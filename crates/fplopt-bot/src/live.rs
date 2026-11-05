//! The deadline alerts (Python `fplopt.bot.live`): `fplopt-rs live tick`, every 15 minutes.
//!
//! Each run:
//! 1. **Due?** From the newest bootstrap, the next deadline d: `alert24` from d − 24 h until
//!    d − 15 min, `alert2` (the silent re-run) from the archiver's pre-deadline snapshot
//!    (after the one alert24 planned on) until d − 30 min; each once per GW.
//! 2. **Fresh flags for alert24:** an FPL snapshot, once per GW.
//! 3. **Context:** rebuilt (`build all` in a subprocess, then `build_context`) when the
//!    archive's fingerprint, the code's commit or the next deadline changed.
//! 4. **Users** (active, with a team, alerts on), once per GW and kind, a failed user retried
//!    up to `MAX_ATTEMPTS` times; the job fails at the end if anyone failed.

use std::rc::Rc;

use anyhow::{Context, Result, bail};
use chrono::{DateTime, TimeDelta, Utc};
use fplopt_archive::Clock;
use fplopt_archive::adapters::FplSource;
use fplopt_archive::jobs::snapshot_fpl;
use fplopt_archive::raw_store::RawStore;
use fplopt_archive::schedule::PRE_DEADLINE_LEAD;
use fplopt_core::seasons::bootstrap_season;
use fplopt_store::DataStore;
use serde_json::Value;

use crate::builder::{FINGERPRINT_SOURCES, build_context, raw_fingerprint};
use crate::context::{ContextStore, PlanContext};
use crate::service::{AlertOutcome, BotService, SendError};
use crate::store::{Record, UserStore};
use crate::util::parse_isoformat;

pub const ALERT24_LEAD: TimeDelta = TimeDelta::hours(24);
const ALERT24_CUTOFF: TimeDelta = TimeDelta::minutes(15);
const ALERT2_CUTOFF: TimeDelta = TimeDelta::minutes(30);
pub const MAX_ATTEMPTS: usize = 3;
const DONE_ERRORS: [&str; 2] = ["EntryError", "undeliverable"];

#[derive(Clone, Debug, PartialEq)]
pub struct Due {
    pub season: i64,
    pub gw: i64,
    pub deadline: DateTime<Utc>,
    pub kinds: Vec<&'static str>,
}

#[derive(Debug, Default)]
pub struct TickResult {
    pub due: Option<Due>,
    pub snapshot: bool,
    pub context_id: Option<String>,
    pub rebuilt: bool,
    pub sent: Vec<i64>,
    pub silent: Vec<i64>,
    pub failed: Vec<i64>,
}

/// (season, GW, deadline) of the first deadline after `now` in a bootstrap.
pub fn next_event(
    bootstrap: &Value,
    now: DateTime<Utc>,
) -> Result<Option<(i64, i64, DateTime<Utc>)>> {
    let mut events = Vec::new();
    for event in bootstrap.get("events").and_then(Value::as_array).into_iter().flatten() {
        let text = event["deadline_time"].as_str().context("event: no deadline_time")?;
        let deadline = parse_isoformat(&text.replace('Z', "+00:00"))?;
        if deadline > now {
            events.push((deadline, event["id"].as_i64().context("event: no id")?));
        }
    }
    let Some(&(deadline, gw)) = events.iter().min() else { return Ok(None) };
    Ok(Some((i64::from(bootstrap_season(bootstrap)?), gw, deadline)))
}

/// The alert kinds due at `now`, or None.
pub fn due_alerts(
    users: &UserStore,
    event: Option<(i64, i64, DateTime<Utc>)>,
    now: DateTime<Utc>,
    bootstrap_times: &[DateTime<Utc>],
) -> Result<Option<Due>> {
    let Some((season, gw, deadline)) = event else { return Ok(None) };
    let mut kinds = Vec::new();
    let alert24 = users.alert_run(season, gw, "alert24")?;
    if alert24.is_none() && deadline - ALERT24_LEAD <= now && now < deadline - ALERT24_CUTOFF {
        kinds.push("alert24");
    }
    if users.alert_run(season, gw, "alert2")?.is_none()
        && deadline - PRE_DEADLINE_LEAD <= now
        && now < deadline - ALERT2_CUTOFF
    {
        let after = alert24.map(|run| parse_isoformat(&run.bootstrap_at)).transpose()?;
        let window = deadline - PRE_DEADLINE_LEAD;
        if bootstrap_times
            .iter()
            .any(|&t| window <= t && t < deadline && after.is_none_or(|a| t > a))
        {
            kinds.push("alert2");
        }
    }
    Ok((!kinds.is_empty()).then_some(Due { season, gw, deadline, kinds }))
}

/// Whether a new context is needed: none yet, new archive content, new code, or its
/// deadline has passed.
pub fn context_stale(
    meta: Option<&Value>,
    fingerprint: &str,
    sha: Option<&str>,
    now: DateTime<Utc>,
) -> Result<bool> {
    let Some(meta) = meta else { return Ok(true) };
    let deadline = parse_isoformat(meta["deadline"].as_str().context("meta: no deadline")?)?;
    Ok(meta["fingerprint"].as_str() != Some(fingerprint)
        || meta["git_sha"].as_str() != sha
        || deadline <= now)
}

/// What a context refresh needs from the job: the raw archive, `data/`, the build and the
/// code's commit.
pub struct Refresh<'a> {
    pub raw: &'a RawStore,
    pub data_dir: &'a std::path::Path,
    pub build: &'a dyn Fn() -> Result<()>,
    pub sha: Option<String>,
    pub xp_model: &'a str,
}

/// The newest plan context, rebuilt first if it is stale or `force`; (context, rebuilt).
pub fn refresh_context(
    refresh: &Refresh,
    contexts: &ContextStore,
    clock: &dyn Clock,
    force: bool,
) -> Result<(Option<Rc<PlanContext>>, bool)> {
    let fingerprint = raw_fingerprint(&refresh.raw.root, FINGERPRINT_SOURCES)?;
    let meta = contexts.meta(None)?;
    if force || context_stale(meta.as_ref(), &fingerprint, refresh.sha.as_deref(), clock.now())? {
        log::info!("archive, code or deadline changed: rebuilding data/ and the plan context");
        (refresh.build)()?;
        let times = refresh.raw.times("fpl", "bootstrap-static")?;
        let store = DataStore::open(refresh.data_dir)?;
        let Some(context) = build_context(
            &store,
            clock.now(),
            &fingerprint,
            times.last().copied(),
            refresh.sha.clone(),
            refresh.xp_model,
        )?
        else {
            log::info!("no deadline ahead in the data: no context");
            return Ok((None, false));
        };
        contexts.write(&context)?;
        contexts.prune_default();
        log::info!("plan context {}: {} GW{}", context.context_id, context.season, context.gw);
        return Ok((contexts.latest()?, true));
    }
    Ok((contexts.latest()?, false))
}

/// One run (module docs). `send(chat_id, html)` sends one message piece.
pub fn live_tick(
    refresh: &Refresh,
    users: &UserStore,
    contexts: &ContextStore,
    service: &BotService,
    fpl: &dyn FplSource,
    send: &mut dyn FnMut(i64, &str) -> std::result::Result<(), SendError>,
    clock: &dyn Clock,
) -> Result<TickResult> {
    let mut result = TickResult::default();
    let raw = refresh.raw;
    let Some(latest) = raw.latest("fpl", "bootstrap-static")? else {
        log::info!("no bootstrap snapshot yet: nothing to do");
        return Ok(result);
    };
    let current = clock.now();
    let Some(event) = next_event(&RawStore::read_json(&latest)?, current)? else {
        log::info!("no deadline ahead in the newest bootstrap: nothing to do");
        return Ok(result);
    };
    let times = raw.times("fpl", "bootstrap-static")?;
    let due = due_alerts(users, Some(event), current, &times)?;
    result.due = due.clone();
    if let Some(due) = &due
        && due.kinds.contains(&"alert24")
        && !times.iter().any(|&t| t >= due.deadline - ALERT24_LEAD)
    {
        log::info!("GW{} alert24 due: fresh FPL snapshot", due.gw);
        snapshot_fpl(raw, fpl, clock)?;
        result.snapshot = true;
    }
    let (context, rebuilt) = refresh_context(refresh, contexts, clock, false)?;
    result.rebuilt = rebuilt;
    let Some(context) = context else { return Ok(result) };
    result.context_id = Some(context.context_id.clone());
    let Some(due) = due else { return Ok(result) };
    if (context.season, context.gw) != (due.season, due.gw) {
        bail!(
            "the plan context is for {} GW{} but the next deadline is {} GW{}: data/ is behind \
             the archive",
            context.season,
            context.gw,
            due.season,
            due.gw
        );
    }
    for kind in &due.kinds {
        alert(kind, &context, users, service, send, &mut result)?;
        if !retry_left(kind, &context, users)? {
            users.mark_alert_run(
                due.season,
                due.gw,
                kind,
                &context.context_id,
                context.bootstrap_at.unwrap_or(context.created_at),
            )?;
        }
    }
    if !result.failed.is_empty() {
        bail!(
            "GW{} alerts failed for {} user(s) (user_ids {:?}); the others were sent",
            due.gw,
            result.failed.len(),
            result.failed
        );
    }
    Ok(result)
}

/// The user's records of this kind for the context's GW.
fn records(
    users: &UserStore,
    user_id: i64,
    context: &PlanContext,
    kind: &str,
) -> Result<Vec<crate::store::Recommendation>> {
    Ok(users
        .recommendations(user_id)?
        .into_iter()
        .filter(|r| r.season == context.season && r.gw == context.gw && r.kind == kind)
        .collect())
}

/// Sent, silent, or a final error (told to the user, or undeliverable).
fn done(records: &[crate::store::Recommendation]) -> bool {
    records.iter().any(|r| match &r.error {
        None => true,
        Some(error) => DONE_ERRORS.iter().any(|d| error.starts_with(d)),
    })
}

fn alert(
    kind: &str,
    context: &PlanContext,
    users: &UserStore,
    service: &BotService,
    send: &mut dyn FnMut(i64, &str) -> std::result::Result<(), SendError>,
    result: &mut TickResult,
) -> Result<()> {
    for user in users.alert_users()? {
        let records = records(users, user.user_id, context, kind)?;
        if done(&records) || records.len() >= MAX_ATTEMPTS {
            continue;
        }
        match service.alert_user(kind, context, &user, send) {
            Ok(AlertOutcome::Sent) => result.sent.push(user.user_id),
            Ok(AlertOutcome::Silent) => result.silent.push(user.user_id),
            Ok(AlertOutcome::Told | AlertOutcome::Undeliverable) => {}
            Err(err) => {
                log::error!("GW{} {kind} failed for user {}: {err:#}", context.gw, user.user_id);
                users.record(
                    user.user_id,
                    context.season,
                    context.gw,
                    kind,
                    &Record {
                        context_id: Some(context.context_id.clone()),
                        error: Some("failed (see the job log)".to_string()),
                        ..Record::default()
                    },
                )?;
                result.failed.push(user.user_id);
            }
        }
    }
    Ok(())
}

/// Whether a user's alert of this kind failed and has tries left.
fn retry_left(kind: &str, context: &PlanContext, users: &UserStore) -> Result<bool> {
    for user in users.alert_users()? {
        let records = records(users, user.user_id, context, kind)?;
        if done(&records) {
            continue;
        }
        if !records.is_empty() && records.len() < MAX_ATTEMPTS {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    const SEASON: i64 = 2026;
    const GW: i64 = 6;

    fn deadline() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 9, 26, 10, 0, 0).unwrap()
    }

    fn users(dir: &std::path::Path) -> UserStore {
        UserStore::new(dir.join("bot.sqlite")).unwrap()
    }

    fn kinds(due: Option<Due>) -> Option<Vec<&'static str>> {
        due.map(|d| d.kinds)
    }

    // Python test_due_alerts.
    #[test]
    fn due_alerts_windows() {
        let dir = std::env::temp_dir().join(format!("fplopt-live-due-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let users = users(&dir);
        let d = deadline();
        let event = Some((SEASON, GW, d));
        let h = TimeDelta::hours;
        let m = TimeDelta::minutes;
        assert_eq!(kinds(due_alerts(&users, None, d, &[]).unwrap()), None);
        assert_eq!(kinds(due_alerts(&users, event, d - h(25), &[]).unwrap()), None);
        assert_eq!(
            kinds(due_alerts(&users, event, d - h(24), &[]).unwrap()),
            Some(vec!["alert24"])
        );
        assert_eq!(
            kinds(due_alerts(&users, event, d - m(16), &[]).unwrap()),
            Some(vec!["alert24"])
        );
        assert_eq!(kinds(due_alerts(&users, event, d - m(15), &[]).unwrap()), None);
        users.mark_alert_run(SEASON, GW, "alert24", "c", d - h(23)).unwrap();
        assert_eq!(kinds(due_alerts(&users, event, d - h(10), &[]).unwrap()), None);
        let window = d - h(2);
        // The re-run waits for the archiver's pre-deadline snapshot.
        assert_eq!(kinds(due_alerts(&users, event, window + m(5), &[]).unwrap()), None);
        let snap = [window + m(1)];
        assert_eq!(
            kinds(due_alerts(&users, event, window + m(5), &snap).unwrap()),
            Some(vec!["alert2"])
        );
        assert_eq!(kinds(due_alerts(&users, event, d - m(30), &snap).unwrap()), None);
        users.mark_alert_run(SEASON, GW, "alert2", "c", window).unwrap();
        assert_eq!(kinds(due_alerts(&users, event, window + m(5), &snap).unwrap()), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // Python test_alert24_late_inside_the_window_skips_the_rerun and
    // test_due_alerts_rerun_does_not_wait_for_a_stuck_alert24.
    #[test]
    fn rerun_needs_a_newer_snapshot_but_not_a_finished_alert24() {
        let dir = std::env::temp_dir().join(format!("fplopt-live-rerun-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let d = deadline();
        let event = Some((SEASON, GW, d));
        let window = d - TimeDelta::hours(2);
        {
            let users = users(&dir.join("stuck"));
            let after = [window + TimeDelta::days(1)];
            let due = due_alerts(&users, event, window + TimeDelta::minutes(5), &after).unwrap();
            assert_eq!(kinds(due), Some(vec!["alert24"]));
            let snap = [window + TimeDelta::minutes(1)];
            let due = due_alerts(&users, event, window + TimeDelta::minutes(5), &snap).unwrap();
            assert_eq!(kinds(due), Some(vec!["alert24", "alert2"]));
        }
        let users = users(&dir.join("late"));
        let snap = d - TimeDelta::minutes(110);
        users.mark_alert_run(SEASON, GW, "alert24", "c", snap).unwrap();
        let due = due_alerts(&users, event, d - TimeDelta::minutes(100), &[snap]).unwrap();
        assert_eq!(kinds(due), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // Python test_next_event_and_context_stale.
    #[test]
    fn next_event_and_staleness() {
        let d = deadline();
        let bootstrap = serde_json::json!({
            "events": [
                {"id": 5, "deadline_time": "2026-09-19T10:00:00Z"},
                {"id": 6, "deadline_time": "2026-09-26T10:00:00Z"},
            ],
            "elements": [],
            "phases": [{"id": 1, "start_event": 1, "stop_event": 38}],
            "teams": [],
            "total_players": 0,
            "game_settings": {},
            "element_types": [],
        });
        let found = next_event(&bootstrap, d - TimeDelta::hours(1)).unwrap();
        assert_eq!(found, Some((SEASON, GW, d)));
        assert_eq!(next_event(&bootstrap, d).unwrap(), None);
        let meta = serde_json::json!({"fingerprint": "f", "git_sha": "s", "deadline": crate::util::py_isoformat(d)});
        let before = d - TimeDelta::hours(1);
        assert!(context_stale(None, "f", Some("s"), before).unwrap());
        assert!(!context_stale(Some(&meta), "f", Some("s"), before).unwrap());
        assert!(context_stale(Some(&meta), "g", Some("s"), before).unwrap());
        assert!(context_stale(Some(&meta), "f", Some("t"), before).unwrap());
        assert!(context_stale(Some(&meta), "f", Some("s"), d).unwrap());
    }
}
