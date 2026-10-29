//! Archiver jobs: fetch from adapters and append to the raw store. Port of
//! `fplopt.ingest.jobs`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Result;
use chrono::{Datelike, TimeDelta};
use fplopt_core::pyjson;
use fplopt_core::seasons::{bootstrap_season, football_data_code, season_label, season_start_year};
use serde::Serialize;
use serde_json::Value;

use crate::adapters::{EspnSource, FootballDataSource, FplSource, OddsSource};
use crate::fetch::FetchError;
use crate::pytime::isoformat;
use crate::raw_store::{RawStore, TS_PATTERN, gzip_bytes, parse_ts};
use crate::schedule::{PRE_DEADLINE_LEAD, deadlines_from_bootstrap, pre_deadline_due};
use crate::{Clock, Failure, kind_name, py_repr_ints, stop, summary, truthy};

pub const MAX_CONSECUTIVE_FAILURES: usize = 10;
/// The new-season CSV may not exist yet in July/August.
pub const FOOTBALL_DATA_GRACE_MONTHS: [u32; 2] = [7, 8];
/// Post-lockdown only trusts a bootstrap this recent (normally the daily run's own FPL step).
pub const POST_LOCKDOWN_MAX_BOOTSTRAP_AGE: TimeDelta = TimeDelta::hours(6);
pub const ELEMENT_SUMMARY_PAUSE: Duration = Duration::from_millis(250);
pub const EVENT_LIVE_PAUSE: Duration = Duration::from_millis(250);

/// A named step for [`run_independently`].
pub type Step<'a> = (String, Box<dyn FnOnce() -> Result<()> + 'a>);

pub fn step<'a>(name: impl Into<String>, run: impl FnOnce() -> Result<()> + 'a) -> Step<'a> {
    (name.into(), Box::new(run))
}

/// Run every step even if earlier ones fail; then fail with one `SnapshotError` naming each
/// failed step (single line, so no source's error is cut off when the alert keeps only the
/// first line). A SIGTERM stops the run instead.
pub fn run_independently(steps: Vec<Step<'_>>) -> Result<()> {
    let mut errors = Vec::new();
    for (name, run) in steps {
        if let Err(err) = run() {
            if err.is::<stop::Terminated>() {
                return Err(err);
            }
            log::error!("{name} snapshot failed: {err:#}");
            errors.push(format!("{name}: {}: {}", kind_name(&err), summary(&err)));
        }
    }
    if errors.is_empty() { Ok(()) } else { Err(Failure::error("SnapshotError", errors.join("; "))) }
}

/// Fixtures first, bootstrap last: the tick treats a bootstrap snapshot inside a
/// pre-deadline window as done, so a failure part-way must leave the window open.
pub fn snapshot_fpl(
    store: &RawStore,
    fpl: &dyn FplSource,
    clock: &dyn Clock,
) -> Result<Vec<PathBuf>> {
    let fixtures = fpl.fixtures()?;
    let fixtures = store.write("fpl", "fixtures", &fixtures, &clock.now(), None)?;
    let bootstrap = fpl.bootstrap_static()?;
    let bootstrap = store.write("fpl", "bootstrap-static", &bootstrap, &clock.now(), None)?;
    Ok(vec![fixtures, bootstrap])
}

pub fn snapshot_odds(
    store: &RawStore,
    odds: Option<&dyn OddsSource>,
    clock: &dyn Clock,
) -> Result<Option<PathBuf>> {
    let Some(odds) = odds else {
        log::warn!("ODDS_API_KEY not set; skipping odds snapshot");
        return Ok(None);
    };
    let content = odds.epl_odds()?;
    Ok(Some(store.write("odds", "soccer_epl", &content, &clock.now(), None)?))
}

/// One season's football-data CSV under football-data/E0/<code>, e.g. E0/1617.
pub fn snapshot_football_data(
    store: &RawStore,
    fd: &dyn FootballDataSource,
    start_year: i32,
    clock: &dyn Clock,
) -> Result<PathBuf> {
    let content = fd.epl_season(start_year)?;
    let endpoint = format!("E0/{}", football_data_code(start_year));
    store.write_bytes(
        "football-data",
        &endpoint,
        &gzip_bytes(&content),
        &clock.now(),
        ".csv.gz",
        None,
    )
}

/// Daily: the in-progress season's CSV (results + odds are appended twice a week). A 404 is
/// tolerated in July/August (file not published yet); otherwise it is an error.
pub fn snapshot_football_data_current(
    store: &RawStore,
    fd: &dyn FootballDataSource,
    clock: &dyn Clock,
) -> Result<Option<PathBuf>> {
    let current = clock.now();
    match snapshot_football_data(store, fd, season_start_year(&current), clock) {
        Ok(path) => Ok(Some(path)),
        Err(err)
            if err.downcast_ref::<FetchError>().and_then(FetchError::status) == Some(404)
                && FOOTBALL_DATA_GRACE_MONTHS.contains(&current.month()) =>
        {
            log::warn!("football-data CSV for the new season not published yet");
            Ok(None)
        }
        Err(err) => Err(err),
    }
}

/// The ESPN part of the daily job, with the file its club ids come from.
pub struct DailyEspn<'a> {
    pub source: &'a dyn EspnSource,
    pub teams_csv: &'a Path,
}

/// Daily archive: FPL, odds, the current football-data CSV (if a source is given), the
/// post-lockdown step (after the FPL step, so it reads the bootstrap just archived), then
/// ESPN's recent scoreboards and summaries (if given; its club ids are read inside the step,
/// so a bad file fails that step only). ESPN comes last: it is the slowest step and
/// post-lockdown does not wait on it.
pub fn run_daily(
    store: &RawStore,
    fpl: &dyn FplSource,
    odds: Option<&dyn OddsSource>,
    clock: &dyn Clock,
    football_data: Option<&dyn FootballDataSource>,
    espn: Option<DailyEspn<'_>>,
) -> Result<()> {
    let mut steps = vec![
        step("fpl", || snapshot_fpl(store, fpl, clock).map(drop)),
        step("odds", || snapshot_odds(store, odds, clock).map(drop)),
    ];
    if let Some(fd) = football_data {
        steps.push(step("football-data", move || {
            snapshot_football_data_current(store, fd, clock).map(drop)
        }));
    }
    steps.push(step("post-lockdown", || snapshot_post_lockdown(store, fpl, clock)));
    if let Some(espn) = espn {
        steps.push(step("espn", move || {
            let team_ids = crate::espn::espn_team_ids(espn.teams_csv)?;
            crate::espn::snapshot_espn_recent(
                store,
                espn.source,
                &team_ids,
                clock,
                crate::espn::PAUSE,
            )
        }));
    }
    run_independently(steps)
}

/// Run every ~15 min. Inside each pre-deadline window, snapshot FPL and odds once each; a
/// source that fails is retried on the next tick. Returns true if anything ran.
///
/// The tick that takes a window's FPL snapshot then runs the post-lockdown step on it, so a
/// GW finalised after the 02:30 daily run is archived before the next deadline. It is
/// idempotent, so it usually does nothing. It fails independently of the FPL and odds
/// steps, and isn't retried by later ticks in the window (the daily job picks it up).
pub fn run_tick(
    store: &RawStore,
    fpl: &dyn FplSource,
    odds: Option<&dyn OddsSource>,
    clock: &dyn Clock,
) -> Result<bool> {
    let Some(latest) = store.latest("fpl", "bootstrap-static")? else {
        // FPL and odds only: on a fresh store the post-lockdown step could mean a full
        // element-summary run, which the daily job does.
        log::info!("no bootstrap snapshot yet; taking one");
        run_independently(vec![
            step("fpl", || snapshot_fpl(store, fpl, clock).map(drop)),
            step("odds", || snapshot_odds(store, odds, clock).map(drop)),
        ])?;
        return Ok(true);
    };
    let deadlines = deadlines_from_bootstrap(&RawStore::read_json(&latest)?)?;
    let current = clock.now();
    let fpl_deadline = pre_deadline_due(
        current,
        &deadlines,
        &store.times("fpl", "bootstrap-static")?,
        PRE_DEADLINE_LEAD,
    );
    let odds_due = odds.is_some()
        && pre_deadline_due(
            current,
            &deadlines,
            &store.times("odds", "soccer_epl")?,
            PRE_DEADLINE_LEAD,
        )
        .is_some();
    let mut steps = Vec::new();
    if let Some(deadline) = fpl_deadline {
        log::info!("pre-deadline FPL snapshot for deadline {}", isoformat(&deadline));
        steps.push(step("fpl", || snapshot_fpl(store, fpl, clock).map(drop)));
    }
    if odds_due {
        log::info!("pre-deadline odds snapshot");
        steps.push(step("odds", || snapshot_odds(store, odds, clock).map(drop)));
    }
    if fpl_deadline.is_some() {
        // Last, so it reads the bootstrap the FPL step just archived.
        steps.push(step("post-lockdown", || snapshot_post_lockdown(store, fpl, clock)));
    }
    run_independently(steps)?;
    Ok(fpl_deadline.is_some() || odds_due)
}

#[derive(Serialize)]
struct ElementSummaryManifest<'a> {
    run_at: String,
    finished_at: String,
    season: Option<i32>,
    through_event: i64,
    expected: &'a [i64],
    written: &'a [i64],
    failed: &'a [i64],
}

/// Archive element-summary for every current player under one run timestamp.
///
/// Writes `<run>/_manifest.json.gz` (expected, written and failed ids, plus the `season` and
/// `through_event` of the bootstrap it fetched) even when aborted, so incomplete runs are
/// identifiable. The bootstrap it archives counts as a snapshot for the pre-deadline tick.
/// Element summaries in a run share `run_at` but are fetched later; treat the manifest's
/// `finished_at` as the run's `available_at`.
pub fn backfill_element_summaries(
    store: &RawStore,
    fpl: &dyn FplSource,
    clock: &dyn Clock,
    pause: Duration,
    max_consecutive_failures: usize,
) -> Result<usize> {
    let bootstrap = fpl.bootstrap_static()?;
    let run_at = clock.now();
    if let Err(err) = store.write("fpl", "bootstrap-static", &bootstrap, &run_at, None) {
        if kind_name(&err) != "FileExistsError" {
            return Err(err);
        }
        // The daily FPL step archived a bootstrap in this same second; that one stands.
        log::info!("bootstrap for {} already archived; keeping it", isoformat(&run_at));
    }
    let payload: Value = serde_json::from_slice(crate::fetch::strip_bom(&bootstrap))?;
    let elements =
        payload["elements"].as_array().ok_or_else(|| Failure::error("KeyError", "'elements'"))?;
    let ids = elements
        .iter()
        .map(|e| e["id"].as_i64().ok_or_else(|| Failure::error("KeyError", "'id'")))
        .collect::<Result<Vec<_>>>()?;
    let season = season_or_none(&payload);
    let through_event = checked_events(&payload)?.last().copied().unwrap_or(0);
    let mut written = Vec::new();
    let mut failed = Vec::new();
    let outcome = (|| -> Result<()> {
        let mut consecutive = 0;
        for &element_id in &ids {
            stop::check()?;
            let result = fpl.element_summary(element_id).and_then(|content| {
                store.write(
                    "fpl",
                    "element-summary",
                    &content,
                    &run_at,
                    Some(&element_id.to_string()),
                )
            });
            match result {
                Ok(_) => {
                    written.push(element_id);
                    consecutive = 0;
                }
                Err(err) => {
                    if err.is::<stop::Terminated>() {
                        return Err(err);
                    }
                    log::error!("element-summary {element_id} failed: {err:#}");
                    failed.push(element_id);
                    consecutive += 1;
                    if consecutive >= max_consecutive_failures {
                        return Err(Failure::error(
                            "RuntimeError",
                            format!(
                                "aborting backfill after {consecutive} consecutive failures (last element {element_id}: {})",
                                kind_name(&err)
                            ),
                        ));
                    }
                }
            }
            clock.sleep(pause);
        }
        Ok(())
    })();
    let manifest = ElementSummaryManifest {
        run_at: isoformat(&run_at),
        finished_at: isoformat(&clock.now()),
        season,
        through_event,
        expected: &ids,
        written: &written,
        failed: &failed,
    };
    let saved = pyjson::dumps(&manifest).and_then(|text| {
        store.write("fpl", "element-summary", text.as_bytes(), &run_at, Some("_manifest"))
    });
    if let Err(err) = saved {
        log::error!("could not write backfill manifest: {err:#}");
    }
    outcome?;
    if !failed.is_empty() {
        return Err(Failure::error(
            "RuntimeError",
            format!(
                "{} of {} element summaries failed: {}",
                failed.len(),
                ids.len(),
                py_repr_ints(&failed)
            ),
        ));
    }
    log::info!("archived {} element summaries", ids.len());
    Ok(ids.len())
}

// --- post-lockdown ingest (issue #17) ------------------------------------------------------

/// GWs whose data FPL has finalised (bonus confirmed): `data_checked` is true. Sorted.
pub fn checked_events(bootstrap: &Value) -> Result<Vec<i64>> {
    let events = match &bootstrap["events"] {
        Value::Array(events) => events.as_slice(),
        other if !truthy(other) => &[],
        _ => return Err(Failure::error("TypeError", "bootstrap events is not a list")),
    };
    let mut ids = Vec::new();
    for event in events {
        if truthy(&event["data_checked"]) {
            ids.push(event["id"].as_i64().ok_or_else(|| Failure::error("KeyError", "'id'"))?);
        }
    }
    ids.sort_unstable();
    Ok(ids)
}

/// Season of a bootstrap, or None if it has no usable events (e.g. during the reset).
fn season_or_none(bootstrap: &Value) -> Option<i32> {
    bootstrap_season(bootstrap).ok()
}

/// 'event-live/2026-27/5': season in the path because GW numbers repeat every season.
pub fn event_live_endpoint(season: i32, gw: i64) -> String {
    format!("event-live/{}/{gw}", season_label(season))
}

/// Archive event/{gw}/live/ once per finalised GW, under fpl/event-live/<season>/<gw>. GWs
/// already archived are skipped; every pending GW is attempted even if another fails
/// (SnapshotError naming each failed GW). Returns the GWs written.
pub fn snapshot_event_live(
    store: &RawStore,
    fpl: &dyn FplSource,
    bootstrap: &Value,
    clock: &dyn Clock,
    pause: Duration,
) -> Result<Vec<i64>> {
    let season = bootstrap_season(bootstrap)?;
    let mut pending = Vec::new();
    for gw in checked_events(bootstrap)? {
        if store.latest("fpl", &event_live_endpoint(season, gw))?.is_none() {
            pending.push(gw);
        }
    }
    let written = std::cell::RefCell::new(Vec::new());
    let first = pending.first().copied();
    let steps = pending
        .iter()
        .map(|&gw| {
            let written = &written;
            step(format!("gw {gw}"), move || {
                if Some(gw) != first {
                    clock.sleep(pause);
                }
                stop::check()?;
                let content = fpl.event_live(gw)?;
                store.write(
                    "fpl",
                    &event_live_endpoint(season, gw),
                    &content,
                    &clock.now(),
                    None,
                )?;
                written.borrow_mut().push(gw);
                Ok(())
            })
        })
        .collect();
    let result = run_independently(steps);
    let written = written.into_inner();
    if !written.is_empty() {
        log::info!(
            "archived event-live for {} GWs {}",
            season_label(season),
            py_repr_ints(&written)
        );
    }
    result.map(|()| written)
}

/// (season, through_event) of a manifest whose run wrote every expected element; None for
/// incomplete runs and for older manifests without these keys.
fn complete_run_key(manifest: &Value) -> Option<(i64, i64)> {
    let manifest = manifest.as_object()?;
    // JSON ints only (Python: `type(value) is int`, so not bools or floats).
    let int =
        |key: &str| manifest.get(key).filter(|v| v.is_i64() || v.is_u64()).and_then(Value::as_i64);
    let (season, through_event) = (int("season")?, int("through_event")?);
    let expected = manifest.get("expected")?.as_array()?;
    let written = manifest.get("written")?.as_array()?;
    if manifest.get("failed").is_some_and(truthy) {
        return None;
    }
    let written: HashSet<String> = written.iter().map(Value::to_string).collect();
    if !expected.iter().all(|e| written.contains(&e.to_string())) {
        return None;
    }
    Some((season, through_event))
}

/// (season, through_event) of the newest element-summary run whose manifest shows every
/// expected element written; (0, 0) if none. "Newest" compares (season, through_event), not
/// run time. Non-timestamp directories, runs without a manifest (killed mid-run) and older
/// manifests without these keys count as none.
pub fn latest_complete_element_summary(store: &RawStore) -> Result<(i64, i64)> {
    let root = store.root.join("fpl").join("element-summary");
    let mut best = (0, 0);
    if !root.is_dir() {
        return Ok(best);
    }
    for entry in std::fs::read_dir(&root)? {
        let run_dir = entry?.path();
        let Some(name) = run_dir.file_name().and_then(|n| n.to_str()) else { continue };
        if !(run_dir.is_dir() && TS_PATTERN.is_match(name)) {
            continue;
        }
        let manifest_path = store.path_for(
            "fpl",
            "element-summary",
            &parse_ts(name)?,
            crate::raw_store::SUFFIX,
            Some("_manifest"),
        )?;
        if !manifest_path.is_file() {
            continue;
        }
        let manifest = match RawStore::read_json(&manifest_path) {
            Ok(manifest) => manifest,
            Err(err) => {
                log::warn!(
                    "unreadable element-summary manifest {}: {err:#}",
                    manifest_path.display()
                );
                continue;
            }
        };
        if let Some(key) = complete_run_key(&manifest)
            && key > best
        {
            best = key;
        }
    }
    Ok(best)
}

/// True if a GW has been finalised since the newest complete element-summary run.
pub fn element_summaries_due(store: &RawStore, bootstrap: &Value) -> Result<bool> {
    let checked = checked_events(bootstrap)?;
    let Some(&last) = checked.last() else { return Ok(false) };
    Ok((i64::from(bootstrap_season(bootstrap)?), last) > latest_complete_element_summary(store)?)
}

/// After GW lockdown: event-live for new finalised GWs, and one fresh element-summary run if
/// a GW was finalised since the last complete run. Reads the latest archived bootstrap; the
/// two steps fail independently. A no-op when nothing new is finalised.
///
/// Skipped (warning, not an error) unless that bootstrap is under
/// POST_LOCKDOWN_MAX_BOOTSTRAP_AGE old: event/{gw}/live has no season in its URL, so a stale
/// bootstrap (say today's FPL step failed during the July reset) would file new-season data
/// under the old season.
pub fn snapshot_post_lockdown(
    store: &RawStore,
    fpl: &dyn FplSource,
    clock: &dyn Clock,
) -> Result<()> {
    let entries = store.json_entries("fpl", "bootstrap-static")?;
    let Some((fetched_at, latest)) = entries.last() else {
        log::warn!("no archived bootstrap yet; skipping post-lockdown ingest");
        return Ok(());
    };
    if clock.now() - *fetched_at > POST_LOCKDOWN_MAX_BOOTSTRAP_AGE {
        log::warn!(
            "latest bootstrap ({}) is stale; skipping post-lockdown ingest",
            isoformat(fetched_at)
        );
        return Ok(());
    }
    let bootstrap = RawStore::read_json(latest)?;
    if checked_events(&bootstrap)?.is_empty() {
        log::info!("no finalised GWs yet; nothing to do after lockdown");
        return Ok(());
    }
    let bootstrap = &bootstrap;
    run_independently(vec![
        step("event-live", || {
            snapshot_event_live(store, fpl, bootstrap, clock, EVENT_LIVE_PAUSE).map(drop)
        }),
        step("element-summary", || {
            if element_summaries_due(store, bootstrap)? {
                log::info!("a GW was finalised since the last complete element-summary run");
                backfill_element_summaries(
                    store,
                    fpl,
                    clock,
                    ELEMENT_SUMMARY_PAUSE,
                    MAX_CONSECUTIVE_FAILURES,
                )?;
            }
            Ok(())
        }),
    ])
}
