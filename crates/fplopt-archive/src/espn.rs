//! ESPN scoreboards and match summaries into raw/. Port of `fplopt.ingest.espn`.
//!
//! Layout: `espn/<league>/scoreboard/<period>/<ts>.json.gz` (period 'YYYY' or 'YYYYMM';
//! every scoreboard of a run shares the run's start time, so a run is one schedule
//! snapshot) and `espn/<league>/summary/<event_id>/<ts>.json.gz`. A summary is fetched for a
//! completed event with a club of `team_ids` (every event of `eng.1`) kicking off from
//! 1 July 2016, unless one with a completed status is archived already (resumable; a
//! summary archived before the match finished is fetched again).

use std::cell::Cell;
use std::collections::BTreeSet;
use std::path::Path;
use std::time::Duration;

use anyhow::Result;
use chrono::{DateTime, Datelike, TimeDelta, TimeZone, Utc};
use serde_json::Value;

use crate::adapters::EspnSource;
use crate::adapters::espn::LEAGUES;
use crate::jobs::{Step, run_independently, step};
use crate::pytime::parse_iso_utc;
use crate::raw_store::RawStore;
use crate::{Clock, Failure, stop, truthy};

pub const SOURCE: &str = "espn";
pub const FIRST_YEAR: i32 = 2016;
pub const RECENT_BACK: TimeDelta = TimeDelta::days(7);
/// Covers a 6-GW horizon of cup and European fixtures.
pub const RECENT_AHEAD: TimeDelta = TimeDelta::days(60);
pub const PAUSE: Duration = Duration::from_secs(1);

fn first_kickoff() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2016, 7, 1, 0, 0, 0).unwrap()
}

pub fn scoreboard_endpoint(league: &str, period: &str) -> String {
    format!("{league}/scoreboard/{period}")
}

pub fn summary_endpoint(league: &str, event_id: &str) -> String {
    format!("{league}/summary/{event_id}")
}

/// The `espn` ids of config/teams.csv.
pub fn espn_team_ids(teams_csv: &Path) -> Result<BTreeSet<String>> {
    let mut reader = csv::Reader::from_path(teams_csv)?;
    let column = reader
        .headers()?
        .iter()
        .position(|h| h == "espn")
        .ok_or_else(|| Failure::error("KeyError", "'espn'"))?;
    let mut ids = BTreeSet::new();
    for record in reader.records() {
        let record = record?;
        let id = record.get(column).ok_or_else(|| Failure::error("KeyError", "'espn'"))?.trim();
        if !id.is_empty() {
            ids.insert(id.to_owned());
        }
    }
    Ok(ids)
}

/// `str(value)` for an event id.
fn id_text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Ids of completed events from 1 July 2016 with a club of `team_ids` (eng.1: all).
pub fn events_to_fetch(
    scoreboard: &Value,
    league: &str,
    team_ids: &BTreeSet<String>,
) -> Result<Vec<String>> {
    let missing = |key: &str| Failure::error("KeyError", format!("'{key}'"));
    let mut out = Vec::new();
    let Some(events) = scoreboard.get("events") else { return Ok(out) };
    for ev in events.as_array().ok_or_else(|| missing("events"))? {
        let status =
            ev.get("status").and_then(|s| s.get("type")).ok_or_else(|| missing("status"))?;
        if !truthy(&status["completed"]) {
            continue;
        }
        let date = ev.get("date").and_then(Value::as_str).ok_or_else(|| missing("date"))?;
        if parse_iso_utc(date)? < first_kickoff() {
            continue;
        }
        let competitors = ev["competitions"][0]["competitors"]
            .as_array()
            .ok_or_else(|| missing("competitions"))?;
        let mut ours = false;
        for c in competitors {
            let id = c.get("team").and_then(|t| t.get("id")).ok_or_else(|| missing("team"))?;
            ours |= id.as_str().is_some_and(|id| team_ids.contains(id));
        }
        if league == "eng.1" || ours {
            out.push(id_text(ev.get("id").ok_or_else(|| missing("id"))?));
        }
    }
    Ok(out)
}

pub fn summary_complete(store: &RawStore, league: &str, event_id: &str) -> Result<bool> {
    let Some(path) = store.latest(SOURCE, &summary_endpoint(league, event_id))? else {
        return Ok(false);
    };
    let summary = RawStore::read_json(&path)?;
    let status = summary
        .get("header")
        .and_then(|h| h.get("competitions"))
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("status"))
        .and_then(|s| s.get("type"))
        .ok_or_else(|| Failure::error("KeyError", "'header'"))?;
    Ok(truthy(&status["completed"]))
}

/// Archive one scoreboard (at `run_at`) and the summaries it calls for. Every summary is
/// attempted; failures are raised together (SnapshotError). Returns summaries written.
#[allow(clippy::too_many_arguments)]
pub fn archive_period(
    store: &RawStore,
    espn: &dyn EspnSource,
    league: &str,
    period: &str,
    team_ids: &BTreeSet<String>,
    run_at: &DateTime<Utc>,
    clock: &dyn Clock,
    pause: Duration,
) -> Result<usize> {
    let content = espn.scoreboard(league, period)?;
    store.write(SOURCE, &scoreboard_endpoint(league, period), &content, run_at, None)?;
    let scoreboard: Value = serde_json::from_slice(crate::fetch::strip_bom(&content))?;
    let mut due = Vec::new();
    for event_id in events_to_fetch(&scoreboard, league, team_ids)? {
        if !summary_complete(store, league, &event_id)? {
            due.push(event_id);
        }
    }
    let written = Cell::new(0usize);
    let steps: Vec<Step<'_>> = due
        .iter()
        .map(|event_id| {
            let written = &written;
            step(format!("{league}/{event_id}"), move || {
                clock.sleep(pause);
                stop::check()?;
                let summary = espn.summary(league, event_id)?;
                store.write(
                    SOURCE,
                    &summary_endpoint(league, event_id),
                    &summary,
                    &clock.now(),
                    None,
                )?;
                written.set(written.get() + 1);
                Ok(())
            })
        })
        .collect();
    let result = run_independently(steps);
    log::info!("espn {league} {period}: {} summaries archived ({} due)", written.get(), due.len());
    result.map(|()| written.get())
}

fn run_periods(
    store: &RawStore,
    espn: &dyn EspnSource,
    periods: &[String],
    team_ids: &BTreeSet<String>,
    clock: &dyn Clock,
    pause: Duration,
) -> Result<()> {
    let run_at = clock.now();
    let run_at = &run_at;
    let mut steps = Vec::new();
    for period in periods {
        for league in LEAGUES {
            steps.push(step(format!("espn {league} {period}"), move || {
                stop::check()?;
                archive_period(store, espn, league, period, team_ids, run_at, clock, pause)
                    .map(drop)
            }));
        }
    }
    run_independently(steps)
}

/// Every league's calendar-year scoreboards from `first_year` to this year, and their
/// summaries (~7,000 from 2016, 2-3 h at 1 request/s). Resumable.
pub fn backfill_espn(
    store: &RawStore,
    espn: &dyn EspnSource,
    team_ids: &BTreeSet<String>,
    first_year: i32,
    clock: &dyn Clock,
    pause: Duration,
) -> Result<()> {
    let years: Vec<String> = (first_year..=clock.now().year()).map(|y| y.to_string()).collect();
    run_periods(store, espn, &years, team_ids, clock, pause)
}

/// Months from RECENT_BACK before `at` to RECENT_AHEAD after, as 'YYYYMM'.
pub fn recent_periods(at: DateTime<Utc>) -> Vec<String> {
    let (first, last) = (at - RECENT_BACK, at + RECENT_AHEAD);
    let mut months = Vec::new();
    let (mut year, mut month) = (first.year(), first.month());
    while (year, month) <= (last.year(), last.month()) {
        months.push(format!("{year}{month:02}"));
        (year, month) = if month == 12 { (year + 1, 1) } else { (year, month + 1) };
    }
    months
}

/// Daily: the recent months' scoreboards of every league (the schedule snapshot) and the
/// summaries of matches finished since.
pub fn snapshot_espn_recent(
    store: &RawStore,
    espn: &dyn EspnSource,
    team_ids: &BTreeSet<String>,
    clock: &dyn Clock,
    pause: Duration,
) -> Result<()> {
    run_periods(store, espn, &recent_periods(clock.now()), team_ids, clock, pause)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn event(id: &str, date: &str, completed: bool, teams: [&str; 2]) -> Value {
        json!({"id": id, "date": date, "status": {"type": {"completed": completed}},
               "competitions": [{"competitors": [{"team": {"id": teams[0]}}, {"team": {"id": teams[1]}}]}]})
    }

    #[test]
    fn picks_completed_events_of_our_clubs() {
        let ids: BTreeSet<String> = ["359".to_owned()].into();
        let board = json!({"events": [
            event("1", "2026-10-01T19:00Z", true, ["359", "1"]),
            event("2", "2026-10-01T19:00Z", false, ["359", "1"]),
            event("3", "2026-10-01T19:00Z", true, ["2", "1"]),
            event("4", "2016-06-30T19:00Z", true, ["359", "1"]),
        ]});
        assert_eq!(events_to_fetch(&board, "eng.fa", &ids).unwrap(), ["1"]);
        assert_eq!(events_to_fetch(&board, "eng.1", &ids).unwrap(), ["1", "3"]);
        assert!(events_to_fetch(&json!({}), "eng.1", &ids).unwrap().is_empty());
    }

    #[test]
    fn recent_months() {
        let at = Utc.with_ymd_and_hms(2026, 12, 3, 2, 30, 0).unwrap();
        assert_eq!(recent_periods(at), ["202611", "202612", "202701", "202702"]);
    }

    #[test]
    fn team_ids_from_csv() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("teams.csv");
        std::fs::write(&path, "team_key,espn\n1, 359 \n2,\n").unwrap();
        assert_eq!(espn_team_ids(&path).unwrap(), BTreeSet::from(["359".to_owned()]));
        std::fs::write(&path, "team_key\n1\n").unwrap();
        assert!(espn_team_ids(&path).is_err());
    }
}
