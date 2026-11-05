//! When to take pre-deadline snapshots. Port of `fplopt.ingest.schedule`.

use anyhow::{Context, Result};
use chrono::{DateTime, TimeDelta, Utc};
use serde_json::Value;

use crate::pytime::parse_iso_utc;

pub const PRE_DEADLINE_LEAD: TimeDelta = TimeDelta::hours(2);

/// FPL deadlines as UTC datetimes (FPL publishes them in UTC).
pub fn deadlines_from_bootstrap(bootstrap: &Value) -> Result<Vec<DateTime<Utc>>> {
    let events = bootstrap["events"].as_array().context("bootstrap has no events")?;
    events
        .iter()
        .map(|event| {
            let text = event["deadline_time"].as_str().context("event without deadline_time")?;
            parse_iso_utc(text)
        })
        .collect()
}

/// The deadline whose [deadline - lead, deadline) window contains `now` and has no snapshot
/// yet; otherwise None.
pub fn pre_deadline_due(
    now: DateTime<Utc>,
    deadlines: &[DateTime<Utc>],
    taken: &[DateTime<Utc>],
    lead: TimeDelta,
) -> Option<DateTime<Utc>> {
    deadlines.iter().copied().find(|&deadline| {
        let start = deadline - lead;
        start <= now && now < deadline && !taken.iter().any(|&t| start <= t && t < deadline)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use serde_json::json;

    fn at(h: u32, m: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 8, 21, h, m, 0).unwrap()
    }

    #[test]
    fn parses_deadlines() {
        let b = json!({"events": [{"deadline_time": "2026-08-21T17:30:00Z"}]});
        assert_eq!(deadlines_from_bootstrap(&b).unwrap(), vec![at(17, 30)]);
    }

    #[test]
    fn window_logic() {
        let deadlines = [at(17, 30)];
        let lead = PRE_DEADLINE_LEAD;
        assert_eq!(pre_deadline_due(at(15, 29), &deadlines, &[], lead), None);
        assert_eq!(pre_deadline_due(at(15, 30), &deadlines, &[], lead), Some(at(17, 30)));
        assert_eq!(pre_deadline_due(at(17, 29), &deadlines, &[], lead), Some(at(17, 30)));
        assert_eq!(pre_deadline_due(at(17, 30), &deadlines, &[], lead), None);
        assert_eq!(pre_deadline_due(at(16, 0), &deadlines, &[at(15, 45)], lead), None);
        assert_eq!(pre_deadline_due(at(16, 0), &deadlines, &[at(15, 0)], lead), Some(at(17, 30)));
    }
}
