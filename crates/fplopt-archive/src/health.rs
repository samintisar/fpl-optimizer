//! Archive health checks: a local dead-man's switch for the archiver timers. Port of
//! `fplopt.ingest.health`.
//!
//! `check freshness` runs daily from its own timer and fails (Telegram alert) when the
//! newest bootstrap snapshot is too old, which catches jobs or timers that stopped silently
//! while the server is up. It can't catch the server itself being down.

use anyhow::Result;
use chrono::{DateTime, TimeDelta, Utc};

use crate::Failure;
use crate::pytime::isoformat;
use crate::raw_store::RawStore;

pub const DEFAULT_MAX_AGE_HOURS: f64 = 36.0;

fn hours(delta: TimeDelta) -> String {
    format!("{:.1} h", delta.num_microseconds().unwrap_or(i64::MAX) as f64 / 3.6e9)
}

/// Age of the newest `raw/fpl/bootstrap-static` snapshot. Fails (`StaleArchiveError`) if
/// there is none or it is older than `max_age`.
pub fn check_freshness(
    store: &RawStore,
    now: DateTime<Utc>,
    max_age: TimeDelta,
) -> Result<TimeDelta> {
    let times = store.times("fpl", "bootstrap-static")?;
    let Some(newest) = times.last() else {
        return Err(Failure::error(
            "StaleArchiveError",
            format!("no bootstrap-static snapshot in {}", store.root.display()),
        ));
    };
    let age = now - *newest;
    if age > max_age {
        return Err(Failure::error(
            "StaleArchiveError",
            format!(
                "newest bootstrap-static snapshot is {} old ({}; limit {})",
                hours(age),
                isoformat(newest),
                hours(max_age)
            ),
        ));
    }
    log::info!("newest bootstrap-static snapshot is {} old (limit {})", hours(age), hours(max_age));
    Ok(age)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn fresh_stale_and_missing() {
        let dir = tempfile::tempdir().unwrap();
        let store = RawStore::new(dir.path());
        let t0 = Utc.with_ymd_and_hms(2026, 10, 1, 2, 30, 0).unwrap();
        let limit = TimeDelta::hours(36);
        let err = check_freshness(&store, t0, limit).unwrap_err();
        assert!(err.to_string().starts_with("no bootstrap-static snapshot in "));
        store.write("fpl", "bootstrap-static", b"{}", &t0, None).unwrap();
        assert_eq!(
            check_freshness(&store, t0 + TimeDelta::hours(3), limit).unwrap(),
            TimeDelta::hours(3)
        );
        let err = check_freshness(&store, t0 + TimeDelta::hours(37), limit).unwrap_err();
        assert_eq!(crate::kind_name(&err), "StaleArchiveError");
        assert_eq!(
            err.to_string(),
            "newest bootstrap-static snapshot is 37.0 h old (2026-10-01T02:30:00+00:00; limit 36.0 h)"
        );
    }
}
