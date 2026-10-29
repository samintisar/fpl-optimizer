//! Season labels and codes. A season is identified by its start year (2026 = 2026/27).
//! Port of `fplopt.seasons`.

use std::ops::RangeInclusive;
use std::sync::LazyLock;

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Datelike, TimeZone};
use regex::Regex;
use serde_json::Value;

static LABEL: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(\d{4})-(\d{2})$").unwrap());

/// The 2025/26 holdout (PLAN §9).
pub const HOLDOUT_SEASONS: [i32; 1] = [2025];
/// PLAN §5 *Splits*: hyperparameters are chosen on develop and confirmed on validate.
pub const DEVELOP_SEASONS: RangeInclusive<i32> = 2016..=2022; // 2016/17-2022/23
pub const VALIDATE_SEASONS: RangeInclusive<i32> = 2023..=2024; // 2023/24-2024/25

/// 2026 -> "2026-27" (vaastav folders, config file names).
pub fn season_label(start_year: i32) -> String {
    format!("{start_year}-{:02}", (start_year + 1).rem_euclid(100))
}

/// "2026-27" -> 2026. Rejects malformed or non-consecutive labels.
pub fn parse_season_label(label: &str) -> Result<i32> {
    let parsed = LABEL.captures(label).and_then(|m| {
        let start: i32 = m[1].parse().ok()?;
        let end: i32 = m[2].parse().ok()?;
        ((start + 1) % 100 == end).then_some(start)
    });
    match parsed {
        Some(start) => Ok(start),
        None => bail!("not a season label like 2026-27: {label:?}"),
    }
}

/// 2026 -> "2627" (football-data.co.uk URL segment).
pub fn football_data_code(start_year: i32) -> String {
    format!("{:02}{:02}", start_year.rem_euclid(100), (start_year + 1).rem_euclid(100))
}

/// The season in progress or about to start at `when`: July onwards is the new season.
pub fn season_start_year<Tz: TimeZone>(when: &DateTime<Tz>) -> i32 {
    if when.month() >= 7 { when.year() } else { when.year() - 1 }
}

/// Start year of the season a bootstrap-static payload describes (year of GW1's deadline).
pub fn bootstrap_season(bootstrap: &Value) -> Result<i32> {
    let events = bootstrap["events"].as_array().context("bootstrap has no events")?;
    let first = events
        .iter()
        .min_by_key(|event| event["id"].as_i64().unwrap_or(i64::MAX))
        .context("bootstrap has no events")?;
    let deadline = first["deadline_time"].as_str().context("event without deadline_time")?;
    deadline.get(..4).context("short deadline_time")?.parse().context("bad deadline year")
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use serde_json::json;

    #[test]
    fn labels() {
        assert_eq!(season_label(2026), "2026-27");
        assert_eq!(season_label(2099), "2099-00");
        assert_eq!(parse_season_label("2016-17").unwrap(), 2016);
        assert_eq!(parse_season_label("2099-00").unwrap(), 2099);
    }

    #[test]
    fn parse_rejects_malformed() {
        for label in ["2016-18", "16-17", "2016/17", "2016-17 ", ""] {
            assert!(parse_season_label(label).is_err(), "{label:?}");
        }
    }

    #[test]
    fn football_data() {
        assert_eq!(football_data_code(2016), "1617");
        assert_eq!(football_data_code(2026), "2627");
    }

    #[test]
    fn start_year_turns_over_in_july() {
        let at = |y, m, d| Utc.with_ymd_and_hms(y, m, d, 0, 0, 0).unwrap();
        assert_eq!(season_start_year(&at(2026, 6, 30)), 2025);
        assert_eq!(season_start_year(&at(2026, 7, 1)), 2026);
        assert_eq!(season_start_year(&at(2027, 1, 15)), 2026);
    }

    #[test]
    fn bootstrap_uses_first_gameweek_deadline() {
        let bootstrap = json!({"events": [
            {"id": 2, "deadline_time": "2026-08-28T17:30:00Z"},
            {"id": 1, "deadline_time": "2026-08-21T17:30:00Z"},
        ]});
        assert_eq!(bootstrap_season(&bootstrap).unwrap(), 2026);
    }
}
