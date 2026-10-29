//! Python's datetime text formats, for manifests and log lines that must read the same.

use anyhow::{Result, bail};
use chrono::{DateTime, FixedOffset, NaiveDateTime, Timelike, Utc};

/// `datetime.isoformat()` of a UTC datetime: `2026-10-28T10:22:19.123456+00:00`, without
/// the fraction when the microseconds are zero.
pub fn isoformat(ts: &DateTime<Utc>) -> String {
    let micros = ts.nanosecond() / 1000 % 1_000_000;
    let base = ts.format("%Y-%m-%dT%H:%M:%S");
    if micros == 0 { format!("{base}+00:00") } else { format!("{base}.{micros:06}+00:00") }
}

/// `datetime.fromisoformat` for the forms FPL and ESPN use (`2026-08-21T17:30:00Z`,
/// `2016-08-13T11:30Z`, offsets, fractions), as UTC. Text without an offset is read as UTC,
/// as `deadlines_from_bootstrap` does.
pub fn parse_iso_utc(text: &str) -> Result<DateTime<Utc>> {
    let text = text.trim();
    let normalized = match text.strip_suffix('Z').or_else(|| text.strip_suffix('z')) {
        Some(rest) => format!("{rest}+00:00"),
        None => text.to_owned(),
    };
    for format in ["%Y-%m-%dT%H:%M:%S%.f%:z", "%Y-%m-%dT%H:%M%:z", "%Y-%m-%d %H:%M:%S%.f%:z"] {
        if let Ok(ts) = DateTime::<FixedOffset>::parse_from_str(&normalized, format) {
            return Ok(ts.with_timezone(&Utc));
        }
    }
    for format in ["%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S%.f"] {
        if let Ok(ts) = NaiveDateTime::parse_from_str(&normalized, format) {
            return Ok(ts.and_utc());
        }
    }
    bail!("Invalid isoformat string: {text:?}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn isoformat_like_python() {
        let ts = Utc.with_ymd_and_hms(2026, 10, 28, 10, 22, 19).unwrap();
        assert_eq!(isoformat(&ts), "2026-10-28T10:22:19+00:00");
        let ts = ts + chrono::TimeDelta::microseconds(1234);
        assert_eq!(isoformat(&ts), "2026-10-28T10:22:19.001234+00:00");
    }

    #[test]
    fn parses_fpl_and_espn_forms() {
        let want = Utc.with_ymd_and_hms(2016, 8, 13, 11, 30, 0).unwrap();
        for text in [
            "2016-08-13T11:30Z",
            "2016-08-13T11:30:00Z",
            "2016-08-13T11:30:00+00:00",
            "2016-08-13T12:30:00+01:00",
            "2016-08-13T11:30:00",
            "2016-08-13T11:30:00.000Z",
        ] {
            assert_eq!(parse_iso_utc(text).unwrap(), want, "{text}");
        }
        assert!(parse_iso_utc("13/08/2016").is_err());
    }
}
