//! Python's text formats the bot's messages and logs use.

use chrono::{DateTime, Timelike, Utc};
use chrono_tz::Europe::London;

/// Python's `datetime.isoformat()` of a UTC time: microseconds only when non-zero.
pub fn py_isoformat(t: DateTime<Utc>) -> String {
    if t.nanosecond() / 1000 == 0 {
        t.format("%Y-%m-%dT%H:%M:%S+00:00").to_string()
    } else {
        t.format("%Y-%m-%dT%H:%M:%S%.6f+00:00").to_string()
    }
}

/// Python's `datetime.fromisoformat` of an aware time (as UTC); a time without an offset is
/// read as UTC.
pub fn parse_isoformat(text: &str) -> anyhow::Result<DateTime<Utc>> {
    if let Ok(t) = DateTime::parse_from_rfc3339(text) {
        return Ok(t.with_timezone(&Utc));
    }
    if let Ok(t) = DateTime::parse_from_str(text, "%Y-%m-%dT%H:%M:%S%.f%:z") {
        return Ok(t.with_timezone(&Utc));
    }
    let naive = chrono::NaiveDateTime::parse_from_str(text, "%Y-%m-%dT%H:%M:%S%.f")
        .map_err(|e| anyhow::anyhow!("bad time {text:?}: {e}"))?;
    Ok(naive.and_utc())
}

/// 'Sat 17 Oct 11:00' in UK time.
pub fn uk_time(when: DateTime<Utc>) -> String {
    when.with_timezone(&London).format("%a %d %b %H:%M").to_string()
}

/// Python's `html.escape(text, quote=False)`.
pub fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            c => out.push(c),
        }
    }
    out
}

/// Python's `str.casefold()` for the names the bot matches: lower case, with the full case
/// foldings that lower case misses.
pub fn casefold(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            'ß' | 'ẞ' => out.push_str("ss"),
            'ς' => out.push('σ'),
            'ſ' => out.push('s'),
            c => out.extend(c.to_lowercase()),
        }
    }
    out
}

/// Python's `repr` of a string.
pub fn py_repr_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::from(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32))
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Python's `format(x, "g")`: 6 significant digits, trailing zeros dropped, scientific below
/// 1e-4 or from 1e6.
pub fn py_g(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0".into() } else { "0".into() };
    }
    let sci = format!("{x:.5e}"); // e.g. "1.50000e-5"
    let (mantissa, exp) = sci.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    if (-4..6).contains(&exp) {
        let decimals = (5 - exp).max(0) as usize;
        let fixed = format!("{x:.decimals$}");
        trim_zeros(&fixed)
    } else {
        let esign = if exp < 0 { '-' } else { '+' };
        format!("{}e{esign}{:02}", trim_zeros(mantissa), exp.abs())
    }
}

fn trim_zeros(s: &str) -> String {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s.to_string()
    }
}

/// Python's `str.split()`: runs of whitespace (`str.isspace`, which includes U+001C..U+001F).
pub fn py_split(text: &str) -> Vec<&str> {
    text.split(|c: char| c.is_whitespace() || (''..='').contains(&c))
        .filter(|w| !w.is_empty())
        .collect()
}

/// Python's `str.isdigit()` for ASCII input (what `int()` then accepts).
pub fn is_digits(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats() {
        assert_eq!(py_g(10.0), "10");
        assert_eq!(py_g(5.5), "5.5");
        assert_eq!(py_g(0.0001), "0.0001");
        assert_eq!(py_g(1e-5), "1e-05");
        assert_eq!(py_g(1234567.0), "1.23457e+06");
        assert_eq!(py_g(123456.0), "123456");
        assert_eq!(py_g(2.0 / 3.0), "0.666667");
        let t = parse_isoformat("2026-10-31T11:00:00+00:00").unwrap();
        assert_eq!(uk_time(t), "Sat 31 Oct 11:00"); // after the clocks went back
        assert_eq!(uk_time(parse_isoformat("2026-10-17T10:00:00Z").unwrap()), "Sat 17 Oct 11:00");
        assert_eq!(py_isoformat(t), "2026-10-31T11:00:00+00:00");
        let micro = parse_isoformat("2026-10-31T11:00:00.000123+00:00").unwrap();
        assert_eq!(py_isoformat(micro), "2026-10-31T11:00:00.000123+00:00");
        assert_eq!(parse_isoformat("2026-10-31T11:00:00").unwrap(), t);
        assert_eq!(escape("a<b>&\"c'"), "a&lt;b&gt;&amp;\"c'");
        assert_eq!(casefold("Straße ÖDEGAARD"), "strasse ödegaard");
        assert_eq!(py_repr_str("O'Brien"), "\"O'Brien\"");
        assert_eq!(py_repr_str("Saka"), "'Saka'");
    }
}
