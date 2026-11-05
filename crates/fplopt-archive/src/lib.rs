//! The archiver (migration batch 1): source adapters and the jobs that append their
//! responses to `raw/`. Port of `fplopt.adapters` and `fplopt.ingest`, writing the same
//! paths, timestamps and file contents.

pub mod adapters;
pub mod espn;
pub mod fetch;
pub mod health;
pub mod history;
pub mod jobs;
pub mod pytime;
pub mod raw_store;
pub mod schedule;
pub mod stop;

use std::fmt;
use std::time::Duration;

use chrono::{DateTime, DurationRound, TimeDelta, Utc};

/// Time and pauses, injectable for tests (Python's `now`/`sleep` arguments).
pub trait Clock {
    fn now(&self) -> DateTime<Utc>;
    fn sleep(&self, duration: Duration);
}

/// The wall clock, at Python's microsecond resolution.
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> DateTime<Utc> {
        let now = Utc::now();
        now.duration_trunc(TimeDelta::microseconds(1)).unwrap_or(now)
    }

    fn sleep(&self, duration: Duration) {
        std::thread::sleep(duration);
    }
}

/// An error with the name of the Python exception it stands for (`ValueError`,
/// `RuntimeError`, `SnapshotError`, ...), so failure lines and manifests read the same as
/// Python's `"<name>: <Type>: <message>"`.
#[derive(Debug)]
pub struct Failure {
    pub kind: &'static str,
    pub message: String,
}

impl Failure {
    pub fn error(kind: &'static str, message: impl Into<String>) -> anyhow::Error {
        anyhow::Error::new(Self { kind, message: message.into() })
    }
}

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Failure {}

/// The Python exception name an error stands for.
pub fn kind_name(err: &anyhow::Error) -> &'static str {
    if let Some(failure) = err.downcast_ref::<Failure>() {
        failure.kind
    } else if let Some(fetch) = err.downcast_ref::<fetch::FetchError>() {
        fetch.kind()
    } else if err.is::<stop::Terminated>() {
        "SystemExit"
    } else if err.is::<serde_json::Error>() {
        "JSONDecodeError"
    } else if let Some(io) = err.downcast_ref::<std::io::Error>() {
        io_kind(io)
    } else if let Some(csv) = err.downcast_ref::<csv::Error>() {
        match csv.kind() {
            csv::ErrorKind::Io(io) => io_kind(io),
            _ => "Error",
        }
    } else {
        "Error"
    }
}

fn io_kind(io: &std::io::Error) -> &'static str {
    match io.kind() {
        std::io::ErrorKind::AlreadyExists => "FileExistsError",
        std::io::ErrorKind::NotFound => "FileNotFoundError",
        std::io::ErrorKind::PermissionDenied => "PermissionError",
        std::io::ErrorKind::UnexpectedEof => "EOFError",
        _ => "OSError",
    }
}

/// The first line of the error's own message ("" if it has none).
pub fn summary(err: &anyhow::Error) -> String {
    err.to_string().lines().next().unwrap_or("").to_owned()
}

/// `"<Type>: <first line>"`, or just the type for an empty message (Python's `_one_line`).
pub fn one_line(err: &anyhow::Error) -> String {
    let summary = summary(err);
    if summary.is_empty() {
        kind_name(err).to_owned()
    } else {
        format!("{}: {summary}", kind_name(err))
    }
}

/// Python's `repr` of a list of strings: `['a', "it's"]`.
pub fn py_repr_strs<S: AsRef<str>>(items: &[S]) -> String {
    let parts: Vec<String> = items.iter().map(|s| py_repr_str(s.as_ref())).collect();
    format!("[{}]", parts.join(", "))
}

/// Python's `repr` of a string (printable text; other control characters as `\xNN`).
pub fn py_repr_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
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
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Python's `repr` of a list of ints: `[1, 2]`.
pub fn py_repr_ints(items: &[i64]) -> String {
    let parts: Vec<String> = items.iter().map(i64::to_string).collect();
    format!("[{}]", parts.join(", "))
}

/// Python truthiness of a JSON value.
pub fn truthy(value: &serde_json::Value) -> bool {
    use serde_json::Value;
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|x| x != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn reprs_match_python() {
        assert_eq!(py_repr_strs(&["a", "it's", "x\\y"]), r#"['a', "it's", 'x\\y']"#);
        assert_eq!(py_repr_ints(&[1, 22]), "[1, 22]");
        assert_eq!(py_repr_ints(&[]), "[]");
    }

    #[test]
    fn one_line_names_the_kind() {
        let err = Failure::error("ValueError", "bad\nsecond line");
        assert_eq!(one_line(&err), "ValueError: bad");
        assert_eq!(one_line(&Failure::error("RuntimeError", "")), "RuntimeError");
        let io = anyhow::Error::from(std::io::Error::from(std::io::ErrorKind::AlreadyExists));
        assert_eq!(kind_name(&io), "FileExistsError");
    }

    #[test]
    fn truthiness() {
        for (value, expected) in [
            (json!(null), false),
            (json!(0), false),
            (json!(2), true),
            (json!(""), false),
            (json!([]), false),
            (json!([0]), true),
            (json!({}), false),
            (json!(true), true),
        ] {
            assert_eq!(truthy(&value), expected, "{value}");
        }
    }
}
