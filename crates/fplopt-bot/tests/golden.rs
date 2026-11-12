//! The synthetic bot golden (`tests/fixtures/rust_golden/bot/`: script, state and FPL
//! responses written by Python's `tests/bot_replay.py`; `results.json` and `db.json`
//! recorded from this replay with `FPLOPT_UPDATE_GOLDEN=1`): the scripted session must give
//! the recorded replies and alert texts and the same `bot.sqlite` rows (JSON
//! columns as Python's `json.dumps` text). Covers every command, both alerts, the live prices
//! and their fallback, `/transfer` and `/chips` (Python's tests/test_bot.py, as one session).

use std::path::{Path, PathBuf};

use rusqlite::Connection;
use rusqlite::types::ValueRef;
use serde_json::{Map, Value, json};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.path().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

/// `tests/bot_replay.py` `dump_db`.
fn dump_db(path: &Path) -> Value {
    let db = Connection::open(path).unwrap();
    let mut q = db
        .prepare(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' \
             ORDER BY name",
        )
        .unwrap();
    let tables: Vec<String> = q.query_map([], |r| r.get(0)).unwrap().map(Result::unwrap).collect();
    let mut out = Map::new();
    for table in tables {
        let mut q = db.prepare(&format!("SELECT * FROM {table} ORDER BY rowid")).unwrap();
        let names: Vec<String> = q.column_names().map(|s| s.to_string()).collect();
        let rows: Vec<Value> = q
            .query_map([], |r| {
                let mut row = Map::new();
                for (i, name) in names.iter().enumerate() {
                    if ["created_at", "updated_at", "run_at"].contains(&name.as_str()) {
                        continue;
                    }
                    let value = match r.get_ref(i)? {
                        ValueRef::Null => Value::Null,
                        ValueRef::Integer(n) => json!(n),
                        ValueRef::Real(f) => json!(f),
                        ValueRef::Text(t) => json!(String::from_utf8_lossy(t)),
                        ValueRef::Blob(_) => panic!("a blob"),
                    };
                    row.insert(name.clone(), value);
                }
                Ok(Value::Object(row))
            })
            .unwrap()
            .map(Result::unwrap)
            .collect();
        out.insert(table, Value::Array(rows));
    }
    Value::Object(out)
}

/// Two parsed plan logs equal, `mip_gap` within 1e-9 relative: HiGHS's bookkeeping, which
/// highspy's and our HiGHS build round differently on Linux (dev/bot_parity.py's rule).
fn same_log(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len()
                && x.iter().all(|(k, v)| match (k.as_str(), y.get(k)) {
                    ("mip_gap", Some(w)) if v.is_f64() && w.is_f64() => {
                        let (v, w) = (v.as_f64().unwrap(), w.as_f64().unwrap());
                        (v - w).abs() <= 1e-9 * v.abs().max(w.abs())
                    }
                    (_, Some(w)) => same_log(v, w),
                    (_, None) => false,
                })
        }
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(v, w)| same_log(v, w))
        }
        _ => a == b,
    }
}

/// Two plan log texts (JSON strings) the same (`same_log`).
fn same_plans(a: &Value, b: &Value) -> bool {
    match (a.as_str(), b.as_str()) {
        (Some(a), Some(b)) => match (serde_json::from_str(a), serde_json::from_str(b)) {
            (Ok(a), Ok(b)) => same_log(&a, &b),
            _ => false,
        },
        _ => a == b,
    }
}

/// Where two values differ: around the first differing character for strings (CI cuts long
/// log lines), else both values.
fn difference(rust: &Value, python: &Value) -> String {
    let (Some(a), Some(b)) = (rust.as_str(), python.as_str()) else {
        return format!("\n  rust:   {rust}\n  python: {python}");
    };
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let at = a.iter().zip(&b).position(|(x, y)| x != y).unwrap_or(a.len().min(b.len()));
    let window = |s: &[char]| -> String {
        s[at.saturating_sub(80)..(at + 80).min(s.len())].iter().collect()
    };
    format!("first difference at char {at}\n  rust:   {:?}\n  python: {:?}", window(&a), window(&b))
}

/// Prints the bot's log to stderr, so a failed step shows why (cargo prints a failing
/// test's output).
struct StderrLog;

impl log::Log for StderrLog {
    fn enabled(&self, _: &log::Metadata) -> bool {
        true
    }

    fn log(&self, record: &log::Record) {
        eprintln!("{} {}", record.level(), record.args());
    }

    fn flush(&self) {}
}

#[test]
fn bot_session_matches_golden() {
    let _ = log::set_logger(&StderrLog).map(|()| log::set_max_level(log::LevelFilter::Info));
    let root = repo();
    // Rules and chip values are read relative to the repo root, as the CLI runs.
    std::env::set_current_dir(&root).unwrap();
    let golden = root.join("tests/fixtures/rust_golden/bot");
    let read = |name: &str| -> Value {
        serde_json::from_str(&std::fs::read_to_string(golden.join(name)).unwrap()).unwrap()
    };
    let state = std::env::temp_dir().join(format!("fplopt-bot-golden-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&state);
    copy_dir(&golden.join("state"), &state);

    let results =
        fplopt_bot::replay::replay(&read("script.json"), &state, &golden.join("fpl")).unwrap();
    // /chips in the new wording. Checked on the fresh replay, before any re-recording.
    let chips: Vec<&str> = results
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|r| r.get("replies")?.as_array()?.first()?.get(1)?.as_str())
        .filter(|t| t.starts_with("<b>GW6 chips</b>"))
        .collect();
    assert!(!chips.is_empty());
    for text in chips {
        assert!(
            text.contains(
                " · for information: your /plan doesn't use them
"
            ),
            "{text}"
        );
        assert!(text.contains("<b>Each chip in its best week</b>"), "{text}");
        assert!(
            text.contains(
                "
Chips left: "
            ),
            "{text}"
        );
        assert!(text.contains("A chip only shows a gain if it beats saving it for later."));
        assert!(!text.contains("solver") && !text.contains("xP"), "{text}");
        assert!(text.chars().count() <= fplopt_bot::messages::MAX_MESSAGE);
    }
    if std::env::var("FPLOPT_UPDATE_GOLDEN").is_ok_and(|v| v == "1") {
        // The bot's texts are Rust's own since 2026-11-11 (spec
        // docs/superpowers/specs/2026-11-11-bot-messaging-ux-design.md): re-record.
        let write = |name: &str, value: &Value| {
            let text = serde_json::to_string_pretty(value).unwrap() + "\n";
            std::fs::write(golden.join(name), text).unwrap();
        };
        write("results.json", &results);
        write("db.json", &dump_db(&state.join("bot.sqlite")));
        let _ = std::fs::remove_dir_all(&state);
        return;
    }
    let expected = read("results.json");
    let (got, want) = (results.as_array().unwrap(), expected.as_array().unwrap());
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g, w, "step {i}:\nrust:   {g:#}\ngolden: {w:#}");
    }
    let db = dump_db(&state.join("bot.sqlite"));
    let want = read("db.json");
    for (table, rows) in want.as_object().unwrap() {
        let got = db[table].as_array().unwrap();
        assert_eq!(got.len(), rows.as_array().unwrap().len(), "{table}: row count");
        for (i, (g, w)) in got.iter().zip(rows.as_array().unwrap()).enumerate() {
            let differing: Vec<String> = w
                .as_object()
                .unwrap()
                .iter()
                .filter(|(column, value)| {
                    let got = &g[column.as_str()];
                    got != *value && !(column.as_str() == "plans" && same_plans(got, value))
                })
                .map(|(column, value)| format!("{column}: {}", difference(&g[column], value)))
                .collect();
            assert!(differing.is_empty(), "{table} row {i}:\n{}", differing.join("\n"));
        }
    }
    let _ = std::fs::remove_dir_all(&state);
}

// Python test_every_command_maps_to_a_service_method: every command dispatches (here for a
// chat that never sent /start, without a context).
#[test]
fn every_command_runs() {
    use fplopt_bot::context::ContextStore;
    use fplopt_bot::replay::FileFpl;
    use fplopt_bot::service::{BotService, COMMANDS};
    use fplopt_bot::store::UserStore;

    let dir = std::env::temp_dir().join(format!("fplopt-bot-commands-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let fpl = FileFpl { dir: dir.join("fpl"), down: Default::default() };
    let service = BotService::new(
        UserStore::new(dir.join("bot.sqlite")).unwrap(),
        ContextStore::new(dir.join("context")),
        Box::new(fpl),
        Some(1000),
    );
    for (command, _) in COMMANDS {
        let replies = service.run(command, 5, &[]).unwrap();
        assert!(replies.iter().all(|r| !r.text.is_empty()), "{command}");
    }
    assert!(service.run("nope", 5, &[]).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}
