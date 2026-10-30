//! The synthetic bot golden (`tests/fixtures/rust_golden/bot/`, written by Python's
//! `tests/bot_replay.py`): the scripted session replayed through the Rust bot must give
//! Python's replies and alert texts byte for byte, and the same `bot.sqlite` rows (JSON
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
fn bot_session_matches_python() {
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
    let expected = read("results.json");
    let (got, want) = (results.as_array().unwrap(), expected.as_array().unwrap());
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g, w, "step {i}:\nrust:   {g:#}\npython: {w:#}");
    }
    let db = dump_db(&state.join("bot.sqlite"));
    let want = read("db.json");
    for (table, rows) in want.as_object().unwrap() {
        let got = db[table].as_array().unwrap();
        assert_eq!(got.len(), rows.as_array().unwrap().len(), "{table}: row count");
        for (i, (g, w)) in got.iter().zip(rows.as_array().unwrap()).enumerate() {
            assert_eq!(g, w, "{table} row {i}");
        }
    }
    assert_eq!(db, want);
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
