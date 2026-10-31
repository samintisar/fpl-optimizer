//! User state in SQLite (Python `fplopt.bot.store`; PLAN §2-3, Phase 7 plan §2): who may use
//! the bot, their FPL team, settings, per-GW overrides, the imported squads and every
//! recommendation made (the live log of PLAN §8, with the plan contexts).
//!
//! Everything is keyed by `user_id` (CLAUDE.md); the Telegram `chat_id` is only how a user is
//! reached. The same file and schema as Python's: one connection per call, WAL mode, so the
//! bot and the alert job can both write; each call is one transaction.

use std::path::{Path, PathBuf};
use std::thread::sleep;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use fplopt_core::pyjson;
use log::warn;
use rusqlite::{Connection, OptionalExtension, Row, params};
use serde::Serialize;
use serde_json::Value;

use crate::util::py_isoformat;

pub const STATUSES: [&str; 3] = ["pending", "active", "blocked"];
/// Waits before retrying a connection that hit SQLite's "disk I/O error" (seen twice on
/// ubuntuworker, 2026-10-21, opening the database; transient).
const OPEN_RETRIES: [f64; 2] = [1.0, 3.0];
/// Recommendation kinds: the 24-hour alert, the silent re-run before the deadline, /plan.
pub const KINDS: [&str; 3] = ["alert24", "alert2", "plan"];

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL UNIQUE,
    status TEXT NOT NULL CHECK (status IN ('pending', 'active', 'blocked')),
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS user_settings (
    user_id INTEGER PRIMARY KEY REFERENCES users (user_id),
    entry_id INTEGER,
    alerts INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS user_overrides (
    user_id INTEGER NOT NULL REFERENCES users (user_id),
    season INTEGER NOT NULL,
    gw INTEGER NOT NULL,
    free_transfers INTEGER,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, season, gw)
);
CREATE TABLE IF NOT EXISTS user_state (
    user_id INTEGER NOT NULL REFERENCES users (user_id),
    season INTEGER NOT NULL,
    gw INTEGER NOT NULL,
    entry_id INTEGER NOT NULL,
    squad TEXT NOT NULL,
    context_id TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, season, gw)
);
CREATE TABLE IF NOT EXISTS recommendations (
    rec_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users (user_id),
    season INTEGER NOT NULL,
    gw INTEGER NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('alert24', 'alert2', 'plan')),
    context_id TEXT,
    created_at TEXT NOT NULL,
    plans TEXT,
    decision TEXT,
    message TEXT,
    sent INTEGER NOT NULL DEFAULT 0,
    error TEXT
);
CREATE INDEX IF NOT EXISTS recommendations_user_gw
    ON recommendations (user_id, season, gw, kind);
CREATE TABLE IF NOT EXISTS user_transfers (
    transfer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users (user_id),
    season INTEGER NOT NULL,
    gw INTEGER NOT NULL,
    element_out INTEGER NOT NULL,
    element_in INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chip_checks (
    check_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users (user_id),
    season INTEGER NOT NULL,
    gw INTEGER NOT NULL,
    context_id TEXT,
    created_at TEXT NOT NULL,
    best TEXT NOT NULL,
    scenarios TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS alert_runs (
    season INTEGER NOT NULL,
    gw INTEGER NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('alert24', 'alert2')),
    context_id TEXT NOT NULL,
    bootstrap_at TEXT NOT NULL,
    run_at TEXT NOT NULL,
    PRIMARY KEY (season, gw, kind)
);
";

/// Python's `datetime.now(UTC).isoformat(timespec="seconds")`.
fn now() -> String {
    Utc::now().format("%Y-%m-%dT%H:%M:%S+00:00").to_string()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct User {
    pub user_id: i64,
    pub chat_id: i64,
    pub status: String,
    pub entry_id: Option<i64>,
    pub alerts: bool,
}

impl User {
    pub fn active(&self) -> bool {
        self.status == "active"
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Recommendation {
    pub rec_id: i64,
    pub user_id: i64,
    pub season: i64,
    pub gw: i64,
    pub kind: String,
    pub context_id: Option<String>,
    pub created_at: String,
    pub decision: Option<Value>,
    pub sent: bool,
    pub error: Option<String>,
}

/// A recommendation to log (`UserStore::record`): JSON already in Python's `json.dumps` form.
#[derive(Clone, Debug, Default)]
pub struct Record {
    pub context_id: Option<String>,
    pub plans: Option<String>,
    pub decision: Option<String>,
    pub message: Option<String>,
    pub sent: bool,
    pub error: Option<String>,
}

/// One `chip_checks` row (`scenarios` parsed).
#[derive(Clone, Debug, PartialEq)]
pub struct ChipCheck {
    pub check_id: i64,
    pub user_id: i64,
    pub season: i64,
    pub gw: i64,
    pub context_id: Option<String>,
    pub created_at: String,
    pub best: String,
    pub scenarios: Value,
}

/// An `alert_runs` row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AlertRun {
    pub context_id: String,
    pub bootstrap_at: String,
    pub run_at: String,
}

/// `json.dumps(value)`.
pub fn dumps<T: Serialize + ?Sized>(value: &T) -> Result<String> {
    pyjson::dumps(value)
}

/// The bot's SQLite database at `path` (created with its tables on first use).
pub struct UserStore {
    pub path: PathBuf,
}

const USER_COLUMNS: &str = "SELECT u.user_id, u.chat_id, u.status, s.entry_id, s.alerts FROM users u \
                            LEFT JOIN user_settings s USING (user_id)";

impl UserStore {
    pub fn new(path: impl Into<PathBuf>) -> Result<UserStore> {
        let store = UserStore { path: path.into() };
        if let Some(parent) = store.path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        // WAL is a property of the database file: set once here, not on every connection
        // (switching it needs a write lock).
        store.open()?.pragma_update(None, "journal_mode", "WAL")?;
        store.with(|db| Ok(db.execute_batch(SCHEMA)?))?;
        Ok(store)
    }

    /// A connection that has read the database (its journal mode), retrying a transient disk
    /// I/O error (`OPEN_RETRIES`).
    fn open(&self) -> Result<Connection> {
        let mut waits = OPEN_RETRIES.iter();
        loop {
            let attempt = Connection::open(&self.path).and_then(|db| {
                db.busy_timeout(Duration::from_secs(30))?;
                db.query_row("PRAGMA journal_mode", [], |r| r.get::<_, String>(0))?;
                Ok(db)
            });
            match attempt {
                Ok(db) => return Ok(db),
                Err(e) => match waits.next() {
                    Some(wait) if e.to_string().contains("disk I/O error") => {
                        warn!("bot database: {e}; retrying in {wait:.0} s");
                        sleep(Duration::from_secs_f64(*wait));
                    }
                    _ => {
                        return Err(e)
                            .with_context(|| format!("bot database {}", self.path.display()));
                    }
                },
            }
        }
    }

    /// Run `f` in one transaction (committed on success, rolled back on error).
    fn with<T>(&self, f: impl FnOnce(&Connection) -> Result<T>) -> Result<T> {
        let mut db = self.open()?;
        db.pragma_update(None, "foreign_keys", "ON")?;
        let tx = db.transaction()?;
        let out = f(&tx)?;
        tx.commit()?;
        Ok(out)
    }

    // --- users ------------------------------------------------------------------------------

    pub fn user(&self, chat_id: i64) -> Result<Option<User>> {
        self.with(|db| {
            Ok(db
                .query_row(&format!("{USER_COLUMNS} WHERE u.chat_id = ?"), [chat_id], user_row)
                .optional()?)
        })
    }

    /// The user for `chat_id`, created with `status` if new (an existing user is returned
    /// unchanged).
    pub fn add_user(&self, chat_id: i64, status: &str) -> Result<User> {
        check_status(status)?;
        self.with(|db| {
            db.execute(
                "INSERT OR IGNORE INTO users (chat_id, status, created_at) VALUES (?, ?, ?)",
                params![chat_id, status, now()],
            )?;
            Ok(())
        })?;
        self.user(chat_id)?.context("user just added")
    }

    /// Set an existing user's status; None if there is no such user.
    pub fn set_status(&self, chat_id: i64, status: &str) -> Result<Option<User>> {
        check_status(status)?;
        self.with(|db| {
            db.execute("UPDATE users SET status = ? WHERE chat_id = ?", params![status, chat_id])?;
            Ok(())
        })?;
        self.user(chat_id)
    }

    pub fn users(&self, status: Option<&str>) -> Result<Vec<User>> {
        self.with(|db| {
            let rows = match status {
                Some(s) => {
                    let mut q = db.prepare(&format!(
                        "{USER_COLUMNS} WHERE u.status = ? ORDER BY u.user_id"
                    ))?;
                    q.query_map([s], user_row)?.collect::<rusqlite::Result<Vec<_>>>()?
                }
                None => {
                    let mut q = db.prepare(&format!("{USER_COLUMNS} ORDER BY u.user_id"))?;
                    q.query_map([], user_row)?.collect::<rusqlite::Result<Vec<_>>>()?
                }
            };
            Ok(rows)
        })
    }

    /// Active users with a team and alerts on.
    pub fn alert_users(&self) -> Result<Vec<User>> {
        Ok(self
            .users(Some("active"))?
            .into_iter()
            .filter(|u| u.entry_id.is_some() && u.alerts)
            .collect())
    }

    // --- settings and overrides -------------------------------------------------------------

    fn settings(db: &Connection, user_id: i64, key: &str, value: i64) -> Result<()> {
        db.execute(
            "INSERT INTO user_settings (user_id, updated_at) VALUES (?, ?) \
             ON CONFLICT (user_id) DO NOTHING",
            params![user_id, now()],
        )?;
        db.execute(
            &format!("UPDATE user_settings SET {key} = ?, updated_at = ? WHERE user_id = ?"),
            params![value, now(), user_id],
        )?;
        Ok(())
    }

    pub fn set_entry(&self, user_id: i64, entry_id: i64) -> Result<()> {
        self.with(|db| Self::settings(db, user_id, "entry_id", entry_id))
    }

    pub fn set_alerts(&self, user_id: i64, on: bool) -> Result<()> {
        self.with(|db| Self::settings(db, user_id, "alerts", i64::from(on)))
    }

    /// The free transfers to plan GW `gw` with (None clears the override).
    pub fn set_free_transfers(
        &self,
        user_id: i64,
        season: i64,
        gw: i64,
        n: Option<i64>,
    ) -> Result<()> {
        self.with(|db| {
            db.execute(
                "INSERT INTO user_overrides (user_id, season, gw, free_transfers, updated_at) \
                 VALUES (?, ?, ?, ?, ?) ON CONFLICT (user_id, season, gw) DO UPDATE SET \
                 free_transfers = excluded.free_transfers, updated_at = excluded.updated_at",
                params![user_id, season, gw, n, now()],
            )?;
            Ok(())
        })
    }

    pub fn free_transfers(&self, user_id: i64, season: i64, gw: i64) -> Result<Option<i64>> {
        self.with(|db| {
            let n: Option<Option<i64>> = db
                .query_row(
                    "SELECT free_transfers FROM user_overrides \
                     WHERE user_id = ? AND season = ? AND gw = ?",
                    params![user_id, season, gw],
                    |r| r.get(0),
                )
                .optional()?;
            Ok(n.flatten())
        })
    }

    /// A transfer the user already made for GW `gw` (element ids), after any before it.
    pub fn add_transfer(
        &self,
        user_id: i64,
        season: i64,
        gw: i64,
        out: i64,
        into: i64,
    ) -> Result<()> {
        self.with(|db| {
            db.execute(
                "INSERT INTO user_transfers (user_id, season, gw, element_out, element_in, \
                 created_at) VALUES (?, ?, ?, ?, ?, ?)",
                params![user_id, season, gw, out, into, now()],
            )?;
            Ok(())
        })
    }

    /// The transfers the user made for GW `gw`, in order: (element out, element in).
    pub fn transfers(&self, user_id: i64, season: i64, gw: i64) -> Result<Vec<(i64, i64)>> {
        self.with(|db| {
            let mut q = db.prepare(
                "SELECT element_out, element_in FROM user_transfers WHERE user_id = ? AND \
                 season = ? AND gw = ? ORDER BY transfer_id",
            )?;
            let rows = q
                .query_map(params![user_id, season, gw], |r| Ok((r.get(0)?, r.get(1)?)))?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            Ok(rows)
        })
    }

    /// Remove the GW's transfers (or only the newest); returns how many were removed.
    pub fn drop_transfers(
        &self,
        user_id: i64,
        season: i64,
        gw: i64,
        last_only: bool,
    ) -> Result<usize> {
        self.with(|db| {
            let sql = if last_only {
                "DELETE FROM user_transfers WHERE transfer_id = (SELECT MAX(transfer_id) \
                 FROM user_transfers WHERE user_id = ? AND season = ? AND gw = ?)"
            } else {
                "DELETE FROM user_transfers WHERE user_id = ? AND season = ? AND gw = ?"
            };
            Ok(db.execute(sql, params![user_id, season, gw])?)
        })
    }

    // --- squads and recommendations ---------------------------------------------------------

    /// Save the imported squad (`squad`: its JSON, `json.dumps` form).
    pub fn save_squad(
        &self,
        user_id: i64,
        season: i64,
        gw: i64,
        entry_id: i64,
        squad: &str,
        context_id: Option<&str>,
    ) -> Result<()> {
        self.with(|db| {
            db.execute(
                "INSERT INTO user_state (user_id, season, gw, entry_id, squad, context_id, \
                 updated_at) VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT (user_id, season, gw) \
                 DO UPDATE SET entry_id = excluded.entry_id, squad = excluded.squad, \
                 context_id = excluded.context_id, updated_at = excluded.updated_at",
                params![user_id, season, gw, entry_id, squad, context_id, now()],
            )?;
            Ok(())
        })
    }

    pub fn squad(&self, user_id: i64, season: i64, gw: i64) -> Result<Option<Value>> {
        let text: Option<String> = self.with(|db| {
            Ok(db
                .query_row(
                    "SELECT squad FROM user_state WHERE user_id = ? AND season = ? AND gw = ?",
                    params![user_id, season, gw],
                    |r| r.get(0),
                )
                .optional()?)
        })?;
        text.map(|t| Ok(serde_json::from_str(&t)?)).transpose()
    }

    /// Log a recommendation (or a failed attempt, with `error`); returns its rec_id.
    pub fn record(
        &self,
        user_id: i64,
        season: i64,
        gw: i64,
        kind: &str,
        r: &Record,
    ) -> Result<i64> {
        if !KINDS.contains(&kind) {
            bail!("unknown kind '{kind}'");
        }
        self.with(|db| {
            db.execute(
                "INSERT INTO recommendations (user_id, season, gw, kind, context_id, \
                 created_at, plans, decision, message, sent, error) \
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                params![
                    user_id,
                    season,
                    gw,
                    kind,
                    r.context_id,
                    now(),
                    r.plans,
                    r.decision,
                    r.message,
                    i64::from(r.sent),
                    r.error
                ],
            )?;
            Ok(db.last_insert_rowid())
        })
    }

    /// The newest alert (24-hour or re-run) sent to the user for this GW.
    pub fn last_sent_alert(
        &self,
        user_id: i64,
        season: i64,
        gw: i64,
    ) -> Result<Option<Recommendation>> {
        self.with(|db| {
            Ok(db
                .query_row(
                    "SELECT * FROM recommendations WHERE user_id = ? AND season = ? AND gw = ? \
                     AND kind IN ('alert24', 'alert2') AND sent = 1 ORDER BY rec_id DESC LIMIT 1",
                    params![user_id, season, gw],
                    recommendation_row,
                )
                .optional()?)
        })?
        .map(finish_recommendation)
        .transpose()
    }

    pub fn recommendations(&self, user_id: i64) -> Result<Vec<Recommendation>> {
        let rows = self.with(|db| {
            let mut q =
                db.prepare("SELECT * FROM recommendations WHERE user_id = ? ORDER BY rec_id")?;
            let rows = q
                .query_map([user_id], recommendation_row)?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            Ok(rows)
        })?;
        rows.into_iter().map(finish_recommendation).collect()
    }

    /// Log a `/chips` answer: the best scenario and every solved one's objective
    /// (`scenarios`: (label, objective) in Python's dict order).
    pub fn record_chip_check(
        &self,
        user_id: i64,
        season: i64,
        gw: i64,
        context_id: Option<&str>,
        best: &str,
        scenarios: &[(String, f64)],
    ) -> Result<()> {
        let scenarios = ordered_object(scenarios);
        self.with(|db| {
            db.execute(
                "INSERT INTO chip_checks (user_id, season, gw, context_id, created_at, best, \
                 scenarios) VALUES (?, ?, ?, ?, ?, ?, ?)",
                params![user_id, season, gw, context_id, now(), best, scenarios],
            )?;
            Ok(())
        })
    }

    pub fn chip_checks(&self, user_id: i64) -> Result<Vec<ChipCheck>> {
        let rows: Vec<(ChipCheck, String)> = self.with(|db| {
            let mut q =
                db.prepare("SELECT * FROM chip_checks WHERE user_id = ? ORDER BY check_id")?;
            let rows = q
                .query_map([user_id], |r| {
                    Ok((
                        ChipCheck {
                            check_id: r.get("check_id")?,
                            user_id: r.get("user_id")?,
                            season: r.get("season")?,
                            gw: r.get("gw")?,
                            context_id: r.get("context_id")?,
                            created_at: r.get("created_at")?,
                            best: r.get("best")?,
                            scenarios: Value::Null,
                        },
                        r.get::<_, String>("scenarios")?,
                    ))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            Ok(rows)
        })?;
        rows.into_iter()
            .map(|(mut c, text)| {
                c.scenarios = serde_json::from_str(&text)?;
                Ok(c)
            })
            .collect()
    }

    // --- alert runs -------------------------------------------------------------------------

    pub fn alert_run(&self, season: i64, gw: i64, kind: &str) -> Result<Option<AlertRun>> {
        self.with(|db| {
            Ok(db
                .query_row(
                    "SELECT context_id, bootstrap_at, run_at FROM alert_runs \
                     WHERE season = ? AND gw = ? AND kind = ?",
                    params![season, gw, kind],
                    |r| {
                        Ok(AlertRun {
                            context_id: r.get(0)?,
                            bootstrap_at: r.get(1)?,
                            run_at: r.get(2)?,
                        })
                    },
                )
                .optional()?)
        })
    }

    /// Mark the GW's alert of `kind` done; a second mark (e.g. a manual `live tick` next to
    /// the timer's) keeps the first.
    pub fn mark_alert_run(
        &self,
        season: i64,
        gw: i64,
        kind: &str,
        context_id: &str,
        bootstrap_at: DateTime<Utc>,
    ) -> Result<()> {
        self.with(|db| {
            db.execute(
                "INSERT INTO alert_runs (season, gw, kind, context_id, bootstrap_at, run_at) \
                 VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (season, gw, kind) DO NOTHING",
                params![season, gw, kind, context_id, py_isoformat(bootstrap_at), now()],
            )?;
            Ok(())
        })
    }
}

fn check_status(status: &str) -> Result<()> {
    if !STATUSES.contains(&status) {
        bail!("unknown status '{status}'");
    }
    Ok(())
}

/// `json.dumps(dict(pairs))`: keys in order, a repeated key keeping its first position and
/// last value (Python dict semantics).
pub fn ordered_object(pairs: &[(String, f64)]) -> String {
    let mut keys: Vec<&str> = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    for (k, v) in pairs {
        match keys.iter().position(|x| x == k) {
            Some(i) => values[i] = *v,
            None => {
                keys.push(k);
                values.push(*v);
            }
        }
    }
    let parts: Vec<String> = keys
        .iter()
        .zip(&values)
        .map(|(k, v)| format!("{}: {}", dumps(k).expect("a string"), dumps(v).expect("a float")))
        .collect();
    format!("{{{}}}", parts.join(", "))
}

fn user_row(r: &Row) -> rusqlite::Result<User> {
    Ok(User {
        user_id: r.get(0)?,
        chat_id: r.get(1)?,
        status: r.get(2)?,
        entry_id: r.get(3)?,
        alerts: r.get::<_, Option<i64>>(4)?.is_none_or(|a| a != 0),
    })
}

type RawRecommendation = (Recommendation, Option<String>);

fn recommendation_row(r: &Row) -> rusqlite::Result<RawRecommendation> {
    Ok((
        Recommendation {
            rec_id: r.get("rec_id")?,
            user_id: r.get("user_id")?,
            season: r.get("season")?,
            gw: r.get("gw")?,
            kind: r.get("kind")?,
            context_id: r.get("context_id")?,
            created_at: r.get("created_at")?,
            decision: None,
            sent: r.get::<_, i64>("sent")? != 0,
            error: r.get("error")?,
        },
        r.get("decision")?,
    ))
}

fn finish_recommendation((mut rec, decision): RawRecommendation) -> Result<Recommendation> {
    rec.decision = decision.map(|d| serde_json::from_str(&d)).transpose()?;
    Ok(rec)
}

/// The path of the bot's database under a state directory.
pub fn database_path(state_dir: &Path) -> PathBuf {
    state_dir.join("bot.sqlite")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (tempfile_dir::TempDir, UserStore) {
        let dir = tempfile_dir::TempDir::new();
        let users = UserStore::new(dir.path().join("state").join("bot.sqlite")).unwrap();
        (dir, users)
    }

    /// A throwaway directory (no tempfile dependency).
    mod tempfile_dir {
        use std::path::{Path, PathBuf};
        use std::sync::atomic::{AtomicUsize, Ordering};

        pub struct TempDir(PathBuf);

        impl TempDir {
            pub fn new() -> TempDir {
                static N: AtomicUsize = AtomicUsize::new(0);
                let path = std::env::temp_dir().join(format!(
                    "fplopt-bot-test-{}-{}",
                    std::process::id(),
                    N.fetch_add(1, Ordering::SeqCst)
                ));
                std::fs::create_dir_all(&path).unwrap();
                TempDir(path)
            }

            pub fn path(&self) -> &Path {
                &self.0
            }
        }

        impl Drop for TempDir {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
    }

    // Python test_user_store_lifecycle.
    #[test]
    fn user_store_lifecycle() {
        let (_dir, users) = store();
        assert_eq!(users.user(5).unwrap(), None);
        let user = users.add_user(5, "pending").unwrap();
        assert_eq!(
            (user.chat_id, user.status.as_str(), user.entry_id, user.alerts),
            (5, "pending", None, true)
        );
        assert_eq!(users.add_user(5, "active").unwrap(), user); // existing: unchanged
        assert!(users.set_status(5, "active").unwrap().unwrap().active());
        users.set_entry(user.user_id, 123).unwrap();
        users.set_alerts(user.user_id, false).unwrap();
        let again = users.user(5).unwrap().unwrap();
        assert_eq!((again.entry_id, again.alerts), (Some(123), false));
        assert!(users.alert_users().unwrap().is_empty());
        users.set_alerts(user.user_id, true).unwrap();
        assert_eq!(users.alert_users().unwrap().iter().map(|u| u.chat_id).collect::<Vec<_>>(), [5]);
        assert!(users.add_user(6, "nope").is_err());
        assert_eq!(users.set_status(99, "active").unwrap(), None);
        users.add_user(6, "blocked").unwrap();
        let statuses: Vec<_> = users.users(None).unwrap().into_iter().map(|u| u.status).collect();
        assert_eq!(statuses, ["active", "blocked"]);
        assert_eq!(users.users(Some("blocked")).unwrap().len(), 1);
    }

    // Python test_user_store_overrides_and_recommendations.
    #[test]
    fn overrides_and_recommendations() {
        let (_dir, users) = store();
        let uid = users.add_user(5, "active").unwrap().user_id;
        assert_eq!(users.free_transfers(uid, 2023, 6).unwrap(), None);
        users.set_free_transfers(uid, 2023, 6, Some(2)).unwrap();
        assert_eq!(users.free_transfers(uid, 2023, 6).unwrap(), Some(2));
        users.set_free_transfers(uid, 2023, 6, None).unwrap();
        assert_eq!(users.free_transfers(uid, 2023, 6).unwrap(), None);

        users
            .save_squad(uid, 2023, 6, 7, r#"{"players": [[1, 50]], "bank": 5}"#, Some("c1"))
            .unwrap();
        assert_eq!(users.squad(uid, 2023, 6).unwrap().unwrap()["bank"], 5);
        assert_eq!(users.squad(uid, 2023, 7).unwrap(), None);

        let decision = r#"{"transfers": [], "captain": 1, "vice": 2}"#.to_string();
        let first = users
            .record(
                uid,
                2023,
                6,
                "alert24",
                &Record {
                    context_id: Some("c1".into()),
                    decision: Some(decision.clone()),
                    message: Some("hi".into()),
                    sent: true,
                    ..Record::default()
                },
            )
            .unwrap();
        users
            .record(
                uid,
                2023,
                6,
                "alert2",
                &Record {
                    context_id: Some("c1".into()),
                    error: Some("boom".into()),
                    ..Record::default()
                },
            )
            .unwrap();
        assert!(users.record(uid, 2023, 6, "nope", &Record::default()).is_err());
        let last = users.last_sent_alert(uid, 2023, 6).unwrap().unwrap();
        assert_eq!(last.rec_id, first);
        assert_eq!(last.decision.unwrap()["captain"], 1);
        assert_eq!(users.recommendations(uid).unwrap().len(), 2);

        users.add_transfer(uid, 2023, 6, 1, 2).unwrap();
        users.add_transfer(uid, 2023, 6, 3, 4).unwrap();
        assert_eq!(users.transfers(uid, 2023, 6).unwrap(), [(1, 2), (3, 4)]);
        assert_eq!(users.drop_transfers(uid, 2023, 6, true).unwrap(), 1);
        assert_eq!(users.transfers(uid, 2023, 6).unwrap(), [(1, 2)]);
        assert_eq!(users.drop_transfers(uid, 2023, 6, false).unwrap(), 1);

        users
            .record_chip_check(
                uid,
                2023,
                6,
                Some("c1"),
                "none",
                &[("none".into(), 1.5), ("bboost@gw6".into(), 1e-5)],
            )
            .unwrap();
        let checks = users.chip_checks(uid).unwrap();
        assert_eq!(checks[0].scenarios["bboost@gw6"], 1e-5);

        let at = DateTime::parse_from_rfc3339("2023-10-01T10:00:00Z").unwrap().with_timezone(&Utc);
        assert_eq!(users.alert_run(2023, 6, "alert24").unwrap(), None);
        users.mark_alert_run(2023, 6, "alert24", "c1", at).unwrap();
        users.mark_alert_run(2023, 6, "alert24", "c2", at).unwrap();
        let run = users.alert_run(2023, 6, "alert24").unwrap().unwrap();
        assert_eq!(
            (run.context_id.as_str(), run.bootstrap_at.as_str()),
            ("c1", "2023-10-01T10:00:00+00:00")
        );
    }

    #[test]
    fn ordered_object_is_a_python_dict() {
        let pairs = [("b".to_string(), 1.0), ("a".to_string(), 2.5), ("b".to_string(), 3.0)];
        assert_eq!(ordered_object(&pairs), r#"{"b": 3.0, "a": 2.5}"#);
    }
}
