//! A scripted bot session on recorded FPL responses, for the batch 5 parity gate
//! (`dev/bot_parity.py`) and the golden test: Python's `tests/bot_replay.py` runs the same
//! script through Python's `BotService`, and the gate compares the replies, the alert texts
//! and the two `bot.sqlite` files.
//!
//! The script (JSON): `admin` (the admin chat id), optional `params` / `chip_params` (fields
//! replaced in `live_params()` / `chip_params()` for `/plan` and alerts / `/chips`), and
//! `steps`, each one of
//! - `{"now": ISO time}`: the clock from here on;
//! - `{"fpl_down": bool}`: FPL's bootstrap (live prices) fails from here on;
//! - `{"chat": id, "text": message}`: a message, routed as the Telegram layer routes text
//!   (`parse_command`, else "Unknown command");
//! - `{"alert": chat id, "kind": "alert24"|"alert2"}`: that user's deadline alert on the
//!   newest context (Python `live._alert_user`).
//!
//! The result is one entry per step: null for the clock and FPL steps, `{"replies": [[chat,
//! text], ...]}` for a message, `{"alert": outcome, "texts": [[chat, piece], ...]}` for an
//! alert, `{"error": true}` where the step failed (the bot would apologise and alert the
//! admin).
//!
//! Recorded FPL responses (`FileFpl`): `bootstrap-static.json`, and per team
//! `entry/<id>/history.json`, `entry/<id>/transfers.json`, `entry/<id>/picks/<gw>.json`; a
//! team without a directory is unknown to FPL (404).

use std::cell::Cell;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use fplopt_optimize::entry::EntryClient;
use fplopt_optimize::io::params_with;
use serde_json::{Value, json};

use crate::context::ContextStore;
use crate::planner::{BotFpl, NotFound, chip_options, chip_params, live_params, plan_user};
use crate::service::{AlertOutcome, BotService, SendError};
use crate::store::{UserStore, database_path};
use crate::telegram::parse_command;
use crate::util::parse_isoformat;

/// FPL's public API from recorded responses (module docs).
pub struct FileFpl {
    pub dir: PathBuf,
    pub down: Rc<Cell<bool>>,
}

impl FileFpl {
    fn read(&self, path: PathBuf) -> Result<Vec<u8>> {
        match std::fs::read(&path) {
            Ok(body) => Ok(body),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(NotFound.into()),
            Err(e) => Err(e).with_context(|| path.display().to_string()),
        }
    }

    fn entry(&self, entry_id: i64) -> PathBuf {
        self.dir.join("entry").join(entry_id.to_string())
    }
}

impl EntryClient for FileFpl {
    fn entry_history(&self, entry_id: i64) -> Result<Vec<u8>> {
        self.read(self.entry(entry_id).join("history.json"))
    }

    fn entry_picks(&self, entry_id: i64, gw: i64) -> Result<Vec<u8>> {
        self.read(self.entry(entry_id).join("picks").join(format!("{gw}.json")))
    }

    fn entry_transfers(&self, entry_id: i64) -> Result<Vec<u8>> {
        self.read(self.entry(entry_id).join("transfers.json"))
    }
}

impl BotFpl for FileFpl {
    fn bootstrap_static(&self) -> Result<Vec<u8>> {
        if self.down.get() {
            bail!("FPL is down (script)");
        }
        let path = self.dir.join("bootstrap-static.json");
        std::fs::read(&path).with_context(|| path.display().to_string())
    }
}

/// Run `script` (module docs) on the state under `state_dir` (`bot.sqlite`, `context/`) and
/// the FPL responses under `fpl_dir`.
pub fn replay(script: &Value, state_dir: &Path, fpl_dir: &Path) -> Result<Value> {
    let down = Rc::new(Cell::new(false));
    let fpl = FileFpl { dir: fpl_dir.to_path_buf(), down: down.clone() };
    let admin = script["admin"].as_i64();
    let mut service = BotService::new(
        UserStore::new(database_path(state_dir))?,
        ContextStore::new(state_dir.join("context")),
        Box::new(fpl),
        admin,
    );
    let clock: Rc<Cell<DateTime<Utc>>> = Rc::new(Cell::new(Utc::now()));
    let now = clock.clone();
    service.now = Box::new(move || now.get());
    let null = Value::Null;
    let params = params_with(
        live_params(),
        script.get("params").filter(|v| !v.is_null()).unwrap_or(&json!({})),
    )?;
    service.planner = Box::new(move |fpl, entry, context, ft, made| {
        plan_user(fpl, entry, context, ft, &params, made)
    });
    let chips = params_with(
        chip_params(),
        script.get("chip_params").filter(|v| !v.is_null()).unwrap_or(&json!({})),
    )?;
    service.chip_planner = Box::new(move |fpl, entry, context, ft, made| {
        chip_options(fpl, entry, context, ft, &chips, made)
    });

    let mut results = Vec::new();
    for step in script["steps"].as_array().context("script: no steps")? {
        if let Some(t) = step.get("now") {
            clock.set(parse_isoformat(t.as_str().context("now: not a string")?)?);
            results.push(null.clone());
        } else if let Some(d) = step.get("fpl_down") {
            down.set(d.as_bool().context("fpl_down: not a bool")?);
            results.push(null.clone());
        } else if let Some(chat) = step.get("chat") {
            let chat = chat.as_i64().context("chat: not an integer")?;
            let text = step["text"].as_str().context("text: not a string")?;
            let outcome = match parse_command(Some(text)) {
                Some((command, args)) => service.run(&command, chat, &args),
                None => service.unknown(chat),
            };
            results.push(match outcome {
                Ok(replies) => json!({"replies": replies.iter().map(|r| json!([r.chat_id, r.text])).collect::<Vec<_>>()}),
                Err(err) => {
                    log::warn!("step failed: {err:#}");
                    json!({"error": true})
                }
            });
        } else if let Some(chat) = step.get("alert") {
            let chat = chat.as_i64().context("alert: not an integer")?;
            let kind = step["kind"].as_str().context("kind: not a string")?;
            results.push(alert(&service, chat, kind).unwrap_or_else(|err| {
                log::warn!("alert failed: {err:#}");
                json!({"error": true})
            }));
        } else {
            bail!("script: unknown step {step}");
        }
    }
    Ok(Value::Array(results))
}

fn alert(service: &BotService, chat: i64, kind: &str) -> Result<Value> {
    let user = service.users.user(chat)?.context("alert: no such user")?;
    let context = service.contexts.latest()?.context("alert: no context")?;
    let mut texts = Vec::new();
    let mut send = |chat_id: i64, text: &str| -> std::result::Result<(), SendError> {
        texts.push(json!([chat_id, text]));
        Ok(())
    };
    let outcome = service.alert_user(kind, &context, &user, &mut send)?;
    let outcome = match outcome {
        AlertOutcome::Sent => "sent",
        AlertOutcome::Silent => "silent",
        AlertOutcome::Told => "told",
        AlertOutcome::Undeliverable => "undeliverable",
    };
    Ok(json!({"alert": outcome, "texts": texts}))
}
