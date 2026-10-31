//! `fplopt-rs bot run` (the Telegram bot, `fplopt_bot`) and `bot replay` (a scripted session
//! on recorded FPL responses: the batch 5 gate's entry point, not in the Python CLI).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Subcommand;
use fplopt_archive::adapters::fpl::{BASE_URL, FplClient};
use fplopt_archive::adapters::{FplSource, base_url};
use fplopt_archive::fetch::{FetchError, Http};
use fplopt_bot::context::ContextStore;
use fplopt_bot::planner::{BotFpl, NotFound};
use fplopt_bot::service::BotService;
use fplopt_bot::store::{UserStore, database_path};
use fplopt_bot::telegram::{AdminAlerts, Telegram, run_polling};
use fplopt_core::settings::Settings;
use fplopt_optimize::entry::EntryClient;

#[derive(Subcommand)]
pub enum BotJob {
    /// Long polling until stopped.
    Run,
    /// Run a scripted session (`fplopt_bot::replay`) and write the step results as JSON.
    Replay {
        #[arg(long)]
        script: PathBuf,
        /// The state directory (bot.sqlite, context/).
        #[arg(long)]
        state: PathBuf,
        /// Recorded FPL responses.
        #[arg(long)]
        fpl: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
}

/// FPL's public API for the bot: a 404 becomes `NotFound` (no such team).
pub struct FplBot {
    http: Http,
    base: String,
}

impl FplBot {
    pub fn new() -> Result<FplBot> {
        Ok(FplBot { http: Http::new()?, base: base_url("FPL", BASE_URL) })
    }

    fn client(&self) -> FplClient<'_> {
        FplClient::new(&self.http, &self.base)
    }
}

fn not_found(result: Result<Vec<u8>>) -> Result<Vec<u8>> {
    result.map_err(|err| {
        if err.downcast_ref::<FetchError>().and_then(FetchError::status) == Some(404) {
            anyhow::Error::new(NotFound)
        } else {
            err
        }
    })
}

impl EntryClient for FplBot {
    fn entry_history(&self, entry_id: i64) -> Result<Vec<u8>> {
        not_found(self.client().entry_history(entry_id))
    }

    fn entry_picks(&self, entry_id: i64, gw: i64) -> Result<Vec<u8>> {
        not_found(self.client().entry_picks(entry_id, gw))
    }

    fn entry_transfers(&self, entry_id: i64) -> Result<Vec<u8>> {
        not_found(self.client().entry_transfers(entry_id))
    }
}

impl BotFpl for FplBot {
    fn bootstrap_static(&self) -> Result<Vec<u8>> {
        self.client().bootstrap_static()
    }
}

/// The bot's command logic on the state directory (Python `_bot_service`).
pub fn bot_service(settings: &Settings) -> Result<BotService> {
    let admin = match settings.telegram_admin_chat_id.as_deref().filter(|a| !a.is_empty()) {
        Some(a) => Some(a.trim().parse::<i64>().context("TELEGRAM_ADMIN_CHAT_ID: not an integer")?),
        None => None,
    };
    Ok(BotService::new(
        UserStore::new(database_path(&settings.state_dir))?,
        ContextStore::new(settings.state_dir.join("context")),
        Box::new(FplBot::new()?),
        admin,
    ))
}

/// `bot run` (Python `_bot_run`): Ok when stopped by SIGTERM.
pub fn run(settings: &Settings) -> Result<()> {
    let Some(token) = settings.telegram_bot_token.as_deref().filter(|t| !t.is_empty()) else {
        bail!("TELEGRAM_BOT_TOKEN is not set");
    };
    let service = bot_service(settings)?;
    let bot = Telegram::new(token)?;
    let admin = AdminAlerts {
        token: Some(token.to_string()),
        chat_id: settings.telegram_admin_chat_id.clone(),
    };
    run_polling(&service, &bot, &admin, &fplopt_archive::stop::requested)
}

/// `bot replay`.
pub fn replay(script: &Path, state: &Path, fpl: &Path, out: &Path) -> Result<()> {
    let text = std::fs::read_to_string(script).with_context(|| script.display().to_string())?;
    let results = fplopt_bot::replay::replay(&serde_json::from_str(&text)?, state, fpl)?;
    std::fs::write(out, serde_json::to_string_pretty(&results)?)
        .with_context(|| out.display().to_string())
}
