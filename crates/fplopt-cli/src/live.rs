//! `fplopt-rs live tick` (the 15-minute alert job, `fplopt_bot::live`) and `live prepare`
//! (rebuild data/ and the plan context now): Python's `_live`.

use std::path::Path;

use anyhow::{Context, Result, bail};
use clap::ValueEnum;
use fplopt_archive::SystemClock;
use fplopt_archive::adapters::FplSource;
use fplopt_archive::raw_store::RawStore;
use fplopt_bot::builder::{LIVE_XP, git_sha};
use fplopt_bot::live::{Refresh, live_tick, refresh_context};
use fplopt_bot::telegram::Telegram;
use fplopt_core::seasons::season_label;
use fplopt_core::settings::Settings;

use crate::bot::bot_service;

#[derive(Clone, Copy, ValueEnum)]
pub enum LiveJob {
    /// The 15-minute alert job.
    Tick,
    /// Rebuild data/ and the plan context for the next deadline now.
    Prepare,
}

/// `fplopt-rs build all` in its own process (Python runs `fplopt build all` in a subprocess,
/// so its memory is freed before the model runs).
fn build_subprocess(settings: &Settings) -> impl Fn() -> Result<()> + '_ {
    move || {
        let exe = std::env::current_exe().context("the fplopt-rs executable")?;
        let status = std::process::Command::new(exe)
            .args(["build", "all"])
            .env("FPLOPT_RAW_DIR", &settings.raw_dir)
            .env("FPLOPT_DATA_DIR", &settings.data_dir)
            .status()
            .context("fplopt-rs build all")?;
        if !status.success() {
            bail!("fplopt-rs build all failed ({status})");
        }
        Ok(())
    }
}

pub fn run(job: LiveJob, store: &RawStore, fpl: &dyn FplSource, settings: &Settings) -> Result<()> {
    let service = bot_service(settings)?;
    let build = build_subprocess(settings);
    let refresh = Refresh {
        raw: store,
        data_dir: &settings.data_dir,
        build: &build,
        sha: git_sha(Path::new(".")),
        xp_model: LIVE_XP,
    };
    match job {
        LiveJob::Prepare => {
            let (context, _) = refresh_context(&refresh, &service.contexts, &SystemClock, true)?;
            match context {
                None => println!("no deadline ahead in the data: no plan context"),
                Some(c) => println!(
                    "plan context {}: {} GW{}, deadline {}, {} players",
                    c.context_id,
                    season_label(c.season as i32),
                    c.gw,
                    fplopt_bot::util::py_isoformat(c.deadline),
                    c.pool.len()
                ),
            }
            Ok(())
        }
        LiveJob::Tick => {
            let Some(token) = settings.telegram_bot_token.as_deref().filter(|t| !t.is_empty())
            else {
                bail!("TELEGRAM_BOT_TOKEN is not set: alerts can't be sent");
            };
            let telegram = Telegram::new(token)?;
            let mut send = |chat: i64, text: &str| telegram.send_html(chat, text);
            let result = live_tick(
                &refresh,
                &service.users,
                &service.contexts,
                &service,
                fpl,
                &mut send,
                &SystemClock,
            )?;
            log::info!(
                "live tick: due {:?}, context {:?} (rebuilt {}), sent {:?}, silent {:?}",
                result.due,
                result.context_id,
                result.rebuilt,
                result.sent,
                result.silent
            );
            Ok(())
        }
    }
}
