//! `models dump`: xP frames at given deadlines (the batch 6 parity gate; not in the Python
//! CLI, which evaluates models with `models eval`, batch 7).

use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::Subcommand;
use fplopt_features::leakcheck::fmt_micros;

#[derive(Subcommand)]
pub enum ModelsJob {
    /// Build the plan context for the next deadline after --now from data/ (no rebuild) and
    /// write it under <out>/ as `live prepare` would (the batch 6 gate's entry point).
    Context {
        /// An ISO 8601 instant.
        #[arg(long)]
        now: String,
        #[arg(long)]
        out: PathBuf,
    },
    /// Write each model's xP frame at each deadline to <out>/<NN>/<model>.parquet (NN: the
    /// deadline's line number in the file, from 00); a component (`fplopt_models::COMPONENTS`)
    /// writes its frame and <model>.json, its fitted values at that deadline.
    Dump {
        /// Comma-separated model or component names (`fplopt_models::MODELS`, `COMPONENTS`).
        #[arg(long, default_value = "rolling,ep_next,ep_next_fade")]
        models: String,
        /// A file with one ISO 8601 deadline per line.
        #[arg(long)]
        deadlines: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
}

pub fn dump(data_dir: &Path, models: &str, deadlines: &Path, out: &Path) -> Result<()> {
    let text = std::fs::read_to_string(deadlines)?;
    let deadlines: Vec<i64> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(crate::parse_deadline)
        .collect::<Result<_>>()?;
    let models: Vec<&str> = models.split(',').map(str::trim).filter(|m| !m.is_empty()).collect();
    let store = fplopt_store::DataStore::open(data_dir)?;
    for (i, &deadline) in deadlines.iter().enumerate() {
        let dir = out.join(format!("{i:02}"));
        std::fs::create_dir_all(&dir)?;
        let view = store.as_of(deadline);
        for &name in &models {
            let started = std::time::Instant::now();
            let context =
                |e: anyhow::Error| e.context(format!("{name} at {}", fmt_micros(deadline)));
            let frame = if fplopt_models::COMPONENTS.contains(&name) {
                let (frame, fit) = fplopt_models::component(name, &view).map_err(context)?;
                std::fs::write(dir.join(format!("{name}.json")), serde_json::to_string(&fit)?)?;
                frame
            } else {
                fplopt_models::predict(name, &view).map_err(context)?
            };
            fplopt_build::table::write_parquet(&frame, &dir.join(format!("{name}.parquet")))?;
            log::info!(
                "{name} at {}: {} rows, {:.2} s",
                fmt_micros(deadline),
                frame.height(),
                started.elapsed().as_secs_f64()
            );
        }
    }
    Ok(())
}

pub fn context(data_dir: &Path, now: &str, out: &Path) -> Result<()> {
    let now = chrono::DateTime::from_timestamp_micros(crate::parse_deadline(now)?)
        .ok_or_else(|| anyhow::anyhow!("--now out of range"))?;
    let store = fplopt_store::DataStore::open(data_dir)?;
    let built = fplopt_bot::builder::build_context(
        &store,
        now,
        "",
        None,
        None,
        fplopt_bot::builder::LIVE_XP,
    )?;
    match built {
        None => log::info!("no deadline ahead in the data: no context"),
        Some(context) => {
            let dir = fplopt_bot::context::ContextStore::new(out).write(&context)?;
            log::info!("plan context {}: written to {}", context.context_id, dir.display());
        }
    }
    Ok(())
}
