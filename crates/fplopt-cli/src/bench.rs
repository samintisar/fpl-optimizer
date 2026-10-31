//! `optimize bench` (Python `cli._optimize_bench`): the optimizer benchmark on real deadlines
//! (`fplopt_backtest::bench`); prints the summary and writes cases.csv, prune.csv and
//! summary.json to --out. Usage errors exit 2.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use anyhow::Result;
use clap::Args;
use fplopt_backtest::bench::{
    BENCH_SEASONS, FLOORS, XP_MODELS, run_bench, summarize, summary_text, to_csv,
};
use fplopt_backtest::evaluate::record;
use fplopt_core::pyjson::Ordered;
use fplopt_core::settings::Settings;
use fplopt_optimize::OptimizerParams;
use fplopt_store::DataStore;

use crate::backtest::{command_line, write_json};
use crate::optimize_plan::{exit_code, parse_seasons, usage};

/// Wall-clock cap per HiGHS solve for interactive commands only.
const INTERACTIVE_TIME_LIMIT: f64 = 60.0;
const MODELS: [&str; 4] = ["rolling", "ep_next", "ep_next_fade", "v1"];

#[derive(Args)]
pub struct BenchArgs {
    /// Real deadlines.
    #[arg(long, default_value_t = 10)]
    pub deadlines: i64,
    /// Deadline/random-start seed.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// Planning horizon in GWs.
    #[arg(long, default_value_t = 6)]
    pub horizon: i64,
    /// Top-N pruning per position, GK,DEF,MID,FWD (default 20,60,60,30).
    #[arg(long)]
    pub prune_n: Option<String>,
    /// Also run the exhaustive chip scenario search on the first K cases.
    #[arg(long, default_value_t = 0)]
    pub all_chips: i64,
    /// Skip the pruning variants.
    #[arg(long)]
    pub no_prune_study: bool,
    /// Seasons to sample deadlines from (default 2021-2024,2026).
    #[arg(long)]
    pub seasons: Option<String>,
    /// xP models (default ep_next,rolling).
    #[arg(long)]
    pub xp: Option<String>,
    /// Expected-minutes floors over the horizon in the pruning study (default 0,30,60,90,135,180).
    #[arg(long)]
    pub floors: Option<String>,
    /// Output directory (default results/<UTC time>-bench).
    #[arg(long)]
    pub out: Option<PathBuf>,
}

pub fn run(args: &BenchArgs) -> Result<ExitCode> {
    Ok(exit_code(run_inner(args)))
}

fn run_inner(args: &BenchArgs) -> Result<()> {
    if args.deadlines < 1 || args.all_chips < 0 {
        return usage("--deadlines must be >= 1 and --all-chips >= 0");
    }
    if !(1..=6).contains(&args.horizon) {
        return usage("--horizon must be in 1..6 (the xP frames' GWs)");
    }
    let prune_n = match &args.prune_n {
        None => None,
        Some(text) => {
            let values: Vec<i64> = text.split(',').filter_map(|v| v.trim().parse().ok()).collect();
            if values.len() != 4 || values.iter().any(|v| *v < 0) || text.split(',').count() != 4 {
                return usage(format!(
                    "argument --prune-n: --prune-n: expected 4 counts >= 0, got '{text}'"
                ));
            }
            Some((1..=4).zip(values).collect::<BTreeMap<i64, i64>>())
        }
    };
    let xp: Vec<String> = match &args.xp {
        None => XP_MODELS.iter().map(|s| s.to_string()).collect(),
        Some(text) => {
            let m: Vec<String> = text.split(',').map(|s| s.trim().to_owned()).collect();
            let distinct = m.iter().collect::<std::collections::BTreeSet<_>>().len() == m.len();
            if m.is_empty() || !distinct || m.iter().any(|x| !MODELS.contains(&x.as_str())) {
                return usage(format!(
                    "argument --xp: --xp: expected distinct models from {}, got '{text}'",
                    MODELS.join(", ")
                ));
            }
            m
        }
    };
    let floors: Vec<f64> = match &args.floors {
        None => FLOORS.to_vec(),
        Some(text) => {
            let v: Option<Vec<f64>> = text.split(',').map(|x| x.trim().parse().ok()).collect();
            match v {
                Some(v) if !v.is_empty() && v.iter().all(|x| x.is_finite() && *x >= 0.0) => v,
                _ => {
                    return usage(format!(
                        "argument --floors: --floors: expected minutes >= 0, got '{text}'"
                    ));
                }
            }
        }
    };
    let seasons = match &args.seasons {
        None => BENCH_SEASONS.to_vec(),
        Some(text) => parse_seasons(text, "--seasons")?,
    };
    let early: Vec<i64> = seasons.iter().copied().filter(|s| *s < 2021).collect();
    for m in &xp {
        if m.starts_with("ep_next") && !early.is_empty() {
            let labels: Vec<String> =
                early.iter().map(|s| fplopt_core::seasons::season_label(*s as i32)).collect();
            return usage(format!(
                "--xp {m} uses ep_next, which exists only from 2021-22 (FPL snapshots); drop {}",
                labels.join(", ")
            ));
        }
    }
    let params = OptimizerParams {
        horizon: args.horizon,
        time_limit: Some(INTERACTIVE_TIME_LIMIT),
        prune_n: prune_n.or(OptimizerParams::default().prune_n),
        ..OptimizerParams::default()
    };
    let store = DataStore::open(Settings::from_env().data_dir)?;
    let out = args.out.clone().unwrap_or_else(|| {
        PathBuf::from("results")
            .join(format!("{}-bench", chrono::Utc::now().format("%Y%m%dT%H%M%SZ")))
    });
    let began = Instant::now();
    let now = chrono::Utc::now().timestamp_micros();
    let result = run_bench(
        &store,
        args.deadlines as usize,
        args.seed,
        &params,
        args.all_chips as usize,
        !args.no_prune_study,
        &seasons,
        &xp,
        &floors,
        now,
    )?;
    let summary = summarize(&result);
    let runtime = began.elapsed().as_secs_f64();
    std::fs::create_dir_all(&out)?;
    std::fs::write(out.join("cases.csv"), to_csv(&result.cases))?;
    std::fs::write(out.join("prune.csv"), to_csv(&result.prune))?;
    let payload = record(vec![
        ("command", Ordered::Str(command_line())),
        ("config", result.config.clone()),
        ("runtime_seconds", Ordered::Float((runtime * 10.0).round() / 10.0)),
        ("summary", summary.clone()),
    ]);
    write_json(&out.join("summary.json"), &payload)?;
    println!("\n{}\n\n({runtime:.0}s) Written to {}", summary_text(&summary), out.display());
    Ok(())
}
