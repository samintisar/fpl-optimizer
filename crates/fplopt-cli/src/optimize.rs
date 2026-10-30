//! `fplopt-rs optimize ...` (migration batch 4): `plan` and `squad` as in the Python CLI,
//! with the xP frame read from a file (`--xp-file`; the models are ported in batch 6), and
//! `solve`, which runs one planner case for the parity gate (not in the Python CLI).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use fplopt_build::scoring::{backtest_rules, default_config_dir};
use fplopt_optimize::io::{planset_json, read_case};
use fplopt_optimize::{PlanInput, optimize};

/// Wall-clock cap per HiGHS solve for interactive commands (`optimize plan`) only.
pub const INTERACTIVE_TIME_LIMIT: f64 = 60.0;
/// The xP frames' horizon: `--horizon` is at most this.
pub const MAX_PLAN_HORIZON: i64 = 6;

#[derive(Subcommand)]
pub enum OptimizeJob {
    /// Top plans and the roll plan at a real deadline.
    Plan(PlanArgs),
    /// Import an FPL team as a squad file to check or edit.
    Squad(SquadArgs),
    /// Solve one planner case directory (case.json, pool.parquet, xp.parquet) and write the
    /// plans as JSON (the batch 4 parity gate; not in the Python CLI).
    Solve {
        #[arg(long)]
        case: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
}

#[derive(Args)]
pub struct SquadSource {
    /// An FPL team ID (the number in fantasy.premierleague.com/entry/<ID>/...): import its
    /// squad, bank, free transfers and chips from FPL's public API.
    #[arg(long)]
    pub entry: Option<i64>,
    /// A squad file (TOML, see `fplopt-rs optimize squad`); with --entry its values override
    /// the import's.
    #[arg(long)]
    pub squad_file: Option<PathBuf>,
}

#[derive(Args)]
pub struct PlanArgs {
    /// One season, e.g. 2024 or 2024-25.
    #[arg(long)]
    pub season: String,
    /// The GW to plan from (FPL GW number).
    #[arg(long)]
    pub gw: i64,
    /// template (most-owned squad; default) or random:SEED (a seeded random squad).
    #[arg(long)]
    pub start: Option<String>,
    #[command(flatten)]
    pub squad: SquadSource,
    /// Per-GW weight of later GWs' xP (0.85).
    #[arg(long, allow_negative_numbers = true)]
    pub decay: Option<f64>,
    /// The xP frame (Parquet: player_key, gw, gw_index, horizon, xp[, p_play, p_play_gw,
    /// e_minutes]) for this deadline, e.g. from the Python model.
    #[arg(long)]
    pub xp_file: PathBuf,
    /// The xP model's name, for the header only.
    #[arg(long, default_value = "file")]
    pub xp: String,
    /// Search the chip scenarios (default on).
    #[arg(long, overrides_with = "no_chips")]
    pub chips: bool,
    /// Solve the no-chip scenario only.
    #[arg(long, overrides_with = "chips")]
    pub no_chips: bool,
    /// Hits per GW: N or none (default: OptimizerParams').
    #[arg(long, allow_negative_numbers = true)]
    pub max_hits: Option<String>,
    /// Extra cost per hit.
    #[arg(long, allow_negative_numbers = true)]
    pub hit_margin: Option<f64>,
    /// Planning horizon in GWs.
    #[arg(long, default_value_t = 6)]
    pub horizon: i64,
    /// Plans to show besides roll.
    #[arg(long, default_value_t = 3)]
    pub top_k: usize,
}

#[derive(Args)]
pub struct SquadArgs {
    /// The season, e.g. 2026 or 2026-27.
    #[arg(long)]
    pub season: String,
    /// The GW to plan (FPL GW number).
    #[arg(long)]
    pub gw: i64,
    #[command(flatten)]
    pub squad: SquadSource,
    /// Squad file to write.
    #[arg(long, default_value = "squad.toml")]
    pub out: PathBuf,
}

/// `optimize solve`: one case through `PlanInput::from_context` and `optimize`, as the gate
/// runs Python's.
pub fn solve(case: &Path, out: &Path) -> Result<()> {
    let case = read_case(case)?;
    let rules = backtest_rules(case.season, &default_config_dir())?;
    let problem = PlanInput::from_context(&case.state, &case.pool, &case.xp, &rules, &case.params)?;
    let plans = optimize(&problem, &case.params, case.top_k, case.chips, true, case.search)?;
    let text = serde_json::to_string(&planset_json(&plans, &problem))?;
    std::fs::write(out, text).with_context(|| out.display().to_string())?;
    Ok(())
}
