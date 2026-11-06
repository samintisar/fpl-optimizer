//! `research money-value` (Rust only; price sizing study, addendum of 2026-11-06): the
//! expected points per GW that £1m in the bank buys the live planner, from paired solves at
//! fixed states (`fplopt_backtest::money_value`). Writes money_value.parquet and
//! summary.json to --out. Usage errors exit 2.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Result;
use clap::{Args, Subcommand};
use fplopt_backtest::money_value::{
    LIVE_SPEC, POOLED_FROM, measure, rows_table, specs_at, summary,
};
use fplopt_backtest::simulator::Caches;
use fplopt_backtest::specs::parse_policy_spec;
use fplopt_backtest::xp_cache;
use fplopt_build::table::write_parquet;
use fplopt_core::pyjson::dumps_indent;
use fplopt_core::settings::Settings;
use fplopt_store::DataStore;

use crate::backtest::default_jobs;
use crate::optimize_plan::{exit_code, parse_seasons, usage};

#[derive(Subcommand)]
pub enum ResearchJob {
    /// Expected points per GW that £1m buys the live planner (two solves per state).
    MoneyValue(MoneyValueArgs),
}

#[derive(Args, Debug)]
pub struct MoneyValueArgs {
    /// Start years: 2016-2022, 2021,2023 or 2021-22 (the holdout is refused).
    #[arg(long)]
    pub seasons: String,
    /// gw_indexes of the states.
    #[arg(long, value_delimiter = ',', default_value = "1,8,15,22,29")]
    pub gws: Vec<i64>,
    /// Start kinds without @<gw> (template, random[:N], squad:FILE), built at each gw_index;
    /// a refused state is skipped, never moved to another GW.
    #[arg(long, default_value = "template,random:3")]
    pub starts: String,
    /// Tenths of £m added to the bank in the second solve.
    #[arg(long, default_value_t = 10)]
    pub extra_bank: i64,
    /// The planner as a backtest policy spec (default: the live planner).
    #[arg(long, default_value = LIVE_SPEC)]
    pub spec: String,
    /// Threads over states (default: CPUs - 1); results equal --jobs 1.
    #[arg(long, default_value_t = default_jobs())]
    pub jobs: usize,
    /// The data directory (default FPLOPT_DATA_DIR or data).
    #[arg(long)]
    pub data_dir: Option<PathBuf>,
    #[arg(long)]
    pub out: PathBuf,
    /// Where xP frames are cached across runs (keyed by data/ and this binary).
    #[arg(long, default_value = xp_cache::DEFAULT_ROOT)]
    pub xp_cache: PathBuf,
    /// Compute every xP frame afresh, without reading or writing the cache.
    #[arg(long)]
    pub no_xp_cache: bool,
}

pub fn run(job: &ResearchJob) -> Result<ExitCode> {
    Ok(exit_code(match job {
        ResearchJob::MoneyValue(args) => money_value(args),
    }))
}

fn money_value(args: &MoneyValueArgs) -> Result<()> {
    let seasons = parse_seasons(&args.seasons, "--seasons")?;
    if args.jobs < 1 {
        return usage("--jobs must be >= 1");
    }
    if args.extra_bank < 1 {
        return usage("--extra-bank must be >= 1 (tenths of £m)");
    }
    if args.gws.is_empty() || args.gws.iter().any(|g| *g < 1) {
        return usage("--gws: gw_indexes >= 1");
    }
    if let Err(e) = specs_at(&args.starts, 1) {
        return usage(format!("--starts: {e}"));
    }
    let spec = match parse_policy_spec(&args.spec) {
        Ok(s) => s,
        Err(e) => return usage(format!("argument --spec: {e}")),
    };
    if spec.name != "optimizer" {
        return usage(format!("--spec: an optimizer spec, got {spec}"));
    }
    let policy = spec.build()?;
    let data_dir = args.data_dir.clone().unwrap_or_else(|| Settings::from_env().data_dir);
    let store = match DataStore::open(&data_dir) {
        Ok(s) => s,
        Err(e) => return usage(format!("{e:#}")),
    };
    let caches = if args.no_xp_cache {
        Caches::default()
    } else {
        Caches::with_disk(xp_cache::DiskCache::open(&args.xp_cache, &data_dir)?)
    };
    let started = std::time::Instant::now();
    let rows = measure(
        &store,
        &seasons,
        &args.gws,
        &args.starts,
        args.extra_bank,
        &policy,
        &caches,
        args.jobs,
    )?;
    std::fs::create_dir_all(&args.out)?;
    write_parquet(&rows_table(&rows), &args.out.join("money_value.parquet"))?;
    let summary = summary(&rows, args.extra_bank, &spec.to_string());
    std::fs::write(args.out.join("summary.json"), dumps_indent(&summary, 2)? + "\n")?;

    println!(
        "money value: {} state(s), {spec}, +{} tenths, {:.0} s",
        rows.len(),
        args.extra_bank,
        started.elapsed().as_secs_f64()
    );
    let mut gws: Vec<i64> = rows.iter().map(|r| r.gw_index).collect();
    gws.sort_unstable();
    gws.dedup();
    let line = |label: String, v: Vec<f64>| {
        let n = v.len() as f64;
        let mean = v.iter().sum::<f64>() / n;
        let se = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt() / n.sqrt();
        let se = if v.len() < 2 { "-".to_owned() } else { format!("{se:.3}") };
        println!("  {label:<14} v {mean:>7.3} (SE {se}, n {})", v.len());
    };
    for g in gws {
        line(
            format!("gw_index {g}"),
            rows.iter().filter(|r| r.gw_index == g).map(|r| r.v).collect(),
        );
    }
    let pooled: Vec<f64> = rows.iter().filter(|r| r.gw_index >= POOLED_FROM).map(|r| r.v).collect();
    if !pooled.is_empty() {
        line(format!("gw_index >= {POOLED_FROM}"), pooled);
    }
    println!("  written to {}", args.out.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Cli {
        #[command(subcommand)]
        job: ResearchJob,
    }

    #[test]
    fn money_value_args() {
        let cli = Cli::try_parse_from([
            "x",
            "money-value",
            "--seasons",
            "2016-2022",
            "--gws",
            "1,8,15",
            "--extra-bank",
            "5",
            "--jobs",
            "3",
            "--data-dir",
            "d",
            "--out",
            "o",
        ])
        .unwrap();
        let ResearchJob::MoneyValue(a) = cli.job;
        assert_eq!(a.gws, [1, 8, 15]);
        assert_eq!((a.extra_bank, a.jobs), (5, 3));
        assert_eq!(a.starts, "template,random:3");
        assert_eq!(a.spec, LIVE_SPEC);
        assert_eq!(a.data_dir, Some(PathBuf::from("d")));
        let defaults =
            Cli::try_parse_from(["x", "money-value", "--seasons", "2021", "--out", "o"]).unwrap();
        let ResearchJob::MoneyValue(d) = defaults.job;
        assert_eq!((d.gws, d.extra_bank), (vec![1, 8, 15, 22, 29], 10));
        assert!(Cli::try_parse_from(["x", "money-value", "--seasons", "2021"]).is_err());
    }

    #[test]
    fn live_spec_is_the_live_planner() {
        let spec = parse_policy_spec(LIVE_SPEC).unwrap();
        assert_eq!((spec.name.as_str(), spec.xp.as_str()), ("optimizer", "v1"));
        let fplopt_backtest::policies::Policy::Optimizer { params, chips, .. } =
            spec.build().unwrap()
        else {
            panic!("not an optimizer policy")
        };
        assert!(!chips);
        // the bot's planner, minus its wall-clock cap (backtests stop at the node limit)
        let live = fplopt_optimize::OptimizerParams {
            time_limit: None,
            ..fplopt_bot::planner::live_params()
        };
        assert_eq!(params, live);
        assert!(params.bench_from_minutes && params.node_limit.is_some());
    }
}
