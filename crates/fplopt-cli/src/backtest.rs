//! `backtest run` and `backtest compare` (Python `cli._backtest_run`, `_backtest_compare`,
//! `validate_backtest` and the tables): one policy, or a paired comparison of two, over
//! seasons × start states. Writes gws.parquet (and paired.parquet, per_decision.parquet)
//! and summary.json to --out, prints the tables and appends the run to the experiment log.
//! Usage errors exit 2.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use anyhow::Result;
use clap::{Args, Subcommand};
use fplopt_backtest::evaluate::{
    Bootstrap, GridRow, PerDecision, StartSpec, Summary, num, paired_full_run, paired_rows,
    parse_start_specs, per_decision, record, run_grid, run_totals, splits, summarize,
};
use fplopt_backtest::simulator::Caches;
use fplopt_backtest::specs::{PolicySpec, make_policy_spec, parse_policy_spec};
use fplopt_backtest::stats::{Diff, FULL_RUN_WINDOW, full_run_window_test, season_points};
use fplopt_build::table::{Column, Table, write_parquet};
use fplopt_core::experiments::{family_variants, log_experiment};
use fplopt_core::pyjson::{Ordered, dumps_indent};
use fplopt_core::seasons::season_label;
use fplopt_core::settings::Settings;
use fplopt_optimize::start::FAR_FUTURE;
use fplopt_store::DataStore;

use crate::optimize_plan::{exit_code, parse_seasons, usage};

const DEFAULT_STARTS: &str = "template@1,random:5@1,random:3@20";
const DEFAULT_EXPERIMENTS: &str = "results/experiments.csv";
const CI_LEVEL: f64 = 0.8;
const EP_NEXT_FIRST_SEASON: i64 = 2021;
const EP_NEXT_MODELS: [&str; 2] = ["ep_next", "ep_next_fade"];

fn default_jobs() -> usize {
    std::thread::available_parallelism().map_or(2, |n| n.get()).saturating_sub(1).max(1)
}

#[derive(Subcommand)]
pub enum BacktestJob {
    /// One policy over seasons and start states.
    Run(RunArgs),
    /// Paired comparison of two policies.
    Compare(CompareArgs),
}

#[derive(Args, Clone)]
pub struct Common {
    /// Start years: 2016-2024, 2021,2023, 2021-22 or a mix (the holdout is refused).
    #[arg(long)]
    pub seasons: String,
    /// Start states.
    #[arg(long, default_value = DEFAULT_STARTS)]
    pub starts: String,
    /// Output directory (default results/<UTC time>-<command>).
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Experiment log to append to.
    #[arg(long, default_value = DEFAULT_EXPERIMENTS)]
    pub experiments: PathBuf,
    /// Threads over (season, start) units (default: CPUs - 1); results equal --jobs 1.
    #[arg(long, default_value_t = default_jobs())]
    pub jobs: usize,
    /// Experiment-log comparison family (default: compare = the B policy and seasons; run =
    /// 'run <policy> <seasons>').
    #[arg(long)]
    pub family: Option<String>,
}

#[derive(Args)]
pub struct RunArgs {
    #[command(flatten)]
    pub common: Common,
    /// Policy (default greedy).
    #[arg(long)]
    pub policy: Option<String>,
    /// xP model (default rolling).
    #[arg(long)]
    pub xp: Option<String>,
    /// The policy as a spec instead of --policy/--xp/flags.
    #[arg(long)]
    pub spec: Option<String>,
    #[arg(long)]
    pub threshold: Option<String>,
    #[arg(long)]
    pub horizon: Option<String>,
    #[arg(long)]
    pub decay: Option<String>,
    #[arg(long)]
    pub max_transfers: Option<String>,
}

#[derive(Args)]
pub struct CompareArgs {
    #[command(flatten)]
    pub common: Common,
    #[arg(long)]
    pub a: String,
    #[arg(long)]
    pub b: String,
    /// Per-decision continuation: a policy spec both arms follow, or 'own'.
    #[arg(long, default_value = "roll:rolling")]
    pub continuation: String,
    /// Per-decision: whose own run gives the decision states.
    #[arg(long, default_value = "b", value_parser = ["a", "b"])]
    pub reference: String,
    #[arg(long, default_value_t = 4)]
    pub k: usize,
    #[arg(long)]
    pub stride: Option<usize>,
    /// Skip the per-decision evaluation.
    #[arg(long)]
    pub no_per_decision: bool,
    #[arg(long, default_value_t = 4)]
    pub block_length: usize,
    #[arg(long, default_value_t = 2000)]
    pub n_boot: usize,
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
}

pub fn run(job: &BacktestJob) -> Result<ExitCode> {
    Ok(exit_code(match job {
        BacktestJob::Run(args) => backtest_run(args),
        BacktestJob::Compare(args) => backtest_compare(args),
    }))
}

// --- checks -----------------------------------------------------------------------------------

/// `shlex.join(["fplopt", *argv])` of this invocation (as fplopt-rs).
pub(crate) fn command_line() -> String {
    let quote = |a: &str| {
        if !a.is_empty() && a.chars().all(|c| c.is_ascii_alphanumeric() || "@%+=:,./-_".contains(c))
        {
            a.to_owned()
        } else {
            format!("'{}'", a.replace('\'', "'\"'\"'"))
        }
    };
    let mut parts = vec!["fplopt-rs".to_owned()];
    parts.extend(std::env::args().skip(1).map(|a| quote(&a)));
    parts.join(" ")
}

/// (2016, 2017, 2018, 2021) -> '2016-2018,2021'.
pub(crate) fn format_seasons(seasons: &[i64]) -> String {
    let mut runs: Vec<Vec<i64>> = Vec::new();
    for &s in seasons {
        match runs.last_mut() {
            Some(run) if *run.last().unwrap() + 1 == s => run.push(s),
            _ => runs.push(vec![s]),
        }
    }
    runs.iter()
        .map(
            |r| {
                if r.len() == 1 { r[0].to_string() } else { format!("{}-{}", r[0], r[r.len() - 1]) }
            },
        )
        .collect::<Vec<_>>()
        .join(",")
}

fn spec_or_usage(text: &str, flag: &str) -> Result<PolicySpec> {
    parse_policy_spec(text).or_else(|e| usage(format!("argument {flag}: {e}")))
}

pub(crate) fn seasons_with_data(store: &DataStore) -> Result<BTreeSet<i64>> {
    let view = store.as_of(FAR_FUTURE);
    let gws: BTreeSet<i64> =
        view.table("gameweek", Some(&["season"]))?.i64("season")?.iter().copied().collect();
    let matches: BTreeSet<i64> =
        view.table("player_match", Some(&["season"]))?.i64("season")?.iter().copied().collect();
    Ok(gws.intersection(&matches).copied().collect())
}

struct Checked {
    seasons: Vec<i64>,
    specs: Vec<StartSpec>,
    store: DataStore,
    family: String,
}

fn check_common(
    common: &Common,
    policies: &[&PolicySpec],
    default_family: impl Fn(&str) -> String,
) -> Result<Checked> {
    let seasons = parse_seasons(&common.seasons, "--seasons")?;
    if common.jobs < 1 {
        return usage("--jobs must be >= 1");
    }
    let specs = match parse_start_specs(&common.starts) {
        Ok(s) => s,
        Err(e) => return usage(format!("--starts: {e}")),
    };
    let family = common.family.clone().unwrap_or_else(|| default_family(&format_seasons(&seasons)));
    let early: Vec<String> = seasons
        .iter()
        .filter(|s| **s < EP_NEXT_FIRST_SEASON)
        .map(|s| season_label(*s as i32))
        .collect();
    for spec in policies {
        if EP_NEXT_MODELS.contains(&spec.xp.as_str()) && !early.is_empty() {
            return usage(format!(
                "{spec} uses ep_next, which exists only from {} (FPL snapshots); drop {}",
                season_label(EP_NEXT_FIRST_SEASON as i32),
                early.join(", ")
            ));
        }
    }
    let settings = Settings::from_env();
    let store = match DataStore::open(&settings.data_dir) {
        Ok(s) => s,
        Err(e) => return usage(format!("{e:#}")),
    };
    let available = seasons_with_data(&store)?;
    let missing: Vec<String> = seasons
        .iter()
        .filter(|s| !available.contains(s))
        .map(|s| season_label(*s as i32))
        .collect();
    if !missing.is_empty() {
        return usage(format!(
            "no gameweek/player_match data for {} (run fplopt build all?)",
            missing.join(", ")
        ));
    }
    Ok(Checked { seasons, specs, store, family })
}

fn out_dir(out: &Option<PathBuf>, command: &str) -> PathBuf {
    out.clone().unwrap_or_else(|| {
        PathBuf::from("results")
            .join(format!("{}-{command}", chrono::Utc::now().format("%Y%m%dT%H%M%SZ")))
    })
}

fn run_spec(args: &RunArgs) -> Result<PolicySpec> {
    let flags = [
        ("threshold", &args.threshold),
        ("horizon", &args.horizon),
        ("decay", &args.decay),
        ("max_transfers", &args.max_transfers),
    ];
    if let Some(spec) = &args.spec {
        if flags.iter().any(|(_, v)| v.is_some()) || args.policy.is_some() || args.xp.is_some() {
            return usage("--spec replaces --policy, --xp and their flags");
        }
        return spec_or_usage(spec, "--spec");
    }
    let policy = args.policy.clone().unwrap_or_else(|| "greedy".to_owned());
    let xp = args.xp.clone().unwrap_or_else(|| "rolling".to_owned());
    let given: Vec<(String, String)> =
        flags.iter().filter_map(|(k, v)| v.as_ref().map(|v| (k.to_string(), v.clone()))).collect();
    if !given.is_empty() && policy != "greedy" {
        let names: Vec<String> =
            given.iter().map(|(k, _)| format!("--{}", k.replace('_', "-"))).collect();
        return usage(format!("{}: only for --policy greedy", names.join(", ")));
    }
    make_policy_spec(&policy, &xp, &given).or_else(|e| usage(e.to_string()))
}

// --- outputs ----------------------------------------------------------------------------------

fn grid_table(results: &[GridRow]) -> Table {
    let r = |f: &dyn Fn(&GridRow) -> i64| Column::I64(results.iter().map(f).collect());
    let oi = |f: &dyn Fn(&GridRow) -> Option<i64>| Column::OptI64(results.iter().map(f).collect());
    let of = |f: &dyn Fn(&GridRow) -> Option<f64>| Column::OptF64(results.iter().map(f).collect());
    let os =
        |f: &dyn Fn(&GridRow) -> Option<String>| Column::OptStr(results.iter().map(f).collect());
    let s = |f: &dyn Fn(&GridRow) -> String| Column::Str(results.iter().map(f).collect());
    let mut t = Table::new()
        .with("season", r(&|g| g.row.season))
        .with("start_id", s(&|g| g.start_id.clone()))
        .with("policy", s(&|g| g.policy.clone()))
        .with("gw", r(&|g| g.row.gw))
        .with("gw_index", r(&|g| g.row.gw_index))
        .with("deadline", Column::Ts(results.iter().map(|g| g.row.deadline).collect()))
        .with("chip", os(&|g| g.row.chip.clone()))
        .with("n_transfers", r(&|g| g.row.n_transfers))
        .with("hits", r(&|g| g.row.hits))
        .with("hit_points", r(&|g| g.row.hit_points))
        .with("points", r(&|g| g.row.points))
        .with("net_points", r(&|g| g.row.net_points))
        .with("bench_points", r(&|g| g.row.bench_points))
        .with("xg_points", of(&|g| g.row.xg_points))
        .with("xg_net_points", of(&|g| g.row.xg_net_points))
        .with("captain", r(&|g| g.row.captain))
        .with("vice", r(&|g| g.row.vice))
        .with("captain_used", oi(&|g| g.row.captain_used))
        .with("captain_points", r(&|g| g.row.captain_points))
        .with("captain_regret", r(&|g| g.row.captain_regret))
        .with("xi_regret", oi(&|g| g.row.xi_regret))
        .with("bank", r(&|g| g.row.bank))
        .with("free_transfers", r(&|g| g.row.free_transfers))
        .with("squad_value", r(&|g| g.row.squad_value))
        .with("transfers", s(&|g| g.row.transfers.clone()))
        .with("autosubs", s(&|g| g.row.autosubs.clone()))
        .with("pred_gain", of(&|g| g.row.pred_gain))
        .with("real_gain", of(&|g| g.row.real_gain))
        .with("solver_status", os(&|g| g.row.solver_status.clone()))
        .with("mip_gap", of(&|g| g.row.mip_gap));
    for (i, name) in fplopt_backtest::policies::CHIP_DIAGNOSTICS.iter().enumerate() {
        t.push(name, of(&|g| g.row.diagnostics[i]));
    }
    t
}

fn paired_table(paired: &[fplopt_backtest::evaluate::Paired]) -> Table {
    Table::new()
        .with("season", Column::I64(paired.iter().map(|p| p.season).collect()))
        .with("start_id", Column::Str(paired.iter().map(|p| p.start_id.clone()).collect()))
        .with("gw", Column::I64(paired.iter().map(|p| p.gw).collect()))
        .with("gw_index", Column::I64(paired.iter().map(|p| p.gw_index).collect()))
        .with("points_a", Column::I64(paired.iter().map(|p| p.points_a).collect()))
        .with("points_b", Column::I64(paired.iter().map(|p| p.points_b).collect()))
        .with("diff", Column::I64(paired.iter().map(|p| p.diff).collect()))
        .with("xg_a", Column::OptF64(paired.iter().map(|p| p.xg_a).collect()))
        .with("xg_b", Column::OptF64(paired.iter().map(|p| p.xg_b).collect()))
        .with("diff_xg", Column::OptF64(paired.iter().map(|p| p.diff_xg).collect()))
}

fn decisions_table(rows: &[PerDecision]) -> Table {
    let i = |f: &dyn Fn(&PerDecision) -> i64| Column::I64(rows.iter().map(f).collect());
    let s = |f: &dyn Fn(&PerDecision) -> String| Column::Str(rows.iter().map(f).collect());
    let of = |f: &dyn Fn(&PerDecision) -> Option<f64>| Column::OptF64(rows.iter().map(f).collect());
    Table::new()
        .with("season", i(&|r| r.season))
        .with("start_id", s(&|r| r.start_id.clone()))
        .with("policy_a", s(&|r| r.policy_a.clone()))
        .with("policy_b", s(&|r| r.policy_b.clone()))
        .with("reference", s(&|r| r.reference.clone()))
        .with("continuation", s(&|r| r.continuation.clone()))
        .with("gw", i(&|r| r.gw))
        .with("gw_index", i(&|r| r.gw_index))
        .with("k", i(&|r| r.k))
        .with("stride", i(&|r| r.stride))
        .with("same_decision", Column::Bool(rows.iter().map(|r| r.same_decision).collect()))
        .with("transfers_a", i(&|r| r.transfers_a))
        .with("transfers_b", i(&|r| r.transfers_b))
        .with("points_a", i(&|r| r.points_a))
        .with("points_b", i(&|r| r.points_b))
        .with("diff", i(&|r| r.diff))
        .with("xg_a", of(&|r| r.xg_a))
        .with("xg_b", of(&|r| r.xg_b))
        .with("diff_xg", of(&|r| r.diff_xg))
}

pub(crate) fn write_json(path: &Path, value: &Ordered) -> Result<()> {
    std::fs::write(path, dumps_indent(value, 2)? + "\n")?;
    Ok(())
}

fn text(s: &str) -> Ordered {
    Ordered::Str(s.to_owned())
}

fn int(i: i64) -> Ordered {
    Ordered::Int(i128::from(i))
}

fn starts_by_season(results: &[GridRow]) -> Ordered {
    let mut by: BTreeMap<i64, BTreeSet<String>> = BTreeMap::new();
    for r in results {
        by.entry(r.row.season).or_default().insert(r.start_id.clone());
    }
    Ordered::Object(
        by.into_iter()
            .map(|(s, ids)| {
                (
                    season_label(s as i32),
                    Ordered::Array(ids.into_iter().map(Ordered::Str).collect()),
                )
            })
            .collect(),
    )
}

// --- tables -----------------------------------------------------------------------------------

pub(crate) fn num_text(value: f64, digits: usize, sign: bool) -> String {
    if value.is_nan() {
        return "-".to_owned();
    }
    if sign { format!("{value:+.digits$}") } else { format!("{value:.digits$}") }
}

pub(crate) fn text_table(headers: &[String], rows: &[Vec<String>], left: usize) -> String {
    let widths: Vec<usize> = (0..headers.len())
        .map(|c| {
            rows.iter()
                .map(|r| r[c].chars().count())
                .chain([headers[c].chars().count()])
                .max()
                .unwrap()
        })
        .collect();
    let line = |cells: &[String]| {
        let out: Vec<String> = cells
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let pad = widths[i] - c.chars().count();
                if i < left {
                    format!("{c}{}", " ".repeat(pad))
                } else {
                    format!("{}{c}", " ".repeat(pad))
                }
            })
            .collect();
        out.join("  ").trim_end().to_owned()
    };
    let mut lines = vec![line(headers)];
    lines.extend(rows.iter().map(|r| line(r)));
    lines.join("\n")
}

fn h(names: &[&str]) -> Vec<String> {
    names.iter().map(|s| s.to_string()).collect()
}

const TOTALS_TITLE: &str = "Season totals by start GW (net points; the template start, mean/min/max over starts; \
a GW1 start refused there runs from GW2)";
const GAINS_TITLE: &str = "Predicted vs realized transfer gain (optimizer's curse, PLAN §7): GWs after GW1 with \
transfers; gain of the players bought over those sold over that GW and the next 3, gross of hits (net = minus \
the GW's hit points); slope and intercept of realized on predicted (least squares; slope < 1: predictions \
inflated)";

/// One row of `season_totals_by_start`.
struct ByStart {
    season: i64,
    policy: String,
    start_gw: i64,
    starts: usize,
    n_gws: f64,
    template: f64,
    mean: f64,
    min: f64,
    max: f64,
    xg_mean: f64,
    xg_null: f64,
    n_transfers: f64,
    hit_points: f64,
    captain_regret: f64,
    xi_regret: f64,
}

fn start_gw_of(specs: &[StartSpec]) -> BTreeMap<String, i64> {
    let mut out = BTreeMap::new();
    for spec in specs {
        out.insert(spec.start_id(None), spec.gw_index);
        if spec.gw_index == 1 {
            out.entry(spec.start_id(Some(2))).or_insert(1);
        }
    }
    out
}

fn mean_of(v: &[f64]) -> f64 {
    let kept: Vec<f64> = v.iter().copied().filter(|x| !x.is_nan()).collect();
    if kept.is_empty() { f64::NAN } else { kept.iter().sum::<f64>() / kept.len() as f64 }
}

fn season_totals_by_start(results: &[GridRow], specs: &[StartSpec]) -> Vec<ByStart> {
    let gw_of = start_gw_of(specs);
    let mut groups: BTreeMap<(i64, String, i64), Vec<fplopt_backtest::evaluate::RunTotal>> =
        BTreeMap::new();
    for run in run_totals(results) {
        let gw = gw_of.get(&run.start_id).copied().unwrap_or(0);
        groups.entry((run.season, run.policy.clone(), gw)).or_default().push(run);
    }
    groups
        .into_iter()
        .map(|((season, policy, start_gw), runs)| {
            let col = |f: fn(&fplopt_backtest::evaluate::RunTotal) -> f64| {
                runs.iter().map(f).collect::<Vec<f64>>()
            };
            let totals = col(|r| r.total);
            let template: Vec<f64> = runs
                .iter()
                .filter(|r| r.start_id.starts_with("template"))
                .map(|r| r.total)
                .collect();
            let xg = col(|r| r.xg_total);
            ByStart {
                season,
                policy,
                start_gw,
                starts: runs.len(),
                n_gws: mean_of(&col(|r| r.n_gws)),
                template: template.iter().copied().fold(f64::NAN, f64::max),
                mean: mean_of(&totals),
                min: totals.iter().copied().fold(f64::INFINITY, f64::min),
                max: totals.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                xg_mean: if xg.iter().any(|x| x.is_nan()) { f64::NAN } else { mean_of(&xg) },
                xg_null: mean_of(&col(|r| r.xg_null)),
                n_transfers: mean_of(&col(|r| r.n_transfers)),
                hit_points: mean_of(&col(|r| r.hit_points)),
                captain_regret: mean_of(&col(|r| r.captain_regret)),
                xi_regret: mean_of(&col(|r| r.xi_regret)),
            }
        })
        .collect()
}

fn by_start_json(rows: &[ByStart]) -> Ordered {
    Ordered::Array(
        rows.iter()
            .map(|r| {
                record(vec![
                    ("season", int(r.season)),
                    ("policy", text(&r.policy)),
                    ("start_gw", int(r.start_gw)),
                    ("starts", int(r.starts as i64)),
                    ("n_gws", num(r.n_gws)),
                    ("template", num(r.template)),
                    ("mean", num(r.mean)),
                    ("min", num(r.min)),
                    ("max", num(r.max)),
                    ("xg_mean", num(r.xg_mean)),
                    ("xg_null", num(r.xg_null)),
                    ("n_transfers", num(r.n_transfers)),
                    ("hit_points", num(r.hit_points)),
                    ("captain_regret", num(r.captain_regret)),
                    ("xi_regret", num(r.xi_regret)),
                ])
            })
            .collect(),
    )
}

fn season_table(rows: &[ByStart]) -> String {
    let headers = h(&[
        "season",
        "policy",
        "from GW",
        "starts",
        "GWs",
        "template",
        "mean",
        "min",
        "max",
        "xG mean",
        "null xG GWs",
        "transfers",
        "hit pts",
        "capt regret",
        "XI regret",
    ]);
    let body: Vec<Vec<String>> = rows
        .iter()
        .map(|r| {
            vec![
                season_label(r.season as i32),
                r.policy.clone(),
                r.start_gw.to_string(),
                r.starts.to_string(),
                num_text(r.n_gws, 1, false),
                num_text(r.template, 0, false),
                num_text(r.mean, 1, false),
                num_text(r.min, 0, false),
                num_text(r.max, 0, false),
                num_text(r.xg_mean, 1, false),
                num_text(r.xg_null, 1, false),
                num_text(r.n_transfers, 1, false),
                num_text(r.hit_points, 1, false),
                num_text(r.captain_regret, 2, false),
                num_text(r.xi_regret, 2, false),
            ]
        })
        .collect();
    text_table(&headers, &body, 2)
}

fn overall_table(rows: &[ByStart]) -> String {
    let mut xg_seasons: BTreeMap<i64, BTreeSet<i64>> = BTreeMap::new();
    let start_gws: BTreeSet<i64> = rows.iter().map(|r| r.start_gw).collect();
    for gw in &start_gws {
        let part: Vec<&ByStart> = rows.iter().filter(|r| r.start_gw == *gw).collect();
        let policies: BTreeSet<&String> = part.iter().map(|r| &r.policy).collect();
        let seasons: BTreeSet<i64> = part.iter().map(|r| r.season).collect();
        let ok = seasons
            .into_iter()
            .filter(|s| {
                policies.iter().all(|p| {
                    part.iter().any(|r| r.season == *s && &&r.policy == p && !r.xg_mean.is_nan())
                })
            })
            .collect();
        xg_seasons.insert(*gw, ok);
    }
    let mut groups: BTreeMap<(String, i64), Vec<&ByStart>> = BTreeMap::new();
    for r in rows {
        groups.entry((r.policy.clone(), r.start_gw)).or_default().push(r);
    }
    let headers = h(&[
        "policy",
        "from GW",
        "seasons",
        "template",
        "mean",
        "xG mean",
        "xG seasons",
        "transfers",
        "capt regret",
        "XI regret",
    ]);
    let body: Vec<Vec<String>> = groups
        .iter()
        .map(|((policy, gw), part)| {
            let col =
                |f: fn(&ByStart) -> f64| mean_of(&part.iter().map(|r| f(r)).collect::<Vec<_>>());
            let xg: Vec<f64> = part
                .iter()
                .filter(|r| xg_seasons[gw].contains(&r.season))
                .map(|r| r.xg_mean)
                .collect();
            vec![
                policy.clone(),
                gw.to_string(),
                part.len().to_string(),
                num_text(col(|r| r.template), 1, false),
                num_text(col(|r| r.mean), 1, false),
                num_text(mean_of(&xg), 1, false),
                xg_seasons[gw].len().to_string(),
                num_text(col(|r| r.n_transfers), 1, false),
                num_text(col(|r| r.captain_regret), 2, false),
                num_text(col(|r| r.xi_regret), 2, false),
            ]
        })
        .collect();
    text_table(&headers, &body, 2)
}

fn transfer_gain_table(summary: &Summary) -> String {
    let headers = h(&[
        "policy",
        "GWs",
        "transfers",
        "hit pts",
        "pred",
        "real",
        "pred net",
        "real net",
        "slope",
        "intercept",
    ]);
    let body: Vec<Vec<String>> = summary
        .gains
        .iter()
        .map(|g| {
            vec![
                g.policy.clone(),
                g.n_decisions.to_string(),
                num_text(g.transfers, 2, false),
                num_text(g.hit_points, 2, false),
                num_text(g.pred, 2, true),
                num_text(g.real, 2, true),
                num_text(g.pred_net, 2, true),
                num_text(g.real_net, 2, true),
                num_text(g.slope, 3, false),
                num_text(g.intercept, 2, true),
            ]
        })
        .collect();
    text_table(&headers, &body, 1)
}

fn comparison_table(summary: &Summary, k: usize, stride: usize) -> String {
    let headers = h(&[
        "method",
        "split",
        "metric",
        "mean diff",
        "deflated",
        "80% CI",
        "p",
        "season-t p",
        "seasons",
        "cells",
        "per season",
    ]);
    let body: Vec<Vec<String>> = summary
        .comparisons
        .iter()
        .map(|c| {
            let method = if c.method == "per_decision" {
                format!("per-decision (k={k}, stride={stride})")
            } else {
                "full run".to_owned()
            };
            vec![
                method,
                c.split.clone(),
                c.metric.clone(),
                num_text(c.result.mean, 3, true),
                num_text(c.deflated_mean, 3, true),
                format!(
                    "[{}, {}]",
                    num_text(c.result.ci_low, 3, true),
                    num_text(c.result.ci_high, 3, true)
                ),
                num_text(c.result.p_one_sided, 3, false),
                num_text(c.p_season_t, 3, false),
                c.result.n_seasons.to_string(),
                c.result.n_gws.to_string(),
                num_text(c.season_diff, 1, true),
            ]
        })
        .collect();
    text_table(&headers, &body, 3)
}

/// Per season: (Σ, mean) over cells of the start-averaged value.
fn per_season(diffs: &[Diff]) -> BTreeMap<i64, (f64, f64)> {
    let mut out: BTreeMap<i64, Vec<f64>> = BTreeMap::new();
    for ((season, _), v) in fplopt_backtest::stats::cell_means(diffs, false) {
        out.entry(season).or_default().push(v);
    }
    out.into_iter()
        .map(|(s, v)| (s, (v.iter().sum(), v.iter().sum::<f64>() / v.len() as f64)))
        .collect()
}

fn diffs_of(rows: &[fplopt_backtest::evaluate::DiffRow], xg: bool) -> Vec<Diff> {
    rows.iter()
        .map(|r| Diff {
            season: r.season,
            gw_index: r.gw_index,
            value: if xg { r.diff_xg } else { Some(r.diff) },
            weight: r.weight,
        })
        .collect()
}

fn per_season_diff_table(
    paired: &[fplopt_backtest::evaluate::Paired],
    decisions: Option<&[PerDecision]>,
    k: usize,
) -> String {
    let rows = paired_rows(paired);
    let full = per_season(&diffs_of(&rows, false));
    let full_xg = per_season(&diffs_of(&rows, true));
    let mut headers = h(&["season", "full run total", "per GW", "xG per GW"]);
    let dec = decisions.map(|d| {
        let rows = fplopt_backtest::evaluate::decision_rows(d);
        let mut same: BTreeMap<i64, (usize, usize)> = BTreeMap::new();
        for r in d {
            let e = same.entry(r.season).or_default();
            e.0 += usize::from(r.same_decision);
            e.1 += 1;
        }
        (
            season_points(&diffs_of(&rows, false)),
            per_season(&diffs_of(&rows, false)),
            per_season(&diffs_of(&rows, true)),
            same,
        )
    });
    if dec.is_some() {
        headers.extend(h(&[
            &format!("per-decision total (k={k})"),
            "per window",
            "xG per window",
            "decisions differ",
        ]));
    }
    let body: Vec<Vec<String>> = full
        .iter()
        .map(|(season, (sum, mean))| {
            let mut row = vec![
                season_label(*season as i32),
                num_text(*sum, 1, true),
                num_text(*mean, 3, true),
                num_text(full_xg.get(season).map_or(f64::NAN, |x| x.1), 3, true),
            ];
            if let Some((total, per, per_xg, same)) = &dec {
                row.push(num_text(total.get(season).copied().unwrap_or(f64::NAN), 1, true));
                row.push(num_text(per.get(season).map_or(f64::NAN, |x| x.1), 3, true));
                row.push(num_text(per_xg.get(season).map_or(f64::NAN, |x| x.1), 3, true));
                row.push(same.get(season).map_or("-".to_owned(), |(s, n)| {
                    format!("{:.0}%", 100.0 * (1.0 - *s as f64 / *n as f64))
                }));
            }
            row
        })
        .collect();
    text_table(&headers, &body, 1)
}

fn window_tests(
    paired: &[fplopt_backtest::evaluate::Paired],
) -> Vec<(String, String, f64, f64, usize)> {
    let rows = paired_rows(paired);
    let mut out = Vec::new();
    for (split, seasons) in splits() {
        let part: Vec<fplopt_backtest::evaluate::DiffRow> = rows
            .iter()
            .filter(|r| seasons.as_ref().is_none_or(|s| s.contains(&r.season)))
            .cloned()
            .collect();
        if part.is_empty() {
            continue;
        }
        for (metric, xg) in [("realized", false), ("xg", true)] {
            let t = full_run_window_test(&diffs_of(&part, xg), FULL_RUN_WINDOW);
            out.push((split.to_owned(), metric.to_owned(), t.mean, t.p_one_sided, t.n_cells));
        }
    }
    out
}

fn window_tests_json(tests: &[(String, String, f64, f64, usize)]) -> Ordered {
    Ordered::Array(
        tests
            .iter()
            .map(|(split, metric, mean, p, n)| {
                record(vec![
                    ("split", text(split)),
                    ("metric", text(metric)),
                    ("mean_per_gw", num(*mean)),
                    ("p_one_sided", num(*p)),
                    ("n_cells", int(*n as i64)),
                ])
            })
            .collect(),
    )
}

fn window_test_table(tests: &[(String, String, f64, f64, usize)]) -> String {
    let body: Vec<Vec<String>> = tests
        .iter()
        .map(|(split, metric, mean, p, n)| {
            vec![
                split.clone(),
                metric.clone(),
                num_text(*mean, 2, true),
                num_text(*p, 3, false),
                n.to_string(),
            ]
        })
        .collect();
    text_table(&h(&["split", "metric", "mean/GW", "p", "cells"]), &body, 1)
}

fn base_config(
    common: &Common,
    seasons: &[i64],
    policies: &[&PolicySpec],
    out: &Path,
    family: &str,
) -> Vec<(&'static str, Ordered)> {
    let mut config = vec![
        ("seasons", Ordered::Array(seasons.iter().map(|s| int(*s)).collect())),
        ("starts", text(&common.starts)),
        ("policies", Ordered::Array(policies.iter().map(|p| text(&p.to_string())).collect())),
        ("out", text(&out.display().to_string())),
        ("jobs", int(common.jobs as i64)),
        ("family", text(family)),
    ];
    let used: BTreeSet<i64> = policies
        .iter()
        .filter(|p| p.params.iter().any(|(k, _)| k == "chip_shrink"))
        .map(|p| p.params.iter().find(|(k, _)| k == "chip_split").map_or(0, |(_, v)| v.i()))
        .collect();
    if !used.is_empty() {
        let tables = used
            .iter()
            .map(|split| {
                let path = fplopt_backtest::specs::chip_table_file(*split);
                let entry = std::fs::read_to_string(&path)
                    .ok()
                    .and_then(|t| fplopt_core::pyjson::Ordered::parse(t.as_bytes()).ok())
                    .unwrap_or(Ordered::Null);
                (split.to_string(), entry)
            })
            .collect();
        config.push(("chip_values", Ordered::Object(tables)));
    }
    config
}

// --- the commands -----------------------------------------------------------------------------

fn backtest_run(args: &RunArgs) -> Result<()> {
    let spec = run_spec(args)?;
    let checked = check_common(&args.common, &[&spec], |seasons| format!("run {spec} {seasons}"))?;
    let out = out_dir(&args.common.out, "run");
    let policy = spec.build()?;
    let began = Instant::now();
    let caches = Caches::default();
    let results = run_grid(
        &checked.store,
        &checked.seasons,
        &checked.specs,
        std::slice::from_ref(&policy),
        &caches,
        args.common.jobs,
    )?;
    if results.is_empty() {
        anyhow::bail!("no start state could be built in any season");
    }
    let bs = Bootstrap { block_length: 4, n_boot: 2000, seed: 0, ci: CI_LEVEL, n_variants: 1 };
    let summary = summarize(&results, &[], None, &bs)?;
    let runtime = began.elapsed().as_secs_f64();
    let totals = season_totals_by_start(&results, &checked.specs);
    let config = base_config(&args.common, &checked.seasons, &[&spec], &out, &checked.family);
    std::fs::create_dir_all(&out)?;
    write_parquet(&grid_table(&results), &out.join("gws.parquet"))?;
    let command = command_line();
    let payload = record(vec![
        ("command", text(&command)),
        ("config", record(config.clone())),
        ("policy", text(&policy.name())),
        ("runtime_seconds", num((runtime * 10.0).round() / 10.0)),
        ("starts", starts_by_season(&results)),
        ("season_totals_by_start", by_start_json(&totals)),
        ("summary", summary.to_json()),
    ]);
    write_json(&out.join("summary.json"), &payload)?;
    println!(
        "{}, seasons {}, starts {} ({runtime:.0}s)\n\n{TOTALS_TITLE}\n{}\n\nOverall (means over seasons)\n{}\n\n{GAINS_TITLE}\n{}\n\nWritten to {}",
        policy.name(),
        format_seasons(&checked.seasons),
        args.common.starts,
        season_table(&totals),
        overall_table(&totals),
        transfer_gain_table(&summary),
        out.display()
    );
    let n_variants = family_variants(&args.common.experiments, &checked.family)? + 1;
    log_experiment(
        &args.common.experiments,
        &command,
        &record(config).to_value(),
        &summary.metrics().to_value(),
        n_variants,
        &checked.family,
    )
}

fn backtest_compare(args: &CompareArgs) -> Result<()> {
    let a = spec_or_usage(&args.a, "--a")?;
    let b = spec_or_usage(&args.b, "--b")?;
    let own = args.continuation.trim().eq_ignore_ascii_case("own");
    let continuation =
        if own { None } else { Some(spec_or_usage(&args.continuation, "--continuation")?) };
    let (policy_a, policy_b) = (a.build()?, b.build()?);
    if a == b || policy_a.name() == policy_b.name() {
        return usage(format!("--a and --b are the same policy ({a} vs {b})"));
    }
    let stride = args.stride.unwrap_or(args.k);
    if args.k.min(stride).min(args.block_length).min(args.n_boot) < 1 {
        return usage("--k, --stride, --block-length and --n-boot must be >= 1");
    }
    let per_decision_on = !args.no_per_decision;
    let mut specs: Vec<&PolicySpec> = vec![&a, &b];
    if per_decision_on && let Some(c) = &continuation {
        specs.push(c);
    }
    let checked = check_common(&args.common, &specs, |seasons| format!("{b} {seasons}"))?;
    let out = out_dir(&args.common.out, "compare");
    let n_variants = family_variants(&args.common.experiments, &checked.family)? + 1;
    let cont_policy = match &continuation {
        Some(c) => Some(c.build()?),
        None => None,
    };
    let caches = Caches::default();
    let began = Instant::now();
    let results = run_grid(
        &checked.store,
        &checked.seasons,
        &checked.specs,
        &[policy_a.clone(), policy_b.clone()],
        &caches,
        args.common.jobs,
    )?;
    if results.is_empty() {
        anyhow::bail!("no start state could be built in any season");
    }
    let (name_a, name_b) = (policy_a.name(), policy_b.name());
    let paired = paired_full_run(&results, &name_a, &name_b)?;
    let decisions = if per_decision_on {
        Some(per_decision(
            &checked.store,
            &checked.seasons,
            &checked.specs,
            &policy_a,
            &policy_b,
            cont_policy.as_ref(),
            args.k,
            stride,
            &args.reference,
            &caches,
            args.common.jobs,
        )?)
    } else {
        None
    };
    let bs = Bootstrap {
        block_length: args.block_length,
        n_boot: args.n_boot,
        seed: args.seed,
        ci: CI_LEVEL,
        n_variants,
    };
    let summary =
        summarize(&results, &[(name_a.clone(), name_b.clone())], decisions.as_deref(), &bs)?;
    let runtime = began.elapsed().as_secs_f64();
    let tests = window_tests(&paired);
    let totals = season_totals_by_start(&results, &checked.specs);
    let mut config = base_config(&args.common, &checked.seasons, &[&a, &b], &out, &checked.family);
    let cont_text = if own { "own".to_owned() } else { continuation.as_ref().unwrap().to_string() };
    config.extend([
        ("per_decision", Ordered::Bool(per_decision_on)),
        ("continuation", if per_decision_on { text(&cont_text) } else { Ordered::Null }),
        ("reference", if per_decision_on { text(&args.reference) } else { Ordered::Null }),
        ("k", int(args.k as i64)),
        ("stride", int(stride as i64)),
        ("block_length", int(args.block_length as i64)),
        ("n_boot", int(args.n_boot as i64)),
        ("seed", int(args.seed as i64)),
        ("ci", num(CI_LEVEL)),
    ]);
    std::fs::create_dir_all(&out)?;
    write_parquet(&grid_table(&results), &out.join("gws.parquet"))?;
    write_parquet(&paired_table(&paired), &out.join("paired.parquet"))?;
    if let Some(d) = &decisions {
        write_parquet(&decisions_table(d), &out.join("per_decision.parquet"))?;
    }
    let cont_name = if !per_decision_on {
        Ordered::Null
    } else {
        text(&cont_policy.as_ref().map_or("own".to_owned(), |c| c.name()))
    };
    let command = command_line();
    let payload = record(vec![
        ("command", text(&command)),
        ("config", record(config.clone())),
        ("policy_a", text(&name_a)),
        ("policy_b", text(&name_b)),
        ("continuation", cont_name.clone()),
        ("reference", if per_decision_on { text(&args.reference) } else { Ordered::Null }),
        ("family", text(&checked.family)),
        ("n_variants", int(n_variants as i64)),
        ("runtime_seconds", num((runtime * 10.0).round() / 10.0)),
        ("starts", starts_by_season(&results)),
        ("season_totals_by_start", by_start_json(&totals)),
        ("summary", summary.to_json()),
        ("full_run_window_tests", window_tests_json(&tests)),
    ]);
    write_json(&out.join("summary.json"), &payload)?;
    let cont = if per_decision_on {
        let reference = if args.reference == "a" { &name_a } else { &name_b };
        let name = match &cont_name {
            Ordered::Str(s) => s.clone(),
            _ => String::new(),
        };
        format!(", continuation {name}, reference {reference}")
    } else {
        String::new()
    };
    println!(
        "A = {name_a}, B = {name_b}, seasons {}, starts {}{cont} ({runtime:.0}s)\n\n{TOTALS_TITLE}\n{}\n\n\
         Overall (means over seasons)\n{}\n\n\
         Paired differences A - B (full run: mean per GW; per-decision: mean per {}-GW decision window, \
         decisions at gw_index 1 + {stride}j; 80% CI, two-sided (lower bound = one-sided alpha 0.10), from a \
         GW-block bootstrap by season, blocks of {} GWs, {} resamples; p one-sided, H1: A > B; season-t p = \
         one-sided t-test on the per-season mean differences; deflated = mean - SE*sqrt(2 ln N), N = \
         {n_variants} variant(s) of family '{}'; splits: develop 2016-17..2022-23, validate \
         2023-24..2024-25; realized@xg = realized points on the xG metric's sample; cells = GWs or \
         decision windows; per season = difference in season points)\n{}\n\nPer season (A - B)\n{}\n\n\
         {GAINS_TITLE}\n{}\n\nFull-run window t-test (A - B per GW, non-overlapping {FULL_RUN_WINDOW}-GW \
         cells averaged over GWs and starts, one-sided, df = cells - 1; the chip study's statistic)\n{}\n\n\
         Written to {}",
        format_seasons(&checked.seasons),
        args.common.starts,
        season_table(&totals),
        overall_table(&totals),
        args.k,
        args.block_length,
        args.n_boot,
        checked.family,
        comparison_table(&summary, args.k, stride),
        per_season_diff_table(&paired, decisions.as_deref(), args.k),
        transfer_gain_table(&summary),
        window_test_table(&tests),
        out.display()
    );
    let mut metrics = match summary.metrics() {
        Ordered::Object(pairs) => pairs,
        _ => Vec::new(),
    };
    metrics.push(("full_run_window_tests".to_owned(), window_tests_json(&tests)));
    log_experiment(
        &args.common.experiments,
        &command,
        &record(config).to_value(),
        &Ordered::Object(metrics).to_value(),
        n_variants,
        &checked.family,
    )
}
