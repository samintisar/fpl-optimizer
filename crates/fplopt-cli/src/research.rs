//! Research commands (Rust only). `research money-value` (price sizing study, addendum of
//! 2026-11-06): the expected points per GW that £1m in the bank buys the live planner, from
//! paired solves at fixed states (`fplopt_backtest::money_value`); writes
//! money_value.parquet and summary.json to --out. `research price-eval` (price forecast
//! Stage B): the walk-forward evaluation of the price forecast against no change and
//! momentum (`fplopt_backtest::price_eval`); writes predictions.parquet and metrics.json to
//! --out and appends the run to the experiment log. `research night-eval` (Gate N): the
//! walk-forward evaluation of the night price model against a base rate and net transfers
//! per owner, by log loss (`fplopt_backtest::night_eval`); writes predictions.parquet and
//! metrics.json to --out and appends the run to the experiment log. Usage errors exit 2.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Result;
use clap::{Args, Subcommand};
use fplopt_backtest::money_value::{
    LIVE_SPEC, POOLED_FROM, measure, rows_table, specs_at, summary,
};
use fplopt_backtest::night_eval::{self, evaluate_nights};
use fplopt_backtest::price_eval::{self, PriceEval, check_seasons, evaluate_prices};
use fplopt_backtest::simulator::Caches;
use fplopt_backtest::specs::parse_policy_spec;
use fplopt_backtest::xp_cache;
use fplopt_build::table::write_parquet;
use fplopt_core::experiments::{family_variants, log_experiment, read_experiments};
use fplopt_core::pyjson::{Ordered, dumps_indent};
use fplopt_core::seasons::{HOLDOUT_SEASONS, season_label};
use fplopt_core::settings::Settings;
use fplopt_models::night::NightParams;
use fplopt_models::price::PriceParams;
use fplopt_store::DataStore;

use crate::backtest::{
    XpCacheArgs, command_line, default_jobs, format_seasons, num_text, seasons_with_data,
    text_table, write_json,
};
use crate::optimize_plan::{exit_code, parse_seasons, usage};

#[derive(Subcommand)]
pub enum ResearchJob {
    /// Expected points per GW that £1m buys the live planner (two solves per state).
    MoneyValue(MoneyValueArgs),
    /// Walk-forward evaluation of the price forecast against no change and momentum.
    PriceEval(PriceEvalArgs),
    /// Gate N: walk-forward log loss of the night price model against two baselines.
    NightEval(NightEvalArgs),
}

#[derive(Args)]
pub struct NightEvalArgs {
    /// Evaluated start years: 2022 or 2021-2022 (each after --first-fit, 2022/23 at the
    /// latest; later seasons and the holdout are refused).
    #[arg(long)]
    pub seasons: String,
    /// First season of every fit (season S is fitted on --first-fit..S-1; snapshots start in
    /// 2020/21).
    #[arg(long, default_value_t = 2020)]
    pub first_fit: i64,
    /// Experiment-log family.
    #[arg(long, default_value = "night price model")]
    pub family: String,
    #[arg(long, default_value = "results/experiments.csv")]
    pub experiments: PathBuf,
    /// The data directory (default FPLOPT_DATA_DIR or data).
    #[arg(long)]
    pub data_dir: Option<PathBuf>,
    #[arg(long)]
    pub out: PathBuf,
}

#[derive(Args)]
pub struct PriceEvalArgs {
    /// Evaluated start years: 2022, 2022-2024 or 2022-23 (each after --first-fit; the
    /// holdout is refused).
    #[arg(long)]
    pub seasons: String,
    /// The forecast's learning preset (p1..p6).
    #[arg(long)]
    pub params: String,
    /// First season of every fit (season S is fitted on --first-fit..S-1).
    #[arg(long, default_value_t = 2021)]
    pub first_fit: i64,
    /// Threads over seasons (default: CPUs - 1); results equal --jobs 1.
    #[arg(long, default_value_t = default_jobs())]
    pub jobs: usize,
    /// Experiment-log family (default 'price forecast'; 'price forecast gate b' is the gate's
    /// run: once, from a clean checkout at the tag `price-forecast-gate-b`).
    #[arg(long)]
    pub family: Option<String>,
    #[arg(long, default_value = "results/experiments.csv")]
    pub experiments: PathBuf,
    /// The data directory (default FPLOPT_DATA_DIR or data).
    #[arg(long)]
    pub data_dir: Option<PathBuf>,
    #[arg(long)]
    pub out: PathBuf,
    #[command(flatten)]
    pub xp_cache: XpCacheArgs,
}

/// The snapshot features start in 2021/22: an earlier fit would train on all-NaN snapshots.
pub const FIRST_FIT_MIN: i64 = 2021;
/// The family of Gate B's run, which may run once (`gate_refusal`).
pub const GATE_FAMILY: &str = "price forecast gate b";
/// The tag of the commit Gate B's run is made from.
pub const GATE_TAG: &str = "price-forecast-gate-b";
/// Changes `git status` may show in a gate run's checkout (the log it appends to).
const LOG_FILES: [&str; 1] = ["results/experiments.csv"];

/// The code checkout as git sees it, for Gate B's refusals.
#[derive(Debug, Clone, PartialEq)]
pub struct Checkout {
    /// `git status --porcelain` lines, without `LOG_FILES` (empty: clean).
    pub dirty: Vec<String>,
    /// `git rev-parse HEAD`.
    pub head: String,
    /// The commit `GATE_TAG` points at, if the tag exists.
    pub tag: Option<String>,
    /// The commit this binary was built from (`experiments::GIT_SHA`).
    pub built: String,
}

/// Why a `GATE_FAMILY` run must not run (None: it may; other families are never refused):
/// the log already holds a row of the family, the checkout is dirty, HEAD isn't the tagged
/// commit, or the binary wasn't built from HEAD.
pub fn gate_refusal(
    family: &str,
    log: &[Vec<(String, String)>],
    checkout: &Checkout,
) -> Option<String> {
    if family != GATE_FAMILY {
        return None;
    }
    if log.iter().any(|row| row.iter().any(|(k, v)| k == "family" && v == family)) {
        return Some(format!("the experiment log already holds a '{family}' run; it runs once"));
    }
    if !checkout.dirty.is_empty() {
        return Some(format!(
            "the checkout has uncommitted changes ({}); the gate runs from a clean checkout",
            checkout.dirty.join("; ")
        ));
    }
    match &checkout.tag {
        None => return Some(format!("no tag {GATE_TAG} (git fetch --tags?)")),
        Some(tag) if *tag != checkout.head => {
            return Some(format!(
                "HEAD {} is not the commit tagged {GATE_TAG} ({tag})",
                checkout.head
            ));
        }
        Some(_) => {}
    }
    if checkout.built != checkout.head {
        return Some(format!(
            "this binary was built from {}, not HEAD {} (rebuild)",
            checkout.built, checkout.head
        ));
    }
    None
}

fn git(root: &std::path::Path, args: &[&str]) -> Result<(bool, String)> {
    let out = std::process::Command::new("git").args(args).current_dir(root).output();
    match out {
        Ok(o) => Ok((o.status.success(), String::from_utf8_lossy(&o.stdout).into_owned())),
        Err(e) => usage(format!("can't check the checkout at {} with git: {e}", root.display())),
    }
}

/// The checkout this binary's source lives in (the workspace root), as git sees it.
fn checkout() -> Result<Checkout> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
    let (ok, status) = git(&root, &["status", "--porcelain", "--untracked-files=normal"])?;
    let (ok_head, head) = git(&root, &["rev-parse", "HEAD"])?;
    if !ok || !ok_head {
        return usage(format!("can't check the checkout at {} with git", root.display()));
    }
    let dirty = status
        .lines()
        .filter(|l| !LOG_FILES.contains(&l.get(3..).unwrap_or("").trim()))
        .map(str::to_owned)
        .collect();
    let (has_tag, tag) =
        git(&root, &["rev-parse", "--verify", "--quiet", &format!("{GATE_TAG}^{{commit}}")])?;
    Ok(Checkout {
        dirty,
        head: head.trim().to_owned(),
        tag: has_tag.then(|| tag.trim().to_owned()),
        built: fplopt_core::experiments::GIT_SHA.to_owned(),
    })
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
        ResearchJob::PriceEval(args) => price_eval(args),
        ResearchJob::NightEval(args) => night_eval(args),
    }))
}

fn get<'a>(o: &'a Ordered, key: &str) -> Option<&'a Ordered> {
    match o {
        Ordered::Object(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
        _ => None,
    }
}

fn text(o: Option<&Ordered>) -> String {
    match o {
        Some(Ordered::Str(s)) => s.clone(),
        Some(Ordered::Int(i)) => i.to_string(),
        _ => String::new(),
    }
}

fn number(o: Option<&Ordered>) -> f64 {
    match o {
        Some(Ordered::Float(x)) => *x,
        Some(Ordered::Int(i)) => *i as f64,
        _ => f64::NAN,
    }
}

fn items<'a>(o: &'a Ordered, key: &str) -> &'a [Ordered] {
    match get(o, key) {
        Some(Ordered::Array(v)) => v,
        _ => &[],
    }
}

/// The MSE table and the Diebold-Mariano table of a price-eval run's metrics.
fn price_tables(metrics: &Ordered) -> (String, String) {
    let headers = |names: &[&str]| names.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let mse: Vec<Vec<String>> = items(metrics, "mse")
        .iter()
        .map(|r| {
            let n = |k: &str| num_text(number(get(r, k)), 4, false);
            vec![
                text(get(r, "split")),
                text(get(r, "horizon")),
                text(get(r, "n")),
                text(get(r, "n_deadlines")),
                n("forecast"),
                n("n0"),
                n("n1"),
            ]
        })
        .collect();
    let dm: Vec<Vec<String>> = items(metrics, "dm")
        .iter()
        .map(|r| {
            vec![
                text(get(r, "a")),
                text(get(r, "b")),
                text(get(r, "split")),
                text(get(r, "n_clusters")),
                num_text(number(get(r, "value_a")), 4, false),
                num_text(number(get(r, "value_b")), 4, false),
                num_text(number(get(r, "mean_diff")), 4, true),
                num_text(number(get(r, "p_a_better")), 3, false),
                num_text(number(get(r, "p_b_better")), 3, false),
            ]
        })
        .collect();
    (
        text_table(&headers(&["split", "h", "rows", "deadlines", "forecast", "N0", "N1"]), &mse, 2),
        text_table(
            &headers(&["A", "B", "split", "deadlines", "A", "B", "A - B", "p(A<B)", "p(B<A)"]),
            &dm,
            3,
        ),
    )
}

fn price_eval(args: &PriceEvalArgs) -> Result<()> {
    let seasons = parse_seasons(&args.seasons, "--seasons")?;
    if args.jobs < 1 {
        return usage("--jobs must be >= 1");
    }
    let params = match PriceParams::preset(&args.params) {
        Ok(p) => p,
        Err(e) => return usage(format!("argument --params: {e}")),
    };
    if args.first_fit < FIRST_FIT_MIN {
        return usage(format!(
            "argument --first-fit: {} or later (the snapshot features start in {}; an earlier \
             fit would train on all-NaN snapshot features)",
            FIRST_FIT_MIN,
            season_label(FIRST_FIT_MIN as i32)
        ));
    }
    if let Err(e) = check_seasons(&seasons, args.first_fit) {
        return usage(format!("argument --seasons: {e}"));
    }
    let family = args.family.clone().unwrap_or_else(|| "price forecast".to_owned());
    if family == GATE_FAMILY {
        let log = read_experiments(&args.experiments)?;
        if let Some(reason) = gate_refusal(&family, &log, &checkout()?) {
            return usage(format!("refusing the gate run: {reason}"));
        }
    }
    let data_dir = args.data_dir.clone().unwrap_or_else(|| Settings::from_env().data_dir);
    let store = match DataStore::open(&data_dir) {
        Ok(s) => s,
        Err(e) => return usage(format!("{e:#}")),
    };
    let available = seasons_with_data(&store)?;
    let last = seasons.iter().max().copied().unwrap_or(args.first_fit);
    let missing: Vec<String> = (args.first_fit..=last)
        .filter(|s| !available.contains(s) && !HOLDOUT_SEASONS.contains(&(*s as i32)))
        .map(|s| season_label(s as i32))
        .collect();
    if !missing.is_empty() {
        return usage(format!(
            "no gameweek/player_match data for {} (run fplopt build all?)",
            missing.join(", ")
        ));
    }
    let caches = args.xp_cache.caches_for(&data_dir)?;
    let began = std::time::Instant::now();
    let PriceEval { rows, metrics, timings } =
        evaluate_prices(&store, &caches, &seasons, args.first_fit, &params, args.jobs)?;
    let runtime = began.elapsed().as_secs_f64();
    let int = |i: i64| Ordered::Int(i128::from(i));
    let pair = |k: &str, v: Ordered| (k.to_owned(), v);
    let config = Ordered::Object(vec![
        pair("seasons", Ordered::Array(seasons.iter().map(|s| int(*s)).collect())),
        pair("params", Ordered::Str(params.name.clone())),
        pair("first_fit", int(args.first_fit)),
        pair("out", Ordered::Str(args.out.display().to_string())),
        pair("jobs", int(args.jobs as i64)),
        pair("family", Ordered::Str(family.clone())),
    ]);
    let n_variants = family_variants(&args.experiments, &family)? + 1;
    std::fs::create_dir_all(&args.out)?;
    write_parquet(&price_eval::rows_table(&rows), &args.out.join("predictions.parquet"))?;
    let command = command_line();
    let seconds = timings.iter().map(|(s, t)| (s.clone(), Ordered::Float(*t))).collect();
    let payload = Ordered::Object(vec![
        pair("command", Ordered::Str(command.clone())),
        pair("config", config.clone()),
        pair("family", Ordered::Str(family.clone())),
        pair("n_variants", int(n_variants as i64)),
        pair("runtime_seconds", Ordered::Float((runtime * 10.0).round() / 10.0)),
        pair("season_seconds", Ordered::Object(seconds)),
        pair("metrics", metrics.clone()),
    ]);
    write_json(&args.out.join("metrics.json"), &payload)?;
    // Logged before any result reaches stdout, so a run can't be seen and then not logged.
    log_experiment(
        &args.experiments,
        &command,
        &config.to_value(),
        &metrics.to_value(),
        n_variants,
        &family,
    )?;
    let (mse, dm) = price_tables(&metrics);
    println!(
        "price eval: {} ({}), fitted from {}; {} deadline(s), {} row(s), {} with a realized \
         change ({runtime:.0}s)\n\nMSE of the change in tenths of £m, squared (N0 no change, N1 \
         momentum)\n{mse}\n\nDiebold-Mariano, one-sided, per-deadline MSE pooled over h, lag 6, \
         clustered by deadline\n{dm}\n\nWritten to {}",
        format_seasons(&seasons),
        params.name,
        season_label(args.first_fit as i32),
        text(get(&metrics, "n_deadlines")),
        text(get(&metrics, "n_rows")),
        text(get(&metrics, "n_scored")),
        args.out.display()
    );
    Ok(())
}

/// The log-loss table, the Diebold-Mariano table and the gate table of a night-eval run.
fn night_tables(metrics: &Ordered) -> (String, String, String) {
    let headers = |names: &[&str]| names.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let loss: Vec<Vec<String>> = items(metrics, "log_loss")
        .iter()
        .map(|r| {
            let n = |k: &str| num_text(number(get(r, k)), 5, false);
            vec![text(get(r, "target")), n("model"), n("b0"), n("b1")]
        })
        .collect();
    let dm: Vec<Vec<String>> = items(metrics, "dm")
        .iter()
        .map(|r| {
            vec![
                text(get(r, "target")),
                text(get(r, "baseline")),
                text(get(r, "n_clusters")),
                num_text(number(get(r, "value_a")), 5, false),
                num_text(number(get(r, "value_b")), 5, false),
                num_text(number(get(r, "mean_diff")), 5, true),
                num_text(number(get(r, "p_a_better")), 3, false),
                num_text(number(get(r, "p_b_better")), 3, false),
            ]
        })
        .collect();
    let gate: Vec<Vec<String>> = night_eval::gate_of(metrics)
        .iter()
        .map(|g| {
            vec![
                g.target.to_owned(),
                g.baseline.to_owned(),
                num_text(g.p, 3, false),
                (if g.pass { "pass" } else { "fail" }).to_owned(),
            ]
        })
        .collect();
    (
        text_table(&headers(&["target", "model", "B0", "B1"]), &loss, 1),
        text_table(
            &headers(&[
                "target",
                "B",
                "nights",
                "model",
                "B",
                "model - B",
                "p(model<B)",
                "p(B<model)",
            ]),
            &dm,
            2,
        ),
        text_table(&headers(&["target", "B", "p", "result"]), &gate, 2),
    )
}

fn night_eval(args: &NightEvalArgs) -> Result<()> {
    let seasons = parse_seasons(&args.seasons, "--seasons")?;
    if let Err(e) = night_eval::check_seasons(&seasons, args.first_fit) {
        return usage(format!("argument --seasons: {e}"));
    }
    let data_dir = args.data_dir.clone().unwrap_or_else(|| Settings::from_env().data_dir);
    let store = match DataStore::open(&data_dir) {
        Ok(s) => s,
        Err(e) => return usage(format!("{e:#}")),
    };
    let available = seasons_with_data(&store)?;
    let last = seasons.iter().max().copied().unwrap_or(args.first_fit);
    let missing: Vec<String> = (args.first_fit..=last)
        .filter(|s| !available.contains(s))
        .map(|s| season_label(s as i32))
        .collect();
    if !missing.is_empty() {
        return usage(format!(
            "no gameweek/player_match data for {} (run fplopt build all?)",
            missing.join(", ")
        ));
    }
    let params = NightParams::default();
    let began = std::time::Instant::now();
    let (rows, metrics) = evaluate_nights(&store, &seasons, args.first_fit, &params)?;
    let runtime = began.elapsed().as_secs_f64();
    let int = |i: i64| Ordered::Int(i128::from(i));
    let pair = |k: &str, v: Ordered| (k.to_owned(), v);
    let family = args.family.clone();
    let config = Ordered::Object(vec![
        pair("seasons", Ordered::Array(seasons.iter().map(|s| int(*s)).collect())),
        pair("first_fit", int(args.first_fit)),
        pair("out", Ordered::Str(args.out.display().to_string())),
        pair("family", Ordered::Str(family.clone())),
    ]);
    let n_variants = family_variants(&args.experiments, &family)? + 1;
    std::fs::create_dir_all(&args.out)?;
    write_parquet(&night_eval::rows_table(&rows), &args.out.join("predictions.parquet"))?;
    let command = command_line();
    let payload = Ordered::Object(vec![
        pair("command", Ordered::Str(command.clone())),
        pair("config", config.clone()),
        pair("family", Ordered::Str(family.clone())),
        pair("n_variants", int(n_variants as i64)),
        pair("runtime_seconds", Ordered::Float((runtime * 10.0).round() / 10.0)),
        pair("metrics", metrics.clone()),
    ]);
    write_json(&args.out.join("metrics.json"), &payload)?;
    // Logged before any result reaches stdout, so a run can't be seen and then not logged.
    log_experiment(
        &args.experiments,
        &command,
        &config.to_value(),
        &metrics.to_value(),
        n_variants,
        &family,
    )?;
    let (loss, dm, gate) = night_tables(&metrics);
    let verdict = if matches!(get(&metrics, "gate_pass"), Some(Ordered::Bool(true))) {
        "PASS"
    } else {
        "FAIL"
    };
    println!(
        "night eval: {}, fitted from {}; {} night(s), {} scored player-night(s) ({runtime:.0}s)\n\n\
         Log loss (B0 training base rate, B1 net transfers per owner)\n{loss}\n\n\
         Diebold-Mariano, one-sided, per-night mean log loss, lag {}, clustered by night\n{dm}\n\n\
         Gate N: {verdict} (the model beats both baselines on both targets, p < {})\n{gate}\n\n\
         Written to {}",
        format_seasons(&seasons),
        season_label(args.first_fit as i32),
        text(get(&metrics, "n_nights")),
        text(get(&metrics, "n_rows")),
        night_eval::DM_LAG,
        night_eval::ALPHA,
        args.out.display()
    );
    Ok(())
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
        let ResearchJob::MoneyValue(a) = cli.job else { panic!("not money-value") };
        assert_eq!(a.gws, [1, 8, 15]);
        assert_eq!((a.extra_bank, a.jobs), (5, 3));
        assert_eq!(a.starts, "template,random:3");
        assert_eq!(a.spec, LIVE_SPEC);
        assert_eq!(a.data_dir, Some(PathBuf::from("d")));
        let defaults =
            Cli::try_parse_from(["x", "money-value", "--seasons", "2021", "--out", "o"]).unwrap();
        let ResearchJob::MoneyValue(d) = defaults.job else { panic!("not money-value") };
        assert_eq!((d.gws, d.extra_bank), (vec![1, 8, 15, 22, 29], 10));
        assert!(Cli::try_parse_from(["x", "money-value", "--seasons", "2021"]).is_err());
    }

    #[test]
    fn price_eval_args() {
        let cli = Cli::try_parse_from([
            "x",
            "price-eval",
            "--seasons",
            "2022",
            "--params",
            "p3",
            "--first-fit",
            "2020",
            "--jobs",
            "2",
            "--experiments",
            "e.csv",
            "--no-xp-cache",
            "--out",
            "o",
        ])
        .unwrap();
        let ResearchJob::PriceEval(a) = cli.job else { panic!("not price-eval") };
        assert_eq!((a.seasons.as_str(), a.params.as_str()), ("2022", "p3"));
        assert_eq!((a.first_fit, a.jobs), (2020, 2));
        assert_eq!(a.experiments, PathBuf::from("e.csv"));
        assert!(a.xp_cache.no_xp_cache && a.family.is_none());
        let defaults = Cli::try_parse_from([
            "x",
            "price-eval",
            "--seasons",
            "2022",
            "--params",
            "p1",
            "--out",
            "o",
        ])
        .unwrap();
        let ResearchJob::PriceEval(d) = defaults.job else { panic!("not price-eval") };
        assert_eq!(d.first_fit, 2021);
        assert_eq!(d.experiments, PathBuf::from("results/experiments.csv"));
        assert!(
            Cli::try_parse_from(["x", "price-eval", "--seasons", "2022", "--out", "o"]).is_err()
        );
        assert!(d.data_dir.is_none());
    }

    #[test]
    fn night_eval_args() {
        let cli = Cli::try_parse_from([
            "x",
            "night-eval",
            "--seasons",
            "2022",
            "--first-fit",
            "2021",
            "--family",
            "f",
            "--experiments",
            "e.csv",
            "--data-dir",
            "d",
            "--out",
            "o",
        ])
        .unwrap();
        let ResearchJob::NightEval(a) = cli.job else { panic!("not night-eval") };
        assert_eq!((a.seasons.as_str(), a.first_fit, a.family.as_str()), ("2022", 2021, "f"));
        assert_eq!(a.experiments, PathBuf::from("e.csv"));
        assert_eq!(a.data_dir, Some(PathBuf::from("d")));
        let defaults =
            Cli::try_parse_from(["x", "night-eval", "--seasons", "2022", "--out", "o"]).unwrap();
        let ResearchJob::NightEval(d) = defaults.job else { panic!("not night-eval") };
        assert_eq!((d.first_fit, d.family.as_str()), (2020, "night price model"));
        assert_eq!(d.experiments, PathBuf::from("results/experiments.csv"));
        assert!(d.data_dir.is_none());
        assert!(Cli::try_parse_from(["x", "night-eval", "--seasons", "2022"]).is_err());
        // no --params, --jobs or xP cache
        assert!(
            Cli::try_parse_from([
                "x",
                "night-eval",
                "--seasons",
                "2022",
                "--jobs",
                "2",
                "--out",
                "o"
            ])
            .is_err()
        );
    }

    #[test]
    fn night_eval_refuses_seasons_outside_develop_before_reading_data() {
        // a data dir that doesn't exist: the refusal comes first
        for (seasons, first_fit, needle) in
            [("2023", "2020", "2023-24"), ("2025", "2020", "2025-26"), ("2022", "2019", "2020-21")]
        {
            let cli = Cli::try_parse_from([
                "x",
                "night-eval",
                "--seasons",
                seasons,
                "--first-fit",
                first_fit,
                "--data-dir",
                "no-such-dir",
                "--out",
                "o",
            ])
            .unwrap();
            let ResearchJob::NightEval(a) = cli.job else { panic!("not night-eval") };
            let err = night_eval(&a).unwrap_err();
            let usage = err.downcast_ref::<crate::optimize_plan::Usage>().expect("a usage error");
            assert!(usage.to_string().contains(needle), "{usage}");
        }
    }

    fn log_row(family: &str) -> Vec<(String, String)> {
        vec![("command".to_owned(), "c".to_owned()), ("family".to_owned(), family.to_owned())]
    }

    fn clean() -> Checkout {
        Checkout { dirty: vec![], head: "abc".into(), tag: Some("abc".into()), built: "abc".into() }
    }

    #[test]
    fn gate_runs_once_from_the_tagged_clean_build() {
        let other = vec![log_row("price forecast"), log_row("price forecast develop")];
        assert_eq!(gate_refusal(GATE_FAMILY, &other, &clean()), None);
        // Already logged.
        let once = vec![log_row("price forecast"), log_row(GATE_FAMILY)];
        let r = gate_refusal(GATE_FAMILY, &once, &clean()).unwrap();
        assert!(r.contains("already"), "{r}");
        // Dirty checkout.
        let dirty = Checkout { dirty: vec![" M src/x.rs".into()], ..clean() };
        let r = gate_refusal(GATE_FAMILY, &other, &dirty).unwrap();
        assert!(r.contains("uncommitted") && r.contains("src/x.rs"), "{r}");
        // HEAD not the tagged commit, or no tag at all.
        let moved = Checkout { head: "def".into(), built: "def".into(), ..clean() };
        let r = gate_refusal(GATE_FAMILY, &other, &moved).unwrap();
        assert!(r.contains("not the commit tagged") && r.contains(GATE_TAG), "{r}");
        let untagged = Checkout { tag: None, ..clean() };
        assert!(gate_refusal(GATE_FAMILY, &other, &untagged).unwrap().contains("no tag"));
        // A binary built from another commit.
        let stale = Checkout { built: "old".into(), ..clean() };
        assert!(gate_refusal(GATE_FAMILY, &other, &stale).unwrap().contains("rebuild"));
        // Other families are never refused, whatever the checkout and log.
        let worst = Checkout { dirty: vec!["?? a".into()], tag: None, ..moved };
        assert_eq!(gate_refusal("price forecast", &once, &worst), None);
    }

    #[test]
    fn first_fit_before_the_snapshots_is_a_usage_error() {
        let cli = Cli::try_parse_from([
            "x",
            "price-eval",
            "--seasons",
            "2022",
            "--params",
            "p1",
            "--first-fit",
            "2020",
            "--out",
            "o",
        ])
        .unwrap();
        let ResearchJob::PriceEval(a) = cli.job else { panic!("not price-eval") };
        let err = price_eval(&a).unwrap_err();
        let usage = err.downcast_ref::<crate::optimize_plan::Usage>().expect("a usage error");
        assert!(usage.to_string().contains("2021"), "{usage}");
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
