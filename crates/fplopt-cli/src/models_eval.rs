//! `models eval` (Python `cli._models_eval`, `validate_models_eval` and the tables):
//! walk-forward predictions of --models at every deadline of --seasons
//! (`fplopt_backtest::eval`); writes predictions.parquet, regrets.parquet and metrics.json to
//! --out, prints the summary and Diebold-Mariano tables and appends the run to the experiment
//! log. Usage errors exit 2.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use anyhow::Result;
use clap::Args;
use fplopt_backtest::eval::{EvalResult, REALIZED, evaluate};
use fplopt_backtest::evaluate::record;
use fplopt_build::table::{Column, Table, write_parquet};
use fplopt_core::experiments::{family_variants, log_experiment};
use fplopt_core::pyjson::Ordered;
use fplopt_core::seasons::season_label;
use fplopt_core::settings::Settings;
use fplopt_store::DataStore;

use crate::backtest::{
    command_line, format_seasons, num_text, seasons_with_data, text_table, write_json,
};
use crate::optimize_plan::{exit_code, parse_seasons, usage};

const XP_MODELS: [&str; 4] = ["rolling", "ep_next", "ep_next_fade", "v1"];
const EP_NEXT_FIRST_SEASON: i64 = 2021;

#[derive(Args)]
pub struct EvalArgs {
    /// Comma-separated xP models, e.g. rolling,ep_next.
    #[arg(long)]
    pub models: String,
    /// Start years: 2016-2024, 2021,2023, 2021-22 or a mix (the holdout is refused).
    #[arg(long)]
    pub seasons: String,
    /// Output directory (default results/<UTC time>-eval).
    #[arg(long)]
    pub out: Option<PathBuf>,
    #[arg(long, default_value = "results/experiments.csv")]
    pub experiments: PathBuf,
    /// Threads over seasons (default: CPUs - 1); results equal --jobs 1.
    #[arg(long, default_value_t = std::thread::available_parallelism().map_or(2, |n| n.get()).saturating_sub(1).max(1))]
    pub jobs: usize,
    /// Experiment-log family (default 'eval <models> <seasons>').
    #[arg(long)]
    pub family: Option<String>,
    /// Random squads per deadline for the regret metrics, besides the template.
    #[arg(long, default_value_t = 3)]
    pub n_random: i64,
    /// First random squad seed.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
}

pub fn run(args: &EvalArgs) -> Result<ExitCode> {
    Ok(exit_code(run_eval(args)))
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

fn rows<'a>(metrics: &'a Ordered, key: &str) -> &'a [Ordered] {
    match get(metrics, key) {
        Some(Ordered::Array(items)) => items,
        _ => &[],
    }
}

fn summary_table(metrics: &Ordered, models: &[String]) -> String {
    let headers: Vec<String> = [
        "model",
        "split",
        "deadlines",
        "MSE h0",
        "cand MSE h0",
        "MAE h0",
        "MSE h1-5",
        "XI regret",
        "capt regret",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let find = |key: &str, model: &str, split: &str, horizon: Option<&str>| {
        rows(metrics, key).iter().find(|r| {
            text(get(r, "model")) == model
                && text(get(r, "split")) == split
                && horizon.is_none_or(|h| text(get(r, "horizon")) == h)
        })
    };
    let mut body = Vec::new();
    for model in models {
        for split in ["all", "develop", "validate"] {
            let Some(h0) = find("xp", model, split, Some("0")) else { continue };
            let later = find("xp", model, split, Some("1-5"));
            let r = find("regret", model, split, None);
            body.push(vec![
                model.clone(),
                split.to_owned(),
                text(get(h0, "n_deadlines")),
                num_text(number(get(h0, "mse")), 3, false),
                num_text(number(get(h0, "mse_candidates")), 3, false),
                num_text(number(get(h0, "mae")), 3, false),
                num_text(later.map_or(f64::NAN, |l| number(get(l, "mse"))), 3, false),
                num_text(r.map_or(f64::NAN, |r| number(get(r, "xi_regret"))), 2, false),
                num_text(r.map_or(f64::NAN, |r| number(get(r, "captain_regret"))), 2, false),
            ]);
        }
    }
    text_table(&headers, &body, 2)
}

fn dm_table(metrics: &Ordered) -> String {
    let headers: Vec<String> =
        ["A", "B", "metric", "horizon", "split", "seasons", "A", "B", "A - B", "p(A<B)", "p(B<A)"]
            .iter()
            .map(|s| s.to_string())
            .collect();
    let mut body = Vec::new();
    for r in rows(metrics, "dm") {
        let horizon = text(get(r, "horizon"));
        let metric = text(get(r, "metric"));
        if (horizon != "0" && horizon != "1-5") || (horizon == "1-5" && metric != "mse") {
            continue;
        }
        let seasons: Vec<i64> = match get(r, "seasons") {
            Some(Ordered::Array(s)) => s.iter().map(|x| number(Some(x)) as i64).collect(),
            _ => Vec::new(),
        };
        body.push(vec![
            text(get(r, "a")),
            text(get(r, "b")),
            metric,
            horizon,
            text(get(r, "split")),
            format_seasons(&seasons),
            num_text(number(get(r, "value_a")), 3, false),
            num_text(number(get(r, "value_b")), 3, false),
            num_text(number(get(r, "mean_diff")), 3, true),
            num_text(number(get(r, "p_a_better")), 3, false),
            num_text(number(get(r, "p_b_better")), 3, false),
        ]);
    }
    text_table(&headers, &body, 6)
}

fn predictions_table(result: &EvalResult) -> Table {
    let p = &result.predictions;
    let i = |f: &dyn Fn(&fplopt_backtest::eval::Prediction) -> i64| {
        Column::I64(p.iter().map(f).collect())
    };
    let mut t = Table::new()
        .with("model", Column::Str(p.iter().map(|r| result.models[r.model].clone()).collect()))
        .with("season", i(&|r| r.season))
        .with("deadline", Column::Ts(p.iter().map(|r| r.deadline).collect()))
        .with("gw", i(&|r| r.gw))
        .with("gw_index", i(&|r| r.gw_index))
        .with("player_key", i(&|r| r.player_key))
        .with("element_type", i(&|r| r.element_type))
        .with("horizon", i(&|r| r.horizon))
        .with("target_gw", i(&|r| r.target_gw))
        .with("target_gw_index", i(&|r| r.target_gw_index))
        .with("xp", Column::F64(p.iter().map(|r| r.xp).collect()))
        .with("candidate", Column::Bool(p.iter().map(|r| r.candidate).collect()))
        .with("own_candidate", Column::Bool(p.iter().map(|r| r.own_candidate).collect()));
    for (k, name) in REALIZED.iter().enumerate() {
        t.push(name, Column::F64(p.iter().map(|r| r.realized[k]).collect()));
    }
    let mut names: Vec<String> = Vec::new();
    for e in &result.extras {
        for n in e {
            if !names.contains(n) {
                names.push(n.clone());
            }
        }
    }
    for name in &names {
        let positions: Vec<Option<usize>> =
            result.extras.iter().map(|e| e.iter().position(|n| n == name)).collect();
        t.push(
            name,
            Column::F64(
                p.iter().map(|r| positions[r.model].map_or(f64::NAN, |k| r.extras[k])).collect(),
            ),
        );
    }
    t
}

fn regrets_table(result: &EvalResult) -> Table {
    let r = &result.regrets;
    let i =
        |f: &dyn Fn(&fplopt_backtest::eval::Regret) -> i64| Column::I64(r.iter().map(f).collect());
    Table::new()
        .with("model", Column::Str(r.iter().map(|x| result.models[x.model].clone()).collect()))
        .with("season", i(&|x| x.season))
        .with("deadline", Column::Ts(r.iter().map(|x| x.deadline).collect()))
        .with("gw", i(&|x| x.gw))
        .with("gw_index", i(&|x| x.gw_index))
        .with("squad", Column::Str(r.iter().map(|x| x.squad.clone()).collect()))
        .with("points", i(&|x| x.points))
        .with("best_points", i(&|x| x.best_points))
        .with("xi_regret", i(&|x| x.xi_regret))
        .with("captain_points", i(&|x| x.captain_points))
        .with("best_captain_points", i(&|x| x.best_captain_points))
        .with("captain_regret", i(&|x| x.captain_regret))
}

fn run_eval(args: &EvalArgs) -> Result<()> {
    let models: Vec<String> = args.models.split(',').map(|m| m.trim().to_owned()).collect();
    let unknown: Vec<&String> =
        models.iter().filter(|m| !XP_MODELS.contains(&m.as_str())).collect();
    if models.is_empty() || !unknown.is_empty() {
        let names: Vec<&str> = unknown.iter().map(|s| s.as_str()).collect();
        let shown = if names.is_empty() { format!("'{}'", args.models) } else { names.join(", ") };
        return usage(format!(
            "argument --models: unknown xP model(s) {shown} (models: {})",
            XP_MODELS.join(", ")
        ));
    }
    if models.iter().collect::<std::collections::BTreeSet<_>>().len() != models.len() {
        return usage(format!("argument --models: repeated xP model in '{}'", args.models));
    }
    let seasons = parse_seasons(&args.seasons, "--seasons")?;
    if args.jobs < 1 {
        return usage("--jobs must be >= 1");
    }
    if args.n_random < 0 {
        return usage("--n-random must be >= 0");
    }
    let late = seasons.iter().max().is_some_and(|s| *s < EP_NEXT_FIRST_SEASON);
    let early: Vec<&str> =
        models.iter().map(String::as_str).filter(|m| late && m.starts_with("ep_next")).collect();
    if !early.is_empty() {
        return usage(format!(
            "{} use(s) ep_next, which exists only from {}: add a season from then on",
            early.join(", "),
            season_label(EP_NEXT_FIRST_SEASON as i32)
        ));
    }
    let family = args
        .family
        .clone()
        .unwrap_or_else(|| format!("eval {} {}", models.join(","), format_seasons(&seasons)));
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
    let out = args.out.clone().unwrap_or_else(|| {
        PathBuf::from("results")
            .join(format!("{}-eval", chrono::Utc::now().format("%Y%m%dT%H%M%SZ")))
    });
    let began = Instant::now();
    let result = evaluate(&store, &models, &seasons, args.jobs, args.n_random as usize, args.seed)?;
    let runtime = began.elapsed().as_secs_f64();
    let int = |i: i64| Ordered::Int(i128::from(i));
    let config = record(vec![
        ("models", Ordered::Array(models.iter().map(|m| Ordered::Str(m.clone())).collect())),
        ("seasons", Ordered::Array(seasons.iter().map(|s| int(*s)).collect())),
        ("n_random", int(args.n_random)),
        ("seed", int(args.seed as i64)),
        ("out", Ordered::Str(out.display().to_string())),
        ("jobs", int(args.jobs as i64)),
        ("family", Ordered::Str(family.clone())),
    ]);
    let n_variants = family_variants(&args.experiments, &family)? + 1;
    std::fs::create_dir_all(&out)?;
    write_parquet(&predictions_table(&result), &out.join("predictions.parquet"))?;
    write_parquet(&regrets_table(&result), &out.join("regrets.parquet"))?;
    let command = command_line();
    let payload = record(vec![
        ("command", Ordered::Str(command.clone())),
        ("config", config.clone()),
        ("family", Ordered::Str(family.clone())),
        ("n_variants", int(n_variants as i64)),
        ("runtime_seconds", Ordered::Float((runtime * 10.0).round() / 10.0)),
        (
            "season_seconds",
            Ordered::Object(
                result.timings.iter().map(|(s, t)| (s.clone(), Ordered::Float(*t))).collect(),
            ),
        ),
        ("metrics", result.metrics.clone()),
    ]);
    write_json(&out.join("metrics.json"), &payload)?;
    let seasons_text: Vec<String> = models
        .iter()
        .map(|m| {
            let s: Vec<i64> = seasons
                .iter()
                .copied()
                .filter(|s| *s >= fplopt_backtest::eval::first_season(m))
                .collect();
            format!("{m} {}", format_seasons(&s))
        })
        .collect();
    println!(
        "models eval: {}; {} random squads + template per deadline ({runtime:.0}s)\n\nxP per player-GW (MSE; MAE is a \
         diagnostic) and decision regret per squad-GW\n{}\n\nDiebold-Mariano, one-sided, clustered by GW (common \
         sample of each pair; losses: squared error, regrets)\n{}\n\nWritten to {}",
        seasons_text.join(", "),
        args.n_random,
        summary_table(&result.metrics, &models),
        dm_table(&result.metrics),
        out.display()
    );
    let logged: Vec<(String, Ordered)> = ["seasons", "xp", "regret", "dm"]
        .iter()
        .filter_map(|k| get(&result.metrics, k).map(|v| (k.to_string(), v.clone())))
        .collect();
    log_experiment(
        &args.experiments,
        &command,
        &config.to_value(),
        &Ordered::Object(logged).to_value(),
        n_variants,
        &family,
    )
}
