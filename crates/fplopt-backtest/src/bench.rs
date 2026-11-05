//! Optimizer benchmark on real data (Python `fplopt.optimize.bench`; Phase 4 Task 3, issue
//! #10): solve times, gaps, the chip scenario search and the pruning defaults. Cases are
//! sampled (season, gw_index) deadlines × a template and a seeded random start × the xP
//! models; per case the default no-chip solve, top-3 with the roll plan, the bound chip
//! search (and on the first `all_chips` cases the exhaustive one, and the chip search at a
//! tight gap with the default pool vs dominance only), the fixed bench weights for frames
//! with `p_play`, and the pruning variants at a tight gap (with the expected-minutes floors
//! for frames with `e_minutes`). Rows are ordered key/value lists, written as Python's CSV.

use std::collections::BTreeMap;
use std::time::Instant;

use anyhow::{Result, bail};
use fplopt_core::nprng::Generator;
use fplopt_core::pyjson::{Ordered, py_float_repr};
use fplopt_core::seasons::{HOLDOUT_SEASONS, season_label};
use fplopt_optimize::start::{FAR_FUTURE, MAX_BANK, StartStateError, random_state, template_state};
use fplopt_optimize::{
    ChipScenario, OptimizerParams, PlanInput, PlanSet, Search, optimize, solve_plan,
};
use fplopt_store::{DataStore, Micros};

use crate::evaluate::rules_for;
use crate::simulator::{Caches, season_schedule};

pub const BENCH_SEASONS: [i64; 5] = [2021, 2022, 2023, 2024, 2026];
pub const XP_MODELS: [&str; 2] = ["ep_next", "rolling"];
pub const PRUNE_GAP: f64 = 1e-4;
pub const FLOORS: [f64; 6] = [0.0, 30.0, 60.0, 90.0, 135.0, 180.0];

/// A CSV / JSON cell.
#[derive(Clone, Debug, PartialEq)]
pub enum Cell {
    I(i64),
    F(f64),
    S(String),
    B(bool),
    None,
}

impl Cell {
    /// pandas' `to_csv` text.
    pub fn csv(&self) -> String {
        match self {
            Cell::I(i) => i.to_string(),
            Cell::F(f) if f.is_nan() => String::new(),
            Cell::F(f) => py_float_repr(*f),
            Cell::S(s) => s.clone(),
            Cell::B(b) => {
                if *b {
                    "True".into()
                } else {
                    "False".into()
                }
            }
            Cell::None => String::new(),
        }
    }

    pub fn f(&self) -> f64 {
        match self {
            Cell::I(i) => *i as f64,
            Cell::F(f) => *f,
            _ => f64::NAN,
        }
    }
}

pub type Row = Vec<(String, Cell)>;

fn set(row: &mut Row, key: &str, value: Cell) {
    match row.iter_mut().find(|(k, _)| k == key) {
        Some(e) => e.1 = value,
        None => row.push((key.to_owned(), value)),
    }
}

pub fn get<'a>(row: &'a Row, key: &str) -> Option<&'a Cell> {
    row.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

/// A pruning variant's overrides of `OptimizerParams`.
#[derive(Clone, Debug, PartialEq)]
pub struct Variant {
    pub name: String,
    pub prune_n: Option<Option<BTreeMap<i64, i64>>>,
    pub prune_dominated: Option<bool>,
    pub min_minutes: Option<f64>,
}

impl Variant {
    fn apply(&self, p: &OptimizerParams) -> OptimizerParams {
        let mut q = p.clone();
        if let Some(n) = &self.prune_n {
            q.prune_n.clone_from(n);
        }
        if let Some(d) = self.prune_dominated {
            q.prune_dominated = d;
        }
        if let Some(m) = self.min_minutes {
            q.min_minutes = m;
        }
        q
    }

    pub fn to_json(&self) -> Ordered {
        let mut pairs = Vec::new();
        if let Some(n) = &self.prune_n {
            let v = match n {
                None => Ordered::Null,
                Some(m) => Ordered::Object(
                    m.iter().map(|(k, v)| (k.to_string(), Ordered::Int(i128::from(*v)))).collect(),
                ),
            };
            pairs.push(("prune_n".to_owned(), v));
        }
        if let Some(d) = self.prune_dominated {
            pairs.push(("prune_dominated".to_owned(), Ordered::Bool(d)));
        }
        if let Some(m) = self.min_minutes {
            pairs.push(("min_minutes".to_owned(), Ordered::Float(m)));
        }
        Ordered::Object(pairs)
    }
}

fn variant(
    name: &str,
    prune_n: Option<Option<&[(i64, i64)]>>,
    dominated: Option<bool>,
    minutes: Option<f64>,
) -> Variant {
    Variant {
        name: name.to_owned(),
        prune_n: prune_n.map(|n| n.map(|v| v.iter().copied().collect())),
        prune_dominated: dominated,
        min_minutes: minutes,
    }
}

/// `PRUNE_VARIANTS`, plus `floor_<m>` for frames with e_minutes when `floors` is given.
pub fn prune_variants(floors: Option<&[f64]>) -> Vec<Variant> {
    let mut out = vec![
        variant("none", Some(None), Some(false), Some(0.0)),
        variant("dominance", Some(None), None, Some(0.0)),
        variant("default", None, None, None),
        variant("n10_30", Some(Some(&[(1, 10), (2, 30), (3, 30), (4, 15)])), None, None),
        variant("n5_15", Some(Some(&[(1, 5), (2, 15), (3, 15), (4, 8)])), None, None),
    ];
    if let Some(floors) = floors {
        let mut fs: Vec<f64> = floors.to_vec();
        fs.push(0.0);
        fs.sort_by(f64::total_cmp);
        fs.dedup();
        for f in fs {
            // Python's f"{floor:.17g}"
            let name = if f == f.trunc() && f.abs() < 1e17 {
                format!("{}", f as i64)
            } else {
                format!("{f}")
            };
            out.push(variant(&format!("floor_{name}"), None, None, Some(f)));
        }
    }
    out
}

/// One benchmark instance.
#[derive(Clone, Debug)]
pub struct BenchCase {
    pub season: i64,
    pub gw_index: i64,
    pub start: String,
    pub xp: String,
}

impl BenchCase {
    pub fn name(&self) -> String {
        format!(
            "{} gw{} {} {}",
            season_label(self.season as i32),
            self.gw_index,
            self.start,
            self.xp
        )
    }
}

/// `n` distinct (season, gw_index) pairs, round-robin over the seasons, gw_index drawn
/// uniformly (numpy's draws) from the season's GWs whose deadline has passed. Sorted.
pub fn bench_deadlines(
    store: &DataStore,
    n: usize,
    seed: u64,
    seasons: &[i64],
    now: Micros,
) -> Result<Vec<(i64, i64)>> {
    if n < 1 {
        bail!("need at least one deadline, got {n}");
    }
    let t = store
        .as_of(FAR_FUTURE)
        .table("gameweek", Some(&["season", "gw_index", "deadline_time"]))?;
    let (s, idx, d) = (t.opt_i64("season")?, t.opt_i64("gw_index")?, t.opt_i64("deadline_time")?);
    let mut wanted: Vec<i64> =
        seasons.iter().copied().filter(|x| !HOLDOUT_SEASONS.contains(&(*x as i32))).collect();
    wanted.sort_unstable();
    wanted.dedup();
    let mut pools: Vec<(i64, Vec<i64>)> = Vec::new();
    for season in &wanted {
        let mut gws: Vec<i64> = (0..t.height())
            .filter(|&i| s[i] == Some(*season) && d[i].is_some_and(|x| x < now))
            .filter_map(|i| idx[i])
            .collect();
        gws.sort_unstable();
        gws.dedup();
        if !gws.is_empty() {
            pools.push((*season, gws));
        }
    }
    if pools.is_empty() {
        bail!("no gameweeks for seasons {wanted:?}");
    }
    let mut rng = Generator::new(u128::from(seed));
    let mut picked = Vec::new();
    while picked.len() < n && pools.iter().any(|p| !p.1.is_empty()) {
        for (season, gws) in pools.iter_mut() {
            if picked.len() == n {
                break;
            }
            if !gws.is_empty() {
                let k = rng.integer(0, gws.len() as i64) as usize;
                picked.push((*season, gws.remove(k)));
            }
        }
    }
    picked.sort_unstable();
    Ok(picked)
}

/// Template and random:<seed + i> starts at each sampled deadline × each xP model.
pub fn bench_cases(
    store: &DataStore,
    deadlines: usize,
    seed: u64,
    seasons: &[i64],
    xp_models: &[String],
    now: Micros,
) -> Result<Vec<BenchCase>> {
    let mut out = Vec::new();
    for (i, (season, gw_index)) in
        bench_deadlines(store, deadlines, seed, seasons, now)?.into_iter().enumerate()
    {
        for start in ["template".to_owned(), format!("random:{}", seed + i as u64)] {
            for xp in xp_models {
                out.push(BenchCase { season, gw_index, start: start.clone(), xp: xp.clone() });
            }
        }
    }
    Ok(out)
}

fn timed<T>(f: impl FnOnce() -> Result<T>) -> Result<(T, f64)> {
    let start = Instant::now();
    let out = f()?;
    Ok((out, start.elapsed().as_secs_f64()))
}

fn same_plan(bound: &PlanSet, full: &PlanSet) -> Cell {
    if bound.solver_status() != "Optimal" || full.solver_status() != "Optimal" {
        return Cell::None;
    }
    Cell::B(
        full.best().scenario == bound.best().scenario
            && full.best().total_objective() == bound.best().total_objective(),
    )
}

fn solve(problem: &PlanInput, params: &OptimizerParams) -> Result<fplopt_optimize::Plan> {
    solve_plan(problem, params, None, &[], &ChipScenario::none())
        .map_err(|e| anyhow::anyhow!("{e}"))
}

/// The case's row of measurements and its pruning rows.
pub fn bench_case(
    store: &DataStore,
    caches: &Caches,
    case: &BenchCase,
    params: &OptimizerParams,
    chips_all: bool,
    prune_study: bool,
    floors: &[f64],
) -> Result<(Row, Vec<Row>)> {
    let rules = rules_for(case.season)?;
    let deadline = season_schedule(store, case.season, case.gw_index)?[0].deadline;
    let view = store.as_of(deadline);
    let state = match case.start.strip_prefix("random:") {
        Some(seed) => random_state(&view, &rules, seed.parse()?, Some(MAX_BANK))?,
        None => template_state(&view, &rules)?,
    };
    let pool = caches.pool(&view)?;
    let xp = caches.xp(&case.xp, &view)?;
    let frame = &xp.frame;
    let mut row: Row = vec![
        ("case".into(), Cell::S(case.name())),
        ("season".into(), Cell::I(case.season)),
        ("gw_index".into(), Cell::I(case.gw_index)),
        ("start".into(), Cell::S(case.start.clone())),
        ("xp".into(), Cell::S(case.xp.clone())),
    ];
    let (problem, input_s) =
        timed(|| PlanInput::from_context(&state, &pool, frame, &rules, params))?;
    let plan = solve(&problem, params)?;
    for (k, v) in [
        ("horizon", Cell::I(problem.gws.len() as i64)),
        ("n_pool", Cell::I(problem.n_pool as i64)),
        ("n_candidates", Cell::I(problem.players.len() as i64)),
        ("input_s", Cell::F(input_s)),
        ("build_s", Cell::F(plan.build_seconds)),
        ("solve_s", Cell::F(plan.solve_seconds)),
        ("gap", Cell::F(plan.mip_gap)),
        ("status", Cell::S(plan.status.clone())),
        ("nodes", Cell::I(plan.n_nodes)),
        ("objective", Cell::F(plan.total_objective())),
        ("minutes_bench", Cell::B(problem.bench_weights.is_some())),
    ] {
        set(&mut row, k, v);
    }
    if problem.bench_weights.is_some() {
        let fixed = OptimizerParams { bench_from_minutes: false, ..params.clone() };
        let p = solve(&PlanInput::from_context(&state, &pool, frame, &rules, &fixed)?, &fixed)?;
        set(&mut row, "fixed_bench_solve_s", Cell::F(p.solve_seconds));
        set(&mut row, "fixed_bench_objective", Cell::F(p.total_objective()));
        set(&mut row, "fixed_bench_status", Cell::S(p.status));
    }
    let top3 = optimize(&problem, params, 3, false, true, Search::Bound)?;
    set(&mut row, "top3_s", Cell::F(top3.seconds));
    set(&mut row, "top3_plans", Cell::I(top3.plans.len() as i64));
    let chips = optimize(&problem, params, 1, true, false, Search::Bound)?;
    let stats = &chips.search;
    for (k, v) in [
        ("chips_s", Cell::F(chips.seconds)),
        ("n_scenarios", Cell::I(stats.n_scenarios as i64)),
        ("n_relaxations", Cell::I(stats.n_relaxations as i64)),
        ("n_solves", Cell::I(stats.n_solves as i64)),
        ("relax_s", Cell::F(stats.relax_seconds)),
        ("chip_solve_s", Cell::F(stats.solve_seconds)),
        ("chips_scenario", Cell::S(chips.best().scenario.clone())),
        ("chips_objective", Cell::F(chips.best().total_objective())),
    ] {
        set(&mut row, k, v);
    }
    if chips_all {
        let full = optimize(&problem, params, 1, true, false, Search::All)?;
        set(&mut row, "chips_all_s", Cell::F(full.seconds));
        set(&mut row, "chips_all_scenario", Cell::S(full.best().scenario.clone()));
        set(&mut row, "chips_all_objective", Cell::F(full.best().total_objective()));
        set(&mut row, "bound_matches_all", same_plan(&chips, &full));
        let variants = prune_variants(None);
        for name in ["default", "dominance"] {
            let v = variants.iter().find(|v| v.name == name).unwrap();
            let p = OptimizerParams { mip_gap: PRUNE_GAP, ..v.apply(params) };
            let prob = PlanInput::from_context(&state, &pool, frame, &rules, &p)?;
            let res = optimize(&prob, &p, 1, true, false, Search::Bound)?;
            set(&mut row, &format!("chips_tight_{name}_s"), Cell::F(res.seconds));
            set(
                &mut row,
                &format!("chips_tight_{name}_objective"),
                Cell::F(res.best().total_objective()),
            );
        }
        let loss = get(&row, "chips_tight_dominance_objective").unwrap().f()
            - get(&row, "chips_tight_default_objective").unwrap().f();
        set(&mut row, "chips_prune_loss", Cell::F(loss));
    }
    let mut prune_rows: Vec<Row> = Vec::new();
    if prune_study {
        let variants = prune_variants(xp.table.has("e_minutes").then_some(floors));
        let mut solved: Vec<(OptimizerParams, Row)> = Vec::new();
        for v in &variants {
            let p = OptimizerParams { mip_gap: PRUNE_GAP, ..v.apply(params) };
            if let Some((_, r)) = solved.iter().find(|(q, _)| *q == p) {
                let mut r = r.clone();
                set(&mut r, "variant", Cell::S(v.name.clone()));
                prune_rows.push(r);
                continue;
            }
            let (prob, input_s) =
                timed(|| PlanInput::from_context(&state, &pool, frame, &rules, &p))?;
            let sol = solve(&prob, &p)?;
            let measured: Row = vec![
                ("case".into(), Cell::S(case.name())),
                ("variant".into(), Cell::S(v.name.clone())),
                ("n_candidates".into(), Cell::I(prob.players.len() as i64)),
                ("input_s".into(), Cell::F(input_s)),
                ("build_s".into(), Cell::F(sol.build_seconds)),
                ("solve_s".into(), Cell::F(sol.solve_seconds)),
                ("gap".into(), Cell::F(sol.mip_gap)),
                ("status".into(), Cell::S(sol.status.clone())),
                ("objective".into(), Cell::F(sol.total_objective())),
                ("bound".into(), Cell::F(sol.bound)),
            ];
            solved.push((p, measured.clone()));
            prune_rows.push(measured);
        }
        let best = prune_rows
            .iter()
            .map(|r| get(r, "objective").unwrap().f())
            .fold(f64::NEG_INFINITY, f64::max);
        let no_floor = prune_rows
            .iter()
            .find(|r| matches!(get(r, "variant"), Some(Cell::S(v)) if v == "floor_0"))
            .map(|r| get(r, "objective").unwrap().f());
        for r in &mut prune_rows {
            let objective = get(r, "objective").unwrap().f();
            set(r, "loss", Cell::F(best - objective));
            let is_floor = matches!(get(r, "variant"), Some(Cell::S(v)) if v.starts_with("floor_"));
            if let (Some(nf), true) = (no_floor, is_floor) {
                set(r, "floor_loss", Cell::F(nf - objective));
            }
        }
    }
    Ok((row, prune_rows))
}

/// The benchmark's case rows, pruning rows and config.
pub struct BenchResult {
    pub cases: Vec<Row>,
    pub prune: Vec<Row>,
    pub config: Ordered,
}

#[allow(clippy::too_many_arguments)]
pub fn run_bench(
    store: &DataStore,
    deadlines: usize,
    seed: u64,
    params: &OptimizerParams,
    all_chips: usize,
    prune_study: bool,
    seasons: &[i64],
    xp_models: &[String],
    floors: &[f64],
    now: Micros,
) -> Result<BenchResult> {
    let mut seasons = seasons.to_vec();
    seasons.sort_unstable();
    seasons.dedup();
    let caches = Caches::default();
    let cases = bench_cases(store, deadlines, seed, &seasons, xp_models, now)?;
    let (mut rows, mut prune) = (Vec::new(), Vec::new());
    for (k, case) in cases.iter().enumerate() {
        match bench_case(store, &caches, case, params, rows.len() < all_chips, prune_study, floors)
        {
            Ok((row, prows)) => {
                let f = |key: &str| get(&row, key).map_or(f64::NAN, Cell::f);
                let mut line = format!(
                    "[{}/{}] {}: no-chip {:.2}s, chips {:.1}s ({}/{} solved), top-3 {:.1}s",
                    k + 1,
                    cases.len(),
                    case.name(),
                    f("solve_s"),
                    f("chips_s"),
                    f("n_solves"),
                    f("n_scenarios"),
                    f("top3_s")
                );
                if let Some(all) = get(&row, "chips_all_s") {
                    let same = match get(&row, "bound_matches_all") {
                        Some(Cell::B(b)) => {
                            if *b {
                                "True"
                            } else {
                                "False"
                            }
                        }
                        _ => "None",
                    };
                    line += &format!("; exhaustive {:.0}s, same plan: {same}", all.f());
                }
                println!("{line}");
                rows.push(row);
                prune.extend(prows);
            }
            Err(e) if e.downcast_ref::<StartStateError>().is_some() => {
                println!("[{}/{}] {}: skipped ({e})", k + 1, cases.len(), case.name());
            }
            Err(e) => return Err(e),
        }
    }
    if rows.is_empty() {
        bail!("no benchmark case could be built");
    }
    let int = |i: i64| Ordered::Int(i128::from(i));
    let config = Ordered::Object(vec![
        ("deadlines".into(), int(deadlines as i64)),
        ("seed".into(), int(seed as i64)),
        ("all_chips".into(), int(all_chips as i64)),
        ("prune_study".into(), Ordered::Bool(prune_study)),
        ("prune_gap".into(), Ordered::Float(PRUNE_GAP)),
        ("seasons".into(), Ordered::Array(seasons.iter().map(|s| int(*s)).collect())),
        (
            "xp_models".into(),
            Ordered::Array(xp_models.iter().map(|m| Ordered::Str(m.clone())).collect()),
        ),
        ("floors".into(), Ordered::Array(floors.iter().map(|f| Ordered::Float(*f)).collect())),
        (
            "params".into(),
            Ordered::Object(vec![
                ("horizon".into(), int(params.horizon)),
                ("decay".into(), Ordered::Float(params.decay)),
                ("mip_gap".into(), Ordered::Float(params.mip_gap)),
                ("threads".into(), int(params.threads)),
                (
                    "prune_n".into(),
                    params.prune_n.as_ref().map_or(Ordered::Null, |m| {
                        Ordered::Object(m.iter().map(|(k, v)| (k.to_string(), int(*v))).collect())
                    }),
                ),
                ("prune_dominated".into(), Ordered::Bool(params.prune_dominated)),
                ("bench_from_minutes".into(), Ordered::Bool(params.bench_from_minutes)),
                ("min_minutes".into(), Ordered::Float(params.min_minutes)),
            ]),
        ),
        (
            "prune_variants".into(),
            Ordered::Object(
                prune_variants(Some(floors))
                    .into_iter()
                    .map(|v| (v.name.clone(), v.to_json()))
                    .collect(),
            ),
        ),
    ]);
    Ok(BenchResult { cases: rows, prune, config })
}

/// pandas' CSV of rows with the union of their keys (first-seen order).
pub fn to_csv(rows: &[Row]) -> String {
    let mut keys: Vec<String> = Vec::new();
    for r in rows {
        for (k, _) in r {
            if !keys.contains(k) {
                keys.push(k.clone());
            }
        }
    }
    let quote = |s: String| {
        if s.contains(',') || s.contains('"') || s.contains('\n') {
            format!("\"{}\"", s.replace('"', "\"\""))
        } else {
            s
        }
    };
    let mut out = keys.iter().map(|k| quote(k.clone())).collect::<Vec<_>>().join(",") + "\n";
    for r in rows {
        let line: Vec<String> =
            keys.iter().map(|k| quote(get(r, k).map_or(String::new(), Cell::csv))).collect();
        out += &(line.join(",") + "\n");
    }
    out
}

fn stats(values: &[f64]) -> Ordered {
    let mut v: Vec<f64> = values.iter().copied().filter(|x| !x.is_nan()).collect();
    if v.is_empty() {
        return Ordered::Object(vec![
            ("median".into(), Ordered::Null),
            ("p90".into(), Ordered::Null),
            ("max".into(), Ordered::Null),
        ]);
    }
    v.sort_by(f64::total_cmp);
    let q = |p: f64| crate::stats::np_quantile(&v, p);
    Ordered::Object(vec![
        ("median".into(), Ordered::Float(q(0.5))),
        ("p90".into(), Ordered::Float(q(0.9))),
        ("max".into(), Ordered::Float(v[v.len() - 1])),
    ])
}

fn column(rows: &[Row], key: &str) -> Vec<f64> {
    rows.iter().map(|r| get(r, key).map_or(f64::NAN, Cell::f)).collect()
}

/// Timing stats per measurement, the bound-vs-all check and per pruning variant.
pub fn summarize(result: &BenchResult) -> Ordered {
    let cases = &result.cases;
    let mut timings: Vec<(String, Ordered)> = [
        "input_s",
        "build_s",
        "solve_s",
        "gap",
        "top3_s",
        "chips_s",
        "relax_s",
        "chip_solve_s",
        "n_scenarios",
        "n_relaxations",
        "n_solves",
        "n_candidates",
    ]
    .iter()
    .map(|k| (k.to_string(), stats(&column(cases, k))))
    .collect();
    if cases.iter().any(|r| get(r, "fixed_bench_solve_s").is_some()) {
        timings.push(("fixed_bench_solve_s".into(), stats(&column(cases, "fixed_bench_solve_s"))));
    }
    let mut out = vec![
        ("n_cases".to_owned(), Ordered::Int(cases.len() as i128)),
        ("timings".to_owned(), Ordered::Object(timings)),
    ];
    if cases.iter().any(|r| get(r, "chips_all_s").is_some()) {
        let done: Vec<Row> =
            cases.iter().filter(|r| get(r, "chips_all_s").is_some()).cloned().collect();
        let matches =
            done.iter().filter(|r| get(r, "bound_matches_all") == Some(&Cell::B(true))).count();
        let undetermined = done
            .iter()
            .filter(|r| matches!(get(r, "bound_matches_all"), Some(Cell::None) | None))
            .count();
        out.push((
            "chips_all".into(),
            Ordered::Object(vec![
                ("n_cases".into(), Ordered::Int(done.len() as i128)),
                ("matches".into(), Ordered::Int(matches as i128)),
                ("undetermined".into(), Ordered::Int(undetermined as i128)),
                ("all_s".into(), stats(&column(&done, "chips_all_s"))),
                ("bound_s".into(), stats(&column(&done, "chips_s"))),
                ("prune_loss".into(), stats(&column(&done, "chips_prune_loss"))),
            ]),
        ));
    }
    if !result.prune.is_empty() {
        let mut names: Vec<String> = Vec::new();
        for r in &result.prune {
            if let Some(Cell::S(v)) = get(r, "variant")
                && !names.contains(v)
            {
                names.push(v.clone());
            }
        }
        let per = names
            .iter()
            .map(|name| {
                let group: Vec<Row> = result
                    .prune
                    .iter()
                    .filter(|r| get(r, "variant") == Some(&Cell::S(name.clone())))
                    .cloned()
                    .collect();
                let loss = column(&group, "loss");
                let mut pairs = vec![
                    ("n_candidates".to_owned(), stats(&column(&group, "n_candidates"))),
                    ("solve_s".to_owned(), stats(&column(&group, "solve_s"))),
                    ("loss".to_owned(), stats(&loss)),
                    (
                        "n_loss_over_0.1".to_owned(),
                        Ordered::Int(loss.iter().filter(|x| **x > 0.1).count() as i128),
                    ),
                ];
                let floor = column(&group, "floor_loss");
                if floor.iter().any(|x| !x.is_nan()) {
                    pairs.push(("floor_loss".into(), stats(&floor)));
                    pairs.push((
                        "n_floor_loss_over_0.1".into(),
                        Ordered::Int(floor.iter().filter(|x| **x > 0.1).count() as i128),
                    ));
                    pairs.push(("n_cases".into(), Ordered::Int(group.len() as i128)));
                }
                (name.clone(), Ordered::Object(pairs))
            })
            .collect();
        out.push(("prune".into(), Ordered::Object(per)));
    }
    Ordered::Object(out)
}

fn field<'a>(o: &'a Ordered, key: &str) -> Option<&'a Ordered> {
    match o {
        Ordered::Object(p) => p.iter().find(|(k, _)| k == key).map(|(_, v)| v),
        _ => None,
    }
}

fn num(o: Option<&Ordered>) -> f64 {
    match o {
        Some(Ordered::Float(f)) => *f,
        Some(Ordered::Int(i)) => *i as f64,
        _ => f64::NAN,
    }
}

/// The printed summary: timing table, bound-vs-all line and pruning tables.
pub fn summary_text(summary: &Ordered) -> String {
    let fmt = |x: f64, d: usize| if x.is_nan() { "-".to_owned() } else { format!("{x:.d$}") };
    let timings = field(summary, "timings").unwrap();
    let mut rows: Vec<(&str, &str, usize)> = vec![
        ("input build (s)", "input_s", 2),
        ("model build (s)", "build_s", 2),
        ("no-chip solve (s)", "solve_s", 2),
        ("no-chip final gap", "gap", 4),
        ("top-3 + roll (s)", "top3_s", 1),
        ("chips, bound search (s)", "chips_s", 1),
        ("  LP relaxations (s)", "relax_s", 1),
        ("  scenario MILPs (s)", "chip_solve_s", 1),
        ("  scenarios", "n_scenarios", 0),
        ("  LP relaxations", "n_relaxations", 0),
        ("  MILPs solved", "n_solves", 0),
        ("candidates", "n_candidates", 0),
    ];
    if field(timings, "fixed_bench_solve_s").is_some() {
        rows.insert(3, ("  fixed bench weights (s)", "fixed_bench_solve_s", 2));
    }
    let width = rows.iter().map(|r| r.0.len()).max().unwrap();
    let mut lines = vec![format!("{} cases (median / p90 / max)", num(field(summary, "n_cases")))];
    lines[0] = format!("{} cases (median / p90 / max)", num(field(summary, "n_cases")) as i64);
    lines.push(format!("{:width$}  {:>8} {:>8} {:>8}", "", "median", "p90", "max"));
    for (label, key, d) in &rows {
        let s = field(timings, key).unwrap();
        let cells: Vec<String> = ["median", "p90", "max"]
            .iter()
            .map(|k| format!("{:>8}", fmt(num(field(s, k)), *d)))
            .collect();
        lines.push(format!("{label:width$}  {}", cells.join(" ")));
    }
    if let Some(c) = field(summary, "chips_all") {
        let undetermined = num(field(c, "undetermined")) as i64;
        let all = field(c, "all_s").unwrap();
        let loss = field(c, "prune_loss").unwrap();
        lines.push(format!(
            "\nExhaustive chip search on {} cases: median {}s (max {}s) vs bound search {}s; same best plan in {}/{}{}; \
             with chips at gap 0.0001, default pruning loses {} (max {}) vs dominance pruning only",
            num(field(c, "n_cases")) as i64,
            fmt(num(field(all, "median")), 1),
            fmt(num(field(all, "max")), 1),
            fmt(num(field(field(c, "bound_s").unwrap(), "median")), 1),
            num(field(c, "matches")) as i64,
            num(field(c, "n_cases")) as i64,
            if undetermined > 0 { format!(" ({undetermined} undetermined: a solve hit the time limit)") } else { String::new() },
            fmt(num(field(loss, "median")), 3),
            fmt(num(field(loss, "max")), 3)
        ));
    }
    if let Some(Ordered::Object(prune)) = field(summary, "prune") {
        lines.push("\nPruning (no chips, gap 0.0001; loss = best variant's objective - this one's, points)".into());
        let header =
            ["variant", "cands med", "solve med", "solve max", "loss med", "loss max", ">0.1"];
        lines.push(format!(
            "{:<14}{}",
            header[0],
            header[1..].iter().map(|h| format!("{h:>11}")).collect::<String>()
        ));
        for (name, s) in prune {
            let st = |k: &str, q: &str| num(field(field(s, k).unwrap(), q));
            let cells = [
                fmt(st("n_candidates", "median"), 0),
                fmt(st("solve_s", "median"), 2),
                fmt(st("solve_s", "max"), 2),
                fmt(st("loss", "median"), 3),
                fmt(st("loss", "max"), 3),
                (num(field(s, "n_loss_over_0.1")) as i64).to_string(),
            ];
            lines.push(format!(
                "{name:<14}{}",
                cells.iter().map(|c| format!("{c:>11}")).collect::<String>()
            ));
        }
        let floors: Vec<&(String, Ordered)> =
            prune.iter().filter(|(_, s)| field(s, "floor_loss").is_some()).collect();
        if !floors.is_empty() {
            lines.push(
                "\nMinutes floors (loss vs floor 0 = objective(floor 0) - objective(floor), points; frames with e_minutes)".into(),
            );
            let header =
                ["variant", "cases", "cands med", "solve med", "loss med", "loss max", ">0.1"];
            lines.push(format!(
                "{:<14}{}",
                header[0],
                header[1..].iter().map(|h| format!("{h:>11}")).collect::<String>()
            ));
            for (name, s) in floors {
                let st = |k: &str, q: &str| num(field(field(s, k).unwrap(), q));
                let cells = [
                    (num(field(s, "n_cases")) as i64).to_string(),
                    fmt(st("n_candidates", "median"), 0),
                    fmt(st("solve_s", "median"), 2),
                    fmt(st("floor_loss", "median"), 3),
                    fmt(st("floor_loss", "max"), 3),
                    (num(field(s, "n_floor_loss_over_0.1")) as i64).to_string(),
                ];
                lines.push(format!(
                    "{name:<14}{}",
                    cells.iter().map(|c| format!("{c:>11}")).collect::<String>()
                ));
            }
        }
    }
    lines.join("\n")
}
