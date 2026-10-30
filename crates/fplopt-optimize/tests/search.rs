//! Chip scenario search by bounds, the LP-relaxation bound and the MILP dual bound (Python
//! `tests/test_optimize_search.py`).

mod common;

use std::collections::{BTreeMap, HashMap};

use common::*;
use fplopt_optimize::chips::make_scenario;
use fplopt_optimize::model::relaxation_bound;
use fplopt_optimize::search::{chip_extra, search};
use fplopt_optimize::state::{PoolPlayer, SquadState, TRANSFER_CHIPS};
use fplopt_optimize::{
    OptimizerParams, PlanInput, Search, XpFrame, optimize, scenarios, solve_plan, terminal_value,
};

/// Small candidate pools keep the exhaustive searches fast; the properties tested don't
/// depend on the pool.
fn small() -> BTreeMap<i64, i64> {
    BTreeMap::from([(1, 3), (2, 6), (3, 6), (4, 4)])
}

fn exact() -> OptimizerParams {
    OptimizerParams { mip_gap: 0.0, prune_n: Some(small()), ..OptimizerParams::default() }
}

/// Terminal values that make chips worth playing inside a short horizon.
fn cheap_chips() -> BTreeMap<String, f64> {
    chip_values(0.5, 0.5, 0.5, 0.5)
}

fn chip_values(wildcard: f64, freehit: f64, bboost: f64, tc: f64) -> BTreeMap<String, f64> {
    BTreeMap::from([
        ("wildcard".to_string(), wildcard),
        ("freehit".to_string(), freehit),
        ("bboost".to_string(), bboost),
        ("3xc".to_string(), tc),
    ])
}

/// A seeded grid league (6 clubs) with the cheapest squad owned; `negative` shifts a seeded
/// third of the xP below zero (blanks, red cards).
fn instance(
    seed: u64,
    n_gws: usize,
    gw_index: i64,
    ft: i64,
    negative: bool,
) -> (SquadState, Vec<PoolPlayer>, XpFrame) {
    let pool = grid_pool(1..=6, &GRID, Some(seed));
    let mut xp = correlated_xp(&pool, n_gws, seed, gw_index);
    if negative {
        let mut rng = Rng::new(seed.wrapping_add(3000));
        for x in xp.xp.iter_mut() {
            if rng.random() < 0.33 {
                *x = x.map(|v| v - 1.5);
            }
        }
    }
    let state = state_with(&pool, &first_valid_squad(&pool), gw_index, ft);
    (state, pool, xp)
}

fn problem_for(
    state: &SquadState,
    pool: &[PoolPlayer],
    xp: &XpFrame,
    params: &OptimizerParams,
) -> PlanInput {
    problem(state, pool, xp, params, &rules())
}

// --- bounds -------------------------------------------------------------------------------

#[test]
fn lp_and_dual_bounds_bound_every_scenario() {
    // For every scenario: MILP optimum ≤ its dual bound ≤ LP relaxation (gap 0).
    for seed in [0, 1] {
        let (state, pool, xp) = instance(seed, 2, 5, 1, seed == 1);
        let params = OptimizerParams { horizon: 2, chip_value: cheap_chips(), ..exact() };
        let problem = problem_for(&state, &pool, &xp, &params);
        for s in scenarios(&problem) {
            let plan = solve_plan(&problem, &params, None, &[], &s).unwrap();
            let lp = relaxation_bound(&problem, &params, &s).unwrap();
            assert!(plan.total_objective() <= plan.bound + 1e-6, "{}", s.label);
            assert!((plan.bound - plan.total_objective()).abs() <= 1e-6, "{}", s.label);
            assert!(plan.total_objective() <= lp + 1e-9, "{}", s.label);
        }
    }
}

#[test]
fn chip_extras_bound_tc_and_bb() {
    // A scenario's optimum ≤ its base's (WC/FH chips only) optimum + Σ E + the terminal value
    // change, the derived bound of the search (gap 0, negative xP in one instance).
    for seed in [0, 1] {
        let (state, pool, xp) = instance(seed, 2, 5, 1, seed == 1);
        let params = OptimizerParams { horizon: 2, chip_value: cheap_chips(), ..exact() };
        let problem = problem_for(&state, &pool, &xp, &params);
        let mut totals: HashMap<String, f64> = HashMap::new();
        let mut checked = 0;
        for s in scenarios(&problem) {
            let extras: Vec<&(usize, String)> =
                s.chips.iter().filter(|(_, n)| !TRANSFER_CHIPS.contains(&n.as_str())).collect();
            if extras.is_empty() {
                continue;
            }
            let kept: BTreeMap<usize, String> = s
                .chips
                .iter()
                .filter(|(_, n)| TRANSFER_CHIPS.contains(&n.as_str()))
                .map(|(t, n)| (*t, n.clone()))
                .collect();
            let base = make_scenario(&problem, &kept).unwrap();
            for sc in [&s, &base] {
                if !totals.contains_key(&sc.label) {
                    let plan = solve_plan(&problem, &params, None, &[], sc).unwrap();
                    totals.insert(sc.label.clone(), plan.total_objective());
                }
            }
            let extra: f64 = extras.iter().map(|(t, n)| chip_extra(&problem, &params, *t, n)).sum();
            let derived = totals[&base.label] - terminal_value(&problem, &params, &base)
                + extra
                + terminal_value(&problem, &params, &s);
            assert!(totals[&s.label] <= derived + 1e-6, "{}", s.label);
            checked += 1;
        }
        assert!(checked >= 10);
    }
}

#[test]
fn chip_extra_values() {
    let (state, pool, xp) = instance(4, 2, 5, 1, false);
    let params = OptimizerParams { horizon: 2, decay: 0.5, ..Default::default() };
    let problem = problem_for(&state, &pool, &xp, &params);
    let x1: Vec<f64> = problem.players.iter().map(|p| p.xp[1]).collect();
    assert!(x1.iter().all(|x| *x >= 0.0));
    assert_approx(
        chip_extra(&problem, &params, 1, "3xc"),
        0.5 * x1.iter().copied().fold(f64::MIN, f64::max),
    );
    let gk = problem
        .players
        .iter()
        .filter(|p| p.element_type == 1)
        .map(|p| p.xp[1])
        .fold(f64::MIN, f64::max);
    let mut out: Vec<f64> =
        problem.players.iter().filter(|p| p.element_type != 1).map(|p| p.xp[1]).collect();
    out.sort_by(|a, b| b.total_cmp(a));
    let w = &params.bench_weights; // (0.03, 0.21, 0.06, 0.002): |1 − w| desc 0.998, 0.94, 0.79
    let mut expected = 0.5 * ((1.0 - w[0]) * gk + (1.0 - w[3]) * out[0] + (1.0 - w[2]) * out[1]);
    expected += 0.5 * (1.0 - w[1]) * out[2];
    assert_approx(chip_extra(&problem, &params, 1, "bboost"), expected);
    assert_eq!(chip_extra(&problem, &params, 0, "wildcard"), f64::INFINITY);
}

// --- the search ---------------------------------------------------------------------------

type Case = (u64, usize, i64, i64, Option<BTreeMap<String, f64>>, f64, bool);

fn cases() -> Vec<Case> {
    vec![
        // (seed, n_gws, gw_index, ft, chip values, mip_gap, negative xP)
        (0, 2, 5, 1, None, 0.005, false),
        (1, 2, 5, 2, Some(cheap_chips()), 0.005, false),
        (2, 2, 18, 1, None, 0.0, false), // set-1 chips expire inside the horizon
        (3, 2, 19, 3, Some(cheap_chips()), 0.005, false), // both Free Hits in the horizon
        (4, 2, 1, 1, Some(cheap_chips()), 0.005, false), // GW1: unlimited free transfers
        (5, 2, 9, 1, None, 0.005, true),
        (36, 2, 30, 5, Some(cheap_chips()), 0.0, true), // seed 6 takes HiGHS ~25 s here
        (7, 2, 12, 1, Some(chip_values(0.0, 9.0, 0.0, 9.0)), 0.005, false),
        (8, 1, 38, 1, Some(cheap_chips()), 0.005, false), // the season's last GW
        (9, 1, 20, 2, Some(chip_values(2.0, 0.0, 2.0, 0.0)), 0.0, true),
    ]
}

fn bound_search_matches_all(case: Case) {
    // Soundness: the bound search returns exactly the exhaustive search's plan (same
    // scenario, objective and first-GW decision) with fewer solves.
    let (seed, n_gws, gw_index, ft, chip_value, gap, negative) = case;
    let (state, pool, xp) = instance(seed, n_gws, gw_index, ft, negative);
    let mut params = OptimizerParams {
        horizon: n_gws as i64,
        mip_gap: gap,
        prune_n: Some(small()),
        ..Default::default()
    };
    if let Some(values) = chip_value {
        params.chip_value = values;
    }
    let problem = problem_for(&state, &pool, &xp, &params);
    let candidates = scenarios(&problem);
    let full = search(&problem, &params, &candidates, Search::All).unwrap();
    let fast = search(&problem, &params, &candidates, Search::Bound).unwrap();
    assert_eq!(fast.scenario, full.scenario, "seed {seed}");
    assert_eq!(fast.plan.total_objective(), full.plan.total_objective(), "seed {seed}");
    assert_eq!(fast.plan.gws, full.plan.gws, "seed {seed}");
    assert_eq!(full.stats.n_solves, candidates.len());
    assert!(fast.stats.n_solves < full.stats.n_solves, "seed {seed}");
    // Solved scenarios are solved identically; skipped ones were provably worse.
    let full_objectives: HashMap<&str, f64> =
        full.objectives.iter().map(|(l, v)| (l.as_str(), *v)).collect();
    let fast_objectives: HashMap<&str, f64> =
        fast.objectives.iter().map(|(l, v)| (l.as_str(), *v)).collect();
    for (label, value) in &fast_objectives {
        assert_eq!(*value, full_objectives[label], "seed {seed}: {label}");
    }
    for (label, bound) in &fast.bounds {
        assert!(full_objectives[label.as_str()] <= bound + 1e-6, "seed {seed}: {label}");
        if !fast_objectives.contains_key(label.as_str()) {
            assert!(*bound <= fast.plan.total_objective() + 1e-6, "seed {seed}: {label}");
        }
    }
}

macro_rules! bound_search_case {
    ($($name:ident: $i:expr,)*) => {$(
        #[test]
        fn $name() {
            bound_search_matches_all(cases().swap_remove($i));
        }
    )*};
}

bound_search_case! {
    bound_search_returns_what_all_returns_case0: 0,
    bound_search_returns_what_all_returns_case1: 1,
    bound_search_returns_what_all_returns_case2: 2,
    bound_search_returns_what_all_returns_case3: 3,
    bound_search_returns_what_all_returns_case4: 4,
    bound_search_returns_what_all_returns_case5: 5,
    bound_search_returns_what_all_returns_case6: 6,
    bound_search_returns_what_all_returns_case7: 7,
    bound_search_returns_what_all_returns_case8: 8,
    bound_search_returns_what_all_returns_case9: 9,
}

#[test]
fn bound_search_solves_only_no_chip_when_chips_cannot_pay() {
    let (state, pool, xp) = instance(8, 3, 5, 1, false);
    let params = OptimizerParams {
        horizon: 3,
        chip_value: chip_values(500.0, 500.0, 500.0, 500.0),
        prune_n: Some(small()),
        ..Default::default()
    };
    let problem = problem_for(&state, &pool, &xp, &params);
    let result = optimize(&problem, &params, 1, true, false, Search::Bound).unwrap();
    assert_eq!(result.best().scenario, "none");
    assert_eq!(result.search.n_solves, 1);
    assert_eq!(
        result.scenario_objectives,
        vec![("none".to_string(), result.best().total_objective())]
    );
    let labels: Vec<String> = scenarios(&problem).into_iter().map(|s| s.label).collect();
    assert_eq!(result.scenario_bounds.iter().map(|(l, _)| l.clone()).collect::<Vec<_>>(), labels);
}

#[test]
fn search_validation() {
    let (state, pool, xp) = instance(9, 1, 5, 1, false);
    let params = OptimizerParams { horizon: 1, ..Default::default() };
    let problem = problem_for(&state, &pool, &xp, &params);
    let candidates = scenarios(&problem);
    assert_eq!((Search::Bound.name(), Search::All.name()), ("bound", "all"));
    let err = search(&problem, &params, &candidates[1..], Search::Bound).err().expect("an error");
    assert!(err.to_string().contains("no-chip"), "{err}");
}
