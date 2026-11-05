//! The MILP planner without chips (Python `tests/test_optimize_model.py`).

mod common;

use std::collections::{BTreeMap, BTreeSet, HashMap};

use common::*;
use fplopt_build::scoring::Rules;
use fplopt_optimize::plans::Plan;
use fplopt_optimize::state::{Holding, PoolPlayer, SquadState, Transfer, selling_price};
use fplopt_optimize::{
    ChipScenario, OptimizerParams, PlanInput, Search, SolveError, XpFrame, optimize, solve_plan,
};

/// Hits allowed (the default max_hits 0 never takes one): the brute force searches them too.
fn exact() -> OptimizerParams {
    OptimizerParams {
        mip_gap: 0.0,
        prune_n: None,
        prune_dominated: false,
        max_hits: None,
        tie_epsilon: 0.0,
        ..OptimizerParams::default()
    }
}

fn plan_for(
    state: &SquadState,
    pool: &[PoolPlayer],
    xp: &XpFrame,
    params: &OptimizerParams,
    rules: &Rules,
) -> (PlanInput, Plan) {
    let problem = problem(state, pool, xp, params, rules);
    let plan = solve(&problem, params);
    (problem, plan)
}

fn horizon(params: OptimizerParams, h: i64) -> OptimizerParams {
    OptimizerParams { horizon: h, ..params }
}

fn combinations(items: &[i64], r: usize) -> Vec<Vec<i64>> {
    if r == 0 {
        return vec![Vec::new()];
    }
    if items.len() < r {
        return Vec::new();
    }
    let mut out = Vec::new();
    for (i, &first) in items.iter().enumerate() {
        for mut rest in combinations(&items[i + 1..], r - 1) {
            rest.insert(0, first);
            out.push(rest);
        }
    }
    out
}

fn product(choices: &[Vec<Vec<i64>>]) -> Vec<Vec<i64>> {
    let mut out = vec![Vec::new()];
    for options in choices {
        let mut next = Vec::new();
        for prefix in &out {
            for o in options {
                let mut v = prefix.clone();
                v.extend(o);
                next.push(v);
            }
        }
        out = next;
    }
    out
}

// --- brute force ------------------------------------------------------------------------

/// 15 owned players (3 per club 1-5) plus 6 extras: clubs 1 and 6, so buying a club-1 extra
/// needs a club-1 sale (the club cap binds).
fn owned_tiny() -> Vec<i64> {
    let mut owned: Vec<i64> = [1, 2].iter().map(|&c| key(c, 1, 0)).collect();
    owned.extend([1, 2, 3, 4, 5].iter().map(|&c| key(c, 2, 0)));
    owned.extend([1, 3, 4, 5, 2].iter().map(|&c| key(c, 3, 0)));
    owned.extend([3, 4, 5].iter().map(|&c| key(c, 4, 0)));
    owned
}

const EXTRAS: [i64; 6] = [610, 121, 621, 631, 131, 641];

fn tiny_pool(seed: u64) -> (Vec<PoolPlayer>, XpFrame) {
    let mut rng = Rng::new(seed);
    let mut rows = Vec::new();
    for k in owned_tiny().into_iter().chain(EXTRAS) {
        let (et, club) = (et_of(k), club_of(k));
        rows.push((k, et, club, 40 + 5 * et + rng.integers(0, 25)));
    }
    let pool = make_pool(rows);
    let values: BTreeMap<i64, Vec<f64>> =
        pool.iter().map(|p| (p.player_key, vec![round_to(rng.uniform(0.0, 9.0), 2)])).collect();
    (pool, xp_frame(&values, 5))
}

/// Max objective over every transfer set (any size, the pool is tiny), 1-GW horizon.
fn brute_force_one_gw(
    state: &SquadState,
    pool: &[PoolPlayer],
    xp: &XpFrame,
    params: &OptimizerParams,
) -> f64 {
    let rules = rules();
    let et: HashMap<i64, i64> = pool.iter().map(|p| (p.player_key, p.element_type)).collect();
    let club: HashMap<i64, i64> = pool.iter().map(|p| (p.player_key, p.team_key)).collect();
    let price: HashMap<i64, i64> = pool.iter().map(|p| (p.player_key, p.price)).collect();
    let xs = xp_at(xp, 0);
    let owned: Vec<i64> = state.player_keys();
    let sell: HashMap<i64, i64> = state
        .holdings
        .iter()
        .map(|h| (h.player_key, selling_price(h, rules.sell_on_fee)))
        .collect();
    let extras: Vec<i64> =
        pool.iter().map(|p| p.player_key).filter(|k| !owned.contains(k)).collect();
    let hit = rules.hit_cost as f64 + params.hit_margin;
    let mut best = f64::NEG_INFINITY;
    for r in 0..=extras.len() {
        for ins in combinations(&extras, r) {
            let mut need: BTreeMap<i64, usize> = BTreeMap::new();
            for k in &ins {
                *need.entry(et[k]).or_default() += 1;
            }
            let choices: Vec<Vec<Vec<i64>>> = need
                .iter()
                .map(|(p, n)| {
                    let mut at: Vec<i64> = owned.iter().copied().filter(|k| et[k] == *p).collect();
                    at.sort_unstable();
                    combinations(&at, *n)
                })
                .collect();
            for outs in product(&choices) {
                let squad: Vec<i64> = owned
                    .iter()
                    .copied()
                    .filter(|k| !outs.contains(k))
                    .chain(ins.iter().copied())
                    .collect();
                let mut clubs: HashMap<i64, i64> = HashMap::new();
                for k in &squad {
                    *clubs.entry(club[k]).or_default() += 1;
                }
                if ins.iter().any(|k| clubs[&club[k]] > rules.team_limit) {
                    continue;
                }
                let bank = state.bank + outs.iter().map(|k| sell[k]).sum::<i64>()
                    - ins.iter().map(|k| price[k]).sum::<i64>();
                if bank < 0 {
                    continue;
                }
                let n = ins.len() as i64;
                let hits = if state.gw_index == 1 { 0 } else { (n - state.free_transfers).max(0) };
                let value = best_lineup_value(&squad, &xs, &et, &params.bench_weights)
                    + params.itb_value * bank as f64 / 10.0
                    + params.ft_state_value(state.free_transfers.max(0))
                    - hit * hits as f64;
                best = best.max(value);
            }
        }
    }
    best
}

#[test]
fn one_gw_matches_brute_force() {
    let rules = rules();
    for (seed, ft, margin) in
        [(0, 1, 0.0), (1, 1, 0.0), (2, 2, 0.0), (3, 1, 1.5), (4, 3, 0.0), (5, 1, 0.0)]
    {
        let (pool, xp) = tiny_pool(seed);
        let owned = owned_tiny();
        let purchase: HashMap<i64, i64> = owned.iter().step_by(3).map(|&k| (k, 45)).collect();
        let state = make_state(
            &pool,
            &owned,
            StateOpts { ft, bank: Some(15), purchase, ..StateOpts::default() },
        );
        let params = OptimizerParams { hit_margin: margin, ..exact() };
        let (problem, plan) = plan_for(&state, &pool, &xp, &params, &rules);
        let expected = brute_force_one_gw(&problem.state, &pool, &xp, &params);
        assert!(
            approx(plan.objective, expected, 1e-6),
            "seed {seed}: {} vs {expected}",
            plan.objective
        );
        apply_first(&state, &plan, &pool, &rules);
    }
}

#[test]
fn gw1_squad_choice_matches_brute_force() {
    // gw_index 1: unlimited free transfers, so the plan picks the best squad outright.
    let rules = rules();
    let mut rng = Rng::new(1); // a seed whose prices admit a 2/5/5/3 squad under the club cap
    // 3 GK, 6 DEF, 6 MID, 4 FWD over 6 clubs (the club cap binds).
    let n = [(1, 3), (2, 6), (3, 6), (4, 4)];
    let mut rows = Vec::new();
    for (et, count) in n {
        for i in 0..count {
            let club = 1 + (i + 2 * et) % 6;
            rows.push((key(club, et, i), et, club, 35 + 5 * et + rng.integers(0, 30)));
        }
    }
    let pool = make_pool(rows);
    let values: BTreeMap<i64, Vec<f64>> =
        pool.iter().map(|p| (p.player_key, vec![round_to(rng.uniform(0.0, 9.0), 2)])).collect();
    let xp = xp_frame(&values, 1);
    let owned = first_valid_squad(&pool);
    let state = state_with(&pool, &owned, 1, 1);
    let params = exact();
    let (_, plan) = plan_for(&state, &pool, &xp, &params, &rules);

    let et: HashMap<i64, i64> = pool.iter().map(|p| (p.player_key, p.element_type)).collect();
    let club: HashMap<i64, i64> = pool.iter().map(|p| (p.player_key, p.team_key)).collect();
    let price: HashMap<i64, i64> = pool.iter().map(|p| (p.player_key, p.price)).collect();
    let flat: HashMap<i64, f64> = values.iter().map(|(k, v)| (*k, v[0])).collect();
    let budget = state.bank + owned.iter().map(|k| price[k]).sum::<i64>();
    let choices: Vec<Vec<Vec<i64>>> = n
        .iter()
        .map(|(p, _)| {
            let keys: Vec<i64> =
                pool.iter().filter(|q| q.element_type == *p).map(|q| q.player_key).collect();
            combinations(&keys, rules.squad_select[p] as usize)
        })
        .collect();
    let mut best = f64::NEG_INFINITY;
    for squad in product(&choices) {
        let mut clubs: HashMap<i64, i64> = HashMap::new();
        for k in &squad {
            *clubs.entry(club[k]).or_default() += 1;
        }
        if clubs.values().max().copied().unwrap_or(0) > rules.team_limit {
            continue;
        }
        let bank = budget - squad.iter().map(|k| price[k]).sum::<i64>();
        if bank < 0 {
            continue;
        }
        let value = best_lineup_value(&squad, &flat, &et, &params.bench_weights);
        best = best.max(value + params.itb_value * bank as f64 / 10.0 + params.ft_state_value(1));
    }
    assert!(approx(plan.objective, best, 1e-6), "{} vs {best}", plan.objective);
    assert_eq!(plan.first().hits, 0);
    apply_first(&state, &plan, &pool, &rules);
}

// --- decisions and replay -----------------------------------------------------------------

fn medium_instance(
    seed: u64,
    ft: i64,
    gw_index: i64,
    n_gws: usize,
) -> (SquadState, Vec<PoolPlayer>, XpFrame) {
    let pool = grid_pool(1..=8, &GRID, Some(seed));
    let xp = correlated_xp(&pool, n_gws, seed, gw_index);
    let state = state_with(&pool, &first_valid_squad(&pool), gw_index, ft);
    (state, pool, xp)
}

#[test]
fn multi_gw_plan_replays_through_the_rules_engine() {
    let rules = rules();
    for seed in [0, 1] {
        let (state, pool, xp) = medium_instance(seed, 1, 5, 4);
        let params = OptimizerParams { horizon: 4, ..OptimizerParams::default() };
        let (_, plan) = plan_for(&state, &pool, &xp, &params, &rules);
        assert_eq!(plan.gws.len(), 4);
        assert_eq!(plan.status, "Optimal");
        assert!(plan.gws.iter().any(|g| g.n_transfers() > 0), "seed {seed}: no transfers");
        replay(&state, &plan, &pool, &rules);
    }
}

#[test]
fn gw1_plan_replays_with_one_ft_after() {
    let rules = rules();
    let (state, pool, xp) = medium_instance(3, 1, 1, 3);
    let (_, plan) =
        plan_for(&state, &pool, &xp, &OptimizerParams { horizon: 3, ..Default::default() }, &rules);
    assert!(plan.first().hits == 0 && plan.first().n_transfers() > 1);
    assert_eq!(plan.gws[1].ft, 1);
    replay(&state, &plan, &pool, &rules);
}

#[test]
fn ft_topups_follow_the_rules() {
    let rules = Rules { ft_topups: vec![(6, 2)], ..common::rules() };
    let (state, pool, mut xp) = medium_instance(4, 1, 5, 3);
    // Nothing worth a transfer (no xP, and no value on money in the bank): the FTs bank.
    xp.xp.iter_mut().for_each(|x| *x = Some(0.0));
    let params = OptimizerParams { horizon: 3, itb_value: 0.0, ..Default::default() };
    let (_, plan) = plan_for(&state, &pool, &xp, &params, &rules);
    assert_eq!(plan.gws.iter().map(|g| g.ft).collect::<Vec<_>>(), vec![1, 4, 5]);
    assert!(plan.gws.iter().all(|g| g.n_transfers() == 0));
    replay(&state, &plan, &pool, &rules);
}

#[test]
fn first_gw_decision_passes_apply_decision_and_matches_best_lineup() {
    let rules = rules();
    let (state, pool, xp) = medium_instance(2, 2, 5, 3);
    let (_, plan) = plan_for(&state, &pool, &xp, &OptimizerParams::default(), &rules);
    let gw_state = apply_first(&state, &plan, &pool, &rules);
    let g = plan.first();
    let xs = xp_at(&xp, 0);
    let top = g.starters.iter().map(|k| xs[k]).fold(f64::NEG_INFINITY, f64::max);
    assert_eq!(xs[&g.captain], top);
    let vice = g
        .starters
        .iter()
        .copied()
        .filter(|k| *k != g.captain)
        .min_by(|a, b| xs[b].total_cmp(&xs[a]).then(a.cmp(b)))
        .unwrap();
    assert_eq!(g.vice, vice);
    let keys: BTreeSet<i64> = gw_state.player_keys().into_iter().collect();
    assert_eq!(squad_of(g), keys);
}

// --- money --------------------------------------------------------------------------------

/// A owned (bought 50, now 60: sells for 55 first); C costs 50. A scores in GW 1 only, C in
/// GWs 0 and 2, so the plan churns A → C → A → C, all on free transfers.
fn churn_instance() -> (SquadState, Vec<PoolPlayer>, XpFrame, i64, i64) {
    let clubs = 1..=5;
    let mut rows = Vec::new();
    for c in clubs.clone() {
        for et in 1..=4 {
            rows.push((key(c, et, 0), et, c, 45));
        }
    }
    rows.extend(clubs.clone().map(|c| (key(c, 3, 1), 3, c, 45)));
    rows.extend(clubs.clone().map(|c| (key(c, 2, 1), 2, c, 45)));
    rows.extend([1, 2].map(|c| (key(c, 4, 1), 4, c, 45)));
    rows.push((key(1, 1, 1), 1, 1, 45));
    let (a, c_key) = (key(6, 3, 0), key(7, 3, 0));
    rows.extend([(a, 3, 6, 60), (c_key, 3, 7, 50)]);
    let pool = make_pool(rows);
    let mut owned = vec![key(1, 1, 0), key(2, 1, 0)];
    owned.extend(clubs.clone().map(|c| key(c, 2, 0)));
    owned.extend([1, 2, 3, 4].map(|c| key(c, 3, 0)));
    owned.push(a);
    owned.extend([1, 2, 3].map(|c| key(c, 4, 0)));
    let mut values: BTreeMap<i64, Vec<f64>> =
        pool.iter().map(|p| (p.player_key, vec![0.5; 3])).collect();
    for k in &owned {
        values.insert(*k, vec![1.0; 3]); // no other move is worth making
    }
    values.insert(a, vec![0.0, 20.0, 0.0]);
    values.insert(c_key, vec![10.0, 0.0, 10.0]);
    let xp = xp_frame(&values, 5);
    let state = make_state(
        &pool,
        &owned,
        StateOpts {
            ft: 2,
            bank: Some(5),
            purchase: HashMap::from([(a, 50)]),
            ..Default::default()
        },
    );
    (state, pool, xp, a, c_key)
}

#[test]
fn first_sale_at_selling_price_later_sales_at_buy_price() {
    let rules = rules();
    let (state, pool, xp, a, c) = churn_instance();
    let params =
        OptimizerParams { horizon: 3, ft_value: BTreeMap::new(), itb_value: 0.0, ..exact() };
    let (problem, plan) = plan_for(&state, &pool, &xp, &params, &rules);
    assert_eq!(player(&problem, a).sell_price, 55);
    let moves: Vec<(Vec<i64>, Vec<i64>)> =
        plan.gws.iter().map(|g| (transfers_out(g), transfers_in(g))).collect();
    assert_eq!(moves, vec![(vec![a], vec![c]), (vec![c], vec![a]), (vec![a], vec![c])]);
    // first sale +55 −50, then −60 +50 (C at its buy price), then +60 (A at buy price) −50.
    assert_eq!(plan.gws.iter().map(|g| g.bank).collect::<Vec<_>>(), vec![10, 0, 10]);
    replay(&state, &plan, &pool, &rules);
}

#[test]
fn departed_player_is_sold_at_last_price_and_never_bought() {
    let rules = rules();
    let pool = grid_pool(1..=6, &[(1, 2), (2, 4), (3, 4), (4, 2)], None);
    let mut owned = first_valid_squad(&pool);
    let gone_def = *owned.iter().find(|k| et_of(**k) == 2).unwrap();
    owned.retain(|k| *k != gone_def);
    let departed =
        Holding { player_key: 9921, element_type: 2, team_key: 9, purchase_price: 56, price: 60 };
    let mut values: BTreeMap<i64, Vec<f64>> =
        pool.iter().map(|p| (p.player_key, vec![2.0, 2.0])).collect();
    values.insert(key(6, 2, 3), vec![6.0, 6.0]);
    let state = make_state(
        &pool,
        &owned,
        StateOpts { ft: 1, extra: vec![departed], bank: Some(100), ..Default::default() },
    );
    let xp = xp_frame(&values, 5);
    let (problem, plan) = plan_for(&state, &pool, &xp, &horizon(exact(), 2), &rules);
    let p = player(&problem, 9921);
    assert_eq!((p.owned, p.buyable, p.sell_price, p.xp.clone()), (true, false, 58, vec![0.0, 0.0]));
    assert_eq!(transfers_out(plan.first()), vec![9921]);
    assert!(plan.gws.iter().all(|g| !transfers_in(g).contains(&9921)));
    assert_eq!(plan.first().bank, 100 + 58 - 50);
    replay(&state, &plan, &pool, &rules);
}

// --- free transfers and hits -------------------------------------------------------------

/// All owned score 2 (a FWD 20, the captain); two non-owned MIDs score 2 + gains.
fn upgrade_instance(
    ft: i64,
    n_gws: usize,
    gains: [f64; 2],
) -> (SquadState, Vec<PoolPlayer>, XpFrame, [i64; 2]) {
    let pool = grid_pool(1..=8, &GRID5, None);
    let owned = first_valid_squad(&pool);
    let mut values: BTreeMap<i64, Vec<f64>> =
        pool.iter().map(|p| (p.player_key, vec![0.0; n_gws])).collect();
    for k in &owned {
        values.insert(*k, vec![2.0; n_gws]);
    }
    let captain = *owned.iter().find(|k| et_of(**k) == 4).unwrap();
    values.insert(captain, vec![20.0; n_gws]);
    let ups = [key(8, 3, 0), key(8, 3, 1)]; // club 8 has room
    assert!(ups.iter().all(|u| !owned.contains(u)) && owned.iter().all(|k| club_of(*k) != 8));
    for (k, gain) in ups.iter().zip(gains) {
        values.insert(*k, vec![2.0 + gain; n_gws]);
    }
    let state = state_with(&pool, &owned, 5, ft);
    (state, pool, xp_frame(&values, 5), ups)
}

#[test]
fn hit_margin() {
    let rules = rules();
    for (margin, n_transfers, hits) in [(0.0, 2, 1), (1.5, 1, 0)] {
        let (state, pool, xp, ups) = upgrade_instance(1, 1, [6.0, 5.0]);
        let params = OptimizerParams { horizon: 1, hit_margin: margin, ..exact() };
        let (_, plan) = plan_for(&state, &pool, &xp, &params, &rules);
        assert_eq!((plan.first().n_transfers(), plan.first().hits), (n_transfers, hits));
        assert!(transfers_in(plan.first()).contains(&ups[0]));
        replay(&state, &plan, &pool, &rules);
    }
}

#[test]
fn max_hits_caps_the_hits_per_gw() {
    // Three upgrades worth 6/5/5 against a 4-point hit: unlimited takes two hits; max_hits
    // caps them (0 = only the free transfer).
    let rules = rules();
    for (max_hits, n_transfers, hits) in
        [(None, 3, 2), (Some(2), 3, 2), (Some(1), 2, 1), (Some(0), 1, 0)]
    {
        let (state, pool, mut xp, ups) = upgrade_instance(1, 1, [6.0, 5.0]);
        let third = key(7, 3, 0);
        assert!(xp_value(&xp, third, 0) == 0.0 && !state.player_keys().contains(&third));
        set_xp(&mut xp, third, &[7.0]);
        let params = OptimizerParams { horizon: 1, max_hits, ..exact() };
        let (_, plan) = plan_for(&state, &pool, &xp, &params, &rules);
        assert_eq!(
            (plan.first().n_transfers(), plan.first().hits),
            (n_transfers, hits),
            "{max_hits:?}"
        );
        assert!(transfers_in(plan.first()).contains(&ups[0])); // the best upgrade goes first
        replay(&state, &plan, &pool, &rules);
    }
}

#[test]
fn max_hits_holds_in_every_horizon_gw() {
    let rules = rules();
    for seed in [0, 1] {
        let (state, pool, mut churny) = medium_instance(seed, 1, 5, 3);
        for i in 0..churny.len() {
            let m = 1 + 8 * ((churny.player_key[i] * (churny.horizon[i] + 1)) % 3);
            churny.xp[i] = churny.xp[i].map(|x| x * m as f64);
        }
        // Form swings big enough to pay for hits.
        let free = OptimizerParams { horizon: 3, max_hits: None, ..Default::default() };
        let (_, unlimited) = plan_for(&state, &pool, &churny, &free, &rules);
        assert!(unlimited.gws.iter().map(|g| g.hits).max().unwrap() > 1, "seed {seed}");
        for cap in [0, 1] {
            let params = OptimizerParams { max_hits: Some(cap), ..free.clone() };
            let (_, plan) = plan_for(&state, &pool, &churny, &params, &rules);
            assert!(
                plan.gws.iter().all(|g| g.hits <= cap),
                "{:?}",
                plan.gws.iter().map(|g| g.hits).collect::<Vec<_>>()
            );
            assert!(plan.objective <= unlimited.objective * (1.0 + free.mip_gap) + 1e-6);
            replay(&state, &plan, &pool, &rules);
        }
    }
}

#[test]
fn max_hits_validation_and_defaults() {
    let err = OptimizerParams { max_hits: Some(-1), ..Default::default() }.validate().unwrap_err();
    assert!(err.to_string().contains("max_hits"));
    let default = OptimizerParams::default();
    assert_eq!((default.max_hits, default.itb_value), (Some(0), 0.0));
    default.validate().unwrap();
    OptimizerParams { prune_n: None, max_hits: None, ..Default::default() }.validate().unwrap();
}

#[test]
fn params_reject_values_the_model_cannot_honour() {
    // Dominance pruning assumes a cheaper player with >= xP is never worse (a negative
    // itb_value breaks that); negative hit margins/epsilons and empty horizons are errors.
    let d = OptimizerParams::default;
    let cases: Vec<(&str, OptimizerParams)> = vec![
        ("itb_value", OptimizerParams { itb_value: -0.1, ..d() }),
        ("hit_margin", OptimizerParams { hit_margin: -1.0, ..d() }),
        ("tie_epsilon", OptimizerParams { tie_epsilon: -1e-4, ..d() }),
        ("max_hits", OptimizerParams { max_hits: Some(-1), ..d() }),
        ("decay", OptimizerParams { decay: 0.0, ..d() }),
        ("decay", OptimizerParams { decay: 1.5, ..d() }),
        ("horizon", OptimizerParams { horizon: 0, ..d() }),
        ("node_limit", OptimizerParams { node_limit: Some(0), ..d() }),
        ("time_limit", OptimizerParams { time_limit: Some(0.0), ..d() }),
        ("itb_value", OptimizerParams { itb_value: f64::NAN, ..d() }),
    ];
    for (field, params) in cases {
        let err = params.validate().expect_err(field).to_string();
        assert!(err.contains(field), "{field}: {err}");
    }
}

#[test]
fn default_solver_limits_are_deterministic() {
    // Backtests never depend on wall-clock time: no time limit by default, a node limit as
    // the safety net.
    let default = OptimizerParams::default();
    assert!(default.time_limit.is_none() && default.node_limit.is_some_and(|n| n >= 10_000));
}

#[test]
fn a_node_limit_stop_is_reported_not_silent() {
    // A solve stopped by the node limit keeps its plan but is not `Optimal`; without any
    // plan it is a LimitReached error.
    let rules = rules();
    let (state, pool, xp) = medium_instance(2, 2, 5, 3); // a seed whose exact solve needs > 1 node
    let params = OptimizerParams { horizon: 3, mip_gap: 0.0, node_limit: Some(1), ..exact() };
    let problem = problem(&state, &pool, &xp, &params, &rules);
    match solve_plan(&problem, &params, None, &[], &ChipScenario::none()) {
        Err(SolveError::LimitReached { status, .. }) => assert_ne!(status, "Optimal"),
        Err(e) => panic!("unexpected error {e}"),
        Ok(plan) => {
            let exact = solve(&problem, &OptimizerParams { node_limit: None, ..params.clone() });
            assert!(exact.optimal() && exact.n_nodes > 1, "instance too easy to hit the limit");
            assert!(!plan.optimal() && plan.status != "Optimal");
            assert!(plan.objective <= exact.objective + 1e-6);
        }
    }
}

#[test]
fn tie_epsilon_prefers_fewer_transfers_when_xp_ties() {
    // All-zero xP at GW1: without the tie-break any squad is optimal; with it the planner
    // keeps the squad it has.
    let rules = rules();
    let pool = grid_pool(1..=8, &GRID5, None);
    let owned = first_valid_squad(&pool);
    let values: BTreeMap<i64, Vec<f64>> =
        pool.iter().map(|p| (p.player_key, vec![0.0, 0.0])).collect();
    let xp = xp_frame(&values, 1);
    let state = state_with(&pool, &owned, 1, 1);
    let params =
        OptimizerParams { horizon: 2, prune_n: None, prune_dominated: false, ..Default::default() };
    let (problem, plan) = plan_for(&state, &pool, &xp, &params, &rules);
    assert!(plan.gws.iter().all(|g| g.n_transfers() == 0));
    assert!(approx(plan.objective, 0.85 * params.ft_state_value(1), 1e-12));
    // The epsilon is part of the objective: one forced transfer costs exactly epsilon.
    let mut clubs: HashMap<i64, i64> = HashMap::new();
    for k in &owned {
        *clubs.entry(pool_player(&pool, *k).team_key).or_default() += 1;
    }
    let out = pool_player(&pool, *owned.last().unwrap());
    let buy = pool
        .iter()
        .find(|p| {
            !owned.contains(&p.player_key)
                && p.element_type == out.element_type
                && clubs.get(&p.team_key).copied().unwrap_or(0)
                    - i64::from(p.team_key == out.team_key)
                    < rules.team_limit
                && p.price <= state.bank + out.price
        })
        .unwrap();
    let swap = (BTreeSet::from([out.player_key]), BTreeSet::from([buy.player_key]));
    let forced = solve_plan(&problem, &params, Some(&swap), &[], &ChipScenario::none()).unwrap();
    let diff = plan.objective - forced.objective;
    assert!((diff - params.tie_epsilon).abs() <= 1e-6 * params.tie_epsilon, "{diff}");
}

#[test]
fn high_ft_value_rolls_zero_ft_value_transfers() {
    let rules = rules();
    let (state, pool, xp, ups) = upgrade_instance(1, 2, [1.0, 0.5]);
    let rolled_params =
        OptimizerParams { horizon: 2, ft_value: BTreeMap::from([(2, 10.0)]), ..exact() };
    let (_, rolled) = plan_for(&state, &pool, &xp, &rolled_params, &rules);
    assert!(rolled.first().n_transfers() == 0 && rolled.gws[1].ft == 2);
    let moved_params = OptimizerParams { horizon: 2, ft_value: BTreeMap::new(), ..exact() };
    let (_, moved) = plan_for(&state, &pool, &xp, &moved_params, &rules);
    assert_eq!(transfers_in(moved.first()), vec![ups[0]]);
}

#[test]
fn fts_bank_to_the_cap() {
    let rules = rules();
    let (state, pool, xp, _) = upgrade_instance(4, 3, [0.0, 0.0]);
    let params = horizon(exact(), 3);
    let (_, plan) = plan_for(&state, &pool, &xp, &params, &rules);
    assert_eq!(plan.gws.iter().map(|g| g.ft).collect::<Vec<_>>(), vec![4, 5, 5]);
    assert!(plan.gws.iter().all(|g| g.n_transfers() == 0));
    // The gain in FT value (V(5) − V(4) = 1.1) is credited in GW 1, decayed.
    let g = &plan.gws[1];
    let lhs = g.objective - 0.85 * (g.xp + g.bench_xp);
    let rhs = 0.85 * (1.1 + params.itb_value * g.bank as f64 / 10.0);
    assert!(approx(lhs, rhs, 1e-12), "{lhs} vs {rhs}");
}

#[test]
fn hits_never_buy_free_transfers() {
    // With a huge FT value, paying a hit for a transfer you had an FT for would raise next
    // GW's FTs under a ≤ formulation; the exact dynamics forbid it.
    let rules = rules();
    let (state, pool, xp, _) = upgrade_instance(2, 2, [0.0, 0.0]);
    let params = OptimizerParams { horizon: 2, ft_value: BTreeMap::from([(3, 50.0)]), ..exact() };
    let (_, plan) = plan_for(&state, &pool, &xp, &params, &rules);
    assert!(plan.first().hits == 0 && plan.gws[1].ft == 3);
    replay(&state, &plan, &pool, &rules);
}

// --- club cap -----------------------------------------------------------------------------

/// Club 1 has 4 owned players after a club change; a club-1 star is on the market.
fn moved_club_instance(star_xp: f64) -> (SquadState, Vec<PoolPlayer>, XpFrame, i64, i64) {
    let mut pool = grid_pool(1..=8, &GRID5, None);
    let owned = first_valid_squad(&pool);
    assert_eq!(owned.iter().filter(|k| club_of(**k) == 1).count(), 3);
    let mover = *owned.iter().find(|k| club_of(**k) != 1 && et_of(**k) == 3).unwrap();
    pool.iter_mut().filter(|p| p.player_key == mover).for_each(|p| p.team_key = 1); // moved
    let star = key(1, 3, 4);
    assert!(!owned.contains(&star));
    let mut values: BTreeMap<i64, Vec<f64>> =
        pool.iter().map(|p| (p.player_key, vec![1.0])).collect();
    for k in &owned {
        values.insert(*k, vec![2.0]);
    }
    values.insert(star, vec![star_xp]);
    let state =
        make_state(&pool, &owned, StateOpts { ft: 5, bank: Some(50), ..Default::default() });
    (state, pool, xp_frame(&values, 5), star, mover)
}

#[test]
fn club_over_the_cap_must_shed_two_to_buy_one() {
    let rules = rules();
    let (state, pool, xp, star, mover) = moved_club_instance(40.0);
    let (_, plan) = plan_for(&state, &pool, &xp, &horizon(exact(), 1), &rules);
    assert!(transfers_in(plan.first()).contains(&star));
    let gw_state = apply_first(&state, &plan, &pool, &rules);
    assert!(
        gw_state.holdings.iter().filter(|h| h.team_key == 1).count() as i64 <= rules.team_limit
    );
    // Selling just one club-1 player for the star is what the rules forbid:
    let bad = Decision {
        transfers: vec![Transfer { out_key: mover, in_key: star }],
        ..decision(plan.first())
    };
    let err = apply_decision(&state, &bad, &pool, &rules).unwrap_err();
    assert!(err.0.contains("club 1"), "{err}");
}

#[test]
fn club_over_the_cap_is_tolerated_without_buys() {
    let rules = rules();
    let (state, pool, xp, _, _) = moved_club_instance(0.0);
    let (_, plan) = plan_for(&state, &pool, &xp, &horizon(exact(), 1), &rules);
    assert_eq!(plan.first().n_transfers(), 0);
    apply_first(&state, &plan, &pool, &rules);
}

// --- horizon, hooks, determinism ----------------------------------------------------------

#[test]
fn horizon_shortens_near_season_end() {
    let rules = rules();
    let (state, pool, xp) = medium_instance(5, 1, 36, 3);
    let (problem, plan) =
        plan_for(&state, &pool, &xp, &OptimizerParams { horizon: 6, ..Default::default() }, &rules);
    assert_eq!(problem.gws.iter().map(|g| g.gw_index).collect::<Vec<_>>(), vec![36, 37, 38]);
    assert_eq!(plan.gws.len(), 3);
    let short = common::problem(
        &state,
        &pool,
        &xp,
        &OptimizerParams { horizon: 2, ..Default::default() },
        &rules,
    );
    assert_eq!(short.gws.iter().map(|g| g.horizon).collect::<Vec<_>>(), vec![0, 1]);
}

#[test]
fn xp_frame_must_start_at_the_states_gw() {
    let rules = rules();
    let (state, pool, xp) = medium_instance(5, 1, 7, 3);
    let moved = SquadState { gw_index: 8, ..state };
    let err = PlanInput::from_context(&moved, &pool, &xp, &rules, &OptimizerParams::default())
        .unwrap_err();
    assert!(err.to_string().contains("gw_index 8"), "{err}");
}

#[test]
fn missing_xp_counts_zero() {
    let rules = rules();
    let (state, pool, xp) = medium_instance(6, 1, 5, 2);
    let dropped = pool.last().unwrap().player_key;
    let mut xp = filter_rows(&xp, |i| xp.player_key[i] != dropped);
    xp.xp[0] = Some(f64::NAN);
    let problem = problem(&state, &pool, &xp, &exact(), &rules);
    assert_eq!(player(&problem, dropped).xp, vec![0.0, 0.0]);
    assert_eq!(problem.players[0].xp[0], 0.0);
}

#[test]
fn fix_and_exclude_first_gw() {
    let rules = rules();
    let (state, pool, xp) = medium_instance(27, 2, 5, 3); // seed 7 takes HiGHS ~15 s here
    let params = OptimizerParams::default();
    let problem = problem(&state, &pool, &xp, &params, &rules);
    let best = solve(&problem, &params);
    assert!(best.first().n_transfers() > 0);
    let empty = (BTreeSet::new(), BTreeSet::new());
    let roll = solve_plan(&problem, &params, Some(&empty), &[], &ChipScenario::none()).unwrap();
    assert_eq!(roll.first().n_transfers(), 0);
    assert!(roll.objective <= best.objective + 1e-6);
    let first = best.first().transfer_set();
    let second =
        solve_plan(&problem, &params, None, std::slice::from_ref(&first), &ChipScenario::none())
            .unwrap();
    assert_ne!(second.first().transfer_set(), first);
    assert!(second.objective <= best.objective * (1.0 + params.mip_gap) + 1e-6);
}

fn same_input(a: &PlanInput, b: &PlanInput) {
    assert_eq!(a.state, b.state);
    assert_eq!(a.players, b.players);
    assert_eq!(a.gws, b.gws);
    assert_eq!(
        (a.bank, a.free_transfers, a.gw1, a.n_pool),
        (b.bank, b.free_transfers, b.gw1, b.n_pool)
    );
    assert_eq!(a.bench_weights, b.bench_weights);
}

#[test]
fn same_input_same_plan() {
    let rules = rules();
    let (state, pool, xp) = medium_instance(18, 1, 5, 3); // a seed HiGHS solves fast
    let params = OptimizerParams::default();
    let a = problem(&state, &pool, &xp, &params, &rules);
    let mut rng = Rng::new(1);
    let shuffled: Vec<PoolPlayer> =
        rng.permutation(pool.len()).into_iter().map(|i| pool[i]).collect();
    let xp_shuffled = take_rows(&xp, &rng.permutation(xp.len()));
    let b = problem(&state, &shuffled, &xp_shuffled, &params, &rules);
    same_input(&a, &b);
    let p = optimize(&a, &params, 1, false, true, Search::Bound).unwrap();
    let q = optimize(&b, &params, 1, false, true, Search::Bound).unwrap();
    assert_eq!(p.best().gws, q.best().gws);
    assert_eq!(p.best().objective, q.best().objective);
}

#[test]
fn plan_reports() {
    let rules = rules();
    let (state, pool, xp) = medium_instance(9, 1, 5, 2);
    let (_, plan) =
        plan_for(&state, &pool, &xp, &OptimizerParams { horizon: 2, ..Default::default() }, &rules);
    assert!(plan.build_seconds > 0.0 && plan.solve_seconds > 0.0);
    assert!((0.0..=0.005 + 1e-9).contains(&plan.mip_gap));
    assert!(approx(plan.objective, plan.gws.iter().map(|g| g.objective).sum(), 1e-9));
    for g in &plan.gws {
        assert!(g.chip.is_none() && g.starters.len() == 11 && g.bench.len() == 4);
        assert!(
            g.vice != g.captain && g.starters.contains(&g.captain) && g.starters.contains(&g.vice)
        );
    }
}

#[test]
fn random_xp_instance_replays() {
    // A churny instance (independent xP per GW) still replays exactly.
    let rules = rules();
    let pool = grid_pool(1..=6, &GRID, Some(11));
    let xp = random_xp(&pool, 2, 11, 5);
    let state = state_with(&pool, &first_valid_squad(&pool), 5, 1);
    let (_, plan) =
        plan_for(&state, &pool, &xp, &OptimizerParams { horizon: 2, ..Default::default() }, &rules);
    replay(&state, &plan, &pool, &rules);
}
