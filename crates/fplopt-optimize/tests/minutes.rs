//! The minutes model in the planner: bench weights from P(starter doesn't play) and the
//! expected-minutes candidate floor (Python `tests/test_optimize_minutes.py`).

mod common;

use std::collections::{BTreeSet, HashMap};

use common::*;
use fplopt_optimize::minutes::{
    horizon_minutes, minutes_bench_weights, not_play_probabilities, poisson_binomial_tail,
    projected_xi, skip_aware_tail,
};
use fplopt_optimize::params::DEFAULT_BENCH_WEIGHTS;
use fplopt_optimize::plans::Plan;
use fplopt_optimize::search::chip_extra;
use fplopt_optimize::state::{PoolPlayer, SquadState};
use fplopt_optimize::{OptimizerParams, PlanInput, XpFrame, solve_plan};

fn exact() -> OptimizerParams {
    OptimizerParams {
        mip_gap: 0.0,
        prune_n: None,
        prune_dominated: false,
        tie_epsilon: 0.0,
        ..OptimizerParams::default()
    }
}

fn instance(seed: u64, n_gws: usize, gw_index: i64) -> (SquadState, Vec<PoolPlayer>, XpFrame) {
    let pool = grid_pool(1..=8, &GRID, Some(seed));
    let xp = correlated_xp(&pool, n_gws, seed, gw_index);
    let state = state_with(&pool, &first_valid_squad(&pool), gw_index, 1);
    (state, pool, xp)
}

/// A minutes column: one value for every row, or player_key -> per-horizon values (1.0 for
/// players not listed).
enum Col {
    All(f64),
    By(HashMap<i64, Vec<f64>>),
}

fn column(xp: &XpFrame, values: &Col) -> Vec<Option<f64>> {
    (0..xp.len())
        .map(|i| {
            Some(match values {
                Col::All(v) => *v,
                Col::By(m) => m.get(&xp.player_key[i]).map_or(1.0, |v| v[xp.horizon[i] as usize]),
            })
        })
        .collect()
}

/// The frame with `p_play` and, optionally, `e_minutes`.
fn with_minutes(xp: &XpFrame, p_play: Col, e_minutes: Option<Col>) -> XpFrame {
    let mut out = xp.clone();
    out.p_play = Some(column(xp, &p_play));
    out.e_minutes = e_minutes.map(|c| column(xp, &c));
    out
}

fn squad_of_state(state: &SquadState) -> Vec<(i64, i64)> {
    state.holdings.iter().map(|h| (h.player_key, h.element_type)).collect()
}

/// Everything a plan decides plus its objective (no timings).
fn plan_summary(plan: &Plan) -> Vec<String> {
    let mut out: Vec<String> = plan
        .gws
        .iter()
        .map(|g| {
            format!(
                "{:?}",
                (
                    &g.transfers,
                    &g.starters,
                    &g.bench,
                    g.captain,
                    g.vice,
                    g.xp,
                    g.bench_xp,
                    g.objective
                )
            )
        })
        .collect();
    out.push(format!("{:?} {}", plan.objective, plan.status));
    out
}

fn weights(squad: &[(i64, i64)], frame: &XpFrame, horizons: &[i64]) -> Vec<Vec<f64>> {
    minutes_bench_weights(squad, frame, horizons, &rules(), &DEFAULT_BENCH_WEIGHTS)
        .expect("weights")
}

fn tail_with(first: f64, rest: Vec<f64>) -> Vec<f64> {
    let mut v = vec![first];
    v.extend(rest);
    v
}

// --- Poisson-binomial ---------------------------------------------------------------------

#[test]
fn poisson_binomial_tail_hand_values() {
    assert_approx_vec(&poisson_binomial_tail(&[0.1, 0.2], 3), &[0.28, 0.02, 0.0]);
    assert_approx_vec(&poisson_binomial_tail(&[0.5; 3], 3), &[0.875, 0.5, 0.125]);
    // 0.1, 0.2, 0.3: P(0) = 0.504, P(1) = 0.398, P(2) = 0.092, P(3) = 0.006.
    assert_approx_vec(&poisson_binomial_tail(&[0.1, 0.2, 0.3], 3), &[0.496, 0.098, 0.006]);
    assert_eq!(poisson_binomial_tail(&[], 2), vec![0.0, 0.0]);
    assert_approx_vec(&poisson_binomial_tail(&[1.0, 0.0, 1.0], 3), &[1.0, 1.0, 0.0]);
    assert_approx_vec(&poisson_binomial_tail(&[1.5, -0.2], 2), &[1.0, 0.0]); // clipped
}

#[test]
fn projected_xi_follows_the_formation_rules() {
    // 2 GK, 5 DEF, 5 MID, 3 FWD; DEFs score most, but at most 5 DEF and 1 GK start.
    let mut squad = vec![(1, 1), (2, 1)];
    squad.extend((0..5).map(|i| (10 + i, 2)));
    squad.extend((0..5).map(|i| (20 + i, 3)));
    squad.extend((0..3).map(|i| (30 + i, 4)));
    let mut xp: HashMap<i64, f64> = HashMap::from([(1, 1.0), (2, 9.0)]);
    xp.extend((0..5).map(|i| (10 + i, 8.0)));
    xp.extend((0..5).map(|i| (20 + i, i as f64)));
    xp.extend([(30, 0.5), (31, 0.4), (32, 0.3)]);
    let xi: BTreeSet<i64> = projected_xi(&squad, &xp, &rules()).into_iter().collect();
    assert_eq!(xi, BTreeSet::from([2, 10, 11, 12, 13, 14, 24, 23, 22, 30, 21]));
}

// --- bench weights ------------------------------------------------------------------------

#[test]
fn weights_from_the_projected_xi() {
    let (state, _, xp) = instance(0, 1, 5);
    let squad = squad_of_state(&state);
    let xi = projected_xi(&squad, &xp_at(&xp, 0), &rules());
    let et: HashMap<i64, i64> = squad.iter().copied().collect();
    let gk = *xi.iter().find(|k| et[k] == 1).unwrap();
    let outfield: Vec<i64> = xi.iter().copied().filter(|k| et[k] != 1).collect();
    let bench: Vec<i64> = squad.iter().map(|(k, _)| *k).filter(|k| !xi.contains(k)).collect();
    // Starters: GK plays w.p. 0.7, outfield starters w.p. 0.9, 0.8 and 1 for the rest.
    let mut p: HashMap<i64, Vec<f64>> =
        HashMap::from([(gk, vec![0.7]), (outfield[0], vec![0.9]), (outfield[1], vec![0.8])]);
    p.extend(outfield[2..].iter().map(|k| (*k, vec![1.0])));
    // Bench players who always play: no skips, slot k = P(at least k starters miss).
    let mut certain = p.clone();
    certain.extend(bench.iter().map(|k| (*k, vec![1.0])));
    let w = weights(&squad, &with_minutes(&xp, Col::By(certain), None), &[0]);
    assert_eq!(w.len(), 1);
    assert_approx_vec(&w[0], &[0.3, 0.28, 0.02, 0.0]);
    // Bench players who play w.p. 0.1: a later slot also comes on when an earlier one is
    // skipped (M = starters out: P(0) 0.72, P(1) 0.26, P(2) 0.02).
    let mut unlikely = p.clone();
    unlikely.extend(bench.iter().map(|k| (*k, vec![0.1])));
    let w = weights(&squad, &with_minutes(&xp, Col::By(unlikely), None), &[0]);
    let slot2 = 0.9 * 0.28 + 0.1 * 0.02;
    let slot3 = 0.81 * 0.28 + 0.18 * 0.02;
    assert_approx_vec(&w[0], &[0.3, 0.28, slot2, slot3]);
}

#[test]
fn skip_aware_tail_hand_values() {
    // Bench players certain to play: the plain Poisson-binomial tail.
    assert_approx_vec(&skip_aware_tail(&[0.1, 0.2], &[1.0, 1.0, 1.0]), &[0.28, 0.02, 0.0]);
    // Bench players who never play: every slot is needed as soon as one starter misses.
    assert_approx_vec(&skip_aware_tail(&[0.1, 0.2], &[0.0, 0.0, 0.0]), &[0.28, 0.28, 0.28]);
    assert_approx_vec(&skip_aware_tail(&[0.5], &[0.5, 0.5]), &[0.5, 0.25]);
    assert_eq!(skip_aware_tail(&[], &[0.3]), vec![0.0]);
}

#[test]
fn gk_slot_is_the_starting_keepers_risk_only() {
    let (state, _, xp) = instance(1, 2, 5);
    let squad = squad_of_state(&state);
    let keepers: Vec<i64> = squad.iter().filter(|(_, et)| *et == 1).map(|(k, _)| *k).collect();
    // Every outfielder certain to play; the keepers 0.6 / 0.4 in GW 0 and 1.0 / 0.5 in GW 1.
    let mut p: HashMap<i64, Vec<f64>> =
        squad.iter().filter(|(_, et)| *et != 1).map(|(k, _)| (*k, vec![1.0, 1.0])).collect();
    p.insert(keepers[0], vec![0.6, 1.0]);
    p.insert(keepers[1], vec![0.4, 0.5]);
    let frame = with_minutes(&xp, Col::By(p.clone()), None);
    let w = weights(&squad, &frame, &[0, 1]);
    for (t, wt) in w.iter().enumerate() {
        let xi = projected_xi(&squad, &xp_at(&frame, t as i64), &rules());
        let starter = *xi.iter().find(|k| keepers.contains(k)).unwrap();
        assert_approx(wt[0], 1.0 - p[&starter][t]);
        assert_approx_vec(&wt[1..], &[0.0, 0.0, 0.0]);
    }
}

#[test]
fn double_gw_uses_p_play_gw_and_unknown_falls_back_to_fixed_weights() {
    let (state, _, xp) = instance(2, 2, 5);
    let squad = squad_of_state(&state);
    let mut frame = with_minutes(&xp, Col::All(1.0), None);
    // GW 1 is a double: per-GW p_play is NaN, as in v1's frame.
    let double: Vec<bool> = frame.horizon.iter().map(|h| *h == 1).collect();
    for (i, d) in double.iter().enumerate() {
        if *d {
            frame.p_play.as_mut().unwrap()[i] = Some(f64::NAN);
        }
    }
    let w = weights(&squad, &frame, &[0, 1]);
    assert_approx_vec(&w[0], &[0.0, 0.0, 0.0, 0.0]);
    assert_eq!(w[1], DEFAULT_BENCH_WEIGHTS.to_vec()); // unknown: the fixed weights for that GW
    // With p_play_gw (P(plays in at least one fixture)) the double is weighted from it.
    let p_play = frame.p_play.clone().unwrap();
    frame.p_play_gw =
        Some(double.iter().zip(&p_play).map(|(d, p)| if *d { Some(0.5) } else { *p }).collect());
    let w = weights(&squad, &frame, &[0, 1]);
    assert_approx_vec(&w[1], &tail_with(0.5, skip_aware_tail(&[0.5; 10], &[0.5; 3])));
}

#[test]
fn rows_per_fixture_multiply_and_missing_players_never_play() {
    let frame = XpFrame {
        player_key: vec![1, 1, 2, 3],
        gw: vec![5; 4],
        gw_index: vec![5; 4],
        horizon: vec![0; 4],
        xp: vec![Some(1.0), Some(1.0), Some(2.0), Some(0.5)],
        p_play: Some(vec![Some(0.5), Some(0.6), Some(f64::NAN), Some(0.9)]),
        p_play_gw: None,
        e_minutes: None,
    };
    let q = not_play_probabilities(&frame).unwrap();
    assert_approx(q[&(1, 0)], 0.5 * 0.4); // plays in neither fixture
    assert!(q[&(2, 0)].is_nan());
    assert_approx(q[&(3, 0)], 0.1);
    assert!(not_play_probabilities(&XpFrame { p_play: None, ..frame }).is_none());
    // A starter missing from the frame (left the game) counts as not playing: with every xP
    // 0, the keeper with the smaller key starts (ties by key), rows or not.
    let (state, _, mut xp) = instance(3, 1, 5);
    let squad = squad_of_state(&state);
    xp.xp.iter_mut().for_each(|x| *x = Some(0.0));
    let frame = with_minutes(&xp, Col::All(1.0), None);
    let gk = squad.iter().filter(|(_, et)| *et == 1).map(|(k, _)| *k).min().unwrap();
    assert!(projected_xi(&squad, &HashMap::new(), &rules()).contains(&gk));
    let frame = filter_rows(&frame, |i| frame.player_key[i] != gk);
    let w = weights(&squad, &frame, &[0]);
    assert_eq!(w.len(), 1);
    assert_approx_vec(&w[0], &[1.0, 0.0, 0.0, 0.0]);
}

#[test]
fn projected_xi_sums_a_double_gws_rows() {
    // One row per fixture: the projected XI ranks players by the GW's summed xP.
    let (state, _, xp) = instance(4, 1, 5);
    let squad = squad_of_state(&state);
    let frame = with_minutes(&xp, Col::All(1.0), None);
    let target = xp_at(&frame, 0);
    let xi = projected_xi(&squad, &target, &rules());
    let et: HashMap<i64, i64> = squad.iter().copied().collect();
    let mut benched_rows: Vec<i64> =
        squad.iter().map(|(k, _)| *k).filter(|k| !xi.contains(k) && et[k] != 1).collect();
    // Python's min returns the first minimal one in squad order.
    let low = benched_rows.iter().map(|k| target[k]).fold(f64::INFINITY, f64::min);
    benched_rows.retain(|k| target[k] == low);
    let benched = benched_rows[0];
    // Give the benched player two fixture rows, xP above every starter's and 0: on the sum he
    // starts (on the last row alone he would not), and P(play) multiplies over the rows.
    let top = target.values().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut split = filter_rows(&frame, |i| frame.player_key[i] != benched);
    let row = (0..frame.len()).find(|&i| frame.player_key[i] == benched).unwrap();
    for x in [top + 1.0, 0.0] {
        split.player_key.push(benched);
        split.gw.push(frame.gw[row]);
        split.gw_index.push(frame.gw_index[row]);
        split.horizon.push(frame.horizon[row]);
        split.xp.push(Some(x));
        split.p_play.as_mut().unwrap().push(Some(0.5));
    }
    let w = weights(&squad, &split, &[0]);
    // The benched outfielder now starts (q = 0.25); everyone else always plays.
    assert_eq!(w.len(), 1);
    assert_approx_vec(&w[0], &[0.0, 0.25, 0.0, 0.0]);
}

#[test]
fn weights_never_increase_down_the_bench_and_blanks_never_play() {
    let mut rng = Rng::new(0);
    let (state, _, xp) = instance(5, 3, 5);
    let squad = squad_of_state(&state);
    for _ in 0..20 {
        let mut p: HashMap<i64, Vec<f64>> =
            squad.iter().map(|(k, _)| (*k, (0..3).map(|_| rng.random()).collect())).collect();
        // Several incumbents blank in GW 1 (P(play) 0): they never start or come on.
        for (k, _) in &squad[..4] {
            p.get_mut(k).unwrap()[1] = 0.0;
        }
        for w in weights(&squad, &with_minutes(&xp, Col::By(p), None), &[0, 1, 2]) {
            assert!(w.iter().all(|v| (0.0..=1.0).contains(v)));
            assert!(w[1] >= w[2] - 1e-12 && w[2] - 1e-12 >= w[3] - 2e-12, "{w:?}");
        }
    }
}

// --- the planner --------------------------------------------------------------------------

#[test]
fn without_p_play_the_fixed_weights_and_plans_are_unchanged() {
    // Baseline-model frames (no p_play, no e_minutes): the fixed weights, and the minutes
    // settings change nothing. (Python also pins the plan to a golden from the Phase 4 code
    // on its numpy-seeded instance; that instance can't be rebuilt here.)
    let rules = rules();
    let (state, pool, xp) = instance(3, 3, 5);
    let params = OptimizerParams { horizon: 3, ..Default::default() };
    let p = problem(&state, &pool, &xp, &params, &rules);
    assert!(p.bench_weights.is_none());
    assert_eq!(p.gw_bench_weights(1, &params), DEFAULT_BENCH_WEIGHTS);
    let plan = solve(&p, &params);
    let off = OptimizerParams { bench_from_minutes: false, min_minutes: 60.0, ..params.clone() };
    let same = solve(&problem(&state, &pool, &xp, &off, &rules), &off);
    assert_eq!(plan_summary(&same), plan_summary(&plan));
    // A frame with minutes, with the minutes features off, plans as the baseline frame.
    let minutes = with_minutes(&xp, Col::All(0.5), Some(Col::All(0.0)));
    let off = OptimizerParams { bench_from_minutes: false, min_minutes: 0.0, ..params.clone() };
    let ignored = solve(&problem(&state, &pool, &minutes, &off, &rules), &off);
    assert_eq!(plan_summary(&ignored), plan_summary(&plan));
}

fn bench_xp_of(frame: &XpFrame, t: usize, bench: &[i64], w: &[f64]) -> f64 {
    let values = xp_at(frame, t as i64);
    w.iter().zip(bench).map(|(w, k)| w * values[k]).sum()
}

#[test]
fn per_gw_weights_reach_the_objective() {
    let rules = rules();
    let (state, pool, xp) = instance(4, 2, 5);
    let squad = squad_of_state(&state);
    let owned: Vec<i64> = squad.iter().map(|(k, _)| *k).collect();
    let pool: Vec<PoolPlayer> =
        pool.into_iter().filter(|p| owned.contains(&p.player_key)).collect(); // no transfers
    let xp = filter_rows(&xp, |i| owned.contains(&xp.player_key[i]));
    let p: HashMap<i64, Vec<f64>> = owned.iter().map(|k| (*k, vec![0.9, 0.6])).collect();
    let frame = with_minutes(&xp, Col::By(p), None);
    let params = OptimizerParams { horizon: 2, ..exact() };
    let pi = problem(&state, &pool, &frame, &params, &rules);
    let w = pi.bench_weights.clone().expect("minutes-based weights");
    assert_ne!(w[0], w[1]);
    // Bench outfielders play w.p. 0.9 / 0.6 too (skip rule, `skip_aware_tail`).
    assert_approx_vec(&w[0], &tail_with(0.1, skip_aware_tail(&[0.1; 10], &[0.9; 3])));
    assert_approx_vec(&w[1], &tail_with(0.4, skip_aware_tail(&[0.4; 10], &[0.6; 3])));
    let plan = solve(&pi, &params);
    let et: HashMap<i64, i64> = squad.iter().copied().collect();
    for (t, g) in plan.gws.iter().enumerate() {
        assert_approx(g.bench_xp, bench_xp_of(&frame, t, &g.bench, &w[t]));
        // The MILP's lineup is the best for this GW's weights.
        let best = best_lineup_value(&owned, &xp_at(&frame, t as i64), &et, &w[t]);
        assert_approx(g.xp + g.bench_xp, best);
    }
    // Different weights than the fixed ones: the plan values the bench differently.
    let fixed = solve(&problem(&state, &pool, &xp, &params, &rules), &params);
    assert!(!approx(fixed.objective, plan.objective, 1e-12));
}

#[test]
fn bench_boost_weights_stay_one() {
    let rules = rules();
    let (state, pool, xp) = instance(5, 2, 5);
    let frame = with_minutes(&xp, Col::All(0.8), None);
    let params = OptimizerParams { horizon: 2, ..exact() };
    let pi = problem(&state, &pool, &frame, &params, &rules);
    let w = pi.bench_weights.clone().expect("minutes-based weights");
    let plan = solve_plan(&pi, &params, None, &[], &scenario(&pi, &[(0, "bboost")])).unwrap();
    let g = &plan.gws[0];
    assert_eq!(g.chip.as_deref(), Some("bboost"));
    assert_approx(g.bench_xp, bench_xp_of(&frame, 0, &g.bench, &[1.0; 4]));
    // GW 1 has no chip: minutes-based weights.
    assert_approx(plan.gws[1].bench_xp, bench_xp_of(&frame, 1, &plan.gws[1].bench, &w[1]));
    // The Bench Boost bound of the chip search uses the GW's own weights (GW 1: decayed).
    let keepers = pi
        .players
        .iter()
        .filter(|p| p.element_type == 1)
        .map(|p| p.xp[1].abs())
        .fold(f64::MIN, f64::max);
    let mut outfield: Vec<f64> =
        pi.players.iter().filter(|p| p.element_type != 1).map(|p| p.xp[1].abs()).collect();
    outfield.sort_by(|a, b| b.total_cmp(a));
    let mut slots: Vec<f64> = w[1][1..].iter().map(|x| (1.0 - x).abs()).collect();
    slots.sort_by(|a, b| b.total_cmp(a));
    let expected =
        (1.0 - w[1][0]) * keepers + slots.iter().zip(&outfield).map(|(a, x)| a * x).sum::<f64>();
    assert_approx(chip_extra(&pi, &params, 1, "bboost"), params.decay * expected);
    let fixed = PlanInput { bench_weights: None, ..pi.clone() };
    assert!(!approx(chip_extra(&fixed, &params, 1, "bboost"), params.decay * expected, 1e-12));
}

#[test]
fn rebuilt_squads_use_the_fixed_weights() {
    let rules = rules();
    // GW1's unlimited transfers rebuild the squad: no minutes-based weights.
    let (state, pool, xp) = instance(6, 3, 1);
    let params = OptimizerParams { horizon: 3, ..exact() };
    let pi = problem(&state, &pool, &with_minutes(&xp, Col::All(0.8), None), &params, &rules);
    assert!(pi.gw1 && pi.bench_weights.is_none());
    // A Free Hit's GW and every GW from a Wildcard on use the fixed weights; others the
    // minutes-based ones.
    let (state, pool, xp) = instance(6, 3, 5);
    let frame = with_minutes(&xp, Col::All(0.8), None);
    let pi = problem(&state, &pool, &frame, &params, &rules);
    let w = pi.bench_weights.clone().expect("minutes-based weights");
    for (chip, rebuilt) in [("freehit", vec![1]), ("wildcard", vec![1, 2])] {
        let plan = solve_plan(&pi, &params, None, &[], &scenario(&pi, &[(1, chip)])).unwrap();
        for (t, (g, minutes)) in plan.gws.iter().zip(&w).enumerate() {
            let weights = if rebuilt.contains(&t) { &params.bench_weights } else { minutes };
            let expected = bench_xp_of(&frame, t, &g.bench, weights);
            assert!(approx(g.bench_xp, expected, 1e-12), "{chip} GW {t}");
        }
    }
}

// --- minutes floor ------------------------------------------------------------------------

fn kept(
    state: &SquadState,
    pool: &[PoolPlayer],
    frame: &XpFrame,
    params: &OptimizerParams,
) -> BTreeSet<i64> {
    problem(state, pool, frame, params, &rules()).players.iter().map(|p| p.player_key).collect()
}

#[test]
fn floor_drops_only_non_owned_low_minute_players() {
    let (state, pool, xp) = instance(6, 2, 5);
    let owned: BTreeSet<i64> = state.player_keys().into_iter().collect();
    let others: Vec<i64> =
        pool.iter().map(|p| p.player_key).filter(|k| !owned.contains(k)).collect();
    let low_owned: Vec<i64> = owned.iter().take(3).copied().collect();
    let low_others: Vec<i64> = others[..5].to_vec();
    // Expected minutes per GW: 20 for the "low" players (40 over the horizon), 80 for the rest.
    let minutes: HashMap<i64, Vec<f64>> = pool
        .iter()
        .map(|p| {
            let low = low_owned.contains(&p.player_key) || low_others.contains(&p.player_key);
            (p.player_key, if low { vec![20.0, 20.0] } else { vec![80.0, 80.0] })
        })
        .collect();
    let frame = with_minutes(&xp, Col::All(1.0), Some(Col::By(minutes)));
    let params = OptimizerParams { horizon: 2, min_minutes: 60.0, ..exact() };
    let k = kept(&state, &pool, &frame, &params);
    assert!(low_owned.iter().all(|x| k.contains(x))); // owned players are always candidates
    assert!(low_others.iter().all(|x| !k.contains(x)));
    assert!(others[5..].iter().all(|x| k.contains(x)));
    // No floor (0), a floor below their minutes, or a frame without e_minutes: all kept.
    let all: BTreeSet<i64> = pool.iter().map(|p| p.player_key).collect();
    let no_minutes = XpFrame { e_minutes: None, ..frame.clone() };
    for (p, f) in [
        (OptimizerParams { min_minutes: 0.0, ..params.clone() }, &frame),
        (OptimizerParams { min_minutes: 40.0, ..params.clone() }, &frame),
        (params.clone(), &no_minutes),
    ] {
        assert_eq!(kept(&state, &pool, f, &p), all);
    }
}

#[test]
fn floor_runs_before_top_n() {
    // A floored player doesn't take a top-N slot: the next best player gets it.
    let (state, pool, xp) = instance(7, 1, 5);
    let owned = state.player_keys();
    let values = xp_at(&xp, 0);
    let mut ranked: Vec<i64> = pool
        .iter()
        .filter(|p| p.element_type == 4 && !owned.contains(&p.player_key))
        .map(|p| p.player_key)
        .collect();
    ranked.sort_by(|a, b| values[b].total_cmp(&values[a]).then(a.cmp(b)));
    let best = ranked[0];
    let minutes: HashMap<i64, Vec<f64>> = pool
        .iter()
        .map(|p| (p.player_key, vec![if p.player_key == best { 0.0 } else { 90.0 }]))
        .collect();
    let frame = with_minutes(&xp, Col::All(1.0), Some(Col::By(minutes)));
    let params = OptimizerParams {
        horizon: 1,
        prune_n: Some([(1, 0), (2, 0), (3, 0), (4, 1)].into_iter().collect()),
        min_minutes: 30.0,
        ..exact()
    };
    let k = kept(&state, &pool, &frame, &params);
    assert!(!k.contains(&best));
    assert!(k.contains(&ranked[1]));
}

#[test]
fn min_minutes_validation() {
    let err = OptimizerParams { min_minutes: -1.0, ..Default::default() }.validate().unwrap_err();
    assert!(err.to_string().contains("min_minutes"), "{err}");
    assert_eq!(OptimizerParams::default().min_minutes, 180.0);
}

#[test]
fn nan_expected_minutes_raise() {
    let frame = XpFrame {
        player_key: vec![1, 2],
        gw: vec![5, 5],
        gw_index: vec![5, 5],
        horizon: vec![0, 0],
        xp: vec![Some(1.0), Some(2.0)],
        p_play: None,
        p_play_gw: None,
        e_minutes: Some(vec![Some(90.0), Some(f64::NAN)]),
    };
    let err = horizon_minutes(&frame, &[0], &[1, 2]).unwrap_err();
    assert!(err.to_string().contains("NaN e_minutes"), "{err}");
    // A player without rows (left the game) counts 0.
    let first = take_rows(&frame, &[0]);
    assert_eq!(horizon_minutes(&first, &[0], &[1, 2]).unwrap(), Some(vec![90.0, 0.0]));
}

#[test]
fn floor_is_pro_rata_on_a_short_horizon() {
    // The floor is for a full horizon: with 2 of 6 GWs left, 180 becomes 60.
    let (state, pool, xp) = instance(8, 2, 5);
    let owned = state.player_keys();
    let others: Vec<i64> =
        pool.iter().map(|p| p.player_key).filter(|k| !owned.contains(k)).collect();
    // 25 or 35 expected minutes per GW: 50 / 70 over the 2 GWs left.
    let minutes: HashMap<i64, Vec<f64>> = pool
        .iter()
        .map(|p| {
            (
                p.player_key,
                if others[..4].contains(&p.player_key) { vec![25.0; 2] } else { vec![35.0; 2] },
            )
        })
        .collect();
    let frame = with_minutes(&xp, Col::All(1.0), Some(Col::By(minutes)));
    let params = OptimizerParams { horizon: 6, min_minutes: 180.0, ..exact() };
    let k = kept(&state, &pool, &frame, &params);
    assert!(others[..4].iter().all(|x| !k.contains(x)));
    assert!(others[4..].iter().chain(&owned).all(|x| k.contains(x)));
}
