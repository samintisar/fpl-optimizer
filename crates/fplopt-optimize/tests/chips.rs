//! Chip scenarios, chip terms in the MILP, top-k plans and the roll plan (Python
//! `tests/test_optimize_chips.py`).

mod common;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use common::*;
use fplopt_build::scoring::Rules;
use fplopt_optimize::chips::{chip_value, make_scenario};
use fplopt_optimize::state::{PoolPlayer, SquadState, next_state, selling_price};
use fplopt_optimize::{
    ChipScenario, OptimizerParams, PlanInput, Search, XpFrame, optimize, scenarios, solve_plan,
    terminal_value,
};

// 2026-27 rules: two sets of every chip, set 1 on gw_index 1/2..19, set 2 on 20..38; ids WC
// 1/2, FH 3/6, BB 4/7, TC 5/8; FTs `retain` after WC/FH.
const WC1: i64 = 1;
const WC2: i64 = 2;
const FH1: i64 = 3;
const BB1: i64 = 4;
const TC1: i64 = 5;
const FH2: i64 = 6;
const BB2: i64 = 7;
const TC2: i64 = 8;

fn exact() -> OptimizerParams {
    OptimizerParams {
        mip_gap: 0.0,
        prune_n: None,
        prune_dominated: false,
        tie_epsilon: 0.0,
        ..OptimizerParams::default()
    }
}

fn values(pairs: &[(&str, f64)]) -> BTreeMap<String, f64> {
    pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
}

/// Exact params where every chip costs 100 to play (its terminal value) unless `chip_value`
/// says otherwise, so one chip at a time can be made attractive.
fn chip_params(n_gws: i64, chip_value: &[(&str, f64)]) -> OptimizerParams {
    let mut table =
        values(&[("wildcard", 100.0), ("freehit", 100.0), ("bboost", 100.0), ("3xc", 100.0)]);
    table.extend(values(chip_value));
    OptimizerParams { horizon: n_gws, chip_value: table, ..exact() }
}

/// A PlanInput over an xP frame of `n_gws` GWs from `gw_index` (only the state and the horizon
/// matter for scenario enumeration).
fn problem_at(
    gw_index: i64,
    n_gws: usize,
    chips_used: &[(i64, i64)],
    rules: &Rules,
    horizon: Option<i64>,
) -> PlanInput {
    let pool = grid_pool(1..=6, &GRID, None);
    let mut state = state_with(&pool, &first_valid_squad(&pool), gw_index, 1);
    state.chips_used = chips_used.to_vec();
    let xp = correlated_xp(&pool, n_gws, 0, gw_index);
    let params = OptimizerParams { horizon: horizon.unwrap_or(n_gws as i64), ..exact() };
    problem(&state, &pool, &xp, &params, rules)
}

fn labels(problem: &PlanInput) -> Vec<String> {
    scenarios(problem).into_iter().map(|s| s.label).collect()
}

struct Team {
    owned_xp: Vec<f64>,
    other_xp: Option<Vec<f64>>,
    special: BTreeMap<i64, Vec<f64>>,
    ft: i64,
    gw_index: i64,
    purchase: HashMap<i64, i64>,
}

impl Team {
    fn new(owned_xp: &[f64]) -> Team {
        Team {
            owned_xp: owned_xp.to_vec(),
            other_xp: None,
            special: BTreeMap::new(),
            ft: 1,
            gw_index: 5,
            purchase: HashMap::new(),
        }
    }

    /// A grid pool (6 clubs, 13 players each) with the cheapest valid squad owned; owned
    /// players score `owned_xp`, the others `other_xp` (default 0) per GW, `special`
    /// overrides single players.
    fn build(self) -> (SquadState, Vec<PoolPlayer>, XpFrame, Vec<i64>) {
        let n_gws = self.owned_xp.len();
        let pool = grid_pool(1..=6, &GRID, None);
        let owned = first_valid_squad(&pool);
        let other = self.other_xp.unwrap_or_else(|| vec![0.0; n_gws]);
        let mut values: BTreeMap<i64, Vec<f64>> =
            pool.iter().map(|p| (p.player_key, other.clone())).collect();
        for k in &owned {
            values.insert(*k, self.owned_xp.clone());
        }
        values.extend(self.special);
        let state = make_state(
            &pool,
            &owned,
            StateOpts {
                ft: self.ft,
                gw_index: self.gw_index,
                purchase: self.purchase,
                ..Default::default()
            },
        );
        let xp = xp_frame(&values, self.gw_index);
        (state, pool, xp, owned)
    }
}

// --- scenario enumeration -----------------------------------------------------------------

#[test]
fn scenarios_mid_season_singles_and_pairs() {
    let rules = rules();
    let problem = problem_at(5, 3, &[], &rules, None);
    let got = labels(&problem);
    let names = ["3xc", "bboost", "freehit", "wildcard"];
    let singles: Vec<String> =
        [5, 6, 7].iter().flat_map(|g| names.iter().map(move |n| format!("{n}@gw{g}"))).collect();
    let mut pairs = Vec::new();
    for (g1, g2) in [(5, 6), (5, 7), (6, 7)] {
        for a in names {
            for b in names {
                if a != b {
                    // a second window of the same chip isn't open before GW20
                    pairs.push(format!("{a}@gw{g1}+{b}@gw{g2}"));
                }
            }
        }
    }
    assert_eq!(got[0], "none");
    assert_eq!(got[1..13], singles[..]);
    let mut rest = got[13..].to_vec();
    rest.sort();
    pairs.sort();
    assert_eq!(rest, pairs);
    assert_eq!(got.len(), 1 + 12 + 36);
    assert_eq!(got.iter().collect::<HashSet<_>>().len(), got.len());
    // Pairs come in (first position, first name, second position, second name) order.
    let keys: Vec<Vec<(usize, String)>> =
        scenarios(&problem)[13..].iter().map(|s| s.chips.clone()).collect();
    let mut sorted = keys.clone();
    sorted.sort();
    assert_eq!(keys, sorted);
}

#[test]
fn scenarios_skip_used_chips_and_record_chip_ids() {
    let rules = rules();
    let problem = problem_at(5, 3, &[(WC1, 3), (TC1, 4)], &rules, None);
    let found = scenarios(&problem);
    let first: Vec<&str> = found[..7].iter().map(|s| s.label.as_str()).collect();
    assert_eq!(
        first,
        [
            "none",
            "bboost@gw5",
            "freehit@gw5",
            "bboost@gw6",
            "freehit@gw6",
            "bboost@gw7",
            "freehit@gw7"
        ]
    );
    assert_eq!(found.len(), 1 + 6 + 6);
    let s = found.iter().find(|s| s.label == "freehit@gw5+bboost@gw7").unwrap();
    assert_eq!(s.chip_ids, vec![FH1, BB1]);
    assert_eq!(s.by_t(), BTreeMap::from([(0, "freehit"), (2, "bboost")]));
}

#[test]
fn free_hits_across_the_set_boundary() {
    for consecutive in [false, true] {
        let rules = Rules { freehit_consecutive: consecutive, ..common::rules() };
        let problem = problem_at(18, 4, &[], &rules, None); // GWs 18-21: set 1 ends at 19
        let found = scenarios(&problem);
        let by_label: HashMap<&str, &ChipScenario> =
            found.iter().map(|s| (s.label.as_str(), s)).collect();
        assert_eq!(by_label["freehit@gw18+freehit@gw20"].chip_ids, vec![FH1, FH2]);
        assert_eq!(by_label.contains_key("freehit@gw19+freehit@gw20"), consecutive);
        assert_eq!(by_label["wildcard@gw19+wildcard@gw20"].chip_ids, vec![WC1, WC2]);
        assert!(!by_label.contains_key("wildcard@gw18+wildcard@gw19")); // one set-1 wildcard
        // A Free Hit played the GW before the deadline blocks one now (unless allowed).
        let after_fh = labels(&problem_at(20, 2, &[(FH1, 19)], &rules, None));
        assert_eq!(after_fh.contains(&"freehit@gw20".to_string()), consecutive);
        assert!(after_fh.contains(&"freehit@gw21".to_string()));
        assert!(!after_fh.contains(&"freehit@gw20+freehit@gw21".to_string())); // one set-2 FH
    }
}

fn chips(pairs: &[(usize, &str)]) -> BTreeMap<usize, String> {
    pairs.iter().map(|(t, n)| (*t, n.to_string())).collect()
}

#[test]
fn make_scenario_rejects_what_the_rules_forbid() {
    let rules = rules();
    let problem = problem_at(5, 3, &[(WC1, 3)], &rules, None);
    let err = make_scenario(&problem, &chips(&[(0, "wildcard")])).unwrap_err();
    assert!(err.0.contains("already used"), "{err}");
    let err = make_scenario(&problem, &chips(&[(0, "bboost"), (2, "bboost")])).unwrap_err();
    assert!(err.0.contains("already used"), "{err}");
    assert_eq!(make_scenario(&problem, &BTreeMap::new()).unwrap(), ChipScenario::none());
}

#[test]
#[should_panic(expected = "outside the horizon")]
fn make_scenario_rejects_positions_outside_the_horizon() {
    // Python raises ValueError; the port treats it as a programming error (panic).
    let rules = rules();
    let problem = problem_at(5, 3, &[(WC1, 3)], &rules, None);
    let _ = make_scenario(&problem, &chips(&[(3, "bboost")]));
}

#[test]
fn terminal_values() {
    let rules = rules();
    let params = OptimizerParams {
        chip_value: values(&[("wildcard", 6.0), ("freehit", 4.0), ("bboost", 2.0)]),
        ..exact()
    };
    let early = problem_at(5, 3, &[], &rules, None); // GWs 5-7: every window extends past GW7
    let none = ChipScenario::none();
    assert_eq!(terminal_value(&early, &params, &none), 2.0 * (6.0 + 4.0 + 2.0 + 0.0));
    assert_eq!(terminal_value(&early, &params, &scenario(&early, &[(1, "bboost")])), 24.0 - 2.0);
    // GWs 17-19: set 1 closes inside the horizon → worth 0 whether played or not.
    let closing = problem_at(17, 3, &[], &rules, None);
    assert_eq!(terminal_value(&closing, &params, &none), 12.0);
    assert_eq!(terminal_value(&closing, &params, &scenario(&closing, &[(0, "wildcard")])), 12.0);
    // The state's used chips don't count.
    let used = problem_at(5, 3, &[(WC1, 2), (WC2, 21)], &rules, None);
    assert_eq!(terminal_value(&used, &params, &none), 24.0 - 12.0);
}

#[test]
fn horizon_near_season_end() {
    let rules = rules();
    let problem = problem_at(36, 3, &[], &rules, Some(6)); // only GWs 36-38 in the xP frame
    assert_eq!(problem.gws.iter().map(|g| g.gw_index).collect::<Vec<_>>(), vec![36, 37, 38]);
    let found = scenarios(&problem);
    assert!(found.iter().all(|s| terminal_value(&problem, &OptimizerParams::default(), s) == 0.0));
    let singles: Vec<&ChipScenario> = found.iter().filter(|s| s.chips.len() == 1).collect();
    assert_eq!(singles.len(), 12);
    let ids: BTreeSet<i64> = singles.iter().map(|s| s.chip_ids[0]).collect();
    assert_eq!(ids, BTreeSet::from([WC2, FH2, BB2, TC2]));
}

// --- each chip on a hand-built instance ----------------------------------------------------

fn best_scenario(
    state: &SquadState,
    pool: &[PoolPlayer],
    xp: &XpFrame,
    params: &OptimizerParams,
) -> String {
    let problem = problem(state, pool, xp, params, &rules());
    optimize(&problem, params, 1, true, true, Search::Bound).unwrap().best().scenario.clone()
}

#[test]
fn bench_boost_when_the_bench_scores() {
    let rules = rules();
    let (state, pool, xp, _) = Team::new(&[2.0, 6.0]).build();
    let params = chip_params(2, &[("bboost", 1.0)]);
    let problem = problem(&state, &pool, &xp, &params, &rules);
    let plans = optimize(&problem, &params, 1, true, true, Search::Bound).unwrap();
    let best = plans.best();
    assert_eq!(best.scenario, "bboost@gw6");
    let g = &best.gws[1];
    assert_eq!(g.chip.as_deref(), Some("bboost"));
    assert_approx(g.bench_xp, 4.0 * 6.0);
    assert_approx(best.terminal_value, 6.0 * 100.0 + 1.0); // BB1 used, BB2 held
    replay(&state, best, &pool, &rules);
    // Worth 0.85 · 24 = 20.4 now; holding the chip is worth more → no chip.
    assert_eq!(best_scenario(&state, &pool, &xp, &chip_params(2, &[("bboost", 25.0)])), "none");
}

#[test]
fn triple_captain_on_the_big_gw() {
    let rules = rules();
    let (state, pool, mut xp, owned) = Team::new(&[2.0, 2.0]).build();
    let star = *owned.last().unwrap(); // a forward
    set_xp(&mut xp, star, &[5.0, 15.0]);
    let params = chip_params(2, &[("3xc", 3.0)]);
    let problem = problem(&state, &pool, &xp, &params, &rules);
    let plans = optimize(&problem, &params, 1, true, true, Search::Bound).unwrap();
    let best = plans.best();
    assert_eq!(best.scenario, "3xc@gw6");
    let g = &best.gws[1];
    assert_eq!(g.captain, star);
    assert_approx(g.xp, 10.0 * 2.0 + 3.0 * 15.0);
    let no_chip = solve(&problem, &params);
    // TC adds the captain's xP once more, decayed: 0.85 · 15; the terminal value drops by 3.
    assert_approx(best.objective - no_chip.objective, 0.85 * 15.0);
    assert_approx(no_chip.terminal_value - best.terminal_value, 3.0);
    replay(&state, best, &pool, &rules);
    assert_eq!(best_scenario(&state, &pool, &xp, &chip_params(2, &[("3xc", 13.0)])), "none");
}

#[test]
fn wildcard_rebuilds_for_free_and_keeps_the_fts() {
    let rules = rules();
    let team = Team { other_xp: Some(vec![5.0, 5.0]), ft: 2, ..Team::new(&[1.0, 1.0]) };
    let (state, pool, xp, _) = team.build();
    let params = chip_params(2, &[("wildcard", 6.0)]);
    let problem = problem(&state, &pool, &xp, &params, &rules);
    let plans = optimize(&problem, &params, 1, true, true, Search::Bound).unwrap();
    let best = plans.best();
    assert_eq!(best.scenario, "wildcard@gw5");
    let first = best.first();
    assert!(
        first.chip.as_deref() == Some("wildcard") && first.hits == 0 && first.n_transfers() >= 10
    );
    assert_eq!(best.gws[1].ft, 2); // chip_week_ft "retain": the 2 FTs carry over
    let end = replay(&state, best, &pool, &rules);
    assert_eq!(end.chips_used, vec![(WC1, 5)]);
    // The roll plan doesn't waste the wildcard on no transfers.
    let roll = plans.roll.as_ref().unwrap();
    assert!(roll.first().n_transfers() == 0 && roll.first().chip.is_none());
    assert!(best.gain_vs_roll.unwrap() > 0.0);
    // Holding the wildcard is worth more than the rebuild → no chip, hits instead.
    assert_eq!(best_scenario(&state, &pool, &xp, &chip_params(2, &[("wildcard", 200.0)])), "none");
}

#[test]
fn wildcard_under_retain_plus_one() {
    let rules = Rules { chip_week_ft: "retain_plus_one".to_string(), ..common::rules() };
    let team = Team { other_xp: Some(vec![5.0, 5.0]), ft: 2, ..Team::new(&[1.0, 1.0]) };
    let (state, pool, xp, _) = team.build();
    let params = chip_params(2, &[("wildcard", 0.0)]);
    let problem = problem(&state, &pool, &xp, &params, &rules);
    let plan =
        solve_plan(&problem, &params, None, &[], &scenario(&problem, &[(0, "wildcard")])).unwrap();
    assert!(plan.first().hits == 0 && plan.gws[1].ft == 3);
    replay(&state, &plan, &pool, &rules);
}

#[test]
fn free_hit_for_one_big_gw_reverts() {
    // Owned players score 1; the market scores 8 in GW6 only, and one MID is worth buying for
    // good in GW5. Some owned players were bought cheaper (selling price < price).
    let rules = rules();
    let pool_probe = grid_pool(1..=6, &GRID, None);
    let owned_probe = first_valid_squad(&pool_probe);
    let mut per_club: HashMap<i64, i64> = HashMap::new();
    for k in &owned_probe {
        *per_club.entry(club_of(*k)).or_default() += 1;
    }
    let keeper = pool_probe
        .iter()
        .map(|p| p.player_key)
        .find(|k| {
            !owned_probe.contains(k)
                && et_of(*k) == 3
                && per_club.get(&club_of(*k)).copied().unwrap_or(0) < rules.team_limit
        })
        .unwrap(); // a MID from a club with room: one FT buys him
    let purchase: HashMap<i64, i64> = owned_probe.iter().step_by(4).map(|&k| (k, 40)).collect();
    let team = Team {
        other_xp: Some(vec![0.0, 8.0, 0.0]),
        special: BTreeMap::from([(keeper, vec![10.0, 10.0, 10.0])]),
        purchase,
        ..Team::new(&[1.0, 1.0, 1.0])
    };
    let (state, pool, xp, _) = team.build();
    let params = chip_params(3, &[("freehit", 4.0)]);
    let problem = problem(&state, &pool, &xp, &params, &rules);
    let plans = optimize(&problem, &params, 1, true, true, Search::Bound).unwrap();
    let best = plans.best();
    assert_eq!(best.scenario, "freehit@gw6");
    let (g0, g1, g2) = (&best.gws[0], &best.gws[1], &best.gws[2]);
    assert!(transfers_in(g0) == vec![keeper] && g0.chip.is_none());
    assert!(g1.chip.as_deref() == Some("freehit") && g1.hits == 0 && g1.n_transfers() >= 10);
    assert!(g1.ft == 1 && g2.ft == 1); // "retain": the FT used in GW5 regenerates once
    // The squad comes back after the Free Hit, and so does the bank.
    assert!(squad_of(g2) == squad_of(g0) && g2.n_transfers() == 0);
    assert_eq!(g2.bank, g0.bank);
    // FH sales: never-sold owned players at their selling price, the GW5 buy at his price.
    let price: HashMap<i64, i64> = pool.iter().map(|p| (p.player_key, p.price)).collect();
    let mut value: HashMap<i64, i64> = state
        .holdings
        .iter()
        .map(|h| (h.player_key, selling_price(h, rules.sell_on_fee)))
        .collect();
    value.insert(keeper, price[&keeper]);
    assert!(transfers_out(g1).iter().any(|k| value[k] < price[k]));
    let expected = g0.bank + transfers_out(g1).iter().map(|k| value[k]).sum::<i64>()
        - transfers_in(g1).iter().map(|k| price[k]).sum::<i64>();
    assert_eq!(g1.bank, expected);
    let end = replay(&state, best, &pool, &rules);
    assert_eq!(end.chips_used, vec![(FH1, 6)]);
    assert_eq!(end.bank, g2.bank);
    assert_eq!(end.player_keys().into_iter().collect::<BTreeSet<_>>(), squad_of(g2));
}

#[test]
fn free_hit_in_the_first_gw() {
    let rules = rules();
    let team = Team { other_xp: Some(vec![8.0, 0.0]), ..Team::new(&[1.0, 1.0]) };
    let (state, pool, xp, _) = team.build();
    let params = chip_params(2, &[("freehit", 4.0)]);
    let problem = problem(&state, &pool, &xp, &params, &rules);
    let plans = optimize(&problem, &params, 2, true, true, Search::Bound).unwrap();
    let best = plans.best();
    assert_eq!(best.scenario, "freehit@gw5");
    let d = decision(best.first());
    assert!(d.chip.as_deref() == Some("freehit") && d.transfers.len() >= 10);
    let (gw_state, record) = apply_decision(&state, &d, &pool, &rules).unwrap();
    assert_eq!(gw_state.freehit_backup.as_ref(), Some(&state.holdings));
    let after = next_state(&gw_state, &record.transfer_record(), &rules, 6).unwrap();
    assert!(after.holdings == state.holdings && after.bank == state.bank);
    assert_eq!(squad_of(&best.gws[1]), state.player_keys().into_iter().collect());
    // Plan #2 is another Free Hit squad (cuts act on the FH transfers); the roll plays no FH.
    assert_eq!(plans.plans[1].scenario, "freehit@gw5");
    assert_ne!(plans.plans[1].first().transfer_set(), best.first().transfer_set());
    let roll = plans.roll.as_ref().unwrap();
    assert!(roll.first().n_transfers() == 0 && roll.first().chip.is_none());
    replay(&state, best, &pool, &rules);
}

#[test]
fn a_chip_pair() {
    let rules = rules();
    let (state, pool, mut xp, owned) = Team::new(&[2.0, 6.0]).build();
    let star = *owned.last().unwrap();
    set_xp(&mut xp, star, &[15.0, 6.0]);
    let params = chip_params(2, &[("bboost", 1.0), ("3xc", 1.0)]);
    let problem = problem(&state, &pool, &xp, &params, &rules);
    let plans = optimize(&problem, &params, 1, true, true, Search::Bound).unwrap();
    let best = plans.best();
    assert_eq!(best.scenario, "3xc@gw5+bboost@gw6");
    let chips: Vec<Option<&str>> = best.gws.iter().map(|g| g.chip.as_deref()).collect();
    assert_eq!(chips, vec![Some("3xc"), Some("bboost")]);
    let end = replay(&state, best, &pool, &rules);
    assert_eq!(end.chips_used, vec![(TC1, 5), (BB1, 6)]);
}

#[test]
fn chip_on_gw1_keeps_the_gw1_rules() {
    let rules = rules();
    let (state, pool, xp, _) = Team { gw_index: 1, ..Team::new(&[2.0, 6.0]) }.build();
    let params = chip_params(2, &[("bboost", 0.0)]);
    let problem = problem(&state, &pool, &xp, &params, &rules);
    let plan =
        solve_plan(&problem, &params, None, &[], &scenario(&problem, &[(0, "bboost")])).unwrap();
    assert!(plan.first().hits == 0 && plan.gws[1].ft == 1);
    replay(&state, &plan, &pool, &rules);
}

// --- top-k, roll, determinism --------------------------------------------------------------

fn medium(seed: u64, ft: i64, n_gws: usize) -> (SquadState, Vec<PoolPlayer>, XpFrame) {
    let pool = grid_pool(1..=6, &GRID, Some(seed));
    let xp = correlated_xp(&pool, n_gws, seed, 5);
    let state = state_with(&pool, &first_valid_squad(&pool), 5, ft);
    (state, pool, xp)
}

#[test]
fn top3_distinct_first_gw_transfers_and_roll() {
    let rules = rules();
    for seed in [0, 1] {
        let (state, pool, xp) = medium(seed, 2, 2);
        let params = OptimizerParams { horizon: 2, ..exact() };
        let problem = problem(&state, &pool, &xp, &params, &rules);
        let result = optimize(&problem, &params, 3, false, true, Search::Bound).unwrap();
        assert_eq!(result.plans.len(), 3);
        let sets: HashSet<_> = result.plans.iter().map(|p| p.first().transfer_set()).collect();
        assert_eq!(sets.len(), 3);
        let totals: Vec<f64> = result.plans.iter().map(|p| p.total_objective()).collect();
        assert!(totals.windows(2).all(|w| w[0] >= w[1] - 1e-9));
        assert!(result.plans.iter().all(|p| p.scenario == "none"));
        let roll = result.roll.as_ref().unwrap();
        assert_eq!(roll.first().n_transfers(), 0);
        assert!(result.best().gain_vs_roll.unwrap() >= -1e-9);
        for p in result.plans.iter().chain([roll]) {
            assert!(approx(
                p.gain_vs_roll.unwrap(),
                p.total_objective() - roll.total_objective(),
                1e-12
            ));
            apply_first(&state, p, &pool, &rules);
        }
        assert_eq!(result.scenario_objectives.len(), 1);
        assert_eq!(result.scenario_objectives[0].0, "none");
        assert_approx(result.scenario_objectives[0].1, totals[0]);
    }
}

#[test]
fn optimize_with_chips_records_every_scenario() {
    let rules = rules();
    let (state, pool, xp) = medium(2, 2, 2);
    let params = OptimizerParams { horizon: 2, ..Default::default() };
    let problem = problem(&state, &pool, &xp, &params, &rules);
    let result = optimize(&problem, &params, 3, true, true, Search::All).unwrap();
    let names = labels(&problem);
    assert_eq!(
        result.scenario_objectives.iter().map(|(l, _)| l.clone()).collect::<Vec<_>>(),
        names
    );
    assert_eq!(result.scenario_bounds.iter().map(|(l, _)| l.clone()).collect::<Vec<_>>(), names);
    assert_eq!(result.search.n_solves, names.len());
    let mut best = &result.scenario_objectives[0];
    for entry in &result.scenario_objectives {
        if entry.1 > best.1 {
            best = entry;
        }
    }
    assert_approx(result.best().total_objective(), best.1);
    assert!(result.plans.iter().all(|p| p.scenario == result.best().scenario));
    assert!(result.best().gain_vs_roll.unwrap() >= -1e-9);
    let sets: HashSet<_> = result.plans.iter().map(|p| p.first().transfer_set()).collect();
    assert_eq!(sets.len(), result.plans.len());
    apply_first(&state, result.best(), &pool, &rules);
}

#[test]
fn roll_is_plan_one_when_plan_one_rolls() {
    let rules = rules();
    let (state, pool, mut xp) = medium(3, 2, 2);
    xp.xp.iter_mut().for_each(|x| *x = Some(0.0)); // nothing worth a transfer
    let params = OptimizerParams { horizon: 2, ..exact() };
    let problem = problem(&state, &pool, &xp, &params, &rules);
    let result = optimize(&problem, &params, 2, false, true, Search::Bound).unwrap();
    assert_eq!(result.best().first().n_transfers(), 0);
    let roll = result.roll.as_ref().unwrap();
    assert!(roll.gws == result.best().gws && result.best().gain_vs_roll == Some(0.0));
    assert!(
        result.plans[1].first().n_transfers() > 0 && result.plans[1].gain_vs_roll.unwrap() <= 1e-9
    );
}

#[test]
fn optimize_is_deterministic() {
    let rules = rules();
    let (state, pool, xp) = medium(4, 2, 1);
    let params = OptimizerParams { horizon: 1, ..Default::default() };
    let a = problem(&state, &pool, &xp, &params, &rules);
    let mut rng = Rng::new(3);
    let shuffled: Vec<PoolPlayer> =
        rng.permutation(pool.len()).into_iter().map(|i| pool[i]).collect();
    let b =
        problem(&state, &shuffled, &take_rows(&xp, &rng.permutation(xp.len())), &params, &rules);
    let p = optimize(&a, &params, 3, true, true, Search::Bound).unwrap();
    let q = optimize(&b, &params, 3, true, true, Search::Bound).unwrap();
    let gws =
        |s: &fplopt_optimize::PlanSet| s.plans.iter().map(|x| x.gws.clone()).collect::<Vec<_>>();
    assert_eq!(gws(&p), gws(&q));
    assert_eq!(p.roll.unwrap().gws, q.roll.unwrap().gws);
    assert_eq!(p.scenario_objectives, q.scenario_objectives);
}

#[test]
fn top_k_validation_and_no_roll() {
    let rules = rules();
    let (state, pool, xp) = medium(5, 2, 1);
    let params = OptimizerParams { horizon: 1, ..exact() };
    let problem = problem(&state, &pool, &xp, &params, &rules);
    let err = optimize(&problem, &params, 0, true, true, Search::Bound).unwrap_err();
    assert!(err.to_string().contains("top_k"), "{err}");
    let result = optimize(&problem, &params, 1, false, false, Search::Bound).unwrap();
    assert!(result.roll.is_none() && result.best().gain_vs_roll.is_none());
}

#[test]
fn terminal_values_by_window_left() {
    let rules = rules();
    let params = OptimizerParams {
        chip_value: values(&[("wildcard", 6.0), ("freehit", 4.0), ("bboost", 2.0)]),
        chip_value_left: Some(BTreeMap::from([("wildcard".to_string(), vec![1.0, 2.0, 3.0])])),
        ..exact()
    };
    assert_eq!(chip_value(&params, "wildcard", 1), 1.0);
    assert_eq!(chip_value(&params, "wildcard", 2), 2.0);
    assert_eq!(chip_value(&params, "wildcard", 30), 3.0); // longer: the last entry
    assert_eq!(chip_value(&params, "freehit", 2), 4.0); // not in the table: the flat value
    assert_eq!(chip_value(&params, "3xc", 2), 0.0);
    let none = ChipScenario::none();
    // GWs 5-7: 12 GWs left in set 1, 31 in set 2 -> the Wildcard's last entry twice.
    let early = problem_at(5, 3, &[], &rules, None);
    assert_eq!(terminal_value(&early, &params, &none), 3.0 + 3.0 + 2.0 * 4.0 + 2.0 * 2.0);
    // GWs 15-17: 2 GWs left in set 1 (18, 19), 21 in set 2.
    let later = problem_at(15, 3, &[], &rules, None);
    assert_eq!(terminal_value(&later, &params, &none), 2.0 + 3.0 + 2.0 * 4.0 + 2.0 * 2.0);
}

#[test]
fn chip_value_left_is_validated() {
    for bad in [vec![], vec![1.0, -1.0]] {
        let params = OptimizerParams {
            chip_value_left: Some(BTreeMap::from([("wildcard".to_string(), bad)])),
            ..Default::default()
        };
        let err = params.validate().unwrap_err();
        assert!(err.to_string().contains("chip_value_left"), "{err}");
    }
    OptimizerParams {
        chip_value_left: Some(BTreeMap::from([("bboost".to_string(), vec![1.0, 2.0])])),
        ..Default::default()
    }
    .validate()
    .unwrap();
}
