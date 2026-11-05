//! The chip study's rule baseline H (docs/superpowers/plans/2026-10-11-chip-policy.md) and
//! the per-GW chip quantities it reads, shared by the backtester (`chiprules`, `chipgains`)
//! and the bot's chip suggestions.

use std::collections::{BTreeMap, HashMap};

use anyhow::Result;
use fplopt_build::scoring::Rules;

use crate::chips::{ChipScenario, make_scenario};
use crate::model::{SolveError, solve_plan};
use crate::params::OptimizerParams;
use crate::plans::Plan;
use crate::problem::{PlanInput, XpFrame};
use crate::state::{InvalidDecision, PoolPlayer, SquadState, chip_window};

/// Order the rule policy considers chips in (one per GW), and forced "last chance" plays.
pub const CHIP_PRIORITY: [&str; 4] = ["freehit", "wildcard", "bboost", "3xc"];
/// Per-GW chip quantities (`chip_gains`), recorded by the simulator.
pub const CHIP_DIAGNOSTICS: [&str; 7] =
    ["captain_xp", "bench_xp", "gain_tc", "gain_bb", "gain_fh", "gain_wc", "blanks"];

/// The rule thresholds: Triple Captain if the captain's xP >= `t_tc`, Bench Boost if the
/// bench's xP >= `t_bb`, Free Hit if >= `k_fh` squad players blank, Wildcard if its plan beats
/// the no-chip plan by >= `t_wc`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RuleThresholds {
    pub t_tc: f64,
    pub t_bb: f64,
    pub k_fh: i64,
    pub t_wc: f64,
}

/// H as chosen on develop: `chiprules:v1:...,t_tc=8.3,t_bb=10.9,k_fh=4,t_wc=12.87`.
pub const H: RuleThresholds = RuleThresholds { t_tc: 8.3, t_bb: 10.9, k_fh: 4, t_wc: 12.87 };

/// Whether chip `name` can be played at `state.gw_index`.
fn chip_available(state: &SquadState, name: &str, rules: &Rules) -> bool {
    chip_window(state, name, rules).is_ok()
}

/// The chip the rules must play now or lose (more unused chips in a window than GWs left in
/// it, first in `CHIP_PRIORITY`), or None.
pub fn last_chance(state: &SquadState, rules: &Rules) -> Result<Option<String>> {
    let playable: Vec<&str> =
        CHIP_PRIORITY.iter().copied().filter(|c| chip_available(state, c, rules)).collect();
    let mut windows = HashMap::new();
    for c in &playable {
        windows.insert(*c, chip_window(state, c, rules)?);
    }
    let stops: HashMap<i64, i64> = rules.chips.iter().map(|w| (w.chip_id, w.stop)).collect();
    for chip in &playable {
        let stop = stops[&windows[chip]];
        let same: Vec<&str> =
            playable.iter().copied().filter(|c| stops[&windows[c]] == stop).collect();
        if same.len() as i64 > stop - state.gw_index {
            return Ok(Some(same[0].to_owned()));
        }
    }
    Ok(None)
}

/// The chip H plays this GW: the last chance, else the first chip in `CHIP_PRIORITY` whose
/// threshold fires, else None.
pub fn rule_chip(
    state: &SquadState,
    rules: &Rules,
    gains: &BTreeMap<String, Option<f64>>,
    t: &RuleThresholds,
) -> Result<Option<String>> {
    if let Some(chip) = last_chance(state, rules)? {
        return Ok(Some(chip));
    }
    let g = |k: &str| gains.get(k).copied().flatten();
    let fires = |c: &str| match c {
        "freehit" => g("blanks").unwrap_or(0.0) >= t.k_fh as f64,
        "wildcard" => g("gain_wc").is_some_and(|x| x >= t.t_wc),
        "bboost" => g("bench_xp").unwrap_or(0.0) >= t.t_bb,
        "3xc" => g("captain_xp").unwrap_or(0.0) >= t.t_tc,
        _ => false,
    };
    Ok(CHIP_PRIORITY
        .iter()
        .copied()
        .filter(|c| chip_available(state, c, rules))
        .find(|c| fires(c))
        .map(str::to_owned))
}

/// `player_key`'s target-GW xP in `frame` (horizon 0, summed over a double; null -> 0).
pub fn target_xp_of(frame: &XpFrame, key: i64) -> f64 {
    let mut total = 0.0;
    for i in 0..frame.len() {
        if frame.player_key[i] == key && frame.horizon[i] == 0 {
            total += frame.xp[i].unwrap_or(0.0);
        }
    }
    total
}

/// `params` for a 1-GW plan (the Free Hit gain): the expected-minutes floor scaled to one GW.
pub fn one_gw_params(params: &OptimizerParams) -> OptimizerParams {
    OptimizerParams {
        horizon: 1,
        min_minutes: params.min_minutes / params.horizon as f64,
        ..params.clone()
    }
}

/// The objective (no terminal value) of the plan playing `chip` first, or None if the
/// rules don't allow it there or the solve found no plan.
fn chip_objective(
    problem: &PlanInput,
    params: &OptimizerParams,
    chip: &str,
) -> Result<Option<f64>> {
    match make_scenario(problem, &BTreeMap::from([(0, chip.to_owned())])) {
        Err(InvalidDecision(_)) => Ok(None),
        Ok(scenario) => match solve_plan(problem, params, None, &[], &scenario) {
            Ok(plan) => Ok(Some(plan.objective)),
            Err(SolveError::Infeasible(_) | SolveError::LimitReached { .. }) => Ok(None),
            Err(SolveError::Other(e)) => Err(e),
        },
    }
}

fn no_chip_objective(problem: &PlanInput, params: &OptimizerParams) -> Result<Option<f64>> {
    let none = ChipScenario::none();
    match solve_plan(problem, params, None, &[], &none) {
        Ok(plan) => Ok(Some(plan.objective)),
        Err(SolveError::Infeasible(_) | SolveError::LimitReached { .. }) => Ok(None),
        Err(SolveError::Other(e)) => Err(e),
    }
}

/// The `CHIP_DIAGNOSTICS` for the no-chip `plan` of `problem` (built from `state`, `pool`,
/// `frame`, `rules` with `params`); `blanks` is the caller's count of squad players whose club
/// has no target-GW fixture.
#[allow(clippy::too_many_arguments)]
pub fn chip_gains(
    state: &SquadState,
    pool: &[PoolPlayer],
    frame: &XpFrame,
    rules: &Rules,
    problem: &PlanInput,
    params: &OptimizerParams,
    plan: &Plan,
    blanks: i64,
) -> Result<BTreeMap<String, Option<f64>>> {
    let first = plan.first();
    let weights = problem.gw_bench_weights(0, params).to_vec();
    let bench: Vec<f64> = first.bench.iter().map(|k| target_xp_of(frame, *k)).collect();
    let captain = target_xp_of(frame, first.captain);
    let one = one_gw_params(params);
    let short = PlanInput::from_context(state, pool, frame, rules, &one)?;
    let mut fh = None;
    if chip_available(&problem.state, "freehit", rules) {
        let base = no_chip_objective(&short, &one)?;
        let with_fh = chip_objective(&short, &one, "freehit")?;
        if let (Some(b), Some(f)) = (base, with_fh) {
            fh = Some(f - b);
        }
    }
    let mut wc = None;
    if chip_available(&problem.state, "wildcard", rules) {
        wc = chip_objective(problem, params, "wildcard")?.map(|w| w - plan.objective);
    }
    let tc_ok = chip_available(&problem.state, "3xc", rules);
    let bb_ok = chip_available(&problem.state, "bboost", rules);
    let bench_sum = bench.iter().fold(0.0, |acc, x| acc + x);
    let bb_gain = weights.iter().zip(&bench).fold(0.0, |acc, (w, x)| acc + (1.0 - w) * x);
    Ok(BTreeMap::from([
        ("captain_xp".to_owned(), Some(captain)),
        ("bench_xp".to_owned(), Some(bench_sum)),
        ("gain_tc".to_owned(), tc_ok.then_some(captain)),
        ("gain_bb".to_owned(), bb_ok.then_some(bb_gain)),
        ("gain_fh".to_owned(), fh),
        ("gain_wc".to_owned(), wc),
        ("blanks".to_owned(), Some(blanks as f64)),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Holding;
    use fplopt_build::scoring::{ChipWindow, backtest_rules};
    use std::path::Path;

    fn gains(pairs: &[(&str, f64)]) -> BTreeMap<String, Option<f64>> {
        pairs.iter().map(|(k, v)| (k.to_string(), Some(*v))).collect()
    }

    /// A squad at `gw_index` with `chips_used` (chip_id, gw_index), and 2026/27's chip
    /// windows: two halves, GW2/GW1-19 and GW20-38.
    fn state_at(gw_index: i64, chips_used: &[(i64, i64)]) -> (SquadState, Rules) {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config/scoring");
        let mut rules = backtest_rules(2026, &dir).unwrap();
        let w = |chip_id, name: &str, start, stop| ChipWindow {
            chip_id,
            name: name.to_owned(),
            start,
            stop,
        };
        rules.chips = vec![
            w(1, "wildcard", 2, 19),
            w(2, "wildcard", 20, 38),
            w(3, "freehit", 2, 19),
            w(6, "freehit", 20, 38),
            w(4, "bboost", 1, 19),
            w(7, "bboost", 20, 38),
            w(5, "3xc", 1, 19),
            w(8, "3xc", 20, 38),
        ];
        let holdings = (1..=15)
            .map(|k| Holding {
                player_key: k,
                element_type: 1,
                team_key: k,
                purchase_price: 50,
                price: 50,
            })
            .collect();
        let mut state = SquadState::new(2026, gw_index, holdings, 0, 1).unwrap();
        state.chips_used = chips_used.to_vec();
        (state, rules)
    }

    #[test]
    fn thresholds_fire_in_priority_order() {
        let (state, rules) = state_at(10, &[]);
        let g = gains(&[("captain_xp", 9.0), ("bench_xp", 12.0), ("blanks", 0.0)]);
        // bboost is before 3xc in CHIP_PRIORITY
        assert_eq!(rule_chip(&state, &rules, &g, &H).unwrap().as_deref(), Some("bboost"));
        let g = gains(&[("captain_xp", 9.0), ("bench_xp", 5.0), ("blanks", 0.0)]);
        assert_eq!(rule_chip(&state, &rules, &g, &H).unwrap().as_deref(), Some("3xc"));
        let g = gains(&[("captain_xp", 5.0), ("bench_xp", 5.0), ("blanks", 3.0)]);
        assert_eq!(rule_chip(&state, &rules, &g, &H).unwrap(), None);
        let g = gains(&[("captain_xp", 5.0), ("bench_xp", 5.0), ("blanks", 4.0)]);
        assert_eq!(rule_chip(&state, &rules, &g, &H).unwrap().as_deref(), Some("freehit"));
    }

    #[test]
    fn last_chance_when_chips_outnumber_gws() {
        // gw_index 19: four first-half chips unused, none after -> freehit first (priority)
        let (state, rules) = state_at(19, &[]);
        assert_eq!(last_chance(&state, &rules).unwrap().as_deref(), Some("freehit"));
        // wildcard and 3xc used: freehit and bboost left. The check is stop - gw_index
        // (not +1), so two chips fire from gw_index 18, not 17.
        let (state, rules) = state_at(17, &[(1, 4), (5, 7)]);
        assert_eq!(last_chance(&state, &rules).unwrap(), None);
        let (state, rules) = state_at(18, &[(1, 4), (5, 7)]);
        assert_eq!(last_chance(&state, &rules).unwrap().as_deref(), Some("freehit"));
        // and rule_chip returns the forced chip whatever the gains
        let g = gains(&[("captain_xp", 0.0), ("bench_xp", 0.0), ("blanks", 0.0)]);
        assert_eq!(rule_chip(&state, &rules, &g, &H).unwrap().as_deref(), Some("freehit"));
    }

    #[test]
    fn h_is_the_chosen_rule_baseline() {
        // docs/superpowers/plans/2026-10-11-chip-policy.md: t_tc=8.3,t_bb=10.9,k_fh=4,t_wc=12.87
        assert_eq!(H, RuleThresholds { t_tc: 8.3, t_bb: 10.9, k_fh: 4, t_wc: 12.87 });
    }
}
