//! The chip suggestions in the bot's alerts (spec docs/superpowers/specs/
//! 2026-11-05-chip-alerts-design.md): the chip planner P (`/chips`'s search), the rule
//! baseline H (`fplopt_optimize::chip_rules`), the open chips with expiry, and from GW20 the
//! frozen chip-policy-2 V6 (logged only).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use anyhow::Result;
use fplopt_optimize::chip_rules::{H, chip_gains, last_chance, rule_chip};
use fplopt_optimize::state::PoolPlayer;
use fplopt_optimize::{OptimizerParams, PlanInput, XpFrame};

use crate::planner::{ChipOptions, UserPlan, live_params, load_terminal_values};

/// One unused chip whose window is open at the context's GW.
#[derive(Clone, Debug, PartialEq)]
pub struct OpenChip {
    pub name: String,         // "bboost", ...
    pub last_gw: i64,         // the window's stop as a GW number
    pub best_gw: Option<i64>, // the planner's most promising GW for it alone, if solved
    pub expiring: bool,       // stop - gw_index <= 3
}

/// The chip calls for one alert (logged in `chip_calls`, shown by `messages::chip_section`).
#[derive(Clone, Debug, PartialEq)]
pub struct ChipCalls {
    pub gw: i64,
    /// The planner's chip for this GW (None: hold) and its gain over the no-chip scenario.
    pub planner: Option<String>,
    pub planner_gain: f64,
    /// H's chip for this GW and the numbers it read.
    pub rules: Option<String>,
    pub gains: BTreeMap<String, Option<f64>>,
    /// The chip H must play now or lose (overrides the planner's line).
    pub forced: Option<String>,
    pub open: Vec<OpenChip>,
    /// V6's chip for this GW: None before GW20 (not computed), Some(None) = hold.
    pub v6: Option<Option<String>>,
    /// P's solved scenarios (label, total objective), as `chip_checks` stores them.
    pub scenarios: Vec<(String, f64)>,
}

/// Chip policy 2's frozen V6, live variant (docs/research/2026-11-chip-policy-2.md §5).
pub const V6_SHRINK: f64 = 0.8;
pub const V6_FH_SCALE: f64 = 0.4;
pub const V6_TABLE: &str = "config/chips/terminal-values-window.json";
/// V6 is logged from the second chip set's first GW.
pub const V6_FROM_GW_INDEX: i64 = 20;
/// Expiry warning when a window's stop is this close (stop − gw_index ≤ 3: four GWs left).
pub const EXPIRY_GWS: i64 = 3;

pub fn is_expiring(gw_index: i64, stop: i64) -> bool {
    stop - gw_index <= EXPIRY_GWS
}

/// `bboost@gw16` -> 16 (a single-chip scenario's GW); None for `none` or a malformed label.
pub fn scenario_gw(label: &str) -> Option<i64> {
    label.split_once("@gw").and_then(|(_, g)| g.parse().ok())
}

/// V6's settings: the live settings, the per-window table × 0.8, Free Hit gains × 0.4.
pub fn v6_params() -> Result<OptimizerParams> {
    v6_params_from(Path::new(V6_TABLE))
}

/// `v6_params` with the per-window table read from `table`.
pub fn v6_params_from(table: &Path) -> Result<OptimizerParams> {
    let params = OptimizerParams {
        chip_value_left: Some(load_terminal_values(V6_SHRINK, table)?),
        fh_scale: V6_FH_SCALE,
        ..live_params()
    };
    params.validate()?;
    Ok(params)
}

/// Squad players (by club, `squad_teams`) whose club blanks the target GW: every pool player
/// of the club has xP 0 or null at horizon 0 (a v1 frame gives a blank 0; a club that plays
/// always has someone above 0).
pub fn blanks_from_frame(frame: &XpFrame, pool: &[PoolPlayer], squad_teams: &[i64]) -> i64 {
    let team_of: HashMap<i64, i64> = pool.iter().map(|p| (p.player_key, p.team_key)).collect();
    let mut playing: HashSet<i64> = HashSet::new();
    for i in 0..frame.len() {
        if frame.horizon[i] == 0
            && frame.xp[i].unwrap_or(0.0) > 0.0
            && let Some(t) = team_of.get(&frame.player_key[i])
        {
            playing.insert(*t);
        }
    }
    squad_teams.iter().filter(|t| !playing.contains(t)).count() as i64
}

/// Whether the call shown to the user changed (planner, rules or forced chip).
pub fn calls_changed(a: &ChipCalls, b: &ChipCalls) -> bool {
    (a.planner.as_deref(), a.rules.as_deref(), a.forced.as_deref())
        != (b.planner.as_deref(), b.rules.as_deref(), b.forced.as_deref())
}

/// The calls for `plan`'s user from P's search `options` and, from GW20, V6's `v6` search.
pub fn compute_chip_calls(
    plan: &UserPlan,
    options: &ChipOptions,
    v6: Option<&ChipOptions>,
) -> Result<ChipCalls> {
    let context = &plan.context;
    let gw_of: HashMap<i64, i64> = context.gameweeks.iter().map(|(g, i)| (*i, *g)).collect();
    // P
    let objectives = &options.plans.scenario_objectives;
    let base = objectives.iter().rev().find(|(l, _)| l == "none").map_or(0.0, |(_, v)| *v);
    let best = options.plans.best();
    let planner = best.first().chip.clone();
    let planner_gain = if planner.is_some() { best.total_objective() - base } else { 0.0 };
    // H on the bot's no-chip plan
    let problem = PlanInput::from_context(
        &plan.state,
        &context.pool,
        &context.xp,
        &plan.rules,
        &plan.params,
    )?;
    let squad_teams: Vec<i64> = plan.state.holdings.iter().map(|h| h.team_key).collect();
    let blanks = blanks_from_frame(&context.xp, &context.pool, &squad_teams);
    let no_chip = crate::planner::ranked(&plan.plans)[0];
    let gains = chip_gains(
        &plan.state,
        &context.pool,
        &context.xp,
        &plan.rules,
        &problem,
        &plan.params,
        no_chip,
        blanks,
    )?;
    let rules = rule_chip(&plan.state, &plan.rules, &gains, &H)?;
    let forced = last_chance(&plan.state, &plan.rules)?;
    // open chips, in window order
    let mut windows: Vec<_> = plan.rules.chips.iter().collect();
    windows.sort_by_key(|w| (w.start, w.chip_id));
    let used: HashSet<i64> = plan.state.chips_used.iter().map(|(c, _)| *c).collect();
    let open = windows
        .into_iter()
        .filter(|w| {
            !used.contains(&w.chip_id) && w.start <= context.gw_index && context.gw_index <= w.stop
        })
        .map(|w| OpenChip {
            name: w.name.clone(),
            last_gw: gw_of.get(&w.stop).copied().unwrap_or(w.stop),
            best_gw: options.single(&w.name).and_then(|(label, _)| scenario_gw(label)),
            expiring: is_expiring(context.gw_index, w.stop),
        })
        .collect();
    let v6 = v6.map(|o| o.plans.best().first().chip.clone());
    Ok(ChipCalls {
        gw: context.gw,
        planner,
        planner_gain,
        rules,
        gains,
        forced,
        open,
        v6,
        scenarios: objectives.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool_player(key: i64, team: i64) -> PoolPlayer {
        PoolPlayer { player_key: key, element_type: 3, team_key: team, price: 50 }
    }

    fn calls(planner: Option<&str>, rules: Option<&str>, forced: Option<&str>) -> ChipCalls {
        ChipCalls {
            gw: 10,
            planner: planner.map(Into::into),
            planner_gain: 2.0,
            rules: rules.map(Into::into),
            gains: BTreeMap::new(),
            forced: forced.map(Into::into),
            open: vec![],
            v6: None,
            scenarios: vec![],
        }
    }

    #[test]
    fn blanks_count_squad_players_of_clubs_with_all_zero_xp() {
        let frame = XpFrame {
            player_key: vec![10, 11, 20, 21, 30, 31, 10],
            gw: vec![9; 7],
            gw_index: vec![9, 9, 9, 9, 9, 9, 10],
            horizon: vec![0, 0, 0, 0, 0, 0, 1],
            xp: vec![Some(0.0), Some(3.2), Some(0.0), Some(0.0), None, Some(0.0), Some(5.0)],
            p_play: None,
            p_play_gw: None,
            e_minutes: None,
        };
        let pool = vec![
            pool_player(10, 1),
            pool_player(11, 1),
            pool_player(20, 2),
            pool_player(21, 2),
            pool_player(30, 3),
            pool_player(31, 3),
        ];
        let squad_teams = [1, 1, 2, 3, 3];
        assert_eq!(blanks_from_frame(&frame, &pool, &squad_teams), 3);
    }

    #[test]
    fn expiring_is_three_gws_or_fewer() {
        assert!(is_expiring(16, 19));
        assert!(!is_expiring(15, 19));
    }

    #[test]
    fn best_gw_from_scenario_label() {
        assert_eq!(scenario_gw("bboost@gw16"), Some(16));
        assert_eq!(scenario_gw("none"), None);
    }

    #[test]
    fn changed_compares_planner_rules_and_forced_only() {
        let a = calls(Some("bboost"), None, None);
        let mut b = a.clone();
        b.planner_gain += 1.0;
        b.scenarios.clear();
        assert!(!calls_changed(&a, &b));
        b.rules = Some("3xc".into());
        assert!(calls_changed(&a, &b));
    }

    // The backtest builds optimizer:v1:...,chips=1,chip_shrink=0.8,chip_split=1,fh_scale=0.4
    // as: the window table x 0.8, fh_scale 0.4, the other keys at the live settings.
    #[test]
    fn v6_is_the_window_table_times_shrink_at_the_live_settings() {
        let table = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../").join(V6_TABLE);
        let v6 = v6_params_from(&table).unwrap();
        let raw: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&table).unwrap()).unwrap();
        let values = raw["values"].as_object().unwrap();
        let left = v6.chip_value_left.clone().unwrap();
        assert_eq!(left.len(), values.len());
        for (name, list) in values {
            let want: Vec<f64> =
                list.as_array().unwrap().iter().map(|x| 0.8 * x.as_f64().unwrap()).collect();
            assert_eq!(left[name], want, "{name}");
        }
        assert_eq!(v6.fh_scale, 0.4);
        assert_eq!(OptimizerParams { chip_value_left: None, fh_scale: 1.0, ..v6 }, live_params());
    }
}
