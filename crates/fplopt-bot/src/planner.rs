//! A user's plan on a plan context (Python `fplopt.bot.planner`; Phase 7 plan §1, §4):
//! import their squad from FPL's public API (`fplopt_optimize::entry`), apply the transfers
//! they already made and their `/setft` override, and run the optimizer with the settings of
//! the system that passed Phase 6 (`live_params`, `holdout.SYSTEM`), no chips (chips stay the
//! manager's call until a chip policy passes its own test, PLAN §12).

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::path::Path;

use anyhow::Result;
use fplopt_build::scoring::{Rules, backtest_rules, default_config_dir};
use fplopt_optimize::entry::{
    EntryClient, EntryError, SquadPlayer, SquadSpec, import_squad, squad_state,
};
use fplopt_optimize::model::SolveError;
use fplopt_optimize::state::{Holding, SquadState, chip_window, selling_price};
use fplopt_optimize::{
    OptimizerParams, Plan, PlanInput, PlanSet, Search, optimize, scenarios, solve_plan,
};
use log::warn;
use serde::Serialize;
use serde_json::Value;

use crate::context::PlanContext;
use crate::util::py_isoformat;

pub const TOP_K: usize = 3;
/// The chip study's chosen P (chip pre-registration §6c).
pub const CHIP_SHRINK: f64 = 1.0;
pub const CHIP_LABELS: [(&str, &str); 4] = [
    ("wildcard", "Wildcard"),
    ("freehit", "Free Hit"),
    ("bboost", "Bench Boost"),
    ("3xc", "Triple Captain"),
];
/// `config/chips/terminal-values.json` (relative to the repo root, like the Python CLI).
pub const CHIP_VALUES_FILE: &str = "config/chips/terminal-values.json";

pub fn chip_label(name: &str) -> &str {
    CHIP_LABELS.iter().find(|(n, _)| *n == name).map_or(name, |(_, label)| label)
}

/// `holdout.SYSTEM` (optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2; the other
/// settings at their defaults) with the interactive wall-clock cap per solve
/// (`INTERACTIVE_TIME_LIMIT`).
pub fn live_params() -> OptimizerParams {
    OptimizerParams {
        horizon: 6,
        decay: 0.75,
        max_hits: Some(1),
        hit_margin: 2.0,
        time_limit: Some(60.0),
        ..OptimizerParams::default()
    }
}

/// FPL has no team with that ID (Python `UnknownTeam`, an `EntryError`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnknownTeam(pub String);

impl fmt::Display for UnknownTeam {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for UnknownTeam {}

/// The marker an FPL client returns for an HTTP 404 (no such team).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NotFound;

impl fmt::Display for NotFound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("HTTP 404")
    }
}

impl std::error::Error for NotFound {}

pub fn is_not_found(err: &anyhow::Error) -> bool {
    err.downcast_ref::<NotFound>().is_some()
}

/// The message of an `EntryError` (an `UnknownTeam` too), if `err` is one: what Python's
/// `except EntryError` catches.
pub fn entry_message(err: &anyhow::Error) -> Option<String> {
    if let Some(e) = err.downcast_ref::<UnknownTeam>() {
        return Some(e.0.clone());
    }
    err.downcast_ref::<EntryError>().map(|e| e.0.clone())
}

/// FPL's public API as the bot uses it: the entry endpoints and the bootstrap (live prices).
pub trait BotFpl: EntryClient {
    fn bootstrap_static(&self) -> Result<Vec<u8>>;
}

/// A user's squad (`spec`, `state`) and plans on `context`.
#[derive(Clone, Debug)]
pub struct UserPlan {
    pub context: PlanContext,
    pub entry_id: i64,
    pub spec: SquadSpec,
    pub state: SquadState,
    pub plans: PlanSet,
    pub rules: Rules,
    pub ft_override: Option<i64>,
    pub params: OptimizerParams,
}

fn name_of(element: i64, keys: &HashMap<i64, i64>, names: &HashMap<i64, String>) -> String {
    keys.get(&element)
        .and_then(|k| names.get(k))
        .cloned()
        .unwrap_or_else(|| format!("element {element}"))
}

/// The squad after the transfers the manager already made for the planned GW (`made`:
/// (element out, element in) in order; FPL doesn't publish them before the deadline): each
/// sale at its FPL selling price, each buy at the current price (its new purchase price), the
/// bank and free transfers updated (a move beyond the free transfers was a hit, already
/// paid). EntryError when a move isn't possible.
pub fn apply_made_transfers(
    spec: &SquadSpec,
    made: &[(i64, i64)],
    context: &PlanContext,
) -> Result<SquadSpec> {
    if made.is_empty() {
        return Ok(spec.clone());
    }
    let keys = context.element_keys();
    let price: HashMap<i64, i64> = context.pool.iter().map(|p| (p.player_key, p.price)).collect();
    let names = context.names();
    let mut players: Vec<SquadPlayer> = spec.players.clone();
    let mut bank = spec.bank;
    let err = |m: String| -> Result<SquadSpec> { Err(EntryError(m).into()) };
    for &(out, into) in made {
        let Some(at) = players.iter().position(|p| p.element_id == out) else {
            return err(format!("{} isn't in the squad", name_of(out, &keys, &names)));
        };
        if players.iter().any(|p| p.element_id == into) {
            return err(format!("{} is already in the squad", name_of(into, &keys, &names)));
        }
        let priced = |e: i64| keys.get(&e).and_then(|k| price.get(k)).copied();
        let (Some(now_out), Some(cost)) = (priced(out), priced(into)) else {
            return err(format!("{} isn't in this GW's player pool", name_of(into, &keys, &names)));
        };
        let held = Holding {
            player_key: keys[&out],
            element_type: 0,
            team_key: 0,
            purchase_price: players[at].purchase_price,
            price: now_out,
        };
        // Python's `selling_price(held)`: FPL's fee, 0.5 (the default, not the rules').
        bank += selling_price(&held, 0.5) - cost;
        players.remove(at);
        let name = names.get(&keys[&into]).cloned().unwrap_or_default();
        players.push(SquadPlayer { element_id: into, purchase_price: cost, name });
    }
    if bank < 0 {
        return err(format!(
            "those transfers need £{:.1}m more than you have",
            -bank as f64 / 10.0
        ));
    }
    let moves: Vec<String> = made
        .iter()
        .map(|&(o, i)| format!("{} → {}", name_of(o, &keys, &names), name_of(i, &keys, &names)))
        .collect();
    let n = made.len() as i64;
    let hits = (n - spec.free_transfers).max(0);
    let left = (spec.free_transfers - n).max(0);
    let mut note = format!(
        "transfers you told the bot about (/transfer): {}; bank now £{:.1}m, {left} free \
         transfer(s) left",
        moves.join(", "),
        bank as f64 / 10.0
    );
    if hits > 0 {
        note += &format!("; {hits} of them beyond your free transfers (hits already taken)");
    }
    let mut out = spec.clone();
    out.players = players;
    out.bank = bank;
    out.free_transfers = left;
    out.notes.push(note);
    Ok(out)
}

/// The season's rules (`config/scoring`).
pub fn rules_for(season: i64) -> Result<Rules> {
    backtest_rules(season, &default_config_dir())
}

/// The FPL team's squad before the context's GW, after the transfers already `made` for it
/// (`apply_made_transfers`), as a spec and a state (EntryError / UnknownTeam when it can't be
/// imported or isn't usable).
pub fn import_user_squad(
    fpl: &dyn EntryClient,
    entry_id: i64,
    context: &PlanContext,
    ft_override: Option<i64>,
    made: &[(i64, i64)],
) -> Result<(SquadSpec, SquadState, Rules)> {
    let rules = rules_for(context.season)?;
    let gw_index = context.gw_index_map();
    let mut initial_prices = |gw: i64| Ok(context.initial_prices(gw));
    let spec = match import_squad(
        fpl,
        entry_id,
        context.season,
        context.gw,
        &rules,
        &gw_index,
        &mut initial_prices,
    ) {
        Ok(spec) => spec,
        Err(e) if is_not_found(&e) => {
            return Err(UnknownTeam(format!("FPL has no team with ID {entry_id}")).into());
        }
        Err(e) => return Err(e),
    };
    let mut spec = apply_made_transfers(&spec, made, context)?;
    if let Some(ft) = ft_override {
        spec.free_transfers = ft;
        spec.notes.push(format!("free transfers {ft}: set by you (/setft)"));
    }
    let state = squad_state(&spec, &context.pool, &context.element_keys(), &rules, &gw_index)?;
    Ok((spec, state, rules))
}

/// Import the squad (after the transfers already `made`) and plan it: the top `TOP_K` plans
/// and the roll plan, no chips.
pub fn plan_user(
    fpl: &dyn EntryClient,
    entry_id: i64,
    context: &PlanContext,
    ft_override: Option<i64>,
    params: &OptimizerParams,
    made: &[(i64, i64)],
) -> Result<UserPlan> {
    let (spec, state, rules) = import_user_squad(fpl, entry_id, context, ft_override, made)?;
    let problem = PlanInput::from_context(&state, &context.pool, &context.xp, &rules, params)?;
    let plans = optimize(&problem, params, TOP_K, false, true, Search::Bound)?;
    Ok(UserPlan {
        context: context.clone(),
        entry_id,
        spec,
        state,
        plans,
        rules,
        ft_override,
        params: params.clone(),
    })
}

/// `/chips`: the chip scenario search on a user's squad (information only: the chip planner
/// isn't validated, PLAN §11). `plans.scenario_objectives` holds every solved scenario's best
/// total objective (`none` = no chip); `plans.best()` is the winner. `singles`: chip name ->
/// (scenario label, total objective), each available chip on its own in its most promising
/// horizon GW, in Python's dict order.
#[derive(Clone, Debug)]
pub struct ChipOptions {
    pub context: PlanContext,
    pub entry_id: i64,
    pub spec: SquadSpec,
    pub state: SquadState,
    pub plans: PlanSet,
    pub rules: Rules,
    pub params: OptimizerParams,
    pub singles: Vec<(String, (String, f64))>,
}

impl ChipOptions {
    pub fn single(&self, name: &str) -> Option<&(String, f64)> {
        self.singles.iter().find(|(n, _)| n == name).map(|(_, v)| v)
    }
}

/// The estimated terminal value table times `shrink` (`OptimizerParams.chip_value_left`);
/// an error if it hasn't been estimated (no file) or `shrink` < 0.
pub fn load_terminal_values(shrink: f64, path: &Path) -> Result<BTreeMap<String, Vec<f64>>> {
    if !shrink.is_finite() || shrink < 0.0 {
        anyhow::bail!("chip_shrink must be a finite number >= 0, got {shrink}");
    }
    if !path.is_file() {
        anyhow::bail!(
            "chip_shrink: no estimated chip values at {} (fplopt chips values)",
            path.display()
        );
    }
    let table: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    let values = table
        .get("values")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow::anyhow!("no 'values'"))?;
    values
        .iter()
        .map(|(name, v)| {
            let list = v.as_array().ok_or_else(|| anyhow::anyhow!("values.{name}: not a list"))?;
            let floats = list
                .iter()
                .map(|x| {
                    x.as_f64()
                        .map(|f| shrink * f)
                        .ok_or_else(|| anyhow::anyhow!("values.{name}: not a number"))
                })
                .collect::<Result<Vec<f64>>>()?;
            Ok((name.clone(), floats))
        })
        .collect()
}

/// `/chips`'s planner settings: `live_params` with the chip study's estimated terminal values
/// (chip_shrink 1.0: the study's P, PLAN §12), or the flat placeholders if the table is
/// missing or unusable. Information only: P didn't beat the rule baseline at the
/// pre-registered threshold.
pub fn chip_params() -> OptimizerParams {
    let base = live_params();
    match load_terminal_values(CHIP_SHRINK, Path::new(CHIP_VALUES_FILE)) {
        Ok(values) => {
            let params = OptimizerParams { chip_value_left: Some(values), ..base.clone() };
            match params.validate() {
                Ok(()) => params,
                Err(e) => {
                    warn!("chip values unusable ({e}): flat placeholders");
                    base
                }
            }
        }
        Err(e) => {
            warn!("{e}: flat chip placeholders");
            base
        }
    }
}

/// Import the squad and search the chip scenarios over the horizon (the bound search: only
/// scenarios that could win are solved; same best plan as solving them all), with `params`.
pub fn chip_options(
    fpl: &dyn EntryClient,
    entry_id: i64,
    context: &PlanContext,
    ft_override: Option<i64>,
    params: &OptimizerParams,
    made: &[(i64, i64)],
) -> Result<ChipOptions> {
    let (spec, state, rules) = import_user_squad(fpl, entry_id, context, ft_override, made)?;
    let problem = PlanInput::from_context(&state, &context.pool, &context.xp, &rules, params)?;
    let plans = optimize(&problem, params, 1, true, false, Search::Bound)?;
    let singles = singles(&problem, params, &plans)?;
    Ok(ChipOptions {
        context: context.clone(),
        entry_id,
        spec,
        state,
        plans,
        rules,
        params: params.clone(),
        singles,
    })
}

/// Per chip name, its single-chip scenario with the highest bound, solved unless the search
/// already did (one extra MILP per chip at most).
fn singles(
    problem: &PlanInput,
    params: &OptimizerParams,
    plans: &PlanSet,
) -> Result<Vec<(String, (String, f64))>> {
    let bound =
        |label: &str| plans.scenario_bounds.iter().find(|(l, _)| l == label).map(|(_, b)| *b);
    let mut promising: Vec<(String, fplopt_optimize::ChipScenario)> = Vec::new();
    for scenario in scenarios(problem) {
        let Some(b) = bound(&scenario.label) else { continue };
        if scenario.chips.len() != 1 {
            continue;
        }
        let name = scenario.chips[0].1.clone();
        match promising.iter_mut().find(|(n, _)| *n == name) {
            None => promising.push((name, scenario)),
            Some((_, best)) => {
                if b > bound(&best.label).expect("a promising scenario has a bound") {
                    *best = scenario;
                }
            }
        }
    }
    let mut out = Vec::new();
    for (name, scenario) in promising {
        let solved =
            plans.scenario_objectives.iter().find(|(l, _)| *l == scenario.label).map(|(_, v)| *v);
        let value = match solved {
            Some(v) => v,
            None => match solve_plan(problem, params, None, &[], &scenario) {
                Ok(plan) => plan.total_objective(),
                Err(SolveError::Infeasible(_)) => continue,
                Err(SolveError::LimitReached { status, .. }) => {
                    anyhow::bail!(
                        "chip scenario {}: no plan before the solver stopped ({status})",
                        scenario.label
                    )
                }
                Err(SolveError::Other(e)) => return Err(e),
            },
        };
        out.push((name, (scenario.label.clone(), value)));
    }
    Ok(out)
}

/// The chips the manager can still play at the planned GW (labels, in a fixed order).
pub fn chips_left(state: &SquadState, rules: &Rules) -> Vec<&'static str> {
    CHIP_LABELS
        .iter()
        .filter(|(name, _)| chip_window(state, name, rules).is_ok())
        .map(|(_, label)| *label)
        .collect()
}

/// The top-k plans by total objective, best first. `optimize` returns them in solve order,
/// which can be inverted within the MIP gap (PLAN §8 live finding: plan #3 above plan #1), so
/// the bot shows the best plan found first.
pub fn ranked(plans: &PlanSet) -> Vec<&Plan> {
    let mut order: Vec<&Plan> = plans.plans.iter().collect();
    order.sort_by(|a, b| {
        (-a.total_objective())
            .partial_cmp(&-b.total_objective())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order
}

/// What the alerts compare: the best plan's moves for this GW (player keys, sorted) and its
/// captain and vice.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Decision {
    pub transfers: Vec<[i64; 2]>,
    pub captain: i64,
    pub vice: i64,
}

pub fn decision_summary(plan: &UserPlan) -> Decision {
    let first = ranked(&plan.plans)[0].first();
    let mut transfers: Vec<[i64; 2]> =
        first.transfers.iter().map(|t| [t.out_key, t.in_key]).collect();
    transfers.sort();
    Decision { transfers, captain: first.captain, vice: first.vice }
}

#[derive(Serialize)]
struct GwJson<'a> {
    gw: i64,
    transfers: Vec<[i64; 2]>,
    starters: &'a [i64],
    bench: &'a [i64],
    captain: i64,
    vice: i64,
    chip: &'a Option<String>,
    xp: f64,
    hits: i64,
    bank: i64,
    ft: i64,
}

#[derive(Serialize)]
struct PlanJson<'a> {
    total_objective: f64,
    gain_vs_roll: Option<f64>,
    status: &'a str,
    mip_gap: f64,
    gws: Vec<GwJson<'a>>,
}

#[derive(Serialize)]
struct ParamsJson {
    horizon: i64,
    decay: f64,
    max_hits: Option<i64>,
    hit_margin: f64,
}

#[derive(Serialize)]
struct PlansJson<'a> {
    entry_id: i64,
    prices_at: Option<String>,
    bank: i64,
    free_transfers: i64,
    squad: Vec<[i64; 2]>,
    chips_used: Vec<(&'a str, i64)>,
    params: ParamsJson,
    plans: Vec<PlanJson<'a>>,
    roll: Option<PlanJson<'a>>,
}

fn plan_json(p: &Plan) -> PlanJson<'_> {
    PlanJson {
        total_objective: p.total_objective(),
        gain_vs_roll: p.gain_vs_roll,
        status: &p.status,
        mip_gap: p.mip_gap,
        gws: p
            .gws
            .iter()
            .map(|g| GwJson {
                gw: g.gw,
                transfers: g.transfers.iter().map(|t| [t.out_key, t.in_key]).collect(),
                starters: &g.starters,
                bench: &g.bench,
                captain: g.captain,
                vice: g.vice,
                chip: &g.chip,
                xp: g.xp,
                hits: g.hits,
                bank: g.bank,
                ft: g.ft,
            })
            .collect(),
    }
}

/// The plans as plain data for the live log, as Python's `json.dumps(plans_json(plan))`.
pub fn plans_json(plan: &UserPlan) -> Result<String> {
    let p = &plan.params;
    fplopt_core::pyjson::dumps(&PlansJson {
        entry_id: plan.entry_id,
        prices_at: plan.context.prices_at.map(py_isoformat),
        bank: plan.spec.bank,
        free_transfers: plan.spec.free_transfers,
        squad: plan.spec.players.iter().map(|p| [p.element_id, p.purchase_price]).collect(),
        chips_used: plan.spec.chips_used.iter().map(|(n, g)| (n.as_str(), *g)).collect(),
        params: ParamsJson {
            horizon: p.horizon,
            decay: p.decay,
            max_hits: p.max_hits,
            hit_margin: p.hit_margin,
        },
        plans: ranked(&plan.plans).into_iter().map(plan_json).collect(),
        roll: plan.plans.roll.as_ref().map(plan_json),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Python test_live_params_are_the_phase6_system: holdout.SYSTEM is
    // optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2 with the 60 s interactive cap.
    #[test]
    fn live_params_are_the_phase6_system() {
        let system = OptimizerParams {
            horizon: 6,
            decay: 0.75,
            max_hits: Some(1),
            hit_margin: 2.0,
            ..OptimizerParams::default()
        };
        let live = live_params();
        assert_eq!(live.time_limit, Some(60.0));
        assert_eq!(OptimizerParams { time_limit: None, ..live }, system);
    }

    // Python test_chip_params_use_the_estimated_table (the loader; chip_params reads the
    // repo's table).
    #[test]
    fn terminal_values() {
        let dir = std::env::temp_dir().join(format!("fplopt-chip-values-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let table = dir.join("terminal-values.json");
        assert!(
            load_terminal_values(1.0, &table)
                .unwrap_err()
                .to_string()
                .contains("no estimated chip values")
        );
        std::fs::write(&table, r#"{"values": {"bboost": [2.0, 4.0]}}"#).unwrap();
        let values = load_terminal_values(1.0, &table).unwrap();
        assert_eq!(values, BTreeMap::from([("bboost".to_string(), vec![2.0, 4.0])]));
        assert_eq!(load_terminal_values(0.5, &table).unwrap()["bboost"], [1.0, 2.0]);
        assert!(load_terminal_values(-1.0, &table).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
