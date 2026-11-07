//! Decision policies (Python `fplopt.backtest.policies`; PLAN §5 *Decision policy is
//! pluggable*). A policy holds parameters, names the xP model it reads and decides one GW
//! from a `DecisionContext` (the deadline view, the refreshed state, the rules, the pool and
//! the xP frame); it keeps no state between calls.
//!
//! - `best_lineup`: the best XI on target-GW xP under the formation limits; bench GK first,
//!   then outfield by xP; captain and vice the two best starters.
//! - `Policy::Roll`: no transfers. `Policy::Greedy`: the best same-position transfer by
//!   discounted horizon xP while its gain beats the threshold and a free transfer is left.
//! - `Policy::Optimizer`: the MILP planner's plan #1 first-GW decision (chip policy 2's
//!   leave-one-season-out tables, known doubles and rule Triple Captain included).
//! - `Policy::ChipGains` (the chip study's S0, recording chip gains) and `Policy::ChipRules`
//!   (the heuristic chip baseline H).
//!
//! Ties are broken by `player_key` everywhere.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use anyhow::{Result, bail};
use fplopt_build::scoring::Rules;
use fplopt_core::pyjson::py_float_repr;
use fplopt_features::baseline::pool_coverage;
use fplopt_models::MAX_HORIZON;
use fplopt_optimize::chips::make_scenario;
use fplopt_optimize::model::{SolveError, solve_plan};
use fplopt_optimize::plans::{GwPlan, Plan};
use fplopt_optimize::state::best_xi;
use fplopt_optimize::{OptimizerParams, PlanInput, PlanSet, Search, XpFrame, optimize, py_sum};
use fplopt_store::{AsOfView, Table};
use sha1::{Digest, Sha1};

use crate::forecast_prices::ForecastPrices;
use crate::gw_score::Lineup;
use crate::oracle::FuturePrices;
use crate::state::{
    Decision, GOALKEEPER, PoolPlayer, SquadState, Transfer, chip_available, chip_window, refresh,
    selling_price,
};

pub use fplopt_optimize::chip_rules::{CHIP_DIAGNOSTICS, CHIP_PRIORITY, one_gw_params};
use fplopt_optimize::chip_rules::{RuleThresholds, rule_chip, target_xp_of};
/// Chip policy 2's known-double lookahead (GWs after the decision GW).
pub const DOUBLE_LEAD: i64 = 10;

/// Everything a policy may use for one GW's decision: `xp` is the policy's xP frame for the
/// view (`player_key, gw, gw_index, horizon, xp`, ...) and `frame` the planner's copy of it.
pub struct DecisionContext<'a> {
    pub view: &'a AsOfView<'a>,
    pub state: &'a SquadState,
    pub rules: &'a Rules,
    pub pool: &'a [PoolPlayer],
    pub xp: &'a Table,
    pub frame: &'a XpFrame,
    /// The price paths' source for a policy that plans on moving prices
    /// (`Policy::price_source`), set only by the backtester's decision paths.
    pub prices: Option<PriceInput<'a>>,
}

/// What a non-fixed `PriceSource` plans on: the season's future deadline prices
/// (`prices=oracle`) or its fitted price forecast (`prices=forecast`).
#[derive(Clone, Copy)]
pub enum PriceInput<'a> {
    Oracle(&'a FuturePrices),
    Forecast(&'a ForecastPrices),
}

// --- xP helpers ---------------------------------------------------------------------------

/// player_key -> xP of the target GW (horizon 0; NaN for a null xP).
pub fn target_xp(frame: &XpFrame) -> HashMap<i64, f64> {
    (0..frame.len())
        .filter(|&i| frame.horizon[i] == 0)
        .map(|i| (frame.player_key[i], frame.xp[i].unwrap_or(f64::NAN)))
        .collect()
}

/// Per player_key Σ decay^h · xp_h over the horizons h < `horizon` (pandas' grouped sum:
/// Kahan in row order, nulls skipped).
pub fn horizon_xp(frame: &XpFrame, horizon: i64, decay: f64) -> HashMap<i64, f64> {
    let mut sums: HashMap<i64, (f64, f64)> = HashMap::new();
    for i in 0..frame.len() {
        if frame.horizon[i] >= horizon {
            continue;
        }
        let entry = sums.entry(frame.player_key[i]).or_insert((0.0, 0.0));
        let Some(x) = frame.xp[i] else { continue };
        let value = x * decay.powf(frame.horizon[i] as f64);
        if value.is_nan() {
            continue;
        }
        let (sum, comp) = entry;
        let y = value - *comp;
        let t = *sum + y;
        *comp = t - *sum - y;
        *sum = t;
    }
    sums.into_iter().map(|(k, (s, _))| (k, s)).collect()
}

fn value(xp: &HashMap<i64, f64>, key: i64) -> f64 {
    let v = xp.get(&key).copied().unwrap_or(0.0);
    if v.is_nan() { 0.0 } else { v }
}

// --- lineup -------------------------------------------------------------------------------

/// The XI with the most target-GW xP under the formation limits, for a squad of (player_key,
/// element_type): each position's `play_min` best start, then the best of the rest while
/// their position is under `play_max`. Bench: the other GK first, then the outfield players
/// by xP. Captain and vice: the two best starters. Starters listed by position, then xP.
pub fn best_lineup(
    squad: &[(i64, i64)],
    xp_target: &HashMap<i64, f64>,
    rules: &Rules,
) -> Result<Lineup> {
    let rank = |p: &(i64, i64)| (-value(xp_target, p.0), p.0);
    let cmp = |a: &(i64, i64), b: &(i64, i64)| {
        let (ra, rb) = (rank(a), rank(b));
        ra.0.total_cmp(&rb.0).then(ra.1.cmp(&rb.1))
    };
    let mut order = squad.to_vec();
    order.sort_by(cmp);
    let starters = best_xi(&order, rules);
    if starters.len() as i64 != rules.squad_play {
        let mut counts: Vec<(i64, i64)> = Vec::new();
        for (_, et) in &starters {
            match counts.iter_mut().find(|(e, _)| e == et) {
                Some(c) => c.1 += 1,
                None => counts.push((*et, 1)),
            }
        }
        bail!(
            "squad cannot field {} starters: {}",
            rules.squad_play,
            crate::pyrepr::counter_dict(&counts)
        );
    }
    let chosen: BTreeSet<(i64, i64)> = starters.iter().copied().collect();
    let mut bench: Vec<(i64, i64)> =
        order.iter().filter(|p| !chosen.contains(p) && p.1 == GOALKEEPER).copied().collect();
    bench.extend(order.iter().filter(|p| !chosen.contains(p) && p.1 != GOALKEEPER));
    let mut by_rank = starters.clone();
    by_rank.sort_by(cmp);
    let mut listed = starters;
    listed.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| cmp(a, b)));
    Ok(Lineup {
        starters: listed.iter().map(|p| p.0).collect(),
        bench: bench.iter().map(|p| p.0).collect(),
        captain: by_rank[0].0,
        vice: by_rank[1].0,
    })
}

fn squad_of(state: &SquadState) -> Vec<(i64, i64)> {
    state.holdings.iter().map(|h| (h.player_key, h.element_type)).collect()
}

// --- transfers ----------------------------------------------------------------------------

/// Up to `limit` transfers, each the best (out i, in j) by gain hxp_j − hxp_i among: i held
/// at the start (not bought in this decision), j in the pool, neither held at the start nor
/// already bought, same position, price_j ≤ bank + selling_price(i), and j's club within
/// `team_limit` after the swap. Stops when the best gain is not above `threshold`. Ties:
/// smallest in_key, then smallest out_key. Players without hxp count as 0.
pub fn greedy_transfers(
    state: &SquadState,
    pool: &[PoolPlayer],
    hxp: &HashMap<i64, f64>,
    rules: &Rules,
    threshold: f64,
    limit: i64,
) -> Vec<Transfer> {
    let get = |k: i64| hxp.get(&k).copied().filter(|v| !v.is_nan()).unwrap_or(0.0);
    let original: BTreeSet<i64> = state.player_keys().into_iter().collect();
    let mut buyable: Vec<bool> = pool.iter().map(|p| !original.contains(&p.player_key)).collect();
    // (key, element_type, team, selling price, hxp)
    let mut squad: Vec<(i64, i64, i64, i64, f64)> = state
        .holdings
        .iter()
        .map(|h| {
            let hx = hxp.get(&h.player_key).copied().unwrap_or(0.0);
            (h.player_key, h.element_type, h.team_key, selling_price(h, rules.sell_on_fee), hx)
        })
        .collect();
    let mut sellable = vec![true; squad.len()];
    let mut bank = state.bank;
    let mut transfers = Vec::new();
    while (transfers.len() as i64) < limit {
        if !sellable.iter().any(|s| *s) || !buyable.iter().any(|b| *b) {
            break;
        }
        let mut clubs: HashMap<i64, i64> = HashMap::new();
        for p in &squad {
            *clubs.entry(p.2).or_default() += 1;
        }
        // (gain, in_key, out_key, squad row, pool row)
        let mut best: Option<(f64, i64, i64, usize, usize)> = None;
        for (i, out) in squad.iter().enumerate() {
            if !sellable[i] {
                continue;
            }
            for (j, p) in pool.iter().enumerate() {
                if !buyable[j] || p.element_type != out.1 || p.price > bank + out.3 {
                    continue;
                }
                let after = clubs.get(&p.team_key).copied().unwrap_or(0)
                    - i64::from(out.2 == p.team_key)
                    + 1;
                if after > rules.team_limit {
                    continue;
                }
                let gain = get(p.player_key) - out.4;
                let better = match best {
                    None => true,
                    Some((g, in_key, out_key, _, _)) => {
                        gain > g
                            || (gain == g
                                && (p.player_key < in_key
                                    || (p.player_key == in_key && out.0 < out_key)))
                    }
                };
                if better {
                    best = Some((gain, p.player_key, out.0, i, j));
                }
            }
        }
        let Some((gain, in_key, out_key, i, j)) = best else { break };
        if gain.partial_cmp(&threshold) != Some(std::cmp::Ordering::Greater) {
            break;
        }
        transfers.push(Transfer { out_key, in_key });
        bank += squad[i].3 - pool[j].price;
        squad[i] = (in_key, pool[j].element_type, pool[j].team_key, 0, get(in_key));
        sellable[i] = false;
        buyable[j] = false;
    }
    transfers
}

/// (player_key, element_type) of the squad after `transfers`.
fn after(state: &SquadState, transfers: &[Transfer], pool: &[PoolPlayer]) -> Vec<(i64, i64)> {
    let outs: BTreeSet<i64> = transfers.iter().map(|t| t.out_key).collect();
    let ins: BTreeSet<i64> = transfers.iter().map(|t| t.in_key).collect();
    let mut squad: Vec<(i64, i64)> =
        squad_of(state).into_iter().filter(|(k, _)| !outs.contains(k)).collect();
    squad.extend(
        pool.iter().filter(|p| ins.contains(&p.player_key)).map(|p| (p.player_key, p.element_type)),
    );
    squad
}

// --- the policies ---------------------------------------------------------------------------

/// Chip policy 2's options on the optimizer: season → terminal value table (replacing
/// `params.chip_value_left` in that season), the known-double lookahead, and Triple Captain
/// by rule at this captain xP (needs "3xc" in `params.exclude_chips`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChipPolicy2 {
    pub season_tables: Option<BTreeMap<i64, BTreeMap<String, Vec<f64>>>>,
    pub double_lead: Option<i64>,
    pub tc_rule: Option<f64>,
}

/// Where the optimizer's candidate prices come from: `Fixed` (today's prices over the whole
/// horizon, the default), `Oracle` (the real deadline prices of the horizon's GWs, read
/// from the future: a backtest-only foresight bound, `crate::oracle`) or `Forecast` (the
/// price forecast's expected paths, `crate::forecast_prices`; v1 xP only).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PriceSource {
    #[default]
    Fixed,
    Oracle,
    Forecast,
}

impl PriceSource {
    /// The `prices=` value.
    pub fn label(self) -> &'static str {
        match self {
            PriceSource::Fixed => "fixed",
            PriceSource::Oracle => "oracle",
            PriceSource::Forecast => "forecast",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Policy {
    Roll {
        xp_model: String,
    },
    Greedy {
        xp_model: String,
        threshold: f64,
        horizon: i64,
        decay: f64,
        max_transfers: i64,
    },
    Optimizer {
        xp_model: String,
        params: OptimizerParams,
        chips: bool,
        extra: ChipPolicy2,
        prices: PriceSource,
    },
    ChipGains {
        xp_model: String,
        params: OptimizerParams,
    },
    ChipRules {
        xp_model: String,
        params: OptimizerParams,
        t_tc: f64,
        t_bb: f64,
        k_fh: i64,
        t_wc: f64,
    },
}

fn check_model(xp_model: &str) -> Result<()> {
    if !fplopt_models::MODELS.contains(&xp_model) {
        let mut known = fplopt_models::MODELS.to_vec();
        known.sort_unstable();
        bail!("unknown xp_model '{xp_model}' (MODELS: {known:?})");
    }
    Ok(())
}

fn check_horizon(horizon: i64) -> Result<()> {
    if !(1..=MAX_HORIZON).contains(&horizon) {
        bail!("horizon must be in 1..{MAX_HORIZON} (the xP frames' GWs), got {horizon}");
    }
    Ok(())
}

impl Policy {
    /// Python's `__post_init__` checks.
    pub fn validate(&self) -> Result<()> {
        match self {
            Policy::Roll { xp_model } => check_model(xp_model),
            Policy::Greedy { xp_model, horizon, max_transfers, .. } => {
                check_model(xp_model)?;
                check_horizon(*horizon)?;
                if *max_transfers < 0 {
                    bail!("max_transfers must be >= 0, got {max_transfers}");
                }
                Ok(())
            }
            Policy::Optimizer { xp_model, params, chips, extra, prices } => {
                check_model(xp_model)?;
                if *prices == PriceSource::Forecast && xp_model != crate::price_eval::XP_MODEL {
                    bail!(
                        "prices=forecast with {xp_model} xP: the price forecast uses {} xP",
                        crate::price_eval::XP_MODEL
                    );
                }
                params.validate()?;
                check_horizon(params.horizon)?;
                if extra.tc_rule.is_some()
                    && (!chips || !params.exclude_chips.iter().any(|c| c == "3xc"))
                {
                    bail!("tc_rule needs chips and '3xc' in params.exclude_chips");
                }
                if let Some(lead) = extra.double_lead
                    && lead < 1
                {
                    bail!("double_lead must be >= 1, got {lead}");
                }
                Ok(())
            }
            Policy::ChipGains { xp_model, params } => {
                check_model(xp_model)?;
                check_horizon(params.horizon)
            }
            Policy::ChipRules { xp_model, params, k_fh, .. } => {
                check_model(xp_model)?;
                check_horizon(params.horizon)?;
                if *k_fh < 1 {
                    bail!("k_fh must be >= 1, got {k_fh}");
                }
                Ok(())
            }
        }
    }

    pub fn xp_model(&self) -> &str {
        match self {
            Policy::Roll { xp_model }
            | Policy::Greedy { xp_model, .. }
            | Policy::Optimizer { xp_model, .. }
            | Policy::ChipGains { xp_model, .. }
            | Policy::ChipRules { xp_model, .. } => xp_model,
        }
    }

    /// Where deciding takes candidate prices from (`DecisionContext::prices` for a source
    /// other than `Fixed`); `Fixed` for every policy but the optimizer.
    pub fn price_source(&self) -> PriceSource {
        match self {
            Policy::Optimizer { prices, .. } => *prices,
            _ => PriceSource::Fixed,
        }
    }

    /// Python's policy names (the experiment log keys on them).
    pub fn name(&self) -> String {
        let r = |x: f64| py_float_repr(x);
        match self {
            Policy::Roll { xp_model } => format!("roll({xp_model})"),
            Policy::Greedy { xp_model, threshold, horizon, decay, max_transfers } => {
                let mut extra = String::new();
                if *horizon != 6 {
                    extra += &format!(",h={horizon}");
                }
                if *decay != 0.85 {
                    extra += &format!(",d={}", r(*decay));
                }
                if *max_transfers != 1 {
                    extra += &format!(",n={max_transfers}");
                }
                format!("greedy({xp_model},t={}{extra})", r(*threshold))
            }
            Policy::Optimizer { xp_model, params, chips, extra, prices } => {
                let max_hits = params.max_hits.map_or("inf".to_owned(), |h| h.to_string());
                let mut parts = vec![
                    xp_model.clone(),
                    format!("mh={max_hits}"),
                    format!("m={}", r(params.hit_margin)),
                ];
                parts.extend(optimizer_extras(params));
                if *prices != PriceSource::Fixed {
                    parts.push(prices.label().to_owned());
                }
                if let Some(tables) = &extra.season_tables {
                    parts.push(format!("loso#{}", season_tables_tag(tables)));
                }
                if let Some(lead) = extra.double_lead {
                    parts.push(format!("dl={lead}"));
                }
                if let Some(t) = extra.tc_rule {
                    parts.push(format!("tcr={}", r(t)));
                }
                if *chips {
                    parts.push("chips".to_owned());
                }
                format!("optimizer({})", parts.join(","))
            }
            Policy::ChipGains { xp_model, params } => {
                format!("chipgains({})", inner_name(xp_model, params))
            }
            Policy::ChipRules { xp_model, params, t_tc, t_bb, k_fh, t_wc } => format!(
                "chiprules({},tc={},bb={},fh={k_fh},wc={})",
                inner_name(xp_model, params),
                r(*t_tc),
                r(*t_bb),
                r(*t_wc)
            ),
        }
    }

    /// One GW's decision.
    pub fn decide(&self, ctx: &DecisionContext) -> Result<Decision> {
        match self {
            Policy::Roll { .. } => {
                let lineup = best_lineup(&squad_of(ctx.state), &target_xp(ctx.frame), ctx.rules)?;
                Ok(Decision { lineup, ..Decision::default() })
            }
            Policy::Greedy { threshold, horizon, decay, max_transfers, .. } => {
                let state = refresh(ctx.state, ctx.pool);
                let limit = if state.gw_index == 1 {
                    ctx.rules.squad_size
                } else {
                    (*max_transfers).min(state.free_transfers.max(0))
                };
                let hxp = horizon_xp(ctx.frame, *horizon, *decay);
                let transfers =
                    greedy_transfers(&state, ctx.pool, &hxp, ctx.rules, *threshold, limit);
                let squad = after(&state, &transfers, ctx.pool);
                let lineup = best_lineup(&squad, &target_xp(ctx.frame), ctx.rules)?;
                Ok(Decision { transfers, lineup, ..Decision::default() })
            }
            Policy::Optimizer { extra, .. } => self.decide_optimizer(ctx, extra),
            Policy::ChipGains { params, .. } => {
                let (problem, plans) = no_chip_plan(ctx, params)?;
                let gains = chip_gains(ctx, &problem, params, plans.best())?;
                Ok(Decision {
                    solver_status: Some(plans.solver_status()),
                    mip_gap: Some(plans.best().mip_gap),
                    diagnostics: Some(gains),
                    ..decision_of(plans.best().first())
                })
            }
            Policy::ChipRules { params, .. } => {
                let (problem, plans) = no_chip_plan(ctx, params)?;
                let gains = chip_gains(ctx, &problem, params, plans.best())?;
                if let Some(chip) = self.choose(ctx, &gains)? {
                    match solve_with_chip(&problem, params, &chip)? {
                        Some(plan) => {
                            return Ok(Decision {
                                solver_status: Some(plan.status.clone()),
                                mip_gap: Some(plan.mip_gap),
                                diagnostics: Some(gains),
                                ..decision_of(plan.first())
                            });
                        }
                        None => log::warn!(
                            "{}: no plan with {chip} at gw_index {}",
                            self.name(),
                            ctx.state.gw_index
                        ),
                    }
                }
                Ok(Decision {
                    solver_status: Some(plans.solver_status()),
                    mip_gap: Some(plans.best().mip_gap),
                    diagnostics: Some(gains),
                    ..decision_of(plans.best().first())
                })
            }
        }
    }

    /// The optimizer policy's planner input, parameters and plans at a context: what
    /// `decide` executes the first GW of plan #1 from (before the rule Triple Captain).
    pub fn optimizer_plans(
        &self,
        ctx: &DecisionContext,
    ) -> Result<(PlanInput, OptimizerParams, PlanSet)> {
        let Policy::Optimizer { params, chips, extra, prices, .. } = self else {
            bail!("optimizer_plans: {} is not an optimizer policy", self.name());
        };
        let mut params = params.clone();
        if let Some(tables) = &extra.season_tables {
            let Some(table) = tables.get(&ctx.state.season) else {
                bail!("no leave-one-season-out table for season {}", ctx.state.season);
            };
            params.chip_value_left = Some(table.clone());
        }
        let mut problem =
            PlanInput::from_context(ctx.state, ctx.pool, ctx.frame, ctx.rules, &params)?;
        if let Some(lead) = extra.double_lead {
            problem.doubles = known_doubles(ctx.view, ctx.state.season, ctx.state.gw_index, lead)?;
        }
        match (*prices, ctx.prices) {
            (PriceSource::Fixed, _) => {}
            (PriceSource::Oracle, Some(PriceInput::Oracle(future))) => {
                let season = ctx.state.season;
                let gws = problem.gws.clone();
                problem.set_price_paths(|p| Some(future.path(season, p, &gws)));
            }
            (PriceSource::Forecast, Some(PriceInput::Forecast(forecast))) => {
                if forecast.season != ctx.state.season {
                    bail!(
                        "prices=forecast: the forecast is {}'s, the decision is in {}",
                        forecast.season,
                        ctx.state.season
                    );
                }
                let paths = forecast.paths(ctx.view, ctx.xp, &problem.gws)?;
                problem.set_price_paths(|p| paths.get(&p.player_key).cloned());
            }
            (source, _) => bail!(
                "prices={} needs its price paths, which only the backtester's decision paths \
                 provide (DecisionContext.prices is not {0})",
                source.label()
            ),
        }
        let plans = optimize(&problem, &params, 1, *chips, false, Search::Bound)?;
        Ok((problem, params, plans))
    }

    fn decide_optimizer(&self, ctx: &DecisionContext, extra: &ChipPolicy2) -> Result<Decision> {
        let (problem, params, plans) = self.optimizer_plans(ctx)?;
        let status = plans.solver_status();
        if status != "Optimal" {
            log::warn!(
                "{} at {} gw_index {}: solver stopped at a limit ({status}, gap {:.2}%); \
                 executing the best plan found",
                self.name(),
                ctx.view.deadline(),
                ctx.state.gw_index,
                100.0 * plans.best().mip_gap
            );
        }
        let decision = Decision {
            solver_status: Some(status),
            mip_gap: Some(plans.best().mip_gap),
            ..decision_of(plans.best().first())
        };
        if let Some(tc_rule) = extra.tc_rule
            && decision.chip.is_none()
            && let Some(tc) = self.rule_tc(ctx, &problem, &params, plans.best(), tc_rule)?
        {
            return Ok(tc);
        }
        Ok(decision)
    }

    /// Triple Captain this GW when available and plan #1's captain's xP ≥ `tc_rule` or this
    /// is its window's last GW: re-planned with TC fixed here (None otherwise, or if that
    /// plan is infeasible).
    fn rule_tc(
        &self,
        ctx: &DecisionContext,
        problem: &PlanInput,
        params: &OptimizerParams,
        best: &Plan,
        tc_rule: f64,
    ) -> Result<Option<Decision>> {
        let (state, rules) = (ctx.state, ctx.rules);
        if !chip_available(state, "3xc", rules) {
            return Ok(None);
        }
        let window = chip_window(state, "3xc", rules)?;
        let stop = rules.chips.iter().rev().find(|w| w.chip_id == window).map_or(0, |w| w.stop);
        let captain = target_xp_of(ctx.frame, best.first().captain);
        if captain < tc_rule && state.gw_index != stop {
            return Ok(None);
        }
        match solve_with_chip(problem, params, "3xc")? {
            Some(plan) => Ok(Some(Decision {
                solver_status: Some(plan.status.clone()),
                mip_gap: Some(plan.mip_gap),
                ..decision_of(plan.first())
            })),
            None => {
                log::warn!("{}: no plan with 3xc at gw_index {}", self.name(), state.gw_index);
                Ok(None)
            }
        }
    }

    /// The chip the rule policy plays this GW (last chance, then the rules), or None.
    fn choose(
        &self,
        ctx: &DecisionContext,
        gains: &BTreeMap<String, Option<f64>>,
    ) -> Result<Option<String>> {
        let Policy::ChipRules { t_tc, t_bb, k_fh, t_wc, .. } = self else {
            bail!("choose: not a chip rule policy");
        };
        let t = RuleThresholds { t_tc: *t_tc, t_bb: *t_bb, k_fh: *k_fh, t_wc: *t_wc };
        rule_chip(ctx.state, ctx.rules, gains, &t)
    }
}

fn inner_name(xp_model: &str, params: &OptimizerParams) -> String {
    let name = Policy::Optimizer {
        xp_model: xp_model.to_owned(),
        params: params.clone(),
        chips: false,
        extra: ChipPolicy2::default(),
        prices: PriceSource::Fixed,
    }
    .name();
    name.trim_start_matches("optimizer(").trim_end_matches(')').to_owned()
}

/// The first GW's decision of a plan.
pub fn decision_of(first: &GwPlan) -> Decision {
    Decision {
        transfers: first.transfers.clone(),
        lineup: Lineup {
            starters: first.starters.clone(),
            bench: first.bench.clone(),
            captain: first.captain,
            vice: first.vice,
        },
        chip: first.chip.clone(),
        ..Decision::default()
    }
}

/// First-last value and a weighted checksum per key, so a re-estimated table gives another
/// policy name.
fn table_tag(table: &BTreeMap<String, Vec<f64>>) -> String {
    let parts: Vec<String> = table
        .iter()
        .map(|(k, v)| {
            // Python's `sum()` (compensated): a plain loop can differ in the last bit and so
            // in the 3rd decimal (2559.4185 rounds either way; chip_shrink=0.75's FH table)
            let checksum = py_sum(v.iter().enumerate().map(|(i, x)| (i + 1) as f64 * x));
            format!("{k}:{:.2}-{:.2}#{checksum:.3}", v[0], v[v.len() - 1])
        })
        .collect();
    parts.join("/")
}

/// 10 hex digits of the SHA-1 of the per-season tables' canonical JSON (sorted keys, no
/// spaces, Python's float reprs).
fn season_tables_tag(tables: &BTreeMap<i64, BTreeMap<String, Vec<f64>>>) -> String {
    let mut seasons: Vec<(String, &BTreeMap<String, Vec<f64>>)> =
        tables.iter().map(|(s, t)| (s.to_string(), t)).collect();
    seasons.sort_by(|a, b| a.0.cmp(&b.0));
    let body: Vec<String> = seasons
        .iter()
        .map(|(s, t)| {
            let keys: Vec<String> = t
                .iter()
                .map(|(k, v)| {
                    let values: Vec<String> = v.iter().map(|x| py_float_repr(*x)).collect();
                    format!("{}:[{}]", serde_json::to_string(k).unwrap(), values.join(","))
                })
                .collect();
            format!("\"{s}\":{{{}}}", keys.join(","))
        })
        .collect();
    let canonical = format!("{{{}}}", body.join(","));
    let digest = Sha1::digest(canonical.as_bytes());
    digest.iter().map(|b| format!("{b:02x}")).collect::<String>()[..10].to_owned()
}

/// `OptimizerPolicy.name` parts for the parameters (other than the hit safeguards) that
/// differ from `OptimizerParams::default()`.
fn optimizer_extras(params: &OptimizerParams) -> Vec<String> {
    let d = OptimizerParams::default();
    let r = |x: f64| py_float_repr(x);
    let mut parts = Vec::new();
    if params.horizon != d.horizon {
        parts.push(format!("h={}", params.horizon));
    }
    if params.decay != d.decay {
        parts.push(format!("d={}", r(params.decay)));
    }
    if params.itb_value != d.itb_value {
        parts.push(format!("itb={}", r(params.itb_value)));
    }
    if params.ft_value != d.ft_value {
        let items: Vec<String> =
            params.ft_value.iter().map(|(k, v)| format!("{k}:{}", r(*v))).collect();
        parts.push(format!("ftv={}", items.join("/")));
    }
    if params.bench_weights != d.bench_weights {
        let items: Vec<String> = params.bench_weights.iter().map(|w| r(*w)).collect();
        parts.push(format!("bw={}", items.join("/")));
    }
    if params.chip_value != d.chip_value {
        let items: Vec<String> =
            params.chip_value.iter().map(|(k, v)| format!("{k}:{}", r(*v))).collect();
        parts.push(format!("cv={}", items.join("/")));
    }
    if let Some(table) = &params.chip_value_left {
        parts.push(format!("cvl={}", table_tag(table)));
    }
    if params.prune_n != d.prune_n {
        let prune = match &params.prune_n {
            None => "all".to_owned(),
            Some(p) => p.iter().map(|(k, v)| format!("{k}:{v}")).collect::<Vec<_>>().join("/"),
        };
        parts.push(format!("prune={prune}"));
    }
    if params.prune_dominated != d.prune_dominated {
        parts.push(format!("dom={}", i64::from(params.prune_dominated)));
    }
    if params.mip_gap != d.mip_gap {
        parts.push(format!("gap={}", r(params.mip_gap)));
    }
    if params.threads != d.threads {
        parts.push(format!("threads={}", params.threads));
    }
    if params.time_limit != d.time_limit {
        parts.push(format!("tl={}", params.time_limit.map_or("None".to_owned(), r)));
    }
    if params.node_limit != d.node_limit {
        parts.push(format!(
            "nodes={}",
            params.node_limit.map_or("None".to_owned(), |n| n.to_string())
        ));
    }
    if params.tie_epsilon != d.tie_epsilon {
        parts.push(format!("eps={}", r(params.tie_epsilon)));
    }
    if params.bench_from_minutes != d.bench_from_minutes {
        parts.push(format!("bfm={}", i64::from(params.bench_from_minutes)));
    }
    if params.min_minutes != d.min_minutes {
        parts.push(format!("minm={}", r(params.min_minutes)));
    }
    if params.exclude_chips != d.exclude_chips {
        parts.push(format!("nochip={}", params.exclude_chips.join("/")));
    }
    if params.fh_scale != d.fh_scale {
        parts.push(format!("fhs={}", r(params.fh_scale)));
    }
    if params.price_value_k != d.price_value_k {
        parts.push(format!("pk={}", r(params.price_value_k)));
    }
    parts
}

// --- chip study ---------------------------------------------------------------------------

/// The `gw_index`s in (gw_index, gw_index + lead] where a club has ≥ 2 fixtures in the view's
/// as-of schedule (unscheduled fixtures and placeholder sides, team_key ≤ 0, do not count).
pub fn known_doubles(
    view: &AsOfView,
    season: i64,
    gw_index: i64,
    lead: i64,
) -> Result<BTreeSet<i64>> {
    let schedule = view.schedule(season)?;
    let index = schedule.opt_i64("gw_index")?;
    let (home, away) = (schedule.opt_i64("home_team_key")?, schedule.opt_i64("away_team_key")?);
    let mut counts: HashMap<(i64, i64), i64> = HashMap::new();
    for i in 0..schedule.height() {
        let Some(g) = index[i] else { continue };
        for team in [home[i], away[i]].into_iter().flatten() {
            if team > 0 {
                *counts.entry((g, team)).or_default() += 1;
            }
        }
    }
    Ok(counts
        .into_iter()
        .filter(|(_, n)| *n >= 2)
        .map(|((g, _), _)| g)
        .filter(|&g| gw_index < g && g <= gw_index + lead)
        .collect())
}

/// Squad players whose club has no fixture in the target GW.
fn blanks(ctx: &DecisionContext) -> Result<i64> {
    let coverage = pool_coverage(ctx.view)?;
    let (teams, n) = (coverage.i64("team_key")?, coverage.i64("n_fixtures")?);
    let fixtures: HashMap<i64, i64> = teams.iter().copied().zip(n.iter().copied()).collect();
    Ok(ctx
        .state
        .holdings
        .iter()
        .filter(|h| fixtures.get(&h.team_key).copied().unwrap_or(0) == 0)
        .count() as i64)
}

/// The plan playing `chip` in the first horizon GW, or None if the solve found no plan.
fn solve_with_chip(
    problem: &PlanInput,
    params: &OptimizerParams,
    chip: &str,
) -> Result<Option<Plan>> {
    let scenario = make_scenario(problem, &BTreeMap::from([(0, chip.to_owned())]))?;
    match solve_plan(problem, params, None, &[], &scenario) {
        Ok(plan) => Ok(Some(plan)),
        Err(SolveError::Infeasible(_) | SolveError::LimitReached { .. }) => Ok(None),
        Err(SolveError::Other(e)) => Err(e),
    }
}

fn no_chip_plan(
    ctx: &DecisionContext,
    params: &OptimizerParams,
) -> Result<(PlanInput, fplopt_optimize::PlanSet)> {
    let problem = PlanInput::from_context(ctx.state, ctx.pool, ctx.frame, ctx.rules, params)?;
    let plans = optimize(&problem, params, 1, false, false, Search::Bound)?;
    Ok((problem, plans))
}

/// The `CHIP_DIAGNOSTICS` at this GW for the no-chip `plan` of `problem`.
pub fn chip_gains(
    ctx: &DecisionContext,
    problem: &PlanInput,
    params: &OptimizerParams,
    plan: &Plan,
) -> Result<BTreeMap<String, Option<f64>>> {
    fplopt_optimize::chip_rules::chip_gains(
        ctx.state,
        ctx.pool,
        ctx.frame,
        ctx.rules,
        problem,
        params,
        plan,
        blanks(ctx)?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_tag_sums_like_python() {
        // chip_shrink=0.75 × the estimated Free Hit row: Python's sum() gives 2559.4185
        // (→ .419), a plain loop 2559.4184999999998 (→ .418)
        let row = vec![
            4.58955,
            7.264049999999999,
            9.10365,
            10.3932,
            10.56705,
            11.093175,
            11.4309,
            11.604375000000001,
            12.815100000000001,
            13.008375000000001,
            13.199025,
            13.5231,
            13.597724999999999,
            13.604625000000002,
            13.6524,
            13.819875,
            14.4327,
            14.9334,
            16.7889,
        ];
        let table = BTreeMap::from([("freehit".to_string(), row)]);
        assert_eq!(table_tag(&table), "freehit:4.59-16.79#2559.419");
    }
}
