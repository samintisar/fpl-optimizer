//! The season simulator (Python `fplopt.backtest.simulator`; PLAN §5 *Simulator*).
//!
//! `simulate(store, rules, policy, start)` replays one season GW by GW from a state: at each
//! deadline the pool and the policy's xP frame (cached per (model, deadline)), the policy's
//! decision, `apply_decision`, then the outcomes from `store.as_of(lockdown + 1 µs)` scored
//! twice (realized points and xG points, same minutes so same autosubs), the GW row, and
//! `next_state`. A GW without outcomes ends the run (decided, not scored). After the run each
//! GW with transfers gets its predicted and realized transfer gain over the next
//! `GAIN_WINDOW` recorded GWs (one under a Free Hit).

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Result, anyhow, bail};
use fplopt_build::scoring::Rules;
use fplopt_core::seasons::{HOLDOUT_SEASONS, season_label};
use fplopt_features::baseline::player_pool;
use fplopt_models::assemble::{V1Fit, V1Params, fit_v1, predict_v1};
use fplopt_models::fitted::{REFIT_EVERY, refit_deadline};
use fplopt_optimize::XpFrame;
use fplopt_optimize::start::{FAR_FUTURE, pool_players};
use fplopt_store::{AsOfView, DataStore, Micros, Table};

use crate::gw_score::score_gameweek;
use crate::outcomes::{GwOutcomes, read_outcomes};
use crate::policies::{CHIP_DIAGNOSTICS, DecisionContext, Policy, best_lineup};
use crate::state::{
    Decision, PoolPlayer, SquadState, apply_decision, next_state, refresh, squad_value,
};
use crate::xp_cache::DiskCache;

/// GWs over which transfer gains are measured (PLAN §5's per-decision k).
pub const GAIN_WINDOW: usize = 4;

/// A policy's xP frame at a deadline: the model's table and the planner's copy.
pub struct Xp {
    pub table: Table,
    pub frame: XpFrame,
}

/// A map whose values are computed once per key: the first thread to ask computes, the
/// others asking for the same key wait for it (units of one season reach the same deadlines
/// at the same time, and a v1 fit takes seconds). Errors are kept as their text.
struct OnceMap<K, V>(Mutex<HashMap<K, Arc<OnceLock<std::result::Result<Arc<V>, String>>>>>);

impl<K, V> Default for OnceMap<K, V> {
    fn default() -> Self {
        OnceMap(Mutex::new(HashMap::new()))
    }
}

impl<K: Eq + Hash, V> OnceMap<K, V> {
    fn get_or_compute(&self, key: K, compute: impl FnOnce() -> Result<V>) -> Result<Arc<V>> {
        let cell = self.0.lock().unwrap().entry(key).or_default().clone();
        cell.get_or_init(|| compute().map(Arc::new).map_err(|e| format!("{e:#}")))
            .clone()
            .map_err(|e| anyhow!(e))
    }
}

/// xP frames per (model, deadline), v1 fits per cutoff, pools per deadline and outcomes per
/// (season, gw, rules label), shared across policies, start states and threads; the xP
/// frames also on disk when given one (`xp_cache`), shared across runs.
#[derive(Default)]
pub struct Caches {
    disk: Option<DiskCache>,
    xp: OnceMap<(String, Micros), Xp>,
    fits: OnceMap<Micros, V1Fit>,
    pools: OnceMap<Micros, Vec<PoolPlayer>>,
    outcomes: OnceMap<(i64, i64, String), GwOutcomes>,
}

impl Caches {
    /// Caches whose xP frames are also read from and written to `disk`.
    pub fn with_disk(disk: DiskCache) -> Caches {
        Caches { disk: Some(disk), ..Caches::default() }
    }

    pub fn pool(&self, view: &AsOfView) -> Result<Arc<Vec<PoolPlayer>>> {
        self.pools.get_or_compute(view.deadline(), || pool_players(&player_pool(view)?))
    }

    /// The model's xP frame at the view (v1: predicted with the memoized fit at its refit
    /// cutoff, the same as calling the model).
    pub fn xp(&self, model: &str, view: &AsOfView) -> Result<Arc<Xp>> {
        self.xp.get_or_compute((model.to_owned(), view.deadline()), || {
            let cached = match &self.disk {
                Some(disk) => disk.load(model, view.deadline())?,
                None => None,
            };
            let hit = cached.is_some();
            let table = if let Some(t) = cached {
                t
            } else if model == "v1" {
                let cutoff = refit_deadline(view, REFIT_EVERY)?;
                let fit = self.fits.get_or_compute(cutoff, || {
                    fit_v1(&view.earlier(cutoff)?, &V1Params::default())
                })?;
                predict_v1(view, &fit)?
            } else {
                fplopt_models::predict(model, view)?
            };
            if let (Some(disk), false) = (&self.disk, hit) {
                disk.store(model, view.deadline(), &table)?;
            }
            let frame = XpFrame::from_table(&table)?;
            Ok(Xp { table, frame })
        })
    }

    pub fn outcomes(
        &self,
        store: &DataStore,
        rules: &Rules,
        season: i64,
        gw: i64,
        lockdown: Micros,
    ) -> Result<Arc<GwOutcomes>> {
        self.outcomes.get_or_compute((season, gw, rules.label.clone()), || {
            read_outcomes(store, rules, season, gw, lockdown)
        })
    }
}

/// One schedule row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gameweek {
    pub season: i64,
    pub gw: i64,
    pub gw_index: i64,
    pub deadline: Micros,
    pub lockdown: Micros,
}

/// The season's GWs from `gw_index` on (sorted by gw_index), as visible at the deadline of
/// `gw_index`: the start deadline is the one, of every deadline ever listed for that
/// gw_index, whose own view lists exactly one row for it with that deadline.
pub fn season_schedule(store: &DataStore, season: i64, gw_index: i64) -> Result<Vec<Gameweek>> {
    type Row = (i64, i64, Option<Micros>, Option<Micros>);
    let rows = |view: AsOfView| -> Result<Vec<Row>> {
        let t = view.table(
            "gameweek",
            Some(&["season", "gw", "gw_index", "deadline_time", "lockdown_time"]),
        )?;
        let (s, g, i) = (t.opt_i64("season")?, t.opt_i64("gw")?, t.opt_i64("gw_index")?);
        let (d, l) = (t.opt_i64("deadline_time")?, t.opt_i64("lockdown_time")?);
        Ok((0..t.height())
            .filter(|&r| s[r] == Some(season) && i[r].is_some_and(|x| x >= gw_index))
            .map(|r| (g[r].unwrap_or(0), i[r].unwrap(), d[r], l[r]))
            .collect())
    };
    let probe = rows(store.as_of(FAR_FUTURE))?;
    let mut candidates: Vec<Micros> =
        probe.iter().filter(|r| r.1 == gw_index).filter_map(|r| r.2).collect();
    candidates.sort_unstable();
    candidates.dedup();
    let mut found = Vec::new();
    for deadline in candidates {
        let gameweeks = rows(store.as_of(deadline))?;
        let start: Vec<_> = gameweeks.iter().filter(|r| r.1 == gw_index).collect();
        if start.len() == 1 && start[0].2 == Some(deadline) {
            found.push(gameweeks);
        }
    }
    if found.len() != 1 {
        bail!(
            "{}: {} gameweeks with gw_index {gw_index} visible at their own deadline",
            season_label(season as i32),
            found.len()
        );
    }
    let mut gameweeks = found.pop().unwrap();
    gameweeks.sort_by_key(|r| r.1); // stable, as mergesort
    if gameweeks.windows(2).any(|w| w[0].1 == w[1].1) {
        bail!("{}: duplicate gw_index in the gameweek table", season_label(season as i32));
    }
    gameweeks
        .into_iter()
        .map(|(gw, gw_index, d, l)| {
            let (Some(deadline), Some(lockdown)) = (d, l) else {
                bail!("{} GW{gw}: no deadline or lockdown", season_label(season as i32));
            };
            Ok(Gameweek { season, gw, gw_index, deadline, lockdown })
        })
        .collect()
}

/// One recorded GW (Python's `GW_COLUMNS`).
#[derive(Clone, Debug, PartialEq)]
pub struct GwRow {
    pub season: i64,
    pub gw: i64,
    pub gw_index: i64,
    pub deadline: Micros,
    pub chip: Option<String>,
    pub n_transfers: i64,
    pub hits: i64,
    pub hit_points: i64,
    pub points: i64,
    pub net_points: i64,
    pub bench_points: i64,
    pub xg_points: Option<f64>,
    pub xg_net_points: Option<f64>,
    pub captain: i64,
    pub vice: i64,
    pub captain_used: Option<i64>,
    pub captain_points: i64,
    pub captain_regret: i64,
    pub xi_regret: Option<i64>,
    pub bank: i64,
    pub free_transfers: i64,
    pub squad_value: i64,
    pub transfers: String,
    pub autosubs: String,
    pub pred_gain: Option<f64>,
    pub real_gain: Option<f64>,
    pub solver_status: Option<String>,
    pub mip_gap: Option<f64>,
    /// `CHIP_DIAGNOSTICS`, in order.
    pub diagnostics: [Option<f64>; 7],
}

/// A GW's decision: the refreshed pre-decision state, the pool and the policy's decision.
pub struct Step {
    pub state: SquadState,
    pub pool: Arc<Vec<PoolPlayer>>,
    pub decision: Decision,
}

/// Who decides a GW: a policy, or a decision already made for it (a per-decision arm's
/// first GW, decided from the same state at the same deadline).
#[derive(Clone, Copy)]
pub enum Decider<'a> {
    Policy(&'a Policy),
    Decided(&'a Decision),
}

/// Decide one GW: only the deadline view is read.
pub fn decide_step(
    store: &DataStore,
    rules: &Rules,
    decider: Decider,
    state: &SquadState,
    deadline: Micros,
    caches: &Caches,
) -> Result<Step> {
    let view = store.as_of(deadline);
    let pool = caches.pool(&view)?;
    let state = refresh(state, &pool);
    let policy = match decider {
        Decider::Decided(decision) => {
            return Ok(Step { state, pool, decision: decision.clone() });
        }
        Decider::Policy(policy) => policy,
    };
    let xp = caches.xp(policy.xp_model(), &view)?;
    let ctx = DecisionContext {
        view: &view,
        state: &state,
        rules,
        pool: &pool,
        xp: &xp.table,
        frame: &xp.frame,
    };
    let decision = policy.decide(&ctx)?;
    Ok(Step { state, pool, decision })
}

/// The squad's best GW score in hindsight: the best valid XI on realized points plus the
/// best captain among its starters who played (no autosubs, no chip).
pub fn hindsight_points(
    squad: &[(i64, i64)],
    outcomes: &HashMap<i64, (i64, i64)>,
    rules: &Rules,
) -> Result<i64> {
    let points: HashMap<i64, i64> =
        squad.iter().map(|(k, _)| (*k, outcomes.get(k).map_or(0, |o| o.0))).collect();
    let as_xp: HashMap<i64, f64> = points.iter().map(|(k, p)| (*k, *p as f64)).collect();
    let lineup = best_lineup(squad, &as_xp, rules)?;
    let played: Vec<i64> = lineup
        .starters
        .iter()
        .filter(|k| outcomes.get(k).is_some_and(|o| o.1 > 0))
        .map(|k| points[k])
        .collect();
    Ok(lineup.starters.iter().map(|k| points[k]).sum::<i64>()
        + played.iter().max().copied().unwrap_or(0))
}

fn pairs_json(pairs: impl IntoIterator<Item = (i64, i64)>) -> String {
    let items: Vec<String> = pairs.into_iter().map(|(a, b)| format!("[{a}, {b}]")).collect();
    format!("[{}]", items.join(", "))
}

/// Apply the decision, score it, and return (gw_state, record, GW row).
pub fn play_step(
    step: &Step,
    rules: &Rules,
    outcomes: &GwOutcomes,
    gw: i64,
    deadline: Micros,
) -> Result<(SquadState, crate::state::Applied, GwRow)> {
    let (state, decision) = (&step.state, &step.decision);
    let (gw_state, applied) = apply_decision(state, decision, &step.pool, rules)?;
    let positions: HashMap<i64, i64> =
        gw_state.holdings.iter().map(|h| (h.player_key, h.element_type)).collect();
    let lineup = &decision.lineup;
    let chip = decision.chip.as_deref();
    let score = score_gameweek(lineup, &outcomes.realized, &positions, rules, chip)?;
    let xg_score = score_gameweek(lineup, &outcomes.xg, &positions, rules, chip)?;
    let xg_points = (!xg_score.points.is_nan()).then_some(xg_score.points);
    let raw = |key: Option<i64>| key.map_or(0, |k| outcomes.realized.get(&k).map_or(0, |o| o.0));
    let captain_points = raw(score.captain_used);
    let best_counted = score.counted.iter().map(|k| raw(Some(*k))).max().unwrap_or(0);
    let xi_regret = if chip.is_none() {
        let squad: Vec<(i64, i64)> =
            gw_state.holdings.iter().map(|h| (h.player_key, h.element_type)).collect();
        Some(hindsight_points(&squad, &outcomes.realized, rules)? - score.points)
    } else {
        None
    };
    let mut diagnostics = [None; 7];
    if let Some(d) = &decision.diagnostics {
        for (slot, key) in diagnostics.iter_mut().zip(CHIP_DIAGNOSTICS) {
            *slot = d.get(key).copied().flatten();
        }
    }
    let row = GwRow {
        season: state.season,
        gw,
        gw_index: state.gw_index,
        deadline,
        chip: decision.chip.clone(),
        n_transfers: applied.n_transfers,
        hits: applied.hits,
        hit_points: applied.hit_points,
        points: score.points,
        net_points: score.points - applied.hit_points,
        bench_points: score.bench_points,
        xg_points,
        xg_net_points: xg_points.map(|x| x - applied.hit_points as f64),
        captain: lineup.captain,
        vice: lineup.vice,
        captain_used: score.captain_used,
        captain_points,
        captain_regret: best_counted - captain_points,
        xi_regret,
        bank: gw_state.bank,
        free_transfers: state.free_transfers,
        squad_value: squad_value(&gw_state, rules.sell_on_fee),
        transfers: pairs_json(decision.transfers.iter().map(|t| (t.out_key, t.in_key))),
        autosubs: pairs_json(score.autosubs.iter().copied()),
        pred_gain: None,
        real_gain: None,
        solver_status: decision.solver_status.clone(),
        mip_gap: decision.mip_gap,
        diagnostics,
    };
    Ok((gw_state, applied, row))
}

/// pandas' grouped float sum (Kahan in row order, nulls skipped).
fn kahan(values: impl IntoIterator<Item = f64>) -> f64 {
    let (mut sum, mut comp) = (0.0, 0.0);
    for v in values {
        if v.is_nan() {
            continue;
        }
        let y = v - comp;
        let t = sum + y;
        comp = t - sum - y;
        sum = t;
    }
    sum
}

/// Per decision i (decided at schedule row i): (predicted, realized) gain of its transfers
/// over schedule rows i..i+window−1 that were recorded (a Free Hit: row i only), or (None,
/// None) without transfers. xP from the decision deadline's frame (the one the policy
/// saw), points from the outcomes.
pub fn transfer_gains(
    store: &DataStore,
    rules: &Rules,
    xp_model: &str,
    decisions: &[Decision],
    schedule: &[Gameweek],
    caches: &Caches,
    window: usize,
) -> Result<Vec<(Option<f64>, Option<f64>)>> {
    let gameweeks = &schedule[..decisions.len().min(schedule.len())];
    let mut out = Vec::with_capacity(decisions.len());
    for (i, decision) in decisions.iter().enumerate() {
        if decision.transfers.is_empty() {
            out.push((None, None));
            continue;
        }
        let span = if decision.chip.as_deref() == Some("freehit") { 1 } else { window };
        let gws = &gameweeks[i..(i + span).min(gameweeks.len())];
        let ins: Vec<i64> = decision.transfers.iter().map(|t| t.in_key).collect();
        let outs: Vec<i64> = decision.transfers.iter().map(|t| t.out_key).collect();
        let xp = caches.xp(xp_model, &store.as_of(gws[0].deadline))?;
        let offsets: Vec<i64> = gws.iter().map(|g| g.gw_index - gws[0].gw_index).collect();
        let f = &xp.frame;
        let by_key = |key: i64| -> f64 {
            kahan(
                (0..f.len())
                    .filter(|&r| f.player_key[r] == key && offsets.contains(&f.horizon[r]))
                    .map(|r| f.xp[r].unwrap_or(f64::NAN)),
            )
        };
        let side = |keys: &[i64]| -> f64 {
            let values: Vec<f64> = keys.iter().map(|&k| by_key(k)).collect();
            fplopt_core::numpy::np_sum(&values)
        };
        let predicted = side(&ins) - side(&outs);
        let mut realized = 0i64;
        for g in gws {
            let o = caches.outcomes(store, rules, g.season, g.gw, g.lockdown)?;
            realized += ins.iter().map(|k| o.realized.get(k).map_or(0, |x| x.0)).sum::<i64>();
            realized -= outs.iter().map(|k| o.realized.get(k).map_or(0, |x| x.0)).sum::<i64>();
        }
        out.push((Some(predicted), Some(realized as f64)));
    }
    Ok(out)
}

/// One simulated season: rows per GW, the policy's decisions and the refreshed states it
/// saw, the state after the last GW and Σ net points. A run stopped at a GW without outcomes
/// keeps that GW's state and decision as pending.
pub struct SeasonRun {
    pub policy: String,
    pub rows: Vec<GwRow>,
    pub decisions: Vec<Decision>,
    pub states: Vec<SquadState>,
    pub final_state: SquadState,
    pub total: i64,
    pub pending_state: Option<SquadState>,
    pub pending_decision: Option<Decision>,
}

/// What `run_gameweeks` returns: rows, decisions, states, the final state, the unscored step.
pub type Played = (Vec<GwRow>, Vec<Decision>, Vec<SquadState>, SquadState, Option<Step>);

/// Simulate the GWs of `schedule` (consecutive rows from `start.gw_index`); `policy_for(i)`
/// decides the i-th GW. A GW without outcomes stops the run unrecorded.
pub fn run_gameweeks<'p>(
    store: &DataStore,
    rules: &Rules,
    policy_for: impl Fn(usize) -> Decider<'p>,
    start: &SquadState,
    schedule: &[Gameweek],
    caches: &Caches,
) -> Result<Played> {
    let mut state = start.clone();
    let (mut rows, mut decisions, mut states) = (Vec::new(), Vec::new(), Vec::new());
    let mut pending = None;
    for (i, g) in schedule.iter().enumerate() {
        if state.gw_index != g.gw_index {
            bail!("state is at gw_index {}, schedule at {}", state.gw_index, g.gw_index);
        }
        let step = decide_step(store, rules, policy_for(i), &state, g.deadline, caches)?;
        let outcomes = caches.outcomes(store, rules, state.season, g.gw, g.lockdown)?;
        if outcomes.is_empty() {
            log::warn!(
                "{} GW{}: no outcomes as of its lockdown (not played yet?); decided, not \
                 scored; stopping",
                season_label(state.season as i32),
                g.gw
            );
            pending = Some(step);
            break;
        }
        let (gw_state, applied, row) = play_step(&step, rules, &outcomes, g.gw, g.deadline)?;
        rows.push(row);
        states.push(step.state.clone());
        decisions.push(step.decision);
        let following = schedule.get(i + 1).map_or(g.gw_index + 1, |n| n.gw_index);
        state = next_state(&gw_state, &applied.record(), rules, following)?;
    }
    Ok((rows, decisions, states, state, pending))
}

/// Replay `start.season` from `start` with `policy` through the season's last GW, or
/// through `end_gw_index` (inclusive). Refuses a holdout season unless `allow_holdout`.
pub fn simulate(
    store: &DataStore,
    rules: &Rules,
    policy: &Policy,
    start: &SquadState,
    end_gw_index: Option<i64>,
    caches: &Caches,
    allow_holdout: bool,
) -> Result<SeasonRun> {
    let season = start.season;
    if HOLDOUT_SEASONS.contains(&(season as i32)) && !allow_holdout {
        bail!(
            "HoldoutError: {} is a holdout season; simulate refuses it until Phase 6",
            season_label(season as i32)
        );
    }
    if start.freehit_backup.is_some() {
        bail!("start state is a Free Hit gameweek state");
    }
    let mut schedule = season_schedule(store, season, start.gw_index)?;
    if let Some(end) = end_gw_index {
        schedule.retain(|g| g.gw_index <= end);
    }
    let (mut rows, decisions, states, final_state, pending) =
        run_gameweeks(store, rules, |_| Decider::Policy(policy), start, &schedule, caches)?;
    let gains = transfer_gains(
        store,
        rules,
        policy.xp_model(),
        &decisions,
        &schedule,
        caches,
        GAIN_WINDOW,
    )?;
    for (row, (predicted, realized)) in rows.iter_mut().zip(gains) {
        row.pred_gain = predicted;
        row.real_gain = realized;
    }
    let total = rows.iter().map(|r| r.net_points).sum();
    log::info!(
        "{} {} from gw_index {}: {} GWs, {total} points ({} transfers, {} hit points)",
        season_label(season as i32),
        policy.name(),
        start.gw_index,
        rows.len(),
        rows.iter().map(|r| r.n_transfers).sum::<i64>(),
        rows.iter().map(|r| r.hit_points).sum::<i64>()
    );
    let (pending_state, pending_decision) = match pending {
        Some(step) => (Some(step.state), Some(step.decision)),
        None => (None, None),
    };
    Ok(SeasonRun {
        policy: policy.name(),
        rows,
        decisions,
        states,
        final_state,
        total,
        pending_state,
        pending_decision,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn once_map_computes_each_key_once_across_threads() {
        let map: OnceMap<i64, i64> = OnceMap::default();
        let calls = AtomicUsize::new(0);
        let results: Vec<i64> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..16)
                .map(|i| {
                    let (map, calls) = (&map, &calls);
                    scope.spawn(move || {
                        let key = i % 2;
                        *map.get_or_compute(key, || {
                            calls.fetch_add(1, Ordering::SeqCst);
                            std::thread::sleep(std::time::Duration::from_millis(50));
                            Ok(key * 10)
                        })
                        .unwrap()
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        for (i, r) in results.iter().enumerate() {
            assert_eq!(*r, (i as i64 % 2) * 10);
        }
    }

    #[test]
    fn once_map_hands_an_error_to_every_caller() {
        let map: OnceMap<i64, i64> = OnceMap::default();
        let calls = AtomicUsize::new(0);
        for _ in 0..3 {
            let err = map
                .get_or_compute(1, || {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Err(anyhow!("no fit at 1: {}", "cause"))
                })
                .unwrap_err();
            assert_eq!(err.to_string(), "no fit at 1: cause");
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
