//! The planner's input at one deadline (Python `fplopt.optimize.problem`; PLAN §7): the
//! refreshed squad, the horizon GWs of the xP frame, the pruned candidates with their xP
//! per horizon GW, and the minutes-based bench weights. Prices are fixed over the horizon;
//! an owned player's first sale earns his selling price, any other sale his buy price.

use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::{Result, bail};
use fplopt_build::scoring::Rules;
use fplopt_build::table::Table;

use crate::minutes::{horizon_minutes, minutes_bench_weights};
use crate::params::OptimizerParams;
use crate::prune::{Candidates, prune};
use crate::state::{PoolPlayer, SquadState, refresh, selling_price};
use crate::util::Kahan;

pub const XP_COLUMNS: [&str; 5] = ["player_key", "gw", "gw_index", "horizon", "xp"];

/// An xP frame (one row per player-GW-fixture): the model columns the planner reads.
/// `p_play`, `p_play_gw`, `e_minutes` are None when the model has no minutes (baselines).
#[derive(Clone, Debug, Default)]
pub struct XpFrame {
    pub player_key: Vec<i64>,
    pub gw: Vec<i64>,
    pub gw_index: Vec<i64>,
    pub horizon: Vec<i64>,
    pub xp: Vec<Option<f64>>,
    pub p_play: Option<Vec<Option<f64>>>,
    pub p_play_gw: Option<Vec<Option<f64>>>,
    pub e_minutes: Option<Vec<Option<f64>>>,
}

impl XpFrame {
    pub fn len(&self) -> usize {
        self.player_key.len()
    }

    pub fn is_empty(&self) -> bool {
        self.player_key.is_empty()
    }

    /// From a table with `XP_COLUMNS` (and the optional minutes columns).
    pub fn from_table(table: &Table) -> Result<XpFrame> {
        let missing: Vec<_> = XP_COLUMNS.iter().filter(|c| !table.has(c)).collect();
        if !missing.is_empty() {
            bail!("the xP frame is missing columns {missing:?}");
        }
        let ints = |name: &str| -> Result<Vec<i64>> {
            let values = table.opt_i64(name)?;
            if values.iter().any(Option::is_none) {
                bail!("the xP frame has nulls in {name}");
            }
            Ok(values.into_iter().map(Option::unwrap).collect())
        };
        let floats = |name: &str| -> Result<Option<Vec<Option<f64>>>> {
            if !table.has(name) {
                return Ok(None);
            }
            Ok(Some(table.opt_f64(name)?))
        };
        Ok(XpFrame {
            player_key: ints("player_key")?,
            gw: ints("gw")?,
            gw_index: ints("gw_index")?,
            horizon: ints("horizon")?,
            xp: table.opt_f64("xp")?,
            p_play: floats("p_play")?,
            p_play_gw: floats("p_play_gw")?,
            e_minutes: floats("e_minutes")?,
        })
    }
}

/// A candidate: `price` the buy price, `sell_price` what his first sale earns; `xp[t]` his
/// xP in horizon GW t.
#[derive(Clone, Debug, PartialEq)]
pub struct Player {
    pub player_key: i64,
    pub element_type: i64,
    pub team_key: i64,
    pub price: i64,
    pub sell_price: i64,
    pub owned: bool,
    pub buyable: bool,
    pub xp: Vec<f64>,
}

/// A horizon GW: `horizon` is its offset from the target GW (its decay exponent).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HorizonGw {
    pub gw: i64,
    pub gw_index: i64,
    pub horizon: i64,
}

/// Everything the MILP needs (Python `PlanInput`). `players` sorted by key; `bench_weights`
/// per horizon GW from the minutes model, or None for the fixed `params.bench_weights`.
#[derive(Clone, Debug)]
pub struct PlanInput {
    pub state: SquadState,
    pub rules: Rules,
    pub players: Vec<Player>,
    pub gws: Vec<HorizonGw>,
    pub bank: i64,
    pub free_transfers: i64,
    pub gw1: bool,
    pub n_pool: usize,
    pub bench_weights: Option<Vec<Vec<f64>>>,
}

impl PlanInput {
    /// Horizon GW t's bench weights without a chip.
    pub fn gw_bench_weights<'a>(&'a self, t: usize, params: &'a OptimizerParams) -> &'a [f64] {
        match &self.bench_weights {
            None => &params.bench_weights,
            Some(w) => &w[t],
        }
    }

    pub fn owned_keys(&self) -> Vec<i64> {
        self.players.iter().filter(|p| p.owned).map(|p| p.player_key).collect()
    }

    /// Build the input from a state, the deadline's pool, an xP frame and the rules.
    pub fn from_context(
        state: &SquadState,
        pool: &[PoolPlayer],
        xp: &XpFrame,
        rules: &Rules,
        params: &OptimizerParams,
    ) -> Result<PlanInput> {
        if state.freehit_backup.is_some() {
            bail!("state is a Free Hit gameweek state; call next_state first");
        }
        let mut seen = HashSet::new();
        if pool.iter().any(|p| !seen.insert(p.player_key)) {
            bail!("pool has duplicate player_key");
        }
        let state = refresh(state, pool);
        let gws = horizon_gws(xp, &state, params)?;
        let horizons: Vec<i64> = gws.iter().map(|g| g.horizon).collect();
        let column: HashMap<i64, usize> =
            horizons.iter().enumerate().map(|(i, h)| (*h, i)).collect();

        let mut pool = pool.to_vec();
        pool.sort_by_key(|p| p.player_key);
        let owned: HashMap<i64, &crate::state::Holding> =
            state.holdings.iter().map(|h| (h.player_key, h)).collect();
        // xP per player and horizon GW: missing or NaN counts 0, rows of a player-GW add up.
        let mut matrix: BTreeMap<i64, Vec<Kahan>> = BTreeMap::new();
        for i in 0..xp.len() {
            if let Some(&t) = column.get(&xp.horizon[i]) {
                let x = xp.xp[i].filter(|v| !v.is_nan()).unwrap_or(0.0);
                matrix.entry(xp.player_key[i]).or_insert_with(|| vec![Kahan::default(); gws.len()])
                    [t]
                    .add(x);
            }
        }

        // Every pool player plus departed owned players.
        let in_pool: HashSet<i64> = pool.iter().map(|p| p.player_key).collect();
        let mut frame: Vec<PoolPlayer> = pool.clone();
        frame.extend(state.holdings.iter().filter(|h| !in_pool.contains(&h.player_key)).map(|h| {
            PoolPlayer {
                player_key: h.player_key,
                element_type: h.element_type,
                team_key: h.team_key,
                price: h.price,
            }
        }));
        frame.sort_by_key(|p| p.player_key);
        let keys: Vec<i64> = frame.iter().map(|p| p.player_key).collect();
        let xp_rows: Vec<Vec<f64>> = keys
            .iter()
            .map(|k| match matrix.get(k) {
                Some(row) => row.iter().map(Kahan::value).collect(),
                None => vec![0.0; gws.len()],
            })
            .collect();
        let buyable: Vec<bool> = keys.iter().map(|k| in_pool.contains(k)).collect();
        let owned_mask: Vec<bool> = keys.iter().map(|k| owned.contains_key(k)).collect();
        let minutes =
            if params.min_minutes > 0.0 { horizon_minutes(xp, &horizons, &keys)? } else { None };
        let element_type: Vec<i64> = frame.iter().map(|p| p.element_type).collect();
        let team_key: Vec<i64> = frame.iter().map(|p| p.team_key).collect();
        let price: Vec<i64> = frame.iter().map(|p| p.price).collect();
        let kept = prune(
            &Candidates {
                keys: &keys,
                element_type: &element_type,
                team_key: &team_key,
                price: &price,
                xp: &xp_rows,
                owned: &owned_mask,
                buyable: &buyable,
                minutes: minutes.as_deref(),
            },
            params.decay,
            &horizons,
            rules,
            params.prune_n.as_ref(),
            params.prune_dominated,
            // The floor is for a full horizon: pro rata when it is shorter (season's end).
            params.min_minutes * horizons.len() as f64 / params.horizon as f64,
        )?;
        let kept: HashSet<i64> = kept.into_iter().collect();

        let mut players = Vec::new();
        for (row, p) in frame.iter().enumerate() {
            if !kept.contains(&p.player_key) {
                continue;
            }
            let holding = owned.get(&p.player_key);
            players.push(Player {
                player_key: p.player_key,
                element_type: p.element_type,
                team_key: p.team_key,
                price: p.price,
                sell_price: holding.map_or(p.price, |h| selling_price(h, rules.sell_on_fee)),
                owned: holding.is_some(),
                buyable: buyable[row],
                xp: xp_rows[row].clone(),
            });
        }
        // GW1 (unlimited transfers) rebuilds the squad freely: the fixed weights apply.
        let bench_weights = if params.bench_from_minutes && state.gw_index != 1 {
            let squad: Vec<(i64, i64)> =
                state.holdings.iter().map(|h| (h.player_key, h.element_type)).collect();
            minutes_bench_weights(&squad, xp, &horizons, rules, &params.bench_weights)
        } else {
            None
        };
        Ok(PlanInput {
            bank: state.bank,
            free_transfers: state.free_transfers,
            gw1: state.gw_index == 1,
            n_pool: pool.len(),
            state,
            rules: rules.clone(),
            players,
            gws,
            bench_weights,
        })
    }
}

/// The xP frame's GWs with horizon < params.horizon, by horizon; the first must be the
/// state's GW (horizon 0).
fn horizon_gws(
    xp: &XpFrame,
    state: &SquadState,
    params: &OptimizerParams,
) -> Result<Vec<HorizonGw>> {
    let mut seen = HashSet::new();
    let mut gws = Vec::new();
    for i in 0..xp.len() {
        if xp.horizon[i] < params.horizon {
            let g = HorizonGw { gw: xp.gw[i], gw_index: xp.gw_index[i], horizon: xp.horizon[i] };
            if seen.insert(g) {
                gws.push(g);
            }
        }
    }
    gws.sort_by_key(|g| g.horizon);
    if gws.windows(2).any(|w| w[0].horizon == w[1].horizon) {
        bail!("xp frame maps one horizon to several GWs");
    }
    if gws.first().is_none_or(|g| g.horizon != 0 || g.gw_index != state.gw_index) {
        bail!(
            "xp frame must start at the state's GW (gw_index {}, horizon 0); got {:?}",
            state.gw_index,
            gws.first()
        );
    }
    Ok(gws)
}
