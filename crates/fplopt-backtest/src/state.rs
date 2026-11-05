//! Decisions applied under the FPL rules (Python `fplopt.backtest.state.apply_decision`,
//! `squad_value`, `chip_available`; the state types, money, refresh, chip windows and
//! `next_state` are the planner's, `fplopt_optimize::state`).

use std::collections::{BTreeMap, BTreeSet, HashMap};

use fplopt_build::scoring::Rules;
pub use fplopt_optimize::state::{
    GOALKEEPER, Holding, InvalidDecision, PoolPlayer, SquadState, Transfer, TransferRecord,
    chip_window, is_transfer_chip, next_state, refresh, selling_price,
};

use crate::gw_score::{Lineup, validate_lineup};
use crate::pyrepr;

/// One GW's decision: transfers (executed together), the lineup for the post-transfer
/// squad and an optional chip. `solver_status`, `mip_gap` and `diagnostics` (chip
/// quantities, `policies::CHIP_DIAGNOSTICS`) are recorded in the GW rows only and take no
/// part in equality.
#[derive(Clone, Debug, Default)]
pub struct Decision {
    pub transfers: Vec<Transfer>,
    pub lineup: Lineup,
    pub chip: Option<String>,
    pub solver_status: Option<String>,
    pub mip_gap: Option<f64>,
    pub diagnostics: Option<BTreeMap<String, Option<f64>>>,
}

impl PartialEq for Decision {
    fn eq(&self, other: &Decision) -> bool {
        self.transfers == other.transfers && self.lineup == other.lineup && self.chip == other.chip
    }
}

/// What `apply_decision` did: `free_used` counts the transfers that cost nothing (all of
/// them at GW1 and under WC/FH), so `n_transfers == free_used + hits`; `chip_id` the window
/// the chip came from.
#[derive(Clone, Debug, PartialEq)]
pub struct Applied {
    pub n_transfers: i64,
    pub free_used: i64,
    pub hits: i64,
    pub hit_points: i64,
    pub chip: Option<String>,
    pub chip_id: Option<i64>,
}

impl Applied {
    /// What `next_state` needs.
    pub fn record(&self) -> TransferRecord {
        TransferRecord {
            n_transfers: self.n_transfers,
            chip: self.chip.clone(),
            chip_id: self.chip_id,
        }
    }
}

/// Sum of the holdings' selling prices (what the squad would raise), bank excluded.
pub fn squad_value(state: &SquadState, sell_on_fee: f64) -> i64 {
    state.holdings.iter().map(|h| selling_price(h, sell_on_fee)).sum()
}

/// Whether chip `name` can be played at `state.gw_index`.
pub fn chip_available(state: &SquadState, name: &str, rules: &Rules) -> bool {
    chip_window(state, name, rules).is_ok()
}

fn invalid<T>(msg: String) -> Result<T, InvalidDecision> {
    Err(InvalidDecision(msg))
}

fn duplicates(items: &[i64]) -> Vec<i64> {
    let mut seen = BTreeMap::new();
    for &k in items {
        *seen.entry(k).or_insert(0) += 1;
    }
    seen.into_iter().filter(|(_, n)| *n > 1).map(|(k, _)| k).collect()
}

/// Validate and execute a decision; the squad that plays this GW and what was done.
///
/// The state is first refreshed from the pool. Checks, in order: (1) chip available; (2)
/// outs held, ins in the pool and not held, no player twice; (3) position counts equal
/// `squad_select`; (4) no club over `team_limit` that a bought player belongs to; (5) bank ≥
/// 0 after selling at selling price and buying at pool price; (6) the lineup is valid for the
/// new squad. Hits: none at GW1 or under WC/FH; otherwise every transfer beyond the free
/// transfers costs `hit_cost`. A Free Hit keeps the pre-transfer holdings and bank.
pub fn apply_decision(
    state: &SquadState,
    decision: &Decision,
    pool: &[PoolPlayer],
    rules: &Rules,
) -> anyhow::Result<(SquadState, Applied)> {
    if state.freehit_backup.is_some() {
        anyhow::bail!("state is a Free Hit gameweek state; call next_state first");
    }
    Ok(apply(state, decision, pool, rules)?)
}

fn apply(
    state: &SquadState,
    decision: &Decision,
    pool: &[PoolPlayer],
    rules: &Rules,
) -> Result<(SquadState, Applied), InvalidDecision> {
    let players: HashMap<i64, &PoolPlayer> = pool.iter().map(|p| (p.player_key, p)).collect();
    let state = refresh(state, pool);
    let chip = decision.chip.as_deref();
    let chip_id = match chip {
        Some(name) => Some(chip_window(&state, name, rules)?),
        None => None,
    };
    let held: HashMap<i64, &Holding> = state.holdings.iter().map(|h| (h.player_key, h)).collect();
    let outs: Vec<i64> = decision.transfers.iter().map(|t| t.out_key).collect();
    let ins: Vec<i64> = decision.transfers.iter().map(|t| t.in_key).collect();
    if !duplicates(&outs).is_empty() {
        return invalid(format!("player sold twice: {}", pyrepr::list(&duplicates(&outs))));
    }
    if !duplicates(&ins).is_empty() {
        return invalid(format!("player bought twice: {}", pyrepr::list(&duplicates(&ins))));
    }
    for key in &outs {
        if !held.contains_key(key) {
            return invalid(format!("cannot sell {key}: not in the squad"));
        }
    }
    for key in &ins {
        if held.contains_key(key) {
            return invalid(format!("cannot buy {key}: already in the squad"));
        }
        if !players.contains_key(key) {
            return invalid(format!("cannot buy {key}: not in the pool"));
        }
    }

    let sold: BTreeSet<i64> = outs.iter().copied().collect();
    let bought: Vec<Holding> = ins
        .iter()
        .map(|k| {
            let p = players[k];
            Holding {
                player_key: *k,
                element_type: p.element_type,
                team_key: p.team_key,
                purchase_price: p.price,
                price: p.price,
            }
        })
        .collect();
    let mut holdings: Vec<Holding> =
        state.holdings.iter().filter(|h| !sold.contains(&h.player_key)).cloned().collect();
    holdings.extend(bought.iter().cloned());
    holdings.sort_by_key(|h| h.player_key);

    let counts = pyrepr::counter(holdings.iter().map(|h| h.element_type));
    let wanted: Vec<(i64, i64)> =
        rules.squad_select.iter().filter(|(_, n)| **n != 0).map(|(e, n)| (*e, *n)).collect();
    let as_map = |c: &[(i64, i64)]| c.iter().copied().collect::<BTreeMap<i64, i64>>();
    if as_map(&counts) != as_map(&wanted) {
        return invalid(format!(
            "position counts {} != squad_select {}",
            pyrepr::dict(&as_map(&counts)),
            pyrepr::counter_dict(&wanted)
        ));
    }
    let mut clubs: HashMap<i64, i64> = HashMap::new();
    for h in &holdings {
        *clubs.entry(h.team_key).or_default() += 1;
    }
    let bought_clubs: BTreeSet<i64> = bought.iter().map(|h| h.team_key).collect();
    for team in bought_clubs {
        if clubs[&team] > rules.team_limit {
            return invalid(format!(
                "club {team} would have {} players (limit {})",
                clubs[&team], rules.team_limit
            ));
        }
    }
    let bank = state.bank
        + outs.iter().map(|k| selling_price(held[k], rules.sell_on_fee)).sum::<i64>()
        - ins.iter().map(|k| players[k].price).sum::<i64>();
    if bank < 0 {
        return invalid(format!("over budget: bank would be {bank} (tenths of £m)"));
    }
    let positions: HashMap<i64, i64> =
        holdings.iter().map(|h| (h.player_key, h.element_type)).collect();
    if let Err(e) = validate_lineup(&decision.lineup, &positions, rules) {
        return invalid(format!("invalid lineup: {e}"));
    }

    let n = decision.transfers.len() as i64;
    let free = if state.gw_index == 1 || chip.is_some_and(is_transfer_chip) {
        n
    } else {
        n.min(state.free_transfers.max(0))
    };
    let hits = n - free;
    let freehit = chip == Some("freehit");
    let gw_state = SquadState {
        freehit_backup: freehit.then(|| state.holdings.clone()),
        freehit_bank: freehit.then_some(state.bank),
        holdings,
        bank,
        ..state.clone()
    };
    let applied = Applied {
        n_transfers: n,
        free_used: free,
        hits,
        hit_points: hits * rules.hit_cost,
        chip: decision.chip.clone(),
        chip_id,
    };
    Ok((gw_state, applied))
}
