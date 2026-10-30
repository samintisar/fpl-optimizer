//! The squad state and the FPL rules the planner needs (Python `fplopt.backtest.state`:
//! money, refresh, chip windows, free transfers; `backtest.rules.best_xi`). Money is int
//! tenths of £m. `apply_decision` stays in Python until the backtester moves (batch 7).

use std::collections::HashMap;
use std::fmt;

use anyhow::{Result, bail};
use fplopt_build::scoring::Rules;

pub const GOALKEEPER: i64 = 1;
pub const TRANSFER_CHIPS: [&str; 2] = ["wildcard", "freehit"];

pub fn is_transfer_chip(name: &str) -> bool {
    TRANSFER_CHIPS.contains(&name)
}

/// A decision that breaks an FPL rule (Python `InvalidDecision`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InvalidDecision(pub String);

impl fmt::Display for InvalidDecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for InvalidDecision {}

/// A held player: `price` is his latest known current price (a player who left the game
/// keeps his last price and club).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Holding {
    pub player_key: i64,
    pub element_type: i64,
    pub team_key: i64,
    pub purchase_price: i64,
    pub price: i64,
}

/// What carries over between GWs, at the deadline of `gw_index`. Holdings are kept sorted
/// by `player_key`; `chips_used` holds (chip_id, gw_index). `freehit_backup`/`freehit_bank`
/// are set only on a Free Hit GW's state (the squad and bank to restore).
#[derive(Clone, Debug, PartialEq)]
pub struct SquadState {
    pub season: i64,
    pub gw_index: i64,
    pub holdings: Vec<Holding>,
    pub bank: i64,
    pub free_transfers: i64,
    pub chips_used: Vec<(i64, i64)>,
    pub freehit_backup: Option<Vec<Holding>>,
    pub freehit_bank: Option<i64>,
}

impl SquadState {
    pub fn new(
        season: i64,
        gw_index: i64,
        mut holdings: Vec<Holding>,
        bank: i64,
        free_transfers: i64,
    ) -> Result<SquadState> {
        holdings.sort_by_key(|h| h.player_key);
        if holdings.windows(2).any(|w| w[0].player_key == w[1].player_key) {
            bail!("duplicate player_key in holdings");
        }
        Ok(SquadState {
            season,
            gw_index,
            holdings,
            bank,
            free_transfers,
            chips_used: Vec::new(),
            freehit_backup: None,
            freehit_bank: None,
        })
    }

    pub fn player_keys(&self) -> Vec<i64> {
        self.holdings.iter().map(|h| h.player_key).collect()
    }
}

/// One transfer, paired within position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Transfer {
    pub out_key: i64,
    pub in_key: i64,
}

/// What a GW's decision did (Python `TransferRecord`), as `next_state` needs it.
#[derive(Clone, Debug, PartialEq)]
pub struct TransferRecord {
    pub n_transfers: i64,
    pub chip: Option<String>,
    pub chip_id: Option<i64>,
}

/// A pool row: the deadline's buyable players.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolPlayer {
    pub player_key: i64,
    pub element_type: i64,
    pub team_key: i64,
    pub price: i64,
}

/// FPL selling price: purchase + floor((current − purchase) × fee) when the price rose,
/// else the current price. Exact for decimal fees, as Python's `Fraction(str(fee))`.
pub fn selling_price(holding: &Holding, sell_on_fee: f64) -> i64 {
    let rise = holding.price - holding.purchase_price;
    if rise <= 0 {
        return holding.price;
    }
    let (num, den) = decimal_fraction(sell_on_fee);
    holding.purchase_price + (i128::from(rise) * num).div_euclid(den) as i64
}

/// The shortest decimal of `x` as num / den (Python `Fraction(str(x))` for plain decimals).
fn decimal_fraction(x: f64) -> (i128, i128) {
    let text = format!("{x}");
    let negative = text.starts_with('-');
    let digits = text.trim_start_matches('-');
    let (int_part, frac_part) = digits.split_once('.').unwrap_or((digits, ""));
    let den = 10i128.pow(frac_part.len() as u32);
    let num: i128 = format!("{int_part}{frac_part}").parse().unwrap_or(0);
    (if negative { -num } else { num }, den)
}

/// Held players' club and price brought up to the deadline's pool; players absent from it
/// (left the game) keep their last club and price.
pub fn refresh(state: &SquadState, pool: &[PoolPlayer]) -> SquadState {
    let index: HashMap<i64, &PoolPlayer> = pool.iter().map(|p| (p.player_key, p)).collect();
    let mut out = state.clone();
    for h in &mut out.holdings {
        if let Some(p) = index.get(&h.player_key) {
            h.team_key = p.team_key;
            h.price = p.price;
        }
    }
    out
}

/// The `chip_id` that playing `name` at `state.gw_index` uses (the lowest open, unused
/// window), or InvalidDecision (unknown chip, Free Hit right after a Free Hit unless
/// allowed, no open window, already used).
pub fn chip_window(state: &SquadState, name: &str, rules: &Rules) -> Result<i64, InvalidDecision> {
    let names: HashMap<i64, &str> =
        rules.chips.iter().map(|c| (c.chip_id, c.name.as_str())).collect();
    if !names.values().any(|n| *n == name) {
        let mut known: Vec<&str> = names.values().copied().collect();
        known.sort_unstable();
        known.dedup();
        return Err(InvalidDecision(format!("unknown chip '{name}' (rules have {known:?})")));
    }
    let gw = state.gw_index;
    if name == "freehit"
        && !rules.freehit_consecutive
        && state
            .chips_used
            .iter()
            .any(|(cid, g)| *g == gw - 1 && names.get(cid) == Some(&"freehit"))
    {
        return Err(InvalidDecision(format!("Free Hit at gw_index {gw} right after a Free Hit")));
    }
    let mut open: Vec<_> =
        rules.chips.iter().filter(|c| c.name == name && c.start <= gw && gw <= c.stop).collect();
    if open.is_empty() {
        return Err(InvalidDecision(format!("no {name} window contains gw_index {gw}")));
    }
    open.sort_by_key(|c| c.chip_id);
    for window in open {
        if !state.chips_used.iter().any(|(cid, _)| *cid == window.chip_id) {
            return Ok(window.chip_id);
        }
    }
    Err(InvalidDecision(format!("{name} for gw_index {gw} already used")))
}

/// The state at the next deadline: accrue FTs (after GW1 → 1; after WC/FH per
/// `chip_week_ft`; else max(ft − n, 0) + 1; plus top-ups in (gw_index, next], capped),
/// record the chip, revert a Free Hit.
pub fn next_state(
    gw_state: &SquadState,
    record: &TransferRecord,
    rules: &Rules,
    next_gw_index: i64,
) -> Result<SquadState> {
    if next_gw_index <= gw_state.gw_index {
        bail!("next_gw_index {next_gw_index} <= gw_index {}", gw_state.gw_index);
    }
    let ft = gw_state.free_transfers;
    let chip = record.chip.as_deref();
    let mut new_ft = if gw_state.gw_index == 1 {
        1
    } else if chip.is_some_and(is_transfer_chip) {
        match rules.chip_week_ft.as_str() {
            "retain_plus_one" => ft + 1,
            "retain" => ft,
            other => bail!("chip_week_ft '{other}' not in ('retain_plus_one', 'retain')"),
        }
    } else {
        (ft - record.n_transfers).max(0) + 1
    };
    new_ft += rules
        .ft_topups
        .iter()
        .filter(|(g, _)| gw_state.gw_index < *g && *g <= next_gw_index)
        .map(|(_, a)| a)
        .sum::<i64>();
    new_ft = new_ft.min(rules.max_free_transfers);

    let mut chips_used = gw_state.chips_used.clone();
    if chip.is_some() {
        let Some(chip_id) = record.chip_id else { bail!("record has a chip but no chip_id") };
        chips_used.push((chip_id, gw_state.gw_index));
    }
    let (holdings, bank) = if chip == Some("freehit") {
        match (&gw_state.freehit_backup, gw_state.freehit_bank) {
            (Some(h), Some(b)) => (h.clone(), b),
            _ => bail!("Free Hit record but the state has no backup squad"),
        }
    } else if gw_state.freehit_backup.is_some() {
        bail!("state has a Free Hit backup but the record's chip is not freehit");
    } else {
        (gw_state.holdings.clone(), gw_state.bank)
    };
    Ok(SquadState {
        gw_index: next_gw_index,
        holdings,
        bank,
        free_transfers: new_ft,
        chips_used,
        freehit_backup: None,
        freehit_bank: None,
        ..gw_state.clone()
    })
}

/// The starters from `order` ((player_key, element_type), best first): each position's
/// `play_min` first in `order`, then the rest in order while their position is under
/// `play_max`, up to `squad_play` (Python `backtest.rules.best_xi`).
pub fn best_xi(order: &[(i64, i64)], rules: &Rules) -> Vec<(i64, i64)> {
    let mut starters: Vec<(i64, i64)> = Vec::new();
    let mut counts: HashMap<i64, i64> = HashMap::new();
    for (&et, &n) in &rules.play_min {
        let picked: Vec<_> =
            order.iter().filter(|p| p.1 == et).take(n.max(0) as usize).copied().collect();
        *counts.entry(et).or_default() += picked.len() as i64;
        starters.extend(picked);
    }
    for player in order {
        if starters.len() as i64 >= rules.squad_play {
            break;
        }
        let count = counts.get(&player.1).copied().unwrap_or(0);
        if !starters.contains(player) && count < rules.play_max.get(&player.1).copied().unwrap_or(0)
        {
            starters.push(*player);
            *counts.entry(player.1).or_default() += 1;
        }
    }
    starters
}

#[cfg(test)]
mod tests {
    use super::*;

    fn holding(price: i64, purchase: i64) -> Holding {
        Holding { player_key: 1, element_type: 2, team_key: 3, purchase_price: purchase, price }
    }

    #[test]
    fn selling_price_keeps_half_the_rise_rounded_down() {
        assert_eq!(selling_price(&holding(53, 50), 0.5), 51);
        assert_eq!(selling_price(&holding(51, 50), 0.5), 50);
        assert_eq!(selling_price(&holding(48, 50), 0.5), 48);
        assert_eq!(decimal_fraction(0.5), (5, 10));
        assert_eq!(decimal_fraction(1.0), (1, 1));
    }
}
