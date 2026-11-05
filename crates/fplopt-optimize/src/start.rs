//! Start states and deadline lookups for `optimize plan` (Python
//! `fplopt.backtest.start_states.template_state` and what it needs; `cli.plan_gw_index`;
//! `backtest.simulator.season_schedule`'s start deadline), and `random_state` (numpy's RNG,
//! `fplopt_core::nprng`).
//!
//! `template_state`: the most-owned valid squad. Ownership = the newest player snapshot's
//! `selected_by_percent` of the season, else each player's newest visible
//! `player_gw_ownership.selected` of the season. Players are added in descending ownership
//! (ties by player_key) when their position has a slot, their club is under the cap, and the
//! money left can still fill the remaining slots with the cheapest eligible players. The
//! state: purchase price = price, bank = budget − cost, 1 free transfer (0 at gw_index 1), no
//! chips used.
//!
//! `random_state`: a seeded random valid squad. Slots are visited in random order and each is
//! filled with a feasible player drawn with probability ∝ price, feasible meaning also that
//! the dearest fill of the remaining slots leaves at most `max_bank` (£2m). A dead end is
//! redrawn, up to `ATTEMPTS` times. `np.random.default_rng(seed)`, draw for draw.

use std::collections::{BTreeMap, HashMap};
use std::fmt;

use anyhow::{Result, bail};
use fplopt_build::scoring::Rules;
use fplopt_core::nprng::Generator;
use fplopt_core::numpy::np_sum;
use fplopt_features::baseline::{player_pool, pool_coverage};
use fplopt_store::{AsOfView, DataStore, Micros, Table};

use crate::entry::GwIndex;
use crate::state::{Holding, PoolPlayer, SquadState};

/// 2200-01-01T00:00:00Z: a view that sees everything (Python `simulator.FAR_FUTURE`).
pub const FAR_FUTURE: Micros = 7_258_118_400_000_000;

/// No start state can be generated at this deadline (Python `StartStateError`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StartStateError(pub String);

impl fmt::Display for StartStateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for StartStateError {}

/// A deadline as pandas prints a UTC Timestamp (`2024-10-26 10:00:00+00:00`).
pub fn fmt_timestamp(t: Micros) -> String {
    fplopt_features::leakcheck::fmt_micros(t).replacen('T', " ", 1)
}

fn gameweek_rows(view: &AsOfView, season: i64) -> Result<(Vec<i64>, Vec<Option<i64>>)> {
    let table = view.table("gameweek", Some(&["season", "gw", "gw_index"]))?;
    let seasons = table.opt_i64("season")?;
    let gws = table.opt_i64("gw")?;
    let index = table.opt_i64("gw_index")?;
    let mut gw = Vec::new();
    let mut gw_index = Vec::new();
    for i in 0..table.height() {
        if seasons[i] == Some(season)
            && let Some(g) = gws[i]
        {
            gw.push(g);
            gw_index.push(index[i]);
        }
    }
    Ok((gw, gw_index))
}

/// (season, gw, gw_index) of the GW whose deadline is the view's deadline.
pub fn target_gameweek(view: &AsOfView) -> Result<(i64, i64, i64)> {
    let (season, gw) = view.gameweek_for_deadline()?;
    let (gws, index) = gameweek_rows(view, season)?;
    match gws.iter().position(|&g| g == gw).and_then(|i| index[i]) {
        Some(gw_index) => Ok((season, gw, gw_index)),
        None => bail!("no gw_index for season {season} GW{gw}"),
    }
}

/// StartStateError if a club with a fixture in the horizon has no pool players.
pub fn check_coverage(view: &AsOfView) -> Result<()> {
    let coverage = pool_coverage(view)?;
    let horizon = coverage.opt_i64("n_fixtures_horizon")?;
    let players = coverage.opt_i64("n_pool_players")?;
    let teams = coverage.opt_i64("team_key")?;
    let gaps: Vec<String> = (0..coverage.height())
        .filter(|&i| horizon[i].unwrap_or(0) > 0 && players[i] == Some(0))
        .map(|i| teams[i].map_or("nan".to_owned(), |t| t.to_string()))
        .collect();
    if !gaps.is_empty() {
        return Err(StartStateError(format!(
            "pool coverage gap at {}: club(s) [{}] have fixtures in the horizon but no pool \
             players",
            fmt_timestamp(view.deadline()),
            gaps.join(", ")
        ))
        .into());
    }
    Ok(())
}

/// player_key -> ownership visible at the deadline (snapshot percent, else the newest
/// visible `player_gw_ownership.selected` of the season; NaN where null), or None if none is
/// visible for the season.
pub fn ownership(view: &AsOfView) -> Result<Option<HashMap<i64, f64>>> {
    let (season, _, _) = target_gameweek(view)?;
    let snaps = view.latest(
        "player_snapshot",
        &["player_key"],
        Some(&["snapshot_at", "season", "selected_by_percent"]),
    )?;
    let keys = snaps.opt_i64("player_key")?;
    let at = snaps.opt_i64("snapshot_at")?;
    let seasons = snaps.opt_i64("season")?;
    let share = snaps.opt_f64("selected_by_percent")?;
    let rows: Vec<usize> = (0..snaps.height())
        .filter(|&i| seasons[i] == Some(season) && share[i].is_some_and(|x| !x.is_nan()))
        .collect();
    if !rows.is_empty() {
        let newest = rows.iter().filter_map(|&i| at[i]).max();
        return Ok(Some(
            rows.iter()
                .filter(|&&i| at[i] == newest)
                .filter_map(|&i| keys[i].map(|k| (k, share[i].unwrap())))
                .collect(),
        ));
    }
    let table =
        view.table("player_gw_ownership", Some(&["player_key", "season", "gw", "selected"]))?;
    let keys = table.opt_i64("player_key")?;
    let seasons = table.opt_i64("season")?;
    let gws = table.opt_i64("gw")?;
    let selected = table.opt_f64("selected")?;
    let mut rows: Vec<usize> =
        (0..table.height()).filter(|&i| seasons[i] == Some(season)).collect();
    if rows.is_empty() {
        return Ok(None);
    }
    let (gw_list, index_list) = gameweek_rows(view, season)?;
    let mut index: HashMap<i64, Option<i64>> = HashMap::new();
    for (g, i) in gw_list.into_iter().zip(index_list) {
        index.insert(g, i);
    }
    let gw_index = |i: usize| gws[i].and_then(|g| index.get(&g).copied().flatten());
    // sort_values(["player_key", "gw_index"], kind="mergesort"): stable, nulls last.
    let last = |x: Option<i64>| (x.is_none(), x.unwrap_or(0));
    rows.sort_by_key(|&i| (last(keys[i]), last(gw_index(i))));
    let mut out = HashMap::new();
    for i in rows {
        // drop_duplicates(keep="last") after the sort: the newest row per player wins.
        if let Some(k) = keys[i] {
            out.insert(k, selected[i].unwrap_or(f64::NAN));
        }
    }
    Ok(Some(out))
}

/// The pool table's rows as `PoolPlayer`s.
pub fn pool_players(pool: &Table) -> Result<Vec<PoolPlayer>> {
    let (keys, et, team, price) = (
        pool.i64("player_key")?,
        pool.i64("element_type")?,
        pool.i64("team_key")?,
        pool.i64("price")?,
    );
    Ok((0..pool.height())
        .map(|i| PoolPlayer {
            player_key: keys[i],
            element_type: et[i],
            team_key: team[i],
            price: price[i],
        })
        .collect())
}

/// A squad under construction from the pool (players sorted by player_key).
pub struct Builder<'r> {
    rules: &'r Rules,
    players: Vec<PoolPlayer>,
    chosen: Vec<bool>,
    slots: BTreeMap<i64, i64>,
    clubs: HashMap<i64, i64>,
    spent: i64,
    /// Per position: pool positions sorted by (price, key), cheapest first.
    by_price: BTreeMap<i64, Vec<usize>>,
}

impl<'r> Builder<'r> {
    pub fn new(mut players: Vec<PoolPlayer>, rules: &'r Rules) -> Builder<'r> {
        players.sort_by_key(|p| p.player_key);
        let slots: BTreeMap<i64, i64> =
            rules.squad_select.iter().filter(|(_, n)| **n != 0).map(|(e, n)| (*e, *n)).collect();
        let by_price = slots
            .keys()
            .map(|&et| {
                let mut idx: Vec<usize> =
                    (0..players.len()).filter(|&i| players[i].element_type == et).collect();
                idx.sort_by_key(|&i| (players[i].price, players[i].player_key));
                (et, idx)
            })
            .collect();
        Builder {
            rules,
            chosen: vec![false; players.len()],
            players,
            slots,
            clubs: HashMap::new(),
            spent: 0,
            by_price,
        }
    }

    pub fn players(&self) -> &[PoolPlayer] {
        &self.players
    }

    pub fn full(&self) -> bool {
        self.slots.values().sum::<i64>() == 0
    }

    fn slot(&self, et: i64) -> i64 {
        self.slots.get(&et).copied().unwrap_or(0)
    }

    fn club(&self, team: i64) -> i64 {
        self.clubs.get(&team).copied().unwrap_or(0)
    }

    /// Cheapest (or, with `dearest`, the most expensive) cost of filling `slots` with
    /// players not `blocked` (None if impossible).
    fn fill_cost(
        &self,
        slots: &BTreeMap<i64, i64>,
        blocked: &[bool],
        dearest: bool,
    ) -> Option<i64> {
        let mut total = 0;
        for (et, &n) in slots {
            if n <= 0 {
                continue;
            }
            let idx = self.by_price.get(et).map_or(&[][..], Vec::as_slice);
            let open = |i: &&usize| !blocked[**i];
            let picked: Vec<usize> = if dearest {
                idx.iter().rev().filter(open).take(n as usize).copied().collect()
            } else {
                idx.iter().filter(open).take(n as usize).copied().collect()
            };
            if (picked.len() as i64) < n {
                return None;
            }
            total += picked.iter().map(|&i| self.players[i].price).sum::<i64>();
        }
        Some(total)
    }

    /// Player i fits: his position has a slot, his club is under the cap, and the remaining
    /// budget can fill the remaining slots with the cheapest eligible players (and, with
    /// `max_bank`, the dearest eligible fill leaves at most `max_bank`).
    pub fn feasible(&self, i: usize, max_bank: Option<i64>) -> bool {
        let p = self.players[i];
        let limit = self.rules.team_limit;
        if self.chosen[i] || self.slot(p.element_type) <= 0 || self.club(p.team_key) >= limit {
            return false;
        }
        let left = self.rules.budget - self.spent - p.price;
        if left < 0 {
            return false;
        }
        let mut slots = self.slots.clone();
        *slots.entry(p.element_type).or_default() -= 1;
        let mut blocked = self.chosen.clone();
        blocked[i] = true;
        // Clubs at the cap once i is in: none of their players can fill a slot.
        let mut full: Vec<i64> =
            self.clubs.iter().filter(|(_, n)| **n >= limit).map(|(t, _)| *t).collect();
        if self.club(p.team_key) + 1 >= limit {
            full.push(p.team_key);
        }
        if !full.is_empty() {
            for (b, q) in blocked.iter_mut().zip(&self.players) {
                *b |= full.contains(&q.team_key);
            }
        }
        match self.fill_cost(&slots, &blocked, false) {
            Some(cost) if cost <= left => {}
            _ => return false,
        }
        match max_bank {
            None => true,
            Some(max_bank) => {
                self.fill_cost(&slots, &blocked, true).is_some_and(|c| left - c <= max_bank)
            }
        }
    }

    pub fn add(&mut self, i: usize) {
        let p = self.players[i];
        self.chosen[i] = true;
        *self.slots.entry(p.element_type).or_default() -= 1;
        *self.clubs.entry(p.team_key).or_default() += 1;
        self.spent += p.price;
    }

    pub fn state(&self, season: i64, gw_index: i64) -> Result<SquadState> {
        let holdings = (0..self.players.len())
            .filter(|&i| self.chosen[i])
            .map(|i| {
                let p = self.players[i];
                Holding {
                    player_key: p.player_key,
                    element_type: p.element_type,
                    team_key: p.team_key,
                    purchase_price: p.price,
                    price: p.price,
                }
            })
            .collect();
        let ft = if gw_index > 1 { 1 } else { 0 };
        SquadState::new(season, gw_index, holdings, self.rules.budget - self.spent, ft)
    }

    fn no_squad(&self, deadline: Micros) -> anyhow::Error {
        let open: BTreeMap<i64, i64> =
            self.slots.iter().filter(|(_, n)| **n > 0).map(|(e, n)| (*e, *n)).collect();
        StartStateError(format!(
            "the pool at {} cannot form a valid squad (open slots {}, spent {})",
            fmt_timestamp(deadline),
            crate::entry::py_dict(&open),
            self.spent
        ))
        .into()
    }
}

/// The most-owned valid squad among `players` given `share` (player_key -> ownership; missing
/// or NaN = 0): descending share, ties by player_key, each added when feasible.
pub fn template_squad<'r>(
    players: Vec<PoolPlayer>,
    share: &HashMap<i64, f64>,
    rules: &'r Rules,
) -> Builder<'r> {
    let mut builder = Builder::new(players, rules);
    let shares: Vec<f64> = builder
        .players()
        .iter()
        .map(|p| share.get(&p.player_key).copied().filter(|x| !x.is_nan()).unwrap_or(0.0))
        .collect();
    // np.lexsort((keys, -share)): players are sorted by key, so a stable sort on -share.
    let mut order: Vec<usize> = (0..shares.len()).collect();
    order.sort_by(|&a, &b| {
        (-shares[a]).partial_cmp(&-shares[b]).unwrap_or(std::cmp::Ordering::Equal)
    });
    for i in order {
        if builder.full() {
            break;
        }
        if builder.feasible(i, None) {
            builder.add(i);
        }
    }
    builder
}

/// The most-owned valid squad at the view's deadline (module docs).
pub fn template_state(view: &AsOfView, rules: &Rules) -> Result<SquadState> {
    check_coverage(view)?;
    let players = pool_players(&player_pool(view)?)?;
    let Some(owned) = ownership(view)? else {
        return Err(StartStateError(format!(
            "no ownership visible at {}: no template squad",
            fmt_timestamp(view.deadline())
        ))
        .into());
    };
    let builder = template_squad(players, &owned, rules);
    if !builder.full() {
        return Err(builder.no_squad(view.deadline()));
    }
    let (season, _, gw_index) = target_gameweek(view)?;
    builder.state(season, gw_index)
}

/// random_state redraws after a dead end.
pub const ATTEMPTS: usize = 20;
/// random_state: at most £2m left in the bank.
pub const MAX_BANK: i64 = 20;

/// A seeded random valid squad at the view's deadline, players drawn ∝ price, leaving at
/// most `max_bank` in the bank (module docs).
pub fn random_state(
    view: &AsOfView,
    rules: &Rules,
    seed: u64,
    max_bank: Option<i64>,
) -> Result<SquadState> {
    check_coverage(view)?;
    let players = pool_players(&player_pool(view)?)?;
    let mut rng = Generator::new(u128::from(seed));
    let mut builder = Builder::new(players.clone(), rules);
    for _ in 0..ATTEMPTS {
        builder = Builder::new(players.clone(), rules);
        let slots: Vec<i64> = builder
            .slots
            .iter()
            .flat_map(|(&et, &n)| std::iter::repeat_n(et, n as usize))
            .collect();
        for k in rng.permutation(slots.len()) {
            let et = slots[k];
            let candidates: Vec<usize> = (0..builder.players.len())
                .filter(|&i| builder.players[i].element_type == et && builder.feasible(i, max_bank))
                .collect();
            if candidates.is_empty() {
                break;
            }
            let mut weights: Vec<f64> = candidates
                .iter()
                .map(|&i| builder.players[i].price as f64)
                .map(|w| if w > 0.0 { w } else { 0.0 })
                .collect();
            if np_sum(&weights) <= 0.0 {
                weights = vec![1.0; candidates.len()];
            }
            let total = np_sum(&weights);
            let p: Vec<f64> = weights.iter().map(|w| w / total).collect();
            builder.add(candidates[rng.choice_weighted(&p)]);
        }
        if builder.full() {
            let (season, _, gw_index) = target_gameweek(view)?;
            return builder.state(season, gw_index);
        }
    }
    Err(builder.no_squad(view.deadline()))
}

/// The gw_index of GW `gw` of `season` (the first visible row), or None if the data has no
/// such GW (Python `cli.plan_gw_index`).
pub fn plan_gw_index(store: &DataStore, season: i64, gw: i64) -> Result<Option<i64>> {
    let (gws, index) = gameweek_rows(&store.as_of(FAR_FUTURE), season)?;
    Ok(gws.iter().position(|&g| g == gw).and_then(|i| index[i]))
}

/// The season's FPL GW -> gw_index (the last row of a GW wins, as Python's `dict(zip(...))`).
pub fn season_gw_index(store: &DataStore, season: i64) -> Result<GwIndex> {
    let (gws, index) = gameweek_rows(&store.as_of(FAR_FUTURE), season)?;
    let mut out = GwIndex::new();
    for (g, i) in gws.into_iter().zip(index) {
        if let Some(i) = i {
            out.insert(g, i);
        }
    }
    Ok(out)
}

/// The deadline of `gw_index` of `season` as `season_schedule` finds it: of every deadline
/// ever listed for that gw_index, the one whose own view lists exactly one row for it, with
/// that deadline.
pub fn start_deadline(store: &DataStore, season: i64, gw_index: i64) -> Result<Micros> {
    let rows = |view: AsOfView| -> Result<Vec<Option<Micros>>> {
        let table = view.table("gameweek", Some(&["season", "gw_index", "deadline_time"]))?;
        let seasons = table.opt_i64("season")?;
        let index = table.opt_i64("gw_index")?;
        let deadlines = table.opt_i64("deadline_time")?;
        Ok((0..table.height())
            .filter(|&i| seasons[i] == Some(season) && index[i] == Some(gw_index))
            .map(|i| deadlines[i])
            .collect())
    };
    let mut candidates: Vec<Micros> =
        rows(store.as_of(FAR_FUTURE))?.into_iter().flatten().collect();
    candidates.sort_unstable();
    candidates.dedup();
    let mut found = Vec::new();
    for deadline in candidates {
        let start = rows(store.as_of(deadline))?;
        if start.len() == 1 && start[0] == Some(deadline) {
            found.push(deadline);
        }
    }
    if found.len() != 1 {
        bail!(
            "{}: {} gameweeks with gw_index {gw_index} visible at their own deadline",
            crate::entry::season_label(season),
            found.len()
        );
    }
    Ok(found[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use fplopt_build::scoring::backtest_rules;
    use std::path::Path;

    fn rules() -> Rules {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config/scoring");
        backtest_rules(2024, &dir).unwrap()
    }

    /// 20 clubs x (3 GK, 8 DEF, 8 MID, 5 FWD); price rises with the club number.
    fn league() -> Vec<PoolPlayer> {
        let mut out = Vec::new();
        let mut key = 0;
        for team in 1..=20 {
            for (et, n) in [(1, 3), (2, 8), (3, 8), (4, 5)] {
                for j in 0..n {
                    key += 1;
                    out.push(PoolPlayer {
                        player_key: key,
                        element_type: et,
                        team_key: team,
                        price: 40 + et * 5 + j + team,
                    });
                }
            }
        }
        out.reverse(); // the builder sorts by key
        out
    }

    fn valid(state: &SquadState, rules: &Rules) {
        assert_eq!(state.holdings.len() as i64, rules.squad_size);
        let mut counts: BTreeMap<i64, i64> = BTreeMap::new();
        let mut clubs: HashMap<i64, i64> = HashMap::new();
        for h in &state.holdings {
            *counts.entry(h.element_type).or_default() += 1;
            *clubs.entry(h.team_key).or_default() += 1;
            assert_eq!(h.purchase_price, h.price);
        }
        assert_eq!(counts, rules.squad_select);
        assert!(clubs.values().all(|&n| n <= rules.team_limit));
        let cost: i64 = state.holdings.iter().map(|h| h.price).sum();
        assert_eq!(state.bank, rules.budget - cost);
        assert!(state.bank >= 0);
    }

    #[test]
    fn template_takes_the_most_owned_feasible_players() {
        let rules = rules();
        // Everyone at club 20 (the dearest) is owned most, in key order: the club cap stops it
        // at 3, then the next most-owned clubs fill the rest.
        let share: HashMap<i64, f64> =
            league().iter().map(|p| (p.player_key, p.team_key as f64 * 1.5)).collect();
        let builder = template_squad(league(), &share, &rules);
        assert!(builder.full());
        let state = builder.state(2024, 10).unwrap();
        valid(&state, &rules);
        assert_eq!(state.free_transfers, 1);
        let at_20 = state.holdings.iter().filter(|h| h.team_key == 20).count() as i64;
        assert_eq!(at_20, rules.team_limit);
        // Ties (one share per club) go to the lowest keys: club 20's first players.
        let first_key = league().iter().filter(|p| p.team_key == 20).map(|p| p.player_key).min();
        assert!(state.holdings.iter().any(|h| Some(h.player_key) == first_key));
        assert_eq!(builder.state(2024, 1).unwrap().free_transfers, 0);
    }

    #[test]
    fn a_budget_below_the_cheapest_fill_forms_no_squad() {
        let mut rules = rules();
        let share: HashMap<i64, f64> =
            league().iter().map(|p| (p.player_key, p.price as f64)).collect();
        let cheapest: i64 = {
            let builder = Builder::new(league(), &rules);
            let slots = builder.slots.clone();
            builder.fill_cost(&slots, &vec![false; builder.players.len()], false).unwrap()
        };
        rules.budget = cheapest - 1;
        let builder = template_squad(league(), &share, &rules);
        assert!(!builder.full());
        let err = builder.no_squad(0).to_string();
        assert!(err.starts_with("the pool at 1970-01-01 00:00:00+00:00 cannot form a valid squad (open slots {1: 2, 2: 5, 3: 5, 4: 3}, spent 0)"), "{err}");
    }

    #[test]
    fn dearest_fill_bounds_the_bank() {
        let mut rules = rules();
        rules.budget = 100_000;
        let builder = Builder::new(league(), &rules);
        let i = builder.players().iter().position(|p| p.element_type == 1).unwrap();
        assert!(builder.feasible(i, None));
        assert!(builder.feasible(i, Some(rules.budget)));
        assert!(!builder.feasible(i, Some(0)));
    }

    #[test]
    fn far_future_is_2200() {
        assert_eq!(fmt_timestamp(FAR_FUTURE), "2200-01-01 00:00:00+00:00");
    }
}
