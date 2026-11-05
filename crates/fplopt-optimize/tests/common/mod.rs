//! Hand-made pools, states and xP frames for the optimizer tests (Python
//! `tests/optimize_fixtures.py`), an exact brute-force scorer of a squad's best lineup for one
//! GW, a seeded generator (numpy's stream can't be reproduced: instances have the same shape,
//! not the same numbers), and a test-only `apply_decision` with Python's rules-engine checks
//! (`fplopt.backtest.state.apply_decision`, `gw_score.validate_lineup`) to replay plans.

#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::Path;
use std::sync::OnceLock;

use fplopt_build::scoring::{Rules, load_rules};
use fplopt_optimize::plans::{GwPlan, Plan};
use fplopt_optimize::state::{
    Holding, InvalidDecision, PoolPlayer, SquadState, Transfer, TransferRecord, chip_window,
    is_transfer_chip, next_state, refresh, selling_price,
};
use fplopt_optimize::{ChipScenario, OptimizerParams, PlanInput, XpFrame, solve_plan};

pub const SEASON: i64 = 2023;

/// The 2026-27 rules (Python `load_rules("2026-27")`).
pub fn rules() -> Rules {
    static RULES: OnceLock<Rules> = OnceLock::new();
    RULES
        .get_or_init(|| {
            let dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../config/scoring"));
            load_rules("2026-27", None, dir).expect("load the 2026-27 rules")
        })
        .clone()
}

pub fn base_price(et: i64) -> i64 {
    match et {
        1 => 45,
        2 => 50,
        3 => 70,
        4 => 75,
        _ => panic!("element_type {et}"),
    }
}

// --- a seeded generator -------------------------------------------------------------------

/// SplitMix64: a small deterministic generator for the tests' random instances.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut rng = Rng(seed.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ 0x1234_5678_9ABC_DEF0);
        rng.next_u64();
        rng
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn random(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.random()
    }

    /// Uniform integer in [lo, hi).
    pub fn integers(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next_u64() % (hi - lo) as u64) as i64
    }

    /// Standard normal (Box-Muller).
    pub fn normal(&mut self) -> f64 {
        let u1 = 1.0 - self.random();
        let u2 = self.random();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Gamma(shape 2, scale): the sum of two exponentials.
    pub fn gamma2(&mut self, scale: f64) -> f64 {
        -scale * ((1.0 - self.random()).ln() + (1.0 - self.random()).ln())
    }

    /// A random permutation of 0..n (Fisher-Yates).
    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut out: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = self.integers(0, i as i64 + 1) as usize;
            out.swap(i, j);
        }
        out
    }
}

pub fn round_to(x: f64, decimals: i32) -> f64 {
    let f = 10f64.powi(decimals);
    (x * f).round_ties_even() / f
}

// --- pools, states, frames ----------------------------------------------------------------

/// club * 100 + element_type * 10 + i.
pub fn key(club: i64, et: i64, i: i64) -> i64 {
    club * 100 + et * 10 + i
}

pub fn et_of(k: i64) -> i64 {
    k / 10 % 10
}

pub fn club_of(k: i64) -> i64 {
    k / 100
}

/// Pool from (player_key, element_type, team_key, price) rows, sorted by key.
pub fn make_pool(rows: impl IntoIterator<Item = (i64, i64, i64, i64)>) -> Vec<PoolPlayer> {
    let mut pool: Vec<PoolPlayer> = rows
        .into_iter()
        .map(|(player_key, element_type, team_key, price)| PoolPlayer {
            player_key,
            element_type,
            team_key,
            price,
        })
        .collect();
    pool.sort_by_key(|p| p.player_key);
    pool
}

/// Every club gets per_pos[et] players per position; prices base_price(et) (+ a seeded
/// spread of 0..30 when seed is given).
pub fn grid_pool(
    clubs: impl IntoIterator<Item = i64>,
    per_pos: &[(i64, i64)],
    seed: Option<u64>,
) -> Vec<PoolPlayer> {
    let mut rng = Rng::new(seed.unwrap_or(0));
    let mut per_pos = per_pos.to_vec();
    per_pos.sort_unstable();
    let mut rows = Vec::new();
    for club in clubs {
        for &(et, n) in &per_pos {
            for i in 0..n {
                let spread = if seed.is_some() { rng.integers(0, 31) } else { 0 };
                rows.push((key(club, et, i), et, club, base_price(et) + spread));
            }
        }
    }
    make_pool(rows)
}

pub const GRID: [(i64, i64); 4] = [(1, 2), (2, 4), (3, 4), (4, 3)];
pub const GRID5: [(i64, i64); 4] = [(1, 2), (2, 5), (3, 5), (4, 3)];

pub fn pool_player(pool: &[PoolPlayer], k: i64) -> PoolPlayer {
    *pool.iter().find(|p| p.player_key == k).unwrap_or_else(|| panic!("{k} not in the pool"))
}

pub fn pool_keys(pool: &[PoolPlayer]) -> Vec<i64> {
    pool.iter().map(|p| p.player_key).collect()
}

pub struct StateOpts {
    pub gw_index: i64,
    pub ft: i64,
    pub bank: Option<i64>,
    pub purchase: HashMap<i64, i64>,
    pub extra: Vec<Holding>,
}

impl Default for StateOpts {
    fn default() -> Self {
        StateOpts { gw_index: 5, ft: 1, bank: None, purchase: HashMap::new(), extra: Vec::new() }
    }
}

/// A state holding `keys` (bought at pool price unless `purchase` says otherwise) plus
/// `extra` holdings. Bank: what's left of the budget unless given.
pub fn make_state(pool: &[PoolPlayer], keys: &[i64], opts: StateOpts) -> SquadState {
    let mut holdings: Vec<Holding> = keys
        .iter()
        .map(|&k| {
            let p = pool_player(pool, k);
            Holding {
                player_key: k,
                element_type: p.element_type,
                team_key: p.team_key,
                purchase_price: opts.purchase.get(&k).copied().unwrap_or(p.price),
                price: p.price,
            }
        })
        .collect();
    holdings.extend(opts.extra);
    let bank = opts
        .bank
        .unwrap_or_else(|| rules().budget - holdings.iter().map(|h| h.purchase_price).sum::<i64>());
    SquadState::new(SEASON, opts.gw_index, holdings, bank, opts.ft).expect("a valid state")
}

pub fn state_with(pool: &[PoolPlayer], keys: &[i64], gw_index: i64, ft: i64) -> SquadState {
    make_state(pool, keys, StateOpts { gw_index, ft, ..StateOpts::default() })
}

/// xP frame from player_key -> xP per horizon GW (horizon 0 = gw_index; gw = gw_index).
pub fn xp_frame(values: &BTreeMap<i64, Vec<f64>>, gw_index: i64) -> XpFrame {
    let mut frame = XpFrame::default();
    for (&k, xs) in values {
        for (h, &x) in xs.iter().enumerate() {
            frame.player_key.push(k);
            frame.gw.push(gw_index + h as i64);
            frame.gw_index.push(gw_index + h as i64);
            frame.horizon.push(h as i64);
            frame.xp.push(Some(x));
        }
    }
    frame
}

/// Set a player's xP per horizon (rows in horizon order).
pub fn set_xp(frame: &mut XpFrame, k: i64, values: &[f64]) {
    for i in 0..frame.len() {
        if frame.player_key[i] == k {
            frame.xp[i] = Some(values[frame.horizon[i] as usize]);
        }
    }
}

pub fn xp_value(frame: &XpFrame, k: i64, h: i64) -> f64 {
    (0..frame.len())
        .find(|&i| frame.player_key[i] == k && frame.horizon[i] == h)
        .and_then(|i| frame.xp[i])
        .unwrap_or_else(|| panic!("no xP for {k} at horizon {h}"))
}

/// player_key -> xP at horizon `h` (the frame's last row per player wins, as a dict(zip)).
pub fn xp_at(frame: &XpFrame, h: i64) -> HashMap<i64, f64> {
    (0..frame.len())
        .filter(|&i| frame.horizon[i] == h)
        .map(|i| (frame.player_key[i], frame.xp[i].unwrap_or(f64::NAN)))
        .collect()
}

/// The frame's rows where `keep(row)` holds.
pub fn filter_rows(frame: &XpFrame, keep: impl Fn(usize) -> bool) -> XpFrame {
    let rows: Vec<usize> = (0..frame.len()).filter(|&i| keep(i)).collect();
    take_rows(frame, &rows)
}

/// The frame's rows in the order `rows` gives.
pub fn take_rows(frame: &XpFrame, rows: &[usize]) -> XpFrame {
    let pick_f =
        |c: &Option<Vec<Option<f64>>>| c.as_ref().map(|v| rows.iter().map(|&i| v[i]).collect());
    XpFrame {
        player_key: rows.iter().map(|&i| frame.player_key[i]).collect(),
        gw: rows.iter().map(|&i| frame.gw[i]).collect(),
        gw_index: rows.iter().map(|&i| frame.gw_index[i]).collect(),
        horizon: rows.iter().map(|&i| frame.horizon[i]).collect(),
        xp: rows.iter().map(|&i| frame.xp[i]).collect(),
        p_play: pick_f(&frame.p_play),
        p_play_gw: pick_f(&frame.p_play_gw),
        e_minutes: pick_f(&frame.e_minutes),
    }
}

/// Seeded xP per player and GW, loosely increasing with price (2 decimals).
pub fn random_xp(pool: &[PoolPlayer], n_gws: usize, seed: u64, gw_index: i64) -> XpFrame {
    let mut rng = Rng::new(seed.wrapping_add(1000));
    let mut values = BTreeMap::new();
    for p in pool {
        let base = (p.price - 35) as f64 / 12.0;
        let xs = (0..n_gws).map(|_| round_to(rng.gamma2(base / 2.0 + 0.2), 2)).collect();
        values.insert(p.player_key, xs);
    }
    xp_frame(&values, gw_index)
}

/// Seeded xP: a per-player rate rising with price, times a per-club fixture factor per GW
/// (so xP is correlated across GWs, as real projections are).
pub fn correlated_xp(pool: &[PoolPlayer], n_gws: usize, seed: u64, gw_index: i64) -> XpFrame {
    let mut rng = Rng::new(seed.wrapping_add(2000));
    let clubs: BTreeSet<i64> = pool.iter().map(|p| p.team_key).collect();
    let factor: HashMap<i64, Vec<f64>> =
        clubs.iter().map(|&c| (c, (0..n_gws).map(|_| rng.uniform(0.7, 1.3)).collect())).collect();
    let mut values = BTreeMap::new();
    for p in pool {
        let rate = ((p.price - 40) as f64 / 10.0 + rng.normal()).max(0.0);
        values.insert(
            p.player_key,
            factor[&p.team_key].iter().map(|f| round_to(rate * f, 2)).collect(),
        );
    }
    xp_frame(&values, gw_index)
}

/// The cheapest 2/5/5/3 squad with at most team_limit per club (greedy by price).
pub fn first_valid_squad(pool: &[PoolPlayer]) -> Vec<i64> {
    let rules = rules();
    let mut chosen = Vec::new();
    let mut clubs: HashMap<i64, i64> = HashMap::new();
    for (&et, &n) in &rules.squad_select {
        let mut rows: Vec<&PoolPlayer> = pool.iter().filter(|p| p.element_type == et).collect();
        rows.sort_by_key(|p| (p.price, p.player_key));
        let mut picked = 0;
        for p in rows {
            let c = clubs.entry(p.team_key).or_default();
            if picked < n && *c < rules.team_limit {
                chosen.push(p.player_key);
                *c += 1;
                picked += 1;
            }
        }
    }
    assert_eq!(chosen.len() as i64, rules.squad_size);
    assert!(chosen.iter().map(|&k| pool_player(pool, k).price).sum::<i64>() <= 1000);
    chosen
}

/// Exact max over formations of Σ XI xP + captain xP + Σ_k w_k · bench_k xP (bench GK in
/// slot 0, outfield bench by xP descending in slots 1..3).
pub fn best_lineup_value(
    squad: &[i64],
    xp: &HashMap<i64, f64>,
    et: &HashMap<i64, i64>,
    bench_weights: &[f64],
) -> f64 {
    let rules = rules();
    let by_pos = |p: i64| -> Vec<f64> {
        let mut v: Vec<f64> = squad.iter().filter(|k| et[k] == p).map(|k| xp[k]).collect();
        v.sort_by(|a, b| b.total_cmp(a));
        v
    };
    let (gk, def, mid, fwd) = (by_pos(1), by_pos(2), by_pos(3), by_pos(4));
    let range = |p: i64| rules.play_min[&p] as usize..=rules.play_max[&p] as usize;
    let mut best = f64::NEG_INFINITY;
    for d in range(2) {
        for m in range(3) {
            for f in range(4) {
                if (1 + d + m + f) as i64 != rules.squad_play {
                    continue;
                }
                let starters: Vec<f64> = gk[..1]
                    .iter()
                    .chain(&def[..d])
                    .chain(&mid[..m])
                    .chain(&fwd[..f])
                    .copied()
                    .collect();
                let mut bench_out: Vec<f64> =
                    def[d..].iter().chain(&mid[m..]).chain(&fwd[f..]).copied().collect();
                bench_out.sort_by(|a, b| b.total_cmp(a));
                let mut value: f64 = starters.iter().sum::<f64>()
                    + starters.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                    + bench_weights[0] * gk[1];
                for (w, x) in bench_weights[1..].iter().zip(&bench_out) {
                    value += w * x;
                }
                best = best.max(value);
            }
        }
    }
    best
}

// --- planner helpers ----------------------------------------------------------------------

pub fn problem(
    state: &SquadState,
    pool: &[PoolPlayer],
    xp: &XpFrame,
    params: &OptimizerParams,
    rules: &Rules,
) -> PlanInput {
    PlanInput::from_context(state, pool, xp, rules, params).expect("a plan input")
}

/// The no-chip plan (Python `solve_plan(problem, params)`).
pub fn solve(problem: &PlanInput, params: &OptimizerParams) -> Plan {
    solve_plan(problem, params, None, &[], &ChipScenario::none()).expect("a plan")
}

pub fn scenario(problem: &PlanInput, chips: &[(usize, &str)]) -> ChipScenario {
    let map: BTreeMap<usize, String> = chips.iter().map(|(t, n)| (*t, n.to_string())).collect();
    fplopt_optimize::chips::make_scenario(problem, &map).expect("a valid scenario")
}

pub fn player(problem: &PlanInput, k: i64) -> &fplopt_optimize::Player {
    problem
        .players
        .iter()
        .find(|p| p.player_key == k)
        .unwrap_or_else(|| panic!("{k} not a candidate"))
}

pub fn transfers_in(g: &GwPlan) -> Vec<i64> {
    g.transfers.iter().map(|t| t.in_key).collect()
}

pub fn transfers_out(g: &GwPlan) -> Vec<i64> {
    g.transfers.iter().map(|t| t.out_key).collect()
}

pub fn squad_of(g: &GwPlan) -> BTreeSet<i64> {
    g.starters.iter().chain(&g.bench).copied().collect()
}

pub fn approx(a: f64, b: f64, abs: f64) -> bool {
    (a - b).abs() <= abs.max(1e-6 * b.abs())
}

#[track_caller]
pub fn assert_approx(a: f64, b: f64) {
    assert!(approx(a, b, 1e-12), "{a} != {b} (approx)");
}

#[track_caller]
pub fn assert_approx_vec(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len(), "{a:?} != {b:?}");
    for (x, y) in a.iter().zip(b) {
        assert!(approx(*x, *y, 1e-12), "{a:?} != {b:?} (approx)");
    }
}

// --- the rules engine (test-only apply_decision) ------------------------------------------

/// One GW's decision (Python `Decision`).
#[derive(Clone, Debug)]
pub struct Decision {
    pub transfers: Vec<Transfer>,
    pub starters: Vec<i64>,
    pub bench: Vec<i64>,
    pub captain: i64,
    pub vice: i64,
    pub chip: Option<String>,
}

/// The decision a plan's GW makes (Python `GwPlan.decision()`).
pub fn decision(g: &GwPlan) -> Decision {
    Decision {
        transfers: g.transfers.clone(),
        starters: g.starters.clone(),
        bench: g.bench.clone(),
        captain: g.captain,
        vice: g.vice,
        chip: g.chip.clone(),
    }
}

/// What `apply_decision` did (Python `TransferRecord`).
#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    pub n_transfers: i64,
    pub free_used: i64,
    pub hits: i64,
    pub hit_points: i64,
    pub chip: Option<String>,
    pub chip_id: Option<i64>,
}

impl Record {
    pub fn transfer_record(&self) -> TransferRecord {
        TransferRecord {
            n_transfers: self.n_transfers,
            chip: self.chip.clone(),
            chip_id: self.chip_id,
        }
    }
}

fn invalid(m: String) -> InvalidDecision {
    InvalidDecision(m)
}

/// Python `apply_decision`: validate and execute a decision; the squad that plays this GW
/// and a record. Checks in Python's order: chip available; outs held, ins in the pool and
/// not held, nobody twice; position counts; club cap (a pre-existing excess is tolerated
/// if nobody from that club is bought); bank ≥ 0 at selling prices; lineup valid.
pub fn apply_decision(
    state: &SquadState,
    d: &Decision,
    pool: &[PoolPlayer],
    rules: &Rules,
) -> Result<(SquadState, Record), InvalidDecision> {
    assert!(state.freehit_backup.is_none(), "state is a Free Hit gameweek state");
    let players: HashMap<i64, PoolPlayer> = pool.iter().map(|p| (p.player_key, *p)).collect();
    let state = refresh(state, pool);
    let chip_id = match &d.chip {
        Some(c) => Some(chip_window(&state, c, rules)?),
        None => None,
    };
    let held: HashMap<i64, &Holding> = state.holdings.iter().map(|h| (h.player_key, h)).collect();
    let outs: Vec<i64> = d.transfers.iter().map(|t| t.out_key).collect();
    let ins: Vec<i64> = d.transfers.iter().map(|t| t.in_key).collect();
    if outs.iter().collect::<HashSet<_>>().len() != outs.len() {
        return Err(invalid(format!("player sold twice: {outs:?}")));
    }
    if ins.iter().collect::<HashSet<_>>().len() != ins.len() {
        return Err(invalid(format!("player bought twice: {ins:?}")));
    }
    for k in &outs {
        if !held.contains_key(k) {
            return Err(invalid(format!("cannot sell {k}: not in the squad")));
        }
    }
    for k in &ins {
        if held.contains_key(k) {
            return Err(invalid(format!("cannot buy {k}: already in the squad")));
        }
        if !players.contains_key(k) {
            return Err(invalid(format!("cannot buy {k}: not in the pool")));
        }
    }
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
        state.holdings.iter().filter(|h| !outs.contains(&h.player_key)).cloned().collect();
    holdings.extend(bought.iter().cloned());
    holdings.sort_by_key(|h| h.player_key);

    let mut counts: BTreeMap<i64, i64> = BTreeMap::new();
    for h in &holdings {
        *counts.entry(h.element_type).or_default() += 1;
    }
    let wanted: BTreeMap<i64, i64> =
        rules.squad_select.iter().filter(|(_, n)| **n != 0).map(|(e, n)| (*e, *n)).collect();
    if counts != wanted {
        return Err(invalid(format!("position counts {counts:?} != squad_select {wanted:?}")));
    }
    let mut clubs: HashMap<i64, i64> = HashMap::new();
    for h in &holdings {
        *clubs.entry(h.team_key).or_default() += 1;
    }
    let bought_clubs: BTreeSet<i64> = bought.iter().map(|h| h.team_key).collect();
    for team in bought_clubs {
        if clubs[&team] > rules.team_limit {
            return Err(invalid(format!(
                "club {team} would have {} players (limit {})",
                clubs[&team], rules.team_limit
            )));
        }
    }
    let bank = state.bank
        + outs.iter().map(|k| selling_price(held[k], rules.sell_on_fee)).sum::<i64>()
        - ins.iter().map(|k| players[k].price).sum::<i64>();
    if bank < 0 {
        return Err(invalid(format!("over budget: bank would be {bank} (tenths of £m)")));
    }
    let positions: HashMap<i64, i64> =
        holdings.iter().map(|h| (h.player_key, h.element_type)).collect();
    validate_lineup(d, &positions, rules).map_err(|m| invalid(format!("invalid lineup: {m}")))?;

    let n = d.transfers.len() as i64;
    let free = if state.gw_index == 1 || d.chip.as_deref().is_some_and(is_transfer_chip) {
        n
    } else {
        n.min(state.free_transfers.max(0))
    };
    let hits = n - free;
    let freehit = d.chip.as_deref() == Some("freehit");
    let gw_state = SquadState {
        holdings,
        bank,
        freehit_backup: if freehit { Some(state.holdings.clone()) } else { None },
        freehit_bank: if freehit { Some(state.bank) } else { None },
        ..state.clone()
    };
    let record = Record {
        n_transfers: n,
        free_used: free,
        hits,
        hit_points: hits * rules.hit_cost,
        chip: d.chip.clone(),
        chip_id,
    };
    Ok((gw_state, record))
}

/// Python `gw_score.validate_lineup`.
fn validate_lineup(
    d: &Decision,
    positions: &HashMap<i64, i64>,
    rules: &Rules,
) -> Result<(), String> {
    if d.starters.len() as i64 != rules.squad_play {
        return Err(format!("{} starters, need {}", d.starters.len(), rules.squad_play));
    }
    if d.bench.len() as i64 != rules.squad_size - rules.squad_play {
        return Err(format!("{} bench players", d.bench.len()));
    }
    let squad: Vec<i64> = d.starters.iter().chain(&d.bench).copied().collect();
    if squad.iter().collect::<HashSet<_>>().len() != squad.len() {
        return Err(format!("duplicate players in lineup: {squad:?}"));
    }
    let unknown: Vec<i64> = squad.iter().filter(|k| !positions.contains_key(k)).copied().collect();
    if !unknown.is_empty() {
        return Err(format!("no position for players {unknown:?}"));
    }
    let mut counts: BTreeMap<i64, i64> = BTreeMap::new();
    for k in &squad {
        *counts.entry(positions[k]).or_default() += 1;
    }
    if counts != rules.squad_select {
        return Err(format!("squad positions {counts:?} != {:?}", rules.squad_select));
    }
    let mut formation: BTreeMap<i64, i64> = BTreeMap::new();
    for k in &d.starters {
        *formation.entry(positions[k]).or_default() += 1;
    }
    for (et, lo) in &rules.play_min {
        let n = formation.get(et).copied().unwrap_or(0);
        if n < *lo || n > rules.play_max[et] {
            return Err(format!("invalid formation {formation:?}"));
        }
    }
    if positions[&d.bench[0]] != 1 {
        return Err(format!("bench[0] ({}) must be the goalkeeper", d.bench[0]));
    }
    if d.captain == d.vice {
        return Err("captain and vice must differ".to_string());
    }
    for (role, k) in [("captain", d.captain), ("vice", d.vice)] {
        if !d.starters.contains(&k) {
            return Err(format!("{role} {k} is not a starter"));
        }
    }
    Ok(())
}

/// The first GW's decision through the rules engine (panics if invalid).
pub fn apply_first(
    state: &SquadState,
    plan: &Plan,
    pool: &[PoolPlayer],
    rules: &Rules,
) -> SquadState {
    apply_decision(state, &decision(plan.first()), pool, rules).expect("a valid decision").0
}

/// Every GW of the plan through apply_decision/next_state; the plan's ft (before), hits,
/// transfers, chip and bank (after) must match the rules engine. Returns the state after
/// the horizon.
pub fn replay(state: &SquadState, plan: &Plan, pool: &[PoolPlayer], rules: &Rules) -> SquadState {
    let mut state = state.clone();
    for (i, g) in plan.gws.iter().enumerate() {
        assert!(
            state.free_transfers == g.ft || state.gw_index == 1,
            "GW {i}: ft {} vs {g:?}",
            state.free_transfers
        );
        let (gw_state, record) = apply_decision(&state, &decision(g), pool, rules)
            .unwrap_or_else(|e| panic!("GW {i}: {e}"));
        assert_eq!(
            (record.hits, record.n_transfers, record.chip.clone()),
            (g.hits, g.n_transfers() as i64, g.chip.clone()),
            "GW {i}"
        );
        assert_eq!(gw_state.bank, g.bank, "GW {i}: bank");
        let next = plan.gws.get(i + 1).map_or(g.gw_index + 1, |n| n.gw_index);
        state = next_state(&gw_state, &record.transfer_record(), rules, next).expect("next state");
    }
    state
}
