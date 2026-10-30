//! The transfer-planning MILP for one chip scenario (Python `fplopt.optimize.model`; PLAN
//! §7). Variables, constraints, objective and hooks are documented in the Python module
//! docstring; this port builds the identical HiGHS model: the same columns in the order
//! PuLP exports them (creation order), the same rows in the same order, the same objective
//! coefficients computed with the same float operations (repeated variables merged in
//! first-seen order), handed over in bulk as Python's `_BulkHiGHS` does, with the same
//! HiGHS options. Same input, same HiGHS version and `threads=1` → the same plan.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::time::Instant;

use anyhow::{Result, bail};
use fplopt_highs::{Highs, ModelStatus};

use crate::chips::{ChipScenario, terminal_value};
use crate::params::OptimizerParams;
use crate::plans::{GwPlan, Plan};
use crate::problem::PlanInput;
use crate::state::{GOALKEEPER, Transfer, is_transfer_chip};

/// (outs, ins) of a GW's transfers.
pub type TransferSet = (BTreeSet<i64>, BTreeSet<i64>);
const ON: f64 = 0.5;
/// Relative margin added to LP-relaxation bounds (HiGHS's feasibility tolerances).
pub const BOUND_MARGIN: f64 = 1e-6;

/// Why `solve_plan` returned no plan.
#[derive(Debug)]
pub enum SolveError {
    /// HiGHS found no feasible plan.
    Infeasible(String),
    /// HiGHS stopped at a limit before finding any plan; `bound` bounds the scenario's best
    /// total objective (inf if unknown).
    LimitReached { status: String, bound: f64 },
    /// Anything else (a bad scenario, fixed transfers naming non-candidates, HiGHS errors).
    Other(anyhow::Error),
}

impl std::fmt::Display for SolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolveError::Infeasible(m) => write!(f, "{m}"),
            SolveError::LimitReached { status, .. } => {
                write!(f, "no plan found before the solver stopped (status {status})")
            }
            SolveError::Other(e) => write!(f, "{e:#}"),
        }
    }
}

impl std::error::Error for SolveError {}

impl From<anyhow::Error> for SolveError {
    fn from(e: anyhow::Error) -> Self {
        SolveError::Other(e)
    }
}

// --- a PuLP-shaped LP ----------------------------------------------------------------------

/// A linear expression: terms merged per variable in first-seen order, plus a constant.
#[derive(Clone, Debug, Default)]
struct Expr {
    terms: Vec<(usize, f64)>,
    constant: f64,
}

impl Expr {
    fn var(v: usize) -> Expr {
        Expr { terms: vec![(v, 1.0)], constant: 0.0 }
    }

    fn constant(c: f64) -> Expr {
        Expr { terms: Vec::new(), constant: c }
    }

    fn sum(vars: impl IntoIterator<Item = usize>) -> Expr {
        let mut e = Expr::default();
        for v in vars {
            e.add(v, 1.0);
        }
        e
    }

    fn add(&mut self, v: usize, c: f64) {
        match self.terms.iter_mut().find(|(u, _)| *u == v) {
            Some(term) => term.1 += c,
            None => self.terms.push((v, c)),
        }
    }

    fn plus(mut self, other: &Expr, sign: f64) -> Expr {
        for &(v, c) in &other.terms {
            self.add(v, c * sign);
        }
        self.constant += other.constant * sign;
        self
    }

    fn value(&self, x: &[f64]) -> f64 {
        let mut total = self.constant;
        for &(v, c) in &self.terms {
            total += c * x[v];
        }
        total
    }
}

struct Row {
    terms: Vec<(usize, f64)>,
    lower: f64,
    upper: f64,
}

#[derive(Default)]
struct Lp {
    lower: Vec<f64>,
    upper: Vec<f64>,
    integer: Vec<bool>,
    rows: Vec<Row>,
}

impl Lp {
    fn var(&mut self, lower: f64, upper: f64, integer: bool) -> usize {
        self.lower.push(lower);
        self.upper.push(upper);
        self.integer.push(integer);
        self.lower.len() - 1
    }

    fn binary(&mut self) -> usize {
        self.var(0.0, 1.0, true)
    }

    /// `lhs ≤ rhs` (`le`), `lhs ≥ rhs` (`ge`) or `lhs = rhs` (`eq`), as PuLP stores it:
    /// the row is lhs − rhs with the constant moved to the bound.
    fn le(&mut self, lhs: Expr, rhs: &Expr) {
        let e = lhs.plus(rhs, -1.0);
        self.rows.push(Row { terms: e.terms, lower: f64::NEG_INFINITY, upper: -e.constant });
    }

    fn ge(&mut self, lhs: Expr, rhs: &Expr) {
        let e = lhs.plus(rhs, -1.0);
        self.rows.push(Row { terms: e.terms, lower: -e.constant, upper: f64::INFINITY });
    }

    fn eq(&mut self, lhs: Expr, rhs: &Expr) {
        let e = lhs.plus(rhs, -1.0);
        self.rows.push(Row { terms: e.terms, lower: -e.constant, upper: -e.constant });
    }
}

// --- the model -----------------------------------------------------------------------------

type Key = (i64, usize);

/// FT count of a GW: a constant, or Σ s · fts[s] over binaries.
#[derive(Clone)]
enum Ft {
    Const(i64),
    States(Vec<(usize, i64)>),
}

impl Ft {
    fn expr(&self) -> Expr {
        match self {
            Ft::Const(c) => Expr::constant(*c as f64),
            Ft::States(states) => {
                let mut e = Expr::default();
                for &(v, s) in states {
                    e.add(v, s as f64);
                }
                e
            }
        }
    }
}

pub(crate) struct Model<'a> {
    problem: &'a PlanInput,
    params: &'a OptimizerParams,
    lp: Lp,
    t_len: usize,
    chips: BTreeMap<usize, String>,
    fh: BTreeSet<usize>,
    rebuilt: BTreeSet<usize>,
    squad: HashMap<Key, usize>,
    buy: BTreeMap<Key, usize>,
    buy_order: Vec<(Key, usize)>,
    sell_first: HashMap<Key, usize>,
    sell_later: HashMap<Key, usize>,
    lineup: HashMap<Key, usize>,
    captain: HashMap<Key, usize>,
    bench: Vec<((i64, usize, usize), usize)>,
    fh_squad: HashMap<Key, usize>,
    fh_buy: BTreeMap<Key, usize>,
    fh_buy_order: Vec<(Key, usize)>,
    fh_sell_first: HashMap<Key, usize>,
    fh_sell_later: HashMap<Key, usize>,
    bank: Vec<usize>,
    fh_bank: BTreeMap<usize, usize>,
    ft: Vec<Ft>,
    hits: Vec<Option<usize>>,
}

impl<'a> Model<'a> {
    pub(crate) fn new(
        problem: &'a PlanInput,
        params: &'a OptimizerParams,
        scenario: &ChipScenario,
    ) -> Result<Model<'a>> {
        let rules = &problem.rules;
        if rules.squad_size - rules.squad_play != params.bench_weights.len() as i64 {
            bail!(
                "bench_weights has {} slots, the rules' bench {}",
                params.bench_weights.len(),
                rules.squad_size - rules.squad_play
            );
        }
        let t_len = problem.gws.len();
        let chips: BTreeMap<usize, String> =
            scenario.chips.iter().map(|(t, n)| (*t, n.clone())).collect();
        let known = ["wildcard", "freehit", "bboost", "3xc"];
        if chips.iter().any(|(t, c)| *t >= t_len || !known.contains(&c.as_str())) {
            bail!("invalid chip scenario {}", scenario.label);
        }
        if !chips.is_empty()
            && !["retain_plus_one", "retain"].contains(&rules.chip_week_ft.as_str())
        {
            bail!("chip_week_ft '{}' not in ('retain_plus_one', 'retain')", rules.chip_week_ft);
        }
        let fh: BTreeSet<usize> =
            chips.iter().filter(|(_, c)| *c == "freehit").map(|(t, _)| *t).collect();
        let wildcard = chips.iter().filter(|(_, c)| *c == "wildcard").map(|(t, _)| *t).min();
        let mut rebuilt = fh.clone();
        if let Some(w) = wildcard {
            rebuilt.extend(w..t_len);
        }
        let mut model = Model {
            problem,
            params,
            lp: Lp::default(),
            t_len,
            chips,
            fh,
            rebuilt,
            squad: HashMap::new(),
            buy: BTreeMap::new(),
            buy_order: Vec::new(),
            sell_first: HashMap::new(),
            sell_later: HashMap::new(),
            lineup: HashMap::new(),
            captain: HashMap::new(),
            bench: Vec::new(),
            fh_squad: HashMap::new(),
            fh_buy: BTreeMap::new(),
            fh_buy_order: Vec::new(),
            fh_sell_first: HashMap::new(),
            fh_sell_later: HashMap::new(),
            bank: Vec::new(),
            fh_bank: BTreeMap::new(),
            ft: Vec::new(),
            hits: Vec::new(),
        };
        model.variables();
        model.squad_constraints();
        model.lineup_constraints();
        model.money_and_transfers();
        Ok(model)
    }

    fn variables(&mut self) {
        let n_slots = self.params.bench_weights.len();
        let lp = &mut self.lp;
        for p in &self.problem.players {
            let k = p.player_key;
            // A later sale needs an earlier buy (owned: an earlier first sale, then a buy).
            let later_from = if p.owned { 2 } else { 1 };
            for t in 0..self.t_len {
                self.squad.insert((k, t), lp.binary());
                if self.fh.contains(&t) {
                    self.fh_squad.insert((k, t), lp.binary());
                    if p.buyable {
                        let v = lp.binary();
                        self.fh_buy.insert((k, t), v);
                        self.fh_buy_order.push(((k, t), v));
                    }
                    if p.owned {
                        self.fh_sell_first.insert((k, t), lp.binary());
                    }
                    if p.buyable && t >= later_from {
                        self.fh_sell_later.insert((k, t), lp.binary());
                    }
                } else {
                    if p.buyable {
                        let v = lp.binary();
                        self.buy.insert((k, t), v);
                        self.buy_order.push(((k, t), v));
                    }
                    if p.owned {
                        self.sell_first.insert((k, t), lp.binary());
                    }
                    if p.buyable && t >= later_from {
                        self.sell_later.insert((k, t), lp.binary());
                    }
                }
                self.lineup.insert((k, t), lp.binary());
                self.captain.insert((k, t), lp.binary());
                let slots: Vec<usize> =
                    if p.element_type == GOALKEEPER { vec![0] } else { (1..n_slots).collect() };
                for s in slots {
                    self.bench.push(((k, t, s), lp.binary()));
                }
            }
        }
        self.bank = (0..self.t_len).map(|_| lp.var(0.0, f64::INFINITY, false)).collect();
        for &t in &self.fh {
            self.fh_bank.insert(t, lp.var(0.0, f64::INFINITY, false));
        }
    }

    fn sells(&self, k: i64, t: usize) -> Expr {
        Expr::sum(
            self.sell_first.get(&(k, t)).into_iter().chain(self.sell_later.get(&(k, t))).copied(),
        )
    }

    fn fh_sells(&self, k: i64, t: usize) -> Expr {
        Expr::sum(
            self.fh_sell_first
                .get(&(k, t))
                .into_iter()
                .chain(self.fh_sell_later.get(&(k, t)))
                .copied(),
        )
    }

    /// Σ sell_first[k,s] for s < t.
    fn first_sales_before(&self, k: i64, t: usize) -> Expr {
        Expr::sum((0..t).filter_map(|s| self.sell_first.get(&(k, s)).copied()))
    }

    fn n_transfers(&self, t: usize) -> Expr {
        Expr::sum(
            self.problem.players.iter().filter_map(|p| self.buy.get(&(p.player_key, t)).copied()),
        )
    }

    fn selected(&self, k: i64, t: usize) -> usize {
        if self.fh.contains(&t) { self.fh_squad[&(k, t)] } else { self.squad[&(k, t)] }
    }

    /// GW t's bench weights: 1 under Bench Boost; the fixed weights where a Wildcard or Free
    /// Hit rebuilds the squad; else the problem's.
    fn bench_weights(&self, t: usize) -> Vec<f64> {
        if self.chips.get(&t).map(String::as_str) == Some("bboost") {
            return vec![1.0; self.params.bench_weights.len()];
        }
        if self.rebuilt.contains(&t) {
            return self.params.bench_weights.clone();
        }
        self.problem.gw_bench_weights(t, self.params).to_vec()
    }

    fn captain_extra(&self, t: usize) -> f64 {
        if self.chips.get(&t).map(String::as_str) == Some("3xc") { 2.0 } else { 1.0 }
    }

    fn squad_constraints(&mut self) {
        let rules = &self.problem.rules;
        for p in &self.problem.players {
            let k = p.player_key;
            for t in 0..self.t_len {
                let prev = if t == 0 {
                    Expr::constant(if p.owned { 1.0 } else { 0.0 })
                } else {
                    Expr::var(self.squad[&(k, t - 1)])
                };
                let buy = self.buy.get(&(k, t)).copied();
                let mut rhs = prev.clone();
                if let Some(b) = buy {
                    rhs.add(b, 1.0);
                }
                let rhs = rhs.plus(&self.sells(k, t), -1.0);
                let squad = Expr::var(self.squad[&(k, t)]);
                self.lp.eq(squad, &rhs);
                if let Some(b) = buy {
                    self.lp.le(Expr::var(b).plus(&self.sells(k, t), 1.0), &Expr::constant(1.0));
                }
                if self.fh.contains(&t) {
                    let fh_buy = self.fh_buy.get(&(k, t)).copied();
                    let mut rhs = prev.clone();
                    if let Some(b) = fh_buy {
                        rhs.add(b, 1.0);
                    }
                    let rhs = rhs.plus(&self.fh_sells(k, t), -1.0);
                    self.lp.eq(Expr::var(self.fh_squad[&(k, t)]), &rhs);
                    if let Some(b) = fh_buy {
                        self.lp
                            .le(Expr::var(b).plus(&self.fh_sells(k, t), 1.0), &Expr::constant(1.0));
                    }
                    if p.owned {
                        // Selling price only if not sold before; buy price only if re-bought.
                        let earlier = self.first_sales_before(k, t);
                        self.lp.le(
                            Expr::var(self.fh_sell_first[&(k, t)]).plus(&earlier, 1.0),
                            &Expr::constant(1.0),
                        );
                        if let Some(&v) = self.fh_sell_later.get(&(k, t)) {
                            self.lp.le(Expr::var(v), &earlier);
                        }
                    }
                }
            }
            if p.owned {
                let all = Expr::sum(
                    (0..self.t_len).filter_map(|t| self.sell_first.get(&(k, t)).copied()),
                );
                self.lp.le(all, &Expr::constant(1.0));
                for t in 0..self.t_len {
                    if let Some(&v) = self.sell_later.get(&(k, t)) {
                        let before = self.first_sales_before(k, t);
                        self.lp.le(Expr::var(v), &before);
                    }
                }
            }
        }

        let mut by_type: BTreeMap<i64, Vec<i64>> = BTreeMap::new();
        let mut by_club: BTreeMap<i64, Vec<i64>> = BTreeMap::new();
        let mut owned_per_club: HashMap<i64, i64> = HashMap::new();
        for p in &self.problem.players {
            by_type.entry(p.element_type).or_default().push(p.player_key);
            by_club.entry(p.team_key).or_default().push(p.player_key);
            if p.owned {
                *owned_per_club.entry(p.team_key).or_default() += 1;
            }
        }
        for t in 0..self.t_len {
            self.selection(rules, &by_type, &by_club, &owned_per_club, false, t);
            if self.fh.contains(&t) {
                self.selection(rules, &by_type, &by_club, &owned_per_club, true, t);
            }
        }
    }

    /// Positions and the club cap for one GW's squad (the FH squad when `fh`).
    fn selection(
        &mut self,
        rules: &fplopt_build::scoring::Rules,
        by_type: &BTreeMap<i64, Vec<i64>>,
        by_club: &BTreeMap<i64, Vec<i64>>,
        owned_per_club: &HashMap<i64, i64>,
        fh: bool,
        t: usize,
    ) {
        let (squad, buy) =
            if fh { (&self.fh_squad, &self.fh_buy) } else { (&self.squad, &self.buy) };
        let mut rows: Vec<(Expr, Expr, bool)> = Vec::new(); // (lhs, rhs, equality)
        for (et, n) in &rules.squad_select {
            let keys = by_type.get(et).map(Vec::as_slice).unwrap_or(&[]);
            rows.push((
                Expr::sum(keys.iter().map(|k| squad[&(*k, t)])),
                Expr::constant(*n as f64),
                true,
            ));
        }
        for (club, keys) in by_club {
            let count = Expr::sum(keys.iter().map(|k| squad[&(*k, t)]));
            let start = owned_per_club.get(club).copied().unwrap_or(0);
            if start <= rules.team_limit {
                rows.push((count, Expr::constant(rules.team_limit as f64), false));
            } else {
                // Pre-existing excess: tolerated unless a player of the club is bought.
                let excess = start - rules.team_limit;
                for k in keys {
                    if let Some(&b) = buy.get(&(*k, t)) {
                        let mut lhs = count.clone();
                        lhs.add(b, excess as f64);
                        rows.push((lhs, Expr::constant(start as f64), false));
                    }
                }
            }
        }
        for (lhs, rhs, equality) in rows {
            if equality { self.lp.eq(lhs, &rhs) } else { self.lp.le(lhs, &rhs) }
        }
    }

    fn lineup_constraints(&mut self) {
        let rules = &self.problem.rules;
        let n_slots = self.params.bench_weights.len();
        let mut bench_of: HashMap<Key, Vec<usize>> = HashMap::new();
        let mut slot: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for &((k, t, s), v) in &self.bench {
            bench_of.entry((k, t)).or_default().push(v);
            slot.entry((t, s)).or_default().push(v);
        }
        let players = &self.problem.players;
        for t in 0..self.t_len {
            for p in players {
                let k = p.player_key;
                let lhs = Expr::var(self.lineup[&(k, t)])
                    .plus(&Expr::sum(bench_of.get(&(k, t)).cloned().unwrap_or_default()), 1.0);
                let selected = Expr::var(self.selected(k, t));
                self.lp.eq(lhs, &selected);
                self.lp.le(Expr::var(self.captain[&(k, t)]), &Expr::var(self.lineup[&(k, t)]));
            }
            let all = Expr::sum(players.iter().map(|p| self.lineup[&(p.player_key, t)]));
            self.lp.eq(all, &Expr::constant(rules.squad_play as f64));
            for (et, min) in &rules.play_min {
                let starters = Expr::sum(
                    players
                        .iter()
                        .filter(|p| p.element_type == *et)
                        .map(|p| self.lineup[&(p.player_key, t)]),
                );
                self.lp.ge(starters.clone(), &Expr::constant(*min as f64));
                self.lp.le(starters, &Expr::constant(rules.play_max[et] as f64));
            }
            for s in 0..n_slots {
                let vars = slot.get(&(t, s)).cloned().unwrap_or_default();
                self.lp.eq(Expr::sum(vars), &Expr::constant(1.0));
            }
            let captains = Expr::sum(players.iter().map(|p| self.captain[&(p.player_key, t)]));
            self.lp.eq(captains, &Expr::constant(1.0));
        }
    }

    /// Σ sell_price·sell_first + Σ price·sell_later − Σ price·buy at GW t.
    fn flow(&self, t: usize, fh: bool) -> Expr {
        let (sell_first, sell_later, buy) = if fh {
            (&self.fh_sell_first, &self.fh_sell_later, &self.fh_buy)
        } else {
            (&self.sell_first, &self.sell_later, &self.buy)
        };
        let mut e = Expr::default();
        for p in &self.problem.players {
            let k = p.player_key;
            if let Some(&v) = sell_first.get(&(k, t)) {
                e.add(v, p.sell_price as f64);
            }
            if let Some(&v) = sell_later.get(&(k, t)) {
                e.add(v, p.price as f64);
            }
            if let Some(&v) = buy.get(&(k, t)) {
                e.add(v, -(p.price as f64));
            }
        }
        e.terms.retain(|(_, c)| *c != 0.0);
        e
    }

    fn money_and_transfers(&mut self) {
        let problem = self.problem;
        let rules = &problem.rules;
        for t in 0..self.t_len {
            let prev = if t == 0 {
                Expr::constant(problem.bank as f64)
            } else {
                Expr::var(self.bank[t - 1])
            };
            let rhs = prev.clone().plus(&self.flow(t, false), 1.0);
            self.lp.eq(Expr::var(self.bank[t]), &rhs);
            if self.fh.contains(&t) {
                let rhs = prev.plus(&self.flow(t, true), 1.0);
                self.lp.eq(Expr::var(self.fh_bank[&t]), &rhs);
            }
        }

        let cap = rules.max_free_transfers;
        let m = (rules.squad_size + cap + rules.ft_topups.iter().map(|(_, a)| a).sum::<i64>() + 1)
            as f64;
        let chip_extra = i64::from(rules.chip_week_ft == "retain_plus_one");
        self.ft = vec![Ft::Const(problem.free_transfers.max(0))];
        self.hits = Vec::new();
        for t in 0..self.t_len {
            let free_gw = t == 0 && problem.gw1;
            let transfer_chip = self.chips.get(&t).is_some_and(|c| is_transfer_chip(c));
            let n = self.n_transfers(t);
            if free_gw || transfer_chip {
                self.hits.push(None);
            } else {
                let upper = self.params.max_hits.map_or(f64::INFINITY, |h| h as f64);
                let hits = self.lp.var(0.0, upper, true);
                let take = self.lp.binary();
                let ft = self.ft[t].expr();
                let n_minus_ft = n.clone().plus(&ft, -1.0);
                self.lp.ge(Expr::var(hits), &n_minus_ft);
                let mut rhs = n_minus_ft.clone();
                rhs.constant += m;
                rhs.add(take, -m);
                self.lp.le(Expr::var(hits), &rhs);
                let mut rhs = Expr::default();
                rhs.add(take, m);
                self.lp.le(Expr::var(hits), &rhs);
                self.hits.push(Some(hits));
            }
            if t + 1 >= self.t_len {
                break;
            }
            let (from, to) = (problem.gws[t].gw_index, problem.gws[t + 1].gw_index);
            let topup: i64 =
                rules.ft_topups.iter().filter(|(g, _)| from < *g && *g <= to).map(|(_, a)| a).sum();
            // A constant target needs no variables: after GW1 the count restarts at 1; a
            // chip GW keeps a constant count (chip_week_ft).
            let constant_target = match (&self.ft[t], free_gw, transfer_chip) {
                (_, true, _) => Some(1 + topup),
                (Ft::Const(c), false, true) => Some(c + chip_extra + topup),
                _ => None,
            };
            if let Some(c) = constant_target {
                self.ft.push(Ft::Const(c.min(cap)));
                continue;
            }
            // ft + chip_extra + top-ups (chip GW), or max(ft − n, 0) + 1 + top-ups pinned
            // with the hits: ft − n + hits + 1 + top-ups.
            let mut target = self.ft[t].expr();
            if transfer_chip {
                target.constant += (chip_extra + topup) as f64;
            } else {
                target = target.plus(&n, -1.0);
                target.add(self.hits[t].expect("a normal GW has hits"), 1.0);
                target.constant += 1.0;
                target.constant += topup as f64;
            }
            let states: Vec<(usize, i64)> = (1..=cap).map(|s| (self.lp.binary(), s)).collect();
            self.lp.eq(Expr::sum(states.iter().map(|(v, _)| *v)), &Expr::constant(1.0));
            let ft_next = Ft::States(states.clone()).expr();
            let capped = self.lp.binary();
            self.lp.le(ft_next.clone(), &target);
            let mut rhs = target;
            rhs.add(capped, -m);
            self.lp.ge(ft_next.clone(), &rhs);
            let mut rhs = Expr::constant(cap as f64 - m);
            rhs.add(capped, m);
            self.lp.ge(ft_next, &rhs);
            self.ft.push(Ft::States(states));
        }
    }

    // --- hooks -------------------------------------------------------------------------

    /// (sale variables by key, buy variables by key) of the first GW (the FH squad's in a
    /// Free Hit first GW).
    fn first_gw_vars(&self) -> (BTreeMap<i64, usize>, BTreeMap<i64, usize>) {
        let (sell_first, buy) = if self.fh.contains(&0) {
            (&self.fh_sell_first, &self.fh_buy_order)
        } else {
            (&self.sell_first, &self.buy_order)
        };
        let sells = self
            .problem
            .players
            .iter()
            .filter(|p| p.owned)
            .map(|p| (p.player_key, sell_first[&(p.player_key, 0)]))
            .collect();
        let buys = buy.iter().filter(|((_, t), _)| *t == 0).map(|((k, _), v)| (*k, *v)).collect();
        (sells, buys)
    }

    pub(crate) fn fix_first_gw(&mut self, transfers: &TransferSet) -> Result<()> {
        let (outs, ins) = transfers;
        let (sells, buys) = self.first_gw_vars();
        let unknown: Vec<i64> = outs
            .iter()
            .filter(|k| !sells.contains_key(k))
            .chain(ins.iter().filter(|k| !buys.contains_key(k)))
            .copied()
            .collect();
        if !unknown.is_empty() {
            bail!("fixed transfers name non-candidates: {unknown:?}");
        }
        // Python iterates the dicts in insertion order: players by key, then buys by key.
        for (k, v) in &sells {
            self.lp.eq(Expr::var(*v), &Expr::constant(if outs.contains(k) { 1.0 } else { 0.0 }));
        }
        for (k, v) in &buys {
            self.lp.eq(Expr::var(*v), &Expr::constant(if ins.contains(k) { 1.0 } else { 0.0 }));
        }
        Ok(())
    }

    /// No-good cut: the first GW's (outs, ins) must differ from `transfers`.
    pub(crate) fn exclude_first_gw(&mut self, transfers: &TransferSet) {
        let (outs, ins) = transfers;
        let (sells, buys) = self.first_gw_vars();
        if !outs.iter().all(|k| sells.contains_key(k)) || !ins.iter().all(|k| buys.contains_key(k))
        {
            return; // the set is out of reach anyway
        }
        let on: Vec<usize> =
            outs.iter().map(|k| sells[k]).chain(ins.iter().map(|k| buys[k])).collect();
        let mut off: Vec<usize> =
            sells.iter().filter(|(k, _)| !outs.contains(k)).map(|(_, v)| *v).collect();
        off.extend(buys.iter().filter(|(k, _)| !ins.contains(k)).map(|(_, v)| *v));
        let lhs = Expr::sum(off).plus(&Expr::sum(on.iter().copied()), -1.0);
        self.lp.ge(lhs, &Expr::constant(1.0 - on.len() as f64));
    }

    // --- objective ---------------------------------------------------------------------

    /// The horizon part of the objective, as (variable, coefficient) merged in first-seen
    /// order with zero coefficients dropped, and its constant.
    fn objective(&self) -> (Vec<(usize, f64)>, f64) {
        let params = self.params;
        let hit_cost = self.problem.rules.hit_cost as f64 + params.hit_margin;
        let mut terms: Vec<(usize, f64)> = Vec::new();
        let mut constant = 0.0;
        let bench: HashMap<(i64, usize, usize), usize> = self.bench.iter().copied().collect();
        let ft_terms = |t: usize| -> Option<Vec<(usize, f64)>> {
            match &self.ft[t] {
                Ft::Const(_) => None,
                Ft::States(states) => {
                    Some(states.iter().map(|&(v, s)| (v, params.ft_state_value(s))).collect())
                }
            }
        };
        let ft_const = |t: usize| -> f64 {
            match &self.ft[t] {
                Ft::Const(c) => params.ft_state_value(*c),
                Ft::States(_) => 0.0,
            }
        };
        for (t, gw) in self.problem.gws.iter().enumerate() {
            let d = params.decay.powf(gw.horizon as f64);
            let (weights, extra) = (self.bench_weights(t), self.captain_extra(t));
            for p in &self.problem.players {
                let (k, x) = (p.player_key, p.xp[t]);
                terms.push((self.lineup[&(k, t)], d * x));
                terms.push((self.captain[&(k, t)], extra * d * x));
                for (s, w) in weights.iter().enumerate() {
                    if let Some(&v) = bench.get(&(k, t, s)) {
                        terms.push((v, d * w * x));
                    }
                }
            }
            terms.push((self.bank[t], d * params.itb_value / 10.0));
            for (i, sign) in [(Some(t), 1.0), (t.checked_sub(1), -1.0)] {
                match i {
                    None => constant += sign * d * 0.0,
                    Some(i) => match ft_terms(i) {
                        None => constant += sign * d * ft_const(i),
                        Some(ts) => terms.extend(ts.into_iter().map(|(v, c)| (v, sign * d * c))),
                    },
                }
            }
            if let Some(h) = self.hits[t] {
                terms.push((h, -d * hit_cost));
            }
            if params.tie_epsilon != 0.0 {
                let buys = if self.fh.contains(&t) { &self.fh_buy_order } else { &self.buy_order };
                for ((_, tt), v) in buys {
                    if *tt == t {
                        terms.push((*v, -d * params.tie_epsilon));
                    }
                }
            }
        }
        let mut merged: Vec<(usize, f64)> = Vec::new();
        let mut position: HashMap<usize, usize> = HashMap::new();
        for (v, c) in terms {
            match position.get(&v) {
                Some(&i) => merged[i].1 += c,
                None => {
                    position.insert(v, merged.len());
                    merged.push((v, c));
                }
            }
        }
        merged.retain(|(_, c)| *c != 0.0);
        (merged, constant)
    }

    /// The model handed to HiGHS in bulk (Python `_BulkHiGHS.buildSolverModel`): columns
    /// with the negated objective (HiGHS minimises), rows without zero coefficients, then
    /// integrality. The objective constant is left out (as PuLP does).
    fn to_highs(&self, highs: &mut Highs, mip: bool) -> Result<()> {
        let (terms, _) = self.objective();
        let mut cost: Vec<f64> = vec![-0.0; self.lp.lower.len()];
        for (v, c) in terms {
            cost[v] = -c;
        }
        highs.add_cols(&cost, &self.lp.lower, &self.lp.upper)?;
        let (mut starts, mut index, mut value, mut lower, mut upper) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for row in &self.lp.rows {
            starts.push(index.len() as i32);
            for &(v, c) in &row.terms {
                if c != 0.0 {
                    index.push(v as i32);
                    value.push(c);
                }
            }
            lower.push(row.lower);
            upper.push(row.upper);
        }
        highs.add_rows(&lower, &upper, &starts, &index, &value)?;
        if mip {
            let integer: Vec<i32> = (0..self.lp.integer.len())
                .filter(|&j| self.lp.integer[j])
                .map(|j| j as i32)
                .collect();
            highs.set_integer(&integer)?;
        }
        Ok(())
    }
}

/// HiGHS options for `params` in PuLP's order: output off, gap, threads, the wall-clock limit
/// when set, a fixed seed and the deterministic node limit (`mip` False: an LP relaxation,
/// without gap and node limit).
fn configure(highs: &mut Highs, params: &OptimizerParams, mip: bool) -> Result<()> {
    highs.set_bool("output_flag", false)?;
    if mip {
        highs.set_double("mip_rel_gap", params.mip_gap)?;
    }
    highs.set_int("threads", params.threads)?;
    if let Some(limit) = params.time_limit {
        highs.set_double("time_limit", limit)?;
    }
    highs.set_int("random_seed", 0)?;
    if mip && let Some(nodes) = params.node_limit {
        highs.set_int("mip_max_nodes", nodes)?;
    }
    Ok(())
}

/// PuLP's status name for a HiGHS model status, and whether a stop leaves a solution.
fn status_name(status: ModelStatus) -> (&'static str, bool) {
    use ModelStatus::*;
    match status {
        Notset | ModelEmpty => ("NotSolved", false),
        Optimal => ("Optimal", false),
        Infeasible => ("Infeasible", false),
        Unbounded => ("Unbounded", false),
        ObjectiveBound | ObjectiveTarget => ("GapLimit", true),
        Interrupt => ("Interrupted", true),
        TimeLimit => ("TimeLimit", true),
        IterationLimit => ("IterationLimit", true),
        SolutionLimit => ("SolutionLimit", true),
        _ => ("Undefined", false),
    }
}

/// Build and solve one scenario's MILP (Python `model.solve_plan`). `fix_first_gw` pins
/// the first GW's transfers, each `exclude_first_gw` entry is a no-good cut.
pub fn solve_plan(
    problem: &PlanInput,
    params: &OptimizerParams,
    fix_first_gw: Option<&TransferSet>,
    exclude_first_gw: &[TransferSet],
    scenario: &ChipScenario,
) -> Result<Plan, SolveError> {
    let start = Instant::now();
    let mut model = Model::new(problem, params, scenario)?;
    if let Some(fix) = fix_first_gw {
        model.fix_first_gw(fix)?;
    }
    for transfers in exclude_first_gw {
        model.exclude_first_gw(transfers);
    }
    let terminal = terminal_value(problem, params, scenario);
    let (_, constant) = model.objective();
    let constant = constant + terminal;
    let build_seconds = start.elapsed().as_secs_f64();

    let start = Instant::now();
    let mut highs = Highs::new();
    configure(&mut highs, params, true)?;
    model.to_highs(&mut highs, true)?;
    if let Ok(dir) = std::env::var("FPLOPT_HIGHS_DUMP") {
        let n = std::fs::read_dir(&dir).map(|d| d.count()).unwrap_or(0);
        highs.write_model(&format!("{dir}/{n:04}-{}.mps", scenario.label))?;
    }
    highs.run();
    let solve_seconds = start.elapsed().as_secs_f64();
    let model_status = highs.model_status();
    let (status, stopped) = status_name(model_status);
    let has_solution = model_status == ModelStatus::Optimal
        || (stopped && highs.objective_value() != f64::INFINITY);
    let dual_bound = highs.double_info("mip_dual_bound")?;
    let mut bound = constant - dual_bound;
    if !bound.is_finite() {
        bound = f64::INFINITY;
    }
    if !has_solution {
        if model_status == ModelStatus::Infeasible {
            return Err(SolveError::Infeasible(format!(
                "no feasible plan (HiGHS status {status})"
            )));
        }
        return Err(SolveError::LimitReached { status: status.to_string(), bound });
    }
    let x = highs.col_values();
    let gws = read_plan(&model, &x);
    let mut objective = 0.0;
    for g in &gws {
        objective += g.objective;
    }
    Ok(Plan {
        gws,
        objective,
        status: status.to_string(),
        mip_gap: highs.double_info("mip_gap")?,
        n_nodes: highs.int64_info("mip_node_count")?,
        build_seconds,
        solve_seconds,
        n_candidates: problem.players.len(),
        scenario: scenario.label.clone(),
        terminal_value: terminal,
        gain_vs_roll: None,
        bound: bound.max(objective + terminal),
    })
}

/// An upper bound on the best total objective of `scenario`: its LP relaxation's optimum
/// plus a small relative margin; inf when HiGHS doesn't report the LP optimal.
pub fn relaxation_bound(
    problem: &PlanInput,
    params: &OptimizerParams,
    scenario: &ChipScenario,
) -> Result<f64> {
    let model = Model::new(problem, params, scenario)?;
    let (_, constant) = model.objective();
    let constant = constant + terminal_value(problem, params, scenario);
    let mut highs = Highs::new();
    configure(&mut highs, params, false)?;
    model.to_highs(&mut highs, false)?;
    highs.run();
    if highs.model_status() != ModelStatus::Optimal {
        return Ok(f64::INFINITY);
    }
    let value = constant - highs.double_info("objective_function_value")?;
    Ok(value + BOUND_MARGIN * value.abs().max(1.0))
}

/// The solution as GwPlans (vice = best non-captain starter by xP, ties by key).
fn read_plan(model: &Model, x: &[f64]) -> Vec<GwPlan> {
    let problem = model.problem;
    let params = model.params;
    let hit_cost = problem.rules.hit_cost as f64 + params.hit_margin;
    let et: HashMap<i64, i64> =
        problem.players.iter().map(|p| (p.player_key, p.element_type)).collect();
    let on = |v: usize| x[v] > ON;
    let mut out = Vec::new();
    let mut previous_ft_value = 0.0;
    for (t, gw) in problem.gws.iter().enumerate() {
        let xp: HashMap<i64, f64> =
            problem.players.iter().map(|p| (p.player_key, p.xp[t])).collect();
        let freehit = model.fh.contains(&t);
        let outs: Vec<i64> = problem
            .players
            .iter()
            .filter(|p| {
                let sells = if freehit {
                    model.fh_sells(p.player_key, t)
                } else {
                    model.sells(p.player_key, t)
                };
                sells.value(x) > ON
            })
            .map(|p| p.player_key)
            .collect();
        let buys = if freehit { &model.fh_buy_order } else { &model.buy_order };
        let ins: Vec<i64> =
            buys.iter().filter(|((_, tt), v)| *tt == t && on(*v)).map(|((k, _), _)| *k).collect();
        let transfers = pair(&outs, &ins, &et);
        let mut starters: Vec<i64> = problem
            .players
            .iter()
            .map(|p| p.player_key)
            .filter(|k| on(model.lineup[&(*k, t)]))
            .collect();
        starters.sort_by(|a, b| {
            et[a].cmp(&et[b]).then((-xp[a]).partial_cmp(&-xp[b]).unwrap()).then(a.cmp(b))
        });
        let mut slots: Vec<(usize, i64)> = model
            .bench
            .iter()
            .filter(|((_, tt, _), v)| *tt == t && on(*v))
            .map(|((k, _, s), _)| (*s, *k))
            .collect();
        slots.sort_unstable();
        let bench: Vec<i64> = slots.iter().map(|(_, k)| *k).collect();
        let captain = problem
            .players
            .iter()
            .map(|p| p.player_key)
            .find(|k| on(model.captain[&(*k, t)]))
            .expect("one captain");
        let vice = *starters
            .iter()
            .filter(|k| **k != captain)
            .min_by(|a, b| (-xp[a]).partial_cmp(&-xp[b]).unwrap().then(a.cmp(b)))
            .expect("a vice");
        let hits = model.hits[t].map_or(0, |h| py_round(x[h]));
        let kept_bank = py_round(x[model.bank[t]]); // the persistent bank
        let bank = if freehit { py_round(x[model.fh_bank[&t]]) } else { kept_bank };
        let ft = py_round(model.ft[t].expr().value(x));
        let ft_value = params.ft_state_value(ft);
        let mut lineup_xp = 0.0;
        for k in &starters {
            lineup_xp += xp[k];
        }
        lineup_xp += model.captain_extra(t) * xp[&captain];
        let weights = model.bench_weights(t);
        let mut bench_xp = 0.0;
        for (w, k) in weights.iter().zip(&bench) {
            bench_xp += w * xp[k];
        }
        let d = params.decay.powf(gw.horizon as f64);
        let objective = d
            * (lineup_xp + bench_xp + ft_value - previous_ft_value
                + params.itb_value * kept_bank as f64 / 10.0
                - hit_cost * hits as f64
                - params.tie_epsilon * ins.len() as f64);
        previous_ft_value = ft_value;
        out.push(GwPlan {
            gw: gw.gw,
            gw_index: gw.gw_index,
            horizon: gw.horizon,
            transfers,
            starters,
            bench,
            captain,
            vice,
            chip: model.chips.get(&t).cloned(),
            xp: lineup_xp,
            bench_xp,
            objective,
            hits,
            bank,
            ft,
        });
    }
    out
}

/// Python's `round` to an int (half to even).
fn py_round(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// (out, in) pairs within position, by player_key.
fn pair(outs: &[i64], ins: &[i64], et: &HashMap<i64, i64>) -> Vec<Transfer> {
    let positions: BTreeSet<i64> = outs.iter().chain(ins).map(|k| et[k]).collect();
    let mut pairs = Vec::new();
    for position in positions {
        let mut o: Vec<i64> = outs.iter().copied().filter(|k| et[k] == position).collect();
        let mut i: Vec<i64> = ins.iter().copied().filter(|k| et[k] == position).collect();
        o.sort_unstable();
        i.sort_unstable();
        assert_eq!(o.len(), i.len(), "position {position}: {} out, {} in", o.len(), i.len());
        pairs.extend(o.into_iter().zip(i).map(|(out_key, in_key)| Transfer { out_key, in_key }));
    }
    pairs
}
