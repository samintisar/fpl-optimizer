//! Chip scenario search (Python `fplopt.optimize.search`; PLAN §7 *Chips*): the best plan
//! over the scenarios. `Search::All` solves every scenario; `Search::Bound` (default) solves
//! only those whose upper bound (LP relaxation, or one derived from the scenario's
//! Wildcard/Free Hit base) could beat the best plan found, and returns the same plan. The
//! method and its guarantee are documented in the Python module docstring.

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use anyhow::{Result, bail};
use log::warn;

use crate::chips::{ChipScenario, make_scenario, terminal_value};
use crate::model::{BOUND_MARGIN, SolveError, relaxation_bound, solve_plan};
use crate::params::OptimizerParams;
use crate::plans::Plan;
use crate::problem::PlanInput;
use crate::state::{GOALKEEPER, is_transfer_chip};
use crate::util::py_sum;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Search {
    Bound,
    All,
}

impl Search {
    pub fn name(self) -> &'static str {
        match self {
            Search::Bound => "bound",
            Search::All => "all",
        }
    }
}

/// How a search went (Python `SearchStats`).
#[derive(Clone, Debug, PartialEq)]
pub struct SearchStats {
    pub search: String,
    pub n_scenarios: usize,
    pub n_relaxations: usize,
    pub n_solves: usize,
    pub relax_seconds: f64,
    pub solve_seconds: f64,
    /// (scenario label, solver status) of every solve a limit stopped.
    pub not_optimal: Vec<(String, String)>,
}

/// The best plan and its scenario, the solved scenarios' total objectives and every
/// scenario's final bound (enumeration order), and the stats.
pub struct SearchResult {
    pub plan: Plan,
    pub scenario: ChipScenario,
    pub objectives: Vec<(String, f64)>,
    pub bounds: Vec<(String, f64)>,
    pub stats: SearchStats,
}

/// An upper bound on the objective a Triple Captain or Bench Boost at position `t` adds to
/// any plan; inf for any other chip.
pub fn chip_extra(problem: &PlanInput, params: &OptimizerParams, t: usize, name: &str) -> f64 {
    let d = params.decay.powf(problem.gws[t].horizon as f64);
    match name {
        "3xc" => d * problem.players.iter().map(|p| p.xp[t].abs()).fold(0.0, f64::max),
        "bboost" => {
            let w = problem.gw_bench_weights(t, params);
            let mut keepers: Vec<f64> = problem
                .players
                .iter()
                .filter(|p| p.element_type == GOALKEEPER)
                .map(|p| p.xp[t].abs())
                .collect();
            let mut outfield: Vec<f64> = problem
                .players
                .iter()
                .filter(|p| p.element_type != GOALKEEPER)
                .map(|p| p.xp[t].abs())
                .collect();
            keepers.sort_by(f64::total_cmp);
            outfield.sort_by(f64::total_cmp);
            let mut slots: Vec<f64> = w[1..].iter().map(|x| (1.0 - x).abs()).collect();
            slots.sort_by(|a, b| b.total_cmp(a));
            let mut total = (1.0 - w[0]).abs() * keepers.last().copied().unwrap_or(0.0);
            total += py_sum(slots.iter().zip(outfield.iter().rev()).map(|(a, x)| a * x));
            d * total
        }
        _ => f64::INFINITY,
    }
}

struct State<'a> {
    problem: &'a PlanInput,
    params: &'a OptimizerParams,
    candidates: &'a [ChipScenario],
    plans: BTreeMap<usize, Plan>,
    limited: BTreeMap<usize, (String, f64)>, // stopped at a limit without a plan
    lp: BTreeMap<usize, f64>,
    relax_seconds: f64,
    solve_seconds: f64,
}

impl State<'_> {
    fn solve(&mut self, i: usize) -> Result<()> {
        let start = Instant::now();
        let label = &self.candidates[i].label;
        match solve_plan(self.problem, self.params, None, &[], &self.candidates[i]) {
            Ok(plan) => {
                if !plan.optimal() {
                    warn!(
                        "chip scenario {label}: solver stopped at a limit ({}, gap {:.2}%)",
                        plan.status,
                        100.0 * plan.mip_gap
                    );
                }
                self.plans.insert(i, plan);
            }
            Err(SolveError::LimitReached { status, bound }) => {
                if i == 0 {
                    bail!("no plan found before the solver stopped (status {status})");
                }
                warn!(
                    "chip scenario {label}: no plan before the solver stopped ({status}); skipped"
                );
                self.limited.insert(i, (status, bound));
            }
            Err(e @ SolveError::Infeasible(_)) => bail!("{e}"),
            Err(SolveError::Other(e)) => return Err(e),
        }
        self.solve_seconds += start.elapsed().as_secs_f64();
        Ok(())
    }

    fn relax(&mut self, i: usize) -> Result<()> {
        let start = Instant::now();
        let bound = relaxation_bound(self.problem, self.params, &self.candidates[i])?;
        self.lp.insert(i, bound);
        self.relax_seconds += start.elapsed().as_secs_f64();
        Ok(())
    }

    /// The best plan so far by `score` (ties to the earlier scenario).
    fn incumbent(&self) -> usize {
        let mut best: Option<usize> = None;
        for &i in self.plans.keys() {
            if best.is_none_or(|b| self.score(i) > self.score(b)) {
                best = Some(i);
            }
        }
        best.expect("the no-chip plan")
    }

    /// What scenario i's plan is ranked by (Python `search._score`): its total objective,
    /// less (1 − `fh_scale`) of its Free Hit gain over the no-chip plan in the Free Hit GWs
    /// (never below 0). Only lowers a scenario's value, so the upper bounds stay valid.
    fn score(&self, i: usize) -> f64 {
        let plan = &self.plans[&i];
        if self.params.fh_scale == 1.0 {
            return plan.total_objective();
        }
        let base = &self.plans[&0];
        let gain = py_sum(
            self.candidates[i]
                .chips
                .iter()
                .filter(|(_, name)| name == "freehit")
                .map(|(t, _)| plan.gws[*t].objective - base.gws[*t].objective),
        );
        plan.total_objective() - (1.0 - self.params.fh_scale) * gain.max(0.0)
    }
}

/// The best plan over `candidates` (no chip first); ties go to the earlier candidate.
pub fn search(
    problem: &PlanInput,
    params: &OptimizerParams,
    candidates: &[ChipScenario],
    how: Search,
) -> Result<SearchResult> {
    if candidates.first() != Some(&ChipScenario::none()) {
        bail!("candidates must start with the no-chip scenario");
    }
    let n = candidates.len();
    let mut s = State {
        problem,
        params,
        candidates,
        plans: BTreeMap::new(),
        limited: BTreeMap::new(),
        lp: BTreeMap::new(),
        relax_seconds: 0.0,
        solve_seconds: 0.0,
    };
    let bounds: Vec<(String, f64)> = match how {
        Search::All => {
            for i in 0..n {
                s.solve(i)?;
            }
            candidates
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    let b = s.plans.get(&i).map_or_else(|| s.limited[&i].1, |p| p.bound);
                    (c.label.clone(), b)
                })
                .collect()
        }
        Search::Bound => bound_search(&mut s)?,
    };
    let not_optimal: Vec<(String, String)> = candidates
        .iter()
        .enumerate()
        .filter_map(|(i, c)| match (s.plans.get(&i), s.limited.get(&i)) {
            (_, Some((status, _))) => Some((c.label.clone(), status.clone())),
            (Some(p), None) if !p.optimal() => Some((c.label.clone(), p.status.clone())),
            _ => None,
        })
        .collect();
    let stats = SearchStats {
        search: how.name().to_string(),
        n_scenarios: n,
        n_relaxations: s.lp.len(),
        n_solves: s.plans.len() + s.limited.len(),
        relax_seconds: s.relax_seconds,
        solve_seconds: s.solve_seconds,
        not_optimal,
    };
    let best = s.incumbent();
    let objectives =
        s.plans.iter().map(|(i, p)| (candidates[*i].label.clone(), p.total_objective())).collect();
    Ok(SearchResult {
        plan: s.plans.remove(&best).unwrap(),
        scenario: candidates[best].clone(),
        objectives,
        bounds,
        stats,
    })
}

/// Best-first search with bounds; returns every scenario's final bound by label.
fn bound_search(s: &mut State) -> Result<Vec<(String, f64)>> {
    let (problem, params, candidates) = (s.problem, s.params, s.candidates);
    let n = candidates.len();
    let index: HashMap<&ChipScenario, usize> =
        candidates.iter().enumerate().map(|(i, c)| (c, i)).collect();
    let terminal: Vec<f64> =
        candidates.iter().map(|c| terminal_value(problem, params, c)).collect();
    // base[i]: scenario i's base (itself without TC/BB chips, None if not a candidate);
    // offset[i]: Σ E + T(i) − T(base).
    let mut base: Vec<Option<usize>> = Vec::new();
    let mut offset: Vec<f64> = Vec::new();
    for (i, c) in candidates.iter().enumerate() {
        let extras: Vec<&(usize, String)> =
            c.chips.iter().filter(|(_, n)| !is_transfer_chip(n)).collect();
        if extras.is_empty() {
            base.push(Some(i));
            offset.push(0.0);
            continue;
        }
        let kept: BTreeMap<usize, String> = c
            .chips
            .iter()
            .filter(|(_, n)| is_transfer_chip(n))
            .map(|(t, n)| (*t, n.clone()))
            .collect();
        let b = if kept.is_empty() {
            index.get(&ChipScenario::none()).copied()
        } else {
            make_scenario(problem, &kept).ok().and_then(|sc| index.get(&sc).copied())
        };
        base.push(b);
        let extra = py_sum(extras.iter().map(|(t, name)| chip_extra(problem, params, *t, name)));
        offset.push(match b {
            None => f64::INFINITY,
            Some(b) => extra + terminal[i] - terminal[b],
        });
    }

    let own = |s: &State, i: usize| -> f64 {
        let mut value = s.lp.get(&i).copied().unwrap_or(f64::INFINITY);
        let bound = s.plans.get(&i).map(|p| p.bound).or_else(|| s.limited.get(&i).map(|l| l.1));
        if let Some(bound) = bound {
            value = value.min(bound + BOUND_MARGIN * bound.abs().max(1.0));
        }
        value
    };
    let upper = |s: &State, i: usize| -> f64 {
        let derived = match base[i] {
            Some(b) if b != i => own(s, b) + offset[i],
            _ => f64::INFINITY,
        };
        own(s, i).min(derived)
    };

    s.solve(0)?;
    loop {
        let open: Vec<usize> =
            (0..n).filter(|i| !s.plans.contains_key(i) && !s.limited.contains_key(i)).collect();
        if open.is_empty() {
            break;
        }
        // The open scenario with the highest bound (ties: the earlier).
        let mut i = open[0];
        for &j in &open[1..] {
            if upper(s, j) > upper(s, i) {
                i = j;
            }
        }
        let u = upper(s, i);
        let incumbent = s.incumbent();
        let z = s.score(incumbent);
        if u < z || (u <= z && i > incumbent) {
            break; // no open scenario can beat (or tie earlier than) the incumbent
        }
        let b = base[i];
        match b {
            Some(b)
                if b != i
                    && !s.lp.contains_key(&b)
                    && !s.plans.contains_key(&b)
                    && !s.limited.contains_key(&b) =>
            {
                s.relax(b)?
            }
            _ if !s.lp.contains_key(&i) => s.relax(i)?,
            _ => s.solve(i)?,
        }
    }
    Ok(candidates
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let b = match (s.plans.get(&i), s.limited.get(&i)) {
                (Some(p), _) => p.bound,
                (None, Some(l)) => l.1,
                _ => upper(s, i),
            };
            (c.label.clone(), b)
        })
        .collect())
}
