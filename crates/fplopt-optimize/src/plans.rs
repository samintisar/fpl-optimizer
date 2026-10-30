//! Plans (Python `fplopt.optimize.plans`; PLAN §7 *Practicalities*): the MILP's answer per
//! horizon GW, and `optimize`: the best plan over the chip scenarios (`search`), plans
//! #2..k in its scenario (no-good cuts on the first GW's transfers), the roll plan (no
//! first-GW transfers; a Wildcard or Free Hit in the first GW dropped) and every plan's
//! gain over it.

use std::collections::BTreeSet;
use std::time::Instant;

use anyhow::{Result, bail};
use log::warn;

use crate::chips::{ChipScenario, scenarios};
use crate::model::{SolveError, TransferSet, solve_plan};
use crate::params::OptimizerParams;
use crate::problem::PlanInput;
use crate::search::{Search, SearchStats, search};
use crate::state::{Transfer, is_transfer_chip};

/// One horizon GW of a plan (fields as Python's `GwPlan`).
#[derive(Clone, Debug, PartialEq)]
pub struct GwPlan {
    pub gw: i64,
    pub gw_index: i64,
    pub horizon: i64,
    pub transfers: Vec<Transfer>,
    pub starters: Vec<i64>,
    pub bench: Vec<i64>,
    pub captain: i64,
    pub vice: i64,
    pub chip: Option<String>,
    pub xp: f64,
    pub bench_xp: f64,
    pub objective: f64,
    pub hits: i64,
    pub bank: i64,
    pub ft: i64,
}

impl GwPlan {
    pub fn n_transfers(&self) -> usize {
        self.transfers.len()
    }

    /// (outs, ins) as sets: what the top-k cuts and `fix_first_gw` use.
    pub fn transfer_set(&self) -> TransferSet {
        (
            self.transfers.iter().map(|t| t.out_key).collect(),
            self.transfers.iter().map(|t| t.in_key).collect(),
        )
    }
}

/// A solved plan (fields as Python's `Plan`; `objective` excludes the chip terminal value).
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub gws: Vec<GwPlan>,
    pub objective: f64,
    pub status: String,
    pub mip_gap: f64,
    pub build_seconds: f64,
    pub solve_seconds: f64,
    pub n_candidates: usize,
    pub scenario: String,
    pub terminal_value: f64,
    pub gain_vs_roll: Option<f64>,
    pub bound: f64,
    pub n_nodes: i64,
}

impl Plan {
    pub fn optimal(&self) -> bool {
        self.status == "Optimal"
    }

    pub fn first(&self) -> &GwPlan {
        &self.gws[0]
    }

    pub fn total_objective(&self) -> f64 {
        self.objective + self.terminal_value
    }
}

/// The plans of an `optimize` call (fields as Python's `PlanSet`; label maps in scenario
/// enumeration order).
#[derive(Clone, Debug)]
pub struct PlanSet {
    pub plans: Vec<Plan>,
    pub roll: Option<Plan>,
    pub scenario_objectives: Vec<(String, f64)>,
    pub seconds: f64,
    pub scenario_bounds: Vec<(String, f64)>,
    pub search: SearchStats,
}

impl PlanSet {
    pub fn best(&self) -> &Plan {
        &self.plans[0]
    }

    /// `Optimal` when plan #1 and every scenario solve of its search reached the gap.
    pub fn solver_status(&self) -> String {
        if !self.best().optimal() {
            return self.best().status.clone();
        }
        let limited = self.search.not_optimal.len();
        if limited > 0 {
            return format!("Optimal ({limited} chip scenario solve(s) stopped at a limit)");
        }
        self.best().status.clone()
    }
}

/// Best plan over the chip scenarios, plans #2..top_k in its scenario, and the roll plan
/// (Python `plans.optimize`). `chips` false solves the no-chip scenario only.
pub fn optimize(
    problem: &PlanInput,
    params: &OptimizerParams,
    top_k: usize,
    chips: bool,
    roll: bool,
    how: Search,
) -> Result<PlanSet> {
    if top_k < 1 {
        bail!("top_k must be >= 1, got {top_k}");
    }
    let start = Instant::now();
    let candidates = if chips { scenarios(problem, params) } else { vec![ChipScenario::none()] };
    let found = search(problem, params, &candidates, how)?;
    let (plan1, scenario1) = (found.plan, found.scenario);

    let mut plans = vec![plan1.clone()];
    while plans.len() < top_k {
        let cuts: Vec<TransferSet> = plans.iter().map(|p| p.first().transfer_set()).collect();
        match solve_plan(problem, params, None, &cuts, &scenario1) {
            Ok(plan) => plans.push(plan),
            Err(SolveError::LimitReached { status, .. }) => {
                warn!("plan #{}: no plan before the solver stopped ({status})", plans.len() + 1);
                break;
            }
            Err(SolveError::Infeasible(_)) => break,
            Err(SolveError::Other(e)) => return Err(e),
        }
    }

    let mut roll_plan = None;
    if roll {
        let mut roll_scenario = scenario1.clone();
        if scenario1.chip_at(0).is_some_and(is_transfer_chip) {
            roll_scenario = scenario1.without(0, problem)?;
        }
        if roll_scenario == scenario1 && plan1.first().n_transfers() == 0 {
            roll_plan = Some(plan1.clone());
        } else {
            let empty: TransferSet = (BTreeSet::new(), BTreeSet::new());
            match solve_plan(problem, params, Some(&empty), &[], &roll_scenario) {
                Ok(plan) => roll_plan = Some(plan),
                Err(SolveError::LimitReached { status, .. }) => {
                    warn!("roll plan: no plan before the solver stopped ({status})");
                }
                Err(SolveError::Infeasible(m)) => bail!("roll plan: {m}"),
                Err(SolveError::Other(e)) => return Err(e),
            }
        }
        if let Some(r) = &mut roll_plan {
            let base = r.total_objective();
            for p in &mut plans {
                p.gain_vs_roll = Some(p.total_objective() - base);
            }
            r.gain_vs_roll = Some(0.0);
        }
    }
    for (i, plan) in plans.iter().enumerate().skip(1) {
        if !plan.optimal() {
            warn!("plan #{}: solver stopped at a limit ({})", i + 1, plan.status);
        }
    }
    if let Some(r) = &roll_plan
        && !r.optimal()
    {
        warn!("roll plan: solver stopped at a limit ({})", r.status);
    }
    Ok(PlanSet {
        plans,
        roll: roll_plan,
        scenario_objectives: found.objectives,
        seconds: start.elapsed().as_secs_f64(),
        scenario_bounds: found.bounds,
        search: found.stats,
    })
}
