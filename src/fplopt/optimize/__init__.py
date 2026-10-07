"""MILP transfer planner (PLAN §7; Phase 4 plan): parameters, planner input, candidate
pruning, chip scenarios, the model and its plans.

`PlanInput.from_context(state, pool, xp, rules, params)` → `optimize(problem, params)` →
`PlanSet` (top-k plans over the chip scenarios plus the roll plan); `solve_plan` solves one
scenario with fixed/excluded first-GW transfers → `Plan`; `Plan.decision()` is the first
GW's `state.Decision`. Needs the `optimize` extra (PuLP + highspy) for solving.
"""

from fplopt.optimize.chips import ChipScenario, scenarios, terminal_value
from fplopt.optimize.params import OptimizerParams
from fplopt.optimize.plans import GwPlan, Plan, PlanSet, optimize
from fplopt.optimize.problem import HorizonGw, PlanInput, Player
from fplopt.optimize.prune import prune

__all__ = (
    "ChipScenario",
    "GwPlan",
    "HorizonGw",
    "OptimizerParams",
    "Plan",
    "PlanInput",
    "PlanSet",
    "Player",
    "optimize",
    "prune",
    "scenarios",
    "terminal_value",
)
