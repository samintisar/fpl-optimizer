"""Plans: the MILP's answer per horizon GW, its first-GW decision, and the top-k plans,
roll plan and chip scenarios around it (Phase 4 plan, Decisions and Tasks 1-2; PLAN §7
*Practicalities*).

A `Plan` holds one `GwPlan` per horizon GW plus solve statistics. `Plan.decision()` is the
first GW's `state.Decision` (transfers paired within position, lineup, chip), which
`state.apply_decision` accepts.

`optimize(problem, params, top_k=3, chips=True, roll=True, scenario_search="bound") ->
PlanSet`:

1. **Chip scenarios** (`chips.scenarios`; only no-chip when `chips=False`): each is a
   separate MILP, ranked by `Plan.total_objective` (horizon objective + terminal value of
   the unused chips). The best is plan #1; ties go to the earlier scenario (no chip first).
   `scenario_search="bound"` (`search`) solves only the scenarios whose upper bound (LP
   relaxation, or a bound derived from a related scenario) could beat the best plan found,
   and returns the same plan as `"all"`, which solves every scenario. The solved
   scenarios' total objectives are kept in `PlanSet.scenario_objectives`, every
   scenario's bound in `PlanSet.scenario_bounds`.
2. **Top-k:** plans #2..k re-solve **within plan #1's chip scenario**, each with no-good
   cuts excluding the first-GW transfer sets of the plans before it, so the k plans differ
   in what to do this GW given the winning chip plan. (Cutting across scenarios would need
   k solves per scenario, and the alternatives a manager wants are other moves now, not
   other chip timings, which `scenario_objectives` already shows.) Fewer than k plans come
   back if the cuts leave no feasible plan. Objectives are non-increasing up to the MIP gap.
3. **Roll plan** (None if a solver limit stopped its solve before any plan was found, with a
   warning): no first-GW transfers (`fix_first_gw` with empty sets), the rest of the
   horizon optimized, in plan #1's scenario, except that a Wildcard or Free Hit plan #1
   plays in the first GW is dropped (a transfer chip with no transfers is a wasted chip;
   the dropped chip is then valued as unused). If plan #1 already makes no first-GW
   transfers in that same scenario, it is the roll plan (no extra solve).
4. Every plan's `gain_vs_roll` = its `total_objective` − the roll plan's (the decayed xP
   gain over the horizon, net of hits, FT and bank values and chip terminal values).
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING

from fplopt.backtest.gw_score import Lineup
from fplopt.backtest.state import TRANSFER_CHIPS, Decision, Transfer
from fplopt.optimize.chips import NO_CHIP, scenarios
from fplopt.optimize.params import OptimizerParams
from fplopt.optimize.problem import PlanInput

if TYPE_CHECKING:
    from fplopt.optimize.search import SearchStats

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GwPlan:
    """One horizon GW of a plan.

    - `transfers`: (out, in) pairs, paired within position (each GW keeps 2/5/5/3, so
      every position sells as many as it buys); `n_transfers = len(transfers)`. In a Free
      Hit GW they build the one-GW squad from the squad held before it.
    - `starters` (by position, then xP descending), `bench` (slot order, bench[0] the GK),
      `captain`, `vice` (the best other starter by xP, set after the solve), `chip`.
    - `xp`: expected points of the XI with the captain counted twice (three times under
      Triple Captain); no bench, no vice.
    - `bench_xp`: Σ_k w_k · xP(bench[k]), the bench as the objective values it (w = the
      bench weights, all 1 under Bench Boost).
    - `objective`: this GW's decayed contribution to the objective:
      decay**horizon · (xp + bench_xp + ft gain + itb_value · kept bank/10 − hit cost ·
      hits), the ft gain being V(ft) − V(previous ft) (V(previous) = 0 in the first GW)
      and the kept bank the money carried to the next GW (`bank`, except in a Free Hit GW,
      where it is the bank from before the Free Hit).
    - `hits`: transfers paid for; `bank` (tenths of £m) after this GW's transfers, as
      `apply_decision` leaves it (in a Free Hit GW, the FH squad's leftover); `ft`: free
      transfers available for this GW (before its transfers; meaningless at GW1).
    """

    gw: int
    gw_index: int
    horizon: int
    transfers: tuple[Transfer, ...]
    starters: tuple[int, ...]
    bench: tuple[int, ...]
    captain: int
    vice: int
    chip: str | None
    xp: float
    bench_xp: float
    objective: float
    hits: int
    bank: int
    ft: int

    @property
    def n_transfers(self) -> int:
        return len(self.transfers)

    @property
    def transfers_out(self) -> tuple[int, ...]:
        return tuple(t.out_key for t in self.transfers)

    @property
    def transfers_in(self) -> tuple[int, ...]:
        return tuple(t.in_key for t in self.transfers)

    @property
    def transfer_set(self) -> tuple[frozenset[int], frozenset[int]]:
        """(outs, ins) as sets: the identity the top-k cuts and `fix_first_gw` use."""
        return frozenset(self.transfers_out), frozenset(self.transfers_in)

    @property
    def lineup(self) -> Lineup:
        return Lineup(self.starters, self.bench, self.captain, self.vice)

    def decision(self) -> Decision:
        return Decision(transfers=self.transfers, lineup=self.lineup, chip=self.chip)


@dataclass(frozen=True)
class Plan:
    """A solved plan. `objective` = Σ of the GWs' `objective`, recomputed from the
    integral solution (the solver's objective up to its tolerances), excluding the chip
    terminal value; `terminal_value` is the scenario's value of chips left unused
    (`chips.terminal_value`) and `total_objective` their sum, which plans are ranked by.
    `status` is HiGHS's model status as PuLP reports it (`Optimal` also when stopped at
    `mip_gap`; `NodeLimit`/`SolutionLimit`/`TimeLimit` when a limit stopped it with a
    solution, see `OptimizerParams`: then the plan is not within `mip_gap`); `mip_gap` the
    final relative gap; `n_nodes` the branch-and-bound nodes explored; `build_seconds` the
    PuLP model build and `solve_seconds` the HiGHS call (including PuLP's hand-over);
    `n_candidates` the players in the model.
    `scenario` is the chip scenario's label (`none` without chips) and `gain_vs_roll`
    the `total_objective` gain over the roll plan (set by `optimize`, else None).
    `bound` is HiGHS's dual bound mapped to `total_objective` terms: no plan of this
    scenario (with the same fixed/excluded first-GW transfers) beats it, and it is
    ≥ `total_objective` (equal up to `mip_gap`)."""

    gws: tuple[GwPlan, ...]
    objective: float
    status: str
    mip_gap: float
    build_seconds: float
    solve_seconds: float
    n_candidates: int
    scenario: str = "none"
    terminal_value: float = 0.0
    gain_vs_roll: float | None = None
    bound: float = math.inf
    n_nodes: int = 0

    @property
    def optimal(self) -> bool:
        """Solved to `mip_gap` (no limit stopped it)."""
        return self.status == "Optimal"

    @property
    def first(self) -> GwPlan:
        return self.gws[0]

    @property
    def total_xp(self) -> float:
        """Σ over the horizon of the XI's xP with captain (undiscounted)."""
        return sum(g.xp for g in self.gws)

    @property
    def total_objective(self) -> float:
        return self.objective + self.terminal_value

    @property
    def chips(self) -> tuple[tuple[int, str], ...]:
        """(gw, chip) of the GWs that play a chip."""
        return tuple((g.gw, g.chip) for g in self.gws if g.chip is not None)

    def decision(self) -> Decision:
        """The first GW's decision (with its chip)."""
        return self.first.decision()


@dataclass(frozen=True)
class PlanSet:
    """`plans`: the top-k plans by objective, distinct in first-GW transfer sets (plan #1
    first); `roll`: the roll plan (None if not asked for); `scenario_objectives`: chip
    scenario label → that scenario's best `total_objective`, in enumeration order;
    `seconds`: wall time of the whole `optimize` call. With `scenario_search="bound"`
    only the scenarios that were solved are in `scenario_objectives`; `scenario_bounds`
    holds every scenario's upper bound on its best `total_objective` (the MILP's dual bound
    if solved), and `search` how the search went (`search.SearchStats`)."""

    plans: tuple[Plan, ...]
    roll: Plan | None
    scenario_objectives: Mapping[str, float]
    seconds: float = 0.0
    scenario_bounds: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))
    search: SearchStats | None = None

    @property
    def best(self) -> Plan:
        return self.plans[0]

    @property
    def solver_status(self) -> str:
        """`Optimal` when plan #1 and every chip scenario solve of its search reached
        `mip_gap`; else plan #1's status if a limit stopped it, or `Optimal (n chip
        scenario solve(s) stopped at a limit)`."""
        if not self.best.optimal:
            return self.best.status
        limited = 0 if self.search is None else len(self.search.not_optimal)
        if limited:
            return f"Optimal ({limited} chip scenario solve(s) stopped at a limit)"
        return self.best.status

    def decision(self) -> Decision:
        """Plan #1's first-GW decision."""
        return self.best.decision()


def optimize(
    problem: PlanInput,
    params: OptimizerParams,
    *,
    top_k: int = 3,
    chips: bool = True,
    roll: bool = True,
    scenario_search: str = "bound",
) -> PlanSet:
    """Best plan over the chip scenarios, plans #2..top_k in its scenario, and the roll
    plan (see the module docstring). `chips=False` solves the no-chip scenario only.
    `scenario_search` is "bound" (skip scenarios that provably can't win; same result) or
    "all" (solve every scenario), see `search`."""
    from fplopt.optimize.model import (  # model imports this module
        InfeasiblePlan,
        SolveLimitReached,
        solve_plan,
    )
    from fplopt.optimize.search import search

    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    start = time.perf_counter()
    candidates = scenarios(problem, params) if chips else (NO_CHIP,)
    found = search(problem, params, candidates, scenario_search)
    plan1, scenario1 = found.plan, found.scenario

    plans = [plan1]
    while len(plans) < top_k:
        try:
            plans.append(
                solve_plan(
                    problem,
                    params,
                    chips=scenario1,
                    exclude_first_gw=[p.first.transfer_set for p in plans],
                )
            )
        except SolveLimitReached as exc:
            log.warning(
                "plan #%d: no plan before the solver stopped (%s)", len(plans) + 1, exc.status
            )
            break
        except InfeasiblePlan:
            break

    roll_plan = None
    if roll:
        roll_scenario = scenario1
        if scenario1.chip_at(0) in TRANSFER_CHIPS:
            roll_scenario = scenario1.without(0, problem)
        if roll_scenario == scenario1 and plan1.first.n_transfers == 0:
            roll_plan = plan1
        else:
            try:
                roll_plan = solve_plan(
                    problem,
                    params,
                    chips=roll_scenario,
                    fix_first_gw=(frozenset(), frozenset()),
                )
            except SolveLimitReached as exc:
                log.warning("roll plan: no plan before the solver stopped (%s)", exc.status)
        if roll_plan is not None:
            base = roll_plan.total_objective
            plans = [replace(p, gain_vs_roll=p.total_objective - base) for p in plans]
            roll_plan = replace(roll_plan, gain_vs_roll=0.0)
    # Plan #1's solves are checked (and logged) by the search.
    others = [(f"plan #{i}", p) for i, p in enumerate(plans[1:], start=2)]
    for label, plan in [*others, ("roll plan", roll_plan)]:
        if plan is not None and not plan.optimal:
            log.warning("%s: solver stopped at a limit (%s)", label, plan.status)
    return PlanSet(
        plans=tuple(plans),
        roll=roll_plan,
        scenario_objectives=MappingProxyType(found.objectives),
        seconds=time.perf_counter() - start,
        scenario_bounds=MappingProxyType(found.bounds),
        search=found.stats,
    )
