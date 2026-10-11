"""Chip scenario search: the best plan over `chips.scenarios` (Phase 4 plan, Task 3; PLAN §7
*Chips*).

Each chip scenario is a separate MILP (0.2-4 s on real data) and a 6-GW horizon has ~200-240
of them, so solving them all (`search="all"`) takes minutes per deadline. `search="bound"`
(the default) returns the same best plan, solving only the scenarios that could win:

1. Solve the no-chip scenario; its plan is the incumbent.
2. Every other scenario gets an **upper bound** on its best `total_objective`:
   - its **LP relaxation** (`model.relaxation_bound`): the same model without integrality.
     On real data it is within ~0-3 points of the MILP optimum (often equal), and costs
     ~0.06 s against the MILP's 0.2-4 s;
   - or, cheaper, a bound **derived from its base** (the scenario with only its Wildcard
     and Free Hit chips): any plan of scenario s = base + Triple Captain/Bench Boost
     chips is a plan of the base too (those chips change no constraint), worth at most
     `E` less per such chip, so `UB(s) = UB(base) − T(base) + Σ E + T(s)` (T = terminal
     value). `E` bounds the objective a chip adds in GW t (decay d_t, xP x over all
     candidates): Triple Captain `d_t · max |x|` (the captain counts once more); Bench
     Boost `d_t · Σ_k |1 − w_tk| · |x|_(k)` over the bench slots, w_tk GW t's bench
     weights (the GK slot with the largest |x| of a goalkeeper, the outfield slots'
     |1 − w_tk| in descending order with the three largest outfield |x|: a rearrangement
     bound on Σ (1 − w_tk) · xP(bench_k)).
     `UB(base)` is the base's LP bound or its MILP dual bound (`Plan.bound`).
3. Best-first: repeatedly take the open scenario with the highest bound. If the bound is
   below the incumbent, stop: no remaining scenario can beat it. Otherwise tighten it
   (the base's LP bound first, since one base serves many scenarios, then the scenario's
   own LP bound); once it already has its LP bound, solve its MILP and update the
   incumbent.

**Guarantee.** A scenario is skipped only when an upper bound on its *true* optimum is below
a plan already found, so no skipped scenario has a better plan than the one returned, and
every scenario that is solved is solved exactly as `search="all"` solves it (same model,
same HiGHS settings, deterministic). Hence `"bound"` returns the plan `"all"` returns (same
scenario, same objective; ties go to the earlier scenario in enumeration order in both),
and like `"all"` it is within `mip_gap` of the best plan over all scenarios: each solved
scenario's plan is within `mip_gap` of that scenario's optimum, and each skipped scenario's
optimum is below the returned plan. Bounds carry a small relative margin
(`model.BOUND_MARGIN`) for HiGHS's feasibility tolerances.

The guarantee assumes every solve reaches `mip_gap`. A solve stopped by a limit
(`OptimizerParams.node_limit`/`time_limit`) is never silent: its (label, status) goes into
`SearchStats.not_optimal` and a warning is logged. A scenario stopped before HiGHS found any
plan (`model.SolveLimitReached`) is skipped with its dual bound kept (the search continues;
only the no-chip scenario, whose plan the search needs, re-raises).

Only the scenarios that are solved get an objective in `PlanSet.scenario_objectives`; every
scenario's final bound is in `PlanSet.scenario_bounds`.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

from fplopt.backtest.state import GOALKEEPER, TRANSFER_CHIPS, InvalidDecision
from fplopt.optimize.chips import NO_CHIP, ChipScenario, make_scenario, terminal_value
from fplopt.optimize.model import BOUND_MARGIN, SolveLimitReached, relaxation_bound, solve_plan
from fplopt.optimize.params import OptimizerParams
from fplopt.optimize.plans import Plan
from fplopt.optimize.problem import PlanInput

SEARCHES = ("bound", "all")
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchStats:
    """How a scenario search went: `search` ("bound"/"all"), `n_scenarios` enumerated,
    `n_relaxations` LP bounds computed, `n_solves` scenario MILPs solved, and the seconds
    spent on each (`relax_seconds`, `solve_seconds`, wall time incl. model builds).
    `not_optimal`: (scenario label, solver status) of every scenario solve a limit stopped
    (with or without a plan); empty when every solve reached `mip_gap`."""

    search: str
    n_scenarios: int
    n_relaxations: int = 0
    n_solves: int = 0
    relax_seconds: float = 0.0
    solve_seconds: float = 0.0
    not_optimal: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class SearchResult:
    """The best plan and its scenario, the solved scenarios' total objectives and every
    scenario's final upper bound (label → value, enumeration order), and the stats."""

    plan: Plan
    scenario: ChipScenario
    objectives: dict[str, float]
    bounds: dict[str, float]
    stats: SearchStats


def chip_extra(problem: PlanInput, params: OptimizerParams, t: int, name: str) -> float:
    """An upper bound on the objective a Triple Captain or Bench Boost at horizon position
    `t` adds to any plan (module docstring); `math.inf` for any other chip."""
    d = params.decay ** problem.gws[t].horizon
    if name == "3xc":
        return d * max((abs(p.xp[t]) for p in problem.players), default=0.0)
    if name == "bboost":
        w = problem.gw_bench_weights(t, params)
        keepers = sorted(abs(p.xp[t]) for p in problem.players if p.element_type == GOALKEEPER)
        outfield = sorted(abs(p.xp[t]) for p in problem.players if p.element_type != GOALKEEPER)
        slots = sorted((abs(1.0 - x) for x in w[1:]), reverse=True)
        total = abs(1.0 - w[0]) * (keepers[-1] if keepers else 0.0)
        total += sum(a * x for a, x in zip(slots, reversed(outfield), strict=False))
        return d * total
    return math.inf


def search(
    problem: PlanInput,
    params: OptimizerParams,
    candidates: Sequence[ChipScenario],
    how: str = "bound",
) -> SearchResult:
    """The best plan over `candidates` (no chip first). `how="all"` solves every scenario;
    `"bound"` skips those that provably can't win (module docstring). Ties in
    `total_objective` go to the earlier candidate."""
    if how not in SEARCHES:
        raise ValueError(f"scenario search must be one of {SEARCHES}, got {how!r}")
    candidates = list(candidates)
    if not candidates or candidates[0] != NO_CHIP:
        raise ValueError("candidates must start with the no-chip scenario")
    n = len(candidates)
    plans: dict[int, Plan] = {}
    limited: dict[int, SolveLimitReached] = {}  # stopped at a limit without a plan
    lp: dict[int, float] = {}
    clock = {"relax": 0.0, "solve": 0.0}

    def solve(i: int) -> None:
        start = time.perf_counter()
        try:
            plans[i] = solve_plan(problem, params, chips=candidates[i])
        except SolveLimitReached as exc:
            if i == 0:
                raise
            log.warning(
                "chip scenario %s: no plan before the solver stopped (%s); skipped",
                candidates[i].label,
                exc.status,
            )
            limited[i] = exc
        else:
            if not plans[i].optimal:
                log.warning(
                    "chip scenario %s: solver stopped at a limit (%s, gap %.2f%%)",
                    candidates[i].label,
                    plans[i].status,
                    100 * plans[i].mip_gap,
                )
        clock["solve"] += time.perf_counter() - start

    if how == "all":
        for i in range(n):
            solve(i)
        bounds = {
            s.label: plans[i].bound if i in plans else limited[i].bound
            for i, s in enumerate(candidates)
        }
    else:
        bounds = _bound_search(problem, params, candidates, plans, limited, lp, clock, solve)
    not_optimal = [
        (s.label, plans[i].status if i in plans else limited[i].status)
        for i, s in enumerate(candidates)
        if i in limited or (i in plans and not plans[i].optimal)
    ]
    stats = SearchStats(
        search=how,
        n_scenarios=n,
        n_relaxations=len(lp),
        n_solves=len(plans) + len(limited),
        relax_seconds=clock["relax"],
        solve_seconds=clock["solve"],
        not_optimal=tuple(not_optimal),
    )
    best = max(plans, key=lambda i: (plans[i].total_objective, -i))
    objectives = {candidates[i].label: plans[i].total_objective for i in sorted(plans)}
    return SearchResult(plans[best], candidates[best], objectives, bounds, stats)


def _bound_search(
    problem, params, candidates, plans, limited, lp, clock, solve
) -> dict[str, float]:
    """Best-first search with bounds (module docstring); fills `plans`, `limited` and `lp`
    and returns every scenario's final bound by label."""
    n = len(candidates)
    index = {s: i for i, s in enumerate(candidates)}
    terminal = [terminal_value(problem, params, s) for s in candidates]
    # base[i]: the index of scenario i's base (itself if it has no TC/BB chip, None if the
    # base isn't a candidate); offset[i]: Σ E + T(i) − T(base).
    base: list[int | None] = []
    offset: list[float] = []
    for i, s in enumerate(candidates):
        extras = [(t, name) for t, name in s.chips if name not in TRANSFER_CHIPS]
        if not extras:
            base.append(i)
            offset.append(0.0)
            continue
        kept = {t: name for t, name in s.chips if name in TRANSFER_CHIPS}
        try:
            b = index.get(make_scenario(problem, kept) if kept else NO_CHIP)
        except InvalidDecision:
            b = None
        base.append(b)
        extra = sum(chip_extra(problem, params, t, name) for t, name in extras)
        offset.append(math.inf if b is None else extra + terminal[i] - terminal[b])

    def own(i: int) -> float:
        """The tightest bound of scenario i itself (LP or MILP), inf if none yet."""
        value = lp.get(i, math.inf)
        if i in plans or i in limited:
            bound = plans[i].bound if i in plans else limited[i].bound
            value = min(value, bound + BOUND_MARGIN * max(1.0, abs(bound)))
        return value

    def upper(i: int) -> float:
        b = base[i]
        derived = own(b) + offset[i] if b is not None and b != i else math.inf
        return min(own(i), derived)

    def relax(i: int) -> None:
        start = time.perf_counter()
        lp[i] = relaxation_bound(problem, params, candidates[i])
        clock["relax"] += time.perf_counter() - start

    solve(0)
    while True:
        open_ = [i for i in range(n) if i not in plans and i not in limited]
        if not open_:
            break
        i = max(open_, key=lambda j: (upper(j), -j))
        u = upper(i)
        incumbent = max(plans, key=lambda j: (plans[j].total_objective, -j))
        z = plans[incumbent].total_objective
        if u < z or (u <= z and i > incumbent):
            break  # no open scenario can beat (or tie earlier than) the incumbent
        b = base[i]
        if b is not None and b != i and b not in lp and b not in plans and b not in limited:
            relax(b)
        elif i not in lp:
            relax(i)
        else:
            solve(i)
    return {
        s.label: plans[i].bound if i in plans else limited[i].bound if i in limited else upper(i)
        for i, s in enumerate(candidates)
    }
