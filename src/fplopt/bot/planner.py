"""A user's plan on a plan context (Phase 7 plan §1, §4): import their squad from FPL's public
API (`fplopt.entry`), apply their `/setft` override, and run the optimizer with the settings
of the system that passed Phase 6 (`LIVE_PARAMS`, `holdout.SYSTEM`), no chips (chips stay
the manager's call until a chip policy passes its own test, PLAN §12).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import httpx

from fplopt.backtest.rules import Rules, backtest_rules
from fplopt.backtest.state import SquadState, chip_available
from fplopt.bot.context import PlanContext
from fplopt.entry import EntryClient, EntryError, SquadSpec, import_squad, squad_state
from fplopt.optimize import OptimizerParams, Plan, PlanInput, PlanSet, optimize

__all__ = (
    "CHIP_LABELS",
    "ChipOptions",
    "LIVE_PARAMS",
    "TOP_K",
    "UnknownTeam",
    "UserPlan",
    "chip_options",
    "chips_left",
    "decision_summary",
    "import_user_squad",
    "plan_user",
    "plans_json",
    "ranked",
)

# `holdout.SYSTEM` (optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2; the other
# settings at their defaults) with the interactive wall-clock cap per solve
# (cli.INTERACTIVE_TIME_LIMIT); tests check it matches the spec.
LIVE_PARAMS = OptimizerParams(horizon=6, decay=0.75, max_hits=1, hit_margin=2.0, time_limit=60.0)
TOP_K = 3
CHIP_LABELS = {
    "wildcard": "Wildcard",
    "freehit": "Free Hit",
    "bboost": "Bench Boost",
    "3xc": "Triple Captain",
}


class UnknownTeam(EntryError):
    """FPL has no team with that ID."""


@dataclass(frozen=True)
class UserPlan:
    """A user's squad (`spec`, `state`) and plans on `context`."""

    context: PlanContext
    entry_id: int
    spec: SquadSpec
    state: SquadState
    plans: PlanSet
    rules: Rules
    ft_override: int | None = None
    params: OptimizerParams = LIVE_PARAMS


def import_user_squad(
    fpl: EntryClient, entry_id: int, context: PlanContext, ft_override: int | None = None
) -> tuple[SquadSpec, SquadState, Rules]:
    """The FPL team's squad before the context's GW, as a spec and a state (EntryError when it
    can't be imported or isn't usable)."""
    rules = backtest_rules(context.season)
    try:
        spec = import_squad(
            fpl,
            entry_id,
            context.season,
            context.gw,
            rules,
            context.gw_index_map,
            context.initial_prices,
        )
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            raise UnknownTeam(f"FPL has no team with ID {entry_id}") from None
        raise
    if ft_override is not None:
        spec = replace(
            spec,
            free_transfers=ft_override,
            notes=(*spec.notes, f"free transfers {ft_override}: set by you (/setft)"),
        )
    state = squad_state(spec, context.pool, context.element_keys, rules, context.gw_index_map)
    return spec, state, rules


def plan_user(
    fpl: EntryClient,
    entry_id: int,
    context: PlanContext,
    ft_override: int | None = None,
    params: OptimizerParams = LIVE_PARAMS,
) -> UserPlan:
    """Import the squad and plan it: the top `TOP_K` plans and the roll plan, no chips."""
    spec, state, rules = import_user_squad(fpl, entry_id, context, ft_override)
    problem = PlanInput.from_context(state, context.pool, context.xp, rules, params)
    plans = optimize(problem, params, top_k=TOP_K, chips=False, roll=True)
    return UserPlan(context, entry_id, spec, state, plans, rules, ft_override, params)


@dataclass(frozen=True)
class ChipOptions:
    """`/chips`: the chip scenario search on a user's squad (information only: the chip
    planner isn't validated, PLAN §11). `plans.scenario_objectives` holds every solved
    scenario's best total objective (`none` = no chip); `plans.best` is the winner."""

    context: PlanContext
    entry_id: int
    spec: SquadSpec
    state: SquadState
    plans: PlanSet
    rules: Rules
    params: OptimizerParams = LIVE_PARAMS
    # chip name -> (scenario label, total objective): each available chip on its own, in
    # its most promising horizon GW (the single-chip scenario with the highest bound)
    singles: Mapping[str, tuple[str, float]] = field(default_factory=dict)


def chip_options(
    fpl: EntryClient,
    entry_id: int,
    context: PlanContext,
    ft_override: int | None = None,
    params: OptimizerParams = LIVE_PARAMS,
) -> ChipOptions:
    """Import the squad and search the chip scenarios over the horizon (the bound search:
    only scenarios that could win are solved; same best plan as solving them all)."""
    spec, state, rules = import_user_squad(fpl, entry_id, context, ft_override)
    problem = PlanInput.from_context(state, context.pool, context.xp, rules, params)
    plans = optimize(problem, params, top_k=1, chips=True, roll=False)
    return ChipOptions(
        context, entry_id, spec, state, plans, rules, params, _singles(problem, params, plans)
    )


def _singles(
    problem: PlanInput, params: OptimizerParams, plans: PlanSet
) -> dict[str, tuple[str, float]]:
    """Per chip name, its single-chip scenario with the highest bound, solved unless the
    search already did (one extra MILP per chip at most)."""
    from fplopt.optimize.chips import scenarios
    from fplopt.optimize.model import InfeasiblePlan, solve_plan

    bounds = plans.scenario_bounds
    promising: dict[str, Any] = {}
    for scenario in scenarios(problem, params):
        if len(scenario.chips) != 1 or scenario.label not in bounds:
            continue
        name = scenario.chips[0][1]
        if name not in promising or bounds[scenario.label] > bounds[promising[name].label]:
            promising[name] = scenario
    out = {}
    for name, scenario in promising.items():
        value = plans.scenario_objectives.get(scenario.label)
        if value is None:
            try:
                value = solve_plan(problem, params, chips=scenario).total_objective
            except InfeasiblePlan:
                continue
        out[name] = (scenario.label, float(value))
    return out


def chips_left(state: SquadState, rules: Rules) -> list[str]:
    """The chips the manager can still play at the planned GW (labels, in a fixed order)."""
    return [label for name, label in CHIP_LABELS.items() if chip_available(state, name, rules)]


def ranked(plans: PlanSet) -> list[Plan]:
    """The top-k plans by total objective, best first. `optimize` returns them in solve
    order, which can be inverted within the MIP gap (PLAN §8 live finding: plan #3 above
    plan #1), so the bot shows the best plan found first."""
    return sorted(plans.plans, key=lambda p: -p.total_objective)


def decision_summary(plan: UserPlan) -> dict[str, Any]:
    """What the alerts compare: the best plan's moves for this GW (player keys, sorted) and
    its captain and vice."""
    first = ranked(plan.plans)[0].first
    return {
        "transfers": sorted([t.out_key, t.in_key] for t in first.transfers),
        "captain": int(first.captain),
        "vice": int(first.vice),
    }


def plans_json(plan: UserPlan) -> dict[str, Any]:
    """The plans as plain data for the live log."""

    def one(p: Any) -> dict[str, Any] | None:
        if p is None:
            return None
        return {
            "total_objective": p.total_objective,
            "gain_vs_roll": p.gain_vs_roll,
            "status": p.status,
            "mip_gap": p.mip_gap,
            "gws": [
                {
                    "gw": g.gw,
                    "transfers": [[t.out_key, t.in_key] for t in g.transfers],
                    "starters": list(g.starters),
                    "bench": list(g.bench),
                    "captain": g.captain,
                    "vice": g.vice,
                    "chip": g.chip,
                    "xp": g.xp,
                    "hits": g.hits,
                    "bank": g.bank,
                    "ft": g.ft,
                }
                for g in p.gws
            ],
        }

    return {
        "entry_id": plan.entry_id,
        "bank": plan.spec.bank,
        "free_transfers": plan.spec.free_transfers,
        "squad": [[p.element_id, p.purchase_price] for p in plan.spec.players],
        "chips_used": [list(c) for c in plan.spec.chips_used],
        "params": {
            "horizon": plan.params.horizon,
            "decay": plan.params.decay,
            "max_hits": plan.params.max_hits,
            "hit_margin": plan.params.hit_margin,
        },
        "plans": [one(p) for p in ranked(plan.plans)],
        "roll": one(plan.plans.roll),
    }
