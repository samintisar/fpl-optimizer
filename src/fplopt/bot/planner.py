"""A user's plan on a plan context (Phase 7 plan §1, §4): import their squad from FPL's public
API (`fplopt.entry`), apply their `/setft` override, and run the optimizer with the settings
of the system that passed Phase 6 (`LIVE_PARAMS`, `holdout.SYSTEM`), no chips (chips stay
the manager's call until a chip policy passes its own test, PLAN §12).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import httpx

from fplopt.backtest.rules import Rules, backtest_rules
from fplopt.backtest.state import Holding, SquadState, chip_available, selling_price
from fplopt.bot.context import PlanContext
from fplopt.entry import (
    EntryClient,
    EntryError,
    SquadPlayer,
    SquadSpec,
    import_squad,
    squad_state,
)
from fplopt.optimize import OptimizerParams, Plan, PlanInput, PlanSet, optimize

__all__ = (
    "CHIP_LABELS",
    "ChipOptions",
    "LIVE_PARAMS",
    "TOP_K",
    "apply_made_transfers",
    "UnknownTeam",
    "UserPlan",
    "chip_options",
    "chip_params",
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
CHIP_SHRINK = 1.0  # the chip study's chosen P (chip pre-registration §6c)
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


def apply_made_transfers(
    spec: SquadSpec, made: Sequence[tuple[int, int]], context: PlanContext
) -> SquadSpec:
    """The squad after the transfers the manager already made for the planned GW (`made`:
    (element out, element in) in order; FPL doesn't publish them before the deadline): each
    sale at its FPL selling price, each buy at the current price (which becomes its purchase
    price), the bank and free transfers updated (a move beyond the free transfers was a hit,
    already paid). EntryError when a move isn't possible."""
    if not made:
        return spec
    keys = context.element_keys
    price = dict(zip(context.pool["player_key"], context.pool["price"], strict=True))
    names = context.names
    players = {p.element_id: p for p in spec.players}
    bank = spec.bank
    for out, into in made:
        if out not in players:
            raise EntryError(f"{_name(out, keys, names)} isn't in the squad")
        if into in players:
            raise EntryError(f"{_name(into, keys, names)} is already in the squad")
        if keys.get(into) not in price or keys.get(out) not in price:
            raise EntryError(f"{_name(into, keys, names)} isn't in this GW's player pool")
        held = Holding(keys[out], 0, 0, players[out].purchase_price, int(price[keys[out]]))
        cost = int(price[keys[into]])
        bank += selling_price(held) - cost
        del players[out]
        players[into] = SquadPlayer(into, cost, names.get(keys[into], ""))
    if bank < 0:
        raise EntryError(f"those transfers need £{-bank / 10:.1f}m more than you have")
    moves = ", ".join(f"{_name(o, keys, names)} → {_name(i, keys, names)}" for o, i in made)
    hits = max(0, len(made) - spec.free_transfers)
    left = max(0, spec.free_transfers - len(made))
    note = (
        f"transfers you told the bot about (/transfer): {moves}; bank now £{bank / 10:.1f}m, "
        f"{left} free transfer(s) left"
    )
    if hits:
        note += f"; {hits} of them beyond your free transfers (hits already taken)"
    return replace(
        spec,
        players=tuple(players.values()),
        bank=bank,
        free_transfers=left,
        notes=(*spec.notes, note),
    )


def _name(element: int, keys: Mapping[int, int], names: Mapping[int, str]) -> str:
    return names.get(keys.get(element, -1), f"element {element}")


def import_user_squad(
    fpl: EntryClient,
    entry_id: int,
    context: PlanContext,
    ft_override: int | None = None,
    made: Sequence[tuple[int, int]] = (),
) -> tuple[SquadSpec, SquadState, Rules]:
    """The FPL team's squad before the context's GW, after the transfers already `made` for
    it (`apply_made_transfers`), as a spec and a state (EntryError when it can't be imported
    or isn't usable)."""
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
    spec = apply_made_transfers(spec, made, context)
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
    made: Sequence[tuple[int, int]] = (),
) -> UserPlan:
    """Import the squad (after the transfers already `made`) and plan it: the top `TOP_K`
    plans and the roll plan, no chips."""
    spec, state, rules = import_user_squad(fpl, entry_id, context, ft_override, made)
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


def chip_params() -> OptimizerParams:
    """`/chips`'s planner settings: `LIVE_PARAMS` with the chip study's estimated terminal
    values (`config/chips/terminal-values.json`, chip_shrink 1.0: the study's P, PLAN §12),
    or the flat placeholders if the table is missing. Information only: P didn't beat the
    rule baseline at the pre-registered threshold."""
    from fplopt.backtest.chip_values import load_terminal_values

    try:
        return replace(LIVE_PARAMS, chip_value_left=load_terminal_values(CHIP_SHRINK))
    except ValueError:
        return LIVE_PARAMS


def chip_options(
    fpl: EntryClient,
    entry_id: int,
    context: PlanContext,
    ft_override: int | None = None,
    params: OptimizerParams | None = None,
    made: Sequence[tuple[int, int]] = (),
) -> ChipOptions:
    """Import the squad and search the chip scenarios over the horizon (the bound search:
    only scenarios that could win are solved; same best plan as solving them all), with
    `params` (default `chip_params()`)."""
    params = chip_params() if params is None else params
    spec, state, rules = import_user_squad(fpl, entry_id, context, ft_override, made)
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

    prices_at = plan.context.prices_at
    return {
        "entry_id": plan.entry_id,
        "prices_at": None if prices_at is None else prices_at.isoformat(),
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
