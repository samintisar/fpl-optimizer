"""Telegram message texts (Phase 7 plan §5): Telegram HTML (`<b>` and escaped text only),
short lines for a phone. `recommendation_text` is the full recommendation (alerts and
`/plan`); `split_message` cuts a text to Telegram's 4,096-character limit at line breaks.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from fplopt.bot.planner import CHIP_LABELS, ChipOptions, UserPlan, chips_left, ranked
from fplopt.seasons import season_label

__all__ = (
    "MAX_MESSAGE",
    "UK",
    "changes_text",
    "chips_text",
    "recommendation_text",
    "split_message",
    "squad_text",
    "uk_time",
)

UK = ZoneInfo("Europe/London")
MAX_MESSAGE = 4096
POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
ARROW = "→"


def uk_time(when: datetime) -> str:
    """'Sat 17 Oct 11:00' in UK time."""
    return when.astimezone(UK).strftime("%a %d %b %H:%M")


def _e(text: object) -> str:
    return html.escape(str(text), quote=False)


def _moves(g: Any, names: Mapping[int, str], hit_cost: int) -> str:
    moves = ", ".join(f"{names[t.out_key]} {ARROW} {names[t.in_key]}" for t in g.transfers)
    hit = f" (−{hit_cost * g.hits} hit)" if g.hits else ""
    return _e(moves or "no transfers") + hit


def _tied(a: Any, b: Any) -> bool:
    """Plans whose objectives differ by less than the solver's gap: equally good as far as
    the solver can tell (PLAN §8 live finding)."""
    gap = max(a.mip_gap, b.mip_gap) * abs(a.total_objective)
    return abs(a.total_objective - b.total_objective) <= gap


def _lineup(g: Any, plan: UserPlan, names: Mapping[int, str]) -> list[str]:
    positions = dict(
        zip(plan.context.pool["player_key"], plan.context.pool["element_type"], strict=True)
    )
    marks = {g.captain: " (C)", g.vice: " (V)"}
    lines = []
    for pos, label in POSITIONS.items():
        players = [k for k in g.starters if int(positions.get(k, 0)) == pos]
        if players:
            text = ", ".join(f"{names[k]}{marks.get(k, '')}" for k in players)
            lines.append(f"{label} {_e(text)}")
    lines.append("Bench " + _e(", ".join(names[k] for k in g.bench)))
    return lines


def recommendation_text(
    plan: UserPlan, *, title: str | None = None, changes: Sequence[str] = ()
) -> str:
    """The recommendation for `plan` (module docstring); `title` replaces the first line's
    lead, `changes` (from `changes_text`) go right under it."""
    context, spec, plans = plan.context, plan.spec, plan.plans
    names = context.names
    gw = context.gw
    lines = [
        f"<b>{_e(title or f'GW{gw} plan')}</b> · deadline {uk_time(context.deadline)} (UK)",
    ]
    lines += [_e(c) for c in changes]
    chips = ", ".join(chips_left(plan.state, plan.rules)) or "none"
    lines += [
        f"Team {plan.entry_id} · bank £{spec.bank / 10:.1f}m · {spec.free_transfers} FT",
        f"Chips you can play this GW: {_e(chips)}",
        f"xP data from {uk_time(context.bootstrap_at or context.created_at)} (UK)",
        "",
    ]
    if not plans.plans:
        lines.append("The solver found no plan. The admin has been told.")
        return "\n".join(lines)
    order = ranked(plans)
    best = order[0]
    roll = plans.roll
    gain = "" if best.gain_vs_roll is None else f" · {best.gain_vs_roll:+.1f} xP vs saving the FT"
    lines.append(f"<b>Plan 1</b>{gain} (next {len(best.gws)} GWs)")
    for g in best.gws:
        lines.append(f"GW{g.gw}: {_moves(g, names, plan.rules.hit_cost)}")
    first = best.first
    lines += [
        f"Captain {_e(names[first.captain])} · vice {_e(names[first.vice])}",
        "",
        f"<b>GW{gw} team</b> (plan 1)",
        *_lineup(first, plan, names),
        "",
    ]
    for i, other in enumerate(order[1:], start=2):
        gain = "" if other.gain_vs_roll is None else f" ({other.gain_vs_roll:+.1f})"
        lines.append(
            f"<b>Plan {i}</b>{gain}: GW{gw} {_moves(other.first, names, plan.rules.hit_cost)}"
        )
    if roll is not None and all(roll is not p for p in order):
        lines.append(f"<b>Save the FT</b>: GW{gw} xP {roll.first.xp:.1f} (plan 1: {first.xp:.1f})")
    lines.append("")
    tied = [1] + [i for i, other in enumerate(order[1:], start=2) if _tied(best, other)]
    if len(tied) > 1:
        which = ", ".join(str(i) for i in tied[:-1]) + f" and {tied[-1]}"
        lines.append(
            f"Plans {which} are within the solver's precision of each other: treat them as "
            "equally good."
        )
    if roll is not None and all(roll is not p for p in order) and _tied(best, roll):
        lines.append("Saving the FT is as good as plan 1 within the solver's precision.")
    if not best.optimal:
        lines.append(f"The solver stopped early ({_e(best.status)}): plan 1 may not be the best.")
    lines += [
        "Chips are your call: the system doesn't plan them yet.",
        f"Transfers you've already made for GW{gw} can't be seen before the deadline.",
    ]
    notes = [n for n in spec.notes if not n.startswith("transfers or a chip made for")]
    if notes:
        lines += ["", "Notes:"] + [f"· {_e(n)}" for n in notes]
    return "\n".join(lines)


def changes_text(
    before: Mapping[str, Any] | None, after: Mapping[str, Any], names: Mapping[int, str]
) -> list[str]:
    """What changed in plan 1's decision for this GW (`planner.decision_summary`) since the
    last alert; empty if nothing did."""
    if before is None:
        return []
    out = []

    def moves(summary: Mapping[str, Any]) -> str:
        pairs = summary["transfers"]
        return ", ".join(f"{names[o]} {ARROW} {names[i]}" for o, i in pairs) or "no transfers"

    if sorted(map(tuple, before["transfers"])) != sorted(map(tuple, after["transfers"])):
        out.append(f"Transfers changed: was {moves(before)}, now {moves(after)}.")
    if before["captain"] != after["captain"]:
        out.append(
            f"Captain changed: was {names[before['captain']]}, now {names[after['captain']]}."
        )
    if before["vice"] != after["vice"]:
        out.append(f"Vice changed: was {names[before['vice']]}, now {names[after['vice']]}.")
    return out


def squad_text(plan_context: Any, spec: Any, entry_id: int) -> str:
    """The imported squad, for /register's reply."""
    names = {
        int(e): str(n)
        for e, n in zip(
            plan_context.players["element_id"], plan_context.players["web_name"], strict=True
        )
    }
    players = ", ".join(names.get(p.element_id, str(p.element_id)) for p in spec.players)
    used = ", ".join(f"{name} (GW{gw})" for name, gw in spec.chips_used) or "none"
    return "\n".join(
        [
            f"<b>Team {entry_id}</b> before GW{spec.gw} ({_e(season_label(spec.season))})",
            f"Squad: {_e(players)}",
            f"Bank £{spec.bank / 10:.1f}m · {spec.free_transfers} free transfers",
            f"Chips played: {_e(used)}",
        ]
    )


def scenario_label(label: str) -> str:
    """`bboost@gw9+freehit@gw11` -> 'Bench Boost GW9 + Free Hit GW11'; `none` -> 'no chip'."""
    if label == "none":
        return "no chip"
    parts = []
    for part in label.split("+"):
        name, _, gw = part.partition("@gw")
        parts.append(f"{CHIP_LABELS.get(name, name)} GW{gw}")
    return " + ".join(parts)


def chips_text(options: ChipOptions, shown: int = 6) -> str:
    """`/chips` (information only): the chips still unused, the best chip plan over the
    horizon and the solved scenarios against holding every chip."""
    context, rules, params = options.context, options.rules, options.params
    gw_of = {i: g for g, i in context.gw_index_map.items()}
    used = {cid for cid, _ in options.state.chips_used}
    unused = [
        f"{CHIP_LABELS.get(w.name, w.name)} (until GW{gw_of.get(w.stop, w.stop)})"
        for w in sorted(rules.chips, key=lambda w: (w.start, w.chip_id))
        if w.chip_id not in used and w.stop >= context.gw_index
    ]
    objectives = dict(options.plans.scenario_objectives)
    base = objectives.get("none", 0.0)
    best = options.plans.best
    horizon = len(best.gws)
    last = best.gws[-1].gw
    lines = [
        f"<b>GW{context.gw} chips</b> · information only",
        (
            "Tested on 2023-25: this planner beat no chips, but not simple rules convincingly."
            if params.chip_value_left is not None
            else "The chip planner isn't tested yet: weigh this against your own judgement."
        ),
        f"Unused chips: {_e(', '.join(unused) or 'none')}",
        "",
        f"<b>Best over the next {horizon} GWs:</b> {_e(scenario_label(best.scenario))}"
        + (
            ""
            if best.scenario == "none"
            else f", {best.total_objective - base:+.1f} xP vs holding every chip"
        ),
    ]
    if best.scenario != "none" and best.first.chip is not None:
        names = context.names
        lines.append(
            f"This GW with it: {_moves(best.first, names, rules.hit_cost)}; captain "
            f"{_e(names[best.first.captain])}"
        )
    if options.singles:
        lines += ["", "Each chip on its own, in its most promising GW (xP vs no chip):"]
        for name in CHIP_LABELS:
            if name in options.singles:
                label, value = options.singles[name]
                lines.append(f"· {_e(scenario_label(label))}: {value - base:+.1f}")
    ranked_scenarios = sorted(objectives.items(), key=lambda kv: -kv[1])
    lines += ["", "Options the solver checked (xP vs no chip):"]
    for label, value in ranked_scenarios[:shown]:
        lines.append(f"· {_e(scenario_label(label))}: {value - base:+.1f}")
    solved = set(objectives) | {label for label, _ in options.singles.values()}
    unsolved = set(options.plans.scenario_bounds) - solved
    search = options.plans.search
    stopped = unsolved & {label for label, _ in (search.not_optimal if search else ())}
    skipped = len(unsolved - stopped)
    if len(ranked_scenarios) > shown:
        lines.append(f"· … {len(ranked_scenarios) - shown} more, all lower")
    if skipped > 0:
        lines.append(f"{skipped} other options were ruled out: provably no better than the best.")
    if stopped:
        lines.append(
            f"{len(stopped)} option(s) not settled: the solver hit its time limit before "
            "finding a plan for them."
        )
    if params.chip_value_left is not None:
        from fplopt.optimize.chips import chip_value

        last_index = context.gw_index_map.get(last, context.gw_index + len(best.gws) - 1)
        held = []
        for w in sorted(rules.chips, key=lambda w: (w.start, w.chip_id)):
            if w.chip_id in used or w.stop <= last_index or w.start > last_index + 1:
                continue
            left = w.stop - last_index
            value = chip_value(params, w.name, left)
            held.append(f"{CHIP_LABELS.get(w.name, w.name)} {value:.1f}")
        how = (
            f"How to read it: a chip still held after GW{last} counts as worth its estimated "
            f"best use in the rest of its window ({_e(', '.join(held) or 'none')} xP, from "
            "backtests); a chip whose window ends sooner is worth nothing if held. So a chip "
            "wins only if it beats holding it. The bot's plans don't play chips."
        )
    else:
        values = ", ".join(
            f"{CHIP_LABELS[name]} {params.chip_value.get(name, 0.0):g}"
            for name in CHIP_LABELS
            if name in params.chip_value
        )
        how = (
            f"How to read it: a chip still held after GW{last} counts as worth a guessed "
            f"value ({values} xP); a chip whose window ends sooner is worth nothing if held. "
            "So a chip wins only if it beats that guess. The bot's plans don't play chips."
        )
    lines += ["", how]
    return "\n".join(lines)


def split_message(text: str, limit: int = MAX_MESSAGE) -> list[str]:
    """`text` in pieces of at most `limit` characters, cut at line breaks (a single longer
    line is cut hard)."""
    pieces: list[str] = []
    current = ""
    for line in text.split("\n"):
        while len(line) > limit:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(line[:limit])
            line = line[limit:]
        candidate = line if not current else f"{current}\n{line}"
        if len(candidate) > limit:
            pieces.append(current)
            current = line
        else:
            current = candidate
    if current or not pieces:
        pieces.append(current)
    return pieces
