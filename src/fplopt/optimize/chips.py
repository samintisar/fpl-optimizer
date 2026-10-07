"""Chip scenarios and the terminal value of unused chips (PLAN §7 *Chips*; Phase 4 plan,
Decisions and Task 2).

Chips are not free binaries in the MILP (reported to stall the solver): each allowed chip
assignment inside the horizon is a fixed `ChipScenario`, solved separately, and the best
objective wins (`plans.optimize`). The scenarios are

- no chip;
- each chip available at each horizon GW;
- pairs of chips in two different horizon GWs, both available (the second given the first
  is played).

Availability is `state.chip_window`, the rules engine's own check (`state.chip_available`):
the chip's window (on `gw_index`) is open and its `chip_id` unused, and a Free Hit doesn't
follow a Free Hit in the next GW unless `rules.freehit_consecutive`. Chips are counted per
`chip_id`, so a chip whose set-1 and set-2 windows both fall in the horizon can be played
twice (once per window), and two Free Hits in consecutive GWs across the set boundary are
allowed only with `freehit_consecutive`. At most one chip per GW (a `Decision` holds one).
Triples are not enumerated.

`terminal_value` credits each `chip_id` still unused at the horizon's end whose window
extends past the last horizon `gw_index` with `params.chip_value[name]` (a placeholder for
PLAN §7's "expected best use in the rest of the window, estimated from backtest
distributions"); a window closing inside the horizon is worth 0 (use it or lose it).

Scenario keys `t` are positions in `PlanInput.gws` (equal to the horizon offset unless the
xP frame skips a GW). Pure: no I/O.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass, replace

from fplopt.backtest.state import InvalidDecision, chip_window
from fplopt.optimize.params import OptimizerParams
from fplopt.optimize.problem import PlanInput


@dataclass(frozen=True)
class ChipScenario:
    """A fixed chip assignment: `chips` are `(t, name)` pairs sorted by `t` (position in
    the horizon), `chip_ids` the windows they use (aligned with `chips`), `label` a
    readable name (`none`, `bboost@gw7`, `wildcard@gw5+freehit@gw9`; GW numbers, not
    positions)."""

    chips: tuple[tuple[int, str], ...] = ()
    chip_ids: tuple[int, ...] = ()
    label: str = "none"

    @property
    def by_t(self) -> dict[int, str]:
        """Position in the horizon → chip name."""
        return dict(self.chips)

    def chip_at(self, t: int) -> str | None:
        return self.by_t.get(t)

    def without(self, t: int, problem: PlanInput) -> ChipScenario:
        """The scenario with the chip at position `t` removed (chip ids re-assigned)."""
        return make_scenario(problem, {s: name for s, name in self.chips if s != t})


NO_CHIP = ChipScenario()


def make_scenario(problem: PlanInput, chips: Mapping[int, str]) -> ChipScenario:
    """Validate a chip assignment (position in the horizon → chip name) against the rules,
    in GW order, and assign each chip its `chip_id`. Raises InvalidDecision if a chip can't
    be played where asked, ValueError for a position outside the horizon."""
    state, rules = problem.state, problem.rules
    used = list(state.chips_used)
    pairs, ids, parts = [], [], []
    for t in sorted(chips):
        if not 0 <= t < len(problem.gws):
            raise ValueError(f"chip position {t} outside the horizon (0..{len(problem.gws) - 1})")
        gw = problem.gws[t]
        name = chips[t]
        chip_id = chip_window(
            replace(state, gw_index=gw.gw_index, chips_used=tuple(used)), name, rules
        )
        used.append((chip_id, gw.gw_index))
        pairs.append((t, name))
        ids.append(chip_id)
        parts.append(f"{name}@gw{gw.gw}")
    return ChipScenario(tuple(pairs), tuple(ids), "+".join(parts) or "none")


def scenarios(problem: PlanInput, params: OptimizerParams) -> tuple[ChipScenario, ...]:
    """Every scenario the rules allow, in a deterministic order: no chip; singles by
    (position, chip name); pairs by (first position, first name, second position, second
    name). `params` is accepted for future scenario limits; unused."""
    del params
    names = sorted({c.name for c in problem.rules.chips})
    positions = range(len(problem.gws))

    def allowed(chips: Mapping[int, str]) -> ChipScenario | None:
        try:
            return make_scenario(problem, chips)
        except InvalidDecision:
            return None

    singles = [allowed({t: name}) for t in positions for name in names]
    singles = [s for s in singles if s is not None]
    pairs = []
    for s1, s2 in itertools.combinations(singles, 2):
        (t1, n1), (t2, n2) = s1.chips[0], s2.chips[0]
        if t1 != t2:  # singles are sorted by position, so t1 < t2
            pair = allowed({t1: n1, t2: n2})
            if pair is not None:
                pairs.append(pair)
    return (NO_CHIP, *singles, *pairs)


def terminal_value(problem: PlanInput, params: OptimizerParams, scenario: ChipScenario) -> float:
    """Σ `params.chip_value[name]` over the chip windows unused after the scenario whose
    window extends past the last horizon `gw_index` (0 for a window closing inside it, and
    for a chip name missing from `chip_value`). Added to the objective undecayed: it is
    the value of holding the chip at the horizon's end, in this-GW points."""
    used = {cid for cid, _ in problem.state.chips_used} | set(scenario.chip_ids)
    last = problem.gws[-1].gw_index
    return float(
        sum(
            params.chip_value.get(w.name, 0.0)
            for w in sorted(problem.rules.chips, key=lambda w: w.chip_id)
            if w.chip_id not in used and w.stop > last
        )
    )
