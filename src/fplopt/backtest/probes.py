"""Decision probes for the corrupt-the-future check (PLAN §4; Phase 3 plan, Task 5).

A probe is a builder like a feature: one `AsOfView` in, a deterministic DataFrame out. It
generates a start state at the view's deadline, runs a policy on it (with the season's
`backtest_rules` and the policy's xP model) and encodes the start state and the decision,
so a policy or start state that peeks past the deadline changes the frame and is caught by
`fplopt.features.leakcheck` (registered there as 'probe:<name>').

Frame (`PROBE_DTYPES`), one row per item, in this order:
- `held` (slot = holding index, player_key, value = purchase price), `bank`,
  `free_transfers` (value): the start state;
- `out` / `in` (slot = transfer index, player_key): the transfers;
- `starter` / `bench` (slot = position in the lineup, player_key), `captain`, `vice`;
- `chip` (note = chip name, '' for none).
`player_key` is -1 and `value` 0 where they don't apply. If the start state is refused
(`StartStateError`) the frame is a single `refused` row with the reason in `note`, so the
clean and altered runs still compare. Holdout seasons are refused with an error (the check
never samples their deadlines).
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from fplopt.backtest.policies import (
    DecisionContext,
    GreedyPolicy,
    OptimizerPolicy,
    Policy,
    RollPolicy,
)
from fplopt.backtest.rules import Rules, backtest_rules
from fplopt.backtest.start_states import (
    StartStateError,
    random_state,
    target_gameweek,
    template_state,
)
from fplopt.backtest.state import Decision, SquadState, apply_decision
from fplopt.features.baseline import player_pool
from fplopt.features.store import AsOfView
from fplopt.models import MODELS
from fplopt.optimize import OptimizerParams
from fplopt.seasons import HOLDOUT_SEASONS, season_label

PROBE_DTYPES = (
    ("kind", "str"),
    ("slot", "int64"),
    ("player_key", "int64"),
    ("value", "int64"),
    ("note", "str"),
)
StartState = Callable[[AsOfView, Rules], SquadState]


def encode(state: SquadState, decision: Decision) -> pd.DataFrame:
    """The start state and decision as a PROBE_DTYPES frame (see the module docstring)."""
    rows: list[tuple[str, int, int, int, str]] = []
    rows += [("held", i, h.player_key, h.purchase_price, "") for i, h in enumerate(state.holdings)]
    rows += [("bank", 0, -1, state.bank, ""), ("free_transfers", 0, -1, state.free_transfers, "")]
    for i, t in enumerate(decision.transfers):
        rows += [("out", i, t.out_key, 0, ""), ("in", i, t.in_key, 0, "")]
    lineup = decision.lineup
    rows += [("starter", i, key, 0, "") for i, key in enumerate(lineup.starters)]
    rows += [("bench", i, key, 0, "") for i, key in enumerate(lineup.bench)]
    rows += [("captain", 0, lineup.captain, 0, ""), ("vice", 0, lineup.vice, 0, "")]
    rows.append(("chip", 0, -1, 0, decision.chip or ""))
    return _frame(rows)


def _frame(rows: list[tuple[str, int, int, int, str]]) -> pd.DataFrame:
    columns = [name for name, _ in PROBE_DTYPES]
    return pd.DataFrame(rows, columns=columns).astype(dict(PROBE_DTYPES))


def refused(reason: str) -> pd.DataFrame:
    """The one-row frame of a probe whose start state was refused."""
    return _frame([("refused", 0, -1, 0, reason)])


def run_probe(view: AsOfView, policy: Policy, start: StartState) -> pd.DataFrame:
    """`policy`'s decision from `start(view, rules)` at the view's deadline, encoded; the
    decision is validated with `apply_decision` (an invalid one raises)."""
    season, _, _ = target_gameweek(view)
    if season in HOLDOUT_SEASONS:
        raise ValueError(f"decision probes do not run on the holdout season {season_label(season)}")
    rules = backtest_rules(season)
    try:
        state = start(view, rules)
    except StartStateError as exc:
        return refused(str(exc))
    pool = player_pool(view)
    xp = MODELS[policy.xp_model](view)
    decision = policy.decide(DecisionContext(view, state, rules, pool, xp))
    apply_decision(state, decision, pool, rules)
    return encode(state, decision)


def _random0(view: AsOfView, rules: Rules) -> SquadState:
    return random_state(view, rules, seed=0)


def greedy_rolling_random0(view: AsOfView) -> pd.DataFrame:
    """greedy(rolling) from random_state(seed=0)."""
    return run_probe(view, GreedyPolicy("rolling"), _random0)


def greedy_ep_next_template(view: AsOfView) -> pd.DataFrame:
    """greedy(ep_next) from template_state."""
    return run_probe(view, GreedyPolicy("ep_next"), template_state)


def roll_rolling_template(view: AsOfView) -> pd.DataFrame:
    """roll(rolling) from template_state."""
    return run_probe(view, RollPolicy("rolling"), template_state)


def optimizer_ep_next_template(view: AsOfView) -> pd.DataFrame:
    """optimizer(ep_next) from template_state, no chips, with a 3-GW horizon and a small
    candidate pool (8/20/20/10) to keep the leakage check fast."""
    params = OptimizerParams(horizon=3, prune_n={1: 8, 2: 20, 3: 20, 4: 10})
    return run_probe(view, OptimizerPolicy("ep_next", params), template_state)


PROBES: dict[str, Callable[[AsOfView], pd.DataFrame]] = {
    "greedy_rolling_random0": greedy_rolling_random0,
    "greedy_ep_next_template": greedy_ep_next_template,
    "roll_rolling_template": roll_rolling_template,
    "optimizer_ep_next_template": optimizer_ep_next_template,
}

__all__ = ("PROBES", "PROBE_DTYPES", "encode", "refused", "run_probe")
