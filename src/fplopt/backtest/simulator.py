"""The season simulator (PLAN §5 *Simulator*; Phase 3 plan, Task 6).

`simulate(store, rules, policy, start)` replays one season gameweek by gameweek from any
`SquadState`. For each GW (by `gw_index`, from the start state's to the season's last or
`end_gw_index`):

1. `view = store.as_of(deadline)`; `pool = player_pool(view)`; `state = refresh(state, pool)`;
2. `xp = MODELS[policy.xp_model](view)` (cached per (model, deadline));
3. `decision = policy.decide(DecisionContext(view, state, rules, pool, xp))`;
4. `apply_decision` validates it and gives the squad that plays the GW and its hits;
5. outcomes of (season, gw) from `store.as_of(lockdown + 1 µs)` (results are available at the
   lockdown and the view is strict): the GW's `player_match` rows joined to
   `player_season.element_type`, re-scored with `score_matches(rules)` and, as the secondary
   metric, `xg_score_matches` with `team_xg_table(team_match)` from the same view; summed per
   player (`gw_outcomes`); cached per (season, gw, rules.label);
6. `score_gameweek` twice: realized points, and xG points with the same minutes (so the
   same autosubs). A player-GW's xG points are null if any of its rows is; the GW's are null
   if any counted player's are (NaN propagates through the sum);
7. record the GW; `next_state` accrues FTs, records the chip, reverts a Free Hit.

After the run, every recorded GW with transfers gets its **predicted and realized transfer
gain** (PLAN §7 *Optimizer's curse*; `transfer_gains`): over the GW and the next
`GAIN_WINDOW − 1` recorded GWs (just the GW under a Free Hit, whose squad reverts),
`pred_gain` = Σ xP of the players bought − Σ xP of the players sold, from the decision-time
xP frame (`horizon` = gw_index offset), and `real_gain` = the same with their realized
points from the outcome views (players outside the squad included). Both are gross of
hits (`hit_points` is in the row): a slope of real on predicted below 1 means the planner
overrates its moves.

Policies only ever see the deadline view, and state transitions never use outcomes, so
reading outcomes cannot leak into decisions. The season's GW list comes from the `gameweek`
rows visible at the start deadline (schedule rows are available from 1 June).

Each GW is decided before its outcomes are read, so a decision never waits on (or sees)
them. A GW without any `player_match` row in its outcome view (not played yet, e.g. the live
season) ends the run with a warning: it is decided but not scored or recorded, and its
state and decision are kept as `SeasonRun.pending_state` / `pending_decision`.

The simulator is an orchestrator, not a feature builder: it holds the `DataStore` and reads
through `store.as_of(...)` only. `Caches` keep xP frames, pools and outcomes across runs on
one store (models and policies never cache).
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from fplopt.backtest.gw_score import GwScore, Lineup, Points, gw_outcomes, score_gameweek
from fplopt.backtest.policies import DecisionContext, Policy, best_lineup
from fplopt.backtest.rules import Rules
from fplopt.backtest.scoring import score_matches
from fplopt.backtest.state import (
    Decision,
    SquadState,
    TransferRecord,
    apply_decision,
    next_state,
    refresh,
    squad_value,
)
from fplopt.backtest.xg_points import team_xg_table, xg_score_matches
from fplopt.features.baseline import player_pool
from fplopt.features.store import AsOfView, DataStore
from fplopt.models import MODELS
from fplopt.models.fitted import FittedModel
from fplopt.seasons import HOLDOUT_SEASONS, season_label

log = logging.getLogger(__name__)

__all__ = (
    "GW_COLUMNS",
    "Caches",
    "GwOutcomes",
    "HoldoutError",
    "SeasonRun",
    "Step",
    "decide_step",
    "hindsight_points",
    "play_step",
    "read_outcomes",
    "run_gameweeks",
    "season_schedule",
    "simulate",
    "transfer_gains",
)

OUTCOME_DELAY = pd.Timedelta(microseconds=1)
# A view late enough to see every season's schedule rows; only used to find the start
# deadline, which is then checked against the view at that deadline.
FAR_FUTURE = pd.Timestamp("2200-01-01", tz="UTC")
MATCH_COLUMNS = (
    "player_key",
    "season",
    "gw",
    "fixture_key",
    "opponent_team_key",
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "defensive_contribution",
    "clearances_blocks_interceptions",
    "tackles",
    "recoveries",
    "us_xg",
    "us_xa",
    "fpl_xg",
    "fpl_xa",
)
TEAM_COLUMNS = ("fixture_key", "team_key", "season", "us_xg", "fpl_xg", "fd_xg")
GW_COLUMNS = (
    ("season", "int64"),
    ("gw", "int64"),
    ("gw_index", "int64"),
    ("deadline", pd.DatetimeTZDtype("us", "UTC")),
    ("chip", "str"),
    ("n_transfers", "int64"),
    ("hits", "int64"),
    ("hit_points", "int64"),
    ("points", "int64"),
    ("net_points", "int64"),
    ("bench_points", "int64"),
    ("xg_points", "Float64"),
    ("xg_net_points", "Float64"),
    ("captain", "int64"),
    ("vice", "int64"),
    ("captain_used", "Int64"),
    ("captain_points", "int64"),
    ("captain_regret", "int64"),
    ("xi_regret", "Int64"),
    ("bank", "int64"),
    ("free_transfers", "int64"),
    ("squad_value", "int64"),
    ("transfers", "str"),
    ("autosubs", "str"),
    ("pred_gain", "Float64"),
    ("real_gain", "Float64"),
    ("solver_status", "str"),
    ("mip_gap", "Float64"),
)
GAIN_WINDOW = 4  # GWs over which transfer gains are measured (PLAN §5's per-decision k)


class HoldoutError(ValueError):
    """A holdout season (`HOLDOUT_SEASONS`) was simulated without `allow_holdout`."""


@dataclass(frozen=True)
class GwOutcomes:
    """One GW's outcomes per player_key, summed over its fixtures: realized `(points,
    minutes)` (int) and xG-scored `(points, minutes)` (float, NaN when not computable).
    Players without a row (blank, or not in the game) are absent: 0 points, 0 minutes."""

    realized: dict[int, tuple[int, int]]
    xg: dict[int, tuple[float, int]]

    @property
    def empty(self) -> bool:
        return not self.realized


class Caches:
    """xP frames per (model, deadline), walk-forward fits per (model, cutoff) (`FittedModel`:
    the frame is `predict(view, fit)` with the memoized fit, the same as calling the model),
    pools per deadline and outcomes per (season, gw, rules.label), for one `DataStore`
    (using them with another raises). Share one instance across policies and start states;
    the cached frames and fits must not be modified."""

    def __init__(self) -> None:
        self._store: DataStore | None = None
        self._xp: dict[tuple[str, pd.Timestamp], pd.DataFrame] = {}
        self._fits: dict[tuple[str, pd.Timestamp], Any] = {}
        self._pool: dict[pd.Timestamp, pd.DataFrame] = {}
        self._outcomes: dict[tuple[int, int, str], GwOutcomes] = {}
        self.timings: dict[str, float] = {"xp": 0.0, "fit": 0.0, "pool": 0.0, "outcomes": 0.0}

    def _check(self, store: DataStore) -> None:
        if self._store is None:
            self._store = store
        elif self._store is not store:
            raise ValueError("Caches belong to another DataStore")

    def pool(self, store: DataStore, view: AsOfView) -> pd.DataFrame:
        """`player_pool(view)`; `view` must come from `store`."""
        self._check(store)
        if view.deadline not in self._pool:
            start = time.perf_counter()
            self._pool[view.deadline] = player_pool(view)
            self.timings["pool"] += time.perf_counter() - start
        return self._pool[view.deadline]

    def xp(self, store: DataStore, model: str, view: AsOfView) -> pd.DataFrame:
        """`MODELS[model](view)`; `view` must come from `store`."""
        self._check(store)
        if model not in MODELS:
            raise ValueError(f"unknown xp_model {model!r} (MODELS: {sorted(MODELS)})")
        key = (model, view.deadline)
        if key not in self._xp:
            builder = MODELS[model]
            if isinstance(builder, FittedModel):
                fitted = self.fit(store, model, view)
                start = time.perf_counter()
                self._xp[key] = builder.predict(view, fitted)
            else:
                start = time.perf_counter()
                self._xp[key] = builder(view)
            self.timings["xp"] += time.perf_counter() - start
        return self._xp[key]

    def fit(self, store: DataStore, model: str, view: AsOfView) -> Any:
        """The `FittedModel` `model`'s fit for `view`'s deadline: `fit(view.earlier(cutoff))`,
        memoized by cutoff; `view` must come from `store`."""
        self._check(store)
        builder = MODELS[model]
        if not isinstance(builder, FittedModel):
            raise ValueError(f"xp_model {model!r} is not fitted walk-forward")
        cutoff = builder.cutoff(view)
        key = (model, cutoff)
        if key not in self._fits:
            start = time.perf_counter()
            self._fits[key] = builder.fit(view.earlier(cutoff))
            self.timings["fit"] += time.perf_counter() - start
        return self._fits[key]

    def outcomes(
        self, store: DataStore, rules: Rules, season: int, gw: int, lockdown: pd.Timestamp
    ) -> GwOutcomes:
        self._check(store)
        key = (season, gw, rules.label)
        if key not in self._outcomes:
            start = time.perf_counter()
            self._outcomes[key] = read_outcomes(store, rules, season, gw, lockdown)
            self.timings["outcomes"] += time.perf_counter() - start
        return self._outcomes[key]


def read_outcomes(
    store: DataStore, rules: Rules, season: int, gw: int, lockdown: pd.Timestamp
) -> GwOutcomes:
    """Outcomes of (season, gw) from `store.as_of(lockdown + 1 µs)` (module docstring)."""
    view = store.as_of(pd.Timestamp(lockdown) + OUTCOME_DELAY)
    matches = view.table("player_match", columns=list(MATCH_COLUMNS))
    matches = matches[(matches["season"] == season) & (matches["gw"] == gw)]
    if matches.empty:
        return GwOutcomes({}, {})
    positions = view.table("player_season", columns=["player_key", "season", "element_type"])
    positions = positions[positions["season"] == season].drop_duplicates("player_key", keep="last")
    matches = matches.merge(
        positions[["player_key", "element_type"]], on="player_key", how="left", validate="m:1"
    )
    unknown = matches["element_type"].isna()
    if unknown.any():
        log.warning(
            "%s GW%d: %d player_match row(s) without a player_season position are not scored",
            season_label(season),
            gw,
            int(unknown.sum()),
        )
        matches = matches[~unknown]
    matches = matches.astype({"element_type": "int64"}).reset_index(drop=True)
    realized = score_matches(matches, rules)
    teams = view.table("team_match", columns=list(TEAM_COLUMNS))
    teams = teams[teams["fixture_key"].isin(matches["fixture_key"])]
    teams = teams.drop_duplicates(["fixture_key", "team_key"], keep="last")
    xg = xg_score_matches(matches, team_xg_table(teams.reset_index(drop=True)), rules)
    scored = pd.DataFrame(
        {
            "player_key": matches["player_key"].to_numpy(dtype="int64"),
            "minutes": matches["minutes"].to_numpy(dtype="int64"),
            "points": realized["points"].to_numpy(dtype="int64"),
            "xg_points": xg["xg_points"].to_numpy(dtype="float64", na_value=math.nan),
        }
    )
    xg_null = scored["xg_points"].isna().groupby(scored["player_key"]).any()
    xg_sum = gw_outcomes(scored, points="xg_points")
    xg_out = {
        key: ((math.nan if xg_null[key] else float(pts)), mins)
        for key, (pts, mins) in xg_sum.items()
    }
    return GwOutcomes(gw_outcomes(scored), xg_out)


# --- schedule -------------------------------------------------------------------------------


def season_schedule(store: DataStore, season: int, gw_index: int) -> pd.DataFrame:
    """The season's gameweeks from `gw_index` on (`season, gw, gw_index, deadline_time,
    lockdown_time`, sorted by gw_index), as visible at the deadline of `gw_index`."""
    columns = ["season", "gw", "gw_index", "deadline_time", "lockdown_time"]

    def rows(view: AsOfView) -> pd.DataFrame:
        gameweeks = view.table("gameweek", columns=columns)
        return gameweeks[(gameweeks["season"] == season) & (gameweeks["gw_index"] >= gw_index)]

    # Candidate start deadlines from every row ever listed; the start deadline is the one
    # whose own view lists exactly one row for gw_index, with that deadline.
    probe = rows(store.as_of(FAR_FUTURE))
    candidates = probe.loc[probe["gw_index"] == gw_index, "deadline_time"].dropna().unique()
    found = []
    for deadline in sorted(candidates):
        gameweeks = rows(store.as_of(deadline))
        start = gameweeks[gameweeks["gw_index"] == gw_index]
        if len(start) == 1 and start["deadline_time"].iloc[0] == deadline:
            found.append(gameweeks)
    if len(found) != 1:
        raise ValueError(
            f"{season_label(season)}: {len(found)} gameweeks with gw_index {gw_index} "
            "visible at their own deadline"
        )
    gameweeks = found[0].sort_values("gw_index", kind="mergesort").reset_index(drop=True)
    if not gameweeks["gw_index"].is_unique:
        raise ValueError(f"{season_label(season)}: duplicate gw_index in the gameweek table")
    return gameweeks


# --- one gameweek -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    """A GW's decision: the deadline view, pool, refreshed pre-decision state, xP frame and
    the policy's decision."""

    view: AsOfView
    pool: pd.DataFrame
    state: SquadState
    xp: pd.DataFrame
    decision: Decision


def decide_step(
    store: DataStore,
    rules: Rules,
    policy: Policy,
    state: SquadState,
    deadline: pd.Timestamp,
    caches: Caches,
) -> Step:
    """Steps 1-3 of the module docstring for one GW: only the deadline view is read."""
    view = store.as_of(deadline)
    pool = caches.pool(store, view)
    state = refresh(state, pool)
    xp = caches.xp(store, policy.xp_model, view)
    decision = policy.decide(DecisionContext(view, state, rules, pool, xp))
    return Step(view, pool, state, xp, decision)


def hindsight_points(
    squad: pd.DataFrame, outcomes: dict[int, tuple[int, int]], rules: Rules
) -> int:
    """The squad's best GW score in hindsight: the best valid XI on realized points
    (`best_lineup` with the points as xP) plus the best captain among its starters who
    played (no autosubs, no chip). `xi_regret` = this − the actual points."""
    points = {int(k): outcomes.get(int(k), (0, 0))[0] for k in squad["player_key"]}
    lineup = best_lineup(squad, points, rules)
    played = [points[k] for k in lineup.starters if outcomes.get(k, (0, 0))[1] > 0]
    return sum(points[k] for k in lineup.starters) + (max(played) if played else 0)


def _nan_to_none(value: Points) -> float | None:
    return None if isinstance(value, float) and math.isnan(value) else float(value)


def play_step(
    step: Step,
    rules: Rules,
    outcomes: GwOutcomes,
    gw: int,
    deadline: pd.Timestamp,
) -> tuple[SquadState, TransferRecord, dict[str, Any]]:
    """Steps 4-6: apply the decision, score it, and return (gw_state, record, GW row)."""
    state, decision = step.state, step.decision
    gw_state, record = apply_decision(state, decision, step.pool, rules)
    positions = {h.player_key: h.element_type for h in gw_state.holdings}
    lineup: Lineup = decision.lineup
    score: GwScore = score_gameweek(lineup, outcomes.realized, positions, rules, decision.chip)
    xg_score = score_gameweek(lineup, outcomes.xg, positions, rules, decision.chip)
    xg_points = _nan_to_none(xg_score.points)

    def raw(key: int | None) -> int:
        return 0 if key is None else outcomes.realized.get(key, (0, 0))[0]

    captain_points = raw(score.captain_used)
    best_counted = max(raw(k) for k in score.counted)
    xi_regret = None
    if decision.chip is None:
        squad = pd.DataFrame(
            {"player_key": list(positions), "element_type": list(positions.values())},
            dtype="int64",
        )
        xi_regret = hindsight_points(squad, outcomes.realized, rules) - int(score.points)
    row = {
        "season": state.season,
        "gw": gw,
        "gw_index": state.gw_index,
        "deadline": deadline,
        "chip": decision.chip,
        "n_transfers": record.n_transfers,
        "hits": record.hits,
        "hit_points": record.hit_points,
        "points": int(score.points),
        "net_points": int(score.points) - record.hit_points,
        "bench_points": int(score.bench_points),
        "xg_points": xg_points,
        "xg_net_points": None if xg_points is None else xg_points - record.hit_points,
        "captain": lineup.captain,
        "vice": lineup.vice,
        "captain_used": score.captain_used,
        "captain_points": captain_points,
        "captain_regret": best_counted - captain_points,
        "xi_regret": xi_regret,
        "bank": gw_state.bank,
        "free_transfers": state.free_transfers,
        "squad_value": squad_value(gw_state, rules.sell_on_fee),
        "transfers": json.dumps([[t.out_key, t.in_key] for t in decision.transfers]),
        "autosubs": json.dumps([list(pair) for pair in score.autosubs]),
        "solver_status": decision.solver_status,
        "mip_gap": decision.mip_gap,
    }
    return gw_state, record, row


def transfer_gains(
    store: DataStore,
    rules: Rules,
    xp_model: str,
    decisions: Sequence[Decision],
    schedule: pd.DataFrame,
    caches: Caches,
    window: int = GAIN_WINDOW,
) -> list[tuple[float | None, float | None]]:
    """Per decision i (decided at `schedule` row i; the decisions of recorded GWs, in
    order): (predicted, realized) gain of its transfers over schedule rows i..i+window−1
    that were recorded (a Free Hit: row i only), or (None, None) without transfers. See
    the module docstring. xP comes from `caches.xp` at the decision's deadline (the frame
    the policy saw), points from `caches.outcomes`."""
    gameweeks = list(
        schedule[["gw", "gw_index", "deadline_time", "lockdown_time"]].itertuples(index=False)
    )[: len(decisions)]
    season = int(schedule["season"].iloc[0]) if len(schedule) else 0
    out: list[tuple[float | None, float | None]] = []
    for i, decision in enumerate(decisions):
        if not decision.transfers:
            out.append((None, None))
            continue
        span = 1 if decision.chip == "freehit" else window
        gws = gameweeks[i : i + span]
        ins = [t.in_key for t in decision.transfers]
        outs = [t.out_key for t in decision.transfers]
        xp = caches.xp(store, xp_model, store.as_of(gws[0].deadline_time))
        offsets = [int(g.gw_index) - int(gws[0].gw_index) for g in gws]
        rows = xp[xp["horizon"].isin(offsets)]
        by_key = rows.groupby("player_key")["xp"].sum()
        predicted = float(by_key.reindex(ins).fillna(0.0).sum())
        predicted -= float(by_key.reindex(outs).fillna(0.0).sum())
        realized = 0
        for g in gws:
            outcomes = caches.outcomes(store, rules, season, int(g.gw), g.lockdown_time)
            points = outcomes.realized
            realized += sum(points.get(k, (0, 0))[0] for k in ins)
            realized -= sum(points.get(k, (0, 0))[0] for k in outs)
        out.append((predicted, float(realized)))
    return out


def gw_frame(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """GW rows as a frame with `GW_COLUMNS` dtypes."""
    columns = [name for name, _ in GW_COLUMNS]
    frame = pd.DataFrame(list(rows), columns=columns)
    return frame.astype(dict(GW_COLUMNS))


# --- seasons ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SeasonRun:
    """One simulated season. `gws`: one row per GW (`GW_COLUMNS`; `points` include the
    captain, `net_points = points − hit_points`, `bank`/`squad_value` after the decision,
    `free_transfers` before it, `xi_regret` null in chip GWs, `xg_points` null where not
    computable, `solver_status`/`mip_gap` how an optimizer policy's solve finished, null for
    other policies). `states[i]` is the refreshed state the policy saw at GW i, `decisions[i]` its
    decision; `final_state` the state after the last GW (FH reverted). `total` = Σ
    net_points. If the run stopped at a GW without outcomes, `pending_state` /
    `pending_decision` are that GW's refreshed state and decision (not scored)."""

    policy: str
    gws: pd.DataFrame
    decisions: tuple[Decision, ...]
    states: tuple[SquadState, ...]
    final_state: SquadState
    total: int
    pending_state: SquadState | None = None
    pending_decision: Decision | None = None
    timings: dict[str, float] = field(default_factory=dict)


PolicyFor = Callable[[int], Policy]


def run_gameweeks(
    store: DataStore,
    rules: Rules,
    policy_for: PolicyFor,
    start: SquadState,
    schedule: pd.DataFrame,
    caches: Caches,
) -> tuple[list[dict[str, Any]], list[Decision], list[SquadState], SquadState, Step | None]:
    """Simulate the GWs of `schedule` (consecutive rows starting at `start.gw_index`);
    `policy_for(i)` decides the i-th GW. Each GW is decided before its outcomes are read; a
    GW without outcomes stops the run unrecorded. Returns (rows, decisions, states, final
    state, the unscored step of the GW it stopped at or None)."""
    state = start
    pending: Step | None = None
    rows: list[dict[str, Any]] = []
    decisions: list[Decision] = []
    states: list[SquadState] = []
    gameweeks = list(
        schedule[["gw", "gw_index", "deadline_time", "lockdown_time"]].itertuples(index=False)
    )
    for i, gameweek in enumerate(gameweeks):
        gw, gw_index = int(gameweek.gw), int(gameweek.gw_index)
        if state.gw_index != gw_index:
            raise ValueError(f"state is at gw_index {state.gw_index}, schedule at {gw_index}")
        step = decide_step(store, rules, policy_for(i), state, gameweek.deadline_time, caches)
        outcomes = caches.outcomes(store, rules, state.season, gw, gameweek.lockdown_time)
        if outcomes.empty:
            log.warning(
                "%s GW%d: no outcomes as of its lockdown (not played yet?); decided, not "
                "scored; stopping",
                season_label(state.season),
                gw,
            )
            pending = step
            break
        gw_state, record, row = play_step(step, rules, outcomes, gw, gameweek.deadline_time)
        log.debug(
            "%s GW%d (gw_index %d): %d pts, %d transfer(s), chip %s, xg %s",
            season_label(state.season),
            gw,
            gw_index,
            row["net_points"],
            row["n_transfers"],
            row["chip"],
            row["xg_points"],
        )
        rows.append(row)
        decisions.append(step.decision)
        states.append(step.state)
        following = int(gameweeks[i + 1].gw_index) if i + 1 < len(gameweeks) else gw_index + 1
        state = next_state(gw_state, record, rules, following)
    return rows, decisions, states, state, pending


def simulate(
    store: DataStore,
    rules: Rules,
    policy: Policy,
    start: SquadState,
    *,
    end_gw_index: int | None = None,
    caches: Caches | None = None,
    allow_holdout: bool = False,
) -> SeasonRun:
    """Replay `start.season` from `start` with `policy` (module docstring) through the
    season's last GW, or through `end_gw_index` (inclusive). Raises HoldoutError for a
    season in HOLDOUT_SEASONS unless `allow_holdout` (Phase 6 only)."""
    season = start.season
    if season in HOLDOUT_SEASONS and not allow_holdout:
        raise HoldoutError(
            f"{season_label(season)} is a holdout season; simulate refuses it until Phase 6"
        )
    if start.freehit_backup is not None:
        raise ValueError("start state is a Free Hit gameweek state")
    caches = Caches() if caches is None else caches
    began = time.perf_counter()
    schedule = season_schedule(store, season, start.gw_index)
    if end_gw_index is not None:
        schedule = schedule[schedule["gw_index"] <= end_gw_index]
    rows, decisions, states, final, pending = run_gameweeks(
        store, rules, lambda _: policy, start, schedule, caches
    )
    gains = transfer_gains(store, rules, policy.xp_model, decisions, schedule, caches)
    for row, (predicted, realized) in zip(rows, gains, strict=True):
        row["pred_gain"], row["real_gain"] = predicted, realized
    gws = gw_frame(rows)
    total = int(gws["net_points"].sum())
    elapsed = time.perf_counter() - began
    log.info(
        "%s %s from gw_index %d: %d GWs, %d points (%d transfers, %d hit points) in %.1fs",
        season_label(season),
        policy.name,
        start.gw_index,
        len(gws),
        total,
        int(gws["n_transfers"].sum()),
        int(gws["hit_points"].sum()),
        elapsed,
    )
    return SeasonRun(
        policy=policy.name,
        gws=gws,
        decisions=tuple(decisions),
        states=tuple(states),
        final_state=final,
        total=total,
        pending_state=None if pending is None else pending.state,
        pending_decision=None if pending is None else pending.decision,
        timings={"elapsed": elapsed},
    )
