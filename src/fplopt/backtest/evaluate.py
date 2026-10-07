"""Paired policy evaluation (PLAN §5 *Comparing policies*; Phase 3 plan, Task 8).

Season totals are too noisy to compare policies directly, so every comparison is paired:

- `run_grid`: every policy from the same start states (`parse_start_specs`), one
  `SeasonRun` each, as one long frame (one row per season, start, policy and GW), sharing
  one `Caches` (xP per (model, deadline), outcomes per (season, gw, rules)).
- `paired_full_run`: per (season, start, GW) the difference of two policies' net points
  (realized and xG-scored).
- `per_decision`: states come from a reference policy's own trajectory. At a decision GW t
  both arms start from that state; arm A plays A's decision at t, arm B plays B's, and both
  then follow the same continuation policy (same xP) for k − 1 GWs (truncated at the
  season's end). Each arm's score is its net points over t..t+k−1 (hits at t included).
  Decision GWs are the reference run's GWs on the season's grid gw_index ≡ 1 (mod
  `stride`), default stride = k, so the windows don't overlap and tile the season: windows
  sharing outcomes would make neighbouring differences dependent and the bootstrap too
  optimistic. The grid is the same for every start state of a season, so their windows
  coincide and average into one cell; a start off the grid (`template@2`, `random@20`)
  gets its first decision at the next grid GW. Stride 1 = every GW (a diagnostic).
- `block_bootstrap`: average the paired differences over start states per (season,
  gw_index), then resample circular moving blocks of those cells within each season, all
  seasons together; mean, percentile CI (default 80% two-sided, i.e. one-sided α = 0.10,
  the go-live gate's level) and one-sided p (share of bootstrap means ≤ 0; H1: A > B).
- `summarize`: season totals per policy (mean over starts) and the paired differences with
  CIs: realized, realized on the xG metric's sample (`realized@xg`) and xG-scored. Block
  lengths are in GWs: a per-decision cell spans `stride` GWs, so its blocks are
  ceil(block_length / stride) cells (4 GWs = one k = 4 window). `log_experiment` appends a
  run to `results/experiments.csv`.

Start specs: `template@1` (the template squad at gw_index 1), `random:5@1` (random squads
with seeds 0-4 at gw_index 1), `random:3@20`. A refused start state is skipped with a
warning, except at gw_index 1, where it falls back to gw_index 2 (template squads before
2021/22 have no ownership at GW1; 2020/21 GW1 has a pool coverage gap); the start id records
the GW actually used (`template@2`, `random0@2`).
"""

from __future__ import annotations

import csv
import json
import logging
import re
import subprocess
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fplopt.backtest.policies import Policy
from fplopt.backtest.rules import Rules, backtest_rules
from fplopt.backtest.simulator import (
    GW_COLUMNS,
    Caches,
    HoldoutError,
    decide_step,
    run_gameweeks,
    season_schedule,
    simulate,
)
from fplopt.backtest.start_states import StartStateError, random_state, template_state
from fplopt.backtest.state import SquadState
from fplopt.features.store import DataStore
from fplopt.seasons import HOLDOUT_SEASONS, season_label

log = logging.getLogger(__name__)

__all__ = (
    "EXPERIMENT_COLUMNS",
    "BootstrapResult",
    "StartSpec",
    "Summary",
    "block_bootstrap",
    "build_start_states",
    "json_safe",
    "log_experiment",
    "paired_full_run",
    "parse_start_specs",
    "per_decision",
    "run_grid",
    "season_points",
    "summarize",
)

RulesFn = Callable[[int], Rules]
_SPEC = re.compile(r"(template|random(?::(\d+))?)@(\d+)")
EXPERIMENT_COLUMNS = ("timestamp", "git_sha", "command", "config", "metrics", "n_variants")


# --- start states ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StartSpec:
    """One start state: the template squad, or a random squad with `seed`, at `gw_index`."""

    kind: str
    gw_index: int
    seed: int | None = None

    def start_id(self, gw_index: int | None = None) -> str:
        gw = self.gw_index if gw_index is None else gw_index
        return f"template@{gw}" if self.kind == "template" else f"random{self.seed}@{gw}"

    def build(self, store: DataStore, season: int, gw_index: int, rules: Rules) -> SquadState:
        deadline = season_schedule(store, season, gw_index)["deadline_time"].iloc[0]
        view = store.as_of(deadline)
        if self.kind == "template":
            return template_state(view, rules)
        return random_state(view, rules, int(self.seed))


def parse_start_specs(text: str | Iterable[str]) -> tuple[StartSpec, ...]:
    """`"template@1,random:5@1,random:3@20"` -> the template at gw_index 1, random seeds
    0-4 at 1 and seeds 0-2 at 20 (`random@g` = one random squad, seed 0). Duplicates raise."""
    items = text.split(",") if isinstance(text, str) else list(text)
    specs: list[StartSpec] = []
    for item in (i.strip() for i in items):
        match = _SPEC.fullmatch(item)
        if not match:
            raise ValueError(
                f"bad start spec {item!r}: expected template@<gw>, random@<gw> or random:<n>@<gw>"
            )
        gw_index = int(match[3])
        if gw_index < 1:
            raise ValueError(f"bad start spec {item!r}: gw_index must be >= 1")
        if match[1] == "template":
            specs.append(StartSpec("template", gw_index))
        else:
            n = 1 if match[2] is None else int(match[2])
            if n < 1:
                raise ValueError(f"bad start spec {item!r}: need at least one seed")
            specs += [StartSpec("random", gw_index, seed) for seed in range(n)]
    ids = [spec.start_id() for spec in specs]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"duplicate start specs: {duplicates}")
    if not specs:
        raise ValueError("no start specs")
    return tuple(specs)


def _specs(start_specs: str | Sequence[StartSpec] | Sequence[str]) -> tuple[StartSpec, ...]:
    if isinstance(start_specs, str):
        return parse_start_specs(start_specs)
    if all(isinstance(spec, StartSpec) for spec in start_specs):
        return tuple(start_specs)
    return parse_start_specs([str(spec) for spec in start_specs])


def build_start_states(
    store: DataStore,
    season: int,
    start_specs: str | Sequence[StartSpec],
    rules: Rules,
) -> list[tuple[str, SquadState]]:
    """(start_id, state) per spec for `season`. Refused states are skipped with a warning;
    at gw_index 1 a refusal falls back to gw_index 2 (logged; the id records the GW used).
    A fallback that duplicates another spec's id is dropped."""
    out: dict[str, SquadState] = {}
    label = season_label(season)
    for spec in _specs(start_specs):
        try:
            state, gw_index = spec.build(store, season, spec.gw_index, rules), spec.gw_index
        except StartStateError as exc:
            if spec.gw_index != 1:
                log.warning("%s %s refused, skipped: %s", label, spec.start_id(), exc)
                continue
            log.info("%s %s refused (%s); starting at gw_index 2", label, spec.start_id(), exc)
            try:
                state, gw_index = spec.build(store, season, 2, rules), 2
            except StartStateError as exc2:
                log.warning("%s %s refused, skipped: %s", label, spec.start_id(2), exc2)
                continue
        start_id = spec.start_id(gw_index)
        if start_id in out:
            log.warning("%s %s: duplicate start id after fallback, skipped", label, start_id)
            continue
        out[start_id] = state
    return list(out.items())


# --- full runs --------------------------------------------------------------------------------


def _refuse_holdout(seasons: Sequence[int]) -> None:
    holdout = sorted(set(seasons) & HOLDOUT_SEASONS)
    if holdout:
        raise HoldoutError(
            f"{', '.join(map(season_label, holdout))} is a holdout season; backtests refuse "
            "it until Phase 6"
        )


def _unique_names(policies: Sequence[Policy]) -> None:
    names = [p.name for p in policies]
    if len(set(names)) != len(names):
        raise ValueError(f"policy names must be unique: {names}")


GRID_COLUMNS = ("season", "start_id", "policy", *(n for n, _ in GW_COLUMNS if n != "season"))


def run_grid(
    store: DataStore,
    seasons: Iterable[int],
    start_specs: str | Sequence[StartSpec],
    policies: Sequence[Policy],
    rules_fn: RulesFn = backtest_rules,
    *,
    caches: Caches | None = None,
) -> pd.DataFrame:
    """Every policy from every start state of every season (one `simulate` each), as one
    frame: `season, start_id, policy` + the `SeasonRun.gws` columns (net_points, points,
    hit_points, xg_points, xg_net_points, n_transfers, captain_regret, xi_regret, ...).
    One `Caches` is shared by all runs. Holdout seasons raise (`HoldoutError`) before
    anything runs."""
    seasons = list(seasons)
    _refuse_holdout(seasons)
    _unique_names(policies)
    specs = _specs(start_specs)
    caches = Caches() if caches is None else caches
    frames = []
    for season in seasons:
        rules = rules_fn(season)
        for start_id, state in build_start_states(store, season, specs, rules):
            for policy in policies:
                run = simulate(store, rules, policy, state, caches=caches)
                frames.append(run.gws.assign(start_id=start_id, policy=policy.name))
    if not frames:
        empty = pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in GW_COLUMNS})
        frames = [empty.assign(start_id=pd.Series(dtype="str"), policy=pd.Series(dtype="str"))]
    out = pd.concat(frames, ignore_index=True)
    out = out.astype({"start_id": "str", "policy": "str"})
    return out[list(GRID_COLUMNS)]


def _name(policy: Policy | str) -> str:
    return policy if isinstance(policy, str) else policy.name


def paired_full_run(results: pd.DataFrame, a: Policy | str, b: Policy | str) -> pd.DataFrame:
    """Per (season, start_id, gw_index) where both policies ran: `points_a/points_b` (net
    points), `diff = points_a − points_b`, `xg_a/xg_b` (xG net points) and `diff_xg` (null
    unless both are non-null)."""
    a, b = _name(a), _name(b)
    keys = ["season", "start_id", "gw", "gw_index"]
    columns = [*keys, "net_points", "xg_net_points"]
    sides = []
    for name, suffix in ((a, "a"), (b, "b")):
        side = results.loc[results["policy"] == name, columns]
        if side.empty:
            raise ValueError(f"no results for policy {name!r}")
        sides.append(
            side.rename(columns={"net_points": f"points_{suffix}", "xg_net_points": f"xg_{suffix}"})
        )
    out = sides[0].merge(sides[1], on=keys, how="inner", validate="1:1")
    out["diff"] = (out["points_a"] - out["points_b"]).astype("int64")
    out["diff_xg"] = (out["xg_a"] - out["xg_b"]).astype("Float64")
    out = out.sort_values(["season", "start_id", "gw_index"], kind="mergesort")
    columns = [*keys, "points_a", "points_b", "diff", "xg_a", "xg_b", "diff_xg"]
    return out[columns].reset_index(drop=True)


# --- per-decision -----------------------------------------------------------------------------

PER_DECISION_COLUMNS = (
    ("season", "int64"),
    ("start_id", "str"),
    ("policy_a", "str"),
    ("policy_b", "str"),
    ("gw", "int64"),
    ("gw_index", "int64"),
    ("k", "int64"),
    ("stride", "int64"),
    ("same_decision", "bool"),
    ("transfers_a", "int64"),
    ("transfers_b", "int64"),
    ("points_a", "int64"),
    ("points_b", "int64"),
    ("diff", "int64"),
    ("xg_a", "Float64"),
    ("xg_b", "Float64"),
    ("diff_xg", "Float64"),
)


def _window_scores(rows: Sequence[dict[str, Any]]) -> tuple[int, float | None]:
    """Σ net points and Σ xG net points (None if any GW's is null) over an arm's GWs."""
    points = sum(int(r["net_points"]) for r in rows)
    xg = [r["xg_net_points"] for r in rows]
    return points, (None if any(v is None for v in xg) else float(sum(xg)))


def per_decision(
    store: DataStore,
    seasons: Iterable[int],
    start_specs: str | Sequence[StartSpec],
    policy_a: Policy,
    policy_b: Policy,
    continuation: Policy,
    k: int = 4,
    *,
    stride: int | None = None,
    reference: str = "b",
    caches: Caches | None = None,
    rules_fn: RulesFn = backtest_rules,
) -> pd.DataFrame:
    """Per-decision paired differences (module docstring), one row per decision GW of the
    reference trajectory with gw_index ≡ 1 (mod `stride`), `stride` defaulting to `k`
    (non-overlapping windows on one grid per season, whatever the start; 1 = every GW).
    Columns: `k` = the window length actually scored (shorter at the season's end),
    `stride`, `same_decision` (A and B decided the same at t, so the arms are identical and
    diff = 0), each arm's transfers at t, its net points and xG net points over the window,
    and the differences A − B. Holdout seasons raise (`HoldoutError`) before anything
    runs."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    stride = k if stride is None else stride
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if reference not in ("a", "b"):
        raise ValueError(f"reference must be 'a' or 'b', got {reference!r}")
    seasons = list(seasons)
    _refuse_holdout(seasons)
    specs = _specs(start_specs)
    caches = Caches() if caches is None else caches
    ref_policy = policy_a if reference == "a" else policy_b
    rows: list[dict[str, Any]] = []
    for season in seasons:
        rules = rules_fn(season)
        for start_id, start in build_start_states(store, season, specs, rules):
            ref = simulate(store, rules, ref_policy, start, caches=caches)
            schedule = season_schedule(store, season, start.gw_index).iloc[: len(ref.gws)]
            first_row = len(rows)
            for i, gw_index in enumerate(schedule["gw_index"]):
                if (gw_index - 1) % stride:
                    continue  # not on the season's decision grid
                state = ref.states[i]
                window = schedule.iloc[i : i + k]
                deadline = window["deadline_time"].iloc[0]
                arms = {}
                for side, policy in (("a", policy_a), ("b", policy_b)):
                    arms[side] = decide_step(store, rules, policy, state, deadline, caches)
                same = arms["a"].decision == arms["b"].decision
                scores = {}
                for side, policy in (("a", policy_a), ("b", policy_b)):
                    if side == "b" and same:
                        scores["b"] = scores["a"]
                        continue
                    arm_rows, *_ = run_gameweeks(
                        store,
                        rules,
                        lambda j, first=policy: first if j == 0 else continuation,
                        state,
                        window,
                        caches,
                    )
                    if len(arm_rows) != len(window):
                        raise RuntimeError(f"{season_label(season)}: arm stopped early")
                    scores[side] = (_window_scores(arm_rows), arm_rows[0]["n_transfers"])
                (points_a, xg_a), n_a = scores["a"]
                (points_b, xg_b), n_b = scores["b"]
                rows.append(
                    {
                        "season": season,
                        "start_id": start_id,
                        "policy_a": policy_a.name,
                        "policy_b": policy_b.name,
                        "gw": int(window["gw"].iloc[0]),
                        "gw_index": int(window["gw_index"].iloc[0]),
                        "k": len(window),
                        "stride": stride,
                        "same_decision": same,
                        "transfers_a": n_a,
                        "transfers_b": n_b,
                        "points_a": points_a,
                        "points_b": points_b,
                        "diff": points_a - points_b,
                        "xg_a": xg_a,
                        "xg_b": xg_b,
                        "diff_xg": None if xg_a is None or xg_b is None else xg_a - xg_b,
                    }
                )
            log.info(
                "%s %s per-decision %s vs %s: %d decisions, %d differ",
                season_label(season),
                start_id,
                policy_a.name,
                policy_b.name,
                len(rows) - first_row,
                sum(1 for r in rows[first_row:] if not r["same_decision"]),
            )
    columns = [name for name, _ in PER_DECISION_COLUMNS]
    return pd.DataFrame(rows, columns=columns).astype(dict(PER_DECISION_COLUMNS))


# --- bootstrap --------------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    """The mean paired difference per (season, gw_index) cell — per GW for full-run
    differences, per decision window (up to k GWs) for per-decision ones — its percentile
    CI, the one-sided p-value (share of bootstrap means ≤ 0) and the sample: seasons and
    cells (`n_gws`: GWs for full runs, decision windows for per-decision)."""

    mean: float
    ci_low: float
    ci_high: float
    p_one_sided: float
    n_seasons: int
    n_gws: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "mean": self.mean,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_one_sided": self.p_one_sided,
            "n_seasons": self.n_seasons,
            "n_gws": self.n_gws,
        }


def block_bootstrap(
    diffs: pd.DataFrame,
    value: str = "diff",
    block_length: int = 4,
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.8,
) -> BootstrapResult:
    """GW-block bootstrap clustered by season of the mean of `value` (PLAN §5).

    Rows with a null `value` are dropped; the rest are averaged over start states per
    (season, gw_index), giving one series of cells per season ordered by gw_index (a cell is
    a GW for full-run differences, a decision window for per-decision ones). Each bootstrap
    replicate rebuilds every season's series from circular blocks of `block_length`
    consecutive cells of that season (blocks never cross seasons; a season keeps its length)
    and takes the mean over all seasons' cells. Returns the observed mean, the `ci`
    percentile interval of the replicate means (default 80% two-sided, whose lower bound is
    the one-sided α = 0.10 bound) and p = share of replicate means ≤ 0."""
    if block_length < 1 or n_boot < 1 or not 0 < ci < 1:
        raise ValueError("need block_length >= 1, n_boot >= 1 and 0 < ci < 1")
    data = diffs[["season", "gw_index", value]].copy()
    data[value] = pd.to_numeric(data[value]).astype("Float64")
    data = data[data[value].notna()]
    if data.empty:
        nan = float("nan")
        return BootstrapResult(nan, nan, nan, nan, 0, 0)
    cells = data.groupby(["season", "gw_index"], sort=True)[value].mean().astype("float64")
    rng = np.random.default_rng(seed)
    total = np.zeros(n_boot)
    seasons = cells.index.get_level_values("season")
    for season in sorted(set(seasons)):
        series = cells[seasons == season].to_numpy(dtype="float64")
        n = len(series)
        n_blocks = -(-n // block_length)
        starts = rng.integers(0, n, size=(n_boot, n_blocks))
        index = (starts[:, :, None] + np.arange(block_length)[None, None, :]) % n
        index = index.reshape(n_boot, -1)[:, :n]
        total += series[index].sum(axis=1)
    means = total / len(cells)
    alpha = (1 - ci) / 2
    low, high = np.quantile(means, [alpha, 1 - alpha])
    return BootstrapResult(
        mean=float(cells.mean()),
        ci_low=float(low),
        ci_high=float(high),
        p_one_sided=float(np.mean(means <= 0)),
        n_seasons=len(set(seasons)),
        n_gws=len(cells),
    )


# --- summaries -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Summary:
    """`season_totals`: per (season, policy) the means over start states of the season's
    net points (`total`), xG net points (`xg_total`, null if any start's run has a null GW),
    GWs played, transfers and hit points, and the mean captain/XI regret per GW.
    `policies`: the same averaged over seasons (`n_seasons`), except `xg_total`, averaged
    over the `xg_seasons` seasons where every policy's `xg_total` is non-null, so policies
    are compared on the same seasons. `comparisons`: per (a, b, method, metric) the
    bootstrap result (`block_bootstrap`; mean per GW for `full_run`, per decision window for
    `per_decision`) plus `season_diff`, the difference in season points (Σ over the season's
    cells of the start-averaged difference, mean over seasons; windows that overlap,
    stride < k, are weighted stride/k so each GW counts about once). Metrics: `realized`,
    `realized@xg` (realized on the rows whose xG difference is non-null, i.e. the xG
    metric's sample, for the same-sign check) and `xg`."""

    season_totals: pd.DataFrame
    policies: pd.DataFrame
    comparisons: pd.DataFrame

    def to_dict(self) -> dict[str, Any]:
        def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
            return json.loads(frame.to_json(orient="records"))

        return {
            "season_totals": records(self.season_totals),
            "policies": records(self.policies),
            "comparisons": records(self.comparisons),
        }


def _season_totals(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["season", "policy", "start_id"]
    runs = results.groupby(keys, sort=True).agg(
        total=("net_points", "sum"),
        xg_total=("xg_net_points", lambda s: s.sum() if s.notna().all() else np.nan),
        n_gws=("gw_index", "size"),
        n_transfers=("n_transfers", "sum"),
        hit_points=("hit_points", "sum"),
        captain_regret=("captain_regret", "mean"),
        xi_regret=("xi_regret", "mean"),
    )
    runs = runs.astype("float64").reset_index()
    out = runs.groupby(["season", "policy"], sort=True).agg(
        n_starts=("start_id", "size"),
        total=("total", "mean"),
        xg_total=("xg_total", lambda s: s.mean() if s.notna().all() else np.nan),
        n_gws=("n_gws", "mean"),
        n_transfers=("n_transfers", "mean"),
        hit_points=("hit_points", "mean"),
        captain_regret=("captain_regret", "mean"),
        xi_regret=("xi_regret", "mean"),
    )
    return out.reset_index()


def _stride(diffs: pd.DataFrame) -> int:
    """GWs between consecutive cells: 1 for GW rows (and per-decision frames without a
    `stride` column, which evaluated every GW), else the frame's single stride."""
    if "stride" not in diffs.columns or diffs.empty:
        return 1
    strides = diffs["stride"].unique()
    if len(strides) != 1:
        raise ValueError(f"per-decision frame mixes strides {sorted(strides)}")
    return int(strides[0])


def _season_weights(diffs: pd.DataFrame, stride: int) -> pd.Series:
    """Per row, the weight that turns its difference into season points: 1 for GW rows and
    non-overlapping windows, stride/k for windows that overlap (stride < k)."""
    if "k" not in diffs.columns:
        return pd.Series(1.0, index=diffs.index)
    return (stride / diffs["k"].astype("float64")).clip(upper=1.0)


def season_points(diffs: pd.DataFrame, column: str = "diff") -> pd.Series:
    """Per season, the difference in season points of a `paired_full_run` or `per_decision`
    frame: Σ over the season's (season, gw_index) cells of the start-averaged `column`
    (nulls dropped), with overlapping per-decision windows (stride < k) weighted stride/k
    so each GW counts about once."""
    valid = diffs[diffs[column].notna()]
    weighted = pd.to_numeric(valid[column]).astype("float64") * _season_weights(
        valid, _stride(diffs)
    )
    cells = weighted.groupby([valid["season"], valid["gw_index"]]).mean()
    return cells.groupby(level=0).sum().rename_axis("season")


def _comparison(
    diffs: pd.DataFrame,
    a: str,
    b: str,
    method: str,
    *,
    block_length: int,
    **bootstrap: Any,
) -> list[dict[str, Any]]:
    stride = _stride(diffs)
    cell_block = -(-block_length // stride)  # block_length is in GWs
    rows = []
    on_xg = diffs[diffs["diff_xg"].notna()]
    for metric, frame, column in (
        ("realized", diffs, "diff"),
        ("realized@xg", on_xg, "diff"),
        ("xg", diffs, "diff_xg"),
    ):
        result = block_bootstrap(frame, value=column, block_length=cell_block, **bootstrap)
        per_season = season_points(frame, column)
        season_diff = float(per_season.mean()) if len(per_season) else float("nan")
        rows.append(
            {"a": a, "b": b, "method": method, "metric": metric}
            | result.to_dict()
            | {"season_diff": season_diff}
        )
    return rows


def summarize(
    results: pd.DataFrame,
    pairs: Sequence[tuple[Policy | str, Policy | str]] = (),
    per_decision_diffs: pd.DataFrame | Sequence[pd.DataFrame] | None = None,
    *,
    block_length: int = 4,
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.8,
) -> Summary:
    """Season totals per policy (mean over starts) from `run_grid` results, and for each
    (a, b) in `pairs` the full-run paired difference, plus each `per_decision` frame's:
    realized, realized on the xG sample and xG-scored, with block-bootstrap CIs
    (`block_bootstrap` parameters, `block_length` in GWs; `Summary` describes the columns)."""
    bootstrap = {"block_length": block_length, "n_boot": n_boot, "seed": seed, "ci": ci}
    season_totals = _season_totals(results)
    policies = (
        season_totals.drop(columns=["season", "n_starts"])
        .groupby("policy", sort=True)
        .mean()
        .reset_index()
        .assign(n_seasons=season_totals.groupby("policy", sort=True)["season"].nunique().values)
    )
    xg = season_totals.pivot(index="season", columns="policy", values="xg_total")
    common = xg[xg.notna().all(axis=1)]
    policies["xg_total"] = policies["policy"].map(common.mean()).astype("float64")
    policies["xg_seasons"] = len(common)
    comparisons: list[dict[str, Any]] = []
    for a, b in pairs:
        a, b = _name(a), _name(b)
        comparisons += _comparison(paired_full_run(results, a, b), a, b, "full_run", **bootstrap)
    if per_decision_diffs is not None:
        frames = (
            [per_decision_diffs]
            if isinstance(per_decision_diffs, pd.DataFrame)
            else list(per_decision_diffs)
        )
        for frame in frames:
            if frame.empty:
                continue
            for (a, b), part in frame.groupby(["policy_a", "policy_b"], sort=True):
                comparisons += _comparison(part, a, b, "per_decision", **bootstrap)
    columns = ["a", "b", "method", "metric", "mean", "ci_low", "ci_high", "p_one_sided"]
    columns += ["n_seasons", "n_gws", "season_diff"]
    return Summary(season_totals, policies, pd.DataFrame(comparisons, columns=columns))


# --- experiment log ---------------------------------------------------------------------------


def git_sha() -> str:
    """`git rev-parse HEAD` of the code's checkout, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def json_safe(value: Any) -> Any:
    """`value` with numpy scalars as Python numbers and NaN/inf as None, recursively (for
    strict JSON)."""
    if isinstance(value, dict):
        return {key: json_safe(v) for key, v in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _dumps(value: Any) -> str:
    return json.dumps(json_safe(value), sort_keys=True, default=str, allow_nan=False)


def log_experiment(
    path: Path | str,
    command: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    n_variants: int,
) -> None:
    """Append one row (`EXPERIMENT_COLUMNS`: UTC timestamp, git sha, command, config and
    metrics as strict JSON (NaN as null, numpy scalars as numbers), n_variants) to the CSV
    at `path` with LF line endings, writing the header if the file is new or empty."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    row = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "command": command,
        "config": _dumps(config),
        "metrics": _dumps(metrics),
        "n_variants": int(n_variants),
    }
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EXPERIMENT_COLUMNS), lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow(row)
