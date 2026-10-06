"""Paired policy evaluation (PLAN §5 *Comparing policies*; Phase 3 plan, Task 8).

Season totals are too noisy to compare policies directly, so every comparison is paired:

- `run_grid`: every policy from the same start states (`parse_start_specs`), one
  `SeasonRun` each, as one long frame (one row per season, start, policy and GW), sharing
  one `Caches` (xP per (model, deadline), outcomes per (season, gw, rules)).
- `paired_full_run`: per (season, start, GW) the difference of two policies' net points
  (realized and xG-scored).
- `per_decision`: states come from a reference policy's own trajectory. At each GW t both
  arms start from that state; arm A plays A's decision at t, arm B plays B's, and both then
  follow the same continuation policy (same xP) for k − 1 GWs (truncated at the season's
  end). Each arm's score is its net points over t..t+k−1 (hits at t included).
- `block_bootstrap`: average the paired differences over start states per (season,
  gw_index), then resample circular moving blocks of GWs within each season, all seasons
  together; mean, percentile CI and one-sided p (share of bootstrap means ≤ 0; H1: A > B).
- `summarize`: season totals per policy (mean over starts) and the paired differences with
  CIs, realized and xG-scored. `log_experiment` appends a run to `results/experiments.csv`.

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
    decide_step,
    run_gameweeks,
    season_schedule,
    simulate,
)
from fplopt.backtest.start_states import StartStateError, random_state, template_state
from fplopt.backtest.state import SquadState
from fplopt.features.store import DataStore
from fplopt.seasons import season_label

log = logging.getLogger(__name__)

__all__ = (
    "EXPERIMENT_COLUMNS",
    "BootstrapResult",
    "StartSpec",
    "Summary",
    "block_bootstrap",
    "build_start_states",
    "log_experiment",
    "paired_full_run",
    "parse_start_specs",
    "per_decision",
    "run_grid",
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
    One `Caches` is shared by all runs. Holdout seasons raise (`HoldoutError`)."""
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
    reference: str = "b",
    caches: Caches | None = None,
    rules_fn: RulesFn = backtest_rules,
) -> pd.DataFrame:
    """Per-decision paired differences (module docstring), one row per (season, start_id,
    gw_index) of the reference trajectory: `k` = the window length actually scored
    (shorter at the season's end), `same_decision` (A and B decided the same at t, so the
    arms are identical and diff = 0), each arm's transfers at t, its net points and xG net
    points over the window, and the differences A − B."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if reference not in ("a", "b"):
        raise ValueError(f"reference must be 'a' or 'b', got {reference!r}")
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
            for i, state in enumerate(ref.states):
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
                len(ref.states),
                sum(1 for r in rows[first_row:] if not r["same_decision"]),
            )
    columns = [name for name, _ in PER_DECISION_COLUMNS]
    return pd.DataFrame(rows, columns=columns).astype(dict(PER_DECISION_COLUMNS))


# --- bootstrap --------------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    """Mean paired difference per GW, its percentile CI, the one-sided p-value (share of
    bootstrap means ≤ 0) and the sample: seasons and (season, gw_index) cells."""

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
    ci: float = 0.9,
) -> BootstrapResult:
    """GW-block bootstrap clustered by season of the mean of `value` (PLAN §5).

    Rows with a null `value` are dropped; the rest are averaged over start states per
    (season, gw_index), giving one series per season ordered by gw_index. Each bootstrap
    replicate rebuilds every season's series from circular blocks of `block_length`
    consecutive GWs of that season (blocks never cross seasons; a season keeps its length)
    and takes the mean over all seasons' GWs. Returns the observed mean, the `ci` percentile
    interval of the replicate means and p = share of replicate means ≤ 0."""
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
    net points (`total`), xG net points (`xg_total`, null if any GW's is), GWs played,
    transfers and hit points, and the mean captain/XI regret per GW. `policies`: the same
    averaged over seasons. `comparisons`: per (a, b, method, metric) the bootstrap result of
    the per-GW difference plus `season_diff` (the mean difference per season, summed over
    GWs)."""

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


def _comparison(
    diffs: pd.DataFrame, a: str, b: str, method: str, **bootstrap: Any
) -> list[dict[str, Any]]:
    rows = []
    for metric, column in (("realized", "diff"), ("xg", "diff_xg")):
        result = block_bootstrap(diffs, value=column, **bootstrap)
        valid = diffs[diffs[column].notna()]
        cells = valid.groupby(["season", "gw_index"])[column].mean().astype("float64")
        n_seasons = cells.index.get_level_values("season").nunique()
        season_diff = float(cells.sum() / n_seasons) if n_seasons else float("nan")
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
    ci: float = 0.9,
) -> Summary:
    """Season totals per policy (mean over starts) from `run_grid` results, and for each
    (a, b) in `pairs` the full-run paired difference, plus each `per_decision` frame's, both
    realized and xG-scored, with block-bootstrap CIs (`block_bootstrap` parameters)."""
    bootstrap = {"block_length": block_length, "n_boot": n_boot, "seed": seed, "ci": ci}
    season_totals = _season_totals(results)
    policies = (
        season_totals.drop(columns=["season", "n_starts"])
        .groupby("policy", sort=True)
        .mean()
        .reset_index()
        .assign(n_seasons=season_totals.groupby("policy", sort=True)["season"].nunique().values)
    )
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


def log_experiment(
    path: Path | str,
    command: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    n_variants: int,
) -> None:
    """Append one row (`EXPERIMENT_COLUMNS`: UTC timestamp, git sha, command, config and
    metrics as JSON, n_variants) to the CSV at `path`, writing the header if the file is
    new or empty."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    row = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "command": command,
        "config": json.dumps(config, sort_keys=True, default=str),
        "metrics": json.dumps(metrics, sort_keys=True, default=str),
        "n_variants": int(n_variants),
    }
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EXPERIMENT_COLUMNS))
        if new:
            writer.writeheader()
        writer.writerow(row)
