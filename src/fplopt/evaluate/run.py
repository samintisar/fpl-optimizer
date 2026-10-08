"""Walk-forward evaluation of xP models against realized outcomes (Phase 5 plan,
*Evaluation (criterion 1)*; PLAN §5 *Metrics*, *Splits*).

`evaluate(store, models, seasons)` walks every deadline (gw_index 1 to the last) of every
season. At each deadline:

1. `view = store.as_of(deadline)`; each model's full xP frame (horizons 0-5) through
   `Caches.xp`, the backtester's path, so walk-forward `FittedModel`s memoize their fits by
   cutoff. A model runs only in the seasons it has inputs for (`FIRST_SEASON`: `ep_next` and
   `ep_next_fade` need FPL snapshots, 2021/22 on; before that they are all-zero).
2. Every horizon row is joined to the realized per player-GW outcome of its GW
   (`outcomes.fixture_outcomes` / `gw_totals`, read from `as_of(lockdown + 1 µs)` and
   re-scored under the season's backtest rules): a pool player without a row in a played
   GW scored 0 in 0 minutes; a GW not played yet has null outcomes (left out of metrics).
3. Candidates (`candidate`): the union over the models run at the deadline of each model's
   top-N per position by horizon-summed xP (N = the optimizer's pruning defaults,
   `DEFAULT_PRUNE_N`: 20/60/60/30 for GK/DEF/MID/FWD), so every model is scored on the same
   candidate rows at a deadline. That set depends on which models are in the run (the
   per-model `mse_candidates`), so each row also carries `own_candidate` (in the model's
   own top-N), and a pairwise test uses the pair's union (`own_candidate` of either model):
   the same rows whatever else is run.
4. Decisions (`regrets`): squads = the template squad and `n_random` seeded random squads
   (`start_states`; a refused one is skipped), the same for every model. Each model picks
   its XI and captain with `best_lineup` on its horizon-0 xP; regret per `lineup_regret`.

`compute_metrics` turns the predictions and regrets into tables (lists of records) per model,
split (`all`, `develop` 2016/17-2022/23, `validate` 2023/24-2024/25; a split without rows
is left out) and horizon ("0".."5", and "1-5" pooled):
- `xp`: MSE over all pool players, candidate MSE (`mse_candidates`, the same MSE on the
  candidate rows), MAE (diagnostic only), mean xP and mean points;
- `bands`: MSE per band of each model's own predicted xP (`XP_BANDS`; in metrics.json the
  open ends' `lo`/`hi` are null), with mean xP and mean points (calibration per band);
- `by_season`: horizon-0 MSE, candidate MSE and MAE per season;
- `regret`: mean XI and captain regret (horizon 0);
- `dm`: every pair of models on their common sample (deadlines where both ran): one-sided
  Diebold-Mariano tests (`metrics.diebold_mariano`, clustered by deadline, lag = the
  horizon, pooled "1-5" with lag 5) on squared error (`mse`), squared error on the pair's
  candidates (`mse_candidates`: either model's own top-N) and on regrets (`xi_regret`,
  `captain_regret`);
- `components`: for models whose xP frames carry component columns (`metrics.COMPONENTS`,
  the hook for models with components): their metrics per split and horizon.

Parallel by season (`jobs`): the season units run through the backtester's worker pool
(`backtest.evaluate.run_units`), each worker with its own store and `Caches`; results are
concatenated in season order, so `jobs > 1` gives the same output as `jobs = 1`.

Like the simulator this is an orchestrator: it holds the `DataStore` and reads only through
`store.as_of(...)`; models are called on their deadline view only. Holdout seasons are
refused (`HoldoutError`).
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from fplopt.backtest.evaluate import SPLITS, RulesFn, json_safe, run_units
from fplopt.backtest.policies import target_xp
from fplopt.backtest.rules import Rules, backtest_rules
from fplopt.backtest.simulator import Caches, HoldoutError, season_schedule
from fplopt.backtest.start_states import StartStateError, random_state, template_state
from fplopt.backtest.state import SquadState
from fplopt.evaluate.metrics import (
    COMPONENTS,
    band_table,
    component_metrics,
    diebold_mariano,
    lineup_regret,
    mae,
    mse,
)
from fplopt.evaluate.outcomes import fixture_outcomes, gw_totals
from fplopt.features.store import AsOfView, DataStore
from fplopt.models import MODELS
from fplopt.models.baseline import XP_DTYPES
from fplopt.optimize.params import DEFAULT_PRUNE_N
from fplopt.seasons import HOLDOUT_SEASONS, season_label

log = logging.getLogger(__name__)

__all__ = (
    "FIRST_SEASON",
    "N_RANDOM",
    "PREDICTION_COLUMNS",
    "REGRET_COLUMNS",
    "XP_BANDS",
    "EvalResult",
    "candidate_keys",
    "compute_metrics",
    "deadline_squads",
    "evaluate",
    "model_seasons",
)

# Models whose inputs start later: FPL's ep_next is in the snapshots from 2020/21 GW32, so
# before 2021/22 these models are all-zero and are neither run nor compared there.
FIRST_SEASON: Mapping[str, int] = MappingProxyType({"ep_next": 2021, "ep_next_fade": 2021})
N_RANDOM = 3  # random squads per deadline for the regret metrics (besides the template)
# Predicted-xP bands (per player-GW, the model's own prediction): [lo, hi).
XP_BANDS = (-math.inf, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, math.inf)
# Realized per player-GW columns joined to the predictions (null if the GW is not played).
REALIZED = ("points", "minutes", "starts", "n_fixtures", "goals_scored", "assists", "clean_sheets")
PREDICTION_COLUMNS = (
    ("model", "str"),
    ("season", "int64"),
    ("deadline", pd.DatetimeTZDtype("us", "UTC")),
    ("gw", "int64"),
    ("gw_index", "int64"),
    ("player_key", "int64"),
    ("element_type", "int64"),
    ("horizon", "int64"),
    ("target_gw", "int64"),
    ("target_gw_index", "int64"),
    ("xp", "float64"),
    ("candidate", "bool"),
    ("own_candidate", "bool"),
    *((name, "float64") for name in REALIZED),
)
REGRET_COLUMNS = (
    ("model", "str"),
    ("season", "int64"),
    ("deadline", pd.DatetimeTZDtype("us", "UTC")),
    ("gw", "int64"),
    ("gw_index", "int64"),
    ("squad", "str"),
    ("points", "int64"),
    ("best_points", "int64"),
    ("xi_regret", "int64"),
    ("captain_points", "int64"),
    ("best_captain_points", "int64"),
    ("captain_regret", "int64"),
)
_XP_BASE = tuple(name for name, _ in XP_DTYPES)
HORIZON_LABELS = ("0", "1", "2", "3", "4", "5", "1-5")


# --- inputs -----------------------------------------------------------------------------------


def model_seasons(models: Sequence[str], seasons: Sequence[int]) -> dict[str, tuple[int, ...]]:
    """Per model, the seasons it runs in: `seasons` from its `FIRST_SEASON` on."""
    return {m: tuple(s for s in sorted(seasons) if s >= FIRST_SEASON.get(m, 0)) for m in models}


def _check(models: Sequence[str], seasons: Sequence[int]) -> None:
    if not models:
        raise ValueError("no models to evaluate")
    if len(set(models)) != len(models):
        raise ValueError(f"model names must be unique: {list(models)}")
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        raise ValueError(f"unknown xP model(s) {unknown} (MODELS: {sorted(MODELS)})")
    if not seasons:
        raise ValueError("no seasons to evaluate")
    holdout = sorted(set(seasons) & HOLDOUT_SEASONS)
    if holdout:
        raise HoldoutError(
            f"{', '.join(map(season_label, holdout))} is a holdout season; the evaluation "
            "refuses it until Phase 6"
        )
    empty = [m for m, s in model_seasons(models, seasons).items() if not s]
    if empty:
        first = ", ".join(f"{m} from {season_label(FIRST_SEASON[m])}" for m in empty)
        raise ValueError(f"no season to evaluate for {empty} ({first})")


def deadline_squads(
    view: AsOfView, rules: Rules, n_random: int = N_RANDOM, seed: int = 0
) -> list[tuple[str, SquadState]]:
    """(squad id, state): the template squad and random squads with seeds seed..seed +
    n_random − 1 at the view's deadline; a refused one (`StartStateError`) is skipped."""
    builders = [("template", lambda: template_state(view, rules))]
    builders += [
        (f"random{s}", lambda s=s: random_state(view, rules, s))
        for s in range(seed, seed + n_random)
    ]
    out = []
    for squad_id, build in builders:
        try:
            out.append((squad_id, build()))
        except StartStateError as exc:
            log.debug("%s at %s refused: %s", squad_id, view.deadline, exc)
    return out


def candidate_keys(
    frames: Mapping[str, pd.DataFrame],
    pool: pd.DataFrame,
    prune_n: Mapping[int, int] = DEFAULT_PRUNE_N,
) -> np.ndarray:
    """Sorted player_keys in the union over `frames` (xP frames at one deadline) of each
    frame's top `prune_n[element_type]` pool players by horizon-summed xP (ties: smaller
    player_key first); a position missing from `prune_n` keeps every player."""
    positions = pool.set_index("player_key")["element_type"]
    chosen: set[int] = set()
    for frame in frames.values():
        total = frame.groupby("player_key", sort=True)["xp"].sum()
        table = pd.DataFrame(
            {
                "player_key": total.index.to_numpy(dtype="int64"),
                "xp": total.to_numpy(dtype="float64"),
                "et": positions.reindex(total.index).to_numpy(),
            }
        )
        table = table.sort_values(["xp", "player_key"], ascending=[False, True], kind="mergesort")
        for et, part in table.groupby("et", sort=True):
            n = prune_n.get(int(et))
            chosen.update(part["player_key"].astype(int) if n is None else part["player_key"][:n])
    return np.array(sorted(chosen), dtype="int64")


# --- one season ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class _EvalJob:
    models: tuple[str, ...]
    n_random: int
    seed: int


@dataclass(frozen=True)
class _SeasonResult:
    predictions: pd.DataFrame
    regrets: pd.DataFrame
    seconds: float
    n_deadlines: int


def _empty(columns: Sequence[tuple[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in columns})


class _Realized:
    """Per player-GW realized outcomes of one season, read once per GW."""

    def __init__(self, store: DataStore, rules: Rules, season: int, schedule: pd.DataFrame):
        self.store, self.rules, self.season = store, rules, season
        self.lockdowns = dict(
            zip(schedule["gw"].astype(int), schedule["lockdown_time"], strict=True)
        )
        self.frames: dict[int, pd.DataFrame | None] = {}

    def gw(self, gw: int) -> pd.DataFrame | None:
        """`gw_totals` of the GW indexed by player_key, or None if not played / unknown."""
        if gw not in self.frames:
            frame = None
            if gw in self.lockdowns:
                fixtures = fixture_outcomes(
                    self.store, self.rules, self.season, gw, self.lockdowns[gw]
                )
                if not fixtures.empty:
                    frame = gw_totals(fixtures).set_index("player_key")
            self.frames[gw] = frame
        return self.frames[gw]

    def join(self, rows: pd.DataFrame) -> pd.DataFrame:
        """`rows` (with `target_gw`, `player_key`) plus the `REALIZED` columns (float64):
        null if the GW is not played; a player without a row in a played GW did not play
        (0; `starts` null where the source has no starts)."""
        parts = []
        for gw, part in rows.groupby("target_gw", sort=False):
            totals = self.gw(int(gw))
            if totals is None:
                parts.append(part.assign(**{name: np.nan for name in REALIZED}))
                continue
            values = {}
            for name in REALIZED:
                mapped = part["player_key"].map(totals[name].astype("float64"))
                values[name] = mapped if name == "starts" else mapped.fillna(0.0)
            parts.append(part.assign(**values))
        return pd.concat(parts).sort_index()


def _season_unit(
    store: DataStore, caches: Caches, rules: Rules, unit: tuple[int, str, Any], job: _EvalJob
) -> _SeasonResult:
    """Predictions and regrets of every deadline of one season (module docstring)."""
    season = unit[0]
    began = time.perf_counter()
    models = [m for m in job.models if season >= FIRST_SEASON.get(m, 0)]
    schedule = season_schedule(store, season, 1)
    realized = _Realized(store, rules, season, schedule)
    predictions, regrets = [], []
    for row in schedule.itertuples(index=False):
        deadline, gw, gw_index = row.deadline_time, int(row.gw), int(row.gw_index)
        view = store.as_of(deadline)
        pool = caches.pool(store, view)
        frames = {m: caches.xp(store, m, view) for m in models}
        candidates = candidate_keys(frames, pool)
        positions = pool.set_index("player_key")["element_type"]
        for model, frame in frames.items():
            missing = ~frame["player_key"].isin(positions.index)
            if missing.any():
                raise ValueError(
                    f"{model} at {deadline}: {int(missing.sum())} xP row(s) for players "
                    "outside the pool"
                )
            extra = [c for c in frame.columns if c not in _XP_BASE]
            rows = frame.rename(columns={"gw": "target_gw", "gw_index": "target_gw_index"})
            rows = rows.assign(
                model=model,
                season=season,
                deadline=deadline,
                gw=gw,
                gw_index=gw_index,
                element_type=frame["player_key"].map(positions).to_numpy(dtype="int64"),
                candidate=frame["player_key"].isin(candidates).to_numpy(dtype=bool),
                own_candidate=frame["player_key"]
                .isin(candidate_keys({model: frame}, pool))
                .to_numpy(dtype=bool),
            )
            rows = realized.join(rows.reset_index(drop=True))
            base = [name for name, _ in PREDICTION_COLUMNS]
            predictions.append(rows[[*base, *extra]].astype(dict(PREDICTION_COLUMNS)))
        outcomes = realized.gw(gw)
        if outcomes is None:
            continue  # not played: no decision metrics
        points = {
            int(k): (int(p), int(m))
            for k, p, m in zip(outcomes.index, outcomes["points"], outcomes["minutes"], strict=True)
        }
        for squad_id, state in deadline_squads(view, rules, job.n_random, job.seed):
            squad = pd.DataFrame(
                {
                    "player_key": [h.player_key for h in state.holdings],
                    "element_type": [h.element_type for h in state.holdings],
                },
                dtype="int64",
            )
            for model, frame in frames.items():
                result = lineup_regret(squad, target_xp(frame), points, rules)
                regrets.append(
                    {
                        "model": model,
                        "season": season,
                        "deadline": deadline,
                        "gw": gw,
                        "gw_index": gw_index,
                        "squad": squad_id,
                        "points": int(result.points),
                        "best_points": int(result.best_points),
                        "xi_regret": int(result.xi_regret),
                        "captain_points": int(result.captain_points),
                        "best_captain_points": int(result.best_captain_points),
                        "captain_regret": int(result.captain_regret),
                    }
                )
    seconds = time.perf_counter() - began
    log.info(
        "%s: %d deadlines, models %s, %.1fs", season_label(season), len(schedule), models, seconds
    )
    frame = pd.concat(predictions, ignore_index=True) if predictions else _empty(PREDICTION_COLUMNS)
    regret_frame = pd.DataFrame(regrets, columns=[n for n, _ in REGRET_COLUMNS])
    return _SeasonResult(
        predictions=frame,
        regrets=regret_frame.astype(dict(REGRET_COLUMNS)),
        seconds=seconds,
        n_deadlines=len(schedule),
    )


# --- the run ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalResult:
    """`predictions`: one row per model, deadline, pool player and horizon
    (`PREDICTION_COLUMNS`, then any extra component columns of the models' xP frames),
    sorted by (model in the given order, deadline, player_key, horizon). `regrets`: one row
    per model, deadline and squad (`REGRET_COLUMNS`), same order by squad. `metrics`:
    `compute_metrics`. `timings`: seconds per season unit."""

    predictions: pd.DataFrame
    regrets: pd.DataFrame
    metrics: dict[str, Any]
    timings: dict[str, float] = field(default_factory=dict)


def _sort(frame: pd.DataFrame, models: Sequence[str], keys: Sequence[str]) -> pd.DataFrame:
    rank = frame["model"].map({m: i for i, m in enumerate(models)})
    order = frame.assign(_rank=rank).sort_values(["_rank", *keys], kind="mergesort").index
    return frame.loc[order].reset_index(drop=True)


def evaluate(
    store: DataStore,
    models: Sequence[str],
    seasons: Sequence[int],
    *,
    jobs: int = 1,
    n_random: int = N_RANDOM,
    seed: int = 0,
    rules_fn: RulesFn = backtest_rules,
    caches: Caches | None = None,
) -> EvalResult:
    """Walk-forward predictions and metrics of `models` (MODELS keys) over every deadline of
    `seasons` (module docstring). Raises HoldoutError for a holdout season and ValueError
    for unknown or duplicate models or a model with no season to run in (`FIRST_SEASON`),
    before anything runs. `jobs > 1` runs the seasons in worker processes, same result."""
    models, seasons = list(models), sorted(set(seasons))
    _check(models, seasons)
    if n_random < 0:
        raise ValueError(f"n_random must be >= 0, got {n_random}")
    caches = Caches() if caches is None else caches
    units = [(season, "all deadlines", None) for season in seasons]
    job = _EvalJob(tuple(models), n_random, seed)
    results = run_units(store, caches, rules_fn, _season_unit, units, job, jobs)
    predictions = pd.concat([r.predictions for r in results], ignore_index=True)
    regrets = pd.concat([r.regrets for r in results], ignore_index=True)
    predictions = _sort(predictions, models, ["deadline", "player_key", "horizon"])
    regrets = _sort(regrets, models, ["deadline", "squad"])
    timings = {season_label(s): round(r.seconds, 1) for s, r in zip(seasons, results, strict=True)}
    metrics = compute_metrics(predictions, regrets, models)
    metrics["seasons"] = {m: list(s) for m, s in model_seasons(models, seasons).items()}
    metrics["deadlines"] = {
        season_label(s): r.n_deadlines for s, r in zip(seasons, results, strict=True)
    }
    return EvalResult(predictions, regrets, json_safe(metrics), timings)


# --- metrics ----------------------------------------------------------------------------------


def _split_parts(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    parts = []
    for split, seasons in SPLITS:
        part = frame if seasons is None else frame[frame["season"].isin(seasons)]
        if len(part):
            parts.append((split, part))
    return parts


def _horizon_parts(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame, int]]:
    """(label, rows, DM lag) per horizon label present: each horizon, and 1-5 pooled."""
    parts = []
    for label in HORIZON_LABELS:
        if label == "1-5":
            part, lag = frame[frame["horizon"].between(1, 5)], 5
        else:
            part, lag = frame[frame["horizon"] == int(label)], int(label)
        if len(part):
            parts.append((label, part, lag))
    return parts


def _xp_row(part: pd.DataFrame) -> dict[str, Any]:
    cand = part[part["candidate"]]
    return {
        "n": len(part),
        "n_deadlines": int(part["deadline"].nunique()),
        "mse": mse(part["xp"], part["points"]),
        "mse_candidates": mse(cand["xp"], cand["points"]),
        "n_candidates": len(cand),
        "mae": mae(part["xp"], part["points"]),
        "mean_xp": float(part["xp"].mean()),
        "mean_points": float(part["points"].mean()),
    }


def _dm_rows(
    a: str, b: str, merged: pd.DataFrame, metric: str, split: str, horizon: str, lag: int
) -> dict[str, Any]:
    result = diebold_mariano(merged["loss_a"], merged["loss_b"], merged["deadline"], lag)
    return {
        "a": a,
        "b": b,
        "metric": metric,
        "split": split,
        "horizon": horizon,
        "n": len(merged),
        "seasons": sorted(int(s) for s in merged["season"].unique()),
        "value_a": float(merged["loss_a"].mean()),
        "value_b": float(merged["loss_b"].mean()),
    } | result.to_dict()


def compute_metrics(
    predictions: pd.DataFrame, regrets: pd.DataFrame, models: Sequence[str]
) -> dict[str, Any]:
    """The metric tables of the module docstring from `evaluate`'s frames (rows with a null
    realized `points`, GWs not played yet, are left out)."""
    scored = predictions[predictions["points"].notna()]
    xp_rows, band_rows, season_rows, regret_rows, component_rows = [], [], [], [], []
    for model in models:
        mine = scored[scored["model"] == model]
        for split, part in _split_parts(mine):
            for label, rows, _ in _horizon_parts(part):
                xp_rows.append({"model": model, "split": split, "horizon": label} | _xp_row(rows))
                bands = band_table(rows["xp"], rows["points"], XP_BANDS)
                for band in bands.to_dict("records"):
                    band_rows.append({"model": model, "split": split, "horizon": label} | band)
                for component in COMPONENTS:
                    if all(c in rows.columns for c in component.columns):
                        if rows[list(component.columns)].isna().all().all():
                            continue
                        values = component_metrics(rows, component)
                        component_rows.append(
                            {"model": model, "component": component.name, "split": split}
                            | {"horizon": label}
                            | values
                        )
        for season, part in mine[mine["horizon"] == 0].groupby("season", sort=True):
            row = _xp_row(part)
            season_rows.append(
                {"model": model, "season": int(season)}
                | {k: row[k] for k in ("n", "mse", "mse_candidates", "mae", "mean_xp")}
                | {"mean_points": row["mean_points"]}
            )
        theirs = regrets[regrets["model"] == model]
        for split, part in _split_parts(theirs):
            regret_rows.append(
                {
                    "model": model,
                    "split": split,
                    "n": len(part),
                    "n_deadlines": int(part["deadline"].nunique()),
                    "xi_regret": float(part["xi_regret"].mean()),
                    "captain_regret": float(part["captain_regret"].mean()),
                    "points": float(part["points"].mean()),
                    "best_points": float(part["best_points"].mean()),
                }
            )
    dm_rows = []
    keys = ["deadline", "player_key", "horizon"]
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            side_a = scored.loc[
                scored["model"] == a, [*keys, "season", "own_candidate", "xp", "points"]
            ]
            side_b = scored.loc[scored["model"] == b, [*keys, "own_candidate", "xp"]]
            merged = side_a.merge(side_b, on=keys, how="inner", suffixes=("_a", "_b"))
            # The pair's candidates: independent of the other models in the run.
            merged["candidate"] = merged["own_candidate_a"] | merged["own_candidate_b"]
            merged["loss_a"] = (merged["xp_a"] - merged["points"]) ** 2
            merged["loss_b"] = (merged["xp_b"] - merged["points"]) ** 2
            for split, part in _split_parts(merged):
                for label, rows, lag in _horizon_parts(part):
                    dm_rows.append(_dm_rows(a, b, rows, "mse", split, label, lag))
                    cand = rows[rows["candidate"]]
                    if len(cand):
                        dm_rows.append(_dm_rows(a, b, cand, "mse_candidates", split, label, lag))
            regret_keys = ["deadline", "squad", "season"]
            for metric in ("xi_regret", "captain_regret"):
                ra = regrets.loc[regrets["model"] == a, [*regret_keys, metric]]
                rb = regrets.loc[regrets["model"] == b, [*regret_keys, metric]]
                pair = ra.merge(rb, on=regret_keys, how="inner", suffixes=("_a", "_b"))
                pair = pair.rename(columns={f"{metric}_a": "loss_a", f"{metric}_b": "loss_b"})
                for split, part in _split_parts(pair):
                    dm_rows.append(_dm_rows(a, b, part, metric, split, "0", 0))
    return {
        "models": list(models),
        "xp": xp_rows,
        "bands": band_rows,
        "by_season": season_rows,
        "regret": regret_rows,
        "dm": dm_rows,
        "components": component_rows,
    }
