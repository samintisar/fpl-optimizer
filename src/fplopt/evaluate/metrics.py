"""Forecast metrics for xP and its components (PLAN §5 *Metrics*; Phase 5 plan, *Evaluation
(criterion 1)*). Pure functions on arrays and frames, no I/O, so the component tasks
(minutes, goals, clean sheets) reuse them on their own predictions.

- `mse`, `mae`: per row, optionally weighted. MSE is the xP metric (the mean minimizes it);
  MAE is a diagnostic only (the median minimizes it, which drags xP down).
- `diebold_mariano`: one-sided test of equal expected loss on paired losses clustered by GW
  (`cluster_means`, then a t-statistic with Newey-West variance, `newey_west_variance`,
  and the Harvey-Leybourne-Newbold small-sample correction).
- `brier_decomposition`, `reliability_table`: Brier score with the binned Murphy
  decomposition, and per-bin calibration; `band_table`: the same per band of any predicted
  value (xP bands: always by the *predicted* value, never by the outcome).
- `minutes_class`, `ordinal_log_loss`, `ranked_probability_score`: the 3-class minutes
  outcome 0 / 1-59 / 60+.
- `poisson_log_likelihood`: mean log-likelihood of counts under Poisson rates.
- `lineup_regret`: captain and XI regret of the lineup `best_lineup` picks on a model's xP.
- `COMPONENTS`, `component_metrics`: which predicted component columns are scored against
  which realized outcome, and how (the hook for models that expose components).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from fplopt.backtest.evaluate import t_sf
from fplopt.backtest.gw_score import Points, score_gameweek
from fplopt.backtest.policies import best_lineup
from fplopt.backtest.rules import Rules
from fplopt.backtest.simulator import hindsight_points

__all__ = (
    "COMPONENTS",
    "MINUTES_CLASSES",
    "BrierDecomposition",
    "Component",
    "DMResult",
    "LineupRegret",
    "band_table",
    "brier_decomposition",
    "cluster_means",
    "component_metrics",
    "diebold_mariano",
    "lineup_regret",
    "mae",
    "minutes_class",
    "mse",
    "newey_west_variance",
    "ordinal_log_loss",
    "poisson_log_likelihood",
    "ranked_probability_score",
    "reliability_table",
)

MINUTES_CLASSES = ("0", "1-59", "60+")
LOG_EPS = 1e-15  # probabilities are clipped to [LOG_EPS, 1] before taking logs
RATE_EPS = 1e-12  # Poisson rates are floored at this


def _floats(values: Sequence[float] | np.ndarray | pd.Series) -> np.ndarray:
    return np.asarray(values, dtype="float64").reshape(-1)


def _pair(pred: object, actual: object) -> tuple[np.ndarray, np.ndarray]:
    p, a = _floats(pred), _floats(actual)
    if p.shape != a.shape:
        raise ValueError(f"predictions and outcomes differ in length: {len(p)} vs {len(a)}")
    return p, a


# --- point forecasts ------------------------------------------------------------------------


def mse(pred: object, actual: object, weights: object | None = None) -> float:
    """Mean squared error, weighted by `weights` (≥ 0) if given; NaN on an empty sample."""
    p, a = _pair(pred, actual)
    if not len(p):
        return math.nan
    if weights is None:
        return float(np.mean((p - a) ** 2))
    w = _floats(weights)
    if w.shape != p.shape or (w < 0).any() or w.sum() <= 0:
        raise ValueError("weights must be >= 0, one per row, with a positive sum")
    return float(np.sum(w * (p - a) ** 2) / w.sum())


def mae(pred: object, actual: object) -> float:
    """Mean absolute error (a diagnostic only); NaN on an empty sample."""
    p, a = _pair(pred, actual)
    return float(np.mean(np.abs(p - a))) if len(p) else math.nan


# --- Diebold-Mariano --------------------------------------------------------------------------


def cluster_means(values: object, clusters: object) -> np.ndarray:
    """The mean of `values` per cluster, ordered by cluster label ascending (labels must sort
    in time order, e.g. deadlines)."""
    v = _floats(values)
    labels = pd.Series(np.asarray(clusters).reshape(-1))
    if len(labels) != len(v):
        raise ValueError("values and clusters differ in length")
    return pd.Series(v).groupby(labels, sort=True).mean().to_numpy(dtype="float64")


def newey_west_variance(series: object, lag: int = 0) -> float:
    """Long-run variance of a series with Bartlett weights (Newey-West):
    γ0 + 2 Σ_{k=1..lag} (1 − k/(lag + 1)) γk, with γk = (1/n) Σ_t (x_t − x̄)(x_{t−k} − x̄).
    lag 0 is the plain (biased, 1/n) variance. Never negative."""
    if lag < 0:
        raise ValueError(f"lag must be >= 0, got {lag}")
    x = _floats(series)
    n = len(x)
    if n == 0:
        return math.nan
    d = x - x.mean()
    variance = float(d @ d) / n
    for k in range(1, min(lag, n - 1) + 1):
        variance += 2 * (1 - k / (lag + 1)) * float(d[k:] @ d[:-k]) / n
    return variance


@dataclass(frozen=True)
class DMResult:
    """`mean_diff`: mean over clusters of the per-cluster mean loss difference A − B (< 0:
    A has the lower loss); `t_stat`: the Harvey-Leybourne-Newbold corrected statistic
    (`diebold_mariano`); `p_a_better`: one-sided p for H1 "A's expected loss is lower"
    (P(T ≤ t), T ~ Student t with n_clusters − 1 df); `p_b_better` = P(T ≥ t). The
    statistic and p-values are NaN with fewer than 2 clusters or no variance."""

    mean_diff: float
    t_stat: float
    p_a_better: float
    p_b_better: float
    n_clusters: int
    lag: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def diebold_mariano(loss_a: object, loss_b: object, clusters: object, lag: int = 0) -> DMResult:
    """One-sided Diebold-Mariano test on paired losses (one row per player-GW, or per squad
    for regrets) clustered by `clusters` (the GW: the deadline).

    Per cluster d_g = the mean of loss_a − loss_b (clusters ordered by label); then
    DM = d̄ / sqrt(V / G) with V the Newey-West long-run variance of d (`lag`: 0 for the
    target GW; forecasts h GWs ahead overlap, so use lag h), corrected for small samples
    (Harvey, Leybourne & Newbold 1997, forecast step k = lag + 1):
    t = DM · sqrt((G + 1 − 2k + k(k − 1)/G) / G), compared with Student t on G − 1 degrees
    of freedom. With lag 0 this is exactly the one-sample t-test of the d_g (unbiased
    variance)."""
    a, b = _pair(loss_a, loss_b)
    d = cluster_means(a - b, clusters)
    n = len(d)
    nan = math.nan
    if n < 2:
        return DMResult(float(d.mean()) if n else nan, nan, nan, nan, n, lag)
    variance = newey_west_variance(d, lag)
    k = lag + 1
    correction = (n + 1 - 2 * k + k * (k - 1) / n) / n
    if not variance > 0 or not correction > 0:
        return DMResult(float(d.mean()), nan, nan, nan, n, lag)
    t = float(d.mean()) / math.sqrt(variance / n) * math.sqrt(correction)
    return DMResult(float(d.mean()), t, t_sf(-t, n - 1), t_sf(t, n - 1), n, lag)


# --- probabilities ----------------------------------------------------------------------------


def _bin_index(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin k holds edges[k] ≤ v < edges[k + 1]; the last bin includes its upper edge."""
    index = np.searchsorted(edges, values, side="right") - 1
    return np.clip(index, 0, len(edges) - 2)


def _probability_edges(bins: int) -> np.ndarray:
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}")
    return np.linspace(0.0, 1.0, bins + 1)


def band_table(pred: object, actual: object, edges: Sequence[float]) -> pd.DataFrame:
    """Per band of the *predicted* value (`edges` ascending; band k = [edges[k], edges[k+1]),
    the last one closed; values outside go to the first/last band): `band`, `lo`, `hi`, `n`,
    `mean_pred`, `mean_actual`, `mse`. Bands without rows are left out."""
    p, a = _pair(pred, actual)
    edges = np.asarray(edges, dtype="float64")
    if len(edges) < 2 or (np.diff(edges) <= 0).any():
        raise ValueError("edges must be ascending, at least two")
    index = _bin_index(p, edges)
    rows = []
    for k in np.unique(index).tolist():
        mask = index == k
        rows.append(
            {
                "band": int(k),
                "lo": float(edges[k]),
                "hi": float(edges[k + 1]),
                "n": int(mask.sum()),
                "mean_pred": float(p[mask].mean()),
                "mean_actual": float(a[mask].mean()),
                "mse": float(np.mean((p[mask] - a[mask]) ** 2)),
            }
        )
    columns = ["band", "lo", "hi", "n", "mean_pred", "mean_actual", "mse"]
    return pd.DataFrame(rows, columns=columns)


def reliability_table(prob: object, outcome: object, bins: int = 10) -> pd.DataFrame:
    """Per equal-width bin of the predicted probability on [0, 1] (`bins` bins, the last one
    closed): `bin`, `lo`, `hi`, `n`, `mean_pred`, `observed_rate`. Empty bins are left out."""
    table = band_table(prob, outcome, _probability_edges(bins))
    table = table.rename(columns={"band": "bin", "mean_actual": "observed_rate"})
    return table[["bin", "lo", "hi", "n", "mean_pred", "observed_rate"]]


@dataclass(frozen=True)
class BrierDecomposition:
    """Brier score = mean (p − y)², and Murphy's binned decomposition over `n_bins`
    equal-width bins: reliability Σ n_k (p̄_k − ȳ_k)² / n (lower is better), resolution
    Σ n_k (ȳ_k − ȳ)² / n (higher is better), uncertainty ȳ(1 − ȳ). brier = reliability −
    resolution + uncertainty exactly when predictions are constant within each bin; with
    spread inside the bins the two differ by the within-bin terms."""

    brier: float
    reliability: float
    resolution: float
    uncertainty: float
    n: int
    n_bins: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def brier_decomposition(prob: object, outcome: object, bins: int = 10) -> BrierDecomposition:
    """`BrierDecomposition` of probabilities `prob` for binary outcomes `outcome` (0/1)."""
    p, y = _pair(prob, outcome)
    if not len(p):
        nan = math.nan
        return BrierDecomposition(nan, nan, nan, nan, 0, bins)
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("outcomes must be 0 or 1")
    index = _bin_index(p, _probability_edges(bins))
    n, base = len(p), float(y.mean())
    reliability = resolution = 0.0
    for k in np.unique(index).tolist():
        mask = index == k
        reliability += mask.sum() * (p[mask].mean() - y[mask].mean()) ** 2
        resolution += mask.sum() * (y[mask].mean() - base) ** 2
    return BrierDecomposition(
        brier=float(np.mean((p - y) ** 2)),
        reliability=float(reliability / n),
        resolution=float(resolution / n),
        uncertainty=base * (1 - base),
        n=n,
        n_bins=bins,
    )


# --- minutes (ordinal) ------------------------------------------------------------------------


def minutes_class(minutes: object) -> np.ndarray:
    """0 for 0 minutes, 1 for 1-59, 2 for 60+ (`MINUTES_CLASSES`), as int64."""
    m = _floats(minutes)
    return np.where(m >= 60, 2, np.where(m > 0, 1, 0)).astype("int64")


def _probs(probs: object, classes: object) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probs, dtype="float64")
    c = np.asarray(classes).reshape(-1).astype("int64")
    if p.ndim != 2 or len(p) != len(c):
        raise ValueError("probs must be (rows, classes) with one class label per row")
    if ((c < 0) | (c >= p.shape[1])).any():
        raise ValueError(f"class labels must be in 0..{p.shape[1] - 1}")
    if (p < -1e-9).any() or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("each row of probs must be >= 0 and sum to 1")
    return p, c


def ordinal_log_loss(probs: object, classes: object) -> float:
    """Mean −log p(observed class) (probabilities clipped at `LOG_EPS`); NaN if empty."""
    p, c = _probs(probs, classes)
    if not len(c):
        return math.nan
    return float(-np.mean(np.log(np.clip(p[np.arange(len(c)), c], LOG_EPS, 1.0))))


def ranked_probability_score(probs: object, classes: object) -> float:
    """Mean ranked probability score, normalized to [0, 1]: Σ_k (F_k − O_k)² / (K − 1) over
    the K − 1 cumulative thresholds (F predicted, O observed CDF); NaN if empty."""
    p, c = _probs(probs, classes)
    if not len(c):
        return math.nan
    k = p.shape[1]
    predicted = np.cumsum(p, axis=1)[:, :-1]
    observed = (c[:, None] <= np.arange(k - 1)[None, :]).astype("float64")
    return float(np.mean(np.sum((predicted - observed) ** 2, axis=1) / (k - 1)))


# --- counts ---------------------------------------------------------------------------------


def poisson_log_likelihood(rate: object, count: object) -> float:
    """Mean Poisson log-likelihood k log λ − λ − log k! (λ floored at `RATE_EPS`; counts
    must be non-negative integers); NaN if empty. Higher is better."""
    lam, k = _pair(rate, count)
    if not len(k):
        return math.nan
    if (k < 0).any() or not np.array_equal(k, np.round(k)):
        raise ValueError("counts must be non-negative integers")
    if (lam < 0).any():
        raise ValueError("rates must be >= 0")
    lam = np.maximum(lam, RATE_EPS)
    ints = k.astype("int64")
    log_factorial = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, ints.max() + 1)))])
    return float(np.mean(k * np.log(lam) - lam - log_factorial[ints]))


# --- decisions --------------------------------------------------------------------------------


@dataclass(frozen=True)
class LineupRegret:
    """A squad's GW under a model's lineup (`lineup_regret`): `points` its score,
    `best_points` the hindsight-best score, `xi_regret` = best_points − points; the captain
    bonus it got (`captain_points`: the armband wearer's raw points, 0 if neither captain
    nor vice played), the best bonus available in its XI (`best_captain_points`: the most
    raw points of a player who counted) and `captain_regret` = the difference."""

    points: Points
    best_points: Points
    xi_regret: Points
    captain_points: Points
    best_captain_points: Points
    captain_regret: Points


def lineup_regret(
    squad: pd.DataFrame,
    xp_target: Mapping[int, float],
    outcomes: Mapping[int, tuple[Points, int]],
    rules: Rules,
) -> LineupRegret:
    """Regret of the lineup a model picks for `squad` (`player_key, element_type`), the same
    way the backtester's greedy policy and `xi_regret` / `captain_regret` columns do it:

    - the model's lineup is `best_lineup` on its target-GW xP (`xp_target`): the best XI,
      captain = the highest-xP starter, vice = the second;
    - it is scored with `score_gameweek` on the realized `outcomes` (player_key -> (points,
      minutes) summed over the GW): autosubs and the vice-captain apply, no chip;
    - the hindsight best is `hindsight_points`: the best valid XI on realized points plus
      the best captain among its starters who played (an oracle, so regret ≥ 0 in practice);
    - captain regret = the most raw points of a player who counted (the XI after autosubs)
      − the raw points of the player who got the armband's multiplier."""
    positions = dict(
        zip(squad["player_key"].astype(int), squad["element_type"].astype(int), strict=True)
    )
    lineup = best_lineup(squad, xp_target, rules)
    score = score_gameweek(lineup, outcomes, positions, rules)

    def raw(key: int | None) -> Points:
        return 0 if key is None else outcomes.get(key, (0, 0))[0]

    best = hindsight_points(squad, dict(outcomes), rules)
    captain = raw(score.captain_used)
    best_captain = max(raw(k) for k in score.counted)
    return LineupRegret(
        points=score.points,
        best_points=best,
        xi_regret=best - score.points,
        captain_points=captain,
        best_captain_points=best_captain,
        captain_regret=best_captain - captain,
    )


# --- components -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Component:
    """A predicted component scored against a realized per player-GW outcome.

    `kind`: "prob" (`columns[0]` is P(target ≥ threshold): Brier with decomposition and a
    reliability table), "ordinal" (`columns` are the probabilities of `MINUTES_CLASSES` of
    the target: log loss and RPS), "count" (`columns[0]` is a Poisson rate for the target:
    log-likelihood and MSE) or "value" (`columns[0]` is a point forecast: MSE, MAE).
    `single_fixture`: only rows whose player had exactly one fixture in the GW (a
    probability of one match's event is not defined for a double); rows with a null target
    (e.g. `starts` before 2022/23 GW16) are left out."""

    name: str
    kind: str
    columns: tuple[str, ...]
    target: str
    threshold: int = 1
    single_fixture: bool = True


# The hook for models with components (Task 5): a model's xP frame may carry extra columns
# per player-GW; the evaluation scores those named here (all of a component's columns must be
# present) against the realized outcome. Add entries as components appear.
COMPONENTS = (
    Component("p_start", "prob", ("p_start",), "starts"),
    Component("p_play", "prob", ("p_play",), "minutes"),
    Component("p_60", "prob", ("p_60",), "minutes", threshold=60),
    Component("minutes", "ordinal", ("p_min_0", "p_min_1_59", "p_min_60"), "minutes"),
    Component("e_minutes", "value", ("e_minutes",), "minutes", single_fixture=False),
    Component("p_cs", "prob", ("p_cs",), "clean_sheets"),
    Component("p_goal", "prob", ("p_goal",), "goals_scored"),
    Component("e_goals", "count", ("e_goals",), "goals_scored", single_fixture=False),
    Component("e_assists", "count", ("e_assists",), "assists", single_fixture=False),
)


def component_metrics(frame: pd.DataFrame, component: Component, bins: int = 10) -> dict:
    """`component`'s metrics on `frame` (its prediction columns, the realized `target` and,
    for single-fixture components, `n_fixtures`): `n` plus, by kind, `brier`,
    `reliability`, `resolution`, `uncertainty`, `reliability_table` (records) / `log_loss`,
    `rps` / `log_likelihood`, `mse`, `mean_pred`, `mean_actual` / `mse`, `mae`."""
    rows = frame[frame[component.target].notna()]
    if component.single_fixture:
        rows = rows[rows["n_fixtures"] == 1]
    rows = rows.dropna(subset=list(component.columns))
    target = rows[component.target].to_numpy(dtype="float64")
    out: dict = {"n": len(rows)}
    if component.kind == "prob":
        prob = rows[component.columns[0]].to_numpy(dtype="float64")
        outcome = (target >= component.threshold).astype("float64")
        out |= brier_decomposition(prob, outcome, bins).to_dict()
        out["reliability_table"] = reliability_table(prob, outcome, bins).to_dict("records")
    elif component.kind == "ordinal":
        probs = rows[list(component.columns)].to_numpy(dtype="float64")
        classes = minutes_class(target)
        out["log_loss"] = ordinal_log_loss(probs, classes)
        out["rps"] = ranked_probability_score(probs, classes)
    elif component.kind == "count":
        rate = rows[component.columns[0]].to_numpy(dtype="float64")
        out["log_likelihood"] = poisson_log_likelihood(rate, target)
        out["mse"] = mse(rate, target)
        out["mean_pred"] = float(rate.mean()) if len(rate) else math.nan
        out["mean_actual"] = float(target.mean()) if len(target) else math.nan
    elif component.kind == "value":
        value = rows[component.columns[0]].to_numpy(dtype="float64")
        out["mse"] = mse(value, target)
        out["mae"] = mae(value, target)
    else:
        raise ValueError(f"unknown component kind {component.kind!r}")
    return out
