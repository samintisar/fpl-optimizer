"""The minutes model: a LightGBM hurdle model with post-processing (Phase 5 plan, Task 3;
PLAN §6.3).

**Parts** (each a `fplopt.models.gbm.train` fit on `fplopt.features.history.training_frame`
rows, features `LAG_FEATURES`; no status flags, which exist only from 2020/21 GW32 — the
flag adjustment is `fplopt.models.availability`):
- P(start);
- P(60+ | start);
- P(sub appearance | no start).
Expected minutes use three constants instead of a fourth model (a documented simpler
choice): the weighted mean minutes of starts with 60+ minutes, of starts under 60 and of
substitute appearances. A substitute appearance of 60+ minutes is counted as under 60 (1.4%
of sub appearances in 2022/23–2024/25).

**Training rows:** every visible player_match row except those inside a deterministic ban
(`banned`; their outcome is fixed by the rule below), sorted by key, weighted by
`season_decay ** age` where age = the cutoff's season − the row's season (older seasons
count less, none is dropped). A part with fewer than `MIN_ROWS` rows (the first deadlines of
2016/17) predicts its weighted label mean (`DEFAULT_RATE` without rows) instead.

**`fit_minutes(view) -> MinutesFit`** fits on everything visible in `view` (the caller
passes the view at the refit cutoff, `fplopt.models.fitted`); **`predict_minutes(view,
fit)`** returns one row per pool player and horizon fixture of his club: `player_key,
fixture_key, team_key, season, gw, gw_index, horizon` (int64) and `p_start, p_60, p_sub,
e_minutes, p_play` (float64), sorted by (player_key, horizon, fixture_key), RangeIndex. All
probabilities are unconditional: `p_60` = P(minutes ≥ 60) = p_start · P(60+ | start),
`p_sub` = P(no start, minutes > 0), `p_play` = P(minutes > 0) = p_start + p_sub;
`e_minutes` = E[minutes] in that fixture (≤ 90 · p_play).

**Post-processing**, in this order:
1. Horizon decay (h ≥ 1; every horizon fixture has the lags as of the deadline): p_start =
   w · p_model + (1 − w) · long_run, w = `horizon_decay ** h`, long_run = the start rate
   over his last 38 rows shrunk toward p_model with `long_run_prior` pseudo-rows.
2. Suspensions (`fplopt.features.history.BAN_RULES`, `RED_BAN`): the first `ban_remaining`
   fixtures of his club in the horizon get p_start = p_sub = 0 (so every probability and
   e_minutes are 0); with `ban_residual`, p_start and P(sub | no start) are instead the
   weighted rates of the rule-banned training rows (`MinutesFit.ban_start` / `ban_sub`:
   the rules are approximate, and such rows still start ~10% of the time), held fixed.
   Rules: 5 yellow cards reached by gw_index 19 → 1 match, 10 by gw_index 32 → 2, 15 → 3,
   any red card → 1; served by his next league fixtures. Simplified: cups and
   Europe ignored, gw_index stands in for the club's 19th / 32nd match, red-card types are
   not distinguished (straight reds are 3 matches in reality; the model sees `reds_l3`).
3. Team normalization per (fixture, team) over the pool players of that club: a logit shift
   δ, the same for every player not fixed at 0, with |δ| ≤ `max_shift`, so that Σ p_start =
   11 (as close as the bound allows); then E[minutes] is scaled by a factor in
   [1 / `max_scale`, `max_scale`] so that Σ e_minutes = 990, each capped at 90 · p_play.
   Bounded, so a club with an incomplete pool is not inflated.

`finish_minutes` (steps 3 and the final columns) and `conditionals` (its inverse) are
public, so the availability layer can adjust p_start and renormalize.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from fplopt.features.history import (
    LAG_FEATURES,
    PREDICTION_KEYS,
    prediction_frame,
    training_frame,
)
from fplopt.features.store import AsOfView
from fplopt.models.gbm import Gbm, GbmParams, train

__all__ = (
    "MINUTES_COLUMNS",
    "MinutesFit",
    "MinutesParams",
    "conditionals",
    "finish_minutes",
    "fit_minutes",
    "predict_minutes",
)

MIN_ROWS = 1000
DEFAULT_RATE = 0.5
TEAM_STARTERS = 11.0
TEAM_MINUTES = 990.0
MAX_MINUTES = 90.0
EPS = 1e-6
BISECTIONS = 60
KEYS = tuple(name for name, _ in PREDICTION_KEYS if name != "kickoff_time")
# `banned`: the fixture falls in a predicted ban (P(start) is the ban residual or 0, held
# fixed by the team normalization and the availability layer).
MINUTES_COLUMNS = (*KEYS, "p_start", "p_60", "p_sub", "e_minutes", "p_play", "banned")
SORT_BY = ("player_key", "horizon", "fixture_key")


@dataclass(frozen=True)
class MinutesParams:
    """Tuned on develop walk-forward (dev/minutes_eval.py, 2017/18–2022/23): `GbmParams()`
    (200 rounds, 15 leaves, 200 rows per leaf) tied with 300 rounds at learning rate 0.03
    and feature fraction 0.8, and beat 100 rounds × 7 leaves and 400 × 31; season decay 0.7
    ≈ 0.9 > 0.5; horizon decay 0.85 > 0.9 > 0.75 > 1 (no decay); the normalization bounds
    and the long-run prior barely matter (normalization on vs off: log loss 0.532 vs
    0.533)."""

    start: GbmParams = field(default_factory=GbmParams)
    sixty: GbmParams = field(default_factory=GbmParams)
    sub: GbmParams = field(default_factory=GbmParams)
    season_decay: float = 0.7  # training weight per season of age
    horizon_decay: float = 0.85  # weight of the model's p_start at horizon h: decay ** h
    long_run_prior: float = 5.0  # pseudo-rows of the model's p_start in the long-run rate
    max_shift: float = 1.0  # |logit shift| bound of the team normalization
    max_scale: float = 1.25  # bound of the team minutes scaling
    # Banned fixtures: P(start) = 0 (False) or the walk-forward residual start rate of
    # rule-banned training rows (True; the ban rules are approximate). Task 5 tests it.
    ban_residual: bool = False


@dataclass(frozen=True)
class Constant:
    """A part fitted on too few rows: predicts its rate everywhere."""

    rate: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.rate, dtype="float64")


@dataclass(frozen=True)
class MinutesFit:
    """The fitted hurdle model (frozen; no state)."""

    start: Gbm | Constant
    sixty: Gbm | Constant
    sub: Gbm | Constant
    minutes_sixty: float  # E[minutes | start, 60+]
    minutes_short: float  # E[minutes | start, < 60]
    minutes_sub: float  # E[minutes | sub appearance]
    params: MinutesParams
    n_rows: int
    ban_start: float = 0.0  # P(start) in a banned fixture (`ban_residual`; else 0)
    ban_sub: float = 0.0  # P(sub appearance | no start) in a banned fixture


def _weighted_mean(values: pd.Series, weights: pd.Series, default: float) -> float:
    total = float(weights.sum())
    return float((values * weights).sum() / total) if total > 0 else default


def _part(rows: pd.DataFrame, label: str, params: GbmParams) -> Gbm | Constant:
    if len(rows) < MIN_ROWS or rows[label].nunique() < 2:
        return Constant(_weighted_mean(rows[label], rows["weight"], DEFAULT_RATE))
    return train(rows, LAG_FEATURES, label, weight="weight", params=params)


def _weighted(rows: pd.DataFrame, season: int, params: MinutesParams) -> pd.DataFrame:
    """`rows` (a training frame) plus `weight` = season_decay ** (season − row season)."""
    age = (season - rows["season"]).clip(lower=0).astype("float64")
    return rows.assign(weight=np.power(params.season_decay, age))


def fit_minutes(
    view: AsOfView, params: MinutesParams | None = None, rows: pd.DataFrame | None = None
) -> MinutesFit:
    """Fit the hurdle model on every row visible in `view` (see the module docstring).
    `rows`: `training_frame(view)` if the caller already built it (not modified)."""
    params = MinutesParams() if params is None else params
    season, _ = view.gameweek_for_deadline()
    rows = _weighted(training_frame(view) if rows is None else rows, season, params)
    banned = rows[rows["banned"] != 0]
    rows = rows[rows["banned"] == 0].reset_index(drop=True)
    ban_start = ban_sub = 0.0
    if params.ban_residual:
        ban_start = _weighted_mean(banned["start"], banned["weight"], 0.0)
        benched_banned = banned[banned["start"] == 0]
        ban_sub = _weighted_mean(benched_banned["sub"], benched_banned["weight"], 0.0)
    started = rows[rows["start"] == 1].reset_index(drop=True)
    benched = rows[rows["start"] == 0].reset_index(drop=True)
    sixty = started["sixty"] == 1
    subbed = benched[benched["sub"] == 1]
    return MinutesFit(
        start=_part(rows, "start", params.start),
        sixty=_part(started, "sixty", params.sixty),
        sub=_part(benched, "sub", params.sub),
        minutes_sixty=_weighted_mean(
            started.loc[sixty, "minutes"], started.loc[sixty, "weight"], 87.0
        ),
        minutes_short=_weighted_mean(
            started.loc[~sixty, "minutes"], started.loc[~sixty, "weight"], 45.0
        ),
        minutes_sub=_weighted_mean(subbed["minutes"], subbed["weight"], 18.0),
        params=params,
        n_rows=len(rows),
        ban_start=ban_start,
        ban_sub=ban_sub,
    )


def out_of_fold_start(rows: pd.DataFrame, season: int, params: MinutesParams) -> np.ndarray:
    """Per row of `rows` (a training frame) an out-of-fold P(start): the start part fitted
    (as in `fit_minutes`, non-banned rows, season-decay weights relative to `season`) on the
    rows of the other season parity. Used where a fit needs P(start) on training rows that
    behaves like a prediction (the availability layer): in-sample GBM predictions are
    sharper than out-of-sample ones."""
    rows = _weighted(rows, season, params)
    odd = (rows["season"].to_numpy() % 2).astype(bool)
    out = np.empty(len(rows), dtype="float64")
    for fold in (False, True):
        train_rows = rows[(odd != fold) & (rows["banned"].to_numpy() == 0)]
        part = _part(train_rows.reset_index(drop=True), "start", params.start)
        target = odd == fold
        out[target] = part.predict(rows[target])
    return out


# --- post-processing ---------------------------------------------------------------------


def _shift_to_sum(
    p: np.ndarray, groups: np.ndarray, free: np.ndarray, target: float, bound: float
) -> np.ndarray:
    """Per group the logit shift δ in [−bound, bound] with Σ_free expit(logit(p) + δ) +
    Σ_fixed p closest to `target` (bisection: Σ is increasing in δ); returns the new p
    (fixed rows unchanged)."""
    codes, uniques = pd.factorize(groups)
    n_groups = len(uniques)
    z = logit(np.clip(p, EPS, 1 - EPS))
    fixed_sum = np.bincount(codes, weights=np.where(free, 0.0, p), minlength=n_groups)
    lo = np.full(n_groups, -bound)
    hi = np.full(n_groups, bound)
    for _ in range(BISECTIONS):
        mid = (lo + hi) / 2
        q = np.where(free, expit(z + mid[codes]), 0.0)
        total = np.bincount(codes, weights=q, minlength=n_groups) + fixed_sum
        above = total > target
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
    delta = (lo + hi) / 2
    return np.where(free, expit(z + delta[codes]), p)


def finish_minutes(
    frame: pd.DataFrame, minutes_sub: float, max_shift: float, max_scale: float
) -> pd.DataFrame:
    """Team normalization and the output columns. `frame` has the KEYS and per row
    `p_start`, `c60` (P(60+ | start)), `csub` (P(sub | no start)), `m_start` (E[minutes |
    start]) and `fixed` (True: p_start and csub stay as given, e.g. 0 for a ban), and
    optionally `banned` (carried to the output; False without it); returns MINUTES_COLUMNS
    (see the module docstring), sorted, RangeIndex."""
    groups = (
        frame["fixture_key"].astype("int64").to_numpy() * 1_000_000
        + frame["team_key"].astype("int64").to_numpy()
    )
    free = ~frame["fixed"].to_numpy(dtype=bool)
    p_start = _shift_to_sum(
        frame["p_start"].to_numpy(dtype="float64"), groups, free, TEAM_STARTERS, max_shift
    )
    c60 = frame["c60"].to_numpy(dtype="float64")
    csub = frame["csub"].to_numpy(dtype="float64")
    p_sub = (1.0 - p_start) * csub
    p_play = p_start + p_sub
    e_minutes = p_start * frame["m_start"].to_numpy(dtype="float64") + p_sub * minutes_sub
    codes, uniques = pd.factorize(groups)
    total = np.bincount(codes, weights=e_minutes, minlength=len(uniques))
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(total > 0, TEAM_MINUTES / total, 1.0)
    scale = np.clip(scale, 1.0 / max_scale, max_scale)
    e_minutes = np.minimum(e_minutes * scale[codes], MAX_MINUTES * p_play)
    out = frame[list(KEYS)].assign(
        p_start=p_start,
        p_60=p_start * c60,
        p_sub=p_sub,
        e_minutes=e_minutes,
        p_play=p_play,
        banned=frame["banned"].to_numpy(dtype=bool) if "banned" in frame else False,
    )
    dtypes = {
        **{name: "int64" for name in KEYS},
        **{name: "float64" for name in MINUTES_COLUMNS[len(KEYS) : -1]},
        "banned": "bool",
    }
    out = out[list(MINUTES_COLUMNS)].astype(dtypes)
    return out.sort_values(list(SORT_BY), kind="mergesort").reset_index(drop=True)


def conditionals(minutes: pd.DataFrame, minutes_sub: float) -> pd.DataFrame:
    """The inverse of `finish_minutes`' last step: from a MINUTES_COLUMNS frame, its KEYS
    with p_start, c60, csub, m_start (0 where p_start is 0; 1 / 0 for c60 / csub when
    undefined), `banned` and fixed (= banned: a re-normalization keeps bans as they are)."""
    p_start = minutes["p_start"].to_numpy(dtype="float64")
    p_sub = minutes["p_sub"].to_numpy(dtype="float64")
    started = p_start > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        c60 = np.where(started, minutes["p_60"].to_numpy(dtype="float64") / p_start, 1.0)
        csub = np.where(p_start < 1, p_sub / (1.0 - p_start), 0.0)
        start_minutes = minutes["e_minutes"].to_numpy(dtype="float64") - p_sub * minutes_sub
        m_start = np.where(started, start_minutes / p_start, 0.0)
    return minutes[list(KEYS)].assign(
        p_start=p_start,
        c60=np.clip(c60, 0.0, 1.0),
        csub=np.clip(csub, 0.0, 1.0),
        m_start=np.clip(m_start, 0.0, MAX_MINUTES),
        banned=minutes["banned"].to_numpy(dtype=bool),
        fixed=minutes["banned"].to_numpy(dtype=bool),
    )


def predict_minutes(view: AsOfView, fit: MinutesFit) -> pd.DataFrame:
    """Per pool player and horizon fixture: MINUTES_COLUMNS (see the module docstring)."""
    params = fit.params
    frame = prediction_frame(view)
    p_model = fit.start.predict(frame)
    c60 = fit.sixty.predict(frame)
    csub = fit.sub.predict(frame)

    n_long = frame["n_long"].to_numpy(dtype="float64")
    starts_long = np.nan_to_num(frame["start_rate_long"].to_numpy(dtype="float64")) * n_long
    long_run = (starts_long + params.long_run_prior * p_model) / (n_long + params.long_run_prior)
    weight = np.power(params.horizon_decay, frame["horizon"].to_numpy(dtype="float64"))
    p_start = weight * p_model + (1.0 - weight) * long_run

    banned = frame["club_fixture_order"].to_numpy() < frame["ban_remaining"].to_numpy()
    p_start = np.where(banned, fit.ban_start, p_start)
    csub = np.where(banned, fit.ban_sub, csub)
    m_start = c60 * fit.minutes_sixty + (1.0 - c60) * fit.minutes_short
    parts = frame[list(KEYS)].assign(
        p_start=p_start, c60=c60, csub=csub, m_start=m_start, fixed=banned, banned=banned
    )
    return finish_minutes(parts, fit.minutes_sub, params.max_shift, params.max_scale)
