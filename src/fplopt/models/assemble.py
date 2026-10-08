"""xP assembly and the `v1` model (PLAN §6.6, §5 *Calibration*; Phase 5 plan, *Assembly*,
*Calibration* and Task 5).

**Fit** (`fit_v1(view) -> V1Fit`, at the refit cutoff; `fplopt.models.fitted`), in this
order: the team model (`fit_team`), the minutes model (`fit_minutes`), the availability
layer on top of it (`fit_availability(view, minutes_fit)`), the shares (`fit_shares`) and
the other components (`fit_components`).

**Per fixture** (`fixture_components(view, fit)`): one row per pool player and horizon
fixture of his club (`FIXTURE_COLUMNS`): `team_lambdas` → λ_for, λ_against, P(CS);
`predict_minutes` → `adjust_minutes` → p_start, p_60, p_sub, p_play, e_minutes and the
conditionals behind them (c60 = P(60+ | start), csub = P(sub | no start), m_start =
E[minutes | start], m_sub, m_sixty = E[minutes | start, 60+]); `predict_shares` →
e_np_goals, e_pen_goals, e_assists; `predict_components` → the saves, penalty-save, card,
own-goal and penalty-miss rates and the bonus coefficients. Nothing is calibrated yet.

**Calibration** (`calibrate_inputs`, then `fixture_points`, then `calibrate_xp`;
`fplopt.models.calibration`; the season's entry of `calibration_table`, fitted on earlier
seasons only):
1. P(start) through its isotonic map; p_60 = p_start · c60, p_sub = (1 − p_start) · csub,
   p_play and e_minutes = p_start · m_start + p_sub · m_sub are recomputed, and the
   player's expected goals and assists scale with his e_minutes (no team renormalization:
   the calibrated P(start) says he plays more or less);
2. the team's P(CS) through its map;
3. P(goal ≥ 1) = p_play · (1 − exp(−e_goals / p_play)) through its map; e_goals is solved
   back from it and the non-penalty and penalty parts scale with it;
4. after the points: xp' = a · p_play + b · xp per position.

**Points** (`fixture_points(frame, rules)`; `rules` = `fplopt.backtest.rules.
backtest_rules(season)` of the deadline's season, so develop/validate xP is on the
`rescored_points` scale: current rules, no defcon). Per player-fixture, with the minutes
split into "60+" (probability p_60, on the pitch m_sixty / 90 of the match) and "1–59"
(probability p_play − p_60, on the pitch the rest of e_minutes spread over it, at most
59/90 of the match):
- appearance = p_play · short_play + p_60 · (long_play − short_play);
- goals = e_goals · goals_scored[pos] (e_goals = non-penalty + penalty goals);
- assists = e_assists · assists;
- clean sheet = p_60 · P(CS) · clean_sheets[pos] (the plan's approximation: the team's
  clean sheet, not "none conceded while he was on");
- goals conceded = goals_conceded[pos] · Σ_parts P(part) · E[floor(N / per_point)], N ~
  Poisson(λ_against · fraction of the match on the pitch): an approximation, since
  conceding is not uniform over a match and the split has two minute levels only;
- saves (goalkeepers) = saves · Σ_parts P(part) · E[floor(S / saves_per_point)], S ~
  Poisson(saves rate per 90 · fraction); penalty saves = penalties_saved · rate per 90 ·
  e_minutes / 90;
- bonus = bonus · β[pos] · E[features] (`fplopt.models.components`; E[goal_k] = p_play ·
  P(Poisson(e_goals / p_play) ≥ k), assists alike, E[cs] = p_60 · P(CS), E[saves3] =
  E[floor(S / 3)] as above), floored at 0;
- yellow / red cards and own goals = their points · rate per 90 · e_minutes / 90;
- penalty misses = penalties_missed · e_pen_goals · (1 − c) / c (c: the club's conversion);
- defensive contributions: `components.DEFCON` (0; Phase 5b Task 10).
xP = their sum.

**Per GW** (`gw_frame`): every pool player × GW from the target GW to `HORIZON` later
(`upcoming_fixtures`): xP and `GW_COMPONENTS` summed over his fixtures (a double sums two,
a blank is 0). Probabilities are per player-GW only for one fixture: P(start), P(plays),
P(60+), the 3 minutes classes, P(clean sheet) (= p_60 · team P(CS): the player's FPL clean
sheet), P(goal ≥ 1); NaN in a double, 0 in a blank (P(0 minutes) 1). Expected minutes,
goals and assists are sums. The frame keeps the standard xP columns and dtypes first (`XP_DTYPES`); the extra
columns are the ones `fplopt.evaluate.metrics.COMPONENTS` scores. The backtester, policies
and optimizer read only player_key, gw, gw_index, horizon and xp.

`MODELS["v1"]` = `FittedModel(fit_v1, predict_v1)`. No state, no file access except the
rules config (`backtest_rules` reads `config/scoring`, never `data/`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fplopt.backtest.rules import Rules, backtest_rules
from fplopt.features.baseline import HORIZON, player_pool, upcoming_fixtures
from fplopt.features.store import AsOfView
from fplopt.models.availability import AvailabilityFit, adjust_minutes, fit_availability
from fplopt.models.baseline import XP_DTYPES
from fplopt.models.calibration import Calibration, apply_isotonic, calibration_for
from fplopt.models.components import (
    BONUS_COLUMNS,
    DEFCON,
    GOALKEEPER,
    ComponentsFit,
    ComponentsParams,
    fit_components,
    predict_components,
)
from fplopt.models.minutes import MinutesFit, MinutesParams, conditionals, fit_minutes
from fplopt.models.minutes import predict_minutes as minutes_predict
from fplopt.models.shares import SharesFit, SharesParams, fit_shares, predict_shares
from fplopt.models.team import TeamFit, TeamParams, fit_team, team_lambdas

__all__ = (
    "FIXTURE_COLUMNS",
    "GW_COMPONENTS",
    "POINT_COLUMNS",
    "V1Fit",
    "V1Params",
    "calibrate_inputs",
    "calibrate_xp",
    "fit_v1",
    "fixture_components",
    "fixture_points",
    "gw_frame",
    "poisson_expected_floor",
    "poisson_tail",
    "predict_fixtures",
    "predict_v1",
)

KEYS = ("player_key", "fixture_key", "team_key", "season", "gw", "gw_index", "horizon")
MINUTES_INPUTS = ("p_start", "p_60", "p_sub", "p_play", "e_minutes", "c60", "csub", "m_start")
FIXTURE_COLUMNS = (
    *KEYS,
    "element_type",
    "opponent_team_key",
    *MINUTES_INPUTS,
    "m_sub",
    "m_sixty",
    "lambda_for",
    "lambda_against",
    "p_cs",
    "e_np_goals",
    "e_pen_goals",
    "e_assists",
    "saves_rate",
    "pen_save_rate",
    "yellow_rate",
    "red_rate",
    "own_goal_rate",
    "pen_miss_ratio",
    *BONUS_COLUMNS,
)
POINT_COLUMNS = (
    "pts_appearance",
    "pts_goals",
    "pts_assists",
    "pts_clean_sheet",
    "pts_conceded",
    "pts_saves",
    "pts_pen_saves",
    "pts_bonus",
    "pts_yellow",
    "pts_red",
    "pts_own_goals",
    "pts_pen_miss",
    "pts_defcon",
)
# Per player-GW component columns (fplopt.evaluate.metrics.COMPONENTS): probabilities
# (single-fixture GWs only) and sums.
GW_PROBABILITIES = (
    "p_start",
    "p_play",
    "p_60",
    "p_min_0",
    "p_min_1_59",
    "p_min_60",
    "p_cs",
    "p_goal",
)
GW_SUMS = ("e_minutes", "e_goals", "e_assists")
GW_COMPONENTS = (*GW_PROBABILITIES, *GW_SUMS)
POISSON_CAP = 40  # Poisson sums run over N = 0..40 (tail < 1e-15 for rates below 10)
SHORT_MAX_FRACTION = 59.0 / 90.0
SAVES_FOR_BONUS = 3  # the bonus regression's `saves3` = saves // 3


def _v1_minutes() -> MinutesParams:
    """Task 3's tuned settings plus the ban residual: on develop it cut the minutes log loss
    (all horizons) from 0.6302 to 0.6287 and on the banned rows from 11.6 to 0.78, with xP
    MSE at horizon 0 4.2609 → 4.2603 (dev/calibrate_v1.py `ban`)."""
    return MinutesParams(ban_residual=True)


@dataclass(frozen=True)
class V1Params:
    team: TeamParams = field(default_factory=TeamParams)
    minutes: MinutesParams = field(default_factory=_v1_minutes)
    shares: SharesParams = field(default_factory=SharesParams)
    components: ComponentsParams = field(default_factory=ComponentsParams)
    calibrate: bool = True  # apply the season's calibration_table entry


@dataclass(frozen=True)
class V1Fit:
    cutoff: pd.Timestamp
    params: V1Params
    team: TeamFit
    minutes: MinutesFit
    availability: AvailabilityFit
    shares: SharesFit
    components: ComponentsFit


def fit_v1(view: AsOfView, params: V1Params | None = None) -> V1Fit:
    """Every component fit on the rows visible in `view` (the refit cutoff's view)."""
    params = V1Params() if params is None else params
    team = fit_team(view, params.team)
    minutes = fit_minutes(view, params.minutes)
    availability = fit_availability(view, minutes)
    shares = fit_shares(view, params.shares)
    components = fit_components(view, params.components)
    return V1Fit(view.deadline, params, team, minutes, availability, shares, components)


# --- Poisson helpers ------------------------------------------------------------------------


def _poisson_pmf(rate: np.ndarray, cap: int = POISSON_CAP) -> np.ndarray:
    """P(N = n), n = 0..cap, per rate (rows); rate 0 gives P(0) = 1 exactly."""
    rate = np.clip(np.asarray(rate, dtype="float64"), 0.0, None)
    steps = rate[:, None] / np.arange(1, cap + 1)[None, :]
    ones = np.ones((len(rate), 1))
    return np.exp(-rate)[:, None] * np.concatenate([ones, np.cumprod(steps, axis=1)], axis=1)


def poisson_expected_floor(rate: np.ndarray, k: int) -> np.ndarray:
    """E[floor(N / k)] for N ~ Poisson(rate)."""
    return _poisson_pmf(rate) @ (np.arange(POISSON_CAP + 1) // k)


def poisson_tail(rate: np.ndarray, k: int) -> np.ndarray:
    """P(N ≥ k) for N ~ Poisson(rate)."""
    return 1.0 - _poisson_pmf(rate)[:, :k].sum(axis=1)


# --- per fixture ----------------------------------------------------------------------------


def fixture_components(view: AsOfView, fit: V1Fit) -> pd.DataFrame:
    """Uncalibrated per player-fixture inputs, `FIXTURE_COLUMNS`, sorted by (player_key,
    horizon, fixture_key), RangeIndex (module docstring)."""
    team = team_lambdas(view, fit.team)
    minutes = minutes_predict(view, fit.minutes)
    minutes = adjust_minutes(view, minutes, fit.availability)
    shares = predict_shares(view, fit.shares, minutes, team)
    components = predict_components(view, fit.components, minutes, team, fit.shares)
    cond = conditionals(minutes, fit.minutes.minutes_sub)
    frame = minutes.assign(
        c60=cond["c60"].to_numpy(),
        csub=cond["csub"].to_numpy(),
        m_start=cond["m_start"].to_numpy(),
        m_sub=fit.minutes.minutes_sub,
        m_sixty=fit.minutes.minutes_sixty,
    )
    sides = team[["fixture_key", "team_key", "lambda_for", "lambda_against", "p_cs"]]
    frame = frame.merge(sides, on=["fixture_key", "team_key"], how="left", validate="many_to_one")
    keys = list(KEYS)
    frame = frame.merge(
        shares[[*keys, "e_np_goals", "e_pen_goals", "e_assists"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    frame = frame.merge(components, on=keys, how="left", validate="one_to_one")
    for column in ("lambda_for", "lambda_against", "p_cs"):
        frame[column] = frame[column].fillna(0.0)
    dtypes = {name: "int64" for name in (*KEYS, "element_type", "opponent_team_key")}
    dtypes |= {name: "float64" for name in FIXTURE_COLUMNS if name not in dtypes}
    out = frame[list(FIXTURE_COLUMNS)].astype(dtypes)
    return out.sort_values(["player_key", "horizon", "fixture_key"], kind="mergesort").reset_index(
        drop=True
    )


def _p_goal(e_goals: np.ndarray, p_play: np.ndarray) -> np.ndarray:
    """P(goals ≥ 1) = p_play · (1 − exp(−e_goals / p_play)) (0 when p_play is 0)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(p_play > 0, e_goals / p_play, 0.0)
    return p_play * -np.expm1(-rate)


def calibrate_inputs(frame: pd.DataFrame, calibration: Calibration | None) -> pd.DataFrame:
    """Steps 1–3 of the calibration (module docstring) on a `fixture_components` frame; the
    frame unchanged without a calibration."""
    if calibration is None:
        return frame
    out = frame.copy()
    horizon = out["horizon"].to_numpy()
    if calibration.p_start is not None:
        old_p = out["p_start"].to_numpy()
        p = np.clip(apply_isotonic(calibration.p_start, old_p, horizon), 0.0, 1.0)
        changed = p != old_p  # rows a map left alone keep their values exactly
        p_sub = (1.0 - p) * out["csub"].to_numpy()
        e_minutes = p * out["m_start"].to_numpy() + p_sub * out["m_sub"].to_numpy()
        old = out["e_minutes"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(changed & (old > 0), e_minutes / old, 1.0)
        out["p_start"] = p
        out["p_60"] = np.where(changed, p * out["c60"].to_numpy(), out["p_60"].to_numpy())
        out["p_sub"] = np.where(changed, p_sub, out["p_sub"].to_numpy())
        out["p_play"] = np.where(changed, p + p_sub, out["p_play"].to_numpy())
        out["e_minutes"] = np.where(changed, e_minutes, old)
        for column in ("e_np_goals", "e_pen_goals", "e_assists"):
            out[column] = out[column].to_numpy() * ratio
    if calibration.p_cs is not None:
        p_cs = apply_isotonic(calibration.p_cs, out["p_cs"].to_numpy(), horizon)
        out["p_cs"] = np.clip(p_cs, 0.0, 1.0)
    if calibration.p_goal is not None:
        p_play = out["p_play"].to_numpy()
        e_goals = out["e_np_goals"].to_numpy() + out["e_pen_goals"].to_numpy()
        p_goal = apply_isotonic(calibration.p_goal, _p_goal(e_goals, p_play), horizon)
        share = np.clip(np.where(p_play > 0, p_goal / np.where(p_play > 0, p_play, 1.0), 0.0), 0, 1)
        new_goals = p_play * -np.log1p(-np.minimum(share, 1.0 - 1e-9))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(e_goals > 0, new_goals / e_goals, 1.0)
        out["e_np_goals"] = out["e_np_goals"].to_numpy() * ratio
        out["e_pen_goals"] = out["e_pen_goals"].to_numpy() * ratio
    return out


def _by_position(element_type: np.ndarray, values: Mapping[int, int]) -> np.ndarray:
    lookup = dict(values)
    return np.array([lookup.get(int(e), 0) for e in element_type], dtype="float64")


def fixture_points(frame: pd.DataFrame, rules: Rules) -> pd.DataFrame:
    """`frame` (`fixture_components`, possibly calibrated) plus `p_goal`, `e_goals`,
    `e_bonus`, `e_saves`, the `POINT_COLUMNS` and `xp` (module docstring)."""
    et = frame["element_type"].to_numpy(dtype="int64")
    p_play = frame["p_play"].to_numpy(dtype="float64")
    p_60 = np.minimum(frame["p_60"].to_numpy(dtype="float64"), p_play)
    e_minutes = frame["e_minutes"].to_numpy(dtype="float64")
    p_short = np.clip(p_play - p_60, 0.0, None)
    f_long = np.clip(frame["m_sixty"].to_numpy(dtype="float64") / 90.0, 0.0, 1.0)
    rest = np.clip(e_minutes - p_60 * f_long * 90.0, 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        f_short = np.where(p_short > 0, rest / (90.0 * p_short), 0.0)
    f_short = np.clip(f_short, 0.0, SHORT_MAX_FRACTION)

    def split_floor(rate_per_90: np.ndarray, k: int) -> np.ndarray:
        return p_60 * poisson_expected_floor(rate_per_90 * f_long, k) + (
            p_short * poisson_expected_floor(rate_per_90 * f_short, k)
        )

    e_np = frame["e_np_goals"].to_numpy(dtype="float64")
    e_pen = frame["e_pen_goals"].to_numpy(dtype="float64")
    e_goals = e_np + e_pen
    e_assists = frame["e_assists"].to_numpy(dtype="float64")
    p_cs = frame["p_cs"].to_numpy(dtype="float64")
    lam_against = frame["lambda_against"].to_numpy(dtype="float64")
    saves_rate = frame["saves_rate"].to_numpy(dtype="float64")
    keeper = et == GOALKEEPER

    with np.errstate(divide="ignore", invalid="ignore"):
        goal_rate = np.where(p_play > 0, e_goals / p_play, 0.0)
        assist_rate = np.where(p_play > 0, e_assists / p_play, 0.0)
    e_saves_floor = np.where(keeper, split_floor(saves_rate, SAVES_FOR_BONUS), 0.0)
    features = np.column_stack(
        [
            p_play,
            p_60,
            p_play * poisson_tail(goal_rate, 1),
            p_play * poisson_tail(goal_rate, 2),
            p_play * poisson_tail(goal_rate, 3),
            p_play * poisson_tail(assist_rate, 1),
            p_play * poisson_tail(assist_rate, 2),
            p_60 * p_cs,
            e_saves_floor,
        ]
    )
    beta = frame[list(BONUS_COLUMNS)].to_numpy(dtype="float64")
    e_bonus = np.clip((features * beta).sum(axis=1), 0.0, None)
    saves_points = np.where(keeper, split_floor(saves_rate, rules.saves_per_point), 0.0)
    exposure = e_minutes / 90.0

    def per_90(column: str) -> np.ndarray:
        """Expected events of a per-90 rate column over the expected minutes."""
        return frame[column].to_numpy(dtype="float64") * exposure

    points = {
        "pts_appearance": p_play * rules.short_play + p_60 * (rules.long_play - rules.short_play),
        "pts_goals": e_goals * _by_position(et, rules.goals_scored),
        "pts_assists": e_assists * rules.assists,
        "pts_clean_sheet": p_60 * p_cs * _by_position(et, rules.clean_sheets),
        "pts_conceded": _by_position(et, rules.goals_conceded)
        * split_floor(lam_against, rules.goals_conceded_per_point),
        "pts_saves": rules.saves * saves_points,
        "pts_pen_saves": rules.penalties_saved * per_90("pen_save_rate"),
        "pts_bonus": rules.bonus * e_bonus,
        "pts_yellow": rules.yellow_cards * per_90("yellow_rate"),
        "pts_red": rules.red_cards * per_90("red_rate"),
        "pts_own_goals": rules.own_goals * per_90("own_goal_rate"),
        "pts_pen_miss": rules.penalties_missed
        * e_pen
        * frame["pen_miss_ratio"].to_numpy(dtype="float64"),
        # Defensive contributions: the Phase 5b hook (Task 10), 0 until then.
        "pts_defcon": np.full(len(frame), DEFCON),
    }
    xp = np.sum([points[name] for name in POINT_COLUMNS], axis=0) if len(frame) else np.zeros(0)
    return frame.assign(
        p_goal=_p_goal(e_goals, p_play),
        e_goals=e_goals,
        e_bonus=e_bonus,
        e_saves=np.where(keeper, saves_rate * exposure, 0.0),
        **points,
        xp=np.asarray(xp, dtype="float64"),
    )


def calibrate_xp(frame: pd.DataFrame, calibration: Calibration | None) -> pd.DataFrame:
    """Step 4 of the calibration: xp' = a · p_play + b · xp per position (unchanged without
    an xp map; positions missing from it are unchanged)."""
    if calibration is None or calibration.xp is None:
        return frame
    et = frame["element_type"].to_numpy(dtype="int64")
    lookup = {position: (a, b) for position, a, b in calibration.xp}
    coefficients = np.array([lookup.get(int(e), (0.0, 1.0)) for e in et], dtype="float64")
    a, b = coefficients.reshape(len(et), 2).T
    return frame.assign(xp=a * frame["p_play"].to_numpy() + b * frame["xp"].to_numpy())


def predict_fixtures(
    view: AsOfView, fit: V1Fit, calibration: Calibration | None | str = "season"
) -> pd.DataFrame:
    """Per player-fixture components and points at the view's deadline: calibrated with
    the season's table entry (`calibration="season"`, unless `fit.params.calibrate` is
    False), with the given `Calibration`, or not at all (None)."""
    season, _ = view.gameweek_for_deadline()
    if isinstance(calibration, str):
        calibration = calibration_for(season) if fit.params.calibrate else None
    frame = calibrate_inputs(fixture_components(view, fit), calibration)
    frame = fixture_points(frame, backtest_rules(season))
    return calibrate_xp(frame, calibration)


# --- per GW ---------------------------------------------------------------------------------


def gw_frame(view: AsOfView, fixtures: pd.DataFrame) -> pd.DataFrame:
    """The xP frame (module docstring) from per-fixture rows with `xp` (`fixture_points`)."""
    pool = player_pool(view)[["player_key", "team_key"]]
    upcoming = upcoming_fixtures(view)
    gameweeks = upcoming[["gw", "gw_index", "horizon"]].drop_duplicates()
    gameweeks = gameweeks[gameweeks["horizon"] <= HORIZON]
    grid = pool[["player_key"]].merge(gameweeks, how="cross")
    rows = fixtures.assign(
        p_cs=fixtures["p_60"] * fixtures["p_cs"],
        p_min_0=1.0 - fixtures["p_play"],
        p_min_1_59=np.clip(fixtures["p_play"] - fixtures["p_60"], 0.0, None),
        p_min_60=fixtures["p_60"],
    )
    keys = ["player_key", "gw", "gw_index", "horizon"]
    grouped = rows.groupby(keys, sort=True)
    sums = grouped[["xp", *GW_SUMS]].sum()
    firsts = grouped[list(GW_PROBABILITIES)].first()
    counts = grouped.size().rename("n")
    per_gw = sums.join(firsts).join(counts).reset_index()
    double = per_gw["n"].to_numpy() > 1
    for column in GW_PROBABILITIES:
        per_gw[column] = np.where(double, np.nan, per_gw[column].to_numpy(dtype="float64"))
    out = grid.merge(per_gw, on=keys, how="left", validate="one_to_one")
    blank = out["n"].isna().to_numpy()
    for column in ("xp", *GW_COMPONENTS):
        values = out[column].to_numpy(dtype="float64")
        out[column] = np.where(blank, 1.0 if column == "p_min_0" else 0.0, values)
    base = [name for name, _ in XP_DTYPES]
    out = out[[*base, *GW_COMPONENTS]].astype(
        {**dict(XP_DTYPES), **{c: "float64" for c in GW_COMPONENTS}}
    )
    return out.sort_values(["player_key", "horizon"], kind="mergesort").reset_index(drop=True)


def predict_v1(view: AsOfView, fit: V1Fit) -> pd.DataFrame:
    """The v1 xP frame at the view's deadline (module docstring)."""
    return gw_frame(view, predict_fixtures(view, fit))
