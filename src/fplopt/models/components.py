"""The remaining xP components: bonus, goalkeeper saves and penalty saves, cards, own goals
and penalty misses (PLAN §6.5; Phase 5 plan, *Other components* and Task 5). Defensive
contributions are Phase 5b (Task 10): see `DEFCON` below, the hook `fplopt.models.assemble`
reads.

`fit_components(view) -> ComponentsFit` fits on the rows visible in `view` (the caller passes
the view at the refit cutoff, `fplopt.models.fitted`); `predict_components(view, fit,
minutes, team, shares, shares_fit)` returns per row of `minutes` (pool player × horizon
fixture) the per-fixture *rates* the assembly turns into points (`COMPONENT_COLUMNS`). The
rates are per 90 minutes on the pitch or per event, so the assembly (and the calibration,
which can change minutes and goals) applies them to the expected minutes and events.

**Training rows.** `player_match` rows of the last `ComponentsParams.seasons` seasons
(the cutoff's season and the ones before it), with minutes > 0, the player's
`player_season` position, sorted by (player_key, season, fixture_key), weighted
`season_decay ** age` (age = the cutoff's season − the row's season; recent seasons count
more, as PLAN §6.5 asks for bonus, whose BPS rules changed in 2026/27: older seasons'
bonus is known drift, never dropped).

**Bonus** (`BonusFit`): per position a weighted least-squares regression of a played row's
bonus on indicators of the events that drive BPS (`BONUS_FEATURES`):
`play` (minutes > 0, i.e. the intercept), `sixty` (≥ 60), `goal1..3` (goals ≥ 1, 2, 3),
`assist1..2`, `cs` (FPL's clean-sheet stat) and, for goalkeepers, `saves3` = saves // 3
(zero elsewhere), with a small ridge (`BONUS_RIDGE`) so an all-zero column gets 0.
A 0-minute row has bonus 0 and every feature 0, so E[bonus] = β · E[features] holds for any
player-fixture: the assembly takes the expectation of each indicator under the expected
minutes and the Poisson event distributions (P(goals ≥ k) = p_play · P(Poisson(e_goals /
p_play) ≥ k), the clean sheet as p_60 · p_cs, E[saves // 3] under the saves Poisson). That
handles the nonlinearity of bonus in goals (a second goal adds less than the first; bonus
is capped at 3) exactly *within* the linear-in-indicators model; what it misses are the
interactions (a goal and a clean sheet in the same match, a teammate's or opponent's BPS),
which the indicators' expectations treat as independent. Fewer than `MIN_BONUS_ROWS` rows
of a position: all coefficients 0 for it.

**Saves** (`SavesFit`): goalkeeper rows whose opponent's pre-match market λ is visible
(`fplopt.models.team`: the match's own pre-match odds, de-vigged and solved for Dixon-Coles
λ, the same λ the team model uses at a deadline): saves ~ Poisson(minutes / 90 ·
exp(a + b · log λ_against)), weighted MLE (L-BFGS-B from a fixed start, analytic gradient).
A keeper's effect θ = (Σ w · saves + `keeper_prior`) / (Σ w · μ + `keeper_prior`) over his
rows (μ = the fitted league curve), i.e. shrunk toward 1 with `keeper_prior` pseudo-saves.
Without `MIN_SAVES_ROWS` rows with a market λ the curve is flat (b = 0, a = log of the
weighted saves per 90 of every goalkeeper row). Penalty saves per 90 on the pitch: the
weighted league rate over goalkeeper rows, shrunk toward `DEFAULT_PEN_SAVE_RATE`; at
prediction it is scaled by the opponent's penalty rate over the league's (from the shares
fit: `SharesFit.team_pen` / `pen_rate`).

**Cards and own goals** (`RatesFit`): per player per-90 rates of yellow cards, red cards
and own goals shrunk toward his position's league rate (over the same rows):
rate = (Σ w · events + k · position rate) / (Σ w · minutes / 90 + k), with k =
`yellow_prior`, `red_prior`, `own_goal_prior` pseudo-90s. Players without rows get the
position rate.

**Penalty misses**: no fit. FPL's `penalties_missed` counts every attempt not scored (saved
or missed), so E[misses] = E[attempts] · (1 − c) = E[penalty goals] · (1 − c) / c with c the
club's conversion from the shares fit (`SharesFit.team_pen`, else the league's); the
assembly multiplies `pen_miss_ratio` = (1 − c) / c by the shares' (possibly calibrated)
expected penalty goals, so misses stay consistent with the penalty goals.

**Prediction rows** (`predict_components`): the `minutes` frame's keys plus
`element_type` (from the pool), `opponent_team_key`, the saves rate per 90 on the pitch
(θ · exp(a + b · log λ_against); 0 for outfield players), the penalty-save rate per 90 (0 for
outfield players), the three card / own-goal rates per 90, `pen_miss_ratio` and the
position's bonus coefficients `b_<feature>`. Sorted by (player_key, horizon, fixture_key).

No state, no file access.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from fplopt.features.baseline import player_pool
from fplopt.features.store import AsOfView
from fplopt.models.shares import SharesFit
from fplopt.models.team import RHO, market_lambdas, market_probabilities, visible_odds

__all__ = (
    "BONUS_FEATURES",
    "COMPONENT_COLUMNS",
    "DEFCON",
    "BonusFit",
    "ComponentsFit",
    "ComponentsParams",
    "RatesFit",
    "SavesFit",
    "fit_bonus",
    "fit_components",
    "fit_rates",
    "fit_saves",
    "predict_components",
)

GOALKEEPER = 1
POSITIONS = (1, 2, 3, 4)
BONUS_FEATURES = ("play", "sixty", "goal1", "goal2", "goal3", "assist1", "assist2", "cs", "saves3")
BONUS_RIDGE = 1e-6  # × the total weight, added to X'WX's diagonal (an all-zero column gets 0)
MIN_BONUS_ROWS = 200
MIN_SAVES_ROWS = 200
DEFAULT_SAVES_PER_90 = 2.8
DEFAULT_PEN_SAVE_RATE = 0.02  # penalty saves per 90 on the pitch (EPL long run)
PEN_SAVE_PRIOR = 200.0  # pseudo goalkeeper-90s at DEFAULT_PEN_SAVE_RATE
SAVES_B_START = 0.5  # every saves fit starts at b = this, a = the flat rate at the mean log λ
LAMBDA_FLOOR = 0.05  # λ_against below this is clipped before taking logs
# Defensive contributions (Phase 5b, Task 10): no term yet. The assembly adds `DEFCON`
# points per player-fixture; develop/validate rules score no defcon anyway
# (`backtest_rules`: 2026-27 without defcon), so only 2025/26+ xP lacks it until then.
DEFCON = 0.0
KEYS = ("player_key", "fixture_key", "team_key", "season", "gw", "gw_index", "horizon")
RATE_COLUMNS = (
    "saves_rate",
    "pen_save_rate",
    "yellow_rate",
    "red_rate",
    "own_goal_rate",
    "pen_miss_ratio",
)
BONUS_COLUMNS = tuple(f"b_{name}" for name in BONUS_FEATURES)
COMPONENT_COLUMNS = (*KEYS, "element_type", "opponent_team_key", *RATE_COLUMNS, *BONUS_COLUMNS)
SORT_BY = ("player_key", "horizon", "fixture_key")
MATCH_COLUMNS = (
    "player_key",
    "season",
    "fixture_key",
    "team_key",
    "opponent_team_key",
    "was_home",
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "saves",
    "bonus",
    "yellow_cards",
    "red_cards",
    "own_goals",
    "penalties_saved",
)


@dataclass(frozen=True)
class ComponentsParams:
    """Component settings (not tuned beyond sanity checks on develop: the components are
    small parts of xP; PLAN §6.5)."""

    seasons: int = 4  # seasons of training rows (the cutoff's and the 3 before)
    season_decay: float = 0.7  # training weight per season of age
    keeper_prior: float = 30.0  # pseudo saves at the league curve (keeper effect)
    yellow_prior: float = 10.0  # pseudo 90s at the position rate
    red_prior: float = 60.0
    own_goal_prior: float = 60.0

    def __post_init__(self) -> None:
        if self.seasons < 1:
            raise ValueError("seasons must be >= 1")
        if not 0 < self.season_decay <= 1:
            raise ValueError("season_decay must be in (0, 1]")


@dataclass(frozen=True)
class BonusFit:
    """Per position (element_type, coefficients in `BONUS_FEATURES` order)."""

    coefficients: tuple[tuple[int, tuple[float, ...]], ...]
    n_rows: int


@dataclass(frozen=True)
class SavesFit:
    """log saves per 90 = a + b · log λ_against, times a keeper effect θ (`keepers`:
    (player_key, θ), sorted); penalty saves per 90 on the pitch."""

    a: float
    b: float
    keepers: tuple[tuple[int, float], ...]
    pen_save_rate: float
    n_rows: int


@dataclass(frozen=True)
class RatesFit:
    """Per-90 rates: per position (element_type, yellow, red, own goal) and per player
    (player_key, yellow, red, own goal), the player rates already shrunk."""

    positions: tuple[tuple[int, float, float, float], ...]
    players: tuple[tuple[int, float, float, float], ...]


@dataclass(frozen=True)
class ComponentsFit:
    cutoff: pd.Timestamp
    params: ComponentsParams
    bonus: BonusFit
    saves: SavesFit
    rates: RatesFit


# --- training rows ----------------------------------------------------------------------------


def _floats(frame: pd.DataFrame, column: str) -> np.ndarray:
    return frame[column].astype("Float64").to_numpy(dtype="float64", na_value=np.nan)


def training_rows(view: AsOfView, params: ComponentsParams) -> pd.DataFrame:
    """Played `player_match` rows of the last `params.seasons` seasons visible in `view`
    with element_type, float stats (NaN → 0) and `weight`; sorted by key."""
    season, _ = view.gameweek_for_deadline()
    rows = view.table("player_match", columns=list(MATCH_COLUMNS))
    rows = rows[rows["season"] > season - params.seasons]
    stats = MATCH_COLUMNS[6:]
    rows = rows.assign(**{c: np.nan_to_num(_floats(rows, c)) for c in stats})
    rows = rows[rows["minutes"] > 0]
    positions = view.table("player_season", columns=["player_key", "season", "element_type"])
    positions = positions.drop_duplicates(["player_key", "season"], keep="last")
    rows = rows.merge(positions, on=["player_key", "season"], how="inner", validate="many_to_one")
    age = (season - rows["season"]).clip(lower=0).to_numpy(dtype="float64")
    rows = rows.assign(
        weight=np.power(params.season_decay, age),
        element_type=rows["element_type"].astype("int64"),
    )
    rows = rows.sort_values(["player_key", "season", "fixture_key"], kind="mergesort")
    return rows.reset_index(drop=True)


# --- bonus ----------------------------------------------------------------------------------


def bonus_features(rows: pd.DataFrame) -> np.ndarray:
    """[n, len(BONUS_FEATURES)] realized indicators of played rows (minutes, goals_scored,
    assists, clean_sheets, saves, element_type)."""
    minutes = rows["minutes"].to_numpy(dtype="float64")
    goals = rows["goals_scored"].to_numpy(dtype="float64")
    assists = rows["assists"].to_numpy(dtype="float64")
    keeper = rows["element_type"].to_numpy() == GOALKEEPER
    columns = [
        minutes > 0,
        minutes >= 60,
        goals >= 1,
        goals >= 2,
        goals >= 3,
        assists >= 1,
        assists >= 2,
        rows["clean_sheets"].to_numpy(dtype="float64") >= 1,
        np.where(keeper, np.floor(rows["saves"].to_numpy(dtype="float64") / 3), 0.0),
    ]
    return np.column_stack([np.asarray(c, dtype="float64") for c in columns])


def fit_bonus(rows: pd.DataFrame) -> BonusFit:
    """Per position weighted ridge least squares of `bonus` on `bonus_features` (rows:
    `training_rows` output, sorted)."""
    out = []
    for position in POSITIONS:
        part = rows[rows["element_type"] == position]
        if len(part) < MIN_BONUS_ROWS:
            out.append((position, (0.0,) * len(BONUS_FEATURES)))
            continue
        x = bonus_features(part)
        w = part["weight"].to_numpy(dtype="float64")
        y = part["bonus"].to_numpy(dtype="float64")
        gram = x.T @ (x * w[:, None])
        gram += np.eye(len(BONUS_FEATURES)) * BONUS_RIDGE * w.sum()
        beta = np.linalg.solve(gram, x.T @ (w * y))
        out.append((position, tuple(float(v) for v in beta)))
    return BonusFit(tuple(out), len(rows))


# --- saves ----------------------------------------------------------------------------------


def _opponent_lambdas(view: AsOfView, rows: pd.DataFrame) -> np.ndarray:
    """Per row the opponent's market λ from the fixture's own pre-match odds visible in
    the view (NaN without odds or a converged solve)."""
    keys = pd.Series(rows["fixture_key"].unique()).sort_values().reset_index(drop=True)
    odds = visible_odds(view, keys)
    if odds.empty:
        return np.full(len(rows), np.nan)
    solved = market_lambdas(market_probabilities(odds), RHO)
    solved = solved[solved["success"]].set_index("fixture_key")
    home = rows["fixture_key"].map(solved["lambda_home"]).to_numpy(dtype="float64")
    away = rows["fixture_key"].map(solved["lambda_away"]).to_numpy(dtype="float64")
    was_home = rows["was_home"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    return np.where(was_home, away, home)


def fit_saves(rows: pd.DataFrame, keeper_prior: float) -> SavesFit:
    """The saves curve and keeper effects from goalkeeper rows (`training_rows` output
    restricted to element_type 1, with `lambda_against`: NaN = unknown), and the penalty
    save rate (module docstring)."""
    keepers = rows[rows["element_type"] == GOALKEEPER]
    w_all = keepers["weight"].to_numpy(dtype="float64")
    frac_all = keepers["minutes"].to_numpy(dtype="float64") / 90.0
    pen_saves = float((w_all * keepers["penalties_saved"].to_numpy(dtype="float64")).sum())
    exposure = float((w_all * frac_all).sum())
    pen_save_rate = (pen_saves + PEN_SAVE_PRIOR * DEFAULT_PEN_SAVE_RATE) / (
        exposure + PEN_SAVE_PRIOR
    )
    saves_all = float((w_all * keepers["saves"].to_numpy(dtype="float64")).sum())
    flat = math.log(saves_all / exposure) if saves_all > 0 and exposure > 0 else None
    flat = math.log(DEFAULT_SAVES_PER_90) if flat is None else flat

    known = keepers[np.isfinite(keepers["lambda_against"].to_numpy(dtype="float64"))]
    if len(known) < MIN_SAVES_ROWS:
        return SavesFit(flat, 0.0, (), pen_save_rate, len(known))
    w = known["weight"].to_numpy(dtype="float64")
    frac = known["minutes"].to_numpy(dtype="float64") / 90.0
    saves = known["saves"].to_numpy(dtype="float64")
    lam = np.clip(known["lambda_against"].to_numpy(dtype="float64"), LAMBDA_FLOOR, None)
    log_lambda = np.log(lam)
    scale = float(w.sum())

    def loss(theta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = theta[0] + theta[1] * log_lambda
        mu = frac * np.exp(eta)
        value = float(np.sum(w * (mu - saves * eta))) / scale
        residual = w * (mu - saves) / scale
        return value, np.array([residual.sum(), (residual * log_lambda).sum()])

    mean_log_lambda = float((w * log_lambda).sum()) / scale
    start = np.array([flat - SAVES_B_START * mean_log_lambda, SAVES_B_START])
    result = minimize(loss, start, jac=True, method="L-BFGS-B", options={"maxiter": 200})
    a, b = (float(v) for v in result.x)
    mu = frac * np.exp(a + b * log_lambda)
    totals = (
        pd.DataFrame({"player_key": known["player_key"].to_numpy(), "s": w * saves, "m": w * mu})
        .groupby("player_key", sort=True)[["s", "m"]]
        .sum()
    )
    theta = (totals["s"] + keeper_prior) / (totals["m"] + keeper_prior)
    effects = tuple((int(k), float(v)) for k, v in theta.items())
    return SavesFit(a, b, effects, pen_save_rate, len(known))


# --- cards and own goals --------------------------------------------------------------------

RATE_EVENTS = ("yellow_cards", "red_cards", "own_goals")


def fit_rates(rows: pd.DataFrame, params: ComponentsParams) -> RatesFit:
    """Position and shrunk player per-90 rates of `RATE_EVENTS` (`training_rows` output)."""
    priors = (params.yellow_prior, params.red_prior, params.own_goal_prior)
    w = rows["weight"].to_numpy(dtype="float64")
    frame = pd.DataFrame(
        {
            "player_key": rows["player_key"].to_numpy(),
            "element_type": rows["element_type"].to_numpy(),
            "season": rows["season"].to_numpy(),
            "exposure": w * rows["minutes"].to_numpy(dtype="float64") / 90.0,
            **{e: w * rows[e].to_numpy(dtype="float64") for e in RATE_EVENTS},
        }
    )
    by_position = frame.groupby("element_type", sort=True)[["exposure", *RATE_EVENTS]].sum()
    positions = []
    for position in POSITIONS:
        if position in by_position.index and by_position.loc[position, "exposure"] > 0:
            row = by_position.loc[position]
            positions.append((position, *(float(row[e] / row["exposure"]) for e in RATE_EVENTS)))
        else:
            positions.append((position, 0.0, 0.0, 0.0))
    position_rates = {p[0]: p[1:] for p in positions}
    # The player's position: that of his newest row.
    latest = frame.drop_duplicates("player_key", keep="last").set_index("player_key")
    sums = frame.groupby("player_key", sort=True)[["exposure", *RATE_EVENTS]].sum()
    players = []
    for key, row in sums.iterrows():
        base = position_rates[int(latest.loc[key, "element_type"])]
        players.append(
            (
                int(key),
                *(
                    float((row[e] + k * m) / (row["exposure"] + k))
                    for e, k, m in zip(RATE_EVENTS, priors, base, strict=True)
                ),
            )
        )
    return RatesFit(tuple(positions), tuple(players))


# --- fit and predict ------------------------------------------------------------------------


def fit_components(view: AsOfView, params: ComponentsParams | None = None) -> ComponentsFit:
    """Bonus, saves and card/own-goal fits on the rows visible in `view` (module docstring)."""
    params = ComponentsParams() if params is None else params
    rows = training_rows(view, params)
    keepers = rows[rows["element_type"] == GOALKEEPER].reset_index(drop=True)
    keepers = keepers.assign(lambda_against=_opponent_lambdas(view, keepers))
    return ComponentsFit(
        cutoff=view.deadline,
        params=params,
        bonus=fit_bonus(rows),
        saves=fit_saves(keepers, params.keeper_prior),
        rates=fit_rates(rows, params),
    )


def predict_components(
    view: AsOfView,
    fit: ComponentsFit,
    minutes: pd.DataFrame,
    team: pd.DataFrame,
    shares_fit: SharesFit,
) -> pd.DataFrame:
    """Per row of `minutes` (`predict_minutes` / `adjust_minutes` output; `team` is the
    `team_lambdas` frame of the same view): `COMPONENT_COLUMNS` (module docstring)."""
    pool = player_pool(view)[["player_key", "element_type"]]
    frame = minutes[list(KEYS)].merge(pool, on="player_key", how="left", validate="many_to_one")
    sides = team[["fixture_key", "team_key", "opponent_team_key", "lambda_against"]]
    frame = frame.merge(sides, on=["fixture_key", "team_key"], how="left", validate="many_to_one")
    element_type = frame["element_type"].fillna(0).to_numpy(dtype="int64")
    keeper = element_type == GOALKEEPER

    saves = fit.saves
    theta = frame["player_key"].map(dict(saves.keepers)).fillna(1.0).to_numpy(dtype="float64")
    lam = np.clip(frame["lambda_against"].to_numpy(dtype="float64"), LAMBDA_FLOOR, None)
    saves_rate = np.where(keeper, theta * np.exp(saves.a + saves.b * np.log(lam)), 0.0)
    saves_rate = np.nan_to_num(saves_rate)  # no team λ: no saves

    club_pen = {key: (rate, conversion) for key, rate, conversion in shares_fit.team_pen}
    opponents = frame["opponent_team_key"].fillna(-1).to_numpy(dtype="int64")
    opp_rate = np.array(
        [club_pen.get(int(t), (shares_fit.pen_rate, 0.0))[0] for t in opponents], dtype="float64"
    )
    league_rate = shares_fit.pen_rate if shares_fit.pen_rate > 0 else 1.0
    pen_save_rate = np.where(keeper, saves.pen_save_rate * opp_rate / league_rate, 0.0)
    clubs = frame["team_key"].to_numpy(dtype="int64")
    conversion = np.array(
        [club_pen.get(int(t), (0.0, shares_fit.conversion))[1] for t in clubs], dtype="float64"
    )
    conversion = np.clip(conversion, 1e-3, 1.0)

    position_rates = {p[0]: p[1:] for p in fit.rates.positions}
    player_rates = {p[0]: p[1:] for p in fit.rates.players}
    rates = np.array(
        [
            player_rates.get(int(k), position_rates.get(int(e), (0.0, 0.0, 0.0)))
            for k, e in zip(frame["player_key"].to_numpy(), element_type, strict=True)
        ],
        dtype="float64",
    ).reshape(len(frame), 3)

    coefficients = dict(fit.bonus.coefficients)
    zero = (0.0,) * len(BONUS_FEATURES)
    beta = np.array(
        [coefficients.get(int(e), zero) for e in element_type], dtype="float64"
    ).reshape(len(frame), len(BONUS_FEATURES))

    out = frame[list(KEYS)].assign(
        element_type=element_type,
        opponent_team_key=opponents,
        saves_rate=saves_rate,
        pen_save_rate=pen_save_rate,
        yellow_rate=rates[:, 0],
        red_rate=rates[:, 1],
        own_goal_rate=rates[:, 2],
        pen_miss_ratio=(1.0 - conversion) / conversion,
        **{column: beta[:, i] for i, column in enumerate(BONUS_COLUMNS)},
    )
    dtypes = {name: "int64" for name in (*KEYS, "element_type", "opponent_team_key")}
    dtypes |= {name: "float64" for name in (*RATE_COLUMNS, *BONUS_COLUMNS)}
    out = out[list(COMPONENT_COLUMNS)].astype(dtypes)
    return out.sort_values(list(SORT_BY), kind="mergesort").reset_index(drop=True)
