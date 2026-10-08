"""Team model: per-fixture expected goals and clean-sheet probabilities (PLAN §6.1; Phase 5
plan, *Team model* and Task 2).

**Market λ** (fixtures with pre-match odds visible at the view's deadline):
- Odds: per fixture the newest visible pre-match snapshot (`is_closing` False; closing prices
  are only known at kickoff). football-data's market average (`avg`) when its h2h is
  complete (historically every EPL match since 2016/17, visible for the target GW only),
  else the newest Odds API snapshot (live 2026/27): each bookmaker is de-vigged on its own,
  then the median probability per outcome is taken and renormalized. Odds API rows carry no
  Asian handicap.
- De-vig: the power method (`devig_power`: k with Σ (1/odds_i)^k = 1), or Shin's method
  (`devig_shin`), per market: 1X2, O/U 2.5 and the Asian handicap's two sides.
- λ_home, λ_away: weighted least squares (`scipy.optimize.least_squares`, fixed start, fixed
  tolerances, analytic Jacobian, λ in `LAMBDA_RANGE`)
  of the Dixon-Coles score distribution (`dc_matrix`: goals 0..`MAX_GOALS` per side,
  low-score correction ρ = `RHO`, renormalized) against the de-vigged home/draw/away,
  over/under and, when present, both AH sides.
- Asian handicap (`line` = the home side's handicap h): a quarter line (h·4 odd) is half
  the stake on h − 0.25 and half on h + 0.25; on each part the home bet wins if goal
  difference + part > 0, is pushed (stake back) at 0 and loses below. With W / L the
  expected share of the stake won / lost, the fair home price is 1 + L/W, so the model's
  counterpart of the de-vigged home probability is W / (W + L). Half, whole and quarter
  lines are exact under this rule.

**Ratings** (horizon fixtures without visible odds):
`log λ_home = base + home + attack[home] − defence[away]`,
`log λ_away = base + attack[away] − defence[home]`, with
`attack[t] = attack_slope · z[t] + u[t]` and `defence[t] = defence_slope · z[t] + v[t]`:
z = the club's Elo (`rating_after` of its newest visible `team_rating` row, the current
rating) minus the mean Elo of the newest season's clubs, over `ELO_SCALE`. The deviations
u, v carry a ridge penalty `prior_strength / 2 · Σ (u² + v²)`, so a club with little or no
history (promoted) sits at its Elo prior. Fitted by time-decayed (half-life
`half_life_days`, rows older than `MIN_WEIGHT` dropped) weighted quasi-Poisson MLE
(`scipy.optimize.minimize`, L-BFGS-B, analytic gradient, fixed start) on every finished
match visible in the view. Target per side: `w · market λ + (1 − w) · stats` with
stats = `XG_WEIGHT · xG + (1 − XG_WEIGHT) · goals` (xG including penalties: Understat
`us_xg`, else summed FPL `fpl_xg`, else football-data `fd_xg`; goals alone without any), and
the market λ from the match's own pre-match odds (the stats target alone where it has none).
The **stats fallback** is the same fit with w = 0, used when no odds are visible at all
(`TeamFit.source` 'stats').

**P(CS)** = P(the opponent scores 0) under Dixon-Coles with (λ_home, λ_away) and `RHO`.

API: `fit_team(view) -> TeamFit` (the walk-forward fit at a refit cutoff;
`fplopt.models.fitted`) and `team_lambdas(view, fit) -> frame` (one row per fixture and
side of `upcoming_fixtures(view)`, `TEAM_DTYPES`). The module keeps no state.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize
from scipy.special import gammaln

from fplopt.features.baseline import upcoming_fixtures
from fplopt.features.store import AsOfView

log = logging.getLogger(__name__)

MAX_GOALS = 10  # per side in the score matrix (the tail beyond is renormalized away)
# Dixon-Coles low-score correction, fitted once on develop (2016/17–2022/23, 2,660 matches
# with pre-match odds): the argmax of the realized scores' log-likelihood under each
# fixture's market λ, re-solved for every ρ on a grid (step 0.005 near the maximum). The
# profile is flat: −0.03 beats ρ = 0 by 0.6 log-likelihood in total and −0.10 by 4.1.
# Reproduce with `uv run python dev/team_rho.py` (2026-10-08).
RHO = -0.03
TOTALS_LINE = 2.5
LAMBDA_RANGE = (0.05, 8.0)  # a market λ outside it counts as not converged
MARKET_START = (1.4, 1.1)  # (λ_home, λ_away) where every market solve starts
# Least-squares weights per de-vigged probability (each side of each market).
WEIGHT_1X2 = 1.0
WEIGHT_TOTALS = 1.0
WEIGHT_AH = 1.0
LSQ_TOL = 1e-12
XG_WEIGHT = 0.7  # stats target = XG_WEIGHT · xG + (1 − XG_WEIGHT) · goals
ELO_SCALE = 100.0
MIN_WEIGHT = 0.01  # training rows whose decay weight is below this are dropped
MAX_WINDOW_DAYS = 365.25 * 50  # match_history's look-back cap (any half-life)
FIT_GTOL = 1e-9
FIT_MAXITER = 2000
DEVIG_METHODS = ("power", "shin")
BISECTION_STEPS = 100
OUTCOMES_1X2 = ("home", "draw", "away")
ODDS_COLUMNS = (
    "fixture_key",
    "source",
    "bookmaker",
    "market",
    "outcome",
    "line",
    "price",
    "is_closing",
    "snapshot_at",
)
MARKET_COLUMNS = ("p_home", "p_draw", "p_away", "p_over", "ah_line", "q_ah_home")
XG_SOURCES = ("us_xg", "fpl_xg", "fd_xg")  # first non-null per side wins
TEAM_DTYPES = (
    ("fixture_key", "int64"),
    ("season", "int64"),
    ("gw", "int64"),
    ("gw_index", "int64"),
    ("horizon", "int64"),
    ("team_key", "int64"),
    ("opponent_team_key", "int64"),
    ("is_home", "bool"),
    ("lambda_for", "float64"),
    ("lambda_against", "float64"),
    ("p_cs", "float64"),
    ("source", "str"),
)
SOURCES = ("market", "ratings", "stats")
__all__ = (
    "MAX_GOALS",
    "RHO",
    "TEAM_DTYPES",
    "TeamFit",
    "TeamParams",
    "dc_matrix",
    "devig",
    "devig_power",
    "devig_shin",
    "fit_ratings",
    "fit_team",
    "market_lambdas",
    "market_probabilities",
    "match_history",
    "ratings_lambdas",
    "team_lambdas",
    "visible_odds",
)


@dataclass(frozen=True)
class TeamParams:
    """Team-model settings. `half_life_days` and `market_weight` (w) are tuned on develop
    (dev/team_model_eval.py); the rest are fixed."""

    half_life_days: float = 240.0
    market_weight: float = 0.75
    prior_strength: float = 20.0
    rho: float = RHO
    devig: str = "power"

    def __post_init__(self) -> None:
        if self.devig not in DEVIG_METHODS:
            raise ValueError(f"devig must be one of {DEVIG_METHODS}, got {self.devig!r}")
        if not 0.0 <= self.market_weight <= 1.0:
            raise ValueError(f"market_weight must be in [0, 1], got {self.market_weight}")
        if self.half_life_days <= 0 or self.prior_strength <= 0:
            raise ValueError("half_life_days and prior_strength must be > 0")


@dataclass(frozen=True)
class TeamFit:
    """Ratings fitted at `cutoff` (the view's deadline). `teams` is sorted; `attack` /
    `defence` are aligned with it (Elo prior included). `source` is 'ratings' when market
    λ entered the targets, 'stats' when no odds were visible (w = 0). A club missing from
    `teams` is rated from its Elo at prediction time: `slope · (elo − elo_center) / ELO_SCALE`."""

    cutoff: pd.Timestamp
    params: TeamParams
    source: str
    base: float
    home: float
    attack_slope: float
    defence_slope: float
    elo_center: float
    teams: tuple[int, ...]
    attack: tuple[float, ...]
    defence: tuple[float, ...]
    n_matches: int
    n_market: int


# --- de-vig -------------------------------------------------------------------------------


def _implied(prices: np.ndarray | pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(1 / prices with invalid rows set to 0.5, valid-row mask); a row is valid when every
    price is finite and > 0."""
    prices = np.asarray(prices, dtype="float64")
    if prices.ndim != 2:
        raise ValueError(f"prices must be 2-D (rows x outcomes), got shape {prices.shape}")
    valid = (np.isfinite(prices) & (prices > 0)).all(axis=1)
    safe = np.where(valid[:, None], prices, 2.0)
    return 1.0 / safe, valid


def devig_power(prices: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Power method: per row, probabilities (1/odds_i)^k with k such that they sum to 1
    (bisection on k ∈ (0, 50]). Rows with a missing or non-positive price are NaN."""
    implied, valid = _implied(prices)
    lo = np.full(len(implied), 1e-6)
    hi = np.full(len(implied), 50.0)
    for _ in range(BISECTION_STEPS):
        mid = (lo + hi) / 2
        over = np.power(implied, mid[:, None]).sum(axis=1) > 1.0  # sum falls as k grows
        lo = np.where(over, mid, lo)
        hi = np.where(over, hi, mid)
    out = np.power(implied, ((lo + hi) / 2)[:, None])
    out = out / out.sum(axis=1, keepdims=True)
    return np.where(valid[:, None], out, np.nan)


def _shin(implied: np.ndarray, z: np.ndarray) -> np.ndarray:
    beta = implied.sum(axis=1, keepdims=True)
    z = z[:, None]
    return (np.sqrt(z**2 + 4 * (1 - z) * implied**2 / beta) - z) / (2 * (1 - z))


def devig_shin(prices: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Shin's method: p_i = (√(z² + 4(1 − z) π_i² / Σπ) − z) / (2(1 − z)) with π = 1/odds and
    the insider share z ∈ [0, 1) such that Σ p = 1 (bisection). Rows without a margin
    (Σπ ≤ 1, exchanges) are normalized proportionally. Invalid rows are NaN."""
    implied, valid = _implied(prices)
    margin = implied.sum(axis=1) > 1.0
    lo = np.zeros(len(implied))
    hi = np.full(len(implied), 1.0 - 1e-9)
    for _ in range(BISECTION_STEPS):
        mid = (lo + hi) / 2
        over = _shin(implied, mid).sum(axis=1) > 1.0  # the sum falls as z grows
        lo = np.where(over, mid, lo)
        hi = np.where(over, hi, mid)
    out = _shin(implied, (lo + hi) / 2)
    out = np.where(margin[:, None], out, implied)
    out = out / out.sum(axis=1, keepdims=True)
    return np.where(valid[:, None], out, np.nan)


def devig(prices: np.ndarray | pd.DataFrame, method: str = "power") -> np.ndarray:
    """De-vig rows of decimal odds (one column per outcome of one market)."""
    if method == "power":
        return devig_power(prices)
    if method == "shin":
        return devig_shin(prices)
    raise ValueError(f"unknown de-vig method {method!r}; known: {DEVIG_METHODS}")


# --- Dixon-Coles ------------------------------------------------------------------------


def dc_matrix(
    lambda_home: np.ndarray | float,
    lambda_away: np.ndarray | float,
    rho: float = RHO,
    max_goals: int = MAX_GOALS,
) -> np.ndarray:
    """Score probabilities [n, home goals, away goals] (goals 0..max_goals; a scalar pair
    gives n = 1): independent Poisson times the Dixon-Coles correction τ (0-0: 1 − λμρ,
    0-1: 1 + λρ, 1-0: 1 + μρ, 1-1: 1 − ρ; negative cells clipped to 0), renormalized to
    sum to 1."""
    lh = np.atleast_1d(np.asarray(lambda_home, dtype="float64"))
    la = np.atleast_1d(np.asarray(lambda_away, dtype="float64"))
    goals = np.arange(max_goals + 1, dtype="float64")
    log_fact = gammaln(goals + 1)
    ph = np.exp(goals * np.log(lh)[:, None] - lh[:, None] - log_fact)
    pa = np.exp(goals * np.log(la)[:, None] - la[:, None] - log_fact)
    m = ph[:, :, None] * pa[:, None, :]
    m[:, 0, 0] *= 1 - lh * la * rho
    m[:, 0, 1] *= 1 + lh * rho
    m[:, 1, 0] *= 1 + la * rho
    m[:, 1, 1] *= 1 - rho
    m = np.clip(m, 0.0, None)
    return m / m.sum(axis=(1, 2), keepdims=True)


def _goal_grids(max_goals: int = MAX_GOALS) -> tuple[np.ndarray, np.ndarray]:
    """(goal difference home − away, total goals) over the score matrix."""
    goals = np.arange(max_goals + 1, dtype="float64")
    return goals[:, None] - goals[None, :], goals[:, None] + goals[None, :]


def ah_stakes(line: float, max_goals: int = MAX_GOALS) -> tuple[np.ndarray, np.ndarray]:
    """(won, lost) share of a home Asian-handicap stake per score, for home handicap `line`.
    Quarter lines split the stake over line ± 0.25; a part with margin 0 is pushed."""
    if (line * 4) % 1 != 0:
        raise ValueError(f"Asian handicap lines are multiples of 0.25, got {line}")
    diff, _ = _goal_grids(max_goals)
    parts = (line - 0.25, line + 0.25) if (line * 4) % 2 == 1 else (line,)
    share = 1.0 / len(parts)
    won = sum(share * (diff + part > 0) for part in parts)
    lost = sum(share * (diff + part < 0) for part in parts)
    return np.asarray(won, dtype="float64"), np.asarray(lost, dtype="float64")


def ah_home_probability(matrix: np.ndarray, line: float) -> np.ndarray:
    """The de-vigged home AH probability a score matrix [n, i, j] implies: W / (W + L)."""
    won, lost = ah_stakes(line, matrix.shape[-1] - 1)
    w = (matrix * won).sum(axis=(-2, -1))
    lo = (matrix * lost).sum(axis=(-2, -1))
    return w / (w + lo)


def _dc_with_gradient(
    x: np.ndarray, rho: float, max_goals: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unnormalized DC matrix B at (log λ_home, log λ_away) = x and dB/dx[0], dB/dx[1]."""
    lh, la = math.exp(x[0]), math.exp(x[1])
    goals = np.arange(max_goals + 1, dtype="float64")
    log_fact = gammaln(goals + 1)
    ph = np.exp(goals * x[0] - lh - log_fact)
    pa = np.exp(goals * x[1] - la - log_fact)
    base = np.outer(ph, pa)
    tau = np.ones_like(base)
    tau[0, 0] = 1 - lh * la * rho
    tau[0, 1] = 1 + lh * rho
    tau[1, 0] = 1 + la * rho
    tau[1, 1] = 1 - rho
    b = base * tau
    db_h = b * (goals - lh)[:, None]
    db_h[0, 0] += base[0, 0] * (-lh * la * rho)
    db_h[0, 1] += base[0, 1] * (lh * rho)
    db_a = b * (goals - la)[None, :]
    db_a[0, 0] += base[0, 0] * (-lh * la * rho)
    db_a[1, 0] += base[1, 0] * (la * rho)
    return b, db_h, db_a


def _solve_market(
    target: np.ndarray,
    events: np.ndarray,
    ah: tuple[np.ndarray, np.ndarray] | None,
    weights: np.ndarray,
    rho: float,
    max_goals: int,
) -> tuple[float, float, bool]:
    """Least-squares (log λ_home, log λ_away) for one fixture: linear events (probability
    masks [e, i, j]) and optionally the AH pair (won, lost stakes) against `target`
    (events' probabilities, then the AH home and away probabilities)."""
    sqrt_w = np.sqrt(weights)
    log_lo, log_hi = math.log(LAMBDA_RANGE[0]), math.log(LAMBDA_RANGE[1])
    n_events = len(events)
    # Rows: the linear events, then the total mass, then the AH won / lost stakes.
    masks = [events.reshape(n_events, -1), np.ones((1, events[0].size))]
    if ah is not None:
        masks.append(np.stack([ah[0].ravel(), ah[1].ravel()]))
    masks = np.concatenate(masks)
    evaluated: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}  # this solve's x -> model

    def model(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        key = x.tobytes()
        if key not in evaluated:
            evaluated.clear()
            b, db_h, db_a = _dc_with_gradient(np.clip(x, log_lo, log_hi), rho, max_goals)
            s = masks @ b.ravel()
            ds = masks @ np.stack([db_h.ravel(), db_a.ravel()], axis=1)
            z, dz = s[n_events], ds[n_events]
            p = s[:n_events] / z
            dp = (ds[:n_events] - p[:, None] * dz[None, :]) / z
            if ah is not None:
                w, lo = s[n_events + 1], s[n_events + 2]
                dw, dl = ds[n_events + 1], ds[n_events + 2]
                q = w / (w + lo)
                dq = (dw * lo - w * dl) / (w + lo) ** 2
                p = np.concatenate([p, [q, 1 - q]])
                dp = np.vstack([dp, dq, -dq])
            evaluated[key] = (sqrt_w * (p - target), sqrt_w[:, None] * dp)
        return evaluated[key]

    def residuals(x: np.ndarray) -> np.ndarray:
        return model(x)[0]

    def jacobian(x: np.ndarray) -> np.ndarray:
        return model(x)[1]

    result = least_squares(
        residuals,
        np.log(MARKET_START),
        jac=jacobian,
        method="lm",
        ftol=LSQ_TOL,
        xtol=LSQ_TOL,
        gtol=LSQ_TOL,
        max_nfev=200,
    )
    inside = log_lo <= result.x.min() and result.x.max() <= log_hi
    ok = bool(result.success) and bool(np.isfinite(result.x).all()) and bool(inside)
    return math.exp(result.x[0]), math.exp(result.x[1]), ok


def market_lambdas(
    markets: pd.DataFrame, rho: float = RHO, max_goals: int = MAX_GOALS
) -> pd.DataFrame:
    """Per fixture of `markets` (fixture_key and MARKET_COLUMNS: de-vigged p_home, p_draw,
    p_away, p_over (O/U 2.5) and the AH home line with its de-vigged home probability;
    NaN = market missing): λ_home, λ_away fitted to them (`_solve_market`) and `success`.
    Rows without complete 1X2 get NaN. Sorted by fixture_key."""
    diff, total = _goal_grids(max_goals)
    win, draw, loss = (diff > 0) * 1.0, (diff == 0) * 1.0, (diff < 0) * 1.0
    over = (total > TOTALS_LINE) * 1.0
    rows = markets.sort_values("fixture_key", kind="mergesort").reset_index(drop=True)
    lh = np.full(len(rows), np.nan)
    la = np.full(len(rows), np.nan)
    ok = np.zeros(len(rows), dtype=bool)
    values = rows[list(MARKET_COLUMNS)].to_numpy(dtype="float64", na_value=np.nan)
    for i, (p_home, p_draw, p_away, p_over, line, q_home) in enumerate(values):
        if not np.isfinite([p_home, p_draw, p_away]).all():
            continue
        events = [win, draw, loss]
        target = [p_home, p_draw, p_away]
        weights = [WEIGHT_1X2] * 3
        if np.isfinite(p_over):
            events += [over, 1.0 - over]
            target += [p_over, 1.0 - p_over]
            weights += [WEIGHT_TOTALS] * 2
        ah = None
        if np.isfinite(line) and np.isfinite(q_home):
            ah = ah_stakes(float(line), max_goals)
            target += [q_home, 1.0 - q_home]
            weights += [WEIGHT_AH] * 2
        lh[i], la[i], ok[i] = _solve_market(
            np.asarray(target), np.stack(events), ah, np.asarray(weights), rho, max_goals
        )
    return pd.DataFrame(
        {
            "fixture_key": rows["fixture_key"].astype("int64"),
            "lambda_home": lh,
            "lambda_away": la,
            "success": ok,
        }
    )


# --- odds ---------------------------------------------------------------------------------


def visible_odds(view: AsOfView, fixture_keys: pd.Series | list[int]) -> pd.DataFrame:
    """Pre-match odds rows of `fixture_keys` visible in the view, one snapshot per fixture:
    football-data `avg` (newest snapshot) where its h2h is complete, else the newest
    Odds API snapshot (every bookmaker in it)."""
    odds = view.table("odds_snapshot", columns=list(ODDS_COLUMNS))
    odds = odds[odds["fixture_key"].isin(pd.Series(fixture_keys)) & ~odds["is_closing"]]
    fd = odds[(odds["source"] == "football-data") & (odds["bookmaker"] == "avg")]
    fd = fd[fd["snapshot_at"] == fd.groupby("fixture_key")["snapshot_at"].transform("max")]
    complete = fd[(fd["market"] == "h2h") & fd["outcome"].isin(OUTCOMES_1X2)]
    complete = complete.groupby("fixture_key")["outcome"].nunique()
    fd = fd[fd["fixture_key"].isin(complete.index[complete == len(OUTCOMES_1X2)])]
    api = odds[(odds["source"] == "odds-api") & ~odds["fixture_key"].isin(fd["fixture_key"])]
    api = api[api["snapshot_at"] == api.groupby("fixture_key")["snapshot_at"].transform("max")]
    return pd.concat([fd, api], ignore_index=True)


def _devig_market(
    odds: pd.DataFrame, market: str, outcomes: tuple[str, ...], method: str
) -> pd.DataFrame:
    """Per fixture: the median over bookmakers of each outcome's de-vigged probability,
    renormalized; for 'ah' per (fixture, line), on the fixture's most quoted line."""
    rows = odds[(odds["market"] == market) & odds["outcome"].isin(outcomes)]
    keys = ["fixture_key", "bookmaker"] + (["line"] if market == "ah" else [])
    prices = rows.pivot_table("price", keys, "outcome", aggfunc="median").reindex(
        columns=list(outcomes)
    )
    prices = prices[prices.notna().all(axis=1)]
    probabilities = pd.DataFrame(
        devig(prices.to_numpy(dtype="float64"), method), index=prices.index, columns=outcomes
    )
    probabilities = probabilities.reset_index()
    if market == "ah":
        counts = probabilities.groupby(["fixture_key", "line"]).size().rename("n").reset_index()
        counts = counts.assign(abs_line=counts["line"].abs())
        counts = counts.sort_values(
            ["fixture_key", "n", "abs_line", "line"],
            ascending=[True, False, True, True],
            kind="mergesort",
        ).drop_duplicates("fixture_key")
        probabilities = probabilities.merge(counts[["fixture_key", "line"]], how="inner")
    group = ["fixture_key"] + (["line"] if market == "ah" else [])
    median = probabilities.groupby(group)[list(outcomes)].median()
    median = median.div(median.sum(axis=1), axis=0)
    return median.reset_index()


def market_probabilities(odds: pd.DataFrame, method: str = "power") -> pd.DataFrame:
    """Per fixture with complete 1X2 in `odds` (rows: fixture_key, bookmaker, market,
    outcome, line, price; one snapshot per bookmaker): MARKET_COLUMNS (NaN where the
    market is missing). Sorted by fixture_key."""
    h2h = _devig_market(odds, "h2h", OUTCOMES_1X2, method)
    h2h.columns = ["fixture_key", "p_home", "p_draw", "p_away"]
    totals = odds[(odds["market"] != "totals") | (odds["line"] == TOTALS_LINE)]
    totals = _devig_market(totals, "totals", ("over", "under"), method)
    totals = totals.rename(columns={"over": "p_over"})[["fixture_key", "p_over"]]
    ah = _devig_market(odds, "ah", ("home", "away"), method)
    ah = ah.rename(columns={"line": "ah_line", "home": "q_ah_home"})
    out = h2h.merge(totals, on="fixture_key", how="left").merge(
        ah[["fixture_key", "ah_line", "q_ah_home"]], on="fixture_key", how="left"
    )
    out = out.astype({"fixture_key": "int64", **{c: "float64" for c in MARKET_COLUMNS}})
    return out.sort_values("fixture_key", kind="mergesort").reset_index(drop=True)


def _visible_market_lambdas(
    view: AsOfView, fixture_keys: pd.Series | list[int], params: TeamParams
) -> pd.DataFrame:
    """fixture_key, lambda_home, lambda_away for fixtures with visible odds whose solve
    converged."""
    odds = visible_odds(view, fixture_keys)
    if odds.empty:
        return pd.DataFrame(
            {"fixture_key": pd.Series(dtype="int64"), "lambda_home": [], "lambda_away": []}
        )
    solved = market_lambdas(market_probabilities(odds, params.devig), params.rho)
    failed = solved[~solved["success"] & solved["lambda_home"].notna()]
    if len(failed):
        log.warning(
            "team model at %s: market λ did not converge for fixture(s) %s; ratings used",
            view.deadline,
            failed["fixture_key"].tolist(),
        )
    return solved.loc[solved["success"], ["fixture_key", "lambda_home", "lambda_away"]]


# --- ratings ------------------------------------------------------------------------------


def _elo(view: AsOfView) -> pd.Series:
    """team_key -> the Elo of the club's newest visible `team_rating` row (`rating_after`:
    the current rating; equal to `rating_before` on pre-season rows)."""
    ratings = view.table("team_rating", columns=["team_key", "event_time", "rating_after"])
    ratings = ratings.sort_values(["team_key", "event_time"], kind="mergesort")
    newest = ratings.drop_duplicates("team_key", keep="last")
    return newest.set_index("team_key")["rating_after"].astype("float64")


def _current_teams(view: AsOfView) -> list[int]:
    """Clubs of the newest season with a visible schedule (empty without one)."""
    seasons = view.table("gameweek", columns=["season"])["season"]
    if seasons.empty:
        return []
    schedule = view.schedule(int(seasons.max()))
    return sorted(set(schedule["home_team_key"]) | set(schedule["away_team_key"]))


def match_history(view: AsOfView, params: TeamParams | None = None) -> pd.DataFrame:
    """One row per finished match visible in the view (from `team_match`), sorted by
    fixture_key: fixture_key, kickoff_time, available_at, home/away team_key, goals, xG
    (first of XG_SOURCES per side, NaN without) and the market λ of its pre-match odds
    (NaN without odds or a converged solve)."""
    params = TeamParams() if params is None else params
    columns = ["fixture_key", "team_key", "is_home", "goals_for", *XG_SOURCES]
    columns += ["event_time", "available_at"]
    sides = view.table("team_match", columns=columns)
    xg = pd.Series(np.nan, index=sides.index, dtype="float64")
    for source in reversed(XG_SOURCES):
        values = sides[source].astype("Float64").to_numpy(dtype="float64", na_value=np.nan)
        xg = xg.where(np.isnan(values), values)
    sides = sides.assign(xg=xg)
    home = sides[sides["is_home"]].set_index("fixture_key")
    away = sides[~sides["is_home"]].set_index("fixture_key")
    keys = home.index.intersection(away.index).sort_values()
    home, away = home.loc[keys], away.loc[keys]
    matches = pd.DataFrame(
        {
            "fixture_key": keys.astype("int64"),
            "kickoff_time": home["event_time"].to_numpy(),
            "available_at": np.maximum(
                home["available_at"].to_numpy(), away["available_at"].to_numpy()
            ),
            "home_team_key": home["team_key"].to_numpy(dtype="int64"),
            "away_team_key": away["team_key"].to_numpy(dtype="int64"),
            "home_goals": home["goals_for"].to_numpy(dtype="float64"),
            "away_goals": away["goals_for"].to_numpy(dtype="float64"),
            "home_xg": home["xg"].to_numpy(dtype="float64"),
            "away_xg": away["xg"].to_numpy(dtype="float64"),
        }
    )
    window = min(params.half_life_days * math.log2(1.0 / MIN_WEIGHT), MAX_WINDOW_DAYS)
    oldest = view.deadline - pd.Timedelta(days=window)
    matches = matches[matches["kickoff_time"] >= oldest].reset_index(drop=True)
    market = _visible_market_lambdas(view, matches["fixture_key"], params)
    market = market.rename(columns={"lambda_home": "home_market", "lambda_away": "away_market"})
    matches = matches.merge(market, on="fixture_key", how="left", validate="one_to_one")
    return matches.sort_values("fixture_key", kind="mergesort").reset_index(drop=True)


def _targets(matches: pd.DataFrame, market_weight: float) -> tuple[np.ndarray, np.ndarray]:
    """Per match the home and away targets: w · market + (1 − w) · stats (stats alone
    without a market λ)."""
    out = []
    for side in ("home", "away"):
        goals = matches[f"{side}_goals"].to_numpy(dtype="float64")
        xg = matches[f"{side}_xg"].to_numpy(dtype="float64")
        stats = np.where(np.isnan(xg), goals, XG_WEIGHT * xg + (1 - XG_WEIGHT) * goals)
        market = matches[f"{side}_market"].to_numpy(dtype="float64")
        blended = market_weight * market + (1 - market_weight) * stats
        out.append(np.where(np.isnan(market), stats, blended))
    return out[0], out[1]


def fit_ratings(
    matches: pd.DataFrame,
    elo: pd.Series,
    teams: list[int],
    cutoff: pd.Timestamp,
    params: TeamParams | None = None,
) -> TeamFit:
    """The ratings fit on `matches` (`match_history` rows; those available before
    `cutoff` are used) with Elo priors from `elo` (team_key -> rating); `teams` are rated
    even without matches (the season's clubs: their Elo prior). Deterministic: rows sorted
    by fixture_key, fixed start, fixed tolerances."""
    params = TeamParams() if params is None else params
    cutoff = pd.Timestamp(cutoff)
    rows = matches[matches["available_at"] < cutoff]
    rows = rows.sort_values("fixture_key", kind="mergesort").reset_index(drop=True)
    age = (cutoff - pd.to_datetime(rows["kickoff_time"], utc=True)).dt.total_seconds() / 86400
    weight = np.power(0.5, age.to_numpy(dtype="float64") / params.half_life_days)
    keep = weight >= MIN_WEIGHT
    rows, weight = rows[keep].reset_index(drop=True), weight[keep]
    has_market = rows["home_market"].notna().any() if len(rows) else False
    source = "ratings" if has_market else "stats"
    y_home, y_away = _targets(rows, params.market_weight if has_market else 0.0)

    all_teams = sorted(set(teams) | set(rows["home_team_key"]) | set(rows["away_team_key"]))
    n = len(all_teams)
    index = {team: i for i, team in enumerate(all_teams)}
    current = [t for t in teams if t in elo.index] or [t for t in all_teams if t in elo.index]
    center = float(elo.loc[current].mean()) if current else 0.0
    z = np.array([(elo.get(t, center) - center) / ELO_SCALE for t in all_teams])
    z = np.nan_to_num(z, nan=0.0)

    # Observations: home sides then away sides.
    hi = rows["home_team_key"].map(index).to_numpy(dtype="int64")
    ai = rows["away_team_key"].map(index).to_numpy(dtype="int64")
    team = np.concatenate([hi, ai])
    opp = np.concatenate([ai, hi])
    is_home = np.concatenate([np.ones(len(rows)), np.zeros(len(rows))])
    y = np.concatenate([y_home, y_away])
    w = np.concatenate([weight, weight])
    tau = params.prior_strength

    # x = [base, home, attack_slope, defence_slope, u (n), v (n)]
    def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        base, home, a_slope, d_slope = x[:4]
        u, v = x[4 : 4 + n], x[4 + n :]
        attack = a_slope * z + u
        defence = d_slope * z + v
        eta = base + home * is_home + attack[team] - defence[opp]
        mu = np.exp(eta)
        loss = float(np.sum(w * (mu - y * eta)) + 0.5 * tau * (u @ u + v @ v))
        r = w * (mu - y)
        g_attack = np.bincount(team, weights=r, minlength=n)
        g_defence = -np.bincount(opp, weights=r, minlength=n)
        grad = np.concatenate(
            [
                [r.sum(), r @ is_home, g_attack @ z, g_defence @ z],
                g_attack + tau * u,
                g_defence + tau * v,
            ]
        )
        return loss, grad

    x0 = np.zeros(4 + 2 * n)
    x0[0] = math.log(1.35)
    result = minimize(
        objective,
        x0,
        jac=True,
        method="L-BFGS-B",
        options={"gtol": FIT_GTOL, "ftol": 1e-15, "maxiter": FIT_MAXITER, "maxcor": 30},
    )
    if not result.success:
        log.warning("team ratings fit at %s: %s", cutoff, result.message)
    x = result.x
    attack = x[2] * z + x[4 : 4 + n]
    defence = x[3] * z + x[4 + n :]
    return TeamFit(
        cutoff=cutoff,
        params=params,
        source=source,
        base=float(x[0]),
        home=float(x[1]),
        attack_slope=float(x[2]),
        defence_slope=float(x[3]),
        elo_center=center,
        teams=tuple(int(t) for t in all_teams),
        attack=tuple(float(a) for a in attack),
        defence=tuple(float(d) for d in defence),
        n_matches=len(rows),
        n_market=int(rows["home_market"].notna().sum()) if len(rows) else 0,
    )


def fit_team(view: AsOfView, params: TeamParams | None = None) -> TeamFit:
    """The walk-forward team fit at the view's deadline: ratings (with Elo priors) on every
    finished match visible in the view, market-anchored where odds exist."""
    params = TeamParams() if params is None else params
    return fit_ratings(
        match_history(view, params), _elo(view), _current_teams(view), view.deadline, params
    )


def _strengths(fit: TeamFit, team_keys: np.ndarray, elo: pd.Series) -> np.ndarray:
    """[n, 2] (attack, defence) per team key; clubs outside the fit from their Elo."""
    known = dict(zip(fit.teams, zip(fit.attack, fit.defence, strict=True), strict=True))
    out = np.zeros((len(team_keys), 2))
    missing = []
    for i, team in enumerate(team_keys):
        if int(team) in known:
            out[i] = known[int(team)]
            continue
        missing.append(int(team))
        rating = elo.get(int(team), np.nan)
        z = 0.0 if not np.isfinite(rating) else (rating - fit.elo_center) / ELO_SCALE
        out[i] = (fit.attack_slope * z, fit.defence_slope * z)
    if missing:
        log.warning("team model: club(s) %s not in the fit; rated from Elo", sorted(set(missing)))
    return out


def ratings_lambdas(
    fit: TeamFit,
    home_team_keys: np.ndarray | pd.Series,
    away_team_keys: np.ndarray | pd.Series,
    elo: pd.Series | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(λ_home, λ_away) from the fitted ratings for the given pairings."""
    elo = pd.Series(dtype="float64") if elo is None else elo
    home = _strengths(fit, np.asarray(home_team_keys), elo)
    away = _strengths(fit, np.asarray(away_team_keys), elo)
    lambda_home = np.exp(fit.base + fit.home + home[:, 0] - away[:, 1])
    lambda_away = np.exp(fit.base + away[:, 0] - home[:, 1])
    return lambda_home, lambda_away


def team_lambdas(view: AsOfView, fit: TeamFit) -> pd.DataFrame:
    """One row per fixture and side of `upcoming_fixtures(view)` (the target GW and the
    next 5): λ for / against, P(CS) under Dixon-Coles, and the source: 'market' where
    pre-match odds are visible at the view's deadline, else 'ratings' ('stats' when the
    fit had no odds). Columns `TEAM_DTYPES`, sorted by (team_key, gw_index, fixture_key)."""
    params = fit.params
    sides = upcoming_fixtures(view)
    sides = sides[sides["fixture_key"].notna()].astype(
        {"fixture_key": "int64", "opponent_team_key": "int64", "is_home": "bool"}
    )
    home = sides[sides["is_home"]]
    fixtures = pd.DataFrame(
        {
            "fixture_key": home["fixture_key"].to_numpy(dtype="int64"),
            "home_team_key": home["team_key"].to_numpy(dtype="int64"),
            "away_team_key": home["opponent_team_key"].to_numpy(dtype="int64"),
        }
    )
    fixtures = fixtures.sort_values("fixture_key", kind="mergesort").reset_index(drop=True)
    lambda_home, lambda_away = ratings_lambdas(
        fit, fixtures["home_team_key"], fixtures["away_team_key"], _elo(view)
    )
    market = _visible_market_lambdas(view, fixtures["fixture_key"], params)
    market = fixtures[["fixture_key"]].merge(market, on="fixture_key", how="left")
    in_market = market["lambda_home"].notna().to_numpy()
    fixtures = fixtures.assign(
        lambda_home=np.where(in_market, market["lambda_home"], lambda_home),
        lambda_away=np.where(in_market, market["lambda_away"], lambda_away),
        source=np.where(in_market, "market", "stats" if fit.source == "stats" else "ratings"),
    )
    matrix = dc_matrix(
        fixtures["lambda_home"].to_numpy(), fixtures["lambda_away"].to_numpy(), params.rho
    )
    fixtures["cs_home"] = matrix[:, :, 0].sum(axis=1)  # away scores 0
    fixtures["cs_away"] = matrix[:, 0, :].sum(axis=1)  # home scores 0
    out = sides.merge(fixtures, on="fixture_key", how="left", validate="many_to_one")
    is_home = out["is_home"].to_numpy(dtype=bool)
    out = out.assign(
        lambda_for=np.where(is_home, out["lambda_home"], out["lambda_away"]),
        lambda_against=np.where(is_home, out["lambda_away"], out["lambda_home"]),
        p_cs=np.where(is_home, out["cs_home"], out["cs_away"]),
    )
    out = out[[name for name, _ in TEAM_DTYPES]].astype(dict(TEAM_DTYPES))
    order = ["team_key", "gw_index", "fixture_key"]
    return out.sort_values(order, kind="mergesort").reset_index(drop=True)
