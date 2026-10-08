"""Team model (fplopt.models.team): de-vig, Dixon-Coles market λ, ratings, team_lambdas."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq
from scipy.stats import poisson

from fplopt.features.baseline import upcoming_fixtures
from fplopt.features.store import DataStore
from fplopt.models import team
from fplopt.models.team import (
    TEAM_DTYPES,
    TeamParams,
    ah_home_probability,
    ah_stakes,
    dc_matrix,
    devig_power,
    devig_shin,
    fit_ratings,
    fit_team,
    market_lambdas,
    market_probabilities,
    match_history,
    team_lambdas,
)

UTC_US = pd.DatetimeTZDtype("us", "UTC")


# --- de-vig -------------------------------------------------------------------------------


def test_power_devig_of_a_symmetric_two_way_market_is_one_half():
    out = devig_power(np.array([[1.9, 1.9]]))
    np.testing.assert_allclose(out, [[0.5, 0.5]], atol=1e-12)


def test_power_devig_matches_a_hand_solved_exponent():
    prices = np.array([[1.5, 2.6]])
    implied = 1 / prices[0]
    k = brentq(lambda k: (implied**k).sum() - 1, 0.5, 5, xtol=1e-14)
    np.testing.assert_allclose(devig_power(prices), [implied**k], atol=1e-10)
    assert k > 1  # a margin: the exponent shrinks the implied probabilities


def test_power_devig_inverts_a_power_margin():
    p = np.array([0.5, 0.3, 0.2])
    prices = 1 / p ** (1 / 1.1)  # overround from a power margin with k = 1.1
    np.testing.assert_allclose(devig_power(prices[None, :]), [p], atol=1e-10)


def shin_prices(p: np.ndarray, z: float) -> np.ndarray:
    """Shin (1993): pi_i = sqrt(z p_i + (1 - z) p_i^2) * sum_j sqrt(z p_j + (1 - z) p_j^2)."""
    s = np.sqrt(z * p + (1 - z) * p**2)
    return 1 / (s * s.sum())


@pytest.mark.parametrize("p", [[0.5, 0.3, 0.2], [0.72, 0.28], [0.1, 0.2, 0.7]])
def test_shin_devig_inverts_shins_model(p):
    p = np.array(p)
    prices = shin_prices(p, 0.03)
    assert (1 / prices).sum() > 1
    np.testing.assert_allclose(devig_shin(prices[None, :]), [p], atol=1e-9)


def test_shin_devig_without_a_margin_normalizes_proportionally():
    prices = np.array([[2.05, 2.05]])  # an exchange: implied sum < 1
    np.testing.assert_allclose(devig_shin(prices), [[0.5, 0.5]], atol=1e-12)


def test_devig_rows_with_a_missing_or_bad_price_are_nan():
    prices = np.array([[2.0, np.nan, 3.0], [2.0, 3.5, 4.0], [0.0, 3.0, 3.0]])
    for method in ("power", "shin"):
        out = team.devig(prices, method)
        assert np.isnan(out[0]).all() and np.isnan(out[2]).all()
        assert out[1].sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="unknown de-vig"):
        team.devig(prices, "proportional")


# --- Dixon-Coles --------------------------------------------------------------------------


def test_dc_matrix_sums_to_one():
    m = dc_matrix(np.array([0.4, 1.5, 3.2]), np.array([2.1, 1.0, 0.3]), -0.1)
    np.testing.assert_allclose(m.sum(axis=(1, 2)), 1.0, atol=1e-12)
    assert (m >= 0).all()


def test_dc_matrix_with_rho_zero_is_independent_poisson():
    m = dc_matrix(1.7, 0.9, 0.0, max_goals=30)[0]
    goals = np.arange(31)
    expected = np.outer(poisson.pmf(goals, 1.7), poisson.pmf(goals, 0.9))
    np.testing.assert_allclose(m, expected, atol=1e-14)


def test_dc_correction_only_touches_low_scores():
    lh, la, rho = 1.3, 1.1, -0.1
    m = dc_matrix(lh, la, rho, max_goals=30)[0]
    goals = np.arange(31)
    base = np.outer(poisson.pmf(goals, lh), poisson.pmf(goals, la))
    ratio = m / base
    tau = {(0, 0): 1 - lh * la * rho, (0, 1): 1 + lh * rho, (1, 0): 1 + la * rho, (1, 1): 1 - rho}
    for (i, j), t in tau.items():
        assert ratio[i, j] == pytest.approx(t, rel=1e-9)
    assert ratio[2, 3] == pytest.approx(1.0, rel=1e-9)  # τ preserves the mass: no rescaling


# --- Asian handicap -----------------------------------------------------------------------


def stake(line, home, away):
    won, lost = ah_stakes(line, max_goals=5)
    return won[home, away], lost[home, away]


@pytest.mark.parametrize(
    ("line", "score", "expected"),
    [
        (-0.5, (1, 0), (1, 0)),  # half line: no push
        (-0.5, (1, 1), (0, 1)),
        (0.0, (1, 1), (0, 0)),  # whole line: push
        (-1.0, (2, 1), (0, 0)),
        (-1.0, (3, 1), (1, 0)),
        (-0.25, (1, 1), (0, 0.5)),  # quarter: half on 0 (push), half on -0.5 (lost)
        (-0.25, (2, 1), (1, 0)),
        (-0.75, (2, 1), (0.5, 0)),  # half on -0.5 (won), half on -1 (push)
        (-0.75, (3, 1), (1, 0)),
        (0.25, (1, 1), (0.5, 0)),  # half on 0 (push), half on +0.5 (won)
        (0.25, (0, 1), (0, 1)),
        (1.75, (0, 2), (0, 0.5)),  # half on +1.5 (lost), half on +2 (push)
    ],
)
def test_ah_stakes_split_quarter_lines_and_push_whole_lines(line, score, expected):
    assert stake(line, *score) == pytest.approx(expected)


def test_ah_home_probability_is_won_over_won_plus_lost():
    m = dc_matrix(1.6, 1.0, 0.0)
    diff = np.subtract.outer(np.arange(11), np.arange(11))
    p = {d: m[0][diff == d].sum() for d in range(-10, 11)}
    win = sum(v for d, v in p.items() if d > 0)
    loss = sum(v for d, v in p.items() if d < 0)
    assert ah_home_probability(m, -0.5)[0] == pytest.approx(win)
    assert ah_home_probability(m, 0.0)[0] == pytest.approx(win / (win + loss))
    quarter = win / (win + loss + 0.5 * p[0])  # -0.25: a draw loses half the stake
    assert ah_home_probability(m, -0.25)[0] == pytest.approx(quarter)
    big = sum(v for d, v in p.items() if d >= 2)
    w = 0.5 * p[1] + big  # -0.75: a one-goal win wins half, the other half is pushed
    lo = sum(v for d, v in p.items() if d <= 0)
    assert ah_home_probability(m, -0.75)[0] == pytest.approx(w / (w + lo))
    with pytest.raises(ValueError, match="multiples of 0.25"):
        ah_stakes(-0.3)


# --- market λ ------------------------------------------------------------------------------


def fair_probabilities(lh: float, la: float, line: float, rho: float = team.RHO) -> dict:
    m = dc_matrix(lh, la, rho)
    diff = np.subtract.outer(np.arange(11), np.arange(11))
    total = np.add.outer(np.arange(11), np.arange(11))
    q = ah_home_probability(m, line)[0]
    return {
        "h2h": {
            "home": m[0][diff > 0].sum(),
            "draw": m[0][diff == 0].sum(),
            "away": m[0][diff < 0].sum(),
        },
        "totals": {"over": m[0][total > 2.5].sum(), "under": m[0][total < 2.5].sum()},
        "ah": {"home": q, "away": 1 - q},
    }


def priced(
    fixture_key: int,
    lh: float,
    la: float,
    line: float = -0.25,
    bookmaker: str = "avg",
    source: str = "football-data",
    k: float = 1.06,
    markets: tuple[str, ...] = ("h2h", "totals", "ah"),
) -> list[dict]:
    """Odds rows whose power de-vig gives the Dixon-Coles probabilities of (lh, la)."""
    rows = []
    for market, outcomes in fair_probabilities(lh, la, line).items():
        if market not in markets:
            continue
        for outcome, p in outcomes.items():
            rows.append(
                {
                    "fixture_key": fixture_key,
                    "source": source,
                    "bookmaker": bookmaker,
                    "market": market,
                    "outcome": outcome,
                    "line": {"h2h": np.nan, "totals": 2.5, "ah": line}[market],
                    "price": 1 / p ** (1 / k),
                }
            )
    return rows


@pytest.mark.parametrize("markets", [("h2h", "totals", "ah"), ("h2h", "totals"), ("h2h",)])
@pytest.mark.parametrize(
    ("lh", "la", "line"), [(1.8, 0.9, -0.75), (0.7, 2.6, 1.25), (1.3, 1.25, 0.0)]
)
def test_market_lambdas_recover_the_lambdas_behind_the_odds(markets, lh, la, line):
    odds = pd.DataFrame(priced(7, lh, la, line, markets=markets))
    probabilities = market_probabilities(odds)
    assert probabilities["ah_line"].notna().item() == ("ah" in markets)
    solved = market_lambdas(probabilities)
    assert solved["success"].item()
    assert solved["lambda_home"].item() == pytest.approx(lh, abs=1e-6)
    assert solved["lambda_away"].item() == pytest.approx(la, abs=1e-6)


def test_bookmakers_are_devigged_separately_and_their_median_taken():
    rows = []
    for i, (bookmaker, k) in enumerate((("a", 1.02), ("b", 1.08), ("c", 1.15))):
        lh = 1.5 + 0.1 * i
        rows += priced(3, lh, 1.0, bookmaker=bookmaker, source="odds-api", k=k, markets=("h2h",))
    probabilities = market_probabilities(pd.DataFrame(rows))
    middle = fair_probabilities(1.6, 1.0, 0.0)["h2h"]  # bookmaker b
    raw = [fair_probabilities(1.5 + 0.1 * i, 1.0, 0.0)["h2h"] for i in range(3)]
    expected = {o: np.median([r[o] for r in raw]) for o in ("home", "draw", "away")}
    total = sum(expected.values())
    assert probabilities["p_home"].item() == pytest.approx(expected["home"] / total)
    assert probabilities["p_draw"].item() == pytest.approx(middle["draw"] / total, rel=1e-9)
    assert np.isnan(probabilities["p_over"].item())


# --- a synthetic league ----------------------------------------------------------------------


def round_robin(teams: list[int]) -> list[list[tuple[int, int]]]:
    """Double round robin by the circle method: rounds of (home, away)."""
    n = len(teams)
    order = list(teams)
    rounds = []
    for r in range(n - 1):
        pairs = [(order[i], order[n - 1 - i]) for i in range(n // 2)]
        rounds.append([(a, b) if r % 2 else (b, a) for a, b in pairs])
        order = [order[0], order[-1], *order[1:-1]]
    return rounds + [[(b, a) for a, b in rnd] for rnd in rounds]


def league(
    seasons=(2021, 2022, 2023),
    n_clubs: int = 10,
    strength=None,
    elo=None,
    odds: str = "none",  # 'none', 'visible' (pre-deadline) or 'late' (after the deadline)
    seed: int = 0,
    base: float = 0.25,
    home_adv: float = 0.2,
) -> dict[str, pd.DataFrame]:
    """In-memory tables (gameweek, schedule, fixture_snapshot, team_match, team_rating,
    odds_snapshot) of a league whose λ follow `strength(season, team) -> (attack, defence)`:
    log λ_home = base + home_adv + attack[home] − defence[away]. `us_xg` = the true λ,
    goals ~ Poisson(λ)."""
    rng = np.random.default_rng(seed)
    teams = list(range(1, n_clubs + 1))
    if strength is None:
        planted = {t: (0.3 * math.sin(t), 0.25 * math.cos(2 * t)) for t in teams}

        def strength(season, t):
            return planted[t]

    elo = elo or {t: 1500.0 for t in teams}
    gameweeks, schedule, sides, ratings, odds_rows = [], [], [], [], []
    for season in seasons:
        published = pd.Timestamp(f"{season}-06-01", tz="UTC")
        first = pd.Timestamp(f"{season}-08-06 17:30", tz="UTC")
        for t in teams:
            ratings.append(
                {
                    "team_key": t,
                    "season": season,
                    "event_time": published,
                    "available_at": published,
                    "rating_before": elo[t],
                    "rating_after": elo[t],
                }
            )
        for g, rnd in enumerate(round_robin(teams), start=1):
            deadline = first + pd.Timedelta(days=7 * (g - 1))
            kickoff = deadline + pd.Timedelta(days=1)
            lockdown = kickoff + pd.Timedelta(days=1)
            gameweeks.append(
                {
                    "season": season,
                    "gw": g,
                    "gw_index": g,
                    "deadline_time": deadline,
                    "lockdown_time": lockdown,
                    "event_time": deadline,
                    "available_at": published,
                }
            )
            for i, (h, a) in enumerate(rnd):
                key = season * 1000 + g * 10 + i
                schedule.append(
                    {
                        "fixture_key": key,
                        "season": season,
                        "gw": g,
                        "gw_index": g,
                        "kickoff_time": kickoff,
                        "home_team_key": h,
                        "away_team_key": a,
                        "schedule_source": "final",
                        "event_time": kickoff,
                        "available_at": published,
                    }
                )
                (ah_, dh), (aa, da) = strength(season, h), strength(season, a)
                lh = math.exp(base + home_adv + ah_ - da)
                la = math.exp(base + aa - dh)
                gh, ga = rng.poisson(lh), rng.poisson(la)
                for t, home, gf, ga_, xg in (
                    (h, True, gh, ga, lh),
                    (a, False, ga, gh, la),
                ):
                    sides.append(
                        {
                            "fixture_key": key,
                            "team_key": t,
                            "season": season,
                            "is_home": home,
                            "goals_for": gf,
                            "goals_against": ga_,
                            "us_xg": xg,
                            "fpl_xg": None,
                            "fd_xg": None,
                            "event_time": kickoff,
                            "available_at": lockdown,
                        }
                    )
                if odds != "none":
                    at = deadline + pd.Timedelta(hours=-1 if odds == "visible" else 1)
                    for row in priced(key, lh, la, line=-0.25):
                        odds_rows.append(
                            {
                                **row,
                                "season": season,
                                "is_closing": False,
                                "snapshot_at": at,
                                "event_time": kickoff,
                                "available_at": at,
                            }
                        )
                    # A closing price (visible from kickoff) that must never be used.
                    for row in priced(key, 3.0, 0.3, line=-0.25):
                        odds_rows.append(
                            {
                                **row,
                                "season": season,
                                "is_closing": True,
                                "snapshot_at": kickoff,
                                "event_time": kickoff,
                                "available_at": kickoff,
                            }
                        )
    team_match = pd.DataFrame(sides).astype(
        {"us_xg": "Float64", "fpl_xg": "Float64", "fd_xg": "Float64"}
    )
    odds_columns = {
        "fixture_key": "int64",
        "season": "int64",
        "source": "str",
        "bookmaker": "str",
        "market": "str",
        "outcome": "str",
        "line": "Float64",
        "price": "float64",
        "is_closing": "bool",
        "snapshot_at": UTC_US,
        "event_time": UTC_US,
        "available_at": UTC_US,
    }
    odds_frame = pd.DataFrame(odds_rows, columns=list(odds_columns)).astype(odds_columns)
    snapshot_columns = {
        "snapshot_at": UTC_US,
        "season": "int64",
        "fixture_key": "int64",
        "gw": "Int64",
        "kickoff_time": UTC_US,
        "home_team_key": "int64",
        "away_team_key": "int64",
        "event_time": UTC_US,
        "available_at": UTC_US,
    }
    return {
        "gameweek": as_us(pd.DataFrame(gameweeks)),
        "schedule": as_us(pd.DataFrame(schedule)).astype({"gw": "Int64", "gw_index": "Int64"}),
        "fixture_snapshot": pd.DataFrame(columns=list(snapshot_columns)).astype(snapshot_columns),
        "team_match": as_us(team_match),
        "team_rating": as_us(pd.DataFrame(ratings)),
        "odds_snapshot": odds_frame,
    }


def as_us(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype({c: UTC_US for c in df.columns if isinstance(df[c].dtype, pd.DatetimeTZDtype)})


def deadline(tables, season: int, gw: int) -> pd.Timestamp:
    gws = tables["gameweek"]
    return gws.loc[(gws["season"] == season) & (gws["gw"] == gw), "deadline_time"].item()


def rated(fit) -> pd.DataFrame:
    return pd.DataFrame({"attack": fit.attack, "defence": fit.defence}, index=list(fit.teams))


def centered(values: pd.Series) -> pd.Series:
    return values - values.mean()


LOOSE = TeamParams(prior_strength=1e-4, half_life_days=1e5)


def test_ratings_recover_planted_strengths_from_exact_market_lambdas():
    tables = league(odds="visible")
    view = DataStore(tables=tables).as_of(deadline(tables, 2023, 10))
    params = TeamParams(prior_strength=1e-4, half_life_days=1e5, market_weight=1.0)
    fit = fit_team(view, params)
    assert fit.source == "ratings" and fit.n_market == fit.n_matches > 0
    truth = pd.DataFrame(
        {t: (0.3 * math.sin(t), 0.25 * math.cos(2 * t)) for t in range(1, 11)},
        index=["attack", "defence"],
    ).T
    got = rated(fit)
    np.testing.assert_allclose(centered(got["attack"]), centered(truth["attack"]), atol=2e-3)
    np.testing.assert_allclose(centered(got["defence"]), centered(truth["defence"]), atol=2e-3)
    assert fit.home == pytest.approx(0.2, abs=2e-3)


def test_ratings_recover_planted_strengths_from_stats_without_odds():
    tables = league(seasons=(2019, 2020, 2021, 2022, 2023))
    view = DataStore(tables=tables).as_of(deadline(tables, 2023, 10))
    fit = fit_team(view, LOOSE)
    assert fit.source == "stats" and fit.n_market == 0
    truth = {t: (0.3 * math.sin(t), 0.25 * math.cos(2 * t)) for t in range(1, 11)}
    got = rated(fit)
    attack = centered(pd.Series({t: a for t, (a, _) in truth.items()}))
    defence = centered(pd.Series({t: d for t, (_, d) in truth.items()}))
    assert np.corrcoef(centered(got["attack"]), attack)[0, 1] > 0.95
    assert np.corrcoef(centered(got["defence"]), defence)[0, 1] > 0.95
    assert np.abs(centered(got["attack"]) - attack).max() < 0.08


def changing(season, t):
    """Club 1 is strong until 2022, then weak; the rest are equal."""
    if t == 1:
        return (0.5, 0.0) if season < 2023 else (-0.5, 0.0)
    return (0.0, 0.0)


def test_a_short_half_life_follows_recent_form_and_a_long_one_averages():
    tables = league(strength=changing, odds="visible")
    view = DataStore(tables=tables).as_of(deadline(tables, 2023, 18))
    exact = {"prior_strength": 1e-4, "market_weight": 1.0}
    short = rated(fit_team(view, TeamParams(half_life_days=20, **exact)))
    long = rated(fit_team(view, TeamParams(half_life_days=1e5, **exact)))
    gap_short = short.loc[1, "attack"] - short.drop(1)["attack"].mean()
    gap_long = long.loc[1, "attack"] - long.drop(1)["attack"].mean()
    assert gap_short == pytest.approx(-0.5, abs=0.05)
    assert 0.0 < gap_long < 0.4  # two strong seasons, one weak: in between


def promoted_league() -> tuple[dict[str, pd.DataFrame], dict[int, float]]:
    """`league` 2021-2023 with strength linear in Elo, except club 1 (+0.4 attack above its
    Elo line); in 2024 club 11 (Elo 1450, no match yet) replaces club 10."""
    elo = {t: 1500.0 + 40 * (t - 5) for t in range(1, 11)}

    def strength(season, t):
        return (0.004 * (elo[t] - 1500) + (0.4 if t == 1 else 0.0), 0.003 * (elo[t] - 1500))

    tables = league(strength=strength, elo=elo, odds="visible")
    year = pd.Timedelta(days=364)
    published = pd.Timestamp("2024-06-01", tz="UTC")
    rows = tables["schedule"]
    s2024 = rows[rows["season"] == 2023].assign(
        season=2024,
        fixture_key=lambda d: d["fixture_key"] + 1000,
        kickoff_time=lambda d: d["kickoff_time"] + year,
        event_time=lambda d: d["event_time"] + year,
        available_at=published,
    )
    s2024 = s2024.replace({"home_team_key": {10: 11}, "away_team_key": {10: 11}})
    tables["schedule"] = as_us(pd.concat([rows, s2024], ignore_index=True)).astype(
        {"gw": "Int64", "gw_index": "Int64"}
    )
    gws = tables["gameweek"]
    g2024 = gws[gws["season"] == 2023].assign(
        season=2024,
        deadline_time=lambda d: d["deadline_time"] + year,
        event_time=lambda d: d["event_time"] + year,
        lockdown_time=lambda d: d["lockdown_time"] + year,
        available_at=published,
    )
    tables["gameweek"] = as_us(pd.concat([gws, g2024], ignore_index=True))
    ratings = tables["team_rating"]
    new = ratings[ratings["season"] == 2023].assign(
        season=2024, event_time=published, available_at=published
    )
    new.loc[new["team_key"] == 10, ["team_key", "rating_before", "rating_after"]] = [11, 1450, 1450]
    tables["team_rating"] = as_us(pd.concat([ratings, new], ignore_index=True))
    return tables, {**elo, 11: 1450.0}


def deviation(fit, elo: dict[int, float]) -> pd.Series:
    """attack minus the fit's Elo line, per club."""
    got = rated(fit)["attack"]
    line = pd.Series({t: (elo[t] - fit.elo_center) / team.ELO_SCALE for t in got.index})
    return got - fit.attack_slope * line


def test_promoted_clubs_start_at_their_elo_prior():
    tables, elo = promoted_league()
    view = DataStore(tables=tables).as_of(deadline(tables, 2024, 1))
    fit = fit_team(view)
    got = rated(fit)
    assert {10, 11} <= set(got.index)
    z = (1450.0 - fit.elo_center) / team.ELO_SCALE
    assert fit.elo_center == pytest.approx(np.mean([elo[t] for t in (*range(1, 10), 11)]))
    assert got.loc[11, "attack"] == pytest.approx(fit.attack_slope * z, abs=1e-9)
    assert got.loc[11, "defence"] == pytest.approx(fit.defence_slope * z, abs=1e-9)
    assert fit.attack_slope > 0 and fit.defence_slope > 0  # strength follows Elo here
    out = team_lambdas(view, fit)
    assert 11 in set(out["team_key"]) and 10 not in set(out["team_key"])


def test_a_stronger_prior_pulls_clubs_toward_their_elo_line():
    tables, elo = promoted_league()
    view = DataStore(tables=tables).as_of(deadline(tables, 2024, 1))
    weak = deviation(fit_team(view, TeamParams(prior_strength=1e-3)), elo)
    strong = deviation(fit_team(view, TeamParams(prior_strength=1e5)), elo)
    # Nearly free deviations: club 1's +0.4 shows (less what the fitted line absorbs).
    assert weak.loc[1] - weak.drop([1, 11]).mean() > 0.25
    assert weak.loc[11] == pytest.approx(0.0, abs=1e-9)  # no match: on the line
    assert strong.abs().max() < 0.01


def test_rows_available_after_the_cutoff_are_not_used():
    tables = league(odds="visible")
    cut = deadline(tables, 2023, 6)
    view = DataStore(tables=tables).as_of(cut)
    # A later view's history (every match: its look-back window covers the cutoff's).
    later = DataStore(tables=tables).as_of(deadline(tables, 2023, 12))
    matches = match_history(later, TeamParams(half_life_days=1e4))
    assert (matches["available_at"] >= cut).any()
    elo = pd.Series(1500.0, index=range(1, 11))
    direct = fit_ratings(matches, elo, list(range(1, 11)), cut)
    assert direct == fit_team(view)


# --- team_lambdas ----------------------------------------------------------------------------


def corrupt_after(tables, at: pd.Timestamp, seed: int = 1) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    out = {name: df.copy() for name, df in tables.items()}
    tm = out["team_match"]
    late = tm["available_at"] >= at
    tm.loc[late, "goals_for"] = rng.integers(0, 9, int(late.sum()))
    tm.loc[late, "us_xg"] = rng.uniform(0, 5, int(late.sum()))
    odds = out["odds_snapshot"]
    late = odds["available_at"] >= at
    odds.loc[late, "price"] = rng.uniform(1.01, 30, int(late.sum()))
    return out


def test_team_lambdas_schema_and_determinism():
    tables = league(odds="visible")
    view = DataStore(tables=tables).as_of(deadline(tables, 2023, 6))
    fit = fit_team(view.earlier(deadline(tables, 2023, 5)))
    first, second = team_lambdas(view, fit), team_lambdas(view, fit)
    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == [name for name, _ in TEAM_DTYPES]
    assert {c: str(t) for c, t in first.dtypes.items()} == {c: str(t) for c, t in TEAM_DTYPES}
    assert isinstance(first.index, pd.RangeIndex)
    assert first.equals(
        first.sort_values(["team_key", "gw_index", "fixture_key"]).reset_index(drop=True)
    )
    upcoming = upcoming_fixtures(view).dropna(subset=["fixture_key"])
    assert len(first) == len(upcoming)
    assert sorted(first["horizon"].unique()) == list(range(6))
    assert set(first["source"]) <= set(team.SOURCES)
    # Both sides of a fixture agree.
    pair = first.merge(first, on="fixture_key", suffixes=("", "_o"))
    pair = pair[pair["team_key"] != pair["team_key_o"]]
    np.testing.assert_allclose(pair["lambda_for"], pair["lambda_against_o"])
    # Odds are visible for the target GW only: market there, ratings later.
    assert set(first.loc[first["horizon"] == 0, "source"]) == {"market"}
    assert set(first.loc[first["horizon"] > 0, "source"]) == {"ratings"}
    # P(CS) is P(opponent scores 0) under Dixon-Coles.
    home = first[first["is_home"]].iloc[0]
    m = dc_matrix(home["lambda_for"], home["lambda_against"])[0]
    assert home["p_cs"] == pytest.approx(m[:, 0].sum())
    away = first[~first["is_home"]].iloc[0]
    m = dc_matrix(away["lambda_against"], away["lambda_for"])[0]
    assert away["p_cs"] == pytest.approx(m[0, :].sum())


def test_market_rows_reproduce_the_odds_lambdas():
    tables = league(odds="visible")
    view = DataStore(tables=tables).as_of(deadline(tables, 2023, 6))
    out = team_lambdas(view, fit_team(view.earlier(deadline(tables, 2023, 5))))
    target = out[(out["horizon"] == 0) & out["is_home"]]
    tm = tables["team_match"]
    truth = tm[tm["is_home"]].set_index("fixture_key")["us_xg"].astype("float64")
    np.testing.assert_allclose(target["lambda_for"], target["fixture_key"].map(truth), atol=1e-6)


def test_only_odds_visible_at_the_deadline_are_used():
    tables = league(odds="late")  # every price lands 1 h after its GW's deadline
    at = deadline(tables, 2023, 6)
    view = DataStore(tables=tables).as_of(at)
    fit = fit_team(view.earlier(deadline(tables, 2023, 5)))
    assert fit.source == "ratings"  # past matches' odds are visible by now
    out = team_lambdas(view, fit)
    assert set(out["source"]) == {"ratings"}
    corrupted = DataStore(tables=corrupt_after(tables, at)).as_of(at)
    again = team_lambdas(corrupted, fit_team(corrupted.earlier(deadline(tables, 2023, 5))))
    pd.testing.assert_frame_equal(out, again)


def test_corrupting_the_future_leaves_market_predictions_unchanged():
    tables = league(odds="visible")
    at = deadline(tables, 2023, 6)
    out = team_lambdas(
        DataStore(tables=tables).as_of(at), fit_team(DataStore(tables=tables).as_of(at))
    )
    corrupted = DataStore(tables=corrupt_after(tables, at)).as_of(at)
    pd.testing.assert_frame_equal(out, team_lambdas(corrupted, fit_team(corrupted)))


def test_without_any_odds_the_stats_fallback_is_used():
    tables = league()
    view = DataStore(tables=tables).as_of(deadline(tables, 2023, 6))
    out = team_lambdas(view, fit_team(view))
    assert set(out["source"]) == {"stats"}
    assert (out["lambda_for"] > 0).all() and out["p_cs"].between(0, 1).all()


def test_team_params_are_validated():
    with pytest.raises(ValueError, match="devig"):
        TeamParams(devig="multiplicative")
    with pytest.raises(ValueError, match="market_weight"):
        TeamParams(market_weight=1.5)
    with pytest.raises(ValueError, match="half_life_days"):
        TeamParams(half_life_days=0)
