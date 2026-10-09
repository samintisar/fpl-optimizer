"""Bonus, saves, cards/own goals and their per-fixture rates (`fplopt.models.components`) on
hand-made rows and the synthetic league."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import nbinom
from synthetic_season import synthetic_tables

from fplopt.features.store import DataStore
from fplopt.models.components import (
    BONUS_FEATURES,
    COMPONENT_COLUMNS,
    DEFCON_PRIOR_MEAN,
    MIN_BONUS_ROWS,
    ComponentsParams,
    bonus_features,
    defcon_counts,
    defcon_rates,
    fit_bonus,
    fit_components,
    fit_rates,
    fit_saves,
    negbin_tail,
    predict_components,
    training_rows,
)
from fplopt.models.gbm import GbmParams
from fplopt.models.minutes import MinutesParams, fit_minutes, predict_minutes
from fplopt.models.shares import fit_shares
from fplopt.models.team import fit_team, team_lambdas

SMALL = GbmParams(num_boost_round=20, min_data_in_leaf=20)
MINUTES = MinutesParams(start=SMALL, sixty=SMALL, sub=SMALL)


def rows(n: int, **columns) -> pd.DataFrame:
    base = {
        "player_key": np.arange(n) % 50,
        "season": 2023,
        "fixture_key": np.arange(n),
        "minutes": 90.0,
        "goals_scored": 0.0,
        "assists": 0.0,
        "clean_sheets": 0.0,
        "saves": 0.0,
        "bonus": 0.0,
        "yellow_cards": 0.0,
        "red_cards": 0.0,
        "own_goals": 0.0,
        "penalties_saved": 0.0,
        "element_type": 3,
        "weight": 1.0,
    }
    base.update(columns)
    return pd.DataFrame(base)


# --- bonus ----------------------------------------------------------------------------------


def test_bonus_features_are_indicators_of_the_events():
    frame = rows(
        3,
        minutes=[30.0, 90.0, 0.0],
        goals_scored=[2.0, 0.0, 0.0],
        assists=[0.0, 1.0, 0.0],
        clean_sheets=[0.0, 1.0, 0.0],
        saves=[7.0, 7.0, 0.0],
        element_type=[3, 1, 2],
    )
    x = bonus_features(frame)
    assert x.shape == (3, len(BONUS_FEATURES))
    names = list(BONUS_FEATURES)
    assert x[0, names.index("goal2")] == 1 and x[0, names.index("goal3")] == 0
    assert x[0, names.index("sixty")] == 0 and x[1, names.index("sixty")] == 1
    assert x[0, names.index("saves3")] == 0  # outfield saves don't count
    assert x[1, names.index("saves3")] == 2  # 7 // 3
    assert x[2].sum() == 0  # no minutes: every feature 0


def test_bonus_regression_recovers_planted_coefficients():
    rng = np.random.default_rng(0)
    n = 4000
    goals = rng.poisson(0.3, n).astype("float64")
    assists = rng.poisson(0.2, n).astype("float64")
    cs = (rng.random(n) < 0.3).astype("float64")
    minutes = np.where(rng.random(n) < 0.8, 90.0, 30.0)
    frame = rows(n, goals_scored=goals, assists=assists, clean_sheets=cs, minutes=minutes)
    planted = np.array([0.05, 0.1, 1.2, 0.8, 0.3, 0.6, 0.4, 0.2, 0.0])
    frame["bonus"] = bonus_features(frame) @ planted
    fit = fit_bonus(frame)
    coefficients = dict(fit.coefficients)
    np.testing.assert_allclose(coefficients[3][:8], planted[:8], atol=2e-3)
    assert coefficients[1] == (0.0,) * len(BONUS_FEATURES)  # too few goalkeeper rows


def test_bonus_weights_favour_recent_rows():
    n = 2 * MIN_BONUS_ROWS
    goals = np.tile([0.0, 1.0], n // 2)
    old = rows(n, goals_scored=goals, bonus=goals * 3.0, weight=0.01, season=2020)
    new = rows(n, goals_scored=goals, bonus=goals * 1.0, weight=1.0)
    fit = fit_bonus(pd.concat([old, new], ignore_index=True))
    goal1 = dict(fit.coefficients)[3][BONUS_FEATURES.index("goal1")]
    assert goal1 == pytest.approx(1.0 + 2.0 * 0.01 / 1.01, abs=0.01)


# --- saves ----------------------------------------------------------------------------------


def keeper_rows(n: int, a: float, b: float, seed: int = 0, keepers: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lam = rng.uniform(0.6, 2.5, n)
    minutes = np.where(rng.random(n) < 0.9, 90.0, 45.0)
    saves = rng.poisson(minutes / 90 * np.exp(a + b * np.log(lam))).astype("float64")
    return rows(
        n,
        player_key=np.arange(n) % keepers,
        element_type=1,
        minutes=minutes,
        saves=saves,
        lambda_against=lam,
        penalties_saved=(rng.random(n) < 0.03).astype("float64"),
    )


def test_saves_curve_is_recovered_and_keepers_are_shrunk():
    frame = keeper_rows(6000, a=0.9, b=0.6)
    fit = fit_saves(frame, keeper_prior=30.0)
    assert fit.a == pytest.approx(0.9, abs=0.05)
    assert fit.b == pytest.approx(0.6, abs=0.05)
    effects = np.array([v for _, v in fit.keepers])
    assert len(effects) == 20 and np.all(np.abs(effects - 1) < 0.15)  # no real keeper effect
    assert fit.pen_save_rate == pytest.approx(0.03, abs=0.01)
    # A keeper who saves twice as much is pulled up, but shrunk toward 1.
    better = frame.copy()
    better.loc[better["player_key"] == 0, "saves"] *= 2
    effect = dict(fit_saves(better, keeper_prior=30.0).keepers)[0]
    assert 1.5 < effect < 2.0
    looser = dict(fit_saves(better, keeper_prior=3000.0).keepers)[0]
    assert 1.0 < looser < effect


def test_saves_fall_back_to_a_flat_rate_without_market_lambdas():
    frame = keeper_rows(500, a=1.0, b=0.6).assign(lambda_against=np.nan)
    fit = fit_saves(frame, keeper_prior=30.0)
    assert fit.b == 0.0 and fit.keepers == ()
    expected = np.log(frame["saves"].sum() / (frame["minutes"].sum() / 90))
    assert fit.a == pytest.approx(expected)


# --- cards and own goals --------------------------------------------------------------------


def test_card_rates_are_shrunk_toward_the_position():
    frame = rows(
        4,
        player_key=[1, 1, 2, 3],
        element_type=[2, 2, 2, 3],
        yellow_cards=[1.0, 1.0, 0.0, 0.0],
        minutes=[90.0, 90.0, 90.0, 90.0],
    )
    params = ComponentsParams(yellow_prior=10.0)
    fit = fit_rates(frame, params)
    positions = {p[0]: p[1:] for p in fit.positions}
    assert positions[2][0] == pytest.approx(2 / 3)  # 2 yellows in 3 DEF 90s
    assert positions[3][0] == 0.0
    players = {p[0]: p[1:] for p in fit.players}
    assert players[1][0] == pytest.approx((2 + 10 * 2 / 3) / (2 + 10))
    assert players[2][0] == pytest.approx((0 + 10 * 2 / 3) / (1 + 10))


def test_params_validate():
    with pytest.raises(ValueError):
        ComponentsParams(seasons=0)
    with pytest.raises(ValueError, match="defcon_k"):
        ComponentsParams(defcon_k=((2, 3.0), (3, 2.0)))  # FWD missing
    with pytest.raises(ValueError, match="defcon_r"):
        ComponentsParams(defcon_r=((2, 0.0), (3, 15.0), (4, 15.0)))
    with pytest.raises(ValueError):
        ComponentsParams(season_decay=0.0)


# --- on the synthetic league ----------------------------------------------------------------


@pytest.fixture(scope="module")
def league():
    tables = synthetic_tables(seasons=(2022, 2023), n_clubs=6, seed=3)
    gameweeks = tables["gameweek"]
    deadline = gameweeks[(gameweeks["season"] == 2023) & (gameweeks["gw"] == 5)]
    view = DataStore(tables=tables).as_of(deadline["deadline_time"].iloc[0])
    minutes = predict_minutes(view, fit_minutes(view, MINUTES))
    team = team_lambdas(view, fit_team(view))
    return view, minutes, team, fit_shares(view)


def test_training_rows_are_visible_played_and_weighted(league):
    view = league[0]
    frame = training_rows(view, ComponentsParams(season_decay=0.5))
    assert (frame["minutes"] > 0).all()
    assert set(frame["season"]) == {2022, 2023}
    assert set(frame.loc[frame["season"] == 2022, "weight"]) == {0.5}
    assert set(frame.loc[frame["season"] == 2023, "weight"]) == {1.0}
    keys = frame[["player_key", "season", "fixture_key"]]
    assert keys.equals(keys.sort_values(list(keys.columns), kind="mergesort"))
    visible = view.table("player_match", columns=["fixture_key"])["fixture_key"]
    assert frame["fixture_key"].isin(visible).all()
    one = training_rows(view, ComponentsParams(seasons=1))
    assert set(one["season"]) == {2023}


def test_predicted_rates_schema_and_goalkeeper_only_saves(league):
    view, minutes, team, shares_fit = league
    fit = fit_components(view)
    out = predict_components(view, fit, minutes, team, shares_fit)
    assert list(out.columns) == list(COMPONENT_COLUMNS)
    assert len(out) == len(minutes)
    assert out[["player_key", "fixture_key"]].equals(minutes[["player_key", "fixture_key"]])
    keeper = out["element_type"] == 1
    assert (out.loc[keeper, "saves_rate"] > 0).all()
    assert (out.loc[~keeper, ["saves_rate", "pen_save_rate"]] == 0).all().all()
    assert (out[["yellow_rate", "red_rate", "own_goal_rate", "pen_miss_ratio"]] >= 0).all().all()
    # Every row of a position carries that position's bonus coefficients.
    for position, coefficients in fit.bonus.coefficients:
        part = out[out["element_type"] == position]
        values = part[[f"b_{n}" for n in BONUS_FEATURES]].to_numpy()
        assert np.all(values == np.array(coefficients))
    again = predict_components(view, fit_components(view), minutes, team, shares_fit)
    pd.testing.assert_frame_equal(out, again)


# --- defensive contributions ----------------------------------------------------------------


def test_negbin_tail_matches_scipy():
    means = np.array([0.5, 3.0, 7.5, 12.0, 20.0])
    for r in (0.5, 2.0, 13.2, 100.0):
        for k in (1, 2, 10, 12):
            expected = nbinom.sf(k - 1, r, r / (r + means))
            np.testing.assert_allclose(negbin_tail(means, r, k), expected, rtol=1e-9, atol=1e-13)
    # Elementwise mean, r and threshold; mean 0 never reaches k >= 1; k <= 0 always does.
    out = negbin_tail(
        np.array([0.0, 6.0, 6.0]), np.array([13.2, 15.2, 15.2]), np.array([10, 0, 12])
    )
    assert out[0] == 0.0 and out[1] == 1.0
    assert out[2] == pytest.approx(nbinom.sf(11, 15.2, 15.2 / 21.2), rel=1e-9)
    with pytest.raises(ValueError):
        negbin_tail(np.array([1.0]), np.array([0.0]), np.array([10]))


def test_defcon_counts_prefer_fpl_and_fall_back_to_the_position_stats():
    frame = pd.DataFrame(
        {
            "element_type": [2, 2, 3, 4, 3, 1, 1],
            "defensive_contribution": pd.array([7, None, None, None, None, 5, None], "Int64"),
            "clearances_blocks_interceptions": pd.array([1, 6, 3, 1, 3, 2, 2], "Int64"),
            "tackles": pd.array([1, 2, 2, 1, None, 0, 0], "Int64"),
            "recoveries": pd.array([9, 9, 5, 4, 5, 1, 1], "Int64"),
        }
    )
    count = defcon_counts(frame)
    # FPL's count; DEF CBIT (no recoveries); MID/FWD CBIRT; a null component: unknown;
    # goalkeepers only through FPL's count.
    np.testing.assert_array_equal(count[:4], [7.0, 8.0, 10.0, 6.0])
    assert np.isnan(count[4]) and count[5] == 5.0 and np.isnan(count[6])


def planted(tables: dict, seed: int = 0) -> dict:
    """The tables with an FPL defcon count on every played player_match row."""
    out = {name: frame.copy() for name, frame in tables.items()}
    matches = out["player_match"]
    rng = np.random.default_rng(seed)
    count = rng.poisson(7.0, len(matches)) * (matches["minutes"].to_numpy() > 0)
    matches["defensive_contribution"] = pd.array(count, dtype="Int64")
    return out


@pytest.fixture(scope="module")
def defcon_tables():
    return planted(synthetic_tables(seasons=(2025, 2026), n_clubs=6, seed=3))


def view_at(tables: dict, season: int, gw: int):
    gameweeks = tables["gameweek"]
    row = gameweeks[(gameweeks["season"] == season) & (gameweeks["gw"] == gw)]
    return DataStore(tables=tables).as_of(row["deadline_time"].iloc[0])


def test_defcon_rates_shrink_current_season_rows_toward_the_position_mean(defcon_tables):
    view = view_at(defcon_tables, 2026, 5)
    params = ComponentsParams(defcon_k=((2, 3.0), (3, 2.0), (4, 2.0)))
    means, players = defcon_rates(view, params)
    matches = defcon_tables["player_match"]
    seen = matches[
        (matches["season"] == 2026)
        & (matches["available_at"] < view.deadline)
        & (matches["minutes"] > 0)
    ]
    assert set(seen["gw"]) == {1, 2, 3, 4}
    positions = defcon_tables["player_season"]
    positions = positions[positions["season"] == 2026][["player_key", "element_type"]]
    seen = seen.merge(positions, on="player_key")
    seen = seen[seen["element_type"] > 1]
    seen = seen.assign(
        count=seen["defensive_contribution"].astype("float64"), e=seen["minutes"] / 90.0
    )
    totals = seen.groupby("element_type")[["count", "e"]].sum()
    for position in (2, 3, 4):
        expected = totals.loc[position, "count"] / totals.loc[position, "e"]
        assert means[position] == pytest.approx(expected)
    k = {2: 3.0, 3: 2.0, 4: 2.0}
    per_player = seen.groupby(["player_key", "element_type"])[["count", "e"]].sum()
    assert set(players) == {key for key, _ in per_player.index}
    for (key, position), row in per_player.iterrows():
        expected = (row["count"] + k[position] * means[position]) / (row["e"] + k[position])
        assert players[key] == pytest.approx(expected)
    goalkeepers = positions.loc[positions["element_type"] == 1, "player_key"]
    assert not set(goalkeepers) & set(players)


def test_defcon_rates_see_only_the_visible_current_season(defcon_tables):
    view = view_at(defcon_tables, 2026, 5)
    before = defcon_rates(view, ComponentsParams())
    changed = {name: frame.copy() for name, frame in defcon_tables.items()}
    matches = changed["player_match"]
    future = (matches["season"] == 2026) & (matches["available_at"] >= view.deadline)
    earlier = matches["season"] == 2025
    assert future.any() and earlier.any()
    matches.loc[future | earlier, "defensive_contribution"] = 99
    # A planted future row of a player without visible rows changes nothing either.
    extra = matches[future].iloc[[0]].assign(player_key=999999, defensive_contribution=99)
    changed["player_match"] = pd.concat([matches, extra], ignore_index=True)
    after = defcon_rates(view_at(changed, 2026, 5), ComponentsParams())
    assert after == before


def test_defcon_rates_before_any_row_use_the_prior_means(defcon_tables):
    means, players = defcon_rates(view_at(defcon_tables, 2026, 1), ComponentsParams())
    assert means == dict(DEFCON_PRIOR_MEAN)
    assert players == {}
    # Seasons without a known count (2019/20-2024/25: all null) behave the same.
    unknown = synthetic_tables(seasons=(2025, 2026), n_clubs=6, seed=3)
    means, players = defcon_rates(view_at(unknown, 2026, 5), ComponentsParams())
    assert means == dict(DEFCON_PRIOR_MEAN) and players == {}


def test_predicted_defcon_rates_by_position(defcon_tables):
    view = view_at(defcon_tables, 2026, 5)
    minutes = predict_minutes(view, fit_minutes(view, MINUTES))
    team = team_lambdas(view, fit_team(view))
    fit = fit_components(view)
    out = predict_components(view, fit, minutes, team, fit_shares(view))
    means, players = defcon_rates(view, fit.params)
    keeper = out["element_type"] == 1
    assert keeper.any()
    assert (out.loc[keeper, ["defcon_rate", "defcon_r"]] == 0).all().all()
    outfield = out[~keeper]
    expected = [
        players.get(key, means[position])
        for key, position in zip(outfield["player_key"], outfield["element_type"], strict=True)
    ]
    np.testing.assert_allclose(outfield["defcon_rate"], expected)
    assert (outfield["defcon_rate"] > 0).all()
    np.testing.assert_array_equal(
        outfield["defcon_r"], outfield["element_type"].map(dict(fit.params.defcon_r))
    )
