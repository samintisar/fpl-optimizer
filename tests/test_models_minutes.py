"""The minutes hurdle model (`fplopt.models.minutes`) on the synthetic league."""

import dataclasses

import numpy as np
import pandas as pd
import pytest
from synthetic_season import synthetic_tables

from fplopt.features.history import prediction_frame
from fplopt.features.store import DataStore
from fplopt.models.gbm import Gbm, GbmParams
from fplopt.models.minutes import (
    MINUTES_COLUMNS,
    Constant,
    MinutesParams,
    finish_minutes,
    fit_minutes,
    predict_minutes,
)

SMALL = GbmParams(num_boost_round=30, min_data_in_leaf=20)
PARAMS = MinutesParams(start=SMALL, sixty=SMALL, sub=SMALL)


@pytest.fixture(scope="module")
def league():
    return synthetic_tables(seasons=(2022, 2023), n_clubs=10, seed=2)


def deadline(tables, season, gw):
    gameweeks = tables["gameweek"]
    match = gameweeks[(gameweeks["season"] == season) & (gameweeks["gw"] == gw)]
    return match["deadline_time"].iloc[0]


def view(tables, season=2023, gw=5):
    return DataStore(tables=tables).as_of(deadline(tables, season, gw))


@pytest.fixture(scope="module")
def fitted(league):
    return fit_minutes(view(league), PARAMS)


def test_fit_has_gbm_parts_and_minute_constants(fitted):
    assert isinstance(fitted.start, Gbm) and isinstance(fitted.sixty, Gbm)
    assert 80 < fitted.minutes_sixty <= 90
    assert 0 < fitted.minutes_short < 60
    assert 0 < fitted.minutes_sub < 60
    assert fitted.n_rows > 3000


def test_predictions_are_probabilities_with_fixed_schema(league, fitted):
    out = predict_minutes(view(league), fitted)
    assert list(out.columns) == list(MINUTES_COLUMNS)
    assert out.index.equals(pd.RangeIndex(len(out)))
    assert (out.dtypes.iloc[:7] == "int64").all() and (out.dtypes.iloc[7:-1] == "float64").all()
    assert out["banned"].dtype == bool
    keys = ["player_key", "horizon", "fixture_key"]
    assert out.equals(out.sort_values(keys, kind="mergesort").reset_index(drop=True))
    assert not out.duplicated(["player_key", "fixture_key"]).any()
    for column in ("p_start", "p_60", "p_sub", "p_play"):
        assert out[column].between(0, 1).all(), column
    assert (out["p_60"] <= out["p_start"] + 1e-12).all()
    np.testing.assert_allclose(out["p_play"], out["p_start"] + out["p_sub"])
    assert (out["e_minutes"] >= 0).all()
    assert (out["e_minutes"] <= 90 * out["p_play"] + 1e-9).all()
    per_player = out.groupby(["player_key", "gw"])["e_minutes"].sum()
    fixtures = out.groupby(["player_key", "gw"]).size()
    assert (per_player <= 90 * fixtures).all()
    pool = prediction_frame(view(league))
    assert len(out) == len(pool)


def test_team_sums_are_normalized(league, fitted):
    wide = dataclasses.replace(PARAMS, max_shift=5.0, max_scale=2.0)
    out = predict_minutes(view(league), dataclasses.replace(fitted, params=wide))
    sums = out.groupby(["fixture_key", "team_key"])[["p_start", "e_minutes"]].sum()
    np.testing.assert_allclose(sums["p_start"], 11.0, atol=1e-6)
    np.testing.assert_allclose(sums["e_minutes"], 990.0, atol=1e-6)
    # With the default bounds a team whose pool can't reach 11 starters stays below.
    bounded = predict_minutes(view(league), fitted)
    sums = bounded.groupby(["fixture_key", "team_key"])[["p_start", "e_minutes"]].sum()
    assert (sums["p_start"] <= 11.0 + 1e-6).all() and (sums["p_start"] > 10.0).all()
    assert (sums["p_start"] > 11.0 - 1e-6).mean() > 0.9


def test_normalization_is_bounded_and_keeps_fixed_rows():
    frame = pd.DataFrame(
        {
            "player_key": range(1, 7),
            "fixture_key": 1,
            "team_key": 1,
            "season": 2023,
            "gw": 1,
            "gw_index": 1,
            "horizon": 0,
            "p_start": [0.5] * 5 + [0.0],
            "c60": 0.9,
            "csub": [0.3] * 5 + [0.0],
            "m_start": 80.0,
            "fixed": [False] * 5 + [True],
        }
    )
    out = finish_minutes(frame, minutes_sub=20.0, max_shift=1.0, max_scale=1.25)
    expected = 1 / (1 + np.exp(-1.0))  # the shift stops at its bound
    np.testing.assert_allclose(out["p_start"].iloc[:5], expected)
    assert out["p_start"].iloc[5] == 0 and out["e_minutes"].iloc[5] == 0
    raw = expected * 80 + (1 - expected) * 0.3 * 20
    p_play = expected + (1 - expected) * 0.3
    np.testing.assert_allclose(out["e_minutes"].iloc[:5], min(raw * 1.25, 90 * p_play))
    assert raw * 1.25 > 90 * p_play  # the cap at 90 minutes x P(plays) binds here
    out = finish_minutes(frame.assign(m_start=60.0), 20.0, 1.0, 1.25)
    raw = expected * 60 + (1 - expected) * 0.3 * 20
    np.testing.assert_allclose(out["e_minutes"].iloc[:5], raw * 1.25)  # the scale bound
    big = frame.assign(p_start=[0.99] * 5 + [0.0], fixed=False)
    out = finish_minutes(big.iloc[:5].assign(player_key=range(1, 6)), 20.0, 1.0, 1.25)
    assert out["p_start"].sum() <= 5


def test_horizon_decay_moves_later_gws_to_the_long_run_rate(league, fitted):
    v = view(league)
    frame = prediction_frame(v)
    p_model = fitted.start.predict(frame)
    still = dataclasses.replace(PARAMS, max_shift=0.0, max_scale=1.0)
    flat = predict_minutes(
        v, dataclasses.replace(fitted, params=dataclasses.replace(still, horizon_decay=1.0))
    )
    np.testing.assert_allclose(flat["p_start"], p_model, rtol=1e-9)
    decayed = predict_minutes(
        v, dataclasses.replace(fitted, params=dataclasses.replace(still, horizon_decay=0.0))
    )
    n = frame["n_long"].to_numpy()
    long_run = (np.nan_to_num(frame["start_rate_long"].to_numpy()) * n + 5.0 * p_model) / (n + 5.0)
    expected = np.where(frame["horizon"] == 0, p_model, long_run)
    np.testing.assert_allclose(decayed["p_start"], expected, rtol=1e-9)
    half = predict_minutes(
        v, dataclasses.replace(fitted, params=dataclasses.replace(still, horizon_decay=0.5))
    )
    w = 0.5 ** frame["horizon"].to_numpy()
    np.testing.assert_allclose(half["p_start"], w * p_model + (1 - w) * long_run, rtol=1e-9)


def regular_starter(tables, season=2023, gw=4):
    pm = tables["player_match"]
    rows = pm[(pm["season"] == season) & (pm["gw"] <= gw)]
    started = rows.groupby("player_key")["minutes"].min()
    return int(started[started == 90].index[0])


@pytest.mark.parametrize(("cards", "banned"), [((0, 1), 1), ((5, 0), 1)])
def test_suspensions_zero_the_banned_fixtures(league, fitted, cards, banned):
    tables = {name: df.copy() for name, df in league.items()}
    key = regular_starter(tables)
    pm = tables["player_match"]
    at = (pm["player_key"] == key) & (pm["season"] == 2023) & (pm["gw"] == 4)
    pm.loc[at, ["yellow_cards", "red_cards"]] = cards
    wide = dataclasses.replace(PARAMS, max_shift=5.0, max_scale=2.0)
    out = predict_minutes(view(tables), dataclasses.replace(fitted, params=wide))
    mine = out[out["player_key"] == key].sort_values("horizon")
    assert (
        (mine[["p_start", "p_60", "p_sub", "p_play", "e_minutes"]].iloc[:banned] == 0).all().all()
    )
    assert (mine["p_start"].iloc[banned:] > 0.3).all()
    sums = out.groupby(["fixture_key", "team_key"])["p_start"].sum()
    np.testing.assert_allclose(sums, 11.0, atol=1e-6)


def test_ban_residual_uses_the_banned_rows_start_rate(league, fitted):
    from fplopt.features.history import training_frame

    assert fitted.ban_start == 0.0 and fitted.ban_sub == 0.0  # off by default
    v = view(league)
    rows = training_frame(v)
    banned = rows[rows["banned"] != 0]
    assert len(banned) > 0
    weight = 0.7 ** (2023 - banned["season"]).to_numpy(dtype="float64")
    residual = fit_minutes(v, dataclasses.replace(PARAMS, ban_residual=True))
    expected = float((banned["start"] * weight).sum() / weight.sum())
    assert residual.ban_start == pytest.approx(expected)
    assert residual.start.booster.model_to_string() == fitted.start.booster.model_to_string()

    tables = {name: df.copy() for name, df in league.items()}
    key = regular_starter(tables)
    pm = tables["player_match"]
    at = (pm["player_key"] == key) & (pm["season"] == 2023) & (pm["gw"] == 4)
    pm.loc[at, "red_cards"] = 1
    planted = dataclasses.replace(residual, ban_start=0.2, ban_sub=0.5)
    out = predict_minutes(view(tables), planted)
    first = out[out["player_key"] == key].sort_values("horizon").iloc[0]
    assert first["p_start"] == pytest.approx(0.2)  # held fixed through the normalization
    assert first["p_sub"] == pytest.approx(0.8 * 0.5)


def test_fits_are_deterministic(league, fitted):
    again = fit_minutes(view(league), PARAMS)
    assert again.start.booster.model_to_string() == fitted.start.booster.model_to_string()
    pd.testing.assert_frame_equal(
        predict_minutes(view(league), again), predict_minutes(view(league), fitted)
    )


def test_only_visible_data_is_used(league, fitted):
    tables = {name: df.copy() for name, df in league.items()}
    at = deadline(tables, 2023, 5)
    rng = np.random.default_rng(0)
    for name in ("player_match", "player_gw", "player_snapshot"):
        frame = tables[name]
        future = frame["available_at"] >= at
        if name == "player_match":
            frame.loc[future, "minutes"] = rng.integers(0, 91, int(future.sum()))
            frame.loc[future, "red_cards"] = 1
            frame.loc[future, "starts"] = 0
        if name == "player_gw":
            frame.loc[future, "value"] = 99
        if name == "player_snapshot":
            frame.loc[future, "now_cost"] = 99
    corrupted = fit_minutes(view(tables), PARAMS)
    assert corrupted.start.booster.model_to_string() == fitted.start.booster.model_to_string()
    pd.testing.assert_frame_equal(
        predict_minutes(view(tables), corrupted), predict_minutes(view(league), fitted)
    )


def test_too_few_rows_fall_back_to_constants():
    tables = synthetic_tables(seasons=(2023,), n_clubs=6, seed=1)
    v = view(tables, 2023, 2)  # one GW of 90 rows visible
    fit = fit_minutes(v, PARAMS)
    assert isinstance(fit.start, Constant) and isinstance(fit.sub, Constant)
    out = predict_minutes(v, fit)
    sums = out.groupby(["fixture_key", "team_key"])["p_start"].sum()
    np.testing.assert_allclose(sums, 11.0, atol=1e-6)
    first = fit_minutes(view(tables, 2023, 1), PARAMS)  # nothing visible at all
    assert first.n_rows == 0 and first.start.rate == 0.5
