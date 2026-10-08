"""The availability adjustment layer (`fplopt.models.availability`): news parsing, flag
categories, the walk-forward fit and its application to minutes predictions."""

import numpy as np
import pandas as pd
import pytest
from synthetic_season import synthetic_tables

from fplopt.features.store import DataStore
from fplopt.models.availability import (
    AvailabilityFit,
    adjust_minutes,
    expected_back,
    fit_availability,
    flag_category,
)
from fplopt.models.gbm import GbmParams
from fplopt.models.minutes import MINUTES_COLUMNS, MinutesParams, fit_minutes, predict_minutes

SMALL = GbmParams(num_boost_round=30, min_data_in_leaf=20)
PARAMS = MinutesParams(start=SMALL, sixty=SMALL, sub=SMALL)
UTC_US = pd.DatetimeTZDtype("us", "UTC")


def utc(text):
    return pd.Timestamp(text, tz="UTC")


def deadline(tables, season, gw):
    gameweeks = tables["gameweek"]
    match = gameweeks[(gameweeks["season"] == season) & (gameweeks["gw"] == gw)]
    return match["deadline_time"].iloc[0]


def view(tables, season=2023, gw=5):
    return DataStore(tables=tables).as_of(deadline(tables, season, gw))


@pytest.fixture(scope="module")
def league():
    # Three seasons: the out-of-fold bases of the flag fit need a season of each parity.
    return synthetic_tables(seasons=(2021, 2022, 2023), n_clubs=10, seed=2)


@pytest.fixture(scope="module")
def fits(league):
    minutes = fit_minutes(view(league), PARAMS)
    return minutes, fit_availability(view(league), minutes)


# --- parsing ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("news", "added", "back"),
    [
        ("Hamstring injury - Expected back 09 Dec", "2022-12-01", "2022-12-09"),
        ("Knock - expected back 3 jan", "2022-12-28", "2023-01-03"),  # year rollover
        ("Suspended until 13 Dec", "2022-12-10", "2022-12-13"),
        ("Calf injury - Expected back 28 Dec", "2023-01-10", "2022-12-28"),  # just passed
        ("Joined Lyon on loan - Expected back 01 Jul", "2022-09-15", "2023-07-01"),
        ("Expected back 29 Feb", "2024-02-10", "2024-02-29"),
        ("Expected back 29 Feb", "2023-02-01", None),  # no valid year in the window
        ("Knee injury - Unknown return date", "2022-12-01", None),
        ("Knock - 75% chance of playing", "2022-12-01", None),
        (None, "2022-12-01", None),
    ],
)
def test_expected_back(news, added, back):
    parsed = expected_back(pd.Series([news]), pd.Series([utc(added)]))
    assert parsed.dtype == UTC_US
    if back is None:
        assert parsed.isna().all()
    else:
        assert parsed.iloc[0] == utc(back)


def test_flag_categories():
    status = pd.Series(["a", "a", "d", "d", "d", "i", "s", "u", "n", None, "x"])
    chance = pd.array([None, 100, 25, 50, 75, 0, 0, 0, 0, None, None], dtype="Int64")
    categories = flag_category(status, pd.Series(chance))
    assert categories.tolist() == ["a", "a100", "d25", "d50", "d75", "i", "s", "u", "n", "", ""]


# --- the fit ------------------------------------------------------------------------------


def test_the_fit_learns_that_injured_players_do_not_start(fits):
    _, fit = fits
    offsets = {(c, b): v for c, b, v in fit.offsets}
    assert fit.n_rows > 1000
    assert offsets[("i", 0)] < -2
    assert offsets[("i", 0)] < offsets[("d50", 0)] < offsets[("a", 0)] + 1
    # Out-of-fold bases on the small synthetic league are flat, so the slope sharpens them
    # (real data, 2021/22-2022/23 cutoffs: 0.77-0.78).
    assert 0.2 < fit.slope < 5


def test_the_fit_is_deterministic_and_uses_only_visible_flags(league, fits):
    minutes, fit = fits
    assert fit_availability(view(league), minutes) == fit
    tables = {name: df.copy() for name, df in league.items()}
    snaps = tables["player_snapshot"]
    future = snaps["available_at"] >= deadline(tables, 2023, 5)
    snaps.loc[future, "status"] = "i"
    snaps.loc[future, "chance_of_playing_next_round"] = 0
    snaps.loc[future, "news"] = "Expected back 01 Jan"
    assert fit_availability(view(tables), minutes) == fit


def test_without_snapshots_nothing_is_fitted_or_adjusted():
    tables = synthetic_tables(seasons=(2022, 2023), n_clubs=10, seed=2, snapshots=False)
    v = view(tables)
    minutes = fit_minutes(v, PARAMS)
    fit = fit_availability(v, minutes)
    assert fit.n_rows == 0 and fit.offsets == () and fit.slope == 1.0
    predicted = predict_minutes(v, minutes)
    assert adjust_minutes(v, predicted, fit) is predicted


# --- adjusting ----------------------------------------------------------------------------


def newest_snapshot(tables, at):
    snaps = tables["player_snapshot"]
    newest = snaps.loc[snaps["snapshot_at"] < at, "snapshot_at"].max()
    return snaps["snapshot_at"] == newest


def test_flags_move_p_start_and_teams_stay_normalized(league, fits):
    minutes, fit = fits
    v = view(league)
    predicted = predict_minutes(v, minutes)
    adjusted = adjust_minutes(v, predicted, fit)
    assert list(adjusted.columns) == list(MINUTES_COLUMNS)
    assert (adjusted.dtypes == predicted.dtypes).all()
    pd.testing.assert_frame_equal(
        adjusted[list(MINUTES_COLUMNS[:7])], predicted[list(MINUTES_COLUMNS[:7])]
    )
    snaps = league["player_snapshot"]
    newest = snaps[newest_snapshot(league, v.deadline)]
    injured = newest.loc[newest["status"] == "i", "player_key"]
    assert len(injured)
    now = adjusted[adjusted["player_key"].isin(injured) & (adjusted["horizon"] == 0)]
    assert (now["p_start"] < 0.1).all()
    for column in ("p_start", "p_60", "p_sub", "p_play"):
        assert adjusted[column].between(0, 1).all()
    assert (adjusted["e_minutes"] <= 90 * adjusted["p_play"] + 1e-9).all()
    sums = adjusted.groupby(["fixture_key", "team_key"])["p_start"].sum()
    # The team shift is bounded (|δ| ≤ max_shift), so a club can stay a little off 11.
    assert (sums <= 11.5).all() and (sums > 10).all()
    assert (sums - 11).abs().median() < 1e-6


def test_banned_fixtures_keep_their_residual(league, fits):
    """A predicted ban keeps the minutes model's P(start) through the flag mapping and the
    team re-normalization, even for a flagged player."""
    minutes, fit = fits
    v = view(league)
    predicted = predict_minutes(v, minutes)
    snaps = league["player_snapshot"]
    newest = snaps[newest_snapshot(league, v.deadline)]
    injured = newest.loc[newest["status"] == "i", "player_key"]
    ban = predicted["player_key"].isin(injured)
    assert ban.any()
    adjusted = adjust_minutes(v, predicted.assign(banned=ban.to_numpy()), fit)
    assert adjusted["banned"].tolist() == ban.tolist()
    np.testing.assert_allclose(adjusted.loc[ban, "p_start"], predicted.loc[ban, "p_start"])
    np.testing.assert_allclose(adjusted.loc[ban, "p_sub"], predicted.loc[ban, "p_sub"])


def test_a_return_date_zeroes_the_fixtures_before_it(league, fits):
    minutes, _ = fits
    identity = AvailabilityFit(1.0, (), minutes.minutes_sub, 5.0, 2.0, 0)
    tables = {name: df.copy() for name, df in league.items()}
    v = view(tables)
    predicted = predict_minutes(v, minutes)
    key = int(predicted.loc[predicted["p_start"].idxmax(), "player_key"])
    schedule = tables["schedule"].set_index("fixture_key")["kickoff_time"]
    mine = predicted[predicted["player_key"] == key].sort_values("horizon")
    back = schedule.loc[mine["fixture_key"].iloc[2]].normalize()  # the day of his 3rd
    snaps = tables["player_snapshot"]
    at = newest_snapshot(tables, v.deadline) & (snaps["player_key"] == key)
    snaps.loc[at, "status"] = "i"
    snaps.loc[at, "chance_of_playing_next_round"] = 0
    snaps.loc[at, "news"] = f"Calf injury - Expected back {back:%d %b}"
    snaps.loc[at, "news_added"] = v.deadline - pd.Timedelta(days=3)
    adjusted = adjust_minutes(view(tables), predicted, identity)
    out = adjusted[adjusted["player_key"] == key].sort_values("horizon")
    assert (out[["p_start", "p_sub", "p_play", "e_minutes"]].iloc[:2] == 0).all().all()
    np.testing.assert_allclose(out["p_start"].iloc[2:], mine["p_start"].iloc[2:], atol=0.05)
    sums = adjusted.groupby(["fixture_key", "team_key"])["p_start"].sum()
    np.testing.assert_allclose(sums, 11.0, atol=1e-6)
