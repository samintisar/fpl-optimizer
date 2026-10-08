"""xP assembly, calibration and the `v1` model (`fplopt.models.assemble`, `.calibration`)."""

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest
from synthetic_season import synthetic_tables

import fplopt.models.assemble as assemble
from fplopt.backtest.rules import backtest_rules
from fplopt.backtest.simulator import Caches
from fplopt.features.baseline import HORIZON, player_pool
from fplopt.features.store import DataStore
from fplopt.models import MODELS
from fplopt.models.assemble import (
    FIXTURE_COLUMNS,
    GW_COMPONENTS,
    POINT_COLUMNS,
    V1Params,
    calibrate_inputs,
    calibrate_xp,
    fit_v1,
    fixture_components,
    fixture_points,
    gw_frame,
    poisson_expected_floor,
    poisson_tail,
    predict_fixtures,
    predict_v1,
)
from fplopt.models.baseline import XP_DTYPES
from fplopt.models.calibration import Calibration, Isotonic
from fplopt.models.components import BONUS_FEATURES
from fplopt.models.fitted import FittedModel
from fplopt.models.gbm import GbmParams
from fplopt.models.minutes import MinutesParams

RULES = backtest_rules(2023)  # 2026-27 rules without defcon: GK goal 10, DEF 6, MID 5, FWD 4
SMALL = GbmParams(num_boost_round=20, min_data_in_leaf=20)
FAST = V1Params(minutes=MinutesParams(start=SMALL, sixty=SMALL, sub=SMALL))


def poisson(n: int, rate: float) -> float:
    return math.exp(-rate) * rate**n / math.factorial(n)


def expected_floor(rate: float, k: int) -> float:
    return sum(poisson(n, rate) * (n // k) for n in range(60))


def tail(rate: float, k: int) -> float:
    return 1.0 - sum(poisson(n, rate) for n in range(k))


def fixture_row(**values) -> dict:
    row = {name: 0.0 for name in FIXTURE_COLUMNS}
    row.update(
        player_key=1,
        fixture_key=1,
        team_key=1,
        season=2023,
        gw=1,
        gw_index=1,
        horizon=0,
        opponent_team_key=2,
        m_sixty=90.0,
        m_sub=20.0,
    )
    row.update(values)
    return row


def frame(*rows: dict) -> pd.DataFrame:
    out = pd.DataFrame(list(rows))[list(FIXTURE_COLUMNS)]
    return out.reset_index(drop=True)


# --- Poisson helpers ------------------------------------------------------------------------


def test_poisson_helpers_match_explicit_sums():
    rates = np.array([0.0, 0.3, 1.0, 2.7, 6.0])
    for k in (1, 2, 3):
        np.testing.assert_allclose(
            poisson_expected_floor(rates, k), [expected_floor(r, k) for r in rates], atol=1e-12
        )
        np.testing.assert_allclose(poisson_tail(rates, k), [tail(r, k) for r in rates], atol=1e-12)


# --- points per fixture: hand values ----------------------------------------------------------


def bonus(**coefficients) -> dict:
    return {f"b_{name}": coefficients.get(name, 0.0) for name in BONUS_FEATURES}


def test_points_of_a_forward_by_hand():
    # Starts and plays 60+ with probability 0.7; 0.2 more plays 1-59 minutes (30 on average).
    row = fixture_row(
        element_type=4,
        p_play=0.9,
        p_60=0.7,
        e_minutes=0.7 * 90 + 0.2 * 30,
        e_np_goals=0.4,
        e_pen_goals=0.1,
        e_assists=0.2,
        p_cs=0.3,
        lambda_against=1.2,
        yellow_rate=0.1,
        pen_miss_ratio=0.25,
        **bonus(play=0.1),
    )
    out = fixture_points(frame(row), RULES).iloc[0]
    assert out["pts_appearance"] == pytest.approx(0.9 * 1 + 0.7 * 1)
    assert out["pts_goals"] == pytest.approx(0.5 * 4)
    assert out["pts_assists"] == pytest.approx(0.2 * 3)
    assert out["pts_clean_sheet"] == 0.0 and out["pts_conceded"] == 0.0
    assert out["pts_bonus"] == pytest.approx(0.09)
    assert out["pts_yellow"] == pytest.approx(-0.1 * 69 / 90)
    assert out["pts_pen_miss"] == pytest.approx(-2 * 0.1 * 0.25)
    assert out["xp"] == pytest.approx(1.6 + 2.0 + 0.6 + 0.09 - 0.1 * 69 / 90 - 0.05)
    assert out["xp"] == pytest.approx(out[list(POINT_COLUMNS)].sum())


def test_points_of_a_defender_by_hand():
    row = fixture_row(
        element_type=2,
        p_play=1.0,
        p_60=1.0,
        e_minutes=90.0,
        e_np_goals=0.1,
        e_assists=0.1,
        p_cs=0.4,
        lambda_against=1.0,
        red_rate=0.01,
        own_goal_rate=0.02,
        **bonus(cs=1.0),
    )
    out = fixture_points(frame(row), RULES).iloc[0]
    conceded = expected_floor(1.0, 2)
    assert out["pts_clean_sheet"] == pytest.approx(0.4 * 4)
    assert out["pts_conceded"] == pytest.approx(-conceded)
    assert out["pts_bonus"] == pytest.approx(0.4)
    assert out["xp"] == pytest.approx(
        2 + 0.1 * 6 + 0.1 * 3 + 1.6 - conceded + 0.4 - 3 * 0.01 - 2 * 0.02
    )


def test_points_of_a_goalkeeper_by_hand():
    row = fixture_row(
        element_type=1,
        p_play=1.0,
        p_60=1.0,
        e_minutes=90.0,
        p_cs=0.3,
        lambda_against=1.5,
        saves_rate=3.0,
        pen_save_rate=0.02,
        **bonus(saves3=0.5),
    )
    out = fixture_points(frame(row), RULES).iloc[0]
    saves = expected_floor(3.0, 3)
    assert out["pts_saves"] == pytest.approx(saves)
    assert out["e_saves"] == pytest.approx(3.0)
    assert out["pts_pen_saves"] == pytest.approx(5 * 0.02)
    assert out["pts_bonus"] == pytest.approx(0.5 * saves)
    assert out["xp"] == pytest.approx(
        2 + 0.3 * 4 - expected_floor(1.5, 2) + saves + 0.1 + 0.5 * saves
    )


def test_points_of_a_midfielder_by_hand():
    # Plays 60+ (90 minutes) with probability 0.5, never 1-59.
    row = fixture_row(
        element_type=3,
        p_play=0.5,
        p_60=0.5,
        e_minutes=45.0,
        e_np_goals=0.3,
        e_assists=0.2,
        p_cs=0.2,
        lambda_against=2.0,
        **bonus(goal1=1.0, assist2=2.0),
    )
    out = fixture_points(frame(row), RULES).iloc[0]
    p_goal = 0.5 * (1 - math.exp(-0.6))  # goals | plays ~ Poisson(0.3 / 0.5)
    assert out["p_goal"] == pytest.approx(p_goal)
    assert out["pts_bonus"] == pytest.approx(p_goal + 2 * 0.5 * tail(0.4, 2))
    assert out["pts_clean_sheet"] == pytest.approx(0.5 * 0.2 * 1)
    assert out["pts_conceded"] == 0.0  # midfielders lose nothing for goals conceded
    assert out["xp"] == pytest.approx(0.5 * 2 + 0.3 * 5 + 0.2 * 3 + 0.1 + out["pts_bonus"])


def test_the_minutes_split_for_the_floor_terms():
    # A defender who plays 60+ (90 min) w.p. 0.5 and 1-59 w.p. 0.3, 30 minutes on average.
    row = fixture_row(
        element_type=2, p_play=0.8, p_60=0.5, e_minutes=0.5 * 90 + 0.3 * 30, lambda_against=2.4
    )
    out = fixture_points(frame(row), RULES).iloc[0]
    expected = 0.5 * expected_floor(2.4, 2) + 0.3 * expected_floor(2.4 / 3, 2)
    assert out["pts_conceded"] == pytest.approx(-expected)


def test_nobody_scores_without_minutes():
    row = fixture_row(element_type=2, lambda_against=2.0, p_cs=0.5, saves_rate=0.0)
    out = fixture_points(frame(row), RULES).iloc[0]
    assert out["xp"] == 0.0 and out["p_goal"] == 0.0


# --- calibration ----------------------------------------------------------------------------


def conditional_row(**values) -> dict:
    # p_start 0.5, P(60 | start) 0.8, P(sub | no start) 0.4, 80 min per start, 20 per sub.
    p_start, c60, csub = 0.5, 0.8, 0.4
    p_sub = (1 - p_start) * csub
    base = dict(
        element_type=3,
        p_start=p_start,
        p_60=p_start * c60,
        p_sub=p_sub,
        p_play=p_start + p_sub,
        e_minutes=p_start * 80 + p_sub * 20,
        c60=c60,
        csub=csub,
        m_start=80.0,
        e_np_goals=0.2,
        e_pen_goals=0.05,
        e_assists=0.1,
        p_cs=0.3,
    )
    base.update(values)
    return fixture_row(**base)


def doubling(first: int = 0) -> tuple[tuple[int, Isotonic], ...]:
    return ((first, Isotonic((0.01, 0.5), (0.02, 1.0))),)


def test_calibrating_p_start_carries_through_minutes_and_goals():
    before = frame(conditional_row())
    after = calibrate_inputs(before, Calibration((2022,), p_start=doubling())).iloc[0]
    assert after["p_start"] == pytest.approx(1.0)
    assert after["p_60"] == pytest.approx(0.8)
    assert after["p_sub"] == pytest.approx(0.0)
    assert after["p_play"] == pytest.approx(1.0)
    assert after["e_minutes"] == pytest.approx(80.0)
    ratio = 80.0 / before["e_minutes"].iloc[0]
    assert after["e_np_goals"] == pytest.approx(0.2 * ratio)
    assert after["e_assists"] == pytest.approx(0.1 * ratio)
    # Unchanged without a calibration, or with an identity map.
    assert calibrate_inputs(before, None) is before
    identity = ((0, Isotonic((0.0, 1.0), (0.0, 1.0))),)
    same = calibrate_inputs(before, Calibration((2022,), p_start=identity))
    pd.testing.assert_frame_equal(same, before)


def test_calibrating_p_cs_and_p_goal():
    before = frame(conditional_row(), conditional_row(player_key=2, horizon=1, p_start=0.0))
    cal = Calibration((2022,), p_cs=doubling(), p_goal=((0, Isotonic((0.0, 1.0), (0.0, 0.5))),))
    after = calibrate_inputs(before, cal)
    assert after["p_cs"].tolist() == pytest.approx([0.6, 0.6])
    p_play = before["p_play"].iloc[0]
    p_goal = p_play * (1 - math.exp(-0.25 / p_play))
    goals = after["e_np_goals"].iloc[0] + after["e_pen_goals"].iloc[0]
    assert p_play * (1 - math.exp(-goals / p_play)) == pytest.approx(p_goal / 2)
    assert after["e_np_goals"].iloc[0] / after["e_pen_goals"].iloc[0] == pytest.approx(4.0)


def test_isotonic_maps_by_horizon_group_and_zero_stays_zero():
    before = frame(
        conditional_row(),
        conditional_row(player_key=2, horizon=3),
        conditional_row(player_key=3, horizon=0, p_start=0.0),
    )
    split = (*doubling(0)[:1], (1, Isotonic((0.0, 1.0), (0.0, 1.0))))
    after = calibrate_inputs(before, Calibration((2022,), p_start=split))
    assert after["p_start"].tolist() == pytest.approx([1.0, 0.5, 0.0])


def test_linear_xp_recalibration_per_position():
    rows = frame(
        fixture_row(element_type=3, p_play=0.5),
        fixture_row(player_key=2, element_type=4, p_play=1.0),
    ).assign(xp=[2.0, 3.0])
    cal = Calibration((2022,), xp=((3, 0.2, 0.9),))
    out = calibrate_xp(rows, cal)
    assert out["xp"].tolist() == pytest.approx([0.2 * 0.5 + 0.9 * 2.0, 3.0])
    assert calibrate_xp(rows, None) is rows


# --- v1 on the synthetic league ----------------------------------------------------------------


@pytest.fixture(scope="module")
def league():
    # Club 1 blanks in GW5 (that fixture moves to GW7, a double for both clubs).
    return synthetic_tables(seasons=(2022, 2023), n_clubs=6, seed=3, blank=(2023, 5, 1))


def view_at(tables, gw: int, season: int = 2023):
    gameweeks = tables["gameweek"]
    row = gameweeks[(gameweeks["season"] == season) & (gameweeks["gw"] == gw)]
    return DataStore(tables=tables).as_of(row["deadline_time"].iloc[0])


def test_v1_frame_schema_doubles_and_blanks(league):
    view = view_at(league, 4)
    fit = fit_v1(view, FAST)
    fixtures = predict_fixtures(view, fit)
    out = gw_frame(view, fixtures)
    base = [name for name, _ in XP_DTYPES]
    assert list(out.columns) == [*base, *GW_COMPONENTS]
    assert dict(out.dtypes) == {
        **{name: np.dtype(dtype) for name, dtype in XP_DTYPES},
        **{name: np.dtype("float64") for name in GW_COMPONENTS},
    }
    assert out.index.equals(pd.RangeIndex(len(out)))
    pool = player_pool(view)
    assert len(out) == len(pool) * (HORIZON + 1)
    assert out[["player_key", "horizon"]].equals(
        out[["player_key", "horizon"]].sort_values(["player_key", "horizon"], kind="mergesort")
    )
    club = pool.loc[pool["team_key"] == 1, "player_key"]
    mine = out[out["player_key"].isin(club)]
    blank = mine[mine["gw"] == 5]
    assert (blank["xp"] == 0).all() and (blank[list(GW_COMPONENTS)] == 0).all().all()
    double = mine[mine["gw"] == 7]
    in_double = fixtures[(fixtures["gw"] == 7) & fixtures["player_key"].isin(club)]
    assert (in_double.groupby("player_key").size() == 2).all()
    sums = in_double.groupby("player_key")[["xp", "e_minutes"]].sum()
    np.testing.assert_allclose(double["xp"], sums.loc[double["player_key"], "xp"])
    np.testing.assert_allclose(double["e_minutes"], sums.loc[double["player_key"], "e_minutes"])
    assert double["p_start"].isna().all() and double["p_cs"].isna().all()
    single = out[(out["gw"] == 6)].merge(
        fixtures[fixtures["gw"] == 6], on="player_key", suffixes=("", "_fixture")
    )
    np.testing.assert_allclose(single["p_cs"], single["p_60_fixture"] * single["p_cs_fixture"])
    np.testing.assert_allclose(single["p_min_0"] + single["p_min_1_59"] + single["p_min_60"], 1.0)
    assert (out["xp"].abs() < 20).all() and out["xp"].notna().all()


def test_fixture_components_cover_the_minutes_rows(league):
    view = view_at(league, 4)
    fit = fit_v1(view, FAST)
    raw = fixture_components(view, fit)
    assert list(raw.columns) == list(FIXTURE_COLUMNS)
    assert not raw.isna().any().any()
    assert raw[["player_key", "fixture_key"]].duplicated().sum() == 0
    pd.testing.assert_frame_equal(raw, fixture_components(view, fit_v1(view, FAST)))


def test_v1_is_deterministic_and_registered(league):
    model = MODELS["v1"]
    assert isinstance(model, FittedModel)
    assert model.fit is fit_v1 and model.predict is predict_v1
    first = model(view_at(league, 6))
    second = model(
        view_at(synthetic_tables(seasons=(2022, 2023), n_clubs=6, seed=3, blank=(2023, 5, 1)), 6)
    )
    pd.testing.assert_frame_equal(first, second)


def test_v1_fits_in_order_at_the_cutoff(league, monkeypatch):
    calls = []

    def recorder(name, real):
        def wrapped(view, *args, **kwargs):
            calls.append((name, view.deadline))
            return real(view, *args, **kwargs)

        return wrapped

    for name in ("fit_team", "fit_minutes", "fit_availability", "fit_shares", "fit_components"):
        monkeypatch.setattr(assemble, name, recorder(name, getattr(assemble, name)))
    view = view_at(league, 7)
    cutoff = MODELS["v1"].cutoff(view)
    assert cutoff == view_at(league, 5).deadline
    fit_v1(view.earlier(cutoff), FAST)
    assert [name for name, _ in calls] == [
        "fit_team",
        "fit_minutes",
        "fit_availability",
        "fit_shares",
        "fit_components",
    ]
    assert {deadline for _, deadline in calls} == {cutoff}


FITS: list = []


def counting_fit(view):
    FITS.append(view.deadline)
    return fit_v1(view, FAST)


def test_caches_memoize_v1_fits_by_cutoff(league, monkeypatch):
    monkeypatch.setitem(MODELS, "v1_counted", FittedModel(counting_fit, predict_v1))
    FITS.clear()
    store, caches = DataStore(tables=league), Caches()
    gameweeks = league["gameweek"]
    deadlines = gameweeks[gameweeks["season"] == 2023].sort_values("gw")["deadline_time"]
    frames = [caches.xp(store, "v1_counted", store.as_of(d)) for d in deadlines.iloc[4:9]]
    assert FITS == [deadlines.iloc[4], deadlines.iloc[8]]  # GW5-8 share a fit, GW9 refits
    direct = MODELS["v1_counted"](store.as_of(deadlines.iloc[5]))
    pd.testing.assert_frame_equal(frames[1], direct)


def test_calibration_switch_and_season_table(league, monkeypatch):
    view = view_at(league, 4)
    fit = fit_v1(view, FAST)
    plain = predict_fixtures(view, fit, calibration=None)
    cal = Calibration((2022,), xp=((1, 0.0, 2.0), (2, 0.0, 2.0), (3, 0.0, 2.0), (4, 0.0, 2.0)))
    doubled = predict_fixtures(view, fit, calibration=cal)
    np.testing.assert_allclose(doubled["xp"], 2 * plain["xp"])
    monkeypatch.setattr(assemble, "calibration_for", lambda season: cal)
    np.testing.assert_allclose(predict_fixtures(view, fit)["xp"], doubled["xp"])
    off = dataclasses.replace(fit, params=dataclasses.replace(fit.params, calibrate=False))
    np.testing.assert_allclose(predict_fixtures(view, off)["xp"], plain["xp"])
