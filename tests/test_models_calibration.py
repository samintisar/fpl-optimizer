"""Isotonic and linear recalibration and the season table (`fplopt.models.calibration`)."""

import numpy as np
import pytest

from fplopt.models.calibration import (
    Calibration,
    Isotonic,
    apply_isotonic,
    calibration_for,
    from_table,
    isotonic,
    linear_by_position,
    to_table,
)
from fplopt.models.calibration_table import CALIBRATION_TABLE


def test_pav_pools_violators_with_weights():
    # Outcomes 1, 0 at increasing predictions violate monotonicity: they pool to 0.5.
    curve = isotonic(np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.0, 1.0, 0.0, 1.0]))
    assert curve.y == (0.0, 0.5, 1.0)
    assert curve.x == pytest.approx((0.1, 0.25, 0.4))
    weighted = isotonic(
        np.array([0.1, 0.2, 0.3]), np.array([1.0, 0.0, 0.0]), np.array([1.0, 3.0, 1.0])
    )
    assert weighted.y == pytest.approx((0.2,))  # (1·1 + 0·3 + 0·1) / 5


def test_isotonic_recovers_a_miscalibrated_probability():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.99, 50_000)
    outcome = (rng.random(len(p)) < p**2).astype("float64")  # truth: p², the model says p
    curve = isotonic(p, outcome)
    assert len(curve.x) <= 200
    assert all(np.diff(curve.y) >= 0) and all(np.diff(curve.x) > 0)
    grid = np.array([0.2, 0.5, 0.8])
    np.testing.assert_allclose(curve.apply(grid), grid**2, atol=0.03)


def test_isotonic_keeps_zero_and_is_flat_beyond_the_ends():
    curve = Isotonic((0.1, 0.5), (0.2, 0.6))
    out = curve.apply(np.array([0.0, 0.05, 0.3, 0.9]))
    assert out.tolist() == pytest.approx([0.0, 0.2, 0.4, 0.6])
    assert isotonic(np.zeros(3), np.ones(3)) == Isotonic((), ())  # zeros are never fitted
    assert Isotonic((), ()).apply(np.array([0.3])).tolist() == [0.3]


def test_apply_isotonic_by_horizon_group():
    maps = ((0, Isotonic((0.0, 1.0), (0.0, 0.5))), (2, Isotonic((0.0, 1.0), (0.0, 1.0))))
    out = apply_isotonic(maps, np.array([0.4, 0.4, 0.4, 0.4]), np.array([0, 1, 2, 5]))
    assert out.tolist() == pytest.approx([0.2, 0.2, 0.4, 0.4])


def test_linear_by_position_recovers_planted_maps():
    rng = np.random.default_rng(1)
    n = 5000
    et = rng.integers(1, 5, n)
    p_play = rng.uniform(0, 1, n)
    xp = rng.uniform(0, 6, n)
    a = np.array([0.0, 0.3, -0.2, 0.1, 0.5])[et]
    b = np.array([0.0, 0.9, 0.8, 1.1, 0.7])[et]
    points = a * p_play + b * xp + rng.normal(scale=0.01, size=n)
    out = {p: (aa, bb) for p, aa, bb in linear_by_position(et, p_play, xp, points)}
    for position, (aa, bb) in {1: (0.3, 0.9), 2: (-0.2, 0.8), 3: (0.1, 1.1), 4: (0.5, 0.7)}.items():
        assert out[position] == pytest.approx((aa, bb), abs=0.01)


def test_table_round_trip_and_season_lookup():
    cal = Calibration(
        fitted_on=(2016, 2017),
        p_start=((0, Isotonic((0.1, 0.9), (0.2, 0.8))), (1, Isotonic((0.5,), (0.4,)))),
        p_goal=((0, Isotonic((0.1,), (0.1,))),),
        xp=((1, 0.1, 0.9),),
    )
    assert from_table(to_table(cal)) == cal
    other = Calibration(fitted_on=(2016, 2017, 2018), p_cs=((0, Isotonic((0.3,), (0.3,))),))
    table = ((2018, to_table(cal)), (2019, to_table(other)))
    assert calibration_for(2017, table) is None
    assert calibration_for(2018, table) == cal
    assert calibration_for(2019, table) == other
    assert calibration_for(2026, table) == other  # later seasons use the last entry
    assert calibration_for(2020, ()) is None


def test_the_checked_in_table_is_fitted_on_earlier_seasons_only():
    for season, entry in CALIBRATION_TABLE:
        cal = from_table(entry)
        assert cal.fitted_on and max(cal.fitted_on) < season
        assert list(cal.fitted_on) == sorted(cal.fitted_on)
        for maps in (cal.p_start, cal.p_cs, cal.p_goal):
            for _, curve in maps or ():
                assert all(np.diff(curve.y) >= 0)
    seasons = [season for season, _ in CALIBRATION_TABLE]
    assert seasons == sorted(seasons)
