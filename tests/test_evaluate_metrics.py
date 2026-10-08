import math

import numpy as np
import pandas as pd
import pytest

from fplopt.backtest.rules import backtest_rules
from fplopt.evaluate.metrics import (
    COMPONENTS,
    Component,
    band_table,
    brier_decomposition,
    cluster_means,
    component_metrics,
    diebold_mariano,
    lineup_regret,
    mae,
    minutes_class,
    mse,
    newey_west_variance,
    ordinal_log_loss,
    poisson_log_likelihood,
    ranked_probability_score,
    reliability_table,
)


def test_mse_and_mae_by_hand():
    assert mse([1, 2, 3], [1, 1, 1]) == pytest.approx(5 / 3)
    assert mse([1, 2, 3], [1, 1, 1], weights=[1, 1, 2]) == pytest.approx(9 / 4)
    assert mae([1, 2, 3], [1, 1, 1]) == pytest.approx(1.0)
    assert math.isnan(mse([], []))
    with pytest.raises(ValueError, match="length"):
        mse([1, 2], [1])
    with pytest.raises(ValueError, match="weights"):
        mse([1, 2], [1, 1], weights=[-1, 2])


def test_cluster_means_are_ordered_by_label():
    means = cluster_means([1.0, 3.0, 10.0, 0.0], ["b", "b", "a", "c"])
    assert means.tolist() == [10.0, 2.0, 0.0]


def test_newey_west_variance_by_hand():
    # mean 2.5, deviations -1.5, -0.5, 0.5, 1.5: gamma0 = 1.25, gamma1 = 1.25 / 4.
    assert newey_west_variance([1, 2, 3, 4]) == pytest.approx(1.25)
    assert newey_west_variance([1, 2, 3, 4], lag=1) == pytest.approx(1.25 + 2 * 0.5 * 0.3125)
    with pytest.raises(ValueError):
        newey_west_variance([1, 2], lag=-1)


def test_diebold_mariano_known_case():
    # Cluster means of a - b: g1 (0 + 2) / 2 = 1, g2 = 2, g3 = 3, g4 = 4 (unequal sizes:
    # clusters are averaged first, rows are not pooled).
    loss_a = [0.0, 2.0, 2.0, 3.0, 4.0]
    loss_b = [0.0, 0.0, 0.0, 0.0, 0.0]
    clusters = [1, 1, 2, 3, 4]
    result = diebold_mariano(loss_a, loss_b, clusters)
    # lag 0: V = 1.25 (1/n), HLN factor sqrt((n - 1) / n): the one-sample t-test.
    expected_t = 2.5 / math.sqrt(1.25 / 4) * math.sqrt(3 / 4)
    assert result.mean_diff == pytest.approx(2.5)
    assert result.t_stat == pytest.approx(expected_t)
    assert result.n_clusters == 4 and result.lag == 0
    stats = pytest.importorskip("scipy.stats")
    ttest = stats.ttest_1samp([1.0, 2.0, 3.0, 4.0], 0.0, alternative="greater")
    assert result.t_stat == pytest.approx(ttest.statistic)
    assert result.p_b_better == pytest.approx(ttest.pvalue)
    assert result.p_a_better == pytest.approx(stats.t.cdf(expected_t, 3))
    # Swapping the arms flips the sign and the one-sided p-values.
    swapped = diebold_mariano(loss_b, loss_a, clusters)
    assert swapped.t_stat == pytest.approx(-expected_t)
    assert swapped.p_a_better == pytest.approx(result.p_b_better)


def test_diebold_mariano_with_lag_uses_newey_west_and_hln():
    d = [1.0, 2.0, 3.0, 4.0]
    result = diebold_mariano(d, [0.0] * 4, [1, 2, 3, 4], lag=1)
    k, n = 2, 4
    hln = math.sqrt((n + 1 - 2 * k + k * (k - 1) / n) / n)
    expected = 2.5 / math.sqrt(1.5625 / 4) * hln
    assert result.t_stat == pytest.approx(expected)
    stats = pytest.importorskip("scipy.stats")
    assert result.p_a_better == pytest.approx(stats.t.cdf(expected, 3))
    assert result.p_b_better == pytest.approx(stats.t.sf(expected, 3))


def test_diebold_mariano_degenerate_cases():
    one = diebold_mariano([1.0, 2.0], [0.0, 0.0], [1, 1])
    assert one.n_clusters == 1 and one.mean_diff == pytest.approx(1.5)
    assert math.isnan(one.p_a_better)
    flat = diebold_mariano([1.0, 1.0], [0.0, 0.0], [1, 2])
    assert flat.mean_diff == 1.0 and math.isnan(flat.t_stat)


def test_brier_decomposition_by_hand():
    prob = [0.15, 0.15, 0.85, 0.85]
    outcome = [0, 1, 1, 1]
    result = brier_decomposition(prob, outcome)
    assert result.brier == pytest.approx(0.79 / 4)
    assert result.reliability == pytest.approx((2 * 0.35**2 + 2 * 0.15**2) / 4)
    assert result.resolution == pytest.approx(0.0625)
    assert result.uncertainty == pytest.approx(0.1875)
    # Predictions constant within bins: the decomposition is exact.
    assert result.brier == pytest.approx(
        result.reliability - result.resolution + result.uncertainty
    )
    assert (result.n, result.n_bins) == (4, 10)
    with pytest.raises(ValueError, match="0 or 1"):
        brier_decomposition([0.5], [2])


def test_reliability_table_by_hand():
    table = reliability_table([0.15, 0.15, 0.85, 0.85, 1.0], [0, 1, 1, 1, 1], bins=10)
    assert table["bin"].tolist() == [1, 8, 9]
    first, eighth, last = table.to_dict("records")
    assert (first["lo"], first["hi"], first["n"]) == (pytest.approx(0.1), pytest.approx(0.2), 2)
    assert first["mean_pred"] == pytest.approx(0.15) and first["observed_rate"] == 0.5
    assert eighth["observed_rate"] == 1.0
    assert last["n"] == 1 and last["hi"] == 1.0  # 1.0 falls in the last (closed) bin


def test_band_table_uses_the_predicted_value():
    table = band_table([0.2, 0.4, 2.5, 7.0], [1, 0, 2, 2], edges=[-np.inf, 1, 3, np.inf])
    assert table["n"].tolist() == [2, 1, 1]
    low = table.iloc[0]
    assert low["mean_pred"] == pytest.approx(0.3) and low["mean_actual"] == pytest.approx(0.5)
    assert low["mse"] == pytest.approx((0.8**2 + 0.4**2) / 2)
    assert table.iloc[2]["mse"] == pytest.approx(25.0)
    with pytest.raises(ValueError, match="ascending"):
        band_table([1.0], [1.0], edges=[1, 0])


def test_minutes_class():
    assert minutes_class([0, 1, 59, 60, 90]).tolist() == [0, 1, 1, 2, 2]


def test_ordinal_log_loss_and_rps_by_hand():
    probs = [[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]]
    classes = [2, 0]
    assert ordinal_log_loss(probs, classes) == pytest.approx(-(math.log(0.5) + math.log(0.6)) / 2)
    # Row 1: F = (0.2, 0.5), O = (0, 0) -> 0.29 / 2; row 2: F = (0.6, 0.9), O = (1, 1) -> 0.17 / 2.
    assert ranked_probability_score(probs, classes) == pytest.approx((0.145 + 0.085) / 2)
    assert ranked_probability_score([[0, 0, 1]], [2]) == 0.0
    assert ranked_probability_score([[1, 0, 0]], [2]) == 1.0
    with pytest.raises(ValueError, match="sum to 1"):
        ordinal_log_loss([[0.5, 0.2, 0.2]], [0])
    with pytest.raises(ValueError, match="class labels"):
        ordinal_log_loss([[0.5, 0.3, 0.2]], [3])


def test_poisson_log_likelihood_by_hand():
    expected = (-1.0 + (3 * math.log(2) - 2 - math.log(6))) / 2
    assert poisson_log_likelihood([1.0, 2.0], [0, 3]) == pytest.approx(expected)
    stats = pytest.importorskip("scipy.stats")
    rates, counts = np.array([0.3, 1.7, 2.2, 0.05]), np.array([0, 2, 5, 1])
    assert poisson_log_likelihood(rates, counts) == pytest.approx(
        stats.poisson.logpmf(counts, rates).mean()
    )
    with pytest.raises(ValueError, match="integers"):
        poisson_log_likelihood([1.0], [0.5])


def _squad():
    keys = [1, 2, 11, 12, 13, 14, 15, 21, 22, 23, 24, 25, 31, 32, 33]
    types = [1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4]
    return pd.DataFrame({"player_key": keys, "element_type": types}, dtype="int64")


def test_lineup_regret_by_hand():
    xp = {1: 5, 2: 1, 11: 5, 12: 4, 13: 3, 14: 2, 15: 1, 21: 9, 22: 8, 23: 7, 24: 6, 25: 1}
    xp |= {31: 10, 32: 4, 33: 1}
    points = {1: 2, 2: 0, 11: 1, 12: 6, 13: 2, 14: 0, 15: 8, 21: 3, 22: 2, 23: 5, 24: 1}
    points |= {25: 12, 31: 2, 32: 9, 33: 0}
    outcomes = {k: (v, 0 if k in (2, 14, 33) else 90) for k, v in points.items()}
    result = lineup_regret(_squad(), xp, outcomes, backtest_rules(2023))
    # Model XI (4-4-2): GK 1; DEF 11-14; MID 21-24; FWD 31, 32; captain 31. DEF 14 didn't
    # play: bench DEF 15 comes on. 2+1+6+2+8+3+2+5+1+2+9 = 41, + captain 31's 2.
    assert result.points == 43
    assert (result.captain_points, result.best_captain_points, result.captain_regret) == (2, 9, 7)
    # Hindsight: GK 1; DEF 15, 12, 13, 11; MID 25, 23, 21, 22; FWD 32, 31 = 52, captain 25.
    assert result.best_points == 64 and result.xi_regret == 21


def test_lineup_regret_is_zero_when_xp_is_the_outcome():
    points = {k: (k % 7) for k in _squad()["player_key"]}
    outcomes = {k: (v, 90) for k, v in points.items()}
    result = lineup_regret(_squad(), points, outcomes, backtest_rules(2023))
    assert result.xi_regret == 0 and result.captain_regret == 0


def test_component_metrics_by_kind():
    frame = pd.DataFrame(
        {
            "p_cs": [0.15, 0.15, 0.85, 0.85, 0.5],
            "clean_sheets": [0, 1, 1, 1, 2],
            "n_fixtures": [1, 1, 1, 1, 2],  # the double is left out of probability metrics
            "e_goals": [1.0, 2.0, 0.5, 0.5, 1.0],
            "goals_scored": [0, 3, 0, 1, 2],
            "p_min_0": [0.2, 0.6, 0.1, 0.1, 0.1],
            "p_min_1_59": [0.3, 0.3, 0.2, 0.2, 0.2],
            "p_min_60": [0.5, 0.1, 0.7, 0.7, 0.7],
            "minutes": [90, 0, np.nan, 90, 180],  # a null target is left out
        }
    )
    by_name = {c.name: c for c in COMPONENTS}
    cs = component_metrics(frame, by_name["p_cs"])
    assert cs["n"] == 4 and cs["brier"] == pytest.approx(0.79 / 4)
    assert [row["n"] for row in cs["reliability_table"]] == [2, 2]
    goals = component_metrics(frame, by_name["e_goals"])
    assert goals["n"] == 5
    assert goals["log_likelihood"] == pytest.approx(
        poisson_log_likelihood(frame["e_goals"], frame["goals_scored"])
    )
    minutes = component_metrics(frame, by_name["minutes"])
    assert minutes["n"] == 3
    probs = frame.loc[[0, 1, 3], ["p_min_0", "p_min_1_59", "p_min_60"]].to_numpy()
    assert minutes["rps"] == pytest.approx(ranked_probability_score(probs, [2, 0, 2]))
    value = component_metrics(
        frame.assign(e_minutes=[80.0, 10.0, 0.0, 70.0, 150.0]), by_name["e_minutes"]
    )
    assert value["n"] == 4 and value["mae"] == pytest.approx((10 + 10 + 20 + 30) / 4)
    with pytest.raises(ValueError, match="kind"):
        component_metrics(frame, Component("x", "nope", ("e_goals",), "goals_scored"))
