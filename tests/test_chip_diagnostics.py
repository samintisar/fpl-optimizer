"""Tests for fplopt.backtest.chip_diagnostics (spec 2026-10-29-chip-diagnostics-design)."""

import numpy as np
import pandas as pd
import pytest

from fplopt.backtest import chip_diagnostics as cd
from fplopt.backtest.chip_values import estimate_terminal_values
from fplopt.backtest.rules import backtest_rules
from fplopt.seasons import DEVELOP_SEASONS


def gws(season=2018, start="template@1", n=10, chips=None, points=None):
    """GW rows of one run: gw_index 1..n, `chips` = {gw_index: chip}, `points` = {gw_index: net}."""
    chips, points = chips or {}, points or {}
    return pd.DataFrame(
        {
            "season": season,
            "start_id": start,
            "gw": range(1, n + 1),
            "gw_index": range(1, n + 1),
            "chip": [chips.get(i) for i in range(1, n + 1)],
            "net_points": [points.get(i, 50) for i in range(1, n + 1)],
            "xg_net_points": [float(points.get(i, 50)) for i in range(1, n + 1)],
        }
    )


def test_a_wildcard_is_attributed_its_gw_and_the_next_five():
    run = gws(chips={2: "wildcard", 4: "3xc"})
    got = list(cd.attribute_chips(run))
    assert got == [
        "none", "wildcard", "wildcard", "3xc", "wildcard", "wildcard", "wildcard",
        "none", "none", "none",
    ]  # fmt: skip


def test_attribution_is_per_run():
    a = gws(start="template@1", n=3, chips={3: "wildcard"})
    b = gws(start="random0@1", n=3)
    got = cd.attribute_chips(pd.concat([a, b], ignore_index=True))
    assert list(got) == ["none", "none", "wildcard", "none", "none", "none"]


def test_gap_by_chip_contributions_sum_to_the_mean_gap():
    a = gws(chips={3: "3xc"}, points={3: 70, 5: 60})
    b = gws(points={5: 40})
    table = cd.gap_by_chip(a, b)
    assert table["contribution_per_gw"].sum() == pytest.approx((20 + 20) / 10)
    row = table.set_index(["chip_a", "chip_b"]).loc[("3xc", "none")]
    assert row["n"] == 1 and row["mean_diff"] == 20


def test_gap_by_cell_uses_8_gw_cells():
    a = gws(n=16, points={1: 58, 9: 42})
    b = gws(n=16)
    table = cd.gap_by_cell(a, b).set_index("cell")
    assert table.loc[1, "mean_diff"] == pytest.approx(8 / 8)
    assert table.loc[2, "mean_diff"] == pytest.approx(-8 / 8)


def test_paired_refuses_runs_that_dont_match():
    with pytest.raises(ValueError, match="rows"):
        cd.paired(gws(n=10), gws(n=9))


def schedule_rows(rows):
    """rows = [(season, gw, home, away)]."""
    return pd.DataFrame(
        [
            {"fixture_key": k, "season": s, "gw": g, "home_team_key": h, "away_team_key": a}
            for k, (s, g, h, a) in enumerate(rows)
        ]
    )


def test_gw_kind_counts_the_captains_fixtures_and_the_gws_doubles_and_blanks():
    # clubs 1-4; GW1 normal, GW2 clubs 1 and 2 play each other twice, clubs 3-4 blank
    schedule = schedule_rows([(2018, 1, 1, 2), (2018, 1, 3, 4), (2018, 2, 1, 2), (2018, 2, 1, 2)])
    player_gw = pd.DataFrame(
        {"player_key": [7, 7, 8], "season": 2018, "gw": [1, 2, 2], "team_key": [1, 1, 3]}
    )
    rows = pd.DataFrame(
        {
            "season": 2018,
            "start_id": "t",
            "gw": [1, 2, 2],
            "gw_index": [1, 2, 2],
            "captain": [7, 7, 8],
        }
    )
    got = cd.gw_kind(schedule, player_gw, rows)
    assert list(got["captain_fixtures"]) == [1, 2, 0]
    assert list(got["captain_kind"]) == ["single", "double", "blank"]
    assert list(got["n_double_clubs"]) == [0, 2, 2]
    assert list(got["n_blank_clubs"]) == [0, 2, 2]
    assert list(got["gw_type"]) == ["normal", "double+blank", "double+blank"]


def test_gain_distribution_quantiles_per_chip_and_group():
    rows = pd.DataFrame(
        {
            "start_id": ["template@1"] * 4 + ["random0@1"] * 4,
            "gain_tc": [1.0, 2.0, 3.0, 4.0, 10.0, 10.0, 10.0, 10.0],
            "gain_bb": np.nan,
            "gain_fh": np.nan,
            "gain_wc": np.nan,
        }
    )
    rows["start_type"] = cd.start_type(rows["start_id"])
    table = cd.gain_distribution(rows, ["start_type"])
    tc = table[table["chip"] == "3xc"].set_index("start_type")
    assert tc.loc["template", "median"] == 2.5 and tc.loc["random", "p90"] == 10
    assert tc.loc["template", "n"] == 4
    assert set(table["chip"]) == {"3xc"}  # chips without gains are left out


def test_forecast_slope_recovers_a_planted_slope():
    rng = np.random.default_rng(0)
    predicted = pd.Series(rng.uniform(2, 12, 2000))
    realized = 1.0 + 0.5 * predicted + rng.normal(0, 0.5, 2000)
    got = cd.forecast_slope(predicted, realized)
    assert got["slope"] == pytest.approx(0.5, abs=0.02)
    assert got["intercept"] == pytest.approx(1.0, abs=0.15)
    assert got["se"] < 0.01 and got["n"] == 2000
    assert got["bias"] == pytest.approx(float((realized - predicted).mean()))


def test_forecast_slope_skips_missing_pairs_and_needs_three():
    got = cd.forecast_slope(pd.Series([1.0, 2.0, np.nan, 4.0]), pd.Series([1.0, 2.0, 3.0, np.nan]))
    assert got["n"] == 2 and np.isnan(got["slope"])


def season_rows(seed, start="template@1", season=2018):
    rng = np.random.default_rng(seed)
    n = 38
    return pd.DataFrame(
        {
            "season": season,
            "start_id": start,
            "policy": "s0",
            "gw_index": np.arange(1, n + 1),
            "gain_tc": rng.uniform(3, 12, n),
            "gain_bb": rng.uniform(0, 8, n),
            "gain_fh": rng.uniform(-2, 10, n),
            "gain_wc": rng.uniform(-5, 20, n),
            "captain_points": rng.integers(0, 20, n),
            "bench_points": rng.integers(0, 15, n),
        }
    )


def test_holding_value_ex_ante_matches_the_terminal_value_estimate():
    # estimate_terminal_values pools a chip's windows per L: compare the n-weighted mean
    rows = pd.concat([season_rows(1), season_rows(2, start="random0@1")], ignore_index=True)
    table = {c: (1.0,) * 19 for c in ("3xc", "bboost", "freehit", "wildcard")}
    got = cd.holding_value(rows, {"3xc": "captain_points", "bboost": "bench_points"}, table)
    pooled = (
        got.assign(w=got["ex_ante"] * got["n"])
        .groupby(["chip", "left"])[["w", "n"]]
        .sum()
        .pipe(lambda t: t["w"] / t["n"])
    )
    for chip, values in estimate_terminal_values(rows).items():
        mine = pooled.loc[chip].sort_index().to_numpy()[: len(values)]
        np.testing.assert_allclose(np.clip(mine, 0, None), values)


def test_holding_value_realized_is_taken_at_the_ex_ante_best_gw():
    rows = season_rows(3)
    rules = backtest_rules(max(DEVELOP_SEASONS))
    stop = max(w.stop for w in rules.chips if w.name == "3xc" and w.start == 1)
    # L = 1: the window's last GW only
    last = rows.loc[rows["gw_index"] == stop].iloc[0]
    # L = 2: the better of the last two by gain_tc
    two = rows[rows["gw_index"].isin([stop - 1, stop])].sort_values("gain_tc").iloc[-1]
    got = cd.holding_value(rows, {"3xc": "captain_points"}, {"3xc": (2.0,) * 19})
    tc = got[(got["chip"] == "3xc") & (got["window_start"] == 1)].set_index("left")
    assert tc.loc[1, "realized"] == last["captain_points"]
    assert tc.loc[2, "realized"] == two["captain_points"]
    assert tc.loc[1, "table"] == 2.0
    assert tc.loc[1, "realized_ratio"] == pytest.approx(last["captain_points"] / 2.0)
    fh = got[got["chip"] == "freehit"]
    assert fh["realized"].isna().all()  # no realized column given


def test_captain_congestion_flags_a_european_week_or_short_rest():
    features = pd.DataFrame(
        {
            "player_key": [7, 7, 7, 8],
            "season": 2018,
            "gw": [1, 2, 2, 2],
            "europe_week": [0.0, 0.0, 1.0, 0.0],
            "rest_days_all": [7.0, 6.0, 3.0, 2.5],
        }
    )
    rows = pd.DataFrame({"season": 2018, "gw": [1, 2, 2, 3], "captain": [7, 7, 8, 7]})
    got = cd.captain_congestion(rows, features)
    assert list(got) == [False, True, True, False]  # GW3: no feature row -> not flagged


def test_ratio_split_and_weighted_share():
    rows = pd.DataFrame(
        {
            "flag": [True, True, False, False],
            "xp": [5.0, 5.0, 5.0, 5.0],
            "real": [2.0, 4.0, 6.0, 6.0],
        }
    )
    got = cd.ratio_split(rows, "flag", "xp", "real").set_index("flag")
    assert got.loc[True, "ratio"] == pytest.approx(0.6) and got.loc[False, "ratio"] == 1.2
    assert got.loc[True, "n"] == 2
    assert cd.weighted_share(pd.Series([True, False, False]), pd.Series([2.0, 1.0, 1.0])) == 0.5


def test_attribution_works_positionally_with_duplicate_index_labels():
    a = gws(start="a", n=4, chips={1: "3xc"})
    b = gws(start="b", n=4)
    got = cd.attribute_chips(pd.concat([a, b]))  # no ignore_index: labels repeat
    assert list(got) == ["3xc", "none", "none", "none", "none", "none", "none", "none"]


def test_paired_refuses_duplicate_keys_that_mask_a_missing_one():
    a, b = gws(n=3), gws(n=3)
    a["gw_index"] = [1, 1, 3]
    with pytest.raises(ValueError, match="runs don't match"):
        cd.paired(a, b)


def test_holding_value_refuses_nan_realized_and_non_develop_seasons():
    rows = season_rows(1)
    rows.loc[5, "captain_points"] = np.nan
    with pytest.raises(ValueError, match="captain_points"):
        cd.holding_value(rows, {"3xc": "captain_points"}, {})
    with pytest.raises(ValueError, match="develop seasons only"):
        cd.holding_value(season_rows(1, season=2024), {}, {})


def test_gain_distribution_refuses_rows_without_any_gain():
    rows = pd.DataFrame({"g": ["x"], "gain_tc": np.nan, "gain_bb": np.nan,
                         "gain_fh": np.nan, "gain_wc": np.nan})  # fmt: skip
    with pytest.raises(ValueError, match="no chip gains"):
        cd.gain_distribution(rows, ["g"])
