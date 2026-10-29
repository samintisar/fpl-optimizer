"""Tests for fplopt.backtest.chip_diagnostics (spec 2026-10-29-chip-diagnostics-design)."""

import numpy as np
import pandas as pd
import pytest

from fplopt.backtest import chip_diagnostics as cd


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
