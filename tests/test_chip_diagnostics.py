"""Tests for fplopt.backtest.chip_diagnostics (spec 2026-10-29-chip-diagnostics-design)."""

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
