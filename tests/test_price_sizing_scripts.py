"""Unit tests of the price sizing scripts in dev/ (spec 2026-11-06-price-sizing-design.md)."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "dev" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gap = _load("price_value_gap")


def test_sample_ranks_is_seeded_and_in_range():
    a, b = gap.sample_ranks(0, 500), gap.sample_ranks(0, 500)
    assert a == b
    assert len(a["top10k"]) == 500 and len(set(a["top10k"])) == 500
    assert min(a["top10k"]) >= 1 and max(a["top10k"]) <= 10_000
    assert min(a["10k-100k"]) >= 10_001 and max(a["10k-100k"]) <= 100_000
    assert gap.sample_ranks(1, 500) != a


def test_page_and_slot_of_rank():
    assert gap.page_of(1) == 1 and gap.slot_of(1) == 0
    assert gap.page_of(50) == 1 and gap.slot_of(50) == 49
    assert gap.page_of(51) == 2 and gap.slot_of(51) == 0


def test_value_gap_at_last_common_gw():
    managers = pd.DataFrame(
        {
            "group": ["top10k"] * 4,
            "entry": [1, 1, 2, 2],
            "gw": [1, 2, 1, 2],
            "value": [1000, 1010, 1000, 1030],
        }
    )
    ours = pd.DataFrame({"gw": [1, 2, 3], "market_value": [1000, 1005, 1007]})
    out = gap.value_gap(managers, ours, n_gws=38)
    g = out["top10k"]
    assert g["gw"] == 2
    assert g["median"] == 1020 and g["ours"] == 1005 and g["gap"] == 15
    assert g["gap_season"] == pytest.approx(15 * 38 / 2)


timing = _load("price_timing")


def test_selling_price_rule():
    assert timing.sell_price(50, 53) == 51  # half the rise, rounded down
    assert timing.sell_price(50, 54) == 52
    assert timing.sell_price(50, 48) == 48  # a fall is taken in full
    assert timing.sell_price(50, 50) == 50


def test_timing_gain_sell_before_drop_buy_before_rise():
    # sold player: purchase 50, was 52 in the window, 50 at the deadline -> sell at 51 not 50
    # bought player: 60 in the window, 61 at the deadline -> 1 saved
    assert timing.timing_gain(50, [52, 51], 50, [60, 61], 61) == (1, 1)


def test_timing_gain_is_zero_without_changes():
    assert timing.timing_gain(50, [50, 50], 50, [60, 60], 60) == (0, 0)
    assert timing.timing_gain(50, [], 50, [], 60) == (0, 0)


def test_replay_tracks_purchase_prices():
    gws = pd.DataFrame(
        {
            "season": [2021, 2021, 2021],
            "start_id": ["template@1"] * 3,
            "gw": [1, 2, 3],
            "gw_index": [1, 2, 3],
            "deadline": pd.to_datetime(
                ["2021-08-13 17:30", "2021-08-21 10:00", "2021-08-28 10:00"], utc=True
            ),
            "transfers": ["[]", "[[1, 2]]", "[[2, 3]]"],
        }
    )
    deadline_price = {
        (2021, 1, 1): 50,
        (2021, 2, 1): 50,
        (2021, 2, 2): 60,
        (2021, 3, 2): 61,
        (2021, 3, 3): 70,
    }
    snaps = {1: [], 2: [(pd.Timestamp("2021-08-28 01:30", tz="UTC"), 62)], 3: []}
    rows = timing.replay(gws, deadline_price, snaps, window=pd.Timedelta(hours=24))
    # GW3: player 2 bought at 60, 62 on the night before the deadline, 61 at the deadline:
    # selling price 61 vs 60 -> 1 tenth; nothing on the buy side
    assert [(r["gw"], r["sell_gain"], r["buy_gain"]) for r in rows] == [(2, 0, 0), (3, 1, 0)]


sizing = _load("price_sizing")


def _gws(start_ids, points):
    rows = []
    for sid, pts in zip(start_ids, points, strict=True):
        for gw, p in enumerate(pts, start=1):
            rows.append({"season": 2020, "start_id": sid, "gw": gw, "net_points": p})
    return pd.DataFrame(rows)


def test_points_per_million_pairs_by_start():
    base = _gws(["template@1", "random0@1", "template@20"], [[50, 50], [40, 40], [30]])
    bank = _gws(
        ["template+bank10@1", "random0+bank10@1", "template+bank10@20"], [[52, 51], [41, 42], [30]]
    )
    out = sizing.points_per_million(base, bank, extra_bank=10)
    assert out["1"]["n"] == 2 and out["1"]["mean"] == pytest.approx((3 + 3) / 2)
    assert out["20"]["n"] == 1 and out["20"]["mean"] == 0


def test_decision_thresholds():
    d = sizing.decide(planning_points=19.9, timing_points=10.0)
    assert d == {"planning": False, "timing": True}
    assert sizing.decide(20.0, 9.9) == {"planning": True, "timing": False}


def test_deadline_prices_use_last_snapshot_before_deadline():
    snap = pd.DataFrame(
        {
            "season": [2021] * 4,
            "player_key": [1, 1, 1, 2],
            "snapshot_at": pd.to_datetime(
                ["2021-08-01", "2021-08-10", "2021-08-14", "2021-08-10"], utc=True
            ),
            "now_cost": [50, 51, 99, 60],
        }
    )
    deadlines = {(2021, 1): pd.Timestamp("2021-08-13 17:30", tz="UTC")}
    out = timing.deadline_prices(deadlines, snap, fallback={(2021, 1, 1): 1, (2021, 1, 3): 70})
    assert out == {(2021, 1, 1): 51, (2021, 1, 2): 60, (2021, 1, 3): 70}


def _unit_gws(start_id, first_gw, transfers, chips=None):
    n = len(transfers)
    return pd.DataFrame(
        {
            "season": [2021] * n,
            "start_id": [start_id] * n,
            "gw": range(first_gw, first_gw + n),
            "gw_index": range(first_gw, first_gw + n),
            "deadline": pd.date_range("2021-09-01", periods=n, freq="7D", tz="UTC"),
            "transfers": transfers,
            "chip": chips or [None] * n,
        }
    )


def test_summarise_means_only_full_season_units_and_counts_empty_units():
    gws = pd.concat(
        [
            _unit_gws("template@1", 1, ["[]", "[[1, 2]]"]),
            _unit_gws("random0@1", 1, ["[]", "[]"]),
            _unit_gws("template@20", 20, ["[]", "[[1, 2]]"]),
        ]
    )
    rows = [
        {"season": 2021, "start_id": "template@1", "gw": 2, "sell_gain": 4, "buy_gain": 2},
        {"season": 2021, "start_id": "template@20", "gw": 21, "sell_gain": 10, "buy_gain": 0},
    ]
    out = timing.summarise(rows, gws)
    assert out["per_unit"]["2021 random0@1"] == 0
    assert out["mean_per_season"] == pytest.approx(3.0)  # (6 + 0) / 2, the @20 unit excluded
    assert out["mean_gw20_38"] == 10.0


def test_replay_skips_first_row_freehit_and_missing_prices():
    # @20 unit: the first row's transfers are not counted; GW22 is a Free Hit (skipped); GW21
    # sells 1 with no GW20 price (skipped, counted); GW23 brings in 6 with no price (counted)
    gws = _unit_gws(
        "template@20",
        20,
        ["[[8, 9]]", "[[1, 3]]", "[[3, 5]]", "[[3, 6]]"],
        chips=[None, None, "freehit", None],
    )
    price = {
        (2021, 20, 8): 50,
        (2021, 21, 3): 60,
        (2021, 21, 1): 55,
        (2021, 22, 3): 60,
        (2021, 22, 5): 40,
        (2021, 23, 3): 61,
    }
    stats = {}
    rows = timing.replay(gws, price, {}, pd.Timedelta(hours=24), stats)
    assert [r["gw"] for r in rows] == [21, 23]
    assert stats["skipped"] == 2


def test_value_gap_fpl_lag_shifts_managers_forward():
    managers = pd.DataFrame(
        {"group": ["top10k"] * 2, "entry": [1, 1], "gw": [1, 2], "value": [1000, 1010]}
    )
    ours = pd.DataFrame({"gw": [1, 2, 3], "market_value": [1000, 1005, 1007]})
    g = gap.value_gap(managers, ours, n_gws=38, fpl_lag=1)["top10k"]
    # managers' event 2 (1010) is compared with our row 3 (1007)
    assert g["gw"] == 3 and g["median"] == 1010 and g["ours"] == 1007 and g["gap"] == 3
