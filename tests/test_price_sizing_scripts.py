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
