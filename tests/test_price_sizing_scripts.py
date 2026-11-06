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
