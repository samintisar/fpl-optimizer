import importlib.util
import math
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("info_value", REPO / "dev" / "info_value.py")
iv = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = iv
spec.loader.exec_module(iv)


def _gws(points_by_unit: dict) -> pd.DataFrame:
    """{(season, start_id, start_gw): points per GW} -> a 3-GW gws frame."""
    rows = []
    for (season, start_id, start_gw), pts in points_by_unit.items():
        for i in range(3):
            rows.append(
                dict(
                    season=season,
                    start_id=start_id,
                    gw_index=start_gw + i,
                    net_points=pts,
                    n_transfers=1,
                    hit_points=0,
                    xi_regret=5,
                    captain_regret=4,
                )
            )
    return pd.DataFrame(rows)


def _runs(gain_by_season: dict, late_gain: float = 1.0):
    base, arm = {}, {}
    for season, gain in gain_by_season.items():
        for start in ("template@1", "random:0@1"):
            base[(season, start, 1)] = 50
            arm[(season, start, 1)] = 50 + gain / 3
        base[(season, "template@20", 20)] = 50
        arm[(season, "template@20", 20)] = 50 + late_gain / 3
    return iv.units(_gws(base)), iv.units(_gws(arm))


def test_units_sum_points_and_keep_the_start_gw():
    u = iv.units(_gws({(2021, "template@1", 1): 50, (2021, "template@20", 20): 40}))
    assert u.loc[(2021, "template@1"), "points"] == 150
    assert u.loc[(2021, "template@20"), "start_gw"] == 20
    assert u.loc[(2021, "template@1"), "transfers"] == 3


def test_headline_is_the_full_season_starts_by_season():
    base, arm = _runs({2019: 90.0, 2020: 110.0, 2021: 100.0}, late_gain=7.0)
    r = iv.compare(base, arm)
    assert r["n_units"] == 9
    f = r["full_season"]
    assert f["n_seasons"] == 3
    assert math.isclose(f["mean"], 100.0, abs_tol=1e-9)
    # SE over the 3 season means (90, 110, 100): 10 / sqrt(3); t(0.9, 2) = 1.886
    assert math.isclose(f["se"], 10 / math.sqrt(3), rel_tol=1e-9)
    assert math.isclose(f["ci80"][0], 100 - 1.8856 * 10 / math.sqrt(3), abs_tol=0.01)
    assert math.isclose(r["late_starts"]["mean"], 7.0, abs_tol=1e-9)
    assert r["decision"] == "source"


def test_decision_boundaries():
    assert iv.decision(60.0) == "source"
    assert iv.decision(59.99) == "noted"
    assert iv.decision(20.0) == "noted"
    assert iv.decision(19.99) == "closed"
    assert iv.decision(-5.0) == "closed"
    assert iv.decision(float("nan")) == "undetermined"


def test_a_noisy_arm_is_judged_on_its_lower_end():
    base, arm = _runs({2019: 10.0, 2020: 190.0, 2021: 100.0})
    r = iv.compare(base, arm)
    assert math.isclose(r["full_season"]["mean"], 100.0, abs_tol=1e-9)
    assert r["full_season"]["ci80"][0] < 20
    assert r["decision"] == "closed"


def test_an_arm_with_fewer_seasons_uses_its_own_units():
    base, _ = _runs({2018: 0.0, 2019: 0.0, 2020: 0.0, 2021: 0.0})
    _, arm = _runs({2019: 80.0, 2020: 90.0, 2021: 85.0})
    r = iv.compare(base, arm)
    assert r["full_season"]["n_seasons"] == 3
    assert len(r["units_in_one_run_only"]) == 3
    assert iv.table({"R": r}).splitlines()[1].split()[1] == "3"


def test_one_season_has_no_interval():
    base, arm = _runs({2021: 100.0})
    r = iv.compare(base, arm)
    assert r["decision"] == "undetermined"
