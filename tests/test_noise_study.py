import importlib.util
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("noise_study", REPO / "dev" / "noise_study.py")
ns = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ns
_spec.loader.exec_module(ns)


def test_comparisons():
    c = ns.comparisons()
    assert list(c)[:2] == ["C1", "C2"] and len(c) == 12
    assert c["C1"]["a"] == ns.LIVE + ",prices=forecast,price_k=0.5" and c["C1"]["b"] == ns.LIVE
    assert c["C1"]["seasons"] == "2022" and c["C2"]["seasons"] == "2021,2022"
    assert "hit_margin=3" in c["C2"]["a"] and "hit_margin=2" not in c["C2"]["a"]
    assert c["P1"]["a"].endswith("xp_jitter=0.05,jitter_seed=1")
    assert c["P1"]["b"].endswith("xp_jitter=0.05,jitter_seed=2")
    assert c["P10"]["b"].endswith("jitter_seed=20")


def test_compare_cmd():
    comp = ns.comparisons()["C2"]
    cmd = ns.compare_cmd("bin", comp, 2, Path("o"), jobs=14, experiments=Path("e.csv"))
    assert cmd[:3] == ["bin", "backtest", "compare"]
    j = " ".join(cmd)
    assert "--k 2" in j and "--continuation own" in j and "--reference b" in j
    assert "--seasons 2021,2022" in j and f"--starts {ns.STARTS}" in j
    assert "--family" in cmd and cmd[cmd.index("--family") + 1] == ns.FAMILY
    full = ns.compare_cmd("bin", comp, None, Path("o"), jobs=14, experiments=Path("e.csv"))
    assert "--no-per-decision" in full and "--k" not in full


def _paired():
    # 1 season, 2 starts, 8 GWs: start s0 diff 1 each GW, s1 diff 3 -> per-GW mean 2
    rows = [
        {"season": 2022, "start_id": s, "gw_index": g, "diff": d, "diff_xg": d / 2}
        for s, d in (("s0", 1), ("s1", 3))
        for g in range(1, 9)
    ]
    return pd.DataFrame(rows)


def test_whole_season_scale_and_se():
    mean, se, n = ns.whole_season(_paired(), "diff")
    assert n == 2 and mean == pytest.approx(2 * 38) and se == 0.0
    df = _paired()
    df.loc[(df.start_id == "s0") & (df.gw_index <= 4), "diff"] = 5  # block 1 mean 4, block 2 mean 2
    mean, se, n = ns.whole_season(df, "diff")
    assert mean == pytest.approx(3 * 38)
    assert se == pytest.approx(math.sqrt(2) / math.sqrt(2) * 38)  # sd of (4, 2) = sqrt(2)


def _decisions(k):
    rows = []
    for g, d in ((1, 2), (1 + k, 4), (1 + 2 * k, 6)):
        for s in ("s0", "s1"):
            rows.append({"season": 2022, "start_id": s, "gw_index": g, "diff": d, "diff_xg": None})
    return pd.DataFrame(rows)


def test_per_decision_scale_and_se():
    mean, se, n = ns.per_decision(_decisions(2), 2, "diff")
    assert n == 3 and mean == pytest.approx(4 * 3)  # 3 windows in one season
    assert se == pytest.approx(2 / math.sqrt(3) * 3)
    mean, se, n = ns.per_decision(_decisions(2), 2, "diff_xg")  # all null -> no windows
    assert n == 0 and math.isnan(mean) and math.isnan(se)


def test_whole_season_averages_over_live_starts():
    rows = [{"season": 2022, "start_id": "s0", "gw_index": g, "diff": 1.0} for g in range(1, 9)]
    rows += [{"season": 2022, "start_id": "s1", "gw_index": g, "diff": 3.0} for g in range(5, 9)]
    df = pd.DataFrame(rows)
    df["diff_xg"] = df["diff"]
    mean, se, n = ns.whole_season(df, "diff")
    # block 1 (GW1-4): s0 only = 1; block 2 (GW5-8): mean(1, 3) = 2
    assert n == 2 and mean == pytest.approx(1.5 * 38)
    assert se == pytest.approx(math.sqrt(0.5) / math.sqrt(2) * 38)


def test_per_decision_ten_windows_a_season():
    rows = [
        {"season": 2022, "start_id": s, "gw_index": g, "diff": float(g), "diff_xg": None}
        for g in range(1, 38, 4)
        for s in ("s0", "s1")
    ]
    mean, se, n = ns.per_decision(pd.DataFrame(rows), 4, "diff")
    assert n == 10 and mean == pytest.approx(19.0 * 10)  # mean of 1,5,..,37 is 19; scale 10


def test_per_decision_pools_seasons_not_gw_index():
    rows = [
        {"season": y, "start_id": "s0", "gw_index": g, "diff": d, "diff_xg": None}
        for y, off in ((2021, 0), (2022, 10))
        for g, d in ((1, 2 + off), (3, 4 + off))
    ]
    mean, se, n = ns.per_decision(pd.DataFrame(rows), 2, "diff")
    assert n == 4  # not merged by gw_index
    assert mean == pytest.approx((2 + 4 + 12 + 14) / 4 * 2)  # 2 windows a season
    xs = [2, 4, 12, 14]
    sd = math.sqrt(sum((x - 8) ** 2 for x in xs) / 3)
    assert se == pytest.approx(sd / math.sqrt(4) * 2)


def test_season_clustered_se_uses_windows_per_season():
    rows = [
        {"season": y, "start_id": "s0", "gw_index": g, "diff": float(d), "diff_xg": None}
        for y, d in ((2021, 1), (2022, 3))
        for g in (1, 5, 9)
    ]
    se = ns.season_clustered_se(pd.DataFrame(rows), 4, "diff")
    assert se == pytest.approx(
        math.sqrt(2) / math.sqrt(2) * 3
    )  # per-season (3, 9): sd sqrt(18)/sqrt(2)


def test_z_sd_and_rms():
    assert ns.z_sd([(1.0, 1.0), (-1.0, 1.0)]) == pytest.approx(math.sqrt(2))
    assert ns.z_rms([(1.0, 1.0), (-1.0, 1.0)]) == pytest.approx(1.0)
    biased = [(1.5, 1.0)] * 4  # sd about the mean hides the bias, rms does not
    assert ns.z_sd(biased) == pytest.approx(0.0) and ns.z_rms(biased) == pytest.approx(1.5)
    assert math.isnan(ns.z_sd([(1.0, 0.0)])) and math.isnan(ns.z_rms([(1.0, 0.0)]))
    assert math.isnan(ns.z_rms([(1.0, 1.0)]))
    assert ns.z_count([(1.0, 1.0), (1.0, 0.0), (1.0, float("nan")), (2.0, 2.0)]) == 2


def _row(method, c1, c2, z, minutes):
    return {"method": method, "se_C1": c1, "se_C2": c2, "z_rms": z, "z_sd": z, "minutes": minutes}


def test_choose():
    rows = [
        _row("whole/realized", 9.0, 8.0, 1.0, 10),
        _row("k1/xg", 4.0, 4.4, 1.1, 60),
        _row("k2/xg", 4.2, 4.0, 1.2, 40),
        _row("k4/xg", 3.0, 3.0, 1.6, 30),
    ]
    assert ns.choose(rows) == "k2/xg"
    assert ns.choose(rows[:1]) is None
    assert ns.choose([_row("edge", 4.5, 4.5, 1.25, 5)]) == "edge"  # boundary passes
    assert ns.choose([_row("nan", 4.0, 4.0, float("nan"), 5)]) is None
    assert ns.choose([_row("nanse", float("nan"), 4.0, 1.0, 5)]) is None
    assert ns.choose([_row("zero", 0.0, 4.0, 1.0, 5)]) is None
    assert ns.choose([_row("zero2", 4.0, 0.0, 1.0, 5)]) is None
