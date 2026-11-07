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
    assert n == 3 and mean == pytest.approx(4 * 19)  # 38 / k = 19 decisions a season
    assert se == pytest.approx(2 / math.sqrt(3) * 19)
    mean, se, n = ns.per_decision(_decisions(2), 2, "diff_xg")  # all null -> no windows
    assert n == 0 and math.isnan(mean) and math.isnan(se)


def test_z_sd():
    assert ns.z_sd([(1.0, 1.0), (-1.0, 1.0)]) == pytest.approx(math.sqrt(2))
    assert math.isnan(ns.z_sd([(1.0, 0.0)]))


def test_choose():
    rows = [
        {"method": "whole/realized", "se_C1": 9.0, "se_C2": 8.0, "z_sd": 1.0, "minutes": 10},
        {"method": "k1/xg", "se_C1": 4.0, "se_C2": 4.4, "z_sd": 1.1, "minutes": 60},
        {"method": "k2/xg", "se_C1": 4.2, "se_C2": 4.0, "z_sd": 1.2, "minutes": 40},
        {"method": "k4/xg", "se_C1": 3.0, "se_C2": 3.0, "z_sd": 1.6, "minutes": 30},
    ]
    assert ns.choose(rows) == "k2/xg"
    assert ns.choose(rows[:1]) is None
