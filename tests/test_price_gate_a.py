import importlib.util
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("price_gate_a", REPO / "dev" / "price_gate_a.py")
gate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate
spec.loader.exec_module(gate)


def _run(points, end_values):
    rows = []
    for (sid, pts), end in zip(points.items(), end_values, strict=True):
        for gw, p in enumerate(pts, start=1):
            rows.append(
                {
                    "season": 2020,
                    "start_id": sid,
                    "gw": gw,
                    "gw_index": gw,
                    "net_points": p,
                    "market_value": 1000 if gw < len(pts) else end,
                }
            )
    return pd.DataFrame(rows)


def test_unit_totals_and_end_value():
    df = _run({"template@1": [50, 60], "random0@1": [40, 40]}, [1010, 990])
    u = gate.units(df)
    assert u.loc[(2020, "template@1"), "points"] == 110
    assert u.loc[(2020, "random0@1"), "end_value"] == 990


def test_gate_a_pass_and_fail():
    base = _run({"template@1": [50, 50], "random0@1": [50, 50]}, [1000, 1000])
    o = _run({"template@1": [56, 56], "random0@1": [55, 55]}, [1025, 1022])
    r = gate.compare(gate.units(base), gate.units(o))
    assert r["n"] == 2 and r["points"]["mean"] == 11.0 and r["end_value"]["mean"] == 23.5
    assert gate.decide(r) is True
    o2 = _run({"template@1": [56, 56], "random0@1": [55, 55]}, [1010, 1010])
    assert gate.decide(gate.compare(gate.units(base), gate.units(o2))) is False
