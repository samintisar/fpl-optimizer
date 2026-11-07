import importlib.util
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("price_gate_c", REPO / "dev" / "price_gate_c.py")
gate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate
spec.loader.exec_module(gate)


def _ok(**kw):
    args = dict(log_families=["other"], dirty=False, head="abc", tag="abc", out_nonempty=False)
    args.update(kw)
    return gate.refusal(**args)


def test_refusals():
    assert _ok() is None
    assert "price check" in _ok(log_families=["x", "price check"])
    assert _ok(log_families=["price check verdict"]) is None
    assert "uncommitted" in _ok(dirty=True)
    assert "price-check" in _ok(head="def")
    assert "price-check" in _ok(tag=None)
    assert "not empty" in _ok(out_nonempty=True)


def _res(pm, se, ev=0.0):
    return {"points": {"mean": pm, "se": se}, "end_value": {"mean": ev, "se": 1.0}}


def test_tune_choice_deflates_noisy_arm():
    # raw mean favours "1", but its SE is large; deflation picks the steady "0.5"
    res = {"0": _res(0.0, 1.0), "0.5": _res(4.0, 1.0), "1": _res(6.0, 5.0)}
    out = gate.tune_choice(res)
    assert out["chosen"] == "0.5"
    assert abs(out["arms"]["0"]["deflated"] - (0.0 - (2 * 1.0986122887) ** 0.5)) < 1e-6


def test_tune_choice_tie_broken_by_end_value():
    res = {"0": _res(1.0, 1.0, ev=5.0), "0.25": _res(1.0, 1.0, ev=9.0)}
    assert gate.tune_choice(res)["chosen"] == "0.25"
    res = {"0": _res(1.0, 1.0, ev=5.0), "0.25": _res(1.0 + 1e-12, 1.0, ev=1.0)}
    assert gate.tune_choice(res)["chosen"] == "0"


def _gws(policy_vals):
    rows = []
    for policy, ends in policy_vals.items():
        for i, end in enumerate(ends):
            for gw in (1, 2):
                rows.append(
                    {
                        "policy": policy,
                        "season": 2023,
                        "start_id": f"s{i}",
                        "gw_index": gw,
                        "net_points": 50,
                        "market_value": 1000 if gw == 1 else end,
                    }
                )
    return pd.DataFrame(rows)


def test_team_value_test():
    df = _gws({"A:forecast": [1030, 1025, 1040, 1020], "B": [1000, 1000, 1000, 1000]})
    mean, p, n = gate.team_value_test(df, "A:forecast", "B")
    assert mean == 28.75 and n == 4 and p < 0.01
    df = _gws({"A:forecast": [990, 1010, 1000, 1000], "B": [1000, 1000, 1000, 1000]})
    mean, p, n = gate.team_value_test(df, "A:forecast", "B")
    assert mean == 0.0 and 0.4 < p < 0.6


def test_verdict_boundaries():
    assert gate.verdict(0.05, -5.0)
    assert not gate.verdict(0.10, -5.0)
    assert not gate.verdict(0.05, -10.0)
    assert gate.verdict(0.0999, -9.99)


def test_season_conversion_and_verdict():
    assert abs(gate.season_bound(-0.25) - (-9.5)) < 1e-12
    assert gate.SEASON_GWS == 38
    # per-GW -0.25 -> -9.5 season points passes; -0.27 -> -10.26 fails
    assert gate.verdict(0.05, gate.season_bound(-0.25))
    assert not gate.verdict(0.05, gate.season_bound(-0.27))


def test_lower_bound_extractor():
    metrics = {
        "comparisons": [
            {"method": "full_run", "split": "develop", "metric": "realized", "ci_low": 9.0},
            {"method": "full_run", "split": "all", "metric": "xg", "ci_low": 8.0},
            {"method": "per_decision", "split": "all", "metric": "realized", "ci_low": 7.0},
            {"method": "full_run", "split": "all", "metric": "realized", "ci_low": -3.5},
        ]
    }
    assert gate.points_lower_bound(metrics) == -3.5
    assert "method=full_run" in gate.LOWER_BOUND_PATH
    try:
        gate.points_lower_bound({"comparisons": []})
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")
