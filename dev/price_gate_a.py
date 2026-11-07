"""Gate A of the price planner (spec docs/superpowers/specs/2026-11-06-price-planner-design.md):
O (oracle prices, price_k=1) vs L (the live planner), and F (price_k=1, fixed prices) vs L,
on develop 2016/17-2022/23, paired by (season, start). Pass: O - L mean end team value
>= +20 tenths and mean season net points >= +10.

    uv run python dev/price_gate_a.py --l results/price-a-l --o results/price-a-o \
        --f results/price-a-f --out results/price-gate-a
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from scipy import stats

MIN_VALUE = 20.0  # tenths of £m (spec, fixed before the runs)
MIN_POINTS = 10.0


def units(gws: pd.DataFrame) -> pd.DataFrame:
    g = gws.sort_values("gw_index").groupby(["season", "start_id"])
    return pd.DataFrame({"points": g["net_points"].sum(), "end_value": g["market_value"].last()})


def _stats(d: pd.Series) -> dict:
    n = int(d.size)
    se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    half = float(stats.t.ppf(0.9, n - 1) * se) if n > 1 else float("nan")
    return {
        "mean": float(d.mean()),
        "se": se,
        "ci80": [float(d.mean()) - half, float(d.mean()) + half],
    }


def compare(base: pd.DataFrame, arm: pd.DataFrame) -> dict:
    joined = base.join(arm, lsuffix="_b", rsuffix="_a", how="inner")
    missing = sorted(set(base.index) ^ set(arm.index))
    if missing:
        print(f"units in only one run, dropped: {missing}")
    return {
        "n": int(len(joined)),
        "points": _stats(joined["points_a"] - joined["points_b"]),
        "end_value": _stats(joined["end_value_a"] - joined["end_value_b"]),
    }


def decide(r: dict) -> bool:
    return r["end_value"]["mean"] >= MIN_VALUE and r["points"]["mean"] >= MIN_POINTS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for name in ("l", "o", "f", "out"):
        ap.add_argument(f"--{name}", type=Path, required=True)
    args = ap.parse_args()

    def load(p: Path) -> pd.DataFrame:
        return units(pd.read_parquet(p / "gws.parquet"))

    l_units, o_units, f_units = load(args.l), load(args.o), load(args.f)
    o_vs_l, f_vs_l = compare(l_units, o_units), compare(l_units, f_units)
    result = {
        "o_vs_l": o_vs_l,
        "f_vs_l": f_vs_l,
        "thresholds": {"end_value_tenths": MIN_VALUE, "points": MIN_POINTS},
        "pass": decide(o_vs_l),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "gate_a.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
