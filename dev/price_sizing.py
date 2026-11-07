"""Price sizing: points per £1m (develop backtests with and without +bank10), the two upper
bounds and the decision, with the spec's fixed thresholds
(docs/superpowers/specs/2026-11-06-price-sizing-design.md §3).

    uv run python dev/price_sizing.py --base results/price-sizing-base \
        --bank results/price-sizing-bank10 --gap results/price-sizing-managers/gap.json \
        --timing results/price-sizing-timing/timing.json --out results/price-sizing
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from scipy import stats

PLANNING_MIN = 20.0  # points a season (spec §3, fixed before any number was seen)
TIMING_MIN = 10.0
GAP_GROUPS = ("top10k", "10k-100k")  # the upper bound uses the larger gap_season


def _totals(gws: pd.DataFrame) -> pd.Series:
    return gws.groupby(["season", "start_id"])["net_points"].sum()


def points_per_million(base: pd.DataFrame, bank: pd.DataFrame, extra_bank: int) -> dict:
    """Mean (bank − base) season total per start GW, scaled to £1m, with an 80% t CI."""
    b = _totals(base)
    m = _totals(bank)
    m.index = pd.MultiIndex.from_tuples(
        [(s, i.replace(f"+bank{extra_bank}", "")) for s, i in m.index], names=b.index.names
    )
    only = sorted(set(b.index) ^ set(m.index))
    if only:
        print(f"units present in only one run (dropped): {only}")
    diff = (m - b).dropna() * 10 / extra_bank
    out = {}
    for gw, d in diff.groupby(diff.index.get_level_values("start_id").str.split("@").str[1]):
        n = int(d.size)
        se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
        half = float(stats.t.ppf(0.9, n - 1) * se) if n > 1 else float("nan")
        out[gw] = {
            "n": n,
            "mean": float(d.mean()),
            "se": se,
            "ci80": [float(d.mean()) - half, float(d.mean()) + half],
        }
    return out


def decide(planning_points: float, timing_points: float) -> dict:
    return {"planning": planning_points >= PLANNING_MIN, "timing": timing_points >= TIMING_MIN}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for name in ("base", "bank", "out"):
        ap.add_argument(f"--{name}", type=Path, required=True)
    ap.add_argument("--gap", type=Path, required=True)
    ap.add_argument("--timing", type=Path, required=True)
    ap.add_argument("--extra-bank", type=int, default=10)
    args = ap.parse_args()
    rate = points_per_million(
        pd.read_parquet(args.base / "gws.parquet"),
        pd.read_parquet(args.bank / "gws.parquet"),
        args.extra_bank,
    )
    # GW1 starts before 2021/22 fall back to gw_index 2; pool them with GW1 as the season rate
    early = [v for k, v in rate.items() if k in ("1", "2")]
    if not early:
        raise SystemExit(
            f"no full-season start groups ('1' or '2') in the runs; found {sorted(rate)}"
        )
    n = sum(v["n"] for v in early)
    season_rate = sum(v["mean"] * v["n"] for v in early) / n
    gaps = json.loads(args.gap.read_text())
    gap_group = max(GAP_GROUPS, key=lambda g: gaps[g]["gap_season"])
    gap = gaps[gap_group]
    timing = json.loads(args.timing.read_text())
    planning_points = gap["gap_season"] / 10 * season_rate
    timing_points = timing["24h"]["mean_per_season"] / 10 * season_rate
    result = {
        "points_per_million": rate,
        "season_rate": season_rate,
        "gap": gap,
        "gap_group_used": gap_group,
        "gaps": {g: gaps[g] for g in GAP_GROUPS},
        "ci_note": "the CI treats (season, start) units as independent, although starts "
        "within a season share outcomes, so it is too narrow",
        "planning_points": planning_points,
        "timing_points_24h": timing_points,
        "timing_points_week": timing["week"]["mean_per_season"] / 10 * season_rate,
        "thresholds": {"planning": PLANNING_MIN, "timing": TIMING_MIN},
        "decision": decide(planning_points, timing_points),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "sizing.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
