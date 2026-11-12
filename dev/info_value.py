"""Value-of-information report.

Spec: docs/superpowers/specs/2026-11-09-value-of-information-design.md. Each oracle arm
(v1 with one input's truth) against the live planner, paired by (season, start).
Headline: the season net points gained on the full-season starts (start GW 1 or 2),
averaged per season, with an 80% t interval over the season means; the decision uses the
interval's lower end (>= 60 source, 20-60 noted, < 20 closed, fixed before any run).

    uv run python dev/info_value.py --base results/voi/base --arm M0=results/voi/m0 \
        --arm A0=results/voi/a0 --out results/voi
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from scipy import stats

SOURCE = 60.0  # points a season (spec, fixed before the runs)
NOTED = 20.0
FULL_SEASON_MAX_START = 2  # template starts before 2021/22 begin at GW2


def units(gws: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, start_id): season totals and per-season counts."""
    g = gws.sort_values("gw_index").groupby(["season", "start_id"])
    return pd.DataFrame(
        {
            "points": g["net_points"].sum(),
            "start_gw": g["gw_index"].first(),
            "transfers": g["n_transfers"].sum(),
            "hit_points": g["hit_points"].sum(),
            "xi_regret": g["xi_regret"].mean(),
            "captain_regret": g["captain_regret"].mean(),
        }
    )


def interval(season_means: pd.Series) -> dict:
    """Mean and 80% t interval over season means (df = seasons - 1)."""
    n = int(season_means.size)
    mean = float(season_means.mean()) if n else float("nan")
    if n < 2:
        return {"n_seasons": n, "mean": mean, "se": float("nan"), "ci80": [float("nan")] * 2}
    se = float(season_means.std(ddof=1) / math.sqrt(n))
    half = float(stats.t.ppf(0.9, n - 1) * se)
    return {"n_seasons": n, "mean": mean, "se": se, "ci80": [mean - half, mean + half]}


def decision(lower: float) -> str:
    if math.isnan(lower):
        return "undetermined"
    if lower >= SOURCE:
        return "source"
    if lower >= NOTED:
        return "noted"
    return "closed"


def compare(base: pd.DataFrame, arm: pd.DataFrame) -> dict:
    """The arm against the base on the units both runs have."""
    joined = base.join(arm, lsuffix="_b", rsuffix="_a", how="inner")
    dropped = sorted(set(base.index) ^ set(arm.index))
    joined["diff"] = joined["points_a"] - joined["points_b"]
    full = joined[joined["start_gw_b"] <= FULL_SEASON_MAX_START]
    late = joined[joined["start_gw_b"] > FULL_SEASON_MAX_START]
    head = interval(full.groupby(level="season")["diff"].mean())
    out = {
        "n_units": int(len(joined)),
        "units_in_one_run_only": [list(map(str, u)) for u in dropped],
        "full_season": head,
        "late_starts": interval(late.groupby(level="season")["diff"].mean()),
        "decision": decision(head["ci80"][0]),
    }
    for col in ("transfers", "hit_points", "xi_regret", "captain_regret"):
        out[col] = {
            "arm": float(full[f"{col}_a"].mean()) if len(full) else float("nan"),
            "base": float(full[f"{col}_b"].mean()) if len(full) else float("nan"),
        }
    return out


def table(results: dict) -> str:
    header = (
        f"{'arm':<5} {'seasons':>7} {'gain':>8} {'80% interval':>18} {'late starts':>12} "
        f"{'XI regret':>14} {'decision':>13}"
    )
    lines = [header]
    for name, r in results.items():
        full = r["full_season"]
        low, high = full["ci80"]
        span = f"{low:+.1f} to {high:+.1f}"
        regret = f"{r['xi_regret']['arm']:.2f} / {r['xi_regret']['base']:.2f}"
        lines.append(
            f"{name:<5} {full['n_seasons']:>7} {full['mean']:>+8.1f} {span:>18} "
            f"{r['late_starts']['mean']:>+12.1f} {regret:>14} {r['decision']:>13}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=RUN")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    def load(p: Path) -> pd.DataFrame:
        return units(pd.read_parquet(p / "gws.parquet"))

    base = load(args.base)
    results = {}
    for item in args.arm:
        name, _, run = item.partition("=")
        if not run:
            ap.error(f"--arm {item!r}: want NAME=RUN")
        results[name] = compare(base, load(Path(run)))
    report = {"thresholds": {"source": SOURCE, "noted": NOTED}, "arms": results}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "info_value.json").write_text(json.dumps(report, indent=2))
    print(table(results))


if __name__ == "__main__":
    main()
