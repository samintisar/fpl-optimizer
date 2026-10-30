"""Chip policy 2 comparisons (spec docs/superpowers/specs/2026-10-30-chip-policy-2-design.md):
each arm vs H by the full-run window t-test on 8-GW cells, deflated for N = 5 eligible
variants (V1-V4, V6; V5 is information only: its doubles are known in hindsight); the best
of V1-V3.

    uv run python dev/chip_policy_2.py --h results/cp2-h \
        --arm P=results/cp2-p --arm V1=results/cp2-v1 ... [--out results/chip-policy-2]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from fplopt.backtest import chip_diagnostics as cd
from fplopt.backtest.evaluate import full_run_window_test

N_VARIANTS = 5
INFO_ONLY = ("V5",)  # reported, never the winner (hindsight doubles)


def load(run: Path) -> pd.DataFrame:
    return pd.read_parquet(run / "gws.parquet")


def compare(a: pd.DataFrame, b: pd.DataFrame) -> dict[str, float]:
    m = cd.paired(a, b)
    real, xg = full_run_window_test(m, "diff"), full_run_window_test(m, "diff_xg")
    return {
        "mean": real.mean,
        "se": real.se,
        "p_one_sided": real.p_one_sided,
        "n_cells": real.n_cells,
        # NaN se (fewer than 2 cells) makes this NaN, and NaN > 0 is False: the arm can't pass.
        "deflated": real.mean - real.se * math.sqrt(2 * math.log(N_VARIANTS)),
        "xg_mean": xg.mean,
    }


def passing_arms(vs_h: dict[str, dict[str, float]]) -> list[str]:
    """Arms V1-V6 (not information-only ones) with deflated mean > 0, one-sided p < 0.10 and a
    positive xG-scored mean, best deflated mean first. NaN comparisons are False: never pass."""
    ok = {
        name: r
        for name, r in vs_h.items()
        if name.startswith("V")
        and name not in INFO_ONLY
        and r["deflated"] > 0
        and r["p_one_sided"] < 0.10
        and r["xg_mean"] > 0
    }
    return sorted(ok, key=lambda n: -ok[n]["deflated"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=Path, required=True)
    ap.add_argument("--arm", action="append", default=[], help="NAME=RUN_DIR")
    ap.add_argument("--out", type=Path, default=Path("results/chip-policy-2"))
    args = ap.parse_args()
    h = load(args.h)
    arms = {name: load(Path(path)) for name, path in (a.split("=", 1) for a in args.arm)}
    out: dict = {"vs H": {name: compare(g, h) for name, g in arms.items()}}
    first = [v for v in ("V1", "V2", "V3") if v in arms]
    if first:
        # Ties: the smaller shrink, i.e. the earlier of V1-V3.
        out["best_of_v1_v3"] = max(first, key=lambda v: (out["vs H"][v]["mean"], -first.index(v)))
    out["passing"] = passing_arms(out["vs H"])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for name, r in out["vs H"].items():
        print(
            f"{name} vs H: {r['mean']:+.3f}/GW (se {r['se']:.3f}, deflated {r['deflated']:+.3f}), "
            f"p {r['p_one_sided']:.4f}, xG {r['xg_mean']:+.3f}"
        )
    print(f"best of V1-V3: {out.get('best_of_v1_v3')}; passing: {out['passing'] or 'none'}")


if __name__ == "__main__":
    main()
