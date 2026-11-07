"""Noise study (spec docs/superpowers/specs/2026-11-07-noise-study-design.md): which comparison
method sees 10 points a season.

    uv run python dev/noise_study.py run --out results/noise-study [--jobs 14] [--only C1,P3]
    uv run python dev/noise_study.py report --out results/noise-study

`run` launches `fplopt-rs backtest compare` for every comparison (C1 price, C2 hit margin,
P1-P10 placebo pairs): once without per-decision (the whole-season method's cost) and once per
window length k (1, 2, 4; each arm continues its own plan, states from B). Resumable: a run whose
summary.json exists is skipped. Wall times go to OUT/times.json. `report` puts the 8 methods
(whole season, per decision k = 1, 2, 4; realized and xG-scored points) on one scale, points per
season, and picks the cheapest with SE <= 4.5 on C1 and C2 and placebo z sd <= 1.25.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

LIVE = "optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2"
STARTS = (
    "template@1,random:3@1,template@10,random:3@10,template@20,random:3@20,template@29,random:3@29"
)
KS = (1, 2, 4)
SEASON_GWS = 38
BLOCK = 4  # GWs per block of the whole-season method (PLAN §7's window t-test)
TARGET_SE = 4.5  # points a season
Z_SD_MAX = 1.25
SIGMA = 0.05
N_PLACEBO = 10
FAMILY = "noise study"
SCORES = {"realized": "diff", "xg": "diff_xg"}


def comparisons() -> dict[str, dict]:
    out = {
        "C1": {"a": LIVE + ",prices=forecast,price_k=0.5", "b": LIVE, "seasons": "2022"},
        "C2": {
            "a": LIVE.replace("hit_margin=2", "hit_margin=3"),
            "b": LIVE,
            "seasons": "2021,2022",
        },
    }
    for s in range(1, N_PLACEBO + 1):
        jit = f",xp_jitter={SIGMA},jitter_seed="
        out[f"P{s}"] = {
            "a": f"{LIVE}{jit}{2 * s - 1}",
            "b": f"{LIVE}{jit}{2 * s}",
            "seasons": "2022",
        }
    return out


def compare_cmd(binary, comp: dict, k: int | None, out: Path, jobs: int, experiments: Path):
    """The compare command: k None = whole season only (--no-per-decision)."""
    cmd = [str(binary), "backtest", "compare", "--a", comp["a"], "--b", comp["b"]]
    cmd += ["--seasons", comp["seasons"], "--starts", STARTS, "--jobs", str(jobs)]
    cmd += ["--family", FAMILY, "--experiments", str(experiments), "--out", str(out)]
    if k is None:
        cmd.append("--no-per-decision")
    else:
        cmd += ["--continuation", "own", "--reference", "b", "--k", str(k), "--stride", str(k)]
    return cmd


def _stats(values: pd.Series, scale: float) -> tuple[float, float, int]:
    n = int(values.size)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = float(values.mean()) * scale
    se = float(values.std(ddof=1)) / math.sqrt(n) * scale if n > 1 else float("nan")
    return mean, se, n


def whole_season(paired: pd.DataFrame, col: str) -> tuple[float, float, int]:
    """Per-GW A - B averaged over the starts live in it, averaged per non-overlapping BLOCK-GW
    block within each season; mean and SE over blocks, scaled to a season (x SEASON_GWS)."""
    d = paired.dropna(subset=[col])
    per_gw = d.groupby(["season", "gw_index"])[col].mean().reset_index()
    per_gw["block"] = (per_gw["gw_index"] - 1) // BLOCK
    blocks = per_gw.groupby(["season", "block"])[col].mean()
    return _stats(blocks, SEASON_GWS)


def per_decision(dec: pd.DataFrame, k: int, col: str) -> tuple[float, float, int]:
    """Each decision window's A - B (summed over its k GWs) averaged over the starts live in it;
    windows of all seasons pooled; mean and SE over windows x (SEASON_GWS / k)."""
    d = dec.dropna(subset=[col])
    windows = d.groupby(["season", "gw_index"])[col].mean()
    return _stats(windows, SEASON_GWS / k)


def season_clustered_se(dec_or_paired: pd.DataFrame, k: int | None, col: str) -> float:
    """Diagnostic: SE over per-season means (C2 has 2 seasons, so this is rough)."""
    d = dec_or_paired.dropna(subset=[col])
    scale = SEASON_GWS if k is None else SEASON_GWS / k
    per = d.groupby(["season", "gw_index"])[col].mean().groupby("season").mean() * scale
    return float(per.std(ddof=1) / math.sqrt(per.size)) if per.size > 1 else float("nan")


def z_sd(pairs: list[tuple[float, float]]) -> float:
    """sd of z = mean / SE over placebo pairs (honest SEs: about 1)."""
    zs = [m / s for m, s in pairs if s and not math.isnan(s) and s > 0]
    if len(zs) < 2:
        return float("nan")
    mu = sum(zs) / len(zs)
    return math.sqrt(sum((z - mu) ** 2 for z in zs) / (len(zs) - 1))


def choose(rows: list[dict]) -> str | None:
    """The cheapest method (minutes) with se_C1, se_C2 <= TARGET_SE and z_sd <= Z_SD_MAX."""
    ok = [
        r
        for r in rows
        if r["se_C1"] <= TARGET_SE and r["se_C2"] <= TARGET_SE and r["z_sd"] <= Z_SD_MAX
    ]
    return min(ok, key=lambda r: r["minutes"])["method"] if ok else None
