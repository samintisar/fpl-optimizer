"""Noise study (spec docs/superpowers/specs/2026-11-07-noise-study-design.md): which comparison
method sees 10 points a season.

    uv run python dev/noise_study.py run --out results/noise-study [--jobs 14] [--only C1,P3]
    uv run python dev/noise_study.py report --out results/noise-study

`run` launches `fplopt-rs backtest compare` for every comparison (C1 price, C2 hit margin,
P1-P10 placebo pairs): once without per-decision (the whole-season method's cost) and once per
window length k (1, 2, 4; each arm continues its own plan, states from B). Resumable: a run whose
summary.json exists is skipped. Wall times go to OUT/times.json. `report` puts the 8 methods
(whole season, per decision k = 1, 2, 4; realized and xG-scored points) on one scale, points per
season, and picks the cheapest with SE <= 4.5 on C1 and C2 and placebo z RMS <= 1.25.
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
Z_MAX = 1.25
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
    block within each season; mean and SE over blocks, scaled to a season (x SEASON_GWS).

    Caveat: starts begin at GW 1, 10, 20 and 29, so a block averages starts of different ages and
    late blocks mix young squads. The block means are heterogeneous, so the SE is conservative
    (too high), and the mean is not a true full-season effect for this method."""
    d = paired.dropna(subset=[col])
    per_gw = d.groupby(["season", "gw_index"])[col].mean().reset_index()
    per_gw["block"] = (per_gw["gw_index"] - 1) // BLOCK
    blocks = per_gw.groupby(["season", "block"])[col].mean()
    return _stats(blocks, SEASON_GWS)


def _windows(dec: pd.DataFrame, col: str) -> pd.Series:
    d = dec.dropna(subset=[col])
    return d.groupby(["season", "gw_index"])[col].mean()


def _windows_per_season(windows: pd.Series) -> float:
    n_seasons = windows.index.get_level_values("season").nunique()
    return windows.size / n_seasons if n_seasons else float("nan")


def per_decision(dec: pd.DataFrame, k: int, col: str) -> tuple[float, float, int]:
    """Each decision window's A - B (summed over its k GWs) averaged over the starts live in it;
    windows of all seasons pooled (keyed by season and gw_index); mean and SE over windows,
    scaled by the windows a season (n windows / n seasons; decisions sit at GW 1, 1 + k, ..., so
    k = 4 gives 10, not 38 / 4)."""
    windows = _windows(dec, col)
    return _stats(windows, _windows_per_season(windows))


def season_clustered_se(dec_or_paired: pd.DataFrame, k: int | None, col: str) -> float:
    """Diagnostic: SE over per-season means (C2 has 2 seasons, so this is rough). Per decision
    the scale is the windows a season, as in per_decision."""
    d = dec_or_paired.dropna(subset=[col])
    windows = d.groupby(["season", "gw_index"])[col].mean()
    scale = SEASON_GWS if k is None else _windows_per_season(windows)
    per = windows.groupby("season").mean() * scale
    return float(per.std(ddof=1) / math.sqrt(per.size)) if per.size > 1 else float("nan")


def _zs(pairs: list[tuple[float, float]]) -> list[float]:
    return [m / s for m, s in pairs if s and not math.isnan(s) and s > 0]


def z_count(pairs: list[tuple[float, float]]) -> int:
    """Usable placebo pairs (SE positive and not NaN)."""
    return len(_zs(pairs))


def z_sd(pairs: list[tuple[float, float]]) -> float:
    """Diagnostic: sd of z = mean / SE over placebo pairs, about their own mean (hides a bias)."""
    zs = _zs(pairs)
    if len(zs) < 2:
        return float("nan")
    mu = sum(zs) / len(zs)
    return math.sqrt(sum((z - mu) ** 2 for z in zs) / (len(zs) - 1))


def z_rms(pairs: list[tuple[float, float]]) -> float:
    """sqrt(mean z^2) about 0: honest, unbiased SEs give about 1; a biased method shows here."""
    zs = _zs(pairs)
    if len(zs) < 2:
        return float("nan")
    return math.sqrt(sum(z * z for z in zs) / len(zs))


def choose(rows: list[dict]) -> str | None:
    """The cheapest method (minutes) with 0 < se_C1, se_C2 <= TARGET_SE and z_rms <= Z_MAX
    (an SE of exactly 0 is degenerate, never a pass; NaN never passes)."""
    ok = [
        r
        for r in rows
        if 0 < r["se_C1"] <= TARGET_SE and 0 < r["se_C2"] <= TARGET_SE and r["z_rms"] <= Z_MAX
    ]
    return min(ok, key=lambda r: r["minutes"])["method"] if ok else None
