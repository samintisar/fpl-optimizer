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

import argparse
import json
import math
import subprocess
import time
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


def windows_per_season(k: int) -> int:
    """Decisions a season on the grid GW 1, 1 + k, ...: 38, 19, 10 for k = 1, 2, 4."""
    return math.ceil(SEASON_GWS / k)


def per_decision(dec: pd.DataFrame, k: int, col: str) -> tuple[float, float, int]:
    """Each decision window's A - B (summed over its k GWs) averaged over the starts live in it;
    windows of all seasons pooled (keyed by season and gw_index); mean and SE over windows,
    scaled by the windows a season on the decision grid (windows_per_season(k)), not by the
    windows found in the data."""
    windows = _windows(dec, col)
    return _stats(windows, windows_per_season(k))


def season_clustered_se(dec_or_paired: pd.DataFrame, k: int | None, col: str) -> float:
    """Diagnostic: SE over per-season means (C2 has 2 seasons, so this is rough). Per decision
    the scale is windows_per_season(k), as in per_decision."""
    d = dec_or_paired.dropna(subset=[col])
    windows = d.groupby(["season", "gw_index"])[col].mean()
    scale = SEASON_GWS if k is None else windows_per_season(k)
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


def passes(row: dict) -> bool:
    """The rule: 0 < se_C1, se_C2 <= TARGET_SE and z_rms <= Z_MAX (an SE of exactly 0 is
    degenerate, never a pass; NaN never passes)."""
    return 0 < row["se_C1"] <= TARGET_SE and 0 < row["se_C2"] <= TARGET_SE and row["z_rms"] <= Z_MAX


def choose(rows: list[dict]) -> str | None:
    """The cheapest method (minutes) that passes."""
    ok = [r for r in rows if passes(r)]
    return min(ok, key=lambda r: r["minutes"])["method"] if ok else None


def run_name(k: int | None) -> str:
    return "full" if k is None else f"k{k}"


def _method_runs():
    yield "whole", None
    for k in KS:
        yield f"k{k}", k


def _measure(run_dir: Path, k: int | None, col: str) -> tuple[float, float, int]:
    if k is None:
        return whole_season(pd.read_parquet(run_dir / "paired.parquet"), col)
    return per_decision(pd.read_parquet(run_dir / "per_decision.parquet"), k, col)


def method_rows(out: Path) -> list[dict]:
    """One row per method: C1 and C2 mean / SE / n per season, placebo z RMS (plus z sd, mean z
    and count as diagnostics), C2's season-clustered SE (diagnostic), minutes (C1 + C2 runs)."""
    times_path = out / "times.json"
    times = json.loads(times_path.read_text()) if times_path.exists() else {}
    placebos = [c for c in comparisons() if c.startswith("P")]
    rows = []
    for name, k in _method_runs():
        for score, col in SCORES.items():
            row = {"method": f"{name}/{score}"}
            for cid in ("C1", "C2"):
                mean, se, n = _measure(out / cid / run_name(k), k, col)
                row |= {f"mean_{cid}": mean, f"se_{cid}": se, f"n_{cid}": n}
            src = "paired.parquet" if k is None else "per_decision.parquet"
            row["se_C2_season_clustered"] = season_clustered_se(
                pd.read_parquet(out / "C2" / run_name(k) / src), k, col
            )
            pairs = [_measure(out / p / run_name(k), k, col)[:2] for p in placebos]
            zs = _zs(pairs)
            row["z_rms"] = z_rms(pairs)
            row["z_sd"] = z_sd(pairs)
            row["z_mean"] = sum(zs) / len(zs) if zs else float("nan")
            row["z_count"] = z_count(pairs)
            row["minutes"] = (
                sum(times.get(f"{c}/{run_name(k)}", float("nan")) for c in ("C1", "C2")) / 60
            )
            row["pass"] = passes(row)
            rows.append(row)
    return rows


def cmd_run(args) -> int:
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    times_path = out / "times.json"
    times = json.loads(times_path.read_text()) if times_path.exists() else {}
    only = set(args.only.split(",")) if args.only else None
    for cid, comp in comparisons().items():
        if only and cid not in only:
            continue
        for k in (None, *KS):
            run_dir = out / cid / run_name(k)
            if (run_dir / "summary.json").exists():
                continue
            cmd = compare_cmd(args.binary, comp, k, run_dir, args.jobs, args.experiments)
            run_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.monotonic()
            with (run_dir / "compare.log").open("w", encoding="utf-8") as log:
                r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            if r.returncode != 0:
                print(f"{cid}/{run_name(k)} failed ({r.returncode}); see {run_dir / 'compare.log'}")
                return r.returncode
            times[f"{cid}/{run_name(k)}"] = time.monotonic() - t0
            times_path.write_text(json.dumps(times, indent=2))
            print(f"{cid}/{run_name(k)}: {times[f'{cid}/{run_name(k)}'] / 60:.1f} min", flush=True)
    return 0


def cmd_report(args) -> int:
    rows = method_rows(args.out)
    chosen = choose(rows)
    payload = {
        "target_se": TARGET_SE,
        "z_max": Z_MAX,
        "n_placebo": N_PLACEBO,
        "sigma": SIGMA,
        "starts": STARTS,
        "methods": rows,
        "chosen": chosen,
    }
    (args.out / "noise.json").write_text(json.dumps(payload, indent=2))
    head = f"{'method':<14} {'C1 mean':>8} {'C1 se':>6} {'C2 mean':>8} {'C2 se':>6} {'C2 se/s':>7}"
    print(head + f" {'z rms':>5} {'z sd':>5} {'z mean':>6} {'z n':>3} {'min':>6} pass")
    for r in rows:
        print(
            f"{r['method']:<14} {r['mean_C1']:>8.1f} {r['se_C1']:>6.2f} {r['mean_C2']:>8.1f} "
            f"{r['se_C2']:>6.2f} {r['se_C2_season_clustered']:>7.2f} {r['z_rms']:>5.2f} "
            f"{r['z_sd']:>5.2f} {r['z_mean']:>6.2f} {r['z_count']:>3d} {r['minutes']:>6.1f} "
            f"{'yes' if r['pass'] else 'no'}"
        )
    print(f"chosen: {chosen}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--binary", type=Path, default=Path("target/release/fplopt-rs"))
    r.add_argument("--jobs", type=int, default=14)
    r.add_argument("--experiments", type=Path, default=Path("results/experiments.csv"))
    r.add_argument("--only", help="comma-separated comparison ids, e.g. C1,P3")
    p = sub.add_parser("report")
    p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    raise SystemExit(cmd_run(args) if args.cmd == "run" else cmd_report(args))


if __name__ == "__main__":
    main()
