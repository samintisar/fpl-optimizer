"""Gate C of the price planner (plan docs/superpowers/plans/2026-11-07-price-planner-stage-c.md).

    uv run python dev/price_gate_c.py tune --base RUN --arm 0=RUN --arm 0.25=RUN ... [--out DIR]
    uv run python dev/price_gate_c.py check --out DIR [--experiments results/experiments.csv]

`tune` picks the money weight k on develop 2022/23 (highest deflated points mean). `check` is the
run-once pre-registered gate: refusals, `backtest compare`, verdict, log.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("price_gate_a", REPO / "dev" / "price_gate_a.py")
gate_a = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = gate_a
_spec.loader.exec_module(gate_a)

P_MAX = 0.10
LOWER_MIN = -10.0  # season points
# The bootstrap CI is of the per-GW mean difference (stats.rs block_bootstrap); the spec's -10 is in
# season points, so the bound is scaled by the GWs in a season.
SEASON_GWS = 38
TAG = "price-check"
FAMILY = "price check"
VERDICT_FAMILY = "price check verdict"
LIVE = "optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2"
COLUMNS = ["timestamp", "git_sha", "command", "config", "metrics", "n_variants", "family"]
# metrics.json written by `backtest compare` (evaluate.rs `comparisons_json`): the entry of the
# top-level "comparisons" list with method "full_run", split "all", metric "realized"; its "ci_low"
# is the bootstrap CI's lower bound (ci 0.8) of the mean full-run difference A - B.
LOWER_BOUND_PATH = "comparisons[method=full_run, split=all, metric=realized].ci_low"


# ---- pure functions ----


def refusal(log_families, dirty, head, tag, out_nonempty) -> str | None:
    if FAMILY in set(log_families):
        return f"the experiment log already holds family '{FAMILY}': Gate C runs once"
    if dirty:
        return "the working tree has uncommitted changes"
    if tag is None or head != tag:
        return f"HEAD is not the commit tagged '{TAG}'"
    if out_nonempty:
        return "--out exists and is not empty"
    return None


def tune_choice(results: dict) -> dict:
    """results: arm label -> `price_gate_a.compare` output. Deflate each points mean by
    SE*sqrt(2 ln N), N = number of arms; highest wins, higher end-value mean breaks ties (1e-9)."""
    n = len(results)
    pen = math.sqrt(2 * math.log(n)) if n > 1 else 0.0
    arms = {}
    for k, r in results.items():
        arms[k] = {
            "points_mean": r["points"]["mean"],
            "points_se": r["points"]["se"],
            "deflated": r["points"]["mean"] - r["points"]["se"] * pen,
            "end_value_mean": r["end_value"]["mean"],
        }
    chosen = None
    for k, a in arms.items():
        if chosen is None:
            chosen = k
            continue
        c = arms[chosen]
        gap = a["deflated"] - c["deflated"]
        if gap > 1e-9 or (abs(gap) <= 1e-9 and a["end_value_mean"] > c["end_value_mean"]):
            chosen = k
    return {"chosen": chosen, "n_arms": n, "arms": arms}


def team_value_test(gws: pd.DataFrame, a: str, b: str) -> tuple[float, float, int]:
    """Mean end market_value difference (A - B) over paired units and the one-sided t-test p."""
    u = {p: gate_a.units(gws[gws["policy"] == p]) for p in (a, b)}
    joined = u[a].join(u[b], lsuffix="_a", rsuffix="_b", how="inner")
    d = joined["end_value_a"] - joined["end_value_b"]
    n = int(d.size)
    if n < 2:
        return float(d.mean()), float("nan"), n
    mean = float(d.mean())
    if d.std(ddof=1) == 0:
        return mean, (0.5 if mean == 0 else (0.0 if mean > 0 else 1.0)), n
    t = stats.ttest_1samp(d, 0.0, alternative="greater")
    return mean, float(t.pvalue), n


def season_bound(ci_low_per_gw: float) -> float:
    return ci_low_per_gw * SEASON_GWS


def verdict(p: float, lower: float) -> bool:
    """lower: the points lower bound in season terms (ci_low per GW x SEASON_GWS)."""
    return bool(p < P_MAX and lower > LOWER_MIN)


def points_lower_bound(metrics: dict) -> float:
    for c in metrics["comparisons"]:
        if c["method"] == "full_run" and c["split"] == "all" and c["metric"] == "realized":
            return float(c["ci_low"])
    raise KeyError(f"no {LOWER_BOUND_PATH} in metrics.json")


# ---- commands ----


def cmd_tune(args) -> None:
    def load(p: Path) -> pd.DataFrame:
        return gate_a.units(pd.read_parquet(p / "gws.parquet"))

    base = load(args.base)
    results = {}
    for item in args.arm:
        k, _, path = item.partition("=")
        results[k] = gate_a.compare(base, load(Path(path)))
    out = tune_choice(results)
    out["results"] = results
    print(f"{'k':>6} {'points':>9} {'se':>7} {'deflated':>9} {'end value':>10}")
    for k, a in out["arms"].items():
        print(
            f"{k:>6} {a['points_mean']:>9.2f} {a['points_se']:>7.2f} {a['deflated']:>9.2f} "
            f"{a['end_value_mean']:>10.2f}"
        )
    print(f"chosen k = {out['chosen']}")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "tune.json").write_text(json.dumps(out, indent=2))


def _git(*a: str) -> str:
    r = subprocess.run(["git", *a], capture_output=True, text=True, cwd=REPO)
    return r.stdout.strip() if r.returncode == 0 else ""


def _log_families(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [row.get("family", "") for row in csv.DictReader(f)]


def _dirty() -> bool:
    lines = _git("status", "--porcelain").splitlines()
    return any(ln[3:].strip().replace("\\", "/") != "results/experiments.csv" for ln in lines)


def append_log(path: Path, command: str, config: dict, metrics: dict, n_variants: int, family: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        if new:
            w.writerow(COLUMNS)
        w.writerow(
            [
                datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                _git("rev-parse", "HEAD"),
                command,
                json.dumps(config, sort_keys=True),
                json.dumps(metrics, sort_keys=True),
                n_variants,
                family,
            ]
        )


def cmd_check(args) -> int:
    out: Path = args.out
    nonempty = out.exists() and any(out.iterdir())
    reason = refusal(
        _log_families(args.experiments),
        _dirty(),
        _git("rev-parse", "HEAD"),
        _git("rev-parse", "--verify", "--quiet", f"{TAG}^{{commit}}") or None,
        nonempty,
    )
    if reason:
        print(f"refused: {reason}", file=sys.stderr)
        return 2
    k = json.loads((REPO / "config" / "prices" / "planner.json").read_text())["price_k"]
    a = f"{LIVE},prices=forecast,price_k={k}"
    cmd = [
        str(args.binary),
        "backtest",
        "compare",
        "--a",
        a,
        "--b",
        LIVE,
        "--seasons",
        "2023,2024",
        "--starts",
        "template@1,random:3@1,template@20,random:3@20",
        "--no-per-decision",
        "--family",
        FAMILY,
        "--experiments",
        str(args.experiments),
        "--out",
        str(out / "compare"),
    ]
    if args.jobs:
        cmd += ["--jobs", str(args.jobs)]
    if args.xp_cache:
        cmd += ["--xp-cache", str(args.xp_cache)]
    subprocess.run(cmd, check=True)

    gws = pd.read_parquet(out / "compare" / "gws.parquet")
    names = sorted(gws["policy"].unique())
    a_name = next(n for n in names if "forecast" in n)
    b_name = next(n for n in names if n != a_name)
    tv_mean, tv_p, n = team_value_test(gws, a_name, b_name)
    metrics = json.loads((out / "compare" / "metrics.json").read_text())
    ci_low = points_lower_bound(metrics)
    lower = season_bound(ci_low)
    passed = verdict(tv_p, lower)
    result = {
        "price_k": k,
        "a": a_name,
        "b": b_name,
        "n_units": n,
        "team_value_mean": tv_mean,
        "team_value_p": tv_p,
        "ci_low_per_gw": ci_low,
        "lower_season": lower,
        "season_gws": SEASON_GWS,
        "points_lower_bound_path": LOWER_BOUND_PATH,
        "rule": f"p < {P_MAX} and lower_season > {LOWER_MIN}",
        "pass": passed,
    }
    (out / "gate_c.json").write_text(json.dumps(result, indent=2))
    append_log(
        args.experiments,
        "price_gate_c.py check " + " ".join(sys.argv[2:]),
        {"price_k": k, "out": str(out)},
        result,
        1,
        VERDICT_FAMILY,
    )
    print(json.dumps(result, indent=2))
    print(f"points lower bound read from {LOWER_BOUND_PATH}")
    print(f"ci_low per GW = {ci_low:.4f}; lower_season = {lower:.2f} (x{SEASON_GWS})")
    print("GATE C: " + ("PASS" if passed else "FAIL"))
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tune")
    t.add_argument("--base", type=Path, required=True)
    t.add_argument("--arm", action="append", required=True, metavar="K=RUN")
    t.add_argument("--out", type=Path)
    c = sub.add_parser("check")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--experiments", type=Path, default=Path("results/experiments.csv"))
    c.add_argument("--binary", type=Path, default=Path("target/release/fplopt-rs"))
    c.add_argument("--jobs", type=int)
    c.add_argument("--xp-cache", type=Path)
    args = ap.parse_args()
    if args.cmd == "tune":
        cmd_tune(args)
    else:
        sys.exit(cmd_check(args))


if __name__ == "__main__":
    main()
