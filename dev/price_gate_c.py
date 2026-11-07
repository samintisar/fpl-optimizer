"""Gate C of the price planner (plan docs/superpowers/plans/2026-11-07-price-planner-stage-c.md).

    uv run python dev/price_gate_c.py tune --base RUN --arm 0=RUN --arm 0.25=RUN ... [--out DIR]
    uv run python dev/price_gate_c.py check --out DIR [--experiments results/experiments.csv]

    uv run python dev/price_gate_c.py verdict --out DIR   (recover after a crash past compare)

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
K_GRID = {0, 0.25, 0.5, 1}
CHECK_UNITS = 16  # 2 seasons x 8 starts
TUNE_UNITS = 8  # 2022/23 x 8 starts
POLICY_A_KEY, POLICY_B_KEY = "policy_a", "policy_b"  # top-level keys of compare's summary.json
COLUMNS = ["timestamp", "git_sha", "command", "config", "metrics", "n_variants", "family"]
# summary.json written by `backtest compare` (backtest.rs payload; evaluate.rs `comparisons_json`):
# the entry of summary["summary"]["comparisons"] with method "full_run", split "all", metric
# "realized"; its "ci_low" is the per-GW bootstrap CI's
# lower bound (ci 0.8) of the mean full-run difference A - B.
LOWER_BOUND_PATH = "summary.comparisons[method=full_run, split=all, metric=realized].ci_low"


# ---- pure functions ----


def units(gws: pd.DataFrame) -> pd.DataFrame:
    """Per (season, start): season net points and end team value = market_value + bank at the
    last GW (tenths). Counting the bank keeps a planner from raising the value just by spending it;
    it matches FPL's `value` used in the sizing. Gate A's `units` (market_value only) stays as
    it was run."""
    g = gws.sort_values("gw_index").groupby(["season", "start_id"])
    return pd.DataFrame(
        {
            "points": g["net_points"].sum(),
            "end_value": g["market_value"].last() + g["bank"].last(),
        }
    )


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
    """Mean end team value (market_value + bank) difference (A - B) over paired units and the
    one-sided t-test p."""
    u = {p: units(gws[gws["policy"] == p]) for p in (a, b)}
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


def points_lower_bound(summary: dict) -> float:
    for c in summary["summary"]["comparisons"]:
        if c["method"] == "full_run" and c["split"] == "all" and c["metric"] == "realized":
            return float(c["ci_low"])
    raise KeyError(f"no {LOWER_BOUND_PATH} in summary.json")


# ---- commands ----


def cmd_tune(args) -> None:
    def load(p: Path) -> pd.DataFrame:
        return units(pd.read_parquet(p / "gws.parquet"))

    base = load(args.base)
    if len(base) != TUNE_UNITS:
        raise SystemExit(f"base run has {len(base)} units, expected {TUNE_UNITS}")
    results = {}
    for item in args.arm:
        k, _, path = item.partition("=")
        results[k] = gate_a.compare(base, load(Path(path)))
        if results[k]["n"] != TUNE_UNITS:
            raise SystemExit(f"arm {k}: {results[k]['n']} paired units, expected {TUNE_UNITS}")
    out = tune_choice(results)
    bad = [k for k, a in out["arms"].items() if math.isnan(a["deflated"])]
    if bad:
        raise SystemExit(f"arms {bad}: deflated mean is NaN (fewer than 2 units)")
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


def _resolve(p: Path) -> Path:
    return p if p.is_absolute() else REPO / p


def finalize(out: Path, experiments: Path, k: float) -> dict:
    """Verdict from an existing `backtest compare` output in DIR/compare; writes gate_c.json and
    appends the verdict row."""
    gws = pd.read_parquet(out / "compare" / "gws.parquet")
    summary = json.loads((out / "compare" / "summary.json").read_text())
    a_name, b_name = summary[POLICY_A_KEY], summary[POLICY_B_KEY]
    tv_mean, tv_p, n = team_value_test(gws, a_name, b_name)
    if n != CHECK_UNITS:
        raise SystemExit(f"expected {CHECK_UNITS} paired units, got {n}")
    ci_low = points_lower_bound(summary)
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
        experiments,
        "price_gate_c.py check " + str(out),
        {"price_k": k, "out": str(out)},
        result,
        1,
        VERDICT_FAMILY,
    )
    return result


def _report(result: dict) -> None:
    print(json.dumps(result, indent=2))
    print(f"points lower bound read from {LOWER_BOUND_PATH}")
    print(
        f"ci_low per GW = {result['ci_low_per_gw']:.4f}; "
        f"lower_season = {result['lower_season']:.2f} (x{SEASON_GWS})"
    )
    print("GATE C: " + ("PASS" if result["pass"] else "FAIL"))


def _frozen_k() -> float:
    k = json.loads((REPO / "config" / "prices" / "planner.json").read_text())["price_k"]
    if k not in K_GRID:
        raise SystemExit(f"refused: frozen price_k {k} is not in {sorted(K_GRID)}")
    return k


def cmd_check(args) -> int:
    out: Path = args.out
    experiments = _resolve(args.experiments)
    nonempty = out.exists() and any(out.iterdir())
    reason = refusal(
        _log_families(experiments),
        _dirty(),
        _git("rev-parse", "HEAD"),
        _git("rev-parse", "--verify", "--quiet", f"{TAG}^{{commit}}") or None,
        nonempty,
    )
    if reason:
        print(f"refused: {reason}", file=sys.stderr)
        return 2
    k = _frozen_k()
    subprocess.run(
        ["cargo", "build", "--release", "-p", "fplopt-cli", "-j", str(args.cargo_jobs)],
        check=True,
        cwd=REPO,
    )
    binary = _resolve(args.binary) if args.binary else REPO / "target" / "release" / "fplopt-rs"
    a = f"{LIVE},prices=forecast,price_k={k}"
    cmd = [
        str(binary),
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
        str(experiments),
        "--out",
        str(out / "compare"),
    ]
    if args.jobs:
        cmd += ["--jobs", str(args.jobs)]
    if args.xp_cache:
        cmd += ["--xp-cache", str(args.xp_cache)]
    out.mkdir(parents=True, exist_ok=True)
    log = out / "compare.log"
    with log.open("w", encoding="utf-8") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO)
    if r.returncode != 0:
        print(f"backtest compare failed ({r.returncode}); see {log}", file=sys.stderr)
        return r.returncode
    result = finalize(out, experiments, k)
    tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
    print("\n".join(tail))
    _report(result)
    return 0


def cmd_verdict(args) -> int:
    experiments = _resolve(args.experiments)
    if VERDICT_FAMILY in set(_log_families(experiments)):
        print(f"refused: the log already holds family '{VERDICT_FAMILY}'", file=sys.stderr)
        return 2
    result = finalize(args.out, experiments, _frozen_k())
    _report(result)
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
    c.add_argument("--binary", type=Path)
    c.add_argument("--cargo-jobs", type=int, default=2)
    c.add_argument("--jobs", type=int)
    c.add_argument("--xp-cache", type=Path)
    v = sub.add_parser("verdict", help="recompute the verdict from an existing compare output")
    v.add_argument("--out", type=Path, required=True)
    v.add_argument("--experiments", type=Path, default=Path("results/experiments.csv"))
    args = ap.parse_args()
    if args.cmd == "tune":
        cmd_tune(args)
    elif args.cmd == "verdict":
        sys.exit(cmd_verdict(args))
    else:
        sys.exit(cmd_check(args))


if __name__ == "__main__":
    main()
