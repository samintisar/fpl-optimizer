"""Batch 7 gate: the logged research runs on both implementations.

Every `fplopt backtest run|compare` and `fplopt models eval` command in the experiment log
(`results/experiments.csv`; `holdout run` excluded: it runs once) is re-run with today's
Python (`uv run fplopt ...`) and today's Rust (`target/release/fplopt-rs ...`), each with its
own --out and experiment log (so both log n_variants = 1), and compared:

- the output tables (backtest: gws, paired, per_decision; eval: predictions, regrets) row
  for row, numbers within --rtol (1e-7 by default: v1's xP differs from Python's in the 9th
  digit, batch 6's fits within 5.5e-8; the 2016-22 eval's predictions 3.8e-9 apart, every
  other table exact);
- every metric of the logged row (and of eval's metrics.json) within --metrics-rtol (1e-6,
  the migration plan's tolerance for point estimates; the bootstrap draws numpy's generator
  draw for draw, so CIs fall under it too).

Backtests with `v1` xP run Rust on Python's v1 frames: v1's xP is 1e-8 from Python's (batch
6's gate), enough to tip a MILP stopped at its 0.5% gap onto another plan and the runs apart.
Rust runs once on its own xP (filling a disk cache with the deadlines it needs), those frames
are replaced by Python's (`python_xp_cache`, kept across commands), and the run compared is
Rust's on them; that run must not miss the cache. `--no-python-xp` compares Rust's own.

numpy's dot products round as OpenBLAS's kernel for the CPU does (FMA on AVX-512 Intel, not
on Zen or Haswell); Rust follows the Haswell kernel, so Python runs with
OPENBLAS_CORETYPE=Haswell (a no-op where OpenBLAS picks it anyway).

Since v1r's adoption (2026-11-03) Rust's `v1` is not Python's (that is Rust's `v1-2026-10`),
so the logged v1 commands only compare from a checkout before the rename (`rust-migration`
at 2b49a64, where this gate passed).

The log's rows were made by older commits, so the reference is today's Python, not the
logged values. The logged runs take hours (v1 chip runs: ~3 h each in Python); `--select`
picks commands by regex, `--list` prints them. Results go to `results/experiments.csv`
(family `rust parity`).

    uv run python dev/backtest_parity.py --list
    uv run python dev/backtest_parity.py --select "greedy:rolling --b roll" [--jobs 8]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

# Before numpy loads: this process computes Python's v1 xP too (`python_xp_cache`).
os.environ["OPENBLAS_CORETYPE"] = "Haswell"

import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fplopt.backtest.evaluate import log_experiment  # noqa: E402

FAMILY = "rust parity"
QUIET = {
    "TELEGRAM_BOT_TOKEN": "",
    "TELEGRAM_ADMIN_CHAT_ID": "",
    "HEALTHCHECK_PING_URL": "",
    "OPENBLAS_CORETYPE": "Haswell",
}
DROP = {"--out", "--experiments", "--jobs"}
TABLES = {
    "backtest": ("gws", "paired", "per_decision"),
    "models": ("predictions", "regrets"),
}


def logged_commands(path: Path) -> list[list[str]]:
    """The logged research commands as argv (without the program), in log order, deduped."""
    out, seen = [], set()
    for row in csv.DictReader(path.open(encoding="utf-8")):
        # a trailing "[note]" (e.g. "[run on the server, ...]") is not part of the command
        argv = shlex.split(re.sub(r"\s*\[[^\]]*\]\s*$", "", row["command"]), posix=True)
        if (
            not argv
            or argv[0] != "fplopt"
            or argv[1:3]
            not in (
                ["backtest", "run"],
                ["backtest", "compare"],
                ["models", "eval"],
            )
        ):
            continue
        args, skip = [], False
        for a in argv[1:]:
            if skip:
                skip = False
                continue
            name = a.split("=", 1)[0]
            if name in DROP:
                skip = "=" not in a
                continue
            args.append(a)
        key = tuple(args)
        if key not in seen:
            seen.add(key)
            out.append(args)
    return out


def run(
    argv: list[str], out: Path, log: Path, rust: Path | None, jobs: int, extra: tuple = ()
) -> tuple[bool, float, str]:
    command = [str(rust)] if rust else ["uv", "run", "fplopt"]
    command += [*argv, "--out", str(out), "--experiments", str(log), "--jobs", str(jobs)]
    command += extra
    started = time.monotonic()
    done = subprocess.run(
        command, cwd=REPO, env={**os.environ, **QUIET}, capture_output=True, text=True
    )
    tail = (done.stderr or done.stdout).strip().splitlines()[-1:] if done.returncode else []
    return done.returncode == 0, time.monotonic() - started, " ".join(tail)


def uses_v1(argv: list[str]) -> bool:
    """A backtest whose policy specs (or --xp) use the v1 model."""
    return argv[0] == "backtest" and any(re.search(r"(^|:)v1(:|$)", a) for a in argv)


def _python_v1(files: list[Path]) -> list[tuple[Path, pd.DataFrame]]:
    """Python's v1 frame at each cached deadline (one Caches: refits shared within a chunk)."""
    from fplopt.backtest.simulator import Caches
    from fplopt.features.store import DataStore

    store, caches = DataStore(REPO / "data"), Caches()
    out = []
    for f in files:
        deadline = pd.Timestamp(int(f.stem), unit="us", tz="UTC")
        out.append((f, caches.xp(store, "v1", store.as_of(deadline)).reset_index(drop=True)))
    return out


def python_xp_cache(rust_root: Path, python_root: Path, jobs: int) -> list[str]:
    """Copy Rust's disk cache (`<root>/<key>/<model>/<deadline>.parquet` and `.kinds`) to
    `python_root` with the v1 frames replaced by Python's (Rust's schema and kinds kept);
    deadlines already there are kept. Problems: frames whose columns or rows don't line up."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    todo, problems = [], []
    for f in sorted(rust_root.rglob("*.parquet")):
        target = python_root / f.relative_to(rust_root)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if f.parent.name == "v1":
            todo.append(f)
        else:
            shutil.copy2(f, target)
            shutil.copy2(f.with_suffix(".kinds"), target.with_suffix(".kinds"))
    if not todo:
        return problems
    size = -(-len(todo) // jobs)  # contiguous chunks keep each process's refit cutoffs together
    chunks = [todo[i : i + size] for i in range(0, len(todo), size)]
    with Pool(len(chunks)) as pool:
        frames = [x for chunk in pool.map(_python_v1, chunks) for x in chunk]
    for f, py in frames:
        rs = pq.read_table(f)
        target = python_root / f.relative_to(rust_root)
        keys = [c for c in ("player_key", "horizon") if c in py.columns]
        if list(py.columns) != rs.column_names or len(py) != rs.num_rows:
            problems.append(f"v1 xP at {f.stem}: columns or row count differ")
        elif (py[keys].to_numpy("int64") != rs.select(keys).to_pandas().to_numpy("int64")).any():
            problems.append(f"v1 xP at {f.stem}: rows in another order")
        else:
            table = pa.Table.from_pandas(py, schema=rs.schema, preserve_index=False)
            pq.write_table(table, target)
            shutil.copy2(f.with_suffix(".kinds"), target.with_suffix(".kinds"))
    return problems


def n_cached(root: Path) -> int:
    return sum(1 for _ in root.rglob("*.parquet")) if root.exists() else 0


def compare_tables(a: Path, b: Path, rtol: float) -> list[str]:
    left, right = pd.read_parquet(a), pd.read_parquet(b)
    if list(left.columns) != list(right.columns):
        return [f"{a.name}: columns differ"]
    if len(left) != len(right):
        return [f"{a.name}: {len(left)} vs {len(right)} rows"]
    problems = []
    for c in left.columns:
        x, y = left[c], right[c]
        if x.dtype.kind in "fiu" or str(x.dtype) in ("Float64", "Int64"):
            xf, yf = x.astype("float64"), y.astype("float64")
            ok = (xf.isna() & yf.isna()) | ((xf - yf).abs() <= rtol * (1 + xf.abs()))
        else:
            ok = (x.isna() & y.isna()) | (x.astype(str) == y.astype(str))
        if not ok.all():
            i = (~ok).idxmax()
            problems.append(f"{a.name}.{c}: {int((~ok).sum())} rows, e.g. {x[i]!r} vs {y[i]!r}")
    return problems


def compare_values(p, q, rtol: float, path: str, out: list[str]) -> None:
    if isinstance(p, dict) and isinstance(q, dict):
        if set(p) != set(q):
            out.append(f"{path}: keys {sorted(set(p) ^ set(q))}")
        for k in p:
            if k in q:
                compare_values(p[k], q[k], rtol, f"{path}.{k}", out)
    elif isinstance(p, list) and isinstance(q, list):
        if len(p) != len(q):
            out.append(f"{path}: {len(p)} vs {len(q)} items")
        for i, (x, y) in enumerate(zip(p, q, strict=False)):
            compare_values(x, y, rtol, f"{path}[{i}]", out)
    elif isinstance(p, float) or isinstance(q, float):
        if p is None or q is None:
            if p != q:
                out.append(f"{path}: {p} vs {q}")
        elif abs(p - q) > rtol * max(1.0, abs(p)):
            out.append(f"{path}: {p} vs {q}")
    elif p != q:
        out.append(f"{path}: {p!r} vs {q!r}")


def last_metrics(log: Path) -> dict:
    rows = list(csv.DictReader(log.open(encoding="utf-8")))
    return json.loads(rows[-1]["metrics"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--experiments", type=Path, default=REPO / "results" / "experiments.csv")
    parser.add_argument("--select", help="regex over the commands to run (default: all)")
    parser.add_argument("--list", action="store_true", help="print the commands and exit")
    parser.add_argument("--jobs", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    parser.add_argument("--rtol", type=float, default=1e-7)
    parser.add_argument("--metrics-rtol", type=float, default=1e-6)
    parser.add_argument("--keep", type=Path, help="keep the outputs here")
    parser.add_argument(
        "--no-python-xp", action="store_true", help="v1 backtests on Rust's own xP (module doc)"
    )
    args = parser.parse_args()
    commands = logged_commands(args.experiments)
    if args.select:
        commands = [c for c in commands if re.search(args.select, shlex.join(c))]
    if args.list:
        for c in commands:
            print(shlex.join(c))
        return 0
    if not commands:
        raise SystemExit("no command selected")
    binary = REPO / "target" / "release" / ("fplopt-rs.exe" if os.name == "nt" else "fplopt-rs")
    work = args.keep or Path(os.environ.get("TEMP", "/tmp")) / "backtest-parity"
    shutil.rmtree(work, ignore_errors=True)
    results, timings = {}, {}
    for n, argv in enumerate(commands):
        label = shlex.join(argv)
        case = work / f"{n:02d}"
        print(f"[{n + 1}/{len(commands)}] {label}", flush=True)
        problems = []
        python_xp = uses_v1(argv) and not args.no_python_xp
        rust_root, python_root = work / "xp-rust", work / "xp-python"
        steps = [("python", None, ())]
        if python_xp:
            steps.append(("rust-own", binary, ("--xp-cache", str(rust_root))))
        steps.append(("rust", binary, ("--xp-cache", str(python_root)) if python_xp else ()))
        for who, rust, extra in steps:
            if who == "rust" and python_xp:
                started = time.monotonic()
                problems += python_xp_cache(rust_root, python_root, args.jobs)
                timings[f"{n:02d} python_xp_s"] = round(time.monotonic() - started)
                before = n_cached(python_root)
            ok, seconds, tail = run(argv, case / who, case / f"{who}.csv", rust, args.jobs, extra)
            timings[f"{n:02d} {who}_s"] = round(seconds)
            if not ok:
                problems.append(f"{who} failed: {tail}")
            if who == "rust" and python_xp and n_cached(python_root) != before:
                problems.append("Rust's run missed the Python xP cache (it computed its own)")
        if not problems:
            for table in TABLES[argv[0]]:
                a, b = case / "python" / f"{table}.parquet", case / "rust" / f"{table}.parquet"
                if a.exists() != b.exists():
                    problems.append(f"{table}: written by one side only")
                elif a.exists():
                    problems += compare_tables(a, b, args.rtol)
            compare_values(
                last_metrics(case / "python.csv"), last_metrics(case / "rust.csv"),
                args.metrics_rtol, "log", problems,
            )  # fmt: skip
            if argv[0] == "models":
                pm = json.loads((case / "python" / "metrics.json").read_text(encoding="utf-8"))
                rm = json.loads((case / "rust" / "metrics.json").read_text(encoding="utf-8"))
                compare_values(pm["metrics"], rm["metrics"], args.metrics_rtol, "metrics", problems)
        results[label] = not problems
        py_s, rs_s = timings[f"{n:02d} python_s"], timings[f"{n:02d} rust_s"]
        verdict = "PASS" if not problems else "FAIL"
        on = " on Python's v1 xP" if python_xp else ""
        print(f"  {verdict}{on} (python {py_s} s, rust {rs_s} s)", flush=True)
        for p in problems[:12]:
            print(f"    {p}", flush=True)
    passed = all(results.values())
    print(f"\n{'PASS' if passed else 'FAIL'}: {sum(results.values())}/{len(results)} commands")
    log_experiment(
        REPO / "results" / "experiments.csv",
        "python " + shlex.join(["dev/backtest_parity.py", *sys.argv[1:]]),
        {
            "select": args.select,
            "rtol": args.rtol,
            "metrics_rtol": args.metrics_rtol,
            "python_xp": not args.no_python_xp,
        },
        {"passed": passed, "results": results, "timings": timings},
        1,
        FAMILY,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
