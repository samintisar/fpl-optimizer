"""Rust migration batch 2, the parity gate: `fplopt build all` and `fplopt-rs build all` from
the same `raw/` must write the same tables (docs/superpowers/plans/2026-10-28-rust-migration.md,
Batch 2). Dev only: never imported by `fplopt`.

Both builds run as subprocesses from the repo root (so both read `config/`), each into its
own data directory; alerts and the heartbeat are switched off. Then every table in
`fplopt.build.tables.TABLES` is compared with `fplopt-rs check parity table` on the table's
key (rows, schema and values; floats within rtol 1e-9, null != NaN), and the two builds must
write the same set of files.

    cargo build --release -p fplopt-cli
    uv run python dev/build_parity.py [--raw raw] [--bin target/release/fplopt-rs]
        [--python-data DIR (reuse a Python build)] [--keep DIR]

Takes ~5 minutes on the laptop (Python ~4, Rust ~2). Logs one row to
results/experiments.csv (family `rust parity`) and exits 1 on any difference.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fplopt.backtest.evaluate import family_variants, log_experiment  # noqa: E402
from fplopt.build.tables import TABLES  # noqa: E402

FAMILY = "rust parity"
QUIET = {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_ADMIN_CHAT_ID": "", "HEALTHCHECK_PING_URL": ""}


def run_build(command: list[str], raw: Path, data: Path, log: Path) -> tuple[bool, float]:
    data.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **QUIET, "FPLOPT_RAW_DIR": str(raw), "FPLOPT_DATA_DIR": str(data)}
    started = time.monotonic()
    with open(log, "w", encoding="utf-8") as out:
        result = subprocess.run(command, cwd=REPO, env=env, stdout=out, stderr=subprocess.STDOUT)
    return result.returncode == 0, time.monotonic() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--raw", type=Path, default=REPO / "raw", help="raw/ to build from (read only)"
    )
    parser.add_argument("--bin", type=Path, default=REPO / "target" / "release" / "fplopt-rs")
    parser.add_argument("--python-data", type=Path, help="an existing Python build of --raw")
    parser.add_argument("--keep", type=Path, help="keep the data directories here")
    parser.add_argument("--experiments", type=Path, default=REPO / "results" / "experiments.csv")
    args = parser.parse_args()
    binary = args.bin if args.bin.exists() else args.bin.with_suffix(".exe")
    if not binary.exists():
        parser.error(f"no Rust binary at {args.bin} (cargo build --release -p fplopt-cli)")
    raw = args.raw.resolve()

    work = Path(tempfile.mkdtemp(prefix="build-parity-")) if args.keep is None else args.keep
    work.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    if args.python_data is None:
        py_data = work / "python"
        print("python: fplopt build all", flush=True)
        py_ok, timings["python_s"] = run_build(
            [sys.executable, "-m", "fplopt.cli", "build", "all"], raw, py_data, work / "python.log"
        )
    else:
        py_data, py_ok = args.python_data.resolve(), True
    rs_data = work / "rust"
    print("rust: fplopt-rs build all", flush=True)
    rs_ok, timings["rust_s"] = run_build(
        [str(binary), "build", "all"], raw, rs_data, work / "rust.log"
    )

    problems: list[str] = []
    results: dict[str, bool] = {}
    if not (py_ok and rs_ok):
        problems.append(
            f"build failed: python {'ok' if py_ok else 'FAILED'}, "
            f"rust {'ok' if rs_ok else 'FAILED'} (logs in {work})"
        )
    else:
        py_files = {p.name for p in py_data.glob("*.parquet")}
        rs_files = {p.name for p in rs_data.glob("*.parquet")}
        if py_files != rs_files:
            problems.append(
                f"table sets differ: only python {sorted(py_files - rs_files)}, "
                f"only rust {sorted(rs_files - py_files)}"
            )
        for name, spec in sorted(TABLES.items()):
            command = [
                str(binary), "check", "parity", "table",
                "--python", str(py_data / f"{name}.parquet"),
                "--rust", str(rs_data / f"{name}.parquet"),
                "--keys", ",".join(spec.key),
                "--out", str(work / "parity" / name),
                "--experiments", str(work / "tables.csv"),
            ]  # fmt: skip
            result = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
            results[name] = result.returncode == 0
            lines = (result.stdout + result.stderr).strip().splitlines()
            print("\n  ".join(lines[:6]), flush=True)
            if result.returncode != 0:
                problems.append(f"{name}: {lines[0] if lines else 'no output'}")

    passed = not problems
    print(f"\n{'PASS' if passed else 'FAIL'}: {sum(results.values())}/{len(TABLES)} tables")
    for problem in problems:
        print(f"  {problem}")
    print("  " + ", ".join(f"{k} {v:.0f}" for k, v in timings.items()))
    family_n = family_variants(args.experiments, FAMILY) + 1
    log_experiment(
        args.experiments,
        shlex.join(["python", "dev/build_parity.py", *sys.argv[1:]]),
        {"gate": "batch 2 build", "raw": str(raw), "binary": str(binary)},
        {"passed": passed, "tables": results, **{k: round(v, 1) for k, v in timings.items()}},
        family_n,
        FAMILY,
    )
    if args.keep is None:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
