"""Rust migration batch 3, the parity gate: every feature frame the Python and the Rust as-of
store compute from the same `data/` must be identical (docs/superpowers/plans/
2026-10-28-rust-migration.md, Batch 3). Dev only: never imported by `fplopt`.

Deadlines: the leakage check's edge list plus `--n-random` (default 10) sampled ones
(`fplopt.features.leakcheck.sample_deadlines`, seed 0). At each, Python computes every
registered feature (`fplopt.features.FEATURES`) and the minutes model's history frames
(`history.training_frame`, `history.prediction_frame`) and writes them as Parquet; then
`fplopt-rs features dump` writes the same frames from the same deadlines, and each pair is
compared with `fplopt-rs check parity table` on the frame's key (rows, schema and values;
floats within rtol 1e-9, null != NaN) plus the row order (every frame is sorted by its key).

    cargo build --release -p fplopt-cli
    uv run python dev/features_parity.py [--data data] [--bin target/release/fplopt-rs]
        [--n-random 10] [--keep DIR]

Logs one row to results/experiments.csv (family `rust parity`) and exits 1 on any difference.
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

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fplopt.backtest.evaluate import family_variants, log_experiment  # noqa: E402
from fplopt.features import FEATURES, history  # noqa: E402
from fplopt.features.leakcheck import sample_deadlines  # noqa: E402
from fplopt.features.store import DataStore  # noqa: E402

FAMILY = "rust parity"
QUIET = {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_ADMIN_CHAT_ID": "", "HEALTHCHECK_PING_URL": ""}
KEYS = {
    "player_pool": ["player_key"],
    "availability": ["player_key"],
    "ep_next": ["player_key"],
    "recent_form": ["player_key"],
    "upcoming_fixtures": ["team_key", "gw_index", "fixture_key"],
    "team_strength": ["fixture_key", "team_key"],
    "pool_coverage": ["team_key"],
    "history_training": ["player_key", "fixture_key"],
    "history_prediction": ["player_key", "fixture_key"],
}
BUILDERS = {
    **FEATURES,
    "history_training": history.training_frame,
    "history_prediction": history.prediction_frame,
}


def deadlines(data: Path, n_random: int) -> list[pd.Timestamp]:
    tables = {
        "gameweek": pd.read_parquet(data / "gameweek.parquet"),
        "player_match": pd.read_parquet(data / "player_match.parquet", columns=["available_at"]),
        "player_snapshot": pd.read_parquet(
            data / "player_snapshot.parquet", columns=["snapshot_at"]
        ),
    }
    return sample_deadlines(tables, n_random, seed=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=REPO / "data", help="data/ (read only)")
    parser.add_argument("--bin", type=Path, default=REPO / "target" / "release" / "fplopt-rs")
    parser.add_argument("--n-random", type=int, default=10)
    parser.add_argument("--keep", type=Path, help="keep the frames here")
    parser.add_argument("--experiments", type=Path, default=REPO / "results" / "experiments.csv")
    args = parser.parse_args()
    binary = args.bin if args.bin.exists() else args.bin.with_suffix(".exe")
    if not binary.exists():
        parser.error(f"no Rust binary at {args.bin} (cargo build --release -p fplopt-cli)")
    data = args.data.resolve()
    work = Path(tempfile.mkdtemp(prefix="features-parity-")) if args.keep is None else args.keep
    work.mkdir(parents=True, exist_ok=True)

    picked = deadlines(data, args.n_random)
    listing = work / "deadlines.txt"
    listing.write_text("".join(f"{d.isoformat()}\n" for d in picked), encoding="utf-8")
    print(f"{len(picked)} deadlines from {picked[0]} to {picked[-1]}", flush=True)

    timings: dict[str, float] = {}
    started = time.monotonic()
    store = DataStore(data)
    py_errors: list[str] = []
    for i, deadline in enumerate(picked):
        view = store.as_of(deadline)
        out = work / "python" / f"{i:02d}"
        out.mkdir(parents=True, exist_ok=True)
        for name, builder in BUILDERS.items():
            try:
                builder(view).to_parquet(out / f"{name}.parquet", index=False)
            except Exception as exc:  # a failure is a finding, reported below
                py_errors.append(f"{deadline} {name}: python {type(exc).__name__}: {exc}")
    timings["python_s"] = time.monotonic() - started
    print(f"python: {timings['python_s']:.0f} s", flush=True)

    started = time.monotonic()
    env = {**os.environ, **QUIET, "FPLOPT_DATA_DIR": str(data)}
    command = [str(binary), "features", "dump", "--deadlines", str(listing)]
    command += ["--out", str(work / "rust")]
    with open(work / "rust.log", "w", encoding="utf-8") as log:
        rust = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    timings["rust_s"] = time.monotonic() - started
    print(f"rust: {timings['rust_s']:.0f} s", flush=True)

    problems = list(py_errors)
    if rust.returncode != 0:
        problems.append(f"fplopt-rs features dump failed (log: {work / 'rust.log'})")
    results: dict[str, bool] = {}
    for i, deadline in enumerate(picked):
        for name, keys in KEYS.items():
            label = f"{i:02d} {deadline} {name}"
            py_file = work / "python" / f"{i:02d}" / f"{name}.parquet"
            rs_file = work / "rust" / f"{i:02d}" / f"{name}.parquet"
            if not (py_file.exists() and rs_file.exists()):
                results[label] = False
                problems.append(f"{label}: missing ({'python' if rs_file.exists() else 'rust'})")
                continue
            command = [
                str(binary), "check", "parity", "table",
                "--python", str(py_file), "--rust", str(rs_file), "--keys", ",".join(keys),
                "--out", str(work / "parity" / f"{i:02d}" / name),
                "--experiments", str(work / "frames.csv"),
            ]  # fmt: skip
            result = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
            ok = result.returncode == 0
            if ok:
                # Dtypes are the parity check's job (a nullable Int64 key without nulls is
                # int64 on the Rust side); this compares the key values row by row.
                py_keys = pd.read_parquet(py_file, columns=keys)
                rs_keys = pd.read_parquet(rs_file, columns=keys)
                try:
                    pd.testing.assert_frame_equal(py_keys, rs_keys, check_dtype=False)
                except AssertionError:
                    ok = False
                    problems.append(f"{label}: same rows in a different order")
            else:
                lines = (result.stdout + result.stderr).strip().splitlines()
                problems.append(f"{label}: {lines[0] if lines else 'no output'}")
                print("\n  ".join([label, *lines[:8]]), flush=True)
            results[label] = ok

    passed = not problems and all(results.values())
    print(f"\n{'PASS' if passed else 'FAIL'}: {sum(results.values())}/{len(results)} frames")
    for problem in problems[:40]:
        print(f"  {problem}")
    print("  " + ", ".join(f"{k} {v:.0f}" for k, v in timings.items()))
    family_n = family_variants(args.experiments, FAMILY) + 1
    log_experiment(
        args.experiments,
        shlex.join(["python", "dev/features_parity.py", *sys.argv[1:]]),
        {"gate": "batch 3 features", "data": str(data), "binary": str(binary)},
        {
            "passed": passed,
            "deadlines": [d.isoformat() for d in picked],
            "frames": len(results),
            "failed": sorted(k for k, v in results.items() if not v),
            **{k: round(v, 1) for k, v in timings.items()},
        },
        family_n,
        FAMILY,
    )
    if args.keep is None:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
