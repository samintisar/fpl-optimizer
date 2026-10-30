"""Rust migration batch 6, the xP models' parity gate: every model's xP frame from the Python
and the Rust code on the same `data/` (docs/superpowers/plans/2026-10-28-rust-migration.md,
Batch 6). Dev only: never imported by `fplopt`.

Deadlines: the leakage check's edge list plus `--n-random` (default 10) sampled ones
(`fplopt.features.leakcheck.sample_deadlines`, seed 0). At each, Python computes each
`--models` frame (`fplopt.models.MODELS`) and writes it as Parquet; then `fplopt-rs models
dump` writes the same frames, and each pair is compared with `fplopt-rs check parity table`
on (player_key, horizon) plus the row order. Tolerances: the baselines are arithmetic on the
same rows, so rtol 1e-9; `v1` fits models, so the plan's gate applies (xP within an absolute
0.01). Components (`COMPONENTS`: v1's parts, fitted at the deadline itself) compare their
frame on its key and their fitted values, both within the plan's relative 1e-6.

    cargo build --release -p fplopt-cli
    uv run python dev/models_parity.py [--data data] [--models rolling,ep_next,ep_next_fade]
        [--n-random 10] [--keep DIR]

Logs one row to results/experiments.csv (family `rust parity`) and exits 1 on any difference.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
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
from fplopt.features.history import training_frame  # noqa: E402
from fplopt.features.leakcheck import sample_deadlines  # noqa: E402
from fplopt.features.store import DataStore  # noqa: E402
from fplopt.models import MODELS, availability, minutes, shares, team  # noqa: E402

FAMILY = "rust parity"
QUIET = {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_ADMIN_CHAT_ID": "", "HEALTHCHECK_PING_URL": ""}
KEYS = ["player_key", "horizon"]
TOLERANCE = {"v1": ["--rtol", "0", "--atol", "0.01"]}  # others: the parity check's 1e-9
FIT_RTOL = 1e-6


def _team(view):
    fit = team.fit_team(view)
    values = {k: v for k, v in dataclasses.asdict(fit).items() if k not in ("cutoff", "params")}
    return team.team_lambdas(view, fit), values


def _part(part) -> dict:
    if isinstance(part, minutes.Constant):
        return {"rate": part.rate}
    text = part.booster.model_to_string()
    return {"model": text[: text.rindex("\npandas_categorical:")]}  # the package's suffix


def _minutes(view):
    fit = minutes.fit_minutes(view)
    values = {
        **{k: _part(getattr(fit, k)) for k in ("start", "sixty", "sub")},
        **{
            k: getattr(fit, k)
            for k in (
                "minutes_sixty", "minutes_short", "minutes_sub", "n_rows", "ban_start",
                "ban_sub", "ban_start_likely", "ban_sub_likely",
            )
        },
    }  # fmt: skip
    return minutes.predict_minutes(view, fit), values


def _availability(view):
    history = training_frame(view)
    fit = minutes.fit_minutes(view, rows=history)
    layer = availability.fit_availability(view, fit, history, back_split=True)
    frame = availability.adjust_minutes(view, minutes.predict_minutes(view, fit), layer)
    return frame, json.loads(json.dumps(dataclasses.asdict(layer)))


def _shares(view):
    lambdas = team.team_lambdas(view, team.fit_team(view))
    adjusted, _ = _availability(view)
    fit = shares.fit_shares(view)
    values = {
        k: v
        for k, v in dataclasses.asdict(fit).items()
        if k not in ("cutoff", "params", "coverage")
    }
    return shares.predict_shares(view, fit, adjusted, lambdas), json.loads(json.dumps(values))


# name -> (frame and fitted values at the view, the frame's key)
COMPONENTS = {
    "shares": (_shares, ["player_key", "horizon", "fixture_key"]),
    "availability": (_availability, ["player_key", "horizon", "fixture_key"]),
    "team": (_team, ["team_key", "gw_index", "fixture_key"]),
    "minutes": (_minutes, ["player_key", "horizon", "fixture_key"]),
}
for _name in COMPONENTS:
    TOLERANCE[_name] = ["--rtol", str(FIT_RTOL), "--atol", "0"]


def compare_fits(python: dict, rust: dict, where: str = "") -> list[str]:
    """Differences between two fits' values: numbers within FIT_RTOL, the rest equal."""
    problems = []
    for key in sorted(set(python) | set(rust)):
        a, b = python.get(key), rust.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            problems += compare_fits(a, b, f"{where}{key}.")
            continue
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b):
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                problems += compare_fits({"v": x}, {"v": y}, f"{where}{key}[{i}]")
            continue
        if isinstance(a, float) or isinstance(b, float):
            ok = (
                isinstance(a, (int, float))
                and isinstance(b, (int, float))
                and (math.isclose(a, b, rel_tol=FIT_RTOL, abs_tol=0) or a == b)
            )
        else:
            ok = a == b
        if not ok:
            problems.append(f"{where}{key if key != 'v' else ''}: python {a!r}, rust {b!r}")
    return problems


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
    parser.add_argument("--models", default="rolling,ep_next,ep_next_fade")
    parser.add_argument("--n-random", type=int, default=10)
    parser.add_argument("--keep", type=Path, help="keep the frames here")
    parser.add_argument("--experiments", type=Path, default=REPO / "results" / "experiments.csv")
    args = parser.parse_args()
    binary = args.bin if args.bin.exists() else args.bin.with_suffix(".exe")
    if not binary.exists():
        parser.error(f"no Rust binary at {args.bin} (cargo build --release -p fplopt-cli)")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in MODELS and m not in COMPONENTS]
    if unknown:
        parser.error(f"unknown models {unknown} (known: {', '.join([*MODELS, *COMPONENTS])})")
    data = args.data.resolve()
    work = Path(tempfile.mkdtemp(prefix="models-parity-")) if args.keep is None else args.keep
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
        for name in models:
            try:
                if name in COMPONENTS:
                    frame, fit = COMPONENTS[name][0](view)
                    text = json.dumps(fit)
                    (out / f"{name}.json").write_text(text, encoding="utf-8")
                else:
                    frame = MODELS[name](view)
                frame.to_parquet(out / f"{name}.parquet", index=False)
            except Exception as exc:  # a failure is a finding, reported below
                py_errors.append(f"{deadline} {name}: python {type(exc).__name__}: {exc}")
    timings["python_s"] = time.monotonic() - started
    print(f"python: {timings['python_s']:.0f} s", flush=True)

    started = time.monotonic()
    env = {**os.environ, **QUIET, "FPLOPT_DATA_DIR": str(data)}
    command = [str(binary), "models", "dump", "--models", ",".join(models)]
    command += ["--deadlines", str(listing), "--out", str(work / "rust")]
    with open(work / "rust.log", "w", encoding="utf-8") as log:
        rust = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    timings["rust_s"] = time.monotonic() - started
    print(f"rust: {timings['rust_s']:.0f} s", flush=True)

    problems = list(py_errors)
    if rust.returncode != 0:
        problems.append(f"fplopt-rs models dump failed (log: {work / 'rust.log'})")
    results: dict[str, bool] = {}
    for i, deadline in enumerate(picked):
        for name in models:
            label = f"{i:02d} {deadline} {name}"
            py_file = work / "python" / f"{i:02d}" / f"{name}.parquet"
            rs_file = work / "rust" / f"{i:02d}" / f"{name}.parquet"
            if not (py_file.exists() and rs_file.exists()):
                results[label] = False
                problems.append(f"{label}: missing ({'python' if rs_file.exists() else 'rust'})")
                continue
            keys = COMPONENTS[name][1] if name in COMPONENTS else KEYS
            if name in COMPONENTS:
                py_fit = json.loads((py_file.with_suffix(".json")).read_text(encoding="utf-8"))
                rs_fit = json.loads((rs_file.with_suffix(".json")).read_text(encoding="utf-8"))
                fit_problems = compare_fits(py_fit, rs_fit)
                if fit_problems:
                    problems.append(f"{label} fit: {fit_problems[0]} ({len(fit_problems)})")
                    results[f"{label} fit"] = False
                else:
                    results[f"{label} fit"] = True
            command = [
                str(binary), "check", "parity", "table",
                "--python", str(py_file), "--rust", str(rs_file), "--keys", ",".join(keys),
                *TOLERANCE.get(name, []),
                "--out", str(work / "parity" / f"{i:02d}" / name),
                "--experiments", str(work / "frames.csv"),
            ]  # fmt: skip
            result = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
            ok = result.returncode == 0
            if ok:
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
        shlex.join(["python", "dev/models_parity.py", *sys.argv[1:]]),
        {"gate": "batch 6 models", "models": models, "data": str(data), "binary": str(binary)},
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
