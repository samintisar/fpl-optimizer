"""Rust migration batch 6, the xP models' parity gate: every model's xP frame from the Python
and the Rust code on the same `data/` (docs/superpowers/plans/2026-10-28-rust-migration.md,
Batch 6). Dev only: never imported by `fplopt`.

Deadlines: the leakage check's edge list plus `--n-random` (default 10) sampled ones
(`fplopt.features.leakcheck.sample_deadlines`, seed 0); or, with `--seasons 2025,2026`, every
deadline of those seasons up to the first one ahead of the data's newest snapshot (the plan's
gate 2: "every 2025/26 and 2026/27 deadline so far"; the holdout season is only run, never
scored). At each, Python computes each
`--models` frame (`fplopt.models.MODELS`) and writes it as Parquet; then `fplopt-rs models
dump` writes the same frames, and each pair is compared with `fplopt-rs check parity table`
on (player_key, horizon) plus the row order. Tolerances: the baselines are arithmetic on the
same rows, so rtol 1e-9; `v1` fits models, so the plan's gate applies (xP within an absolute
0.01). Components (`COMPONENTS`: v1's parts, fitted at the deadline itself) compare their
frame on its key and their fitted values, both within the plan's relative 1e-6.

`--plans` (with v1; the plan's gate 3): at each deadline the Rust planner plans the template
squad (`fplopt-rs optimize plan --start template --no-chips`, the bot's setting) on Python's
v1 frame and on Rust's; plan #1 must be the same (its lines, the solver's gap aside). Batch 4
showed the two planners agree on the same frame, so this isolates the xP's effect.

    cargo build --release -p fplopt-cli
    uv run python dev/models_parity.py [--data data] [--models rolling,ep_next,ep_next_fade]
        [--n-random 10 | --seasons 2025,2026] [--keep DIR]

Logs one row to results/experiments.csv (family `rust parity`) and exits 1 on any difference.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
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
from fplopt.models import MODELS, availability, components, minutes, shares, team  # noqa: E402

FAMILY = "rust parity"
QUIET = {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_ADMIN_CHAT_ID": "", "HEALTHCHECK_PING_URL": ""}
KEYS = ["player_key", "horizon"]
TOLERANCE = {"v1": ["--rtol", "0", "--atol", "0.01"]}  # others: the parity check's 1e-9
FIT_RTOL = 1e-6


def _values(fit, skip=("cutoff", "params")) -> dict:
    """A fit's dataclass fields as JSON values (`skip` left out)."""
    values = {k: v for k, v in dataclasses.asdict(fit).items() if k not in skip}
    return json.loads(json.dumps(values))


def _part(part) -> dict:
    if isinstance(part, minutes.Constant):
        return {"rate": part.rate}
    text = part.booster.model_to_string()
    return {"model": text[: text.rindex("\npandas_categorical:")]}  # the package's suffix


def _minutes_values(fit) -> dict:
    values = {k: _part(getattr(fit, k)) for k in ("start", "sixty", "sub")}
    names = (
        "minutes_sixty", "minutes_short", "minutes_sub", "n_rows", "ban_start", "ban_sub",
        "ban_start_likely", "ban_sub_likely",
    )  # fmt: skip
    return {**values, **{k: getattr(fit, k) for k in names}}


def _components_values(fit) -> dict:
    return json.loads(
        json.dumps({k: dataclasses.asdict(getattr(fit, k)) for k in ("bonus", "saves", "rates")})
    )


def _team(view):
    fit = team.fit_team(view)
    return team.team_lambdas(view, fit), _values(fit)


def _minutes(view):
    fit = minutes.fit_minutes(view)
    return minutes.predict_minutes(view, fit), _minutes_values(fit)


def _availability(view):
    history = training_frame(view)
    fit = minutes.fit_minutes(view, rows=history)
    layer = availability.fit_availability(view, fit, history, back_split=True)
    frame = availability.adjust_minutes(view, minutes.predict_minutes(view, fit), layer)
    return frame, _values(layer, ())


def _shares(view):
    lambdas = team.team_lambdas(view, team.fit_team(view))
    adjusted, _ = _availability(view)
    fit = shares.fit_shares(view)
    frame = shares.predict_shares(view, fit, adjusted, lambdas)
    return frame, _values(fit, ("cutoff", "params", "coverage"))


def _components(view):
    lambdas = team.team_lambdas(view, team.fit_team(view))
    adjusted, _ = _availability(view)
    fit = components.fit_components(view)
    frame = components.predict_components(view, fit, adjusted, lambdas, shares.fit_shares(view))
    return frame, _components_values(fit)


def _v1(view):
    """The model itself: fitted at the refit cutoff (gate 1: every component's values),
    predicted at the deadline (gate 2: the xP frame)."""
    model = MODELS["v1"]
    fit = model.fit(view.earlier(model.cutoff(view)))
    values = {
        "team": _values(fit.team),
        "minutes": _minutes_values(fit.minutes),
        "availability": _values(fit.availability, ()),
        "shares": _values(fit.shares, ("cutoff", "params", "coverage")),
        "components": _components_values(fit.components),
    }
    return model.predict(view, fit), values


# name -> (frame and fitted values at the view, the frame's key)
COMPONENTS = {
    "v1": (_v1, ["player_key", "horizon"]),
    "components": (_components, ["player_key", "horizon", "fixture_key"]),
    "shares": (_shares, ["player_key", "horizon", "fixture_key"]),
    "availability": (_availability, ["player_key", "horizon", "fixture_key"]),
    "team": (_team, ["team_key", "gw_index", "fixture_key"]),
    "minutes": (_minutes, ["player_key", "horizon", "fixture_key"]),
}
for _name in COMPONENTS:  # v1's frame keeps the plan's xP tolerance
    TOLERANCE.setdefault(_name, ["--rtol", str(FIT_RTOL), "--atol", "0"])


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


def plan_one(text: str) -> list[str]:
    """Plan #1's lines of an `optimize plan` output, the solver's gap left out."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("Plan #1:")), None)
    if start is None:
        return ["(no plan #1)", *lines[-3:]]
    block = []
    for line in lines[start:]:
        if not line.strip():
            break
        block.append(re.sub(r", gap [0-9.]+%", "", line))
    return block


def compare_plans(binary: Path, work: Path, picked: list, data: Path) -> dict[str, str]:
    """Per deadline: '' if plan #1 is the same on both v1 frames, else what differs."""
    gameweeks = pd.read_parquet(
        data / "gameweek.parquet", columns=["season", "gw", "deadline_time"]
    )
    out = {}
    for i, deadline in enumerate(picked):
        row = gameweeks[gameweeks["deadline_time"] == deadline].iloc[0]
        plans = {}
        for who in ("python", "rust"):
            frame = work / who / f"{i:02d}" / "v1.parquet"
            command = [
                str(binary), "optimize", "plan", "--season", str(int(row["season"])),
                "--gw", str(int(row["gw"])), "--xp", "v1", "--xp-file", str(frame),
                "--start", "template", "--no-chips",
            ]  # fmt: skip
            done = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
            (work / who / f"{i:02d}" / "plan.txt").write_text(done.stdout + done.stderr)
            plans[who] = plan_one(done.stdout + done.stderr)
        same = plans["python"] == plans["rust"]
        out[f"{i:02d} {deadline}"] = "" if same else f"{plans['python'][:2]} vs {plans['rust'][:2]}"
    return out


def season_deadlines(data: Path, seasons: list[int]) -> list[pd.Timestamp]:
    """Every deadline of `seasons` up to the first one after the newest player snapshot."""
    gameweeks = pd.read_parquet(data / "gameweek.parquet", columns=["season", "deadline_time"])
    newest = pd.read_parquet(data / "player_snapshot.parquet", columns=["snapshot_at"]).max()
    newest = newest.iloc[0]
    times = sorted(
        gameweeks.loc[gameweeks["season"].isin(seasons), "deadline_time"].dropna().unique()
    )
    past = [pd.Timestamp(t) for t in times if t <= newest]
    ahead = [pd.Timestamp(t) for t in times if t > newest]
    return past + ahead[:1]


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
    parser.add_argument("--seasons", help="every deadline of these seasons (e.g. 2025,2026)")
    parser.add_argument(
        "--plans", action="store_true", help="v1: compare plan #1 of the template on both frames"
    )
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

    if args.seasons:
        picked = season_deadlines(data, [int(x) for x in args.seasons.split(",") if x.strip()])
    else:
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

    plan_results: dict[str, str] = {}
    if args.plans and "v1" in models:
        started = time.monotonic()
        plan_results = compare_plans(binary, work, picked, data)
        timings["plans_s"] = time.monotonic() - started
        for label, difference in plan_results.items():
            results[f"{label} plan #1"] = not difference
            if difference:
                problems.append(f"{label} plan #1: {difference}")
        same = sum(not d for d in plan_results.values())
        print(f"plan #1 (template): {same}/{len(plan_results)} the same", flush=True)
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
