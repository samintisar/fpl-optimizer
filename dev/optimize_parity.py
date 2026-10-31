"""Rust migration batch 4, the parity gate: the Rust optimizer must make the plans Python's
makes (docs/superpowers/plans/2026-10-28-rust-migration.md, Batch 4). Dev only: never
imported by `fplopt`.

A corpus of planner cases at real deadlines (seeded): each case is a squad state, the
deadline's pool, an xP frame (`ep_next`, and `v1` at some deadlines) and planner settings
(defaults, a few with hits allowed), solved with and without chips. Squads: the template,
random squads (`random:SEED`), and "carried" squads: the template of an earlier GW held to
this deadline (stale prices, maybe departed players and clubs over the cap), some with more
free transfers, a lower purchase price (sell-on fee) and chips already used.

Each case is written once (case.json, pool.parquet, xp.parquet) and solved from those files
by both: Python (`PlanInput.from_context` + `optimize`, in worker processes) and
`fplopt-rs optimize solve`. Checks per case:

- the problem: the same candidates, horizon GWs and minutes-based bench weights;
- plan #1: total objective within 1e-6, and the same plan (every GW's transfers, lineup,
  captain, vice, chip, hits, bank, FTs) unless Python's plan #1 ties its runner-up (plan #2
  or the next chip scenario) within 1e-6, where ties may break differently;
- plans #2..k and the roll plan: the same, under the same tie rule;
- every Rust plan is valid under the Python rule checker: each GW replayed with
  `apply_decision` and `next_state`, and its hits, bank and FTs equal the replay's.

    cargo build --release -p fplopt-cli
    uv run python dev/optimize_parity.py [--data data] [--deadlines 12] [--seed 0]
        [--v1 4] [--jobs N] [--keep DIR]

Logs one row to results/experiments.csv (family `rust parity`) and exits 1 on any failure.
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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fplopt.backtest.evaluate import family_variants, log_experiment  # noqa: E402
from fplopt.backtest.gw_score import Lineup  # noqa: E402
from fplopt.backtest.rules import backtest_rules  # noqa: E402
from fplopt.backtest.simulator import FAR_FUTURE, season_schedule  # noqa: E402
from fplopt.backtest.start_states import (  # noqa: E402
    StartStateError,
    random_state,
    template_state,
)
from fplopt.backtest.state import (  # noqa: E402
    POOL_COLUMNS,
    Decision,
    Holding,
    InvalidDecision,
    SquadState,
    Transfer,
    apply_decision,
    chip_window,
    next_state,
)
from fplopt.features.baseline import player_pool  # noqa: E402
from fplopt.features.store import DataStore  # noqa: E402

FAMILY = "rust parity"
QUIET = {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_ADMIN_CHAT_ID": "", "HEALTHCHECK_PING_URL": ""}
OBJECTIVE_TOL = 1e-6
WEIGHT_TOL = 1e-12
# Deadline pool: develop/validate seasons with ep_next (2021/22+), the holdout and the live
# season (a code check, migration plan §6).
SEASONS = (2021, 2022, 2023, 2024, 2025, 2026)
VARIANTS = (
    {},
    {"max_hits": 1, "hit_margin": 2.0},
    {"max_hits": None},
)


# --- the corpus -----------------------------------------------------------------------------


def _deadline(store: DataStore, season: int, gw_index: int) -> pd.Timestamp:
    return season_schedule(store, season, gw_index)["deadline_time"].iloc[0]


def pick_deadlines(store: DataStore, n: int, seed: int) -> list[tuple[int, int, int]]:
    """(season, gw, gw_index): every season's GW1 and GW2 deadline once among the first
    picks, then seeded random deadlines whose GW has a later GW in the data (a horizon)."""
    gameweeks = store.as_of(FAR_FUTURE).table("gameweek", columns=["season", "gw", "gw_index"])
    gameweeks = gameweeks[gameweeks["season"].isin(SEASONS)].drop_duplicates(["season", "gw"])
    rng = np.random.default_rng(seed)
    fixed = [(2022, 1), (2024, 2), (2026, 1)]
    out = []
    for season, gw in fixed:
        row = gameweeks[(gameweeks["season"] == season) & (gameweeks["gw"] == gw)]
        if len(row):
            out.append((season, gw, int(row["gw_index"].iloc[0])))
    pool = gameweeks[gameweeks["gw"] > 2].reset_index(drop=True)
    last = gameweeks.groupby("season")["gw_index"].max()
    pool = pool[pool["gw_index"] < pool["season"].map(last)].reset_index(drop=True)
    for i in rng.permutation(len(pool)):
        if len(out) >= n:
            break
        row = pool.iloc[int(i)]
        pick = (int(row["season"]), int(row["gw"]), int(row["gw_index"]))
        if pick not in out:
            out.append(pick)
    return sorted(out[:n])


def carried_state(store, season, gw_index, rules, pool, variant: int) -> SquadState | None:
    """The template squad of 4-8 GWs earlier held to this deadline (purchase prices from
    then), with more FTs, a lower purchase price for two players and chips used for the
    odd variants."""
    back = 4 + 2 * variant
    if gw_index - back < 2:
        return None
    try:
        earlier = template_state(store.as_of(_deadline(store, season, gw_index - back)), rules)
    except (StartStateError, ValueError):
        return None
    holdings = list(earlier.holdings)
    if variant % 2 == 1:
        holdings[0] = dataclasses.replace(holdings[0], purchase_price=holdings[0].price - 3)
        holdings[5] = dataclasses.replace(holdings[5], purchase_price=holdings[5].price - 2)
    state = SquadState(
        season=season,
        gw_index=gw_index,
        holdings=tuple(holdings),
        bank=earlier.bank,
        free_transfers=min(2 + variant, rules.max_free_transfers),
    )
    if variant % 2 == 1:
        used = []
        probe = state
        for name, at in (("wildcard", gw_index - back), ("3xc", gw_index - 1)):
            try:
                cid = chip_window(dataclasses.replace(probe, gw_index=at), name, rules)
            except InvalidDecision:
                continue
            used.append((cid, at))
            probe = dataclasses.replace(probe, chips_used=tuple(used))
        state = dataclasses.replace(state, chips_used=tuple(used))
    return state


def state_json(state: SquadState) -> dict:
    return {
        "season": state.season,
        "gw_index": state.gw_index,
        "holdings": [
            [h.player_key, h.element_type, h.team_key, h.purchase_price, h.price]
            for h in state.holdings
        ],
        "bank": state.bank,
        "free_transfers": state.free_transfers,
        "chips_used": [list(c) for c in state.chips_used],
    }


def build_corpus(data: Path, work: Path, n_deadlines: int, n_v1: int, seed: int) -> list[Path]:
    from fplopt.models import MODELS

    store = DataStore(data)
    picks = pick_deadlines(store, n_deadlines, seed)
    v1_at = set(picks[:: max(len(picks) // max(n_v1, 1), 1)][:n_v1]) if n_v1 else set()
    cases = []
    for d, (season, gw, gw_index) in enumerate(picks):
        started = time.monotonic()
        view = store.as_of(_deadline(store, season, gw_index))
        rules = backtest_rules(season)
        pool = player_pool(view)[list(POOL_COLUMNS)].reset_index(drop=True)
        starts: dict[str, SquadState] = {}
        for name, seed in (("template", None), ("random:1", 1), ("random:2", 2)):
            try:
                if seed is None:
                    starts[name] = template_state(view, rules)
                else:
                    starts[name] = random_state(view, rules, seed)
            except StartStateError as exc:
                print(f"  {season} GW{gw}: no {name} start ({exc})", flush=True)
        for variant in (0, 1):
            state = carried_state(store, season, gw_index, rules, pool, variant)
            if state is not None:
                starts[f"carried:{variant}"] = state
        frames = {"ep_next": MODELS["ep_next"](view)}
        if (season, gw, gw_index) in v1_at:
            frames["v1"] = MODELS["v1"](view)
        for xp_name, frame in frames.items():
            frame_file = work / "frames" / f"{d:02d}-{xp_name}.parquet"
            frame_file.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(frame_file, index=False)
            for s, (start, state) in enumerate(starts.items()):
                for chips in (False, True):
                    # Chips on every other start (and every v1 start), settings variants
                    # rotating over the no-chip cases.
                    if chips and xp_name == "ep_next" and s % 2 == 1:
                        continue
                    params = {} if chips else VARIANTS[(d + s) % len(VARIANTS)]
                    case = work / "cases" / f"{len(cases):03d}"
                    case.mkdir(parents=True, exist_ok=True)
                    pool.to_parquet(case / "pool.parquet", index=False)
                    shutil.copyfile(frame_file, case / "xp.parquet")
                    meta = {
                        "season": season,
                        "gw": gw,
                        "label": f"{season} GW{gw} {start} {xp_name} "
                        f"{'chips' if chips else 'no chips'} {params or 'defaults'}",
                        "state": state_json(state),
                        "params": params,
                        "top_k": 3,
                        "chips": chips,
                    }
                    (case / "case.json").write_text(json.dumps(meta), encoding="utf-8")
                    cases.append(case)
        print(
            f"{season} GW{gw}: {len(starts)} starts, {list(frames)}, "
            f"{time.monotonic() - started:.0f} s ({len(cases)} cases so far)",
            flush=True,
        )
    return cases


# --- solving --------------------------------------------------------------------------------


def read_case(case: Path):
    from fplopt.optimize import OptimizerParams

    meta = json.loads((case / "case.json").read_text(encoding="utf-8"))
    s = meta["state"]
    state = SquadState(
        season=s["season"],
        gw_index=s["gw_index"],
        holdings=tuple(Holding(*h) for h in s["holdings"]),
        bank=s["bank"],
        free_transfers=s["free_transfers"],
        chips_used=tuple(tuple(c) for c in s["chips_used"]),
    )
    pool = pd.read_parquet(case / "pool.parquet")
    xp = pd.read_parquet(case / "xp.parquet")
    return meta, state, pool, xp, OptimizerParams(**meta["params"])


def plan_json(plan) -> dict:
    return {
        "scenario": plan.scenario,
        "objective": plan.objective,
        "terminal_value": plan.terminal_value,
        "total_objective": plan.total_objective,
        "status": plan.status,
        "mip_gap": plan.mip_gap,
        "n_nodes": plan.n_nodes,
        "bound": plan.bound if math.isfinite(plan.bound) else "inf",
        "gain_vs_roll": plan.gain_vs_roll,
        "n_candidates": plan.n_candidates,
        "gws": [
            {
                "gw": g.gw,
                "gw_index": g.gw_index,
                "horizon": g.horizon,
                "transfers": [[t.out_key, t.in_key] for t in g.transfers],
                "starters": list(g.starters),
                "bench": list(g.bench),
                "captain": g.captain,
                "vice": g.vice,
                "chip": g.chip,
                "xp": g.xp,
                "bench_xp": g.bench_xp,
                "objective": g.objective,
                "hits": g.hits,
                "bank": g.bank,
                "ft": g.ft,
            }
            for g in plan.gws
        ],
    }


def solve_python(case: Path) -> float:
    """Python's plans for a case, written to python.json; returns the seconds taken."""
    from fplopt.optimize import PlanInput, optimize

    started = time.monotonic()
    meta, state, pool, xp, params = read_case(case)
    rules = backtest_rules(meta["season"])
    try:
        problem = PlanInput.from_context(state, pool, xp, rules, params)
        plans = optimize(problem, params, top_k=meta["top_k"], chips=meta["chips"], roll=True)
        out = {
            "problem": {
                "candidates": [p.player_key for p in problem.players],
                "gws": [[g.gw, g.gw_index, g.horizon] for g in problem.gws],
                "bench_weights": None
                if problem.bench_weights is None
                else [list(w) for w in problem.bench_weights],
                "n_pool": problem.n_pool,
            },
            "plans": [plan_json(p) for p in plans.plans],
            "roll": None if plans.roll is None else plan_json(plans.roll),
            "scenario_objectives": [[k, v] for k, v in plans.scenario_objectives.items()],
            "solver_status": plans.solver_status,
        }
    except Exception as exc:  # a failure is a finding, compared below
        out = {"error": f"{type(exc).__name__}: {exc}"}
    out["seconds"] = time.monotonic() - started
    (case / "python.json").write_text(json.dumps(out), encoding="utf-8")
    return out["seconds"]


def solve_rust(binary: Path, case: Path) -> float:
    started = time.monotonic()
    env = {**os.environ, **QUIET, "RUST_LOG": "error"}
    command = [str(binary), "optimize", "solve", "--case", str(case), "--out"]
    command.append(str(case / "rust.json"))
    result = subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        error = (result.stderr or result.stdout).strip().splitlines()
        (case / "rust.json").write_text(
            json.dumps({"error": error[-1] if error else f"exit {result.returncode}"}),
            encoding="utf-8",
        )
    return time.monotonic() - started


# --- checks -----------------------------------------------------------------------------------


PLAN_FIELDS = ("transfers", "starters", "bench", "captain", "vice", "chip", "hits", "bank", "ft")


def same_plan(a: dict, b: dict) -> bool:
    if a["scenario"] != b["scenario"] or len(a["gws"]) != len(b["gws"]):
        return False
    for ga, gb in zip(a["gws"], b["gws"], strict=True):
        if any(ga[f] != gb[f] for f in PLAN_FIELDS):
            return False
        if ga["transfers"] and sorted(map(tuple, ga["transfers"])) != sorted(
            map(tuple, gb["transfers"])
        ):
            return False
    return True


def runner_up_gap(py: dict, i: int) -> float:
    """How far Python's plan #i+1 is from the next-best alternative it could tie with: the
    next plan, and for plan #1 also the best other chip scenario."""
    total = py["plans"][i]["total_objective"]
    others = [p["total_objective"] for p in py["plans"][i + 1 : i + 2]]
    if i == 0:
        best = py["plans"][0]["scenario"]
        others += [v for k, v in py["scenario_objectives"] if k != best]
    return min((abs(total - o) for o in others), default=math.inf)


def validate(case: Path, plan: dict, state: SquadState, pool: pd.DataFrame, rules) -> str | None:
    """Replay a plan with the Python rule checker; None if valid, else the problem."""
    for i, g in enumerate(plan["gws"]):
        if state.gw_index != g["gw_index"]:
            return f"GW{g['gw']}: state at gw_index {state.gw_index}"
        decision = Decision(
            transfers=tuple(Transfer(o, n) for o, n in g["transfers"]),
            lineup=Lineup(tuple(g["starters"]), tuple(g["bench"]), g["captain"], g["vice"]),
            chip=g["chip"],
        )
        try:
            gw_state, record = apply_decision(state, decision, pool, rules)
        except InvalidDecision as exc:
            return f"GW{g['gw']}: {exc}"
        replay = {"hits": record.hits, "bank": gw_state.bank, "ft": state.free_transfers}
        planned = {k: g[k] for k in replay}
        if replay != planned:
            return f"GW{g['gw']}: plan {planned} != rules {replay}"
        if i + 1 < len(plan["gws"]):
            state = next_state(gw_state, record, rules, plan["gws"][i + 1]["gw_index"])
    return None


def check(case: Path) -> dict:
    py = json.loads((case / "python.json").read_text(encoding="utf-8"))
    rs = json.loads((case / "rust.json").read_text(encoding="utf-8"))
    meta, state, pool, _, _ = read_case(case)
    out = {"case": case.name, "label": meta["label"], "problems": [], "ties": 0, "same": 0}
    out["seconds"] = (py.get("seconds", 0.0), rs.get("seconds", 0.0))
    if "error" in py or "error" in rs:
        if not ("error" in py and "error" in rs):
            out["problems"].append(f"error: python {py.get('error')!r} rust {rs.get('error')!r}")
        return out
    pp, rp = py["problem"], rs["problem"]
    if pp["candidates"] != rp["candidates"]:
        only_py = sorted(set(pp["candidates"]) - set(rp["candidates"]))[:5]
        only_rs = sorted(set(rp["candidates"]) - set(pp["candidates"]))[:5]
        out["problems"].append(f"candidates differ (python only {only_py}, rust only {only_rs})")
    if pp["gws"] != rp["gws"]:
        out["problems"].append(f"horizon {pp['gws']} != {rp['gws']}")
    if (pp["bench_weights"] is None) != (rp["bench_weights"] is None):
        out["problems"].append("bench weights: one side has none")
    elif pp["bench_weights"] is not None:
        diff = float(np.max(np.abs(np.array(pp["bench_weights"]) - np.array(rp["bench_weights"]))))
        out["weight_diff"] = diff
        if diff > WEIGHT_TOL:
            out["problems"].append(f"bench weights differ by {diff:.2e}")
    rules = backtest_rules(meta["season"])
    labelled = [(f"plan #{i + 1}", i) for i in range(len(py["plans"]))]
    if len(py["plans"]) != len(rs["plans"]):
        out["problems"].append(f"{len(py['plans'])} python plans, {len(rs['plans'])} rust")
    out["nodes_equal"] = all(
        a["n_nodes"] == b["n_nodes"] for a, b in zip(py["plans"], rs["plans"], strict=False)
    )
    for label, i in labelled:
        if i >= len(rs["plans"]):
            break
        a, b = py["plans"][i], rs["plans"][i]
        gap = abs(a["total_objective"] - b["total_objective"])
        if gap > OBJECTIVE_TOL:
            out["problems"].append(
                f"{label}: objective {a['total_objective']:.6f} != {b['total_objective']:.6f}"
            )
        if same_plan(a, b):
            out["same"] += 1
        elif runner_up_gap(py, i) <= OBJECTIVE_TOL:
            out["ties"] += 1
        else:
            scenarios = f"{a['scenario']} vs {b['scenario']}"
            out["problems"].append(f"{label}: a different plan ({scenarios})")
    if (py["roll"] is None) != (rs["roll"] is None):
        out["problems"].append("roll plan: one side has none")
    elif py["roll"] is not None:
        a, b = py["roll"], rs["roll"]
        if abs(a["total_objective"] - b["total_objective"]) > OBJECTIVE_TOL:
            out["problems"].append(
                f"roll plan: objective {a['total_objective']:.6f} != {b['total_objective']:.6f}"
            )
        if same_plan(a, b):
            out["same"] += 1
        else:
            out["problems"].append("roll plan: a different plan")
    py_scen = dict(map(tuple, py["scenario_objectives"]))
    rs_scen = dict(map(tuple, rs["scenario_objectives"]))
    if set(py_scen) != set(rs_scen):
        out["problems"].append(
            f"solved chip scenarios differ ({len(py_scen)} python, {len(rs_scen)} rust)"
        )
    for plan in [*rs["plans"], *([rs["roll"]] if rs["roll"] else [])]:
        problem = validate(case, plan, state, pool, rules)
        if problem:
            out["problems"].append(f"rust plan {plan['scenario']} invalid: {problem}")
    return out


# --- main -------------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=REPO / "data", help="data/ (read only)")
    parser.add_argument("--bin", type=Path, default=REPO / "target" / "release" / "fplopt-rs")
    parser.add_argument("--deadlines", type=int, default=12)
    parser.add_argument("--v1", type=int, default=4, help="deadlines with a v1 frame too")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=max((os.cpu_count() or 2) - 2, 1))
    parser.add_argument("--keep", type=Path, help="keep the cases here (or reuse them)")
    parser.add_argument("--experiments", type=Path, default=REPO / "results" / "experiments.csv")
    args = parser.parse_args()
    binary = args.bin if args.bin.exists() else args.bin.with_suffix(".exe")
    if not binary.exists():
        parser.error(f"no Rust binary at {args.bin} (cargo build --release -p fplopt-cli)")
    os.environ.update(QUIET)
    data = args.data.resolve()
    work = Path(tempfile.mkdtemp(prefix="optimize-parity-")) if args.keep is None else args.keep
    work.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}
    started = time.monotonic()
    cases = sorted((work / "cases").glob("*")) if (work / "cases").exists() else []
    if not cases:
        cases = build_corpus(data, work, args.deadlines, args.v1, args.seed)
    timings["corpus_s"] = time.monotonic() - started
    print(f"{len(cases)} cases ({timings['corpus_s']:.0f} s)", flush=True)

    started = time.monotonic()
    with ProcessPoolExecutor(args.jobs) as pool:
        py_seconds = list(pool.map(solve_python, cases))
    timings["python_s"] = time.monotonic() - started
    print(f"python: {timings['python_s']:.0f} s", flush=True)
    started = time.monotonic()
    with ThreadPoolExecutor(args.jobs) as pool:
        rs_seconds = list(pool.map(lambda c: solve_rust(binary, c), cases))
    timings["rust_s"] = time.monotonic() - started
    print(f"rust: {timings['rust_s']:.0f} s", flush=True)
    timings["python_solve_s"] = sum(py_seconds)
    timings["rust_solve_s"] = sum(rs_seconds)

    results = [check(case) for case in cases]
    failed = [r for r in results if r["problems"]]
    plans = sum(r["same"] + r["ties"] for r in results)
    ties = sum(r["ties"] for r in results)
    nodes = sum(bool(r.get("nodes_equal")) for r in results)
    weight_diff = max((r.get("weight_diff", 0.0) for r in results), default=0.0)
    passed = not failed
    print(f"\n{'PASS' if passed else 'FAIL'}: {len(results) - len(failed)}/{len(results)} cases")
    print(
        f"  {plans} plans matched ({ties} as ties), node counts equal in {nodes} cases, "
        f"bench weights max diff {weight_diff:.1e}"
    )
    for r in failed[:30]:
        print(f"  {r['case']} {r['label']}:")
        for problem in r["problems"][:5]:
            print(f"    {problem}")
    print("  " + ", ".join(f"{k} {v:.0f}" for k, v in timings.items()))
    family_n = family_variants(args.experiments, FAMILY) + 1
    log_experiment(
        args.experiments,
        shlex.join(["python", "dev/optimize_parity.py", *sys.argv[1:]]),
        {"gate": "batch 4 optimizer", "data": str(data), "binary": str(binary), "seed": args.seed},
        {
            "passed": passed,
            "cases": len(results),
            "plans_matched": plans,
            "ties": ties,
            "nodes_equal": nodes,
            "bench_weight_max_diff": weight_diff,
            "failed": [f"{r['case']} {r['label']}: {r['problems'][0]}" for r in failed],
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
