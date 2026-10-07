"""Optimizer benchmark on real data (Phase 4 plan, Task 3; issue #10): solve times, gaps,
the chip scenario search and the pruning defaults.

`run_bench(store, ...)` samples `deadlines` (season, gw_index) pairs from 2021/22-2024/25 and
2026/27 (never the holdout; the last season only up to now), builds a template and a
seeded random start state at each (`start_states`, no chips used: every chip scenario is
open) and plans with both xP models (`ep_next`, `rolling`). Per case (deadline × start ×
xP) it records:

- `nochip`: input build (`PlanInput.from_context`), model build and HiGHS solve time,
  final gap and status of the default no-chip solve (`OptimizerParams()`);
- `top3`: `optimize(top_k=3, chips=False)` incl. the roll plan, wall time;
- `chips`: `optimize(top_k=1, chips=True, roll=False)` with the bound search: wall time,
  scenarios, LP relaxations, MILPs, best scenario and objective; on the first
  `all_chips` cases also the exhaustive search (`scenario_search="all"`), checked equal,
  and the chip search at a tight gap with the default pool vs dominance pruning only;
- `prune`: the no-chip solve at a tight gap (`PRUNE_GAP`) per pruning variant
  (`PRUNE_VARIANTS`, from no pruning at all to small N), with candidates, solve time and
  the objective loss vs the best variant (the unpruned pool is the reference: same
  objective, more candidates).

The CLI (`fplopt optimize bench`) prints `summary_text` and writes `cases.csv`,
`prune.csv` and `summary.json`. Reads data only through the store's views.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from fplopt.backtest.rules import backtest_rules
from fplopt.backtest.simulator import FAR_FUTURE, Caches, season_schedule
from fplopt.backtest.start_states import StartStateError, random_state, template_state
from fplopt.features.store import DataStore
from fplopt.optimize.model import solve_plan
from fplopt.optimize.params import OptimizerParams
from fplopt.optimize.plans import optimize
from fplopt.optimize.problem import PlanInput
from fplopt.seasons import HOLDOUT_SEASONS, season_label

log = logging.getLogger(__name__)

BENCH_SEASONS = (2021, 2022, 2023, 2024, 2026)  # ep_next exists from 2021/22; 2025 = holdout
XP_MODELS = ("ep_next", "rolling")
PRUNE_GAP = 1e-4  # tight, so pruning losses aren't hidden by the MIP gap
# Pruning variants: OptimizerParams overrides. "none" (every pool player) is the reference.
PRUNE_VARIANTS: Mapping[str, dict[str, Any]] = {
    "none": {"prune_n": None, "prune_dominated": False},
    "dominance": {"prune_n": None},
    "default": {},  # 20/60/60/30 + dominance
    "n10_30": {"prune_n": {1: 10, 2: 30, 3: 30, 4: 15}},
    "n5_15": {"prune_n": {1: 5, 2: 15, 3: 15, 4: 8}},
}


@dataclass(frozen=True)
class BenchCase:
    """One benchmark instance: the decision at (season, gw_index) from a start state built
    there (`template` or `random:<seed>`), planned with xP model `xp`."""

    season: int
    gw_index: int
    start: str
    xp: str

    @property
    def name(self) -> str:
        return f"{season_label(self.season)} gw{self.gw_index} {self.start} {self.xp}"


def bench_deadlines(
    store: DataStore,
    n: int,
    seed: int,
    seasons: Iterable[int] = BENCH_SEASONS,
    now: pd.Timestamp | None = None,
) -> list[tuple[int, int]]:
    """`n` distinct (season, gw_index) pairs, round-robin over the seasons that have
    gameweeks (holdout excluded), gw_index drawn uniformly from the season's GWs whose
    deadline has passed (`now`, default the current time). Sorted."""
    if n < 1:
        raise ValueError(f"need at least one deadline, got {n}")
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    gameweeks = store.as_of(FAR_FUTURE).table(
        "gameweek", columns=["season", "gw_index", "deadline_time"]
    )
    gameweeks = gameweeks[gameweeks["deadline_time"] < now]
    wanted = [s for s in sorted(set(seasons)) if s not in HOLDOUT_SEASONS]
    pools = {
        s: sorted(set(gameweeks.loc[gameweeks["season"] == s, "gw_index"].tolist())) for s in wanted
    }
    pools = {s: gws for s, gws in pools.items() if gws}
    if not pools:
        raise ValueError(f"no gameweeks for seasons {wanted}")
    rng = np.random.default_rng(seed)
    order = list(pools)
    picked: list[tuple[int, int]] = []
    while len(picked) < n and any(pools.values()):
        for season in order:
            if len(picked) == n:
                break
            if pools[season]:
                gw = pools[season].pop(int(rng.integers(len(pools[season]))))
                picked.append((season, int(gw)))
    return sorted(picked)


def build_case(store: DataStore, caches: Caches, case: BenchCase) -> tuple:
    """(state, pool, xp, rules) for a case; StartStateError if no start state there."""
    rules = backtest_rules(case.season)
    schedule = season_schedule(store, case.season, case.gw_index)
    view = store.as_of(schedule["deadline_time"].iloc[0])
    if case.start == "template":
        state = template_state(view, rules)
    else:
        state = random_state(view, rules, int(case.start.split(":")[1]))
    return state, caches.pool(store, view), caches.xp(store, case.xp, view), rules


def _timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - start


def bench_case(
    store: DataStore,
    caches: Caches,
    case: BenchCase,
    params: OptimizerParams,
    *,
    chips_all: bool,
    prune_study: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The case's row of measurements and its pruning rows (module docstring)."""
    state, pool, xp, rules = build_case(store, caches, case)
    row: dict[str, Any] = {
        "case": case.name,
        "season": case.season,
        "gw_index": case.gw_index,
        "start": case.start,
        "xp": case.xp,
    }
    problem, input_s = _timed(lambda: PlanInput.from_context(state, pool, xp, rules, params))
    plan = solve_plan(problem, params)
    row |= {
        "horizon": len(problem.gws),
        "n_pool": problem.n_pool,
        "n_candidates": len(problem.players),
        "input_s": input_s,
        "build_s": plan.build_seconds,
        "solve_s": plan.solve_seconds,
        "gap": plan.mip_gap,
        "status": plan.status,
        "objective": plan.total_objective,
    }
    top3 = optimize(problem, params, top_k=3, chips=False, roll=True)
    row |= {"top3_s": top3.seconds, "top3_plans": len(top3.plans)}
    chips = optimize(problem, params, top_k=1, chips=True, roll=False)
    stats = chips.search
    row |= {
        "chips_s": chips.seconds,
        "n_scenarios": stats.n_scenarios,
        "n_relaxations": stats.n_relaxations,
        "n_solves": stats.n_solves,
        "relax_s": stats.relax_seconds,
        "chip_solve_s": stats.solve_seconds,
        "chips_scenario": chips.best.scenario,
        "chips_objective": chips.best.total_objective,
    }
    if chips_all:
        full = optimize(problem, params, top_k=1, chips=True, roll=False, scenario_search="all")
        row |= {
            "chips_all_s": full.seconds,
            "chips_all_scenario": full.best.scenario,
            "chips_all_objective": full.best.total_objective,
            "bound_matches_all": full.best.scenario == chips.best.scenario
            and full.best.total_objective == chips.best.total_objective,
        }
        # Pruning with chips (a Free Hit wants the best players of one GW, not of the
        # horizon): the default pool vs dominance pruning only, both at the tight gap.
        for variant in ("default", "dominance"):
            p = replace(params, mip_gap=PRUNE_GAP, **PRUNE_VARIANTS[variant])
            prob = PlanInput.from_context(state, pool, xp, rules, p)
            res = optimize(prob, p, top_k=1, chips=True, roll=False)
            row[f"chips_tight_{variant}_s"] = res.seconds
            row[f"chips_tight_{variant}_objective"] = res.best.total_objective
        row["chips_prune_loss"] = (
            row["chips_tight_dominance_objective"] - row["chips_tight_default_objective"]
        )
    prune_rows: list[dict[str, Any]] = []
    if prune_study:
        for variant, overrides in PRUNE_VARIANTS.items():
            p = replace(params, mip_gap=PRUNE_GAP, **overrides)
            prob, input_s = _timed(lambda p=p: PlanInput.from_context(state, pool, xp, rules, p))
            sol = solve_plan(prob, p)
            prune_rows.append(
                {
                    "case": case.name,
                    "variant": variant,
                    "n_candidates": len(prob.players),
                    "input_s": input_s,
                    "build_s": sol.build_seconds,
                    "solve_s": sol.solve_seconds,
                    "gap": sol.mip_gap,
                    "status": sol.status,
                    "objective": sol.total_objective,
                    "bound": sol.bound,
                }
            )
        best = max(r["objective"] for r in prune_rows)
        for r in prune_rows:
            r["loss"] = best - r["objective"]
    return row, prune_rows


@dataclass(frozen=True)
class BenchResult:
    cases: pd.DataFrame
    prune: pd.DataFrame
    config: dict[str, Any]


def bench_cases(store: DataStore, deadlines: int, seed: int) -> list[BenchCase]:
    """Template and random:<seed + i> starts at each sampled deadline, × both xP models."""
    out = []
    for i, (season, gw_index) in enumerate(bench_deadlines(store, deadlines, seed)):
        for start in ("template", f"random:{seed + i}"):
            out += [BenchCase(season, gw_index, start, xp) for xp in XP_MODELS]
    return out


def run_bench(
    store: DataStore,
    *,
    deadlines: int = 10,
    seed: int = 0,
    params: OptimizerParams | None = None,
    all_chips: int = 0,
    prune_study: bool = True,
    progress: Callable[[str], None] = log.info,
) -> BenchResult:
    """Run the benchmark (module docstring). `all_chips`: on how many cases (the first)
    to also run the exhaustive chip search; `prune_study=False` skips the pruning
    variants."""
    params = OptimizerParams() if params is None else params
    caches = Caches()
    rows, prune_rows = [], []
    cases = bench_cases(store, deadlines, seed)
    for k, case in enumerate(cases):
        try:
            row, prows = bench_case(
                store,
                caches,
                case,
                params,
                chips_all=len(rows) < all_chips,
                prune_study=prune_study,
            )
        except StartStateError as exc:
            progress(f"[{k + 1}/{len(cases)}] {case.name}: skipped ({exc})")
            continue
        rows.append(row)
        prune_rows += prows
        progress(
            f"[{k + 1}/{len(cases)}] {case.name}: no-chip {row['solve_s']:.2f}s, "
            f"chips {row['chips_s']:.1f}s ({row['n_solves']}/{row['n_scenarios']} solved), "
            f"top-3 {row['top3_s']:.1f}s"
            + (
                f"; exhaustive {row['chips_all_s']:.0f}s, same plan: {row['bound_matches_all']}"
                if "chips_all_s" in row
                else ""
            )
        )
    if not rows:
        raise RuntimeError("no benchmark case could be built")
    config = {
        "deadlines": deadlines,
        "seed": seed,
        "all_chips": all_chips,
        "prune_study": prune_study,
        "prune_gap": PRUNE_GAP,
        "params": {
            "horizon": params.horizon,
            "decay": params.decay,
            "mip_gap": params.mip_gap,
            "threads": params.threads,
            "prune_n": None if params.prune_n is None else dict(params.prune_n),
            "prune_dominated": params.prune_dominated,
        },
        "prune_variants": {k: _plain(v) for k, v in PRUNE_VARIANTS.items()},
    }
    return BenchResult(pd.DataFrame(rows), pd.DataFrame(prune_rows), config)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    return value


def _stats(values: Sequence[float] | pd.Series) -> dict[str, float]:
    s = pd.Series(values, dtype="float64").dropna()
    if s.empty:
        return {"median": float("nan"), "p90": float("nan"), "max": float("nan")}
    return {
        "median": float(s.median()),
        "p90": float(s.quantile(0.9)),
        "max": float(s.max()),
    }


def summarize(result: BenchResult) -> dict[str, Any]:
    """Aggregates: timing stats per measurement, the bound-vs-all check and per pruning
    variant candidates, solve time and loss."""
    cases = result.cases
    timings = {
        column: _stats(cases[column])
        for column in (
            "input_s",
            "build_s",
            "solve_s",
            "gap",
            "top3_s",
            "chips_s",
            "relax_s",
            "chip_solve_s",
            "n_scenarios",
            "n_relaxations",
            "n_solves",
            "n_candidates",
        )
    }
    out: dict[str, Any] = {"n_cases": len(cases), "timings": timings}
    if "chips_all_s" in cases:
        done = cases.dropna(subset=["chips_all_s"])
        out["chips_all"] = {
            "n_cases": len(done),
            "matches": int(done["bound_matches_all"].astype(bool).sum()),
            "all_s": _stats(done["chips_all_s"]),
            "bound_s": _stats(done["chips_s"]),
            "prune_loss": _stats(done["chips_prune_loss"]),
        }
    if not result.prune.empty:
        out["prune"] = {
            variant: {
                "n_candidates": _stats(group["n_candidates"]),
                "solve_s": _stats(group["solve_s"]),
                "loss": _stats(group["loss"]),
                "n_loss_over_0.1": int((group["loss"] > 0.1).sum()),
            }
            for variant, group in result.prune.groupby("variant", sort=False)
        }
    return out


def summary_text(summary: Mapping[str, Any]) -> str:
    """The printed summary: timing table, bound-vs-all line and pruning table."""

    def fmt(x: float, digits: int = 2) -> str:
        return "-" if x != x else f"{x:.{digits}f}"

    rows = [
        ("input build (s)", "input_s", 2),
        ("model build (s)", "build_s", 2),
        ("no-chip solve (s)", "solve_s", 2),
        ("no-chip final gap", "gap", 4),
        ("top-3 + roll (s)", "top3_s", 1),
        ("chips, bound search (s)", "chips_s", 1),
        ("  LP relaxations (s)", "relax_s", 1),
        ("  scenario MILPs (s)", "chip_solve_s", 1),
        ("  scenarios", "n_scenarios", 0),
        ("  LP relaxations", "n_relaxations", 0),
        ("  MILPs solved", "n_solves", 0),
        ("candidates", "n_candidates", 0),
    ]
    width = max(len(r[0]) for r in rows)
    lines = [f"{summary['n_cases']} cases (median / p90 / max)"]
    lines.append(f"{'':{width}}  {'median':>8} {'p90':>8} {'max':>8}")
    for label, key, digits in rows:
        s = summary["timings"][key]
        cells = " ".join(f"{fmt(s[k], digits):>8}" for k in ("median", "p90", "max"))
        lines.append(f"{label:{width}}  {cells}")
    if "chips_all" in summary:
        c = summary["chips_all"]
        lines.append(
            f"\nExhaustive chip search on {c['n_cases']} cases: median "
            f"{fmt(c['all_s']['median'], 1)}s (max {fmt(c['all_s']['max'], 1)}s) vs bound "
            f"search {fmt(c['bound_s']['median'], 1)}s; same best plan in "
            f"{c['matches']}/{c['n_cases']}; with chips at gap {PRUNE_GAP:g}, default "
            f"pruning loses {fmt(c['prune_loss']['median'], 3)} (max "
            f"{fmt(c['prune_loss']['max'], 3)}) vs dominance pruning only"
        )
    if "prune" in summary:
        lines.append(
            f"\nPruning (no chips, gap {PRUNE_GAP:g}; loss = best variant's objective - this "
            "one's, points)"
        )
        header = ("variant", "cands med", "solve med", "solve max", "loss med", "loss max", ">0.1")
        lines.append(
            f"{header[0]:<14}" + "".join(f"{h:>11}" for h in header[1:]),
        )
        for variant, s in summary["prune"].items():
            cells = [
                fmt(s["n_candidates"]["median"], 0),
                fmt(s["solve_s"]["median"]),
                fmt(s["solve_s"]["max"]),
                fmt(s["loss"]["median"], 3),
                fmt(s["loss"]["max"], 3),
                str(s["n_loss_over_0.1"]),
            ]
            lines.append(f"{variant:<14}" + "".join(f"{c:>11}" for c in cells))
    return "\n".join(lines)
