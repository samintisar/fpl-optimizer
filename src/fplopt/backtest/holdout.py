"""The Phase 6 holdout run on 2025/26 (PLAN §9 go-live gate; pre-registered in
docs/superpowers/plans/2026-10-11-phase-6-preregistration.md). Every setting is fixed here,
so `fplopt holdout run` has nothing to vary, and it runs once.

- **System** (`SYSTEM`): v1 xP + the Phase 5 optimizer, no chips.
- **Baselines** (`BASELINES`): (a) rolling-average xP + greedy; (b) FPL `ep_next` + the
  same optimizer. The system must beat **each** (an intersection-union test: each at
  α = `ALPHA`, no multiplicity correction).
- **Gate** (`gate`): per-decision windows (k = `K`, stride `K`, each arm continuing with its
  own policy, states from the baseline's run), the one-sided `window_t_test` on realized
  points with p < `ALPHA`, and a positive mean xG-scored difference.
- **Fallback** (`FALLBACK`): run against both baselines in the same job, reported only.
- **Reported, not gating**: the block-bootstrap CIs, full-run differences, season totals.

`already_run` refuses a second run (the experiment log holds `FAMILY`); `dirty_checkout`
and `tagged_head` refuse a run from uncommitted code or from a commit without `TAG`.
`preflight` checks the 2025/26 inputs (schedule, player pool, `ep_next`) at every deadline
without reading any result.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fplopt.backtest.evaluate import (
    BootstrapResult,
    block_bootstrap,
    read_experiments,
    window_t_test,
)
from fplopt.features.store import DataStore

__all__ = (
    "ALPHA",
    "BASELINES",
    "FALLBACK",
    "FAMILY",
    "GateResult",
    "K",
    "REFERENCE",
    "SEASON",
    "STARTS",
    "SYSTEM",
    "TAG",
    "already_run",
    "comparisons_table",
    "dirty_checkout",
    "gate",
    "gate_table",
    "preflight",
    "tagged_head",
)

SEASON = 2025
STARTS = "template@1,random:3@1,random:3@20"
SYSTEM = "optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2"
BASELINES = (
    ("a", "greedy:rolling"),
    ("b", "optimizer:ep_next:horizon=6,decay=0.75,max_hits=1,hit_margin=2"),
)
FALLBACK = "greedy:v1:threshold=2.0"
K = 4  # decision window (GWs) and stride: non-overlapping windows
REFERENCE = "b"  # decision states from the baseline's own run
ALPHA = 0.10
BLOCK_LENGTH = 4
N_BOOT = 2000
SEED = 0
CI = 0.8
FAMILY = "phase6 holdout"
TAG = "phase6-holdout"  # the commit the run is made from
# Tracked files a run itself appends to: not "uncommitted code".
LOG_FILES = ("results/experiments.csv",)


@dataclass(frozen=True)
class GateResult:
    """One candidate vs one baseline on the per-decision windows: the window t-test on
    realized points (`mean`, `p`, `n_windows`), the mean xG-scored difference over the
    windows where both arms have one, the reported block bootstrap, and `passed`
    (p < ALPHA and xg_mean > 0)."""

    candidate: str
    baseline: str
    label: str
    mean: float
    p: float
    n_windows: int
    xg_mean: float
    bootstrap: BootstrapResult
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate": self.candidate,
            "baseline": self.baseline,
            "label": self.label,
            "mean": self.mean,
            "p": self.p,
            "n_windows": self.n_windows,
            "xg_mean": self.xg_mean,
            "passed": self.passed,
            "bootstrap": self.bootstrap.to_dict(),
        }


def gate(decisions: pd.DataFrame, candidate: str, baseline: str, label: str) -> GateResult:
    """The gate for `candidate` vs `baseline` (policy names) on a `per_decision` frame."""
    part = decisions[(decisions["policy_a"] == candidate) & (decisions["policy_b"] == baseline)]
    if part.empty:
        raise ValueError(f"no per-decision rows for {candidate} vs {baseline}")
    realized = window_t_test(part, "diff")
    xg = window_t_test(part, "diff_xg")
    boot = block_bootstrap(part, "diff", block_length=1, n_boot=N_BOOT, seed=SEED, ci=CI)
    passed = realized.p_one_sided < ALPHA and xg.mean > 0  # NaN compares False: fails
    return GateResult(
        candidate,
        baseline,
        label,
        realized.mean,
        realized.p_one_sided,
        realized.n_cells,
        xg.mean,
        boot,
        bool(passed),
    )


def already_run(experiments: Path) -> bool:
    """Whether the experiment log already holds a Phase 6 run."""
    if not Path(experiments).exists():
        return False
    return any(row.get("family") == FAMILY for row in read_experiments(experiments))


def dirty_checkout(root: Path | None = None) -> list[str]:
    """`git status --porcelain` lines of the code's checkout (untracked files included),
    without `LOG_FILES`: empty for a clean checkout. RuntimeError if git can't tell."""
    root = Path(__file__).resolve().parents[3] if root is None else root
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"can't check the checkout at {root} with git: {exc}") from None
    return [line for line in out.splitlines() if line[3:].strip() not in LOG_FILES]


def tagged_head(root: Path | None = None) -> bool:
    """Whether the code's checkout HEAD carries the `TAG` the run is made from (fetch tags
    first: `git fetch --tags`). RuntimeError if git can't tell."""
    root = Path(__file__).resolve().parents[3] if root is None else root
    try:
        out = subprocess.run(
            ["git", "tag", "--points-at", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"can't check the checkout at {root} with git: {exc}") from None
    return TAG in out.split()


def preflight(store: DataStore, season: int = SEASON) -> tuple[pd.DataFrame, list[str]]:
    """The season's inputs at every deadline, without results: per GW the pool size, the
    pool players with a non-null `ep_next`, and the clubs with a fixture in the GW but no
    pool player. Returns (that table, problems): a deadline with no pool, `ep_next` missing
    for more than 5% of the pool, or a club without players."""
    from fplopt.backtest.simulator import season_schedule
    from fplopt.features.baseline import ep_next, player_pool, pool_coverage

    schedule = season_schedule(store, season, 1)
    rows, problems = [], []
    for gw, deadline in zip(schedule["gw"], schedule["deadline_time"], strict=True):
        view = store.as_of(deadline)
        pool = player_pool(view)
        ep = ep_next(view)
        coverage = pool_coverage(view)
        missing = coverage.loc[
            (coverage["n_fixtures"] > 0) & (coverage["n_pool_players"] == 0), "team_key"
        ].tolist()
        n_ep = int(ep["ep_next"].notna().sum())
        rows.append(
            {
                "gw": int(gw),
                "deadline": deadline,
                "pool": len(pool),
                "ep_next": n_ep,
                "clubs_without_players": missing,
            }
        )
        if pool.empty:
            problems.append(f"GW{gw}: empty player pool")
        elif n_ep < 0.95 * len(pool):
            problems.append(f"GW{gw}: ep_next for {n_ep} of {len(pool)} pool players")
        if missing:
            problems.append(f"GW{gw}: clubs with a fixture and no pool players {missing}")
    return pd.DataFrame(rows), problems


def comparisons_table(comparisons: pd.DataFrame) -> str:
    """`Summary.comparisons` (split `all`) with each row's pair: method, metric, mean
    difference, the bootstrap CI and p (reported only), cells."""
    lines = [
        f"{'candidate':<40} {'baseline':<40} {'method':<12} {'metric':<11} {'mean':>8} "
        f"{'80% CI':>17} {'boot p':>7} {'cells':>5}"
    ]
    for r in comparisons[comparisons["split"] == "all"].itertuples(index=False):
        ci = f"{r.ci_low:+.2f}..{r.ci_high:+.2f}"
        lines.append(
            f"{r.a:<40} {r.b:<40} {r.method:<12} {r.metric:<11} {r.mean:>+8.2f} {ci:>17} "
            f"{r.p_one_sided:>7.3f} {int(r.n_gws):>5}"
        )
    return "\n".join(lines)


def gate_table(results: Sequence[GateResult]) -> str:
    """The gate results as a fixed-width table."""
    lines = [
        f"{'candidate':<40} {'vs':<4} {'mean/window':>11} {'t p':>7} {'windows':>7} "
        f"{'xG mean':>8} {'boot 80% CI':>17}  result"
    ]
    for r in results:
        ci = f"{r.bootstrap.ci_low:+.2f}..{r.bootstrap.ci_high:+.2f}"
        lines.append(
            f"{r.candidate:<40} {r.label:<4} {r.mean:>+11.2f} {r.p:>7.3f} {r.n_windows:>7} "
            f"{r.xg_mean:>+8.2f} {ci:>17}  {'PASS' if r.passed else 'fail'}"
        )
    return "\n".join(lines)
