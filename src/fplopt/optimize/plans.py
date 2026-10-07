"""Plans: the MILP's answer, per horizon GW, and its first-GW decision (Phase 4 plan,
Task 1; top-k, roll plan and chip scenarios come with Task 2).

A `Plan` holds one `GwPlan` per horizon GW plus solve statistics. `Plan.decision()` is the
first GW's `state.Decision` (transfers paired within position, lineup, chip), which
`state.apply_decision` accepts.
"""

from __future__ import annotations

from dataclasses import dataclass

from fplopt.backtest.gw_score import Lineup
from fplopt.backtest.state import Decision, Transfer
from fplopt.optimize.params import OptimizerParams
from fplopt.optimize.problem import PlanInput


@dataclass(frozen=True)
class GwPlan:
    """One horizon GW of a plan.

    - `transfers`: (out, in) pairs, paired within position (each GW keeps 2/5/5/3, so
      every position sells as many as it buys); `n_transfers = len(transfers)`.
    - `starters` (by position, then xP descending), `bench` (slot order, bench[0] the GK),
      `captain`, `vice` (the best other starter by xP, set after the solve), `chip`.
    - `xp`: expected points of the XI with the captain counted twice (no bench, no vice).
    - `bench_xp`: Σ_k bench_weights[k] · xP(bench[k]), the bench as the objective values it.
    - `objective`: this GW's decayed contribution to the objective:
      decay**horizon · (xp + bench_xp + ft gain + itb_value · bank/10 − hit cost · hits),
      the ft gain being V(ft) − V(previous ft) (V(previous) = 0 in the first GW).
    - `hits`: transfers paid for; `bank` (tenths of £m) after this GW's transfers; `ft`:
      free transfers available for this GW (before its transfers; meaningless at GW1).
    """

    gw: int
    gw_index: int
    horizon: int
    transfers: tuple[Transfer, ...]
    starters: tuple[int, ...]
    bench: tuple[int, ...]
    captain: int
    vice: int
    chip: str | None
    xp: float
    bench_xp: float
    objective: float
    hits: int
    bank: int
    ft: int

    @property
    def n_transfers(self) -> int:
        return len(self.transfers)

    @property
    def transfers_out(self) -> tuple[int, ...]:
        return tuple(t.out_key for t in self.transfers)

    @property
    def transfers_in(self) -> tuple[int, ...]:
        return tuple(t.in_key for t in self.transfers)

    @property
    def lineup(self) -> Lineup:
        return Lineup(self.starters, self.bench, self.captain, self.vice)

    def decision(self) -> Decision:
        return Decision(transfers=self.transfers, lineup=self.lineup, chip=self.chip)


@dataclass(frozen=True)
class Plan:
    """A solved plan. `objective` = Σ of the GWs' `objective`, recomputed from the
    integral solution (the solver's objective up to its tolerances); `status` HiGHS's model
    status as PuLP reports it (`Optimal` also when stopped at `mip_gap`, `TimeLimit` when
    the safety net stopped it with a solution);
    `mip_gap` the final relative gap; `build_seconds` the PuLP model build and
    `solve_seconds` the HiGHS call (including PuLP's hand-over); `n_candidates` the
    players in the model."""

    gws: tuple[GwPlan, ...]
    objective: float
    status: str
    mip_gap: float
    build_seconds: float
    solve_seconds: float
    n_candidates: int

    @property
    def first(self) -> GwPlan:
        return self.gws[0]

    @property
    def total_xp(self) -> float:
        """Σ over the horizon of the XI's xP with captain (undiscounted)."""
        return sum(g.xp for g in self.gws)

    def decision(self) -> Decision:
        """The first GW's decision."""
        return self.first.decision()


def optimize(problem: PlanInput, params: OptimizerParams) -> Plan:
    """Plan #1 (no chips). Task 2 extends this to top-k plans, the roll plan and chip
    scenarios."""
    from fplopt.optimize.model import solve_plan  # model imports this module's dataclasses

    return solve_plan(problem, params)
