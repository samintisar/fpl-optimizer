"""The transfer-planning MILP for one chip scenario (Phase 4 plan, Decisions and Tasks 1-2;
PLAN §7). Built with PuLP and solved by HiGHS in-process (highspy, no temp files).

Indices: candidate i (`PlanInput.players`, sorted by `player_key`), horizon GW t (0 = the
GW being decided). Money is int tenths of £m; prices are fixed over the horizon.

Variables (binary unless noted):
- `squad[i,t]` (the persistent squad); `buy[i,t]` (buyable players); `sell_first[i,t]`
  (owned players: the first sale, at the selling price); `sell_later[i,t]` (a sale at the
  buy price: a player bought, or bought back, within the plan). No buy/sell variables in a
  Free Hit GW;
- `lineup[i,t]`, `captain[i,t]`, `bench[i,t,k]` (k = 0 for goalkeepers, 1..3 outfield);
- `bank[t]` ≥ 0 (continuous); per GW after the first, `fts[t,s]` for s = 1..cap (one
  binary per FT state, `ft[t] = Σ s · fts[t,s]`); per normal GW `hits[t]` ≥ 0 (integer)
  and `take_hits[t]`; per transition `ft_capped[t]`;
- Free Hit GW t only: `fh_squad[i,t]`, `fh_buy[i,t]`, `fh_sell_first[i,t]`,
  `fh_sell_later[i,t]` and `fh_bank[t]` ≥ 0, the one-GW squad (see *Chips*).

Constraints:
- squad continuity `squad[i,t] = squad[i,t−1] + buy − sell_first − sell_later` (start:
  the owned players); buy + sells ≤ 1 per player and GW; an owned player's first sale at
  most once, and a `sell_later` only after it;
- `squad_select` per position (2/5/5/3, so 15); club cap: ≤ `team_limit` per club, except
  a club already over the cap at the deadline (a held player changed club), which may keep
  its excess as long as none of its players is bought in that GW (`state.apply_decision`);
- lineup + bench slots = the GW's squad (the FH squad in a Free Hit GW); 11 starters within
  `play_min`/`play_max`; one player per bench slot (slot 0 the GK, slots 1..3 outfield;
  their order follows from the weights); one captain, a starter;
- `bank[t] = bank[t−1] + Σ sell_price·sell_first + Σ price·sell_later − Σ price·buy`;
- free transfers, exactly as `state.next_state`: in a normal GW `hits = max(n − ft, 0)`
  (pinned with `take_hits`, so a hit can never buy an extra FT), and
  `ft[t+1] = min(cap, max(ft − n, 0) + 1 + top-ups)` (pinned with `ft_capped`). At
  `gw_index == 1` transfers are unlimited and free and the next GW has 1 FT (+ top-ups);
  this takes precedence over a chip played there, as in `next_state`.

Chips (one fixed `ChipScenario` per solve, `chips.py`):
- **Wildcard** GW: transfers free (no hits), and `ft[t+1] = min(cap, ft[t] + top-ups)`
  under `chip_week_ft == "retain"` (`ft[t] + 1 + top-ups` under `retain_plus_one`).
- **Free Hit** GW: the persistent squad is unchanged (`squad[i,t] = squad[i,t−1]`, no
  buys or sells) and `bank[t] = bank[t−1]`. The one-GW squad `fh_squad[i,t] =
  squad[i,t−1] + fh_buy − fh_sell_first − fh_sell_later` is built from the persistent
  squad with the same selling-price rules (an owned player not yet sold in the plan sells
  at his selling price, anyone else at his buy price), so its budget is bank[t−1] + the
  squad's sale value: `fh_bank[t] = bank[t−1] + Σ sell_price·fh_sell_first +
  Σ price·fh_sell_later − Σ price·fh_buy ≥ 0`. Positions and the club cap (with the
  pre-existing-excess rule) apply to it; the GW's lineup, bench and captain are picked
  from it. Transfers are free and FTs follow `chip_week_ft` as for the Wildcard. Next
  GW continues from the persistent squad and bank (the revert of `state.next_state`).
- **Bench Boost** GW: every bench weight is 1 (all 15 count, no autosub weighting).
- **Triple Captain** GW: the captain's xP counts 3× (2 extra instead of 1).

Objective (maximised), with h_t the GW's horizon offset:
  Σ_t decay**h_t · [ Σ_i xp·lineup + (m_t − 1) Σ_i xp·captain + Σ_k w_tk Σ_i xp·bench_k
                     + V(ft[t]) − V(ft[t−1]) + itb_value · bank[t] / 10
                     − (hit_cost + hit_margin) · hits[t] ]
  + terminal_value(scenario)
with m_t the captain multiplier (2, 3 under TC), w_tk the bench weights (1 under BB),
V(s) = Σ_{n ≤ s} ft_value[n] and V(ft[−1]) = 0 (`params`). `bank[t]` is the persistent
bank, so a Free Hit GW credits the money carried past it, not the FH squad's leftover.
The terminal value of unused chips (`chips.terminal_value`) is a constant per scenario,
undecayed; `Plan.objective` excludes it and `Plan.terminal_value` holds it. This follows
open-fpl-solver (solioanalytics/open-fpl-solver, `dev/solver.py`, Apache-2.0) for the FT
value (its `gw_ft_gain`: the gain in FT-state value, inside the decay), money in the bank
(`itb_value · in_the_bank[w]`, after transfers), hits (`hit_cost · penalized_transfers`,
inside the decay), decay (`decay_base ** (w − next_gw)`) and bench weights. Differences:
its vice-captain term (`vcap_weight · xp · vicecap`) is not modelled (the vice is set after
the solve), nor its optional `ft_use_penalty`; its FT dynamics clamp the same way
(`raw_gw_ft` clamped to 1..5 with indicator binaries).

Hooks: `fix_first_gw` pins the first GW's transfers (the roll plan pins none) and
`exclude_first_gw` adds no-good cuts on first-GW transfer sets (top-k). In a Free Hit first
GW both act on the FH squad's transfers.

The model is built in sorted key order with stable variable names, so the same input
gives the same model (and, with `threads=1`, the same plan).
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import pulp

from fplopt.backtest.rules import CHIP_NAMES, CHIP_WEEK_FT, Rules
from fplopt.backtest.state import GOALKEEPER, TRANSFER_CHIPS, Transfer
from fplopt.optimize.chips import NO_CHIP, ChipScenario, make_scenario, terminal_value
from fplopt.optimize.params import OptimizerParams
from fplopt.optimize.plans import GwPlan, Plan
from fplopt.optimize.problem import PlanInput

TransferSet = tuple[frozenset[int], frozenset[int]]  # (outs, ins)
_ON = 0.5  # binary threshold when reading the solution
# Relative margin added to LP-relaxation bounds: HiGHS's primal/dual feasibility tolerances
# (1e-7) can leave the reported LP optimum a hair below the true one.
BOUND_MARGIN = 1e-6


class InfeasiblePlan(RuntimeError):
    """The solver found no feasible plan (e.g. contradictory fixed transfers)."""


def _sum(terms: Iterable[tuple[Any, float]]) -> pulp.LpAffineExpression:
    """Σ coef · var, with repeated variables merged (in first-seen order)."""
    merged: dict[int, list] = {}
    for v, c in terms:
        if v.id in merged:
            merged[v.id][1] += float(c)
        else:
            merged[v.id] = [v, float(c)]
    pairs = [(v, c) for v, c in merged.values() if c != 0]
    return pulp.lpSum_vars_coefs(pairs) if pairs else pulp.LpAffineExpression.empty()


def _vars(terms: Iterable[Any]) -> pulp.LpAffineExpression:
    terms = list(terms)
    return pulp.lpSum_vars(terms) if terms else pulp.LpAffineExpression.empty()


def _value(x: Any) -> float:
    """Value of a variable, expression or constant in the solution."""
    if isinstance(x, int | float):
        return float(x)
    v = x.value()
    return 0.0 if v is None else float(v)


class _BulkHiGHS(pulp.HiGHS):
    """`pulp.HiGHS` with a bulk hand-over: the model goes to highspy in three array calls
    (`addCols`, `addRows`, `changeColsIntegrality`) instead of one call per column, row
    and integer column, which took longer than the solve itself (~0.2 s per 6-GW model;
    Task 3 benchmark). Same columns, rows, order and coefficients as PuLP's own
    `buildSolverModel` (zero coefficients dropped, the objective constant left out), so
    HiGHS sees the identical model; warm starts are not supported."""

    def buildSolverModel(self, lp: pulp.LpProblem) -> None:  # noqa: N802 (PuLP's name)
        import highspy
        import numpy as np

        if self.optionsDict.get("warmStart", False):
            raise ValueError("_BulkHiGHS does not support warm starts")
        inf = highspy.kHighsInf
        h = lp.solverModel
        sign = -1.0 if lp.sense == pulp.LpMaximize else 1.0
        columns = lp.exported_variables()
        col_of = {v.id: j for j, v in enumerate(columns)}
        objective = lp.objective
        n = len(columns)
        cost = np.array([sign * objective.get(v, 0.0) for v in columns], dtype=np.float64)
        lower = np.array(
            [-inf if v.lowBound is None else v.lowBound for v in columns], dtype=np.float64
        )
        upper = np.array(
            [inf if v.upBound is None else v.upBound for v in columns], dtype=np.float64
        )
        empty_i, empty_f = np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)
        h.addCols(n, cost, lower, upper, 0, np.zeros(n, dtype=np.int32), empty_i, empty_f)

        starts, indices, values, row_lower, row_upper = [], [], [], [], []
        for constraint in lp.constraints():
            starts.append(len(indices))
            for var, coef in constraint.items():
                if coef != 0:
                    indices.append(col_of[var.id])
                    values.append(coef)
            lb, ub = constraint.getLb(), constraint.getUb()
            row_lower.append(-inf if lb is None else lb)
            row_upper.append(inf if ub is None else ub)
        h.addRows(
            len(starts),
            np.array(row_lower, dtype=np.float64),
            np.array(row_upper, dtype=np.float64),
            len(indices),
            np.array(starts, dtype=np.int32),
            np.array(indices, dtype=np.int32),
            np.array(values, dtype=np.float64),
        )
        if self.mip:
            integer = [j for j, v in enumerate(columns) if v.cat == pulp.LpInteger]
            if integer:
                kinds = np.full(len(integer), int(highspy.HighsVarType.kInteger), dtype=np.uint8)
                h.changeColsIntegrality(len(integer), np.array(integer, dtype=np.int32), kinds)


class _Model:
    """Variables and constraints of one solve (one chip scenario)."""

    def __init__(
        self, problem: PlanInput, params: OptimizerParams, scenario: ChipScenario = NO_CHIP
    ) -> None:
        self.problem, self.params = problem, params
        rules = problem.rules
        if rules.squad_size - rules.squad_play != len(params.bench_weights):
            raise ValueError(
                f"bench_weights has {len(params.bench_weights)} slots, the rules' bench "
                f"{rules.squad_size - rules.squad_play}"
            )
        self.lp = pulp.LpProblem("fpl_plan", pulp.LpMaximize)
        self.players = problem.players
        self.T = range(len(problem.gws))
        self.chips = scenario.by_t
        unknown = {c for c in self.chips.values() if c not in CHIP_NAMES}
        if unknown or not set(self.chips) <= set(self.T):
            raise ValueError(f"invalid chip scenario {scenario}")
        if self.chips and rules.chip_week_ft not in CHIP_WEEK_FT:
            raise ValueError(f"chip_week_ft {rules.chip_week_ft!r} not in {CHIP_WEEK_FT}")
        self.fh = frozenset(t for t, c in self.chips.items() if c == "freehit")
        self.cap = rules.max_free_transfers
        self.big_m = rules.squad_size + self.cap + sum(a for _, a in rules.ft_topups) + 1
        self._variables()
        self._squad_constraints(rules)
        self._lineup_constraints(rules)
        self._money_and_transfers(rules)

    # --- variables ----------------------------------------------------------------------

    def _bin(self, name: str) -> Any:
        return self.lp.add_variable(name, cat=pulp.LpBinary)

    def _variables(self) -> None:
        self.squad: dict[tuple[int, int], Any] = {}
        self.buy: dict[tuple[int, int], Any] = {}
        self.sell_first: dict[tuple[int, int], Any] = {}
        self.sell_later: dict[tuple[int, int], Any] = {}
        self.lineup: dict[tuple[int, int], Any] = {}
        self.captain: dict[tuple[int, int], Any] = {}
        self.bench: dict[tuple[int, int, int], Any] = {}
        self.fh_squad: dict[tuple[int, int], Any] = {}
        self.fh_buy: dict[tuple[int, int], Any] = {}
        self.fh_sell_first: dict[tuple[int, int], Any] = {}
        self.fh_sell_later: dict[tuple[int, int], Any] = {}
        n_slots = len(self.params.bench_weights)
        for p in self.players:
            k = p.player_key
            # A later sale needs an earlier buy (owned: an earlier first sale, then a buy).
            later_from = 2 if p.owned else 1
            for t in self.T:
                self.squad[k, t] = self._bin(f"squad_{k}_{t}")
                if t in self.fh:
                    self.fh_squad[k, t] = self._bin(f"fh_squad_{k}_{t}")
                    if p.buyable:
                        self.fh_buy[k, t] = self._bin(f"fh_buy_{k}_{t}")
                    if p.owned:
                        self.fh_sell_first[k, t] = self._bin(f"fh_sell_first_{k}_{t}")
                    if p.buyable and t >= later_from:
                        self.fh_sell_later[k, t] = self._bin(f"fh_sell_later_{k}_{t}")
                else:
                    if p.buyable:
                        self.buy[k, t] = self._bin(f"buy_{k}_{t}")
                    if p.owned:
                        self.sell_first[k, t] = self._bin(f"sell_first_{k}_{t}")
                    if p.buyable and t >= later_from:
                        self.sell_later[k, t] = self._bin(f"sell_later_{k}_{t}")
                self.lineup[k, t] = self._bin(f"lineup_{k}_{t}")
                self.captain[k, t] = self._bin(f"captain_{k}_{t}")
                slots = [0] if p.element_type == GOALKEEPER else range(1, n_slots)
                for s in slots:
                    self.bench[k, t, s] = self._bin(f"bench_{k}_{t}_{s}")
        self.bank = {t: self.lp.add_variable(f"bank_{t}", lowBound=0) for t in self.T}
        self.fh_bank = {
            t: self.lp.add_variable(f"fh_bank_{t}", lowBound=0) for t in sorted(self.fh)
        }
        self.bench_of: dict[tuple[int, int], list[Any]] = defaultdict(list)  # (key, t)
        self.slot: dict[tuple[int, int], list[Any]] = defaultdict(list)  # (t, slot)
        for (k, t, s), v in self.bench.items():
            self.bench_of[k, t].append(v)
            self.slot[t, s].append(v)

    def sells(self, k: int, t: int) -> pulp.LpAffineExpression:
        return _vars(
            v for v in (self.sell_first.get((k, t)), self.sell_later.get((k, t))) if v is not None
        )

    def fh_sells(self, k: int, t: int) -> pulp.LpAffineExpression:
        return _vars(
            v
            for v in (self.fh_sell_first.get((k, t)), self.fh_sell_later.get((k, t)))
            if v is not None
        )

    def first_sales_before(self, k: int, t: int) -> pulp.LpAffineExpression:
        """Σ sell_first[k,s] for s < t: 1 once an owned player has been sold."""
        return _vars(self.sell_first[k, s] for s in range(t) if (k, s) in self.sell_first)

    def n_transfers(self, t: int) -> pulp.LpAffineExpression:
        return _vars(
            self.buy[p.player_key, t] for p in self.players if (p.player_key, t) in self.buy
        )

    def selected(self, k: int, t: int) -> Any:
        """The squad that plays GW t: the FH squad in a Free Hit GW, else the squad."""
        return self.fh_squad[k, t] if t in self.fh else self.squad[k, t]

    def bench_weights(self, t: int) -> tuple[float, ...]:
        if self.chips.get(t) == "bboost":
            return (1.0,) * len(self.params.bench_weights)
        return self.params.bench_weights

    def captain_extra(self, t: int) -> float:
        """The captain's xP counted on top of his starter xP: 1 (×2), 2 under TC (×3)."""
        return 2.0 if self.chips.get(t) == "3xc" else 1.0

    # --- constraints --------------------------------------------------------------------

    def _squad_constraints(self, rules: Rules) -> None:
        lp = self.lp
        for p in self.players:
            k = p.player_key
            for t in self.T:
                prev = (1 if p.owned else 0) if t == 0 else self.squad[k, t - 1]
                buy = self.buy.get((k, t), 0)
                lp += self.squad[k, t] == prev + buy - self.sells(k, t)
                if (k, t) in self.buy:
                    lp += buy + self.sells(k, t) <= 1
                if t in self.fh:
                    fh_buy = self.fh_buy.get((k, t), 0)
                    lp += self.fh_squad[k, t] == prev + fh_buy - self.fh_sells(k, t)
                    if (k, t) in self.fh_buy:
                        lp += fh_buy + self.fh_sells(k, t) <= 1
                    if p.owned:
                        # Selling price only if not sold before; buy price only if re-bought.
                        earlier = self.first_sales_before(k, t)
                        lp += self.fh_sell_first[k, t] + earlier <= 1
                        if (k, t) in self.fh_sell_later:
                            lp += self.fh_sell_later[k, t] <= earlier
            if p.owned:
                lp += _vars(self.sell_first[k, t] for t in self.T if (k, t) in self.sell_first) <= 1
                for t in self.T:
                    if (k, t) in self.sell_later:
                        lp += self.sell_later[k, t] <= self.first_sales_before(k, t)

        by_type: dict[int, list[int]] = defaultdict(list)
        by_club: dict[int, list[int]] = defaultdict(list)
        owned_per_club: dict[int, int] = defaultdict(int)
        for p in self.players:
            by_type[p.element_type].append(p.player_key)
            by_club[p.team_key].append(p.player_key)
            if p.owned:
                owned_per_club[p.team_key] += 1
        groups = (by_type, by_club, owned_per_club)
        for t in self.T:
            self._selection(rules, groups, self.squad, self.buy, t)
            if t in self.fh:
                self._selection(rules, groups, self.fh_squad, self.fh_buy, t)

    def _selection(
        self,
        rules: Rules,
        groups: tuple[Mapping[int, list[int]], Mapping[int, list[int]], Mapping[int, int]],
        squad: Mapping[tuple[int, int], Any],
        buy: Mapping[tuple[int, int], Any],
        t: int,
    ) -> None:
        """Positions and the club cap for one GW's squad variables (`groups`: keys by
        position, keys by club, owned players per club)."""
        lp = self.lp
        by_type, by_club, owned_per_club = groups
        for et, n in sorted(rules.squad_select.items()):
            lp += _vars(squad[k, t] for k in by_type.get(et, [])) == n
        for club, keys in sorted(by_club.items()):
            count = _vars(squad[k, t] for k in keys)
            start = owned_per_club.get(club, 0)
            if start <= rules.team_limit:
                lp += count <= rules.team_limit
            else:
                # Pre-existing excess: tolerated unless a player of the club is bought.
                excess = start - rules.team_limit
                for k in keys:
                    if (k, t) in buy:
                        lp += count + excess * buy[k, t] <= start

    def _lineup_constraints(self, rules: Rules) -> None:
        lp = self.lp
        n_slots = len(self.params.bench_weights)
        for t in self.T:
            for p in self.players:
                k = p.player_key
                lp += self.lineup[k, t] + _vars(self.bench_of[k, t]) == self.selected(k, t)
                lp += self.captain[k, t] <= self.lineup[k, t]
            lp += _vars(self.lineup[p.player_key, t] for p in self.players) == rules.squad_play
            for et in sorted(rules.play_min):
                starters = _vars(
                    self.lineup[p.player_key, t] for p in self.players if p.element_type == et
                )
                lp += starters >= rules.play_min[et]
                lp += starters <= rules.play_max[et]
            for s in range(n_slots):
                lp += _vars(self.slot[t, s]) == 1
            lp += _vars(self.captain[p.player_key, t] for p in self.players) == 1

    def _flow(self, t: int, sell_first: Mapping, sell_later: Mapping, buy: Mapping) -> Any:
        """Σ sell_price·sell_first + Σ price·sell_later − Σ price·buy at GW t."""
        terms: list[tuple[Any, float]] = []
        for p in self.players:
            k = p.player_key
            if (k, t) in sell_first:
                terms.append((sell_first[k, t], p.sell_price))
            if (k, t) in sell_later:
                terms.append((sell_later[k, t], p.price))
            if (k, t) in buy:
                terms.append((buy[k, t], -p.price))
        return _sum(terms)

    def _money_and_transfers(self, rules: Rules) -> None:
        lp, problem = self.lp, self.problem
        for t in self.T:
            prev = problem.bank if t == 0 else self.bank[t - 1]
            lp += self.bank[t] == prev + self._flow(t, self.sell_first, self.sell_later, self.buy)
            if t in self.fh:
                flow = self._flow(t, self.fh_sell_first, self.fh_sell_later, self.fh_buy)
                lp += self.fh_bank[t] == prev + flow

        # Free transfers. ft[t] is a constant or Σ s·fts[t,s]; hits[t] a variable or 0.
        cap, m = self.cap, self.big_m
        gws = problem.gws
        chip_extra = 1 if rules.chip_week_ft == "retain_plus_one" else 0
        self.ft: dict[int, Any] = {0: max(problem.free_transfers, 0)}
        self.ft_value: dict[int, Any] = {0: self.params.ft_state_value(self.ft[0])}
        self.hits: dict[int, Any] = {}
        for t in self.T:
            free_gw = t == 0 and problem.gw1
            transfer_chip = self.chips.get(t) in TRANSFER_CHIPS
            if free_gw or transfer_chip:
                self.hits[t] = 0
            else:
                n = self.n_transfers(t)
                hits = self.lp.add_variable(f"hits_{t}", lowBound=0, cat=pulp.LpInteger)
                take = self._bin(f"take_hits_{t}")
                lp += hits >= n - self.ft[t]
                lp += hits <= n - self.ft[t] + m * (1 - take)
                lp += hits <= m * take
                self.hits[t] = hits
            if t + 1 not in self.T:
                break
            topup = sum(a for g, a in rules.ft_topups if gws[t].gw_index < g <= gws[t + 1].gw_index)
            if free_gw:
                target: Any = 1 + topup  # after GW1 the count restarts at 1
            elif transfer_chip:
                target = self.ft[t] + chip_extra + topup  # chip_week_ft
            else:
                target = self.ft[t] - n + self.hits[t] + 1 + topup  # max(ft − n, 0) + 1 + ...
            if isinstance(target, int):
                self.ft[t + 1] = min(cap, target)
                self.ft_value[t + 1] = self.params.ft_state_value(self.ft[t + 1])
                continue
            states = {s: self._bin(f"fts_{t + 1}_{s}") for s in range(1, cap + 1)}
            lp += _vars(states.values()) == 1
            ft_next = _sum((v, s) for s, v in states.items())
            capped = self._bin(f"ft_capped_{t}")
            lp += ft_next <= target
            lp += ft_next >= target - m * capped
            lp += ft_next >= cap - m * (1 - capped)
            self.ft[t + 1] = ft_next
            # V(ft[t+1]) as (variable, coefficient) terms.
            self.ft_value[t + 1] = [(v, self.params.ft_state_value(s)) for s, v in states.items()]

    # --- hooks -------------------------------------------------------------------------

    def first_gw_vars(self) -> tuple[dict[int, Any], dict[int, Any]]:
        """(sale variables by key, buy variables by key) of the first GW's transfers (the
        FH squad's in a Free Hit first GW)."""
        if 0 in self.fh:
            sell_first, buy = self.fh_sell_first, self.fh_buy
        else:
            sell_first, buy = self.sell_first, self.buy
        sells = {p.player_key: sell_first[p.player_key, 0] for p in self.players if p.owned}
        buys = {k: v for (k, t), v in buy.items() if t == 0}
        return sells, buys

    def fix_first_gw(self, transfers: TransferSet) -> None:
        outs, ins = transfers
        sells, buys = self.first_gw_vars()
        unknown = (set(outs) - set(sells)) | (set(ins) - set(buys))
        if unknown:
            raise ValueError(f"fixed transfers name non-candidates: {sorted(unknown)}")
        for k, v in sells.items():
            self.lp += v == (1 if k in outs else 0)
        for k, v in buys.items():
            self.lp += v == (1 if k in ins else 0)

    def exclude_first_gw(self, transfers: TransferSet) -> None:
        """No-good cut: the first GW's (outs, ins) must differ from `transfers`."""
        outs, ins = transfers
        sells, buys = self.first_gw_vars()
        if not set(outs) <= set(sells) or not set(ins) <= set(buys):
            return  # the set is out of reach anyway
        on = [sells[k] for k in sorted(outs)] + [buys[k] for k in sorted(ins)]
        off = [v for k, v in sorted(sells.items()) if k not in outs]
        off += [v for k, v in sorted(buys.items()) if k not in ins]
        self.lp += _vars(off) - _vars(on) >= 1 - len(on)

    # --- objective ----------------------------------------------------------------------

    def objective(self) -> pulp.LpAffineExpression:
        """The horizon part of the objective (the terminal value is added by the caller)."""
        params, rules = self.params, self.problem.rules
        hit_cost = rules.hit_cost + params.hit_margin
        terms: list[tuple[Any, float]] = []
        constant = 0.0
        for t, gw in zip(self.T, self.problem.gws, strict=True):
            d = params.decay**gw.horizon
            weights, extra = self.bench_weights(t), self.captain_extra(t)
            for p in self.players:
                k, x = p.player_key, p.xp[t]
                terms += [(self.lineup[k, t], d * x), (self.captain[k, t], extra * d * x)]
                for s in range(len(weights)):
                    if (k, t, s) in self.bench:
                        terms.append((self.bench[k, t, s], d * weights[s] * x))
            terms.append((self.bank[t], d * params.itb_value / 10))
            for expr, sign in ((self.ft_value[t], 1.0), (self.ft_value.get(t - 1, 0.0), -1.0)):
                if isinstance(expr, int | float):
                    constant += sign * d * expr
                else:
                    terms += [(v, sign * d * c) for v, c in expr]
            if not isinstance(self.hits[t], int):
                terms.append((self.hits[t], -d * hit_cost))
        return _sum(terms) + constant


def solve_plan(
    problem: PlanInput,
    params: OptimizerParams,
    *,
    fix_first_gw: TransferSet | None = None,
    exclude_first_gw: Iterable[TransferSet] = (),
    chips: ChipScenario | Mapping[int, str] | None = None,
) -> Plan:
    """Build and solve the MILP; return the plan.

    `fix_first_gw=(outs, ins)` pins the first GW's transfers (`(frozenset(), frozenset())`
    = the roll plan's first GW); each `exclude_first_gw` entry is a first-GW (outs, ins) set
    the plan must differ from (a no-good cut, for top-k plans). `chips` fixes the chip
    scenario: a `ChipScenario`, or a mapping position in the horizon → chip name (validated
    with `chips.make_scenario`; InvalidDecision if the rules don't allow it); `None`/empty
    = no chip. The plan's `terminal_value` is the scenario's (`chips.terminal_value`).
    Raises InfeasiblePlan if HiGHS returns no solution."""
    if isinstance(chips, ChipScenario):
        scenario = chips
    else:
        scenario = make_scenario(problem, chips) if chips else NO_CHIP
    start = time.perf_counter()
    model = _Model(problem, params, scenario)
    if fix_first_gw is not None:
        model.fix_first_gw(fix_first_gw)
    for transfers in exclude_first_gw:
        model.exclude_first_gw(transfers)
    terminal = terminal_value(problem, params, scenario)
    model.lp.setObjective(model.objective() + terminal)
    build_seconds = time.perf_counter() - start

    solver = _BulkHiGHS(
        msg=False,
        gapRel=params.mip_gap,
        threads=params.threads,
        timeLimit=params.time_limit,
        random_seed=0,
    )
    start = time.perf_counter()
    stats = model.lp.solve(solver)
    solve_seconds = time.perf_counter() - start
    if not stats.has_solution:
        raise InfeasiblePlan(f"no feasible plan (HiGHS status {stats.status_str})")
    info = model.lp.solverModel.getInfo()
    gws = _read_plan(model)
    objective = sum(g.objective for g in gws)
    # HiGHS minimises −(objective − constant) (PuLP flips the sense and keeps the constant),
    # so its dual bound maps back as constant − bound. Never below the plan itself.
    bound = float(model.lp.objective.constant) - float(info.mip_dual_bound)
    if not math.isfinite(bound):
        bound = math.inf
    return Plan(
        gws=gws,
        objective=objective,
        status=str(stats.status_str),
        mip_gap=float(info.mip_gap),
        build_seconds=build_seconds,
        solve_seconds=solve_seconds,
        n_candidates=len(problem.players),
        scenario=scenario.label,
        terminal_value=terminal,
        bound=max(bound, objective + terminal),
    )


def relaxation_bound(
    problem: PlanInput, params: OptimizerParams, scenario: ChipScenario = NO_CHIP
) -> float:
    """An upper bound on the best `Plan.total_objective` of `scenario`: the optimum of its
    LP relaxation (integrality dropped; same model, objective incl. the terminal value).
    Every integral plan is feasible for the LP, so the LP optimum is ≥ the MILP optimum;
    a small tolerance margin (`BOUND_MARGIN`, relative) covers HiGHS's feasibility
    tolerances. `math.inf` when HiGHS doesn't report the LP optimal (no bound)."""
    model = _Model(problem, params, scenario)
    model.lp.setObjective(model.objective() + terminal_value(problem, params, scenario))
    solver = _BulkHiGHS(mip=False, msg=False, threads=params.threads, timeLimit=params.time_limit)
    stats = model.lp.solve(solver)
    if str(stats.status_str) != "Optimal" or not stats.has_solution:
        return math.inf
    value = float(model.lp.objective.value())
    return value + BOUND_MARGIN * max(1.0, abs(value))


def _on(var: Any) -> bool:
    return var is not None and _value(var) > _ON


def _read_plan(model: _Model) -> tuple[GwPlan, ...]:
    """The solution as GwPlans (vice = best non-captain starter by xP, ties by key)."""
    problem, params = model.problem, model.params
    hit_cost = problem.rules.hit_cost + params.hit_margin
    et = {p.player_key: p.element_type for p in problem.players}
    out = []
    previous_ft_value = 0.0
    for t, gw in enumerate(problem.gws):
        xp = {p.player_key: p.xp[t] for p in problem.players}
        freehit = t in model.fh
        sells = model.fh_sells if freehit else model.sells
        buy = model.fh_buy if freehit else model.buy
        outs = [p.player_key for p in problem.players if _on_expr(sells(p.player_key, t))]
        ins = [k for (k, tt), v in buy.items() if tt == t and _on(v)]
        transfers = _pair(outs, ins, et)
        starters = sorted(
            (k for (k, tt), v in model.lineup.items() if tt == t and _on(v)),
            key=lambda k: (et[k], -xp[k], k),
        )
        slots = sorted((s, k) for (k, tt, s), v in model.bench.items() if tt == t and _on(v))
        bench = tuple(k for _, k in slots)
        captain = next(k for (k, tt), v in model.captain.items() if tt == t and _on(v))
        vice = min((k for k in starters if k != captain), key=lambda k: (-xp[k], k))
        hits = round(_value(model.hits[t]))
        kept_bank = round(_value(model.bank[t]))  # the persistent bank
        bank = round(_value(model.fh_bank[t])) if freehit else kept_bank
        ft = round(_value(model.ft[t]))
        ft_value = params.ft_state_value(ft)
        lineup_xp = sum(xp[k] for k in starters) + model.captain_extra(t) * xp[captain]
        weights = model.bench_weights(t)
        bench_xp = sum(w * xp[k] for w, k in zip(weights, bench, strict=True))
        d = params.decay**gw.horizon
        objective = d * (
            lineup_xp
            + bench_xp
            + ft_value
            - previous_ft_value
            + params.itb_value * kept_bank / 10
            - hit_cost * hits
        )
        previous_ft_value = ft_value
        out.append(
            GwPlan(
                gw=gw.gw,
                gw_index=gw.gw_index,
                horizon=gw.horizon,
                transfers=transfers,
                starters=tuple(starters),
                bench=bench,
                captain=captain,
                vice=vice,
                chip=model.chips.get(t),
                xp=lineup_xp,
                bench_xp=bench_xp,
                objective=objective,
                hits=hits,
                bank=bank,
                ft=ft,
            )
        )
    return tuple(out)


def _on_expr(expr: pulp.LpAffineExpression) -> bool:
    return _value(expr) > _ON


def _pair(outs: list[int], ins: list[int], et: Mapping[int, int]) -> tuple[Transfer, ...]:
    """(out, in) pairs within position, by player_key (equal counts per position)."""
    pairs = []
    for position in sorted({et[k] for k in outs + ins}):
        o = sorted(k for k in outs if et[k] == position)
        i = sorted(k for k in ins if et[k] == position)
        if len(o) != len(i):
            raise AssertionError(f"position {position}: {len(o)} out, {len(i)} in")
        pairs += [Transfer(a, b) for a, b in zip(o, i, strict=True)]
    return tuple(pairs)
