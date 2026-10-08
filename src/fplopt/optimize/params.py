"""Optimizer parameters (PLAN §7 *Objective*, *Defaults*; Phase 4 plan, Decisions).

All defaults are start values, to be tuned by backtest (horizon, decay and FT value are
confounded and tuned jointly). Where PLAN §7 is silent the conventions follow
open-fpl-solver (solioanalytics/open-fpl-solver, Apache-2.0, `dev/solver.py`), so the
reference check (Phase 4 Task 4) compares like with like:

- `ft_value`: marginal value of the n-th free transfer available at a GW, keyed by n. The
  value of holding s FTs is V(s) = Σ_{n ≤ s} ft_value[n] (missing keys count 0). As in
  open-fpl-solver, the objective credits the *gain* V(ft[t]) − V(ft[t−1]) at GW t (inside
  the decay, with V(ft[−1]) = 0), not V(ft[t]) itself. Our states are 1..cap, so a key 1
  (open-fpl-solver's increments for states 0 and 1, both its scalar `ft_value` by default)
  only adds the constant V(ft[0]) to the objective; set it to reproduce that constant.
- `itb_value`: points per £1m in the bank after each GW's transfers, per GW (inside the
  decay); open-fpl-solver's `itb_value` (£m units; our money is tenths, so bank / 10).
  **Default 0** (open-fpl-solver: 0.08; Phase 4 Task 5 backtests, PLAN §7): with 0.08 the
  planner hoards cash (mean bank £3.1m vs greedy's £1.7m), selling premiums for cheap
  players, and with uninformative xP (2016/17 GW1, no history) it sold down to the
  cheapest squad and banked £36m. optimizer(rolling, max_hits 0) gained +0.79 pts/GW
  (80% CI +0.24 to +1.38, 2016/17-2024/25) from 0.08 -> 0.
- `bench_weights`: (GK slot, outfield slots 1–3), open-fpl-solver's `bench_weights`.
- `decay`: GW t (horizon offset t, target GW t = 0) is weighted decay**t, as
  open-fpl-solver's `decay_base ** (w − next_gw)`.
- Hits cost `rules.hit_cost + hit_margin` each, inside the decay (open-fpl-solver's
  `hit_cost × penalized_transfers` sits inside its decayed GW total too).
- `max_hits`: at most this many hits in every horizon GW (`None` = unlimited, `0` = never
  take a hit); a safeguard against inflated xP chasing form with hits. **Default 0**
  (Phase 4 Task 5, PLAN §7): against greedy on ep_next_fade 2021/22-2024/25, max_hits 0
  was -0.9 pts/GW, max_hits 1 with hit_margin 2 -1.4 (28 hits a season) and unlimited
  -4.0 (77 hits a season); realized transfer gains are ~1/4-1/2 of the predicted ones, so
  a hit (4 points) rarely pays. `hit_margin` only matters with `max_hits` > 0.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


def _frozen(mapping: Mapping) -> Mapping:
    return MappingProxyType(dict(sorted(mapping.items())))


DEFAULT_FT_VALUE = MappingProxyType({2: 2.0, 3: 1.6, 4: 1.3, 5: 1.1})
DEFAULT_BENCH_WEIGHTS = (0.03, 0.21, 0.06, 0.002)
# Terminal value (points, undecayed) per chip name of an unused chip whose window extends
# past the horizon (`chips.terminal_value`). PLACEHOLDERS, not estimates: rough guesses at
# what a well-timed chip adds over not playing it (Phase 4 plan, Decisions; PLAN §7 wants
# them estimated from backtest distributions later). A chip is played inside the horizon
# only if it beats holding it by at least this much.
DEFAULT_CHIP_VALUE = MappingProxyType({"wildcard": 6.0, "freehit": 4.0, "bboost": 4.0, "3xc": 3.0})
# Candidates kept per position (element_type) by horizon xP and by horizon xP per price.
# Task 3 benchmark (40 real cases, no chips, gap 1e-4, with club-aware dominance): vs the
# unpruned pool this loses 0 points in the median and at most 0.21 (10/30/30/15: max 1.0,
# at GW1 squad builds), with ~130 candidates and a median solve of ~1.4 s at that gap.
DEFAULT_PRUNE_N = MappingProxyType({1: 20, 2: 60, 3: 60, 4: 30})


@dataclass(frozen=True)
class OptimizerParams:
    """Planner settings. Mappings are copied into read-only, key-sorted mappings.

    `prune_n=None` disables top-N pruning and `prune_dominated=False` dominance pruning
    (both off = every pool player is a candidate). `mip_gap` is HiGHS's `mip_rel_gap`,
    `threads` its thread count (1 for reproducibility), `time_limit` (seconds) a safety net
    only."""

    horizon: int = 6
    decay: float = 0.85
    ft_value: Mapping[int, float] = field(default_factory=lambda: DEFAULT_FT_VALUE)
    itb_value: float = 0.0
    hit_margin: float = 0.0
    max_hits: int | None = 0
    bench_weights: tuple[float, float, float, float] = DEFAULT_BENCH_WEIGHTS
    chip_value: Mapping[str, float] = field(default_factory=lambda: DEFAULT_CHIP_VALUE)
    prune_n: Mapping[int, int] | None = field(default_factory=lambda: DEFAULT_PRUNE_N)
    prune_dominated: bool = True
    mip_gap: float = 0.005
    threads: int = 1
    time_limit: float = 60.0

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.max_hits is not None:
            if self.max_hits < 0:
                raise ValueError(f"max_hits must be >= 0 or None, got {self.max_hits}")
            object.__setattr__(self, "max_hits", int(self.max_hits))
        if not 0 < self.decay <= 1:
            raise ValueError(f"decay must be in (0, 1], got {self.decay}")
        ft_value = {int(k): float(v) for k, v in self.ft_value.items()}
        if any(k < 1 for k in ft_value) or any(v < 0 for v in ft_value.values()):
            # Negative values would make the solver prefer fewer FTs than the rules give,
            # which the FT dynamics (exact) cannot express anyway; keep them meaningful.
            raise ValueError(f"ft_value keys must be >= 1 and values >= 0, got {ft_value}")
        object.__setattr__(self, "ft_value", _frozen(ft_value))
        if len(self.bench_weights) != 4:
            raise ValueError(f"bench_weights needs 4 values, got {self.bench_weights}")
        object.__setattr__(self, "bench_weights", tuple(float(w) for w in self.bench_weights))
        object.__setattr__(
            self, "chip_value", _frozen({str(k): float(v) for k, v in self.chip_value.items()})
        )
        if self.prune_n is not None:
            prune_n = {int(k): int(v) for k, v in self.prune_n.items()}
            if any(v < 0 for v in prune_n.values()):
                raise ValueError(f"prune_n values must be >= 0, got {prune_n}")
            object.__setattr__(self, "prune_n", _frozen(prune_n))
        if not 0 <= self.mip_gap < 1:
            raise ValueError(f"mip_gap must be in [0, 1), got {self.mip_gap}")
        if self.threads < 1:
            raise ValueError(f"threads must be >= 1, got {self.threads}")
        if self.time_limit <= 0:
            raise ValueError(f"time_limit must be > 0, got {self.time_limit}")

    def __reduce__(self) -> tuple:
        """Pickle by the constructor's arguments (the read-only mappings can't be pickled
        as they are), so params travel to backtest worker processes."""
        return (
            OptimizerParams,
            (
                self.horizon,
                self.decay,
                dict(self.ft_value),
                self.itb_value,
                self.hit_margin,
                self.max_hits,
                self.bench_weights,
                dict(self.chip_value),
                None if self.prune_n is None else dict(self.prune_n),
                self.prune_dominated,
                self.mip_gap,
                self.threads,
                self.time_limit,
            ),
        )

    def ft_state_value(self, s: int) -> float:
        """V(s) = Σ_{n ≤ s} ft_value[n]: the value of having s free transfers."""
        return sum(v for n, v in self.ft_value.items() if n <= s)
