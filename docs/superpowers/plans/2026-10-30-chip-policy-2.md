# Chip Policy 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the chip variants V1–V6 of `docs/superpowers/specs/2026-10-30-chip-policy-2-design.md` (per-window and double-aware terminal values estimated leave-one-season-out, a rule Triple Captain, a Free Hit gain shrink), estimate their tables, and run them against H on develop.

**Architecture:** The optimizer's terminal value (`optimize.chips.chip_value`) learns two finer table keys, `name@window` and `name@window+double`, falling back to the plain chip name; `PlanInput.doubles` carries the double GWs known at the deadline. Two new `OptimizerParams` fields restrict the chip search (`exclude_chips`) and shrink a Free Hit's gain when scenarios are ranked (`fh_scale`). `OptimizerPolicy` picks a leave-one-season-out table per season, fills `doubles` from the as-of schedule, and plays Triple Captain by rule (`tc_rule`). The estimator (`backtest.chip_values`) writes v2 table files with a `loso` section; CLI spec keys `chip_split`, `loso`, `tc_rule`, `fh_scale` select it all. A dev driver compares the runs.

**Tech Stack:** Python 3.12, pandas, PuLP/HiGHS (existing optimizer), pytest, uv.

**Shadow logger (spec item 6):** a separate plan, due before the GW20 deadline (~2026-12-29).

---

## Conventions for every task

- Work in the worktree on branch `feat/chip-policy-2` (from `origin/main`).
- Run tests with `uv run pytest <path> -q`; lint with `uv run ruff check . && uv run ruff format .` before each commit.
- Commit messages end with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  ```
- Match the surrounding code: docstrings explain *why*, dataclasses are frozen, mappings in `OptimizerParams` are read-only and key-sorted.
- The rules (`backtest_rules`, 2026/27): chips in two sets, set 1 on `gw_index` 1/2–19, set 2 on 20–38. Chip ids: WC 1/2, FH 3/6, BB 4/7, TC 5/8. "Window ordinal" = the 1-based rank of a chip's window among that chip name's windows, by start (set 1 → 1, set 2 → 2).

## File map

| File | Change |
|---|---|
| `src/fplopt/optimize/chips.py` | `chip_value` looks up `name@k+d`, `name@k`, `name`; `terminal_value` passes window ordinal and double flag; `scenarios` honours `params.exclude_chips` |
| `src/fplopt/optimize/problem.py` | `PlanInput.doubles: frozenset[int]` |
| `src/fplopt/optimize/params.py` | fields `exclude_chips`, `fh_scale` (+ validation, `__reduce__`) |
| `src/fplopt/optimize/search.py` | rank scenarios by `_score` (Free Hit gain × `fh_scale`) |
| `src/fplopt/backtest/chip_doubles.py` (new) | `known_doubles(view, season, gw_index, lead)`, `known_doubles_by_gw(store, seasons, lead)` |
| `src/fplopt/backtest/chip_values.py` | `estimate_terminal_values(..., split, doubles, horizon)`, `estimate_loso`, `load_loso_values`, `WINDOW_VALUES_FILE`, `DOUBLE_VALUES_FILE` |
| `src/fplopt/backtest/policies.py` | `_table_tag`; `OptimizerPolicy.season_tables`, `.double_lead`, `.tc_rule`; params extras `nochip=`, `fhs=` |
| `src/fplopt/backtest/evaluate.py` | `WindowTTest.se` |
| `src/fplopt/cli.py` | spec keys `chip_split`, `loso`, `tc_rule`, `fh_scale`; `chips values --split --loso` |
| `config/chips/terminal-values-window.json`, `config/chips/terminal-values-double.json` (new) | estimated tables |
| `dev/chip_policy_2.py` (new) | comparison driver |
| Tests | `tests/test_optimize_chips.py`, `tests/test_chip_study.py`, `tests/test_chip_doubles.py` (new), `tests/test_backtest_evaluate.py` |
| Docs | spec amendments, `CLAUDE.md`, `docs/PLAN.md` decision row |

---

### Task 0: Spec amendments (pre-registration details found while planning)

**Files:**
- Modify: `docs/superpowers/specs/2026-10-30-chip-policy-2-design.md`

- [ ] **Step 1: Edit the spec**

In *Runs*, replace the starts line with:

```markdown
- Develop 2016/17–2022/23, starts `template@1,random:3@1,random:3@20` (7 runs per season, the
  diagnostics' grid, so H and P are comparable with sub-project 1), v1 xP.
```

In *Variants*, after the table, add:

```markdown
- **Spec strings** (`BASE = horizon=6,decay=0.75,max_hits=1,hit_margin=2`):
  V1–V3 `optimizer:v1:BASE,chips=1,chip_shrink=k,chip_split=1,loso=1` (k = 0.6/0.8/1.0);
  V4 `…,chip_shrink=k*,chip_split=1,loso=1,tc_rule=8.3`;
  V5 `…,chip_shrink=k*,chip_split=2,loso=1`;
  V6 `…,chip_shrink=k*,chip_split=1,loso=1,fh_scale=0.4`.
- **V5's "known double"**: a GW (by `gw_index`) where some club has ≥ 2 league fixtures in the
  as-of schedule (`AsOfView.schedule`), at most `DOUBLE_LEAD = 10` GWs after the decision GW.
  Before 2021/22 there are no fixture snapshots and the as-of schedule is the final one, so a
  double counts as known up to 10 GWs ahead even if it was announced later. Hence V5 must
  also, on 2021/22–2022/23 alone (snapshot schedules), have a positive full-run window mean
  vs the best of V1–V3; otherwise V5 can't be the winner.
- **V6's Free Hit shrink** ranks chip scenarios by total objective − (1 − 0.4) ·
  max(0, Σ over Free Hit GWs t of (plan GW t objective − no-chip plan GW t objective)); the
  MILPs are unchanged.
- **V4's rule**: the optimizer plans without Triple Captain scenarios; if plan #1 plays no
  chip this GW and Triple Captain is available, it is played when the captain's xP ≥ 8.3 or
  this is the last GW of its window (re-planned with TC fixed in this GW).
```

In *Statistic and bar*, add a bullet:

```markdown
- The family's runs include H and P (8 runs); the deflation counts the 6 variants
  (N = 6), as pre-registered here.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-10-30-chip-policy-2-design.md
git commit -m "docs(spec): chip policy 2 pre-registration details (starts, spec strings, V4-V6 rules)"
```

---

### Task 1: Window- and double-keyed terminal values

**Files:**
- Modify: `src/fplopt/optimize/problem.py` (class `PlanInput`, after `bench_weights`)
- Modify: `src/fplopt/optimize/chips.py:118-142` (`terminal_value`, `chip_value`)
- Test: `tests/test_optimize_chips.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_optimize_chips.py`)

```python
def test_terminal_values_by_window() -> None:
    from fplopt.optimize.chips import chip_value

    table = {"wildcard@1": (1.0,), "wildcard@2": (5.0,), "freehit": (2.0,)}
    params = replace(EXACT, chip_value={}, chip_value_left=table)
    assert chip_value(params, "wildcard", 3, window=1) == 1.0
    assert chip_value(params, "wildcard", 3, window=2) == 5.0
    assert chip_value(params, "freehit", 3, window=2) == 2.0  # no window key: the name's
    assert chip_value(params, "wildcard", 3) == 1.0  # default window 1
    # GWs 5-7: WC1 1 + WC2 5 + FH 2 per window; BB/TC not in either table: 0.
    assert terminal_value(problem_at(5, 3), params, NO_CHIP) == 1 + 5 + 2 * 2


def test_terminal_values_with_a_known_double() -> None:
    table = {"3xc@1+1": (9.0,), "3xc@1": (1.0,), "3xc@2": (2.0,)}
    params = replace(EXACT, chip_value={}, chip_value_left=table)
    problem = problem_at(5, 3)
    assert problem.doubles == frozenset()
    assert terminal_value(problem, params, NO_CHIP) == 1 + 2
    # A double in GW12: after the horizon (5-7), inside set 1 (..19), before set 2 (20..).
    doubled = replace(problem, doubles=frozenset({12}))
    assert terminal_value(doubled, params, NO_CHIP) == 9 + 2
    # A double inside the horizon doesn't count (the horizon plans it already).
    inside = replace(problem, doubles=frozenset({6}))
    assert terminal_value(inside, params, NO_CHIP) == 1 + 2
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_optimize_chips.py -q -k "by_window or known_double"`
Expected: FAIL (`chip_value() got an unexpected keyword argument 'window'`, `PlanInput` has no `doubles`).

- [ ] **Step 3: Implement**

In `src/fplopt/optimize/problem.py`, add to `PlanInput` after `bench_weights` (and extend the class docstring with one sentence: "`doubles`: the `gw_index`s of double GWs known at the deadline (any club with ≥ 2 league fixtures), for double-aware terminal values; empty unless the policy fills it."):

```python
    doubles: frozenset[int] = frozenset()
```

In `src/fplopt/optimize/chips.py`, replace `terminal_value` and `chip_value` with:

```python
def terminal_value(problem: PlanInput, params: OptimizerParams, scenario: ChipScenario) -> float:
    """Σ `chip_value` over the chip windows unused after the scenario whose window extends
    past the last horizon `gw_index` (0 for a window closing inside it, and for a chip name
    missing from the tables). Each window is valued by its ordinal among the chip's windows
    and whether `problem.doubles` has a double in its part after the horizon. Added to the
    objective undecayed: it is the value of holding the chip at the horizon's end, in
    this-GW points."""
    used = {cid for cid, _ in problem.state.chips_used} | set(scenario.chip_ids)
    last = problem.gws[-1].gw_index
    windows = sorted(problem.rules.chips, key=lambda w: (w.name, w.start))
    ordinal = {}
    for w in windows:
        ordinal[w.chip_id] = 1 + sum(1 for o in windows if o.name == w.name and o.start < w.start)
    return float(
        sum(
            chip_value(
                params,
                w.name,
                w.stop - last,
                window=ordinal[w.chip_id],
                double=any(max(last, w.start - 1) < g <= w.stop for g in problem.doubles),
            )
            for w in sorted(problem.rules.chips, key=lambda w: w.chip_id)
            if w.chip_id not in used and w.stop > last
        )
    )


def chip_value(
    params: OptimizerParams, name: str, left: int, *, window: int = 1, double: bool = False
) -> float:
    """The value of holding chip `name` with `left` (≥ 1) GWs of its window after the
    horizon, from `params.chip_value_left` (its last entry for a longer window): the first
    key of `name@window+double` (0/1), `name@window`, `name` the table has; else the flat
    `params.chip_value[name]` (0 if missing)."""
    table = params.chip_value_left
    if table is not None:
        for key in (f"{name}@{window}+{int(double)}", f"{name}@{window}", name):
            if key in table:
                values = table[key]
                return values[min(left, len(values)) - 1]
    return params.chip_value.get(name, 0.0)
```

Also update the module docstring's `terminal_value` paragraph: append "Tables may key a chip by window (`3xc@2`) and by a known double ahead (`3xc@2+1`); `chip_value` falls back to coarser keys."

- [ ] **Step 4: Run the chip tests**

Run: `uv run pytest tests/test_optimize_chips.py -q`
Expected: all PASS (the existing `test_terminal_values_by_window_left` still passes: plain-name keys behave as before).

- [ ] **Step 5: Check the other caller**

Run: `uv run pytest tests/test_bot.py -q`
Expected: PASS (`bot/messages.py` calls `chip_value(params, name, left)`, still valid).

- [ ] **Step 6: Commit**

```bash
git add src/fplopt/optimize/problem.py src/fplopt/optimize/chips.py tests/test_optimize_chips.py
git commit -m "feat(optimize): terminal values keyed by chip window and a known double ahead"
```

---

### Task 2: `exclude_chips` and `fh_scale` in OptimizerParams; scenarios honour exclusions

**Files:**
- Modify: `src/fplopt/optimize/params.py` (fields after `chip_value_left`; `__post_init__`; `__reduce__`)
- Modify: `src/fplopt/optimize/chips.py` (`scenarios`)
- Modify: `src/fplopt/backtest/policies.py` (`_optimizer_extras`)
- Test: `tests/test_optimize_chips.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_scenarios_without_excluded_chips() -> None:
    problem = problem_at(5, 2)
    params = replace(EXACT, exclude_chips=("3xc",))
    assert params.exclude_chips == ("3xc",)
    kept = labels(problem, params)
    assert kept[0] == "none" and not any("3xc" in label for label in kept)
    assert set(labels(problem)) - set(kept) == {s for s in labels(problem) if "3xc" in s}


def test_exclude_chips_and_fh_scale_are_validated_and_pickle() -> None:
    import pickle

    with pytest.raises(ValueError, match="exclude_chips"):
        OptimizerParams(exclude_chips=("bogus",))
    with pytest.raises(ValueError, match="fh_scale"):
        OptimizerParams(fh_scale=1.5)
    params = OptimizerParams(exclude_chips=["3xc", "bboost"], fh_scale=0.4)
    assert params.exclude_chips == ("3xc", "bboost")  # sorted
    assert pickle.loads(pickle.dumps(params)) == params
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_optimize_chips.py -q -k "excluded or fh_scale_are"`
Expected: FAIL (`unexpected keyword argument 'exclude_chips'`).

- [ ] **Step 3: Implement**

In `params.py`, after `chip_value_left`:

```python
    # Chip names left out of the chip scenario search (`chips.scenarios`), e.g. ("3xc",) when
    # a rule plays Triple Captain instead (chip policy 2, V4). Sorted.
    exclude_chips: tuple[str, ...] = ()
    # The share of a Free Hit's gain over the no-chip plan that counts when chip scenarios
    # are ranked (`search._score`; chip policy 2, V6: the Free Hit forecast slope). 1 = all.
    fh_scale: float = 1.0
```

In `__post_init__` (after the `chip_value_left` block):

```python
        chips = tuple(sorted({str(c) for c in self.exclude_chips}))
        unknown = [c for c in chips if c not in ("3xc", "bboost", "freehit", "wildcard")]
        if unknown:
            raise ValueError(f"exclude_chips: unknown chip names {unknown}")
        object.__setattr__(self, "exclude_chips", chips)
        if not math.isfinite(self.fh_scale) or not 0 <= self.fh_scale <= 1:
            raise ValueError(f"fh_scale must be in [0, 1], got {self.fh_scale}")
        object.__setattr__(self, "fh_scale", float(self.fh_scale))
```

In `__reduce__`, append after the `chip_value_left` entry:

```python
                self.exclude_chips,
                self.fh_scale,
```

In `chips.py` `scenarios`, replace the docstring's last sentence and the `del params` / `names` lines with:

```python
    name); chip names in `params.exclude_chips` are left out."""
    names = sorted({c.name for c in problem.rules.chips} - set(params.exclude_chips))
```

In `policies.py` `_optimizer_extras`, before `return parts`:

```python
    if params.exclude_chips != default.exclude_chips:
        parts.append("nochip=" + "/".join(params.exclude_chips))
    if params.fh_scale != default.fh_scale:
        parts.append(f"fhs={params.fh_scale!r}")
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_optimize_chips.py tests/test_chip_study.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fplopt/optimize/params.py src/fplopt/optimize/chips.py src/fplopt/backtest/policies.py tests/test_optimize_chips.py
git commit -m "feat(optimize): exclude_chips and fh_scale optimizer params"
```

---

### Task 3: Rank chip scenarios with the Free Hit shrink

**Files:**
- Modify: `src/fplopt/optimize/search.py` (`search`, `_bound_search`)
- Test: `tests/test_optimize_chips.py`

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.parametrize("how", ["bound", "all"])
def test_fh_scale_shrinks_the_free_hit_gain(how) -> None:
    """The Free Hit GW6 of test_free_hit_for_one_big_gw_reverts wins at full value and
    loses when none of its gain counts (holding the chip is worth 4)."""
    state, pool, xp, _ = team([1.0, 1.0, 1.0], other_xp=[0.0, 8.0, 0.0])
    params = chip_params(3, freehit=4.0)
    problem = state_problem(state, pool, xp, params)
    assert optimize(problem, params, top_k=1, scenario_search=how).best.scenario == "freehit@gw6"
    shrunk = replace(params, fh_scale=0.0)
    assert optimize(problem, shrunk, top_k=1, scenario_search=how).best.scenario == "none"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_optimize_chips.py -q -k fh_scale_shrinks`
Expected: FAIL (the shrunk run still picks `freehit@gw6`).

- [ ] **Step 3: Implement**

In `search.py`, add after `chip_extra`:

```python
def _score(plans: dict[int, Plan], i: int, candidates, params: OptimizerParams) -> float:
    """What scenario i's plan is ranked by: its total objective, less (1 − `fh_scale`) of
    its Free Hit gain over the no-chip plan (scenario 0) in the Free Hit GWs (never below
    0). Only lowers a scenario's value, so the search's upper bounds stay valid."""
    plan = plans[i]
    if params.fh_scale == 1.0:
        return plan.total_objective
    base = plans[0]
    gain = sum(
        plan.gws[t].objective - base.gws[t].objective
        for t, name in candidates[i].chips
        if name == "freehit"
    )
    return plan.total_objective - (1.0 - params.fh_scale) * max(0.0, gain)
```

In `search`, replace

```python
    best = max(plans, key=lambda i: (plans[i].total_objective, -i))
```

with

```python
    best = max(plans, key=lambda i: (_score(plans, i, candidates, params), -i))
```

In `_bound_search`, replace

```python
        incumbent = max(plans, key=lambda j: (plans[j].total_objective, -j))
        z = plans[incumbent].total_objective
```

with

```python
        incumbent = max(plans, key=lambda j: (_score(plans, j, candidates, params), -j))
        z = _score(plans, incumbent, candidates, params)
```

Add to the module docstring after the *Guarantee* paragraph: "With `params.fh_scale < 1` scenarios are ranked by `_score`, which never exceeds the total objective, so the bounds still prune soundly."

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_optimize_chips.py tests/test_optimize_search.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fplopt/optimize/search.py tests/test_optimize_chips.py
git commit -m "feat(optimize): rank chip scenarios with a shrunk Free Hit gain (fh_scale)"
```

---

### Task 4: Known doubles from the as-of schedule

**Files:**
- Create: `src/fplopt/backtest/chip_doubles.py`
- Test: `tests/test_chip_doubles.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Known double GWs from the as-of schedule (chip policy 2, V5)."""

import pandas as pd

from fplopt.backtest.chip_doubles import DOUBLE_LEAD, known_doubles


class FakeView:
    def __init__(self, rows):
        self.rows = rows

    def schedule(self, season):
        frame = pd.DataFrame(
            self.rows, columns=["season", "gw_index", "home_team_key", "away_team_key"]
        )
        return frame[frame["season"] == season].astype({"gw_index": "Int64"})


def test_known_doubles_within_the_lead():
    rows = [
        (2020, 5, 1, 2),
        (2020, 12, 1, 2),
        (2020, 12, 1, 3),  # club 1 twice in GW12: a double
        (2020, 30, 4, 5),
        (2020, 30, 6, 4),  # club 4 twice in GW30: beyond the lead from GW5
        (2020, None, 7, 8),  # unscheduled
        (2021, 13, 1, 2),
        (2021, 13, 1, 3),  # another season
    ]
    view = FakeView(rows)
    assert DOUBLE_LEAD == 10
    assert known_doubles(view, 2020, 5) == frozenset({12})
    assert known_doubles(view, 2020, 12) == frozenset()  # only GWs after the decision GW
    assert known_doubles(view, 2020, 25) == frozenset({30})
    assert known_doubles(view, 2020, 5, lead=30) == frozenset({12, 30})
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_chip_doubles.py -q`
Expected: FAIL (`No module named 'fplopt.backtest.chip_doubles'`).

- [ ] **Step 3: Implement** `src/fplopt/backtest/chip_doubles.py`

```python
"""Double GWs known at a deadline (chip policy 2, V5: double-aware terminal values,
docs/superpowers/specs/2026-10-30-chip-policy-2-design.md).

A GW (`gw_index`) is a known double when some club has ≥ 2 league fixtures in it in the
as-of schedule (`AsOfView.schedule`), within `lead` GWs after the decision GW. Before
2021/22 there are no fixture snapshots and the as-of schedule is the final one: the lead
caps how far ahead that hindsight reaches (the spec's V5 sanity condition covers the rest).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

__all__ = ("DOUBLE_LEAD", "known_doubles", "known_doubles_by_gw")

DOUBLE_LEAD = 10  # GWs after the decision GW (pre-registered in the spec)


def known_doubles(
    view: Any, season: int, gw_index: int, lead: int = DOUBLE_LEAD
) -> frozenset[int]:
    """The `gw_index`s in (gw_index, gw_index + lead] where a club has ≥ 2 fixtures in
    `view.schedule(season)`."""
    schedule = view.schedule(season)
    schedule = schedule[schedule["gw_index"].notna()]
    sides = pd.concat(
        [
            schedule[["gw_index", side]].rename(columns={side: "team_key"})
            for side in ("home_team_key", "away_team_key")
        ]
    )
    sides = sides[sides["team_key"] > 0]
    counts = sides.groupby(["gw_index", "team_key"]).size()
    doubles = {int(g) for (g, _), n in counts.items() if n >= 2}
    return frozenset(g for g in doubles if gw_index < g <= gw_index + lead)


def known_doubles_by_gw(
    store: Any, seasons: Iterable[int], lead: int = DOUBLE_LEAD
) -> dict[tuple[int, int], frozenset[int]]:
    """(season, gw_index) → `known_doubles` at that GW's deadline, for the estimator."""
    gameweeks = store.table("gameweek")
    out = {}
    for season in seasons:
        rows = gameweeks[gameweeks["season"] == season].sort_values("gw_index")
        for gw_index, deadline in zip(rows["gw_index"], rows["deadline_time"], strict=True):
            view = store.as_of(deadline)
            out[(int(season), int(gw_index))] = known_doubles(view, season, int(gw_index), lead)
    return out
```

(Check `DataStore` has a `table(name)` method for whole tables; if not, read `pd.read_parquet(data_dir / "gameweek.parquet")` in the CLI and pass the frame instead: change the parameter to `gameweeks: pd.DataFrame` plus `store` for `as_of`.)

- [ ] **Step 4: Add a synthetic-store test** (append to `tests/test_chip_doubles.py`)

```python
def test_known_doubles_by_gw_on_a_synthetic_store():
    from synthetic_season import synthetic_tables

    from fplopt.backtest.chip_doubles import known_doubles_by_gw
    from fplopt.features.store import DataStore

    store = DataStore(tables=synthetic_tables(seasons=(2022, 2023)))
    table = known_doubles_by_gw(store, [2023])
    assert table and all(season == 2023 for season, _ in table)
    assert sorted(gw for _, gw in table) == list(range(1, len(table) + 1))
    assert all(isinstance(v, frozenset) for v in table.values())
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_chip_doubles.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/fplopt/backtest/chip_doubles.py tests/test_chip_doubles.py
git commit -m "feat(backtest): double GWs known at a deadline from the as-of schedule"
```

---

### Task 5: Per-window, double-split and leave-one-season-out estimation

**Files:**
- Modify: `src/fplopt/backtest/chip_values.py`
- Test: `tests/test_chip_study.py` (after `test_estimate_terminal_values_refuses_...`)

- [ ] **Step 1: Write the failing tests**

```python
def test_estimate_terminal_values_by_window():
    table = estimate_terminal_values(gains_frame(), split="window")
    assert set(table) == {f"{c}@{k}" for c in ("3xc", "bboost", "freehit", "wildcard") for k in (1, 2)}
    # TC gain = gw_index: set 1's best of its last L GWs is 19, set 2's is 38.
    assert table["3xc@1"] == (19.0,) * 19
    assert table["3xc@2"] == (38.0,) * 19
    assert len(table["freehit@1"]) == 18 and len(table["freehit@2"]) == 19
    # The pooled table is unchanged by default.
    assert estimate_terminal_values(gains_frame())["3xc"][0] == pytest.approx((19 + 38) / 2)


def test_estimate_terminal_values_split_by_known_doubles():
    # A double known (at every deadline of 2020) in GW15 only.
    doubles = {(2020, gw): frozenset({15}) for gw in range(1, 39)}
    frames = pd.concat([gains_frame(start_id=str(i)) for i in range(40)])  # n ≥ MIN_CELL
    table = estimate_terminal_values(frames, split="double", doubles=doubles, horizon=6)
    # Set 1, L GWs left = GWs 20-L..19: GW15 is among them for L >= 5.
    assert table["3xc@1+1"][4] == 19.0 and table["3xc@1+0"][3] == 19.0
    # Sparse cells fall back to the window value: set 2 never has a double ahead.
    assert table["3xc@2+1"] == table["3xc@2"]
    with pytest.raises(ValueError, match="doubles"):
        estimate_terminal_values(frames, split="double")


def test_estimate_loso_leaves_each_season_out():
    from fplopt.backtest.chip_values import estimate_loso

    two = pd.concat([gains_frame(season=2019), gains_frame(season=2020, values=lambda gw: 0.0)])
    tables = estimate_loso(two, split="window")
    assert set(tables) == {2019, 2020}
    assert tables[2019]["3xc@1"][0] == 0.0  # estimated on 2020 only
    assert tables[2020]["3xc@1"][0] == 19.0  # on 2019 only
    with pytest.raises(ValueError, match="two seasons"):
        estimate_loso(gains_frame(), split="window")


def test_load_loso_values(tmp_path):
    from fplopt.backtest.chip_values import load_loso_values, load_terminal_values

    path = tmp_path / "t.json"
    path.write_text(
        json.dumps(
            {
                "format": 2,
                "split": "window",
                "values": {"3xc@1": [2.0]},
                "loso": {"2019": {"3xc@1": [1.0]}, "2020": {"3xc@1": [3.0]}},
            }
        ),
        encoding="utf-8",
    )
    assert load_loso_values(0.5, path) == {2019: {"3xc@1": (0.5,)}, 2020: {"3xc@1": (1.5,)}}
    assert load_terminal_values(1.0, path) == {"3xc@1": (2.0,)}
    plain = tmp_path / "p.json"
    plain.write_text(json.dumps({"values": {"3xc": [1.0]}}), encoding="utf-8")
    with pytest.raises(ValueError, match="no leave-one-season-out"):
        load_loso_values(1.0, plain)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_chip_study.py -q -k "estimate or loso"`
Expected: the new tests FAIL (`unexpected keyword argument 'split'`, missing functions).

- [ ] **Step 3: Implement** in `src/fplopt/backtest/chip_values.py`

Update `__all__` to

```python
__all__ = (
    "CHIP_VALUES_FILE",
    "DOUBLE_VALUES_FILE",
    "GAIN_COLUMNS",
    "MIN_CELL",
    "SPLITS",
    "WINDOW_VALUES_FILE",
    "estimate_loso",
    "estimate_terminal_values",
    "load_loso_values",
    "load_terminal_values",
)
```

Add after `CHIP_VALUES_FILE`:

```python
# Chip policy 2 (docs/superpowers/specs/2026-10-30-chip-policy-2-design.md): per-window
# and per-window-and-double tables, each with a leave-one-season-out section.
WINDOW_VALUES_FILE = CHIP_VALUES_FILE.with_name("terminal-values-window.json")
DOUBLE_VALUES_FILE = CHIP_VALUES_FILE.with_name("terminal-values-double.json")
SPLITS = ("pooled", "window", "double")
# A double-split cell with fewer (run, window) observations falls back to its window value.
MIN_CELL = 30
```

Add after `load_terminal_values`:

```python
def load_loso_values(
    shrink: float = 1.0, path: Path | None = None
) -> dict[int, dict[str, tuple[float, ...]]]:
    """The leave-one-season-out tables of a v2 file (season → table, each × `shrink`):
    season s's table was estimated without s. ValueError if the file has none."""
    path = WINDOW_VALUES_FILE if path is None else path
    if not math.isfinite(shrink) or shrink < 0:
        raise ValueError(f"chip_shrink must be a finite number >= 0, got {shrink}")
    if not path.is_file():
        raise ValueError(f"loso: no estimated chip values at {path} (fplopt chips values)")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "loso" not in payload:
        raise ValueError(f"{path}: no leave-one-season-out tables (chips values --loso)")
    return {
        int(season): {name: tuple(shrink * float(v) for v in values) for name, values in t.items()}
        for season, t in payload["loso"].items()
    }
```

Replace `estimate_terminal_values` with:

```python
def estimate_terminal_values(
    gws: pd.DataFrame,
    split: str = "pooled",
    doubles: Mapping[tuple[int, int], frozenset[int]] | None = None,
    horizon: int = 6,
) -> dict[str, tuple[float, ...]]:
    """Per table key, the values for L = 1 .. the longest window (module docstring).
    `split`: "pooled" keys by chip name (both windows together); "window" by `name@k`, k the
    window's ordinal; "double" by `name@k+d` as well as `name@k`, d = 1 when a double was
    known (`doubles[(season, decision gw_index)]`, `chip_doubles.known_doubles_by_gw`) in the
    window's last L GWs at the decision GW max(1, stop − L − horizon + 1); a `+d` cell with
    fewer than `MIN_CELL` observations takes the `name@k` value."""
    from fplopt.backtest.rules import backtest_rules

    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if split == "double" and doubles is None:
        raise ValueError("split='double' needs the known doubles (chip_doubles)")
    seasons = sorted(int(s) for s in gws["season"].unique())
    outside = [s for s in seasons if s not in DEVELOP_SEASONS]
    if outside:
        raise ValueError(f"chip values are estimated on develop seasons only, got {outside}")
    missing = [c for c in GAIN_COLUMNS.values() if c not in gws.columns]
    if missing or gws[list(GAIN_COLUMNS.values())].isna().all().all():
        raise ValueError("no chip gains in these GW rows: run `backtest run --spec chipgains:...`")
    rules = backtest_rules(max(DEVELOP_SEASONS))
    sums: dict[tuple[str, int], float] = defaultdict(float)
    counts: dict[tuple[str, int], int] = defaultdict(int)
    longest: dict[str, int] = {}
    for name, column in GAIN_COLUMNS.items():
        windows = sorted((w.start, w.stop) for w in rules.chips if w.name == name)
        for (season, _, _), run in gws.groupby(["season", "policy", "start_id"], sort=True):
            gains = {
                int(i): float(g)
                for i, g in zip(run["gw_index"], run[column], strict=True)
                if not pd.isna(g)
            }
            for k, (start, stop) in enumerate(windows, start=1):
                keys = [name] if split == "pooled" else [f"{name}@{k}"]
                for key in keys:
                    longest[key] = max(longest.get(key, 0), stop - start + 1)
                best = -math.inf
                for left in range(1, stop - start + 2):  # GWs left: stop-left+1 .. stop
                    gw = stop - left + 1
                    if gw in gains:
                        best = max(best, gains[gw])
                    if best == -math.inf:
                        continue
                    cell = list(keys)
                    if split == "double":
                        decision = max(1, stop - left - horizon + 1)
                        ahead = doubles.get((int(season), decision), frozenset())
                        d = int(any(stop - left < g <= stop for g in ahead))
                        cell.append(f"{name}@{k}+{d}")
                        for dd in (0, 1):
                            longest.setdefault(f"{name}@{k}+{dd}", stop - start + 1)
                    for key in cell:
                        sums[key, left] += best
                        counts[key, left] += 1
    out = {}
    for key, length in longest.items():
        parent = key.split("+")[0] if "+" in key else None
        values: list[float] = []
        for left in range(1, length + 1):
            n = counts.get((key, left), 0)
            if parent is not None and n < MIN_CELL:
                n_parent = counts.get((parent, left), 0)
                value = sums[parent, left] / n_parent if n_parent else None
            else:
                value = sums[key, left] / n if n else None
            values.append(value if value is not None else (values[-1] if values else 0.0))
        out[key] = tuple(max(0.0, v) for v in values)
    return out


def estimate_loso(
    gws: pd.DataFrame,
    split: str = "window",
    doubles: Mapping[tuple[int, int], frozenset[int]] | None = None,
    horizon: int = 6,
) -> dict[int, dict[str, tuple[float, ...]]]:
    """Season s → `estimate_terminal_values` on every other season's rows: the table a
    backtest of season s uses, so no season's own outcomes inform its holding values."""
    seasons = sorted(int(s) for s in gws["season"].unique())
    if len(seasons) < 2:
        raise ValueError(f"leave-one-season-out needs at least two seasons, got {seasons}")
    return {
        s: estimate_terminal_values(gws[gws["season"] != s], split, doubles, horizon)
        for s in seasons
    }
```

Add imports at the top: `from collections import defaultdict` and `from collections.abc import Mapping`.

Note on behaviour: the pooled table must equal the old one. The old code took each chip's longest window over windows in rules order; this code does the same per key. Keys' insertion order is by chip, then window: for "pooled" the dict order is `GAIN_COLUMNS` order, as before.

- [ ] **Step 4: Run tests (old and new)**

Run: `uv run pytest tests/test_chip_study.py tests/test_chip_diagnostics.py -q`
Expected: PASS, including `test_estimate_terminal_values` and the diagnostics' pin against `estimate_terminal_values`.

- [ ] **Step 5: Commit**

```bash
git add src/fplopt/backtest/chip_values.py tests/test_chip_study.py
git commit -m "feat(chips): per-window, double-split and leave-one-season-out terminal values"
```

---

### Task 6: OptimizerPolicy — per-season tables, known doubles, rule Triple Captain

**Files:**
- Modify: `src/fplopt/backtest/policies.py` (`_optimizer_extras` checksum → `_table_tag`; `OptimizerPolicy`)
- Test: `tests/test_chip_study.py`

- [ ] **Step 1: Write the failing tests** (use the module's `league` fixture and `contexts` helper)

```python
def test_optimizer_policy_uses_the_seasons_loso_table(league):
    view, pool, xp = league[8]
    rules = backtest_rules(2023)
    state = random_state(view, rules, 3)
    ctx = DecisionContext(view, state, rules, pool, xp)
    params = replace(SMALL, chip_value={})
    tables = {2023: {"wildcard": (500.0,)}, 2022: {"wildcard": (0.0,)}}
    policy = OptimizerPolicy("rolling", params, chips=True, season_tables=tables)
    assert "loso#" in policy.name
    # Holding a Wildcard is worth 500 in 2023's table: never played.
    assert policy.decide(ctx).chip != "wildcard"
    missing = OptimizerPolicy("rolling", params, chips=True, season_tables={2022: {}})
    with pytest.raises(ValueError, match="no leave-one-season-out table for season 2023"):
        missing.decide(ctx)


def test_optimizer_policy_rule_triple_captain(league):
    params = replace(SMALL, exclude_chips=("3xc",))
    always = OptimizerPolicy("rolling", params, chips=True, tc_rule=0.0)
    never = OptimizerPolicy("rolling", params, chips=True, tc_rule=1e9)
    assert "tcr=0.0" in always.name
    assert any(d.chip == "3xc" for _, d in contexts(league, always))
    assert all(d.chip != "3xc" for _, d in contexts(league, never))
    with pytest.raises(ValueError, match="tc_rule needs"):
        OptimizerPolicy("rolling", SMALL, chips=True, tc_rule=8.3)  # 3xc not excluded


def test_optimizer_policy_fills_known_doubles(league, monkeypatch):
    from fplopt.backtest import policies

    seen = []
    monkeypatch.setattr(
        policies, "known_doubles", lambda view, season, gw, lead: seen.append(lead) or frozenset()
    )
    view, pool, xp = league[8]
    rules = backtest_rules(2023)
    ctx = DecisionContext(view, random_state(view, rules, 3), rules, pool, xp)
    policy = OptimizerPolicy("rolling", SMALL, chips=True, double_lead=10)
    assert "dl=10" in policy.name
    policy.decide(ctx)
    assert seen == [10]
```

(Check `contexts` yields `(ctx, decision)` pairs and applies decisions as in the existing tests; adapt the call if its signature differs.)

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_chip_study.py -q -k "optimizer_policy"`
Expected: FAIL (`unexpected keyword argument 'season_tables'`).

- [ ] **Step 3: Implement**

In `policies.py`, add the import `from fplopt.backtest.chip_doubles import known_doubles` with the other `fplopt.backtest` imports, and a helper above `_optimizer_extras`:

```python
def _table_tag(table: Mapping[str, Sequence[float]]) -> str:
    """First-last value and a weighted checksum per key, so a re-estimated table gives a
    different policy name (the experiment log keys on names)."""
    return "/".join(
        f"{k}:{v[0]:.2f}-{v[-1]:.2f}#{sum((i + 1) * x for i, x in enumerate(v)):.3f}"
        for k, v in table.items()
    )
```

and in `_optimizer_extras` replace the `chip_value_left` block with:

```python
    if params.chip_value_left is not None:
        parts.append("cvl=" + _table_tag(params.chip_value_left))
```

(`Sequence` from `collections.abc`; add the import if missing.)

Replace `OptimizerPolicy`'s fields, `__post_init__`, `name` and `decide` with:

```python
    xp_model: str = "ep_next"
    params: OptimizerParams = field(default_factory=OptimizerParams)
    chips: bool = False
    # Chip policy 2: season → terminal value table (leave-one-season-out), replacing
    # `params.chip_value_left` in that season's decisions; the known-double lookahead for
    # double-aware tables (None = no doubles); Triple Captain by rule at this captain xP
    # (needs "3xc" in `params.exclude_chips`; None = the optimizer plays it).
    season_tables: Mapping[int, Mapping[str, tuple[float, ...]]] | None = None
    double_lead: int | None = None
    tc_rule: float | None = None

    def __post_init__(self) -> None:
        _check_model(self.xp_model)
        if not isinstance(self.params, OptimizerParams):
            raise TypeError(f"params must be OptimizerParams, got {type(self.params).__name__}")
        _check_horizon(self.params.horizon)
        if self.tc_rule is not None and (
            not self.chips or "3xc" not in self.params.exclude_chips
        ):
            raise ValueError("tc_rule needs chips and '3xc' in params.exclude_chips")
        if self.double_lead is not None and self.double_lead < 1:
            raise ValueError(f"double_lead must be >= 1, got {self.double_lead}")

    @property
    def name(self) -> str:
        params = self.params
        max_hits = "inf" if params.max_hits is None else str(params.max_hits)
        parts = [self.xp_model, f"mh={max_hits}", f"m={float(params.hit_margin)!r}"]
        parts += _optimizer_extras(params)
        if self.season_tables is not None:
            parts.append(
                "loso#"
                + ";".join(f"{s}:{_table_tag(t)}" for s, t in sorted(self.season_tables.items()))
            )
        if self.double_lead is not None:
            parts.append(f"dl={self.double_lead}")
        if self.tc_rule is not None:
            parts.append(f"tcr={float(self.tc_rule)!r}")
        if self.chips:
            parts.append("chips")
        return f"optimizer({','.join(parts)})"

    def decide(self, ctx: DecisionContext) -> Decision:
        params = self.params
        if self.season_tables is not None:
            table = self.season_tables.get(ctx.state.season)
            if table is None:
                raise ValueError(
                    f"no leave-one-season-out table for season {ctx.state.season}"
                )
            params = replace(params, chip_value_left=table)
        problem = PlanInput.from_context(ctx.state, ctx.pool, ctx.xp, ctx.rules, params)
        if self.double_lead is not None:
            doubles = known_doubles(
                ctx.view, ctx.state.season, ctx.state.gw_index, self.double_lead
            )
            problem = replace(problem, doubles=doubles)
        plans = optimize(problem, params, top_k=1, chips=self.chips, roll=False)
        status = plans.solver_status
        if status != "Optimal":
            log.warning(
                "%s at %s gw_index %d: solver stopped at a limit (%s, gap %.2f%%); "
                "executing the best plan found",
                self.name,
                ctx.view.deadline,
                ctx.state.gw_index,
                status,
                100 * plans.best.mip_gap,
            )
        decision = replace(plans.decision(), solver_status=status, mip_gap=plans.best.mip_gap)
        if self.tc_rule is not None and decision.chip is None:
            return self._rule_tc(ctx, problem, params, plans) or decision
        return decision

    def _rule_tc(self, ctx, problem, params, plans) -> Decision | None:
        """Triple Captain this GW when available and plan #1's captain's xP ≥ `tc_rule` or
        this is its window's last GW: re-planned with TC fixed here (None otherwise, or if
        that plan is infeasible)."""
        state, rules = ctx.state, ctx.rules
        if not chip_available(state, "3xc", rules):
            return None
        stop = {w.chip_id: w.stop for w in rules.chips}[chip_window(state, "3xc", rules)]
        captain = _target_xp_of(ctx.xp, plans.best.first.captain)
        if captain < self.tc_rule and state.gw_index != stop:
            return None
        try:
            plan = solve_plan(problem, params, chips={0: "3xc"})
        except InfeasiblePlan:
            log.warning("%s: no plan with 3xc at gw_index %d", self.name, state.gw_index)
            return None
        return replace(plan.decision(), solver_status=plan.status, mip_gap=plan.mip_gap)
```

`_target_xp_of` is defined further down the module (chip study section); Python resolves it at call time, so no reordering is needed. `solve_plan`, `InfeasiblePlan`, `chip_available`, `chip_window` are already imported for `ChipRulePolicy`.

Update the class docstring with a sentence: "Chip policy 2 options: `season_tables`, `double_lead`, `tc_rule` (field comments)."

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_chip_study.py tests/test_backtest_policies.py -q`
Expected: PASS. Existing policy names are unchanged (the `_table_tag` refactor produces the same string: compare `OptimizerPolicy("v1", OptimizerParams(chip_value_left={"3xc": (1.0, 2.0)}), chips=True).name` before and after in a quick REPL check, `uv run python -c ...`).

- [ ] **Step 5: Commit**

```bash
git add src/fplopt/backtest/policies.py tests/test_chip_study.py
git commit -m "feat(backtest): optimizer policy with loso tables, known doubles and a rule TC"
```

---

### Task 7: CLI — spec keys and `chips values --split --loso`

**Files:**
- Modify: `src/fplopt/cli.py` (`POLICY_PARAMS["optimizer"]`, `_OPTIMIZER_KEYS`, `PolicySpec.build`, `estimated_chip_values`, `make_policy_spec` checks, the run config block at ~L522, `_chips_values`, the `values` parser)
- Test: `tests/test_chip_study.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_cli_chip_policy_2_specs(tmp_path, monkeypatch):
    window = tmp_path / "w.json"
    window.write_text(
        json.dumps(
            {
                "format": 2,
                "split": "window",
                "values": {"3xc@1": [4.0]},
                "loso": {"2019": {"3xc@1": [2.0]}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(cli.CHIP_TABLE_FILES, 1, window)
    monkeypatch.setitem(cli.CHIP_TABLE_FILES, 2, window)
    base = "horizon=6,decay=0.75,max_hits=1,hit_margin=2,chips=1,chip_shrink=0.5"
    spec = cli.parse_policy_spec(f"optimizer:v1:{base},chip_split=1,loso=1").build()
    assert spec.season_tables == {2019: {"3xc@1": (1.0,)}}
    pooled = cli.parse_policy_spec(f"optimizer:v1:{base},chip_split=1").build()
    assert pooled.params.chip_value_left == {"3xc@1": (2.0,)}
    double = cli.parse_policy_spec(f"optimizer:v1:{base},chip_split=2,loso=1").build()
    assert double.double_lead == 10
    hybrid = cli.parse_policy_spec(f"optimizer:v1:{base},chip_split=1,loso=1,tc_rule=8.3").build()
    assert hybrid.tc_rule == 8.3 and hybrid.params.exclude_chips == ("3xc",)
    fh = cli.parse_policy_spec(f"optimizer:v1:{base},chip_split=1,loso=1,fh_scale=0.4").build()
    assert fh.params.fh_scale == 0.4
    for bad in ("chip_split=1", "loso=1", "chip_split=3,chip_shrink=1"):
        with pytest.raises(argparse.ArgumentTypeError):
            cli.parse_policy_spec(f"optimizer:v1:chips=1,{bad}")


def test_cli_chips_values_loso(tmp_path):
    run = tmp_path / "s0"
    run.mkdir()
    pd.concat([gains_frame(season=2019), gains_frame(season=2020)]).to_parquet(
        run / "gws.parquet"
    )
    out = tmp_path / "t.json"
    assert cli.main(["chips", "values", "--gains", str(run), "--split", "window", "--loso", "--out", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["format"] == 2 and payload["split"] == "window"
    assert set(payload["loso"]) == {"2019", "2020"} and "3xc@1" in payload["values"]
```

(Check the names `cli.parse_policy_spec` and `cli.main` against the existing chip-study CLI tests in this file — e.g. the test that builds `chip_shrink` specs — and use the same entry points; add `import argparse` at the top of the test module if missing.)

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_chip_study.py -q -k cli_chip`
Expected: FAIL (unknown parameter `chip_split`).

- [ ] **Step 3: Implement**

In `POLICY_PARAMS["optimizer"]` after `"chip_shrink": float,`:

```python
        # chip policy 2 (docs/superpowers/specs/2026-10-30-chip-policy-2-design.md):
        # the table file (0 pooled, 1 per window, 2 per window and known double), its
        # leave-one-season-out tables (loso=1), Triple Captain by rule at this captain
        # xP, and the share of a Free Hit's gain that counts in the chip search
        "chip_split": int,
        "loso": parse_flag,
        "tc_rule": float,
        "fh_scale": float,
```

Change `_OPTIMIZER_KEYS`' filter to leave out the chip keys:

```python
_CHIP_KEYS = ("chips", "chip_shrink", "chip_split", "loso", "tc_rule", "fh_scale")
_OPTIMIZER_KEYS = {k: v for k, v in POLICY_PARAMS["optimizer"].items() if k not in _CHIP_KEYS}
```

After `CHIP_VALUES_FILE = ...` add:

```python
# chip_split → table file (patched in tests); 1 and 2 are chip policy 2's.
CHIP_TABLE_FILES: dict[int, Path] = {
    0: CHIP_VALUES_FILE,
    1: CHIP_VALUES_FILE.with_name("terminal-values-window.json"),
    2: CHIP_VALUES_FILE.with_name("terminal-values-double.json"),
}
```

In `PolicySpec.build`, replace the `if self.name == "optimizer":` block with:

```python
        if self.name == "optimizer":
            from fplopt.backtest.chip_doubles import DOUBLE_LEAD
            from fplopt.optimize import OptimizerParams

            chips = bool(params.pop("chips", 0))
            shrink = params.pop("chip_shrink", None)
            split = int(params.pop("chip_split", 0))
            loso = bool(params.pop("loso", 0))
            tc_rule = params.pop("tc_rule", None)
            extra: dict[str, Any] = {}
            if split not in CHIP_TABLE_FILES:
                raise ValueError(f"chip_split must be one of {sorted(CHIP_TABLE_FILES)}")
            if (split or loso) and shrink is None:
                raise ValueError("chip_split and loso need chip_shrink")
            if shrink is not None:
                if not chips:
                    raise ValueError("chip_shrink needs chips=1")
                path = CHIP_TABLE_FILES[split]
                if loso:
                    extra["season_tables"] = estimated_loso_values(float(shrink), path)
                else:
                    params["chip_value_left"] = estimated_chip_values(float(shrink), path)
            if split == 2:
                extra["double_lead"] = DOUBLE_LEAD
            if tc_rule is not None:
                params["exclude_chips"] = ("3xc",)
                extra["tc_rule"] = float(tc_rule)
            return OptimizerPolicy(self.xp, OptimizerParams(**params), chips=chips, **extra)
```

Replace `estimated_chip_values` with:

```python
def estimated_chip_values(shrink: float, path: Path | None = None) -> dict[str, tuple[float, ...]]:
    """`chip_values.load_terminal_values` from `path` (default `CHIP_VALUES_FILE`, patched
    in tests)."""
    from fplopt.backtest.chip_values import load_terminal_values

    return load_terminal_values(shrink, CHIP_VALUES_FILE if path is None else path)


def estimated_loso_values(shrink: float, path: Path) -> dict[int, dict[str, tuple[float, ...]]]:
    """`chip_values.load_loso_values` from `path`."""
    from fplopt.backtest.chip_values import load_loso_values

    return load_loso_values(shrink, path)
```

In the run-config block (~L522), log every table file a spec used:

```python
    used = {
        int(dict(spec.params).get("chip_split", 0))
        for spec in args.policy_specs
        if "chip_shrink" in dict(spec.params)
    }
    if used:
        # The estimated terminal values the chip arms ran with (chip pre-registration §3).
        config["chip_values"] = {
            str(split): json.loads(CHIP_TABLE_FILES[split].read_text(encoding="utf-8"))
            for split in sorted(used)
        }
```

(This changes `config["chip_values"]` from one table to a dict by split; check any test asserting on it and update it to `config["chip_values"]["0"]`.)

Replace `_chips_values` with:

```python
def _chips_values(c: Context) -> object:
    """`fplopt chips values`: the terminal value table (`fplopt.backtest.chip_values`) from a
    chipgains run's GW rows, written as JSON with its source and seasons; `--split window|
    double` keys it by window (and known double; reads `data/`), `--loso` adds the
    leave-one-season-out tables (format 2)."""
    import pandas as pd

    from fplopt.backtest import chip_values

    args = c.args
    gws = pd.read_parquet(args.gains / "gws.parquet")
    seasons = sorted(int(s) for s in gws["season"].unique())
    doubles = None
    if args.split == "double":
        from fplopt.backtest.chip_doubles import known_doubles_by_gw
        from fplopt.features.store import DataStore

        doubles = known_doubles_by_gw(DataStore(c.settings.data_dir), seasons)
    table = chip_values.estimate_terminal_values(gws, args.split, doubles)
    payload: dict[str, Any] = {
        "source": str(args.gains),
        "seasons": [season_label(s) for s in seasons],
        "note": "mean best ex-ante one-GW gain of each chip over the GWs left in its window "
        "(index = GWs left - 1); unshrunk (the policy spec's chip_shrink scales it)",
    }
    if args.split != "pooled" or args.loso:
        payload |= {"format": 2, "split": args.split}
    payload["values"] = {name: [round(v, 4) for v in values] for name, values in table.items()}
    if args.loso:
        loso = chip_values.estimate_loso(gws, args.split, doubles)
        payload["loso"] = {
            str(s): {name: [round(v, 4) for v in values] for name, values in t.items()}
            for s, t in loso.items()
        }
    out = args.out or CHIP_TABLE_FILES[chip_values.SPLITS.index(args.split)]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for name, values in table.items():
        shown = ", ".join(f"{v:.1f}" for v in values[:6])
        print(f"{name}: L=1..6 {shown} ... L={len(values)} {values[-1]:.1f}", flush=True)
    print(f"Written to {out}", flush=True)
    return table
```

(Check how `Context` exposes settings: `c.settings` is used by `Context.odds`; `data_dir` is `Settings.data_dir`.)

In the `values` parser: change `--out`'s default to `None` with help "where to write the table (default: config/chips/terminal-values[-window|-double].json by --split)", and add:

```python
    values.add_argument(
        "--split",
        choices=["pooled", "window", "double"],
        default="pooled",
        help="table keys: chip (pooled), chip@window, chip@window+double (reads data/)",
    )
    values.add_argument(
        "--loso",
        action="store_true",
        help="add leave-one-season-out tables (season s estimated without s)",
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_chip_study.py tests/test_cli.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fplopt/cli.py tests/test_chip_study.py
git commit -m "feat(cli): chip policy 2 spec keys and chips values --split/--loso"
```

---

### Task 8: `WindowTTest.se` and the comparison driver

**Files:**
- Modify: `src/fplopt/backtest/evaluate.py` (`WindowTTest`, `window_t_test`)
- Create: `dev/chip_policy_2.py`
- Test: `tests/test_backtest_evaluate.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_backtest_evaluate.py`)

```python
def test_window_t_test_reports_the_standard_error():
    import math

    import pandas as pd

    from fplopt.backtest.evaluate import window_t_test

    diffs = pd.DataFrame(
        {"season": [2020] * 4, "gw_index": [1, 2, 3, 4], "diff": [1.0, 2.0, 3.0, 4.0]}
    )
    result = window_t_test(diffs)
    assert result.se == pytest.approx(math.sqrt(1.6666666666666667 / 4))
    assert math.isnan(window_t_test(diffs.iloc[:1]).se)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_backtest_evaluate.py -q -k standard_error`
Expected: FAIL (`'WindowTTest' object has no attribute 'se'`).

- [ ] **Step 3: Implement**

In `evaluate.py`:

```python
@dataclass(frozen=True)
class WindowTTest:
    """`window_t_test`: the mean over cells, the one-sided p (H1: mean > 0), the cells and
    the mean's standard error (NaN with fewer than 2 cells)."""

    mean: float
    p_one_sided: float
    n_cells: int
    se: float = float("nan")
```

and in `window_t_test` return `se`:

```python
    n, mean = len(cells), float(cells.mean())
    sd = float(np.std(cells, ddof=1)) if n >= 2 else 0.0
    se = sd / math.sqrt(n) if n >= 2 else float("nan")
    if n < 2 or sd == 0:
        return WindowTTest(mean, float("nan"), n, se)
    return WindowTTest(mean, t_sf(mean / se, n - 1), n, se)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_backtest_evaluate.py tests/test_chip_study.py -q`
Expected: PASS.

- [ ] **Step 5: Write the driver** `dev/chip_policy_2.py`

```python
"""Chip policy 2 comparisons (spec docs/superpowers/specs/2026-10-30-chip-policy-2-design.md):
each arm vs H by the full-run window t-test on 8-GW cells, deflated for N = 6 variants;
the best of V1-V3; V5's sanity condition on 2021/22-2022/23.

    uv run python dev/chip_policy_2.py --h results/cp2-h \
        --arm P=results/cp2-p --arm V1=results/cp2-v1 ... [--out results/chip-policy-2]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from fplopt.backtest import chip_diagnostics as cd
from fplopt.backtest.evaluate import full_run_window_test

N_VARIANTS = 6
SNAPSHOT_SEASONS = (2021, 2022)  # develop seasons with fixture snapshots (V5 sanity check)


def load(run: Path) -> pd.DataFrame:
    return pd.read_parquet(run / "gws.parquet")


def compare(a: pd.DataFrame, b: pd.DataFrame) -> dict[str, float]:
    m = cd.paired(a, b)
    real, xg = full_run_window_test(m, "diff"), full_run_window_test(m, "diff_xg")
    return {
        "mean": real.mean,
        "se": real.se,
        "p_one_sided": real.p_one_sided,
        "n_cells": real.n_cells,
        "deflated": real.mean - real.se * math.sqrt(2 * math.log(N_VARIANTS)),
        "xg_mean": xg.mean,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=Path, required=True)
    ap.add_argument("--arm", action="append", default=[], help="NAME=RUN_DIR")
    ap.add_argument("--out", type=Path, default=Path("results/chip-policy-2"))
    args = ap.parse_args()
    h = load(args.h)
    arms = {name: load(Path(path)) for name, path in (a.split("=", 1) for a in args.arm)}
    out = {"vs H": {name: compare(g, h) for name, g in arms.items()}}
    first = [v for v in ("V1", "V2", "V3") if v in arms]
    if first:
        # Ties: the smaller shrink, i.e. the earlier of V1-V3.
        best = max(first, key=lambda v: (out["vs H"][v]["mean"], -first.index(v)))
        out["best_of_v1_v3"] = best
        if "V5" in arms:
            snap = [arms[x][arms[x]["season"].isin(SNAPSHOT_SEASONS)] for x in ("V5", best)]
            out["v5_sanity_vs_best"] = compare(*snap)
    passing = {
        name: r
        for name, r in out["vs H"].items()
        if name.startswith("V") and r["deflated"] > 0 and r["p_one_sided"] < 0.10 and r["xg_mean"] > 0
    }
    if "V5" in passing and out.get("v5_sanity_vs_best", {}).get("mean", 0) <= 0:
        passing.pop("V5")
    out["passing"] = sorted(passing, key=lambda n: -passing[n]["deflated"])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for name, r in out["vs H"].items():
        print(
            f"{name} vs H: {r['mean']:+.3f}/GW (se {r['se']:.3f}, deflated {r['deflated']:+.3f}), "
            f"p {r['p_one_sided']:.4f}, xG {r['xg_mean']:+.3f}"
        )
    print(f"best of V1-V3: {out.get('best_of_v1_v3')}; passing: {out['passing'] or 'none'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Smoke-test the driver on the diagnostics' runs**

Run: `uv run python dev/chip_policy_2.py --h results/chips-diag-h --arm P=results/chips-diag-p --out results/cp2-smoke`
Expected: prints `P vs H: +0.33…/GW` (the diagnostics' number) and `passing: none`.

- [ ] **Step 7: Commit**

```bash
git add src/fplopt/backtest/evaluate.py tests/test_backtest_evaluate.py dev/chip_policy_2.py
git commit -m "feat(research): window t-test standard error; chip policy 2 comparison driver"
```

---

### Task 9: Estimate the tables; docs

**Files:**
- Create: `config/chips/terminal-values-window.json`, `config/chips/terminal-values-double.json`
- Modify: `CLAUDE.md` (chip study bullet), `docs/PLAN.md` (decision log row after 2026-10-30)

- [ ] **Step 1: Estimate both tables from the S0 develop run** (local `results/chips-diag-s0`, `data/` present)

```bash
uv run fplopt chips values --gains results/chips-diag-s0 --split window --loso
uv run fplopt chips values --gains results/chips-diag-s0 --split double --loso
```

Expected: each prints the per-key values and `Written to config/chips/terminal-values-{window,double}.json`. Sanity: `3xc@2`'s values above `3xc@1`'s for short L (the diagnostics found set-2 holding undervalued by the pooled table).

- [ ] **Step 2: Check a spec builds from the files**

Run: `uv run python -c "from fplopt import cli; p = cli.make_policy_spec('optimizer', 'v1', {'chips': '1', 'chip_shrink': '0.8', 'chip_split': '2', 'loso': '1'}).build(); print(sorted(p.season_tables), p.double_lead)"`
Expected: `[2016, 2017, 2018, 2019, 2020, 2021, 2022] 10`.

- [ ] **Step 3: Docs**

`CLAUDE.md`, in the *Chip study* bullet list, add:

```markdown
  - chip policy 2 (`docs/superpowers/specs/2026-10-30-chip-policy-2-design.md`): optimizer keys `chip_split` (0 pooled table, 1 per window `config/chips/terminal-values-window.json`, 2 per window and known double `…-double.json`), `loso=1` (leave-one-season-out tables), `tc_rule=X` (Triple Captain by rule), `fh_scale=k` (Free Hit gain share in the chip search); `chips values --split window|double --loso` estimates them; `dev/chip_policy_2.py` compares runs vs H.
```

`docs/PLAN.md`, decision log, after the last 2026-10-30 row:

```markdown
| 2026-10-30 | Chip policy 2 pre-registered (`docs/superpowers/specs/2026-10-30-chip-policy-2-design.md`): six variants vs H on develop (per-window terminal values at k 0.6/0.8/1.0 estimated leave-one-season-out; then a rule-TC hybrid, double-aware values, a Free Hit shrink), deflated for N = 6; the winner is frozen before GW20 and shadowed live over set 2. | The diagnostics (`docs/research/2026-10-chip-diagnostics.md`) found P ≈ H (+0.33/GW, p 0.26), set-1 holding overvalued and set-2 undervalued by one pooled table, and chips worth 1.5–2.5× in doubles. Validate is spent; the live shadow is the out-of-sample check. |
```

- [ ] **Step 4: Full test run and lint**

Run: `uv run pytest -q` then `uv run ruff check . && uv run ruff format .`
Expected: all PASS (the `realdata` marker stays skipped), lint clean.

- [ ] **Step 5: Commit and push; open the PR**

```bash
git add config/chips/terminal-values-window.json config/chips/terminal-values-double.json CLAUDE.md docs/PLAN.md
git commit -m "chore(chips): estimated per-window and double-split terminal values (develop, loso)"
git push -u origin feat/chip-policy-2
gh pr create --base main --title "Chip policy 2: variants V1-V6 vs H" --body-file - <<'EOF'
Implements docs/superpowers/specs/2026-10-30-chip-policy-2-design.md (plan docs/superpowers/plans/2026-10-30-chip-policy-2.md): window- and double-keyed terminal values, leave-one-season-out tables, a rule Triple Captain, a Free Hit gain shrink in the chip search, the comparison driver. No change to the live bot (it keeps the pooled table). Runs follow on the server; results go in a later commit.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
```

---

### Task 10: Round 1 on the server — H, P, V1–V3

The runs need the branch's code on the server test checkout. **Never** run in `~/fpl-optimizer`; `nice -n 15`, at most 12 jobs.

- [ ] **Step 1: Put the branch on the test checkout**

```bash
ssh fplopt-server
cd /mnt/ibrokeass/fpl-tests/code && source ../env.sh
git fetch git@github.com:samintisar/fpl-optimizer.git feat/chip-policy-2 && git checkout --detach FETCH_HEAD
~/.local/bin/uv sync --locked --all-extras
git log --oneline -1
```

Expected: HEAD is the branch tip. If the GitHub fetch is refused (deploy key scope), stop and ask the user (merging the PR first, then `git fetch ~/fpl-optimizer main` as usual, is the fallback).

- [ ] **Step 2: Start the five runs, one after another, in the background**

```bash
cd /mnt/ibrokeass/fpl-tests/code && source ../env.sh
BASE="horizon=6,decay=0.75,max_hits=1,hit_margin=2"
COMMON="--seasons 2016-2022 --starts template@1,random:3@1,random:3@20 --jobs 12 --family chip-policy-2026-11"
run() { nice -n 15 ~/.local/bin/uv run fplopt backtest run --spec "$2" $COMMON --out results/cp2-$1 > results/cp2-$1.log 2>&1; }
nohup bash -c "
  $(declare -f run); BASE='$BASE'; COMMON='$COMMON'
  run h  'chiprules:v1:$BASE,t_tc=8.3,t_bb=10.9,k_fh=4,t_wc=12.87' &&
  run p  'optimizer:v1:$BASE,chips=1,chip_shrink=1.0' &&
  run v1 'optimizer:v1:$BASE,chips=1,chip_shrink=0.6,chip_split=1,loso=1' &&
  run v2 'optimizer:v1:$BASE,chips=1,chip_shrink=0.8,chip_split=1,loso=1' &&
  run v3 'optimizer:v1:$BASE,chips=1,chip_shrink=1.0,chip_split=1,loso=1'
" > results/cp2-round1.log 2>&1 &
```

Expected: `results/cp2-h.log` starts filling; each run takes roughly as long as its diagnostics counterpart (check `results/chips-diag-*.log` timestamps on the server for an estimate).

- [ ] **Step 3: When all five finish, compare**

```bash
nice -n 15 ~/.local/bin/uv run python dev/chip_policy_2.py --h results/cp2-h \
  --arm P=results/cp2-p --arm V1=results/cp2-v1 --arm V2=results/cp2-v2 --arm V3=results/cp2-v3 \
  --out results/cp2-round1
```

Expected: five lines `… vs H: …` and `best of V1-V3: Vk`. Record k* (V1 → 0.6, V2 → 0.8, V3 → 1.0).

---

### Task 11: Round 2 — V4–V6 at k*, the verdict

- [ ] **Step 1: Start the three runs** (replace `K` with k*)

```bash
cd /mnt/ibrokeass/fpl-tests/code && source ../env.sh
K=0.8  # k* from round 1
BASE="horizon=6,decay=0.75,max_hits=1,hit_margin=2"
COMMON="--seasons 2016-2022 --starts template@1,random:3@1,random:3@20 --jobs 12 --family chip-policy-2026-11"
run() { nice -n 15 ~/.local/bin/uv run fplopt backtest run --spec "$2" $COMMON --out results/cp2-$1 > results/cp2-$1.log 2>&1; }
nohup bash -c "
  $(declare -f run); BASE='$BASE'; COMMON='$COMMON'; K='$K'
  run v4 'optimizer:v1:$BASE,chips=1,chip_shrink=$K,chip_split=1,loso=1,tc_rule=8.3' &&
  run v5 'optimizer:v1:$BASE,chips=1,chip_shrink=$K,chip_split=2,loso=1' &&
  run v6 'optimizer:v1:$BASE,chips=1,chip_shrink=$K,chip_split=1,loso=1,fh_scale=0.4'
" > results/cp2-round2.log 2>&1 &
```

- [ ] **Step 2: Final comparison**

```bash
nice -n 15 ~/.local/bin/uv run python dev/chip_policy_2.py --h results/cp2-h \
  --arm P=results/cp2-p --arm V1=results/cp2-v1 --arm V2=results/cp2-v2 --arm V3=results/cp2-v3 \
  --arm V4=results/cp2-v4 --arm V5=results/cp2-v5 --arm V6=results/cp2-v6 --out results/cp2-final
```

Expected: `passing: [...]` or `passing: none`.

- [ ] **Step 3: Bring the results back and record them**

Copy `results/cp2-final/summary.json` and the run rows of the server's `results/experiments.csv` (family `chip-policy-2026-11`) to the worktree; append the rows to `results/experiments.csv`; write `docs/research/2026-11-chip-policy-2.md` (verdict, table of the six variants vs H, k*, V5 sanity, what gets frozen); add a PLAN decision row with the verdict. Commit to the PR branch:

```bash
git add results/experiments.csv docs/research/2026-11-chip-policy-2.md docs/PLAN.md
git commit -m "docs(research): chip policy 2 develop results"
git push
```

- [ ] **Step 4: Freeze** (only after the user has seen the verdict): tag the commit `chip-policy-2-frozen` with the winner's spec string in the tag message; the shadow-logger plan follows.
