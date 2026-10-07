# Phase 4: Optimizer (MILP) — Implementation Plan

> **For agentic workers:** Implement task-by-task (one implementer per task or wave, then an independent review). TDD: failing tests first. Before using PuLP or highspy APIs, fetch their current docs (`npx ctx7@latest library PuLP "<question>"`, then `docs`).

**Goal:** A MILP planner (PuLP on HiGHS) over transfers, captain, bench order and chips for a 6-GW rolling horizon. It returns the top 3 plans plus the roll plan, with each plan's xP gain vs roll. An `OptimizerPolicy` executes plan #1 in the Phase 3 backtester.

Phase 4 is done when (PLAN §9):
- the backtest runs end-to-end with the optimizer;
- it beats greedy (paired, same xP);
- it matches open-fpl-solver on no-chip cases;
- solve times are acceptable (#10).

**Spec:** `docs/PLAN.md` §7 (formulation, chips, objective, defaults, practicalities), §5 (policies, paired evaluation), §4 (determinism for the corrupt-the-future check). Phase 3 code: `src/fplopt/backtest/` (rules, state, gw_score, policies, simulator, evaluate).

**Branch:** `phase-4-optimizer`. **Holdout** untouched, as in Phase 3.

---

## Decisions

- **Stack:** PuLP model, solved by HiGHS in-process (`pulp.HiGHS`/highspy, no temp files), `threads=1`, gap-based stopping (`mip_rel_gap` 0.005), and a generous time limit as a safety net only. The model is built in sorted key order, so it is deterministic and the corrupt-the-future check can fingerprint decisions. Model-build and solve time are recorded separately (PLAN §2: switch to highspy's bulk API only if build time dominates).
- **Prices are fixed over the horizon** at the deadline price (no price prediction, PLAN §1). Selling: an owned player sells at his real selling price on his **first** sale (`sell_first`). A player bought within the plan, or re-bought, sells at the plan's buy price (`sell_later`).
- **Free transfers:**
  - Integer `ft[t]` in 1..cap, integer `hits[t]` ≥ 0.
  - Normal GWs: `Σ buy ≤ ft + hits` and `ft[t+1] ≤ ft[t] − (Σ buy − hits) + 1`.
  - WC/FH GWs: transfers free, and `ft[t+1] = ft[t]` (`chip_week_ft: retain`, verified in Phase 3).
  - GW1 (`gw_index == 1`): unlimited free transfers, and `ft[2] = 1`.
  - FT top-ups per the rules.
  - One binary per FT state (1–5) carries the concave FT value.
- **Objective** (PLAN §7):
  - `Σ_t decay^t · [Σ xP·lineup + xP·captain + Σ_k w_k·xP·bench_k + chip terms + ft_value(ft[t]) + itb_value·bank[t] − (hit_cost + hit_margin)·hits[t]]` + the terminal value of unused chips.
  - Hits are inside the decay (a hit in a later GW is valued like that GW's points). FT value and money in the bank follow open-fpl-solver's convention exactly, so the reference check compares like with like. Task 4 confirms the convention and records any difference.
- **Bench:** fixed weights (GK 0.03; outfield 0.21 / 0.06 / 0.002) until Phase 5's minutes model gives P(needed).
- **Vice-captain:** the second-highest-xP starter, set after the solve.
- **Chips:**
  - **Scenario solve.** Enumerate the chip assignments within the horizon that the rules allow:
    - no chip;
    - each available chip in each eligible GW;
    - pairs only when both are available, in different GWs, and FH is not consecutive.
  - Fix each scenario and solve it; keep the best objective.
  - **FH** has separate squad variables for its GW. The persistent squad and bank carry over unchanged, and the FH squad's budget = bank + the squad's sale value.
  - **BB** gives every bench weight 1. **TC** gives a captain multiplier of 3. **WC** makes transfers free.
  - **Terminal value** of an unused chip is a configurable value per chip (`chip_value`) when its window extends past the horizon, and 0 when the window closes inside it. This is a placeholder for PLAN §7's "estimated from backtest distributions"; tuning comes later.
  - The backtest's no-chip mode skips the scenarios.
- **Top-3 plans:**
  - Solve, then add a no-good cut that excludes plan #1's **first-GW transfer set**, and re-solve (×2).
  - The roll plan = no transfers in the first GW, with the rest optimized.
  - Each plan reports its xP gain vs roll over the horizon.
- **Pruning:**
  - Keep owned players.
  - Per position, keep the top-N by horizon xP and the top-N by horizon xP per price.
  - Dominance pruning (cheaper and ≥ xP in every horizon GW, same position, with the dominators kept spanning ≥ the slots that position could need + 5 clubs, so the club cap can't block them all; Task 3).
  - Defaults: N = 20/60/60/30 (GK/DEF/MID/FWD), set by the Task 3 benchmark (was 10/30/30/15).
- **Chip scenario search (Task 3):** best-first by upper bounds (LP relaxation, or the WC/FH base's bound + a Triple Captain/Bench Boost bound); only scenarios that could beat the incumbent are solved, so the result equals the exhaustive search (`scenario_search="all"`).
- **Parallel backtests:** `run_grid` takes `jobs` (process pool over (season, start state); each worker opens its own `DataStore` and `Caches`). Results are identical to `jobs=1` because every piece is deterministic.

---

## File structure

| File | Responsibility |
|---|---|
| `src/fplopt/optimize/params.py` | `OptimizerParams` (frozen): horizon, decay, ft_value, itb_value, hit_margin, bench weights, chip_value, pruning N, mip_gap, threads, time_limit |
| `src/fplopt/optimize/problem.py` | `PlanInput` from (state, pool, xp, rules, params): holdings with selling prices, candidates, xP matrix, horizon GWs, chip availability |
| `src/fplopt/optimize/prune.py` | Candidate pruning |
| `src/fplopt/optimize/model.py` | MILP build + solve for one chip scenario → `Plan` |
| `src/fplopt/optimize/chips.py` | Scenario enumeration, terminal values |
| `src/fplopt/optimize/plans.py` | `optimize(PlanInput) -> PlanSet` (top-k + roll, best over chip scenarios), `Plan`/`PlanSet` dataclasses, conversion to `state.Decision` |
| `src/fplopt/backtest/policies.py` | `OptimizerPolicy(xp_model, params, chips=False)` |
| `src/fplopt/backtest/probes.py` | Optimizer decision probe |
| `src/fplopt/backtest/evaluate.py` | `jobs` for `run_grid` / `per_decision` |
| `src/fplopt/cli.py` | `optimizer` policy spec; `backtest --jobs`; `fplopt optimize plan` / `fplopt optimize bench` |
| `dev/reference_check.py` | Differential check against open-fpl-solver (dev only) |
| `.github/workflows/*` | `uv sync --locked --all-extras` (PuLP/highspy) |
| tests | `test_optimize_*.py`, policy/probe/architecture/CLI extensions |

---

### Task 0: Plan, CI extras (lead)
- [ ] Commit this plan. CI installs all extras. PLAN §7 gets a "Phase 4 decisions" note (hits inside the decay, bench fallback weights in use, chip terminal values as config, top-3 cut on first-GW transfers, pruning defaults) and a §12 row.

### Task 1: Core MILP without chips (wave A)
**Files:** `optimize/params.py`, `problem.py`, `prune.py`, `model.py`, `plans.py` (minimal: plan #1 only), tests.
- [ ] `PlanInput.from_context(state, pool, xp, rules, params)` gives the horizon from the xP frame (horizon < params.horizon near season end).
- [ ] Owned players missing from the pool keep their last price and club, can be sold and can't be bought (as `state.py`).
- [ ] The model per the Decisions, without chips. Constraints:
  - squad continuity;
  - 15 players, 2/5/5/3, ≤ team_limit per club; clubs are fixed at the deadline, and the pre-existing violation rule matches `state.py`;
  - lineup 11 with a valid formation; bench slots (slot 0 the GK, slots 1–3 outfield in order); captain;
  - budget/bank;
  - FT dynamics, hits, GW1 unlimited.
- [ ] `Plan`: per horizon GW the transfers (out/in), starters, bench order, captain, vice, chip, xP, hits, bank, ft; plus objective, solve stats (build time, solve time, gap, status).
- [ ] `Plan.decision()` gives a `state.Decision` for the first GW.
- [ ] Tests:
  - tiny instances against brute force: 1-GW horizon on a small pool, enumerating all valid squads/lineups with ≤ 2 transfers;
  - the first-GW decision passes `apply_decision`;
  - multi-GW plans replayed through `apply_decision`/`next_state` match the plan's bank/ft/hits per GW;
  - selling price on first sale vs re-sale;
  - FT banking to the cap and FT-value effects (a high ft_value makes it roll);
  - hit margin;
  - a club over the cap from a club change;
  - a player who left the game;
  - determinism (same input → identical plan);
  - pruning keeps owned players and never removes a player the unpruned optimum uses on test instances.

### Task 2: Chips, top-3, roll plan (wave B; needs Task 1)
**Files:** `chips.py`, `plans.py`, `model.py`, tests.
- [ ] Scenario enumeration from `state.chips_used`, rule windows on gw_index, and the FH-consecutive rule. Only scenarios inside the horizon.
- [ ] FH/WC/BB/TC terms. Terminal values.
- [ ] Best scenario kept, with all scenario objectives recorded.
- [ ] Top-3 via no-good cuts on the first-GW transfer set; roll plan; gain vs roll.
- [ ] Tests:
  - each chip against a hand-built instance where it is clearly best (or clearly not);
  - FH squad reverts in the plan (GW t+1 squad = GW t−1 squad);
  - a set-1 chip expiring inside the horizon has terminal value 0;
  - no FH in consecutive GWs;
  - top-3 plans are distinct in first-GW transfers, with objectives non-increasing;
  - roll plan's first GW has no transfers;
  - the chip decision passes `apply_decision`.

### Task 3: Benchmark and pruning defaults (#10) (wave B, after Task 2's model is in; may run with Task 4)
**Files:** `src/fplopt/optimize/bench.py`, CLI `fplopt optimize bench`.
- [x] On real data, from template and random states at 10+ deadlines across seasons (ep_next and rolling xP), measure model-build and solve time and final gap:
  - no chips / all chip scenarios / top-3;
  - pruning N variations;
  - the objective loss from pruning vs a larger pool.
- [x] Pick the defaults. Record the numbers in PLAN §7 and close #10.

### Task 4: Reference check against open-fpl-solver (wave B; needs Task 1)
**Files:** `dev/reference_check.py`, `dev/README.md` (how to run), optional `tests/test_optimize_reference.py` (skipped unless `OPEN_FPL_SOLVER_DIR` is set).
- [ ] Clone `solioanalytics/open-fpl-solver` (Apache-2.0) into a scratch dir (never vendored).
- [ ] Map our `PlanInput` (squad, selling prices, bank, ft, xP per player and GW) to its inputs (projection CSV + settings + an initial-squad JSON or equivalent). Match settings: horizon, decay, ft_value, itb_value, hit cost, bench weights, no chips, same candidate pool.
- [ ] Compare objective values (and, ideally, first-GW transfers) on ≥ 20 instances from real deadlines (no holdout).
- [ ] Differences must be within the MIP gaps. Explain and fix any convention mismatch (e.g. how FT value and the bank enter the objective) in our model, or document a deliberate difference in PLAN §7.
- [ ] Report the instances, objectives, gaps and runtimes of both.

### Task 5: Optimizer policy, backtests, CLI (wave C; needs Tasks 2–3)
- [ ] `OptimizerPolicy(xp_model, params=OptimizerParams(), chips=False)` gives `decide(ctx)` = plan #1's first-GW decision (top-3 off in backtests).
- [ ] Architecture scan: the optimizer modules (no file reads; allowed imports add pulp/highspy). The probe `optimizer_ep_next_template` goes in `PROBES` (no chips, small pruning to keep the leakage check fast).
- [ ] `run_grid(..., jobs=N)` / `per_decision(..., jobs=N)` with a process pool; equal results for jobs=1 vs 2 (test). CLI `--jobs` (default: CPU count − 1).
- [ ] CLI policy spec `optimizer:xp[:horizon=..,decay=..,chips=1,...]`. `fplopt optimize plan --season S --gw G --start template|random:SEED --xp ep_next` prints top-3 + roll plans with xP gains.
- [ ] **Real-data runs** (report verbatim with timings):
  - (a) optimizer:ep_next vs greedy:ep_next, 2021–2024, `template@1,random:5@1`;
  - (b) optimizer:rolling vs greedy:rolling, 2016–2024, `template@1,random:5@1`;
  - (c) optimizer:ep_next:chips=1 vs optimizer:ep_next, 2021–2024 (full run is the chip metric, PLAN §5).
- [ ] The optimizer must beat greedy in (a) and (b) (paired, one-sided p < 0.10 on the full run and per-decision). If it doesn't, investigate before tuning (bugs first).
- [ ] Docs: CLAUDE.md commands, README status, PLAN §7/§9 results.

### Task 6: Review and PR (lead)
- [ ] Independent reviews: MILP correctness vs state rules, chips, determinism/leakage, performance. Fix the findings.
- [ ] Full tests, `-m realdata`, `fplopt check leakage`. PR → CI → CodeRabbit → merge → server sync (with extras).

## Waves
- **A:** Task 1. Lead: Task 0.
- **B:** Task 2, then Task 3 ∥ Task 4.
- **C:** Task 5, then Task 6.
