# Chip policy (Phase 8, first item): pre-registration

**Status: APPROVED by the user, 2026-10-11** (terminal values estimated per window left (§3); gate "P, else H" at α 0.05 each (§4); the rest as written). Nothing has been run.

**Goal.** Decide, with a paired test fixed in advance, whether the bot may recommend chips, and how. Until then, chips stay the manager's call. `/chips` shows the untested planner as information only (PR #36).

**Why now.** The user's first-set Free Hit, Bench Boost and Triple Captain expire at the GW19 deadline (`gw_index` 19, 2026/27). The aim is a verdict by about GW10, so a passing policy still has about 9 GWs to play them.

**What's known.**
- Phase 4: chips on vs off was worth +4.3 points/GW. That is "chips used vs chips wasted": the no-chip arm never plays its chips, and the gain is mostly Wildcard rebuilds (PLAN §7). It says nothing about *timing*.
- Timing is set by the terminal values for holding a chip past the horizon: WC 6, FH 4, BB 4, TC 3, all **placeholders** (PLAN §11).
- The 2025/26 holdout has been used (Phase 6), so it can't serve as a test set again.
- Development data: develop 2016/17–2022/23, validate 2023/24–2024/25 (PLAN §5).

---

## 1. Arms

All arms use v1 xP and the Phase 6 system's transfer settings (`optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2`). They differ only in chips.

| Arm | Chips |
|---|---|
| **S0** | none. The live system today. |
| **H**, heuristic | Chosen by fixed rules (§2). In a GW where a rule fires, the optimizer plans with that chip fixed; otherwise, no chip. |
| **P**, planner | The chip scenario search (`chips=1`) with **estimated** terminal values (§3) instead of the placeholders. |

S0 is the floor. **H is the bar that matters:** a chip planner that can't beat simple rules shouldn't drive chip advice.

## 2. The heuristic baseline H (decision 1)

Each rule uses only what's visible at the deadline: the v1 xP frame and the as-of fixtures. One chip per GW, in priority order FH > WC > BB > TC.

**Every chip is also played in the last GW of its window if still unused** ("use it or lose it"). The rules:

- **Triple Captain:** play when the best captain's xP ≥ `T_tc`. A double gameweek for a premium usually clears it.
- **Bench Boost:** play when the four bench players' summed xP ≥ `T_bb`.
- **Free Hit:** play when the planned XI has ≥ `K_fh` starters without a fixture this GW (a blank).
- **Wildcard:** play at the **first GW of the window whose 6-GW horizon objective with the Wildcard beats the no-chip plan by ≥ `T_wc`** (one extra solve per GW while the Wildcard is unused).

**Thresholds** (`T_tc`, `T_bb`, `K_fh`, `T_wc`) are chosen on **develop only**, from a small grid of 3 values each, tuned one chip at a time. The grid points are the 50th, 75th and 90th percentiles of each quantity's distribution in S0's develop runs, and for `K_fh` the values {2, 3, 4}. The final thresholds are logged before validate runs.

## 3. Terminal values for P (decision 2)

**Recommended: estimate them, then shrink.**

The estimate: for each chip and number of GWs `L` left in its window after the horizon, the terminal value is the mean over S0's develop deadlines of the **best ex-ante one-GW gain** of that chip in those `L` GWs:
- **TC:** the captain's xP.
- **BB:** the bench's xP net of its bench weights.
- **FH:** the Free Hit squad's xP minus the XI's xP (one solve).
- **WC:** the Wildcard plan's 6-GW gain (one solve).

That is "the expected best use of the chip in the rest of its window" (PLAN §7). A best-of-`L` from noisy forecasts overstates what can be had, so it is multiplied by a shrink factor `k` ∈ {0.5, 0.75, 1.0}, chosen on develop.

The result is a table per chip by `L`, stored in config and logged. It replaces `OptimizerParams.chip_value`, which is one number per chip that ignores how much of the window is left.

*Alternative:* scale the placeholders by s ∈ {0.5, 1, 1.5, 2}, chosen on develop. It's cheaper, but it keeps the "window length doesn't matter" flaw.

## 4. Test (decision 3)

**Design: full run only.** The per-decision windows can't charge a chip's use to later GWs (PLAN §5 *Known limitation*), so chips are judged on whole-season paired differences.

| Setting | Value |
|---|---|
| Starts | `template@1,random:3@1` (set 1 + set 2, whole seasons) and `random:3@20` (set 2 only) |
| Statistic | **as replaced by the size check (below):** one-sided **full-run window t-test** (`evaluate.full_run_window_test`). Per-GW paired differences in realized points go into non-overlapping 8-GW cells (gw_index 1–8, 9–16, …, 33–38). Each cell is averaged over its GWs and starts. The t-test runs over the cells, df = n − 1, about 10 cells on validate. Plus the same sign on xG-scored points. |
| Size check first | **Done 2026-10-11, before any run.** The pre-registered GW-block bootstrap rejects about 12.5% at nominal 10% on two seasons, which is outside 0.08–0.12 (measured 0.131 / 0.124 for 1 / 4 seasons, `test_bootstrap_size_under_no_effect`), so it was replaced, as this row allowed. The window t-test's A/A size on two seasons is in range: 0.10 with independent GWs, 0.108 with per-GW AR(1) 0.3 (`tests/test_chip_study.py`); in a simulation, 8-GW cells hold ≤ 0.114 up to AR 0.5, where 4-GW cells reach 0.141. Measured lag-1–4 autocorrelations of full-run differences are about 0 (−0.05 to +0.04, Phase 5b develop runs). The bootstrap CI is still reported by `backtest compare`. |

**Size on real differences (added after code review, before any run).** The A/A size above assumes Gaussian cells, but chip differences are skewed (Triple Captain and Bench Boost spikes). After the develop runs, and before validate, run `fplopt chips size --paired <develop P vs S0 run>` and the same on H vs S0. This resamples the real develop 8-GW cells, centred to no effect, 10 at a time. If the rejection rate at α 0.05 is above 0.07, the validate gate uses the **calibrated critical p** that run reports (the 5% quantile of the null p values) instead of 0.05. The number used is recorded before validate runs.

**Order:**
1. **Develop:** tune H's thresholds and P's `k`. Logged, with the number of variants tried, for deflation.
2. **Validate, once:**
   - P vs H;
   - P vs S0;
   - H vs S0.

**Gate for the bot to recommend chips (decision 4), recommended:**
- **P goes live** if it beats **both H and S0** on validate: an intersection-union test at α 0.05 each.
- **H goes live instead** if P fails but **H beats S0** at α 0.05.
- The two candidates split α = 0.10 (Bonferroni), so either may go live without a second look.
- If neither passes, chips stay manual all season, and `/chips` stays information only.

Validate is 2 seasons, which is thin. As in Phase 6, a fail is plausible even if P is better.

## 5. What going live means

**If P passes:**
- Plans and alerts include chip scenarios.
- The message shows the chip, its GW and the gain over holding it.
- `/chips` loses its "information only" label.

**If H passes:** the alerts say e.g. "Bench Boost this GW (rule: bench xP 14.2 ≥ 12.0)", and the transfers are planned with it.

**Either way,** the user still decides, and chip messages are logged like the rest.

## 6. Work and timeline (estimates)

1. **Code:**
   - H as a policy (`policies.HeuristicChipPolicy`, a fixed-chip wrapper around `OptimizerPolicy`);
   - per-`L` terminal values in `OptimizerParams` and `chips.terminal_value`;
   - the gain extraction for §3;
   - the two-season A/A size test.

   One PR, reviewed, before any run: ~1–2 days.
2. **Develop runs on the dev PC at `--jobs 14`:**
   - S0 is cached;
   - H, 4 chips × 3 values, tuned sequentially;
   - P, 3 values of `k`.

   Several hours.
3. **Validate:** one run of three comparisons, about 2–3 hours, then record the result in PLAN §7, §9 and §12.

Target: verdict by about GW10 (late October).

## 6b. As built (code PR, 2026-10-11)

- **Specs:**
  - `chipgains:v1:<optimizer settings>`: S0, recording each chip's ex-ante gain per GW in the GW rows (`gain_tc`, `gain_bb`, `gain_fh`, `gain_wc`, `captain_xp`, `bench_xp`, `blanks`);
  - `chiprules:v1:<optimizer settings>,t_tc=,t_bb=,k_fh=,t_wc=`: H;
  - `optimizer:v1:...,chips=1,chip_shrink=k`: P, using the table in `config/chips/terminal-values.json` times k.
- **`fplopt chips values --gains <chipgains run>`** writes that table from develop seasons only (`fplopt.backtest.chip_values`).
- **`backtest compare`** prints and logs `full_run_window_tests` per split, for realized and xG.
- **Review fixes (before any run):**
  - `OptimizerParams` pickling now keeps `chip_value_left`. Without it, worker processes silently ran P with the placeholders.
  - P's policy name carries a checksum of the whole table.
  - The table P ran with is written into the run's config (`summary.json`, experiment log).
  - `fplopt chips size` checks the size on real differences.
- **Develop run 1 (S0 chip gains, `results/chips-s0-develop`) found a bug, fixed before any tuning.** The Free Hit gain's 1-GW plan kept the expected-minutes floor of a full horizon: 180 minutes in one GW, so it had no candidates and every Free Hit gain was "keep the squad" (median −1.2). The fix is `policies.one_gw_params`, which scales the floor to one GW, with a test that fails without it. Only `gain_fh` was affected; Triple Captain, Bench Boost and the 6-GW Wildcard solves were not. S0 was rerun from the fix (`results/chips-s0-develop-v2`, commit 785e97a; its experiment-log row is committed with this note). The rerun's decisions and its TC/BB/WC gains are identical to the first run's, and `gain_fh` now has median 6.6 (11.3 in GWs with blanks). The first run stays in the experiment log.
- **To keep in mind when interpreting the runs (from review):**
  - **Bench weights.** A Wildcard or Free Hit GW uses the fixed bench weights, while the no-chip plan uses v1's minutes-based ones. Part of `gain_wc`/`gain_fh` is that switch, not the squad: measure it on a few develop deadlines before setting H's `T_wc` grid.
  - **Undiscounted terminal values.** They are added undiscounted, while chips played inside the horizon are discounted by `decay^h`. So P leans towards holding chips until the window's end comes inside the horizon (PLAN §7 design; `k` partly offsets it).
- **H's rules** use the raw quantities as written in §2: captain xP, the bench's summed xP, squad players with no fixture, and the Wildcard plan's gain. The grid points are their develop percentiles from the chipgains run.

## 6c. Develop results and the settings fixed for validate (2026-10-11, before any validate run)

All runs cover develop 2016/17–2022/23 from starts `template@1,random:3@1,random:3@20` (49 runs, 1,448 GWs each, every solve Optimal). Differences are full-run mean net points per GW against S0 (`results/chips-s0-develop-v2`). These are **in-sample**: the settings were chosen on these seasons.

- **Terminal values:** `config/chips/terminal-values.json`, from the S0 rerun, values for L = 1 → 19:

  | Chip | L = 1 | L = 19 |
  |---|---|---|
  | TC | 8.0 | 12.8 |
  | BB | 5.2 | 9.4 |
  | FH | 6.1 | 22.4 |
  | WC | 4.8 | 23.3 |

- **H, tuned one chip at a time** (TC, BB, FH, WC; the others at their current value; `results/chips-h-tune.log`; 9 variants):

  | Setting | Grid values: mean vs S0 per GW | Chosen |
  |---|---|---|
  | `t_tc` | 7.04: +1.99; 8.3: +2.19; 10.88: +2.17 | **8.3** |
  | `t_bb` | 7.54: +1.90; 9.42: +2.19; 10.9: +2.36 | **10.9** |
  | `k_fh` | 2: +2.24; 3: +2.36; 4: +2.39 | **4** |
  | `t_wc` | 6.75: +2.20; 12.87: +2.39; 19.65: +2.08 | **12.87** |

  - Chosen H: `chiprules:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2,t_tc=8.3,t_bb=10.9,k_fh=4,t_wc=12.87`, **+2.39/GW vs S0**.
  - Every chip is played in every window it has (77 of each; the last-chance rule).
- **P, chip_shrink k** (`results/chips-p-tune.log`; 3 variants):

  | k | vs S0 per GW | vs H per GW |
  |---|---|---|
  | 0.5 | +1.77 | −0.62 |
  | 0.75 | +2.07 | −0.32 |
  | 1.0 | +2.53 | +0.14 |

  - **Chosen k = 1.0:** `optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2,chips=1,chip_shrink=1.0`.
  - The best k is the grid's upper edge, and the grid is not widened, since that would be another look.
  - P's develop edge over H is small (+0.14/GW).
- **Size on real develop differences** (`fplopt chips size`, 10 cells resampled from the centred develop 8-GW cells; rates at α 0.05):
  - P vs S0: 0.061;
  - H vs S0: 0.025;
  - P vs H: **0.075**, with calibrated critical p 0.0296.
- **Validate thresholds, fixed now.**
  - The rule (§4) names P vs S0 and H vs S0, which are both ≤ 0.07, so they stay at **p < 0.05**.
  - P vs H over-rejects, and it is part of P's gate, so it uses the stricter **calibrated p < 0.0296**. This is the conservative reading.
  - Each comparison also needs a positive mean xG-scored difference.
  - **P goes live** if P vs H and P vs S0 both pass.
  - **Otherwise H goes live** if H vs S0 passes.
  - **Otherwise chips stay manual.**
- **Validate run, once:**
  - seasons 2023/24–2024/25, the same starts;
  - each arm run once with `backtest run` (S0 as `optimizer:v1:...` without chips: the same decisions as `chipgains`, which is cheaper);
  - the three comparisons with `evaluate.full_run_window_test` on the paired per-GW differences.

## 6d. Validate result (2026-10-11, the one run): **chips stay manual**

`results/chips-validate/`: seasons 2023/24–2024/25, starts `template@1,random:3@1,random:3@20`, 22 runs and 418 GWs per arm, every solve Optimal. Settings and thresholds were as fixed in §6c (commit 0f8f3cb). The statistic is the full-run window t-test on 8-GW cells (10 cells).

| Comparison | Mean per GW | One-sided p | Threshold | xG per GW | Per season (mean over starts) | Result |
|---|---|---|---|---|---|---|
| P vs H | **+1.70** | 0.083 | < 0.0296 | +1.57 | 2023/24 +68, 2024/25 +26 | fail |
| P vs S0 | **+4.18** | **0.025** | < 0.05 | +4.30 | +131, +140 | pass |
| H vs S0 | +2.49 | 0.106 | < 0.05 | +2.74 | +63, +114 | fail |

- **Verdict, as pre-registered: chips stay manual.**
  - P beats S0 but not H at the fixed threshold.
  - P vs H would also fail at a nominal 0.05, so the calibrated threshold didn't decide it.
  - H doesn't beat S0.
- **Season totals** (mean over starts):

  | Arm | 2023/24 | 2024/25 |
  |---|---|---|
  | S0 | 1,840 | 1,814 |
  | H | 1,903 | 1,928 |
  | P | 1,971 | 1,954 |

- **Reading it.** P was the best arm in both validate seasons and on xG. Its edge over the rule baseline (+1.7/GW, about 65 points a season) is positive but not established on two seasons. That fits the develop edge (+0.14/GW) being small and the test having little power.
  - This is not evidence against P.
  - It is not enough to let P drive alerts.
- **Live, it stays as is.** `/chips` stays information only, and plans and alerts play no chips. Whether `/chips` should show P's estimated terminal values instead of the placeholders is a separate display decision, untested as such. The live 2026/27 log can add evidence at season end (a second look on new data, not on these seasons).

## 7. Open (VERIFY before the code PR)

- **Free Hit in consecutive GWs in 2026/27** (PLAN §11): check the rules config, since it constrains H and P.
- **Blank GW visibility:** blanks are only known from the as-of schedule from 2026-10-05 on (PLAN §3 limitation). Earlier seasons see the final fixture list, so H's Free Hit rule and P's Free Hit timing look better in backtests than they will live. This is reported, not fixable.
