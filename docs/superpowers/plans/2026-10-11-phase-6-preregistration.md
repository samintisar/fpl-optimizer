# Phase 6 pre-registration: the 2025/26 holdout run

**Status: APPROVED by the user, 2026-10-11** (all five decisions as recommended). Recorded in PLAN §9 (*Phase 6 design*), §11 and §12. The code in *Before the run* is in `fplopt.backtest.holdout` and `fplopt holdout check|run`. Then the run happens **once**. Nothing here has looked at 2025/26 results.

## 1. What is tested

**System under test (fixed 2026-10-09, PLAN §12):** v1 xP + `optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2`. Bench weights are minutes-based, except fixed at GW1, in a Free Hit GW and from a Wildcard on. Expected-minutes floor 180. **No chips** (see §4).

**Baselines (PLAN §9 go-live gate): the system must beat each one:**
- **(a)** `greedy:rolling`: rolling-average xP + the greedy policy at its Phase 3 defaults (threshold 1.0, max 1 transfer).
- **(b)** `optimizer:ep_next:horizon=6,decay=0.75,max_hits=1,hit_margin=2`: FPL's `ep_next` with the same optimizer and settings, and the fixed bench weights (`ep_next` has no minutes). This separates the model's value from the optimizer's.

**Season and starts:** 2025/26 under its native rules (with the AFCON top-up, §5). Starts `template@1,random:3@1,random:3@20`, the same as Phase 5b.

## 2. The test (decision 1: a test with the right size on one season)

**Problem.** The pre-registered statistic is the GW-block bootstrap. Measured by A/A placebo, it rejects ~14% of the time at nominal 10% on a single season (PLAN §5). On ~10 cells, a percentile bootstrap is anti-conservative.

**Recommendation: a one-sided t-test on the per-decision window differences, df = n − 1.**
- **Unit:** one non-overlapping 4-GW decision window: decisions at gw_index 1, 5, 9, …, 37, so n ≈ 10 (the last window is 2 GWs). Each window's difference is averaged over the start states that are live in it (the GW20 starts join from window 21).
- **Why a plain t-test:** the windows don't overlap, so they're close to independent. With df = n − 1 the t critical value accounts for estimating the variance from ~10 values, which the bootstrap does not. No HAC correction is needed at stride = k.
- **Gate for each baseline:** mean window difference in realized points > 0, with one-sided t p < 0.10, **and** a mean xG-scored difference > 0 (same sign).
- **Two baselines, no correction:** "beats each" is an intersection-union test, so each comparison at α 0.10 keeps the overall error at 0.10. No Bonferroni.
- **Size check before the run:** extend the A/A placebo test (`tests/test_backtest_evaluate.py`) to the window t-test. Its rejection rate must be 0.08–0.12 for one season (expected ≈ 0.10). If it isn't, stop and revisit before running.
- **Still reported, not gating:**
  - the bootstrap 80% CI and p;
  - the full-run difference (mean, block-bootstrap CI);
  - the season totals;
  - the average-manager benchmark;
  - per-start results.

**Rough power (honest expectation).** Phase 5b validate's per-decision edge over greedy:v1 was +7.3 points per window, with a one-season SE of about 4.5. At t(9) the bar is a mean of about 1.38 × SE ≈ +6.2 per window, so with a +7 true edge the power is only around 55–60%. The binding baseline is likely (b): its decision edge has never been measured. **A fail is a real possibility even if the system is better.** PLAN already accepts that go-live may slip.

## 3. Per-decision design (decision 2: continuation and reference)

**Recommendation: `--continuation own --reference b`, as in the Phase 5b validate run.**
- **Reference = the baseline's own run:** each comparison's decision states come from the baseline's trajectory. The states are therefore not chosen by the system under test.
- **Own continuation:** after each decision, each arm keeps playing its own policy for the rest of the 4-GW window. This credits a plan's follow-up moves; the shared roll continuation never does. Every window still starts both arms from the same state, so there's no carry-over between windows.
- **k = 4, stride 4, block length 4:** unchanged.

## 4. Chips (part of decision 1)

**Recommendation: chips off in every arm.**
- The system was tuned and validated without chips.
- The chip terminal values are placeholders (PLAN §11).
- There is no heuristic chip baseline yet.
- The per-decision windows can't charge chip use across windows (PLAN §5).

Including chips would make the test about untuned chip timing. Live, chips stay a manual decision until a chip policy passes its own paired test (Phase 8).

## 5. 2025/26 rules (decision 3: the AFCON top-up)

**The 2025/26 AFCON top-up:** free transfers were topped up to five for GW16, whatever was banked (announced by FPL, July 2025; it applied straight after the GW15 deadline). In `config/scoring/2025-26.supplement.json`:

```json
"ft_topups": [{"gw_index": <gw_index of FPL GW16>, "amount": 5}]
```

`next_state` adds the amount and caps at `max_free_transfers` (5), so the result is exactly five, as announced. The gw_index comes from the 2025/26 schedule (`gameweek` table: GW numbers and deadlines only, no results).

**VERIFY #9 (FT reconstruction):**
- WC/FH was resolved on 2026-10-06.
- The AFCON top-up is settled by the announcement.
- The rebuild was checked live on 2026-10-11: team 4570069's inferred four FTs for GW6 matched the FPL site.

Recommendation: close #9 with this note.

Other 2025/26 rules are already in the supplement (defcon thresholds; no Free Hit in consecutive GWs, which doesn't matter with chips off).

## 6. Before the run (one PR, reviewed and merged first)

1. **Window t-test:**
   - add `window_t_test` in `fplopt.backtest.evaluate`;
   - add the A/A size test (§2);
   - report it next to the bootstrap.
2. **AFCON top-up** in `config/scoring/2025-26.supplement.json` (§5), with a rules test.
3. **`fplopt holdout run`:**
   - one command with every setting above fixed in code, so there are no flags to vary;
   - runs both comparisons and writes `results/phase6-holdout/`;
   - appends to `experiments.csv` (family `phase6 holdout`);
   - prints the gate verdict;
   - refuses to run if that family is already in the log;
   - runs from a clean checkout of a tagged commit (`phase6-holdout`), with its git sha logged.
4. **Pre-flight checks (metadata only, no results):**
   - `ep_next` is present at every 2025/26 deadline (snapshot counts);
   - the GW1 pool covers every club (every deadline, in fact).

   Any gap is reported and decided **before** the run. A v1 failure at a 2025/26 deadline isn't pre-checked: it would crash the run before any verdict, which §7 allows to fix and rerun.
5. **Docs:**
   - PLAN §9: the gate's statistic becomes the window t-test;
   - §12: these decisions;
   - §11: the two gate items resolved;
   - close #9.

## 7. Running it, and what happens next

- **One run.**
  - If it crashes before printing a verdict (a bug, a timeout), fix it and run again: no result has been seen.
  - Once any comparison result is printed, the run stands. No reruns, no setting changes, no extra variants.
- **Pass** (both gates): Phase 7 (Telegram bot, go-live for the user's team), with chips manual.
- **Fail** (either gate): no go-live from this run. The fallback `greedy:v1:threshold=2.0` is **reported only** (it's run in the same job against both baselines). Testing it for go-live on the holdout we just used would be a second look. Instead, the live 2026/27 shadow log (PLAN §8) becomes the evidence, judged with the same test at season end. Go-live slips, as PLAN allows.
- **Recorded either way:**
  - PLAN §7, a Phase 6 results section;
  - §9, the phase row;
  - §12, the verdict;
  - `experiments.csv`.
