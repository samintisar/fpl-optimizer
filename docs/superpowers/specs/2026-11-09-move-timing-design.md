# Move timing: make a move early only when a price change is due tonight

**Date:** 2026-11-09. **Status:** parked 2026-11-09 after Gate N failed (rises overconfident; `docs/research/2026-11-night-price-model.md`); T was not built or run. Supersedes the fixed-time study (`2026-11-07-early-moves-design.md`, policy E), which is parked before any server run; its code (early views, prices at the decision moment, money kept, the leakage probe) is reused here.

## Why

Prices change overnight (about 02:30 UK, at most £0.1m a night and £0.3m a GW) when net transfers pass a threshold tied to ownership. The live planner L makes its moves at the deadline, after the changes of the nights before it:
- moving right after the previous deadline would have kept about £2.3m a season (sizing, `docs/research/2026-11-price-sizing.md`); E, moving every transfer at the lockdown, kept £2.3m on 2022/23's template start in a local smoke run;
- but moving every transfer early means deciding on staler news for all of them.

The user's direction (2026-11-09): don't move at a fixed time. Move as close to the deadline as possible, except when a price change on a player being moved is due tonight; then make that move tonight. The price only decides **when**, never **who** (Stage C chased price moves in its choice of players and lost points; PLAN §12, 2026-11-07).

## The policy

**Policy T** is L (`optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2`, no chips) with `timing=night`:

1. **Nights.** For a GW after the season's first, the decision nights are those from the previous GW's lockdown up to the night before the deadline. Night n's decision moment is 22:00 UK that evening: the view sees the data as of then (the evening snapshot at about 18–19 UK in 2021/22–2022/23, before the ~02:30 change). The first GW has no nights.
2. **Each night:**
   - L plans on the night's view and pool (`as_of_early(deadline, night)`), from the squad as it stands (moves made on earlier nights applied).
   - Plan #1's moves for this GW are paired out → in by position, as `Transfer { out_key, in_key }` already pairs them.
   - The nightly model gives each player P(rise tonight) and P(fall tonight). A pair's expected saving tonight, in tenths: P(in rises) + P(out falls) × s(out), where s is the fall in the out-player's selling price if his price drops by 0.1 (1 if he's at or below his purchase price, else 0 or 1 by FPL's rounding).
   - A pair is made tonight if its expected saving is at least τ (default 0.5: more likely than not to save £0.1m). Made pairs are bought and sold at the night's prices.
   - Free transfers and hits count as usual: a move made tonight uses this GW's free transfers, so the state carries `free_transfers − moves made` (floored at 0), which gives the same next-GW free transfers and hits as FPL's rule.
3. **At the deadline:** L plans from the squad with the night moves applied: the remaining moves (at deadline prices) and the lineup, captain and vice. It may make further moves, or even reverse a night move (at a cost), as a manager could.
4. **Seasons:** T needs intraday snapshots and is refused before 2021/22 (as E is).

Nothing changes when no change is due on a planned move: T then is L.

## The nightly model

A player-night classifier: rise / fall / no change tonight.

- **Rows:** every pool player × night with an evening snapshot, 2020/21's last 65 nights and 2021/22 for training; label = the next morning's price (first snapshot after 04:00 UK) minus the evening's.
- **Counts (checked 2026-11-09):** a season has about 245,000–290,000 player-nights, with 450–620 rises and 1,340–1,560 falls (2021/22: 563 / 1,354; 2022/23: 454 / 1,557).
- **Features** (from the view only):
  - net transfers since the player's last price change (cumulative from `transfers_in_event` / `transfers_out_event`, which reset each GW) and today's;
  - ownership; net transfers relative to ownership;
  - price; changes so far this GW (the £0.3m cap) and nights since the last change (another rise soon after one needs many more transfers);
  - status and chance of playing; a flag that went straight from red to none (price locked for a week);
  - days to the deadline; GW index.
- **Model:** LightGBM (the pinned `fplopt-lgbm`), two binary boosters (rise, fall). **VERIFY** that `fplopt-lgbm` runs the `binary` objective as LightGBM does; if not, a regression on the change in tenths, clipped.
- **Gate N (2022/23, develop):** log loss of each booster against two baselines: the base rate and a threshold rule on net transfers per owner (fitted on the training seasons); one-sided Diebold–Mariano by night, p < 0.10; calibration table in the note. If N fails, T is not run.
- **Calibration (amended 2026-11-09, before Gate N's run, after the 2021/22 smoke run):**
  - **The problem:** the smoke run's raw probabilities were overconfident above 0.1. For example, predicted 0.55 against an observed 0.14 for rises. T acts on the probabilities themselves (τ = 0.5), so ranking well is not enough.
  - **The fix:**
    - Both the model and B1 are recalibrated by Platt scaling: a logistic regression of the label on logit(p).
    - The scaling is fitted on out-of-fold predictions over the training rows only: 5 folds of contiguous training nights, with each fold's boosters fitted on the other four, using the same parameters.
    - The final boosters are fitted on all training rows, and the map is applied to their predictions.
  - **What the gate scores:** the log-loss comparisons use the calibrated probabilities. The raw ones are reported too.
  - **An added gate condition, per target, for the model:** among the evaluated rows with calibrated p ≥ 0.3 (the region where T can act), there must be at least 30 rows, and the observed rate must be at least the mean calibrated p − 0.10.
  - **The overall rule:** N passes only if all four log-loss tests pass and both calibration conditions hold.

## Measurement

Develop season 2022/23 only (2021/22 trains the model). 2023/24–2024/25 stay clean until a frozen T passes; 2025/26 is never read.

- **Value gain V:** the money kept by T's night moves (each pair: deadline buy price minus the night's, plus the night's selling price minus the deadline's, as `early_kept`), cumulated over the season and converted at 0.395 points per £1m per GW, on the full-season starts (the 4 `@1` starts).
- **Information cost C:** the standard comparison T vs L (PLAN §5: per decision, k = 1, realized points, 16 starts), on 2022/23. T makes L's moves at L's deadline in most GWs, so the arms should agree often and C's SE should be small (the SE grows with disagreement).
- **Net = V + C,** SEs combined; **gate:** net > 0, one-sided p < 0.10 (normal), family `move timing develop`.
- **Also reported:** the share of GWs and of moves made at night; the share of night moves whose price change happened (hit rate); money kept per night move; the news risk (night buys flagged before the deadline).
- **τ:** 0.5 is the pre-registered setting. 0.3 and 0.7 run as sensitivity rows in the same family; the gate is judged on 0.5 only.
- **If T passes:** frozen and tagged; one run on 2023/24–2024/25 with the same gate; only then the bot's "move tonight" alert (its own spec).

## Build (Rust)

1. **Night features and labels** (`fplopt-features`, view-only, as `price_features`), with leakage checks.
2. **Nightly model** (`fplopt-models`): fit, predict, a walk-forward eval command (`research night-eval`), logged.
3. **Simulator:** several decision moments per GW: each night's partial decision (a set of pairs, each with its own `priced_at`), then the deadline decision on the updated squad; per-move `priced_at` replaces the decision-level one for T; `early_kept` summed over a GW's night moves; per-decision replay at each move's own pool.
4. **Spec keys:** `timing=night` (default `deadline`), `tau=`; refused with chips, moving prices, `tc_rule` and `early=1`.
5. **Probes:** `probe:timing_*` runs a night step with the check's deadline as the cutoff.
6. **Driver** `dev/move_timing.py`: Gate N, T's backtest, the comparison and the report; note `docs/research/2026-11-move-timing.md`.

## Not in this spec

- Letting price forecasts change which players are moved.
- Moving before a fall in a player not already planned to be sold.
- The bot alert.
- Chips.

## Risks

- **Snapshot timing:** the evening snapshot is 3–8 h before the change; transfers in the last hours are missed, so the model is a little worse than a live alert at 01:00 would be.
- **One develop season:** the policy's points are judged on 2022/23 alone; the money and the model's accuracy are measured well, the points cost less so.
- **Price rule drift:** FPL's threshold changes between seasons; the model is trained on one season back.
- **Pairing:** splitting a GW's plan into pairs and making some early assumes the remaining moves still fit the budget at the deadline; the deadline re-plan handles it (it may change the remaining moves).
