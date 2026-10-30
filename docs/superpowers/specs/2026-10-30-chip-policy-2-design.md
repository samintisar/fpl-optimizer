# Chip policy 2: beat the rule baseline H (design)

Sub-project 2 of the chip research (sub-project 1: `2026-10-29-chip-diagnostics-design.md`,
report `docs/research/2026-10-chip-diagnostics.md`). Approved 2026-10-30.

## Goal

A chip policy that beats H (`chiprules:v1:…,t_tc=8.3,t_bb=10.9,k_fh=4,t_wc=12.87`) on develop
after deflation, frozen before the GW20 deadline (~2026-12-29) and shadowed live over set 2
(GW20–38). Validate (2023/24–2024/25) and the holdout (2025/26) are not used.

## What the diagnostics showed (develop, today's model)

- P (optimizer, pooled terminal values × 1.0) vs H: +0.33/GW, p 0.26; both beat S0 by ~2.5/GW.
- Holding a chip is overvalued in set 1 (realized ÷ table 0.52–0.62 for TC) and undervalued in
  set 2 (1.13–1.31): one table pooled over both windows.
- Chips are worth 1.5–2.5× in doubles; the TC forecast is calibrated on doubles (slope 1.04),
  flat on singles (0.39).
- Free Hit forecast slope 0.4; Wildcard forecast uninformative; Bench Boost calibrated.

## Variants (fixed list, pre-registered here)

All are `optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2,chips=1` with a different
terminal (holding) value or chip rule. Every terminal-value table is estimated
**leave-one-season-out**: the table used in develop season s comes from the S0 gains of the
other six develop seasons.

| id | variant | change |
|---|---|---|
| V1 | per-window, k = 0.6 | separate tables for set 1 and set 2 (by `chip_id` window), × 0.6 |
| V2 | per-window, k = 0.8 | as V1, × 0.8 |
| V3 | per-window, k = 1.0 | as V1, × 1.0 |
| V4 | hybrid | best of V1–V3 for FH/BB/WC; TC by H's rule (`t_tc = 8.3`), never by the optimizer |
| V5 | double-aware | best of V1–V3; the TC and BB tables split by whether a double GW is known (schedule as of the deadline) in the rest of the window after the horizon |
| V6 | FH shrink | best of V1–V3; chip scenarios ranked with the Free Hit's gain over the no-chip plan × 0.4 |

"Best of V1–V3" = the highest raw (undeflated) full-run window t-test mean vs H; V4–V6 run
after V1–V3. Ties: the smaller k.

- **Spec strings** (`BASE = horizon=6,decay=0.75,max_hits=1,hit_margin=2`):
  V1–V3 `optimizer:v1:BASE,chips=1,chip_shrink=k,chip_split=1,loso=1` (k = 0.6/0.8/1.0);
  V4 `…,chip_shrink=k*,chip_split=1,loso=1,tc_rule=8.3`;
  V5 `…,chip_shrink=k*,chip_split=2,loso=1`;
  V6 `…,chip_shrink=k*,chip_split=1,loso=1,fh_scale=0.4`.
- **V5's "known double"**: a GW (by `gw_index`) where some club has ≥ 2 league fixtures in the
  as-of schedule (`AsOfView.schedule`), at most `DOUBLE_LEAD = 10` GWs after the decision GW.
  Fixture snapshots exist only for seasons archived live (2026/27 on), so in every develop
  season the as-of schedule is the final one: a double counts as known up to 10 GWs ahead even
  if it was announced later. **V5 is therefore information only** (amended 2026-10-30, user
  decision): it is run and reported but can't be the winner, and the deflation counts the
  other five variants (N = 5).
- **V6's Free Hit shrink** ranks chip scenarios by total objective − (1 − 0.4) ·
  max(0, Σ over Free Hit GWs t of (plan GW t objective − no-chip plan GW t objective)); the
  MILPs are unchanged.
- **V4's rule**: the optimizer plans without Triple Captain scenarios; if plan #1 plays no
  chip this GW and Triple Captain is available, it is played when the captain's xP ≥ 8.3 or
  this is the last GW of its window (re-planned with TC fixed in this GW).

## Runs

- Develop 2016/17–2022/23, starts `template@1,random:3@1,random:3@20` (7 runs per season, the
  diagnostics' grid, so H and P are comparable with sub-project 1), v1 xP.
- Arms: V1–V6, plus fresh H and P (pooled table, in-sample, k = 1.0) for reference.
- On the server test checkout (`/mnt/ibrokeass/fpl-tests`, `nice -n 15`, ≤ 12 jobs), never in
  `~/fpl-optimizer`.
- Logged to `results/experiments.csv`, family `chip-policy-2026-11` (N = 5 eligible variants).

## Statistic and bar

- Each variant vs H: `backtest compare`'s full-run window t-test on 8-GW cells (realized
  points), with the xG-scored mean reported alongside.
- **Pass:** the best variant's deflated mean (mean − SE·√(2 ln 5)) > 0 **and** one-sided
  p < 0.10, with a positive xG-scored mean.
- The family's runs include H, P and V5 (9 runs); the deflation counts the 5 eligible
  variants V1–V4, V6 (N = 5).
- Known bias against the variants: H's thresholds were tuned in-sample on develop; the
  variants' tables are out of season.
- No pass: nothing changes. `/chips` keeps its "information only" label.

## Freeze and live shadow (set 2)

- The winner (or, with no pass, the best variant by deflated mean, for information) is
  frozen in git before the GW20 deadline: spec string, table files and commit.
- From GW20 to GW38, the live job logs, for each registered user's squad, both the winner's
  and H's chip call for the coming GW (SQLite, keyed by `user_id`, deadline). Advice doesn't
  change.
- At season end each logged call is scored on its own GW only: TC = captain points, BB = bench
  points, FH = best-XI difference between the FH squad and the squad held. Wildcards are
  reported, not scored (their effect spans six GWs and the user's later transfers).
- The winner drives chip advice in 2027/28 only if it beats H in the shadow (and passed on
  develop).

## Code

1. **Per-window tables:** `chip_values` estimates values per (chip, window) into
   `config/chips/terminal-values-window.json` (format 2, keys `name@k`, k the window's
   ordinal); `chips.chip_value` looks up `name@k`, falling back to `name` (the pooled
   `terminal-values.json` is unchanged and still loads).
2. **Leave-one-season-out:** `fplopt chips values --loso` writes one table per held-out
   season; the policy spec loads the table for the season being replayed (`loso=1`).
3. **Hybrid:** optimizer key `tc_rule=<threshold>`: TC excluded from the chip search and
   played by H's rule.
4. **Double-aware:** `terminal-values-double.json`, keys `name@k+d` (d = a double known in
   the as-of schedule at the deadline, lead 10); a `+d` cell with fewer than 30 observations
   falls back to `name@k`. On the 7-start grid only 4 runs per season cover set 1 (28 per
   cell, 24 leave-one-season-out), so set 1 always falls back: V5 is double-aware in set 2
   only.
5. **FH shrink:** optimizer key `fh_scale=<k>`: chip scenarios are ranked by total objective
   − (1 − k) · max(0, the Free Hit GWs' objective over the no-chip plan's); MILPs unchanged.
6. **Shadow logger:** in `fplopt.bot.live` after the recommendation; a table in
   `state/bot.sqlite`; a scoring command at season end.

Each piece has unit tests with pinned values; items 1–5 land before the runs, item 6 before
GW20.
