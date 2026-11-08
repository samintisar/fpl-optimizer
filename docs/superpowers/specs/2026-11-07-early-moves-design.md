# Early moves: transfers at the lockdown, before prices move

**Date:** 2026-11-07. **Status:** approved design.

## Why

Prices change overnight in the days before a deadline, mostly for the players managers are buying and selling. The live planner makes its transfers at the deadline, after those changes.

- **The sizing** (develop 2022/23, the live planner's own transfers) found that making the same transfers right after the previous deadline would have kept about £2.3m a season. 50% of moves gained, 6% lost, and 3.3% of the bought players were flagged before the deadline. Moving 24 h before the deadline kept only about £0.6m.
- **What the money is worth:** £1m is worth about 0.395 xP per GW to the planner (`docs/research/2026-11-price-sizing.md`). So money kept gradually over a season is worth roughly 10–20 points, before the cost of deciding on older information.

That moment was unrealistic, because the gameweek's results aren't known right after the deadline. This design tests a realistic one, and if early moves pay, the bot's "move tonight" alert follows.

## The policy

**Time A** is the gameweek's lockdown: 09:00 UK the day after its last kickoff, when the archive makes its results visible (PLAN §3). That is usually 2–4 days before the next deadline, and about a day before a midweek one. It misses the first night of price changes after the last match, so it slightly understates the gain.

**Policy E** is the live planner L (`optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2`, no chips) with `early=1`:

1. **At A:**
   - E plans for the coming deadline with a view cut at A, so its data and snapshot prices are as of A.
   - It makes plan #1's transfers for that GW at A's prices: buying at A's price, selling at A's selling price under FPL's selling rule.
   - Free transfers and hits are counted as usual.
2. **At the deadline:** E re-solves with those transfers fixed, choosing only the captain, vice and XI with deadline data.
3. **Afterwards:** prices move as they did in reality, so E's bank, squad value and purchase prices differ from L's from then on.
4. **The first decision of a unit** (start states) and GWs without a lockdown before them (GW1) are decided at the deadline, as L does.

## Measurement

Develop seasons 2021/22–2022/23 only: the snapshots that give prices at A start in 2021. The 2025/26 holdout is never read, and 2023/24–2024/25 stay unused until a frozen version passes on develop.

Amended 2026-11-08, before any build, after the noise study's placebos:
- the standard comparison (PLAN §5) counts only one GW's points per decision, so it cannot see money kept;
- its SE grows with how often the arms disagree.

So the net effect is split into a value gain and an information cost:

- **Pre-check: how often E's transfers at A differ from L's at the deadline.** At every decision of L's own runs, plan at A and at the deadline and count the differences. It is cheap (no simulation) and gives a power estimate before the comparison runs. The placebo pairs disagreed in 94% of windows, with an SE of 9–13 a season over 7 seasons, about 18–25 over 2.
- **Value gain V:**
  - the money kept by E's moves (bought at A rather than at the deadline, and sold at A's selling price rather than the deadline's), from the snapshots, along E's own runs;
  - in £m a season, and in points as Σ over the remaining GWs of the cumulative money kept (£m) × 0.395;
  - nearly deterministic, so its SE (over starts and seasons) is small.
- **Information cost C:**
  - E − L per decision from the standard comparison (`backtest compare --continuation own --reference b --k 1 --stride 1`, real points, 16 starts a season: `template@1,random:3@1,template@10,random:3@10,template@20,random:3@20,template@29,random:3@29`), which counts the points the earlier transfer choice wins or loses in its GW;
  - with 2 seasons the SE over season means has 1 degree of freedom, so the pooled-window SE is reported next to it and the larger of the two is used.
- **Net = V + (E − L per decision),** with the SEs combined, treating the two as independent.
- **News risk:** the share of players E buys at A who are flagged (status not `a`) at some snapshot between A and the deadline.

**Gate:** E is worth an alert if the net > 0 with one-sided p < 0.10. Logged as family `early moves develop`. No tuning grid: E has no parameters. If the pre-check shows E disagrees with L so often that a 10-point net would be under 1.3 SE, the note reports V, the disagreement rate and the news risk, and we decide before running the comparison.

## Build (Rust)

1. **Store** (`crates/fplopt-store`): `DataStore::as_of_early(deadline, cutoff)`, with cutoff ≤ deadline.
   - It is a view for the deadline's gameweek (`gameweek_for_deadline` and everything keyed on the deadline unchanged) that keeps only rows with `available_at < cutoff`.
   - With cutoff = deadline it is the same view as `as_of(deadline)` (test).
   - It never shows a row with `available_at ≥ cutoff` (test).
   - The player pool and prices then read the last snapshot before A.
   - **VERIFY:**
     - the snapshot density in 2021/22–2022/23 (fplcache), i.e. how close to A the last price before A is;
     - that the pool and features run on a view whose cutoff isn't a deadline. Rows with `available_at = deadline_t − 1h` (PLAN §3) are hidden at A, so the pool must fall back to the latest snapshot.
2. **xP at A:** the model's prediction on the early view, cached on disk like the deadline frames, keyed by (deadline, cutoff).
3. **Simulator** (`crates/fplopt-backtest`):
   - with `early=1`, each GW with a lockdown before its deadline (not the unit's first decision) runs the transfer step on the early view at A's prices;
   - then the lineup step at the deadline with transfers fixed.
   - Purchase prices are A's prices.
   - The output gains `decided_at` per GW.
4. **Spec key:** `early` (0/1, default 0, optimizer only; refused with chips and with `prices=forecast`). Label suffix `early`.
5. **Tests:**
   - `early=0` leaves the planner and backtest goldens unchanged;
   - the early view never sees anything at or after A;
   - E's buy and sell prices are A's;
   - the captain and XI step can't change transfers;
   - `check leakage` covers the early view (corrupt everything at or after A, and E's decision at A must not change).
6. **Driver** `dev/early_moves.py`: runs the pre-check (needs a Rust entry point that plans at A and at the deadline along L's runs) and the comparison, and computes the value gain, information cost and news risk; writes `docs/research/2026-11-early-moves.md`.

## Not in this spec

- A variant that moves early only when a price change is forecast.
- The bot's "move tonight" alert. It follows only if E passes, and the 2023/24–2024/25 check runs first, once.
- Chips with early moves.

## Risks

- **Two seasons are noisy:** the information cost's SE depends on the disagreement rate. The pre-check gives it before the comparison runs.
- **News between A and the deadline** (cups, Europe, press conferences) is the cost being measured. Before 2021 there are no status flags, which is one more reason these seasons are out.
- **Lockdown at 09:00 is an approximation** for seasons before 2026/27. The real first chance to move after the last match is earlier, so the gain is understated.
