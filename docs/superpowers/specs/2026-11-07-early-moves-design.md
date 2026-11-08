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

- **Main result: net points of E − L,** from the standard comparison:
  - `backtest compare --continuation own --reference b --k 1 --stride 1`, scored on real points;
  - 16 starts a season (`template@1,random:3@1,template@10,random:3@10,template@20,random:3@20,template@29,random:3@29`);
  - the SE is grouped by season.
  - This assumes the noise study's seven-season placebos pass. If they don't, the comparison method is fixed first.
  - With 2 seasons the SE grouped by season has 1 degree of freedom, so the pooled-window SE is reported next to it, and the main result uses the larger of the two.
- **Value gain:** money kept by E's moves at A versus at the deadline, from the snapshots (same transfers, priced at both moments), in £m a season and converted to points at 0.395 xP per £1m per GW.
- **Information cost:** the share of decisions where E's transfers at A differ from L's at the deadline, and the per-decision point difference on those decisions only.
- **News risk:** the share of players E buys at A who are flagged (status not `a`) at some snapshot between A and the deadline.

**Gate:** E is worth an alert if E − L > 0 with one-sided p < 0.10. A 10-point effect is about 1.7 SE at the expected SE of ~6. Logged as family `early moves develop`. No tuning grid: E has no parameters.

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
6. **Driver** `dev/early_moves.py`: runs the comparison and computes the value gain, information cost and news risk; writes `docs/research/2026-11-early-moves.md`.

## Not in this spec

- A variant that moves early only when a price change is forecast.
- The bot's "move tonight" alert. It follows only if E passes, and the 2023/24–2024/25 check runs first, once.
- Chips with early moves.

## Risks

- **Two seasons are noisy:** an SE of ~6 sees 10 points only about half the time. The value gain and information cost are reported so a near miss can be read.
- **News between A and the deadline** (cups, Europe, press conferences) is the cost being measured. Before 2021 there are no status flags, which is one more reason these seasons are out.
- **Lockdown at 09:00 is an approximation** for seasons before 2026/27. The real first chance to move after the last match is earlier, so the gain is understated.
