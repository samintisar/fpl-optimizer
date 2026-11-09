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
4. **The season's first GW** has no lockdown before it and is decided at the deadline, as L does. Every later GW is decided at A, a start state's first GW included (both arms share the start state, built at the deadline).
5. **Seasons:** E is refused before 2021/22. Without snapshots, the pool at A has no prices: deadline prices are visible only from an hour before the deadline.

## Measurement

Develop seasons 2021/22–2022/23 only: the snapshots that give prices at A start in 2021. The 2025/26 holdout is never read, and 2023/24–2024/25 stay unused until a frozen version passes on develop.

Amended 2026-11-08, before any build, after the noise study's placebos:
- the standard comparison (PLAN §5) counts only one GW's points per decision, so it cannot see money kept;
- its SE grows with how often the arms disagree.

So the net effect is split into a value gain and an information cost:

- **Disagreement: how often E's decisions at A differ from L's at the deadline.** This is the comparison's `same_decision` share (transfers and lineup; priced moment ignored). The comparison takes about 35 min on 2 seasons, so it needs no separate pre-check (amended 2026-11-08 while planning). The placebo pairs disagreed in 94% of windows, with an SE of 9–13 a season over 7 seasons, about 18–25 over 2.
- **Value gain V:**
  - the money kept by E's moves (bought at A rather than at the deadline, and sold at A's selling price rather than the deadline's), from the snapshots, along E's own runs;
  - in £m a season, and in points as Σ over the GWs of the cumulative money kept (£m) × 0.395;
  - per season from the full-season units only (the 4 `@1` starts a season; 8 units), so it is a full season's money;
  - nearly deterministic, so its SE (over units) is small.
- **Information cost C:**
  - E − L per decision from the standard comparison (`backtest compare --continuation own --reference b --k 1 --stride 1`, real points, 16 starts a season: `template@1,random:3@1,template@10,random:3@10,template@20,random:3@20,template@29,random:3@29`), which counts the points the earlier transfer choice wins or loses in its GW;
  - with 2 seasons the SE over season means has 1 degree of freedom, so the pooled-window SE is reported next to it and the larger of the two is used.
- **Net = V + (E − L per decision),** with the SEs combined, treating the two as independent.
- **News risk:** the share of players E buys at A who are flagged (status not `a`) at some snapshot between A and the deadline.

**Gate:** E is worth an alert if the net > 0 with one-sided p < 0.10 (normal approximation on the combined SE). Logged as family `early moves develop`. No tuning grid: E has no parameters. The note always reports V, C, the disagreement rate and the news risk, so a near miss can be read.

## Build (Rust)

1. **Store** (`crates/fplopt-store`): a view keeps its deadline (which GW it is for) apart from its cutoff (what it sees).
   - `AsOfView::for_deadline(d)` gives the view for a later deadline d with the same cutoff, and `DataStore::as_of_early(deadline, cutoff)` = `as_of(cutoff).for_deadline(deadline)`.
   - `as_of(d)` is the same view as before (cutoff = deadline).
   - `earlier(c)` keeps the cutoff at most c.
   - Visibility (`table`, `latest`, `schedule`) uses the cutoff; `gameweek_for_deadline` and `deadline()` use the deadline.
   - **Verified 2026-11-08** (local `data/`):
     - in 2021/22–2022/23 the last snapshot before A is at most 2.6 h earlier, with a median of 11–12 more before the deadline;
     - the coming GW's row is visible at A for every GW;
     - A is a median of 3 days (at least 4.6 h) before the deadline.
     - The pool at A comes from that snapshot (`player_gw` deadline prices are hidden until an hour before the deadline).
2. **Caches:** pools, xP frames (memory and disk) and v1 fits are keyed by (deadline, cutoff). A deadline view's disk path is unchanged, so existing caches stay valid.
3. **Simulator** (`crates/fplopt-backtest`):
   - with `early=1`, a GW after the season's first runs the policy on the early view at A = previous lockdown + 1 µs (the outcomes become visible at the lockdown, as `read_outcomes` sees them);
   - the lineup is re-solved at the deadline with the GW's transfers fixed (`solve_plan` with the post-transfer squad and no further transfers);
   - the decision carries `priced_at = A`, so it is applied at A's pool, and replayed at A's pool in the per-decision arms.
   - `gws.parquet` gains `decided_at` and `early_kept` (tenths: the deadline's buy prices of the ins minus A's, plus A's selling prices of the outs minus the deadline's).
4. **Spec key:** `early` (0/1, default 0, optimizer only; refused with chips, `prices` other than fixed, and `tc_rule`). Label suffix `early`.
5. **Tests:**
   - `early=0` leaves the policy names and the backtest output unchanged (diffed on a local run);
   - the early view never sees a row at or after the cutoff;
   - E's buy and sell prices are A's;
   - the deadline lineup step can't change transfers;
   - a leakage probe `probe:early_optimizer_ep_next_template` runs E's step at the next deadline with the check's deadline as the cutoff, so `check leakage` corrupts everything at or after A.
6. **Driver** `dev/early_moves.py`: runs E's backtest and the comparison, and computes the value gain, information cost, disagreement and news risk; the note `docs/research/2026-11-early-moves.md` is written from its report.

## Not in this spec

- A variant that moves early only when a price change is forecast.
- The bot's "move tonight" alert. It follows only if E passes, and the 2023/24–2024/25 check runs first, once.
- Chips with early moves.

## Risks

- **Two seasons are noisy:** the information cost's SE depends on the disagreement rate, which the comparison reports.
- **A's snapshot is after the first night's price change** (snapshots at about 06:30–07:30, before the 09:00 lockdown), so A's prices already include that night's moves.
- **News between A and the deadline** (cups, Europe, press conferences) is the cost being measured. Before 2021 there are no status flags, which is one more reason these seasons are out.
- **Lockdown at 09:00 is an approximation** for seasons before 2026/27. The real first chance to move after the last match is earlier, so the gain is understated.
