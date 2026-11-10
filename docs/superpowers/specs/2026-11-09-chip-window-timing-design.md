# Chip timing inside a closing window, and what the alert says

**Date:** 2026-11-09. **Status:** proposed, not approved. Nothing is built. Review: `docs/research/2026-11-system-review.md` §3.

## Why

- Playing all eight chips is worth about +85 points a season (SE 13) over playing none, by the rules or by the planner. That is the largest single item between the holdout's 2,240 and the target.
- The chip planner has a defect in the last gameweeks of a window. A Triple Captain or Bench Boost gain in horizon gameweek h is multiplied by `decay^h` (`crates/fplopt-optimize/src/model.rs:789-795`), while the value of holding the chip is undecayed and becomes 0 once the window's end is inside the horizon (`chips.rs:122-126`). With decay 0.75 a gameweek three ahead must be worth 2.4 times this one to be preferred, so the planner plays the chip at once.
- In the runs it plays one chip a gameweek from five gameweeks before a window closes (validate set 1: all 8 Triple Captains and all 8 Free Hits at gameweeks 14–15). On develop, set 2, the gameweek it picks realized 6.6 (TC) and 6.0 (BB) against 14.4 and 11.3 for the best forecast gameweek left.
- **Live:** set 1 closes at gameweek 19. From the gameweek-14 deadline the alert will tell the user to play every remaining set-1 chip now.

**Size:** about **+5 points a season (0 to +10)** for the fix, from the set-2 cost weighted by how often the planner reaches that point with the chip unused (19 of 49 runs for TC, 15 of 49 for BB). The wording changes are worth no points; they stop the alert from claiming more than was tested.

## Change 1: chip gains are compared undecayed

New optimizer key `chip_decay` (0/1, default 1 = today's behaviour, so every existing spec and golden is unchanged).

With `chip_decay=0`:
- the Triple Captain's extra captain term and the Bench Boost's bench term in horizon gameweek t enter the objective without the `decay^h` factor (the ordinary lineup, captain and bench terms keep it);
- the chip search's Triple Captain and Bench Boost bounds use the same undecayed gains;
- Wildcard and Free Hit scenarios are unchanged. Their gain is a difference of whole plans, and the review found no timing signal in them (the Wildcard's forecast gain has a slope of 0.10–0.22 on the realized one).

The holding value stays as it is: it is already in undecayed points, so with this change a chip played in the horizon and a chip held are on the same scale.

## Change 2: the alert's chip section

1. When a chip's window ends inside the horizon, the line reads "play X in GW g: the best gameweek left by forecast (+a xP; GW now +b)" and never "vs keeping it".
2. The evidence line states what was tested with the live model: "over nine seasons, playing every chip was worth about 85 points a season; the planner is not proven better than the simple rules".
3. When the planner's call is a Wildcard or Free Hit, the alert says that `/chips` holds the transfer plan that goes with it. (Already so for those two; unchanged, listed for completeness.)

Change 2 has no effect on which chip or gameweek is suggested.

## Measurement

Validate is spent for chips and 2025/26 is the holdout, so there is no unseen season. The bar is therefore a no-harm bar on develop plus a mechanism check, and the out-of-sample evidence is the live shadow.

- **Arms** (develop 2016/17–2022/23, current v1, starts and settings of chip policy 2): P = `optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2,chips=1,chip_shrink=0.8` as live; P′ = P + `chip_decay=0`. One variant, so no deflation. Family `chip window timing`.
- **Statistic:** the chip study's, the whole-season window t-test on 8-gameweek cells, P′ − P.
- **Mechanism table (reported):** per chip and set, the share of plays in the window's last five gameweeks, and realized points per play.
- **Gate, all of:**
  1. P′ − P ≥ 0 in realized points and in xG-scored points (means; no significance asked, since one develop family cannot give it);
  2. set 2's Triple Captain and Bench Boost realize more per play under P′ than under P;
  3. P′ − H is not below P − H.
- **Cost:** two chip runs of about 35 minutes each on the server at `--jobs 14`.

**If the gate passes:** the live chip suggestion uses `chip_decay=0`, and `chip_calls` logs P, P′, H and V6 from then on. Chips stay the user's decision, as now.
**If it fails:** only change 2 ships, and the note records the numbers.

Change 2 does not wait for the gate. It should be live before the gameweek-14 deadline.

## Shadow

From gameweek 20 the `chip_calls` log already holds H and V6. P′ is added. At the end of set 2 the three are scored on the user's squads against what was played. Three or four users give no significance; the log is for the direction and for next season's design.

## Not in this spec

- A chip policy that plays chips without the user. No arm has beaten the rules at the pre-registered bar.
- Double-aware rules for Triple Captain and Bench Boost (+2 to +4 over H by an offline bound; review §5, item 8). They can be sized with no solves and get their own note first.
- V6's one-sided Free Hit shrink. V6 is frozen; the finding is recorded in the review.

## Risks

- **Backtests know doubles all season** (final fixture list before 2026), so the develop gain of waiting for a double is optimistic. Live, a double is visible only once scheduled, which inside a six-gameweek horizon it usually is.
- **Undecayed chip gains weigh later forecasts fully**, and later xP is less accurate. The holding value already has this property, and the mechanism table shows whether late plays realize what they forecast.
