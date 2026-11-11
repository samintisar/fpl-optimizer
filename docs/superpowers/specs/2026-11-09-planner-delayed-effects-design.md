# Hits and decay, judged on windows that count what a hit buys

**Date:** 2026-11-09. **Status:** closed unbuilt by the owner, 2026-11-10 (the review was closed; `docs/research/2026-11-system-review.md`, *Closed*). Nothing is built. Review: `docs/research/2026-11-system-review.md` §2.

## Why

- The standard comparison (PLAN §5) sums `net_points` over the decision's own gameweek (`crates/fplopt-backtest/src/evaluate.rs:419-421`). A hit is charged 4 points there and credited one gameweek of a player who is then held a mean of 9. Any setting that trades this gameweek for later ones is judged with a bias toward doing nothing.
- "Hit margin 3 beats 2 by +2.8 (SE 2.1)" comes from that method. On the same runs the whole-season difference is −12.6 (SE 7.7). Margin 3 removed 5.3 hit points a unit and gained 2.8 in the first gameweek, so the hits it removed were earning back two thirds of their cost in one gameweek of nine.
- Decay 0.75 was the best of 8 cells in one develop grid with the October v1. Decay 0.85 wins 3 of the 4 matched pairs in that grid, and the chosen cell fell from +2.22 a gameweek on develop to +0.74 on validate. Effective horizon at 0.75 is 3.3 gameweeks.
- None of this has been run with the current v1.

**Size:** +5 to +15 points a season if the direction holds, low confidence. It can also come out at 0.

## Change

No code in the planner. Existing spec keys only. One addition to the comparison's report (below).

## The comparison

- **Design:** `backtest compare --continuation own --reference b --k 4 --stride 4`, all seven develop seasons, the standard 16 starts, realized points. Four-gameweek windows with each arm continuing its own plan credit a hit with four gameweeks of the player it bought. The whole-season difference is reported next to it.
- **Expected SE:** about 5–6 points a season at k = 4 (the noise study's 10.6 on two seasons, scaled to seven) and about 8 for whole seasons. Only an effect of about 10 points or more can be seen. That is stated here so that a null is read as "smaller than 10", not "zero".
- **Placebo first:** the four jitter pairs of the noise study, rerun at k = 4. The method is used only if the z-scores' RMS is at most 1.2 and the mean of the four is within 2 SE of zero. About 8 hours of server time; the pairs' xP caches exist.
- **Report addition:** `backtest compare` prints, per arm, hit points and transfers a season, and the mean points of hit gameweeks' bought players over the window. No change to any existing number.

## Variants (develop, family `planner delayed effects`, N = 5)

Base B = `optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2`.

| Arm | Change from B |
|---|---|
| A1 | `hit_margin=1` |
| A2 | `hit_margin=0` |
| A3 | `max_hits=2` |
| A4 | `decay=0.85` |
| A5 | `decay=0.85,hit_margin=1` |

Each is one compare run against B, about 2 hours at `--jobs 14`: 10 hours, plus the placebos.

## Develop gate

The best arm by the k = 4 mean is frozen if all of:

1. its k = 4 mean, deflated for five variants (mean − SE·√(2 ln 5), with the larger of the pooled SE and the SE over season means), is above 0;
2. its whole-season mean is above 0;
3. at least 5 of the 7 seasons are positive at k = 4.

If no arm passes, nothing changes and the note records the means. The live settings stay.

## Check (2023/24–2024/25, once)

The frozen arm against B, same design, tag `planner-delayed-check`, family `planner delayed check`, logged before printing, refused on a dirty checkout or a second run.

- **Pass:** k = 4 mean ≥ 0 in realized and xG-scored points, and whole-season mean ≥ 0. A no-harm bar, as the v1 promotion used: two seasons cannot confirm a 10-point effect, so develop carries the evidence and the check guards against a reversal.
- These seasons were used once for the v1 promotion's no-harm bar and are reserved for the timing policy's check too. Each frozen policy gets one run; the note lists every run made on them.

## Not in this spec

- Free-transfer values and a squad value after the horizon (review §5, item 9). Both are confounded with decay and wait for this result.
- The MIP gap and best-of-three (item 5). They have no delayed effect, so the standard k = 1 comparison is fair for them, and they need a spec key first.
- A gain shrink. Realized gain per free transfer is flat and positive below a predicted gain of 8 (+3.4 to +4.3), so there is nothing to shrink.

## Risks

- **Power.** If the true effect is 5 points, this study will most likely end with no arm frozen.
- **k = 4 has its own blind spot:** money and free transfers carried past the window are still not counted. The whole-season condition is the guard.
- **The placebo may fail.** Then the fallback is the whole-season difference alone on seven seasons (SE about 8), which can see only effects above about 15 points.
