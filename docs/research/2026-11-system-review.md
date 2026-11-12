# System review (2026-11-09)

A fresh read of the price work, the planner, chips, the xP model and rank play. Every number below comes from the repo's own logs and local outputs, or from a small check described where it is used. Nothing here read 2025/26. Checks on 2022/23 are diagnosis on a spent season and are marked as such. Night labels of 2023/24–2024/25 were not read.

Local inputs: `data/` (built to 2026-10-29), `results/experiments.csv`, Gate N's `predictions.parquet` and the Gate A runs (`results/night-gate-n`, `results/price-a-l` in the `feat/price-sizing` worktree), the chip-study runs in the main checkout. Stage C's and the noise study's per-GW outputs are on the server only and were not read.

## Closed (2026-11-10)

The owner closed this review on 2026-11-10. Outcome:

- **Nothing from it is built.** The three proposed specs (`2026-11-09-night-model-v2-design.md`, `2026-11-09-chip-window-timing-design.md`, `2026-11-09-planner-delayed-effects-design.md`) are closed unbuilt: their sizes sum to 15–35 points a season.
- **The value-of-information study ran** (`docs/research/2026-11-value-of-information.md`): no data source is worth buying. The one large input is who starts over the next three gameweeks among unflagged players, league-wide, and that is the minutes model's ground (the research loop).
- **The system is plausibly already at the target** (correction below: 2,400–2,500 before chips with the current v1). The "gap to 2,400" in the summary, in §5 and in the value-of-information spec's *Why* was measured from the October model and does not describe the live system.
- **Still true and not acted on:** the standard comparison's bias against hits (summary item 3) and the one-snapshot-a-night archive (item 4). Both matter only if price timing or hit settings are reopened.

## Correction (2026-11-10): the baseline of 2,240 is the October model's

The review measured every gap from the holdout's 2,240. That run used the October v1 (`v1-2026-10`). The model live since 2026-11-03 scores well above it on every season where both have been run, so the sizes below are right but the "gap to 2,400" they are compared with is not.

Season points, live planner, no chips, gameweek-1 starts (develop scoring: current rules, no defensive-contribution points):

| Seasons | October v1 | Current v1 | Source |
|---|---|---|---|
| 2023/24, 2024/25 (validate; 6 starts each) | 2,350 and 2,093 | 2,510 and 2,387 | the v1 promotion's stage-2 run (`results/v1r-stage2`, local) |
| 2021/22, 2022/23 (develop, flags all season; 4 starts) | 2,482 and 2,407 | 2,513 and 2,449 | `results/v1r-stage2-develop`, `results/voi/base` |
| 2016/17–2019/20 (develop, no status flags) | 1,998–2,160 | 2,040–2,150 | the same |
| 2025/26 (holdout, native rules) | 2,240 | not run, and it cannot be: the holdout ran once | PLAN §7 |

- On validate the October model averages 2,220, in line with its 2,240 on the holdout. The current model averages 2,450 there, about 230 more.
- So the system as it runs today is plausibly at **2,400–2,500 before chips**, and about 85 more with all chips played. The target of 2,350–2,400 is likely already met on expected points; what is left is variance and execution.
- **Why this is an estimate and not a result:** two validate seasons, 120 points apart; before 2026 the backtest sees the final fixture list, which flatters seasons with many doubles and postponements (2021/22 above all); develop scoring is not the real game's scoring, so these totals do not line up with what managers scored in those years; the current model has no holdout number.
- Scoring level is not the cause of the jump after 2020: the 50 highest-scoring players total 7,830–8,380 re-scored points in every season from 2016/17 to 2024/25. The planner is better where status flags and snapshots exist.

Every "gap", "remaining" and "puts the system at" figure in the summary and in §5 below was written against 2,240 and should be read with this in mind. The per-item sizes do not depend on the baseline.

## Summary

1. **More than half the gap to ~2,400 is chips, not prices.** *(Superseded in part by the correction above: the gap was measured from the October model's 2,240.)* The holdout's 2,240 has no chips. From gameweek-1 starts over nine seasons, playing all eight chips by either the rules H or the planner P is worth **+85 points a season (SE 13)** over playing none (§3). That puts the system at about 2,325–2,345. Chip timing beyond the rules is not established (P − H = +3 ± 24). The remaining 55–75 points to 2,400 have no single source: the items in §5 sum to roughly 15–35.
2. **Prices are worth about 5–15 points a season, and 20 at the very most.** The earlier work is mostly right about that. Three of its conclusions are wrong in the detail, though, and all three make prices look more dead than they are:
   - Gate N's test season contains a 44-night price freeze (the 2022 World Cup). It accounts for 17% of the rows where the rise model was judged, with no rises possible. Before the freeze the rise model was calibrated at T's threshold (0.63 predicted, 0.56 realized).
   - The night model's transfer counter drops about 15% of every gameweek's transfers and is not scaled for the number of managers, which grew 24% between the training and test seasons.
   - Stage C's money term values cash over every remaining gameweek, undiscounted, against points discounted over about 3.3 gameweeks. Its verdict rests on 8 starts of one season with SEs of 14–27 points.
3. **The standard comparison is biased against hits.** It scores one gameweek per decision, so a hit is charged in full and credited for one of the roughly nine gameweeks the player is held. "Hit margin 3 beats 2 by +2.8" comes from that method; the whole-season number from the same runs is −12.6 (SE 7.7).
4. **The 2026/27 archive has one snapshot a night**, not 15-minute snapshots. A live shadow of any night model needs evening snapshots, which the archiver does not take. Every night without them is lost.

No bug was found in the MILP, the selling rule, the night labels or the simulator's accounting.

## 1. Prices and move timing

### 1.1 What is right

- **The size of the prize.** £1m is worth 0.395 xP a gameweek (132 paired solves). My independent count from the live planner's own transfers agrees with the sizing's timing number:

  | Develop 2016/17–2022/23, 3,392 transfer legs (`price-a-l`) | Bought | Sold |
  |---|---|---|
  | Price change in the gameweek before the move (tenths) | +0.355 | −0.295 |
  | Share that moved in that gameweek | 31% | 27% |
  | Change over the 3 gameweeks before | +0.57 | −0.47 |
  | Change over the 3 gameweeks after | +0.40 | −0.47 |

  At about 50 transfers a full season that is £2.5–3.2m a season paid for moves that happened in the week before the transfer, against the sizing's £2.3–2.8m. Kept gradually, £2.75m is 2.75 × 19 × 0.395 × 0.93 ≈ **19 points a season with perfect timing**. That is the ceiling for timing.
- **Gate A's reading.** The oracle's +255 is look-ahead on form. The 47-point figure (closing the whole top-10k gap) is the better bound for everything price-related together.
- **Gate B.** The forecast's accuracy result stands. I found no leak.
- **Night labels.** I rebuilt them from `player_snapshot` for 2021/22 and 2022/23: the same 203,480 scored rows as Gate N, 450 of the season's 454 rises and 1,553 of 1,557 falls captured, no label spanning two changes. The label cannot catch another night's change.
- **The selling rule in the MILP** (`problem.rs:104-137`, `model.rs:590-616`): gameweek 0 trades at FPL's exact integer prices, half the rise is kept on later sales, cohorts carry the buy price. Money is counted once (`model.rs:858-898` subtracts the starting value).

### 1.2 Gate N: the failure is partly an artefact (diagnosis on 2022/23)

Gate N's verdict was that rises are overconfident: at p ≥ 0.5, predicted 0.63 against 0.31 realized. Splitting the gate's own predictions by date:

| Period | Nights | Rises | Rows with p ≥ 0.3: mean p → rate | Rows with p ≥ 0.5: mean p → rate |
|---|---|---|---|---|
| Before the World Cup break (to 11 Nov) | 99 | 213 | 308: 0.49 → **0.40** | 126: 0.63 → **0.56** |
| The break (12 Nov – 25 Dec), prices frozen | 44 | 3 | 179: 0.54 → 0.00 | 93: 0.67 → 0.00 |
| Three weeks after | 21 | 36 | 189: 0.53 → 0.15 | 107: 0.63 → 0.20 |
| The rest (16 Jan on) | 131 | 198 | 370: 0.44 → 0.23 | 99: 0.60 → 0.38 |
| **All (the gate)** | 295 | 450 | 1,046: 0.49 → 0.23 | 425: 0.63 → 0.31 |

- **FPL froze prices during the World Cup** and gave unlimited transfers. No price changed for 43 nights (0 rises and 0 falls in 28,681 player-nights by my labels) while transfer counts exploded. The model had never seen a freeze, and predicted 161 rises there.
- **The counter stayed polluted afterwards.** `net_since_change` kept the break's transfers until each player's next change; FPL evidently did not count them (2 rises a night after the unfreeze against 6–15 predicted).
- **Before the break the gate's condition holds** (0.40 ≥ 0.49 − 0.10), and at T's threshold the model is close to calibrated.
- **The failure does not go away entirely.** From mid-January, with a counter that restarted after the unfreeze, rises still run at 0.25 against 0.44 predicted (283 rows), and 0.42 against 0.59 at p ≥ 0.5. Excluding the break and the three weeks after, the gate's statistic is 0.46 → 0.31, still a fail by 0.05. So there is a real second-half overconfidence of about 0.15–0.2, on top of an artefact that doubled it.
- **Falls** are calibrated in every period outside the freeze (0.60 → 0.59 before it, 0.58 → 0.60 after).

**Online recalibration fixes what is left, at a cost in volume** (post hoc, same predictions). Refitting the Platt map each morning on the season's own last 60 nights:

| | p ≥ 0.3: n, mean p → rate | p ≥ 0.5: n, mean p → rate | Log loss |
|---|---|---|---|
| Gate N as run | 1,046, 0.49 → 0.23 | 425, 0.63 → 0.31 | 0.00840 |
| Recalibrated nightly | 343, 0.46 → 0.36 | 113, 0.62 → 0.56 | 0.00699 |

After mid-January only 59 rows reach p ≥ 0.3 (0.38 → 0.36): the model ranks less sharply there once it is honest. This was chosen after seeing the data, so it is a lead and not a result.

### 1.3 Defects in the night features

1. **The cumulative counter loses 15% of each gameweek's transfers.** `accumulate` (`crates/fplopt-features/src/night.rs:150-162`) carries a player's last net before each deadline into the season total. The last fplcache snapshot is 3.4–3.6 hours before the deadline. Against the final totals in `player_gw_ownership`, the last snapshot holds 86% of the gross transfers; for players with more than 50,000 net transfers in, 15.4% of the net is missing (2021/22 and 2022/23, 46,170 player-gameweeks). The exact total is visible in the store once the deadline has passed and should replace the carried value.
2. **No scaling for the number of managers.** `net_since_change` is a raw count and `net_per_own` divides by ownership in percent, so both grow with the manager count. Estimated from `selected` ÷ `selected_by_percent`, 2022/23 had 24% more managers than 2021/22 at every gameweek (10.9m against 8.8m at gameweek 16), and the count grows 40% within a season. The median `net_since_change` at a rise went from 160,000 to 205,000. A simplified refit with both features divided by the manager count brought predicted rises before the break from 268 to 214 against 213 realized. It did not align the thresholds fully (median scaled counter at a rise: 187k training, 208k before the break, 172k after), so scaling removes part of the drift, not all.
3. **Frozen periods.** Nights when no price in the game moves should be neither trained on nor scored, and the counter should restart when prices unfreeze. It is a one-off, but it sits in the only gate season.
4. **Not fixable from our data:** FPL is believed to ignore transfers made on a Wildcard or Free Hit. The public counters include them.

### 1.4 Stage C: the valuation was mis-scaled, and the test could not see anything

- **Scale.** λ = k · 0.395 · 0.93 · (38 − last gameweek) / 10 points per tenth, **not decayed** (`params.rs:158-166`). At gameweek 10 with k = 1 that is 0.84 points per £0.1m. The points side of the objective is decayed at 0.75, so six gameweeks of a player count as 3.3. A buy forecast to rise £0.2m is therefore credited about 1.7 points for the saving, the size of a whole typical transfer gain. Meanwhile the squad itself has no value after the horizon (PLAN §7). Cash is credited for 23 future gameweeks and player quality for none of them. F (the money term with fixed prices) losing 10.5 points is this imbalance.
- **Power.** One season, 8 starts, SEs of 14–27 points. The arm means (−10 to −36) are all within about 1.5 SE of zero. The logged season totals: base 1,852.5, k = 0 1,842.0, k = 1 1,835.1 (`results/experiments.csv`, family `price planner develop`). The arms made the same number of transfers (37.4–37.9) and hits (16–19.5 points). Stage C shows no benefit; it does not show harm.
- **The fragility is the planner's, not the forecast's.** The placebo pairs of the noise study changed 94% of decisions with a 5% jitter on xP and scored the same. So "it changed transfers in 80% of gameweeks" is what any perturbation does: at the 0.5% MIP gap (about 1 objective point) the planner picks among many near-equal plans. That is also the opening: choosing among those near-equal plans by expected price change costs no points by construction.
- **Not traced:** "expected +158, got +68". The log row for k = 0 has a transfer-gain slope of 1.22, so this is another quantity; its definition is in the server outputs.
- **Team value fell** (−£0.6m to −£1.2m with the bank). This is the one result that argues against the mechanism, and it has no SE in the notes. It needs the per-GW outputs to explain.

### 1.5 Uses never tried

| Use | What it is | Evidence for size |
|---|---|---|
| Sell a planned sale before a fall | Falls are calibrated, 3.4× as common as rises | 27% of sold players fell in the gameweek before the sale, −0.295 tenths a sale on average: about £1m a season at the ceiling, 7 points |
| Buy a planned buy before a rise | Policy T, with a model that passes | 31% of buys rose the week before: about £1.8m a season at the ceiling, 12 points |
| Price as a tie-breaker | Among plans within ε of the best xP objective, take the one with the most expected money | Bought players rise +0.27 tenths in the next gameweek already; unknown how much choice the ε-set gives. Measured by an oracle run first |
| Bank management | Hold cash for a planned later buy | The planner already does this inside the horizon; nothing found |

### 1.6 The 2026/27 archive cannot support a night shadow today

`player_snapshot` for 2026/27 has 36 snapshots from our archiver in 23 days: 29 between 00:00 and 03:00 UK and one before each deadline. The tick job runs every 15 minutes but takes a snapshot once per pre-deadline window (`crates/fplopt-archive/src/schedule.rs`). fplcache continues at about four a day at irregular hours. A night model's features need the evening state (transfers so far, a few hours before the change) and its label needs the morning. **Adding an evening and a late-night `bootstrap-static` snapshot is one request each a day, and is the only item here with a deadline: unarchived nights cannot be recovered.**

### 1.7 Verdict on prices

- **Not dead:** timing planned moves. The night model ranks well, falls are calibrated, the rise failure was half artefact, and the remaining half responds to nightly recalibration. Ceiling 19 points a season; a realistic share is a third to a half less an information cost: **+3 to +8**.
- **Not dead, unmeasured:** price as a tie-breaker. **0 to +5.**
- **Dead as built:** money in the objective at an undiscounted rate (Stage C's k > 0). F lost 10.5 points (80% CI −18.5 to −2.4) on 56 units.
- **Dead:** moving every transfer at the lockdown regardless of price (policy E). It buys the same £2.3m as a conditional policy but takes the news risk on all 50 transfers instead of the 15 that need it.

The whole area is worth roughly **5–15 points a season**. The team-value gap to the top 10k (£3.9m at gameweek 9) is mostly a by-product of owning in-form players early, as Gate A showed, and cannot be bought back with a price model.

## 2. The planner

No bug found: hits, free-transfer transitions, decay and the selling rule are consistent (`model.rs:647-701`, `789-822`), and every local solve ended `Optimal`.

- **The standard comparison is biased against anything that pays later.** It sums `net_points` over a one-gameweek window (`evaluate.rs:419-421`). A hit costs 4 in that window and is credited one gameweek of a player held a mean of 9 (median 6). Margin 3 cut hit points from 15.5 to 10.2 a unit and gained only 2.8 at k = 1, so the forgone hits were earning back about two thirds of their cost in their first gameweek alone. On the same runs the whole-season difference is −12.6 (SE 7.7). The evidence leans toward margin 2 being too *high*, not too low.
- **Decay 0.75 was the best of 8 noisy cells.** Decay 0.85 wins 3 of the 4 matched pairs in PLAN §7's table. The chosen cell fell from +2.22 a gameweek on develop to +0.74 on validate. Effective horizon is 3.3 gameweeks against a 9-gameweek mean hold.
- **Free-transfer values were never tuned with v1** (PLAN §7 says "tuned"; no log row varies them and there is no spec key). The planner transfers in 89% of gameweeks and holds two or more free transfers in 12%.
- **No case for a gain shrink.** Realized gain per free transfer by predicted gain (develop, old v1, 4-gameweek points):

  | Predicted | ≤ 2 | 2–4 | 4–6 | 6–8 | 8–12 | > 12 |
  |---|---|---|---|---|---|---|
  | n | 56 | 156 | 179 | 232 | 309 | 130 |
  | Realized | +3.7 | +4.3 | +3.4 | +4.0 | +9.0 | +13.5 |

  Small predicted gains do not lose. The slope figures quoted in PLAN (1.08, 0.93) are regression slopes; the ratio of means on develop with the current v1 is 1.02.
- **The MIP gap is the size of a decision.** 0.5% of a ~200-point objective is 1 point; 41% of develop solves end above 0.3%. PLAN §8 already records plans #1–#3 inside the gap.
- **Live is not quite the tested system.** The bot shows the best of three plans by objective (`crates/fplopt-bot/src/planner.rs:424-435`); the backtest executes the first solved. The live solve has a 60-second cap on a box shared with research.
- **Bench weights at gameweek 1 and after a Wildcard** are the fixed 0.03 / 0.21 / 0.06 / 0.002, against slots needed 27–56% of the time for template squads (PLAN §7).

## 3. Chips

Numbers from the local chip-study runs use the October v1; current-v1 numbers are quoted from `docs/research/2026-11-chip-policy-2.md`, whose outputs are on the server.

### 3.1 What chips are worth

Season points over no chips, gameweek-1 starts (mean of 4 starts a season), both chip sets:

| Arm − no chips | Develop, 7 seasons | Validate, 2 seasons | All 9 (SE over seasons) |
|---|---|---|---|
| Rules H | +90 ± 13 | +69 | **+85 ± 13** |
| Planner P | +67 ± 16 | +161 | **+88 ± 19** |
| P − H | −23 | +92 | **+3 ± 24** |

- With the current v1 on develop: H +93, P +106, V6 +115.
- The per-gameweek figures in PLAN (+4.18 for P, +2.49 for H) are means over 8-gameweek cells that include the gameweek-20 starts, so multiplying them by 38 overstates a season (159 and 95).
- **Chip timing beyond the rules is dead on the evidence we can get:** P − H pooled over develop (current v1) and validate is about +23 ± 19 a season. Validate is spent and develop cannot separate them.
- **Backtest chip gains are optimistic** by an unmeasured amount: before 2026 the simulator sees the final fixture list, so doubles and blanks are known all season (PLAN §3).

### 3.2 A defect in how the planner times a chip inside its window

- **Code.** A Triple Captain or Bench Boost played in horizon gameweek h is credited `decay^h` × its gain (`crates/fplopt-optimize/src/model.rs:789-795`). The value of holding the chip is added undecayed, and only while the window ends after the horizon (`chips.rs:122-126`).
- **Effect.** Once the window's end is inside the horizon the holding value is 0 and a later gameweek's gain is discounted. With a captain worth 7.0 now and 12.0 in a double three gameweeks on, the planner sees 0.75³ × 12 = 5.1 and plays now.
- **In the runs.** P plays one chip a gameweek from five gameweeks before a window closes. Validate, set 1: all 8 Triple Captains and all 8 Free Hits at gameweeks 14–15. Develop, set 2: 15 of 49 Triple Captains at exactly gameweek 33.
- **Cost** (develop, October v1, the no-chip arm's path, runs that reach that point with the chip unused): in set 2 the gameweek P plays realizes 6.6 (TC) and 6.0 (BB) against 14.4 and 11.3 for the best forecast gameweek left, +7.8 ± 3.8 and +5.4 ± 2.7. Set 1 shows no cost. Weighted by how often it happens, about **+5 points a season (0 to +10)**.
- **Live.** Set 1 closes at gameweek 19, so from the gameweek-14 deadline the alert's headline will say to play each remaining set-1 chip now, labelled "+g xP vs keeping it" (`crates/fplopt-bot/src/messages.rs:479`). From then on that means against never playing it.

### 3.3 Smaller findings

- **The alert's evidence line** (`messages.rs:556-560`) quotes +4.2 points a gameweek, measured with the October v1 and the October value table. The live planner is the current v1 with a re-estimated table, which was never run on validate.
- **V6's Free Hit shrink is one-sided** (read from code, not from runs): the in-horizon gain is × 0.4 (`search.rs:155-169`) but the holding value is × 0.8, so V6 holds the Free Hit until the window forces it. This affects only the shadow log.
- **The alert pairs the planner's chip call with the no-chip transfer plan.** The tested planner made its chip plan's own transfers.
- **Checked and fine:** both chip sets and their windows, the consecutive Free Hit ban, set-1 expiry, H's last-chance rule, the develop-only guard on the value tables, the V6 shadow log from gameweek 20.

## 4. xP, minutes and captaincy

Not audited in depth: the nightly loop owns the xP model, and nothing here duplicates it.

- **Fragility is an xP finding.** The eight jittered arms of the placebo runs scored about 25 points a season below the unjittered planner on the same starts (SD across arms 9). A 5% multiplicative error on xP costs that much, so xP accuracy is worth more than any planner setting.
- **Captaincy needs no work.** The captain is the highest xP (`model.rs:795`) and the vice is picked after the solve. The exact value adds (1 − P(captain plays)) × the vice's xP. Captains failed to play in 3.2% of develop gameweeks; re-ranking with that term on validate changed 9 of 76 picks for +0.05 ± 0.33 points a gameweek. Simulating from v1's own calibration bands reproduces the measured captain regret (6.80 against 6.79 ± 0.63 on a 15-player pool), so the regret is noise, not model error. Risk has no place in an expected-points captain.
- **Rival ownership** belongs to rank play (§6), not to expected points.

## 5. Ranked opportunities

Points a season against the no-chip 2,240, with the range I would bet on. "Test" is the cheapest valid one.

| # | Opportunity | Size | Where the number comes from | Test |
|---|---|---|---|---|
| 1 | Play all eight chips, by the rules or the planner | **+85 ± 13** | §3.1, nine seasons | None needed. It is a matter of the user playing them |
| 2 | Hits and decay re-tested with a statistic that counts later gameweeks | +5 to +15, low confidence | §2: whole-season −12.6 for the stricter margin; 0.85 wins 3 of 4 pairs | Develop, 7 seasons, 4-gameweek windows and whole season; one check on 2023/24–2024/25 |
| 3 | Chip timing inside the window (§3.2) | +5 (0 to +10) | Set-2 dump cost on develop | Develop rerun; live shadow over set 2 |
| 4 | Timing planned moves with a repaired night model | +3 to +8 | §1.5: ceiling 19, a third to a half captured, less news risk | Prospective 2026/27 nights; then one check on 2023/24–2024/25 |
| 5 | Tighter MIP gap, or best of three plans as the bot does | 0 to +6 | §2 | Standard comparison (fair here: no delayed effect) |
| 6 | Bench weights at gameweek 1 and after a Wildcard | +3 to +8 on gameweek-1 starts | §2 | Whole-season, gameweek-1 starts |
| 7 | Price as a tie-breaker among near-equal plans | 0 to +5 | §1.4, §1.5 | Oracle prices first (2021/22–2022/23), as Gate A did |
| 8 | Double-aware Triple Captain and Bench Boost rules | +2 to +4 over H | Set 2: best forecast gameweek 17.5 against H's 11.4 | Offline on the no-chip path, no solves |
| 9 | Free-transfer values, squad value after the horizon | −3 to +10 | Never tested with v1 | After #2 |
| — | Gain shrink, money in the bank, captain risk, injury dates, policy E | ≈ 0 or negative | §1.7, §2, §4 | Not worth a run |

Items 2–9 sum to roughly +15 to +35 if each lands at the middle of its range, which is optimistic for a list picked after looking. With chips that is about 2,340–2,380 from the October model's 2,240 (see the correction at the top: the current model starts about 230 higher). The template start alone scored 2,317 without chips on the holdout; one unit is not evidence, but a real manager starts from a sensible squad, not a random one.

## 6. Rank and mini-league play

Neither queued idea earns expected points. They move the probability of winning a league.

- **Rival ownership as a tie-break within about 0.5 xP** costs nothing (such plans are inside the MIP gap) and is worth perhaps 1–2 percentage points of win probability in a close race. The bot already reads public entries; rivals' picks are public after each deadline, not before. Minimal build: `/league <id>`, "owned by k of n rivals" on each plan, tie-break among the top plans.
- **Endgame mode** matters only when the gap is within about one standard deviation of the points difference. Trailing by 30 with an SD of 40 gives a 23% chance; 15% more variance gives 26%. It needs joint distributions, and v1 has marginals without team correlation. Validate the simulated squad totals on develop (variance, tails, teammates) before any policy is built on them.

## 7. Proposals

Four specs, in `docs/superpowers/specs/`, in the order agreed with the owner on 2026-11-09 (the gains of the first three were judged too small to lead with):

1. `2026-11-09-value-of-information-design.md`: what each xP input is worth in season points if known perfectly, to decide which new source of information is worth building. Added after the owner's review; it is the only item aimed at the 55–75 points that chips leave open.
2. `2026-11-09-chip-window-timing-design.md`: the fix of §3.2 and the alert wording, before set 1's last five gameweeks.
3. `2026-11-09-night-model-v2-design.md`: of this, only part 0 (evening snapshots) goes ahead now, to keep the option open. The repairs of §1.3, nightly recalibration, the prospective gate and policy T wait.
4. `2026-11-09-planner-delayed-effects-design.md`: hit margin, hit count and decay judged on windows that count what a hit buys. Runs when server time is free.

