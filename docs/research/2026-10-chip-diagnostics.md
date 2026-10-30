# Chip diagnostics, 2026-10 (chip research, sub-project 1)

- **Spec:** `docs/superpowers/specs/2026-10-29-chip-diagnostics-design.md`
- **Plan:** `docs/superpowers/plans/2026-10-29-chip-diagnostics.md`

Develop seasons only (2016/17–2022/23), except §8. Validate and the 2025/26 holdout were not read.

**Runs**
- Three develop backtests under today's v1: the #47 promotions, ESPN bans and congestion.
- Starts: `template@1,random:3@1,random:3@20`.
- 1,448 GWs each, every solve Optimal. Logged under family `chip-diagnostics-2026-10`.
- Outputs: `results/chips-diag-{s0,h,p}`.

| Arm | Spec |
|---|---|
| S0 | `chipgains:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2`: no chips, each chip's ex-ante gain recorded at every deadline |
| H | `chiprules:v1:…,t_tc=8.3,t_bb=10.9,k_fh=4,t_wc=12.87`: the rule baseline at its Oct 11 thresholds |
| P | `optimizer:v1:…,chips=1,chip_shrink=1.0`: the chip planner with today's `config/chips/terminal-values.json` |

**Tables:** `results/chips-diag/` (`dev/chip_diagnostics.py`, functions in `fplopt.backtest.chip_diagnostics`). Each section names its file.

**Caveat for every paired number below:** a chip's "gain" is the arm's points minus S0's on the same GWs. After the first chip the arms' squads drift apart, so this includes transfer differences, not just the chip.

## 1. The gaps under today's model (`comparisons.json`)

Full-run window t-test on 8-GW cells, develop (realized; xG-scored in brackets):

| Comparison | Today's model | Oct 11 (old v1) | One-sided p |
|---|---|---|---|
| P vs H | **+0.33/GW** (+0.36) | +0.14 | 0.26 |
| P vs S0 | +2.77/GW (+2.49) | ≈ +2.5 | < 0.001 |
| H vs S0 | +2.44/GW (+2.19) | +2.39 | < 0.001 |

- **Per season, P vs H** (points over the season, mean over starts): +49, +9, +13, +39, −18, +20, −40. Positive in 5 of 7 seasons; 2022/23 is the big miss.
- **Reading.** Today's model widens P's edge over H a little. It is still not distinguishable from noise on develop. Both arms are worth about +2.5–2.8/GW over playing no chips, roughly 95–105 points a season.

## 2. Where the gap comes from (`q1-gap-chip-*.csv`, `q1-timing.csv`)

**Realized gain per play against S0, by chip and window** (mean points; 28 plays in set 1, 49 in set 2, except P's set-1 Wildcard: 25; `q1-gain-per-play.csv`):

| Chip | H set 1 | H set 2 | P set 1 | P set 2 |
|---|---|---|---|---|
| Triple Captain | 13.5 | 15.6 | **4.7** | 12.9 |
| Bench Boost | 10.1 | 11.3 | 12.0 | 11.6 |
| Free Hit | 8.4 | 6.4 | 9.5 | **16.9** |
| Wildcard (that GW) | 8.4 | 9.2 | 6.9 | 8.2 |

**When they're played** (median `gw_index`, set 1 / set 2):

| Chip | H | P |
|---|---|---|
| TC | 19 / 23 | 14 (all GW12–16) / 33 |
| BB | 14 / 26 | 1 / 28 |
| FH | 17 / 28 | 14.5 / 27 |
| WC | 18 / 26 | 13 / 30 |

Both arms use the "last chance" rule, so every chip is played in every window, except P's set-1 Wildcard (25 of 28 runs). It's timing, not use, that differs.

**What this shows:**
- **P's Free Hit is much better than H's** in set 2: 16.9 against 6.4 per play. H's rule ("≥ 4 squad players blank") fires on blanks, but P finds the doubles.
- **H's Triple Captain is much better than P's** in set 1: 13.5 against 4.7. P plays every set-1 TC in GW12–16, as soon as its plan finds a good-enough GW. H mostly waits until GW19, because its threshold of captain xP 8.3 rarely fires. n = 28 a side, so the gap is about ±3 per play. That is large, but the table is noisy.
- **H vs S0** splits evenly across the chips: Wildcard 0.82/GW, TC 0.79, BB 0.58, FH 0.38 (`q1-gap-chip-H-S0.csv`).
- **P vs H** nets out from P's better FH and BB against H's better TC (`q1-gap-chip-P-H.csv`).

## 3. What a chip is worth, and when (`q2-gain-*.csv`)

**S0's ex-ante gains** (median):

| Chip | Captain on a double | Single | GW with doubles | Normal GW | GW with blanks |
|---|---|---|---|---|---|
| TC | **10.6** | 6.8 | 11.0 | 6.9 | 6.5 |
| BB | 6.7 | 4.5 | 7.3 | 5.0 | 0.6 |
| FH | 14.4 | 5.5 | 14.3 | 5.2 | 8.6 |
| WC | 13.8 | 5.4 | 13.1 | 4.5 | 8.8 |

- **Doubles dominate.** Every chip is worth roughly 1.5–2.5× as much in a GW with doubles. In develop seasons, doubles fall almost entirely in set 2.
- **By time of season** (median by `gw_index` band): TC is flat at 6.8–7.1 through GW19, then rises to 8.1 in GW36–38 (`q2-gain-gw_band.csv`). BB, FH and WC have no peak in set 1. FH and WC jump from about 4–6 in set 1 to about 7–9.5 in set 2.
- **Squad state:**
  - Random squads gain more from WC (median 6.9 vs 4.7) and FH than the template squad.
  - FH is worth less the more free transfers are banked: median 6.7 with 1 FT, 2.7 with 3.
  - Bank had no clear effect.

## 4. Forecast vs outcome (`q3-forecast-slopes.csv`)

| Forecast | Slope of realized on predicted | SE | Bias (realized − predicted) | n |
|---|---|---|---|---|
| TC, all S0 GWs | 0.70 | 0.07 | +0.3 | 1,448 |
| TC, top decile | 2.0 | 0.35 | −1.2 | 151 |
| TC, captain on a single | **0.39** | 0.14 | +0.6 | 1,172 |
| TC, captain on a double | 1.04 | 0.17 | −0.7 | 276 |
| BB, all S0 GWs | **1.03** | 0.04 | +0.4 | 1,448 |
| BB, top decile | 0.80 | 0.43 | +0.8 | 145 |
| FH, H's plays (that GW) | **0.43** | 0.24 | −3.5 | 77 |
| FH, P's plays | 0.42 | 0.25 | −0.3 | 77 |
| WC, H's plays (6 GWs) | **0.10** | 0.28 | +15.8 | 77 |
| WC, P's plays (6 GWs) | 0.22 | 0.27 | +16.4 | 74 |
| WC, H's plays with a full 6-GW span | −0.03 | 0.30 | +20.1 | 64 |
| WC, P's plays with a full 6-GW span | 0.15 | 0.27 | +19.1 | 67 |

- **Bench Boost forecasts are calibrated.**
- **Triple Captain** is calibrated on doubles but not on singles:
  - **Doubles:** slope 1.04, with the forecast about 0.7 points too high on average.
  - **Singles:** slope 0.39. A single-fixture captain forecast at 9 has realized little more than one forecast at 7.
  - Realized TC gain was 10.0 on a double against 7.5 on a single.
  - The top-decile slope of 2.0 is not meaningful. That decile is 140 doubles and 11 singles, so the slope is driven by range restriction.
- **Free Hit gains are over-stated** by about half (slope 0.4).
- **The size of the Wildcard's forecast gain carries no information** about its realized gain (slope 0.1–0.2 ± 0.3). A Wildcard is still worth about +16 points over 6 GWs in both arms, whatever was forecast.
  - The WC row's realized side sums 6 GWs that can contain other chips, so it is the noisiest row here.

## 5. The value of holding a chip (`q4-holding-value.csv`)

For each chip and number of GWs left `L`: the expected best use over the rest of the window, against the terminal value P plans with (`table`). The table is one set of numbers for both windows.

**Triple Captain** (`table` 8.0 / 11.1 / 12.1 / 12.8 at L = 1 / 5 / 10 / 19):

| Window | L = 1 | L = 5 | L = 10 | L = 19 |
|---|---|---|---|---|
| set 1, realized | 6.3 | 6.8 | 6.3 | 6.8 |
| set 1, realized / table | 0.78 | 0.62 | **0.52** | **0.53** |
| set 2, realized | 7.2 | 12.5 | 13.9 | 16.8 |
| set 2, realized / table | 0.90 | 1.13 | **1.15** | **1.31** |

**Bench Boost** (`table` 5.2 / 8.2 / 8.9 / 9.4):

| Window | L = 1 | L = 5 | L = 10 | L = 19 |
|---|---|---|---|---|
| set 1, realized / table | 1.59 | 0.49 | 0.78 | 0.51 |
| set 2, realized / table | 0.82 | 1.24 | 1.35 | 1.31 |

**FH and WC** (ex-ante only: S0 never plays them, so the realized side is unknown). With 4 or more GWs left, the set-1 ex-ante values are 0.64–0.80 of the table for FH and 0.50–0.88 for WC; set 2's are 1.03–1.24 and 1.01–1.27. With §4's slopes, the realized values would be lower still.

**The finding:**
- **Set 1:** in a set-1 window, waiting doesn't pay. The realized best use of a TC is about 6.5 whether 1 or 19 GWs are left, because set 1 has almost no doubles. The table credits holding at up to twice that.
- *Note:* windows are keyed on `gw_index`, and 2022/23 reaches only index 37, so set 2's L = 1 has 42 runs, not 49.
- **Set 2:** it under-credits holding, because the doubles come late.
- **One table for both windows is wrong in both directions.** That is consistent with P's poor set-1 TC (§2): it holds and plans around inflated values, so the plays it does make are not well timed.

## 6. Congestion (`q5-*`)

- **Captains in congested fixtures realize as well as forecast.** That means a European week, or rest of 3 days or less (536 of 1,448 S0 GWs). Their realized/xP ratio is 1.03, against 1.05 for the others.
- **Where the TC picks land:** 23% of H's picks and 39% of P's fall in congested fixtures, against a 37% gain-weighted baseline.
- **No chip idea here.** The minutes model already prices congestion (#51), and what's left after that is negligible.

## 7. Ranked ideas for sub-project 2

Ranked by plausible develop gain × strength of evidence. Gains are rough per-GW estimates against H.

1. **Separate terminal values per window, set 1 and set 2.** Re-estimate the table from this S0 run, one per window (`holding_value` already splits them), with a shrink chosen on develop. *Evidence:* §5, consistent across TC, BB and the ex-ante FH/WC, and §2's set-1 TC. *Plausible:* +0.2–0.5/GW. This is the core fix for P's timing.
2. **A hybrid policy: P's Free Hit, Bench Boost and Wildcard, with an H-style rule for Triple Captain.** Or equivalently, P with the TC decision taken out of the search. *Evidence:* §2 (P's FH 16.9 vs H 6.4 in set 2; H's TC 13.5 vs P 4.7 in set 1). *Plausible:* about +0.3/GW from TC and about +0.3/GW from FH. Sample sizes are small (28 and 49 per cell). Idea 1 may close the TC gap on its own, so test 1 first, then 2 on top of it.
3. **Make chip decisions double-aware.** Rules or planner features that hold BB, TC and FH for doubles in set 2. *Evidence:* §3 (1.5–2.5× gains in doubles), §4 (realized TC gain 10.0 on a double against 7.5 on a single, and single-fixture TC forecasts barely separate good weeks from average ones), §5 (set-2 holding is under-credited). Partly covered by idea 1, since that table already credits set-2 doubles.
4. **Shrink the Free Hit forecast** by about 0.4–0.5 at decision time (§4), and treat the Wildcard gain as uninformative. That means a fixed-value WC with timing by a simple rule, not by the forecast gain. This fits the transfer-gain shrink test planned alongside.
5. **Dropped: congestion-based chip rules** (§6).

## 8. Your set-1 chips (untested research)

Set 1 expires at the GW19 deadline. These are pointers from develop history plus this season's schedule, not a tested policy.

You've used your set-1 Wildcard (GW4) and Triple Captain (GW7). **Bench Boost and Free Hit are left.**

- **No doubles or blanks are scheduled in GW9–19** as of today (`q6-upcoming.csv`). Cup rearrangements can still create some.
- **Waiting doesn't pay in set 1** without doubles (§5). For Bench Boost, the best realized use over the rest of set 1 is no higher with 19 GWs left than with 1. Holding is only worth it if a double or blank is announced.
- **Bench Boost:** set-1 plays averaged 10–12 points in both arms. Pick the GW where all four of your bench players have a fixture and are nailed. The bot's GW9 check gives BB −0.3 vs holding (`q6-user-chips.txt`: the bot's chip search for entry 4570069 on 2026-10-29, run on the server with the live plan context), which is a tie. A GW where your bench is stronger is the time.
- **Free Hit:** in set 1 without blanks or doubles it's worth about 6–9 points (§3, §5). The bot's GW9 check gives it −13.9 vs holding. Keep it for a GW where injuries or bans leave three or more of your starters out, or for any blank that's announced.
- **For set 2,** looking ahead:
  - The doubles come late, and holding BB, TC and FH for them has paid (§3, §5).
  - The Wildcard's forecast gain doesn't predict its outcome (§4). Use it when the squad needs a rebuild, not when a forecast number peaks.
