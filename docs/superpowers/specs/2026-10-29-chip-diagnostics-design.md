# Chip research, sub-project 1: what drives chip value (design)

**Status:** approved in conversation 2026-10-29; spec for review.

## Context

The chip study (`docs/superpowers/plans/2026-10-11-chip-policy.md`) ended with **chips stay manual**:
- On validate, P (the chip planner with estimated terminal values) beat S0 (no chips) by +4.18/GW (p 0.025).
- It did not beat the rule baseline H: +1.70/GW, p 0.083. On develop the edge was +0.14/GW.
- `/chips` is information only.

The user wants a second round in three sub-projects, each with its own spec, plan and implementation:

1. **Diagnostics** (this spec): where chip value comes from, under today's model.
2. **A better chip policy**, tested against H. Its candidate ideas come from 1: re-estimated terminal values, timing by doubles/blanks and fixtures, Free Hit on blanks.
3. **A chip track in the nightly research loop**: a frozen chip judge on develop.

**Promotion bar for 2 and 3 (decided):**
- Ideas are judged on develop, with the mean deflated for the variants tried (PLAN §5 *Multiple testing*).
- The winner is frozen **before GW20 of 2026/27** and shadows the live system for set 2 (GW20–38), on the user's squad plus template and random squads.
- Chips stay manual until it holds up there.
- Validate (2023/24–2024/25) is not used again.

## Goal

A report that tells sub-project 2 which chip ideas have room to win, and by about how much. It also gives the user labelled-untested guidance for their set-1 chips, which expire at the GW19 deadline. Everything is measured under today's v1: the #47 promotions, the ESPN bans (#50), congestion and cup minutes (#51).

## Runs

**Where:** the dev PC, `--jobs 14`.

**Common settings:**
- seasons: develop 2016/17–2022/23;
- starts: `template@1,random:3@1,random:3@20` (as on Oct 11);
- experiment-log family: `chip diagnostics 2026-10`.

| Arm | Spec | Out |
|---|---|---|
| S0 | `chipgains:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2` | `results/chips-diag-s0` |
| H | `chiprules:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2,t_tc=8.3,t_bb=10.9,k_fh=4,t_wc=12.87` | `results/chips-diag-h` |
| P | `optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2,chips=1,chip_shrink=1.0` | `results/chips-diag-p` |

- **P keeps today's `config/chips/terminal-values.json`.** Re-estimating the table is one of sub-project 2's candidates. It is not changed here, and the bot's `/chips` keeps reading it.
- **The paired comparisons are logged as well:** P vs H, P vs S0 and H vs S0, with `backtest compare`, full run, develop. Together they show whether today's model on its own moves the Oct 11 develop gaps (P vs H +0.14/GW, H vs S0 +2.39/GW).

## Questions

All inputs are the runs' `gws.parquet`, joined to the final schedule (`data/schedule.parquet`) for each club's fixtures per GW.

**Notation:**
- *single / double / blank GW*: a GW with a club playing 1 / 2 / 0 league fixtures in it. Per chip, the relevant club is the captain's (TC), the bench's (BB) or the squad's (FH).
- *cell*: the 8-GW cells of the chip study's window t-test.

1. **Where the gap comes from.**
   - P − H and H − S0 realized (and xG-scored) net points, split by the chip played in that GW (none / TC / BB / FH / WC) and by cell.
   - For a Wildcard, the GW it is played in plus the next 5 GWs are attributed to it.
   - When each arm plays each chip: a histogram of `gw_index` per window (set 1: 1–19, set 2: 20–38).
2. **What a chip is worth, and when.**
   - S0's ex-ante gains (`gain_tc`, `gain_bb`, `gain_fh`, `gain_wc`) by `gw_index`.
   - The same split by single / double / blank, start type (template / random), bank and free transfers.
   - Distributions shown as median and 75th / 90th percentiles.
3. **Forecast vs outcome.**
   - TC: the predicted gain (captain xP) against the realized gain (the captain's points), in the GWs H and P played it.
   - BB: the bench's xP net of bench weights against the bench's realized points.
   - FH and WC: the predicted gain against the realized paired difference vs S0 (FH: that GW; WC: that GW plus the next 5).
   - Reported per chip: the OLS slope of realized on predicted, its SE, and the mean bias. A slope below 1 means chip decisions are overconfident (the optimizer's curse).
4. **The value of holding a chip.**
   - Per chip and GWs left `L`, compare the table P plans with against two things: its ex-ante estimate from this S0 run (`estimate_terminal_values`), and what a manager realizes.
   - **TC and BB:** the realized gain of playing the chip at the ex-ante best GW among the `L` left. The realized gain is computable for every S0 row: the captain's points, and the bench's points net of autosubs.
   - **FH and WC:** S0 never plays them, so their realized gain is unknown for most GWs. They get the ex-ante table times Q3's slope instead.
   - The output is the ratio realized / table by chip and `L`. This is the evidence for the right shrink, possibly per chip.
5. **Congestion.**
   - For the TC captains and BB benches that H and P picked: the realized/xP ratio in fixtures with `europe_week` = 1 or `rest_days_all` ≤ 3, against the rest, with counts.
   - Whether either arm's picks are under- or over-represented in such fixtures, against S0's chip-gain-weighted baseline.
6. **The user's set-1 chips** (labelled *untested research*).
   - Doubles and blanks of 2026/27 GW10–19 in the as-of schedule now.
   - The user's `/chips` output (FPL's live prices, `made` transfers).
   - What findings 1–5 imply for when TC / BB / FH are worth most before GW19.
   - No model is scored on 2026/27 data.

## Code

- **`src/fplopt/backtest/chip_diagnostics.py`**: pure functions over frames, with no file reads (callers pass frames in):
  - `gw_kind(schedule, gws, club_of)`: single / double / blank per row and chip;
  - `gap_by_chip(a, b)` and `gap_by_cell(a, b)`: paired differences on (season, start_id, gw_index), the WC 6-GW attribution included;
  - `gain_distribution(gws, by)`;
  - `forecast_slope(predicted, realized)`: slope, SE, bias, n;
  - `holding_value(gws, table)`: realized and ex-ante best-of-L per chip and `L`, and the ratio to `table`;
  - `congestion_split(rows, features)`.

  These live in the package because sub-projects 2 and 3 reuse `forecast_slope` and `holding_value` as judge metrics.
- **`tests/test_chip_diagnostics.py`**: small synthetic frames with planted answers. Examples:
  - a planted double gives `gw_kind` = double;
  - a planted gap appears in the right chip bucket and cell;
  - a planted realized = 0.5 · predicted gives slope 0.5;
  - a planted best-of-L is recovered.
- **`dev/chip_diagnostics.py`**: a flat script, like `dev/calibrate_v1.py`. It loads the three run dirs, the schedule and (for Q5) the features of the picked players at their deadlines, through `DataStore(...).as_of(deadline)`. It writes the tables as CSV to `results/chips-diag/` (gitignored) and prints a summary.
- **Run identity:** `backtest run` already logs commit and spec to `results/experiments.csv`. The script records the run dirs it read in its output folder.

## Report

**`docs/research/2026-10-chip-diagnostics.md`** (committed):
- one section per question: the key table and two or three sentences on what it means;
- then **ranked ideas for sub-project 2**, each with the develop gain it could plausibly be worth and the evidence for that;
- then the user's set-1 section.

It is also published as a private page (Artifact) for reading on the phone.

## Data rules

- Develop seasons only (2016/17–2022/23) for every number in 1–5.
- Validate and the 2025/26 holdout are not read.
- 2026/27 appears only in section 6, as the schedule known now and the bot's `/chips` output, with no scoring on live data.
- `raw/` and the bot's server are untouched. Everything runs on the dev PC with its local `data/`, rebuilt from main before the runs (`fplopt build all`).

## Out of scope

- Any policy change: re-estimated terminal values, new rules, tuning. That is sub-project 2.
- Changes to the bot.
- The research loop. That is sub-project 3.

## Done when

- The three runs and three comparisons are logged.
- `chip_diagnostics.py` is tested (`uv run pytest`, ruff clean).
- The report is committed and published, and its ranked idea list is agreed with the user as sub-project 2's starting point.
