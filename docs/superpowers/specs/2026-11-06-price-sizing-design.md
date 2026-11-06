# Price changes: sizing the upside before building a model

**Date:** 2026-11-06. **Status:** approved by the user in chat. Branch `feat/price-sizing`, from `main` (8f8c8f9). Written in Rust where it touches the backtester; the analysis scripts live in `dev/`.

## Why

- PLAN §1 lists price-change prediction as a non-goal. The user asked (2026-11-05) to "fully develop and validate a price change model", because team value is one of the gaps between our system and top-10k managers.
- What the code does today:
  - backtests replay real deadline prices (`player_gw.value`, 2016/17 →) and FPL's selling rule (purchase + floor(half the rise));
  - the planner holds prices fixed over its horizon (PLAN §7);
  - the live bot uses FPL's live prices but ignores changes between now and the deadline.
- A price model can earn points in two separate ways:
  1. **Planning:** the planner sees forecast price paths, prefers likely risers, sells likely fallers and counts the money gained.
  2. **Timing:** within a GW, make a transfer before a rise or after a drop.
- The user's rule is "measure first". This study sizes the **upper bound** of each, with perfect foresight, and decides with thresholds fixed here, before any number is seen, whether to build either.

## 1. Planning: the quick estimate

**Upper bound = value gap over a season × points per £1m.**

### 1a. Value gap (2026/27, GW1–9)

- **Our system:** `backtest run` on season 2026, GW1–9, the live settings (`holdout.SYSTEM`'s spec, v1 xP), template start at GW1. `gws.parquet` already records `bank` and `squad_value` per GW. Team value = squad value + bank.
  - **VERIFY:** `backtest run` accepts season 2026 (the current season, not the holdout) and the template start at GW1. If it refuses, report it and use the user's own team as the comparison instead of building a workaround.
- **Managers:** the overall league's standings (`leagues-classic/314/standings/`) give entry ids by rank. Sample 500 entries uniformly from ranks 1–10,000 and 500 from 10,001–100,000 (seed 0), and fetch each one's `entry/{id}/history/`: `value` and `bank` per GW.
  - The standings page holding each sampled rank (50 per page; up to ~1,000 pages) plus 1,000 histories: about 2,000 requests (~35 min), one at a time, 1 s apart, with the project's User-Agent. The raw responses are saved as `.json.gz` under the scratch output folder, not in `raw/` (this is a one-off study, not an archived source).
  - Ranks are as of the fetch, so the sample is "managers ranked there now". That's survivorship: it overstates what an average top manager had. The report says so.
- **Also shown:** the user's entry (4570069) and the GW1 → GW9 value path of each group (median and quartiles).
- **Gap** = the group's median team value at GW9 minus ours, per group. Scaled to a season by assuming the gap grows linearly over 38 GWs (× 38/9); the report states the assumption and also gives the unscaled GW9 gap.

### 1b. Points per £1m (develop seasons 2016/17–2022/23)

- **Backtester change (Rust, `crates/fplopt-backtest`):** a start modifier that adds money to the bank: `template+bank10@1` means the template start at GW1 with £1.0m (10 tenths) more in the bank. It works on any start kind (`template`, `random:SEED`, `squad:FILE`) and appears in the start label in `gws.parquet` and the log. Unit test: the state's bank is the base start's bank + 10, everything else equal.
- **Runs:** `backtest compare --a <live spec> with bank10 --b <live spec> without`, seasons 2016–2022, starts template and random:3, at GW1 and at GW20. Report points per £1m per start GW with the 80% CI.
  - If `compare` can't pair different starts, run two `backtest run`s and pair the season totals in the analysis script.
- **Conversion rate:** the GW1 figure scales the full-season gap. The GW20 figure is reported for context.

## 2. Timing: perfect foresight

- **Transfers:** every transfer our system made in the develop backtests for 2021/22 and 2022/23 (the only develop seasons with daily snapshots). Re-use logged `gws.parquet` runs of the live spec if they exist, otherwise run them.
- **Prices:** `player_snapshot` price per snapshot time.
- **For each transfer at deadline t** (player out `o`, player in `i`), over two windows ending at deadline t:
  - **24 h:** from deadline − 24 h (when the alert arrives);
  - **week:** from the previous deadline.
- **Perfect timing gain** = (o's best selling price in the window − o's selling price at the deadline) + (i's price at the deadline − i's lowest price in the window). The selling price follows FPL's rule from o's purchase price. Gains are ≥ 0 by construction.
- **Season gain** in £m, converted to points with 1b's GW1 rate (money saved early in a season compounds like money gained by planning).

## 3. Decision rule (fixed before any number is seen)

- **Planning:** if the upper bound is **≥ 20 points a season**, write the full foresight design next (planner with per-GW prices, backtest with real future prices vs fixed). A forecast usually captures a third to a half of perfect foresight.
- **Timing:** if the 24 h upper bound is **≥ 10 points a season**, design the live "transfer tonight or wait" hint. The week window is reported for context only, because alerts arrive 24 h before the deadline.
- **Both below:** stop. PLAN §1 keeps price prediction as a non-goal, with the measured numbers and a §12 row.

## Output

- `docs/research/2026-11-price-sizing.md`: the numbers, the assumptions (survivorship, linear scaling), the decision.
- Analysis scripts in `dev/` (`price_value_gap.py`, `price_timing.py`), run with `uv run python`. They read `data/` and `results/` only; the fetch writes to its output folder.
- Experiment log rows (`results/experiments.csv`) from the backtests, family `price sizing`.
- PLAN §12: a row for the decision.

## Not in scope

- The price forecast itself, any planner change, any bot change.
- The 2025/26 holdout, in any form.
- Archiving manager histories in `raw/`.

## Tests

- Rust: the bank modifier's unit test (start state) and a parse test for `template+bank10@1`, `random:3+bank10@20`; `cargo test --workspace`, `cargo clippy`.
- Scripts: a small unit test of the timing gain on a hand-made price series (sell before a drop, buy before a rise, no change → 0).
