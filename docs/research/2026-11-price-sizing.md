# Price changes: how much could a price model be worth? (2026-11)

**Question.** Is a price-change model worth building? It could earn points in two ways:
1. **Planning:** the planner sees future prices, keeps risers, sells fallers, and ends up with more money.
2. **Timing:** within a gameweek, make each transfer before a rise or after a drop.

**Method.** Upper bounds with perfect foresight, on develop seasons and 2026/27 GW1–9; never the 2025/26 holdout. The thresholds were fixed before any number was seen: planning ≥ 20 points a season, timing ≥ 10. Spec and addendum: `docs/superpowers/specs/2026-11-06-price-sizing-design.md`; scripts `dev/price_value_gap.py`, `dev/price_timing.py`, `dev/price_sizing.py`; Rust `fplopt-rs research money-value`.

**Decision: planning passes (≈ 47 points a season, 80% range 38–57). Timing fails (≈ 5 points a season). Next: the full-foresight design (planner with prices that change by GW).**

## 1. Team value: we fall behind every week

FPL team value (squad at current prices plus bank) at each GW of 2026/27:

| GW | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| Our system (backtest, live settings, template start) | 100.0 | 99.9 | 99.9 | 99.7 | 99.5 | 99.3 | 99.2 | 99.2 | 99.2 |
| Top-10k median (n 500) | 100.0 | 100.2 | 100.4 | 100.7 | 101.2 | 101.7 | 102.3 | 102.9 | 103.1 |
| 10k–100k median (n 500) | | | | | | | | | 102.7 |
| The user's team | 100.0 | 100.3 | 100.5 | 100.8 | 101.2 | 101.8 | 102.4 | 102.8 | 103.0 |

- **The gap:** at GW9 it is **£3.9m** to the top 10k (quartiles £102.6–103.6m) and £3.5m to 10k–100k.
- **Our system loses value:** it buys at market prices and ignores price trends. In the 2016/17–2022/23 develop backtests it also ends seasons at about £99.4m.
- **Caveats:**
  - The sample is "ranked there now", so it is survivor-biased: it overstates what a typical top manager had.
  - FPL's `value` for GW g is the squad priced at g's deadline plus the bank. This was checked on the user's entry, GW4–6.

## 2. What £1m is worth to the planner

**The first measurement was too noisy to read.** Paired backtests (2016/17–2022/23, start with and without +£1m) moved the season total by:

| Start GW | Change in season total | n | SE |
|---|---|---|---|
| GW1 | −7.8 | 20 | 10.0 |
| GW2 | +19.3 | 8 | 22.8 |
| GW20 | +10.6 | 28 | 10.2 |

One extra £1m changes every later decision, so the two runs drift apart. The pre-registered pooled rate came out at about 0 points per £1m, which made the planning upside 0 by construction.

**The re-measurement** (addendum; the user chose it before seeing any planning number) solved the live planner twice at the same state, with and without £1m. The states were 132 fresh template and random squads at GW1, 8, 15, 22 and 29 of 2016/17–2022/23.

v = the planner's expected points per GW per £1m:

| gw_index | 1 | 8 | 15 | 22 | 29 | 8–29 pooled |
|---|---|---|---|---|---|---|
| v | 0.31 | 0.45 | 0.33 | 0.54 | 0.27 | **0.395** |
| SE | 0.05 | 0.09 | 0.15 | 0.13 | 0.12 | 0.062 |
| n | 20 | 28 | 28 | 28 | 28 | 112 |

- **Noise:** individual states are noisy, because the planner stops within a 0.5% MIP gap, which is about the size of the effect. The mean is not biased by this.
- **Fresh squads:** the states are freshly built squads, not squads that evolved through a season.

## 3. Planning upside

- **Discount for over-prediction:** the expected gains are discounted by the base run's realized-on-predicted transfer-gain slope, c = **0.93**.
- **Exposure:** the measured top-10k gap for GW1–9, then held at £3.9m for GW10–38: **E = 130 £m·GW**.
- **Upside** = v × c × E = 0.395 × 0.93 × 130 ≈ **47 points a season**. Using v ± 1.28 SE, the 80% range is 38–57. That clears the 20-point bar comfortably.
- **If the gap kept growing linearly** to £16.5m at GW38 (reported, not decided on): E = 318, upside ≈ 116.
- **What 47 points is not:** it is not what a model would earn.
  - It assumes perfect foresight closes the whole gap to the top-10k median.
  - A real forecast gets part of it: typically a third to a half.
  - Prices that change also constrain plans in ways this bound ignores.
- **Why it still matters:** even a third (~15 points a season) is worth having. With ~2,320 points at the 100k cutoff and ~2,400 at 10k, 15–25 points is a fifth to a third of that 80-point distance.

## 4. Timing within the gameweek

Perfect timing on our system's transfers (2021/22 and 2022/23, full-season starts) would have saved:

| Window | Saved per season | Sell side |
|---|---|---|
| 24 h before the deadline (when the alert arrives) | **£0.78m** | 34% |
| Since the previous deadline | £2.8m | 36% |

- **In points:** at 0.395 × 0.93 per £1m per GW, and spread over the season (£0.78m reached gradually), that is about **5 points a season** (E = 0.78 × 38/2 = 15 £m·GW). Even counting all £0.78m from GW1, it is ~11, an upper bound that assumes every rise and drop is known.
- **Threshold:** the rule's own rate (the pre-registered backtest rate, ≈ 0) gives 0. Either way it is below the 10-point bar. **No timing hint.**

## 5. Decision and next step

- **Planning: build.** The next step is the full-foresight design (spec §3):
  - the planner gets per-GW prices, including FPL's selling rule for players bought and sold within the plan;
  - a develop backtest then runs it with the real future prices against fixed prices. That gives the true upper bound, and the price forecast later plugs into the same code.
  - Only then a forecast, a develop check, and a pre-registered check before it drives advice.
- **Timing: stop.** No live "transfer tonight" hint.
- **Diagnosis:** our system's slow value loss is itself a finding: it pays market price for players at their peak and holds fallers.

Experiment log: `results/experiments.csv`, family `price sizing` (the 2026 run, base and +£1m develop runs). Raw outputs: `results/price-sizing-*` (not tracked).
