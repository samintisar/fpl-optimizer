# Price-aware planner: Gate A (2026-11-06)

**Question (spec `docs/superpowers/specs/2026-11-06-price-planner-design.md`, Stage A).** If the planner knew future prices, would it turn them into points and team value? The answer decides whether to build a price forecast.

**Runs.** Develop seasons 2016/17–2022/23, starts `template@1,random:3@1,template@20,random:3@20` (56 season × start units). Template GW1 starts before 2021/22 fall back to GW2. Family `price planner A` in `results/experiments.csv`. The three arms:

- **L:** the live planner (`optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2`).
- **O:** L + `prices=oracle,price_k=1`. The real future deadline prices, plus the end-of-horizon money value.
- **F:** L + `price_k=1`. The money value alone, with fixed prices.

**Equivalence check.** L reproduces the price sizing's `price-sizing-base` run exactly: all 1,580 rows on every column. The new code leaves the default planner unchanged.

## Results

Paired by (season, start), arm − L, with 80% t intervals (units treated as independent, so the intervals are a little narrow):

| | Season points | End team value |
|---|---|---|
| **O − L**, all 56 units | **+140** (116 to 164) | **+£3.1m** (2.7 to 3.6) |
| O − L, full-season starts | +255 (2,463 vs 2,208) | +£5.3m (£104.4m vs £99.1m) |
| O − L, GW20 starts | +25 (1,184 vs 1,159) | +£1.0m |
| **F − L**, all 56 units | **−10.5** (−18.5 to −2.4) | **−£0.2m** (−0.3 to −0.1) |

Run times at `--jobs 14`: L 1,168 s (it filled the xP cache), O 1,398 s, F 1,051 s. Solves stay fast with cohort variables.

**Gate A: pass.** The bar was O − L ≥ +£2m end value and ≥ +10 points.

## Reading it

- **The oracle bound overstates what a forecast can earn.** +255 points over a full season is more than five times the sizing's upper bound (about 47). Prices rise because managers buy players who are playing and scoring, so future prices carry information about future form. O's gain mixes:
  - money from price changes (what a forecast can capture);
  - that hidden peek at future returns (which no forecast can have).

  The bound is still useful: it says the planner can act on price paths. But it is not a size estimate. The sizing's ~47 points remains the better estimate of the money effect.
- **The money value alone hurts slightly (F).** With fixed prices, valuing end-of-horizon money makes the planner reluctant to sell owned players who have risen (FPL keeps half the rise). It also leans toward cash over points: −10.5 points and slightly less team value. So the value of a price model comes from the price information, not the money term. Stage C should therefore include small k values and k = 0 with forecast prices, and judge them on points.

## Decision and next step

- Stage B (the price forecast) goes ahead.
- Its own gate (accuracy against "no change" and momentum, on 2023/24–2024/25) and Stage C's pre-registered backtest are the real tests. A forecast built only from information available at each deadline can't use the form leak that inflates O.
- **Stage C change, recorded before any forecast exists:** the k grid becomes {0, 0.25, 0.5, 1}. With k = 0 and forecast paths, the planner still uses prices in its budget within the horizon. This adds one variant to the deflation family (N = 4).
