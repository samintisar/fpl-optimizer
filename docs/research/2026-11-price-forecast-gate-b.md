# Price forecast: Gate B (2026-11-07)

**Question (spec `docs/superpowers/specs/2026-11-06-price-planner-design.md`, Stage B and addendum).** Does a forecast of each player's price change 1–6 deadlines ahead, built only from what is known at each deadline, beat two baselines on seasons it has never seen? The baselines are "no change" (N0, today's planner) and momentum (N1).

**Forecast.** One LightGBM regression per horizon h = 1..6 on:
- snapshot features: price changes over 1, 7 and 28 days; net transfers this GW and 1 and 3 days earlier, scaled by ownership and by market activity; ownership; price relative to the position; status, chance of playing and days since the news;
- v1 xP for the next 1 and 3 GWs, and its rank within the position;
- the gw_index and the days to the next deadline.

Predictions are clipped to ±£0.3m per GW. Code: `fplopt_features::price`, `fplopt_models::price`, `fplopt-rs research price-eval`.

**Leakage checks.**
- `check leakage` passed with `price_features` (18 deadlines).
- A unit test pins that training rows contain no targets past the fit cutoff.
- A review confirmed the input price equals the target's base price on all 103,247 rows of 2021/22–2024/25.

## Develop (2022/23, fitted on 2021/22): choosing the setting

Six presets were tried and every one was logged (family `price forecast develop`). Pooled MSE over h = 1..6, in squared tenths of £m:

| p1 | p2 | p3 | p4 | p5 | p6 | N0 | N1 |
|---|---|---|---|---|---|---|---|
| 0.7526 | 0.7626 | 0.7594 | **0.7468** | 0.7659 | 0.7588 | 0.9327 | 2.6017 |

p4 (learning rate 0.02, 31 leaves, min 200 per leaf) was frozen in `config/prices/forecast.json` and tagged `price-forecast-gate-b` (e51668b) before the gate ran. The spread across presets is small, so the result doesn't hinge on the choice.

## Gate B (2023/24 and 2024/25, run once from the tag)

The 2023/24 fit used 2021/22–2022/23 and the 2024/25 fit used 2021/22–2023/24; 2025/26 was never used. The run covered 76 deadlines and 67,344 scored candidate rows, and took 1,144 s. Family `price forecast gate b`; the log row was written before the result was printed.

| h | rows | forecast | N0 | N1 |
|---|---|---|---|---|
| 1 | 12,059 | **0.122** | 0.177 | 0.263 |
| 2 | 11,723 | **0.320** | 0.454 | 0.845 |
| 3 | 11,405 | **0.608** | 0.795 | 1.815 |
| 4 | 11,046 | **0.948** | 1.164 | 3.163 |
| 5 | 10,757 | **1.324** | 1.555 | 4.904 |
| 6 | 10,354 | **1.688** | 1.931 | 7.074 |
| **pooled 1–6** | 67,344 | **0.807** | 0.982 | 2.891 |

Diebold–Mariano tests, one-sided, on per-deadline MSE pooled over h, lag 6, 74 deadline clusters:
- **forecast vs N0:** mean difference −0.164, t −6.27, **p 1.1e-8**;
- **forecast vs N1:** mean difference −1.975, t −11.18, **p 8.7e-18**.

**Gate B: pass.** The rule was lower pooled MSE than both baselines, with p < 0.10 against each.

## Reading it

- **Error reduction against "no change":** 18% pooled (0.807 vs 0.982). That is 31% at one deadline ahead and 13% at six, as expected, since transfers foretell the next few nights best.
- **Develop and check agree:** the reductions were 20% and 18%. There is no sign of overfitting to 2022/23.
- **Momentum is far worse than "no change".** Price moves don't persist mechanically; the forecast's edge comes from transfers, ownership and form.
- **Error is not points.** A lower price error is necessary but not enough. Stage C tests whether the planner turns these paths into team value without losing points:
  - develop tuning of k ∈ {0, 0.25, 0.5, 1};
  - then a pre-registered backtest on 2023/24–2024/25, which must give higher end team value and a season-points 80% lower bound above −10.

## Next

Stage C's plan:
- `prices=forecast` in the backtester, fitted per season as here;
- the k grid on develop 2022/23;
- the frozen settings and the run-once check on 2023/24–2024/25.
