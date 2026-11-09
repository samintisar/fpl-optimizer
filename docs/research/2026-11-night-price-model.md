# Night price model: Gate N fails on calibration of rises

**Date:** 2026-11-09. **Run:** once, from tag `night-model-gate-n` (commit d30d17c), family `night price model gate n`, logged in `results/experiments.csv`; output `results/night-gate-n/` (gitignored). Spec: `docs/superpowers/specs/2026-11-09-move-timing-design.md` (Gate N, with the calibration amendment made before the run). Plan: `docs/superpowers/plans/2026-11-09-night-price-model.md`.

**Verdict: FAIL.** The model ranks nights very well and beats both baselines decisively on both targets. But its calibrated rise probabilities are far too high where policy T would act: rises happen about half as often as predicted. By the spec, T is not run.

## What was evaluated

- **Rows:** every pool player at 22:00 UK on each night of 2022/23: 295 nights and 203,480 scored player-nights. The label is the first snapshot between 04:00 and 12:00 UK the next morning, minus the evening price. A row is unscored if the player's last snapshot is older than 04:00 that day.
- **Model:** two LightGBM boosters (rise and fall tonight) on 14 view-only features (`fplopt_features::night`). They are fitted on 2020/21's snapshot nights and 2021/22. Each is then Platt-calibrated on out-of-fold predictions over 5 contiguous blocks of training nights.
- **Baselines:**
  - B0 is the training base rate.
  - B1 is a booster on net transfers per owner, calibrated the same way.
- **Gate:** for each target, a one-sided Diebold–Mariano test of the model against B0 and against B1, per night, lag 7, p < 0.10. Each target must also pass the calibration condition: among the rows with calibrated p ≥ 0.3, n ≥ 30 and an observed rate ≥ the mean p − 0.10.

## Results

**Log loss** (calibrated; raw in brackets):

| target | model | B0 | B1 |
|---|---|---|---|
| rise | 0.00840 (0.00962) | 0.01587 | 0.01378 (0.01381) |
| fall | 0.02962 (0.02982) | 0.04485 | 0.04200 (0.04235) |

All four DM tests: p < 0.001. Against B0, the model's per-night loss is 47% lower on rises and 34% lower on falls.

**Platt maps** (a, b): rise model −0.50, 0.67; fall model −0.31, 0.86; B1 rise −1.74, 0.64; B1 fall −1.21, 0.73.

**Calibration condition** (p ≥ 0.3):

| target | n | mean p | rate | result |
|---|---|---|---|---|
| rise | 1,046 | 0.488 | 0.226 | **fail** |
| fall | 791 | 0.424 | 0.353 | pass |

**At T's threshold τ = 0.5** (report only, added before the run):
- rise: n 425, mean p 0.631, rate 0.306 (90% Wilson 0.270–0.344);
- fall: n 171, mean p 0.597, rate 0.567 (0.504–0.628).

**Calibration bins** (calibrated; n, mean p → rate):
- rise: 0.1–0.2: 1,010, 0.14 → 0.07; 0.3–0.4: 354, 0.35 → 0.17; 0.5–0.6: 194, 0.55 → 0.27; 0.6–0.7: 129, 0.65 → 0.33; 0.7–0.8: 76, 0.74 → 0.32; 0.8–0.9: 24, 0.84 → 0.42.
- fall: 0.1–0.2: 2,489, 0.14 → 0.12; 0.3–0.4: 421, 0.34 → 0.27; 0.5–0.6: 97, 0.55 → 0.45; 0.6–0.7: 63, 0.64 → 0.67.
- Raw rise probabilities were worse still: 0.94 → 0.37 in the top bin.

## Reading

- **Falls are usable.** They are close to calibrated, and at τ the interval covers the predicted rate.
- **Rises are not.** At p ≥ 0.5, about 3 in 10 predicted rises happen. The out-of-fold Platt map halved the overconfidence but could not remove it.
  - The miscalibration looks like a shift between seasons. The 2021/22 smoke run, fitted on 2020/21's 65 nights, was overconfident in the same direction (p ≥ 0.3: 0.457 → 0.146), so one season's map does not carry to the next.
  - A plausible cause is FPL's rise threshold changing between 2020/21–2021/22 and 2022/23. It is a hidden rule and is retuned, which is the spec's "price rule drift" risk.
- **Ranking is not the same as acting.** T acts on the probability itself (expected saving ≥ 0.5 tenths), so a model that ranks well but rises at 0.3 where it says 0.63 would make many night buys that save nothing and decide on staler news.

## Disclosures

- **A smoke run on 2021/22 happened before the gate.** It was fitted on 2020/21 and evaluated on 2021/22, a training season for the gate, so it was not a look at 2022/23. It showed the overconfidence, and two things followed from it:
  - The calibration amendment (Platt plus the condition) was registered before the run.
  - Step-halving was added to the Platt fit after plain Newton diverged (a = −1.9e13) on those overconfident predictions.
  - Nothing was tuned on it: the booster parameters are the plan's defaults.
- **Two report-only additions were made before the run:** the τ = 0.5 row, and `fit_platt` failing instead of returning an unconverged map.
- **Platt's out-of-fold blocks come from the training seasons only.** About 80% of those nights are in 2021/22, so the map cannot learn a change between seasons. That is why the condition is out of sample.
- **Pre-season transfers are carried into the cumulative net transfers.** A snapshot's net transfers are offset at the first deadline after it. 2020/21's series starts in April 2021, when the snapshots begin.
- **The condition is a coarse test.** It pools p ≥ 0.3, and at n = 30 its SE would be about 0.09. Here n is over 1,000, so the rise failure is not noise: the rate's 90% interval at p ≥ 0.5 is 0.27–0.34 against a mean p of 0.63.

## What next (not decided)

Gate N ran once and is spent. Any change is a new model with its own pre-registered gate, on seasons not yet used for evaluation. 2022/23 is now seen. 2023/24–2024/25 stay clean until a frozen version passes on develop. The options are for the user to choose:
1. **A fall-only T:** make a planned sale tonight when a fall is likely. Falls passed here, but a fall-only design needs its own spec and a new develop season to test it.
2. **Rework the rise model for drift:** recency weighting, a per-season threshold feature, or calibration on the most recent nights. It would need a new evaluation season, and 2022/23 is spent.
3. **Park move timing.** The sizing put the money at about £2.3m a season with perfect timing, so T's realistic share was smaller to begin with.
