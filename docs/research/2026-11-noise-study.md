# Noise study: no comparison method reaches 4.5 points a season

**Date:** 2026-11-07. Design: `docs/superpowers/specs/2026-11-07-noise-study-design.md`. Driver: `dev/noise_study.py`. Runs on the server in `~/fpl-pricec/results/noise-study` (commit d8e2da3); log family `noise study` (8 rows).

## Question

Which way of comparing two planners has a standard error of at most 4.5 points a season (enough to see a 10-point effect), with honest errors, at the least server time?

## Results

C1 = the price planner (`prices=forecast,price_k=0.5`) vs the live planner L on 2022/23. C2 = L with `hit_margin=3` vs L on 2021/22–2022/23. 16 starts a season. Points per season (A − B); n = blocks or decision windows.

| Method | C1 mean | C1 SE | n | C2 mean | C2 SE | n |
|---|---|---|---|---|---|---|
| Whole season, realized | −37.5 | 29.7 | 10 | −23.5 | 18.2 | 20 |
| Whole season, xG | −15.9 | 21.7 | 10 | −12.7 | 14.4 | 20 |
| Per decision k=1, realized | +14.9 | 10.5 | 37 | +11.6 | 5.8 | 75 |
| Per decision k=1, xG | +8.4 | 8.3 | 37 | +4.3 | **4.3** | 75 |
| Per decision k=2, realized | +16.9 | 10.3 | 19 | +3.6 | 7.3 | 38 |
| Per decision k=2, xG | +9.9 | **5.6** | 19 | −1.7 | 6.1 | 38 |
| Per decision k=4, realized | +15.2 | 19.9 | 10 | +12.6 | 10.6 | 20 |
| Per decision k=4, xG | +16.0 | 13.0 | 10 | +1.9 | 8.5 | 20 |

Server time per compare run (14 jobs, warm xP cache): whole season 14.5 min (C1, 1 season) and 17.2 min (C2, 2 seasons); each per-decision run 30–35 min.

## Verdict

- **No method passes.** None has SE ≤ 4.5 on both C1 and C2. The best are k=1 xG (8.3 / 4.3) and k=2 xG (5.6 / 6.1).
- **The placebo pairs were not run.** They could not change the verdict, and the method that follows (approach B) needs its own placebo check, so checking these SEs was mostly wasted server time (10–15 h). One placebo run was started and stopped before it finished; nothing from it was logged.
- **Nothing is adopted** as the standard comparison; PLAN §7 and CLAUDE.md are unchanged.

## What we learned

- **xG scoring and short decision windows cut the SE 3–4×** against whole-season realized points (C1 29.7 → 5.6, C2 18.2 → 4.3). Per-decision windows remove the luck that compounds as the two squads drift apart; xG scoring removes finishing luck.
- **Whole-season and per-decision results disagree for the price planner** (−16 to −38 vs +8 to +17). The per-decision method measures one decision with each arm continuing its own plan, which is not a full season's effect when effects compound (the spec's second risk). All C1 numbers are within about 2 SE of 0, so this is not a verdict on the price planner.
- **The season-clustered SE tells us nothing with 2 seasons** (1 df: it is half the gap between the two season means, from 1.2 to 9.7 here). Whether pooled windows understate the error is still open; only a placebo check answers it.
- **Unchecked:** whether these SEs are honest (no placebos), and the per-decision transfer asymmetry under `--reference b`.

## Seven seasons and placebos (2026-11-08): the standard comparison

C2 on all seven develop seasons (`results/noise-seasons/` on the server, family `noise study seasons`, about 1 h 55 min a run at `--jobs 14`, with the xP cache cold before 2021):

| Method | Mean | SE (pooled) | SE (season means, 6 df) |
|---|---|---|---|
| Per decision k=1, realized | +2.8 | **2.1** | 2.5 |
| Per decision k=2, realized | −0.1 | 2.8 | 3.4 |
| Per decision k=1, xG (2019/20+ only) | +0.9 | 2.5 | 2.1 |
| Per decision k=2, xG (2019/20+ only) | −3.1 | 3.5 | 3.0 |
| Whole season, realized | −12.6 | 7.7 | – |
| Whole season, xG | −6.1 | 8.5 | – |

xG points are missing before 2019/20 (an input is null; not traced), so on seven seasons real points win.

Four placebo pairs (`xp_jitter=0.05`, seeds 1/2 … 7/8), per decision k=1 realized, seven seasons (family `noise study seasons placebo`):

| Pair | Mean | SE pooled | SE season | Same decision | A − B transfers a window |
|---|---|---|---|---|---|
| P1 | −2.2 | 9.3 | 12.5 | 5% | +0.035 |
| P2 | −2.7 | 9.4 | 11.3 | 6% | +0.029 |
| P3 | −0.8 | 10.5 | 9.4 | 6% | +0.029 |
| P4 | −6.4 | 9.8 | 13.4 | 6% | +0.038 |

- **The SEs are honest or conservative:** the RMS of the normal-score z is 0.38 pooled and 0.27 by season, not above 1.
- **No bias detected:** the mean of the four is −3.1, against an SE of about 5. All four being negative has a 1-in-8 chance under no bias; A makes about 0.03 more transfers a window under `--reference b`.
- **The SE depends on how often the arms disagree.** The placebos (5% jitter) change almost every decision, hence an SE of 9–13. Hit margin 3 vs 2 matches in ~70% of windows, hence 2.1–2.5.

**Adopted** as the standard comparison (PLAN §5, CLAUDE.md): per decision k=1, realized points, all seven develop seasons, 16 starts, judged on the larger of the two SEs. It counts only the decision's own GW, so effects that come later (money kept, FTs banked, chips) are measured separately.

The 10-pair placebo criterion of the original spec was not run; four pairs are a rough check (they catch a bias of several points or SEs off by a factor, not finer).

## Next (approach B, as planned before the seven-season runs)

- **More seasons.** The SE falls with √(seasons). C2 is running on all seven develop seasons (2016/17–2022/23) with k=2 and k=1, xG and realized (`results/noise-seasons/` on the server, family `noise study seasons`). Expected k=2 xG SE ≈ 6.1 × √(2/7) ≈ 3.3. Anything that needs price snapshots has only 2021/22–2022/23 (≈ 4 for the price planner at k=2 xG).
- **Dropped: a control variate on the arms' predicted xP.** A control variate needs a quantity with a known mean; the predicted xP difference between two planners carries the very effect being measured, so subtracting it would remove the effect. xG scoring is the valid form of the idea.
- **Then one placebo check** of the method that comes out, before it becomes the standard comparison.
