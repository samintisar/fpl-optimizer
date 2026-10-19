# Ideas

Ranked: the loop takes the top `open` idea it can test tonight. Each idea says what to
change, the evidence, and how it is judged. Update the status and add what was learned; move
finished ideas to *Done*. New ideas from the weekly monitor go in with their live evidence.

## Open

1. **P(start) still compressed: next steps after `club_continuity`** (model).
   - Live GW2–6 (report 2026-10-19-gw6): by band ≤0.3 0.069 vs 0.037, 0.7–0.8 0.754 vs
     0.859, 0.8–0.9 0.855 vs 0.929; regulars 0.763 vs 0.824 every GW from GW2. Develop
     walk: same direction, 3–4 pp in GW2–12, largest in 2021/22 (0.787 vs 0.833) and
     2022/23 (0.784 vs 0.849).
   - Tried: global sharpening `start_sharpen=1.15` (DISCARD: helps 2021–22, hurts 2017–20);
     `club_continuity` feature (KEEP, MSE h0 −0.0017). Re-slice the develop walk at the new
     head first: how much of the band gap is left, by team-continuity quartile and season.
   - Candidates if a gap remains: continuity over a longer window or within the season
     only; the player's own share of the club's last 5 XIs (a stability-weighted start
     rate). One per night; judge with `model_eval.sh`.
2. **Defcon calibration** (model; the measurement is kept on `research`, 243c39f).
   - Live check (PLAN §6.5, GW2–5, horizon 0): DEF 0.088 predicted vs 0.104 realized, MID
     0.047 vs 0.038. Weekly monitor GW1–6 realized: DEF 0.104 (60+ min 0.266), MID 0.043
     (0.136), FWD 0.002; predicted can't be read until the tooling reaches `main` (the
     monitor runs main's frames). DEF points under-predicted by 0.14 per player-GW at h0.
   - Develop has no defcon points, so test on develop via P(threshold reached) on
     2016/17–2018/19 rows that carry the old detailed stats only if their definitions
     match (PLAN §3, VERIFY); otherwise live-evidence-only: write the case for the user.
3. **#43 finishing above xG** (model).
   - A long-window goals − npxG term, shrunk hard (empirical Bayes), in `shares`. Live
     GW1–6 over-performers: João Pedro +3.3, Cherki +3.2, Rogers +3.0, Haaland +2.5 (noise
     at 6 GWs). Test on develop with `model_eval.sh` (goals_ll_h0 is in the judge).
4. **Bonus season weighting** (model).
   - BPS changed in 2026/27; the bonus fit weights 4 seasons by 0.7 per year. Develop can
     only test the weighting itself (current-season data weighted more), not the new rules.
     `e_bonus` is now scored by the xP evaluation (243c39f): read it on develop first.
5. **Free-transfer value** (decision; `ft_scale` key on `research`, c6eb489).
   - 1.5: DISCARD on realized (+0.045/GW, CI [−0.318, +0.429]) but xG-scored +0.446/GW,
     CI [+0.051, +0.842], p 0.073; seasons split 4–3. Next value: 2.0 (the xG signal says
     higher), then 0.5 only if 2.0 is flat. Two values left under the 3-value rule.
6. **P(CS) team level** (model; watch).
   - Live GW1–6: team P(CS) 0.25 predicted vs 0.27–0.33 realized by position, but GW to GW
     0.13–0.40: a few team-weeks. Recheck at GW10; brier_cs on develop is flat so far.

## Needs the user (not the loop's to do)

- **#39 `back_split` on** (was idea 1). The loop can't judge it: the pinned harness's walk
  builds `V1Params(..., back_split=args.back_split)` and `model_eval.sh` passes no
  `--back-split`, so a flipped default is never scored; and a flipped `V1Params` default
  changes `calibration_fingerprint`, so the frozen calibration table stops matching. Needs
  a harness option (or a manual walk) and a regenerated table. Live GW1–6 is mixed: players
  back from a 0-minute GW started 0.095 vs 0.155 predicted (n 346; p_start > 0.3: 0.333 vs
  0.402, n 45), i.e. over-predicted, not the #39 cell but worth reading before promoting.
- **Calibration table after `club_continuity`**: the table was fitted on walks without the
  feature; the fingerprint doesn't see feature changes. Regenerate (`dev/calibrate_v1.py
  walk` + `fit --write`) as part of promoting it.
- European / cup fixtures for rotation risk: no source in `raw/`; a new adapter and build
  change.
- Shadow-policy scoring of the bot's live recommendations (PLAN §8): reads the bot's
  `state/bot.sqlite`.

## Done

- **Measure defcon and bonus** (tooling, KEEP 2026-10-18, 243c39f + 31a9115): `p_defcon`,
  `e_bonus` components and the summed `pts_*` columns in the GW frame; the xP evaluation
  joins realized bonus and defcon and scores them. model_eval metrics identical.
- **Club continuity** (model, KEEP 2026-10-19, ef9715e + c56a1da): `club_continuity` in
  `LAG_FEATURES`. MSE h0 4.26092 → 4.25920, h1–5 4.68633 → 4.68530, 4 seasons better / 2
  worse (2017 +0.0013, 2021 +0.0010), P(start) Brier h0 −0.14 %.
- **Solver gap** (decision, DISCARD 2026-10-19; `mip_gap` key kept, e3bf5a0): 0.0005 vs
  0.005, realized −0.092/GW, CI [−0.402, +0.202]; seasons swing ±60 points from the gap
  alone, so the 0.5 % gap is not the binding source of noise. Not worth another value.
- **Global P(start) sharpening** (model, DISCARD 2026-10-19): `start_sharpen=1.15`, MSE h0
  +0.0016; 2021/22 −0.0027, 2022/23 −0.0075, 2017–2020 all worse.
- **Minutes: absence counted in club fixtures** (model, DISCARD 2026-10-19): days_since_app
  with gaps capped at 10 days; worse in all six seasons (MSE h0 +0.0012). Calendar days
  carry signal beyond `absent_run` (which already counts missed fixtures).
