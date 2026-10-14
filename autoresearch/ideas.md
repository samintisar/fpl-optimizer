# Ideas

Ranked: the loop takes the top `open` idea it can test tonight. Each idea says what to
change, the evidence, and how it is judged. Update the status and add what was learned; move
finished ideas to *Done*. New ideas from the weekly monitor go in with their live evidence.

## Open

1. **#39 back from a missed match: turn `V1Params.back_split` on** (model).
   - Code is on `research` (commit 9cd0078), off by default. Flip the default to True.
   - Evidence: develop 2021/22–2022/23, `a100` regulars after one miss started 69–71%
     against 0.54 predicted; with the cells, 0.66. A manual walk (2026-10-14) gave MSE h0
     4.2609 → 4.2590, h1–5 4.6863 → 4.6855, earlier seasons identical.
   - Judge: `model_eval.sh` + `judge.py` (the first real run of the loop's harness).
2. **Measure defcon and bonus** (tooling: no xP change).
   - The GW xP frame drops per-component points (`assemble.gw_frame` keeps only `xp`,
     probabilities and sums); `evaluate.metrics.COMPONENTS` scores neither defcon nor bonus.
   - Add `p_defcon` and the summed `pts_*` columns as GW extras (not scored as xP), and
     `p_defcon` (target: defcon threshold reached) to `COMPONENTS`.
   - Keep rule: tests pass and `model_eval.sh` metrics are identical to the baseline.
3. **Defcon calibration** (model; needs 2).
   - Live check (PLAN §6.5, GW2–5, horizon 0): DEF 0.088 predicted vs 0.104 realized, MID
     0.047 vs 0.038; defcon drives some picks (Silva, Thiaw, PLAN §8 live findings).
   - Develop has no defcon points (scoring rules), so test on develop via P(threshold
     reached) on 2016/17–2018/19 rows that carry the old detailed stats only if their
     definitions match (PLAN §3, VERIFY); otherwise this is a live-evidence-only idea: write
     the case in the report for the user, don't tune on live.
4. **Minutes: absence counted in club fixtures** (model).
   - `days_since_app` counts calendar days, so one missed match before an international
     break looks like a 4-week absence (João Pedro, GW5 → GW6 gap 22 days).
   - Candidate: replace or complement it with the club fixtures since his last appearance,
     or the days excluding the gaps between club fixtures over 10 days. Retrain minutes.
5. **#43 finishing above xG** (model).
   - A long-window goals − npxG term, shrunk hard (empirical Bayes), in `shares`. Watch the
     high over-performers on develop. See the issue for the live case.
6. **Bonus season weighting** (model).
   - BPS changed in 2026/27; the bonus fit weights 4 seasons by 0.7 per year. Develop can
     only test the weighting itself (current-season data weighted more), not the new rules.
7. **Free-transfer value** (decision; needs a spec key).
   - Open-fpl-solver's defaults, never tuned; the live settings hold 1 FT in 87% of
     develop decisions. Add `ft_scale` (multiplies `DEFAULT_FT_VALUE`), test 0.5, 1.5, 2.0.
8. **Solver gap** (decision; needs a spec key).
   - The live gap is 0.5%: ~1 point on a 6-GW objective, about the size of plan 1's gain
     over saving the FT (GW6 +1.4, GW7 +0.2). Add `mip_gap` as a key; test 0.0005 (watch
     solve times in the GW rows).

## Needs the user (not the loop's to do)

- European / cup fixtures for rotation risk: no source in `raw/`; a new adapter and build
  change.
- Shadow-policy scoring of the bot's live recommendations (PLAN §8): reads botbox's
  `state/bot.sqlite`.

## Done

(none yet)
