# Ideas

Ranked: the loop takes the top `open` idea it can test tonight. Each idea says what to
change, the evidence, and how it is judged. Update the status and add what was learned; move
finished ideas to *Done*. New ideas from the weekly monitor go in with their live evidence.

## Open

1. **P(start) still compressed: next steps** (model).
   - Live GW2–6 (report 2026-10-19-gw6): by band ≤0.3 0.069 vs 0.037, 0.7–0.8 0.754 vs
     0.859, 0.8–0.9 0.855 vs 0.929; regulars 0.763 vs 0.824 every GW from GW2. Develop
     walk: same direction, 3–4 pp in GW2–12, largest in 2021/22 (0.787 vs 0.833) and
     2022/23 (0.784 vs 0.849); the ≤0.3 band is over-predicted too (GW2–6 0.078 vs 0.065).
   - Tried: `start_sharpen=1.15` (DISCARD); `club_continuity` (KEEP); `club_fixtures_season`
     (KEEP 2026-10-20, MSE h0 −0.0013, 5 of 6 seasons better); `five_subs` rule indicator
     (DISCARD: 2022 MSE +0.003, though minutes log loss −0.33 %, see idea 2).
   - Ruled out (diagnostic 2026-10-20, `~/fpl-research/results/research/diag-norm.txt`): the
     team normalization. The GBM's raw P(start) sums to 11.0–11.2 per club at h0 and the
     median logit shift is ±0.05 (2019/2021/2022 × GW4/10/22), so the shift moves a regular
     by < 0.01. The compression is in the GBM itself.
   - Open leads: label noise from inferred starts (before 2022/23 GW16; 569 of the 1,016
     wrong rows are 45-minute half-time ties, a coin flip between a regular and a sub):
     weight 0 for training rows tied at the 11th/12th spot of an inferred fixture. Re-slice
     the P(start) bands after tonight's keeps first: they may have moved the gap.
2. **Five subs: better minutes likelihood, worse points MSE** (model; new 2026-10-20).
   - `five_subs` (2019/20 GW39+, 2022/23 on) as a minutes feature: minutes_ll_h0 −0.33 %
     (the biggest minutes gain of any change so far), |mean xP − points| 0.0110 → 0.0092,
     but MSE h0 +0.0005, all of it 2022/23 (+0.0029). In the 60+/sub parts only: the same
     (the gain is there, not in P(start)). Before another variant, find which rows' error
     grows in 2022/23 (by position and minutes band; appearance vs goal points): probably the
     points assembly reacting to the new P(60+). Diagnostic first, no variant.
3. **Defcon calibration** (model; the measurement is kept on `research`, 243c39f).
   - Live check (PLAN §6.5, GW2–5, horizon 0): DEF 0.088 predicted vs 0.104 realized, MID
     0.047 vs 0.038. Weekly monitor GW1–6 realized: DEF 0.104 (60+ min 0.266), MID 0.043
     (0.136), FWD 0.002; predicted can't be read until the tooling reaches `main` (the
     monitor runs main's frames). DEF points under-predicted by 0.14 per player-GW at h0.
   - Develop has no defcon points, so test on develop via P(threshold reached) on
     2016/17–2018/19 rows that carry the old detailed stats only if their definitions
     match (PLAN §3, VERIFY); otherwise live-evidence-only: write the case for the user.
4. **Bonus season weighting** (model; low value on develop).
   - BPS changed in 2026/27; the bonus fit weights 4 seasons by 0.7 per year. Develop can
     only test the weighting itself (current-season data weighted more), not the new rules,
     and `season_decay` is shared with saves and cards. `e_bonus` is now scored by the xP
     evaluation (243c39f): read it on develop first.
5. **Free-transfer value** (decision; `ft_scale` key on `research`, c6eb489). **Blocked on
   `research`** since `back_split` is on: calibrated v1 raises (fingerprint), so every
   `optimizer:v1` backtest at the head fails until the table is regenerated (*Needs the
   user*).
   - 1.5: DISCARD on realized (+0.045/GW, CI [−0.318, +0.429]) but xG-scored +0.446/GW,
     CI [+0.051, +0.842], p 0.073; seasons split 4–3. Next value: 2.0 (the xG signal says
     higher), then 0.5 only if 2.0 is flat. Two values left under the 3-value rule.
6. **P(CS) team level** (model; watch).
   - Live GW1–6: team P(CS) 0.25 predicted vs 0.27–0.33 realized by position, but GW to GW
     0.13–0.40: a few team-weeks. Recheck at GW10; brier_cs on develop is flat so far.

## Needs the user (not the loop's to do)

- **The calibration table on `research`** (blocks decision experiments): with
  `back_split=True` and `finishing_prior=25` the fingerprint no longer matches, so
  calibrated v1 raises at the `research` head. 5 model tests fail by design (the table test
  and 4 that call calibrated v1), and `optimizer:v1` backtests can't run there. The features
  kept since the table (`club_continuity`, `club_fixtures_season`) change predictions
  without changing the fingerprint. Regenerating the table on `research` (the calibration
  walk + fit --write, frozen for the loop) would unblock decision experiments; otherwise it
  happens at promotion.
- European / cup fixtures for rotation risk: no source in `raw/`; a new adapter and build
  change.
- Shadow-policy scoring of the bot's live recommendations (PLAN §8): reads the bot's
  `state/bot.sqlite`.

## Done

- **#39 `back_split` on** (model, KEEP 2026-10-20, fe3e683): MSE h0 −0.0019, h1–5 −0.0008;
  2021 −0.0102, 2022 −0.0013, 2017–2020 identical. Live caveat stands: read the `a`/`a100`
  back-from-a-miss cell at the next weekly monitor before promoting (GW1–6: players back from
  a 0-minute GW started 0.095 vs 0.155 predicted, n 346; not exactly this cell).
- **Season phase** (model, KEEP 2026-10-20, 2161931): `club_fixtures_season` in
  `LAG_FEATURES`; MSE h0 −0.0013, h1–5 −0.0007, 5 seasons better (2019 +0.0014).
- **#43 finishing above npxG** (model, KEEP 2026-10-20, 7da0e49): goal share × (np goals +
  25) / (npxG + 25) over his own-npxG rows; MSE h0 −0.0020, h1–5 −0.0026, 5 seasons better,
  goal Brier −0.11 %. k = 12: flat (DISCARD). FPL assists over xA, same form: flat (DISCARD).
  A third value only if a re-slice says the multiplier over-reacts.
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
