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
   - Live GW1–7 with the promoted model (report 2026-10-21-gw7): regulars 0.773 vs 0.818;
     0.8–0.9 band 0.855 vs 0.928; by GW 0.79 vs 0.82–0.87 from GW2 (GW7 0.799 vs 0.824,
     the smallest gap yet). The promotion (back_split, continuity, season phase, the
     calibration refit) did not close it.
   - Tried 2026-10-21: tie label noise (`tied_start_weight=0`, DISCARD: brier_start_h0
     unchanged, so the ties are not the cause); `start_per_app_l10` (DISCARD: the GBM
     already has it). Develop walk (`diag-pstart.txt`): **regulars are under-predicted in
     2021 and 2022 in every phase** (2021 GW7–12 0.795 vs 0.842; 2022 GW13+ 0.766 vs 0.815),
     while 2017–2020 are within ±0.02. That points to an era shift, not the early season.
   - **Found 2026-10-23: what changed from 2021/22 is the flag layer** (FPL flags exist from
     2020/21 GW32). `diag-avail.txt` (18 deadlines 2021–22, h0): the layer fits one logit
     slope (0.73–0.83) for every flagged player, plain `a` included; a raw 0.93 became
     0.90–0.91 against 0.94–0.96 realized, the ≤ 0.3 band 0.05 → 0.07 against 0.04–0.06.
     Overall Brier still improves (the i/u/d cells), but regulars are compressed.
   - Why the slope is low (`diag-slopes.txt`, `diag-slope-h.txt`, `diag-oof.txt`): not the
     other categories (plain `a` alone fits 0.79) but **the horizon**. Raw P(start) is
     over-confident at longer horizons: recal slope of the real raw predictions for plain
     `a` 1.06 at h0, 0.91 h1, 0.84 h2, 0.73 h3+ (a100 1.00 → 0.67); the fit rows (out-of-fold
     base) show the same (0.96–1.01 → 0.71–0.75), so the base is fine. h3+ is half the fit
     rows, so the one slope is set by them and over-compresses h0 (after the layer: recal
     slope 1.19 at h0, 0.91 at h3+).
   - **And h ≥ 1 is corrected twice**: the harness calibrates P(start) isotonically at h ≥ 1
     only (`--parts p_start:1+`, h0 left as is), with maps learned on earlier seasons'
     walks, mostly pre-flag. So a layer slope fitted to h ≥ 1 outcomes double-counts there.
   - Tried 2026-10-23: `category_slopes` (KEEP: MSE h0 −0.00088, 2022 −0.0047); per
     (category, bucket) cell slopes (DISCARD: MSE h0 −0.0029, 2021 −0.0068, 2022 −0.0108,
     brier_start_h0 −0.39 %, but h1–5 +0.0012); own slope at h0 and one for h ≥ 1 per
     category (DISCARD: h0 the same −0.0029, h1–5 +0.0009). Three values of the slope
     setting: closed for the loop.
   - **Fixed at the source 2026-10-24: `horizon_shrink` (KEEP, 8eaf71c).** The over-confidence
     at h ≥ 1 is in the minutes model: a regular's P(start) stays ~0.90 at every horizon (the
     decay can't go below his long-run rate) while his realized rate falls 0.91 → 0.82 by
     h3+. logit p_start × 0.92^h after the decay (minutes model and the flag fit's base):
     MSE h0 −0.00193, h1–5 −0.00234, 2021 −0.0028, 2022 −0.0087, every component better.
     The flag-layer slopes it then fits (`diag-shrink-sums.txt`): plain `a` 0.79–0.80 →
     0.96–0.97, a100 0.74–0.76 → 0.88–0.91; real club sums stay exactly 11 at every horizon.
     2021–22 h0 calibration (`diag-shrink-cal.txt`): 0.95–1 band 0.958/0.989 → 0.961/0.963,
     0.9–0.95 0.921/0.951 → 0.924/0.935, 0.3–0.5 0.401/0.334 → 0.397/0.372. 0.88: flat h0,
     h1–5 +0.0034 (DISCARD); no 3rd value planned. This supersedes the 4th slope variant
     request (the slope is ~1 now); the user's h0 isotonic option stays open.
   - What's left at h0 (head walk 2021–22): MID p_start > 0.5 0.767 vs 0.788; first-choice
     GK 0.899 vs 0.935 (minutes model −1.5 pp, the layer another −1.5 pp; club GK sums are
     already ≈ 1, `diag-gk-layer.txt`, small n). **Live check at GW8**: does the regulars
     gap (0.77 vs 0.82) shrink once horizon_shrink is promoted? (Live is h0, where the
     shrink acts only through the flag layer's slopes.)
2. **Five subs: closed 2026-10-24** (model). Retest of the full feature on top of the shrink:
   2022 still +0.0025 (DISCARD); in the 60+ part only: 2022 +0.0013, minutes_ll −0.06 % (the
   −0.32 % gain was the sub part; DISCARD). Three variants: closed. Remaining fact: in
   2022/23 sub appearances are under-predicted (non-starters 0.111 vs 0.165,
   `diag-sub-season.txt`), so live (five subs) P(play) is likely low too.
   Older notes:
   - **Located 2026-10-23** (`diag-5subs-rows.txt`: per-row uncalibrated xP from the 2022
     walks via `fixture_points`, h0; MSE 3.6405 → 3.6427): the change is `pts_appearance`
     (+0.036 per row in the P(play) 0.5–0.8 band, other components < 0.005). The loss sits in
     the 0.1–0.8 bands (DEF/MID/FWD: rows whose xP rose and who then got 0 or 1–59 minutes,
     +375 total), against a gain on 60+ rows (−351). In that band v1 already over-predicts
     (xP 1.80 vs 1.70 realized) while 0.8–1 under-predicts (2.91 vs 3.09): the 2022 xP is
     compressed (idea 1, the flag layer's h0 slope), so more appearance probability lands
     where xP is too high. **Retest `five_subs` only after the h0 compression is fixed**
     (idea 1's 4th variant); it may then pass.
   - `five_subs` (2019/20 GW39+, 2022/23 on) as a minutes feature: minutes_ll_h0 −0.33 %
     (the biggest minutes gain of any change so far), |mean xP − points| 0.0110 → 0.0092,
     but MSE h0 +0.0005, all of it 2022/23 (+0.0029). In the 60+/sub parts only: the same
     (the gain is there, not in P(start)). Before another variant, find which rows' error
     grows in 2022/23 (by position and minutes band; appearance vs goal points): probably the
     points assembly reacting to the new P(60+). Diagnostic first, no variant.
   - Diagnostic 2026-10-21 (2022/23 walks with and without the feature,
     `~/fpl-research/results/research/diag-5subs.txt`): at h0 the feature mainly raises
     **P(play)**. Sub appearances are under-predicted in every band without it (MID p_start
     0.5–0.8: P(play) 0.827 vs 0.908 realized; with five_subs 0.864), and the Brier of P(play)
     improves 0.0841 → 0.0833 (P(60+) 0.0852 → 0.0850). Appearance-point bias −0.035 →
     −0.023, but appearance-point MSE 0.2451 → 0.2454. Deadlines 1–4 are identical (no
     five-sub rows in the fit until the first refit). The walk has no per-row xP, so the
     +0.003 points MSE in 2022 isn't located yet. Next: per-row xP from the walk's
     components (the bonus and goal terms scale with p_play) to find the rows.
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
6. **Clean sheet while on the pitch** (model; new 2026-10-21, two variants DISCARD, narrowly).
   - Mechanism: FPL credits a clean sheet to a 60+ player whose team concedes none *while he
     is on*, so a player subbed off before a late goal keeps it. v1 uses the full-match team
     clean sheet (the plan's documented approximation). Live GW1–7 (60+ minutes, single
     fixture): realized minus team P(CS) by position is GK +0.023 (never subbed), DEF +0.036,
     MID +0.082, FWD +0.073. The excess grows with how often the position is subbed.
   - Develop 2022/23 team level: e^−λ 0.273 vs realized 0.272, so the team model is
     calibrated and the gap is the player-level credit.
   - `cs-while-on` (P(CS)^(m_sixty/90)): MSE h0 −0.00033, h1–5 −0.00084, 2020–22 better,
     |mean xP − points| 0.0110 → 0.0004. `cs-while-on-bypos` (plus E[min | 60+] per
     position): MSE h0 −0.00047, h1–5 −0.00066, worst season +0.0001, bias 0.0061. Both
     fail only the minimum gain. **For the user**: a principled correction of a known
     approximation that is consistently small on develop. Live matters more than develop
     here (defcon era: DEF bias −0.134 per player-GW at h0). Don't run a third variant
     without a new mechanism; the user may want to judge it on its merits.
   - Team-level live P(CS) stays a watch item (GW to GW 0.13–0.40). Recheck at GW10.
   - Head walk 2021–22 (`diag-components.txt`), starters (p_start > 0.8): P(CS)·P(60+) vs
     realized CS·60+: GK 0.261/0.272, DEF 0.236/0.246, MID 0.235/0.260, FWD 0.219/0.255.
     The same subbing pattern on develop.
7. **The flag layer cuts sub appearances** (model; new 2026-10-24, one variant DISCARD on
   min gain). In 2021 the minutes model's P(sub) is calibrated before the layer (0.082 vs
   0.082 realized) and low after it (0.065). Players whose P(start) the flag raised lose
   sub appearances they make (0.153 before → 0.127 after vs 0.199 realized;
   `diag-sub-layer.txt`). `keep_sub` (a/a100 keep the unconditional P(sub)): MSE h0
   −0.00009, 2021 −0.0005, 2022 −0.0003, 2020 +0.0002; minutes_ll −0.15 %, bias 0.0100 →
   0.0080. **For the user**, like idea 6: right mechanism, small in points. Commit 68519f9 on
   the server hub (`try/keep-sub`). Don't retry without a new angle.
8. **Penalties and assists: looked at, no lead** (2026-10-24). Penalty goals swing with the
   league's season rate (2020: 102 realized vs 74 predicted) and the main-taker split flips
   sign between seasons (`diag-pens.txt`). Assists: FPL assists / xA by position (FWD 1.41,
   MID 1.20, DEF 1.13) as a share multiplier: DISCARD (4 of 6 seasons worse).

## Needs the user (not the loop's to do)

- **Two new keeps on `research`** (2026-10-24): `horizon_shrink=0.92` (8eaf71c) and
  `position_par` (c5a10d2). Each also breaks one synthetic test, not a by-design one:
  `test_flags_move_p_start_and_teams_stay_normalized` (a small synthetic pool sums to 11.67
  at a far horizon fixture; real clubs sum to exactly 11, `diag-shrink-sums.txt`) and
  `test_planted_shares_are_recovered` (the planted DEF has npxG without goals, the same
  artifact as finishing-12). Both tests need a decision at promotion (adjust the fixture or
  the assertion).
- **Flag-layer h0 slope, 4th variant** (2026-10-23, idea 1): **probably moot** after
  horizon_shrink (plain `a` slope now 0.96–0.97). The h0 isotonic option in the harness
  (`--parts p_start` instead of `p_start:1+`) is still yours.
- **Guard false positives** (2026-10-23, no harm): the guard refuses any command whose text
  contains `data/<file>` (a read in a diagnostic script; worked around by reading through
  `DataStore("data")`, the sanctioned interface) or mentions `calibrate_v1.py` in a grep
  without `--seasons`. Worth a look if the reads should be allowed.
- Still pending since 2026-10-21: merging `main` into `research` (below).

- **Merge `main` into `research`** (new 2026-10-21): PR #47 promoted back_split,
  club_continuity, season phase and finishing with a refit calibration table
  (064b07d) and harness fixes. `research` still has the stale table: the loop can't merge
  it (calibration_table.py is frozen). After the merge, the item below is resolved, and
  decision experiments (idea 5) can run again.
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

- **P(start) horizon shrink** (model, KEEP 2026-10-24, 8eaf71c): `MinutesParams.horizon_shrink
  = 0.92`, logit p_start × 0.92^h after the horizon decay; MSE h0 −0.00193, h1–5 −0.00234,
  2021 −0.0028, 2022 −0.0087, 2017–2020 identical; brier_start h0 −0.31 %, h1–5 −0.23 %. 0.88
  over-shrinks (DISCARD). Details in idea 1.
- **Finishing par per position** (model, KEEP 2026-10-24, c5a10d2): `SharesParams.position_par`,
  finishing shrinks toward the position's np goals / npxG (FWD ~0.89, DEF ~0.87, MID ~1.0;
  fitted walk-forward over 3 seasons) instead of 1. MSE h0 −0.00094, h1–5 −0.00097, all 6
  seasons better, goals_ll +0.05 %. Understat npxG over-states forwards' goals in every
  develop season from 2018/19.
- **Flag-layer slope per category** (model, KEEP 2026-10-23, 3dca529): `category_slopes`, one
  slope per flag category (prior N(1, 1)); MSE h0 −0.00088, h1–5 −0.00050, 2021 −0.0009,
  2022 −0.0047, 2017–2019 identical (no flags). Small: plain `a` alone still fits ~0.79,
  because the compression is the horizon (idea 1). Changes the calibration fingerprint
  (same 5 by-design test failures; regenerate the table at promotion).
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
