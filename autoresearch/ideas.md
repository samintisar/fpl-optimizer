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
   - Tried 2026-10-25: `position_minutes` (E[minutes | start, 60+ / <60] per position, in
     e_minutes and m_sixty; GK e_minutes 82.3 vs 86.5 at p_start > 0.8): DISCARD, MSE h0
     +0.00007, 2 of 6 seasons better. The GK minutes gap costs nothing in points. Not a lead.
   - Live GW1–8 (report 2026-10-27-gw8; main still lacks horizon_shrink and
     category_slopes): regulars 0.776 vs 0.817, GW8 0.803 vs 0.812 (smallest gap yet);
     0.3–0.5 band 0.399 vs 0.322, 0.8–0.9 0.856 vs 0.921; GK p_start > 0.5 0.874 vs 0.944
     (n 161; develop first-choice GK 0.90 vs 0.93, a keeper-only fix is diluted in the
     pooled MSE like idea 10). Back from a miss with p_start > 0.3: 0.419 vs 0.400 (n 65),
     the over-prediction of GW6–7 is gone.
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
   - **GW8 (2026-10-27), first read of main's `p_defcon`** (`slice4.txt`, `slice5.txt` in
     `~/fpl-live/results/live/2026-10-27T0945Z`): DEF under-predicted, MID fine. DEF
     starters (p_start > 0.8) 0.206 vs 0.274; P(defcon | 60+) ≈ p_defcon / p_60 0.228 vs
     0.277 (n 642), realized ≥ predicted in every GW from GW2, band 0.3–0.4 0.344 vs 0.429,
     0.4–0.5 0.439 vs 0.540. MID 0.155 vs 0.137. DEF starters' pts_defcon 0.41 vs ~0.55
     realized: most of the DEF h0 bias (−0.151).
   - **The count model is calibrated on develop** (`diag-defcon-dev.txt`: 2016/17–2018/19,
     CBIT from the old stats, in-season rates from `defcon_rates` at every 2nd deadline,
     P(count ≥ 10 / 12) given the realized minutes): by band DEF 0.150 / 0.139, 0.249 /
     0.251, 0.348 / 0.361, 0.445 / 0.452; by rate quintile within ±0.04; implied NB r DEF
     12.7 (model 13.2), MID 18.6 (15.2); early season same as late. So the rate shrinkage,
     `DEFCON_K` and `DEFCON_R` are right **on old-definition counts**. The live gap is
     something live-specific: 2025+ Opta CBIT counts more dispersed (smaller r) or higher
     per 90 for the same players, the prior mean at GW1 (0.142 vs 0.224), or opponent
     strength (not in the model). Not testable by the judge (develop scores no defcon).
     **For the user**: a live dispersion check (implied r of 2026 DEF counts, the same
     calc as the develop diagnostic) decides between r and the rate; the loop can't read
     live counts outside `weekly.sh`.
4. **Bonus season weighting** (model; low value on develop).
   - BPS changed in 2026/27; the bonus fit weights 4 seasons by 0.7 per year. Develop can
     only test the weighting itself (current-season data weighted more), not the new rules,
     and `season_decay` is shared with saves and cards. `e_bonus` is now scored by the xP
     evaluation (243c39f): read it on develop first.
   - 2026-10-26: teammates competing (`team_goals` feature, the club's goals in the match):
     starters' residual at λ_for > 2 is negative everywhere (GK −0.10, DEF −0.02, MID −0.03,
     FWD −0.05; `diag-bonus-lambda.txt`); DISCARD on the min gain (MSE h0 −0.00018, 5 of 6
     seasons better, 2021 +0.0012). Premium MIDs (≥ 9.5) get 0.09 less bonus per start than
     predicted in 5 of 6 seasons (`diag-bonus-price.txt`). See idea 10 for the keeper term.
   - Player term, 2026-10-25: a player's bonus residual vs the position regression persists
     (corr 0.13–0.30 into the next season in every develop season, MID 0.32; best k ≈ 80
     pseudo matches, next-season residual SSE −7 %, `diag-bonus-player.txt`). As an offset on
     `b_play` (`bonus_player_prior=80`): DISCARD, MSE h0 +0.00001, 3–3 seasons. Real but
     ~0.03 points per played match after shrinkage: too small to see. Don't retry the
     offset; the season weighting is the only bonus item left, and it is low value.
   - 2026-11-05, the season weighting: `comp-decay-05` (`ComponentsParams.season_decay` 0.7
     → 0.5; bonus, saves, cards, defcon): DISCARD, MSE h0 +0.00003, 2–4 seasons, every
     component metric unchanged to 4 digits. Develop has no BPS drift to adapt to, so it
     can't show a gain. On live, the faster decay is harmless, so it's a judgment call for
     the new BPS rules (**for the user**, no more loop variants). The idea is closed for the
     loop.
   - **GW10 (2026-11-10), concentration**: live e_bonus is calibrated over all rows (FWD
     0.168 vs 0.171, DEF 0.084 vs 0.081) but under for predicted starters (p_start > 0.8:
     FWD 0.524 vs 0.697 n 99, GK 0.192 vs 0.280, MID 0.284 vs 0.313, DEF 0.188 vs 0.206).
     Bonus looks too evenly spread. Before any variant: the same cut on a develop head walk.
     If develop starters are calibrated, it's the new BPS (yours); if not, a mechanism to test.
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
   - **GW9 (2026-11-04)**: team P(CS) 0.245–0.250 for every position vs 60+ realized GK
     0.294 (n 180, ~1.5 SE), DEF 0.301, MID 0.354, FWD 0.355; above 0.25 in 7 of 9 GWs
     (GW9 0.367). GW9's top-10 misses: 9 of 10 clean sheets. GK h0 bias −0.081 → −0.100.
     Part level shift (keepers), part subbing (MID / FWD > GK). Develop team model is
     calibrated, so the level part isn't testable there.
   - **GW10 (2026-11-10)**: team P(CS) 0.245–0.255 vs 60+ realized GK 0.285 (n 200), DEF
     0.301, MID 0.346, FWD 0.369; GW10 0.299, above 0.25 in 8 of 10 GWs. Fourth week.
   - Team-level live P(CS) stays a watch item (GW to GW 0.13–0.43).
     GW8: team P(CS) 0.247 predicted vs keepers 60+ 0.288 (n 160, ~1.1 SE), GW8 alone
     0.433; DEF 0.297, MID 0.349, FWD 0.351 (the subbing pattern again).
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
   - 2026-10-26: penalties realized / expected by season 0.69, 1.10, 1.02, 1.18, 0.96, 0.81
     (2017–22), 458 vs 442 overall: noise around the right level (`diag-pens-walk.txt`).
     A season's over-prediction lowers λ_np (team goals minus penalties), so open-play
     goals move the other way. FPL assists by price: no trend (premium MID 0.90, FWD 0.89
     realized / expected; `diag-assists-price.txt`).
9. **Goalkeepers' P(start) at longer horizons** (model; new 2026-10-26, one variant DISCARD
   by the h0 rule). `horizon_shrink` (logit × 0.92^h) pulls every P(start) toward 0.5, but
   first-choice keepers are rarely rotated: develop 2017–22, p_start > 0.5, predicted 0.917
   → 0.796 from h0 to h5 against 0.928 → 0.857 realized; outfield within 0.01
   (`diag-horizon-components.txt`: keeper starters' xP under by +0.09 at h0 → +0.31 at h5,
   from starts, clean sheets and appearance). `keeper-shrink` (no shrink for keepers):
   **MSE h1–5 −0.00339, every season better (−0.0014 to −0.0043)**, the largest h1–5 gain
   in the ledger; h0 +0.00008 (unchanged before flags; 2022 +0.0007 through the flag
   layer's refit slopes). The judge needs an h0 gain, which an h ≥ 1 fix can't give.
   **For the user** (commit d018641 on the server hub). The planner's horizon is 6 GWs.
10. **Goalkeeper bonus: clean sheet *and* saves** (model; new 2026-10-26, DISCARD on the min
    gain, all 6 seasons better). Predicted keeper bonus is flat in λ_against (0.23–0.26),
    realized 0.17 for strong-team keepers, 0.34 mid, 0.26 weak (`diag-gk-saves.txt`). The
    additive indicators miss the conjunction: 2018–22 keepers 60+, CS without 3 saves 0.20
    realized vs 0.61 additive, CS with 3–5 saves 1.39 vs 1.02; the product cs × saves//3
    cuts the bonus SSE 20 % (`diag-gk-bonus-cells.txt`). `keeper-cs-saves`: MSE h0 −0.00024,
    h1–5 −0.00018, all 6 seasons better, worst −0.00001. Keepers are ~11 % of rows, so the
    pooled MSE dilutes it. **For the user** (commit 4098778). The defender analogue (CS ×
    goal / assist: realized bonus SSE −4.8 %) is flat in xP (DISCARD): the products'
    expectations are tiny.
11. **Goals: what's left after `price_par`** (model; 2026-10-26).
    - Before it (head walk 2021–22 h0, `diag-np-pen-ast.txt`): E[np goals] split MID 54.6 % /
      FWD 31.2 % / DEF 14.2 % against realized 57.1 / 29.3 / 13.6, short of even the
      par-implied split (56.1 / 30.0 / 13.9). Finishing tracks price (`diag-goals-price.txt`):
      `price_par` KEEP (see Done).
    - Tried and flat: FWD goal-share prior 960 instead of 1920 (`fwd-goal-prior`; the FWD
      starters' goals slope 1.17 is not shrinkage); `finishing_prior` 50 (third value: the
      setting is closed at 25); season-weighted finishing sums (`finishing-weighted`).
    - Still there with price_par (2017–22, `diag-horizon-components.txt`,
      `diag-goals-price-pp.txt`): FWD starters' goals over-predicted ~6 % at every horizon
      (−0.016 at h0, −0.021 to −0.026 goals per row at h1–5; 5 of 6 seasons), concentrated in
      mid-price starters (£5.0–8.0m: E[np] / npxG 0.86–0.89 vs realized 0.80–0.81; premium
      FWDs on target). A log-linear price par may be too flat for FWDs (fitted +2.5–4 % per
      £1m). Bench players (p_start ≤ 0.5) over-predicted 10–25 % (cheap bench MID 0.73,
      DEF 0.78): they convert below npxG, but they're rarely picked.
    - **2026-10-27: two keeps.** `price_par_quadratic` (z² in the finishing par's price
      regression; MSE h0 −0.00133, all 6 seasons) and `price_prior_quadratic` (z² in the
      goal / assist share priors; the MID prior was concave: ≤ £5.0m 0.69–0.89 of the
      log-linear prior, £5.5–8.0m 1.14–1.60, ≥ £9.5m 0.71–0.85, `diag-prior-bands.txt`;
      MSE h0 −0.00083, 5 of 6 seasons). Two variants of the price-curvature line.
    - The fitted finishing curves (`diag-price-curve.txt`) are not the convex FWD story
      alone: MID c −0.010 to −0.019 (caps premium mids), FWD +0.005 to +0.009 in 2020/22
      (−0.017 in 2018), **DEF unstable: 2018 c −0.065** (par 0.43 at £10.0m, 0.13 at
      £12.0m; 2022 −0.011). The gain is mostly taming the log-linear extrapolation at the
      price extremes. **Check at promotion**: the DEF par at live prices with the
      2023–25 fit (a premium DEF at z ≈ 2.5 gets ×0.67 if c is 2018-like).
    - After both (head walk d228f87, `diag-goals-price-pq.txt`): MID bands now 0.89–1.10
      (cheap bench 0.73 → 0.89, cheap starters 0.90 → 1.02). FWD still over. With FPL
      goals (`diag-goals-fpl.txt`; Understat `us_npg` is incomplete before 2019/20, so the
      earlier Understat-only FWD numbers are partly coverage): starters FWD 0.936, MID
      1.025, DEF 0.958, FWD < 1 in 5 of 6 seasons. 2020–22 (`diag-fwd-pens.txt`): it's
      open play (FWD 0.907, MID 1.058); penalties are fine.
    - **Located (`diag-fwd-finish.txt`, 2020–22 starters): FWD own finishing is
      over-trusted.** By last season's np goals − npxG: ≤ −3 1.05, −3 to −1 0.92, ±1
      0.86, +1 to +3 0.70, > +3 0.72 (17 / 31 / 34 / 7 / 6 FWDs); MIDs flatter (0.97 at
      +1 to +3, 1.37 at ≤ −3, 7 players). FWD finishing doesn't persist as much as k = 25
      assumes. `finishing_prior` is closed (12, 25, 50 tried globally, before the price
      pars); a per-position k (FWD stronger shrinkage) is the same setting: **for the
      user** to reopen or not. Small n in the high bands.
    - Bench-player minutes are not it: `sub-minutes` (E[minutes | sub app] by P(start))
      DISCARD on the min gain, idea 12.
    - DEF goal spread (2026-10-29): over-dispersed when pooled, small within a season;
      closed, idea 16.
    - **Still there at the gbm-400 head** (2026-11-02, walk 05d4b12 h0, `diag-regulars.txt`,
      `diag-fwd-starters.txt`): FWD starters at P(start) 0.7–0.85 lose 0.134 pts per row in
      goals (5 of 6 seasons), 0.85–0.95 −0.066. Premium FWDs at 0.95+ gain +0.128 (n 246).
      FWD starters' goals at their realized minutes are 0.94 of expected overall, 0.88 at
      0.7–0.85 (2017–22: 0.94 0.91 0.85 0.80 0.77 1.05). No clean price pattern within the
      band (£7.5–9.5m 0.84, ≥ £9.5m 0.82 at 0.7–0.85). It's the same over-trusted FWD
      finishing as above, and the fix (a per-position `finishing_prior`) is still the user's
      call.
12. **Minutes given the appearance** (model; new 2026-10-27, one variant DISCARD on the min
    gain). Develop P(start) is calibrated in every band (head walk d228f87 h0,
    `diag-low-minutes.txt`), but E[minutes] is over-predicted for low-p_start players at
    every position (FWD 0.1–0.3: 22.0 vs 17.8; `diag-fwd-minutes.txt`) because the
    conditional minutes are flat: E[minutes | sub app] is one constant 17.8 against 13.9
    realized at p_start ≤ 0.1 → 23.1 at 0.8+, and E[minutes | start] 81.9–86.5 against
    78.5–87.1. `sub-minutes` (m_sub linear in the row's P(start)): MSE h0 −0.00011, 5 of 6
    seasons better, components unchanged (commit 5b645bc on the server hub). The start
    half is smaller still. Don't retry without a points-level reason.
    - 2026-11-02, head walk 05d4b12 h0 (`diag-fwd-starters.txt`), the start half: MID / FWD
      starters play less than `m_start` at low P(start) (MID ≤ 0.5 82.7 vs 76.9, 0.5–0.7 84.4
      vs 80.2, 0.7–0.85 85.3 vs 82.8; FWD 82.9 vs 75.0, 84.5 vs 78.8, 85.5 vs 82.3). DEF and GK
      are within 1–2 minutes. It is not a points lead. MID goals per 90 at the *realized*
      minutes are 1.03–1.04 of expected in those bands, so the minutes excess cancels a rate
      deficit and MID goals are right in points. Fixing m_start would help FWDs and hurt MIDs.
      Not tested.
    - 2026-11-07: a FWD-only `m_start` slope in P(start) was considered and not run. FWDs
      are a minority of rows, and the sub half (all positions) gave only −0.00011 against the
      judge's MIN_GAIN of 0.0005. Expected to fail the min gain, and it would cost a variant.
13. **The crowd: ownership and deadline transfers** (model; new 2026-10-28, **blocked on
    the harness**, see *Needs the user*; re-checked 2026-10-29 after the harness hash
    changed: still the same KeyErrors). No model reads FPL managers' choices. Within a
    P(start) band they predict who starts, most in the rotation band:
    - Ownership (`player_gw_ownership`, GW t − 1's row at deadline t; 2017–22 head walk
      d228f87, `diag-ownership.txt`): P(start) 0.3–0.7 at h0, top quintile +4.2 pp over
      prediction (+0.12 pts), bottom −2.1 pp (−0.075); h1–5 the same. Offline, a logit
      adjustment fitted on 2017–19 and applied to 2020–22 (`diag-ownership-oos.txt`): MSE h0
      −0.0016 (2021 −0.0046), h1–5 −0.0025, Brier start h0 −0.13 %.
    - Deadline net transfers (`player_snapshot.transfers_in/out_event`, the newest snapshot
      ~3.6 h before the deadline; 2021–22, status `a`, `diag-snap-crowd.txt`): rotation band
      top quintile +9.7 pp (+0.47 pts), middle −8.3 pp (−0.23). Offline, fitted on 2021/22
      and applied to 2022/23 (`diag-snap-crowd-oos.txt`): MSE h0 −0.0036, Brier start h0
      −0.35 %; recalibration without it −0.0003. About −0.001 on the 6-season mean (only
      2021–22 have snapshots). This is the press-conference news the lags can't see.
    - Built and tested (commits on the server hub): `ownership` (f2fb2fa) and
      `net_transfers` (ad62831), each a minutes lag feature with an as-of test. Both fail in
      the pinned walk: its store has no `player_gw_ownership` table and loads
      `player_snapshot` without the transfer columns. Not a guard; the harness's input list.
    - Scoreable proxy, `price_change` (price − first price of the season; prices follow
      net transfers): DISCARD (MSE h0 −0.00028, 2020 +0.0031; Brier −0.01 %): too coarse.
    - 2026-11-04: rebased onto the head as b3a642b (`try/net-transfers-head`, history tests
      pass) and re-scored after the hash change to adeb6c64: same KeyError. 2026-11-10: again
      after 2c77d376 (a develop data refresh, baseline identical): same KeyError. Score b3a642b
      directly once the walk reads the columns; `ownership` needs the same rebase.
    - Next, once the harness reads them: score `net_transfers`, then `ownership` (snapshot
      `selected_by_percent` for 2021+, `player_gw_ownership` before), one at a time.
      Promotion check: live `player_gw_ownership` / snapshot transfer coverage for 2026/27,
      and the bot's snapshot timing (it plans 24 h before the deadline, when the GW's
      transfers are only partly in).
14. **Keeper saves points** (model; new 2026-10-28, too small for the judge). 60+ keepers,
    h0 (`diag-gk-rest.txt`): the saves rate is calibrated in the bulk but its spread is too
    wide (top band 4.54 predicted vs 4.18 realized), and realized floor(saves / 3) beats the
    Poisson expectation by ~0.045 per appearance in bands below 3.5 saves per 90 (saves
    over-dispersed, var / mean 1.1–1.3). Keepers starting are ~3 % of rows: ≤ 0.0001 of
    pooled MSE. Note for a per-position view, not a loop experiment.

15. **Match strength and big-club rotation** (model; new 2026-10-29, **two keeps**). The
    minutes model saw no opponent. Head walk d228f87 h0 (`diag-match-strength.txt`):
    regulars (P(start) > 0.7) started 3.7 pp under prediction when λ_for − λ_against > 1.6,
    in all 6 seasons (−1.8 to −4.9 pp), and +1 to +2 pp in even matches; rotation players
    +1.1 pp in the same lopsided matches.
    - `win_expectancy` (the club's Elo expected score in the fixture): KEEP, MSE h0 −0.00216,
      5 of 6 seasons. Its walk (`diag-we-walk`, 4092f44; `diag-match-strength-we.txt`)
      halves the big-favourite gap (−3.7 → −2.0 pp).
    - What it left (`diag-own-opp.txt`): top-6 clubs' regulars still −1.3 to −1.7 pp against
      mid and bottom sides, the other clubs' +0.4 to +2.4 pp: big clubs rotate (European
      fixtures, not in our data), whatever the opponent. `club_rating` (own Elo): KEEP,
      MSE h0 −0.00212, **all 6 seasons better**.
    - Forward congestion, `next_rest_days` (days to the next league fixture; big favourites
      with the next match ≤ 3.5 days away −5.1 pp, `diag-next-gap.txt`): DISCARD, seasons
      3–3, minutes log loss worse. The two Elo features and `rest_days` already carry it.
    - Two variants kept; a third Elo variable (the opponent's own rating) adds nothing the
      two don't imply (with home advantage). Next: re-slice on a walk at ba78638 (the big-club
      residual left, and 2022's regulars +3.5 pp in every band, `diag-match-strength-we.txt`,
      which is the five-subs season of idea 2). The real fix for the club effect is
      European / cup fixtures as a source (`main` has the ESPN adapter and congestion
      features since #51, which `research` lacks: see *Needs the user*).
    - **2026-10-30, after the main merge** (6439125: `europe_week`, `rest_days_all`,
      `days_to_next`, cup minutes alongside the two Elo features). The merge alone moved the
      baseline MSE h0 4.24287 → 4.22904. Head walk (`diag-europe-head.txt`): top-6 regulars
      vs mid / bottom sides −0.9 / −1.3 pp (from −1.3 / −1.7), so the club effect is smaller
      but not gone. Regulars at clubs without a UEFA match are +0.8 pp overall, but that is
      2022 (+3.3 pp, the five-subs season, idea 2) plus keepers (+1.6, idea 9) and MIDs
      (+1.4); other seasons are within ±0.75 (`diag-europe-head2.txt`).
    - `no-club-rating` (drop `club_rating` now that the real European data is in): DISCARD,
      MSE h0 +0.00128, 5 of 6 seasons worse. **Keep both Elo features at promotion**: they
      add beyond `europe_week`.
    - `europe-next` (a UEFA match within 4 days *after* the fixture; top-6 regulars −2.3 pp
      with it vs −0.3 pp with it before, 5 of 6 seasons): DISCARD, flat (+0.00004, 3–3).
      `days_to_next` with `europe_week` already carries the direction. The line is done unless
      a new source arrives (e.g. squad-rotation news, the crowd of idea 13).
16. **DEF goal spread: closed** (model; new 2026-10-29, one variant DISCARD). Starters'
    realized DEF goals on E[goals] have slope 0.68 at h0 / 0.66 at h1–5 (MID 0.95, FWD
    0.98), top share quintile 0.85 of expected, bottom 1.19 (`diag-def-spread.txt`).
    `goal_prior_by_position=((2, 3840),)`: MSE h0 −0.00019 (min gain), 4 of 6 seasons. The
    pooled slope is mostly between-season level: in-season slopes are 0.69–0.97, and even a
    perfect in-season rescale of DEF goal points is worth ≤ −0.0005 pooled in any season
    (`diag-def-ceiling.txt`). Not worth another value.

17. **MID xP band shape** (model; new 2026-10-30, diagnostic only). Head walk 6439125 h0
    (`diag-head-scan.txt`, `diag-mid-band.txt`): MID xP 2.5–3.5 under-predicted +0.139
    (z 4.2, 5 of 6 seasons) and 3.5–4.5 +0.129; ≥ 5.5 over-predicted −0.273 (5 of 6), 4.5–5.5
    −0.133. Components: the 2.5–4.5 gap is spread thin (appearance +0.03, goals +0.03–0.06,
    CS +0.03 (idea 6's subbing pattern), bonus +0.02; 2019 −0.03); the top band's is bonus
    −0.12 in all 6 seasons (idea 4's premium-MID finding again) plus goals −0.19 that swing
    by season (2020 −0.73, 2021 +0.32). Removable SSE of every MID cell together ≈ 4 bp of
    total (≤ ~0.002 MSE even if fully removed). No single-component fix; a bonus cap or
    interaction would repeat `bonus-team-goals` / `cs-attack-bonus`. Not a loop
    experiment without a new mechanism. Keepers at xP 2.5–3.5 +0.152 (6 of 6) are ideas 9–10.

18. **Minutes GBM capacity** (model; new 2026-10-31, **one keep**). `GbmParams()` (200
    rounds × 15 leaves, lr 0.05, 200 rows per leaf) was tuned in Phase 5 with about half of
    today's 32 `LAG_FEATURES`. `gbm-400` (400 rounds in all three parts, bee446a): **KEEP,
    MSE h0 −0.00539, h1–5 −0.00160, all 6 seasons better (2020 −0.0124)**, Brier start h0
    −0.35 %; the largest single keep in the ledger. `gbm-800`: DISCARD (+0.00206, minutes
    log loss +0.52 %, worst in 2017–19 where the fit has fewest seasons): rounds closed at
    400 (3 values). `gbm-ff08` (feature_fraction 0.8 on top): DISCARD (+0.00070, 2–4).
    Settings not re-checked: `num_leaves`, `min_data_in_leaf`, learning rate (one value
    each at most, and only after a fresh head walk shows where the minutes error is).
    Note: a sampling change swings single seasons by ±0.005, so judge GBM settings by how
    many seasons move, not by size alone.
    - Eval time 274 → 346 s; the live `live prepare` fit grows ~proportionally (minutes is
      ~11 s per cutoff of ~4 min).
    - **2026-11-01, head walk 05d4b12 vs 6439125 (200 rounds)** (`diag-minutes-400.txt`):
      each part's log loss on its own rows. Start: −0.38 % h0, −0.15 % h1+, better in all 6
      seasons. Sixty (P(60+ | start), starters only): +1.85 % h0 / +1.62 % h1+, worse in all
      6. Sub (P(sub | no start)): +0.36 % / +0.77 %, worse in all 6. Worst in 2017, when the fit
      has the least data. The gain is all in the start part; the two small parts overfit.
    - `gbm-parts` (sixty / sub back to 200, start 400): DISCARD. minutes_ll h0 −0.16 % as
      predicted, but MSE h0 +0.00026 (1–5). The sharper, worse-calibrated 60+ / sub
      probabilities score better in points. `start-leaves31` (31 leaves in the start part
      only): DISCARD, MSE h0 +0.00146, 2017–19 +0.0024 to +0.0039, Brier start +0.10 %. The
      start part is at capacity. **The GBM line is closed**: rounds (3 values), feature
      fraction, per-part rounds and leaves all tried. Learning rate and min_data_in_leaf stay
      untried; there's no mechanism for them.
    - Also in that walk: P(start) by horizon is level-biased high at long horizons in every
      season (h3+ 0.379 predicted vs 0.372 realized; h1–2 −0.15 to −0.34 pp, h0 ≈ 0). Recal
      slopes are ~1 (h0 1.02, h1–2 0.99, h3+ 1.03). The likely cause is pool churn
      (players who leave the club keep their horizon rows). It's h ≥ 1 only, so the judge's
      h0 rule can't see a fix. Keepers at h3+ still show idea 9's pattern after gbm-400
      (bands 0.1–0.5 −9 to −13 pp, 0.5–0.8 +11 to +13 pp, 6 of 6 seasons).
19. **Relative price and end of season: looked at 2026-10-31.** `price-rank` (rank by price
    among same-position club-mates, a minutes lag feature; the most expensive at a
    club-position starts +2.1 pp at P(start) 0.7–0.9, z 4.0, `diag-club-price.txt`):
    DISCARD, flat (+0.00013, 3–3); `price` and `comp_rank` already carry it. Dead rubbers
    (`diag-season-end.txt`: GW31+, clubs safe and out of the European race by a crude
    table): regulars −3.5 pp (z −2.7, 5 of 6 seasons), rotation +3.5 pp, but 2.4 % of rows
    (≈ 0.0001 MSE): not a loop experiment.

Looked at 2026-10-28, no lead: the team model (λ_against calibrated within 2 % at every
horizon, P(CS) 0.279 vs 0.281, slope 0.92–0.95, `diag-team-lambda.txt`); P(60+ | start)
(within 0.01 at h0; `diag-c60.txt`); P(start) by price (premium outfield ±0.02 by season,
keepers as idea 9; `diag-start-price.txt`); keeper P(start) sharpening within the club
(best γ 1.25: h0 −0.00028, h1–5 −0.0029, the idea 9 story; `diag-gk-sharpen.txt`); FWD
finishing × price (season noise dominates, 2021 0.72–0.76 in every band;
`diag-finish-price-over.txt`).

## Needs the user (not the loop's to do)

- **2026-11-10 (GW10 report)**: `last_checked_gw` 10, monitor done (`reports/2026-11-10-gw10.md`).
  The eval hash changed (adeb6c64 → 2c77d376) after `weekly.sh` refreshed the develop data;
  baseline `c04be445fd-2c77d376` identical (MSE h0 4.223654). Re-scored `net_transfers`
  (b3a642b): same KeyError, so it was the data, not the harness (ledger `crowd-recheck3`,
  ERROR). No variants spent. Ledger: 54 scored (14 KEEP, 40 DISCARD), 5 ERROR. Live GW10:
  regulars' P(start) gap widest since GW3 (0.797 vs 0.872), team P(CS) under in 8 of 10 GWs,
  DEF defcon under, starters' bonus under (FWD 0.524 vs 0.697) while all-rows bonus is
  calibrated. After the Rust switch (main checkout 14f1fc8), check which model `weekly.sh`
  evaluates and whether the loop's harness should move to `fplopt-rs`.

- **2026-11-09: no experiments tonight.** `last_checked_gw` is still 9 (bootstrap
  2026-11-09T023113Z; status still reports next GW 10 with the 2026-11-06 deadline, so
  FPL hasn't marked GW10 checked yet). No monitor; the next run should do the GW10 report.
  `sync.sh raw` copied 33 new files. The harness hash is still adeb6c64 (head baseline
  `039552b60b-adeb6c64`), so the crowd (idea 13, b3a642b) is still unscoreable, and the
  calibration table still blocks `optimizer:v1` decision runs (idea 5). No open line has an
  untried variant with a mechanism behind it (idea 18's learning rate / min_data_in_leaf
  have none), so I spent no variants. The ledger is unchanged: 54 scored (14 KEEP, 40
  DISCARD), 4 ERROR. Fifth night idle: the loop needs a harness or table change to resume.

- **2026-11-08: no experiments tonight.** `last_checked_gw` is still 9 (bootstrap
  2026-11-08T023213Z; GW10's matches not yet checked by FPL), so there was no monitor; the
  next run should do the GW10 report. `sync.sh raw` copied 34 new files. The harness hash is
  still adeb6c64 (head baseline `41ac913037-adeb6c64`), so the crowd (idea 13, b3a642b) is
  still unscoreable and the calibration table still blocks `optimizer:v1` decision runs (idea
  5). No open line has an untried variant with a mechanism behind it, so I spent no
  variants. The ledger is unchanged: 54 scored (14 KEEP, 40 DISCARD), 4 ERROR. Fourth night
  in a row without work: the loop is idle until one of the blockers is lifted.

- **2026-11-07: no experiments tonight.** `last_checked_gw` is still 9 (bootstrap
  2026-11-07T023030Z; GW10's deadline passed 2026-11-06 18:30Z, but its matches aren't
  checked yet), so there was no monitor; the next run should do the GW10 report. `sync.sh raw`
  copied 42 new files. The harness hash is still adeb6c64 (head baseline
  `9fea614165-adeb6c64`), so the crowd (idea 13, b3a642b) is still unscoreable, and the
  calibration table still blocks `optimizer:v1` decision runs (idea 5). The guard refuses
  reading `~/fpl-harness` beyond its four entry points, so the hash is the only check on the
  crowd columns. I spent no variants (idea 12's FWD start half is below MIN_GAIN, see there).
  The ledger is unchanged: 54 scored (14 KEEP, 40 DISCARD), 4 ERROR.

- **2026-11-06: no experiments tonight.** `last_checked_gw` is still 9 (bootstrap
  2026-11-06T023113Z; the GW10 deadline is 18:30Z today), so there was no monitor. The next
  run should do the GW10 report. `sync.sh raw` copied 42 new files. The harness hash is
  still adeb6c64 (new head baseline `4b343e96c5-adeb6c64`), so the crowd (idea 13,
  b3a642b) is still unscoreable. The calibration table still blocks `optimizer:v1` decision
  runs (idea 5). No open model line has an untried variant with a mechanism behind it, so
  I spent no variants. The ledger is unchanged: 54 scored (14 KEEP, 40 DISCARD), 4 ERROR.

- **2026-11-05**: `last_checked_gw` still 9 (GW10 deadline 2026-11-06 18:30Z), so no
  monitor; the next report is GW10. Harness hash unchanged (adeb6c64; baseline
  `043aa3bc2a-adeb6c64`), so the crowd (idea 13, b3a642b) is still unscoreable, and the
  calibration table still blocks decision runs. One variant tonight: `comp-decay-05` (idea
  4), DISCARD, flat. Every open model line is now closed for the loop or waiting on you.
  Until the harness reads the crowd columns or the table is regenerated on `research`, the
  nightly run can only do the weekly monitor.

- **2026-11-04 (GW9 report)**: the harness hash changed (b7e2afde → adeb6c64; baseline
  metrics identical), so `net_transfers` was cherry-picked onto the head (b3a642b on the
  server hub, `try/net-transfers-head`) and re-scored: same KeyError (no
  `transfers_in_event` / `transfers_out_event` in the walk's `player_snapshot`). Ledger
  `crowd-recheck2`, ERROR. Nothing else run: model lines closed, decisions blocked by the
  calibration table. GW9 live: clean sheets under-predicted for the 3rd week (team P(CS)
  0.245 vs keepers 60+ 0.294, 7 of 9 GWs above), DEF defcon still under (0.230 vs 0.279 |
  60+), P(start) regulars gap back to 5 pp in GW9; see the report.

- **2026-11-03: no experiments tonight, again.** `last_checked_gw` is still 8 (GW9's
  deadline was 2026-10-31; FPL hasn't marked it checked as of the 02:32Z bootstrap), so no
  monitor; the next run should do GW9. Both blockers re-checked and unchanged: the harness
  hash is still b7e2afde (baseline `4200b4b7a0-b7e2afde` reused), so the crowd (idea 13)
  stays blocked; `test_the_calibration_table_matches_the_default_settings` still fails at
  the head (the table predates gbm-400 and the Elo features' defaults), so `optimizer:v1`
  decision runs (idea 5) can't run. With every model line closed under the 3-value rule or
  waiting on you (ideas 6, 7, 9, 10, 11), I didn't spend variants on settings without a
  mechanism. Unblocking either the crowd or the table is what would let the loop do work.

- **2026-11-02: no experiments tonight.** GW9 wasn't checked yet (`last_checked_gw` 8), so
  there was no monitor. Three new slices of the gbm-400 head walk (05d4b12, h0, 2017–22)
  found nothing testable with a mechanism:
  - `diag-newcomers.txt`: promoted clubs, new to the club, new to the league and season
    phase are all within ±0.08 pts, and the largest cell is < 0.25 bp of SSE.
  - `diag-regulars.txt`: regulars (P(start) > 0.7) are under-predicted +0.03 to +0.05 pts,
    and the largest piece is the clean sheet (+0.02 to +0.035; DEF +0.05 to +0.06). That's
    idea 6's CS-while-on, tried twice. Starting keepers are +0.16 (ideas 9–10).
  - `diag-fwd-starters.txt`: FWD rotation starters' goals (idea 11, finishing) and the start
    minutes (idea 12).

  Every cell left is ≤ ~3 bp of total SSE (≤ 0.001 MSE if fully removed). The remaining
  gains need new information in the minutes model: the crowd in the harness (idea 13).
  The other open items are still yours: the calibration table for decision experiments,
  the per-position finishing prior (idea 11), and the h1–5 / keeper path (ideas 9, 10).

- **2026-11-01: the loop is out of testable model lines.** Two variants tonight (GBM
  per-part rounds, start-part leaves; both DISCARD), and the GBM line is now closed (idea 18).
  What's left is the user's: the crowd inputs in the pinned harness (idea 13; harness hash
  still b7e2afde); a calibration table on `research` (decision experiments); and an h1–5
  path in the judge, or a per-position view, for the h ≥ 1 / keeper fixes (ideas 9, 10).
  The keeper h3+ miscalibration is still the largest clean signal left in the minutes model
  (6 of 6 seasons). Until one of those changes, the nightly run is mostly the weekly monitor.

- **New keep on `research`** (2026-10-31, idea 18): `gbm-400` (bee446a), 400 boosting
  rounds in each minutes part. MSE h0 −0.00539, all 6 develop seasons. Promotion checks:
  (1) the Rust port's LightGBM golden (`tests/fixtures/rust_golden/lgbm.json.gz`) and the
  calibration table both change with it; (2) `live prepare` time on the server (minutes fit
  roughly doubles); (3) `check leakage` is unaffected (no new input). Also: `main` has moved
  on (c7869f2: chip policy 2, player-props archive) and `research` lacks it.

- **2026-10-30**: after the main merge, the model ideas the loop can test are used up. Two
  tried tonight (both DISCARD), and I stopped there rather than spend variants on knobs. What
  would open new lines: (1) the crowd inputs in the pinned harness (idea 13; still not in
  `dev/calibrate_v1.py`'s `TABLES` / `SNAPSHOT_COLUMNS` on `research`); (2) a calibration
  table regenerated on `research` (decision experiments, idea 5 FT value 2.0, are still
  blocked: `research` runs `main`'s table with other defaults); (3) your call on the
  for-the-user DISCARDs (ideas 6, 7, 9, 10). Also: `club_rating` and `win_expectancy`
  both still pay their way on top of the ESPN features (idea 15), so promote both.

- **Two new keeps on `research`** (2026-10-29, idea 15): `win_expectancy` (1ff8914) and
  `club_rating` (6e09a1c), Elo match strength and the club's own Elo as minutes lag
  features. Together MSE h0 −0.00428 (4.24715 → 4.24287), h1–5 −0.00225, 2017–22 all
  better. Promotion checks: (1) `fplopt check leakage` on the two features (they read
  `team_rating` per fixture: the training rows take the match's own pre-match rating, the
  horizon fixtures the deadline rating); (2) live `team_rating` coverage for 2026/27 (built
  from football-data results: is the current season's Elo up to date at each deadline?);
  (3) it overlaps with `main`'s ESPN congestion features (#51): compare against them, or
  merge `main` into `research` first so the loop tests on top of them.
- **Crowd still blocked** (2026-10-29): the harness hash changed (62719625 → b7e2afde,
  same baseline metrics), but the walk still lacks `player_gw_ownership` and the snapshot
  transfer columns. Both re-runs failed with the same KeyErrors (ledger `crowd-recheck`).

- **The harness can't see the crowd** (2026-10-28, idea 13; the night's main finding). Two
  input-list changes in the pinned `dev/calibrate_v1.py` (yours): add
  `"player_gw_ownership"` to `TABLES`, and `transfers_in_event`, `transfers_out_event`,
  `selected_by_percent` to `SNAPSHOT_COLUMNS`. Then the loop scores `net_transfers`
  (ad62831) and `ownership` (f2fb2fa). Offline they're the largest model gains since
  finishing (h0 −0.0016 and −0.0036 on their seasons).

- **Two new keeps on `research`** (2026-10-27): `price_par_quadratic` (625a96f) and
  `price_prior_quadratic` (dbb3265), the price curvature of the finishing par and of the
  goal / assist share priors. Same 7 head test failures, 3 new tests pass. Promotion
  check: the DEF finishing curve at live prices (idea 11).
- **FWD finishing over-trusted** (2026-10-27, idea 11): reopen `finishing_prior` as a
  per-position setting (FWD stronger)? Evidence is clear in direction, thin in players.
- **Live DEF defcon −0.05 per 60+ appearance** (2026-10-27, idea 3): the model is right
  on develop counts, so it's the 2025+ counts. A dispersion check on live counts is yours.
- **Guard** (2026-10-27, no harm): `judge.py compare` followed by a `python -c` in the
  same ssh command is refused as touching the frozen judge; the plain compare works.
  `dev/calibrate_v1.py walk --seasons 2016-2022` (research copy) runs; the pinned
  harness's copy is refused, as designed.

- **New keep on `research`** (2026-10-26): `price_par` (e81e161), finishing par by position
  and price. Same 7 test failures as the head (no new one).
- **Two judge-DISCARDs worth your look** (2026-10-26): `keeper-shrink` (idea 9: h1–5
  −0.0034, every season; the h0 rule can't see an h ≥ 1 fix) and `keeper-cs-saves`
  (idea 10: all 6 seasons better, below the min gain because keepers are 11 % of rows).
  Maybe the judge needs an h1–5 path, or a per-position view.
- **Process race** (2026-10-26, no harm done): `sync.sh try` checks the candidate out in
  `~/fpl-research` immediately, even while a `model_eval.sh` baseline for the previous head
  is still walking there. I stopped that run and deleted its partial dir
  (`84e5a968fb-62719625`). The loop now waits for model_eval before `sync.sh try`; a lock
  check in sync.sh (frozen) would make it safe.

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
- Still pending since 2026-10-21: merging `main` into `research` (below). As of 2026-10-25
  `main` also has #46 (measure-bonus-defcon) and #48 (bot fixes) that `research` lacks.
- **The model ideas are thin** (2026-10-25): every open model line is closed under the
  3-value rule, waiting on you (ideas 6, 7) or blocked (idea 5, the calibration table). The
  next real leads will come from the GW8+ monitor (does live P(start) close after the
  promotion; the CS/defcon bias, now that main's frames carry p_defcon).
- Guard false positive, 2026-10-25: one Bash command that appends to `ledger.tsv` (`>>`)
  and also runs `bash autoresearch/sync.sh` is refused as a write to the frozen sync.sh.
  Separate commands work.

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

- **Club rating** (model, KEEP 2026-10-29, 6e09a1c): `club_rating` in `LAG_FEATURES`, the
  club's own Elo before the fixture. MSE h0 −0.00212, h1–5 −0.00126, all 6 seasons better,
  Brier start h0 −0.13 %. Big clubs rotate whatever the opponent (idea 15).
- **Win expectancy** (model, KEEP 2026-10-29, 1ff8914): `win_expectancy` in `LAG_FEATURES`,
  the club's Elo expected score in the fixture (home advantage 60). MSE h0 −0.00216, h1–5
  −0.00099, 5 of 6 seasons better (2020 +0.0008), Brier start h0 −0.11 %.

- **Share prior curvature** (model, KEEP 2026-10-27, dbb3265):
  `SharesParams.price_prior_quadratic`, a ridged z² term in the goal and assist share price
  priors (`SharesFit.price_goal_curve`, `price_assist_curve`). MSE h0 −0.00083, h1–5
  −0.00022, 5 of 6 seasons better (2018 +0.0003). Evidence: the MID prior is concave in
  price in every fit season (`diag-prior-bands.txt`).
- **Finishing par curvature** (model, KEEP 2026-10-27, 625a96f):
  `SharesParams.price_par_quadratic`, a ridged z² term in the finishing par's price
  regression (`SharesFit.price_finishing_curve`). MSE h0 −0.00133, h1–5 −0.00054, all 6
  seasons better, goals_ll +0.03 %. See idea 11 for the fitted curves (DEF unstable).
- **Finishing par by price** (model, KEEP 2026-10-26, e81e161): `SharesParams.price_par`,
  the finishing par per position and price, quasi-Poisson np goals ~ npxG·exp(a + b·z) over
  the position par's own-npxG player-seasons (`SharesFit.price_finishing`; fitted per +£1m:
  MID ×1.02–1.03, FWD ×1.03–1.04, DEF ×1.06–1.13 at GW10 of 2018/2020/2022). MSE h0
  −0.00094, h1–5 −0.00140, 5 of 6 seasons better (2020 +0.0003), goals_ll +0.06 %. Evidence:
  cheap players finish below npxG (bench MID ≤ 5.0 0.74), premium ones above (MID ≥ 9.5
  1.10, FWD 1.06).

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
