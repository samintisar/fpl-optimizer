# Chip policy 2, 2026-11: no variant beats H

- **Spec:** `docs/superpowers/specs/2026-10-30-chip-policy-2-design.md` (amended 2026-11-03: run on the new v1)
- **Previous:** `docs/research/2026-10-chip-diagnostics.md`

Develop seasons only (2016/17–2022/23). Validate and the 2025/26 holdout were not read.

**Runs**
- All on the new v1 (v1r adopted, dd5c6ba), with `fplopt-rs` in the server's migration checkout `~/fpl-optimizer-rs/repo`.
  - Binary built at 4348dff; code identical to 9f3472d.
  - `nice -n 15`, 12 jobs.
- Starts: `template@1,random:3@1,random:3@20`, so 7 runs per season.
- Logged under family `chip-policy-2026-11`, 2026-11-04 07:14–11:43 UTC.
- Outputs on the server: `results/cp2-<arm>`. Comparison: `results/cp2-final/summary.json` (`dev/chip_policy_2.py`).
- The terminal-value tables (pooled, per window, per window and known double; leave-one-season-out) were re-estimated from S0 and committed (9f3472d) before any arm ran.

`BASE = horizon=6,decay=0.75,max_hits=1,hit_margin=2`:

| Arm | Spec |
|---|---|
| S0 | `chipgains:v1:BASE`: no chips, gains recorded |
| H | `chiprules:v1:BASE,t_tc=8.3,t_bb=10.9,k_fh=4,t_wc=12.87` |
| P | `optimizer:v1:BASE,chips=1,chip_shrink=1.0` (pooled table, in-sample; reference) |
| V1 / V2 / V3 | `optimizer:v1:BASE,chips=1,chip_shrink=k,chip_split=1,loso=1`, k = 0.6 / 0.8 / 1.0 |
| V4 | V2 + `tc_rule=8.3` |
| V5 | `…,chip_shrink=0.8,chip_split=2,loso=1` (information only) |
| V6 | V2 + `fh_scale=0.4` |

## 1. The test

Each arm vs H uses the full-run window t-test on 8-GW cells (35 cells), realized points, with the xG-scored mean alongside.
- Deflated mean = mean − SE·√(2 ln 5).
- Pass: deflated > 0, one-sided p < 0.10, and a positive xG-scored mean.

| Arm | Mean vs H (/GW) | SE | Deflated | One-sided p | xG-scored |
|---|---|---|---|---|---|
| P (reference) | +0.35 | 0.56 | −0.65 | 0.265 | +0.04 |
| V1 (k 0.6) | +0.22 | 0.67 | −0.98 | 0.374 | −0.42 |
| **V2 (k 0.8)** | +0.28 | 0.57 | −0.74 | 0.313 | +0.48 |
| V3 (k 1.0) | −0.39 | 0.61 | −1.48 | 0.735 | −0.04 |
| V4 (V2 + rule TC) | +0.02 | 0.57 | −1.00 | 0.485 | +0.50 |
| V5 (double-aware, info) | +0.16 | 0.55 | −0.82 | 0.388 | +0.42 |
| **V6 (V2 + FH × 0.4)** | **+0.50** | 0.50 | **−0.40** | **0.164** | **+0.79** |

- **k\*** = 0.8 (V2): the highest raw mean of V1–V3.
- **No variant passes.** Every deflated mean is negative, and the best p is 0.16.
- **Best by deflated mean:** V6 (−0.40), the variant to freeze for information under the spec.

## 2. Season means (points; means over seasons of the per-season mean over starts)

| Arm | From GW1, template | From GW1, mean | From GW1, xG mean | From GW20, mean | Transfer-gain slope | Run time |
|---|---|---|---|---|---|---|
| S0 | 2169.0 | 2208.0 | 2259.0 | 1152.1 | 0.907 | 36 min |
| H | 2314.7 | 2301.3 | 2421.6 | 1231.5 | 0.683 | 30 min |
| P | 2304.4 | 2314.3 | 2406.9 | 1235.4 | 0.707 | 35 min |
| V1 | 2260.3 | 2312.8 | 2411.4 | 1230.5 | 0.702 | 30 min |
| V2 | 2290.7 | 2309.9 | 2422.8 | 1240.6 | 0.686 | 33 min |
| V3 | 2295.4 | 2282.4 | 2404.9 | 1231.9 | 0.722 | 34 min |
| V4 | 2276.6 | 2296.0 | 2421.4 | 1239.3 | 0.676 | 30 min |
| V5 | 2283.1 | 2304.0 | 2417.2 | 1237.3 | 0.686 | 32 min |
| V6 | 2280.6 | 2322.9 | 2441.9 | 1231.1 | 0.695 | 43 min |

- **Every chip arm is about 75–115 points a season above S0** from GW1. Playing chips at all matters; how they are timed hardly does on develop.
- **P vs H per season, from GW1** (mean over the 4 starts): +22, −15, +58, +29, −14, +121, −110.

## 3. Reading

- **Split tables don't help on their own.** The diagnostics showed set-1 holding overvalued and set-2 undervalued by one pooled table. Splitting by window and estimating out of season (V1–V3) does no better than the pooled in-sample table P. The best shrink, 0.8, is only slightly ahead of 0.6, and full values (k 1.0) are the worst arm.
- **Rule TC (V4) costs the optimizer its edge.** The optimizer's Triple Captain timing is about as good as H's threshold.
- **Shrinking the Free Hit's predicted gain (V6) is the most promising change.** It gives the largest realized and xG-scored edge, consistent with the diagnostics' Free Hit forecast slope of 0.4. With N = 5 and 35 cells, though, it is not distinguishable from noise.
- **Double-aware values (V5) don't add anything,** even with doubles known in hindsight.
- **Known bias, both ways:** H's thresholds were tuned in-sample on develop under the old v1, while the variants' tables are out of season.

## 4. What changes

As pre-registered: no pass, so nothing changes.
- `/chips` keeps its "information only" label.
- The bot's plans don't play chips.

Under the spec, the best variant by deflated mean (V6) is frozen in git before GW20 for information, and logged next to H's call over GW20–38 in the live shadow. The spec lets the frozen variant drive chip advice in 2027/28 only if it passed on develop and then beats H in the shadow. V6 didn't pass, so the shadow is information only.

## 5. Freeze (2026-11-04, user decision)

V6 is frozen at tag `chip-policy-2-frozen`:

- **Backtested spec:** `optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2,chips=1,chip_shrink=0.8,chip_split=1,loso=1,fh_scale=0.4`.
- **Live spec (GW20–38 shadow):** the same without `loso=1`, i.e. `optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2,chips=1,chip_shrink=0.8,chip_split=1,fh_scale=0.4`.
  - It reads the all-seasons per-window table: `values` in `config/chips/terminal-values-window.json`, estimated on all seven develop seasons.
  - Live has no held-out season, so there's nothing to leave out.
- **Tables:** `config/chips/` as committed in 9f3472d, unchanged since.
- **Comparison arm:** H, `chiprules:v1:…,t_tc=8.3,t_bb=10.9,k_fh=4,t_wc=12.87`, as pre-registered.
- **Before GW20 (~2026-12-29):** the shadow logger (spec *Code* item 6, written in Rust) logs both calls per registered user and deadline. Advice doesn't change.
