# v1r: promoting the research loop's keeps (design)

Date: 2026-11-02. Status: approved in brainstorming, not built.

## Goal

Port every unpromoted keep on the research loop's `research` branch (`fpl-autoresearch`) into the Rust xP model as one stack, test it once on data the loop never saw, and adopt it live if it passes. Points this season come first (PLAN §6.3 minutes model, §6.4 adjustment layer); this is the minutes/xP item of that order.

Not in scope: the general gain shrink (the 2026-11-01 diagnostic found no general optimizer's curse: ratio 0.98, 80% CI 0.94–1.02; the GW2–5 overconfidence goes to the early-season squad item), the crowd features `net_transfers` and `ownership` (blocked on `player_gw_ownership` and snapshot columns, a separate change), chip policy 2, the top-10k archive.

## What gets ported

The keeps on `research` at a pinned commit (recorded in the plan before porting starts), as listed in `autoresearch/ideas.md`:

| keep | commit | note |
|---|---|---|
| `gbm-400` | bee446a | MSE h0 −0.00539 on develop |
| `win_expectancy` | 1ff8914 | with `club_rating`, −0.00428 |
| `club_rating` | 6e09a1c | needs live `team_rating` coverage |
| `horizon_shrink=0.92` | 8eaf71c | breaks one synthetic test |
| `position_par` | c5a10d2 | breaks one synthetic test |
| `category_slopes` | 3dca529 | availability layer |
| `price_par`, `price_par_quadratic`, `price_prior_quadratic` | e81e161, 625a96f, dbb3265 | |

The two broken synthetic tests are resolved during the port: each test is either updated with a written reason (the test encoded the old behaviour) or the keep is dropped from the stack.

## Section 1: build

- **Model name.** The stack goes behind a new xP model name `v1r`; `v1` stays as it is. On adoption `v1r` becomes `v1` and the old model is kept as `v1-2026-10`.
- **Calibration.** The calibration table is refit for `v1r` with its own `V1Params` fingerprint, using the same expanding-window procedure and the same part as v1's table (`p_start`, horizons 1+ pooled, from 2016). Fitting is Python-only today (`dev/calibrate_v1.py`) and batch 8 deletes Python, so the plan ports the needed part to Rust (`fplopt-rs models calibrate walk|fit`). The port is gated on reproducing v1's current table.
- **Where.** Rust only (`crates/fplopt-models`, `crates/fplopt-features`), on top of `rust-migration`, started after the switch's gates pass (about GW13–14). Nothing is added to the Python package.
- **Parity.** The port is checked against Python `research` at the pinned commit with `fplopt-rs check parity table`, at batch 6's tolerances: fitted parameters within 1e-6, plan #1 identical (a tie within the 0.5% MIP gap counts as identical).
- **Leakage.** `fplopt-rs check leakage` with `model:v1r` must pass. The new Elo-style inputs (`win_expectancy`, `club_rating`) are the main risk.

## Section 2: pre-registered test

Fixed here and committed before any run. Logged to `results/experiments.csv`, family `v1r-promotion-2026-11`, one candidate (N=1, no deflation).

**Data.** Validate, 2023/24–2024/25: the loop tuned on develop only. 2026/27 so far is reported for information only, because the loop's weekly monitor read live results while choosing ideas. The 2025/26 holdout is not touched.

**Stage 1, components.** `fplopt-rs models eval --models v1,v1r --seasons 2023-2024`.
- Pass: MSE per player-GW lower for `v1r` at h0 and at h1–5, pooled over both seasons, and one-sided Diebold–Mariano p < 0.10 at h0.
- Also required: minutes log loss and P(start) calibration by band not worse, reported per season.

**Stage 2, decisions.** `fplopt-rs backtest compare --a optimizer:v1r:horizon=6,decay=0.75,max_hits=1,hit_margin=2 --b optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2 --seasons 2023-2024 --continuation own --starts template@1,random:5@1,random:3@20`.
- Pass (no-harm bar): per-decision mean (4-GW windows) ≥ 0 on realized points **and** ≥ 0 xG-scored.
- Why no-harm and not significance: two seasons can't detect a gain of the size the component metrics imply, and the project tunes on component metrics. Stage 2 catches a more accurate model that makes worse plans.
- Develop is also run and reported; it is in-sample for the loop and doesn't count toward the verdict.

**Verdict.**
- Both stages pass → adopt.
- Stage 1 fails → no change; the result goes back to the loop's ledger.
- Stage 1 passes, stage 2 fails → no change; find which keep hurt planning before any retry (a retry is a new candidate, N=2).

## Section 3: going live

After the verdict and the server switch to Rust:
1. Time `live prepare` with `v1r` on the server. `gbm-400` roughly doubles the minutes fit, so about 4 → 6–8 min; it must stay under the 15-minute tick.
2. Switch the bot's planner settings to `v1r` between two GWs (after one deadline, before the next alert). Rollback: point them back to `v1`.
3. Close #39 once the `a100_back` cell has been checked live.

## Risks

- The `research` branch lacks main's latest changes (c7869f2); the pinned commit's harness is what parity compares against, so this matters only if a keep touches code main changed since.
- `club_rating` needs `team_rating` for every live club; a gap fails `live prepare` rather than silently filling.
- The switch's gates slipping delays this; it does not start before them.
