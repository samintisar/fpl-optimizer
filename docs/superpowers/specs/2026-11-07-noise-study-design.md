# Noise study: comparisons that can see 10 points a season

**Date:** 2026-11-07. **Status:** approved design.

## Why

Small planner changes are now worth 5–15 points a season, and our comparisons can't see them:

- Stage C's develop runs (2022/23, 8 starts, whole-season paired differences) had standard errors of 14–27 points a season (`docs/superpowers/specs/2026-11-06-price-planner-design.md`, Stage C addendum).
- The two versions picked different transfers in ~80% of GWs. Each different pick adds its own luck, and over a season the squads drift apart, so the luck adds up.

The next experiments (the price planner re-judged, early transfers, a "move tonight" alert) need comparisons that can see about 10 points a season.

## Goal and success

- **Target:** a comparison method whose standard error of the A − B difference is **≤ 4.5 points a season**. A 10-point effect then clears one-sided p < 0.10 with ~80% power: (1.28 + 0.84) × 4.5 ≈ 9.5.
- **Honest errors:** on placebo pairs (true difference 0), the method's standardized differences z = mean / SE must have a root mean square (about 0, so a bias counts too) of no more than 1.25 over the 10 pairs. Ten pairs give a rough check: the RMS of 10 draws from N(0, 1) is below 1.25 about 89% of the time. The sd of z is reported as a diagnostic. (Amended 2026-11-07 before any run: the first version used the sd, which hides a bias.)
- **Cheapest wins:** among the methods that pass both, the one with the least server time per comparison.
- **If none passes:** the note reports the best standard error reached, and we decide on approach B (new variance-reduction machinery) then.

## Comparisons

All comparisons use the live planner L = `optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2`. Starts S = `template@1,random:3@1,template@10,random:3@10,template@20,random:3@20,template@29,random:3@29` (16 per season).

| # | A | B | Seasons | Why |
|---|---|---|---|---|
| C1 | L + `,prices=forecast,price_k=0.5` | L | 2022 | the price planner (Stage C) |
| C2 | L with `hit_margin=3` | L | 2021, 2022 | a typical small tweak; the seasons an early-move test would use |
| P1–P10 | L + `,xp_jitter=0.05,jitter_seed=2s−1` | L + `,xp_jitter=0.05,jitter_seed=2s` | 2022 | placebo: true difference 0 |

Develop seasons only. The 2025/26 holdout is never read, and 2023/24–2024/25 stay unused (they remain clean for later checks).

## Methods

Every comparison is run with `fplopt-rs backtest compare --continuation own --reference b`, once per window length k ∈ {1, 2, 4} (`--k k`, stride = k). Each run gives the whole-season paired differences (identical across the three runs) and per-decision differences for its k. Scoring: realized net points and xG-scored net points (both already in compare's outputs). That gives 8 methods per comparison:

- whole season × {realized, xG};
- per decision k = 1, 2, 4 × {realized, xG}.

**On one scale.** Every method is reported as points per season:

- **Whole season:**
  - the per-GW differences are averaged over the starts live in each GW, then over non-overlapping 4-GW blocks within each season (as PLAN §7's gate statistic does);
  - mean and SE are taken over the blocks and scaled by 38.
  - Starts begin at GW 1, 10, 20 and 29, so a block averages squads of different ages. That makes the block means uneven, so this method's SE is conservative and its mean is not exactly a full-season effect.
- **Per decision (k):**
  - each decision window's difference is averaged over the starts live in it;
  - the windows of all seasons are pooled;
  - mean and SE are taken over the windows, then scaled by the windows per season (38, 19 and 10 for k = 1, 2, 4; for k = 4 the last window covers GW37–38).

The SE is the plain SE over windows, which treats windows as independent. The placebo check is what tests that assumption.

- **Server time:** the wall time of each compare run (logged by the driver), and per method the time of the run it needs.

## Build

1. **Rust, `xp_jitter` and `jitter_seed` spec keys** (`crates/fplopt-backtest/src/specs.rs`, applied where the policy reads its xP frame):
   - each player's xP at each deadline is multiplied by (1 + σ·ε);
   - ε ~ N(0, 1) is drawn from a hash of (`jitter_seed`, `player_key`, `deadline`, horizon GW), so it is deterministic and independent of thread order;
   - the same ε is used for every xP column of that player and GW.
   - The default is 0, which leaves the planner unchanged (pin it with a fingerprint test).
   - The label gets a suffix `jit=σ/seed`.
   - Tests cover determinism, different seeds giving different ε, a mean of ε near 0 over many keys, and σ = 0 leaving the frame unchanged.
2. **`dev/noise_study.py`:**
   - **`run`** writes and launches the compare commands for C1, C2 and P1–P10 into an out directory, logging each command's wall time. Family `noise study`. It runs on the server in a separate worktree.
   - **`report`** reads each compare's `paired.parquet`, `per_decision.parquet` and `summary.json`. It writes `noise.json` and prints, per method:
     - the C1 and C2 mean and SE per season;
     - the placebo z sd;
     - server time;
     - pass/fail;
     - the chosen method.
   - Pure functions are tested in `tests/test_noise_study.py` (scaling, z sd, the choice rule, reading real compare outputs from a tiny local run).
3. **Docs:**
   - `docs/research/2026-11-noise-study.md` (results and the chosen method);
   - a PLAN §12 row;
   - the chosen method written into PLAN (§7, where the evaluation design lives) and CLAUDE.md as the standard comparison for new experiments.

## Not in this spec

- Re-judging the price planner on 2022/23 with the chosen method (descriptive; Gate C stays unrun).
- The early-move experiment: plan at the previous deadline with data as of then, at that moment's prices, optionally waiting if flagged.
- The "move tonight" alert in the bot.

## Risks

- **Pooled windows understate the error.** If one season's windows are correlated, pooling makes the SE too small. The placebo check catches gross cases only (10 pairs). Mitigation: also report the season-clustered SE for C2 (2 seasons) as a diagnostic.
- **Per-decision differences measure one decision's effect with each arm continuing its own plan.** Summing them over a season is not exactly the whole-season effect when effects compound. The note states this, and the whole-season method stays reported next to the per-decision ones.
- **xG scoring needs xG data.** 2021 and 2022 have it.
