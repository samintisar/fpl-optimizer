# dev/

Developer tools that are not part of the `fplopt` package.

## `reference_check.py`: differential check against open-fpl-solver

Phase 4 plan, Task 4 (PLAN §7 *Reference check*). It solves the same no-chip planning problems
with our MILP (`fplopt.optimize.model.solve_plan`) and with
[open-fpl-solver](https://github.com/solioanalytics/open-fpl-solver) (Apache-2.0), then compares
the results.

### Setup

open-fpl-solver is never vendored. Clone it outside this repo and give it its own venv
(it needs Python ≥ 3.14; `uv` installs that):

```sh
git clone https://github.com/solioanalytics/open-fpl-solver <somewhere>/open-fpl-solver
cd <somewhere>/open-fpl-solver && uv sync
export OPEN_FPL_SOLVER_DIR=<somewhere>/open-fpl-solver   # PowerShell: $env:OPEN_FPL_SOLVER_DIR=...
```

The check also reads the built `data/` tables (read-only), from `--data-dir`, `FPLOPT_DATA_DIR` or
the repo's `data/`.

### Run

```sh
uv run python dev/reference_check.py [--data-dir data] [--gap 0] [--quick] [--out table.md]
```

Both solvers use the same `mip_rel_gap` (`--gap`, default 0:
proven optima; the 24 instances take about 10 minutes). `--quick` runs 3 small instances. The script
exits 1 on any mismatch. The pytest version, `tests/test_optimize_reference.py`, runs the quick
instances and is skipped unless `OPEN_FPL_SOLVER_DIR` is set and `data/` exists.

### How open-fpl-solver is driven

Its data layer (projection CSV, FPL API by team id) is bypassed. The script runs itself in
driver mode under the clone's venv (`uv run --project $OPEN_FPL_SOLVER_DIR`, one subprocess for
all instances). There it builds the `data` dict that `prep_data` would return and calls
`dev.solver.solve_multi_period_fpl(data, options)` directly. Its code is not modified. The
driver only subclasses `highspy.Highs` to read the final MIP gap, and stdout is discarded.

The mapping (the module docstring has the details):

| ours | open-fpl-solver |
|---|---|
| candidates `PlanInput.players` (pruned, or the full pool) | `merged_data`, the same players |
| xP per horizon GW | `{w}_Pts`, with w = gw_index + horizon offset |
| prices, selling prices, bank (tenths of £m) | `buy_price`, `sell_price`, `itb` in tenths (exact integers); `itb_value / 10` |
| owned player with selling price ≠ price | `price_modified_players` (first sale at `sell_price`) |
| owned player who left the game (not buyable) | buy price 1e6: first sale at `sell_price`, can't be bought back |
| `free_transfers` | `ft`, with `ft_base` 1 |
| `ft_value` {2: 2.0, 3: 1.6, 4: 1.3, 5: 1.1} | `ft_value_list` {"2".."5"}; states 0 and 1 get their scalar `ft_value` 1.5 |
| GW1: unlimited free transfers, then 1 FT | `use_wc = [GW]` (free transfers, then `fts` = `ft_base` = 1) |
| decay, bench weights, `hit_cost + hit_margin` | `decay_base`, `bench_weights`, `hit_cost` |
| vice set after the solve; no FT-use penalty | `vcap_weight` 0, `ft_use_penalty` and `itb_loss_per_transfer` off |
| no chips | `chip_limits` all 0 |

**Constant offset.** Their FT states run 0..5 with V(s) = Σ_{n=0..s} v[n]. Ours run 1..5 with
V(s) = Σ_{1≤n≤s} ft_value[n]. Both credit the gain V(ft[t]) − V(ft[t−1]) inside the decay, with
V(ft[−1]) = 0. With the mapping above, V_theirs(s) − V_ours(s) = c = 3.0 for every reachable
state s ≥ 1. The objectives therefore differ by the constant
Σ_t decay^h_t (c_t − c_{t−1}) = 3.0. It is computed per instance (column `offset`) and
subtracted from their score.

**Not comparable** (detected and reported, not compared):
- a club over the team limit at the deadline. Theirs tolerates the excess only in a GW with no
  transfers at all; ours tolerates it as long as nobody from that club is bought;
- FT top-ups (none in our rules);
- a horizon that is not contiguous in gw_index.

### What is compared

For each instance:
- the objective (ours vs theirs − offset). They agree when the difference is within the sum of
  the two final relative MIP gaps × |objective|, plus 1e-4;
- the first-GW transfer sets;
- the captain per GW;
- the XI xP (captain counted twice) per GW;
- hits and FTs per GW.

When the first-GW transfers differ, ours is re-solved with their first-GW transfers fixed. An
equal objective means an alternative optimum, not a model difference.

The instances (`SPECS`, 24) use real deadlines in 2021/22–2024/25 and 2026/27 (never the
2025/26 holdout), with:
- template and seeded random start states, some rolled forward with `RollPolicy` so FTs bank
  (up to 5) and selling prices drift from prices;
- `ep_next` and `rolling` xP;
- 1- and 6-GW horizons (shorter at the season's end);
- the default pruning, plus three instances with pruning off (the full pool, about 700 players);
- two GW1 instances.

### Latest results

2026-10-07, open-fpl-solver `ec65f5e`, `--gap 0`, so both solvers prove optimality. Columns:
- `cand`: number of candidates;
- `FT`: free transfers at the decision;
- `sell != price (sold)`: owned players whose selling price differs from the price, and how many
  of them our plan sells;
- `offset`: the FT-state constant;
- `tol`: (sum of the final gaps) × |objective| + 1e-4;
- `s`: seconds for the build and solve (ours on 1 thread, theirs with HiGHS `parallel` on).

| instance | cand | FT | sell != price (sold) | ours | theirs | offset | ours - (theirs - offset) | tol | gap ours/theirs | s ours/theirs | first-GW transfers equal | captains equal | per-GW xP equal | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2021/22 gw1 template ep_next h6 | 79 | 0 | 0 (0) | 208.4166 | 211.4166 | 3.0000 | -5.68e-14 | 1.0e-04 | 0.0e+00/0.0e+00 | 0.6/1.1 | no: ours >18073, theirs 171314>121145,222017; ours with theirs fixed 208.4166 | no | yes | agree (alternative optimum) |
| 2021/22 gw8 template ep_next h1 | 59 | 1 | 0 (0) | 71.2180 | 74.2180 | 3.0000 | +0.00e+00 | 1.0e-04 | 0.0e+00/0.0e+00 | 0.1/0.2 | yes | yes | yes | agree |
| 2021/22 gw8 template ep_next h6 | 61 | 1 | 0 (0) | 377.4151 | 380.4151 | 3.0000 | +0.00e+00 | 1.0e-04 | 0.0e+00/0.0e+00 | 2.5/2.9 | yes | yes | yes | agree |
| 2021/22 gw20 random:1+roll3 rolling h6 | 98 | 4 | 2 (0) | 397.8823 | 400.8823 | 3.0000 | -5.12e-13 | 1.0e-04 | 0.0e+00/0.0e+00 | 11.2/26.0 | yes | yes | yes | agree |
| 2021/22 gw27 random:11+roll1 ep_next h6 full | 711 | 2 | 0 (0) | 427.3226 | 430.3226 | 3.0000 | +0.00e+00 | 1.0e-04 | 0.0e+00/0.0e+00 | 78.9/112.7 | yes | yes | yes | agree |
| 2022/23 gw5 random:2 ep_next h6 | 76 | 1 | 0 (0) | 376.4447 | 379.4447 | 3.0000 | +0.00e+00 | 1.0e-04 | 0.0e+00/0.0e+00 | 3.5/9.7 | yes | yes | yes | agree |
| 2022/23 gw15 template+roll2 rolling h1 | 45 | 3 | 3 (1) | 83.6524 | 86.6524 | 3.0000 | +0.00e+00 | 1.0e-04 | 0.0e+00/0.0e+00 | 0.1/0.2 | yes | yes | yes | agree |
| 2022/23 gw15 template+roll2 rolling h6 | 67 | 3 | 3 (1) | 362.5439 | 365.5439 | 3.0000 | +0.00e+00 | 1.0e-04 | 0.0e+00/0.0e+00 | 4.8/16.5 | yes | yes | yes | agree |
| 2022/23 gw30 random:3+roll4 rolling h6 | 91 | 5 | 1 (1) | 349.2347 | 352.2347 | 3.0000 | +5.68e-14 | 1.0e-04 | 0.0e+00/0.0e+00 | 0.5/15.5 | yes | no | yes | agree |
| 2023/24 gw10 template rolling h6 full | 730 | 1 | 0 (0) | 381.3952 | 384.3952 | 3.0000 | -5.68e-14 | 1.0e-04 | 0.0e+00/0.0e+00 | 6.3/31.3 | yes | yes | yes | agree |
| 2023/24 gw25 random:4+roll1 ep_next h6 | 91 | 2 | 1 (1) | 446.9286 | 449.9286 | 3.0000 | +0.00e+00 | 1.0e-04 | 0.0e+00/0.0e+00 | 2.8/14.5 | yes | yes | yes | agree |
| 2023/24 gw25 random:4+roll1 ep_next h1 | 58 | 2 | 1 (0) | 151.0920 | 154.0920 | 3.0000 | +0.00e+00 | 1.0e-04 | 0.0e+00/0.0e+00 | 0.0/0.1 | yes | yes | yes | agree |
| 2023/24 gw36 template ep_next h6 | 69 | 1 | 0 (0) | 295.8281 | 298.8281 | 3.0000 | -5.68e-14 | 1.0e-04 | 0.0e+00/0.0e+00 | 0.2/0.6 | yes | yes | yes | agree |
| 2024/25 gw4 template ep_next h6 | 55 | 1 | 0 (0) | 428.9354 | 431.9354 | 3.0000 | +5.68e-14 | 1.0e-04 | 0.0e+00/0.0e+00 | 5.7/12.1 | yes | yes | yes | agree |
| 2024/25 gw12 random:5+roll3 rolling h6 | 62 | 4 | 2 (0) | 329.1609 | 332.1609 | 3.0000 | -5.68e-14 | 1.0e-04 | 0.0e+00/0.0e+00 | 0.3/5.4 | yes | yes | yes | agree |
| 2024/25 gw22 random:6+roll2 ep_next h1 | 58 | 3 | 2 (1) | 81.6460 | 84.6460 | 3.0000 | -1.42e-14 | 1.0e-04 | 0.0e+00/0.0e+00 | 0.1/0.2 | yes | yes | yes | agree |
| 2024/25 gw22 random:6+roll2 ep_next h6 | 65 | 3 | 2 (1) | 451.2468 | 454.2468 | 3.0000 | -1.93e-11 | 1.0e-04 | 0.0e+00/0.0e+00 | 2.0/7.6 | no: ours 501770>, theirs 488464>; ours with theirs fixed 451.2468 | yes | yes | agree (alternative optimum) |
| 2024/25 gw33 template rolling h6 full | 778 | 1 | 0 (0) | 356.6125 | 359.6125 | 3.0000 | +0.00e+00 | 1.0e-04 | 0.0e+00/0.0e+00 | 6.9/24.9 | yes | yes | yes | agree |
| 2026/27 gw1 template ep_next h1 | 67 | 0 | 0 (0) | 38.4450 | 41.4450 | 3.0000 | +0.00e+00 | 1.0e-04 | 0.0e+00/0.0e+00 | 0.0/0.1 | no: ours 446008>60689,430871,448047, theirs >154566,209244; ours with theirs fixed 38.4450 | no | yes | agree (alternative optimum) |
| 2026/27 gw3 template ep_next h6 | 48 | 1 | 0 (0) | 502.2978 | 505.2978 | 3.0000 | +0.00e+00 | 1.0e-04 | 0.0e+00/0.0e+00 | 2.8/6.2 | yes | yes | yes | agree |
| 2026/27 gw5 random:7+roll2 rolling h6 | 57 | 3 | 4 (4) | 353.8460 | 356.8460 | 3.0000 | +0.00e+00 | 1.0e-04 | 0.0e+00/0.0e+00 | 0.3/1.4 | yes | yes | yes | agree |
| 2026/27 gw6 random:8 ep_next h1 | 54 | 1 | 0 (0) | 90.8560 | 93.8560 | 3.0000 | +0.00e+00 | 1.0e-04 | 0.0e+00/0.0e+00 | 0.1/0.1 | no: ours 212319>, theirs 470313>; ours with theirs fixed 90.8560 | yes | yes | agree (alternative optimum) |
| 2026/27 gw6 random:8+roll4 rolling h6 | 59 | 5 | 1 (1) | 376.0661 | 379.0661 | 3.0000 | +0.00e+00 | 1.0e-04 | 0.0e+00/0.0e+00 | 10.7/11.2 | yes | yes | yes | agree |
| 2022/23 gw22 random:9+roll6 ep_next h6 | 92 | 5 | 2 (2) | 434.5498 | 437.5498 | 3.0000 | -5.68e-14 | 1.0e-04 | 0.0e+00/0.0e+00 | 0.5/13.1 | yes | yes | yes | agree |

24/24 agree within the MIP gaps (gap 0.0); max |diff| 1.93e-11; first-GW transfers equal in 20/24

All 24 instances agree to within floating-point error (|diff| ≤ 2e-11). The 4 instances whose
first-GW transfers differ are alternative optima: fixing their first GW in our model gives the
same objective (the plans differ only in tied choices). The captains that differ are ties in
xP, since the per-GW XI xP are equal.

With `--gap 0.001` on both sides, all 24 also agree within the gaps (max |diff| 0.125; tol 0.1–0.8),
and first-GW transfers are equal in 19 of 24.

Not exercised by these instances: an owned player who left the game, and a club over the
team limit (the latter is not comparable, see above).

## `team_rho.py` and `team_model_eval.py`: the team model on develop

Phase 5 plan, Task 2 (PLAN §6.1; `fplopt.models.team`). Both read the built `data/` tables
read-only, as of 2023-07-01: develop results only, never validate or the 2025/26 holdout.

```sh
uv run python dev/team_rho.py [--data-dir data] [--devig power|shin]          # ~1 min
uv run python dev/team_model_eval.py [--data-dir data] [--quick] [--out DIR]  # ~10 min
```

**ρ** (`team_rho.py`): for each ρ on a grid, every develop match's market λ is re-solved
under that ρ and the realized scores are scored under Dixon-Coles; ρ = argmax. On the 2,660
develop matches with pre-match odds the profile is flat around **ρ = −0.03** (+0.6
log-likelihood in total over ρ = 0; ρ = −0.10 is 4.1 worse). Hard-coded as `team.RHO`.

**Evaluation** (`team_model_eval.py`): every develop deadline (265), fitted at the
walk-forward cutoff (`refit_deadline`), predicting the target GW and the next 5. Scores per
side: Poisson log-likelihood of the team's goals and the P(clean sheet) Brier score, per
horizon, for the model (market λ where odds are visible, else ratings), the ratings alone,
an Elo-only reference (log λ linear in the Elo difference + home, Poisson-fitted on the 3
years before the cutoff, Elo at the deadline) and the market λ. It writes
`results/<UTC>-team-eval/metrics.json`.

Results (2026-10-08; chosen parameters: half-life 45 d, w 0.75, prior 2; all develop
seasons, mean per side):

| horizon | model ll | ratings ll | Elo ll | model Brier | ratings Brier | Elo Brier |
|---|---|---|---|---|---|---|
| 0 | −1.4471 | −1.4542 | −1.4656 | 0.1859 | 0.1870 | 0.1891 |
| 1 | −1.4550 | −1.4550 | −1.4657 | 0.1871 | 0.1872 | 0.1889 |
| 2 | −1.4551 | −1.4551 | −1.4655 | 0.1870 | 0.1870 | 0.1888 |
| 3 | −1.4548 | −1.4548 | −1.4656 | 0.1873 | 0.1873 | 0.1891 |
| 4 | −1.4533 | −1.4533 | −1.4634 | 0.1882 | 0.1882 | 0.1898 |
| 5 | −1.4531 | −1.4531 | −1.4629 | 0.1884 | 0.1884 | 0.1901 |

- Horizon 0, the 2,454 fixtures with visible odds: market −1.4477 / 0.1854, ratings
  −1.4554 / 0.1866, Elo −1.4670 / 0.1890. Shin's de-vig: −1.4477 / 0.1854 (no difference).
- Odds visible at the deadline (target GW): 2016/17 95.5%, 2017/18 92.9%, 2018/19 94.5%,
  2019/20 91.8%, 2020/21 91.6%, 2021/22 88.4%, 2022/23 91.1%. The misses are fixtures
  whose football-data collection time (Tuesday/Friday 15:00 UK) falls after the GW's
  deadline, mostly midweek games in a GW that starts at the weekend. Horizons 1–5: ≤ 0.3%.
- The model beats the Elo reference in every develop season (all horizons pooled).
- Tuning grid (half-life × w × prior strength, 120 variants; mean over 2017/18–2022/23 of
  the ratings' log-likelihood over horizons 1–5): best 45 d / 0.75 / 2 (−1.4516), flat
  nearby (60 d / 0.75 / 2: −1.4517; 90 d / 0.75 / 5: −1.4521). Stats only (w 0): best
  −1.4551; market only (w 1): −1.4522. Long half-lives are worse (365 d: ≤ −1.4546).
- Time (one process): `fit_team` 0.2–0.5 s per cutoff, `team_lambdas` ≤ 0.2 s per
  deadline.

## `minutes_eval.py`: minutes model measurements

Phase 5 plan, Task 3 (PLAN §6.3, §6.4). Two subcommands, both reading the built `data/` tables
(read-only, `--data-dir`):

```sh
uv run python dev/minutes_eval.py --data-dir data inference
uv run python dev/minutes_eval.py --data-dir data walk --seasons 2017-2022 [--grid g.json] \
    [--horizon-decays 1,0.85 | --predict-grid p.json] [--jobs 6] [--out <dir>]
```

- `inference`: accuracy of the inferred starters (11 most minutes per team-fixture, ties by
  `player_key`) where FPL's `starts` exist (2022/23 GW16 – 2024/25; this measures the proxy,
  not a model).
- `walk`: develop only (refuses seasons after 2022/23). At every GW deadline: fit at the refit
  cutoff (`refit_deadline`, every 4 GWs), predict, apply the availability layer (from 2021/22),
  score each (player, fixture) with a `player_match` row. 3-class (0 / 1–59 / 60+) log loss and
  RPS, Brier and reliability of P(start), per horizon group (0, 1–5), against the last-5
  reference (smoothed class frequencies over the player's last 5 rows). The in-memory store holds
  only seasons ≤ the last requested one. ~33 min for 6 GBM variants × 5 horizon decays at
  `--jobs 6`.

### Results (2026-10-08)

Start inference, 75,022 rows: accuracy 98.65%, false positives 0.96% of real non-starts, false
negatives 2.30% of real starts; 441 of 2,008 team-fixtures have an error, 569 of the 1,016 wrong
rows are 45-minute half-time ties.

Develop walk-forward (2017/18–2022/23, mean over seasons; defaults: `GbmParams()`,
`season_decay` 0.7, `horizon_decay` 0.85):

| | log loss h0 | RPS h0 | Brier P(start) h0 | log loss h1–5 | RPS h1–5 | Brier h1–5 |
|---|---|---|---|---|---|---|
| model, no flags | 0.532 | 0.105 | 0.101 | 0.650 | 0.140 | 0.138 |
| last-5 reference | 0.632 | 0.124 | 0.122 | 0.736 | 0.153 | 0.151 |
| 2021/22–2022/23: no flags | 0.512 | 0.099 | 0.094 | 0.622 | 0.132 | 0.129 |
| 2021/22–2022/23: with flags | 0.476 | 0.089 | 0.087 | 0.607 | 0.127 | 0.125 |

## `shares_eval.py`: goal/assist shares and penalties

Phase 5 plan, Task 4 (PLAN §6.2, §3; `fplopt.models.shares`). Two subcommands, both reading the
built `data/` tables read-only into an in-memory store (`--data-dir`):

```sh
uv run python dev/shares_eval.py --data-dir data npxg-check                      # ~1 min
uv run python dev/shares_eval.py --data-dir data walk --seasons 2017-2022 \
    [--grid g.json] [--jobs 6] [--out <dir>]                          # ~2.5 min + ~0.2 min / variant
```

- `npxg-check`: the PLAN §3 VERIFY. On the 27,825 played rows of 2022/23 GW16 – 2024/25 (to
  2025-04-07) where Understat and FPL/Opta both exist (a data-source check, so these validate
  seasons may be read; nothing from 2025/26 is loaded), Understat `us_npxg` vs FPL `fpl_xg` −
  penalty xG × attempts, and `us_xa` vs `fpl_xa`.
- `walk`: develop only (deadlines of 2017/18–2022/23; anything else is refused). At every GW
  deadline the team model, the minutes model (+ the availability layer from 2021/22) and the
  shares, each fitted at the refit cutoff; every (player, fixture) with a `player_match` row is
  scored at horizon 0 and pooled over horizons 1–5: Poisson log-likelihood of goals and of FPL
  assists, Brier and reliability of P(goal ≥ 1) = 1 − exp(−e_goals). The reference: the
  position's goals (assists) per 90 over the cutoff's last 3 seasons × e_minutes / 90 × λ_for /
  the league's mean goals per team-match. `--grid` is a JSON list of predict-time
  `SharesParams` overrides (default: the 27-variant first round below); all share the fits.
  Writes `summary.json` and `timings.csv`.

### Results (2026-10-08)

**npxG source check** (closes the PLAN §3 VERIFY):
- FPL/Opta gives every penalty exactly 0.79 xG (all 30 rows whose only shot was a penalty);
  Understat 0.761. So the subtraction uses 0.79, not 0.76 (the difference is small: 234
  attempts).
- With the true attempts (Understat's penalty goals + FPL misses): per match r 0.932, MAE 0.032;
  per player-season (1,123 with ≥ 450 min) r 0.992, per-90 r 0.987, but Σ FPL / Σ Understat =
  0.914 (slope 0.88): Opta's npxG is ~9% lower. A scale factor is needed when sources are mixed:
  k_goals = 1.095 (fitted walk-forward on the visible overlap by `fit_shares`).
- Without Understat, penalty goals must be inferred. Only 58% of penalty goals are scored by the
  club's rank-1 listed taker at kickoff (`penalties_order`), so a hard taker rule fails
  (a scratch check: counting the candidates of the rank-1 taker only finds ~57% of the
  penalty rows and adds ~75 false ones; counting everyone's adds ~380 false ones). The module uses the expected value: candidates = min(goals, ⌊(fpl_xg − 0.79·missed) /
  0.79⌋), × π = 0.55 for the main taker and 0.20 for others (in-sample; from 2022/23 alone
  0.49 / 0.23). Per match r 0.902, MAE 0.036; per player-season r 0.985, per-90 r 0.981, ratio
  0.914; among takers (≥ 3 attempts) ratio 0.970 vs 0.908 with the true attempts. Out of sample
  (π from 2022/23, scored on 2023/24–2024/25): r 0.985, ratio 0.897, takers 0.998. Ignoring
  penalties (misses only): ratio 0.970 overall but 1.207 for takers (+21% npxG for penalty
  takers).
- xA: FPL `fpl_xa` = 0.80 × Understat (per player-season r 0.946, per-90 r 0.908, per match
  0.746); k_assists = 1.257.
- Recommendation: the substitution is acceptable for shares with the scale factors (k, fitted on
  the overlap) and the expected-value penalty inference; shares are ratios within one
  team-fixture, so a uniform scale cancels unless sources mix (Understat history with FPL
  current season, i.e. from 2025/26).

**Develop walk** (mean over 2017/18–2022/23; per player-fixture; chosen `SharesParams()`:
1920 pseudo-minutes, season weights 2-2-1-1, club change 0.5, goals fallback 0.5, price prior
for everyone):

| | goals LL h0 | assists LL h0 | Brier P(goal) h0 | goals LL h1–5 | assists LL h1–5 | Brier h1–5 |
|---|---|---|---|---|---|---|
| model | −0.13552 | −0.13467 | 0.03245 | −0.14074 | −0.13915 | 0.03311 |
| reference | −0.14039 | −0.13822 | 0.03328 | −0.14540 | −0.14266 | 0.03382 |

- The model beats the reference in every season, at both horizon groups, on goals, assists and
  Brier. Brier resolution 0.0040 vs 0.0031 (h0), reliability 1.5e-5 vs 1.8e-5. Mean e_goals
  0.0439 vs 0.0430 realized (h0), e_assists 0.0399 vs 0.0388: both ~2–3% high (the reference
  too: the team λ sums over pool players, some of whom have no `player_match` row).
- Reliability is good below P(goal) 0.3; above it the model is a little high at h1–5 (0.34 →
  0.31, 0.44 → 0.39).
- Tuning (38 variants; objective: mean over seasons of goals + assists LL, all horizons): round 1
  pseudo-minutes 240/480/960 × current-season weight 2/3/5 × club-change weight 1/0.5/0.25
  (27); round 2 1440/1920/2880 and current weight 1, price prior for everyone, goals weight
  0.25/1 (9); round 3 1920/2880 with the price prior for everyone (2). Pseudo-minutes 240 →
  −0.2803 … 960 → −0.2786, flat from 1440 (−0.27815 to −0.27823 with the price prior);
  current weight 2 best (1: −0.2788, 5: −0.2791); the club-change weight moves the objective
  by ≤ 0.00005 between 0.25 and 1; goals weight 0.25/0.5/1 within 0.00008.
- Time (6 processes in parallel, the chosen settings): `fit_shares` 0.15–0.59 s per cutoff
  (1.0 s on the full `data/` store at 2022/23), `predict_shares` 0.10–0.27 s per deadline
  (median 0.17 s). The walk takes ~2.5 min for one variant at `--jobs 6`.

## `calibrate_v1.py`: v1 calibration and the ban residual

> **Re-run after the review (2026-10-08):** the flag mapping now uses out-of-fold P(start).
> `walk --seasons 2016-2024 --jobs 9 --ban-residual --out results/p5-review-walk` (3.2 min),
> then `fit --walk results/p5-review-walk --first-fit 2017 --parts p_start:1+ --write`.
> Develop means: none MSE h0 4.2608 / h1–5 4.6867; chosen (p_start:1+) 4.2608 / 4.6845. The
> other parts are still worse. The walk writes `params.txt` (the `V1Params` fingerprint),
> which `fit --write` stores as `CALIBRATION_PARAMS`. The numbers below are from the first
> run.

Phase 5 plan, Task 5 (PLAN §5 *Calibration*; `fplopt.models.assemble`, `.calibration`). Reads
the built `data/` tables read-only into an in-memory store (`--data-dir`):

```sh
uv run python dev/calibrate_v1.py --data-dir data walk --seasons 2016-2024 --jobs 6 \
    --out results/p5-task5-walk [--ban-residual]                           # ~4.5 min
uv run python dev/calibrate_v1.py fit --walk results/p5-task5-walk-ban --first-fit 2017 \
    --parts "p_start:1+" [--variant SPECS ...] [--write]                   # ~5 min
uv run python dev/calibrate_v1.py ban --off results/p5-task5-walk --on results/p5-task5-walk-ban
```

- `walk`: v1's uncalibrated per player-fixture components (every horizon) at every deadline of
  2016/17–2024/25 (fitted at the refit cutoffs), joined to each fixture's realized minutes,
  start (real, else inferred), goals, clean sheet, the club's goals against and the re-scored
  points. One parquet per season. The validate seasons are walked only so that the 2024/25
  and 2025/26+ calibration entries can be fitted; nothing scores them.
- `fit`: the expanding-window calibrations (season S: fitted on the walk rows of seasons in
  [`--first-fit`, S)) and their effect on develop (2017/18–2022/23 only): per-fixture xP MSE
  (horizon 0, 1–5), Brier of P(start), of the team's P(CS) and of P(goal ≥ 1), minutes log
  loss and goals log-likelihood, means over seasons. Part specs: `name` (all horizons), `name:0`
  or `name:1+`. `--write` regenerates `src/fplopt/models/calibration_table.py`.
- `ban`: the ban residual (`MinutesParams.ban_residual`) against P(start) = 0 on develop.

### Results (2026-10-08)

**Ban residual** (develop, mean over 2017/18–2022/23; ~47 banned player-fixtures per season,
31% of which started): adopted.

| | minutes log loss (all horizons) | on banned rows | Brier P(start) on banned rows | xP MSE h0 | xP MSE h1–5 |
|---|---|---|---|---|---|
| P(start) = 0 | 0.6302 | 11.62 | 0.311 | 4.26088 | 4.68698 |
| residual | 0.6287 | 0.78 | 0.229 | 4.26034 | 4.68695 |

**Calibration** (on the ban-residual walk, `--first-fit 2017`: 2016/17 is burn-in, so 2017/18
is uncalibrated and 2018/19 is the first calibrated season; develop means over 2017/18–2022/23,
per player-fixture):

| variant | xP MSE h0 | xP MSE h1–5 | Brier P(start) h0 / h1–5 | team P(CS) Brier | P(goal) Brier h0 |
|---|---|---|---|---|---|
| none | 4.26034 | 4.68695 | 0.09848 / 0.13618 | 0.18740 | 0.03244 |
| P(start), all horizons | 4.27054 | 4.68355 | 0.09958 / 0.13572 | | |
| P(start), h0 only | 4.26104 | 4.68740 | 0.09852 / 0.13616 | | |
| **P(start), h1+ (kept)** | **4.26034** | **4.68426** | **0.09848 / 0.13577** | | |
| team P(CS) | 4.26222 | 4.68763 | | 0.18783 | |
| team P(CS), h1+ | 4.26034 | 4.68820 | | 0.18778 | |
| P(goal ≥ 1) | 4.26316 | 4.68645 | | | 0.03246 |
| P(goal ≥ 1), h1+ | 4.26034 | 4.68630 | | | |
| linear xP per position | 4.26717 | 4.68560 | | | |

- Uncalibrated v1 is already calibrated in the large (horizon 0: mean xP 1.317 vs 1.327 realized;
  P(start) 0.353 vs 0.352; e_goals 0.0437 vs 0.0428; e_bonus 0.104 vs 0.103), so most maps only
  add noise. P(CS) as p_60 · team P(CS) runs 8% under the player's realized FPL clean sheets
  (0.094 vs 0.102: a player off after 60+ minutes keeps a clean sheet the team later loses);
  p_cs^(m_sixty/90) closes half of it but moved xP MSE by < 0.002 either way, so the plan's
  formula stays.
- Only isotonic P(start) at horizons ≥ 1 improves its component and xP; it is the only part in
  the table. P(start) + P(goal) at h1+ together: 4.68596 (worse than P(start) alone).
- With `--first-fit 2016` (2017/18 calibrated on 2016/17 alone) every part was worse still
  (e.g. all parts: xP MSE h0 4.2756; on the walk without the ban residual).
- Time: the walk fits 1.1 s (2016/17 cutoffs) to 21 s (2024/25) per cutoff, predicts 0.5–1.5 s
  per deadline; ~4.5 min for 2016–2024 at `--jobs 6`.

## `defcon_prior.py`: defensive contributions (#14, `k` and `r`, live check)

Phase 5b, Task 10 (PLAN §6.5, §11; issue #14). Reads FPL-Core-Insights (FCI) files as plain
CSV data and our built tables read-only (only 2024/25 and 2026/27; `live` filters 2025/26 out
on read). Run it with `python -I` (isolated mode), never from inside the data folder:

```sh
.venv/Scripts/python.exe -I dev/defcon_prior.py --data-dir <data> verify   # #14, seconds
.venv/Scripts/python.exe -I dev/defcon_prior.py --data-dir <data> fit [--tackles f_tackles_attempted]  # ~1 min
.venv/Scripts/python.exe -I dev/defcon_prior.py --data-dir <data> live --gws 2-7   # ~2 min
```

### FPL-Core-Insights files (research only, never shipped)

FCI has no licence (PLAN §10, §12): its files are only for setting the prior once, live in the
git-ignored `research/fpl-core-insights/` (never `raw/`, `data/` or the package) and are
treated as untrusted data. Downloaded 2026-10-08 from
[olbauday/FPL-Core-Insights](https://github.com/olbauday/FPL-Core-Insights) at commit
`d6148f8743a6dd64b6ec1968ed86974f56725e20`, with
`curl -sSfL -o <local> https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/d6148f8743a6dd64b6ec1968ed86974f56725e20/<path>`:

| local (`research/fpl-core-insights/`) | repo path (`data/...`) | bytes |
|---|---|---|
| `2024-2025/playermatchstats.csv` | `2024-2025/playermatchstats/playermatchstats.csv` | 1,903,185 |
| `2024-2025/matches.csv` | `2024-2025/matches/matches.csv` | 163,575 |
| `2024-2025/players.csv` | `2024-2025/players/players.csv` | 39,340 |
| `2024-2025/teams.csv` | `2024-2025/teams/teams.csv` | 1,363 |
| `2026-2027/GW{1..5}/playermatchstats.csv` | `2026-2027/By%20Gameweek/GW{1..5}/playermatchstats.csv` | 84,497 / 128,640 / 84,113 / 135,421 / 143,415 |
| `2026-2027/GW{1..5}/matches.csv` | `2026-2027/By%20Gameweek/GW{1..5}/matches.csv` | 8,205 / 14,215 / 8,095 / 14,126 / 14,743 |
| `2026-2027/GW{1..5}/players.csv` | `2026-2027/By%20Gameweek/GW{1..5}/players.csv` | 28,954 / 29,726 / 31,101 / 31,335 / 31,713 |
| `2026-2027/teams.csv` | `2026-2027/teams.csv` | 926 |

Nothing from 2025-2026. The `teams.csv` files turned out not to be needed (FCI's club ids are
FPL team codes).

### #14: FCI reproduces FPL's counts (2026/27 GW1–5, 2026-10-08)

Players mapped by FPL `player_code` (= our `player_key`), fixtures by (home, away) club code
(FCI's club ids are FPL team codes): all 50 matches and all 1,537 FCI rows with minutes > 0
mapped; 1,537 of our 1,538 played `player_match` rows matched (one 90-minute row of player
566213 in fixture 2026047 is missing from FCI). Minutes agree exactly on 73% of rows (mean
|difference| 0.27). On the 1,437 outfield rows (FCI leaves `blocks` blank for goalkeepers):

| | rows | exact match | mean abs diff | threshold outcome agrees | FPL / FCI hit rate |
|---|---|---|---|---|---|
| DEF CBIT (≥ 10) | 532 | 100% | 0 | 100% | 20.1% / 20.1% |
| MID CBIRT (≥ 12) | 729 | 100% | 0 | 100% | 7.7% / 7.7% |
| FWD CBIRT (≥ 12) | 176 | 100% | 0 | 100% | 0.6% / 0.6% |

Each component (clearances + blocks + interceptions vs FPL's CBI, tackles, recoveries) matches
exactly too, and FPL's `defensive_contribution` equals our CBI + tackles (DEF) / + recoveries
(MID/FWD) on every row. **The tackle count is FCI's `tackles_won` column**: FCI's `tackles` is
blank in 2026/27. In 2024/25 FCI has both, and its `tackles` is a different, larger count
(summed per team-match it equals FCI's team tackles won / tackles-won %, i.e. attempts: 17.5
per team-match vs 10.5 won). The team tackles-won stat itself is 15.4 per team-match in
2026/27 GW1–5 vs 10.5 in 2024/25, so FotMob's tackle definition probably changed between the
seasons: per 90, FPL's 2026/27 tackles (DEF 1.58, MID 1.75, FWD 0.56) sit between FCI 2024/25
`tackles_won` (1.11, 1.12, 0.48) and `tackles` (1.79, 1.94, 0.80). CBI (DEF 6.06 vs 5.46 per
90) and recoveries (3.53 vs 3.87) are within early-season noise of 2024/25. `k` and `r` barely
depend on this (below).

### `k` and `r` (FCI 2024/25)

10,797 played outfield rows of 380 matches (positions from our `player_season` 2024/25: all
mapped). Count: DEF CBIT, MID/FWD CBIRT, tackles = `tackles_won`. Each row is predicted from
the player's earlier rows of the season (`rate = (actions + k · pos_mean) / (minutes/90 + k)`,
`pos_mean` from the position's earlier rows) with NB(rate · minutes/90, r); (k, r) maximize
that walk-forward log-likelihood. Mean log-likelihood per row and the threshold calibration
(given the realized minutes):

| group | rows | k | r | mean LL (Poisson) | hit rate | mean P(hit) | k, r with FCI `tackles` | half-season k, r |
|---|---|---|---|---|---|---|---|---|
| DEF | 3,787 | **3.49** | **13.2** | −2.2955 (−2.3305) | 13.6% | 12.2% | 3.90, 14.6 | 2.72, 13.1 |
| MID | 5,738 | 1.83 | 16.1 | −2.2201 (−2.2435) | 7.5% | 8.1% | 1.63, 15.5 | 2.24, 16.3 |
| FWD | 1,272 | 2.57 | 8.8 | −1.8399 (−1.8585) | 0.6% | 0.9% | 2.03, 8.3 | 6.69, 9.3 |
| **MID + FWD** | 7,010 | **1.94** | **15.2** | −2.1517 (−2.1738) | 6.2% | 6.8% | 1.71, 14.7 | 2.55, 15.7 |

The profile likelihood is flat near the optimum (DEF: −2.2974 at k = 2, −2.2964 at 5; MID+FWD:
−2.1517 at 2, −2.1529 at 3). DEF and MID/FWD differ materially in k (3.5 vs 1.9), so they get
their own (k, r); MID and FWD are pooled (FWD alone gives r 8.8, which moves a forward's
P(CBIRT ≥ 12) by hundredths of a percent). The half-season check (first-half totals
predicting the second half) agrees. Hard-coded in `fplopt.models.components`
(`DEFCON_K`, `DEFCON_R`). `DEFCON_PRIOR_MEAN` (DEF 6.57, MID 7.73, FWD 4.07 per 90) are the
2024/25 FCI season means, used only while a position has no row of the season (a season's
first deadline); with FCI `tackles` they would be 7.25, 8.55, 4.39 (FPL 2026/27 GW1–5: 7.63,
8.14, 4.27).

### The v1 term: calibration table and the live check (2026-10-08)

The new `ComponentsParams` fields change `calibration_fingerprint`, so the table was
regenerated (`calibrate_v1.py walk --seasons 2016-2024 --jobs 9 --ban-residual`, 4.4 min, then
`fit --first-fit 2017 --parts p_start:1+ --write`). Develop rules score no defcon: the walk's
per-fixture frames are identical to a walk on the previous commit apart from the two new
columns (all 9 seasons), every develop number of `fit` is unchanged (none: xP MSE h0 4.260925 /
h1–5 4.686697; chosen: 4.260925 / 4.684454), and so are the table's entries; only
`CALIBRATION_PARAMS` changed.

`live --gws 2-7` (2026/27, tables read without 2025/26; GW6–7 not played yet, so GW2–5): v1
at each deadline (uncalibrated: horizon 0, where the table's h1+ P(start) map does nothing),
P(defcon) per pool player-fixture at horizon 0 against FPL's realized count reaching the
threshold (no `player_match` row = 0 minutes = no defcon):

| position | player-fixtures | mean P(defcon) | realized | Brier | of those who played: P / realized |
|---|---|---|---|---|---|
| DEF | 849 | 0.088 | 0.104 | 0.078 | 0.150 / 0.208 (424) |
| MID | 1,146 | 0.047 | 0.038 | 0.032 | 0.084 / 0.076 (578) |
| FWD | 307 | 0.0014 | 0.0033 (1 award) | 0.003 | 0.003 / 0.007 (140) |

By predicted bin (all positions): ≤ 0.02 → 0.001 realized (1,090 rows), 0.02–0.05 → 0.034,
0.05–0.1 → 0.048, 0.1–0.2 → 0.138, 0.2–0.3 → 0.370 (119), 0.3–0.4 → 0.381 (42). Defenders run
about 15% under their realized rate (as in the 2024/25 fit check, 12.2% vs 13.6%, and DEF
CBIT per 90 is higher so far this season than FCI's 2024/25), midfielders slightly over; GW2
rates rest on a single match each.

## `bench_weights_eval.py`: bench-weight reliability

Phase 5 plan, Task 8 (PLAN §7 *Bench*). Develop only (refuses anything but 2017/18–2022/23):

```sh
uv run python dev/bench_weights_eval.py --data-dir data [--seasons 2017-2022] [--random 3] \
    [--jobs 6] [--out <dir>] [--max-gws N]
```

At every GW deadline: the template squad and `--random` seeded random squads
(`fplopt.backtest.start_states`), each with its lineup from `best_lineup` on `v1`'s horizon-0 xP
(the same rule as the planner's projected XI). Per bench slot (0 = GK, 1–3 outfield) it compares
the fixed weights (0.03 / 0.21 / 0.06 / 0.002) and the minutes-based ones
(`fplopt.optimize.minutes.minutes_bench_weights`) with the realized autosubs under
`gw_score`'s rules:

- `needed`: slot k's player would have come on had he played (the lineup is rescored with his
  minutes set to 1 if he had none). This is what the weight estimates: the objective values the
  slot at w_k · xP_k, and xP_k already includes his own P(play).
- `came_on`: he did come on, compared with w_k · P(he plays).
- `skip` (diagnostic, not used by the planner): outfield slot k with the skip rule,
  P(M ≥ 1 + Σ_{j<k} B_j), with M the absent outfield starters (Poisson-binomial) and B_j the
  earlier outfield bench players playing (Bernoulli of their P(play)).

It prints means and Brier scores per slot (all rows, single-fixture and double GWs) and a
reliability table per slot (bins of the minutes-based weight); `--out` also writes
`rows.parquet`, `summary.json` and `summary.txt`. The in-memory store holds only seasons ≤ the
season being run. About 4 min for 2017/18–2022/23 at `--jobs 6` (v1 fits dominate).

### Results (2026-10-09)

898 squad-GWs (2017/18–2022/23, template + 3 random squads per deadline). `needed` is the
realized rate; Brier is of the `needed` event:

| slot | needed | fixed | minutes | skip | Brier fixed | Brier minutes | Brier skip |
|---|---|---|---|---|---|---|---|
| GK | 0.237 | 0.030 | 0.263 | 0.263 | 0.224 | 0.033 | 0.033 |
| 1 | 0.787 | 0.210 | 0.894 | 0.894 | 0.501 | 0.155 | 0.155 |
| 2 | 0.757 | 0.060 | 0.677 | 0.805 | 0.670 | 0.135 | 0.124 |
| 3 | 0.723 | 0.002 | 0.430 | 0.768 | 0.720 | 0.247 | 0.130 |

- The fixed weights are far too low for these squads: a bench slot was needed 24% (GK) to 79%
  (slot 1) of the time. The minutes-based weights track that; the GK slot's reliability is
  close to the diagonal in every bin.
- Slot 1 is overpredicted (0.89 vs 0.79; template squads 0.80 vs 0.56): a small underestimate of
  each starter's P(play) compounds over ten starters.
- Slots 2–3 are underpredicted (slot 3: 0.43 vs 0.72): the skip rule (a bench player who doesn't
  play passes his turn) is ignored. Accounting for it (`skip`) halves slot 3's Brier score.
- Random squads (678 of 898) hold more non-playing players than template squads (220): template
  `needed` rates are 0.27 / 0.56 / 0.56 / 0.51, random 0.23 / 0.86 / 0.82 / 0.79.
