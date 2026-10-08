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

