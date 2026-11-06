# Price Forecast, Stage B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a LightGBM forecast of each player's price change 1–6 deadlines ahead, built only from information available at the deadline, plus a walk-forward evaluation against "no change" and momentum. The evaluation decides Gate B.

**Architecture:**
- **Snapshot features** (price, transfers, ownership, availability) are a feature frame in `fplopt-features` (`price.rs`, registered in `FRAMES`, so `check leakage` covers it).
- **The forecast** lives in `fplopt-models` (`price.rs`): training rows, six regressions (one per horizon), prediction, clipping, baselines. It takes v1 xP from a caller-supplied function, because `Caches` lives in the backtester.
- **The evaluation** is a Rust research command (`fplopt-rs research price-eval`) in `fplopt-backtest` + `fplopt-cli`. It reuses `season_schedule`, `Caches` and the Diebold–Mariano test from `eval.rs`.

**Tech Stack:** Rust (`fplopt-features`, `fplopt-models`, `fplopt-backtest`, `fplopt-cli`), LightGBM through `fplopt_models::gbm`, pytest only for the unchanged Python tests.

**Spec:** `docs/superpowers/specs/2026-11-06-price-planner-design.md`, Stage B. Gate A passed (`docs/research/2026-11-price-planner-gate-a.md`). Branch `feat/price-sizing`, worktree `C:\Users\samin\Documents\GitHub\fpl-optimizer\.claude\worktrees\v1r`.

**Facts (code read on 2026-11-07):**
- **Feature builders.** `fplopt-features/src/lib.rs:17` defines `pub type FeatureBuilder = fn(&AsOfView) -> Result<Table>;`. `FEATURES` is at `:20` and `FRAMES` at `:32`.
- **Helpers in `baseline.rs`** (crate-visible):
  - `gameweeks(view, season)` (:63), `target(view) -> (season, gw, gw_index)` (:72);
  - `newest_snapshot(view, season)` (:84), `pool_with_snapshot(view, cols)` (:275), `SNAPSHOT_COLUMNS` (:17), `player_pool` (:115);
  - `util.rs` has `days_between`, `median`, `floats`, `opt_floats`.
- **`player_snapshot` columns** (`fplopt-build/src/snapshots.rs:97-125`): `snapshot_at, source, season, player_key, team_key, element_type, now_cost, status, chance_of_playing_next_round (OptI64), news_added (OptTs), selected_by_percent (OptF64), transfers_in_event, transfers_out_event, cost_change_event`.
  - The `*_event` counts are cumulative within the current GW and reset at each GW.
  - **No total-managers count exists anywhere.**
- **`AsOfView`** (`fplopt-store/src/lib.rs`): `deadline()`, `earlier(cutoff)`, `table(name, cols)` (every visible row), `latest(name, by, cols)`, `schedule(season)`. `DataStore::from_tables` is for tests.
- **`Table` / `Column`** (`fplopt-build/src/table.rs`): `Table::new().with(name, Column::F64(..))`, readers `i64(..)`, `f64(..)`, `opt_f64(..)`, `str(..)`. No joins; joins are done with HashMaps.
- **LightGBM:**
  - `fplopt-models/src/gbm.rs`: `GbmParams { objective, num_boost_round, learning_rate, num_leaves, min_data_in_leaf, feature_fraction, lambda_l2 }` (default objective "binary"; use `"regression"`).
  - `train(table, features, label, weight, params) -> Result<Gbm>`, `Gbm::predict(&Table)`. Training is deterministic (`seed=0 num_threads=1 deterministic=True`).
- **v1 xP:** `fplopt_models::assemble::xp_v1(view)` or `predict("v1", view)` gives `player_key, gw, gw_index, horizon, xp`. The backtester's `Caches::xp(model, view)` (`simulator.rs:89`) memoizes and disk-caches it.
- **Leakage:**
  - `fplopt_models::LEAKAGE_BUILDERS` (lib.rs:113) and `FRAMES` are checked by `check leakage` (`fplopt-cli/src/main.rs:385-440`).
  - Each builder runs on clean, corrupted-future and truncated stores, and the outputs must be bit-identical.
  - `player_snapshot` is already in `FEATURE_TABLES`.
- **Evaluation pieces** (`fplopt-backtest/src/eval.rs`):
  - `season_unit` (:222) loops `season_schedule(store, season, 1)` deadlines.
  - `candidate_keys(frames, positions)` (:157) is the union of each model's top `DEFAULT_PRUNE_N = [(1,20),(2,60),(3,60),(4,30)]` per position by horizon-summed xP.
  - `diebold_mariano(loss: &[(Micros, f64, f64)], lag)` (:853, private) clusters by deadline.
  - Splits are at :507. The metrics/log pattern is in `fplopt-cli/src/models_eval.rs:233-358`, writing through `fplopt_core::experiments::log_experiment`.
- **Schedule and gws:** `season_schedule(store, season, gw_index)` (`simulator.rs:156`) returns `Gameweek { season, gw, gw_index, deadline, lockdown }`. gw ≠ gw_index in 2019/20 and 2022/23.
- **Deadline prices:** `player_gw.value` at (season, gw) is the deadline price, the same as the last pre-deadline snapshot.
- **Existing price code:** `PriceSource { Fixed, Oracle }` (`policies.rs:259`). `prices=forecast` is refused at `specs.rs:322-328` until Stage C wires it; this plan does not wire it.

**Rules for every task:**
- Cargo always with `-j 2`.
- Never use season 2025 for fitting, tuning or evaluation, and never touch `raw/`.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- No push or server work without the controller's go-ahead.
- Features read only through the view.

---

### Task 0: Spec addendum (controller)

The map of the code showed four places where the spec doesn't fit. Record the resolutions in the spec, under a new "Addendum (2026-11-07): Stage B as built", before any code:

1. **Ownership scaling.** There is no total-manager count. Transfers are scaled two ways:
   - by ownership: `net / (selected_by_percent + 0.1)`;
   - by the deadline's market activity: `net / Σ_players |net|` at the same snapshot.
2. **Candidate set.** "The planner's candidates" is the union of the top `DEFAULT_PRUNE_N` per position by summed v1 xP, as `models eval` defines candidate MSE. This replaces "the optimizer's pruned set from template and random states".
3. **No fixtures feature.** The fixture count is dropped: v1 xP for the next 1 and 3 GWs already carries fixtures and blanks.
4. **Leakage.**
   - `check leakage` covers the snapshot feature frame (`price_features`).
   - Training-set visibility is enforced by construction and covered by a unit test: rows are built from a view at the fit cutoff, so a target is only present when its deadline price (`player_gw`, available at deadline − 1h) is visible.
   - The xP inputs are v1's, already covered by `model:v1`.
   - Refitting six boosters with v1 at every leakage deadline would take hours and adds nothing beyond these checks.

- [ ] Commit: `docs: price forecast Stage B addendum (as built)`.

---

### Task 1: Snapshot price features (`fplopt-features/src/price.rs`)

**Files:**
- Create: `crates/fplopt-features/src/price.rs`
- Modify: `crates/fplopt-features/src/lib.rs` (module; `("price_features", price::price_features)` in `FRAMES`, not `FEATURES`)

**Output.** `price_features(view) -> Result<Table>` has one row per pool player (`player_pool`'s keys, sorted by `player_key`) with these columns:

| column | type | definition (all as of the view's deadline D, current season only) |
|---|---|---|
| `player_key` | I64 | pool key |
| `price` | F64 | `now_cost` at the newest snapshot (the pool price) |
| `dprice_1d`, `dprice_7d`, `dprice_28d` | F64 | price now − price at the newest snapshot at or before D − d days. NaN if there is none |
| `net_event` | F64 | `transfers_in_event − transfers_out_event` at the newest snapshot |
| `net_event_1d`, `net_event_3d` | F64 | the same at the newest snapshot at or before D − 1 / D − 3 days. NaN if there is none |
| `net_per_own` | F64 | `net_event / (selected_by_percent + 0.1)`. NaN if ownership is null |
| `net_share` | F64 | `net_event / Σ_pool |net_event|` (0 if the sum is 0) |
| `own` | F64 | `selected_by_percent`. NaN if null |
| `price_rel_pos` | F64 | price / median price of the same `element_type` in the pool |
| `status_code` | F64 | `a`=0, `d`=1, `i`=2, `s`=3, `u`=4, `n`=5, else 6 |
| `chance_next` | F64 | `chance_of_playing_next_round`. NaN if null |
| `days_since_news` | F64 | days from `news_added` to D. NaN if null |

**How to read the snapshots:** `view.table("player_snapshot", Some(&["snapshot_at","season","player_key","now_cost","transfers_in_event","transfers_out_event","selected_by_percent","status","chance_of_playing_next_round","news_added","source"]))`.
- Keep the current season (`target(view).0`).
- For each player, sort his rows by `snapshot_at`.
- Where two sources share a timestamp, prefer `fpl`, as `latest` does: look at how `newest_snapshot` breaks ties and mirror it.
- **Players with no snapshot** (seasons before 2021/22): every column but `player_key` and `price` is NaN. `price` comes from `player_pool`.

- [ ] **Step 1: Failing test** in `price.rs` (`#[cfg(test)] mod tests`). Build a `DataStore::from_tables` with:
  - a `gameweek` table (season 2021, gw 1–2, deadlines 2021-08-13 17:30 and 2021-08-21 10:00 UTC);
  - `player_gw` and `player_snapshot` rows for two players (MID 1 and MID 2).
  - Player 1's snapshots:
    - 2021-08-14 00:00: cost 75, in 1,000, out 0, own 10.0;
    - 2021-08-18 00:00: cost 76, in 5,000, out 0, own 12.0;
    - 2021-08-20 12:00: cost 76, in 9,000, out 1,000, own 13.9, status `a`.
  - Player 2's snapshots: cost 50 throughout, net −2,000 at the last snapshot, own 5.0, status `d`, chance 75, `news_added` 2021-08-19 10:00.
  - A snapshot at 2021-08-22 (after the GW2 deadline), cost 99, must not change anything.

  Assert at the GW2 deadline view:
  - player 1: `price` 76, `dprice_1d` 0, `dprice_7d` 1 (D − 7 days = 2021-08-14 10:00, so the 08-14 00:00 snapshot at 75), `net_event` 8,000, `net_event_3d` 5,000, `net_per_own` 8,000/14.0, `net_share` 8,000/10,000;
  - `price_rel_pos`: 76/63 for both, since the median MID price is (76 + 50)/2 = 63;
  - player 2: `status_code` 1, `chance_next` 75, `days_since_news` = (D − 2021-08-19 10:00) in days = 2.0.

  Also write a test that a pre-2021 season (no snapshots) gives NaN everywhere except `price`.
- [ ] **Step 2:** `cargo test -j 2 -p fplopt-features price` and see it fail.
- [ ] **Step 3: Implement** with `util` helpers. No file reads (architecture rule).
- [ ] **Step 4:** run `cargo test -j 2 -p fplopt-features`, clippy and fmt. Then run `cargo run -j 2 --release -p fplopt-cli -- check leakage --deadlines 2` against the main checkout's data (`FPLOPT_DATA_DIR=C:/Users/samin/Documents/GitHub/fpl-optimizer/data`): `price_features` must pass on the edge deadlines. This takes about 20+ minutes because `model:v1` runs too; if `check leakage` accepts a builder filter, use it and say so.
- [ ] **Step 5: Commit** `feat(features): price_features (snapshot price, transfers, ownership, availability)`.

---

### Task 2: Training rows and targets (`fplopt-models/src/price.rs`)

**Files:**
- Create: `crates/fplopt-models/src/price.rs`
- Modify: `crates/fplopt-models/src/lib.rs` (module)

**API:**

```rust
/// Horizons forecast: price change from deadline t to the deadline of gw_index t + h.
pub const HORIZONS: [usize; 6] = [1, 2, 3, 4, 5, 6];
/// Feature columns, in model order (price_features' columns + xP features + calendar).
pub const FEATURES: &[&str] = &[
    "price", "dprice_1d", "dprice_7d", "dprice_28d", "net_event", "net_event_1d", "net_event_3d",
    "net_per_own", "net_share", "own", "price_rel_pos", "status_code", "chance_next",
    "days_since_news", "xp_1", "xp_3", "xp_rank_pos", "gw_index", "days_to_next",
];

/// One deadline's feature rows: price_features(view) joined with v1 xP features
/// (`xp_1` = xP at horizon 0, `xp_3` = Σ horizons 0..3, `xp_rank_pos` = rank of xp_3 within
/// element_type, 1 = best, ties by player_key) and calendar (`gw_index` of the target GW,
/// `days_to_next` = days from this deadline to the next one; NaN at the season's last GW).
pub fn deadline_features(view: &AsOfView, xp: &Table) -> Result<Table>;

/// Training rows for a fit at `cutoff`: for every deadline D of `seasons` with D < cutoff,
/// the features at D plus `dp_h` for each h (deadline price at gw_index(D)+h − price at D,
/// from `player_gw` as visible at the cutoff; NaN when that GW's price isn't visible yet or
/// is past the season's end). `xp_at` returns v1 xP for a deadline view.
pub fn training_rows(
    store: &DataStore,
    cutoff: Micros,
    seasons: &[i64],
    xp_at: &dyn Fn(&AsOfView) -> Result<Table>,
) -> Result<Table>;
```

Rows also carry `season`, `deadline` (Ts) and `player_key`, for evaluation and tests. The deadline price at gw_index g is `player_gw.value` at (season, gw of g), read from `store.as_of(cutoff)`. Map gw_index to gw with `baseline::gameweeks` (make it `pub` in fplopt-features if needed; it's crate-visible today), or with `view.schedule(season)`.

- [ ] **Step 1: Failing tests** with an in-memory store (reuse Task 1's fixture shape, extended to 4 GWs and 2 players, and an `xp_at` closure returning a constant xP table):
  - `targets_are_deadline_price_differences`: a player at 50, 51, 51 and 53 at GW1–4 deadlines gives `dp_1` = 1, `dp_2` = 1, `dp_3` = 3 at GW1, and `dp_1` = 0 at GW2.
  - `no_target_past_the_cutoff`: with the cutoff 1 minute after GW2's deadline:
    - the GW1 rows have `dp_1` (GW2's price, visible from deadline − 1h) and `dp_2` = NaN (GW3's price is not visible yet);
    - the GW2 rows are present (their deadline is before the cutoff), with every `dp_h` NaN;
    - there are no GW3 or GW4 rows.
  - `holdout_is_refused`: `seasons` containing 2025 errors.
- [ ] **Step 2:** see them fail. **Step 3:** implement. **Step 4:** `cargo test -j 2 -p fplopt-models price`, clippy, fmt.
- [ ] **Step 5: Commit** `feat(models): price forecast training rows and targets`.

---

### Task 3: Fit, predict, clip and baselines

**Files:**
- Modify: `crates/fplopt-models/src/price.rs`

**API:**

```rust
pub struct PriceParams { pub gbm: GbmParams }   // objective "regression"
impl PriceParams {
    /// Named presets for the develop grid (at most 6, spec Stage B).
    pub fn preset(name: &str) -> Result<PriceParams>;  // "p1".."p6"
    pub const PRESETS: [&'static str; 6] = ["p1", "p2", "p3", "p4", "p5", "p6"];
}
pub struct PriceFit { pub boosters: Vec<Gbm> /* index h-1 */, pub params_name: String, pub seasons: Vec<i64> }

/// One regression per h on the rows whose dp_h is finite.
pub fn fit_price(rows: &Table, params: &PriceParams) -> Result<PriceFit>;
/// Expected change per player and h, clipped to [-3h, +3h] tenths (prices move at most
/// 0.1 a night, much less on average; spec: ±0.3h £m … in tenths that is ±3h).
pub fn predict_changes(fit: &PriceFit, features: &Table) -> Result<Table>; // player_key, h, dp
/// Baselines: N0 (no change) and N1 (momentum h · (p_t − p_{t−1 GW}), clipped the same way;
/// p_{t−1 GW} = the previous deadline's price, from player_gw; 0 at a season's first GW).
pub fn baseline_changes(view: &AsOfView, keys: &[i64]) -> Result<(Table, Table)>;
```

**Clipping units:** the spec writes the bound as "[p_t − 0.3h, p_t + 0.3h]". Prices are in tenths of £m, and 0.3 per GW-night budget means 3 tenths per GW. Use ±3h tenths, and record this reading in a doc comment.

**Presets:** all with `objective: "regression"`, `num_boost_round` 300, `feature_fraction` 0.9, `lambda_l2` 1.0, varying:

| preset | learning_rate | num_leaves | min_data_in_leaf |
|---|---|---|---|
| p1 | 0.05 | 15 | 200 |
| p2 | 0.05 | 31 | 200 |
| p3 | 0.05 | 63 | 500 |
| p4 | 0.02 | 31 | 200 |
| p5 | 0.1 | 15 | 500 |
| p6 | 0.05 | 31 | 1000 |

- [ ] **Step 1: Failing tests** on a synthetic table, 2,000 rows, seeded with `fplopt_core::nprng`:
  - with `dp_1 = 2 · sign(net_share)` and noise, p1's predictions have the sign of `net_share` on 90% or more of held-out rows;
  - the clip holds: a hand-made row with an extreme feature can't exceed ±3h;
  - N1 on a hand-made player with prices 50 then 52 at the last two deadlines predicts `dp_h` = min(2h, 3h) = 2h.
- [ ] **Steps 2–4:** fail, implement, then run tests, clippy and fmt.
- [ ] **Step 5: Commit** `feat(models): price forecast fit, predict and baselines`.

---

### Task 4: `fplopt-rs research price-eval` (walk-forward evaluation)

**Files:**
- Create: `crates/fplopt-backtest/src/price_eval.rs`
- Modify: `crates/fplopt-backtest/src/lib.rs`, `crates/fplopt-backtest/src/eval.rs` (make `diebold_mariano` and `candidate_keys` `pub(crate)`), `crates/fplopt-cli/src/research.rs` (subcommand), `CLAUDE.md` (one line)

**Command:**

`fplopt-rs research price-eval --seasons 2022 --params p1 [--first-fit 2021] [--jobs N] [--xp-cache DIR | --no-xp-cache] [--family NAME] [--experiments PATH] --out DIR`

**What it does, for each evaluated season S (refuse 2025, and any S ≤ `--first-fit`):**

1. **Fit.** `training_rows(store, cutoff = S's first deadline, seasons = first_fit..S−1 excluding 2025, xp_at = Caches::xp("v1", view))`, then `fit_price(rows, preset)`.
2. **Predict.** At every deadline D of S (`season_schedule(store, S, 1)`):
   - build `deadline_features(view, xp)`;
   - forecast with `predict_changes`, and the baselines with `baseline_changes`;
   - take the truth `dp_h` from `player_gw` (read from `store.as_of(<after the season>)`; this is evaluation, not a model input);
   - restrict to the candidates: `candidate_keys(&[xp], DEFAULT_PRUNE_N)`.
3. **Write** `predictions.parquet` with `season, deadline, player_key, h, truth, forecast, n0, n1`.
4. **Write** `metrics.json`:
   - MSE per h and pooled over h = 1..6 for forecast, N0 and N1;
   - one-sided Diebold–Mariano tests, forecast vs N0 and forecast vs N1, on per-deadline mean squared error pooled over h, at lag 6, clustered by deadline (`eval::diebold_mariano`);
   - the number of deadlines, the number of rows, the params name and the fit seasons.
5. **Log** to the experiment log (`log_experiment`), default family `price forecast`, with `n_variants` from `family_variants`, as `models eval` does.

**`--jobs`:** seasons run in parallel, with results identical to `--jobs 1`.

- [ ] **Step 1: Failing tests** in `price_eval.rs`:
  - the metric arithmetic on hand-made prediction rows (MSE per h and pooled; the DM input rows are per deadline);
  - seasons ≤ first-fit and 2025 are refused.
- [ ] **Steps 2–4:** fail, implement, test. Smoke on the main checkout's data: `--seasons 2022 --params p1 --first-fit 2021 --jobs 1` into the scratchpad (`C:\Users\samin\AppData\Local\Temp\claude\6ff2f6c53ca963aa\50295afa-4ab0-4ac1-a579-1e2ddcc3dc5e\scratchpad\price-eval-smoke`). Report the pooled MSEs, DM p-values and run time. The first v1 xP pass over 2021–2022 may take 20–40 minutes. If it is much longer, stop and report.
- [ ] **Step 5: Commit** `feat(rust): research price-eval, walk-forward price forecast evaluation`.

---

### Task 5: Develop grid and freezing the settings (controller)

- [ ] **Develop run.** On the server, in a separate worktree as before, after asking the user: run `research price-eval --seasons 2022 --first-fit 2021 --params pX` for p1–p6, family `price forecast develop` (6 variants).
- [ ] **Choose.** Pick the preset with the lowest pooled forecast MSE on 2022/23, and record the choice and all six results.
- [ ] **Develop sanity check (not a gate).** The chosen preset should beat N0 and N1 on 2022/23. If it doesn't, report to the user before Gate B. A forecast that can't beat the baselines in-sample won't pass out of sample.
- [ ] **Freeze.** Write `config/prices/forecast.json` with `{"params": "<preset>", "first_fit": 2021, "horizons": [1,2,3,4,5,6], "clip_tenths_per_gw": 3}`, commit it, and tag the commit `price-forecast-gate-b`.

---

### Task 6: Gate B (controller, run once)

- [ ] **Run once,** from the tagged commit: `research price-eval --seasons 2023,2024 --first-fit 2021 --params <frozen> --family "price forecast gate b"`. The 2023 fit uses 2021–2022; the 2024 fit uses 2021–2023.
  - If the experiment log already holds family `price forecast gate b`, do not rerun.
  - The log row is written before the result is read.
- [ ] **Pass rule:** pooled MSE below N0's and N1's, and one-sided DM p < 0.10 against each.
- [ ] **Write-up:** `docs/research/2026-11-price-forecast-gate-b.md`: per-h and pooled MSEs, the DM tests, develop vs check, and the decision. Add a PLAN §12 row and the experiment-log rows, then commit.
- [ ] **Next:** on a pass, write Stage C's plan (`prices=forecast` in the planner, the k grid {0, 0.25, 0.5, 1}, and the pre-registered backtest). On a fail, record the stop in PLAN §1's price line.
