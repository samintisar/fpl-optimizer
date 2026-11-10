# Value of Information Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** backtest-only xP models `v1+oracle-<ARM>` that replace one v1 input with the truth, six develop runs against the live planner, and a note that says which input is worth a source.

**Spec:** `docs/superpowers/specs/2026-11-09-value-of-information-design.md` (approved by the owner in chat, 2026-11-09). Branch `claude/fpl-optimizer-review-455d79`.

**Architecture:**
- `fplopt-models` gets one pure hook: `predict_v1_with(view, fit, edit)`, where `edit: &dyn Fn(&mut [Fixture])` runs after `calibrate_inputs` and before `fixture_points`. It reads nothing. `predict_v1` is that with a no-op.
- `fplopt-backtest` gets `info_oracle.rs`, next to `oracle.rs`. It loads a season's truth once (`store.as_of(i64::MAX)`, as `FuturePrices::load` does) and builds the `edit` for an arm.
- `Caches::xp` recognises `v1+oracle-<ARM>`: v1's memoized fit, then `predict_v1_with`. The disk cache stores it under its own directory, so v1's cache is untouched.
- The specs accept the names for backtest policies only. Runs go through the existing `backtest run`; a small `dev/info_value.py` pairs the arms with the base and writes the table.

**Tech stack:** Rust (`fplopt-models`, `fplopt-backtest`, `fplopt-cli`), Python + pandas + scipy for the driver.

**Facts (checked 2026-11-09):**
- `Fixture` (`crates/fplopt-models/src/assemble.rs:202-240`) has `key: minutes::Key { player_key, fixture_key, team_key, season, gw, gw_index, horizon }`, `p_start`, `p_60`, `p_sub`, `p_play`, `e_minutes`, `c60`, `csub`, `m_start`, `m_sub`, `lambda_for`, `lambda_against`, `p_cs`, `e_np_goals`, `e_pen_goals`, `e_assists`.
- `predict_fixtures` (`assemble.rs:448-463`) is `fixture_components` → `calibrate_inputs` → `fixture_points` → `calibrate_xp`. `calibrate_inputs` (`:311-347`) shows the rescale to copy: new `e_minutes = p·m_start + p_sub·m_sub`, and `e_np_goals`, `e_pen_goals`, `e_assists` multiplied by new/old `e_minutes`.
- `Caches::xp` (`crates/fplopt-backtest/src/simulator.rs:111-140`) keys frames by (model, deadline, cutoff) and fits by the fit view; the disk path is `<dir>/<model>/<deadline>.parquet` (`xp_cache.rs:104-111`). `+` and `-` are valid in Windows and Linux directory names; `:` is not, and the spec grammar splits on it. Hence `v1+oracle-M0`, not the spec draft's `v1+oracle:M0`.
- `XP_MODELS` and `MINUTES_XP_MODELS` (`specs.rs:17-18`) gate model names; the planner's minutes keys need a model in `MINUTES_XP_MODELS`.
- `LEAKAGE_EXCLUDED` (`oracle.rs:17`) is printed by `check leakage` (`crates/fplopt-cli/src/main.rs:423`).
- Truth: `player_match` has `minutes` for every season and `starts` from 2022/23 gameweek 16 (0 non-null rows before 2022, 18,014 of 26,505 in 2022). `fplopt_features::history::infer_starts(fixture, team, minutes, player)` (`history.rs:125`) gives the inferred starts. `us_npxg`, `us_xa` cover 2019/20+ well. `team_match.goals_for` is complete.
- Closing odds (`odds_snapshot.is_closing`) exist for every develop season: 1,140 rows a season to 2018/19, about 5,320 from 2019/20.
- A release binary of `feat/price-sizing` exists at `../v1r/target/release/fplopt-rs.exe` with a warm xP cache nearby; this worktree has no `target/` and its submodules are not initialised (`git submodule update --init --recursive` first, CMake and a C++ compiler needed).

**Rules:**
- Never read 2025/26. Every loader refuses `HOLDOUT_SEASONS`.
- The oracle names are refused by `optimize plan`, `live`, `bot`, `models eval` and `models dump`; only backtest specs accept them.
- With no oracle name in a spec, every output is byte-identical: goldens and the backtest golden must pass unchanged.
- Cargo with `-j 2` locally. `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --all` before every commit.
- No push and no server use without the owner's go-ahead. Task 8 needs it.
- Commits end with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`.

---

### Task 1: the hook in `fplopt-models`

**Files:** modify `crates/fplopt-models/src/assemble.rs`.

- [ ] **Step 1: failing test.** In `assemble.rs` tests: with a hand-made `Vec<Fixture>` helper already used there (or the smallest existing fixture-level test), assert that `predict_fixtures_with(view, fit, &|_| {})` equals `predict_fixtures(view, fit)`, and that an edit setting `p_play = 0` on one fixture gives that fixture `xp == 0` (appearance, goals and clean-sheet terms all scale with `p_play`/`p_60`; set `p_60`, `e_np_goals`, `e_pen_goals`, `e_assists`, `e_minutes` to 0 in the same edit). If no view-free test seam exists, test `fixture_points` directly on two `Fixture`s and keep the equality test under `#[ignore]` with the real-data tests.
- [ ] **Step 2: see it fail** (`cargo test -j 2 -p fplopt-models assemble`).
- [ ] **Step 3: implement.**
  ```rust
  /// `predict_fixtures` with `edit` applied to the calibrated inputs before the points are
  /// assembled. `edit` must not read data: callers that pass the truth are backtest-only
  /// oracles (`fplopt_backtest::info_oracle`).
  pub fn predict_fixtures_with(
      view: &AsOfView,
      fit: &V1Fit,
      edit: &dyn Fn(&mut [Fixture]),
  ) -> Result<Vec<Fixture>> { /* body of predict_fixtures, edit(&mut frame) after calibrate_inputs */ }

  pub fn predict_fixtures(view: &AsOfView, fit: &V1Fit) -> Result<Vec<Fixture>> {
      predict_fixtures_with(view, fit, &|_| {})
  }

  pub fn predict_v1_with(view: &AsOfView, fit: &V1Fit, edit: &dyn Fn(&mut [Fixture])) -> Result<Table> {
      gw_frame(view, &predict_fixtures_with(view, fit, edit)?)
  }
  ```
  Also `pub fn set_play(f: &mut Fixture, p_start: f64, p_play: f64)`: sets `p_start`, `p_sub = p_play − p_start`, `p_play`, `p_60 = p_start · c60`, `e_minutes = p_start · m_start + p_sub · m_sub`, and multiplies `e_np_goals`, `e_pen_goals`, `e_assists` by new/old `e_minutes` (1 when old is 0 and new is 0; when old is 0 and new > 0 leave the expectations at 0 and count the fixture in a returned `bool` so the caller can report how many played rows had no rates). Unit-test it on one `Fixture`.
- [ ] **Step 4:** workspace tests, clippy, fmt. The models golden and `lgbm` golden must be unchanged.
- [ ] **Step 5: commit** `feat(models): predict_v1_with, an edit hook on v1's calibrated inputs`.

### Task 2: the truth and the minutes arms (`info_oracle.rs`)

**Files:** create `crates/fplopt-backtest/src/info_oracle.rs`; modify `crates/fplopt-backtest/src/lib.rs`, `oracle.rs` (`LEAKAGE_EXCLUDED` gains `("xp=v1+oracle-*", "backtest-only value-of-information bound")`).

**Design:**
```rust
pub enum Arm { M0, M0h, MA, A0, T0, R }
impl Arm { pub fn parse(model: &str) -> Option<Arm>; pub fn model(&self) -> &'static str; }  // "v1+oracle-M0" …
pub const MODELS: [&str; 6];

/// One season's truth: (player_key, fixture_key) -> (played, started). **Reads the future.**
pub struct Truth { play: HashMap<(i64, i64), (bool, bool)>, /* Task 6, 7 add fields */ }
impl Truth {
    pub fn load(store: &DataStore, season: i64) -> Result<Truth>;   // refuses HOLDOUT_SEASONS
    pub fn edit(&self, arm: Arm) -> impl Fn(&mut [Fixture]) + '_;
}
```
- `load` reads `player_match` (`season`, `player_key`, `fixture_key`, `team_key`, `minutes`, `starts`) from `store.as_of(i64::MAX)`. `played = minutes > 0`. `started = starts == 1` where `starts` is non-null, else `infer_starts` over the season's rows.
- A fixture with no `player_match` row for the player: `played = started = false` (he was not in the squad), **except** when the fixture has no rows at all for his club (postponed after the deadline: the final schedule's known leak) → leave v1's values.
- Arms, per `Fixture f` with truth `(played, started)`:
  - **M0:** if `f.key.horizon == 0`: `set_play(f, started as f64, played as f64)`.
  - **MA:** the same at every horizon.
  - **A0:** if horizon 0: `played` false → `set_play(f, 0, 0)`; `played` true → `set_play(f, f.p_start / f.p_play, 1.0)` (v1's start share given he plays; `p_play == 0` → `set_play(f, 0.5, 1.0)` and counted).
  - **M0h:** M0 for the players with `splitmix64(player_key ⊕ gw_index·0x9E37…) & 1 == 0`. Deterministic, no RNG state.
- [ ] **Step 1: failing tests** on a hand-made `Truth` (`Truth::from_rows` under `#[cfg(test)]`) and two or three `Fixture`s: each arm's effect at horizon 0 and 1; no row → not playing; club with no rows → unchanged; M0h splits a 1,000-player list within 45–55% and gives the same answer twice; `load` refuses 2025.
- [ ] **Step 2: see them fail. Step 3: implement.**
- [ ] **Step 4:** workspace tests, clippy, fmt.
- [ ] **Step 5: commit** `feat(backtest): info_oracle, truth for who plays and the M0/MA/A0/M0h arms`.

### Task 3: wire into `Caches::xp` and the specs

**Files:** modify `crates/fplopt-backtest/src/simulator.rs` (`Caches`), `specs.rs`, `crates/fplopt-cli/src/{models.rs,models_eval.rs,optimize_plan.rs,live.rs}` only if they route names through `XP_MODELS` (check; they should keep using `fplopt_models::MODELS`, which is not extended).

- `Caches` gains `truth: OnceMap<i64, Truth>` and needs the `DataStore`: `Caches::xp` takes only the view today, so add `Caches::xp_for(&self, store: &DataStore, model, view)` that handles oracle names (v1's fit from `self.fits` under the model key `"v1"`, then `predict_v1_with(view, &fit, &truth.edit(arm))`, disk cache under the oracle's own name) and delegates everything else to `xp`. Call sites that have a store (`decide_step`, `money_value::solve_at`, the per-decision replay) switch to `xp_for`; `xp` itself bails on an oracle name with "needs Caches::xp_for".
- `specs.rs`: `XP_MODELS` stays as is. `make_policy_spec` accepts `xp` when `XP_MODELS.contains(xp) || info_oracle::Arm::parse(xp).is_some()`, and treats oracle names as minutes models (`MINUTES_XP_MODELS` check). Refuse an oracle name with `prices` other than fixed, `early=1` and chips (keep the study's arms clean).
- [ ] **Step 1: failing tests.** `specs.rs`: `optimizer:v1+oracle-M0:horizon=6,decay=0.75,max_hits=1,hit_margin=2` builds and its label contains `v1+oracle-M0`; `optimizer:v1+oracle-ZZ` errors with the model list; `…,chips=1` and `…,prices=oracle` error. A `fplopt-cli` test (or the existing refusal tests' pattern) that `optimize plan --xp v1+oracle-M0` is refused.
- [ ] **Step 2–3.**
- [ ] **Step 4:** workspace tests, clippy, fmt; backtest golden unchanged. **Smoke (local, needs a build):** `backtest run --seasons 2022 --spec optimizer:v1+oracle-M0:horizon=6,decay=0.75,max_hits=1,hit_margin=2 --starts template@1 --jobs 1 --experiments <scratch>/exp.csv --out <scratch>/voi-smoke-m0` and the same with plain `v1`, sharing an `--xp-cache`. Report both season totals, XI regret and run time. M0's XI regret should drop sharply (it knows who plays); if it does not, stop and debug before any server run.
- [ ] **Step 5: commit** `feat(backtest): v1+oracle-* xP models in backtest specs`.

### Task 4: the leak guard

**Files:** modify `crates/fplopt-backtest/src/info_oracle.rs` (tests), `crates/fplopt-cli/src/main.rs` only if the exclusion list needs a second source.

- [ ] A real-data test (`#[ignore]`, as `night_real`): for 2022/23 gameweek 10, the M0 frame equals v1's frame on every column at horizons ≥ 1, and differs at horizon 0 only in rows whose `p_play` changed. MA differs at later horizons too. This pins that M0 leaks nothing beyond the coming gameweek.
- [ ] `check leakage` prints the new exclusion line; no `model:` or `probe:` entry is added for oracle names.
- [ ] **Commit** `test(backtest): the oracle arms change only what they say they change`.

### Task 5: `dev/info_value.py`

**Files:** create `dev/info_value.py`, `tests/test_info_value.py`.

- `report --base RUN --arm NAME=RUN ... --out DIR`: per arm, paired by (season, start_id) against the base, the season net-points difference; mean over units, mean over season means with an 80% t interval (df = seasons − 1), the lower end; the same split by start gameweek (`@1`/`@2` vs `@20`); per arm transfers, hit points, XI regret and captain regret a season. Copy `units` from `dev/price_gate_a.py`.
- The decision column from the spec's table: lower end ≥ 60 → `source`, 20–60 → `noted`, < 20 → `closed`.
- Writes `info_value.json` and prints the table. No logging of its own: each `backtest run` logs its row under family `value of information`.
- [ ] Tests on synthetic frames: pairing, the interval, the three decision boundaries (60 → source, 20 → noted), an arm with fewer seasons (R) reported with its own n.
- [ ] `uv run pytest tests/test_info_value.py -q`, ruff. **Commit** `feat(dev): value-of-information report`.

### Task 6: arm R (true rates)

**Files:** modify `info_oracle.rs`.

- `Truth` gains per (player_key): season `us_npxg`, `us_xa`, minutes (rows with non-null `us_npxg`), and per team the season mean of `team_match.goals_for`.
- **R**, at every horizon, for players with ≥ 450 season minutes with xG: `e_np_goals = (npxg / minutes · 90) · (e_minutes / 90) · (lambda_for / team_mean_goals)`; `e_assists = (xa / minutes · 90) · K_A · (e_minutes / 90) · (lambda_for / team_mean_goals)`, with `K_A` = the season's ratio of FPL assists to Understat xA over all players (FPL awards assists xA does not count; PLAN §6.2). Penalty goals stay v1's. Others unchanged.
- Refused for seasons before 2019 (xG coverage 47–75%).
- This is a rate-times-difficulty stand-in for the shares model, not the shares model with true inputs. The note must say so: R measures knowing each player's level, with a cruder fixture adjustment than v1's.
- [ ] Failing tests (formula on a hand-made truth; < 450 minutes untouched; 2018 refused), implement, workspace tests.
- [ ] **Sanity (local):** on 2022/23, the sum of R's `e_np_goals` over all player-fixtures at horizon 0 is within 10% of v1's (the level is preserved; only its split between players changes). If not, fix the scaling before any run.
- [ ] **Commit** `feat(backtest): oracle arm R, true season scoring rates`.

### Task 7: arm T0 (closing odds), investigate first

- [ ] **Read** `fplopt_models::team::team_lambdas` (`team.rs:937`) and how it picks odds rows. Wanted: the market-implied λ of the target gameweek's fixtures from the `is_closing` rows, with the fit unchanged.
- [ ] **If** it can be had by calling `team_lambdas` on a view whose cutoff is just after the gameweek's last kickoff **and** that call reads nothing but odds and the fit (no results, no later ratings): `Truth` caches that table per (season, gw), and T0 overwrites `lambda_for`, `lambda_against`, `p_cs` at horizon 0, rescaling `e_np_goals`, `e_pen_goals`, `e_assists` by new/old `lambda_for`. Tests as Task 2; a real-data test that fixtures without closing rows are unchanged.
- [ ] **Otherwise** T0 is dropped: record why in the note and remove it from `Arm`. It is the arm with the smallest expected ceiling, so the study stands without it.
- [ ] **Commit** either way.

### Task 8: the runs (controller; server, after the owner's go-ahead)

- [ ] **Setup:** a separate server worktree at this branch, never `~/fpl-optimizer`; `cargo build --release -p fplopt-cli`; the shared xP cache for `v1`.
- [ ] **Base:** `backtest run --seasons 2016-2022 --spec optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2 --starts template@1,random:3@1,template@20,random:3@20 --jobs 14 --family 'value of information' --out results/voi/base`.
- [ ] **Arms:** the same with `v1+oracle-M0`, `-M0h`, `-MA`, `-A0`, `-T0` (if built), and `-R` on `--seasons 2019-2022`. About 20–25 minutes each.
- [ ] **Before reading any total,** check each run's log for the count of played rows with no rates and of fixtures left unchanged; more than 2% of horizon-0 rows in either means a join is wrong.
- [ ] **Report:** `uv run python dev/info_value.py report --base results/voi/base --arm M0=… … --out results/voi`.

### Task 9: the note

- [ ] Write `docs/research/2026-11-value-of-information.md`: the table, the three comparisons the spec names (M0 vs A0, M0 vs MA, M0 vs M0h), each arm's decision, and what was not built (T0 if dropped, R's approximation).
- [ ] Add a PLAN §12 row and copy the experiment-log rows. If any arm is `source`, name the first source to try and stop there: the sourcing spec is the owner's call.
- [ ] Commit. Push only on the owner's go-ahead.

---

## Spec deviations (recorded before any code)

1. **Model names** are `v1+oracle-<ARM>`, not `v1+oracle:<ARM>`: the spec grammar splits on `:` and `:` is not valid in the cache's directory names on Windows.
2. **No `models eval` rows.** The spec asked for each arm's xP error removed. `models eval` takes its names from `fplopt_models::MODELS`, and extending that list would let an oracle name reach non-backtest commands. The report gives XI and captain regret from the backtest runs instead.
3. **R is an approximation** (Task 6) and **T0 may be dropped** (Task 7).
