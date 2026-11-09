# Early Moves Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Policy E (the live planner with `early=1`) makes its transfers at the previous gameweek's lockdown, at that moment's data and prices, and picks its captain and XI at the deadline; a driver measures its value gain, information cost, disagreement and news risk against the live planner on 2021/22–2022/23.

**Architecture:** The store's view gets a cutoff separate from its deadline, so a view can be "for the deadline of GW g, seeing only what was known at time A". The caches key on (deadline, cutoff). The simulator's `decide_step` runs an `early` optimizer on that view, re-solves the lineup at the deadline with the transfers fixed, and tags the decision with `priced_at = A` so it is applied (and replayed in per-decision arms) at A's prices. A Python driver in `dev/` runs E's backtest and the standard comparison and computes the report.

**Tech Stack:** Rust (`crates/fplopt-store`, `crates/fplopt-backtest`, `crates/fplopt-cli`), Python 3.12 driver with pandas and scipy, pytest.

**Spec:** `docs/superpowers/specs/2026-11-07-early-moves-design.md` (read it first). Standing rules: new production code in Rust only (nothing new in `src/fplopt`); never read the 2025/26 holdout; don't use 2023/24–2024/25; `raw/` is append-only. Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

**Worktree:** `C:\Users\samin\Documents\GitHub\fpl-optimizer\.claude\worktrees\v1r`, branch `feat/price-sizing`. The worktree has no `data/`; local runs set `FPLOPT_DATA_DIR=C:/Users/samin/Documents/GitHub/fpl-optimizer/data` (2021/22–2022/23 are complete there). Use plain git commands, one per call.

---

## File structure

| File | Change |
|---|---|
| `crates/fplopt-store/src/lib.rs` | `AsOfView` gets `cutoff`; `as_of_early`, `for_deadline`, `cutoff()`; visibility on the cutoff |
| `crates/fplopt-store/src/tests.rs` | early-view tests |
| `crates/fplopt-backtest/src/xp_cache.rs` | disk path keyed by (deadline, cutoff) |
| `crates/fplopt-backtest/src/simulator.rs` | caches keyed by (deadline, cutoff); `previous_lockdown`, `early_kept`; early path in `decide_step`; `Step.early_kept`; `GwRow.decided_at`, `GwRow.early_kept` |
| `crates/fplopt-backtest/src/state.rs` | `Decision.priced_at` (not part of equality) |
| `crates/fplopt-backtest/src/policies.rs` | `Policy::Optimizer.early`; name label; `deadline_lineup` |
| `crates/fplopt-backtest/src/specs.rs` | `early` key, default, refusals, test |
| `crates/fplopt-backtest/src/probes.rs` | `early: false` in `optimizer()`; `probe:early_optimizer_ep_next_template` |
| `crates/fplopt-backtest/src/evaluate.rs` | the per-decision "same" shortcut also needs equal `priced_at` |
| `crates/fplopt-cli/src/backtest.rs` | `decided_at`, `early_kept` columns |
| `crates/fplopt-cli/src/main.rs` | `check leakage --only PREFIX` |
| `dev/backtest_parity.py` | the two new columns in `RUST_ONLY` |
| `dev/early_moves.py` | driver: `run`, `report` |
| `tests/test_early_moves.py` | driver's pure functions |
| `CLAUDE.md` | one line for `early=1` and the driver |

---

### Task 0: Baseline binary for the "early=0 changes nothing" check

**Files:** none (scratch only).

- [ ] **Step 1: Build the current binary and keep a copy**

Run from the worktree:
```bash
cargo build --release -p fplopt-cli
```
```bash
cp target/release/fplopt-rs.exe "C:/Users/samin/AppData/Local/Temp/claude/6ff2f6c53ca963aa/50295afa-4ab0-4ac1-a579-1e2ddcc3dc5e/scratchpad/fplopt-rs-base.exe"
```
Expected: `Finished release` and the copy exists.

- [ ] **Step 2: Record the baseline run** (ep_next xP: no model fitting, a few minutes)

```bash
FPLOPT_DATA_DIR=C:/Users/samin/Documents/GitHub/fpl-optimizer/data "C:/Users/samin/AppData/Local/Temp/claude/6ff2f6c53ca963aa/50295afa-4ab0-4ac1-a579-1e2ddcc3dc5e/scratchpad/fplopt-rs-base.exe" backtest run --spec optimizer:ep_next:horizon=6,decay=0.75,max_hits=1,hit_margin=2 --seasons 2022 --starts template@1,random:1@20 --jobs 4 --no-xp-cache --family "early smoke" --experiments "C:/Users/samin/AppData/Local/Temp/claude/6ff2f6c53ca963aa/50295afa-4ab0-4ac1-a579-1e2ddcc3dc5e/scratchpad/early-smoke/exp.csv" --out "C:/Users/samin/AppData/Local/Temp/claude/6ff2f6c53ca963aa/50295afa-4ab0-4ac1-a579-1e2ddcc3dc5e/scratchpad/early-smoke/base"
```
Expected: `Written to …/early-smoke/base` with `gws.parquet`. (The log goes to scratch, not `results/experiments.csv`.)

---

### Task 1: Store — a view's cutoff apart from its deadline

**Files:**
- Modify: `crates/fplopt-store/src/lib.rs` (module doc line 5; `as_of` ~352; `AsOfView` ~375–416; `latest` ~464; `schedule` ~510)
- Test: `crates/fplopt-store/src/tests.rs`

- [ ] **Step 1: Write the failing tests** (append to `crates/fplopt-store/src/tests.rs`)

```rust
#[test]
fn an_early_view_is_for_its_deadline_but_sees_only_rows_before_its_cutoff() {
    let rows = Table::new()
        .with("player_key", ints(&[1, 2, 3]))
        .with("fixture_key", ints(&[10, 10, 10]))
        .with("available_at", ts(&[1, 3, 5]));
    let gameweeks = Table::new()
        .with("season", ints(&[2022, 2022]))
        .with("gw", ints(&[1, 2]))
        .with("gw_index", ints(&[1, 2]))
        .with("deadline_time", Column::OptI64(vec![Some(2 * H), Some(6 * H)]))
        .with("available_at", ts(&[0, 0]));
    let s = store(vec![("player_match", rows), ("gameweek", gameweeks)]);
    let early = s.as_of_early(6 * H, 4 * H).unwrap();
    assert_eq!(early.deadline(), 6 * H);
    assert_eq!(early.cutoff(), 4 * H);
    assert!(early.is_early());
    // visibility follows the cutoff ...
    let seen = early.table("player_match", Some(&["player_key"])).unwrap();
    assert_eq!(seen.i64("player_key").unwrap(), &[1, 2]);
    // ... the gameweek follows the deadline
    assert_eq!(early.gameweek_for_deadline().unwrap(), (2022, 2));
    // as_of is the cutoff = deadline case
    let full = s.as_of(6 * H);
    assert_eq!(full.cutoff(), 6 * H);
    assert!(!full.is_early());
    assert_eq!(full.table("player_match", None).unwrap().height(), 3);
    // a deadline before the cutoff is refused
    assert!(s.as_of(4 * H).for_deadline(3 * H).is_err());
    // earlier() never widens what the view sees
    let e = early.earlier(5 * H).unwrap();
    assert_eq!((e.deadline(), e.cutoff()), (5 * H, 4 * H));
    let e = early.earlier(2 * H).unwrap();
    assert_eq!((e.deadline(), e.cutoff()), (2 * H, 2 * H));
    assert!(early.earlier(7 * H).is_err());
}

#[test]
fn latest_on_an_early_view_takes_the_newest_snapshot_before_the_cutoff() {
    let rows = Table::new()
        .with("snapshot_at", ts(&[1, 3, 5]))
        .with("source", strs(&["fpl", "fpl", "fpl"]))
        .with("player_key", Column::OptI64(vec![Some(7), Some(7), Some(7)]))
        .with("now_cost", ints(&[50, 51, 52]))
        .with("available_at", ts(&[1, 3, 5]));
    let s = store(vec![("player_snapshot", rows)]);
    let early = s.as_of_early(6 * H, 4 * H).unwrap();
    let out = early.latest("player_snapshot", &["player_key"], Some(&["now_cost"])).unwrap();
    assert_eq!(out.i64("now_cost").unwrap(), &[51]);
}
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `cargo test -p fplopt-store early`
Expected: compile errors (`as_of_early`, `cutoff`, `is_early` not found).

- [ ] **Step 3: Implement**

In `crates/fplopt-store/src/lib.rs`:

Module doc (line 5): replace "A view exposes only rows with `available_at < deadline`" with "A view exposes only rows with `available_at < cutoff`, where the cutoff is the deadline for `as_of(deadline)` and earlier for an early view (`as_of_early`, the early-move study)". Keep the rest of the sentence.

Replace `as_of` and add `as_of_early`:
```rust
    pub fn as_of(&self, deadline: Micros) -> AsOfView<'_> {
        AsOfView { store: self, deadline, cutoff: deadline }
    }

    /// The view for `deadline`'s gameweek as known strictly before `cutoff` (≤ deadline): the
    /// early-move study's decision at the previous lockdown.
    pub fn as_of_early(&self, deadline: Micros, cutoff: Micros) -> Result<AsOfView<'_>> {
        self.as_of(cutoff).for_deadline(deadline)
    }
```

Replace the struct and the first methods:
```rust
/// The data as known strictly before `cutoff` (µs since the epoch, UTC), for the gameweek
/// whose deadline is `deadline` (cutoff ≤ deadline; equal for `as_of`).
#[derive(Clone, Copy)]
pub struct AsOfView<'a> {
    store: &'a DataStore,
    deadline: Micros,
    cutoff: Micros,
}

impl<'a> AsOfView<'a> {
    pub fn deadline(&self) -> Micros {
        self.deadline
    }

    /// What the view sees: rows with available_at < cutoff.
    pub fn cutoff(&self) -> Micros {
        self.cutoff
    }

    /// Whether the view sees less than its deadline would (an early-move view).
    pub fn is_early(&self) -> bool {
        self.cutoff < self.deadline
    }

    /// The view for a later `deadline` that sees exactly what this one sees.
    pub fn for_deadline(&self, deadline: Micros) -> Result<AsOfView<'a>> {
        if deadline < self.cutoff {
            bail!("deadline {deadline} is before the view's cutoff {}", self.cutoff);
        }
        Ok(AsOfView { store: self.store, deadline, cutoff: self.cutoff })
    }

    /// The same store as of `cutoff` (walk-forward fits); a cutoff after this view's deadline
    /// fails, and the result never sees more than this view, so nothing derived from a view
    /// sees more than the view.
    pub fn earlier(&self, cutoff: Micros) -> Result<AsOfView<'a>> {
        if cutoff > self.deadline {
            bail!("cutoff {cutoff} is after the view's deadline {}", self.deadline);
        }
        Ok(AsOfView { store: self.store, deadline: cutoff, cutoff: cutoff.min(self.cutoff) })
    }
```

Replace `self.deadline` with `self.cutoff` in the three visibility calls: `table.visible(self.deadline)` in `table` (~line 408) and `latest` (~line 464), and `snapshots.visible(self.deadline)` in `schedule` (~line 510). Update the doc comments of `table` ("Rows with available_at < cutoff"), `latest` ("the newest snapshot before the cutoff") and `schedule` ("before the cutoff"). `gameweek_for_deadline` keeps `self.deadline`.

- [ ] **Step 4: Run the store tests and the workspace build**

Run: `cargo test -p fplopt-store`
Expected: all pass, including the two new tests and the existing `table_keeps_rows_strictly_before_the_deadline_in_available_order`.

Run: `cargo build --workspace`
Expected: builds (no caller constructs `AsOfView` directly).

- [ ] **Step 5: Commit**

```bash
git add crates/fplopt-store/src/lib.rs crates/fplopt-store/src/tests.rs
```
```bash
git commit -m "feat(store): early views: a cutoff apart from the deadline" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Caches keyed by (deadline, cutoff)

**Files:**
- Modify: `crates/fplopt-backtest/src/xp_cache.rs` (`path`, `load`, `store`, tests)
- Modify: `crates/fplopt-backtest/src/simulator.rs` (`Caches` fields and `pool`, `xp`, ~lines 71–133)

- [ ] **Step 1: Write the failing test** (append inside `mod tests` of `xp_cache.rs`; update the two existing tests' calls to the new signature: `cache.load("v1", 42, 42)`, `cache.store("v1", 42, 42, &t)`, `cache.load("v1", 43, 43)`, `cache.load("rolling", 42, 42)`, `cache.store("v1", 7, 7, &frame())`, `cache.load("v1", 7, 7)`)

```rust
    #[test]
    fn an_early_frame_has_its_own_file_and_a_deadline_frame_keeps_its_name() {
        let dir = scratch("early");
        let cache = DiskCache::at(dir.clone());
        cache.store("v1", 42, 42, &frame()).unwrap();
        assert!(dir.join("v1").join("42.parquet").exists());
        assert!(cache.load("v1", 42, 40).unwrap().is_none());
        cache.store("v1", 42, 40, &frame()).unwrap();
        assert!(dir.join("v1").join("42-at-40.parquet").exists());
        assert!(cache.load("v1", 42, 40).unwrap().is_some());
    }
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `cargo test -p fplopt-backtest xp_cache`
Expected: compile error (wrong number of arguments).

- [ ] **Step 3: Implement**

In `xp_cache.rs` (module doc: add "an early-move frame, seen at `cutoff` < deadline, is `<deadline>-at-<cutoff>.parquet`"):
```rust
    fn path(&self, model: &str, deadline: Micros, cutoff: Micros) -> PathBuf {
        let name = if cutoff == deadline {
            format!("{deadline}.parquet")
        } else {
            format!("{deadline}-at-{cutoff}.parquet")
        };
        self.dir.join(model).join(name)
    }
```
Change `load(&self, model: &str, deadline: Micros)` to `load(&self, model: &str, deadline: Micros, cutoff: Micros)` and `store(&self, model: &str, deadline: Micros, table: &Table)` to `store(&self, model: &str, deadline: Micros, cutoff: Micros, table: &Table)`, each calling `self.path(model, deadline, cutoff)`. Update their doc comments to "(model, deadline, cutoff)".

In `simulator.rs`, `Caches`:
```rust
    xp: OnceMap<(String, Micros, Micros), Xp>,
    fits: OnceMap<(String, Micros, Micros), V1Fit>,
    pools: OnceMap<(Micros, Micros), Vec<PoolPlayer>>,
```
Update the doc comment above `Caches` to "xP frames per (model, deadline, cutoff), v1 fits per (deadline, cutoff) of the fit view, pools per (deadline, cutoff) …".

```rust
    pub fn pool(&self, view: &AsOfView) -> Result<Arc<Vec<PoolPlayer>>> {
        self.pools
            .get_or_compute((view.deadline(), view.cutoff()), || pool_players(&player_pool(view)?))
    }

    pub fn xp(&self, model: &str, view: &AsOfView) -> Result<Arc<Xp>> {
        let key = (model.to_owned(), view.deadline(), view.cutoff());
        self.xp.get_or_compute(key, || {
            let cached = match &self.disk {
                Some(disk) => disk.load(model, view.deadline(), view.cutoff())?,
                None => None,
            };
            let hit = cached.is_some();
            let table = if let Some(t) = cached {
                t
            } else if fplopt_models::is_v1_family(model) {
                let refit = refit_deadline(view, REFIT_EVERY)?;
                let fit_view = view.earlier(refit)?;
                let fit_key = (model.to_owned(), fit_view.deadline(), fit_view.cutoff());
                let fit = self.fits.get_or_compute(fit_key, || {
                    fit_v1(&fit_view, &V1Params::for_model(model)?)
                })?;
                predict_v1(view, &fit)?
            } else {
                fplopt_models::predict(model, view)?
            };
            if let (Some(disk), false) = (&self.disk, hit) {
                disk.store(model, view.deadline(), view.cutoff(), &table)?;
            }
            let frame = XpFrame::from_table(&table)?;
            Ok(Xp { table, frame })
        })
    }
```
Note: for a deadline view `fit_view` is `view.earlier(refit)` with cutoff = refit, so the fit key equals the old `(model, refit)` key's information; for an early view whose refit deadline is after the cutoff the fit sees only rows before the cutoff and gets its own key.

- [ ] **Step 4: Run the tests**

Run: `cargo test -p fplopt-backtest`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add crates/fplopt-backtest/src/xp_cache.rs crates/fplopt-backtest/src/simulator.rs
```
```bash
git commit -m "feat(backtest): caches keyed by (deadline, cutoff)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: The `early` spec key and the deadline lineup step

**Files:**
- Modify: `crates/fplopt-backtest/src/policies.rs` (`Policy::Optimizer` ~299; `validate` ~351; `name` ~425; `inner_name` ~660; new `deadline_lineup` after `optimizer_plans`)
- Modify: `crates/fplopt-backtest/src/specs.rs` (keys ~114; build ~321–381; `key_default` ~426; test)
- Modify: `crates/fplopt-backtest/src/probes.rs` (`optimizer()` ~124)

- [ ] **Step 1: Write the failing test** (append to `mod tests` in `specs.rs`)

```rust
    #[test]
    fn early_key_parses_labels_and_refuses_what_it_cannot_combine_with() {
        let base = "optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2";
        let policy = parse_policy_spec(&format!("{base},early=1")).unwrap().build().unwrap();
        assert_eq!(policy.name(), "optimizer(v1,mh=1,m=2.0,d=0.75,early)");
        let Policy::Optimizer { early, .. } = &policy else { panic!("not an optimizer") };
        assert!(*early);
        // the default (0) is dropped: the live planner's spec and name are unchanged
        let plain = parse_policy_spec(&format!("{base},early=0")).unwrap();
        assert_eq!(plain, parse_policy_spec(base).unwrap());
        assert_eq!(plain.build().unwrap().name(), "optimizer(v1,mh=1,m=2.0,d=0.75)");
        // only 0/1, and not with chips, moving prices or the rule Triple Captain
        assert!(parse_policy_spec(&format!("{base},early=2")).unwrap().build().is_err());
        for extra in ["chips=1", "prices=forecast", "prices=oracle", "tc_rule=8"] {
            let spec = parse_policy_spec(&format!("{base},early=1,{extra}"));
            assert!(spec.and_then(|s| s.build()).is_err(), "{extra}");
        }
        assert!(parse_policy_spec("chipgains:v1:early=1").is_err());
    }
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `cargo test -p fplopt-backtest early_key`
Expected: FAIL (unknown key `early`) or compile error on `early`.

- [ ] **Step 3: Implement**

`policies.rs`, in `Policy::Optimizer` after `jitter`:
```rust
        /// The early-move study's policy E (`docs/superpowers/specs/2026-11-07-early-moves-design.md`):
        /// transfers at the previous lockdown, lineup at the deadline (`simulator::decide_step`).
        early: bool,
```
In `name()`, destructure `early` too (`Policy::Optimizer { xp_model, params, chips, extra, prices, jitter, early }`) and after the `jit=` push:
```rust
                if *early {
                    parts.push("early".to_owned());
                }
```
In `validate()` for `Policy::Optimizer { xp_model, params, chips, extra, prices, early, .. }`, before `Ok(())`:
```rust
                if *early && (*chips || *prices != PriceSource::Fixed || extra.tc_rule.is_some()) {
                    bail!("early=1 runs without chips, with fixed prices and without tc_rule");
                }
```
In `inner_name`, add `early: false,` after `jitter: None,`. In `probes.rs` `optimizer()`, add `early: false,` after `jitter: None,`.

Add to `impl Policy`, after `optimizer_plans`:
```rust
    /// The early policy's deadline step: the lineup, captain and vice for `ctx.state`, the
    /// squad after the GW's transfers (made at the lockdown), from the planner with this GW's
    /// transfers fixed to none and no chip. Returns the lineup and the solve's status and gap.
    pub fn deadline_lineup(&self, ctx: &DecisionContext) -> Result<(Lineup, String, f64)> {
        let Policy::Optimizer { params, jitter, .. } = self else {
            bail!("deadline_lineup: {} is not an optimizer policy", self.name());
        };
        let jittered;
        let frame = match jitter {
            Some(j) => {
                jittered = j.apply(ctx.frame, ctx.state.season, ctx.state.gw_index);
                &jittered
            }
            None => ctx.frame,
        };
        let problem = PlanInput::from_context(ctx.state, ctx.pool, frame, ctx.rules, params)?;
        let none: fplopt_optimize::TransferSet = (BTreeSet::new(), BTreeSet::new());
        let plan = match solve_plan(
            &problem,
            params,
            Some(&none),
            &[],
            &fplopt_optimize::ChipScenario::none(),
        ) {
            Ok(plan) => plan,
            Err(SolveError::LimitReached { status, .. }) => {
                bail!("deadline lineup: no plan before the solver stopped ({status})")
            }
            Err(SolveError::Infeasible(m)) => bail!("deadline lineup: {m}"),
            Err(SolveError::Other(e)) => return Err(e),
        };
        let first = plan.first();
        if !first.transfers.is_empty() {
            bail!("deadline lineup: the planner changed the squad");
        }
        let lineup = Lineup {
            starters: first.starters.clone(),
            bench: first.bench.clone(),
            captain: first.captain,
            vice: first.vice,
        };
        Ok((lineup, plan.status.clone(), plan.mip_gap))
    }
```
(Check that `SolveError::LimitReached` has the `status` field as used in `plans.rs`; it does, `Err(SolveError::LimitReached { status, .. })`.)

`specs.rs`: add `("early", Kind::Int),` after `("jitter_seed", Kind::Int),`; add `("optimizer", "early") => Some(Value::Int(0)),` after the `jitter_seed` default; in the optimizer build, after the jitter checks:
```rust
                let early = match params.remove("early").map_or(0, Value::i) {
                    0 => false,
                    1 => true,
                    v => bail!("early must be 0 or 1, got {v}"),
                };
```
and construct `Policy::Optimizer { xp_model, params: opt, chips, extra, prices, jitter, early }`. `opt.validate()?` stays; `build` ends with `policy.validate()?` (specs.rs ~414), which refuses the combinations.

- [ ] **Step 4: Run the tests**

Run: `cargo test -p fplopt-backtest`
Expected: all pass, including `jitter_keys_parse_and_label` (names unchanged).

- [ ] **Step 5: Commit**

```bash
git add crates/fplopt-backtest/src/policies.rs crates/fplopt-backtest/src/specs.rs crates/fplopt-backtest/src/probes.rs
```
```bash
git commit -m "feat(backtest): early=1 spec key and the deadline lineup step" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: The early decision in the simulator

**Files:**
- Modify: `crates/fplopt-backtest/src/state.rs` (`Decision` ~21)
- Modify: `crates/fplopt-backtest/src/simulator.rs` (`GwRow` ~252, `Step` ~288, `decide_step` ~303, `play_step` ~362, new helpers, tests)
- Modify: `crates/fplopt-backtest/src/evaluate.rs` (~476)
- Modify: `crates/fplopt-cli/src/backtest.rs` (~330)
- Modify: `dev/backtest_parity.py` (line 187)

- [ ] **Step 1: Write the failing tests** (append to `mod tests` in `simulator.rs`)

```rust
    use crate::state::{Holding, PoolPlayer, Transfer};
    use fplopt_build::table::Column;

    fn pool(prices: &[(i64, i64)]) -> Vec<PoolPlayer> {
        prices
            .iter()
            .map(|&(k, p)| PoolPlayer { player_key: k, element_type: 3, team_key: 1, price: p })
            .collect()
    }

    #[test]
    fn previous_lockdown_is_the_lockdown_of_the_seasons_last_earlier_gameweek() {
        let gameweeks = Table::new()
            .with("season", Column::I64(vec![2021, 2022, 2022, 2022]))
            .with("deadline_time", Column::OptI64(vec![Some(5), Some(10), Some(20), Some(30)]))
            .with("lockdown_time", Column::OptI64(vec![Some(8), Some(15), Some(25), Some(35)]));
        assert_eq!(previous_lockdown(&gameweeks, 2022, 30).unwrap(), Some(25));
        assert_eq!(previous_lockdown(&gameweeks, 2022, 20).unwrap(), Some(15));
        // the season's first GW has none (the previous season's doesn't count)
        assert_eq!(previous_lockdown(&gameweeks, 2022, 10).unwrap(), None);
    }

    #[test]
    fn early_kept_counts_the_buy_and_sell_prices_saved_by_moving_at_the_lockdown() {
        let state = SquadState {
            season: 2022,
            gw_index: 5,
            holdings: vec![Holding {
                player_key: 1,
                element_type: 3,
                team_key: 1,
                purchase_price: 50,
                price: 50,
            }],
            bank: 0,
            free_transfers: 1,
            chips_used: vec![],
            freehit_backup: None,
            freehit_bank: None,
        };
        let decision = Decision {
            transfers: vec![Transfer { out_key: 1, in_key: 2 }],
            ..Decision::default()
        };
        // out 1: 52 at A (sells for 51), 49 at the deadline; in 2: 60 at A, 61 at the deadline
        let at_a = pool(&[(1, 52), (2, 60)]);
        let at_d = pool(&[(1, 49), (2, 61)]);
        let kept = early_kept(&state, &decision, &at_a, &at_d, 0.5);
        assert_eq!(kept, (61 - 60) + (51 - 49));
        // nothing moved: nothing kept
        assert_eq!(early_kept(&state, &Decision::default(), &at_a, &at_d, 0.5), 0);
    }
```
(`Transfer` is `{ out_key, in_key }` and `SquadState`'s fields are as in the literal, `fplopt_optimize::state`.)

- [ ] **Step 2: Run them to make sure they fail**

Run: `cargo test -p fplopt-backtest previous_lockdown early_kept`
Expected: compile errors (`previous_lockdown`, `early_kept` not found).

- [ ] **Step 3: Implement**

`state.rs`, `Decision` (after `diagnostics`), and extend the doc comment ("`priced_at` … takes no part in equality"):
```rust
    /// The moment the transfers were priced when not at the deadline (the early policy's
    /// lockdown + 1 µs): the decision is applied, and replayed, at that moment's pool.
    pub priced_at: Option<fplopt_store::Micros>,
```
(`PartialEq` stays as it is: it ignores `priced_at`.)

`simulator.rs`:

`GwRow`, after `market_value`:
```rust
    /// When the transfers were decided and priced: the deadline, or the previous lockdown +
    /// 1 µs for the early policy (Rust only; dev/backtest_parity.py ignores it).
    pub decided_at: Micros,
    /// Early policy only: tenths of £m its moves kept against making them at the deadline
    /// (`early_kept`; Rust only).
    pub early_kept: Option<i64>,
```
`Step`, after `decision`:
```rust
    /// `early_kept` of an early decision.
    pub early_kept: Option<i64>,
```
Set `early_kept: None` in the deadline paths of `decide_step`.

New helpers (above `decide_step`):
```rust
/// The lockdown of `season`'s last GW whose deadline is before `deadline`, from a gameweek
/// table (season, deadline_time, lockdown_time); None for the season's first GW.
pub fn previous_lockdown(gameweeks: &Table, season: i64, deadline: Micros) -> Result<Option<Micros>> {
    let seasons = gameweeks.i64("season")?;
    let deadlines = gameweeks.opt_i64("deadline_time")?;
    let lockdowns = gameweeks.opt_i64("lockdown_time")?;
    Ok((0..gameweeks.height())
        .filter(|&i| seasons[i] == season && deadlines[i].is_some_and(|d| d < deadline))
        .max_by_key(|&i| deadlines[i])
        .and_then(|i| lockdowns[i]))
}

/// Tenths of £m kept by making `decision`'s transfers at the `early` pool's prices rather than
/// the `deadline` pool's: per in-player the deadline's price minus the early one, per out-player
/// the early selling price minus the deadline's (FPL's selling rule on `state`'s purchase price).
pub fn early_kept(
    state: &SquadState,
    decision: &Decision,
    early: &[PoolPlayer],
    deadline: &[PoolPlayer],
    sell_on_fee: f64,
) -> i64 {
    let price = |pool: &[PoolPlayer], key: i64| pool.iter().find(|p| p.player_key == key).map(|p| p.price);
    let (at_a, at_d) = (refresh(state, early), refresh(state, deadline));
    let selling = |s: &SquadState, key: i64| {
        s.holdings.iter().find(|h| h.player_key == key).map(|h| selling_price(h, sell_on_fee))
    };
    let mut kept = 0;
    for t in &decision.transfers {
        if let (Some(a), Some(d)) = (price(early, t.in_key), price(deadline, t.in_key)) {
            kept += d - a;
        }
        if let (Some(a), Some(d)) = (selling(&at_a, t.out_key), selling(&at_d, t.out_key)) {
            kept += a - d;
        }
    }
    kept
}

/// First season the early policy runs in: before it there are no player snapshots, so the
/// pool at the lockdown has no prices.
pub const EARLY_FIRST_SEASON: i64 = 2021;
```
Add `selling_price` to the `crate::state::{…}` import list.

Replace `decide_step` with:
```rust
/// Decide one GW. A deadline decision reads only the deadline view. The early policy, at a GW
/// after the season's first, decides its transfers on the view for this deadline cut at the
/// previous lockdown + 1 µs (A: the previous GW's outcomes are visible, as `read_outcomes`
/// sees them), at A's pool, and its lineup at the deadline with those transfers fixed; the
/// decision carries `priced_at = A`. A decision made earlier (`Decider::Decided`) is replayed
/// at its own pool.
pub fn decide_step(
    store: &DataStore,
    rules: &Rules,
    decider: Decider,
    state: &SquadState,
    deadline: Micros,
    caches: &Caches,
) -> Result<Step> {
    let view = store.as_of(deadline);
    let deadline_pool = caches.pool(&view)?;
    let policy = match decider {
        Decider::Decided(decision) => {
            let Some(at) = decision.priced_at else {
                let state = refresh(state, &deadline_pool);
                return Ok(Step {
                    state,
                    pool: deadline_pool,
                    decision: decision.clone(),
                    early_kept: None,
                });
            };
            let pool = caches.pool(&store.as_of_early(deadline, at)?)?;
            let kept = early_kept(state, decision, &pool, &deadline_pool, rules.sell_on_fee);
            let state = refresh(state, &pool);
            return Ok(Step { state, pool, decision: decision.clone(), early_kept: Some(kept) });
        }
        Decider::Policy(policy) => policy,
    };
    let early_at = match policy {
        Policy::Optimizer { early: true, .. } => {
            if state.season < EARLY_FIRST_SEASON {
                bail!(
                    "early=1 needs player snapshots (from {}); {} has none",
                    season_label(EARLY_FIRST_SEASON as i32),
                    season_label(state.season as i32)
                );
            }
            let gameweeks =
                view.table("gameweek", Some(&["season", "deadline_time", "lockdown_time"]))?;
            previous_lockdown(&gameweeks, state.season, deadline)?.map(|l| l + 1)
        }
        _ => None,
    };
    let Some(at) = early_at else {
        let state = refresh(state, &deadline_pool);
        let xp = caches.xp(policy.xp_model(), &view)?;
        let prices = caches.prices_for(policy, store, state.season)?;
        let ctx = DecisionContext {
            view: &view,
            state: &state,
            rules,
            pool: &deadline_pool,
            xp: &xp.table,
            frame: &xp.frame,
            prices: prices.as_ref().map(PriceData::input),
        };
        let decision = policy.decide(&ctx)?;
        return Ok(Step { state, pool: deadline_pool, decision, early_kept: None });
    };
    // Transfers at A.
    let early_view = store.as_of_early(deadline, at)?;
    let pool = caches.pool(&early_view)?;
    let state_a = refresh(state, &pool);
    let xp_a = caches.xp(policy.xp_model(), &early_view)?;
    let ctx_a = DecisionContext {
        view: &early_view,
        state: &state_a,
        rules,
        pool: &pool,
        xp: &xp_a.table,
        frame: &xp_a.frame,
        prices: None,
    };
    let at_a = policy.decide(&ctx_a)?;
    // Lineup at the deadline, transfers fixed.
    let (squad, _) = apply_decision(&state_a, &at_a, &pool, rules)?;
    let squad = refresh(&squad, &deadline_pool);
    let xp_d = caches.xp(policy.xp_model(), &view)?;
    let ctx_d = DecisionContext {
        view: &view,
        state: &squad,
        rules,
        pool: &deadline_pool,
        xp: &xp_d.table,
        frame: &xp_d.frame,
        prices: None,
    };
    let (lineup, status, gap) = policy.deadline_lineup(&ctx_d)?;
    let decision = Decision {
        transfers: at_a.transfers,
        lineup,
        chip: None,
        solver_status: at_a.solver_status.or(Some(status)),
        mip_gap: at_a.mip_gap.map(|g| g.max(gap)).or(Some(gap)),
        diagnostics: None,
        priced_at: Some(at),
    };
    let kept = early_kept(state, &decision, &pool, &deadline_pool, rules.sell_on_fee);
    Ok(Step { state: state_a, pool, decision, early_kept: Some(kept) })
}
```
Notes for the implementer:
- `apply_decision(&state_a, &at_a, …)` validates A's own lineup for the post-transfer squad; we only use the squad it returns.
- `refresh(&squad, &deadline_pool)` on the GW state: `squad.free_transfers` is the pre-transfer count, which doesn't matter: this GW's transfers are fixed to none, and only the GW's lineup is used.
- The decision is played with `Step.pool` = A's pool, so `play_step`'s `apply_decision` buys and sells at A's prices.

`play_step`: fill the two new `GwRow` fields:
```rust
        decided_at: decision.priced_at.unwrap_or(deadline),
        early_kept: step.early_kept,
```

`evaluate.rs` (~476), keep `same` (the `same_decision` column) and guard the shortcut:
```rust
            let same = decisions["a"] == decisions["b"];
            // equal decisions priced at different moments leave different banks
            let same_priced = same && decisions["a"].priced_at == decisions["b"].priced_at;
```
and use `same_priced` in `if side == "b" && same && continuation.is_some()` (→ `same_priced`).

`crates/fplopt-cli/src/backtest.rs`, after the `market_value` line:
```rust
        .with("decided_at", Column::Ts(results.iter().map(|g| g.row.decided_at).collect()))
        .with("early_kept", oi(&|g| g.row.early_kept))
```

`dev/backtest_parity.py` line 187:
```python
# gws.parquet columns Python never wrote (price sizing and early moves, 2026-11)
RUST_ONLY = ("market_value", "decided_at", "early_kept")
```

- [ ] **Step 4: Run the tests and lints**

Run: `cargo test -p fplopt-backtest`
Expected: all pass.

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: no warnings. Then `cargo fmt --all`.

- [ ] **Step 5: Commit**

```bash
git add crates/fplopt-backtest/src/state.rs crates/fplopt-backtest/src/simulator.rs crates/fplopt-backtest/src/evaluate.rs crates/fplopt-cli/src/backtest.rs dev/backtest_parity.py
```
```bash
git commit -m "feat(backtest): early=1 decides transfers at the previous lockdown" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Leakage probe for the early view

**Files:**
- Modify: `crates/fplopt-backtest/src/probes.rs`
- Modify: `crates/fplopt-cli/src/main.rs` (`Check::Leakage` ~192, dispatch ~289, `check_leakage` ~385)

- [ ] **Step 1: Add the probe**

In `probes.rs`, below `optimizer_rolling_random0`:
```rust
/// The early policy's step with the check's deadline as its cutoff: E's decision for the next
/// deadline of the season, made at the view's deadline (as at a lockdown), so the check
/// corrupts everything at or after A. A view without a later deadline in its season is one
/// `refused` row.
fn early_optimizer_ep_next_template(view: &AsOfView) -> Result<Table> {
    let (season, _) = view.gameweek_for_deadline()?;
    let gameweeks = view.table("gameweek", Some(&["season", "deadline_time"]))?;
    let (seasons, deadlines) = (gameweeks.i64("season")?, gameweeks.opt_i64("deadline_time")?);
    let next = (0..gameweeks.height())
        .filter(|&i| seasons[i] == season)
        .filter_map(|i| deadlines[i])
        .filter(|&d| d > view.deadline())
        .min();
    let Some(next) = next else {
        return Ok(frame(vec![("refused".to_owned(), 0, -1, 0, "no later deadline".to_owned())]));
    };
    let early = view.for_deadline(next)?;
    let mut policy = optimizer("ep_next");
    if let Policy::Optimizer { early: e, .. } = &mut policy {
        *e = true;
    }
    run_probe(&early, &policy, template_state)
}
```
Add `("probe:early_optimizer_ep_next_template", early_optimizer_ep_next_template),` to `PROBE_BUILDERS`. Extend the module doc: "`probe:early_*` runs a policy on a view for a later deadline cut at the check's deadline (`AsOfView::for_deadline`)".

(`run_probe` calls `policy.decide`, the transfer step at A. The deadline lineup step reads the deadline view, which is meant to see more, so it is not in the probe.)

- [ ] **Step 2: Add `--only` to `check leakage`**

In `main.rs`, `Check::Leakage`:
```rust
        /// Check only the frames whose name starts with this (e.g. `probe:early`).
        #[arg(long)]
        only: Option<String>,
```
Dispatch: `Check::Leakage { deadlines, seed, only }` → `check_leakage(&ctx.settings.data_dir, deadlines, seed, only.as_deref())`. In `check_leakage(data_dir, n_deadlines, seed, only: Option<&str>)`, after collecting `builders`:
```rust
    let builders: Vec<_> = builders
        .into_iter()
        .filter(|(name, _)| only.is_none_or(|p| name.starts_with(p)))
        .collect();
    if builders.is_empty() {
        anyhow::bail!("no frame matches --only: the leakage check was not performed");
    }
```

- [ ] **Step 3: Run the probe's leakage check on real data**

Run: `cargo build --release -p fplopt-cli`
Run:
```bash
FPLOPT_DATA_DIR=C:/Users/samin/Documents/GitHub/fpl-optimizer/data target/release/fplopt-rs.exe check leakage --deadlines 2 --only probe:early
```
Expected: exits 0 with no `leak:` lines. ep_next before 2021/22 may be refused or empty at some edge deadlines; that is the same as the existing `probe:optimizer_ep_next_template`. If the clean run fails at an edge deadline where `probe:optimizer_ep_next_template` passes, fix the probe; don't skip the deadline.

- [ ] **Step 4: Commit**

```bash
git add crates/fplopt-backtest/src/probes.rs crates/fplopt-cli/src/main.rs
```
```bash
git commit -m "feat(check): early-view leakage probe; check leakage --only" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Local smoke runs

**Files:** none (scratch).

- [ ] **Step 1: `early=0` changes nothing**

Run the Task 0 command with `target/release/fplopt-rs.exe` and `--out …/early-smoke/after`. Then:
```bash
uv run python -c "import pandas as pd; s='C:/Users/samin/AppData/Local/Temp/claude/6ff2f6c53ca963aa/50295afa-4ab0-4ac1-a579-1e2ddcc3dc5e/scratchpad/early-smoke/'; a=pd.read_parquet(s+'base/gws.parquet'); b=pd.read_parquet(s+'after/gws.parquet').drop(columns=['decided_at','early_kept']); pd.testing.assert_frame_equal(a, b); print('identical', len(a))"
```
Expected: `identical <n>`.

- [ ] **Step 2: E runs and moves at A**

```bash
FPLOPT_DATA_DIR=C:/Users/samin/Documents/GitHub/fpl-optimizer/data target/release/fplopt-rs.exe backtest run --spec optimizer:ep_next:horizon=6,decay=0.75,max_hits=1,hit_margin=2,early=1 --seasons 2022 --starts template@1,random:1@20 --jobs 4 --no-xp-cache --family "early smoke" --experiments "C:/Users/samin/AppData/Local/Temp/claude/6ff2f6c53ca963aa/50295afa-4ab0-4ac1-a579-1e2ddcc3dc5e/scratchpad/early-smoke/exp.csv" --out "C:/Users/samin/AppData/Local/Temp/claude/6ff2f6c53ca963aa/50295afa-4ab0-4ac1-a579-1e2ddcc3dc5e/scratchpad/early-smoke/early"
```
Then check:
```bash
uv run python -c "import pandas as pd; g=pd.read_parquet('C:/Users/samin/AppData/Local/Temp/claude/6ff2f6c53ca963aa/50295afa-4ab0-4ac1-a579-1e2ddcc3dc5e/scratchpad/early-smoke/early/gws.parquet'); e=g[g.decided_at<g.deadline]; print('policy', g.policy.iloc[0]); print('early GWs', len(e), 'of', len(g)); print('GW1 at deadline', bool((g[g.gw_index==1].decided_at==g[g.gw_index==1].deadline).all())); print('kept tenths', int(e.early_kept.sum()), 'nulls on early GWs', int(e.early_kept.isna().sum()))"
```
Expected: policy name ends `,early)`; every GW but the season's first has `decided_at < deadline`; GW1 at the deadline; `early_kept` set on every early GW and positive in total (the sizing found gains in half the moves).

- [ ] **Step 3: E is refused before 2021/22**

```bash
FPLOPT_DATA_DIR=C:/Users/samin/Documents/GitHub/fpl-optimizer/data target/release/fplopt-rs.exe backtest run --spec optimizer:rolling:early=1 --seasons 2020 --starts template@1 --jobs 1 --no-xp-cache --family "early smoke" --experiments "C:/Users/samin/AppData/Local/Temp/claude/6ff2f6c53ca963aa/50295afa-4ab0-4ac1-a579-1e2ddcc3dc5e/scratchpad/early-smoke/exp.csv" --out "C:/Users/samin/AppData/Local/Temp/claude/6ff2f6c53ca963aa/50295afa-4ab0-4ac1-a579-1e2ddcc3dc5e/scratchpad/early-smoke/refused"
```
Expected: fails with "early=1 needs player snapshots (from 2021-22); 2020-21 has none".

- [ ] **Step 4: Per-decision compare runs** (2022, one start, ep_next)

```bash
FPLOPT_DATA_DIR=C:/Users/samin/Documents/GitHub/fpl-optimizer/data target/release/fplopt-rs.exe backtest compare --a optimizer:ep_next:horizon=6,decay=0.75,max_hits=1,hit_margin=2,early=1 --b optimizer:ep_next:horizon=6,decay=0.75,max_hits=1,hit_margin=2 --seasons 2022 --starts template@1 --jobs 4 --no-xp-cache --continuation own --reference b --k 1 --stride 1 --family "early smoke" --experiments "C:/Users/samin/AppData/Local/Temp/claude/6ff2f6c53ca963aa/50295afa-4ab0-4ac1-a579-1e2ddcc3dc5e/scratchpad/early-smoke/exp.csv" --out "C:/Users/samin/AppData/Local/Temp/claude/6ff2f6c53ca963aa/50295afa-4ab0-4ac1-a579-1e2ddcc3dc5e/scratchpad/early-smoke/compare"
```
Expected: `Written to …`; `per_decision.parquet` has 38 rows; `same_decision` is true at gw_index 1 and on some later GWs. Check that the per-decision `diff` is 0 on every `same_decision` row: `uv run python -c "import pandas as pd; d=pd.read_parquet('…/early-smoke/compare/per_decision.parquet'); print(d.same_decision.mean(), d[d.same_decision]['diff'].abs().max())"` → second number 0.

No commit (scratch only). If any expectation fails, fix the code in the task that owns it and rerun.

---

### Task 7: Driver `dev/early_moves.py`

**Files:**
- Create: `dev/early_moves.py`
- Test: `tests/test_early_moves.py`

- [ ] **Step 1: Write the failing tests** (`tests/test_early_moves.py`)

```python
import importlib.util
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("early_moves", REPO / "dev" / "early_moves.py")
em = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = em
_spec.loader.exec_module(em)

H = pd.Timedelta(hours=1)


def _gws(kept_by_unit):
    rows = []
    for start_id, kept in kept_by_unit.items():
        for i, k in enumerate(kept, start=1):
            rows.append({"season": 2022, "start_id": start_id, "gw_index": i, "early_kept": k})
    return pd.DataFrame(rows)


def test_value_points_count_money_from_the_gw_after_it_was_kept():
    # 10 tenths kept at GW1 is worth 0.395 points in each later GW (2 here)
    g = _gws({"template@1": [10, 0, 0], "random0@1": [0, 0, 0], "template@10": [99, 99, 99]})
    v = em.value_points(g)
    assert list(v.index) == ["random0@1", "template@1"]  # full-season units only
    assert v["template@1"] == pytest.approx(2 * 0.395)
    assert v["random0@1"] == 0


def test_value_points_treat_a_missing_kept_as_zero():
    g = _gws({"template@1": [None, 10, 0]})
    assert em.value_points(g)["template@1"] == pytest.approx(0.395)


def test_gate_combines_the_two_errors():
    out = em.gate(v_mean=6.0, v_se=1.0, c_mean=-2.0, c_se=2.0)
    assert out["net"] == pytest.approx(4.0)
    assert out["se"] == pytest.approx(math.sqrt(5))
    assert out["p"] == pytest.approx(0.0368, abs=1e-3)
    assert out["pass"] is True
    assert em.gate(v_mean=1.0, v_se=1.0, c_mean=-1.0, c_se=1.0)["pass"] is False


def test_news_risk_flags_buys_that_were_available_at_a_and_flagged_before_the_deadline():
    t0 = pd.Timestamp("2022-09-01T09:00Z")
    snaps = pd.DataFrame(
        {
            "player_key": [7, 7, 8, 8, 9],
            "snapshot_at": [t0 - H, t0 + 5 * H, t0 - H, t0 + 5 * H, t0 + 5 * H],
            "status": ["a", "d", "a", "a", "i"],
        }
    )
    buys = pd.DataFrame(
        {
            "player_key": [7, 8, 9],
            "decided_at": [t0, t0, t0],
            "deadline": [t0 + 48 * H] * 3,
        }
    )
    out = em.news_risk(buys, snaps)
    # 7 flagged after A, 8 never, 9 has no snapshot before A (not counted)
    assert out == {"n": 2, "flagged": 1, "share": 0.5}


def test_buys_reads_the_ins_of_early_decisions_only():
    t0 = pd.Timestamp("2022-09-01T09:00Z")
    g = pd.DataFrame(
        {
            "transfers": ["[[1, 2], [3, 4]]", "[[5, 6]]", "[]"],
            "decided_at": [t0, t0 + 48 * H, t0],
            "deadline": [t0 + 48 * H, t0 + 48 * H, t0 + 48 * H],
        }
    )
    b = em.buys(g)
    assert list(b.player_key) == [2, 4]
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `uv run pytest tests/test_early_moves.py -q`
Expected: FAIL (`dev/early_moves.py` does not exist). The import style is `tests/test_noise_study.py`'s.

- [ ] **Step 3: Implement `dev/early_moves.py`**

```python
"""Early moves (docs/superpowers/specs/2026-11-07-early-moves-design.md): policy E (the live
planner with early=1: transfers at the previous lockdown, lineup at the deadline) vs the live
planner L on the develop seasons with player snapshots.

    run    --out DIR [--jobs 14] [--binary target/release/fplopt-rs] [--experiments PATH]
    report --out DIR [--data-dir data]

`run` writes E's backtest (DIR/E) and the standard comparison E vs L (DIR/compare); `report`
computes the value gain V (money kept, full-season units, at 0.395 points per £1m per GW),
the information cost C (per decision, k = 1), the disagreement, the news risk and the gate,
writes DIR/early.json and prints it.
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
import noise_study as ns  # noqa: E402

LIVE = "optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2"
EARLY = LIVE + ",early=1"
SEASONS = "2021,2022"
STARTS = ns.STARTS
FAMILY = "early moves develop"
POINTS_PER_M = 0.395  # money value, docs/research/2026-11-price-sizing.md
ALPHA = 0.10


def run_cmd(binary, out: Path, jobs: int, experiments: Path) -> list[str]:
    cmd = [str(binary), "backtest", "run", "--spec", EARLY, "--seasons", SEASONS]
    cmd += ["--starts", STARTS, "--jobs", str(jobs), "--family", FAMILY + " E run"]
    cmd += ["--experiments", str(experiments), "--out", str(out / "E")]
    return cmd


def compare_cmd(binary, out: Path, jobs: int, experiments: Path) -> list[str]:
    cmd = [str(binary), "backtest", "compare", "--a", EARLY, "--b", LIVE]
    cmd += ["--seasons", SEASONS, "--starts", STARTS, "--jobs", str(jobs)]
    cmd += ["--family", FAMILY, "--experiments", str(experiments), "--out", str(out / "compare")]
    cmd += ["--continuation", "own", "--reference", "b", "--k", "1", "--stride", "1"]
    return cmd


def value_points(gws: pd.DataFrame) -> pd.Series:
    """Per full-season unit (start_id ending '@1'): points from the money kept, Σ over GWs of
    the money kept before that GW (£m) × POINTS_PER_M. Money kept at GW t helps from t + 1."""
    g = gws[gws["start_id"].str.endswith("@1")].sort_values(["season", "start_id", "gw_index"])
    out = {}
    for (_, sid), unit in g.groupby(["season", "start_id"]):
        kept = unit["early_kept"].fillna(0).astype(float)
        before = kept.cumsum().shift(1, fill_value=0.0)
        out[sid if g["season"].nunique() == 1 else f"{unit['season'].iloc[0]}:{sid}"] = (
            float(before.sum()) / 10 * POINTS_PER_M
        )
    return pd.Series(out).sort_index()


def gate(v_mean: float, v_se: float, c_mean: float, c_se: float) -> dict:
    net = v_mean + c_mean
    se = math.sqrt(v_se**2 + c_se**2)
    p = float(stats.norm.sf(net / se)) if se > 0 else float("nan")
    return {"net": net, "se": se, "p": p, "pass": bool(se > 0 and p < ALPHA)}


def buys(gws: pd.DataFrame) -> pd.DataFrame:
    """The in-players of early decisions (decided_at < deadline), with A and the deadline."""
    rows = []
    for r in gws[gws["decided_at"] < gws["deadline"]].itertuples():
        for _, inp in json.loads(r.transfers):
            rows.append({"player_key": inp, "decided_at": r.decided_at, "deadline": r.deadline})
    return pd.DataFrame(rows, columns=["player_key", "decided_at", "deadline"])


def news_risk(buys: pd.DataFrame, snaps: pd.DataFrame) -> dict:
    """Among buys whose last snapshot before A says available ('a'): how many have a snapshot
    in [A, deadline) with another status."""
    by_key = {k: v.sort_values("snapshot_at") for k, v in snaps.groupby("player_key")}
    n = flagged = 0
    for b in buys.itertuples():
        s = by_key.get(b.player_key)
        if s is None:
            continue
        before = s[s["snapshot_at"] < b.decided_at]
        if before.empty or before["status"].iloc[-1] != "a":
            continue
        n += 1
        window = s[(s["snapshot_at"] >= b.decided_at) & (s["snapshot_at"] < b.deadline)]
        flagged += bool((window["status"] != "a").any())
    return {"n": n, "flagged": flagged, "share": flagged / n if n else float("nan")}


def cmd_run(args) -> int:
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    times = {}
    for name, cmd in (
        ("E", run_cmd(args.binary, out, args.jobs, args.experiments)),
        ("compare", compare_cmd(args.binary, out, args.jobs, args.experiments)),
    ):
        if (out / name / "summary.json").exists():
            print(f"{name}: done, skipped")
            continue
        (out / name).mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()
        with (out / name / "run.log").open("w", encoding="utf-8") as log:
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        times[name] = time.monotonic() - t0
        print(f"{name}: exit {r.returncode}, {times[name] / 60:.1f} min", flush=True)
        if r.returncode != 0:
            return r.returncode
    (out / "times.json").write_text(json.dumps(times, indent=2))
    return 0


def cmd_report(args) -> int:
    out: Path = args.out
    gws = pd.read_parquet(out / "E" / "gws.parquet")
    dec = pd.read_parquet(out / "compare" / "per_decision.parquet")
    v = value_points(gws)
    v_mean = float(v.mean())
    v_se = float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan")
    c_mean, c_se_pooled, n = ns.per_decision(dec, 1, "diff")
    c_se_season = ns.season_clustered_se(dec, 1, "diff")
    c_se = max(c_se_pooled, c_se_season)
    early = gws[gws["decided_at"] < gws["deadline"]]
    full = gws[gws["start_id"].str.endswith("@1")]
    kept_season = full.groupby(["season", "start_id"])["early_kept"].sum().mean() / 10
    snaps = pd.read_parquet(
        Path(args.data_dir) / "player_snapshot.parquet",
        columns=["player_key", "season", "snapshot_at", "status"],
    )
    snaps = snaps[snaps["season"].isin([2021, 2022])]
    report = {
        "value": {
            "points_mean": v_mean,
            "points_se": v_se,
            "units": len(v),
            "kept_m_per_season": float(kept_season),
            "kept_per_unit_points": v.to_dict(),
        },
        "information_cost": {
            "mean": c_mean,
            "se_pooled": c_se_pooled,
            "se_season": c_se_season,
            "se_used": c_se,
            "windows": n,
        },
        "disagreement": float(1 - dec["same_decision"].mean()),
        "early_share_of_gws": float(len(early) / len(gws)),
        "news_risk": news_risk(buys(gws), snaps),
        "gate": gate(v_mean, v_se, c_mean, c_se),
    }
    (out / "early.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--jobs", type=int, default=14)
    r.add_argument("--binary", type=Path, default=Path("target/release/fplopt-rs"))
    r.add_argument("--experiments", type=Path, default=Path("results/experiments.csv"))
    rep = sub.add_parser("report")
    rep.add_argument("--out", type=Path, required=True)
    rep.add_argument("--data-dir", default="data")
    args = p.parse_args()
    raise SystemExit(cmd_run(args) if args.cmd == "run" else cmd_report(args))


if __name__ == "__main__":
    main()
```

The test for `value_points` uses one season, so its keys are bare start ids; with two seasons they become `season:start_id`.

- [ ] **Step 4: Run the tests and lint**

Run: `uv run pytest tests/test_early_moves.py -q`
Expected: 6 passed.
Run: `uv run ruff check dev/early_moves.py tests/test_early_moves.py && uv run ruff format dev/early_moves.py tests/test_early_moves.py`
Expected: clean.

- [ ] **Step 5: Report on the smoke outputs**

Copy the smoke outputs into the driver's layout and report on them (ep_next, so the numbers mean nothing; this checks the plumbing):
```bash
uv run python -c "import shutil; s='C:/Users/samin/AppData/Local/Temp/claude/6ff2f6c53ca963aa/50295afa-4ab0-4ac1-a579-1e2ddcc3dc5e/scratchpad/early-smoke/'; shutil.copytree(s+'early', s+'drv/E', dirs_exist_ok=True); shutil.copytree(s+'compare', s+'drv/compare', dirs_exist_ok=True)"
```
```bash
uv run python dev/early_moves.py report --out "C:/Users/samin/AppData/Local/Temp/claude/6ff2f6c53ca963aa/50295afa-4ab0-4ac1-a579-1e2ddcc3dc5e/scratchpad/early-smoke/drv" --data-dir C:/Users/samin/Documents/GitHub/fpl-optimizer/data
```
Expected: a JSON report with finite `value.points_mean` and `information_cost.mean`, `news_risk.n` > 0 and a `gate` block. On this one-season smoke `se_season` is NaN and `se_used` is the pooled SE (`max(pooled, nan)` returns `pooled`); `value.points_se` is NaN with one full-season unit, so the gate's `p` is NaN. Both are finite on the real run (2 seasons, 8 full-season units).

- [ ] **Step 6: Commit**

```bash
git add dev/early_moves.py tests/test_early_moves.py
```
```bash
git commit -m "feat(dev): early moves driver (run, report)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Docs and push

**Files:**
- Modify: `CLAUDE.md` (Commands, after the `research price-eval` line)

- [ ] **Step 1: Add the CLAUDE.md line**

```markdown
- `optimizer:v1:...,early=1` and `uv run python dev/early_moves.py run|report --out DIR` — the early-move study (`docs/superpowers/specs/2026-11-07-early-moves-design.md`): `early=1` makes the GW's transfers at the previous lockdown + 1 µs (the view for the deadline cut there, `DataStore::as_of_early`; A's snapshot prices) and re-solves the lineup at the deadline with them fixed; the season's first GW stays at the deadline; seasons before 2021/22 are refused (no snapshots); not with chips, moving prices or `tc_rule`. `gws.parquet` gains `decided_at` and `early_kept` (tenths kept vs the same moves at the deadline). The driver runs E's backtest and the standard comparison (k = 1) on 2021/22–2022/23 and reports the value gain (full-season units, 0.395 points per £1m per GW), the information cost, the disagreement, the news risk and the gate (family `early moves develop`). `check leakage --only PREFIX` checks a subset (e.g. `probe:early`)
```

- [ ] **Step 2: Full checks**

Run: `cargo test --workspace` → all pass.
Run: `cargo clippy --workspace --all-targets -- -D warnings` → clean.
Run: `cargo fmt --all -- --check` → clean.
Run: `uv run pytest -q` → all pass (realdata tests skipped).
Run: `uv run ruff check . && uv run ruff format --check .` → clean.

- [ ] **Step 3: Commit and push**

```bash
git add CLAUDE.md
```
```bash
git commit -m "docs: early moves commands" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
```bash
git push
```

---

### Task 9 (controller, after the user's go-ahead for server time): the develop run and the note

**Files:** `docs/research/2026-11-early-moves.md`, `docs/PLAN.md` §12, `results/experiments.csv`.

- [ ] **Step 1: Update the server worktree and build**

On the server (`ssh fplopt-server`), in `~/fpl-pricec` (never `~/fpl-optimizer`): `git fetch` then `git checkout --detach origin/feat/price-sizing`, then `PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH cargo build --release -p fplopt-cli` and `uv sync --all-extras`.

- [ ] **Step 2: Run detached** (about 20 min for E's backtest and 35–60 min for the comparison, plus v1 xP at the lockdowns: the first run computes ~70 new frames)

`setsid nohup .venv/bin/python dev/early_moves.py run --out results/early-moves --jobs 14 > results/early-moves.out 2>&1 < /dev/null &`

Wait with a watcher whose pattern can't match itself: `while pgrep -f "[e]arly_moves.py run" > /dev/null; do sleep 120; done`.

- [ ] **Step 3: Report**

`.venv/bin/python dev/early_moves.py report --out results/early-moves --data-dir data`. Copy `early.json` and the new `experiments.csv` rows (family `early moves develop` and `early moves develop E run`) into the repo.

- [ ] **Step 4: Write the note and the PLAN row**

`docs/research/2026-11-early-moves.md`:
- the question;
- the numbers: V (£m and points a season), C (pooled and season SE), the net and the gate, the disagreement, the news risk, the times;
- the verdict: alert next, or not.

Then add a PLAN §12 row, commit and push.
