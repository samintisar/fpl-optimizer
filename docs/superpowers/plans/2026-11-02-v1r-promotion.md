# v1r promotion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the research loop's nine unpromoted keeps into the Rust xP model as a new model `v1r`, prove it matches Python `research`, refit its calibration in Rust, run the pre-registered validate test, and adopt it live if it passes.

**Architecture:** Every keep becomes a parameter on the existing Rust structs (`MinutesParams`, `SharesParams`, `V1Params`) whose default keeps today's `v1` bit-identical; `V1Params::v1r()` turns all of them on. `v1r` is registered next to `v1` everywhere a model name is accepted, and has its own calibration table, fitted by a new Rust port of `dev/calibrate_v1.py`'s `walk` and `fit` (only the part v1 uses, `p_start:1+`). Parity against Python `research` at a pinned commit uses the batch 6 tooling (`fplopt-rs models dump`, `dev/models_parity.py`).

**Tech Stack:** Rust workspace (`crates/fplopt-features`, `crates/fplopt-models`, `crates/fplopt-cli`), LightGBM C API (`fplopt-lgbm`), Python `research` branch of `fpl-autoresearch` as the parity reference.

**Spec:** `docs/superpowers/specs/2026-11-02-v1r-promotion-design.md`. Read it first; its section 2 (the pre-registered test) is binding and must not be changed after Task 14.

---

## Ground rules

- **Branch.** Start only after the Rust switch's gates pass (migration plan `docs/superpowers/plans/2026-10-28-rust-migration.md`, batch 7 gate and the shadow week). Branch `feat/v1r` from `rust-migration` (or from `main` if the switch has merged it). Do not touch the server's live checkout until Task 16.
- **Python is the reference, not a target.** Nothing is added to `src/fplopt`. The Python code quoted below is the `research` branch of `C:/Users/samin/Documents/GitHub/fpl-autoresearch` (on the server `~/fpl-research`); read it there with `git show <commit>`; never commit to that repo (its guard hook blocks it, and the loop owns it).
- **`v1` must not move.** After every task, `cargo test --workspace` and the existing golden files (`tests/fixtures/rust_golden/`) must pass unchanged. A diff in a `v1` golden means a default changed: fix it, don't regenerate the golden.
- **Sums and order.** Follow the migration's rules (migration plan, batch 6 *As built*): numpy pairwise sums are `np_sum`, pandas group sums are `Kahan`, `bincount`/`cumsum` are sequential, `logit`/`expit` from `crate::util`. Each task below names which one a line uses.
- **Commands.** `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --all`. Release binary: `cargo build --release -p fplopt-cli` → `target/release/fplopt-rs`.

## File map

| File | Change |
|---|---|
| `crates/fplopt-features/src/history.rs` | `ELO_FEATURES`, `match_elo(view)`, `horizon_elo(view)` |
| `crates/fplopt-models/src/gbm.rs` | none (features are already per booster) |
| `crates/fplopt-models/src/minutes.rs` | `horizon_shrink`, `elo_features` params; `horizon_start`; `lag_features`; `with_match_elo` / `with_horizon_elo` |
| `crates/fplopt-models/src/availability.rs` | `horizon_start` in the fit rows; `category_slopes` |
| `crates/fplopt-models/src/shares.rs` | `position_par`, `price_par`, `price_par_quadratic`, `price_prior_quadratic`; quadratic `price_model`; `price_curve` |
| `crates/fplopt-models/src/assemble.rs` | `category_slopes` in `V1Params`; `V1Params::v1r()`; `xp_v1r`, `xp_v1r_with_fit`; calibration chosen by params |
| `crates/fplopt-models/src/calibration.rs` | tables keyed by model; `isotonic` (fit) |
| `crates/fplopt-models/calibration_table_v1r.json` | new, written by Task 12 |
| `crates/fplopt-models/src/lib.rs` | `v1r` in `MODELS`, `COMPONENTS`, `LEAKAGE_BUILDERS`, `predict`, `component` |
| `crates/fplopt-cli/src/models_eval.rs`, `bench.rs` | `v1r` in the model lists |
| `crates/fplopt-cli/src/calibrate.rs` | new: `fplopt-rs models calibrate walk|fit` |
| `crates/fplopt-cli/src/models.rs` | the `Calibrate` subcommand |
| `dev/models_parity.py` | `--python-src`, `--rust-model` |
| `crates/fplopt-bot/src/builder.rs` | `LIVE_XP` (Task 16 only) |

---

### Task 0: Pin the reference and branch

**Files:** none (records go in this plan's *Pins* section at the bottom).

- [ ] **Step 1: Check the switch gates passed.** Read the migration plan's batch 7 and shadow-week status on the branch. If either is open, stop: this plan doesn't start before them (spec, *Risks*).

- [ ] **Step 2: Branch.**

```bash
git fetch origin
git switch -c feat/v1r origin/rust-migration
```

- [ ] **Step 3: Pin the research commit.** In the research checkout:

```bash
cd C:/Users/samin/Documents/GitHub/fpl-autoresearch
git fetch
git rev-parse research
git log --oneline research -- src/fplopt/models src/fplopt/features | head -20
```

Record the SHA under *Pins* below. Check that every keep in the spec's table is an ancestor of it (`git merge-base --is-ancestor <keep> <pin>`), and that no later `try:` commit on `research` changed `src/fplopt/models` or `src/fplopt/features` and was kept (the ledger in `autoresearch/ideas.md` says *kept*). A keep after 2026-11-02 is **out of scope**: it waits for a second promotion. If one landed, pin the last commit before it and note that.

- [ ] **Step 4: Find the two breaking synthetic tests.** In a scratch worktree of the pin (`git worktree add ../research-pin <pin>`), run `uv run pytest -x -q` and record the two failing tests and their messages under *Pins*. Expected: one in `tests/test_models_minutes.py` (horizon_shrink) and one in `tests/test_models_shares.py` or a model-level test (position_par). Each needs a written verdict in Task 4 / Task 7: "the test encodes the old behaviour" (keep stays, the Rust test asserts the new one), or "the keep is wrong" (drop it from `v1r`, note it in the spec).

- [ ] **Step 5: Commit the pins.**

```bash
git add docs/superpowers/plans/2026-11-02-v1r-promotion.md
git commit -m "docs(v1r): pin the research commit and the breaking tests"
```

---

### Task 1: Register `v1r` as a model (still identical to `v1`)

**Files:**
- Modify: `crates/fplopt-models/src/assemble.rs` (V1Params, `xp_v1`, `xp_v1_with_fit`, `predict_fixtures`)
- Modify: `crates/fplopt-models/src/calibration.rs`
- Modify: `crates/fplopt-models/src/lib.rs`
- Modify: `crates/fplopt-cli/src/models_eval.rs:27`, `crates/fplopt-cli/src/bench.rs:26`
- Test: `crates/fplopt-models/src/assemble.rs` (tests module)

At the end of this task `v1r` exists, its params equal `v1`'s, and it refuses to calibrate (no table yet). Later tasks flip its switches one at a time.

- [ ] **Step 1: Write the failing test** (in `assemble.rs`'s `#[cfg(test)] mod tests`, create it if absent):

```rust
#[test]
fn v1r_is_registered_and_v1_is_unchanged() {
    assert!(crate::MODELS.contains(&"v1r"));
    assert!(crate::COMPONENTS.contains(&"v1r"));
    assert_eq!(V1Params::default().model, "v1");
    assert_eq!(V1Params::v1r().model, "v1r");
    // No v1r table yet: a calibrated v1r is refused, an uncalibrated one is fine.
    assert!(crate::calibration::calibration_for("v1r", 2023).unwrap().is_none());
    assert!(crate::calibration::calibration_for("v1", 2023).unwrap().is_some());
}
```

- [ ] **Step 2: Run it:** `cargo test -p fplopt-models v1r_is_registered` → FAIL (no `v1r`, no `model` field).

- [ ] **Step 3: Implement.**

`assemble.rs` — add a `model` field (the calibration table key) and the constructor:

```rust
#[derive(Clone, Debug, PartialEq)]
pub struct V1Params {
    /// The model name: picks the calibration table (`calibration::calibration_for`).
    pub model: &'static str,
    pub team: TeamParams,
    pub minutes: MinutesParams,
    pub shares: SharesParams,
    pub components: ComponentsParams,
    pub back_split: bool,
    pub calibrate: bool,
}

impl Default for V1Params {
    fn default() -> Self {
        V1Params {
            model: "v1",
            team: TeamParams::default(),
            minutes: MinutesParams { ban_residual: true, ..MinutesParams::default() },
            shares: SharesParams::default(),
            components: ComponentsParams::default(),
            back_split: true,
            calibrate: true,
        }
    }
}

impl V1Params {
    /// `v1` with the research loop's keeps on (spec 2026-11-02-v1r-promotion). Each task of
    /// the plan turns one on here.
    pub fn v1r() -> Self {
        V1Params { model: "v1r", ..V1Params::default() }
    }

    /// The params a model name stands for.
    pub fn for_model(name: &str) -> Result<Self> {
        match name {
            "v1" => Ok(V1Params::default()),
            "v1r" => Ok(V1Params::v1r()),
            other => bail!("not a v1-family model: {other:?}"),
        }
    }
}
```

Replace the calibration check in `predict_fixtures`:

```rust
pub fn predict_fixtures(view: &AsOfView, fit: &V1Fit) -> Result<Vec<Fixture>> {
    let (season, _) = view.gameweek_for_deadline()?;
    let params = &fit.params;
    let calibration =
        if params.calibrate { calibration_for(params.model, season)? } else { None };
    if calibration.is_some() && *params != V1Params::for_model(params.model)? {
        bail!(
            "the {} calibration table was fitted for that model's default params; use calibrate=false",
            params.model
        );
    }
    let mut frame = fixture_components(view, fit)?;
    calibrate_inputs(&mut frame, calibration.as_ref());
    fixture_points(&mut frame, &backtest_rules(season, &default_config_dir())?);
    calibrate_xp(&mut frame, calibration.as_ref());
    Ok(frame)
}
```

Generalize the two entry points (keep the old names as thin wrappers so callers don't change):

```rust
/// `MODELS[name](view)` for the v1 family: fit at the refit cutoff, predict at the view.
pub fn xp_v1_family(name: &str, view: &AsOfView) -> Result<Table> {
    let cutoff = refit_deadline(view, REFIT_EVERY)?;
    let fit = fit_v1(&view.earlier(cutoff)?, &V1Params::for_model(name)?)?;
    predict_v1(view, &fit)
}

pub fn xp_v1(view: &AsOfView) -> Result<Table> {
    xp_v1_family("v1", view)
}

pub fn xp_v1r(view: &AsOfView) -> Result<Table> {
    xp_v1_family("v1r", view)
}

/// `xp_v1_family` with the fit's values (the parity gate's gate 1).
pub fn xp_v1_family_with_fit(name: &str, view: &AsOfView) -> Result<(Table, serde_json::Value)> {
    let cutoff = refit_deadline(view, REFIT_EVERY)?;
    let fit = fit_v1(&view.earlier(cutoff)?, &V1Params::for_model(name)?)?;
    Ok((predict_v1(view, &fit)?, fit.to_json()?))
}

pub fn xp_v1_with_fit(view: &AsOfView) -> Result<(Table, serde_json::Value)> {
    xp_v1_family_with_fit("v1", view)
}
```

`calibration.rs` — key the tables by model. The v1r file doesn't exist yet, so it is optional at compile time through a `Option<&str>` constant that Task 12 fills:

```rust
pub static TABLE: LazyLock<Value> = LazyLock::new(|| {
    serde_json::from_str(include_str!("../calibration_table.json")).expect("calibration_table.json")
});

/// v1r's table (`fplopt-rs models calibrate fit --model v1r --write`); None until fitted.
const V1R_TABLE_TEXT: Option<&str> = None;

pub static V1R_TABLE: LazyLock<Option<Value>> = LazyLock::new(|| {
    V1R_TABLE_TEXT.map(|t| serde_json::from_str(t).expect("calibration_table_v1r.json"))
});

/// The table of `model`, None if it has none.
pub fn table_for(model: &str) -> Option<&'static Value> {
    match model {
        "v1" => Some(&TABLE),
        "v1r" => V1R_TABLE.as_ref(),
        _ => None,
    }
}
```

and change `calibration_for(season)` to `calibration_for(model: &str, season: i64)`, starting with:

```rust
pub fn calibration_for(model: &str, season: i64) -> Result<Option<Calibration>> {
    let Some(table) = table_for(model) else { return Ok(None) };
    let table = table["table"].as_array().context("calibration table: no 'table'")?;
    // ... unchanged from here
```

Fix every caller (`git grep -n "calibration_for(" crates`), passing `"v1"` where the caller is not v1-family-aware.

`lib.rs`:

```rust
pub const MODELS: &[&str] = &["rolling", "ep_next", "ep_next_fade", "v1", "v1r"];
pub const COMPONENTS: &[&str] =
    &["team", "minutes", "availability", "shares", "components", "v1", "v1r"];
```

In `component`: `"v1r" => assemble::xp_v1_family_with_fit("v1r", view),`. In `predict`: `"v1r" => assemble::xp_v1r(view),`. In `LEAKAGE_BUILDERS`: `("model:v1r", assemble::xp_v1r),`.

`models_eval.rs:27` and `bench.rs:26`: append `"v1r"` and bump the array length to 5. Then `git grep -n '"v1"' crates` and check each remaining hit: the backtester's spec parsing, `optimize_plan.rs`, and `bench.rs` checks such as "frames with `e_minutes` only" or "`bench_from_minutes` for `v1`" must treat `v1r` like `v1`. Where the code tests `xp == "v1"`, replace with a helper in `fplopt-models/src/lib.rs`:

```rust
/// A v1-family model (its frame has `p_play`, `e_minutes`, …).
pub fn is_v1_family(name: &str) -> bool {
    matches!(name, "v1" | "v1r")
}
```

- [ ] **Step 4: Run the tests:** `cargo test --workspace` → PASS, goldens unchanged.

- [ ] **Step 5: Check the CLI accepts it.** Write one deadline (e.g. `2024-10-19T10:00:00Z`) to a file and run:

```bash
cargo run --release -p fplopt-cli -- --data-dir data models dump --models v1r --deadlines <file> --out <scratch dir>
```

Expected: exit 0 and `<scratch dir>/00/v1r.parquet` written (uncalibrated: v1r has no table yet).

- [ ] **Step 6: Commit.**

```bash
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings
git add -A crates
git commit -m "feat(models): register v1r (v1's params for now), calibration tables keyed by model"
```

---

### Task 2: `gbm-400` — 400 boosting rounds per minutes part

Python (keep bee446a): `MinutesParams.start/sixty/sub = GbmParams(num_boost_round=400)`.

**Files:**
- Modify: `crates/fplopt-models/src/assemble.rs` (`V1Params::v1r`)
- Test: `assemble.rs` tests

- [ ] **Step 1: Failing test.**

```rust
#[test]
fn v1r_minutes_parts_have_400_rounds() {
    let p = V1Params::v1r();
    for part in [&p.minutes.start, &p.minutes.sixty, &p.minutes.sub] {
        assert_eq!(part.num_boost_round, 400);
    }
    assert_eq!(V1Params::default().minutes.start.num_boost_round, 200);
}
```

- [ ] **Step 2:** `cargo test -p fplopt-models v1r_minutes_parts` → FAIL.

- [ ] **Step 3: Implement** in `V1Params::v1r()`:

```rust
pub fn v1r() -> Self {
    let base = V1Params::default();
    let rounds = GbmParams { num_boost_round: 400, ..GbmParams::default() };
    V1Params {
        model: "v1r",
        minutes: MinutesParams {
            start: rounds.clone(),
            sixty: rounds.clone(),
            sub: rounds,
            ..base.minutes.clone()
        },
        ..base
    }
}
```

(`use crate::gbm::GbmParams;` at the top of `assemble.rs`.)

- [ ] **Step 4:** `cargo test --workspace` → PASS.

- [ ] **Step 5: Commit.** `git commit -am "feat(v1r): 400 boosting rounds per minutes part (research bee446a)"`

---

### Task 3: Elo inputs — `win_expectancy` and `club_rating`

Python (keeps 1ff8914, 6e09a1c, `src/fplopt/features/history.py`): two more minutes lag features.
- Training rows: the match's `team_rating` row for (team_key, fixture_key): `expected_score` → `win_expectancy`, `rating_before` → `club_rating` (rows with a null fixture_key or expected_score dropped; duplicates keep the last).
- Horizon rows: each club's newest `rating_after` before the deadline (sorted by team_key, event_time; last wins; NaN if none); `win_expectancy = 1 / (1 + 10^(−edge/400))`, `edge = elo[team] − elo[opponent] + (±1)·60` (+ at home); `club_rating = elo[team]`.
- Both appended **after** `club_fixtures_season` in `LAG_FEATURES` (column order matters to LightGBM).

In Rust, `v1`'s frames and `LAG_FEATURES` must not change (goldens), so the two columns are built by separate one-argument view functions in `fplopt-features` and appended by the minutes model only when `MinutesParams.elo_features` is on.

**Files:**
- Modify: `crates/fplopt-features/src/history.rs`
- Modify: `crates/fplopt-models/src/minutes.rs`, `availability.rs` (the `part` call), `assemble.rs`, `lib.rs` (component probes call `fit_minutes` with rows)
- Test: `crates/fplopt-features/src/history.rs` tests, `minutes.rs` tests

- [ ] **Step 1: Failing feature test** (`history.rs` tests; use the synthetic store helper the existing history tests use — `git grep -n "fn synthetic" crates/fplopt-features` — and its `team_rating` table):

```rust
#[test]
fn match_elo_is_the_matchs_rating_row() {
    let store = synthetic_store();
    let view = store.as_of(last_deadline(&store)).unwrap();
    let elo = match_elo(&view).unwrap();
    let ratings = view
        .table("team_rating", Some(&["team_key", "fixture_key", "expected_score", "rating_before"]))
        .unwrap();
    let (team, fixture) = (ratings.i64("team_key").unwrap(), ratings.opt_i64("fixture_key").unwrap());
    let (score, before) =
        (ratings.opt_f64("expected_score").unwrap(), ratings.f64("rating_before").unwrap());
    let mut checked = 0;
    for i in 0..ratings.height() {
        let (Some(f), Some(s)) = (fixture[i], score[i]) else { continue };
        let (we, rating) = elo[&(team[i], f)];
        assert_eq!(we, s);
        assert_eq!(rating, before[i]);
        checked += 1;
    }
    assert!(checked > 0);
}

#[test]
fn horizon_elo_sides_sum_to_one() {
    let store = synthetic_store();
    let view = store.as_of(deadline_of_gw(&store, 6)).unwrap();
    let elo = horizon_elo(&view).unwrap();
    assert!(!elo.is_empty());
    let mut by_fixture: HashMap<i64, f64> = HashMap::new();
    for (&(_, fixture), &(we, _)) in &elo {
        assert!(we > 0.0 && we < 1.0);
        *by_fixture.entry(fixture).or_default() += we;
    }
    for total in by_fixture.values() {
        assert!((total - 1.0).abs() < 1e-12);
    }
}
```

(Name the helpers after what the existing tests use; if there is no `last_deadline` / `deadline_of_gw`, read the `gameweek` table's `deadline_time` as the existing tests do.)

- [ ] **Step 2:** `cargo test -p fplopt-features elo` → FAIL.

- [ ] **Step 3: Implement** in `history.rs` (near `horizon_fixtures`):

```rust
/// fplopt.build.elo's home advantage (`win_expectancy` at prediction).
const ELO_HOME_ADVANTAGE: f64 = 60.0;

/// The Elo lag features the v1r minutes model adds after `LAG_FEATURES` (research keeps
/// 1ff8914, 6e09a1c): his club's expected score in the fixture and its own rating before it.
pub const ELO_FEATURES: &[&str] = &["win_expectancy", "club_rating"];

/// Per played (team_key, fixture_key) in `team_rating`: (expected_score, rating_before) of
/// the side before the match (Python `_match_expectancy`; a duplicate keeps the last row).
pub fn match_elo(view: &AsOfView) -> Result<HashMap<(i64, i64), (f64, f64)>> {
    let ratings = view.table(
        "team_rating",
        Some(&["team_key", "fixture_key", "expected_score", "rating_before"]),
    )?;
    let team = ratings.i64("team_key")?;
    let fixture = ratings.opt_i64("fixture_key")?;
    let score = ratings.opt_f64("expected_score")?;
    let before = ratings.opt_f64("rating_before")?;
    let mut out = HashMap::new();
    for i in 0..ratings.height() {
        let (Some(f), Some(s)) = (fixture[i], score[i]) else { continue };
        out.insert((team[i], f), (s, before[i].unwrap_or(f64::NAN)));
    }
    Ok(out)
}

/// Per horizon (team_key, fixture_key): (win_expectancy, club_rating) from both clubs'
/// newest `rating_after` before the deadline (NaN without one; Python `_horizon_fixtures`).
pub fn horizon_elo(view: &AsOfView) -> Result<HashMap<(i64, i64), (f64, f64)>> {
    let ratings = view.table("team_rating", Some(&["team_key", "event_time", "rating_after"]))?;
    let team = ratings.i64("team_key")?;
    let time = opt_ts(&ratings, "event_time")?;
    let rating = ratings.opt_f64("rating_after")?;
    let mut order: Vec<usize> = (0..ratings.height()).collect();
    order.sort_by_key(|&i| (team[i], time[i].is_none(), time[i])); // stable: mergesort
    let mut elo: HashMap<i64, f64> = HashMap::new();
    for i in order {
        elo.insert(team[i], rating[i].unwrap_or(f64::NAN));
    }
    let upcoming = upcoming_fixtures(view)?;
    let teams = upcoming.i64("team_key")?;
    let fixtures = upcoming.opt_i64("fixture_key")?;
    let opponents = upcoming.opt_i64("opponent_team_key")?;
    let home = upcoming.opt_bool("is_home")?;
    let lookup = |t: Option<i64>| t.and_then(|t| elo.get(&t).copied()).unwrap_or(f64::NAN);
    let mut out = HashMap::new();
    for i in 0..upcoming.height() {
        let Some(fixture) = fixtures[i] else { continue };
        let sign = match home[i] {
            Some(true) => 1.0,
            Some(false) => -1.0,
            None => f64::NAN,
        };
        let own = lookup(Some(teams[i]));
        let edge = own - lookup(opponents[i]) + sign * ELO_HOME_ADVANTAGE;
        out.insert((teams[i], fixture), (1.0 / (1.0 + 10f64.powf(-edge / 400.0)), own));
    }
    Ok(out)
}
```

Check `opt_bool` exists on `Table` (`git grep -n "fn opt_bool" crates/fplopt-core`); if not, read `is_home` the way `baseline.rs` reads it. Add `"team_rating"` to `leakcheck::FEATURE_TABLES` if it is not already there (it is listed at `leakcheck.rs:53`; confirm).

- [ ] **Step 4:** `cargo test -p fplopt-features elo` → PASS.

- [ ] **Step 5: Failing minutes test** (`minutes.rs` tests):

```rust
#[test]
fn elo_features_only_when_asked() {
    let off = MinutesParams::default();
    let on = MinutesParams { elo_features: true, ..MinutesParams::default() };
    assert_eq!(lag_features(&off), LAG_FEATURES.to_vec());
    let with = lag_features(&on);
    assert_eq!(&with[..LAG_FEATURES.len()], LAG_FEATURES);
    assert_eq!(&with[LAG_FEATURES.len()..], ELO_FEATURES);
}
```

- [ ] **Step 6:** FAIL (no `elo_features`, no `lag_features`).

- [ ] **Step 7: Implement** in `minutes.rs`:

```rust
pub struct MinutesParams {
    // ... existing fields ...
    pub ban_residual: bool,
    /// Append `history::ELO_FEATURES` to the lag features (v1r).
    pub elo_features: bool,
}
// Default: elo_features: false
```

```rust
/// The boosters' features: `LAG_FEATURES`, then `ELO_FEATURES` with `elo_features`.
pub fn lag_features(params: &MinutesParams) -> Vec<&'static str> {
    let mut out = LAG_FEATURES.to_vec();
    if params.elo_features {
        out.extend_from_slice(ELO_FEATURES);
    }
    out
}

/// `table` (rows with team_key, fixture_key) with the Elo columns from `elo` (NaN when a
/// key is missing); unchanged without `elo_features`.
fn with_elo(
    table: Table,
    params: &MinutesParams,
    elo: impl FnOnce() -> Result<HashMap<(i64, i64), (f64, f64)>>,
) -> Result<Table> {
    if !params.elo_features {
        return Ok(table);
    }
    let elo = elo()?;
    let (team, fixture) = (table.i64("team_key")?, table.i64("fixture_key")?);
    let get = |i: usize| elo.get(&(team[i], fixture[i])).copied().unwrap_or((f64::NAN, f64::NAN));
    let n = table.height();
    let we: Vec<f64> = (0..n).map(|i| get(i).0).collect();
    let rating: Vec<f64> = (0..n).map(|i| get(i).1).collect();
    Ok(table.with("win_expectancy", floats(we)).with("club_rating", floats(rating)))
}

/// The minutes model's training frame: `training_frame` plus the Elo columns if asked
/// (shared with the availability fit, like Python's `history`).
pub fn history_frame(view: &AsOfView, params: &MinutesParams) -> Result<Table> {
    with_elo(training_frame(view)?, params, || match_elo(view))
}
```

(`floats` is the features crate's NaN-as-null float column helper; import it, or use `Column::F64` if that is what `prediction_frame` uses for the lag columns — match whichever the training frame uses for `club_fixtures_season`.) Check `team_key`/`fixture_key` are `I64` columns in both frames (they are in `training_frame`; in `prediction_frame` they come from `h(|x| x.team_key)`).

Change `part` to take the features:

```rust
pub(crate) fn part(
    table: &Table,
    weight: &[f64],
    label: &str,
    params: &GbmParams,
    features: &[&'static str],
) -> Result<Part> {
    let labels = float_column(table, label)?;
    if table.height() < MIN_ROWS || nunique(&labels) < 2 {
        return Ok(Part::Constant(weighted_mean(&labels, weight, DEFAULT_RATE)));
    }
    let with_weight = table.clone().with("weight", Column::F64(weight.to_vec()));
    Ok(Part::Gbm(train(&with_weight, features, label, Some("weight"), params)?))
}
```

and pass `&lag_features(params)` at every call (`fit_minutes`' three parts; `availability::out_of_fold_start`). In `fit_minutes`, build the default rows with `history_frame(view, params)` instead of `training_frame(view)`. In `predict_minutes`:

```rust
let frame = with_elo(prediction_frame(view)?, params, || horizon_elo(view))?;
```

In `assemble::fit_v1`, `let history = history_frame(view, &params.minutes)?;`. In `lib.rs`'s component probes, replace `training_frame(view)` with `minutes::history_frame(view, &params)` where `params` is the probe's `MinutesParams`.

Turn it on for v1r in `V1Params::v1r()`: `minutes: MinutesParams { start…, elo_features: true, ..base.minutes.clone() }`.

- [ ] **Step 8:** `cargo test --workspace` → PASS, goldens unchanged.

- [ ] **Step 9: Commit.** `git commit -am "feat(v1r): win_expectancy and club_rating minutes features (research 1ff8914, 6e09a1c)"`

---

### Task 4: `horizon_shrink=0.92` — logit shrink of P(start) by horizon

Python (keep 8eaf71c, `minutes.py` `horizon_start`, used by `predict_minutes` and `availability._fit_rows`):

```python
weight = np.power(params.horizon_decay, horizon)
p_start = weight * p_model + (1.0 - weight) * long_run
if params.horizon_shrink == 1.0:
    return p_start
z = logit(np.clip(p_start, EPS, 1.0 - EPS))
return expit(np.power(params.horizon_shrink, horizon) * z)
```

`EPS` in Python `minutes.py` is the same 1e-6 as Rust's `const EPS` (check both).

**Files:** `crates/fplopt-models/src/minutes.rs`, `availability.rs`, `assemble.rs`; tests in `minutes.rs`.

- [ ] **Step 1: Failing test** (`minutes.rs` tests):

```rust
#[test]
fn horizon_start_decays_then_shrinks() {
    let off = MinutesParams { horizon_decay: 0.5, ..MinutesParams::default() };
    let on = MinutesParams { horizon_shrink: 0.8, ..off.clone() };
    for (p, long, h) in [(0.9, 0.6, 0.0), (0.9, 0.6, 1.0), (0.3, 0.5, 3.0)] {
        let w = 0.5f64.powf(h);
        let blended = w * p + (1.0 - w) * long;
        assert_eq!(horizon_start(p, long, h, &off), blended);
        let z = logit(np_clip(blended, EPS, 1.0 - EPS));
        let shrunk = expit(0.8f64.powf(h) * z);
        assert!((horizon_start(p, long, h, &on) - shrunk).abs() < 1e-15);
    }
    // h = 0 is not moved by the shrink beyond the clip round trip.
    assert!((horizon_start(0.9, 0.6, 0.0, &on) - 0.9).abs() < 1e-12);
}
```

- [ ] **Step 2:** FAIL.

- [ ] **Step 3: Implement.** Field `pub horizon_shrink: f64` on `MinutesParams`, default `1.0`. Function:

```rust
/// P(start) at a horizon: the decay toward `long_run`, then logit × `horizon_shrink ** h`
/// (post-processing step 1; Python `horizon_start`).
pub fn horizon_start(p_model: f64, long_run: f64, horizon: f64, params: &MinutesParams) -> f64 {
    let weight = params.horizon_decay.powf(horizon);
    let p_start = weight * p_model + (1.0 - weight) * long_run;
    if params.horizon_shrink == 1.0 {
        return p_start;
    }
    expit(params.horizon_shrink.powf(horizon) * logit(np_clip(p_start, EPS, 1.0 - EPS)))
}
```

In `predict_minutes` replace the two `weight`/`blended` lines with `let blended = horizon_start(p_model[i], long_run, horizon[i], params);`. In `availability.rs`'s fit rows (around line 249) replace `base: weight * p_model[b] + (1.0 - weight) * long_run` with `base: horizon_start(p_model[b], long_run, horizon as f64, params)` and drop the `weight` binding. `np.power` on float arrays and `f64::powf` agree for these inputs; if parity (Task 13) shows a last-bit difference, use `crate::util`'s power helper if one exists.

v1r: `horizon_shrink: 0.92` in `V1Params::v1r()`'s minutes.

- [ ] **Step 4: The breaking synthetic test.** Translate nothing from it; write its verdict under *Pins* (from Task 0 Step 4): the Python test `test_horizon_decay_moves_later_gws_to_the_long_run_rate` asserted the plain decay with the shrink on, and 1ff8914 relaxed an exact `==` at h0 to `rtol=1e-12` (the clip round trip). Verdict expected: "encodes the old behaviour". If the failing test is a different one, read it and decide.

- [ ] **Step 5:** `cargo test --workspace` → PASS.

- [ ] **Step 6: Commit.** `git commit -am "feat(v1r): horizon_shrink 0.92 on P(start) (research 8eaf71c)"`

---

### Task 5: `category_slopes` — one flag-layer slope per category

Python (keep 3dca529, `availability.py`): θ = [slopes (one per category, or one shared), offsets per cell, back slopes]; each category slope has a Gaussian prior around 1 with sd `SLOPE_SD = 1.0` (in the loss and Hessian, on top of the offsets' `OFFSET_SD` prior, which no longer covers the slopes); start at slopes 1. The fit stores `slopes: [(category, slope)]` for categories with rows and `slope = 1.0`; at prediction a category uses its own slope, a back category its plain category's, an unseen one `fit.slope`.

**Files:** `crates/fplopt-models/src/availability.rs`, `assemble.rs`, `lib.rs` (the three `fit_availability(...)` probe calls); tests in `availability.rs`.

- [ ] **Step 1: Failing tests** (`availability.rs` tests; build the fit on the synthetic store the existing availability tests use):

```rust
#[test]
fn shared_slope_is_unchanged_without_category_slopes() {
    let (view, minutes, history) = synthetic_fit_inputs();
    let fit = fit_availability(&view, &minutes, Some(&history), true, false).unwrap();
    assert!(fit.slopes.is_empty());
}

#[test]
fn category_slopes_one_per_seen_category() {
    let (view, minutes, history) = synthetic_fit_inputs();
    let fit = fit_availability(&view, &minutes, Some(&history), true, true).unwrap();
    if fit.n_rows < MIN_ROWS {
        return; // the identity mapping
    }
    assert_eq!(fit.slope, 1.0);
    assert!(!fit.slopes.is_empty());
    let seen: HashSet<&str> = fit.offsets.iter().map(|(c, _, _)| c.as_str()).collect();
    for (category, slope) in &fit.slopes {
        assert!(seen.contains(category.as_str()));
        assert!(slope.is_finite());
    }
}

#[test]
fn slope_lookup_falls_back_to_plain_then_shared() {
    let fit = AvailabilityFit {
        slope: 0.7,
        slopes: vec![("a".into(), 0.95), ("d50".into(), 0.6)],
        ..identity_fit()
    };
    assert_eq!(slope_for(&fit, "a"), 0.95);
    assert_eq!(slope_for(&fit, "a_back"), 0.95);
    assert_eq!(slope_for(&fit, "i"), 0.7);
    let shared = AvailabilityFit { slope: 0.7, ..identity_fit() };
    assert_eq!(slope_for(&shared, "a"), 0.7);
}
```

(`synthetic_fit_inputs` and `identity_fit` are small helpers to add in the tests module if the existing tests don't have them: the store, `fit_minutes(&view, &MinutesParams::default(), Some(&history))`, and an `AvailabilityFit` with empty vectors.)

- [ ] **Step 2:** FAIL.

- [ ] **Step 3: Implement.**

```rust
const SLOPE_SD: f64 = 1.0; // `category_slopes`: prior sd of each category's slope around 1

pub struct AvailabilityFit {
    // ... existing fields ...
    /// With `category_slopes`, each seen category's slope (Python `slopes`).
    pub slopes: Vec<(String, f64)>,
}

/// The slope a category uses: its own, its plain category's for a back cell, else the
/// shared slope (Python `adjust_minutes`).
fn slope_for(fit: &AvailabilityFit, category: &str) -> f64 {
    let find = |c: &str| fit.slopes.iter().find(|(k, _)| k == c).map(|(_, s)| *s);
    find(category)
        .or_else(|| category.strip_suffix(BACK_SUFFIX).and_then(find))
        .unwrap_or(fit.slope)
}
```

`fit_availability` gets a `category_slopes: bool` parameter. Inside, translate the Python index layout exactly (this is the whole change to the optimization; everything not shown stays):

```rust
// The slopes come first in θ: one shared, or one per category (a cell's category is
// cell / HORIZON_BUCKETS, the product's order).
let n_s = if category_slopes { categories.len() } else { 1 };
let cell_slope: Vec<usize> = (0..n_cells)
    .map(|k| if category_slopes { k / HORIZON_BUCKETS as usize } else { 0 })
    .collect();
let sidx: Vec<usize> = cell.iter().map(|&c| cell_slope[c]).collect();
let n_theta = n_s + n_cells + n_back;
let slope_sd2 = SLOPE_SD * SLOPE_SD;
// per_slope: the shared slope is a numpy sum (np_sum, as before); per category a bincount.
let per_slope = |weights: &dyn Fn(usize) -> f64| -> Vec<f64> {
    if n_s == 1 {
        vec![np_sum(&(0..y.len()).map(weights).collect::<Vec<_>>())]
    } else {
        bincount(&sidx, weights, n_s)
    }
};
let eta_of = |theta: &[f64]| -> Vec<f64> {
    (0..y.len())
        .map(|i| {
            let mut eta = theta[sidx[i]] * z[i] + theta[n_s + cell[i]];
            if n_back > 0 {
                eta += theta[n_s + n_cells + bcell[i]] * x[i];
            }
            eta
        })
        .collect()
};
```

Loss: the offset prior sums `theta[n_s..]`; gradient `grad[..n_s] = per_slope(&|i| residual[i] * z[i])`, `grad[n_s..n_s + n_cells] = by_cell`, back at `n_s + n_cells..`; the prior gradient `theta[j] / sd2` for `j in n_s..n_theta`; and with `category_slopes`:

```rust
if category_slopes {
    let squares: Vec<f64> = theta[..n_s].iter().map(|t| (t - 1.0) * (t - 1.0)).collect();
    value += 0.5 * np_sum(&squares) / slope_sd2;
    for j in 0..n_s {
        grad[j] += (theta[j] - 1.0) / slope_sd2;
    }
}
```

Hessian: diagonal of the slope block `per_slope(&|i| w[i] * z[i] * z[i])`; the slope–offset cross term goes to `(cell_slope[k], n_s + k)` and its transpose; offsets' diagonal at `n_s + k`; back block at `b0 = n_s + n_cells`, its slope cross term at `(cell_slope[back_cells[j]], b0 + j)`; the offset prior `1/sd2` on `n_s..n_theta`; with `category_slopes` add `1/slope_sd2` on `0..n_s`. Start: `theta[..n_s] = 1.0`, rest 0.

Python sums the per-category Hessian cross terms with `np.bincount(cell, weights=w * z)`, then scatters with `out[cell_slope, offset_index] = cross` (assignment, not addition: each offset index appears once). Keep it an assignment.

Result: `slope: if category_slopes { 1.0 } else { theta[0] }`; offsets read `theta[n_s + k]`; back slopes `theta[n_s + n_cells + j]`; `slopes` = `(categories[j], theta[j])` for `j` with at least one row in `sidx` (Python `used_categories`), empty without `category_slopes`. The identity early return and its struct literal get `slopes: Vec::new()`.

In `adjust_minutes`, replace `fit.slope * logit(...)` with `slope_for(fit, &category) * logit(...)`, where `category` is the (possibly back) label of that row — the same variable the offset lookup uses.

`to_json`: add `"slopes": self.slopes.iter().map(|(c, s)| serde_json::json!([c, s])).collect::<Vec<_>>()` **only when non-empty**, so `v1`'s fit JSON (a golden) is unchanged:

```rust
let mut out = serde_json::json!({ /* existing keys */ });
if !self.slopes.is_empty() {
    out["slopes"] = serde_json::json!(
        self.slopes.iter().map(|(c, s)| serde_json::json!([c, s])).collect::<Vec<_>>()
    );
}
out
```

`V1Params`: add `pub category_slopes: bool` (default `false`; `v1r`: `true`), and pass it in `fit_v1`. The three probe calls in `lib.rs` pass `false`.

- [ ] **Step 4:** `cargo test --workspace` → PASS.

- [ ] **Step 5: Commit.** `git commit -am "feat(v1r): one flag-layer slope per category (research 3dca529)"`

---

### Task 6: Quadratic price regression (shared by Tasks 7 and 8)

Python (625a96f, `_price_model(..., quadratic)`): with `quadratic`, columns `[1, z, z²]`, θ₀ = log(Σx/Σd), ridge `PRICE_RIDGE` on b and c, `PRICE_ITERATIONS` Newton steps; returns `(a, b, center, c)`; the flat fallback returns `(…, 0.0)`. Python's gradient and Hessian are `design.T @ …` (BLAS dot products); the Rust 2-term version uses sequential `.sum()`, which the batch 6 gate accepted for `price_goal` (fits 2.9e-14). Use the same sequential sums.

**Files:** `crates/fplopt-models/src/shares.rs`; tests there.

- [ ] **Step 1: Failing tests.**

```rust
#[test]
fn quadratic_price_model_recovers_a_convex_curve() {
    // x = d · exp(a + b z + c z²) exactly, z = (price − center)/10, weights d.
    let (a, b, c) = (-1.0, 0.3, 0.8);
    let prices: Vec<f64> = (0..40).map(|i| 4.0 + 0.25 * i as f64).collect();
    let d = 10.0;
    let center = prices.iter().sum::<f64>() / prices.len() as f64;
    let rows: Vec<(f64, f64, f64)> = prices
        .iter()
        .map(|&p| {
            let z = (p - center) / PRICE_SCALE;
            (p, d * (a + b * z + c * z * z).exp(), d)
        })
        .collect();
    let (fa, fb, fcenter, fc) = price_model_quadratic(&rows);
    assert!((fcenter - center).abs() < 1e-12);
    // The ridge (1.0 on b and c) pulls a little: a loose tolerance.
    assert!((fa - a).abs() < 0.05 && (fb - b).abs() < 0.1 && (fc - c).abs() < 0.2);
}

#[test]
fn linear_price_model_is_unchanged() {
    let rows = vec![(5.0, 1.0, 10.0), (7.0, 2.0, 10.0), (10.0, 4.0, 10.0)];
    let (a, b, center) = price_model(&rows);
    let (qa, qb, qcenter, _) = price_model_quadratic(&rows);
    assert_eq!(center, qcenter);
    assert!(a.is_finite() && b.is_finite() && qa.is_finite() && qb.is_finite());
}
```

- [ ] **Step 2:** FAIL.

- [ ] **Step 3: Implement** next to `price_model`:

```rust
/// `_price_model(..., quadratic=True)`: (a, b, center, c), a c·z² term in the exponent,
/// ridged like b.
fn price_model_quadratic(seasons: &[(f64, f64, f64)]) -> (f64, f64, f64, f64) {
    let rows: Vec<&(f64, f64, f64)> =
        seasons.iter().filter(|(p, _, d)| *d > 0.0 && !p.is_nan()).collect();
    let xs: Vec<f64> = rows.iter().map(|r| r.1).collect();
    if rows.is_empty() || pd_sum(&xs) <= 0.0 {
        let (a, b, center) = price_model(seasons); // the same flat fallback
        return (a, b, center, 0.0);
    }
    let weights: Vec<f64> = rows.iter().map(|r| r.2).collect();
    let price: Vec<f64> = rows.iter().map(|r| r.0).collect();
    let products: Vec<f64> = price.iter().zip(&weights).map(|(p, w)| p * w).collect();
    let center = np_sum(&products) / np_sum(&weights);
    let z: Vec<f64> = price.iter().map(|p| (p - center) / PRICE_SCALE).collect();
    let cols = |i: usize| [1.0, z[i], z[i] * z[i]];
    let mut theta = [(np_sum(&xs) / np_sum(&weights)).ln(), 0.0, 0.0];
    for _ in 0..PRICE_ITERATIONS {
        let mu: Vec<f64> = (0..z.len())
            .map(|i| {
                let c = cols(i);
                weights[i] * (theta[0] * c[0] + theta[1] * c[1] + theta[2] * c[2]).exp()
            })
            .collect();
        let mut g = [0.0; 3];
        let mut h = [[0.0; 3]; 3];
        for (j, gj) in g.iter_mut().enumerate() {
            *gj = (0..z.len()).map(|i| cols(i)[j] * (mu[i] - xs[i])).sum::<f64>();
            for k in 0..3 {
                h[j][k] = (0..z.len()).map(|i| mu[i] * cols(i)[j] * cols(i)[k]).sum::<f64>();
            }
        }
        for j in 1..3 {
            g[j] += PRICE_RIDGE * theta[j];
            h[j][j] += PRICE_RIDGE;
        }
        let step = solve3(h, g);
        for j in 0..3 {
            theta[j] -= step[j];
        }
    }
    (theta[0], theta[1], center, theta[2])
}

/// `np.linalg.solve` of a 3 × 3 system (LU with partial pivoting, as LAPACK's getrf/getrs).
fn solve3(a: [[f64; 3]; 3], b: [f64; 3]) -> [f64; 3] {
    let (mut a, mut b) = (a, b);
    for col in 0..3 {
        let pivot = (col..3).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs())).unwrap();
        a.swap(col, pivot);
        b.swap(col, pivot);
        let inv = 1.0 / a[col][col];
        for row in col + 1..3 {
            let l = a[row][col] * inv;
            for k in col..3 {
                a[row][k] -= l * a[col][k];
            }
            b[row] -= l * b[col];
        }
    }
    let mut x = [0.0; 3];
    for row in (0..3).rev() {
        let s: f64 = (row + 1..3).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - s) / a[row][row];
    }
    x
}

/// exp(c·z²) for a row (z from the position's center in `table`), 1 without a curve
/// (Python `_price_curve`).
fn price_curve(
    table: &[(i64, f64, f64, f64)],
    curve: &[(i64, f64)],
    element_type: f64,
    price: f64,
) -> f64 {
    for &(position, _, _, center) in table {
        if element_type == position as f64 {
            let c = curve.iter().find(|t| t.0 == position).map_or(0.0, |t| t.1);
            let z = (price - center) / PRICE_SCALE;
            return (c * z * z).exp();
        }
    }
    1.0
}
```

`max_by` returns the last of equal maxima; LAPACK's `idamax` returns the first. Use a loop that keeps the first strict maximum instead if Task 13 shows a difference (it only matters on exact ties).

- [ ] **Step 4:** `cargo test -p fplopt-models price_model` → PASS.

- [ ] **Step 5: Commit.** `git commit -am "feat(models): quadratic price regression and price curve (for v1r's shares)"`

---

### Task 7: Finishing par — `position_par`, `price_par`, `price_par_quadratic`

Python (c5a10d2, e81e161, 625a96f, `shares.py`). In `fit_shares`, per position over `recent` rows that have their own npxG (`~fb_goals`):
- `position_finishing[pos] = shrunk_ratio(Σ max(np_goals, 0), Σ x_goals, 1.0, PAR_PRIOR)` with `np_goals = nan_to_num(goals_scored) − nan_to_num(pen_goals)`, `PAR_PRIOR = 25.0`; the sums are pandas/numpy sums over the group (`float(np.maximum(...).sum())` is a numpy pairwise sum: `np_sum`; `own["x_goals"].sum()` is a pandas sum with NaN as 0: `pd_sum`).
- With `price_par`: per (player, season) of those rows, `fin = Σ max(np_goals,0)`, `fin_xg = Σ x_goals` (pandas groupby sums: `Kahan`, sorted by key), joined to the same `first_price` as the price prior; if `Σ fin_xg ≥ PAR_PRIOR`, fit `price_model(_quadratic)` on `(price, fin, fin_xg)` and store `price_finishing[pos] = (a, b, center)` and, quadratic, `price_finishing_curve[pos] = c`.

In `marcel_shares`: `par = position_finishing[pos]` (1 if absent or NaN position); with `price_par` and a non-empty `price_finishing`: `by_price = price_prior(price_finishing, pos, price)` × (quadratic) `price_curve(...)`; `par = by_price` unless NaN. Finishing becomes `(fin_goals + k·par) / (fin_xg + k)`.

**Files:** `crates/fplopt-models/src/shares.rs`, `assemble.rs`; tests in `shares.rs`.

- [ ] **Step 1: Failing tests** (on the synthetic store the existing shares tests use):

```rust
#[test]
fn finishing_par_fields_empty_by_default() {
    let fit = fit_shares(&synthetic_view(), &SharesParams::default()).unwrap();
    assert!(fit.position_finishing.is_empty());
    assert!(fit.price_finishing.is_empty());
    assert!(fit.price_finishing_curve.is_empty());
}

#[test]
fn position_par_is_np_goals_over_npxg_shrunk_to_one() {
    let params = SharesParams { position_par: true, ..SharesParams::default() };
    let fit = fit_shares(&synthetic_view(), &params).unwrap();
    assert!(!fit.position_finishing.is_empty());
    for (_, par) in &fit.position_finishing {
        assert!(*par > 0.0 && par.is_finite());
    }
}

#[test]
fn price_par_curve_only_when_quadratic() {
    let linear = SharesParams { position_par: true, price_par: true, ..SharesParams::default() };
    let quad = SharesParams { price_par_quadratic: true, ..linear.clone() };
    let view = synthetic_view();
    let l = fit_shares(&view, &linear).unwrap();
    let q = fit_shares(&view, &quad).unwrap();
    assert!(l.price_finishing_curve.is_empty());
    assert_eq!(q.price_finishing.len(), q.price_finishing_curve.len());
}
```

If the synthetic league has too few goals for any position to reach `PAR_PRIOR` npxG, `price_finishing` is empty and the third test's `assert_eq!` passes trivially; add an assertion that `q.price_finishing` is non-empty only if the synthetic data supports it (check once, then keep or drop that line).

- [ ] **Step 2:** FAIL.

- [ ] **Step 3: Implement.**

```rust
const PAR_PRIOR: f64 = 25.0; // pseudo-npxG at 1 behind each position's finishing par

pub struct SharesParams {
    // ... existing ...
    /// Finishing par per position (np goals / npxG of the position) instead of 1.
    pub position_par: bool,
    /// With position_par: the par by position and price (`price_finishing`).
    pub price_par: bool,
    /// With price_par: a z² term in that regression (`price_finishing_curve`).
    pub price_par_quadratic: bool,
}
// Default: all three false.

pub struct SharesFit {
    // ... existing ...
    pub position_finishing: Vec<(i64, f64)>,
    pub price_finishing: Vec<(i64, f64, f64, f64)>,
    pub price_finishing_curve: Vec<(i64, f64)>,
}
```

In `fit_shares`, inside `for (position, group) in by_position`, after `position_assist.push(...)`:

```rust
if params.position_par {
    let own: Vec<&&ShareRow> = group.iter().filter(|s| !s.fb_goals).collect();
    let np_goals: Vec<f64> = own
        .iter()
        .map(|s| (nan_to_num(s.row.goals_scored) - nan_to_num(s.row.pen_goals)).max(0.0))
        .collect();
    let xg: Vec<f64> = own.iter().map(|s| s.x_goals).collect();
    position_finishing.push((position, shrunk_ratio(np_sum(&np_goals), pd_sum(&xg), 1.0, PAR_PRIOR)));
    if params.price_par {
        let mut by_season: BTreeMap<(i64, i64), [Kahan; 2]> = BTreeMap::new();
        for (k, s) in own.iter().enumerate() {
            let e = by_season.entry((s.row.player_key, s.row.season)).or_default();
            e[0].add(np_goals[k]);
            if !xg[k].is_nan() {
                e[1].add(xg[k]);
            }
        }
        let table: Vec<(f64, f64, f64)> = by_season
            .iter()
            .map(|(key, v)| {
                (first_price.get(key).copied().unwrap_or(f64::NAN), v[0].value(), v[1].value())
            })
            .collect();
        let total_xg = pd_sum(&table.iter().map(|t| t.2).collect::<Vec<_>>());
        if total_xg >= PAR_PRIOR {
            if params.price_par_quadratic {
                let (a, b, center, c) = price_model_quadratic(&table);
                price_finishing.push((position, a, b, center));
                price_finishing_curve.push((position, c));
            } else {
                let (a, b, center) = price_model(&table);
                price_finishing.push((position, a, b, center));
            }
        }
    }
}
```

`first_price` is built a few lines later in the loop today; move its construction above this block. Python's `np.maximum(np_goals, 0.0)` per row then sums; `np_goals` above is already clipped per row, matching. Python's `pen_goals` is `own["pen_goals"]` with `nan_to_num`, same as here.

Declare `let (mut position_finishing, mut price_finishing, mut price_finishing_curve) = (Vec::new(), Vec::new(), Vec::new());` before the loop and add the three fields to the `SharesFit` literal.

In `marcel_shares`:

```rust
let finishing = match params.finishing_prior {
    None => 1.0,
    Some(k) => {
        let mut par = if params.position_par {
            position_finishing_par(&fit.position_finishing, element[i])
        } else {
            1.0
        };
        if params.position_par && params.price_par && !fit.price_finishing.is_empty() {
            let mut by_price = price_prior(&fit.price_finishing, element[i], prices[i]);
            if params.price_par_quadratic && !fit.price_finishing_curve.is_empty() {
                by_price *= price_curve(
                    &fit.price_finishing,
                    &fit.price_finishing_curve,
                    element[i],
                    prices[i],
                );
            }
            if !by_price.is_nan() {
                par = by_price;
            }
        }
        (v(|s| &s.fin_goals) + k * par) / (v(|s| &s.fin_xg) + k)
    }
};
```

```rust
/// A position's finishing par, 1 for an unknown or missing position.
fn position_finishing_par(table: &[(i64, f64)], element_type: f64) -> f64 {
    if element_type.is_nan() {
        return 1.0;
    }
    table.iter().find(|t| t.0 as f64 == element_type).map_or(1.0, |t| t.1)
}
```

`to_json`: add the three keys only when non-empty (as in Task 5), so `v1`'s golden is unchanged.

v1r: `shares: SharesParams { position_par: true, price_par: true, price_par_quadratic: true, ..base.shares.clone() }`.

- [ ] **Step 4: The breaking synthetic test.** Write its verdict under *Pins* (Task 0 Step 4). With par below 1 for forwards, a test that expects finishing ≥ 1 for a player at npxG par fails; that encodes the old par of 1. If the failing test is something else, read it and decide.

- [ ] **Step 5:** `cargo test --workspace` → PASS.

- [ ] **Step 6: Commit.** `git commit -am "feat(v1r): finishing par by position and price (research c5a10d2, e81e161, 625a96f)"`

---

### Task 8: `price_prior_quadratic` — z² term in the goal and assist priors

Python (dbb3265): with `price_prior_quadratic`, `price_goal`/`price_assist` come from `_price_model(..., quadratic=True)` (the stored tuple keeps `(a, b, center)`, the c's go to `price_goal_curve` / `price_assist_curve`), and in `marcel_shares` the price prior is multiplied by `price_curve`. The `use_price` / NaN fallbacks are unchanged.

**Files:** `crates/fplopt-models/src/shares.rs`, `assemble.rs`; tests in `shares.rs`.

- [ ] **Step 1: Failing test.**

```rust
#[test]
fn prior_curves_only_when_quadratic() {
    let view = synthetic_view();
    let off = fit_shares(&view, &SharesParams::default()).unwrap();
    assert!(off.price_goal_curve.is_empty() && off.price_assist_curve.is_empty());
    let on = SharesParams { price_prior_quadratic: true, ..SharesParams::default() };
    let fit = fit_shares(&view, &on).unwrap();
    assert_eq!(fit.price_goal_curve.len(), fit.price_goal.len());
    assert_eq!(fit.price_assist_curve.len(), fit.price_assist.len());
}
```

- [ ] **Step 2:** FAIL.

- [ ] **Step 3: Implement.** `pub price_prior_quadratic: bool` (default false), `pub price_goal_curve: Vec<(i64, f64)>`, `pub price_assist_curve: Vec<(i64, f64)>`. In `fit_shares` replace the two `price_model` calls:

```rust
let goal_rows: Vec<(f64, f64, f64)> = table.iter().map(|(p, v)| (*p, v[0], v[1])).collect();
let assist_rows: Vec<(f64, f64, f64)> = table.iter().map(|(p, v)| (*p, v[2], v[3])).collect();
if params.price_prior_quadratic {
    let (a, b, center, c) = price_model_quadratic(&goal_rows);
    price_goal.push((position, a, b, center));
    price_goal_curve.push((position, c));
    let (a, b, center, c) = price_model_quadratic(&assist_rows);
    price_assist.push((position, a, b, center));
    price_assist_curve.push((position, c));
} else {
    let goal = price_model(&goal_rows);
    let assist = price_model(&assist_rows);
    price_goal.push((position, goal.0, goal.1, goal.2));
    price_assist.push((position, assist.0, assist.1, assist.2));
}
```

In `marcel_shares`, Python computes `by_price` for every row and multiplies by the curve, then picks; mirror it:

```rust
let mut by_price_g = price_prior(&fit.price_goal, element[i], prices[i]);
let mut by_price_a = price_prior(&fit.price_assist, element[i], prices[i]);
if params.price_prior_quadratic {
    by_price_g *= price_curve(&fit.price_goal, &fit.price_goal_curve, element[i], prices[i]);
    by_price_a *= price_curve(&fit.price_assist, &fit.price_assist_curve, element[i], prices[i]);
}
let mut prior_g =
    if use_price { by_price_g } else { position_prior(&fit.position_goal, element[i]) };
let mut prior_a =
    if use_price { by_price_a } else { position_prior(&fit.position_assist, element[i]) };
// NaN fallbacks unchanged
```

`to_json`: the two curve keys only when non-empty. v1r: `price_prior_quadratic: true`.

- [ ] **Step 4:** `cargo test --workspace` → PASS.

- [ ] **Step 5: Commit.** `git commit -am "feat(v1r): quadratic goal and assist price priors (research dbb3265)"`

---

### Task 9: Check `V1Params::v1r()` against the research defaults

**Files:** `crates/fplopt-models/src/assemble.rs` tests.

- [ ] **Step 1: Test the whole constructor** (one place that lists every switch, so a missed one fails loudly):

```rust
#[test]
fn v1r_turns_on_every_keep() {
    let p = V1Params::v1r();
    assert_eq!(p.minutes.start.num_boost_round, 400);
    assert_eq!(p.minutes.sixty.num_boost_round, 400);
    assert_eq!(p.minutes.sub.num_boost_round, 400);
    assert!(p.minutes.elo_features);
    assert_eq!(p.minutes.horizon_shrink, 0.92);
    assert!(p.minutes.ban_residual);
    assert!(p.category_slopes && p.back_split);
    assert!(p.shares.position_par && p.shares.price_par && p.shares.price_par_quadratic);
    assert!(p.shares.price_prior_quadratic);
    // Everything else is v1's.
    let v1 = V1Params::default();
    assert_eq!(p.team, v1.team);
    assert_eq!(p.components, v1.components);
    assert_eq!(p.minutes.horizon_decay, v1.minutes.horizon_decay);
    assert_eq!(p.shares.finishing_prior, v1.shares.finishing_prior);
}
```

- [ ] **Step 2:** Compare with the research pin's Python defaults: `git show <pin>:src/fplopt/models/assemble.py` (`V1Params`), `minutes.py` (`MinutesParams`), `shares.py` (`SharesParams`). Every Python field that differs from main's must be one of the switches above. Any other difference: stop and add it to the spec (it's an unlisted keep).

- [ ] **Step 3:** `cargo test --workspace` → PASS. Commit: `git commit -am "test(v1r): every keep on, the rest v1"`

---

### Task 10: Parity tooling against Python `research`

`dev/models_parity.py` imports `fplopt` in-process and calls `fplopt-rs models dump`. Python's model is `v1` with the research defaults; Rust's is `v1r`.

**Files:** `dev/models_parity.py`.

- [ ] **Step 1: Add the two flags.** Before the `from fplopt ...` imports (which follow a `sys.path` setup, hence the `# noqa: E402`), pre-parse `--python-src`:

```python
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--python-src", type=Path)
_known, _ = _pre.parse_known_args()
if _known.python_src is not None:
    sys.path.insert(0, str(_known.python_src.resolve()))
```

In `main()`'s parser:

```python
parser.add_argument(
    "--python-src",
    type=Path,
    help="import fplopt from this src/ (e.g. a research checkout); read before the imports",
)
parser.add_argument(
    "--rust-model",
    action="append",
    default=[],
    help="PY=RUST: compare Python model PY with Rust model RUST (e.g. v1=v1r)",
)
```

and where the script builds the Rust `models dump --models ...` list and pairs frames by name, map each Python name through `dict(s.split("=") for s in args.rust_model)` (default: same name). Print `fplopt imported from <fplopt.__file__>` at start so the run log shows which Python was compared.

Add `--no-calibrate` (store_true). Rust's `v1r` is uncalibrated until Task 12 (no table), so the Python side must be too. In `_v1`, when the flag is set, fit and predict with calibration off:

```python
def _v1(view):
    model = MODELS["v1"]
    cutoff = model.cutoff(view)
    if NO_CALIBRATE:
        fit = fit_v1(view.earlier(cutoff), dataclasses.replace(V1Params(), calibrate=False))
        frame = predict_v1(view, fit)
    else:
        fit = model.fit(view.earlier(cutoff))
        frame = model.predict(view, fit)
    values = {
        "team": _values(fit.team),
        "minutes": _minutes_values(fit.minutes),
        "availability": _values(fit.availability, ()),
        "shares": _values(fit.shares, ("cutoff", "params", "coverage")),
        "components": _components_values(fit.components),
    }
    return frame, values
```

with `NO_CALIBRATE` a module global set in `main()` from the flag, and `from fplopt.models.assemble import V1Params, fit_v1, predict_v1` (check those names exist in the research pin's `assemble.py`). `V1Params()` there is research's, with every keep on.

- [ ] **Step 2: Smoke test on v1 vs v1** (no mapping): `uv run python dev/models_parity.py --models v1 --n-random 1` → same result as before the change.

- [ ] **Step 3: Commit.** `git commit -am "dev(models_parity): --python-src and --rust-model for the v1r gate"`

---

### Task 11: The v1r parity gate

Batch 6's gate, Python `research` at the pin vs Rust `v1r`, both **uncalibrated** (`--no-calibrate`, Task 10; Rust's `v1r` has no table yet, so it is uncalibrated by construction).

- [ ] **Step 1: Build and run** (locally or on the server, data from the same `data/` build; the research checkout at the pin):

```bash
cargo build --release -p fplopt-cli
cd <research pin worktree> && uv sync && cd -
uv run --project <research pin worktree> python dev/models_parity.py \
  --python-src <research pin worktree>/src --models v1 --rust-model v1=v1r \
  --seasons 2025,2026 --plans --no-calibrate --keep results/parity/v1r
```

Also run `--models minutes,availability,shares --rust-model minutes=minutes ...` with the component probes set to v1r's params if the per-component diff is needed to localize a failure.

- [ ] **Step 2: Gate** (spec section 1; batch 6's tolerances): every fitted value within 1e-6 relative; xP per player-GW max abs diff ≤ 0.01; plan #1 identical for the template squad (a tie within the planner's 0.5% MIP gap counts as identical, shown by the ±1e-9 noise test batch 6 used). Expected first-run failures and where to look: booster text differs → feature column order or values (Task 3; compare the training frames' Elo columns at rtol 0 with `check parity table`); availability fit off by > 1e-6 → θ layout or prior terms (Task 5); shares fit → sum kinds (Task 7).

- [ ] **Step 3: Record** the gate result (deadlines, worst differences) under *Pins* and in the experiment log (the script logs family `rust parity`). Commit: `git commit -am "gate(v1r): parity with research <pin>"`

---

### Task 12: Rust calibration fitter (`fplopt-rs models calibrate`)

Port the part of `dev/calibrate_v1.py` v1 uses: `walk` (per season, the walk-forward uncalibrated per player-fixture components at every deadline, fitted at the refit cutoff, joined to the realized start) and `fit` for `p_start` with horizon groups, `--first-fit 2016`, and `--write`. Not ported: `score`, the variant report, `ban`, the `p_cs`/`p_goal`/`xp` parts (YAGNI: neither table uses them; spec: v1r uses v1's parts). v1's current table says *Parts: p_start:1+; fitted from season 2016; all horizons pooled* — i.e. one isotonic map for horizons ≥ 1, none for h0.

**Files:**
- Create: `crates/fplopt-cli/src/calibrate.rs`
- Modify: `crates/fplopt-cli/src/models.rs` (subcommand), `crates/fplopt-cli/src/main.rs` (dispatch)
- Modify: `crates/fplopt-models/src/calibration.rs` (`isotonic` fit, `pav`)
- Create (by running): `crates/fplopt-models/calibration_table_v1r.json`
- Test: `calibration.rs` tests, `crates/fplopt-cli/tests/calibrate.rs`

- [ ] **Step 1: Failing tests for the isotonic fit** (`calibration.rs`), against Python's `fplopt.models.calibration.isotonic` on fixed inputs. Generate the expected values once with Python and paste them in:

```bash
uv run python -c "
import numpy as np
from fplopt.models.calibration import isotonic
rng = np.random.default_rng(0)
p = rng.uniform(0.01, 0.99, 1000); y = (rng.uniform(size=1000) < p**1.3).astype(float)
m = isotonic(p, y)
print(len(m.x)); print(list(m.x[:5]), list(m.y[:5])); print(list(m.x[-3:]), list(m.y[-3:]))
np.savetxt('tests/fixtures/rust_golden/isotonic_in.csv', np.column_stack([p, y]), delimiter=',', fmt='%.17g')
np.savetxt('tests/fixtures/rust_golden/isotonic_out.csv', np.column_stack([m.x, m.y]), delimiter=',', fmt='%.17g')
"
```

```rust
#[test]
fn isotonic_matches_python() {
    let read = |name: &str| -> Vec<(f64, f64)> {
        let path = format!("{}/../../tests/fixtures/rust_golden/{name}", env!("CARGO_MANIFEST_DIR"));
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|l| {
                let mut it = l.split(',').map(|v| v.parse::<f64>().unwrap());
                (it.next().unwrap(), it.next().unwrap())
            })
            .collect()
    };
    let input = read("isotonic_in.csv");
    let expected = read("isotonic_out.csv");
    let (p, y): (Vec<f64>, Vec<f64>) = input.into_iter().unzip();
    let map = isotonic(&p, &y, None, MAX_BINS);
    assert_eq!(map.x.len(), expected.len());
    for (k, (x, v)) in expected.iter().enumerate() {
        assert_eq!(map.x[k], *x);
        assert_eq!(map.y[k], *v);
    }
}

#[test]
fn isotonic_skips_zeros_and_is_monotone() {
    let p = [0.0, 0.2, 0.4, 0.6, 0.8];
    let y = [1.0, 0.0, 1.0, 0.0, 1.0];
    let map = isotonic(&p, &y, None, MAX_BINS);
    assert!(map.x.iter().all(|&x| x > 0.0));
    assert!(map.y.windows(2).all(|w| w[0] <= w[1]));
}
```

- [ ] **Step 2:** FAIL.

- [ ] **Step 3: Implement** in `calibration.rs` (Python `isotonic`, `_pav`; `DIGITS = 6` rounding with `np.round`'s half-to-even — use `crate::util::np_round` if it exists, else add it next to the other numpy helpers):

```rust
pub const MAX_BINS: usize = 200;
const DIGITS: i32 = 6;

/// Weighted pool-adjacent-violators: the non-decreasing fit of `y` (Python `_pav`).
fn pav(y: &[f64], w: &[f64]) -> Vec<f64> {
    let mut level: Vec<f64> = Vec::new();
    let mut weight: Vec<f64> = Vec::new();
    let mut count: Vec<usize> = Vec::new();
    for (&yi, &wi) in y.iter().zip(w) {
        level.push(yi);
        weight.push(wi);
        count.push(1);
        while level.len() > 1 && level[level.len() - 2] > level[level.len() - 1] {
            let (l1, w1, c1) = (level.pop().unwrap(), weight.pop().unwrap(), count.pop().unwrap());
            let k = level.len() - 1;
            let total = weight[k] + w1;
            level[k] = (level[k] * weight[k] + l1 * w1) / total;
            weight[k] = total;
            count[k] += c1;
        }
    }
    level.iter().zip(&count).flat_map(|(&l, &c)| std::iter::repeat_n(l, c)).collect()
}
```

**Before writing `pav`, read Python's `_pav` (`src/fplopt/models/calibration.py:67`) and translate it line for line instead if it differs from the stack form above** (merge order changes the last bits, and the golden test compares exactly).

```rust
/// The isotonic map of `outcome` on `pred` (Python `isotonic`): rows with pred ≤ 0 or a
/// non-finite value or weight ≤ 0 left out; pooled into at most `bins` equal-count bins;
/// one breakpoint per PAV block, rounded to `DIGITS`.
pub fn isotonic(pred: &[f64], outcome: &[f64], weight: Option<&[f64]>, bins: usize) -> Isotonic {
    let ones = vec![1.0; pred.len()];
    let weight = weight.unwrap_or(&ones);
    let mut rows: Vec<usize> = (0..pred.len())
        .filter(|&i| {
            pred[i] > 0.0 && pred[i].is_finite() && outcome[i].is_finite() && weight[i] > 0.0
        })
        .collect();
    if rows.is_empty() {
        return Isotonic::default();
    }
    rows.sort_by(|&a, &b| pred[a].total_cmp(&pred[b])); // stable, as mergesort
    let n = rows.len();
    let n_bins = bins.min(n);
    // np.unique(np.linspace(0, n, n_bins + 1).astype(int64))
    let mut edges: Vec<usize> =
        (0..=n_bins).map(|k| (k as f64 * n as f64 / n_bins as f64) as usize).collect();
    edges.dedup();
    let mut w = vec![0.0; edges.len() - 1];
    let mut wx = vec![0.0; edges.len() - 1];
    let mut wy = vec![0.0; edges.len() - 1];
    let mut code = 0;
    for (pos, &i) in rows.iter().enumerate() {
        while code + 1 < edges.len() - 1 && pos >= edges[code + 1] {
            code += 1;
        }
        w[code] += weight[i];
        wx[code] += weight[i] * pred[i];
        wy[code] += weight[i] * outcome[i];
    }
    let x: Vec<f64> = (0..w.len()).map(|k| wx[k] / w[k]).collect();
    let y: Vec<f64> = (0..w.len()).map(|k| wy[k] / w[k]).collect();
    let fitted = pav(&y, &w);
    let (mut bx, mut by, mut bw) = (Vec::new(), Vec::new(), Vec::new());
    for k in 0..fitted.len() {
        if k == 0 || fitted[k] != fitted[k - 1] {
            bx.push(0.0);
            by.push(0.0);
            bw.push(0.0);
        }
        let b = bw.len() - 1;
        bw[b] += w[k];
        bx[b] += w[k] * x[k];
        by[b] += w[k] * fitted[k];
    }
    Isotonic {
        x: (0..bw.len()).map(|b| np_round(bx[b] / bw[b], DIGITS)).collect(),
        y: (0..bw.len()).map(|b| np_round(by[b] / bw[b], DIGITS)).collect(),
    }
}
```

`np.linspace(0, n, m+1)` computes `start + k*step` with `step = n/m` (and sets the last point to `n` exactly); `astype(int64)` truncates. Check the edges against Python on n = 1000, bins 200 and on n = 7, bins 200 in a test (`[0..=7]`).

- [ ] **Step 4:** `cargo test -p fplopt-models isotonic` → PASS.

- [ ] **Step 5: The CLI.** `crates/fplopt-cli/src/models.rs`:

```rust
    /// Fit a v1-family model's calibration table (Python `dev/calibrate_v1.py`, the
    /// p_start part only).
    #[command(subcommand)]
    Calibrate(crate::calibrate::CalibrateJob),
```

`crates/fplopt-cli/src/calibrate.rs`:

```rust
//! `models calibrate walk|fit`: the walk-forward rows of a v1-family model and the
//! expanding-window P(start) calibration fitted on them (Python `dev/calibrate_v1.py`'s
//! `walk` and `fit`, part `p_start` only; `score`, `ban` and the other parts not ported).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Subcommand;

pub const LAST_FIT_SEASON: i64 = 2024; // never the 2025/26 holdout

#[derive(Subcommand)]
pub enum CalibrateJob {
    /// Per season, the uncalibrated per player-fixture P(start) at every deadline and the
    /// realized start; writes <out>/<season>.parquet.
    Walk {
        #[arg(long)]
        model: String,
        #[arg(long)]
        seasons: String,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        jobs: Option<usize>,
    },
    /// From a walk, the table: for each season S, the P(start) map(s) fitted on the walk
    /// rows of seasons [first_fit, S); with --write, the model's calibration JSON.
    Fit {
        #[arg(long)]
        model: String,
        #[arg(long)]
        walk: PathBuf,
        /// Horizon groups of the p_start part: "1+" (v1's: horizons ≥ 1 pooled, h0 none),
        /// "all", or "0,1+".
        #[arg(long, default_value = "1+")]
        groups: String,
        #[arg(long, default_value_t = 2016)]
        first_fit: i64,
        #[arg(long)]
        write: bool,
    },
}
```

Walk, per season (port `walk_season`; parallel over seasons with the same pool helper `models eval` uses):

```rust
/// One season's walk rows: (season, deadline_gw_index, player_key, fixture_key, horizon,
/// p_start, r_start). Fits memoized per refit cutoff, as Python's `fits` dict.
fn walk_season(data: &Path, model: &str, season: i64) -> Result<Table> {
    let store = fplopt_store::DataStore::open(data)?;
    let mut params = fplopt_models::assemble::V1Params::for_model(model)?;
    params.calibrate = false;
    let realized = realized_starts(&store, season)?; // (player_key, fixture_key) → r_start
    let played = played_sides(&store, season)?; // (fixture_key, team_key) in team_match
    let mut fits: HashMap<i64, fplopt_models::assemble::V1Fit> = HashMap::new();
    let mut out = WalkRows::default();
    for (gw_index, deadline) in season_deadlines(&store, season)? {
        let view = store.as_of(deadline)?;
        let cutoff = fplopt_models::fitted::refit_deadline(&view, fplopt_models::fitted::REFIT_EVERY)?;
        if !fits.contains_key(&cutoff) {
            fits.insert(cutoff, fplopt_models::assemble::fit_v1(&view.earlier(cutoff)?, &params)?);
        }
        for f in fplopt_models::assemble::fixture_components(&view, &fits[&cutoff])? {
            if !played.contains(&(f.key.fixture_key, f.key.team_key)) {
                continue; // Python's inner join to team_match
            }
            let r = realized.get(&(f.key.player_key, f.key.fixture_key)).copied().unwrap_or(0.0);
            out.push(season, gw_index, &f, r);
        }
    }
    Ok(out.into_table())
}
```

(`Key`'s field names: check `crate::minutes::Key`; `WalkRows` is a small column builder in this file.) `realized_starts`: per (player_key, fixture_key) of the season's `player_match` rows (inner-joined to `player_season` positions, as Python), `starts` where present else the inferred start; the training frame's `start` column at a deadline after the season is the same quantity — reuse it (`fplopt_features::history::training_frame` on a view past the season's last deadline, filtered to the season) and confirm with the gate in Step 7 rather than re-implementing `infer_starts`.

Fit (port `fit` + `fit_calibration` for `p_start` + `write_table`):

```rust
fn fit_table(walk: &Path, groups: &[(i64, Option<i64>)], first_fit: i64) -> Result<Vec<(i64, Calibration)>> {
    let seasons = walk_seasons(walk)?; // the <season>.parquet files present, sorted
    let mut table = Vec::new();
    for &s in &seasons {
        if s <= first_fit {
            continue;
        }
        let used: Vec<i64> = seasons.iter().copied().filter(|&t| t >= first_fit && t < s).collect();
        let rows = load_walk(walk, &used)?; // concatenated in season order
        let maps = groups
            .iter()
            .map(|&(first, stop)| {
                let keep: Vec<usize> = (0..rows.len())
                    .filter(|&i| rows.horizon[i] >= first && rows.horizon[i] < stop.unwrap_or(99))
                    .collect();
                let p: Vec<f64> = keep.iter().map(|&i| rows.p_start[i]).collect();
                let y: Vec<f64> = keep.iter().map(|&i| rows.r_start[i]).collect();
                (first, isotonic(&p, &y, None, MAX_BINS))
            })
            .collect();
        table.push((s, Calibration { p_start: Some(maps), ..Calibration::default() }));
    }
    // The last entry, fitted on every season through LAST_FIT_SEASON, also covers later
    // seasons (`calibration_for` uses the last entry): Python appends S = last + 1.
    Ok(table)
}
```

Read Python's `fit()` (`dev/calibrate_v1.py:363`) for exactly which S values get entries (it iterates `--apply` seasons and adds the entry used after the walk's last season) and match it; the Step 7 gate compares the whole table.

`--write` writes `crates/fplopt-models/calibration_table_<model>.json` in `calibration_table.json`'s layout: `{"params": <fingerprint>, "table": [[S, {"p_start": [[first, {"x": [...], "y": [...]}], ...]}], ...]}` — check the existing file for the exact key names (`git show HEAD:crates/fplopt-models/calibration_table.json | head -c 2000`) and write floats with `fplopt_core::pyjson` so the file diffs cleanly. The fingerprint: a `V1Params::fingerprint()` method (`format!("{:?}", params with calibrate = true)`) — it only has to be stable for Rust; v1's file keeps Python's string.

For `v1r`, set `V1R_TABLE_TEXT` in `calibration.rs` to `Some(include_str!("../calibration_table_v1r.json"))` after the file exists, and make `calibration_for` refuse a table whose `params` differs from `V1Params::for_model(model)?.fingerprint()` for `v1r` (v1 keeps the existing equality check).

- [ ] **Step 6: Integration test** (`crates/fplopt-cli/tests/calibrate.rs`): on the synthetic store fixture the CLI tests use, `walk` over two synthetic seasons then `fit --groups 1+` produces one entry whose map is non-decreasing with every `x > 0`. Run, see it fail, implement, see it pass.

- [ ] **Step 7: Gate — reproduce v1's table.** On the server (the walk is heavy: v1 fits per refit cutoff for 2016–2024; budget hours with `nice -n 15`, at most 12 jobs):

```bash
target/release/fplopt-rs --data-dir data models calibrate walk --model v1 --seasons 2016-2024 --out results/calibrate-v1-rs
target/release/fplopt-rs models calibrate fit --model v1 --walk results/calibrate-v1-rs --groups 1+
```

Pass: every entry's breakpoints equal `calibration_table.json`'s after the 6-digit rounding (allow a difference of one unit in the 6th digit at ≤ 1% of breakpoints, from rounding ties; anything more is a port bug). Log the run (family `rust parity`, name `calibrate v1`).

- [ ] **Step 8: Fit v1r's table** (same command, `--model v1r --out results/calibrate-v1r-rs`, then `fit --model v1r --groups 1+ --write`). Same parts and groups as v1, fixed here, not chosen by score (spec section 1). Set `calibrate: true` in `V1Params::v1r()` (Task 1's review left it `false` until this table exists, so no run is logged as `v1r` uncalibrated by accident) and update that test. Commit the JSON, the `V1R_TABLE_TEXT` change and the flag.

- [ ] **Step 9:** `cargo test --workspace` → PASS. Commit: `git commit -am "feat(models): Rust calibration fitter (p_start); v1r's table"`

---

### Task 13: Leakage and v1 non-regression

- [ ] **Step 1: Leakage.**

```bash
target/release/fplopt-rs check leakage --deadlines 12
```

It runs every `LEAKAGE_BUILDERS` entry, now including `model:v1r`. Pass: no feature or model fails. The Elo inputs are the risk: `match_elo` reads `team_rating` rows for past matches only through the view, and `horizon_elo` reads `rating_after` before the deadline; a failure here means a row's `available_at` lets a post-deadline rating in.

- [ ] **Step 2: v1 unchanged.** Re-run the batch 6 gate for `v1` on the branch: `uv run python dev/models_parity.py --models v1 --seasons 2025,2026` → identical to batch 6's result (fits ≤ 1e-6, xP ≤ 0.01). And `cargo test --workspace` goldens unchanged (they are, if every task kept them).

- [ ] **Step 3: `live prepare` timing.** On the server's shadow checkout, time `fplopt-rs live prepare` with `LIVE_XP` temporarily `"v1r"` (a local edit, not committed): expect 6–8 min vs about 4 for v1; must stay under the 15-minute tick with margin (≤ 10 min). Record under *Pins*. Revert the edit.

- [ ] **Step 4: Commit the records.** `git commit -am "gate(v1r): leakage, v1 unchanged, live prepare timing"`

---

### Task 14: Pre-register and freeze

Nothing in the spec's section 2 changes after this task.

- [ ] **Step 1:** Re-read the spec's section 2 against the code: the CLI flags it names exist and mean what it says (`models eval --models v1,v1r --seasons 2023-2024`; `backtest compare` with `--continuation own` and those starts; the per-decision window mean; xG-scored mean). If a flag differs, fix the spec's command **now** (before any run) and note the change in its header.

- [ ] **Step 2: Tag.**

```bash
git tag -a v1r-preregistered -m "v1r promotion: code, v1r calibration table and test fixed before the validate runs"
git push origin feat/v1r v1r-preregistered
```

---

### Task 15: The pre-registered runs and the verdict

On the server checkout of the tag (`nice -n 15`, at most 12 jobs, a quiet window with the research loop paused if it shares the box). Both runs log to `results/experiments.csv` with family `v1r-promotion-2026-11`.

- [x] **Step 1: Stage 1, components.**

```bash
target/release/fplopt-rs models eval --models v1,v1r --seasons 2023-2024 --family v1r-promotion-2026-11 --out results/v1r-stage1
```

Read `metrics.json`. Pass (spec): pooled MSE per player-GW lower for `v1r` at h0 **and** h1–5, one-sided Diebold–Mariano p < 0.10 at h0; minutes log loss and P(start) calibration by band not worse, per season (computed from `predictions.parquet` at h0 per the spec; `models eval` reports components per split only). Write the numbers and the pass/fail under *Results* below.

- [x] **Step 2: Stage 2, decisions** (only if stage 1 passed; spec: otherwise stop, no change):

```bash
target/release/fplopt-rs backtest compare \
  --a optimizer:v1r:horizon=6,decay=0.75,max_hits=1,hit_margin=2 \
  --b optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2 \
  --seasons 2023-2024 --continuation own \
  --starts template@1,random:5@1,random:3@20 \
  --family v1r-promotion-2026-11 --out results/v1r-stage2
```

(If `backtest compare` takes starts by another flag name, use it; the starts are fixed by the spec.) Pass: per-decision mean ≥ 0 realized **and** ≥ 0 xG-scored. Then run the same with `--seasons 2016-2022 --family v1r-promotion-2026-11-develop --out results/v1r-stage2-develop` (develop: in-sample for the loop, information only, not part of the verdict).

- [ ] **Step 3: Verdict.** Per the spec: both pass → adopt (Task 16). Stage 1 fails → no change; append the result to the loop's ledger (the maintainer's job: tell the user; don't write to the research repo). Stage 1 passes, stage 2 fails → no change; find which keep hurt planning before any retry (a retry is N=2). Write the verdict under *Results*, add a line to `docs/PLAN.md` §12 (decision log), commit.

---

### Task 16: Go live (only on an adopt verdict)

- [ ] **Step 1: Swap names** so `v1` is the live model's name everywhere (spec: `v1r` becomes `v1`, the old model stays as `v1-2026-10`):
  - `V1Params::default()` takes `v1r()`'s settings with `model: "v1"`; the old settings move to `V1Params::v1_2026_10()` with `model: "v1-2026-10"`; `v1r` stays as an alias of `v1` (`for_model("v1r")` returns `default()`), so logged specs still parse.
  - Calibration: `calibration_table.json` ← the v1r table (re-keyed to `v1`'s fingerprint); the old file → `calibration_table_v1-2026-10.json`.
  - `MODELS`, `LEAKAGE_BUILDERS`, `is_v1_family`, eval/bench lists: add `v1-2026-10`.
  - Goldens: this is the one place they are regenerated (`uv run python tests/test_rust_golden.py` won't apply once Python is retired; regenerate the Rust-side goldens with whatever batch 8 left as the generator) — review the diff: only v1-model outputs may change.
- [ ] **Step 2:** `cargo test --workspace`, clippy, fmt. PR to `main` (`🤖 Generated with [Claude Code](https://claude.com/claude-code)` at the end of the body), user review, merge.
- [ ] **Step 3: Deploy between two GWs** (after one deadline's alerts, before the next 24 h alert): `git pull && cargo build --release --locked` in `~/fpl-optimizer`, restart `fplopt-live.timer`'s next run picks it up (the context fingerprint changes with the code, so `live tick` rebuilds the context). Watch the first `live tick` log and the first alert, and record the context rebuild's time under *Pins* (Task 13's server timing; must be ≤ 10 min). Rollback: revert the merge commit, rebuild, the next tick rebuilds the context with the old model.
- [ ] **Step 4:** Tell the user the research loop's baseline moved (its harness compares against `v1`): re-pinning the harness is the maintainer's job.
- [ ] **Step 5: #39.** After two GWs live, check the `a100_back` cell (players flagged available who missed their last match: predicted vs realized start rate) in the weekly monitor; if within the band, close issue #39 with the numbers (`gh issue close 39 --repo samintisar/fpl-optimizer --comment "..."`), only with the user's OK.

---

## Pins

Filled in by Task 0 and later tasks.

- **Start ahead of the gates (user, 2026-11-02):** the user said "start" while the switch's gates were still open (batch 7 v1 backtests and chip study, shadow week). Tasks 0–10 run now on `feat/v1r`; nothing touches the server or `rust-migration` itself. Merge `rust-migration` into `feat/v1r` whenever it moves.
- Research commit: **dcacbcb** (`research`, 2026-11-01 run; local-only branch in `C:/Users/samin/Documents/GitHub/fpl-autoresearch`). All nine keeps are ancestors, and no kept `try:` commit after them touches models/features.
- Breaking synthetic tests at the pin (`pytest` on the model and history tests: 4 failed, 126 passed):
  - `test_models_availability.py::test_flags_move_p_start_and_teams_stay_normalized` (horizon_shrink): a small synthetic club's P(start) sums to 11.67 at a far horizon fixture, above the test's 11.5 bound. The shrink pulls toward 0.5 and the bounded normalization can't bring a thin pool back to 11. Real clubs sum to exactly 11 (ledger: `diag-shrink-sums.txt`). **Verdict: fixture artifact, keep stays.** The Rust v1 test is unchanged; any v1r version of it needs a realistic pool size.
  - `test_models_shares.py::test_planted_shares_are_recovered` (position_par): the planted DEF has npxG but no goals, so the position par pulls its share down (the same artifact as the ledger's finishing-12). **Verdict: fixture artifact, keep stays.** A v1r version would plant goals that match npxG.
  - `test_models_assemble.py::test_v1_is_deterministic_and_registered` and `::test_the_calibration_table_matches_the_default_settings`: research's calibration table was fitted for the old params, so calibrated v1 refuses it. Not a keep problem: Task 12 fits v1r's own table.
- **Parity gate (Task 11, run locally 2026-11-02 on `data/` built 2026-10-29; Python = research pin dcacbcb via `--python-src`, Rust = `v1r`, both `--no-calibrate`): PASS.**
  - Gates 1–2, 47 deadlines (2025/26 GW1 → 2026/27 GW10): every fitted value and xP frame within tolerance (94/94). Python 1,308 s, Rust 873 s. `results/parity/v1r/`.
  - Gate 3 (plan #1, template): the first run's "47/47" was false. Every solve errored and two identical errors counted as a match: the plan subprocess got no `FPLOPT_DATA_DIR`, and `optimize plan` refuses the 2025/26 holdout. `dev/models_parity.py` is fixed: it passes the data dir, leaves out holdout deadlines and says so, and treats a missing plan #1 as a difference. **This bug predates v1r: batch 6's "46/47" plan check most likely only solved its 2026/27 deadlines.**
  - Re-run on 2026/27 (9 deadlines, `results/parity/v1r-plans/`): plan #1 transfers, captains and objective are identical at 9/9. At GW5 and GW6 only the printed "gain vs roll" differs. Both roll solves stop at the 0.5% MIP gap on different near-equal plans (objective 241.20 vs 240.64, 264.62 vs 264.69): batch 6's tie rule, as at its 2026/27 GW8.
- **Calibration reproduction (Task 12, run locally 2026-11-02 on the same `data/`): PASS, exact.** `fplopt-rs models calibrate walk --model v1 --seasons 2016-2024 --jobs 4` (~5 min) then `fit --groups 1+ --compare crates/fplopt-models/calibration_table.json`: all 973 breakpoints of the 9 entries (2017–2025) equal, max |diff| 0, no rounding allowance used. The full store vs Python's 2016..S store changes nothing (only `team_rating` has pre-2016 rows, never the newest row a deadline reads). Logged (`rust parity`, n_variants 17).
- **v1r's table:** the same walk with `--model v1r` (~6 min), `fit --groups 1+ --write` → `crates/fplopt-models/calibration_table_v1r.json` (p_start only, horizons 1+ pooled, from 2016; 75–136 breakpoints per entry). v1r now inherits v1's `calibrate: true`; `calibration_for("v1r")` refuses a table whose params aren't v1r's fingerprint.
- **Leakage (Task 13, local 2026-11-03): PASS.** `fplopt-rs check leakage --deadlines 12`: 28 deadlines (2016/17 GW1 → 2026/27 GW10) × 19 frames (features, history, `model:*` incl. `model:v1r`, probes) × 2 variants, all ok; `team_rating` is among the corrupted tables. 1,850 s.
- **v1 unchanged (Task 13): PASS.** `dev/models_parity.py --models v1 --seasons 2025,2026 --plans` (Python = this branch's `src`): every fitted value and xP frame within tolerance on 47 deadlines (94/94). Plan #1 with the fixed check on the 9 solvable 2026/27 deadlines: 7/9 identical. At GW8 only the roll plan differs, and at GW3 plan #1 moves one transfer a week (same moves, objective 247.53 vs 247.39). Both solves stop at the 0.5% gap: batch 6's tie rule. **This settles the batch 6 question:** its plan gate really solved only its 9 2026/27 deadlines, and on those it holds. `results/parity/v1-nonregression/`. Goldens unchanged (`cargo test --workspace`).
- **`live prepare` timing (Task 13, local proxy):** the xP fit and frame at the live deadline (`models dump` at 2026/27 GW10) take 13 s for v1 and 21 s for v1r on the dev machine, so v1r adds seconds to a prepare dominated by `build all` (~1.5 min) and the context. The server's timing is measured at go-live (Task 16 step 3, the first `live prepare`); the ≤ 10 min bar has a wide margin.

## Results

Filled in by Task 15.

**Stage 1 (components): PASS.** 2026-11-03, locally at the tag `v1r-preregistered` (`models eval --models v1,v1r --seasons 2023-2024 --family v1r-promotion-2026-11 --out results/v1r-stage1`, 293 s; 76 deadlines).

| criterion | v1 | v1r | result |
|---|---|---|---|
| MSE per player-GW, h0, pooled | 3.465 | 3.446 | lower |
| MSE per player-GW, h1–5, pooled | 3.969 | 3.960 | lower |
| Diebold–Mariano at h0, one-sided p (v1r better) | | 0.000 | < 0.10 |
| minutes log loss, h0, 2023/24 | 0.43147 | 0.42546 | not worse |
| minutes log loss, h0, 2024/25 | 0.45795 | 0.45041 | not worse |
| P(start) reliability (10 bins), h0, 2023/24 | 0.001662 | 0.000584 | not worse |
| P(start) reliability (10 bins), h0, 2024/25 | 0.001469 | 0.000326 | not worse |

Also h1–5 DM p 0.002 and candidate MSE at h0 10.183 → 10.127. The per-season numbers come from `predictions.parquet` (a scratch script whose pooled values matched `metrics.json`'s components exactly). v1r's P(start) bins no longer overpredict in the 0.2–0.5 range (2024/25 bin 0.3–0.4: v1 predicts 0.350 vs 0.249 observed, v1r 0.350 vs 0.330).

Not a criterion, recorded as a warning for stage 2: XI and captain regret on the template + 3 random squads are worse for v1r (XI 4.59 → 5.04, captain 4.02 → 4.48; one-sided DM p 0.035 that v1 is better).

**Stage 2 (decisions): PASS.** 2026-11-03, locally, same binary (built at the tag; only the stage 1 write-up committed since). The pre-registered command with `--family v1r-promotion-2026-11 --out results/v1r-stage2`, 1,247 s.

| v1r − v1, validate | mean | 80% CI | p (one-sided) | bar |
|---|---|---|---|---|
| **per-decision, 4-GW windows, realized** | **+3.27** | [−0.51, +7.07] | 0.139 | ≥ 0: pass |
| **per-decision, 4-GW windows, xG-scored** | **+1.26** | [−1.19, +3.82] | 0.259 | ≥ 0: pass |
| full run, per GW, realized | +4.56 | [+2.83, +6.33] | 0.000 | (info) |
| full run, per GW, xG-scored | +3.21 | [+1.98, +4.50] | 0.000 | (info) |

- Full-run season totals, v1r − v1: +121.6 (2023/24), +225.2 (2024/25); template from GW1 2,510 vs 2,350 and 2,387 vs 2,093. The one arm where v1r is behind: the three GW20 random starts in 2024/25 (mean 1,124 vs 1,171). Full-run window t-test on 8-GW cells (the chip study's statistic): +4.63/GW, p 0.001.
- 20 decision windows; plans differ at 83% / 99% of decisions. The stage 1 regret warning didn't carry over: XI regret is higher for v1r in the backtest (11.03 vs 9.62 from GW1) but the squads it picks score more.
- Transfer-gain slope (realized on predicted) 0.75 for v1r vs 1.32 for v1: v1r's predicted gains are less flat but somewhat inflated (intercept +5.3); no general optimizer's-curse correction is in scope here (spec, *Not in scope*).
- The log's `n_variants` for the family is 2 because the stage 1 `models eval` row shares the family; the spec's N=1 is one candidate, and the bar is the raw mean (deflated values are printed only).

**Verdict: adopt** (both stages pass). Task 16 next; the develop run below is information only.
