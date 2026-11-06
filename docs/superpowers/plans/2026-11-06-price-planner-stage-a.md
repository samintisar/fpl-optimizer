# Price-Aware Planner, Stage A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The MILP planner accepts per-GW expected price paths (with FPL's selling rule for owned players and for players bought within the plan) and an end-of-horizon money value. The backtester can feed it the real future prices ("oracle"). The Gate A runs decide whether a price forecast is worth building.

**Architecture:**
- **Planner:** `Player` gains an optional price path and its purchase price.
- **Model (`model.rs`):**
  - prices money flows per GW from the path;
  - adds "cohort" sale variables (bought in GW t, sold in GW s) only for players whose path moves;
  - adds λ·M to the objective, where M is the money at the horizon's end.
- **Default equivalence:** with no paths and `price_value_k = 0`, no variable, row or coefficient changes, so every golden and parity check holds.
- **Backtester:**
  - a `FuturePrices` reader (`player_gw`, all deadlines of a season) fills paths when the spec says `prices=oracle`;
  - spec keys `prices` and `price_k`.
- **Gate A:** a `dev/` script pairs the three develop runs.

**Tech Stack:** Rust (`crates/fplopt-optimize`, `crates/fplopt-backtest`, `crates/fplopt-cli`), HiGHS through the existing LP builder, Python + pandas for the gate script, pytest.

**Spec:** `docs/superpowers/specs/2026-11-06-price-planner-design.md`, Stage A. Branch `feat/price-sizing`, worktree `C:\Users\samin\Documents\GitHub\fpl-optimizer\.claude\worktrees\v1r`.

**Facts (from reading the code on 2026-11-06):**
- **`Player` (`crates/fplopt-optimize/src/problem.rs:77-87`).**
  - Fields: `{ player_key, element_type, team_key, price: i64, sell_price: i64, owned, buyable, xp: Vec<f64> }`.
  - It has exactly one struct literal, at `problem.rs:218-227`, inside `PlanInput::from_context` (lines 129-249).
  - `PlanInput` (lines 99-113) has `doubles`, a field the policy fills after `from_context` (`crates/fplopt-backtest/src/policies.rs:486-487`). Price paths follow that precedent.
- **`model.rs` variables (`variables`, lines 273-321).**
  - `later_from = if p.owned {2} else {1}`.
  - In a normal GW the model creates `buy` (if buyable), `sell_first` (if owned) and `sell_later` (if buyable and t ≥ later_from). Free Hit GWs get `fh_*` copies. `squad[(k,t)]` is always created.
  - `sells(k,t)` (lines 323-327) is `sell_first + sell_later`.
  - **Column order is creation order and is pinned for parity.**
- **`model.rs` constraints and money.**
  - Squad balance and linking: lines 370-428.
  - The money `flow` (lines 531-553) puts `sell_price` on `sell_first`, `price` on `sell_later` and `-price` on `buy`, then drops zero coefficients.
  - Bank balance: lines 558-570. `prev` is `problem.bank` at t = 0, otherwise `bank[t-1]`. The Free Hit bank is `fh_bank[t] = prev + flow(t,true)`.
- **`model.rs` objective (`objective()`, lines 707-775).**
  - Terms per GW are multiplied by `d = decay^horizon`. `itb_value` is at line 740.
  - Terms merge in first-seen order and zero terms are dropped. The function returns `(terms, constant)`.
  - The chip terminal value is added as a constant in `solve_plan` (lines 865-867), and `Plan.terminal_value` holds it.
- **`model.rs` readout (`read_plan`, lines 938-1031).**
  - Outs come from `sells` / `fh_sells`, and the bank is `py_round(x[bank[t]])`.
  - The per-GW objective is recomputed at lines 1006-1010, and `Plan.objective` is their sum (line 899). `Plan::total_objective` = objective + terminal_value (`plans.rs:80`).
- **Params.**
  - `crates/fplopt-optimize/src/params.rs`: fields at lines 25-50, defaults at lines 52-77, validation list at lines 93-98.
  - `crates/fplopt-optimize/src/io.rs` `params_with` (lines 57-145) rejects unknown keys.
- **Backtest specs (`crates/fplopt-backtest/src/specs.rs`).**
  - Allowed keys: `params_of` (lines 80-123).
  - `make_policy_spec` (lines 394-459) rejects unknown keys. Mapping: `optimizer_params` (lines 276-293).
  - `chip_key_default` (lines 376-383) drops keys given at their default.
  - The label comes from `Policy::name` (`policies.rs:371-391`) and `optimizer_extras` (lines 635-706). It is pinned by `specs.rs:505-517`.
- **Backtest context.** `DecisionContext` (`policies.rs:45-52`) is `{ view, state, rules, pool, xp, frame }` and has no `DataStore`. Contexts are built in three places:
  - `simulator.rs:261` (`decide_step`, which has `store` and `caches`);
  - `money_value.rs:154` (`solve_at`, which has `store` and `caches`);
  - `probes.rs:87`.
- **`player_gw` readers.** `fplopt-features/src/baseline.rs:135-196` reads `view.table("player_gw", Some(&[...,"value"]))`. `value` at (season, gw) is the deadline price: equal to the last pre-deadline snapshot on 47,875 checked rows.
- **Optimizer tests** are in `crates/fplopt-optimize/tests/` (helpers in `tests/common/mod.rs`: `make_pool`, `make_state`/`StateOpts`, `xp_frame`, `problem`, `solve`, `player`, `transfers_in/out`, `replay`). The model for money tests is `churn_instance` / `first_sale_at_selling_price_later_sales_at_buy_price` (`tests/model.rs:332-387`) with `plan_for` and `exact()`.
- **What pins the MILP:**
  - the bot golden (`crates/fplopt-bot/tests/golden.rs`);
  - `dev/optimize_parity.py` (through `optimize solve --case`);
  - `dev/backtest_parity.py`;
  - the `model.rs` header (lines 1-7).

**Rules for every task:**
- Cargo always with `-j 2` (the machine runs out of memory otherwise).
- Never touch season 2025 or `raw/`.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- No push and no server work without the controller's go-ahead.
- **With defaults, no new LP column, row or coefficient may appear.** Every new variable and term is guarded by "a path moves" or "k > 0".

---

### Task 1: Price paths on `Player` and `PlanInput`

**Files:**
- Modify: `crates/fplopt-optimize/src/problem.rs` (Player, its literal, new methods, module doc)
- Test: `crates/fplopt-optimize/tests/model.rs`

- [ ] **Step 1: Write the failing test** (append to `tests/model.rs`'s money section, reusing `churn_instance`):

```rust
#[test]
fn price_path_prices_buys_and_first_sales() {
    let rules = rules();
    let (state, pool, xp, a, c) = churn_instance();
    let params =
        OptimizerParams { horizon: 3, ft_value: BTreeMap::new(), itb_value: 0.0, ..exact() };
    let mut problem = problem(&state, &pool, &xp, &params, &rules);
    // no path: today's prices
    assert_eq!(player(&problem, a).buy_price(2), 60.0);
    assert_eq!(player(&problem, a).first_sale_value(2), 55.0);
    problem.set_price_paths(|p| match p.player_key {
        k if k == a => Some(vec![60.0, 62.0, 66.0]),
        k if k == c => Some(vec![50.0, 52.0, 54.0]),
        _ => None,
    });
    assert_eq!(player(&problem, a).buy_price(1), 62.0);
    // A was bought at 50: at 66 he sells for 50 + (66 - 50) / 2 = 58 (not floored)
    assert_eq!(player(&problem, a).first_sale_value(2), 58.0);
    assert_eq!(player(&problem, a).purchase_price, Some(50));
    assert_eq!(player(&problem, c).purchase_price, None);
    assert!(player(&problem, c).path_moves());
    assert!(!player(&problem, key(1, 1, 0)).path_moves());
}
```

- [ ] **Step 2: Run it and see it fail** — `cargo test -j 2 -p fplopt-optimize --test model price_path`; expected: compile errors (`set_price_paths`, `buy_price`, ... missing).

- [ ] **Step 3: Implement** in `problem.rs`:

```rust
/// A candidate: `price` the buy price, `sell_price` what his first sale earns; `xp[t]` his
/// xP in horizon GW t. `price_path[t]`, when set, is his expected price at horizon GW t's
/// deadline (tenths, fractional; `[0]` is today's price) and replaces `price`/`sell_price`
/// in the money flows; `purchase_price` is an owned player's purchase price.
#[derive(Clone, Debug, PartialEq)]
pub struct Player {
    pub player_key: i64,
    pub element_type: i64,
    pub team_key: i64,
    pub price: i64,
    pub sell_price: i64,
    pub owned: bool,
    pub buyable: bool,
    pub xp: Vec<f64>,
    pub purchase_price: Option<i64>,
    pub price_path: Option<Vec<f64>>,
}

/// A path "moves" when it spans at least this many tenths over the horizon.
pub const PATH_MOVES: f64 = 0.05;

impl Player {
    /// What buying him in horizon GW t costs.
    pub fn buy_price(&self, t: usize) -> f64 {
        self.price_path.as_ref().map_or(self.price as f64, |p| p[t])
    }

    /// What an owned player's first sale in GW t earns: FPL's selling rule on the path's
    /// price, not floored (expected prices are fractional); `sell_price` without a path.
    pub fn first_sale_value(&self, t: usize) -> f64 {
        match (&self.price_path, self.purchase_price) {
            (Some(p), Some(purchase)) => selling_value(purchase as f64, p[t]),
            _ => self.sell_price as f64,
        }
    }

    pub fn path_moves(&self) -> bool {
        self.price_path.as_ref().is_some_and(|p| {
            let (lo, hi) = p.iter().fold((f64::MAX, f64::MIN), |(l, h), &x| (l.min(x), h.max(x)));
            hi - lo >= PATH_MOVES
        })
    }
}

/// FPL's selling rule without the floor: bought at `bought`, now `now`.
pub fn selling_value(bought: f64, now: f64) -> f64 {
    if now > bought { bought + 0.5 * (now - bought) } else { now }
}

impl PlanInput {
    /// Set each candidate's price path (`None` keeps today's prices). Paths must have one
    /// entry per horizon GW.
    pub fn set_price_paths(&mut self, mut path: impl FnMut(&Player) -> Option<Vec<f64>>) {
        let n = self.gws.len();
        for p in &mut self.players {
            p.price_path = path(p);
            if let Some(v) = &p.price_path {
                assert_eq!(v.len(), n, "price path of {} has {} GWs, want {n}", p.player_key, v.len());
            }
        }
    }
}
```

In the single `Player` literal (`problem.rs:218-227`), add `purchase_price: holding.map(|h| h.purchase_price), price_path: None,`. Update the module doc (lines 3-4) to say prices may follow a per-GW path (`set_price_paths`).

- [ ] **Step 4: Run** `cargo test -j 2 -p fplopt-optimize` — all pass (the new test included).

- [ ] **Step 5: Commit** — `git commit -am "feat(optimize): price paths on planner candidates"` (plus the Co-Authored-By line).

---

### Task 2: Money flows follow the paths, with cohort sales

**Files:**
- Modify: `crates/fplopt-optimize/src/model.rs` (variables, `sells`, linking constraints, `flow`, readout)
- Test: `crates/fplopt-optimize/tests/model.rs`

**Design (implement exactly this):**

- **Cohort variables.** For a candidate with `path_moves()` and `buyable`, in a **normal** (non–Free Hit) GW s ≥ 1, replace the single `sell_later[(k,s)]` with cohort binaries `cohort[(k, tb, s)]` for every normal GW `tb < s` that has a `buy[(k,tb)]` variable.
  - Owned players keep `later_from = 2`: a cohort needs a buy, and an owned player can only be re-bought after his first sale.
  - Store the cohorts in a new `BTreeMap<(i64, usize, usize), usize>` field, and create them where `sell_later` would have been created, in the same loop position, so column order is unchanged for non-moving players.
- **`sells(k, s)`** also includes `Σ_tb cohort[(k,tb,s)]`.
- **New rows,** only for moving players:
  - `Σ_{s>tb} cohort[(k,tb,s)] ≤ buy[(k,tb)]` for every tb with a buy (a cohort is sold at most once, and only if bought);
  - the existing `sell_later ≤ first_sales_before` rule applies to each cohort of an owned player.
- **`flow(t, fh=false)` coefficients:**
  - `sell_first[(k,t)]`: `p.first_sale_value(t)`;
  - `sell_later[(k,t)]`: `p.buy_price(t)` (non-moving players; equals `price` with no path);
  - `cohort[(k,tb,t)]`: `selling_value(p.buy_price(tb), p.buy_price(t))`;
  - `buy[(k,t)]`: `-p.buy_price(t)`.

  Without paths these are the same numbers as today (`sell_price`, `price`, `-price`), so the coefficients are unchanged.
- **Free Hit (`fh = true`):** `fh_sell_first` gets `p.first_sale_value(t)`, `fh_sell_later` gets `p.buy_price(t)`, and `fh_buy` gets `-p.buy_price(t)`. No cohorts in Free Hit GWs.
- **Readout:** `read_plan`'s outs include cohort sales (through `sells`). The bank stays `py_round(x[bank[t]])`, which is fractional in the LP and rounded for the report.

- [ ] **Step 1: Write the failing test** (append):

```rust
#[test]
fn moving_prices_change_the_money_flows() {
    // churn_instance with rising prices: the same churn A -> C -> A -> C, but the money
    // follows the paths and FPL's selling rule per cohort.
    let rules = rules();
    let (state, pool, xp, a, c) = churn_instance();
    let state = SquadState { bank: 15, ..state };
    let params =
        OptimizerParams { horizon: 3, ft_value: BTreeMap::new(), itb_value: 0.0, ..exact() };
    let mut problem = problem(&state, &pool, &xp, &params, &rules);
    problem.set_price_paths(|p| match p.player_key {
        k if k == a => Some(vec![60.0, 62.0, 66.0]),
        k if k == c => Some(vec![50.0, 52.0, 54.0]),
        _ => None,
    });
    let plan = solve(&problem, &params);
    let moves: Vec<(Vec<i64>, Vec<i64>)> =
        plan.gws.iter().map(|g| (transfers_out(g), transfers_in(g))).collect();
    assert_eq!(moves, vec![(vec![a], vec![c]), (vec![c], vec![a]), (vec![a], vec![c])]);
    // GW0: 15 + 55 (A's first sale at 60) - 50 (C) = 20
    // GW1: + 51 (C bought at 50, sold at 52: 50 + 1) - 62 (A) = 9
    // GW2: + 64 (A bought at 62, sold at 66: 62 + 2) - 54 (C) = 19
    assert_eq!(plan.gws.iter().map(|g| g.bank).collect::<Vec<_>>(), vec![20, 9, 19]);
}

#[test]
fn without_paths_the_model_is_unchanged() {
    // same columns, rows and coefficients as before the feature: compare the HiGHS model
    // of churn_instance with and without an explicit `set_price_paths(|_| None)`.
    let rules = rules();
    let (state, pool, xp, _, _) = churn_instance();
    let params =
        OptimizerParams { horizon: 3, ft_value: BTreeMap::new(), itb_value: 0.0, ..exact() };
    let plain = problem(&state, &pool, &xp, &params, &rules);
    let mut with_none = plain.clone();
    with_none.set_price_paths(|_| None);
    assert_eq!(model_fingerprint(&plain, &params), model_fingerprint(&with_none, &params));
}
```

`SquadState` may not support struct-update syntax if it has private fields; then build the state with `make_state(..., StateOpts { bank: Some(15), ... })` as `churn_instance` does (copy its body into a `churn_instance_with_bank(bank)` helper and have `churn_instance` call it with 5). `model_fingerprint`: if `model.rs` already exposes the built LP (for example a test hook or `to_highs`), hash its column count, row count and coefficient list. If nothing is exposed, add a `#[doc(hidden)] pub fn model_size(problem, params) -> (usize, usize, usize)` returning (columns, rows, nonzeros) for the no-chip scenario and compare those. Also check that the existing `first_sale_at_selling_price_later_sales_at_buy_price` still passes unchanged.

- [ ] **Step 2: Run them and see them fail** — `cargo test -j 2 -p fplopt-optimize --test model moving_prices without_paths`.

- [ ] **Step 3: Implement** the design above in `model.rs`. Keep the existing loop structure and add the cohort branch inside it:

```rust
// in variables(), where sell_later is created for a normal GW t:
if p.buyable && t >= later_from {
    if p.path_moves() {
        for tb in 0..t {
            if self.buy.contains_key(&(k, tb)) {
                self.cohort.insert((k, tb, t), lp.binary());
            }
        }
    } else {
        self.sell_later.insert((k, t), lp.binary());
    }
}
```

```rust
// sells(): add the cohorts sold in GW t
fn cohort_sales(&self, k: i64, t: usize) -> impl Iterator<Item = usize> + '_ {
    self.cohort.range((k, 0, t)..=(k, usize::MAX, t)).filter(|((_, _, s), _)| *s == t).map(|(_, &v)| v)
}
```

(The range scan over a `(k, tb, s)` key needs a filter on `s`; alternatively also keep a `HashMap<(i64, usize), Vec<usize>>` of cohorts by sale GW, built at creation, and use it in `sells` and `flow`.)

```rust
// squad_constraints(): for each moving player and each tb with a buy
let sold: Vec<usize> = (tb + 1..self.t_len).filter_map(|s| self.cohort.get(&(k, tb, s)).copied()).collect();
if !sold.is_empty() {
    self.lp.le(Expr::sum(sold), &Expr::var(self.buy[&(k, tb)]));
}
// owned players: every cohort sale in GW s also obeys `≤ first_sales_before(k, s)`
```

```rust
// flow(): replace the three coefficient lines
if let Some(&v) = sell_first.get(&(k, t)) { e.add(v, p.first_sale_value(t)); }
if let Some(&v) = sell_later.get(&(k, t)) { e.add(v, p.buy_price(t)); }
if !fh { for (tb, v) in self.cohorts_sold_at(k, t) { e.add(v, selling_value(p.buy_price(tb), p.buy_price(t))); } }
if let Some(&v) = buy.get(&(k, t)) { e.add(v, -p.buy_price(t)); }
```

Keep the order of `e.add` calls as today for the three existing variables, so the coefficient order stays pinned.

- [ ] **Step 4: Run** `cargo test -j 2 -p fplopt-optimize`, then `cargo test -j 2 --workspace` (includes the bot golden), `cargo clippy -j 2 --workspace --all-targets -- -D warnings`, `cargo fmt --all`. All pass, with no golden change.

- [ ] **Step 5: Commit** — `"feat(optimize): money flows follow price paths, cohort selling values"`.

---

### Task 3: End-of-horizon money value (`price_value_k`)

**Files:**
- Modify: `crates/fplopt-optimize/src/params.rs`, `crates/fplopt-optimize/src/io.rs`, `crates/fplopt-optimize/src/model.rs`, `crates/fplopt-optimize/src/plans.rs`
- Test: `crates/fplopt-optimize/tests/model.rs`

**Design:**
- **New param** `OptimizerParams.price_value_k: f64`, default 0.0. Validate it finite and ≥ 0 in the list at `params.rs:93-98`. JSON key `"price_value_k"` in `io.rs` `params_with`.
- **Constants** in `params.rs`: `pub const MONEY_V: f64 = 0.395;` and `pub const MONEY_C: f64 = 0.93;`, with a doc comment citing `docs/research/2026-11-price-sizing.md`.
- **The rate:** `λ = price_value_k · MONEY_V · MONEY_C · max(0, 38 − gw_index_last) / 10` points per tenth. `gw_index_last = problem.gws.last().gw_index`.
- **Only when λ > 0,** add to the objective terms (not decayed), with T = t_len − 1:
  - `λ · bank[T]` (the persistent bank, also when T is a Free Hit GW);
  - for each owned player k: `λ · first_sale_value(T)` on the constant (held all along), plus `−λ · first_sale_value(T)` on each `sell_first[(k,t)]` (selling him removes his end value). This adds owned × (1 − Σ sell_first) × value;
  - for each moving, buyable player and each tb with a buy: `λ · selling_value(buy_price(tb), buy_price(T))` on `buy[(k,tb)]`, minus the same on each of that cohort's `cohort[(k,tb,s)]`;
  - for each non-moving player bought within the plan: `λ · buy_price(T)` on each `buy[(k,t)]` and `−λ · buy_price(T)` on each `sell_later[(k,t)]`. Selling an owned player and buying him back is covered: the re-buy is a buy.

  Append these terms after all existing terms, so the merge order of existing terms is unchanged.
- **Readout:** `Plan` gets `pub money_value: f64`, which is λ·M evaluated at the solution with the same formula, and 0 when λ = 0. `Plan::total_objective` adds it. `read_plan`'s per-GW objectives stay as they are: the money value is a horizon-end term like `terminal_value`, reported separately.
  - Where the solver's objective is compared with `Plan.objective` (search `objective` asserts or `bound` in `model.rs`/`search.rs`), include `money_value` the way `terminal_value` is included.
  - JSON output (`io.rs` `plan_json`): add `"money_value"` only when it is non-zero, so existing JSON stays byte-identical.

- [ ] **Step 1: Write the failing tests** (append):

```rust
/// One owned MID X (xP 0) to replace; candidates C (rising) and D (flat, 0.01 more xP).
fn riser_instance() -> (SquadState, Vec<PoolPlayer>, XpFrame, i64, i64) {
    let clubs = 1..=5;
    let mut rows = Vec::new();
    for c in clubs.clone() {
        for et in 1..=4 {
            rows.push((key(c, et, 0), et, c, 45));
        }
    }
    rows.extend(clubs.clone().map(|c| (key(c, 3, 1), 3, c, 45)));
    rows.extend(clubs.clone().map(|c| (key(c, 2, 1), 2, c, 45)));
    rows.extend([1, 2].map(|c| (key(c, 4, 1), 4, c, 45)));
    rows.push((key(1, 1, 1), 1, 1, 45));
    let (c_key, d_key) = (key(6, 3, 0), key(7, 3, 0));
    rows.extend([(c_key, 3, 6, 50), (d_key, 3, 7, 50)]);
    let pool = make_pool(rows);
    let mut owned = vec![key(1, 1, 0), key(2, 1, 0)];
    owned.extend(clubs.clone().map(|c| key(c, 2, 0)));
    owned.extend([1, 2, 3, 4, 5].map(|c| key(c, 3, 0)));
    owned.extend([1, 2, 3].map(|c| key(c, 4, 0)));
    let mut values: BTreeMap<i64, Vec<f64>> =
        pool.iter().map(|p| (p.player_key, vec![0.5; 3])).collect();
    for k in &owned {
        values.insert(*k, vec![1.0; 3]);
    }
    let x = key(5, 3, 0); // the owned MID worth replacing
    values.insert(x, vec![0.0; 3]);
    values.insert(c_key, vec![5.0, 5.0, 5.0]);
    values.insert(d_key, vec![5.0, 5.0, 5.01]);
    let xp = xp_frame(&values, 5);
    let state = make_state(&pool, &owned, StateOpts { ft: 1, bank: Some(10), ..Default::default() });
    (state, pool, xp, c_key, d_key)
}

fn riser_plan(k: f64) -> (Vec<i64>, f64) {
    let rules = rules();
    let (state, pool, xp, c, _) = riser_instance();
    let params = OptimizerParams {
        horizon: 3,
        ft_value: BTreeMap::new(),
        itb_value: 0.0,
        price_value_k: k,
        ..exact()
    };
    let mut problem = problem(&state, &pool, &xp, &params, &rules);
    problem.set_price_paths(|p| (p.player_key == c).then(|| vec![50.0, 52.0, 54.0]));
    let plan = solve(&problem, &params);
    (transfers_in(&plan.gws[0]), plan.money_value)
}

#[test]
fn money_value_prefers_the_riser_only_when_k_is_positive() {
    let (c, d) = (key(6, 3, 0), key(7, 3, 0));
    let (ins, money) = riser_plan(0.0);
    assert_eq!(ins, vec![d]);
    assert_eq!(money, 0.0);
    let (ins, money) = riser_plan(1.0);
    assert_eq!(ins, vec![c]);
    // λ = 0.395 · 0.93 · (38 − 7) / 10; C held at the end is worth 50 + (54 − 50)/2 = 52
    let lambda = 0.395 * 0.93 * 31.0 / 10.0;
    assert!(money > lambda * 52.0); // plus the bank and the rest of the squad
}

#[test]
fn money_value_is_zero_late_in_the_season() {
    // with the horizon ending at gw_index ≥ 38, λ = 0: build riser_instance at gw_index 36
    // (xp_frame(&values, 36), StateOpts { gw_index: 36, .. }) and check money_value == 0
    // and the D choice, like k = 0.
}
```

(Write the second test's body in full along the lines of the comment. `StateOpts` has a `gw_index` field.)

- [ ] **Step 2: Run them and see them fail** (compile errors on `price_value_k` / `money_value`).

- [ ] **Step 3: Implement** the design. In `model.rs`, write a `fn money_terms(&self, lambda: f64) -> (Vec<(usize, f64)>, f64)`, called from `objective()` only when `lambda > 0.0`. Its terms are appended after the existing ones and its constant is added to `constant`. Write a twin `fn money_value(&self, x: &[f64], lambda: f64) -> f64` for the readout, using the same coefficients so the two can't diverge. A unit test asserts `money_value(x) == Σ terms·x + constant` on the riser solution.

- [ ] **Step 4: Run** `cargo test -j 2 --workspace`, clippy, fmt. All pass, goldens unchanged.

- [ ] **Step 5: Commit** — `"feat(optimize): end-of-horizon money value (price_value_k)"`.

---

### Task 4: Spec keys `prices` and `price_k` in the backtester

**Files:**
- Modify: `crates/fplopt-backtest/src/specs.rs`, `crates/fplopt-backtest/src/policies.rs`
- Test: the `specs.rs` tests module

**Design:**
- **`price_k`** is an optimizer key that maps to `price_value_k`. Its default of 0 is dropped (`chip_key_default`). The label is `pk={k}` in `optimizer_extras`, emitted only when ≠ 0.
- **`prices`** is an optimizer key with values `fixed` (default, dropped) or `oracle`. `forecast` is reserved and refused for now with "prices=forecast arrives with Stage B".
  - Store it on `Policy::Optimizer` as a new field `prices: PriceSource` (`enum PriceSource { Fixed, Oracle }`).
  - Label `prices=oracle` shows as `oracle` in `optimizer_extras`.
- **Accepted on both policies:** `optimizer` and `chipgains`/`chiprules` if they share `optimizer_keys`. Put both keys in `optimizer_keys`.

- [ ] **Step 1: Failing test:**

```rust
#[test]
fn price_keys_parse_and_label() {
    let spec = parse_policy_spec("optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2,prices=oracle,price_k=1").unwrap();
    let policy = spec.build().unwrap();
    assert!(policy.name().contains("oracle") && policy.name().contains("pk=1"));
    let plain = parse_policy_spec("optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2,prices=fixed,price_k=0").unwrap();
    assert_eq!(plain.build().unwrap().name(), parse_policy_spec("optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2").unwrap().build().unwrap().name());
    assert!(parse_policy_spec("optimizer:v1:prices=forecast").and_then(|s| s.build()).is_err());
    assert!(parse_policy_spec("optimizer:v1:prices=bogus").and_then(|s| s.build()).is_err());
}
```

Adapt the calls to the real API (`parse_policy_spec(...)?.build()` as in `specs.rs`'s tests, and the label check as at `specs.rs:505-517`).

- [ ] **Step 2: Run it and see it fail.**

- [ ] **Step 3: Implement.** Keep the existing label test (`specs.rs:505-517`) passing unchanged.

- [ ] **Step 4: Run** `cargo test -j 2 -p fplopt-backtest`, then the workspace tests, clippy and fmt.

- [ ] **Step 5: Commit** — `"feat(backtest): prices and price_k spec keys"`.

---

### Task 5: Oracle price paths in the backtester

**Files:**
- Create: `crates/fplopt-backtest/src/oracle.rs`
- Modify:
  - `crates/fplopt-backtest/src/lib.rs` (module)
  - `crates/fplopt-backtest/src/policies.rs` (`DecisionContext`, `optimizer_plans`)
  - `crates/fplopt-backtest/src/simulator.rs` (`decide_step`, `Caches`)
  - `crates/fplopt-backtest/src/money_value.rs` (`solve_at`)
  - `crates/fplopt-backtest/src/probes.rs` (context literal)
  - `crates/fplopt-cli/src/main.rs` (leakage note)

**Design:**
- **`FuturePrices`:** `pub struct FuturePrices { by_key: HashMap<(i64 /*season*/, i64 /*gw*/, i64 /*player_key*/), i64> }`.
  - `FuturePrices::load(store: &DataStore, season: i64) -> Result<Self>` reads `player_gw` (`season, gw, player_key, value`) for that season from a view that sees everything. Use `store.as_of(<an instant after the season's last deadline>)`: check `DataStore::as_of`'s argument type and use the latest representable instant, or the season schedule's last deadline + 30 days.
  - Doc comment: **reads the future by design; backtest-only foresight bound (spec Stage A).**
- **`path(&self, season, player: &Player, gws: &[HorizonGw]) -> Vec<f64>`:**
  - `[0]` is `player.price` (today's price, as the pool has it);
  - for t ≥ 1, `value` at `(season, gws[t].gw, key)` if present, else the previous entry (carry forward).
- **Cache:** `Caches` gets `future: RefCell<HashMap<i64, Rc<FuturePrices>>>` (or whatever the existing caches use), loaded per season on first use, only when a policy asks for oracle prices.
- **`DecisionContext`** gets `pub future: Option<&'a FuturePrices>`:
  - `decide_step` and `solve_at` set it when `decider`'s policy has `prices == Oracle` (add `Policy::wants_future_prices()`), otherwise `None`;
  - `probes.rs` sets `None`.
- **`optimizer_plans`:** after `from_context` and the `doubles` line, if `prices == Oracle`:

  ```rust
  let future = ctx.future.context("prices=oracle needs future prices (backtest only)")?;
  let season = ctx.state.season;
  let gws = problem.gws.clone();
  problem.set_price_paths(|p| Some(future.path(season, p, &gws)));
  ```
- **`check leakage`** (`crates/fplopt-cli/src/main.rs:385-416`): print one line before the run: `not checked (reads future prices by design): prices=oracle (backtest-only foresight bound)`. Put the text in a `pub const LEAKAGE_EXCLUDED: &[(&str, &str)]` in `fplopt-backtest/src/oracle.rs`.

- [ ] **Step 1: Failing unit test** in `oracle.rs`:

```rust
#[test]
fn path_starts_at_todays_price_and_carries_forward() {
    let fp = FuturePrices::from_rows([(2020, 6, 1, 52), (2020, 8, 1, 55)]);
    let gws: Vec<HorizonGw> = (5..=8).enumerate().map(|(h, gw)| HorizonGw { gw, gw_index: gw, horizon: h as i64 }).collect();
    let player = /* a Player with player_key 1, price 50 */;
    assert_eq!(fp.path(2020, &player, &gws), vec![50.0, 52.0, 52.0, 55.0]);
}
```

Add a `#[cfg(test)] fn from_rows(...)` constructor. Build the `Player` literal with every field (`purchase_price: None, price_path: None`, ...).

- [ ] **Step 2: Run it and see it fail.**

- [ ] **Step 3: Implement** the design.

- [ ] **Step 4: Smoke test:** with the main checkout's data, `cargo run -j 2 --release -p fplopt-cli -- backtest run --seasons 2021 --spec optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2,prices=oracle,price_k=1 --starts template@1 --jobs 2 --data-dir C:/Users/samin/Documents/GitHub/fpl-optimizer/data --out <scratch dir>`. Check `backtest run` accepts `--data-dir`, otherwise run from a directory whose `data` points there. Report the season total and the end `market_value` against the same command without the two keys.

- [ ] **Step 5: Run** the workspace tests, clippy and fmt. **Commit** — `"feat(backtest): oracle price paths (prices=oracle), backtest-only"`.

---

### Task 6: Gate A script

**Files:**
- Create: `dev/price_gate_a.py`, `tests/test_price_gate_a.py`

- [ ] **Step 1: Failing tests** (`tests/test_price_gate_a.py`):

```python
import importlib.util
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("price_gate_a", REPO / "dev" / "price_gate_a.py")
gate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate
spec.loader.exec_module(gate)


def _run(points, end_values):
    rows = []
    for (sid, pts), end in zip(points.items(), end_values, strict=True):
        for gw, p in enumerate(pts, start=1):
            rows.append({"season": 2020, "start_id": sid, "gw": gw, "gw_index": gw,
                         "net_points": p, "market_value": 1000 if gw < len(pts) else end})
    return pd.DataFrame(rows)


def test_unit_totals_and_end_value():
    df = _run({"template@1": [50, 60], "random0@1": [40, 40]}, [1010, 990])
    u = gate.units(df)
    assert u.loc[(2020, "template@1"), "points"] == 110
    assert u.loc[(2020, "random0@1"), "end_value"] == 990


def test_gate_a_pass_and_fail():
    l = _run({"template@1": [50, 50], "random0@1": [50, 50]}, [1000, 1000])
    o = _run({"template@1": [56, 56], "random0@1": [55, 55]}, [1025, 1022])
    r = gate.compare(gate.units(l), gate.units(o))
    assert r["n"] == 2 and r["points"]["mean"] == 11.0 and r["end_value"]["mean"] == 23.5
    assert gate.decide(r) is True
    o2 = _run({"template@1": [56, 56], "random0@1": [55, 55]}, [1010, 1010])
    assert gate.decide(gate.compare(gate.units(l), gate.units(o2))) is False
```

- [ ] **Step 2: Run them and see them fail** — `uv run pytest tests/test_price_gate_a.py -q`.

- [ ] **Step 3: Implement** `dev/price_gate_a.py`:

```python
"""Gate A of the price planner (spec docs/superpowers/specs/2026-11-06-price-planner-design.md):
O (oracle prices, price_k=1) vs L (the live planner), and F (price_k=1, fixed prices) vs L,
on develop 2016/17-2022/23, paired by (season, start). Pass: O - L mean end team value
>= +20 tenths and mean season net points >= +10.

    uv run python dev/price_gate_a.py --l results/price-a-l --o results/price-a-o \
        --f results/price-a-f --out results/price-gate-a
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from scipy import stats

MIN_VALUE = 20.0  # tenths of £m (spec, fixed before the runs)
MIN_POINTS = 10.0


def units(gws: pd.DataFrame) -> pd.DataFrame:
    g = gws.sort_values("gw_index").groupby(["season", "start_id"])
    return pd.DataFrame({"points": g["net_points"].sum(), "end_value": g["market_value"].last()})


def _stats(d: pd.Series) -> dict:
    n = int(d.size)
    se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    half = float(stats.t.ppf(0.9, n - 1) * se) if n > 1 else float("nan")
    return {"mean": float(d.mean()), "se": se, "ci80": [float(d.mean()) - half, float(d.mean()) + half]}


def compare(base: pd.DataFrame, arm: pd.DataFrame) -> dict:
    joined = base.join(arm, lsuffix="_b", rsuffix="_a", how="inner")
    missing = sorted(set(base.index) ^ set(arm.index))
    if missing:
        print(f"units in only one run, dropped: {missing}")
    return {
        "n": int(len(joined)),
        "points": _stats(joined["points_a"] - joined["points_b"]),
        "end_value": _stats(joined["end_value_a"] - joined["end_value_b"]),
    }


def decide(r: dict) -> bool:
    return r["end_value"]["mean"] >= MIN_VALUE and r["points"]["mean"] >= MIN_POINTS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for name in ("l", "o", "f", "out"):
        ap.add_argument(f"--{name}", type=Path, required=True)
    args = ap.parse_args()
    load = lambda p: units(pd.read_parquet(p / "gws.parquet"))  # noqa: E731
    l_units, o_units, f_units = load(args.l), load(args.o), load(args.f)
    o_vs_l, f_vs_l = compare(l_units, o_units), compare(l_units, f_units)
    result = {
        "o_vs_l": o_vs_l,
        "f_vs_l": f_vs_l,
        "thresholds": {"end_value_tenths": MIN_VALUE, "points": MIN_POINTS},
        "pass": decide(o_vs_l),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "gate_a.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run** `uv run pytest tests/test_price_gate_a.py -q` (2 passed), then ruff check and format on both files.

- [ ] **Step 5: Commit** — `"feat(dev): price planner Gate A script"`.

---

### Task 7: Gate A runs and write-up (controller)

**Ask the user before pushing and before using the server.** Use a separate server worktree, as in the sizing (`deploy`: `git worktree add ~/price-a origin/feat/price-sizing`, `ln -s ~/fpl-optimizer/data data`, submodules, `cargo build --release -p fplopt-cli`). Never touch `~/fpl-optimizer`'s checkout.

- [ ] **Step 1: Equivalence check.** Run L with the new binary (`optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2`, seasons 2016-2022, starts `template@1,random:3@1,template@20,random:3@20`, `--jobs 12`, family `price planner A`, `--out results/price-a-l`). Its `gws.parquet` must equal the sizing's `results/price-sizing-base/gws.parquet` (local copy) on every column but `market_value`. Any difference means the defaults moved: stop and fix.
- [ ] **Step 2: Arms.** O = L's spec + `,prices=oracle,price_k=1` (`--out results/price-a-o`). F = L's spec + `,price_k=1` (`--out results/price-a-f`). Same starts, seasons, jobs and family.
- [ ] **Step 3: Gate.** Copy the results back and run `uv run python dev/price_gate_a.py --l results/price-a-l --o results/price-a-o --f results/price-a-f --out results/price-gate-a`.
- [ ] **Step 4: Write-up.** `docs/research/2026-11-price-planner-gate-a.md`: O vs L and F vs L (points and end value, with 80% CIs and n), solve times vs L, the decision. Add a PLAN §12 row. Copy the experiment-log rows (family `price planner A`) into `results/experiments.csv`. Commit.
- [ ] **Step 5: Next.** On a pass: write Stage B's plan (the forecast). On a fail: record the stop in PLAN §1's price line, and Stages B–D are not built.
