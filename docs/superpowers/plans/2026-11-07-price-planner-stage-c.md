# Price-Aware Planner, Stage C Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the backtested planner uses the Gate-B forecast's price paths (`prices=forecast`). Tune the money weight k on develop 2022/23, freeze it, and run the pre-registered Gate C once on 2023/24–2024/25.

**Architecture:**
- A `ForecastPrices` source in `fplopt-backtest` (next to `oracle.rs`) fits the frozen forecast once per season. It uses `training_rows` and `fit_price` with v1 xP from `Caches`, fitted on seasons `first_fit..S−1` excluding 2025, exactly as `research price-eval` does.
- At each decision it predicts the candidates' changes and sets the price paths: path[t] = price + dp_{h=t} for t ≥ 1.
- The spec key `prices=forecast` turns it on.
- Gate C runs through `backtest compare`, which already writes both arms' `gws.parquet` (with `market_value`) and the GW-block bootstrap of the full-run points difference. A `dev/price_gate_c.py` script enforces run-once and reads the verdict.

**Tech Stack:** Rust (`fplopt-backtest`, `fplopt-cli`), Python + pandas + scipy for the gate script.

**Spec:** `docs/superpowers/specs/2026-11-06-price-planner-design.md`, Stage C. The k grid is {0, 0.25, 0.5, 1}, deflated over N = 4. Gate B passed (`docs/research/2026-11-price-forecast-gate-b.md`). The frozen forecast settings are `config/prices/forecast.json` (`params` p4, `first_fit` 2021). Branch `feat/price-sizing`.

**Facts:**
- **`PriceSource { Fixed, Oracle }`** (`crates/fplopt-backtest/src/policies.rs:257-259`).
  - `prices=forecast` is refused at `specs.rs:325-326`; the test is at `specs.rs:563-564`, and `PRICE_SOURCES` is at `specs.rs:174`.
  - `DecisionContext.future: Option<&FuturePrices>` is at `policies.rs:53-55`. It is filled in `simulator.rs` `decide_step` and in `money_value.rs` `solve_at` through `Caches::future_prices_for`.
  - `optimizer_plans` (`policies.rs:~505-520`) calls `problem.set_price_paths(...)`. `PlanInput::set_price_paths` overwrites path[0] with today's price.
- **Forecast API** (`crates/fplopt-models/src/price.rs`): `training_rows(store, cutoff, seasons, xp_at)`, `PriceParams::preset(name)`, `fit_price(rows, params) -> PriceFit`, `deadline_features(view, xp) -> Table`, `predict_changes(fit, features) -> Table(player_key, h, dp)`, `refuse_holdout`.
  - `research price-eval` (`crates/fplopt-backtest/src/price_eval.rs`) shows the fitting pattern: the cutoff is S's first deadline, `fit_seasons(S, first_fit)`, and xP comes from `caches.xp("v1", view)`.
- **`backtest compare`** (`crates/fplopt-cli/src/backtest.rs` ~950-1070) writes `gws.parquet` (both arms, a `policy` column, `market_value`), `paired.parquet` and `metrics.json`. The metrics include the full-run difference with GW-block bootstrap 80% CIs. `--no-per-decision` skips the per-decision windows. It logs to the experiment log under `--family`.
- Forecast prices need snapshot seasons, so they can only run for seasons ≥ `first_fit` + 1 = 2022.

**Rules:**
- Cargo with `-j 2`.
- Never use season 2025.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- No push or server use without the controller's go-ahead.
- With `prices=fixed`, behaviour must be unchanged (labels, plans, goldens).

---

### Task 1: `prices=forecast` in the backtester

**Files:**
- Create: `crates/fplopt-backtest/src/forecast_prices.rs`
- Modify:
  - `crates/fplopt-backtest/src/lib.rs`
  - `crates/fplopt-backtest/src/policies.rs` (`PriceSource::Forecast`, the `DecisionContext` field, `optimizer_plans`, the name suffix `forecast`)
  - `crates/fplopt-backtest/src/specs.rs` (accept `forecast`; update the refusal test)
  - `crates/fplopt-backtest/src/simulator.rs` (`Caches`)
  - `crates/fplopt-backtest/src/money_value.rs` (context literal)
  - `crates/fplopt-backtest/src/probes.rs` (context literal)

**Design:**
- **`ForecastConfig`:** `{ params: String, first_fit: i64 }`, read from `config/prices/forecast.json` by `ForecastConfig::load(path)`. The default path is `config/prices/forecast.json` relative to the working directory, like other config files; check how `config/chips/terminal-values.json` is located and do the same.
- **`ForecastPrices`:** `{ season, fit: PriceFit }`. `ForecastPrices::fit(store, caches, season, config) -> Result<Self>`:
  - refuse 2025 and any season ≤ `first_fit`;
  - cutoff = the season's first deadline (`season_schedule(store, season, 1)?[0].deadline`);
  - rows = `training_rows(store, cutoff, &fit_seasons, &|v| caches.xp("v1", v).map(|x| x.table.clone()))`, where `fit_seasons` = first_fit..season−1 without 2025;
  - `fit_price(&rows, &PriceParams::preset(&config.params)?)`.
- **`ForecastPrices::paths(&self, view, xp: &Table, gws: &[HorizonGw]) -> Result<HashMap<i64, Vec<f64>>>`:**
  - `deadline_features(view, xp)`, then `predict_changes`;
  - for each player, path[0] = 0.0 (overwritten later), and path[t] = features' `price` + dp at h = t for t = 1..gws.len()−1 (h ≤ 6; if the horizon is longer than 6, carry h = 6 forward);
  - players without a feature row are absent, so they keep fixed prices.
- **Generalize the context field:** `DecisionContext.future: Option<&FuturePrices>` becomes `prices: Option<PriceInput<'a>>`, with `enum PriceInput<'a> { Oracle(&'a FuturePrices), Forecast(&'a ForecastPrices) }`.
  - `Policy::wants_future_prices()` becomes `Policy::price_source() -> PriceSource`.
  - `Caches` gets `forecast: OnceMap<i64, ForecastPrices>` and `forecast_prices_for(policy, store, season, config)`. The config is loaded once per `Caches`; store it in `Caches` when the policy asks for it.
  - In `optimizer_plans`, `Forecast` computes `paths(ctx.view, ctx.xp, &problem.gws)` and calls `problem.set_price_paths(|p| paths.get(&p.player_key).cloned())`. `ctx.xp` is the v1 xP `Table` the policy already gets; check that `DecisionContext.xp` is v1's table for a v1 policy.
- **Spec:** `prices=forecast` is accepted on `optimizer` only, and the label suffix is `forecast`. The spec also refuses `prices=forecast` with a non-v1 xP model ("the price forecast uses v1 xP").

- [ ] **Step 1: Failing tests.**
  - `specs.rs`: `optimizer:v1:...,prices=forecast,price_k=0.5` builds, and its name contains `pk=0.5` and `forecast`. `optimizer:rolling:prices=forecast` errors. The old "arrives with Stage B" assertion is replaced.
  - `forecast_prices.rs`: a pure helper `paths_from_changes(prices: &HashMap<i64,f64>, changes: &Table, n_gws: usize) -> HashMap<i64, Vec<f64>>`, unit-tested on a hand-made changes table:
    - player 1 at price 50 with dp 1, 2, 2 for h = 1..3 and n_gws = 4 gives `[0, 51, 52, 52]` (path[0] is a placeholder);
    - with n_gws = 8, h = 6's value is carried forward;
    - a player absent from `prices` is skipped.
  - `ForecastConfig::load` on a temp file; a season ≤ first_fit and 2025 are refused.
- [ ] **Step 2: See them fail. Step 3: Implement.**
- [ ] **Step 4:**
  - Run `cargo test -j 2 --workspace`, clippy and fmt. The goldens and label tests must be unchanged.
  - **Smoke test:** with `FPLOPT_DATA_DIR=C:/Users/samin/Documents/GitHub/fpl-optimizer/data` and `--xp-cache C:\Users\samin\AppData\Local\Temp\claude\6ff2f6c53ca963aa\50295afa-4ab0-4ac1-a579-1e2ddcc3dc5e\scratchpad\price-eval-smoke\xp-cache` (warm for 2021–2024), run `backtest run --seasons 2022 --spec optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2,prices=forecast,price_k=0.5 --starts template@1 --jobs 1 --experiments <scratch>/exp.csv --out <scratch>/smoke-c`.
  - Report the season total, end `market_value` and run time, against the same command without the two keys.
- [ ] **Step 5: Commit** `feat(backtest): prices=forecast, the planner on the price forecast's paths`.

---

### Task 2: `dev/price_gate_c.py` (tuning choice and the run-once check)

**Files:**
- Create: `dev/price_gate_c.py`, `tests/test_price_gate_c.py`

**Behaviour:**
- **`tune --base RUN --arm K=RUN ...`** (develop 2022/23): for each k, the paired unit differences against the base run are computed with Gate A's `units`/`compare` logic:
  - season net points, and end `market_value` at each unit's last GW;
  - each points mean is deflated as mean − SE·√(2 ln N) with N = number of arms (4);
  - the chosen k has the highest deflated points mean, with the higher end-value mean breaking ties to 1e-9.
  - It prints a table and writes `tune.json`. Copy `units`/`compare` from `dev/price_gate_a.py`, or import that module by path the way the tests import dev modules.
- **`check --out DIR [--experiments results/experiments.csv] [--binary target/release/fplopt-rs]`:** the run-once gate.
  1. **Refusals, before anything runs:**
     - the experiment log already has family `price check`;
     - the working tree is dirty (`git status --porcelain`, ignoring `results/experiments.csv`);
     - HEAD isn't the commit tagged `price-check`;
     - `--out` exists and isn't empty.
  2. **Read** the frozen `config/prices/planner.json` (`{"price_k": K}`).
  3. **Run** `<binary> backtest compare --a optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2,prices=forecast,price_k=K --b optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2 --seasons 2023,2024 --starts template@1,random:3@1,template@20,random:3@20 --no-per-decision --family "price check" --out DIR/compare` (plus `--jobs` and `--xp-cache` passthrough). It logs its own row.
  4. **Verdict:**
     - *Team value:* from `DIR/compare/gws.parquet`, split by `policy`, the units' end `market_value` differences (A − B), one-sided t-test p (H1: mean > 0).
     - *Points:* from `DIR/compare/metrics.json`, the full-run realized difference's bootstrap 80% CI lower bound for split "all". Find the key by reading the file; print the path used.
     - Pass = p < 0.10 and lower bound > −10.
  5. **Write** `DIR/gate_c.json` and append a row to the experiment log (family `price check verdict`, through a small CSV append matching the columns of `results/experiments.csv`), then print.
- **Tests** (pure functions on synthetic frames, no subprocess):
  - `refusal(log_families, dirty, head, tag, out_nonempty)` returns a reason or None;
  - `tune_choice` with deflation (a high-variance arm loses to a steady one);
  - `team_value_test` on hand-made units;
  - `verdict(p, lower)` at the boundaries (p = 0.10 fails; lower = −10 fails).

- [ ] **Steps 1–5:** write the tests, see them fail, implement, run `uv run pytest tests/test_price_gate_c.py -q` and ruff, then commit `feat(dev): price planner Gate C (tuning choice, run-once check)`.

---

### Task 3: Develop tuning (controller; server, after the user's go-ahead)

- [ ] **Setup:** a separate server worktree, as for Gate A, with a warm xP cache if possible.
- [ ] **Runs** on 2022/23 with starts `template@1,random:3@1,template@20,random:3@20` (8 units), family `price planner develop`:
  - the base: the live spec;
  - four arms: live spec + `,prices=forecast,price_k=K` for K ∈ {0, 0.25, 0.5, 1}.
- [ ] **Choose:** `uv run python dev/price_gate_c.py tune --base … --arm 0=… --arm 0.25=… --arm 0.5=… --arm 1=…`. Record the result in `docs/research/2026-11-price-planner-gate-c.md`'s develop section.
- [ ] **Sanity check (not a gate):** if every arm's deflated points mean is below −10, report to the user before freezing. Gate C would almost surely fail.

### Task 4: Freeze and run Gate C once (controller)

- [ ] **Freeze:** write `config/prices/planner.json` `{"prices": "forecast", "price_k": K, "chosen_on": "...", "frozen": "..."}`, commit it with the develop log rows, and tag `price-check`.
- [ ] **Build** the release binary at that commit on the server, in the worktree checked out at the tag.
- [ ] **Run once:** `uv run python dev/price_gate_c.py check --out results/price-gate-c --jobs 14`.
- [ ] **Write-up:** finish `docs/research/2026-11-price-planner-gate-c.md`, add a PLAN §12 row, copy the experiment-log rows, commit and push.
- [ ] **Next:** on a pass, write Stage D's plan (live: the forecast in the plan context, one shadow GW, then on). On a fail, record the stop: PLAN §1's price line, and no change to advice.
