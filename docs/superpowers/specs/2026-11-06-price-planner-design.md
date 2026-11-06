# Price-aware planner and price forecast: design

**Date:** 2026-11-06. **Status:** approved by the user in chat. Written in Rust, on a branch from `feat/price-sizing` (or `main` once that merges).

**Why:** the price sizing (`docs/research/2026-11-price-sizing.md`, PLAN §12 2026-11-06):
- Our system loses team value: £99.2m at GW9 of 2026/27, against £103.1m for the top-10k median.
- Planning with perfect price foresight is worth up to about 47 points a season (80% range 38–57).
- Timing transfers within a GW was dropped.

**Scope.** This spec covers four stages, each ending in a stop-or-go gate fixed here:
- the planner's support for changing prices, with a foresight upper bound;
- the price forecast;
- tuning plus the pre-registered check;
- going live.

Each stage gets its own implementation plan.

## Stage A: the planner models changing prices (`crates/fplopt-optimize`)

### Price paths
- **Paths per candidate.** `Player` gets an optional price path `price_path: Option<Vec<f64>>`: the expected price (tenths of £m) at each horizon GW's deadline.
  - With no path, the price is today's `price` at every GW, so behaviour is unchanged.
- **Buying** at horizon GW t costs `price_path[t]`.
- **Owned player's first sale** at t pays the selling value `purchase + 0.5 · max(0, price_path[t] − purchase)` when it rose, and `price_path[t]` otherwise.
  - The value is not floored, because expected prices are fractional.
  - With fixed prices it equals today's `sell_price` up to FPL's floor. The planner keeps the exact `sell_price` when the player has no path.
- **Bought within the plan.** A player bought in GW t and sold in GW s > t earns `p_t + 0.5 · max(0, p_s − p_t)`, which depends on t. It is modelled with cohort variables `sell_later[k, t, s]`, linked to the buy in GW t.
  - These variables exist only for candidates whose path moves: max − min ≥ 0.05 over the horizon.
  - Other candidates keep today's `sell_later[k, s]` at the buy price.
- **Free Hit GWs** are unchanged: a Free Hit squad's buys and sales happen at that GW's prices and are reverted.

### End-of-horizon money value
- **What counts as money.** M = bank after the last horizon GW + Σ over the squad held then of each player's selling value at the last horizon GW's price. For a player bought within the plan, his cohort's buy price is used.
- **Objective term.** The objective adds `λ · M` with λ (points per tenth) = `k · v · c · max(0, 38 − gw_index_last) / 10`:
  - v = 0.395 expected points per GW per £1m (`research money-value`, develop GW8–29);
  - c = 0.93 (the realized-on-predicted slope);
  - `gw_index_last` = the last horizon GW's index.
- **Not decayed,** like chip terminal values. The constant part of M (the starting money) drops out of comparisons.
- **New key:** `OptimizerParams.price_value_k` (spec key `price_k`), default 0. `itb_value` stays as it is.
- **Default equivalence:** with no paths and k = 0, plans are identical to today's. The existing optimizer tests, the optimizer parity golden and the backtest golden must pass unchanged.

### Foresight (oracle) prices
- **Source.** Backtest-only price-path source `oracle`: the real price at each horizon GW's deadline, from `player_gw.value` (equal to the last pre-deadline snapshot, checked 2026-11-06 on 47,875 rows), covering 2016/17 onward.
  - A player without a row at a future GW (his club blanks) keeps his latest known price.
- **It reads the future by design:**
  - only the backtester accepts it, through spec key `prices=oracle`;
  - `optimize plan`, the bot and `live` refuse it;
  - `check leakage` lists it as a deliberate exception and does not run it.
- **Spec key** `prices=fixed|oracle|forecast` (default `fixed`).

### Gate A (develop 2016/17–2022/23)
- **Setup:** starts `template@1,random:3@1,template@20,random:3@20`, as in the sizing.
- **Arms:**
  - L = the live planner (`optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2`);
  - O = L + `prices=oracle,price_k=1`;
  - F = L + `price_k=1` with fixed prices: the end value's own effect, a diagnostic only.
- **Pass:** O − L has a mean end-of-season team value (`market_value` at the last GW) ≥ +20 tenths (£2m), and a mean season net points ≥ +10, over (season, start) units.
- **Otherwise stop:** even perfect foresight doesn't pay through this planner. Write it up and keep price prediction a non-goal.

## Stage B: the price forecast (`crates/fplopt-features`, `crates/fplopt-models`)

### Data
- **Prices:** the last `player_snapshot` row before each deadline, as the backtest pool uses.
- **Seasons:** daily snapshots (about four a day, with `transfers_in_event`, `transfers_out_event`, `cost_change_event`, `selected_by_percent`) cover 2021/22 through 2024/25 and 2026/27.
- **2025/26 is never used** for fitting, tuning or checking.

### Target
- For each player in the pool at deadline t and each h = 1..6: Δp_h = price at the deadline of gw_index t+h minus price at t, in tenths. Rows whose t+h falls past the season's end are dropped.
- One LightGBM regression per h (L2), with v1's LightGBM wrapper (`crates/fplopt-lgbm`).
- The expected path is p_t + predicted Δp_h, clipped to [p_t − 0.3h, p_t + 0.3h] tenths (prices move at most 0.1 a night, and much less on average).

### Features
- **Rule:** as of the deadline, built only from the `DataStore` view (the feature-builder rule: one argument, the view).
- **Transfer and ownership signals:**
  - net transfers over the last 1, 3 and 7 days, each divided by ownership (`selected_by_percent` × managers);
  - net transfers this GW so far;
  - ownership.
- **Price signals:** price changes over the last 1, 7 and 28 days; price; price relative to the position's median.
- **Availability:** status, chance of playing, days since the news was added.
- **Expected points:** v1 expected points for the next 1 and 3 GWs, and the player's v1 expected points rank within his position.
- **Calendar:** the number of fixtures in the next 3 GWs, the gw_index, and days to the deadline of t+1.
- **Leakage:** `check leakage` gains `price:forecast`, with the same corrupt-the-future check as the other models.

### Fits
- **Walk-forward:** a forecast for season S is fitted on the snapshot seasons before S, excluding 2025.
  - Develop: fit on 2021/22, predict 2022/23.
  - Check: fit 2021/22–2022/23 → predict 2023/24; fit 2021/22–2023/24 → predict 2024/25.
  - Live 2026/27: fit 2021/22–2024/25 plus 2026/27's earlier deadlines (refit when the context is rebuilt).
- **Hyperparameters:** chosen on develop only (2022/23 predictions). At most 6 settings, and every one is logged.

### Baselines
- **N0:** no change (today's planner).
- **N1:** momentum, h × (p_t − p_{t−1 GW}), clipped the same way.

### Gate B (check 2023/24–2024/25)
- **Measure:** squared error of the predicted price at t+h, over the planner's candidates: the optimizer's pruned candidate set at each deadline from the template and random:3 states, the way `models eval` defines candidate MSE. It is pooled over h = 1..6.
- **Pass:** the forecast's pooled MSE is lower than both N0's and N1's, and a one-sided Diebold–Mariano test, clustered by GW, gives p < 0.10 against each.
- **Also reported:** per h, and develop vs check.
- **Otherwise stop:** write it up.

### Addendum (2026-11-07): Stage B as built

Written before any Stage B code, after mapping the code (plan `docs/superpowers/plans/2026-11-07-price-forecast-stage-b.md`):

1. **Ownership scaling.** No total-manager count exists in the data, so transfers are scaled two ways: by ownership, `net / (selected_by_percent + 0.1)`, and by the deadline's market activity, `net / Σ_players |net|` at the same snapshot.
2. **Candidates.** "The planner's candidates" for Gate B are the union of the top `DEFAULT_PRUNE_N` per position by summed v1 xP, as `models eval` defines candidate MSE. This replaces the optimizer's pruned set from template and random states.
3. **No fixtures feature.** It is dropped: v1 xP for the next 1 and 3 GWs already carries fixtures and blanks.
4. **Leakage.**
   - `check leakage` covers the snapshot feature frame (`price_features`).
   - Training-set visibility holds by construction and is unit-tested: rows come from the view at the fit cutoff, so a target exists only when its deadline price is visible.
   - xP inputs are v1's, covered by `model:v1`.
5. **Clip.** The clip is ±3h tenths, i.e. ±£0.3m per GW (the bound above meant £m).

## Stage C: tuning and the pre-registered check

- **Tuning (develop 2022/23 only, forecast prices from Stage B's develop fit):**
  - choose k ∈ {0, 0.25, 0.5, 1} for `prices=forecast,price_k=k` (0 added on 2026-11-06 after Gate A, before any forecast existed: the money value alone cost 10.5 points, `docs/research/2026-11-price-planner-gate-a.md`);
  - the criterion is season points net, with team value as the tie-break;
  - the family is deflated for 4 variants (mean − SE·√(2 ln 4)) and logged as family `price planner`;
  - the chosen settings are written to `config/prices/settings.json` and tagged `price-check` before the check runs.
- **Gate C (check 2023/24–2024/25):** starts as in Gate A; the forecast planner (chosen k) vs L.
  - **Team value:** the forecast planner must end seasons with higher team value than L: mean difference > 0, one-sided t-test over (season, start) units, p < 0.10.
  - **Points:** the 80% lower bound of the season-points difference (GW-block bootstrap, as `backtest compare` does) must be > −10.
  - **Rules for the run:**
    - It is run once, logged before printing, and refused if the log already holds it.
    - It is refused on a dirty checkout or when HEAD isn't the tag, like `holdout run`.
    - The command is `fplopt-rs backtest compare` with family `price check`.
- **Otherwise stop:** write it up; the price forecast stays out of advice.

## Stage D: live

- **Plan contexts:**
  - `live prepare` (`fplopt_bot::builder`) fits or loads the forecast and writes each candidate's expected price path into the plan context, next to the xP.
  - Plans (`/plan`, `/chips`, alerts) use `prices=forecast` with the chosen k through `live_params`.
  - The context records the forecast's fit seasons and settings.
- **Shadow:** for the first GW after Gate C passes, the bot computes both plans, shows the fixed-price plan, and logs the price plan in `recommendations` (a new `variant` column, default `live`). After one GW, the price plan is switched on.
- **Message text** is unchanged.
- **Goldens:** the bot replay golden and `dev/bot_parity.py` run with prices off (`price_k` 0), so they stay byte-for-byte.
- **Prices update:** live FPL prices (`with_live_prices`) replace p_t, and the path shifts by the same amount.

## Tests

- **Optimizer (unit, tiny synthetic problems):**
  - a rising player is preferred to an equal-xP flat one only when k > 0;
  - an owned riser's sale pays half the rise;
  - a player bought in GW1 and sold in GW3 earns the cohort selling value;
  - with no paths and k = 0 the plan equals today's;
  - the Free Hit reverts.
- **Goldens:** existing optimizer, backtest and bot goldens unchanged.
- **Oracle:** refused outside the backtester. `check leakage` lists it as excluded.
- **Forecast:**
  - a leakage check (`price:forecast`);
  - a features unit test on a hand-made view;
  - walk-forward fits never include the predicted season or 2025.
- **Gates:** each gate's script has unit tests on hand-made frames, as the sizing scripts do.

## Not in scope

- Timing transfers within a GW (dropped by the sizing).
- Price-change alerts.
- Chip-specific price handling beyond the planner's existing chip logic.
- Rank or ownership objectives.
