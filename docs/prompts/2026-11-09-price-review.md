# Prompt: a fresh look at prices and transfer timing

You're picking up a research line in `fpl-optimizer`, a Fantasy Premier League system. It forecasts expected points (xP), plans transfers with a MILP, and sends advice through a Telegram bot. The owner wants the system to earn **team value as well as points**. That means predicting FPL price changes well enough to act on them, in the choice of players, the timing of moves, or both.

Five attempts so far have produced one working forecast and no improvement to advice. The owner is not convinced that's the ceiling, and neither should you be. Your job is to go through all of it with fresh eyes, find what the earlier work missed or got wrong, and propose (then, once the owner approves, build and test) something that does better.

Don't take the earlier conclusions on trust. Some may rest on bugs, weak features, poor framing or bad luck. Equally, don't wave away a real negative result: if something is truly dead, say why.

## Read first

- `CLAUDE.md`: commands, invariants and conventions. The rules there are binding.
- `docs/PLAN.md`: §3 (data and timing), §5 (*Comparing policies*, the standard comparison) and §12 (the decision log; the rows dated 2026-11-06 to 2026-11-09 are this line of work).
- The history of the line, in order:
  1. Sizing: `docs/superpowers/specs/2026-11-06-price-sizing-design.md`, `docs/research/2026-11-price-sizing.md`.
  2. Price-aware planner, Stage A / Gate A: `docs/superpowers/specs/2026-11-06-price-planner-design.md`, `docs/research/2026-11-price-planner-gate-a.md`.
  3. Price forecast, Stage B / Gate B (passed): `docs/superpowers/plans/2026-11-07-price-forecast-stage-b.md`, `docs/research/2026-11-price-forecast-gate-b.md`.
  4. Planner using the forecast, Stage C (stopped): `docs/superpowers/plans/2026-11-07-price-planner-stage-c.md` and its PLAN §12 row.
  5. Noise study, which produced the standard comparison: `docs/research/2026-11-noise-study.md`.
  6. Early moves (policy E: move at the lockdown, parked before it ran): `docs/superpowers/specs/2026-11-07-early-moves-design.md`.
  7. Move timing (policy T: move tonight only if a price change is due) and the night model, Gate N (failed): `docs/superpowers/specs/2026-11-09-move-timing-design.md`, `docs/research/2026-11-night-price-model.md`.
- Code, all Rust (new production code must be Rust; Python under `dev/` is fine):
  - price forecast: `crates/fplopt-models/src/price.rs`;
  - price features: `crates/fplopt-features/src/price.rs`;
  - night model: `crates/fplopt-features/src/night.rs`, `crates/fplopt-models/src/night.rs`;
  - evaluation: `crates/fplopt-backtest/src/price_eval.rs`, `crates/fplopt-backtest/src/night_eval.rs`, `crates/fplopt-backtest/src/money_value.rs`;
  - the planner's price handling, selling rule and early-move simulation: `crates/fplopt-optimize`, `crates/fplopt-backtest`;
  - the as-of store: `crates/fplopt-store` (`as_of`, `as_of_early`, cutoff vs deadline).

## What's established (check it, don't assume it)

- **Price changes:**
  - they happen overnight, about 02:30 UK;
  - each one is at most ±£0.1m, with at most £0.3m per gameweek;
  - they're driven by net transfers relative to ownership, against a hidden threshold that FPL retunes;
  - a player who goes from flagged red to unflagged is price-locked for a while;
  - a seller gets only half the profit on a rise, rounded down to £0.1m.
- **What money is worth:** £1m is worth about 0.395 xP per gameweek to the planner, measured by paired solves at fixed states (`fplopt-rs research money-value`).
- **We lose team value:** £99.2m at GW9 of 2026/27, against a top-10k median of £103.1m.
- **Planning with perfect knowledge of future prices** is worth about 47 points a season. Gate A found most of that is hindsight, because prices follow form.
- **Gate B passed.** On 2023/24–2024/25 the forecast's error on price changes 1–6 deadlines ahead is 0.807, against 0.982 for "no change" and 2.891 for momentum (squared tenths). That's 31% better than no change at one deadline and 13% at six.
- **Stage C lost.** The planner using forecast prices on develop 2022/23 lost 10–36 points a season and £0.6–1.2m of team value. It changed transfers in about 80% of gameweeks and expected +158 against a realized +68: it overrated price moves.
- **Timing:** moving every transfer right after the previous deadline would have kept about £2.3m a season, with 3.3% of bought players flagged before the deadline.
- **The night model ranks well but is miscalibrated:**
  - its log loss is 47% (rises) and 34% (falls) lower than the base rate;
  - rises are badly overconfident: at p ≥ 0.5, predicted 0.63 against a realized 0.31;
  - falls are calibrated: 0.60 against 0.57;
  - it was trained on 2020/21's last 65 nights plus 2021/22, and tested on 2022/23.

## Angles worth checking (starting points, not conclusions)

1. **Night-model rises:**
   - **Is the label right?** The evening snapshot is at about 18–19 UK and the label is the first snapshot 04:00–12:00 the next morning. Could the label catch a change from another night, or miss one because of snapshot gaps?
   - **Is the feature right?** Is cumulative net transfers since the last change correct across gameweek resets and wildcard or Free Hit transfers (which may not count toward the threshold)?
   - **Is it drift?** Is the overconfidence a threshold change between seasons, or something the model could learn: early season vs late, ownership bands, how many players are near the threshold on the same night?
   - **Would a mechanistic model do better?** For example, a "progress to threshold" state per player with a fitted threshold per season, updated online from observed changes. It might carry across seasons better than a booster.
   - **What about online recalibration within the season?** Each night's outcomes are known by the next morning, so a model that recalibrates on the season's own recent nights is legitimate and may remove the drift.
2. **Stage C's overrating:**
   - Check that price gains were valued correctly: selling price under the half-profit rule, discounting over the horizon, and money counted once (bank vs squad value).
   - Check how much 0.395 xP per £1m was multiplied by.
   - A forecast that's 31% better than "no change" still has large errors, and the MILP exploits whichever errors favour it. Try shrinking the forecast toward zero, valuing price only as a tie-breaker, or capping price's share of the objective.
3. **Uses not tried:**
   - avoiding imminent falls among players we already own (sell earlier);
   - holding back a rise-bound transfer target versus buying now;
   - bank management;
   - price as a secondary objective inside a small xP tolerance (a lexicographic objective).
4. **Clean data the earlier work didn't use:**
   - The live season 2026/27 has dense 15-minute snapshots and keeps growing.
   - A **prospective live shadow** (log the model's night calls now, score them each morning) is a valid test on data no model has seen. It may be the honest way to test any new night model, since 2022/23 is now spent.
5. **Anything else you find.** Bugs, leakage, wrong assumptions in the plans, or better framings.

## Data rules (binding)

- **Never read 2025/26.** It's the holdout.
- **2022/23 has been used:**
  - as Gate N's test season (the night model) and as Stage C's develop season, so a new night model can't be judged on it;
  - you may use it as training data or for diagnosis, but say so.
- **2023/24–2024/25:**
  - Gate B already evaluated the price forecast's accuracy on them;
  - no planner or timing policy has been backtested on them;
  - keep them clean for a frozen policy's single pre-registered check.
- **Intraday snapshots start in 2021**; earlier seasons have deadline prices only.
- **Every change must be tested before it drives advice:**
  - tune on develop seasons, deflated for the number of variants tried;
  - pre-register a gate, written into the spec and committed before the run;
  - run on data the change hasn't seen, once, logged in `results/experiments.csv` before anything is printed.
  - Compare planner changes with the standard comparison (PLAN §5). Measure money kept separately: the comparison counts one gameweek's points per decision.
- **Data directories:**
  - `raw/` is append-only;
  - never write `data/` or `raw/` by hand;
  - read only through `DataStore::as_of(...)` / `as_of_early(...)`;
  - feature builders take only the view;
  - never key on FPL player ids (use `player_key`).
- **No scraping** of fpl.page or other price-predictor sites. Use your own archive.
- **Don't touch** the live server checkout `~/fpl-optimizer` or the bot. Ask the owner before any server use or push.

## Deliverables, in order

1. **A diagnosis.** Write `docs/research/2026-11-price-review.md`, covering:
   - what the earlier work got right;
   - what it got wrong or missed, with evidence (queries, small checks, file and line references, not opinions);
   - the bugs you found;
   - where you think the real value is, with a rough size in points per season using 0.395 xP per £1m per gameweek.
2. **A proposal.** Write one or two designs as specs in `docs/superpowers/specs/`. Each states the policy or model, the measurement, the gate and the data it will be judged on. Stop there and get the owner's approval.
3. **After approval:** a plan, then build in Rust with tests (`cargo test --workspace`, clippy `-D warnings`, `cargo fmt`), develop runs, and the pre-registered gate.

Be concrete and quantitative. If you conclude something can't work, show the number that says so. If you find something that might work, size it before building it.
