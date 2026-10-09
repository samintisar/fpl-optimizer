# Phase 5: Real xP models — Implementation Plan

> **For agentic workers:** Implement task-by-task (one implementer per task or wave, then an independent review). TDD: failing tests first. Before using LightGBM, scipy or scikit-learn APIs, fetch their current docs (`npx ctx7@latest library LightGBM "<question>"`, then `docs`).

**Goal:** Replace the baseline xP with PLAN §6's component models, then show it helps decisions:
- market-implied team model;
- player goal and assist shares;
- a LightGBM minutes model;
- bonus, saves, cards and the other components;
- calibration.

Phase 5 is done when (PLAN §9):
1. the new xP beats both baselines (rolling, `ep_next`) on component metrics in validation;
2. the optimizer beats greedy with the new xP: paired, same xP, chosen on develop, confirmed on validate, deflated for the variants tried.

**Two PRs.** The phase is too large for one review.
- **5a** (`phase-5a-models`, Tasks 0–7) builds the models and meets criterion 1.
- **5b** (`phase-5b-decisions`, Tasks 8–11) feeds the minutes model into the optimizer, pre-registers the comparison, and meets criterion 2.

**Spec:**
- `docs/PLAN.md` §6 (models), §5 (*Metrics*, *Splits*, *Baselines*, *Comparing policies*) and §4 (determinism, leakage).
- §7 for bench and pruning in 5b.
- Phase 2–4 code: `fplopt.features`, `fplopt.models`, `fplopt.backtest`, `fplopt.optimize`.

**Holdout:** 2025/26 stays untouched. Model fits, evaluation and backtests refuse it, as in Phases 3–4.

---

## Decisions

### Model structure

- **Fit and predict are separate.** A component model has two parts:
  - `fit(view) -> Fitted`: a frozen dataclass of parameters, or a LightGBM booster wrapped in one;
  - `predict(view, fitted) -> frame`.
- **Walk-forward fits use `AsOfView.earlier(cutoff)`.** This is a new method that returns `store.as_of(cutoff)` and refuses a cutoff after the view's deadline, so a fit can't see more than the view.
- **Refit cadence.** The fit for deadline `d` uses the view at the season's most recent refit GW deadline ≤ `d`, with refit GWs at `gw_index` 1, 5, 9, … (every 4 GWs; tunable). The registered model is still a plain `Callable[[AsOfView], DataFrame]`:
  ```
  v1(view) = assemble(view, fit(view.earlier(refit_deadline(view))))
  ```
  The leakage check therefore covers both fit and predict unchanged.
- **Model modules keep no state.** Fits are memoized by the caller: `Caches` keys them on (model, component, cutoff). This needs a small `FittedModel` protocol (`refit_deadline`, `fit`, `predict`) next to the plain callable.
  - Expected cost: about 10 fits per season, each ≤ 30 s, and ≤ 2 s to predict each deadline.
  - The leakage check refits at every deadline it checks, so expect it to take about 10 minutes longer.
- **Determinism (PLAN §4):**
  - LightGBM only through `fplopt/models/gbm.py`, which pins `seed`, `num_threads=1`, `deterministic=True` and `force_row_wise=True`.
  - scipy optimizers start from fixed points with fixed tolerances.
  - Rows are sorted by key before every fit.
  - The architecture scanner lets `scipy.optimize` / `scipy.stats` / `scipy.special` into `fplopt.models`, and `lightgbm` only into `gbm.py` (`module_attrs`).
- **Components are stored, not just xP.**
  - `components(view)` returns one row per pool player and horizon fixture (doubles get two rows) with: `p_start`, `p_60`, `p_sub`, `e_minutes`, `lambda_team`, `lambda_opp`, `p_cs`, `e_goals`, `e_assists`, `e_pen_goals`, `e_bonus`, `e_saves_pts`, `e_cards_pts`, `e_conceded_pts`, `xp`.
  - The xP frame is its per-GW sum.
  - The model is registered as `MODELS["v1"]`, which works with every existing `--xp` / spec.

### Training data and weighting

- **Seasons:** training uses every visible season (2016/17+), weighted by time decay rather than dropping old seasons (PLAN §3). 2016/17 is burn-in: it is evaluated but not used for tuning.
- **Starts before 2022/23 GW16 are inferred.** `starts` is null before 2022/23 GW16.
  - Starters per team-fixture = the 11 players with the most minutes (ties by `player_key`), inferred inside the features from `player_match`.
  - Task 3 measures the accuracy where the real `starts` exists (2022/23 GW16 – 2024/25) and records it in PLAN §6.3.
- **Player npxG sources:**
  - Understat `us_npxg` where present.
  - Otherwise `fpl_xg − 0.76 × penalty attempts`. Penalty attempts = penalty goals (`us_goals − us_npg`, or inferred from the taker) + `penalties_missed`.
  - PLAN §3 marks this approximation as VERIFY; Task 4 checks it on the 2022/23–2024/25 overlap.
- **Missing xG:** 2016/17–2018/19 cover only 46–72% of played rows. There, shares fall back to goals and assists, with stronger shrinkage, and the coverage is logged.

### Team model (PLAN §6.1)

- **Market λ for every fixture whose pre-match odds are visible at the deadline.**
  - Historically that is the football-data pre-match `avg` rows, so the target GW only.
  - Live it is The Odds API bookmakers: the median of each outcome's de-vigged probability over bookmakers.
  - The bookmaker margin is removed with the power method; Shin's method is tested as an alternative in Task 2.
  - λ_home and λ_away come from a weighted least-squares fit of a Dixon-Coles score distribution to 1X2 + O/U 2.5, plus the AH line when available.
  - ρ is fixed: fitted once on develop results and stored as a constant in the code.
- **Ratings, for horizon GWs without odds:**
  - Model: `log λ = base + home + attack − defence`.
  - Fitted by time-decayed weighted Poisson MLE (scipy). The targets are past matches' market-implied λ blended with `0.7·xG + 0.3·goals`, using one fixed blend weight that is tuned, never per fold.
  - Every played match since 2016/17 has pre-match odds, so the ratings are market-anchored historically too.
  - Priors: our Elo (`team_rating`). Promoted clubs start at their Elo prior.
- **Stats fallback:** the same ratings fitted on `0.7·xG + 0.3·goals` alone. Used when no odds are visible at all.
- **P(CS)** = P(opponent scores 0) under Dixon-Coles.

### Player model (PLAN §6.2–6.5)

- **Minutes, a LightGBM hurdle model:**
  - three parts: P(start); P(60+ | start); P(sub appearance | not start) with E[sub minutes].
  - Features: only what the data has. Recent starts and minutes, starts' share of team matches, days of rest, return from absence, positional competition (team-position starts concentration), price, new signing (`team_join_date`, 2021/22+), and whether the player has a snapshot.
  - Europe and cup matches, manager tenure and age have no source, so they are dropped and noted in PLAN §6.3.
  - Post-processing:
    - team normalization (Σ P(start) ≈ 11, Σ E[minutes] ≈ 990);
    - suspensions from yellow-card thresholds and red cards;
    - horizon decay toward the player's long-run start rate.
- **Adjustment layer (2021/22+ only, where flags exist):**
  - mapping from `chance_of_playing_next_round` and status to P(start), fit on 2021/22–2022/23 and checked on 2023/24–2024/25;
  - "Expected back DD Mon" parsed from `news` into per-GW availability.
  - Per-user overrides belong to the bot (Phase 7).
- **Shares:**
  - npxG/90 and xA/90 shares of team npxG while on the pitch, with a Marcel prior: previous seasons weighted 2-1-1, plus 480 minutes at the position's league average. New players get position + price.
  - Team consistency: Σ E[non-penalty goals] = team non-penalty λ over the minutes-weighted expected XI.
  - FPL assists = λ × the empirical share of goals that carry an FPL assist, split by xA share.
  - Penalties: P(on pitch) × P(taker), where P(taker) comes from `penalties_order` in 2021/22+ and from the team's past penalty-goal scorers before that, × team penalties per match × conversion.
- **Other components:**
  - bonus = E[bonus | goals, assists, CS, minutes], from a per-position regression on history, weighted toward recent seasons (the BPS rules changed in 2026/27);
  - GK saves: Poisson on the opponent's λ;
  - cards, own goals and penalty misses: shrunk per-player rates.
  - **Defcon is in 5b (Task 10):** it only scores in 2025/26+, and its prior depends on #14.
- **Assembly (PLAN §6.6):**
  - per fixture, using Poisson floors and thresholds;
  - points under the backtest rules of the season (`backtest_rules`), so develop/validate xP is on the same scale as `rescored_points`.
- **Calibration:**
  - isotonic fits for P(start), P(CS) and P(goal ≥ 1), and per-position linear recalibration of xP;
  - fit on develop predictions and checked on validate;
  - applied inside `v1` as fixed parameters: a refit only on develop, never per validate fold.

### Evaluation (criterion 1)

- **`fplopt models eval`** runs the walk-forward predictions over chosen seasons' deadlines (`--jobs`, by season) and writes `results/<UTC>-eval/predictions.parquet` (components + xP, every horizon) and `metrics.json`. It appends to `results/experiments.csv`.
- **Criterion 1, as tested:** the baselines have no components, so "beats both baselines" is tested on PLAN §5's xP and decision metrics. On validate (2023/24–2024/25), `v1` vs `rolling` and vs `ep_next` must pass all of:
  - lower MSE per player-GW at horizon 0 (one-sided Diebold-Mariano, clustered by GW, p < 0.10);
  - lower candidate-weighted MSE (optimizer pool candidates);
  - no worse MSE in any predicted-xP band, and no worse over horizons 1–5;
  - no worse captain regret and XI regret.
- **Component metrics** are reported against simple references (last-5 start rate; team goals from Elo) to diagnose each part:
  - minutes: log loss and RPS on 0 / 1–59 / 60+;
  - goals and assists: Poisson log-likelihood;
  - P(CS) and P(≥1 goal): Brier with decomposition;
  - reliability diagrams for each.
- **Tuning:** hyperparameters are chosen on develop walk-forward metrics (2017/18–2022/23, mean over seasons), then confirmed once on validate. Every variant is logged with its family, so the deflation counts it.

---

## File structure

| File | Responsibility |
|---|---|
| `src/fplopt/features/store.py` | `AsOfView.earlier(cutoff)` |
| `src/fplopt/features/history.py` | Training frames from the view: per player-fixture history with lags, inferred starts, rest days, team context |
| `src/fplopt/models/gbm.py` | The only LightGBM entry point (deterministic params) |
| `src/fplopt/models/team.py` | De-vig, Dixon-Coles market λ, ratings fit, stats fallback, P(CS) |
| `src/fplopt/models/minutes.py` | Hurdle model, normalization, suspensions, horizon decay |
| `src/fplopt/models/availability.py` | Flag mapping, news parsing |
| `src/fplopt/models/shares.py` | npxG/xA shares, Marcel prior, penalties, FPL assist fraction |
| `src/fplopt/models/components.py` | Bonus, saves, cards/OG/pen-miss rates (defcon in 5b) |
| `src/fplopt/models/assemble.py` | Per-fixture xP from components; calibration; `v1`, `FittedModel` |
| `src/fplopt/models/__init__.py` | `MODELS["v1"]`, `FITTED` registry |
| `src/fplopt/evaluate/` | Outcomes per player-fixture, metrics, DM test, reliability tables, `models eval` |
| `src/fplopt/backtest/simulator.py` | `Caches` memoizes fits by cutoff |
| `src/fplopt/cli.py` | `fplopt models eval`; `v1` accepted wherever `--xp` is |
| tests | `test_models_*.py`, `test_evaluate_*.py`, architecture / leakage / CLI extensions |

---

### Task 0: Plan, scaffolding (lead)
- [ ] Commit this plan, with a PLAN §12 row (fit/predict split, refit every 4 GWs, criterion 1 as tested above) and §6 notes (dropped minutes features, inferred starts).
- [ ] `AsOfView.earlier`, the `FittedModel` protocol, fit memoization in `Caches`, `gbm.py`, and the architecture-scanner allowances, each with tests. The determinism test runs a LightGBM fit twice and checks that the booster dumps are byte-identical.

### Task 1: Evaluation harness (wave A)
- [ ] Outcomes per player-fixture from `as_of(lockdown + 1 µs)`, the backtester's access path, re-scored under `backtest_rules`.
- [ ] The metrics above; Diebold-Mariano clustered by GW; reliability tables; captain/XI regret.
- [ ] `fplopt models eval --models v1,rolling,ep_next --seasons … [--jobs N]`.
- [ ] Baseline numbers for `rolling` / `ep_next` on develop and validate go into PLAN §6.

### Task 2: Team model (wave A)
- [ ] De-vig (power method, with Shin as the alternative), Dixon-Coles λ from 1X2 + O/U (+ AH), ρ fitted once on develop.
- [ ] Ratings with time decay and Elo priors; the stats fallback.
- [ ] Metrics: team-goals Poisson log-likelihood and P(CS) Brier vs the Elo-only reference and vs the market λ on fixtures with odds.
- [ ] Task 2 also tunes the half-life and the market/stats blend weight on develop.

### Task 3: Minutes and availability (wave A)
- [ ] Start inference, with its measured accuracy.
- [ ] History features; the hurdle model; post-processing.
- [ ] The flag mapping and news parsing.
- [ ] Minutes metrics vs the last-5 reference, both without flags (all seasons) and with flags (2021/22+).

### Task 4: Shares and penalties (wave B; needs Tasks 2–3)
- [ ] The npxG source check: Understat vs FPL xG minus penalty xG on 2022/23–2024/25, with the result in PLAN §3 (closes that VERIFY).
- [ ] Shares, the Marcel prior, team consistency, the FPL-assist fraction, penalties.
- [ ] Goals/assists log-likelihood vs a position-average reference.

### Task 5: Other components, assembly, calibration (wave C; needs Task 4)
- [ ] Bonus, saves, cards/OG/pen-miss; per-fixture assembly; calibration fit on develop.
- [ ] `MODELS["v1"]` registered. `fplopt check leakage` passes with `model:v1` (and `pytest -m realdata`).

### Task 6: Tuning and criterion 1 (lead, after Task 5)
- [ ] Develop tuning: refit cadence, decay half-lives, shrinkage strengths, LightGBM size.
- [ ] One validate run. Results go in PLAN §6 (a *Phase 5 results* block like §7's) and §12.
- [ ] If criterion 1 fails, stop and report before 5b.

### Task 7: Review and PR 5a (lead)
- [ ] An independent review (`/code-review`), then a PR with CI and CodeRabbit, merge with the user's go-ahead, and a server update.

### Task 8: Minutes in the optimizer (5b, wave A)
- [ ] Bench weights from P(starter doesn't play) and autosub order (PLAN §7 *Bench*).
- [ ] The expected-minutes pruning floor (§7 *Pruning*); pruning loss re-measured with `optimize bench`.

### Task 9: Pre-register the optimizer-vs-greedy test (5b, lead, with the user)
- [ ] Settle §11's per-decision design (continuation, reference) and the start set, and log them in §12 **before** any 5b comparison runs.
- [ ] Proposed:
  - The parameters (horizon, decay and FT value jointly; `max_hits` / `hit_margin`) are chosen on develop.
  - Validate confirms on both:
    - the full run (`template@1,random:3@1,random:3@20`): 80% CI lower bound > 0;
    - per decision with each arm's own continuation, at greedy's states: 80% CI lower bound > 0.
  - Plus: the same sign on xG, and a deflated mean > 0 over all seasons.

**Pre-registered 2026-10-08 (approved by the user before any 5b comparison):**
- **Arms:** A = `optimizer:v1`, B = `greedy:v1`. Same xP (v1), no chips. Bench weights and the minutes-based pruning floor are those Task 8 fixes, decided before any comparison below.
- **Starts:** `template@1,random:3@1,random:3@20`.
- **Selection on develop** (2017/18–2022/23; 2016/17 is left out, since v1 has no earlier data there):
  - greedy's `threshold` ∈ {0.5, 1.0, 2.0}, chosen by greedy's full-run paired difference vs `greedy:v1` defaults;
  - then the optimizer vs the chosen greedy over `max_hits` ∈ {0, 1 with `hit_margin` 2} × `horizon` ∈ {4, 6} × `decay` ∈ {0.75, 0.85};
  - each run `backtest compare --continuation own --reference b`, with family `optimizer:v1 vs greedy:v1 develop`;
  - the variant with the largest full-run realized mean per GW wins;
  - N = the number of optimizer variants run.
- **Confirmation on validate, one run:** `backtest compare --a optimizer:v1:<chosen> --b greedy:v1:<chosen> --seasons 2023-2024 --starts template@1,random:3@1,random:3@20 --continuation own --reference b`.
- **Passes if all three hold:**
  1. the full-run realized paired mean per GW is > 0, with the 80% GW-block-bootstrap CI's lower bound > 0;
  2. the xG-scored full-run mean is > 0 (same sign);
  3. the deflated validate mean, mean − SE·√(2 ln N), is > 0.

  The per-decision result (own continuation, B's states, k = 4) is reported but does not gate (two seasons are too few for it, Phase 4).

**Procedure amendment (2026-10-09, user-approved, before any selection run):** the develop selection runs use `--no-per-decision`. Selection is decided on the full-run mean alone, which per-decision windows cannot change; the validate run keeps the per-decision report. The runs are split between the local machine and the server, on the same commit and an identical copy of `data/`, checked first on one case with identical results.

### Task 10: Defcon (5b, wave A)
- [ ] #14: check that FPL-Core-Insights reproduces FPL CBIT/CBIRT on 2026/27 GW1–5, then set `k` and `r` once from its 2024/25 data (research use only, never shipped).
- [ ] The defcon component, active only for 2025/26+ rules, so develop/validate xP is unchanged. Close #14.

### Task 11: Criterion 2, review, PR 5b (lead)
- [ ] Develop tuning runs, then the single validate confirmation. Results in PLAN §7 and §12, and the §9 row marked done.
- [ ] Review, PR, merge with the go-ahead, server update.

---

## Waves

| Wave | Tasks | Notes |
|---|---|---|
| 5a-0 | 0 | Lead; everything else builds on the protocol |
| 5a-A | 1, 2, 3 | Independent; parallel implementers |
| 5a-B | 4 | Needs team λ and expected minutes |
| 5a-C | 5, 6, 7 | Assembly, then tuning (long runs), then review |
| 5b | 8, 10 in parallel; 9 before any comparison; then 11 | 9 needs the user |

Run-time budget: `models eval` over 2016/17–2024/25 at `--jobs 14` ≤ ~20 min; the leakage check with `v1` ≤ ~20 min.
