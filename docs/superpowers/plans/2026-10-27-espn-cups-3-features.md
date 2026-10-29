# ESPN cup data, PR 3 (congestion and cup-minutes features) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the v1 minutes model what it could not see: fixture congestion (cup and European matches around a PL fixture) and each player's own cup/European minutes and starts.

**Architecture:** Spec `docs/superpowers/specs/2026-10-27-espn-cup-data-design.md` §3. PRs 1–2 (merged) give `club_match` (every non-league match of a PL club with `available_at` = when it was known), `espn_player_match` (minutes, starter) and `espn_map`. This PR adds six columns to `fplopt.features.history.LAG_FEATURES`, computed for training rows (as of each row's GW deadline, exactly like the existing lags) and prediction rows (as of the view's deadline; the schedule features per horizon fixture), then refits the calibration table and validates like PR 2 (#50).

**Tech Stack:** pandas, numpy, LightGBM via `fplopt.models.gbm`; `dev/calibrate_v1.py`, `fplopt models eval`, `fplopt check leakage`.

---

## The features

All read through the view only (`club_match`, `espn_player_match`, `espn_map`); "known" means `available_at` < the reference time, "played" means kickoff < the reference time. Reference time: a training row's GW deadline (`deadline_time`, the same `ref_time` the other lags use); the view's deadline for prediction rows.

| feature | per | definition | cap / missing |
|---|---|---|---|
| `rest_days_all` | fixture | days from the club's previous match in any competition (PL fixture or played/known `club_match`) to this fixture's kickoff | cap 30; 30 without one |
| `days_to_next` | fixture | days from this fixture's kickoff to the club's next known match in any competition (PL fixture from the schedule, or known `club_match`) | cap 30; 30 without one |
| `europe_week` | fixture | 1 if the club has a known UEFA `club_match` (`uefa.*`) within 4 days before or after the kickoff, else 0 | — |
| `cup_minutes_7d` | player | his minutes in played non-league matches (mapped ESPN rows, `league != "eng.1"`) in the 7 days before the reference time | 0 without |
| `cup_started_last` | player | over his non-league ESPN rows since his last PL row before the reference time: 1 if he started the latest, 0 if he was in the squad and did not start; NaN if none | NaN |
| `days_since_app` | player | (changed) also counts non-league appearances (minutes > 0) before the reference time | as now |

`rest_days` (league-only) stays. Training/prediction consistency is the invariant: a training row's features equal what `prediction_frame` would compute at that row's GW deadline for that fixture (the existing docstring's "training and prediction features match by construction").

Leakage: the schedule features use only `club_match` rows known before the reference time; the player features only ESPN rows with `available_at` < reference time (kickoff + 3 h). `fplopt check leakage` covers them (corrupting future rows must not change any value).

## File map

- Modify `src/fplopt/features/history.py`: `LAG_FEATURES`, the docstring's feature list (and remove "European/cup matches" from *Dropped*), `_lags` (`days_since_app`), training/prediction frame plumbing; new helpers for the schedule and player features (or a new sibling module `fplopt/features/congestion.py` if history.py grows past ~900 lines — follow the existing file's patterns).
- Modify `tests/test_features_history.py` (+ `tests/synthetic_season.py` helpers if needed).
- Regenerated `src/fplopt/models/calibration_table.py`.
- Docs: PLAN §6.3 features line, `history.py` docstring.

### Task 1: schedule features (`rest_days_all`, `days_to_next`, `europe_week`)

- [ ] Failing tests (view-level, synthetic tables with planted `club_match` rows): a UEFA match 3 days before a PL fixture → `rest_days_all` 3, `europe_week` 1; a cup match 2 days after → `days_to_next` 2; a cup match not yet known at the reference time (available_at after it) → ignored (training and prediction); without ESPN data `rest_days_all == rest_days`, `days_to_next` = days to the next PL fixture, `europe_week` 0.
- [ ] Implement for training rows (per row: club timeline known at its GW deadline) and prediction rows (per horizon fixture, known at the view deadline). Vectorise with `merge_asof`/searchsorted per club; no per-row Python loops over all rows.
- [ ] Commit `feat(features): congestion — rest days and next match in any competition, a European week`.

### Task 2: player features (`cup_minutes_7d`, `cup_started_last`, `days_since_app`)

- [ ] Failing tests: a player with 90' in an EFL Cup match 3 days before a GW deadline → `cup_minutes_7d` 90 (and 0 for a match 8 days before); started the cup tie after his last PL row → `cup_started_last` 1, on the bench → 0, no cup match since → NaN; a player injured for 4 PL rows who played 60' in a cup match before the deadline → `days_since_app` counts from the cup match; without ESPN data all equal to today's values (days_since_app unchanged).
- [ ] Implement (ESPN rows mapped through `view.lookup("espn_map", ...)`; NA minutes count as 0 for `cup_minutes_7d` and are skipped for appearances).
- [ ] Commit `feat(features): cup and European minutes and starts`.

### Task 3: verification and timing

- [ ] `uv run pytest tests/test_features_history.py tests/test_features_architecture.py tests/test_models_minutes.py tests/test_models_assemble.py -q`, then the full suite (check the summary line), `ruff`.
- [ ] Report `prediction_frame` time on the synthetic tables before/after.

### Task 4 (controller, on the server): refit and validation

- [ ] Server checkout of the branch sharing the live `raw/`; `fplopt build all` into a scratch `data/`; a main checkout reading the same `data/`.
- [ ] `dev/calibrate_v1.py walk --seasons 2016-2024` for both; paired walk comparison (P(start) Brier, minutes log loss) overall and for rows with `europe_week` 1 or `cup_minutes_7d` > 0; `fit --parts p_start:1+ --write`; commit the table with develop MSE.
- [ ] Feature importance (gain) of the new features in the start/sixty/sub boosters at a recent cutoff.
- [ ] `fplopt check leakage --deadlines 12`; `models eval --models v1` on validate (2023–2024) and live (2026) vs main, paired (MSE h0, h1–5, candidates, captain regret). Gate as #47/#50: validate MSE not worse, captain regret not significantly worse.
- [ ] PR; merge and deploy only on the user's go-ahead, after GW9's deadline (2026-10-31 11:00 UTC) unless the user says otherwise.
