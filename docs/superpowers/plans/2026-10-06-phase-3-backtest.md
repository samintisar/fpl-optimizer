# Phase 3: Backtester + Baselines + Greedy Policy + Paired Evaluation — Implementation Plan

> **For agentic workers:** Implement task-by-task (one implementer per task or wave, then an independent review). Checkbox (`- [ ]`) steps. TDD: write failing tests first. Pure modules come with exhaustive unit tests; anything touching `data/` is also checked on the real tables (`-m realdata`).

**Goal:** Replay any develop/validate season gameweek by gameweek, starting from any squad state. Each policy sees only `as_of(deadline)` data. Every gameweek is scored with actual outcomes, re-scored under the backtest rules (PLAN §5 table). Two baseline xP models (rolling average and FPL `ep_next`), a roll policy and the greedy policy. Paired evaluation: full-run paired differences and per-decision differences, each with a GW-block bootstrap clustered by season, plus the xG-scored secondary metric.

Phase 3 is done when (PLAN §9): seasons can be simulated from arbitrary states, and the baselines are scored per §5 on real data with paired comparisons reported.

**Spec:** `docs/PLAN.md` §3 (rules config, tables), §4 (as-of semantics), §5 (backtester), §7 (FT/selling/chip rules shared with the optimizer), §9.

**Branch:** `phase-3-backtest`.

**Holdout:** 2025/26 is untouched. `simulate` refuses seasons in `HOLDOUT_SEASONS` unless `allow_holdout=True`. Nothing in Phase 3 passes `allow_holdout=True`, and there is no CLI flag for it until Phase 6. The scorer is validated on 2016/17–2024/25 (legacy rules) and on 2026/27 GW1–5 (native rules, including defcon), never on 2025/26.

---

## Decisions (record in PLAN §5/§12 in Task 0)

- **Reading outcomes.** The simulator scores GW t from `store.as_of(lockdown_t + 1 µs)` (outcomes are available at lockdown, strict `<`). Policies only ever get the deadline view. State transitions never use outcomes, so reading outcomes cannot leak into decisions.
- **Rules per season** (`backtest_rules(season)`):
  - 2016/17–2024/25: the current rules (`2026-27`) with defcon off.
  - 2025/26: native `2025-26`, holdout only.
  - 2026/27: native `2026-27`.
  - Chips: develop/validate seasons use the 2026/27 chip windows, applied to **`gw_index`** (1–19 / 20–38). This is because 2019/20 numbers its GWs 39–47 and 2022/23 has no GW7.
- **Hand-maintained supplement** (`config/scoring/<season>.supplement.json`, next to the generated files; PLAN §3) for what the API lacks:
  - Thresholds: 60 minutes for long play, 1 point per 3 saves, −1 per 2 goals conceded.
  - Defcon thresholds: DEF 10 CBIT, MID/FWD 12 CBIRT, GK not eligible.
  - Hit cost 4.
  - WC/FH effect on banked FTs (`chip_week_ft`: `"retain_plus_one"` or `"retain"`; VERIFY in Task 12).
  - Free Hit in consecutive GWs not allowed.
  - FT top-ups (2025/26 AFCON: empty until #9 verifies GW/amount).
- **GW1** (`gw_index == 1`): transfers are unlimited and free (pre-season), and the next GW starts with 1 FT. Everywhere else, FTs bank up to `1 + max_extra_free_transfers` (5). Each transfer beyond the FTs costs a hit, except under WC/FH.
- **Money** is int tenths of £m. Selling price = purchase price + floor((current − purchase) × `transfers_sell_on_fee`) when current > purchase, else current. A held player missing from the pool (left the game) keeps his last known price and club, can be sold, and cannot be bought back.
- **"Played"** (for autosubs and captaincy) = minutes > 0, summed over the GW's fixtures. A blank counts as did-not-play. Autosubs follow FPL: the bench GK can only replace the GK. Outfield bench players, in order, each replace the first non-playing outfield starter for whom the formation stays valid (≥3 DEF, ≥2 MID, ≥1 FWD). Bench Boost: all 15 count and there are no autosubs. If the captain didn't play, the vice gets the multiplier (if the vice played). Triple Captain: ×3.
- **Club cap** (≤ 3 per club) is checked after transfers, with clubs as of the deadline. A club already over the cap without any transfer in (a player you hold moved club) is allowed, as long as the transfers don't add to it.
- **xP frames** (models and policies): one row per pool player and GW for the target GW and the next `HORIZON` (5) GWs by `gw_index`, i.e. 6 GWs, the optimizer default. Columns: `player_key, gw, gw_index, horizon (0 = target), xp (float64, 0 when unknown)`. Sorted by (player_key, horizon).
- **Start states:**
  - **template** = the most-owned valid squad, using ownership visible at the deadline (the newest snapshot `selected_by_percent`, else the newest visible `player_gw_ownership.selected`). Before 2021/22 nothing is visible at GW1, so template starts begin at GW2 there.
  - **random** = a seeded random valid squad from the pool, sampled with weight ∝ price so it spends realistically.
  - Both are refused at deadlines where `pool_coverage` shows a club with a horizon fixture and 0 pool players (only 2020/21 GW1).
- **Per-decision evaluation:** states come from a reference trajectory (default: policy B's own run). At each GW, arm A applies A's decision and arm B applies B's. Both then continue for `k − 1` GWs with a fixed continuation (roll policy on a fixed xP model). Scored over GWs t..t+k−1 including hits at t; default k = 4.
- **Bootstrap:** average the paired differences over start states per (season, gw_index). Then a circular moving-block bootstrap of GW blocks (default length 4) within each season, with every season resampled together. Reports the mean, a percentile CI, and a one-sided p-value (fraction of bootstrap means ≤ 0) with H1: A > B.
- **xG-scored points** (§5): realized minutes. Goals become xG × goal points and assists become xA × 3. For players with ≥ 60 minutes, the clean sheet becomes P(0 conceded) × CS points, with conceded ~ Poisson(opponent team xG × minutes/90). Goals conceded (GK/DEF) become −E[floor(G/2)] under the same Poisson. Everything else is realized (appearance, saves, pens, cards, own goals, bonus, defcon).
  - Sources: player xG/xA from Understat (`us_xg`, `us_xa`), else FPL (`fpl_xg`, `fpl_xa`); team xG from `team_match` Understat, else FPL-summed, else football-data.
  - A row is null if any input it needs is missing. A GW's xG score is null if any counted player's is. Comparisons use only GWs where both arms are non-null.
- **Experiments:** `results/experiments.csv`, appended by every CLI backtest run (timestamp, git sha, command/config JSON, metrics JSON, `n_variants`). Tracked in git. The rest of `results/` is gitignored.

---

## File structure

| File | Responsibility |
|---|---|
| `src/fplopt/build/players.py` | #24: `player_match.available_at = max(lockdown, player_season.available_at)` |
| `config/scoring/2025-26.supplement.json`, `2026-27.supplement.json` | Hand-maintained rules supplement |
| `src/fplopt/backtest/rules.py` | `Rules`, `ChipWindow`, `load_rules`, `backtest_rules` |
| `src/fplopt/backtest/scoring.py` | `score_matches` (per player-match points by component) |
| `src/fplopt/backtest/gw_score.py` | `Lineup`, `GwScore`, `score_gameweek` (autosubs, captaincy, chips) |
| `src/fplopt/backtest/state.py` | `Holding`, `SquadState`, `Transfer`, `Decision`, `selling_price`, `apply_decision`, `next_state`, validation |
| `src/fplopt/models/baseline.py`, `models/__init__.py` | `xp_rolling`, `xp_ep_next`; `MODELS` registry |
| `src/fplopt/backtest/policies.py` | `DecisionContext`, `Policy`, `best_lineup`, `RollPolicy`, `GreedyPolicy` |
| `src/fplopt/backtest/start_states.py` | `template_state`, `random_state`, `StartStateError` |
| `src/fplopt/backtest/probes.py` | Decision probes for the leakage check |
| `src/fplopt/backtest/simulator.py` | `simulate`, `SeasonRun`, outcome + xP caches |
| `src/fplopt/backtest/xg_points.py` | xG-scored points |
| `src/fplopt/backtest/evaluate.py` | Paired full-run diffs, per-decision evaluation, block bootstrap, experiment log |
| `src/fplopt/features/leakcheck.py` | Also checks `MODELS` and decision probes |
| `src/fplopt/cli.py` | `fplopt backtest run`, `fplopt backtest compare` |
| tests | `tests/synthetic_season.py` (in-memory tables for backtests), `test_backtest_*.py`, `test_models_baseline.py`, architecture test extended to `fplopt/models` and `fplopt/backtest/policies.py` |

---

### Task 0: PLAN + plan commit (lead)

- [ ] Add the Decisions above to PLAN §5 (Simulator / Scoring / Comparing policies), the supplement file to §3 *Rules config*, and a §12 decision-log row ("chip windows on gw_index for develop seasons; outcomes read at lockdown; per-decision reference trajectory = baseline's run"). Commit with this plan.

---

### Task 1: #24, player_match availability (wave A)

**Files:** `src/fplopt/build/players.py` (and/or wherever `player_match.available_at` is set), `tests/test_build_players.py`.

- [ ] Failing test (synthetic build): a player whose `player_season.available_at` is after a GW's lockdown has `player_match` rows for that GW with `available_at == player_season.available_at`; everyone else keeps the lockdown.
- [ ] Implement `available_at = max(lockdown, player_season.available_at)`, joined on (player_key, season). `event_time` stays the kickoff.
- [ ] Measure on real data (`uv run fplopt build player_match`, or `all` if the dependency order needs it): rows changed per season, and the max/median delay. Report the numbers in the PR and close #24 with them.
- [ ] `uv run pytest`, `uv run pytest -m realdata -k leak` (or `fplopt check leakage --deadlines 4`) still pass.

### Task 2: Rules, match scoring, GW scoring (wave A)

**Files:** `config/scoring/*.supplement.json`, `src/fplopt/backtest/rules.py`, `scoring.py`, `gw_score.py`, `tests/test_backtest_rules.py`, `test_backtest_scoring.py`, `test_backtest_gw_score.py`.

- [ ] `Rules` (frozen dataclass; mappings are read-only: `MappingProxyType` or tuples):
  - per position (element_type 1–4): `goals_scored`, `clean_sheets`, `goals_conceded`, `defensive_contribution`, `defcon_threshold` (None for GK);
  - scalars: `assists`, `saves`, `penalties_saved`, `penalties_missed`, `yellow_cards`, `red_cards`, `own_goals`, `bonus`, `short_play`, `long_play`, `long_play_minutes`, `saves_per_point`, `goals_conceded_per_point`, `defcon_enabled`;
  - squad: `squad_size` 15, `squad_select` {1:2, 2:5, 3:5, 4:3}, `play_min` / `play_max` (from `element_types` squad_min_play/squad_max_play), `squad_play` 11, `team_limit` 3, `budget` 1000, `sell_on_fee` 0.5, `max_free_transfers` (1 + `max_extra_free_transfers`), `hit_cost`;
  - chips: `chips: tuple[ChipWindow(chip_id, name, start, stop)]` (name ∈ wildcard/freehit/bboost/3xc), `chip_week_ft`, `freehit_consecutive`, `ft_topups: tuple[(gw_index, amount)]`;
  - `label` (e.g. `"2026-27"` or `"2026-27-nodefcon"`).
  - Map the API position names GKP/DEF/MID/FWD to 1–4.
- [ ] `load_rules(label, *, defcon=None, config_dir=...)` reads `<label>.json` + `<label>.supplement.json` and fails loudly on missing keys. `backtest_rules(season)` follows the Decisions. `legacy_rules()` = 2026-27 with GK goals 6 and defcon off (FPL scoring 2016/17–2024/25), used only to validate the scorer.
- [ ] `score_matches(matches, rules) -> DataFrame` (same index):
  - **Inputs:** `element_type`, `minutes`, `goals_scored`, `assists`, `clean_sheets`, `goals_conceded`, `own_goals`, `penalties_saved`, `penalties_missed`, `yellow_cards`, `red_cards`, `saves`, `bonus`, plus optionally `defensive_contribution`, `clearances_blocks_interceptions`, `tackles`, `recoveries`.
  - **Output:** int64 columns `appearance, goals, assists, clean_sheets, goals_conceded, saves, penalties_saved, penalties_missed, yellow_cards, red_cards, own_goals, bonus, defcon, points`.
  - **Clean sheets:** use the `clean_sheets` column as is (FPL already applies the 60-minute rule).
  - **Defcon:** only when `defcon_enabled`. Use FPL's `defensive_contribution` count when non-null. Otherwise DEF: CBI + tackles, MID/FWD: CBI + tackles + recoveries. A missing component gives 0 points.
- [ ] Unit tests for every component, every position, thresholds, and defcon on/off.
- [ ] Real-data test (`@pytest.mark.realdata`):
  - `score_matches(..., legacy_rules())` reproduces `total_points` for 2016/17–2024/25. Report the match rate per season and assert ≥ 99%; the PR lists the mismatch patterns.
  - `backtest_rules(2026)` reproduces 2026/27 rows. Assert ≥ 99%; investigate any mismatch.
  - Never read 2025/26 rows (filter `season != 2025` before scoring).
- [ ] `gw_score.py`:
  - `Lineup(starters: tuple[int, ...] (11), bench: tuple[int, ...] (4, bench[0] = GK), captain, vice)`, with `validate(lineup, positions, rules)` (formation, distinct players, captain ≠ vice, both starters).
  - `score_gameweek(lineup, outcomes: Mapping[player_key → (points: int, minutes: int)], positions, rules, chip=None) -> GwScore(points, counted: tuple, multipliers: Mapping, captain_used, autosubs: tuple[(out, in)], bench_points)`. Players missing from `outcomes` = 0 points, 0 minutes.
  - Generic over the points (realized or xG-scored floats): pass `points` as float and keep it exact.
- [ ] Autosub tests (each a separate case):
  - GK swap;
  - outfield order;
  - a formation-blocked sub, where the next bench player is used;
  - a bench player who didn't play is skipped;
  - captain didn't play → vice; neither played → no multiplier;
  - TC; BB (no autosubs, all 15 count);
  - a double GW summing two fixtures;
  - a blank = didn't play.

### Task 3: Squad state and transfers (wave A)

**Files:** `src/fplopt/backtest/state.py`, `tests/test_backtest_state.py`. Pure: no I/O, no pandas beyond reading the pool frame.

- [ ] Data types:
  - `Holding(player_key, element_type, team_key, purchase_price, price)`, where `price` is the latest known current price.
  - `SquadState(season, gw_index, holdings: tuple[Holding] sorted by key, bank, free_transfers, chips_used: tuple[(chip_id, gw_index)], freehit_backup: tuple[Holding] | None, freehit_bank: int | None)`. Keep immutable `replace`-style updates.
  - `Transfer(out_key, in_key)`; `Decision(transfers: tuple[Transfer], lineup: Lineup, chip: str | None)`.
- [ ] `selling_price(holding)`, `squad_value(state)`.
- [ ] `refresh(state, pool)` updates held players' `team_key`/`price` from the pool (`player_key, element_type, team_key, price`); players absent from the pool keep theirs.
- [ ] `apply_decision(state, decision, pool, rules) -> (gw_state, TransferRecord(n_transfers, free_used, hits, hit_points, chip))`. It validates in this order:
  1. chip available: there is a window containing `gw_index` with that name whose `chip_id` is unused, and at most one chip per GW. A Free Hit right after a Free Hit is rejected unless `freehit_consecutive`;
  2. outs held, ins in pool and not held, no duplicates;
  3. final position counts = `squad_select`;
  4. club cap (see Decisions);
  5. budget ≥ 0 after selling at selling price and buying at pool price;
  6. lineup valid for the post-transfer squad.

  Purchase price of an incoming player = pool price. FH stores a backup of the pre-transfer holdings and bank. Raise `InvalidDecision` with a clear message.
- [ ] `next_state(gw_state, record, rules, next_gw_index) -> SquadState`:
  - FT accrual: GW1 → 1; under WC/FH per `chip_week_ft`; otherwise `min(cap, max(ft − n, 0) + 1)`; plus top-ups; capped.
  - FH revert: restore the backup holdings, with `price`/`team_key` refreshed later by the next `refresh`, and the backup bank.
  - Record the chip in `chips_used`.
- [ ] Tests:
  - FT banking to 5 and capping; hits;
  - GW1 unlimited;
  - WC and FH FT behaviour for both `chip_week_ft` values;
  - FH revert of squad and bank; FH consecutive rule;
  - chip windows (set 1 expires at 19, set 2 from 20; a set-1 chip unused is not usable at 20);
  - selling price rounding (e.g. bought 50, now 53 → 51; now 48 → 48);
  - a departed player can be sold but not bought;
  - club cap including a pre-existing violation;
  - budget failures.

### Task 4: Baseline xP models (wave A)

**Files:** `src/fplopt/models/baseline.py`, `src/fplopt/models/__init__.py`, `tests/test_models_baseline.py`, `tests/test_features_architecture.py` (extend), `src/fplopt/features/leakcheck.py` (register).

- [ ] `xp_rolling(view)`. Per pool player (`player_pool(view)`), the per-fixture rate is the mean `total_points` over his last `ROLLING_N = 5` visible `player_match` rows (any season, ordered by kickoff, 0-minute rows included). Players without rows get rate 0. xP per horizon GW = rate × his club's `n_fixtures` in that GW (`upcoming_fixtures`, club = pool `team_key`).
- [ ] `xp_ep_next(view)`. Target GW: xP = `ep_next` (null → 0). For later horizon GWs, the per-fixture rate = `ep_next / n_fixtures_target` when the target isn't a blank, else the snapshot `form` (null → 0); xP = rate × n_fixtures. Without any snapshot (pre-2021), every xP is 0 and the model logs a warning once per call (the CLI refuses `ep_next` for seasons before 2021). `form` comes from `player_snapshot` via the view: add it to a feature (e.g. extend `ep_next` with `form`) or read `view.latest` directly; models may read the view.
- [ ] `MODELS: dict[str, Callable[[AsOfView], DataFrame]] = {"rolling": xp_rolling, "ep_next": xp_ep_next}`. Output dtypes and sort order are fixed per the Decisions.
- [ ] Architecture: the static scan now also covers `fplopt/models/`. Same rules; its allowlist adds `fplopt.features` (and its submodules except `store`/`leakcheck`; from the store only `AsOfView`). `MODELS` is the allowed registry. Generalize the scanner rather than duplicating it, and keep every existing test passing.
- [ ] Leakage: `check_leakage` / `run_leakage_check` check `FEATURES | MODELS` (names prefixed `model:`) by default. Synthetic leakage tests cover the models.
- [ ] Tests: synthetic in-memory tables (see `tests/synthetic_season.py`, created in this task if not already there):
  - rolling over a season boundary, 0-minute rows, doubles ×2 and blanks 0;
  - ep_next with a blank target, and no snapshot;
  - dtypes and sort order.

`tests/synthetic_season.py` (shared fixture for waves A–C): a function `synthetic_tables(seasons=(2023,), n_clubs=20, players_per_club=(2, 5, 5, 3), seed=0, snapshots=True)` returning an in-memory `{name: DataFrame}` for `DataStore(tables=...)`.
- It covers `gameweek` (38 GWs, deadlines, lockdowns, `gw_index`), `schedule`/`fixture` (round robin), `player_season`, `player_gw` (value), `player_gw_ownership`, `player_match` (minutes/goals/etc. from a seeded generator, `total_points` consistent with `legacy_rules`), `player_snapshot` (one snapshot per GW at deadline − 2h: price, status, `ep_next`, `form`, `selected_by_percent`), `team_rating`, `odds_snapshot` (may be empty), `fixture_snapshot` (empty) and `team_match`.
- Options to add a blank GW (one club's fixture moved) and a double GW.
- `available_at` follows PLAN §4 exactly; reuse the Phase 2 conventions.

Wave A agents coordinate on this file: **Task 4 owns it**. Tasks 2/3 use plain hand-made frames in their unit tests.

### Task 5: Policies and start states (wave B; needs Tasks 2–4)

**Files:** `src/fplopt/backtest/policies.py`, `start_states.py`, `probes.py`, tests, architecture test (extend to `policies.py`, `start_states.py`, `probes.py`: same no-file-read rules; they may import `fplopt.features`, `fplopt.models`, `fplopt.backtest.{rules,state,gw_score}`).

- [ ] `DecisionContext(view, state, rules, pool, xp)` (frozen). `Policy` protocol: `name: str`, `xp_model: str` (a `MODELS` key), `decide(ctx) -> Decision`. Policies are immutable parameter holders and keep no state between calls.
- [ ] `best_lineup(squad: DataFrame[player_key, element_type], xp_target: Mapping, rules) -> Lineup`. GK = best GK by xP. Fill the formation minimums by xP, then the remaining slots with the best of the rest under the maximums (optimal for these constraints). Bench: the other GK first, then outfield by xP descending. Captain = top starter by xP, vice = second. Ties broken by player_key ascending.
- [ ] `RollPolicy(xp_model)`: no transfers; `best_lineup` on target-GW xP.
- [ ] `GreedyPolicy(xp_model, threshold=1.0, horizon=6, decay=0.85, max_transfers=1)`:
  - hxp = Σ_h decay^h · xp_h over h < horizon.
  - Candidates: (out i, in j) with the same element_type, j in the pool and not held, price_j ≤ bank + selling_price(i), club cap OK.
  - Gain = hxp_j − hxp_i. Pick the max gain (ties by in_key, then out_key).
  - Make it if gain > threshold and a free transfer is available (no hits ever). Repeat up to `max_transfers` while FTs remain. At `gw_index == 1` repeat while the gain exceeds the threshold (up to 15).
  - Then `best_lineup` on the post-transfer squad. No chips.
- [ ] `template_state(view, rules)` and `random_state(view, rules, seed)` → `SquadState`:
  - template: add players in descending ownership if their position has a slot, the club cap holds, and the remaining budget can still fill the remaining slots with the cheapest eligible players.
  - random: random slot order; sample feasible players with weight ∝ price (`numpy.random.default_rng(seed)`).
  - `free_transfers` = 1 (gw_index > 1) or 0 (GW1, unlimited anyway); `bank` = budget − cost; purchase price = price.
  - `StartStateError` if no ownership is visible (template), if the pool can't form a valid squad, or if the pool coverage has a gap.
- [ ] `probes.py`: `PROBES: dict[str, Callable[[AsOfView], DataFrame]]`, e.g. `greedy_rolling_random0` = greedy(rolling) decision from `random_state(view, seed=0)` with `backtest_rules(season)`, encoded as a DataFrame (transfers, lineup slots, captain/vice, chip). Register them in the leakage check (prefix `probe:`). Probes are skipped (and noted) when a start state is refused. Both template and random probes are included.
- [ ] Tests:
  - `best_lineup` optimality: brute force over all valid XIs on random small squads;
  - greedy picks the max-gain affordable transfer; respects the threshold, FTs, budget and club cap; tie-breaking;
  - roll never transfers;
  - start states are valid, deterministic per seed, budget-respecting, and refused on a coverage gap / without ownership;
  - the probes pass the synthetic leakage check.

### Task 6: Simulator (wave C; needs Task 5)

**Files:** `src/fplopt/backtest/simulator.py`, `tests/test_backtest_simulator.py`.

- [ ] `simulate(store, rules, policy, start, *, end_gw_index=None, caches=None, allow_holdout=False) -> SeasonRun`. For each GW from `start.gw_index` to the season's last (or `end_gw_index`):
  1. `view = store.as_of(deadline)`;
  2. `pool = player_pool(view)`;
  3. `state = refresh(state, pool)`;
  4. `xp = caches.xp(policy.xp_model, view)`;
  5. `decision = policy.decide(ctx)`;
  6. `apply_decision`;
  7. outcomes for (season, gw) from `store.as_of(lockdown + 1 µs)`: `player_match` rows of that GW, joined to `player_season.element_type`, `score_matches(rules)` → per player (points, minutes);
  8. `score_gameweek`;
  9. record;
  10. `next_state`.
- [ ] `SeasonRun`:
  - `gws` DataFrame, one row per GW: season, gw, gw_index, deadline, chip, n_transfers, hits, hit_points, points (incl. captain), net_points (points − hit_points), bench_points, captain, vice, captain_used, captain_points, captain_regret, xi_regret, bank, free_transfers (before the decision), squad_value, transfers (JSON string), autosubs (JSON string), plus later `xg_points`;
  - `decisions`, `final_state`, `total` (Σ net_points).
  - captain_regret = max realized points among counted players − captain's realized points (raw, before the multiplier). xi_regret = best valid XI from the squad in hindsight (realized points, + best captain) − actual points, for no-chip GWs.
- [ ] `Caches`: xP per (model, deadline) and outcomes per (season, gw, rules.label). Simulator objects own them; models and policies never cache.
- [ ] Holdout guard: raises `HoldoutError` for a season in `HOLDOUT_SEASONS` unless `allow_holdout`.
- [ ] Tests on `synthetic_tables`:
  - a full season with roll and with greedy is deterministic (two runs are identical);
  - totals equal Σ GW points; FTs and bank evolve as `state` says;
  - a forced chip policy (test-only) exercises WC/FH/BB/TC end to end;
  - a mid-season start; `end_gw_index`; the holdout guard;
  - outcomes after the deadline don't change decisions (corrupt the future at deadline t and assert decisions up to t are identical; the scores of later GWs may differ);
  - double/blank GWs score correctly.

### Task 7: xG-scored points (wave C; needs Task 2)

**Files:** `src/fplopt/backtest/xg_points.py`, tests; wire into the simulator as an extra per-GW column `xg_points` (null where not computable).

- [ ] `xg_score_matches(matches, team_xg, rules) -> Series[float]` per the Decisions, with a `source` column (`understat`/`fpl`/`football-data`/null). Poisson terms in closed form (scipy.stats or explicit sums up to 15 goals).
- [ ] The simulator calls `score_gameweek` a second time with the xG points (same minutes, so autosubs are the same).
- [ ] Tests: hand-computed cases (CS probability, E[floor(G/2)]), null propagation, and source precedence. Real-data test: coverage per season (share of non-null player-GWs among players with minutes) is reported; ≥ 95% for 2019/20–2023/24.

### Task 8: Evaluation (wave C; needs Task 6)

**Files:** `src/fplopt/backtest/evaluate.py`, tests.

- [ ] `run_grid(store, seasons, start_specs, policies, rules_fn=backtest_rules) -> DataFrame` with columns `season, start_id, policy, gw_index, net_points, xg_points, ...`. It shares `Caches` across policies and start states.
  - `start_specs` are parsed from strings: `template@1`, `random:5@1` (seeds 0–4), `random:3@20`.
  - Refused start states are skipped and logged; template@1 falls back to @2 for pre-2021 seasons with a logged note.
- [ ] `paired_full_run(results, a, b) -> DataFrame[season, start_id, gw_index, diff, diff_xg]`.
- [ ] `per_decision(store, seasons, start_specs, policy_a, policy_b, continuation, k=4, reference="b") -> DataFrame[season, start_id, gw_index, points_a, points_b, diff, diff_xg]`.
- [ ] `block_bootstrap(diffs, value="diff", block_length=4, n_boot=2000, seed=0) -> BootstrapResult(mean, ci_low, ci_high, p_one_sided, n_seasons, n_gws)`.
- [ ] `summarize(...)`: season totals per policy (mean over starts) and paired diffs with CIs, for realized and xG.
- [ ] `log_experiment(path, command, config, metrics, n_variants)` appends a CSV row (with header if new) including `git rev-parse HEAD` (or `unknown`).
- [ ] Tests:
  - bootstrap on a known constant diff (CI collapses), a zero-mean noise series (p ≈ 0.5), determinism per seed, block resampling stays within seasons;
  - per-decision with identical policies gives all-zero diffs;
  - per-decision arms differ only at t (same continuation);
  - the experiment log appends rows.

### Task 9: CLI, real-data runs, docs (wave D; lead + agent)

**Files:** `src/fplopt/cli.py`, `tests/test_cli.py`, `.gitignore` (`/results/*`, `!/results/experiments.csv`), `README.md`, `CLAUDE.md`, `docs/PLAN.md`.

- [ ] `fplopt backtest run --seasons 2016-2024 --policy greedy|roll --xp rolling|ep_next [--threshold ...] --starts "template@1,random:5@1,random:3@20" [--out results/<name>]`. Writes `gws.parquet` and `summary.json`, prints the summary table, and logs the experiment.
- [ ] `fplopt backtest compare --a greedy:ep_next --b greedy:rolling --seasons 2021-2024 --starts ... [--k 4] [--per-decision/--no-per-decision]`. Prints paired results with CIs (realized and xG), writes parquet, logs the experiment.
- [ ] Seasons in `HOLDOUT_SEASONS` are rejected with a clear message. `ep_next` with seasons before 2021 is rejected.
- [ ] The leakage check (`fplopt check leakage`) passes with models and probes (real data).
- [ ] **Real-data runs, reported in the PR:**
  - (a) greedy+rolling vs roll+rolling, 2016–2024;
  - (b) greedy+ep_next vs greedy+rolling, 2021–2024;
  - season totals per policy and season (sanity: template greedy seasons should be in a plausible 1,700–2,400 range);
  - captain/XI regret;
  - runtime.
- [ ] Docs: CLAUDE.md commands; README phase status; PLAN §5 notes with measured numbers and §9.

### Task 10: Review and PR (lead)

- [ ] Independent review (agents: correctness of rules/FT/autosubs, leakage of the backtest path, test coverage). Fix the findings.
- [ ] Full `uv run pytest`, `ruff check`, `ruff format --check`, `pytest -m realdata`, `fplopt check leakage`.
- [ ] PR → CI → CodeRabbit → merge → server `git pull && uv sync --locked`.

### Task 12: VERIFY the WC/FH free-transfer rule (lead, parallel with wave A)

- [ ] From public `entry/{id}/history` for a sample of 2026/27 managers who played a WC (or FH) in GW2–5: reconstruct FTs under both `chip_week_ft` values and find the one never contradicted by free transfers used (`event_transfers − event_transfers_cost / 4`). Use a polite rate (≤ 2 req/s) and a few hundred entries from the overall league's top pages.
- [ ] Set the supplement default accordingly, record it in PLAN §3/§11, and comment on #9 (keep #9 open for the rest of FT reconstruction, Phase 7).

---

## Waves

- **A** (parallel, separate worktrees, disjoint files): Task 1, Task 2, Task 3, Task 4. Lead: Task 0, Task 12.
- **B:** Task 5.
- **C:** Task 6, then Tasks 7 and 8 (7 can run in parallel with 6).
- **D:** Task 9, then Task 10.
