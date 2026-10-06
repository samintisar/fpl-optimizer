# FPL Optimizer — Project Plan

A model that recommends Fantasy Premier League decisions (transfers, captain, chips, bench order) to maximize **expected total points**, delivered through a Telegram bot. The user makes the final call.

This document is the source of truth for design decisions. Items marked **VERIFY** are assumptions to check before building on them.

---

## 1. Goals and non-goals

**Goals**
- Maximize expected total season points (pure EV; no rank/variance strategy).
- Recommend, don't automate: top 3 transfer plans + "roll transfer" baseline, captain, bench order, chip timing.
- Usable this season (2026/27) if possible — but only after the holdout validates it (see the go-live gate in §9).
- Multi-user from day one (everything keyed by `user_id`), single user (me) for now. Possible paid product later.
- Realistic target once live: consistently above the average manager; a top-1–5% season is the best-case benchmark set by public model-driven teams. Single-season rank is luck-dominated and is not a success criterion.

**Non-goals (for now)**
- Mini-league / rank optimization, effective ownership, differentials.
- LLM-based news extraction or YouTube/expert opinion signals.
- Price-change prediction.
- Fully automated team submission.

---

## 2. Infrastructure

| Concern | Decision |
|---|---|
| Dev | Windows + Claude Code; exploration in notebooks (flat, copy-pasteable cells), pipeline as a Python package |
| Runtime | Ubuntu server (Tailscale). Bot as `systemd` service (long polling, no public URL). Pipeline via `systemd` user timers (timezone-aware schedules; see `deploy/README.md`) |
| Snapshot backup | No self-hosted backup. [Randdalf/fplcache](https://github.com/Randdalf/fplcache) (public domain, `bootstrap-static` 4×/day) is the independent backup for bootstrap snapshots; the Ubuntu archiver is the only source for fixtures, element-summary, odds and exact pre-deadline timing |
| Analytical storage | Immutable `raw/` + Parquet tables queried with DuckDB |
| User state | SQLite (DuckDB is single-writer; bot + pipeline would clash) |
| Secrets | `.env` locally, env vars on server; never in the repo |
| Solver | HiGHS. Models built with PuLP first (simplest to write/debug); the Phase 4 benchmark records model-build time separately from solve time, and if build time dominates, switch to `highspy`'s bulk API |
| Models | LightGBM (minutes); scipy MLE + closed-form empirical-Bayes shrinkage, statsmodels (team + player models). No MCMC (PyMC) unless a model needs full posteriors |

### Repo layout
```
fpl-optimizer/
  src/fplopt/
    adapters/      # one module per data source (fpl, fplcache, understat, odds, football_data)
    ingest/        # raw snapshot writers, backfill
    build/         # raw -> Parquet tables, ID mapping
    features/      # point-in-time feature builders (all take a deadline)
    models/        # team, shares, minutes, bonus, defcon, assemble_xp
    optimize/      # MILP formulation, chips, top-k plans
    backtest/      # season simulator, state generation, policies, metrics
    bot/           # Telegram handlers, alerts, user state
  notebooks/
  tests/           # incl. leakage tests
  config/          # scoring rules per season, overrides.csv, settings
  raw/             # gitignored, immutable
  data/            # gitignored, Parquet
```

---

## 3. Data layer

### Principles
1. **`raw/` is append-only and never edited.** Every API response saved as gzipped JSON with timestamp in path, e.g. `raw/fpl/bootstrap/2026-10-05T0300.json.gz`.
2. **Everything downstream is rebuildable** from `raw/` with one command. Fix code, rebuild — never patch data.
3. **Every derived row has `event_time` and `available_at`.** `available_at` = when it could have been known. All reads in features/backtest go through `as_of(deadline)`.
4. **Every data source sits behind an adapter**, so free sources can later be swapped for licensed ones without touching models. Adapters are thin and our own (no `soccerdata`: it now hard-depends on a browser stack and ships CAPTCHA solving).

### Sources and schedule

| Source | What | When |
|---|---|---|
| FPL `bootstrap-static/` | Prices, positions, status flags, news, ownership, GW deadlines, `ep_next`, set-piece orders, `team_join_date`, chip windows, `game_config` | Daily after price changes (~02:00 UK) + 2h pre-deadline |
| FPL `fixtures/` | Schedule, kickoffs (as-of fixture lists for blanks/doubles) | Same as above |
| FPL `event/{gw}/live/`, `element-summary/{id}/` | Per-player GW stats/points (incl. CBIT/recoveries/defensive_contribution) | After GW lockdown (09:00 UK the day after the GW's last match); `event-status/` confirms bonus is final |
| FPL `entry/{id}/…` | User picks, history, transfers, chips | Pre-deadline, per registered user |
| fplcache | Historical `bootstrap-static` snapshots ~4×/day, April 2021 → today | One-time backfill (2021/22–2025/26), copied into `raw/` |
| Understat (via vaastav's mirror only) | Per-team match xG/npxG/xGA (2019/20–2024/25); per-player career match logs with npxG, xA, minutes (files exist 2021/22–2024/25; rows reach back to 2014 for players still active then) | Part of the vaastav backfill. **No direct Understat requests:** its robots.txt disallows all crawling |
| The Odds API (free tier: 500 credits/month, live odds only) | EPL h2h + totals (2 credits/call, ~80/month) | Daily + pre-deadline |
| vaastav/Fantasy-Premier-League | Historical FPL per-GW data, 2016/17+ | One-time backfill, pinned to a commit, copied into `raw/` |
| football-data.co.uk | Historical results + match odds (1X2, O/U 2.5, Asian handicap); team match xG (`HxG`/`AxG`) from 2026/27 | One-time backfill; current season re-fetched daily |

**Odds columns (football-data):** use an **explicit allowlist** of pre-match columns, never a regex. Closing columns carry a `C` after the bookmaker code (`B365CH`, `PSCH`, `AvgC>2.5`, `AHCh`) from 2019/20 (2018/19: `PSC*` only) — but `CLH` is Coral, not closing. Prefer market averages (`Avg*`) or Betfair Exchange (`BFE*`); Pinnacle (`PS*`) is sparse in 2025/26. Pre-match odds are collected Friday afternoon (weekend) / Tuesday afternoon (midweek), i.e. close to the deadline — treated as as-of-deadline odds.

**Not sources:** FBref (Opta stats removed January 2026); FotMob/Sofascore/WhoScored (ToS forbid scraping); ClubElo API (down since ~September 2026 — we compute our own Elo from football-data results instead). [FPL-Core-Insights](https://github.com/olbauday/FPL-Core-Insights) is used **for research only** (§6.5), never in the product.

**Start the snapshot archiver immediately** — every day without it is lost point-in-time data. Phase 0 also backfills 2026/27 GW1–5 per-GW stats from `element-summary/` (status flags for those GWs are covered by fplcache).

### Rules config
- One file per season in `config/scoring/` (`2025-26.json`, `2026-27.json`), generated by `fplopt rules export` from `bootstrap-static` `game_config` (`scoring`, `rules`, `settings`), per-GW `events[].overrides`, `chips` and `element_types`, then checked in. Chip windows and FT caps are read from these, never hard-coded.
- **Not in the API** (checked across every 2025/26 fplcache snapshot), so they need a hand-maintained supplement next to the generated files: the defcon thresholds (10 CBIT / 12 CBIRT; the API only has the 2-point award), the BPS rules, and the 2025/26 AFCON free-transfer top-up (no event override ever carried it; GW and amount **VERIFY** from entry transfer histories, issue #9).
- 2026/27 (confirmed against the live API, unchanged from 2025/26 except BPS):
  - **Defensive contributions:** DEF 2 pts at ≥10 CBIT (clearances, blocks, interceptions, tackles); MID/FWD 2 pts at ≥12 CBIRT (CBIT + recoveries); GK not eligible.
  - **Chips:** two sets. Set 1: Bench Boost and Triple Captain GW1–19, Wildcard and Free Hit **GW2–19**; expires at the GW19 deadline. Set 2: all four GW20–38. Free Hit cannot be played in consecutive GWs (per 2025/26 rules; VERIFY still applies).
  - **Free transfers:** bank up to 5 (`max_extra_free_transfers` = 4). No AFCON top-up in 2026/27 (2025/26 had one; not in the API — see above).
  - **BPS changed:** 1 BPS per 3 CBI (was per 2), no −1 for being tackled, GK save BPS restructured.
  - **GW lockdown:** 09:00 UK the day after the GW's last match.

### Tables (Parquet)
- `player_dim` — stable `player_key` (= FPL player `code`, stable across seasons; `opta_code` = `"p"+code`) ↔ Understat id; identity only (names), static (`available_at` = epoch), so no first/last season. **FPL ids reset every season; never use them as keys.**
- `player_season` — per player and season: FPL id (`element_id`), position, names; no club (per-match club in `player_match`, as-of club in `player_snapshot`). `available_at` = the earliest `available_at` of the player's `player_gw` rows that season (`event_time` = that row's deadline).
- `team_dim` — `team_key` (= FPL team `code`, stable across seasons; synthetic codes ≥ 1000 for clubs never in FPL's era, needed for Elo burn-in) ↔ names in every source, from the hand-maintained `config/teams.csv`.
- `fixture` — every EPL match: season, FPL fixture id and GW, kickoff (from FPL), teams, result; football-data and Understat match ids.
- `gameweek` — per season and GW: deadline (from bootstrap snapshots; 2016/17–2019/20 approximated as first kickoff − 90 min and flagged), lockdown time.
- `gameweek_result` — per season and GW outcomes known at lockdown (`available_at` = lockdown): `average_entry_score`.
- `player_snapshot` — price, position, status, chance_of_playing, news, ownership, `ep_next`, penalty / set-piece order, `team_join_date` per snapshot time.
- `schedule` — the final fixture list (no results), available from publication; the fallback as-of schedule before `fixture_snapshot` exists.
- `fixture_snapshot` — the fixture list as it looked at each snapshot. **Only from our own archive (2026-10-05 on):** fplcache has bootstrap only, so earlier seasons have just the final fixture list — the as-of blank/double leak (§4) cannot be fully avoided historically. **This includes the decision GW itself:** before 2026-10-05 `AsOfView.schedule()` is the final schedule, so the target GW already lacks fixtures postponed *after* its deadline (and shows rescheduled ones). Example: 2021/22 GW18 shows only 4 of its 10 fixtures (COVID postponements of December 2021, several announced after the deadline); the same COVID run covers 2021/22 GW16–20 (9, 7, 4, 7, 7 fixtures), and 2020/21 had GW1 (8), GW16–18 (8, 9, 6). Not measurable exactly (no as-of fixture lists before our archive). Upper bound, 2016/17–2026/27 without the holdout: of 379 GWs, 40 have fewer than 10 fixtures in the final schedule and 48 more than 10; some of those blanks/doubles were announced weeks ahead (cup rounds), the rest after the deadline. `upcoming_fixtures` / `team_strength` / `pool_coverage` on such GWs see the final schedule, so backtest results there are optimistic.
- `player_gw` — per player and GW: registered club, position, price at the deadline (`available_at` = deadline − 1h if the player was registered with that club by then, else later: vaastav gives players added or moved during GW t a row for GW t). With `player_snapshot` coverage (from 2021-04-18): the first snapshot listing him with that club if after deadline − 1h; before it: deadline − 1h only for players registered at game launch (first row at the club's first fixture GW) and unchanged clubs, else the deadline. `player_gw_ownership` — `selected`, `transfers_in/out` (`available_at` = deadline, or the `player_gw` row's if later). From vaastav 2016/17–2025/26 and our element-summary archive.
- `player_match` — minutes, starts, goals, assists, CS, saves, cards, BPS, bonus, CBIT/recoveries (where available), npxG, xA, penalties.
- `team_match` — per fixture and side: goals, Understat xG/npxG (`us_*`), football-data xG (`fd_*`, 2026/27), summed FPL xG (`fpl_*`, 2022/23 GW16+; FPL published zeros before GW16, stored as null).
- `understat_map` / `understat_player_match` — Understat id ↔ `player_key` (matched on per-fixture minutes + names; 100% agreement with vaastav's `id_dict` for 2021/22–2022/23) and the per-match Understat rows behind `player_match.us_*`.
- `team_rating` — our Elo ratings from football-data results, per date. Burn-in from 2005/06 (football-data E0 backfilled back to 2005/06 for this only); promoted clubs enter at the mean rating of the clubs they replace. One pre-season row per club and season carries the season-start rating (available like the season's schedule: 1 June, or the previous season's last lockdown if later), so every club has a rating at GW1.
- `odds_snapshot` — long format: fixture, source, bookmaker (`avg`/`pinnacle`/`betfair_ex` from football-data; Odds API bookmaker keys), market (`h2h`/`totals` 2.5/`ah`), outcome, line, price, `is_closing`, snapshot time. football-data pre-match odds get `available_at` = the Friday (weekend) / Tuesday (midweek) 15:00 UK collection time, capped at kickoff − 1h; closing odds `available_at` = kickoff.

User state (SQLite): `users`, `user_state` (squad, purchase prices, bank, FTs, chips remaining, per GW), `user_overrides`, `user_settings`.

### Backfill rules
- Backfill from 2016/17. Snapshot-derived fields (status flags, news, ownership, `ep_next`, set-piece orders) exist from 2021/22 via fplcache; earlier seasons lack them.
- Store raw stat components and **re-score every season under the current season's rules** (scoring config per season in `config/`). Exception: 2019/20–2024/25 have no CBIT/recoveries data in our sources, so they are re-scored **without** defensive-contribution points. 2016/17–2018/19 vaastav files *do* carry `clearances_blocks_interceptions`, `recoveries`, `tackles` (FPL's old detailed stats) — **VERIFY** the definitions match 2025/26+ before scoring defcon for those seasons (if they do, they could also set the defcon prior without FPL-Core-Insights, issue #14).
- **xG coverage** (no direct Understat — see §12):
  - Player npxG/xA: Understat mirror. Share of FPL minutes covered: 2016/17 47%, 2017/18 62%, 2018/19 75%, 2019/20 91%, 2020/21–2023/24 100%, 2024/25 80% (the mirror stops at 2025-04-07). From 2025/26: FPL's own `expected_goals` / `expected_assists` (Opta, in vaastav and our `element-summary` archive; available from 2022/23 GW16, so it overlaps the mirror for calibration). FPL xG includes penalties; npxG for 2025/26+ subtracts penalty xG using penalty attempts inferred from takers — approximate, **VERIFY** against the 2022/23–2024/25 overlap.
  - Team xG: Understat mirror 2019/20–2024/25 (2024/25 only to 2025-04-07), football-data `HxG`/`AxG` 2026/27, summed FPL player xG (2022/23 GW16+) otherwise. 2016/17–2018/19: goals only.
  - Derived tables keep each source in its own columns (e.g. `us_npxg`, `fpl_xg`); blending is a modelling decision (Phase 5).
- Weight older seasons lower in training rather than dropping them.
- Handle quirks: 2019/20 COVID GWs (numbered 39–47; vaastav also has phantom GW29 copies of the rescheduled fixture), 2022/23 has no GW7, duplicated rows (2025/26), 2024/25 assistant managers (`element_type` 5 / position `AM` — dropped), player names that change mid-season (key on ids, never names), Understat files that include other leagues and stale previous-season team files.
- `available_at` for GW outcomes: the GW lockdown — 09:00 UK the day after the GW's last kickoff (official from 2026/27; the same rule applied to earlier seasons as an approximation). Snapshots (ours and fplcache) are exact to their timestamp.

### ID mapping
- Fuzzy match with `rapidfuzz` + hand-maintained `config/overrides.csv`.
- Validation: every player with minutes > 0 must map, or the pipeline fails loudly.

### Data quality
- Schema checks with `pandera`; retries for the "game is being updated" window.
- Pipeline failures → Telegram alert to admin.

---

## 4. Leakage prevention

- **Single access path:** feature builders take a `deadline` and only read via `as_of(df, deadline)` — concretely `DataStore(data_dir).as_of(deadline)`; nothing else in `fplopt.features` reads files. Enforced statically (`tests/test_features_architecture.py`, every module under `fplopt/features/` except `store.py`/`leakcheck.py`): an import allowlist (pandas, numpy, typing, collections, dataclasses, math, logging; only `AsOfView` from the store; sibling feature modules), no dynamic access (`getattr`, `vars`, `globals`, `__dict__`, `__import__`, …) or private attributes, and no state between calls (no `global`/`nonlocal`, cache decorators, mutable module-level values except the `FEATURES` registry). The leakage check runs builders with `DataStore(data_dir=...)` blocked, and computes the clean/corrupted/truncated variants in a seeded random order per deadline.
- **`as_of` semantics:**
  - Keeps rows with `available_at < deadline` (strict); a row available exactly at the deadline is for the next decision.
  - State fixed before deadline t (price at t, registration/club at t): `available_at = deadline_t − 1h` (prices change overnight) — but never before the player was registered with that club (`player_gw`: players added after the deadline are only visible from the first snapshot listing them, or the next GW without snapshots).
  - State that keeps moving until the deadline (ownership after GW t transfers, GW t transfer totals): `available_at = deadline_t`, i.e. visible from deadline t+1.
  - Schedules (fixture list, GW deadlines): `available_at` = publication, approximated as 1 June of the season's start year — or, if later, the lockdown after the previous season's last kickoff, so a season's schedule (which reveals promotion/relegation) is never visible at a deadline of the previous season (2019/20 ended 26 July 2020 → 2020/21 known from 2020-07-27; real publication 2020-08-20, no deadline in between). The build fails if a `schedule`/`gameweek` row of season S is available by S−1's last deadline; Elo pre-season rows use the same rule. Historically only the final schedule exists (accepted leak, §3); from 2026-10-05 the as-of schedule comes from `fixture_snapshot`.
  - Results keep the GW lockdown; snapshot tables are read as "newest snapshot before the deadline".
  - Static tables (`team_dim`, `player_dim`, `understat_map`) are not as-of — `player_dim` lists future debutants, `team_dim.in_fpl` covers every season, `understat_map` stats are fitted on all seasons — so `AsOfView.table()` refuses them; `AsOfView.lookup(name, keys)` returns only identity columns (`TableSpec.public_columns`: key and names) for keys visible at the deadline (in a visible row of `player_season` / `team_rating` / `understat_player_match`).
- **Corrupt-the-future test** (in CI): randomize all data after a deadline, rerun, assert predictions are byte-identical. Requires deterministic models: LightGBM with fixed `seed`, `num_threads=1`, `deterministic=True`; fixed seeds everywhere else; solver with fixed threads and gap-based (not time-based) stopping.
- Known FPL leaks to guard against:
  - Per-GW outcome fields (minutes, points, bonus) used as same-GW features.
  - Current `bootstrap-static` values (form, status, totals) used for past dates.
  - **vaastav's `xP` column** — scraped after the GW, contains lookahead. Never use it; historical `ep_next` comes from fplcache snapshots.
  - Closing odds for matches kicking off after the deadline → use odds as of deadline (historically: the football-data pre-match allowlist).
  - Final fixture list instead of as-of fixture list (rescheduled doubles/blanks).
  - Current positions/prices instead of as-of values.
  - Models/scalers/priors/hyperparameters fit on the full season → refit walk-forward.
- **Status flags:** available as-of from 2021/22 (fplcache). The minutes model is trained without flags (they don't exist before 2021/22); the flag adjustment layer (§6.4) is fit on 2021/22–2022/23 and checked on 2023/24–2024/25.
- **vaastav per-GW fields** (verified against fplcache 2021/22–2024/25, #6): `value` = price at deadline t (100% match with the last pre-deadline snapshot, never the end-of-GW price) → usable at deadline t. `transfers_in/out/balance` = totals for the window ending at deadline t and `selected` = ownership after GW t's transfers; both include the last hours before the deadline (~17% of the window's transfers) that a live pre-deadline read can't see → `available_at` after deadline t, i.e. features for GW t+1 onward. For the GW t decision use `player_snapshot` as of the deadline; before 2021/22, vaastav round t−1 values.

---

## 5. Backtester

### Simulator
- Replays a season GW by GW. At each deadline it sees only `as_of(deadline)` data, builds xP, runs the decision policy, executes this GW's decisions, and scores them with actual outcomes (re-scored per the table below).
- **Starts from any state:** squad, purchase prices, bank, FTs, chips remaining, current GW. This supports both GW1 starts and mid-season opt-ins.
- **Incomplete player pools:** before player snapshots (to 2020/21 GW32) the pool comes from `player_gw` rows visible at the deadline, and a club can have no honest as-of source: at 2020/21 GW1 Man Utd, Man City, Burnley and Aston Villa (GW1 matches postponed, no earlier row) are missing entirely. The `pool_coverage` feature lists per club its fixtures and pool size, and `player_pool` warns. The backtester must not start from — or must flag — a deadline where a club with a fixture in the horizon has 0 pool players (measured over all non-holdout deadlines: only 2020/21 GW1).
- Historical real-manager squads aren't available from the API for past seasons → generate start states from **template squads** (most-owned) and **random valid squads** at various GWs.
- Chips included from the start. Develop/validate seasons use the current season's chip rules. The holdout and live season use their own native rules (2025/26 includes the AFCON free-transfer top-up).
- **Decision policy is pluggable.** Policies:
  - **Greedy** (Phase 3): best single transfer if its horizon xP gain exceeds a threshold, else roll; captain = highest xP starter; no chips.
  - **Optimizer** (Phase 4): the MILP, executing plan #1. Must beat greedy.
  - **Sensitivity** (Phase 8): most frequent plan across noisy re-solves (§7). Must beat the plain optimizer to be adopted.

### Scoring the backtest
| Seasons | Scored under | Compared against |
|---|---|---|
| Develop / validate (2016/17–2024/25) | Current rules, no defcon points | Baselines, greedy policy, previous model versions |
| Holdout (2025/26) | 2025/26 native rules | Above + average manager |
| Live (2026/27) | 2026/27 rules | Above + average manager |

The average-manager benchmark is used only where the rules match; older seasons' averages were scored under different rules and chips. The 2025/26 per-GW `average_entry_score` comes from the season's final fplcache snapshot.

### Splits
| Split | Seasons |
|---|---|
| Develop | 2016/17 – 2022/23 |
| Validate | 2023/24 – 2024/25 |
| Holdout (touch once, at the end) | 2025/26 |
| Live out-of-sample | 2026/27 (log predictions pre-deadline, score after) |

Hyperparameters are chosen by leave-one-season-out within develop, then confirmed on validate.

### Comparing policies (season totals are too noisy to compare directly)
Luck moves a season total by roughly ±80–100 points, so across 9 develop+validate seasons only gaps of ~45+ points/season are detectable from totals alone. Therefore:
- **Paired comparisons only:** every policy runs from identical start states; report the per-state difference.
- **Per-decision evaluation:** at each GW, both policies start from the same state and differ only in this GW's decision; score the difference over the next k GWs under a fixed continuation (roll). ~38 paired samples per season per start state.
- **Confidence intervals:** GW-block bootstrap clustered by season. Start states within one season share realized outcomes, so more states ≠ more independent samples.
- **xG-scored points** as a second, lower-variance metric: realized minutes, with goals/assists/clean sheets replaced by xG/xA/opponent-xG-based expectations. It favours xG-based models, so its sign must agree with realized points.
- **Multiple testing:** `experiments.csv` records the number of variants tried; the validation winner gets a deflation haircut (best-of-N inflation ≈ SE·√(2 ln N)) before it is believed.

### Metrics
- **Tune on component metrics, not season totals:**
  - minutes: ordinal outcome 0 / 1–59 / 60+ — log loss + ranked probability score, reliability diagrams.
  - goals/assists: Poisson/NB log-likelihood; P(≥1) and P(CS): Brier score with decomposition, reliability diagrams.
  - xP: **MSE** per player-GW (MAE is minimized by the median and drags xP down; report it only as a diagnostic). Also MSE stratified by **predicted** xP band and weighted toward optimizer candidates — never grouped by realized outcome.
  - decision-relevant: captain regret and starting-XI regret per GW, Diebold-Mariano tests clustered by GW.
  - team model: implied probabilities vs odds.
- **Calibration:** isotonic/Platt recalibration of P(start), P(CS), P(goal ≥ 1) and per-position linear recalibration of xP, fit on develop and checked on validate.
- Season points per the scoring table above.
- Log every experiment to a CSV (`experiments.csv`: timestamp, git sha, config, metrics, variant count).

### Baselines
- **Rolling average** of points, decisions by the greedy policy.
- **FPL's own `ep_next`** (2021/22+ via fplcache, live from our archive), decisions by greedy and by the optimizer. Running the optimizer on `ep_next` separates the value of our model from the value of the optimizer.
- Every model must beat both.

---

## 6. Models

Philosophy: **market where it is strong, structure elsewhere.** Betting markets are the best public forecast of team goals and clean sheets; structured Poisson models with empirical-Bayes shrinkage handle player shares and rare events; GBM handles minutes. **Minutes is the biggest lever** on xP accuracy and gets the most effort. A direct-GBM xP model (OpenFPL-style) is a low-priority challenger.

**Store components, not just xP:** P(start), P(60+), Poisson rates, etc. Mean xP for now; full distributions (simulation) later become a small add-on.

Blending rule everywhere: equal weights or a single fixed weight — never weights estimated per fold (they overfit).

### 6.1 Team model — market primary
**Market-implied λ (primary for GWs with odds, typically GW+1, sometimes +2):**
- Remove the bookmaker margin with the power method (Shin as an alternative; both beat proportional scaling when there is a clear favourite).
- Solve for Dixon-Coles λ_home, λ_away (ρ fixed from history) by least squares against **1X2 + over/under 2.5** (+ Asian handicap when available). 1X2 alone misstates clean-sheet rates.

**Market-anchored ratings (GWs beyond the odds horizon):**
```
log λ_home = base + home_adv + attack[home] − defence[away]
log λ_away = base            + attack[away] − defence[home]
```
- Fit attack/defence to recent market-implied λs blended with the stats fit below, with time-decay weights, so the market's view carries across the 6-GW horizon.

**Stats model (fallback when no odds; backtest seasons without usable odds):**
- Same structure, fit on **0.7·xG + 0.3·goals** with time-decay weights.
- Priors from our own Elo ratings (football-data results); promoted teams get Elo-based priors.
- P(CS) = P(opponent scores 0); Dixon-Coles low-score correction. Conway-Maxwell-Poisson (under-dispersed goals) as a challenger.

### 6.2 Player shares
- Goal share = player **npxG**/90 ÷ team npxG/90 **while he's on the pitch**; same for xA → assists. Non-penalty xG avoids double-counting with the penalty term.
- "Team npxG while on the pitch" is approximated as team npxG × minutes/90 (no per-shot data without direct Understat access).
- **Hierarchical (Marcel-style) prior:** the player's own previous seasons (weights 2-1-1) plus ~480 minutes of league-average for his position; for new players, position + FPL price. Shrinkage is strong — per-90 npxG/xA are unreliable within half a season.
- Players who change club keep their old share with extra shrinkage.
- **Team consistency:** normalize shares over the minutes-weighted expected XI so Σ players' E[goals] = team non-penalty λ.
- **FPL assists ≠ xA:** FPL awards assists Understat never counts (penalties won, rebounds, deflections). Team E[FPL assists] = λ_team × empirical fraction of goals with an FPL assist; distribute by xA share.
- Penalty term: P(on pitch) × P(penalty taker) × team penalties per match × conversion. Taker order as-of from `penalties_order` (2021/22+); before that, a team-level prior.
- E[goals] = λ_team(non-penalty) × goal_share × E[fraction of match on pitch] + penalty term.

### 6.3 Minutes model (LightGBM)
Hurdle structure:
- P(start)
- P(60+ | start)
- P(sub appearance | not start) and expected sub minutes

Features: recent starts/minutes, starts share of **team** matches, days rest, midweek European/cup match, recent return from absence, positional competition, price, new signing (`team_join_date`), manager tenure, age.

Post-processing:
- **Team normalization:** Σ P(start) ≈ 11 and Σ expected minutes ≈ 990 per team per fixture.
- **Deterministic suspensions:** yellow-card thresholds and red-card bans zero P(start) for the affected GWs.
- **Horizon decay:** for GW+2 onward, decay predictions toward the player's long-run start rate.

### 6.4 Adjustment layer (prediction time only)
- Apply `chance_of_playing_next_round` / status flags (mapping fit on 2021/22+ snapshots, §4).
- Parse "Expected back DD Mon" from `news` into GW-specific availability across the horizon.
- Apply **per-user** P(start) overrides (`/override`). Only overridden players are recomputed per user.

### 6.5 Other components
- **Bonus:** E[bonus | goals, assists, CS, minutes] using the event probabilities already in xP (captures the dominant player in lopsided fixtures); full BPS simulation later. BPS rules changed in 2026/27, so weight recent data heavily and treat older-season bonus as a known source of drift.
- **Defensive contributions — fixed in-season form, prior set once:**
  - Rate = player's CBIT (DEF) or CBIRT (MID/FWD) per 90, shrunk toward the position mean: `rate = (actions_i + k·pos_mean) / (minutes_i/90 + k)`. Player actions and position mean come from the **current season only**, as of the deadline.
  - P(threshold reached | minutes) from a negative binomial with that rate scaled by expected minutes and dispersion `r`.
  - `k` and `r` are set **once** from FPL-Core-Insights 2024/25 data (a validate season; research use only), after checking that its components reproduce FPL's own CBIT/CBIRT on 2026/27 GW1–5. Never re-tuned; the 2025/26 holdout is untouched.
  - Identical in the holdout and live. No defcon term before 2025/26.
- **Saves (GK):** Poisson on saves vs opponent shots/xG.
- **Cards / own goals / penalty misses:** small per-player rates.

### 6.6 xP assembly (per fixture, summed for doubles)
```
appearance = P(plays)·1 + P(60+)·1
attack     = E[goals]·goal_pts(pos) + E[assists]·3
clean_sh   = P(60+)·P(CS)·cs_pts(pos)
conceded   = −E[floor(goals_conceded/2)]            # GK/DEF
+ E[floor(saves/3)] + bonus + defcon + cards
```
Use the Poisson distributions for threshold/floor terms; the output is still one mean xP.

---

## 7. Optimizer (MILP)

### Variables (player i, GW t in horizon)
- Binary: `squad`, `lineup`, `captain`, `vice`, `bench_slot[k]`, `buy`, `sell_first`, `sell_later`
- Chips: fixed per solve (see *Chips*), plus separate free-hit squad variables when FH is in the scenario
- State: `bank[t]` (continuous), `ft[t]`, `hits[t]` (integers), one binary per FT state (1–5)

### Constraints
```
squad[i,t] = squad[i,t-1] + buy[i,t] − sell[i,t]
Σ squad = 15; positions 2/5/5/3; ≤3 per club
lineup ≤ squad; Σ lineup = 11; valid formation (1 GK, 3–5 DEF, 2–5 MID, 1–3 FWD)
Σ captain = 1; captain ≤ lineup
bank[t] = bank[t-1] + Σ sell_price·sell − Σ buy_price·buy ≥ 0
Σ buy[i,t] ≤ ft[t] + hits[t]                 # waived under wildcard / free hit
ft[t+1] ≤ ft[t] − (Σ buy − hits) + 1;  1 ≤ ft ≤ cap        # cap from config; WC/FH effect on FTs from config
```
- **Selling prices:** owned players sell at the user's real selling price (purchase price + half the rise, rounded down to £0.1m) on their **first** sale; a player bought and re-sold within the plan sells at the plan's buy price (`sell_first` vs `sell_later`).
- Binary products (e.g. captain × triple captain) linearized with auxiliary variables.

### Chips
- Chip availability is per chip **and per window** (from the rules config): a set-1 chip not used by the GW19 deadline is lost. Free Hit not in consecutive GWs.
- **Solve by scenario:** enumerate candidate chip assignments within the horizon (no chip; each available chip in each eligible GW; chip pairs only where both are available), fix them, solve each sub-MILP in parallel, keep the best. Modelling chips as free binaries was reported to stall the solver (85% gap after 30 s vs ~5 s with chips fixed).
- **Unused chips have terminal value:** the expected best use of that chip in the rest of its window beyond the horizon (zero once the window closes), estimated from backtest distributions. A chip is played now only if it beats waiting.

### Objective
```
max Σ_t decay^t · [ Σ xP·lineup + xP·captain + Σ_k bench_xP_k + chip terms
                    + ft_state_value(ft[t]) + itb_value·bank[t] ]
    − (4 + hit_margin) · Σ hits
    + terminal_value(squad[T], unused chips)
```
- **FT value is concave:** marginal value of the n-th banked FT, starting at {2: 2.0, 3: 1.6, 4: 1.3, 5: 1.1} (open-fpl-solver defaults), tuned.
- **Money in the bank:** `itb_value` ≈ 0.08 pts per £1m per GW, tuned.
- **Bench:** bench_xP_k = xP × P(bench slot k is needed), from the minutes model's P(starter doesn't play) and autosub order. Fallback fixed weights 0.21 / 0.06 / 0.002, GK 0.03.

### Defaults (tuned by backtest)
| Parameter | Start value |
|---|---|
| Horizon | 6 GWs |
| Decay | 0.85 per GW |
| Hit margin | tuned (hits only if gain > 4 + margin) |
| FT value | concave, see above |
| Bench | P(needed)-weighted; fixed fallback above |
| Vice-captain | 2nd-highest xP starter |

Horizon, decay and FT value are confounded — tune them jointly.

### Practicalities
- **Rolling horizon:** plan 6 GWs, execute only this GW, re-solve next week. Warm-start from last week's plan shifted by one GW.
- **Pruning:** owned players + top-N by xP per position, a minimum expected-minutes floor, an xP-per-price cutoff, and dominance pruning (cheaper and higher xP in every GW).
- **Top 3 plans:** solve → add no-good cut excluding that GW's transfer set → re-solve (×2). Plus the "roll transfer" plan as baseline; show each plan's xP gain vs roll over the horizon.
- **Solve budget:** gap-based stopping (start: 0.5%) with fixed thread count for reproducibility; a generous time cap only as a safety net. Record solve time and final gap per GW.
- **Reference check:** open-fpl-solver (Apache-2.0) is the external reference. A differential test feeds identical xP to both and checks objectives match on no-chip cases.
- **Optimizer's curse:** the chosen plan's predicted gain is biased upward. Log predicted vs realized gain of executed transfers (backtest and live); a slope < 1 means raising the hit margin / FT value or shrinking later-GW xP harder.
- **Sensitivity analysis (Phase 8):** re-solve N times with xP perturbed by **estimate uncertainty** from the component models (not outcome noise); report how often each move appears ("robustness %") and backtest "most frequent plan" as a policy.
- **Uncertain fixtures (Phase 8):** weight scenarios for possible blanks/doubles.
- **VERIFY:** solve times with chip scenarios over 6 GWs; benchmark early.

---

## 8. Telegram bot

### Commands
| Command | Purpose |
|---|---|
| `/register <team_id>` | Rebuild squad, purchase/selling prices, bank, chips used, FTs from public API |
| `/setft <n>` | Correct free-transfer count (reconstruction is fiddly) |
| `/plan` | Top 3 plans + roll, with xP gain over horizon |
| `/captain` | Captain + vice options with xP |
| `/xp <player>` | xP breakdown by component and fixture |
| `/override <player> <p_start>` | Per-user P(start) override for the coming GW |

### Alerts
- 24h before each deadline: full recommendation.
- Silent re-run ~2h before deadline; message only if the recommendation changes.

### Recommendation message contents
- Plans 1–3 + roll, each with transfers, hits, xP gain vs roll.
- Captain/vice, starting XI, bench order.
- Chip recommendation (and the GW it's planned for, if not now).
- Short "why" per transfer from xP components; robustness % once sensitivity analysis exists.

### Live logging (starts as soon as Phase 5 models exist, before go-live)
- Before each deadline, store: plans, xP vectors and components, config, git sha, and the hash of the input snapshot.
- Score a **shadow policy** that follows its own state from GW1 of logging, separately from my real team.
- Run the same optimizer on FPL's `ep_next` for comparison.

---

## 9. Build phases

| # | Phase | Done when |
|---|---|---|
| 0 | Snapshot archiver (systemd timers) + 2026/27 GW1–5 stats backfill | Daily bootstrap/fixtures/odds snapshots landing in `raw/` |
| 1 | Backfill (vaastav incl. its Understat mirror, fplcache, football-data) + ID mapping + Parquet tables + rules config + Elo | All seasons 2016/17+ built from raw; mapping validation passes; `config/scoring/` per season |
| 2 | `as_of` layer + leakage tests | Corrupt-the-future test passes |
| 3 | Backtester + baselines (rolling avg, `ep_next`) + greedy policy + paired evaluation | Simulated seasons from arbitrary states; baselines scored per §5 |
| 4 | Optimizer (transfers, captain, bench, chip scenarios, top-3) | Backtest runs end-to-end; beats greedy; matches open-fpl-solver on no-chip cases; solve times acceptable |
| 5 | Real models (market-implied team model, shares, minutes, components, calibration) | Beat both baselines on component metrics in validation |
| 6 | Holdout evaluation | Single pre-registered run on 2025/26; result recorded |
| 7 | Telegram bot + go live | `/register`, `/plan`, alerts working for my team |
| 8 | Distributions, sensitivity analysis, uncertain fixtures, polish | Each adopted only if it beats the current policy in paired validation |

**Go-live gate (pre-registered):** no recommendations are used live until Phase 6 is recorded and, on the 2025/26 holdout, the full system (our models + optimizer) beats **each** of (a) rolling-average xP + greedy policy and (b) FPL `ep_next` + optimizer:
- one-sided paired test at α = 0.10 on per-decision paired differences in realized points (GW-block bootstrap), **and**
- the same sign on the xG-scored metric.

The test and threshold are fixed now; any change before Phase 6 runs must be logged in §12. Go-live may slip past 2026/27.

---

## 10. Commercial considerations (later)

- Payments: Telegram requires **Telegram Stars** for digital goods sold in bots.
- Move hosting off the home server before anyone pays.
- **Data licensing review** before charging: FPL API terms, Understat data via vaastav (no published terms; contact support@understat.com), odds providers' commercial terms, fplcache (public domain), FPL-Core-Insights (no licence, likely FotMob-derived — research only, must not be needed by the product).
- open-fpl-solver is Apache-2.0: if any of its code is ever copied, keep LICENSE/NOTICE and mark changes.
- Benchmark against existing paid tools' public track records.

---

## 11. Open items to VERIFY

- Free-transfer reconstruction rules from public transfer history, including whether a WC/FH GW grants the +1 FT in 2026/27, and the 2025/26 AFCON top-up (GW and amount; absent from the API).
- Free Hit consecutive-GW restriction in 2026/27.
- FPL-Core-Insights components reproduce FPL CBIT/CBIRT (2026/27 GW1–5) before setting defcon `k`, `r`.
- Solver performance with chip scenarios over a 6-GW horizon.

**Resolved (2026-10-06):**
- vaastav per-GW `selected` / `transfers` / `value` timing → §4 (#6).

**Resolved (2026-10-05):**
- 2026/27 rules → see §3 *Rules config* (checked against the live API).
- FBref → advanced stats removed January 2026; dropped as a source.
- The Odds API → free tier 500 credits/month, live odds only; historical odds paid; player props US bookmakers only.
- football-data.co.uk odds timing → pre-match odds collected Friday/Tuesday afternoon; closing columns identified (§3).
- 2025/26 per-GW `average_entry_score` → final fplcache snapshot of the season.

---

## 12. Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-10-05 | No live recommendations until the Phase 6 holdout passes, even if go-live slips past this season. Archiver runs now. | Keep validation honest; archived data is needed either way. |
| 2026-10-05 | ~~Defensive contributions use a fixed, untuned in-season model.~~ Superseded below. | — |
| 2026-10-05 | Pre-2025/26 seasons re-scored under current rules minus defcon; average-manager benchmark only for holdout and live seasons. | Historical averages were scored under different rules and chips. |
| 2026-10-05 | Goal shares use npxG; penalties modelled separately. | Full xG double-counts penalties. |
| 2026-10-05 | Phase 3 includes a greedy decision policy. | The backtester needs a decision-maker before the optimizer exists. |
| 2026-10-05 | Chip availability tracked per chip per window, read from the API. | 2026/27 chip rules; WC/FH set 1 opens at GW2. |
| 2026-10-05 | Experiment log is a CSV, not MLflow. | YAGNI for a single developer. |
| 2026-10-05 | Market odds are the primary team model from Phase 5; stats model is fallback and long-range anchor. | Bookmaker odds beat xG/goals models on goals and clean sheets. |
| 2026-10-05 | Defcon: fixed in-season form; `k` and dispersion set once from FPL-Core-Insights 2024/25 (research use only). | Only way to set the prior without touching the 2025/26 holdout; licence unclear, so never shipped. |
| 2026-10-05 | fplcache is the historical snapshot source (2021/22+) and the bootstrap backup; no self-hosted GitHub Actions backup. | Public-domain archive 4×/day since April 2021; one less moving part. |
| 2026-10-05 | Policies compared only with paired tests; go-live gate is a pre-registered one-sided paired test. | Season totals are too noisy (±80–100 pts) to compare directly. |
| 2026-10-05 | xP tuned on MSE, not MAE; component metrics are proper scoring rules. | MAE rewards the median and biases xP down. |
| 2026-10-05 | Chips solved by fixed-chip scenarios; FT value concave; bench weighted by P(needed). | Solver performance; public-tool evidence. |
| 2026-10-05 | Own thin adapters; no `soccerdata` dependency; own Elo instead of ClubElo. | soccerdata needs a browser + CAPTCHA solving; ClubElo API is down. |
| 2026-10-05 | Drop PyMC; models use scipy MLE + closed-form shrinkage. PuLP builds the MILP on HiGHS; switch to `highspy` bulk API only if model build time dominates. | Hundreds of walk-forward refits per backtest make MCMC impractical; one modelling layer, decided by measurement. |
| 2026-10-05 | systemd user timers instead of cron; pre-deadline snapshots via a 15-min tick that reads deadlines from the latest archived bootstrap and tracks FPL and odds windows separately. | Ubuntu cron has no per-job timezone (daily run follows UK time); deadlines change, so they are read, not scheduled. |
| 2026-10-06 | No direct Understat requests; use vaastav's Understat mirror (≤2024/25) and FPL's Opta xG (2022/23+). No per-shot data. | Understat's robots.txt disallows all crawling and it publishes no API or terms. |
