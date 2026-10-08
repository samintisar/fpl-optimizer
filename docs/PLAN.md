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
- **Not in the API** (checked across every 2025/26 fplcache snapshot), so they need a hand-maintained supplement next to the generated files (`config/scoring/<season>.supplement.json`: scoring thresholds, defcon thresholds, hit cost, WC/FH effect on banked FTs, Free Hit consecutive rule, FT top-ups): the defcon thresholds (10 CBIT / 12 CBIRT; the API only has the 2-point award), the BPS rules, and the 2025/26 AFCON free-transfer top-up (no event override ever carried it; GW and amount **VERIFY** from entry transfer histories, issue #9).
- 2026/27 (confirmed against the live API, unchanged from 2025/26 except BPS):
  - **Defensive contributions:** DEF 2 pts at ≥10 CBIT (clearances, blocks, interceptions, tackles); MID/FWD 2 pts at ≥12 CBIRT (CBIT + recoveries); GK not eligible.
  - **Chips:** two sets. Set 1: Bench Boost and Triple Captain GW1–19, Wildcard and Free Hit **GW2–19**; expires at the GW19 deadline. Set 2: all four GW20–38. Free Hit cannot be played in consecutive GWs (per 2025/26 rules; VERIFY still applies).
  - **Free transfers:** bank up to 5 (`max_extra_free_transfers` = 4). No AFCON top-up in 2026/27 (2025/26 had one; not in the API — see above). A Wildcard/Free Hit GW keeps the banked FTs but adds **no** +1 (verified 2026-10-06, #9: all 795 sampled top-800 2026/27 histories fit; the +1 variant is contradicted by 27 of them).
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
- `player_match` — minutes, starts, goals, assists, CS, saves, cards, BPS, bonus, CBIT/recoveries (where available), npxG, xA, penalties. `available_at` = the GW lockdown, or the player's `player_season.available_at` if later (FPL can add a player days after a GW, and his history then has rows for GWs before he was listed).
- `team_match` — per fixture and side: goals, Understat xG/npxG (`us_*`), football-data xG (`fd_*`, 2026/27), summed FPL xG (`fpl_*`, 2022/23 GW16+; FPL published zeros before GW16, stored as null).
- `understat_map` / `understat_player_match` — Understat id ↔ `player_key` (matched on per-fixture minutes + names; 100% agreement with vaastav's `id_dict` for 2021/22–2022/23) and the per-match Understat rows behind `player_match.us_*`.
- `team_rating` — our Elo ratings from football-data results, per date. Burn-in from 2005/06 (football-data E0 backfilled back to 2005/06 for this only); promoted clubs enter at the mean rating of the clubs they replace. One pre-season row per club and season carries the season-start rating (available like the season's schedule: 1 June, or the previous season's last lockdown if later), so every club has a rating at GW1.
- `odds_snapshot` — long format: fixture, source, bookmaker (`avg`/`pinnacle`/`betfair_ex` from football-data; Odds API bookmaker keys), market (`h2h`/`totals` 2.5/`ah`), outcome, line, price, `is_closing`, snapshot time. football-data pre-match odds get `available_at` = the Friday (weekend) / Tuesday (midweek) 15:00 UK collection time, capped at kickoff − 1h; closing odds `available_at` = kickoff.

User state (SQLite): `users`, `user_state` (squad, purchase prices, bank, FTs, chips remaining, per GW), `user_overrides`, `user_settings`.

### Backfill rules
- Backfill from 2016/17. Snapshot-derived fields (status flags, news, ownership, `ep_next`, set-piece orders) exist from 2021/22 via fplcache; earlier seasons lack them.
- Store raw stat components and **re-score every season under the current season's rules** (scoring config per season in `config/`). Exception: 2019/20–2024/25 have no CBIT/recoveries data in our sources, so they are re-scored **without** defensive-contribution points. 2016/17–2018/19 vaastav files *do* carry `clearances_blocks_interceptions`, `recoveries`, `tackles` (FPL's old detailed stats) — **VERIFY** the definitions match 2025/26+ before scoring defcon for those seasons (if they do, they could also set the defcon prior without FPL-Core-Insights, issue #14).
- **xG coverage** (no direct Understat — see §12):
  - Player npxG/xA: Understat mirror. Share of FPL minutes covered: 2016/17 47%, 2017/18 62%, 2018/19 75%, 2019/20 91%, 2020/21–2023/24 100%, 2024/25 80% (the mirror stops at 2025-04-07). From 2025/26: FPL's own `expected_goals` / `expected_assists` (Opta, in vaastav and our `element-summary` archive; available from 2022/23 GW16, so it overlaps the mirror for calibration). FPL xG includes penalties.
    - **npxG for 2025/26+** = FPL xG − 0.79 × penalty attempts (Opta values a penalty at exactly 0.79). Penalty attempts are inferred as an expected value from `penalties_order` and goals.
    - **Verified 2026-10-08** on the 2022/23 GW16 – 2025-04-07 overlap (27,825 player-matches, Phase 5 Task 4):
      - per player-season, the inferred npxG correlates with Understat at r 0.985 (0.992 with true penalty attempts);
      - Opta npxG runs about 9% below Understat (ratio 0.914), and FPL xA about 20% below (0.80);
      - so `fplopt.models.shares` scales FPL values to Understat's level (k_goals ≈ 1.095, k_assists ≈ 1.257, fitted walk-forward on the overlap). The scale only matters where the two sources mix (2025/26+);
      - out of sample (fitted on 2022/23, scored on 2023/24–2024/25): r 0.985;
      - leaving penalties in overstates takers' npxG by 21%.
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
  - **Optimizer** (Phase 4): the MILP, executing plan #1. Must beat greedy (with Phase 5's xP: not met out of sample with the baseline xP, §7 *Phase 4 results*, §12).
  - **Sensitivity** (Phase 8): most frequent plan across noisy re-solves (§7). Must beat the plain optimizer to be adopted.
- **Mechanics (Phase 3, `fplopt.backtest`):**
  - Outcomes for GW t are read from `as_of(lockdown_t + 1 µs)` and re-scored with `score_matches`. Policies get only the deadline view, and state transitions never use outcomes.
  - Develop/validate chip windows apply the 2026/27 windows to `gw_index` (1–19 / 20–38), because 2019/20 numbers GWs 39–47 and 2022/23 has no GW7.
  - GW1 transfers are unlimited and free; then FTs bank to 5 and extra transfers cost 4 each (WC/FH exempt). Money is int tenths of £m.
  - Selling price = purchase + floor(half the rise). A held player who left the game keeps his last price and club.
  - Autosubs and captaincy follow FPL ("played" = minutes > 0 over the GW).
  - xP frames cover the target GW and the next 5 GWs by `gw_index`.
  - Start states: *template* = most-owned valid squad from ownership visible at the deadline (not available at GW1 before 2021/22, so those start at GW2); *random* = seeded, price-weighted valid squad. Both are refused where `pool_coverage` has a gap.
  - `simulate` refuses holdout seasons unless `allow_holdout` (Phase 6 only).

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
- **Per-decision evaluation:** at each GW, both policies start from the same state and differ only in this GW's decision; score the difference over the next k GWs under a fixed continuation (roll). ~38 paired samples per season per start state. States come from a reference trajectory (default: the baseline policy's own run); k = 4 by default.
  - **Windows don't overlap:** decisions at gw_index 1, 1+k, 1+2k, … on one grid per season for every start state (stride = k). Every-GW windows share k−1 GWs of outcomes, and an A/A simulation showed the bootstrap rejecting ~17% at nominal 10%.
  - **Known limitation:** each window starts from the reference state, so a policy's chip or FT spending at t is never charged at t+1, and the roll continuation never uses chips or banked FTs. Judge chips (and FT banking) on full-run paired differences. The go-live test must account for this (pre-register before Phase 6).
- **Confidence intervals:** GW-block bootstrap clustered by season. Start states within one season share realized outcomes, so more states ≠ more independent samples. Implementation: average the paired diffs over start states per (season, GW or decision window), then resample circular blocks of 4 GWs (one window for k = 4 per-decision) within each season; one-sided p = share of bootstrap means ≤ 0. Reported CI: two-sided 80%, whose lower bound is the one-sided α = 0.10 test.
  - **Measured size at nominal 0.10** (A/A placebo, 2000 replicates): per-decision (k = 4, non-overlapping) 0.141 / 0.122 / 0.122 for 1 / 4 / 9 seasons; full run 0.131 / 0.124 / 0.122. The circular block bootstrap understates variance by about (n − b)/(n − 1). The single-season holdout gate is therefore ~0.14, not 0.10 (open item, §11).
  - `realized@xg`: realized points on exactly the sample where the xG metric is defined, so the "same sign" check compares like with like.
- **xG-scored points** as a second, lower-variance metric: realized minutes, with goals/assists/clean sheets replaced by xG/xA/opponent-xG-based expectations. It favours xG-based models, so its sign must agree with realized points.
  - Goals = xG × goal points and assists = xA × 3.
  - For players with ≥ 60 minutes, CS = P(0 conceded), with conceded ~ Poisson(opponent xG × minutes/90). GK/DEF conceded = −E[floor(G/2)].
  - The rest is realized.
  - Understat xG first, then FPL, then football-data (team).
  - Null where inputs are missing, and compared only where both arms are non-null.
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
- **`ep_next_fade`** (Phase 4): `ep_next` in the target GW; in horizon GW h the per-fixture rate is 0.5^h · ep_next's + (1 − 0.5^h) · the player's mean points over his last 10 matches. ep_next is form-driven and copied flat it inflates later GWs; `ep_next` itself is unchanged so Phase 3 results stay valid. A diagnostic, not a baseline.
- Every model must beat both.
- **Phase 3 results** (2026-10-06, `results/experiments.csv`; GW1 starts, means over template + 5 random squads):
  - **greedy(rolling) vs roll(rolling), 2016/17–2024/25:** +11.4 pts/GW full run (80% CI +10.4 to +12.6), +431 per season. Per-decision +9.3 per 4-GW window (+7.6 to +11.1). xG metric agrees (+12.2/GW over the 6 seasons with xG).
  - Template greedy(rolling) seasons: 1,717–2,117 (mean 1,943). Greedy never takes hits; it changes the squad or lineup at nearly every GW.
  - **greedy(ep_next) vs greedy(rolling), 2021/22–2024/25:** −0.6 pts/GW full run (−1.8 to +0.5), no detectable difference. Per-decision −0.5 per window (−3.2 to +2.6). xG agrees.
  - **xG-metric coverage:** a GW's xG score is null if any counted player lacks xG. 2016/17–2018/19 are essentially all null (no team xG; Understat mirror gaps). 2019/20 is partly null: 115 players never got an Understat career log (e.g. David Silva). 2020/21+ is complete.

---

## 6. Models

Philosophy: **market where it is strong, structure elsewhere.** Betting markets are the best public forecast of team goals and clean sheets; structured Poisson models with empirical-Bayes shrinkage handle player shares and rare events; GBM handles minutes. **Minutes is the biggest lever** on xP accuracy and gets the most effort. A direct-GBM xP model (OpenFPL-style) is a low-priority challenger.

**Store components, not just xP:** P(start), P(60+), Poisson rates, etc. Mean xP for now; full distributions (simulation) later become a small add-on.

Blending rule everywhere: equal weights or a single fixed weight — never weights estimated per fold (they overfit).

**Phase 5 decisions** (plan `docs/superpowers/plans/2026-10-08-phase-5-models.md`):
- Each component model is split into `fit(view)` and `predict(view, fitted)`. The fit for deadline d uses `view.earlier(cutoff)`, where the cutoff is the deadline of the season's latest refit GW ≤ d (`gw_index` 1, 5, 9, …). The registered model is still one callable on the view, so the leakage check covers the fit. The backtester's `Caches` memoizes fits by cutoff; model modules keep no state.
- LightGBM runs only through `fplopt.models.gbm`: fixed `seed`, `num_threads=1`, `deterministic=True`, `force_row_wise=True`.
- `starts` is null before 2022/23 GW16. Before then, a team's 11 players with the most minutes in a fixture count as its starters; Task 3 measures the accuracy (§6.3).
- Minutes features with no source are dropped: European/cup matches, manager tenure, age.
- Phase 5 criterion 1 ("beat both baselines on component metrics") is tested on xP and decision metrics, because the baselines have no components. On validate, `v1` vs `rolling` and vs `ep_next` must have:
  - lower MSE per player-GW (one-sided Diebold-Mariano clustered by GW, p < 0.10);
  - lower candidate-weighted MSE;
  - no worse MSE in any predicted-xP band or over horizons 1–5;
  - no worse captain or XI regret.

  Component metrics (minutes, goals, assists, P(CS)) are reported against simple references for diagnosis.

**Phase 5 results so far** (develop 2016/17–2022/23 only; validate not looked at):
- **Baselines** (`fplopt models eval`, Task 1):
  - MSE per player-GW at horizon 0: rolling 5.33 (2016/17–2022/23), ep_next 5.33 (2021/22–2022/23). On their common seasons, rolling 5.10 vs ep_next 5.33, rolling better (DM p 0.001).
  - MSE over horizons 1–5: rolling 5.80, ep_next 6.31, ep_next_fade 5.17.
  - XI regret per GW: rolling 6.4, ep_next 5.8. Captain regret: 5.5 and 4.9.
  - Both overpredict above ~3 xP (rolling: predicted 9.8 → realized 5.8).
  - About 10% of ep_next horizon-0 values are negative (FPL's own numbers).
- **Team model** (Task 2, `fplopt.models.team`):
  - Poisson log-likelihood per side: −1.447 at horizon 0 (Elo-only reference −1.466); about −1.454 at horizons 1–5 (reference about −1.465).
  - P(CS) Brier: 0.186 vs 0.189.
  - Beats the reference in every develop season.
  - Settings: ρ −0.03 (flat profile), power de-vig (Shin equal), ratings half-life 45 days, market weight 0.75, Elo prior strength 2 (grid of 120 variants).
  - Odds are visible at the deadline for 88–96% of target-GW fixtures per season, and ≤ 0.3% beyond.
  - Fit 0.2–0.5 s per cutoff.
- **Minutes model** (Task 3, `fplopt.models.minutes`, `.availability`, `fplopt.features.history`):
  - Inferred starts (the 11 players with the most minutes) vs real `starts` on 2022/23 GW16–2024/25: 98.65% accurate (false positives 0.96%, false negatives 2.30%; 569 of 1,016 misses are 45-minute half-time ties).
  - 3-class minutes log loss at horizon 0: 0.532 vs the last-5 reference 0.632. Horizons 1–5: 0.650 vs 0.736. Beats the reference in every season.
  - With flags (2021/22–2022/23): 0.476 vs 0.512 without.
  - Expected minutes: three per-class means, not a fourth regression.
  - Bans zero P(start), but rule-banned develop rows still started 10% of the time: the rules are approximate.
  - Flags exist from 2020/21 GW32 (fplcache), not only from 2021/22. The flag mapping is fit walk-forward on every visible snapshot season; the §4 check on 2023/24–2024/25 is still open (Task 6).
  - Settings: `GbmParams()` defaults, season decay 0.7, horizon decay 0.85 (36 variants).
  - Fit ~11 s per cutoff (minutes + flags), predict 0.4 s.
- **Shares and penalties** (Task 4, `fplopt.models.shares`), per player-fixture, mean over 2017/18–2022/23:
  - Poisson log-likelihood at horizon 0: goals −0.1355 vs the position-average reference −0.1404; FPL assists −0.1347 vs −0.1382.
  - P(goal ≥ 1) Brier 0.0325 vs 0.0333. Horizons 1–5 are similar.
  - Beats the reference in every season.
  - Means run 2–3% high (e_goals 0.0439 vs 0.0430 realized).
  - Prior: 1920 pseudo-minutes at a price-based mean for every player, with season weights 2-2-1-1 (current, −1, −2, −3). That is stronger than §6.2's ~480 minutes at the position average, which scored worse (38 variants tried).
  - Club changes: half weight on the old club's data. Goals stand in for xG at half weight where xG is missing (2016/17–2018/19).
  - Own goals (~3.5%) are removed from the non-penalty λ. FPL-assisted fraction of goals ≈ 0.87.
  - Penalties: P(taker) comes from `penalties_order` (main taker takes 90% of attempts when on the pitch) or from recent attempts, chained down the order by minutes. Club attempt rates and conversion are shrunk to the league.
  - Fit ≤ 1.3 s, predict ≤ 0.3 s.
- **`v1`** (Task 5, `fplopt.models.assemble`, `.components`, `.calibration`; `MODELS["v1"]`):
  - Points under `backtest_rules(season)`, per fixture, summed per GW:
    - **bonus:** per-position regression on event indicators (last 4 seasons, weighted 0.7^age), applied to expected events;
    - **GK saves:** Poisson on the opponent's market λ, with a shrunk keeper effect;
    - **goals conceded:** −E[⌊N/2⌋] with N ~ Poisson(λ_against × fraction on pitch);
    - **cards and own goals:** shrunk per-90 rates;
    - **penalty misses:** from the shares' penalty goals and conversion;
    - **clean sheets:** P(60+)·P(CS), which runs ~8% under realized because a player subbed off after 60 keeps a clean sheet his team later loses.
  - **Bans:** a walk-forward residual P(start) instead of 0 (banned rows started 31% of the time).
  - **Calibration:** expanding window by season (season S uses maps fit on the walk-forward predictions of 2017/18 … S−1; 2016/17 is burn-in). Only isotonic P(start) at horizons ≥ 1 improved develop metrics, so it is the only map kept. P(CS), P(goal) and the per-position linear xP map were worse and were dropped. Raw xP is already calibrated in the large: mean 1.317 predicted vs 1.327 realized.
  - **Develop results** (`models eval`, 2016/17–2022/23):

    | | MSE h0 | MSE h1–5 | candidate MSE | XI regret | captain regret |
    |---|---|---|---|---|---|
    | `v1` | 4.31 | 4.79 | 9.16 | 5.46 | 4.77 |
    | rolling | 5.33 | 5.80 | 11.68 | 6.41 | 5.50 |

    On 2021/22–2022/23 vs `ep_next`: MSE h0 3.99 vs 5.33, XI regret 5.26 vs 5.84 (DM p 0.087), captain regret 4.46 vs 4.85 (p 0.17). `v1` wins every season and is calibrated in every predicted-xP band.
  - **Component metrics:** minutes log loss 0.527 at h0; player P(CS) Brier 0.077; P(goal) Brier 0.033.
  - **Runtime:** fit 1–21 s per cutoff (about 30 s with predict at live 2026/27 cutoffs; the flag fit dominates), predict 0.5–1.5 s. `models eval` over develop with all four models: about 4 min at `--jobs 14`.
  - **Checks:** `fplopt check leakage` passes with `model:v1` (22.6 min; was ~9), and `pytest -m realdata` passes (19.8 min).
  - **Not tuned yet:** the component priors (cards, keeper, bonus seasons).
- **Review fixes before the validate run** (2026-10-08, `/code-review` of the 5a branch):
  - **Flag mapping:** the flag mapping is now fit on out-of-fold P(start): the start model refit on the other season parity, `minutes.out_of_fold_start`. It used to be fit on the minutes model's in-sample predictions on its own training rows, which are sharper than its predictions at a new deadline. Real-data slope at 2021/22–2022/23 cutoffs: 0.77–0.78; injured players' offset about −5.
  - **Faster fit:** the flag fit now uses Newton's method with the exact Hessian (0.2 s, was 3.2 s over 371 L-BFGS steps). News is parsed once per fit, and the history frame is built once per v1 fit. `fit_availability` at a 2022/23 cutoff: 6 s, was 10 s, including the new out-of-fold fits.
  - **Calibration tied to settings:** the table records the `V1Params` it was fitted with (`CALIBRATION_PARAMS`, `assemble.calibration_fingerprint`); v1 refuses it with other settings, and a test checks that it matches the defaults. Regenerated after the fixes (`dev/calibrate_v1.py walk` + `fit --write`). Again only P(start) at horizons ≥ 1 helps (MSE h1–5 4.687 → 4.684).
  - **No silent zeros:** a horizon fixture missing a team λ, or a pool player whose club plays but who has no per-fixture rows, now raises instead of passing silently as 0 xP.
  - **Candidate test:** `models eval` pairwise candidate tests use the pair's own top-N union (`own_candidate`), so they no longer depend on which other models are in the run.
  - **Variant counts:** the Phase 5 tuning grids are logged in `results/experiments.csv` with their variant counts (team 120, minutes 42, shares 38, v1 calibration 8).
  - **Develop after the fixes:** v1 MSE h0 4.31 vs rolling 5.33 (2016/17–2022/23) and vs ep_next 5.33 (v1 3.99 on 2021/22–2022/23); XI regret 5.46 vs 6.41; captain regret 4.77 vs 5.50. These are the same as before the fixes to two decimals.
- **Criterion 1, validate (2023/24–2024/25), one run** (2026-10-08, after the review fixes; `fplopt models eval --models v1,rolling,ep_next --seasons 2023-2024`, `results/p5-validate`, 4 min). **Met.**

  | | MSE h0 | candidate MSE h0 | MSE h1–5 | XI regret | captain regret |
  |---|---|---|---|---|---|
  | `v1` | **3.47** | **8.79** | **3.98** | **4.60** | **4.04** |
  | rolling | 4.35 | 11.20 | 4.80 | 5.29 | 4.63 |
  | ep_next | 4.46 | 11.31 | 5.24 | 4.68 | 4.11 |

  - **MSE:** lower than both baselines at horizon 0 and at every horizon 1–5 separately. One-sided DM p < 0.001 everywhere. Candidate MSE on each pair's own top-N union: 9.07 vs 11.63 (rolling) and 9.38 vs 12.10 (ep_next), p < 0.001.
  - **Seasons:** v1 wins both (3.45 / 3.50 vs rolling 4.36 / 4.35, ep_next 4.44 / 4.48).
  - **Regrets: no worse** (better, not significantly):
    - XI regret 4.60 vs 5.29 (rolling, p 0.10) and 4.68 (ep_next, p 0.44);
    - captain regret 4.04 vs 4.63 (p 0.13) and 4.11 (p 0.44).
  - **Predicted-xP bands.** The harness's band table puts each model on its *own* bands, so each band holds different players. In bands 4–8 xP v1's MSE is higher there (e.g. 5–6 xP: 22.9 vs 16.6 / 16.1). The reason: v1's high bands hold players who do score high (5–6 xP band: realized 5.96), and high scorers vary more. The baselines' high bands are full of overrated low scorers (realized 3.45 / 3.67).
    - On identical rows, v1's MSE is lower than both baselines in every band, whether the bands are drawn by rolling's xP, ep_next's, v1's own or the three models' mean (e.g. by mean xP, 5–6: 13.7 vs 19.6 / 22.9; ≥ 8: 40.4 vs 53.1 / 60.5).
    - The criterion's "no worse in any band" is read as this paired comparison, the only one that compares like with like. The own-band table is kept as a calibration diagnostic: v1's mean realized points track its prediction in every band, the baselines overpredict above ~3 xP.
  - **Components, validate:**
    - minutes log loss 0.450 at h0 and 0.610 at h1–5;
    - P(start) Brier 0.073 at h0 (it underpredicts high P(start) slightly: 0.85 predicted → 0.92 observed);
    - player P(CS) Brier 0.056; P(goal) Brier 0.030;
    - e_goals mean 0.0402 vs 0.0402 realized.
  - **Flag check (§4) on validate:** no separate with/without-flags run was made; the P(start) reliability above includes the flag layer.
- **After the validate run** (PR #28 review, 2026-10-08): `adjust_minutes` reset every row to "free", so the flag mapping and the team re-normalization also moved fixtures in a predicted ban, overriding the fitted ban residual. The minutes frame now carries `banned`; banned fixtures are left out of the mapping and held fixed. This affects only banned fixtures of flagged-era deadlines (~47 banned player-fixtures per season). The calibration table was regenerated: develop MSE h0 4.2609 (was 4.2608), MSE h1–5 4.6845 (unchanged). Criterion 1 was re-checked on the final code (`results/p5-validate-final`; the same command, as asked in the PR #28 review): MSE h0 3.473 vs 4.352 / 4.458, MSE h1–5 3.976, regrets 4.60 / 4.04. These are identical to the first run to three decimals, so the conclusion stands. A return date ("Expected back" / "Suspended until") still zeroes a banned fixture before that date: it is direct evidence the player is out.

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
- **Solve by scenario:** enumerate candidate chip assignments within the horizon (no chip; each available chip in each eligible GW; chip pairs only where both are available), fix them, solve each sub-MILP, keep the best. Scenarios are searched best-first by upper bound (LP relaxation, or a WC/FH base's bound + a Triple Captain/Bench Boost bound) and only those that could beat the best plan found are solved: the same plan as solving all of them, ~2 MILPs instead of ~205 (see *Solve times* below). Modelling chips as free binaries was reported to stall the solver (85% gap after 30 s vs ~5 s with chips fixed).
- **Unused chips have terminal value:** the expected best use of that chip in the rest of its window beyond the horizon (zero once the window closes), estimated from backtest distributions. A chip is played now only if it beats waiting.

### Objective
```
max Σ_t decay^t · [ Σ xP·lineup + xP·captain + Σ_k bench_xP_k + chip terms
                    + ft_state_value(ft[t]) − ft_state_value(ft[t−1]) + itb_value·bank[t]
                    − (4 + hit_margin) · hits[t] − ε · Σ_i buy[i,t] ]
    + terminal_value(unused chips)
```
- **Terminal value is chips-only:** unused chips are credited (placeholder values, *Chips*); the squad and the banked FTs at the horizon's end are not. Known limitation: a short horizon undervalues moves that pay off after it (and FTs banked for later). The FT term credits the gain in FT-state value inside the decay (open-fpl-solver's convention), not the state value itself.
- **Tie-break ε** (Phase 4 review): 1e-4 points per player bought, decayed, so equal-xP plans make fewer moves (2016/17 GW1 under `rolling`, where every xP is 0, made 5 arbitrary transfers with predicted gain 0). It is part of the reported objective; the open-fpl-solver comparison sets it to 0 (`dev/reference_check.py`), so objectives there are unchanged.
- **FT value is concave:** marginal value of the n-th banked FT, starting at {2: 2.0, 3: 1.6, 4: 1.3, 5: 1.1} (open-fpl-solver defaults), tuned.
- **Money in the bank:** `itb_value` pts per £1m per GW. Default 0 (open-fpl-solver: 0.08): with 0.08 the planner hoarded cash (Phase 4 results below).
- **Bench:** bench_xP_k = xP × P(bench slot k is needed), from the minutes model's P(starter doesn't play) and autosub order. Fallback fixed weights 0.21 / 0.06 / 0.002, GK 0.03.

### Defaults (tuned by backtest)
| Parameter | Start value |
|---|---|
| Horizon | 6 GWs |
| Decay | 0.85 per GW |
| Hits | `max_hits` 0: never (chosen conservatively, Phase 4 results; re-test with Phase 5 models); `hit_margin` (hits only if gain > 4 + margin) only applies with `max_hits` > 0 |
| Money in the bank | `itb_value` 0 |
| FT value | concave, see above |
| Bench | P(needed)-weighted; fixed fallback above |
| Vice-captain | 2nd-highest xP starter |

Horizon, decay and FT value are confounded — tune them jointly.

### Practicalities
- **Rolling horizon:** plan 6 GWs, execute only this GW, re-solve next week. Warm-start from last week's plan shifted by one GW.
- **Pruning:** owned players + top-N by horizon xP and by xP per price per position (20/60/60/30 GK/DEF/MID/FWD), and dominance pruning (cheaper and ≥ xP in every GW, with the dominators spanning ≥ the position's squad slots + 5 clubs so the club cap can't block them all). A minimum expected-minutes floor waits for Phase 5's minutes model.
- **Top 3 plans:** solve → add no-good cut excluding that GW's transfer set → re-solve (×2). Plus the "roll transfer" plan as baseline; show each plan's xP gain vs roll over the horizon.
- **Solve budget:** gap-based stopping (0.5%) with one thread and a fixed seed for reproducibility. Backtests use no wall-clock limit: the safety net is a deterministic node limit (HiGHS `mip_max_nodes` 20,000; real no-chip instances explore at most 11 nodes at 0.5% and 255 at 1e-4, over 78 cases, while their wall time reached 306 s on a loaded machine, where the old 60 s cap would have stopped them at load-dependent plans). A 60 s wall-clock cap applies only to `fplopt optimize plan` / `bench`. A solve stopped by a limit is never silent: `Plan.status`, a warning, and the backtest's `solver_status` / `mip_gap` columns per GW (Phase 4 review). Record solve time and final gap per GW.
- **Reference check:** open-fpl-solver (Apache-2.0) is the external reference. A differential test feeds identical xP to both and checks objectives match on no-chip cases.
- **Optimizer's curse:** the chosen plan's predicted gain is biased upward. Log predicted vs realized gain of executed transfers (backtest and live); a slope < 1 means raising the hit margin / FT value or shrinking later-GW xP harder. Backtest: every simulated GW records `pred_gain` / `real_gain` (players bought minus sold, decision-time xP vs realized points, over the GW and the next 3), and `backtest run|compare` print the slope per policy.
- **Sensitivity analysis (Phase 8):** re-solve N times with xP perturbed by **estimate uncertainty** from the component models (not outcome noise); report how often each move appears ("robustness %") and backtest "most frequent plan" as a policy.
- **Uncertain fixtures (Phase 8):** weight scenarios for possible blanks/doubles.
- **Solve times** (#10, `fplopt optimize bench`, 2026-10-07; 40 cases = 10 deadlines (2021/22 GW2, 33; 2022/23 GW3, 24; 2023/24 GW1, 20; 2024/25 GW7, 11; 2026/27 GW2, 5) × template/random start × ep_next/rolling xP; 6-GW horizon, no chips used yet (205 scenarios, 173 at GW1); defaults; one thread, one process; median / p90 / max):
  | | median | p90 | max |
  |---|---|---|---|
  | model build (PuLP) | 0.06 s | 0.13 s | 0.18 s |
  | no-chip solve (HiGHS, gap 0.5%) | 1.4 s | 6.1 s | 31 s |
  | final gap | 0 | 0.36% | 0.50% |
  | top-3 + roll plan | 11 s | 28 s | 105 s |
  | chips, bound search | 14 s | 30 s | 54 s |
  | — LP relaxations computed / time | 49 / 10 s | 73 / 18 s | 106 / 26 s |
  | — scenario MILPs solved / time | 2 / 3.6 s | 3 / 13 s | 3 / 29 s |
  - Solving every chip scenario took 107–750 s per deadline (5 instances, earlier code); the bound search returns the same plan in 5–15 s there. Same plan as the exhaustive search on every instance checked: those 5; 8 benchmark cases with a 4-GW horizon (`--all-chips`, 2021/22 GW28 and 2022/23 GW35: exhaustive 37–340 s, bound 2.6–16 s); 2 `-m realdata` tests (3-GW); 10 synthetic test instances. Guarantee: within `mip_gap` of the best plan over all scenarios, as the exhaustive search.
  - PuLP's per-column/row hand-over to highspy took ~0.2 s per model (longer than most solves); a bulk hand-over (three array calls, identical model) cut it to ~0.02 s. Model build is not the bottleneck, so no switch to highspy's API.
  - Hard cases are early-season and ep_next instances (ep_next is one number per fixture, so many plans tie); `rolling` mid-season solves in ~0.2 s. With top-N 10/30/30/15 the chip search takes 8.7 / 16 / 34 s.
  - Parallel scenario solves: not needed (~2 MILPs per deadline); backtests parallelise over (season, start) instead.
- **Pruning loss** (same 40 cases, no chips, gap 1e-4; loss = best variant's objective − this one's; "none" hits the 60 s limit in 5 cases):
  | pool | candidates (median) | solve median / p90 | loss median / max | cases > 0.1 |
  |---|---|---|---|---|
  | none (every pool player) | 662 | 18 s / 60 s | — | — |
  | dominance only | 204 | 1.6 s / 20 s | 0 / 0.015 | 0 |
  | **20/60/60/30 + dominance (default)** | 131 | 2.0 s / 15 s | 0 / 0.21 | 2 |
  | 10/30/30/15 + dominance | 99 | 1.2 s / 15 s | 0 / 1.0 | 4 |
  | 5/15/15/8 + dominance | 66 | 0.6 s / 11 s | 0 / 1.3 | 7 |
  - The first dominance rule (≥ squad-slots dominators, any club) lost up to 6.6 points: at GW1/GW2 the dominators of a mid-priced player were often all of one club. Losses > 0.1 left are GW1/GW2 ep_next squad builds.
  - With chips (21 cases, gap 1e-4) the default pool loses median 0, max 0.22 points vs dominance-only pruning (the same winning chip plan in 20/21).
- **Phase 4 decisions** (plan `docs/superpowers/plans/2026-10-07-phase-4-optimizer.md`):
  - Hits sit inside the decay sum, like the points they cost.
  - FT value and money in the bank follow open-fpl-solver's convention (checked by the reference check).
  - Bench uses the fixed fallback weights until Phase 5's minutes model.
  - Chip terminal values are config per chip (0 once the window closes inside the horizon) until backtest-estimated.
  - Top-3 cuts exclude plan #1's first-GW transfer set. Plans #2–3 are re-solved within plan #1's chip scenario (other moves now, given the winning chip plan; every scenario's objective is reported alongside). The roll plan uses the same scenario, minus a Wildcard/Free Hit in the first GW (a transfer chip with no transfers is wasted).
  - Chip terminal values are added undecayed (the value of holding the chip at the horizon's end, in this-GW points); placeholder defaults WC 6, FH 4, BB 4, TC 3. A Free Hit GW credits `itb_value` on the bank carried past it, not the FH squad's leftover.
  - Prices are held fixed over the horizon.
- **Phase 4 results** (Task 5, 2026-10-08; `results/experiments.csv`, `results/phase4-*`; GW1 starts, template + 5 random squads; `backtest compare --jobs 14`; paired A − B, 80% CI, one-sided p; per-decision = per 4-GW window, roll(rolling) continuation; curse = least-squares slope of realized on predicted transfer gain over the GW and the next 3, GWs after GW1 with transfers, gross of hits, with mean predicted → realized gain per such GW):
  - **Hit safeguards** (A = optimizer(ep_next_fade, itb 0.08), B = greedy(ep_next_fade), 2021/22–2024/25, ~13 min each):

    | A | hits / season | full run / GW | per-decision / window | curse slope (pred → real) |
    |---|---|---|---|---|
    | max_hits 0 | 0 | −0.86 (−2.26 to +0.47), p 0.80 | −1.77 (−3.73 to +0.01), p 0.90 | 0.46 (19.9 → 6.5) |
    | max_hits 1, hit_margin 2 | 28 | −1.40 (−2.68 to −0.06), p 0.91 | +0.19 (−2.90 to +3.30), p 0.49 | 0.51 (28.2 → 8.2; net of hits 24.9 → 5.0) |
    | unlimited, hit_margin 0 | 81 | −3.99 (−5.32 to −2.63), p 1.00 | −7.13 (−10.96 to −3.17), p 0.99 | 0.33 (39.0 → 12.0; net 29.8 → 2.8) |

    greedy: slope 0.23 (18.6 → 6.2). **Unlimited hits clearly lose** (−3.1/GW vs `max_hits` 0). **Max 1 hit with margin 2 is undetermined:** the full run favours `max_hits` 0 by +0.54/GW (p 0.26), the per-decision test favours max 1 / margin 2 by 1.95 per window. **Default `max_hits = 0`, chosen conservatively** (realized transfer gains are a quarter to a half of the predicted ones, so a 4-point hit rarely pays with these xP models); `hit_margin` then unused, left at 0. Re-test with the Phase 5 models.
  - **Money in the bank:** with `itb_value` 0.08 (open-fpl-solver's) the planner hoarded cash (mean bank £3.1m vs greedy's £1.7m; sold premiums for cheap players), and at 2016/17 GW1, where `rolling` xP is 0 for everyone (no earlier matches), it sold down to the cheapest squad and banked £36m (2016/17: −221 points vs greedy). optimizer(max_hits 0) with `itb_value` 0 vs 0.08 (same B, paired on A): rolling 2016/17–2024/25 **+0.79/GW** (+0.24 to +1.38, p 0.034; xG +0.64, p 0.04); ep_next_fade +0.21 (−0.54 to +1.03, p 0.37). **Default `itb_value = 0`**.
  - **Defaults (max_hits 0, itb_value 0) vs greedy, same xP** (as run in Task 5):

    | xP, seasons (wall time) | full run / GW | xG full run | per-decision / window | curse slope optimizer / greedy |
    |---|---|---|---|---|
    | ep_next_fade, 2021–24 (15 min) | −0.65 (−2.06 to +0.73), p 0.72 | −0.26, p 0.64 | −1.06 (−3.01 to +0.68), p 0.75 | 0.34 / 0.23 |
    | ep_next, 2021–24 (9 min) | +1.92 (+1.01 to +2.87), p 0.005; +73 / season | +1.70, p 0.002 | −0.72 (−3.31 to +1.89), p 0.62 | 0.25 / 0.17 |
    | rolling, 2016–24 (19 min) | +0.90 (+0.38 to +1.41), p 0.013; +34 / season | +1.30, p 0.001 | −0.86 (−1.99 to +0.30), p 0.84 (on the xG sample, 2019/20+: +0.12, p 0.46) | 0.48 / 0.26 |

  - **The full-run edge does not survive out of sample** (review of the rows above, develop = 2016/17–2022/23, validate = 2023/24–2024/25, full run per GW, 80% CI):

    | xP | all | develop | validate | season-level t-test p (one-sided) | deflated edge (N = 8) |
    |---|---|---|---|---|---|
    | rolling | +0.90 | +1.16 | **−0.03 (−0.95 to +0.88)** | 0.14 | ≈ +0.08 |
    | ep_next | +1.92 | +4.01 | **−0.13 (−1.28 to +0.98)** | 0.11 | ≈ +0.44 |

    - One season carries it: **2021/22 contributes 207 of the 305 season-points** (rolling, sum over seasons) and **190 of 290** (ep_next).
    - Seasons are the independent units (start states and GWs of a season share outcomes): on the per-season mean differences the one-sided t-test gives p 0.14 / 0.11, not < 0.10. The GW-block bootstrap's p 0.013 / 0.005 is too optimistic for a season-clustered effect this uneven.
    - Deflation for best-of-N (PLAN §5 *Multiple testing*: mean − SE·√(2 ln N), N = 8 optimizer-vs-greedy variants of Task 5 incl. the chip run): the edge shrinks to ≈ +0.08 (rolling) and ≈ +0.44 (ep_next) points per GW.
    - **Mechanism** (where the full-run difference comes from): XI/bench structure, not better transfers. The optimizer's XI scores +0.93 (rolling) / +2.20 (ep_next) per GW more than greedy's and its bench 1.63 / 2.09 less: it builds a stronger XI over a weaker bench (its objective values the XI fully and the bench at the fallback weights 0.03–0.21); its transfers are not better per move (greedy's single moves have the larger realized gain at the same states, below); the FT-banking explanation (the optimizer banks FTs for better later moves) is refuted by the data. Plus the GW1 random-start squad rebuilds (unlimited free transfers at GW1: the optimizer rebuilds a random squad wholesale).
  - **Chips** (optimizer(ep_next_fade) with vs without chip scenarios, 2021/22–2024/25, template + 3 random, full run only, 41 min): +4.33/GW (+2.78 to +5.94), p < 0.001, +163 / season (xG +3.80); every season positive (+59 to +342). **This is chips used vs chips wasted**, not evidence of chip timing: the no-chip arm never plays its chips, and the gain is dominated by Wildcard rebuilds; when the chips are played is set by the placeholder terminal values (WC 6, FH 4, BB 4, TC 3). Judging chip timing needs a heuristic chip baseline (e.g. WC at the first international break, BB/TC at the best double) and terminal values estimated from backtest distributions (§11).
  - **Per decision** (roll(rolling) continuation, B's states): every mean negative, none significant. At greedy's states (rolling, 2016/17, 2017/18, 2018/19, 2022/23; 105 decisions where both make one transfer) greedy's move has the larger standalone predicted gain (23.0 vs 20.9 over the 4-GW window; 23.9 vs 21.7 over 6 GWs decayed) and realized gain (4.6 vs 1.1): greedy maximizes the gain of one move by construction, and the roll continuation never carries out the rest of the optimizer's plan (§5 *Known limitation*). Bugs checked first: every simulated decision passes `apply_decision`; at 13 deadlines (2024/25, greedy's states) the MILP's objective is ≥ that of greedy's move fixed in the same model; the plan's XI xP equals the lineup's xP from the frame (captain twice); horizons match the frame's.
  - **Per decision with each arm's own continuation** (Phase 4 review, 2026-10-08, after the review fixes; `results/phase4-review-own-{b,a}`; optimizer(rolling) vs greedy(rolling), 2016/17–2024/25, `template@1,random:3@20`, `--continuation own`, 13.6 / 16.9 min at `--jobs 14`; all 846 optimizer decisions `Optimal`; the full-run rows of the two runs are identical, run independently):

    | | all | develop | validate | season-level t p (all) |
    |---|---|---|---|---|
    | full run / GW (this start set) | +0.69 (+0.06 to +1.28), p 0.085 | +0.11 (−0.66 to +0.86), p 0.43 | +2.66 (+1.68 to +3.62), p < 0.001 | 0.24 |
    | per decision / window, B's (greedy's) states | +2.41 (+0.79 to +4.06), p 0.031 | +2.71 (+0.63 to +4.77), p 0.049 | +1.43 (−0.69 to +3.55), p 0.20 | 0.22 |
    | per decision / window, A's (optimizer's) states | +0.71 (−0.70 to +2.19), p 0.27 | +1.27 (−0.44 to +3.00), p 0.18 | −1.13 (−3.31 to +1.04), p 0.74 | 0.38 |

    - Letting each arm carry out its own plan turns the per-decision sign positive (the roll continuation never credited the optimizer's follow-up moves), but the edge depends on whose states are used, is not significant at the season level, and 2021/22 again dominates (+181 of +205 season-points per decision with B's states; +203 of the full run's +231).
    - The develop/validate pattern of the full run flips with the start set: validate ≈ 0 with the Task 5 GW1 starts (template + 5 random), +2.66/GW with `template@1,random:3@20` (driven by the mid-season random starts and 2024/25's template; develop then +0.11). An edge that changes sign with the start states is not an established edge.
  - **Optimizer's curse:** realized / predicted transfer gain (ratio of the means, GWs after GW1 with transfers) 0.20 (rolling), 0.21 (ep_next), 0.33 (ep_next_fade); greedy(ep_next_fade) 0.33 (18.6 → 6.2). The least-squares slopes (0.25–0.48, greedy 0.17–0.26) are unstable across variants and seasons and are not used for decisions. The baseline xP models are the cause (Phase 5); until then `max_hits = 0`.
  - **Acceptance (PLAN §9 "beats greedy"): not met out of sample with the baseline xP.** With the Task 5 starts the full-run edge is a develop-season effect (validate ≈ 0, 2021/22 dominant, season-level p > 0.10, small after deflation); with another start set the split pattern flips (validate +2.66, develop +0.11), so the edge is not stable; per decision it is negative with the roll continuation and, with each arm's own continuation, positive but reference-dependent and not significant at the season level (p ≥ 0.22). Decision (2026-10-08, §12): merge Phase 4 (backtest end-to-end, open-fpl-solver match, solve times — the other three criteria — are met) and move "beats greedy" to Phase 5, tested with the real xP models: develop-selected, validate-confirmed, deflated.
  - **Review fixes after these runs** (Phase 4 review, 2026-10-08): deterministic solver limits (no wall-clock limit in backtests; node limit), the solver status recorded per GW, the tie-break ε, stricter parameter validation, a robust worker pool. The rows above were run before them; the own-continuation runs above after them.
  - Multiple testing: 8 optimizer-vs-greedy variants in Task 5 (3 hit variants, itb 0 and 0.08 on each xP model, the chip run) — the N used for the deflation above. The experiment log now counts variants per comparison family (`family` column, default `<B spec> <seasons>`; `backtest compare` prints the deflated mean with that family's N, which is smaller than 8 because the Task 5 variants span several baselines and season ranges). Rows logged before the family column keep their old `n_variants` (2 per compare); their family was backfilled from the command line.

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
| 4 | Optimizer (transfers, captain, bench, chip scenarios, top-3) | Backtest runs end-to-end; ~~beats greedy~~ (deferred to Phase 5, §12); matches open-fpl-solver on no-chip cases; solve times acceptable — **done 2026-10-08** |
| 5 | Real models (market-implied team model, shares, minutes, components, calibration) | Beat both baselines on component metrics in validation (tested on xP and decision metrics, §6 *Phase 5 decisions*); **and the optimizer beats greedy with Phase 5 xP** (paired, same xP: chosen on develop, confirmed on validate, deflated for the variants tried; full run and the per-decision design fixed in §11) |
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

- Free-transfer reconstruction rules from public transfer history, and the 2025/26 AFCON top-up (GW and amount; absent from the API). (WC/FH effect resolved below.)
- Free Hit consecutive-GW restriction in 2026/27.
- FPL-Core-Insights components reproduce FPL CBIT/CBIRT (2026/27 GW1–5) before setting defcon `k`, `r`.
- Go-live test size on a single season (~0.14 at nominal 0.10 with the block bootstrap, §5): choose a size-correct test (e.g. a HAC t-test with t critical values, or calibrate the threshold by A/A placebo) and how chips enter it, before Phase 6.
- Per-decision gate design (Phase 4 review): continuation (shared roll vs each arm's own policy, `--continuation own`) and reference (whose run gives the states, `--reference a|b`); the roll continuation never credits a plan's follow-up moves. Decide and log in §12 before Phase 6.
- A heuristic chip baseline (e.g. Wildcard at the first international break, Bench Boost / Triple Captain at the best double), so chip timing is judged against something better than never playing chips.
- Chip terminal values estimated from backtest distributions (now placeholders: WC 6, FH 4, BB 4, TC 3); they set when the planner plays chips.
- Mid-season start states for transfer-quality tests (the GW1 starts let squad rebuilds dominate full-run differences).
- Exclude the degenerate 2016/17 GW1 under `rolling` (no earlier matches: every xP is 0) from comparisons, or start 2016/17 at GW2.

**Resolved (2026-10-08):**
- FPL xG minus penalty xG as npxG for 2025/26+ → §3 *Backfill rules* (verified on the overlap; scale factors fitted walk-forward).

**Resolved (2026-10-07):**
- Solver performance with chip scenarios over a 6-GW horizon → §7 *Solve times* (#10).

**Resolved (2026-10-06):**
- WC/FH GW effect on banked FTs (2026/27): FTs kept, no +1 → §3 *Rules config*, `chip_week_ft: retain` (#9).
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
| 2026-10-06 | Backtester: outcomes read at lockdown via `as_of`; develop chip windows on `gw_index`; per-decision states from the baseline's own run; greedy never takes hits. | Same single access path as features; GW numbering gaps (2019/20, 2022/23); a neutral, reproducible reference. |
| 2026-10-06 | Per-decision windows don't overlap (stride = k, one grid per season); bootstrap blocks counted in GWs; 80% two-sided CIs; `realized@xg` companion rows. | Overlapping windows made the bootstrap reject ~17% at nominal 10%; the 80% lower bound is the gate's one-sided α = 0.10; same-sample sign check. |
| 2026-10-07 | Optimizer: PuLP on in-process HiGHS (1 thread, gap stop), chips by fixed scenarios, hits decayed, open-fpl-solver objective conventions, parallel backtests by (season, start). | PLAN §2/§7; deterministic decisions for the leakage check; like-for-like reference check; hundreds of solves per backtest. |
| 2026-10-08 | Optimizer defaults `max_hits = 0` (conservative: unlimited hits clearly lose, max 1 with margin 2 undetermined) and `itb_value = 0`; `ep_next_fade` xP; parallel backtests (`--jobs`). Phase 4 full-run gains vs greedy (rolling +0.90/GW, ep_next +1.92/GW) are in-sample only (see the next row); chips +4.3/GW is chips used vs wasted. | Cash in the bank was hoarded (2016/17 GW1 sell-off); realized transfer gains are ~⅕–⅓ of predicted (§7 Phase 4 results). |
| 2026-10-08 | Merge Phase 4 without its "optimizer beats greedy" criterion; the criterion moves to Phase 5 (with the real xP models: develop-selected, validate-confirmed, deflated). | Out of sample the full-run edge vanishes (validate: rolling −0.03, ep_next −0.13 per GW; 2021/22 carries ~⅔ of it; season-level p 0.14 / 0.11; deflated ≈ +0.08 / +0.44) the split pattern flips with the start set; per decision the optimizer doesn't beat greedy (negative with the roll continuation; with each arm's own continuation positive but reference-dependent, season-level p ≥ 0.22) (§7 *Phase 4 results*). The planner itself is done and correct (reference check, solve times, leakage checks); its edge depends on xP quality. |
| 2026-10-08 | Phase 5 models fit walk-forward: refit every 4 GWs via `AsOfView.earlier`, with fits memoized by the caller; LightGBM only through `fplopt.models.gbm` (deterministic). Criterion 1 is tested on xP MSE and decision metrics against `rolling` and `ep_next` on validate. Phase 5 ships as two PRs (5a models, 5b decisions). | ~10 fits per season instead of one per deadline, still leak-checked end to end; the baselines have no components to compare; one review per PR stays manageable. |
| 2026-10-08 | Phase 5 criterion 1 met on validate (2023/24–2024/25). The "no worse in any predicted-xP band" condition is judged on identical rows (bands by each model's xP and by their mean), not on each model's own bands. | v1 MSE 3.47 vs 4.35 / 4.46 (p < 0.001 at every horizon), regrets no worse. Own-band MSE compares different players: v1's high bands hold real high scorers, whose outcomes vary more; on identical rows v1 is lower in every band under every banding. |
| 2026-10-08 | Backtests use a deterministic node limit, no wall-clock limit (60 s only for `optimize plan`/`bench`); solver status recorded per GW; tie-break ε 1e-4 per buy; parallel units on a pipe-per-worker pool. | Results must not depend on machine load (solves took up to 306 s on a loaded machine); equal-xP GWs made arbitrary transfers; ProcessPoolExecutor's queue semaphores broke under load on Windows (Phase 4 review). |
| 2026-10-07 | Chip scenarios searched best-first by LP/derived upper bounds (same plan as solving all); bulk PuLP→highspy hand-over; club-aware dominance pruning; top-N 20/60/60/30. | #10 benchmark: 107–750 s → median 14 s per deadline; old dominance lost up to 6.6 pts, 10/30/30/15 up to 1.0. |
