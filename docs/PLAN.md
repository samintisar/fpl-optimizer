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
| Runtime | Ubuntu server (Tailscale). Bot as `systemd` service (long polling, no public URL). Pipeline via `cron` |
| Snapshot backup | GitHub Actions cron → commits gzipped JSON snapshots to a private repo (Actions cron can run late; the pre-deadline snapshot stays on the Ubuntu box) |
| Analytical storage | Immutable `raw/` + Parquet tables queried with DuckDB |
| User state | SQLite (DuckDB is single-writer; bot + pipeline would clash) |
| Secrets | `.env` locally, env vars on server; never in the repo |
| Solver | HiGHS (`highspy`), PuLP as fallback modeling layer |
| Models | LightGBM (minutes), statsmodels / PyMC (team + player models) |

### Repo layout (proposed)
```
fpl-optimizer/
  src/fplopt/
    adapters/      # one module per data source (fpl, understat, odds, football_data)
    ingest/        # raw snapshot writers, backfill
    build/         # raw -> Parquet tables, ID mapping
    features/      # point-in-time feature builders (all take a deadline)
    models/        # team, shares, minutes, bonus, defcon, assemble_xp
    optimize/      # MILP formulation, chips, top-k plans
    backtest/      # season simulator, state generation, metrics
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
4. **Every data source sits behind an adapter**, so free sources can later be swapped for licensed ones without touching models.

### Sources and schedule

| Source | What | When |
|---|---|---|
| FPL `bootstrap-static/` | Prices, positions, status flags, news, ownership, GW deadlines | Daily after price changes (~02:00 UK) + 2h pre-deadline |
| FPL `fixtures/` | Schedule, kickoffs (as-of fixture lists for blanks/doubles) | Same as above |
| FPL `event/{gw}/live/`, `element-summary/{id}/` | Per-player GW stats/points (incl. CBIT/recoveries/defensive_contribution) | After GW lockdown (09:00 UK the day after the GW's last match) |
| FPL `entry/{id}/…` | User picks, history, transfers, chips | Pre-deadline, per registered user |
| Understat | Per-match player + team xG/xA; per-shot data (minute, xG, penalty flag) | After each GW |
| The Odds API (free tier first) | Match result, over/under | Daily + pre-deadline |
| vaastav/Fantasy-Premier-League | Historical FPL per-GW data, 2016/17+ | One-time backfill, pinned to a commit, copied into `raw/` |
| football-data.co.uk | Historical match odds — **pre-match columns only, never closing (`…C…`) columns** | One-time backfill |

FBref is **not** a source: Opta's advanced stats were removed from FBref in January 2026.

**Start the snapshot archiver immediately** — every day without it is lost point-in-time data (especially status flags, which don't exist historically). Phase 0 also backfills 2026/27 GW1–7 per-GW stats from `element-summary/`; status flags and ownership for those GWs are permanently lost.

### Rules config
- One file per season in `config/scoring/` (e.g. `config/scoring/2026-27`). The 2026/27 file is generated from `bootstrap-static` `game_config` and checked in.
- 2026/27 (confirmed, unchanged from 2025/26 except BPS):
  - **Defensive contributions:** DEF 2 pts at ≥10 CBIT (clearances, blocks, interceptions, tackles); MID/FWD 2 pts at ≥12 CBIRT (CBIT + recoveries); GK not eligible.
  - **Chips:** two sets of WC / FH / TC / BB. Set 1 usable GW1–19, expires at the GW19 deadline; set 2 usable GW20–38. No other chips.
  - **Free transfers:** bank up to 5. No AFCON top-up in 2026/27 (2025/26 had one).
  - **BPS changed:** 1 BPS per 3 CBI (was per 2), no −1 for being tackled, GK save BPS restructured.
  - **GW lockdown:** 09:00 UK the day after the GW's last match.

### Tables (Parquet)
- `player_dim` — stable `player_key` ↔ FPL id per season ↔ Understat id. **FPL ids reset every season; never use them as keys.**
- `team_dim` — name mappings across sources.
- `player_snapshot` — price, position, status, chance_of_playing, news, ownership per snapshot time.
- `fixture_snapshot` — the fixture list as it looked at each snapshot.
- `player_match` — minutes, goals, assists, CS, saves, cards, BPS, bonus, CBIT/recoveries (where available), xG, xA, penalties.
- `team_match` — team xG for/against, goals.
- `odds_snapshot` — fixture, market, outcome, price, snapshot time.

User state (SQLite): `users`, `user_state` (squad, purchase prices, bank, FTs, chips remaining, per GW), `user_overrides`, `user_settings`.

### Backfill rules
- Backfill from 2016/17.
- Store raw stat components and **re-score every season under the current season's rules** (scoring config per season in `config/`). Exception: seasons before 2025/26 have no CBIT/recoveries data, so they are re-scored **without** defensive-contribution points.
- Defensive contribution data exists only from 2025/26 (FPL API), and 2025/26 is the holdout → defcon uses a fixed, untuned in-season model (§6.5).
- Use Understat for xG throughout (FPL's own xG fields only exist in recent seasons).
- Weight older seasons lower in training rather than dropping them.
- Handle quirks: 2019/20 COVID GWs (numbered 39–47), postponed GWs (e.g. 2022/23).
- `available_at` for GW outcomes: from 2026/27, the official lockdown (09:00 UK the day after the GW's last match). Earlier seasons: kickoff + fixed delay (approximate). Only archived snapshots are exact.

### ID mapping
- Fuzzy match with `rapidfuzz` + hand-maintained `config/overrides.csv`.
- Validation: every player with minutes > 0 must map, or the pipeline fails loudly.

### Data quality
- Schema checks with `pandera`; retries for the "game is being updated" window.
- Pipeline failures → Telegram alert to admin.

---

## 4. Leakage prevention

- **Single access path:** feature builders take a `deadline` and only read via `as_of(df, deadline)`.
- **Corrupt-the-future test** (in CI): randomize all data after a deadline, rerun, assert predictions are byte-identical. Requires deterministic models: LightGBM with fixed `seed`, `num_threads=1`, `deterministic=True`; fixed seeds everywhere else.
- Known FPL leaks to guard against:
  - Per-GW outcome fields (minutes, points, bonus) used as same-GW features.
  - Current `bootstrap-static` values (form, status, totals) used for past dates.
  - Closing odds for matches kicking off after the deadline → use odds as of deadline (historically: football-data pre-match columns only).
  - Final fixture list instead of as-of fixture list (rescheduled doubles/blanks).
  - Current positions/prices instead of as-of values.
  - Models/scalers/priors/hyperparameters fit on the full season → refit walk-forward.
- **Train/serve skew on status flags:** historical data has no injury flags. Train minutes model without flags; apply flags + overrides in a post-model adjustment layer.
- **VERIFY:** when vaastav's ownership/transfer fields were snapshotted relative to deadlines before using them as features.

---

## 5. Backtester

### Simulator
- Replays a season GW by GW. At each deadline it sees only `as_of(deadline)` data, builds xP, runs the optimizer, executes the recommended GW decisions, and scores them with actual outcomes (re-scored under current rules).
- **Starts from any state:** squad, purchase prices, bank, FTs, chips remaining, current GW. This supports both GW1 starts and mid-season opt-ins.
- Historical real-manager squads aren't available from the API for past seasons → generate start states from **template squads** (most-owned) and **random valid squads** at various GWs.
- Chips included from the start. Develop/validate seasons use the current season's chip rules (two sets, GW19 expiry). The holdout and live season use their own native rules (2025/26 includes the AFCON free-transfer top-up).
- Recommended plan = best plan (backtest executes plan #1).
- **Decision policy is pluggable.** Phase 3 ships a **greedy policy**: best single transfer if its horizon xP gain exceeds a threshold, else roll; captain = highest xP starter; no chips. The Phase 4 optimizer replaces it and must beat it.

### Scoring the backtest
| Seasons | Scored under | Compared against |
|---|---|---|
| Develop / validate (2016/17–2024/25) | Current rules, no defcon points | Baseline xP + greedy policy, previous model versions |
| Holdout (2025/26) | 2025/26 native rules | Above + average manager |
| Live (2026/27) | 2026/27 rules | Above + average manager |

The average-manager benchmark is used only where the rules match; older seasons' averages were scored under different rules and chips.

### Splits
| Split | Seasons |
|---|---|
| Develop | 2016/17 – 2022/23 |
| Validate | 2023/24 – 2024/25 |
| Holdout (touch once, at the end) | 2025/26 |
| Live out-of-sample | 2026/27 (log predictions pre-deadline, score after) |

### Metrics
- **Tune on component metrics, not season totals** (season totals are luck-dominated):
  - minutes: log loss, calibration
  - goals/assists/CS: log loss, calibration curves
  - xP: RMSE / MAE per player-GW
  - team model: compare implied probabilities to odds
- Season points per the scoring table above (`average_entry_score` from the API for the average manager).
- Log every experiment to a CSV (`experiments.csv`: timestamp, git sha, config, metrics). The more variants tried, the more the best backtest score is inflated.

### Baseline
- xP = rolling average of points, decisions by the greedy policy. Every model must beat it.

---

## 6. Models

Philosophy: **hybrid.** Structured Poisson/Bayesian models for rare events (goals, assists, CS); GBM for minutes. Keep a GBM goals model as a challenger — switch if it wins in validation.

**Store components, not just xP:** P(start), P(60+), Poisson rates, etc. Mean xP for now; full distributions (simulation) later become a small add-on.

### 6.1 Team model
```
log λ_home = base + home_adv + attack[home] − defence[away]
log λ_away = base            + attack[away] − defence[home]
```
- Fit on xG with time-decay weights (Dixon-Coles style); promoted-team priors.
- P(CS) = P(opponent scores 0); Dixon-Coles low-score correction.
- Validate against odds-implied probabilities. When odds are available, consider blending.

### 6.2 Player shares
- Goal share = player **npxG**/90 ÷ team npxG/90 **while he's on the pitch**; same for xA → assists. Non-penalty xG avoids double-counting with the penalty term.
- "Team npxG while on the pitch" is built from Understat per-shot data (shot minute vs the player's minutes).
- Shrinkage toward position prior: `share = (npxG_i + k·prior_pos) / (team_npxG_on_i + k)`.
- Separate penalty term: P(penalty taker) × team penalties per match × conversion rate.
- E[goals] = λ_team(non-penalty) × goal_share × E[fraction of match on pitch] + penalty term.

### 6.3 Minutes model (LightGBM)
Hurdle structure:
- P(start)
- P(60+ | start)
- P(sub appearance | not start) and expected sub minutes

Candidate features: recent starts/minutes, days rest, midweek European/cup match, recent return from absence (derived), positional competition, price (importance proxy), new signing, manager tenure.

### 6.4 Adjustment layer (prediction time only)
- Apply `chance_of_playing_next_round` / status flags.
- Apply **per-user** P(start) overrides (`/override`). Only overridden players are recomputed per user.

### 6.5 Other components
- **Bonus:** per-player rate initially; BPS-based model later. BPS rules changed in 2026/27 (fewer CBI points, no tackled penalty, GK saves restructured), so weight recent data heavily and treat older-season bonus as a known source of drift.
- **Defensive contributions — fixed in-season model, never tuned:**
  - Rate = player's CBIT (DEF) or CBIRT (MID/FWD) per 90, shrunk toward the position mean with a prior weight `k` fixed up front: `rate = (actions_i + k·pos_mean) / (minutes_i/90 + k)`.
  - Both the player's actions and the position mean come from the **current season only**, as of the deadline.
  - P(threshold reached | minutes) from a negative binomial with that rate scaled by expected minutes and a dispersion also fixed up front.
  - Identical in the holdout and live. Weak early in a season by design. No defcon term before 2025/26.
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
- Binary: `squad`, `lineup`, `captain`, `vice`, `bench_slot[k]`, `buy`, `sell`
- Chips per GW: `wc[t]`, `fh[t]`, `bb[t]`, `tc[t]` (+ separate free-hit squad variables)
- State: `bank[t]` (continuous), `ft[t]`, `hits[t]` (integers)

### Constraints
```
squad[i,t] = squad[i,t-1] + buy[i,t] − sell[i,t]
Σ squad = 15; positions 2/5/5/3; ≤3 per club
lineup ≤ squad; Σ lineup = 11; valid formation (1 GK, 3–5 DEF, 2–5 MID, 1–3 FWD)
Σ captain = 1; captain ≤ lineup
bank[t] = bank[t-1] + Σ sell_price·sell − Σ buy_price·buy ≥ 0
Σ buy[i,t] ≤ ft[t] + hits[t]                 # waived under wildcard / free hit
ft[t+1] ≤ ft[t] − (Σ buy − hits) + 1;  1 ≤ ft ≤ cap        # cap = 5 (from config)
Σ_{t ∈ half h} chip_c[t] ≤ available[c, h]   for each chip c, half h   # h1 = GW1–19, h2 = GW20–38
≤ 1 chip per GW
```
Chip availability is per chip **and per half**: a set-1 chip not used by the GW19 deadline is lost, never carried into GW20+. A horizon spanning GW19→20 sees both halves' availability. WC/FH effects on banked FTs follow the season's rules config.
Binary products (e.g. captain × triple captain) linearized with auxiliary variables.

### Objective
```
max Σ_t decay^t · [ Σ xP·lineup + xP·captain + Σ_k w_k·xP·bench_k + chip terms ]
    − (4 + hit_margin) · Σ hits
    + terminal_value(ft[T+1], squad[T])
```

### Defaults (tuned by backtest)
| Parameter | Start value |
|---|---|
| Horizon | 6 GWs |
| Decay | 0.85 per GW |
| Hit margin | tuned (hits only if gain > 4 + margin) |
| Bench weights | 0.10 / 0.05 / 0.02 (outfield), GK separate |
| Vice-captain | 2nd-highest xP starter |
| Terminal value | value per banked FT + squad xP beyond horizon (tuned) |

### Practicalities
- **Rolling horizon:** plan 6 GWs, execute only this GW, re-solve next week.
- **Selling prices:** user's real selling prices for owned players (purchase price + half the rise, rounded down to £0.1m); buy = sell for players bought within the plan.
- **Pruning:** owned players + top-N by xP per position.
- **Top 3 plans:** solve → add no-good cut excluding that GW's transfer set → re-solve (×2). Plus the "roll transfer" plan as baseline; show each plan's xP gain vs roll over the horizon.
- **Solve budget:** tuning backtests run with a MIP time limit and gap tolerance (start: 10 s, 0.5%); final runs use tighter settings. Record solve time and final gap per GW.
- **VERIFY:** solve times with chips over 6 GWs; benchmark early.

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
- Short "why" per transfer from xP components.

---

## 9. Build phases

| # | Phase | Done when |
|---|---|---|
| 0 | Snapshot archiver (Ubuntu cron + GH Actions backup) + 2026/27 GW1–7 stats backfill | Daily bootstrap/fixtures/odds snapshots landing in `raw/` |
| 1 | Backfill + ID mapping + Parquet tables + rules config | All seasons 2016/17+ built from raw; mapping validation passes; `config/scoring/` per season |
| 2 | `as_of` layer + leakage tests | Corrupt-the-future test passes |
| 3 | Backtester + baseline xP + greedy policy | Simulated seasons from arbitrary states; baseline + greedy scored per §5 |
| 4 | Optimizer (transfers, captain, bench, chips, top-3) | Backtest runs end-to-end with optimizer; beats greedy policy; solve times acceptable |
| 5 | Real models (team, shares, minutes, components) | Beat baseline on component metrics in validation |
| 6 | Holdout evaluation | Single run on 2025/26; result recorded |
| 7 | Telegram bot + go live | `/register`, `/plan`, alerts working for my team |
| 8 | Odds integration, distributions, polish | Odds evaluated (free first, paid props only if backtest justifies) |

**Go-live gate:** no recommendations are used live until Phase 6 is recorded and the model beats the baseline + greedy policy on the holdout, even if that pushes go-live past 2026/27.

---

## 10. Commercial considerations (later)

- Payments: Telegram requires **Telegram Stars** for digital goods sold in bots.
- Move hosting off the home server before anyone pays.
- **Data licensing review** before charging: FPL API terms, Understat scraping, odds providers' commercial terms.
- Benchmark against existing paid tools' public track records.

---

## 11. Open items to VERIFY

- vaastav field snapshot timing (ownership, transfers, value).
- The Odds API free-tier coverage (EPL markets; player props likely paid).
- football-data.co.uk pre-match odds collection time relative to FPL deadlines (and which columns are closing odds per season).
- A source for 2025/26 per-GW `average_entry_score` (that season is no longer in the live API; check vaastav or other archives).
- Free-transfer reconstruction rules from public transfer history (WC/FH effects on FTs).
- Solver performance with chips over a 6-GW horizon.

**Resolved (2026-10-05):**
- 2026/27 rules → see §3 *Rules config*. Scoring and defcon unchanged from 2025/26; two chip sets with GW19 expiry; FT cap 5; BPS changed; lockdown 09:00 UK next day.
- FBref → advanced stats removed January 2026; dropped as a source.

---

## 12. Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-10-05 | No live recommendations until the Phase 6 holdout passes, even if go-live slips past this season. Archiver runs now. | Keep validation honest; archived data is needed either way. |
| 2026-10-05 | Defensive contributions use a fixed, untuned in-season model; no defcon term before 2025/26. | Free CBIT data exists only from 2025/26 (= holdout); FBref lost Opta data. |
| 2026-10-05 | Pre-2025/26 seasons re-scored under current rules minus defcon; average-manager benchmark only for holdout and live seasons. | Historical averages were scored under different rules and chips. |
| 2026-10-05 | Goal shares use npxG; penalties modelled separately. | Full xG double-counts penalties. |
| 2026-10-05 | Phase 3 includes a greedy decision policy. | The backtester needs a decision-maker before the optimizer exists. |
| 2026-10-05 | Chip availability tracked per chip per half (GW19 expiry). | 2026/27 chip rules. |
| 2026-10-05 | Experiment log is a CSV, not MLflow. | YAGNI for a single developer. |
