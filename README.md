# fpl-optimizer

Recommends Fantasy Premier League decisions — transfers, captain, bench order, chip timing — to maximize **expected total points**, delivered through a Telegram bot. It recommends; you decide.

The design lives in [`docs/PLAN.md`](docs/PLAN.md), which is the source of truth. Progress is tracked as GitHub milestones (one per build phase) and issues.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync                    # core + dev tools
uv sync --all-extras       # + models, optimizer, bot
uv run pytest
uv run ruff check .
```

Extras: `models` (LightGBM, statsmodels, scikit-learn, scipy), `optimize` (HiGHS, PuLP), `bot` (python-telegram-bot).

### Environment variables

Put these in a local `.env` (gitignored) or set them as env vars on the server:

| Variable | Purpose |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Bot token from @BotFather |
| `TELEGRAM_ADMIN_CHAT_ID` | Chat that receives pipeline-failure alerts |
| `ODDS_API_KEY` | The Odds API key |
| `HEALTHCHECK_PING_URL` | Optional. Pinged (GET) after each successful `snapshot daily`/`tick`, for an external dead-man's switch such as healthchecks.io (see `deploy/README.md`). Treated as a secret: never logged |
| `FPLOPT_RAW_DIR` | Raw snapshot root (default `raw`) |
| `FPLOPT_DATA_DIR` | Parquet root (default `data`) |
| `FPLOPT_USER_DB` | SQLite user-state path (default `data/users.sqlite`) |

## Layout

```
src/fplopt/
  adapters/   one module per data source (fpl, odds, football_data, vaastav, fplcache)
  ingest/     raw snapshot writers (daily/tick jobs), one-off historical backfills
  build/      raw -> Parquet tables (`fplopt build all`), ID mapping, rules export
  features/   point-in-time feature builders (all take a deadline)
  models/     team, shares, minutes, bonus, defcon, assemble_xp
  optimize/   MILP formulation, chips, top-k plans
  backtest/   season simulator, state generation, metrics
  bot/        Telegram handlers, alerts, user state
notebooks/    exploration
tests/        incl. leakage tests
config/       scoring rules per season, teams.csv (club names per source), overrides.csv
raw/          (gitignored) immutable gzipped API snapshots
data/         (gitignored) Parquet tables, SQLite user state
```

## Ground rules

- `raw/` is append-only. Everything downstream is rebuildable from it — fix code and rebuild, never patch data.
- Every derived row carries `event_time` and `available_at`; features and backtests read only through `as_of(deadline)`.
- FPL player ids reset every season — key on `player_key`, never the FPL id.
- Secrets live in `.env` / server env vars, never in the repo.

## Build phases

| # | Phase | Status |
|---|---|---|
| 0 | Snapshot archiver | done, running on the server |
| 1 | Backfill + ID mapping + Parquet tables | done: raw backfills on the server; `fplopt build all` builds every table |
| 2 | `as_of` layer + leakage tests | done: `DataStore(...).as_of(deadline)`, baseline features, corrupt-the-future check (CI + `fplopt check leakage`) |
| 3 | Backtester + baselines + greedy policy | done: simulator from any squad state, rolling-average and `ep_next` xP, roll/greedy policies, paired full-run and per-decision evaluation with GW-block bootstrap (`fplopt backtest run`/`compare`, log in `results/experiments.csv`) |
| 4 | Optimizer | done (criterion deferred to Phase 5): MILP planner (PuLP + HiGHS; transfers, captain, bench, chip scenarios, top-3 + roll plan), `OptimizerPolicy` in the backtester, parallel backtests (`--jobs`), `fplopt optimize plan`/`bench`; hits off by default (`max_hits=0`). "Beats greedy" is **not** met out of sample with the baseline xP (develop +, validate ≈ 0; PLAN §7 *Phase 4 results*) and moves to Phase 5 |
| 5 | Real models (market-implied team model first) | |
| 6 | Holdout evaluation | |
| 7 | Telegram bot + go live | |
| 8 | Distributions, sensitivity analysis, polish | |
