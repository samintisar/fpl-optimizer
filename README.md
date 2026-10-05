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

Extras: `models` (LightGBM, statsmodels, PyMC), `optimize` (HiGHS, PuLP), `bot` (python-telegram-bot).

### Environment variables

Put these in a local `.env` (gitignored) or set them as env vars on the server:

| Variable | Purpose |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Bot token from @BotFather |
| `TELEGRAM_ADMIN_CHAT_ID` | Chat that receives pipeline-failure alerts |
| `ODDS_API_KEY` | The Odds API key |
| `FPLOPT_RAW_DIR` | Raw snapshot root (default `raw`) |
| `FPLOPT_DATA_DIR` | Parquet root (default `data`) |
| `FPLOPT_USER_DB` | SQLite user-state path (default `data/users.sqlite`) |

## Layout

```
src/fplopt/
  adapters/   one module per data source (fpl, understat, odds, football_data)
  ingest/     raw snapshot writers, backfill
  build/      raw -> Parquet tables, ID mapping
  features/   point-in-time feature builders (all take a deadline)
  models/     team, shares, minutes, bonus, defcon, assemble_xp
  optimize/   MILP formulation, chips, top-k plans
  backtest/   season simulator, state generation, metrics
  bot/        Telegram handlers, alerts, user state
notebooks/    exploration
tests/        incl. leakage tests
config/       scoring rules per season, overrides.csv, settings
raw/          (gitignored) immutable gzipped API snapshots
data/         (gitignored) Parquet tables, SQLite user state
```

## Ground rules

- `raw/` is append-only. Everything downstream is rebuildable from it — fix code and rebuild, never patch data.
- Every derived row carries `event_time` and `available_at`; features and backtests read only through `as_of(deadline)`.
- FPL player ids reset every season — key on `player_key`, never the FPL id.
- Secrets live in `.env` / server env vars, never in the repo.

## Build phases

| # | Phase |
|---|---|
| 0 | Snapshot archiver |
| 1 | Backfill + ID mapping + Parquet tables |
| 2 | `as_of` layer + leakage tests |
| 3 | Backtester + baseline xP |
| 4 | Optimizer |
| 5 | Real models |
| 6 | Holdout evaluation |
| 7 | Telegram bot + go live |
| 8 | Odds integration, distributions, polish |
