# CLAUDE.md

`docs/PLAN.md` is the source of truth for design decisions — read the relevant section before building a component. Items marked **VERIFY** there are unconfirmed assumptions; check them (and close the matching GitHub issue) before relying on them.

## Commands

- `uv sync --all-extras` — install everything
- `uv run pytest` — tests
- `uv run ruff check . && uv run ruff format .` — lint/format
- `uv run fplopt snapshot daily|tick` / `uv run fplopt backfill element-summary` — archiver jobs (see `deploy/README.md`)
- `uv run fplopt check freshness [--max-age-hours 36]` — fails (and alerts) if the newest bootstrap snapshot is missing or too old
- `uv run fplopt backfill football-data|vaastav|fplcache` — one-off historical backfills into `raw/` (run on the server, see `deploy/README.md`)
- `uv run fplopt rules export <season>` (e.g. `2026-27`, `--out` default `config/scoring`) — rules config from an archived bootstrap
- `uv run fplopt check leakage [--deadlines 12]` — corrupt-the-future check of every feature, xP model (`model:*`) and decision probe (`probe:*`) on real `data/` at a fixed edge list (every season's GW1, 2019/20 GW39/GW47, 2021/22 GW18, 2022/23 GW8, first deadline with snapshots, the live deadline) plus N random deadlines (~9 min); `uv run pytest -m realdata` runs the same edges plus 10 fixed deadlines (~7 min; plain `pytest` skips it)
- `uv run fplopt build all` (or `build <table>`) — rebuild `data/*.parquet` from `raw/` (~1.5 min; needs a local `raw/` copy, see `deploy/README.md`). Club names across sources live in `config/teams.csv`; Understat mapping overrides in `config/overrides.csv`.
- `uv run fplopt backtest run --seasons 2016-2024 [--policy greedy|roll] [--xp rolling|ep_next] [--threshold 1.0 --horizon 6 --decay 0.85 --max-transfers 1] [--starts "template@1,random:5@1,random:3@20"] [--out results/<name>]` — replay one policy over seasons × start states (~1.5 min for that grid); writes `gws.parquet` + `summary.json`, prints season totals by start GW with captain/XI regret
- `uv run fplopt backtest compare --a greedy:ep_next --b greedy:rolling --seasons 2021-2024 [--continuation roll:rolling --k 4] [--no-per-decision] [--block-length 4 --n-boot 2000 --seed 0]` — paired comparison of two policy specs (`name:xp[:key=value,...]`, e.g. `greedy:rolling:threshold=2.0`): full-run and per-decision differences with GW-block bootstrap CIs, realized and xG-scored (~3 min for 2016–2024 × 6 starts). Seasons: `2016-2024`, `2021,2023`, `2021-22` or mixed; the holdout season, `ep_next` before 2021/22 and seasons without data are refused (exit 2). Both commands append to `results/experiments.csv` (tracked; the rest of `results/` is gitignored)

## Invariants (do not break)

- `raw/` is append-only, never edited. Every file is compressed (`.json.gz`, `.csv.gz`; fplcache `.json.xz` copied verbatim) with a timestamp in the path.
- Derived tables are rebuilt from `raw/`; never hand-patch Parquet.
- Every derived row has `event_time` and `available_at`. Feature builders and the backtester read only through `DataStore(data_dir).as_of(deadline)` (keeps `available_at < deadline`); feature builders take exactly one argument, the view. Nothing else in `fplopt.features` reads files (enforced by `tests/test_features_architecture.py`). Table kinds and keys live in `fplopt.build.tables.TABLES`.
- Never key on FPL player ids (they reset each season); use `player_key` from `player_dim`.
- Each data source sits behind an adapter in `src/fplopt/adapters/`.
- User state goes in SQLite (not DuckDB — the bot and pipeline would both write).
- Everything user-facing is keyed by `user_id`.

## Conventions

- Notebooks: flat, copy-pasteable cells; anything reused moves into `src/fplopt/`.
- Tune on component metrics (log loss, calibration, RMSE), not season totals.
- Don't touch the 2025/26 holdout season until Phase 6.
