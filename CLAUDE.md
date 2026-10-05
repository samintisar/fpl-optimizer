# CLAUDE.md

`docs/PLAN.md` is the source of truth for design decisions — read the relevant section before building a component. Items marked **VERIFY** there are unconfirmed assumptions; check them (and close the matching GitHub issue) before relying on them.

## Commands

- `uv sync --all-extras` — install everything
- `uv run pytest` — tests
- `uv run ruff check . && uv run ruff format .` — lint/format

## Invariants (do not break)

- `raw/` is append-only, never edited. Snapshots are gzipped JSON with a timestamp in the path.
- Derived tables are rebuilt from `raw/`; never hand-patch Parquet.
- Every derived row has `event_time` and `available_at`. Feature builders and the backtester take a `deadline` and read only via `as_of(df, deadline)`.
- Never key on FPL player ids (they reset each season); use `player_key` from `player_dim`.
- Each data source sits behind an adapter in `src/fplopt/adapters/`.
- User state goes in SQLite (not DuckDB — the bot and pipeline would both write).
- Everything user-facing is keyed by `user_id`.

## Conventions

- Notebooks: flat, copy-pasteable cells; anything reused moves into `src/fplopt/`.
- Tune on component metrics (log loss, calibration, RMSE), not season totals.
- Don't touch the 2025/26 holdout season until Phase 6.
