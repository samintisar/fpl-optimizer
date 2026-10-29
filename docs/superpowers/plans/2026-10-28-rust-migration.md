# Rust migration — Staged Plan

**Status: APPROVED by the user, 2026-10-28**, as drafted: bot (batch 5) before models (6); a `v1rs` fallback goes live only with the user's go-ahead; random start states frozen as a fixture (batch 7); notebooks decided in batch 8; one PR per batch. Logged in PLAN §12.

**Amended by the user, 2026-10-29: one branch, one switch.** Every batch lands on the long-lived `rust-migration` branch (a draft PR to `main` runs CI); nothing goes live batch by batch. The server switches to Rust once, when all batches are done and every gate has passed. Batch 0 was merged to `main` before this (no runtime effect); batch 1 onwards live only on the branch. `main` keeps the running Python system: fixes there are merged into the branch, and the gates re-run on the branch before the switch.

**Goal:** the whole project runs in Rust: archiver, build, features, models, optimizer, backtester and bot. We port one batch at a time. Python and Rust run side by side until the last batch deletes Python. Each batch goes live only after it **matches the Python output** on real data (the parity gate), so the live bot keeps the behaviour that passed Phase 6.

**Scope today:** about 27k lines in `src/fplopt`, 23k lines of tests. Third-party code that matters:

| Python | Used for | Rust |
|---|---|---|
| pandas, pyarrow | every table | `polars` (lazy where it helps), `arrow`/`parquet` |
| httpx, tenacity | FPL, odds, football-data, Understat fetches | `reqwest` + `backoff` (or a hand-written retry) |
| pandera | schema checks in build | explicit schema asserts on `polars` frames |
| rapidfuzz `token_set_ratio` | Understat name matching | own port of the exact algorithm (golden test against rapidfuzz) |
| scipy `minimize` (L-BFGS-B, trust-exact), `least_squares` (lm), `expit`/`logit`/`gammaln` | availability, components, team models | `argmin` (L-BFGS) + own Newton/trust-region with an exact Hessian, `levenberg-marquardt`, `statrs`/own special functions |
| LightGBM (deterministic, 1 thread, seed 0) | minutes model | the same LightGBM C library through its C API (`lightgbm3` or our own FFI), **same version** |
| highspy / pulp | MILP planner | `highs` crate on the **same HiGHS version** |
| python-telegram-bot | bot | `teloxide` |
| sqlite3 | `state/bot.sqlite` | `rusqlite` (same schema, same file) |

Neither DuckDB, statsmodels nor scikit-learn is imported in `src/`: no ports needed.

---

## Design

### 1. Python and Rust talk through files, not through bindings

The project's boundaries are already files: `raw/`, `data/*.parquet`, `config/`, `state/context/<id>/`, `state/bot.sqlite`, `results/`. Each batch replaces everything that **writes** one of those, so the next layer (Python until it's ported) doesn't care who wrote it.

- No PyO3. A Python ↔ Rust in-process bridge would be throwaway work, since the goal is full Rust.
- The one place an in-process call crosses a batch boundary: the Python backtester calls the optimizer (§4, batch 4). Python keeps its own optimizer until batch 7 ports the backtester.

### 2. Repo layout

A Cargo workspace at the repo root, next to `src/fplopt` (which shrinks one batch at a time):

```
Cargo.toml              # workspace
crates/
  fplopt-core/          # seasons, settings, alerts, heartbeat, redact, shared types (PlayerKey, Gw, Money)
  fplopt-adapters/      # one module per source (invariant: every source sits behind an adapter)
  fplopt-ingest/        # snapshot/backfill jobs -> raw/
  fplopt-build/         # raw/ -> data/*.parquet, TABLES registry, rules export
  fplopt-store/         # DataStore::as_of(deadline)
  fplopt-features/      # depends on fplopt-store ONLY (Cargo enforces what test_features_architecture.py checks today)
  fplopt-models/
  fplopt-optimize/
  fplopt-backtest/      # simulator, policies, evaluate/compare, stats
  fplopt-bot/
  fplopt-cli/           # binary `fplopt-rs`; renamed `fplopt` in batch 8
  fplopt-parity/        # compares Python vs Rust outputs (§3)
```

- Subcommands and flags stay **the same as the Python CLI**, so a migrated job changes `fplopt` to `fplopt-rs` in a systemd unit and nothing else.
- `clippy -D warnings`, `rustfmt`, `cargo test` in CI next to ruff/pytest.

### 3. Parity gate (every batch)

`fplopt-rs check parity <area>` runs both implementations on the same inputs and diffs them:

- Tables: sort by the table's keys (`TABLES`), then integers/strings/dates/booleans equal, floats equal within relative `1e-9` unless the batch says otherwise. Same schema (names, dtypes, nullability).
- Results are written to `results/parity/<UTC>-<area>/` and appended to `results/experiments.csv` (family `rust parity`), like every other run.
- **Every batch's gate passes on the branch; the switch waits for all of them.** If it can't pass (§6 models), the Rust version is a new variant and goes through the usual model/backtest comparison before it can replace Python.
- Python tests for a ported module are translated to Rust tests in the same batch. The Python modules and their tests stay until after the switch: they are deleted (batch 8) once the Rust system has run one full gameweek without a parity regression.

### 4. Deploy

- ubuntuworker, user `fplopt`, no sudo: install Rust with `rustup` in the user's home and `cargo build --release --locked` in `~/fpl-optimizer` after `git pull` (the LightGBM C library and `libgomp1` are already there for Python's LightGBM).
- The switch (once, at the end) = merge `rust-migration`, build, edit every unit to `fplopt-rs …`, `daemon-reload`, restart, between two gameweeks. Rollback = revert the units (the Python commands stay installed until batch 8). Each batch's "Cutover" line below names the units it moves at the switch.
- Shadow runs (batch 1 gate b, and any later ones) come right before the switch, from a separate checkout of the branch, never from `~/fpl-optimizer`.
- Shadow runs write to a separate directory (`raw-shadow/`, `data-shadow/`, `state/context-shadow/`), never to the live ones. `raw/` stays append-only.

### 5. Research loop

The nightly loop's pinned harness (`~/fpl-harness`: `model_eval.sh`, `dev/calibrate_v1.py`, `judge.py`) calls the Python CLI. It stays on Python until batch 7 ships `models eval` / `backtest compare` in Rust. Then the user re-pins the harness to the Rust commands (the maintainer's job, see the guard). Nothing in batches 0–6 touches `~/fpl-research`, `~/fpl-live` or `~/fpl-harness`.

### 6. Holdout

- `holdout run` is **not** ported: it ran once, its record is the `phase6-holdout` tag and the log row, and both stay.
- `holdout check` is ported (batch 7).
- Phase 6 is done, so running the Python and Rust code on 2025/26 inputs to compare their outputs is allowed. It's a code check, not tuning.

---

## Batches

Order follows the import graph (`seasons/adapters → ingest → build → store/features → models → optimize ↔ backtest → evaluate → bot`). The live path goes Rust by batch 6. Research tooling follows.

### Batch 0 — Foundations (~1 week)

- Workspace, CI, `fplopt-core` (`seasons`, `settings`, `alerts` incl. the `fplopt-failure@` Telegram alert, `heartbeat`, `redact`).
- `fplopt-parity` with the table diff and the log row.
- Rust on ubuntuworker; `fplopt-rs --version` from a unit to check the deploy path.
- **Gate:** CI green; `seasons` golden test (every season's id/name/GW mapping identical to Python's).
- *As built:* `fplopt-core` also holds the experiment logger (`log_experiment`, same columns and JSON layout; `git_sha` is the commit the binary was built from). Golden files live in `tests/fixtures/rust_golden/`, written by `tests/test_rust_golden.py`, which also fails if Python drifts from them. `check parity table` self-compares every `data/` table (largest 5.7M rows in 5.5 s).

### Batch 1 — Archiver (~1–2 weeks)

`adapters/*`, `ingest/*`: `snapshot daily|tick`, `backfill element-summary|football-data|vaastav|fplcache`, `check freshness`.

- Same paths, file names, timestamps, compression (`.json.gz`, `.csv.gz`, `.json.xz` copied verbatim).
- **Gate:** (a) recorded HTTP responses → identical decompressed files and paths in both implementations; (b) one week of shadow runs into `raw-shadow/`: same file set per tick (± fetch-time jitter), every file valid and parsable.
- Cutover: `fplopt-tick`, `fplopt-daily`, `fplopt-freshness` units.
- *As built:* `crates/fplopt-archive` (adapters behind traits, raw store, jobs, ESPN, backfills; `backfill espn` too, added to Python by #51). Gzip output is deterministic (mtime 0) but its compressed bytes differ from zlib's; the gate compares decompressed contents, as planned. Run manifests are written field for field like Python's `json.dumps` (`fplopt_core::pyjson`, also now used by the experiment log: `ensure_ascii` included). Error lines keep Python's exception names (`kind_name`: `HTTPStatusError`, `InvalidPayload`, `SnapshotError`, ...), so manifests and alerts read the same; alert texts start `fplopt-rs` so shadow alerts are told apart. Status errors name the URL without its query string. SIGTERM sets a flag the long loops check, so a stopped run still writes its manifest (exit 143, no alert, as Python's `SystemExit`). Base URLs can be redirected by `FPLOPT_<SOURCE>_URL` for the harness only. Gate (a): `dev/archiver_parity.py`, 17 scenarios (daily, tick on an empty store / in a window / idle, every backfill, freshness fresh/stale/empty, and five with 404s, 5xx and invalid payloads) all identical (2026-10-29). Gate (b): shadow units in `deploy/systemd-shadow/`, compared by `dev/shadow_compare.py`; it runs in the week before the switch.

### Batch 2 — Build (~3–4 weeks)

`build/*` (`snapshots`, `players`, `fixtures`, `teams`, `elo`, `odds`, `understat`, `history`), `rules export`, plus `backtest/rules` and `backtest/scoring`, which `build/players` already uses.

- `config/teams.csv`, `config/overrides.csv` read as-is.
- Port `token_set_ratio` exactly; golden test on every name pair the build compares on real data.
- **Gate:** `build all` from the same `raw/`: every table in `data/` identical under §3. Run on the server copy of `raw/` (about 1.5 min Python).
- Cutover: `live prepare`'s `build all` subprocess calls `fplopt-rs build all`.

### Batch 3 — As-of store and features (~2 weeks)

`features/store` (`DataStore::as_of`, the `available_at < deadline` filter), `features/baseline`, `features/history`, the feature part of `features/leakcheck`.

- **Gate:** every feature frame identical at the leakcheck edge list plus 10 fixed random deadlines; `check leakage` (features only) passes in Rust.

### Batch 4 — Optimizer (~3 weeks)

`optimize/model`, `optimize/plans`, the chip search, `optimize plan|squad` given an xP frame file, `entry` (FPL team import).

- HiGHS pinned to the version highspy bundles; same node limit and time cap.
- **Gate:** a corpus of ~300 solves (real deadlines × template/random/real squads × `ep_next`/`v1` frames, with and without chips): objective within `1e-6`; every plan valid under the Python rule checker; same plan #1 wherever the top-2 gap is above `1e-6` (ties may break differently, and that's accepted).
- No cutover yet: the bot moves in batch 5. Python backtests keep the Python optimizer until batch 7.

### Batch 5 — Bot (~3 weeks)

`bot/app`, `service`, `store` (same `bot.sqlite` schema, migrations included), `messages`, `planner` (`LIVE_PARAMS`), `context` reader, `with_live_prices`, `/register /plan /chips /transfer /setft /alerts`.

- Reads the context Python's `live prepare` still writes.
- **Gate:** message golden tests (each command and alert rendered byte-for-byte from the same state); for every registered user, Rust and Python plans on the current context match under the batch 4 rule. Test against a copy of `bot.sqlite`, never the live file.
- Cutover: `fplopt-bot.service` between two gameweeks, not in the 24 h before a deadline.

### Batch 6 — Models and the live job (~5–7 weeks, the riskiest batch)

`models/*` (`team`, `shares`, `minutes` + `gbm`, `availability`, `components`, `calibration_table`, `assemble`), the baselines (`rolling`, `ep_next`, `ep_next_fade`), `bot/context` writer, `live prepare|tick`.

- LightGBM through its C API, same version and `DETERMINISTIC` params: identical inputs should give bit-identical trees.
- Optimizer fits (L-BFGS-B, trust-exact, lm) can't be bit-identical across libraries; they're convex/well-conditioned enough to agree to a tight tolerance.
- **Gate (for v1):**
  1. fitted parameters: relative `1e-6` at the batch 3 deadlines;
  2. xP per player-GW: max abs difference ≤ `0.01` at every 2025/26 and 2026/27 deadline so far;
  3. plan #1 identical (batch 4 rule) for the template squad and every registered user at those deadlines.
- **If the gate fails:** Rust v1 is a new variant (`v1rs`). It goes live only after `models eval` and `backtest compare` (Python backtester, xP frames from Rust) show it is no worse than Python v1 on develop + validate. That takes it outside the Phase 6 system (PLAN §12), so it is the user's decision.
- Cutover: `fplopt-live.timer`. After it, the live system is all Rust.

### Batch 7 — Backtester and research tooling (~5–6 weeks)

`backtest/*` (simulator, state, policies, start states, evaluate/compare, bootstrap, t-tests, family deflation), `evaluate/*` (`models eval`, Diebold-Mariano), `optimize bench`, chip study (`chipgains`, `chiprules`, `chips values`), model probes in `check leakage`, `holdout check`.

- `random:SEED` start states depend on numpy's RNG. Either port PCG64 + `SeedSequence` + numpy's bounded-integer sampling exactly, or freeze the start states once as a fixture file that both implementations read. **Recommended: freeze them**, simpler and just as reproducible.
- Bootstrap CIs use resampling: point estimates must match exactly, CIs within Monte Carlo error (or use the same RNG port).
- **Gate:** re-run the logged `backtest compare` / `models eval` rows in `results/experiments.csv` (develop + validate seasons): point estimates within `1e-6`, CIs within Monte Carlo error, same verdicts.
- Then the user re-pins the research harness (§5).

### Batch 8 — Retire Python (~1 week)

- Rename `fplopt-rs` → `fplopt`; update every unit, `deploy/README.md`, CLAUDE.md commands and invariants (the features rule becomes the crate dependency rule).
- Delete `src/fplopt`, `tests/`, `pyproject.toml`, `uv.lock`.
- Notebooks: keep them as Python reading `data/*.parquet` (read-only analysis), or drop them. **Open question for the user.**

**Total: about 6–8 months part-time.** Batches 0–5 (about 3.5 months) get the archiver, build and bot onto Rust with no change to what the models produce.

---

## Risks

| Risk | Mitigation |
|---|---|
| A model port changes xP and with it the validated system | Batch 6 gate; on failure, a variant that needs the user's go-ahead |
| HiGHS/LightGBM versions drift between Python and Rust during the overlap | Pin both; a CI check that the versions match |
| polars semantics differ from pandas (NaN vs null, sort stability, groupby order, float sums) | The table diff catches it; write the null rule into each table's schema |
| Fuzzy-match score off by one → a different Understat mapping | Exact port + golden test on every real name pair |
| A bot cutover breaks during a deadline | Cutover only between gameweeks; rollback by reverting the unit |
| Research loop breaks | Harness stays on Python until batch 7, re-pinned by the user |
| Two implementations to maintain for months | No new features in Python for ported areas; Python fixes on `main` merged into the branch, gates re-run before the switch; Python deleted one gameweek after the switch |

## Open questions for the user

1. Approve the batch order, especially bot (5) **before** models (6)?
2. Batch 6 fallback: is a `v1rs` variant that passes the comparisons acceptable live, or must Rust match Python v1 exactly?
3. Batch 7: freeze random start states as a fixture (recommended) or port numpy's RNG?
4. Batch 8: keep notebooks as Python analysis over Parquet, or drop them?
5. One PR per batch (as with the earlier phases)? *Changed 2026-10-29: one branch, one switch (see Status).*
