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
- *As built:* `crates/fplopt-build`, with builders that follow the Python modules one to one:
  - The modules: `teams`, `fixtures`, `snapshots`, `players`, `understat`, `odds`, `elo`, `espn` and `rules`, plus `scoring` (the `backtest/rules` loader and the per-match scorer, for `rescored_points`) and `fuzz` (`token_set_ratio`, `normalise_name`).
  - How rows are built: as plain Rust structs, not dataframes. Polars is used only to read and write Parquet, with the same dtypes pandas writes (nullable Int64/Float64, str, bool, `datetime64[us, UTC]`, date32).
  - Raw CSVs are read with pandas' NA strings and boolean spellings. JSON floats are parsed correctly rounded, as Python does.
  - `player_snapshot` is parsed in parallel and written one row group per chunk.
- *Gate (2026-10-29):* `dev/build_parity.py` passes on a frozen copy of the local `raw/` (synced from the server that day, ESPN included).
  - All 22 tables are identical, including `player_snapshot` (5.7M rows) and the Understat and ESPN maps.
  - `build all` takes 140 s in Rust against 354 s in Python.
  - `rules export` is byte-identical for 2025-26 and 2026-27.
  - `token_set_ratio` matches rapidfuzz on all 1,844,337 name pairs the maps score. The committed golden keeps the 49k pairs that can change a map.
  - The schema checks are ported as explicit checks per builder, and they raise as pandera did. Their failure texts differ from pandera's.
- *Still open:*
  - The Python build tests are covered by the real-data gate rather than translated one by one. The Rust unit tests cover the helpers (lockdown and clock changes, CSV NA handling, the float repr, defcon scoring, rapidfuzz).
  - The xz decoder (`lzma-rs`) dominates `player_snapshot` (~75 s).
  - The gate re-runs on the server before the switch.

### Batch 3 — As-of store and features (~2 weeks)

`features/store` (`DataStore::as_of`, the `available_at < deadline` filter), `features/baseline`, `features/history`, the feature part of `features/leakcheck`.

- **Gate:** every feature frame identical at the leakcheck edge list plus 10 fixed random deadlines; `check leakage` (features only) passes in Rust.
- *As built:* two crates.
  - `crates/fplopt-store`: `DataStore` (Parquet or in-memory tables, loaded lazily column by column and kept in `available_at` order) and `AsOfView` (`table`, `latest`, `lookup`, `schedule`, `gameweek_for_deadline`, `earlier`), `files_blocked`, `load_tables`. The `TABLES` registry is ported into `fplopt-build` (`tables.rs`), next to the `Table` type both crates use.
  - `crates/fplopt-features`, which depends on `fplopt-store` only: the seven `FEATURES`, the minutes model's `history.training_frame` / `prediction_frame` and `bans`, and the leakage check.
  - Python's row orders are kept wherever an order decides a value (first or last of duplicates, ties, `merge_asof` picks). Grouped float sums use pandas' Kahan summation, and day fractions use its nanosecond timedeltas.
  - `check leakage` covers the features and the history frames; models and probes join it in batches 4–7. Its corruption uses this port's own RNG (a seeded xoshiro256**), not numpy's: the edge deadlines are Python's, the sampled ones differ.
  - `fplopt-rs features dump` (not in the Python CLI) writes the frames for the gate.
- *Gate (2026-10-30):* `dev/features_parity.py` on the local `data/` (built 2026-10-29).
  - All 234 frames are identical: 9 frames (7 features, 2 history frames) at 26 deadlines, the 16 edge deadlines plus 10 sampled ones from 2016/17 GW1 to 2026/27 GW10. Each frame was diffed on its key, and its row order was checked too.
  - Python takes 80 s and Rust 33 s for all of them.
  - `fplopt-rs check leakage` passes: 28 deadlines (16 edge, 12 sampled) × 9 frames × 2 variants, no leaks, 17 min (about 35 s per deadline: two corrupted copies of 13 tables in memory).
- *Still open:* the Python feature tests are covered by the real-data gate rather than translated one by one. The Rust unit tests cover the store's visibility rules, the ban rules and the leakage harness (it must catch a stateful and a failing builder).

### Batch 4 — Optimizer (~3 weeks)

`optimize/model`, `optimize/plans`, the chip search, `optimize plan|squad` given an xP frame file, `entry` (FPL team import).

- HiGHS pinned to the version highspy bundles; same node limit and time cap.
- **Gate:** a corpus of ~300 solves (real deadlines × template/random/real squads × `ep_next`/`v1` frames, with and without chips): objective within `1e-6`; every plan valid under the Python rule checker; same plan #1 wherever the top-2 gap is above `1e-6` (ties may break differently, and that's accepted).
- No cutover yet: the bot moves in batch 5. Python backtests keep the Python optimizer until batch 7.
- *As built:*
  - `crates/fplopt-highs`: HiGHS from the pinned submodule `third_party/HiGHS` (v1.15.1, commit 04024d7, the version highspy 1.15.1 bundles), built with CMake in Release with CMake's own flags, and a hand-written binding of the C API calls the optimizer makes (no bindgen). CI checks the submodule out.
  - `crates/fplopt-optimize`: `state` (the squad state and rules the planner needs: selling price, refresh, chip windows, `next_state`, `best_xi`), `params`, `problem` (`PlanInput::from_context`), `prune`, `minutes`, `chips`, `model`, `search`, `plans` (`optimize`), `io` (gate cases), `entry` (FPL team import, squad files) and `start` (the template squad). The model hands HiGHS what PuLP's bulk hand-over does: columns in creation order, the same rows, objective coefficients from the same float operations (merged in first-seen order), the same options. With HiGHS's fixed seed and one thread, solves follow the same path (equal node counts).
  - CLI: `optimize plan` and `optimize squad` with Python's flags, except that the xP frame comes from `--xp-file` (models: batch 6) and `--start random:SEED` waits for numpy's RNG (batch 7). `optimize solve` runs one gate case (not in the Python CLI). The FPL entry endpoints are on the archiver's `FplClient`.
  - Tests: the Python optimizer tests translated as integration tests (`crates/fplopt-optimize/tests/`, 90 tests: model, chips, search, prune, minutes; a test-only `apply_decision` replays plans; seeded instances use the tests' own generator), and the entry and start tests as unit tests. Skipped: tests of PuLP/highspy internals (bulk hand-over, warm starts, pickling, monkeypatched solves) and the `realdata` one, which the gate covers.
- *Gate (2026-10-30):* `dev/optimize_parity.py` on the local `data/`: seed 0 (12 deadlines) 120/120 cases and seed 1 (20 deadlines) 173/173, i.e. 293 planner calls (template, random and carried squads; `ep_next` everywhere, `v1` at 8 deadlines (68 cases); 119 with chips): all 1,172 plans (#1–#3 and roll) identical to Python's, objectives equal, node counts equal, bench weights identical, and every Rust plan valid under Python's rule checker. `optimize plan`/`squad` print Python's text on template, chip, imported (`--entry`, live FPL) and squad-file runs.
- *Still open:* run the gate on the server (Linux: gcc-built HiGHS vs the manylinux highspy wheel) before the switch.

### Batch 5 — Bot (~3 weeks)

`bot/app`, `service`, `store` (same `bot.sqlite` schema, migrations included), `messages`, `planner` (`LIVE_PARAMS`), `context` reader, `with_live_prices`, `/register /plan /chips /transfer /setft /alerts`.

- Reads the context Python's `live prepare` still writes.
- **Gate:** message golden tests (each command and alert rendered byte-for-byte from the same state); for every registered user, Rust and Python plans on the current context match under the batch 4 rule. Test against a copy of `bot.sqlite`, never the live file.
- Cutover: `fplopt-bot.service` between two gameweeks, not in the 24 h before a deadline.
- *As built:* `crates/fplopt-bot`.
  - `store`: `UserStore` on `rusqlite` (bundled SQLite) with Python's schema, SQL and transactions, WAL, the disk-I/O retry; JSON columns written as Python's `json.dumps` (`pyjson`, which now writes floats as Python's repr: `1e-05`).
  - `context`: `PlanContext` and `ContextStore` (read, latest with its cache, prune), `with_live_prices`. Writing contexts stays in Python's `live prepare` until batch 6.
  - `planner`: `live_params` (`holdout.SYSTEM` with the 60 s cap), the squad import with the transfers already made and the `/setft` override, `plan_user`, `/chips` (`chip_params` with the estimated terminal values, the single-chip scenarios), `ranked`, `decision_summary`, `plans_json`.
  - `messages`, `service`: every text and command of Python's, and `alert_user` (Python `live._alert_user`, for batch 6's alert job).
  - `telegram`: long polling over the Bot API with `reqwest` instead of `teloxide`: python-telegram-bot's routing (a command entity for this bot, else the text handler with `parse_command`), one update at a time, private chats only, refused chats skipped, errors and panics alerted to the admin with the apology. `fplopt-rs bot run` stops between polls on SIGTERM.
  - `replay` and `fplopt-rs bot replay` (not in the Python CLI): scripted sessions on recorded FPL responses; Python's side is `tests/bot_replay.py`.
  - Found on the way: CPython 3.12's `sum()` of floats is compensated (Neumaier), so the optimizer's plan xP, bench xP, objective, chip bounds and terminal values now use `py_sum`. Batch 4's gate compared those values within 1e-6, so it didn't see the last-bit differences. Re-run afterwards (seed 2, 5 deadlines): 47/47 cases, 188 plans identical, node counts equal.
  - Tests: the synthetic golden `tests/fixtures/rust_golden/bot/` (a 55-step session: access, every command and its usage errors, both alerts with changes, `/transfer` accepted and refused, `/chips`, live prices and their fallback, after the deadline), replayed by `cargo test` with replies and database rows compared, and checked against Python by `tests/test_rust_golden.py`; unit tests for the store, texts, routing, the sender (never the token) and the live settings. Not ported here: the context builder, fingerprint and alert job tests (batch 6).
- *Gate (2026-10-30):* `dev/bot_parity.py` on the server's GW9 context and a backup of its `bot.sqlite` (copied that day), and v1 contexts for 2026/27 GW4 and GW8 built from the local `data/`; the registered user's team and 11 public FPL teams (recorded responses): 3/3 sessions identical, 303 steps (every command, both alerts, `/chips`, `/setft`, FPL down, after the deadline), 333 messages and every database row, 93 plans logged, no ties needed. The synthetic golden passes in `cargo test`.
- *Still open:* re-run the gate on the server against a fresh backup before the switch (needs the Rust build there, as batch 4); the live polling loop is exercised only by unit tests (routing, sending) until the cutover.

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
- *As built:* two crates, and the live path in `fplopt-bot`.
  - `fplopt-lgbm`: LightGBM v4.7.0 (the wheel's) from the pinned submodule `third_party/LightGBM`, built static with CMake (Release, OpenMP like the wheel; the source is copied into OUT_DIR because LightGBM's CMakeLists writes its libraries into the source tree). The C API is declared by hand and called in `lightgbm.train`'s order: the dataset and booster parameter strings as `_param_dict_to_str` writes them, label and weight as float32 (an all-ones weight dropped), one `UpdateOneIter` per round, the model saved to text and loaded back (what `train` returns). Model text byte-identical and predictions bit-identical to the wheel's on the minutes model's real training calls (69k–257k rows), on Windows (MSVC) and Linux (gcc, CI) alike; golden `tests/fixtures/rust_golden/lgbm.json.gz`.
  - `fplopt-models`: the baselines, `fitted` (refit cutoff), `team`, `gbm`, `minutes`, `availability`, `shares`, `components`, `calibration` (applies `calibration_table.json`, exported from Python's table by `tests/test_rust_golden.py`, which checks it stays current) and `assemble` (v1). `optim` ports scipy's trust-exact (Moré–Sorensen on Cholesky, its λ state across rejected steps), L-BFGS-B for the unbounded case (the L-BFGS step its Cauchy point and subspace minimization reduce to, the first-step and `stp = 1` rules, MINPACK-2's Moré–Thuente search) and MINPACK's `lmder` (`qrfac` with pivoting, `lmpar`, `qrsolv`, `enorm`), so fits follow Python's iterates: an exact solve of the saves curve would land 1e-6 from scipy's L-BFGS-B, the port lands within 1e-10. Sums follow what each Python line uses: numpy's pairwise sum (`np_sum`, `Series.sum` with NaN as 0), pandas' Kahan group sums, sequential `bincount` / `cumsum`; `np.round` is half to even, `logit` switches to `log1p` on [0.3, 0.65] as scipy's does, NaN propagates through `np.minimum` / `np.clip`, and frames write NaN as null as pandas does.
  - Found on the way: `club_continuity` (batch 3) used a fresh mean per window where pandas' `rolling().mean()` runs an online Kahan sum; the last bits differed, which batch 3's 1e-9 gate allowed but LightGBM's bins don't (other trees from 2018 on). `features::util::rolling_mean` is pandas' algorithm (3,000 random series identical), and every features frame now matches at rtol 0 (234/234).
  - `fplopt-bot`: `builder` (`build_context`, `ContextStore::write`, `raw_fingerprint`, `git_sha`, `next_gameweek`) and `live` (`next_event`, `due_alerts`, `context_stale`, `refresh_context`, `live_tick`). `fplopt-rs live tick|prepare` run them (`build all` in a subprocess, as Python). `check leakage` also covers the models (`model:<name>`; `team_match` loaded for them).
  - Gate tooling: `fplopt-rs models dump` (xP frames; each component's frame and fitted values; v1's fit at the refit cutoff) and `models context` (a context from `data/` without the rebuild), neither in the Python CLI; `dev/models_parity.py` (`--seasons`, `--plans`); `dev/bot_parity.py --rust-contexts`.
  - Also fixed: the bot golden's plan context was never committed (`*.parquet` is ignored repo-wide), so it passed locally and failed on CI; and on Linux HiGHS's reported `mip_gap` differs in its last digits between highspy and our build (the plans the same), so the golden and the bot gate compare it within 1e-9 relative.
- *Gate (2026-10-30):* on the local `data/` (built 2026-10-29).
  - Components (`dev/models_parity.py`, 18–26 deadlines each): the baselines 78/78 frames identical; team fits within 5.5e-8, frames 4.5e-9; minutes boosters byte-identical, frames exact; availability ~1e-14; shares fits 2.9e-14, frames 7.3e-10; components (saves fit) 1e-10.
  - Gates 1 and 2 (`--models v1 --seasons 2025,2026 --plans`, all 47 deadlines from 2025/26 GW1 to 2026/27 GW9): every fit within 1e-6 relative (worst 5.5e-8), xP within 1.4e-9 (limit 0.01).
  - Gate 3, template: plan #1 the same at 46/47 deadlines. At 2026/27 GW8 plan #1's moves and captains are the same and only its printed gain vs roll differs (the roll plan differs); at 2026/27 GW3 plan #1 swaps Thiago and Gonzalo between GW4 and GW5 (objective 247.39 vs 247.53). Both solves stop at the planner's 0.5% MIP gap after one node, and the difference is the solver's, not the port's: Python's own frame with ±1e-9 noise gives Python's plan for one seed and Rust's for two. The user counted it as a tie (2026-10-30): a plan that differs only where the solve stopped at its MIP gap passes gate 3, so the gate passes and Rust v1 is v1 (no `v1rs`).
  - Gate 3, registered users (`dev/bot_parity.py --rust-contexts`, the server's `bot.sqlite` backup and GW9 context, plus 2026/27 GW4, GW8 and GW9 from `data/`): 4/4 sessions, 404 steps, 441 messages and every row identical (plan-log numbers within 1e-9, `/chips` scenario values within the 0.5% gap, 16 ties at GW8/GW9).
  - Contexts built by both at GW4 and GW8: pool, players, prices and gameweeks identical, xP within 1.4e-9; Python reads the Rust contexts. `check leakage` (Rust) passes with the models: 18 deadlines × 13 frames.
- *Still open:* the gates on the server before the switch (Linux builds, a fresh `bot.sqlite` backup); `live tick`'s end-to-end tests (Python's synthetic league) not ported, the Rust side covered by unit tests and the bot gate; merging main for the chip PR's optimizer settings (`exclude_chips`, `fh_scale`, `doubles`, the finer hold-value keys, Free Hit ranking) and later the set-2 shadow logger (its `live tick` step and `bot.sqlite` table).

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
2. Batch 6 fallback: is a `v1rs` variant that passes the comparisons acceptable live, or must Rust match Python v1 exactly? *Not needed: the gate passed (2026-10-30); a plan #1 that differs only where both solves stopped at the 0.5% MIP gap counts as a tie.*
3. Batch 7: freeze random start states as a fixture (recommended) or port numpy's RNG?
4. Batch 8: keep notebooks as Python analysis over Parquet, or drop them?
5. One PR per batch (as with the earlier phases)? *Changed 2026-10-29: one branch, one switch (see Status).*
