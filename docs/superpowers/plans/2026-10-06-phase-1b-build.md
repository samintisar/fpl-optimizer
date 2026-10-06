# Phase 1b: Build Layer (raw → Parquet) Implementation Plan

> **For agentic workers:** Implement task-by-task (subagent-driven: one implementer per task, then review). Checkbox (`- [ ]`) steps. TDD: failing tests first (small synthetic raw stores / DataFrames), then implement; then run the builder on the real `raw/` copy locally and check the stated invariants.

**Goal:** `fplopt build all` turns `raw/` into validated Parquet tables under `data/`, for every season 2016/17 → 2026/27, with stable keys (`player_key` = FPL player `code`, `team_key` = FPL team `code`), `event_time` + `available_at` on every row, and loud failures when mapping or consistency checks fail. Phase 1 "done when": all seasons built from raw; mapping validation passes; `config/scoring/` per season (done in 1a).

**Architecture:** `fplopt.build` holds one module per table. Each builder is a pure-ish function `build_<table>(ctx) -> pd.DataFrame` that reads `raw/` (via `RawStore`) and/or earlier-built tables (via `ctx.table(name)`), and `write_table` validates it against a pandera schema and writes `data/<table>.parquet` atomically. `fplopt build <table>|all` runs builders in dependency order. Nothing is ever patched by hand: fixes go in code or in the two hand-maintained configs (`config/teams.csv`, `config/overrides.csv`).

**Tech Stack:** pandas 3.0 (copy-on-write; default `str` dtype), pyarrow 25 (zstd Parquet), pandera 0.34 (`import pandera.pandas as pa`), rapidfuzz, stdlib `zoneinfo`. No scipy in the build layer.

**Spec:** `docs/PLAN.md` §3 (Tables, Backfill rules, ID mapping, Data quality), §4 (leakage). Data facts below come from profiling the real `raw/` on 2026-10-06.

**Branch:** `phase-1b-build` (exists; PLAN.md data-fact edits uncommitted — commit in Task 0).

**Data location for dev:** a full copy of the server's `raw/` is at the repo root (`raw/`, gitignored). Builders default to `FPLOPT_RAW_DIR=raw`, `FPLOPT_DATA_DIR=data`. **Do not look at 2025/26 outcomes beyond structural checks** (holdout, CLAUDE.md) — counts, keys and nulls are fine; no metric/model work on it.

---

## Data facts (profiled 2026-10-06)

Raw inputs:
- vaastav run: newest dir under `raw/vaastav/data/` whose `_manifest.json.gz` has `failed == []` (currently `2026-10-06T032609Z`). Files: `<season>/gws/merged_gw.csv.gz`, `players_raw.csv.gz`, `player_idlist.csv.gz`, `cleaned_players.csv.gz`, `fixtures.csv.gz` (2018-19+), `teams.csv.gz` (2019-20+), `id_dict.csv.gz` (2021-22, 2022-23), `understat/*.csv.gz`; root `master_team_list.csv.gz`.
- Encodings: try strict `utf-8-sig`, fall back to `latin-1` (16-17…18-19 merged_gw, 16-17 cleaned_players, 19-20 understat_player are latin-1).
- `raw/fplcache/bootstrap-static/*.json.xz` (7,946, 2021-04-18 → 2026-10-05), `raw/fpl/bootstrap-static/*.json.gz`, `raw/fpl/fixtures/*.json.gz`, `raw/fpl/element-summary/<run>/` (+ `_manifest`), `raw/fpl/event-live/<season>/<gw>/`, `raw/odds/soccer_epl/*.json.gz`, `raw/football-data/E0/<YYZZ>/*.csv.gz`.

merged_gw (vaastav), per season rows 21.8k–29.8k; common columns `name, element, fixture, kickoff_time (ISO Z), round, GW, was_home, opponent_team, team_h_score, team_a_score, minutes, goals_scored, assists, clean_sheets, goals_conceded, own_goals, penalties_missed, penalties_saved, saves, yellow_cards, red_cards, bonus, bps, influence, creativity, threat, ict_index, total_points, value, selected, transfers_in/out/balance`. Extras: 16-17…18-19 `clearances_blocks_interceptions, recoveries, tackles` (+ other old detailed stats); 20-21+ `team` (short name), `position`, `xP` (**never read xP** — lookahead leak); 22-23+ `starts` (22-23 only populated from GW16), `expected_goals, expected_assists, expected_goal_involvements, expected_goals_conceded`; 24-25 `mng_*`, `modified`; 25-26 `clearances_blocks_interceptions, recoveries, tackles, defensive_contribution`. Column order varies — always select by name.
- `element` = `players_raw.id` (100% both ways). `round == GW`. `fixture` = fixtures.csv `id`.
- Team for a row: `team` only exists from 20-21 → derive uniformly from the fixture: `home_team_key if was_home else away_team_key` (validate against `team` where present).
- 52–62% of rows have minutes 0 (every registered player of a team with a fixture). Doubles: one row per fixture. Blanks: no rows.
- Quirks: 19-20 GW 1–29 then 39–47, plus 59 phantom GW29 rows for fixture 275 (0 min, old kickoff) duplicating the real GW39 rows → keep the row whose GW equals the fixture's event. 25-26: 10 exact duplicate rows → drop exact duplicates. 22-23 has no GW7. 24-25 assistant managers (`position == "AM"`, `element_type == 5`) → drop. 21-22 GW37 position label "GKP" vs "GK" elsewhere.
- `value` is tenths of £m.

players_raw: `id, code, first_name, second_name, web_name, team, team_code, element_type` every season; `opta_code` (= `"p"+code`), `birth_date`, `team_join_date` from 24-25; `known_name` 25-26. `code` and `id` unique per season. `team`→`team_code` is 1:1 per season (gives team codes for 16-17…18-19). `team` is end-of-season team.

Teams: FPL team `code` is stable per club across all seasons (35 clubs 2016-17 → 2026-27; mapping table in Task 2). FPL team `id` resets each season. FPL `name` changes (code 40 "Ipswich" → "Ipswich Town"; 2026-27 "Coventry City", "Hull City").

fixtures.csv (18-19+): `id, code, event, kickoff_time, team_h, team_a, team_h_score, team_a_score, finished, …` 380 rows, no nulls. 18-19 also has deadline columns. 16-17/17-18: derive fixtures from merged_gw (`fixture, kickoff_time, was_home, opponent_team, team_h_score, team_a_score, round`). Every season: all 380 fixtures have both sides present in merged_gw.

football-data: 380 rows/season (2627: 50 so far), UTF-8 (BOM in some) — decode `utf-8-sig`. `Date` `dd/mm/yy` (1617) else `dd/mm/yyyy`. Joins to FPL fixtures on (UK kickoff date, home, away) for all 3,090 played matches with 0 date mismatches; take kickoff times from FPL (football-data `Time` is off by 10–30 min in 7 cases). Odds allowlist in Task 8.

Understat player files (vaastav `understat/<Name>_<understat_id>.csv`, in 21-22…24-25 folders): columns `goals, shots, xG, time, position, h_team, a_team, h_goals, a_goals, date (YYYY-MM-DD), id (understat match id), season (start year), roster_id, xA, assists, key_passes, npg, npxG, xGChain, xGBuildup`. Career logs incl. **other leagues** (filter to EPL by team names). Same (understat_id, match) appears in several folders with equal values (±5e-6) → dedupe keeping the newest folder. No player-side column (side = the EPL team the player's FPL record says). Names may contain HTML entities (`&#039;`). EPL coverage of FPL minutes: 2016 47%, 2017 62%, 2018 75%, 2019 91%, 2020–2023 100%, 2024 80% (stops 2025-04-07).
Understat team files (`understat_<Team>.csv`, 19-20…24-25): `h_a, xG, xGA, npxG, npxGA, scored, missed, date (YYYY-MM-DD HH:MM:SS, ~UTC), …` one row per team match; each folder also carries ~3 stale previous-season files → filter by the season's date window. Team names in file names use `_` for spaces.
id_dict (21-22, 22-23): `Understat_ID, FPL_ID, Understat_Name, FPL_Name` (strip header whitespace) — 100% coverage of minutes>0 players; use as a **validation oracle**.

fplcache bootstrap elements: keys grow over time (`expected_goals` 2022-11, `team_join_date` 2024-12, `opta_code` 2025-01, …). `now_cost` int tenths; `selected_by_percent`, `ep_next`, `ep_this`, `form`, `expected_*` are **strings**; `chance_of_playing_*` null = no flag. 2024-25 Jan–Jul snapshots contain managers (`element_type` 5) → drop. Every snapshot has 38 events, 20 teams, non-empty elements; season = `bootstrap_season(...)`, never the timestamp (307 snapshots disagree with the calendar rule). ~5.7M player rows total; ~33 ms/snapshot to parse → ~4–5 min single-threaded.

element-summary `history` rows (2026-27): per fixture; fields as merged_gw plus `defensive_contribution`, `clearances_blocks_interceptions`, `recoveries`, `tackles`, `starts`, `expected_*` (strings), `modified`.

Odds API: list of events `{id, commence_time, home_team, away_team, bookmakers:[{key, title, last_update, markets:[{key, outcomes:[{name, price, point?}]}]}]}`; markets `h2h`, `totals` (points 2.5 and 3.5 — keep 2.5), unrequested `h2h_lay` (drop). Full club names ("Brighton and Hove Albion").

---

## Conventions

- Keys: `season` = start year (int). `team_key` = FPL team code (int; synthetic ≥ 1000 for pre-2016 clubs). `player_key` = FPL player code (int). `fixture_key` = `season * 1000 + fpl_fixture_id` (int, e.g. 2016001).
- Times: tz-aware UTC pandas timestamps (`datetime64[us, UTC]`). UK-local logic via `zoneinfo.ZoneInfo("Europe/London")`.
- Every table has `event_time` and `available_at` (UTC). `available_at` must never be earlier than what could really have been known.
- GW lockdown (`lockdown_time`): 09:00 Europe/London on the day after the UK-local date of the GW's last kickoff.
- Table schemas: pandera `DataFrameSchema` per table, `strict=True`, explicit dtypes and nullability, unique-key checks. Validation failure = build failure.
- Parquet: zstd, written to `<name>.parquet.tmp` then `os.replace`. Rows sorted by the table's key for deterministic output; re-running a build on the same raw gives byte-identical files (test this for one small table).
- Code style as in Phase 0/1a; module docstrings; tests flat in `tests/` (`tests/test_build_<module>.py`). Commit trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## File structure

| File | Responsibility |
|---|---|
| `src/fplopt/build/common.py` | `BuildContext`, raw CSV reading with encoding fallback, latest complete run dirs, `write_table`, time helpers (`lockdown_time`, UK date) |
| `src/fplopt/build/teams.py` | `team_dim` from `config/teams.csv`; name→key resolvers per source; coverage validation |
| `src/fplopt/build/fixtures.py` | `fixture`, `gameweek` |
| `src/fplopt/build/players.py` | `player_dim`, `player_season`, `player_match` |
| `src/fplopt/build/understat.py` | `understat_map` (understat_id ↔ player_key), Understat columns on `player_match`, `team_match` |
| `src/fplopt/build/snapshots.py` | `player_snapshot` |
| `src/fplopt/build/odds.py` | `odds_snapshot` |
| `src/fplopt/build/elo.py` | `team_rating` |
| `src/fplopt/build/__init__.py` | `BUILDERS` registry + order, `build(names, ctx)` |
| `config/teams.csv` | Hand-maintained club mapping (Task 2) |
| `config/overrides.csv` | Existing header `source,source_id,season,player_key,note` — Understat mapping overrides |

---

### Task 0: Commit PLAN.md data facts + this plan

- [ ] `git add docs/PLAN.md docs/superpowers/plans/2026-10-06-phase-1b-build.md && git commit -m "docs: Phase 1b plan; PLAN data facts from profiling raw/"`

---

### Task 1: Build infrastructure + CLI

**Files:** `src/fplopt/build/common.py`, `src/fplopt/build/__init__.py`, `src/fplopt/settings.py` (+ `data_dir` from `FPLOPT_DATA_DIR`, default `data`, resolved like `raw_dir`), `src/fplopt/cli.py`, tests `tests/test_build_common.py`, `tests/test_cli.py`, `tests/test_settings.py` if it exists.

- `BuildContext(store: RawStore, data_dir: Path, config_dir: Path = Path("config"))` with `table(name) -> pd.DataFrame` (reads `data/<name>.parquet`; raises a clear error naming the builder to run first) and a small in-memory cache.
- `read_raw_csv(path) -> pd.DataFrame`: `RawStore.read_bytes`, decode strict `utf-8-sig`, fallback `latin-1`; strip whitespace from column names; `pd.read_csv(io.StringIO(text))`.
- `latest_complete_run(store, source, endpoint) -> Path`: newest timestamp-named run dir under `<source>/<endpoint>/` whose `_manifest.json.gz` has `failed == []` and `written` == `expected` (vaastav and element-summary manifests both have these). Raise `LookupError` if none.
- `write_table(df, name, schema, data_dir, sort_by) -> Path`: validate (`schema.validate(df, lazy=True)` — on failure raise with a readable summary of failure cases, first ~20), sort, `df.to_parquet(tmp, compression="zstd", index=False)`, `os.replace`. Return path.
- `lockdown_time(last_kickoff_utc: pd.Timestamp) -> pd.Timestamp` and a vectorised variant over a Series; `uk_date(ts)`.
- `fplcache_and_own_bootstraps(store) -> list[tuple[datetime, Path, str]]` (snapshot time, path, source `fplcache`|`fpl`) sorted by time — used by gameweek, player_dim (2026-27) and player_snapshot.
- `BUILDERS: dict[str, Callable[[BuildContext], pd.DataFrame]]` + `ORDER` list; `build(names, ctx)` runs each, writes, logs `table: rows, seasons, seconds`. Unknown name → error listing valid names.
- CLI: `fplopt build TABLE|all` (group `build`, positional `target`), job `lambda c: build(...)` using `BuildContext(c.store, c.settings.data_dir)`. Builders don't need HTTP; the Context still creates the client — fine.

Tests: encoding fallback (latin-1 bytes `b"name\nJos\xe9\n"` gzipped), BOM header stripped, whitespace headers stripped; `latest_complete_run` skips a newer failed run and non-timestamp dirs; `write_table` writes, rejects an invalid frame with a message naming the column, output is byte-identical on rewrite; `lockdown_time` for a Sunday 16:30 BST kickoff → Monday 08:00 UTC, and for a GMT date (January) → 09:00 UTC; CLI `build fixture` dispatches; `build nope` → exit 1.

Commit: `feat(build): build context, table writer, build CLI`.

---

### Task 2: `team_dim` + `config/teams.csv` (+ football-data back to 2005/06 for Elo)

**Files:** `config/teams.csv`, `src/fplopt/build/teams.py`, `tests/test_build_teams.py`; `src/fplopt/ingest/history.py` + `src/fplopt/cli.py` (football-data `--from-season`).

1. Extend `backfill_football_data(store, fd, now, sleep, pause_s, first_season=FIRST_SEASON)` and add CLI option `fplopt backfill football-data --from-season 2005` (default 2016). Pre-2016 files are only used for Elo burn-in. Test the range. (Running it on the server + re-copying raw/ is done by the controller, not the implementer.)
2. `config/teams.csv` columns: `team_key,short_name,fpl_names,football_data,understat,odds_api`. `fpl_names` = every FPL `name` seen for that code, `;`-separated. Understat names with spaces (file names' `_` → space). Odds API names only where observed — leave blank otherwise (validation only requires names that actually occur). Seed rows (FPL code | short | FPL names | football-data | Understat | Odds API):
   ```
   3 ARS Arsenal | Arsenal | Arsenal | Arsenal
   7 AVL Aston Villa | Aston Villa | Aston Villa | Aston Villa
   91 BOU Bournemouth | Bournemouth | Bournemouth | Bournemouth
   94 BRE Brentford | Brentford | Brentford | Brentford
   36 BHA Brighton | Brighton | Brighton | Brighton and Hove Albion
   90 BUR Burnley | Burnley | Burnley | Burnley
   97 CAR Cardiff | Cardiff | Cardiff | 
   8 CHE Chelsea | Chelsea | Chelsea | Chelsea
   9 COV Coventry City | Coventry | | Coventry City
   31 CRY Crystal Palace | Crystal Palace | Crystal Palace | Crystal Palace
   11 EVE Everton | Everton | Everton | Everton
   54 FUL Fulham | Fulham | Fulham | Fulham
   38 HUD Huddersfield | Huddersfield | Huddersfield | 
   88 HUL Hull;Hull City | Hull | Hull | Hull City
   40 IPS Ipswich;Ipswich Town | Ipswich | Ipswich | Ipswich Town
   2 LEE Leeds | Leeds | Leeds | Leeds United
   13 LEI Leicester | Leicester | Leicester | 
   14 LIV Liverpool | Liverpool | Liverpool | Liverpool
   102 LUT Luton | Luton | Luton | 
   43 MCI Man City | Man City | Manchester City | Manchester City
   1 MUN Man Utd | Man United | Manchester United | Manchester United
   25 MID Middlesbrough | Middlesbrough | Middlesbrough | 
   4 NEW Newcastle | Newcastle | Newcastle United | Newcastle United
   45 NOR Norwich | Norwich | Norwich | 
   17 NFO Nott'm Forest | Nott'm Forest | Nottingham Forest | Nottingham Forest
   49 SHU Sheffield Utd | Sheffield United | Sheffield United | 
   20 SOU Southampton | Southampton | Southampton | 
   110 STK Stoke | Stoke | Stoke | 
   56 SUN Sunderland | Sunderland | Sunderland | Sunderland
   80 SWA Swansea | Swansea | Swansea | 
   6 TOT Spurs | Tottenham | Tottenham | Tottenham Hotspur
   57 WAT Watford | Watford | Watford | 
   35 WBA West Brom | West Brom | West Bromwich Albion | 
   21 WHU West Ham | West Ham | West Ham | 
   39 WOL Wolves | Wolves | Wolverhampton Wanderers | 
   ```
   Verify every cell against the data (FPL names per code from players_raw/teams.csv/bootstraps; Understat names from player-file `h_team`/`a_team` and team-file names, which may differ from the guesses above, e.g. 2016–18 clubs; football-data names). Then add synthetic rows (`team_key` 1001, 1002, … in alphabetical order of name, `short_name` 3 letters, `fpl_names` empty) for every football-data club that only appears in 2005/06–2015/16 (Wigan, Blackburn, Bolton, Portsmouth, Birmingham, Blackpool, QPR, Reading, Derby, Charlton, …) — derive the list from the backfilled CSVs once they exist; until then the implementer adds the rows the controller provides.
3. `build_team_dim(ctx)`: read `config/teams.csv` → columns `team_key:int, short_name, fpl_names (str, ';'-joined), football_data_name, understat_name (nullable), odds_api_name (nullable), in_fpl:bool`; `event_time`/`available_at` = constant `1970-01-01 UTC` (static reference data).
4. Resolvers in `teams.py`: `TeamResolver(team_dim)` with `.football_data(name)`, `.understat(name)`, `.odds_api(name)`, `.fpl_code(code)` — each raises `KeyError` listing the unknown name.
5. `validate_team_coverage(ctx)` (called by `build_team_dim`, fails the build): every FPL team code in vaastav players_raw `team_code` and bootstrap `teams[].code` exists; every football-data `HomeTeam`/`AwayTeam` (all seasons present in raw) resolves; every Understat EPL team name observed in the team-file names resolves; every Odds API `home_team`/`away_team` resolves. Report all unknowns at once.

Tests: CSV parsing → frame types; resolver lookups incl. `Nott'm Forest`; coverage validation passes on a tiny synthetic store and fails listing two unknown names; football-data backfill `first_season`.

Commit(s): `feat(ingest): football-data backfill from an earlier season` and `feat(build): team_dim from config/teams.csv with cross-source coverage checks`.

---

### Task 3: `fixture` + `gameweek`

**Files:** `src/fplopt/build/fixtures.py`, `tests/test_build_fixtures.py`.

`fixture` (one row per EPL match 2016/17 → 2026/27):
`fixture_key:int, season:int, fpl_fixture_id:int, gw:Int64 (nullable — unscheduled), gw_index:Int64 (dense rank of gw within season: 2019/20 39–47 → 30–38), kickoff_time:UTC (nullable only if unscheduled), home_team_key:int, away_team_key:int, home_goals:Int64, away_goals:Int64, finished:bool, fd_date:date (nullable), event_time (= kickoff_time), available_at`. `available_at` = the lockdown_time of the fixture's GW (when its result is final; future for unplayed fixtures). Unscheduled fixtures (no gw/kickoff) get `2100-01-01 UTC` and null results. The schedule itself being known earlier is Phase 2's concern (`fixture_snapshot`).
Sources:
- 2016-17, 2017-18: from merged_gw rows: per `fixture` id take `kickoff_time`, `round`; home team = the team whose rows have `was_home` True. A team's id = the `opponent_team` of the other side's rows; map season team id → code via players_raw `team`→`team_code`. Scores `team_h_score`/`team_a_score`.
- 2018-19 … 2025-26: vaastav `fixtures.csv` (`id, event, kickoff_time, team_h, team_a, team_h_score, team_a_score, finished`); team id → code via `teams.csv` (2019-20+) or players_raw mapping (2018-19).
- 2026-27: newest `raw/fpl/fixtures` snapshot + team id → code from the newest `raw/fpl/bootstrap-static` snapshot (same season check via `bootstrap_season`).
- Join football-data (all seasons present) on (season, UK kickoff date, home_team_key, away_team_key) for finished matches → `fd_date`. **Fail** if a finished FPL fixture has no football-data row (except 2026-27 rows after the latest football-data snapshot's last date), if a football-data row is unmatched, or if goals differ.
- Validation: 380 fixtures per season; unique `fixture_key`; `(season, home, away)` unique; each team plays 38.

`gameweek` (`season, gw, gw_index, deadline_time, deadline_source ('bootstrap'|'fixtures_csv'|'approx'), first_kickoff, last_kickoff, lockdown_time, average_entry_score:Int64, event_time (= deadline_time), available_at (= deadline_time; average_entry_score is only known after lockdown — store it in a separate column `average_entry_score_available_at` = lockdown_time)`):
- Deadlines + average_entry_score from the newest bootstrap of each season (own archive for 2026-27, fplcache for 2020-21 … 2025-26; 2020-21 only if its final snapshot is present — fplcache starts 2021-04-18, mid 2020-21, which still lists all 38 events).
- 2018-19: deadlines from fixtures.csv deadline columns if present (check names), else approx.
- 2016-17, 2017-18, 2019-20 (if no bootstrap): `deadline_time = first_kickoff − 90 min`, `deadline_source = 'approx'`.
- first/last kickoff from `fixture` (finished + scheduled rows of that gw).
- Validation: gw count per season (38; 2022-23 has 37 GWs in vaastav — check how FPL's own bootstrap numbers 2022-23 and document the result; don't invent a GW7 row), deadlines strictly increasing within a season, deadline < first_kickoff.

Tests: synthetic merged_gw-derived fixtures (two teams, two fixtures), id→code mapping, home detection, football-data join failure on a goal mismatch, `gw_index` for 39–47 numbering, approx deadline, lockdown.

Commit: `feat(build): fixture and gameweek tables with football-data cross-check`.

---

### Task 4: `player_dim`, `player_season`, `player_match`

**Files:** `src/fplopt/build/players.py`, `tests/test_build_players.py`.

`player_season` (`player_key, season, element_id, element_type (1–4), team_key (end of season / latest), first_name, second_name, web_name, event_time, available_at`): vaastav players_raw per season (drop element_type 5) + 2026-27 from the newest own bootstrap (`elements[].code`, `team_code`). event_time = available_at = season's first deadline (the registration is known by then — approximate; document).
`player_dim` (`player_key, first_name, second_name, web_name (newest season), opta_code (= "p"+code), understat_id:Int64 (filled by Task 5; null here), first_season, last_season, event_time, available_at` = static epoch like team_dim).

`player_match` (one row per player per fixture the player's team played while registered):
`player_key, season, fixture_key, gw, element_id, team_key, opponent_team_key, was_home, kickoff_time, minutes, starts:Int64, goals_scored, assists, clean_sheets, goals_conceded, own_goals, penalties_saved, penalties_missed, yellow_cards, red_cards, saves, bonus, bps, total_points, clearances_blocks_interceptions:Int64, recoveries:Int64, tackles:Int64, defensive_contribution:Int64, fpl_xg:Float64, fpl_xa:Float64, fpl_xgc:Float64, source ('vaastav'|'fpl'), event_time (= kickoff_time), available_at (= GW lockdown_time from gameweek)`.
- Not included on purpose: `value, selected, transfers_*` (timing unverified, issue #6), `xP` (leak), ICT.
- `starts`: null where the source lacks it (pre-22-23; 22-23 GW1–15).
- 2016-17 … 2025-26 from vaastav merged_gw; 2026-27 from the newest complete element-summary run (`latest_complete_run(store, "fpl", "element-summary")`, element → code via the bootstrap archived at that run's `run_at`: `raw/fpl/bootstrap-static/<run_at>.json.gz`; fall back to the newest bootstrap with the same season if absent). `expected_*` strings → floats.
- Dedupe: drop exact duplicate rows; for duplicated `(element, fixture)` keep the row whose `round` equals the fixture's `gw` (19-20 phantom GW29); any remaining duplicate → fail.
- Team: from the fixture: `home_team_key if was_home else away_team_key`; validate against merged_gw `team` short name (20-21+) via team_dim `fpl_names`/short names, and that `opponent_team` maps to the other side.
- Validation: unique `(player_key, fixture_key)`; every `fixture_key` in `fixture`; minutes in 0–120 (stoppage time; check real max and set bound accordingly); every element maps to a `player_key`; per fixture, sum of goals_scored + own goals of the other side == team goals (allow own-goal attribution: home_goals == sum(home players' goals_scored) + sum(away players' own_goals)) — report mismatching fixtures; fail if more than a handful (document real count; vaastav may have known gaps).

Tests: synthetic merged_gw with a phantom duplicate, an exact duplicate, a manager row; team derivation; element-summary path with string xG; goal-sum check failure.

Commit: `feat(build): player_dim, player_season and player_match`.

---

### Task 5: Understat mapping, xG columns, `team_match`

**Files:** `src/fplopt/build/understat.py`, `tests/test_build_understat.py`; update `player_dim` (understat_id) and `player_match` (us_* columns) — implement as a step that re-writes both tables after mapping (builder `understat` in ORDER after `player_match`).

1. `understat_player_match` (intermediate table, also written for inspection): from the vaastav run's `*/understat/*.csv.gz` player files (exclude `understat_*.csv` team/aggregate files). `understat_id` from the filename suffix. Keep EPL rows only: both `h_team` and `a_team` resolve via `TeamResolver.understat`; map to `fixture_key` by `(season, home_team_key, away_team_key)` (unique per season); drop rows whose fixture doesn't exist and report the count. Dedupe `(understat_id, us_match_id)` keeping the newest folder. Columns: `understat_id, fixture_key, us_match_id, us_minutes (time), us_goals, us_npg, us_xg, us_npxg, us_xa, us_assists, us_shots, us_key_passes, us_position`, names (`html.unescape`).
2. Mapping `understat_id → player_key`:
   - Candidate pairs: an Understat player-fixture and an FPL player_match row on the same fixture, FPL minutes > 0, and the FPL player's team is one of the fixture's teams (always true) — restrict to pairs where the Understat player's appearance and the FPL appearance are in the same fixture.
   - Pair score over **all** seasons: `shared` = fixtures where both appear with |us_minutes − minutes| ≤ 5; `overlap = shared / min(us_apps, fpl_apps_in_understat_covered_fixtures)`; `name` = rapidfuzz `token_set_ratio` on normalised names (html-unescape, NFKD strip accents, lowercase; FPL full name vs Understat name, also try `web_name`).
   - Greedy one-to-one assignment by (overlap, shared, name) descending. Accept if (`overlap ≥ 0.8` and `shared ≥ 3`) or (`overlap ≥ 0.5` and `name ≥ 85`) or (`shared ≥ 1` and `name ≥ 95`). Thresholds are starting points: tune so that agreement with id_dict (oracle, 2021-22 + 2022-23) is **100%** and coverage is maximal; record final thresholds + results in the module docstring.
   - Overrides: `config/overrides.csv` rows with `source=understat` (`source_id` = understat id, `player_key`; `player_key` empty = "no mapping") are applied last and win.
   - Write `understat_map` (`understat_id, player_key, method ('auto'|'override'), shared, overlap, name_score`).
3. Validation (fails the build, listing offenders): every FPL player with minutes > 0 in a fixture where Understat has rows for that player's team in that fixture, in seasons 2020–2023, must be mapped (or overridden to none). One understat_id per player_key and vice versa. id_dict agreement 100%.
4. Join: `player_dim.understat_id`; `player_match` gets `us_minutes, us_goals, us_npg, us_xg, us_npxg, us_xa, us_shots, us_key_passes` (nullable) by `(player_key, fixture_key)`. Report coverage per season.
5. `team_match` (`fixture_key, team_key, season, is_home, goals_for, goals_against, us_xg, us_xga, us_npxg, us_npxga (Understat team files), fd_xg, fd_xga (football-data HxG/AxG, 2026-27), fpl_xg, fpl_xga (sum of player fpl_xg per side, 2022-23+; null if any player row lacks it), event_time (kickoff), available_at (lockdown)`): Understat team rows → keep rows inside the season's window (first–last kickoff of that season ± 1 day), join on `(team_key, is_home = h_a=="h", UTC date of kickoff)`; validate `scored`/`missed` == goals; report unmatched. Two rows per fixture.

Tests: synthetic files incl. a non-EPL row, a duplicate across folders, an HTML-entity name; mapping picks the minutes-consistent candidate over a similar name (Kyle Walker vs Walker-Peters); override wins; unmapped covered player fails validation; team_match join and goal check.

Commit: `feat(build): Understat↔FPL mapping, xG columns and team_match`.

---

### Task 6: `player_snapshot`

**Files:** `src/fplopt/build/snapshots.py`, `tests/test_build_snapshots.py`.

Columns: `snapshot_at (UTC), source ('fplcache'|'fpl'), season, element_id, player_key, team_key, element_type, now_cost, status, chance_of_playing_next_round:Int64, chance_of_playing_this_round:Int64, news, news_added (UTC, nullable), selected_by_percent:Float64, ep_next:Float64, ep_this:Float64, form:Float64, penalties_order:Int64, corners_and_indirect_freekicks_order:Int64, direct_freekicks_order:Int64, team_join_date (date, nullable), transfers_in_event, transfers_out_event, cost_change_event, event_time (= snapshot_at), available_at (= snapshot_at)`. Missing keys in older snapshots → null.
- All fplcache snapshots + all own bootstrap snapshots (both kept; `source` distinguishes; overlap from 2026-10-05).
- Drop `element_type == 5`. `team_key` via that snapshot's `teams[]` id → code.
- Memory: parse per snapshot into column lists/arrow arrays; write with `pyarrow.parquet.ParquetWriter` one row group per season (or every ~500 snapshots) so peak memory stays well under 2 GB; validate each chunk with the pandera schema before writing. Optional `ProcessPoolExecutor` for parsing (keep a `jobs=1` path for tests).
- Validation: unique `(snapshot_at, source, element_id)`; every `player_key` in `player_dim` (2021-22+ players should all be in players_raw/bootstraps — report any missing); seasons present 2020…2026.
- Log total rows and seconds.

Tests: two synthetic snapshots (old schema without `team_join_date`, new with), string→float conversion, manager dropped, team id→code.

Commit: `feat(build): player_snapshot from fplcache and own bootstraps`.

---

### Task 7: `odds_snapshot`

**Files:** `src/fplopt/build/odds.py`, `tests/test_build_odds.py`.

Long format: `fixture_key, season, source ('football-data'|'odds-api'), bookmaker, market ('h2h'|'totals'|'ah'), outcome ('home'|'draw'|'away'|'over'|'under'), line:Float64 (totals 2.5; AH home handicap; null for h2h), price:float, is_closing:bool, snapshot_at (UTC), event_time (= kickoff), available_at`.

football-data (**explicit allowlist — never regex**; era A 1617–1819, era B 1920+):

| canonical (bookmaker `avg`) | era A | era B |
|---|---|---|
| h2h home/draw/away | `BbAvH, BbAvD, BbAvA` | `AvgH, AvgD, AvgA` |
| totals over/under (line 2.5) | `BbAv>2.5, BbAv<2.5` | `Avg>2.5, Avg<2.5` |
| ah line / home / away | `BbAHh, BbAvAHH, BbAvAHA` | `AHh, AvgAHH, AvgAHA` |
| closing (era B only) | — | `AvgCH, AvgCD, AvgCA`, `AvgC>2.5, AvgC<2.5`, `AHCh` + `AvgCAHH, AvgCAHA` |

Extra bookmakers: `pinnacle`: `PSH, PSD, PSA` (+ `P>2.5, P<2.5, AHh`-paired `PAHH, PAHA` in era B), closing `PSCH, PSCD, PSCA` (all eras), `PC>2.5, PC<2.5`, `PCAHH, PCAHA` (paired with `AHCh`); `betfair_ex`: `BFEH, BFED, BFEA, BFE>2.5, BFE<2.5, BFEAHH, BFEAHA` (+ `BFEC*` closing) where present. **Traps:** `CLH/CLD/CLA` = Coral (not closing); `BFH/BFD/BFA` (2425) = Betfair sportsbook; `BFDH/BFDD/BFDA` = Betfred; `BFCH` = Betfair sportsbook closing — none are in the allowlist. Closing AH prices pair with `AHCh`, never `AHh`. Skip null cells.
- `available_at` for pre-match football-data odds: UK-local kickoff weekday Tue/Wed/Thu → that week's Tuesday 15:00 UK; Fri/Sat/Sun/Mon → the Friday on or before 15:00 UK; then `min(that, kickoff − 1h)`. Closing odds: `available_at = kickoff_time` (never usable before kickoff). `snapshot_at = available_at`.
- Odds API: every `raw/odds/soccer_epl/*.json.gz`: events → `fixture_key` via `(season, home, away)` names through `TeamResolver.odds_api` (season of the event = `season_start_year(commence_time)`); unmatched events (e.g. not yet in the fixture table) → reported, skipped. Markets: `h2h` (outcome by name: home team / away team / "Draw"), `totals` with `point == 2.5` only; drop `h2h_lay`. `bookmaker` = key; `snapshot_at = available_at` = file timestamp; `is_closing = False`.
- Validation: unique `(fixture_key, source, bookmaker, market, outcome, line, is_closing, snapshot_at)`; prices > 1.0; per fixture/bookmaker/snapshot, h2h implied probabilities sum within [1.0, 1.2] (report outliers, fail if > 1%).

Tests: era A and era B rows, Coral column ignored, AH closing pairing, available_at for Sat 12:30 and Wed 19:45 kickoffs, Odds API h2h + totals parsing with a 3.5 line dropped and `h2h_lay` dropped.

Commit: `feat(build): odds_snapshot from football-data allowlist and The Odds API`.

---

### Task 8: `team_rating` (Elo)

**Files:** `src/fplopt/build/elo.py`, `tests/test_build_elo.py`.

- Input: football-data results for every season present (2005/06+ after the backfill), parsed `Date` (two formats), `HomeTeam, AwayTeam, FTHG, FTAG`, teams via `TeamResolver.football_data`. For 2016/17+ attach `fixture_key` and FPL `kickoff_time` via the `fixture` table; earlier matches: kickoff = `Date` 15:00 UK.
- Model (World-Football-Elo style, parameters as module constants, documented; tuned later in Phase 5): initial 1500 for every club in the first season; home advantage `H = 60`; `K = 20`; goal-difference multiplier `G = 1` (|d| ≤ 1), `1.5` (2), `(11 + |d|) / 8` (≥ 3); `E_home = 1 / (1 + 10^(−(R_h + H − R_a)/400))`; draws S = 0.5. Season rollover: clubs promoted into the league start at the mean end-of-season rating of the clubs relegated the season before (first season: 1500). No mean reversion.
- Output rows, two per match: `team_key, season, fixture_key (nullable pre-2016), kickoff_time, opponent_team_key, is_home, rating_before, rating_after, expected_score, event_time (= kickoff_time), available_at (= kickoff_time + 2h)`.
- Deterministic: process matches sorted by (kickoff, home_team_key).
- Validation: every club's ratings continuous within a season; total rating change per match sums to 0; the mean rating of the 20 league clubs stays within 1500 ± 50 in 2016/17+ (sanity — report the value).

Tests: one match updates both sides symmetrically; draw between equal teams with home advantage lowers the home rating; G multiplier; promoted-club seeding.

Commit: `feat(build): Elo team_rating with promoted-club seeding`.

---

### Task 9: `build all`, docs, full local build, PR

- [ ] `ORDER` = `team_dim, fixture, gameweek, player_season, player_dim, player_match, understat, team_match, player_snapshot, odds_snapshot, team_rating` (adjust to the real dependencies).
- [ ] Run `uv run fplopt build all` on the local `raw/` copy; record per-table rows, size and time in the PR. Fix real-data failures in code (or `config/teams.csv` / `config/overrides.csv` with a `note`), never by editing data.
- [ ] Docs: CLAUDE.md commands (`uv run fplopt build all|<table>`), README (tables), PLAN §3 Tables (any schema changes), `deploy/README.md` (where to run builds: dev machine after copying raw/; the server doesn't need data/ yet).
- [ ] `uv run pytest`, ruff check/format clean. PR "Phase 1b: build layer", independent review, fix, merge. Close the Phase 1 milestone issues that are done.
