# Phase 1a: Raw Backfills Implementation Plan

> **For agentic workers:** Implement task-by-task (subagent-driven: one implementer per task, then spec + quality review). Steps use checkbox (`- [ ]`) syntax for tracking. TDD: write the failing tests first, watch them fail, implement, watch them pass, commit.

**Goal:** Get every historical source into the append-only `raw/` store on the server, so Phase 1b can build Parquet tables purely from `raw/`:
- football-data.co.uk EPL CSVs (2016/17 → current; current season re-fetched daily);
- vaastav/Fantasy-Premier-League at a pinned commit (per-GW stats, player lists, its Understat mirror);
- Randdalf/fplcache `bootstrap-static` snapshots (April 2021 → today), stored byte-for-byte;
- recurring post-lockdown FPL ingest (`event/{gw}/live/` + a fresh `element-summary` run per finished GW) — GitHub issue #17;
- `config/scoring/<season>.json` exported from archived bootstraps (2025-26 and 2026-27).

**Architecture:** `RawStore` gains `write_bytes` (verbatim bytes with a compression suffix: `.json.gz`, `.csv.gz`, `.json.xz`) next to the existing JSON `write`; reads dispatch on suffix. HTTP gains a generic `get_response(..., validate=...)`; `get_json_response` becomes a thin wrapper. New thin adapters return raw bytes. One-off backfills live in `fplopt.ingest.history`; recurring steps stay in `fplopt.ingest.jobs` (`run_daily` gains a football-data step and a post-lockdown step). The CLI passes a `Context` to jobs instead of `(store, fpl, odds)`.

**Tech Stack:** Python 3.12 stdlib (`gzip`, `lzma`, `tarfile`, `hashlib`), httpx (+ `MockTransport`), tenacity, pytest, ruff.

**Spec:** `docs/PLAN.md` §3 (sources, rules config, backfill rules), §12 (2026-10-06 Understat decision). GitHub issues #15, #17.

**Branch:** `phase-1a-raw-backfill` (already created; PLAN.md Understat update is uncommitted on it — commit it in Task 0).

**Research facts this plan relies on (checked 2026-10-06):**
- vaastav: default branch `master`, pinned commit `9779cdbc0c07f6c900c2d0c181ddf6bb9c800f88`. Git tree API (`/repos/vaastav/Fantasy-Premier-League/git/trees/<sha>?recursive=1`) is not truncated (26,675 entries). 2016-17…2018-19 gw CSVs are latin-1 — irrelevant here, raw bytes are stored verbatim. `fixtures.csv` exists from 2018-19, `teams.csv` from 2019-20, `id_dict.csv` only 2021-22/2022-23, `understat/` 2019-20…2024-25. 2026-27 folder is stale (GW1 only) → not mirrored; our own archive covers 2026/27.
- fplcache: branch `main`, files `cache/{year}/{month}/{day}/{HHMM}.json.xz` (month/day not zero-padded, UTC), ~7,950 files, ~880 MB, xz-compressed pretty JSON, not Git LFS. Bootstrap-static only. Public domain.
- football-data: `https://www.football-data.co.uk/mmz4281/{YYZZ}/E0.csv` (302 → `https://football-data.co.uk/...`; client follows redirects). Header row starts with `Div,` (2526+ has a UTF-8 BOM). 2627 exists (in-progress season).
- Live bootstrap: `events[]` have `id`, `finished`, `data_checked`, `deadline_time`, `overrides` (`{"rules": {}, "scoring": {}, "element_types": [], "pick_multiplier": null}` when empty). Top-level `game_config` = `{rules, scoring, settings, status}`, `chips[]` with `start_event`/`stop_event`, `element_types[]`.
- Existing element-summary manifest keys: `run_at, finished_at, expected, written, failed`.

---

## File structure

| File | Responsibility |
|---|---|
| `src/fplopt/seasons.py` (new) | Season labels/codes; season of a date; season of a bootstrap payload |
| `src/fplopt/ingest/raw_store.py` | + `write_bytes`, `path_for`, public `entries`, suffix-aware `times`/`latest`, `read_bytes`; `gzip_bytes` helper |
| `src/fplopt/adapters/http.py` | + `get_response(validate=)`, `require_json`; `get_json_response` wraps it |
| `src/fplopt/adapters/football_data.py` (new) | EPL season CSV bytes; CSV validator |
| `src/fplopt/adapters/vaastav.py` (new) | Pinned commit, git tree listing, file fetch with git-blob-sha check |
| `src/fplopt/adapters/fplcache.py` (new) | Resolve `main` HEAD sha; stream the tarball |
| `src/fplopt/adapters/fpl.py` | + `event_live(gw)` |
| `src/fplopt/ingest/history.py` (new) | One-off backfills: football-data seasons, vaastav, fplcache |
| `src/fplopt/ingest/jobs.py` | + current-season football-data step, post-lockdown step; element-summary manifest gains `season`, `through_event` |
| `src/fplopt/build/rules.py` (new) | Export `config/scoring/<season>.json` from an archived bootstrap |
| `src/fplopt/cli.py` | `Context`; new commands `backfill football-data|vaastav|fplcache`, `rules export SEASON` |
| `deploy/README.md`, `CLAUDE.md`, `README.md` | Runbook + commands |
| `tests/test_seasons.py`, `tests/test_football_data_adapter.py`, `tests/test_vaastav_adapter.py`, `tests/test_fplcache_adapter.py`, `tests/test_history.py`, `tests/test_rules.py` (new); existing test modules extended |

Raw layout added:
```
raw/football-data/E0/1617/2026-10-06T030000Z.csv.gz
raw/vaastav/data/2026-10-06T031500Z/_manifest.json.gz
raw/vaastav/data/2026-10-06T031500Z/2016-17/gws/merged_gw.csv.gz
raw/vaastav/data/2026-10-06T031500Z/2021-22/understat/Aaron_Cresswell_534.csv.gz
raw/vaastav/data/2026-10-06T031500Z/master_team_list.csv.gz
raw/fplcache/bootstrap-static/2021-04-18T164100Z.json.xz      # timestamp = fplcache snapshot time
raw/fplcache/runs/2026-10-06T040000Z.json.gz                   # run manifest (commit, counts)
raw/fpl/event-live/2026-27/5/2026-10-07T013000Z.json.gz
```

Conventions (match the existing code): `from __future__ import annotations`, module docstring, injected `now: Clock` and `sleep` for testability, `log = logging.getLogger(__name__)`, flat `tests/` with one module per source module, `httpx.MockTransport` for HTTP tests, ruff line length 100. Commit trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

---

### Task 0: Commit the PLAN.md update

- [ ] `git add docs/PLAN.md docs/superpowers/plans/2026-10-06-phase-1a-raw-backfill.md && git commit -m "docs: Understat via vaastav mirror only; Phase 1a plan"`

---

### Task 1: Season helpers

**Files:** Create `src/fplopt/seasons.py`, `tests/test_seasons.py`.

A season is identified by its **start year** (`2026` = 2026/27).

```python
"""Season labels and codes. A season is identified by its start year (2026 = 2026/27)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any

_LABEL = re.compile(r"^(\d{4})-(\d{2})$")


def season_label(start_year: int) -> str:
    """2026 -> '2026-27' (vaastav folders, config file names)."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def parse_season_label(label: str) -> int:
    """'2026-27' -> 2026. Rejects malformed or non-consecutive labels."""
    match = _LABEL.match(label)
    if not match or (int(match[1]) + 1) % 100 != int(match[2]):
        raise ValueError(f"not a season label like 2026-27: {label!r}")
    return int(match[1])


def football_data_code(start_year: int) -> str:
    """2026 -> '2627' (football-data.co.uk URL segment)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_start_year(when: datetime) -> int:
    """The season in progress or about to start at `when`: July onwards is the new season."""
    return when.year if when.month >= 7 else when.year - 1


def bootstrap_season(bootstrap: Mapping[str, Any]) -> int:
    """Start year of the season a bootstrap-static payload describes (year of GW1's deadline)."""
    first = min(bootstrap["events"], key=lambda event: event["id"])
    return int(first["deadline_time"][:4])
```

Tests (`tests/test_seasons.py`):
- `season_label(2026) == "2026-27"`, `season_label(2099) == "2099-00"`.
- `parse_season_label("2016-17") == 2016`; `"2016-18"`, `"16-17"`, `"2016/17"` raise `ValueError`.
- `football_data_code(2016) == "1617"`, `football_data_code(2026) == "2627"`.
- `season_start_year(datetime(2026, 6, 30, tzinfo=UTC)) == 2025`, `(2026, 7, 1) == 2026`, `(2027, 1, 15) == 2026`.
- `bootstrap_season({"events": [{"id": 2, "deadline_time": "2026-08-28T17:30:00Z"}, {"id": 1, "deadline_time": "2026-08-21T17:30:00Z"}]}) == 2026`.

Commit: `feat(seasons): season labels, codes and bootstrap season`.

---

### Task 2: RawStore — verbatim bytes, suffixes, nested names

**Files:** Modify `src/fplopt/ingest/raw_store.py`, extend `tests/test_raw_store.py`.

Changes (keep every existing behaviour and test passing):

```python
import lzma

COMPRESSED_SUFFIXES = (".gz", ".xz")
_DECOMPRESS = {".gz": gzip.decompress, ".xz": lzma.decompress}


def gzip_bytes(content: bytes) -> bytes:
    """Deterministic gzip (no mtime), as used for every `.gz` file in the store."""
    return gzip.compress(content, mtime=0)


def _check_relative(text: str, what: str) -> None:
    """Reject anything that could escape the store: absolute paths, '..', empty parts, '\\'."""
    parts = text.split("/")
    if not text or text.startswith("/") or "\\" in text or any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"unsafe {what}: {text!r}")
```

`RawStore` methods:
- `path_for(source, endpoint, fetched_at, *, suffix=SUFFIX, name=None) -> Path` — validates `source`, `endpoint` (may contain `/`, e.g. `E0/1617`, `event-live/2026-27/5`) and `name` (may contain `/`, e.g. `2016-17/gws/merged_gw`) with `_check_relative`; `suffix` must start with `.` and end with one of `COMPRESSED_SUFFIXES` (raw is always compressed) else `ValueError`. Returns `root/source/endpoint/<ts><suffix>` or `root/source/endpoint/<ts>/<name><suffix>`.
- `write_bytes(source, endpoint, data, fetched_at, *, suffix, name=None) -> Path` — stores `data` **verbatim** (caller compresses; suffix says how). Same tmp → fsync → `os.link` → `_fsync_dir` → unlink tmp sequence as today; `mkdir(parents=True)` on the final parent; raises `FileExistsError` instead of overwriting.
- `write(...)` (unchanged signature) becomes: validate JSON, then `self.write_bytes(source, endpoint, gzip_bytes(content), fetched_at, suffix=SUFFIX, name=name)`.
- `entries(source, endpoint, suffix=SUFFIX) -> list[tuple[datetime, Path]]` — the current `_entries`, made public and suffix-aware (glob `*{suffix}`, ignore non-timestamp names). Keep `_entries` callers working (rename them).
- `times(source, endpoint, suffix=SUFFIX)`, `latest(source, endpoint, suffix=SUFFIX)` — pass the suffix through.
- `read_bytes(path) -> bytes` (static) — decompress by the final suffix (`.gz`/`.xz`); `ValueError` for anything else.
- `read_json(path)` (static) — `json.loads(read_bytes(path))` (now reads `.json.xz` too).
- Update the class docstring: single-file snapshots vs grouped runs; `write` = JSON → gzip; `write_bytes` = verbatim.

Tests to add:
- `write_bytes` stores bytes verbatim (`path.read_bytes() == data`), path ends with the suffix, and refuses to overwrite (`FileExistsError`).
- `write_bytes` with `name="2016-17/gws/merged_gw", suffix=".csv.gz"` lands at `<endpoint>/<ts>/2016-17/gws/merged_gw.csv.gz`.
- unsafe inputs raise `ValueError`: endpoint `"../x"`, name `"a/../b"`, name `"/abs"`, name `"a\\b"`, source `""`.
- suffix `".csv"` (uncompressed) raises `ValueError`.
- `times("fplcache", "bootstrap-static", suffix=".json.xz")` lists `.json.xz` files only; `.json.gz` files in the same dir are not listed and vice versa.
- `read_json` round-trips a `.json.xz` written with `lzma.compress(b'{"a": 1}')` and a `.json.gz` written by `write`.
- `read_bytes` on a `.csv.gz` returns the original CSV bytes.

Commit: `feat(raw-store): verbatim byte writes, compression suffixes, nested names`.

---

### Task 3: Generic validated GET

**Files:** Modify `src/fplopt/adapters/http.py`, extend `tests/test_http.py`.

```python
Validator = Callable[[bytes], None]


class InvalidPayload(Exception):
    """A 2xx response whose body fails validation (e.g. FPL's HTML 'game is being updated' page)."""


def require_json(content: bytes) -> None:
    try:
        json.loads(content)
    except ValueError as exc:
        raise InvalidPayload("response is not JSON") from exc


def get_response(client, url, params=None, *, validate: Validator | None = None,
                 attempts: int = 5, wait=None) -> httpx.Response:
    """GET `url`, retrying transient failures (429, 5xx, transport errors, invalid payloads).
    If `validate` is given it must raise InvalidPayload for a bad body."""
    ...same Retrying loop; after raise_for_status(): if validate is not None: validate(response.content)


def get_json_response(client, url, params=None, *, attempts=5, wait=None) -> httpx.Response:
    """GET `url`; the returned body is guaranteed to be JSON."""
    return get_response(client, url, params, validate=require_json, attempts=attempts, wait=wait)
```

Never put `params` into exception messages (the odds key travels in `params`).

Tests to add: `get_response` without a validator returns non-JSON bodies (e.g. `text="a,b\n1,2"`) after one call; with a validator that raises `InvalidPayload`, it retries `attempts` times then raises `InvalidPayload`. All existing tests keep passing.

Commit: `refactor(http): generic validated GET; JSON GET wraps it`.

---

### Task 4: football-data adapter + jobs

**Files:** Create `src/fplopt/adapters/football_data.py`, `tests/test_football_data_adapter.py`, `src/fplopt/ingest/history.py`, `tests/test_history.py`; modify `src/fplopt/ingest/jobs.py`, `tests/test_jobs.py`.

Adapter:

```python
"""football-data.co.uk: EPL results + pre-match odds, one CSV per season. Returns raw bytes."""

BASE_URL = "https://www.football-data.co.uk/mmz4281"


def require_csv(content: bytes) -> None:
    """football-data CSVs start with the `Div` column (after an optional UTF-8 BOM)."""
    head = content[:64].decode("utf-8-sig", errors="replace").lstrip()
    if not head.startswith("Div,"):
        raise InvalidPayload("response is not a football-data CSV")


class FootballDataClient:
    def __init__(self, client: httpx.Client, base_url: str = BASE_URL) -> None: ...

    def epl_season(self, start_year: int) -> bytes:
        url = f"{self._base}/{football_data_code(start_year)}/E0.csv"
        return get_response(self._client, url, validate=require_csv, attempts=3).content
```

Before finalising `require_csv`, check it against the real headers saved by the research agent in the session scratchpad (`E0_*.csv` under `C:\Users\samin\AppData\Local\Temp\claude\C--Users-samin-Documents-GitHub-fpl-optimizer\5303fed5-4bcc-43af-8baa-bdb5fd0a6625\scratchpad\`) if present; otherwise fetch `1617/E0.csv` and `2526/E0.csv` once with curl and look at the first bytes (1617 may be quoted or have a different first column — adapt the validator so every season 1617…2627 passes, and add a test for each variant seen).

`jobs.py` additions:

```python
FOOTBALL_DATA_GRACE_MONTHS = (7, 8)  # new-season CSV may not exist yet in July/August


class FootballDataSource(Protocol):
    def epl_season(self, start_year: int) -> bytes: ...


def snapshot_football_data(store, fd, start_year, now=utc_now) -> Path:
    content = fd.epl_season(start_year)
    return store.write_bytes(
        "football-data",
        f"E0/{football_data_code(start_year)}",
        gzip_bytes(content),
        now(),
        suffix=".csv.gz",
    )


def snapshot_football_data_current(store, fd, now=utc_now) -> Path | None:
    """Daily: the in-progress season's CSV (results + odds are appended twice a week).
    A 404 is tolerated in July/August (file not published yet); otherwise it is an error."""
    current = now()
    try:
        return snapshot_football_data(store, fd, season_start_year(current), now)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404 and current.month in FOOTBALL_DATA_GRACE_MONTHS:
            log.warning("football-data CSV for the new season not published yet")
            return None
        raise
```

`run_daily(store, fpl, odds, now=utc_now, *, football_data=None, sleep=time.sleep)` adds a step `("football-data", lambda: snapshot_football_data_current(store, football_data, now))` only when `football_data is not None` (Task 7 adds the post-lockdown step; `sleep` is for it).

`history.py`:

```python
"""One-off historical backfills into raw/: football-data seasons, vaastav, fplcache."""

FIRST_SEASON = 2016


def backfill_football_data(store, fd, now=utc_now, sleep=time.sleep, pause_s=1.0) -> int:
    """Every EPL season CSV from 2016/17 to the current season. All seasons are attempted;
    failures are collected and raised together (SnapshotError, one line)."""
```
Implement with `jobs._run_independently` (rename it to public `run_independently` and update callers) over one step per season, sleeping `pause_s` between requests. Return the number of seasons written.

Tests:
- adapter: hits `https://www.football-data.co.uk/mmz4281/1617/E0.csv` for 2016; returns bytes unchanged; accepts a BOM-prefixed body; rejects HTML (`<html>…`) with `InvalidPayload` after retries (use `wait=`? — the adapter does not expose `wait`; instead test `require_csv` directly for the rejection, and the URL/bytes via MockTransport).
- `snapshot_football_data` writes `football-data/E0/1617/<ts>.csv.gz` whose `RawStore.read_bytes` equals the CSV.
- `snapshot_football_data_current`: 404 in August → returns `None`, nothing written; 404 in October → raises `httpx.HTTPStatusError`. (Fake source raising `httpx.HTTPStatusError` built with `httpx.Response(404, request=httpx.Request("GET", "https://x"))`.)
- `run_daily(..., football_data=Fake)` writes FPL, odds and the current-season CSV; a failing football-data source does not block FPL/odds and the combined error names `football-data`.
- `backfill_football_data` with `now` in 2026-10 fetches 2016…2026 (11 seasons) in order; one failing season → others still written, raises with that season's code in the message.

Commit: `feat(ingest): football-data adapter, season backfill, daily current-season snapshot`.

---

### Task 5: vaastav mirror

**Files:** Create `src/fplopt/adapters/vaastav.py`, `tests/test_vaastav_adapter.py`; extend `src/fplopt/ingest/history.py`, `tests/test_history.py`.

Adapter:

```python
"""vaastav/Fantasy-Premier-League at a pinned commit: git tree listing + raw file bytes."""

REPO = "vaastav/Fantasy-Premier-League"
PINNED_COMMIT = "9779cdbc0c07f6c900c2d0c181ddf6bb9c800f88"  # 2026-08-28, "Add 26/27 gw1 data"
API_URL = "https://api.github.com"
RAW_URL = "https://raw.githubusercontent.com"


def git_blob_sha(data: bytes) -> str:
    """The sha git uses for a file's content, to verify downloads against the tree listing."""
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


class BlobMismatch(Exception):
    """Downloaded bytes do not match the blob sha in the pinned tree."""


class VaastavClient:
    def __init__(self, client, api_url=API_URL, raw_url=RAW_URL): ...

    def tree(self, commit: str) -> dict[str, str]:
        """path -> blob sha for every file at `commit`. Refuses a truncated listing."""
        # GET {api}/repos/{REPO}/git/trees/{commit}?recursive=1 via get_json_response
        # raise RuntimeError if data["truncated"]; keep entries with type == "blob"

    def file(self, commit: str, path: str, blob_sha: str) -> bytes:
        # GET {raw}/{REPO}/{commit}/{quote(path, safe='/')} via get_response (no validator)
        # raise BlobMismatch if git_blob_sha(content) != blob_sha
```

Selection (pure function in `history.py`):

```python
VAASTAV_SEASONS = range(2016, 2026)  # 2016-17 … 2025-26; 2026/27 comes from our own archive
_VAASTAV_FILE = re.compile(
    r"^data/(?P<season>\d{4}-\d{2})/"
    r"(?:players_raw|player_idlist|cleaned_players|fixtures|teams|id_dict|gws/merged_gw"
    r"|understat/[^/]+)\.csv$"
)
VAASTAV_ROOT_FILES = ("data/master_team_list.csv",)


def select_vaastav_paths(paths: Iterable[str]) -> list[str]:
    seasons = {season_label(year) for year in VAASTAV_SEASONS}
    return sorted(
        p
        for p in paths
        if p in VAASTAV_ROOT_FILES
        or ((m := _VAASTAV_FILE.match(p)) is not None and m["season"] in seasons)
    )
```
Excluded on purpose: `gws/xP*.csv` (lookahead leak, PLAN §4), `gws/gwN.csv` (duplicated by `merged_gw`), `players/` (duplicated), `fbref/`, `managers/`, `cleaned_merged_seasons*.csv` (incomplete), the stale `2026-27/`.

Job:

```python
def backfill_vaastav(
    store,
    vaastav,
    commit=PINNED_COMMIT,
    now=utc_now,
    sleep=time.sleep,
    pause_s=0.05,
    max_consecutive_failures=MAX_CONSECUTIVE_FAILURES,
) -> int:
    """Mirror the selected vaastav files at `commit` under one run directory:
    raw/vaastav/data/<run_at>/<path without 'data/' and '.csv'>.csv.gz, plus _manifest.json.gz
    (commit, run_at, finished_at, expected, written, failed) written even when aborted.
    Continues past single failures; aborts after `max_consecutive_failures` in a row; raises if
    anything failed."""
```
Same structure as `backfill_element_summaries` (look at it and mirror it; factor out a shared helper only if it stays simpler). File bytes are gzipped with `gzip_bytes` and stored with `write_bytes(..., suffix=".csv.gz", name=...)`; the manifest with `store.write("vaastav", "data", ..., run_at, name="_manifest")`.

Tests:
- `git_blob_sha(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"` (known git value).
- `tree` builds the right URL (`…/git/trees/<sha>?recursive=1`), returns only blobs, raises on `"truncated": true`.
- `file` URL-quotes paths (e.g. `data/2021-22/understat/Martin_Ødegaard_1.csv` → `%C3%98`), returns bytes, raises `BlobMismatch` on wrong sha.
- `select_vaastav_paths` keeps merged_gw/players_raw/understat/master_team_list for 2016-17…2025-26; drops `gws/xP3.csv`, `gws/gw1.csv`, `players/x/gw.csv`, `2026-27/players_raw.csv`, `2015-16/players_raw.csv`, `cleaned_merged_seasons.csv`.
- `backfill_vaastav` with a fake client (tree of 3 selected + 1 unselected file): writes 3 gzipped CSVs at the expected nested paths (bytes round-trip via `read_bytes`) + manifest with `commit`; one failing file → others written, manifest lists it under `failed`, raises naming the count.

Commit: `feat(ingest): vaastav mirror at a pinned commit with blob-sha verification`.

---

### Task 6: fplcache mirror

**Files:** Create `src/fplopt/adapters/fplcache.py`, `tests/test_fplcache_adapter.py`; extend `src/fplopt/ingest/history.py`, `tests/test_history.py`.

Adapter:

```python
"""Randdalf/fplcache: bootstrap-static snapshots ~4x/day since April 2021 (public domain)."""

REPO = "Randdalf/fplcache"
API_URL = "https://api.github.com"
CODELOAD_URL = "https://codeload.github.com"


class _ChunkReader(io.RawIOBase):
    """File-like view over an iterator of byte chunks, so tarfile can stream an HTTP body."""
    def __init__(self, chunks: Iterator[bytes]) -> None: ...
    def readable(self) -> bool: return True
    def readinto(self, buffer) -> int: ...  # fill from pending chunk; 0 at end of stream


class FplcacheClient:
    def __init__(self, client, api_url=API_URL, codeload_url=CODELOAD_URL): ...

    def head_commit(self, branch: str = "main") -> str:
        # GET {api}/repos/{REPO}/commits/{branch} (get_json_response) -> ["sha"]

    @contextmanager
    def tarball(self, commit: str) -> Iterator[tarfile.TarFile]:
        """Stream the repo tarball at `commit` (~0.9 GB) without holding it in memory."""
        # with self._client.stream("GET", f"{codeload}/{REPO}/tar.gz/{commit}") as response:
        #     response.raise_for_status()
        #     reader = io.BufferedReader(_ChunkReader(response.iter_bytes()), 1 << 20)
        #     with tarfile.open(fileobj=reader, mode="r|gz") as tar:
        #         yield tar
```
No retries around the stream: a failed download fails the job (alert); re-running skips files already archived.

`history.py`:

```python
_FPLCACHE_MEMBER = re.compile(
    r"^[^/]+/cache/(?P<y>\d{4})/(?P<m>\d{1,2})/(?P<d>\d{1,2})/(?P<H>\d{2})(?P<M>\d{2})\.json\.xz$"
)


def fplcache_snapshot_time(member_name: str) -> datetime | None:
    """'fplcache-<sha>/cache/2021/4/18/1641.json.xz' -> 2021-04-18 16:41 UTC; None otherwise."""


def backfill_fplcache(store, fplcache, now=utc_now) -> int:
    """Mirror every fplcache snapshot into raw/fplcache/bootstrap-static/<snapshot time>.json.xz,
    byte-for-byte. Idempotent: identical files already present are skipped; a present file with
    different bytes is a failure (upstream rewrote history). Each new file must decompress to
    JSON. Writes a run manifest raw/fplcache/runs/<run_at>.json.gz with commit, written, skipped,
    failed (member names + reason) — also when failures occurred — then raises if any failed.
    Returns the number of files written."""
```
Use `store.path_for("fplcache", "bootstrap-static", ts, suffix=".json.xz")` + `.exists()` for the skip check, `store.write_bytes(...)` for new files. Iterate `for member in tar:`; skip non-files and non-matching names; read with `tar.extractfile(member).read()`. Log progress every 500 files.

Tests (build a small `.tar.gz` in memory with `tarfile` + `lzma.compress`; serve it and the commits JSON through `httpx.MockTransport` routing on URL):
- `fplcache_snapshot_time` parses unpadded month/day, returns `None` for `README.md` and `cache/2021/4/18/notes.txt`.
- `_ChunkReader` reassembles chunks of uneven sizes (`[b"ab", b"", b"cde"]` → `b"abcde"`).
- `head_commit` returns the sha from the API JSON.
- `backfill_fplcache` writes two snapshots at the right paths, bytes identical to the tar members; second run writes 0 and skips 2; a member with corrupt xz → recorded as failed, others written, raises; a pre-existing file with different bytes → failed.
- Run manifest exists after both a clean and a failing run and contains the commit.

Commit: `feat(ingest): fplcache mirror streamed from the pinned tarball`.

---

### Task 7: Post-lockdown ingest (issue #17)

**Files:** Modify `src/fplopt/adapters/fpl.py`, `tests/test_fpl_adapter.py`, `src/fplopt/ingest/jobs.py`, `tests/test_jobs.py`.

- `FplClient.event_live(gw)` → `event/{gw}/live/`. Add to `FplSource` protocol.
- `backfill_element_summaries` manifest gains `"season": bootstrap_season(bootstrap)` and `"through_event": max(ids of events with data_checked, default 0)` (computed from the bootstrap it fetched). Use `event.get("data_checked")` so fakes without the key work.

```python
def checked_events(bootstrap) -> list[int]:
    """GWs whose data FPL has finalised (bonus confirmed): `data_checked` is true."""


def snapshot_event_live(store, fpl, bootstrap, now=utc_now) -> list[int]:
    """Archive event/{gw}/live/ once per finalised GW, under fpl/event-live/<season>/<gw>
    (season in the path because GW numbers repeat every season). Returns GWs written."""


def latest_complete_element_summary(store) -> tuple[int, int]:
    """(season, through_event) of the newest element-summary run whose manifest shows every
    expected element written; (0, 0) if none (older manifests without these keys count as none)."""


def element_summaries_due(store, bootstrap) -> bool:
    """True if a GW has been finalised since the newest complete element-summary run."""


def snapshot_post_lockdown(store, fpl, now=utc_now, *, sleep=time.sleep) -> None:
    """After GW lockdown: event-live for new finalised GWs, and one fresh element-summary run
    (all players' per-GW history) if a GW was finalised since the last complete run. Reads the
    latest archived bootstrap; the two steps fail independently."""
```
`run_daily` gets a final step `("post-lockdown", lambda: snapshot_post_lockdown(store, fpl, now, sleep=sleep))`, after the FPL step so it sees the fresh bootstrap. It is a no-op when nothing is finalised. Nested `SnapshotError` messages stay single-line.

Update the `FakeFpl` test helper: events get `"id": 1, "data_checked": False, "finished": False`; add `checked=()` parameter marking those GW ids `data_checked: True`; add `event_live(gw)` recording calls and returning `json.dumps({"elements": []}).encode()`. Existing tests must still pass unchanged in meaning.

Tests:
- `event_live(5)` hits `…/api/event/5/live/`.
- no finalised GWs → `run_daily` makes no `event_live` or `element_summary` calls.
- finalised GWs 1–2, empty store → event-live written for both under `fpl/event-live/2026-27/1` and `/2`, and one element-summary run; the manifest has `season: 2026`, `through_event: 2`.
- second `run_daily` with the same bootstrap → no new event-live, no new element-summary run.
- a later bootstrap with GW3 finalised → event-live for 3 only, and a new element-summary run.
- a failed element-summary run (one failing element) → next day it runs again (the failed manifest doesn't count).
- an old-style manifest (no `season`/`through_event`) → counts as none → run is due.

Commit: `feat(ingest): post-lockdown event-live + element-summary refresh (closes #17 core)`.

---

### Task 8: Rules config export

**Files:** Create `src/fplopt/build/rules.py`, `tests/test_rules.py`.

```python
"""Export config/scoring/<season>.json from an archived bootstrap-static (PLAN §3 Rules config)."""


def find_bootstrap(store, start_year) -> tuple[datetime, Path]:
    """Newest archived bootstrap describing season `start_year`: our own archive
    (fpl/bootstrap-static, .json.gz) first, then fplcache (.json.xz). Walks each archive
    newest-first, skipping snapshots taken on/after 1 Aug of start_year + 1, and stops once
    snapshots are older than 1 Jun of start_year. Raises LookupError if none matches."""


def export_rules(store, start_year, out_dir: Path) -> Path:
    """Write out_dir/<season>.json (sorted keys, indent 2, LF, trailing newline):
    {season, source: {path (relative to raw root, POSIX), snapshot_at},
     scoring, rules, settings (from game_config), chips,
     element_types (minus volatile 'element_count'),
     event_overrides: {gw: overrides} for events whose overrides have any truthy value}.
    Raises ValueError if the bootstrap has no game_config or chips."""
```
Use `store.entries(...)` and `bootstrap_season`. Exclude `game_config["status"]` (runtime state, not rules).

Tests (write small fake bootstraps into a temp `RawStore` via `write` and `write_bytes` + `lzma.compress`):
- picks our archive's newest snapshot for the current season;
- for the previous season, skips our archive (all newer) and fplcache snapshots from the new season after the July reset, returning the newest old-season fplcache snapshot;
- `LookupError` when no snapshot matches;
- exported JSON content: keys as above, `element_count` stripped, only non-empty event overrides kept, file ends with `\n`, re-export is byte-identical (deterministic);
- missing `game_config` → `ValueError`.

Commit: `feat(build): export per-season rules config from archived bootstraps`.

---

### Task 9: CLI wiring

**Files:** Modify `src/fplopt/cli.py`, `tests/test_cli.py`.

```python
@dataclass
class Context:
    store: RawStore
    http: httpx.Client
    settings: Settings
    args: argparse.Namespace

    @property
    def fpl(self) -> FplClient:
        return FplClient(self.http)

    @property
    def odds(self) -> OddsClient | None:
        key = self.settings.odds_api_key
        return OddsClient(self.http, key) if key else None


Job = Callable[[Context], object]

JOBS: dict[str, Job] = {
    "snapshot daily": lambda c: jobs.run_daily(
        c.store, c.fpl, c.odds, football_data=FootballDataClient(c.http)
    ),
    "snapshot tick": lambda c: jobs.run_tick(c.store, c.fpl, c.odds),
    "backfill element-summary": lambda c: jobs.backfill_element_summaries(c.store, c.fpl),
    "backfill football-data": lambda c: history.backfill_football_data(
        c.store, FootballDataClient(c.http)
    ),
    "backfill vaastav": lambda c: history.backfill_vaastav(c.store, VaastavClient(c.http)),
    "backfill fplcache": lambda c: history.backfill_fplcache(c.store, FplcacheClient(c.http)),
    "rules export": lambda c: rules.export_rules(
        c.store, parse_season_label(c.args.season), Path(c.args.out)
    ),
}
```
Parser: `backfill` choices `element-summary, football-data, vaastav, fplcache`; new group `rules` with `command` choice `export`, positional `season` (e.g. `2026-27`) and `--out` (default `config/scoring`). `main` builds the `Context`; everything else (logging, alerts, return codes, try block) unchanged. Update the module docstring.

Tests: update existing ones to one-argument jobs (`lambda c: ran.append(c.odds)`); add: `rules export 2026-27` passes `season`/`out` through (monkeypatched job reads `c.args`); `backfill vaastav` is a valid command; a bad season label fails with exit code 1 and an alert (the `ValueError` is raised inside the try).

Commit: `feat(cli): context-based jobs; backfill and rules export commands`.

---

### Task 10: Docs

**Files:** `deploy/README.md`, `CLAUDE.md`, `README.md`.

- `deploy/README.md` new section **Historical backfills (one-off, Phase 1a)** — run on the server as long-lived transient units so they survive SSH disconnects:
  ```bash
  cd ~/fpl-optimizer && git pull && ~/.local/bin/uv sync --locked
  systemd-run --user --unit=fplopt-bf-football-data --working-directory=%h/fpl-optimizer \
    -p OnFailure=fplopt-failure@%n.service .venv/bin/fplopt backfill football-data
  # same for: vaastav (~3.4k files, ~10 min), fplcache (~0.9 GB, ~10–20 min)
  journalctl --user -u fplopt-bf-vaastav -f
  ```
  (Check that `%h`/`%n` expand in `systemd-run`; if not, use `$HOME` and the literal unit name. Verify on the server in Task 12 and fix the doc.)
  Expected results and how to check manifests (`raw/vaastav/data/<run>/_manifest.json.gz`, `raw/fplcache/runs/`). Re-running fplcache later tops up new snapshots.
- Section **Rules config**: `fplopt rules export 2026-27` / `2025-26` → copy `config/scoring/*.json` back and commit.
- Section **Copy raw/ to the dev machine** (for notebooks/Phase 1b): `ssh fplopt-server 'tar -C fpl-optimizer -cf - raw' | tar -C . -xf -` from the repo root (Git Bash).
- Note that the daily job now also archives the current football-data CSV and, after each GW lockdown, `event/{gw}/live/` plus a fresh element-summary run (a few minutes, ~700 requests).
- `CLAUDE.md` Commands: add `uv run fplopt backfill football-data|vaastav|fplcache` and `uv run fplopt rules export <season>`.
- `README.md`: phase table/status line if it lists phases.

Commit: `docs: Phase 1a backfill runbook`.

---

### Task 11: Local read-only smoke test

No writes to the repo; use a scratch raw dir (`FPLOPT_RAW_DIR=<scratchpad>/raw-smoke`).
- `uv run pytest` all green; `uv run ruff check . && uv run ruff format --check .` clean.
- `uv run fplopt backfill football-data` against the scratch dir → 11 CSVs (1617…2627), each `read_bytes` decodes and starts with `Div`.
- Python one-liner: `VaastavClient(make_client()).tree(PINNED_COMMIT)` → `len(select_vaastav_paths(tree))` in the low thousands; print counts per season and per kind (merged_gw / players_raw / understat). Do **not** run the full vaastav or fplcache backfills locally.
- `FplcacheClient(make_client()).head_commit()` returns a 40-hex sha.

---

### Task 12: PR, review, merge, run on the server

- [ ] Push, open PR "Phase 1a: raw backfills" (body: summary, test plan, closes #15 and #17 — move #17's leftover heartbeat/network-wait items into a new issue first). Wait for CI green; independent code review subagent; fix findings; merge.
- [ ] Server (`ssh fplopt-server`): `git pull && ~/.local/bin/uv sync --locked`; re-copy `deploy/systemd/*` only if changed.
- [ ] Run the three backfills per the runbook; verify: football-data 11 seasons; vaastav manifest `failed == []`; fplcache run manifest `failed == []`, file count ≈ upstream count; `du -sh raw/*`.
- [ ] Trigger `systemctl --user start fplopt-daily.service` once; verify event-live GW1–5 under `raw/fpl/event-live/2026-27/` and a new element-summary run with `through_event: 5`.
- [ ] `fplopt rules export 2026-27` and `2025-26` on the server; copy the two JSON files back; check 2026-27 against PLAN §3 (defcon DEF/MID/FWD = 2, chip windows 2–19 / 20–38 for WC/FH, `max_extra_free_transfers` = 4); check 2025-26 includes the AFCON free-transfer override. If 2025-26's bootstrap lacks `game_config`, record that in PLAN §11 instead of guessing.
- [ ] Small follow-up PR adding `config/scoring/2025-26.json` and `2026-27.json`.
