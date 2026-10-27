# ESPN cup data, PR 1 (data) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Archive ESPN's PL, domestic cup and European match summaries (lineups, substitutions, cards) into `raw/`, build `espn_match`, `espn_player_match`, `espn_schedule_snapshot` and `espn_map` from them, and keep them current with the daily job.

**Architecture:** Spec: `docs/superpowers/specs/2026-10-27-espn-cup-data-design.md` §1. A new adapter (`fplopt.adapters.espn`) returns raw bytes; an ingest module (`fplopt.ingest.espn`) writes scoreboards and summaries into the append-only `RawStore`; a build module (`fplopt.build.espn`) parses them into four tables and maps ESPN athletes to `player_key` with the Understat matcher, generalised by id column. Clubs resolve through a new `espn` column (ESPN team id) in `config/teams.csv`.

**Tech Stack:** Python 3.12, httpx + tenacity (`fplopt.adapters.http`), pandas, pandera, rapidfuzz, pytest. Run everything with `uv run`.

**Facts checked on the live API (2026-10-27):**
- `GET {BASE}/{league}/scoreboard?dates=YYYY&limit=1000` returns the calendar year's events (e.g. `eng.1` 2016: 378); `dates=YYYYMM` the month. Date ranges (`YYYYMMDD-YYYYMMDD`) return HTTP 400.
- Event: `id`, `date` (`2026-10-24T16:30Z`), `status.type.{name, completed, state}` (`STATUS_FULL_TIME`, `STATUS_FINAL_AET`, `STATUS_FINAL_PEN`, `STATUS_SCHEDULED`; `state` `pre`/`in`/`post`), `competitions[0].competitors[].{homeAway, team.id}`.
- `GET {BASE}/{league}/summary?event=ID`: `header.id`, `header.competitions[0].{date, status, competitors}`, `rosters[].{homeAway, team.id, roster[]}` with `roster[].{starter, subbedIn, subbedOut, athlete.{id, displayName}}`, and `keyEvents[]` with `type.text` (`Substitution`, `Yellow Card`, `Red Card`), `clock.value` (seconds), `team.id`, `participants[].athlete.id`. A substitution's participants are `[in, out]`. A second yellow is a `Red Card` event after a `Yellow Card` event for the same athlete (text "Second yellow card to …"). Present in 2016/17 EFL Cup, PL and UEL summaries.
- Leagues: `eng.1`, `eng.fa`, `eng.league_cup`, `eng.charity`, `uefa.champions`, `uefa.europa`, `uefa.europa.conf`, `uefa.super_cup` all answer.

---

## File map

- Modify `config/teams.csv`: add `espn` column (ESPN team id).
- Modify `src/fplopt/build/teams.py`: config column, `team_dim.espn_id`, resolver source `espn`.
- Modify `tests/synthetic_season.py:1010` area: synthetic `team_dim` gains `espn_id`.
- Create `src/fplopt/adapters/espn.py`: `EspnClient`.
- Create `src/fplopt/ingest/espn.py`: archive scoreboards/summaries, backfill, daily step, `espn_team_ids`.
- Modify `src/fplopt/ingest/jobs.py`: `run_daily(..., espn=...)`.
- Modify `src/fplopt/cli.py`: `backfill espn`, daily wiring.
- Modify `src/fplopt/build/understat.py`: matcher helpers take an id column.
- Create `src/fplopt/build/espn.py`: parse + four tables + mapping + checks.
- Modify `src/fplopt/build/__init__.py`, `src/fplopt/build/tables.py`: register.
- Modify `tests/synthetic_raw.py`: `World.add_espn_summary`, `World.add_espn_scoreboard`.
- Tests: `tests/test_espn_adapter.py`, `tests/test_ingest_espn.py`, `tests/test_build_espn.py`, edits to `tests/test_build_teams.py`, `tests/test_build_tables.py`, `tests/test_jobs.py`, `tests/test_cli.py`.
- Docs: `docs/PLAN.md` (§3 sources, §6.3 note), `CLAUDE.md` (commands), `deploy/README.md` (backfill).

---

### Task 1: ESPN team ids in `config/teams.csv` and `team_dim`

**Files:**
- Modify: `config/teams.csv`
- Modify: `src/fplopt/build/teams.py:51-60,72-76,133,150-165`
- Modify: `tests/synthetic_season.py` (the `team_dim` frame near line 1010)
- Test: `tests/test_build_teams.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_build_teams.py`)

```python
def test_team_dim_carries_espn_ids_and_the_resolver_finds_them():
    dim = team_dim_from_config(REPO_CONFIG / "teams.csv")
    by_short = dim.set_index("short_name")["espn_id"]
    assert by_short["ARS"] == "359" and by_short["CHE"] == "363" and by_short["MUN"] == "360"
    assert dim.loc[~dim["in_fpl"], "espn_id"].isna().all()  # synthetic pre-2016 clubs
    assert dim.loc[dim["in_fpl"], "espn_id"].notna().all()
    resolver = TeamResolver(dim)
    assert resolver.espn("363") == 8
    assert resolver.find("espn", "999999") is None


def test_teams_config_rejects_a_duplicate_espn_id(tmp_path):
    text = (REPO_CONFIG / "teams.csv").read_text(encoding="utf-8")
    path = tmp_path / "teams.csv"
    path.write_text(text.replace(",363\n", ",359\n"), encoding="utf-8")  # Chelsea -> Arsenal's id
    with pytest.raises(ValueError, match="espn '359' used by team_key"):
        read_teams_config(path)
```

(Use the module's existing imports; add `REPO_CONFIG` from `synthetic_raw` and `read_teams_config`, `TeamResolver` if not already imported.)

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_build_teams.py -k espn -v`
Expected: FAIL (`KeyError: 'espn_id'`).

- [ ] **Step 3: Add the column to `config/teams.csv`**

Header becomes `team_key,short_name,fpl_names,football_data,understat,odds_api,espn`. Append the ESPN id to each FPL club's row (blank for 1001–1010):

| key | short | espn | key | short | espn | key | short | espn |
|---|---|---|---|---|---|---|---|---|
| 3 | ARS | 359 | 13 | LEI | 375 | 20 | SOU | 376 |
| 7 | AVL | 362 | 14 | LIV | 364 | 110 | STK | 336 |
| 91 | BOU | 349 | 102 | LUT | 301 | 56 | SUN | 366 |
| 94 | BRE | 337 | 43 | MCI | 382 | 80 | SWA | 318 |
| 36 | BHA | 331 | 1 | MUN | 360 | 6 | TOT | 367 |
| 90 | BUR | 379 | 25 | MID | 369 | 57 | WAT | 395 |
| 97 | CAR | 347 | 4 | NEW | 361 | 35 | WBA | 383 |
| 8 | CHE | 363 | 45 | NOR | 381 | 21 | WHU | 371 |
| 9 | COV | 388 | 17 | NFO | 393 | 39 | WOL | 380 |
| 31 | CRY | 384 | 49 | SHU | 398 | | | |
| 11 | EVE | 368 | 38 | HUD | 335 | | | |
| 54 | FUL | 370 | 88 | HUL | 306 | | | |
| 40 | IPS | 373 | 2 | LEE | 357 | | | |

Example rows: `3,ARS,Arsenal,Arsenal,Arsenal,Arsenal,359` and `1001,BIR,,Birmingham,,,`.

- [ ] **Step 4: Teach `teams.py` the column**

```python
CONFIG_COLUMNS = (
    "team_key",
    "short_name",
    "fpl_names",
    "football_data",
    "understat",
    "odds_api",
    "espn",
)
...
_SOURCES = {
    "football_data": ("football_data_name", "football-data"),
    "understat": ("understat_name", "Understat"),
    "odds_api": ("odds_api_name", "Odds API"),
    "espn": ("espn_id", "ESPN"),
}
```

In `SCHEMA`, after `odds_api_name`:

```python
        "espn_id": pa.Column(str, pa.Check(_unique_when_present), nullable=True),
```

In `read_teams_config`, the duplicate check's column list:

```python
values = [(c, row[c]) for c in ("short_name", "football_data", "understat", "odds_api", "espn")]
```

In `team_dim_from_config`, after `"odds_api_name": config["odds_api"],`:

```python
            "espn_id": config["espn"],
```

Add to `TeamResolver`:

```python
    def espn(self, team_id: str) -> int:
        return self._get("espn", "ESPN", team_id)
```

Update the module docstring's config-column sentence: "`… odds_api, espn` — the last three only where the source has the club (blank otherwise); `espn` is ESPN's team id."

- [ ] **Step 5: Synthetic team_dim** — in `tests/synthetic_season.py`, next to `"odds_api_name": …` add:

```python
            "espn_id": pd.array([str(9000 + t) for t in teams], dtype="str"),
```

Also update the column list asserted in `tests/test_build_teams.py:123` (add `"espn_id"` after `"odds_api_name"`).

- [ ] **Step 6: Run the team tests**

Run: `uv run pytest tests/test_build_teams.py tests/test_build_tables.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add config/teams.csv src/fplopt/build/teams.py tests/synthetic_season.py tests/test_build_teams.py
git commit -m "feat(teams): ESPN team ids in teams.csv and team_dim"
```

---

### Task 2: `EspnClient` adapter

**Files:**
- Create: `src/fplopt/adapters/espn.py`
- Test: `tests/test_espn_adapter.py`

- [ ] **Step 1: Write the failing tests**

```python
import httpx
import pytest

from fplopt.adapters.espn import BASE_URL, LEAGUES, EspnClient, require_key
from fplopt.adapters.http import InvalidPayload, make_client


def recording_client(body: bytes):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, content=body)

    return make_client(httpx.MockTransport(handler)), seen


def test_scoreboard_asks_for_the_period_and_returns_bytes_unchanged():
    body = b'{"events": []}'
    client, seen = recording_client(body)
    assert EspnClient(client).scoreboard("eng.league_cup", "2016") == body
    assert seen == [f"{BASE_URL}/eng.league_cup/scoreboard?dates=2016&limit=1000"]


def test_summary_asks_for_the_event():
    body = b'{"header": {}, "rosters": []}'
    client, seen = recording_client(body)
    assert EspnClient(client).summary("eng.1", "401878767") == body
    assert seen == [f"{BASE_URL}/eng.1/summary?event=401878767"]


def test_require_key_rejects_json_without_the_key():
    require_key("events")(b'{"events": []}')
    with pytest.raises(InvalidPayload):
        require_key("events")(b'{"code": 404}')
    with pytest.raises(InvalidPayload):
        require_key("events")(b"<html>")


def test_leagues_are_the_eight_competitions():
    assert LEAGUES == (
        "eng.1",
        "eng.fa",
        "eng.league_cup",
        "eng.charity",
        "uefa.champions",
        "uefa.europa",
        "uefa.europa.conf",
        "uefa.super_cup",
    )


def test_unknown_league_is_refused():
    client, _ = recording_client(b"{}")
    with pytest.raises(ValueError, match="unknown ESPN league"):
        EspnClient(client).scoreboard("esp.1", "2016")
```

- [ ] **Step 2: Run to see them fail**

Run: `uv run pytest tests/test_espn_adapter.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
"""ESPN's public site API (unofficial, no key): soccer scoreboards and match summaries.
Returns raw bytes. Formats checked 2026-10-27 (docs/superpowers/plans/2026-10-27-espn-cups-1-data.md)."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx

from fplopt.adapters.http import InvalidPayload, get_response

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer"
# The PL, domestic cups, the Community Shield and UEFA club competitions.
LEAGUES = (
    "eng.1",
    "eng.fa",
    "eng.league_cup",
    "eng.charity",
    "uefa.champions",
    "uefa.europa",
    "uefa.europa.conf",
    "uefa.super_cup",
)
# Red-card bans carry across these (PLAN §6.3).
DOMESTIC = ("eng.1", "eng.fa", "eng.league_cup")
SCOREBOARD_LIMIT = "1000"


def require_key(key: str) -> Callable[[bytes], None]:
    """A validator: the body is a JSON object with `key` (ESPN answers some errors with 200)."""

    def validate(content: bytes) -> None:
        try:
            payload = json.loads(content)
        except ValueError as exc:
            raise InvalidPayload("response is not JSON") from exc
        if not isinstance(payload, dict) or key not in payload:
            raise InvalidPayload(f"response has no {key!r}")

    return validate


class EspnClient:
    def __init__(self, client: httpx.Client, base_url: str = BASE_URL) -> None:
        self._client = client
        self._base = base_url.rstrip("/")

    def _url(self, league: str, endpoint: str) -> str:
        if league not in LEAGUES:
            raise ValueError(f"unknown ESPN league {league!r}")
        return f"{self._base}/{league}/{endpoint}"

    def scoreboard(self, league: str, period: str) -> bytes:
        """Every event of a calendar year (`period` 'YYYY') or month ('YYYYMM')."""
        url = self._url(league, "scoreboard")
        params = {"dates": period, "limit": SCOREBOARD_LIMIT}
        return get_response(self._client, url, params, validate=require_key("events")).content

    def summary(self, league: str, event_id: str) -> bytes:
        """One match: header, rosters (lineups, substitutes) and key events (subs, cards)."""
        url = self._url(league, "summary")
        params = {"event": event_id}
        return get_response(self._client, url, params, validate=require_key("header")).content
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_espn_adapter.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fplopt/adapters/espn.py tests/test_espn_adapter.py
git commit -m "feat(adapters): ESPN scoreboard and match summary client"
```

---

### Task 3: Archive scoreboards and summaries (`fplopt.ingest.espn`)

Raw layout (`RawStore` single-file snapshots; endpoints may contain `/`):
- `espn/<league>/scoreboard/<period>/<ts>.json.gz` (`period` = `YYYY` or `YYYYMM`);
- `espn/<league>/summary/<event_id>/<ts>.json.gz`.

All scoreboards of one run share one timestamp (the run's start), so a run is one schedule snapshot.

**Files:**
- Create: `src/fplopt/ingest/espn.py`
- Test: `tests/test_ingest_espn.py`

- [ ] **Step 1: Write the failing tests**

```python
import json
from datetime import UTC, datetime

import pytest

from fplopt.ingest.espn import (
    archive_period,
    backfill_espn,
    events_to_fetch,
    recent_periods,
    snapshot_espn_recent,
    espn_team_ids,
    summary_endpoint,
)
from fplopt.ingest.jobs import SnapshotError
from fplopt.ingest.raw_store import RawStore

NOW = datetime(2026, 10, 27, 3, 0, tzinfo=UTC)


def event(event_id, date, home, away, completed=True, state="post"):
    return {
        "id": event_id,
        "date": date,
        "status": {
            "type": {
                "name": "STATUS_FULL_TIME" if completed else "STATUS_SCHEDULED",
                "completed": completed,
                "state": state,
            }
        },
        "competitions": [
            {
                "competitors": [
                    {"homeAway": "home", "team": {"id": home}},
                    {"homeAway": "away", "team": {"id": away}},
                ]
            }
        ],
    }


def summary(event_id, completed=True):
    return {
        "header": {
            "id": event_id,
            "competitions": [{"status": {"type": {"completed": completed}}}],
        },
        "rosters": [],
        "keyEvents": [],
    }


class FakeEspn:
    def __init__(self, boards, fail=()):
        self.boards = boards  # (league, period) -> list of events
        self.fail = set(fail)
        self.calls = []

    def scoreboard(self, league, period):
        self.calls.append(("scoreboard", league, period))
        return json.dumps({"events": self.boards.get((league, period), [])}).encode()

    def summary(self, league, event_id):
        self.calls.append(("summary", league, event_id))
        if event_id in self.fail:
            raise RuntimeError(f"boom {event_id}")
        return json.dumps(summary(event_id)).encode()


PL = frozenset({"359", "363"})


def test_events_to_fetch_keeps_completed_events_with_a_pl_club_after_july_2016():
    events = [
        event("1", "2016-10-25T18:45Z", "359", "1000"),  # Arsenal v lower club
        event("2", "2016-10-25T18:45Z", "1000", "1001"),  # no PL club
        event("3", "2016-05-01T18:45Z", "359", "1000"),  # before 2016/17
        event("4", "2026-11-04T20:00Z", "363", "1000", completed=False, state="pre"),
    ]
    assert events_to_fetch({"events": events}, "eng.league_cup", PL) == ["1"]


def test_events_to_fetch_keeps_every_pl_match_in_eng_1():
    events = [event("9", "2016-08-13T14:00Z", "1000", "1001")]
    assert events_to_fetch({"events": events}, "eng.1", PL) == ["9"]


def test_archive_period_writes_the_scoreboard_and_new_summaries_once(tmp_path):
    store = RawStore(tmp_path)
    espn = FakeEspn({("eng.league_cup", "2016"): [event("1", "2016-10-25T18:45Z", "359", "1000")]})
    sleeps = []
    assert (
        archive_period(store, espn, "eng.league_cup", "2016", PL, NOW, lambda: NOW, sleeps.append)
        == 1
    )
    assert store.latest("espn", summary_endpoint("eng.league_cup", "1")) is not None
    assert store.latest("espn", "eng.league_cup/scoreboard/2016") is not None
    # A second run re-reads the scoreboard but skips the completed summary.
    later = datetime(2026, 10, 28, tzinfo=UTC)
    assert (
        archive_period(
            store, espn, "eng.league_cup", "2016", PL, later, lambda: later, sleeps.append
        )
        == 0
    )
    assert [c[0] for c in espn.calls] == ["scoreboard", "summary", "scoreboard"]


def test_archive_period_fetches_the_others_when_one_summary_fails(tmp_path):
    store = RawStore(tmp_path)
    events = [event(i, "2016-10-25T18:45Z", "359", "1000") for i in ("1", "2")]
    espn = FakeEspn({("eng.league_cup", "2016"): events}, fail={"1"})
    with pytest.raises(SnapshotError, match="eng.league_cup/1"):
        archive_period(store, espn, "eng.league_cup", "2016", PL, NOW, lambda: NOW, lambda s: None)
    assert store.latest("espn", summary_endpoint("eng.league_cup", "2")) is not None


def test_recent_periods_cover_a_week_back_and_sixty_days_ahead():
    assert recent_periods(NOW) == ["202610", "202611", "202612"]


def test_backfill_walks_every_league_and_year(tmp_path):
    espn = FakeEspn({})
    backfill_espn(
        RawStore(tmp_path), espn, PL, first_year=2025, now=lambda: NOW, sleep=lambda s: None
    )
    boards = {(c[1], c[2]) for c in espn.calls if c[0] == "scoreboard"}
    assert ("eng.1", "2025") in boards and ("uefa.super_cup", "2026") in boards
    assert len(boards) == 16


def test_snapshot_recent_fetches_the_recent_months_of_every_league(tmp_path):
    espn = FakeEspn({})
    snapshot_espn_recent(RawStore(tmp_path), espn, PL, now=lambda: NOW, sleep=lambda s: None)
    assert len([c for c in espn.calls if c[0] == "scoreboard"]) == 8 * 3


def test_espn_team_ids_reads_the_config_column():
    from synthetic_raw import REPO_CONFIG

    ids = espn_team_ids(REPO_CONFIG / "teams.csv")
    assert {"359", "363", "360"} <= ids and len(ids) == 35
```

- [ ] **Step 2: Run to see them fail**

Run: `uv run pytest tests/test_ingest_espn.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `src/fplopt/ingest/espn.py`**

```python
"""ESPN scoreboards and match summaries into raw/ (spec 2026-10-27 §1).

Layout: `espn/<league>/scoreboard/<period>/<ts>.json.gz` (period 'YYYY' or 'YYYYMM'; every
scoreboard of a run shares the run's start time, so a run is one schedule snapshot) and
`espn/<league>/summary/<event_id>/<ts>.json.gz`. A summary is fetched for a completed event
with a club of `team_ids` (every event of `eng.1`) kicking off from 1 July 2016, unless one
with a completed status is archived already (resumable; a summary archived before the match
finished is fetched again). Requests are paced by `pause_s`.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from fplopt.adapters.espn import LEAGUES
from fplopt.ingest.jobs import Clock, run_independently, utc_now
from fplopt.ingest.raw_store import RawStore

log = logging.getLogger(__name__)

SOURCE = "espn"
FIRST_YEAR = 2016
FIRST_KICKOFF = datetime(2016, 7, 1, tzinfo=UTC)
RECENT_BACK = timedelta(days=7)
RECENT_AHEAD = timedelta(days=60)  # covers a 6-GW horizon of cup and European fixtures


class EspnSource(Protocol):
    def scoreboard(self, league: str, period: str) -> bytes: ...

    def summary(self, league: str, event_id: str) -> bytes: ...


def scoreboard_endpoint(league: str, period: str) -> str:
    return f"{league}/scoreboard/{period}"


def summary_endpoint(league: str, event_id: str) -> str:
    return f"{league}/summary/{event_id}"


def espn_team_ids(teams_csv: Path) -> frozenset[str]:
    """The `espn` ids of config/teams.csv (csv only: the daily job does not import pandas)."""
    with open(teams_csv, encoding="utf-8", newline="") as fh:
        return frozenset(row["espn"].strip() for row in csv.DictReader(fh) if row["espn"].strip())


def _kickoff(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def events_to_fetch(
    scoreboard: Mapping[str, Any], league: str, team_ids: frozenset[str]
) -> list[str]:
    """Ids of completed events from FIRST_KICKOFF with a club of `team_ids` (eng.1: all)."""
    out = []
    for ev in scoreboard.get("events", []):
        if not ev["status"]["type"].get("completed") or _kickoff(ev["date"]) < FIRST_KICKOFF:
            continue
        clubs = {c["team"]["id"] for c in ev["competitions"][0]["competitors"]}
        if league == "eng.1" or clubs & team_ids:
            out.append(str(ev["id"]))
    return out


def summary_complete(store: RawStore, league: str, event_id: str) -> bool:
    path = store.latest(SOURCE, summary_endpoint(league, event_id))
    if path is None:
        return False
    status = store.read_json(path)["header"]["competitions"][0]["status"]["type"]
    return bool(status.get("completed"))


def archive_period(
    store: RawStore,
    espn: EspnSource,
    league: str,
    period: str,
    team_ids: frozenset[str],
    run_at: datetime,
    now: Clock = utc_now,
    sleep: Callable[[float], None] = time.sleep,
    pause_s: float = 1.0,
) -> int:
    """Archive one scoreboard (at `run_at`) and the summaries it calls for. Every summary is
    attempted; failures are raised together (SnapshotError). Returns summaries written."""
    content = espn.scoreboard(league, period)
    store.write(SOURCE, scoreboard_endpoint(league, period), content, run_at)
    due = [
        event_id
        for event_id in events_to_fetch(json.loads(content), league, team_ids)
        if not summary_complete(store, league, event_id)
    ]
    written: list[str] = []

    def step(event_id: str) -> Callable[[], None]:
        def run() -> None:
            sleep(pause_s)
            store.write(
                SOURCE, summary_endpoint(league, event_id), espn.summary(league, event_id), now()
            )
            written.append(event_id)

        return run

    try:
        run_independently([(f"{league}/{event_id}", step(event_id)) for event_id in due])
    finally:
        log.info(
            "espn %s %s: %d summaries archived (%d due)", league, period, len(written), len(due)
        )
    return len(written)


def _run_periods(
    store: RawStore,
    espn: EspnSource,
    periods: list[str],
    team_ids: frozenset[str],
    now: Clock,
    sleep: Callable[[float], None],
    pause_s: float,
) -> None:
    run_at = now()
    steps = [
        (
            f"espn {league} {period}",
            lambda league=league, period=period: archive_period(
                store, espn, league, period, team_ids, run_at, now, sleep, pause_s
            ),
        )
        for period in periods
        for league in LEAGUES
    ]
    run_independently(steps)


def backfill_espn(
    store: RawStore,
    espn: EspnSource,
    team_ids: frozenset[str],
    *,
    first_year: int = FIRST_YEAR,
    now: Clock = utc_now,
    sleep: Callable[[float], None] = time.sleep,
    pause_s: float = 1.0,
) -> None:
    """Every league's calendar-year scoreboards from `first_year` to this year, and their
    summaries (~7,000 from 2016, 2–3 h at 1 request/s). Resumable."""
    years = [str(year) for year in range(first_year, now().year + 1)]
    _run_periods(store, espn, years, team_ids, now, sleep, pause_s)


def recent_periods(at: datetime) -> list[str]:
    """Months from RECENT_BACK before `at` to RECENT_AHEAD after, as 'YYYYMM'."""
    first, last = at - RECENT_BACK, at + RECENT_AHEAD
    months, (year, month) = [], (first.year, first.month)
    while (year, month) <= (last.year, last.month):
        months.append(f"{year}{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def snapshot_espn_recent(
    store: RawStore,
    espn: EspnSource,
    team_ids: frozenset[str],
    now: Clock = utc_now,
    sleep: Callable[[float], None] = time.sleep,
    pause_s: float = 1.0,
) -> None:
    """Daily: the recent months' scoreboards of every league (the schedule snapshot) and the
    summaries of matches finished since."""
    _run_periods(store, espn, recent_periods(now()), team_ids, now, sleep, pause_s)
```

Note: `archive_period`'s test passes `run_at` positionally then `now`, `sleep` (see Step 1); keep that order.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_ingest_espn.py -v`
Expected: PASS. (`test_backfill_walks_every_league_and_year`: 2 years × 8 leagues = 16.)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/fplopt/ingest/espn.py tests/test_ingest_espn.py && uv run ruff format src/fplopt/ingest/espn.py tests/test_ingest_espn.py
git add src/fplopt/ingest/espn.py tests/test_ingest_espn.py
git commit -m "feat(ingest): archive ESPN scoreboards and match summaries"
```

---

### Task 4: CLI `backfill espn` and the daily step

**Files:**
- Modify: `src/fplopt/ingest/jobs.py:121-141` (`run_daily`)
- Modify: `src/fplopt/cli.py` (`JOBS`, `build_parser`, module docstring)
- Test: `tests/test_jobs.py`, `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_jobs.py` (reuse the module's fake FPL/odds sources and store fixture; names as in the existing `run_daily` tests):

```python
def test_run_daily_runs_the_espn_step_when_given(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "fplopt.ingest.espn.snapshot_espn_recent",
        lambda store, espn, team_ids, now, sleep: calls.append(team_ids),
    )
    store = RawStore(tmp_path)
    jobs.run_daily(
        store, FakeFpl(), None, espn=(object(), frozenset({"359"})), sleep=lambda s: None
    )
    assert calls == [frozenset({"359"})]
```

(If the module's fake FPL source has another name, use it; the point is that `espn=(source, team_ids)` adds a step named `espn`.)

In `tests/test_cli.py`:

```python
def test_backfill_espn_is_a_command():
    args = cli.build_parser().parse_args(["backfill", "espn", "--from-season", "2020"])
    assert (args.group, args.command, args.from_season) == ("backfill", "espn", 2020)
    assert "backfill espn" in cli.JOBS
```

- [ ] **Step 2: Run to see them fail**

Run: `uv run pytest tests/test_jobs.py tests/test_cli.py -k espn -v`
Expected: FAIL.

- [ ] **Step 3: `run_daily` gains an optional ESPN step** (before `post-lockdown`):

```python
def run_daily(
    store: RawStore,
    fpl: FplSource,
    odds: OddsSource | None,
    now: Clock = utc_now,
    *,
    football_data: FootballDataSource | None = None,
    espn: tuple[Any, frozenset[str]] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Daily archive: FPL, odds, the current football-data CSV (if a source is given), ESPN's
    recent scoreboards and summaries (if `espn` = (source, club ids) is given), then the
    post-lockdown step, last so it reads the bootstrap the FPL step just archived."""
    ...
    if espn is not None:
        from fplopt.ingest import espn as espn_ingest

        source, team_ids = espn
        steps.append(
            ("espn", lambda: espn_ingest.snapshot_espn_recent(store, source, team_ids, now, sleep))
        )
    steps.append(("post-lockdown", ...))  # unchanged
```

(The import is local: `fplopt.ingest.espn` imports `fplopt.ingest.jobs`.)

- [ ] **Step 4: CLI wiring** in `src/fplopt/cli.py`:

```python
from fplopt.adapters.espn import EspnClient
from fplopt.ingest import espn as espn_ingest

TEAMS_CSV = Path("config/teams.csv")
...
    "snapshot daily": lambda c: jobs.run_daily(
        c.store,
        c.fpl,
        c.odds,
        football_data=FootballDataClient(c.http),
        espn=(EspnClient(c.http), espn_ingest.espn_team_ids(TEAMS_CSV)),
    ),
    ...
    "backfill espn": lambda c: espn_ingest.backfill_espn(
        c.store, EspnClient(c.http), espn_ingest.espn_team_ids(TEAMS_CSV), first_year=c.args.from_season
    ),
```

In `build_parser`, the backfill choices become `["element-summary", "football-data", "vaastav", "fplcache", "espn"]` and the `--from-season` help: `"football-data, espn: first season's start year (default 2016)"`. Add `espn` to the module docstring's backfill list ("`espn` (ESPN PL/cup/European summaries; ~2–3 h)").

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_jobs.py tests/test_cli.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/fplopt/ingest/jobs.py src/fplopt/cli.py tests/test_jobs.py tests/test_cli.py
git commit -m "feat(cli): backfill espn, and ESPN in the daily snapshot"
```

---

### Task 5: Parse a summary (`fplopt.build.espn.parse_summary`)

Minutes rule: a clock of `s` seconds is minute `min(s // 60 + 1, end)` (ESPN's display minute; `end` = 120 for `STATUS_FINAL_AET` / `STATUS_FINAL_PEN`, else 90). A starter is on from 0, a substitute from his `Substitution` event (participants `[in, out]`); off at his own substitution, his red card, or `end`. `minutes = off − on` (0 for an unused substitute). Shoot-out events (`shootout: true`) are ignored.

**Files:**
- Create: `src/fplopt/build/espn.py`
- Test: `tests/test_build_espn.py`

- [ ] **Step 1: Write the failing tests** (`tests/test_build_espn.py`; the payload helper is reused by later tasks)

```python
import pandas as pd

from fplopt.build.espn import parse_summary


def player(athlete_id, name, starter, sub_in=False, sub_out=False):
    return {
        "starter": starter,
        "subbedIn": sub_in,
        "subbedOut": sub_out,
        "athlete": {"id": str(athlete_id), "displayName": name},
    }


def key_event(text, seconds, team, *athletes, shootout=False):
    return {
        "type": {"text": text},
        "clock": {"value": float(seconds)},
        "team": {"id": team},
        "participants": [{"athlete": {"id": str(a)}} for a in athletes],
        "shootout": shootout,
    }


def summary_payload(
    event_id, kickoff, home, away, home_roster, away_roster, events=(), status="STATUS_FULL_TIME"
):
    competitors = [
        {"homeAway": "home", "team": {"id": home}},
        {"homeAway": "away", "team": {"id": away}},
    ]
    return {
        "header": {
            "id": str(event_id),
            "competitions": [
                {
                    "date": kickoff,
                    "competitors": competitors,
                    "status": {"type": {"name": status, "completed": True, "state": "post"}},
                }
            ],
        },
        "rosters": [
            {"homeAway": "home", "team": {"id": home}, "roster": home_roster},
            {"homeAway": "away", "team": {"id": away}, "roster": away_roster},
        ],
        "keyEvents": list(events),
    }


def palmer_match():
    """Chelsea 0-2 Spurs, GW8 2026/27, trimmed: Palmer yellow 71', red (second yellow) 80'."""
    home = [
        player(296395, "Cole Palmer", True),
        player(353951, "Jorrel Hato", True, sub_out=True),
        player(362486, "Marco Palestra", False, sub_in=True),
        player(1, "Unused Sub", False),
    ]
    away = [
        player(194450, "James Maddison", True, sub_out=True),
        player(266696, "Conor Gallagher", False, sub_in=True),
    ]
    events = [
        key_event("Substitution", 3015, "367", 266696, 194450),
        key_event("Substitution", 3867, "363", 362486, 353951),
        key_event("Yellow Card", 4253, "363", 296395),
        key_event("Red Card", 4789, "363", 296395),
        key_event("Yellow Card", 5400, "367", 266696),
    ]
    return summary_payload(401878767, "2026-10-24T16:30Z", "363", "367", home, away, events)


def test_parse_summary_match_row():
    match, _ = parse_summary(palmer_match(), "eng.1")
    assert match == {
        "espn_event_id": 401878767,
        "league": "eng.1",
        "season": 2026,
        "kickoff_time": pd.Timestamp("2026-10-24T16:30Z"),
        "home_espn_id": "363",
        "away_espn_id": "367",
        "status": "STATUS_FULL_TIME",
        "extra_time": False,
    }


def test_parse_summary_minutes_and_cards():
    _, players = parse_summary(palmer_match(), "eng.1")
    rows = {p["espn_athlete_id"]: p for p in players}
    assert len(rows) == 6
    palmer = rows[296395]
    assert (palmer["minutes"], palmer["minute_off"], palmer["yellows"]) == (80, 80, 1)
    assert palmer["red"] and palmer["second_yellow"] and palmer["starter"]
    assert (rows[353951]["minutes"], rows[362486]["minute_on"], rows[362486]["minutes"]) == (
        65,
        65,
        25,
    )
    assert (rows[194450]["minutes"], rows[266696]["minutes"]) == (51, 39)
    assert rows[266696]["yellows"] == 1 and not rows[266696]["red"]
    assert rows[1]["minutes"] == 0 and pd.isna(rows[1]["minute_on"])
    assert rows[296395]["espn_team_id"] == "363" and rows[194450]["espn_team_id"] == "367"


def test_a_straight_red_is_not_a_second_yellow_and_extra_time_runs_to_120():
    home = [player(10, "A", True), player(11, "B", False, sub_in=True)]
    away = [player(20, "C", True, sub_out=True), player(21, "D", False, sub_in=True)]
    events = [
        key_event("Red Card", 1200, "359", 10),
        key_event("Substitution", 6600, "359", 11, 10),  # ignored: 10 already off
        key_event("Substitution", 5000, "1000", 21, 20),
        key_event("Yellow Card", 7300, "1000", 21, shootout=True),
    ]
    payload = summary_payload(
        5, "2016-10-25T18:45Z", "359", "1000", home, away, events, status="STATUS_FINAL_PEN"
    )
    match, players = parse_summary(payload, "eng.league_cup")
    rows = {p["espn_athlete_id"]: p for p in players}
    assert match["extra_time"] and match["season"] == 2016
    assert rows[10]["red"] and not rows[10]["second_yellow"] and rows[10]["minutes"] == 21
    assert rows[21]["minutes"] == 120 - 84 and rows[21]["yellows"] == 0
```

- [ ] **Step 2: Run to see them fail**

Run: `uv run pytest tests/test_build_espn.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement the parser** (start of `src/fplopt/build/espn.py`; the module docstring is completed in Task 6)

```python
"""ESPN summaries -> espn_match, espn_player_match, espn_schedule_snapshot, espn_map (spec
2026-10-27 §1). See the module docstring's sections below for the tables and checks."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from fplopt.seasons import season_start_year

log = logging.getLogger(__name__)

EXTRA_TIME_STATUSES = frozenset({"STATUS_FINAL_AET", "STATUS_FINAL_PEN"})


def _minute(seconds: float, end: int) -> int:
    """ESPN's display minute (3015 s -> 51'), capped at the match end (stoppage time)."""
    return min(int(seconds // 60) + 1, end)


def parse_summary(payload: dict[str, Any], league: str) -> tuple[dict, list[dict]]:
    """One summary -> its match row and one row per squad player (lineups, minutes, cards)."""
    competition = payload["header"]["competitions"][0]
    status = competition["status"]["type"]["name"]
    end = 120 if status in EXTRA_TIME_STATUSES else 90
    kickoff = pd.Timestamp(competition["date"]).tz_convert("UTC")
    sides = {c["homeAway"]: str(c["team"]["id"]) for c in competition["competitors"]}
    match = {
        "espn_event_id": int(payload["header"]["id"]),
        "league": league,
        "season": season_start_year(kickoff.to_pydatetime()),
        "kickoff_time": kickoff,
        "home_espn_id": sides["home"],
        "away_espn_id": sides["away"],
        "status": status,
        "extra_time": end == 120,
    }
    on: dict[int, int] = {}
    off: dict[int, int] = {}
    yellows: dict[int, int] = {}
    red_at: dict[int, int] = {}
    second_yellow: set[int] = set()
    events = sorted(payload.get("keyEvents", []), key=lambda e: e.get("clock", {}).get("value", 0))
    for ev in events:
        if ev.get("shootout"):
            continue
        text = ev["type"]["text"]
        ids = [int(p["athlete"]["id"]) for p in ev.get("participants", [])]
        minute = _minute(ev["clock"]["value"], end)
        if text == "Substitution" and len(ids) == 2:
            player_in, player_out = ids
            if player_out in off:  # already sent off or replaced: a data glitch, skip
                continue
            on.setdefault(player_in, minute)
            off[player_out] = minute
        elif text == "Yellow Card" and ids:
            yellows[ids[0]] = yellows.get(ids[0], 0) + 1
        elif text == "Red Card" and ids:
            if yellows.get(ids[0], 0) > 0:
                second_yellow.add(ids[0])
            red_at[ids[0]] = minute
            off.setdefault(ids[0], minute)
    players = []
    for roster in payload.get("rosters", []):
        team = str(roster["team"]["id"])
        for entry in roster.get("roster", []):
            athlete = int(entry["athlete"]["id"])
            start = 0 if entry.get("starter") else on.get(athlete)
            stop = off.get(athlete, end) if start is not None else None
            players.append(
                {
                    "espn_athlete_id": athlete,
                    "espn_event_id": match["espn_event_id"],
                    "espn_team_id": team,
                    "athlete_name": entry["athlete"].get("displayName", ""),
                    "starter": bool(entry.get("starter")),
                    "minute_on": start if start is not None else pd.NA,
                    "minute_off": stop if stop is not None else pd.NA,
                    "minutes": max(stop - start, 0) if start is not None else 0,
                    "yellows": yellows.get(athlete, 0),
                    "red": athlete in red_at,
                    "second_yellow": athlete in second_yellow,
                }
            )
    return match, players
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_build_espn.py -v`
Expected: PASS. (Maddison off at 3015 s = 51'; Gallagher 90 − 51 = 39; Palestra on 65', 25 minutes; D on 84' after 5000 s, `end` 120 → 36.)

- [ ] **Step 5: Commit**

```bash
git add src/fplopt/build/espn.py tests/test_build_espn.py
git commit -m "feat(build): parse ESPN match summaries (minutes, cards)"
```

---

### Task 6: `espn_match`, `espn_player_match`, `espn_schedule_snapshot`

Rules:
- Summaries: the newest file per `(league, event_id)`; only completed ones. A match is kept if either side resolves to a `team_key` (`TeamResolver.find("espn", id)`). `fixture_key` for `eng.1` from `fixture` on `(season, home_team_key, away_team_key)`; an `eng.1` match that does not resolve or join is a build error. `event_time` = kickoff, `available_at` = kickoff + 3 h (`AVAILABLE_AFTER`).
- Players: rows of kept matches whose `espn_team_id` resolves (`team_key`), with the match's `season`, `league`, `event_time`, `available_at`.
- Schedule snapshot: every scoreboard file's events with `status.type.state == "pre"` kicking off after the file's timestamp, with a side that resolves; `snapshot_at` = `event_time` = `available_at` = the file's timestamp.

**Files:**
- Modify: `src/fplopt/build/espn.py`
- Modify: `tests/synthetic_raw.py` (World helpers)
- Test: `tests/test_build_espn.py`

- [ ] **Step 1: World helpers** (`tests/synthetic_raw.py`, methods of `World`):

```python
def add_espn_summary(self, league: str, payload: dict, at: datetime = RUN) -> None:
    event_id = payload["header"]["id"]
    self.store.write("espn", f"{league}/summary/{event_id}", json.dumps(payload).encode(), at)


def add_espn_scoreboard(self, league: str, period: str, events: list, at: datetime = RUN) -> None:
    body = json.dumps({"events": events}).encode()
    self.store.write("espn", f"{league}/scoreboard/{period}", body, at)
```

- [ ] **Step 2: Write the failing tests** (append to `tests/test_build_espn.py`)

```python
from datetime import UTC, datetime

import pytest
from synthetic_raw import REPO_CONFIG, RUN, TEAM_CODES, World

from fplopt.build import build
from fplopt.build.espn import AVAILABLE_AFTER, EspnBuildError, build_espn_tables
from fplopt.build.teams import team_dim_from_config

ESPN_OF = dict(
    team_dim_from_config(REPO_CONFIG / "teams.csv")[["team_key", "espn_id"]]
    .dropna()
    .itertuples(index=False)
)


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


ALL = ["fixture", "gameweek", "player_match", "player_gw", "player_season", "player_dim"]


def built_world(world):
    """Season 2023 (one player per club, 90' every match) built through player_match."""
    world.add_vaastav_season(2023)
    world.write_player_snapshot()
    build(ALL, world.ctx)
    return world.ctx.table("fixture")


def athlete(team_key):
    """The synthetic player of a club as ESPN lists him (players_raw: First<i> Second<i>)."""
    i = TEAM_CODES.index(team_key) + 1
    return player(team_key * 10, f"First{i} Second{i}", True)


def pl_payload(fx, minutes_home=90):
    home, away = ESPN_OF[fx.home_team_key], ESPN_OF[fx.away_team_key]
    events = []
    if minutes_home != 90:  # sent off: off the pitch at `minutes_home`
        events = [key_event("Red Card", (minutes_home - 1) * 60, home, fx.home_team_key * 10)]
    kickoff = fx.kickoff_time.strftime("%Y-%m-%dT%H:%MZ")
    return summary_payload(
        fx.fixture_key,
        kickoff,
        home,
        away,
        [athlete(fx.home_team_key)],
        [athlete(fx.away_team_key)],
        events,
    )


def test_tables_join_pl_fixtures_and_keep_cup_matches_of_pl_clubs(world):
    fixtures = built_world(world)
    first = next(fixtures.itertuples(index=False))
    world.add_espn_summary("eng.1", pl_payload(first))
    ars = ESPN_OF[3]
    cup = summary_payload(
        77, "2023-09-26T18:45Z", ars, "1000", [athlete(3)], [player(5, "Lower", True)]
    )
    world.add_espn_summary("eng.league_cup", cup)
    world.add_espn_summary(
        "eng.league_cup", summary_payload(78, "2023-09-26T18:45Z", "1000", "1001", [], [])
    )
    tables = build_espn_tables(world.ctx)
    match = tables["espn_match"].set_index("espn_event_id")
    assert set(match.index) == {first.fixture_key, 77}
    assert match.loc[first.fixture_key, "fixture_key"] == first.fixture_key
    assert pd.isna(match.loc[77, "fixture_key"]) and match.loc[77, "home_team_key"] == 3
    assert pd.isna(match.loc[77, "away_team_key"])
    assert (match["available_at"] == match["event_time"] + AVAILABLE_AFTER).all()
    players = tables["espn_player_match"]
    assert set(players["espn_athlete_id"]) == {
        first.home_team_key * 10,
        first.away_team_key * 10,
        30,
    }


def test_an_unknown_pl_club_fails_the_build(world):
    built_world(world)
    world.add_espn_summary(
        "eng.1", summary_payload(1, "2023-08-12T14:00Z", "999999", ESPN_OF[3], [], [])
    )
    with pytest.raises(EspnBuildError, match="eng.1 event 1"):
        build_espn_tables(world.ctx)


def test_the_newest_summary_of_an_event_wins(world):
    fixtures = built_world(world)
    first = next(fixtures.itertuples(index=False))
    world.add_espn_summary("eng.1", pl_payload(first, minutes_home=30), at=RUN)
    world.add_espn_summary("eng.1", pl_payload(first), at=datetime(2026, 10, 7, tzinfo=UTC))
    players = build_espn_tables(world.ctx)["espn_player_match"]
    assert players["minutes"].tolist() == [90, 90]


def test_schedule_snapshot_lists_scheduled_matches_of_pl_clubs(world):
    built_world(world)
    at = datetime(2026, 10, 27, 3, tzinfo=UTC)
    events = [
        {
            "id": "501",
            "date": "2026-10-28T19:45Z",
            "status": {"type": {"name": "STATUS_SCHEDULED", "completed": False, "state": "pre"}},
            "competitions": [
                {
                    "competitors": [
                        {"homeAway": "home", "team": {"id": ESPN_OF[14]}},
                        {"homeAway": "away", "team": {"id": ESPN_OF[8]}},
                    ]
                }
            ],
        },
        {
            "id": "502",
            "date": "2026-10-28T19:45Z",
            "status": {"type": {"name": "STATUS_SCHEDULED", "completed": False, "state": "pre"}},
            "competitions": [
                {
                    "competitors": [
                        {"homeAway": "home", "team": {"id": "1000"}},
                        {"homeAway": "away", "team": {"id": "1001"}},
                    ]
                }
            ],
        },
    ]
    world.add_espn_scoreboard("eng.league_cup", "202610", events, at=at)
    snap = build_espn_tables(world.ctx)["espn_schedule_snapshot"]
    assert snap["espn_event_id"].tolist() == [501]
    row = snap.iloc[0]
    assert (row["home_team_key"], row["away_team_key"], row["league"]) == (14, 8, "eng.league_cup")
    assert row["snapshot_at"] == row["available_at"] == pd.Timestamp(at)
```

- [ ] **Step 3: Run to see them fail**

Run: `uv run pytest tests/test_build_espn.py -v`
Expected: FAIL (`ImportError: build_espn_tables`).

- [ ] **Step 4: Implement** (append to `src/fplopt/build/espn.py`; add `import pandera.pandas as pa` to its imports)

```python
from fplopt.build.common import UTC_US, BuildContext
from fplopt.build.teams import TeamResolver
from fplopt.ingest.raw_store import RawStore

SOURCE = "espn"
AVAILABLE_AFTER = pd.Timedelta(hours=3)  # VERIFY (spec §1): lineups/cards final by then


class EspnBuildError(ValueError):
    """ESPN data that does not fit: an unknown PL club, a missing PL fixture, a bad mapping."""


def _item_dirs(store: RawStore, kind: str) -> list[tuple[str, str]]:
    """(league, item) of every `espn/<league>/<kind>/<item>` directory, sorted."""
    root = store.root / SOURCE
    if not root.is_dir():
        return []
    return [(d.parent.parent.name, d.name) for d in sorted(root.glob(f"*/{kind}/*")) if d.is_dir()]


MATCH_SCHEMA = pa.DataFrameSchema(
    {
        "espn_event_id": pa.Column("int64", unique=True),
        "league": pa.Column(str),
        "season": pa.Column("int64"),
        "kickoff_time": pa.Column(UTC_US),
        "home_espn_id": pa.Column(str),
        "away_espn_id": pa.Column(str),
        "home_team_key": pa.Column("Int64", nullable=True),
        "away_team_key": pa.Column("Int64", nullable=True),
        "fixture_key": pa.Column("Int64", nullable=True),
        "status": pa.Column(str),
        "extra_time": pa.Column(bool),
        "event_time": pa.Column(UTC_US),
        "available_at": pa.Column(UTC_US),
    },
    strict=True,
    ordered=True,
)
MATCH_SORT_BY = ("kickoff_time", "espn_event_id")

PLAYER_SCHEMA = pa.DataFrameSchema(
    {
        "espn_athlete_id": pa.Column("int64"),
        "espn_event_id": pa.Column("int64"),
        "season": pa.Column("int64"),
        "league": pa.Column(str),
        "team_key": pa.Column("int64"),
        "athlete_name": pa.Column(str),
        "starter": pa.Column(bool),
        "minute_on": pa.Column("Int64", nullable=True),
        "minute_off": pa.Column("Int64", nullable=True),
        "minutes": pa.Column("int64", pa.Check.in_range(0, 120)),
        "yellows": pa.Column("int64", pa.Check.ge(0)),
        "red": pa.Column(bool),
        "second_yellow": pa.Column(bool),
        "event_time": pa.Column(UTC_US),
        "available_at": pa.Column(UTC_US),
    },
    checks=[
        pa.Check(
            lambda df: ~df.duplicated(["espn_athlete_id", "espn_event_id"]),
            error="(espn_athlete_id, espn_event_id) must be unique",
        )
    ],
    strict=True,
    ordered=True,
)
PLAYER_SORT_BY = ("espn_event_id", "espn_athlete_id")

SCHEDULE_SCHEMA = pa.DataFrameSchema(
    {
        "snapshot_at": pa.Column(UTC_US),
        "espn_event_id": pa.Column("int64"),
        "league": pa.Column(str),
        "season": pa.Column("int64"),
        "kickoff_time": pa.Column(UTC_US),
        "home_team_key": pa.Column("Int64", nullable=True),
        "away_team_key": pa.Column("Int64", nullable=True),
        "event_time": pa.Column(UTC_US),
        "available_at": pa.Column(UTC_US),
    },
    checks=[
        pa.Check(
            lambda df: ~df.duplicated(["snapshot_at", "espn_event_id"]),
            error="(snapshot_at, espn_event_id) must be unique",
        )
    ],
    strict=True,
    ordered=True,
)
SCHEDULE_SORT_BY = ("snapshot_at", "espn_event_id")

_INT64 = ("espn_event_id", "espn_athlete_id", "season", "team_key", "minutes", "yellows")
_NULLABLE_INT = ("home_team_key", "away_team_key", "fixture_key", "minute_on", "minute_off")
_TIMES = ("kickoff_time", "event_time", "available_at", "snapshot_at")
_BOOLS = ("extra_time", "starter", "red", "second_yellow")


def _frame(rows: list[dict], schema: pa.DataFrameSchema) -> pd.DataFrame:
    """`rows` as a frame with exactly the schema's columns and dtypes (also when empty)."""
    df = pd.DataFrame(rows, columns=list(schema.columns))
    for name in df.columns:
        if name in _INT64:
            df[name] = df[name].astype("int64")
        elif name in _NULLABLE_INT:
            df[name] = df[name].astype("Int64")
        elif name in _TIMES:
            df[name] = pd.to_datetime(df[name], utc=True).astype(UTC_US)
        elif name in _BOOLS:
            df[name] = df[name].astype(bool)
        else:
            df[name] = df[name].astype(str)
    return df


def _sides(competitors: list[dict]) -> dict[str, str]:
    return {c["homeAway"]: str(c["team"]["id"]) for c in competitors}


def _summaries(
    ctx: BuildContext, resolver: TeamResolver, problems: list[str]
) -> tuple[list[dict], list[dict]]:
    """Match and player rows from the newest completed summary of every event."""
    fixture = ctx.table("fixture")
    fixture_of = {
        (int(s), int(h), int(a)): int(k)
        for s, h, a, k in zip(
            fixture["season"],
            fixture["home_team_key"],
            fixture["away_team_key"],
            fixture["fixture_key"],
            strict=True,
        )
    }
    matches: list[dict] = []
    players: list[dict] = []
    for league, event_id in _item_dirs(ctx.store, "summary"):
        path = ctx.store.latest(SOURCE, f"{league}/summary/{event_id}")
        payload = ctx.store.read_json(path)
        if not payload["header"]["competitions"][0]["status"]["type"].get("completed"):
            continue
        match, rows = parse_summary(payload, league)
        home = resolver.find("espn", match["home_espn_id"])
        away = resolver.find("espn", match["away_espn_id"])
        fixture_key = None
        if league == "eng.1":
            fixture_key = fixture_of.get((match["season"], home or -1, away or -1))
            if fixture_key is None:
                problems.append(
                    f"eng.1 event {match['espn_event_id']} ({match['home_espn_id']} v "
                    f"{match['away_espn_id']}, {match['kickoff_time']:%Y-%m-%d}) has no fixture"
                )
                continue
        if home is None and away is None:
            continue
        times = {
            "event_time": match["kickoff_time"],
            "available_at": match["kickoff_time"] + AVAILABLE_AFTER,
        }
        matches.append(
            {**match, "home_team_key": home, "away_team_key": away, "fixture_key": fixture_key}
            | times
        )
        for row in rows:
            team_key = resolver.find("espn", row.pop("espn_team_id"))
            if team_key is not None:
                players.append(
                    {**row, "season": match["season"], "league": league, "team_key": team_key}
                    | times
                )
    return matches, players


def _schedule(ctx: BuildContext, resolver: TeamResolver) -> list[dict]:
    """Every scoreboard file is a snapshot (taken at its timestamp) of the scheduled matches
    of PL clubs it lists."""
    rows: list[dict] = []
    for league, period in _item_dirs(ctx.store, "scoreboard"):
        for ts, path in ctx.store.entries(SOURCE, f"{league}/scoreboard/{period}"):
            snapshot_at = pd.Timestamp(ts)
            for ev in ctx.store.read_json(path).get("events", []):
                kickoff = pd.Timestamp(ev["date"]).tz_convert("UTC")
                if ev["status"]["type"].get("state") != "pre" or kickoff <= snapshot_at:
                    continue
                sides = _sides(ev["competitions"][0]["competitors"])
                home = resolver.find("espn", sides["home"])
                away = resolver.find("espn", sides["away"])
                if home is None and away is None:
                    continue
                rows.append(
                    {
                        "snapshot_at": snapshot_at,
                        "espn_event_id": int(ev["id"]),
                        "league": league,
                        "season": season_start_year(kickoff.to_pydatetime()),
                        "kickoff_time": kickoff,
                        "home_team_key": home,
                        "away_team_key": away,
                        "event_time": snapshot_at,
                        "available_at": snapshot_at,
                    }
                )
    return rows


def build_espn_tables(ctx: BuildContext) -> dict[str, pd.DataFrame]:
    """espn_match, espn_player_match and espn_schedule_snapshot (the mapping: `map_espn`)."""
    resolver = TeamResolver(ctx.table("team_dim"))
    problems: list[str] = []
    matches, players = _summaries(ctx, resolver, problems)
    if problems:
        raise EspnBuildError(f"{len(problems)} ESPN problems: " + "; ".join(problems[:20]))
    schedule = _frame(_schedule(ctx, resolver), SCHEDULE_SCHEMA)
    return {
        "espn_match": _frame(matches, MATCH_SCHEMA),
        "espn_player_match": _frame(players, PLAYER_SCHEMA),
        "espn_schedule_snapshot": schedule.drop_duplicates(["snapshot_at", "espn_event_id"]),
    }
```

Also use `_sides` in `parse_summary` (`sides = _sides(competition["competitors"])`).

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_build_espn.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
uv run ruff check src/fplopt/build/espn.py tests && uv run ruff format src/fplopt/build/espn.py tests/test_build_espn.py tests/synthetic_raw.py
git add src/fplopt/build/espn.py tests/test_build_espn.py tests/synthetic_raw.py
git commit -m "feat(build): espn_match, espn_player_match and espn_schedule_snapshot"
```

---

### Task 7: Generalise the Understat matcher by id column

The helpers hard-code `understat_id` / `us_minutes`. Add keyword parameters with the Understat defaults, so existing callers and tests are unchanged.

**Files:**
- Modify: `src/fplopt/build/understat.py:303-478`
- Test: `tests/test_build_understat.py` (existing tests must still pass) and one new test

- [ ] **Step 1: Write the failing test** (append to `tests/test_build_understat.py`)

```python
def test_matcher_helpers_take_another_id_column():
    src = pd.DataFrame({"espn_athlete_id": [7, 7], "fixture_key": [1, 2], "minutes_src": [90, 88]})
    fpl = pd.DataFrame(
        {
            "player_key": [3, 3],
            "fixture_key": [1, 2],
            "minutes": [90, 90],
            "season": [2023, 2023],
            "goals_scored": [0, 0],
        }
    )
    pairs = candidate_pairs(src, fpl, id_col="espn_athlete_id", minutes_col="minutes_src")
    assert pairs[["espn_athlete_id", "player_key", "together", "shared"]].values.tolist() == [
        [7, 3, 2, 2]
    ]
    pairs["name_score"] = name_scores(
        pairs, {7: "Ann Bee"}, {3: {"Ann Bee"}}, id_col="espn_athlete_id"
    )
    empty = read_overrides(REPO_CONFIG, source="espn", id_col="espn_athlete_id")
    mapping = map_understat(pairs, empty, id_col="espn_athlete_id")
    assert mapping[["espn_athlete_id", "player_key", "method"]].values.tolist() == [[7, 3, "auto"]]
```

(Import `REPO_CONFIG` from `synthetic_raw` if not already.)

- [ ] **Step 2: Run to see it fail**

Run: `uv run pytest tests/test_build_understat.py::test_matcher_helpers_take_another_id_column -v`
Expected: FAIL (`unexpected keyword argument 'id_col'`).

- [ ] **Step 3: Parametrise** — in each function replace the literal column names:

```python
def candidate_pairs(
    upm: pd.DataFrame,
    fpl_apps: pd.DataFrame,
    *,
    id_col: str = "understat_id",
    minutes_col: str = "us_minutes",
) -> pd.DataFrame:
    """<id_col> x player_key pairs … (docstring unchanged otherwise)."""
    us = upm[[id_col, "fixture_key", minutes_col]]
    joined = us.merge(fpl_apps[["player_key", "fixture_key", "minutes"]], on="fixture_key")
    joined["close"] = (joined[minutes_col] - joined["minutes"]).abs() <= MINUTES_TOLERANCE
    pairs = (
        joined.groupby([id_col, "player_key"])["close"]
        .agg(together="size", shared="sum")
        .reset_index()
    )
    us_apps = us.groupby(id_col).size().rename("us_apps")
    covered = fpl_apps[fpl_apps["fixture_key"].isin(set(us["fixture_key"]))]
    fpl_counts = covered.groupby("player_key").size().rename("fpl_apps")
    pairs = pairs.join(us_apps, on=id_col).join(fpl_counts, on="player_key")
    ...  # overlap / coverage unchanged


def name_scores(pairs, us_names, fpl_names, *, id_col: str = "understat_id") -> pd.Series:
    ...
        [score(u, k) for u, k in zip(pairs[id_col], pairs["player_key"], strict=True)],


def greedy_assign(pairs: pd.DataFrame, *, id_col: str = "understat_id") -> pd.DataFrame:
    ranked = pairs[acceptable(pairs)].sort_values(
        ["coverage", "shared", "name_score", id_col, "player_key"], ...
    )
    ...
    for idx, uid, key in zip(ranked.index, ranked[id_col], ranked["player_key"], strict=True):


def read_overrides(
    config_dir: Path, *, source: str = "understat", id_col: str = "understat_id"
) -> pd.DataFrame:
    """`source` rows of config/overrides.csv as <id_col>, player_key (both Int64): …"""
    out = pd.DataFrame({id_col: pd.Series(dtype="Int64"), "player_key": pd.Series(dtype="Int64")})
    ...
    df = df[df["source"].str.strip() == source]
    ...
    out = pd.DataFrame({id_col: ints(df["source_id"]), "player_key": ints(df["player_key"])})
    both_blank = out[id_col].isna() & out["player_key"].isna()
    if both_blank.any() or out.dropna(subset=[id_col])[id_col].duplicated().any():
        raise ValueError(f"{path}: {source} overrides need a unique source_id or a player_key")
    if out.dropna(subset=["player_key"])["player_key"].duplicated().any():
        raise ValueError(f"{path}: player_key overridden twice for {source}")


def map_understat(
    pairs: pd.DataFrame, overrides: pd.DataFrame, *, id_col: str = "understat_id"
) -> pd.DataFrame:
    chosen = greedy_assign(pairs, id_col=id_col).assign(method="auto")
    if len(overrides):
        overridden_ids = set(overrides[id_col].dropna().astype(int))
        ...
            ~chosen[id_col].isin(overridden_ids) & ~chosen["player_key"].isin(overridden_keys)
        forced = overrides.dropna(subset=[id_col]).astype({id_col: "int64"})
        scored = forced.merge(pairs.astype({"player_key": "Int64"}), on=[id_col, "player_key"], how="left")
        ...
    out = pd.DataFrame({id_col: chosen[id_col].astype("int64"), ...})
    ...
    return out[[id_col, *MAP_COLUMNS[1:]]].reset_index(drop=True)
```

- [ ] **Step 4: Run every Understat test**

Run: `uv run pytest tests/test_build_understat.py -v`
Expected: PASS (all, old and new).

- [ ] **Step 5: Commit**

```bash
git add src/fplopt/build/understat.py tests/test_build_understat.py
git commit -m "refactor(understat): matcher helpers take the source's id column"
```

---

### Task 8: `espn_map` and the build checks; register the builder

Mapping: ESPN `eng.1` appearances (`espn_player_match` with `league == "eng.1"`, `minutes > 0`, `fixture_key` from `espn_match`) against `fpl_appearances(player_match)`, `name_scores` over `player_season` names, `map_understat(..., id_col="espn_athlete_id")`, overrides `source=espn`. Columns: `espn_athlete_id, player_key, method, shared, overlap, coverage, name_score, event_time, available_at` (static: epoch).

Checks (`espn_problems`; raise `EspnBuildError` listing up to 20), only for seasons with any `eng.1` ESPN match (`covered`):
1. every finished `fixture` of a covered season has an `eng.1` match;
2. every FPL player with minutes > 0 in a covered season is mapped (or exempt by an override with a blank `source_id`);
3. minutes oracle: every auto pair agrees (|Δ| ≤ 5) in ≥ 90% of the fixtures where both have a row and either played (`MIN_MINUTES_AGREEMENT`);
4. reds: over covered fixtures and mapped players, ESPN `red` disagrees with `player_match.red_cards > 0` in at most 2% of FPL red cards (`MAX_RED_DISAGREEMENT`; logged in full).

**Files:**
- Modify: `src/fplopt/build/espn.py`, `src/fplopt/build/__init__.py`, `src/fplopt/build/tables.py`
- Test: `tests/test_build_espn.py`, `tests/test_build_tables.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_build_espn.py`)

```python
from fplopt.build.espn import build_espn


def pl_world(world):
    """Season 2023 with an ESPN eng.1 summary for every fixture (one player a side, 90')."""
    fixtures = built_world(world)
    for fx in fixtures.itertuples(index=False):
        world.add_espn_summary("eng.1", pl_payload(fx))
    return fixtures


def test_build_espn_maps_every_player_and_writes_four_tables(world):
    pl_world(world)
    build(["espn"], world.ctx)
    mapping = pd.read_parquet(world.ctx.table_path("espn_map"))
    pm = world.ctx.table("player_match")
    assert len(mapping) == pm["player_key"].nunique() == 20
    assert (mapping["method"] == "auto").all()
    for name in ("espn_match", "espn_player_match", "espn_schedule_snapshot"):
        assert world.ctx.table_path(name).exists()


def test_a_missing_pl_fixture_fails_the_build(world):
    fixtures = built_world(world)
    for fx in list(fixtures.itertuples(index=False))[1:]:
        world.add_espn_summary("eng.1", pl_payload(fx))
    with pytest.raises(EspnBuildError, match="1 finished PL fixtures have no ESPN match"):
        build(["espn"], world.ctx)


def test_no_espn_data_builds_empty_tables(world):
    built_world(world)
    build(["espn"], world.ctx)
    assert pd.read_parquet(world.ctx.table_path("espn_map")).empty
```

In `tests/test_build_tables.py`, the kinds test: `understat_map` and `espn_map` are static, `espn_schedule_snapshot` is a snapshot table:

```python
assert {n for n, k in kinds.items() if k == "static"} == {
    "team_dim",
    "player_dim",
    "understat_map",
    "espn_map",
}
...
assert {n for n, k in kinds.items() if k == "snapshot"} == {
    "player_snapshot",
    "fixture_snapshot",
    "odds_snapshot",
    "espn_schedule_snapshot",
}
```

and `assert TABLES["espn_map"].public_columns == ("espn_athlete_id", "player_key")` in the static-tables test. The full synthetic build (`test_synthetic_build_all_writes_exactly_the_registered_tables`) then needs the ESPN tables written even without ESPN data (empty).

- [ ] **Step 2: Run to see them fail**

Run: `uv run pytest tests/test_build_espn.py tests/test_build_tables.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement** (append to `src/fplopt/build/espn.py`)

```python
from fplopt.build.common import EPOCH, write_table
from fplopt.build.understat import (
    MINUTES_TOLERANCE,
    candidate_pairs,
    fpl_appearances,
    fpl_name_variants,
    map_understat,
    name_scores,
    read_overrides,
)

MIN_MINUTES_AGREEMENT = 0.9
MAX_RED_DISAGREEMENT = 0.02
ID = "espn_athlete_id"

MAP_SCHEMA = pa.DataFrameSchema(
    {
        ID: pa.Column("int64", unique=True),
        "player_key": pa.Column(
            "Int64", pa.Check(lambda s: ~s.duplicated() | s.isna()), nullable=True
        ),
        "method": pa.Column(str, pa.Check.isin(["auto", "override"])),
        "shared": pa.Column("Int64", nullable=True),
        "overlap": pa.Column("Float64", nullable=True),
        "coverage": pa.Column("Float64", nullable=True),
        "name_score": pa.Column("Float64", nullable=True),
        "event_time": pa.Column(UTC_US, pa.Check.eq(EPOCH)),
        "available_at": pa.Column(UTC_US, pa.Check.eq(EPOCH)),
    },
    strict=True,
    ordered=True,
)
MAP_SORT_BY = (ID,)


def pl_appearances(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """ESPN eng.1 rows with a fixture: espn_athlete_id, fixture_key, minutes_src, red, name."""
    match = tables["espn_match"].dropna(subset=["fixture_key"])
    rows = tables["espn_player_match"].merge(
        match[["espn_event_id", "fixture_key"]], on="espn_event_id"
    )
    rows = rows.assign(fixture_key=rows["fixture_key"].astype("int64"), minutes_src=rows["minutes"])
    return rows[[ID, "fixture_key", "minutes_src", "red", "athlete_name"]]


def map_espn(ctx: BuildContext, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    espn = pl_appearances(tables)
    played = espn[espn["minutes_src"] > 0]
    pairs = candidate_pairs(
        played, fpl_appearances(ctx.table("player_match")), id_col=ID, minutes_col="minutes_src"
    )
    names = dict(zip(espn[ID], espn["athlete_name"], strict=False))
    pairs["name_score"] = name_scores(
        pairs, names, fpl_name_variants(ctx.table("player_season")), id_col=ID
    )
    overrides = read_overrides(ctx.config_dir, source="espn", id_col=ID)
    return map_understat(pairs, overrides, id_col=ID)


def espn_problems(
    ctx: BuildContext, tables: dict[str, pd.DataFrame], mapping: pd.DataFrame
) -> list[str]:
    match = tables["espn_match"]
    covered = set(match.loc[match["league"] == "eng.1", "season"])
    if not covered:
        return []
    problems: list[str] = []
    fixture = ctx.table("fixture")
    finished = fixture[fixture["finished"] & fixture["season"].isin(covered)]
    missing = finished[
        ~finished["fixture_key"].isin(set(match["fixture_key"].dropna().astype(int)))
    ]
    if len(missing):
        problems.append(
            f"{len(missing)} finished PL fixtures have no ESPN match: "
            f"{missing['fixture_key'].head(5).tolist()}"
        )
    pm = ctx.table("player_match")
    pm = pm[pm["season"].isin(covered)]
    exempt = set(
        read_overrides(ctx.config_dir, source="espn", id_col=ID)
        .query(f"{ID}.isna()", engine="python")["player_key"]
        .dropna()
        .astype(int)
    )
    mapped = set(mapping["player_key"].dropna().astype(int))
    unmapped = set(pm.loc[pm["minutes"] > 0, "player_key"]) - mapped - exempt
    if unmapped:
        problems.append(
            f"{len(unmapped)} FPL players with minutes are not mapped: {sorted(unmapped)[:10]}"
        )
    espn = pl_appearances(tables).merge(mapping[[ID, "player_key"]].dropna(), on=ID)
    both = espn.merge(
        pm[["player_key", "fixture_key", "minutes", "red_cards"]], on=["player_key", "fixture_key"]
    )
    both = both.assign(player_key=both["player_key"].astype("int64"))
    played = both[(both["minutes"] > 0) | (both["minutes_src"] > 0)]
    agree = (
        ((played["minutes"] - played["minutes_src"]).abs() <= MINUTES_TOLERANCE)
        .groupby(played[ID])
        .mean()
    )
    auto = set(mapping.loc[mapping["method"] == "auto", ID])
    bad = agree[(agree < MIN_MINUTES_AGREEMENT) & agree.index.isin(auto)]
    if len(bad):
        problems.append(
            f"{len(bad)} pairs agree on minutes in < 90% of fixtures: {bad.head(10).round(2).to_dict()}"
        )
    fpl_reds = both["red_cards"] > 0
    wrong = both[fpl_reds != both["red"]]
    if len(wrong):
        log.info(
            "ESPN/FPL red cards disagree in %d rows: %s",
            len(wrong),
            wrong[[ID, "player_key", "fixture_key", "red", "red_cards"]].to_dict("records"),
        )
    if len(wrong) > MAX_RED_DISAGREEMENT * max(int(fpl_reds.sum()), 1):
        problems.append(f"red cards disagree in {len(wrong)} of {int(fpl_reds.sum())} FPL reds")
    return problems


def build_espn(ctx: BuildContext) -> pd.DataFrame:
    """Writes espn_match, espn_player_match, espn_schedule_snapshot and espn_map."""
    tables = build_espn_tables(ctx)
    mapping = map_espn(ctx, tables)
    problems = espn_problems(ctx, tables, mapping)
    if problems:
        raise EspnBuildError(f"{len(problems)} ESPN problems: " + "; ".join(problems))
    specs = {
        "espn_match": (MATCH_SCHEMA, MATCH_SORT_BY),
        "espn_player_match": (PLAYER_SCHEMA, PLAYER_SORT_BY),
        "espn_schedule_snapshot": (SCHEDULE_SCHEMA, SCHEDULE_SORT_BY),
    }
    for name, (schema, sort_by) in specs.items():
        write_table(tables[name], name, schema, ctx.data_dir, sort_by)
    write_table(mapping, "espn_map", MAP_SCHEMA, ctx.data_dir, MAP_SORT_BY)
    log.info(
        "espn_map: %d athletes (%d by override)",
        len(mapping),
        int((mapping["method"] == "override").sum()),
    )
    return tables["espn_player_match"]
```

An empty `map_espn` (no ESPN rows) must return an empty frame with the map columns: if `played` is empty, return `pd.DataFrame({c: pd.Series(dtype=t) for c, t in ...})` matching `MAP_SCHEMA` (int64 id, Int64 key, str method, Int64 shared, Float64 ×3, UTC_US ×2).

Register (`src/fplopt/build/__init__.py`): import `espn`; `"espn": Builder(espn.build_espn, None)` in `BUILDERS` (comment: "Writes espn_match, espn_player_match, espn_schedule_snapshot and espn_map."); append `"espn"` to `ORDER` after `"team_match"`.

`src/fplopt/build/tables.py`:

```python
(
    TableSpec(
        "espn_map",
        "static",
        ("espn_athlete_id",),
        public_columns=("espn_athlete_id", "player_key"),
        visible_via=("espn_player_match", "espn_athlete_id"),
    ),
)
...
(TableSpec("espn_match", "event", ("espn_event_id",)),)
(TableSpec("espn_player_match", "event", ("espn_athlete_id", "espn_event_id")),)
...
(TableSpec("espn_schedule_snapshot", "snapshot", ("snapshot_at", "espn_event_id"), "snapshot_at"),)
```

- [ ] **Step 4: Run the build tests**

Run: `uv run pytest tests/test_build_espn.py tests/test_build_tables.py tests/test_build_understat.py tests/test_features_architecture.py -v`
Expected: PASS.

- [ ] **Step 5: Full suite and lint**

Run: `uv run ruff check . && uv run ruff format --check . && uv run pytest -q`
Expected: all pass (≈10 min).

- [ ] **Step 6: Commit**

```bash
git add src/fplopt/build tests
git commit -m "feat(build): espn_map (ESPN athlete -> player_key) and ESPN build checks"
```

---

### Task 9: Docs

**Files:** `docs/PLAN.md`, `CLAUDE.md`, `deploy/README.md`, `src/fplopt/build/espn.py` (docstring)

- [ ] **Step 1:** `src/fplopt/build/espn.py` module docstring: the four tables (kind, key, `available_at`), the minutes rule, the mapping rules and the four checks, as in Tasks 5–8, plus a "Results on raw/ <date>" paragraph filled in after Task 10 (counts: matches per league, athletes mapped, overrides, minutes agreement, red disagreements).
- [ ] **Step 2:** `docs/PLAN.md`: in the data-sources section add ESPN (what, where, unofficial-API risk, tables); in §6.3 change "Minutes features with no source are dropped: European/cup matches, manager tenure, age" to "… manager tenure, age (European/cup matches: ESPN, spec 2026-10-27; features in PR 3)".
- [ ] **Step 3:** `CLAUDE.md` commands: `uv run fplopt backfill football-data|vaastav|fplcache|espn` and "`espn`: ESPN PL/cup/European summaries (~2–3 h; resumable)". `deploy/README.md`: the same backfill line under the other backfills, and that `snapshot daily` now also fetches ESPN.
- [ ] **Step 4: Commit**

```bash
git add docs/PLAN.md CLAUDE.md deploy/README.md src/fplopt/build/espn.py
git commit -m "docs: ESPN cup and European data"
```

---

### Task 10: Backfill and build on the server, then the PR

The server (`ssh fplopt-server`, `~/fpl-optimizer`) runs the live bot from main; do not deploy the branch there. Use a separate checkout so `raw/` is the live archive but the code is the branch:

- [ ] **Step 1:** Push the branch; on the server: `git -C ~/fpl-optimizer fetch -q origin && git -C ~/fpl-optimizer worktree add ~/fpl-espn origin/feat/espn-cups && cd ~/fpl-espn && ~/.local/bin/uv sync -q --locked --all-extras && ln -s ~/fpl-optimizer/raw raw`. The backfill appends to the live `raw/` (append-only, new `espn/` folder only), which is what the daily job will extend after merge.
- [ ] **Step 2:** Run the backfill in the foreground (Bash `run_in_background`): `cd ~/fpl-espn && ~/.local/bin/uv run fplopt backfill espn`. Expected: ~2–3 h, log lines "espn <league> <year>: N summaries archived". Re-run once if any step failed (resumable).
- [ ] **Step 3:** Build into a scratch data dir: `cd ~/fpl-espn && cp -r ~/fpl-optimizer/data data && FPLOPT_DATA_DIR=$PWD/data ~/.local/bin/uv run fplopt build espn` (check `fplopt.settings` for the data-dir variable name). Expected: passes the checks. On a failure, inspect the listed offenders: add `config/overrides.csv` rows (`source=espn`) for genuine mismatches, or fix the parser; record each in the docstring's results paragraph.
- [ ] **Step 4:** Fill in the results paragraph (Task 9 Step 1) with the real counts; commit.
- [ ] **Step 5:** Open the PR (`gh pr create`, body: summary, results, test plan; ends with the Claude Code line). Merge and deploy after review per the user; after deploy the daily timer (01:31 UTC) runs the ESPN step; `git worktree remove ~/fpl-espn` (keep `raw/espn`).

---

## PR 2 and PR 3

Written after PR 1 lands (they depend on the real tables): `2026-10-27-espn-cups-2-bans.md` (spec §2) and `2026-10-27-espn-cups-3-features.md` (spec §3).
