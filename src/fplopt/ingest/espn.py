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
