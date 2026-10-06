"""One-off historical backfills into raw/: football-data seasons, vaastav, fplcache."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from fplopt.ingest.jobs import (
    Clock,
    FootballDataSource,
    run_independently,
    snapshot_football_data,
    snapshot_football_data_current,
    utc_now,
)
from fplopt.ingest.raw_store import RawStore
from fplopt.seasons import football_data_code, season_start_year

log = logging.getLogger(__name__)

FIRST_SEASON = 2016


def backfill_football_data(
    store: RawStore,
    fd: FootballDataSource,
    now: Clock = utc_now,
    sleep: Callable[[float], None] = time.sleep,
    pause_s: float = 1.0,
) -> int:
    """Every EPL season CSV from 2016/17 to the current season. All seasons are attempted;
    failures are collected and raised together (SnapshotError, one line). The current season
    follows the daily job's rule: a 404 in July/August means "not published yet", not an error.
    Returns the number of seasons written."""
    current = season_start_year(now())
    written: list[int] = []

    def step(start_year: int) -> Callable[[], None]:
        def run() -> None:
            if start_year > FIRST_SEASON:
                sleep(pause_s)
            if start_year == current:
                path = snapshot_football_data_current(store, fd, now)
            else:
                path = snapshot_football_data(store, fd, start_year, now)
            if path is not None:
                written.append(start_year)

        return run

    run_independently(
        [
            (f"E0/{football_data_code(year)}", step(year))
            for year in range(FIRST_SEASON, current + 1)
        ]
    )
    log.info("archived %d football-data seasons", len(written))
    return len(written)
