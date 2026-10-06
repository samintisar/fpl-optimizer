"""One-off historical backfills into raw/: football-data seasons, vaastav, fplcache."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Iterable
from typing import Protocol

from fplopt.adapters.vaastav import PINNED_COMMIT
from fplopt.ingest.jobs import (
    MAX_CONSECUTIVE_FAILURES,
    Clock,
    FootballDataSource,
    run_independently,
    snapshot_football_data,
    snapshot_football_data_current,
    utc_now,
)
from fplopt.ingest.raw_store import RawStore, gzip_bytes
from fplopt.seasons import football_data_code, season_label, season_start_year

log = logging.getLogger(__name__)

FIRST_SEASON = 2016


class VaastavSource(Protocol):
    def tree(self, commit: str) -> dict[str, str]: ...

    def file(self, commit: str, path: str, blob_sha: str) -> bytes: ...


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


# --- vaastav -----------------------------------------------------------------------------

VAASTAV_SEASONS = range(2016, 2026)  # 2016-17 … 2025-26; 2026/27 comes from our own archive
_VAASTAV_FILE = re.compile(
    r"^data/(?P<season>\d{4}-\d{2})/"
    r"(?:players_raw|player_idlist|cleaned_players|fixtures|teams|id_dict|gws/merged_gw"
    r"|understat/[^/]+)\.csv$"
)
VAASTAV_ROOT_FILES = ("data/master_team_list.csv",)


def select_vaastav_paths(paths: Iterable[str]) -> list[str]:
    """The vaastav files we mirror, sorted. Excluded on purpose: `gws/xP*.csv` (lookahead
    leak), `gws/gwN.csv` and `players/` (duplicated by merged_gw), `fbref/`, `managers/`,
    `cleaned_merged_seasons*.csv` (incomplete) and the stale 2026-27 folder."""
    seasons = {season_label(year) for year in VAASTAV_SEASONS}
    return sorted(
        path
        for path in paths
        if path in VAASTAV_ROOT_FILES
        or ((match := _VAASTAV_FILE.match(path)) is not None and match["season"] in seasons)
    )


def vaastav_name(path: str) -> str:
    """Store name of a vaastav file: 'data/2016-17/gws/merged_gw.csv' -> '2016-17/gws/merged_gw'."""
    if not (path.startswith("data/") and path.endswith(".csv")):
        raise ValueError(f"not a vaastav data CSV: {path!r}")
    return path.removeprefix("data/").removesuffix(".csv")


def backfill_vaastav(
    store: RawStore,
    vaastav: VaastavSource,
    commit: str = PINNED_COMMIT,
    now: Clock = utc_now,
    sleep: Callable[[float], None] = time.sleep,
    pause_s: float = 0.05,
    max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES,
) -> int:
    """Mirror the selected vaastav files at `commit` under one run directory:
    raw/vaastav/data/<run_at>/<path without 'data/' and '.csv'>.csv.gz, plus _manifest.json.gz
    (commit, run_at, finished_at, expected, written, failed) written even when aborted.
    Continues past single failures (including paths the store refuses, which are not
    downloaded); aborts after `max_consecutive_failures` in a row; raises if anything failed.
    Returns the number of files written."""
    blobs = vaastav.tree(commit)
    paths = select_vaastav_paths(blobs)
    if not paths:
        raise RuntimeError(f"no vaastav files selected at {commit} (layout changed?)")
    run_at = now()
    written: list[str] = []
    failed: list[str] = []
    consecutive = 0
    try:
        for path in paths:
            try:
                name = vaastav_name(path)
                # Validate the store path before downloading (e.g. ':' is refused).
                store.path_for("vaastav", "data", run_at, suffix=".csv.gz", name=name)
                content = vaastav.file(commit, path, blobs[path])
                store.write_bytes(
                    "vaastav", "data", gzip_bytes(content), run_at, suffix=".csv.gz", name=name
                )
                written.append(path)
                consecutive = 0
            except Exception as exc:
                log.exception("vaastav %s failed", path)
                failed.append(path)
                consecutive += 1
                if consecutive >= max_consecutive_failures:
                    raise RuntimeError(
                        f"aborting vaastav backfill after {consecutive} consecutive failures "
                        f"(last {path}: {type(exc).__name__})"
                    ) from exc
            sleep(pause_s)
    finally:
        manifest = {
            "commit": commit,
            "run_at": run_at.isoformat(),
            "finished_at": now().isoformat(),
            "expected": paths,
            "written": written,
            "failed": failed,
        }
        try:
            store.write("vaastav", "data", json.dumps(manifest).encode(), run_at, name="_manifest")
        except Exception:
            log.exception("could not write vaastav manifest")
    if failed:
        raise RuntimeError(
            f"{len(failed)} of {len(paths)} vaastav files failed (first: {failed[:5]})"
        )
    log.info("archived %d vaastav files at %s", len(paths), commit)
    return len(written)
