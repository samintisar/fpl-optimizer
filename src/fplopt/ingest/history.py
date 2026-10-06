"""One-off historical backfills into raw/: football-data seasons, vaastav, fplcache."""

from __future__ import annotations

import json
import logging
import lzma
import re
import tarfile
import time
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from datetime import UTC, datetime
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


class FplcacheSource(Protocol):
    def head_commit(self) -> str: ...

    def tarball(self, commit: str) -> AbstractContextManager[tarfile.TarFile]: ...


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


# --- fplcache ----------------------------------------------------------------------------

_FPLCACHE_MEMBER = re.compile(
    r"^[^/]+/cache/(?P<y>\d{4})/(?P<m>\d{1,2})/(?P<d>\d{1,2})/(?P<H>\d{2})(?P<M>\d{2})\.json\.xz$"
)
MAX_FPLCACHE_MEMBER_BYTES = 64 << 20  # real snapshots are ~0.1-0.3 MB compressed
FPLCACHE_PROGRESS_EVERY = 500


def fplcache_snapshot_time(member_name: str) -> datetime | None:
    """'fplcache-<sha>/cache/2021/4/18/1641.json.xz' -> 2021-04-18 16:41 UTC; None for any
    other file. A snapshot-shaped name with an impossible date raises ValueError."""
    match = _FPLCACHE_MEMBER.match(member_name)
    if match is None:
        return None
    y, m, d, hh, mm = (int(match[key]) for key in ("y", "m", "d", "H", "M"))
    return datetime(y, m, d, hh, mm, tzinfo=UTC)


def _archive_fplcache_member(
    store: RawStore, tar: tarfile.TarFile, member: tarfile.TarInfo, snapshot_at: datetime
) -> bool:
    """Archive one snapshot member. True if written, False if an identical copy exists;
    raises if it cannot be archived."""
    if member.size > MAX_FPLCACHE_MEMBER_BYTES:
        raise ValueError(f"member too large: {member.size} bytes")
    handle = tar.extractfile(member)
    if handle is None:
        raise ValueError("member has no content")
    data = handle.read()
    path = store.path_for("fplcache", "bootstrap-static", snapshot_at, suffix=".json.xz")
    if path.exists():
        if path.read_bytes() == data:
            return False
        raise ValueError("differs from the archived copy (upstream rewrote history?)")
    json.loads(lzma.decompress(data))
    store.write_bytes("fplcache", "bootstrap-static", data, snapshot_at, suffix=".json.xz")
    return True


def backfill_fplcache(store: RawStore, fplcache: FplcacheSource, now: Clock = utc_now) -> int:
    """Mirror every fplcache snapshot into raw/fplcache/bootstrap-static/<snapshot time>.json.xz,
    byte-for-byte. Idempotent: identical files already present are skipped; a present file with
    different bytes is a failure (upstream rewrote history). Each new file must decompress to
    JSON. Writes a run manifest raw/fplcache/runs/<run_at>.json.gz with commit, written,
    skipped, failed (member names + reason) and error (a download/tar failure that stopped the
    run, else null) — also when failures occurred — then raises if anything failed.
    Returns the number of files written."""
    commit = fplcache.head_commit()
    run_at = now()
    written = skipped = 0
    failed: list[dict[str, str]] = []
    error: str | None = None
    try:
        with fplcache.tarball(commit) as tar:
            for member in tar:
                if not member.isfile():
                    continue
                try:
                    snapshot_at = fplcache_snapshot_time(member.name)
                    if snapshot_at is None:
                        continue
                    if _archive_fplcache_member(store, tar, member, snapshot_at):
                        written += 1
                    else:
                        skipped += 1
                except Exception as exc:
                    log.exception("fplcache %s failed", member.name)
                    failed.append({"member": member.name, "reason": _one_line(exc)})
                done = written + skipped + len(failed)
                if done % FPLCACHE_PROGRESS_EVERY == 0:
                    log.info(
                        "fplcache: %d snapshots (%d written, %d skipped, %d failed)",
                        done,
                        written,
                        skipped,
                        len(failed),
                    )
    except BaseException as exc:
        error = _one_line(exc)
        raise
    finally:
        manifest = {
            "commit": commit,
            "run_at": run_at.isoformat(),
            "finished_at": now().isoformat(),
            "written": written,
            "skipped": skipped,
            "failed": failed,
            "error": error,
        }
        try:
            store.write("fplcache", "runs", json.dumps(manifest).encode(), run_at)
        except Exception:
            log.exception("could not write fplcache run manifest")
    total = written + skipped + len(failed)
    if failed:
        raise RuntimeError(
            f"{len(failed)} of {total} fplcache snapshots failed "
            f"(first: {[entry['member'] for entry in failed[:5]]})"
        )
    log.info("fplcache at %s: %d written, %d already archived", commit, written, skipped)
    return written


def _one_line(exc: BaseException) -> str:
    lines = str(exc).splitlines()
    return f"{type(exc).__name__}: {lines[0]}" if lines else type(exc).__name__
