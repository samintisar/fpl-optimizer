"""Archiver jobs: fetch from adapters and append to the raw store."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import httpx

from fplopt.ingest.raw_store import RawStore, gzip_bytes
from fplopt.ingest.schedule import deadlines_from_bootstrap, pre_deadline_due
from fplopt.seasons import football_data_code, season_start_year

log = logging.getLogger(__name__)

Clock = Callable[[], datetime]

MAX_CONSECUTIVE_FAILURES = 10

FOOTBALL_DATA_GRACE_MONTHS = (7, 8)  # new-season CSV may not exist yet in July/August


def utc_now() -> datetime:
    return datetime.now(UTC)


class FplSource(Protocol):
    def bootstrap_static(self) -> bytes: ...

    def fixtures(self) -> bytes: ...

    def element_summary(self, element_id: int) -> bytes: ...


class OddsSource(Protocol):
    def epl_odds(self) -> bytes: ...


class FootballDataSource(Protocol):
    def epl_season(self, start_year: int) -> bytes: ...


def snapshot_fpl(store: RawStore, fpl: FplSource, now: Clock = utc_now) -> list[Path]:
    """Fixtures first, bootstrap last: the tick treats a bootstrap snapshot inside a
    pre-deadline window as done, so a failure part-way must leave the window open."""
    fixtures_content = fpl.fixtures()
    fixtures = store.write("fpl", "fixtures", fixtures_content, now())
    bootstrap_content = fpl.bootstrap_static()
    bootstrap = store.write("fpl", "bootstrap-static", bootstrap_content, now())
    return [fixtures, bootstrap]


def snapshot_odds(store: RawStore, odds: OddsSource | None, now: Clock = utc_now) -> Path | None:
    if odds is None:
        log.warning("ODDS_API_KEY not set; skipping odds snapshot")
        return None
    content = odds.epl_odds()
    return store.write("odds", "soccer_epl", content, now())


class SnapshotError(RuntimeError):
    """One or more sources failed; the others were still archived."""


def run_independently(steps: list[tuple[str, Callable[[], object]]]) -> None:
    """Run every step even if earlier ones fail; then raise one SnapshotError naming each
    failed step (single line)."""
    errors: list[str] = []
    for name, step in steps:
        try:
            step()
        except Exception as exc:
            log.exception("%s snapshot failed", name)
            # First line only: the combined message must stay one line so no source's
            # error is cut off when the CLI alert keeps just the first line.
            summary = str(exc).splitlines()[0] if str(exc) else ""
            errors.append(f"{name}: {type(exc).__name__}: {summary}")
    if errors:
        raise SnapshotError("; ".join(errors))


def snapshot_football_data(
    store: RawStore, fd: FootballDataSource, start_year: int, now: Clock = utc_now
) -> Path:
    """Archive one season's football-data CSV under football-data/E0/<code>, e.g. E0/1617."""
    content = fd.epl_season(start_year)
    return store.write_bytes(
        "football-data",
        f"E0/{football_data_code(start_year)}",
        gzip_bytes(content),
        now(),
        suffix=".csv.gz",
    )


def snapshot_football_data_current(
    store: RawStore, fd: FootballDataSource, now: Clock = utc_now
) -> Path | None:
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


def run_daily(
    store: RawStore,
    fpl: FplSource,
    odds: OddsSource | None,
    now: Clock = utc_now,
    *,
    football_data: FootballDataSource | None = None,
) -> None:
    steps: list[tuple[str, Callable[[], object]]] = [
        ("fpl", lambda: snapshot_fpl(store, fpl, now)),
        ("odds", lambda: snapshot_odds(store, odds, now)),
    ]
    if football_data is not None:
        steps.append(
            ("football-data", lambda: snapshot_football_data_current(store, football_data, now))
        )
    run_independently(steps)


def run_tick(
    store: RawStore, fpl: FplSource, odds: OddsSource | None, now: Clock = utc_now
) -> bool:
    """Run every ~15 min. Inside each pre-deadline window, snapshot FPL and odds once each;
    a source that fails is retried on the next tick. Returns True if anything ran."""
    latest = store.latest("fpl", "bootstrap-static")
    if latest is None:
        log.info("no bootstrap snapshot yet; taking one")
        run_daily(store, fpl, odds, now)
        return True
    deadlines = deadlines_from_bootstrap(store.read_json(latest))
    current = now()
    fpl_deadline = pre_deadline_due(current, deadlines, store.times("fpl", "bootstrap-static"))
    odds_due = (
        odds is not None
        and pre_deadline_due(current, deadlines, store.times("odds", "soccer_epl")) is not None
    )
    steps: list[tuple[str, Callable[[], object]]] = []
    if fpl_deadline is not None:
        log.info("pre-deadline FPL snapshot for deadline %s", fpl_deadline.isoformat())
        steps.append(("fpl", lambda: snapshot_fpl(store, fpl, now)))
    if odds_due:
        log.info("pre-deadline odds snapshot")
        steps.append(("odds", lambda: snapshot_odds(store, odds, now)))
    run_independently(steps)
    return fpl_deadline is not None or odds_due


def backfill_element_summaries(
    store: RawStore,
    fpl: FplSource,
    now: Clock = utc_now,
    pause_s: float = 0.25,
    sleep: Callable[[float], None] = time.sleep,
    max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES,
) -> int:
    """Archive element-summary for every current player under one run timestamp.

    Writes `<run>/_manifest.json.gz` (expected, written and failed ids) even when aborted,
    so incomplete runs are identifiable. The bootstrap it archives counts as a snapshot for
    the pre-deadline tick, so avoid running this inside a pre-deadline window.

    Element summaries in a run share `run_at` but are fetched later; treat the manifest's
    `finished_at` as the run's `available_at`.
    """
    bootstrap = fpl.bootstrap_static()
    run_at = now()
    store.write("fpl", "bootstrap-static", bootstrap, run_at)
    ids = [element["id"] for element in json.loads(bootstrap)["elements"]]
    written: list[int] = []
    failed: list[int] = []
    consecutive = 0
    try:
        for element_id in ids:
            try:
                content = fpl.element_summary(element_id)
                store.write("fpl", "element-summary", content, run_at, name=str(element_id))
                written.append(element_id)
                consecutive = 0
            except Exception as exc:
                log.exception("element-summary %s failed", element_id)
                failed.append(element_id)
                consecutive += 1
                if consecutive >= max_consecutive_failures:
                    raise RuntimeError(
                        f"aborting backfill after {consecutive} consecutive failures "
                        f"(last element {element_id}: {type(exc).__name__})"
                    ) from exc
            sleep(pause_s)
    finally:
        manifest = {
            "run_at": run_at.isoformat(),
            "finished_at": now().isoformat(),
            "expected": ids,
            "written": written,
            "failed": failed,
        }
        try:
            store.write(
                "fpl", "element-summary", json.dumps(manifest).encode(), run_at, name="_manifest"
            )
        except Exception:
            log.exception("could not write backfill manifest")
    if failed:
        raise RuntimeError(f"{len(failed)} of {len(ids)} element summaries failed: {failed}")
    log.info("archived %d element summaries", len(ids))
    return len(ids)
