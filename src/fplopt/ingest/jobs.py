"""Archiver jobs: fetch from adapters and append to the raw store."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from fplopt.ingest.raw_store import RawStore
from fplopt.ingest.schedule import deadlines_from_bootstrap, pre_deadline_due

log = logging.getLogger(__name__)

Clock = Callable[[], datetime]

MAX_CONSECUTIVE_FAILURES = 10


def utc_now() -> datetime:
    return datetime.now(UTC)


class FplSource(Protocol):
    def bootstrap_static(self) -> bytes: ...

    def fixtures(self) -> bytes: ...

    def element_summary(self, element_id: int) -> bytes: ...


class OddsSource(Protocol):
    def epl_odds(self) -> bytes: ...


def snapshot_fpl(store: RawStore, fpl: FplSource, now: Clock = utc_now) -> list[Path]:
    """Fixtures first, bootstrap last: the tick treats a bootstrap snapshot inside a
    pre-deadline window as done, so a failure part-way must leave the window open."""
    fixtures = store.write("fpl", "fixtures", fpl.fixtures(), now())
    bootstrap = store.write("fpl", "bootstrap-static", fpl.bootstrap_static(), now())
    return [fixtures, bootstrap]


def snapshot_odds(store: RawStore, odds: OddsSource | None, now: Clock = utc_now) -> Path | None:
    if odds is None:
        log.warning("ODDS_API_KEY not set; skipping odds snapshot")
        return None
    return store.write("odds", "soccer_epl", odds.epl_odds(), now())


def run_daily(
    store: RawStore, fpl: FplSource, odds: OddsSource | None, now: Clock = utc_now
) -> None:
    snapshot_fpl(store, fpl, now)
    snapshot_odds(store, odds, now)


def run_tick(
    store: RawStore, fpl: FplSource, odds: OddsSource | None, now: Clock = utc_now
) -> bool:
    """Run every ~15 min: snapshot once inside each pre-deadline window. True if it ran."""
    latest = store.latest("fpl", "bootstrap-static")
    if latest is None:
        log.info("no bootstrap snapshot yet; taking one")
        run_daily(store, fpl, odds, now)
        return True
    deadlines = deadlines_from_bootstrap(store.read_json(latest))
    deadline = pre_deadline_due(now(), deadlines, store.times("fpl", "bootstrap-static"))
    if deadline is None:
        return False
    log.info("pre-deadline snapshot for deadline %s", deadline.isoformat())
    run_daily(store, fpl, odds, now)
    return True


def backfill_element_summaries(
    store: RawStore,
    fpl: FplSource,
    now: Clock = utc_now,
    pause_s: float = 0.25,
    sleep: Callable[[float], None] = time.sleep,
    max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES,
) -> int:
    """Archive element-summary for every current player under one run timestamp."""
    run_at = now()
    bootstrap = fpl.bootstrap_static()
    store.write("fpl", "bootstrap-static", bootstrap, run_at)
    ids = [element["id"] for element in json.loads(bootstrap)["elements"]]
    failed: list[int] = []
    consecutive = 0
    for element_id in ids:
        try:
            content = fpl.element_summary(element_id)
            store.write("fpl", "element-summary", content, run_at, name=str(element_id))
            consecutive = 0
        except Exception:
            log.exception("element-summary %s failed", element_id)
            failed.append(element_id)
            consecutive += 1
            if consecutive >= max_consecutive_failures:
                raise RuntimeError(
                    f"aborting backfill after {consecutive} consecutive failures "
                    f"(last element {element_id})"
                ) from None
        sleep(pause_s)
    if failed:
        raise RuntimeError(f"{len(failed)} of {len(ids)} element summaries failed: {failed[:20]}")
    log.info("archived %d element summaries", len(ids))
    return len(ids)
