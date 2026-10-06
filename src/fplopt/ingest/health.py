"""Archive health checks: a local dead-man's switch for the archiver timers.

`fplopt check freshness` runs daily from its own timer and fails (Telegram alert) when the
newest bootstrap snapshot is too old, which catches jobs or timers that stopped silently
while the server is up. It can't catch the server itself being down.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from fplopt.ingest.raw_store import RawStore

log = logging.getLogger(__name__)

DEFAULT_MAX_AGE = timedelta(hours=36)


class StaleArchiveError(RuntimeError):
    """The newest bootstrap snapshot is missing or older than allowed."""


def _hours(delta: timedelta) -> str:
    return f"{delta.total_seconds() / 3600:.1f} h"


def check_freshness(
    store: RawStore, now: datetime, max_age: timedelta = DEFAULT_MAX_AGE
) -> timedelta:
    """Age of the newest `raw/fpl/bootstrap-static` snapshot (every daily run and every
    pre-deadline tick writes one). Raises StaleArchiveError if there is none or it is older
    than `max_age`."""
    times = store.times("fpl", "bootstrap-static")
    if not times:
        raise StaleArchiveError(f"no bootstrap-static snapshot in {store.root}")
    age = now - times[-1]
    if age > max_age:
        raise StaleArchiveError(
            f"newest bootstrap-static snapshot is {_hours(age)} old "
            f"({times[-1].isoformat()}; limit {_hours(max_age)})"
        )
    log.info("newest bootstrap-static snapshot is %s old (limit %s)", _hours(age), _hours(max_age))
    return age
