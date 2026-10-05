"""When to take pre-deadline snapshots."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Any

PRE_DEADLINE_LEAD = timedelta(hours=2)


def deadlines_from_bootstrap(bootstrap: dict[str, Any]) -> list[datetime]:
    return [datetime.fromisoformat(event["deadline_time"]) for event in bootstrap["events"]]


def pre_deadline_due(
    now: datetime,
    deadlines: Iterable[datetime],
    taken: Iterable[datetime],
    lead: timedelta = PRE_DEADLINE_LEAD,
) -> datetime | None:
    """Return the deadline whose [deadline - lead, deadline) window contains `now`
    and has no snapshot yet; otherwise None."""
    taken = list(taken)
    for deadline in deadlines:
        start = deadline - lead
        if start <= now < deadline and not any(start <= t < deadline for t in taken):
            return deadline
    return None
