"""When to take pre-deadline snapshots."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

PRE_DEADLINE_LEAD = timedelta(hours=2)


def deadlines_from_bootstrap(bootstrap: dict[str, Any]) -> list[datetime]:
    """FPL deadlines as timezone-aware UTC datetimes (FPL publishes them in UTC)."""
    deadlines = []
    for event in bootstrap["events"]:
        deadline = datetime.fromisoformat(event["deadline_time"])
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=UTC)
        deadlines.append(deadline.astimezone(UTC))
    return deadlines


def pre_deadline_due(
    now: datetime,
    deadlines: Iterable[datetime],
    taken: Iterable[datetime],
    lead: timedelta = PRE_DEADLINE_LEAD,
) -> datetime | None:
    """Return the deadline whose [deadline - lead, deadline) window contains `now`
    and has no snapshot yet; otherwise None. All datetimes must be timezone-aware."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    taken = list(taken)
    for deadline in deadlines:
        start = deadline - lead
        if start <= now < deadline and not any(start <= t < deadline for t in taken):
            return deadline
    return None
