"""Season labels and codes. A season is identified by its start year (2026 = 2026/27)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any

_LABEL = re.compile(r"(\d{4})-(\d{2})")

# The 2025/26 holdout (PLAN §9): no outcome statistics, tuning or leakage-check deadlines
# from it until Phase 6.
HOLDOUT_SEASONS = frozenset({2025})


def season_label(start_year: int) -> str:
    """2026 -> '2026-27' (vaastav folders, config file names)."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def parse_season_label(label: str) -> int:
    """'2026-27' -> 2026. Rejects malformed or non-consecutive labels."""
    match = _LABEL.fullmatch(label)
    if not match or (int(match[1]) + 1) % 100 != int(match[2]):
        raise ValueError(f"not a season label like 2026-27: {label!r}")
    return int(match[1])


def football_data_code(start_year: int) -> str:
    """2026 -> '2627' (football-data.co.uk URL segment)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_start_year(when: datetime) -> int:
    """The season in progress or about to start at `when`: July onwards is the new season."""
    return when.year if when.month >= 7 else when.year - 1


def bootstrap_season(bootstrap: Mapping[str, Any]) -> int:
    """Start year of the season a bootstrap-static payload describes (year of GW1's deadline)."""
    first = min(bootstrap["events"], key=lambda event: event["id"])
    return int(first["deadline_time"][:4])
