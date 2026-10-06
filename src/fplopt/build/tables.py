"""`TABLES`: metadata for every table the build writes to `data/` — the single source of truth
for the as-of store, the leakage harness and the tests (Phase 2 plan, Task 1).

Kinds (how `available_at` is set and how as-of reads treat the table):
- `static`: reference data with nothing time-dependent (`available_at` = epoch).
- `schedule`: known from publication (`available_at` = 1 June of the season's start year);
  historically only the final version exists (PLAN §3, §4).
- `event`: one row per thing that happened (or state at a deadline), each with its own
  `available_at`; as-of reads keep rows with `available_at < deadline`.
- `snapshot`: the state at each snapshot time (`snapshot_col`, = `available_at`); as-of
  reads take the newest snapshot before the deadline.

`key` is unique in the table (checked by the tests on a synthetic build).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Kind = Literal["static", "schedule", "event", "snapshot"]
KINDS: tuple[Kind, ...] = ("static", "schedule", "event", "snapshot")


@dataclass(frozen=True)
class TableSpec:
    name: str
    kind: Kind
    key: tuple[str, ...]
    snapshot_col: str | None = None


def _specs(*specs: TableSpec) -> dict[str, TableSpec]:
    return {spec.name: spec for spec in specs}


TABLES: dict[str, TableSpec] = _specs(
    # static
    TableSpec("team_dim", "static", ("team_key",)),
    TableSpec("player_dim", "static", ("player_key",)),
    TableSpec("understat_map", "static", ("understat_id",)),
    # schedule
    TableSpec("schedule", "schedule", ("fixture_key",)),
    TableSpec("gameweek", "schedule", ("season", "gw")),
    # event
    TableSpec("fixture", "event", ("fixture_key",)),
    TableSpec("gameweek_result", "event", ("season", "gw")),
    TableSpec("player_season", "event", ("player_key", "season")),
    TableSpec("player_match", "event", ("player_key", "fixture_key")),
    TableSpec("understat_player_match", "event", ("understat_id", "fixture_key")),
    TableSpec("team_match", "event", ("fixture_key", "team_key")),
    TableSpec("player_gw", "event", ("player_key", "season", "gw")),
    TableSpec("player_gw_ownership", "event", ("player_key", "season", "gw")),
    TableSpec("team_rating", "event", ("team_key", "kickoff_time")),
    # snapshot
    TableSpec(
        "player_snapshot", "snapshot", ("snapshot_at", "source", "player_key"), "snapshot_at"
    ),
    TableSpec("fixture_snapshot", "snapshot", ("snapshot_at", "fixture_key"), "snapshot_at"),
    TableSpec(
        "odds_snapshot",
        "snapshot",
        (
            "fixture_key",
            "source",
            "bookmaker",
            "market",
            "outcome",
            "line",
            "is_closing",
            "snapshot_at",
        ),
        "snapshot_at",
    ),
)
