"""`TABLES`: metadata for every table the build writes to `data/` — the single source of truth
for the as-of store, the leakage harness and the tests (Phase 2 plan, Task 1).

Kinds (how `available_at` is set and how as-of reads treat the table):
- `static`: reference data (`available_at` = epoch). The whole table is not as-of: it lists
  future debutants (`player_dim`), clubs' FPL eras (`team_dim.in_fpl`) and match statistics
  fitted on every season (`understat_map`). Feature code reads it only through
  `AsOfView.lookup`: the `public_columns` (identity: key and names) of keys that are visible
  as of the deadline in `visible_via` (table, column).
- `schedule`: known from publication (`available_at` = 1 June of the season's start year, or
  the lockdown after the previous season's last kickoff if later: `schedule_available_at`);
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
    # static tables only: the columns `AsOfView.lookup` returns (the key first) and the
    # as-of (table, column) whose visible values are the keys it may return.
    public_columns: tuple[str, ...] = ()
    visible_via: tuple[str, str] | None = None


def _specs(*specs: TableSpec) -> dict[str, TableSpec]:
    return {spec.name: spec for spec in specs}


TABLES: dict[str, TableSpec] = _specs(
    # static
    TableSpec(
        "team_dim",
        "static",
        ("team_key",),
        public_columns=("team_key", "short_name", "fpl_names"),
        visible_via=("team_rating", "team_key"),
    ),
    TableSpec(
        "player_dim",
        "static",
        ("player_key",),
        public_columns=("player_key", "first_name", "second_name", "web_name"),
        visible_via=("player_season", "player_key"),
    ),
    TableSpec(
        "understat_map",
        "static",
        ("understat_id",),
        public_columns=("understat_id", "player_key"),
        visible_via=("understat_player_match", "understat_id"),
    ),
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
    TableSpec("team_rating", "event", ("team_key", "event_time")),
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
