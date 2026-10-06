"""raw/ -> Parquet tables and cross-source ID mapping.

`BUILDERS` maps a table name to its `Builder`; `ORDER` is the dependency order `build()` runs
them in (`fplopt build TABLE|all`). A builder's `run(ctx)` returns the table and `build()`
validates it against `schema` and writes `data/<name>.parquet` sorted by `sort_by`. A builder
with `schema=None` writes its own output(s) and returns a frame only for logging.

Rebuilding `player_match` or `player_dim` writes them without their Understat columns, so
`build()` then also runs `understat` (which fills them) and `team_match` (built from
`player_match`), and logs that it did.

`fplopt.build.tables.TABLES` describes every written table (kind, key); a test checks it
matches what a full build writes.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from fplopt.build import elo, fixtures, odds, players, snapshots, teams, understat
from fplopt.build.common import BuildContext, write_table

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Builder:
    run: Callable[[BuildContext], pd.DataFrame]
    schema: pa.DataFrameSchema | None
    sort_by: tuple[str, ...] = ()


BUILDERS: dict[str, Builder] = {
    "team_dim": Builder(teams.build_team_dim, teams.SCHEMA, teams.SORT_BY),
    "fixture": Builder(fixtures.build_fixture, fixtures.FIXTURE_SCHEMA, fixtures.FIXTURE_SORT_BY),
    "gameweek": Builder(
        fixtures.build_gameweek, fixtures.GAMEWEEK_SCHEMA, fixtures.GAMEWEEK_SORT_BY
    ),
    "gameweek_result": Builder(
        fixtures.build_gameweek_result,
        fixtures.GAMEWEEK_RESULT_SCHEMA,
        fixtures.GAMEWEEK_RESULT_SORT_BY,
    ),
    "schedule": Builder(
        fixtures.build_schedule, fixtures.SCHEDULE_SCHEMA, fixtures.SCHEDULE_SORT_BY
    ),
    "fixture_snapshot": Builder(
        fixtures.build_fixture_snapshot,
        fixtures.FIXTURE_SNAPSHOT_SCHEMA,
        fixtures.FIXTURE_SNAPSHOT_SORT_BY,
    ),
    "player_season": Builder(
        players.build_player_season, players.PLAYER_SEASON_SCHEMA, players.PLAYER_SEASON_SORT_BY
    ),
    "player_dim": Builder(
        players.build_player_dim, players.PLAYER_DIM_SCHEMA, players.PLAYER_DIM_SORT_BY
    ),
    "player_gw": Builder(
        players.build_player_gw, players.PLAYER_GW_SCHEMA, players.PLAYER_GW_SORT_BY
    ),
    "player_gw_ownership": Builder(
        players.build_player_gw_ownership,
        players.PLAYER_GW_OWNERSHIP_SCHEMA,
        players.PLAYER_GW_SORT_BY,
    ),
    "player_match": Builder(
        players.build_player_match, players.PLAYER_MATCH_SCHEMA, players.PLAYER_MATCH_SORT_BY
    ),
    # Writes understat_player_match + understat_map and rewrites player_dim / player_match.
    "understat": Builder(understat.build_understat, None),
    "team_match": Builder(
        understat.build_team_match, understat.TEAM_MATCH_SCHEMA, understat.TEAM_MATCH_SORT_BY
    ),
    "player_snapshot": Builder(snapshots.build_player_snapshot, None),
    "odds_snapshot": Builder(odds.build_odds_snapshot, odds.SCHEMA, odds.SORT_BY),
    "team_rating": Builder(elo.build_team_rating, elo.SCHEMA, elo.SORT_BY),
}
ORDER: list[str] = [
    "team_dim",
    "fixture",
    "gameweek",
    "gameweek_result",
    "schedule",
    "player_match",
    "player_season",
    "player_dim",
    "player_gw",
    "player_gw_ownership",
    "understat",
    "team_match",
    "player_snapshot",
    "fixture_snapshot",
    "odds_snapshot",
    "team_rating",
]


# Tables whose rebuild clears columns the `understat` builder fills, and what then reruns.
UNDERSTAT_TARGETS = ("player_match", "player_dim")
UNDERSTAT_AND_DEPENDENTS = ("understat", "team_match")


def _resolve(names: Iterable[str]) -> list[str]:
    requested = set()
    for name in names:
        if name == "all":
            requested.update(ORDER)
        elif name in BUILDERS:
            requested.add(name)
        else:
            raise ValueError(f"unknown table {name!r}; valid: {', '.join(['all', *ORDER])}")
    if requested.intersection(UNDERSTAT_TARGETS):
        added = [name for name in UNDERSTAT_AND_DEPENDENTS if name not in requested]
        if added:
            log.info(
                "also rebuilding %s: rebuilding %s clears the Understat columns",
                ", ".join(added),
                " and ".join(name for name in UNDERSTAT_TARGETS if name in requested),
            )
            requested.update(added)
    return [name for name in ORDER if name in requested]


def _seasons(df: pd.DataFrame) -> str:
    if "season" not in df.columns or df.empty:
        return "-"
    seasons = sorted(df["season"].dropna().unique())
    return f"{seasons[0]}-{seasons[-1]} ({len(seasons)})"


def build(names: Iterable[str], ctx: BuildContext) -> list[Path]:
    """Build the named tables ('all' = every table) in dependency order. Unknown names fail
    before anything runs. Returns the written paths."""
    paths = []
    for name in _resolve(names):
        builder = BUILDERS[name]
        started = time.perf_counter()
        df = builder.run(ctx)
        if builder.schema is not None:
            write_table(df, name, builder.schema, ctx.data_dir, builder.sort_by)
        ctx.forget(name)
        paths.append(ctx.table_path(name))
        log.info(
            "%s: %d rows, seasons %s, %.1fs",
            name,
            len(df),
            _seasons(df),
            time.perf_counter() - started,
        )
    return paths
