"""raw/ -> Parquet tables and cross-source ID mapping.

`BUILDERS` maps a table name to its `Builder`; `ORDER` is the dependency order `build()` runs
them in (`fplopt build TABLE|all`). A builder's `run(ctx)` returns the table and `build()`
validates it against `schema` and writes `data/<name>.parquet` sorted by `sort_by`. A builder
with `schema=None` writes its own output(s) and returns a frame only for logging.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from fplopt.build import fixtures, players, teams
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
    "player_season": Builder(
        players.build_player_season, players.PLAYER_SEASON_SCHEMA, players.PLAYER_SEASON_SORT_BY
    ),
    "player_dim": Builder(
        players.build_player_dim, players.PLAYER_DIM_SCHEMA, players.PLAYER_DIM_SORT_BY
    ),
    "player_match": Builder(
        players.build_player_match, players.PLAYER_MATCH_SCHEMA, players.PLAYER_MATCH_SORT_BY
    ),
}
ORDER: list[str] = [
    "team_dim",
    "fixture",
    "gameweek",
    "player_season",
    "player_dim",
    "player_match",
]


def _resolve(names: Iterable[str]) -> list[str]:
    requested = set()
    for name in names:
        if name == "all":
            requested.update(ORDER)
        elif name in BUILDERS:
            requested.add(name)
        else:
            raise ValueError(f"unknown table {name!r}; valid: {', '.join(['all', *ORDER])}")
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
