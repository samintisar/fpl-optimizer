"""`player_snapshot`: one row per player per archived bootstrap-static snapshot, from the
fplcache mirror (2021-04 onwards) and our own archive (both kept; `source` distinguishes them,
they overlap from 2026-10-05). PLAN §3; Phase 1b plan Task 6.

- `season` is `bootstrap_season(payload)`, never the snapshot's timestamp (hundreds of
  snapshots around the summer disagree with the calendar rule).
- Assistant managers (`element_type == 5`, 2024-25) are dropped. `team_key` is the snapshot's
  own `teams[]` id -> code (team ids reshuffle every season).
- `selected_by_percent`, `ep_next`, `ep_this`, `form` arrive as strings -> Float64. Keys an
  older snapshot lacks (e.g. `team_join_date` before 2024-12) -> null; the always-present keys
  (`id, code, team, element_type, now_cost, status, transfers_*_event, cost_change_event`)
  are required and a missing one fails the build naming the file.
- `event_time = available_at = snapshot_at`: everything in a snapshot was public when it was
  taken.

Memory: ~8,000 snapshots, ~5.7M rows. Snapshots are parsed one at a time (optionally in a
process pool, results consumed in order through a bounded window), collected into chunks of
at most `chunk_snapshots` snapshots that never straddle a season, and each chunk is
pandera-validated and written as one Parquet row group. Rows come out in key order
(snapshot_at, source, element_id) without a global sort, and the bytes depend only on the
raw files and the chunk size, not on `jobs`. The file is written to `.tmp` and renamed only
after every check passed.

Checks: the per-chunk schema (dtypes, ranges, unique key — a snapshot never spans chunks, so
per-chunk uniqueness is global uniqueness); every season from `first_season` (2020: the
mirror starts in April 2021) to the newest is present; and, only if `player_dim` has been
built, every player_key is in it — missing keys are logged as a warning, not a failure (a
player deleted mid-season may be absent from end-of-season players_raw).
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pyarrow
import pyarrow.parquet as pq

from fplopt.build.common import UTC_US, BuildContext, fplcache_and_own_bootstraps, validate_table
from fplopt.ingest.raw_store import RawStore
from fplopt.seasons import bootstrap_season

log = logging.getLogger(__name__)

NAME = "player_snapshot"
FIRST_SEASON = 2020
CHUNK_SNAPSHOTS = 500
DEFAULT_JOBS = max(1, min(8, (os.cpu_count() or 2) - 1))
IN_FLIGHT_PER_JOB = 8  # parsed-ahead snapshots per worker (bounds memory)
MANAGER = 5
STATUSES = ("a", "d", "i", "n", "s", "u")
SOURCES = ("fplcache", "fpl")
MAX_MISSING_SHOWN = 20

DATE32 = pd.ArrowDtype(pyarrow.date32())

# Element key -> (column, kind). kinds: int (required), Int64, float_str, str (required),
# str_opt, utc_str, date_str. Optional kinds read missing keys as null.
_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("id", "element_id", "int"),
    ("code", "player_key", "int"),
    ("element_type", "element_type", "int"),
    ("now_cost", "now_cost", "int"),
    ("status", "status", "str"),
    ("chance_of_playing_next_round", "chance_of_playing_next_round", "Int64"),
    ("chance_of_playing_this_round", "chance_of_playing_this_round", "Int64"),
    ("news", "news", "str_opt"),
    ("news_added", "news_added", "utc_str"),
    ("selected_by_percent", "selected_by_percent", "float_str"),
    ("ep_next", "ep_next", "float_str"),
    ("ep_this", "ep_this", "float_str"),
    ("form", "form", "float_str"),
    ("penalties_order", "penalties_order", "Int64"),
    ("corners_and_indirect_freekicks_order", "corners_and_indirect_freekicks_order", "Int64"),
    ("direct_freekicks_order", "direct_freekicks_order", "Int64"),
    ("team_join_date", "team_join_date", "date_str"),
    ("transfers_in_event", "transfers_in_event", "int"),
    ("transfers_out_event", "transfers_out_event", "int"),
    ("cost_change_event", "cost_change_event", "int"),
)
_REQUIRED = frozenset({"int", "str"})

COLUMNS = (
    "snapshot_at",
    "source",
    "season",
    "element_id",
    "player_key",
    "team_key",
    "element_type",
    "now_cost",
    "status",
    "chance_of_playing_next_round",
    "chance_of_playing_this_round",
    "news",
    "news_added",
    "selected_by_percent",
    "ep_next",
    "ep_this",
    "form",
    "penalties_order",
    "corners_and_indirect_freekicks_order",
    "direct_freekicks_order",
    "team_join_date",
    "transfers_in_event",
    "transfers_out_event",
    "cost_change_event",
    "event_time",
    "available_at",
)

_ORDER_CHECK = pa.Check.ge(1)
SCHEMA = pa.DataFrameSchema(
    {
        "snapshot_at": pa.Column(UTC_US),
        "source": pa.Column(str, pa.Check.isin(SOURCES)),
        "season": pa.Column("int64", pa.Check.ge(2016)),
        "element_id": pa.Column("int64", pa.Check.gt(0)),
        "player_key": pa.Column("int64", pa.Check.gt(0)),
        "team_key": pa.Column("int64", pa.Check.gt(0)),
        "element_type": pa.Column("int64", pa.Check.isin([1, 2, 3, 4])),
        "now_cost": pa.Column("int64", pa.Check.gt(0)),
        "status": pa.Column(str, pa.Check.isin(STATUSES)),
        "chance_of_playing_next_round": pa.Column(
            "Int64", pa.Check.in_range(0, 100), nullable=True
        ),
        "chance_of_playing_this_round": pa.Column(
            "Int64", pa.Check.in_range(0, 100), nullable=True
        ),
        "news": pa.Column(str, nullable=True),
        "news_added": pa.Column(UTC_US, nullable=True),
        "selected_by_percent": pa.Column("Float64", pa.Check.in_range(0, 100), nullable=True),
        "ep_next": pa.Column("Float64", nullable=True),
        "ep_this": pa.Column("Float64", nullable=True),
        "form": pa.Column("Float64", nullable=True),
        "penalties_order": pa.Column("Int64", _ORDER_CHECK, nullable=True),
        "corners_and_indirect_freekicks_order": pa.Column("Int64", _ORDER_CHECK, nullable=True),
        "direct_freekicks_order": pa.Column("Int64", _ORDER_CHECK, nullable=True),
        "team_join_date": pa.Column(DATE32, nullable=True),
        "transfers_in_event": pa.Column("int64", pa.Check.ge(0)),
        "transfers_out_event": pa.Column("int64", pa.Check.ge(0)),
        "cost_change_event": pa.Column("int64"),
        "event_time": pa.Column(UTC_US),
        "available_at": pa.Column(UTC_US),
    },
    checks=[
        pa.Check(
            lambda df: (
                (df["event_time"] == df["snapshot_at"]) & (df["available_at"] == df["snapshot_at"])
            ),
            error="event_time and available_at must equal snapshot_at",
        ),
    ],
    strict=True,
    ordered=True,
    unique=["snapshot_at", "source", "element_id"],
)
SORT_BY = ("snapshot_at", "source", "element_id")


# --- parsing -----------------------------------------------------------------------------


def _float(value: Any) -> float | None:
    return None if value is None else float(value)


def _column(values: list[Any], kind: str) -> Any:
    if kind == "int":
        return np.array(values, dtype=np.int64)
    if kind == "Int64":
        return pd.array(values, dtype="Int64")
    if kind == "float_str":
        return pd.array([_float(v) for v in values], dtype="Float64")
    if kind in ("str", "str_opt"):
        return pd.array(values, dtype="str")
    if kind == "utc_str":
        parsed = pd.to_datetime(pd.Series(values, dtype=object), utc=True, format="ISO8601")
        return parsed.astype(UTC_US).array
    if kind == "date_str":
        return pd.array(
            pyarrow.array(values, type=pyarrow.string()).cast(pyarrow.date32()), dtype=DATE32
        )
    raise ValueError(f"unknown field kind {kind!r}")


def parse_snapshot(entry: tuple[datetime, Path, str]) -> tuple[int, pd.DataFrame]:
    """(season, player rows) of one archived bootstrap-static file. Module-level so a process
    pool can run it. Raises ValueError naming the file on a malformed element."""
    snapshot_at, path, source = entry
    try:
        payload = RawStore.read_json(path)
        season = bootstrap_season(payload)
        team_codes = {int(team["id"]): int(team["code"]) for team in payload["teams"]}
        elements = sorted(
            (e for e in payload["elements"] if e["element_type"] != MANAGER),
            key=lambda e: int(e["id"]),
        )
        raw: dict[str, list[Any]] = {column: [] for _, column, _ in _FIELDS}
        team_keys = []
        for element in elements:
            for key, column, kind in _FIELDS:
                raw[column].append(element[key] if kind in _REQUIRED else element.get(key))
            team_keys.append(team_codes[int(element["team"])])
        data = {column: _column(raw[column], kind) for _, column, kind in _FIELDS}
    except (KeyError, TypeError, ValueError, pyarrow.ArrowInvalid) as exc:
        raise ValueError(f"{source} bootstrap {Path(path).name}: {exc!r}") from exc
    n = len(elements)
    at = pd.Series([pd.Timestamp(snapshot_at)] * n, dtype=UTC_US)
    data |= {
        "snapshot_at": at,
        "source": pd.array([source] * n, dtype="str"),
        "season": np.full(n, season, dtype=np.int64),
        "team_key": np.array(team_keys, dtype=np.int64),
        "event_time": at,
        "available_at": at,
    }
    return season, pd.DataFrame({column: data[column] for column in COLUMNS})


def _parsed(
    entries: list[tuple[datetime, Path, str]], jobs: int
) -> Iterator[tuple[int, pd.DataFrame]]:
    """`parse_snapshot` over `entries`, in order. With jobs > 1 a process pool parses ahead,
    at most `jobs * IN_FLIGHT_PER_JOB` snapshots in flight so results cannot pile up in memory."""
    if jobs <= 1:
        yield from map(parse_snapshot, entries)
        return
    window = jobs * IN_FLIGHT_PER_JOB
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        remaining = iter(entries)
        pending: deque[Future[tuple[int, pd.DataFrame]]] = deque(
            pool.submit(parse_snapshot, entry) for entry in islice(remaining, window)
        )
        while pending:
            result = pending.popleft().result()
            following = next(remaining, None)
            if following is not None:
                pending.append(pool.submit(parse_snapshot, following))
            yield result


def _chunks(
    parsed: Iterable[tuple[int, pd.DataFrame]], chunk_snapshots: int
) -> Iterator[pd.DataFrame]:
    """Consecutive snapshots of one season, at most `chunk_snapshots` per chunk."""
    parts: list[pd.DataFrame] = []
    current = None
    for season, frame in parsed:
        if parts and (season != current or len(parts) >= chunk_snapshots):
            yield pd.concat(parts, ignore_index=True)
            parts = []
        parts.append(frame)
        current = season
    if parts:
        yield pd.concat(parts, ignore_index=True)


# --- checks ------------------------------------------------------------------------------


def _check_seasons(seasons: set[int], first_season: int) -> None:
    expected = set(range(first_season, max(seasons, default=first_season) + 1))
    missing = sorted(expected - seasons)
    if missing:
        raise ValueError(
            f"{NAME}: missing season(s) {', '.join(map(str, missing))} "
            f"(expected every season from {first_season}; got {sorted(seasons)})"
        )


def _report_missing_players(ctx: BuildContext, keys: pd.DataFrame) -> None:
    """Warn about player_keys absent from player_dim; skipped if player_dim isn't built."""
    if not ctx.table_path("player_dim").exists():
        log.info("%s: player_dim not built yet; player_key check skipped", NAME)
        return
    known = set(ctx.table("player_dim")["player_key"].astype(int))
    missing = keys[~keys["player_key"].isin(known)]
    if missing.empty:
        return
    by_season = missing.groupby("season")["player_key"].nunique().to_dict()
    sample = sorted(set(missing["player_key"].astype(int)))[:MAX_MISSING_SHOWN]
    log.warning(
        "%s: %d player_key(s) not in player_dim (per season: %s); first: %s",
        NAME,
        missing["player_key"].nunique(),
        by_season,
        sample,
    )


# --- build -------------------------------------------------------------------------------


def build_player_snapshot(
    ctx: BuildContext,
    jobs: int = DEFAULT_JOBS,
    chunk_snapshots: int = CHUNK_SNAPSHOTS,
    first_season: int | None = FIRST_SEASON,
) -> pd.DataFrame:
    """Write `data/player_snapshot.parquet` chunk by chunk (see module doc). Returns only the
    `season` of every row, for build logging. `first_season=None` skips the season-coverage
    check."""
    started = time.perf_counter()
    entries = fplcache_and_own_bootstraps(ctx.store)
    if not entries:
        raise ValueError(f"{NAME}: no bootstrap-static snapshots in raw/")
    data_dir = Path(ctx.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = ctx.table_path(NAME)
    tmp = data_dir / f"{NAME}.parquet.tmp"
    season_counts: dict[int, int] = {}
    player_keys: list[pd.DataFrame] = []
    writer: pq.ParquetWriter | None = None
    row_groups = 0
    try:
        for chunk in _chunks(_parsed(entries, jobs), chunk_snapshots):
            chunk = validate_table(chunk, NAME, SCHEMA)
            table = pyarrow.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            elif not table.schema.equals(writer.schema, check_metadata=True):
                raise ValueError(f"{NAME}: chunk schema differs from the first chunk's")
            writer.write_table(table, row_group_size=len(table))
            row_groups += 1
            for season, count in chunk["season"].value_counts().items():
                season_counts[int(season)] = season_counts.get(int(season), 0) + int(count)
            player_keys.append(chunk[["season", "player_key"]].drop_duplicates())
            log.debug("%s: chunk %d, %d rows", NAME, row_groups, len(chunk))
        writer.close()
        writer = None
        if first_season is not None:
            _check_seasons(set(season_counts), first_season)
        _report_missing_players(ctx, pd.concat(player_keys, ignore_index=True).drop_duplicates())
        os.replace(tmp, path)
    finally:
        if writer is not None:
            writer.close()
        tmp.unlink(missing_ok=True)
    seasons = sorted(season_counts)
    log.info(
        "%s: %d rows from %d snapshots in %d row groups, %.1f MB, %.1fs",
        NAME,
        sum(season_counts.values()),
        len(entries),
        row_groups,
        path.stat().st_size / 1e6,
        time.perf_counter() - started,
    )
    return pd.DataFrame(
        {
            "season": np.repeat(
                np.array(seasons, dtype=np.int64), [season_counts[s] for s in seasons]
            )
        }
    )
