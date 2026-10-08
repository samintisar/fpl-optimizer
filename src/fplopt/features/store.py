"""The as-of read path: `DataStore(data_dir).as_of(deadline)` is how everything downstream of
`data/` reads tables (PLAN §4 "Single access path"). It is the only module in
`fplopt.features` that reads files.

A view exposes only rows with `available_at < deadline` (strict; the column, never
`event_time`). Snapshot tables are read with `latest()` (newest snapshot per group before the
deadline); the as-of fixture list with `schedule()`. Static tables are not as-of (they list
future debutants, whole-history flags and statistics), so `table()` refuses them: `lookup()`
returns only their identity columns (`TableSpec.public_columns`) for keys visible at the
deadline.

Performance: each table is loaded once per store, lazily and column by column, and kept
sorted by `available_at` (for snapshot tables: ties ordered so the preferred source comes
last). A view's visible rows are therefore a prefix found with `searchsorted`, and `latest()`
uses a per-(table, by) group index built once per store, so a per-deadline call never scans or
copies a whole table. Stores are not invalidated: build a new one after `data/` changes.

`files_blocked()`: while it is active (in the current context), opening `data/` through a
`DataStore` raises `FilesBlockedError`; the leakage harness runs the feature builders inside
it, so a builder cannot read the real tables behind the view it is given.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from fplopt.build.tables import TABLES, TableSpec

# At equal snapshot times, rows of these sources win in `latest()` (our own archive is
# authoritative over the fplcache mirror). Other sources rank below, in input order.
PREFERRED_SOURCES = ("fpl",)
SCHEDULE_COLUMNS = [
    "fixture_key",
    "season",
    "gw",
    "gw_index",
    "kickoff_time",
    "home_team_key",
    "away_team_key",
    "schedule_source",
]


_FILES_BLOCKED: ContextVar[str | None] = ContextVar("fplopt_files_blocked", default=None)


class FilesBlockedError(RuntimeError):
    """A `DataStore` tried to read `data/` while `files_blocked()` was active."""


@contextmanager
def files_blocked(reason: str) -> Iterator[None]:
    """While active, `DataStore(data_dir=...)` (and loading a table of an existing data_dir
    store) raises FilesBlockedError naming `reason`. In-memory stores still work."""
    token = _FILES_BLOCKED.set(reason)
    try:
        yield
    finally:
        _FILES_BLOCKED.reset(token)


def _check_files_allowed() -> None:
    reason = _FILES_BLOCKED.get()
    if reason is not None:
        raise FilesBlockedError(f"reading data/ through a DataStore is blocked {reason}")


def _utc(deadline: Any) -> pd.Timestamp:
    ts = pd.Timestamp(deadline)
    if ts.tzinfo is None:
        raise ValueError(f"deadline must be tz-aware (UTC), got naive {ts}")
    return ts.tz_convert("UTC")


def _unknown(name: str) -> KeyError:
    return KeyError(f"unknown table {name!r}; known tables: {', '.join(sorted(TABLES))}")


def _available_ns(values: pd.Series, name: str) -> np.ndarray:
    if not isinstance(values.dtype, pd.DatetimeTZDtype):
        raise ValueError(f"{name}.available_at must be tz-aware datetimes, got {values.dtype}")
    if values.isna().any():
        raise ValueError(f"{name}.available_at has {int(values.isna().sum())} missing value(s)")
    return pd.DatetimeIndex(values).tz_convert("UTC").as_unit("ns").asi8


class _Table:
    """One table's columns, loaded on demand and stored in `available_at` order."""

    def __init__(
        self, spec: TableSpec, columns: list[str], load: Callable[[list[str]], pd.DataFrame]
    ) -> None:
        self.spec = spec
        self.columns = columns
        self._load = load
        self._arrays: dict[str, Any] = {}
        self._groups: dict[tuple[str, ...], tuple[np.ndarray, np.ndarray, int]] = {}
        first = ["available_at"]
        if spec.kind == "snapshot" and "source" in columns:
            first.append("source")
        raw = load(first)
        available = _available_ns(raw["available_at"], spec.name)
        if spec.kind == "snapshot" and "source" in raw:
            preferred = raw["source"].isin(PREFERRED_SOURCES).to_numpy(dtype=bool)
            order = np.lexsort((preferred, available))
        else:
            order = np.argsort(available, kind="stable")
        self._order = None if np.array_equal(order, np.arange(len(order))) else order
        self.available = available if self._order is None else available[self._order]
        self.n = len(self.available)
        self._store(raw)

    def _store(self, raw: pd.DataFrame) -> None:
        for column in raw.columns:
            array = raw[column].array
            self._arrays[column] = array if self._order is None else array.take(self._order)

    def require(self, columns: Sequence[str]) -> None:
        unknown = [c for c in columns if c not in self.columns]
        if unknown:
            raise KeyError(
                f"table {self.spec.name!r} has no column(s) {unknown}; "
                f"columns: {', '.join(self.columns)}"
            )
        missing = [c for c in dict.fromkeys(columns) if c not in self._arrays]
        if missing:
            self._store(self._load(missing))

    def visible(self, deadline_ns: int) -> int:
        """Number of rows with available_at < deadline (they are the first rows)."""
        return int(np.searchsorted(self.available, deadline_ns, side="left"))

    def frame(self, columns: Sequence[str], rows: slice | np.ndarray) -> pd.DataFrame:
        self.require(columns)
        return pd.DataFrame({c: self._arrays[c][rows] for c in columns}, copy=True)

    def groups(self, by: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, int]:
        """(keys, positions, n_groups): positions grouped by `by` (nulls form groups),
        increasing within a group; keys = group * n + position, ascending."""
        if by not in self._groups:
            self.require(by)
            codes = np.zeros(self.n, dtype=np.int64)
            n_codes = 1
            for column in by:
                part, uniques = pd.factorize(self._arrays[column], use_na_sentinel=False)
                codes, seen = pd.factorize(codes * len(uniques) + part)
                n_codes = len(seen)
            positions = np.argsort(codes, kind="stable")
            keys = codes[positions].astype(np.int64) * self.n + positions
            self._groups[by] = (keys, positions, n_codes if self.n else 0)
        return self._groups[by]


class DataStore:
    """Loads `data/` tables once (lazily, selected columns) and hands out deadline views.
    The only place in fplopt.features that reads files.

    `tables` supplies in-memory frames instead (tests, the leakage harness); they are not
    modified. With `tables`, only those tables exist."""

    def __init__(
        self, data_dir: Path | str | None = None, tables: Mapping[str, pd.DataFrame] | None = None
    ) -> None:
        if (data_dir is None) == (tables is None):
            raise ValueError("pass exactly one of data_dir and tables")
        if data_dir is not None:
            _check_files_allowed()
        self.data_dir = None if data_dir is None else Path(data_dir)
        self._frames: dict[str, pd.DataFrame] | None = None
        if tables is not None:
            for name in tables:
                if name not in TABLES:
                    raise _unknown(name)
            self._frames = dict(tables)
        self._tables: dict[str, _Table] = {}

    def as_of(self, deadline: Any) -> AsOfView:
        return AsOfView(self, _utc(deadline))

    @property
    def provided_tables(self) -> Mapping[str, pd.DataFrame] | None:
        """The in-memory tables given as `tables` (read-only), or None for a `data_dir`
        store: with `data_dir`, enough to open an equivalent store elsewhere (backtest
        worker processes)."""
        return None if self._frames is None else MappingProxyType(self._frames)

    def _table(self, name: str) -> _Table:
        if name not in TABLES:
            raise _unknown(name)
        if name not in self._tables:
            self._tables[name] = self._open(TABLES[name])
        return self._tables[name]

    def _open(self, spec: TableSpec) -> _Table:
        if self._frames is not None:
            if spec.name not in self._frames:
                raise KeyError(
                    f"table {spec.name!r} was not provided; provided: "
                    f"{', '.join(sorted(self._frames))}"
                )
            frame = self._frames[spec.name].reset_index(drop=True)
            return _Table(spec, list(frame.columns), lambda cols: frame[cols])
        _check_files_allowed()
        path = self.data_dir / f"{spec.name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"table {spec.name!r} has not been built ({path}); run `fplopt build all`"
            )
        names = [n for n in pq.read_schema(path).names if not n.startswith("__index_level_")]

        def load(columns: list[str]) -> pd.DataFrame:
            _check_files_allowed()  # columns load lazily, possibly later
            return pd.read_parquet(path, columns=columns)

        return _Table(spec, names, load)


class AsOfView:
    """The data as known strictly before `deadline` (tz-aware, UTC)."""

    def __init__(self, store: DataStore, deadline: pd.Timestamp) -> None:
        self._store = store
        self.deadline = _utc(deadline)
        self._deadline_ns = self.deadline.as_unit("ns").value

    def _visible(self, name: str) -> tuple[_Table, int]:
        table = self._store._table(name)
        return table, table.visible(self._deadline_ns)

    def table(self, name: str, columns: Sequence[str] | None = None) -> pd.DataFrame:
        """Rows with available_at < deadline (a copy), in available_at order. Not for static
        tables (use `lookup`)."""
        if name in TABLES and TABLES[name].kind == "static":
            raise ValueError(
                f"{name!r} is a static table (not as-of: it covers every season); "
                f"use lookup({name!r}, keys) for its identity columns"
            )
        table, k = self._visible(name)
        return table.frame(table.columns if columns is None else list(columns), slice(0, k))

    def lookup(self, name: str, keys: pd.Series | Sequence[Any]) -> pd.DataFrame:
        """Static tables only: the `public_columns` of the rows whose key is in `keys` and
        is visible as of the deadline (appears in a visible row of the spec's `visible_via`
        table); one row per key, sorted by key. Other keys are dropped."""
        if name not in TABLES:
            raise _unknown(name)
        spec = TABLES[name]
        if spec.kind != "static" or spec.visible_via is None:
            raise ValueError(f"lookup() is for static tables; {name!r} is {spec.kind}")
        key = spec.key[0]
        via_table, via_column = spec.visible_via
        visible = self.table(via_table, columns=[via_column])[via_column]
        wanted = pd.Series(keys).dropna()
        wanted = wanted[wanted.isin(visible)]
        table = self._store._table(name)
        rows = table.frame([key], slice(0, table.n))[key]
        positions = np.flatnonzero(rows.isin(wanted).to_numpy(dtype=bool))
        out = table.frame(list(spec.public_columns), positions)
        out = out.drop_duplicates(key)
        return out.sort_values(key, kind="mergesort").reset_index(drop=True)

    def latest(
        self, name: str, by: Sequence[str], columns: Sequence[str] | None = None
    ) -> pd.DataFrame:
        """Snapshot tables only: per `by` group (nulls are a group; `by=[]` = one group) the
        row of the newest snapshot before the deadline, a 'fpl' row winning a tie. One row
        per group, sorted by `by`; columns = `by` then `columns` (default: all)."""
        table = self._store._table(name)
        if table.spec.kind != "snapshot":
            raise ValueError(f"latest() is for snapshot tables; {name!r} is {table.spec.kind}")
        by = tuple(by)
        wanted = table.columns if columns is None else list(columns)
        out_columns = list(dict.fromkeys([*by, *wanted]))
        k = table.visible(self._deadline_ns)
        keys, positions, n_groups = table.groups(by)
        groups = np.arange(n_groups, dtype=np.int64)
        idx = np.searchsorted(keys, groups * table.n + k, side="left") - 1
        found = (idx >= 0) & (keys[np.maximum(idx, 0)] // max(table.n, 1) == groups)
        rows = np.sort(positions[idx[found]])
        out = table.frame(out_columns, rows)
        if by:
            out = out.sort_values(list(by), kind="mergesort").reset_index(drop=True)
        return out

    def schedule(self, season: int) -> pd.DataFrame:
        """The as-of fixture list of `season`: the newest `fixture_snapshot` of that season
        before the deadline (schedule_source 'snapshot'; GW index from `gameweek`), else the
        final `schedule` ('final'). Sorted by fixture_key."""
        snapshots, k = self._visible("fixture_snapshot")
        newest = self.latest("fixture_snapshot", by=["season"], columns=["snapshot_at"])
        newest = newest.loc[newest["season"] == season, "snapshot_at"]
        if newest.empty:
            out = self.table("schedule", columns=SCHEDULE_COLUMNS)
            out = out[out["season"] == season]
        else:
            # Visible rows are sorted by available_at (= snapshot_at): the snapshot is a block.
            at = pd.Timestamp(newest.iloc[0]).as_unit("ns").value
            lo = int(np.searchsorted(snapshots.available[:k], at, side="left"))
            hi = int(np.searchsorted(snapshots.available[:k], at, side="right"))
            columns = [c for c in SCHEDULE_COLUMNS if c not in ("gw_index", "schedule_source")]
            out = snapshots.frame([*columns, "snapshot_at"], slice(lo, hi))
            out = out[(out["season"] == season) & (out["snapshot_at"] == newest.iloc[0])]
            gameweeks = self.table("gameweek", columns=["season", "gw", "gw_index"])
            gameweeks = gameweeks[gameweeks["season"] == season]
            index = dict(zip(gameweeks["gw"], gameweeks["gw_index"], strict=True))
            gw_index = [index.get(int(gw)) if pd.notna(gw) else None for gw in out["gw"]]
            out = out.assign(gw_index=pd.array(gw_index, dtype="Int64"), schedule_source="snapshot")
        out = out[SCHEDULE_COLUMNS].astype({"gw": "Int64", "gw_index": "Int64"})
        out["schedule_source"] = out["schedule_source"].astype("str")
        return out.sort_values("fixture_key", kind="mergesort").reset_index(drop=True)

    def gameweek_for_deadline(self) -> tuple[int, int]:
        """(season, gw) of the gameweek whose deadline is this view's deadline."""
        gameweeks = self.table("gameweek", columns=["season", "gw", "deadline_time"])
        match = gameweeks[gameweeks["deadline_time"] == self.deadline]
        if len(match) != 1:
            raise LookupError(f"{len(match)} gameweeks have the deadline {self.deadline}")
        return int(match["season"].iloc[0]), int(match["gw"].iloc[0])
