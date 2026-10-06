"""Shared build-layer plumbing: the build context, raw CSV reading, run discovery, the
validated Parquet writer and UK time helpers (PLAN §3, Phase 1b plan "Conventions")."""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pandera.pandas as pa

from fplopt.ingest.raw_store import TS_PATTERN, RawStore, parse_ts

log = logging.getLogger(__name__)

UK = ZoneInfo("Europe/London")
UTC_US = pd.DatetimeTZDtype("us", "UTC")
EPOCH = pd.Timestamp("1970-01-01", tz="UTC").as_unit("us")
MAX_FAILURES_SHOWN = 20
MANIFEST = "_manifest.json.gz"


class TableValidationError(ValueError):
    """A built table failed its schema; the first line names the table and columns."""


@dataclass
class BuildContext:
    """What a builder reads from: the raw store, earlier-built tables in `data_dir`, and the
    hand-maintained configs in `config_dir`. Tables read via `table()` are cached."""

    store: RawStore
    data_dir: Path
    config_dir: Path = Path("config")
    _cache: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)

    def table_path(self, name: str) -> Path:
        return Path(self.data_dir) / f"{name}.parquet"

    def table(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            path = self.table_path(name)
            if not path.exists():
                raise FileNotFoundError(
                    f"table {name!r} has not been built ({path}); run `fplopt build {name}` first"
                )
            self._cache[name] = pd.read_parquet(path)
        return self._cache[name]

    def forget(self, name: str) -> None:
        """Drop a cached table (after it has been rebuilt)."""
        self._cache.pop(name, None)


def decode_text(data: bytes) -> str:
    """Strict UTF-8 (BOM stripped), else latin-1 (some old vaastav files are latin-1)."""
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def read_raw_csv(path: Path, usecols: list[str] | None = None, **kwargs: Any) -> pd.DataFrame:
    """A compressed raw CSV as a DataFrame, with whitespace stripped from column names.
    `usecols` names columns after stripping; other kwargs go to `pd.read_csv`."""
    text = decode_text(RawStore.read_bytes(path))
    if usecols is not None:
        wanted = set(usecols)
        kwargs["usecols"] = lambda column: column.strip() in wanted
    df = pd.read_csv(io.StringIO(text), **kwargs)
    df.columns = [str(column).strip() for column in df.columns]
    return df


def latest_complete_run(store: RawStore, source: str, endpoint: str) -> Path:
    """Newest timestamp-named run directory under `<source>/<endpoint>/` whose manifest says
    every expected file was written and none failed. Raises LookupError if there is none."""
    directory = store.root / source / endpoint
    runs = (
        sorted(
            (p for p in directory.iterdir() if p.is_dir() and TS_PATTERN.match(p.name)),
            key=lambda p: parse_ts(p.name),
            reverse=True,
        )
        if directory.is_dir()
        else []
    )
    for run in runs:
        manifest_path = run / MANIFEST
        if not manifest_path.exists():
            continue
        manifest = RawStore.read_json(manifest_path)
        expected, written = manifest.get("expected"), manifest.get("written")
        if (
            manifest.get("failed") == []
            and isinstance(expected, list)
            and isinstance(written, list)
            and sorted(map(str, expected)) == sorted(map(str, written))
        ):
            return run
        log.warning("skipping incomplete run %s", run)
    raise LookupError(f"no complete run under {source}/{endpoint}")


def fplcache_and_own_bootstraps(store: RawStore) -> list[tuple[datetime, Path, str]]:
    """Every archived bootstrap-static snapshot as (snapshot time, path, source), source being
    'fplcache' (mirror, .json.xz) or 'fpl' (our archive, .json.gz); sorted by time."""
    entries = [
        (ts, path, "fplcache")
        for ts, path in store.entries("fplcache", "bootstrap-static", ".json.xz")
    ] + [(ts, path, "fpl") for ts, path in store.entries("fpl", "bootstrap-static", ".json.gz")]
    return sorted(entries, key=lambda entry: (entry[0], entry[2]))


def write_table(
    df: pd.DataFrame,
    name: str,
    schema: pa.DataFrameSchema,
    data_dir: Path,
    sort_by: list[str] | tuple[str, ...],
) -> Path:
    """Validate `df` against `schema` (all failures collected), sort by `sort_by` and write
    `<data_dir>/<name>.parquet` (zstd) atomically. Same input -> byte-identical file."""
    try:
        df = schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        cases = exc.failure_cases
        columns = sorted({str(c) for c in cases["column"].dropna()}) or ["<frame>"]
        raise TableValidationError(
            f"table {name!r} failed validation: {len(cases)} failure case(s) in "
            f"{', '.join(columns)}\n"
            + cases.head(MAX_FAILURES_SHOWN)[
                ["column", "check", "failure_case", "index"]
            ].to_string()
        ) from None
    df = df.sort_values(list(sort_by), kind="mergesort").reset_index(drop=True)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{name}.parquet"
    tmp = data_dir / f"{name}.parquet.tmp"
    try:
        df.to_parquet(tmp, compression="zstd", index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


# --- time --------------------------------------------------------------------------------


def uk_date(ts: pd.Timestamp) -> date:
    """The UK-local calendar date of a tz-aware timestamp."""
    return pd.Timestamp(ts).tz_convert(UK).date()


def lockdown_time(last_kickoff: pd.Timestamp) -> pd.Timestamp:
    """GW lockdown: 09:00 UK on the day after the UK-local date of the GW's last kickoff."""
    day = uk_date(last_kickoff) + timedelta(days=1)
    # Localise the wall-clock time on the new date: it can be across a clock change.
    local = pd.Timestamp(datetime(day.year, day.month, day.day, 9)).tz_localize(UK)
    return local.tz_convert("UTC").as_unit("us")


def lockdown_times(last_kickoffs: pd.Series) -> pd.Series:
    """Vectorised `lockdown_time`; nulls stay null. Returns datetime64[us, UTC]."""
    local_day = last_kickoffs.dt.tz_convert(UK).dt.tz_localize(None).dt.normalize()
    local = local_day + pd.Timedelta(days=1, hours=9)
    return local.dt.tz_localize(UK).dt.tz_convert("UTC").astype(UTC_US)
