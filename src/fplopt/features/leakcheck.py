"""Corrupt-the-future harness (PLAN §4; Phase 2 plan, Task 4).

For a deadline, every feature is computed three times, each from its own `DataStore`, in a
per-deadline random order (`variant_order`, seeded), so a builder that keeps state between
calls cannot rely on always seeing the clean data first:
- **clean**: the tables as they are;
- **corrupted** (`corrupt_future`): every row with `available_at >= deadline` gets random
  values of the same dtype in every column except `available_at` (keys, event/snapshot
  times, strings, nullable types included; values drawn partly from the column's own values,
  so corrupted keys collide with real past keys), plus scrambled copies of random past rows
  moved to or after the deadline (keeping their key and becoming the newest row of it more
  often than not), and the future rows stored interleaved with the past ones;
- **truncated** (`truncate_future`): the future rows deleted.

All three must give byte-identical features (`feature_fingerprint`). What each catches:
- a read that bypasses the view's `available_at < deadline` filter, whatever it computes
  from the future rows (values, keys, max/min, counts): corrupted and/or truncated;
- latest-per-key / dedup / sort before filtering: ghosts are the newest row of a past key
  (corrupted), the true latest row disappears (truncated);
- reliance on storage order of the whole table: corrupted (interleaved);
- reliance on `max(available_at)` or on future rows merely existing (corruption keeps
  `available_at`): truncated;
- filtering on `event_time` instead of `available_at`: whenever a row's event precedes the
  deadline but its availability does not;
- a builder that fails on one variant only: reported with the exception.
A difference can also mean the builder is not deterministic; either way it fails the check.

Past rows (`available_at < deadline`) are never modified and keep their relative order.
`available_at` is never corrupted (the store's visibility depends on it).

The builders run inside `files_blocked()`: a builder that opens `data/` itself
(`DataStore(data_dir=...)`) fails, which is reported as a clean-data failure. What the
harness cannot see by construction — rows marked available too early are "past" and never
corrupted — is covered by the build-time availability rules and checks, and the static
rules of tests/test_features_architecture.py.
"""

from __future__ import annotations

import hashlib
import logging
import string
import time
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from fplopt.build.tables import TABLES
from fplopt.features import FEATURES, FeatureBuilder
from fplopt.features.store import DataStore, files_blocked
from fplopt.seasons import HOLDOUT_SEASONS

log = logging.getLogger(__name__)

AVAILABLE = "available_at"
GHOST_FRACTION = 0.01  # ghosts per table: this share of the past rows ...
MAX_GHOSTS = 2_000  # ... at most
GHOST_HORIZON = pd.Timedelta(days=400)
POOL_SIZE = 4096  # corrupted cells drawn from the column's own values: from this many rows
P_FRESH = 0.45  # corrupted cell: a fresh random value; else one of the column's values
P_NULL = 0.1  # corrupted cell: null (if the dtype can hold one)
P_KEEP_KEY = 0.75  # ghost cell of a key column: keep the past row's value
P_KEEP = 0.4  # ghost cell of another column: keep the past row's value
DEFAULT_DEADLINES = 12
VARIANTS = ("corrupted", "truncated")
RUNS = ("clean", *VARIANTS)


class LeakageError(RuntimeError):
    """The corrupt-the-future check found features that depend on future data."""


@dataclass(frozen=True)
class Leak:
    deadline: pd.Timestamp
    feature: str
    variant: str  # 'corrupted' | 'truncated'
    detail: str

    def __str__(self) -> str:
        return f"{self.feature} @ {self.deadline} ({self.variant}): {self.detail}"


def _utc(deadline: Any) -> pd.Timestamp:
    ts = pd.Timestamp(deadline)
    if ts.tzinfo is None:
        raise ValueError(f"deadline must be tz-aware (UTC), got naive {ts}")
    return ts.tz_convert("UTC")


def _future_mask(df: pd.DataFrame, name: str, deadline: pd.Timestamp) -> np.ndarray:
    available = df[AVAILABLE] if AVAILABLE in df else None
    if available is None or not isinstance(available.dtype, pd.DatetimeTZDtype):
        dtype = None if available is None else available.dtype
        raise ValueError(f"{name}.{AVAILABLE} must be tz-aware datetimes, got {dtype}")
    if available.isna().any():
        raise ValueError(f"{name}.{AVAILABLE} has missing values")
    return (available >= deadline).to_numpy(dtype=bool)


# --- random values -----------------------------------------------------------------------


def _can_hold_null(dtype: Any) -> bool:
    if isinstance(dtype, np.dtype):
        return dtype.kind in "fMmO"
    return True  # extension dtypes (masked, arrow, str, categorical, datetimetz)


def _null(dtype: Any) -> Any:
    if isinstance(dtype, np.dtype):
        nulls = {"f": np.nan, "M": np.datetime64("NaT", "ns"), "m": np.timedelta64("NaT", "ns")}
        return nulls.get(dtype.kind)
    return getattr(dtype, "na_value", None)


def _random_strings(s: pd.Series, k: int, rng: np.random.Generator) -> Any:
    """k strings drawn from 64 random ones, as an array of `s`'s dtype."""
    alphabet = np.array(list(string.ascii_letters + string.digits + " -_'"))
    pool = ["".join(rng.choice(alphabet, rng.integers(1, 13))) for _ in range(64)]
    values = np.array(pool, dtype=object) if isinstance(s.dtype, np.dtype) else pool
    return pd.array(values, dtype=s.dtype).take(rng.integers(0, len(pool), k))


def _span(lo: float, hi: float, minimum: float) -> tuple[float, float]:
    span = max(hi - lo, minimum)
    return lo - span, hi + span


def _bounds(s: pd.Series, default: tuple[Any, Any]) -> tuple[Any, Any]:
    """(min, max) of the non-null values of `s` (no copy), or `default` if there are none."""
    lo, hi = s.min(), s.max()
    return default if pd.isna(lo) or pd.isna(hi) else (lo, hi)


def _instant_ns(value: Any) -> int:
    ts = pd.Timestamp(value)
    return (ts if ts.tzinfo is not None else ts.tz_localize("UTC")).as_unit("ns").value


def _random_ns(s: pd.Series, k: int, rng: np.random.Generator) -> np.ndarray:
    """k random instants (int64 ns since the epoch, UTC) around the range of `s`."""
    lo, hi = _bounds(s, (pd.Timestamp("2000-01-01"), pd.Timestamp("2030-01-01")))
    low, high = _span(_instant_ns(lo), _instant_ns(hi), pd.Timedelta(days=365).value)
    return rng.integers(int(low), int(high), k, dtype=np.int64)


def _fresh(s: pd.Series, k: int, rng: np.random.Generator) -> Any:
    """k random values of `s`'s dtype (an array of that dtype) around its value range, or
    None for a dtype it cannot generate (then only the column's own values are used)."""
    dtype = s.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        if not len(dtype.categories):
            return None
        return pd.Categorical.from_codes(rng.integers(0, len(dtype.categories), k), dtype=dtype)
    if isinstance(dtype, pd.ArrowDtype):
        arrow = dtype.pyarrow_dtype
        if pa.types.is_date(arrow) or pa.types.is_timestamp(arrow):
            ns = pa.array(_random_ns(s, k, rng).astype("datetime64[ns]"))
            if pa.types.is_timestamp(arrow) and arrow.tz is not None:
                ns = ns.cast(pa.timestamp("ns", tz="UTC"))
            return pd.arrays.ArrowExtensionArray(ns.cast(arrow, safe=False))
    if pd.api.types.is_bool_dtype(dtype):
        return pd.array(rng.random(k) < 0.5, dtype=dtype)
    if pd.api.types.is_datetime64_any_dtype(dtype):
        stamps = pd.to_datetime(_random_ns(s, k, rng), unit="ns", utc=True)
        tz = getattr(dtype, "tz", None)
        stamps = stamps.tz_localize(None) if tz is None else stamps.tz_convert(tz)
        return pd.array(stamps).astype(dtype)
    if pd.api.types.is_timedelta64_dtype(dtype):
        lo, hi = _bounds(s, (pd.Timedelta(0), pd.Timedelta(days=1)))
        low, high = _span(lo.value, hi.value, pd.Timedelta(days=1).value)
        values = pd.to_timedelta(rng.integers(int(low), int(high), k), unit="ns")
        return pd.array(values).astype(dtype)
    if pd.api.types.is_integer_dtype(dtype):
        info = np.iinfo(getattr(dtype, "numpy_dtype", dtype))
        lo, hi = _bounds(s, (0, 100))
        low, high = _span(int(lo), int(hi), 10)
        low, high = max(int(low), int(info.min)), min(int(high), int(info.max) - 1)
        return pd.array(rng.integers(low, high + 1, k), dtype=dtype)
    if pd.api.types.is_float_dtype(dtype):
        lo, hi = _bounds(s, (0.0, 1.0))
        low, high = _span(float(lo), float(hi), 1.0)
        return pd.array(rng.uniform(low, high, k), dtype=dtype)
    if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
        return _random_strings(s, k, rng)
    return None


def _random_values(s: pd.Series, k: int, rng: np.random.Generator, past_rows: np.ndarray) -> Any:
    """k corrupted cells for column `s`: its own values (from a sample of POOL_SIZE rows,
    half of them past rows, so corrupted keys collide with real past keys), fresh random
    values of its dtype, or nulls. An array of `s`'s dtype."""
    if len(s):
        rows = rng.integers(0, len(s), POOL_SIZE)
        if len(past_rows):
            rows[: POOL_SIZE // 2] = rng.choice(past_rows, POOL_SIZE // 2)
        out = s.array.take(rows).take(rng.integers(0, POOL_SIZE, k))
        use = rng.random(k) < P_FRESH
        fresh = _fresh(s, int(use.sum()), rng)
        if fresh is not None:
            out[use] = fresh
    else:
        out = _fresh(s, k, rng)
        if out is None:
            return s.array.take(np.full(k, -1, dtype=np.intp), allow_fill=True)
    if _can_hold_null(s.dtype):
        out[rng.random(k) < P_NULL] = _null(s.dtype)
    return out


# --- variants ----------------------------------------------------------------------------


def _ghost_available(deadline: pd.Timestamp, k: int, rng: np.random.Generator) -> pd.Series:
    """Availability of ghosts: some exactly at the deadline, some within the hour, the rest
    up to GHOST_HORIZON later."""
    horizon_us = GHOST_HORIZON.value // 1000
    offsets = rng.integers(0, horizon_us, k)
    close = rng.random(k)
    offsets = np.where(close < 0.3, rng.integers(0, 3_600_000_000, k), offsets)
    offsets = np.where(close < 0.1, 0, offsets)
    return pd.Series(deadline.as_unit("us") + pd.to_timedelta(offsets, unit="us"))


def _corrupt_table(
    name: str,
    df: pd.DataFrame,
    deadline: pd.Timestamp,
    rng: np.random.Generator,
    ghost_fraction: float,
    max_ghosts: int,
    interleave: bool,
) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    future = _future_mask(df, name, deadline)
    future_rows = np.flatnonzero(future)
    past_rows = np.flatnonzero(~future)
    n_ghosts = (
        min(max_ghosts, max(1, int(np.ceil(ghost_fraction * len(past_rows)))))
        if len(past_rows) and ghost_fraction > 0
        else 0
    )
    sources = rng.choice(past_rows, n_ghosts) if n_ghosts else np.array([], dtype=np.intp)
    ghost_available = _ghost_available(deadline, n_ghosts, rng).astype(df[AVAILABLE].dtype)
    spec = TABLES.get(name)
    keys = set(spec.key) if spec else set()
    columns: dict[str, Any] = {}
    for column in df.columns:
        s = df[column]
        corrupted = s.array.copy()
        if column == AVAILABLE:
            ghosts = ghost_available.array
        else:
            if len(future_rows):
                corrupted[future_rows] = _random_values(s, len(future_rows), rng, past_rows)
            ghosts = s.array.take(sources)
            if n_ghosts:
                keep = P_KEEP_KEY if column in keys else P_KEEP
                scramble = rng.random(n_ghosts) >= keep
                ghosts[scramble] = _random_values(s, n_ghosts, rng, past_rows)[scramble]
                if pd.api.types.is_datetime64_any_dtype(s.dtype) and not isinstance(
                    s.dtype, pd.ArrowDtype
                ):
                    # Event/snapshot times: often the ghost's new availability, so the ghost
                    # is the newest row of its key.
                    moved = rng.random(n_ghosts) < 0.4
                    times = ghost_available
                    if getattr(s.dtype, "tz", None) is None:
                        times = times.dt.tz_localize(None)
                    ghosts[moved] = times.astype(s.dtype).array[moved]
        columns[column] = type(corrupted)._concat_same_type([corrupted, ghosts])
    out = pd.DataFrame(columns, columns=df.columns)
    out = out.astype(df.dtypes.to_dict()) if list(out.dtypes) != list(df.dtypes) else out
    if interleave and len(out):
        # Random storage order in which the past rows keep their relative order.
        is_past = np.concatenate([~future, np.zeros(n_ghosts, dtype=bool)])
        position = rng.random(len(out))
        position[is_past] = np.sort(position[is_past])
        out = out.iloc[np.argsort(position, kind="stable")].reset_index(drop=True)
    if list(out.dtypes) != list(df.dtypes):
        raise AssertionError(f"corrupting {name} changed its dtypes")  # harness bug
    return out


def corrupt_future(
    tables: Mapping[str, pd.DataFrame],
    deadline: Any,
    seed: int,
    *,
    ghost_fraction: float = GHOST_FRACTION,
    max_ghosts: int = MAX_GHOSTS,
    interleave: bool = True,
) -> dict[str, pd.DataFrame]:
    """A corrupted copy of `tables` for `deadline` (see the module docstring). The inputs
    are not modified; the same seed gives the same copy."""
    deadline = _utc(deadline)
    out = {}
    for name, df in tables.items():
        rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
        out[name] = _corrupt_table(name, df, deadline, rng, ghost_fraction, max_ghosts, interleave)
    return out


def truncate_future(tables: Mapping[str, pd.DataFrame], deadline: Any) -> dict[str, pd.DataFrame]:
    """`tables` without their rows with available_at >= deadline."""
    deadline = _utc(deadline)
    return {
        name: df[~_future_mask(df, name, deadline)].reset_index(drop=True)
        for name, df in tables.items()
    }


# --- fingerprints ------------------------------------------------------------------------


def _values_bytes(s: pd.Series) -> bytes:
    """Canonical bytes of the non-null values of `s` (exact: no rounding)."""
    dtype = s.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        s = s.astype(object)
        dtype = s.dtype
    array = s.array
    if hasattr(array, "asi8"):  # datetime / timedelta (any unit, tz in the dtype string)
        return np.asarray(array.asi8).tobytes()
    if pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_numeric_dtype(dtype):
        if not isinstance(dtype, pd.ArrowDtype):
            numpy_dtype = getattr(dtype, "numpy_dtype", dtype)
            return np.ascontiguousarray(s.to_numpy(dtype=numpy_dtype)).tobytes()
    return "\x1f".join(f"{type(v).__name__}:{v!r}" for v in s.tolist()).encode()


def _frame_digest(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(f"rows={len(df)};cols={len(df.columns)}\n".encode())
    if not df.index.equals(pd.RangeIndex(len(df))) or type(df.index) is not pd.RangeIndex:
        index = df.index.to_frame(index=False)
        h.update(b"index:" + _frame_digest(index).encode())
    for name, s in df.items():
        null = s.isna().to_numpy(dtype=bool)
        h.update(f"\ncol={name!r};dtype={s.dtype}\n".encode())
        h.update(np.packbits(null).tobytes())
        h.update(_values_bytes(s[~null]))
    return h.hexdigest()


def feature_fingerprint(features: Mapping[str, pd.DataFrame]) -> dict[str, str]:
    """sha256 per feature over its exact contents: column names and order, dtypes, index,
    null positions and values."""
    return {name: _frame_digest(df) for name, df in features.items()}


def _describe(clean: Any, other: Any) -> str:
    """One line on how two results differ."""
    if isinstance(clean, str) or isinstance(other, str):
        return f"clean: {_short(clean)} | variant: {_short(other)}"
    if list(clean.columns) != list(other.columns):
        return f"columns differ: {list(clean.columns)} vs {list(other.columns)}"
    if clean.shape != other.shape:
        return f"shape {clean.shape} vs {other.shape}"
    dtypes = [c for c in clean.columns if clean[c].dtype != other[c].dtype]
    if dtypes:
        return f"dtypes differ in {dtypes}"
    differing = []
    for column in clean.columns:
        a, b = clean[column], other[column].set_axis(clean.index)
        try:
            same = (a == b).fillna(False).astype(bool) | (a.isna() & b.isna())
        except (TypeError, ValueError):
            differing.append(column)
            continue
        if not same.all():
            differing.append(f"{column} ({int((~same).sum())} rows)")
    return "values differ: " + (", ".join(differing) or "index or exact bytes")


def _short(result: Any) -> str:
    return result if isinstance(result, str) else f"{len(result)} rows"


# --- the check ---------------------------------------------------------------------------


def _compute(
    store: DataStore, deadline: pd.Timestamp, features: Mapping[str, FeatureBuilder]
) -> dict[str, Any]:
    """Per feature its frame, or 'error: <type>: <message>' if the builder raised."""
    view = store.as_of(deadline)
    out: dict[str, Any] = {}
    for name, builder in features.items():
        try:
            out[name] = builder(view)
        except Exception as exc:  # a builder failing on one variant only is a finding
            first = str(exc).splitlines()[0] if str(exc) else ""
            out[name] = f"error: {type(exc).__name__}: {first}"
    return out


def _fingerprints(results: Mapping[str, Any]) -> dict[str, str]:
    return {
        name: result if isinstance(result, str) else _frame_digest(result)
        for name, result in results.items()
    }


def variant_order(seed: int, i: int) -> tuple[str, ...]:
    """The order in which the clean, corrupted and truncated features of the i-th deadline
    are computed: random, deterministic for (seed, i)."""
    rng = np.random.default_rng([seed, i, 0x0DE7])
    return tuple(RUNS[j] for j in rng.permutation(len(RUNS)))


def check_leakage(
    tables: Mapping[str, pd.DataFrame],
    deadlines: Sequence[Any],
    seed: int = 0,
    features: Mapping[str, FeatureBuilder] | None = None,
) -> list[Leak]:
    """For each deadline compute `features` (default: the registry) on the clean,
    corrupted and truncated tables (each variant from a fresh DataStore; the clean one is
    shared across deadlines), in `variant_order`, with `files_blocked()` active; every
    difference is a Leak. A feature that fails on the clean data is reported too (variant
    'clean')."""
    features = FEATURES if features is None else features
    clean_store = DataStore(tables=tables)
    leaks: list[Leak] = []
    with files_blocked("while the leakage check runs (features read only the view)"):
        for i, raw in enumerate(deadlines):
            deadline = _utc(raw)
            started = time.perf_counter()
            results: dict[str, dict[str, Any]] = {}
            for run in variant_order(seed, i):
                if run == "clean":
                    results[run] = _compute(clean_store, deadline, features)
                    continue
                if run == "corrupted":
                    altered = corrupt_future(tables, deadline, seed=seed + i)
                else:
                    altered = truncate_future(tables, deadline)
                results[run] = _compute(DataStore(tables=altered), deadline, features)
                del altered
            clean = results["clean"]
            expected = _fingerprints(clean)
            for name, result in clean.items():
                if isinstance(result, str):
                    leaks.append(Leak(deadline, name, "clean", result))
            for variant in VARIANTS:
                for name, digest in _fingerprints(results[variant]).items():
                    if digest != expected[name]:
                        detail = _describe(clean[name], results[variant][name])
                        leaks.append(Leak(deadline, name, variant, detail))
            log.info(
                "leakage check %d/%d at %s: %s (%.1f s)",
                i + 1,
                len(deadlines),
                deadline,
                "ok" if not any(leak.deadline == deadline for leak in leaks) else "LEAKS",
                time.perf_counter() - started,
            )
    return leaks


# --- real data ---------------------------------------------------------------------------


def load_tables(
    data_dir: Path | str, names: Sequence[str] | None = None
) -> dict[str, pd.DataFrame]:
    """Every registered table (or `names`) from `data_dir`, all columns."""
    data_dir = Path(data_dir)
    out = {}
    for name in TABLES if names is None else names:
        path = data_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"table {name!r} has not been built ({path})")
        out[name] = pd.read_parquet(path)
    return out


def sample_deadlines(
    tables: Mapping[str, pd.DataFrame], n: int, seed: int = 0
) -> list[pd.Timestamp]:
    """n GW deadlines spread over the data, deterministic for a seed: the candidates are
    the deadlines outside HOLDOUT_SEASONS up to the first one after the newest result
    (`player_match`), split into n consecutive chunks with one random deadline per chunk;
    the last chunk always contributes the newest candidate (the live decision)."""
    gameweeks = tables["gameweek"]
    newest = tables["player_match"][AVAILABLE].max()
    deadlines = gameweeks.loc[~gameweeks["season"].isin(HOLDOUT_SEASONS), "deadline_time"]
    upcoming = deadlines[deadlines > newest]
    if len(upcoming):
        deadlines = deadlines[deadlines <= upcoming.min()]
    candidates = sorted(pd.Timestamp(d) for d in deadlines.unique())
    if n <= 0 or not candidates:
        return []
    rng = np.random.default_rng(seed)
    chunks = np.array_split(np.arange(len(candidates)), min(n, len(candidates)))
    picked = [candidates[int(rng.choice(chunk))] for chunk in chunks[:-1]]
    return [*picked, candidates[-1]]


def run_leakage_check(
    data_dir: Path | str, n_deadlines: int = DEFAULT_DEADLINES, seed: int = 0
) -> list[pd.Timestamp]:
    """`fplopt check leakage`: check the registered features on `data_dir` at n sampled
    deadlines; raises LeakageError listing the leaks. Returns the deadlines checked."""
    started = time.perf_counter()
    tables = load_tables(data_dir)
    deadlines = sample_deadlines(tables, n_deadlines, seed)
    if not deadlines:
        # Checking nothing must not read as a pass.
        raise ValueError("no eligible deadlines: the leakage check was not performed")
    log.info(
        "leakage check: %d deadline(s) from %s to %s, %d feature(s), seed %d",
        len(deadlines),
        deadlines[0],
        deadlines[-1],
        len(FEATURES),
        seed,
    )
    leaks = check_leakage(tables, deadlines, seed=seed)
    elapsed = time.perf_counter() - started
    if leaks:
        for leak in leaks:
            log.error("leak: %s", leak)
        raise LeakageError(
            f"{len(leaks)} leak(s) in {len({leak.feature for leak in leaks})} feature(s) at "
            f"{len({leak.deadline for leak in leaks})} deadline(s); first: {leaks[0]}"
        )
    log.info(
        "leakage check passed: %d deadline(s) x %d feature(s) x %d variants in %.0f s",
        len(deadlines),
        len(FEATURES),
        len(VARIANTS),
        elapsed,
    )
    return deadlines
