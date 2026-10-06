import json
from datetime import UTC, datetime

import pandas as pd
import pandera.pandas as pa
import pytest

import fplopt.build as build_pkg
from fplopt.build import BUILDERS, ORDER, Builder, build
from fplopt.build.common import (
    BuildContext,
    TableValidationError,
    fplcache_and_own_bootstraps,
    latest_complete_run,
    lockdown_time,
    lockdown_times,
    read_raw_csv,
    uk_date,
    write_table,
)
from fplopt.ingest.raw_store import RawStore, gzip_bytes

T0 = datetime(2026, 10, 1, 3, 0, tzinfo=UTC)


def write_raw(tmp_path, rel, content: bytes):
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip_bytes(content))
    return path


# --- read_raw_csv -----------------------------------------------------------------------


def test_read_raw_csv_falls_back_to_latin1(tmp_path):
    path = write_raw(tmp_path, "x.csv.gz", b"name\nJos\xe9\n")
    df = read_raw_csv(path)
    assert df["name"].tolist() == ["José"]


def test_read_raw_csv_strips_bom_and_header_whitespace(tmp_path):
    path = write_raw(tmp_path, "x.csv.gz", "﻿ Understat_ID , FPL_ID\n1,2\n".encode())
    df = read_raw_csv(path)
    assert list(df.columns) == ["Understat_ID", "FPL_ID"]
    assert df.iloc[0].tolist() == [1, 2]


def test_read_raw_csv_usecols_match_stripped_names(tmp_path):
    path = write_raw(tmp_path, "x.csv.gz", b"a , b,c\n1,2,3\n")
    df = read_raw_csv(path, usecols=["a", "c"])
    assert list(df.columns) == ["a", "c"]


def test_read_raw_csv_utf8_is_not_misread_as_latin1(tmp_path):
    path = write_raw(tmp_path, "x.csv.gz", "name\nØdegaard\n".encode())
    assert read_raw_csv(path)["name"].tolist() == ["Ødegaard"]


# --- latest_complete_run ----------------------------------------------------------------


def write_manifest(tmp_path, endpoint, run, **manifest):
    write_raw(
        tmp_path, f"vaastav/{endpoint}/{run}/_manifest.json.gz", json.dumps(manifest).encode()
    )


def test_latest_complete_run_skips_failed_incomplete_and_non_timestamp_dirs(tmp_path):
    ok = {"expected": ["a", "b"], "written": ["b", "a"], "failed": []}
    write_manifest(tmp_path, "data", "2026-10-01T000000Z", **ok)
    write_manifest(tmp_path, "data", "2026-10-02T000000Z", **ok)
    write_manifest(tmp_path, "data", "2026-10-03T000000Z", expected=["a"], written=[], failed=["a"])
    write_manifest(tmp_path, "data", "2026-10-04T000000Z", expected=["a"], written=[], failed=[])
    (tmp_path / "vaastav/data/2026-10-05T000000Z").mkdir()  # no manifest (still running)
    write_manifest(tmp_path, "data", "zz-not-a-run", **ok)
    store = RawStore(tmp_path)
    assert (
        latest_complete_run(store, "vaastav", "data")
        == tmp_path / "vaastav/data/2026-10-02T000000Z"
    )


def test_latest_complete_run_raises_when_none(tmp_path):
    write_manifest(tmp_path, "data", "2026-10-03T000000Z", expected=["a"], written=[], failed=["a"])
    with pytest.raises(LookupError, match="vaastav/data"):
        latest_complete_run(RawStore(tmp_path), "vaastav", "data")
    with pytest.raises(LookupError):
        latest_complete_run(RawStore(tmp_path), "fpl", "element-summary")


# --- write_table ------------------------------------------------------------------------

SCHEMA = pa.DataFrameSchema(
    {
        "team_key": pa.Column(int, pa.Check.ge(1)),
        "name": pa.Column(str),
        "event_time": pa.Column(pd.DatetimeTZDtype("us", "UTC")),
    },
    strict=True,
    unique=["team_key"],
)


def frame(keys=(3, 1, 2)):
    return pd.DataFrame(
        {
            "team_key": list(keys),
            "name": [f"t{k}" for k in keys],
            "event_time": pd.Series(
                pd.to_datetime(["2026-01-01"] * len(keys), utc=True)
            ).dt.as_unit("us"),
        }
    )


def test_write_table_sorts_writes_and_is_byte_identical(tmp_path):
    path = write_table(frame(), "team_dim", SCHEMA, tmp_path / "data", sort_by=["team_key"])
    assert path == tmp_path / "data" / "team_dim.parquet"
    first = path.read_bytes()
    out = pd.read_parquet(path)
    assert out["team_key"].tolist() == [1, 2, 3]
    assert str(out["event_time"].dtype) == "datetime64[us, UTC]"
    assert not list((tmp_path / "data").glob("*.tmp"))
    write_table(frame((2, 3, 1)), "team_dim", SCHEMA, tmp_path / "data", sort_by=["team_key"])
    assert path.read_bytes() == first


def test_write_table_rejects_invalid_frame_naming_the_column(tmp_path):
    with pytest.raises(TableValidationError, match="team_key") as info:
        write_table(frame((1, 1, 0)), "team_dim", SCHEMA, tmp_path, sort_by=["team_key"])
    assert str(info.value).splitlines()[0].startswith("table 'team_dim' failed validation")
    assert not (tmp_path / "team_dim.parquet").exists()


def test_write_table_rejects_extra_column(tmp_path):
    df = frame().assign(extra=1)
    with pytest.raises(TableValidationError, match="extra"):
        write_table(df, "team_dim", SCHEMA, tmp_path, sort_by=["team_key"])


# --- BuildContext -----------------------------------------------------------------------


def test_context_table_reads_and_caches(tmp_path):
    ctx = BuildContext(RawStore(tmp_path / "raw"), tmp_path / "data")
    write_table(frame(), "team_dim", SCHEMA, ctx.data_dir, sort_by=["team_key"])
    first = ctx.table("team_dim")
    assert first["team_key"].tolist() == [1, 2, 3]
    assert ctx.table("team_dim") is first
    ctx.forget("team_dim")
    assert ctx.table("team_dim") is not first


def test_context_missing_table_names_the_builder(tmp_path):
    ctx = BuildContext(RawStore(tmp_path / "raw"), tmp_path / "data")
    with pytest.raises(FileNotFoundError, match="fplopt build fixture"):
        ctx.table("fixture")


# --- time helpers -----------------------------------------------------------------------


def test_lockdown_after_sunday_bst_kickoff_is_monday_0800_utc():
    kickoff = pd.Timestamp("2026-09-27 15:30", tz="UTC")  # Sunday 16:30 BST
    assert lockdown_time(kickoff) == pd.Timestamp("2026-09-28 08:00", tz="UTC")


def test_lockdown_in_gmt_is_0900_utc():
    kickoff = pd.Timestamp("2026-01-17 17:30", tz="UTC")
    assert lockdown_time(kickoff) == pd.Timestamp("2026-01-18 09:00", tz="UTC")


def test_lockdown_uses_uk_date_and_handles_dst_change():
    # 23:30 UTC on 30 Sep is 00:30 BST on 1 Oct -> lockdown 2 Oct 09:00 BST.
    assert lockdown_time(pd.Timestamp("2026-09-30 23:30", tz="UTC")) == pd.Timestamp(
        "2026-10-02 08:00", tz="UTC"
    )
    # Saturday 28 Mar 2026 kickoff (GMT); clocks go forward overnight -> Sunday 09:00 BST.
    assert lockdown_time(pd.Timestamp("2026-03-28 15:00", tz="UTC")) == pd.Timestamp(
        "2026-03-29 08:00", tz="UTC"
    )


def test_lockdown_times_matches_scalar_and_keeps_nulls():
    kickoffs = pd.Series(
        pd.to_datetime(["2026-09-27 15:30", None, "2026-01-17 17:30", "2026-03-28 15:00"], utc=True)
    )
    out = lockdown_times(kickoffs)
    assert str(out.dtype) == "datetime64[us, UTC]"
    assert out.isna().tolist() == [False, True, False, False]
    for k, v in zip(kickoffs, out, strict=True):
        if pd.notna(k):
            assert v == lockdown_time(k)


def test_uk_date():
    assert str(uk_date(pd.Timestamp("2026-09-30 23:30", tz="UTC"))) == "2026-10-01"
    assert str(uk_date(pd.Timestamp("2026-01-31 23:30", tz="UTC"))) == "2026-01-31"


# --- bootstraps -------------------------------------------------------------------------


def test_fplcache_and_own_bootstraps_sorted_with_source(tmp_path):
    store = RawStore(tmp_path)
    store.write_bytes(
        "fplcache",
        "bootstrap-static",
        b"x",
        datetime(2026, 10, 5, 3, tzinfo=UTC),
        suffix=".json.xz",
    )
    store.write_bytes(
        "fplcache", "bootstrap-static", b"x", datetime(2021, 4, 18, tzinfo=UTC), suffix=".json.xz"
    )
    store.write("fpl", "bootstrap-static", b"{}", datetime(2026, 10, 5, 1, tzinfo=UTC))
    out = fplcache_and_own_bootstraps(store)
    assert [(ts.year, ts.hour, source) for ts, _, source in out] == [
        (2021, 0, "fplcache"),
        (2026, 1, "fpl"),
        (2026, 3, "fplcache"),
    ]
    assert all(path.exists() for _, path, _ in out)


# --- registry ---------------------------------------------------------------------------


@pytest.fixture
def registry(monkeypatch):
    """Two fake tables, `b` depending on nothing but ordered after `a`."""
    ran = []

    def fake(name, keys):
        def run(ctx):
            ran.append(name)
            return frame(keys)

        return Builder(run, SCHEMA, ("team_key",))

    monkeypatch.setattr(build_pkg, "BUILDERS", {"b": fake("b", (3, 4)), "a": fake("a", (1, 2))})
    monkeypatch.setattr(build_pkg, "ORDER", ["a", "b"])
    return ran


def test_build_runs_requested_tables_in_dependency_order(tmp_path, registry):
    ctx = BuildContext(RawStore(tmp_path / "raw"), tmp_path / "data")
    paths = build(["b", "a"], ctx)
    assert registry == ["a", "b"]
    assert paths == [tmp_path / "data" / "a.parquet", tmp_path / "data" / "b.parquet"]
    assert ctx.table("b")["team_key"].tolist() == [3, 4]


def test_build_all_runs_everything(tmp_path, registry):
    build(["all"], BuildContext(RawStore(tmp_path), tmp_path / "data"))
    assert registry == ["a", "b"]


def test_build_unknown_name_lists_valid_names(tmp_path, registry):
    ctx = BuildContext(RawStore(tmp_path), tmp_path / "data")
    with pytest.raises(ValueError, match="unknown table 'nope'; valid: all, a, b"):
        build(["a", "nope"], ctx)
    assert registry == []  # nothing runs when a name is unknown


def test_order_matches_registry():
    assert sorted(ORDER) == sorted(BUILDERS)
