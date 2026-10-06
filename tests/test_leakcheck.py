"""Corrupt-the-future harness (PLAN Â§4): the harness itself, deliberately leaky builders it
must catch, the registered features on the synthetic world (CI) and on the real data
(`-m realdata`)."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fplopt.build.tables import TABLES
from fplopt.features import FEATURES
from fplopt.features.leakcheck import (
    Leak,
    check_leakage,
    corrupt_future,
    feature_fingerprint,
    load_tables,
    run_leakage_check,
    sample_deadlines,
    truncate_future,
)
from fplopt.features.store import AsOfView
from fplopt.seasons import HOLDOUT_SEASONS

UTC_US = pd.DatetimeTZDtype("us", "UTC")
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
T0 = pd.Timestamp("2024-01-01", tz="UTC").as_unit("us")
DEADLINE = T0 + pd.Timedelta(days=50)


def mixed(n=400, seed=1):
    """A table with one column of every dtype the build writes (and a few more)."""
    rng = np.random.default_rng(seed)
    available = T0 + pd.to_timedelta(rng.integers(0, 100 * 24, n), unit="h")
    available = pd.Series(available).astype(UTC_US)
    nulls = rng.random(n) < 0.1
    return pd.DataFrame(
        {
            "player_key": rng.integers(1, 40, n),
            "i64": rng.integers(0, 10, n),
            "f64": rng.random(n),
            "Int64": pd.array(rng.integers(0, 100, n), dtype="Int64").copy(),
            "Float64": pd.array(rng.random(n), dtype="Float64"),
            "boolean": pd.array(rng.random(n) < 0.5, dtype="boolean"),
            "bool": rng.random(n) < 0.5,
            "str": pd.array([f"name{i % 7}" for i in range(n)], dtype="str"),
            "object": pd.Series([("a", "b", "c")[i % 3] for i in range(n)], dtype=object),
            "ts": (available - pd.Timedelta(hours=1)).astype(UTC_US),
            "naive": pd.Series(pd.date_range("2024-01-01", periods=n, freq="h")).astype(
                "datetime64[us]"
            ),
            "date": pd.Series(available.dt.date, dtype="date32[day][pyarrow]"),
            "cat": pd.Categorical([("x", "y")[i % 2] for i in range(n)]),
            "event_time": available,
            "available_at": available,
        }
    ).assign(Int64=lambda df: df["Int64"].mask(nulls))


def is_future(df, deadline=DEADLINE):
    return (df["available_at"] >= deadline).to_numpy()


# --- corrupt_future / truncate_future ----------------------------------------------------


def test_corrupt_future_keeps_past_rows_and_dtypes():
    df = mixed()
    out = corrupt_future({"t": df}, DEADLINE, seed=0)["t"]
    assert list(out.columns) == list(df.columns)
    assert dict(out.dtypes) == dict(df.dtypes)
    assert out.index.equals(pd.RangeIndex(len(out)))
    past = df[~is_future(df)].reset_index(drop=True)
    pd.testing.assert_frame_equal(out[~is_future(out)].reset_index(drop=True), past)
    assert df.equals(mixed())  # input not modified


def test_corrupt_future_scrambles_every_column_of_future_rows():
    df = mixed()
    out = corrupt_future({"t": df}, DEADLINE, seed=0, ghost_fraction=0, interleave=False)["t"]
    assert len(out) == len(df)
    future = is_future(df)
    assert future.sum() > 100
    pd.testing.assert_series_equal(out["available_at"], df["available_at"])
    for column in df.columns.drop("available_at"):
        before, after = df.loc[future, column], out.loc[future, column]
        changed = ~(before.eq(after).fillna(False).astype(bool))
        assert changed.mean() > 0.3, column
        if column not in ("bool", "boolean", "cat"):
            new = ~after.dropna().isin(df[column].dropna().unique())
            assert new.any(), f"{column}: no value outside the original ones"
        if column not in ("player_key", "i64", "bool"):  # numpy ints and bools hold no nulls
            assert after.isna().any(), f"{column}: no nulls"


def test_corrupt_future_appends_ghosts_of_past_rows_after_the_deadline():
    """Scrambled copies of past rows moved after the deadline, often keeping their key and
    newer than every past row of that key: a builder that takes the latest row per key (or
    dedups) before filtering returns a ghost."""
    df = mixed(n=2000)
    out = corrupt_future({"t": df}, DEADLINE, seed=0, ghost_fraction=0.5, max_ghosts=300)["t"]
    assert len(out) == len(df) + 300
    ghosts = out[is_future(out)]
    assert (ghosts["available_at"] >= DEADLINE).all()
    assert (ghosts["available_at"] == DEADLINE).any()  # exactly at the deadline is future too
    past = df[~is_future(df)]
    newest = past.groupby("player_key")["event_time"].max()
    colliding = ghosts[ghosts["player_key"].isin(newest.index)]
    newer = colliding["event_time"] > colliding["player_key"].map(newest)
    assert len(colliding) > len(ghosts) / 2 and newer.sum() > 50
    # Future rows are interleaved with past rows, not appended at the end.
    positions = np.flatnonzero(is_future(out))
    assert positions.min() < np.flatnonzero(~is_future(out)).max()


def test_corrupt_future_is_seeded_and_covers_every_table():
    tables = {"a": mixed(seed=1), "b": mixed(seed=2)}
    one = corrupt_future(tables, DEADLINE, seed=7)
    pd.testing.assert_frame_equal(one["a"], corrupt_future(tables, DEADLINE, seed=7)["a"])
    assert not one["a"].equals(corrupt_future(tables, DEADLINE, seed=8)["a"])
    assert set(one) == {"a", "b"}


def test_corrupt_future_rejects_bad_inputs():
    with pytest.raises(ValueError, match="tz-aware"):
        corrupt_future({"t": mixed()}, DEADLINE.tz_localize(None), seed=0)
    broken = mixed().assign(available_at=lambda df: df["available_at"].dt.tz_localize(None))
    with pytest.raises(ValueError, match="available_at"):
        corrupt_future({"t": broken}, DEADLINE, seed=0)


def test_truncate_future_drops_rows_at_or_after_the_deadline():
    df = mixed()
    out = truncate_future({"t": df}, DEADLINE)["t"]
    pd.testing.assert_frame_equal(out, df[~is_future(df)].reset_index(drop=True))


# --- fingerprints ------------------------------------------------------------------------


def test_feature_fingerprint_is_exact():
    df = mixed()
    base = feature_fingerprint({"f": df})["f"]
    assert base == feature_fingerprint({"f": df.copy()})["f"]
    changed = df.copy()
    changed.loc[3, "f64"] = np.nextafter(changed.loc[3, "f64"], 2)
    variants = [
        changed,
        df.astype({"i64": "Int64"}),  # same values, other dtype
        df[list(reversed(df.columns))],  # column order
        df.rename(columns={"i64": "j64"}),
        df.iloc[:-1],
        df.set_axis(df.index + 1),  # index
        df.assign(Int64=df["Int64"].fillna(0)),  # null vs value
        df.assign(str=df["str"].str.upper()),
        df.assign(ts=df["ts"] + pd.Timedelta(microseconds=1)),
    ]
    for variant in variants:
        assert feature_fingerprint({"f": variant})["f"] != base


# --- check_leakage: leaky builders are caught --------------------------------------------


def backdoor(view: AsOfView, name: str) -> pd.DataFrame:
    """Test-only back door: the store's whole, unfiltered table (what a builder that bypasses
    the view would read, e.g. from a cached DataFrame or the Parquet file)."""
    return view._store._frames[name]


def leaky_latest_price(view: AsOfView) -> pd.DataFrame:
    """Latest price per player, deduplicated *before* filtering to the deadline: when a
    player's newest row is in the future he silently drops out."""
    rows = backdoor(view, "player_gw").sort_values(["season", "gw"], kind="mergesort")
    latest = rows.groupby("player_key").tail(1)
    latest = latest[latest["available_at"] < view.deadline]
    out = latest[["player_key", "value"]].sort_values("player_key", kind="mergesort")
    return out.reset_index(drop=True)


def leaky_recency_weighted_points(view: AsOfView) -> pd.DataFrame:
    """Reads its rows correctly through the view, but takes the reference time for the
    recency weights from `max(event_time)` of the whole table."""
    matches = view.table("player_match", columns=["player_key", "kickoff_time", "total_points"])
    now = backdoor(view, "player_match")["event_time"].max()
    age_days = (now - matches["kickoff_time"]).dt.total_seconds() / 86400
    weighted = matches["total_points"] * np.exp(-age_days / 30)
    out = weighted.groupby(matches["player_key"]).sum().rename("points").reset_index()
    return out.sort_values("player_key", kind="mergesort").reset_index(drop=True)


def leaky_first_seen_team(view: AsOfView) -> pd.DataFrame:
    """Club per player from the first row per player in *storage order*, then filtered:
    right while the table is stored sorted, wrong once a future row is stored first."""
    rows = backdoor(view, "player_gw").drop_duplicates("player_key")
    rows = rows[rows["available_at"] < view.deadline]
    return rows[["player_key", "team_key"]].sort_values("player_key").reset_index(drop=True)


def leaky_data_horizon(view: AsOfView) -> pd.DataFrame:
    """Depends only on how far the data reaches (`max(available_at)`), which corruption
    leaves alone: only the truncated variant catches it."""
    horizon = backdoor(view, "fixture")["available_at"].max()
    return pd.DataFrame({"days_of_data_left": [(horizon - view.deadline).days]})


def leaky_event_time_filter(view: AsOfView) -> pd.DataFrame:
    """Filters on `event_time` (the match happened) instead of `available_at` (its result
    is in)."""
    rows = backdoor(view, "player_match")
    rows = rows[rows["event_time"] < view.deadline]
    out = rows.groupby("player_key")["total_points"].sum().reset_index()
    return out.sort_values("player_key", kind="mergesort").reset_index(drop=True)


def deadline_of(tables, season, gw):
    gameweeks = tables["gameweek"]
    row = gameweeks[(gameweeks["season"] == season) & (gameweeks["gw"] == gw)]
    return row["deadline_time"].iloc[0]


def flagged(leaks):
    return {(leak.feature, leak.variant) for leak in leaks}


def test_leaky_builders_are_caught(tables):
    deadlines = [deadline_of(tables, 2023, 10), deadline_of(tables, 2024, 20)]
    leaky = {
        "latest_price": leaky_latest_price,
        "recency_weighted_points": leaky_recency_weighted_points,
        "first_seen_team": leaky_first_seen_team,
        "data_horizon": leaky_data_horizon,
        "honest": FEATURES["recent_form"],
    }
    leaks = check_leakage(tables, deadlines, seed=0, features=leaky)
    assert all(isinstance(leak, Leak) and leak.detail for leak in leaks)
    caught = flagged(leaks)
    assert {feature for feature, _ in caught} == set(leaky) - {"honest"}
    # Each variant has leaks only it catches:
    assert ("latest_price", "truncated") in caught
    assert ("recency_weighted_points", "corrupted") in caught
    assert ("recency_weighted_points", "truncated") in caught
    # Storage order only changes in the corrupted copy (future rows interleaved).
    assert {v for f, v in caught if f == "first_seen_team"} == {"corrupted"}
    # Corruption keeps available_at: only truncation moves max(available_at).
    assert {v for f, v in caught if f == "data_horizon"} == {"truncated"}
    assert {leak.deadline for leak in leaks} == set(deadlines)


def test_event_time_filter_is_caught_when_results_arrive_after_the_deadline(tables):
    """A GW9 match whose result is only in after the GW10 deadline (event_time before it,
    available_at after): filtering on event_time sees it."""
    deadline = deadline_of(tables, 2023, 10)
    pm = tables["player_match"]
    late = (pm["season"] == 2023) & (pm["gw"] == 9) & (pm["player_key"] == pm["player_key"].min())
    pm.loc[late, "available_at"] = deadline + pd.Timedelta(hours=3)
    features = {"event_time_filter": leaky_event_time_filter, "honest": FEATURES["recent_form"]}
    leaks = check_leakage(tables, [deadline], seed=0, features=features)
    expected = {("event_time_filter", "corrupted"), ("event_time_filter", "truncated")}
    assert flagged(leaks) == expected


def test_a_builder_that_fails_only_on_one_variant_is_a_leak(tables):
    def fragile(view):
        if backdoor(view, "player_gw")["season"].max() > 2026:
            raise RuntimeError("season from the future")
        return pd.DataFrame({"ok": [True]})

    leaks = check_leakage(tables, [deadline_of(tables, 2023, 10)], seed=0, features={"f": fragile})
    assert leaks and all("RuntimeError" in leak.detail for leak in leaks)


# --- check_leakage: the registered features ----------------------------------------------

SYNTHETIC_DEADLINES = [(2023, 1), (2023, 10), (2024, 38), (2026, 1), (2026, 3), (2026, 6)]


def test_registered_features_do_not_leak_on_the_synthetic_world(built):
    """Phase 2 'done when' (CI): every feature is byte-identical on the clean, corrupted
    and truncated data at deadlines covering GW1, mid-season, the last GW, the player_gw
    and snapshot pool paths and the as-of snapshot schedule."""
    deadlines = [deadline_of(built, season, gw) for season, gw in SYNTHETIC_DEADLINES]
    for seed in (0, 1):
        assert check_leakage(built, deadlines, seed=seed) == []


# --- deadline sampling -------------------------------------------------------------------


def test_sample_deadlines_spreads_over_seasons_and_skips_the_holdout(built):
    tables = dict(built)
    gameweeks = tables["gameweek"]
    holdout = gameweeks[gameweeks["season"] == 2023]
    moved = holdout["deadline_time"] + pd.DateOffset(years=2)
    holdout = holdout.assign(season=2025, deadline_time=moved)
    tables["gameweek"] = pd.concat([gameweeks, holdout], ignore_index=True)
    picked = sample_deadlines(tables, 6, seed=0)
    assert picked == sample_deadlines(tables, 6, seed=0)
    assert len(picked) == 6 and picked == sorted(picked)
    by_deadline = tables["gameweek"].set_index("deadline_time")["season"]
    seasons = [by_deadline[d] for d in picked]
    assert not set(seasons) & HOLDOUT_SEASONS
    assert {2023, 2024, 2026} <= set(seasons)
    # Nothing after the next deadline following the newest result.
    newest = tables["player_match"]["available_at"].max()
    upcoming = gameweeks.loc[gameweeks["deadline_time"] > newest, "deadline_time"].min()
    assert max(picked) == upcoming


# --- real data ---------------------------------------------------------------------------

REAL_DEADLINES = [
    (2016, 1),
    (2017, 1),  # Brighton and Huddersfield promoted: Elo from pre-season rows
    (2018, 20),
    (2019, 25),
    (2020, 1),  # COVID-delayed season start, postponed GW1 fixtures
    (2021, 7),  # first season with fplcache snapshots
    (2022, 25),
    (2023, 38),
    (2024, 15),
    (2026, 6),  # live season: own snapshots, as-of fixture list
]


@pytest.mark.realdata
def test_registered_features_do_not_leak_on_the_real_data():
    if not (DATA_DIR / "player_snapshot.parquet").exists():
        pytest.skip("data/ not built (run `uv run fplopt build all`)")
    tables = load_tables(DATA_DIR)
    deadlines = [deadline_of(tables, season, gw) for season, gw in REAL_DEADLINES]
    assert not {s for s, _ in REAL_DEADLINES} & HOLDOUT_SEASONS
    start = time.perf_counter()
    leaks = check_leakage(tables, deadlines, seed=0)
    elapsed = time.perf_counter() - start
    assert leaks == [], "\n".join(map(str, leaks))
    print(f"real-data leakage check: {len(deadlines)} deadlines in {elapsed:.0f} s")


def test_load_tables_reads_every_registered_table(tmp_path, built):
    for name, df in built.items():
        df.to_parquet(tmp_path / f"{name}.parquet")
    loaded = load_tables(tmp_path)
    assert set(loaded) == set(TABLES)
    pd.testing.assert_frame_equal(loaded["gameweek"], built["gameweek"])


def test_leakage_check_with_no_eligible_deadlines_fails(tmp_path, built):
    """Checking nothing must not report a pass."""
    for name, df in built.items():
        df.to_parquet(tmp_path / f"{name}.parquet")
    with pytest.raises(ValueError, match="no eligible deadlines"):
        run_leakage_check(tmp_path, n_deadlines=0)
