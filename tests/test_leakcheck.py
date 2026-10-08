"""Corrupt-the-future harness (PLAN §4): the harness itself, deliberately leaky builders it
must catch, the registered features and xP models on the synthetic worlds (CI) and on the
real data (`-m realdata`)."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from synthetic_season import synthetic_tables

from fplopt.backtest.policies import RollPolicy
from fplopt.backtest.probes import PROBES, run_probe
from fplopt.backtest.rules import Rules
from fplopt.backtest.start_states import template_state
from fplopt.backtest.state import Holding, SquadState
from fplopt.build.tables import TABLES
from fplopt.features import FEATURES
from fplopt.features.baseline import player_pool
from fplopt.features.leakcheck import (
    Leak,
    check_leakage,
    checked_builders,
    corrupt_future,
    describe_builders,
    edge_deadlines,
    feature_fingerprint,
    load_tables,
    run_leakage_check,
    sample_deadlines,
    truncate_future,
    variant_order,
)
from fplopt.features.store import AsOfView, DataStore, FilesBlockedError, files_blocked
from fplopt.models import MODELS
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


def test_builders_cannot_open_the_data_directory_during_the_check(tables, tmp_path):
    for name in ("gameweek", "player_match"):
        tables[name].to_parquet(tmp_path / f"{name}.parquet")
    opened = DataStore(data_dir=tmp_path)  # outside the check: allowed
    opened.as_of(deadline_of(tables, 2023, 10)).table("gameweek")

    def own_store(view):
        """Reads the real tables itself instead of through the view."""
        store = DataStore(data_dir=tmp_path).as_of(view.deadline)
        return store.table("player_match", columns=["player_key"]).head(1)

    def kept_store(view):
        """Reuses a data_dir store opened before the check (a not-yet-loaded table)."""
        return opened.as_of(view.deadline).table("player_match", columns=["player_key"]).head(1)

    deadline = deadline_of(tables, 2023, 10)
    features = {"own_store": own_store, "kept_store": kept_store}
    leaks = check_leakage(tables, [deadline], seed=0, features=features)
    assert flagged(leaks) == {("own_store", "clean"), ("kept_store", "clean")}
    assert all("FilesBlockedError" in leak.detail for leak in leaks)
    DataStore(data_dir=tmp_path).as_of(deadline).table("player_match")  # allowed again


def test_files_blocked_is_scoped_and_leaves_in_memory_stores_alone(tables, tmp_path):
    with files_blocked("in this test"):
        with pytest.raises(FilesBlockedError, match="in this test"):
            DataStore(data_dir=tmp_path)
        DataStore(tables={"gameweek": tables["gameweek"]}).as_of(DEADLINE).table("gameweek")
    DataStore(data_dir=tmp_path)


def test_variants_run_in_a_seeded_random_order_per_deadline(tables, monkeypatch):
    orders = [variant_order(0, i) for i in range(40)]
    assert all(sorted(order) == ["clean", "corrupted", "truncated"] for order in orders)
    assert {order[0] for order in orders} == {"clean", "corrupted", "truncated"}
    assert orders == [variant_order(0, i) for i in range(40)]
    assert orders != [variant_order(1, i) for i in range(40)]

    # check_leakage follows it: tell the variants apart by their (test-only) raw tables.
    seen = []
    n_rows = len(tables["player_match"])

    def spy(view):
        rows = len(backdoor(view, "player_match"))
        seen.append("clean" if rows == n_rows else "truncated" if rows < n_rows else "corrupted")
        return pd.DataFrame({"ok": [True]})

    deadlines = [deadline_of(tables, 2023, gw) for gw in (5, 10, 15, 20, 25, 30)]
    assert check_leakage(tables, deadlines, seed=3, features={"spy": spy}) == []
    expected = [run for i in range(len(deadlines)) for run in variant_order(3, i)]
    assert seen == expected
    assert {seen[3 * i] for i in range(len(deadlines))} != {"clean"}


# --- check_leakage: the registered features ----------------------------------------------

SYNTHETIC_DEADLINES = [(2023, 1), (2023, 10), (2024, 38), (2026, 1), (2026, 3), (2026, 6)]


def test_registered_features_do_not_leak_on_the_synthetic_world(built):
    """Phase 2 'done when' (CI): every feature (and xP model) is byte-identical on the
    clean, corrupted and truncated data at deadlines covering GW1, mid-season, the last GW,
    the player_gw and snapshot pool paths and the as-of snapshot schedule."""
    deadlines = [deadline_of(built, season, gw) for season, gw in SYNTHETIC_DEADLINES]
    for seed in (0, 1):
        assert check_leakage(built, deadlines, seed=seed) == []


def test_checked_builders_are_the_features_models_and_probes():
    builders = checked_builders()
    assert list(builders) == [
        *FEATURES,
        *(f"model:{name}" for name in MODELS),
        *(f"probe:{name}" for name in PROBES),
    ]
    assert builders["model:rolling"] is MODELS["rolling"]
    assert builders["probe:greedy_rolling_random0"] is PROBES["greedy_rolling_random0"]
    assert describe_builders(builders) == (
        f"{len(FEATURES)} feature(s), {len(MODELS)} model(s), {len(PROBES)} probe(s)"
    )
    assert describe_builders({"a": 1, "probe:x": 2, "probe:y": 3}) == "1 feature(s), 2 probe(s)"


@pytest.mark.parametrize(
    ("kwargs", "gameweeks"),
    [
        # Season boundary (rolling reads the previous season), a blank target (ep_next falls
        # back to form) next to a double, mid-season and the last GW.
        (
            {"seasons": (2022, 2023), "blank": (2023, 10, 1), "double": (2023, 11, 4)},
            [(2022, 38), (2023, 1), (2023, 2), (2023, 10), (2023, 11), (2023, 38)],
        ),
        ({"snapshots": False}, [(2023, 1), (2023, 20)]),  # pool from player_gw, ep_next 0
    ],
)
def test_models_do_not_leak_on_the_synthetic_league(kwargs, gameweeks):
    tables = synthetic_tables(**kwargs)
    deadlines = [deadline_of(tables, season, gw) for season, gw in gameweeks]
    builders = {name: b for name, b in checked_builders().items() if name.startswith("model:")}
    assert check_leakage(tables, deadlines, seed=0, features=builders) == []
    assert check_leakage(tables, deadlines[:2], seed=1) == []  # default: features + models


def leaky_rolling(view: AsOfView) -> pd.DataFrame:
    """xp_rolling's frame, but the rate from the player's last 5 matches in the whole
    table (the future included)."""
    honest = MODELS["rolling"](view)
    rows = backdoor(view, "player_match").sort_values(["player_key", "kickoff_time"])
    rate = rows.groupby("player_key")["total_points"].apply(lambda s: s.tail(5).mean())
    return honest.assign(xp=honest["player_key"].map(rate).fillna(0.0).astype("float64"))


def test_the_default_check_catches_a_leaky_model(monkeypatch):
    tables = synthetic_tables()
    monkeypatch.setitem(MODELS, "leaky", leaky_rolling)
    deadlines = [deadline_of(tables, 2023, 10)]
    leaks = check_leakage(tables, deadlines, seed=0)
    assert {feature for feature, _ in flagged(leaks)} == {"model:leaky"}
    assert ("model:leaky", "truncated") in flagged(leaks)


# --- decision probes (fplopt.backtest.probes) ---------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "gameweeks"),
    [
        # Snapshots: GW1 (template from the pre-season snapshot), mid-season, a blank target
        # next to a double, the last GW; the previous season feeds the rolling model.
        (
            {"seasons": (2022, 2023), "blank": (2023, 10, 1), "double": (2023, 11, 4)},
            [(2023, 1), (2023, 2), (2023, 10), (2023, 11), (2023, 38)],
        ),
        # No snapshots: the template is refused at GW1 (no ownership), from GW2 it uses
        # player_gw_ownership; ep_next is 0 everywhere.
        ({"snapshots": False}, [(2023, 1), (2023, 2), (2023, 20)]),
    ],
)
def test_probes_do_not_leak_on_the_synthetic_league(kwargs, gameweeks):
    tables = synthetic_tables(**kwargs)
    deadlines = [deadline_of(tables, season, gw) for season, gw in gameweeks]
    probes = {name: b for name, b in checked_builders().items() if name.startswith("probe:")}
    assert len(probes) == len(PROBES) == 5
    assert check_leakage(tables, deadlines, seed=0, features=probes) == []
    view = DataStore(tables=tables).as_of(deadlines[0])
    frames = {name: probe(view) for name, probe in probes.items()}
    if kwargs.get("snapshots", True):
        assert all(len(frame) >= 35 for frame in frames.values())
        assert (frames["probe:greedy_rolling_random0"]["kind"] == "held").sum() == 15
    else:
        refused = frames["probe:roll_rolling_template"]
        assert refused["kind"].tolist() == ["refused"]
        assert "no ownership" in refused["note"].iloc[0]
        assert len(frames["probe:greedy_rolling_random0"]) >= 35
        # Without ep_next the optimizer's rolling probe still decides (not degenerate).
        rolling = frames["probe:optimizer_rolling_random0"]
        assert len(rolling) >= 35 and (rolling["kind"] == "starter").sum() == 11


def test_probes_are_refused_on_a_coverage_gap_and_still_compare():
    tables = synthetic_tables(snapshots=False)
    rows = tables["player_gw"]
    club1 = (rows["season"] == 2023) & (rows["gw"] == 1) & (rows["team_key"] == 1)
    tables["player_gw"] = rows[~club1].reset_index(drop=True)
    deadlines = [deadline_of(tables, 2023, 1)]
    view = DataStore(tables=tables).as_of(deadlines[0])
    frame = PROBES["greedy_rolling_random0"](view)
    assert frame["kind"].tolist() == ["refused"] and "coverage gap" in frame["note"].iloc[0]
    probes = {f"probe:{name}": probe for name, probe in PROBES.items()}
    assert check_leakage(tables, deadlines, seed=0, features=probes) == []


def test_probes_refuse_the_holdout_season():
    tables = synthetic_tables(seasons=(2025,))
    view = DataStore(tables=tables).as_of(deadline_of(tables, 2025, 5))
    for probe in PROBES.values():
        with pytest.raises(ValueError, match="holdout"):
            probe(view)


def leaky_start(view: AsOfView, rules: Rules) -> SquadState:
    """The template squad, but its first player swapped for the same-position player who
    scores the most points in the target GW itself (an outcome after the deadline)."""
    state = template_state(view, rules)
    first = state.holdings[0]
    season, gw = view.gameweek_for_deadline()
    matches = backdoor(view, "player_match")
    matches = matches[(matches["season"] == season) & (matches["gw"] == gw)]
    points = matches.groupby("player_key")["total_points"].sum()
    pool = player_pool(view)
    candidates = pool[
        (pool["element_type"] == first.element_type) & ~pool["player_key"].isin(state.player_keys)
    ]
    candidates = candidates.assign(points=candidates["player_key"].map(points).fillna(0))
    best = candidates.sort_values(["points", "player_key"], ascending=[False, True]).iloc[0]
    swapped = Holding(
        int(best["player_key"]), first.element_type, int(best["team_key"]), first.price, first.price
    )
    return SquadState(state.season, state.gw_index, (swapped, *state.holdings[1:]), state.bank, 1)


def leaky_probe(view: AsOfView) -> pd.DataFrame:
    return run_probe(view, RollPolicy("rolling"), leaky_start)


def test_the_default_check_catches_a_leaky_probe(monkeypatch):
    tables = synthetic_tables()
    monkeypatch.setitem(PROBES, "leaky", leaky_probe)
    deadlines = [deadline_of(tables, 2023, 10)]
    leaks = check_leakage(tables, deadlines, seed=0)
    assert {feature for feature, _ in flagged(leaks)} == {"probe:leaky"}


def test_run_leakage_check_logs_what_it_checks(tmp_path, built, caplog):
    for name, df in built.items():
        df.to_parquet(tmp_path / f"{name}.parquet")
    with caplog.at_level("INFO", logger="fplopt.features.leakcheck"):
        deadlines = run_leakage_check(tmp_path, n_deadlines=0)
    builders = checked_builders()
    counts = describe_builders(builders)
    assert "probe(s)" in counts
    assert f"{len(deadlines)} deadline(s)" in caplog.text and counts in caplog.text
    assert f"x {len(builders)} builder(s) ({counts})" in caplog.text


# --- deadline sampling -------------------------------------------------------------------


def test_edge_deadlines_are_always_checked(built):
    """Every season's GW1, the fixed edge GWs present in the data, the first deadline after
    snapshot coverage starts and the newest (live) deadline."""
    tables = dict(built)
    gameweeks = tables["gameweek"]
    # Pretend 2023 is 2021/22 (GW18 is an edge) and give 2024 a snapshot-coverage start
    # just after its GW5 deadline.
    tables["gameweek"] = gameweeks.assign(season=gameweeks["season"].replace({2023: 2021}))
    snaps = tables["player_snapshot"]
    start = deadline_of(built, 2024, 5) + pd.Timedelta(minutes=1)
    tables["player_snapshot"] = snaps.assign(snapshot_at=start.as_unit("us"))
    by = tables["gameweek"].set_index(["season", "gw"])["deadline_time"]
    edges = edge_deadlines(tables)
    newest = tables["player_match"]["available_at"].max()
    live = gameweeks.loc[gameweeks["deadline_time"] > newest, "deadline_time"].min()
    expected = [by[(2021, 1)], by[(2021, 18)], by[(2024, 1)], by[(2024, 6)], by[(2026, 1)], live]
    assert edges == sorted(expected)
    for n, seed in ((0, 0), (5, 0), (5, 1)):
        picked = sample_deadlines(tables, n, seed=seed)
        assert set(edges) <= set(picked) and len(picked) == len(edges) + n


def test_sample_deadlines_spreads_over_seasons_and_skips_the_holdout(built):
    tables = dict(built)
    gameweeks = tables["gameweek"]
    holdout = gameweeks[gameweeks["season"] == 2023]
    moved = holdout["deadline_time"] + pd.DateOffset(years=2)
    holdout = holdout.assign(season=2025, deadline_time=moved)
    tables["gameweek"] = pd.concat([gameweeks, holdout], ignore_index=True)
    picked = sample_deadlines(tables, 6, seed=0)
    assert picked == sample_deadlines(tables, 6, seed=0)
    assert picked != sample_deadlines(tables, 6, seed=1)
    edges = edge_deadlines(tables)
    assert len(picked) == len(edges) + 6 and picked == sorted(picked)
    by_deadline = tables["gameweek"].set_index("deadline_time")["season"]
    seasons = [by_deadline[d] for d in picked]
    assert not set(seasons) & HOLDOUT_SEASONS
    assert {2023, 2024, 2026} <= set(seasons)
    # Nothing after the next deadline following the newest result.
    newest = tables["player_match"]["available_at"].max()
    upcoming = gameweeks.loc[gameweeks["deadline_time"] > newest, "deadline_time"].min()
    assert max(picked) == upcoming


# --- real data ---------------------------------------------------------------------------

REAL_DEADLINES = [  # mid-season deadlines, checked with the edges (`edge_deadlines`)
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
    edges = edge_deadlines(tables)
    fixed = [deadline_of(tables, season, gw) for season, gw in REAL_DEADLINES]
    deadlines = sorted(set(edges) | set(fixed))
    assert not {s for s, _ in REAL_DEADLINES} & HOLDOUT_SEASONS
    by = tables["gameweek"].set_index(["season", "gw"])["deadline_time"]
    for season_gw in [(2019, 39), (2019, 47), (2021, 18), (2022, 8), (2016, 1), (2026, 1)]:
        assert by[season_gw] in edges, season_gw
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
    """Checking nothing must not report a pass (here: every season is the holdout)."""
    for name, df in built.items():
        if name == "gameweek":
            df = df.assign(season=2025)
        df.to_parquet(tmp_path / f"{name}.parquet")
    with pytest.raises(ValueError, match="no eligible deadlines"):
        run_leakage_check(tmp_path, n_deadlines=0)
