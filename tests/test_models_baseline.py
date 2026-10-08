"""Baseline xP models (`fplopt.models`) and the synthetic league they are tested on
(`tests/synthetic_season.py`)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from synthetic_season import (
    SCORED_STATS,
    first_deadline,
    fpl_points,
    player_key,
    synthetic_tables,
)

from fplopt.build.tables import TABLES
from fplopt.features import FEATURES, compute_features
from fplopt.features.baseline import ep_next, player_pool
from fplopt.features.store import DataStore
from fplopt.models import MODELS
from fplopt.models.baseline import (
    FADE,
    HORIZON,
    LONG_RUN_N,
    ROLLING_N,
    xp_ep_next,
    xp_ep_next_fade,
    xp_rolling,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
XP_COLUMNS = ["player_key", "gw", "gw_index", "horizon", "xp"]
XP_DTYPES = {"player_key": "int64", "gw": "int64", "gw_index": "int64", "horizon": "int64"}


def deadline(tables, season, gw):
    gameweeks = tables["gameweek"]
    row = gameweeks[(gameweeks["season"] == season) & (gameweeks["gw"] == gw)]
    return row["deadline_time"].iloc[0]


def view(tables, season, gw):
    return DataStore(tables=tables).as_of(deadline(tables, season, gw))


def club_players(tables, club):
    season = tables["player_season"]
    keys = season["player_key"].unique()
    return sorted(int(k) for k in keys if (k - 100_000) // 100 == club)


def fixtures_of(tables, season, club):
    """Number of fixtures of `club` per GW of `season` in the final schedule."""
    schedule = tables["schedule"]
    s = schedule[schedule["season"] == season]
    plays = s[(s["home_team_key"] == club) | (s["away_team_key"] == club)]
    return plays["gw"].value_counts().to_dict()


def latest_snapshot(tables, season, gw):
    snaps = tables["player_snapshot"]
    at = snaps.loc[snaps["snapshot_at"] < deadline(tables, season, gw), "snapshot_at"].max()
    return snaps[snaps["snapshot_at"] == at].set_index("player_key")


def last_matches(tables, season, gw, key, n=ROLLING_N):
    pm = tables["player_match"]
    seen = pm[(pm["player_key"] == key) & (pm["available_at"] < deadline(tables, season, gw))]
    return seen.sort_values(["kickoff_time", "fixture_key"]).tail(n)


def xp_of(frame, key, horizon):
    row = frame[(frame["player_key"] == key) & (frame["horizon"] == horizon)]
    assert len(row) == 1
    return float(row["xp"].iloc[0])


# --- the synthetic league ------------------------------------------------------------------


@pytest.fixture(scope="module")
def league():
    """Two seasons, snapshots, no blanks. Copy before modifying."""
    return synthetic_tables(seasons=(2022, 2023))


def test_synthetic_tables_cover_every_table_but_understat(league):
    assert set(league) == set(TABLES) - {"understat_map", "understat_player_match"}
    for name, df in league.items():
        assert df.index.equals(pd.RangeIndex(len(df))), name
        assert isinstance(df["available_at"].dtype, pd.DatetimeTZDtype), name
        assert not df["available_at"].isna().any(), name
        key = list(TABLES[name].key)
        assert not df.duplicated(key).any(), name
    assert league["odds_snapshot"].empty and league["fixture_snapshot"].empty


def test_synthetic_tables_match_the_real_schemas(league):
    """Same columns, order and dtypes as data/*.parquet (when built locally)."""
    if not (DATA_DIR / "player_snapshot.parquet").exists():
        pytest.skip("data/ not built")
    for name, df in league.items():
        real = pq.read_schema(DATA_DIR / f"{name}.parquet").empty_table().to_pandas()
        real_dtypes = {c: str(t) for c, t in real.dtypes.items() if not c.startswith("__")}
        assert {c: str(t) for c, t in df.dtypes.items()} == real_dtypes, name
        assert list(df.columns) == list(real_dtypes), name


def test_synthetic_availability_follows_plan_section_4(league):
    gw = league["gameweek"].set_index(["season", "gw"])
    deadlines, lockdowns = gw["deadline_time"], gw["lockdown_time"]
    assert (gw["lockdown_time"] > gw["last_kickoff"]).all()
    assert (gw["first_kickoff"] > gw["deadline_time"]).all()
    nxt = gw.groupby(level="season")["deadline_time"].shift(-1).dropna()
    assert (gw.loc[nxt.index, "lockdown_time"] < nxt).all()
    june = league["gameweek"]["season"].map(lambda s: pd.Timestamp(f"{s}-06-01", tz="UTC"))
    assert (league["gameweek"]["available_at"] == june).all()
    assert (league["schedule"]["available_at"].dt.strftime("%m-%d") == "06-01").all()

    def at(table):
        return pd.MultiIndex.from_frame(league[table][["season", "gw"]])

    pg = league["player_gw"]
    assert (
        pg["available_at"].values == (deadlines[at("player_gw")] - pd.Timedelta("1h")).values
    ).all()
    own = league["player_gw_ownership"]
    assert (own["available_at"].values == deadlines[at("player_gw_ownership")].values).all()
    for table in ("player_match", "fixture", "gameweek_result"):
        assert (league[table]["available_at"].values == lockdowns[at(table)].values).all(), table
    ps = league["player_season"]
    gw1 = ps["season"].map(lambda s: deadlines[(s, 1)])
    assert (ps["available_at"] == gw1 - pd.Timedelta("1h")).all()
    snaps = league["player_snapshot"]
    assert (snaps["available_at"] == snaps["snapshot_at"]).all()
    assert (snaps["event_time"] == snaps["snapshot_at"]).all()
    assert set(snaps["snapshot_at"]) == set(gw["deadline_time"] - pd.Timedelta("2h"))
    rating = league["team_rating"]
    matches = rating["fixture_key"].notna()
    delay = rating.loc[matches, "available_at"] - rating.loc[matches, "kickoff_time"]
    assert (delay == pd.Timedelta("2h")).all()


def test_synthetic_points_follow_fpl_scoring(league):
    pm = league["player_match"]
    positions = league["player_season"].drop_duplicates("player_key")
    pm = pm.merge(positions[["player_key", "element_type"]], on="player_key")
    expected = [
        fpl_points(row["element_type"], row["minutes"], **{s: row[s] for s in SCORED_STATS})
        for _, row in pm.iterrows()
    ]
    assert (pm["total_points"] == expected).all()
    # Clean sheets only with 60+ minutes and nothing conceded.
    cs = pm[pm["clean_sheets"] == 1]
    assert (cs["minutes"] >= 60).all() and (cs["goals_conceded"] == 0).all()
    assert pm["total_points"].between(-4, 30).all() and pm["bonus"].sum() > 0


def test_synthetic_league_has_zero_minute_players_and_price_changes(league):
    pm = league["player_match"]
    minutes = pm.groupby("player_key")["minutes"].sum()
    assert (minutes == 0).sum() >= 20  # every backup GK, at least
    assert (pm["minutes"] == 0).mean() > 0.1
    pg = league["player_gw"]
    changes = pg.sort_values("gw").groupby(["player_key", "season"])["value"].nunique()
    assert (changes > 1).mean() > 0.5
    assert pg["value"].between(38, 135).all()
    snaps = league["player_snapshot"]
    assert set(snaps["status"]) == {"a", "d", "i"}
    assert snaps["ep_next"].isna().any() and (snaps["form"] > 0).any()


def test_a_squad_fits_the_budget(league):
    """The 15 cheapest players per position quota with <= 3 per club cost <= 1000."""
    pool = player_pool(view(league, 2023, 1))
    assert pool["price"].between(38, 135).all()
    chosen = []
    per_club: dict[int, int] = {}
    for element_type, quota in ((1, 2), (2, 5), (3, 5), (4, 3)):
        candidates = pool[pool["element_type"] == element_type].sort_values(["price", "player_key"])
        taken = 0
        for _, player in candidates.iterrows():
            if taken == quota:
                break
            if per_club.get(player["team_key"], 0) < 3:
                per_club[player["team_key"]] = per_club.get(player["team_key"], 0) + 1
                chosen.append(player["price"])
                taken += 1
        assert taken == quota
    assert sum(chosen) <= 1000


def test_synthetic_tables_are_deterministic_per_seed():
    one, two = synthetic_tables(seed=3), synthetic_tables(seed=3)
    for name in one:
        pd.testing.assert_frame_equal(one[name], two[name])
    one["player_match"].loc[0, "minutes"] = -1  # copies: the cache is not modified
    assert synthetic_tables(seed=3)["player_match"].loc[0, "minutes"] != -1
    other = synthetic_tables(seed=4)
    assert not other["player_match"]["total_points"].equals(two["player_match"]["total_points"])


def test_blank_and_double_options_move_fixtures():
    tables = synthetic_tables(blank=(2023, 10, 1), double=(2023, 20, 2, 25))
    one = fixtures_of(tables, 2023, 1)
    assert 10 not in one and one[12] == 2
    two = fixtures_of(tables, 2023, 2)
    assert two[20] == 2 and 25 not in two
    pg = tables["player_gw"]
    assert pg[(pg["team_key"] == 1) & (pg["gw"] == 10)].empty
    assert not pg[(pg["team_key"] == 1) & (pg["gw"] == 12)].empty
    gw = tables["gameweek"].set_index("gw")
    assert (gw["last_kickoff"] < gw["lockdown_time"]).all()
    with pytest.raises(ValueError, match="fixtures"):
        synthetic_tables(blank=(2023, 10, 1), double=(2023, 10, 1))


def test_without_snapshots_the_tables_are_otherwise_identical():
    with_snaps, without = synthetic_tables(), synthetic_tables(snapshots=False)
    assert without["player_snapshot"].empty
    assert dict(without["player_snapshot"].dtypes) == dict(with_snaps["player_snapshot"].dtypes)
    pd.testing.assert_frame_equal(with_snaps["player_match"], without["player_match"])
    pd.testing.assert_frame_equal(with_snaps["player_gw"], without["player_gw"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seasons": (2022, 2023)},
        {"snapshots": False},
        {"blank": (2023, 10, 1), "double": (2023, 11, 5)},
    ],
)
def test_every_feature_and_model_runs_on_the_synthetic_league(kwargs):
    tables = synthetic_tables(**kwargs)
    store = DataStore(tables=tables)
    season = max(kwargs.get("seasons", (2023,)))
    for gw in (1, 2, 10, 11, 37, 38):
        v = store.as_of(deadline(tables, season, gw))
        features = compute_features(v)
        assert list(features) == list(FEATURES)
        assert len(features["player_pool"]) == 300
        assert len(features["upcoming_fixtures"]) >= 20
        for name, model in MODELS.items():
            xp = model(v)
            assert len(xp) == 300 * min(HORIZON + 1, 39 - gw), (name, gw)


# --- the xP models -------------------------------------------------------------------------


def test_registry():
    assert MODELS == {
        "rolling": xp_rolling,
        "ep_next": xp_ep_next,
        "ep_next_fade": xp_ep_next_fade,
    }


@pytest.mark.parametrize("model", [xp_rolling, xp_ep_next, xp_ep_next_fade])
def test_xp_frames_have_fixed_columns_dtypes_and_order(model, league):
    xp = model(view(league, 2023, 10))
    assert list(xp.columns) == XP_COLUMNS
    assert dict(xp.dtypes) == {**XP_DTYPES, "xp": np.dtype("float64")}
    assert xp.index.equals(pd.RangeIndex(len(xp)))
    assert xp.equals(xp.sort_values(["player_key", "horizon"], kind="mergesort"))
    assert not xp.duplicated(["player_key", "horizon"]).any()
    pool = player_pool(view(league, 2023, 10))
    assert set(xp["player_key"]) == set(pool["player_key"])
    assert sorted(xp["horizon"].unique()) == list(range(HORIZON + 1))
    assert (xp["gw_index"] == 10 + xp["horizon"]).all() and (xp["gw"] == xp["gw_index"]).all()
    assert not xp["xp"].isna().any() and (xp["xp"] != 0).any()
    end = model(view(league, 2023, 36))
    assert sorted(end["horizon"].unique()) == [0, 1, 2]  # GW36-38: the season ends


@pytest.mark.parametrize("model", [xp_rolling, xp_ep_next, xp_ep_next_fade])
def test_xp_models_are_deterministic(model, league):
    first = model(view(league, 2023, 15))
    pd.testing.assert_frame_equal(first, model(view(league, 2023, 15)))
    store = DataStore(tables=league)
    v = store.as_of(deadline(league, 2023, 15))
    pd.testing.assert_frame_equal(first, model(v))
    pd.testing.assert_frame_equal(first, model(v))


def test_rolling_rate_spans_the_season_boundary(league):
    xp = xp_rolling(view(league, 2023, 3))
    spanning = 0
    for key in [player_key(club, slot) for club in (1, 7, 13) for slot in range(15)]:
        last = last_matches(league, 2023, 3, key)
        assert len(last) == ROLLING_N
        spanning += set(last["season"]) == {2022, 2023}
        rate = last["rescored_points"].mean()
        for horizon in range(HORIZON + 1):
            assert xp_of(xp, key, horizon) == pytest.approx(rate)  # one fixture per GW
    assert spanning >= 40  # 2 matches of 2023 (GW1-2) and 3 of 2022


def test_rolling_counts_zero_minute_rows_and_unknown_players_get_zero(league):
    tables = {name: df.copy() for name, df in league.items()}
    key = player_key(4, 9)  # a regular MID
    last = last_matches(tables, 2023, 10, key)
    pm = tables["player_match"]
    benched = last.index[-1]
    pm.loc[benched, ["minutes", "total_points", "rescored_points"]] = 0
    pm.loc[benched, SCORED_STATS[:5]] = 0
    expected = pm.loc[last.index, "rescored_points"].sum() / ROLLING_N
    # A new signing in the newest snapshot only: no matches anywhere.
    snaps = tables["player_snapshot"]
    newest = snaps[snaps["snapshot_at"] == latest_snapshot(tables, 2023, 10)["snapshot_at"].iloc[0]]
    signing = newest.iloc[[0]].assign(player_key=999_999, element_id=999)
    tables["player_snapshot"] = pd.concat([snaps, signing], ignore_index=True)

    xp = xp_rolling(view(tables, 2023, 10))
    assert xp_of(xp, key, 0) == pytest.approx(expected)
    backup_gk = player_key(4, 1)
    assert (xp.loc[xp["player_key"] == backup_gk, "xp"] == 0).all()  # five 0-minute rows
    new = xp[xp["player_key"] == 999_999]
    assert len(new) == HORIZON + 1 and (new["xp"] == 0).all()


def test_doubles_count_twice_and_blanks_score_zero():
    tables = synthetic_tables(blank=(2023, 12, 1))  # club 1 and its opponent: GW12 -> GW14
    opponent = next(c for c in range(2, 21) if fixtures_of(tables, 2023, c).get(14) == 2)
    rolling, ep = xp_rolling(view(tables, 2023, 10)), xp_ep_next(view(tables, 2023, 10))
    snapshot = latest_snapshot(tables, 2023, 10)
    for club in (1, opponent):
        for key in club_players(tables, club):
            rate = last_matches(tables, 2023, 10, key)["rescored_points"].mean()
            for horizon, n in ((0, 1), (1, 1), (2, 0), (3, 1), (4, 2), (5, 1)):
                assert xp_of(rolling, key, horizon) == pytest.approx(n * rate)
            ep_next_ = snapshot.loc[key, "ep_next"]
            ep_next_ = 0.0 if pd.isna(ep_next_) else float(ep_next_)
            for horizon, n in ((0, 1), (2, 0), (4, 2)):
                assert xp_of(ep, key, horizon) == pytest.approx(n * ep_next_)
    other = player_key(next(c for c in range(2, 21) if c != opponent), 9)
    assert xp_of(rolling, other, 2) == pytest.approx(xp_of(rolling, other, 0)) != 0


def test_ep_next_is_the_target_xp_and_the_rate_after():
    tables = synthetic_tables()
    snaps = tables["player_snapshot"]
    newest = latest_snapshot(tables, 2023, 10)["snapshot_at"].iloc[0]
    nulled = player_key(5, 9)
    snaps.loc[(snaps["snapshot_at"] == newest) & (snaps["player_key"] == nulled), "ep_next"] = None
    xp = xp_ep_next(view(tables, 2023, 10))
    snapshot = latest_snapshot(tables, 2023, 10)
    for key in [player_key(club, slot) for club in (2, 5, 9) for slot in range(15)]:
        expected = snapshot.loc[key, "ep_next"]
        expected = 0.0 if pd.isna(expected) else float(expected)
        for horizon in range(HORIZON + 1):
            assert xp_of(xp, key, horizon) == pytest.approx(expected)
    assert (xp.loc[xp["player_key"] == nulled, "xp"] == 0).all()
    features = ep_next(view(tables, 2023, 10))
    assert list(features.columns) == ["player_key", "ep_next", "form"]
    assert str(features["form"].dtype) == "Float64"


def test_ep_next_falls_back_to_form_after_a_blank_target():
    tables = synthetic_tables(blank=(2023, 10, 1))  # club 1: blank in GW10, double in GW12
    snaps = tables["player_snapshot"]
    newest = latest_snapshot(tables, 2023, 10)["snapshot_at"].iloc[0]
    players = club_players(tables, 1)
    no_form = players[3]
    snaps.loc[(snaps["snapshot_at"] == newest) & (snaps["player_key"] == no_form), "form"] = None
    xp = xp_ep_next(view(tables, 2023, 10))
    snapshot = latest_snapshot(tables, 2023, 10)
    assert (snapshot.loc[players, "ep_next"].fillna(0) == 0).all()  # FPL: 0 in a blank
    for key in players:
        form = snapshot.loc[key, "form"]
        form = 0.0 if pd.isna(form) else float(form)
        for horizon, n in ((0, 0), (1, 1), (2, 2), (3, 1)):
            assert xp_of(xp, key, horizon) == pytest.approx(n * form)
    assert (xp.loc[xp["player_key"] == no_form, "xp"] == 0).all()
    assert (snapshot.loc[players, "form"].fillna(0) > 0).any()


def test_ep_next_without_snapshots_is_zero_and_warns(caplog):
    tables = synthetic_tables(snapshots=False)
    with caplog.at_level("WARNING", logger="fplopt.models.baseline"):
        xp = xp_ep_next(view(tables, 2023, 10))
    assert (xp["xp"] == 0).all() and len(xp) == 300 * (HORIZON + 1)
    assert list(xp.columns) == XP_COLUMNS and dict(xp.dtypes) == {
        **XP_DTYPES,
        "xp": np.dtype("float64"),
    }
    warnings = [r for r in caplog.records if r.name == "fplopt.models.baseline"]
    assert len(warnings) == 1 and "no player snapshot" in warnings[0].getMessage()
    # The rolling model does not need snapshots.
    assert (xp_rolling(view(tables, 2023, 10))["xp"] != 0).any()


def test_ep_next_fade_starts_at_ep_next_and_fades_to_the_long_run_rate():
    tables = synthetic_tables(seasons=(2022, 2023), blank=(2023, 12, 1))  # club 1: GW12 -> 14
    v = view(tables, 2023, 10)
    fade, ep = xp_ep_next_fade(v), xp_ep_next(v)
    target = ep[ep["horizon"] == 0].reset_index(drop=True)
    pd.testing.assert_frame_equal(fade[fade["horizon"] == 0].reset_index(drop=True), target)
    assert (FADE, LONG_RUN_N) == (0.5, 10)
    snapshot = latest_snapshot(tables, 2023, 10)
    for club in (1, 2, 9):
        n_by_gw = fixtures_of(tables, 2023, club)
        for key in club_players(tables, club):
            ep_rate = snapshot.loc[key, "ep_next"]
            ep_rate = 0.0 if pd.isna(ep_rate) else float(ep_rate)  # one fixture in GW10
            last = last_matches(tables, 2023, 10, key, n=LONG_RUN_N)
            assert len(last) == LONG_RUN_N
            long_run = last["rescored_points"].mean()
            for horizon in range(1, HORIZON + 1):
                w = 0.5**horizon
                n = n_by_gw.get(10 + horizon, 0)
                expected = n * (w * ep_rate + (1 - w) * long_run)
                assert xp_of(fade, key, horizon) == pytest.approx(expected), (key, horizon)
    assert xp_of(fade, club_players(tables, 1)[5], 2) == 0  # the blank


def test_ep_next_fade_without_matches_fades_to_zero_and_without_snapshots_is_zero(caplog):
    tables = synthetic_tables()
    snaps = tables["player_snapshot"]
    newest = latest_snapshot(tables, 2023, 2)["snapshot_at"].iloc[0]
    signing = snaps[snaps["snapshot_at"] == newest].iloc[[0]]
    signing = signing.assign(player_key=999_999, element_id=999, ep_next=4.0)
    tables["player_snapshot"] = pd.concat([snaps, signing], ignore_index=True)
    fade = xp_ep_next_fade(view(tables, 2023, 2))
    new = fade[fade["player_key"] == 999_999].set_index("horizon")["xp"]
    n_by_gw = fixtures_of(tables, 2023, int(signing["team_key"].iloc[0]))
    for horizon, value in new.items():
        assert value == pytest.approx(n_by_gw.get(2 + horizon, 0) * 4.0 * 0.5**horizon)
    tables = synthetic_tables(snapshots=False)
    with caplog.at_level("WARNING", logger="fplopt.models.baseline"):
        xp = xp_ep_next_fade(view(tables, 2023, 10))
    assert (xp["xp"] == 0).all() and "no player snapshot" in caplog.text


def test_first_deadline_is_a_friday_in_august():
    for season in (2016, 2022, 2023):
        d = first_deadline(season)
        assert d.weekday() == 4 and d.month == 8 and 8 <= d.day <= 14
