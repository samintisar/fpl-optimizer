import inspect
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from synthetic_raw import CURRENT_AT, TEAM_CODES

from fplopt.features import FEATURES, compute_features
from fplopt.features.baseline import (
    FORM_STATS,
    availability,
    ep_next,
    player_pool,
    recent_form,
    team_strength,
    upcoming_fixtures,
)
from fplopt.features.store import DataStore

UTC_US = pd.DatetimeTZDtype("us", "UTC")


def deadline(tables, season, gw):
    gameweeks = tables["gameweek"]
    row = gameweeks[(gameweeks["season"] == season) & (gameweeks["gw"] == gw)]
    return row["deadline_time"].iloc[0]


def view(tables, season, gw):
    return DataStore(tables=tables).as_of(deadline(tables, season, gw))


def player_of(team_code):
    """The synthetic world has one player per club: code 100000 + team id."""
    return 100000 + TEAM_CODES.index(team_code) + 1


def move_fixture(tables, season, from_gw, to_gw, n=0):
    """Move the n-th fixture of `from_gw` to `to_gw` (two days after that GW's first kickoff)
    in the final schedule, and drop its players' `from_gw` player_gw rows. Returns the
    fixture's two team keys."""
    schedule = tables["schedule"]
    in_season = schedule["season"] == season
    fixture = schedule[in_season & (schedule["gw"] == from_gw)].iloc[n]
    first = schedule.loc[in_season & (schedule["gw"] == to_gw), "kickoff_time"].min()
    at = schedule["fixture_key"] == fixture["fixture_key"]
    schedule.loc[at, ["gw", "gw_index"]] = to_gw
    schedule.loc[at, "kickoff_time"] = first + pd.Timedelta(days=2)
    teams = [int(fixture["home_team_key"]), int(fixture["away_team_key"])]
    pg = tables["player_gw"]
    drop = (pg["season"] == season) & (pg["gw"] == from_gw) & pg["team_key"].isin(teams)
    tables["player_gw"] = pg[~drop].reset_index(drop=True)
    return teams


# --- registry -----------------------------------------------------------------------------


def test_registry_and_builder_signatures():
    assert list(FEATURES) == [
        "player_pool",
        "availability",
        "ep_next",
        "recent_form",
        "upcoming_fixtures",
        "team_strength",
    ]
    for builder in FEATURES.values():
        assert len(inspect.signature(builder).parameters) == 1


def test_compute_features_is_deterministic_with_stable_dtypes(tables):
    mid = compute_features(view(tables, 2023, 10))
    again = compute_features(view(tables, 2023, 10))
    first = compute_features(view(tables, 2023, 1))
    live = compute_features(view(tables, 2026, 6))
    assert list(mid) == list(FEATURES)
    for name in FEATURES:
        pd.testing.assert_frame_equal(mid[name], again[name])
        assert mid[name].index.equals(pd.RangeIndex(len(mid[name])))
        assert dict(first[name].dtypes) == dict(mid[name].dtypes), name
        assert dict(live[name].dtypes) == dict(mid[name].dtypes), name


# --- player pool / availability / ep_next -------------------------------------------------


def test_player_pool_from_player_gw_without_snapshots(tables):
    pg = tables["player_gw"]
    tables["player_gw"] = pg.assign(value=40 + pg["gw"])  # price varies by GW
    pool = player_pool(view(tables, 2023, 10))
    assert len(pool) == 20
    assert pool["player_key"].is_monotonic_increasing
    assert list(pool.columns) == ["player_key", "element_type", "team_key", "price", "source"]
    assert (pool["source"] == "player_gw").all()
    assert (pool["price"] == 50).all()  # the GW10 row (deadline - 1h), not GW9 or GW11


def test_player_pool_keeps_blank_gw_players_and_drops_departed_ones(tables):
    pg = tables["player_gw"]
    tables["player_gw"] = pg.assign(value=40 + pg["gw"])
    blank = move_fixture(tables, 2023, from_gw=10, to_gw=12)
    # A player who left after GW5 (no rows from GW6) is not in the GW10 pool.
    departed = next(p for p in map(player_of, TEAM_CODES) if p not in map(player_of, blank))
    pg = tables["player_gw"]
    tables["player_gw"] = pg[~((pg["player_key"] == departed) & (pg["gw"] > 5))]

    pool = player_pool(view(tables, 2023, 10)).set_index("player_key")
    assert len(pool) == 19 and departed not in pool.index
    for team in blank:
        assert pool.loc[player_of(team), "price"] == 49  # their newest earlier row (GW9)
        assert pool.loc[player_of(team), "team_key"] == team
    assert (pool.drop([player_of(t) for t in blank])["price"] == 50).all()


def test_player_pool_from_latest_snapshot(tables):
    snaps = tables["player_snapshot"]
    tables["player_snapshot"] = snaps.assign(now_cost=55)
    pool = player_pool(view(tables, 2026, 6))  # 2026 bootstrap of 1 Sep is visible
    assert len(pool) == 20
    assert (pool["source"] == "snapshot").all()
    assert (pool["price"] == 55).all()
    # GW1 of 2026 (1 Aug): no 2026 snapshot yet -> player_gw rows of GW1.
    assert (player_pool(view(tables, 2026, 1))["source"] == "player_gw").all()


def test_player_pool_uses_only_the_newest_snapshot(tables):
    """A player missing from the newest snapshot (removed from the game) is not in the pool,
    and an equal-time 'fpl' row beats the 'fplcache' row."""
    snaps = tables["player_snapshot"]
    current = snaps[snaps["season"] == 2026]
    removed = int(current["player_key"].iloc[0])
    newer = current[current["player_key"] != removed]
    later = CURRENT_AT + timedelta(days=2)
    newer = newer.assign(snapshot_at=later, event_time=later, available_at=later)
    cache = newer.assign(source="fplcache", now_cost=1)
    newer = newer.assign(source="fpl", now_cost=61)
    for frame in (newer, cache):
        for column in ("snapshot_at", "event_time", "available_at"):
            frame[column] = frame[column].astype(UTC_US)
    tables["player_snapshot"] = pd.concat([snaps, cache, newer], ignore_index=True)
    pool = player_pool(view(tables, 2026, 6))
    assert len(pool) == 19 and removed not in set(pool["player_key"])
    assert (pool["price"] == 61).all()


def test_availability_and_ep_next(tables):
    snaps = tables["player_snapshot"]
    news_at = pd.Timestamp("2026-08-30 10:00", tz="UTC")
    tables["player_snapshot"] = snaps.assign(
        status="d",
        chance_of_playing_next_round=pd.array([75] * len(snaps), dtype="Int64"),
        news_added=pd.Series([news_at] * len(snaps), dtype=UTC_US),
        ep_next=pd.array([4.5] * len(snaps), dtype="Float64"),
    )
    live = view(tables, 2026, 6)
    avail = availability(live)
    assert list(avail.columns) == [
        "player_key",
        "status",
        "chance_of_playing_next_round",
        "news_added",
    ]
    assert len(avail) == 20
    assert (avail["status"] == "d").all()
    assert (avail["chance_of_playing_next_round"] == 75).all()
    assert (avail["news_added"] == news_at).all()
    assert (ep_next(live)["ep_next"] == 4.5).all()
    # No snapshot in 2023 before the deadline: pool players with nulls.
    old = view(tables, 2023, 10)
    avail = availability(old)
    assert len(avail) == 20
    assert avail[["status", "chance_of_playing_next_round", "news_added"]].isna().all().all()
    assert ep_next(old)["ep_next"].isna().all()


# --- recent form -------------------------------------------------------------------------


def test_recent_form_windows_exclude_the_target_gameweek(tables):
    target = view(tables, 2023, 10)
    visible = target.table("player_match")
    assert visible.loc[visible["season"] == 2023, "gw"].max() == 9  # not available yet
    pm = tables["player_match"]
    pm = pm[pm["season"] == 2023]
    tables["player_match"] = tables["player_match"].assign(
        total_points=tables["player_match"]["gw"]
    )

    form = recent_form(view(tables, 2023, 10)).set_index("player_key")
    assert len(form) == 20
    assert (form["matches_season"] == 9).all() and (form["matches_last5"] == 5).all()
    assert (form["apps_last5"] == 5).all()
    assert (form["minutes_last5"] == 450).all() and (form["minutes_season"] == 810).all()
    assert (form["total_points_last5"] == sum(range(5, 10))).all()
    assert (form["total_points_season"] == sum(range(1, 10))).all()
    player = player_of(TEAM_CODES[0])
    goals = pm[(pm["player_key"] == player) & (pm["gw"] < 10)].sort_values("gw")["goals_scored"]
    assert form.loc[player, "goals_scored_season"] == goals.sum()
    assert form.loc[player, "goals_scored_last5"] == goals.tail(5).sum()
    assert form.loc[player, "goals_scored_per90_season"] == pytest.approx(goals.sum() / 9)
    assert form.loc[player, "us_npxg_per90_last5"] == pytest.approx(0.3)
    assert form["fpl_xg_last5"].isna().all() and form["fpl_xg_per90_last5"].isna().all()


def test_recent_form_rates_are_null_without_minutes(tables):
    pm = tables["player_match"]
    player = player_of(TEAM_CODES[0])
    benched = (pm["player_key"] == player) & (pm["season"] == 2023) & (pm["gw"].between(5, 9))
    tables["player_match"] = pm.assign(minutes=pm["minutes"].mask(benched, 0))
    form = recent_form(view(tables, 2023, 10)).set_index("player_key")
    assert form.loc[player, "minutes_last5"] == 0 and form.loc[player, "apps_last5"] == 0
    assert pd.isna(form.loc[player, "total_points_per90_last5"])
    assert form.loc[player, "minutes_season"] == 360


def test_recent_form_is_empty_before_the_first_match(tables):
    form = recent_form(view(tables, 2023, 1))
    assert form.empty
    for stat in FORM_STATS:
        assert f"{stat}_last5" in form.columns and f"{stat}_season" in form.columns


# --- upcoming fixtures ---------------------------------------------------------------------


def test_upcoming_fixtures_blank_and_double_counts(tables):
    blank = move_fixture(tables, 2023, from_gw=10, to_gw=12)
    up = upcoming_fixtures(view(tables, 2023, 10))
    assert sorted(up["gw"].unique()) == list(range(10, 16))
    assert up["horizon"].min() == 0 and up["horizon"].max() == 5
    assert len(up) == 20 * 6 + 2  # one row per team-GW, plus the second fixture of doubles
    counts = up.drop_duplicates(["team_key", "gw"]).set_index(["team_key", "gw"])["n_fixtures"]
    for team in blank:
        assert counts[(team, 10)] == 0 and counts[(team, 12)] == 2
        blank_row = up[(up["team_key"] == team) & (up["gw"] == 10)]
        assert blank_row["fixture_key"].isna().all() and blank_row["opponent_team_key"].isna().all()
        double = up[(up["team_key"] == team) & (up["gw"] == 12)]
        assert len(double) == 2 and blank[1 - blank.index(team)] in set(double["opponent_team_key"])
    others = counts.drop([(t, gw) for t in blank for gw in (10, 12)])
    assert (others == 1).all()
    home = up[(up["team_key"] == blank[0]) & (up["gw"] == 12)]
    assert home["is_home"].tolist().count(True) >= 1
    # Near the end of the season the horizon is shorter.
    assert sorted(upcoming_fixtures(view(tables, 2023, 36))["gw"].unique()) == [36, 37, 38]


def test_upcoming_fixtures_use_the_as_of_snapshot_schedule(tables):
    up = upcoming_fixtures(view(tables, 2026, 6))
    assert (up["schedule_source"] == "snapshot").all()
    assert 2026380 not in set(up["fixture_key"].dropna())  # postponed: no GW in the snapshot
    assert sorted(up["gw"].unique()) == list(range(6, 12))


# --- team strength -----------------------------------------------------------------------


def odds_rows(rows):
    df = pd.DataFrame(
        rows, columns=["fixture_key", "source", "bookmaker", "outcome", "price", "snapshot_at"]
    )
    df["snapshot_at"] = pd.to_datetime(df["snapshot_at"], utc=True).astype(UTC_US)
    return df.assign(
        season=2023,
        market="h2h",
        line=pd.array([None] * len(df), dtype="Float64"),
        is_closing=False,
        event_time=df["snapshot_at"],
        available_at=df["snapshot_at"],
    )


def test_team_strength_elo_and_as_of_odds(tables):
    target = view(tables, 2023, 10)
    d = target.deadline
    fixtures = target.schedule(2023)
    fixtures = fixtures[fixtures["gw"] == 10].reset_index(drop=True)
    fd_fx, api_fx, none_fx = (int(k) for k in fixtures["fixture_key"].iloc[:3])
    kickoff = fixtures["kickoff_time"].iloc[0]
    before, earlier, after = (
        d - pd.Timedelta(hours=2),
        d - pd.Timedelta(days=1),
        d + pd.Timedelta(1, "h"),
    )
    rows = []
    for outcome, price, late in (("home", 2.0, 9.0), ("draw", 4.0, 9.0), ("away", 4.0, 1.1)):
        rows += [
            (fd_fx, "football-data", "avg", outcome, price, before),
            (fd_fx, "football-data", "avg", outcome, late, after),  # after the deadline
            (fd_fx, "odds-api", "bk1", outcome, 9.9, before),  # football-data preferred
        ]
    closing = odds_rows(
        [
            (fd_fx, "football-data", "avg", o, p, kickoff)
            for o, p in (("home", 1.5), ("draw", 4.0), ("away", 8.0))
        ]
    ).assign(is_closing=True)
    for outcome, prices in (
        ("home", (2.0, 3.0, 2.5)),
        ("draw", (3.0, 3.4, 3.2)),
        ("away", (4.0, 2.0, 3.0)),
    ):
        for bookmaker, price in zip(("bk1", "bk2", "bk3"), prices, strict=True):
            rows.append((api_fx, "odds-api", bookmaker, outcome, price, before))
        rows += [
            (api_fx, "odds-api", "bk4", outcome, 50.0, earlier),  # not in the newest snapshot
            (api_fx, "odds-api", "bk1", outcome, 50.0, after),
        ]
    tables["odds_snapshot"] = pd.concat([odds_rows(rows), closing], ignore_index=True)

    strength = team_strength(view(tables, 2023, 10))
    assert len(strength) == 2 * len(fixtures)
    by_side = strength.set_index(["fixture_key", "is_home"])
    fd_home = by_side.loc[(fd_fx, True)]
    raw = np.array([1 / 2.0, 1 / 4.0, 1 / 4.0])
    expected = raw / raw.sum()
    assert fd_home["odds_source"] == "football-data"
    assert [fd_home["p_win"], fd_home["p_draw"], fd_home["p_loss"]] == pytest.approx(expected)
    fd_away = by_side.loc[(fd_fx, False)]
    assert fd_away["p_win"] == pytest.approx(expected[2])
    api_home = by_side.loc[(api_fx, True)]
    raw = np.array([1 / 2.5, 1 / 3.2, 1 / 3.0])  # medians of bk1-bk3
    assert api_home["odds_source"] == "odds-api"
    assert [api_home["p_win"], api_home["p_draw"], api_home["p_loss"]] == pytest.approx(
        raw / raw.sum()
    )
    assert pd.isna(by_side.loc[(none_fx, True), "p_win"])
    assert pd.isna(by_side.loc[(none_fx, True), "odds_source"])
    sums = strength[["p_win", "p_draw", "p_loss"]].sum(axis=1, min_count=3).dropna()
    assert np.allclose(sums, 1.0)

    ratings = tables["team_rating"]
    ratings = ratings[ratings["available_at"] < d].sort_values("event_time")
    elo = ratings.groupby("team_key")["rating_after"].last()
    for row in strength.itertuples():
        assert row.elo == pytest.approx(elo[row.team_key])
        assert row.opponent_elo == pytest.approx(elo[row.opponent_team_key])


def test_team_strength_has_elo_for_every_club_at_gameweek_one(tables):
    """Pre-season team_rating rows give every club (promoted ones too) a rating before its
    first match."""
    for season in (2023, 2024, 2026):
        strength = team_strength(view(tables, season, 1))
        assert len(strength) == 20, season
        assert strength["elo"].notna().all() and strength["opponent_elo"].notna().all(), season
