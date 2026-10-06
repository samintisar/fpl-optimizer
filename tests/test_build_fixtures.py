from datetime import UTC, datetime

import pandas as pd
import pytest
from synthetic_raw import TEAM_CODES, World, bootstrap, football_data, season_fixtures

from fplopt.build import build
from fplopt.build.common import TableValidationError
from fplopt.build.fixtures import (
    UNSCHEDULED_AT,
    FixtureCrossCheckError,
    assemble_fixtures,
    assemble_gameweek_results,
    assemble_gameweeks,
    attach_football_data,
    fixtures_from_merged_gw,
    last_bootstrap_per_season,
)


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def utc(text):
    return pd.Timestamp(text, tz="UTC").as_unit("us")


# --- merged_gw-derived fixtures ---------------------------------------------------------


def test_fixtures_from_merged_gw_detects_home_and_maps_codes():
    merged = pd.DataFrame(
        {
            "fixture": [1, 1, 1, 2, 2],
            "round": [1, 1, 1, 2, 2],
            "kickoff_time": ["2016-08-13T14:00:00Z"] * 3 + ["2016-08-20T14:00:00Z"] * 2,
            # fixture 1: team 5 (home) v team 9; fixture 2: team 9 (home) v team 5
            "was_home": [True, True, False, False, True],
            "opponent_team": [9, 9, 5, 9, 5],
            "team_h_score": [2, 2, 2, 0, 0],
            "team_a_score": [1, 1, 1, 3, 3],
        }
    )
    out = fixtures_from_merged_gw(merged, {5: 3, 9: 88}).set_index("fpl_fixture_id")
    assert out.loc[1, ["home_team_key", "away_team_key"]].tolist() == [3, 88]
    assert out.loc[2, ["home_team_key", "away_team_key"]].tolist() == [88, 3]
    assert out.loc[1, ["home_goals", "away_goals"]].tolist() == [2, 1]
    assert out.loc[2, "kickoff_time"] == utc("2016-08-20T14:00")
    assert out["finished"].all()


def test_fixtures_from_merged_gw_rejects_conflicting_kickoffs():
    merged = pd.DataFrame(
        {
            "fixture": [1, 1],
            "round": [1, 1],
            "kickoff_time": ["2016-08-13T14:00:00Z", "2016-08-14T14:00:00Z"],
            "was_home": [True, False],
            "opponent_team": [9, 5],
            "team_h_score": [0, 0],
            "team_a_score": [0, 0],
        }
    )
    with pytest.raises(ValueError, match="kickoff_time"):
        fixtures_from_merged_gw(merged, {5: 3, 9: 88})


# --- assembly ---------------------------------------------------------------------------


def raw_rows(gws, season=2019):
    n = len(gws)
    return pd.DataFrame(
        {
            "season": season,
            "fpl_fixture_id": range(1, n + 1),
            "gw": pd.array(gws, dtype="Int64"),
            "kickoff_time": [
                None if gw is None else utc("2020-06-20T18:00") + pd.Timedelta(days=i)
                for i, gw in enumerate(gws)
            ],
            "home_team_key": TEAM_CODES[:n],
            "away_team_key": TEAM_CODES[n : 2 * n],
            "home_goals": [1] * n,
            "away_goals": [0] * n,
            "finished": True,
        }
    )


def no_fd():
    return pd.DataFrame(
        columns=["season", "fd_date", "home_team_key", "away_team_key"]
        + ["fd_home_goals", "fd_away_goals"]
    )


def test_gw_index_is_dense_rank_for_covid_numbering():
    raw = raw_rows([29, 39, 39, 47])
    raw["finished"] = False
    raw[["home_goals", "away_goals"]] = pd.NA
    out = assemble_fixtures(raw, no_fd())
    assert out["gw_index"].tolist() == [1, 2, 2, 3]
    assert out["fixture_key"].tolist() == [2019001, 2019002, 2019003, 2019004]
    # available_at = lockdown of the GW's last kickoff (09:00 UK the day after, BST).
    gw39 = out[out["gw"] == 39]
    assert (gw39["available_at"] == utc("2020-06-23T08:00")).all()


def test_unscheduled_fixture_gets_far_future_and_no_result():
    raw = raw_rows([1, None], season=2026)
    raw["kickoff_time"] = [utc("2026-08-22T14:00"), pd.NaT]
    raw["finished"] = [True, False]
    raw["home_goals"] = pd.array([1, pd.NA], dtype="Int64")
    raw["away_goals"] = pd.array([0, pd.NA], dtype="Int64")
    out = assemble_fixtures(raw, no_fd())
    later = out.iloc[1]
    assert pd.isna(later["gw"]) and pd.isna(later["gw_index"]) and pd.isna(later["kickoff_time"])
    assert later["available_at"] == later["event_time"] == UNSCHEDULED_AT


def fd_rows(fixtures):
    df = football_data(fixtures, 2022)
    return pd.DataFrame(
        {
            "season": 2022,
            "fd_date": pd.to_datetime(df["Date"], format="%d/%m/%Y").dt.date,
            "home_team_key": [TEAM_CODES[i - 1] for i in fixtures["team_h"]],
            "away_team_key": [TEAM_CODES[i - 1] for i in fixtures["team_a"]],
            "fd_home_goals": df["FTHG"],
            "fd_away_goals": df["FTAG"],
        }
    )


def keyed(fixtures, season=2022):
    return pd.DataFrame(
        {
            "season": season,
            "fpl_fixture_id": fixtures["id"],
            "kickoff_time": pd.to_datetime(fixtures["kickoff_time"], utc=True).astype(
                "datetime64[us, UTC]"
            ),
            "home_team_key": [TEAM_CODES[i - 1] for i in fixtures["team_h"]],
            "away_team_key": [TEAM_CODES[i - 1] for i in fixtures["team_a"]],
            "home_goals": fixtures["team_h_score"].astype("Int64"),
            "away_goals": fixtures["team_a_score"].astype("Int64"),
            "finished": fixtures["finished"],
        }
    )


def test_football_data_join_sets_date_and_fails_on_goal_mismatch():
    fixtures = season_fixtures(2022).head(3)
    out = attach_football_data(keyed(fixtures), fd_rows(fixtures))
    assert out["fd_date"].tolist() == [datetime(2022, 8, 6).date()] * 3
    fd = fd_rows(fixtures)
    fd.loc[1, "fd_home_goals"] += 1
    with pytest.raises(FixtureCrossCheckError, match="1 goal mismatch"):
        attach_football_data(keyed(fixtures), fd)


def test_football_data_join_fails_on_unmatched_row_and_date_mismatch():
    fixtures = season_fixtures(2022).head(3)
    fd = fd_rows(fixtures)
    fd.loc[0, "fd_date"] = datetime(2022, 8, 7).date()
    with pytest.raises(FixtureCrossCheckError, match="1 date mismatch"):
        attach_football_data(keyed(fixtures), fd)
    with pytest.raises(FixtureCrossCheckError, match="without an FPL fixture"):
        attach_football_data(keyed(fixtures.head(2)), fd_rows(fixtures))


def test_missing_football_data_row_fails_except_recent_current_season():
    fixtures = season_fixtures(2022).head(12)  # rounds 1 and 2
    with pytest.raises(FixtureCrossCheckError, match="2 finished FPL fixture"):
        attach_football_data(keyed(fixtures), fd_rows(fixtures.drop([3, 4])))
    # Round 2 not yet published by football-data: fine for the newest season (and so are
    # fixtures on football-data's newest date, which may be published part-way through).
    out = attach_football_data(keyed(fixtures), fd_rows(fixtures.head(10)))
    assert out["fd_date"].isna().sum() == 2


# --- gameweek ---------------------------------------------------------------------------


def test_assemble_gameweeks_sources_and_bootstrap_event_without_fixtures():
    fixture = pd.DataFrame(
        {
            "season": [2016, 2016, 2022, 2022],
            "gw": pd.array([1, 1, 6, 8], dtype="Int64"),
            "gw_index": pd.array([1, 1, 6, 7], dtype="Int64"),
            "kickoff_time": [
                utc("2016-08-13T11:45"),
                utc("2016-08-14T15:00"),
                utc("2022-09-03T11:30"),
                utc("2022-09-16T19:00"),
            ],
        }
    )
    events = pd.DataFrame(
        {
            "gw": [6, 7, 8],
            "deadline_time": [
                utc("2022-09-03T10:00"),
                utc("2022-09-10T10:00"),
                utc("2022-09-16T17:30"),
            ],
            "average_entry_score": pd.array([45, 0, pd.NA], dtype="Int64"),
        }
    )
    gameweek = assemble_gameweeks(fixture, {2022: events}, {})
    assert "average_entry_score" not in gameweek.columns  # known at lockdown: gameweek_result
    out = gameweek.set_index(["season", "gw"])
    assert (2022, 7) not in out.index  # FPL's cancelled 2022-23 GW7: no fixtures, no row
    approx = out.loc[(2016, 1)]
    assert approx["deadline_source"] == "approx"
    assert approx["deadline_time"] == utc("2016-08-13T10:15")
    assert approx["first_kickoff"] == utc("2016-08-13T11:45")
    assert approx["lockdown_time"] == utc("2016-08-15T08:00")  # Monday 09:00 BST
    gw6 = out.loc[(2022, 6)]
    assert gw6["deadline_source"] == "bootstrap"
    # A schedule: known from publication (1 June of the season's start year), timed at the
    # deadline.
    assert (out["event_time"] == out["deadline_time"]).all()
    assert gw6["available_at"] == utc("2022-06-01T00:00")
    assert approx["available_at"] == utc("2016-06-01T00:00")

    results = assemble_gameweek_results(gameweek, {2022: events}).set_index(["season", "gw"])
    assert list(results.index) == list(out.index)
    assert results.loc[(2022, 6), "average_entry_score"] == 45
    assert pd.isna(results.loc[(2016, 1), "average_entry_score"])
    assert pd.isna(results.loc[(2022, 8), "average_entry_score"])  # not finished
    assert (results["event_time"] == out["lockdown_time"]).all()
    assert (results["available_at"] == out["lockdown_time"]).all()


def test_last_bootstrap_per_season_uses_payload_season_not_timestamp(world):
    fx21, fx22 = season_fixtures(2021), season_fixtures(2022)
    world.add_fplcache_bootstrap(datetime(2022, 5, 1, tzinfo=UTC), bootstrap(2021, fx21))
    # A July snapshot that still describes 2021-22 (FPL resets mid-July).
    world.add_fplcache_bootstrap(datetime(2022, 7, 10, tzinfo=UTC), bootstrap(2021, fx21))
    world.add_fplcache_bootstrap(datetime(2022, 7, 20, tzinfo=UTC), bootstrap(2022, fx22))
    world.add_own_bootstrap(datetime(2022, 9, 1, tzinfo=UTC), bootstrap(2022, fx22))
    found = {
        season: (ts, source)
        for season, (ts, _, source) in last_bootstrap_per_season(world.store).items()
    }
    assert found == {
        2021: (datetime(2022, 7, 10, tzinfo=UTC), "fplcache"),
        2022: (datetime(2022, 9, 1, tzinfo=UTC), "fpl"),
    }


# --- end to end -------------------------------------------------------------------------


def test_build_fixture_and_gameweek_end_to_end(world):
    world.add_vaastav_season(2016, fixtures_csv=False, teams_csv=False)
    fx23 = world.add_vaastav_season(2023)
    world.add_fplcache_bootstrap(datetime(2024, 6, 1, tzinfo=UTC), bootstrap(2023, fx23))
    world.add_current_season()
    build(["fixture", "gameweek", "gameweek_result"], world.ctx)

    fixture = world.ctx.table("fixture")
    assert fixture.groupby("season").size().to_dict() == {2016: 380, 2023: 380, 2026: 380}
    assert fixture["fixture_key"].min() == 2016001
    f16 = fixture[fixture["season"] == 2016].set_index("fpl_fixture_id").sort_index()
    expected = season_fixtures(2016)  # derived from merged_gw rows: same teams and results
    assert f16["home_team_key"].tolist() == [TEAM_CODES[i - 1] for i in expected["team_h"]]
    assert f16["away_team_key"].tolist() == [TEAM_CODES[i - 1] for i in expected["team_a"]]
    assert f16["home_goals"].tolist() == expected["team_h_score"].tolist()
    assert f16["fd_date"].notna().all()
    f26 = fixture[fixture["season"] == 2026].set_index("fpl_fixture_id")
    assert f26.loc[380, "available_at"] == UNSCHEDULED_AT
    assert f26["fd_date"].notna().sum() == 15
    assert f26.loc[f26["finished"], "home_goals"].notna().all()
    assert f26.loc[~f26["finished"], "home_goals"].isna().all()

    gameweek = world.ctx.table("gameweek")
    sources = gameweek.groupby("season")["deadline_source"].unique().map(list).to_dict()
    assert sources == {2016: ["approx"], 2023: ["bootstrap"], 2026: ["bootstrap"]}
    gw26 = gameweek[gameweek["season"] == 2026].set_index("gw")
    assert len(gw26) == 38
    result = world.ctx.table("gameweek_result")
    r26 = result[result["season"] == 2026].set_index("gw")
    assert r26.loc[2, "average_entry_score"] == 52
    assert pd.isna(r26.loc[3, "average_entry_score"])
    assert (r26["available_at"] == gw26["lockdown_time"]).all()


def test_fixture_build_fails_when_a_team_misses_matches(world):
    fixtures = season_fixtures(2022).iloc[:-1]
    world.add_vaastav_season(2022, fixtures=fixtures)
    with pytest.raises(TableValidationError, match="380 fixtures per season"):
        build(["fixture"], world.ctx)


# --- schedule and fixture_snapshot ------------------------------------------------------

RESULT_COLUMNS = {"home_goals", "away_goals", "finished", "fd_date"}


def test_schedule_is_the_fixture_list_without_results(world):
    world.add_vaastav_season(2023)
    world.add_current_season()
    build(["fixture", "schedule"], world.ctx)

    schedule = world.ctx.table("schedule")
    assert list(schedule.columns) == [
        "fixture_key",
        "season",
        "gw",
        "gw_index",
        "kickoff_time",
        "home_team_key",
        "away_team_key",
        "schedule_source",
        "event_time",
        "available_at",
    ]
    assert not RESULT_COLUMNS & set(schedule.columns)
    assert (schedule["schedule_source"] == "final").all()
    fixture = world.ctx.table("fixture").set_index("fixture_key")
    s = schedule.set_index("fixture_key")
    assert list(s.index) == list(fixture.index)
    same = ["season", "gw", "gw_index", "kickoff_time", "home_team_key", "away_team_key"]
    pd.testing.assert_frame_equal(s[same], fixture[same])
    # Known from publication, 1 June of the season's start year.
    published = s["season"].map({2023: utc("2023-06-01"), 2026: utc("2026-06-01")})
    assert (s["available_at"] == published).all()
    assert (s.loc[s["kickoff_time"].notna(), "event_time"] == s["kickoff_time"].dropna()).all()
    postponed = s.loc[2026380]
    assert pd.isna(postponed["gw"]) and postponed["event_time"] == UNSCHEDULED_AT
    assert postponed["available_at"] == utc("2026-06-01")


def test_fixture_snapshot_keeps_every_snapshot(world):
    first_at = datetime(2026, 9, 1, tzinfo=UTC)
    fx26 = world.add_current_season(at=first_at)  # fixture 380 postponed
    later = fx26.copy()
    later.loc[later["id"] == 1, "kickoff_time"] = "2026-08-02T19:30:00Z"  # rescheduled
    later.loc[later["id"] == 380, ["event", "kickoff_time"]] = [10, "2026-10-20T19:00:00Z"]
    later["started"] = later["finished"]
    second_at = datetime(2026, 9, 2, 3, tzinfo=UTC)
    world.add_own_fixtures(second_at, later)
    build(["fixture_snapshot"], world.ctx)

    snap = world.ctx.table("fixture_snapshot")
    assert list(snap.columns) == [
        "snapshot_at",
        "season",
        "fixture_key",
        "fpl_fixture_id",
        "gw",
        "kickoff_time",
        "home_team_key",
        "away_team_key",
        "started",
        "finished",
        "finished_provisional",
        "event_time",
        "available_at",
    ]
    assert not {"home_goals", "away_goals", "team_h_score", "team_a_score"} & set(snap.columns)
    assert snap.groupby("snapshot_at").size().to_dict() == {
        utc("2026-09-01"): 380,
        utc("2026-09-02T03:00"): 380,
    }
    assert (snap["event_time"] == snap["snapshot_at"]).all()
    assert (snap["available_at"] == snap["snapshot_at"]).all()
    assert (snap["season"] == 2026).all()
    s = snap.set_index(["snapshot_at", "fixture_key"])
    one, two = utc("2026-09-01"), utc("2026-09-02T03:00")
    assert s.loc[(one, 2026001), "kickoff_time"] == utc("2026-08-01T14:00")
    assert s.loc[(two, 2026001), "kickoff_time"] == utc("2026-08-02T19:30")
    assert pd.isna(s.loc[(one, 2026380), "gw"]) and pd.isna(s.loc[(one, 2026380), "kickoff_time"])
    assert s.loc[(two, 2026380), "gw"] == 10
    expected = season_fixtures(2026).set_index("id")
    first = s.loc[one].sort_index()
    assert first["home_team_key"].tolist() == [TEAM_CODES[i - 1] for i in expected["team_h"]]
    assert first["away_team_key"].tolist() == [TEAM_CODES[i - 1] for i in expected["team_a"]]
    assert first["started"].isna().all()  # absent from the first payload
    assert s.loc[(two, 2026001), "started"] and not s.loc[(two, 2026021), "started"]


def test_fixture_snapshot_without_own_archive_is_empty(world):
    world.add_vaastav_season(2023)
    build(["fixture_snapshot"], world.ctx)
    assert world.ctx.table("fixture_snapshot").empty


def test_fixture_snapshot_needs_a_full_fixture_list(world):
    fx26 = world.add_current_season()
    world.add_own_fixtures(datetime(2026, 9, 2, tzinfo=UTC), fx26.iloc[:-1])
    with pytest.raises(TableValidationError, match="380 fixtures per snapshot"):
        build(["fixture_snapshot"], world.ctx)
