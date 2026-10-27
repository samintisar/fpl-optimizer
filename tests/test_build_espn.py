from datetime import UTC, datetime

import pandas as pd
import pytest
from synthetic_raw import REPO_CONFIG, RUN, TEAM_CODES, World
from test_build_understat import ALL as UNDERSTAT_BUILD
from test_build_understat import add_understat_season

from fplopt.build import build
from fplopt.build.espn import (
    AVAILABLE_AFTER,
    MAP_SCHEMA,
    EspnBuildError,
    build_espn_tables,
    parse_summary,
)
from fplopt.build.teams import team_dim_from_config


def player(athlete_id, name, starter, sub_in=False, sub_out=False):
    return {
        "starter": starter,
        "subbedIn": sub_in,
        "subbedOut": sub_out,
        "athlete": {"id": str(athlete_id), "displayName": name},
    }


def key_event(text, seconds, team, *athletes, shootout=False, period=None):
    if period is None:
        period = 5 if shootout else 1 + (seconds >= 2700) + (seconds >= 5400) + (seconds >= 6300)
    return {
        "type": {"text": text},
        "period": {"number": period},
        "clock": {"value": float(seconds)},
        "team": {"id": team},
        "participants": [{"athlete": {"id": str(a)}} for a in athletes],
        "shootout": shootout,
    }


def summary_payload(
    event_id,
    kickoff,
    home,
    away,
    home_roster,
    away_roster,
    events=(),
    status="STATUS_FULL_TIME",
    season=None,
):
    competitors = [
        {"homeAway": "home", "team": {"id": home}},
        {"homeAway": "away", "team": {"id": away}},
    ]
    header_season = {} if season is None else {"season": {"year": season, "type": 1}}
    return {
        "header": {
            "id": str(event_id),
            **header_season,
            "competitions": [
                {
                    "date": kickoff,
                    "competitors": competitors,
                    "status": {"type": {"name": status, "completed": True, "state": "post"}},
                }
            ],
        },
        "rosters": [
            {"homeAway": "home", "team": {"id": home}, "roster": home_roster},
            {"homeAway": "away", "team": {"id": away}, "roster": away_roster},
        ],
        "keyEvents": list(events),
    }


def palmer_match():
    """Chelsea 0-2 Spurs, GW8 2026/27, trimmed: Palmer yellow 71', red (second yellow) 80'."""
    home = [
        player(296395, "Cole Palmer", True),
        player(353951, "Jorrel Hato", True, sub_out=True),
        player(362486, "Marco Palestra", False, sub_in=True),
        player(1, "Unused Sub", False),
    ]
    away = [
        player(194450, "James Maddison", True, sub_out=True),
        player(266696, "Conor Gallagher", False, sub_in=True),
    ]
    events = [
        key_event("Substitution", 3015, "367", 266696, 194450),
        key_event("Substitution", 3867, "363", 362486, 353951),
        key_event("Yellow Card", 4253, "363", 296395),
        key_event("Red Card", 4789, "363", 296395),
        key_event("Yellow Card", 5400, "367", 266696),
    ]
    return summary_payload(401878767, "2026-10-24T16:30Z", "363", "367", home, away, events)


def test_parse_summary_match_row():
    match, _ = parse_summary(palmer_match(), "eng.1")
    assert match == {
        "espn_event_id": 401878767,
        "league": "eng.1",
        "season": 2026,
        "kickoff_time": pd.Timestamp("2026-10-24T16:30Z"),
        "home_espn_id": "363",
        "away_espn_id": "367",
        "status": "STATUS_FULL_TIME",
        "extra_time": False,
    }


def test_parse_summary_minutes_and_cards():
    _, players = parse_summary(palmer_match(), "eng.1")
    rows = {p["espn_athlete_id"]: p for p in players}
    assert len(rows) == 6
    palmer = rows[296395]
    assert (palmer["minutes"], palmer["minute_off"], palmer["yellows"]) == (80, 80, 1)
    assert palmer["red"] and palmer["second_yellow"] and palmer["starter"]
    assert (rows[353951]["minutes"], rows[362486]["minute_on"], rows[362486]["minutes"]) == (
        65,
        65,
        25,
    )
    assert (rows[194450]["minutes"], rows[266696]["minutes"]) == (51, 39)
    assert rows[266696]["yellows"] == 1 and not rows[266696]["red"]
    assert rows[1]["minutes"] == 0 and pd.isna(rows[1]["minute_on"])
    assert rows[296395]["espn_team_id"] == "363" and rows[194450]["espn_team_id"] == "367"


def test_a_straight_red_is_not_a_second_yellow_and_extra_time_runs_to_120():
    home = [player(10, "A", True), player(11, "B", False, sub_in=True)]
    away = [player(20, "C", True, sub_out=True), player(21, "D", False, sub_in=True)]
    events = [
        key_event("Red Card", 1200, "359", 10),
        key_event("Substitution", 6600, "359", 11, 10),  # ignored: 10 already off
        key_event("Substitution", 5000, "1000", 21, 20),
        key_event("Yellow Card", 7300, "1000", 21, shootout=True),
    ]
    payload = summary_payload(
        5, "2016-10-25T18:45Z", "359", "1000", home, away, events, status="STATUS_FINAL_PEN"
    )
    match, players = parse_summary(payload, "eng.league_cup")
    rows = {p["espn_athlete_id"]: p for p in players}
    assert match["extra_time"] and match["season"] == 2016
    assert rows[10]["red"] and not rows[10]["second_yellow"] and rows[10]["minutes"] == 21
    assert rows[21]["minutes"] == 120 - 84 and rows[21]["yellows"] == 0


def test_a_substitution_flag_without_its_event_leaves_minutes_unknown():
    home = [
        player(10, "Off", True, sub_out=True),  # no event takes him off
        player(11, "On", False, sub_in=True),  # no event brings him on
        player(12, "Starter", True),
        player(13, "Swapped off", True, sub_out=True),
        player(14, "Swapped on", False, sub_in=True),
    ]
    events = [key_event("Substitution", 3600, "359", 14, 13)]
    payload = summary_payload(7, "2024-09-17T18:45Z", "359", "1000", home, [], events)
    rows = {p["espn_athlete_id"]: p for p in parse_summary(payload, "eng.1")[1]}
    assert pd.isna(rows[10]["minutes"]) and pd.isna(rows[10]["minute_off"])
    assert rows[10]["minute_on"] == 0
    assert pd.isna(rows[11]["minutes"]) and pd.isna(rows[11]["minute_on"])
    assert [rows[k]["minutes"] for k in (12, 13, 14)] == [90, 61, 29]


def shootout_payload(events, status="STATUS_FINAL_PEN"):
    home = [player(10, "A", True, sub_out=True), player(11, "B", False, sub_in=True)]
    events = [key_event("Substitution", 5000, "359", 11, 10), *events]
    return summary_payload(6, "2024-09-17T18:45Z", "359", "1000", home, [], events, status=status)


def test_a_shoot_out_straight_after_90_minutes_ends_at_90():
    shootout = [key_event("Penalty - Scored", 5500, "359", 11, shootout=True)]
    match, players = parse_summary(shootout_payload(shootout), "eng.league_cup")
    rows = {p["espn_athlete_id"]: p for p in players}
    assert not match["extra_time"]
    assert (rows[11]["minute_off"], rows[11]["minutes"]) == (90, 90 - 84)


def test_a_shoot_out_after_extra_time_ends_at_120():
    for events in (
        [key_event("Yellow Card", 6000, "359", 11)],  # an event in extra time (period 3)
        [key_event("Start Extra Time", 5400, "359", period=2)],
    ):
        match, players = parse_summary(shootout_payload(events), "eng.league_cup")
        rows = {p["espn_athlete_id"]: p for p in players}
        assert match["extra_time"] and rows[11]["minutes"] == 120 - 84


def test_after_extra_time_without_a_shoot_out_ends_at_120():
    match, _ = parse_summary(shootout_payload([], status="STATUS_FINAL_AET"), "eng.fa")
    assert match["extra_time"]


ESPN_OF = dict(
    team_dim_from_config(REPO_CONFIG / "teams.csv")[["team_key", "espn_id"]]
    .dropna()
    .itertuples(index=False)
)


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


ALL = ["fixture"]


def built_world(world):
    """Season 2023 (one player per club, 90' every match) built through fixture."""
    world.add_vaastav_season(2023)
    build(ALL, world.ctx)
    return world.ctx.table("fixture")


def athlete(team_key):
    """The synthetic player of a club as ESPN lists him (players_raw: First<i> Second<i>)."""
    i = TEAM_CODES.index(team_key) + 1
    return player(team_key * 10, f"First{i} Second{i}", True)


def pl_payload(fx, minutes_home=90):
    home, away = ESPN_OF[fx.home_team_key], ESPN_OF[fx.away_team_key]
    events = []
    if minutes_home != 90:  # sent off: off the pitch at `minutes_home`
        events = [key_event("Red Card", (minutes_home - 1) * 60, home, fx.home_team_key * 10)]
    kickoff = fx.kickoff_time.strftime("%Y-%m-%dT%H:%MZ")
    return summary_payload(
        fx.fixture_key,
        kickoff,
        home,
        away,
        [athlete(fx.home_team_key)],
        [athlete(fx.away_team_key)],
        events,
    )


def test_tables_join_pl_fixtures_and_keep_cup_matches_of_pl_clubs(world):
    fixtures = built_world(world)
    first = next(fixtures.itertuples(index=False))
    world.add_espn_summary("eng.1", pl_payload(first))
    ars = ESPN_OF[3]
    cup = summary_payload(
        77, "2023-09-26T18:45Z", ars, "1000", [athlete(3)], [player(5, "Lower", True)]
    )
    world.add_espn_summary("eng.league_cup", cup)
    world.add_espn_summary(
        "eng.league_cup", summary_payload(78, "2023-09-26T18:45Z", "1000", "1001", [], [])
    )
    tables = build_espn_tables(world.ctx)
    match = tables["espn_match"].set_index("espn_event_id")
    assert set(match.index) == {first.fixture_key, 77}
    assert match.loc[first.fixture_key, "fixture_key"] == first.fixture_key
    assert pd.isna(match.loc[77, "fixture_key"]) and match.loc[77, "home_team_key"] == 3
    assert pd.isna(match.loc[77, "away_team_key"])
    assert (match["available_at"] == match["event_time"] + AVAILABLE_AFTER).all()
    players = tables["espn_player_match"]
    assert set(players["espn_athlete_id"]) == {
        first.home_team_key * 10,
        first.away_team_key * 10,
        30,
    }


def test_an_unknown_pl_club_fails_the_build(world):
    built_world(world)
    world.add_espn_summary(
        "eng.1", summary_payload(1, "2023-08-12T14:00Z", "999999", ESPN_OF[3], [], [])
    )
    with pytest.raises(EspnBuildError, match="eng.1 event 1"):
        build_espn_tables(world.ctx)


def test_the_header_season_wins_over_the_kickoff_date():
    # 2019/20 restarted after lockdown: an eng.1 match on 2020-07-04 is still season 2019.
    payload = summary_payload(9, "2020-07-04T11:30Z", "359", "363", [], [], season=2019)
    assert parse_summary(payload, "eng.1")[0]["season"] == 2019
    del payload["header"]["season"]
    assert parse_summary(payload, "eng.1")[0]["season"] == 2020


def test_a_match_played_after_june_joins_its_seasons_fixture(world):
    world.add_vaastav_season(2019)
    build(ALL, world.ctx)
    first = next(world.ctx.table("fixture").itertuples(index=False))
    payload = pl_payload(first)
    payload["header"]["competitions"][0]["date"] = "2020-07-04T11:30Z"
    payload["header"]["season"] = {"year": 2019, "type": 1}
    world.add_espn_summary("eng.1", payload)
    match = build_espn_tables(world.ctx)["espn_match"]
    assert match["fixture_key"].tolist() == [first.fixture_key]
    assert match["season"].tolist() == [2019]


def test_the_newest_summary_of_an_event_wins(world):
    fixtures = built_world(world)
    first = next(fixtures.itertuples(index=False))
    world.add_espn_summary("eng.1", pl_payload(first, minutes_home=30), at=RUN)
    world.add_espn_summary("eng.1", pl_payload(first), at=datetime(2026, 10, 7, tzinfo=UTC))
    players = build_espn_tables(world.ctx)["espn_player_match"]
    assert players["minutes"].tolist() == [90, 90]


def test_schedule_snapshot_lists_scheduled_matches_of_pl_clubs(world):
    built_world(world)
    at = datetime(2026, 10, 27, 3, tzinfo=UTC)
    events = [
        {
            "id": "501",
            "date": "2026-10-28T19:45Z",
            "status": {"type": {"name": "STATUS_SCHEDULED", "completed": False, "state": "pre"}},
            "competitions": [
                {
                    "competitors": [
                        {"homeAway": "home", "team": {"id": ESPN_OF[14]}},
                        {"homeAway": "away", "team": {"id": ESPN_OF[8]}},
                    ]
                }
            ],
        },
        {
            "id": "502",
            "date": "2026-10-28T19:45Z",
            "status": {"type": {"name": "STATUS_SCHEDULED", "completed": False, "state": "pre"}},
            "competitions": [
                {
                    "competitors": [
                        {"homeAway": "home", "team": {"id": "1000"}},
                        {"homeAway": "away", "team": {"id": "1001"}},
                    ]
                }
            ],
        },
    ]
    world.add_espn_scoreboard("eng.league_cup", "202610", events, at=at)
    snap = build_espn_tables(world.ctx)["espn_schedule_snapshot"]
    assert snap["espn_event_id"].tolist() == [501]
    row = snap.iloc[0]
    assert (row["home_team_key"], row["away_team_key"], row["league"]) == (14, 8, "eng.league_cup")
    assert row["snapshot_at"] == row["available_at"] == pd.Timestamp(at)
    assert row["season"] == 2026


def test_schedule_snapshot_takes_the_events_season(world):
    built_world(world)
    event = {
        "id": "601",
        "date": "2020-07-04T11:30Z",
        "season": {"year": 2019, "type": 1},
        "status": {"type": {"name": "STATUS_SCHEDULED", "completed": False, "state": "pre"}},
        "competitions": [
            {
                "competitors": [
                    {"homeAway": "home", "team": {"id": ESPN_OF[14]}},
                    {"homeAway": "away", "team": {"id": ESPN_OF[8]}},
                ]
            }
        ],
    }
    world.add_espn_scoreboard("eng.1", "202007", [event], at=datetime(2020, 7, 1, tzinfo=UTC))
    snap = build_espn_tables(world.ctx)["espn_schedule_snapshot"]
    assert snap["season"].tolist() == [2019]


# --- espn_map and the build checks ---------------------------------------------------------


def pl_world(world, *, payload=None, skip=()):
    """Season 2023 built through player_match / player_season (Understat files included, so
    the Understat builder passes) with an ESPN eng.1 summary for every fixture but the
    `skip` indices (`payload(fx)`, default: one player a side, 90')."""
    raw = world.add_vaastav_season(2023)
    add_understat_season(world, 2023, raw, "2023-24")
    world.write_player_snapshot()
    build(UNDERSTAT_BUILD, world.ctx)
    fixtures = world.ctx.table("fixture")
    for i, fx in enumerate(fixtures.itertuples(index=False)):
        if i not in skip:
            world.add_espn_summary("eng.1", (payload or pl_payload)(fx))
    return fixtures


def test_build_espn_maps_every_player_and_writes_four_tables(world):
    pl_world(world)
    build(["espn"], world.ctx)
    mapping = world.ctx.table("espn_map")
    pm = world.ctx.table("player_match")
    assert len(mapping) == pm["player_key"].nunique() == 20
    assert (mapping["method"] == "auto").all()
    # Club i's player (team_key TEAM_CODES[i-1]) is player_key 100000 + i.
    expected = {code * 10: 100000 + i for i, code in enumerate(TEAM_CODES, start=1)}
    assert dict(zip(mapping["espn_athlete_id"], mapping["player_key"], strict=True)) == expected
    assert (mapping["shared"] == 38).all() and (mapping["coverage"] == 1).all()
    for name in ("espn_match", "espn_player_match", "espn_schedule_snapshot"):
        assert world.ctx.table_path(name).exists()
    assert len(world.ctx.table("espn_match")) == 380


def test_a_missing_pl_fixture_fails_the_build(world):
    pl_world(world, skip={0})
    with pytest.raises(EspnBuildError, match="1 finished PL fixtures have no ESPN match"):
        build(["espn"], world.ctx)
    assert not world.ctx.table_path("espn_map").exists()


def test_an_fpl_player_missing_from_espn_fails_the_build(world):
    gone = TEAM_CODES[0]

    def payload(fx):
        out = pl_payload(fx)
        for roster in out["rosters"]:
            roster["roster"] = [p for p in roster["roster"] if p["athlete"]["id"] != str(gone * 10)]
        return out

    pl_world(world, payload=payload)
    with pytest.raises(
        EspnBuildError, match=r"1 FPL players with minutes are not mapped: \[100001\]"
    ):
        build(["espn"], world.ctx)


def test_an_override_exempts_an_fpl_player_without_espn_rows(world, tmp_path):
    config = tmp_path / "config"
    config.mkdir()
    (config / "teams.csv").write_text((REPO_CONFIG / "teams.csv").read_text("utf-8"), "utf-8")
    (config / "overrides.csv").write_text(
        "source,source_id,season,player_key,note\nespn,,,100001,not on ESPN\n", "utf-8"
    )
    world.ctx.config_dir = config
    gone = TEAM_CODES[0]

    def payload(fx):
        out = pl_payload(fx)
        for roster in out["rosters"]:
            roster["roster"] = [p for p in roster["roster"] if p["athlete"]["id"] != str(gone * 10)]
        return out

    pl_world(world, payload=payload)
    build(["espn"], world.ctx)
    assert 100001 not in set(world.ctx.table("espn_map")["player_key"])


def substituted(minute, times):
    """pl_payload with club 1's player replaced at `minute` in its first `times` matches."""
    club = TEAM_CODES[0]
    done = []

    def payload(fx):
        out = pl_payload(fx)
        if club in (fx.home_team_key, fx.away_team_key) and len(done) < times:
            side = 0 if fx.home_team_key == club else 1
            team = out["rosters"][side]["team"]["id"]
            out["keyEvents"].append(
                key_event("Substitution", (minute - 1) * 60, team, 999, club * 10)
            )
            done.append(fx.fixture_key)
        return out

    return payload


def test_minutes_that_disagree_too_often_fail_the_build(world):
    # 34 of 38 agree (89%): still an auto pair (overlap >= 0.8, shared >= 3), below 90%.
    pl_world(world, payload=substituted(30, times=4))
    with pytest.raises(EspnBuildError, match=r"1 auto pairs agree on minutes in < 90% of fixtures"):
        build(["espn"], world.ctx)


def test_unknown_minutes_are_null_and_skipped_by_the_checks(world, caplog):
    club = TEAM_CODES[0]
    done = []

    def payload(fx):  # club 1's player flagged as a substitute without an event, 4 times
        out = pl_payload(fx)
        if club in (fx.home_team_key, fx.away_team_key) and len(done) < 4:
            side = 0 if fx.home_team_key == club else 1
            out["rosters"][side]["roster"] = [player(club * 10, "First1 Second1", False, True)]
            done.append(fx.fixture_key)
        return out

    pl_world(world, payload=payload)
    with caplog.at_level("INFO", logger="fplopt.build.espn"):
        build(["espn"], world.ctx)
    minutes = world.ctx.table("espn_player_match")["minutes"]
    assert minutes.dtype == "Int64" and minutes.isna().sum() == 4
    assert "eng.1: 4 row(s)" in caplog.text
    mapping = world.ctx.table("espn_map")
    assert mapping.loc[mapping["espn_athlete_id"] == club * 10, "player_key"].tolist() == [100001]


def test_minutes_that_mostly_agree_pass(world):
    pl_world(world, payload=substituted(30, times=3))  # 35 of 38 (92%)
    build(["espn"], world.ctx)
    assert len(world.ctx.table("espn_map")) == 20


def test_an_espn_red_card_fpl_does_not_have_fails_the_build(world):
    fixtures = pl_world(world, skip={0})
    world.add_espn_summary("eng.1", pl_payload(fixtures.iloc[0], minutes_home=88))
    with pytest.raises(EspnBuildError, match="red cards disagree in 1 of 0 FPL reds"):
        build(["espn"], world.ctx)


def test_no_espn_data_builds_empty_tables(world):
    built_world(world)
    build(["espn"], world.ctx)
    for name in ("espn_map", "espn_match", "espn_player_match", "espn_schedule_snapshot"):
        assert world.ctx.table(name).empty, name
    mapping = pd.read_parquet(world.ctx.table_path("espn_map"))
    MAP_SCHEMA.validate(mapping)
    assert list(mapping.columns) == list(MAP_SCHEMA.columns)
