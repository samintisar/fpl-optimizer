from datetime import UTC, datetime

import pandas as pd
import pytest
from synthetic_raw import REPO_CONFIG, RUN, TEAM_CODES, World

from fplopt.build import build
from fplopt.build.espn import AVAILABLE_AFTER, EspnBuildError, build_espn_tables, parse_summary
from fplopt.build.teams import team_dim_from_config


def player(athlete_id, name, starter, sub_in=False, sub_out=False):
    return {
        "starter": starter,
        "subbedIn": sub_in,
        "subbedOut": sub_out,
        "athlete": {"id": str(athlete_id), "displayName": name},
    }


def key_event(text, seconds, team, *athletes, shootout=False):
    return {
        "type": {"text": text},
        "clock": {"value": float(seconds)},
        "team": {"id": team},
        "participants": [{"athlete": {"id": str(a)}} for a in athletes],
        "shootout": shootout,
    }


def summary_payload(
    event_id, kickoff, home, away, home_roster, away_roster, events=(), status="STATUS_FULL_TIME"
):
    competitors = [
        {"homeAway": "home", "team": {"id": home}},
        {"homeAway": "away", "team": {"id": away}},
    ]
    return {
        "header": {
            "id": str(event_id),
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
