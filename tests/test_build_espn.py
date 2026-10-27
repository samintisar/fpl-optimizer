import pandas as pd

from fplopt.build.espn import parse_summary


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
