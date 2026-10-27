import pandas as pd

from fplopt.features.bans import (
    BAN_CERTAIN,
    BAN_LIKELY,
    ban_events,
    club_timelines,
    serve_bans,
)

T = pd.Timestamp


def league(rows):
    cols = [
        "player_key",
        "season",
        "gw_index",
        "fixture_key",
        "team_key",
        "kickoff_time",
        "yellow_cards",
        "red_cards",
    ]
    return pd.DataFrame(rows, columns=cols)


def fixtures(team_key, start, n, first_key=100):
    return pd.DataFrame(
        {
            "team_key": team_key,
            "fixture_key": range(first_key, first_key + n),
            "kickoff_time": [T(start) + pd.Timedelta(weeks=i) for i in range(n)],
        }
    )


NO_REDS = pd.DataFrame(columns=["player_key", "fixture_key", "second_yellow"])
NO_CUP = pd.DataFrame(columns=["player_key", "team_key", "kickoff_time", "second_yellow"])
NO_CUP_MATCHES = pd.DataFrame(columns=["team_key", "kickoff_time"])


def kinds(served):
    return dict(zip(served["fixture_key"], served["ban_kind"], strict=True))


def test_palmer_second_yellow_is_served_in_the_cup_tie():
    # Chelsea (8): PL fixture 100 on 24 Oct (red, second yellow), EFL Cup 28 Oct, PL 101 31 Oct.
    rows = league([[7, 2026, 8, 100, 8, T("2026-10-24T16:30Z"), 1, 1]])
    reds = pd.DataFrame({"player_key": [7], "fixture_key": [100], "second_yellow": [True]})
    events = ban_events(rows, reds, NO_CUP)
    lines = club_timelines(
        fixtures(8, "2026-10-24T16:30Z", 3),
        pd.DataFrame({"team_key": [8], "kickoff_time": [T("2026-10-28T19:45Z")]}),
    )
    assert kinds(serve_bans(events, lines)) == {}


def test_without_the_cup_tie_the_next_pl_fixture_is_banned():
    rows = league([[7, 2026, 8, 100, 8, T("2026-10-24T16:30Z"), 1, 1]])
    reds = pd.DataFrame({"player_key": [7], "fixture_key": [100], "second_yellow": [True]})
    lines = club_timelines(fixtures(8, "2026-10-24T16:30Z", 3), NO_CUP_MATCHES)
    assert kinds(serve_bans(ban_events(rows, reds, NO_CUP), lines)) == {101: BAN_CERTAIN}


def test_a_straight_red_is_one_certain_and_two_likely():
    rows = league([[7, 2026, 8, 100, 8, T("2026-10-24T16:30Z"), 0, 1]])
    reds = pd.DataFrame({"player_key": [7], "fixture_key": [100], "second_yellow": [False]})
    lines = club_timelines(fixtures(8, "2026-10-24T16:30Z", 5), NO_CUP_MATCHES)
    served = kinds(serve_bans(ban_events(rows, reds, NO_CUP), lines))
    assert served == {101: BAN_CERTAIN, 102: BAN_LIKELY, 103: BAN_LIKELY}


def test_a_red_without_espn_data_is_one_match():
    rows = league([[7, 2026, 8, 100, 8, T("2026-10-24T16:30Z"), 0, 1]])
    lines = club_timelines(fixtures(8, "2026-10-24T16:30Z", 3), NO_CUP_MATCHES)
    assert kinds(serve_bans(ban_events(rows, NO_REDS, NO_CUP), lines)) == {101: BAN_CERTAIN}


def test_a_missing_second_yellow_flag_counts_as_one_match():
    rows = league([[7, 2026, 8, 100, 8, T("2026-10-24T16:30Z"), 0, 1]])
    reds = pd.DataFrame({"player_key": [7], "fixture_key": [100], "second_yellow": [pd.NA]})
    lines = club_timelines(fixtures(8, "2026-10-24T16:30Z", 3), NO_CUP_MATCHES)
    assert kinds(serve_bans(ban_events(rows, reds, NO_CUP), lines)) == {101: BAN_CERTAIN}


def test_a_cup_red_bans_the_next_pl_fixture():
    rows = league([])
    cup = pd.DataFrame(
        {
            "player_key": [7],
            "team_key": [8],
            "kickoff_time": [T("2026-10-28T19:45Z")],
            "second_yellow": [True],
        }
    )
    lines = club_timelines(
        fixtures(8, "2026-10-24T16:30Z", 3),
        pd.DataFrame({"team_key": [8], "kickoff_time": [T("2026-10-28T19:45Z")]}),
    )
    assert kinds(serve_bans(ban_events(rows, NO_REDS, cup), lines)) == {101: BAN_CERTAIN}


def test_pl_yellow_bans_ignore_cup_ties():
    # Fifth yellow in gw_index 10 -> one PL match, served in PL fixture 105 despite the cup tie.
    rows = league(
        [
            [
                7,
                2026,
                gw,
                100 + gw - 6,
                8,
                T("2026-10-03T14:00Z") + pd.Timedelta(weeks=gw - 6),
                1,
                0,
            ]
            for gw in range(6, 11)
        ]
    )
    lines = club_timelines(
        fixtures(8, "2026-10-03T14:00Z", 7),
        pd.DataFrame({"team_key": [8], "kickoff_time": [T("2026-11-03T19:45Z")]}),
    )
    assert kinds(serve_bans(ban_events(rows, NO_REDS, NO_CUP), lines)) == {105: BAN_CERTAIN}


def test_overlapping_bans_keep_the_stronger_kind():
    rows = league(
        [
            [7, 2026, 8, 100, 8, T("2026-10-24T16:30Z"), 0, 1],
            [7, 2026, 9, 101, 8, T("2026-10-31T16:30Z"), 0, 1],
        ]
    )
    reds = pd.DataFrame(
        {"player_key": [7, 7], "fixture_key": [100, 101], "second_yellow": [False, True]}
    )
    lines = club_timelines(fixtures(8, "2026-10-24T16:30Z", 5), NO_CUP_MATCHES)
    served = kinds(serve_bans(ban_events(rows, reds, NO_CUP), lines))
    assert served == {101: BAN_CERTAIN, 102: BAN_CERTAIN, 103: BAN_LIKELY}


def test_empty_inputs_give_typed_empty_frames():
    events = ban_events(league([]), NO_REDS, NO_CUP)
    lines = club_timelines(fixtures(8, "2026-10-24T16:30Z", 0), NO_CUP_MATCHES)
    assert events.empty and lines.empty
    served = serve_bans(events, lines)
    assert served.empty
    assert list(served.columns) == ["player_key", "fixture_key", "ban_kind"]
    assert (served.dtypes == "int64").all()
