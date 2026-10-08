import json
from datetime import UTC, datetime

import pandas as pd
import pytest
from synthetic_raw import REPO_CONFIG, TEAM_CODES, World, football_data

from fplopt.build import build
from fplopt.build.odds import (
    OddsValidationError,
    archived_live,
    check_overround,
    football_data_first_seen,
    football_data_odds,
    odds_api_rows,
    prematch_available_at,
)
from fplopt.build.teams import TeamResolver, team_dim_from_config
from fplopt.features.store import DataStore

CONFIG = pd.read_csv(REPO_CONFIG / "teams.csv")
ODDS_NAMES = dict(zip(CONFIG["team_key"], CONFIG["odds_api"], strict=True))
RESOLVER = TeamResolver(team_dim_from_config(REPO_CONFIG / "teams.csv"))
LATER = datetime(2026, 10, 6, 5, tzinfo=UTC)  # a one-time backfill after the season


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def utc(text):
    return pd.Timestamp(text, tz="UTC").as_unit("us")


def fd_row(season, **odds):
    """One football-data row (Arsenal v Chelsea) with the given odds cells."""
    return pd.DataFrame(
        [{"season": season, "fd_date": None, "home_team_key": 3, "away_team_key": 8, **odds}]
    )


def as_set(df):
    return {
        (r.bookmaker, r.market, r.outcome, None if pd.isna(r.line) else r.line, r.is_closing)
        for r in df.itertuples()
    }


# --- football-data allowlist ------------------------------------------------------------


def test_era_a_columns_and_coral_ignored():
    row = fd_row(
        2017,
        BbAvH=2.0,
        BbAvD=3.4,
        BbAvA=4.0,
        **{"BbAv>2.5": 1.9, "BbAv<2.5": 1.95},
        BbAHh=-0.5,
        BbAvAHH=1.98,
        BbAvAHA=1.92,
        PSH=2.05,
        PSD=3.5,
        PSA=4.1,
        PSCH=2.1,
        PSCD=3.4,
        PSCA=4.0,
        CLH=9.0,  # Coral, not closing
        CLD=9.0,
        CLA=9.0,
        AvgH=7.0,  # era B column name: not read in era A
    )
    out = football_data_odds(row)
    assert as_set(out) == {
        ("avg", "h2h", "home", None, False),
        ("avg", "h2h", "draw", None, False),
        ("avg", "h2h", "away", None, False),
        ("avg", "totals", "over", 2.5, False),
        ("avg", "totals", "under", 2.5, False),
        ("avg", "ah", "home", -0.5, False),
        ("avg", "ah", "away", -0.5, False),
        ("pinnacle", "h2h", "home", None, False),
        ("pinnacle", "h2h", "draw", None, False),
        ("pinnacle", "h2h", "away", None, False),
        ("pinnacle", "h2h", "home", None, True),
        ("pinnacle", "h2h", "draw", None, True),
        ("pinnacle", "h2h", "away", None, True),
    }
    assert 9.0 not in set(out["price"]) and 7.0 not in set(out["price"])
    closing_home = out[(out["bookmaker"] == "pinnacle") & out["is_closing"]]
    assert closing_home.set_index("outcome").loc["home", "price"] == 2.1


def test_era_b_closing_ah_pairs_with_ahch_and_traps_ignored():
    row = fd_row(
        2024,
        AvgH=2.0,
        AvgD=3.4,
        AvgA=4.0,
        AHh=-0.5,
        AvgAHH=1.98,
        AvgAHA=1.92,
        AHCh=-0.75,
        AvgCAHH=2.02,
        AvgCAHA=1.88,
        AvgCH=2.1,
        AvgCD=3.3,
        AvgCA=3.9,
        **{"AvgC>2.5": 1.8, "AvgC<2.5": 2.05},
        BFEH=2.1,
        BFED=3.6,
        BFEA=None,  # null cell skipped
        BFH=8.0,  # Betfair sportsbook
        BFDH=8.0,  # Betfred
        BFCH=8.0,  # Betfair sportsbook closing
        BbAvH=8.0,  # era A column name: not read in era B
    )
    out = football_data_odds(row)
    ah = out[out["market"] == "ah"].set_index(["is_closing", "outcome"])
    assert ah.loc[(False, "home"), ["line", "price"]].tolist() == [-0.5, 1.98]
    assert ah.loc[(True, "home"), ["line", "price"]].tolist() == [-0.75, 2.02]
    assert ah.loc[(True, "away"), ["line", "price"]].tolist() == [-0.75, 1.88]
    closing_totals = out[(out["market"] == "totals") & out["is_closing"]]
    assert set(closing_totals["outcome"]) == {"over", "under"}
    assert (closing_totals["line"] == 2.5).all()
    betfair = out[out["bookmaker"] == "betfair_ex"]
    assert set(betfair["outcome"]) == {"home", "draw"}
    assert 8.0 not in set(out["price"])


def test_prematch_available_at_collection_days():
    kickoffs = pd.Series(
        [
            utc("2024-08-17T11:30"),  # Sat 12:30 BST -> Fri 15:00 BST
            utc("2024-09-25T18:45"),  # Wed 19:45 BST -> Tue 15:00 BST
            utc("2024-12-30T20:00"),  # Mon 20:00 GMT -> Fri 15:00 GMT (previous week)
            utc("2025-01-14T12:00"),  # Tue 12:00 GMT -> kickoff - 1h (before Tue 15:00)
            utc("2024-08-16T19:00"),  # Fri 20:00 BST -> same Friday 15:00 BST
        ]
    )
    assert prematch_available_at(kickoffs).tolist() == [
        utc("2024-08-16T14:00"),
        utc("2024-09-24T14:00"),
        utc("2024-12-27T15:00"),
        utc("2025-01-14T11:00"),
        utc("2024-08-16T14:00"),
    ]


# --- The Odds API -----------------------------------------------------------------------


def fixture_rows():
    return pd.DataFrame(
        {
            "fixture_key": [2026070, 2026071],
            "season": [2026, 2026],
            "home_team_key": [3, 14],
            "away_team_key": [2, 43],
            "kickoff_time": [utc("2026-10-10T11:30"), utc("2026-10-11T15:30")],
            "event_time": [utc("2026-10-10T11:30"), utc("2026-10-11T15:30")],
        }
    )


def odds_event(home, away, commence, bookmakers):
    return {
        "id": f"{home}-{away}",
        "commence_time": commence,
        "home_team": home,
        "away_team": away,
        "bookmakers": bookmakers,
    }


def test_odds_api_rows_keep_h2h_and_totals_2_5_only():
    book = {
        "key": "williamhill",
        "title": "William Hill",
        "last_update": "2026-10-06T03:32:57Z",
        "markets": [
            {
                "key": "h2h",
                "outcomes": [
                    {"name": "Arsenal", "price": 1.37},
                    {"name": "Leeds United", "price": 8.0},
                    {"name": "Draw", "price": 5.0},
                ],
            },
            {
                "key": "totals",
                "outcomes": [
                    {"name": "Over", "price": 1.7, "point": 2.5},
                    {"name": "Under", "price": 2.08, "point": 2.5},
                    {"name": "Over", "price": 2.9, "point": 3.5},
                    {"name": "Under", "price": 1.4, "point": 3.5},
                ],
            },
            {
                "key": "h2h_lay",
                "outcomes": [
                    {"name": "Arsenal", "price": 1.42},
                    {"name": "Leeds United", "price": 9.4},
                    {"name": "Draw", "price": 5.5},
                ],
            },
        ],
    }
    events = [
        odds_event("Arsenal", "Leeds United", "2026-10-10T11:30:00Z", [book]),
        # not in the fixture table -> skipped
        odds_event("Chelsea", "Everton", "2026-10-10T14:00:00Z", [book]),
        # already kicked off at snapshot time (in-play prices) -> skipped
        odds_event("Liverpool", "Manchester City", "2026-10-05T15:30:00Z", [book]),
    ]
    taken_at = utc("2026-10-06T03:33:44")
    out, skipped = odds_api_rows(events, taken_at, fixture_rows(), RESOLVER)
    assert skipped == {"no fixture": 1, "started": 1}
    assert (out["fixture_key"] == 2026070).all()
    assert as_set(out) == {
        ("williamhill", "h2h", "home", None, False),
        ("williamhill", "h2h", "draw", None, False),
        ("williamhill", "h2h", "away", None, False),
        ("williamhill", "totals", "over", 2.5, False),
        ("williamhill", "totals", "under", 2.5, False),
    }
    h2h = out[out["market"] == "h2h"].set_index("outcome")["price"].to_dict()
    assert h2h == {"home": 1.37, "draw": 5.0, "away": 8.0}
    assert (out["snapshot_at"] == taken_at).all()
    assert (out["available_at"] == taken_at).all()
    assert (out["source"] == "odds-api").all()


# --- validation -------------------------------------------------------------------------


def h2h_frame(sums):
    rows = []
    for i, total in enumerate(sums):
        for outcome in ("home", "draw", "away"):
            rows.append(
                {
                    "fixture_key": 2026001 + i,
                    "source": "odds-api",
                    "bookmaker": "b",
                    "market": "h2h",
                    "outcome": outcome,
                    "is_closing": False,
                    "snapshot_at": utc("2026-10-06"),
                    "price": 3 / total,
                }
            )
    return pd.DataFrame(rows)


def test_overround_outliers_fail_above_one_percent():
    check_overround(h2h_frame([1.05] * 199 + [1.5]))  # 0.5% outliers: reported only
    with pytest.raises(OddsValidationError, match="overround"):
        check_overround(h2h_frame([1.05] * 98 + [1.5, 0.9]))


# --- end to end -------------------------------------------------------------------------


def with_odds(fd, season):
    n = len(fd)
    if season < 2019:
        cols = {"BbAvH": 2.0, "BbAvD": 3.5, "BbAvA": 4.0, "BbAHh": -0.25, "BbAvAHH": 1.9}
        cols |= {"BbAvAHA": 1.95, "PSCH": 2.1, "PSCD": 3.4, "PSCA": 3.9}
    else:
        cols = {"AvgH": 2.0, "AvgD": 3.5, "AvgA": 4.0, "AvgCH": 2.1, "AvgCD": 3.4}
        cols |= {"AvgCA": 3.9, "Avg>2.5": 1.9, "Avg<2.5": 1.9, "CLH": 9.0}
    return fd.assign(**{c: [v] * n for c, v in cols.items()})


def test_build_odds_snapshot_end_to_end(world):
    fx16 = world.add_vaastav_season(2016, fixtures_csv=False, teams_csv=False)
    fx23 = world.add_vaastav_season(2023)
    later = datetime(2026, 10, 6, 5, tzinfo=UTC)
    world.football_data(2016, with_odds(football_data(fx16, 2016), 2016), at=later)
    world.football_data(2023, with_odds(football_data(fx23, 2023), 2023), at=later)
    current = world.add_current_season()
    named = current["team_h"].map(lambda i: isinstance(ODDS_NAMES[TEAM_CODES[i - 1]], str))
    named &= current["team_a"].map(lambda i: isinstance(ODDS_NAMES[TEAM_CODES[i - 1]], str))
    upcoming = current[(current["event"] == 3) & named].iloc[0]
    home, away = TEAM_CODES[upcoming["team_h"] - 1], TEAM_CODES[upcoming["team_a"] - 1]
    book = {
        "key": "betfair_ex_uk",
        "markets": [
            {
                "key": "h2h",
                "outcomes": [
                    {"name": ODDS_NAMES[home], "price": 2.0},
                    {"name": ODDS_NAMES[away], "price": 4.0},
                    {"name": "Draw", "price": 3.6},
                ],
            }
        ],
    }
    events = [odds_event(ODDS_NAMES[home], ODDS_NAMES[away], upcoming["kickoff_time"], [book])]
    world.store.write(
        "odds", "soccer_epl", json.dumps(events).encode(), datetime(2026, 8, 10, tzinfo=UTC)
    )
    build(["fixture", "odds_snapshot"], world.ctx)

    odds = world.ctx.table("odds_snapshot")
    fixture = world.ctx.table("fixture").set_index("fixture_key")
    counts = odds.groupby(["season", "source", "is_closing"]).size().to_dict()
    assert counts == {
        (2016, "football-data", False): 380 * 5,
        (2016, "football-data", True): 380 * 3,
        (2023, "football-data", False): 380 * 5,
        (2023, "football-data", True): 380 * 3,
        (2026, "odds-api", False): 3,
    }
    kickoff = odds["fixture_key"].map(fixture["kickoff_time"])
    assert (odds["event_time"] == kickoff).all()
    closing = odds[odds["is_closing"]]
    assert (closing["available_at"] == closing["event_time"]).all()
    pre = odds[~odds["is_closing"]]
    assert (pre["available_at"] < pre["event_time"]).all()
    # synthetic kickoffs are Saturdays -> collected Friday 15:00 UK
    fd_pre = pre[pre["source"] == "football-data"]
    local = fd_pre["available_at"].dt.tz_convert("Europe/London")
    assert (fd_pre["event_time"].dt.weekday == 5).all()
    assert (local.dt.weekday == 4).all() and (local.dt.strftime("%H:%M") == "15:00").all()
    assert (odds["snapshot_at"] == odds["available_at"]).all()
    assert 9.0 not in set(odds["price"])
    api = odds[odds["source"] == "odds-api"]
    assert set(api["bookmaker"]) == {"betfair_ex_uk"}
    assert (api["fixture_key"] == 2026000 + upcoming["id"]).all()


def test_first_seen_is_the_first_file_carrying_the_price():
    gw1 = fd_row(2026, AvgH=2.0, AvgD=3.5, AvgA=4.0)
    gw1_moved = fd_row(2026, AvgH=2.2, AvgD=3.5, AvgA=4.0)  # a revised home price
    gw2 = fd_row(2026, AvgH=1.5, AvgD=4.0, AvgA=6.0).assign(home_team_key=8, away_team_key=3)
    files = [
        (utc("2026-08-03T02:30"), gw1),
        (utc("2026-08-10T02:30"), pd.concat([gw1_moved, gw2], ignore_index=True)),
    ]
    seen = football_data_first_seen(files)
    first = {(r.home_team_key, r.outcome, r.price): r.first_seen for r in seen.itertuples()}
    assert first == {
        (3, "home", 2.0): utc("2026-08-03T02:30"),
        (3, "draw", 3.5): utc("2026-08-03T02:30"),
        (3, "away", 4.0): utc("2026-08-03T02:30"),
        (3, "home", 2.2): utc("2026-08-10T02:30"),
        (8, "home", 1.5): utc("2026-08-10T02:30"),
        (8, "draw", 4.0): utc("2026-08-10T02:30"),
        (8, "away", 6.0): utc("2026-08-10T02:30"),
    }
    assert football_data_first_seen([]).empty


def test_archived_live_means_first_file_before_the_last_kickoff():
    kickoffs = pd.Series([utc("2026-08-01T14:00"), utc("2027-05-23T15:00"), pd.NaT])
    assert archived_live(utc("2026-10-06T03:26"), kickoffs)
    assert not archived_live(utc("2027-06-01T02:30"), kickoffs)
    assert archived_live(utc("2026-10-06T03:26"), pd.Series([pd.NaT]))


def test_live_season_rows_available_from_first_archived_file(world):
    """2026/27 is re-fetched daily and E0.csv lists played matches only: each price is
    available from the first file carrying it, never the assumed Friday collection."""
    fx23 = world.add_vaastav_season(2023)
    world.football_data(2023, with_odds(football_data(fx23, 2023), 2023), at=LATER)
    current = world.add_current_season(fd_rows=0)  # file at RUN, no odds columns
    played = current[current["event"].isin([1, 2])].astype({"event": "int64"})
    gw1 = with_odds(football_data(played[played["event"] == 1], 2026), 2026)
    both = with_odds(football_data(played, 2026), 2026)
    moved = (both["HomeTeam"] == gw1["HomeTeam"].iloc[0]) & (
        both["AwayTeam"] == gw1["AwayTeam"].iloc[0]
    )
    both.loc[moved, "AvgH"] = 1.9  # revised after the first archive
    after_gw1 = datetime(2026, 8, 3, 2, 30, tzinfo=UTC)
    after_gw2 = datetime(2026, 8, 10, 2, 30, tzinfo=UTC)
    world.football_data(2026, gw1, at=after_gw1)
    world.football_data(2026, both, at=after_gw2)
    world.football_data(2026, both, at=datetime(2026, 10, 7, 1, 30, tzinfo=UTC))  # newest
    build(["fixture", "odds_snapshot"], world.ctx)

    odds = world.ctx.table("odds_snapshot")
    fixture = world.ctx.table("fixture").set_index("fixture_key")
    live = odds[(odds["season"] == 2026) & (odds["source"] == "football-data")]
    assert len(live) == 20 * 8
    gw = live["fixture_key"].map(fixture["gw"])
    # the revised home price delays its whole market (h2h avg, pre-match), not the others
    revised = live["fixture_key"].eq(2026000 + played["id"].iloc[0]) & live["market"].eq("h2h")
    revised &= ~live["is_closing"]
    expected = pd.Series(pd.Timestamp(after_gw2).as_unit("us"), index=live.index)
    expected = expected.where((gw != 1) | revised, pd.Timestamp(after_gw1).as_unit("us"))
    assert (live["available_at"] == expected).all()
    assert (live["snapshot_at"] == live["available_at"]).all()
    assert (live["available_at"] > live["event_time"]).all()  # all after kickoff
    assert 1.9 in set(live.loc[revised, "price"])
    # the historical season keeps the assumed collection time and kickoff
    old = odds[odds["season"] == 2023]
    assert (old["available_at"] == old["event_time"]).eq(old["is_closing"]).all()
    pre = old[~old["is_closing"]]
    assert (pre["available_at"] == prematch_available_at(pre["event_time"])).all()
    # a GW2 deadline (after the assumed Friday collection) sees no GW2 football-data odds
    first_gw2 = played.loc[played["event"] == 2, "kickoff_time"].min()
    view = DataStore(tables={"odds_snapshot": odds}).as_of(
        pd.Timestamp(first_gw2) - pd.Timedelta(minutes=90)
    )
    visible = view.latest("odds_snapshot", by=["fixture_key", "source"], columns=["price"])
    visible = visible[visible["fixture_key"] // 1000 == 2026]
    assert set(visible["fixture_key"].map(fixture["gw"])) == {1}


def test_build_fails_on_football_data_date_mismatch(world):
    fx23 = world.add_vaastav_season(2023)
    build(["fixture"], world.ctx)
    fd = with_odds(football_data(fx23, 2023), 2023)
    fd.loc[0, "Date"] = "01/01/2024"  # date no longer matches the fixture
    world.football_data(2023, fd, at=datetime(2026, 10, 6, 5, tzinfo=UTC))
    with pytest.raises(ValueError, match="date"):
        build(["odds_snapshot"], world.ctx)
