import json
import lzma
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from fplopt.build import BUILDERS, ORDER, build
from fplopt.build.common import BuildContext
from fplopt.build.teams import (
    TeamCoverageError,
    TeamResolver,
    read_teams_config,
    team_dim_from_config,
)
from fplopt.ingest.raw_store import RawStore, gzip_bytes

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config"
RUN = datetime(2026, 10, 6, 3, 26, 9, tzinfo=UTC)

TEAMS_CSV = """team_key,short_name,fpl_names,football_data,understat,odds_api
3,ARS,Arsenal,Arsenal,Arsenal,Arsenal
17,NFO,Nott'm Forest,Nott'm Forest,Nottingham Forest,Nottingham Forest
88,HUL,Hull;Hull City,Hull,Hull,Hull City
9,COV,Coventry City,Coventry,,
1001,WIG,,Wigan,,
"""


def csv_gz(text: str) -> bytes:
    return gzip_bytes(text.encode())


def bootstrap(teams):
    return json.dumps(
        {
            "events": [{"id": 1, "deadline_time": "2026-08-14T17:30:00Z"}],
            "teams": [
                {"id": i, "code": code, "name": name, "short_name": "X"}
                for i, (code, name) in enumerate(teams, start=1)
            ],
            "elements": [],
        }
    ).encode()


@pytest.fixture
def ctx(tmp_path):
    """A tiny raw store whose team names are all covered by TEAMS_CSV."""
    config = tmp_path / "config"
    config.mkdir()
    (config / "teams.csv").write_text(TEAMS_CSV, encoding="utf-8")
    store = RawStore(tmp_path / "raw")

    def vaastav(name, text):
        store.write_bytes("vaastav", "data", csv_gz(text), RUN, suffix=".csv.gz", name=name)

    vaastav("2016-17/players_raw", "id,team,team_code\n1,1,3\n2,2,88\n")
    vaastav("2022-23/players_raw", "id,team,team_code\n1,1,3\n2,2,17\n")
    vaastav("2022-23/teams", "code,id,name,short_name\n3,1,Arsenal,ARS\n17,2,Nott'm Forest,NFO\n")
    vaastav("master_team_list", "season,team,team_name\n2016-17,1,Arsenal\n2016-17,2,Hull\n")
    vaastav("2022-23/understat/understat_Arsenal", "h_a,xG,date\nh,1.0,2022-08-05 19:00:00\n")
    vaastav(
        "2022-23/understat/understat_Nottingham_Forest", "h_a,xG,date\na,1.0,2022-08-06 14:00:00\n"
    )
    vaastav("2022-23/understat/understat_player", "id,player_name,team_title\n1,X,Arsenal\n")
    vaastav(
        "2022-23/understat/Bukayo_Saka_7322",
        "goals,h_team,a_team,date\n"
        "0,Arsenal,Nottingham Forest,2022-11-30\n"
        "1,Barcelona,Real Madrid,2014-05-01\n",  # another league: ignored
    )
    manifest = {"expected": ["a"], "written": ["a"], "failed": []}
    store.write("vaastav", "data", json.dumps(manifest).encode(), RUN, name="_manifest")

    xz = lzma.compress(bootstrap([(3, "Arsenal"), (88, "Hull City")]))
    store.write_bytes(
        "fplcache", "bootstrap-static", xz, datetime(2026, 7, 23, tzinfo=UTC), suffix=".json.xz"
    )
    store.write(
        "fpl",
        "bootstrap-static",
        bootstrap([(3, "Arsenal"), (9, "Coventry City")]),
        datetime(2026, 10, 6, tzinfo=UTC),
    )
    store.write_bytes(
        "football-data",
        "E0/0506",
        csv_gz("Div,Date,HomeTeam,AwayTeam\nE0,13/08/05,Wigan,Arsenal\n,,,\n"),
        RUN,
        suffix=".csv.gz",
    )
    store.write_bytes(
        "football-data",
        "E0/2627",
        csv_gz("﻿Div,Date,Time,HomeTeam,AwayTeam\nE0,15/08/2026,15:00,Hull,Coventry\n"),
        RUN,
        suffix=".csv.gz",
    )
    odds = [{"id": "e1", "home_team": "Hull City", "away_team": "Nottingham Forest"}]
    store.write("odds", "soccer_epl", json.dumps(odds).encode(), RUN)
    return BuildContext(store, tmp_path / "data", config_dir=config)


def add_football_data(ctx, code, text):
    ctx.store.write_bytes("football-data", f"E0/{code}", csv_gz(text), RUN, suffix=".csv.gz")


# --- config -> team_dim -----------------------------------------------------------------


def test_team_dim_from_config_types(tmp_path):
    path = tmp_path / "teams.csv"
    path.write_text(TEAMS_CSV, encoding="utf-8")
    df = team_dim_from_config(path)
    assert list(df.columns) == [
        "team_key",
        "short_name",
        "fpl_names",
        "football_data_name",
        "understat_name",
        "odds_api_name",
        "in_fpl",
        "event_time",
        "available_at",
    ]
    assert str(df["team_key"].dtype) == "int64"
    assert str(df["in_fpl"].dtype) == "bool"
    assert str(df["event_time"].dtype) == "datetime64[us, UTC]"
    wigan = df.set_index("team_key").loc[1001]
    assert not wigan["in_fpl"]
    assert pd.isna(wigan["fpl_names"]) and pd.isna(wigan["understat_name"])
    assert df.set_index("team_key").loc[88, "fpl_names"] == "Hull;Hull City"
    assert (df["available_at"] == pd.Timestamp("1970-01-01", tz="UTC")).all()


def test_config_rejects_duplicate_names_and_bad_columns(tmp_path):
    path = tmp_path / "teams.csv"
    path.write_text(TEAMS_CSV + "1002,WGN,,Wigan,,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="football_data.*Wigan"):
        read_teams_config(path)
    path.write_text("team_key,short_name\n3,ARS\n", encoding="utf-8")
    with pytest.raises(ValueError, match="columns"):
        read_teams_config(path)


def test_config_rejects_synthetic_key_with_fpl_name(tmp_path):
    path = tmp_path / "teams.csv"
    path.write_text(TEAMS_CSV.replace("1001,WIG,,", "1001,WIG,Wigan,"), encoding="utf-8")
    with pytest.raises(ValueError, match="1001"):
        read_teams_config(path)


# --- resolver ---------------------------------------------------------------------------


def test_resolver_lookups(tmp_path):
    path = tmp_path / "teams.csv"
    path.write_text(TEAMS_CSV, encoding="utf-8")
    resolver = TeamResolver(team_dim_from_config(path))
    assert resolver.football_data("Nott'm Forest") == 17
    assert resolver.understat("Nottingham Forest") == 17
    assert resolver.odds_api("Nottingham Forest") == 17
    assert resolver.fpl_name("Nott'm Forest") == 17
    assert resolver.fpl_name("Hull") == resolver.fpl_name("Hull City") == 88
    assert resolver.fpl_code(88) == 88
    assert resolver.football_data("Wigan") == 1001
    with pytest.raises(KeyError, match="Wigan Athletic"):
        resolver.odds_api("Wigan Athletic")
    with pytest.raises(KeyError, match="1001"):
        resolver.fpl_code(1001)  # synthetic clubs are not FPL codes
    with pytest.raises(KeyError, match="Coventry"):
        resolver.understat("Coventry")


# --- coverage ---------------------------------------------------------------------------


def test_build_team_dim_passes_coverage_and_writes(ctx):
    assert ORDER[0] == "team_dim" and "team_dim" in BUILDERS
    build(["team_dim"], ctx)
    df = ctx.table("team_dim")
    assert df["team_key"].tolist() == [3, 9, 17, 88, 1001]


def test_coverage_lists_every_unknown_name_at_once(ctx):
    add_football_data(ctx, "0607", "Div,Date,HomeTeam,AwayTeam\nE0,19/08/06,Sheffield Weds,Hull\n")
    odds = [{"id": "e2", "home_team": "Arsenal", "away_team": "Wigan Athletic"}]
    ctx.store.write(
        "odds", "soccer_epl", json.dumps(odds).encode(), datetime(2026, 10, 7, tzinfo=UTC)
    )
    with pytest.raises(TeamCoverageError) as info:
        build(["team_dim"], ctx)
    first_line = str(info.value).splitlines()[0]
    assert "2 unknown" in first_line
    assert "'Sheffield Weds'" in first_line and "'Wigan Athletic'" in first_line
    assert not ctx.table_path("team_dim").exists()


def test_coverage_flags_unknown_fpl_code_and_name(ctx):
    ctx.store.write(
        "fpl",
        "bootstrap-static",
        bootstrap([(3, "Arsenal FC"), (999, "Nowhere")]),
        datetime(2026, 10, 7, tzinfo=UTC),
    )
    with pytest.raises(TeamCoverageError) as info:
        build(["team_dim"], ctx)
    message = str(info.value)
    assert "FPL team code 999" in message
    assert "'Arsenal FC'" in message


def test_coverage_flags_one_sided_understat_match(ctx):
    ctx.store.write_bytes(
        "vaastav",
        "data",
        csv_gz("goals,h_team,a_team,date\n0,Queens Park Rangers,Arsenal,2014-08-16\n"),
        RUN,
        suffix=".csv.gz",
        name="2022-23/understat/Some_Player_1",
    )
    with pytest.raises(TeamCoverageError, match="Queens Park Rangers"):
        build(["team_dim"], ctx)


def test_coverage_flags_unknown_understat_team_file(ctx):
    ctx.store.write_bytes(
        "vaastav",
        "data",
        csv_gz("h_a,xG,date\nh,1.0,2022-08-05 19:00:00\n"),
        RUN,
        suffix=".csv.gz",
        name="2022-23/understat/understat_Coventry",
    )
    with pytest.raises(TeamCoverageError, match="'Coventry'"):
        build(["team_dim"], ctx)


# --- the checked-in config --------------------------------------------------------------


def test_repo_teams_config_is_well_formed():
    df = team_dim_from_config(REPO_CONFIG / "teams.csv")
    assert df["team_key"].is_unique and df["short_name"].is_unique
    fpl = df[df["in_fpl"]]
    assert len(fpl) == 35
    synthetic = df[~df["in_fpl"]].sort_values("team_key")
    assert synthetic["team_key"].tolist() == list(range(1001, 1011))
    assert synthetic["football_data_name"].tolist() == sorted(synthetic["football_data_name"])
    resolver = TeamResolver(df)
    assert resolver.football_data("Man United") == resolver.understat("Manchester United") == 1
    assert resolver.odds_api("Brighton and Hove Albion") == 36
    assert resolver.fpl_name("Ipswich Town") == resolver.fpl_name("Ipswich") == 40
