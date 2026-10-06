"""A small but complete synthetic raw/ store for build-layer tests: full 20-club seasons
(380 fixtures, double round robin) in each source's shape, with one player per club."""

from __future__ import annotations

import io
import json
import lzma
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from fplopt.build import teams
from fplopt.build.common import UTC_US, BuildContext, fplcache_and_own_bootstraps, write_table
from fplopt.build.snapshots import build_player_snapshot
from fplopt.build.teams import team_dim_from_config
from fplopt.ingest.raw_store import RawStore, gzip_bytes
from fplopt.seasons import football_data_code, season_label

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config"
RUN = datetime(2026, 10, 6, 3, 26, 9, tzinfo=UTC)
CURRENT_AT = datetime(2026, 9, 1, tzinfo=UTC)
# 20 FPL team codes (config/teams.csv); team id i+1 in every synthetic season.
TEAM_CODES = [3, 7, 91, 94, 36, 90, 8, 31, 11, 54, 2, 13, 14, 43, 1, 4, 17, 20, 6, 21]
FD_NAMES = dict(
    pd.read_csv(REPO_CONFIG / "teams.csv")[["team_key", "football_data"]].itertuples(index=False)
)

MERGED_STATS = [
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "bps",
    "total_points",
]


def csv_gz(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return gzip_bytes(buf.getvalue().encode())


def round_robin(n: int = 20) -> list[list[tuple[int, int]]]:
    """Double round robin of team ids 1..n (circle method): 2(n-1) rounds of n/2 matches."""
    ids = list(range(1, n + 1))
    rounds = []
    for r in range(n - 1):
        pairs = [(ids[i], ids[n - 1 - i]) for i in range(n // 2)]
        rounds.append([(a, b) if r % 2 else (b, a) for a, b in pairs])
        ids = [ids[0], ids[-1], *ids[1:-1]]
    return rounds + [[(b, a) for a, b in rnd] for rnd in rounds]


def season_fixtures(season: int, gw_numbers: list[int] | None = None) -> pd.DataFrame:
    """FPL fixtures-endpoint rows for a season: 38 weekly rounds from the first Saturday of
    August, kickoffs 14:00 UTC (10 per round, staggered by a minute), all finished."""
    start = datetime(season, 8, 1, 14, tzinfo=UTC)
    start += timedelta(days=(5 - start.weekday()) % 7)
    gw_numbers = gw_numbers or list(range(1, 39))
    rows = []
    for r, matches in enumerate(round_robin()):
        for m, (home, away) in enumerate(matches):
            fixture_id = r * 10 + m + 1
            rows.append(
                {
                    "id": fixture_id,
                    "event": gw_numbers[r],
                    "kickoff_time": (start + timedelta(weeks=r, minutes=m)).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                    "team_h": home,
                    "team_a": away,
                    "team_h_score": fixture_id % 3,
                    "team_a_score": fixture_id % 2,
                    "finished": True,
                }
            )
    return pd.DataFrame(rows)


def players_raw(season: int) -> pd.DataFrame:
    """One player per club: element id = team id, code = 100000 + team id."""
    return pd.DataFrame(
        {
            "id": range(1, 21),
            "code": [100000 + i for i in range(1, 21)],
            "first_name": [f"First{i}" for i in range(1, 21)],
            "second_name": [f"Second{i}" for i in range(1, 21)],
            "web_name": [f"Web{i}" for i in range(1, 21)],
            "team": range(1, 21),
            "team_code": TEAM_CODES,
            "element_type": [(i % 4) + 1 for i in range(1, 21)],
        }
    )


def merged_gw(fixtures: pd.DataFrame) -> pd.DataFrame:
    """One row per fixture side for the club's only player: 90 minutes, all the side's
    goals (so goal sums match), GW = round = event."""
    rows = []
    for fx in fixtures.itertuples(index=False):
        for was_home, team, opp, scored, conceded in (
            (True, fx.team_h, fx.team_a, fx.team_h_score, fx.team_a_score),
            (False, fx.team_a, fx.team_h, fx.team_a_score, fx.team_h_score),
        ):
            row = dict.fromkeys(MERGED_STATS, 0)
            row.update(
                name=f"First{team} Second{team}",
                element=team,
                fixture=fx.id,
                kickoff_time=fx.kickoff_time,
                round=fx.event,
                GW=fx.event,
                was_home=was_home,
                opponent_team=opp,
                team_h_score=fx.team_h_score,
                team_a_score=fx.team_a_score,
                minutes=90,
                goals_scored=scored,
                goals_conceded=conceded,
                clean_sheets=int(conceded == 0),
                total_points=2 + 4 * scored,
                value=50,
                selected=1000,
                transfers_in=0,
                transfers_out=0,
                xP=9.9,
            )
            rows.append(row)
    return pd.DataFrame(rows)


def football_data(fixtures: pd.DataFrame, season: int) -> pd.DataFrame:
    kickoffs = pd.to_datetime(fixtures["kickoff_time"], utc=True).dt.tz_convert("Europe/London")
    fmt = "%d/%m/%y" if season <= 2016 else "%d/%m/%Y"
    return pd.DataFrame(
        {
            "Div": "E0",
            "Date": kickoffs.dt.strftime(fmt),
            "HomeTeam": [FD_NAMES[TEAM_CODES[i - 1]] for i in fixtures["team_h"]],
            "AwayTeam": [FD_NAMES[TEAM_CODES[i - 1]] for i in fixtures["team_a"]],
            "FTHG": fixtures["team_h_score"],
            "FTAG": fixtures["team_a_score"],
        }
    )


def bootstrap(season: int, fixtures: pd.DataFrame, finished_through: int = 38) -> dict:
    """bootstrap-static with events (deadline = first kickoff − 90 min), teams and elements."""
    kickoffs = pd.to_datetime(fixtures["kickoff_time"], utc=True)
    events = []
    for gw, first in kickoffs.groupby(fixtures["event"]).min().items():
        deadline = first - pd.Timedelta(minutes=90)
        events.append(
            {
                "id": int(gw),
                "deadline_time": deadline.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "average_entry_score": 50 + int(gw) if gw <= finished_through else 0,
                "finished": bool(gw <= finished_through),
            }
        )
    players = players_raw(season)
    return {
        "events": events,
        "teams": [
            {"id": i, "code": code, "name": f"T{code}"} for i, code in enumerate(TEAM_CODES, 1)
        ],
        "elements": [
            {
                "id": int(p.id),
                "code": int(p.code),
                "first_name": p.first_name,
                "second_name": p.second_name,
                "web_name": p.web_name,
                "team": int(p.team),
                "team_code": int(p.team_code),
                "element_type": int(p.element_type),
                "now_cost": 50,
                "status": "a",
                "transfers_in_event": 0,
                "transfers_out_event": 0,
                "cost_change_event": 0,
            }
            for p in players.itertuples(index=False)
        ],
    }


class World:
    """Writes synthetic seasons into a raw store and builds a context over it."""

    def __init__(self, tmp_path: Path) -> None:
        self.store = RawStore(tmp_path / "raw")
        self.ctx = BuildContext(self.store, tmp_path / "data", config_dir=REPO_CONFIG)
        write_table(
            team_dim_from_config(REPO_CONFIG / "teams.csv"),
            "team_dim",
            teams.SCHEMA,
            self.ctx.data_dir,
            teams.SORT_BY,
        )
        manifest = {"expected": ["a"], "written": ["a"], "failed": []}
        self.store.write("vaastav", "data", json.dumps(manifest).encode(), RUN, name="_manifest")

    def vaastav(self, name: str, df: pd.DataFrame) -> None:
        self.store.write_bytes("vaastav", "data", csv_gz(df), RUN, suffix=".csv.gz", name=name)

    def football_data(self, season: int, df: pd.DataFrame, at: datetime = RUN) -> None:
        self.store.write_bytes(
            "football-data", f"E0/{football_data_code(season)}", csv_gz(df), at, suffix=".csv.gz"
        )

    def add_vaastav_season(
        self,
        season: int,
        *,
        fixtures_csv: bool = True,
        teams_csv: bool = True,
        fixtures: pd.DataFrame | None = None,
        merged: pd.DataFrame | None = None,
        players: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        label = season_label(season)
        fixtures = season_fixtures(season) if fixtures is None else fixtures
        self.vaastav(f"{label}/players_raw", players_raw(season) if players is None else players)
        self.vaastav(f"{label}/gws/merged_gw", merged_gw(fixtures) if merged is None else merged)
        if fixtures_csv:
            self.vaastav(f"{label}/fixtures", fixtures)
        if teams_csv:
            ids = pd.DataFrame({"id": range(1, 21), "code": TEAM_CODES})
            self.vaastav(f"{label}/teams", ids.assign(name=[f"T{c}" for c in TEAM_CODES]))
        self.football_data(season, football_data(fixtures, season))
        return fixtures

    def add_fplcache_bootstrap(self, at: datetime, payload: dict) -> None:
        data = lzma.compress(json.dumps(payload).encode())
        self.store.write_bytes("fplcache", "bootstrap-static", data, at, suffix=".json.xz")

    def add_own_bootstrap(self, at: datetime, payload: dict) -> None:
        self.store.write("fpl", "bootstrap-static", json.dumps(payload).encode(), at)

    def add_own_fixtures(self, at: datetime, fixtures: pd.DataFrame) -> None:
        records = json.loads(fixtures.to_json(orient="records"))
        self.store.write("fpl", "fixtures", json.dumps(records).encode(), at)

    def add_current_season(
        self, at: datetime = CURRENT_AT, finished_through: int = 2, fd_rows: int = 15
    ) -> pd.DataFrame:
        """2026-27 from our own archive: GWs up to `finished_through` finished, fixture 380
        postponed (no GW/kickoff), football-data published for the first `fd_rows`."""
        fixtures = season_fixtures(2026)
        fixtures["finished"] = fixtures["event"] <= finished_through
        fixtures[["team_h_score", "team_a_score"]] = fixtures[
            ["team_h_score", "team_a_score"]
        ].where(fixtures["finished"])
        fixtures["event"] = fixtures["event"].astype("Int64")
        fixtures.loc[fixtures["id"] == 380, ["event", "kickoff_time"]] = None
        payload = bootstrap(2026, season_fixtures(2026), finished_through=finished_through)
        self.add_own_bootstrap(at, payload)
        self.add_own_fixtures(at, fixtures)
        played = fixtures[fixtures["finished"]]
        self.football_data(2026, football_data(played.head(fd_rows), 2026))
        return fixtures

    def write_player_snapshot(self) -> None:
        """player_snapshot (player_gw needs it): built in one process without the season
        coverage check, or, for a world without bootstraps, an empty table with the columns
        the player builders read."""
        if fplcache_and_own_bootstraps(self.store):
            build_player_snapshot(self.ctx, jobs=1, first_season=None)
            return
        empty = pd.DataFrame(
            {
                "snapshot_at": pd.Series(dtype=UTC_US),
                "season": pd.Series(dtype="int64"),
                "player_key": pd.Series(dtype="int64"),
                "team_key": pd.Series(dtype="int64"),
            }
        )
        self.ctx.data_dir.mkdir(parents=True, exist_ok=True)
        empty.to_parquet(self.ctx.table_path("player_snapshot"), index=False)

    def add_element_summary(
        self,
        at: datetime,
        history: pd.DataFrame,
        season: int | None,
        through_event: int,
        elements: list[int] | None = None,
    ) -> None:
        """An element-summary run: one file per element (default: those in `history`) with
        its `history` rows."""
        if elements is None:
            elements = sorted(int(e) for e in history["element"].unique())
        for element in elements:
            rows = history[history["element"] == element]
            payload = {
                "fixtures": [],
                "history": json.loads(rows.to_json(orient="records")),
                "history_past": [],
            }
            self.store.write(
                "fpl", "element-summary", json.dumps(payload).encode(), at, name=str(element)
            )
        manifest = {
            "run_at": at.isoformat(),
            "season": season,
            "through_event": through_event,
            "expected": [str(e) for e in elements],
            "written": [str(e) for e in elements],
            "failed": [],
        }
        self.store.write(
            "fpl", "element-summary", json.dumps(manifest).encode(), at, name="_manifest"
        )
