"""`fixture` (one row per EPL match, 2016/17 →), `gameweek` (one row per season and GW) and
`gameweek_result` (per GW outcomes: `average_entry_score`).

fixture sources:
- 2016-17, 2017-18 (no fixtures.csv): vaastav merged_gw. Per `fixture` id, the home side's
  rows (`was_home`) name the away club in `opponent_team` and vice versa; kickoff, round and
  scores must be unique per fixture.
- 2018-19 … 2025-26: vaastav `fixtures.csv`.
- Seasons vaastav doesn't cover (2026-27 →): per season, the newest own `raw/fpl/fixtures`
  snapshot of that season (season = start year of its earliest kickoff), so a new season's
  fixture list does not hide the previous season's.
Season team ids -> FPL team codes via vaastav `teams.csv` (2019-20+) or `players_raw`
`team`→`team_code` (earlier), and for own snapshots the newest bootstrap of that season.

`available_at` is the lockdown of the fixture's GW (when results and stats are final); a
fixture without a GW/kickoff (postponed, not yet rescheduled) gets 2100-01-01. That the
*schedule* was known earlier is Phase 2's concern (`fixture_snapshot`).

football-data cross-check (fails the build): every football-data row joins one fixture on
(season, home, away); its date must equal the UK-local kickoff date and its goals the FPL
goals; every finished FPL fixture needs a football-data row, except current-season fixtures
dated after the newest football-data row of that season (not published yet). Measured on
raw/ 2026-10-06: 3,850 played matches (2016/17 – 2026/27 GW5) join with 0 date and 0 goal
mismatches.

gameweek: GWs that have fixtures. Deadlines from the newest bootstrap of each season
(`deadline_source='bootstrap'`: 2020-21 … 2026-27, fplcache from 2021-04-18 still lists
2020-21's 38 events), 2018-19 from vaastav fixtures.csv `deadline_time` ('fixtures_csv'),
else first kickoff − 90 min ('approx': 2016-17, 2017-18, 2019-20). Measured leads (first
kickoff − deadline): 2018-19 always 60 min, 2020-21 … 2026-27 90 min (one 240-min GW in
2024-25); so 'approx' is at or before the true deadline — conservative for as-of reads.

gameweek_result: one row per gameweek row; `average_entry_score` only from bootstraps and
only for finished events (null otherwise). An outcome, so `event_time` = `available_at` =
the GW's lockdown (`gameweek` rows are available at their deadline).

2022-23 GW7: FPL's bootstrap keeps event 7 (deadline 2022-09-10 10:00 UTC,
average_entry_score 0, no fixtures: the round was cancelled after the Queen's death and its
matches moved into other GWs). There are no fixtures and no player rows for it, so
`gameweek` has no GW7 row (37 GWs; FPL's numbering 1–6, 8–38 is kept). `gw_index` is the
dense rank of `gw` among a season's GWs with fixtures: 2019-20 39–47 -> 30–38 and 2022-23
8–38 -> 7–37.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from fplopt.build.common import (
    UK,
    UTC_US,
    BuildContext,
    fplcache_and_own_bootstraps,
    latest_complete_run,
    lockdown_times,
    read_raw_csv,
)
from fplopt.build.teams import TeamResolver
from fplopt.ingest.raw_store import RawStore
from fplopt.seasons import bootstrap_season, parse_season_label, season_start_year

log = logging.getLogger(__name__)

FIRST_SEASON = 2016
FIXTURES_PER_SEASON = 380
MATCHES_PER_TEAM = 38
GWS_PER_SEASON = 38
# Seasons whose FPL game has fewer rounds with fixtures (see module docstring).
GW_COUNT_EXCEPTIONS = {2022: 37}
UNSCHEDULED_AT = pd.Timestamp("2100-01-01", tz="UTC").as_unit("us")
APPROX_DEADLINE_LEAD = pd.Timedelta(minutes=90)

# --- shared helpers (used by the other builders too) -------------------------------------


def to_utc(values: pd.Series) -> pd.Series:
    """ISO-8601 strings (or timestamps) -> datetime64[us, UTC]; nulls stay null."""
    return pd.to_datetime(values, utc=True, format="ISO8601").astype(UTC_US)


def uk_dates(times: pd.Series) -> pd.Series:
    """UK-local calendar date of each UTC timestamp (object dtype of `date`, nulls None)."""
    local = times.dt.tz_convert(UK)
    return pd.Series(
        [None if pd.isna(t) else t.date() for t in local], index=times.index, dtype=object
    )


def vaastav_run(ctx: BuildContext) -> Path:
    return latest_complete_run(ctx.store, "vaastav", "data")


def vaastav_seasons(run: Path) -> dict[int, Path]:
    """Season start year -> season folder of a vaastav run (2016 → …)."""
    seasons = {}
    for path in sorted(run.iterdir()):
        if path.is_dir():
            try:
                season = parse_season_label(path.name)
            except ValueError:
                continue
            if season >= FIRST_SEASON:
                seasons[season] = path
    return seasons


def season_team_codes(season_dir: Path) -> dict[int, int]:
    """FPL team id (resets each season) -> team code for a vaastav season folder:
    `teams.csv` where present (2019-20+), else `players_raw` `team`→`team_code`."""
    teams_path = season_dir / "teams.csv.gz"
    if teams_path.exists():
        teams = read_raw_csv(teams_path, usecols=["id", "code"])
        pairs = teams.rename(columns={"id": "team", "code": "team_code"})
    else:
        players = read_raw_csv(season_dir / "players_raw.csv.gz", usecols=["team", "team_code"])
        pairs = players.drop_duplicates()
    if pairs["team"].duplicated().any() or pairs["team_code"].duplicated().any():
        raise ValueError(f"{season_dir.name}: team id -> code is not 1:1")
    return {int(t): int(c) for t, c in zip(pairs["team"], pairs["team_code"], strict=True)}


def bootstrap_team_codes(bootstrap: dict) -> dict[int, int]:
    return {int(team["id"]): int(team["code"]) for team in bootstrap["teams"]}


def newest_per_season[T](entries: list[T], season_of: Callable[[T], int]) -> dict[int, T]:
    """The newest of time-ordered `entries` for each season. Seasons only move forward in
    time, so a binary search per season reads a few entries instead of all of them."""
    seasons: dict[int, int] = {}

    def season_at(i: int) -> int:
        if i not in seasons:
            seasons[i] = season_of(entries[i])
        return seasons[i]

    def last_at_most(season: int) -> int:
        lo, hi = 0, len(entries)
        while lo < hi:
            mid = (lo + hi) // 2
            if season_at(mid) <= season:
                lo = mid + 1
            else:
                hi = mid
        return lo - 1

    if not entries:
        return {}
    result = {}
    for season in range(season_at(0), season_at(len(entries) - 1) + 1):
        i = last_at_most(season)
        if i >= 0 and season_at(i) == season:
            result[season] = entries[i]
    return result


def last_bootstrap_per_season(store: RawStore) -> dict[int, tuple[datetime, Path, str]]:
    """The newest archived bootstrap (fplcache or own) describing each season (~13 of ~8,000
    files parsed); the season is `bootstrap_season(...)`, never the snapshot timestamp."""
    return newest_per_season(
        fplcache_and_own_bootstraps(store),
        lambda entry: bootstrap_season(RawStore.read_json(entry[1])),
    )


def fixtures_snapshot_season(records: list[dict], snapshot_at: datetime) -> int:
    """Season of an FPL fixtures payload: start year of its earliest kickoff (the snapshot's
    own season if no fixture has a kickoff)."""
    kickoffs = [r["kickoff_time"] for r in records if r.get("kickoff_time")]
    if not kickoffs:
        return season_start_year(snapshot_at)
    return season_start_year(to_utc(pd.Series([min(kickoffs)])).iloc[0].to_pydatetime())


def own_fixtures_per_season(store: RawStore) -> dict[int, tuple[datetime, Path]]:
    """The newest own `fpl/fixtures` snapshot of each season."""
    return newest_per_season(
        store.entries("fpl", "fixtures"),
        lambda entry: fixtures_snapshot_season(RawStore.read_json(entry[1]), entry[0]),
    )


# --- fixture -----------------------------------------------------------------------------

FIXTURE_COLUMNS = [
    "fixture_key",
    "season",
    "fpl_fixture_id",
    "gw",
    "gw_index",
    "kickoff_time",
    "home_team_key",
    "away_team_key",
    "home_goals",
    "away_goals",
    "finished",
    "fd_date",
    "event_time",
    "available_at",
]


def _fixture_frame_checks(df: pd.DataFrame) -> pd.Series:
    scheduled = df["gw"].notna()
    return (
        (scheduled == df["kickoff_time"].notna())
        & (scheduled == df["gw_index"].notna())
        & (~df["finished"] | (scheduled & df["home_goals"].notna() & df["away_goals"].notna()))
        & (df["finished"] | (df["home_goals"].isna() & df["away_goals"].isna()))
        & (df["home_team_key"] != df["away_team_key"])
        & (df["fixture_key"] == df["season"] * 1000 + df["fpl_fixture_id"])
    )


def _season_counts_ok(df: pd.DataFrame) -> bool:
    if not (df.groupby("season").size() == FIXTURES_PER_SEASON).all():
        return False
    sides = pd.concat(
        [
            df[["season", "home_team_key"]].set_axis(["season", "team"], axis=1),
            df[["season", "away_team_key"]].set_axis(["season", "team"], axis=1),
        ]
    )
    return bool((sides.groupby(["season", "team"]).size() == MATCHES_PER_TEAM).all())


def _kickoff_in_season(df: pd.DataFrame) -> pd.Series:
    years = df["kickoff_time"].dt.year.astype("Int64")
    months = df["kickoff_time"].dt.month.astype("Int64")
    # Seasons start in August and end by July (2019/20 ended 2020-07-26).
    start = years.where(months >= 8, years - 1)
    return df["kickoff_time"].isna() | (start == df["season"])


FIXTURE_SCHEMA = pa.DataFrameSchema(
    {
        "fixture_key": pa.Column("int64", unique=True),
        "season": pa.Column("int64", pa.Check.ge(FIRST_SEASON)),
        "fpl_fixture_id": pa.Column("int64", pa.Check.in_range(1, 999)),
        "gw": pa.Column("Int64", pa.Check.in_range(1, 47), nullable=True),
        "gw_index": pa.Column("Int64", pa.Check.in_range(1, GWS_PER_SEASON), nullable=True),
        "kickoff_time": pa.Column(UTC_US, nullable=True),
        "home_team_key": pa.Column("int64"),
        "away_team_key": pa.Column("int64"),
        "home_goals": pa.Column("Int64", pa.Check.ge(0), nullable=True),
        "away_goals": pa.Column("Int64", pa.Check.ge(0), nullable=True),
        "finished": pa.Column(bool),
        "fd_date": pa.Column(pa.Date, nullable=True),
        "event_time": pa.Column(UTC_US),
        "available_at": pa.Column(UTC_US),
    },
    checks=[
        pa.Check(_fixture_frame_checks, error="fixture row consistency"),
        pa.Check(_kickoff_in_season, error="kickoff outside its season"),
        pa.Check(_season_counts_ok, error="380 fixtures per season, 38 per team"),
        pa.Check(
            lambda df: ~df.duplicated(["season", "home_team_key", "away_team_key"]),
            error="(season, home, away) must be unique",
        ),
        pa.Check(
            lambda df: (df["available_at"] >= df["event_time"]) | df["kickoff_time"].isna(),
            error="available_at before kickoff",
        ),
    ],
    strict=True,
    ordered=True,
)
FIXTURE_SORT_BY = ("fixture_key",)


class FixtureCrossCheckError(ValueError):
    """FPL fixtures and football-data results disagree."""


def fixtures_from_merged_gw(merged: pd.DataFrame, team_codes: dict[int, int]) -> pd.DataFrame:
    """Raw fixture rows (fpl_fixture_id, gw, kickoff, home/away codes, goals) derived from a
    season's merged_gw (2016-17, 2017-18 have no fixtures.csv)."""
    rows = merged[
        ["fixture", "round", "kickoff_time", "was_home", "opponent_team"]
        + ["team_h_score", "team_a_score"]
    ].drop_duplicates()
    per_fixture = rows.groupby("fixture")
    for column in ("round", "kickoff_time", "team_h_score", "team_a_score"):
        bad = per_fixture[column].nunique()
        if (bad > 1).any():
            raise ValueError(f"merged_gw: {column} differs within fixture(s) {bad[bad > 1].index}")
    # The home side's rows name the away club as their opponent and vice versa.
    away_ids = rows[rows["was_home"]].groupby("fixture")["opponent_team"].agg(_only)
    home_ids = rows[~rows["was_home"]].groupby("fixture")["opponent_team"].agg(_only)
    first = per_fixture.first()
    out = pd.DataFrame(
        {
            "fpl_fixture_id": first.index.astype("int64"),
            "gw": first["round"].to_numpy(),
            "kickoff_time": to_utc(first["kickoff_time"]).to_numpy(),
            "home_id": home_ids.reindex(first.index).to_numpy(),
            "away_id": away_ids.reindex(first.index).to_numpy(),
            "home_goals": first["team_h_score"].to_numpy(),
            "away_goals": first["team_a_score"].to_numpy(),
            "finished": True,
        }
    )
    if out[["home_id", "away_id"]].isna().any().any():
        missing = out.loc[out[["home_id", "away_id"]].isna().any(axis=1), "fpl_fixture_id"]
        raise ValueError(f"merged_gw: fixture(s) {missing.tolist()} lack one side's rows")
    return _codes(out, team_codes)


def _only(values: pd.Series) -> int:
    unique = values.unique()
    if len(unique) != 1:
        raise ValueError(f"merged_gw: ambiguous opponent ids {list(unique)}")
    return int(unique[0])


def fixtures_from_api(fixtures: pd.DataFrame, team_codes: dict[int, int]) -> pd.DataFrame:
    """Raw fixture rows from FPL's fixtures endpoint shape (vaastav fixtures.csv or our
    `raw/fpl/fixtures` snapshots)."""
    finished = fixtures["finished"].astype(bool)
    out = pd.DataFrame(
        {
            "fpl_fixture_id": fixtures["id"].astype("int64"),
            "gw": fixtures["event"].astype("Int64"),
            "kickoff_time": to_utc(fixtures["kickoff_time"]),
            "home_id": fixtures["team_h"].astype("int64"),
            "away_id": fixtures["team_a"].astype("int64"),
            # Goals only once the result is final (a live match has provisional scores).
            "home_goals": fixtures["team_h_score"].astype("Int64").where(finished),
            "away_goals": fixtures["team_a_score"].astype("Int64").where(finished),
            "finished": finished,
        }
    )
    return _codes(out, team_codes)


def _codes(rows: pd.DataFrame, team_codes: dict[int, int]) -> pd.DataFrame:
    unknown = sorted((set(rows["home_id"]) | set(rows["away_id"])) - set(team_codes))
    if unknown:
        raise ValueError(f"team id(s) {unknown} missing from the season's team list")
    rows = rows.assign(
        home_team_key=rows["home_id"].map(team_codes).astype("int64"),
        away_team_key=rows["away_id"].map(team_codes).astype("int64"),
    )
    return rows.drop(columns=["home_id", "away_id"])


def _raw_fixtures(ctx: BuildContext) -> pd.DataFrame:
    """Fixture rows for every season (no keys/timing columns yet)."""
    run = vaastav_run(ctx)
    frames = []
    for season, season_dir in vaastav_seasons(run).items():
        team_codes = season_team_codes(season_dir)
        fixtures_path = season_dir / "fixtures.csv.gz"
        if fixtures_path.exists():
            raw = fixtures_from_api(read_raw_csv(fixtures_path), team_codes)
        else:
            merged = read_raw_csv(
                season_dir / "gws" / "merged_gw.csv.gz",
                usecols=[
                    "fixture",
                    "round",
                    "kickoff_time",
                    "was_home",
                    "opponent_team",
                    "team_h_score",
                    "team_a_score",
                ],
            )
            raw = fixtures_from_merged_gw(merged, team_codes)
        frames.append(raw.assign(season=season))
    frames += _current_season_fixtures(ctx, {f["season"].iloc[0] for f in frames})
    return pd.concat(frames, ignore_index=True)


def _current_season_fixtures(ctx: BuildContext, have: set[int]) -> list[pd.DataFrame]:
    """Per season vaastav doesn't cover, its newest own fixtures snapshot."""
    frames = []
    bootstraps = None
    for season, (_, path) in sorted(own_fixtures_per_season(ctx.store).items()):
        if season in have:
            log.info("fixtures snapshot %s is for season %d, covered by vaastav", path, season)
            continue
        bootstraps = bootstraps or last_bootstrap_per_season(ctx.store)
        if season not in bootstraps:
            raise LookupError(f"no archived bootstrap for season {season} (team ids of {path})")
        team_codes = bootstrap_team_codes(RawStore.read_json(bootstraps[season][1]))
        fixtures = pd.DataFrame(RawStore.read_json(path))
        frames.append(fixtures_from_api(fixtures, team_codes).assign(season=season))
    return frames


def football_data_files(
    ctx: BuildContext, first_season: int | None = FIRST_SEASON
) -> Iterator[tuple[int, Path]]:
    """(season, newest file) per football-data E0 season folder, oldest first; `first_season`
    None = every season present (2005/06+ after the Elo backfill). The season comes from the
    folder code (`0506` -> 2005), never from the July-cutoff rule: the 2019/20 restart ran
    into July 2020."""
    base = ctx.store.root / "football-data" / "E0"
    for season_dir in sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []:
        season = 2000 + int(season_dir.name[:2])
        if first_season is not None and season < first_season:
            continue
        path = ctx.store.latest("football-data", f"E0/{season_dir.name}", suffix=".csv.gz")
        if path is not None:
            yield season, path


def football_data_frames(
    ctx: BuildContext,
    resolver: TeamResolver,
    columns: list[str] | tuple[str, ...] = (),
    first_season: int | None = FIRST_SEASON,
) -> pd.DataFrame:
    """football-data rows of every season (newest file per season): season, fd_date,
    home/away team keys, plus those of `columns` the season's file has (absent -> missing
    column for that season, NaN after the concat). Rows without a `Date` are dropped
    (2014/15 has a trailing empty row)."""
    keys = ["season", "fd_date", "home_team_key", "away_team_key"]
    frames = []
    for season, path in football_data_files(ctx, first_season):
        df = read_raw_csv(path, usecols=["Date", "HomeTeam", "AwayTeam", *columns])
        df = df.dropna(subset=["Date"]).reset_index(drop=True)
        frames.append(
            pd.concat(
                [
                    pd.DataFrame(
                        {
                            "season": season,
                            "fd_date": [_fd_date(text) for text in df["Date"]],
                            "home_team_key": [resolver.football_data(n) for n in df["HomeTeam"]],
                            "away_team_key": [resolver.football_data(n) for n in df["AwayTeam"]],
                        }
                    ),
                    df[[c for c in columns if c in df.columns]],
                ],
                axis=1,
            )
        )
    if not frames:
        return pd.DataFrame(columns=keys)
    out = pd.concat(frames, ignore_index=True)
    out["season"] = out["season"].astype("int64")
    return out


def football_data_results(
    ctx: BuildContext, resolver: TeamResolver, first_season: int | None = FIRST_SEASON
) -> pd.DataFrame:
    """football-data results (newest file per season, season >= `first_season`; None = all):
    season, fd_date, home/away team keys, goals."""
    df = football_data_frames(ctx, resolver, ["FTHG", "FTAG"], first_season)
    columns = ["season", "fd_date", "home_team_key", "away_team_key"]
    if df.empty:
        return pd.DataFrame(columns=[*columns, "fd_home_goals", "fd_away_goals"])
    return pd.DataFrame(
        {
            **{c: df[c] for c in columns},
            "fd_home_goals": df["FTHG"].astype("int64"),
            "fd_away_goals": df["FTAG"].astype("int64"),
        }
    )


def _fd_date(text: str) -> date:
    """football-data `Date`: dd/mm/yy (2016/17 and earlier) or dd/mm/yyyy."""
    fmt = "%d/%m/%Y" if len(str(text).strip()) == 10 else "%d/%m/%y"
    return datetime.strptime(str(text).strip(), fmt).date()


def attach_football_data(fixtures: pd.DataFrame, fd: pd.DataFrame) -> pd.DataFrame:
    """Join football-data results onto fixtures by (season, home, away) and cross-check date
    and goals; sets `fd_date`. Raises FixtureCrossCheckError listing every problem."""
    keys = ["season", "home_team_key", "away_team_key"]
    if fd.duplicated(keys).any():
        raise FixtureCrossCheckError(
            f"duplicate football-data matches: {fd[fd.duplicated(keys, keep=False)]}"
        )
    merged = fixtures.merge(fd, on=keys, how="outer", indicator=True, validate="one_to_one")
    # Outer-join rows from football-data alone have no `finished`: treat as not finished.
    merged["finished"] = merged["finished"].eq(True)
    problems: list[str] = []

    def show(rows: pd.DataFrame, what: str) -> None:
        if len(rows):
            sample = rows[[*keys, "fpl_fixture_id", "kickoff_time", "fd_date"]].head(10)
            problems.append(f"{len(rows)} {what}:\n{sample.to_string()}")

    show(merged[merged["_merge"] == "right_only"], "football-data row(s) without an FPL fixture")
    both = merged[merged["_merge"] == "both"]
    kickoff_dates = uk_dates(both["kickoff_time"])
    show(both[kickoff_dates != both["fd_date"]], "date mismatch(es) (UK kickoff date)")
    played = both[both["finished"]]
    goals_differ = (played["home_goals"] != played["fd_home_goals"]) | (
        played["away_goals"] != played["fd_away_goals"]
    )
    show(played[goals_differ.fillna(True)], "goal mismatch(es)")
    unfinished = both[~both["finished"]]
    if len(unfinished):
        log.warning(
            "%d football-data result(s) for fixtures FPL has not finished yet", len(unfinished)
        )
    missing = merged[(merged["_merge"] == "left_only") & merged["finished"]]
    if len(missing):
        # Current season: football-data publishes with a delay (possibly part of a day), so
        # fixtures on or after its newest date are not missing yet.
        newest = fd.groupby("season")["fd_date"].max()
        current = fixtures["season"].max()
        pending = (missing["season"] == current) & (
            uk_dates(missing["kickoff_time"]) >= newest.get(current, date.min)
        ).astype(bool)
        show(missing[~pending], "finished FPL fixture(s) without a football-data row")
    if problems:
        raise FixtureCrossCheckError("football-data cross-check failed: " + "\n".join(problems))
    return fixtures.merge(fd[[*keys, "fd_date"]], on=keys, how="left", validate="one_to_one")


def assemble_fixtures(raw: pd.DataFrame, fd: pd.DataFrame) -> pd.DataFrame:
    """Keys, gw_index, timing columns and the football-data join for raw fixture rows."""
    df = raw.copy()
    df["season"] = df["season"].astype("int64")
    df["fixture_key"] = df["season"] * 1000 + df["fpl_fixture_id"]
    df["gw"] = df["gw"].astype("Int64")
    df["gw_index"] = (
        df.groupby("season")["gw"].rank(method="dense").astype("Int64").where(df["gw"].notna())
    )
    df["kickoff_time"] = df["kickoff_time"].astype(UTC_US)
    df["home_goals"] = df["home_goals"].astype("Int64")
    df["away_goals"] = df["away_goals"].astype("Int64")
    df["finished"] = df["finished"].astype(bool)
    last_kickoff = df.groupby(["season", "gw"])["kickoff_time"].transform("max")
    df["event_time"] = df["kickoff_time"].fillna(UNSCHEDULED_AT).astype(UTC_US)
    df["available_at"] = lockdown_times(last_kickoff).fillna(UNSCHEDULED_AT).astype(UTC_US)
    df = attach_football_data(df, fd)
    df["fd_date"] = df["fd_date"].astype(object).where(df["fd_date"].notna(), None)
    return df[FIXTURE_COLUMNS]


def build_fixture(ctx: BuildContext) -> pd.DataFrame:
    resolver = TeamResolver(ctx.table("team_dim"))
    raw = _raw_fixtures(ctx)
    for key in ("home_team_key", "away_team_key"):
        for code in raw[key].unique():
            resolver.fpl_code(int(code))  # every FPL team code is a known club
    return assemble_fixtures(raw, football_data_results(ctx, resolver))


# --- gameweek ----------------------------------------------------------------------------

GAMEWEEK_COLUMNS = [
    "season",
    "gw",
    "gw_index",
    "deadline_time",
    "deadline_source",
    "first_kickoff",
    "last_kickoff",
    "lockdown_time",
    "event_time",
    "available_at",
]


def _gw_counts_ok(df: pd.DataFrame) -> pd.Series:
    counts = df.groupby("season")["gw"].transform("size")
    expected = df["season"].map(lambda s: GW_COUNT_EXCEPTIONS.get(s, GWS_PER_SEASON))
    return counts == expected


def _deadlines_increasing(df: pd.DataFrame) -> bool:
    ordered = df.sort_values(["season", "gw"])
    return bool(
        ordered.groupby("season")["deadline_time"]
        .apply(lambda s: s.is_monotonic_increasing and s.is_unique)
        .all()
    )


GAMEWEEK_SCHEMA = pa.DataFrameSchema(
    {
        "season": pa.Column("int64", pa.Check.ge(FIRST_SEASON)),
        "gw": pa.Column("int64", pa.Check.in_range(1, 47)),
        "gw_index": pa.Column("int64", pa.Check.in_range(1, GWS_PER_SEASON)),
        "deadline_time": pa.Column(UTC_US),
        "deadline_source": pa.Column(str, pa.Check.isin(["bootstrap", "fixtures_csv", "approx"])),
        "first_kickoff": pa.Column(UTC_US),
        "last_kickoff": pa.Column(UTC_US),
        "lockdown_time": pa.Column(UTC_US),
        "event_time": pa.Column(UTC_US),
        "available_at": pa.Column(UTC_US),
    },
    checks=[
        pa.Check(lambda df: ~df.duplicated(["season", "gw"]), error="(season, gw) unique"),
        pa.Check(_gw_counts_ok, error="GWs per season (38; 2022-23: 37)"),
        pa.Check(_deadlines_increasing, error="deadlines strictly increasing within a season"),
        pa.Check(lambda df: df["deadline_time"] < df["first_kickoff"], error="deadline < kickoff"),
        pa.Check(lambda df: df["first_kickoff"] <= df["last_kickoff"], error="kickoff order"),
        pa.Check(lambda df: df["lockdown_time"] > df["last_kickoff"], error="lockdown order"),
    ],
    strict=True,
    ordered=True,
)
GAMEWEEK_SORT_BY = ("season", "gw")


def bootstrap_events(bootstrap: dict) -> pd.DataFrame:
    """gw, deadline_time, average_entry_score (null until the event is finished)."""
    events = pd.DataFrame(bootstrap["events"])
    finished = events["finished"].astype(bool)
    return pd.DataFrame(
        {
            "gw": events["id"].astype("int64"),
            "deadline_time": to_utc(events["deadline_time"]),
            "average_entry_score": events["average_entry_score"].astype("Int64").where(finished),
        }
    )


def fixtures_csv_deadlines(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Per-event deadlines from a vaastav fixtures.csv that carries `deadline_time`."""
    per_event = fixtures.groupby("event")["deadline_time"]
    if (per_event.nunique() > 1).any():
        raise ValueError("fixtures.csv: deadline_time differs within an event")
    first = per_event.first()
    return pd.DataFrame(
        {"gw": first.index.astype("int64"), "deadline_time": to_utc(first).to_numpy()}
    )


def assemble_gameweeks(
    fixture: pd.DataFrame,
    bootstrap_deadlines: dict[int, pd.DataFrame],
    csv_deadlines: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """gameweek rows from the fixture table plus deadline sources per season
    (bootstrap first, then fixtures.csv, else approximated)."""
    scheduled = fixture[fixture["gw"].notna()]
    gws = (
        scheduled.groupby(["season", "gw"])
        .agg(
            gw_index=("gw_index", "first"),
            first_kickoff=("kickoff_time", "min"),
            last_kickoff=("kickoff_time", "max"),
        )
        .reset_index()
    )
    gws["gw"] = gws["gw"].astype("int64")
    gws["gw_index"] = gws["gw_index"].astype("int64")
    frames = []
    for season, rows in gws.groupby("season", sort=True):
        rows = rows.copy()
        if season in bootstrap_deadlines:
            events = bootstrap_deadlines[season]
            extra = sorted(set(events["gw"]) - set(rows["gw"]))
            if extra:
                log.info(
                    "season %d: bootstrap event(s) %s have no fixtures; skipped", season, extra
                )
            rows = rows.merge(
                events[["gw", "deadline_time"]], on="gw", how="left", validate="one_to_one"
            )
            if rows["deadline_time"].isna().any():
                missing = rows.loc[rows["deadline_time"].isna(), "gw"].tolist()
                raise ValueError(f"season {season}: GW(s) {missing} missing from the bootstrap")
            rows["deadline_source"] = "bootstrap"
        elif season in csv_deadlines:
            rows = rows.merge(csv_deadlines[season], on="gw", how="left", validate="one_to_one")
            if rows["deadline_time"].isna().any():
                raise ValueError(f"season {season}: fixtures.csv lacks deadlines for some GWs")
            rows["deadline_source"] = "fixtures_csv"
        else:
            rows["deadline_time"] = rows["first_kickoff"] - APPROX_DEADLINE_LEAD
            rows["deadline_source"] = "approx"
        frames.append(rows)
    df = pd.concat(frames, ignore_index=True)
    df["season"] = df["season"].astype("int64")
    df["deadline_time"] = df["deadline_time"].astype(UTC_US)
    df["first_kickoff"] = df["first_kickoff"].astype(UTC_US)
    df["last_kickoff"] = df["last_kickoff"].astype(UTC_US)
    df["deadline_source"] = df["deadline_source"].astype(str)
    df["lockdown_time"] = lockdown_times(df["last_kickoff"])
    df["event_time"] = df["deadline_time"]
    df["available_at"] = df["deadline_time"]
    return df[GAMEWEEK_COLUMNS]


def season_bootstrap_events(ctx: BuildContext, seasons: set[int]) -> dict[int, pd.DataFrame]:
    """`bootstrap_events` of the newest bootstrap of each of `seasons` that has one."""
    return {
        season: bootstrap_events(RawStore.read_json(path))
        for season, (_, path, _) in last_bootstrap_per_season(ctx.store).items()
        if season in seasons
    }


def build_gameweek(ctx: BuildContext) -> pd.DataFrame:
    fixture = ctx.table("fixture")
    seasons = set(fixture["season"].unique())
    bootstrap_deadlines = season_bootstrap_events(ctx, seasons)
    csv_deadlines = {}
    for season, season_dir in vaastav_seasons(vaastav_run(ctx)).items():
        path = season_dir / "fixtures.csv.gz"
        if season in seasons and season not in bootstrap_deadlines and path.exists():
            fixtures = read_raw_csv(path)
            if "deadline_time" in fixtures.columns:
                csv_deadlines[season] = fixtures_csv_deadlines(fixtures)
    return assemble_gameweeks(fixture, bootstrap_deadlines, csv_deadlines)


# --- gameweek_result ---------------------------------------------------------------------

GAMEWEEK_RESULT_COLUMNS = ["season", "gw", "average_entry_score", "event_time", "available_at"]

GAMEWEEK_RESULT_SCHEMA = pa.DataFrameSchema(
    {
        "season": pa.Column("int64", pa.Check.ge(FIRST_SEASON)),
        "gw": pa.Column("int64", pa.Check.in_range(1, 47)),
        "average_entry_score": pa.Column("Int64", pa.Check.ge(0), nullable=True),
        "event_time": pa.Column(UTC_US),
        "available_at": pa.Column(UTC_US),
    },
    checks=[
        pa.Check(lambda df: ~df.duplicated(["season", "gw"]), error="(season, gw) unique"),
        pa.Check(lambda df: df["available_at"] == df["event_time"], error="available_at"),
    ],
    strict=True,
    ordered=True,
)
GAMEWEEK_RESULT_SORT_BY = ("season", "gw")


def assemble_gameweek_results(
    gameweek: pd.DataFrame, bootstrap_events_by_season: dict[int, pd.DataFrame]
) -> pd.DataFrame:
    """One row per gameweek row: average_entry_score (bootstrap seasons, finished events;
    else null), timed at the GW's lockdown."""
    events = [
        frame[["gw", "average_entry_score"]].assign(season=season)
        for season, frame in bootstrap_events_by_season.items()
    ]
    scores = (
        pd.concat(events, ignore_index=True)
        if events
        else pd.DataFrame({"gw": [], "average_entry_score": [], "season": []})
    ).astype({"season": "int64", "gw": "int64", "average_entry_score": "Int64"})
    df = gameweek[["season", "gw", "lockdown_time"]].merge(
        scores, on=["season", "gw"], how="left", validate="one_to_one"
    )
    df["average_entry_score"] = df["average_entry_score"].astype("Int64")
    df["event_time"] = df["lockdown_time"].astype(UTC_US)
    df["available_at"] = df["event_time"]
    return df[GAMEWEEK_RESULT_COLUMNS]


def build_gameweek_result(ctx: BuildContext) -> pd.DataFrame:
    gameweek = ctx.table("gameweek")
    events = season_bootstrap_events(ctx, set(gameweek["season"].unique()))
    return assemble_gameweek_results(gameweek, events)
