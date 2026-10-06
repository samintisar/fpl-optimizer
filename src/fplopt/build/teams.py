"""`team_dim` from the hand-maintained `config/teams.csv`, name -> team_key resolvers per
source, and a cross-source coverage check that fails the build on any unknown club.

`team_key` is the FPL team `code` (stable per club across seasons); clubs that only appear in
football-data before 2016/17 (Elo burn-in) get synthetic keys from 1001. Config columns:
`team_key, short_name, fpl_names (every FPL name for the code, ';'-separated), football_data,
understat, odds_api` — the last two only where the source has the club (blank otherwise).

Coverage (`validate_team_coverage`), all unknowns reported at once:
- FPL: every `team_code` in vaastav `players_raw` and every bootstrap `teams[].code` is a key;
  every FPL name seen for a code (vaastav `teams.csv`, vaastav `master_team_list` joined via
  `players_raw`, bootstrap `teams[].name`) is in its `fpl_names`. Bootstraps are sampled: the
  first and last snapshot of each calendar month per archive (~130 of ~8,000 files, a few
  seconds instead of minutes). Teams only change between seasons (summer), so every season's
  teams are seen; a mid-month rename would surface at the month's last snapshot.
- football-data: `HomeTeam`/`AwayTeam` of the newest file of every season present (rows
  without a `Date` are dropped — 2014/15 has a trailing empty row).
- Understat: team names from the team-file names (`understat_<Team>.csv`, `_` = space; the
  aggregate `understat_player`/`understat_team` files are skipped), plus every player-file
  match with exactly one side resolving: player files are career logs across leagues, but a
  match against a known EPL club is an EPL match, so its other side must resolve too (reads
  only `h_team`/`a_team`; ~2,800 small files, a few seconds).
- The Odds API: `home_team`/`away_team` of every archived `odds/soccer_epl` snapshot.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from fplopt.build.common import (
    EPOCH,
    UTC_US,
    BuildContext,
    fplcache_and_own_bootstraps,
    latest_complete_run,
    read_raw_csv,
)
from fplopt.ingest.raw_store import RawStore
from fplopt.seasons import parse_season_label

log = logging.getLogger(__name__)

CONFIG_FILE = "teams.csv"
CONFIG_COLUMNS = ("team_key", "short_name", "fpl_names", "football_data", "understat", "odds_api")
SYNTHETIC_KEY_MIN = 1000
UNDERSTAT_AGGREGATE_FILES = frozenset({"understat_player", "understat_team"})
MAX_WHERE_SHOWN = 3

# Resolver sources: config column -> team_dim column / label used in messages.
_SOURCES = {
    "football_data": ("football_data_name", "football-data"),
    "understat": ("understat_name", "Understat"),
    "odds_api": ("odds_api_name", "Odds API"),
}


def _unique_when_present(series: pd.Series) -> bool:
    return not series.dropna().duplicated().any()


SCHEMA = pa.DataFrameSchema(
    {
        "team_key": pa.Column("int64", pa.Check.gt(0), unique=True),
        "short_name": pa.Column(str, pa.Check.str_matches(r"^[A-Z]{3}$"), unique=True),
        "fpl_names": pa.Column(str, nullable=True),
        "football_data_name": pa.Column(str, unique=True),
        "understat_name": pa.Column(str, pa.Check(_unique_when_present), nullable=True),
        "odds_api_name": pa.Column(str, pa.Check(_unique_when_present), nullable=True),
        "in_fpl": pa.Column(bool),
        "event_time": pa.Column(UTC_US),
        "available_at": pa.Column(UTC_US),
    },
    checks=[
        pa.Check(
            lambda df: df["in_fpl"] == (df["team_key"] < SYNTHETIC_KEY_MIN),
            error="in_fpl must equal team_key < 1000",
        ),
        pa.Check(
            lambda df: df["in_fpl"] == df["fpl_names"].notna(),
            error="FPL clubs (and only they) have fpl_names",
        ),
    ],
    strict=True,
    ordered=True,
)
SORT_BY = ("team_key",)


class TeamCoverageError(ValueError):
    """A team code or name seen in raw/ is missing from config/teams.csv."""


# --- config ------------------------------------------------------------------------------


def read_teams_config(path: Path) -> pd.DataFrame:
    """Parse and check config/teams.csv: config columns, blanks as None, team_key int.
    Raises ValueError listing every problem (bad keys, duplicates, synthetic clubs with FPL
    names or FPL clubs without)."""
    path = Path(path)
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if tuple(reader.fieldnames or ()) != CONFIG_COLUMNS:
            raise ValueError(
                f"{path}: columns must be {','.join(CONFIG_COLUMNS)}, got {reader.fieldnames}"
            )
        rows = [{k: (v or "").strip() or None for k, v in row.items()} for row in reader]
    problems: list[str] = []
    seen: dict[tuple[str, str], int] = {}
    for line, row in enumerate(rows, start=2):
        key_text = row["team_key"] or ""
        if not key_text.isdigit() or int(key_text) <= 0:
            problems.append(f"line {line}: bad team_key {key_text!r}")
            continue
        key = int(key_text)
        row["team_key"] = key
        names = [n.strip() for n in (row["fpl_names"] or "").split(";") if n.strip()]
        row["fpl_names"] = ";".join(names) or None
        synthetic = key >= SYNTHETIC_KEY_MIN
        if synthetic and row["fpl_names"]:
            problems.append(f"team_key {key}: synthetic (>= 1000) but has fpl_names")
        if not synthetic and not row["fpl_names"]:
            problems.append(f"team_key {key}: FPL club without fpl_names")
        if not row["football_data"]:
            problems.append(f"team_key {key}: no football_data name")
        values = [(c, row[c]) for c in ("short_name", "football_data", "understat", "odds_api")]
        values = [("team_key", str(key)), *((c, v) for c, v in values if v)]
        values += [("fpl_names", name) for name in names]
        for column, value in values:
            other = seen.setdefault((column, value), key)
            if other != key:
                problems.append(f"{column} {value!r} used by team_key {other} and {key}")
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    df = pd.DataFrame(rows, columns=list(CONFIG_COLUMNS))
    df["team_key"] = df["team_key"].astype("int64")
    return df


def team_dim_from_config(path: Path) -> pd.DataFrame:
    """The `team_dim` table for a teams.csv (static reference data: event_time =
    available_at = 1970-01-01 UTC)."""
    config = read_teams_config(path)
    n = len(config)
    return pd.DataFrame(
        {
            "team_key": config["team_key"],
            "short_name": config["short_name"],
            "fpl_names": config["fpl_names"],
            "football_data_name": config["football_data"],
            "understat_name": config["understat"],
            "odds_api_name": config["odds_api"],
            "in_fpl": config["team_key"] < SYNTHETIC_KEY_MIN,
            "event_time": pd.Series([EPOCH] * n, dtype=UTC_US),
            "available_at": pd.Series([EPOCH] * n, dtype=UTC_US),
        }
    )


# --- resolver ----------------------------------------------------------------------------


class TeamResolver:
    """Name/code -> team_key lookups per source. The named methods raise KeyError naming the
    unknown value; `find(source, name)` returns None instead (source: 'football_data',
    'understat', 'odds_api' or 'fpl_name')."""

    def __init__(self, team_dim: pd.DataFrame) -> None:
        self._maps: dict[str, dict[str, int]] = {}
        for source, (column, _) in _SOURCES.items():
            present = team_dim[team_dim[column].notna()]
            self._maps[source] = dict(
                zip(present[column], present["team_key"].astype(int), strict=True)
            )
        fpl = team_dim[team_dim["in_fpl"]]
        self._maps["fpl_name"] = {
            name: int(key)
            for key, names in zip(fpl["team_key"], fpl["fpl_names"], strict=True)
            for name in names.split(";")
        }
        self._fpl_codes = frozenset(int(key) for key in fpl["team_key"])
        self._fpl_names = {
            int(key): frozenset(names.split(";"))
            for key, names in zip(fpl["team_key"], fpl["fpl_names"], strict=True)
        }

    def find(self, source: str, name: str) -> int | None:
        return self._maps[source].get(name)

    def _get(self, source: str, label: str, name: str) -> int:
        key = self.find(source, name)
        if key is None:
            raise KeyError(f"unknown {label} team name {name!r} (config/teams.csv)")
        return key

    def football_data(self, name: str) -> int:
        return self._get("football_data", "football-data", name)

    def understat(self, name: str) -> int:
        return self._get("understat", "Understat", name)

    def odds_api(self, name: str) -> int:
        return self._get("odds_api", "Odds API", name)

    def fpl_name(self, name: str) -> int:
        return self._get("fpl_name", "FPL", name)

    def fpl_code(self, code: int) -> int:
        if int(code) not in self._fpl_codes:
            raise KeyError(f"unknown FPL team code {code} (config/teams.csv)")
        return int(code)

    def fpl_names(self, code: int) -> frozenset[str]:
        """Every FPL name configured for an FPL team code."""
        return self._fpl_names[self.fpl_code(code)]


# --- coverage ----------------------------------------------------------------------------


def _vaastav_seasons(run: Path) -> list[tuple[str, Path]]:
    seasons = []
    for path in sorted(run.iterdir()):
        if path.is_dir():
            try:
                parse_season_label(path.name)
            except ValueError:
                continue
            seasons.append((path.name, path))
    return seasons


def _fpl_teams(store: RawStore, run: Path) -> Iterator[tuple[int, str | None, str]]:
    """(team code, FPL name or None, where) from vaastav and sampled bootstraps."""
    master_path = run / "master_team_list.csv.gz"
    master = read_raw_csv(master_path) if master_path.exists() else None
    for season, path in _vaastav_seasons(run):
        players_path = path / "players_raw.csv.gz"
        if players_path.exists():
            players = read_raw_csv(players_path, usecols=["team", "team_code"])
            codes = players.drop_duplicates()
            where = f"vaastav {season} players_raw"
            for code in codes["team_code"]:
                yield int(code), None, where
            if master is not None:
                named = codes.merge(master[master["season"] == season], on="team")
                for code, name in zip(named["team_code"], named["team_name"], strict=True):
                    yield int(code), str(name), f"vaastav master_team_list {season}"
        teams_path = path / "teams.csv.gz"
        if teams_path.exists():
            teams = read_raw_csv(teams_path, usecols=["code", "name"])
            for code, name in zip(teams["code"], teams["name"], strict=True):
                yield int(code), str(name), f"vaastav {season} teams"
    for taken_at, path, source in _sample_bootstraps(store):
        where = f"{source} bootstrap {taken_at:%Y-%m-%dT%H%MZ}"
        for team in RawStore.read_json(path)["teams"]:
            yield int(team["code"]), str(team["name"]), where


def _sample_bootstraps(store: RawStore) -> list[tuple[datetime, Path, str]]:
    """First and last bootstrap of each calendar month in each archive (see module doc)."""
    months: dict[tuple[str, int, int], list[tuple[datetime, Path, str]]] = {}
    for entry in fplcache_and_own_bootstraps(store):
        taken_at, _, source = entry
        months.setdefault((source, taken_at.year, taken_at.month), []).append(entry)
    sample = []
    for entries in months.values():
        sample.append(entries[0])
        if len(entries) > 1:
            sample.append(entries[-1])
    return sorted(sample, key=lambda entry: (entry[0], entry[2]))


def _football_data_names(store: RawStore) -> Iterator[tuple[str, str]]:
    base = store.root / "football-data" / "E0"
    if not base.is_dir():
        return
    for season_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        path = store.latest("football-data", f"E0/{season_dir.name}", suffix=".csv.gz")
        if path is None:
            continue
        df = read_raw_csv(path, usecols=["Date", "HomeTeam", "AwayTeam"])
        df = df.dropna(subset=["Date"])
        for name in sorted(set(df["HomeTeam"]) | set(df["AwayTeam"])):
            yield str(name), f"football-data E0/{season_dir.name}"


def _understat_names(run: Path, resolver: TeamResolver) -> Iterator[tuple[str, str]]:
    """Understat names that must resolve: team-file names, and the unresolved side of
    player-file matches whose other side resolves."""
    for path in sorted(run.glob("*/understat/*.csv.gz")):
        stem = path.name.removesuffix(".csv.gz")
        season = path.parent.parent.name
        if stem.startswith("understat_"):
            if stem not in UNDERSTAT_AGGREGATE_FILES:
                yield stem.removeprefix("understat_").replace("_", " "), f"vaastav {season} {stem}"
            continue
        df = read_raw_csv(path, usecols=["h_team", "a_team"])
        for home, away in set(zip(df["h_team"], df["a_team"], strict=True)):
            home_known = resolver.find("understat", home) is not None
            away_known = resolver.find("understat", away) is not None
            if home_known != away_known:
                yield (away if home_known else home), f"vaastav {season} understat/{stem}"


def _odds_api_names(store: RawStore) -> Iterator[tuple[str, str]]:
    for taken_at, path in store.entries("odds", "soccer_epl"):
        for event in RawStore.read_json(path):
            where = f"odds soccer_epl {taken_at:%Y-%m-%dT%H%MZ}"
            yield str(event["home_team"]), where
            yield str(event["away_team"]), where


def validate_team_coverage(ctx: BuildContext, team_dim: pd.DataFrame) -> None:
    """Fail (TeamCoverageError, one line listing every unknown) if any team code or name in
    raw/ is missing from `team_dim`. Sources and sampling: see the module docstring."""
    resolver = TeamResolver(team_dim)
    problems: dict[str, set[str]] = {}

    def flag(problem: str, where: str) -> None:
        problems.setdefault(problem, set()).add(where)

    run = latest_complete_run(ctx.store, "vaastav", "data")
    for code, name, where in _fpl_teams(ctx.store, run):
        try:
            known = resolver.fpl_names(code)
        except KeyError:
            flag(f"FPL team code {code}", where)
            continue
        if name is not None and name not in known:
            flag(f"FPL name {name!r} for team code {code}", where)
    for name, where in _football_data_names(ctx.store):
        if resolver.find("football_data", name) is None:
            flag(f"football-data name {name!r}", where)
    for name, where in _understat_names(run, resolver):
        if resolver.find("understat", name) is None:
            flag(f"Understat name {name!r}", where)
    for name, where in _odds_api_names(ctx.store):
        if resolver.find("odds_api", name) is None:
            flag(f"Odds API name {name!r}", where)
    if problems:
        details = []
        for problem in sorted(problems):
            wheres = sorted(problems[problem])
            more = (
                f" +{len(wheres) - MAX_WHERE_SHOWN} more" if len(wheres) > MAX_WHERE_SHOWN else ""
            )
            details.append(f"{problem} ({', '.join(wheres[:MAX_WHERE_SHOWN])}{more})")
        raise TeamCoverageError(
            f"{len(problems)} unknown team code(s)/name(s), add them to config/teams.csv: "
            + "; ".join(details)
        )


def build_team_dim(ctx: BuildContext) -> pd.DataFrame:
    """team_dim from `<config_dir>/teams.csv`, after the cross-source coverage check."""
    team_dim = team_dim_from_config(Path(ctx.config_dir) / CONFIG_FILE)
    validate_team_coverage(ctx, team_dim)
    return team_dim
