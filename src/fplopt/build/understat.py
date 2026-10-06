"""Understat (vaastav mirror) -> `understat_player_match`, the `understat_map`
(understat_id <-> player_key), `us_*` columns on `player_match`, `player_dim.understat_id`,
and `team_match`.

Player files: `<season>/understat/<Name>_<understat_id>.csv` in the 2021-22 … 2024-25
folders are career match logs across leagues. A row is an EPL match when both `h_team` and
`a_team` are Understat names in team_dim; it maps to `fixture_key` by (Understat `season`,
home, away), unique per season. EPL rows before 2016/17 have no fixture and are dropped
(counted in the log). The same (understat_id, match) appears in several folders with equal
values (±5e-6): the newest folder wins. Names come from the folders' `understat_player`
aggregates (HTML entities unescaped), else the file name. `event_time` = the fixture's
kickoff, `available_at` = the fixture's `available_at` (GW lockdown). understat_map is
reference data like team_dim (`event_time` = `available_at` = 1970-01-01).

Mapping understat_id -> player_key (all seasons at once):
- Candidates: an Understat appearance and an FPL appearance (minutes > 0) in the same
  fixture. `together` = such common fixtures; `shared` = those with |us_minutes − minutes|
  ≤ 5; with us_apps = the Understat id's EPL appearances and fpl_apps = the FPL player's
  appearances in fixtures that have any Understat row: `overlap` = shared / min(us_apps,
  fpl_apps) and `coverage` = shared / max(us_apps, fpl_apps); `name` = rapidfuzz
  token_set_ratio of normalised names (unescaped, accents stripped/transliterated, lower
  case, letters/digits only), best over the FPL full names and web_names of all seasons.
- Accept (any): overlap ≥ 0.8 and shared ≥ 3; overlap ≥ 0.5 and name ≥ 85; together ≥ 1 and
  name ≥ 95. Greedy one-to-one by (coverage, shared, name) descending.
- Overrides: `config/overrides.csv` rows with source=understat are applied last and win:
  `source_id` + `player_key` = force the pair; `player_key` blank = the Understat id maps to
  nobody; `source_id` blank = the FPL player has no Understat id (exempt from validation).
- Validation (fails the build, listing offenders): one-to-one; agreement with vaastav's
  id_dict (2021-22, 2022-23; FPL_ID -> player_key via player_season) is 100% — no id_dict
  pair contradicted, and every pair whose players both appear (Understat EPL rows, FPL
  minutes) is mapped; every FPL player with minutes > 0 in 2020/21 – 2023/24 is mapped
  (the mirror covers those seasons fully: team files for every side, a player file for
  every FPL player with minutes).

Results on raw/ 2026-10-06 (2,805 player files, 1,213 Understat ids with EPL rows from
2016/17 on; 7,190 EPL rows from 2014/15–2015/16 have no fixture): all 1,213 mapped, no
overrides; 1,051 by the minutes rule, 158 by overlap + name, 4 by name alone (2 of them —
Klaesson 2021-22 and Ben Parkinson 2023-24, single substitute appearances — share no
fixture within 5 minutes: Understat 24'/18' vs FPL 35'/24'); 2 mapped with name < 85
(Zambo Anguissa as "Franck Zambo", "Vinicius Souza" as "Vini de Souza Costa"; minutes agree
in 58/58 and 35/36). id_dict: 1,229 rows (818 distinct pairs), 1,205 rows where both
players appear — all agree; the other 24 never played. Share of FPL minutes with Understat
data: 2016 47%, 2017 62%, 2018 75%, 2019 91%, 2020–2023 100%, 2024 79% (mirror stops
2025-04-07), 2025– 0%.
Deviations from the plan's starting point, both forced by the oracle/coverage checks:
ranking by `overlap` (min-based) lets a one-appearance FPL player claim a regular's
Understat id (6 id_dict contradictions, e.g. Benteke, Grealish, Mitrović), so the greedy
ranks by two-sided `coverage`; and "shared ≥ 1 and name ≥ 95" left the two substitutes
above unmapped, so the name rule needs only a common fixture (`together ≥ 1`).

team_match: two rows per finished fixture. Understat team files (`understat_<Team>.csv`,
2019-20 … 2024-25 folders; each folder also has ~3 stale previous-season files) are pooled
and joined on (team_key, is_home, UK-local kickoff date); a row is preferred from the
folder of the fixture's season, else the newest folder. `scored`/`missed` must equal the
goals. Measured: 4,874 team-file rows, all join a fixture with matching goals; 4,418 sides
after dropping repeats (2019/20–2023/24 every side, 2024/25 618 of 760). `fd_xg`/`fd_xga`
from football-data `HxG`/`AxG` (2026-27). `fpl_xg`/`fpl_xga` = sum of the side's players'
`fpl_xg` (null if any player row lacks it, i.e. before 2022/23 GW16).
"""

from __future__ import annotations

import html
import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from rapidfuzz import fuzz

from fplopt.build.common import EPOCH, UK, UTC_US, BuildContext, read_raw_csv, write_table
from fplopt.build.fixtures import FIRST_SEASON, vaastav_run
from fplopt.build.players import (
    MAX_MINUTES,
    PLAYER_DIM_COLUMNS,
    PLAYER_DIM_SCHEMA,
    PLAYER_DIM_SORT_BY,
    PLAYER_MATCH_SCHEMA,
    PLAYER_MATCH_SORT_BY,
    UNDERSTAT_COLUMNS,
)
from fplopt.build.teams import UNDERSTAT_AGGREGATE_FILES, TeamResolver
from fplopt.seasons import parse_season_label

log = logging.getLogger(__name__)

MINUTES_TOLERANCE = 5
# Accepted if (overlap >= 0.8 and shared >= 3) or (overlap >= 0.5 and name >= 85) or
# (together >= 1 and name >= 95); see the module docstring.
ACCEPT_RULES = ((0.8, 3), (0.5, 85.0), (1, 95.0))
VALIDATED_SEASONS = range(2020, 2024)
OVERRIDES_FILE = "overrides.csv"

US_SOURCE_COLUMNS = {
    "time": "us_minutes",
    "goals": "us_goals",
    "npg": "us_npg",
    "xG": "us_xg",
    "npxG": "us_npxg",
    "xA": "us_xa",
    "assists": "us_assists",
    "shots": "us_shots",
    "key_passes": "us_key_passes",
    "position": "us_position",
}


class UnderstatMappingError(ValueError):
    """The Understat <-> FPL mapping failed validation."""


# --- reading -----------------------------------------------------------------------------


def _folder_season(path: Path) -> int:
    return parse_season_label(path.parent.parent.name)


def read_player_files(run: Path) -> pd.DataFrame:
    """Every Understat player-file row (all leagues) + understat_id and folder season."""
    frames = []
    for path in sorted(run.glob("*/understat/*.csv.gz")):
        stem = path.name.removesuffix(".csv.gz")
        if stem.startswith("understat_"):
            continue
        df = read_raw_csv(path)
        df["understat_id"] = int(stem.rsplit("_", 1)[1])
        df["folder_season"] = _folder_season(path)
        df["file_name"] = stem.rsplit("_", 1)[0].replace("_", " ")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def read_player_names(run: Path) -> dict[int, str]:
    """understat_id -> name from the folders' `understat_player` aggregates (newest wins)."""
    names: dict[int, str] = {}
    for path in sorted(run.glob("*/understat/understat_player.csv.gz")):
        df = read_raw_csv(path, usecols=["id", "player_name"])
        names.update(zip(df["id"].astype(int), df["player_name"].astype(str), strict=True))
    return {key: html.unescape(name) for key, name in names.items()}


def read_team_files(run: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(run.glob("*/understat/understat_*.csv.gz")):
        stem = path.name.removesuffix(".csv.gz")
        if stem in UNDERSTAT_AGGREGATE_FILES:
            continue
        df = read_raw_csv(
            path, usecols=["h_a", "xG", "xGA", "npxG", "npxGA", "scored", "missed", "date"]
        )
        df["team_name"] = stem.removeprefix("understat_").replace("_", " ")
        df["folder_season"] = _folder_season(path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# --- understat_player_match --------------------------------------------------------------

UPM_COLUMNS = [
    "understat_id",
    "fixture_key",
    "season",
    "us_match_id",
    *US_SOURCE_COLUMNS.values(),
    "understat_name",
    "event_time",
    "available_at",
]

UPM_SCHEMA = pa.DataFrameSchema(
    {
        "understat_id": pa.Column("int64"),
        "fixture_key": pa.Column("int64"),
        "season": pa.Column("int64", pa.Check.ge(FIRST_SEASON)),
        "us_match_id": pa.Column("int64"),
        "us_minutes": pa.Column("int64", pa.Check.in_range(0, MAX_MINUTES)),
        "us_goals": pa.Column("int64", pa.Check.ge(0)),
        "us_npg": pa.Column("int64", pa.Check.ge(0)),
        "us_xg": pa.Column("float64", pa.Check.ge(0)),
        "us_npxg": pa.Column("float64", pa.Check.ge(0)),
        "us_xa": pa.Column("float64", pa.Check.ge(0)),
        "us_assists": pa.Column("int64", pa.Check.ge(0)),
        "us_shots": pa.Column("int64", pa.Check.ge(0)),
        "us_key_passes": pa.Column("int64", pa.Check.ge(0)),
        "us_position": pa.Column(str),
        "understat_name": pa.Column(str),
        "event_time": pa.Column(UTC_US),
        "available_at": pa.Column(UTC_US),
    },
    checks=[
        pa.Check(
            lambda df: ~df.duplicated(["understat_id", "fixture_key"]),
            error="(understat_id, fixture_key) unique",
        ),
        pa.Check(lambda df: df["available_at"] > df["event_time"], error="available_at order"),
        pa.Check(
            lambda df: df.groupby("us_match_id")["fixture_key"].transform("nunique") == 1,
            error="us_match_id -> one fixture",
        ),
    ],
    strict=True,
    ordered=True,
)
UPM_SORT_BY = ("fixture_key", "understat_id")


def understat_player_match(
    raw: pd.DataFrame, names: dict[int, str], fixture: pd.DataFrame, resolver: TeamResolver
) -> pd.DataFrame:
    """EPL rows of the player files, mapped to fixtures and deduplicated (newest folder)."""
    home = raw["h_team"].map(lambda n: resolver.find("understat", n))
    away = raw["a_team"].map(lambda n: resolver.find("understat", n))
    epl = raw[home.notna() & away.notna()].assign(
        home_team_key=home.astype("Int64"), away_team_key=away.astype("Int64")
    )
    epl = epl.sort_values("folder_season", ascending=False, kind="mergesort")
    epl = epl.drop_duplicates(["understat_id", "id"])
    keys = fixture[
        ["season", "home_team_key", "away_team_key", "fixture_key", "kickoff_time", "available_at"]
    ]
    merged = epl.merge(
        keys.astype({"home_team_key": "Int64", "away_team_key": "Int64"}),
        on=["season", "home_team_key", "away_team_key"],
        how="left",
        validate="many_to_one",
    )
    no_fixture = merged["fixture_key"].isna()
    log.info(
        "understat: %d player rows, %d EPL (after dedupe), %d without a fixture (seasons %s)",
        len(raw),
        len(epl),
        int(no_fixture.sum()),
        sorted(merged.loc[no_fixture, "season"].unique().tolist()),
    )
    merged = merged[~no_fixture]
    us_dates = pd.to_datetime(merged["date"]).dt.normalize()
    kickoff_dates = merged["kickoff_time"].dt.tz_convert(UK).dt.tz_localize(None).dt.normalize()
    off = (us_dates - kickoff_dates).dt.days.abs() > 1
    if off.any():
        bad = merged.loc[off, ["understat_id", "id", "date", "kickoff_time"]]
        raise UnderstatMappingError(
            f"{len(bad)} Understat row(s) dated away from their fixture:\n{bad.head(10)}"
        )
    out = merged.rename(columns={"id": "us_match_id", **US_SOURCE_COLUMNS})
    out["understat_name"] = [
        names.get(uid, file_name)
        for uid, file_name in zip(out["understat_id"], out["file_name"], strict=True)
    ]
    out["fixture_key"] = out["fixture_key"].astype("int64")
    out["us_position"] = out["us_position"].astype(str)
    out["event_time"] = out["kickoff_time"].astype(UTC_US)
    out["available_at"] = out["available_at"].astype(UTC_US)
    return out[UPM_COLUMNS].reset_index(drop=True)


# --- mapping -----------------------------------------------------------------------------


# Letters NFKD doesn't decompose into an ASCII base letter.
_TRANSLITERATE = str.maketrans(
    {"ø": "o", "æ": "ae", "œ": "oe", "ß": "ss", "ł": "l", "đ": "d", "ð": "d", "þ": "th", "ı": "i"}
)


def normalise_name(name: str) -> str:
    """HTML-unescaped, accents stripped, lower case, letters/digits/spaces only."""
    text = unicodedata.normalize("NFKD", html.unescape(str(name)).lower())
    text = "".join(c for c in text if not unicodedata.combining(c)).translate(_TRANSLITERATE)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def fpl_appearances(player_match: pd.DataFrame) -> pd.DataFrame:
    """FPL appearances (minutes > 0): player_key, fixture_key, season, minutes."""
    played = player_match[player_match["minutes"] > 0]
    return played[["player_key", "fixture_key", "season", "minutes"]].reset_index(drop=True)


def fpl_name_variants(player_season: pd.DataFrame) -> dict[int, set[str]]:
    """player_key -> every FPL full name and web_name seen across seasons."""
    names: dict[int, set[str]] = {}
    for key, first, second, web in zip(
        player_season["player_key"],
        player_season["first_name"],
        player_season["second_name"],
        player_season["web_name"],
        strict=True,
    ):
        names.setdefault(int(key), set()).update({f"{first} {second}", str(web)})
    return names


def candidate_pairs(upm: pd.DataFrame, fpl_apps: pd.DataFrame) -> pd.DataFrame:
    """understat_id x player_key pairs that appeared in at least one common fixture:
    together (common fixtures), shared (… with |minutes diff| <= 5), us_apps, fpl_apps (the
    FPL player's appearances in fixtures with any Understat row), overlap (shared / min of the
    two) and coverage (shared / max of the two)."""
    us = upm[["understat_id", "fixture_key", "us_minutes"]]
    joined = us.merge(fpl_apps[["player_key", "fixture_key", "minutes"]], on="fixture_key")
    joined["close"] = (joined["us_minutes"] - joined["minutes"]).abs() <= MINUTES_TOLERANCE
    pairs = (
        joined.groupby(["understat_id", "player_key"])["close"]
        .agg(together="size", shared="sum")
        .reset_index()
    )
    us_apps = us.groupby("understat_id").size().rename("us_apps")
    covered = fpl_apps[fpl_apps["fixture_key"].isin(set(us["fixture_key"]))]
    fpl_counts = covered.groupby("player_key").size().rename("fpl_apps")
    pairs = pairs.join(us_apps, on="understat_id").join(fpl_counts, on="player_key")
    apps = pairs[["us_apps", "fpl_apps"]]
    pairs["overlap"] = pairs["shared"] / apps.min(axis=1)
    pairs["coverage"] = pairs["shared"] / apps.max(axis=1)
    return pairs


def name_scores(
    pairs: pd.DataFrame, us_names: dict[int, str], fpl_names: dict[int, set[str]]
) -> pd.Series:
    """Best token_set_ratio between the pair's Understat name and any FPL name variant."""
    us_norm = {uid: normalise_name(name) for uid, name in us_names.items()}
    fpl_norm = {key: {normalise_name(n) for n in names} for key, names in fpl_names.items()}
    cache: dict[tuple[str, str], float] = {}

    def score(uid: int, key: int) -> float:
        best = 0.0
        a = us_norm.get(uid, "")
        for b in fpl_norm.get(key, ()):
            if (a, b) not in cache:
                cache[(a, b)] = fuzz.token_set_ratio(a, b)
            best = max(best, cache[(a, b)])
        return best

    return pd.Series(
        [score(u, k) for u, k in zip(pairs["understat_id"], pairs["player_key"], strict=True)],
        index=pairs.index,
        dtype="float64",
    )


def acceptable(pairs: pd.DataFrame) -> pd.Series:
    """Pairs meeting any of ACCEPT_RULES (see module docstring)."""
    (min_overlap_1, min_shared), (min_overlap_2, min_name_2), (min_together, min_name_3) = (
        ACCEPT_RULES
    )
    return (
        ((pairs["overlap"] >= min_overlap_1) & (pairs["shared"] >= min_shared))
        | ((pairs["overlap"] >= min_overlap_2) & (pairs["name_score"] >= min_name_2))
        | ((pairs["together"] >= min_together) & (pairs["name_score"] >= min_name_3))
    )


def greedy_assign(pairs: pd.DataFrame) -> pd.DataFrame:
    """One-to-one assignment of acceptable pairs by (coverage, shared, name) descending."""
    ranked = pairs[acceptable(pairs)].sort_values(
        ["coverage", "shared", "name_score", "understat_id", "player_key"],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    )
    used_us: set[int] = set()
    used_fpl: set[int] = set()
    keep = []
    for idx, uid, key in zip(
        ranked.index, ranked["understat_id"], ranked["player_key"], strict=True
    ):
        if uid in used_us or key in used_fpl:
            continue
        used_us.add(uid)
        used_fpl.add(key)
        keep.append(idx)
    return ranked.loc[keep]


def read_overrides(config_dir: Path) -> pd.DataFrame:
    """Understat rows of config/overrides.csv as understat_id, player_key (both Int64):
    both set = force the pair; player_key blank = that Understat id maps to nobody;
    source_id blank = that FPL player has no Understat id."""
    path = Path(config_dir) / OVERRIDES_FILE
    out = pd.DataFrame(
        {"understat_id": pd.Series(dtype="Int64"), "player_key": pd.Series(dtype="Int64")}
    )
    if not path.exists():
        return out
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df[df["source"].str.strip() == "understat"]
    if df.empty:
        return out

    def ints(values: pd.Series) -> pd.arrays.IntegerArray:
        return pd.array([int(v) if v else pd.NA for v in values.str.strip()], dtype="Int64")

    out = pd.DataFrame(
        {"understat_id": ints(df["source_id"]), "player_key": ints(df["player_key"])}
    )
    both_blank = out["understat_id"].isna() & out["player_key"].isna()
    if both_blank.any() or out.dropna(subset=["understat_id"])["understat_id"].duplicated().any():
        raise ValueError(f"{path}: understat overrides need a unique source_id or a player_key")
    if out.dropna(subset=["player_key"])["player_key"].duplicated().any():
        raise ValueError(f"{path}: player_key overridden twice for understat")
    return out.reset_index(drop=True)


MAP_COLUMNS = [
    "understat_id",
    "player_key",
    "method",
    "shared",
    "overlap",
    "coverage",
    "name_score",
    "event_time",
    "available_at",
]

MAP_SCHEMA = pa.DataFrameSchema(
    {
        "understat_id": pa.Column("int64", unique=True),
        "player_key": pa.Column(
            "Int64", pa.Check(lambda s: ~s.duplicated() | s.isna()), nullable=True
        ),
        "method": pa.Column(str, pa.Check.isin(["auto", "override"])),
        "shared": pa.Column("Int64", nullable=True),
        "overlap": pa.Column("Float64", nullable=True),
        "coverage": pa.Column("Float64", nullable=True),
        "name_score": pa.Column("Float64", nullable=True),
        "event_time": pa.Column(UTC_US, pa.Check.eq(EPOCH)),
        "available_at": pa.Column(UTC_US, pa.Check.eq(EPOCH)),
    },
    strict=True,
    ordered=True,
)
MAP_SORT_BY = ("understat_id",)


def map_understat(pairs: pd.DataFrame, overrides: pd.DataFrame) -> pd.DataFrame:
    """understat_map from scored candidate pairs (`candidate_pairs` + `name_score`): greedy
    auto assignment, then overrides, which win (see `read_overrides`). An override that
    only names an FPL player removes his auto mapping and adds no row."""
    chosen = greedy_assign(pairs).assign(method="auto")
    if len(overrides):
        overridden_ids = set(overrides["understat_id"].dropna().astype(int))
        overridden_keys = set(overrides["player_key"].dropna().astype(int))
        chosen = chosen[
            ~chosen["understat_id"].isin(overridden_ids)
            & ~chosen["player_key"].isin(overridden_keys)
        ]
        forced = overrides.dropna(subset=["understat_id"]).astype({"understat_id": "int64"})
        scored = forced.merge(
            pairs.astype({"player_key": "Int64"}), on=["understat_id", "player_key"], how="left"
        )
        chosen = pd.concat([chosen, scored.assign(method="override")], ignore_index=True)
    out = pd.DataFrame(
        {
            "understat_id": chosen["understat_id"].astype("int64"),
            "player_key": chosen["player_key"].astype("Int64"),
            "method": chosen["method"].astype(str),
            "shared": chosen["shared"].astype("Int64"),
            "overlap": chosen["overlap"].astype("Float64"),
            "coverage": chosen["coverage"].astype("Float64"),
            "name_score": chosen["name_score"].astype("Float64"),
        }
    )
    out["event_time"] = pd.Series(EPOCH, index=out.index, dtype=UTC_US)
    out["available_at"] = out["event_time"]
    return out[MAP_COLUMNS].reset_index(drop=True)


# --- validation --------------------------------------------------------------------------


def id_dict_oracle(run: Path, player_season: pd.DataFrame) -> pd.DataFrame:
    """vaastav id_dict pairs: understat_id, player_key (FPL_ID via that season), season."""
    frames = []
    for path in sorted(run.glob("*/id_dict.csv.gz")):
        season = parse_season_label(path.parent.name)
        ids = read_raw_csv(path, usecols=["Understat_ID", "FPL_ID"])
        players = player_season.loc[player_season["season"] == season, ["element_id", "player_key"]]
        joined = ids.merge(players, left_on="FPL_ID", right_on="element_id", how="left")
        if joined["player_key"].isna().any():
            missing = joined.loc[joined["player_key"].isna(), "FPL_ID"].tolist()
            raise UnderstatMappingError(
                f"id_dict {season}: FPL id(s) {missing} not in player_season"
            )
        frames.append(
            pd.DataFrame(
                {
                    "understat_id": joined["Understat_ID"].astype("int64"),
                    "player_key": joined["player_key"].astype("int64"),
                    "season": season,
                }
            )
        )
    if not frames:
        return pd.DataFrame(
            {name: pd.Series(dtype="int64") for name in ("understat_id", "player_key", "season")}
        )
    return pd.concat(frames, ignore_index=True)


def mapping_problems(
    mapping: pd.DataFrame,
    upm: pd.DataFrame,
    fpl_apps: pd.DataFrame,
    oracle: pd.DataFrame,
    exempt: set[int] = frozenset(),
) -> list[str]:
    """Everything wrong with a mapping (see module docstring); empty = valid."""
    problems = []
    mapped = mapping.dropna(subset=["player_key"]).astype({"player_key": "int64"})
    for column in ("understat_id", "player_key"):
        if mapped[column].duplicated().any():
            dupes = mapped[mapped[column].duplicated(keep=False)]
            problems.append(f"{column} mapped more than once:\n{dupes}")
    by_us = dict(zip(mapped["understat_id"], mapped["player_key"], strict=True))
    by_fpl = dict(zip(mapped["player_key"], mapped["understat_id"], strict=True))
    decided = set(mapping["understat_id"])  # incl. overrides to "no mapping"

    # id_dict: no contradiction, and every pair whose players both appear is mapped.
    ours_for_us = oracle["understat_id"].map(by_us)
    ours_for_fpl = oracle["player_key"].map(by_fpl)
    wrong = oracle[
        (ours_for_us.notna() & ours_for_us.ne(oracle["player_key"]))
        | (ours_for_fpl.notna() & ours_for_fpl.ne(oracle["understat_id"]))
    ]
    if len(wrong):
        problems.append(f"{len(wrong)} id_dict pair(s) contradicted:\n{wrong.head(20)}")
    appear = oracle["understat_id"].isin(set(upm["understat_id"])) & oracle["player_key"].isin(
        set(fpl_apps["player_key"])
    )
    unmapped = oracle[appear & ~oracle["understat_id"].isin(decided)]
    if len(unmapped):
        problems.append(f"{len(unmapped)} id_dict pair(s) not mapped:\n{unmapped.head(20)}")

    # Seasons the mirror covers fully (Understat team files for every side, every FPL
    # player with minutes has a player file): every FPL player with minutes > 0 is mapped.
    apps = fpl_apps[fpl_apps["season"].isin(VALIDATED_SEASONS)]
    missing = apps[~apps["player_key"].isin(set(by_fpl) | set(exempt))]
    if len(missing):
        summary = missing.groupby(["season", "player_key"])["minutes"].agg(["size", "sum"])
        problems.append(
            f"{missing['player_key'].nunique()} FPL player(s) with minutes in "
            f"{VALIDATED_SEASONS.start}-{VALIDATED_SEASONS.stop - 1} have no Understat id "
            f"(add an override if that is right):\n{summary.head(20)}"
        )
    return problems


# --- the `understat` builder -------------------------------------------------------------


def attach_understat(
    player_match: pd.DataFrame, upm: pd.DataFrame, mapping: pd.DataFrame
) -> pd.DataFrame:
    """player_match with `us_*` filled by (player_key, fixture_key)."""
    mapped = mapping.dropna(subset=["player_key"]).astype({"player_key": "int64"})
    keyed = upm.merge(mapped[["understat_id", "player_key"]], on="understat_id")
    columns = list(UNDERSTAT_COLUMNS)
    out = player_match.drop(columns=columns).merge(
        keyed[["player_key", "fixture_key", *columns]],
        on=["player_key", "fixture_key"],
        how="left",
        validate="one_to_one",
    )
    unjoined = len(keyed) - int(out["us_minutes"].notna().sum())
    if unjoined:
        log.info("understat: %d mapped appearance(s) without a player_match row", unjoined)
    for name, dtype in UNDERSTAT_COLUMNS.items():
        out[name] = out[name].astype(dtype)
    return out[list(player_match.columns)]


def minutes_coverage(player_match: pd.DataFrame) -> dict[int, float]:
    """Per season: share of FPL minutes in rows that have Understat data."""
    minutes = player_match["minutes"]
    covered = minutes.where(player_match["us_minutes"].notna(), 0)
    by_season = (
        covered.groupby(player_match["season"]).sum()
        / minutes.groupby(player_match["season"]).sum()
    )
    return {int(s): round(float(v), 3) for s, v in by_season.items()}


def build_understat(ctx: BuildContext) -> pd.DataFrame:
    """Writes understat_player_match and understat_map, and rewrites player_dim
    (understat_id) and player_match (us_* columns). Returns understat_map."""
    run = vaastav_run(ctx)
    resolver = TeamResolver(ctx.table("team_dim"))
    fixture = ctx.table("fixture")
    player_season = ctx.table("player_season")
    player_match = ctx.table("player_match")

    upm = understat_player_match(read_player_files(run), read_player_names(run), fixture, resolver)
    write_table(upm, "understat_player_match", UPM_SCHEMA, ctx.data_dir, UPM_SORT_BY)

    fpl_apps = fpl_appearances(player_match)
    pairs = candidate_pairs(upm, fpl_apps)
    us_names = dict(zip(upm["understat_id"], upm["understat_name"], strict=True))
    pairs["name_score"] = name_scores(pairs, us_names, fpl_name_variants(player_season))
    overrides = read_overrides(ctx.config_dir)
    mapping = map_understat(pairs, overrides)
    no_understat = overrides.loc[overrides["understat_id"].isna(), "player_key"]
    problems = mapping_problems(
        mapping,
        upm,
        fpl_apps,
        id_dict_oracle(run, player_season),
        exempt={int(k) for k in no_understat},
    )
    if problems:
        raise UnderstatMappingError("Understat mapping failed: " + "\n".join(problems))
    write_table(mapping, "understat_map", MAP_SCHEMA, ctx.data_dir, MAP_SORT_BY)
    log.info(
        "understat_map: %d of %d Understat ids mapped (%d override row(s))",
        int(mapping["player_key"].notna().sum()),
        upm["understat_id"].nunique(),
        int(mapping["method"].eq("override").sum()),
    )

    ids = mapping.dropna(subset=["player_key"]).astype({"player_key": "int64"})
    player_dim = ctx.table("player_dim").drop(columns=["understat_id"])
    player_dim = player_dim.merge(
        ids[["player_key", "understat_id"]], on="player_key", how="left", validate="one_to_one"
    )
    player_dim["understat_id"] = player_dim["understat_id"].astype("Int64")
    write_table(
        player_dim[PLAYER_DIM_COLUMNS],
        "player_dim",
        PLAYER_DIM_SCHEMA,
        ctx.data_dir,
        PLAYER_DIM_SORT_BY,
    )
    player_match = attach_understat(player_match, upm, mapping)
    write_table(
        player_match, "player_match", PLAYER_MATCH_SCHEMA, ctx.data_dir, PLAYER_MATCH_SORT_BY
    )
    log.info(
        "understat: share of FPL minutes with Understat data by season: %s",
        minutes_coverage(player_match),
    )
    ctx.forget("player_dim")
    ctx.forget("player_match")
    return mapping


# --- team_match --------------------------------------------------------------------------

TEAM_MATCH_COLUMNS = [
    "fixture_key",
    "team_key",
    "season",
    "is_home",
    "goals_for",
    "goals_against",
    "us_xg",
    "us_xga",
    "us_npxg",
    "us_npxga",
    "fd_xg",
    "fd_xga",
    "fpl_xg",
    "fpl_xga",
    "event_time",
    "available_at",
]
XG_COLUMNS = ["us_xg", "us_xga", "us_npxg", "us_npxga", "fd_xg", "fd_xga", "fpl_xg", "fpl_xga"]

TEAM_MATCH_SCHEMA = pa.DataFrameSchema(
    {
        "fixture_key": pa.Column("int64"),
        "team_key": pa.Column("int64"),
        "season": pa.Column("int64", pa.Check.ge(FIRST_SEASON)),
        "is_home": pa.Column(bool),
        "goals_for": pa.Column("int64", pa.Check.ge(0)),
        "goals_against": pa.Column("int64", pa.Check.ge(0)),
        **{name: pa.Column("Float64", pa.Check.ge(0), nullable=True) for name in XG_COLUMNS},
        "event_time": pa.Column(UTC_US),
        "available_at": pa.Column(UTC_US),
    },
    checks=[
        pa.Check(
            lambda df: ~df.duplicated(["fixture_key", "team_key"]),
            error="(fixture_key, team_key) unique",
        ),
        pa.Check(
            lambda df: df.groupby("fixture_key")["is_home"].transform("sum") == 1,
            error="one home and one away row per fixture",
        ),
        pa.Check(
            lambda df: df.groupby("fixture_key")["is_home"].transform("size") == 2,
            error="two rows per fixture",
        ),
        pa.Check(lambda df: df["available_at"] > df["event_time"], error="available_at order"),
    ],
    strict=True,
    ordered=True,
)
TEAM_MATCH_SORT_BY = ("fixture_key", "team_key")


def team_sides(fixture: pd.DataFrame) -> pd.DataFrame:
    """Two rows per finished fixture: team, opponent, is_home, goals for/against, timing."""
    played = fixture[fixture["finished"]]
    common = played[["fixture_key", "season", "kickoff_time", "available_at"]]
    home = common.assign(
        team_key=played["home_team_key"],
        opponent_key=played["away_team_key"],
        is_home=True,
        goals_for=played["home_goals"],
        goals_against=played["away_goals"],
    )
    away = common.assign(
        team_key=played["away_team_key"],
        opponent_key=played["home_team_key"],
        is_home=False,
        goals_for=played["away_goals"],
        goals_against=played["home_goals"],
    )
    sides = pd.concat([home, away], ignore_index=True)
    return sides.astype({"goals_for": "int64", "goals_against": "int64"})


def understat_team_rows(
    raw: pd.DataFrame, sides: pd.DataFrame, resolver: TeamResolver
) -> pd.DataFrame:
    """Understat team-file rows joined to sides on (team, is_home, UK-local kickoff date),
    one per side (the fixture's season folder first, else the newest). Fails on rows that
    match no fixture or disagree with the result."""
    rows = raw.assign(
        team_key=raw["team_name"].map(resolver.understat),
        is_home=raw["h_a"].eq("h"),
        date=pd.to_datetime(raw["date"]).dt.date,
    )
    keys = sides.assign(date=sides["kickoff_time"].dt.tz_convert(UK).dt.date)[
        ["fixture_key", "season", "team_key", "is_home", "date", "goals_for", "goals_against"]
    ]
    joined = rows.merge(
        keys, on=["team_key", "is_home", "date"], how="left", validate="many_to_one"
    )
    unmatched = joined[joined["fixture_key"].isna()]
    if len(unmatched):
        raise UnderstatMappingError(
            f"{len(unmatched)} Understat team row(s) match no fixture:\n"
            f"{unmatched[['team_name', 'h_a', 'date', 'folder_season']].head(20)}"
        )
    differ = (joined["scored"] != joined["goals_for"]) | (
        joined["missed"] != joined["goals_against"]
    )
    if differ.any():
        shown = ["team_name", "date", "scored", "missed", "goals_for", "goals_against"]
        raise UnderstatMappingError(
            f"{int(differ.sum())} Understat team row(s) disagree with the result:\n"
            f"{joined.loc[differ, shown].head(20)}"
        )
    joined["own_folder"] = joined["folder_season"] == joined["season"]
    joined = joined.sort_values(["own_folder", "folder_season"], ascending=False, kind="mergesort")
    unique = joined.drop_duplicates(["fixture_key", "team_key"])
    return unique[["fixture_key", "team_key", "xG", "xGA", "npxG", "npxGA"]].astype(
        {"fixture_key": "int64"}
    )


def football_data_xg(
    ctx: BuildContext, fixture: pd.DataFrame, resolver: TeamResolver
) -> pd.DataFrame:
    """football-data `HxG`/`AxG` (seasons whose files carry them) by fixture_key."""
    base = ctx.store.root / "football-data" / "E0"
    frames = []
    for season_dir in sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []:
        season = 2000 + int(season_dir.name[:2])
        path = ctx.store.latest("football-data", f"E0/{season_dir.name}", suffix=".csv.gz")
        if season < FIRST_SEASON or path is None:
            continue
        df = read_raw_csv(path)
        if "HxG" not in df.columns:
            continue
        df = df.dropna(subset=["Date"])
        frames.append(
            pd.DataFrame(
                {
                    "season": season,
                    "home_team_key": [resolver.football_data(n) for n in df["HomeTeam"]],
                    "away_team_key": [resolver.football_data(n) for n in df["AwayTeam"]],
                    "fd_home_xg": pd.to_numeric(df["HxG"]).to_numpy(),
                    "fd_away_xg": pd.to_numeric(df["AxG"]).to_numpy(),
                }
            )
        )
    columns = ["fixture_key", "fd_home_xg", "fd_away_xg"]
    if not frames:
        return pd.DataFrame({name: pd.Series(dtype="float64") for name in columns}).astype(
            {"fixture_key": "int64"}
        )
    fd = pd.concat(frames, ignore_index=True)
    keys = fixture[["season", "home_team_key", "away_team_key", "fixture_key"]]
    joined = fd.merge(keys, on=["season", "home_team_key", "away_team_key"], how="left")
    if joined["fixture_key"].isna().any():
        raise ValueError("football-data xG rows without a fixture (see `fixture` cross-check)")
    return joined[columns].astype({"fixture_key": "int64"})


def side_fpl_xg(player_match: pd.DataFrame) -> pd.DataFrame:
    """Sum of the side's players' fpl_xg; null if any player row lacks it (before 2022-23
    GW16, see `players.LATE_COLUMNS_FIRST_GW`)."""
    grouped = player_match.groupby(["fixture_key", "team_key"])["fpl_xg"]
    total = grouped.sum(min_count=1)
    complete = grouped.count() == grouped.size()
    return total.where(complete).rename("fpl_xg").reset_index()


def build_team_match(ctx: BuildContext) -> pd.DataFrame:
    resolver = TeamResolver(ctx.table("team_dim"))
    fixture = ctx.table("fixture")
    sides = team_sides(fixture)
    us = understat_team_rows(read_team_files(vaastav_run(ctx)), sides, resolver)
    df = sides.merge(us, on=["fixture_key", "team_key"], how="left", validate="one_to_one")
    fd = football_data_xg(ctx, fixture, resolver)
    df = df.merge(fd, on="fixture_key", how="left", validate="many_to_one")
    home = df["is_home"]
    df["fd_xg"] = df["fd_home_xg"].where(home, df["fd_away_xg"])
    df["fd_xga"] = df["fd_away_xg"].where(home, df["fd_home_xg"])
    fpl = side_fpl_xg(ctx.table("player_match"))
    df = df.merge(fpl, on=["fixture_key", "team_key"], how="left", validate="one_to_one")
    df = df.merge(
        fpl.rename(columns={"team_key": "opponent_key", "fpl_xg": "fpl_xga"}),
        on=["fixture_key", "opponent_key"],
        how="left",
        validate="one_to_one",
    )
    df = df.rename(columns={"xG": "us_xg", "xGA": "us_xga", "npxG": "us_npxg", "npxGA": "us_npxga"})
    for name in XG_COLUMNS:
        df[name] = df[name].astype("Float64")
    df["event_time"] = df["kickoff_time"]
    return df[TEAM_MATCH_COLUMNS].reset_index(drop=True)
