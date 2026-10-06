"""`player_season`, `player_dim`, `player_match`, `player_gw` and `player_gw_ownership`.

Keys: `player_key` is the FPL player `code` (stable across seasons); `element_id` is the FPL
id, which resets every season and is never a key.

Registered players per season (`registered_players`): vaastav `players_raw` (2016-17 …
2025-26; managers, `element_type` 5, dropped) and, for seasons vaastav doesn't cover
(2026-27 →), the newest archived bootstrap of that season. player_match maps elements to
player_keys through them; player_season is built from them after player_gw.

player_season: one row per player and season, no club (`players_raw.team_code` is the
end-of-season club: the club per match is in player_match, the as-of club in
player_snapshot). Timing from the player's player_gw row with the earliest `available_at`
that season: `event_time` = its deadline, `available_at` = its `available_at` (see
player_gw below: a player added to FPL after a deadline is not visible at it, although
vaastav gives him a row for that GW). A player without rows: `event_time` = `available_at`
= the snapshot time of the bootstrap listing him (current seasons), else the season's last
lockdown (end-of-season players_raw). Measured on raw/ 2026-10-06: every one of the 8,005
player-seasons has rows.

player_dim: one row per `player_key`; names from the newest season; `opta_code` = "p" + code;
`understat_id` is filled by the `understat` builder (null here). Static reference data
(event_time = available_at = 1970-01-01), like team_dim, so it carries nothing that reveals
the future (the seasons a player plays are in player_season, with their own `available_at`).

player_match: one row per player per fixture his club played while he was registered (FPL
lists every registered player of a club with a fixture: 52–62% of rows have 0 minutes).
2016-17 … 2025-26 from vaastav merged_gw; each later season from the newest complete
element-summary run whose manifest `season` is that season (runs without a season are
skipped with a warning; team ids via the bootstrap archived with that run, else the newest
bootstrap of the season): only finished fixtures of GWs up to the run's `through_event`. A
season without such a run has no rows if none of its fixtures is finished, else fails.
- Dropped on purpose: `value`, `selected`, `transfers_*` (per GW, not per match: in
  `player_gw` / `player_gw_ownership`), `xP` (lookahead leak), ICT.
- Cleaning: exact duplicate rows dropped (2025-26: 10); managers dropped (2024-25: 20
  elements, position "AM"); for a duplicated (element, fixture) the row whose `round` equals
  the fixture's GW is kept (2019-20: 59 phantom GW29 copies of fixture 275); any other
  duplicate fails, and so does a pair none of whose rows is filed under the fixture's GW
  (dropping them all would lose the pair).
- Team: from the fixture (`home_team_key` if `was_home` else `away_team_key`), checked
  against `opponent_team` (season id -> code) and, where present (2020-21+), the `team` name.
  Kickoff, GW and timing come from `fixture`; `available_at` = the GW's lockdown.
- `starts` and `fpl_x*` (FPL/Opta expected goals / assists / goals conceded): from 2022-23
  GW16 (null before 2022-23: no columns; null for 2022-23 GW1–15, where vaastav has the
  columns but every value is a 0 placeholder). CBIT/recoveries/tackles where the source has
  them (2016-17 … 2018-19, 2025-26 →), `defensive_contribution` 2025-26 →; null elsewhere.
- Understat columns `us_*` are null here; the `understat` builder fills them and rewrites the
  table (`build()` reruns it whenever player_match or player_dim is rebuilt).

Checks: placeholder zeros — no (season, GW) with goals where every non-null value of an
optional stat column (`OPTIONAL_STATS`) is 0 (measured on raw/ 2026-10-06: the only such
blocks are 2022-23 GW1–15, `starts` and `fpl_x*`, nulled above); unique (player_key,
fixture_key); minutes 0–90 (measured on raw/ 2026-10-06: the maximum is 90 in every season,
FPL caps minutes); every row's fixture, GW and element are known; goal sums: home_goals =
Σ home goals_scored + Σ away own_goals (and vice versa) for every fixture with player rows —
measured 0 mismatches over all 3,850 played fixtures, so any mismatch fails the build
(`MAX_GOAL_SUM_MISMATCHES = 0`).

player_gw / player_gw_ownership: one row per (player_key, season, gw) with player_match rows
(the same source rows, cleaning and checks; all minutes incl. 0; a blank GW has no row).
Per-GW fields (verified against fplcache, issue #6): `value` is the price at deadline t, so
player_gw (with the club of that GW's fixture and the season's `element_type`) has
`event_time` = deadline t and `available_at` = when the player's registration with that club
was public (`registration_available_at`), at the earliest deadline t − 1h
(`KNOWN_BEFORE_DEADLINE`: state fixed before the deadline, PLAN §4). vaastav lists a player
added to FPL (or moved to a club) during GW t from GW t on, so the row can postdate the
deadline:
- once player_snapshot coverage starts (fplcache, 2021-04-18): deadline t − 1h if the
  newest snapshot by then lists him with that club, else the first later snapshot that does
  (else deadline t). Measured on raw/ 2026-10-06: 788 of 140,269 rows are later (760
  players new to FPL, listed a median 32 h after the deadline, at most 11.6 days; 28 club
  switches); `value` equals the last pre-deadline snapshot price in all 139,510 rows that
  have one;
- before it: deadline t − 1h only for players registered at game launch (first row of the
  season at his club's first fixture GW) with an unchanged club; a later signing's first row
  and a row after a club change get deadline t (visible from GW t+1): 778 of 107,372 rows.
`selected` / `transfers_in` / `transfers_out` include transfers up to deadline t, so
player_gw_ownership has `event_time` = deadline t and `available_at` = deadline t, or the
player_gw row's `available_at` if later (visible from GW t+1). In a double GW both fixture
rows must carry the same club and values (measured on raw/ 2026-10-06: they always do),
else the build fails.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from fplopt.build.common import (
    EPOCH,
    KNOWN_BEFORE_DEADLINE,
    MANIFEST,
    UTC_US,
    BuildContext,
    latest_complete_season_run,
    read_raw_csv,
)
from fplopt.build.fixtures import (
    FIRST_SEASON,
    bootstrap_team_codes,
    last_bootstrap_per_season,
    season_team_codes,
    vaastav_run,
    vaastav_seasons,
)
from fplopt.build.snapshots import report_missing_players
from fplopt.build.teams import TeamResolver
from fplopt.ingest.raw_store import RawStore, parse_ts
from fplopt.seasons import bootstrap_season

log = logging.getLogger(__name__)

MANAGER_TYPE = 5
MAX_MINUTES = 90
# First GW whose `starts` and FPL expected-stat columns hold data, for seasons where the
# columns exist but earlier GWs carry 0 placeholders (vaastav 2022-23: GW1–15).
LATE_COLUMNS_FIRST_GW = {2022: 16}
MAX_GOAL_SUM_MISMATCHES = 0
PLAYER_FIELDS = ["id", "code", "first_name", "second_name", "web_name"]


class PlayerMatchError(ValueError):
    """player_match source rows are inconsistent with fixtures or players."""


def current_seasons(ctx: BuildContext) -> list[int]:
    """Seasons in `fixture` that vaastav doesn't cover (read from our own archive)."""
    covered = set(vaastav_seasons(vaastav_run(ctx)))
    return sorted(int(s) for s in ctx.table("fixture")["season"].unique() if s not in covered)


# --- player_season -----------------------------------------------------------------------

PLAYER_SEASON_COLUMNS = [
    "player_key",
    "season",
    "element_id",
    "element_type",
    "first_name",
    "second_name",
    "web_name",
    "event_time",
    "available_at",
]

PLAYER_SEASON_SCHEMA = pa.DataFrameSchema(
    {
        "player_key": pa.Column("int64", pa.Check.gt(0)),
        "season": pa.Column("int64", pa.Check.ge(FIRST_SEASON)),
        "element_id": pa.Column("int64", pa.Check.gt(0)),
        "element_type": pa.Column("int64", pa.Check.in_range(1, 4)),
        "first_name": pa.Column(str),
        "second_name": pa.Column(str),
        "web_name": pa.Column(str),
        "event_time": pa.Column(UTC_US),
        "available_at": pa.Column(UTC_US),
    },
    checks=[
        pa.Check(lambda df: ~df.duplicated(["player_key", "season"]), error="player per season"),
        pa.Check(lambda df: ~df.duplicated(["season", "element_id"]), error="element per season"),
        pa.Check(
            lambda df: df["available_at"] >= df["event_time"] - KNOWN_BEFORE_DEADLINE,
            error="available_at not before deadline - 1h",
        ),
    ],
    strict=True,
    ordered=True,
)
PLAYER_SEASON_SORT_BY = ("season", "player_key")


def season_players(
    df: pd.DataFrame, season: int, listed_at: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Registered players (player_season columns without timing, plus `listed_at`: when the
    source listing them was taken, NaT if unknown) from players_raw / bootstrap `elements`."""
    df = df[df["element_type"] != MANAGER_TYPE]
    return pd.DataFrame(
        {
            "player_key": df["code"].astype("int64"),
            "season": season,
            "element_id": df["id"].astype("int64"),
            "element_type": df["element_type"].astype("int64"),
            "first_name": df["first_name"].astype(str),
            "second_name": df["second_name"].astype(str),
            "web_name": df["web_name"].astype(str),
            "listed_at": pd.Series(listed_at, index=df.index, dtype=UTC_US),
        }
    )


def players_raw(season_dir: Path) -> pd.DataFrame:
    return read_raw_csv(season_dir / "players_raw.csv.gz", usecols=[*PLAYER_FIELDS, "element_type"])


def bootstrap_elements(bootstrap: dict) -> pd.DataFrame:
    return pd.DataFrame(bootstrap["elements"])[[*PLAYER_FIELDS, "element_type"]]


def registered_players(ctx: BuildContext) -> pd.DataFrame:
    """`season_players` rows of every season: vaastav players_raw, and for current seasons the
    newest bootstrap of the season (`listed_at` = its snapshot time)."""
    frames = [
        season_players(players_raw(season_dir), season)
        for season, season_dir in vaastav_seasons(vaastav_run(ctx)).items()
    ]
    bootstraps = last_bootstrap_per_season(ctx.store)
    for season in current_seasons(ctx):
        if season not in bootstraps:
            raise LookupError(f"no archived bootstrap for season {season}")
        snapshot_at, path, _ = bootstraps[season]
        listed_at = pd.Timestamp(snapshot_at).tz_convert("UTC")
        elements = bootstrap_elements(RawStore.read_json(path))
        frames.append(season_players(elements, season, listed_at))
    return pd.concat(frames, ignore_index=True)


def first_visible_rows(player_gw: pd.DataFrame) -> pd.DataFrame:
    """season, player_key, first_event_time, first_available_at: per player and season the
    player_gw row with the earliest `available_at` (ties: the earliest deadline)."""
    rows = player_gw.sort_values(
        ["season", "player_key", "available_at", "event_time"], kind="mergesort"
    ).drop_duplicates(["season", "player_key"])
    return rows[["season", "player_key", "event_time", "available_at"]].rename(
        columns={"event_time": "first_event_time", "available_at": "first_available_at"}
    )


def player_season_from(
    players: pd.DataFrame, player_gw: pd.DataFrame, gameweek: pd.DataFrame
) -> pd.DataFrame:
    """player_season rows: `registered_players` + timing (see module docstring)."""
    missing = sorted(set(players["season"]) - set(gameweek["season"]))
    if missing:
        raise ValueError(f"no gameweek rows for season(s) {missing}")
    df = players.merge(
        first_visible_rows(player_gw),
        on=["season", "player_key"],
        how="left",
        validate="one_to_one",
    )
    no_rows = df["first_available_at"].isna()
    if no_rows.any():
        log.info(
            "player_season: %d player-season(s) without player_gw rows (per season: %s)",
            int(no_rows.sum()),
            df.loc[no_rows, "season"].value_counts().sort_index().to_dict(),
        )
    last_lockdown = gameweek.groupby("season")["lockdown_time"].max()
    fallback = df["listed_at"].fillna(df["season"].map(last_lockdown)).astype(UTC_US)
    df["event_time"] = df["first_event_time"].astype(UTC_US).fillna(fallback)
    df["available_at"] = df["first_available_at"].astype(UTC_US).fillna(fallback)
    return df[PLAYER_SEASON_COLUMNS]


def build_player_season(ctx: BuildContext) -> pd.DataFrame:
    return player_season_from(
        registered_players(ctx), ctx.table("player_gw"), ctx.table("gameweek")
    )


# --- player_dim --------------------------------------------------------------------------

PLAYER_DIM_COLUMNS = [
    "player_key",
    "first_name",
    "second_name",
    "web_name",
    "opta_code",
    "understat_id",
    "event_time",
    "available_at",
]

PLAYER_DIM_SCHEMA = pa.DataFrameSchema(
    {
        "player_key": pa.Column("int64", pa.Check.gt(0), unique=True),
        "first_name": pa.Column(str),
        "second_name": pa.Column(str),
        "web_name": pa.Column(str),
        "opta_code": pa.Column(str, pa.Check.str_matches(r"^p\d+$"), unique=True),
        "understat_id": pa.Column(
            "Int64", pa.Check(lambda s: ~s.duplicated() | s.isna()), nullable=True
        ),
        "event_time": pa.Column(UTC_US),
        "available_at": pa.Column(UTC_US),
    },
    checks=[
        pa.Check(
            lambda df: df["opta_code"] == "p" + df["player_key"].astype(str), error="opta_code"
        ),
    ],
    strict=True,
    ordered=True,
)
PLAYER_DIM_SORT_BY = ("player_key",)


def player_dim_from_seasons(player_season: pd.DataFrame) -> pd.DataFrame:
    ordered = player_season.sort_values(["player_key", "season"], kind="mergesort")
    newest = ordered.groupby("player_key").last()
    n = len(newest)
    epoch = pd.Series([EPOCH] * n, dtype=UTC_US)
    return pd.DataFrame(
        {
            "player_key": newest.index.astype("int64"),
            "first_name": newest["first_name"].to_numpy(),
            "second_name": newest["second_name"].to_numpy(),
            "web_name": newest["web_name"].to_numpy(),
            "opta_code": ("p" + newest.index.astype(str)).to_numpy(),
            "understat_id": pd.array([pd.NA] * n, dtype="Int64"),
            "event_time": epoch,
            "available_at": epoch,
        }
    )[PLAYER_DIM_COLUMNS]


def build_player_dim(ctx: BuildContext) -> pd.DataFrame:
    dim = player_dim_from_seasons(ctx.table("player_season"))
    path = ctx.table_path("player_snapshot")
    if path.exists():
        keys = pd.read_parquet(path, columns=["season", "player_key"]).drop_duplicates()
        report_missing_players(dim, keys)
    return dim


# --- player_match ------------------------------------------------------------------------

COUNT_STATS = [
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
SIGNED_STATS = ("bps", "total_points")  # can be negative
# Optional source columns -> table columns (null where the source lacks them).
OPTIONAL_INTS = {
    "starts": "starts",
    "clearances_blocks_interceptions": "clearances_blocks_interceptions",
    "recoveries": "recoveries",
    "tackles": "tackles",
    "defensive_contribution": "defensive_contribution",
}
OPTIONAL_FLOATS = {
    "expected_goals": "fpl_xg",
    "expected_assists": "fpl_xa",
    "expected_goals_conceded": "fpl_xgc",
}
# Per-GW state on every source row (player_gw / player_gw_ownership; not in player_match).
GW_STATE = ["value", "selected", "transfers_in", "transfers_out"]
# Filled by the `understat` builder.
UNDERSTAT_COLUMNS = {
    "us_minutes": "Int64",
    "us_goals": "Int64",
    "us_npg": "Int64",
    "us_xg": "Float64",
    "us_npxg": "Float64",
    "us_xa": "Float64",
    "us_shots": "Int64",
    "us_key_passes": "Int64",
}
DEFENSIVE = ["clearances_blocks_interceptions", "recoveries", "tackles", "defensive_contribution"]
# Columns nulled before LATE_COLUMNS_FIRST_GW.
LATE_COLUMNS = ["starts", *OPTIONAL_FLOATS.values()]
# Optional stats checked for placeholder-zero blocks (see `placeholder_zero_blocks`).
OPTIONAL_STATS = ["starts", *DEFENSIVE, *OPTIONAL_FLOATS.values()]

PLAYER_MATCH_COLUMNS = [
    "player_key",
    "season",
    "fixture_key",
    "gw",
    "element_id",
    "team_key",
    "opponent_team_key",
    "was_home",
    "kickoff_time",
    "minutes",
    "starts",
    *COUNT_STATS[1:],
    *DEFENSIVE,
    *OPTIONAL_FLOATS.values(),
    *UNDERSTAT_COLUMNS,
    "source",
    "event_time",
    "available_at",
]

PLAYER_MATCH_SCHEMA = pa.DataFrameSchema(
    {
        "player_key": pa.Column("int64", pa.Check.gt(0)),
        "season": pa.Column("int64", pa.Check.ge(FIRST_SEASON)),
        "fixture_key": pa.Column("int64"),
        "gw": pa.Column("int64", pa.Check.in_range(1, 47)),
        "element_id": pa.Column("int64", pa.Check.gt(0)),
        "team_key": pa.Column("int64"),
        "opponent_team_key": pa.Column("int64"),
        "was_home": pa.Column(bool),
        "kickoff_time": pa.Column(UTC_US),
        "minutes": pa.Column("int64", pa.Check.in_range(0, MAX_MINUTES)),
        "starts": pa.Column("Int64", pa.Check.isin([0, 1]), nullable=True),
        **{
            name: pa.Column("int64", None if name in SIGNED_STATS else pa.Check.ge(0))
            for name in COUNT_STATS[1:]
        },
        **{name: pa.Column("Int64", pa.Check.ge(0), nullable=True) for name in DEFENSIVE},
        **{
            name: pa.Column("Float64", pa.Check.ge(0), nullable=True)
            for name in OPTIONAL_FLOATS.values()
        },
        **{
            name: pa.Column(dtype, pa.Check.ge(0), nullable=True)
            for name, dtype in UNDERSTAT_COLUMNS.items()
        },
        "source": pa.Column(str, pa.Check.isin(["vaastav", "fpl"])),
        "event_time": pa.Column(UTC_US),
        "available_at": pa.Column(UTC_US),
    },
    checks=[
        pa.Check(
            lambda df: ~df.duplicated(["player_key", "fixture_key"]),
            error="(player_key, fixture_key) unique",
        ),
        pa.Check(lambda df: df["team_key"] != df["opponent_team_key"], error="team != opponent"),
        pa.Check(lambda df: df["event_time"] == df["kickoff_time"], error="event_time = kickoff"),
        pa.Check(lambda df: df["available_at"] > df["event_time"], error="available_at order"),
    ],
    strict=True,
    ordered=True,
)
PLAYER_MATCH_SORT_BY = ("fixture_key", "player_key")


def match_rows(df: pd.DataFrame, season: int, source: str) -> pd.DataFrame:
    """Source rows (merged_gw or element-summary history; any column order) -> the common raw
    shape: season, element_id, fpl_fixture_id, round, was_home, opponent_id, team_name (null
    when absent), stats, optional columns (null when absent), source."""
    out = pd.DataFrame(
        {
            "season": season,
            "element_id": df["element"].astype("int64"),
            "fpl_fixture_id": df["fixture"].astype("int64"),
            "round": df["round"].astype("int64"),
            "was_home": df["was_home"].astype(bool),
            "opponent_id": df["opponent_team"].astype("int64"),
            "team_name": df["team"].astype(str) if "team" in df.columns else None,
        },
        index=df.index,
    )
    for name in COUNT_STATS:
        out[name] = df[name].astype("int64")
    for column, name in OPTIONAL_INTS.items():
        values = df[column] if column in df.columns else pd.Series(pd.NA, index=df.index)
        out[name] = values.astype("Int64")
    for column, name in OPTIONAL_FLOATS.items():
        values = pd.to_numeric(df[column]) if column in df.columns else pd.NA
        out[name] = pd.Series(values, index=df.index).astype("Float64")
    for name in GW_STATE:
        values = df[name] if name in df.columns else pd.Series(pd.NA, index=df.index)
        out[name] = values.astype("Int64")
    out["source"] = source
    return out.reset_index(drop=True)


def drop_duplicates_and_managers(
    merged: pd.DataFrame, managers: set[int], season: int
) -> pd.DataFrame:
    exact = merged.duplicated()
    if exact.any():
        log.info("season %d: dropped %d exact duplicate row(s)", season, int(exact.sum()))
    merged = merged[~exact]
    is_manager = merged["element"].isin(managers)
    if is_manager.any():
        log.info("season %d: dropped %d manager row(s)", season, int(is_manager.sum()))
    return merged[~is_manager]


def _fail_if(mask: pd.Series, df: pd.DataFrame, what: str) -> None:
    if mask.any():
        columns = [c for c in ("season", "element_id", "fpl_fixture_id", "round") if c in df]
        raise PlayerMatchError(f"{int(mask.sum())} {what}:\n{df.loc[mask, columns].head(10)}")


def assemble_player_match(
    raw: pd.DataFrame,
    fixture: pd.DataFrame,
    gameweek: pd.DataFrame,
    players: pd.DataFrame,
    team_codes: dict[int, dict[int, int]],
    resolver: TeamResolver,
    keep: list[str] | tuple[str, ...] = (),
) -> pd.DataFrame:
    """player_match rows from raw source rows (`match_rows` shape); `players` maps (season,
    element_id) -> player_key. Fails (PlayerMatchError) on unknown fixtures, GWs or elements,
    leftover duplicates and team inconsistencies. `keep`: raw columns to carry along."""
    fx = fixture[
        ["season", "fpl_fixture_id", "fixture_key", "gw", "kickoff_time"]
        + ["home_team_key", "away_team_key"]
    ]
    df = raw.merge(fx, on=["season", "fpl_fixture_id"], how="left", validate="many_to_one")
    _fail_if(df["fixture_key"].isna() | df["gw"].isna(), df, "row(s) without a played fixture")
    df["gw"] = df["gw"].astype("int64")

    # A duplicated (element, fixture): keep the row filed under the fixture's GW.
    keys = ["season", "element_id", "fpl_fixture_id"]
    phantom = df.duplicated(keys, keep=False) & (df["round"] != df["gw"])
    # Every (element, fixture) must keep a row: one whose copies are all phantoms fails.
    lost = phantom.groupby([df[k] for k in keys]).transform("all")
    _fail_if(lost, df, "duplicate row(s) filed under another GW only (the pair would be lost)")
    if phantom.any():
        log.info("dropped %d duplicate row(s) filed under another GW", int(phantom.sum()))
    df = df[~phantom]
    _fail_if(df.duplicated(keys, keep=False), df, "duplicate (element, fixture) row(s)")

    player_keys = players[["season", "element_id", "player_key"]]
    df = df.merge(player_keys, on=["season", "element_id"], how="left", validate="many_to_one")
    _fail_if(df["player_key"].isna(), df, "row(s) whose element is not a registered player")

    home = df["was_home"]
    df["team_key"] = df["home_team_key"].where(home, df["away_team_key"]).astype("int64")
    df["opponent_team_key"] = df["away_team_key"].where(home, df["home_team_key"]).astype("int64")
    opponent_codes = pd.Series(
        [team_codes[s].get(i, -1) for s, i in zip(df["season"], df["opponent_id"], strict=True)],
        index=df.index,
    )
    _fail_if(opponent_codes != df["opponent_team_key"], df, "opponent_team mismatch(es)")
    named = df[df["team_name"].notna()]
    name_codes = named["team_name"].map(lambda name: resolver.find("fpl_name", name))
    _fail_if(name_codes != named["team_key"], named, "`team` name mismatch(es)")

    first_gw = df["season"].map(LATE_COLUMNS_FIRST_GW).fillna(0)
    for name in LATE_COLUMNS:
        df[name] = df[name].where(df["gw"] >= first_gw)

    lockdown = gameweek.set_index(["season", "gw"])["lockdown_time"]
    at = lockdown.reindex(pd.MultiIndex.from_frame(df[["season", "gw"]]))
    df["available_at"] = at.to_numpy()
    _fail_if(df["available_at"].isna(), df, "row(s) without a gameweek")
    df["available_at"] = df["available_at"].astype(UTC_US)
    df["event_time"] = df["kickoff_time"]
    for name, dtype in UNDERSTAT_COLUMNS.items():
        df[name] = pd.Series(pd.NA, index=df.index, dtype=dtype)
    df["player_key"] = df["player_key"].astype("int64")
    df["fixture_key"] = df["fixture_key"].astype("int64")
    return df[[*PLAYER_MATCH_COLUMNS, *keep]].reset_index(drop=True)


def goal_sum_mismatches(player_match: pd.DataFrame, fixture: pd.DataFrame) -> pd.DataFrame:
    """Fixtures with player rows whose result differs from the players' goals plus the
    opponents' own goals: home_goals = Σ home goals_scored + Σ away own_goals (and vice
    versa). Returns fixture_key, the sums and the fixture's goals."""
    sides = player_match.groupby(["fixture_key", "was_home"])[["goals_scored", "own_goals"]].sum()
    sides = sides.unstack("was_home", fill_value=0)
    sums = pd.DataFrame(
        {
            "home_sum": sides[("goals_scored", True)] + sides[("own_goals", False)],
            "away_sum": sides[("goals_scored", False)] + sides[("own_goals", True)],
        }
    )
    goals = fixture.set_index("fixture_key")[["home_goals", "away_goals"]]
    joined = sums.join(goals, how="left")
    differ = (joined["home_sum"] != joined["home_goals"]) | (
        joined["away_sum"] != joined["away_goals"]
    )
    return joined[differ.fillna(True)].reset_index()


def placeholder_zero_blocks(player_match: pd.DataFrame) -> pd.DataFrame:
    """(season, gw, column) where goals were scored but every non-null value of an optional
    stat column is 0: a source placeholder (like vaastav 2022-23 GW1–15), not data."""
    by_gw = player_match.groupby(["season", "gw"])
    scored = by_gw["goals_scored"].sum() > 0
    blocks = []
    for name in OPTIONAL_STATS:
        values = player_match[name]
        present = by_gw[name].count() > 0
        nonzero = values.fillna(0).ne(0).groupby([player_match["season"], player_match["gw"]])
        bad = scored & present & ~nonzero.any()
        blocks += [(season, gw, name) for season, gw in bad[bad].index]
    return pd.DataFrame(blocks, columns=["season", "gw", "column"])


def check_placeholder_zeros(player_match: pd.DataFrame) -> None:
    blocks = placeholder_zero_blocks(player_match)
    if len(blocks):
        raise PlayerMatchError(
            f"{len(blocks)} placeholder-zero block(s): goals scored but the column is 0 for "
            f"every player of the GW (null it, see LATE_COLUMNS_FIRST_GW):\n"
            f"{blocks.head(20).to_string(index=False)}"
        )


def check_goal_sums(player_match: pd.DataFrame, fixture: pd.DataFrame) -> None:
    mismatches = goal_sum_mismatches(player_match, fixture)
    if len(mismatches):
        log.warning("%d fixture(s) with goal-sum mismatches:\n%s", len(mismatches), mismatches)
    if len(mismatches) > MAX_GOAL_SUM_MISMATCHES:
        raise PlayerMatchError(
            f"{len(mismatches)} fixture(s) whose result != players' goals + opponents' own "
            f"goals (allowed {MAX_GOAL_SUM_MISMATCHES}):\n{mismatches.head(20)}"
        )


# --- sources -----------------------------------------------------------------------------


def vaastav_match_rows(season: int, season_dir: Path) -> pd.DataFrame:
    merged = read_raw_csv(season_dir / "gws" / "merged_gw.csv.gz")
    players = players_raw(season_dir)
    managers = set(players.loc[players["element_type"] == MANAGER_TYPE, "id"].astype(int))
    merged = drop_duplicates_and_managers(merged, managers, season)
    return match_rows(merged, season, "vaastav")


def element_summary_bootstrap(ctx: BuildContext, run: Path, season: int) -> dict:
    """The bootstrap archived with an element-summary run, else the season's newest."""
    path = ctx.store.path_for("fpl", "bootstrap-static", parse_ts(run.name))
    if path.exists():
        bootstrap = RawStore.read_json(path)
        if bootstrap_season(bootstrap) == season:
            return bootstrap
    bootstraps = last_bootstrap_per_season(ctx.store)
    if season not in bootstraps:
        raise LookupError(f"no archived bootstrap for season {season}")
    return RawStore.read_json(bootstraps[season][1])


# Source columns `match_rows` needs (an empty run's history frame has none).
SOURCE_COLUMNS = ["element", "fixture", "round", "was_home", "opponent_team", *COUNT_STATS]


def element_summary_match_rows(
    ctx: BuildContext, season: int, fixture: pd.DataFrame
) -> tuple[pd.DataFrame, dict[int, int]]:
    """A current season's rows from its newest complete element-summary run (finished
    fixtures of GWs up to the run's `through_event`) and the team id -> code map for them."""
    finished = fixture.loc[(fixture["season"] == season) & fixture["finished"], "fpl_fixture_id"]
    run = latest_complete_season_run(ctx.store, "fpl", "element-summary", season)
    if run is None:
        if len(finished):
            raise LookupError(
                f"no complete element-summary run for season {season}, which has "
                f"{len(finished)} finished fixture(s)"
            )
        log.info("season %d: no element-summary run and no finished fixture yet", season)
        return match_rows(pd.DataFrame(columns=SOURCE_COLUMNS), season, "fpl"), {}
    manifest = RawStore.read_json(run / MANIFEST)
    histories = [
        pd.DataFrame(RawStore.read_json(path)["history"])
        for path in sorted(run.glob("*.json.gz"))
        if path.name != MANIFEST
    ]
    histories = [h for h in histories if len(h)]
    history = (
        pd.concat(histories, ignore_index=True)
        if histories
        else pd.DataFrame(columns=SOURCE_COLUMNS)
    )
    rows = match_rows(history, season, "fpl")
    keep = rows["fpl_fixture_id"].isin(finished)
    through = manifest.get("through_event")
    if through is not None:
        keep &= rows["round"] <= int(through)
    if (~keep).any():
        log.info("element-summary %s: skipped %d unfinished row(s)", run.name, int((~keep).sum()))
    return rows[keep], bootstrap_team_codes(element_summary_bootstrap(ctx, run, season))


def assembled_match_rows(
    ctx: BuildContext, keep: list[str] | tuple[str, ...] = ()
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(`assemble_player_match` over every source season, `registered_players`)."""
    fixture = ctx.table("fixture")
    players = registered_players(ctx)
    resolver = TeamResolver(ctx.table("team_dim"))
    frames, team_codes = [], {}
    for season, season_dir in vaastav_seasons(vaastav_run(ctx)).items():
        frames.append(vaastav_match_rows(season, season_dir))
        team_codes[season] = season_team_codes(season_dir)
    for season in current_seasons(ctx):
        rows, team_codes[season] = element_summary_match_rows(ctx, season, fixture)
        frames.append(rows)
    raw = pd.concat([f for f in frames if len(f)] or frames[:1], ignore_index=True)
    gameweek = ctx.table("gameweek")
    df = assemble_player_match(raw, fixture, gameweek, players, team_codes, resolver, keep)
    return df, players


def build_player_match(ctx: BuildContext) -> pd.DataFrame:
    df, _ = assembled_match_rows(ctx)
    fixture = ctx.table("fixture")
    check_goal_sums(df, fixture)
    check_placeholder_zeros(df)
    return df


# --- player_gw / player_gw_ownership -----------------------------------------------------

GW_KEY = ["player_key", "season", "gw"]
PLAYER_GW_COLUMNS = [*GW_KEY, "team_key", "element_type", "value", "event_time", "available_at"]
PLAYER_GW_OWNERSHIP_COLUMNS = [
    *GW_KEY,
    "selected",
    "transfers_in",
    "transfers_out",
    "event_time",
    "available_at",
]


def _gw_key_columns() -> dict[str, pa.Column]:
    return {
        "player_key": pa.Column("int64", pa.Check.gt(0)),
        "season": pa.Column("int64", pa.Check.ge(FIRST_SEASON)),
        "gw": pa.Column("int64", pa.Check.in_range(1, 47)),
    }


def _gw_key_unique(df: pd.DataFrame) -> pd.Series:
    return ~df.duplicated(GW_KEY)


PLAYER_GW_SCHEMA = pa.DataFrameSchema(
    {
        **_gw_key_columns(),
        "team_key": pa.Column("int64"),
        "element_type": pa.Column("int64", pa.Check.in_range(1, 4)),
        "value": pa.Column("int64", pa.Check.gt(0)),
        "event_time": pa.Column(UTC_US),
        "available_at": pa.Column(UTC_US),
    },
    checks=[
        pa.Check(_gw_key_unique, error="(player_key, season, gw) unique"),
        pa.Check(
            lambda df: df["available_at"] >= df["event_time"] - KNOWN_BEFORE_DEADLINE,
            error="available_at not before deadline - 1h",
        ),
    ],
    strict=True,
    ordered=True,
)
PLAYER_GW_OWNERSHIP_SCHEMA = pa.DataFrameSchema(
    {
        **_gw_key_columns(),
        "selected": pa.Column("int64", pa.Check.ge(0)),
        "transfers_in": pa.Column("int64", pa.Check.ge(0)),
        "transfers_out": pa.Column("int64", pa.Check.ge(0)),
        "event_time": pa.Column(UTC_US),
        "available_at": pa.Column(UTC_US),
    },
    checks=[
        pa.Check(_gw_key_unique, error="(player_key, season, gw) unique"),
        pa.Check(lambda df: df["available_at"] >= df["event_time"], error="available_at"),
    ],
    strict=True,
    ordered=True,
)
PLAYER_GW_SORT_BY = ("season", "gw", "player_key")


def player_gw_from(
    rows: pd.DataFrame, players: pd.DataFrame, gameweek: pd.DataFrame
) -> pd.DataFrame:
    """One row per (player_key, season, gw) of assembled match rows carrying `GW_STATE`:
    club, `element_type`, the GW state and the GW's `deadline_time`. Fails (PlayerMatchError)
    if a GW's rows (a double) disagree, a value is missing or a GW is not in `gameweek`."""
    columns = ["team_key", *GW_STATE]
    groups = rows.groupby(GW_KEY, sort=True)
    distinct = groups[columns].nunique(dropna=False)
    differ = distinct[(distinct > 1).any(axis=1)]
    if len(differ):
        which = ", ".join(c for c in columns if (differ[c] > 1).any())
        raise PlayerMatchError(
            f"{len(differ)} (player, season, GW)(s) whose rows differ within a GW ({which}):\n"
            f"{differ.head(10)}"
        )
    df = groups[columns].first().reset_index()
    missing = [c for c in GW_STATE if df[c].isna().any()]
    if missing:
        raise PlayerMatchError(f"player GW row(s) without {', '.join(missing)}")
    df = df.astype(dict.fromkeys(GW_STATE, "int64"))
    df = df.merge(
        players[["season", "player_key", "element_type"]],
        on=["season", "player_key"],
        how="left",
        validate="many_to_one",
    )
    df = df.merge(
        gameweek[["season", "gw", "gw_index", "deadline_time"]],
        on=["season", "gw"],
        how="left",
        validate="many_to_one",
    )
    _fail_if(df["deadline_time"].isna(), df, "player GW row(s) whose GW is not in gameweek")
    df["element_type"] = df["element_type"].astype("int64")
    df["gw_index"] = df["gw_index"].astype("int64")
    df["deadline_time"] = df["deadline_time"].astype(UTC_US)
    return df


def player_gw_table(gw: pd.DataFrame, available_at: pd.Series) -> pd.DataFrame:
    """player_gw: price and club at deadline t; `available_at` from
    `registration_available_at` (deadline t - 1h unless the player or his club was not
    public by then)."""
    df = gw.assign(event_time=gw["deadline_time"], available_at=available_at.astype(UTC_US))
    return df[PLAYER_GW_COLUMNS]


def player_gw_ownership_table(gw: pd.DataFrame, player_gw: pd.DataFrame) -> pd.DataFrame:
    """player_gw_ownership: final only at deadline t, and never before the player's
    player_gw row of that GW (whose existence it would reveal)."""
    registered = player_gw[[*GW_KEY, "available_at"]].rename(columns={"available_at": "known"})
    df = gw.merge(registered, on=GW_KEY, how="left", validate="one_to_one")
    _fail_if(df["known"].isna(), df, "ownership row(s) without a player_gw row")
    later = df["known"] > df["deadline_time"]
    df["event_time"] = df["deadline_time"]
    df["available_at"] = df["known"].where(later, df["deadline_time"]).astype(UTC_US)
    return df[PLAYER_GW_OWNERSHIP_COLUMNS]


def player_gw_rows(ctx: BuildContext) -> pd.DataFrame:
    rows, players = assembled_match_rows(ctx, keep=GW_STATE)
    return player_gw_from(rows, players, ctx.table("gameweek"))


def build_player_gw(ctx: BuildContext) -> pd.DataFrame:
    gw = player_gw_rows(ctx)
    stints, coverage_start = read_registration_stints(ctx)
    club_first = club_first_gameweeks(ctx.table("fixture"))
    available_at = registration_available_at(gw, stints, coverage_start, club_first)
    late = available_at > gw["deadline_time"] - KNOWN_BEFORE_DEADLINE
    if late.any():
        log.info(
            "player_gw: %d row(s) not known an hour before their deadline (per season: %s)",
            int(late.sum()),
            gw.loc[late, "season"].value_counts().sort_index().to_dict(),
        )
    return player_gw_table(gw, available_at)


def build_player_gw_ownership(ctx: BuildContext) -> pd.DataFrame:
    return player_gw_ownership_table(player_gw_rows(ctx), ctx.table("player_gw"))


# --- registration timing -----------------------------------------------------------------

REGISTRATION_COLUMNS = ["snapshot_at", "season", "player_key", "team_key"]


def registration_stints(snapshots: pd.DataFrame) -> pd.DataFrame:
    """From player_snapshot rows: per player and season, the first snapshot of each run of
    consecutive snapshots listing him with the same club (snapshot_at, season, player_key,
    team_key), sorted by player, season and time."""
    df = snapshots[REGISTRATION_COLUMNS].sort_values(
        ["player_key", "season", "snapshot_at"], kind="mergesort"
    )
    key = df[["player_key", "season", "team_key"]]
    starts = (key != key.shift()).any(axis=1)
    return df[starts].reset_index(drop=True)


def read_registration_stints(ctx: BuildContext) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """(`registration_stints` of the built player_snapshot table, its first snapshot time or
    None if it is empty). Reads only the four columns it needs."""
    path = ctx.table_path("player_snapshot")
    if not path.exists():
        raise FileNotFoundError(
            f"table 'player_snapshot' has not been built ({path}); player_gw needs it: run "
            "`fplopt build player_snapshot` first"
        )
    snapshots = pd.read_parquet(path, columns=REGISTRATION_COLUMNS)
    start = pd.Timestamp(snapshots["snapshot_at"].min()) if len(snapshots) else None
    return registration_stints(snapshots), start


def club_first_gameweeks(fixture: pd.DataFrame) -> pd.DataFrame:
    """season, team_key, first_gw_index: the GW of each club's first fixture of a season
    (final fixture list)."""
    scheduled = fixture[fixture["gw_index"].notna()]
    columns = ["season", "team_key", "gw_index"]
    home = scheduled.rename(columns={"home_team_key": "team_key"})[columns]
    away = scheduled.rename(columns={"away_team_key": "team_key"})[columns]
    sides = pd.concat([home, away], ignore_index=True)
    first = sides.groupby(["season", "team_key"], as_index=False)["gw_index"].min()
    return first.rename(columns={"gw_index": "first_gw_index"}).astype("int64")


def registration_available_at(
    gw: pd.DataFrame,
    stints: pd.DataFrame,
    coverage_start: pd.Timestamp | None,
    club_first: pd.DataFrame,
) -> pd.Series:
    """When each player GW row (player_key, season, gw_index, team_key, deadline_time) was
    public, i.e. the player registered with that club; `known` = deadline - 1h.
    - Rows with snapshots (`known` >= `coverage_start`, the first player_snapshot): `known`
      if the newest snapshot by then lists the player with that club, else the first later
      snapshot that does (it can be after the deadline), else the deadline.
    - Earlier rows: `known`, except the deadline (visible from GW t+1) for the player's first
      row of the season when it is not at his club's first fixture GW (registered after game
      launch) and for a row whose club differs from his previous row's that season (moved).
    Returns datetime64[us, UTC] aligned with `gw`."""
    deadline = gw["deadline_time"].astype(UTC_US)
    known = deadline - KNOWN_BEFORE_DEADLINE
    out = known.copy()
    if coverage_start is None:
        with_snapshots = pd.Series(False, index=gw.index)
    else:
        with_snapshots = known >= coverage_start

    early = gw[~with_snapshots].sort_values(["player_key", "season", "gw_index"], kind="mergesort")
    previous_team = early.groupby(["player_key", "season"])["team_key"].shift()
    first_gw = early.merge(club_first, on=["season", "team_key"], how="left")["first_gw_index"]
    first_gw.index = early.index
    joined_late = previous_team.isna() & (early["gw_index"] != first_gw)
    moved = previous_team.notna() & (early["team_key"] != previous_team)
    late = early.index[joined_late | moved]
    out[late] = deadline[late]

    rows = gw[with_snapshots]
    if len(rows):
        left = rows[["player_key", "season", "team_key"]].assign(known=known[rows.index])
        left = left.reset_index(names="row").sort_values("known", kind="mergesort")
        listed = stints.rename(columns={"snapshot_at": "listed_at", "team_key": "listed_team"})
        newest = pd.merge_asof(
            left,
            listed.sort_values("listed_at", kind="mergesort"),
            left_on="known",
            right_on="listed_at",
            by=["player_key", "season"],
            direction="backward",
        )
        joined = pd.merge_asof(
            left,
            stints.rename(columns={"snapshot_at": "joined_at"}).sort_values(
                "joined_at", kind="mergesort"
            ),
            left_on="known",
            right_on="joined_at",
            by=["player_key", "season", "team_key"],
            direction="forward",
        )
        same_club = newest["listed_team"] == newest["team_key"]
        at = newest["known"].where(same_club, joined["joined_at"])
        at = at.fillna(deadline.loc[newest["row"]].reset_index(drop=True))
        out.loc[newest["row"].to_numpy()] = at.astype(UTC_US).to_numpy()
    return out.astype(UTC_US)
