"""ESPN summaries and scoreboards (`fplopt.ingest.espn`) -> `espn_match`, `espn_player_match`,
`espn_schedule_snapshot` and `espn_map` (spec §1:
docs/superpowers/specs/2026-10-27-espn-cup-data-design.md).

Tables:
- `espn_match` (event, key `espn_event_id`): the newest completed summary of every match with a
  side in team_dim (`TeamResolver.find("espn", team id)`; the other side's key is null for a
  club outside the PL); `fixture_key` for `eng.1`, joined on (season, home, away) — an `eng.1`
  match that does not join is a build error. `event_time` = kickoff, `available_at` = kickoff +
  `AVAILABLE_AFTER`.
- `espn_player_match` (event, key `espn_athlete_id`, `espn_event_id`): every squad player of a
  side in team_dim, with the match's times.
- `espn_schedule_snapshot` (snapshot): every scoreboard file is a snapshot at its timestamp of
  the scheduled (`state` 'pre') matches of PL clubs after it.
- `espn_map` (static): ESPN athlete -> player_key, the Understat matcher (`fplopt.build.understat`)
  on ESPN's `eng.1` appearances (minutes > 0) against FPL's; overrides in config/overrides.csv
  with source=espn. Athletes without PL minutes are not in it.

Minutes (`parse_summary`): a key-event clock of s seconds is minute min(s // 60 + 1, end), end =
120 after extra time else 90. A starter is on from 0, a substitute from his substitution
(participants [in, out]); off at his own substitution, his red card or the end. A red after a
yellow for the same player is a second yellow. Shoot-out events are ignored.

Checks (`espn_problems`, fail the build), for seasons with any `eng.1` match: every finished PL
fixture has one; every FPL player with minutes is mapped (or exempt by an override with a blank
source_id); every auto pair agrees on minutes (±5) in >= `MIN_MINUTES_AGREEMENT` of the
fixtures where both have a row and either played; ESPN and FPL red cards disagree in at most
`MAX_RED_DISAGREEMENT` of FPL reds (each disagreement logged).
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import pandera.pandas as pa

from fplopt.build import understat
from fplopt.build.common import UTC_US, BuildContext, write_table
from fplopt.build.teams import TeamResolver
from fplopt.build.understat import (
    MINUTES_TOLERANCE,
    candidate_pairs,
    fpl_appearances,
    fpl_name_variants,
    map_understat,
    name_scores,
    read_overrides,
)
from fplopt.ingest.raw_store import RawStore
from fplopt.seasons import season_start_year

log = logging.getLogger(__name__)

EXTRA_TIME_STATUSES = frozenset({"STATUS_FINAL_AET", "STATUS_FINAL_PEN"})


def _minute(seconds: float, end: int) -> int:
    """ESPN's display minute (3015 s -> 51'), capped at the match end (stoppage time)."""
    return min(int(seconds // 60) + 1, end)


def parse_summary(payload: dict[str, Any], league: str) -> tuple[dict, list[dict]]:
    """One summary -> its match row and one row per squad player (lineups, minutes, cards)."""
    competition = payload["header"]["competitions"][0]
    status = competition["status"]["type"]["name"]
    end = 120 if status in EXTRA_TIME_STATUSES else 90
    kickoff = pd.Timestamp(competition["date"]).tz_convert("UTC")
    sides = _sides(competition["competitors"])
    match = {
        "espn_event_id": int(payload["header"]["id"]),
        "league": league,
        "season": season_start_year(kickoff.to_pydatetime()),
        "kickoff_time": kickoff,
        "home_espn_id": sides["home"],
        "away_espn_id": sides["away"],
        "status": status,
        "extra_time": end == 120,
    }
    on: dict[int, int] = {}
    off: dict[int, int] = {}
    yellows: dict[int, int] = {}
    red_at: dict[int, int] = {}
    second_yellow: set[int] = set()
    events = sorted(payload.get("keyEvents", []), key=lambda e: e.get("clock", {}).get("value", 0))
    for ev in events:
        if ev.get("shootout"):
            continue
        text = ev["type"]["text"]
        ids = [int(p["athlete"]["id"]) for p in ev.get("participants", [])]
        minute = _minute(ev["clock"]["value"], end)
        if text == "Substitution" and len(ids) == 2:
            player_in, player_out = ids
            if player_out in off:  # already sent off or replaced: a data glitch, skip
                continue
            on.setdefault(player_in, minute)
            off[player_out] = minute
        elif text == "Yellow Card" and ids:
            yellows[ids[0]] = yellows.get(ids[0], 0) + 1
        elif text == "Red Card" and ids:
            if yellows.get(ids[0], 0) > 0:
                second_yellow.add(ids[0])
            red_at[ids[0]] = minute
            off.setdefault(ids[0], minute)
    players = []
    for roster in payload.get("rosters", []):
        team = str(roster["team"]["id"])
        for entry in roster.get("roster", []):
            athlete = int(entry["athlete"]["id"])
            start = 0 if entry.get("starter") else on.get(athlete)
            stop = off.get(athlete, end) if start is not None else None
            players.append(
                {
                    "espn_athlete_id": athlete,
                    "espn_event_id": match["espn_event_id"],
                    "espn_team_id": team,
                    "athlete_name": entry["athlete"].get("displayName", ""),
                    "starter": bool(entry.get("starter")),
                    "minute_on": start if start is not None else pd.NA,
                    "minute_off": stop if stop is not None else pd.NA,
                    "minutes": max(stop - start, 0) if start is not None else 0,
                    "yellows": yellows.get(athlete, 0),
                    "red": athlete in red_at,
                    "second_yellow": athlete in second_yellow,
                }
            )
    return match, players


SOURCE = "espn"
AVAILABLE_AFTER = pd.Timedelta(hours=3)  # VERIFY (spec §1): lineups/cards final by then


class EspnBuildError(ValueError):
    """ESPN data that does not fit: an unknown PL club, a missing PL fixture, a bad mapping."""


def _item_dirs(store: RawStore, kind: str) -> list[tuple[str, str]]:
    """(league, item) of every `espn/<league>/<kind>/<item>` directory, sorted."""
    root = store.root / SOURCE
    if not root.is_dir():
        return []
    return [(d.parent.parent.name, d.name) for d in sorted(root.glob(f"*/{kind}/*")) if d.is_dir()]


MATCH_SCHEMA = pa.DataFrameSchema(
    {
        "espn_event_id": pa.Column("int64", unique=True),
        "league": pa.Column(str),
        "season": pa.Column("int64"),
        "kickoff_time": pa.Column(UTC_US),
        "home_espn_id": pa.Column(str),
        "away_espn_id": pa.Column(str),
        "home_team_key": pa.Column("Int64", nullable=True),
        "away_team_key": pa.Column("Int64", nullable=True),
        "fixture_key": pa.Column("Int64", nullable=True),
        "status": pa.Column(str),
        "extra_time": pa.Column(bool),
        "event_time": pa.Column(UTC_US),
        "available_at": pa.Column(UTC_US),
    },
    strict=True,
    ordered=True,
)
MATCH_SORT_BY = ("kickoff_time", "espn_event_id")

PLAYER_SCHEMA = pa.DataFrameSchema(
    {
        "espn_athlete_id": pa.Column("int64"),
        "espn_event_id": pa.Column("int64"),
        "season": pa.Column("int64"),
        "league": pa.Column(str),
        "team_key": pa.Column("int64"),
        "athlete_name": pa.Column(str),
        "starter": pa.Column(bool),
        "minute_on": pa.Column("Int64", nullable=True),
        "minute_off": pa.Column("Int64", nullable=True),
        "minutes": pa.Column("int64", pa.Check.in_range(0, 120)),
        "yellows": pa.Column("int64", pa.Check.ge(0)),
        "red": pa.Column(bool),
        "second_yellow": pa.Column(bool),
        "event_time": pa.Column(UTC_US),
        "available_at": pa.Column(UTC_US),
    },
    checks=[
        pa.Check(
            lambda df: ~df.duplicated(["espn_athlete_id", "espn_event_id"]),
            error="(espn_athlete_id, espn_event_id) must be unique",
        )
    ],
    strict=True,
    ordered=True,
)
PLAYER_SORT_BY = ("espn_event_id", "espn_athlete_id")

SCHEDULE_SCHEMA = pa.DataFrameSchema(
    {
        "snapshot_at": pa.Column(UTC_US),
        "espn_event_id": pa.Column("int64"),
        "league": pa.Column(str),
        "season": pa.Column("int64"),
        "kickoff_time": pa.Column(UTC_US),
        "home_team_key": pa.Column("Int64", nullable=True),
        "away_team_key": pa.Column("Int64", nullable=True),
        "event_time": pa.Column(UTC_US),
        "available_at": pa.Column(UTC_US),
    },
    checks=[
        pa.Check(
            lambda df: ~df.duplicated(["snapshot_at", "espn_event_id"]),
            error="(snapshot_at, espn_event_id) must be unique",
        )
    ],
    strict=True,
    ordered=True,
)
SCHEDULE_SORT_BY = ("snapshot_at", "espn_event_id")

_INT64 = ("espn_event_id", "espn_athlete_id", "season", "team_key", "minutes", "yellows")
_NULLABLE_INT = ("home_team_key", "away_team_key", "fixture_key", "minute_on", "minute_off")
_TIMES = ("kickoff_time", "event_time", "available_at", "snapshot_at")
_BOOLS = ("extra_time", "starter", "red", "second_yellow")


def _frame(rows: list[dict], schema: pa.DataFrameSchema) -> pd.DataFrame:
    """`rows` as a frame with exactly the schema's columns and dtypes (also when empty)."""
    df = pd.DataFrame(rows, columns=list(schema.columns))
    for name in df.columns:
        if name in _INT64:
            df[name] = df[name].astype("int64")
        elif name in _NULLABLE_INT:
            df[name] = df[name].astype("Int64")
        elif name in _TIMES:
            df[name] = pd.to_datetime(df[name], utc=True).astype(UTC_US)
        elif name in _BOOLS:
            df[name] = df[name].astype(bool)
        else:
            df[name] = df[name].astype(str)
    return df


def _sides(competitors: list[dict]) -> dict[str, str]:
    return {c["homeAway"]: str(c["team"]["id"]) for c in competitors}


def _summaries(
    ctx: BuildContext, resolver: TeamResolver, problems: list[str]
) -> tuple[list[dict], list[dict]]:
    """Match and player rows from the newest completed summary of every event."""
    fixture = ctx.table("fixture")
    fixture_of = {
        (int(s), int(h), int(a)): int(k)
        for s, h, a, k in zip(
            fixture["season"],
            fixture["home_team_key"],
            fixture["away_team_key"],
            fixture["fixture_key"],
            strict=True,
        )
    }
    matches: list[dict] = []
    players: list[dict] = []
    for league, event_id in _item_dirs(ctx.store, "summary"):
        path = ctx.store.latest(SOURCE, f"{league}/summary/{event_id}")
        payload = ctx.store.read_json(path)
        if not payload["header"]["competitions"][0]["status"]["type"].get("completed"):
            continue
        match, rows = parse_summary(payload, league)
        home = resolver.find("espn", match["home_espn_id"])
        away = resolver.find("espn", match["away_espn_id"])
        fixture_key = None
        if league == "eng.1":
            fixture_key = fixture_of.get((match["season"], home or -1, away or -1))
            if fixture_key is None:
                problems.append(
                    f"eng.1 event {match['espn_event_id']} ({match['home_espn_id']} v "
                    f"{match['away_espn_id']}, {match['kickoff_time']:%Y-%m-%d}) has no fixture"
                )
                continue
        if home is None and away is None:
            continue
        times = {
            "event_time": match["kickoff_time"],
            "available_at": match["kickoff_time"] + AVAILABLE_AFTER,
        }
        matches.append(
            {**match, "home_team_key": home, "away_team_key": away, "fixture_key": fixture_key}
            | times
        )
        for row in rows:
            team_key = resolver.find("espn", row.pop("espn_team_id"))
            if team_key is not None:
                players.append(
                    {**row, "season": match["season"], "league": league, "team_key": team_key}
                    | times
                )
    return matches, players


def _schedule(ctx: BuildContext, resolver: TeamResolver) -> list[dict]:
    """Every scoreboard file is a snapshot (taken at its timestamp) of the scheduled matches
    of PL clubs it lists."""
    rows: list[dict] = []
    for league, period in _item_dirs(ctx.store, "scoreboard"):
        for ts, path in ctx.store.entries(SOURCE, f"{league}/scoreboard/{period}"):
            snapshot_at = pd.Timestamp(ts)
            for ev in ctx.store.read_json(path).get("events", []):
                kickoff = pd.Timestamp(ev["date"]).tz_convert("UTC")
                if ev["status"]["type"].get("state") != "pre" or kickoff <= snapshot_at:
                    continue
                sides = _sides(ev["competitions"][0]["competitors"])
                home = resolver.find("espn", sides["home"])
                away = resolver.find("espn", sides["away"])
                if home is None and away is None:
                    continue
                rows.append(
                    {
                        "snapshot_at": snapshot_at,
                        "espn_event_id": int(ev["id"]),
                        "league": league,
                        "season": season_start_year(kickoff.to_pydatetime()),
                        "kickoff_time": kickoff,
                        "home_team_key": home,
                        "away_team_key": away,
                        "event_time": snapshot_at,
                        "available_at": snapshot_at,
                    }
                )
    return rows


def build_espn_tables(ctx: BuildContext) -> dict[str, pd.DataFrame]:
    """espn_match, espn_player_match and espn_schedule_snapshot (the mapping: `map_espn`)."""
    resolver = TeamResolver(ctx.table("team_dim"))
    problems: list[str] = []
    matches, players = _summaries(ctx, resolver, problems)
    if problems:
        raise EspnBuildError(f"{len(problems)} ESPN problems: " + "; ".join(problems[:20]))
    schedule = _frame(_schedule(ctx, resolver), SCHEDULE_SCHEMA)
    return {
        "espn_match": _frame(matches, MATCH_SCHEMA),
        "espn_player_match": _frame(players, PLAYER_SCHEMA),
        "espn_schedule_snapshot": schedule.drop_duplicates(["snapshot_at", "espn_event_id"]),
    }


# --- espn_map and the build checks ---------------------------------------------------------

ID = "espn_athlete_id"
MIN_MINUTES_AGREEMENT = 0.9  # auto pairs: share of common fixtures with |Δ minutes| <= 5
MAX_RED_DISAGREEMENT = 0.02  # disagreeing rows, as a share of FPL red cards
MAX_LISTED = 20

MAP_SCHEMA = understat.MAP_SCHEMA.rename_columns({"understat_id": ID})
MAP_SORT_BY = (ID,)


def empty_map() -> pd.DataFrame:
    """espn_map with no rows (MAP_SCHEMA's columns and dtypes)."""
    return pd.DataFrame(
        {
            ID: pd.Series(dtype="int64"),
            "player_key": pd.Series(dtype="Int64"),
            "method": pd.Series(dtype=str),
            "shared": pd.Series(dtype="Int64"),
            "overlap": pd.Series(dtype="Float64"),
            "coverage": pd.Series(dtype="Float64"),
            "name_score": pd.Series(dtype="Float64"),
            "event_time": pd.Series(dtype=UTC_US),
            "available_at": pd.Series(dtype=UTC_US),
        }
    )


def pl_appearances(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """ESPN rows of matches with a PL fixture (eng.1): espn_athlete_id, fixture_key,
    minutes_src, red, athlete_name (minutes 0 included)."""
    match = tables["espn_match"].dropna(subset=["fixture_key"])
    rows = tables["espn_player_match"].merge(
        match[["espn_event_id", "fixture_key"]], on="espn_event_id"
    )
    rows = rows.assign(fixture_key=rows["fixture_key"].astype("int64"), minutes_src=rows["minutes"])
    return rows[[ID, "fixture_key", "minutes_src", "red", "athlete_name"]]


def map_espn(ctx: BuildContext, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """espn_map: the Understat matcher on ESPN's PL appearances (minutes > 0) against FPL's,
    names from espn_player_match vs player_season, overrides with source=espn."""
    espn = pl_appearances(tables)
    played = espn[espn["minutes_src"] > 0]
    if played.empty:
        return empty_map()
    pairs = candidate_pairs(
        played, fpl_appearances(ctx.table("player_match")), id_col=ID, minutes_col="minutes_src"
    )
    names = dict(zip(played[ID], played["athlete_name"], strict=True))
    pairs["name_score"] = name_scores(
        pairs, names, fpl_name_variants(ctx.table("player_season")), id_col=ID
    )
    overrides = read_overrides(ctx.config_dir, source=SOURCE, id_col=ID)
    return map_understat(pairs, overrides, id_col=ID)


def espn_problems(
    ctx: BuildContext, tables: dict[str, pd.DataFrame], mapping: pd.DataFrame
) -> list[str]:
    """The build checks (module docstring) over seasons with any eng.1 match; empty = valid."""
    match = tables["espn_match"]
    covered = set(match.loc[match["league"] == "eng.1", "season"])
    if not covered:
        return []
    problems: list[str] = []

    fixture = ctx.table("fixture")
    finished = fixture[fixture["finished"] & fixture["season"].isin(covered)]
    missing = finished[~finished["fixture_key"].isin(set(match["fixture_key"].dropna()))]
    if len(missing):
        problems.append(
            f"{len(missing)} finished PL fixtures have no ESPN match: "
            f"{missing['fixture_key'].head(MAX_LISTED).tolist()}"
        )

    pm = ctx.table("player_match")
    pm = pm[pm["season"].isin(covered)]
    overrides = read_overrides(ctx.config_dir, source=SOURCE, id_col=ID)
    exempt = set(overrides.loc[overrides[ID].isna(), "player_key"].astype(int))
    mapped = mapping.dropna(subset=["player_key"]).astype({"player_key": "int64"})
    unmapped = set(pm.loc[pm["minutes"] > 0, "player_key"]) - set(mapped["player_key"]) - exempt
    if unmapped:
        problems.append(
            f"{len(unmapped)} FPL players with minutes are not mapped: "
            f"{sorted(unmapped)[:MAX_LISTED]}"
        )

    both = (
        pl_appearances(tables)
        .merge(mapped[[ID, "player_key", "method"]], on=ID)
        .merge(
            pm[["player_key", "fixture_key", "minutes", "red_cards"]],
            on=["player_key", "fixture_key"],
        )
    )
    played = both[(both["method"] == "auto") & ((both["minutes"] > 0) | (both["minutes_src"] > 0))]
    close = (played["minutes"] - played["minutes_src"]).abs() <= MINUTES_TOLERANCE
    agreement = close.groupby(played[ID]).mean()
    low = agreement[agreement < MIN_MINUTES_AGREEMENT].sort_values()
    if len(low):
        problems.append(
            f"{len(low)} auto pairs agree on minutes in < {MIN_MINUTES_AGREEMENT:.0%} of "
            f"fixtures (add an override if the pair is right): "
            f"{low.head(MAX_LISTED).round(3).to_dict()}"
        )

    fpl_red = both["red_cards"] > 0
    wrong = both[fpl_red != both["red"]]
    if len(wrong):
        log.info(
            "espn: red cards disagree with FPL in %d row(s): %s",
            len(wrong),
            wrong[[ID, "player_key", "fixture_key", "red", "red_cards"]].to_dict("records"),
        )
    n_red = int(fpl_red.sum())
    if len(wrong) > MAX_RED_DISAGREEMENT * max(n_red, 1):
        problems.append(f"red cards disagree in {len(wrong)} of {n_red} FPL reds")
    return problems


def build_espn(ctx: BuildContext) -> pd.DataFrame:
    """Writes espn_match, espn_player_match, espn_schedule_snapshot and espn_map (nothing if a
    check fails). Returns espn_player_match (for the build log)."""
    tables = build_espn_tables(ctx)
    mapping = map_espn(ctx, tables)
    problems = espn_problems(ctx, tables, mapping)
    if problems:
        raise EspnBuildError(f"{len(problems)} ESPN problems: " + "; ".join(problems))
    specs = {
        "espn_match": (MATCH_SCHEMA, MATCH_SORT_BY),
        "espn_player_match": (PLAYER_SCHEMA, PLAYER_SORT_BY),
        "espn_schedule_snapshot": (SCHEDULE_SCHEMA, SCHEDULE_SORT_BY),
    }
    for name, (schema, sort_by) in specs.items():
        write_table(tables[name], name, schema, ctx.data_dir, sort_by)
        ctx.forget(name)
    write_table(mapping, "espn_map", MAP_SCHEMA, ctx.data_dir, MAP_SORT_BY)
    ctx.forget("espn_map")
    log.info(
        "espn_map: %d athletes mapped (%d override row(s)) of %d with PL minutes",
        int(mapping["player_key"].notna().sum()),
        int((mapping["method"] == "override").sum()),
        pl_appearances(tables).query("minutes_src > 0")[ID].nunique(),
    )
    return tables["espn_player_match"]
