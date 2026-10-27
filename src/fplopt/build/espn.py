"""ESPN summaries and scoreboards (`fplopt.ingest.espn`) -> `espn_match`, `espn_player_match`,
`espn_schedule_snapshot` and `espn_map` (spec §1:
docs/superpowers/specs/2026-10-27-espn-cup-data-design.md).

Tables:
- `espn_match` (event, key `espn_event_id`): the newest summary of every match with a side in
  team_dim (`TeamResolver.find("espn", team id)`; the other side's key is null for a club
  outside the PL), if finished (`FINISHED_STATUSES`: full time, after extra time, on
  penalties; abandoned or postponed matches are dropped); `fixture_key` for `eng.1`, joined on
  (season, home, away) — an `eng.1` match that does not join, or a fixture with two `eng.1`
  events, is a build error. `season` is ESPN's (`season.year` of the summary header or
  scoreboard event, the start year: 2019/20's matches of July 2020 are 2019), else the season
  in progress at kickoff. `event_time` = kickoff, `available_at` = kickoff + `AVAILABLE_AFTER`.
- `espn_player_match` (event, key `espn_athlete_id`, `espn_event_id`): every squad player of a
  side in team_dim, with the match's times.
- `espn_schedule_snapshot` (snapshot): every scoreboard file is a snapshot at its timestamp of
  the scheduled (`state` 'pre') matches of PL clubs after it.
- `espn_map` (static): ESPN athlete -> player_key, the Understat matcher (`fplopt.build.understat`)
  on ESPN's `eng.1` appearances (minutes > 0) against FPL's; overrides in config/overrides.csv
  with source=espn. Athletes without PL minutes are not in it.

Minutes (`parse_summary`): a key-event clock of s seconds is minute min(s // 60 + 1, end), end =
120 after extra time else 90: STATUS_FINAL_AET, or STATUS_FINAL_PEN with a non-shoot-out key
event in period 3+ or a "Start Extra Time" event (an EFL Cup shoot-out straight after 90' has
neither). A starter is on from 0, a substitute from his substitution (participants [in, out]);
off at his own substitution, his red card or the end; a player who came on gets at least 1
minute (FPL's convention for a stoppage-time substitute). A red after a yellow for the same player
is a second yellow. Key events are read in (period, clock) order (stable); shoot-out events are
ignored, as is (logged) a substitution or card without a clock. A roster `subbedIn` without a
Substitution event bringing him on, or `subbedOut` without one taking him off (nor a red card),
leaves his `minutes` (and the unknown minute_on/off) null; the count per league is logged. The
matcher and the minutes check skip null rows.

- `club_match` (event, key `espn_event_id`): every non-league match of a PL club, played
  (`espn_match`) or scheduled (an `espn_schedule_snapshot` event not played yet; kickoff and
  sides from its newest snapshot, a small accepted look-ahead on the date of a rescheduled
  match). `available_at` = when it was known: min(first snapshot listing it, rule time); the
  rule time per PL side = the kickoff of the club's previous match in the same league and
  season (the draw follows the previous round), or `kickoff - FIRST_ROUND_NOTICE` without one;
  the row's rule time is the latest over its PL sides; never after `kickoff - 1 us`.
  `event_time` = kickoff.

Checks (`espn_problems`, fail the build), for seasons with any `eng.1` match: every finished PL
fixture that kicked off over `GRACE` (2 days) before the newest ESPN `eng.1` match has one
(matches are archived daily, so the last days may be missing); for each season before the
newest of them: every FPL player with minutes is mapped (or exempt by an override with a blank
source_id); every auto pair agrees on minutes (±5) in >= `MIN_MINUTES_AGREEMENT` of the
fixtures where both have a row (ESPN minutes known) and either played; ESPN and FPL red cards
disagree in at most `MAX_RED_DISAGREEMENT` of FPL reds (each disagreement logged). The newest
season's player checks (in progress, its data still settling) are logged as warnings, so they
never fail the build. `espn` is the last table of `fplopt.build.ORDER`.

Results on raw/ 2026-10-27 (backfill 2016/17 – 2026/27 GW8, 6,002 summaries): matches eng.1
3,880 (every PL fixture), eng.league_cup 651, eng.fa 644, uefa.champions 442, uefa.europa
297, uefa.europa.conf 69, eng.charity 11, uefa.super_cup 6; 201,320 player rows of PL clubs.
All 2,013 athletes with PL minutes mapped, 1 by override (Roman Dixon, listed as Roman
Quintyne); 0 red-card disagreements with FPL (461 PL reds, 45% second yellows); 4 rows with
unknown minutes (1 eng.1, 1 eng.fa, 2 eng.league_cup); in the newest season one pair agrees on
minutes in 6 of 7 fixtures (a warning). Found on the way: stoppage-time substitutes need the
1-minute floor (35 of the first 36 unmapped players), and ESPN lists a goalkeeper twice in 25
matches (one row kept).
"""

from __future__ import annotations

import logging
from collections import Counter
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

AFTER_EXTRA_TIME = "STATUS_FINAL_AET"
ON_PENALTIES = "STATUS_FINAL_PEN"  # straight after 90' (EFL Cup) or after extra time
# Played to a finish; abandoned, postponed and cancelled matches are dropped.
FINISHED_STATUSES = frozenset({"STATUS_FULL_TIME", AFTER_EXTRA_TIME, ON_PENALTIES})


def _minute(seconds: float, end: int) -> int:
    """ESPN's display minute (3015 s -> 51'), capped at the match end (stoppage time)."""
    return min(int(seconds // 60) + 1, end)


MINUTE_EVENTS = frozenset({"Substitution", "Yellow Card", "Red Card"})


def _event_order(ev: dict) -> tuple[int, float]:
    """(period, clock): ESPN's second-half clock restarts at 45', so a first-half stoppage-time
    event can have a later clock than an early second-half one."""
    return (ev.get("period") or {}).get("number", 0), (ev.get("clock") or {}).get("value", 0)


def _season(payload: dict[str, Any], kickoff: pd.Timestamp) -> int:
    """ESPN's season (its start year: an eng.1 match on 2020-07-04 is 2019), else the one in
    progress at kickoff."""
    year = (payload.get("season") or {}).get("year")
    return int(year) if year is not None else season_start_year(kickoff.to_pydatetime())


def _went_to_extra_time(status: str, events: list[dict]) -> bool:
    """AET, or on penalties with a non-shoot-out key event in period 3+ (extra time; the
    shoot-out is period 5) or a "Start Extra Time" event."""
    if status == AFTER_EXTRA_TIME:
        return True
    if status != ON_PENALTIES:
        return False
    return any(
        not ev.get("shootout")
        and (
            (ev.get("period") or {}).get("number", 0) >= 3
            or ev.get("type", {}).get("text") == "Start Extra Time"
        )
        for ev in events
    )


def parse_summary(payload: dict[str, Any], league: str) -> tuple[dict, list[dict]]:
    """One summary -> its match row and one row per squad player (lineups, minutes, cards)."""
    competition = payload["header"]["competitions"][0]
    status = competition["status"]["type"]["name"]
    end = 120 if _went_to_extra_time(status, payload.get("keyEvents", [])) else 90
    kickoff = pd.Timestamp(competition["date"]).tz_convert("UTC")
    sides = _sides(competition["competitors"])
    match = {
        "espn_event_id": int(payload["header"]["id"]),
        "league": league,
        "season": _season(payload["header"], kickoff),
        "kickoff_time": kickoff,
        "home_espn_id": sides["home"],
        "away_espn_id": sides["away"],
        "status": status,
        "extra_time": end == 120,
    }
    on: dict[int, int] = {}
    off: dict[int, int] = {}
    replaced: set[int] = set()
    yellows: dict[int, int] = {}
    red_at: dict[int, int] = {}
    second_yellow: set[int] = set()
    no_clock = 0
    for ev in sorted(payload.get("keyEvents", []), key=_event_order):
        text = ev["type"]["text"]
        if ev.get("shootout") or text not in MINUTE_EVENTS:
            continue
        seconds = (ev.get("clock") or {}).get("value")
        if seconds is None:
            no_clock += 1
            continue
        ids = [int(p["athlete"]["id"]) for p in ev.get("participants", [])]
        minute = _minute(seconds, end)
        if text == "Substitution" and len(ids) == 2:
            player_in, player_out = ids
            if player_out in off:  # already sent off or replaced: a data glitch, skip
                continue
            on.setdefault(player_in, minute)
            off[player_out] = minute
            replaced.add(player_out)
        elif text == "Yellow Card" and ids:
            yellows[ids[0]] = yellows.get(ids[0], 0) + 1
        elif text == "Red Card" and ids:
            if yellows.get(ids[0], 0) > 0:
                second_yellow.add(ids[0])
            red_at[ids[0]] = minute
            off.setdefault(ids[0], minute)
    if no_clock:
        log.warning(
            "espn event %d: %d key event(s) without a clock skipped",
            match["espn_event_id"],
            no_clock,
        )
    players = []
    seen: set[int] = set()
    for roster in payload.get("rosters", []):
        team = str(roster["team"]["id"])
        for entry in roster.get("roster", []):
            athlete = int(entry["athlete"]["id"])
            if athlete in seen:  # ESPN lists some goalkeepers twice (25 matches, 2016-2026)
                continue
            seen.add(athlete)
            # A substitution flag without its event: when he came on or off is unknown.
            unknown = (entry.get("subbedIn") and athlete not in on) or (
                entry.get("subbedOut") and athlete not in replaced and athlete not in red_at
            )
            start = 0 if entry.get("starter") else on.get(athlete)
            stop = off.get(athlete, end) if start is not None and not unknown else None
            players.append(
                {
                    "espn_athlete_id": athlete,
                    "espn_event_id": match["espn_event_id"],
                    "espn_team_id": team,
                    "athlete_name": entry["athlete"].get("displayName", ""),
                    "starter": bool(entry.get("starter")),
                    "minute_on": start if start is not None else pd.NA,
                    "minute_off": stop if stop is not None else pd.NA,
                    # A player who came on plays at least a minute (FPL's convention: a
                    # stoppage-time substitute, on and off at the capped 90', gets 1).
                    "minutes": (
                        pd.NA if unknown else max(stop - start, 1) if start is not None else 0
                    ),
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
        "minutes": pa.Column("Int64", pa.Check.in_range(0, 120), nullable=True),
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

_INT64 = ("espn_event_id", "espn_athlete_id", "season", "team_key", "yellows")
_NULLABLE_INT = (
    "home_team_key",
    "away_team_key",
    "fixture_key",
    "minute_on",
    "minute_off",
    "minutes",
)
_TIMES = ("kickoff_time", "event_time", "available_at", "snapshot_at")
_BOOLS = ("extra_time", "starter", "red", "second_yellow", "played")


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


FIRST_ROUND_NOTICE = pd.Timedelta(days=14)  # VERIFY (spec §3): a competition's first match
CLUB_MATCH_COLUMNS = (
    "espn_event_id",
    "league",
    "season",
    "kickoff_time",
    "home_team_key",
    "away_team_key",
)
CLUB_MATCH_SCHEMA = pa.DataFrameSchema(
    {
        "espn_event_id": pa.Column("int64", unique=True),
        "league": pa.Column(str, pa.Check.ne("eng.1")),
        "season": pa.Column("int64"),
        "kickoff_time": pa.Column(UTC_US),
        "home_team_key": pa.Column("Int64", nullable=True),
        "away_team_key": pa.Column("Int64", nullable=True),
        "played": pa.Column(bool),
        "event_time": pa.Column(UTC_US),
        "available_at": pa.Column(UTC_US),
    },
    checks=[
        pa.Check(
            lambda df: df["available_at"] < df["kickoff_time"],
            error="a club match is known before it kicks off",
        )
    ],
    strict=True,
    ordered=True,
)
CLUB_MATCH_SORT_BY = ("kickoff_time", "espn_event_id")


def club_match(played: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Every non-league match of a PL club: played ones (`espn_match` rows) and scheduled ones
    (`espn_schedule_snapshot` events not played yet; kickoff and sides from the newest
    snapshot), with `available_at` = when it was known (module docstring, *club_match*)."""
    cols = list(CLUB_MATCH_COLUMNS)
    sort_by = list(CLUB_MATCH_SORT_BY)
    done = played.loc[played["league"] != "eng.1", cols].assign(played=True)
    snaps = schedule[schedule["league"] != "eng.1"].sort_values("snapshot_at", kind="mergesort")
    first_seen = snaps.groupby("espn_event_id")["snapshot_at"].min()
    pending = snaps.drop_duplicates("espn_event_id", keep="last")
    pending = pending.loc[~pending["espn_event_id"].isin(done["espn_event_id"]), cols]
    out = pd.concat([done, pending.assign(played=False)], ignore_index=True)
    out = out.sort_values(sort_by, kind="mergesort").reset_index(drop=True)
    # Rule time per PL side: the club's previous match in the league and season.
    sides = pd.concat(
        [
            out[["espn_event_id", "league", "season", "kickoff_time", c]].rename(
                columns={c: "team_key"}
            )
            for c in ("home_team_key", "away_team_key")
        ]
    ).dropna(subset=["team_key"])
    sides = sides.sort_values(sort_by, kind="mergesort")
    previous = sides.groupby(["team_key", "league", "season"])["kickoff_time"].shift()
    rule = previous.fillna(sides["kickoff_time"] - FIRST_ROUND_NOTICE)
    rule = rule.groupby(sides["espn_event_id"]).max()
    known = pd.to_datetime(out["espn_event_id"].map(rule), utc=True)
    seen = pd.to_datetime(out["espn_event_id"].map(first_seen), utc=True)
    known = known.where(~(seen < known), seen)  # NaT compares False: no snapshot, keep the rule
    latest = pd.to_datetime(out["kickoff_time"], utc=True) - pd.Timedelta(microseconds=1)
    out["available_at"] = known.where(known < latest, latest)
    out["event_time"] = out["kickoff_time"]
    return _frame(out.to_dict("records"), CLUB_MATCH_SCHEMA)


def _sides(competitors: list[dict]) -> dict[str, str]:
    return {c["homeAway"]: str(c["team"]["id"]) for c in competitors}


def _summaries(
    ctx: BuildContext, resolver: TeamResolver, problems: list[str]
) -> tuple[list[dict], list[dict]]:
    """Match and player rows from the newest summary of every finished event."""
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
    unknown: Counter[str] = Counter()
    for league, event_id in _item_dirs(ctx.store, "summary"):
        path = ctx.store.latest(SOURCE, f"{league}/summary/{event_id}")
        payload = ctx.store.read_json(path)
        status = payload["header"]["competitions"][0]["status"]["type"]["name"]
        if status not in FINISHED_STATUSES:
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
                unknown[league] += pd.isna(row["minutes"])
                players.append(
                    {**row, "season": match["season"], "league": league, "team_key": team_key}
                    | times
                )
    events_of = Counter(m["fixture_key"] for m in matches if m["fixture_key"] is not None)
    for fixture_key, n in sorted(events_of.items()):
        if n > 1:
            problems.append(f"fixture {fixture_key} has {n} eng.1 events")
    for league, n in sorted(unknown.items()):
        if n:
            log.info(
                "espn: minutes unknown (substitution flag without its event) in %s: %d row(s)",
                league,
                n,
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
                        "season": _season(ev, kickoff),
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
GRACE = pd.Timedelta(days=2)  # fixtures this close to the newest ESPN eng.1 match may be missing

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
    minutes_src (Int64, NA = unknown), red, athlete_name (minutes 0 included)."""
    match = tables["espn_match"].dropna(subset=["fixture_key"])
    rows = tables["espn_player_match"].merge(
        match[["espn_event_id", "fixture_key"]], on="espn_event_id"
    )
    rows = rows.assign(fixture_key=rows["fixture_key"].astype("int64"), minutes_src=rows["minutes"])
    return rows[[ID, "fixture_key", "minutes_src", "red", "athlete_name"]]


def map_espn(ctx: BuildContext, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """espn_map: the Understat matcher on ESPN's PL appearances (minutes > 0) against FPL's,
    names from espn_player_match vs player_season, overrides with source=espn."""
    espn = pl_appearances(tables).dropna(subset=["minutes_src"])
    played = espn[espn["minutes_src"] > 0].astype({"minutes_src": "int64"})
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
    # A pair override applies only to an athlete with PL minutes in the data; blank-id rows
    # (an FPL player without an ESPN id) always apply.
    overrides = overrides[overrides[ID].isna() | overrides[ID].isin(set(played[ID]))]
    return map_understat(pairs, overrides.reset_index(drop=True), id_col=ID)


def espn_problems(
    ctx: BuildContext, tables: dict[str, pd.DataFrame], mapping: pd.DataFrame
) -> list[str]:
    """The build checks (module docstring) over seasons with any eng.1 match; empty = valid.
    The player checks of the newest covered season are logged as warnings instead."""
    match = tables["espn_match"]
    pl = match[match["league"] == "eng.1"]
    covered = set(pl["season"])
    if not covered:
        return []
    problems: list[str] = []

    fixture = ctx.table("fixture")
    cutoff = pl["kickoff_time"].max() - GRACE
    finished = fixture[
        fixture["finished"] & fixture["season"].isin(covered) & (fixture["kickoff_time"] < cutoff)
    ]
    missing = finished[~finished["fixture_key"].isin(set(match["fixture_key"].dropna()))]
    if len(missing):
        problems.append(
            f"{len(missing)} finished PL fixtures have no ESPN match: "
            f"{missing['fixture_key'].head(MAX_LISTED).tolist()}"
        )

    newest = max(covered)
    problems += _player_problems(ctx, tables, mapping, covered - {newest})
    for warning in _player_problems(ctx, tables, mapping, {newest}):
        log.warning("espn (season %d, in progress or newest): %s", newest, warning)
    return problems


def _player_problems(
    ctx: BuildContext, tables: dict[str, pd.DataFrame], mapping: pd.DataFrame, seasons: set[int]
) -> list[str]:
    """The mapping, minutes and red-card checks over FPL's rows of `seasons`."""
    if not seasons:
        return []
    problems: list[str] = []
    pm = ctx.table("player_match")
    pm = pm[pm["season"].isin(seasons)]
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
    known = both.dropna(subset=["minutes_src"]).astype({"minutes_src": "int64"})
    played = known[
        (known["method"] == "auto") & ((known["minutes"] > 0) | (known["minutes_src"] > 0))
    ]
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
    clubs = club_match(tables["espn_match"], tables["espn_schedule_snapshot"])
    write_table(clubs, "club_match", CLUB_MATCH_SCHEMA, ctx.data_dir, CLUB_MATCH_SORT_BY)
    ctx.forget("club_match")
    log.info(
        "espn_map: %d athletes mapped (%d override row(s)) of %d with PL minutes",
        int(mapping["player_key"].notna().sum()),
        int((mapping["method"] == "override").sum()),
        pl_appearances(tables)
        .dropna(subset=["minutes_src"])
        .query("minutes_src > 0")[ID]
        .nunique(),
    )
    return tables["espn_player_match"]
