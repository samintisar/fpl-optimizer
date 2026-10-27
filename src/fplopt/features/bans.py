"""Suspensions, served where they really are (spec docs/superpowers/specs/2026-10-27-espn-cup-
data-design.md §2). Pure functions over frames (no view), so they are tested directly.

- A red card in a domestic match (PL, FA Cup, EFL Cup) bans the club's next *domestic* matches,
  so a cup tie in between uses a match of the ban up. A straight red is 1 certain match plus
  `STRAIGHT_RED_EXTRA` likely ones (a third match is the usual ban; rescinded reds and appeals
  are absorbed by the per-kind fit in `fplopt.models.minutes`), a second yellow is 1. A red
  without ESPN data counts as 1.
- PL yellow-card bans (`BAN_RULES`) are served in PL fixtures only.

`ban_kind` codes: `BAN_NONE` 0, `BAN_LIKELY` 1, `BAN_CERTAIN` 2 (overlapping bans combine with
`max`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BAN_NONE = 0
BAN_LIKELY = 1
BAN_CERTAIN = 2
STRAIGHT_RED_EXTRA = 2  # likely matches after the certain one for a straight red
# (yellow cards this season, last gw_index it counts by, matches banned): PLAN §6.3
BAN_RULES = ((5, 19, 1), (10, 32, 2), (15, 99, 3))
SCOPE_DOMESTIC = "domestic"
SCOPE_LEAGUE = "league"

EVENT_COLUMNS = ("player_key", "team_key", "time", "scope", "n_certain", "n_likely")
TIMELINE_COLUMNS = ("team_key", "kickoff_time", "fixture_key", "domestic_cup")
SERVED_COLUMNS = ("player_key", "fixture_key", "ban_kind")


def _utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True).astype("datetime64[us, UTC]")


def _red_lengths(second_yellow: pd.Series) -> tuple[pd.Series, pd.Series]:
    """(n_certain, n_likely) of red cards by `second_yellow` (True / False / missing)."""
    straight = second_yellow.astype("boolean").eq(False).fillna(False).to_numpy(dtype=bool)
    certain = pd.Series(1, index=second_yellow.index, dtype="int64")
    likely = pd.Series(np.where(straight, STRAIGHT_RED_EXTRA, 0), index=second_yellow.index)
    return certain, likely.astype("int64")


def _events(frame: pd.DataFrame, scope: str, certain, likely) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_key": frame["player_key"].to_numpy(),
            "team_key": frame["team_key"].to_numpy(),
            "time": _utc(frame["kickoff_time"]).to_numpy(),
            "scope": scope,
            "n_certain": np.asarray(certain, dtype="int64"),
            "n_likely": np.asarray(likely, dtype="int64"),
        }
    )


def ban_events(
    league_rows: pd.DataFrame, red_kinds: pd.DataFrame, cup_reds: pd.DataFrame
) -> pd.DataFrame:
    """One row per ban: player_key, team_key, time, scope ('domestic' | 'league'), n_certain,
    n_likely.

    `league_rows`: player_key, season, gw_index, fixture_key, team_key, kickoff_time,
    yellow_cards, red_cards (history's match rows). `red_kinds`: player_key, fixture_key,
    second_yellow from ESPN's PL rows. `cup_reds`: player_key, team_key, kickoff_time,
    second_yellow of reds in domestic cup matches."""
    parts = []

    rows = league_rows.sort_values(["player_key", "season", "gw_index"], kind="mergesort")
    yellow = rows["yellow_cards"].to_numpy(dtype="float64")
    total = (
        pd.Series(yellow, index=rows.index)
        .groupby([rows["player_key"], rows["season"]])
        .cumsum()
        .to_numpy()
    )
    before = total - yellow
    gw_index = rows["gw_index"].to_numpy()
    length = np.zeros(len(rows), dtype="int64")
    for cards, last_gw, matches in BAN_RULES:
        length += matches * ((before < cards) & (total >= cards) & (gw_index <= last_gw))
    hit = length > 0
    parts.append(_events(rows[hit], SCOPE_LEAGUE, length[hit], np.zeros(hit.sum())))

    reds = rows[rows["red_cards"].to_numpy() > 0]
    kinds = red_kinds.drop_duplicates(["player_key", "fixture_key"])[
        ["player_key", "fixture_key", "second_yellow"]
    ]
    reds = reds.merge(kinds, on=["player_key", "fixture_key"], how="left")
    certain, likely = _red_lengths(reds["second_yellow"])
    parts.append(_events(reds, SCOPE_DOMESTIC, certain, likely))

    certain, likely = _red_lengths(cup_reds["second_yellow"])
    parts.append(_events(cup_reds, SCOPE_DOMESTIC, certain, likely))

    out = pd.concat(parts, ignore_index=True)
    out["time"] = _utc(out["time"])
    return out[list(EVENT_COLUMNS)]


def club_timelines(league_fixtures: pd.DataFrame, cup_matches: pd.DataFrame) -> pd.DataFrame:
    """Each club's domestic matches in time order: team_key, kickoff_time, fixture_key (Int64;
    null for a cup match), domestic_cup. `league_fixtures`: team_key, fixture_key,
    kickoff_time; `cup_matches`: team_key, kickoff_time, one row per PL side of every
    domestic cup match."""
    league = league_fixtures[["team_key", "fixture_key", "kickoff_time"]].assign(domestic_cup=False)
    cups = cup_matches[["team_key", "kickoff_time"]].assign(fixture_key=pd.NA, domestic_cup=True)
    out = pd.concat([league, cups], ignore_index=True)
    out["kickoff_time"] = _utc(out["kickoff_time"])
    out["fixture_key"] = out["fixture_key"].astype("Int64")
    out["domestic_cup"] = out["domestic_cup"].astype(bool)
    out = out.sort_values(["team_key", "kickoff_time"], kind="mergesort").reset_index(drop=True)
    return out[list(TIMELINE_COLUMNS)]


def serve_bans(events: pd.DataFrame, timelines: pd.DataFrame) -> pd.DataFrame:
    """player_key, fixture_key, ban_kind for the league fixtures each ban is served in (kind >
    0): the club's matches after the ban's `time`, in order (scope 'league': league fixtures
    only); the first `n_certain` are `BAN_CERTAIN`, the next `n_likely` `BAN_LIKELY`; a cup
    match uses up its position without a row. Overlapping bans keep the maximum."""
    by_team = {
        team: (
            _micros(group["kickoff_time"]),
            group["fixture_key"].to_numpy(dtype="float64", na_value=np.nan),
        )
        for team, group in timelines.groupby("team_key")
    }
    rows = []
    for event in events.itertuples(index=False):
        if event.team_key not in by_team:
            continue
        kickoffs, fixtures = by_team[event.team_key]
        after = np.searchsorted(kickoffs, _micros(pd.Series([event.time]))[0], side="right")
        upcoming = fixtures[after:]
        if event.scope == SCOPE_LEAGUE:
            upcoming = upcoming[~np.isnan(upcoming)]
        n_certain, n_likely = int(event.n_certain), int(event.n_likely)
        for position, fixture in enumerate(upcoming[: n_certain + n_likely]):
            if not np.isnan(fixture):
                kind = BAN_CERTAIN if position < n_certain else BAN_LIKELY
                rows.append((event.player_key, int(fixture), kind))
    out = pd.DataFrame(rows, columns=list(SERVED_COLUMNS))
    out = out.groupby(["player_key", "fixture_key"], as_index=False)["ban_kind"].max()
    return out.astype({"player_key": "int64", "fixture_key": "int64", "ban_kind": "int64"})


def _micros(times: pd.Series) -> np.ndarray:
    """UTC instants as int64 microseconds (comparable whatever the frame's time unit)."""
    return _utc(times).astype("int64").to_numpy()
