"""ESPN summaries -> espn_match, espn_player_match, espn_schedule_snapshot, espn_map (spec
2026-10-27 §1). See the module docstring's sections below for the tables and checks."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

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
    sides = {c["homeAway"]: str(c["team"]["id"]) for c in competition["competitors"]}
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
