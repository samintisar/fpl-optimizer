"""A synthetic league as in-memory derived tables, for `DataStore(tables=...)` (Phase 3 plan,
Task 4). Shared by the model, policy, start-state and simulator tests.

`synthetic_tables()` returns `{name: DataFrame}` with the columns, dtypes and `available_at`
rules of the real `data/*.parquet` tables (PLAN §3, §4):

- `gameweek`, `schedule`: one double round robin per season (`2 * (n_clubs - 1)` GWs, 38 for
  20 clubs), weekly Friday 17:30 UTC deadlines from early August; kickoffs Friday evening to
  Monday; lockdown 08:00 UTC the day after the GW's last kickoff. `available_at` = 1 June of
  the season's start year (or the previous generated season's last lockdown, if later).
- `fixture`, `team_match`, `gameweek_result`, `player_match`: results, `available_at` = the
  GW lockdown (`event_time` = kickoff; `gameweek_result.event_time` = lockdown).
- `player_season`: `available_at` = GW1 deadline - 1h (`event_time` = GW1 deadline).
- `player_gw` (registered club, position, price at the deadline): `available_at` = deadline
  - 1h. No row for a GW in which the player's club blanks (as vaastav).
- `player_gw_ownership`: `available_at` = deadline.
- `player_snapshot` (if `snapshots`): one snapshot per GW at deadline - 2h listing every
  player of the season (source `fplcache`; `available_at` = `event_time` = `snapshot_at`).
  Otherwise an empty frame with the same columns.
- `team_rating`: Elo, one pre-season row per club and season (available like the schedule)
  and one row per fixture and side (`available_at` = kickoff + 2h).
- `odds_snapshot`, `fixture_snapshot`: empty, with the real columns and dtypes.
- `team_dim`, `player_dim`: static identity tables (`available_at` = epoch).

Players stay with their club (and keep their `player_key`) across the generated seasons;
`element_id` changes per season. Per club `players_per_club` = (GK, DEF, MID, FWD); the first
GK, 4 DEF, 4 MID and 2 FWD are regular starters, the rest rotate in rarely, and the second GK
never plays (0-minute rows). Players get injured now and then (snapshot status `i` with
chance 0: they don't play; then `d` with chance 50 for one GW). Prices start at 40-130 by
position and quality and move by 0.1 between GWs now and then. Match stats come from a seeded
generator; `total_points` follows FPL scoring of 2016/17-2024/25 (`fpl_points`). `ep_next` is
the player's season mean points per fixture (with noise) x his club's fixtures in the GW
(0 in a blank) x his availability; `form` is his mean points over the matches of the 30 days
before the snapshot.

Blanks and doubles (`blank`, `double`; one tuple or a sequence of tuples; `club` is the
`team_key`, clubs are numbered 1..n_clubs):
- `blank=(season, gw, club[, to_gw])`: the club's fixture of `gw` is played in `to_gw`
  (default gw + 2) instead: both clubs blank in `gw` and double in `to_gw`;
- `double=(season, gw, club[, from_gw])`: the club's fixture of `from_gw` (default gw + 2) is
  played in `gw`: both clubs double in `gw` and blank in `from_gw`.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from functools import lru_cache

import numpy as np
import pandas as pd

UTC_US = pd.DatetimeTZDtype("us", "UTC")
EPOCH = pd.Timestamp(0, tz="UTC").as_unit("us")
POSITION_NAMES = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
STARTERS = {1: 1, 2: 4, 3: 4, 4: 2}  # regular starters per club and position
GOAL_POINTS = {1: 6, 2: 6, 3: 5, 4: 4}
CLEAN_SHEET_POINTS = {1: 4, 2: 4, 3: 1, 4: 0}
ATTACK = {1: 0.0, 2: 0.12, 3: 0.45, 4: 1.0}  # share of a club's goals, x quality
CREATE = {1: 0.03, 2: 0.3, 3: 0.8, 4: 0.5}  # share of its assists, x quality
PRICE = {1: (40, 20), 2: (40, 35), 3: (45, 85), 4: (45, 80)}  # (base, spread) in £0.1m
PRICE_RANGE = (38, 135)
KICKOFF_HOURS = (1.5, 19, 21.5, 21.5, 21.5, 21.5, 24, 44.5, 47, 74.5)  # after the deadline
MOVED_KICKOFF_HOURS = 50.0  # a rearranged fixture: Sunday evening of its new GW
SNAPSHOT_LEAD = pd.Timedelta(hours=2)
REGISTRATION_LEAD = pd.Timedelta(hours=1)
RATING_DELAY = pd.Timedelta(hours=2)
FORM_DAYS = pd.Timedelta(days=30)

Move = tuple[int, ...]
SCORED_STATS = (
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
)


def fpl_points(
    element_type: int,
    minutes: int,
    goals_scored: int = 0,
    assists: int = 0,
    clean_sheets: int = 0,
    goals_conceded: int = 0,
    own_goals: int = 0,
    penalties_saved: int = 0,
    penalties_missed: int = 0,
    yellow_cards: int = 0,
    red_cards: int = 0,
    saves: int = 0,
    bonus: int = 0,
) -> int:
    """FPL points of one player-match under the 2016/17-2024/25 rules (no defcon)."""
    if minutes <= 0:
        return 0
    points = 2 if minutes >= 60 else 1
    points += GOAL_POINTS[element_type] * goals_scored + 3 * assists
    points += CLEAN_SHEET_POINTS[element_type] * clean_sheets
    if element_type in (1, 2):
        points -= goals_conceded // 2
    points += saves // 3 + 5 * penalties_saved - 2 * penalties_missed
    points += -yellow_cards - 3 * red_cards - 2 * own_goals + bonus
    return points


def team_keys(n_clubs: int = 20) -> list[int]:
    return list(range(1, n_clubs + 1))


def player_key(team_key: int, slot: int) -> int:
    """Slot = index within the club, positions in order (GK first)."""
    return 100_000 + 100 * team_key + slot


def first_deadline(season: int) -> pd.Timestamp:
    """GW1 deadline: the first Friday on or after 8 August, 17:30 UTC."""
    day = pd.Timestamp(f"{season}-08-08", tz="UTC")
    return (day + pd.Timedelta(days=(4 - day.weekday()) % 7, hours=17, minutes=30)).as_unit("us")


def synthetic_tables(
    seasons: Sequence[int] = (2023,),
    n_clubs: int = 20,
    players_per_club: Sequence[int] = (2, 5, 5, 3),
    seed: int = 0,
    snapshots: bool = True,
    blank: Move | Sequence[Move] | None = None,
    double: Move | Sequence[Move] | None = None,
) -> dict[str, pd.DataFrame]:
    """The synthetic league's tables (see the module docstring); a fresh copy per call,
    identical for identical arguments."""
    built = _build(
        tuple(seasons),
        n_clubs,
        tuple(players_per_club),
        seed,
        snapshots,
        _moves(blank, double),
    )
    return {name: df.copy() for name, df in built.items()}


def _as_list(moves: Move | Sequence[Move] | None) -> list[Move]:
    if moves is None:
        return []
    if len(moves) and isinstance(moves[0], int | np.integer):
        return [tuple(moves)]
    return [tuple(m) for m in moves]


def _moves(blank, double) -> tuple[tuple[int, int, int, int], ...]:
    """(season, from_gw, club, to_gw) per moved fixture."""
    out = []
    for move in _as_list(blank):
        season, gw, club, *rest = move
        out.append((season, gw, club, rest[0] if rest else gw + 2))
    for move in _as_list(double):
        season, gw, club, *rest = move
        out.append((season, rest[0] if rest else gw + 2, club, gw))
    return tuple(out)


# --- the league ----------------------------------------------------------------------------


def _round_robin(teams: list[int], rng: np.random.Generator) -> list[list[tuple[int, int]]]:
    """Circle-method double round robin: per round the (home, away) pairs."""
    order = [int(t) for t in rng.permutation(teams)]
    n = len(order)
    rounds = []
    for r in range(n - 1):
        pairs = []
        for i in range(n // 2):
            a, b = order[i], order[n - 1 - i]
            pairs.append((a, b) if (r + i) % 2 == 0 else (b, a))
        rounds.append(pairs)
        order = [order[0], order[-1], *order[1:-1]]
    return rounds + [[(b, a) for a, b in pairs] for pairs in rounds]


def _players(teams: list[int], per_club: tuple[int, ...], rng: np.random.Generator) -> list[dict]:
    players = []
    for team in teams:
        slot = 0
        for element_type, count in zip((1, 2, 3, 4), per_club, strict=True):
            for i in range(count):
                starter = i < STARTERS[element_type]
                if starter:
                    p_start = 0.97 if element_type == 1 else float(rng.uniform(0.7, 0.97))
                    quality = 0.9 + 0.7 * float(rng.beta(1.5, 4))  # a few premiums
                else:
                    never = element_type == 1  # the backup GK never plays
                    p_start = 0.0 if never else float(rng.uniform(0.02, 0.25))
                    quality = float(rng.uniform(0.5, 1.0))
                base, spread = PRICE[element_type]
                price = base + spread * min(max((quality - 0.5) / 1.1, 0.0), 1.0)
                players.append(
                    {
                        "player_key": player_key(team, slot),
                        "team_key": team,
                        "element_type": element_type,
                        "p_start": p_start,
                        "p_sub": 0.0 if element_type == 1 else (0.1 if starter else 0.35),
                        "quality": quality,
                        "attack": ATTACK[element_type] * quality,
                        "create": CREATE[element_type] * quality,
                        "price": int(round(price)),
                        "popularity": quality**4 * max(p_start, 0.05),
                    }
                )
                slot += 1
    return players


def _schedule_available(season: int, previous_lockdown: pd.Timestamp | None) -> pd.Timestamp:
    june = pd.Timestamp(f"{season}-06-01", tz="UTC").as_unit("us")
    return june if previous_lockdown is None else max(june, previous_lockdown)


def _fixtures(season: int, teams: list[int], moves, rng: np.random.Generator) -> pd.DataFrame:
    rounds = _round_robin(teams, rng)
    deadline0 = first_deadline(season)
    rows = []
    for gw, pairs in enumerate(rounds, start=1):
        deadline = deadline0 + pd.Timedelta(weeks=gw - 1)
        for slot, (home, away) in enumerate(pairs):
            hours = KICKOFF_HOURS[slot % len(KICKOFF_HOURS)]
            rows.append([gw, home, away, deadline + pd.Timedelta(hours=hours)])
    fixtures = pd.DataFrame(rows, columns=["gw", "home_team_key", "away_team_key", "kickoff_time"])
    for move_season, from_gw, club, to_gw in moves:
        if move_season != season:
            continue
        if not 1 <= to_gw <= len(rounds):
            raise ValueError(f"cannot move a GW{from_gw} fixture to GW{to_gw}")
        at = (fixtures["gw"] == from_gw) & (
            (fixtures["home_team_key"] == club) | (fixtures["away_team_key"] == club)
        )
        if at.sum() != 1:
            raise ValueError(f"club {club} has {int(at.sum())} fixtures in {season} GW{from_gw}")
        moved_deadline = deadline0 + pd.Timedelta(weeks=to_gw - 1)
        fixtures.loc[at, "gw"] = to_gw
        fixtures.loc[at, "kickoff_time"] = moved_deadline + pd.Timedelta(hours=MOVED_KICKOFF_HOURS)
    # FPL numbers fixtures by kickoff.
    fixtures = fixtures.sort_values(["kickoff_time", "home_team_key"], kind="mergesort")
    fixtures = fixtures.reset_index(drop=True)
    fixtures["fpl_fixture_id"] = np.arange(1, len(fixtures) + 1)
    fixtures["fixture_key"] = season * 1000 + fixtures["fpl_fixture_id"]
    fixtures["season"] = season
    fixtures["kickoff_time"] = fixtures["kickoff_time"].astype(UTC_US)
    return fixtures


def _gameweeks(season: int, n_gws: int, fixtures: pd.DataFrame) -> pd.DataFrame:
    deadline0 = first_deadline(season)
    kickoffs = fixtures.groupby("gw")["kickoff_time"]
    out = pd.DataFrame({"season": season, "gw": np.arange(1, n_gws + 1)})
    out["gw_index"] = out["gw"]
    out["deadline_time"] = [deadline0 + pd.Timedelta(weeks=gw - 1) for gw in out["gw"]]
    out["first_kickoff"] = out["gw"].map(kickoffs.min())
    out["last_kickoff"] = out["gw"].map(kickoffs.max())
    # A GW without fixtures (every club blanks) is not generated; lockdown from its deadline.
    last = out["last_kickoff"].fillna(out["deadline_time"])
    out["lockdown_time"] = last.dt.normalize() + pd.Timedelta(days=1, hours=8)
    return out


# --- matches -------------------------------------------------------------------------------


def _availability(players: list[dict], n_gws: int, rng: np.random.Generator) -> dict:
    """Per player key: (status, chance) per GW index 0..n_gws-1."""
    out = {}
    for player in players:
        remaining = 0
        states = []
        for _ in range(n_gws):
            if remaining == 0 and rng.random() < 0.02:
                remaining = int(rng.integers(2, 5))
            if remaining > 1:
                states.append(("i", 0))
            elif remaining == 1:
                states.append(("d", 50))
            else:
                states.append(("a", None))
            remaining = max(remaining - 1, 0)
        out[player["player_key"]] = states
    return out


def _side_minutes(squad: list[dict], states: dict, gw: int, rng: np.random.Generator) -> list:
    """Per squad player: (minutes, started, on, off) for one match."""
    out = []
    for player in squad:
        status, _ = states[player["player_key"]][gw - 1]
        factor = {"i": 0.0, "d": 0.5, "a": 1.0}[status]
        if rng.random() < player["p_start"] * factor:
            minutes = 90 if rng.random() < 0.7 else int(rng.integers(55, 90))
            out.append((minutes, 1, 0, minutes))
        elif rng.random() < player["p_sub"] * factor:
            minutes = int(rng.integers(1, 36))
            out.append((minutes, 0, 90 - minutes, 90))
        else:
            out.append((0, 0, 0, 0))
    return out


def _pick(weights: np.ndarray, rng: np.random.Generator) -> int | None:
    total = weights.sum()
    if total <= 0:
        return None
    return int(rng.choice(len(weights), p=weights / total))


def _match_rows(
    fixture: pd.Series,
    squads: dict[int, list[dict]],
    states: dict,
    strength: dict[int, tuple[float, float]],
    element_ids: dict[int, int],
    rng: np.random.Generator,
) -> tuple[list[dict], list[dict], tuple[int, int]]:
    """player_match rows of one fixture, its two team_match rows and the score."""
    home, away = int(fixture["home_team_key"]), int(fixture["away_team_key"])
    gw = int(fixture["gw"])
    lam = {
        home: 1.45 * strength[home][0] / strength[away][1],
        away: 1.15 * strength[away][0] / strength[home][1],
    }
    sides = {team: _side_minutes(squads[team], states, gw, rng) for team in (home, away)}
    goals = {team: int(rng.poisson(lam[team])) for team in (home, away)}
    goal_minutes = {team: np.sort(rng.integers(1, 91, goals[team])) for team in (home, away)}
    stats = {}
    for team in (home, away):
        squad, minutes = squads[team], sides[team]
        n = len(squad)
        scored, assisted = np.zeros(n, int), np.zeros(n, int)
        for minute in goal_minutes[team]:
            on = np.array([m > 0 and on_ <= minute <= off for (m, _, on_, off) in minutes])
            scorer = _pick(np.array([p["attack"] for p in squad]) * on, rng)
            if scorer is None:
                continue
            scored[scorer] += 1
            if rng.random() < 0.7:
                weights = np.array([p["create"] for p in squad]) * on
                weights[scorer] = 0
                helper = _pick(weights, rng)
                if helper is not None:
                    assisted[helper] += 1
        stats[team] = (scored, assisted)
    rows, team_rows = [], []
    for team, opponent in ((home, away), (away, home)):
        squad, minutes = squads[team], sides[team]
        scored, assisted = stats[team]
        attack_on = sum(p["attack"] * m[0] / 90 for p, m in zip(squad, minutes, strict=True))
        create_on = sum(p["create"] * m[0] / 90 for p, m in zip(squad, minutes, strict=True))
        side_rows = []
        for i, player in enumerate(squad):
            mins, started, on, off = minutes[i]
            played = mins > 0
            element_type = player["element_type"]
            conceded = int(sum(on <= m <= off for m in goal_minutes[opponent])) if played else 0
            row = {
                "minutes": mins,
                "starts": started,
                "goals_scored": int(scored[i]),
                "assists": int(assisted[i]),
                "clean_sheets": int(mins >= 60 and conceded == 0),
                "goals_conceded": conceded,
                "own_goals": int(played and rng.random() < 0.002),
                "penalties_saved": int(played and element_type == 1 and rng.random() < 0.02),
                "penalties_missed": int(played and element_type in (3, 4) and rng.random() < 0.004),
                "yellow_cards": 0,
                "red_cards": 0,
                "saves": int(rng.poisson(2.5 * mins / 90)) if played and element_type == 1 else 0,
            }
            if played:
                if rng.random() < 0.1 * mins / 90:
                    row["yellow_cards"] = 1
                elif rng.random() < 0.004 * mins / 90:
                    row["red_cards"] = 1
            base = fpl_points(element_type, mins, **{k: row[k] for k in SCORED_STATS[:-1]})
            row["bps"] = 3 * base + int(rng.integers(0, 5)) if played else 0
            if played:
                share = mins / 90
                xg = lam[team] * player["attack"] * share / max(attack_on, 1e-9)
                xa = 0.7 * lam[team] * player["create"] * share / max(create_on, 1e-9)
                xg *= float(rng.lognormal(0, 0.3))
                xa *= float(rng.lognormal(0, 0.3))
            else:
                xg = xa = 0.0
            row.update(
                player_key=player["player_key"],
                element_type=element_type,
                team_key=team,
                opponent_team_key=opponent,
                was_home=team == home,
                xg=xg,
                xa=xa,
                xgc=lam[opponent] * mins / 90,
                shots=int(rng.poisson(3 * xg + 0.2)) if played else 0,
                key_passes=int(rng.poisson(3 * xa + 0.2)) if played else 0,
            )
            side_rows.append(row)
        rows += side_rows
        team_rows.append(
            {
                "team_key": team,
                "is_home": team == home,
                "goals_for": goals[team],
                "goals_against": goals[opponent],
                "xg": lam[team] * float(rng.lognormal(0, 0.2)),
                "fpl_xg": round(sum(r["xg"] for r in side_rows), 2),
            }
        )
    # Bonus: 3/2/1 to the top three BPS of the match (ties by player key).
    ranked = sorted(
        (r for r in rows if r["minutes"] > 0), key=lambda r: (-r["bps"], r["player_key"])
    )
    for r in rows:
        r["bonus"] = 0
    for r, bonus in zip(ranked, (3, 2, 1), strict=False):
        r["bonus"] = bonus
    for r in rows:
        stats_only = {k: r[k] for k in SCORED_STATS}
        r["total_points"] = fpl_points(r["element_type"], r["minutes"], **stats_only)
        r["element_id"] = element_ids[r["player_key"]]
    for row in team_rows:
        opponent = next(r for r in team_rows if r is not row)
        row["xga"] = opponent["xg"]
        row["fpl_xga"] = opponent["fpl_xg"]
    return rows, team_rows, (goals[home], goals[away])


# --- tables --------------------------------------------------------------------------------


def _player_match_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    played = df["minutes"] > 0
    nullable = lambda s: pd.array(s.where(played), dtype="Int64")  # noqa: E731
    floats = lambda s, d: pd.array(s.round(d).where(played), dtype="Float64")  # noqa: E731
    out = pd.DataFrame(
        {
            "player_key": df["player_key"].astype("int64"),
            "season": df["season"].astype("int64"),
            "fixture_key": df["fixture_key"].astype("int64"),
            "gw": df["gw"].astype("int64"),
            "element_id": df["element_id"].astype("int64"),
            "team_key": df["team_key"].astype("int64"),
            "opponent_team_key": df["opponent_team_key"].astype("int64"),
            "was_home": df["was_home"].astype("bool"),
            "kickoff_time": df["kickoff_time"].astype(UTC_US),
            "minutes": df["minutes"].astype("int64"),
            "starts": pd.array(df["starts"], dtype="Int64"),
        }
    )
    for column in (
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
    ):
        out[column] = df[column].astype("int64")
    for column in (
        "clearances_blocks_interceptions",
        "recoveries",
        "tackles",
        "defensive_contribution",
    ):
        out[column] = pd.array([None] * len(df), dtype="Int64")
    out["fpl_xg"] = pd.array(df["xg"].round(2), dtype="Float64")
    out["fpl_xa"] = pd.array(df["xa"].round(2), dtype="Float64")
    out["fpl_xgc"] = pd.array(df["xgc"].round(2), dtype="Float64")
    out["us_minutes"] = nullable(df["minutes"])
    out["us_goals"] = nullable(df["goals_scored"])
    out["us_npg"] = nullable(df["goals_scored"])
    out["us_xg"] = floats(df["xg"], 6)
    out["us_npxg"] = floats(df["xg"], 6)
    out["us_xa"] = floats(df["xa"], 6)
    out["us_shots"] = nullable(df["shots"])
    out["us_key_passes"] = nullable(df["key_passes"])
    out["source"] = pd.array(["vaastav"] * len(df), dtype="str")
    out["event_time"] = out["kickoff_time"]
    out["available_at"] = df["available_at"].astype(UTC_US)
    return out.sort_values(["season", "gw", "fixture_key", "player_key"], kind="mergesort")


def _empty(columns: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series([], dtype=dtype) for name, dtype in columns.items()})


SNAPSHOT_DTYPES = {
    "snapshot_at": UTC_US,
    "source": "str",
    "season": "int64",
    "element_id": "int64",
    "player_key": "int64",
    "team_key": "int64",
    "element_type": "int64",
    "now_cost": "int64",
    "status": "str",
    "chance_of_playing_next_round": "Int64",
    "chance_of_playing_this_round": "Int64",
    "news": "str",
    "news_added": UTC_US,
    "selected_by_percent": "Float64",
    "ep_next": "Float64",
    "ep_this": "Float64",
    "form": "Float64",
    "penalties_order": "Int64",
    "corners_and_indirect_freekicks_order": "Int64",
    "direct_freekicks_order": "Int64",
    "team_join_date": "date32[day][pyarrow]",
    "transfers_in_event": "int64",
    "transfers_out_event": "int64",
    "cost_change_event": "int64",
    "event_time": UTC_US,
    "available_at": UTC_US,
}
ODDS_DTYPES = {
    "fixture_key": "int64",
    "season": "int64",
    "source": "str",
    "bookmaker": "str",
    "market": "str",
    "outcome": "str",
    "line": "Float64",
    "price": "float64",
    "is_closing": "bool",
    "snapshot_at": UTC_US,
    "event_time": UTC_US,
    "available_at": UTC_US,
}
FIXTURE_SNAPSHOT_DTYPES = {
    "snapshot_at": UTC_US,
    "season": "int64",
    "fixture_key": "int64",
    "fpl_fixture_id": "int64",
    "gw": "Int64",
    "kickoff_time": UTC_US,
    "home_team_key": "int64",
    "away_team_key": "int64",
    "started": "boolean",
    "finished": "boolean",
    "finished_provisional": "boolean",
    "event_time": UTC_US,
    "available_at": UTC_US,
}


def _form(history: tuple[list[int], list[int]], at: pd.Timestamp) -> float:
    """Mean points over the matches of the FORM_DAYS before `at` (history: kickoffs in ns,
    ascending, and points)."""
    kickoffs, points = history
    hi = bisect.bisect_left(kickoffs, at.value)
    lo = bisect.bisect_left(kickoffs, (at - FORM_DAYS).value)
    return round(sum(points[lo:hi]) / (hi - lo), 1) if hi > lo else 0.0


@lru_cache(maxsize=16)
def _build(
    seasons: tuple[int, ...],
    n_clubs: int,
    per_club: tuple[int, ...],
    seed: int,
    snapshots: bool,
    moves: tuple[tuple[int, int, int, int], ...],
) -> dict[str, pd.DataFrame]:
    if n_clubs < 2 or n_clubs % 2:
        raise ValueError("n_clubs must be even and at least 2")
    if list(seasons) != sorted(set(seasons)):
        raise ValueError("seasons must be increasing")
    rng = np.random.default_rng(seed)
    teams = team_keys(n_clubs)
    players = _players(teams, per_club, rng)
    squads = {team: [p for p in players if p["team_key"] == team] for team in teams}
    strength = {t: (float(rng.uniform(0.75, 1.35)), float(rng.uniform(0.75, 1.35))) for t in teams}
    elo = {t: 1500 + 400 * (strength[t][0] * strength[t][1] - 1) for t in teams}
    n_gws = 2 * (n_clubs - 1)

    out: dict[str, list[pd.DataFrame]] = {}
    add = lambda name, df: out.setdefault(name, []).append(df)  # noqa: E731
    previous_lockdown = None
    # Per player: kickoffs (ns, ascending) and points of his matches so far (for `form`).
    history = {p["player_key"]: ([], []) for p in players}
    for season in seasons:
        available = _schedule_available(season, previous_lockdown)
        fixtures = _fixtures(season, teams, moves, rng)
        gameweeks = _gameweeks(season, n_gws, fixtures)
        lockdown = gameweeks.set_index("gw")["lockdown_time"].to_dict()
        deadline = gameweeks.set_index("gw")["deadline_time"].to_dict()
        fixtures["gw_index"] = fixtures["gw"]
        states = _availability(players, n_gws, rng)
        order = rng.permutation(len(players))
        element_ids = {players[j]["player_key"]: i + 1 for i, j in enumerate(order)}

        # gameweek, schedule
        gw_frame = gameweeks.assign(
            deadline_source="bootstrap",
            event_time=gameweeks["deadline_time"],
            available_at=available,
        )
        gw_frame = gw_frame[
            [
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
        ]
        add("gameweek", gw_frame)
        add(
            "schedule",
            pd.DataFrame(
                {
                    "fixture_key": fixtures["fixture_key"],
                    "season": fixtures["season"],
                    "gw": pd.array(fixtures["gw"], dtype="Int64"),
                    "gw_index": pd.array(fixtures["gw_index"], dtype="Int64"),
                    "kickoff_time": fixtures["kickoff_time"],
                    "home_team_key": fixtures["home_team_key"],
                    "away_team_key": fixtures["away_team_key"],
                    "schedule_source": "final",
                    "event_time": fixtures["kickoff_time"],
                    "available_at": available,
                }
            ),
        )
        add(
            "team_rating",
            pd.DataFrame(
                {
                    "team_key": teams,
                    "season": season,
                    "fixture_key": pd.array([None] * n_clubs, dtype="Int64"),
                    "kickoff_time": pd.Series([pd.NaT] * n_clubs, dtype=UTC_US),
                    "opponent_team_key": pd.array([None] * n_clubs, dtype="Int64"),
                    "is_home": pd.array([None] * n_clubs, dtype="boolean"),
                    "rating_before": [elo[t] for t in teams],
                    "rating_after": [elo[t] for t in teams],
                    "expected_score": np.nan,
                    "event_time": available,
                    "available_at": available,
                }
            ),
        )

        # matches
        match_rows, team_rows, fixture_rows, rating_rows = [], [], [], []
        for _, fixture in fixtures.iterrows():
            gw = int(fixture["gw"])
            rows, sides, (hg, ag) = _match_rows(fixture, squads, states, strength, element_ids, rng)
            kickoff = fixture["kickoff_time"]
            common = {
                "season": season,
                "fixture_key": int(fixture["fixture_key"]),
                "gw": gw,
                "kickoff_time": kickoff,
                "available_at": lockdown[gw],
            }
            match_rows += [{**r, **common} for r in rows]
            team_rows += [{**r, **common} for r in sides]
            fixture_rows.append({**fixture.to_dict(), "home_goals": hg, "away_goals": ag})
            home, away = int(fixture["home_team_key"]), int(fixture["away_team_key"])
            expected = 1 / (1 + 10 ** ((elo[away] - elo[home] - 60) / 400))
            score = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
            for team, opponent, exp, res in (
                (home, away, expected, score),
                (away, home, 1 - expected, 1 - score),
            ):
                before = elo[team]
                elo[team] = before + 20 * (res - exp)
                rating_rows.append(
                    {
                        "team_key": team,
                        "season": season,
                        "fixture_key": int(fixture["fixture_key"]),
                        "kickoff_time": kickoff,
                        "opponent_team_key": opponent,
                        "is_home": team == home,
                        "rating_before": before,
                        "rating_after": elo[team],
                        "expected_score": exp,
                    }
                )
        add("player_match", _player_match_frame(match_rows))
        for r in match_rows:  # fixtures are generated in kickoff order
            kickoffs, points = history[r["player_key"]]
            kickoffs.append(r["kickoff_time"].value)
            points.append(r["total_points"])
        tm = pd.DataFrame(team_rows)
        add(
            "team_match",
            pd.DataFrame(
                {
                    "fixture_key": tm["fixture_key"].astype("int64"),
                    "team_key": tm["team_key"].astype("int64"),
                    "season": tm["season"].astype("int64"),
                    "is_home": tm["is_home"].astype("bool"),
                    "goals_for": tm["goals_for"].astype("int64"),
                    "goals_against": tm["goals_against"].astype("int64"),
                    "us_xg": pd.array(tm["xg"].round(6), dtype="Float64"),
                    "us_xga": pd.array(tm["xga"].round(6), dtype="Float64"),
                    "us_npxg": pd.array(tm["xg"].round(6), dtype="Float64"),
                    "us_npxga": pd.array(tm["xga"].round(6), dtype="Float64"),
                    "fd_xg": pd.array([None] * len(tm), dtype="Float64"),
                    "fd_xga": pd.array([None] * len(tm), dtype="Float64"),
                    "fpl_xg": pd.array(tm["fpl_xg"], dtype="Float64"),
                    "fpl_xga": pd.array(tm["fpl_xga"], dtype="Float64"),
                    "event_time": tm["kickoff_time"].astype(UTC_US),
                    "available_at": tm["available_at"].astype(UTC_US),
                }
            ),
        )
        fx = pd.DataFrame(fixture_rows)
        add(
            "fixture",
            pd.DataFrame(
                {
                    "fixture_key": fx["fixture_key"].astype("int64"),
                    "season": fx["season"].astype("int64"),
                    "fpl_fixture_id": fx["fpl_fixture_id"].astype("int64"),
                    "gw": pd.array(fx["gw"], dtype="Int64"),
                    "gw_index": pd.array(fx["gw_index"], dtype="Int64"),
                    "kickoff_time": fx["kickoff_time"].astype(UTC_US),
                    "home_team_key": fx["home_team_key"].astype("int64"),
                    "away_team_key": fx["away_team_key"].astype("int64"),
                    "home_goals": pd.array(fx["home_goals"], dtype="Int64"),
                    "away_goals": pd.array(fx["away_goals"], dtype="Int64"),
                    "finished": True,
                    "fd_date": pd.Series([k.date() for k in fx["kickoff_time"]], dtype=object),
                    "event_time": fx["kickoff_time"].astype(UTC_US),
                    "available_at": fx["gw"].map(lockdown).astype(UTC_US),
                }
            ),
        )
        rr = pd.DataFrame(rating_rows)
        add(
            "team_rating",
            pd.DataFrame(
                {
                    "team_key": rr["team_key"].astype("int64"),
                    "season": rr["season"].astype("int64"),
                    "fixture_key": pd.array(rr["fixture_key"], dtype="Int64"),
                    "kickoff_time": rr["kickoff_time"].astype(UTC_US),
                    "opponent_team_key": pd.array(rr["opponent_team_key"], dtype="Int64"),
                    "is_home": pd.array(rr["is_home"], dtype="boolean"),
                    "rating_before": rr["rating_before"].astype("float64"),
                    "rating_after": rr["rating_after"].astype("float64"),
                    "expected_score": rr["expected_score"].astype("float64"),
                    "event_time": rr["kickoff_time"].astype(UTC_US),
                    "available_at": (rr["kickoff_time"] + RATING_DELAY).astype(UTC_US),
                }
            ),
        )
        add(
            "gameweek_result",
            pd.DataFrame(
                {
                    "season": season,
                    "gw": gameweeks["gw"],
                    "average_entry_score": pd.array(
                        rng.integers(35, 75, len(gameweeks)), dtype="Int64"
                    ),
                    "event_time": gameweeks["lockdown_time"],
                    "available_at": gameweeks["lockdown_time"],
                }
            ),
        )

        # per player and GW: price, ownership, snapshots
        gw1 = deadline[1]
        add(
            "player_season",
            pd.DataFrame(
                {
                    "player_key": [p["player_key"] for p in players],
                    "season": season,
                    "element_id": [element_ids[p["player_key"]] for p in players],
                    "element_type": [p["element_type"] for p in players],
                    "first_name": pd.array(
                        [f"First{p['player_key']}" for p in players], dtype="str"
                    ),
                    "second_name": pd.array(
                        [f"Second{p['player_key']}" for p in players], dtype="str"
                    ),
                    "web_name": pd.array([f"P{p['player_key']}" for p in players], dtype="str"),
                    "event_time": gw1,
                    "available_at": gw1 - REGISTRATION_LEAD,
                }
            ),
        )
        n_fixtures = {(team, gw): 0 for team in teams for gw in range(1, n_gws + 1)}
        for _, fixture in fixtures.iterrows():
            for team in (fixture["home_team_key"], fixture["away_team_key"]):
                n_fixtures[(int(team), int(fixture["gw"]))] += 1
        season_points: dict[int, list[int]] = {p["player_key"]: [] for p in players}
        for r in match_rows:
            season_points[r["player_key"]].append(r["total_points"])
        gw_rows, own_rows, snap_rows = [], [], []
        prices = {p["player_key"]: p["price"] for p in players}
        # Snapshot-only draws use their own generator: `snapshots` changes nothing else.
        snap_rng = np.random.default_rng([seed, season, 1])
        noise = {p["player_key"]: float(snap_rng.uniform(0.85, 1.15)) for p in players}
        total_pop = sum(p["popularity"] for p in players)
        for gw in range(1, n_gws + 1):
            snap_at = deadline[gw] - SNAPSHOT_LEAD
            for player in players:
                key = player["player_key"]
                change = 0
                if gw > 1:
                    u = rng.random()
                    step = -1 if u < 0.08 else 1 if u > 0.92 else 0
                    new = min(max(prices[key] + step, PRICE_RANGE[0]), PRICE_RANGE[1])
                    change, prices[key] = new - prices[key], new
                share = 15 * player["popularity"] / total_pop * float(rng.uniform(0.8, 1.2))
                percent = round(min(100 * share, 95.0), 1)
                selected = int(percent / 100 * 9_000_000)
                fixtures_gw = n_fixtures[(player["team_key"], gw)]
                status, chance = states[key][gw - 1]
                if fixtures_gw:
                    gw_rows.append(
                        (
                            key,
                            season,
                            gw,
                            player["team_key"],
                            player["element_type"],
                            prices[key],
                            deadline[gw],
                        )
                    )
                    own_rows.append(
                        (
                            key,
                            season,
                            gw,
                            selected,
                            int(rng.integers(0, 50_000)),
                            int(rng.integers(0, 50_000)),
                            deadline[gw],
                        )
                    )
                if not snapshots:
                    continue
                points = season_points[key]
                rate = sum(points) / len(points) * noise[key] if points else 0.0
                factor = {"a": 1.0, "d": 0.5, "i": 0.0}[status]
                ep = round(rate * fixtures_gw * factor, 1)
                snap_rows.append(
                    {
                        "snapshot_at": snap_at,
                        "player_key": key,
                        "element_id": element_ids[key],
                        "team_key": player["team_key"],
                        "element_type": player["element_type"],
                        "now_cost": prices[key],
                        "status": status,
                        "chance": chance,
                        "news": "" if status == "a" else f"Knock - {chance}% chance of playing",
                        "news_added": pd.NaT if status == "a" else snap_at - pd.Timedelta(days=2),
                        "selected_by_percent": percent,
                        "ep_next": None if snap_rng.random() < 0.01 else ep,
                        "form": _form(history[key], snap_at),
                        "transfers_in_event": int(snap_rng.integers(0, 50_000)),
                        "transfers_out_event": int(snap_rng.integers(0, 50_000)),
                        "cost_change_event": change,
                    }
                )
        pg = pd.DataFrame(
            gw_rows,
            columns=[
                "player_key",
                "season",
                "gw",
                "team_key",
                "element_type",
                "value",
                "event_time",
            ],
        )
        add(
            "player_gw",
            pg.assign(
                event_time=pg["event_time"].astype(UTC_US),
                available_at=(pg["event_time"] - REGISTRATION_LEAD).astype(UTC_US),
            ).astype(
                {
                    c: "int64"
                    for c in ("player_key", "season", "gw", "team_key", "element_type", "value")
                }
            ),
        )
        po = pd.DataFrame(
            own_rows,
            columns=[
                "player_key",
                "season",
                "gw",
                "selected",
                "transfers_in",
                "transfers_out",
                "event_time",
            ],
        )
        add(
            "player_gw_ownership",
            po.assign(
                event_time=po["event_time"].astype(UTC_US),
                available_at=po["event_time"].astype(UTC_US),
            ).astype(
                {
                    c: "int64"
                    for c in (
                        "player_key",
                        "season",
                        "gw",
                        "selected",
                        "transfers_in",
                        "transfers_out",
                    )
                }
            ),
        )
        if snapshots:
            sn = pd.DataFrame(snap_rows)
            n = len(sn)
            add(
                "player_snapshot",
                pd.DataFrame(
                    {
                        "snapshot_at": sn["snapshot_at"],
                        "source": "fplcache",
                        "season": season,
                        "element_id": sn["element_id"],
                        "player_key": sn["player_key"],
                        "team_key": sn["team_key"],
                        "element_type": sn["element_type"],
                        "now_cost": sn["now_cost"],
                        "status": sn["status"],
                        "chance_of_playing_next_round": pd.array(sn["chance"], dtype="Int64"),
                        "chance_of_playing_this_round": pd.array(sn["chance"], dtype="Int64"),
                        "news": sn["news"],
                        "news_added": sn["news_added"],
                        "selected_by_percent": pd.array(sn["selected_by_percent"], dtype="Float64"),
                        "ep_next": pd.array(sn["ep_next"], dtype="Float64"),
                        "ep_this": pd.array([None] * n, dtype="Float64"),
                        "form": pd.array(sn["form"], dtype="Float64"),
                        "penalties_order": pd.array([None] * n, dtype="Int64"),
                        "corners_and_indirect_freekicks_order": pd.array([None] * n, dtype="Int64"),
                        "direct_freekicks_order": pd.array([None] * n, dtype="Int64"),
                        "team_join_date": pd.Series([None] * n, dtype="date32[day][pyarrow]"),
                        "transfers_in_event": sn["transfers_in_event"],
                        "transfers_out_event": sn["transfers_out_event"],
                        "cost_change_event": sn["cost_change_event"],
                        "event_time": sn["snapshot_at"],
                        "available_at": sn["snapshot_at"],
                    }
                ).astype(SNAPSHOT_DTYPES),
            )
        previous_lockdown = gameweeks["lockdown_time"].max()

    tables = {name: pd.concat(frames, ignore_index=True) for name, frames in out.items()}
    if not snapshots:
        tables["player_snapshot"] = _empty(SNAPSHOT_DTYPES)
    tables["odds_snapshot"] = _empty(ODDS_DTYPES)
    tables["fixture_snapshot"] = _empty(FIXTURE_SNAPSHOT_DTYPES)
    tables["team_dim"] = pd.DataFrame(
        {
            "team_key": pd.array(teams, dtype="int64"),
            "short_name": pd.array([f"T{t:02d}" for t in teams], dtype="str"),
            "fpl_names": pd.array([f"Team {t}" for t in teams], dtype="str"),
            "football_data_name": pd.array([f"Team {t}" for t in teams], dtype="str"),
            "understat_name": pd.array([f"Team {t}" for t in teams], dtype="str"),
            "odds_api_name": pd.array([f"Team {t} FC" for t in teams], dtype="str"),
            "in_fpl": True,
            "event_time": EPOCH,
            "available_at": EPOCH,
        }
    )
    keys = [p["player_key"] for p in players]
    tables["player_dim"] = pd.DataFrame(
        {
            "player_key": pd.array(keys, dtype="int64"),
            "first_name": pd.array([f"First{k}" for k in keys], dtype="str"),
            "second_name": pd.array([f"Second{k}" for k in keys], dtype="str"),
            "web_name": pd.array([f"P{k}" for k in keys], dtype="str"),
            "opta_code": pd.array([f"p{k}" for k in keys], dtype="str"),
            "understat_id": pd.array([None] * len(keys), dtype="Int64"),
            "event_time": EPOCH,
            "available_at": EPOCH,
        }
    )
    for name in ("gameweek", "schedule"):
        tables[name] = tables[name].astype(
            {"deadline_source": "str"} if name == "gameweek" else {"schedule_source": "str"}
        )
    return tables
