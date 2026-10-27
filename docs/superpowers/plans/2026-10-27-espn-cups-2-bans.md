# ESPN cup data, PR 2 (bans) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve suspensions where they are really served: a red card in any domestic match (PL, FA Cup, EFL Cup) bans the club's next *domestic* matches, so a cup tie in between uses up the ban; a straight red is 1 certain + 2 likely matches, a second yellow 1; PL yellow-card bans stay PL-only.

**Architecture:** Spec `docs/superpowers/specs/2026-10-27-espn-cup-data-design.md` §2 (and §3 *Visibility*). PR 1 (merged first) gives `espn_match`, `espn_player_match`, `espn_map`, `espn_schedule_snapshot`. This PR adds one derived table, `club_match` (every non-league match of a PL club, with `available_at` = when it was known), a pure ban module `fplopt.features.bans` that turns card events and club timelines into a per player-fixture `ban_kind` (0 none, 1 likely, 2 certain), and wires it into `fplopt.features.history` (replacing the league-only ban code in `_lags`) and `fplopt.models.minutes` (one fitted residual per kind). Then the calibration table is refit and the change is validated like #47.

**Tech Stack:** pandas, numpy, pandera, pytest; `dev/calibrate_v1.py`; `fplopt models eval`, `fplopt check leakage`.

**Codes:** `BAN_NONE = 0`, `BAN_LIKELY = 1`, `BAN_CERTAIN = 2` (so overlapping bans combine with `max`).

---

## File map

- Modify `src/fplopt/build/espn.py`: `club_match` (built and written by `build_espn`).
- Modify `src/fplopt/build/tables.py`: register `club_match` (event, key `espn_event_id`).
- Create `src/fplopt/features/bans.py`: `ban_events`, `club_timelines`, `serve_bans`.
- Modify `src/fplopt/features/history.py`: drop the ban block of `_lags`; `training_frame` gets `ban_kind` (+ `banned` = kind > 0); `prediction_frame` gets `ban_kind` instead of `ban_remaining` / `club_fixture_order`.
- Modify `src/fplopt/models/minutes.py`: `MinutesFit` per-kind residuals; post-processing reads `ban_kind`.
- Modify `tests/synthetic_season.py`: empty ESPN tables + `club_match` in `synthetic_tables()`; revert the PR 1 exclusion in `tests/test_models_baseline.py` if the synthetic tables now cover them.
- Tests: `tests/test_build_espn.py`, new `tests/test_features_bans.py`, `tests/test_features_history.py`, `tests/test_models_minutes.py`.
- Regenerated: `src/fplopt/models/calibration_table.py` (Task 6).
- Docs: `history.py` / `minutes.py` docstrings, PLAN §6.3 *Deterministic suspensions*.

---

### Task 1: `club_match` table

Rows: every match in `espn_match` with `league != "eng.1"`, plus every event in `espn_schedule_snapshot` not in `espn_match` (scheduled, not played yet; kickoff and sides from its newest snapshot). Columns: `espn_event_id, league, season, kickoff_time, home_team_key, away_team_key, played, event_time, available_at`.

`available_at` = when the match was known = min(first snapshot listing it, rule time); the rule time per PL side = the kickoff of that club's previous match in the same league and season (the draw follows the previous round), or `kickoff − FIRST_ROUND_NOTICE` (14 days) when it has none; the row's rule time is the latest over its PL sides; never after `kickoff − 1 µs`. `event_time` = kickoff.

**Files:** `src/fplopt/build/espn.py`, `src/fplopt/build/tables.py`, `tests/test_build_espn.py`, `tests/test_build_tables.py`

- [ ] **Step 1: Failing tests** (append to `tests/test_build_espn.py`)

```python
from fplopt.build.espn import FIRST_ROUND_NOTICE, club_match


def cup(event_id, kickoff, home, away, league="eng.league_cup"):
    return {
        "espn_event_id": event_id,
        "league": league,
        "season": 2026,
        "kickoff_time": pd.Timestamp(kickoff),
        "home_team_key": home,
        "away_team_key": away,
    }


def test_club_match_known_after_the_previous_round_or_two_weeks_before_the_first():
    played = pd.DataFrame(
        [
            cup(1, "2026-08-26T18:45Z", 8, None),  # Chelsea's round 2
            cup(2, "2026-09-23T18:45Z", 8, 14),  # round 3 v Liverpool
            cup(3, "2026-09-17T19:00Z", 8, None, league="uefa.champions"),
            cup(9, "2026-09-16T19:00Z", 1, None, league="eng.1"),  # league: not a club match
        ]
    )
    out = club_match(played, pd.DataFrame(columns=["snapshot_at", *played.columns]))
    out = out.set_index("espn_event_id")
    assert set(out.index) == {1, 2, 3}
    assert out.loc[1, "available_at"] == pd.Timestamp("2026-08-26T18:45Z") - FIRST_ROUND_NOTICE
    # Round 3: known once both clubs' previous league-cup match kicked off (Liverpool: none).
    assert out.loc[2, "available_at"] == pd.Timestamp("2026-08-26T18:45Z")
    assert out.loc[3, "available_at"] == pd.Timestamp("2026-09-17T19:00Z") - FIRST_ROUND_NOTICE
    assert out["played"].all() and (out["event_time"] == out["kickoff_time"]).all()


def test_club_match_adds_scheduled_matches_known_from_their_first_snapshot():
    played = pd.DataFrame([cup(1, "2026-09-23T18:45Z", 8, None)])
    snaps = pd.DataFrame(
        [
            {
                "snapshot_at": pd.Timestamp("2026-10-01T02:00Z"),
                **cup(2, "2026-10-28T19:45Z", 14, 8),
            },
            {
                "snapshot_at": pd.Timestamp("2026-10-20T02:00Z"),
                **cup(2, "2026-10-28T20:00Z", 14, 8),
            },
        ]
    )
    out = club_match(played, snaps).set_index("espn_event_id")
    assert not out.loc[2, "played"]
    assert out.loc[2, "kickoff_time"] == pd.Timestamp("2026-10-28T20:00Z")  # newest snapshot
    # Rule: Chelsea's previous league-cup match (event 1) kicked off 2026-09-23 < first snapshot.
    assert out.loc[2, "available_at"] == pd.Timestamp("2026-09-23T18:45Z")
```

- [ ] **Step 2: Run** `uv run pytest tests/test_build_espn.py -k club_match -v` → FAIL (ImportError).

- [ ] **Step 3: Implement** in `src/fplopt/build/espn.py`:

```python
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
    done = played.loc[played["league"] != "eng.1", cols].assign(played=True)
    snaps = schedule[schedule["league"] != "eng.1"].sort_values("snapshot_at", kind="mergesort")
    first_seen = snaps.groupby("espn_event_id")["snapshot_at"].min()
    pending = snaps.drop_duplicates("espn_event_id", keep="last")
    pending = pending.loc[~pending["espn_event_id"].isin(done["espn_event_id"]), cols]
    out = pd.concat([done, pending.assign(played=False)], ignore_index=True)
    out = out.sort_values(["kickoff_time", "espn_event_id"], kind="mergesort").reset_index(
        drop=True
    )
    # Rule time per PL side: the club's previous match in the league and season.
    sides = (
        pd.concat(
            [
                out[["espn_event_id", "league", "season", "kickoff_time", c]].rename(
                    columns={c: "team_key"}
                )
                for c in ("home_team_key", "away_team_key")
            ]
        )
        .dropna(subset=["team_key"])
        .sort_values(["kickoff_time", "espn_event_id"], kind="mergesort")
    )
    previous = sides.groupby(["team_key", "league", "season"])["kickoff_time"].shift()
    rule = previous.fillna(sides["kickoff_time"] - FIRST_ROUND_NOTICE)
    rule = rule.groupby(sides["espn_event_id"]).max()
    known = out["espn_event_id"].map(rule)
    seen = out["espn_event_id"].map(first_seen)
    known = known.where(seen.isna() | (known <= seen), seen)
    latest = out["kickoff_time"] - pd.Timedelta(microseconds=1)
    out["available_at"] = known.where(known < latest, latest)
    out["event_time"] = out["kickoff_time"]
    return _frame(out.to_dict("records"), CLUB_MATCH_SCHEMA)
```

Add `"played"` to `_BOOLS`. In `build_espn`, after the checks: `write_table(club_match(tables["espn_match"], tables["espn_schedule_snapshot"]), "club_match", CLUB_MATCH_SCHEMA, ctx.data_dir, CLUB_MATCH_SORT_BY)` (and `ctx.forget("club_match")`). Register `TableSpec("club_match", "event", ("espn_event_id",))` in `TABLES`. Add a *club_match* paragraph to the module docstring (the rule above, and that a rescheduled kickoff takes the newest value: a small, accepted look-ahead on the date only).

- [ ] **Step 4: Run** `uv run pytest tests/test_build_espn.py tests/test_build_tables.py -v` → PASS (the full synthetic build writes an empty `club_match`).
- [ ] **Step 5: Commit** `feat(build): club_match, every cup and European match of a PL club with when it was known`

---

### Task 2: the ban module (`fplopt.features.bans`)

Pure functions over frames (no view), so they are tested directly.

- `ban_events(league_rows, red_kinds, cup_reds)` → one row per ban: `player_key, team_key, time, scope ('domestic'|'league'), n_certain, n_likely`.
  - `league_rows`: `player_key, season, gw_index, fixture_key, team_key, kickoff_time, yellow_cards, red_cards` (history's match rows).
  - PL yellow thresholds (`BAN_RULES`, unchanged: 5 by gw_index 19 → 1, 10 by 32 → 2, 15 → 3): scope `league`, `n_certain` = matches, `n_likely` 0.
  - `red_kinds`: `player_key, fixture_key, second_yellow` from ESPN `eng.1` rows (mapped). A league red with `second_yellow` True → (1, 0); False → (1, `STRAIGHT_RED_EXTRA` = 2); no ESPN row → (1, 0) (today's rule).
  - `cup_reds`: `player_key, team_key, kickoff_time, second_yellow` (ESPN `eng.fa` / `eng.league_cup` rows with `red`, mapped) → scope `domestic`, lengths as above.
  - League reds: scope `domestic`. `time` = the match's kickoff.
- `club_timelines(league_fixtures, cup_matches)` → `team_key, kickoff_time, fixture_key (Int64; null for a cup match), domestic_cup (bool)` sorted by (team_key, kickoff_time): league fixtures (`team_key, fixture_key, kickoff_time`) and the domestic cup matches (`club_match` rows with `league` in `eng.fa`/`eng.league_cup`, one row per PL side).
- `serve_bans(events, timelines)` → `player_key, fixture_key, ban_kind` (league fixtures only, kind > 0): for each event, the club's timeline entries with `kickoff_time > time` (scope `league`: league fixtures only), in order; the first `n_certain` → `BAN_CERTAIN`, the next `n_likely` → `BAN_LIKELY`; a cup entry consumes its position without a row; overlapping bans: max.

**Files:** create `src/fplopt/features/bans.py`, `tests/test_features_bans.py`

- [ ] **Step 1: Failing tests** (`tests/test_features_bans.py`)

```python
import pandas as pd

from fplopt.features.bans import (
    BAN_CERTAIN,
    BAN_LIKELY,
    ban_events,
    club_timelines,
    serve_bans,
)

T = pd.Timestamp


def league(rows):
    cols = [
        "player_key",
        "season",
        "gw_index",
        "fixture_key",
        "team_key",
        "kickoff_time",
        "yellow_cards",
        "red_cards",
    ]
    return pd.DataFrame(rows, columns=cols)


def fixtures(team_key, start, n, first_key=100):
    return pd.DataFrame(
        {
            "team_key": team_key,
            "fixture_key": range(first_key, first_key + n),
            "kickoff_time": [T(start) + pd.Timedelta(weeks=i) for i in range(n)],
        }
    )


NO_REDS = pd.DataFrame(columns=["player_key", "fixture_key", "second_yellow"])
NO_CUP = pd.DataFrame(columns=["player_key", "team_key", "kickoff_time", "second_yellow"])
NO_CUP_MATCHES = pd.DataFrame(columns=["team_key", "kickoff_time"])


def kinds(served):
    return dict(zip(served["fixture_key"], served["ban_kind"], strict=True))


def test_palmer_second_yellow_is_served_in_the_cup_tie():
    # Chelsea (8): PL fixture 100 on 24 Oct (red, second yellow), EFL Cup 28 Oct, PL 101 31 Oct.
    rows = league([[7, 2026, 8, 100, 8, T("2026-10-24T16:30Z"), 1, 1]])
    reds = pd.DataFrame({"player_key": [7], "fixture_key": [100], "second_yellow": [True]})
    events = ban_events(rows, reds, NO_CUP)
    lines = club_timelines(
        fixtures(8, "2026-10-24T16:30Z", 3),
        pd.DataFrame({"team_key": [8], "kickoff_time": [T("2026-10-28T19:45Z")]}),
    )
    assert kinds(serve_bans(events, lines)) == {}


def test_without_the_cup_tie_the_next_pl_fixture_is_banned():
    rows = league([[7, 2026, 8, 100, 8, T("2026-10-24T16:30Z"), 1, 1]])
    reds = pd.DataFrame({"player_key": [7], "fixture_key": [100], "second_yellow": [True]})
    lines = club_timelines(fixtures(8, "2026-10-24T16:30Z", 3), NO_CUP_MATCHES)
    assert kinds(serve_bans(ban_events(rows, reds, NO_CUP), lines)) == {101: BAN_CERTAIN}


def test_a_straight_red_is_one_certain_and_two_likely():
    rows = league([[7, 2026, 8, 100, 8, T("2026-10-24T16:30Z"), 0, 1]])
    reds = pd.DataFrame({"player_key": [7], "fixture_key": [100], "second_yellow": [False]})
    lines = club_timelines(fixtures(8, "2026-10-24T16:30Z", 5), NO_CUP_MATCHES)
    served = kinds(serve_bans(ban_events(rows, reds, NO_CUP), lines))
    assert served == {101: BAN_CERTAIN, 102: BAN_LIKELY, 103: BAN_LIKELY}


def test_a_red_without_espn_data_is_one_match():
    rows = league([[7, 2026, 8, 100, 8, T("2026-10-24T16:30Z"), 0, 1]])
    lines = club_timelines(fixtures(8, "2026-10-24T16:30Z", 3), NO_CUP_MATCHES)
    assert kinds(serve_bans(ban_events(rows, NO_REDS, NO_CUP), lines)) == {101: BAN_CERTAIN}


def test_a_cup_red_bans_the_next_pl_fixture():
    rows = league([])
    cup = pd.DataFrame(
        {
            "player_key": [7],
            "team_key": [8],
            "kickoff_time": [T("2026-10-28T19:45Z")],
            "second_yellow": [True],
        }
    )
    lines = club_timelines(
        fixtures(8, "2026-10-24T16:30Z", 3),
        pd.DataFrame({"team_key": [8], "kickoff_time": [T("2026-10-28T19:45Z")]}),
    )
    assert kinds(serve_bans(ban_events(rows, NO_REDS, cup), lines)) == {101: BAN_CERTAIN}


def test_pl_yellow_bans_ignore_cup_ties():
    # Fifth yellow in gw_index 10 -> one PL match, served in PL fixture 101 despite the cup tie.
    rows = league(
        [
            [
                7,
                2026,
                gw,
                100 + gw - 6,
                8,
                T("2026-10-03T14:00Z") + pd.Timedelta(weeks=gw - 6),
                1,
                0,
            ]
            for gw in range(6, 11)
        ]
    )
    lines = club_timelines(
        fixtures(8, "2026-10-03T14:00Z", 7),
        pd.DataFrame({"team_key": [8], "kickoff_time": [T("2026-11-03T19:45Z")]}),
    )
    assert kinds(serve_bans(ban_events(rows, NO_REDS, NO_CUP), lines)) == {105: BAN_CERTAIN}


def test_overlapping_bans_keep_the_stronger_kind():
    rows = league(
        [
            [7, 2026, 8, 100, 8, T("2026-10-24T16:30Z"), 0, 1],
            [7, 2026, 9, 101, 8, T("2026-10-31T16:30Z"), 0, 1],
        ]
    )
    reds = pd.DataFrame(
        {"player_key": [7, 7], "fixture_key": [100, 101], "second_yellow": [False, True]}
    )
    lines = club_timelines(fixtures(8, "2026-10-24T16:30Z", 5), NO_CUP_MATCHES)
    served = kinds(serve_bans(ban_events(rows, reds, NO_CUP), lines))
    assert served == {101: BAN_CERTAIN, 102: BAN_CERTAIN, 103: BAN_LIKELY}
```

- [ ] **Step 2: Run** `uv run pytest tests/test_features_bans.py -v` → FAIL.

- [ ] **Step 3: Implement** `src/fplopt/features/bans.py` (pure: pandas/numpy only; module-level values immutable — `tests/test_features_architecture.py` scans it):

```python
"""Suspensions served where they really are (spec 2026-10-27 §2): a red card in any domestic
match (PL, FA Cup, EFL Cup) bans the club's next domestic matches in date order, so a cup tie
in between serves part of the ban; a PL yellow-card ban (`BAN_RULES`) is served in PL matches
only. A second yellow is one match; a straight red is one certain match and
`STRAIGHT_RED_EXTRA` likely ones (three matches, but one for denying an obvious goal-scoring
opportunity, which ESPN does not distinguish); a red without ESPN data counts as one match.
Codes per league fixture: BAN_NONE, BAN_LIKELY, BAN_CERTAIN (overlaps take the max)."""

from __future__ import annotations

import numpy as np
import pandas as pd

BAN_NONE, BAN_LIKELY, BAN_CERTAIN = 0, 1, 2
# (yellow cards this season, last gw_index at which reaching them bans, PL matches banned).
BAN_RULES = ((5, 19, 1), (10, 32, 2), (15, 99, 3))
STRAIGHT_RED_EXTRA = 2
EVENT_COLUMNS = ("player_key", "team_key", "time", "scope", "n_certain", "n_likely")


def _red_lengths(second_yellow: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    straight = second_yellow.eq(False).to_numpy()  # NaN (no ESPN row) -> one match
    return np.ones(len(second_yellow), dtype=int), np.where(straight, STRAIGHT_RED_EXTRA, 0)


def ban_events(
    league_rows: pd.DataFrame, red_kinds: pd.DataFrame, cup_reds: pd.DataFrame
) -> pd.DataFrame:
    rows = league_rows.sort_values(
        ["player_key", "season", "gw_index", "kickoff_time", "fixture_key"], kind="mergesort"
    )
    yellow = rows["yellow_cards"].astype("float64")
    total = yellow.groupby([rows["player_key"], rows["season"]]).cumsum()
    before = total - yellow
    length = np.zeros(len(rows), dtype=int)
    for cards, last_gw, matches in BAN_RULES:
        hit = (before < cards) & (total >= cards) & (rows["gw_index"] <= last_gw)
        length = length + matches * hit.to_numpy()
    base = rows[["player_key", "team_key"]].assign(time=rows["kickoff_time"])
    yellows = base[length > 0].assign(scope="league", n_certain=length[length > 0], n_likely=0)
    reds = rows[rows["red_cards"] > 0].merge(
        red_kinds, on=["player_key", "fixture_key"], how="left"
    )
    certain, likely = _red_lengths(reds["second_yellow"])
    league_reds = reds[["player_key", "team_key"]].assign(
        time=reds["kickoff_time"], scope="domestic", n_certain=certain, n_likely=likely
    )
    certain, likely = _red_lengths(cup_reds["second_yellow"].astype("boolean"))
    cup = cup_reds[["player_key", "team_key"]].assign(
        time=cup_reds["kickoff_time"], scope="domestic", n_certain=certain, n_likely=likely
    )
    out = pd.concat([yellows, league_reds, cup], ignore_index=True)
    return out[list(EVENT_COLUMNS)].astype({"player_key": "int64", "team_key": "int64"})


def club_timelines(league_fixtures: pd.DataFrame, cup_matches: pd.DataFrame) -> pd.DataFrame:
    league = league_fixtures[["team_key", "fixture_key", "kickoff_time"]].assign(domestic_cup=False)
    cups = cup_matches[["team_key", "kickoff_time"]].assign(fixture_key=pd.NA, domestic_cup=True)
    out = pd.concat([league, cups], ignore_index=True)
    out = out.astype({"team_key": "int64", "fixture_key": "Int64"})
    out["kickoff_time"] = pd.to_datetime(out["kickoff_time"], utc=True)
    return out.sort_values(["team_key", "kickoff_time"], kind="mergesort").reset_index(drop=True)


def serve_bans(events: pd.DataFrame, timelines: pd.DataFrame) -> pd.DataFrame:
    """player_key, fixture_key, ban_kind for every banned league fixture (see the module
    docstring). A ban is served by the club of the event (a mid-ban transfer is ignored)."""
    kinds: dict[tuple[int, int], int] = {}
    by_club = {int(k): g for k, g in timelines.groupby("team_key", sort=False)}
    for ev in events.itertuples(index=False):
        line = by_club.get(int(ev.team_key))
        if line is None:
            continue
        if ev.scope == "league":
            line = line[~line["domestic_cup"]]
        times = line["kickoff_time"].to_numpy()
        start = int(
            np.searchsorted(
                times, np.datetime64(pd.Timestamp(ev.time).tz_convert(None)), side="right"
            )
        )
        ahead = line.iloc[start : start + ev.n_certain + ev.n_likely]
        for position, fixture_key in enumerate(ahead["fixture_key"]):
            if pd.isna(fixture_key):
                continue
            kind = BAN_CERTAIN if position < ev.n_certain else BAN_LIKELY
            key = (int(ev.player_key), int(fixture_key))
            kinds[key] = max(kinds.get(key, BAN_NONE), kind)
    out = pd.DataFrame(
        [(p, f, k) for (p, f), k in kinds.items()],
        columns=["player_key", "fixture_key", "ban_kind"],
    )
    return out.astype({"player_key": "int64", "fixture_key": "int64", "ban_kind": "float64"})
```

The `np.searchsorted` call must compare like with like: convert `times` and the event time the same way (both tz-aware UTC → `datetime64[ns]` via `.dt.tz_convert(None)`); write a tiny helper `_ns(series_or_ts)` if cleaner. Module-level `kinds`/`by_club` are function locals (allowed). Remove `BAN_RULES` and `RED_BAN`/`MAX_BAN` from `history.py` in Task 3 (history re-exports `BAN_RULES` from `bans` if anything imports it from there: grep).

- [ ] **Step 4: Run** `uv run pytest tests/test_features_bans.py tests/test_features_architecture.py -v` → PASS.
- [ ] **Step 5: Commit** `feat(features): bans served in domestic cup matches, straight reds likely 3`

---

### Task 3: history frames carry `ban_kind`

In `src/fplopt/features/history.py`:

- Remove the "Bans" block of `_lags` (the `length`/`remaining` loop, `ban_remaining`, `banned`), `RED_BAN`, `MAX_BAN`, and `BAN_RULES` (now `fplopt.features.bans.BAN_RULES`; keep a re-export only if something imports it from history).
- New helper `_bans(view, matches, league_fixtures)` → `player_key, fixture_key, ban_kind`:
  - `red_kinds`: `view.table("espn_player_match", columns=["espn_athlete_id", "espn_event_id", "league", "red", "second_yellow", "team_key", "event_time"])`, `view.table("espn_match", columns=["espn_event_id", "fixture_key"])`, `view.lookup("espn_map", ids)` (→ `player_key`); `eng.1` rows with `red` joined to `fixture_key` → `player_key, fixture_key, second_yellow`.
  - `cup_reds`: rows with `league` in `fplopt.adapters.espn.DOMESTIC` minus `eng.1` and `red`, mapped → `player_key, team_key, kickoff_time (= event_time), second_yellow`. (Import `DOMESTIC` only if `tests/test_features_architecture.py` allows `fplopt.adapters.espn`; otherwise define `DOMESTIC_CUPS = ("eng.fa", "eng.league_cup")` in `bans.py`.)
  - `cup_matches`: `view.table("club_match", columns=["league", "kickoff_time", "home_team_key", "away_team_key"])`, domestic cups only, one row per non-null side → `team_key, kickoff_time`.
  - `ban_events(...)` from `matches` (the visible league rows) → `serve_bans(events, club_timelines(league_fixtures, cup_matches))`.
- `training_frame`: `league_fixtures` = the visible league fixtures per side (`_team_fixtures(matches)` gives team_key, fixture_key, kickoff_time — check its columns); merge `_bans(...)` onto the rows by (player_key, fixture_key) → `ban_kind` (fill 0); `banned` = `(ban_kind > 0)` as float64. Training rows only count bans from events before their own kickoff — true by construction (`kickoff_time > time`).
- `prediction_frame`: `league_fixtures` = visible played fixtures plus the horizon fixtures (`_horizon_fixtures(view, season)`: team_key, fixture_key, kickoff_time); merge `_bans(...)` onto the output by (player_key, fixture_key) → `ban_kind` (fill 0). `PREDICTION_COLUMNS`: replace `"ban_remaining", "club_fixture_order"` by `"ban_kind"`; drop `club_fixture_order` from `_horizon_fixtures` if nothing else uses it (grep).
- Docstring: rewrite **Suspensions** ("`ban_kind` per league fixture from `fplopt.features.bans`: …; training rows carry `ban_kind` and `banned`, prediction rows `ban_kind`").
- Synthetic tables (`tests/synthetic_season.py` `synthetic_tables`): add empty `espn_match`, `espn_player_match`, `espn_map`, `espn_schedule_snapshot`, `club_match` frames with the PR 1 / Task 1 schemas' columns and dtypes (helper: build them from the pandera schemas, e.g. `_frame([], SCHEMA)` exported from `fplopt.build.espn`), so every view-reading test still runs; then drop the ESPN/club_match exclusion added to `tests/test_models_baseline.py` in PR 1 if the synthetic tables now cover them.

**Files:** `src/fplopt/features/history.py`, `tests/test_features_history.py`, `tests/synthetic_season.py`, `tests/test_models_baseline.py`

- [ ] **Step 1: Failing tests** in `tests/test_features_history.py`: adapt the existing ban tests (grep `ban_remaining`, `banned`) to `ban_kind`, and add one view-level test: a synthetic league where a player is sent off (second yellow, ESPN row mapped) in GW k and his club has a domestic cup `club_match` between GW k and k+1 → his GW k+1 prediction row has `ban_kind == 0`; without the cup match → 2. Build the view the way the existing history tests do (look at their fixtures/helpers first).
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** as above.
- [ ] **Step 4: Run** `uv run pytest tests/test_features_history.py tests/test_features_bans.py tests/test_features_architecture.py tests/test_features_store.py -q` → PASS.
- [ ] **Step 5: Commit** `feat(features): history frames take bans from fplopt.features.bans (ban_kind)`

---

### Task 4: minutes model, one residual per kind

In `src/fplopt/models/minutes.py`:

- `MinutesFit`: keep `ban_start` / `ban_sub` for **certain** bans; add `ban_start_likely: float = 0.0`, `ban_sub_likely: float = 0.0`.
- `fit_minutes`: training rows with `banned != 0` are excluded as now; with `ban_residual`, `ban_start`/`ban_sub` from rows with `ban_kind == BAN_CERTAIN`, the `_likely` pair from `ban_kind == BAN_LIKELY` (same `_weighted_mean` formulas; default 0.0 / 0.0 when there are none).
- Post-processing (around line 319): `kind = frame["ban_kind"]`; `certain = kind == BAN_CERTAIN`, `likely = kind == BAN_LIKELY`; `p_start = where(certain, ban_start, where(likely, ban_start_likely, p_start))`, same for `csub`; `banned = certain | likely` (fixed, carried to the output as now).
- Docstring item 2 (*Suspensions*): rewrite for `ban_kind` and the two residuals.
- `availability.py` needs no change (`banned` keeps its meaning) — confirm by grep.

**Files:** `src/fplopt/models/minutes.py`, `tests/test_models_minutes.py`

- [ ] **Step 1: Failing tests**: adapt the existing ban-residual tests to `ban_kind`; add one: training rows with `ban_kind` 2 (start rate 0.1) and 1 (start rate 0.6) give `ban_start ≈ 0.1`, `ban_start_likely ≈ 0.6`, and a prediction row with `ban_kind == 1` gets `p_start == ban_start_likely`.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest tests/test_models_minutes.py tests/test_models_availability.py tests/test_models_assemble.py tests/test_models_fitted.py -q` → PASS; then the full suite `uv run pytest -q` (~20 min) and `uv run ruff check . && uv run ruff format --check .`.
- [ ] **Step 5: Commit** `feat(models): a fitted start rate per ban kind (certain, likely)`

---

### Task 5: real-data check of the rules

On the server checkout of the branch (`~/fpl-espn`, `raw` → the live archive; `data/` built there with `fplopt build all`, ~2 min):

- [ ] Count per season (develop): league fixtures with `ban_kind` 2 and 1 in `training_frame` at each season's last deadline; their start rates (should be ≈ 0.1 for certain; likely between); how many bans main's rule (`RED_BAN` 1, league only) puts on a different fixture (served in a cup instead). Script in the scratchpad, read-only on `data/`.
- [ ] Palmer: `prediction_frame` at the GW9 2026/27 deadline (2026-10-31 11:00 UTC) on the server's data: Palmer's (element 154) GW9 row has `ban_kind == 0` (the EFL Cup at Liverpool on 2026-10-28 is in `club_match`, known from its schedule snapshot).
- [ ] Record the numbers in the `bans.py` docstring ("Measured on raw/ <date>: …") and PLAN §6.3; commit.

### Task 6: calibration refit and validation

- [ ] `uv run python dev/calibrate_v1.py --data-dir data walk --seasons 2016-2024 --jobs 7 --out results/espn-bans-walk` then `uv run python dev/calibrate_v1.py fit --walk results/espn-bans-walk --write` (check `dev/calibrate_v1.py --help` for the exact flags; ~1–2 h on the server, run via Bash `run_in_background`). Commit the regenerated `src/fplopt/models/calibration_table.py` with the develop MSE it prints.
- [ ] `uv run fplopt check leakage --deadlines 12` → passes (~25 min).
- [ ] `uv run fplopt models eval --models v1 --seasons 2016-2022 --out results/espn-bans-develop` and the same on `main` (a second checkout, e.g. `~/fpl-live`, same `data/` copy) → paired comparison (reuse the session's `promote_compare.py` approach: MSE h0 / h1–5 per player-GW with GW-clustered SE, captain/XI regret). Then validate (`--seasons 2023-2024`) and live (`--seasons 2026`, monitor only).
- [ ] Gate (as #47): validate MSE h0 and h1–5 not worse (paired diff ≤ 0 or within 1 SE), captain regret not significantly worse. Record the numbers in the PR body and append rows to `results/experiments.csv` as the commands do.

### Task 7: PR, merge, deploy

- [ ] PR "Bans served in cup matches (ESPN)": summary, Task 5 numbers, Task 6 results, test plan. After the user approves: merge, deploy (`cd ~/fpl-optimizer && git pull -q --ff-only && ~/.local/bin/uv sync -q --locked --all-extras && systemctl --user restart fplopt-bot.service`), then `fplopt live prepare` (rebuild data and the GW9 context) and check Palmer's GW9 xP in the new context.
