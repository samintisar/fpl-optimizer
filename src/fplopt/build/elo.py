"""`team_rating`: our own Elo ratings from football-data results (ClubElo is down; PLAN §3),
two rows per match (one per side) with the rating before and after it. Burn-in from the
first football-data season present (2005/06 after the backfill); `fixture_key` from 2016/17.

Model (World-Football-Elo style; constants below, to be tuned in Phase 5):
- `E_home = 1 / (1 + 10^(−(R_home + H − R_away) / 400))`, H = 60 (home advantage).
- Update `Δ = K · G · (S_home − E_home)`, K = 20; home gains Δ, away loses Δ (zero sum).
  S = 1 / 0.5 / 0. Goal-difference multiplier G = 1 (|d| ≤ 1), 1.5 (|d| = 2),
  (11 + |d|) / 8 (|d| ≥ 3).
- Every club of the first season starts at 1500. At each later season, clubs that were not
  in the previous season (promoted) start at the mean end-of-season rating of the clubs
  that dropped out of it (relegated); clubs that stay keep their rating. No mean reversion.
  A season's clubs = the teams in its football-data rows plus (2016/17+) its `fixture`
  rows, so the current season is complete before every club has played.
- Matches are processed in (kickoff, home_team_key) order: deterministic.

Inputs: football-data `Date, HomeTeam, AwayTeam, FTHG, FTAG` of every season (season from
the folder code), teams via `TeamResolver.football_data`. From 2016/17 each match is joined
to `fixture` on (season, home, away) for `fixture_key` and the FPL `kickoff_time` (its date
must equal `fixture.fd_date`, else the build fails); earlier matches kick off at `Date`
15:00 UK. `event_time` = kickoff; `available_at` = kickoff + 2 h (the result is in).

Validation: unique (team_key, kickoff_time); fixture_key present iff season ≥ 2016; each
club's ratings continuous within a season (rating_before = previous rating_after); every
match two rows whose rating changes sum to 0; at most 20 clubs per season; the mean
end-of-season rating of a season's clubs within 1500 ± 50 from 2016/17 (it is exactly 1500
while promoted and relegated counts match: changes are zero-sum and seeding preserves the
total). Measured on raw/ 2026-10-06: 16,060 rows (2005/06 – 2026/27 GW5, 8,030 matches,
20 clubs every season, < 1 s); league mean 1500.000 in every season (3 up / 3 down each
year); ratings range 1192–1877.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import pandera.pandas as pa

from fplopt.build.common import UK, UTC_US, BuildContext
from fplopt.build.fixtures import FIRST_SEASON, football_data_results
from fplopt.build.teams import TeamResolver

log = logging.getLogger(__name__)

INITIAL_RATING = 1500.0
HOME_ADVANTAGE = 60.0
K_FACTOR = 20.0
CLUBS_PER_SEASON = 20
MEAN_TOLERANCE = 50.0
PRE_FPL_KICKOFF_HOUR = 15  # UK local, football-data has dates only before 2016/17 here
RESULT_DELAY = pd.Timedelta(hours=2)
ZERO_SUM_TOLERANCE = 1e-9

COLUMNS = [
    "team_key",
    "season",
    "fixture_key",
    "kickoff_time",
    "opponent_team_key",
    "is_home",
    "rating_before",
    "rating_after",
    "expected_score",
    "event_time",
    "available_at",
]


class EloInputError(ValueError):
    """football-data results that cannot be placed on a fixture."""


def goal_multiplier(goal_difference: int) -> float:
    d = abs(int(goal_difference))
    if d <= 1:
        return 1.0
    if d == 2:
        return 1.5
    return (11 + d) / 8


def expected_home_score(home_rating: float, away_rating: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(home_rating + HOME_ADVANTAGE - away_rating) / 400.0))


def elo_ratings(
    matches: pd.DataFrame, season_clubs: dict[int, set[int]] | None = None
) -> pd.DataFrame:
    """Run the Elo model over `matches` (season, fixture_key, kickoff_time, home/away team
    keys, home/away goals) and return the two-rows-per-match table. `season_clubs` (season
    -> clubs) defaults to the teams of each season's matches."""
    ordered = matches.sort_values(["kickoff_time", "home_team_key"], kind="mergesort")
    if season_clubs is None:
        season_clubs = {}
    clubs = {int(s): set(c) for s, c in season_clubs.items()}
    for season, rows in ordered.groupby("season"):
        clubs.setdefault(int(season), set()).update(
            map(int, set(rows["home_team_key"]) | set(rows["away_team_key"]))
        )
    ratings: dict[int, float] = {}
    previous: set[int] | None = None
    out = []
    for season in sorted(int(s) for s in ordered["season"].unique()):
        current = clubs[season]
        if previous is None:
            ratings.update(dict.fromkeys(current, INITIAL_RATING))
        else:
            relegated = previous - current
            seed = (
                sum(ratings[c] for c in relegated) / len(relegated) if relegated else INITIAL_RATING
            )
            for club in sorted(current - previous):
                ratings[club] = seed
        rows = ordered[ordered["season"] == season]
        for m in rows.itertuples(index=False):
            home, away = int(m.home_team_key), int(m.away_team_key)
            before_h, before_a = ratings[home], ratings[away]
            e_home = expected_home_score(before_h, before_a)
            hg, ag = int(m.home_goals), int(m.away_goals)
            score = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
            delta = K_FACTOR * goal_multiplier(hg - ag) * (score - e_home)
            ratings[home], ratings[away] = before_h + delta, before_a - delta
            common = (season, m.fixture_key, m.kickoff_time)
            out.append((home, *common, away, True, before_h, ratings[home], e_home))
            out.append((away, *common, home, False, before_a, ratings[away], 1.0 - e_home))
        previous = current
    df = pd.DataFrame(out, columns=COLUMNS[:9])
    df = df.astype(
        {
            "team_key": "int64",
            "season": "int64",
            "fixture_key": "Int64",
            "kickoff_time": UTC_US,
            "opponent_team_key": "int64",
            "is_home": bool,
            "rating_before": "float64",
            "rating_after": "float64",
            "expected_score": "float64",
        }
    )
    df["event_time"] = df["kickoff_time"]
    df["available_at"] = (df["kickoff_time"] + RESULT_DELAY).astype(UTC_US)
    return df[COLUMNS]


# --- validation --------------------------------------------------------------------------


def _continuous(df: pd.DataFrame) -> bool:
    ordered = df.sort_values(["team_key", "season", "kickoff_time"], kind="mergesort")
    previous = ordered.groupby(["team_key", "season"])["rating_after"].shift()
    return bool((previous.isna() | (previous == ordered["rating_before"])).all())


def _zero_sum(df: pd.DataFrame) -> bool:
    low = df[["team_key", "opponent_team_key"]].min(axis=1)
    high = df[["team_key", "opponent_team_key"]].max(axis=1)
    change = (df["rating_after"] - df["rating_before"]).groupby([df["kickoff_time"], low, high])
    sizes, sums = change.size(), change.sum()
    return bool((sizes == 2).all() and (sums.abs() < ZERO_SUM_TOLERANCE).all())


def league_means(df: pd.DataFrame) -> pd.Series:
    """Mean end-of-season (or latest) rating of each season's clubs."""
    ordered = df.sort_values("kickoff_time", kind="mergesort")
    last = ordered.groupby(["season", "team_key"])["rating_after"].last()
    return last.groupby(level="season").mean()


def _means_ok(df: pd.DataFrame) -> bool:
    means = league_means(df)
    recent = means[means.index >= FIRST_SEASON]
    return bool(((recent - INITIAL_RATING).abs() <= MEAN_TOLERANCE).all())


SCHEMA = pa.DataFrameSchema(
    {
        "team_key": pa.Column("int64"),
        "season": pa.Column("int64"),
        "fixture_key": pa.Column("Int64", nullable=True),
        "kickoff_time": pa.Column(UTC_US),
        "opponent_team_key": pa.Column("int64"),
        "is_home": pa.Column(bool),
        "rating_before": pa.Column("float64"),
        "rating_after": pa.Column("float64"),
        "expected_score": pa.Column("float64", pa.Check.between(0, 1, include_min=False)),
        "event_time": pa.Column(UTC_US),
        "available_at": pa.Column(UTC_US),
    },
    checks=[
        pa.Check(
            lambda df: ~df.duplicated(["team_key", "kickoff_time"]),
            error="(team_key, kickoff_time) must be unique",
        ),
        pa.Check(
            lambda df: df["fixture_key"].notna() == (df["season"] >= FIRST_SEASON),
            error="fixture_key present iff season >= 2016",
        ),
        pa.Check(
            lambda df: (
                (df["event_time"] == df["kickoff_time"])
                & (df["available_at"] == df["kickoff_time"] + RESULT_DELAY)
                & (df["team_key"] != df["opponent_team_key"])
            ),
            error="row consistency",
        ),
        pa.Check(_continuous, error="ratings must be continuous within a season"),
        pa.Check(_zero_sum, error="two rows per match with rating changes summing to 0"),
        pa.Check(
            lambda df: bool((df.groupby("season")["team_key"].nunique() <= CLUBS_PER_SEASON).all()),
            error="at most 20 clubs per season",
        ),
        pa.Check(_means_ok, error="league mean rating within 1500 ± 50 from 2016/17"),
    ],
    strict=True,
    ordered=True,
)
SORT_BY = ("kickoff_time", "team_key")


# --- inputs ------------------------------------------------------------------------------


def pre_fpl_kickoffs(dates: pd.Series) -> pd.Series:
    """`Date` at 15:00 UK, in UTC (matches before 2016/17 have no FPL kickoff)."""
    local = pd.to_datetime(dates.map(date.isoformat)) + pd.Timedelta(hours=PRE_FPL_KICKOFF_HOUR)
    return local.dt.tz_localize(UK).dt.tz_convert("UTC").astype(UTC_US)


def elo_matches(results: pd.DataFrame, fixture: pd.DataFrame) -> pd.DataFrame:
    """Results (football_data_results) with fixture_key and kickoff_time."""
    keys = ["season", "home_team_key", "away_team_key"]
    fixtures = fixture[[*keys, "fixture_key", "kickoff_time", "fd_date"]].rename(
        columns={"fd_date": "fixture_fd_date"}
    )
    df = results.merge(fixtures, on=keys, how="left", validate="one_to_one")
    fpl_era = df["season"] >= FIRST_SEASON
    bad = df[fpl_era & (df["fixture_key"].isna() | (df["fd_date"] != df["fixture_fd_date"]))]
    if len(bad):
        raise EloInputError(
            f"{len(bad)} football-data result(s) from 2016/17 without a fixture of the same "
            "date (rebuild `fixture` after a football-data refresh):\n"
            + bad[[*keys, "fd_date", "fixture_fd_date"]].head(10).to_string()
        )
    pre = ~fpl_era
    kickoff = df["kickoff_time"].astype(UTC_US)
    if pre.any():
        kickoff = kickoff.where(~pre, pre_fpl_kickoffs(df.loc[pre, "fd_date"]))
    return pd.DataFrame(
        {
            "season": df["season"].astype("int64"),
            "fixture_key": df["fixture_key"].astype("Int64"),
            "kickoff_time": kickoff.astype(UTC_US),
            "home_team_key": df["home_team_key"].astype("int64"),
            "away_team_key": df["away_team_key"].astype("int64"),
            "home_goals": df["fd_home_goals"].astype("int64"),
            "away_goals": df["fd_away_goals"].astype("int64"),
        }
    )


def build_team_rating(ctx: BuildContext) -> pd.DataFrame:
    resolver = TeamResolver(ctx.table("team_dim"))
    fixture = ctx.table("fixture")
    matches = elo_matches(football_data_results(ctx, resolver, first_season=None), fixture)
    season_clubs: dict[int, set[int]] = {}
    for column in ("home_team_key", "away_team_key"):
        for source in (matches, fixture):
            for season, team in zip(source["season"], source[column], strict=True):
                season_clubs.setdefault(int(season), set()).add(int(team))
    df = elo_ratings(matches, season_clubs)
    means = league_means(df)
    log.info(
        "team_rating: league mean end-of-season rating %.3f to %.3f (2016/17+: %.3f to %.3f)",
        means.min(),
        means.max(),
        means[means.index >= FIRST_SEASON].min(),
        means[means.index >= FIRST_SEASON].max(),
    )
    return df
