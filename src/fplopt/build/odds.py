"""`odds_snapshot`: bookmaker prices per fixture in long format (one row per fixture, source,
bookmaker, market, outcome, line, closing flag and snapshot time), 2016/17 →.

football-data (newest file per season; joined to `fixture` on (season, home, away), the
football-data date must equal `fixture.fd_date`, else the build fails). An **explicit
allowlist** of columns, never a regex (PLAN §3 "Odds columns"); era A = 2016/17–2018/19,
era B = 2019/20+:

- `avg` (market average). Era A: `BbAvH/D/A`, `BbAv>2.5`/`BbAv<2.5`, `BbAHh` +
  `BbAvAHH/AHA`. Era B: `AvgH/D/A`, `Avg>2.5`/`Avg<2.5`, `AHh` + `AvgAHH/AHA`; closing
  `AvgCH/CD/CA`, `AvgC>2.5`/`AvgC<2.5`, `AHCh` + `AvgCAHH/CAHA`.
- `pinnacle`. Era A: `PSH/D/A`; closing `PSCH/CD/CA`. Era B: `PSH/D/A`, `P>2.5`/`P<2.5`,
  `AHh` + `PAHH/PAHA`; closing `PSCH/CD/CA`, `PC>2.5`/`PC<2.5`, `AHCh` + `PCAHH/PCAHA`.
- `betfair_ex` (Betfair Exchange, files from 2024/25). Era B: `BFEH/D/A`,
  `BFE>2.5`/`BFE<2.5`, `AHh` + `BFEAHH/AHA`; closing `BFECH/CD/CA`, `BFEC>2.5`/`BFEC<2.5`,
  `AHCh` + `BFECAHH/CAHA`.

Closing AH prices pair with the closing line `AHCh`, never `AHh` (they differ in 30–39% of
matches). Traps kept out by the allowlist: `CLH/CLD/CLA` (Coral, not closing),
`BFH/BFD/BFA` (Betfair sportsbook), `BFDH/BFDD/BFDA` (Betfred), `BFCH` (Betfair sportsbook
closing). Null cells are skipped (Pinnacle stops in Feb 2026; the 2026/27 file has none).
`line` is the home handicap for both AH outcomes and 2.5 for totals.

Timing (`snapshot_at = available_at`): football-data collects pre-match odds on Friday
afternoon (weekend) and Tuesday afternoon (midweek), so a pre-match price is available from
the collection time: UK-local kickoff weekday Tue/Wed/Thu -> that week's Tuesday 15:00 UK,
Fri/Sat/Sun/Mon -> the Friday on or before, 15:00 UK; capped at kickoff − 1h. Closing
prices are only known at kickoff: `available_at = kickoff_time`. `event_time` = the fixture's
`event_time` (its FPL kickoff).

The Odds API (`raw/odds/soccer_epl/*.json.gz`, 2026/27 →): events -> fixtures by (season =
July-cutoff season of `commence_time`, home, away) via `TeamResolver.odds_api`; markets
`h2h` (outcomes by team name / "Draw") and `totals` with `point == 2.5` (3.5 dropped);
`h2h_lay` (exchange lay prices, not requested) dropped. `bookmaker` = the API key,
`snapshot_at = available_at` = file timestamp, never closing. Skipped and reported: events
that had already kicked off at the snapshot (in-play prices) and events without a fixture.

Validation: unique (fixture_key, source, bookmaker, market, outcome, line, is_closing,
snapshot_at); prices > 1; market/outcome/line consistent; pre-match rows available before
kickoff, closing rows at kickoff; h2h overround (Σ 1/price over home/draw/away per fixture,
source, bookmaker, closing flag and snapshot) within [OVERROUND_MIN, OVERROUND_MAX] — groups
outside, or missing an outcome, are reported, and more than 1% of them fails the build.

Measured on raw/ 2026-10-06 (101,074 rows: football-data 98,612 for 2016/17 – 2026/27 GW5,
Odds API 2,462 from 2 snapshots; ~1 s): every football-data match joins its fixture with
the same date; prices 1.04–42.94; AH lines −3.75…+3.0. h2h overround ranges: `avg`
1.027–1.073, `pinnacle` 1.0005–1.059, `betfair_ex` 1.00006–1.031, Odds API bookmakers
1.002–1.103 — except one Smarkets (exchange) group at 1.384, the only outlier of 16,120
groups (reported, not fatal). So [1.0, 1.2] holds with room on both sides; exchanges sit just
above 1.0. Pre-match collection lead (kickoff − available_at) is 1 h – 3 d 6 h; 15 matches
(Friday/Tuesday kickoffs before 16:00 UK) are capped at kickoff − 1 h.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
import pandera.pandas as pa

from fplopt.build.common import UK, UTC_US, BuildContext
from fplopt.build.fixtures import FIRST_SEASON, football_data_frames
from fplopt.build.teams import TeamResolver
from fplopt.ingest.raw_store import RawStore
from fplopt.seasons import season_start_year

log = logging.getLogger(__name__)

ERA_B_FIRST_SEASON = 2019
TOTALS_LINE = 2.5
MARKET_OUTCOMES = {
    "h2h": ("home", "draw", "away"),
    "totals": ("over", "under"),
    "ah": ("home", "away"),
}
OVERROUND_MIN = 1.0
OVERROUND_MAX = 1.2
MAX_OUTLIER_SHARE = 0.01
COLLECTION_HOUR = 15  # UK local
LAST_COLLECTION_BEFORE_KICKOFF = pd.Timedelta(hours=1)
KEY = (
    "fixture_key",
    "source",
    "bookmaker",
    "market",
    "outcome",
    "line",
    "is_closing",
    "snapshot_at",
)
COLUMNS = [
    "fixture_key",
    "season",
    "source",
    "bookmaker",
    "market",
    "outcome",
    "line",
    "price",
    "is_closing",
    "snapshot_at",
    "event_time",
    "available_at",
]


class OddsValidationError(ValueError):
    """Odds rows that cannot be placed on a fixture, or implausible prices."""


# --- football-data allowlist -------------------------------------------------------------


@dataclass(frozen=True)
class OddsColumn:
    """One allowlisted football-data price column."""

    column: str
    bookmaker: str
    market: str
    outcome: str
    is_closing: bool = False
    line_column: str | None = None  # AH: the handicap column the price belongs to


def _market(
    bookmaker: str,
    market: str,
    columns: tuple[str, ...],
    closing: bool = False,
    line_column: str | None = None,
) -> tuple[OddsColumn, ...]:
    outcomes = MARKET_OUTCOMES[market]
    return tuple(
        OddsColumn(column, bookmaker, market, outcome, closing, line_column)
        for column, outcome in zip(columns, outcomes, strict=True)
    )


ERA_A: tuple[OddsColumn, ...] = (
    *_market("avg", "h2h", ("BbAvH", "BbAvD", "BbAvA")),
    *_market("avg", "totals", ("BbAv>2.5", "BbAv<2.5")),
    *_market("avg", "ah", ("BbAvAHH", "BbAvAHA"), line_column="BbAHh"),
    *_market("pinnacle", "h2h", ("PSH", "PSD", "PSA")),
    *_market("pinnacle", "h2h", ("PSCH", "PSCD", "PSCA"), closing=True),
)
ERA_B: tuple[OddsColumn, ...] = (
    *_market("avg", "h2h", ("AvgH", "AvgD", "AvgA")),
    *_market("avg", "totals", ("Avg>2.5", "Avg<2.5")),
    *_market("avg", "ah", ("AvgAHH", "AvgAHA"), line_column="AHh"),
    *_market("avg", "h2h", ("AvgCH", "AvgCD", "AvgCA"), closing=True),
    *_market("avg", "totals", ("AvgC>2.5", "AvgC<2.5"), closing=True),
    *_market("avg", "ah", ("AvgCAHH", "AvgCAHA"), closing=True, line_column="AHCh"),
    *_market("pinnacle", "h2h", ("PSH", "PSD", "PSA")),
    *_market("pinnacle", "totals", ("P>2.5", "P<2.5")),
    *_market("pinnacle", "ah", ("PAHH", "PAHA"), line_column="AHh"),
    *_market("pinnacle", "h2h", ("PSCH", "PSCD", "PSCA"), closing=True),
    *_market("pinnacle", "totals", ("PC>2.5", "PC<2.5"), closing=True),
    *_market("pinnacle", "ah", ("PCAHH", "PCAHA"), closing=True, line_column="AHCh"),
    *_market("betfair_ex", "h2h", ("BFEH", "BFED", "BFEA")),
    *_market("betfair_ex", "totals", ("BFE>2.5", "BFE<2.5")),
    *_market("betfair_ex", "ah", ("BFEAHH", "BFEAHA"), line_column="AHh"),
    *_market("betfair_ex", "h2h", ("BFECH", "BFECD", "BFECA"), closing=True),
    *_market("betfair_ex", "totals", ("BFEC>2.5", "BFEC<2.5"), closing=True),
    *_market("betfair_ex", "ah", ("BFECAHH", "BFECAHA"), closing=True, line_column="AHCh"),
)
ALLOWLIST_COLUMNS: tuple[str, ...] = tuple(
    sorted(
        {c.column for c in (*ERA_A, *ERA_B)}
        | {c.line_column for c in (*ERA_A, *ERA_B) if c.line_column}
    )
)


def allowlist(season: int) -> tuple[OddsColumn, ...]:
    return ERA_B if season >= ERA_B_FIRST_SEASON else ERA_A


def football_data_odds(fd: pd.DataFrame) -> pd.DataFrame:
    """Long odds rows (season, fd_date, home/away team keys, bookmaker, market, outcome, line,
    price, is_closing) from football-data rows carrying the allowlisted columns."""
    keys = ["season", "fd_date", "home_team_key", "away_team_key"]
    frames = []
    for season, rows in fd.groupby("season", sort=True):
        for spec in allowlist(int(season)):
            if spec.column not in rows.columns:
                continue
            price = pd.to_numeric(rows[spec.column])
            if spec.line_column is not None:
                if spec.line_column not in rows.columns:
                    continue
                line = pd.to_numeric(rows[spec.line_column])
            elif spec.market == "totals":
                line = pd.Series(TOTALS_LINE, index=rows.index)
            else:
                line = pd.Series(float("nan"), index=rows.index)
            keep = price.notna() & (line.notna() | (spec.market == "h2h"))
            if not keep.any():
                continue
            frames.append(
                rows.loc[keep, keys].assign(
                    bookmaker=spec.bookmaker,
                    market=spec.market,
                    outcome=spec.outcome,
                    line=line[keep],
                    price=price[keep],
                    is_closing=spec.is_closing,
                )
            )
    columns = [*keys, "bookmaker", "market", "outcome", "line", "price", "is_closing"]
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)[columns]


def prematch_available_at(kickoff: pd.Series) -> pd.Series:
    """When football-data's pre-match odds were collected (see module docstring)."""
    local = kickoff.dt.tz_convert(UK)
    weekday = local.dt.weekday  # Monday 0
    midweek = weekday.between(1, 3)
    days_back = weekday.where(~midweek, weekday - 1)
    days_back = days_back.where(midweek, (weekday - 4) % 7)
    day = local.dt.tz_localize(None).dt.normalize() - pd.to_timedelta(days_back, unit="D")
    collected = (day + pd.Timedelta(hours=COLLECTION_HOUR)).dt.tz_localize(UK)
    collected = collected.dt.tz_convert("UTC").astype(UTC_US)
    cap = (kickoff - LAST_COLLECTION_BEFORE_KICKOFF).astype(UTC_US)
    return collected.where(collected <= cap, cap)


def football_data_snapshots(odds: pd.DataFrame, fixture: pd.DataFrame) -> pd.DataFrame:
    """Place football-data odds rows on fixtures (by season, home, away; the date must equal
    `fixture.fd_date`) and set the timing columns."""
    keys = ["season", "home_team_key", "away_team_key"]
    fixtures = fixture[[*keys, "fixture_key", "kickoff_time", "event_time", "fd_date"]].rename(
        columns={"fd_date": "fixture_fd_date"}
    )
    df = odds.merge(fixtures, on=keys, how="left", validate="many_to_one")
    problems = []
    unmatched = df[df["fixture_key"].isna()].drop_duplicates(keys)
    if len(unmatched):
        problems.append(
            f"{len(unmatched)} football-data match(es) without a fixture:\n"
            + unmatched[[*keys, "fd_date"]].head(10).to_string()
        )
    matched = df[df["fixture_key"].notna()]
    wrong = matched[matched["fd_date"] != matched["fixture_fd_date"]].drop_duplicates(keys)
    if len(wrong):
        problems.append(
            f"{len(wrong)} football-data date mismatch(es) with fixture.fd_date (rebuild "
            "`fixture` after a football-data refresh):\n"
            + wrong[[*keys, "fd_date", "fixture_fd_date"]].head(10).to_string()
        )
    if problems:
        raise OddsValidationError("football-data odds: " + "\n".join(problems))
    closing = df["is_closing"].astype(bool)
    available = prematch_available_at(df["kickoff_time"]).where(~closing, df["kickoff_time"])
    return df.assign(
        fixture_key=df["fixture_key"].astype("int64"),
        source="football-data",
        is_closing=closing,
        snapshot_at=available,
        available_at=available,
    )[COLUMNS]


# --- The Odds API ------------------------------------------------------------------------


def odds_api_rows(
    events: list[dict[str, Any]],
    taken_at: pd.Timestamp,
    fixture: pd.DataFrame,
    resolver: TeamResolver,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Long odds rows from one Odds API snapshot, plus counts of skipped events by reason
    ('started': kicked off by the snapshot; 'no fixture': not in the fixture table)."""
    taken_at = pd.Timestamp(taken_at).tz_convert("UTC").as_unit("us")
    lookup = {
        (int(r.season), int(r.home_team_key), int(r.away_team_key)): r
        for r in fixture[
            ["season", "home_team_key", "away_team_key", "fixture_key", "kickoff_time"]
            + ["event_time"]
        ].itertuples(index=False)
    }
    skipped: Counter[str] = Counter()
    rows = []
    for event in events:
        commence = pd.Timestamp(event["commence_time"]).tz_convert("UTC")
        if commence <= taken_at:
            skipped["started"] += 1
            continue
        home, away = event["home_team"], event["away_team"]
        season = season_start_year(commence.to_pydatetime())
        fx = lookup.get((season, resolver.odds_api(home), resolver.odds_api(away)))
        if fx is None:
            skipped["no fixture"] += 1
            continue
        if pd.notna(fx.kickoff_time) and fx.kickoff_time <= taken_at:
            skipped["started"] += 1
            continue
        if pd.notna(fx.kickoff_time) and fx.kickoff_time != commence:
            log.info(
                "Odds API %s v %s: commence %s, FPL kickoff %s",
                home,
                away,
                commence,
                fx.kickoff_time,
            )
        h2h_names = {home: "home", away: "away", "Draw": "draw"}
        for book in event["bookmakers"]:
            for market in book["markets"]:
                for outcome in market["outcomes"]:
                    if market["key"] == "h2h":
                        name, line = h2h_names.get(outcome["name"]), None
                    elif market["key"] == "totals" and outcome.get("point") == TOTALS_LINE:
                        name, line = {"Over": "over", "Under": "under"}.get(outcome["name"]), 2.5
                    else:
                        continue
                    if name is None:
                        raise OddsValidationError(
                            f"Odds API {home} v {away} {book['key']} {market['key']}: "
                            f"unknown outcome {outcome['name']!r}"
                        )
                    rows.append(
                        (fx.fixture_key, season, book["key"], market["key"], name, line)
                        + (float(outcome["price"]), fx.event_time)
                    )
    df = pd.DataFrame(
        rows,
        columns=["fixture_key", "season", "bookmaker", "market", "outcome", "line", "price"]
        + ["event_time"],
    )
    df = df.assign(
        fixture_key=df["fixture_key"].astype("int64"),
        season=df["season"].astype("int64"),
        line=df["line"].astype("Float64"),
        event_time=df["event_time"].astype(UTC_US),
        source="odds-api",
        is_closing=False,
        snapshot_at=pd.Series(taken_at, index=df.index, dtype=UTC_US),
        available_at=pd.Series(taken_at, index=df.index, dtype=UTC_US),
    )
    return df[COLUMNS], dict(skipped)


# --- validation --------------------------------------------------------------------------


def check_overround(df: pd.DataFrame) -> pd.DataFrame:
    """h2h overround per (fixture, source, bookmaker, closing, snapshot). Logs the groups
    outside [OVERROUND_MIN, OVERROUND_MAX] or missing an outcome; raises if they exceed
    MAX_OUTLIER_SHARE of all groups. Returns the per-group overrounds."""
    h2h = df[df["market"] == "h2h"]
    groups = h2h.assign(inv=1.0 / h2h["price"]).groupby(
        ["fixture_key", "source", "bookmaker", "is_closing", "snapshot_at"], sort=True
    )
    stats = groups.agg(overround=("inv", "sum"), outcomes=("outcome", "nunique")).reset_index()
    bad = (
        (stats["outcomes"] != len(MARKET_OUTCOMES["h2h"]))
        | (stats["overround"] < OVERROUND_MIN)
        | (stats["overround"] > OVERROUND_MAX)
    )
    if bad.any():
        share = bad.mean()
        message = (
            f"h2h overround outside [{OVERROUND_MIN}, {OVERROUND_MAX}] or incomplete in "
            f"{bad.sum()} of {len(stats)} group(s) ({share:.2%}):\n"
            + stats[bad].head(10).to_string()
        )
        if share > MAX_OUTLIER_SHARE:
            raise OddsValidationError(message)
        log.warning(message)
    return stats


def _consistent(df: pd.DataFrame) -> pd.Series:
    valid = pd.Series(False, index=df.index)
    for market, outcomes in MARKET_OUTCOMES.items():
        valid |= (df["market"] == market) & df["outcome"].isin(outcomes)
    line = df["line"]
    h2h, totals, ah = (df["market"] == m for m in ("h2h", "totals", "ah"))
    quarter = (line * 4).round() == line * 4
    line_ok = (h2h & line.isna()) | (totals & (line == TOTALS_LINE)) | (ah & quarter)
    timing = (df["is_closing"] & (df["available_at"] == df["event_time"])) | (
        ~df["is_closing"] & (df["available_at"] < df["event_time"])
    )
    return (
        valid
        & line_ok.fillna(False).astype(bool)
        & timing
        & (df["snapshot_at"] == df["available_at"])
        & (df["fixture_key"] // 1000 == df["season"])
        & (~df["is_closing"] | (df["source"] == "football-data"))
    )


SCHEMA = pa.DataFrameSchema(
    {
        "fixture_key": pa.Column("int64"),
        "season": pa.Column("int64", pa.Check.ge(FIRST_SEASON)),
        "source": pa.Column(str, pa.Check.isin(["football-data", "odds-api"])),
        "bookmaker": pa.Column(str),
        "market": pa.Column(str, pa.Check.isin(list(MARKET_OUTCOMES))),
        "outcome": pa.Column(str),
        "line": pa.Column("Float64", nullable=True),
        "price": pa.Column("float64", pa.Check.gt(1.0)),
        "is_closing": pa.Column(bool),
        "snapshot_at": pa.Column(UTC_US),
        "event_time": pa.Column(UTC_US),
        "available_at": pa.Column(UTC_US),
    },
    checks=[
        pa.Check(_consistent, error="market/outcome/line/timing consistency"),
        pa.Check(lambda df: ~df.duplicated(list(KEY)), error=f"{KEY} must be unique"),
    ],
    strict=True,
    ordered=True,
)
SORT_BY = KEY


def build_odds_snapshot(ctx: BuildContext) -> pd.DataFrame:
    resolver = TeamResolver(ctx.table("team_dim"))
    fixture = ctx.table("fixture")
    fd = football_data_frames(ctx, resolver, ALLOWLIST_COLUMNS)
    frames = [football_data_snapshots(football_data_odds(fd), fixture)]
    skipped: Counter[str] = Counter()
    for taken_at, path in ctx.store.entries("odds", "soccer_epl"):
        rows, skips = odds_api_rows(RawStore.read_json(path), _ts(taken_at), fixture, resolver)
        frames.append(rows)
        skipped.update(skips)
    if skipped:
        log.info("Odds API events skipped: %s", dict(skipped))
    df = pd.concat([f for f in frames if len(f)] or [frames[0]], ignore_index=True)
    df = df.astype(
        {
            "fixture_key": "int64",
            "season": "int64",
            "line": "Float64",
            "price": "float64",
            "is_closing": bool,
            "snapshot_at": UTC_US,
            "event_time": UTC_US,
            "available_at": UTC_US,
        }
    )
    for column in ("source", "bookmaker", "market", "outcome"):
        df[column] = df[column].astype(str)
    stats = check_overround(df)
    log.info(
        "h2h overround by source/bookmaker/closing:\n%s",
        stats.groupby(["source", "bookmaker", "is_closing"])["overround"]
        .describe(percentiles=[0.001, 0.5, 0.999])
        .to_string(),
    )
    return df[COLUMNS]


def _ts(taken_at: datetime) -> pd.Timestamp:
    return pd.Timestamp(taken_at).tz_convert("UTC").as_unit("us")
