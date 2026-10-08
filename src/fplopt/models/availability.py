"""The availability adjustment layer (Phase 5 plan, Task 3; PLAN §6.4). It applies FPL's
status flags and news to `fplopt.models.minutes.predict_minutes` output. Flags exist only
where player snapshots do (fplcache from 2020/21 GW32; PLAN §4 says 2021/22+); at earlier
deadlines there is nothing to apply and the frame comes back unchanged.

**Flag categories** (newest snapshot of the season before the deadline, as `player_pool`):
`a` (available, no chance given), `a100` (available, chance 100: news cleared), `d25`,
`d50`, `d75` (doubtful, by `chance_of_playing_next_round`), `i` (injured), `s` (suspended),
`u` (unavailable: loan, left the club), `n` (not eligible, e.g. against his parent club). A
pool player missing from the newest snapshot gets no adjustment.

**Return dates.** `expected_back` parses "Expected back DD Mon" (and FPL's "Suspended until
DD Mon") from `news`; the year is the first one that puts the date no more than
`PAST_DAYS` before the news was added (`news_added`, else the snapshot time) and less than a
year after that, so a December note about January rolls over. A fixture kicking off (UTC
date) before the date gets p_start = p_sub = 0; from the date on the model's prediction
stands (no flag adjustment). A date no year makes valid in that window (e.g. 29 Feb outside
a leap year) is ignored.

**Mapping** (players without a parsed date): logit p' = slope · logit p + γ[category,
min(h, 3)], one shared slope, one offset per category and horizon bucket (the flag is about
the next round; later GWs regress). p_sub's conditional P(sub | no start) is scaled by
min(1, p'/p): a player unlikely to start is also less likely to come off the bench. Then the
team normalization (`finish_minutes`) runs again, with zeroed players fixed at 0.

**Fit (`fit_availability(view, minutes_fit)`), leak-free and walk-forward:** every GW
deadline d of the visible seasons with a player snapshot before d, before the view's
deadline. The flags are those of the newest snapshot before d (`view.earlier(d)`, exactly
what a prediction at d sees). Outcomes are the visible training rows (`training_frame`) of
the player's club fixtures in GWs gw_index(d) … + 5, minus deterministic bans and players
with a parsed return date. The base p_start is an out-of-fold P(start) on his GW-d training
row (lags as of d; `minutes.out_of_fold_start`: the start part refitted on the other
season parity, since the minutes fit's own predictions on its training rows are sharper
than its predictions at a new deadline and would bias the mapping), horizon-decayed as
`predict_minutes` does (normalization not applied). Weighted log loss with a Gaussian prior
on the offsets (sd `OFFSET_SD`), minimized from slope 1 and offsets 0 (Newton,
`trust-exact` with the exact Hessian; deterministic). Fewer than `MIN_ROWS` rows: the
identity mapping (return dates still apply). PLAN §4 fixes the fit seasons as
2021/22–2022/23 with a check on 2023/24–2024/25; walk-forward refits use every snapshot
season visible at the cutoff instead, which is the same rule applied at each cutoff (no
validate-season row is ever used for a validate-season prediction it precedes).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

from fplopt.features.history import training_frame
from fplopt.features.store import AsOfView
from fplopt.models.minutes import (
    EPS,
    MinutesFit,
    conditionals,
    finish_minutes,
    out_of_fold_start,
)

__all__ = (
    "AvailabilityFit",
    "adjust_minutes",
    "expected_back",
    "flag_category",
    "fit_availability",
)

UTC_US = pd.DatetimeTZDtype("us", "UTC")
CATEGORIES = ("a", "a100", "d25", "d50", "d75", "i", "s", "u", "n")
HORIZON_BUCKETS = 4  # offsets for h = 0, 1, 2 and 3+
LOOKAHEAD = 5  # horizon GWs after the deadline's GW used in the fit
MIN_ROWS = 500
OFFSET_SD = 4.0
PAST_DAYS = 60
YEAR_DAYS = 366
MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
RETURN_PATTERN = r"(?i)(?:expected back|suspended until)\s+(\d{1,2})\s+([a-z]{3})"
FLAG_COLUMNS = (
    "snapshot_at",
    "season",
    "status",
    "chance_of_playing_next_round",
    "news",
    "news_added",
)


@dataclass(frozen=True)
class AvailabilityFit:
    """The fitted mapping: logit p' = slope · logit p + offset[(category, bucket)]."""

    slope: float
    offsets: tuple[tuple[str, int, float], ...]
    minutes_sub: float
    max_shift: float
    max_scale: float
    n_rows: int


def flag_category(status: pd.Series, chance: pd.Series) -> pd.Series:
    """CATEGORIES label per row ('' = none: no or unknown status)."""
    status = status.fillna("").astype("str")
    chance = pd.to_numeric(chance, errors="coerce").astype("float64")
    out = status.where(status.isin(("i", "s", "u", "n")), "")
    out = out.mask((status == "a") & chance.isna(), "a")
    out = out.mask((status == "a") & (chance == 100), "a100")
    for value in (25, 50, 75):
        out = out.mask((status == "d") & (chance == value), f"d{value}")
    return out.astype("str")


def expected_back(news: pd.Series, reference: pd.Series) -> pd.Series:
    """The return date (UTC midnight) in "Expected back DD Mon" / "Suspended until DD Mon"
    news, NaT if none. The year: the first of reference's year − 1, year, year + 1 that
    puts the date in [reference − PAST_DAYS, that + YEAR_DAYS) (`reference` = when the news
    was added); NaT if no year does (e.g. 29 Feb outside a leap year)."""
    parts = news.fillna("").astype("str").str.extract(RETURN_PATTERN)
    month_of = {name: number for number, name in enumerate(MONTHS, start=1)}
    day = pd.to_numeric(parts[0], errors="coerce")
    month = parts[1].str.lower().map(month_of)
    reference = pd.to_datetime(reference, utc=True)
    earliest = reference.dt.normalize() - pd.Timedelta(days=PAST_DAYS)
    out = pd.Series(pd.NaT, index=news.index, dtype=UTC_US)
    for offset in (1, 0, -1):  # the latest year first; earlier valid ones override
        year = reference.dt.year + offset
        parts = pd.DataFrame({"year": year, "month": month, "day": day}, dtype="float64")
        candidate = pd.to_datetime(parts, errors="coerce")
        candidate = pd.Series(candidate, index=news.index).dt.tz_localize("UTC").astype(UTC_US)
        ok = (candidate >= earliest) & (candidate < earliest + pd.Timedelta(days=YEAR_DAYS))
        ok = ok.fillna(False).astype("bool")
        out = out.mask(ok, candidate)
    return out


def _snapshot_flags(view: AsOfView) -> pd.DataFrame:
    """The newest player snapshot of the view's season before its deadline, one row per
    player: player_key, category, news and its reference time (empty without one)."""
    season, _ = view.gameweek_for_deadline()
    snaps = view.latest("player_snapshot", by=["player_key"], columns=list(FLAG_COLUMNS))
    snaps = snaps[snaps["season"] == season]
    snaps = snaps[snaps["snapshot_at"] == snaps["snapshot_at"].max()]
    reference = snaps["news_added"].where(snaps["news_added"].notna(), snaps["snapshot_at"])
    return pd.DataFrame(
        {
            "player_key": snaps["player_key"].astype("int64").to_numpy(),
            "category": flag_category(
                snaps["status"], snaps["chance_of_playing_next_round"]
            ).to_numpy(),
            "news": snaps["news"].to_numpy(),
            "reference": reference.to_numpy(),
        }
    )


def _with_return_dates(flags: pd.DataFrame) -> pd.DataFrame:
    """`_snapshot_flags` rows with the parsed return date `back` instead of news/reference
    (one parse for any number of concatenated snapshots)."""
    back = expected_back(flags["news"], flags["reference"])
    return flags.drop(columns=["news", "reference"]).assign(back=back.array)


def _flags(view: AsOfView) -> pd.DataFrame:
    """Per player of the newest player snapshot of the view's season before its deadline:
    category and parsed return date (empty without a snapshot)."""
    return _with_return_dates(_snapshot_flags(view))


def _identity(minutes_fit: MinutesFit, n_rows: int = 0) -> AvailabilityFit:
    params = minutes_fit.params
    return AvailabilityFit(
        1.0, (), minutes_fit.minutes_sub, params.max_shift, params.max_scale, n_rows
    )


def _fit_rows(
    view: AsOfView, minutes_fit: MinutesFit, rows: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Rows (base p, category, bucket, outcome start) for the fit, sorted by key."""
    gameweeks = view.table("gameweek", columns=["season", "gw", "gw_index", "deadline_time"])
    gameweeks = gameweeks[gameweeks["deadline_time"] < view.deadline]
    snapshot_times = view.table("player_snapshot", columns=["snapshot_at"])["snapshot_at"]
    if snapshot_times.empty:
        return pd.DataFrame()
    gameweeks = gameweeks[gameweeks["deadline_time"] > snapshot_times.min()]
    flags = []
    for row in gameweeks.sort_values("deadline_time", kind="mergesort").itertuples(index=False):
        found = _snapshot_flags(view.earlier(row.deadline_time))
        if len(found):
            flags.append(found.assign(season=row.season, base_index=row.gw_index))
    if not flags:
        return pd.DataFrame()
    flags = _with_return_dates(pd.concat(flags, ignore_index=True))
    flags = flags[(flags["category"] != "") & flags["back"].isna()]

    rows = training_frame(view) if rows is None else rows
    params = minutes_fit.params
    season, _ = view.gameweek_for_deadline()
    p_model = out_of_fold_start(rows, season, params)
    n_long = rows["n_long"].to_numpy(dtype="float64")
    starts_long = np.nan_to_num(rows["start_rate_long"].to_numpy(dtype="float64")) * n_long
    long_run = (starts_long + params.long_run_prior * p_model) / (n_long + params.long_run_prior)
    bases = rows[["player_key", "season", "gw_index"]].assign(p_model=p_model, long_run=long_run)
    bases = bases.drop_duplicates(["player_key", "season", "gw_index"], keep="first")
    bases = bases.rename(columns={"gw_index": "base_index"})
    flags = flags.merge(
        bases, on=["player_key", "season", "base_index"], how="inner", validate="one_to_one"
    )
    outcomes = rows.loc[rows["banned"] == 0, ["player_key", "season", "gw_index", "start"]]
    horizons = pd.DataFrame({"horizon": np.arange(LOOKAHEAD + 1)})
    expanded = flags.merge(horizons, how="cross")
    expanded["gw_index"] = expanded["base_index"] + expanded["horizon"]
    out = expanded.merge(
        outcomes, on=["player_key", "season", "gw_index"], how="inner", validate="many_to_many"
    )
    weight = np.power(params.horizon_decay, out["horizon"].to_numpy(dtype="float64"))
    out["base"] = weight * out["p_model"] + (1.0 - weight) * out["long_run"]
    out["bucket"] = out["horizon"].clip(upper=HORIZON_BUCKETS - 1)
    sort_by = ["season", "base_index", "player_key", "gw_index", "horizon"]
    return out.sort_values(sort_by, kind="mergesort").reset_index(drop=True)


def fit_availability(
    view: AsOfView, minutes_fit: MinutesFit, rows: pd.DataFrame | None = None
) -> AvailabilityFit:
    """Fit the flag mapping walk-forward on the view (see the module docstring). `rows`:
    `training_frame(view)` if the caller already built it (not modified)."""
    rows = _fit_rows(view, minutes_fit, rows)
    if len(rows) < MIN_ROWS:
        return _identity(minutes_fit, len(rows))
    cells = pd.MultiIndex.from_product([CATEGORIES, range(HORIZON_BUCKETS)])
    cell = cells.get_indexer(pd.MultiIndex.from_arrays([rows["category"], rows["bucket"]]))
    z = logit(np.clip(rows["base"].to_numpy(dtype="float64"), EPS, 1 - EPS))
    y = rows["start"].to_numpy(dtype="float64")
    n_cells = len(cells)

    n = len(y)  # the objective is per row, so the tolerance does not depend on the sample

    def loss(theta: np.ndarray) -> tuple[float, np.ndarray]:
        slope, offsets = theta[0], theta[1:]
        eta = slope * z + offsets[cell]
        p = expit(eta)
        # Log loss in a numerically stable form: log(1 + e^eta) - y * eta.
        value = float(np.sum(np.logaddexp(0.0, eta) - y * eta))
        value += float(0.5 * np.sum(offsets**2) / OFFSET_SD**2)
        residual = p - y
        grad_offsets = np.bincount(cell, weights=residual, minlength=n_cells)
        grad_offsets += offsets / OFFSET_SD**2
        grad = np.concatenate([[float(np.sum(residual * z))], grad_offsets])
        return value / n, grad / n

    def hessian(theta: np.ndarray) -> np.ndarray:
        p = expit(theta[0] * z + theta[1:][cell])
        w = p * (1.0 - p)
        out = np.zeros((n_cells + 1, n_cells + 1))
        out[0, 0] = float(np.sum(w * z * z))
        cross = np.bincount(cell, weights=w * z, minlength=n_cells)
        out[0, 1:] = cross
        out[1:, 0] = cross
        diagonal = np.bincount(cell, weights=w, minlength=n_cells) + 1.0 / OFFSET_SD**2
        out[np.arange(1, n_cells + 1), np.arange(1, n_cells + 1)] = diagonal
        return out / n

    start = np.concatenate([[1.0], np.zeros(n_cells)])
    result = minimize(
        loss, start, jac=True, hess=hessian, method="trust-exact", options={"gtol": 1e-9}
    )
    theta = result.x
    used = np.bincount(cell, minlength=n_cells) > 0
    offsets = tuple(
        (str(category), int(bucket), float(theta[1 + k]))
        for k, (category, bucket) in enumerate(cells)
        if used[k]
    )
    params = minutes_fit.params
    return AvailabilityFit(
        slope=float(theta[0]),
        offsets=offsets,
        minutes_sub=minutes_fit.minutes_sub,
        max_shift=params.max_shift,
        max_scale=params.max_scale,
        n_rows=len(rows),
    )


def adjust_minutes(view: AsOfView, minutes: pd.DataFrame, fit: AvailabilityFit) -> pd.DataFrame:
    """`predict_minutes` output with the flags and return dates applied and the team
    normalization redone; the same columns, dtypes and order. Unchanged without a player
    snapshot of the season before the deadline."""
    flags = _flags(view)
    if flags.empty or minutes.empty:
        return minutes
    season, _ = view.gameweek_for_deadline()
    kickoffs = view.schedule(season)[["fixture_key", "kickoff_time"]]
    rows = minutes[["player_key", "fixture_key", "horizon"]].merge(
        flags, on="player_key", how="left", validate="many_to_one"
    )
    rows = rows.merge(kickoffs, on="fixture_key", how="left", validate="many_to_one")
    cond = conditionals(minutes, fit.minutes_sub)
    p = cond["p_start"].to_numpy(dtype="float64")
    csub = cond["csub"].to_numpy(dtype="float64")

    back = rows["back"]
    kickoff_day = pd.to_datetime(rows["kickoff_time"], utc=True).dt.normalize()
    out_until_back = (back.notna() & (kickoff_day < back)).to_numpy(dtype=bool)

    offsets = {(category, bucket): value for category, bucket, value in fit.offsets}
    bucket = np.minimum(rows["horizon"].to_numpy(dtype="int64"), HORIZON_BUCKETS - 1)
    category = rows["category"].fillna("").astype("str").to_numpy()
    offset = np.array(
        [offsets.get((c, int(b)), np.nan) for c, b in zip(category, bucket, strict=True)],
        dtype="float64",
    )
    mapped = back.isna().to_numpy(dtype=bool) & ~np.isnan(offset) & (p > 0)
    adjusted = expit(fit.slope * logit(np.clip(p, EPS, 1 - EPS)) + np.nan_to_num(offset))
    ratio = np.where(p > 0, adjusted / np.where(p > 0, p, 1.0), 1.0)
    new_p = np.where(mapped, adjusted, p)
    new_csub = np.where(mapped, csub * np.minimum(ratio, 1.0), csub)
    new_p = np.where(out_until_back, 0.0, new_p)
    new_csub = np.where(out_until_back, 0.0, new_csub)
    cond = cond.assign(p_start=new_p, csub=new_csub, fixed=new_p <= 0)
    return finish_minutes(cond, fit.minutes_sub, fit.max_shift, fit.max_scale)
