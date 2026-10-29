"""Chip diagnostics (chip research, sub-project 1; spec
docs/superpowers/specs/2026-10-29-chip-diagnostics-design.md): where chip value comes from,
read from backtest GW rows (`backtest run` -> gws.parquet) and the schedule. Pure functions
over frames; `dev/chip_diagnostics.py` loads the inputs and writes the tables.

S0 rows (`chipgains`) carry each chip's ex-ante gain (`gain_tc`, `gain_bb`, `gain_fh`,
`gain_wc`); since S0 plays no chip, its `captain_points` and `bench_points` are the realized
gains a Triple Captain and a Bench Boost would have added in that GW.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fplopt.backtest.chip_values import GAIN_COLUMNS

__all__ = (
    "CELL",
    "KEYS",
    "WC_SPAN",
    "attribute_chips",
    "forecast_slope",
    "gain_distribution",
    "gap_by_cell",
    "gap_by_chip",
    "gw_kind",
    "paired",
    "start_type",
)

KEYS = ["season", "start_id", "gw_index"]
WC_SPAN = 6  # GWs a Wildcard is credited with: its own and the next 5 (the plan horizon)
CELL = 8  # GWs per cell, as the chip study's full-run window t-test


def attribute_chips(gws: pd.DataFrame) -> pd.Series:
    """Per row, the chip its points are attributed to: the chip played that GW, else
    "wildcard" within `WC_SPAN` GWs of a Wildcard in the same run (season, start_id), else
    "none". Index as `gws`."""
    out = pd.Series("none", index=gws.index, dtype="object")
    ordered = gws.sort_values(KEYS, kind="mergesort")
    for _, run in ordered.groupby(["season", "start_id"], sort=False):
        left = 0  # GWs of a Wildcard's span still to credit
        for idx, chip in zip(run.index, run["chip"], strict=True):
            if isinstance(chip, str) and chip:
                out[idx] = chip
                if chip == "wildcard":
                    left = WC_SPAN - 1
                    continue
            elif left > 0:
                out[idx] = "wildcard"
            left = max(0, left - 1)
    return out


def paired(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """The GW rows of `a` and `b` matched on `KEYS`: `diff` / `diff_xg` (a − b net points,
    realized / xG-scored) and each side's attributed chip (`chip_a`, `chip_b`)."""
    cols = KEYS + ["net_points", "xg_net_points"]
    left = a[cols].assign(chip=attribute_chips(a))
    right = b[cols].assign(chip=attribute_chips(b))
    m = left.merge(right, on=KEYS, suffixes=("_a", "_b"))
    if not len(m) == len(a) == len(b):
        raise ValueError(f"runs don't match: {len(a)} and {len(b)} rows, {len(m)} paired")
    m["diff"] = (m["net_points_a"] - m["net_points_b"]).astype("float64")
    m["diff_xg"] = (m["xg_net_points_a"] - m["xg_net_points_b"]).astype("float64")
    return m


def gap_by_chip(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """a − b by the pair of attributed chips: n, mean_diff, mean_diff_xg and
    contribution_per_gw (the group's summed diff over all paired rows: the contributions sum
    to the mean gap per GW). Sorted by contribution, largest first."""
    m = paired(a, b)
    total = len(m)
    table = (
        m.groupby(["chip_a", "chip_b"])
        .agg(
            n=("diff", "size"),
            mean_diff=("diff", "mean"),
            mean_diff_xg=("diff_xg", "mean"),
            sum_diff=("diff", "sum"),
        )
        .reset_index()
    )
    table["contribution_per_gw"] = table["sum_diff"] / total
    return table.drop(columns="sum_diff").sort_values(
        "contribution_per_gw", ascending=False, key=np.abs, ignore_index=True
    )


def gap_by_cell(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """a − b per `CELL`-GW cell of gw_index (1 = GWs 1–8): n, mean_diff, mean_diff_xg."""
    m = paired(a, b)
    m["cell"] = (m["gw_index"] - 1) // CELL + 1
    return (
        m.groupby("cell")
        .agg(n=("diff", "size"), mean_diff=("diff", "mean"), mean_diff_xg=("diff_xg", "mean"))
        .reset_index()
    )


def start_type(start_id: pd.Series) -> pd.Series:
    """ "template" or "random" from a start id (template@1, random0@20, ...)."""
    return start_id.str.extract(r"^(template|random)", expand=False)


def _kind(n: int) -> str:
    return "blank" if n == 0 else "single" if n == 1 else "double"


def gw_kind(schedule: pd.DataFrame, player_gw: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Per row of `rows` (season, gw, captain = player_key), in row order:
    `captain_fixtures` (his club's league fixtures that GW, his club from `player_gw`),
    `captain_kind` (blank / single / double), `n_double_clubs` and `n_blank_clubs` (the
    season's clubs with ≥ 2 / 0 fixtures that GW; a season's clubs are the schedule's) and
    `gw_type` (normal, double, blank or double+blank)."""
    sides = pd.concat(
        [
            schedule[["season", "gw", "home_team_key"]].rename(
                columns={"home_team_key": "team_key"}
            ),
            schedule[["season", "gw", "away_team_key"]].rename(
                columns={"away_team_key": "team_key"}
            ),
        ]
    )
    sides = sides[sides["team_key"] > 0]
    counts = sides.groupby(["season", "gw", "team_key"]).size().rename("n").reset_index()
    clubs = sides.drop_duplicates(["season", "team_key"])[["season", "team_key"]]
    gws_ = schedule.drop_duplicates(["season", "gw"])[["season", "gw"]]
    grid = gws_.merge(clubs, on="season").merge(counts, on=["season", "gw", "team_key"], how="left")
    grid["n"] = grid["n"].fillna(0).astype("int64")
    per_gw = grid.groupby(["season", "gw"]).agg(
        n_double_clubs=("n", lambda s: int((s >= 2).sum())),
        n_blank_clubs=("n", lambda s: int((s == 0).sum())),
    )
    club = player_gw[["player_key", "season", "gw", "team_key"]].rename(
        columns={"player_key": "captain"}
    )
    out = rows[["season", "gw", "captain"]].reset_index(drop=True)
    out = out.merge(club, on=["season", "gw", "captain"], how="left")
    out = out.merge(counts, on=["season", "gw", "team_key"], how="left")
    out["captain_fixtures"] = out["n"].fillna(0).astype("int64")
    out["captain_kind"] = out["captain_fixtures"].map(_kind)
    out = out.join(per_gw, on=["season", "gw"])
    out[["n_double_clubs", "n_blank_clubs"]] = (
        out[["n_double_clubs", "n_blank_clubs"]].fillna(0).astype("int64")
    )
    double, blank = out["n_double_clubs"] > 0, out["n_blank_clubs"] > 0
    out["gw_type"] = np.select(
        [double & blank, double, blank], ["double+blank", "double", "blank"], "normal"
    )
    out.index = rows.index
    return out[["captain_fixtures", "captain_kind", "n_double_clubs", "n_blank_clubs", "gw_type"]]


def gain_distribution(rows: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Each chip's ex-ante gain (`GAIN_COLUMNS`) per group of `by`: n, mean, median, p75,
    p90. Rows without a gain (chip unavailable) are left out, chips without any gain too."""
    parts = []
    for chip, column in GAIN_COLUMNS.items():
        part = rows[by + [column]].rename(columns={column: "gain"}).dropna(subset=["gain"])
        if part.empty:
            continue
        g = part.groupby(by)["gain"]
        table = pd.DataFrame(
            {
                "n": g.size(),
                "mean": g.mean(),
                "median": g.median(),
                "p75": g.quantile(0.75),
                "p90": g.quantile(0.9),
            }
        ).reset_index()
        parts.append(table.assign(chip=chip))
    return pd.concat(parts, ignore_index=True)[["chip", *by, "n", "mean", "median", "p75", "p90"]]


def forecast_slope(predicted: pd.Series, realized: pd.Series) -> dict[str, float]:
    """OLS of realized on predicted over the pairs with both: slope, its SE, intercept, bias
    (mean realized − predicted) and n. Slope < 1: the larger the forecast gain, the more it
    overstates (the optimizer's curse). NaN statistics under 3 pairs."""
    frame = pd.DataFrame({"x": predicted, "y": realized}).astype("float64").dropna()
    n = len(frame)
    nan = float("nan")
    if n < 3:
        return {"slope": nan, "se": nan, "intercept": nan, "bias": nan, "n": n}
    x, y = frame["x"].to_numpy(), frame["y"].to_numpy()
    xc = x - x.mean()
    sxx = float(xc @ xc)
    slope = float(xc @ (y - y.mean()) / sxx) if sxx > 0 else nan
    intercept = float(y.mean() - slope * x.mean())
    resid = y - (intercept + slope * x)
    se = float(np.sqrt(resid @ resid / (n - 2) / sxx)) if sxx > 0 else nan
    return {"slope": slope, "se": se, "intercept": intercept, "bias": float((y - x).mean()), "n": n}
