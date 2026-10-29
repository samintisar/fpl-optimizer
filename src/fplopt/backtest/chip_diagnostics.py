"""Chip diagnostics (chip research, sub-project 1; spec
docs/superpowers/specs/2026-10-29-chip-diagnostics-design.md): where chip value comes from,
read from backtest GW rows (`backtest run` -> gws.parquet) and the schedule. Pure functions
over frames; `dev/chip_diagnostics.py` loads the inputs and writes the tables.

S0 rows (`chipgains`) carry each chip's ex-ante gain (`gain_tc`, `gain_bb`, `gain_fh`,
`gain_wc`); since S0 plays no chip, its `captain_points` and `bench_points` are the realized
gains a Triple Captain and a Bench Boost would have added in that GW.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from fplopt.backtest.chip_values import GAIN_COLUMNS
from fplopt.backtest.rules import backtest_rules
from fplopt.seasons import DEVELOP_SEASONS

__all__ = (
    "CELL",
    "CONGESTED_REST",
    "KEYS",
    "WC_SPAN",
    "attribute_chips",
    "captain_congestion",
    "forecast_slope",
    "gain_distribution",
    "gap_by_cell",
    "gap_by_chip",
    "gw_kind",
    "holding_value",
    "paired",
    "ratio_split",
    "start_type",
    "weighted_share",
)

KEYS = ["season", "start_id", "gw_index"]
WC_SPAN = 6  # GWs a Wildcard is credited with: its own and the next 5 (the plan horizon)
CELL = 8  # GWs per cell, as the chip study's full-run window t-test


def attribute_chips(gws: pd.DataFrame) -> pd.Series:
    """Per row, the chip its points are attributed to: the chip played that GW, else
    "wildcard" within `WC_SPAN` GWs of a Wildcard in the same run (season, start_id), else
    "none". Index as `gws`."""
    out = np.full(len(gws), "none", dtype=object)  # filled by position: labels may repeat
    ordered = gws[[*KEYS, "chip"]].reset_index(drop=True)
    ordered = ordered.sort_values(KEYS, kind="mergesort")
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
    return pd.Series(out, index=gws.index, dtype="object")


def paired(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """The GW rows of `a` and `b` matched on `KEYS`: `diff` / `diff_xg` (a − b net points,
    realized / xG-scored) and each side's attributed chip (`chip_a`, `chip_b`)."""
    cols = KEYS + ["net_points", "xg_net_points"]
    left = a[cols].assign(chip=attribute_chips(a))
    right = b[cols].assign(chip=attribute_chips(b))
    try:
        m = left.merge(right, on=KEYS, suffixes=("_a", "_b"), validate="one_to_one")
    except ValueError as exc:  # pandas' MergeError: duplicate keys on a side
        raise ValueError(f"runs don't match: duplicate rows per key ({exc})") from exc
    if not len(m) == len(a) == len(b):
        raise ValueError(f"runs don't match: {len(a)} and {len(b)} rows, {len(m)} paired")
    m["diff"] = (m["net_points_a"] - m["net_points_b"]).astype("float64")
    m["diff_xg"] = (m["xg_net_points_a"] - m["xg_net_points_b"]).astype("float64")
    return m


def gap_by_chip(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """a − b by the pair of attributed chips: n, mean_diff, mean_diff_xg and
    contribution_per_gw (the group's summed diff over all paired rows: the contributions sum
    to the mean gap per GW). Sorted by |contribution|, largest first."""
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
    `gw_type` (normal, double, blank or double+blank). A captain with no `player_gw` row
    that GW counts as blank (0 fixtures)."""
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
    if not parts:
        raise ValueError("no chip gains in these rows: run `backtest run --spec chipgains:...`")
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


def holding_value(
    rows: pd.DataFrame, realized: dict[str, str], table: dict[str, tuple[float, ...]]
) -> pd.DataFrame:
    """Per chip, window and GWs left `L`: `ex_ante` (the mean over runs of the best ex-ante
    gain among the window's last L GWs: `chip_values.estimate_terminal_values`, unclipped
    and per window), `realized` (the mean realized gain at that ex-ante best GW, from the
    column `realized[chip]`; NaN for chips without one), `table` (`table[chip][L − 1]`, the
    value P plans with), `exante_ratio` and `realized_ratio` (each over `table`), and n."""
    outside = sorted(int(x) for x in rows["season"].unique() if x not in DEVELOP_SEASONS)
    if outside:
        raise ValueError(f"chip values are estimated on develop seasons only, got {outside}")
    for column in {c for c in realized.values()}:
        if rows[column].isna().any():
            raise ValueError(f"realized column {column!r} has missing values")
    rules = backtest_rules(max(DEVELOP_SEASONS))
    out = []
    for chip, column in GAIN_COLUMNS.items():
        real_col = realized.get(chip)
        for window in [w for w in rules.chips if w.name == chip]:
            span = window.stop - window.start + 1
            sums = np.zeros(span)
            real_sums = np.zeros(span)
            counts = np.zeros(span, dtype="int64")
            for _, run in rows.groupby(["season", "policy", "start_id"], sort=True):
                by_gw = run.set_index("gw_index")
                best, best_real = -math.inf, math.nan
                for left in range(1, span + 1):
                    gw = window.stop - left + 1
                    if gw in by_gw.index and not pd.isna(by_gw.at[gw, column]):
                        gain = float(by_gw.at[gw, column])
                        if gain > best:
                            best = gain
                            best_real = float(by_gw.at[gw, real_col]) if real_col else math.nan
                    if best > -math.inf:
                        sums[left - 1] += best
                        real_sums[left - 1] += best_real
                        counts[left - 1] += 1
            values = table.get(chip, ())
            for left in range(1, span + 1):
                n = int(counts[left - 1])
                ex_ante = sums[left - 1] / n if n else math.nan
                real = real_sums[left - 1] / n if n and real_col else math.nan
                tab = float(values[left - 1]) if left <= len(values) else math.nan
                out.append(
                    {
                        "chip": chip,
                        "window_start": window.start,
                        "left": left,
                        "n": n,
                        "ex_ante": ex_ante,
                        "realized": real,
                        "table": tab,
                        "exante_ratio": ex_ante / tab if tab else math.nan,
                        "realized_ratio": real / tab if tab else math.nan,
                    }
                )
    return pd.DataFrame(out)


CONGESTED_REST = 3.0  # rest_days_all at or below: a short-rest fixture


def captain_congestion(rows: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    """Per row (season, gw, captain = player_key): True if any of his fixtures that GW has
    `europe_week` = 1 or `rest_days_all` ≤ `CONGESTED_REST`. `features`: player_key, season,
    gw, europe_week, rest_days_all (one row per player fixture, e.g. `training_frame`). Rows
    without a feature row are False."""
    flag = (features["europe_week"] == 1) | (features["rest_days_all"] <= CONGESTED_REST)
    per = (
        features.assign(flag=flag)
        .groupby(["player_key", "season", "gw"])["flag"]
        .any()
        .rename_axis(["captain", "season", "gw"])
    )
    keys = pd.MultiIndex.from_frame(rows[["captain", "season", "gw"]])
    return pd.Series(per.reindex(keys).fillna(False).to_numpy(dtype=bool), index=rows.index)


def ratio_split(rows: pd.DataFrame, flag: str, xp: str, real: str) -> pd.DataFrame:
    """Per value of `flag`: n, the summed `xp` and `real`, and ratio = Σ real / Σ xp."""
    g = rows.groupby(flag)
    table = pd.DataFrame({"n": g.size(), "xp": g[xp].sum(), "realized": g[real].sum()})
    table["ratio"] = table["realized"] / table["xp"]
    return table.reset_index()


def weighted_share(flag: pd.Series, weight: pd.Series) -> float:
    """Σ weight over flagged rows / Σ weight: the share a chip-gain-weighted pick would land
    in flagged fixtures."""
    w = weight.astype("float64").clip(lower=0)
    return float(w[flag.astype(bool)].sum() / w.sum())
