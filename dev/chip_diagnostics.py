"""Chip diagnostics driver (spec docs/superpowers/specs/2026-10-29-chip-diagnostics-design.md):
reads the S0 / H / P develop runs, writes the tables to results/chips-diag/.

    FPLOPT_DATA_DIR=<data> uv run python dev/chip_diagnostics.py \
        --s0 results/chips-diag-s0 --h results/chips-diag-h --p results/chips-diag-p \
        [--out results/chips-diag] [--upcoming 2026:10-19]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fplopt.backtest import chip_diagnostics as cd
from fplopt.backtest.chip_values import load_terminal_values
from fplopt.backtest.evaluate import full_run_window_test
from fplopt.features.history import training_frame
from fplopt.features.store import DataStore
from fplopt.settings import Settings

ARMS = ("S0", "H", "P")
COMPARISONS = (("P", "H"), ("P", "S0"), ("H", "S0"))


def load(run: Path) -> pd.DataFrame:
    return pd.read_parquet(run / "gws.parquet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s0", type=Path, required=True)
    ap.add_argument("--h", type=Path, required=True)
    ap.add_argument("--p", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/chips-diag"))
    ap.add_argument("--upcoming", default=None, help="SEASON:FROM-TO, e.g. 2026:10-19")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Settings.from_env().data_dir
    runs = {"S0": args.s0, "H": args.h, "P": args.p}
    gws = {arm: load(path) for arm, path in runs.items()}
    (out / "inputs.json").write_text(
        json.dumps({arm: str(p.resolve()) for arm, p in runs.items()}, indent=2), encoding="utf-8"
    )

    # Q1: comparisons, gap by chip and by cell, chip timing
    comparisons = {}
    for a, b in COMPARISONS:
        m = cd.paired(gws[a], gws[b])
        real, xg = full_run_window_test(m, "diff"), full_run_window_test(m, "diff_xg")
        comparisons[f"{a} vs {b}"] = {
            "mean_per_gw": real.mean,
            "p_one_sided": real.p_one_sided,
            "n_cells": real.n_cells,
            "xg_mean_per_gw": xg.mean,
            "season_totals": m.groupby(["season", "start_id"])["diff"]
            .sum()
            .groupby("season")
            .mean()
            .to_dict(),
        }
        cd.gap_by_chip(gws[a], gws[b]).to_csv(out / f"q1-gap-chip-{a}-{b}.csv", index=False)
        cd.gap_by_cell(gws[a], gws[b]).to_csv(out / f"q1-gap-cell-{a}-{b}.csv", index=False)
        print(f"{a} vs {b}: {real.mean:+.3f}/GW, p {real.p_one_sided:.4f}, xG {xg.mean:+.3f}/GW")
    (out / "comparisons.json").write_text(
        json.dumps(comparisons, indent=2, default=str), encoding="utf-8"
    )
    timing = pd.concat(
        [
            g.dropna(subset=["chip"]).assign(arm=arm)[["arm", "chip", "gw_index"]]
            for arm, g in gws.items()
        ]
    )
    timing.groupby(["arm", "chip", "gw_index"]).size().rename("n").reset_index().to_csv(
        out / "q1-timing.csv", index=False
    )

    # Q2: gain distributions (S0)
    s0 = gws["S0"].copy()
    schedule = pd.read_parquet(data_dir / "schedule.parquet")
    player_gw = pd.read_parquet(data_dir / "player_gw.parquet")
    s0 = s0.join(cd.gw_kind(schedule, player_gw, s0))
    s0["start_type"] = cd.start_type(s0["start_id"])
    s0["ft"] = s0["free_transfers"].clip(upper=3)
    s0["bank_band"] = pd.cut(
        s0["bank"], [-1, 5, 20, 50, 10_000], labels=["<=0.5", "0.6-2.0", "2.1-5.0", ">5.0"]
    )
    for by in (["gw_index"], ["captain_kind"], ["gw_type"], ["start_type"], ["ft"], ["bank_band"]):
        cd.gain_distribution(s0, by).to_csv(out / f"q2-gain-{by[0]}.csv", index=False)

    # Q3: forecast vs outcome
    slopes = {
        "3xc (S0, all rows)": cd.forecast_slope(s0["gain_tc"], s0["captain_points"]),
        "bboost (S0, all rows)": cd.forecast_slope(s0["gain_bb"], s0["bench_points"]),
    }
    for chip, gain, real in (
        ("3xc", "gain_tc", "captain_points"),
        ("bboost", "gain_bb", "bench_points"),
    ):
        top = s0[s0[gain] >= s0[gain].quantile(0.9)]
        slopes[f"{chip} (S0, top decile)"] = cd.forecast_slope(top[gain], top[real])
    for arm in ("H", "P"):
        m = cd.paired(gws[arm], gws["S0"]).merge(s0[cd.KEYS + ["gain_fh", "gain_wc"]], on=cd.KEYS)
        played = gws[arm].set_index(cd.KEYS)["chip"]
        m["played"] = played.reindex(pd.MultiIndex.from_frame(m[cd.KEYS])).to_numpy()
        fh = m[m["played"] == "freehit"]
        slopes[f"freehit ({arm} vs S0, that GW)"] = cd.forecast_slope(fh["gain_fh"], fh["diff"])
        m = m.sort_values(cd.KEYS)
        m["diff_6"] = m.groupby(["season", "start_id"])["diff"].transform(
            lambda s: s[::-1].rolling(cd.WC_SPAN, min_periods=1).sum()[::-1]
        )
        wc = m[m["played"] == "wildcard"]
        slopes[f"wildcard ({arm} vs S0, 6 GWs)"] = cd.forecast_slope(wc["gain_wc"], wc["diff_6"])
    pd.DataFrame(slopes).T.to_csv(out / "q3-forecast-slopes.csv")

    # Q4: holding value
    table = load_terminal_values(1.0)
    hv = cd.holding_value(s0, {"3xc": "captain_points", "bboost": "bench_points"}, table)
    hv.to_csv(out / "q4-holding-value.csv", index=False)

    # Q5: captain congestion (training_frame as of the first deadline after develop)
    gameweek = pd.read_parquet(data_dir / "gameweek.parquet")
    after = gameweek.loc[
        (gameweek["season"] == 2023) & (gameweek["gw"] == 1), "deadline_time"
    ].iloc[0]
    features = training_frame(DataStore(data_dir).as_of(after))
    features = features[["player_key", "season", "gw", "europe_week", "rest_days_all"]]
    s0["congested"] = cd.captain_congestion(s0, features)
    cd.ratio_split(s0, "congested", "captain_xp", "captain_points").to_csv(
        out / "q5-captain-congestion-s0.csv", index=False
    )
    shares = {"S0 gain-weighted": cd.weighted_share(s0["congested"], s0["gain_tc"])}
    for arm in ("H", "P"):
        tc = gws[arm][gws[arm]["chip"] == "3xc"]
        shares[f"{arm} TC picks"] = float(cd.captain_congestion(tc, features).mean())
    (out / "q5-tc-congestion-share.json").write_text(json.dumps(shares, indent=2), encoding="utf-8")

    # Q6: doubles and blanks ahead (as-of schedule now; no scoring)
    if args.upcoming:
        season, span = args.upcoming.split(":")
        lo, hi = (int(x) for x in span.split("-"))
        store = DataStore(data_dir)
        future = gameweek[
            (gameweek["season"] == int(season))
            & (gameweek["deadline_time"] > pd.Timestamp.now(tz="UTC"))
        ]
        view = store.as_of(future["deadline_time"].min())
        sched = view.table("schedule")
        sched = sched[(sched["season"] == int(season)) & sched["gw"].between(lo, hi)]
        rows = sched[["season", "gw"]].drop_duplicates().assign(captain=-1)
        kinds = cd.gw_kind(sched, player_gw.iloc[:0], rows)
        rows.join(kinds)[["season", "gw", "n_double_clubs", "n_blank_clubs", "gw_type"]].to_csv(
            out / "q6-upcoming.csv", index=False
        )
    print(f"tables in {out}")


if __name__ == "__main__":
    main()
