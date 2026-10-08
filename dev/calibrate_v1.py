"""v1 calibration and the ban residual (Phase 5 plan, Task 5; PLAN §5 *Calibration*). Dev
only: never imported by `fplopt`.

Subcommands:
- `walk`: the walk-forward, uncalibrated per player-fixture components of `v1`
  (`fplopt.models.assemble.fixture_components`, every horizon) at every GW deadline of the
  given seasons, fitted at the refit cutoff (memoized per cutoff), joined to the realized
  outcome of each fixture: minutes, start (FPL's `starts` where known, else the inferred
  starters, as the minutes model's targets), goals, clean sheet, the club's goals against
  and the points re-scored under the season's backtest rules. A player without a row in a
  played fixture scored nothing in 0 minutes; fixtures not played (postponed beyond the
  data) are dropped. One parquet per season in `--out`. `--ban-residual` uses
  `MinutesParams(ban_residual=True)`.
- `fit`: from a `walk` directory, the expanding-window calibrations: for each season S of
  `--apply` the maps fitted on the walk-forward rows of the seasons in [`--first-fit`, S)
  (`fplopt.models.calibration`). Effects are reported on develop seasons only (2017/18–
  2022/23; validate rows are only *fitted on*, for the 2024/25 and 2025/26+ entries, never
  scored): per variant (each part alone, then the chosen set) the per player-fixture xP MSE
  (horizon 0 and 1–5), Brier of P(start), of the team's P(CS) and of P(goal ≥ 1), the
  minutes 3-class log loss and the goals Poisson log-likelihood, as means over seasons.
  `--write` writes `src/fplopt/models/calibration_table.py` with the parts given by
  `--parts`.
- `ban`: compares two `walk` directories (ban residual off / on) on develop: P(start) Brier
  and minutes log loss on the banned rows and overall, and xP MSE.

    uv run python dev/calibrate_v1.py --data-dir <data> walk --seasons 2016-2024 --jobs 6 \
        --out results/p5-task5-walk [--ban-residual]
    uv run python dev/calibrate_v1.py fit --walk results/p5-task5-walk [--parts ...] [--write]
    uv run python dev/calibrate_v1.py ban --off results/p5-task5-walk --on results/p5-task5-ban
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pprint
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

from fplopt.backtest.rules import backtest_rules
from fplopt.backtest.scoring import score_matches
from fplopt.evaluate.metrics import minutes_class, ordinal_log_loss
from fplopt.features.history import infer_starts
from fplopt.features.store import DataStore
from fplopt.models.assemble import (
    V1Params,
    calibrate_inputs,
    calibrate_xp,
    calibration_fingerprint,
    fit_v1,
    fixture_components,
    fixture_points,
)
from fplopt.models.calibration import (
    Calibration,
    isotonic,
    linear_by_position,
    to_table,
)
from fplopt.models.fitted import refit_deadline
from fplopt.models.minutes import MinutesParams

DEVELOP = range(2017, 2023)  # scored seasons (2016/17 is burn-in)
LAST_FIT_SEASON = 2024  # validate rows are fitted on, never scored
TABLES = (
    "gameweek",
    "schedule",
    "fixture_snapshot",
    "player_match",
    "player_gw",
    "player_season",
    "player_snapshot",
    "team_match",
    "team_rating",
    "odds_snapshot",
)
SNAPSHOT_COLUMNS = [
    "snapshot_at",
    "source",
    "season",
    "player_key",
    "team_key",
    "element_type",
    "now_cost",
    "status",
    "chance_of_playing_next_round",
    "news",
    "news_added",
    "ep_next",
    "form",
    "penalties_order",
    "event_time",
    "available_at",
]
PARTS = ("p_start", "p_cs", "p_goal", "xp")
TABLE_PATH = Path(__file__).resolve().parents[1] / "src/fplopt/models/calibration_table.py"


def load_tables(data_dir: Path, first: int, last: int) -> dict[str, pd.DataFrame]:
    out = {}
    for name in TABLES:
        columns = SNAPSHOT_COLUMNS if name == "player_snapshot" else None
        frame = pd.read_parquet(data_dir / f"{name}.parquet", columns=columns)
        if "season" in frame:
            frame = frame[frame["season"].between(first, last)]
        out[name] = frame.reset_index(drop=True)
    return out


def parse_seasons(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out += list(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


# --- walk -----------------------------------------------------------------------------------


def realized(tables: dict[str, pd.DataFrame], season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per player-fixture outcomes of the season's played fixtures, per team-fixture goals
    against)."""
    matches = tables["player_match"]
    matches = matches[matches["season"] == season].reset_index(drop=True)
    positions = tables["player_season"][["player_key", "season", "element_type"]]
    positions = positions.drop_duplicates(["player_key", "season"], keep="last")
    matches = matches.merge(positions, on=["player_key", "season"], how="inner")
    stats = [
        "minutes",
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
    ]
    for column in stats:
        matches[column] = matches[column].fillna(0).astype("int64")
    points = score_matches(matches, backtest_rules(season))["points"]
    real = pd.to_numeric(matches["starts"], errors="coerce")
    start = real.where(real.notna(), infer_starts(matches).astype("float64"))
    outcomes = pd.DataFrame(
        {
            "player_key": matches["player_key"].astype("int64"),
            "fixture_key": matches["fixture_key"].astype("int64"),
            "r_minutes": matches["minutes"].astype("float64"),
            "r_start": start.astype("float64"),
            "r_goals": matches["goals_scored"].astype("float64"),
            "r_cs": matches["clean_sheets"].astype("float64"),
            "r_bonus": matches["bonus"].astype("float64"),
            "r_points": points.astype("float64"),
        }
    )
    outcomes = outcomes.drop_duplicates(["player_key", "fixture_key"], keep="last")
    teams = tables["team_match"]
    teams = teams[teams["season"] == season]
    against = pd.DataFrame(
        {
            "fixture_key": teams["fixture_key"].astype("int64"),
            "team_key": teams["team_key"].astype("int64"),
            "r_team_against": teams["goals_against"].astype("float64"),
        }
    ).dropna()
    return outcomes, against


def walk_season(job: tuple) -> dict:
    data_dir, season, out_dir, ban_residual = job
    tables = load_tables(Path(data_dir), 2016, season)
    store = DataStore(tables=tables)
    params = V1Params(minutes=MinutesParams(ban_residual=ban_residual), calibrate=False)
    outcomes, against = realized(tables, season)
    gameweeks = tables["gameweek"]
    gameweeks = gameweeks[gameweeks["season"] == season].sort_values("gw_index")
    fits: dict = {}
    rows, fit_s, predict_s = [], [], []
    for gw in gameweeks.itertuples(index=False):
        view = store.as_of(gw.deadline_time)
        cutoff = refit_deadline(view)
        if cutoff not in fits:
            t0 = time.perf_counter()
            fits[cutoff] = fit_v1(view.earlier(cutoff), params)
            fit_s.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        frame = fixture_components(view, fits[cutoff])
        predict_s.append(time.perf_counter() - t0)
        frame = frame.merge(against, on=["fixture_key", "team_key"], how="inner")
        frame = frame.merge(outcomes, on=["player_key", "fixture_key"], how="left")
        for column in ("r_minutes", "r_start", "r_goals", "r_cs", "r_bonus", "r_points"):
            frame[column] = frame[column].fillna(0.0)
        rows.append(frame.assign(deadline_gw_index=int(gw.gw_index)))
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(Path(out_dir) / f"{season}.parquet", index=False)
    print(f"season {season} done: {len(out)} rows", file=sys.stderr, flush=True)
    return {"season": season, "fit_s": fit_s, "predict_s": predict_s, "rows": len(out)}


def walk(args) -> None:
    seasons = parse_seasons(args.seasons)
    if any(s > LAST_FIT_SEASON for s in seasons):
        sys.exit("walk covers develop and validate only (never the 2025/26 holdout)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(str(args.data_dir), s, str(out), args.ban_residual) for s in seasons]
    params = V1Params(minutes=MinutesParams(ban_residual=args.ban_residual))
    (out / "params.txt").write_text(calibration_fingerprint(params), encoding="utf-8")
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(args.jobs, len(jobs))) as pool:
        results = list(pool.map(walk_season, sorted(jobs, key=lambda j: -j[1])))
    timing = {
        str(r["season"]): {
            "fit_s_median": float(np.median(r["fit_s"])),
            "fit_s_max": float(np.max(r["fit_s"])),
            "n_fits": len(r["fit_s"]),
            "predict_s_median": float(np.median(r["predict_s"])),
            "predict_s_max": float(np.max(r["predict_s"])),
            "rows": r["rows"],
        }
        for r in results
    }
    timing["wall_s"] = time.perf_counter() - start
    (out / "timing.json").write_text(json.dumps(timing, indent=1), encoding="utf-8")
    print(json.dumps(timing, indent=1))


# --- fit ------------------------------------------------------------------------------------


def load_walk(directory: Path, seasons: list[int]) -> dict[int, pd.DataFrame]:
    return {s: pd.read_parquet(directory / f"{s}.parquet") for s in seasons}


def horizon_groups(spec: str, split: bool) -> tuple[tuple[int, int | None], ...]:
    """Horizon groups of a part spec: `name` (pooled, or 0 / 1+ with `--split`), `name:0`
    (the target GW only) or `name:1+` (later GWs only; the target GW is left as is)."""
    suffix = spec.split(":", 1)[1] if ":" in spec else ""
    if suffix == "0":
        return ((0, 1),)
    if suffix == "1+":
        return ((1, None),)
    return ((0, 1), (1, None)) if split else ((0, None),)


def _maps(frame: pd.DataFrame, pred: np.ndarray, outcome: np.ndarray, groups: tuple) -> tuple:
    out = []
    horizon = frame["horizon"].to_numpy()
    for first, stop in groups:
        rows = (horizon >= first) & (horizon < (stop if stop is not None else 99))
        out.append((first, isotonic(pred[rows], outcome[rows])))
    return tuple(out)


def fit_calibration(
    window: pd.DataFrame, parts: tuple[str, ...], split: bool, seasons: tuple[int, ...]
) -> Calibration:
    """The maps of `parts` (specs, `horizon_groups`) fitted on `window` (walk rows), in the
    application order."""
    specs = {spec.split(":", 1)[0]: horizon_groups(spec, split) for spec in parts}
    calibration = Calibration(fitted_on=seasons)
    frame = window
    if "p_start" in specs:
        pred, outcome = frame["p_start"].to_numpy(), frame["r_start"].to_numpy()
        calibration = dataclasses.replace(
            calibration, p_start=_maps(frame, pred, outcome, specs["p_start"])
        )
        frame = calibrate_inputs(frame, calibration)
    if "p_cs" in specs:
        team = frame.drop_duplicates(["deadline_gw_index", "season", "fixture_key", "team_key"])
        cs = (team["r_team_against"].to_numpy() == 0).astype("float64")
        maps = _maps(team, team["p_cs"].to_numpy(), cs, specs["p_cs"])
        calibration = dataclasses.replace(calibration, p_cs=maps)
    if "p_goal" in specs:
        frame = calibrate_inputs(window, calibration)
        p_play = frame["p_play"].to_numpy()
        e_goals = frame["e_np_goals"].to_numpy() + frame["e_pen_goals"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(p_play > 0, e_goals / p_play, 0.0)
        p_goal = p_play * -np.expm1(-rate)
        outcome = (frame["r_goals"].to_numpy() >= 1).astype("float64")
        maps = _maps(frame, p_goal, outcome, specs["p_goal"])
        calibration = dataclasses.replace(calibration, p_goal=maps)
    if "xp" in specs:
        frame = calibrate_inputs(window, calibration)
        frame = fixture_points(frame, backtest_rules(int(seasons[-1])))
        linear = linear_by_position(
            frame["element_type"].to_numpy(),
            frame["p_play"].to_numpy(),
            frame["xp"].to_numpy(),
            frame["r_points"].to_numpy(),
        )
        calibration = dataclasses.replace(calibration, xp=linear)
    return calibration


def score(frame: pd.DataFrame, season: int, calibration: Calibration | None) -> dict:
    """Develop metrics of one season's walk rows under `calibration`."""
    rules = backtest_rules(season)
    out = calibrate_xp(fixture_points(calibrate_inputs(frame, calibration), rules), calibration)
    h0 = out["horizon"].to_numpy() == 0
    err = (out["xp"].to_numpy() - out["r_points"].to_numpy()) ** 2
    start = out["r_start"].to_numpy()
    p_start = out["p_start"].to_numpy()
    team = out.drop_duplicates(["deadline_gw_index", "fixture_key", "team_key"])
    cs = (team["r_team_against"].to_numpy() == 0).astype("float64")
    goal = (out["r_goals"].to_numpy() >= 1).astype("float64")
    e_goals = out["e_goals"].to_numpy()
    goals = out["r_goals"].to_numpy()
    rate = np.clip(e_goals, 1e-12, None)
    goals_ll = goals * np.log(rate) - rate - gammaln(goals + 1.0)
    p60 = out["p_60"].to_numpy()
    probs = np.column_stack([1 - out["p_play"].to_numpy(), out["p_play"].to_numpy() - p60, p60])
    probs = np.clip(probs, 0, 1)
    classes = minutes_class(out["r_minutes"].to_numpy())
    return {
        "season": season,
        "mse_h0": float(err[h0].mean()),
        "mse_h15": float(err[~h0].mean()),
        "mean_xp_h0": float(out["xp"].to_numpy()[h0].mean()),
        "mean_points_h0": float(out["r_points"].to_numpy()[h0].mean()),
        "brier_start_h0": float(((p_start - start) ** 2)[h0].mean()),
        "brier_start_h15": float(((p_start - start) ** 2)[~h0].mean()),
        "minutes_ll_h0": ordinal_log_loss(probs[h0], classes[h0]),
        "brier_cs": float(((team["p_cs"].to_numpy() - cs) ** 2).mean()),
        "brier_goal_h0": float(((out["p_goal"].to_numpy() - goal) ** 2)[h0].mean()),
        "goals_ll_h0": float(goals_ll[h0].mean()),
    }


def fit(args) -> None:
    directory = Path(args.walk)
    available = sorted(int(p.stem) for p in directory.glob("*.parquet"))
    frames = load_walk(directory, available)
    for season, frame in frames.items():
        frame["season"] = season
    first_fit = args.first_fit
    apply_seasons = [s for s in range(first_fit + 1, LAST_FIT_SEASON + 2)]

    def window(season: int) -> tuple[pd.DataFrame, tuple[int, ...]]:
        used = tuple(s for s in available if first_fit <= s < season)
        return pd.concat([frames[s] for s in used], ignore_index=True), used

    variants = {"none": (), **{f"+{p}": (p,) for p in PARTS}}
    for extra in args.variant or ():
        variants[f"+{extra}"] = tuple(extra.split(","))
    variants["chosen"] = tuple(args.parts.split(",")) if args.parts else PARTS
    scored = [s for s in available if s in DEVELOP]
    results = []
    calibrations: dict[str, dict[int, Calibration | None]] = {}
    for name, parts in variants.items():
        calibrations[name] = {}
        for season in scored:
            if not parts or season <= first_fit:
                cal = None
            else:
                rows, used = window(season)
                cal = fit_calibration(rows, parts, args.split, used)
            calibrations[name][season] = cal
            results.append({"variant": name} | score(frames[season], season, cal))
        print(f"variant {name} scored", file=sys.stderr, flush=True)
    table = pd.DataFrame(results)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    mean = table.groupby("variant", sort=False).mean(numeric_only=True).drop(columns="season")
    print(f"\nfirst fit season {first_fit}, split horizons {args.split}; develop seasons {scored}")
    print("mean over seasons:")
    print(mean.to_string())
    per_season = table.pivot(index="season", columns="variant", values="mse_h0")
    print("\nper season mse_h0:")
    print(per_season.to_string())
    report = {"first_fit": first_fit, "split": args.split, "rows": results}
    (directory / f"calibration-{first_fit}-{int(args.split)}.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8"
    )
    if args.write:
        parts = variants["chosen"]
        entries = []
        for season in apply_seasons:
            rows, used = window(season)
            entries.append((season, to_table(fit_calibration(rows, parts, args.split, used))))
        fingerprint = (directory / "params.txt").read_text(encoding="utf-8")
        write_table(entries, parts, first_fit, args.split, fingerprint)


def write_table(
    entries: list, parts: tuple[str, ...], first_fit: int, split: bool, fingerprint: str
) -> None:
    lines = [
        '"""Calibration parameters of `v1` (`fplopt.models.calibration`), generated by',
        "`uv run python dev/calibrate_v1.py fit --write`; do not edit by hand.",
        "",
        "One entry per season S: (S, parts), the maps fitted on the walk-forward predictions and",
        "outcomes of the seasons before S (`fitted_on`). Seasons before the first entry are",
        "uncalibrated; seasons after the last use the last entry.",
        "",
        "`CALIBRATION_PARAMS`: the `V1Params` the walk ran with",
        "(`fplopt.models.assemble.calibration_fingerprint`); v1 refuses the table with others.",
        "",
        f"Parts: {', '.join(parts)}; fitted from season {first_fit}; isotonic maps per horizon",
        f"group: {'0 and 1+' if split else 'all horizons pooled'}.",
        '"""',
        "",
        "# fmt: off",
        "CALIBRATION_PARAMS = (",
        *(f"    {fingerprint[i : i + 88]!r}" for i in range(0, len(fingerprint), 88)),
        ")",
        "",
        "CALIBRATION_TABLE: tuple = (",
    ]
    for season, entry in entries:
        text = pprint.pformat((season, entry), width=92, compact=True)
        lines += [f"    {line}" for line in text.splitlines()]
        lines[-1] += ","
    lines += [")", "# fmt: on", ""]
    TABLE_PATH.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"wrote {TABLE_PATH}")


# --- ban residual ---------------------------------------------------------------------------


def ban(args) -> None:
    seasons = [s for s in DEVELOP]
    off = load_walk(Path(args.off), seasons)
    on = load_walk(Path(args.on), seasons)
    rows = []
    for season in seasons:
        a, b = off[season], on[season]
        keys = ["deadline_gw_index", "player_key", "fixture_key"]
        merged = a.merge(b[[*keys, "p_start", "p_play", "p_60"]], on=keys, suffixes=("", "_on"))
        banned = (merged["p_start"].to_numpy() == 0) & (merged["p_start_on"].to_numpy() > 0)
        start = merged["r_start"].to_numpy()
        classes = minutes_class(merged["r_minutes"].to_numpy())
        row = {"season": season, "n_banned": int(banned.sum())}
        for tag, suffix in (("off", ""), ("on", "_on")):
            p = merged[f"p_start{suffix}"].to_numpy()
            p60 = merged[f"p_60{suffix}"].to_numpy()
            play = merged[f"p_play{suffix}"].to_numpy()
            probs = np.clip(np.column_stack([1 - play, play - p60, p60]), 0, 1)
            row[f"brier_start_{tag}"] = float(((p - start) ** 2).mean())
            row[f"brier_start_banned_{tag}"] = float(((p - start) ** 2)[banned].mean())
            row[f"minutes_ll_{tag}"] = ordinal_log_loss(probs, classes)
            row[f"minutes_ll_banned_{tag}"] = ordinal_log_loss(probs[banned], classes[banned])
        row["start_rate_banned"] = float(start[banned].mean())
        for tag, frame in (("off", a), ("on", b)):
            pts = fixture_points(frame, backtest_rules(season))
            err = (pts["xp"] - pts["r_points"]) ** 2
            row[f"mse_h0_{tag}"] = float(err[pts["horizon"] == 0].mean())
            row[f"mse_h15_{tag}"] = float(err[pts["horizon"] > 0].mean())
        rows.append(row)
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(table.to_string(index=False))
    print("\nmean over seasons:")
    print(table.mean(numeric_only=True).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", default=os.environ.get("FPLOPT_DATA_DIR", "data"))
    sub = parser.add_subparsers(dest="command", required=True)
    w = sub.add_parser("walk")
    w.add_argument("--seasons", default="2016-2024")
    w.add_argument("--jobs", type=int, default=6)
    w.add_argument("--out", required=True)
    w.add_argument("--ban-residual", action="store_true")
    f = sub.add_parser("fit")
    f.add_argument("--walk", required=True)
    f.add_argument("--first-fit", type=int, default=2016)
    f.add_argument("--split", action="store_true", help="isotonic maps per horizon 0 / 1+")
    f.add_argument(
        "--parts",
        default=None,
        help=f"comma-separated specs of {PARTS} (`name`, `name:0`, `name:1+`): the chosen set",
    )
    f.add_argument("--variant", action="append", help="a further spec set to score")
    f.add_argument("--write", action="store_true")
    b = sub.add_parser("ban")
    b.add_argument("--off", required=True)
    b.add_argument("--on", required=True)
    args = parser.parse_args()
    args.data_dir = Path(args.data_dir)
    {"walk": walk, "fit": fit, "ban": ban}[args.command](args)


if __name__ == "__main__":
    main()
