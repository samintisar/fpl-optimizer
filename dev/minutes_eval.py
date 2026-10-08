"""Minutes model measurements (Phase 5 plan, Task 3; PLAN §6.3). Dev only: never imported
by `fplopt`.

Subcommands:
- `inference`: accuracy of the inferred starters (`fplopt.features.history.infer_starts`)
  where FPL's real `starts` exist (2022/23 GW16 – 2024/25). This measures the proxy, not a
  model, so these validate seasons may be read.
- `walk`: develop-only walk-forward metrics. For every GW deadline of the chosen seasons
  (2016/17–2022/23 only; anything later is refused) the minutes model is fitted at the
  refit cutoff (`fplopt.models.fitted.refit_deadline`, memoized per cutoff), predicted at
  the deadline, adjusted with the availability layer, and scored against the outcome of
  each (player, fixture) with a `player_match` row. Metrics per horizon: 3-class (0 / 1–59 /
  60+) log loss and RPS, Brier and reliability of P(start), for the model without flags
  ("raw"), with flags ("flags") and the last-5 reference (smoothed class frequencies over
  the player's last 5 rows before the deadline's GW, α = 1 pseudo-row at the previous
  seasons' class frequencies). Also fit time per cutoff and predict time per deadline.
  Variants: a JSON list of `MinutesParams` overrides (`--grid`), each × `--horizon-decays`
  (predict-only, no refit).

The store is in memory (`DataStore(tables=...)`) with only the tables the model reads,
restricted to seasons ≤ the last requested season, so no validate or holdout row is even
loaded for `walk`.

    uv run python dev/minutes_eval.py --data-dir <data> inference
    uv run python dev/minutes_eval.py --data-dir <data> walk --seasons 2017-2022 \
        [--grid grid.json] [--horizon-decays 1,0.85] [--jobs 6] [--out results/minutes-x]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from fplopt.features.history import infer_starts
from fplopt.features.store import DataStore
from fplopt.models.availability import adjust_minutes, fit_availability
from fplopt.models.fitted import refit_deadline
from fplopt.models.gbm import GbmParams
from fplopt.models.minutes import MinutesParams, fit_minutes, predict_minutes

DEVELOP = range(2016, 2023)
CLASSES = ("p0", "p1", "p60")
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
    "event_time",
    "available_at",
]
TABLES = (
    "gameweek",
    "schedule",
    "fixture_snapshot",
    "player_match",
    "player_gw",
    "player_season",
    "player_snapshot",
)
RELIABILITY_BINS = np.linspace(0, 1, 11)


def load_tables(data_dir: Path, last_season: int) -> dict[str, pd.DataFrame]:
    out = {}
    for name in TABLES:
        columns = SNAPSHOT_COLUMNS if name == "player_snapshot" else None
        frame = pd.read_parquet(data_dir / f"{name}.parquet", columns=columns)
        if "season" in frame:
            frame = frame[frame["season"] <= last_season]
        out[name] = frame.reset_index(drop=True)
    return out


# --- start inference -----------------------------------------------------------------------


def inference(data_dir: Path) -> dict:
    columns = ["player_key", "season", "fixture_key", "gw", "team_key", "minutes", "starts"]
    matches = pd.read_parquet(data_dir / "player_match.parquet", columns=columns)
    matches = matches[matches["season"].between(2022, 2024) & matches["starts"].notna()]
    matches = matches.reset_index(drop=True)
    inferred = infer_starts(matches).to_numpy(dtype=bool)
    real = (matches["starts"] == 1).to_numpy(dtype=bool)
    wrong = inferred != real
    teams = matches[["fixture_key", "team_key"]].drop_duplicates()
    tie = matches[wrong & (matches["minutes"] == 45).to_numpy()]
    return {
        "rows": len(matches),
        "first_gw_2022": int(matches.loc[matches["season"] == 2022, "gw"].min()),
        "accuracy": float((~wrong).mean()),
        "false_positive_rate": float((inferred & ~real).sum() / (~real).sum()),
        "false_negative_rate": float((~inferred & real).sum() / real.sum()),
        "wrong_rows": int(wrong.sum()),
        "team_fixtures": len(teams),
        "team_fixtures_with_error": int(
            matches[wrong].groupby(["fixture_key", "team_key"]).ngroups
        ),
        "wrong_rows_at_45_minutes": len(tie),
    }


# --- outcomes and the reference -------------------------------------------------------------


def outcomes(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    matches = tables["player_match"]
    real = pd.to_numeric(matches["starts"], errors="coerce").astype("float64")
    start = real.where(real.notna(), infer_starts(matches).astype("float64"))
    minutes = matches["minutes"].to_numpy(dtype="float64")
    return pd.DataFrame(
        {
            "player_key": matches["player_key"].to_numpy(),
            "fixture_key": matches["fixture_key"].to_numpy(),
            "minutes": minutes,
            "y_start": start.to_numpy(dtype="float64"),
            "y_class": np.select([minutes <= 0, minutes < 60], [0, 1], 2),
        }
    )


def reference_counts(tables: dict[str, pd.DataFrame], result: pd.DataFrame) -> pd.DataFrame:
    """Per player row in order: cumulative counts of the 3 classes and starts, with an
    ordinal (season, gw_index) key, for the last-5 reference."""
    gameweeks = tables["gameweek"][["season", "gw", "gw_index"]]
    matches = tables["player_match"][["player_key", "season", "gw", "kickoff_time", "fixture_key"]]
    matches = matches.merge(gameweeks, on=["season", "gw"], how="inner")
    matches = matches.merge(result, on=["player_key", "fixture_key"], how="inner")
    matches = matches.sort_values(
        ["player_key", "season", "gw_index", "kickoff_time", "fixture_key"], kind="mergesort"
    ).reset_index(drop=True)
    matches["order"] = matches["season"] * 100 + matches["gw_index"]
    matches["idx"] = matches.groupby("player_key").cumcount()
    for k in range(3):
        matches[f"c{k}"] = (matches["y_class"] == k).astype("float64")
        matches[f"c{k}"] = matches.groupby("player_key")[f"c{k}"].cumsum()
    matches["cs"] = matches.groupby("player_key")["y_start"].cumsum()
    return matches[["player_key", "season", "order", "idx", "c0", "c1", "c2", "cs"]]


def reference(
    counts: pd.DataFrame, players: pd.DataFrame, season: int, gw_index: int
) -> pd.DataFrame:
    """Last-5 smoothed class frequencies and start rate per player as of (season, gw_index)."""
    order = season * 100 + gw_index
    prior_rows = counts[counts["season"] < season]
    last = prior_rows.groupby("player_key").tail(1)
    totals = last[["c0", "c1", "c2", "cs"]].sum()
    n_prior = totals[["c0", "c1", "c2"]].sum()
    if n_prior > 0:
        pi = (totals[["c0", "c1", "c2"]] / n_prior).to_numpy()
        pi_start = float(totals["cs"] / n_prior)
    else:
        pi, pi_start = np.array([0.55, 0.15, 0.30]), 0.36
    before = counts[counts["order"] < order]
    end = before.groupby("player_key").tail(1).set_index("player_key")
    out = players[["player_key"]].drop_duplicates().set_index("player_key")
    out = out.join(end[["idx", "c0", "c1", "c2", "cs"]])
    back = counts.set_index(["player_key", "idx"])[["c0", "c1", "c2", "cs"]]
    keys = pd.MultiIndex.from_arrays([out.index, out["idx"] - 5])
    start = back.reindex(keys).fillna(0.0).to_numpy()
    n = np.minimum(out["idx"].fillna(-1).to_numpy() + 1, 5)
    window = out[["c0", "c1", "c2", "cs"]].fillna(0.0).to_numpy() - start
    window[n <= 0] = 0.0
    n = np.maximum(n, 0)
    alpha = 1.0
    probs = (window[:, :3] + alpha * pi) / (n[:, None] + alpha)
    p_start = (window[:, 3] + alpha * pi_start) / (n + alpha)
    return pd.DataFrame(
        {
            "player_key": out.index.to_numpy(),
            "r0": probs[:, 0],
            "r1": probs[:, 1],
            "r60": probs[:, 2],
            "r_start": p_start,
        }
    )


# --- metrics --------------------------------------------------------------------------------


def score(p: np.ndarray, y_class: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per row 3-class log loss and RPS of probabilities p (n x 3) for classes y."""
    p = np.clip(p, 1e-6, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    log_loss = -np.log(p[np.arange(len(p)), y_class])
    observed = np.zeros_like(p)
    observed[np.arange(len(p)), y_class] = 1.0
    cumulative = np.cumsum(p, axis=1)[:, :2] - np.cumsum(observed, axis=1)[:, :2]
    rps = (cumulative**2).sum(axis=1) / 2.0
    return log_loss, rps


def model_classes(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    p_play = frame[f"{prefix}p_play"].to_numpy()
    p_60 = frame[f"{prefix}p_60"].to_numpy()
    return np.column_stack([1.0 - p_play, p_play - p_60, p_60])


def row_scores(frame: pd.DataFrame) -> pd.DataFrame:
    y = frame["y_class"].to_numpy(dtype="int64")
    s = frame["y_start"].to_numpy(dtype="float64")
    out = frame[["season", "deadline_index", "horizon", "variant"]].copy()
    for name, probs, p_start in (
        ("raw", model_classes(frame, "raw_"), frame["raw_p_start"].to_numpy()),
        ("flags", model_classes(frame, "adj_"), frame["adj_p_start"].to_numpy()),
        ("ref", frame[["r0", "r1", "r60"]].to_numpy(), frame["r_start"].to_numpy()),
    ):
        log_loss, rps = score(probs, y)
        out[f"{name}_logloss"] = log_loss
        out[f"{name}_rps"] = rps
        out[f"{name}_brier"] = (p_start - s) ** 2
        out[f"{name}_pstart"] = p_start
    out["y_start"] = s
    return out


# --- the walk -------------------------------------------------------------------------------


def walk_season(job: tuple) -> dict:
    data_dir, season, variants, horizon_decays, flags_from = job
    tables = load_tables(Path(data_dir), season)
    store = DataStore(tables=tables)
    result = outcomes(tables)
    counts = reference_counts(tables, result)
    gameweeks = tables["gameweek"]
    gameweeks = gameweeks[gameweeks["season"] == season].sort_values("gw_index")
    rows, timings = [], []
    for name, params in variants:
        fits: dict = {}
        for gw in gameweeks.itertuples(index=False):
            view = store.as_of(gw.deadline_time)
            cutoff = refit_deadline(view)
            if cutoff not in fits:
                t0 = time.perf_counter()
                cut_view = view.earlier(cutoff)
                fit = fit_minutes(cut_view, params)
                t1 = time.perf_counter()
                afit = fit_availability(cut_view, fit)
                t2 = time.perf_counter()
                fits[cutoff] = (fit, afit)
                timings.append(
                    {
                        "variant": name,
                        "season": season,
                        "cutoff_gw_index": int(gw.gw_index),
                        "fit_minutes_s": t1 - t0,
                        "fit_availability_s": t2 - t1,
                        "n_rows": fit.n_rows,
                        "n_flag_rows": afit.n_rows,
                    }
                )
            fit, afit = fits[cutoff]
            for decay in horizon_decays:
                variant_fit = dataclasses.replace(
                    fit, params=dataclasses.replace(fit.params, horizon_decay=decay)
                )
                t0 = time.perf_counter()
                raw = predict_minutes(view, variant_fit)
                t1 = time.perf_counter()
                use_flags = (season, int(gw.gw_index)) >= flags_from
                adjusted = adjust_minutes(view, raw, afit) if use_flags else raw
                t2 = time.perf_counter()
                timings.append(
                    {
                        "variant": f"{name}|hd={decay}",
                        "season": season,
                        "deadline_gw_index": int(gw.gw_index),
                        "predict_s": t1 - t0,
                        "adjust_s": t2 - t1,
                    }
                )
                frame = raw.merge(
                    adjusted[["player_key", "fixture_key", "p_start", "p_60", "p_play"]],
                    on=["player_key", "fixture_key"],
                    suffixes=("", "_adj"),
                )
                frame = frame.rename(
                    columns={
                        "p_start": "raw_p_start",
                        "p_60": "raw_p_60",
                        "p_play": "raw_p_play",
                        "p_start_adj": "adj_p_start",
                        "p_60_adj": "adj_p_60",
                        "p_play_adj": "adj_p_play",
                    }
                )
                frame = frame.merge(result, on=["player_key", "fixture_key"], how="inner")
                ref = reference(counts, frame, season, int(gw.gw_index))
                frame = frame.merge(ref, on="player_key", how="left")
                frame = frame.assign(deadline_index=int(gw.gw_index), variant=f"{name}|hd={decay}")
                rows.append(row_scores(frame))
        print(f"season {season} variant {name} done", file=sys.stderr, flush=True)
    return {
        "scores": pd.concat(rows, ignore_index=True),
        "timings": pd.DataFrame(timings),
    }


def summarize(scores: pd.DataFrame, flag_seasons: tuple[int, ...]) -> dict:
    scores = scores.assign(hgroup=np.where(scores["horizon"] == 0, "h0", "h1-5"))
    metric_columns = [c for c in scores.columns if c.endswith(("_logloss", "_rps", "_brier"))]
    per_season = scores.groupby(["variant", "hgroup", "season"])[metric_columns].mean()
    by_season_mean = per_season.groupby(["variant", "hgroup"]).mean()
    flagged = scores[scores["season"].isin(flag_seasons)]
    flag_means = (
        flagged.groupby(["variant", "hgroup", "season"])[metric_columns]
        .mean()
        .groupby(["variant", "hgroup"])
        .mean()
    )
    reliability = {}
    for (variant, hgroup), group in scores.groupby(["variant", "hgroup"]):
        table = {}
        for name in ("raw", "flags", "ref"):
            bins = pd.cut(group[f"{name}_pstart"], RELIABILITY_BINS, include_lowest=True)
            agg = group.groupby(bins, observed=True).agg(
                n=("y_start", "size"),
                predicted=(f"{name}_pstart", "mean"),
                observed=("y_start", "mean"),
            )
            table[name] = [
                {"bin": str(b), **{k: float(v) for k, v in row.items()}}
                for b, row in agg.iterrows()
            ]
        reliability[f"{variant}|{hgroup}"] = table
    return {
        "mean_over_seasons": json.loads(by_season_mean.reset_index().to_json(orient="records")),
        "per_season": json.loads(per_season.reset_index().to_json(orient="records")),
        "flag_seasons_mean": json.loads(flag_means.reset_index().to_json(orient="records")),
        "reliability": reliability,
    }


def parse_seasons(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out += list(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def variants_from(path: str | None) -> list[tuple[str, MinutesParams]]:
    if path is None:
        return [("default", MinutesParams())]
    out = []
    for spec in json.loads(Path(path).read_text()):
        name = spec.pop("name")
        gbm = {k: GbmParams(**spec.pop(k)) for k in ("start", "sixty", "sub") if k in spec}
        if "all" in spec:
            shared = GbmParams(**spec.pop("all"))
            gbm = {k: gbm.get(k, shared) for k in ("start", "sixty", "sub")}
        out.append((name, MinutesParams(**gbm, **spec)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", default=os.environ.get("FPLOPT_DATA_DIR", "data"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inference")
    walk = sub.add_parser("walk")
    walk.add_argument("--seasons", default="2017-2022")
    walk.add_argument("--grid")
    walk.add_argument("--horizon-decays", default="0.85")
    walk.add_argument("--jobs", type=int, default=6)
    walk.add_argument("--out")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    if args.command == "inference":
        print(json.dumps(inference(data_dir), indent=2))
        return
    seasons = parse_seasons(args.seasons)
    if any(s not in DEVELOP for s in seasons):
        sys.exit("walk is develop-only (2016-2022)")
    variants = variants_from(args.grid)
    decays = [float(x) for x in args.horizon_decays.split(",")]
    flags_from = (2021, 1)
    jobs = [(str(data_dir), s, variants, decays, flags_from) for s in seasons]
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(args.jobs, len(jobs))) as pool:
        results = list(pool.map(walk_season, jobs))
    scores = pd.concat([r["scores"] for r in results], ignore_index=True)
    timings = pd.concat([r["timings"] for r in results], ignore_index=True)
    summary = summarize(scores, flag_seasons=(2021, 2022))
    fits = timings.dropna(subset=["fit_minutes_s"]) if "fit_minutes_s" in timings else timings
    predicts = timings.dropna(subset=["predict_s"]) if "predict_s" in timings else timings
    summary["timing"] = {
        "fit_minutes_s": fits["fit_minutes_s"].describe().to_dict(),
        "fit_availability_s": fits["fit_availability_s"].describe().to_dict(),
        "predict_s": predicts["predict_s"].describe().to_dict(),
        "adjust_s": predicts["adjust_s"].describe().to_dict(),
        "wall_s": time.perf_counter() - start,
    }
    summary["seasons"] = seasons
    summary["variants"] = [{"name": n, "params": dataclasses.asdict(p)} for n, p in variants]
    text = json.dumps(summary, indent=1, default=str)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(text)
        timings.to_csv(out / "timings.csv", index=False)
    table = pd.DataFrame(summary["mean_over_seasons"])
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(table.to_string(index=False))
    print(pd.DataFrame(summary["flag_seasons_mean"]).to_string(index=False))
    print(json.dumps(summary["timing"], indent=1, default=str))


if __name__ == "__main__":
    main()
