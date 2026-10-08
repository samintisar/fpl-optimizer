"""Player goal/assist shares and penalties: measurements (Phase 5 plan, Task 4; PLAN §6.2,
§3). Dev only: never imported by `fplopt`.

Subcommands:
- `npxg-check`: the PLAN §3 VERIFY on the source substitution. On the 2022/23 GW16–2024/25
  rows where both Understat (`us_npxg`, `us_xa`) and FPL/Opta (`fpl_xg`, `fpl_xa`) exist
  (a data-source check, not model tuning, so these validate seasons may be read), compare
  Understat npxG with FPL xG − penalty xG × attempts: the brief's 0.76 per attempt and
  Opta's 0.79 with the true (Understat) attempts, the inferred attempts of
  `fplopt.models.shares.match_xg` (π fitted in-sample here, and π from 2022/23 only for
  2023/24–2024/25), and no penalty correction (misses only). Per match: correlation, mean
  absolute error, bias. Per player-season (≥ 450 minutes): correlation of totals and of
  per-90 rates, ratio Σ FPL / Σ Understat, the regression slope, and the ratio among
  takers (≥ 3 attempts). Also the xG of a penalty in FPL (rows whose only shot was a
  penalty) and the xA comparison. No 2025/26 row is loaded.
- `walk`: develop walk-forward (deadlines of 2017/18–2022/23 only; later seasons are
  refused). At every GW deadline: the team model (`fit_team` / `team_lambdas`), the minutes
  model with the availability layer (`fit_minutes` / `predict_minutes` / `adjust_minutes`,
  flags from 2021/22) and the shares (`fit_shares` / `predict_shares`), each fitted at the
  refit cutoff (`refit_deadline`, memoized per cutoff); every (player, fixture) prediction
  with a `player_match` row is scored against his realized goals and FPL assists:
  Poisson log-likelihood of goals and of assists, Brier of P(goal ≥ 1) = 1 − exp(−e_goals)
  and its reliability, at horizon 0 and pooled over horizons 1–5, against the reference
  (position-average goals / assists per 90 over the cutoff's last 3 seasons × e_minutes /
  90 × λ_for / the league's mean goals per team-match). Predict-time `SharesParams`
  variants (`--grid`, a JSON list of overrides; default: the tuning grid below) share the
  fits. Fit time per cutoff and predict time per deadline are reported.

The store is in memory with only the tables the models read, restricted to seasons ≤ the
season walked (and, for `npxg-check`, to 2022/23–2024/25).

    uv run python dev/shares_eval.py --data-dir <data> npxg-check
    uv run python dev/shares_eval.py --data-dir <data> walk [--seasons 2017-2022] \
        [--grid g.json] [--jobs 6] [--out results/<name>]
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fplopt.evaluate.metrics import brier_decomposition, reliability_table
from fplopt.features.baseline import player_pool
from fplopt.features.store import DataStore
from fplopt.models import shares
from fplopt.models.availability import adjust_minutes, fit_availability
from fplopt.models.fitted import refit_deadline
from fplopt.models.minutes import fit_minutes, predict_minutes
from fplopt.models.team import fit_team, team_lambdas

DEVELOP = range(2017, 2023)  # deadlines scored (2016/17 is burn-in)
OVERLAP = (2022, 2023, 2024)
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
FLAGS_FROM = 2021  # availability layer from 2021/22 (as dev/minutes_eval.py)
REF_SEASONS = 3


def load_tables(data_dir: Path, first: int, last: int, names=TABLES) -> dict[str, pd.DataFrame]:
    out = {}
    for name in names:
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


# --- npxG source check ----------------------------------------------------------------------


def _per_match(us: np.ndarray, est: np.ndarray) -> dict:
    return {
        "match_corr": float(np.corrcoef(us, est)[0, 1]),
        "match_mae": float(np.mean(np.abs(est - us))),
        "match_bias": float(np.mean(est - us)),
    }


def _per_player_season(rows: pd.DataFrame, us: str, est: str) -> dict:
    ps = rows.groupby(["player_key", "season"]).agg(
        us=(us, "sum"), est=(est, "sum"), minutes=("minutes", "sum"), att=("att", "sum")
    )
    ps = ps[ps["minutes"] >= 450]
    takers = ps[ps["att"] >= 3]
    per90_us = ps["us"] / ps["minutes"] * 90
    per90_est = ps["est"] / ps["minutes"] * 90
    return {
        "player_seasons": len(ps),
        "season_corr": float(np.corrcoef(ps["us"], ps["est"])[0, 1]),
        "per90_corr": float(np.corrcoef(per90_us, per90_est)[0, 1]),
        "ratio": float(ps["est"].sum() / ps["us"].sum()),
        "slope": float(np.polyfit(ps["us"], ps["est"], 1)[0]),
        "takers": len(takers),
        "takers_ratio": float(takers["est"].sum() / takers["us"].sum()) if len(takers) else None,
    }


def npxg_check(data_dir: Path) -> dict:
    tables = load_tables(data_dir, OVERLAP[0], OVERLAP[-1], ("player_match", "player_snapshot"))
    tables["player_snapshot"] = tables["player_snapshot"][
        ["snapshot_at", "source", "season", "player_key", "team_key", "penalties_order"]
        + ["event_time", "available_at"]
    ]
    store = DataStore(tables=tables)
    view = store.as_of("2025-07-01T00:00Z")  # after 2024/25; no 2025/26 row is loaded
    columns = [*shares.MATCH_COLUMNS, "us_xg"]
    matches = view.table("player_match", columns=columns)
    matches = matches.assign(**{c: shares._floats(matches, c) for c in columns[6:]})
    rows = matches[
        matches["us_npxg"].notna() & matches["fpl_xg"].notna() & (matches["minutes"] > 0)
    ].reset_index(drop=True)
    rows["main_taker"] = (
        shares._main_taker_at_kickoff(rows, shares._main_takers(view))
        == rows["player_key"].to_numpy()
    )
    rows["pen_us"] = rows["us_goals"] - rows["us_npg"]
    rows["att"] = rows["pen_us"] + rows["penalties_missed"]
    report: dict = {
        "rows": len(rows),
        "first": str(rows["kickoff_time"].min()),
        "last": str(rows["kickoff_time"].max()),
        "attempts": float(rows["att"].sum()),
        "understat_xg_per_attempt": float(
            (rows["us_xg"] - rows["us_npxg"]).sum() / rows["att"].sum()
        ),
    }
    only_pen = rows[(rows["att"] == 1) & (rows["us_npxg"] == 0)]
    report["fpl_xg_penalty_only_rows"] = {
        "n": len(only_pen),
        "mean": float(only_pen["fpl_xg"].mean()),
        "std": float(only_pen["fpl_xg"].std()),
    }

    candidates = shares._candidates(rows)
    main = rows["main_taker"].to_numpy(dtype=bool)
    pen_us = rows["pen_us"].to_numpy()

    def pi_from(mask: np.ndarray) -> tuple[float, float]:
        out = []
        for group in (main, ~main):
            m = mask & group & (candidates > 0)
            out.append(float(pen_us[m].sum() / candidates[m].sum()))
        return out[0], out[1]

    pi_all = pi_from(np.ones(len(rows), dtype=bool))
    pi_2022 = pi_from((rows["season"] == 2022).to_numpy())
    report["pi_in_sample"] = pi_all
    report["pi_2022_only"] = pi_2022
    report["main_taker_share_of_penalty_goals"] = float(pen_us[main].sum() / pen_us.sum())
    inferred = rows.assign(us_npxg=np.nan, us_xa=np.nan, us_goals=np.nan, us_npg=np.nan)
    variants = {
        "brief_0.76_true_attempts": rows["fpl_xg"] - 0.76 * rows["att"],
        "opta_0.79_true_attempts": rows["fpl_xg"] - 0.79 * rows["att"],
        "inferred_attempts_pi_in_sample": shares.match_xg(inferred, 1.0, 1.0, pi_all)["npxg"],
        "misses_only": rows["fpl_xg"] - 0.79 * rows["penalties_missed"],
    }
    estimates = {}
    for name, est in variants.items():
        rows["est"] = np.clip(est.to_numpy(dtype="float64"), 0, None)
        estimates[name] = {
            **_per_match(rows["us_npxg"].to_numpy(), rows["est"].to_numpy()),
            **_per_player_season(rows, "us_npxg", "est"),
        }
    later = rows[rows["season"] > 2022].copy()
    later_inferred = later.assign(us_npxg=np.nan, us_xa=np.nan, us_goals=np.nan, us_npg=np.nan)
    later["est"] = shares.match_xg(later_inferred, 1.0, 1.0, pi_2022)["npxg"].to_numpy()
    estimates["inferred_attempts_pi_2022_on_2023_2024"] = {
        **_per_match(later["us_npxg"].to_numpy(), later["est"].to_numpy()),
        **_per_player_season(later, "us_npxg", "est"),
    }
    report["npxg"] = estimates
    xa = rows[rows["us_xa"].notna() & rows["fpl_xa"].notna()].copy()
    xa["est"] = xa["fpl_xa"]
    report["xa"] = {
        **_per_match(xa["us_xa"].to_numpy(), xa["est"].to_numpy()),
        **_per_player_season(xa, "us_xa", "est"),
    }
    # The recommended scale factors (what fit_shares estimates walk-forward).
    report["k_goals"] = 1 / estimates["opta_0.79_true_attempts"]["ratio"]
    report["k_assists"] = 1 / report["xa"]["ratio"]
    return report


# --- the walk -------------------------------------------------------------------------------


def default_grid() -> list[dict]:
    grid = []
    for prior, current, club in itertools.product(
        (240.0, 480.0, 960.0), (2.0, 3.0, 5.0), (1.0, 0.5, 0.25)
    ):
        grid.append(
            {
                "prior_minutes": prior,
                "season_weights": [current, 2.0, 1.0, 1.0],
                "club_change_weight": club,
            }
        )
    return grid


def variant_label(overrides: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in overrides.items()) or "default"


def reference_inputs(view) -> dict:
    """Per position goals / assists per 90 over the view's last REF_SEASONS seasons, and the
    league's mean goals per team-match over them."""
    season, _ = view.gameweek_for_deadline()
    matches = shares._matches(view)
    matches = matches[(matches["season"] > season - REF_SEASONS) & (matches["minutes"] > 0)]
    per_position = matches.groupby("element_type")[["goals_scored", "assists", "minutes"]].sum()
    rates = {
        int(position): (
            float(row["goals_scored"] / row["minutes"] * 90),
            float(row["assists"] / row["minutes"] * 90),
        )
        for position, row in per_position.iterrows()
    }
    teams = view.table("team_match", columns=["season", "goals_for"])
    teams = teams[teams["season"] > season - REF_SEASONS]
    mean_goals = float(teams["goals_for"].mean()) if len(teams) else 1.35
    return {"rates": rates, "mean_goals": mean_goals}


def walk_season(job: tuple) -> dict:
    data_dir, season, grid = job
    tables = load_tables(Path(data_dir), 2016, season)
    store = DataStore(tables=tables)
    matches = tables["player_match"][["player_key", "fixture_key", "goals_scored", "assists"]]
    gameweeks = tables["gameweek"]
    gameweeks = gameweeks[gameweeks["season"] == season].sort_values("gw_index")
    fits: dict = {}
    rows, timings = [], []
    for gw in gameweeks.itertuples(index=False):
        view = store.as_of(gw.deadline_time)
        cutoff = refit_deadline(view)
        if cutoff not in fits:
            cut = view.earlier(cutoff)
            t0 = time.perf_counter()
            team_fit = fit_team(cut)
            t1 = time.perf_counter()
            minutes_fit = fit_minutes(cut)
            flags_fit = fit_availability(cut, minutes_fit)
            t2 = time.perf_counter()
            shares_fit = shares.fit_shares(cut)
            t3 = time.perf_counter()
            fits[cutoff] = (team_fit, minutes_fit, flags_fit, shares_fit, reference_inputs(cut))
            timings.append(
                {
                    "season": season,
                    "cutoff_gw_index": int(gw.gw_index),
                    "fit_team_s": t1 - t0,
                    "fit_minutes_s": t2 - t1,
                    "fit_shares_s": t3 - t2,
                }
            )
        team_fit, minutes_fit, flags_fit, shares_fit, ref = fits[cutoff]
        team = team_lambdas(view, team_fit)
        minutes = predict_minutes(view, minutes_fit)
        if season >= FLAGS_FROM:
            minutes = adjust_minutes(view, minutes, flags_fit)
        pool = player_pool(view)[["player_key", "element_type"]]
        base = minutes[["player_key", "fixture_key", "team_key", "horizon", "e_minutes"]]
        base = base.merge(pool, on="player_key", how="left")
        base = base.merge(
            team[["fixture_key", "team_key", "lambda_for"]], on=["fixture_key", "team_key"]
        )
        rates = ref["rates"]
        positions = base["element_type"].to_numpy(dtype="int64")
        goal_rate = np.array([rates.get(int(e), (0.0, 0.0))[0] for e in positions])
        assist_rate = np.array([rates.get(int(e), (0.0, 0.0))[1] for e in positions])
        scale = base["e_minutes"] / 90 * base["lambda_for"] / ref["mean_goals"]
        base = base.assign(ref_goals=goal_rate * scale, ref_assists=assist_rate * scale)
        base = base.merge(matches, on=["player_key", "fixture_key"], how="inner")
        for overrides in grid:
            params = dataclasses.replace(
                shares_fit.params,
                **{k: tuple(v) if isinstance(v, list) else v for k, v in overrides.items()},
            )
            fit = dataclasses.replace(shares_fit, params=params)
            t0 = time.perf_counter()
            out = shares.predict_shares(view, fit, minutes, team)
            t1 = time.perf_counter()
            timings.append(
                {"season": season, "deadline_gw_index": int(gw.gw_index), "predict_s": t1 - t0}
            )
            scored = base.merge(
                out[["player_key", "fixture_key", "e_goals", "e_assists", "e_pen_goals"]],
                on=["player_key", "fixture_key"],
                how="inner",
            )
            rows.append(
                scored.assign(
                    season=season,
                    deadline_gw_index=int(gw.gw_index),
                    variant=variant_label(overrides),
                )[
                    [
                        "season",
                        "deadline_gw_index",
                        "variant",
                        "horizon",
                        "goals_scored",
                        "assists",
                        "e_goals",
                        "e_assists",
                        "e_pen_goals",
                        "ref_goals",
                        "ref_assists",
                    ]
                ]
            )
    print(f"season {season} done", file=sys.stderr, flush=True)
    coverage = fits[max(fits)][3].coverage if fits else ()
    return {
        "rows": pd.concat(rows, ignore_index=True),
        "timings": pd.DataFrame(timings),
        "coverage": coverage,
    }


def _poisson_ll(rate: np.ndarray, count: np.ndarray) -> np.ndarray:
    from scipy.special import gammaln

    rate = np.maximum(rate, 1e-12)
    return count * np.log(rate) - rate - gammaln(count + 1)


def score_rows(rows: pd.DataFrame) -> pd.DataFrame:
    goals = rows["goals_scored"].to_numpy(dtype="float64")
    assists = rows["assists"].to_numpy(dtype="float64")
    scored = (goals >= 1).astype("float64")
    out = rows[["season", "variant", "horizon"]].copy()
    out["hgroup"] = np.where(rows["horizon"] == 0, "h0", "h1-5")
    for name, g, a in (
        ("model", rows["e_goals"], rows["e_assists"]),
        ("ref", rows["ref_goals"], rows["ref_assists"]),
    ):
        g = g.to_numpy(dtype="float64")
        a = a.to_numpy(dtype="float64")
        out[f"{name}_goals_ll"] = _poisson_ll(g, goals)
        out[f"{name}_assists_ll"] = _poisson_ll(a, assists)
        out[f"{name}_p_goal"] = 1 - np.exp(-g)
        out[f"{name}_brier"] = (out[f"{name}_p_goal"] - scored) ** 2
        out[f"{name}_e_goals"] = g
        out[f"{name}_e_assists"] = a
    out["goals"] = goals
    out["assists"] = assists
    out["scored"] = scored
    return out


METRICS = (
    "model_goals_ll",
    "ref_goals_ll",
    "model_assists_ll",
    "ref_assists_ll",
    "model_brier",
    "ref_brier",
    "model_e_goals",
    "ref_e_goals",
    "goals",
    "model_e_assists",
    "ref_e_assists",
    "assists",
)


def summarize(scores: pd.DataFrame) -> dict:
    per_season = scores.groupby(["variant", "hgroup", "season"])[list(METRICS)].mean()
    mean = per_season.groupby(["variant", "hgroup"]).mean()
    pooled = scores.groupby(["variant", "season"])[list(METRICS)].mean()
    objective = pooled.groupby("variant").mean()
    objective = objective["model_goals_ll"] + objective["model_assists_ll"]
    return {
        "per_season": per_season,
        "mean": mean,
        "objective": objective.sort_values(ascending=False),
    }


def reliability(scores: pd.DataFrame) -> dict:
    out = {}
    for hgroup, group in scores.groupby("hgroup"):
        out[hgroup] = {}
        for name in ("model", "ref"):
            p = group[f"{name}_p_goal"].to_numpy()
            y = group["scored"].to_numpy()
            out[hgroup][name] = {
                "brier": brier_decomposition(p, y).to_dict(),
                "table": reliability_table(p, y).to_dict(orient="records"),
            }
    return out


def walk(args) -> None:
    seasons = parse_seasons(args.seasons)
    if any(s not in DEVELOP for s in seasons):
        sys.exit("walk is develop-only: deadlines of 2017/18-2022/23")
    grid = json.loads(Path(args.grid).read_text()) if args.grid else default_grid()
    if not grid:
        grid = [{}]
    jobs = [(str(args.data_dir), s, grid) for s in seasons]
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(args.jobs, len(jobs))) as pool:
        results = list(pool.map(walk_season, jobs))
    rows = pd.concat([r["rows"] for r in results], ignore_index=True)
    timings = pd.concat([r["timings"] for r in results], ignore_index=True)
    scores = score_rows(rows)
    summary = summarize(scores)
    best = summary["objective"].index[0]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 200)
    print(f"\nvariants: {len(grid)}; objective = mean over seasons of goals + assists LL")
    print(summary["objective"].to_string())
    print(f"\nbest: {best}")
    print(summary["mean"].loc[best].to_string())
    print("\nper season (best):")
    print(summary["per_season"].loc[best].to_string())
    best_scores = scores[scores["variant"] == best]
    rel = reliability(best_scores)
    fits = timings.dropna(subset=["fit_shares_s"])
    predicts = timings.dropna(subset=["predict_s"])
    timing = {
        "fit_shares_s": fits["fit_shares_s"].describe().to_dict(),
        "fit_minutes_s": fits["fit_minutes_s"].describe().to_dict(),
        "fit_team_s": fits["fit_team_s"].describe().to_dict(),
        "predict_shares_s": predicts["predict_s"].describe().to_dict(),
        "wall_s": time.perf_counter() - start,
    }
    print(json.dumps(timing, indent=1, default=str))
    for hgroup, tables in rel.items():
        for name, table in tables.items():
            print(f"\nreliability {hgroup} {name}: Brier {table['brier']}")
            print(pd.DataFrame(table["table"]).to_string(index=False))
    report = {
        "seasons": seasons,
        "variants": grid,
        "n_variants": len(grid),
        "objective": summary["objective"].to_dict(),
        "best": best,
        "mean": summary["mean"].reset_index().to_dict(orient="records"),
        "per_season": summary["per_season"].reset_index().to_dict(orient="records"),
        "reliability_best": rel,
        "timing": timing,
        "coverage_at_last_cutoff": {str(r[0]): r for r in results[-1]["coverage"]},
    }
    out = Path(args.out or f"results/{datetime.now(UTC):%Y%m%dT%H%M%SZ}-shares-eval")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    timings.to_csv(out / "timings.csv", index=False)
    print(f"\nwrote {out / 'summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", default=os.environ.get("FPLOPT_DATA_DIR", "data"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("npxg-check")
    walk_parser = sub.add_parser("walk")
    walk_parser.add_argument("--seasons", default="2017-2022")
    walk_parser.add_argument("--grid")
    walk_parser.add_argument("--jobs", type=int, default=6)
    walk_parser.add_argument("--out")
    args = parser.parse_args()
    args.data_dir = Path(args.data_dir)
    if args.command == "npxg-check":
        print(json.dumps(npxg_check(args.data_dir), indent=1, default=str))
        return
    walk(args)


if __name__ == "__main__":
    main()
