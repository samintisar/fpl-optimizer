"""Bench-weight reliability on develop (Phase 5 plan, Task 8; PLAN §7 *Bench*). Dev only:
never imported by `fplopt`.

At every GW deadline of the chosen develop seasons (2017/18–2022/23; anything else is
refused) it builds the template squad and `--random` seeded random squads
(`fplopt.backtest.start_states`), picks each one's lineup with `best_lineup` on `v1`'s
horizon-0 xP (the planner's projected XI is the same rule), and compares, per bench slot
(0 = GK, 1–3 outfield):

- predicted weights: the fixed `DEFAULT_BENCH_WEIGHTS` and the minutes-based ones
  (`fplopt.optimize.minutes.minutes_bench_weights`, horizon 0); as a diagnostic also
  "skip": the outfield slot k weight with the skip rule, P(M ≥ 1 + Σ_{j<k} B_j), M the
  absent starters (Poisson-binomial) and B_j ~ Bernoulli(P(bench player j plays)), which
  the planner can't use (the bench is chosen by the MILP);
- realized, under `gw_score`'s autosub rules (formation limits and skips included):
  `needed` = slot k's player would have come on had he played (the lineup rescored with
  his minutes set to 1 if he had none), which is what the weight multiplies (the objective
  values slot k at w_k · xP_k, and xP_k already includes his own P(play));
  `came_on` = he did come on, compared with w_k · P(he plays) (`p_play_gw`).

Reports the means per slot (all rows; single-fixture vs double GWs), Brier scores of the
`needed` event, and a reliability table per slot (bins of the minutes-based weight). The
in-memory store holds only seasons ≤ the season being run (rows of tables without a season
column are filtered by the views' `available_at` as always).

    uv run python dev/bench_weights_eval.py --data-dir <data> [--seasons 2017-2022] \
        [--random 3] [--jobs 6] [--out results/p5b-bench-weights]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from fplopt.backtest.gw_score import score_gameweek
from fplopt.backtest.policies import best_lineup, target_xp
from fplopt.backtest.rules import backtest_rules
from fplopt.backtest.simulator import Caches, season_schedule
from fplopt.backtest.start_states import StartStateError, random_state, template_state
from fplopt.build.tables import TABLES
from fplopt.features.store import DataStore
from fplopt.optimize.minutes import (
    minutes_bench_weights,
    not_play_probabilities,
)
from fplopt.optimize.params import DEFAULT_BENCH_WEIGHTS

DEVELOP = range(2017, 2023)
MODEL = "v1"
BINS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0001)


def load_tables(data_dir: Path, last: int) -> dict[str, pd.DataFrame]:
    out = {}
    for name in TABLES:
        path = data_dir / f"{name}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if "season" in frame:
            frame = frame[frame["season"] <= last]
        out[name] = frame.reset_index(drop=True)
    return out


def skip_weights(q_starters: list[float], p_bench: list[float]) -> list[float]:
    """Outfield slot k (1..len(p_bench)): P(M ≥ 1 + Σ_{j<k} B_j), M Poisson-binomial over
    the outfield starters' q, B_j ~ Bernoulli(p_bench[j]), all independent."""
    dist_m = np.ones(1)
    for q in q_starters:
        nxt = np.append(dist_m * (1 - q), 0.0)
        nxt[1:] += dist_m * q
        dist_m = nxt
    tail = np.cumsum(dist_m[::-1])[::-1]  # P(M >= m)
    out = []
    dist_b = np.ones(1)  # distribution of Σ_{j<k} B_j
    for p in p_bench:
        need = np.arange(len(dist_b)) + 1
        reach = [tail[n] if n < len(tail) else 0.0 for n in need]
        out.append(float(sum(d * r for d, r in zip(dist_b, reach, strict=True))))
        nxt = np.append(dist_b * (1 - p), 0.0)
        nxt[1:] += dist_b * p
        dist_b = nxt
    return out


def season_rows(job: tuple) -> tuple[list[dict], dict]:
    data_dir, season, n_random, max_gws = job
    tables = load_tables(Path(data_dir), season)
    store = DataStore(tables=tables)
    rules = backtest_rules(season)
    caches = Caches()
    gameweeks = store.as_of(pd.Timestamp("2100-01-01", tz="UTC")).table(
        "gameweek", columns=["season", "gw", "gw_index"]
    )
    gw_indices = sorted(gameweeks.loc[gameweeks["season"] == season, "gw_index"].unique())
    if max_gws:
        gw_indices = gw_indices[:max_gws]
    rows: list[dict] = []
    skipped = 0
    began = time.perf_counter()
    for gw_index in gw_indices:
        schedule = season_schedule(store, season, int(gw_index))
        first = schedule.iloc[0]
        view = store.as_of(first["deadline_time"])
        frame = caches.xp(store, MODEL, view)
        outcomes = caches.outcomes(
            store, rules, season, int(first["gw"]), first["lockdown_time"]
        ).realized
        if not outcomes:
            continue
        target = target_xp(frame)
        q = not_play_probabilities(frame)
        q0 = {int(k): float(v) for (k, h), v in q.items() if h == 0}
        h0 = frame[frame["horizon"] == 0]
        in_double = set(h0.loc[h0["p_play"].isna(), "player_key"].tolist())
        starts = [("template", None)] + [(f"random:{s}", s) for s in range(n_random)]
        for start, seed in starts:
            try:
                if seed is None:
                    state = template_state(view, rules)
                else:
                    state = random_state(view, rules, seed)
            except StartStateError:
                skipped += 1
                continue
            squad = [(h.player_key, h.element_type) for h in state.holdings]
            positions = dict(squad)
            lineup = best_lineup(
                pd.DataFrame(squad, columns=["player_key", "element_type"]), target, rules
            )
            weights = minutes_bench_weights(squad, frame, [0], rules, DEFAULT_BENCH_WEIGHTS)[0]
            outfield_q = [q0.get(k, 1.0) for k in lineup.starters if positions[k] != 1]
            bench_p = [1.0 - q0.get(k, 1.0) for k in lineup.bench]
            skip = skip_weights(outfield_q, bench_p[1:])
            actual = score_gameweek(lineup, outcomes, positions, rules)
            came_on = {sub for _, sub in actual.autosubs}
            double = int(any(k in in_double for k in lineup.starters))
            for slot, sub in enumerate(lineup.bench):
                points, minutes = outcomes.get(sub, (0, 0))
                forced = dict(outcomes)
                if minutes <= 0:
                    forced[sub] = (points, 1)
                needed = any(
                    s == sub for _, s in score_gameweek(lineup, forced, positions, rules).autosubs
                )
                rows.append(
                    {
                        "season": season,
                        "gw_index": int(gw_index),
                        "start": start,
                        "slot": slot,
                        "player_key": sub,
                        "w_fixed": DEFAULT_BENCH_WEIGHTS[slot],
                        "w_minutes": weights[slot],
                        "w_skip": weights[0] if slot == 0 else skip[slot - 1],
                        "p_play": bench_p[slot],
                        "needed": int(needed),
                        "came_on": int(sub in came_on),
                        "played": int(minutes > 0),
                        "double": double,  # a projected starter has a double GW
                    }
                )
    stats = {
        "season": season,
        "seconds": round(time.perf_counter() - began, 1),
        "skipped_starts": skipped,
        "timings": {k: round(v, 1) for k, v in caches.timings.items()},
    }
    return rows, stats


def summarize(df: pd.DataFrame) -> dict:
    out: dict = {"n_rows": len(df), "n_squad_gws": len(df) // 4}
    means = []
    for slot, group in df.groupby("slot"):
        for subset, part in (
            ("all", group),
            ("single", group[group["double"] == 0]),
            ("double", group[group["double"] == 1]),
        ):
            if part.empty:
                continue
            means.append(
                {
                    "slot": int(slot),
                    "rows": subset,
                    "n": len(part),
                    "needed": part["needed"].mean(),
                    "fixed": part["w_fixed"].mean(),
                    "minutes": part["w_minutes"].mean(),
                    "skip": part["w_skip"].mean(),
                    "brier_fixed": ((part["w_fixed"] - part["needed"]) ** 2).mean(),
                    "brier_minutes": ((part["w_minutes"] - part["needed"]) ** 2).mean(),
                    "brier_skip": ((part["w_skip"] - part["needed"]) ** 2).mean(),
                    "came_on": part["came_on"].mean(),
                    "pred_came_on_fixed": (part["w_fixed"] * part["p_play"]).mean(),
                    "pred_came_on_minutes": (part["w_minutes"] * part["p_play"]).mean(),
                }
            )
    out["means"] = means
    reliability = []
    for slot, group in df.groupby("slot"):
        bins = pd.cut(group["w_minutes"], BINS, right=False)
        for interval, part in group.groupby(bins, observed=True):
            reliability.append(
                {
                    "slot": int(slot),
                    "bin": f"[{interval.left:.2f}, {min(interval.right, 1.0):.2f})",
                    "n": len(part),
                    "minutes": part["w_minutes"].mean(),
                    "needed": part["needed"].mean(),
                    "fixed": part["w_fixed"].mean(),
                }
            )
    out["reliability"] = reliability
    by_season = (
        df.groupby(["season", "slot"])[["needed", "w_fixed", "w_minutes"]].mean().reset_index()
    )
    out["by_season"] = by_season.to_dict(orient="records")
    return out


def text(summary: dict) -> str:
    lines = [f"{summary['n_squad_gws']} squad-GWs ({summary['n_rows']} bench rows)", ""]
    lines.append(
        "slot rows      n   needed  fixed  minutes   skip | Brier fixed minutes  skip"
        " | came_on  fixed·p minutes·p"
    )
    for m in summary["means"]:
        lines.append(
            f"{m['slot']:>4} {m['rows']:<6} {m['n']:>5}  {m['needed']:.3f}  {m['fixed']:.3f}"
            f"  {m['minutes']:.3f}  {m['skip']:.3f} |    {m['brier_fixed']:.4f} "
            f"{m['brier_minutes']:.4f} {m['brier_skip']:.4f} |  {m['came_on']:.3f}    "
            f"{m['pred_came_on_fixed']:.3f}     {m['pred_came_on_minutes']:.3f}"
        )
    lines += ["", "Reliability (bins of the minutes-based weight; realized = needed)"]
    lines.append("slot bin               n  minutes  needed  fixed")
    for r in summary["reliability"]:
        lines.append(
            f"{r['slot']:>4} {r['bin']:<14} {r['n']:>5}  {r['minutes']:.3f}   "
            f"{r['needed']:.3f}  {r['fixed']:.3f}"
        )
    return "\n".join(lines)


def parse_seasons(value: str) -> list[int]:
    out: list[int] = []
    for part in value.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return sorted(set(out))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", default=os.environ.get("FPLOPT_DATA_DIR", "data"))
    parser.add_argument("--seasons", default="2017-2022")
    parser.add_argument("--random", type=int, default=3, help="random squads per deadline")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-gws", type=int, default=0, help="first N GWs only (smoke test)")
    args = parser.parse_args()
    seasons = parse_seasons(args.seasons)
    bad = [s for s in seasons if s not in DEVELOP]
    if bad:
        sys.exit(f"develop seasons only (2017/18-2022/23): refusing {bad}")
    jobs = [(str(args.data_dir), s, args.random, args.max_gws) for s in seasons]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=min(args.jobs, len(jobs))) as pool:
        for season_rows_, stats in pool.map(season_rows, jobs):
            print(json.dumps(stats), flush=True)
            rows += season_rows_
    df = pd.DataFrame(rows)
    summary = summarize(df)
    print(text(summary))
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out / "rows.parquet", index=False)
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
        (out / "summary.txt").write_text(text(summary))


if __name__ == "__main__":
    main()
