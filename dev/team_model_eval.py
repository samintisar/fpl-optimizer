"""Walk-forward evaluation and tuning of the team model on develop (Phase 5 plan, Task 2;
PLAN §6.1). Dev only: never imported by `fplopt`.

For every develop deadline (2016/17–2022/23, every GW) the team model is fitted at the
walk-forward cutoff (`fplopt.models.fitted.refit_deadline`: the season's latest refit GW)
and predicts every fixture of the target GW and the next 5 (`upcoming_fixtures`). Each side's
prediction is scored against its realized goals:
- team-goals Poisson log-likelihood (mean per side);
- P(clean sheet) Brier score (P(opponent scores 0) under Dixon-Coles with `RHO`);
per horizon 0..5, for
- `model`: what `team_lambdas` returns (market λ where pre-match odds are visible at the
  deadline, else the ratings);
- `ratings`: the fitted ratings alone (every horizon);
- `elo`: the Elo-only reference: log λ = c0 + c_home · home + c_elo · (Elo − opponent Elo) /
  100, a Poisson regression on the matches of the 3 years before the cutoff (pre-match Elo,
  `rating_before`), predicting with the clubs' Elo at the deadline;
- `market`: the market λ itself, on the fixtures where odds are visible (horizon 0).
Also the odds coverage (share of target-GW fixtures with visible odds) per season, and the
market λ under Shin's de-vig instead of the power method.

Tuning (half-life, market weight w, prior strength): a joint grid; the objective is the
mean over seasons 2017/18–2022/23 (2016/17 is burn-in) of the ratings' mean
log-likelihood over horizons 1–5, where the ratings are what the model uses. The match
history (market λ included) is built once from a view after develop's last match and cut
at each cutoff (`fit_ratings` uses only rows available before it); one cutoff is checked
against `fit_team` on the cutoff's view.

Validate (2023/24–2024/25) and the holdout (2025/26) are never scored: the newest view is
as of 2023-07-01, after develop's last match.

    uv run python dev/team_model_eval.py [--data-dir PATH] [--quick] [--out DIR]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from fplopt.features.baseline import upcoming_fixtures
from fplopt.features.store import DataStore
from fplopt.models import team
from fplopt.models.fitted import refit_deadline

DEVELOP = tuple(range(2016, 2023))
TUNE_SEASONS = tuple(range(2017, 2023))
AS_OF = "2023-07-01T00:00Z"  # after develop's last match, before 2023/24
HALF_LIVES = (30.0, 45.0, 60.0, 90.0, 180.0, 365.0)
MARKET_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
PRIOR_STRENGTHS = (1.0, 2.0, 5.0, 20.0)
ELO_YEARS = 3
HORIZONS = range(6)


def season_label(season: int) -> str:
    return f"{season}/{str(season + 1)[-2:]}"


# --- inputs --------------------------------------------------------------------------------


def deadlines(store: DataStore, seasons) -> pd.DataFrame:
    gws = store.as_of(AS_OF).table(
        "gameweek", columns=["season", "gw", "gw_index", "deadline_time"]
    )
    gws = gws[gws["season"].isin(seasons)]
    return gws.sort_values(["season", "gw_index"]).reset_index(drop=True)


def outcomes(store: DataStore) -> pd.DataFrame:
    fx = store.as_of(AS_OF).table(
        "fixture", columns=["fixture_key", "season", "home_goals", "away_goals"]
    )
    fx = fx[fx["season"].isin(DEVELOP) & fx["home_goals"].notna()]
    return fx.astype({"home_goals": "int64", "away_goals": "int64"}).set_index("fixture_key")


def prematch_elo(store: DataStore) -> pd.DataFrame:
    """Per fixture: both clubs' pre-match Elo (rating_before)."""
    r = store.as_of(AS_OF).table(
        "team_rating", columns=["team_key", "fixture_key", "is_home", "rating_before"]
    )
    r = r[r["fixture_key"].notna()].astype({"fixture_key": "int64", "is_home": "bool"})
    home = r[r["is_home"]].set_index("fixture_key")["rating_before"].rename("elo_home")
    away = r[~r["is_home"]].set_index("fixture_key")["rating_before"].rename("elo_away")
    return pd.concat([home, away], axis=1).dropna()


def prediction_rows(store: DataStore, gws: pd.DataFrame, params: team.TeamParams) -> pd.DataFrame:
    """One row per (deadline, fixture): horizon, clubs, cutoff, the deadline's Elo and the
    visible market λ under the power and Shin de-vig (NaN without odds)."""
    frames = []
    shin = replace(params, devig="shin")
    for row in gws.itertuples():
        view = store.as_of(row.deadline_time)
        up = upcoming_fixtures(view)
        up = up[up["fixture_key"].notna() & up["is_home"].fillna(False).astype(bool)]
        fixtures = pd.DataFrame(
            {
                "fixture_key": up["fixture_key"].astype("int64").to_numpy(),
                "home_team_key": up["team_key"].astype("int64").to_numpy(),
                "away_team_key": up["opponent_team_key"].astype("int64").to_numpy(),
                "horizon": up["horizon"].astype("int64").to_numpy(),
            }
        )
        elo = team._elo(view)
        fixtures["elo_home"] = fixtures["home_team_key"].map(elo)
        fixtures["elo_away"] = fixtures["away_team_key"].map(elo)
        for name, p in (("power", params), ("shin", shin)):
            market = team._visible_market_lambdas(view, fixtures["fixture_key"], p)
            market = market.set_index("fixture_key")
            fixtures[f"{name}_home"] = fixtures["fixture_key"].map(market["lambda_home"])
            fixtures[f"{name}_away"] = fixtures["fixture_key"].map(market["lambda_away"])
        fixtures = fixtures.assign(
            season=row.season,
            gw=row.gw,
            deadline=row.deadline_time,
            cutoff=refit_deadline(view),
        )
        frames.append(fixtures)
    return pd.concat(frames, ignore_index=True)


# --- the Elo reference ----------------------------------------------------------------------


def fit_elo_reference(history: pd.DataFrame, elo: pd.DataFrame, cutoff: pd.Timestamp) -> np.ndarray:
    """(c0, c_home, c_elo) by Poisson MLE on the matches of the ELO_YEARS before `cutoff`."""
    rows = history[
        (history["available_at"] < cutoff)
        & (history["kickoff_time"] >= cutoff - pd.Timedelta(days=365.25 * ELO_YEARS))
    ].join(elo, on="fixture_key", how="inner")
    diff = (rows["elo_home"] - rows["elo_away"]).to_numpy() / 100
    x = np.concatenate([diff, -diff])
    home = np.concatenate([np.ones(len(rows)), np.zeros(len(rows))])
    y = np.concatenate([rows["home_goals"].to_numpy(), rows["away_goals"].to_numpy()])

    def objective(c):
        eta = c[0] + c[1] * home + c[2] * x
        mu = np.exp(eta)
        r = mu - y
        return float(np.sum(mu - y * eta)), np.array([r.sum(), r @ home, r @ x])

    return minimize(objective, np.array([0.3, 0.0, 0.0]), jac=True, method="L-BFGS-B").x


# --- scoring ---------------------------------------------------------------------------------


def score(lambda_home, lambda_away, goals_home, goals_away) -> dict[str, np.ndarray]:
    """Per fixture: the two sides' Poisson log-likelihoods summed and the two CS Brier
    terms summed (divide by 2 for per-side means)."""
    lh, la = np.asarray(lambda_home, float), np.asarray(lambda_away, float)
    gh, ga = np.asarray(goals_home, float), np.asarray(goals_away, float)
    ll = gh * np.log(lh) - lh - gammaln(gh + 1) + ga * np.log(la) - la - gammaln(ga + 1)
    m = team.dc_matrix(lh, la, team.RHO)
    cs_home = m[:, :, 0].sum(axis=1)  # away scores 0
    cs_away = m[:, 0, :].sum(axis=1)
    brier = (cs_home - (ga == 0)) ** 2 + (cs_away - (gh == 0)) ** 2
    return {"ll": ll, "brier": brier}


def summarize(rows: pd.DataFrame, predictor: str) -> pd.DataFrame:
    """Mean per-side log-likelihood and Brier by (season, horizon) for `predictor`'s
    λ columns (`{predictor}_home`, `{predictor}_away`), on rows where it exists."""
    sub = rows[rows[f"{predictor}_home"].notna()]
    s = score(
        sub[f"{predictor}_home"], sub[f"{predictor}_away"], sub["home_goals"], sub["away_goals"]
    )
    sub = sub.assign(ll=s["ll"] / 2, brier=s["brier"] / 2)
    return sub.groupby(["season", "horizon"]).agg(
        ll=("ll", "mean"), brier=("brier", "mean"), n=("ll", "size")
    )


def tuning_objective(rows: pd.DataFrame) -> tuple[float, float, float]:
    """Mean over TUNE_SEASONS of the ratings' mean log-likelihood (and Brier) over horizons
    1–5; and the same log-likelihood at horizon 0."""
    t = summarize(rows[rows["season"].isin(TUNE_SEASONS)], "ratings").reset_index()
    ahead = t[t["horizon"] >= 1]
    weighted = ahead.assign(ll_n=ahead["ll"] * ahead["n"], br_n=ahead["brier"] * ahead["n"])
    per_season = weighted.groupby("season")[["ll_n", "br_n", "n"]].sum()
    h0 = t[t["horizon"] == 0].set_index("season")["ll"]
    return (
        float((per_season["ll_n"] / per_season["n"]).mean()),
        float((per_season["br_n"] / per_season["n"]).mean()),
        float(h0.mean()),
    )


# --- fits --------------------------------------------------------------------------------------


def cutoff_inputs(store: DataStore, cutoffs) -> dict:
    """Per cutoff: the Elo and the clubs `fit_team` would use."""
    out = {}
    for cutoff in cutoffs:
        view = store.as_of(cutoff)
        out[cutoff] = (team._elo(view), team._current_teams(view))
    return out


def ratings_predictions(
    rows: pd.DataFrame, history: pd.DataFrame, inputs: dict, params: team.TeamParams
) -> tuple[pd.DataFrame, list[float]]:
    """rows + ratings_home / ratings_away from fits at each row's cutoff; fit seconds."""
    rows = rows.copy()
    rows["ratings_home"] = np.nan
    rows["ratings_away"] = np.nan
    seconds = []
    for cutoff, idx in rows.groupby("cutoff").groups.items():
        elo, teams = inputs[cutoff]
        start = time.perf_counter()
        fit = team.fit_ratings(history, elo, teams, cutoff, params)
        seconds.append(time.perf_counter() - start)
        sub = rows.loc[idx]
        deadline_elo = pd.concat(
            [
                pd.Series(sub["elo_home"].to_numpy(), index=sub["home_team_key"]),
                pd.Series(sub["elo_away"].to_numpy(), index=sub["away_team_key"]),
            ]
        )
        deadline_elo = deadline_elo[~deadline_elo.index.duplicated()]
        lh, la = team.ratings_lambdas(fit, sub["home_team_key"], sub["away_team_key"], deadline_elo)
        rows.loc[idx, "ratings_home"] = lh
        rows.loc[idx, "ratings_away"] = la
    return rows, seconds


def elo_predictions(rows: pd.DataFrame, history: pd.DataFrame, elo: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["elo_ref_home"] = np.nan
    rows["elo_ref_away"] = np.nan
    for cutoff, idx in rows.groupby("cutoff").groups.items():
        c = fit_elo_reference(history, elo, cutoff)
        sub = rows.loc[idx]
        diff = (sub["elo_home"] - sub["elo_away"]).to_numpy() / 100
        rows.loc[idx, "elo_ref_home"] = np.exp(c[0] + c[1] + c[2] * diff)
        rows.loc[idx, "elo_ref_away"] = np.exp(c[0] - c[2] * diff)
    return rows


def with_model(rows: pd.DataFrame) -> pd.DataFrame:
    """model = market (power) where visible, else the ratings (what team_lambdas returns)."""
    market = rows["power_home"].notna()
    return rows.assign(
        model_home=np.where(market, rows["power_home"], rows["ratings_home"]),
        model_away=np.where(market, rows["power_away"], rows["ratings_away"]),
        market_home=rows["power_home"],
        market_away=rows["power_away"],
    )


# --- main ------------------------------------------------------------------------------------


def table(frame: pd.DataFrame) -> str:
    return frame.to_string(float_format=lambda v: f"{v:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default = os.environ.get("FPLOPT_DATA_DIR", str(Path(__file__).parent.parent / "data"))
    parser.add_argument("--data-dir", default=default)
    parser.add_argument("--quick", action="store_true", help="2 seasons, a 2 x 2 grid")
    parser.add_argument("--out", default=None, help="default results/<UTC>-team-eval")
    args = parser.parse_args()
    seasons = (2021, 2022) if args.quick else DEVELOP
    half_lives = HALF_LIVES[1:3] if args.quick else HALF_LIVES
    weights = MARKET_WEIGHTS[2:4] if args.quick else MARKET_WEIGHTS
    priors = PRIOR_STRENGTHS[1:] if args.quick else PRIOR_STRENGTHS
    store = DataStore(data_dir=args.data_dir)
    base = team.TeamParams()
    report: dict = {"defaults": asdict(base), "rho": team.RHO}

    t0 = time.perf_counter()
    gws = deadlines(store, seasons)
    rows = prediction_rows(store, gws, base)
    goals = outcomes(store)
    rows = rows.join(goals[["home_goals", "away_goals"]], on="fixture_key", how="inner")
    print(
        f"{len(gws)} deadlines, {len(rows)} fixture predictions ({time.perf_counter() - t0:.0f} s)"
    )

    # Match history once, every match (fit_ratings applies each cutoff and half-life).
    t0 = time.perf_counter()
    widest = replace(base, half_life_days=1e4)  # every visible match: any cutoff, any grid point
    history = team.match_history(store.as_of(AS_OF), widest)
    print(f"match history: {len(history)} matches ({time.perf_counter() - t0:.1f} s)")
    cutoffs = sorted(rows["cutoff"].unique())
    inputs = cutoff_inputs(store, cutoffs)

    # Check: the cut history reproduces fit_team on the cutoff's own view.
    check = cutoffs[len(cutoffs) // 2]
    direct = team.fit_team(store.as_of(check), base)
    elo, teams = inputs[check]
    assert team.fit_ratings(history, elo, teams, check, base) == direct, "history cut differs"

    # Timing of the production path (match history + fit, on the cutoff's view).
    times = []
    for cutoff in cutoffs[:: max(1, len(cutoffs) // 6)]:
        start = time.perf_counter()
        fit = team.fit_team(store.as_of(cutoff), base)
        times.append(time.perf_counter() - start)
    predict = []
    for d in gws["deadline_time"].iloc[:: max(1, len(gws) // 6)]:
        view = store.as_of(d)
        fit = team.fit_team(view.earlier(refit_deadline(view)), base)
        start = time.perf_counter()
        team.team_lambdas(view, fit)
        predict.append(time.perf_counter() - start)
    report["timing"] = {
        "fit_team_seconds": [round(t, 2) for t in times],
        "team_lambdas_seconds": [round(t, 3) for t in predict],
        "n_matches_last_fit": fit.n_matches,
    }
    print("fit_team s:", report["timing"]["fit_team_seconds"])
    print("team_lambdas s:", report["timing"]["team_lambdas_seconds"])

    # Coverage: target-GW fixtures with visible odds.
    h0 = rows[rows["horizon"] == 0]
    coverage = h0.groupby("season")["power_home"].apply(lambda s: s.notna().mean())
    ahead = (
        rows[rows["horizon"] > 0].groupby("season")["power_home"].apply(lambda s: s.notna().mean())
    )
    report["coverage"] = {season_label(s): round(float(v), 4) for s, v in coverage.items()}
    report["coverage_horizon_1_5"] = {season_label(s): round(float(v), 4) for s, v in ahead.items()}
    print("odds coverage, target GW:", report["coverage"])
    print("odds coverage, horizons 1-5:", report["coverage_horizon_1_5"])

    # Grid: half-life x market weight x prior strength.
    grid = []
    for half_life, w, strength in itertools.product(half_lives, weights, priors):
        params = replace(base, half_life_days=half_life, market_weight=w, prior_strength=strength)
        scored, seconds = ratings_predictions(rows, history, inputs, params)
        ll, brier, ll0 = tuning_objective(scored)
        grid.append(
            {
                "half_life_days": half_life,
                "market_weight": w,
                "prior_strength": strength,
                "ll_h1_5": ll,
                "brier_h1_5": brier,
                "ll_h0": ll0,
                "fit_seconds_median": float(np.median(seconds)),
            }
        )
    grid = pd.DataFrame(grid)
    print(f"\ngrid ({len(grid)} variants): ratings' mean log-likelihood over horizons 1-5")
    pivot = grid.pivot_table("ll_h1_5", ["half_life_days", "prior_strength"], "market_weight")
    print(table(pivot))
    best = grid.loc[grid["ll_h1_5"].idxmax()]
    chosen = replace(
        base,
        half_life_days=float(best["half_life_days"]),
        market_weight=float(best["market_weight"]),
        prior_strength=float(best["prior_strength"]),
    )
    report["grid"] = grid.to_dict(orient="records")
    print(
        f"best: half-life {chosen.half_life_days:.0f} d, w {chosen.market_weight}, "
        f"prior {chosen.prior_strength}: {best.to_dict()}"
    )

    # Metrics at the chosen point vs the references, per horizon (all develop seasons, and
    # the tuning seasons).
    scored, _ = ratings_predictions(rows, history, inputs, chosen)
    elo_pre = prematch_elo(store)
    scored = with_model(elo_predictions(scored, history, elo_pre))
    predictors = ("model", "ratings", "elo_ref", "market")
    metrics = {}
    for label, subset in (
        ("develop", scored),
        ("tune_seasons", scored[scored["season"].isin(TUNE_SEASONS)]),
    ):
        per_h = {}
        for p in predictors:
            sub = subset[subset[f"{p}_home"].notna()]
            s = score(sub[f"{p}_home"], sub[f"{p}_away"], sub["home_goals"], sub["away_goals"])
            g = sub.assign(ll=s["ll"] / 2, brier=s["brier"] / 2).groupby("horizon")
            per_h[p] = g.agg(ll=("ll", "mean"), brier=("brier", "mean"), n=("ll", "size"))
        both = pd.concat(per_h, axis=1)
        both.columns = [f"{p}_{m}" for p, m in both.columns]
        metrics[label] = both
        print(f"\n{label}: mean per-side Poisson log-likelihood / CS Brier by horizon")
        print(table(both))
    # Horizon 0, fixtures with visible odds: market vs ratings vs Elo on the same rows.
    h0 = scored[(scored["horizon"] == 0) & scored["market_home"].notna()]
    same = {}
    for p in ("market", "ratings", "elo_ref"):
        s = score(h0[f"{p}_home"], h0[f"{p}_away"], h0["home_goals"], h0["away_goals"])
        same[p] = {"ll": float(s["ll"].mean() / 2), "brier": float(s["brier"].mean() / 2)}
    s = score(h0["shin_home"], h0["shin_away"], h0["home_goals"], h0["away_goals"])
    same["market_shin"] = {"ll": float(s["ll"].mean() / 2), "brier": float(s["brier"].mean() / 2)}
    same["n_fixtures"] = len(h0)
    print("\nhorizon 0 with visible odds (same fixtures):", json.dumps(same, indent=1))
    per_season = {}
    for season, sub in scored.groupby("season"):
        row = {}
        for p in ("model", "ratings", "elo_ref"):
            s = score(sub[f"{p}_home"], sub[f"{p}_away"], sub["home_goals"], sub["away_goals"])
            row[p] = round(float(s["ll"].mean() / 2), 5)
        per_season[season_label(season)] = row
    print("per season (all horizons) log-likelihood:", json.dumps(per_season, indent=1))
    report["chosen"] = asdict(chosen)
    report["metrics"] = {k: v.reset_index().to_dict(orient="records") for k, v in metrics.items()}
    report["horizon0_with_odds"] = same
    report["per_season_ll"] = per_season

    out = Path(args.out or f"results/{datetime.now(UTC):%Y%m%dT%H%M%SZ}-team-eval")
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\nwrote {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
