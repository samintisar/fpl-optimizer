"""Defensive contributions: check FPL-Core-Insights against FPL, set the prior once, and
sanity-check the v1 term live (Phase 5b, Task 10; PLAN §6.5, §11, issue #14). Dev only:
never imported by `fplopt`.

FPL-Core-Insights (FCI) is research data with no licence (PLAN §10, §12): its files live in
the git-ignored `research/fpl-core-insights/` (dev/README.md lists the URLs and the pinned
commit), are read here as plain CSV data and never reach `raw/`, `data/` or the package. Run
this script with `python -I` (isolated mode: nothing next to the data or in the current
directory is importable). Our tables are read read-only from `--data-dir`, only the
seasons named below (never 2025/26, the holdout).

Subcommands:
- `verify` (#14): FCI's tackles + interceptions + blocks + clearances (CBIT) and + recoveries
  (CBIRT) against FPL's own counts in our `player_match` for 2026/27 GW1–5. Players are
  mapped by FPL `player_code` (= our `player_key`), fixtures by (season, home club code, away
  club code) (FCI's club ids are FPL team codes = our `team_key`). Reports the mapping
  coverage both ways, then per position (FPL's `element_type` of the season) the exact-match
  rate, mean absolute difference and agreement of the threshold outcome (DEF CBIT ≥ 10,
  MID/FWD CBIRT ≥ 12) of FPL's `defensive_contribution`, and the same per component.
  FCI's tackle count is its `tackles_won` column: its `tackles` column is blank in 2026/27
  and, in 2024/25, a larger count (≈ attempts: it sums to the team's tackles won / won %).
- `fit`: `k` (shrinkage pseudo-90s) and `r` (negative-binomial dispersion) from FCI 2024/25,
  per group (DEF CBIT, MID CBIRT, FWD CBIRT, MID+FWD CBIRT); `--tackles f_tackles_attempted`
  uses FCI's `tackles` column instead (a sensitivity check). Every played Premier League row
  (minutes > 0, outfield, position = our `player_season` 2024/25 `element_type`) is predicted
  from the player's earlier rows of the season only (kickoff strictly before):
  `rate = (actions + k · pos_mean) / (minutes / 90 + k)`, `pos_mean` the pooled per-90 rate of
  the position's earlier rows (the season mean before any), and the count ~ NB(mean = rate ·
  minutes / 90, dispersion r). (k, r) maximize that walk-forward predictive log-likelihood
  (Nelder–Mead on logs). Check: the same with the first half's rows (GW ≤ 19) predicting
  the second half's. Also prints the season per-90 means (the early-season fallback).
- `live`: the v1 term on 2026/27 (the live season, never the holdout): at each deadline of
  `--gws` with realized outcomes, P(defcon) per pool player-fixture (v1's minutes split and
  the deadline's rates), against the realized defcon rate (FPL's count reaches the
  threshold), by position. Tables are loaded without the 2025/26 season, so the model is fit
  on 2016/17–2024/25 + the visible 2026/27 rows only.

    .venv/Scripts/python.exe -I dev/defcon_prior.py --data-dir <data> verify
    .venv/Scripts/python.exe -I dev/defcon_prior.py --data-dir <data> fit
    .venv/Scripts/python.exe -I dev/defcon_prior.py --data-dir <data> live --gws 2-7
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import nbinom, poisson

REPO = Path(__file__).resolve().parents[1]
FCI_DIR = REPO / "research" / "fpl-core-insights"
LIVE = 2026
FIT_SEASON = 2024
THRESHOLD = {2: 10, 3: 12, 4: 12}
POSITION_NAMES = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
FILES = ("playermatchstats", "matches", "players")
# FCI column -> our name for it (prefixed `f_`). FCI's `tackles_won` is FPL's (Opta's)
# `tackles`: FCI's `tackles` is a larger count (attempts) in 2024/25 and empty in 2026/27.
FCI_COLUMNS = {
    "minutes_played": "f_minutes",
    "tackles_won": "f_tackles",
    "tackles": "f_tackles_attempted",
    "interceptions": "f_interceptions",
    "blocks": "f_blocks",
    "clearances": "f_clearances",
    "recoveries": "f_recoveries",
    "defensive_contributions": "f_defensive_contributions",
}
CBIT = ("f_tackles", "f_interceptions", "f_blocks", "f_clearances")
CBIRT = (*CBIT, "f_recoveries")
FIRST_HALF_LAST_GW = 19


def read_csv(path: Path) -> pd.DataFrame:
    """An FCI CSV as data (names may carry stray bytes: replaced, never interpreted)."""
    return pd.read_csv(path, encoding="utf-8", encoding_errors="replace", low_memory=False)


def our_table(data_dir: Path, name: str, season: int, columns=None) -> pd.DataFrame:
    """One season of one of our tables (filtered on read: no other season is loaded)."""
    return pd.read_parquet(
        data_dir / f"{name}.parquet", columns=columns, filters=[("season", "==", season)]
    )


def positions(data_dir: Path, season: int) -> pd.DataFrame:
    frame = our_table(data_dir, "player_season", season, ["player_key", "element_type"])
    return frame.drop_duplicates("player_key", keep="last")


def fci_rows(season_dir: Path, gws: list[int] | None) -> pd.DataFrame:
    """FCI Premier League player-match rows: (match_id, gameweek, kickoff_time, home_team,
    away_team, player_code) and the `FCI_COLUMNS` as floats (NaN = blank), from per-GW
    folders (`gws`) or the season files."""
    if gws is None:
        stats = read_csv(season_dir / "playermatchstats.csv")
        matches = read_csv(season_dir / "matches.csv")
        players = read_csv(season_dir / "players.csv")
    else:
        parts = [
            {name: read_csv(season_dir / f"GW{gw}" / f"{name}.csv") for name in FILES} for gw in gws
        ]
        stats = pd.concat([p["playermatchstats"] for p in parts], ignore_index=True)
        matches = pd.concat([p["matches"] for p in parts], ignore_index=True)
        players = pd.concat([p["players"] for p in parts], ignore_index=True)
    if "tournament" in matches:
        matches = matches[matches["tournament"] == "prem"]
    matches = matches.drop_duplicates("match_id", keep="last")
    matches = matches[["match_id", "gameweek", "kickoff_time", "home_team", "away_team"]]
    players = players.drop_duplicates("player_id", keep="last")[["player_id", "player_code"]]
    stats = stats.drop_duplicates(["player_id", "match_id"], keep="last")
    stats = stats[["player_id", "match_id", *(c for c in FCI_COLUMNS if c in stats)]]
    stats = stats.rename(columns=FCI_COLUMNS)
    out = stats.merge(matches, on="match_id", how="inner").merge(
        players, on="player_id", how="left"
    )
    for column in ("gameweek", "home_team", "away_team", "player_code"):
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")
    out["kickoff_time"] = pd.to_datetime(out["kickoff_time"], utc=True, format="mixed")
    for column in FCI_COLUMNS.values():
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce").astype("float64")
        else:
            out[column] = np.nan
    out["f_minutes"] = out["f_minutes"].fillna(0.0)
    if gws is not None:
        out = out[out["gameweek"].isin(gws)]
    return out.reset_index(drop=True)


# --- verify -----------------------------------------------------------------------------------


def compare(name: str, ours: np.ndarray, theirs: np.ndarray) -> dict:
    return {
        "stat": name,
        "n": len(ours),
        "exact": float((ours == theirs).mean()),
        "mae": float(np.abs(ours - theirs).mean()),
        "mean_diff": float((theirs - ours).mean()),
    }


def verify(args) -> None:
    gws = [1, 2, 3, 4, 5]
    fci = fci_rows(FCI_DIR / "2026-2027", gws)
    played = fci[fci["f_minutes"] > 0].copy()
    fixtures = our_table(
        args.data_dir, "fixture", LIVE, ["fixture_key", "gw", "home_team_key", "away_team_key"]
    )
    fixtures = fixtures[fixtures["gw"].isin(gws)]
    played = played.merge(
        fixtures.rename(columns={"home_team_key": "home_team", "away_team_key": "away_team"}),
        on=["home_team", "away_team"],
        how="left",
    )
    matches = our_table(args.data_dir, "player_match", LIVE)
    matches = matches[matches["gw"].isin(gws) & (matches["minutes"] > 0)]
    matches = matches.merge(positions(args.data_dir, LIVE), on="player_key", how="left")
    print(
        f"FCI 2026/27 GW1-5: {fci['match_id'].nunique()} Premier League matches "
        f"(ours: {fixtures['fixture_key'].nunique()}), {len(played)} player rows with minutes > 0"
    )
    print(
        f"  FCI rows: fixture mapped {played['fixture_key'].notna().mean():.4f}; "
        f"player_code present {played['player_code'].notna().mean():.4f}"
    )
    played = played.dropna(subset=["fixture_key", "player_code"])
    played["fixture_key"] = played["fixture_key"].astype("int64")
    played["player_key"] = played["player_code"].astype("int64")
    joined = matches.merge(played, on=["player_key", "fixture_key"], how="outer", indicator=True)
    both = joined[joined["_merge"] == "both"].copy()
    n_ours = int((joined["_merge"] != "right_only").sum())
    print(
        f"  our player_match rows with minutes > 0: {n_ours}; matched {len(both)} "
        f"({len(both) / n_ours:.4f}); FCI rows matched {len(both) / len(played):.4f}"
    )
    for tag, side in (("ours", "left_only"), ("FCI", "right_only")):
        rows = joined[joined["_merge"] == side]
        detail = rows[["player_key", "fixture_key", "minutes", "f_minutes"]].to_dict("records")
        print(f"  unmatched {tag}: {len(rows)} {detail[:5]}")
    print(
        f"  matched rows with equal minutes: {(both['minutes'] == both['f_minutes']).mean():.4f}"
        f"; mean |minutes diff| {(both['minutes'] - both['f_minutes']).abs().mean():.3f}"
    )
    print(
        "  FCI blank share among matched rows: "
        + ", ".join(f"{c} {both[c].isna().mean():.3f}" for c in FCI_COLUMNS.values())
    )
    blank = both[list(CBIRT)].isna().any(axis=1)
    print(
        f"  matched rows with a blank CBIRT component: {int(blank.sum())}, by position "
        f"{both.loc[blank, 'element_type'].map(POSITION_NAMES).value_counts().to_dict()}"
    )
    outfield = both[~blank & both["element_type"].isin((2, 3, 4))].copy()

    def ours(column: str) -> np.ndarray:
        return outfield[column].astype("float64").to_numpy()

    def theirs(*columns: str) -> np.ndarray:
        return outfield[list(columns)].sum(axis=1).to_numpy(dtype="float64")

    et = outfield["element_type"].to_numpy()
    fpl = ours("defensive_contribution")
    our_cbit = ours("clearances_blocks_interceptions") + ours("tackles")
    our_cbirt = our_cbit + ours("recoveries")
    fci_cbit, fci_cbirt = theirs(*CBIT), theirs(*CBIRT)
    fci_count = np.where(et == 2, fci_cbit, fci_cbirt)
    our_count = np.where(et == 2, our_cbit, our_cbirt)
    print(
        "  FPL defensive_contribution = our CBI + tackles (DEF) / + recoveries (MID/FWD): "
        f"{(fpl == our_count).mean():.4f}"
    )
    table = []
    for label, ets in (("DEF", (2,)), ("MID", (3,)), ("FWD", (4,)), ("all", (2, 3, 4))):
        rows = np.isin(et, ets)
        t = np.array([THRESHOLD[int(e)] for e in et[rows]])
        row = compare("CBIT/CBIRT", fpl[rows], fci_count[rows])
        hit_fpl, hit_fci = fpl[rows] >= t, fci_count[rows] >= t
        row |= {
            "position": label,
            "threshold_agree": float((hit_fpl == hit_fci).mean()),
            "fpl_hit_rate": float(hit_fpl.mean()),
            "fci_hit_rate": float(hit_fci.mean()),
        }
        table.append(row)
        components = (
            (
                "CBI",
                ours("clearances_blocks_interceptions"),
                theirs("f_clearances", "f_blocks", "f_interceptions"),
            ),
            ("tackles (FCI tackles_won)", ours("tackles"), theirs("f_tackles")),
            ("recoveries", ours("recoveries"), theirs("f_recoveries")),
        )
        for name, a, b in components:
            table.append(compare(name, a[rows], b[rows]) | {"position": label})
    pd.set_option("display.width", 200)
    frame = pd.DataFrame(table).set_index(["position", "stat"])
    print(frame.to_string(float_format=lambda v: f"{v:.4f}"))
    attempted = outfield["f_tackles_attempted"].notna().mean()
    print(f"FCI `tackles` column non-blank on matched outfield rows: {attempted:.4f}")
    own = outfield["f_defensive_contributions"].notna().mean()
    print(f"FCI `defensive_contributions` non-blank on matched outfield rows: {own:.4f}")
    print("\nFCI tackle columns against FCI's team stats (which column is which):")
    tackle_columns("2024/25", [FCI_DIR / "2024-2025"])
    tackle_columns("2026/27 GW1-5", [FCI_DIR / "2026-2027" / f"GW{gw}" for gw in gws])


def tackle_columns(label: str, folders: list[Path]) -> None:
    """Per team-match, the sum of each player tackle column against the team's `tackles_won`
    and its total tackles (= won / won %; blank in 2026/27)."""
    stats = pd.concat([read_csv(f / "playermatchstats.csv") for f in folders])
    matches = pd.concat([read_csv(f / "matches.csv") for f in folders])
    players = pd.concat([read_csv(f / "players.csv") for f in folders])
    if "tournament" in matches:
        matches = matches[matches["tournament"] == "prem"]
    matches = matches.drop_duplicates("match_id")
    players = players.drop_duplicates("player_id")[["player_id", "team_code"]]
    stats = stats.drop_duplicates(["player_id", "match_id"]).merge(players, on="player_id")
    columns = ["tackles", "tackles_won"]
    for column in columns:
        stats[column] = pd.to_numeric(stats[column], errors="coerce")
    sums = stats.groupby(["match_id", "team_code"])[columns].sum(min_count=1).reset_index()
    sides = []
    for side in ("home", "away"):
        part = matches[
            ["match_id", f"{side}_team", f"{side}_tackles_won", f"{side}_tackles_won_pct"]
        ]
        sides.append(part.set_axis(["match_id", "team_code", "won", "pct"], axis=1))
    teams = pd.concat(sides)
    for column in ("team_code", "won", "pct"):
        teams[column] = pd.to_numeric(teams[column], errors="coerce")
    teams["total"] = teams["won"] / (teams["pct"] / 100.0)
    sums["team_code"] = pd.to_numeric(sums["team_code"], errors="coerce")
    joined = sums.merge(teams, on=["match_id", "team_code"])
    print(
        f"  {label}: {len(joined)} team-matches; team tackles won {joined['won'].mean():.2f}, "
        f"won / won % {joined['total'].mean():.2f} per team-match"
    )
    for column in columns:
        values = joined[column]
        print(
            f"    player `{column}` summed: mean {values.mean():.2f}; equals the team's tackles "
            f"won {(values == joined['won']).mean():.3f}; within 1 of won / won % "
            f"{((values - joined['total']).abs() < 1).mean():.3f}"
        )


# --- fit --------------------------------------------------------------------------------------


def nb_loglik(y: np.ndarray, mu: np.ndarray, r: float) -> np.ndarray:
    """log NB(y; mean mu, dispersion r) (var = mu + mu² / r)."""
    mu = np.clip(mu, 1e-12, None)
    return (
        gammaln(y + r)
        - gammaln(r)
        - gammaln(y + 1.0)
        + r * np.log(r / (r + mu))
        + y * np.log(mu / (r + mu))
    )


def per_90(frame: pd.DataFrame, label: str, tackles, attempted, recoveries, cbi=None) -> None:
    """Per-90 means of the components by position (a definition check across sources)."""
    if cbi is None:
        frame = frame.assign(
            f_cbi=frame[["f_clearances", "f_blocks", "f_interceptions"]].sum(axis=1)
        )
        cbi = "f_cbi"
    out = {}
    for position in (2, 3, 4):
        part = frame[frame["element_type"] == position]
        e = part["e"].sum()
        row = {
            "CBI": part[cbi].astype("float64").sum() / e,
            "tackles": part[tackles].astype("float64").sum() / e,
            "recoveries": part[recoveries].astype("float64").sum() / e,
        }
        if attempted is not None:
            row["FCI tackles (attempts)"] = part[attempted].sum() / e
        out[POSITION_NAMES[position]] = {k: round(float(v), 3) for k, v in row.items()}
    print(f"per-90 means, {label}: {out}")


def fit_rows(data_dir: Path, tackles: str = "f_tackles") -> pd.DataFrame:
    """FCI 2024/25 played outfield rows with element_type, count (CBIT for DEF, CBIRT for
    MID/FWD) and the player's / position's earlier totals of the season."""
    fci = fci_rows(FCI_DIR / "2024-2025", None)
    fci["f_tackles"] = fci[tackles]  # FCI tackles_won (default) or tackles (attempts)
    fci = fci[fci["f_minutes"] > 0].dropna(subset=["player_code"])
    pos = positions(data_dir, FIT_SEASON).rename(columns={"player_key": "player_code"})
    pos["player_code"] = pos["player_code"].astype("Int64")
    n_all = len(fci)
    fci = fci.merge(pos, on="player_code", how="left")
    print(
        f"FCI 2024/25: {fci['match_id'].nunique()} matches, {n_all} played rows; "
        f"position mapped {fci['element_type'].notna().mean():.4f}"
    )
    fci = fci[fci["element_type"].isin((2, 3, 4))].copy()
    missing = fci[list(CBIRT)].isna().any(axis=1)
    print(f"  outfield rows {len(fci)}, missing a component {int(missing.sum())} (dropped)")
    fci = fci[~missing]
    cbit = fci[list(CBIT)].sum(axis=1)
    fci["count"] = np.where(fci["element_type"] == 2, cbit, cbit + fci["f_recoveries"])
    fci["e"] = fci["f_minutes"] / 90.0
    per_90(fci, "FCI 2024/25", "f_tackles", "f_tackles_attempted", "f_recoveries")
    ours = our_table(data_dir, "player_match", LIVE).merge(
        positions(data_dir, LIVE), on="player_key"
    )
    ours = ours[(ours["minutes"] > 0) & ours["element_type"].isin((2, 3, 4))].copy()
    ours["e"] = ours["minutes"] / 90.0
    ours["f_cbi"] = ours["clearances_blocks_interceptions"].astype("float64")
    per_90(
        ours.rename(columns={"tackles": "t", "recoveries": "rec"}),
        "FPL 2026/27 GW1-5",
        "t",
        None,
        "rec",
        cbi="f_cbi",
    )
    fci = fci.sort_values(["kickoff_time", "player_code"], kind="mergesort").reset_index(drop=True)
    # Player's earlier totals (one match per kickoff per player: cumsum minus self).
    by_player = fci.groupby("player_code", sort=False)
    fci["prior_count"] = by_player["count"].cumsum() - fci["count"]
    fci["prior_e"] = by_player["e"].cumsum() - fci["e"]
    # Position's totals over rows with an earlier kickoff.
    per_time = fci.groupby(["element_type", "kickoff_time"], sort=True)[["count", "e"]].sum()
    cum = per_time.groupby(level=0).cumsum() - per_time
    fci = fci.merge(
        cum.rename(columns={"count": "pos_count", "e": "pos_e"}).reset_index(),
        on=["element_type", "kickoff_time"],
        how="left",
    )
    season_mean = (
        fci.groupby("element_type")["count"].sum() / fci.groupby("element_type")["e"].sum()
    )
    fallback = fci["element_type"].map(season_mean)
    with np.errstate(divide="ignore", invalid="ignore"):
        pos_mean = np.where(fci["pos_e"] > 0, fci["pos_count"] / fci["pos_e"], fallback)
    fci["pos_mean"] = pos_mean
    print(
        "season per-90 means (count per 90 on the pitch):",
        {POSITION_NAMES[int(k)]: round(float(v), 4) for k, v in season_mean.items()},
    )
    return fci


def fit_group(rows: pd.DataFrame, prior_count, prior_e, pos_mean) -> tuple[float, float, float]:
    y = rows["count"].to_numpy(dtype="float64")
    e = rows["e"].to_numpy(dtype="float64")

    def nll(theta: np.ndarray) -> float:
        k, r = np.exp(theta)
        rate = (prior_count + k * pos_mean) / (prior_e + k)
        return -float(nb_loglik(y, rate * e, r).sum())

    best = None
    for start in ((0.0, 1.0), (1.5, 2.0), (3.0, 3.0)):
        result = minimize(
            nll,
            np.array(start),
            method="Nelder-Mead",
            options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 4000},
        )
        if best is None or result.fun < best.fun:
            best = result
    k, r = np.exp(best.x)
    return float(k), float(r), float(-best.fun / len(y))


def fit(args) -> None:
    rows = fit_rows(args.data_dir, args.tackles)
    groups = {"DEF": (2,), "MID": (3,), "FWD": (4,), "MID+FWD": (3, 4)}
    print("\nwalk-forward (each row from the player's earlier rows of the season):")
    for name, ets in groups.items():
        part = rows[rows["element_type"].isin(ets)]
        k, r, ll = fit_group(
            part,
            part["prior_count"].to_numpy(dtype="float64"),
            part["prior_e"].to_numpy(dtype="float64"),
            part["pos_mean"].to_numpy(dtype="float64"),
        )
        y = part["count"].to_numpy(dtype="float64")
        t = np.array([THRESHOLD[int(x)] for x in part["element_type"]])
        hit = y >= t
        rate = (part["prior_count"] + k * part["pos_mean"]) / (part["prior_e"] + k)
        mu = rate.to_numpy() * part["e"].to_numpy()
        p_hit = nbinom.sf(t - 1, r, r / (r + mu))
        poisson_ll = float(poisson.logpmf(y, mu).mean())
        brier = float(((p_hit - hit) ** 2).mean())
        print(
            f"  {name:8s} n={len(part):5d} k={k:7.3f} r={r:7.3f} mean LL={ll:.5f} "
            f"(Poisson {poisson_ll:.5f}); threshold hit rate {hit.mean():.4f}, mean P(hit) "
            f"{p_hit.mean():.4f}, Brier {brier:.5f}"
        )
        profile = []
        for kk in (0.5, 1, 2, 3, 5, 8, 12, 20, 40):
            rate = (part["prior_count"] + kk * part["pos_mean"]) / (part["prior_e"] + kk)
            mu_k = rate.to_numpy() * part["e"].to_numpy()
            res = minimize(
                lambda th, mu_k=mu_k, y=y: -float(nb_loglik(y, mu_k, float(np.exp(th[0]))).sum()),
                np.array([1.0]),
                method="Nelder-Mead",
            )
            profile.append(f"{kk}:{-res.fun / len(part):.4f}")
        print("    profile LL by k:", " ".join(profile))

    print("\nhalf-season check (GW <= 19 rows predict GW >= 20 rows):")
    first = rows[rows["gameweek"] <= FIRST_HALF_LAST_GW]
    second = rows[rows["gameweek"] > FIRST_HALF_LAST_GW]
    totals = first.groupby("player_code")[["count", "e"]].sum()
    pos_totals = first.groupby("element_type")[["count", "e"]].sum()
    pos_rate = pos_totals["count"] / pos_totals["e"]
    for name, ets in groups.items():
        part = second[second["element_type"].isin(ets)]
        pc = part["player_code"].map(totals["count"]).fillna(0).to_numpy(dtype="float64")
        pe = part["player_code"].map(totals["e"]).fillna(0).to_numpy(dtype="float64")
        pm = part["element_type"].map(pos_rate).to_numpy(dtype="float64")
        k, r, ll = fit_group(part, pc, pe, pm)
        print(f"  {name:8s} n={len(part):5d} k={k:7.3f} r={r:7.3f} mean LL={ll:.5f}")


# --- live -------------------------------------------------------------------------------------


def live_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    """The tables v1 needs, without any 2025/26 (holdout) row: filtered on read."""
    out = {}
    for name in TABLES:
        frame = pd.read_parquet(
            data_dir / f"{name}.parquet",
            columns=SNAPSHOT_COLUMNS if name == "player_snapshot" else None,
            filters=[("season", "!=", HOLDOUT)],
        )
        out[name] = frame.reset_index(drop=True)
    return out


def live(args) -> None:
    from fplopt.backtest.rules import backtest_rules
    from fplopt.features.store import DataStore
    from fplopt.models.assemble import V1Params, fit_v1, fixture_components, fixture_points
    from fplopt.models.fitted import refit_deadline

    lo, hi = (int(x) for x in args.gws.split("-"))
    tables = live_tables(args.data_dir)
    store = DataStore(tables=tables)
    gameweeks = tables["gameweek"]
    gameweeks = gameweeks[(gameweeks["season"] == LIVE) & gameweeks["gw"].between(lo, hi)]
    matches = tables["player_match"]
    matches = matches[matches["season"] == LIVE]
    played_gws = set(matches["gw"].unique())
    rules = backtest_rules(LIVE)
    params = V1Params(calibrate=False)  # horizon 0: the table's P(start) map is h1+ only
    fits: dict = {}
    out = []
    for gw in gameweeks.sort_values("gw").itertuples(index=False):
        if gw.gw not in played_gws:
            print(f"GW{gw.gw}: no outcomes yet, skipped")
            continue
        view = store.as_of(gw.deadline_time)
        cutoff = refit_deadline(view)
        if cutoff not in fits:
            fits[cutoff] = fit_v1(view.earlier(cutoff), params)
        frame = fixture_points(fixture_components(view, fits[cutoff]), rules)
        frame = frame[frame["horizon"] == 0]
        real = matches[matches["gw"] == gw.gw]
        real = real[["player_key", "fixture_key", "minutes", "defensive_contribution"]]
        frame = frame.merge(real, on=["player_key", "fixture_key"], how="left")
        threshold = frame["element_type"].map(THRESHOLD)
        count = frame["defensive_contribution"].astype("float64")
        frame = frame.assign(
            gw=gw.gw,
            hit=(count >= threshold).fillna(False).astype("float64"),  # no row: 0 minutes
            played=(frame["minutes"].fillna(0) > 0).astype("float64"),
        )
        out.append(frame)
        print(f"GW{gw.gw}: {len(frame)} player-fixtures", flush=True)
    result = pd.concat(out, ignore_index=True)
    result = result[result["element_type"].isin((2, 3, 4))].copy()
    result["position"] = result["element_type"].map(POSITION_NAMES)
    result["sq"] = (result["p_defcon"] - result["hit"]) ** 2

    def table(frame: pd.DataFrame, keys) -> pd.DataFrame:
        return frame.groupby(keys, observed=True).agg(
            n=("hit", "size"),
            predicted=("p_defcon", "mean"),
            realized=("hit", "mean"),
            brier=("sq", "mean"),
            p_play=("p_play", "mean"),
            played=("played", "mean"),
            rate=("defcon_rate", "mean"),
        )

    fmt = {"float_format": lambda v: f"{v:.4f}"}
    print("\nby position and GW (every pool player-fixture at horizon 0):")
    print(table(result, ["position", "gw"]).to_string(**fmt))
    print("\nby position, GW pooled:")
    print(table(result, "position").to_string(**fmt))
    print("\nby position, rows that played (minutes > 0):")
    print(table(result[result["played"] > 0], "position").to_string(**fmt))
    bins = pd.cut(result["p_defcon"], [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 1.0])
    print("\nby predicted P(defcon) (all positions):")
    print(table(result.assign(bin=bins), "bin").to_string(**fmt))


HOLDOUT = 2025
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


TABLES = (
    "gameweek",
    "schedule",
    "fixture_snapshot",
    "fixture",
    "player_match",
    "player_gw",
    "player_season",
    "player_snapshot",
    "team_match",
    "team_rating",
    "odds_snapshot",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    default = os.environ.get("FPLOPT_DATA_DIR", str(REPO / "data"))
    parser.add_argument("--data-dir", default=default)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify")
    ft = sub.add_parser("fit")
    ft.add_argument("--tackles", default="f_tackles", choices=("f_tackles", "f_tackles_attempted"))
    lv = sub.add_parser("live")
    lv.add_argument("--gws", default="2-7")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    args.data_dir = Path(args.data_dir)
    {"verify": verify, "fit": fit, "live": live}[args.command](args)


if __name__ == "__main__":
    main()
