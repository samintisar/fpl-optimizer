"""Command-line entry point.

`fplopt snapshot daily|tick` (archiver timers), `fplopt backfill element-summary|football-data|
vaastav|fplcache` (one-off backfills into raw/; football-data takes `--from-season YEAR`),
`fplopt rules export SEASON [--out DIR]` (config/scoring/<season>.json from an archived
bootstrap), `fplopt build TABLE|all` (raw/ -> data/<table>.parquet; no network),
`fplopt check freshness [--max-age-hours H]` (fails if the newest bootstrap snapshot is
missing or older than H hours, default 36: a dead-man's switch for the timers),
`fplopt check leakage [--deadlines N] [--seed S]` (the corrupt-the-future check of the
registered features, xP models and decision probes on data/ at a fixed list of edge deadlines
plus N sampled ones, default 12, outside the holdout), `fplopt backtest run|compare` (season
replays and paired policy comparisons on data/, written to results/ and logged to
results/experiments.csv; see `_backtest_run`, `_backtest_compare`). Every job
gets a `Context`; failures are logged and alerted to Telegram, and the exit code is 1.
Bad backtest arguments (season syntax, holdout seasons, `ep_next` before 2021/22, seasons
without data) are usage errors: exit 2, no alert.
After a successful `snapshot daily|tick`, HEALTHCHECK_PING_URL (if set) gets a best-effort
GET, for an external dead-man's switch that also notices the server being down.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import shlex
import signal
import socket
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from dotenv import load_dotenv

from fplopt.adapters.football_data import FootballDataClient
from fplopt.adapters.fpl import FplClient
from fplopt.adapters.fplcache import FplcacheClient
from fplopt.adapters.http import make_client
from fplopt.adapters.odds import OddsClient
from fplopt.adapters.vaastav import VaastavClient
from fplopt.alerts import send_admin_alert
from fplopt.heartbeat import send_heartbeat
from fplopt.ingest import health, history, jobs
from fplopt.ingest.raw_store import RawStore
from fplopt.redact import redact
from fplopt.seasons import HOLDOUT_SEASONS, parse_season_label, season_label
from fplopt.settings import Settings

if TYPE_CHECKING:
    import pandas as pd

    from fplopt.backtest.evaluate import Summary
    from fplopt.backtest.policies import Policy
    from fplopt.features.store import DataStore

log = logging.getLogger("fplopt")


@dataclass
class Context:
    """What a job needs: the raw store, a shared HTTP client, settings and parsed arguments."""

    store: RawStore
    http: httpx.Client
    settings: Settings
    args: argparse.Namespace

    @property
    def fpl(self) -> FplClient:
        return FplClient(self.http)

    @property
    def odds(self) -> OddsClient | None:
        key = self.settings.odds_api_key
        return OddsClient(self.http, key) if key else None


Job = Callable[[Context], object]

# Jobs whose success pings HEALTHCHECK_PING_URL: the scheduled archiver runs.
HEARTBEAT_JOBS = frozenset({"snapshot daily", "snapshot tick"})


# The build layer pulls in pandas and pandera (~2 s to import), which the archiver jobs that
# run every 15 minutes don't need, so it is imported only by the jobs that use it.
def _rules_export(c: Context) -> object:
    from fplopt.build import rules

    # The season label is parsed inside the job, so a bad label is alerted like any failure.
    return rules.export_rules(c.store, parse_season_label(c.args.season), Path(c.args.out))


def _build(c: Context) -> object:
    from fplopt.build import build
    from fplopt.build.common import BuildContext

    # Unknown table names fail inside the job, so they are logged and alerted (exit 1).
    return build([c.args.target], BuildContext(c.store, c.settings.data_dir))


def _check_leakage(c: Context) -> object:
    from fplopt.features.leakcheck import run_leakage_check

    return run_leakage_check(c.settings.data_dir, c.args.deadlines, c.args.seed)


# --- backtests --------------------------------------------------------------------------------
# Argument parsing and validation stay pandas-free at import time; the backtester is imported
# only by the backtest commands.

BACKTEST_SEASONS = range(2016, 2027)  # seasons with built tables and backtest rules
EP_NEXT_FIRST_SEASON = 2021  # FPL's ep_next is in the snapshots from 2020/21 GW32 on
XP_MODELS = ("rolling", "ep_next")  # fplopt.models.MODELS keys (a test checks they agree)
POLICY_PARAMS: dict[str, dict[str, type]] = {
    "greedy": {"threshold": float, "horizon": int, "decay": float, "max_transfers": int},
    "roll": {},
}
DEFAULT_STARTS = "template@1,random:5@1,random:3@20"
DEFAULT_EXPERIMENTS = Path("results/experiments.csv")
# Two-sided bootstrap CI level: its lower bound is the one-sided α = 0.10 bound of the
# go-live gate (PLAN §9).
CI_LEVEL = 0.8
_SEASON_ITEM = re.compile(r"(\d{4})(?:-(\d{4}|\d{2}))?")


def parse_seasons(text: str) -> tuple[int, ...]:
    """`2016-2024` (a range of start years, inclusive), `2021,2023`, a season label `2021-22`,
    or a mix (`2016-2018,2021-22,2023`) -> sorted unique start years. Seasons outside
    2016/17-2026/27 and holdout seasons (`HOLDOUT_SEASONS`; no override until Phase 6) are
    rejected with argparse.ArgumentTypeError."""
    seasons: set[int] = set()
    for item in (part.strip() for part in text.split(",")):
        match = _SEASON_ITEM.fullmatch(item)
        if not match:
            raise argparse.ArgumentTypeError(
                f"bad season {item!r}: expected YYYY, YYYY-YYYY (a range) or YYYY-YY (a label)"
            )
        first = last = int(match[1])
        if match[2] is not None and len(match[2]) == 2:
            try:
                first = last = parse_season_label(item)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(str(exc)) from None
        elif match[2] is not None:
            last = int(match[2])
            if last < first:
                raise argparse.ArgumentTypeError(f"bad season range {item!r}: end before start")
        seasons.update(range(first, last + 1))
    outside = sorted(s for s in seasons if s not in BACKTEST_SEASONS)
    if outside:
        raise argparse.ArgumentTypeError(
            f"no backtest data for {', '.join(map(season_label, outside))} (seasons "
            f"{season_label(BACKTEST_SEASONS[0])} to {season_label(BACKTEST_SEASONS[-1])})"
        )
    holdout = sorted(seasons & HOLDOUT_SEASONS)
    if holdout:
        raise argparse.ArgumentTypeError(
            f"{', '.join(map(season_label, holdout))} is the holdout season: backtests refuse "
            "it until Phase 6 (e.g. use 2016-2024,2026)"
        )
    return tuple(sorted(seasons))


def format_seasons(seasons: Sequence[int]) -> str:
    """(2016, 2017, 2018, 2021) -> '2016-2018,2021'."""
    runs: list[list[int]] = []
    for season in sorted(seasons):
        if runs and season == runs[-1][-1] + 1:
            runs[-1].append(season)
        else:
            runs.append([season])
    return ",".join(str(r[0]) if len(r) == 1 else f"{r[0]}-{r[-1]}" for r in runs)


@dataclass(frozen=True)
class PolicySpec:
    """A policy on the command line: `name:xp[:key=value,...]`, e.g. `greedy:rolling`,
    `roll:ep_next`, `greedy:rolling:threshold=2.0,horizon=4`."""

    name: str
    xp: str
    params: tuple[tuple[str, float | int], ...] = ()

    def __str__(self) -> str:
        params = ",".join(f"{key}={value}" for key, value in self.params)
        return f"{self.name}:{self.xp}" + (f":{params}" if params else "")

    def build(self) -> Policy:
        from fplopt.backtest.policies import GreedyPolicy, RollPolicy

        cls = GreedyPolicy if self.name == "greedy" else RollPolicy
        return cls(self.xp, **dict(self.params))


def make_policy_spec(name: str, xp: str, params: dict[str, str]) -> PolicySpec:
    """A checked PolicySpec (argparse.ArgumentTypeError on an unknown policy, model or
    parameter, a value of the wrong type, or one the policy rejects)."""
    if name not in POLICY_PARAMS:
        raise argparse.ArgumentTypeError(
            f"unknown policy {name!r} (policies: {', '.join(POLICY_PARAMS)})"
        )
    if xp not in XP_MODELS:
        models = ", ".join(XP_MODELS)
        raise argparse.ArgumentTypeError(f"unknown xP model {xp!r} (models: {models})")
    allowed = POLICY_PARAMS[name]
    values: list[tuple[str, float | int]] = []
    for key, raw in params.items():
        if key not in allowed:
            known = ", ".join(allowed) or "none"
            raise argparse.ArgumentTypeError(
                f"policy {name!r} has no parameter {key!r} (parameters: {known})"
            )
        try:
            value = allowed[key](raw)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"{name} {key}={raw!r}: not a valid {allowed[key].__name__}"
            ) from None
        if isinstance(value, float) and not math.isfinite(value):
            raise argparse.ArgumentTypeError(f"{name} {key}={raw!r}: not finite")
        values.append((key, value))
    spec = PolicySpec(name, xp, tuple(sorted(values)))
    if name == "greedy":
        horizon = dict(spec.params).get("horizon", 1)
        max_transfers = dict(spec.params).get("max_transfers", 0)
        if horizon < 1 or max_transfers < 0:
            raise argparse.ArgumentTypeError(
                f"bad policy {spec}: need horizon >= 1 and max_transfers >= 0"
            )
    return spec


def parse_policy_spec(text: str) -> PolicySpec:
    """`name:xp[:key=value,...]` -> PolicySpec (argparse.ArgumentTypeError when invalid)."""
    parts = [part.strip() for part in text.strip().split(":")]
    if len(parts) not in (2, 3) or not all(parts[:2]):
        raise argparse.ArgumentTypeError(
            f"bad policy {text!r}: expected name:xp[:key=value,...], e.g. greedy:rolling or "
            "greedy:rolling:threshold=2.0"
        )
    params: dict[str, str] = {}
    for item in parts[2].split(",") if len(parts) == 3 else ():
        key, sep, value = (piece.strip() for piece in item.partition("="))
        if not sep or not key or not value:
            raise argparse.ArgumentTypeError(f"bad policy parameter {item!r} in {text!r}")
        if key in params:
            raise argparse.ArgumentTypeError(f"duplicate policy parameter {key!r} in {text!r}")
        params[key] = value
    return make_policy_spec(parts[0], parts[1], params)


def open_data_store(data_dir: Path) -> DataStore:
    """The backtests' store over data/ (tests replace this factory with in-memory tables)."""
    from fplopt.features.store import DataStore

    return DataStore(data_dir)


def seasons_with_data(store: DataStore) -> set[int]:
    """Seasons with both gameweek and player_match rows (as of any time)."""
    from fplopt.backtest.simulator import FAR_FUTURE

    view = store.as_of(FAR_FUTURE)
    gameweeks = set(view.table("gameweek", columns=["season"])["season"].tolist())
    matches = set(view.table("player_match", columns=["season"])["season"].unique().tolist())
    return gameweeks & matches


def _run_policy_spec(args: argparse.Namespace) -> PolicySpec:
    params = {key: getattr(args, key) for key in POLICY_PARAMS["greedy"]}
    given = {key: str(value) for key, value in params.items() if value is not None}
    if given and args.policy != "greedy":
        flags = ", ".join(f"--{key.replace('_', '-')}" for key in given)
        raise argparse.ArgumentTypeError(f"{flags}: only for --policy greedy")
    return make_policy_spec(args.policy, args.xp, given)


def validate_backtest(
    parser: argparse.ArgumentParser, args: argparse.Namespace, store: DataStore
) -> None:
    """Cross-argument and data checks of `backtest run|compare`; `parser.error` (exit 2, no
    alert) on failure. Sets `args.policy_specs` and `args.start_specs`."""
    from fplopt.backtest.evaluate import parse_start_specs

    try:
        if args.command == "run":
            args.policy_specs = [_run_policy_spec(args)]
        else:
            args.policy_specs = [args.a, args.b]
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if args.command == "compare":
        # Equal specs, or specs that build the same policy (a default spelled out, e.g.
        # greedy:rolling vs greedy:rolling:threshold=1.0): run_grid needs distinct names.
        try:
            same = args.a == args.b or args.a.build().name == args.b.build().name
        except ValueError as exc:
            parser.error(f"bad policy: {exc}")
        if same:
            parser.error(f"--a and --b are the same policy ({args.a} vs {args.b})")
        if args.stride is None:
            args.stride = args.k  # non-overlapping per-decision windows
        if min(args.k, args.stride, args.block_length, args.n_boot) < 1:
            parser.error("--k, --stride, --block-length and --n-boot must be >= 1")
    try:
        args.start_specs = parse_start_specs(args.starts)
    except ValueError as exc:
        parser.error(f"--starts: {exc}")
    specs = list(args.policy_specs)
    if args.command == "compare" and args.per_decision:
        specs.append(args.continuation)
    early = [season for season in args.seasons if season < EP_NEXT_FIRST_SEASON]
    for spec in specs:
        if spec.xp == "ep_next" and early:
            parser.error(
                f"{spec} uses ep_next, which exists only from "
                f"{season_label(EP_NEXT_FIRST_SEASON)} (FPL snapshots); drop "
                f"{', '.join(map(season_label, early))}"
            )
    try:
        available = seasons_with_data(store)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    missing = sorted(set(args.seasons) - available)
    if missing:
        parser.error(
            f"no gameweek/player_match data for {', '.join(map(season_label, missing))} "
            "(run fplopt build all?)"
        )


def _out_dir(args: argparse.Namespace) -> Path:
    if args.out:
        out = Path(args.out)
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out = Path("results") / f"{stamp}-{args.command}"
    return out


def _backtest_config(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    config: dict[str, Any] = {
        "seasons": list(args.seasons),
        "starts": args.starts,
        "policies": [str(spec) for spec in args.policy_specs],
        "out": str(out),
    }
    if args.command == "compare":
        config |= {
            "per_decision": args.per_decision,
            "continuation": str(args.continuation) if args.per_decision else None,
            "k": args.k,
            "stride": args.stride,
            "block_length": args.block_length,
            "n_boot": args.n_boot,
            "seed": args.seed,
            "ci": CI_LEVEL,
        }
    return config


def _jsonable(value: Any) -> Any:
    """NaN/inf -> None, recursively (summary.json is strict JSON)."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _jsonable(v) for key, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(_jsonable(payload), indent=2, default=str, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")


def _num(value: Any, digits: int = 1, sign: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{float(value):{'+' if sign else ''}.{digits}f}"


def text_table(headers: Sequence[str], rows: Sequence[Sequence[str]], left: int = 1) -> str:
    """A plain aligned table: the first `left` columns left-aligned, the rest right-aligned."""
    widths = [max(len(c) for c in column) for column in zip(headers, *rows, strict=True)]

    def line(cells: Sequence[str]) -> str:
        out = [
            cell.ljust(width) if i < left else cell.rjust(width)
            for i, (cell, width) in enumerate(zip(cells, widths, strict=True))
        ]
        return "  ".join(out).rstrip()

    return "\n".join([line(headers), *(line(row) for row in rows)])


TOTALS_TITLE = (
    "Season totals by start GW (net points; the template start, mean/min/max over starts; "
    "a GW1 start refused there runs from GW2)"
)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def start_gw_of(start_specs: Sequence[Any]) -> dict[str, int]:
    """start_id -> the gw_index its spec asked for (a gw_index-1 start that fell back to
    gw_index 2 keeps 1), so full-season and mid-season starts are reported apart."""
    out: dict[str, int] = {}
    for spec in start_specs:
        out[spec.start_id()] = spec.gw_index
        if spec.gw_index == 1:
            out.setdefault(spec.start_id(2), 1)
    return out


def season_totals_by_start(results: pd.DataFrame, start_specs: Sequence[Any]) -> pd.DataFrame:
    """Per (season, policy, start GW): starts, mean GWs played, the template start's total
    (null without one), mean/min/max total over starts, mean xG total (null if any start's
    has a null GW), GWs with a null xG score, transfers, hit points (means per start) and
    captain and XI regret per GW (means over starts)."""
    import numpy as np

    frame = results.assign(start_gw=results["start_id"].map(start_gw_of(start_specs)))
    runs = frame.groupby(["season", "policy", "start_gw", "start_id"], sort=True).agg(
        total=("net_points", "sum"),
        xg_total=("xg_net_points", lambda s: s.sum() if s.notna().all() else np.nan),
        xg_null=("xg_net_points", lambda s: s.isna().sum()),
        n_gws=("gw_index", "size"),
        n_transfers=("n_transfers", "sum"),
        hit_points=("hit_points", "sum"),
        captain_regret=("captain_regret", "mean"),
        xi_regret=("xi_regret", "mean"),
    )
    runs = runs.astype("float64").reset_index()
    runs["template"] = runs["total"].where(runs["start_id"].str.startswith("template"))
    out = runs.groupby(["season", "policy", "start_gw"], sort=True).agg(
        starts=("start_id", "size"),
        n_gws=("n_gws", "mean"),
        template=("template", "max"),
        mean=("total", "mean"),
        min=("total", "min"),
        max=("total", "max"),
        xg_mean=("xg_total", lambda s: s.mean() if s.notna().all() else np.nan),
        xg_null=("xg_null", "mean"),
        n_transfers=("n_transfers", "mean"),
        hit_points=("hit_points", "mean"),
        captain_regret=("captain_regret", "mean"),
        xi_regret=("xi_regret", "mean"),
    )
    return out.reset_index()


def season_table(totals: pd.DataFrame) -> str:
    """`season_totals_by_start` as text."""
    headers = ["season", "policy", "from GW", "starts", "GWs", "template", "mean", "min", "max"]
    headers += ["xG mean", "null xG GWs", "transfers", "hit pts", "capt regret", "XI regret"]
    rows = [
        [
            season_label(int(r.season)),
            str(r.policy),
            str(int(r.start_gw)),
            str(int(r.starts)),
            _num(r.n_gws),
            _num(r.template, 0),
            _num(r.mean),
            _num(r.min, 0),
            _num(r.max, 0),
            _num(r.xg_mean),
            _num(r.xg_null),
            _num(r.n_transfers),
            _num(r.hit_points),
            _num(r.captain_regret, 2),
            _num(r.xi_regret, 2),
        ]
        for r in totals.itertuples(index=False)
    ]
    return text_table(headers, rows, left=2)


def overall_table(totals: pd.DataFrame) -> str:
    """Per (policy, start GW), means over seasons of `season_totals_by_start`; the xG mean
    over the seasons where every policy's xG mean (for that start GW) is defined, counted in
    `xG seasons`, so policies are compared on the same seasons."""
    xg_seasons = {}
    for start_gw, part in totals.groupby("start_gw"):
        wide = part.pivot(index="season", columns="policy", values="xg_mean")
        xg_seasons[start_gw] = set(wide.index[wide.notna().all(axis=1)])
    groups = totals.groupby(["policy", "start_gw"], sort=True)
    headers = ["policy", "from GW", "seasons", "template", "mean", "xG mean", "xG seasons"]
    headers += ["transfers", "capt regret", "XI regret"]
    rows = [
        [
            str(policy),
            str(int(start_gw)),
            str(len(part)),
            _num(part["template"].mean()),
            _num(part["mean"].mean()),
            _num(part.loc[part["season"].isin(xg_seasons[start_gw]), "xg_mean"].mean()),
            str(len(xg_seasons[start_gw])),
            _num(part["n_transfers"].mean()),
            _num(part["captain_regret"].mean(), 2),
            _num(part["xi_regret"].mean(), 2),
        ]
        for (policy, start_gw), part in groups
    ]
    return text_table(headers, rows, left=2)


def comparison_table(summary: Summary, k: int, stride: int, ci: float = CI_LEVEL) -> str:
    """The paired comparisons of `summarize`: mean difference (per GW for the full run, per
    k-GW decision window for per-decision), `ci` CI, one-sided p, sample (seasons and cells:
    GWs or decision windows), difference in season points."""
    headers = ["method", "metric", "mean diff", f"{ci:.0%} CI", "p", "seasons", "cells"]
    headers += ["per season"]
    rows = []
    for r in summary.comparisons.itertuples(index=False):
        method = "full run"
        if r.method == "per_decision":
            method = f"per-decision (k={k}, stride={stride})"
        rows.append(
            [
                method,
                str(r.metric),
                _num(r.mean, 3, True),
                f"[{_num(r.ci_low, 3, True)}, {_num(r.ci_high, 3, True)}]",
                _num(r.p_one_sided, 3),
                str(int(r.n_seasons)),
                str(int(r.n_gws)),
                _num(r.season_diff, 1, True),
            ]
        )
    return text_table(headers, rows, left=2)


def _per_season(diffs: pd.DataFrame, column: str) -> pd.DataFrame:
    """Per season: sum and mean over GWs of the start-averaged difference (nulls dropped)."""
    valid = diffs[diffs[column].notna()]
    cells = valid.groupby(["season", "gw_index"])[column].mean().astype("float64")
    return cells.groupby(level="season").agg(["sum", "mean"])


def per_season_diff_table(paired: pd.DataFrame, decisions: pd.DataFrame | None, k: int) -> str:
    """Per season: the full-run difference (season total and per GW realized, per GW xG) and
    the per-decision difference (season points — `season_points` — and per decision window,
    realized; xG per window) with the share of decisions at which A and B differed."""
    from fplopt.backtest.evaluate import season_points

    full = _per_season(paired, "diff")
    full_xg = _per_season(paired, "diff_xg")["mean"]
    headers = ["season", "full run total", "per GW", "xG per GW"]
    if decisions is not None:
        headers += [f"per-decision total (k={k})", "per window", "xG per window"]
        headers += ["decisions differ"]
        dec_total = season_points(decisions, "diff")
        dec = _per_season(decisions, "diff")["mean"]
        dec_xg = _per_season(decisions, "diff_xg")["mean"]
        differ = 1 - decisions.groupby("season")["same_decision"].mean()
    rows = []
    for season in full.index:
        row = [
            season_label(int(season)),
            _num(full.loc[season, "sum"], 1, True),
            _num(full.loc[season, "mean"], 3, True),
            _num(full_xg.get(season), 3, True),
        ]
        if decisions is not None:
            share = differ.get(season)
            row += [
                _num(dec_total.get(season), 1, True),
                _num(dec.get(season), 3, True),
                _num(dec_xg.get(season), 3, True),
                "-" if share is None else f"{share:.0%}",
            ]
        rows.append(row)
    return text_table(headers, rows)


def _starts_by_season(results: pd.DataFrame) -> dict[str, list[str]]:
    starts = results.groupby("season")["start_id"].unique()
    return {season_label(int(s)): sorted(ids) for s, ids in starts.items()}


def _experiment_metrics(summary: Summary) -> dict[str, Any]:
    """The experiment log's metrics: per-policy means, season totals, comparisons."""
    tables = summary.to_dict()
    season_totals = {
        f"{season_label(int(r.season))} {r.policy}": round(float(r.total), 2)
        for r in summary.season_totals.itertuples(index=False)
    }
    return _jsonable(
        {
            "policies": tables["policies"],
            "season_totals": season_totals,
            "comparisons": tables["comparisons"],
        }
    )


def _backtest_store(c: Context) -> DataStore:
    store = getattr(c.args, "store", None)
    return store if store is not None else open_data_store(c.settings.data_dir)


def _backtest_run(c: Context) -> object:
    """`fplopt backtest run`: one policy from every start state of every season (`run_grid`);
    writes gws.parquet and summary.json to --out, prints the season table and appends the
    run to the experiment log."""
    from fplopt.backtest.evaluate import log_experiment, run_grid, summarize

    args = c.args
    store = _backtest_store(c)
    out = _out_dir(args)
    (policy,) = [spec.build() for spec in args.policy_specs]
    began = time.perf_counter()
    results = run_grid(store, args.seasons, args.start_specs, [policy])
    if results.empty:
        raise RuntimeError("no start state could be built in any season")
    summary = summarize(results)
    runtime = time.perf_counter() - began
    totals = season_totals_by_start(results, args.start_specs)
    config = _backtest_config(args, out)
    out.mkdir(parents=True, exist_ok=True)  # only now: a failed run leaves nothing behind
    results.to_parquet(out / "gws.parquet", index=False)
    payload = {
        "command": args.command_line,
        "config": config,
        "policy": policy.name,
        "runtime_seconds": round(runtime, 1),
        "starts": _starts_by_season(results),
        "season_totals_by_start": _records(totals),
        "summary": summary.to_dict(),
    }
    _write_json(out / "summary.json", payload)
    print(
        f"{policy.name}, seasons {format_seasons(args.seasons)}, starts {args.starts} "
        f"({runtime:.0f}s)\n\n{TOTALS_TITLE}\n{season_table(totals)}\n\n"
        f"Overall (means over seasons)\n{overall_table(totals)}\n\nWritten to {out}",
        flush=True,
    )
    log_experiment(args.experiments, args.command_line, config, _experiment_metrics(summary), 1)
    return summary


def _backtest_compare(c: Context) -> object:
    """`fplopt backtest compare`: policies A and B from the same start states (`run_grid`),
    the paired full-run and (unless --no-per-decision) per-decision differences with
    GW-block bootstrap CIs (`summarize`). Writes gws.parquet, paired.parquet,
    per_decision.parquet and summary.json, prints the tables and appends the run to the
    experiment log (2 variants)."""
    from fplopt.backtest.evaluate import (
        log_experiment,
        paired_full_run,
        per_decision,
        run_grid,
        summarize,
    )
    from fplopt.backtest.simulator import Caches

    args = c.args
    store = _backtest_store(c)
    out = _out_dir(args)
    a, b = (spec.build() for spec in args.policy_specs)
    continuation = args.continuation.build() if args.per_decision else None
    caches = Caches()
    began = time.perf_counter()
    results = run_grid(store, args.seasons, args.start_specs, [a, b], caches=caches)
    if results.empty:
        raise RuntimeError("no start state could be built in any season")
    paired = paired_full_run(results, a, b)
    decisions = None
    if continuation is not None:
        decisions = per_decision(
            store,
            args.seasons,
            args.start_specs,
            a,
            b,
            continuation,
            k=args.k,
            stride=args.stride,
            caches=caches,
        )
    summary = summarize(
        results,
        [(a, b)],
        decisions,
        block_length=args.block_length,
        n_boot=args.n_boot,
        seed=args.seed,
        ci=CI_LEVEL,
    )
    runtime = time.perf_counter() - began
    totals = season_totals_by_start(results, args.start_specs)
    config = _backtest_config(args, out)
    out.mkdir(parents=True, exist_ok=True)  # only now: a failed run leaves nothing behind
    results.to_parquet(out / "gws.parquet", index=False)
    paired.to_parquet(out / "paired.parquet", index=False)
    if decisions is not None:
        decisions.to_parquet(out / "per_decision.parquet", index=False)
    payload = {
        "command": args.command_line,
        "config": config,
        "policy_a": a.name,
        "policy_b": b.name,
        "continuation": None if continuation is None else continuation.name,
        "runtime_seconds": round(runtime, 1),
        "starts": _starts_by_season(results),
        "season_totals_by_start": _records(totals),
        "summary": summary.to_dict(),
    }
    _write_json(out / "summary.json", payload)
    cont = "" if continuation is None else f", continuation {continuation.name}"
    print(
        f"A = {a.name}, B = {b.name}, seasons {format_seasons(args.seasons)}, starts "
        f"{args.starts}{cont} ({runtime:.0f}s)\n\n"
        f"{TOTALS_TITLE}\n{season_table(totals)}\n\n"
        f"Overall (means over seasons)\n{overall_table(totals)}\n\n"
        f"Paired differences A - B (full run: mean per GW; per-decision: mean per {args.k}-GW "
        f"decision window, decisions at gw_index 1 + {args.stride}j; {CI_LEVEL:.0%} CI, two-sided "
        f"(lower bound = one-sided alpha {(1 - CI_LEVEL) / 2:.2f}), from a GW-block bootstrap "
        f"by season, blocks of {args.block_length} GWs, {args.n_boot} resamples; p "
        "one-sided, H1: A > B; realized@xg = realized points on the xG metric's sample; "
        "cells = GWs or decision windows; per season = difference in season points)\n"
        f"{comparison_table(summary, args.k, args.stride)}\n\nPer season (A - B)\n"
        f"{per_season_diff_table(paired, decisions, args.k)}\n\nWritten to {out}",
        flush=True,
    )
    log_experiment(args.experiments, args.command_line, config, _experiment_metrics(summary), 2)
    return summary


JOBS: dict[str, Job] = {
    "snapshot daily": lambda c: jobs.run_daily(
        c.store, c.fpl, c.odds, football_data=FootballDataClient(c.http)
    ),
    "snapshot tick": lambda c: jobs.run_tick(c.store, c.fpl, c.odds),
    "backfill element-summary": lambda c: jobs.backfill_element_summaries(c.store, c.fpl),
    "backfill football-data": lambda c: history.backfill_football_data(
        c.store, FootballDataClient(c.http), first_season=c.args.from_season
    ),
    "backfill vaastav": lambda c: history.backfill_vaastav(c.store, VaastavClient(c.http)),
    "backfill fplcache": lambda c: history.backfill_fplcache(c.store, FplcacheClient(c.http)),
    "rules export": _rules_export,
    "build": _build,
    "check freshness": lambda c: health.check_freshness(
        c.store, jobs.utc_now(), timedelta(hours=c.args.max_age_hours)
    ),
    "check leakage": _check_leakage,
    "backtest run": _backtest_run,
    "backtest compare": _backtest_compare,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fplopt")
    groups = parser.add_subparsers(dest="group", required=True)
    snapshot = groups.add_parser("snapshot", help="archive API snapshots into raw/")
    snapshot.add_argument("command", choices=["daily", "tick"])
    backfill = groups.add_parser("backfill", help="one-off backfills into raw/")
    backfill.add_argument(
        "command", choices=["element-summary", "football-data", "vaastav", "fplcache"]
    )
    backfill.add_argument(
        "--from-season",
        type=int,
        default=history.FIRST_SEASON,
        metavar="YEAR",
        help=f"football-data only: first season's start year (default {history.FIRST_SEASON})",
    )
    rules_group = groups.add_parser("rules", help="per-season rules config")
    rules_group.add_argument("command", choices=["export"])
    rules_group.add_argument("season", help="season label, e.g. 2026-27")
    rules_group.add_argument(
        "--out", default="config/scoring", help="output directory (default: config/scoring)"
    )
    build_group = groups.add_parser("build", help="build data/<table>.parquet from raw/")
    build_group.add_argument("target", help="table name, or 'all'")
    check = groups.add_parser("check", help="archive health and leakage checks")
    check.add_argument("command", choices=["freshness", "leakage"])
    check.add_argument(
        "--max-age-hours",
        type=float,
        default=health.DEFAULT_MAX_AGE.total_seconds() / 3600,
        metavar="H",
        help="fail if the newest bootstrap snapshot is older than this (default 36)",
    )
    check.add_argument(
        "--deadlines",
        type=int,
        default=12,
        metavar="N",
        help="leakage only: random GW deadlines to check besides the fixed edges (default 12)",
    )
    check.add_argument(
        "--seed", type=int, default=0, metavar="S", help="leakage only: corruption seed"
    )
    _add_backtest_parsers(groups)
    return parser


def _add_backtest_parsers(groups: Any) -> None:
    backtest = groups.add_parser("backtest", help="season replays and paired policy comparisons")
    commands = backtest.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="one policy over seasons and start states")
    compare = commands.add_parser("compare", help="paired comparison of two policies")
    for sub in (run, compare):
        sub.set_defaults(subparser=sub)  # for usage errors found after parsing
        sub.add_argument(
            "--seasons",
            type=parse_seasons,
            required=True,
            help="start years: 2016-2024, 2021,2023, 2021-22 or a mix (the holdout is refused)",
        )
        sub.add_argument(
            "--starts", default=DEFAULT_STARTS, help=f"start states (default {DEFAULT_STARTS})"
        )
        sub.add_argument(
            "--out", default=None, help="output directory (default results/<UTC time>-<command>)"
        )
        sub.add_argument(
            "--experiments",
            type=Path,
            default=DEFAULT_EXPERIMENTS,
            help=f"experiment log to append to (default {DEFAULT_EXPERIMENTS.as_posix()})",
        )
    run.add_argument("--policy", choices=list(POLICY_PARAMS), default="greedy")
    run.add_argument("--xp", choices=list(XP_MODELS), default="rolling", help="xP model")
    run.add_argument("--threshold", type=float, help="greedy: minimum gain (default 1.0)")
    run.add_argument("--horizon", type=int, help="greedy: horizon in GWs (default 6)")
    run.add_argument("--decay", type=float, help="greedy: discount per GW (default 0.85)")
    run.add_argument("--max-transfers", type=int, help="greedy: transfers per GW (default 1)")
    spec_help = "policy name:xp[:key=value,...], e.g. greedy:rolling:threshold=2.0"
    compare.add_argument("--a", type=parse_policy_spec, required=True, help=f"A: {spec_help}")
    compare.add_argument("--b", type=parse_policy_spec, required=True, help=f"B: {spec_help}")
    compare.add_argument(
        "--continuation",
        type=parse_policy_spec,
        default="roll:rolling",
        help="per-decision continuation policy (default roll:rolling)",
    )
    compare.add_argument("--k", type=int, default=4, help="per-decision window in GWs (4)")
    compare.add_argument(
        "--stride",
        type=int,
        default=None,
        help="per-decision windows start at gw_index 1, 1 + stride, ... for every start state "
        "(default k: windows don't overlap; 1 = every GW, a diagnostic)",
    )
    compare.add_argument(
        "--per-decision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also run the per-decision evaluation (default on)",
    )
    compare.add_argument("--block-length", type=int, default=4, help="bootstrap block (4 GWs)")
    compare.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples (2000)")
    compare.add_argument("--seed", type=int, default=0, help="bootstrap seed (0)")


def job_name(args: argparse.Namespace) -> str:
    """The JOBS key for parsed arguments: '<group> <command>', or just '<group>' for groups
    without a command (build)."""
    command = getattr(args, "command", None)
    return f"{args.group} {command}" if command else args.group


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # httpx logs full request URLs at INFO, which would expose API keys and bot tokens.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _exit_on_sigterm(signum: int, frame: object) -> None:
    raise SystemExit(143)  # 128 + SIGTERM, the status a shell reports for a TERM-killed process


@contextmanager
def _sigterm_raises_system_exit() -> Iterator[None]:
    """Turn SIGTERM (systemd stop or TimeoutStartSec) into SystemExit while a job runs, so its
    `finally` blocks still write run manifests; the default action kills the process without
    unwinding. Restores the previous handler afterwards. A no-op where there is no SIGTERM
    or off the main thread (signal handlers can only be set there)."""
    sigterm = getattr(signal, "SIGTERM", None)
    if sigterm is None or threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.signal(sigterm, _exit_on_sigterm)
    try:
        yield
    finally:
        signal.signal(sigterm, previous)


def main(argv: Sequence[str] | None = None, settings: Settings | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    args.command_line = shlex.join(["fplopt", *argv])
    name = job_name(args)
    if settings is None:
        load_dotenv(Path.cwd() / ".env")
        settings = Settings.from_env()
    configure_logging()
    if args.group == "backtest":
        # Usage errors (exit 2, no alert); needs the data to check which seasons exist.
        args.store = open_data_store(settings.data_dir)
        validate_backtest(args.subparser, args, args.store)
    if not (settings.telegram_bot_token and settings.telegram_admin_chat_id):
        log.warning(
            "TELEGRAM_BOT_TOKEN/TELEGRAM_ADMIN_CHAT_ID not set: failures will not be alerted"
        )
    log.info("job %r starting (raw dir %s)", name, settings.raw_dir)
    try:
        store = RawStore(settings.raw_dir)
        with make_client() as http, _sigterm_raises_system_exit():
            JOBS[name](Context(store=store, http=http, settings=settings, args=args))
    except Exception as exc:
        log.exception("job %r failed", name)
        summary = str(exc).splitlines()[0] if str(exc) else ""
        send_admin_alert(
            redact(
                f"fplopt {name} failed on {socket.gethostname()}: {type(exc).__name__}: {summary}"
            ),
            token=settings.telegram_bot_token,
            chat_id=settings.telegram_admin_chat_id,
        )
        return 1
    log.info("job %r done", name)
    if name in HEARTBEAT_JOBS and settings.healthcheck_ping_url:
        send_heartbeat(settings.healthcheck_ping_url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
