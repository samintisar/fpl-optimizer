"""Differential check of our MILP planner against open-fpl-solver on no-chip cases (Phase 4
plan, Task 4; PLAN §7 *Reference check*). Dev only: never imported by `fplopt`.

open-fpl-solver (solioanalytics/open-fpl-solver, Apache-2.0) is cloned outside this repo
(never vendored) and run in its own venv (`uv sync` in the clone). We bypass its data layer
(projection CSV, FPL API) and call `dev.solver.solve_multi_period_fpl(data, options)`
directly with a `data` dict built from our `PlanInput`, so both solvers see the same
candidates, xP, prices, selling prices, bank and free transfers.

Two modes, one file:
- `uv run python dev/reference_check.py --data-dir <data/>` (our venv): builds instances from
  real deadlines, solves them with `solve_plan`, writes them to JSON, runs this file in
  driver mode under the clone's venv (one subprocess for all instances), compares, prints
  a table.
- `--driver IN OUT` (the clone's venv, started by the first mode): stdlib + pandas +
  open-fpl-solver only, no `fplopt` import.

Mapping (ours -> theirs):
- players: `merged_data` indexed by player_key; club `name` = str(team_key); xP per GW as
  `{w}_Pts` with w = gw_index + horizon offset (their GW numbers; the horizon must be
  contiguous and end by GW38); `{w}_xMins` = 90 (only used by options we leave off).
- money in tenths of £m on both sides (their `buy_price`, `sell_price`, `itb` are unit-free;
  `itb_value` is divided by 10 to stay per £m), so budgets are exact integers.
- selling prices: their `price_modified_players` (first sale at `sell_price`) = owned players
  whose selling price differs from the buy price. An owned player who left the game (not
  buyable for us) gets a prohibitive buy price (1e6), which makes him price-modified, keeps
  his first sale at his selling price, and makes buying him back impossible, as ours.
- FTs: `ft` = our free_transfers, `ft_base` 1. Their FT states are 0..5 with
  V(s) = Σ_{n=0..s} v[n], v[n] = `ft_value_list[str(n)]` or the scalar `ft_value`. We pass
  `ft_value_list` = our ft_value for n = 2..5 (0 where missing) and leave n = 0, 1 to their
  scalar default 1.5 (unless ours has a key 1), so V_theirs(s) − V_ours(s) is a constant c
  for every reachable state s ≥ 1. The objectives then differ by the constant
  Σ_t decay^h_t (c_t − c_{t−1}) (c_0 from the first GW's states, c_t = c after), computed per
  instance and subtracted (column `offset`): 3.0 with our default ft_value.
- GW1 (`gw_index == 1`, transfers unlimited and free, next GW 1 FT): their `use_wc = [GW]`
  (transfers free, `fts[GW] = ft_base` = 1, next GW `fts = fts` = 1), the only chip used,
  as a mapping of the rule. Our `ft[0]` there is the state's 0, so c_0 uses V_ours(0).
- settings: horizon (number of horizon GWs), `decay_base`, `bench_weights`, `hit_cost` =
  rules.hit_cost + hit_margin, `vcap_weight` 0 (we do not model the vice), `ft_use_penalty`
  and `itb_loss_per_transfer` off, `chip_limits` all 0, `objective` "decay", `gap` = the
  same `mip_rel_gap` as ours, `random_seed` 0. Team limit, squad and lineup sizes are their
  hard-coded FPL constants (asserted equal to our rules).

Not mappable (checked per instance and reported instead of compared): a club over the
team limit at the deadline (theirs allows the excess only in a GW without any transfer,
ours as long as nobody of that club is bought); FT top-ups (none in our rules); a horizon
that is not contiguous in gw_index.

Comparison per instance: objective (theirs − offset vs ours; agreement when the difference
is within the two final relative MIP gaps, plus 1e-4), first-GW transfer sets, captain and
XI xP (with captain) per GW. When first-GW transfers differ, ours is re-solved with their
first-GW transfers fixed (`solve_plan(fix_first_gw=...)`): equal objectives mean an
alternative optimum, not a model difference.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
THEIR_FT_SCALAR = 1.5  # open-fpl-solver's default `ft_value` (FT states 0 and 1 by default)
FT_STATES = range(0, 6)  # open-fpl-solver's FT states
THEIR_SQUAD, THEIR_LINEUP, THEIR_TEAM_LIMIT, THEIR_MAX_GW = 15, 11, 3, 38
NOT_BUYABLE_PRICE = 1_000_000  # tenths of £m: an owned player who left the game
POSITION_SHORT = {1: ("G", "GKP"), 2: ("D", "DEF"), 3: ("M", "MID"), 4: ("F", "FWD")}
EPS = 1e-4  # absolute slack on objective comparisons (float sums, LP tolerances)


# --- instances --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Spec:
    """One instance: the decision at (season, gw_index) from a start state built at
    `gw_index − roll` (template or random:seed) and rolled forward `roll` GWs with
    `RollPolicy` (no transfers, so FTs bank and prices drift from purchase prices)."""

    season: int
    gw_index: int
    start: str  # "template" | "random:<seed>"
    xp: str  # fplopt.models.MODELS key
    horizon: int
    roll: int = 0
    prune: bool = True

    @property
    def name(self) -> str:
        roll = f"+roll{self.roll}" if self.roll else ""
        prune = "" if self.prune else " full"
        return (
            f"{self.season}/{(self.season + 1) % 100:02d} gw{self.gw_index} "
            f"{self.start}{roll} {self.xp} h{self.horizon}{prune}"
        )


# Seasons 2021/22-2024/25 and 2026/27 (2025/26 is the holdout); template and random starts;
# both xP models; 1- and 6-GW horizons; banked FTs (roll) and pruning off on some.
SPECS: tuple[Spec, ...] = (
    Spec(2021, 1, "template", "ep_next", 6),
    Spec(2021, 8, "template", "ep_next", 1),
    Spec(2021, 8, "template", "ep_next", 6),
    Spec(2021, 20, "random:1", "rolling", 6, roll=3),
    Spec(2021, 27, "random:11", "ep_next", 6, roll=1, prune=False),
    Spec(2022, 5, "random:2", "ep_next", 6),
    Spec(2022, 15, "template", "rolling", 1, roll=2),
    Spec(2022, 15, "template", "rolling", 6, roll=2),
    Spec(2022, 30, "random:3", "rolling", 6, roll=4),
    Spec(2023, 10, "template", "rolling", 6, prune=False),
    Spec(2023, 25, "random:4", "ep_next", 6, roll=1),
    Spec(2023, 25, "random:4", "ep_next", 1, roll=1),
    Spec(2023, 36, "template", "ep_next", 6),
    Spec(2024, 4, "template", "ep_next", 6),
    Spec(2024, 12, "random:5", "rolling", 6, roll=3),
    Spec(2024, 22, "random:6", "ep_next", 1, roll=2),
    Spec(2024, 22, "random:6", "ep_next", 6, roll=2),
    Spec(2024, 33, "template", "rolling", 6, prune=False),
    Spec(2026, 1, "template", "ep_next", 1),
    Spec(2026, 3, "template", "ep_next", 6),
    Spec(2026, 5, "random:7", "rolling", 6, roll=2),
    Spec(2026, 6, "random:8", "ep_next", 1),
    Spec(2026, 6, "random:8", "rolling", 6, roll=4),
    Spec(2022, 22, "random:9", "ep_next", 6, roll=6),
)
QUICK_SPECS: tuple[Spec, ...] = (
    Spec(2023, 25, "random:4", "ep_next", 1, roll=1),
    Spec(2024, 4, "template", "ep_next", 2),
    Spec(2026, 6, "random:8", "rolling", 2, roll=4),
)


@dataclass
class Instance:
    spec: Spec
    problem: object  # fplopt.optimize.PlanInput
    params: object  # fplopt.optimize.OptimizerParams
    unmappable: list[str] = field(default_factory=list)


def _ours():
    """Our imports, only in our venv (the driver mode must not need fplopt)."""
    from fplopt.backtest import rules as rules_mod
    from fplopt.backtest import simulator, start_states
    from fplopt.backtest.policies import RollPolicy
    from fplopt.features.baseline import player_pool
    from fplopt.features.store import DataStore
    from fplopt.models import MODELS
    from fplopt.optimize import OptimizerParams, PlanInput
    from fplopt.optimize.model import solve_plan
    from fplopt.seasons import HOLDOUT_SEASONS

    return dict(
        backtest_rules=rules_mod.backtest_rules,
        season_schedule=simulator.season_schedule,
        simulate=simulator.simulate,
        Caches=simulator.Caches,
        template_state=start_states.template_state,
        random_state=start_states.random_state,
        RollPolicy=RollPolicy,
        player_pool=player_pool,
        DataStore=DataStore,
        MODELS=MODELS,
        OptimizerParams=OptimizerParams,
        PlanInput=PlanInput,
        solve_plan=solve_plan,
        HOLDOUT_SEASONS=HOLDOUT_SEASONS,
    )


def build_instances(data_dir: Path, specs, mip_gap: float, time_limit: float) -> list[Instance]:
    """Instances from the real data (read-only): start state, roll, pool, xP, PlanInput."""
    o = _ours()
    store = o["DataStore"](data_dir)
    caches = o["Caches"]()
    out = []
    for spec in specs:
        if spec.season in o["HOLDOUT_SEASONS"]:
            raise ValueError(f"{spec.name}: holdout season")
        rules = o["backtest_rules"](spec.season)
        start_gw = spec.gw_index - spec.roll
        schedule = o["season_schedule"](store, spec.season, start_gw)
        deadline = schedule.loc[schedule["gw_index"] == start_gw, "deadline_time"].iloc[0]
        view = store.as_of(deadline)
        if spec.start == "template":
            state = o["template_state"](view, rules)
        else:
            state = o["random_state"](view, rules, int(spec.start.split(":")[1]))
        if spec.roll:
            run = o["simulate"](
                store,
                rules,
                o["RollPolicy"](spec.xp),
                state,
                end_gw_index=spec.gw_index - 1,
                caches=caches,
            )
            state = run.final_state
        assert state.gw_index == spec.gw_index, (spec, state.gw_index)
        deadline = schedule.loc[schedule["gw_index"] == spec.gw_index, "deadline_time"].iloc[0]
        view = store.as_of(deadline)
        pool = caches.pool(store, view)
        xp = caches.xp(store, spec.xp, view)
        defaults = o["OptimizerParams"]()
        params = o["OptimizerParams"](
            horizon=spec.horizon,
            prune_n=defaults.prune_n if spec.prune else None,
            prune_dominated=spec.prune,
            mip_gap=mip_gap,
            time_limit=time_limit,
        )
        problem = o["PlanInput"].from_context(state, pool, xp, rules, params)
        out.append(Instance(spec, problem, params, unmappable(problem)))
    return out


def unmappable(problem) -> list[str]:
    """Reasons this instance cannot be compared like for like (module docstring)."""
    rules, reasons = problem.rules, []
    clubs: dict[int, int] = {}
    for p in problem.players:
        if p.owned:
            clubs[p.team_key] = clubs.get(p.team_key, 0) + 1
    over = sorted(c for c, n in clubs.items() if n > rules.team_limit)
    if over:
        reasons.append(f"club(s) {over} over the team limit (club-cap conventions differ)")
    first, last = problem.gws[0].gw_index, problem.gws[-1].gw_index
    if [g.horizon for g in problem.gws] != list(range(len(problem.gws))) or [
        g.gw_index for g in problem.gws
    ] != list(range(first, first + len(problem.gws))):
        reasons.append("horizon not contiguous in gw_index")
    if last > THEIR_MAX_GW:
        reasons.append(f"horizon ends at gw_index {last} > {THEIR_MAX_GW}")
    if rules.ft_topups:
        reasons.append("FT top-ups (not modelled by open-fpl-solver)")
    if (
        rules.squad_size != THEIR_SQUAD
        or rules.squad_play != THEIR_LINEUP
        or rules.team_limit != THEIR_TEAM_LIMIT
        or rules.max_free_transfers != max(FT_STATES)
    ):
        reasons.append("squad/lineup/team-limit/FT-cap constants differ from theirs")
    return reasons


# --- mapping ----------------------------------------------------------------------------------


def their_ft_values(ft_value: dict[int, float]) -> dict[str, float]:
    """`ft_value_list` for open-fpl-solver: ours for n = 2..5 (0 if missing), n = 1 only if
    ours has it; n = 0 (and 1) fall back to their scalar (module docstring)."""
    values = {str(n): float(ft_value.get(n, 0.0)) for n in range(2, max(FT_STATES) + 1)}
    if 1 in ft_value:
        values["1"] = float(ft_value[1])
    return values


def their_state_value(values: dict[str, float], s: int) -> float:
    """open-fpl-solver's `ft_state_value[s]` = Σ_{n=0..s} (list value or scalar)."""
    return sum(values.get(str(n), THEIR_FT_SCALAR) for n in range(0, s + 1))


def objective_offset(problem, params) -> float:
    """theirs − ours from the FT-state values alone: Σ_t decay^h_t (c_t − c_{t−1})."""
    values = their_ft_values(dict(params.ft_value))
    diffs = {s: their_state_value(values, s) - params.ft_state_value(s) for s in range(1, 6)}
    c = diffs[1]
    if any(abs(d - c) > EPS for d in diffs.values()):
        raise AssertionError(f"FT-state values differ by a non-constant: {diffs}")
    ours_ft0 = 0 if problem.gw1 else max(problem.free_transfers, 0)
    theirs_ft0 = 1 if problem.gw1 else max(problem.free_transfers, 0)
    c0 = their_state_value(values, theirs_ft0) - params.ft_state_value(ours_ft0)
    offset = c0
    if len(problem.gws) > 1:
        offset += params.decay ** problem.gws[1].horizon * (c - c0)
    return offset


def export(inst: Instance) -> dict:
    """The JSON the driver turns into open-fpl-solver's `data` and `options`."""
    problem, params, rules = inst.problem, inst.params, inst.problem.rules
    return {
        "name": inst.spec.name,
        "next_gw": problem.gws[0].gw_index,
        "n_gws": len(problem.gws),
        "gw1": problem.gw1,
        "bank": problem.bank,
        "ft": problem.free_transfers,
        "players": [
            {
                "key": p.player_key,
                "et": p.element_type,
                "team": p.team_key,
                "buy": p.price if p.buyable else NOT_BUYABLE_PRICE,
                "sell": p.sell_price,
                "owned": p.owned,
                "xp": list(p.xp),
            }
            for p in problem.players
        ],
        "squad_select": {str(k): v for k, v in rules.squad_select.items()},
        "play_min": {str(k): v for k, v in rules.play_min.items()},
        "play_max": {str(k): v for k, v in rules.play_max.items()},
        "options": {
            "horizon": len(problem.gws),
            "objective": "decay",
            "decay_base": params.decay,
            "bench_weights": {str(i): w for i, w in enumerate(params.bench_weights)},
            "ft_value": THEIR_FT_SCALAR,
            "ft_value_list": their_ft_values(dict(params.ft_value)),
            "ft_use_penalty": None,
            "itb_value": params.itb_value / 10,  # money in tenths on both sides
            "itb_loss_per_transfer": None,
            "hit_cost": rules.hit_cost + params.hit_margin,
            "vcap_weight": 0.0,
            "chip_limits": {"wc": 0, "bb": 0, "fh": 0, "tc": 0},
            "allowed_chip_gws": {},
            "forced_chip_gws": {},
            "use_wc": [problem.gws[0].gw_index] if problem.gw1 else [],
            "preseason": False,
            "num_iterations": 1,
            "gap": params.mip_gap,
            "secs": params.time_limit,
            "random_seed": 0,
            "presolve": "on",
            "verbose": False,
        },
    }


# --- driver (runs in open-fpl-solver's venv) --------------------------------------------------


def drive(ofs_dir: Path, in_path: Path, out_path: Path) -> None:
    """Solve every exported instance with open-fpl-solver; write the results as JSON."""
    import pandas as pd

    sys.path.insert(0, str(ofs_dir))
    os.chdir(ofs_dir)
    import dev.solver as ofs  # open-fpl-solver's dev/solver.py
    import highspy

    created: list = []

    class RecordingHighs(highspy.Highs):  # observe the model to read its final MIP gap
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    ofs.highspy.Highs = RecordingHighs
    results = []
    for inst in json.loads(Path(in_path).read_text()):
        next_gw, n_gws = inst["next_gw"], inst["n_gws"]
        gws = list(range(next_gw, next_gw + n_gws))
        players = inst["players"]
        rows = []
        for p in players:
            row = {
                "id_x": p["key"],
                "ID": p["key"],
                "element_type": p["et"],
                "Pos": POSITION_SHORT[p["et"]][0],
                "name": str(p["team"]),
                "web_name": str(p["key"]),
                "now_cost": p["buy"],
            }
            for w, x in zip(gws, p["xp"], strict=True):
                row[f"{w}_Pts"], row[f"{w}_xMins"] = x, 90.0
            rows.append(row)
        merged = pd.DataFrame(rows).set_index("id_x")
        teams = sorted({str(p["team"]) for p in players}, key=int)
        type_data = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "singular_name_short": [POSITION_SHORT[t][1] for t in (1, 2, 3, 4)],
                "squad_select": [inst["squad_select"][str(t)] for t in (1, 2, 3, 4)],
                "squad_min_play": [inst["play_min"][str(t)] for t in (1, 2, 3, 4)],
                "squad_max_play": [inst["play_max"][str(t)] for t in (1, 2, 3, 4)],
            }
        ).set_index("id")
        owned = [p for p in players if p["owned"]]
        clubs: dict[int, int] = {}
        for p in owned:
            clubs[p["team"]] = clubs.get(p["team"], 0) + 1
        data = {
            "merged_data": merged,
            "team_data": pd.DataFrame({"id": range(len(teams)), "name": teams}),
            "type_data": type_data,
            "next_gw": next_gw,
            "initial_squad": [p["key"] for p in owned],
            "sell_price": {p["key"]: p["sell"] for p in owned},
            "buy_price": {p["key"]: p["buy"] for p in players},
            "price_modified_players": [p["key"] for p in owned if p["buy"] != p["sell"]],
            "itb": inst["bank"],
            "ft": inst["ft"],
            "ft_base": 1,
            "fixtures": [],
            "max_players_from_team": max(clubs.values()),
        }
        options = dict(inst["options"])
        options["bench_weights"] = {int(k): v for k, v in options["bench_weights"].items()}
        if not options["use_wc"]:
            del options["use_wc"]
        created.clear()
        start = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            solutions = ofs.solve_multi_period_fpl(data, options)
        seconds = time.perf_counter() - start
        sol, m = solutions[0], created[-1]
        info = m.getInfo()
        picks = sol["picks"]
        first = picks[picks["week"] == next_gw]
        stats = sol["statistics"]
        results.append(
            {
                "name": inst["name"],
                "score": float(sol["score"]),
                "mip_gap": float(info.mip_gap),
                "status": m.modelStatusToString(m.getModelStatus()),
                "seconds": seconds,
                "first_in": sorted(int(k) for k in first.loc[first["transfer_in"] == 1, "id"]),
                "first_out": sorted(int(k) for k in first.loc[first["transfer_out"] == 1, "id"]),
                "captains": [
                    int(picks.loc[(picks["week"] == w) & (picks["captain"] == 1), "id"].iloc[0])
                    for w in gws
                ],
                "xp": [float(stats[w]["xP"]) for w in gws],
                "hits": [round(stats[w]["pt"]) for w in gws],
                "ft": [round(stats[w]["ft"]) for w in gws],
                "bank": [round(stats[w]["itb"]) for w in gws],
            }
        )
    Path(out_path).write_text(json.dumps(results))


def run_theirs(ofs_dir: Path, payload: list[dict]) -> list[dict]:
    """Run this file in driver mode under the clone's venv (`uv run --project`)."""
    with tempfile.TemporaryDirectory() as tmp:
        in_path, out_path = Path(tmp) / "in.json", Path(tmp) / "out.json"
        in_path.write_text(json.dumps(payload))
        cmd = [
            "uv",
            "run",
            "--project",
            str(ofs_dir),
            "--no-dev",
            "python",
            str(Path(__file__).resolve()),
            "--driver",
            str(in_path),
            str(out_path),
            "--ofs-dir",
            str(ofs_dir),
        ]
        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        done = subprocess.run(cmd, cwd=ofs_dir, env=env, capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"open-fpl-solver driver failed:\n{done.stdout}\n{done.stderr}")
        return json.loads(out_path.read_text())


# --- comparison -------------------------------------------------------------------------------


@dataclass
class Row:
    name: str
    n_candidates: int
    ft: int
    price_modified: str  # owned players with selling price != price (sold by our plan)
    ours: float
    theirs: float
    offset: float
    diff: float
    tol: float
    ours_gap: float
    theirs_gap: float
    ours_seconds: float
    theirs_seconds: float
    first_equal: bool
    first_ours: str
    first_theirs: str
    captains_equal: bool
    xp_equal: bool
    fixed_theirs: float | None  # ours with their first-GW transfers fixed
    verdict: str
    notes: str = ""


def _moves(outs, ins) -> str:
    """'out1,out2>in1,in2' (player_keys), '-' for none."""
    if not outs and not ins:
        return "-"
    return f"{','.join(map(str, sorted(outs)))}>{','.join(map(str, sorted(ins)))}"


def compare(inst: Instance, theirs: dict) -> Row:
    """Solve ours and compare it with their result (module docstring)."""
    solve_plan = _ours()["solve_plan"]
    problem, params = inst.problem, inst.params
    plan = solve_plan(problem, params)
    offset = objective_offset(problem, params)
    adjusted = theirs["score"] - offset
    diff = plan.objective - adjusted
    scale = max(abs(plan.objective), abs(adjusted), 1.0)
    tol = (max(plan.mip_gap, 0.0) + max(theirs["mip_gap"], 0.0)) * scale + EPS
    first = plan.first
    ours_first = (frozenset(first.transfers_out), frozenset(first.transfers_in))
    theirs_first = (frozenset(theirs["first_out"]), frozenset(theirs["first_in"]))
    first_equal = ours_first == theirs_first
    fixed, fixed_ok = None, True
    if not first_equal:
        # Their first GW is feasible for ours: the optimum with it fixed must match theirs
        # within the two gaps (an alternative optimum), else the models disagree on it.
        fixed_plan = solve_plan(problem, params, fix_first_gw=theirs_first)
        fixed = fixed_plan.objective
        fixed_tol = (fixed_plan.mip_gap + max(theirs["mip_gap"], 0.0)) * scale + EPS
        fixed_ok = abs(fixed - adjusted) <= fixed_tol
    captains_equal = [g.captain for g in plan.gws] == theirs["captains"]
    xp_equal = all(abs(g.xp - x) <= 1e-6 for g, x in zip(plan.gws, theirs["xp"], strict=True))
    modified = {p.player_key for p in problem.players if p.owned and p.sell_price != p.price}
    sold = modified & {k for g in plan.gws for k in g.transfers_out}
    notes = []
    if [g.hits for g in plan.gws] != theirs["hits"]:
        notes.append(f"hits {[g.hits for g in plan.gws]} vs {theirs['hits']}")
    if not problem.gw1 and [g.ft for g in plan.gws] != theirs["ft"]:
        notes.append(f"ft {[g.ft for g in plan.gws]} vs {theirs['ft']}")
    if inst.unmappable:
        verdict = "not comparable"
        notes += inst.unmappable
    elif abs(diff) > tol:
        verdict = "MISMATCH"
    elif not fixed_ok:
        verdict = "MISMATCH (first GW)"
    else:
        verdict = "agree" if first_equal else "agree (alternative optimum)"
    return Row(
        name=inst.spec.name,
        n_candidates=len(problem.players),
        ft=problem.free_transfers,
        price_modified=f"{len(modified)} ({len(sold)})",
        ours=plan.objective,
        theirs=theirs["score"],
        offset=offset,
        diff=diff,
        tol=tol,
        ours_gap=plan.mip_gap,
        theirs_gap=theirs["mip_gap"],
        ours_seconds=plan.build_seconds + plan.solve_seconds,
        theirs_seconds=theirs["seconds"],
        first_equal=first_equal,
        # Only the moves the two first GWs do not share.
        first_ours=_moves(ours_first[0] - theirs_first[0], ours_first[1] - theirs_first[1]),
        first_theirs=_moves(theirs_first[0] - ours_first[0], theirs_first[1] - ours_first[1]),
        captains_equal=captains_equal,
        xp_equal=xp_equal,
        fixed_theirs=fixed,
        verdict=verdict,
        notes="; ".join(notes),
    )


def table(rows: list[Row]) -> str:
    """The results as a Markdown table."""
    head = (
        "| instance | cand | FT | sell != price (sold) | ours | theirs | offset "
        "| ours - (theirs - offset) | tol "
        "| gap ours/theirs | s ours/theirs | first-GW transfers equal | captains equal "
        "| per-GW xP equal | verdict |"
    )
    lines = [head, "|" + "---|" * 15]
    for r in rows:
        first = "yes"
        if not r.first_equal:
            first = (
                f"no: ours {r.first_ours}, theirs {r.first_theirs}; "
                f"ours with theirs fixed {r.fixed_theirs:.4f}"
            )
        lines.append(
            f"| {r.name} | {r.n_candidates} | {r.ft} | {r.price_modified} "
            f"| {r.ours:.4f} | {r.theirs:.4f} "
            f"| {r.offset:.4f} | {r.diff:+.2e} | {r.tol:.1e} "
            f"| {r.ours_gap:.1e}/{r.theirs_gap:.1e} "
            f"| {r.ours_seconds:.1f}/{r.theirs_seconds:.1f} | {first} "
            f"| {'yes' if r.captains_equal else 'no'} | {'yes' if r.xp_equal else 'no'} "
            f"| {r.verdict}{' (' + r.notes + ')' if r.notes else ''} |"
        )
    return "\n".join(lines)


def check(data_dir: Path, ofs_dir: Path, specs, mip_gap: float, time_limit: float) -> list[Row]:
    """Build, solve both ways and compare every spec."""
    instances = build_instances(data_dir, specs, mip_gap, time_limit)
    theirs = run_theirs(ofs_dir, [export(i) for i in instances])
    return [compare(i, t) for i, t in zip(instances, theirs, strict=True)]


def default_data_dir() -> Path:
    return Path(os.environ.get("FPLOPT_DATA_DIR", REPO / "data"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, default=None, help="data/ (read-only)")
    parser.add_argument("--ofs-dir", type=Path, default=os.environ.get("OPEN_FPL_SOLVER_DIR"))
    parser.add_argument("--gap", type=float, default=0.0, help="mip_rel_gap on both sides")
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--quick", action="store_true", help="3 small instances")
    parser.add_argument("--out", type=Path, help="also write the table (Markdown) here")
    parser.add_argument(
        "--driver", nargs=2, type=Path, metavar=("IN", "OUT"), help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    if args.ofs_dir is None:
        parser.error("set OPEN_FPL_SOLVER_DIR or pass --ofs-dir (a clone of open-fpl-solver)")
    ofs_dir = Path(args.ofs_dir).resolve()
    if args.driver:
        drive(ofs_dir, *args.driver)
        return 0
    data_dir = args.data_dir or default_data_dir()
    rows = check(data_dir, ofs_dir, QUICK_SPECS if args.quick else SPECS, args.gap, args.time_limit)
    text = table(rows)
    agree = sum(r.verdict.startswith("agree") for r in rows)
    summary = (
        f"{agree}/{len(rows)} agree within the MIP gaps (gap {args.gap}); "
        f"max |diff| {max(abs(r.diff) for r in rows):.2e}; first-GW transfers equal in "
        f"{sum(r.first_equal for r in rows)}/{len(rows)}"
    )
    print(text)
    print()
    print(summary)
    if args.out:
        args.out.write_text(text + "\n\n" + summary + "\n", encoding="utf-8")
    return 0 if not any(r.verdict.startswith("MISMATCH") for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
