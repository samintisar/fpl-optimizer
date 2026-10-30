"""Rust migration batch 5, the parity gate: the Rust bot must answer every command and alert
as Python's does (docs/superpowers/plans/2026-10-28-rust-migration.md, Batch 5). Dev only:
never imported by `fplopt`.

Sessions (scripted, `tests/bot_replay.py` / `fplopt-rs bot replay`), each run twice on
copies of the same state, once by Python's `BotService` and once by the Rust bot:

- `server`: a copy of the live state (a backup of `bot.sqlite`, never the live file, and the
  newest plan context, copied from the server; `--server DIR`). Every registered user plans
  on it as they would now: /plan, /chips, /transfer list and both deadline alerts; the
  public teams (`--teams`) register on new chats and do the same, with /setft.
- `local-<season>-<gw>`: a plan context built from `data/` at a past deadline with the live
  xP model (v1), on a new database: the same teams and commands.

FPL's responses for the teams (history, transfers, every GW's picks) and one bootstrap (the
live prices) are recorded once into `--fpl` and served to both from the files, so both see
the same bytes. Checks per session:

- every step's replies and alert texts identical, byte for byte;
- the databases identical (every table and row; write times aside), the plan logs included;
- where a plan differs: the batch 4 rule (plan #1's objective within 1e-6, the same plan #1
  unless it ties plan #2 within 1e-6) decides whether it is a tie broken differently.

    cargo build --release -p fplopt-cli
    uv run python dev/bot_parity.py --server DIR [--data data] [--deadlines 2026:4,2026:8]
        [--teams 1,2,...] [--fpl DIR] [--keep DIR]

Logs one row to results/experiments.csv (family `rust parity`) and exits 1 on any failure.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

FAMILY = "rust parity"
QUIET = {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_ADMIN_CHAT_ID": "", "HEALTHCHECK_PING_URL": ""}
OBJECTIVE_TOL = 1e-6
ADMIN = 1  # a chat id no real user has
FIRST_CHAT = 900_001  # the public teams' chats
# Public FPL teams of all kinds: long-standing, new, inactive this season, unknown.
TEAMS = (1, 2, 3, 13, 100, 1000, 10_000, 123_456, 1_000_000, 5_000_000, 99_999_999)
CHIP_TEAMS = 3  # /chips for this many teams per session (the chip search is the slow part)


# --- recorded FPL responses -------------------------------------------------------------------


def record_fpl(root: Path, teams: list[int]) -> None:
    """Fetch the bootstrap and each team's history, transfers and picks into `root` (a team
    FPL doesn't know gets no directory: 404 on replay). Files already there are kept."""
    import httpx

    from fplopt.adapters.fpl import BASE_URL

    root.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=30, headers={"User-Agent": "fplopt-parity"}) as http:

        def get(path: str) -> httpx.Response:
            for attempt in range(4):
                response = http.get(f"{BASE_URL}/{path}")
                if response.status_code < 500:
                    return response
                time.sleep(2**attempt)
            response.raise_for_status()
            return response

        if not (root / "bootstrap-static.json").exists():
            response = get("bootstrap-static/")
            response.raise_for_status()
            (root / "bootstrap-static.json").write_bytes(response.content)
        for team in teams:
            directory = root / "entry" / str(team)
            if (directory / "transfers.json").exists() or (root / f"unknown-{team}").exists():
                continue
            history = get(f"entry/{team}/history/")
            if history.status_code == 404:
                (root / f"unknown-{team}").write_text("404")
                continue
            history.raise_for_status()
            (directory / "picks").mkdir(parents=True, exist_ok=True)
            for row in history.json().get("current", []):
                picks = get(f"entry/{team}/event/{row['event']}/picks/")
                if picks.status_code == 200:
                    (directory / "picks" / f"{row['event']}.json").write_bytes(picks.content)
                time.sleep(0.2)
            (directory / "history.json").write_bytes(history.content)
            transfers = get(f"entry/{team}/transfers/")
            transfers.raise_for_status()
            (directory / "transfers.json").write_bytes(transfers.content)


# --- sessions ---------------------------------------------------------------------------------


def team_steps(chat: int, team: int, *, new: bool, chips: bool, setft: bool) -> list[dict]:
    steps: list[dict[str, Any]] = []
    if new:
        steps += [
            {"chat": chat, "text": "/start"},
            {"chat": ADMIN, "text": f"/allow {chat}"},
            {"chat": chat, "text": f"/register {team}"},
        ]
    steps += [
        {"chat": chat, "text": "/plan"},
        {"chat": chat, "text": "/transfer list"},
        {"alert": chat, "kind": "alert24"},
    ]
    if chips:
        steps.append({"chat": chat, "text": "/chips"})
    if setft:
        steps += [
            {"chat": chat, "text": "/setft 1"},
            {"alert": chat, "kind": "alert2"},
            {"chat": chat, "text": "/setft clear"},
        ]
    steps.append({"alert": chat, "kind": "alert2"})
    return steps


def session_script(context_deadline, users: list[tuple[int, int]], teams: list[int]) -> dict:
    """The script on a context: the existing users (chat, team) and the public teams."""
    now = (context_deadline - timedelta(hours=20)).isoformat()
    steps: list[dict[str, Any]] = [{"now": now}, {"chat": ADMIN, "text": "/start"}]
    n = 0
    for chat, team in users:
        steps += team_steps(chat, team, new=False, chips=n < CHIP_TEAMS, setft=True)
        n += 1
    for i, team in enumerate(teams):
        chat = FIRST_CHAT + i
        steps += team_steps(chat, team, new=True, chips=n < CHIP_TEAMS, setft=i < 2)
        n += 1
    steps += [
        {"chat": ADMIN, "text": "/users"},
        {"fpl_down": True},
        {"chat": FIRST_CHAT, "text": "/plan"},
        {"fpl_down": False},
        {"now": context_deadline.isoformat()},
        {"chat": FIRST_CHAT, "text": "/plan"},
    ]
    return {"admin": ADMIN, "steps": steps}


def local_context(data: Path, season: int, gw: int, root: Path) -> Path:
    """A v1 plan context for `season` GW `gw` from `data/`, written under `root` (reused)."""
    import pandas as pd

    from fplopt.backtest.simulator import FAR_FUTURE
    from fplopt.bot.context import ContextStore, build_context
    from fplopt.features.store import DataStore

    contexts = ContextStore(root / "context")
    if contexts.ids():
        return root
    store = DataStore(data)
    gameweeks = store.as_of(FAR_FUTURE).table("gameweek")
    row = gameweeks[(gameweeks["season"] == season) & (gameweeks["gw"] == gw)].iloc[0]
    now = pd.Timestamp(row["deadline_time"]).to_pydatetime() - timedelta(days=2)
    context = build_context(store, now, fingerprint="bot-parity")
    if context is None or (context.season, context.gw) != (season, gw):
        raise SystemExit(f"no context for {season} GW{gw} in {data}")
    contexts.write(context)
    return root


def replay_python(script: Path, state: Path, fpl: Path, out: Path) -> None:
    from bot_replay import replay

    results = replay(json.loads(script.read_text(encoding="utf-8")), state, fpl)
    out.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")


def run_session(name: str, base: Path, script: dict, fpl: Path, binary: Path, work: Path) -> dict:
    """Both runs of one session; their results and databases."""
    from bot_replay import dump_db

    root = work / name
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    (root / "script.json").write_text(json.dumps(script, indent=1), encoding="utf-8")
    env = {**os.environ, **QUIET, "RUST_LOG": "error", "PYTHONIOENCODING": "utf-8"}
    commands = {
        "python": [sys.executable, __file__, "--replay-python", str(root / "script.json")],
        "rust": [str(binary), "bot", "replay", "--script", str(root / "script.json")],
    }
    seconds = {}

    def run(who: str) -> None:
        state = root / who
        shutil.copytree(base, state)
        if who == "python":
            command = [*commands[who], str(state), str(fpl), str(root / f"{who}.json")]
        else:
            command = [
                *commands[who],
                "--state", str(state), "--fpl", str(fpl), "--out", str(root / f"{who}.json"),
            ]  # fmt: skip
        started = time.monotonic()
        done = subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True)
        seconds[who] = time.monotonic() - started
        (root / f"{who}.log").write_text(done.stdout + done.stderr, encoding="utf-8")
        if done.returncode != 0:
            raise RuntimeError(f"{name}: {who} replay failed (see {root / who}.log)")

    with ThreadPoolExecutor(2) as pool:
        list(pool.map(run, ["python", "rust"]))
    read = lambda who: json.loads((root / f"{who}.json").read_text(encoding="utf-8"))  # noqa: E731
    return {
        "name": name,
        "script": script,
        "python": read("python"),
        "rust": read("rust"),
        "python_db": dump_db(root / "python" / "bot.sqlite"),
        "rust_db": dump_db(root / "rust" / "bot.sqlite"),
        "seconds": seconds,
    }


# --- checks -----------------------------------------------------------------------------------


def plan_one(plans: dict) -> tuple[Any, float, float]:
    """(plan #1's decision for every GW, its objective, the gap to plan #2)."""
    first = plans["plans"][0]
    key = [
        (g["transfers"], g["starters"], g["bench"], g["captain"], g["vice"], g["chip"])
        for g in first["gws"]
    ]
    second = plans["plans"][1]["total_objective"] if len(plans["plans"]) > 1 else float("-inf")
    return key, first["total_objective"], first["total_objective"] - second


def tie_explains(py_text: str | None, rs_text: str | None) -> str | None:
    """None if two plan logs differ only by a tie broken differently (batch 4 rule), else
    why not."""
    if py_text is None or rs_text is None:
        return "a plan log is missing"
    py, rs = json.loads(py_text), json.loads(rs_text)
    if not py["plans"] or not rs["plans"]:
        return "no plans"
    py_key, py_obj, py_gap = plan_one(py)
    rs_key, rs_obj, _ = plan_one(rs)
    if abs(py_obj - rs_obj) > OBJECTIVE_TOL:
        return f"plan #1 objective {py_obj!r} vs {rs_obj!r}"
    if py_key != rs_key and py_gap > OBJECTIVE_TOL:
        return f"a different plan #1 (Python's leads plan #2 by {py_gap:.2e})"
    return None


def check(session: dict) -> dict:
    problems: list[str] = []
    ties = 0
    steps = session["script"]["steps"]
    py, rs = session["python"], session["rust"]
    if len(py) != len(rs):
        problems.append(f"{len(py)} vs {len(rs)} step results")
    tie_steps: set[int] = set()
    py_recs = session["python_db"].get("recommendations", [])
    rs_recs = session["rust_db"].get("recommendations", [])
    for a, b in zip(py_recs, rs_recs, strict=False):
        if a == b:
            continue
        rest = {k for k in a if a.get(k) != b.get(k)}
        if rest <= {"plans", "decision", "message"}:
            reason = tie_explains(a.get("plans"), b.get("plans"))
            if reason is None:
                ties += 1
                continue
            problems.append(f"recommendation {a['rec_id']}: {reason}")
        else:
            problems.append(f"recommendation {a['rec_id']}: {sorted(rest)} differ")
    for table in sorted(set(session["python_db"]) | set(session["rust_db"])):
        a, b = session["python_db"].get(table, []), session["rust_db"].get(table, [])
        if table == "recommendations":
            if len(a) != len(b):
                problems.append(f"recommendations: {len(a)} vs {len(b)} rows")
            continue
        if table == "user_state" and ties:
            continue  # the squad saved after a tie-broken plan is the same import; checked below
        if a != b:
            bad = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), None)
            problems.append(f"{table}: {len(a)} vs {len(b)} rows, first difference at {bad}")
    for i, (a, b) in enumerate(zip(py, rs, strict=False)):
        if a == b:
            continue
        planned = isinstance(a, dict) and ("replies" in a or "alert" in a)
        if ties and planned and isinstance(b, dict):
            tie_steps.add(i)  # a plan text after a tie broken differently: logged above
            continue
        step = steps[i] if i < len(steps) else {}
        problems.append(f"step {i} {json.dumps(step, ensure_ascii=False)}: {_diff(a, b)}")
    errors = sum(1 for r in py if isinstance(r, dict) and r.get("error"))
    replies = sum(len(r.get("replies") or r.get("texts") or []) for r in py if isinstance(r, dict))
    plans = sum(1 for r in py_recs if r.get("plans"))
    return {
        "name": session["name"],
        "steps": len(steps),
        "replies": replies,
        "plans": plans,
        "ties": ties,
        "tie_steps": sorted(tie_steps),
        "errors": errors,
        "problems": problems,
        "seconds": session["seconds"],
    }


def _diff(a: Any, b: Any) -> str:
    sa, sb = json.dumps(a, ensure_ascii=False), json.dumps(b, ensure_ascii=False)
    same = (k for k, (x, y) in enumerate(zip(sa, sb, strict=False)) if x != y)
    i = next(same, min(len(sa), len(sb)))
    return f"python …{sa[max(0, i - 60) : i + 80]!r} / rust …{sb[max(0, i - 60) : i + 80]!r}"


# --- main -------------------------------------------------------------------------------------


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--replay-python":
        _, _, script, state, fpl, out = sys.argv
        replay_python(Path(script), Path(state), Path(fpl), Path(out))
        return 0
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--server", type=Path, help="a copy of the server's state/ (bot.sqlite, context/)"
    )
    parser.add_argument("--data", type=Path, default=REPO / "data", help="data/ (read only)")
    parser.add_argument("--deadlines", default="2026:4,2026:8", help="season:gw,... ('' for none)")
    parser.add_argument("--teams", default=",".join(map(str, TEAMS)))
    parser.add_argument("--fpl", type=Path, help="recorded FPL responses (fetched if missing)")
    parser.add_argument("--bin", type=Path, default=REPO / "target" / "release" / "fplopt-rs")
    parser.add_argument("--keep", type=Path, help="keep the work directory here")
    parser.add_argument("--experiments", type=Path, default=REPO / "results" / "experiments.csv")
    args = parser.parse_args()
    binary = args.bin if args.bin.exists() else args.bin.with_suffix(".exe")
    if not binary.exists():
        parser.error(f"no Rust binary at {args.bin} (cargo build --release -p fplopt-cli)")
    os.environ.update(QUIET)

    import sqlite3
    from contextlib import closing

    from fplopt.backtest.evaluate import family_variants, log_experiment
    from fplopt.bot.context import ContextStore

    work = Path(tempfile.mkdtemp(prefix="bot-parity-")) if args.keep is None else args.keep
    work.mkdir(parents=True, exist_ok=True)
    teams = [int(t) for t in args.teams.split(",") if t.strip()]
    users: list[tuple[int, int]] = []
    if args.server is not None:
        with closing(sqlite3.connect(args.server / "bot.sqlite")) as db:
            users = db.execute(
                "SELECT u.chat_id, s.entry_id FROM users u JOIN user_settings s USING (user_id) "
                "WHERE u.status = 'active' AND s.entry_id IS NOT NULL ORDER BY u.user_id"
            ).fetchall()
    fpl = args.fpl or work / "fpl"
    started = time.monotonic()
    record_fpl(fpl, sorted({*teams, *(t for _, t in users)}))
    print(f"FPL responses in {fpl} ({time.monotonic() - started:.0f} s)", flush=True)

    sessions: list[tuple[str, Path, dict]] = []
    if args.server is not None:
        context = ContextStore(args.server / "context").latest()
        if context is None:
            parser.error(f"no context under {args.server / 'context'}")
        base = work / "base-server"
        if not base.exists():
            (base / "context").mkdir(parents=True)
            name = context.context_id
            shutil.copytree(args.server / "context" / name, base / "context" / name)
            shutil.copy(args.server / "bot.sqlite", base / "bot.sqlite")
        sessions.append(("server", base, session_script(context.deadline, users, teams)))
    for item in [d for d in args.deadlines.split(",") if d.strip()]:
        season, gw = (int(x) for x in item.split(":"))
        started = time.monotonic()
        base = local_context(args.data.resolve(), season, gw, work / f"base-{season}-{gw}")
        context = ContextStore(base / "context").latest()
        took = time.monotonic() - started
        print(f"context {season} GW{gw}: {context.context_id} ({took:.0f} s)", flush=True)
        script = session_script(context.deadline, [], sorted({*teams, *(t for _, t in users)}))
        sessions.append((f"local-{season}-{gw}", base, script))

    started = time.monotonic()
    with ThreadPoolExecutor(len(sessions) or 1) as pool:
        done = list(pool.map(lambda s: run_session(*s, fpl, binary, work), sessions))
    print(f"sessions replayed ({time.monotonic() - started:.0f} s)", flush=True)
    results = []
    for session in done:
        result = check(session)
        results.append(result)
        name = result["name"]
        verdict = "ok" if not result["problems"] else f"{len(result['problems'])} problem(s)"
        print(
            f"{name}: {result['steps']} steps, {result['replies']} messages, {result['plans']} "
            f"plans logged, {result['ties']} ties, {result['errors']} failed steps (both): "
            f"{verdict} (python {result['seconds']['python']:.0f} s, rust "
            f"{result['seconds']['rust']:.0f} s)",
            flush=True,
        )
        for problem in result["problems"][:10]:
            print(f"    {problem}")
    passed = all(not r["problems"] for r in results) and bool(results)
    ok = sum(not r["problems"] for r in results)
    print(f"\n{'PASS' if passed else 'FAIL'}: {ok}/{len(results)} sessions")
    family_n = family_variants(args.experiments, FAMILY) + 1
    log_experiment(
        args.experiments,
        shlex.join(["python", "dev/bot_parity.py", *sys.argv[1:]]),
        {"gate": "batch 5 bot", "data": str(args.data), "binary": str(binary), "teams": teams},
        {
            "passed": passed,
            "sessions": len(results),
            "steps": sum(r["steps"] for r in results),
            "messages": sum(r["replies"] for r in results),
            "plans": sum(r["plans"] for r in results),
            "ties": sum(r["ties"] for r in results),
            "failed": [f"{r['name']}: {p}" for r in results for p in r["problems"][:3]],
            "python_s": round(sum(r["seconds"]["python"] for r in results), 1),
            "rust_s": round(sum(r["seconds"]["rust"] for r in results), 1),
        },
        family_n,
        FAMILY,
    )
    if args.keep is None:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
