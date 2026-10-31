"""Python's side of the bot replay (`fplopt_bot::replay` in the Rust port, migration batch 5):
a scripted session through `fplopt.bot.service.BotService` on recorded FPL responses, so the
Rust bot can be checked against Python's replies, alert texts and `bot.sqlite` rows.

The script format and the recorded responses' layout are documented in
`crates/fplopt-bot/src/replay.rs`. Used by `tests/test_rust_golden.py` (the synthetic golden
the Rust tests replay) and `dev/bot_parity.py` (real contexts and real teams).
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

__all__ = ("FileFpl", "dump_db", "golden", "replay")

log = logging.getLogger(__name__)

# Columns that hold the time a row was written: they differ between any two runs.
TIME_COLUMNS = {"created_at", "updated_at", "run_at"}


def _not_found() -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://fpl.test/")
    return httpx.HTTPStatusError(
        "404", request=request, response=httpx.Response(404, request=request)
    )


class FileFpl:
    """FPL's public API from recorded responses: `bootstrap-static.json` and, per team,
    `entry/<id>/history.json`, `transfers.json`, `picks/<gw>.json` (a missing file: 404)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.down = False

    def _read(self, path: Path) -> bytes:
        if not path.is_file():
            raise _not_found()
        return path.read_bytes()

    def entry_history(self, entry_id: int) -> bytes:
        return self._read(self.root / "entry" / str(entry_id) / "history.json")

    def entry_picks(self, entry_id: int, gw: int) -> bytes:
        return self._read(self.root / "entry" / str(entry_id) / "picks" / f"{gw}.json")

    def entry_transfers(self, entry_id: int) -> bytes:
        return self._read(self.root / "entry" / str(entry_id) / "transfers.json")

    def bootstrap_static(self) -> bytes:
        if self.down:
            raise httpx.ConnectError("FPL is down (script)")
        return (self.root / "bootstrap-static.json").read_bytes()


def texts_sender(texts: list[list[Any]]) -> Any:
    """A `send(chat_id, text)` that keeps what it is given."""

    def send(chat_id: int, text: str) -> None:
        texts.append([chat_id, text])

    return send


def replay(script: dict[str, Any], state_dir: Path, fpl_dir: Path) -> list[Any]:
    """Run `script` on the state under `state_dir` (`bot.sqlite`, `context/`); one result per
    step (`fplopt_bot::replay`)."""
    from fplopt.bot import live
    from fplopt.bot.app import COMMANDS, parse_command
    from fplopt.bot.context import ContextStore
    from fplopt.bot.planner import LIVE_PARAMS, chip_options, chip_params, plan_user
    from fplopt.bot.service import BotService
    from fplopt.bot.store import UserStore

    fpl = FileFpl(fpl_dir)
    clock = [datetime.now(UTC)]
    params = replace(LIVE_PARAMS, **(script.get("params") or {}))
    chips = replace(chip_params(), **(script.get("chip_params") or {}))

    def planner(f, entry_id, context, ft=None, made=()):
        return plan_user(f, entry_id, context, ft, params=params, made=made)

    def chip_planner(f, entry_id, context, ft=None, made=()):
        return chip_options(f, entry_id, context, ft, params=chips, made=made)

    service = BotService(
        UserStore(Path(state_dir) / "bot.sqlite"),
        ContextStore(Path(state_dir) / "context"),
        fpl,
        script.get("admin"),
        now=lambda: clock[0],
        planner=planner,
        chip_planner=chip_planner,
    )
    results: list[Any] = []
    for step in script["steps"]:
        if "now" in step:
            clock[0] = datetime.fromisoformat(step["now"])
            results.append(None)
        elif "fpl_down" in step:
            fpl.down = bool(step["fpl_down"])
            results.append(None)
        elif "chat" in step:
            chat = int(step["chat"])
            try:
                parsed = parse_command(step["text"])
                if parsed is None:
                    replies = service.unknown(chat)
                else:
                    command, args = parsed
                    method, with_args = COMMANDS[command]
                    call = getattr(service, method)
                    replies = call(chat, args) if with_args else call(chat)
                results.append({"replies": [[r.chat_id, r.text] for r in replies]})
            except Exception:
                log.warning("step failed", exc_info=True)
                results.append({"error": True})
        elif "alert" in step:
            try:
                user = service.users.user(int(step["alert"]))
                context = service.contexts.latest()
                texts: list[list[Any]] = []
                result = live.TickResult()
                live._alert_user(
                    step["kind"], context, user, service.users, service, texts_sender(texts), result
                )
                outcome = (
                    "sent"
                    if user.user_id in result.sent
                    else "silent"
                    if user.user_id in result.silent
                    else "told"
                )
                results.append({"alert": outcome, "texts": texts})
            except Exception:
                log.warning("alert failed", exc_info=True)
                results.append({"error": True})
        else:
            raise ValueError(f"script: unknown step {step}")
    return results


def dump_db(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Every table of a `bot.sqlite`, rows in rowid order, without the write-time columns
    (JSON columns as stored: both writers must write Python's `json.dumps` text)."""
    out: dict[str, list[dict[str, Any]]] = {}
    with closing(sqlite3.connect(path)) as db:
        db.row_factory = sqlite3.Row
        tables = [
            r[0]
            for r in db.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        for table in tables:
            rows = db.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
            out[table] = [{k: r[k] for k in r.keys() if k not in TIME_COLUMNS} for r in rows]
    return out


# --- the synthetic golden ---------------------------------------------------------------------

SEASON = 2023
GW = 6
ADMIN = 1000
TEAM = 7
MEMBER = 5


def _swap_for(context, squad: list[int]) -> tuple[int, int]:
    """The squad's first player for the cheapest pool player of the same position from a
    club the squad doesn't hold (tests/test_bot.py `swap_for`)."""
    pool = context.pool.set_index("player_key")
    keys = context.element_keys
    elements = {k: e for e, k in keys.items()}
    out = squad[0]
    position = pool.at[keys[out], "element_type"]
    clubs = {pool.at[keys[e], "team_key"] for e in squad}
    options = pool[
        (pool["element_type"] == position)
        & ~pool["team_key"].isin(clubs)
        & ~pool.index.isin([keys[e] for e in squad])
    ].sort_values("price", kind="mergesort")
    return out, elements[int(options.index[0])]


def golden(root: Path) -> None:
    """Write the synthetic golden to `root`: a GW6 plan context of the synthetic league
    (`state/context/`), team 7's recorded FPL responses (`fpl/`), the script, Python's
    results and the resulting database rows. The Rust test replays the script on a copy of
    the state and compares."""
    import pandas as pd
    from synthetic_season import synthetic_tables

    from fplopt.backtest.rules import backtest_rules
    from fplopt.backtest.simulator import season_schedule
    from fplopt.backtest.start_states import template_state
    from fplopt.bot.context import ContextStore, build_context
    from fplopt.features.store import DataStore

    shutil.rmtree(root, ignore_errors=True)
    league = DataStore(tables=synthetic_tables(seasons=(2022, SEASON), n_clubs=10))
    schedule = season_schedule(league, SEASON, 1)
    deadlines = {
        int(g): pd.Timestamp(d).to_pydatetime()
        for g, d in zip(schedule["gw"], schedule["deadline_time"], strict=True)
    }
    context = build_context(
        league, deadlines[GW - 1] + timedelta(hours=1), fingerprint="fp", xp_model="rolling"
    )
    assert context is not None
    # A fixed stamp, so regenerating doesn't churn the fixture's paths.
    context = replace(context, context_id="2023-09-30T120000000000Z", created_at=deadlines[GW - 1])
    state = root / "state"
    ContextStore(state / "context").write(context)

    view = league.as_of(pd.Timestamp(context.deadline))
    keys = context.element_keys
    by_key = {k: e for e, k in keys.items()}
    squad = [by_key[k] for k in template_state(view, backtest_rules(SEASON)).player_keys]
    fpl = root / "fpl"
    team = fpl / "entry" / str(TEAM)
    (team / "picks").mkdir(parents=True)
    rows = [{"event": e, "bank": 5, "event_transfers": 0} for e in range(1, GW)]
    (team / "history.json").write_text(json.dumps({"current": rows, "chips": []}))
    (team / "transfers.json").write_text("[]")
    for gw in range(1, GW):
        (team / "picks" / f"{gw}.json").write_text(
            json.dumps({"picks": [{"element": e} for e in squad]})
        )
    # FPL's live prices: the context's, the first squad player £0.3m dearer.
    price = dict(zip(context.pool["player_key"], context.pool["price"], strict=True))
    elements = [
        {"id": e, "now_cost": int(price[k]) + (3 if e == squad[0] else 0)}
        for e, k in keys.items()
        if k in price
    ]
    (fpl / "bootstrap-static.json").write_text(json.dumps({"elements": elements}))

    names = context.names
    out, into = _swap_for(context, squad)
    out_name, in_name = names[keys[out]], names[keys[into]]
    before = (context.deadline - timedelta(hours=20)).isoformat()
    later = (context.deadline - timedelta(hours=2)).isoformat()
    steps: list[dict[str, Any]] = [
        {"now": before},
        {"chat": MEMBER, "text": "/plan"},
        {"chat": MEMBER, "text": "/start"},
        {"chat": MEMBER, "text": "/start"},
        {"chat": MEMBER, "text": "/register 7"},
        {"chat": 6, "text": "/register 7"},
        {"chat": MEMBER, "text": "/allow 5"},
        {"chat": ADMIN, "text": "/help"},
        {"chat": ADMIN, "text": "/allow x"},
        {"chat": ADMIN, "text": "/allow 5"},
        {"chat": ADMIN, "text": "/users"},
        {"chat": MEMBER, "text": "/help"},
        {"chat": MEMBER, "text": "hello"},
        {"chat": MEMBER, "text": "/nope"},
        {"chat": MEMBER, "text": "/plan"},
        {"chat": MEMBER, "text": "/chips"},
        {"chat": MEMBER, "text": "/register"},
        {"chat": MEMBER, "text": "/register abc"},
        {"chat": MEMBER, "text": "/register 0"},
        {"chat": MEMBER, "text": "/register 8"},
        {"chat": MEMBER, "text": "/register 7"},
        {"chat": MEMBER, "text": "/plan"},
        {"alert": MEMBER, "kind": "alert24"},
        {"chat": MEMBER, "text": "/setft 9"},
        {"chat": MEMBER, "text": "/setft 2"},
        {"chat": MEMBER, "text": "/plan"},
        {"chat": MEMBER, "text": "/setft clear"},
        {"chat": MEMBER, "text": "/transfer"},
        {"chat": MEMBER, "text": "/transfer a b c"},
        {"chat": MEMBER, "text": "/transfer Nobody -> X"},
        {"chat": MEMBER, "text": f"/transfer {out_name} -> {in_name}"},
        {"chat": MEMBER, "text": f"/transfer {out_name} -> {in_name}"},
        {"chat": MEMBER, "text": f"/transfer {out} -> {in_name}"},
        {"chat": MEMBER, "text": "/transfer list"},
        {"chat": MEMBER, "text": "/plan"},
        {"chat": MEMBER, "text": "/chips"},
        {"now": later},
        {"alert": MEMBER, "kind": "alert2"},
        {"chat": MEMBER, "text": "/transfer undo"},
        {"alert": MEMBER, "kind": "alert2"},
        {"fpl_down": True},
        {"chat": MEMBER, "text": "/plan"},
        {"fpl_down": False},
        {"chat": MEMBER, "text": "  /PLAN@fplopt_bot "},
        {"chat": MEMBER, "text": "/alerts"},
        {"chat": MEMBER, "text": "/alerts off"},
        {"chat": MEMBER, "text": "/alerts maybe"},
        {"chat": ADMIN, "text": "/block 1000"},
        {"chat": ADMIN, "text": "/block 5"},
        {"chat": MEMBER, "text": "/plan"},
        {"chat": MEMBER, "text": "/start"},
        {"chat": ADMIN, "text": "/users"},
        {"now": context.deadline.isoformat()},
        {"chat": ADMIN, "text": "/register 7"},
        {"chat": ADMIN, "text": "/plan"},
    ]
    script = {
        "admin": ADMIN,
        # The bot's code path with a 1-GW horizon (tests/test_bot.py TEST_PARAMS).
        "params": {"horizon": 1, "time_limit": None},
        "chip_params": {"horizon": 1, "time_limit": None, "chip_value_left": None},
        "steps": steps,
    }
    (root / "script.json").write_text(json.dumps(script, indent=1) + "\n", newline="\n")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        run = Path(tmp) / "state"
        shutil.copytree(state, run)
        results = replay(script, run, fpl)
        rows = dump_db(run / "bot.sqlite")
    for name, value in (("results.json", results), ("db.json", rows)):
        (root / name).write_text(
            json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
        )
