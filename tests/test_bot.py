"""The Telegram bot (`fplopt.bot`, Phase 7): user store, plan context, planner, messages,
command logic and the deadline alerts, on the synthetic league with fake FPL and Telegram
endpoints (no network)."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import timedelta

import httpx
import pandas as pd
import pytest
from synthetic_season import synthetic_tables

from fplopt.bot import live
from fplopt.bot.context import ContextStore, build_context, next_gameweek, raw_fingerprint
from fplopt.bot.messages import changes_text, recommendation_text, split_message
from fplopt.bot.planner import (
    LIVE_PARAMS,
    chips_left,
    decision_summary,
    plan_user,
    plans_json,
    ranked,
)
from fplopt.bot.service import HELP, NOT_READY, PENDING, BotService
from fplopt.bot.store import UserStore
from fplopt.entry import EntryError
from fplopt.features.store import DataStore
from fplopt.ingest.raw_store import RawStore

SEASON = 2023
GW = 6
ADMIN = 1000
TEAM = 7


@pytest.fixture(scope="module")
def league():
    return DataStore(tables=synthetic_tables(seasons=(2022, SEASON), n_clubs=10))


@pytest.fixture(scope="module")
def deadlines(league):
    """FPL GW -> deadline of the synthetic 2023/24."""
    from fplopt.backtest.simulator import season_schedule

    schedule = season_schedule(league, SEASON, 1)
    return {
        int(g): pd.Timestamp(d).to_pydatetime()
        for g, d in zip(schedule["gw"], schedule["deadline_time"], strict=True)
    }


@pytest.fixture(scope="module")
def context(league, deadlines):
    """The plan context for GW6 (rolling xP: fast, and the bot's code path is the same)."""
    return build_context(
        league, deadlines[GW - 1] + timedelta(hours=1), fingerprint="fp", xp_model="rolling"
    )


@pytest.fixture(scope="module")
def squad(league, context):
    """element_ids of the GW6 template squad (team 7 holds it from GW1)."""
    from fplopt.backtest.rules import backtest_rules
    from fplopt.backtest.start_states import template_state

    view = league.as_of(pd.Timestamp(context.deadline))
    state = template_state(view, backtest_rules(SEASON))
    keys = {k: e for e, k in context.element_keys.items()}
    return [keys[k] for k in state.player_keys]


def not_found() -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://fpl.test/")
    return httpx.HTTPStatusError(
        "404", request=request, response=httpx.Response(404, request=request)
    )


class FakeEntries:
    """FPL's public entry endpoints: team 7 holds `squad` since GW1, no transfers or chips,
    £0.5m in the bank; every other team is unknown (404)."""

    def __init__(self, squad):
        self.squad = squad
        self.calls = 0

    def entry_history(self, entry_id):
        self.calls += 1
        if entry_id != TEAM:
            raise not_found()
        rows = [{"event": e, "bank": 5, "event_transfers": 0} for e in range(1, GW)]
        return json.dumps({"current": rows, "chips": []}).encode()

    def entry_picks(self, entry_id, gw):
        return json.dumps({"picks": [{"element": e} for e in self.squad]}).encode()

    def entry_transfers(self, entry_id):
        return b"[]"


@pytest.fixture
def entries(squad):
    return FakeEntries(squad)


@pytest.fixture
def users(tmp_path):
    return UserStore(tmp_path / "state" / "bot.sqlite")


@pytest.fixture
def contexts(tmp_path, context):
    store = ContextStore(tmp_path / "state" / "context")
    store.write(context)
    return store


# The live settings with a 1-GW horizon: the synthetic league's 6-GW MILPs take minutes
# (rolling xP has many ties), and the bot's code path is the same.
TEST_PARAMS = replace(LIVE_PARAMS, horizon=1, time_limit=None)


def rolling_planner(fpl, entry_id, context, ft=None):
    return plan_user(fpl, entry_id, context, ft, params=TEST_PARAMS)


@pytest.fixture
def service(users, contexts, entries, context):
    return BotService(
        users,
        contexts,
        entries,
        ADMIN,
        now=lambda: context.deadline - timedelta(hours=20),
        planner=rolling_planner,
    )


def texts(replies, chat_id=None):
    return [r.text for r in replies if chat_id is None or r.chat_id == chat_id]


# --- user store -------------------------------------------------------------------------------


def test_user_store_lifecycle(users):
    assert users.user(5) is None
    user = users.add_user(5)
    assert (user.status, user.entry_id, user.alerts) == ("pending", None, True)
    assert users.add_user(5, "active").status == "pending"  # existing users are unchanged
    assert users.set_status(5, "active").active
    assert users.set_status(6, "active") is None
    users.set_entry(user.user_id, 4570069)
    users.set_alerts(user.user_id, False)
    assert users.user(5) == replace(user, status="active", entry_id=4570069, alerts=False)
    assert users.alert_users() == []
    users.set_alerts(user.user_id, True)
    assert [u.chat_id for u in users.alert_users()] == [5]
    with pytest.raises(ValueError, match="unknown status"):
        users.add_user(7, "admin")


def test_user_store_overrides_and_recommendations(users):
    user = users.add_user(5, "active")
    assert users.free_transfers(user.user_id, SEASON, GW) is None
    users.set_free_transfers(user.user_id, SEASON, GW, 2)
    users.set_free_transfers(user.user_id, SEASON, GW, 3)
    assert users.free_transfers(user.user_id, SEASON, GW) == 3
    assert users.free_transfers(user.user_id, SEASON, GW + 1) is None
    users.set_free_transfers(user.user_id, SEASON, GW, None)
    assert users.free_transfers(user.user_id, SEASON, GW) is None

    decision = {"transfers": [[1, 2]], "captain": 3, "vice": 4}
    users.record(user.user_id, SEASON, GW, "alert24", context_id="c1", decision=decision, sent=True)
    users.record(user.user_id, SEASON, GW, "alert2", context_id="c2", decision={"x": 1})
    last = users.last_sent_alert(user.user_id, SEASON, GW)
    assert last.kind == "alert24" and last.decision == decision and last.sent
    assert users.last_sent_alert(user.user_id, SEASON, GW + 1) is None
    assert [r.kind for r in users.recommendations(user.user_id)] == ["alert24", "alert2"]
    with pytest.raises(ValueError, match="unknown kind"):
        users.record(user.user_id, SEASON, GW, "nudge", context_id=None)

    users.save_squad(user.user_id, SEASON, GW, 7, {"bank": 5}, "c1")
    users.save_squad(user.user_id, SEASON, GW, 7, {"bank": 6}, "c2")
    assert users.squad(user.user_id, SEASON, GW) == {"bank": 6}

    from datetime import UTC, datetime

    assert users.alert_run(SEASON, GW, "alert24") is None
    users.mark_alert_run(SEASON, GW, "alert24", "c1", datetime(2023, 9, 1, tzinfo=UTC))
    assert users.alert_run(SEASON, GW, "alert24")["context_id"] == "c1"
    users.mark_alert_run(SEASON, GW, "alert24", "c2", datetime(2023, 9, 2, tzinfo=UTC))
    assert users.alert_run(SEASON, GW, "alert24")["context_id"] == "c1"  # first mark kept


# --- plan context -----------------------------------------------------------------------------


def test_build_context(context, deadlines, league):
    assert (context.season, context.gw, context.gw_index) == (SEASON, GW, GW)
    assert context.deadline == deadlines[GW]
    assert context.xp_model == "rolling" and context.fingerprint == "fp"
    assert set(context.xp["gw"]) == set(range(GW, GW + 6))
    assert len(context.pool) and context.pool["player_key"].is_unique
    assert set(context.prices["gw"]) == set(range(1, GW + 1))  # GW1..the planned GW
    assert context.gw_index_map[GW] == GW
    first = context.initial_prices(1)
    assert len(first) and all(p > 0 for p in first.values())
    # The next deadline after the last one: none.
    assert next_gameweek(league, max(deadlines.values()) + timedelta(days=1)) is None
    assert build_context(league, max(deadlines.values()) + timedelta(days=1)) is None


def test_context_store_round_trip(tmp_path, context):
    store = ContextStore(tmp_path / "ctx")
    assert store.latest() is None and store.meta() is None
    path = store.write(context)
    assert sorted(p.name for p in path.iterdir()) == sorted(
        [
            "meta.json",
            "xp.parquet",
            "pool.parquet",
            "players.parquet",
            "prices.parquet",
            "gameweeks.parquet",
        ]
    )
    assert [p.name for p in (tmp_path / "ctx").iterdir()] == [context.context_id]  # no tmp
    read = store.latest()
    assert read.meta() == context.meta()
    for name in ("xp", "pool", "players", "prices", "gameweeks"):
        pd.testing.assert_frame_equal(getattr(read, name), getattr(context, name))
    assert store.latest() is read  # cached
    with pytest.raises(FileExistsError):
        store.write(context)
    # A half-written context (a temporary directory) is never listed.
    (tmp_path / "ctx" / ".zzz.1.tmp").mkdir()
    assert store.ids() == [context.context_id]
    for i in range(3):
        store.write(replace(context, context_id=f"{context.context_id}-{i}"))
    assert store.prune(keep=2) == [context.context_id, f"{context.context_id}-0"]
    assert store.ids() == [f"{context.context_id}-1", f"{context.context_id}-2"]
    assert store.latest().context_id == f"{context.context_id}-2"


def test_raw_fingerprint_changes_with_new_files(tmp_path):
    raw = RawStore(tmp_path / "raw")
    empty = raw_fingerprint(raw.root)
    from datetime import UTC, datetime

    raw.write("fpl", "bootstrap-static", b"{}", datetime(2026, 10, 1, tzinfo=UTC))
    one = raw_fingerprint(raw.root)
    assert one != empty
    (raw.root / "fpl" / "bootstrap-static" / ".partial.tmp").write_bytes(b"x")
    assert raw_fingerprint(raw.root) == one  # temporary files don't count
    raw.write("vaastav", "data", b"{}", datetime(2026, 10, 2, tzinfo=UTC))
    assert raw_fingerprint(raw.root) == one  # only FPL, odds and football-data
    raw.write("odds", "soccer_epl", b"[]", datetime(2026, 10, 2, tzinfo=UTC))
    assert raw_fingerprint(raw.root) != one


# --- planner ----------------------------------------------------------------------------------


def test_live_params_are_the_phase6_system():
    from fplopt.backtest import holdout
    from fplopt.cli import INTERACTIVE_TIME_LIMIT, parse_policy_spec

    spec = parse_policy_spec(holdout.SYSTEM)
    policy = spec.build()
    assert spec.xp == "v1"
    assert replace(LIVE_PARAMS, time_limit=None) == policy.params
    assert LIVE_PARAMS.time_limit == INTERACTIVE_TIME_LIMIT
    from fplopt.bot.context import LIVE_XP

    assert LIVE_XP == spec.xp


def test_plan_user(entries, context):
    plan = rolling_planner(entries, TEAM, context)
    assert plan.spec.bank == 5 and len(plan.spec.players) == 15
    assert 1 <= len(plan.plans.plans) <= 3 and plan.plans.roll is not None
    order = ranked(plan.plans)
    assert [p.total_objective for p in order] == sorted(
        (p.total_objective for p in order), reverse=True
    )
    summary = decision_summary(plan)
    assert summary["captain"] == order[0].first.captain
    assert summary["transfers"] == sorted([t.out_key, t.in_key] for t in order[0].first.transfers)
    logged = json.loads(json.dumps(plans_json(plan)))  # plain data
    assert logged["entry_id"] == TEAM and len(logged["plans"]) == len(order)
    assert logged["params"]["max_hits"] == 1
    assert set(chips_left(plan.state, plan.rules)) <= {
        "Wildcard",
        "Free Hit",
        "Bench Boost",
        "Triple Captain",
    }
    overridden = rolling_planner(entries, TEAM, context, 1)
    assert overridden.spec.free_transfers == 1 and overridden.state.free_transfers == 1
    assert "set by you (/setft)" in overridden.spec.notes[-1]
    with pytest.raises(EntryError, match="FPL has no team with ID 8"):
        rolling_planner(entries, 8, context)


# --- messages ---------------------------------------------------------------------------------


def test_recommendation_text(entries, context):
    plan = rolling_planner(entries, TEAM, context)
    text = recommendation_text(plan)
    names = context.names
    best = ranked(plan.plans)[0].first
    assert text.startswith(f"<b>GW{GW} plan</b> · deadline ")
    assert f"Team {TEAM} · bank £0.5m" in text
    assert f"Captain {names[best.captain]} · vice {names[best.vice]}" in text
    assert "Save the FT" in text or plan.plans.roll is ranked(plan.plans)[0]
    assert "Chips are your call" in text and "can't be seen before the deadline" in text
    assert "aren't public" not in text  # the squad-file note is for the CLI
    assert len(text) < 4096
    titled = recommendation_text(plan, title="GW6 update", changes=["Captain changed: a > b"])
    assert titled.startswith("<b>GW6 update</b>") and "Captain changed: a &gt; b" in titled


def test_ties_are_called_ties(entries, context):
    plan = rolling_planner(entries, TEAM, context)
    best = ranked(plan.plans)[0]
    # Two plans within the gap of each other are labelled as equally good.
    twin = replace(best, objective=best.objective - 1e-9, mip_gap=0.01)
    tied = replace(plan, plans=replace(plan.plans, plans=(best, twin)))
    assert "Plans 1 and 2 are within the solver's precision" in recommendation_text(tied)
    far = replace(best, objective=best.objective - 1000, mip_gap=0.0)
    apart = replace(plan, plans=replace(plan.plans, plans=(replace(best, mip_gap=0.0), far)))
    assert "within the solver's precision" not in recommendation_text(apart).split("Save")[0]


def test_changes_text():
    names = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}
    before = {"transfers": [[1, 2]], "captain": 3, "vice": 4}
    assert changes_text(None, before, names) == []
    assert changes_text(before, dict(before), names) == []
    after = {"transfers": [[1, 5]], "captain": 4, "vice": 3}
    assert changes_text(before, after, names) == [
        "Transfers changed: was A → B, now A → E.",
        "Captain changed: was C, now D.",
        "Vice changed: was D, now C.",
    ]


def test_split_message():
    assert split_message("short") == ["short"]
    assert split_message("") == [""]
    lines = [f"line {i:03d}" for i in range(100)]
    pieces = split_message("\n".join(lines), limit=50)
    assert all(len(p) <= 50 for p in pieces)
    assert "\n".join(pieces) == "\n".join(lines)
    long = "x" * 120
    assert split_message(f"a\n{long}\nb", limit=50) == ["a", "x" * 50, "x" * 50, "x" * 20 + "\nb"]


# --- commands ---------------------------------------------------------------------------------


def test_access_is_invite_only(service, users):
    replies = service.start(5)
    assert texts(replies, 5) == [PENDING]
    assert texts(replies, ADMIN) == ["New user asks for access: chat 5. /allow 5"]
    assert users.user(5).status == "pending"
    assert texts(service.start(5)) == [PENDING]  # the admin is told once
    assert texts(service.plan(5)) == [PENDING]
    assert texts(service.register(5, ["7"])) == [PENDING]
    assert texts(service.register(6, ["7"])) == ["Send /start first."]
    # Admin-only commands look unknown to others.
    assert texts(service.allow(5, ["5"])) == [PENDING]
    replies = service.allow(ADMIN, ["5"])
    assert texts(replies, ADMIN) == ["Chat 5: active."]
    assert texts(replies, 5)[0].startswith("You have access.")
    assert users.user(5).active
    assert texts(service.block(ADMIN, ["5"])) == ["Chat 5: blocked."]
    assert service.plan(5) == [] and service.start(5) == []  # blocked: silence
    assert texts(service.block(ADMIN, [str(ADMIN)])) == ["That's you."]
    assert texts(service.allow(ADMIN, ["x"])) == ["Usage: /allow &lt;chat id&gt;"]
    assert "chat 5: blocked" in texts(service.list_users(ADMIN))[0]


def test_admin_is_active_from_the_start(service, users):
    assert texts(service.start(ADMIN))[0].startswith(HELP)
    assert users.user(ADMIN).active
    assert "Admin:" in texts(service.help(ADMIN))[0] and "Admin:" not in texts(service.help(5))[0]


@pytest.fixture
def member(service, users):
    users.add_user(5, "active")
    return 5


def test_register(service, users, member, entries):
    usage = texts(service.register(member, []))[0]
    assert usage.startswith("Usage: /register")
    assert texts(service.register(member, ["abc"]))[0].startswith("Usage")
    assert texts(service.register(member, ["8"])) == ["FPL has no team with ID 8."]
    assert users.user(member).entry_id is None
    text = texts(service.register(member, [str(TEAM)]))[0]
    assert text.startswith("Registered.") and f"<b>Team {TEAM}</b> before GW{GW}" in text
    assert "Bank £0.5m" in text and f"/plan for the GW{GW} plan" in text
    user = users.user(member)
    assert user.entry_id == TEAM
    assert users.squad(user.user_id, SEASON, GW)["bank"] == 5


def test_register_without_a_context(users, entries, member):
    service = BotService(users, ContextStore(users.path.parent / "none"), entries, ADMIN)
    assert texts(service.register(member, ["8"])) == ["FPL has no team with ID 8."]
    assert texts(service.register(member, [str(TEAM)])) == [f"Registered team {TEAM}.\n{NOT_READY}"]
    assert texts(service.plan(member)) == [NOT_READY]


def test_plan(service, users, member):
    assert texts(service.plan(member))[0].startswith("Register your team first")
    service.register(member, [str(TEAM)])
    text = texts(service.plan(member))[0]
    assert text.startswith(f"<b>GW{GW} plan</b>")
    (rec,) = [r for r in users.recommendations(users.user(member).user_id) if r.kind == "plan"]
    assert rec.sent and rec.decision["captain"] > 0 and rec.season == SEASON and rec.gw == GW


def test_plan_after_the_deadline_waits_for_the_next_context(
    users, contexts, entries, context, member
):
    late = BotService(users, contexts, entries, ADMIN, now=lambda: context.deadline)
    users.set_entry(users.user(member).user_id, TEAM)
    assert texts(late.plan(member)) == [NOT_READY]


def test_setft_and_alerts(service, users, member):
    user = users.user(member)
    text = texts(service.setft(member, ["2"]))[0]
    assert text.startswith(f"GW{GW} (deadline ") and "planning with 2 free transfers" in text
    assert users.free_transfers(user.user_id, SEASON, GW) == 2
    assert texts(service.setft(member, ["9"]))[0].startswith("Usage: /setft &lt;0-")
    assert texts(service.setft(member, ["clear"]))[0].endswith("from the import again.")
    assert users.free_transfers(user.user_id, SEASON, GW) is None
    assert texts(service.alerts(member, []))[0].startswith("Alerts are on.")
    assert texts(service.alerts(member, ["off"])) == ["Alerts off."]
    assert not users.user(member).alerts
    assert texts(service.unknown(member)) == ["Unknown command. /help lists them."]


def test_plan_uses_the_ft_override(service, users, member):
    service.register(member, [str(TEAM)])
    service.setft(member, ["1"])
    assert "1 FT" in texts(service.plan(member))[0]


# --- deadline alerts --------------------------------------------------------------------------


class FakeSource:
    """FPL's bootstrap and fixtures for snapshots: the synthetic 2023/24 deadlines."""

    def __init__(self, deadlines):
        self.events = [
            {"id": g, "deadline_time": d.isoformat().replace("+00:00", "Z")}
            for g, d in sorted(deadlines.items())
        ]
        self.calls = 0

    def bootstrap_static(self):
        self.calls += 1
        return json.dumps({"events": self.events}).encode()

    def fixtures(self):
        return b"[]"


class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def alerts_env(tmp_path, league, deadlines, users, entries):
    """A live tick on the synthetic league: raw store with one bootstrap a day before the
    GW6 alert, a member with team 7, and a recording Telegram."""
    raw = RawStore(tmp_path / "raw")
    source = FakeSource(deadlines)
    deadline = deadlines[GW]
    raw.write("fpl", "bootstrap-static", source.bootstrap_static(), deadline - timedelta(hours=40))
    contexts = ContextStore(tmp_path / "state" / "context")
    clock = Clock(deadline - timedelta(hours=30))
    service = BotService(users, contexts, entries, ADMIN, now=clock, planner=rolling_planner)
    member = users.add_user(5, "active")
    users.set_entry(member.user_id, TEAM)
    sent: list[tuple[int, str]] = []
    builds: list[int] = []
    env = {
        "raw": raw,
        "source": source,
        "contexts": contexts,
        "clock": clock,
        "service": service,
        "sent": sent,
        "builds": builds,
        "deadline": deadline,
        "member": member,
    }

    def tick(send=None):
        return live.live_tick(
            raw=raw,
            data=lambda: league,
            users=users,
            contexts=contexts,
            service=service,
            fpl=source,
            send=send or (lambda chat, text: sent.append((chat, text))),
            build=lambda: builds.append(1),
            sha="abc",
            now=clock,
            xp_model="rolling",
        )

    env["tick"] = tick
    return env


def test_due_alerts(users, deadlines):
    from datetime import UTC

    d = deadlines[GW]
    event = (SEASON, GW, d)
    assert live.due_alerts(users, None, d, []) is None
    assert live.due_alerts(users, event, d - timedelta(hours=25), []) is None
    assert live.due_alerts(users, event, d - timedelta(hours=24), []).kinds == ("alert24",)
    assert live.due_alerts(users, event, d - timedelta(minutes=16), []).kinds == ("alert24",)
    assert live.due_alerts(users, event, d - timedelta(minutes=15), []) is None
    users.mark_alert_run(SEASON, GW, "alert24", "c", d - timedelta(hours=23))
    assert live.due_alerts(users, event, d - timedelta(hours=10), []) is None
    window = d - timedelta(hours=2)
    # The re-run waits for the archiver's pre-deadline snapshot.
    assert live.due_alerts(users, event, window + timedelta(minutes=5), []) is None
    snap = [window + timedelta(minutes=1)]
    assert live.due_alerts(users, event, window + timedelta(minutes=5), snap).kinds == ("alert2",)
    assert live.due_alerts(users, event, d - timedelta(minutes=30), snap) is None
    users.mark_alert_run(SEASON, GW, "alert2", "c", window)
    assert live.due_alerts(users, event, window + timedelta(minutes=5), snap) is None
    assert d.tzinfo is not None and d.utcoffset() == UTC.utcoffset(None)


def test_alert24_late_inside_the_window_skips_the_rerun(users, deadlines):
    d = deadlines[GW]
    event = (SEASON, GW, d)
    # The alert24 ran on the pre-deadline snapshot itself: no newer snapshot, no re-run.
    snap = d - timedelta(minutes=110)
    users.mark_alert_run(SEASON, GW, "alert24", "c", snap)
    assert live.due_alerts(users, event, d - timedelta(minutes=100), [snap]) is None


def test_next_event_and_context_stale(deadlines):
    source = FakeSource(deadlines)
    bootstrap = json.loads(source.bootstrap_static())
    d = deadlines[GW]
    assert live.next_event(bootstrap, d - timedelta(hours=1)) == (SEASON, GW, d)
    assert live.next_event(bootstrap, max(deadlines.values())) is None
    meta = {"fingerprint": "f", "git_sha": "s", "deadline": d.isoformat()}
    assert live.context_stale(None, "f", "s", d - timedelta(hours=1))
    assert not live.context_stale(meta, "f", "s", d - timedelta(hours=1))
    assert live.context_stale(meta, "g", "s", d - timedelta(hours=1))
    assert live.context_stale(meta, "f", "t", d - timedelta(hours=1))
    assert live.context_stale(meta, "f", "s", d)


def test_live_tick_alert24_then_silent_rerun(alerts_env, users):
    env = alerts_env
    tick, clock, deadline = env["tick"], env["clock"], env["deadline"]
    result = tick()  # a day and a half out: a context, no alert
    assert result.due is None and result.rebuilt and env["builds"] == [1] and env["sent"] == []
    assert tick().rebuilt is False  # nothing new archived: no rebuild

    clock.now = deadline - timedelta(hours=23)
    result = tick()
    assert result.due.kinds == ("alert24",) and result.snapshot and env["source"].calls == 2
    assert result.rebuilt  # the fresh snapshot is new archive content
    assert result.sent == [env["member"].user_id]
    ((chat, text),) = env["sent"]
    assert chat == 5 and text.startswith(f"<b>GW{GW} plan</b>")
    assert users.alert_run(SEASON, GW, "alert24") is not None
    clock.now += timedelta(minutes=15)
    assert tick().due is None and len(env["sent"]) == 1  # once per GW

    # The archiver's pre-deadline snapshot, then the re-run: same plan, no message.
    window = deadline - timedelta(hours=2)
    env["raw"].write(
        "fpl", "bootstrap-static", env["source"].bootstrap_static(), window + timedelta(minutes=1)
    )
    clock.now = window + timedelta(minutes=5)
    result = tick()
    assert result.due.kinds == ("alert2",) and result.rebuilt
    assert result.silent == [env["member"].user_id] and len(env["sent"]) == 1
    kinds = [r.kind for r in users.recommendations(env["member"].user_id)]
    assert kinds == ["alert24", "alert2"]


def test_live_tick_rerun_sends_what_changed(alerts_env, users, monkeypatch):
    env = alerts_env
    clock, deadline = env["clock"], env["deadline"]
    clock.now = deadline - timedelta(hours=23)
    env["tick"]()
    user_id = env["member"].user_id
    first = users.last_sent_alert(user_id, SEASON, GW).decision
    window = deadline - timedelta(hours=2)
    env["raw"].write(
        "fpl", "bootstrap-static", env["source"].bootstrap_static(), window + timedelta(minutes=1)
    )
    clock.now = window + timedelta(minutes=5)
    # A different captain now.
    real = live.decision_summary
    monkeypatch.setattr(
        live,
        "decision_summary",
        lambda plan: {**real(plan), "captain": first["vice"], "vice": first["captain"]},
    )
    result = env["tick"]()
    assert result.sent == [user_id]
    text = env["sent"][-1][1]
    assert text.startswith(f"<b>GW{GW} update</b>") and "Captain changed: was " in text


def test_live_tick_retries_failed_sends(alerts_env, users):
    env = alerts_env
    env["clock"].now = env["deadline"] - timedelta(hours=23)

    def broken(chat, text):
        raise RuntimeError("telegram down")

    with pytest.raises(RuntimeError, match=r"alerts failed for 1 user\(s\)"):
        env["tick"](send=broken)
    assert users.alert_run(SEASON, GW, "alert24") is None  # not done: retried
    env["clock"].now += timedelta(minutes=15)
    result = env["tick"]()
    assert result.sent == [env["member"].user_id] and len(env["sent"]) == 1
    assert users.alert_run(SEASON, GW, "alert24") is not None


def test_live_tick_tells_the_user_when_the_squad_cant_be_imported(alerts_env, users):
    env = alerts_env
    users.set_entry(env["member"].user_id, 8)  # FPL doesn't know it
    env["clock"].now = env["deadline"] - timedelta(hours=23)
    result = env["tick"]()
    assert result.sent == [] and result.failed == []
    ((chat, text),) = env["sent"]
    assert chat == 5 and "Couldn't plan GW6 for team 8: FPL has no team with ID 8" in text
    assert users.alert_run(SEASON, GW, "alert24") is not None


def test_live_tick_refuses_a_context_behind_the_archive(alerts_env, deadlines, monkeypatch):
    env = alerts_env
    env["clock"].now = env["deadline"] - timedelta(hours=23)
    # data/ that still ends a GW earlier than the archive's next deadline.
    monkeypatch.setattr(
        live, "build_context", lambda *a, **k: replace(build_context(*a, **k), gw=GW - 1)
    )
    with pytest.raises(RuntimeError, match="data/ is behind the archive"):
        env["tick"]()


# --- the Telegram layer -----------------------------------------------------------------------


def test_every_command_maps_to_a_service_method():
    from fplopt.bot.app import COMMANDS

    for method, _ in COMMANDS.values():
        assert callable(getattr(BotService, method))


def test_build_application_registers_the_commands(service):
    from telegram.ext import CommandHandler

    from fplopt.bot.app import COMMANDS, build_application

    application = build_application("123456:TEST", service)
    commands = {
        c
        for handlers in application.handlers.values()
        for h in handlers
        if isinstance(h, CommandHandler)
        for c in h.commands
    }
    assert commands == set(COMMANDS)


def test_send_replies_splits_long_messages():
    from fplopt.bot.app import send_replies
    from fplopt.bot.service import Reply

    class Bot:
        def __init__(self):
            self.sent = []

        async def send_message(self, chat_id, text):
            self.sent.append((chat_id, text))

    bot = Bot()
    long = "\n".join("x" * 100 for _ in range(60))  # ~6,000 characters
    asyncio.run(send_replies(bot, [Reply(1, "hi"), Reply(2, long)]))
    assert bot.sent[0] == (1, "hi")
    assert [c for c, _ in bot.sent[1:]] == [2, 2]
    assert all(len(t) <= 4096 for _, t in bot.sent)


def test_telegram_sender_posts_html_and_never_leaks_the_token():
    from fplopt.bot.sender import TelegramSendError, telegram_sender

    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        status = 403 if seen[-1]["chat_id"] == 2 else 200
        return httpx.Response(status, json={"ok": status == 200})

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        send = telegram_sender(http, "123:SECRET")
        send(1, "<b>hi</b>")
        assert seen[0] == {
            "chat_id": 1,
            "text": "<b>hi</b>",
            "parse_mode": "HTML",
            "link_preview_options": {"is_disabled": True},
        }
        with pytest.raises(TelegramSendError) as info:
            send(2, "x")
    assert "HTTP 403" in str(info.value) and "SECRET" not in str(info.value)
    assert info.value.__cause__ is None and info.value.__suppress_context__


# --- review fixes: retries, undeliverable chats, off-season, polling errors -------------------


def test_due_alerts_rerun_does_not_wait_for_a_stuck_alert24(users, deadlines):
    d = deadlines[GW]
    event = (SEASON, GW, d)
    window = d - timedelta(hours=2)
    # alert24 never marked done (a user kept failing): the re-run is still due.
    due = live.due_alerts(users, event, window + timedelta(minutes=5), [window + timedelta(1)])
    assert due.kinds == ("alert24",)  # a snapshot after the deadline doesn't count
    snap = [window + timedelta(minutes=1)]
    due = live.due_alerts(users, event, window + timedelta(minutes=5), snap)
    assert due.kinds == ("alert24", "alert2")


def test_live_tick_gives_up_after_max_attempts_and_snapshots_once(alerts_env, users):
    env = alerts_env
    env["clock"].now = env["deadline"] - timedelta(hours=23)

    def broken(chat, text):
        raise RuntimeError("telegram down")

    for attempt in range(live.MAX_ATTEMPTS):
        with pytest.raises(RuntimeError, match="alerts failed"):
            env["tick"](send=broken)
        assert users.alert_run(SEASON, GW, "alert24") is None or attempt == live.MAX_ATTEMPTS - 1
        env["clock"].now += timedelta(minutes=15)
    assert users.alert_run(SEASON, GW, "alert24") is not None  # given up: done
    assert env["source"].calls == 2  # one fresh snapshot for the GW, not one per retry
    result = env["tick"](send=broken)  # nothing due, nothing retried
    assert result.due is None and result.failed == []


def test_live_tick_undeliverable_chat_turns_alerts_off(alerts_env, users):
    from fplopt.bot.sender import TelegramSendError

    env = alerts_env
    env["clock"].now = env["deadline"] - timedelta(hours=23)

    def blocked(chat, text):
        raise TelegramSendError("sendMessage failed: HTTPStatusError (HTTP 403)", permanent=True)

    result = env["tick"](send=blocked)
    assert result.failed == [] and result.sent == []
    assert not users.user(5).alerts
    assert users.alert_run(SEASON, GW, "alert24") is not None
    (rec,) = users.recommendations(env["member"].user_id)
    assert rec.error.startswith("undeliverable") and not rec.sent


def test_live_tick_one_users_failure_does_not_stop_the_others(alerts_env, users, monkeypatch):
    env = alerts_env
    other = users.add_user(6, "active")
    users.set_entry(other.user_id, TEAM)
    env["clock"].now = env["deadline"] - timedelta(hours=23)
    real = live.recommendation_text
    calls = []

    def flaky(plan, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise KeyError("player")
        return real(plan, **kwargs)

    monkeypatch.setattr(live, "recommendation_text", flaky)
    with pytest.raises(RuntimeError, match=r"failed for 1 user\(s\)"):
        env["tick"]()
    assert [chat for chat, _ in env["sent"]] == [6]


def test_live_tick_does_nothing_off_season(alerts_env, deadlines):
    env = alerts_env
    env["clock"].now = max(deadlines.values()) + timedelta(days=1)
    result = env["tick"]()
    assert result.due is None and not result.rebuilt and env["builds"] == []


def test_polling_network_errors_are_not_alerted(service, monkeypatch):
    from telegram.error import Conflict, NetworkError

    from fplopt.bot import app

    alerts = []

    def alert(text, **kwargs):
        alerts.append(text)

    monkeypatch.setattr(app, "send_admin_alert", alert)
    application = app.build_application("123456:TEST", service)
    (on_error,) = application.error_handlers

    class Ctx:
        def __init__(self, error):
            self.error = error

    for error in (NetworkError("Bad Gateway"), Conflict("terminated by other getUpdates")):
        asyncio.run(on_error(None, Ctx(error)))
    assert alerts == []
    asyncio.run(on_error(None, Ctx(ValueError("boom"))))
    assert alerts == ["fplopt bot: ValueError: boom"]


def test_send_replies_skips_chats_that_refuse():
    from telegram.error import Forbidden

    from fplopt.bot.app import send_replies
    from fplopt.bot.service import Reply

    class Bot:
        def __init__(self):
            self.sent = []

        async def send_message(self, chat_id, text):
            if chat_id == 2:
                raise Forbidden("bot can't initiate conversation with a user")
            self.sent.append(chat_id)

    bot = Bot()
    asyncio.run(send_replies(bot, [Reply(1, "a"), Reply(2, "b"), Reply(3, "c")]))
    assert bot.sent == [1, 3]


# --- /chips (information only) ----------------------------------------------------------------


def chip_planner(fpl, entry_id, context, ft=None):
    from fplopt.bot.planner import chip_options

    return chip_options(fpl, entry_id, context, ft, params=TEST_PARAMS)


def test_scenario_label():
    from fplopt.bot.messages import scenario_label

    assert scenario_label("none") == "no chip"
    assert scenario_label("bboost@gw9") == "Bench Boost GW9"
    assert scenario_label("wildcard@gw5+3xc@gw9") == "Wildcard GW5 + Triple Captain GW9"


def test_chip_options_and_text(entries, context):
    from fplopt.bot.messages import chips_text

    options = chip_planner(entries, TEAM, context)
    objectives = options.plans.scenario_objectives
    assert "none" in objectives and options.plans.best.scenario in objectives
    available = {"bboost", "3xc", "freehit", "wildcard"} & {
        c.name for c in options.rules.chips if c.start <= GW <= c.stop
    }
    assert set(options.singles) <= available and options.singles
    for name, (label, value) in options.singles.items():
        assert label == f"{name}@gw{GW}"  # a 1-GW horizon: the only GW
        assert value <= options.plans.best.total_objective + 1e-6
    text = chips_text(options)
    assert text.startswith(f"<b>GW{GW} chips</b> · information only")
    assert "Each chip on its own" in text and "Best over the next 1 GWs" in text
    assert "Unused chips: " in text and "The bot's plans don't play chips." in text
    assert len(text) < 4096


def test_chips_command(users, contexts, entries, context, member):
    service = BotService(
        users,
        contexts,
        entries,
        ADMIN,
        now=lambda: context.deadline - timedelta(hours=20),
        chip_planner=chip_planner,
    )
    assert texts(service.chips(member))[0].startswith("Register your team first")
    users.set_entry(users.user(member).user_id, TEAM)
    text = texts(service.chips(member))[0]
    assert text.startswith(f"<b>GW{GW} chips</b>")
    (check,) = users.chip_checks(users.user(member).user_id)
    assert check["gw"] == GW and "none" in check["scenarios"] and check["best"]
    assert texts(service.chips(6)) == ["Send /start first."]
