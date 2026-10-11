"""The deadline alerts (Phase 7 plan §3): `fplopt live tick`, every 15 minutes from a
systemd timer, in its own process (it exits and frees the model's memory; failures alert
the admin like every job).

Each run:
1. **Due?** From the newest bootstrap, the next deadline d and its GW.
   - `alert24` is due from d − 24 h until d − 15 min, once per GW.
   - `alert2` (the silent re-run) is due from the archiver's pre-deadline snapshot (taken in
     [d − 2 h, d), after the one alert24 planned on, if it ran) until d − 30 min, once per
     GW.
   Off-season (no deadline ahead) nothing runs, not even a rebuild.
2. **Fresh flags for alert24:** an FPL snapshot (fixtures + bootstrap, like the tick's),
   once per GW (a retry reuses it).
3. **Context:** rebuilt (`build all` in a subprocess, then `context.build_context`) when the
   archive's fingerprint, the code's commit or the next deadline changed.
4. **Users** (active, with a team, alerts on), each once per GW and kind (a crashed run
   resumes where it stopped):
   - alert24: the full recommendation;
   - alert2: sent only if plan 1's moves, captain or vice differ from the last alert sent
     (the message says what changed), else just logged. A user who got no alert24 gets the
     full recommendation.
   An `EntryError` (the squad can't be imported) is sent to the user and logged. A chat
   Telegram refuses for good (403, or 400: the user blocked the bot or the chat is gone)
   turns the user's alerts off and is logged. Any other failure is logged, the other users
   still get theirs, the job fails at the end, and the next run retries the failed users,
   up to MAX_ATTEMPTS tries each (the GW's run is marked done once nobody is left to
   retry).
"""

from __future__ import annotations

import html
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from fplopt.bot.context import (
    LIVE_XP,
    ContextStore,
    PlanContext,
    build_context,
    raw_fingerprint,
)
from fplopt.bot.messages import changes_text, recommendation_text, split_message
from fplopt.bot.planner import decision_summary, plans_json
from fplopt.bot.sender import TelegramSendError
from fplopt.bot.service import BotService
from fplopt.bot.store import User, UserStore
from fplopt.entry import EntryError
from fplopt.features.store import DataStore
from fplopt.ingest.jobs import FplSource, snapshot_fpl
from fplopt.ingest.raw_store import RawStore
from fplopt.ingest.schedule import PRE_DEADLINE_LEAD
from fplopt.seasons import bootstrap_season

__all__ = (
    "ALERT24_LEAD",
    "Due",
    "TickResult",
    "context_stale",
    "due_alerts",
    "live_tick",
    "next_event",
    "refresh_context",
)

log = logging.getLogger(__name__)

ALERT24_LEAD = timedelta(hours=24)
ALERT24_CUTOFF = timedelta(minutes=15)  # no alert24 later than this before the deadline
ALERT2_CUTOFF = timedelta(minutes=30)  # no re-run later than this before the deadline
MAX_ATTEMPTS = 3  # tries per user, GW and kind before an alert is given up
# Recorded errors that end a user's alert for the GW: a squad that can't be imported
# (told to the user) and an undeliverable chat.
DONE_ERRORS = ("EntryError", "undeliverable")


@dataclass(frozen=True)
class Due:
    season: int
    gw: int
    deadline: datetime
    kinds: tuple[str, ...]


@dataclass
class TickResult:
    due: Due | None = None
    snapshot: bool = False
    context_id: str | None = None
    rebuilt: bool = False
    sent: list[int] = field(default_factory=list)  # user_ids messaged
    silent: list[int] = field(default_factory=list)  # user_ids re-run without a change
    failed: list[int] = field(default_factory=list)


def next_event(bootstrap: dict[str, Any], now: datetime) -> tuple[int, int, datetime] | None:
    """(season, GW, deadline) of the first deadline after `now` in a bootstrap."""
    events = []
    for event in bootstrap.get("events") or []:
        deadline = datetime.fromisoformat(event["deadline_time"].replace("Z", "+00:00"))
        if deadline > now:
            events.append((deadline, int(event["id"])))
    if not events:
        return None
    deadline, gw = min(events)
    return bootstrap_season(bootstrap), gw, deadline


def due_alerts(
    users: UserStore,
    event: tuple[int, int, datetime] | None,
    now: datetime,
    bootstrap_times: list[datetime],
) -> Due | None:
    """The alert kinds due at `now` (module docstring), or None. The re-run doesn't wait
    for the 24-hour alert to be marked done (one user whose alerts keep failing must not
    cost everyone the re-run); it needs a pre-deadline snapshot newer than the one the
    24-hour alert planned on, if that ran."""
    if event is None:
        return None
    season, gw, deadline = event
    kinds = []
    alert24 = users.alert_run(season, gw, "alert24")
    if alert24 is None and deadline - ALERT24_LEAD <= now < deadline - ALERT24_CUTOFF:
        kinds.append("alert24")
    if (
        users.alert_run(season, gw, "alert2") is None
        and deadline - PRE_DEADLINE_LEAD <= now < deadline - ALERT2_CUTOFF
    ):
        after = None if alert24 is None else datetime.fromisoformat(alert24["bootstrap_at"])
        window = deadline - PRE_DEADLINE_LEAD
        if any(window <= t < deadline and (after is None or t > after) for t in bootstrap_times):
            kinds.append("alert2")
    return Due(season, gw, deadline, tuple(kinds)) if kinds else None


def context_stale(
    meta: dict[str, Any] | None, fingerprint: str, sha: str | None, now: datetime
) -> bool:
    """Whether a new context is needed: none yet, new archive content, new code, or its
    deadline has passed."""
    if meta is None:
        return True
    return (
        meta["fingerprint"] != fingerprint
        or meta["git_sha"] != sha
        or datetime.fromisoformat(meta["deadline"]) <= now
    )


def refresh_context(
    *,
    raw: RawStore,
    data: Callable[[], DataStore],
    contexts: ContextStore,
    build: Callable[[], None],
    sha: str | None,
    now: Callable[[], datetime],
    force: bool = False,
    xp_model: str = LIVE_XP,
) -> tuple[PlanContext | None, bool]:
    """The newest plan context, rebuilt first (`build()`, then `build_context` on `data()`,
    the rebuilt `data/`) if it is stale or `force`; returns (context or None, whether it was
    rebuilt)."""
    fingerprint = raw_fingerprint(raw.root)
    if force or context_stale(contexts.meta(), fingerprint, sha, now()):
        log.info("archive, code or deadline changed: rebuilding data/ and the plan context")
        build()
        times = raw.times("fpl", "bootstrap-static")
        context = build_context(
            data(),
            now(),
            fingerprint=fingerprint,
            bootstrap_at=times[-1] if times else None,
            sha=sha,
            xp_model=xp_model,
        )
        if context is None:
            log.info("no deadline ahead in the data: no context")
            return None, False
        contexts.write(context)
        contexts.prune()
        log.info("plan context %s: %d GW%d", context.context_id, context.season, context.gw)
        return contexts.latest(), True
    return contexts.latest(), False


def live_tick(
    *,
    raw: RawStore,
    data: Callable[[], DataStore],
    users: UserStore,
    contexts: ContextStore,
    service: BotService,
    fpl: FplSource,
    send: Callable[[int, str], None],
    build: Callable[[], None],
    sha: str | None,
    now: Callable[[], datetime],
    xp_model: str = LIVE_XP,
) -> TickResult:
    """One run (module docstring). `send(chat_id, html)` raises on failure; `build()`
    rebuilds `data/` from `raw/` and `data()` opens it."""
    result = TickResult()
    latest = raw.latest("fpl", "bootstrap-static")
    if latest is None:
        log.info("no bootstrap snapshot yet: nothing to do")
        return result
    current = now()
    event = next_event(raw.read_json(latest), current)
    if event is None:  # off-season: no deadline ahead, nothing to plan or rebuild
        log.info("no deadline ahead in the newest bootstrap: nothing to do")
        return result
    times = raw.times("fpl", "bootstrap-static")
    due = due_alerts(users, event, current, times)
    result.due = due
    if (
        due is not None
        and "alert24" in due.kinds
        and not any(t >= due.deadline - ALERT24_LEAD for t in times)
    ):
        # Once per GW: a retried alert24 (failed users) reuses this snapshot, so retries
        # don't grow raw/ or rebuild the context every run.
        log.info("GW%d alert24 due: fresh FPL snapshot", due.gw)
        snapshot_fpl(raw, fpl, now)
        result.snapshot = True
    context, result.rebuilt = refresh_context(
        raw=raw, data=data, contexts=contexts, build=build, sha=sha, now=now, xp_model=xp_model
    )
    if context is None:
        return result
    result.context_id = context.context_id
    if due is None:
        return result
    if (context.season, context.gw) != (due.season, due.gw):
        raise RuntimeError(
            f"the plan context is for {context.season} GW{context.gw} but the next deadline "
            f"is {due.season} GW{due.gw}: data/ is behind the archive"
        )
    for kind in due.kinds:
        _alert(kind, context, users, service, send, result)
        if not _retry_left(kind, context, users):
            users.mark_alert_run(
                due.season,
                due.gw,
                kind,
                context.context_id,
                context.bootstrap_at or context.created_at,
            )
    if result.failed:
        raise RuntimeError(
            f"GW{due.gw} alerts failed for {len(result.failed)} user(s) (user_ids "
            f"{result.failed}); the others were sent"
        )
    return result


def _alert(
    kind: str,
    context: PlanContext,
    users: UserStore,
    service: BotService,
    send: Callable[[int, str], None],
    result: TickResult,
) -> None:
    for user in users.alert_users():
        records = [
            r
            for r in users.recommendations(user.user_id)
            if r.season == context.season and r.gw == context.gw and r.kind == kind
        ]
        if any(r.error is None or r.error.startswith(DONE_ERRORS) for r in records):
            continue  # sent, silent, or a final error (told to the user, or undeliverable)
        if len(records) >= MAX_ATTEMPTS:
            continue  # failed MAX_ATTEMPTS times: given up (each failure was alerted)
        try:
            _alert_user(kind, context, user, users, service, send, result)
        except Exception:
            log.exception("GW%d %s failed for user %d", context.gw, kind, user.user_id)
            users.record(
                user.user_id,
                context.season,
                context.gw,
                kind,
                context_id=context.context_id,
                error="failed (see the job log)",
            )
            result.failed.append(user.user_id)


def _alert_user(
    kind: str,
    context: PlanContext,
    user: User,
    users: UserStore,
    service: BotService,
    send: Callable[[int, str], None],
    result: TickResult,
) -> None:
    try:
        plan = service.make_plan(user, context, kind)
    except EntryError as exc:
        try:
            _send(
                send,
                user.chat_id,
                f"Couldn't plan GW{context.gw} for team {user.entry_id}: "
                + html.escape(str(exc), quote=False),
            )
        except Exception:
            log.exception("telling user %d about their squad failed", user.user_id)
        return
    decision = decision_summary(plan)
    last = users.last_sent_alert(user.user_id, context.season, context.gw)
    record = {
        "context_id": context.context_id,
        "plans": plans_json(plan),
        "decision": decision,
    }
    if kind == "alert2" and last is not None:
        changes = changes_text(last.decision, decision, context.names)
        if not changes:
            users.record(user.user_id, context.season, context.gw, kind, **record)
            result.silent.append(user.user_id)
            return
        text = recommendation_text(plan, title=f"GW{context.gw} update", changes=changes)
    else:
        text = recommendation_text(plan)
    try:
        _send(send, user.chat_id, text)
    except TelegramSendError as exc:
        if not exc.permanent:
            raise
        # The user blocked the bot or the chat is gone: stop alerting them (/alerts on
        # turns it back on) and don't retry.
        log.warning("GW%d %s undeliverable to user %d: %s", context.gw, kind, user.user_id, exc)
        users.set_alerts(user.user_id, False)
        users.record(
            user.user_id,
            context.season,
            context.gw,
            kind,
            message=text,
            error=f"undeliverable: {exc}",
            **record,
        )
        return
    users.record(user.user_id, context.season, context.gw, kind, message=text, sent=True, **record)
    result.sent.append(user.user_id)


def _retry_left(kind: str, context: PlanContext, users: UserStore) -> bool:
    """Whether a user's alert of this kind failed and has tries left (the next run retries
    it, so the GW's run isn't done yet)."""
    for user in users.alert_users():
        records = [
            r
            for r in users.recommendations(user.user_id)
            if r.season == context.season and r.gw == context.gw and r.kind == kind
        ]
        if any(r.error is None or r.error.startswith(DONE_ERRORS) for r in records):
            continue
        if records and len(records) < MAX_ATTEMPTS:
            return True
    return False


def _send(send: Callable[[int, str], None], chat_id: int, text: str) -> None:
    for piece in split_message(text):
        send(chat_id, piece)
