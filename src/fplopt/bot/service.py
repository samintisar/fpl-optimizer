"""The bot's commands as plain functions (Phase 7 plan §4, §6): each takes the sender's chat
id and the command's arguments and returns the replies to send, so the logic is tested
without Telegram (`fplopt.bot.app` is the thin python-telegram-bot layer around it).

Access is invite-only: `/start` from a new chat makes a `pending` user and tells the admin
(`TELEGRAM_ADMIN_CHAT_ID`, active from the start), who answers `/allow <chat_id>`.
"""

from __future__ import annotations

import html
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

from fplopt.bot.context import ContextStore, PlanContext
from fplopt.bot.messages import chips_text, recommendation_text, squad_text, uk_time
from fplopt.bot.planner import (
    ChipOptions,
    UnknownTeam,
    UserPlan,
    chip_options,
    decision_summary,
    import_user_squad,
    plan_user,
    plans_json,
)
from fplopt.bot.store import User, UserStore
from fplopt.entry import EntryClient, EntryError, SquadSpec

__all__ = ("HELP", "BotService", "Reply")

log = logging.getLogger(__name__)

HELP = "\n".join(
    [
        "<b>FPL optimizer</b>: transfer plans for your FPL team, from the model that passed "
        "its 2025/26 test. You make the final call.",
        "",
        "/register &lt;team ID&gt; · your FPL team (the number in your team's URL)",
        "/plan · plans for the next deadline",
        "/chips · what the (untested) chip planner would do: information only",
        "/setft &lt;n&gt; · correct the free transfers for the next GW (/setft clear to undo)",
        "/alerts on|off · the message 24 hours before each deadline (and a re-run about 2 "
        "hours before, only if the plan changes)",
        "/help · this list",
    ]
)
ADMIN_HELP = "Admin: /allow &lt;chat id&gt;, /block &lt;chat id&gt;, /users"
PENDING = "Waiting for access: the admin has been asked."
NOT_READY = (
    "The data for the next deadline isn't ready yet (it's rebuilt after each new snapshot). "
    "Try again in a few minutes."
)


@dataclass(frozen=True)
class Reply:
    chat_id: int
    text: str


def _e(text: object) -> str:
    return html.escape(str(text), quote=False)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class BotService:
    """Command logic over the user store, the plan contexts and FPL's public API."""

    def __init__(
        self,
        users: UserStore,
        contexts: ContextStore,
        fpl: EntryClient,
        admin_chat_id: int | None,
        *,
        now: Callable[[], datetime] = _utc_now,
        planner: Callable[..., UserPlan] = plan_user,
        chip_planner: Callable[..., ChipOptions] = chip_options,
    ) -> None:
        self.users = users
        self.contexts = contexts
        self.fpl = fpl
        self.admin_chat_id = admin_chat_id
        self.now = now
        self.planner = planner
        self.chip_planner = chip_planner

    # --- access ---------------------------------------------------------------------------

    def is_admin(self, chat_id: int) -> bool:
        return self.admin_chat_id is not None and int(chat_id) == int(self.admin_chat_id)

    def _active(self, chat_id: int) -> tuple[User | None, list[Reply]]:
        """The active user for `chat_id`, or the replies that say why not."""
        user = self.users.user(chat_id)
        if user is None and self.is_admin(chat_id):
            user = self.users.add_user(chat_id, "active")
        if user is None:
            return None, [Reply(chat_id, "Send /start first.")]
        if user.status == "pending":
            return None, [Reply(chat_id, PENDING)]
        if user.status == "blocked":
            return None, []
        return user, []

    def start(self, chat_id: int) -> list[Reply]:
        existing = self.users.user(chat_id)
        if self.is_admin(chat_id):
            if existing is None or existing.status != "active":
                self.users.add_user(chat_id, "active")
                self.users.set_status(chat_id, "active")
            return [Reply(chat_id, f"{HELP}\n\n{ADMIN_HELP}")]
        if existing is None:
            self.users.add_user(chat_id, "pending")
            replies = [Reply(chat_id, PENDING)]
            if self.admin_chat_id is not None:
                replies.append(
                    Reply(
                        int(self.admin_chat_id),
                        f"New user asks for access: chat {chat_id}. /allow {chat_id}",
                    )
                )
            return replies
        if existing.status == "pending":
            return [Reply(chat_id, PENDING)]
        if existing.status == "blocked":
            return []
        return [Reply(chat_id, HELP)]

    def help(self, chat_id: int) -> list[Reply]:
        extra = f"\n\n{ADMIN_HELP}" if self.is_admin(chat_id) else ""
        return [Reply(chat_id, HELP + extra)]

    @staticmethod
    def _admin_target(args: Sequence[str]) -> int | None:
        """The chat id argument of /allow and /block (group chats' ids are negative)."""
        if len(args) != 1 or not args[0].lstrip("-").isdigit():
            return None
        return int(args[0])

    def allow(self, chat_id: int, args: Sequence[str]) -> list[Reply]:
        return self._set_status(chat_id, args, "allow", "active")

    def block(self, chat_id: int, args: Sequence[str]) -> list[Reply]:
        return self._set_status(chat_id, args, "block", "blocked")

    def _set_status(
        self, chat_id: int, args: Sequence[str], command: str, status: str
    ) -> list[Reply]:
        if not self.is_admin(chat_id):
            return self.unknown(chat_id)
        target = self._admin_target(args)
        if target is None:
            return [Reply(chat_id, f"Usage: /{command} &lt;chat id&gt;")]
        if self.is_admin(target) and status == "blocked":
            return [Reply(chat_id, "That's you.")]
        if self.users.user(target) is None:
            self.users.add_user(target, status)
        self.users.set_status(target, status)
        replies = [Reply(chat_id, f"Chat {target}: {status}.")]
        if status == "active":
            replies.append(Reply(target, f"You have access.\n\n{HELP}"))
        return replies

    def list_users(self, chat_id: int) -> list[Reply]:
        if not self.is_admin(chat_id):
            return self.unknown(chat_id)
        rows = [
            f"chat {u.chat_id}: {u.status}, team {u.entry_id or '-'}, alerts "
            f"{'on' if u.alerts else 'off'}"
            for u in self.users.users()
        ]
        return [Reply(chat_id, "\n".join(rows) or "No users.")]

    def unknown(self, chat_id: int) -> list[Reply]:
        user, replies = self._active(chat_id)
        if user is None:
            return replies
        return [Reply(chat_id, "Unknown command. /help lists them.")]

    # --- the user's team ------------------------------------------------------------------

    def _context(self) -> PlanContext | None:
        """The newest context, if it is for a deadline still ahead."""
        context = self.contexts.latest()
        if context is None or context.deadline <= self.now():
            return None
        return context

    def register(self, chat_id: int, args: Sequence[str]) -> list[Reply]:
        user, replies = self._active(chat_id)
        if user is None:
            return replies
        if len(args) != 1 or not args[0].isdigit() or int(args[0]) < 1:
            return [
                Reply(
                    chat_id,
                    "Usage: /register &lt;team ID&gt;, the number in your FPL team's URL "
                    "(fantasy.premierleague.com/entry/<b>1234567</b>/event/…).",
                )
            ]
        entry_id = int(args[0])
        context = self._context()
        if context is None:
            try:
                self.fpl.entry_history(entry_id)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    return [Reply(chat_id, f"FPL has no team with ID {entry_id}.")]
                raise
            self.users.set_entry(user.user_id, entry_id)
            return [Reply(chat_id, f"Registered team {entry_id}.\n{NOT_READY}")]
        ft = self.users.free_transfers(user.user_id, context.season, context.gw)
        try:
            spec, _, _ = import_user_squad(self.fpl, entry_id, context, ft)
        except UnknownTeam as exc:
            return [Reply(chat_id, f"{_e(exc)}.")]
        except EntryError as exc:
            self.users.set_entry(user.user_id, entry_id)
            return [
                Reply(
                    chat_id,
                    f"Registered team {entry_id}, but its squad can't be planned yet: {_e(exc)}",
                )
            ]
        self.users.set_entry(user.user_id, entry_id)
        self._save_squad(user, entry_id, spec, context)
        text = squad_text(context, spec, entry_id)
        return [Reply(chat_id, f"Registered.\n\n{text}\n\n/plan for the GW{context.gw} plan.")]

    def _save_squad(self, user: User, entry_id: int, spec: SquadSpec, context: PlanContext) -> None:
        self.users.save_squad(
            user.user_id,
            context.season,
            context.gw,
            entry_id,
            {
                "players": [[p.element_id, p.purchase_price] for p in spec.players],
                "bank": spec.bank,
                "free_transfers": spec.free_transfers,
                "chips_used": [list(c) for c in spec.chips_used],
                "notes": list(spec.notes),
            },
            context.context_id,
        )

    def plan(self, chat_id: int) -> list[Reply]:
        user, replies = self._active(chat_id)
        if user is None:
            return replies
        if user.entry_id is None:
            return [Reply(chat_id, "Register your team first: /register &lt;team ID&gt;")]
        context = self._context()
        if context is None:
            return [Reply(chat_id, NOT_READY)]
        try:
            result = self.make_plan(user, context, "plan")
        except EntryError as exc:
            return [Reply(chat_id, f"Can't plan team {user.entry_id}: {_e(exc)}")]
        return [Reply(chat_id, recommendation_text(result))]

    def chips(self, chat_id: int) -> list[Reply]:
        """`/chips`: the chip scenario search on the user's squad, as information (the bot's
        plans don't play chips until a chip policy passes its own test)."""
        user, replies = self._active(chat_id)
        if user is None:
            return replies
        if user.entry_id is None:
            return [Reply(chat_id, "Register your team first: /register &lt;team ID&gt;")]
        context = self._context()
        if context is None:
            return [Reply(chat_id, NOT_READY)]
        ft = self.users.free_transfers(user.user_id, context.season, context.gw)
        try:
            options = self.chip_planner(self.fpl, user.entry_id, context, ft)
        except EntryError as exc:
            return [Reply(chat_id, f"Can't plan team {user.entry_id}: {_e(exc)}")]
        self.users.record_chip_check(
            user.user_id,
            context.season,
            context.gw,
            context_id=context.context_id,
            best=options.plans.best.scenario,
            scenarios=options.plans.scenario_objectives,
        )
        return [Reply(chat_id, chips_text(options))]

    def make_plan(self, user: User, context: PlanContext, kind: str) -> UserPlan:
        """Plan the user's team on `context` and log it (`kind`; EntryError is logged as
        the recommendation's error and re-raised)."""
        assert user.entry_id is not None
        ft = self.users.free_transfers(user.user_id, context.season, context.gw)
        try:
            result = self.planner(self.fpl, user.entry_id, context, ft)
        except EntryError as exc:
            self.users.record(
                user.user_id,
                context.season,
                context.gw,
                kind,
                context_id=context.context_id,
                error=f"EntryError: {exc}",
            )
            raise
        self._save_squad(user, user.entry_id, result.spec, context)
        if kind == "plan":
            self.users.record(
                user.user_id,
                context.season,
                context.gw,
                kind,
                context_id=context.context_id,
                plans=plans_json(result),
                decision=decision_summary(result),
                message=recommendation_text(result),
                sent=True,
            )
        return result

    def setft(self, chat_id: int, args: Sequence[str]) -> list[Reply]:
        user, replies = self._active(chat_id)
        if user is None:
            return replies
        context = self._context()
        if context is None:
            return [Reply(chat_id, NOT_READY)]
        from fplopt.backtest.rules import backtest_rules

        cap = backtest_rules(context.season).max_free_transfers
        if len(args) == 1 and args[0].lower() == "clear":
            self.users.set_free_transfers(user.user_id, context.season, context.gw, None)
            return [Reply(chat_id, f"GW{context.gw}: free transfers from the import again.")]
        if len(args) != 1 or not args[0].isdigit() or not 0 <= int(args[0]) <= cap:
            return [Reply(chat_id, f"Usage: /setft &lt;0-{cap}&gt; or /setft clear")]
        n = int(args[0])
        self.users.set_free_transfers(user.user_id, context.season, context.gw, n)
        return [
            Reply(
                chat_id,
                f"GW{context.gw} (deadline {uk_time(context.deadline)}): planning with {n} "
                "free transfers.",
            )
        ]

    def alerts(self, chat_id: int, args: Sequence[str]) -> list[Reply]:
        user, replies = self._active(chat_id)
        if user is None:
            return replies
        if len(args) != 1 or args[0].lower() not in ("on", "off"):
            state = "on" if user.alerts else "off"
            return [Reply(chat_id, f"Alerts are {state}. Usage: /alerts on|off")]
        on = args[0].lower() == "on"
        self.users.set_alerts(user.user_id, on)
        return [Reply(chat_id, f"Alerts {'on' if on else 'off'}.")]
