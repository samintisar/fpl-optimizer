"""The Telegram bot (`fplopt bot run`, a long-polling systemd user service; Phase 7 plan §4):
a thin python-telegram-bot layer over `service.BotService`.

Updates are handled one at a time (python-telegram-bot's default), each command's logic in a
worker thread (`asyncio.to_thread`: SQLite, FPL's API and the solver block), so a `/plan`
waits for the one before it. Private chats only; group messages are ignored. An unexpected
error is logged, alerted to the admin and answered with a short apology.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from typing import Any

from fplopt.alerts import send_admin_alert
from fplopt.bot.messages import split_message
from fplopt.bot.service import BotService, Reply
from fplopt.redact import redact

__all__ = ("COMMANDS", "build_application", "send_replies")

log = logging.getLogger(__name__)

# command -> (BotService method, takes the command's arguments)
COMMANDS: dict[str, tuple[str, bool]] = {
    "start": ("start", False),
    "help": ("help", False),
    "register": ("register", True),
    "plan": ("plan", False),
    "setft": ("setft", True),
    "alerts": ("alerts", True),
    "allow": ("allow", True),
    "block": ("block", True),
    "users": ("list_users", False),
}


async def send_replies(bot: Any, replies: Sequence[Reply]) -> None:
    """Send each reply (split to Telegram's length limit)."""
    for reply in replies:
        for piece in split_message(reply.text):
            await bot.send_message(reply.chat_id, piece)


def build_application(
    token: str,
    service: BotService,
    *,
    admin_token: str | None = None,
    admin_chat_id: str | None = None,
) -> Any:
    """The python-telegram-bot Application with a handler per `COMMANDS` entry."""
    from telegram import LinkPreviewOptions, Update
    from telegram.constants import ParseMode
    from telegram.ext import (
        Application,
        CommandHandler,
        ContextTypes,
        Defaults,
        MessageHandler,
        filters,
    )

    defaults = Defaults(
        parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True)
    )
    application = Application.builder().token(token).defaults(defaults).build()
    private = filters.ChatType.PRIVATE

    def handler(method: str, with_args: bool) -> Callable[..., Any]:
        call = getattr(service, method)

        async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            chat = update.effective_chat
            if chat is None:
                return
            args = list(context.args or [])
            if method == "plan":
                await context.bot.send_chat_action(chat.id, "typing")
            replies = await asyncio.to_thread(
                (lambda: call(chat.id, args)) if with_args else (lambda: call(chat.id))
            )
            await send_replies(context.bot, replies)

        return handle

    for command, (method, with_args) in COMMANDS.items():
        application.add_handler(CommandHandler(command, handler(method, with_args), private))

    async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat = update.effective_chat
        if chat is not None:
            await send_replies(context.bot, await asyncio.to_thread(service.unknown, chat.id))

    application.add_handler(MessageHandler(private & (filters.COMMAND | filters.TEXT), unknown))

    async def on_error(update: object | None, context: ContextTypes.DEFAULT_TYPE) -> None:
        error = context.error
        log.error("bot update failed", exc_info=error)
        summary = str(error).splitlines()[0] if error is not None and str(error) else ""
        await asyncio.to_thread(
            send_admin_alert,
            redact(f"fplopt bot: {type(error).__name__}: {summary}"),
            token=admin_token,
            chat_id=admin_chat_id,
        )
        chat = getattr(update, "effective_chat", None)
        if chat is not None:
            try:
                await context.bot.send_message(
                    chat.id, "Something went wrong. The admin has been told; try again later."
                )
            except Exception:
                log.exception("couldn't tell the user about the error")

    application.add_error_handler(on_error)
    return application
