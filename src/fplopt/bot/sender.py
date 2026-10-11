"""Sending a user message with the Telegram Bot API over plain HTTP (the alert job has no
bot framework; `fplopt.bot.app` uses python-telegram-bot's own client)."""

from __future__ import annotations

from collections.abc import Callable

import httpx

__all__ = ("TelegramSendError", "telegram_sender")


class TelegramSendError(RuntimeError):
    """A message wasn't sent. The message never contains the request URL (it holds the bot
    token)."""


def telegram_sender(http: httpx.Client, token: str) -> Callable[[int, str], None]:
    """`send(chat_id, html)`: one sendMessage with HTML parse mode, link previews off;
    TelegramSendError on any failure."""

    def send(chat_id: int, text: str) -> None:
        try:
            response = http.post(
                f"https://api.telegram.org/bot{token.strip()}/sendMessage",
                json={
                    "chat_id": int(chat_id),
                    "text": text,
                    "parse_mode": "HTML",
                    "link_preview_options": {"is_disabled": True},
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            detail = f" (HTTP {status})" if status else ""
            raise TelegramSendError(
                f"sendMessage to chat {chat_id} failed: {type(exc).__name__}{detail}"
            ) from None

    return send
