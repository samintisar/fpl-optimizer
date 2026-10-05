"""Admin alerts via the Telegram Bot API (plain HTTP; no bot framework needed)."""

from __future__ import annotations

import logging

import httpx

log = logging.getLogger(__name__)

MAX_LEN = 4000  # Telegram's limit is 4096 characters


def send_admin_alert(
    text: str,
    *,
    token: str | None,
    chat_id: str | None,
    client: httpx.Client | None = None,
) -> bool:
    """Best effort: never raises, so a failed alert can't hide the original error."""
    if not token or not chat_id:
        log.warning("telegram alert not configured; message was: %s", text)
        return False
    own_client = client is None
    http = client or httpx.Client(timeout=15.0)
    try:
        response = http.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text[:MAX_LEN]},
        )
        response.raise_for_status()
        return True
    except httpx.HTTPError as exc:
        # Only the type name: the exception message would contain the bot token in the URL.
        log.error("telegram alert failed: %s", type(exc).__name__)
        return False
    finally:
        if own_client:
            http.close()
