"""Admin alerts via the Telegram Bot API (plain HTTP; no bot framework needed)."""

from __future__ import annotations

import contextlib
import logging

import httpx

from fplopt.redact import redact  # importing also installs httpx log redaction

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
        log.warning("telegram alert not configured; message was: %s", redact(str(text)))
        return False
    http = client
    try:
        if http is None:
            http = httpx.Client(timeout=httpx.Timeout(10.0, connect=5.0))
        response = http.post(
            f"https://api.telegram.org/bot{token.strip()}/sendMessage",
            json={"chat_id": str(chat_id), "text": str(text)[:MAX_LEN]},
        )
        response.raise_for_status()
        return True
    except Exception as exc:
        # Never log the exception message: it can contain the bot token from the URL.
        status = getattr(getattr(exc, "response", None), "status_code", None)
        detail = f" (HTTP {status})" if status else ""
        log.error("telegram alert failed: %s%s", type(exc).__name__, detail)
        return False
    finally:
        if client is None and http is not None:
            with contextlib.suppress(Exception):
                http.close()
