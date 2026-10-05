"""Masking secrets (API keys, bot tokens) out of log output. Importing installs the filter."""

from __future__ import annotations

import logging
import re

_SECRET_QUERY = re.compile(r"(?i)\b(apikey|api_key|token)=[^&\s\"']+")
_BOT_TOKEN = re.compile(r"/bot[^/]+/")


def redact(text: str) -> str:
    """Mask API keys in query strings and Telegram bot tokens in URL paths."""
    return _BOT_TOKEN.sub("/botREDACTED/", _SECRET_QUERY.sub(r"\1=REDACTED", text))


class _RedactSecrets(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = None
        return True


_FILTER = _RedactSecrets()


def install() -> None:
    """Attach the redaction filter to the httpx logger (idempotent)."""
    logger = logging.getLogger("httpx")
    if _FILTER not in logger.filters:
        logger.addFilter(_FILTER)


# httpx logs every request URL at INFO; keys and tokens must never reach the logs.
install()
