"""Masking secrets (API keys, bot tokens, heartbeat ping URLs) out of log output. Importing
installs the filter."""

from __future__ import annotations

import logging
import re

_SECRET_QUERY = re.compile(r"(?i)\b(apikey|api_key|token)=[^&\s\"']+")
_BOT_TOKEN = re.compile(r"/bot[^/]+/")
# healthchecks.io ping URLs are https://hc-ping.com/<uuid>: the UUID is the secret.
_UUID = re.compile(r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")

_SECRETS: set[str] = set()


def register_secret(value: str) -> None:
    """Mask this exact string from now on, e.g. a ping URL whose path is a secret but not a
    UUID (healthchecks.io ping-key URLs, self-hosted instances)."""
    value = value.strip()
    if len(value) >= 8:  # never mask short, common strings
        _SECRETS.add(value)


def redact(text: str) -> str:
    """Mask registered secrets, UUIDs, API keys in query strings and Telegram bot tokens in
    URL paths."""
    for secret in sorted(_SECRETS, key=len, reverse=True):
        text = text.replace(secret, "REDACTED")
    text = _UUID.sub("REDACTED", text)
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
