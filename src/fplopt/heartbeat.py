"""Best-effort heartbeat pings to an external dead-man's switch such as healthchecks.io,
which alerts when the pings stop (including when the whole server is down)."""

from __future__ import annotations

import contextlib
import logging

import httpx

from fplopt.redact import register_secret  # importing also installs httpx log redaction

log = logging.getLogger(__name__)

TIMEOUT = httpx.Timeout(10.0, connect=5.0)


def send_heartbeat(url: str, *, client: httpx.Client | None = None) -> bool:
    """GET `url`. Never raises, and never logs the URL: its path is usually a secret."""
    register_secret(url)
    http = client
    try:
        register_secret(str(httpx.URL(url)))  # the form httpx logs, if it normalises the URL
        if http is None:
            http = httpx.Client(timeout=TIMEOUT)
        http.get(url).raise_for_status()
        log.info("heartbeat ping sent")
        return True
    except Exception as exc:
        # Never log the exception message: it can contain the URL.
        status = getattr(getattr(exc, "response", None), "status_code", None)
        detail = f" (HTTP {status})" if status else ""
        log.warning("heartbeat ping failed: %s%s", type(exc).__name__, detail)
        return False
    finally:
        if client is None and http is not None:
            with contextlib.suppress(Exception):
                http.close()
