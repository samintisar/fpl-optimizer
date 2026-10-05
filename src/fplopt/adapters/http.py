"""HTTP fetching with retries for flaky public APIs."""

from __future__ import annotations

import json
from collections.abc import Mapping

import httpx
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential
from tenacity.wait import wait_base

USER_AGENT = "fpl-optimizer/0.1 (+https://github.com/samintisar/fpl-optimizer)"


class InvalidPayload(Exception):
    """A 2xx response whose body is not JSON (e.g. FPL's 'game is being updated' page)."""


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError | InvalidPayload):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return False


def make_client(transport: httpx.BaseTransport | None = None) -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=httpx.Timeout(30.0),
        follow_redirects=True,
        transport=transport,
    )


def get_json_response(
    client: httpx.Client,
    url: str,
    params: Mapping[str, str] | None = None,
    *,
    attempts: int = 5,
    wait: wait_base | None = None,
) -> httpx.Response:
    """GET `url`, retrying transient failures. The returned body is guaranteed to be JSON."""
    for attempt in Retrying(
        stop=stop_after_attempt(attempts),
        wait=wait if wait is not None else wait_exponential(multiplier=2, max=60),
        retry=retry_if_exception(_retryable),
        reraise=True,
    ):
        with attempt:
            response = client.get(url, params=params)
            response.raise_for_status()
            try:
                json.loads(response.content)
            except ValueError as exc:
                raise InvalidPayload(f"non-JSON response from {url}") from exc
            return response
    raise AssertionError("unreachable")
