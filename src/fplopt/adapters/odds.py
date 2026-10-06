"""The Odds API (v4): live EPL match odds. Methods return the raw response bytes."""

from __future__ import annotations

import logging

import httpx

from fplopt.adapters.http import get_json_response

log = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "soccer_epl"
MARKETS = ("h2h", "totals")
REGIONS = ("uk",)
LOW_CREDIT_WARNING = 50


class OddsApiError(Exception):
    """Odds request failed. The message never contains the request URL (it holds the key)."""


class OddsClient:
    def __init__(self, client: httpx.Client, api_key: str, base_url: str = BASE_URL) -> None:
        self._client = client
        self._api_key = api_key
        self._base = base_url.rstrip("/")

    def epl_odds(self) -> bytes:
        """Current EPL odds. Costs len(MARKETS) * len(REGIONS) credits per call."""
        try:
            response = get_json_response(
                self._client,
                f"{self._base}/sports/{SPORT}/odds/",
                params={
                    "apiKey": self._api_key,
                    "regions": ",".join(REGIONS),
                    "markets": ",".join(MARKETS),
                    "oddsFormat": "decimal",
                    "dateFormat": "iso",
                },
                attempts=3,
            )
        except httpx.HTTPStatusError as exc:
            failure = f"HTTP {exc.response.status_code} {_error_code(exc.response)}".rstrip()
        else:
            _log_credits(response)
            return response.content
        # Raised outside the except block so no exception context (whose URL holds the key)
        # is attached to the error.
        raise OddsApiError(f"odds request failed: {failure}")


def _error_code(response: httpx.Response) -> str:
    """The API's error_code (e.g. OUT_OF_USAGE_CREDITS), never echoing anything else."""
    try:
        code = response.json().get("error_code", "")
    except (ValueError, AttributeError):
        return ""
    return code if isinstance(code, str) and code.isidentifier() else ""


def _log_credits(response: httpx.Response) -> None:
    remaining = response.headers.get("x-requests-remaining")
    log.info(
        "odds api credits remaining=%s used=%s last=%s",
        remaining,
        response.headers.get("x-requests-used"),
        response.headers.get("x-requests-last"),
    )
    try:
        low = float(remaining) < LOW_CREDIT_WARNING
    except (TypeError, ValueError):
        low = False
    if low:
        log.warning("odds api credits low: %s remaining this month", remaining)
