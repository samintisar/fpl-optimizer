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
            )
        except httpx.HTTPStatusError as exc:
            raise OddsApiError(f"odds request failed: HTTP {exc.response.status_code}") from None
        log.info(
            "odds api credits remaining=%s used=%s last=%s",
            response.headers.get("x-requests-remaining"),
            response.headers.get("x-requests-used"),
            response.headers.get("x-requests-last"),
        )
        return response.content
