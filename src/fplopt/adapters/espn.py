"""ESPN's public site API (unofficial, no key): soccer scoreboards and match summaries.
Returns raw bytes. Formats checked 2026-10-27
(docs/superpowers/plans/2026-10-27-espn-cups-1-data.md)."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx

from fplopt.adapters.http import InvalidPayload, get_response

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer"
# The PL, domestic cups, the Community Shield and UEFA club competitions.
LEAGUES = (
    "eng.1",
    "eng.fa",
    "eng.league_cup",
    "eng.charity",
    "uefa.champions",
    "uefa.europa",
    "uefa.europa.conf",
    "uefa.super_cup",
)
# Red-card bans carry across these (PLAN §6.3).
DOMESTIC = ("eng.1", "eng.fa", "eng.league_cup")
SCOREBOARD_LIMIT = "1000"
ATTEMPTS = 3  # per request: a hanging ESPN costs at most three timeouts


def require_key(key: str) -> Callable[[bytes], None]:
    """A validator: the body is a JSON object with `key` (ESPN answers some errors with 200)."""

    def validate(content: bytes) -> None:
        try:
            payload = json.loads(content)
        except ValueError as exc:
            raise InvalidPayload("response is not JSON") from exc
        if not isinstance(payload, dict) or key not in payload:
            raise InvalidPayload(f"response has no {key!r}")

    return validate


class EspnClient:
    def __init__(self, client: httpx.Client, base_url: str = BASE_URL) -> None:
        self._client = client
        self._base = base_url.rstrip("/")

    def _url(self, league: str, endpoint: str) -> str:
        if league not in LEAGUES:
            raise ValueError(f"unknown ESPN league {league!r}")
        return f"{self._base}/{league}/{endpoint}"

    def scoreboard(self, league: str, period: str) -> bytes:
        """Every event of a calendar year (`period` 'YYYY') or month ('YYYYMM')."""
        url = self._url(league, "scoreboard")
        params = {"dates": period, "limit": SCOREBOARD_LIMIT}
        return get_response(
            self._client, url, params, validate=require_key("events"), attempts=ATTEMPTS
        ).content

    def summary(self, league: str, event_id: str) -> bytes:
        """One match: header, rosters (lineups, substitutes) and key events (subs, cards)."""
        url = self._url(league, "summary")
        params = {"event": event_id}
        return get_response(
            self._client, url, params, validate=require_key("header"), attempts=ATTEMPTS
        ).content
