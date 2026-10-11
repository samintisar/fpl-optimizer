"""Fantasy Premier League public API. Methods return the raw response bytes."""

from __future__ import annotations

import httpx

from fplopt.adapters.http import get_json_response

BASE_URL = "https://fantasy.premierleague.com/api"


class FplClient:
    def __init__(self, client: httpx.Client, base_url: str = BASE_URL) -> None:
        self._client = client
        self._base = base_url.rstrip("/")

    def _get(self, path: str) -> bytes:
        return get_json_response(self._client, f"{self._base}/{path}").content

    def bootstrap_static(self) -> bytes:
        return self._get("bootstrap-static/")

    def fixtures(self) -> bytes:
        return self._get("fixtures/")

    def element_summary(self, element_id: int) -> bytes:
        return self._get(f"element-summary/{element_id}/")

    def event_live(self, gw: int) -> bytes:
        return self._get(f"event/{gw}/live/")

    # A manager's team (public, no login; `fplopt.entry`).
    def entry_history(self, entry_id: int) -> bytes:
        return self._get(f"entry/{entry_id}/history/")

    def entry_picks(self, entry_id: int, gw: int) -> bytes:
        return self._get(f"entry/{entry_id}/event/{gw}/picks/")

    def entry_transfers(self, entry_id: int) -> bytes:
        return self._get(f"entry/{entry_id}/transfers/")
