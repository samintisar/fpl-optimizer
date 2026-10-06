"""football-data.co.uk: EPL results + pre-match odds, one CSV per season. Returns raw bytes."""

from __future__ import annotations

import httpx

from fplopt.adapters.http import InvalidPayload, get_response
from fplopt.seasons import football_data_code

BASE_URL = "https://www.football-data.co.uk/mmz4281"


def require_csv(content: bytes) -> None:
    """football-data CSVs start with the `Div` column (after an optional UTF-8 BOM).

    Seen 2016/17-2026/27: `Div,Date,HomeTeam,...` (to 2018/19), `Div,Date,Time,...` (from
    2019/20), and a UTF-8 BOM before `Div` in some seasons (2021/22, 2024/25 onwards).
    """
    head = content[:64].decode("utf-8-sig", errors="replace").lstrip()
    if not head.startswith("Div,"):
        raise InvalidPayload("response is not a football-data CSV")


class FootballDataClient:
    def __init__(self, client: httpx.Client, base_url: str = BASE_URL) -> None:
        self._client = client
        self._base = base_url.rstrip("/")

    def epl_season(self, start_year: int) -> bytes:
        """The season's E0 (Premier League) CSV, e.g. 2016 -> mmz4281/1617/E0.csv."""
        url = f"{self._base}/{football_data_code(start_year)}/E0.csv"
        return get_response(self._client, url, validate=require_csv, attempts=3).content
