import httpx
import pytest

from fplopt.adapters.http import make_client
from fplopt.adapters.odds import OddsApiError, OddsClient


def test_requests_epl_h2h_and_totals_in_decimal():
    seen = []

    def handler(request):
        seen.append(request.url)
        return httpx.Response(200, json=[], headers={"x-requests-remaining": "498"})

    body = OddsClient(make_client(httpx.MockTransport(handler)), "KEY").epl_odds()
    assert body == b"[]"
    assert seen[0].path == "/v4/sports/soccer_epl/odds/"
    assert dict(seen[0].params) == {
        "apiKey": "KEY",
        "regions": "uk",
        "markets": "h2h,totals",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }


def test_errors_do_not_leak_api_key():
    def handler(request):
        return httpx.Response(401, json={"message": "invalid key"})

    with pytest.raises(OddsApiError) as info:
        OddsClient(make_client(httpx.MockTransport(handler)), "SECRET").epl_odds()
    assert "SECRET" not in str(info.value)
    assert info.value.__cause__ is None
