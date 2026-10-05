import logging
import traceback

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

    secret_api_key = "SECRET"
    with pytest.raises(OddsApiError) as info:
        OddsClient(make_client(httpx.MockTransport(handler)), secret_api_key).epl_odds()
    rendered = "".join(traceback.format_exception(info.value))
    assert "SECRET" not in rendered
    assert info.value.__context__ is None
    assert "HTTP 401" in str(info.value)


def test_error_includes_api_error_code():
    def handler(request):
        return httpx.Response(401, json={"message": "quota", "error_code": "OUT_OF_USAGE_CREDITS"})

    with pytest.raises(OddsApiError, match="HTTP 401 OUT_OF_USAGE_CREDITS"):
        OddsClient(make_client(httpx.MockTransport(handler)), "KEY").epl_odds()


def test_warns_when_credits_low(caplog):
    def handler(request):
        return httpx.Response(200, json=[], headers={"x-requests-remaining": "12"})

    caplog.set_level(logging.INFO)
    OddsClient(make_client(httpx.MockTransport(handler)), "KEY").epl_odds()
    assert "credits low: 12" in caplog.text
    assert "KEY" not in caplog.text
