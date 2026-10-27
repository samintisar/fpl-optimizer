import httpx
import pytest

from fplopt.adapters.espn import BASE_URL, LEAGUES, EspnClient, require_key
from fplopt.adapters.http import InvalidPayload, make_client


def recording_client(body: bytes):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, content=body)

    return make_client(httpx.MockTransport(handler)), seen


def test_scoreboard_asks_for_the_period_and_returns_bytes_unchanged():
    body = b'{"events": []}'
    client, seen = recording_client(body)
    assert EspnClient(client).scoreboard("eng.league_cup", "2016") == body
    assert seen == [f"{BASE_URL}/eng.league_cup/scoreboard?dates=2016&limit=1000"]


def test_summary_asks_for_the_event():
    body = b'{"header": {}, "rosters": []}'
    client, seen = recording_client(body)
    assert EspnClient(client).summary("eng.1", "401878767") == body
    assert seen == [f"{BASE_URL}/eng.1/summary?event=401878767"]


def test_require_key_rejects_json_without_the_key():
    require_key("events")(b'{"events": []}')
    with pytest.raises(InvalidPayload):
        require_key("events")(b'{"code": 404}')
    with pytest.raises(InvalidPayload):
        require_key("events")(b"<html>")


def test_leagues_are_the_eight_competitions():
    assert LEAGUES == (
        "eng.1",
        "eng.fa",
        "eng.league_cup",
        "eng.charity",
        "uefa.champions",
        "uefa.europa",
        "uefa.europa.conf",
        "uefa.super_cup",
    )


def test_unknown_league_is_refused():
    client, _ = recording_client(b"{}")
    with pytest.raises(ValueError, match="unknown ESPN league"):
        EspnClient(client).scoreboard("esp.1", "2016")


def test_requests_are_tried_at_most_three_times(monkeypatch):
    seen = []

    def fake_get_response(client, url, params, **kwargs):
        seen.append(kwargs["attempts"])
        return httpx.Response(200, content=b"{}")

    monkeypatch.setattr("fplopt.adapters.espn.get_response", fake_get_response)
    client, _ = recording_client(b"{}")
    EspnClient(client).scoreboard("eng.1", "2016")
    EspnClient(client).summary("eng.1", "1")
    assert seen == [3, 3]
