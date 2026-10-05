import httpx
import pytest
from tenacity import wait_none

from fplopt.adapters.http import USER_AGENT, InvalidPayload, get_json_response, make_client

URL = "https://example.test/api"


def client_with(responses):
    calls = []

    def handler(request):
        calls.append(request)
        return responses[min(len(calls), len(responses)) - 1]

    return make_client(httpx.MockTransport(handler)), calls


def test_returns_json_response_and_sends_user_agent():
    client, calls = client_with([httpx.Response(200, json={"ok": True})])
    assert get_json_response(client, URL, wait=wait_none()).json() == {"ok": True}
    assert calls[0].headers["User-Agent"] == USER_AGENT


def test_retries_503_then_succeeds():
    client, calls = client_with(
        [httpx.Response(503, text="The game is being updated."), httpx.Response(200, json=[])]
    )
    assert get_json_response(client, URL, wait=wait_none()).json() == []
    assert len(calls) == 2


def test_retries_non_json_200_then_gives_up():
    client, calls = client_with([httpx.Response(200, text="<html>updating</html>")])
    with pytest.raises(InvalidPayload):
        get_json_response(client, URL, attempts=3, wait=wait_none())
    assert len(calls) == 3


def test_does_not_retry_404():
    client, calls = client_with([httpx.Response(404, json={"detail": "Not found."})])
    with pytest.raises(httpx.HTTPStatusError):
        get_json_response(client, URL, wait=wait_none())
    assert len(calls) == 1
