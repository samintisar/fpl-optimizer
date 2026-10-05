import logging

import httpx
import pytest
from tenacity import RetryCallState, Retrying, wait_none

from fplopt.adapters.http import (
    USER_AGENT,
    InvalidPayload,
    default_wait,
    get_json_response,
    make_client,
    redact,
)

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


def test_retries_429_and_transport_errors():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429)
        if len(calls) == 2:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json={})

    client = make_client(httpx.MockTransport(handler))
    assert get_json_response(client, URL, wait=wait_none()).json() == {}
    assert len(calls) == 3


def test_passes_query_params():
    client, calls = client_with([httpx.Response(200, json=[])])
    get_json_response(client, URL, params={"a": "1"}, wait=wait_none())
    assert calls[0].url.params["a"] == "1"


def test_gives_up_after_repeated_5xx():
    client, calls = client_with([httpx.Response(500)])
    with pytest.raises(httpx.HTTPStatusError):
        get_json_response(client, URL, attempts=3, wait=wait_none())
    assert len(calls) == 3


def retry_state_for(response):
    exc = httpx.HTTPStatusError("x", request=httpx.Request("GET", URL), response=response)
    state = RetryCallState(retry_object=Retrying(), fn=None, args=(), kwargs={})
    state.set_exception((type(exc), exc, None))
    return state


def test_default_wait_honours_retry_after_on_429():
    assert default_wait(retry_state_for(httpx.Response(429, headers={"Retry-After": "30"}))) == 30
    assert default_wait(retry_state_for(httpx.Response(429, headers={"Retry-After": "999"}))) == 120
    assert 0 < default_wait(retry_state_for(httpx.Response(503))) <= 60


def test_redact_masks_query_keys_and_bot_tokens():
    url = "https://x.test/v4/odds/?apiKey=abc123&regions=uk"
    assert redact(url) == "https://x.test/v4/odds/?apiKey=REDACTED&regions=uk"
    assert redact("https://api.telegram.org/bot123:AA-bb_c/sendMessage") == (
        "https://api.telegram.org/botREDACTED/sendMessage"
    )


def test_httpx_logger_output_is_redacted(caplog):
    caplog.set_level(logging.INFO)
    logging.getLogger("httpx").info("HTTP Request: GET %s", "https://x.test/?apiKey=SECRET")
    assert "SECRET" not in caplog.text
    assert "apiKey=REDACTED" in caplog.text
