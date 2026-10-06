import json
import logging

import httpx

from fplopt.alerts import send_admin_alert


def capturing_client(status=200):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(status, json={"ok": status == 200})

    return httpx.Client(transport=httpx.MockTransport(handler)), seen


def test_posts_message_to_chat():
    client, seen = capturing_client()
    assert send_admin_alert("boom", token="T", chat_id="42", client=client) is True
    assert seen[0].url.path == "/botT/sendMessage"
    assert json.loads(seen[0].content) == {"chat_id": "42", "text": "boom"}


def test_unconfigured_is_noop():
    assert send_admin_alert("boom", token=None, chat_id="42") is False


def test_failure_returns_false_without_raising():
    client, _ = capturing_client(status=500)
    assert send_admin_alert("boom", token="T", chat_id="42", client=client) is False


def test_long_messages_are_truncated():
    client, seen = capturing_client()
    send_admin_alert("x" * 5000, token="T", chat_id="42", client=client)
    assert len(json.loads(seen[0].content)["text"]) == 4000


def test_transport_error_returns_false():
    def handler(request):
        raise httpx.ConnectError("down", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert send_admin_alert("boom", token="T", chat_id="42", client=client) is False


def test_bad_token_does_not_raise():
    client, _ = capturing_client(status=401)
    assert send_admin_alert("boom", token="1:A\nB C", chat_id="42", client=client) is False


def test_token_not_logged(caplog):
    caplog.set_level(logging.DEBUG)
    client, _ = capturing_client(status=401)
    send_admin_alert("boom", token="123:SECRET", chat_id="42", client=client)
    assert "SECRET" not in caplog.text
    assert "HTTP 401" in caplog.text


def test_unconfigured_log_redacts_text(caplog):
    caplog.set_level(logging.WARNING)
    send_admin_alert("failed GET https://x.test/?apiKey=K3Y", token=None, chat_id="42")
    assert "K3Y" not in caplog.text
