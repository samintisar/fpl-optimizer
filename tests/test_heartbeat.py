import logging

import httpx
import pytest

from fplopt.heartbeat import send_heartbeat
from fplopt.redact import redact

UUID = "0b4c1f7e-9a2d-4e4b-8f3a-5d6c7b8a9e01"
URL = f"https://hc-ping.com/{UUID}"


def client_for(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_pings_the_url_with_get():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, text="OK")

    assert send_heartbeat(URL, client=client_for(handler)) is True
    assert [(r.method, str(r.url)) for r in seen] == [("GET", URL)]


@pytest.mark.parametrize("status", [404, 500])
def test_http_error_returns_false_without_raising(status):
    assert send_heartbeat(URL, client=client_for(lambda r: httpx.Response(status))) is False


def test_transport_error_returns_false_without_raising():
    def handler(request):
        raise httpx.ConnectError(f"cannot reach {request.url}", request=request)

    assert send_heartbeat(URL, client=client_for(handler)) is False


@pytest.mark.parametrize("status", [200, 404])
def test_url_is_never_logged(caplog, status):
    # httpx logs every request URL at INFO; the CLI raises that logger to WARNING, but the
    # URL must stay out of the logs even at DEBUG.
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="httpx")
    send_heartbeat(URL, client=client_for(lambda r: httpx.Response(status)))
    assert any(r.name == "httpx" for r in caplog.records)  # httpx did log the request
    assert UUID not in caplog.text
    assert "hc-ping.com" not in caplog.text or "REDACTED" in caplog.text


def test_transport_error_log_names_the_failure_type_only(caplog):
    def handler(request):
        raise httpx.ConnectError(f"cannot reach {request.url}", request=request)

    send_heartbeat(URL, client=client_for(handler))
    assert "ConnectError" in caplog.text
    assert UUID not in caplog.text


def test_non_uuid_ping_urls_are_redacted_once_used():
    # healthchecks.io also offers <ping-key>/<slug> URLs, and self-hosted instances exist.
    url = "https://hc.example.test/ping/AbCdEfGhIjKlMnOpQrStUv/fplopt-archiver"
    send_heartbeat(url, client=client_for(lambda r: httpx.Response(200)))
    assert "AbCdEfGhIjKlMnOpQrStUv" not in redact(f"GET {url} failed")


def test_redact_masks_uuids():
    assert UUID not in redact(f"HTTP Request: GET {URL} 200 OK")
