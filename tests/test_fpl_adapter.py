import httpx

from fplopt.adapters.fpl import FplClient
from fplopt.adapters.http import make_client


def recording_client(body=b"{}"):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, content=body)

    return make_client(httpx.MockTransport(handler)), seen


def test_endpoints_hit_expected_urls():
    client, seen = recording_client()
    fpl = FplClient(client)
    fpl.bootstrap_static()
    fpl.fixtures()
    fpl.element_summary(17)
    assert seen == [
        "https://fantasy.premierleague.com/api/bootstrap-static/",
        "https://fantasy.premierleague.com/api/fixtures/",
        "https://fantasy.premierleague.com/api/element-summary/17/",
    ]


def test_returns_raw_bytes_unchanged():
    client, _ = recording_client(b'{"events": []}')
    assert FplClient(client).bootstrap_static() == b'{"events": []}'
