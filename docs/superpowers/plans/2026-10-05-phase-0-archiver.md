# Phase 0: Snapshot Archiver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Archive FPL `bootstrap-static` + `fixtures` and The Odds API EPL odds into append-only gzipped JSON under `raw/`, daily and ~2h before every deadline, on the Ubuntu box with Telegram failure alerts; plus a one-off backfill of 2026/27 `element-summary` (GW1–5).

**Architecture:** Thin adapters (`fplopt.adapters.*`) return raw response bytes; an append-only `RawStore` writes them once to `raw/<source>/<endpoint>/<UTC timestamp>.json.gz`; jobs (`fplopt.ingest.jobs`) compose adapters + store; a CLI (`fplopt …`) wraps every job with logging and a Telegram alert on failure. Scheduling uses systemd **user** timers: a daily run at 02:30 Europe/London and a 15-minute "tick" that snapshots once inside each 2h pre-deadline window (deadlines read from the latest archived bootstrap).

**Tech Stack:** Python 3.12, httpx (+ `MockTransport` in tests), tenacity, python-dotenv, pytest, ruff, systemd timers.

**Spec:** `docs/PLAN.md` §2–§4, §9 Phase 0. GitHub issues #1, #2, #4, #13.

**Branch:** do all work on `phase-0-archiver`; open a PR at the end.

---

## File structure

| File | Responsibility |
|---|---|
| `src/fplopt/ingest/raw_store.py` | Append-only gzipped-JSON store; timestamp formatting; listing/latest |
| `src/fplopt/adapters/http.py` | Shared httpx client (User-Agent, timeout) + GET with retries and JSON validation |
| `src/fplopt/adapters/fpl.py` | FPL endpoints → raw bytes |
| `src/fplopt/adapters/odds.py` | The Odds API EPL odds → raw bytes; never leaks the API key in errors |
| `src/fplopt/alerts.py` | Best-effort Telegram admin alert (plain HTTP) |
| `src/fplopt/ingest/schedule.py` | Pure logic: deadlines from bootstrap, "is a pre-deadline snapshot due?" |
| `src/fplopt/ingest/jobs.py` | `run_daily`, `run_tick`, `backfill_element_summaries` |
| `src/fplopt/settings.py` | Env-var settings |
| `src/fplopt/cli.py` | `fplopt` entry point: argparse, logging, failure alerts |
| `deploy/systemd/*.service`, `*.timer` | systemd user units |
| `deploy/README.md` | Server runbook |
| `tests/test_*.py` | One test module per source module (flat `tests/`, unique names) |

Raw layout produced:
```
raw/fpl/bootstrap-static/2026-10-05T013000Z.json.gz
raw/fpl/fixtures/2026-10-05T013001Z.json.gz
raw/odds/soccer_epl/2026-10-05T013002Z.json.gz
raw/fpl/element-summary/2026-10-05T140000Z/1.json.gz   # one directory per backfill run
```

---

### Task 0: Branch

- [ ] **Step 1: Create the branch**

```bash
git checkout -b phase-0-archiver
```

---

### Task 1: Append-only raw store

**Files:**
- Create: `src/fplopt/ingest/raw_store.py`
- Test: `tests/test_raw_store.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_raw_store.py`:
```python
import gzip
from datetime import UTC, datetime, timedelta, timezone

import pytest

from fplopt.ingest.raw_store import RawStore, format_ts, parse_ts

T0 = datetime(2026, 10, 5, 3, 0, 0, tzinfo=UTC)


def test_write_creates_gzipped_file_with_timestamp_path(tmp_path):
    store = RawStore(tmp_path)
    path = store.write("fpl", "bootstrap-static", b'{"a": 1}', T0)
    assert path == tmp_path / "fpl" / "bootstrap-static" / "2026-10-05T030000Z.json.gz"
    assert gzip.decompress(path.read_bytes()) == b'{"a": 1}'


def test_write_never_overwrites(tmp_path):
    store = RawStore(tmp_path)
    store.write("fpl", "fixtures", b"[]", T0)
    with pytest.raises(FileExistsError):
        store.write("fpl", "fixtures", b"[1]", T0)
    assert store.read_json(store.latest("fpl", "fixtures")) == []


def test_write_rejects_non_json(tmp_path):
    with pytest.raises(ValueError):
        RawStore(tmp_path).write("fpl", "fixtures", b"<html>The game is being updated.</html>", T0)
    assert not (tmp_path / "fpl").exists()


def test_write_with_name_groups_files_under_run_timestamp(tmp_path):
    path = RawStore(tmp_path).write("fpl", "element-summary", b"{}", T0, name="17")
    assert path == tmp_path / "fpl" / "element-summary" / "2026-10-05T030000Z" / "17.json.gz"


def test_times_and_latest(tmp_path):
    store = RawStore(tmp_path)
    assert store.times("fpl", "bootstrap-static") == []
    assert store.latest("fpl", "bootstrap-static") is None
    later = T0 + timedelta(hours=2)
    store.write("fpl", "bootstrap-static", b'{"n": 2}', later)
    store.write("fpl", "bootstrap-static", b'{"n": 1}', T0)
    assert store.times("fpl", "bootstrap-static") == [T0, later]
    assert store.read_json(store.latest("fpl", "bootstrap-static")) == {"n": 2}


def test_timestamps_are_normalised_to_utc():
    plus_one = timezone(timedelta(hours=1))
    assert format_ts(datetime(2026, 10, 5, 4, 0, 0, tzinfo=plus_one)) == "2026-10-05T030000Z"
    assert parse_ts("2026-10-05T030000Z") == T0


def test_naive_timestamp_rejected(tmp_path):
    with pytest.raises(ValueError):
        RawStore(tmp_path).write("fpl", "fixtures", b"[]", datetime(2026, 10, 5))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_raw_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fplopt.ingest.raw_store'`

- [ ] **Step 3: Implement**

`src/fplopt/ingest/raw_store.py`:
```python
"""Append-only store for raw API responses: gzipped JSON with a UTC timestamp in the path."""

from __future__ import annotations

import gzip
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TS_FORMAT = "%Y-%m-%dT%H%M%SZ"
SUFFIX = ".json.gz"


def format_ts(ts: datetime) -> str:
    if ts.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return ts.astimezone(UTC).strftime(TS_FORMAT)


def parse_ts(text: str) -> datetime:
    return datetime.strptime(text, TS_FORMAT).replace(tzinfo=UTC)


class RawStore:
    """Writes each response exactly once; existing files are never overwritten."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def write(
        self,
        source: str,
        endpoint: str,
        content: bytes,
        fetched_at: datetime,
        name: str | None = None,
    ) -> Path:
        json.loads(content)  # refuse to archive non-JSON, e.g. an HTML error page
        stamp = format_ts(fetched_at)
        directory = self.root / source / endpoint
        if name is None:
            path = directory / f"{stamp}{SUFFIX}"
        else:
            directory = directory / stamp
            path = directory / f"{name}{SUFFIX}"
        directory.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_bytes(gzip.compress(content, mtime=0))
        try:
            os.link(tmp, path)  # atomic, and raises FileExistsError instead of overwriting
        finally:
            tmp.unlink()
        return path

    def times(self, source: str, endpoint: str) -> list[datetime]:
        directory = self.root / source / endpoint
        if not directory.is_dir():
            return []
        return sorted(parse_ts(p.name.removesuffix(SUFFIX)) for p in directory.glob(f"*{SUFFIX}"))

    def latest(self, source: str, endpoint: str) -> Path | None:
        times = self.times(source, endpoint)
        if not times:
            return None
        return self.root / source / endpoint / f"{format_ts(times[-1])}{SUFFIX}"

    @staticmethod
    def read_json(path: Path) -> Any:
        return json.loads(gzip.decompress(Path(path).read_bytes()))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_raw_store.py -v`
Expected: 7 passed

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add src/fplopt/ingest/raw_store.py tests/test_raw_store.py
git commit -m "feat(ingest): append-only gzipped raw store"
```

---

### Task 2: HTTP fetching with retries

**Files:**
- Create: `src/fplopt/adapters/http.py`
- Test: `tests/test_http.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_http.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_http.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fplopt.adapters.http'`

- [ ] **Step 3: Implement**

`src/fplopt/adapters/http.py`:
```python
"""HTTP fetching with retries for flaky public APIs."""

from __future__ import annotations

import json
from collections.abc import Mapping

import httpx
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential
from tenacity.wait import wait_base

USER_AGENT = "fpl-optimizer/0.1 (+https://github.com/samintisar/fpl-optimizer)"


class InvalidPayload(Exception):
    """A 2xx response whose body is not JSON (e.g. FPL's 'game is being updated' page)."""


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError | InvalidPayload):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return False


def make_client(transport: httpx.BaseTransport | None = None) -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=httpx.Timeout(30.0),
        follow_redirects=True,
        transport=transport,
    )


def get_json_response(
    client: httpx.Client,
    url: str,
    params: Mapping[str, str] | None = None,
    *,
    attempts: int = 5,
    wait: wait_base | None = None,
) -> httpx.Response:
    """GET `url`, retrying transient failures. The returned body is guaranteed to be JSON."""
    for attempt in Retrying(
        stop=stop_after_attempt(attempts),
        wait=wait if wait is not None else wait_exponential(multiplier=2, max=60),
        retry=retry_if_exception(_retryable),
        reraise=True,
    ):
        with attempt:
            response = client.get(url, params=params)
            response.raise_for_status()
            try:
                json.loads(response.content)
            except ValueError as exc:
                raise InvalidPayload(f"non-JSON response from {url}") from exc
            return response
    raise AssertionError("unreachable")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_http.py -v`
Expected: 4 passed

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add src/fplopt/adapters/http.py tests/test_http.py
git commit -m "feat(adapters): HTTP GET with retries and JSON validation"
```

---

### Task 3: FPL adapter

**Files:**
- Create: `src/fplopt/adapters/fpl.py`
- Test: `tests/test_fpl_adapter.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_fpl_adapter.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_fpl_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fplopt.adapters.fpl'`

- [ ] **Step 3: Implement**

`src/fplopt/adapters/fpl.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_fpl_adapter.py -v`
Expected: 2 passed

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add src/fplopt/adapters/fpl.py tests/test_fpl_adapter.py
git commit -m "feat(adapters): FPL bootstrap, fixtures, element-summary"
```

---

### Task 4: Odds adapter

The Odds API v4: `GET /v4/sports/soccer_epl/odds/?apiKey=…&regions=uk&markets=h2h,totals&oddsFormat=decimal&dateFormat=iso`. Cost = markets × regions = 2 credits. Credit headers: `x-requests-remaining`, `x-requests-used`, `x-requests-last`. The key is a query parameter, so **errors must not echo the URL**.

**Files:**
- Create: `src/fplopt/adapters/odds.py`
- Test: `tests/test_odds_adapter.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_odds_adapter.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_odds_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fplopt.adapters.odds'`

- [ ] **Step 3: Implement**

`src/fplopt/adapters/odds.py`:
```python
"""The Odds API (v4): live EPL match odds. Methods return the raw response bytes."""

from __future__ import annotations

import logging

import httpx

from fplopt.adapters.http import get_json_response

log = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "soccer_epl"
MARKETS = ("h2h", "totals")
REGIONS = ("uk",)


class OddsApiError(Exception):
    """Odds request failed. The message never contains the request URL (it holds the key)."""


class OddsClient:
    def __init__(self, client: httpx.Client, api_key: str, base_url: str = BASE_URL) -> None:
        self._client = client
        self._api_key = api_key
        self._base = base_url.rstrip("/")

    def epl_odds(self) -> bytes:
        """Current EPL odds. Costs len(MARKETS) * len(REGIONS) credits per call."""
        try:
            response = get_json_response(
                self._client,
                f"{self._base}/sports/{SPORT}/odds/",
                params={
                    "apiKey": self._api_key,
                    "regions": ",".join(REGIONS),
                    "markets": ",".join(MARKETS),
                    "oddsFormat": "decimal",
                    "dateFormat": "iso",
                },
            )
        except httpx.HTTPStatusError as exc:
            raise OddsApiError(f"odds request failed: HTTP {exc.response.status_code}") from None
        log.info(
            "odds api credits remaining=%s used=%s last=%s",
            response.headers.get("x-requests-remaining"),
            response.headers.get("x-requests-used"),
            response.headers.get("x-requests-last"),
        )
        return response.content
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_odds_adapter.py -v`
Expected: 2 passed

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add src/fplopt/adapters/odds.py tests/test_odds_adapter.py
git commit -m "feat(adapters): The Odds API EPL odds without leaking the key"
```

---

### Task 5: Telegram admin alerts

**Files:**
- Create: `src/fplopt/alerts.py`
- Test: `tests/test_alerts.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_alerts.py`:
```python
import json

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_alerts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fplopt.alerts'`

- [ ] **Step 3: Implement**

`src/fplopt/alerts.py`:
```python
"""Admin alerts via the Telegram Bot API (plain HTTP; no bot framework needed)."""

from __future__ import annotations

import logging

import httpx

log = logging.getLogger(__name__)

MAX_LEN = 4000  # Telegram's limit is 4096 characters


def send_admin_alert(
    text: str,
    *,
    token: str | None,
    chat_id: str | None,
    client: httpx.Client | None = None,
) -> bool:
    """Best effort: never raises, so a failed alert can't hide the original error."""
    if not token or not chat_id:
        log.warning("telegram alert not configured; message was: %s", text)
        return False
    own_client = client is None
    http = client or httpx.Client(timeout=15.0)
    try:
        response = http.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text[:MAX_LEN]},
        )
        response.raise_for_status()
        return True
    except httpx.HTTPError as exc:
        # Only the type name: the exception message would contain the bot token in the URL.
        log.error("telegram alert failed: %s", type(exc).__name__)
        return False
    finally:
        if own_client:
            http.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_alerts.py -v`
Expected: 4 passed

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add src/fplopt/alerts.py tests/test_alerts.py
git commit -m "feat: best-effort Telegram admin alerts"
```

---

### Task 6: Pre-deadline scheduling logic

**Files:**
- Create: `src/fplopt/ingest/schedule.py`
- Test: `tests/test_schedule.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_schedule.py`:
```python
from datetime import UTC, datetime, timedelta

from fplopt.ingest.schedule import deadlines_from_bootstrap, pre_deadline_due

D = datetime(2026, 10, 10, 10, 0, tzinfo=UTC)


def test_parses_deadlines():
    bootstrap = {"events": [{"deadline_time": "2026-10-10T10:00:00Z"}]}
    assert deadlines_from_bootstrap(bootstrap) == [D]


def test_due_inside_window_without_snapshot():
    assert pre_deadline_due(D - timedelta(minutes=110), [D], []) == D


def test_not_due_before_window():
    assert pre_deadline_due(D - timedelta(hours=3), [D], []) is None


def test_not_due_after_deadline():
    assert pre_deadline_due(D + timedelta(minutes=1), [D], []) is None


def test_not_due_when_window_already_has_snapshot():
    taken = [D - timedelta(minutes=100)]
    assert pre_deadline_due(D - timedelta(minutes=50), [D], taken) is None


def test_snapshot_before_window_does_not_count():
    taken = [D - timedelta(hours=8)]
    assert pre_deadline_due(D - timedelta(minutes=50), [D], taken) == D
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_schedule.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fplopt.ingest.schedule'`

- [ ] **Step 3: Implement**

`src/fplopt/ingest/schedule.py`:
```python
"""When to take pre-deadline snapshots."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Any

PRE_DEADLINE_LEAD = timedelta(hours=2)


def deadlines_from_bootstrap(bootstrap: dict[str, Any]) -> list[datetime]:
    return [datetime.fromisoformat(event["deadline_time"]) for event in bootstrap["events"]]


def pre_deadline_due(
    now: datetime,
    deadlines: Iterable[datetime],
    taken: Iterable[datetime],
    lead: timedelta = PRE_DEADLINE_LEAD,
) -> datetime | None:
    """Return the deadline whose [deadline - lead, deadline) window contains `now`
    and has no snapshot yet; otherwise None."""
    taken = list(taken)
    for deadline in deadlines:
        start = deadline - lead
        if start <= now < deadline and not any(start <= t < deadline for t in taken):
            return deadline
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_schedule.py -v`
Expected: 6 passed

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add src/fplopt/ingest/schedule.py tests/test_schedule.py
git commit -m "feat(ingest): pre-deadline snapshot window logic"
```

---

### Task 7: Archiver jobs

**Files:**
- Create: `src/fplopt/ingest/jobs.py`
- Test: `tests/test_jobs.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_jobs.py`:
```python
import json
from datetime import UTC, datetime, timedelta

import pytest

from fplopt.ingest.jobs import backfill_element_summaries, run_daily, run_tick
from fplopt.ingest.raw_store import RawStore

DEADLINE = datetime(2026, 10, 10, 10, 0, tzinfo=UTC)
BOOTSTRAP = json.dumps(
    {
        "events": [{"deadline_time": "2026-10-10T10:00:00Z"}],
        "elements": [{"id": 1}, {"id": 2}],
    }
).encode()


class FakeFpl:
    def __init__(self, fail_ids=()):
        self.fail_ids = set(fail_ids)

    def bootstrap_static(self):
        return BOOTSTRAP

    def fixtures(self):
        return b"[]"

    def element_summary(self, element_id):
        if element_id in self.fail_ids:
            raise RuntimeError("boom")
        return json.dumps({"id": element_id, "history": []}).encode()


class FakeOdds:
    def epl_odds(self):
        return b"[]"


class Clock:
    """Advances one second per call so consecutive writes get distinct timestamps."""

    def __init__(self, start):
        self.t = start

    def __call__(self):
        self.t += timedelta(seconds=1)
        return self.t


def no_sleep(_seconds):
    pass


def test_daily_writes_bootstrap_fixtures_and_odds(tmp_path):
    store = RawStore(tmp_path)
    run_daily(store, FakeFpl(), FakeOdds(), Clock(DEADLINE - timedelta(hours=8)))
    assert len(store.times("fpl", "bootstrap-static")) == 1
    assert len(store.times("fpl", "fixtures")) == 1
    assert len(store.times("odds", "soccer_epl")) == 1


def test_daily_without_odds_key_still_archives_fpl(tmp_path):
    store = RawStore(tmp_path)
    run_daily(store, FakeFpl(), None, Clock(DEADLINE - timedelta(hours=8)))
    assert store.times("odds", "soccer_epl") == []
    assert len(store.times("fpl", "fixtures")) == 1


def test_tick_snapshots_an_empty_store(tmp_path):
    store = RawStore(tmp_path)
    assert run_tick(store, FakeFpl(), None, Clock(DEADLINE - timedelta(hours=8))) is True
    assert len(store.times("fpl", "bootstrap-static")) == 1


def test_tick_runs_once_per_pre_deadline_window(tmp_path):
    store = RawStore(tmp_path)
    fpl = FakeFpl()
    run_daily(store, fpl, None, Clock(DEADLINE - timedelta(hours=8)))
    assert run_tick(store, fpl, None, Clock(DEADLINE - timedelta(hours=3))) is False
    assert run_tick(store, fpl, None, Clock(DEADLINE - timedelta(minutes=110))) is True
    assert run_tick(store, fpl, None, Clock(DEADLINE - timedelta(minutes=95))) is False
    assert len(store.times("fpl", "bootstrap-static")) == 2


def test_backfill_archives_every_element_under_one_run(tmp_path):
    store = RawStore(tmp_path)
    count = backfill_element_summaries(store, FakeFpl(), Clock(DEADLINE), sleep=no_sleep)
    assert count == 2
    run_dirs = list((tmp_path / "fpl" / "element-summary").iterdir())
    assert len(run_dirs) == 1
    assert sorted(p.name for p in run_dirs[0].iterdir()) == ["1.json.gz", "2.json.gz"]


def test_backfill_continues_past_failures_then_raises(tmp_path):
    store = RawStore(tmp_path)
    with pytest.raises(RuntimeError, match="1 of 2"):
        backfill_element_summaries(store, FakeFpl(fail_ids={1}), Clock(DEADLINE), sleep=no_sleep)
    run_dir = next((tmp_path / "fpl" / "element-summary").iterdir())
    assert [p.name for p in run_dir.iterdir()] == ["2.json.gz"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_jobs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fplopt.ingest.jobs'`

- [ ] **Step 3: Implement**

`src/fplopt/ingest/jobs.py`:
```python
"""Archiver jobs: fetch from adapters and append to the raw store."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from fplopt.ingest.raw_store import RawStore
from fplopt.ingest.schedule import deadlines_from_bootstrap, pre_deadline_due

log = logging.getLogger(__name__)

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


class FplSource(Protocol):
    def bootstrap_static(self) -> bytes: ...

    def fixtures(self) -> bytes: ...

    def element_summary(self, element_id: int) -> bytes: ...


class OddsSource(Protocol):
    def epl_odds(self) -> bytes: ...


def snapshot_fpl(store: RawStore, fpl: FplSource, now: Clock = utc_now) -> list[Path]:
    return [
        store.write("fpl", "bootstrap-static", fpl.bootstrap_static(), now()),
        store.write("fpl", "fixtures", fpl.fixtures(), now()),
    ]


def snapshot_odds(store: RawStore, odds: OddsSource | None, now: Clock = utc_now) -> Path | None:
    if odds is None:
        log.warning("ODDS_API_KEY not set; skipping odds snapshot")
        return None
    return store.write("odds", "soccer_epl", odds.epl_odds(), now())


def run_daily(
    store: RawStore, fpl: FplSource, odds: OddsSource | None, now: Clock = utc_now
) -> None:
    snapshot_fpl(store, fpl, now)
    snapshot_odds(store, odds, now)


def run_tick(
    store: RawStore, fpl: FplSource, odds: OddsSource | None, now: Clock = utc_now
) -> bool:
    """Run every ~15 min: snapshot once inside each pre-deadline window. True if it ran."""
    latest = store.latest("fpl", "bootstrap-static")
    if latest is None:
        log.info("no bootstrap snapshot yet; taking one")
        run_daily(store, fpl, odds, now)
        return True
    deadlines = deadlines_from_bootstrap(store.read_json(latest))
    deadline = pre_deadline_due(now(), deadlines, store.times("fpl", "bootstrap-static"))
    if deadline is None:
        return False
    log.info("pre-deadline snapshot for deadline %s", deadline.isoformat())
    run_daily(store, fpl, odds, now)
    return True


def backfill_element_summaries(
    store: RawStore,
    fpl: FplSource,
    now: Clock = utc_now,
    pause_s: float = 0.25,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Archive element-summary for every current player under one run timestamp."""
    run_at = now()
    bootstrap = fpl.bootstrap_static()
    store.write("fpl", "bootstrap-static", bootstrap, run_at)
    ids = [element["id"] for element in json.loads(bootstrap)["elements"]]
    failed: list[int] = []
    for element_id in ids:
        try:
            content = fpl.element_summary(element_id)
            store.write("fpl", "element-summary", content, run_at, name=str(element_id))
        except Exception:
            log.exception("element-summary %s failed", element_id)
            failed.append(element_id)
        sleep(pause_s)
    if failed:
        raise RuntimeError(f"{len(failed)} of {len(ids)} element summaries failed: {failed[:20]}")
    log.info("archived %d element summaries", len(ids))
    return len(ids)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_jobs.py -v`
Expected: 6 passed

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add src/fplopt/ingest/jobs.py tests/test_jobs.py
git commit -m "feat(ingest): daily, pre-deadline tick and element-summary backfill jobs"
```

---

### Task 8: Settings and `fplopt` CLI

**Files:**
- Create: `src/fplopt/settings.py`, `src/fplopt/cli.py`
- Modify: `pyproject.toml` (add `[project.scripts]` after `[project.optional-dependencies]`)
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_cli.py`:
```python
import logging
from pathlib import Path

import pytest

from fplopt import cli
from fplopt.settings import Settings


def make_settings(tmp_path):
    return Settings(
        raw_dir=tmp_path,
        odds_api_key=None,
        telegram_bot_token="T",
        telegram_admin_chat_id="42",
    )


def test_successful_job_returns_zero(tmp_path, monkeypatch):
    ran = []
    monkeypatch.setitem(cli.JOBS, "snapshot daily", lambda store, fpl, odds: ran.append(odds))
    assert cli.main(["snapshot", "daily"], settings=make_settings(tmp_path)) == 0
    assert ran == [None]  # no odds key -> no odds client


def test_failed_job_alerts_and_returns_one(tmp_path, monkeypatch):
    def boom(store, fpl, odds):
        raise RuntimeError("fpl down")

    alerts = []
    monkeypatch.setitem(cli.JOBS, "snapshot tick", boom)
    monkeypatch.setattr(cli, "send_admin_alert", lambda text, **kw: alerts.append((text, kw)))
    assert cli.main(["snapshot", "tick"], settings=make_settings(tmp_path)) == 1
    text, kw = alerts[0]
    assert "snapshot tick failed" in text
    assert "fpl down" in text
    assert kw == {"token": "T", "chat_id": "42"}


def test_unknown_command_exits():
    with pytest.raises(SystemExit):
        cli.main(["snapshot", "weekly"])


def test_logging_hides_httpx_request_urls():
    cli.configure_logging()
    assert logging.getLogger("httpx").level == logging.WARNING


def test_settings_from_env():
    s = Settings.from_env({"FPLOPT_RAW_DIR": "/srv/raw", "ODDS_API_KEY": ""})
    assert s.raw_dir == Path("/srv/raw")
    assert s.odds_api_key is None
    assert Settings.from_env({}).raw_dir == Path("raw")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — `ImportError: cannot import name 'cli' from 'fplopt'`

- [ ] **Step 3: Implement settings**

`src/fplopt/settings.py`:
```python
"""Runtime settings from environment variables (the CLI loads .env first)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    raw_dir: Path
    odds_api_key: str | None
    telegram_bot_token: str | None
    telegram_admin_chat_id: str | None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        return cls(
            raw_dir=Path(env.get("FPLOPT_RAW_DIR") or "raw"),
            odds_api_key=env.get("ODDS_API_KEY") or None,
            telegram_bot_token=env.get("TELEGRAM_BOT_TOKEN") or None,
            telegram_admin_chat_id=env.get("TELEGRAM_ADMIN_CHAT_ID") or None,
        )
```

- [ ] **Step 4: Implement the CLI**

`src/fplopt/cli.py`:
```python
"""Command-line entry point: `fplopt snapshot daily|tick`, `fplopt backfill element-summary`."""

from __future__ import annotations

import argparse
import logging
import socket
import sys
from collections.abc import Callable, Sequence

from dotenv import find_dotenv, load_dotenv

from fplopt.adapters.fpl import FplClient
from fplopt.adapters.http import make_client
from fplopt.adapters.odds import OddsClient
from fplopt.alerts import send_admin_alert
from fplopt.ingest import jobs
from fplopt.ingest.raw_store import RawStore
from fplopt.settings import Settings

log = logging.getLogger("fplopt")

Job = Callable[[RawStore, FplClient, OddsClient | None], object]

JOBS: dict[str, Job] = {
    "snapshot daily": lambda store, fpl, odds: jobs.run_daily(store, fpl, odds),
    "snapshot tick": lambda store, fpl, odds: jobs.run_tick(store, fpl, odds),
    "backfill element-summary": lambda store, fpl, odds: jobs.backfill_element_summaries(
        store, fpl
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fplopt")
    groups = parser.add_subparsers(dest="group", required=True)
    snapshot = groups.add_parser("snapshot", help="archive API snapshots into raw/")
    snapshot.add_argument("command", choices=["daily", "tick"])
    backfill = groups.add_parser("backfill", help="one-off backfills into raw/")
    backfill.add_argument("command", choices=["element-summary"])
    return parser


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # httpx logs full request URLs at INFO, which would expose API keys and bot tokens.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def main(argv: Sequence[str] | None = None, settings: Settings | None = None) -> int:
    args = build_parser().parse_args(argv)
    job_name = f"{args.group} {args.command}"
    if settings is None:
        load_dotenv(find_dotenv(usecwd=True))
        settings = Settings.from_env()
    configure_logging()
    store = RawStore(settings.raw_dir)
    with make_client() as http:
        fpl = FplClient(http)
        odds = OddsClient(http, settings.odds_api_key) if settings.odds_api_key else None
        try:
            JOBS[job_name](store, fpl, odds)
        except Exception as exc:
            log.exception("job %r failed", job_name)
            send_admin_alert(
                f"fplopt {job_name} failed on {socket.gethostname()}: {type(exc).__name__}: {exc}",
                token=settings.telegram_bot_token,
                chat_id=settings.telegram_admin_chat_id,
            )
            return 1
    log.info("job %r done", job_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Register the console script**

In `pyproject.toml`, add after the `[project.optional-dependencies]` table:
```toml
[project.scripts]
fplopt = "fplopt.cli:main"
```

Then: `uv sync`

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest -v`
Expected: all tests pass (8 existing + 7 + 4 + 2 + 2 + 4 + 6 + 6 + 5 = 44 passed)

Run: `uv run fplopt --help`
Expected: usage text listing `snapshot` and `backfill`

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add src/fplopt/settings.py src/fplopt/cli.py tests/test_cli.py pyproject.toml uv.lock
git commit -m "feat: fplopt CLI with failure alerts"
```

---

### Task 9: Live smoke test against the real APIs (local, read-only)

Uses a throwaway raw directory; nothing is committed. Requires network. Skip the odds part if no `ODDS_API_KEY` is configured locally.

- [ ] **Step 1: Daily snapshot**

```bash
FPLOPT_RAW_DIR="$(mktemp -d)" && export FPLOPT_RAW_DIR && uv run fplopt snapshot daily && ls -R "$FPLOPT_RAW_DIR"
```
Expected: exit 0; `fpl/bootstrap-static/<ts>.json.gz` and `fpl/fixtures/<ts>.json.gz` exist (plus `odds/soccer_epl/<ts>.json.gz` if a key is set; otherwise a "skipping odds snapshot" warning).

- [ ] **Step 2: Content check**

```bash
uv run python -c "import os; from fplopt.ingest.raw_store import RawStore; s=RawStore(os.environ['FPLOPT_RAW_DIR']); b=s.read_json(s.latest('fpl','bootstrap-static')); print(len(b['elements']), 'players;', next(e['deadline_time'] for e in b['events'] if not e['finished']))"
```
Expected: roughly 700–850 players and the next deadline (GW6: `2026-10-10T10:00:00Z` until it passes).

- [ ] **Step 3: Tick is a no-op outside a pre-deadline window**

```bash
uv run fplopt snapshot tick && ls "$FPLOPT_RAW_DIR/fpl/bootstrap-static" | wc -l
```
Expected: exit 0 and still `1` file (unless run within 2h of a deadline).

- [ ] **Step 4: Clean up**

```bash
rm -rf "$FPLOPT_RAW_DIR" && unset FPLOPT_RAW_DIR
```

---

### Task 10: systemd units, runbook, plan update

**Files:**
- Create: `deploy/systemd/fplopt-daily.service`, `deploy/systemd/fplopt-daily.timer`, `deploy/systemd/fplopt-tick.service`, `deploy/systemd/fplopt-tick.timer`, `deploy/README.md`
- Modify: `docs/PLAN.md` (§2 Runtime row, §12 decision log), `CLAUDE.md` (Commands)

systemd timers replace cron: Ubuntu's cron has no per-job timezone, and the daily run must follow UK time across BST/GMT. User units avoid hard-coding a username.

- [ ] **Step 1: Write the units**

`deploy/systemd/fplopt-daily.service`:
```ini
[Unit]
Description=fpl-optimizer daily snapshot (bootstrap, fixtures, odds)
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=%h/fpl-optimizer
ExecStart=%h/fpl-optimizer/.venv/bin/fplopt snapshot daily
```

`deploy/systemd/fplopt-daily.timer`:
```ini
[Unit]
Description=fpl-optimizer daily snapshot after FPL price changes

[Timer]
OnCalendar=*-*-* 02:30:00 Europe/London
Persistent=true
RandomizedDelaySec=120

[Install]
WantedBy=timers.target
```

`deploy/systemd/fplopt-tick.service`:
```ini
[Unit]
Description=fpl-optimizer pre-deadline snapshot check
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=%h/fpl-optimizer
ExecStart=%h/fpl-optimizer/.venv/bin/fplopt snapshot tick
```

`deploy/systemd/fplopt-tick.timer`:
```ini
[Unit]
Description=fpl-optimizer pre-deadline check every 15 minutes

[Timer]
OnCalendar=*:0/15
AccuracySec=1min

[Install]
WantedBy=timers.target
```

- [ ] **Step 2: Write the runbook**

`deploy/README.md`:
````markdown
# Deploying the archiver (Ubuntu)

Runs as your normal user with systemd **user** timers. Raw data lands in `~/fpl-optimizer/raw/` (gitignored).

## One-time setup

1. Clone and install:
   ```bash
   git clone https://github.com/samintisar/fpl-optimizer.git ~/fpl-optimizer
   curl -LsSf https://astral.sh/uv/install.sh | sh
   cd ~/fpl-optimizer && uv sync --locked
   ```
2. Telegram alerts: create a bot with @BotFather, send it any message, then read your chat id from
   `https://api.telegram.org/bot<token>/getUpdates` (`message.chat.id`).
3. Create `~/fpl-optimizer/.env` (then `chmod 600 .env`):
   ```
   TELEGRAM_BOT_TOKEN=...
   TELEGRAM_ADMIN_CHAT_ID=...
   ODDS_API_KEY=...
   ```
4. Smoke test:
   ```bash
   .venv/bin/fplopt snapshot daily && ls raw/fpl/bootstrap-static
   ```
5. Alert test (should fail and send a Telegram message):
   ```bash
   FPLOPT_RAW_DIR=/proc/fplopt-alert-test .venv/bin/fplopt snapshot daily; echo "exit=$?"
   ```
6. Install and start the timers:
   ```bash
   mkdir -p ~/.config/systemd/user
   cp deploy/systemd/* ~/.config/systemd/user/
   systemctl --user daemon-reload
   systemctl --user enable --now fplopt-daily.timer fplopt-tick.timer
   sudo loginctl enable-linger "$USER"   # keep timers running when logged out
   ```
7. One-off backfill of this season's per-GW stats (issue #13, ~5 minutes):
   ```bash
   .venv/bin/fplopt backfill element-summary
   ```

## Operating

- Timers: `systemctl --user list-timers 'fplopt-*'`
- Logs: `journalctl --user -u fplopt-daily.service -u fplopt-tick.service --since today`
- Update: `cd ~/fpl-optimizer && git pull && uv sync --locked` (re-copy units if `deploy/systemd/` changed, then `systemctl --user daemon-reload`)
- Odds credits: each odds snapshot logs `credits remaining=…` (free tier: 500/month; ~80 used).
````

- [ ] **Step 3: Update the plan and CLAUDE.md**

In `docs/PLAN.md` §2, replace the Runtime row with:
```markdown
| Runtime | Ubuntu server (Tailscale). Bot as `systemd` service (long polling, no public URL). Pipeline via `systemd` user timers (timezone-aware schedules; see `deploy/README.md`) |
```

Append to the §12 decision log table:
```markdown
| 2026-10-05 | systemd user timers instead of cron; pre-deadline snapshot via a 15-min tick that reads deadlines from the latest archived bootstrap. | Ubuntu cron has no per-job timezone (daily run follows UK time); deadlines change, so they are read, not scheduled. |
```

In `CLAUDE.md` under `## Commands`, add:
```markdown
- `uv run fplopt snapshot daily|tick` / `uv run fplopt backfill element-summary` — archiver jobs (see `deploy/README.md`)
```

- [ ] **Step 4: Verify and commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest -q
git add deploy docs/PLAN.md CLAUDE.md
git commit -m "ops: systemd user timers and archiver runbook"
```
Expected: lint clean, 44 passed.

---

### Task 11: PR

- [ ] **Step 1: Push and open the PR**

```bash
git push -u origin phase-0-archiver
gh pr create --title "Phase 0: snapshot archiver" --body "$(cat <<'EOF'
Archiver for docs/PLAN.md Phase 0: append-only raw store, FPL + Odds API adapters, daily / pre-deadline tick / element-summary backfill jobs, `fplopt` CLI with Telegram failure alerts, systemd user timers + runbook.

Closes #1, closes #4. #2 and #13 close after server deployment (deploy/README.md steps 6–7).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 2: After merge and server deployment** (user runs `deploy/README.md` on the Ubuntu box): close #2 once `list-timers` shows both timers, and #13 once the backfill run completes.
