import logging
from datetime import UTC, datetime, timedelta

import pytest

from fplopt.ingest.health import StaleArchiveError, check_freshness
from fplopt.ingest.raw_store import RawStore

NOW = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)
MAX_AGE = timedelta(hours=36)


def archive_bootstrap(store, when):
    store.write("fpl", "bootstrap-static", b'{"events": []}', when)


def test_fresh_archive_passes_and_logs_the_age(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    store = RawStore(tmp_path)
    archive_bootstrap(store, NOW - timedelta(days=3))
    archive_bootstrap(store, NOW - timedelta(hours=9, minutes=30))
    assert check_freshness(store, NOW, MAX_AGE) == timedelta(hours=9, minutes=30)
    assert "9.5 h old" in caplog.text


def test_stale_archive_raises(tmp_path):
    store = RawStore(tmp_path)
    archive_bootstrap(store, NOW - timedelta(hours=37))
    with pytest.raises(StaleArchiveError, match=r"37\.0 h old") as info:
        check_freshness(store, NOW, MAX_AGE)
    assert "\n" not in str(info.value)  # the CLI alert keeps the first line only


def test_age_exactly_at_the_limit_passes(tmp_path):
    store = RawStore(tmp_path)
    archive_bootstrap(store, NOW - MAX_AGE)
    assert check_freshness(store, NOW, MAX_AGE) == MAX_AGE


def test_empty_archive_raises(tmp_path):
    with pytest.raises(StaleArchiveError, match="no bootstrap-static snapshot"):
        check_freshness(RawStore(tmp_path), NOW, MAX_AGE)
