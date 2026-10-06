from datetime import UTC, datetime

import pytest

from fplopt.seasons import (
    bootstrap_season,
    football_data_code,
    parse_season_label,
    season_label,
    season_start_year,
)


def test_season_label():
    assert season_label(2026) == "2026-27"
    assert season_label(2099) == "2099-00"


def test_parse_season_label():
    assert parse_season_label("2016-17") == 2016
    assert parse_season_label("2099-00") == 2099


@pytest.mark.parametrize("label", ["2016-18", "16-17", "2016/17", "2016-17 ", ""])
def test_parse_season_label_rejects_malformed(label):
    with pytest.raises(ValueError):
        parse_season_label(label)


def test_football_data_code():
    assert football_data_code(2016) == "1617"
    assert football_data_code(2026) == "2627"


def test_season_start_year_turns_over_in_july():
    assert season_start_year(datetime(2026, 6, 30, tzinfo=UTC)) == 2025
    assert season_start_year(datetime(2026, 7, 1, tzinfo=UTC)) == 2026
    assert season_start_year(datetime(2027, 1, 15, tzinfo=UTC)) == 2026


def test_bootstrap_season_uses_first_gameweek_deadline():
    bootstrap = {
        "events": [
            {"id": 2, "deadline_time": "2026-08-28T17:30:00Z"},
            {"id": 1, "deadline_time": "2026-08-21T17:30:00Z"},
        ]
    }
    assert bootstrap_season(bootstrap) == 2026
