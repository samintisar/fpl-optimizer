import json
import lzma
from datetime import UTC, datetime

import pytest

from fplopt.build.rules import export_rules, find_bootstrap
from fplopt.ingest.raw_store import RawStore

EMPTY_OVERRIDES = {"rules": {}, "scoring": {}, "element_types": [], "pick_multiplier": None}


def ts(year, month, day, hour=3):
    return datetime(year, month, day, hour, tzinfo=UTC)


def bootstrap(start_year, *, gw2_overrides=None, game_config=True, chips=True):
    payload = {
        "events": [
            {
                "id": 2,
                "deadline_time": f"{start_year}-08-22T17:30:00Z",
                "overrides": gw2_overrides or EMPTY_OVERRIDES,
            },
            {
                "id": 1,
                "deadline_time": f"{start_year}-08-15T17:30:00Z",
                "overrides": EMPTY_OVERRIDES,
            },
        ],
        "element_types": [
            {"id": 1, "singular_name_short": "GKP", "squad_select": 2, "element_count": 80},
            {"id": 2, "singular_name_short": "DEF", "squad_select": 5, "element_count": 260},
        ],
        "elements": [{"id": 1, "now_cost": 45}],
        "teams": [],
        "total_players": 11_000_000,
    }
    if game_config:
        payload["game_config"] = {
            "rules": {"squad_squadsize": 15, "max_extra_free_transfers": 4},
            "scoring": {"goals_scored": {"GKP": 10, "DEF": 6}, "defensive_contribution": 2},
            "settings": {"timezone": "UTC"},
            "status": {"leagues_updated": True},
        }
    if chips:
        payload["chips"] = [
            {"id": 1, "name": "wildcard", "start_event": 2, "stop_event": 19},
            {"id": 2, "name": "bboost", "start_event": 1, "stop_event": 19},
        ]
    return payload


def write_own(store, when, payload):
    return store.write("fpl", "bootstrap-static", json.dumps(payload).encode(), when)


def write_fplcache(store, when, payload):
    data = lzma.compress(json.dumps(payload, indent=2).encode())
    return write_fplcache_raw(store, when, data)


def write_fplcache_raw(store, when, data):
    return store.write_bytes("fplcache", "bootstrap-static", data, when, suffix=".json.xz")


def test_find_bootstrap_prefers_newest_snapshot_in_our_archive(tmp_path):
    store = RawStore(tmp_path)
    write_own(store, ts(2026, 8, 1), bootstrap(2026))
    newest = write_own(store, ts(2026, 10, 5), bootstrap(2026))
    write_fplcache(store, ts(2026, 10, 6), bootstrap(2026))  # newer, but ours wins

    assert find_bootstrap(store, 2026) == (ts(2026, 10, 5), newest)


def test_find_bootstrap_previous_season_falls_back_to_fplcache(tmp_path):
    store = RawStore(tmp_path)
    write_own(store, ts(2026, 10, 5), bootstrap(2026))
    write_fplcache(store, ts(2025, 9, 1), bootstrap(2025))
    last_old = write_fplcache(store, ts(2026, 7, 5), bootstrap(2025))
    write_fplcache(store, ts(2026, 7, 20), bootstrap(2026))  # after the July reset
    # On/after 1 Aug of the next year: never read (corrupt bytes would raise if it were).
    write_fplcache_raw(store, ts(2026, 8, 15), b"not xz")

    assert find_bootstrap(store, 2025) == (ts(2026, 7, 5), last_old)


def test_find_bootstrap_stops_before_june_of_start_year(tmp_path):
    store = RawStore(tmp_path)
    write_fplcache(store, ts(2026, 7, 20), bootstrap(2026))
    # Older than 1 Jun 2025: never read when looking for 2025/26.
    write_fplcache_raw(store, ts(2025, 5, 1), b"not xz")

    with pytest.raises(LookupError):
        find_bootstrap(store, 2025)


def test_find_bootstrap_skips_snapshots_without_events(tmp_path):
    store = RawStore(tmp_path)
    good = write_own(store, ts(2026, 9, 1), bootstrap(2026))
    write_own(store, ts(2026, 9, 2), {"events": []})

    assert find_bootstrap(store, 2026) == (ts(2026, 9, 1), good)


def test_find_bootstrap_raises_lookup_error_when_nothing_matches(tmp_path):
    store = RawStore(tmp_path)
    with pytest.raises(LookupError):
        find_bootstrap(store, 2026)

    write_own(store, ts(2026, 10, 5), bootstrap(2026))
    with pytest.raises(LookupError):
        find_bootstrap(store, 2024)


def test_export_rules_content(tmp_path):
    store = RawStore(tmp_path / "raw")
    afcon = {**EMPTY_OVERRIDES, "rules": {"max_extra_free_transfers": 5}}
    write_own(store, ts(2026, 10, 5), bootstrap(2026, gw2_overrides=afcon))
    out_dir = tmp_path / "config" / "scoring"  # does not exist yet

    path = export_rules(store, 2026, out_dir)

    assert path == out_dir / "2026-27.json"
    raw = path.read_bytes()
    assert raw.endswith(b"}\n")
    assert b"\r\n" not in raw
    config = json.loads(raw)
    assert set(config) == {
        "season",
        "source",
        "scoring",
        "rules",
        "settings",
        "chips",
        "element_types",
        "event_overrides",
    }
    assert config["season"] == "2026-27"
    assert config["source"] == {
        "path": "fpl/bootstrap-static/2026-10-05T030000Z.json.gz",
        "snapshot_at": "2026-10-05T03:00:00Z",
    }
    assert config["rules"] == {"squad_squadsize": 15, "max_extra_free_transfers": 4}
    assert config["scoring"]["defensive_contribution"] == 2
    assert config["settings"] == {"timezone": "UTC"}
    assert [chip["name"] for chip in config["chips"]] == ["wildcard", "bboost"]
    assert config["element_types"] == [
        {"id": 1, "singular_name_short": "GKP", "squad_select": 2},
        {"id": 2, "singular_name_short": "DEF", "squad_select": 5},
    ]
    assert config["event_overrides"] == {"2": afcon}
    assert raw == json.dumps(config, sort_keys=True, indent=2).encode() + b"\n"


def test_export_rules_from_fplcache_snapshot(tmp_path):
    store = RawStore(tmp_path / "raw")
    write_fplcache(store, ts(2026, 5, 20, 16), bootstrap(2025))

    path = export_rules(store, 2025, tmp_path / "out")

    config = json.loads(path.read_bytes())
    assert path.name == "2025-26.json"
    assert config["source"] == {
        "path": "fplcache/bootstrap-static/2026-05-20T160000Z.json.xz",
        "snapshot_at": "2026-05-20T16:00:00Z",
    }
    assert config["event_overrides"] == {}


def test_export_rules_is_deterministic(tmp_path):
    store = RawStore(tmp_path / "raw")
    write_own(store, ts(2026, 10, 5), bootstrap(2026))
    out_dir = tmp_path / "out"

    first = export_rules(store, 2026, out_dir).read_bytes()
    second = export_rules(store, 2026, out_dir).read_bytes()

    assert first == second


@pytest.mark.parametrize("missing", ["game_config", "chips"])
def test_export_rules_requires_game_config_and_chips(tmp_path, missing):
    store = RawStore(tmp_path / "raw")
    write_own(store, ts(2022, 10, 5), bootstrap(2022, **{missing: False}))
    out_dir = tmp_path / "out"

    with pytest.raises(ValueError, match=missing):
        export_rules(store, 2022, out_dir)
    assert not (out_dir / "2022-23.json").exists()


def test_export_rules_keeps_overrides_that_set_a_falsy_value(tmp_path):
    store = RawStore(tmp_path / "raw")
    no_transfers = {**EMPTY_OVERRIDES, "rules": {"max_extra_free_transfers": 0}}
    write_own(store, ts(2026, 10, 5), bootstrap(2026, gw2_overrides=no_transfers))

    config = json.loads(export_rules(store, 2026, tmp_path / "out").read_bytes())

    assert config["event_overrides"] == {"2": no_transfers}
