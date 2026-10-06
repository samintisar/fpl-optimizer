"""Export config/scoring/<season>.json from an archived bootstrap-static (PLAN §3 Rules config)."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fplopt.ingest.raw_store import RawStore
from fplopt.seasons import bootstrap_season, season_label

log = logging.getLogger(__name__)

# Archives searched in order: our own snapshots first, then the fplcache mirror.
ARCHIVES = (
    ("fpl", "bootstrap-static", ".json.gz"),
    ("fplcache", "bootstrap-static", ".json.xz"),
)


def _season_of(payload: Any) -> int | None:
    """Season a payload describes, or None if it has no usable events (e.g. mid-reset)."""
    try:
        return bootstrap_season(payload)
    except (KeyError, TypeError, ValueError):
        return None


def find_bootstrap(store: RawStore, start_year: int) -> tuple[datetime, Path]:
    """Newest archived bootstrap describing season `start_year`: our own archive
    (fpl/bootstrap-static, .json.gz) first, then fplcache (.json.xz). Walks each archive
    newest-first, skipping snapshots taken on/after 1 Aug of start_year + 1, and stops once
    snapshots are older than 1 Jun of start_year. Raises LookupError if none matches."""
    too_new = datetime(start_year + 1, 8, 1, tzinfo=UTC)
    too_old = datetime(start_year, 6, 1, tzinfo=UTC)
    for source, endpoint, suffix in ARCHIVES:
        for taken_at, path in reversed(store.entries(source, endpoint, suffix)):
            if taken_at >= too_new:
                continue
            if taken_at < too_old:
                break
            season = _season_of(store.read_json(path))
            if season == start_year:
                return taken_at, path
            if season is None:
                log.warning("skipping %s: no events to date its season", path)
    raise LookupError(f"no archived bootstrap-static for season {season_label(start_year)}")


def _has_value(value: Any) -> bool:
    """True if an overrides entry sets anything: some non-null leaf inside it. Empty dicts,
    empty lists and nulls do not count; a leaf of 0 or False does (it overrides a rule)."""
    if isinstance(value, Mapping):
        return any(_has_value(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_value(item) for item in value)
    return value is not None


def rules_config(bootstrap: Mapping[str, Any], start_year: int) -> dict[str, Any]:
    """The season's rules from a bootstrap payload (without the `source` provenance)."""
    for key in ("game_config", "chips"):
        if key not in bootstrap:
            raise ValueError(f"bootstrap has no {key} (pre-2025/26 snapshot?)")
    game_config = bootstrap["game_config"]
    missing = [key for key in ("rules", "scoring", "settings") if key not in game_config]
    if missing:
        raise ValueError(f"bootstrap game_config lacks {', '.join(missing)}")
    return {
        "season": season_label(start_year),
        "scoring": game_config["scoring"],
        "rules": game_config["rules"],
        "settings": game_config["settings"],
        "chips": bootstrap["chips"],
        "element_types": [
            {key: value for key, value in element_type.items() if key != "element_count"}
            for element_type in bootstrap["element_types"]
        ],
        "event_overrides": {
            str(event["id"]): event["overrides"]
            for event in bootstrap["events"]
            if _has_value(event.get("overrides"))
        },
    }


def export_rules(store: RawStore, start_year: int, out_dir: Path) -> Path:
    """Write out_dir/<season>.json (sorted keys, indent 2, LF, trailing newline):
    {season, source: {path (relative to raw root, POSIX), snapshot_at},
     scoring, rules, settings (from game_config), chips,
     element_types (minus volatile 'element_count'),
     event_overrides: {gw: overrides} for events whose overrides have any truthy value}.
    Raises ValueError if the bootstrap has no game_config or chips."""
    taken_at, path = find_bootstrap(store, start_year)
    config = rules_config(store.read_json(path), start_year)
    config["source"] = {
        "path": path.relative_to(store.root).as_posix(),
        "snapshot_at": taken_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{season_label(start_year)}.json"
    text = json.dumps(config, sort_keys=True, indent=2) + "\n"
    out_path.write_bytes(text.encode("utf-8"))  # bytes: LF on every platform
    log.info("wrote %s from %s", out_path, config["source"]["path"])
    return out_path
