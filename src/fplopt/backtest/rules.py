"""Game rules for the backtester (PLAN §3 Rules config, §5 Scoring the backtest).

A `Rules` object is read from two files in `config/scoring/`: `<label>.json`, generated from
the API by `fplopt rules export` (never hand-edited), and `<label>.supplement.json`, the
hand-maintained rest the API lacks (scoring thresholds, defcon thresholds, hit cost, the
WC/FH free-transfer effect, the Free Hit consecutive rule, FT top-ups). Every key is
required: a missing key raises instead of falling back to a guess, because a silently wrong
rule skews every backtest scored with it.

Which rules score which season (`backtest_rules`) follows PLAN §5: develop/validate seasons
use the current rules without defcon points, the holdout and live season their own.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from fplopt.seasons import season_label

# src/fplopt/backtest/rules.py -> repo root (the package is installed editable from the repo).
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config" / "scoring"

POSITIONS = MappingProxyType({"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4})
CHIP_NAMES = frozenset({"wildcard", "freehit", "bboost", "3xc"})
CHIP_WEEK_FT = frozenset({"retain_plus_one", "retain"})

# Develop/validate seasons (PLAN §5): scored under the current rules, defcon off.
CURRENT_RULES = "2026-27"
DEVELOP_VALIDATE_SEASONS = range(2016, 2025)
NATIVE_SEASONS = (2025, 2026)


@dataclass(frozen=True)
class ChipWindow:
    """One chip the manager owns: usable once, at a `gw_index` in [start, stop].

    `chip_id` is the API id, distinct per window (set 1 and set 2 of the same chip are two
    windows), so "chip used" is tracked per `chip_id`."""

    chip_id: int
    name: str
    start: int
    stop: int

    def contains(self, gw_index: int) -> bool:
        return self.start <= gw_index <= self.stop


@dataclass(frozen=True)
class Rules:
    """Scoring, squad, transfer and chip rules of one season.

    Per-position mappings are keyed by element_type (1 GK, 2 DEF, 3 MID, 4 FWD) and are
    read-only, as are all other containers (tuples), so a `Rules` can be shared freely.
    Money is int tenths of £m (`budget` 1000 = £100.0m)."""

    label: str
    # scoring, per position
    goals_scored: Mapping[int, int]
    clean_sheets: Mapping[int, int]
    goals_conceded: Mapping[int, int]
    defensive_contribution: Mapping[int, int]
    defcon_threshold: Mapping[int, int | None]
    # scoring, scalars
    assists: int
    saves: int
    penalties_saved: int
    penalties_missed: int
    yellow_cards: int
    red_cards: int
    own_goals: int
    bonus: int
    short_play: int
    long_play: int
    long_play_minutes: int
    saves_per_point: int
    goals_conceded_per_point: int
    defcon_enabled: bool
    # squad
    squad_size: int
    squad_select: Mapping[int, int]
    play_min: Mapping[int, int]
    play_max: Mapping[int, int]
    squad_play: int
    team_limit: int
    budget: int
    sell_on_fee: float
    max_free_transfers: int
    hit_cost: int
    # chips and free transfers
    chips: tuple[ChipWindow, ...]
    chip_week_ft: str
    freehit_consecutive: bool
    ft_topups: tuple[tuple[int, int], ...]


def best_xi(order: Sequence[tuple[int, int]], rules: Rules) -> list[tuple[int, int]]:
    """The starters from `order` ((player_key, element_type), best first): each position's
    `play_min` first in `order`, then the rest in order while their position is under
    `play_max`, up to `squad_play` (fewer when the players can't field them). Optimal for
    any ranking, since the only constraints are per-position lower and upper bounds on a
    fixed-size selection (`backtest.policies.best_lineup`, `optimize.minutes.projected_xi`)."""
    starters: list[tuple[int, int]] = []
    counts: Counter[int] = Counter()
    for et, n in sorted(rules.play_min.items()):
        picked = [p for p in order if p[1] == et][:n]
        starters += picked
        counts[et] += len(picked)
    for player in order:
        if len(starters) >= rules.squad_play:
            break
        if player not in starters and counts[player[1]] < rules.play_max[player[1]]:
            starters.append(player)
            counts[player[1]] += 1
    return starters


def _get(mapping: Any, key: str, where: str) -> Any:
    """mapping[key], or a KeyError naming the file and path of the missing key."""
    if not isinstance(mapping, Mapping) or key not in mapping:
        raise KeyError(f"{where}: missing key {key!r}")
    return mapping[key]


def _per_position(mapping: Any, where: str) -> Mapping[int, Any]:
    """{'GKP': .., 'DEF': .., 'MID': .., 'FWD': ..} -> read-only {1: .., ..., 4: ..}."""
    return MappingProxyType({et: _get(mapping, name, where) for name, et in POSITIONS.items()})


def _int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where}: expected an integer, got {value!r}")
    return value


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"rules config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_rules(
    label: str, *, defcon: bool | None = None, config_dir: Path = DEFAULT_CONFIG_DIR
) -> Rules:
    """Rules from `<label>.json` + `<label>.supplement.json` in `config_dir`.

    `defcon=None` keeps the season's own defcon rule (on iff any position gets points for
    it); `False` switches defcon points off and suffixes the label with `-nodefcon` (so
    caches keyed by label never mix the two); `True` forces it on. Raises KeyError on a
    missing key, ValueError on a malformed value, and NotImplementedError if the season has
    per-GW event overrides (not modelled: they would change scoring for some GWs)."""
    config_dir = Path(config_dir)
    main_path = config_dir / f"{label}.json"
    supp_path = config_dir / f"{label}.supplement.json"
    main, supp = _read(main_path), _read(supp_path)
    m, s = main_path.name, supp_path.name
    for name, config in ((m, main), (s, supp)):
        if _get(config, "season", name) != label:
            raise ValueError(f"{name}: season {config['season']!r} is not {label!r}")
    if _get(main, "event_overrides", m):
        raise NotImplementedError(f"{m}: per-GW event overrides are not supported")

    scoring = _get(main, "scoring", m)
    rules = _get(main, "rules", m)
    supp_scoring = _get(supp, "scoring", s)
    transfers = _get(supp, "transfers", s)
    supp_chips = _get(supp, "chips", s)

    def score(key: str) -> int:
        return _int(_get(scoring, key, f"{m} scoring"), f"{m} scoring.{key}")

    def per_position(key: str) -> Mapping[int, int]:
        where = f"{m} scoring.{key}"
        values = _per_position(_get(scoring, key, f"{m} scoring"), where)
        return MappingProxyType({et: _int(v, where) for et, v in values.items()})

    thresholds = _per_position(
        _get(supp_scoring, "defcon_threshold", f"{s} scoring"), f"{s} scoring.defcon_threshold"
    )
    for et, value in thresholds.items():
        if value is not None:
            _int(value, f"{s} scoring.defcon_threshold[{et}]")
    defcon_points = per_position("defensive_contribution")

    element_types: dict[int, Mapping[str, Any]] = {}
    for item in _get(main, "element_types", m):
        short = _get(item, "singular_name_short", f"{m} element_types")
        et = _int(_get(item, "id", f"{m} element_types"), f"{m} element_types.id")
        if POSITIONS.get(short) != et:
            raise ValueError(f"{m}: element_type {et} is {short!r}, expected {POSITIONS}")
        element_types[et] = item
    if set(element_types) != set(POSITIONS.values()):
        raise ValueError(f"{m}: element_types {sorted(element_types)} are not 1-4")

    def squad(key: str) -> Mapping[int, int]:
        where = f"{m} element_types"
        return MappingProxyType(
            {
                et: _int(_get(element_types[et], key, where), f"{where}[{et}].{key}")
                for et in sorted(element_types)
            }
        )

    chips = []
    for item in _get(main, "chips", m):
        where = f"{m} chips"
        name = _get(item, "name", where)
        if name not in CHIP_NAMES:
            raise ValueError(f"{where}: unknown chip {name!r}")
        chips.append(
            ChipWindow(
                chip_id=_int(_get(item, "id", where), f"{where}.id"),
                name=name,
                start=_int(_get(item, "start_event", where), f"{where}.start_event"),
                stop=_int(_get(item, "stop_event", where), f"{where}.stop_event"),
            )
        )
    if len({chip.chip_id for chip in chips}) != len(chips):
        raise ValueError(f"{m}: duplicate chip ids")

    chip_week_ft = _get(transfers, "chip_week_ft", f"{s} transfers")
    if chip_week_ft not in CHIP_WEEK_FT:
        raise ValueError(f"{s}: chip_week_ft {chip_week_ft!r} not in {sorted(CHIP_WEEK_FT)}")
    freehit_consecutive = _get(supp_chips, "freehit_consecutive", f"{s} chips")
    if not isinstance(freehit_consecutive, bool):
        raise ValueError(f"{s}: freehit_consecutive must be true/false")
    topups = tuple(
        (
            _int(_get(item, "gw_index", f"{s} ft_topups"), f"{s} ft_topups.gw_index"),
            _int(_get(item, "amount", f"{s} ft_topups"), f"{s} ft_topups.amount"),
        )
        for item in _get(transfers, "ft_topups", f"{s} transfers")
    )

    enabled = any(defcon_points.values()) if defcon is None else defcon

    def rule(key: str) -> Any:
        return _get(rules, key, f"{m} rules")

    return Rules(
        label=label if defcon is not False else f"{label}-nodefcon",
        goals_scored=per_position("goals_scored"),
        clean_sheets=per_position("clean_sheets"),
        goals_conceded=per_position("goals_conceded"),
        defensive_contribution=defcon_points,
        defcon_threshold=thresholds,
        assists=score("assists"),
        saves=score("saves"),
        penalties_saved=score("penalties_saved"),
        penalties_missed=score("penalties_missed"),
        yellow_cards=score("yellow_cards"),
        red_cards=score("red_cards"),
        own_goals=score("own_goals"),
        bonus=score("bonus"),
        short_play=score("short_play"),
        long_play=score("long_play"),
        long_play_minutes=_int(
            _get(supp_scoring, "long_play_minutes", f"{s} scoring"), f"{s} long_play_minutes"
        ),
        saves_per_point=_int(
            _get(supp_scoring, "saves_per_point", f"{s} scoring"), f"{s} saves_per_point"
        ),
        goals_conceded_per_point=_int(
            _get(supp_scoring, "goals_conceded_per_point", f"{s} scoring"),
            f"{s} goals_conceded_per_point",
        ),
        defcon_enabled=bool(enabled),
        squad_size=_int(rule("squad_squadsize"), f"{m} rules.squad_squadsize"),
        squad_select=squad("squad_select"),
        play_min=squad("squad_min_play"),
        play_max=squad("squad_max_play"),
        squad_play=_int(rule("squad_squadplay"), f"{m} rules.squad_squadplay"),
        team_limit=_int(rule("squad_team_limit"), f"{m} rules.squad_team_limit"),
        budget=_int(rule("squad_total_spend"), f"{m} rules.squad_total_spend"),
        sell_on_fee=float(rule("transfers_sell_on_fee")),
        max_free_transfers=1
        + _int(rule("max_extra_free_transfers"), f"{m} rules.max_extra_free_transfers"),
        hit_cost=_int(_get(transfers, "hit_cost", f"{s} transfers"), f"{s} hit_cost"),
        chips=tuple(chips),
        chip_week_ft=chip_week_ft,
        freehit_consecutive=freehit_consecutive,
        ft_topups=topups,
    )


def backtest_rules(season: int, *, config_dir: Path = DEFAULT_CONFIG_DIR) -> Rules:
    """The rules a season is scored under in backtests (PLAN §5 table).

    2016/17-2024/25 (develop/validate): the current rules with defcon off, label
    `2026-27-nodefcon`; their chip windows are applied to `gw_index`. 2025/26 (holdout) and
    2026/27 (live): their native rules. Any other season raises ValueError."""
    if season in DEVELOP_VALIDATE_SEASONS:
        return load_rules(CURRENT_RULES, defcon=False, config_dir=config_dir)
    if season in NATIVE_SEASONS:
        return load_rules(season_label(season), config_dir=config_dir)
    raise ValueError(f"no backtest rules for season {season_label(season)}")


def legacy_rules(*, config_dir: Path = DEFAULT_CONFIG_DIR) -> Rules:
    """FPL's scoring for 2016/17-2024/25: the current rules with GK goals worth 6 (10 since
    2025/26) and no defcon. Only for checking `score_matches` against the archived
    `total_points` of those seasons; backtests score them with `backtest_rules`."""
    current = load_rules(CURRENT_RULES, defcon=False, config_dir=config_dir)
    goals = MappingProxyType({**current.goals_scored, 1: 6})
    return dataclasses.replace(current, label="legacy", goals_scored=goals)
