"""A real manager's squad for `fplopt optimize plan` (Phase 7 groundwork, pulled forward):
imported from FPL's public API by team ID (`--entry`), from a squad file
(`--squad-file`), or both, the file's values overriding the import's.

**Import** (`import_squad`, public endpoints only, no login): the manager's season history
(`entry/{id}/history/`: per GW bank and transfers made, chips played), the picks of the
last finished GW before the planned one (`entry/{id}/event/{gw}/picks/`) and the transfer
history (`entry/{id}/transfers/`):

- squad and bank: those of the last GW before the planned one; when that GW was a Free
  Hit, the GW before it (the Free Hit squad reverts);
- purchase prices: a player's latest transfer in (`element_in_cost`), Free Hit GWs
  excepted; a player held since the manager's first GW, their price at that GW's deadline
  (the newest player snapshot before it: `initial_prices`);
- free transfers: the season replayed with the backtester's own rule
  (`backtest.state.next_state`: transfers made and chips played per GW, top-ups and cap);
- chips used: the history's chips, by GW.

What the public API can't show: transfers or a chip made for the planned GW before its
deadline. Those (and anything inferred wrongly) go in the squad file.

**Squad file** (TOML, prices in £m; `write_squad_file` writes the import as one to edit):

    season = "2026-27"
    gw = 7
    bank = 0.5
    free_transfers = 2
    chips_used = [["wildcard", 3]]

    [[players]]
    element_id = 229        # or name = "Tarkowski" (web name, unique in the season)
    purchase_price = 5.5

Every key is optional when the file overrides an import (a file with only
`free_transfers = 2` fixes just that); without `--entry` it needs `bank`,
`free_transfers` and the 15 `players`. `players`, when given, replaces the whole squad.

`squad_state` turns the spec into the backtester's `SquadState` at the planned GW
(player keys via `player_season.element_id`, clubs and current prices from the pool).
"""

from __future__ import annotations

import json
import math
import tomllib
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from fplopt.backtest.rules import Rules
from fplopt.backtest.state import (
    Holding,
    InvalidDecision,
    SquadState,
    TransferRecord,
    chip_window,
    next_state,
)
from fplopt.seasons import parse_season_label, season_label

__all__ = (
    "EntryError",
    "SquadPlayer",
    "SquadSpec",
    "apply_overrides",
    "import_squad",
    "read_squad_file",
    "squad_from_api",
    "squad_state",
    "write_squad_file",
)


class EntryError(ValueError):
    """The squad can't be imported or read (a message for the user)."""


class EntryClient(Protocol):
    """The `FplClient` methods the import uses."""

    def entry_history(self, entry_id: int) -> bytes: ...
    def entry_picks(self, entry_id: int, gw: int) -> bytes: ...
    def entry_transfers(self, entry_id: int) -> bytes: ...


@dataclass(frozen=True)
class SquadPlayer:
    """A held player: FPL `element_id` (this season's), purchase price in tenths of £m;
    `name` is informational (the import's web name)."""

    element_id: int
    purchase_price: int
    name: str = ""


@dataclass(frozen=True)
class SquadSpec:
    """A manager's squad before GW `gw` of `season`: bank in tenths of £m, free transfers
    for GW `gw`, chips played as (name, FPL GW). `notes`: how values were inferred."""

    season: int
    gw: int
    players: tuple[SquadPlayer, ...]
    bank: int
    free_transfers: int
    chips_used: tuple[tuple[str, int], ...] = ()
    notes: tuple[str, ...] = ()


# --- import ---------------------------------------------------------------------------------


def import_squad(
    fpl: EntryClient,
    entry_id: int,
    season: int,
    gw: int,
    rules: Rules,
    gw_index: Mapping[int, int],
    initial_prices: Callable[[int], Mapping[int, int]],
) -> SquadSpec:
    """The squad of FPL team `entry_id` before GW `gw` (module docstring). `gw_index` maps the
    season's FPL GW numbers to gw_index; `initial_prices(gw)` gives element_id -> price at
    that GW's deadline."""
    history = _json(fpl.entry_history(entry_id), f"team {entry_id}'s history")
    transfers = _json(fpl.entry_transfers(entry_id), f"team {entry_id}'s transfers")
    return squad_from_api(
        season=season,
        gw=gw,
        history=history,
        picks=lambda event: _json(fpl.entry_picks(entry_id, event), f"GW{event} picks"),
        transfers=transfers,
        rules=rules,
        gw_index=gw_index,
        initial_prices=initial_prices,
    )


def squad_from_api(
    *,
    season: int,
    gw: int,
    history: Mapping[str, Any],
    picks: Callable[[int], Mapping[str, Any]],
    transfers: Sequence[Mapping[str, Any]],
    rules: Rules,
    gw_index: Mapping[int, int],
    initial_prices: Callable[[int], Mapping[int, int]],
) -> SquadSpec:
    """`import_squad` on parsed responses (`picks(gw)` fetches a GW's picks)."""
    rows = sorted(
        (r for r in history.get("current", []) if int(r["event"]) < gw),
        key=lambda r: int(r["event"]),
    )
    if not rows:
        raise EntryError(
            f"the team has no finished GW before GW{gw}: there is no squad to import yet"
        )
    if int(rows[-1]["event"]) != gw - 1:
        raise EntryError(
            f"GW{gw - 1}'s deadline hasn't passed (the history ends at GW{rows[-1]['event']}): "
            f"plan GW{int(rows[-1]['event']) + 1}, or give the squad in a squad file"
        )
    chips = {int(c["event"]): str(c["name"]) for c in history.get("chips", [])}
    unknown = sorted({n for n in chips.values()} - {c.name for c in rules.chips})
    if unknown:
        raise EntryError(f"unknown chip(s) in the history: {unknown}")
    notes = []
    squad_row = rows[-1]
    if chips.get(int(squad_row["event"])) == "freehit":
        if len(rows) < 2:
            raise EntryError("the team's only GW was a Free Hit: no squad to revert to")
        squad_row = rows[-2]
        notes.append(
            f"GW{rows[-1]['event']} was a Free Hit: squad and bank of GW{squad_row['event']}"
        )
    squad_event = int(squad_row["event"])
    elements = [int(p["element"]) for p in picks(squad_event)["picks"]]
    if len(elements) != rules.squad_size:
        raise EntryError(f"GW{squad_event} picks hold {len(elements)} players")

    free_transfers = replay_free_transfers(season, gw, rows, chips, rules, gw_index)
    freehits = {e for e, name in chips.items() if name == "freehit"}
    bought: dict[int, int] = {}
    for t in sorted(transfers, key=lambda t: (int(t["event"]), str(t.get("time", "")))):
        event = int(t["event"])
        if event in freehits or event > squad_event:
            continue
        bought[int(t["element_in"])] = int(t["element_in_cost"])
    first = int(rows[0]["event"])
    start_prices: Mapping[int, int] = {}
    players = []
    for element in elements:
        if element in bought:
            players.append(SquadPlayer(element, bought[element]))
            continue
        if not start_prices:
            start_prices = initial_prices(first)
        if element not in start_prices:
            raise EntryError(
                f"no GW{first} price for element {element} (held since GW{first}): "
                "give its purchase_price in the squad file"
            )
        players.append(SquadPlayer(element, int(start_prices[element])))
    if any(p.element_id not in bought for p in players):
        notes.append(
            f"purchase price of players held since GW{first}: their price at the GW{first} "
            "deadline (newest snapshot before it)"
        )

    notes.append(
        f"free transfers {free_transfers}: replayed from the transfers and chips of "
        f"GW{first}-GW{rows[-1]['event']}"
    )
    notes.append(
        f"transfers or a chip made for GW{gw} before its deadline aren't public: if you made "
        "any, edit the squad file"
    )
    return SquadSpec(
        season=season,
        gw=gw,
        players=tuple(players),
        bank=int(squad_row["bank"]),
        free_transfers=free_transfers,
        chips_used=tuple(sorted(((name, e) for e, name in chips.items()), key=lambda c: c[1])),
        notes=tuple(notes),
    )


def replay_free_transfers(
    season: int,
    gw: int,
    rows: Sequence[Mapping[str, Any]],
    chips: Mapping[int, str],
    rules: Rules,
    gw_index: Mapping[int, int],
) -> int:
    """Free transfers for GW `gw` after the history `rows` (FPL GWs before `gw`, ascending,
    with `event_transfers`): `next_state` GW by GW. The manager's first GW is a fresh squad
    (0 FTs carried in, so 1 after it, as after GW1)."""
    state = SquadState(season, _index(gw_index, int(rows[0]["event"])), (), 0, 0)
    events = [int(r["event"]) for r in rows]
    for row, nxt in zip(rows, [*events[1:], gw], strict=True):
        event = int(row["event"])
        current = replace(state, gw_index=_index(gw_index, event))
        chip = chips.get(event)
        chip_id = None
        if chip is not None:
            try:
                chip_id = chip_window(current, chip, rules)
            except InvalidDecision as exc:
                raise EntryError(f"GW{event} {chip}: {exc}") from None
        if chip == "freehit":
            current = replace(current, freehit_backup=(), freehit_bank=0)
        n = int(row.get("event_transfers", 0))
        record = TransferRecord(n, 0, 0, 0, chip, chip_id)
        state = next_state(current, record, rules, _index(gw_index, nxt))
    return state.free_transfers


def _index(gw_index: Mapping[int, int], gw: int) -> int:
    if gw not in gw_index:
        raise EntryError(f"GW{gw} is not in the season's schedule")
    return int(gw_index[gw])


def _json(body: bytes, what: str) -> Any:
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise EntryError(f"{what}: not JSON ({exc})") from None


# --- squad file -----------------------------------------------------------------------------


def read_squad_file(path: Path) -> dict[str, Any]:
    """The file's keys (module docstring), checked; prices in tenths of £m. Players come
    back as dicts with `element_id` or `name`, and `purchase_price`."""
    try:
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise EntryError(f"squad file {path}: {exc}") from None
    known = {"season", "gw", "bank", "free_transfers", "chips_used", "players"}
    extra = sorted(set(raw) - known)
    if extra:
        raise EntryError(f"squad file {path}: unknown key(s) {extra} (known: {sorted(known)})")
    out: dict[str, Any] = {}
    if "season" in raw:
        try:
            out["season"] = parse_season_label(str(raw["season"]))
        except ValueError as exc:
            raise EntryError(f"squad file {path}: season: {exc}") from None
    if "gw" in raw:
        out["gw"] = _int_value(raw["gw"], "gw", path, minimum=1)
    if "bank" in raw:
        out["bank"] = _tenths(raw["bank"], "bank", path)
    if "free_transfers" in raw:
        out["free_transfers"] = _int_value(raw["free_transfers"], "free_transfers", path)
    if "chips_used" in raw:
        chips = []
        for item in raw["chips_used"]:
            if not (isinstance(item, list) and len(item) == 2):
                raise EntryError(f"squad file {path}: chips_used entries are [name, gw]")
            chips.append((str(item[0]), _int_value(item[1], "chips_used gw", path, minimum=1)))
        out["chips_used"] = tuple(chips)
    if "players" in raw:
        players = []
        for i, item in enumerate(raw["players"], start=1):
            if not isinstance(item, dict) or "purchase_price" not in item:
                raise EntryError(f"squad file {path}: player {i} needs purchase_price")
            if ("element_id" in item) == ("name" in item):
                raise EntryError(f"squad file {path}: player {i}: give element_id or name")
            player = {"purchase_price": _tenths(item["purchase_price"], "purchase_price", path)}
            if "element_id" in item:
                player["element_id"] = _int_value(item["element_id"], "element_id", path)
            else:
                player["name"] = str(item["name"])
            players.append(player)
        out["players"] = players
    return out


def apply_overrides(
    spec: SquadSpec | None,
    overrides: Mapping[str, Any],
    season: int,
    gw: int,
    resolve: Callable[[str], int],
) -> SquadSpec:
    """The spec with the file's values (`read_squad_file`) in place of the import's; without
    an import (`spec` None) the file must give bank, free_transfers and players. The file's
    season and gw, when given, must be the planned ones. `resolve(name)` -> element_id."""
    for key, value in (("season", season), ("gw", gw)):
        if key in overrides and overrides[key] != value:
            shown = season_label(overrides[key]) if key == "season" else overrides[key]
            raise EntryError(f"the squad file is for {key} {shown}, the plan is for {value}")
    if spec is None:
        missing = [k for k in ("bank", "free_transfers", "players") if k not in overrides]
        if missing:
            raise EntryError(f"without --entry the squad file needs {', '.join(missing)}")
        spec = SquadSpec(season, gw, (), 0, 0)
    notes = list(spec.notes)
    changed = {}
    if "players" in overrides:
        players = []
        for item in overrides["players"]:
            element = item["element_id"] if "element_id" in item else resolve(item["name"])
            players.append(SquadPlayer(element, item["purchase_price"], item.get("name", "")))
        changed["players"] = tuple(players)
    for key in ("bank", "free_transfers", "chips_used"):
        if key in overrides:
            changed[key] = overrides[key]
    if changed:
        notes.append(f"from the squad file: {', '.join(sorted(changed))}")
    return replace(spec, **changed, notes=tuple(notes))


def write_squad_file(spec: SquadSpec, path: Path, names: Mapping[int, str]) -> None:
    """The spec as a squad file to edit (module docstring); `names`: element_id -> name,
    written as comments."""
    lines = [
        "# Squad for `fplopt optimize plan --squad-file`. Prices in £m.",
        *(f"# {note}" for note in spec.notes),
        f'season = "{season_label(spec.season)}"',
        f"gw = {spec.gw}",
        f"bank = {spec.bank / 10:.1f}",
        f"free_transfers = {spec.free_transfers}",
        "chips_used = [" + ", ".join(f'["{name}", {g}]' for name, g in spec.chips_used) + "]",
    ]
    for p in spec.players:
        name = names.get(p.element_id, p.name)
        lines += [
            "",
            "[[players]]",
            f"element_id = {p.element_id}" + (f"  # {name}" if name else ""),
            f"purchase_price = {p.purchase_price / 10:.1f}",
        ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _tenths(value: Any, key: str, path: Path) -> int:
    """£m (e.g. 7.5) -> tenths (75); a whole number of tenths, >= 0."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise EntryError(f"squad file {path}: {key} must be a number of £m, got {value!r}")
    tenths = round(float(value) * 10)
    if not math.isfinite(float(value)) or abs(float(value) * 10 - tenths) > 1e-6 or tenths < 0:
        raise EntryError(f"squad file {path}: {key} must be >= 0 in steps of 0.1, got {value}")
    return int(tenths)


def _int_value(value: Any, key: str, path: Path, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EntryError(f"squad file {path}: {key} must be an integer >= {minimum}")
    return int(value)


# --- the backtester's state -----------------------------------------------------------------


def squad_state(
    spec: SquadSpec,
    pool: pd.DataFrame,
    element_keys: Mapping[int, int],
    rules: Rules,
    gw_index: Mapping[int, int],
) -> SquadState:
    """The spec as a `SquadState` at GW `spec.gw` (module docstring): `pool` is the planned
    GW's `player_pool`, `element_keys` this season's element_id -> player_key. Checks the
    squad's size, duplicates and position counts; chips are mapped to their windows."""
    holdings = []
    by_key = pool.set_index("player_key")
    for p in spec.players:
        if p.element_id not in element_keys:
            raise EntryError(f"element {p.element_id} is not a {season_label(spec.season)} player")
        key = int(element_keys[p.element_id])
        if key not in by_key.index:
            raise EntryError(
                f"element {p.element_id} ({p.name or key}) is not in the GW{spec.gw} player "
                "pool (left the league?): sell them in the squad file"
            )
        row = by_key.loc[key]
        holdings.append(
            Holding(
                player_key=key,
                element_type=int(row["element_type"]),
                team_key=int(row["team_key"]),
                purchase_price=int(p.purchase_price),
                price=int(row["price"]),
            )
        )
    _check_squad(holdings, rules)
    gw = _index(gw_index, spec.gw)
    state = SquadState(spec.season, gw, tuple(holdings), spec.bank, spec.free_transfers)
    if spec.free_transfers > rules.max_free_transfers:
        raise EntryError(
            f"free_transfers {spec.free_transfers} > the rules' cap {rules.max_free_transfers}"
        )
    return replace(state, chips_used=chip_ids(spec, rules, gw_index))


def chip_ids(
    spec: SquadSpec, rules: Rules, gw_index: Mapping[int, int]
) -> tuple[tuple[int, int], ...]:
    """The spec's chips as `SquadState.chips_used` ((chip_id, gw_index)), in GW order; each
    takes the window `chip_window` gives at its GW."""
    state = SquadState(spec.season, 1, (), 0, 0)
    for name, gw in sorted(spec.chips_used, key=lambda c: c[1]):
        if gw >= spec.gw:
            raise EntryError(f"chips_used: {name} in GW{gw}, not before GW{spec.gw}")
        at = replace(state, gw_index=_index(gw_index, gw))
        try:
            chip_id = chip_window(at, name, rules)
        except InvalidDecision as exc:
            raise EntryError(f"chips_used: {name} in GW{gw}: {exc}") from None
        state = replace(state, chips_used=(*state.chips_used, (chip_id, at.gw_index)))
    return state.chips_used


def _check_squad(holdings: Iterable[Holding], rules: Rules) -> None:
    holdings = list(holdings)
    keys = [h.player_key for h in holdings]
    if len(set(keys)) != len(keys):
        raise EntryError("the squad holds a player twice")
    if len(holdings) != rules.squad_size:
        raise EntryError(f"the squad has {len(holdings)} players, not {rules.squad_size}")
    counts = Counter(h.element_type for h in holdings)
    if dict(counts) != dict(rules.squad_select):
        raise EntryError(
            f"position counts {dict(sorted(counts.items()))} != {dict(rules.squad_select)}"
        )
