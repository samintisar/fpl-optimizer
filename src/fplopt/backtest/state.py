"""Squad state, transfers, free transfers and chips for the backtester (PLAN §5, §7).

The simulator carries one `SquadState` from gameweek to gameweek. Each GW:

1. `refresh(state, pool)` brings held players' club and price up to the deadline's pool;
2. the policy returns a `Decision` (transfers, lineup, chip);
3. `apply_decision` validates it against the FPL rules and returns the squad that plays this
   GW (`gw_state`) plus a `TransferRecord` (hits, chip);
4. after scoring, `next_state(gw_state, record, rules, next_gw_index)` accrues free
   transfers, records the chip and reverts a Free Hit.

FPL rules implemented here (values come from the rules config, PLAN §3):

- **Money** is int tenths of £m. A player sells for his purchase price plus
  `floor((current − purchase) × sell_on_fee)` if his price rose, else his current price (FPL
  keeps half of every rise, rounded down to £0.1m, and passes on every fall).
- **Squad:** `squad_select` players per position (2/5/5/3), at most `team_limit` per club,
  bank ≥ 0. A club already over the cap (a held player moved club) is tolerated as long as
  no player of that club is bought.
- **Free transfers:** at `gw_index == 1` transfers are unlimited and free (pre-season squad
  building) and GW2 starts with 1 FT. Otherwise each transfer beyond the FTs costs
  `hit_cost` points; unused FTs bank up to `max_free_transfers`:
  `ft' = min(cap, max(ft − n, 0) + 1)`. Wildcard and Free Hit make every transfer free, and
  the next GW's FTs follow `chip_week_ft` (`retain_plus_one`: ft + 1, `retain`: ft). FT
  top-ups (`ft_topups`, e.g. 2025/26 AFCON) are added at their `gw_index`. Always capped.
- **Chips:** a chip is a window (`chip_id`, name, start..stop on `gw_index`); each `chip_id`
  can be used once, so a set-1 chip not played by its last GW is lost. At most one chip per
  GW (a `Decision` holds one chip). A Free Hit can't follow a Free Hit in the next GW unless
  `freehit_consecutive`. A Free Hit's squad lasts one GW: the pre-transfer squad and bank
  come back at `next_state`.

Pure: no I/O, no pandas beyond reading the pool frame. States are frozen and updated with
`dataclasses.replace`.

"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import TYPE_CHECKING, NamedTuple

from fplopt.backtest.gw_score import InvalidLineup, Lineup, validate_lineup

if TYPE_CHECKING:
    import pandas as pd

    from fplopt.backtest.rules import Rules

GOALKEEPER = 1
TRANSFER_CHIPS = frozenset({"wildcard", "freehit"})
POOL_COLUMNS = ("player_key", "element_type", "team_key", "price")
CHIP_WEEK_FT = ("retain_plus_one", "retain")


class InvalidDecision(ValueError):
    """A decision that breaks an FPL rule (chip, transfers, squad, budget or lineup)."""


@dataclass(frozen=True)
class Holding:
    """A held player. `price` is his latest known current price (tenths of £m); a player
    who left the game keeps his last price and club."""

    player_key: int
    element_type: int
    team_key: int
    purchase_price: int
    price: int


@dataclass(frozen=True)
class SquadState:
    """Everything that carries over between GWs, at the deadline of `gw_index`.

    `free_transfers` are those available for this GW's decision (ignored at GW1, where
    transfers are unlimited). `chips_used` holds `(chip_id, gw_index)` pairs.
    `freehit_backup` / `freehit_bank` are set only on the state returned by
    `apply_decision` for a Free Hit GW: the squad and bank to restore afterwards.
    Holdings are kept sorted by `player_key`."""

    season: int
    gw_index: int
    holdings: tuple[Holding, ...]
    bank: int
    free_transfers: int
    chips_used: tuple[tuple[int, int], ...] = ()
    freehit_backup: tuple[Holding, ...] | None = None
    freehit_bank: int | None = None

    def __post_init__(self) -> None:
        holdings = _sorted(self.holdings)
        keys = [h.player_key for h in holdings]
        if len(set(keys)) != len(keys):
            raise ValueError(f"duplicate player_key in holdings: {_duplicates(keys)}")
        object.__setattr__(self, "holdings", holdings)
        object.__setattr__(self, "chips_used", tuple(tuple(c) for c in self.chips_used))
        if self.freehit_backup is not None:
            object.__setattr__(self, "freehit_backup", _sorted(self.freehit_backup))
        if (self.freehit_backup is None) != (self.freehit_bank is None):
            raise ValueError("freehit_backup and freehit_bank must be set together")

    @property
    def player_keys(self) -> tuple[int, ...]:
        return tuple(h.player_key for h in self.holdings)


@dataclass(frozen=True)
class Transfer:
    out_key: int
    in_key: int


@dataclass(frozen=True)
class Decision:
    """One GW's decision: transfers (executed together), the lineup for the post-transfer
    squad, and an optional chip name (`wildcard`, `freehit`, `bboost`, `3xc`)."""

    transfers: tuple[Transfer, ...]
    lineup: Lineup
    chip: str | None = None


@dataclass(frozen=True)
class TransferRecord:
    """What `apply_decision` did. `free_used` counts the transfers that cost nothing (all of
    them at GW1 and under WC/FH), so `n_transfers == free_used + hits`. `chip_id` is the
    window the chip was taken from (recorded in `chips_used` by `next_state`)."""

    n_transfers: int
    free_used: int
    hits: int
    hit_points: int
    chip: str | None
    chip_id: int | None = None


class _PoolPlayer(NamedTuple):
    element_type: int
    team_key: int
    price: int


# --- money ------------------------------------------------------------------------------


def selling_price(holding: Holding, sell_on_fee: float = 0.5) -> int:
    """FPL selling price in tenths of £m: purchase + floor((current − purchase) × fee) when
    the price rose, else the current price. Exact for decimal fees (0.5 → half the rise,
    rounded down): bought 50, now 53 → 51; now 51 → 50; now 48 → 48."""
    rise = holding.price - holding.purchase_price
    if rise <= 0:
        return holding.price
    return holding.purchase_price + math.floor(rise * Fraction(str(sell_on_fee)))


def squad_value(state: SquadState, sell_on_fee: float = 0.5) -> int:
    """Sum of the holdings' selling prices (what the squad would raise), bank excluded."""
    return sum(selling_price(h, sell_on_fee) for h in state.holdings)


# --- pool ---------------------------------------------------------------------------------


def _pool_index(pool: pd.DataFrame) -> dict[int, _PoolPlayer]:
    missing = [c for c in POOL_COLUMNS if c not in pool.columns]
    if missing:
        raise ValueError(f"pool is missing columns {missing}")
    keys = pool["player_key"].tolist()
    if len(set(keys)) != len(keys):
        raise ValueError(f"pool has duplicate player_key: {_duplicates(keys)}")
    rows = zip(*(pool[c].tolist() for c in POOL_COLUMNS), strict=True)
    return {int(k): _PoolPlayer(int(et), int(team), int(price)) for k, et, team, price in rows}


def refresh(state: SquadState, pool: pd.DataFrame) -> SquadState:
    """Update held players' `team_key` and `price` from the deadline's pool (columns
    `player_key, element_type, team_key, price`). Players absent from the pool (left the
    game) keep their last known club and price, so they can still be sold."""
    return _refresh(state, _pool_index(pool))


def _refresh(state: SquadState, players: Mapping[int, _PoolPlayer]) -> SquadState:
    holdings = tuple(_refreshed(h, players) for h in state.holdings)
    return state if holdings == state.holdings else replace(state, holdings=holdings)


def _refreshed(holding: Holding, players: Mapping[int, _PoolPlayer]) -> Holding:
    p = players.get(holding.player_key)
    if p is None or (p.team_key == holding.team_key and p.price == holding.price):
        return holding
    return replace(holding, team_key=p.team_key, price=p.price)


# --- chips --------------------------------------------------------------------------------


def chip_available(state: SquadState, name: str, rules: Rules) -> bool:
    """Whether chip `name` can be played at `state.gw_index` (window open and unused, Free Hit
    not right after a Free Hit unless allowed)."""
    try:
        _chip_window(state, name, rules)
    except InvalidDecision:
        return False
    return True


def chip_window(state: SquadState, name: str, rules: Rules) -> int:
    """The `chip_id` that playing chip `name` at `state.gw_index` would use (the lowest
    open, unused window), or InvalidDecision if the chip can't be played there (same rules
    as `chip_available`; the optimizer's chip scenarios use it)."""
    return _chip_window(state, name, rules)


def _chip_window(state: SquadState, name: str, rules: Rules) -> int:
    """The `chip_id` of the open, unused window for `name`, or InvalidDecision."""
    names = {c.chip_id: c.name for c in rules.chips}
    if name not in names.values():
        raise InvalidDecision(f"unknown chip {name!r} (rules have {sorted(set(names.values()))})")
    gw = state.gw_index
    # One chip per GW needs no check: a Decision holds a single chip, and `chips_used` only
    # records chips of earlier GWs (next_state moves on to the following gw_index).
    if (
        name == "freehit"
        and not rules.freehit_consecutive
        and any(g == gw - 1 and names.get(cid) == "freehit" for cid, g in state.chips_used)
    ):
        raise InvalidDecision(f"Free Hit at gw_index {gw} right after a Free Hit")
    used = {cid for cid, _ in state.chips_used}
    open_windows = [c for c in rules.chips if c.name == name and c.start <= gw <= c.stop]
    if not open_windows:
        raise InvalidDecision(f"no {name} window contains gw_index {gw}")
    for window in sorted(open_windows, key=lambda c: c.chip_id):
        if window.chip_id not in used:
            return window.chip_id
    raise InvalidDecision(f"{name} for gw_index {gw} already used")


# --- decisions ------------------------------------------------------------------------------


def apply_decision(
    state: SquadState, decision: Decision, pool: pd.DataFrame, rules: Rules
) -> tuple[SquadState, TransferRecord]:
    """Validate and execute a decision; return the squad that plays this GW and a record.

    The state is first refreshed from the pool (clubs and prices as of the deadline). Checks,
    in order: (1) chip available; (2) outs held, ins in the pool and not held, no player
    twice; (3) final position counts equal `squad_select`; (4) club cap: no club over
    `team_limit` that a bought player belongs to (a pre-existing excess from a held player
    changing club is tolerated if nobody from that club is bought); (5) bank ≥ 0 after
    selling at selling price and buying at pool price; (6) lineup valid for the new squad.

    A bought player's purchase price is his pool price. Hits: none at GW1 or under WC/FH;
    otherwise every transfer beyond `free_transfers` costs `hit_cost`. A Free Hit keeps the
    pre-transfer holdings and bank in `freehit_backup` / `freehit_bank`.
    `gw_state.free_transfers` stays at the pre-decision value; `next_state` accrues it."""
    if state.freehit_backup is not None:
        raise ValueError("state is a Free Hit gameweek state; call next_state first")
    players = _pool_index(pool)
    state = _refresh(state, players)
    chip = decision.chip
    chip_id = _chip_window(state, chip, rules) if chip is not None else None

    held = {h.player_key: h for h in state.holdings}
    outs = [t.out_key for t in decision.transfers]
    ins = [t.in_key for t in decision.transfers]
    if len(set(outs)) != len(outs):
        raise InvalidDecision(f"player sold twice: {_duplicates(outs)}")
    if len(set(ins)) != len(ins):
        raise InvalidDecision(f"player bought twice: {_duplicates(ins)}")
    for key in outs:
        if key not in held:
            raise InvalidDecision(f"cannot sell {key}: not in the squad")
    for key in ins:
        if key in held:
            raise InvalidDecision(f"cannot buy {key}: already in the squad")
        if key not in players:
            raise InvalidDecision(f"cannot buy {key}: not in the pool")

    sold = set(outs)
    bought = [
        Holding(k, players[k].element_type, players[k].team_key, players[k].price, players[k].price)
        for k in ins
    ]
    holdings = _sorted([h for h in state.holdings if h.player_key not in sold] + bought)

    counts = Counter(h.element_type for h in holdings)
    wanted = {et: n for et, n in rules.squad_select.items() if n}
    if dict(counts) != wanted:
        raise InvalidDecision(
            f"position counts {dict(sorted(counts.items()))} != squad_select {wanted}"
        )

    clubs = Counter(h.team_key for h in holdings)
    for team in sorted({h.team_key for h in bought}):
        if clubs[team] > rules.team_limit:
            raise InvalidDecision(
                f"club {team} would have {clubs[team]} players (limit {rules.team_limit})"
            )

    bank = (
        state.bank
        + sum(selling_price(held[k], rules.sell_on_fee) for k in outs)
        - sum(players[k].price for k in ins)
    )
    if bank < 0:
        raise InvalidDecision(f"over budget: bank would be {bank} (tenths of £m)")

    _validate_lineup(decision.lineup, {h.player_key: h.element_type for h in holdings}, rules)

    n = len(decision.transfers)
    if state.gw_index == 1 or chip in TRANSFER_CHIPS:
        free = n
    else:
        free = min(n, max(state.free_transfers, 0))
    hits = n - free
    freehit = chip == "freehit"
    gw_state = replace(
        state,
        holdings=holdings,
        bank=bank,
        freehit_backup=state.holdings if freehit else None,
        freehit_bank=state.bank if freehit else None,
    )
    record = TransferRecord(n, free, hits, hits * rules.hit_cost, chip, chip_id)
    return gw_state, record


def next_state(
    gw_state: SquadState, record: TransferRecord, rules: Rules, next_gw_index: int
) -> SquadState:
    """The state at the next deadline: accrue FTs, record the chip, revert a Free Hit.

    FTs: after GW1 → 1; after WC/FH → per `chip_week_ft` (`retain_plus_one`: ft + 1,
    `retain`: ft); otherwise `max(ft − n, 0) + 1`. Then add `ft_topups` falling in
    (gw_index, next_gw_index] and cap at `max_free_transfers`. A Free Hit restores the
    backup holdings (their club/price are refreshed by the next `refresh`) and bank."""
    if next_gw_index <= gw_state.gw_index:
        raise ValueError(f"next_gw_index {next_gw_index} <= gw_index {gw_state.gw_index}")
    ft = gw_state.free_transfers
    if gw_state.gw_index == 1:
        new_ft = 1
    elif record.chip in TRANSFER_CHIPS:
        if rules.chip_week_ft == "retain_plus_one":
            new_ft = ft + 1
        elif rules.chip_week_ft == "retain":
            new_ft = ft
        else:
            raise ValueError(f"chip_week_ft {rules.chip_week_ft!r} not in {CHIP_WEEK_FT}")
    else:
        new_ft = max(ft - record.n_transfers, 0) + 1
    new_ft += sum(a for g, a in rules.ft_topups if gw_state.gw_index < g <= next_gw_index)
    new_ft = min(rules.max_free_transfers, new_ft)

    chips_used = gw_state.chips_used
    if record.chip is not None:
        if record.chip_id is None:
            raise ValueError("record has a chip but no chip_id")
        chips_used = (*chips_used, (record.chip_id, gw_state.gw_index))

    holdings, bank = gw_state.holdings, gw_state.bank
    if record.chip == "freehit":
        if gw_state.freehit_backup is None or gw_state.freehit_bank is None:
            raise ValueError("Free Hit record but the state has no backup squad")
        holdings, bank = gw_state.freehit_backup, gw_state.freehit_bank
    elif gw_state.freehit_backup is not None:
        raise ValueError("state has a Free Hit backup but the record's chip is not freehit")

    return replace(
        gw_state,
        gw_index=next_gw_index,
        holdings=holdings,
        bank=bank,
        free_transfers=new_ft,
        chips_used=chips_used,
        freehit_backup=None,
        freehit_bank=None,
    )


# --- lineup -----------------------------------------------------------------------------


def _validate_lineup(lineup: Lineup, positions: Mapping[int, int], rules: Rules) -> None:
    """`gw_score.validate_lineup` for the post-transfer squad, as an InvalidDecision."""
    try:
        validate_lineup(lineup, positions, rules)
    except InvalidLineup as exc:
        raise InvalidDecision(f"invalid lineup: {exc}") from exc


# --- helpers ------------------------------------------------------------------------------


def _sorted(holdings: Iterable[Holding]) -> tuple[Holding, ...]:
    return tuple(sorted(holdings, key=lambda h: h.player_key))


def _duplicates(items: Iterable[int]) -> list[int]:
    return sorted(k for k, n in Counter(items).items() if n > 1)
