"""Squad state, transfers, free transfers and chips (fplopt.backtest.state)."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import pandas as pd
import pytest

from fplopt.backtest.state import (
    Decision,
    Holding,
    InvalidDecision,
    SquadState,
    Transfer,
    TransferRecord,
    apply_decision,
    chip_available,
    next_state,
    refresh,
    selling_price,
    squad_value,
)

# --- local stand-ins for Task 2's Rules / ChipWindow / Lineup ------------------------------


@dataclass(frozen=True)
class ChipWindow:
    chip_id: int
    name: str
    start: int
    stop: int


# The 2026/27 windows (config/scoring/2026-27.json): set 1 to GW19, set 2 from GW20;
# Wildcard and Free Hit set 1 start at GW2.
CHIPS = (
    ChipWindow(1, "wildcard", 2, 19),
    ChipWindow(2, "wildcard", 20, 38),
    ChipWindow(3, "freehit", 2, 19),
    ChipWindow(4, "bboost", 1, 19),
    ChipWindow(5, "3xc", 1, 19),
    ChipWindow(6, "freehit", 20, 38),
    ChipWindow(7, "bboost", 20, 38),
    ChipWindow(8, "3xc", 20, 38),
)


def _ro(d: dict[int, int]) -> Mapping[int, int]:
    return MappingProxyType(d)


@dataclass(frozen=True)
class Rules:
    squad_select: Mapping[int, int] = field(default_factory=lambda: _ro({1: 2, 2: 5, 3: 5, 4: 3}))
    play_min: Mapping[int, int] = field(default_factory=lambda: _ro({1: 1, 2: 3, 3: 2, 4: 1}))
    play_max: Mapping[int, int] = field(default_factory=lambda: _ro({1: 1, 2: 5, 3: 5, 4: 3}))
    squad_play: int = 11
    team_limit: int = 3
    budget: int = 1000
    sell_on_fee: float = 0.5
    max_free_transfers: int = 5
    hit_cost: int = 4
    chips: tuple[ChipWindow, ...] = CHIPS
    chip_week_ft: str = "retain_plus_one"
    freehit_consecutive: bool = False
    ft_topups: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class Lineup:
    starters: tuple[int, ...]
    bench: tuple[int, ...]
    captain: int
    vice: int


RULES = Rules()

# --- hand-made pool -----------------------------------------------------------------------
# key = club * 100 + element_type * 10 + i; 8 clubs, each with 2 GK, 5 DEF, 5 MID, 3 FWD.
BASE_PRICE = {1: 45, 2: 50, 3: 70, 4: 75}
N_PER_POS = {1: 2, 2: 5, 3: 5, 4: 3}
CLUBS = range(1, 9)


def make_pool(
    overrides: Mapping[int, int] | None = None, drop: tuple[int, ...] = ()
) -> pd.DataFrame:
    rows = [
        (club * 100 + et * 10 + i, et, club, BASE_PRICE[et])
        for club in CLUBS
        for et, n in N_PER_POS.items()
        for i in range(n)
    ]
    pool = pd.DataFrame(rows, columns=["player_key", "element_type", "team_key", "price"])
    for key, price in (overrides or {}).items():
        pool.loc[pool["player_key"] == key, "price"] = price
    pool = pool[~pool["player_key"].isin(drop)].reset_index(drop=True)
    return pool.astype("int64")


POOL = make_pool()
# 3 players from each of clubs 1-5: GK 2, DEF 5, MID 5, FWD 3. Cost 915 → bank 85.
BASE_KEYS = (110, 120, 121, 210, 220, 221, 320, 330, 331, 430, 431, 440, 530, 540, 541)


def holding(key: int, purchase: int | None = None, price: int | None = None) -> Holding:
    et, club = key // 10 % 10, key // 100
    base = BASE_PRICE[et]
    return Holding(key, et, club, base if purchase is None else purchase, price or base)


def make_state(
    gw_index: int = 5,
    ft: int = 1,
    keys: tuple[int, ...] = BASE_KEYS,
    bank: int | None = None,
    chips_used: tuple[tuple[int, int], ...] = (),
) -> SquadState:
    holdings = tuple(holding(k) for k in keys)
    if bank is None:
        bank = RULES.budget - sum(h.purchase_price for h in holdings)
    return SquadState(2023, gw_index, holdings, bank, ft, chips_used)


def lineup_for(positions: Mapping[int, int]) -> Lineup:
    """A valid 4-4-2 lineup: lowest keys start, bench = other GK then outfield."""
    by_pos = {et: sorted(k for k, p in positions.items() if p == et) for et in (1, 2, 3, 4)}
    starters = by_pos[1][:1] + by_pos[2][:4] + by_pos[3][:4] + by_pos[4][:2]
    bench = by_pos[1][1:] + by_pos[2][4:] + by_pos[3][4:] + by_pos[4][2:]
    return Lineup(tuple(starters), tuple(bench), starters[1], starters[2])


def positions_after(state: SquadState, transfers, pool: pd.DataFrame) -> dict[int, int]:
    pos = {h.player_key: h.element_type for h in state.holdings}
    pool_pos = dict(zip(pool["player_key"], pool["element_type"], strict=True))
    for t in transfers:
        pos.pop(t.out_key, None)
        pos[t.in_key] = int(pool_pos.get(t.in_key, t.in_key // 10 % 10))
    return pos


def decide(state, pairs=(), chip=None, pool=POOL) -> Decision:
    transfers = tuple(Transfer(o, i) for o, i in pairs)
    return Decision(transfers, lineup_for(positions_after(state, transfers, pool)), chip)


def run(state, pairs=(), chip=None, pool=POOL, rules=RULES):
    gw_state, record = apply_decision(state, decide(state, pairs, chip, pool), pool, rules)
    return gw_state, record, next_state(gw_state, record, rules, state.gw_index + 1)


# --- money --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("purchase", "current", "expected"),
    [
        (50, 53, 51),
        (50, 51, 50),
        (50, 52, 51),
        (50, 50, 50),
        (50, 48, 48),
        (50, 60, 55),
        (45, 46, 45),
    ],
)
def test_selling_price_keeps_half_the_rise_rounded_down(purchase, current, expected):
    assert selling_price(Holding(1, 3, 1, purchase, current)) == expected


def test_selling_price_other_fee():
    h = Holding(1, 3, 1, 50, 60)
    assert selling_price(h, sell_on_fee=0.0) == 50
    assert selling_price(h, sell_on_fee=1.0) == 60
    assert selling_price(Holding(1, 3, 1, 50, 53), sell_on_fee=0.3) == 50  # floor(0.9)


def test_squad_value_sums_selling_prices():
    state = replace(make_state(), holdings=(Holding(1, 3, 1, 50, 53), Holding(2, 3, 2, 60, 58)))
    assert squad_value(state) == 51 + 58


# --- state basics ------------------------------------------------------------------------


def test_holdings_sorted_and_unique():
    state = make_state(keys=tuple(reversed(BASE_KEYS)))
    assert state.player_keys == tuple(sorted(BASE_KEYS))
    with pytest.raises(ValueError, match="duplicate"):
        make_state(keys=BASE_KEYS[:-1] + (110,))


def test_refresh_updates_club_and_price_and_keeps_departed():
    pool = make_pool(overrides={330: 73}, drop=(541,))
    pool.loc[pool["player_key"] == 120, "team_key"] = 7
    state = make_state()
    state = replace(
        state,
        holdings=tuple(replace(h, price=49) if h.player_key == 541 else h for h in state.holdings),
    )
    new = refresh(state, pool)
    by_key = {h.player_key: h for h in new.holdings}
    assert by_key[330].price == 73 and by_key[330].purchase_price == 70
    assert by_key[120].team_key == 7
    assert by_key[541] == holding(541, price=49)  # left the game: last known club and price
    assert refresh(new, pool) is new  # nothing to change


def test_pool_validation():
    with pytest.raises(ValueError, match="missing columns"):
        refresh(make_state(), POOL.drop(columns="price"))
    with pytest.raises(ValueError, match="duplicate"):
        refresh(make_state(), pd.concat([POOL, POOL.head(1)]))


# --- free transfers and hits ------------------------------------------------------------


def test_ft_banking_caps_at_five():
    state = make_state(gw_index=2, ft=1)
    seen = []
    for _ in range(6):
        _, record, state = run(state)
        assert record == TransferRecord(0, 0, 0, 0, None, None)
        seen.append(state.free_transfers)
    assert seen == [2, 3, 4, 5, 5, 5]
    assert state.gw_index == 8


def test_transfers_use_free_transfers_then_cost_hits():
    state = make_state(ft=2)
    _, record, nxt = run(state, [(120, 620)])
    assert (record.n_transfers, record.free_used, record.hits, record.hit_points) == (1, 1, 0, 0)
    assert nxt.free_transfers == 2  # 2 - 1 + 1

    _, record, nxt = run(state, [(120, 620), (330, 630), (540, 640)])
    assert (record.n_transfers, record.free_used, record.hits, record.hit_points) == (3, 2, 1, 4)
    assert nxt.free_transfers == 1  # max(2 - 3, 0) + 1

    _, record, nxt = run(make_state(ft=1), [(120, 620), (330, 630), (540, 640)])
    assert (record.hits, record.hit_points) == (2, 8)
    assert nxt.free_transfers == 1


def test_hit_cost_from_rules():
    _, record, _ = run(make_state(ft=1), [(120, 620), (330, 630)], rules=replace(RULES, hit_cost=6))
    assert record.hit_points == 6


def test_gw1_transfers_unlimited_and_next_gw_has_one_ft():
    state = make_state(gw_index=1, ft=0)
    pairs = [(120, 620), (121, 621), (330, 730), (331, 731), (540, 840), (541, 841)]
    _, record, nxt = run(state, pairs)
    assert (record.n_transfers, record.free_used, record.hits, record.hit_points) == (6, 6, 0, 0)
    assert nxt.free_transfers == 1 and nxt.gw_index == 2
    _, _, nxt = run(state)  # rolling at GW1 still gives 1, not 2
    assert nxt.free_transfers == 1


@pytest.mark.parametrize("chip", ["wildcard", "freehit"])
@pytest.mark.parametrize(
    ("chip_week_ft", "ft", "expected"),
    [("retain_plus_one", 3, 4), ("retain_plus_one", 5, 5), ("retain", 3, 3), ("retain", 1, 1)],
)
def test_wildcard_and_freehit_transfers_are_free_and_ft_follows_rules(
    chip, chip_week_ft, ft, expected
):
    rules = replace(RULES, chip_week_ft=chip_week_ft)
    state = make_state(gw_index=6, ft=ft)
    pairs = [(120, 620), (121, 621), (330, 730), (331, 731), (540, 840), (541, 841)]
    _, record, nxt = run(state, pairs, chip=chip, rules=rules)
    assert (record.n_transfers, record.free_used, record.hits, record.hit_points) == (6, 6, 0, 0)
    assert nxt.free_transfers == expected


def test_unknown_chip_week_ft_raises():
    gw_state, record = apply_decision(
        make_state(), decide(make_state(), chip="wildcard"), POOL, RULES
    )
    with pytest.raises(ValueError, match="chip_week_ft"):
        next_state(gw_state, record, replace(RULES, chip_week_ft="reset"), 6)


def test_bench_boost_and_triple_captain_keep_normal_ft_rules():
    for chip in ("bboost", "3xc"):
        _, record, nxt = run(make_state(ft=1), [(120, 620), (330, 630)], chip=chip)
        assert (record.hits, record.chip) == (1, chip)
        assert nxt.free_transfers == 1


def test_ft_topups_added_at_their_gw_and_capped():
    rules = replace(RULES, ft_topups=((16, 4),))
    _, _, nxt = run(make_state(gw_index=15, ft=1), rules=rules)
    assert nxt.free_transfers == 5  # 2 + 4, capped
    _, _, nxt = run(make_state(gw_index=15, ft=1), [(120, 620)], rules=rules)
    assert nxt.free_transfers == 5  # 1 + 4
    _, _, nxt = run(make_state(gw_index=14, ft=1), rules=rules)
    assert nxt.free_transfers == 2  # top-up is for gw_index 16 only


def test_next_state_rejects_going_backwards():
    gw_state, record = apply_decision(make_state(), decide(make_state()), POOL, RULES)
    with pytest.raises(ValueError, match="next_gw_index"):
        next_state(gw_state, record, RULES, 5)


# --- Free Hit -------------------------------------------------------------------------------


def test_freehit_reverts_squad_and_bank():
    state = make_state(gw_index=8, ft=2)
    pairs = [(120, 620), (121, 621), (330, 730), (331, 731), (540, 840), (541, 841)]
    pool = make_pool(overrides={620: 40, 621: 40})  # cheaper ins: bank changes
    gw_state, record = apply_decision(state, decide(state, pairs, "freehit", pool), pool, RULES)
    assert gw_state.bank == state.bank + 2 * 10
    assert gw_state.freehit_backup == state.holdings and gw_state.freehit_bank == state.bank
    assert {620, 621, 730, 841} <= set(gw_state.player_keys)
    nxt = next_state(gw_state, record, RULES, 9)
    assert nxt.holdings == state.holdings and nxt.bank == state.bank
    assert nxt.freehit_backup is None and nxt.freehit_bank is None
    assert nxt.chips_used == ((3, 8),)
    assert nxt.free_transfers == 3


def test_freehit_backup_holds_deadline_prices_and_is_refreshed_later():
    state = make_state(gw_index=8)
    pool = make_pool(overrides={330: 74})
    gw_state, record = apply_decision(
        state, decide(state, [(330, 630)], "freehit", pool), pool, RULES
    )
    assert {h.player_key: h.price for h in gw_state.freehit_backup}[330] == 74
    nxt = next_state(gw_state, record, RULES, 9)
    later = refresh(nxt, make_pool(overrides={330: 75}))
    assert {h.player_key: h.price for h in later.holdings}[330] == 75


def test_apply_on_freehit_gw_state_raises():
    state = make_state(gw_index=8)
    gw_state, _ = apply_decision(state, decide(state, chip="freehit"), POOL, RULES)
    with pytest.raises(ValueError, match="Free Hit"):
        apply_decision(gw_state, decide(gw_state), POOL, RULES)


def test_freehit_not_in_consecutive_gameweeks():
    _, _, at20 = run(make_state(gw_index=19), chip="freehit")
    assert at20.chips_used == ((3, 19),)
    assert not chip_available(at20, "freehit", RULES)
    with pytest.raises(InvalidDecision, match="right after a Free Hit"):
        run(at20, chip="freehit")
    # allowed when the rules allow it
    _, record, _ = run(at20, chip="freehit", rules=replace(RULES, freehit_consecutive=True))
    assert record.chip_id == 6
    # and fine a GW later; other chips are fine right after a Free Hit
    _, _, at21 = run(at20)
    assert chip_available(at21, "freehit", RULES)
    _, record, _ = run(at20, chip="wildcard")
    assert record.chip_id == 2


# --- chip windows ----------------------------------------------------------------------------


def test_chip_windows_on_gw_index():
    gw1 = make_state(gw_index=1, ft=0)
    assert not chip_available(gw1, "wildcard", RULES)  # set 1 WC/FH start at GW2
    assert not chip_available(gw1, "freehit", RULES)
    assert chip_available(gw1, "bboost", RULES) and chip_available(gw1, "3xc", RULES)
    with pytest.raises(InvalidDecision, match="no wildcard window"):
        run(gw1, chip="wildcard")
    _, record, _ = run(make_state(gw_index=19), chip="wildcard")
    assert record.chip_id == 1
    _, record, _ = run(make_state(gw_index=20), chip="wildcard")
    assert record.chip_id == 2


def test_set1_chip_unused_is_lost_at_gw20():
    # set-2 bench boost used at 20; the unused set-1 bench boost can't be played later
    _, _, at21 = run(make_state(gw_index=20), chip="bboost")
    assert at21.chips_used == ((7, 20),)
    assert not chip_available(at21, "bboost", RULES)
    with pytest.raises(InvalidDecision, match="already used"):
        run(at21, chip="bboost")


def test_each_chip_id_used_once():
    _, _, at6 = run(make_state(gw_index=5), chip="3xc")
    assert not chip_available(at6, "3xc", RULES)
    with pytest.raises(InvalidDecision, match="already used"):
        run(replace(at6, gw_index=10), chip="3xc")
    _, record, _ = run(replace(at6, gw_index=20), chip="3xc")  # set 2 is separate
    assert record.chip_id == 8


def test_one_chip_per_gameweek_and_unknown_chip():
    state = make_state(gw_index=5, chips_used=((4, 5),))
    assert not chip_available(state, "3xc", RULES)
    with pytest.raises(InvalidDecision, match="already played"):
        run(state, chip="3xc")
    assert not chip_available(make_state(), "doubletrouble", RULES)
    with pytest.raises(InvalidDecision, match="unknown chip"):
        run(make_state(), chip="doubletrouble")


def test_chip_is_checked_before_transfers():
    with pytest.raises(InvalidDecision, match="no wildcard window"):
        run(make_state(gw_index=1), [(999, 620)], chip="wildcard")


# --- transfer validation ----------------------------------------------------------------------


def test_bought_player_purchase_price_is_pool_price():
    pool = make_pool(overrides={620: 52})
    gw_state, _, _ = run(make_state(), [(120, 620)], pool=pool)
    bought = {h.player_key: h for h in gw_state.holdings}[620]
    assert bought == Holding(620, 2, 6, 52, 52)
    assert gw_state.bank == 85 + 50 - 52


@pytest.mark.parametrize(
    ("pairs", "message"),
    [
        ([(999, 620)], "not in the squad"),
        ([(120, 121)], "already in the squad"),
        ([(120, 620), (120, 621)], "sold twice"),
        ([(120, 620), (121, 620)], "bought twice"),
        ([(120, 999)], "not in the pool"),
        ([(120, 630)], "position counts"),
    ],
)
def test_invalid_transfers(pairs, message):
    with pytest.raises(InvalidDecision, match=message):
        run(make_state(ft=5), pairs)


def test_departed_player_can_be_sold_at_last_price_but_not_bought():
    pool = make_pool(drop=(541,))
    state = replace(
        make_state(),
        holdings=tuple(
            Holding(541, 4, 5, 75, 79) if h.player_key == 541 else h for h in make_state().holdings
        ),
    )
    roll, _, _ = run(state, pool=pool)  # can still be held and named in the lineup
    assert 541 in roll.player_keys
    gw_state, _, _ = run(state, [(541, 641)], pool=pool)
    assert gw_state.bank == state.bank + 77 - 75  # sold at 75 + floor(4 / 2)
    with pytest.raises(InvalidDecision, match="not in the pool"):
        run(gw_state, [(641, 541)], pool=pool)


def test_club_cap():
    # clubs 1-5 already have 3 each: a 4th from club 1 is rejected
    with pytest.raises(InvalidDecision, match="club 1 would have 4"):
        run(make_state(), [(320, 122)])
    # swapping within the same club keeps 3
    gw_state, _, _ = run(make_state(), [(120, 122)])
    assert 122 in gw_state.player_keys


def test_club_cap_preexisting_violation():
    # held 320 (club 3) moved to club 1 → 4 players from club 1 without any transfer in
    pool = POOL.copy()
    pool.loc[pool["player_key"] == 320, "team_key"] = 1
    state = make_state()
    gw_state, _, _ = run(state, pool=pool)  # rolling is fine
    assert Counter(h.team_key for h in gw_state.holdings)[1] == 4
    gw_state, _, _ = run(state, [(330, 630)], pool=pool)  # unrelated transfer is fine
    assert Counter(h.team_key for h in gw_state.holdings)[1] == 4
    gw_state, _, _ = run(state, [(120, 620)], pool=pool)  # selling one of them is fine
    assert Counter(h.team_key for h in gw_state.holdings)[1] == 3
    with pytest.raises(InvalidDecision, match="club 1 would have 5"):
        run(state, [(330, 132)], pool=pool)  # adding to it is not
    with pytest.raises(InvalidDecision, match="club 1 would have 4"):
        run(state, [(120, 122)], pool=pool)  # nor replacing one of them with another


def test_budget():
    state = make_state(bank=0)
    gw_state, _, _ = run(state, [(120, 620)])  # same price: bank stays 0
    assert gw_state.bank == 0
    with pytest.raises(InvalidDecision, match="over budget: bank would be -1"):
        run(state, [(120, 620)], pool=make_pool(overrides={620: 51}))
    # selling price, not current price, funds the purchase: bought 50, now 60 → sells 55
    risen = replace(
        state,
        holdings=tuple(
            Holding(120, 2, 1, 50, 60) if h.player_key == 120 else h for h in state.holdings
        ),
    )
    pool = make_pool(overrides={120: 60, 620: 55})
    gw_state, _, _ = run(risen, [(120, 620)], pool=pool)
    assert gw_state.bank == 0
    with pytest.raises(InvalidDecision, match="over budget"):
        run(risen, [(120, 620)], pool=make_pool(overrides={120: 60, 620: 56}))


def test_budget_on_freehit_and_wildcard():
    state = make_state(gw_index=8, bank=0)
    pool = make_pool(overrides={630: 71})
    for chip in ("wildcard", "freehit"):
        with pytest.raises(InvalidDecision, match="over budget"):
            run(state, [(330, 630)], chip=chip, pool=pool)


# --- lineup validation ---------------------------------------------------------------------


def _valid_lineup() -> Lineup:
    return lineup_for({h.player_key: h.element_type for h in make_state().holdings})


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda lu: replace(lu, starters=lu.starters[:10]), "10 starters"),
        (lambda lu: replace(lu, bench=lu.bench[:3]), "3 bench players"),
        (
            lambda lu: replace(lu, bench=(lu.bench[0], lu.bench[1], lu.bench[2], lu.starters[5])),
            "twice",
        ),
        (lambda lu: replace(lu, bench=(lu.bench[0], lu.bench[1], lu.bench[2], 999)), "not in"),
        (lambda lu: replace(lu, bench=(lu.bench[1], lu.bench[0], *lu.bench[2:])), "goalkeeper"),
        (lambda lu: replace(lu, vice=lu.captain), "same player"),
        (lambda lu: replace(lu, captain=lu.bench[1]), "captain .* not a starter"),
        (lambda lu: replace(lu, vice=lu.bench[1]), "vice .* not a starter"),
    ],
)
def test_invalid_lineups(change, message):
    state = make_state()
    with pytest.raises(InvalidDecision, match=message):
        apply_decision(state, Decision((), change(_valid_lineup())), POOL, RULES)


def test_formation_limits():
    state = make_state()
    lu = _valid_lineup()  # starters: GK, DEF×4, MID×4, FWD×2; bench: GK, DEF, MID, FWD
    gk, defs, mids, fwds = lu.starters[0], lu.starters[1:5], lu.starters[5:9], lu.starters[9:]
    bench_gk, bench_def, bench_mid, bench_fwd = lu.bench
    # 5-4-1 and 3-5-2 are valid
    ok = [
        Lineup(
            (gk, *defs, bench_def, *mids, fwds[0]),
            (bench_gk, bench_mid, fwds[1], bench_fwd),
            defs[0],
            defs[1],
        ),
        Lineup(
            (gk, *defs[:3], *mids, bench_mid, *fwds),
            (bench_gk, defs[3], bench_def, bench_fwd),
            defs[0],
            defs[1],
        ),
    ]
    for lineup in ok:
        apply_decision(state, Decision((), lineup), POOL, RULES)
    bad = [
        # 2 DEF
        (
            Lineup(
                (gk, *defs[:2], *mids, bench_mid, *fwds, bench_fwd),
                (bench_gk, defs[2], defs[3], bench_def),
                mids[0],
                mids[1],
            ),
            "element_type 2",
        ),
        # 0 FWD
        (
            Lineup(
                (gk, *defs, bench_def, *mids, bench_mid),
                (bench_gk, *fwds, bench_fwd),
                defs[0],
                defs[1],
            ),
            "element_type 4",
        ),
        # 2 GK
        (
            Lineup(
                (gk, bench_gk, *defs, *mids, fwds[0]),
                (bench_def, bench_mid, fwds[1], bench_fwd),
                defs[0],
                defs[1],
            ),
            "goalkeeper|element_type 1",
        ),
    ]
    for lineup, message in bad:
        with pytest.raises(InvalidDecision, match=message):
            apply_decision(state, Decision((), lineup), POOL, RULES)


# --- property-style: random transfer sequences --------------------------------------------


def _random_pool(rng: random.Random, prices: dict[int, int]) -> pd.DataFrame:
    for key in prices:
        if rng.random() < 0.1:
            prices[key] = max(40, prices[key] + rng.choice((-1, 1)))
    return make_pool(overrides=prices)


def _random_decision(rng: random.Random, state: SquadState, pool: pd.DataFrame, gw: int):
    held = set(state.player_keys)
    k = rng.choice((0, 0, 1, 1, 2, 3, 15 if gw == 1 else 4))
    outs = rng.sample(sorted(held), min(k, len(held)))
    ins: list[int] = []
    for out in outs:
        et = out // 10 % 10
        options = [
            p
            for p in pool["player_key"][pool["element_type"] == et]
            if p not in held and p not in ins
        ]
        ins.append(rng.choice(options))
    chip = None
    if rng.random() < 0.15:
        chip = rng.choice(("wildcard", "freehit", "bboost", "3xc"))
    return decide(state, list(zip(outs, ins, strict=True)), chip, pool)


@pytest.mark.parametrize("chip_week_ft", ["retain_plus_one", "retain"])
@pytest.mark.parametrize("seed", range(8))
def test_random_sequences_preserve_invariants(seed, chip_week_ft):
    rng = random.Random(seed)
    rules = replace(RULES, chip_week_ft=chip_week_ft)
    prices = {int(k): BASE_PRICE[int(k) // 10 % 10] for k in POOL["player_key"]}
    state = make_state(gw_index=1, ft=0)
    applied = 0
    for gw in range(1, 39):
        assert state.gw_index == gw
        pool = _random_pool(rng, prices)
        state = refresh(state, pool)
        decision = _random_decision(rng, state, pool, gw)
        try:
            gw_state, record = apply_decision(state, decision, pool, rules)
        except InvalidDecision:
            decision = replace(
                decision,
                transfers=(),
                chip=None,
                lineup=lineup_for({h.player_key: h.element_type for h in state.holdings}),
            )
            gw_state, record = apply_decision(state, decision, pool, rules)
        else:
            applied += 1

        for s in (gw_state,):
            assert len(s.holdings) == 15 and len(set(s.player_keys)) == 15
            assert Counter(h.element_type for h in s.holdings) == {1: 2, 2: 5, 3: 5, 4: 3}
            assert max(Counter(h.team_key for h in s.holdings).values()) <= 3
            assert s.bank >= 0
        assert record.n_transfers == record.free_used + record.hits
        assert record.hit_points == 4 * record.hits
        if gw == 1 or record.chip in ("wildcard", "freehit"):
            assert record.hits == 0
        else:
            assert record.hits == max(0, record.n_transfers - state.free_transfers)

        nxt = next_state(gw_state, record, rules, gw + 1)
        assert 1 <= nxt.free_transfers <= 5
        if record.chip == "freehit":
            assert nxt.holdings == state.holdings and nxt.bank == state.bank
        else:
            assert nxt.holdings == gw_state.holdings and nxt.bank == gw_state.bank
        ids = [cid for cid, _ in nxt.chips_used]
        assert len(ids) == len(set(ids))
        windows = {c.chip_id: c for c in CHIPS}
        assert all(windows[cid].start <= g <= windows[cid].stop for cid, g in nxt.chips_used)
        assert len({g for _, g in nxt.chips_used}) == len(nxt.chips_used)
        state = nxt
    assert applied >= 15  # the generator exercises real transfers, not only rejections
