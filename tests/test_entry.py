"""A real manager's squad (`fplopt.entry`): import from FPL's public API responses, the squad
file and its overrides, and the backtester's state."""

from __future__ import annotations

import pandas as pd
import pytest

from fplopt.backtest.rules import backtest_rules
from fplopt.entry import (
    EntryError,
    SquadPlayer,
    SquadSpec,
    apply_overrides,
    read_squad_file,
    replay_free_transfers,
    squad_from_api,
    squad_state,
    write_squad_file,
)

RULES = backtest_rules(2026)  # max 5 FTs, WC/FH retain FTs, chips per half
GW_INDEX = {gw: gw for gw in range(1, 39)}
# 15 elements: 2 GK (1-2), 5 DEF (3-7), 5 MID (8-12), 3 FWD (13-15); club = element % 10.
ELEMENTS = list(range(1, 16))
POSITIONS = {e: 1 if e <= 2 else 2 if e <= 7 else 3 if e <= 12 else 4 for e in range(1, 30)}
START_PRICES = {e: 40 + e for e in range(1, 30)}  # price at the first GW's deadline


def history(rows, chips=()):
    return {
        "current": [{"event": e, "bank": bank, "event_transfers": n} for e, bank, n in rows],
        "chips": [{"name": name, "event": e} for name, e in chips],
    }


def picks_for(squads):
    return lambda event: {"picks": [{"element": e} for e in squads[event]]}


def transfer(event, out, into, cost, time="2026-09-01T10:00:00Z"):
    return {
        "event": event,
        "element_out": out,
        "element_in": into,
        "element_in_cost": cost,
        "time": time,
    }


def imported(rows, squads, transfers=(), chips=(), gw=None):
    gw = max(e for e, _, _ in rows) + 1 if gw is None else gw
    return squad_from_api(
        season=2026,
        gw=gw,
        history=history(rows, chips),
        picks=picks_for(squads),
        transfers=list(transfers),
        rules=RULES,
        gw_index=GW_INDEX,
        initial_prices=lambda gw: START_PRICES,
    )


def test_import_squad_bank_prices_and_free_transfers():
    squad = [*ELEMENTS[:14], 20]  # element 15 sold for 20 in GW3
    spec = imported(
        rows=[(1, 5, 0), (2, 5, 0), (3, 3, 1), (4, 3, 0)],
        squads={4: squad},
        transfers=[transfer(3, 15, 20, 62)],
    )
    assert spec.gw == 5 and spec.bank == 3
    prices = {p.element_id: p.purchase_price for p in spec.players}
    assert prices[20] == 62  # bought: the transfer's cost
    assert prices[1] == START_PRICES[1]  # held since GW1: the GW1 price
    # GW1 -> 1, GW2 (0 made) -> 2, GW3 (1 made) -> 2, GW4 (0 made) -> 3.
    assert spec.free_transfers == 3
    assert spec.chips_used == ()
    assert any("GW5 before its deadline aren't public" in n for n in spec.notes)


def test_the_latest_transfer_in_sets_the_price():
    squad = ELEMENTS
    spec = imported(
        rows=[(1, 0, 0), (2, 0, 1), (3, 0, 1)],
        squads={3: squad},
        transfers=[
            transfer(2, 5, 21, 50),
            transfer(3, 21, 5, 47, time="2026-09-10T10:00:00Z"),  # rebought
        ],
    )
    assert {p.element_id: p.purchase_price for p in spec.players}[5] == 47


def test_wildcard_keeps_the_free_transfers_and_its_transfers_set_prices():
    spec = imported(
        rows=[(1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 9), (5, 0, 0)],
        squads={5: ELEMENTS},
        transfers=[transfer(4, 21, 3, 55)],
        chips=[("wildcard", 4)],
    )
    # 1, 2, 3 after GW1-3; the Wildcard GW retains 3 (rules: "retain"); GW5 -> 4.
    assert spec.free_transfers == 4
    assert spec.chips_used == (("wildcard", 4),)
    assert {p.element_id: p.purchase_price for p in spec.players}[3] == 55


def test_free_hit_reverts_squad_and_bank_and_its_buys_set_no_price():
    before = ELEMENTS
    free_hit = [*ELEMENTS[:14], 25]
    spec = imported(
        rows=[(1, 4, 0), (2, 4, 0), (3, 9, 6)],
        squads={2: before, 3: free_hit},
        transfers=[transfer(3, 15, 25, 70)],
        chips=[("freehit", 3)],
    )
    assert [p.element_id for p in spec.players] == before
    assert spec.bank == 4
    assert all(p.purchase_price == START_PRICES[p.element_id] for p in spec.players)
    assert spec.free_transfers == 2  # 1, 2, Free Hit retains 2
    assert any("Free Hit" in n for n in spec.notes)


def test_a_late_starter_gets_one_free_transfer_after_their_first_gw():
    rows = [{"event": 3, "event_transfers": 0}, {"event": 4, "event_transfers": 0}]
    assert replay_free_transfers(2026, 5, rows, {}, RULES, GW_INDEX) == 2
    spec = imported(rows=[(3, 0, 0)], squads={3: ELEMENTS})
    assert spec.free_transfers == 1


def test_free_transfers_cap_at_five():
    rows = [(e, 0, 0) for e in range(1, 10)]
    assert imported(rows=rows, squads={9: ELEMENTS}).free_transfers == 5


def test_import_errors():
    with pytest.raises(EntryError, match="no finished GW before GW1"):
        imported(rows=[(1, 0, 0)], squads={1: ELEMENTS}, gw=1)
    with pytest.raises(EntryError, match="GW5's deadline hasn't passed.*plan GW5"):
        imported(rows=[(1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)], squads={4: ELEMENTS}, gw=6)
    with pytest.raises(EntryError, match="GW9 is not in the season's schedule"):
        squad_from_api(
            season=2026,
            gw=10,
            history=history([(9, 0, 0)]),
            picks=picks_for({9: ELEMENTS}),
            transfers=[],
            rules=RULES,
            gw_index={g: g for g in range(1, 9)} | {10: 10},
            initial_prices=lambda gw: START_PRICES,
        )
    with pytest.raises(EntryError, match="hold 14 players"):
        imported(rows=[(1, 0, 0)], squads={1: ELEMENTS[:14]})
    with pytest.raises(EntryError, match="already used"):
        imported(
            rows=[(1, 0, 0), (2, 0, 0), (3, 0, 0)],
            squads={3: ELEMENTS},
            chips=[("wildcard", 2), ("wildcard", 3)],
        )


# --- squad file ---------------------------------------------------------------------------


def write(tmp_path, text):
    path = tmp_path / "squad.toml"
    path.write_text(text, encoding="utf-8")
    return path


def spec_of(players=ELEMENTS, bank=0, ft=1, chips=()):
    return SquadSpec(
        2026, 6, tuple(SquadPlayer(e, START_PRICES[e]) for e in players), bank, ft, chips
    )


def test_a_partial_file_overrides_only_its_keys(tmp_path):
    path = write(tmp_path, "free_transfers = 2\n")
    spec = apply_overrides(spec_of(ft=4), read_squad_file(path), 2026, 6, lambda n: 0)
    assert spec.free_transfers == 2 and spec.players == spec_of().players
    assert spec.notes[-1] == "from the squad file: free_transfers"


def test_a_full_file_without_an_import(tmp_path):
    players = "".join(
        f"[[players]]\nelement_id = {e}\npurchase_price = {START_PRICES[e] / 10}\n\n"
        for e in ELEMENTS[1:]
    )
    path = write(
        tmp_path,
        'season = "2026-27"\ngw = 6\nbank = 1.5\nfree_transfers = 2\n'
        'chips_used = [["wildcard", 4]]\n\n'
        f'[[players]]\nname = "Keeper"\npurchase_price = 4.1\n\n{players}',
    )
    spec = apply_overrides(None, read_squad_file(path), 2026, 6, {"Keeper": 1}.__getitem__)
    assert spec.bank == 15 and spec.free_transfers == 2
    assert spec.chips_used == (("wildcard", 4),)
    assert [p.element_id for p in spec.players] == ELEMENTS
    assert spec.players[0].purchase_price == 41


def test_file_errors(tmp_path):
    with pytest.raises(EntryError, match="needs bank, players"):
        apply_overrides(
            None, read_squad_file(write(tmp_path, "free_transfers = 1\n")), 2026, 6, int
        )
    with pytest.raises(EntryError, match="for gw 7"):
        apply_overrides(spec_of(), read_squad_file(write(tmp_path, "gw = 7\n")), 2026, 6, int)
    with pytest.raises(EntryError, match="unknown key"):
        read_squad_file(write(tmp_path, "free_transfer = 1\n"))
    with pytest.raises(EntryError, match="steps of 0.1"):
        read_squad_file(write(tmp_path, "bank = 0.55\n"))
    with pytest.raises(EntryError, match="element_id or name"):
        read_squad_file(write(tmp_path, "[[players]]\npurchase_price = 5.0\n"))


def test_written_file_reads_back(tmp_path):
    spec = spec_of(bank=7, ft=3, chips=(("wildcard", 4),))
    path = tmp_path / "out.toml"
    write_squad_file(spec, path, {1: "Raya"})
    assert "element_id = 1  # Raya" in path.read_text(encoding="utf-8")
    back = apply_overrides(None, read_squad_file(path), 2026, 6, int)
    assert (back.players, back.bank, back.free_transfers, back.chips_used) == (
        spec.players,
        spec.bank,
        spec.free_transfers,
        spec.chips_used,
    )


# --- the backtester's state ---------------------------------------------------------------


def pool(elements=range(1, 30)):
    return pd.DataFrame(
        {
            "player_key": [1000 + e for e in elements],
            "element_type": [POSITIONS[e] for e in elements],
            "team_key": [e % 10 for e in elements],
            "price": [START_PRICES[e] + 3 for e in elements],
        }
    )


KEYS = {e: 1000 + e for e in range(1, 30)}


def test_squad_state():
    state = squad_state(
        spec_of(bank=5, ft=2, chips=(("wildcard", 4),)), pool(), KEYS, RULES, GW_INDEX
    )
    assert state.gw_index == 6 and state.bank == 5 and state.free_transfers == 2
    assert state.chips_used == ((1, 4),)  # the first-half Wildcard window
    held = {h.player_key: h for h in state.holdings}
    assert held[1001].purchase_price == START_PRICES[1] and held[1001].price == START_PRICES[1] + 3


def test_squad_state_errors():
    with pytest.raises(EntryError, match="position counts"):
        squad_state(spec_of([*ELEMENTS[1:], 16]), pool(), KEYS, RULES, GW_INDEX)
    with pytest.raises(EntryError, match="not in the GW6 player pool"):
        squad_state(spec_of(), pool(range(2, 30)), KEYS, RULES, GW_INDEX)
    with pytest.raises(EntryError, match="cap 5"):
        squad_state(spec_of(ft=6), pool(), KEYS, RULES, GW_INDEX)
    with pytest.raises(EntryError, match="not before GW6"):
        squad_state(spec_of(chips=(("bboost", 6),)), pool(), KEYS, RULES, GW_INDEX)
