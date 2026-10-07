import pandas as pd
import pytest

from fplopt.backtest.gw_score import (
    InvalidLineup,
    Lineup,
    gw_outcomes,
    score_gameweek,
    validate_lineup,
)
from fplopt.backtest.rules import load_rules

RULES = load_rules("2026-27")

# Squad: GKs 1-2, DEFs 10-14, MIDs 20-24, FWDs 30-32.
POSITIONS = {1: 1, 2: 1} | {k: 2 for k in range(10, 15)} | {k: 3 for k in range(20, 25)}
POSITIONS |= {k: 4 for k in range(30, 33)}
# 3-4-3; bench: GK, DEF 13, MID 24, DEF 14.
LINEUP = Lineup(
    starters=(1, 10, 11, 12, 20, 21, 22, 23, 30, 31, 32),
    bench=(2, 13, 24, 14),
    captain=30,
    vice=20,
)
# Distinct points per player so sums identify who counted.
POINTS = {key: key % 10 + 1 for key in POSITIONS}


def outcomes(dnp=(), points=POINTS, blank=()):
    """Everyone played 90 minutes except `dnp` (0 minutes, 0 points) and `blank` (absent)."""
    out = {}
    for key in POSITIONS:
        if key in blank:
            continue
        out[key] = (0, 0) if key in dnp else (points[key], 90)
    return out


def total(keys, captain=None, multiplier=2):
    return sum(POINTS[k] for k in keys) + (multiplier - 1) * (POINTS[captain] if captain else 0)


def test_everyone_plays():
    result = score_gameweek(LINEUP, outcomes(), POSITIONS, RULES)
    assert result.counted == LINEUP.starters
    assert result.autosubs == ()
    assert result.captain_used == 30
    assert result.multipliers[30] == 2 and result.multipliers[20] == 1
    assert result.points == total(LINEUP.starters, captain=30)
    assert result.bench_points == total(LINEUP.bench)


def test_goalkeeper_swap():
    result = score_gameweek(LINEUP, outcomes(dnp={1}), POSITIONS, RULES)
    assert result.autosubs == ((1, 2),)
    assert 2 in result.counted and 1 not in result.counted
    assert result.points == total([2, *LINEUP.starters[1:]], captain=30)


def test_bench_goalkeeper_only_replaces_the_goalkeeper():
    result = score_gameweek(LINEUP, outcomes(dnp={20, 13, 24, 14}), POSITIONS, RULES)
    assert result.autosubs == ()
    assert 2 not in result.counted


def test_goalkeeper_not_swapped_when_bench_goalkeeper_did_not_play():
    result = score_gameweek(LINEUP, outcomes(dnp={1, 2}), POSITIONS, RULES)
    assert result.autosubs == ()
    assert result.counted[0] == 1


def test_outfield_subs_in_bench_order():
    """MID 21 and FWD 30 didn't play: bench DEF 13 replaces the first (in pick order: the
    MID, 4-3-3 is valid), bench MID 24 the FWD (2 FWDs stay); DEF 14 stays on the bench."""
    lineup = Lineup(LINEUP.starters, LINEUP.bench, captain=31, vice=20)
    result = score_gameweek(lineup, outcomes(dnp={21, 30}), POSITIONS, RULES)
    assert result.autosubs == ((21, 13), (30, 24))
    assert set(result.counted) == set(LINEUP.starters) - {21, 30} | {13, 24}
    assert result.bench_points == POINTS[2] + POINTS[14]  # 21 and 30 scored 0


def test_pick_order_is_by_position_not_lineup_order():
    """Starters listed FWD first: the non-playing MID is still the first replaced."""
    starters = (30, 31, 32, 1, 10, 11, 12, 20, 21, 22, 23)
    lineup = Lineup(starters, LINEUP.bench, captain=31, vice=20)
    result = score_gameweek(lineup, outcomes(dnp={21, 30, 13}), POSITIONS, RULES)
    assert result.autosubs == ((21, 24), (30, 14))


def test_formation_blocked_sub_uses_the_next_bench_player():
    """DEF 10 didn't play in a 3-4-3: MID 24 (first on the bench) would leave 2 DEFs, so
    DEF 13 comes on instead and 24 stays on the bench."""
    lineup = Lineup(LINEUP.starters, bench=(2, 24, 13, 14), captain=30, vice=20)
    result = score_gameweek(lineup, outcomes(dnp={10}), POSITIONS, RULES)
    assert result.autosubs == ((10, 13),)
    assert 24 not in result.counted


def test_forward_minimum_blocks_a_sub():
    """5-4-1 with the lone FWD out: only a FWD may replace him."""
    starters = (1, 10, 11, 12, 13, 14, 20, 21, 22, 23, 30)
    lineup = Lineup(starters, bench=(2, 24, 31, 32), captain=20, vice=21)
    result = score_gameweek(lineup, outcomes(dnp={30}), POSITIONS, RULES)
    assert result.autosubs == ((30, 31),)


def test_bench_player_who_did_not_play_is_skipped():
    result = score_gameweek(LINEUP, outcomes(dnp={21, 13}), POSITIONS, RULES)
    assert result.autosubs == ((21, 24),)


def test_no_sub_when_no_bench_player_fits():
    result = score_gameweek(LINEUP, outcomes(dnp={21, 13, 24, 14}), POSITIONS, RULES)
    assert result.autosubs == ()
    assert 21 in result.counted


def test_captain_did_not_play_vice_doubles():
    result = score_gameweek(LINEUP, outcomes(dnp={30}), POSITIONS, RULES)
    assert result.captain_used == 20
    assert result.multipliers[20] == 2
    assert result.autosubs == ((30, 13),)
    counted = set(LINEUP.starters) - {30} | {13}
    assert result.points == total(counted, captain=20)


def test_neither_captain_nor_vice_played():
    result = score_gameweek(LINEUP, outcomes(dnp={30, 20}), POSITIONS, RULES)
    assert result.captain_used is None
    assert set(result.multipliers.values()) == {1}
    assert result.points == sum(POINTS[k] for k in result.counted)


def test_triple_captain():
    result = score_gameweek(LINEUP, outcomes(), POSITIONS, RULES, chip="3xc")
    assert result.multipliers[30] == 3
    assert result.points == total(LINEUP.starters, captain=30, multiplier=3)


def test_triple_captain_passes_to_vice():
    result = score_gameweek(LINEUP, outcomes(dnp={30}), POSITIONS, RULES, chip="3xc")
    assert result.captain_used == 20
    assert result.multipliers[20] == 3


def test_bench_boost_counts_all_fifteen_without_autosubs():
    result = score_gameweek(LINEUP, outcomes(dnp={1, 21}), POSITIONS, RULES, chip="bboost")
    assert result.autosubs == ()
    assert result.counted == LINEUP.starters + LINEUP.bench
    assert result.bench_points == 0
    played = set(POSITIONS) - {1, 21}
    assert result.points == total(played, captain=30)


@pytest.mark.parametrize("chip", ["wildcard", "freehit"])
def test_transfer_chips_do_not_change_the_score(chip):
    plain = score_gameweek(LINEUP, outcomes(dnp={21}), POSITIONS, RULES)
    assert score_gameweek(LINEUP, outcomes(dnp={21}), POSITIONS, RULES, chip=chip) == plain


def test_unknown_chip_raises():
    with pytest.raises(ValueError, match="unknown chip"):
        score_gameweek(LINEUP, outcomes(), POSITIONS, RULES, chip="assistant")


def test_blank_counts_as_did_not_play():
    result = score_gameweek(LINEUP, outcomes(blank={21, 30}), POSITIONS, RULES)
    assert result.autosubs == ((21, 13), (30, 24))
    assert result.captain_used == 20


def test_double_gameweek_sums_both_fixtures():
    """Player 30 had a double: 0 minutes in one fixture, 30 in the other -> he played and
    both fixtures' points count (doubled as captain)."""
    rows = [{"player_key": k, "points": POINTS[k], "minutes": 90} for k in POSITIONS if k != 30]
    rows += [
        {"player_key": 30, "points": 0, "minutes": 0},
        {"player_key": 30, "points": 7, "minutes": 30},
        {"player_key": 31, "points": 5, "minutes": 90},  # 31's second fixture
    ]
    per_player = gw_outcomes(pd.DataFrame(rows))
    assert per_player[30] == (7, 30)
    assert per_player[31] == (POINTS[31] + 5, 180)
    result = score_gameweek(LINEUP, per_player, POSITIONS, RULES)
    assert result.autosubs == ()
    assert result.captain_used == 30
    expected = sum(POINTS[k] for k in LINEUP.starters if k != 30) + 5 + 2 * 7
    assert result.points == expected


def test_gw_outcomes_keeps_int_and_float_points():
    ints = gw_outcomes(pd.DataFrame({"player_key": [1, 1], "points": [2, 3], "minutes": [9, 1]}))
    assert ints == {1: (5, 10)} and isinstance(ints[1][0], int)
    floats = gw_outcomes(
        pd.DataFrame({"player_key": [1, 1], "points": [0.25, 0.5], "minutes": [9, 1]})
    )
    assert floats == {1: (0.75, 10)} and isinstance(floats[1][0], float)


def test_float_points_are_kept_exact():
    points = {key: (key % 10 + 1) / 4 for key in POSITIONS}  # exact binary fractions
    out = {key: (points[key], 90) for key in POSITIONS}
    result = score_gameweek(LINEUP, out, POSITIONS, RULES)
    assert isinstance(result.points, float)
    assert result.points == sum(points[k] for k in LINEUP.starters) + points[30]
    assert result.bench_points == sum(points[k] for k in LINEUP.bench)


def test_multipliers_are_read_only():
    result = score_gameweek(LINEUP, outcomes(), POSITIONS, RULES)
    with pytest.raises(TypeError):
        result.multipliers[1] = 5


# --- lineup validation ---------------------------------------------------------------


def test_valid_lineup_passes():
    validate_lineup(LINEUP, POSITIONS, RULES)


@pytest.mark.parametrize(
    ("lineup", "message"),
    [
        (Lineup(LINEUP.starters[:10], LINEUP.bench, 30, 20), "starters"),
        (Lineup(LINEUP.starters, LINEUP.bench[:3], 30, 20), "bench"),
        (Lineup(LINEUP.starters, (2, 13, 24, 13), 30, 20), "duplicate"),
        (Lineup(LINEUP.starters, LINEUP.bench, 30, 30), "differ"),
        (Lineup(LINEUP.starters, LINEUP.bench, 13, 20), "captain 13 is not a starter"),
        (Lineup(LINEUP.starters, LINEUP.bench, 30, 24), "vice 24 is not a starter"),
        (Lineup(LINEUP.starters, (13, 2, 24, 14), 30, 20), "goalkeeper"),
        # 2-5-3: swap DEF 12 for MID 24
        (
            Lineup((1, 10, 11, 24, 20, 21, 22, 23, 30, 31, 32), (2, 13, 12, 14), 30, 20),
            "formation",
        ),
        (Lineup(LINEUP.starters, (2, 13, 24, 99), 30, 20), "no position"),
    ],
)
def test_invalid_lineups(lineup, message):
    with pytest.raises(InvalidLineup, match=message):
        validate_lineup(lineup, POSITIONS, RULES)


def test_invalid_squad_composition():
    positions = POSITIONS | {14: 1}  # three goalkeepers, four defenders
    with pytest.raises(InvalidLineup, match="squad positions"):
        validate_lineup(LINEUP, positions, RULES)


def test_score_gameweek_validates_the_lineup():
    with pytest.raises(InvalidLineup):
        score_gameweek(Lineup(LINEUP.starters, LINEUP.bench, 30, 30), outcomes(), POSITIONS, RULES)
