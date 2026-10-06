from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fplopt.backtest.rules import backtest_rules, legacy_rules, load_rules
from fplopt.backtest.scoring import COMPONENTS, REQUIRED, score_matches
from fplopt.seasons import HOLDOUT_SEASONS

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RULES = load_rules("2026-27")  # native: defcon on, GK goals 10
NODEFCON = load_rules("2026-27", defcon=False)
GK, DEF, MID, FWD = 1, 2, 3, 4


def frame(*rows: dict) -> pd.DataFrame:
    """Player-match rows: every REQUIRED stat defaults to 0, minutes to 90."""
    base = dict.fromkeys(REQUIRED, 0) | {"minutes": 90}
    return pd.DataFrame([base | row for row in rows])


def score(*rows: dict, rules=RULES) -> pd.DataFrame:
    return score_matches(frame(*rows), rules)


def test_output_columns_dtypes_and_index():
    matches = frame({"element_type": MID}, {"element_type": DEF}).set_axis([7, 3])
    out = score_matches(matches, RULES)
    assert list(out.columns) == [*COMPONENTS, "points"]
    assert (out.dtypes == np.int64).all()
    assert list(out.index) == [7, 3]


def test_appearance_thresholds():
    out = score(*({"element_type": MID, "minutes": m} for m in (0, 1, 59, 60, 90)))
    assert out["appearance"].tolist() == [0, 1, 1, 2, 2]
    assert out["points"].tolist() == [0, 1, 1, 2, 2]


@pytest.mark.parametrize(("element_type", "points"), [(GK, 10), (DEF, 6), (MID, 5), (FWD, 4)])
def test_goals_by_position(element_type, points):
    out = score({"element_type": element_type, "goals_scored": 2})
    assert out["goals"].item() == 2 * points


def test_legacy_goalkeeper_goal_is_six():
    assert score({"element_type": GK, "goals_scored": 1}, rules=legacy_rules())["goals"].item() == 6


@pytest.mark.parametrize("element_type", [GK, DEF, MID, FWD])
def test_assists(element_type):
    assert score({"element_type": element_type, "assists": 2})["assists"].item() == 6


@pytest.mark.parametrize(("element_type", "points"), [(GK, 4), (DEF, 4), (MID, 1), (FWD, 0)])
def test_clean_sheets_by_position(element_type, points):
    assert score({"element_type": element_type, "clean_sheets": 1})["clean_sheets"].item() == points


def test_clean_sheet_column_used_as_is():
    """FPL already applied the 60-minute rule to the stat; the scorer does not re-apply it."""
    out = score({"element_type": DEF, "minutes": 30, "clean_sheets": 1})
    assert out["clean_sheets"].item() == 4


@pytest.mark.parametrize(
    ("element_type", "expected"),
    [(GK, [0, 0, -1, -1, -2, -2]), (DEF, [0, 0, -1, -1, -2, -2]), (MID, [0] * 6), (FWD, [0] * 6)],
)
def test_goals_conceded_per_two(element_type, expected):
    out = score(*({"element_type": element_type, "goals_conceded": g} for g in range(6)))
    assert out["goals_conceded"].tolist() == expected


def test_saves_per_three():
    out = score(*({"element_type": GK, "saves": s} for s in (0, 2, 3, 5, 6, 10)))
    assert out["saves"].tolist() == [0, 0, 1, 1, 2, 3]


@pytest.mark.parametrize(
    ("column", "component", "per_unit"),
    [
        ("penalties_saved", "penalties_saved", 5),
        ("penalties_missed", "penalties_missed", -2),
        ("yellow_cards", "yellow_cards", -1),
        ("red_cards", "red_cards", -3),
        ("own_goals", "own_goals", -2),
        ("bonus", "bonus", 1),
    ],
)
@pytest.mark.parametrize("element_type", [GK, DEF, MID, FWD])
def test_flat_components(column, component, per_unit, element_type):
    out = score({"element_type": element_type, column: 2})
    assert out[component].item() == 2 * per_unit


def test_points_is_the_sum_of_components():
    out = score(
        {
            "element_type": MID,
            "goals_scored": 1,
            "assists": 1,
            "clean_sheets": 1,
            "bonus": 2,
            "yellow_cards": 1,
        },
        {"element_type": GK, "saves": 7, "goals_conceded": 3, "penalties_saved": 1, "bonus": 1},
    )
    assert out["points"].tolist() == [2 + 5 + 3 + 1 + 2 - 1, 2 + 2 - 1 + 5 + 1]
    assert (out[list(COMPONENTS)].sum(axis=1) == out["points"]).all()


# --- defensive contributions ---------------------------------------------------------


@pytest.mark.parametrize(
    ("element_type", "count", "points"),
    [
        (DEF, 9, 0),
        (DEF, 10, 2),
        (MID, 11, 0),
        (MID, 12, 2),
        (FWD, 11, 0),
        (FWD, 15, 2),
        (GK, 30, 0),
    ],
)
def test_defcon_threshold_from_fpl_count(element_type, count, points):
    out = score({"element_type": element_type, "defensive_contribution": count})
    assert out["defcon"].item() == points


def test_defcon_off_scores_nothing():
    out = score({"element_type": DEF, "defensive_contribution": 20}, rules=NODEFCON)
    assert out["defcon"].item() == 0
    assert out["points"].item() == 2


def test_defcon_fallback_def_counts_cbit_only():
    out = score(
        {"element_type": DEF, "clearances_blocks_interceptions": 6, "tackles": 4, "recoveries": 0},
        {"element_type": DEF, "clearances_blocks_interceptions": 6, "tackles": 3, "recoveries": 9},
    )
    assert out["defcon"].tolist() == [2, 0]


def test_defcon_fallback_mid_fwd_count_recoveries():
    out = score(
        {"element_type": MID, "clearances_blocks_interceptions": 5, "tackles": 3, "recoveries": 4},
        {"element_type": FWD, "clearances_blocks_interceptions": 5, "tackles": 3, "recoveries": 3},
    )
    assert out["defcon"].tolist() == [2, 0]


def test_defcon_fpl_count_wins_over_components():
    stats = {"clearances_blocks_interceptions": 10, "tackles": 5, "recoveries": 5}
    out = score(
        {"element_type": MID, "defensive_contribution": 4, **stats},
        {"element_type": MID, "defensive_contribution": None, **stats},
    )
    assert out["defcon"].tolist() == [0, 2]


def test_defcon_missing_component_scores_zero():
    out = score(
        {
            "element_type": MID,
            "clearances_blocks_interceptions": 20,
            "tackles": 5,
            "recoveries": None,
        },
        {"element_type": DEF, "clearances_blocks_interceptions": None, "tackles": 20},
    )
    assert out["defcon"].tolist() == [0, 0]


def test_defcon_without_any_defensive_columns_scores_zero():
    assert score({"element_type": DEF})["defcon"].item() == 0


def test_defcon_never_for_goalkeepers_via_components():
    out = score({"element_type": GK, "clearances_blocks_interceptions": 30, "tackles": 10})
    assert out["defcon"].item() == 0


# --- input checks --------------------------------------------------------------------


def test_missing_required_column_raises():
    with pytest.raises(KeyError, match="saves"):
        score_matches(frame({"element_type": GK}).drop(columns="saves"), RULES)


def test_null_required_stat_raises():
    matches = frame({"element_type": GK}).astype({"bonus": "Int64"})
    matches.loc[0, "bonus"] = pd.NA
    with pytest.raises(ValueError, match="bonus"):
        score_matches(matches, RULES)


def test_unknown_position_raises():
    with pytest.raises(ValueError, match="element_type"):
        score({"element_type": 5})


# --- real data -----------------------------------------------------------------------


def real_matches() -> pd.DataFrame:
    """Non-holdout player_match rows with the season's position. The holdout season is
    dropped before anything else is done with the rows (never scored or inspected)."""
    if not (DATA_DIR / "player_match.parquet").exists():
        pytest.skip("data/ not built (run `uv run fplopt build all`)")
    matches = pd.read_parquet(DATA_DIR / "player_match.parquet")
    matches = matches[~matches["season"].isin(HOLDOUT_SEASONS)]
    positions = pd.read_parquet(
        DATA_DIR / "player_season.parquet", columns=["player_key", "season", "element_type"]
    )
    positions = positions[~positions["season"].isin(HOLDOUT_SEASONS)]
    return matches.merge(positions, on=["player_key", "season"], how="left", validate="m:1")


@pytest.mark.realdata
def test_scorer_reproduces_fpl_total_points_on_the_real_data():
    """legacy_rules on 2016/17-2024/25 and the native 2026/27 rules (defcon on) reproduce
    FPL's archived `total_points` row by row. Measured 2026-10-06: 100.000% in every season
    (2016-2024 vaastav rows, 2026 GW1-5 from our own archive, 164 defcon awards); dropping
    `defensive_contribution` and counting from CBI/tackles/recoveries also gives 100% on
    2026. No 2024/25 manager rows (element_type 5) reach player_match."""
    matches = real_matches()
    assert matches["element_type"].notna().all()
    assert set(matches["element_type"]) <= {1, 2, 3, 4}
    rates = {}
    for season, rows in matches.groupby("season"):
        rules = legacy_rules() if season < 2025 else backtest_rules(season)
        points = score_matches(rows, rules)["points"]
        rates[season] = (points == rows["total_points"]).mean()
    print("scorer match rate per season:", {s: f"{r:.5f}" for s, r in rates.items()})
    assert set(range(2016, 2025)) <= set(rates)
    assert 2026 in rates
    assert not set(rates) & HOLDOUT_SEASONS
    low = {season: rate for season, rate in rates.items() if rate < 0.99}
    assert not low, low
