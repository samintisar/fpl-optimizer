import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fplopt.backtest.rules import backtest_rules, load_rules
from fplopt.backtest.scoring import REQUIRED, score_matches
from fplopt.backtest.xg_points import (
    poisson_expected_floor,
    poisson_pmf,
    team_xg_table,
    xg_score_matches,
)
from fplopt.seasons import HOLDOUT_SEASONS

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RULES = load_rules("2026-27", defcon=False)  # GK goals 10, DEF 6, MID 5, FWD 4
DEFCON = load_rules("2026-27")
GK, DEF, MID, FWD = 1, 2, 3, 4
XG_COLUMNS = ("us_xg", "us_xa", "fpl_xg", "fpl_xa")


def frame(*rows: dict) -> pd.DataFrame:
    """Player-match rows of team 10 against team 20 in fixture 1: stats 0, minutes 90, no
    xG unless given."""
    base = dict.fromkeys(REQUIRED, 0) | {"minutes": 90, "fixture_key": 1, "team_key": 10}
    base |= {"opponent_team_key": 20} | dict.fromkeys(XG_COLUMNS)
    out = pd.DataFrame([base | row for row in rows])
    return out.astype(dict.fromkeys(XG_COLUMNS, "Float64"))


def teams(*rows: dict) -> pd.DataFrame:
    """team_match rows; every xG source null unless given."""
    base = {"fixture_key": 1} | dict.fromkeys(("us_xg", "fpl_xg", "fd_xg"))
    columns = ["fixture_key", "team_key", "us_xg", "fpl_xg", "fd_xg"]
    out = pd.DataFrame([base | row for row in rows], columns=columns)
    return out.astype(
        {"fixture_key": "int64", "team_key": "int64"} | dict.fromkeys(columns[2:], "Float64")
    )


OPP_1_2 = team_xg_table(teams({"team_key": 20, "us_xg": 1.2}, {"team_key": 10, "us_xg": 0.4}))


def xg_points(*rows: dict, team_xg=OPP_1_2, rules=RULES) -> pd.DataFrame:
    return xg_score_matches(frame(*rows), team_xg, rules)


def expected_floor_direct(rate: float, k: int, cap: int = 80) -> float:
    return sum((n // k) * math.exp(-rate) * rate**n / math.factorial(n) for n in range(cap))


def expected_half_closed_form(rate: float) -> float:
    """E[floor(N/2)] = (E[N] - P(N odd)) / 2, P(N odd) = (1 - e^{-2 rate}) / 2."""
    return (rate - (1 - math.exp(-2 * rate)) / 2) / 2


# --- Poisson helpers --------------------------------------------------------------------


def test_pmf_sums_to_one_and_rate_zero_is_a_point_mass():
    pmf = poisson_pmf(np.array([0.0, 0.7, 3.5]))
    assert pmf.sum(axis=1) == pytest.approx([1, 1, 1], abs=1e-12)
    assert pmf[0, 0] == 1.0
    assert pmf[0, 1:].sum() == 0.0
    assert pmf[1, 2] == pytest.approx(math.exp(-0.7) * 0.7**2 / 2)


@pytest.mark.parametrize("rate", [0.0, 0.3, 1.0, 1.2 * 75 / 90, 2.5, 6.0])
def test_expected_floor_half_matches_closed_form_and_direct_sum(rate):
    value = poisson_expected_floor(np.array([rate]), 2)[0]
    assert value == pytest.approx(expected_half_closed_form(rate), abs=1e-12)
    assert value == pytest.approx(expected_floor_direct(rate, 2), abs=1e-12)


@pytest.mark.parametrize("k", [1, 3])
def test_expected_floor_other_divisors(k):
    rates = np.array([0.5, 1.7, 4.0])
    values = poisson_expected_floor(rates, k)
    assert values == pytest.approx([expected_floor_direct(r, k) for r in rates], abs=1e-12)
    if k == 1:
        assert values == pytest.approx(rates, abs=1e-12)


# --- hand-computed rows -----------------------------------------------------------------


def test_defender_75_minutes():
    """λ = 1.2 × 75/90 = 1.0: 2 (appearance) + 0.3 × 6 + 0.1 × 3 + 4 e^-1 - E[floor(N/2)]."""
    out = xg_points({"element_type": DEF, "minutes": 75, "us_xg": 0.3, "us_xa": 0.1})
    rate = 1.2 * 75 / 90
    expected = 2 + 1.8 + 0.3 + 4 * math.exp(-rate) - expected_half_closed_form(rate)
    assert out["xg_points"].item() == pytest.approx(expected, abs=1e-12)
    assert out["xg_source"].item() == "understat"
    assert out["opp_xg_source"].item() == "understat"


def test_goalkeeper_90_minutes_keeps_realized_saves_and_ignores_realized_outcomes():
    """Saves are realized; the realized goals, clean sheet and goals conceded are not."""
    out = xg_points(
        {
            "element_type": GK,
            "saves": 4,
            "goals_conceded": 3,
            "clean_sheets": 0,
            "fpl_xg": 0.0,
            "fpl_xa": 0.02,
        }
    )
    expected = 2 + 1 + 0.02 * 3 + 4 * math.exp(-1.2) - expected_half_closed_form(1.2)
    assert out["xg_points"].item() == pytest.approx(expected, abs=1e-12)
    assert out["xg_source"].item() == "fpl"


def test_midfielder_90_minutes_clean_sheet_point_no_goals_conceded():
    out = xg_points(
        {"element_type": MID, "goals_scored": 2, "assists": 1, "us_xg": 0.5, "us_xa": 0.25}
    )
    expected = 2 + 0.5 * 5 + 0.25 * 3 + 1 * math.exp(-1.2)
    assert out["xg_points"].item() == pytest.approx(expected, abs=1e-12)


def test_forward_needs_no_team_xg():
    out = xg_points(
        {"element_type": FWD, "us_xg": 0.8, "us_xa": 0.1}, team_xg=team_xg_table(teams())
    )
    assert out["xg_points"].item() == pytest.approx(2 + 0.8 * 4 + 0.3, abs=1e-12)
    assert pd.isna(out["opp_xg_source"].item())


def test_goalkeeper_goal_is_ten_points():
    out = xg_points({"element_type": GK, "us_xg": 0.1, "us_xa": 0.0})
    expected = 2 + 0.1 * 10 + 4 * math.exp(-1.2) - expected_half_closed_form(1.2)
    assert out["xg_points"].item() == pytest.approx(expected, abs=1e-12)


def test_under_60_minutes_no_clean_sheet_but_goals_conceded_applies():
    out = xg_points(
        {"element_type": DEF, "minutes": 45, "us_xg": 0.0, "us_xa": 0.0},
        {"element_type": GK, "minutes": 59, "us_xg": 0.0, "us_xa": 0.0},
        {"element_type": MID, "minutes": 45, "us_xg": 0.0, "us_xa": 0.0},
    )
    assert out["xg_points"].tolist() == pytest.approx(
        [
            1 - expected_half_closed_form(1.2 * 45 / 90),
            1 - expected_half_closed_form(1.2 * 59 / 90),
            1,
        ],
        abs=1e-12,
    )
    assert out["opp_xg_source"].tolist()[:2] == ["understat", "understat"]
    assert pd.isna(out["opp_xg_source"].iloc[2])


def test_midfielder_under_60_needs_no_team_xg():
    out = xg_points(
        {"element_type": MID, "minutes": 45, "us_xg": 0.2, "us_xa": 0.0},
        team_xg=team_xg_table(teams()),
    )
    assert out["xg_points"].item() == pytest.approx(1 + 0.2 * 5, abs=1e-12)


def test_opponent_xg_is_the_opponent_side():
    """Team 10 (xG 0.4) faces team 20 (xG 1.2): team 10's defender uses 1.2, team 20's 0.4."""
    out = xg_points(
        {"element_type": DEF, "us_xg": 0.0, "us_xa": 0.0},
        {"element_type": DEF, "us_xg": 0.0, "us_xa": 0.0, "team_key": 20, "opponent_team_key": 10},
    )
    assert out["xg_points"].tolist() == pytest.approx(
        [2 + 4 * math.exp(-r) - expected_half_closed_form(r) for r in (1.2, 0.4)], abs=1e-12
    )


def test_realized_components_are_kept():
    """Appearance, saves, pens, cards, own goals, bonus and defcon come from score_matches."""
    row = {
        "element_type": DEF,
        "minutes": 30,
        "yellow_cards": 1,
        "red_cards": 1,
        "own_goals": 1,
        "penalties_missed": 1,
        "bonus": 2,
        "defensive_contribution": 10,
        "us_xg": 0.0,
        "us_xa": 0.0,
    }
    realized = score_matches(frame(row), DEFCON).iloc[0]
    expected = (
        realized[["appearance", "yellow_cards", "red_cards", "own_goals", "penalties_missed"]].sum()
        + realized[["bonus", "defcon"]].sum()
    )
    assert realized["defcon"] == 2
    gc = -expected_half_closed_form(1.2 * 30 / 90)
    out = xg_points(row, rules=DEFCON)
    assert out["xg_points"].item() == pytest.approx(expected + gc, abs=1e-12)


def test_penalties_saved_realized():
    out = xg_points({"element_type": GK, "penalties_saved": 1, "us_xg": 0.0, "us_xa": 0.0})
    expected = 2 + 5 + 4 * math.exp(-1.2) - expected_half_closed_form(1.2)
    assert out["xg_points"].item() == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("element_type", [GK, DEF, MID, FWD])
def test_zero_minutes_scores_realized_points_without_any_xg(element_type):
    row = {"element_type": element_type, "minutes": 0, "yellow_cards": 1}
    out = xg_points(row, team_xg=team_xg_table(teams()))
    assert out["xg_points"].item() == score_matches(frame(row), RULES)["points"].item() == -1
    assert pd.isna(out["xg_source"].item())
    assert pd.isna(out["opp_xg_source"].item())


# --- null propagation -------------------------------------------------------------------


def test_null_when_player_xg_missing():
    out = xg_points(
        {"element_type": FWD},
        {"element_type": MID, "minutes": 10, "us_xa": 0.1},
        {"element_type": DEF, "minutes": 90, "us_xg": 0.1, "us_xa": 0.0},
    )
    assert out["xg_points"].isna().tolist() == [True, True, False]
    assert out["xg_source"].isna().tolist() == [True, True, False]


@pytest.mark.parametrize(
    ("element_type", "minutes", "null"),
    [(GK, 90, True), (DEF, 90, True), (DEF, 1, True), (MID, 60, True), (MID, 59, False)]
    + [(FWD, 90, False)],
)
def test_null_when_needed_opponent_xg_missing(element_type, minutes, null):
    no_opponent = team_xg_table(teams({"team_key": 10, "us_xg": 1.0}, {"team_key": 20}))
    out = xg_points(
        {"element_type": element_type, "minutes": minutes, "us_xg": 0.1, "us_xa": 0.1},
        team_xg=no_opponent,
    )
    assert out["xg_points"].isna().item() is null
    assert pd.isna(out["opp_xg_source"].item())


def test_null_when_fixture_absent_from_team_xg():
    out = xg_points({"element_type": DEF, "us_xg": 0.1, "us_xa": 0.1, "fixture_key": 99})
    assert out["xg_points"].isna().item()


# --- source precedence ------------------------------------------------------------------


def test_player_source_precedence_uses_one_source_for_both():
    """Understat when both us_xg and us_xa exist, else FPL when both exist; never mixed."""
    out = xg_points(
        {"element_type": FWD, "us_xg": 0.5, "us_xa": 0.2, "fpl_xg": 0.9, "fpl_xa": 0.9},
        {"element_type": FWD, "us_xg": 0.5, "fpl_xg": 0.9, "fpl_xa": 0.1},
        {"element_type": FWD, "fpl_xg": 0.9, "fpl_xa": 0.1},
        {"element_type": FWD, "us_xg": 0.5, "fpl_xa": 0.1},
    )
    assert out["xg_source"].iloc[:3].tolist() == ["understat", "fpl", "fpl"]
    assert pd.isna(out["xg_source"].iloc[3])
    assert out["xg_points"].iloc[:3].tolist() == pytest.approx(
        [2 + 0.5 * 4 + 0.2 * 3, 2 + 0.9 * 4 + 0.1 * 3, 2 + 0.9 * 4 + 0.1 * 3], abs=1e-12
    )
    assert out["xg_points"].isna().iloc[3]


def test_team_xg_table_precedence():
    table = team_xg_table(
        teams(
            {"team_key": 1, "us_xg": 1.1, "fpl_xg": 2.2, "fd_xg": 3.3},
            {"team_key": 2, "fpl_xg": 2.2, "fd_xg": 3.3},
            {"team_key": 3, "fd_xg": 3.3},
            {"team_key": 4},
        )
    )
    assert list(table.columns) == ["fixture_key", "team_key", "xg", "source"]
    assert table["xg"].dtype == "Float64"
    assert table["xg"].iloc[:3].tolist() == [1.1, 2.2, 3.3]
    assert table["source"].iloc[:3].tolist() == ["understat", "fpl", "football-data"]
    assert table["xg"].isna().iloc[3]
    assert pd.isna(table["source"].iloc[3])


def test_opponent_source_reported():
    team_xg = team_xg_table(teams({"team_key": 20, "fd_xg": 1.2}))
    out = xg_points({"element_type": DEF, "us_xg": 0.0, "us_xa": 0.0}, team_xg=team_xg)
    assert out["opp_xg_source"].item() == "football-data"
    expected = 2 + 4 * math.exp(-1.2) - expected_half_closed_form(1.2)
    assert out["xg_points"].item() == pytest.approx(expected, abs=1e-12)


# --- shape and validation ---------------------------------------------------------------


def test_output_index_and_dtypes():
    matches = frame(
        {"element_type": MID, "us_xg": 0.1, "us_xa": 0.1}, {"element_type": DEF, "minutes": 0}
    ).set_axis([7, 3])
    out = xg_score_matches(matches, OPP_1_2, RULES)
    assert list(out.columns) == ["xg_points", "xg_source", "opp_xg_source"]
    assert list(out.index) == [7, 3]
    assert out["xg_points"].dtype == "Float64"
    assert out["xg_source"].dtype == "str"
    assert out["opp_xg_source"].dtype == "str"


def test_missing_xg_column_raises():
    with pytest.raises(KeyError, match="us_xa"):
        xg_score_matches(frame({"element_type": FWD}).drop(columns="us_xa"), OPP_1_2, RULES)


def test_duplicate_team_xg_raises():
    duplicated = pd.concat([OPP_1_2, OPP_1_2])
    with pytest.raises(ValueError, match="duplicate"):
        xg_score_matches(frame({"element_type": FWD}), duplicated, RULES)


# --- real data -----------------------------------------------------------------------


@pytest.mark.realdata
def test_xg_points_coverage_on_the_real_data():
    """Share of non-null xg_points among player-match rows with minutes > 0, and the mean
    of xg_points vs realized points on covered rows (backtest rules). The holdout season is
    dropped before anything else is done with the rows.

    Measured 2026-10-06 (coverage, mean xg_points vs realized):
    2016 0.116 2.44/2.64, 2017 0.153 2.52/2.55, 2018 0.186 2.55/2.55 (no team xG before
    2019/20, so only FWDs and sub-60 MIDs with Understat rows are covered),
    2019 0.889 3.09/3.02, 2020-2023 1.000 (3.07/3.03, 3.06/3.00, 2.86/2.80, 2.84/2.75),
    2024 1.000 2.80/2.71, 2026 1.000 2.96/3.02.
    2019/20 misses the >= 0.95 target: every uncovered row belongs to one of 115 players
    with no Understat career log (the mirror only has players still active in 2021/22+), and
    FPL xG starts in 2022/23, so there is no fallback; it is held to >= 0.85."""
    if not (DATA_DIR / "player_match.parquet").exists():
        pytest.skip("data/ not built (run `uv run fplopt build all`)")
    matches = pd.read_parquet(DATA_DIR / "player_match.parquet")
    matches = matches[~matches["season"].isin(HOLDOUT_SEASONS)]
    positions = pd.read_parquet(
        DATA_DIR / "player_season.parquet", columns=["player_key", "season", "element_type"]
    )
    positions = positions[~positions["season"].isin(HOLDOUT_SEASONS)]
    matches = matches.merge(positions, on=["player_key", "season"], how="left", validate="m:1")
    team_match = pd.read_parquet(DATA_DIR / "team_match.parquet")
    team_xg = team_xg_table(team_match[~team_match["season"].isin(HOLDOUT_SEASONS)])

    report = {}
    for season, rows in matches.groupby("season"):
        rules = backtest_rules(season)
        out = xg_score_matches(rows, team_xg, rules)
        realized = score_matches(rows, rules)["points"]
        played = rows["minutes"] > 0
        covered = played & out["xg_points"].notna()
        report[season] = (
            covered.sum() / played.sum(),
            out.loc[covered, "xg_points"].mean(),
            realized[covered].mean(),
        )
    print("season: coverage, mean xg_points, mean realized points (covered rows)")
    for season, (coverage, xg_mean, points_mean) in report.items():
        print(f"{season}: {coverage:.4f} {xg_mean:.3f} {points_mean:.3f}")

    assert set(range(2016, 2025)) | {2026} <= set(report)
    assert not set(report) & HOLDOUT_SEASONS
    low = {s: report[s][0] for s in range(2020, 2024) if report[s][0] < 0.95}
    assert not low, low
    assert report[2019][0] >= 0.85
    for season in (*range(2019, 2025), 2026):
        _, xg_mean, points_mean = report[season]
        assert abs(xg_mean - points_mean) < 0.25, (season, xg_mean, points_mean)
