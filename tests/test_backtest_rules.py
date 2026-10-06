import dataclasses
import json
import shutil

import pytest

from fplopt.backtest.rules import (
    DEFAULT_CONFIG_DIR,
    ChipWindow,
    backtest_rules,
    legacy_rules,
    load_rules,
)


def test_default_config_dir_is_the_repo_config():
    assert (DEFAULT_CONFIG_DIR / "2026-27.json").exists()
    assert (DEFAULT_CONFIG_DIR / "2026-27.supplement.json").exists()
    assert (DEFAULT_CONFIG_DIR / "2025-26.supplement.json").exists()


def test_2026_27_scoring():
    rules = load_rules("2026-27")
    assert rules.label == "2026-27"
    assert dict(rules.goals_scored) == {1: 10, 2: 6, 3: 5, 4: 4}
    assert dict(rules.clean_sheets) == {1: 4, 2: 4, 3: 1, 4: 0}
    assert dict(rules.goals_conceded) == {1: -1, 2: -1, 3: 0, 4: 0}
    assert dict(rules.defensive_contribution) == {1: 0, 2: 2, 3: 2, 4: 2}
    assert dict(rules.defcon_threshold) == {1: None, 2: 10, 3: 12, 4: 12}
    assert rules.defcon_enabled
    assert (rules.assists, rules.saves, rules.penalties_saved, rules.penalties_missed) == (
        3,
        1,
        5,
        -2,
    )
    assert (rules.yellow_cards, rules.red_cards, rules.own_goals, rules.bonus) == (-1, -3, -2, 1)
    assert (rules.short_play, rules.long_play, rules.long_play_minutes) == (1, 2, 60)
    assert (rules.saves_per_point, rules.goals_conceded_per_point) == (3, 2)


def test_2026_27_squad_transfers_and_chips():
    rules = load_rules("2026-27")
    assert rules.squad_size == 15
    assert dict(rules.squad_select) == {1: 2, 2: 5, 3: 5, 4: 3}
    assert dict(rules.play_min) == {1: 1, 2: 3, 3: 2, 4: 1}
    assert dict(rules.play_max) == {1: 1, 2: 5, 3: 5, 4: 3}
    assert (rules.squad_play, rules.team_limit, rules.budget) == (11, 3, 1000)
    assert rules.sell_on_fee == 0.5
    assert rules.max_free_transfers == 5
    assert rules.hit_cost == 4
    assert rules.chip_week_ft == "retain"
    assert rules.freehit_consecutive is False
    assert rules.ft_topups == ()
    windows = {(c.name, c.start, c.stop) for c in rules.chips}
    assert windows == {
        ("wildcard", 2, 19),
        ("freehit", 2, 19),
        ("bboost", 1, 19),
        ("3xc", 1, 19),
        ("wildcard", 20, 38),
        ("freehit", 20, 38),
        ("bboost", 20, 38),
        ("3xc", 20, 38),
    }
    assert len({c.chip_id for c in rules.chips}) == 8


def test_2025_26_native_rules_have_defcon():
    rules = load_rules("2025-26")
    assert rules.label == "2025-26"
    assert rules.defcon_enabled
    assert dict(rules.goals_scored) == {1: 10, 2: 6, 3: 5, 4: 4}


def test_defcon_switch():
    off = load_rules("2026-27", defcon=False)
    assert off.label == "2026-27-nodefcon"
    assert not off.defcon_enabled
    assert dict(off.defensive_contribution) == {1: 0, 2: 2, 3: 2, 4: 2}  # award kept, unused
    forced = load_rules("2026-27", defcon=True)
    assert forced.label == "2026-27"
    assert forced.defcon_enabled


@pytest.mark.parametrize("season", range(2016, 2025))
def test_backtest_rules_for_develop_and_validate_seasons(season):
    rules = backtest_rules(season)
    assert rules.label == "2026-27-nodefcon"
    assert not rules.defcon_enabled
    assert rules.goals_scored[1] == 10


def test_backtest_rules_native_seasons():
    assert backtest_rules(2025).label == "2025-26"
    assert backtest_rules(2025).defcon_enabled
    assert backtest_rules(2026).label == "2026-27"
    assert backtest_rules(2026).defcon_enabled


@pytest.mark.parametrize("season", [2015, 2027, 1999])
def test_backtest_rules_unknown_season(season):
    with pytest.raises(ValueError, match="no backtest rules"):
        backtest_rules(season)


def test_legacy_rules():
    legacy, current = legacy_rules(), load_rules("2026-27", defcon=False)
    assert legacy.label == "legacy"
    assert dict(legacy.goals_scored) == {1: 6, 2: 6, 3: 5, 4: 4}
    assert not legacy.defcon_enabled
    same = dataclasses.replace(legacy, label=current.label, goals_scored=current.goals_scored)
    assert same == current


def test_rules_are_read_only():
    rules = load_rules("2026-27")
    with pytest.raises(dataclasses.FrozenInstanceError):
        rules.hit_cost = 0
    with pytest.raises(TypeError):
        rules.goals_scored[1] = 0
    with pytest.raises(TypeError):
        rules.squad_select[1] = 3
    assert isinstance(rules.chips, tuple)


def test_chip_window_contains():
    window = ChipWindow(chip_id=1, name="wildcard", start=2, stop=19)
    assert [window.contains(gw) for gw in (1, 2, 19, 20)] == [False, True, True, False]


# --- failing loudly on a broken config -----------------------------------------------


@pytest.fixture
def config_dir(tmp_path):
    for name in ("2026-27.json", "2026-27.supplement.json"):
        shutil.copy(DEFAULT_CONFIG_DIR / name, tmp_path / name)
    return tmp_path


def edit(path, change):
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_copied_config_loads(config_dir):
    assert load_rules("2026-27", config_dir=config_dir) == load_rules("2026-27")


@pytest.mark.parametrize(
    ("file", "change", "message"),
    [
        ("2026-27.json", lambda d: d["scoring"].pop("assists"), "'assists'"),
        ("2026-27.json", lambda d: d["scoring"]["goals_scored"].pop("MID"), "'MID'"),
        ("2026-27.json", lambda d: d["rules"].pop("max_extra_free_transfers"), "max_extra"),
        ("2026-27.json", lambda d: d["chips"][0].pop("stop_event"), "'stop_event'"),
        ("2026-27.json", lambda d: d["element_types"][1].pop("squad_min_play"), "squad_min"),
        ("2026-27.supplement.json", lambda d: d["scoring"].pop("saves_per_point"), "saves_per"),
        (
            "2026-27.supplement.json",
            lambda d: d["scoring"]["defcon_threshold"].pop("GKP"),
            "'GKP'",
        ),
        ("2026-27.supplement.json", lambda d: d["transfers"].pop("hit_cost"), "'hit_cost'"),
        ("2026-27.supplement.json", lambda d: d["transfers"].pop("chip_week_ft"), "chip_week"),
        ("2026-27.supplement.json", lambda d: d.pop("chips"), "'chips'"),
    ],
)
def test_missing_key_raises(config_dir, file, change, message):
    edit(config_dir / file, change)
    with pytest.raises(KeyError, match=message):
        load_rules("2026-27", config_dir=config_dir)


@pytest.mark.parametrize(
    ("file", "change", "message"),
    [
        ("2026-27.supplement.json", lambda d: d["transfers"].update(chip_week_ft="x"), "chip_week"),
        ("2026-27.supplement.json", lambda d: d.update(season="2025-26"), "season"),
        ("2026-27.supplement.json", lambda d: d["transfers"].update(hit_cost="4"), "integer"),
        ("2026-27.json", lambda d: d["chips"][0].update(name="assistant"), "unknown chip"),
        ("2026-27.json", lambda d: d["element_types"][0].update(singular_name_short="GK"), "GK"),
    ],
)
def test_malformed_value_raises(config_dir, file, change, message):
    edit(config_dir / file, change)
    with pytest.raises(ValueError, match=message):
        load_rules("2026-27", config_dir=config_dir)


def test_event_overrides_are_refused(config_dir):
    edit(config_dir / "2026-27.json", lambda d: d.update(event_overrides={"5": {"x": 1}}))
    with pytest.raises(NotImplementedError, match="event overrides"):
        load_rules("2026-27", config_dir=config_dir)


def test_missing_supplement_file_raises(config_dir):
    (config_dir / "2026-27.supplement.json").unlink()
    with pytest.raises(FileNotFoundError, match="supplement"):
        load_rules("2026-27", config_dir=config_dir)


def test_ft_topups_are_read(config_dir):
    edit(
        config_dir / "2026-27.supplement.json",
        lambda d: d["transfers"].update(ft_topups=[{"gw_index": 16, "amount": 5}]),
    )
    assert load_rules("2026-27", config_dir=config_dir).ft_topups == ((16, 5),)
