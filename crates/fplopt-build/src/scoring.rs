//! Game rules and the per-match points scorer. Port of `fplopt.backtest.rules` (loading,
//! `backtest_rules`) and `fplopt.backtest.scoring` (`score_matches`): `player_match`'s
//! `rescored_points` are every match re-scored under the backtest rules of its season.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use fplopt_core::seasons::season_label;
use serde_json::Value;

pub const POSITIONS: [(&str, i64); 4] = [("GKP", 1), ("DEF", 2), ("MID", 3), ("FWD", 4)];
const CHIP_NAMES: [&str; 4] = ["wildcard", "freehit", "bboost", "3xc"];
const CHIP_WEEK_FT: [&str; 2] = ["retain_plus_one", "retain"];
pub const CURRENT_RULES: &str = "2026-27";
pub const DEVELOP_VALIDATE_SEASONS: std::ops::Range<i64> = 2016..2025;
pub const NATIVE_SEASONS: [i64; 2] = [2025, 2026];

#[derive(Clone, Debug, PartialEq)]
pub struct ChipWindow {
    pub chip_id: i64,
    pub name: String,
    pub start: i64,
    pub stop: i64,
}

/// Scoring, squad, transfer and chip rules of one season (per-position maps keyed by
/// element_type 1-4; money in tenths of £m).
#[derive(Clone, Debug, PartialEq)]
pub struct Rules {
    pub label: String,
    pub goals_scored: BTreeMap<i64, i64>,
    pub clean_sheets: BTreeMap<i64, i64>,
    pub goals_conceded: BTreeMap<i64, i64>,
    pub defensive_contribution: BTreeMap<i64, i64>,
    pub defcon_threshold: BTreeMap<i64, Option<i64>>,
    pub assists: i64,
    pub saves: i64,
    pub penalties_saved: i64,
    pub penalties_missed: i64,
    pub yellow_cards: i64,
    pub red_cards: i64,
    pub own_goals: i64,
    pub bonus: i64,
    pub short_play: i64,
    pub long_play: i64,
    pub long_play_minutes: i64,
    pub saves_per_point: i64,
    pub goals_conceded_per_point: i64,
    pub defcon_enabled: bool,
    pub squad_size: i64,
    pub squad_select: BTreeMap<i64, i64>,
    pub play_min: BTreeMap<i64, i64>,
    pub play_max: BTreeMap<i64, i64>,
    pub squad_play: i64,
    pub team_limit: i64,
    pub budget: i64,
    pub sell_on_fee: f64,
    pub max_free_transfers: i64,
    pub hit_cost: i64,
    pub chips: Vec<ChipWindow>,
    pub chip_week_ft: String,
    pub freehit_consecutive: bool,
    pub ft_topups: Vec<(i64, i64)>,
}

/// The repo's `config/scoring` (the binary runs from the repo root, like the Python CLI).
pub fn default_config_dir() -> PathBuf {
    PathBuf::from("config").join("scoring")
}

fn get<'a>(map: &'a Value, key: &str, where_: &str) -> Result<&'a Value> {
    match map.as_object().and_then(|m| m.get(key)) {
        Some(v) => Ok(v),
        None => bail!("{where_}: missing key '{key}'"),
    }
}

fn int(value: &Value, where_: &str) -> Result<i64> {
    match value {
        Value::Number(n) if n.is_i64() => Ok(n.as_i64().unwrap()),
        other => bail!("{where_}: expected an integer, got {other}"),
    }
}

fn per_position<'a>(map: &'a Value, where_: &str) -> Result<BTreeMap<i64, &'a Value>> {
    POSITIONS.iter().map(|(name, et)| Ok((*et, get(map, name, where_)?))).collect()
}

fn read(path: &Path) -> Result<Value> {
    if !path.exists() {
        bail!("rules config not found: {}", path.display());
    }
    Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
}

/// Rules from `<label>.json` + `<label>.supplement.json`. `defcon` None keeps the season's
/// own rule, `Some(false)` switches it off (label suffixed `-nodefcon`).
pub fn load_rules(label: &str, defcon: Option<bool>, config_dir: &Path) -> Result<Rules> {
    let main_path = config_dir.join(format!("{label}.json"));
    let supp_path = config_dir.join(format!("{label}.supplement.json"));
    let (main, supp) = (read(&main_path)?, read(&supp_path)?);
    let m = main_path.file_name().unwrap().to_string_lossy().into_owned();
    let s = supp_path.file_name().unwrap().to_string_lossy().into_owned();
    for (name, config) in [(&m, &main), (&s, &supp)] {
        let season = get(config, "season", name)?;
        if season.as_str() != Some(label) {
            bail!("{name}: season {season} is not '{label}'");
        }
    }
    let overrides = get(&main, "event_overrides", &m)?;
    if crate::common::json_bool(overrides) {
        bail!("{m}: per-GW event overrides are not supported");
    }
    let scoring = get(&main, "scoring", &m)?;
    let rules = get(&main, "rules", &m)?;
    let supp_scoring = get(&supp, "scoring", &s)?;
    let transfers = get(&supp, "transfers", &s)?;
    let supp_chips = get(&supp, "chips", &s)?;
    let score = |key: &str| -> Result<i64> {
        int(get(scoring, key, &format!("{m} scoring"))?, &format!("{m} scoring.{key}"))
    };
    let per_pos = |key: &str| -> Result<BTreeMap<i64, i64>> {
        let where_ = format!("{m} scoring.{key}");
        per_position(get(scoring, key, &format!("{m} scoring"))?, &where_)?
            .into_iter()
            .map(|(et, v)| Ok((et, int(v, &where_)?)))
            .collect()
    };
    let thresholds: BTreeMap<i64, Option<i64>> = per_position(
        get(supp_scoring, "defcon_threshold", &format!("{s} scoring"))?,
        &format!("{s} scoring.defcon_threshold"),
    )?
    .into_iter()
    .map(|(et, v)| {
        Ok((
            et,
            if v.is_null() {
                None
            } else {
                Some(int(v, &format!("{s} scoring.defcon_threshold[{et}]"))?)
            },
        ))
    })
    .collect::<Result<_>>()?;
    let defcon_points = per_pos("defensive_contribution")?;

    let mut element_types: BTreeMap<i64, &Value> = BTreeMap::new();
    for item in
        get(&main, "element_types", &m)?.as_array().context("element_types is not a list")?
    {
        let short = get(item, "singular_name_short", &format!("{m} element_types"))?;
        let et =
            int(get(item, "id", &format!("{m} element_types"))?, &format!("{m} element_types.id"))?;
        let expected = POSITIONS.iter().find(|(n, _)| Some(*n) == short.as_str()).map(|(_, e)| *e);
        if expected != Some(et) {
            bail!("{m}: element_type {et} is {short}, expected GKP/DEF/MID/FWD = 1-4");
        }
        element_types.insert(et, item);
    }
    if element_types.keys().copied().collect::<Vec<_>>() != [1, 2, 3, 4] {
        bail!("{m}: element_types {:?} are not 1-4", element_types.keys().collect::<Vec<_>>());
    }
    let squad = |key: &str| -> Result<BTreeMap<i64, i64>> {
        let where_ = format!("{m} element_types");
        element_types
            .iter()
            .map(|(et, item)| {
                Ok((*et, int(get(item, key, &where_)?, &format!("{where_}[{et}].{key}"))?))
            })
            .collect()
    };
    let mut chips = Vec::new();
    for item in get(&main, "chips", &m)?.as_array().context("chips is not a list")? {
        let where_ = format!("{m} chips");
        let name = get(item, "name", &where_)?.as_str().unwrap_or_default().to_string();
        if !CHIP_NAMES.contains(&name.as_str()) {
            bail!("{where_}: unknown chip '{name}'");
        }
        chips.push(ChipWindow {
            chip_id: int(get(item, "id", &where_)?, &format!("{where_}.id"))?,
            name,
            start: int(get(item, "start_event", &where_)?, &format!("{where_}.start_event"))?,
            stop: int(get(item, "stop_event", &where_)?, &format!("{where_}.stop_event"))?,
        });
    }
    let ids: std::collections::HashSet<i64> = chips.iter().map(|c| c.chip_id).collect();
    if ids.len() != chips.len() {
        bail!("{m}: duplicate chip ids");
    }
    let chip_week_ft = get(transfers, "chip_week_ft", &format!("{s} transfers"))?
        .as_str()
        .unwrap_or_default()
        .to_string();
    if !CHIP_WEEK_FT.contains(&chip_week_ft.as_str()) {
        bail!("{s}: chip_week_ft '{chip_week_ft}' not in ['retain', 'retain_plus_one']");
    }
    let freehit_consecutive = match get(supp_chips, "freehit_consecutive", &format!("{s} chips"))? {
        Value::Bool(b) => *b,
        _ => bail!("{s}: freehit_consecutive must be true/false"),
    };
    let mut ft_topups = Vec::new();
    for item in
        get(transfers, "ft_topups", &format!("{s} transfers"))?.as_array().context("ft_topups")?
    {
        ft_topups.push((
            int(
                get(item, "gw_index", &format!("{s} ft_topups"))?,
                &format!("{s} ft_topups.gw_index"),
            )?,
            int(get(item, "amount", &format!("{s} ft_topups"))?, &format!("{s} ft_topups.amount"))?,
        ));
    }
    let supp_int = |key: &str| -> Result<i64> {
        int(get(supp_scoring, key, &format!("{s} scoring"))?, &format!("{s} {key}"))
    };
    let rule = |key: &str| get(rules, key, &format!("{m} rules"));
    let enabled = defcon.unwrap_or_else(|| defcon_points.values().any(|&v| v != 0));
    Ok(Rules {
        label: if defcon == Some(false) { format!("{label}-nodefcon") } else { label.to_string() },
        goals_scored: per_pos("goals_scored")?,
        clean_sheets: per_pos("clean_sheets")?,
        goals_conceded: per_pos("goals_conceded")?,
        defensive_contribution: defcon_points,
        defcon_threshold: thresholds,
        assists: score("assists")?,
        saves: score("saves")?,
        penalties_saved: score("penalties_saved")?,
        penalties_missed: score("penalties_missed")?,
        yellow_cards: score("yellow_cards")?,
        red_cards: score("red_cards")?,
        own_goals: score("own_goals")?,
        bonus: score("bonus")?,
        short_play: score("short_play")?,
        long_play: score("long_play")?,
        long_play_minutes: supp_int("long_play_minutes")?,
        saves_per_point: supp_int("saves_per_point")?,
        goals_conceded_per_point: supp_int("goals_conceded_per_point")?,
        defcon_enabled: enabled,
        squad_size: int(rule("squad_squadsize")?, &format!("{m} rules.squad_squadsize"))?,
        squad_select: squad("squad_select")?,
        play_min: squad("squad_min_play")?,
        play_max: squad("squad_max_play")?,
        squad_play: int(rule("squad_squadplay")?, &format!("{m} rules.squad_squadplay"))?,
        team_limit: int(rule("squad_team_limit")?, &format!("{m} rules.squad_team_limit"))?,
        budget: int(rule("squad_total_spend")?, &format!("{m} rules.squad_total_spend"))?,
        sell_on_fee: crate::common::json_f64(
            rule("transfers_sell_on_fee")?,
            "transfers_sell_on_fee",
        )?,
        max_free_transfers: 1 + int(
            rule("max_extra_free_transfers")?,
            &format!("{m} rules.max_extra_free_transfers"),
        )?,
        hit_cost: int(
            get(transfers, "hit_cost", &format!("{s} transfers"))?,
            &format!("{s} hit_cost"),
        )?,
        chips,
        chip_week_ft,
        freehit_consecutive,
        ft_topups,
    })
}

/// The rules a season is scored under in backtests (PLAN §5): 2016/17-2024/25 the current
/// rules without defcon, 2025/26 and 2026/27 their own.
pub fn backtest_rules(season: i64, config_dir: &Path) -> Result<Rules> {
    if DEVELOP_VALIDATE_SEASONS.contains(&season) {
        return load_rules(CURRENT_RULES, Some(false), config_dir);
    }
    if NATIVE_SEASONS.contains(&season) {
        return load_rules(&season_label(season as i32), None, config_dir);
    }
    bail!("no backtest rules for season {}", season_label(season as i32))
}

/// One player-match stat line (`score_matches` input).
#[derive(Clone, Debug, Default)]
pub struct StatLine {
    pub element_type: i64,
    pub minutes: i64,
    pub goals_scored: i64,
    pub assists: i64,
    pub clean_sheets: i64,
    pub goals_conceded: i64,
    pub own_goals: i64,
    pub penalties_saved: i64,
    pub penalties_missed: i64,
    pub yellow_cards: i64,
    pub red_cards: i64,
    pub saves: i64,
    pub bonus: i64,
    pub defensive_contribution: Option<i64>,
    pub clearances_blocks_interceptions: Option<i64>,
    pub tackles: Option<i64>,
    pub recoveries: Option<i64>,
}

/// Defensive-contribution count: FPL's when present, else the position's stats summed (a
/// missing component makes it missing); GKs have none.
fn defcon_count(m: &StatLine) -> Option<i64> {
    if m.defensive_contribution.is_some() {
        return m.defensive_contribution;
    }
    let cbit = || Some(m.clearances_blocks_interceptions? + m.tackles?);
    match m.element_type {
        2 => cbit(),
        3 | 4 => Some(cbit()? + m.recoveries?),
        _ => None,
    }
}

/// FPL points of one match under `rules` (the sum of `score_matches`' components).
pub fn score(m: &StatLine, rules: &Rules) -> Result<i64> {
    let et = m.element_type;
    if !(1..=4).contains(&et) {
        bail!("score_matches: element_type must be 1-4, got {et}");
    }
    let appearance = if m.minutes >= rules.long_play_minutes {
        rules.long_play
    } else if m.minutes > 0 {
        rules.short_play
    } else {
        0
    };
    let defcon = if rules.defcon_enabled {
        match (defcon_count(m), rules.defcon_threshold[&et]) {
            (Some(count), Some(threshold)) if count >= threshold => {
                rules.defensive_contribution[&et]
            }
            _ => 0,
        }
    } else {
        0
    };
    Ok(appearance
        + m.goals_scored * rules.goals_scored[&et]
        + m.assists * rules.assists
        + m.clean_sheets * rules.clean_sheets[&et]
        + m.goals_conceded.div_euclid(rules.goals_conceded_per_point) * rules.goals_conceded[&et]
        + m.saves.div_euclid(rules.saves_per_point) * rules.saves
        + m.penalties_saved * rules.penalties_saved
        + m.penalties_missed * rules.penalties_missed
        + m.yellow_cards * rules.yellow_cards
        + m.red_cards * rules.red_cards
        + m.own_goals * rules.own_goals
        + m.bonus * rules.bonus
        + defcon)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repo_config() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config/scoring")
    }

    #[test]
    fn current_rules_load() {
        let rules = backtest_rules(2020, &repo_config()).unwrap();
        assert_eq!(rules.label, "2026-27-nodefcon");
        assert!(!rules.defcon_enabled);
        let native = backtest_rules(2026, &repo_config()).unwrap();
        assert!(native.defcon_enabled);
        assert_eq!(native.defcon_threshold[&1], None);
        assert!(backtest_rules(2027, &repo_config()).is_err());
    }

    #[test]
    fn defcon_and_floors() {
        let rules = backtest_rules(2026, &repo_config()).unwrap();
        let line = StatLine {
            element_type: 2,
            minutes: 90,
            clean_sheets: 1,
            goals_conceded: 3,
            clearances_blocks_interceptions: Some(7),
            tackles: Some(3),
            ..Default::default()
        };
        // 2 (90') + 4 (CS) - 1 (3 conceded) + 2 (CBIT 10)
        assert_eq!(
            score(&line, &rules).unwrap(),
            2 + rules.clean_sheets[&2] - 1 + rules.defensive_contribution[&2]
        );
        let no_count = StatLine { tackles: None, ..line };
        assert_eq!(score(&no_count, &rules).unwrap(), 2 + rules.clean_sheets[&2] - 1);
    }
}
