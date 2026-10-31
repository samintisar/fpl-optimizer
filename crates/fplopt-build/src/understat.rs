//! Understat (vaastav mirror) -> `understat_player_match`, `understat_map`, the `us_*`
//! columns of `player_match`, `player_dim.understat_id`, and `team_match`. Port of
//! `fplopt.build.understat` (its module docstring describes the matching rules, the
//! validation and the measured results). The candidate/greedy matcher is shared with
//! the ESPN map (`id` = the source's id column).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result, bail};
use chrono::NaiveDateTime;
use fplopt_core::seasons::parse_season_label;

use crate::common::{
    BuildContext, Csv, EPOCH, as_str, days_from_date, file_name, latest_entry, parse_int,
    parse_opt_float, parse_opt_int, subdirs, uk_date,
};
use crate::fixtures::{FIRST_SEASON, Fixture, vaastav_run};
use crate::fuzz::{html_unescape, normalise_name, token_set_ratio};
use crate::players::{
    MAX_MINUTES, PlayerMatch, PlayerSeason, player_dim_from_seasons, validated_player_match_table,
};
use crate::table::{Column, Micros, Table};
use crate::teams::{TeamResolver, UNDERSTAT_AGGREGATE_FILES, understat_files};

pub const MINUTES_TOLERANCE: i64 = 5;
const MIN_AUTO_NAME_SCORE: f64 = 50.0;
const MIN_GOAL_AGREEMENT: f64 = 0.9;
const VALIDATED_SEASONS: std::ops::Range<i64> = 2020..2024;
const OVERRIDES_FILE: &str = "overrides.csv";

pub const UPM_SORT_BY: &[&str] = &["fixture_key", "understat_id"];
pub const MAP_SORT_BY: &[&str] = &["understat_id"];
pub const TEAM_MATCH_SORT_BY: &[&str] = &["fixture_key", "team_key"];

fn folder_season(path: &Path) -> Result<i64> {
    Ok(i64::from(parse_season_label(file_name(path.parent().unwrap().parent().unwrap()))?))
}

fn stem(path: &Path) -> &str {
    file_name(path).trim_end_matches(".csv.gz")
}

// --- understat_player_match ------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Upm {
    pub understat_id: i64,
    pub fixture_key: i64,
    pub season: i64,
    pub us_match_id: i64,
    pub us_minutes: i64,
    pub us_goals: i64,
    pub us_npg: i64,
    pub us_xg: f64,
    pub us_npxg: f64,
    pub us_xa: f64,
    pub us_assists: i64,
    pub us_shots: i64,
    pub us_key_passes: i64,
    pub us_position: String,
    pub understat_name: String,
    pub event_time: Micros,
    pub available_at: Micros,
}

struct PlayerFileRow {
    understat_id: i64,
    folder_season: i64,
    file_name: String,
    season: i64,
    match_id: i64,
    home: String,
    away: String,
    date: String,
    time: i64,
    goals: i64,
    npg: i64,
    xg: f64,
    npxg: f64,
    xa: f64,
    assists: i64,
    shots: i64,
    key_passes: i64,
    position: String,
}

fn read_player_files(run: &Path) -> Result<Vec<PlayerFileRow>> {
    let mut rows = Vec::new();
    for path in understat_files(run)? {
        let stem = stem(&path).to_string();
        if stem.starts_with("understat_") {
            continue;
        }
        let (name, id) = stem.rsplit_once('_').context("player file without an id")?;
        let understat_id: i64 = id.parse()?;
        let folder = folder_season(&path)?;
        let csv = Csv::read(&path)?;
        let cell = |name: &str| csv.cells(name);
        let (season, match_id, h, a, date) =
            (cell("season"), cell("id"), cell("h_team"), cell("a_team"), cell("date"));
        let (time, goals, npg, xg, npxg, xa) =
            (cell("time"), cell("goals"), cell("npg"), cell("xG"), cell("npxG"), cell("xA"));
        let (assists, shots, key_passes, position) =
            (cell("assists"), cell("shots"), cell("key_passes"), cell("position"));
        for i in 0..csv.len() {
            // Only EPL rows are kept; others are parsed lazily below (non-EPL columns may be
            // odd), so keep the raw text and parse when the row is EPL.
            rows.push(PlayerFileRow {
                understat_id,
                folder_season: folder,
                file_name: name.replace('_', " "),
                season: parse_opt_int(season[i], "season")?.unwrap_or(-1),
                match_id: parse_int(match_id[i], "id")?,
                home: as_str(h[i]),
                away: as_str(a[i]),
                date: as_str(date[i]),
                time: parse_opt_int(time[i], "time")?.unwrap_or(0),
                goals: parse_opt_int(goals[i], "goals")?.unwrap_or(0),
                npg: parse_opt_int(npg[i], "npg")?.unwrap_or(0),
                xg: parse_opt_float(xg[i], "xG")?.unwrap_or(f64::NAN),
                npxg: parse_opt_float(npxg[i], "npxG")?.unwrap_or(f64::NAN),
                xa: parse_opt_float(xa[i], "xA")?.unwrap_or(f64::NAN),
                assists: parse_opt_int(assists[i], "assists")?.unwrap_or(0),
                shots: parse_opt_int(shots[i], "shots")?.unwrap_or(0),
                key_passes: parse_opt_int(key_passes[i], "key_passes")?.unwrap_or(0),
                position: as_str(position[i]),
            });
        }
    }
    Ok(rows)
}

/// understat_id -> name from the folders' `understat_player` aggregates (newest wins).
fn read_player_names(run: &Path) -> Result<HashMap<i64, String>> {
    let mut names = HashMap::new();
    for path in understat_files(run)? {
        if stem(&path) != "understat_player" {
            continue;
        }
        let csv = Csv::read(&path)?;
        for (id, name) in csv.cells("id").into_iter().zip(csv.cells("player_name")) {
            names.insert(parse_int(id, "id")?, as_str(name));
        }
    }
    Ok(names.into_iter().map(|(k, v)| (k, html_unescape(&v))).collect())
}

fn understat_date(text: &str) -> Result<chrono::NaiveDate> {
    let text = text.trim();
    for format in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"] {
        if let Ok(ts) = NaiveDateTime::parse_from_str(text, format) {
            return Ok(ts.date());
        }
    }
    chrono::NaiveDate::parse_from_str(text, "%Y-%m-%d")
        .with_context(|| format!("bad Understat date {text:?}"))
}

fn understat_player_match(
    raw: Vec<PlayerFileRow>,
    names: &HashMap<i64, String>,
    fixtures: &[Fixture],
    resolver: &TeamResolver,
) -> Result<Vec<Upm>> {
    let total = raw.len();
    let mut epl: Vec<(PlayerFileRow, i64, i64)> = raw
        .into_iter()
        .filter_map(|r| {
            let home = resolver.find("understat", &r.home)?;
            let away = resolver.find("understat", &r.away)?;
            Some((r, home, away))
        })
        .collect();
    // Newest folder wins for a repeated (understat_id, match).
    epl.sort_by_key(|(r, _, _)| std::cmp::Reverse(r.folder_season));
    let mut seen = HashSet::new();
    epl.retain(|(r, _, _)| seen.insert((r.understat_id, r.match_id)));
    let n_epl = epl.len();
    let by_sides: HashMap<(i64, i64, i64), &Fixture> =
        fixtures.iter().map(|f| ((f.season, f.home_team_key, f.away_team_key), f)).collect();
    let mut no_fixture_seasons = BTreeSet::new();
    let mut no_fixture = 0;
    let mut out = Vec::new();
    let mut off = Vec::new();
    for (r, home, away) in epl {
        let Some(f) = by_sides.get(&(r.season, home, away)) else {
            no_fixture += 1;
            no_fixture_seasons.insert(r.season);
            continue;
        };
        let kickoff = f.kickoff_time.context("fixture without a kickoff")?;
        let us_date = days_from_date(understat_date(&r.date)?);
        let kickoff_date = days_from_date(uk_date(kickoff));
        if (us_date - kickoff_date).abs() > 1 {
            off.push(format!("{} {} {} {}", r.understat_id, r.match_id, r.date, kickoff));
        }
        out.push(Upm {
            understat_id: r.understat_id,
            fixture_key: f.fixture_key,
            season: r.season,
            us_match_id: r.match_id,
            us_minutes: r.time,
            us_goals: r.goals,
            us_npg: r.npg,
            us_xg: r.xg,
            us_npxg: r.npxg,
            us_xa: r.xa,
            us_assists: r.assists,
            us_shots: r.shots,
            us_key_passes: r.key_passes,
            us_position: r.position,
            understat_name: names.get(&r.understat_id).cloned().unwrap_or(r.file_name),
            event_time: kickoff,
            available_at: f.available_at,
        });
    }
    log::info!(
        "understat: {total} player rows, {n_epl} EPL (after dedupe), {no_fixture} without a fixture (seasons {:?})",
        no_fixture_seasons
    );
    if !off.is_empty() {
        bail!(
            "{} Understat row(s) dated away from their fixture:\n{}",
            off.len(),
            off[..off.len().min(10)].join("\n")
        );
    }
    Ok(out)
}

fn validate_upm(rows: &[Upm]) -> Result<()> {
    let mut keys = HashSet::new();
    let mut match_fixture: HashMap<i64, i64> = HashMap::new();
    for r in rows {
        let ok = keys.insert((r.understat_id, r.fixture_key))
            && r.available_at > r.event_time
            && *match_fixture.entry(r.us_match_id).or_insert(r.fixture_key) == r.fixture_key
            && r.season >= FIRST_SEASON
            && (0..=MAX_MINUTES).contains(&r.us_minutes)
            && [r.us_goals, r.us_npg, r.us_assists, r.us_shots, r.us_key_passes]
                .iter()
                .all(|&v| v >= 0)
            && [r.us_xg, r.us_npxg, r.us_xa].iter().all(|&v| v >= 0.0);
        if !ok {
            bail!(
                "table 'understat_player_match' failed validation at {} {}",
                r.understat_id,
                r.fixture_key
            );
        }
    }
    Ok(())
}

fn upm_table(rows: &[Upm]) -> Table {
    macro_rules! ints {
        ($f:ident) => {
            Column::I64(rows.iter().map(|r| r.$f).collect())
        };
    }
    Table::new()
        .with("understat_id", ints!(understat_id))
        .with("fixture_key", ints!(fixture_key))
        .with("season", ints!(season))
        .with("us_match_id", ints!(us_match_id))
        .with("us_minutes", ints!(us_minutes))
        .with("us_goals", ints!(us_goals))
        .with("us_npg", ints!(us_npg))
        .with("us_xg", Column::F64(rows.iter().map(|r| r.us_xg).collect()))
        .with("us_npxg", Column::F64(rows.iter().map(|r| r.us_npxg).collect()))
        .with("us_xa", Column::F64(rows.iter().map(|r| r.us_xa).collect()))
        .with("us_assists", ints!(us_assists))
        .with("us_shots", ints!(us_shots))
        .with("us_key_passes", ints!(us_key_passes))
        .with("us_position", Column::Str(rows.iter().map(|r| r.us_position.clone()).collect()))
        .with(
            "understat_name",
            Column::Str(rows.iter().map(|r| r.understat_name.clone()).collect()),
        )
        .with("event_time", Column::Ts(rows.iter().map(|r| r.event_time).collect()))
        .with("available_at", Column::Ts(rows.iter().map(|r| r.available_at).collect()))
}

// --- the matcher (shared with ESPN) ----------------------------------------------------------

/// An FPL appearance (minutes > 0).
#[derive(Clone, Debug)]
pub struct FplApp {
    pub player_key: i64,
    pub fixture_key: i64,
    pub season: i64,
    pub minutes: i64,
    pub goals_scored: i64,
}

pub fn fpl_appearances(player_match: &[PlayerMatch]) -> Vec<FplApp> {
    player_match
        .iter()
        .filter(|r| r.count("minutes") > 0)
        .map(|r| FplApp {
            player_key: r.player_key,
            fixture_key: r.fixture_key,
            season: r.season,
            minutes: r.count("minutes"),
            goals_scored: r.count("goals_scored"),
        })
        .collect()
}

/// player_key -> every FPL full name and web_name seen across seasons.
pub fn fpl_name_variants(player_season: &[PlayerSeason]) -> HashMap<i64, BTreeSet<String>> {
    let mut names: HashMap<i64, BTreeSet<String>> = HashMap::new();
    for p in player_season {
        let set = names.entry(p.player_key).or_default();
        set.insert(format!("{} {}", p.first_name, p.second_name));
        set.insert(p.web_name.clone());
    }
    names
}

/// A source id x player_key candidate pair.
#[derive(Clone, Debug)]
pub struct Pair {
    pub id: i64,
    pub player_key: i64,
    pub together: i64,
    pub shared: i64,
    pub us_apps: i64,
    pub fpl_apps: i64,
    pub overlap: f64,
    pub coverage: f64,
    pub name_score: f64,
}

/// Pairs that appeared in at least one common fixture (`apps`: (id, fixture_key, minutes)).
pub fn candidate_pairs(apps: &[(i64, i64, i64)], fpl: &[FplApp]) -> Vec<Pair> {
    let mut by_fixture: HashMap<i64, Vec<&FplApp>> = HashMap::new();
    for a in fpl {
        by_fixture.entry(a.fixture_key).or_default().push(a);
    }
    let mut stats: BTreeMap<(i64, i64), (i64, i64)> = BTreeMap::new();
    let mut us_apps: HashMap<i64, i64> = HashMap::new();
    let mut fixtures = HashSet::new();
    for &(id, fixture, minutes) in apps {
        *us_apps.entry(id).or_default() += 1;
        fixtures.insert(fixture);
        for a in by_fixture.get(&fixture).into_iter().flatten() {
            let e = stats.entry((id, a.player_key)).or_default();
            e.0 += 1;
            e.1 += i64::from((minutes - a.minutes).abs() <= MINUTES_TOLERANCE);
        }
    }
    let mut fpl_apps: HashMap<i64, i64> = HashMap::new();
    for a in fpl.iter().filter(|a| fixtures.contains(&a.fixture_key)) {
        *fpl_apps.entry(a.player_key).or_default() += 1;
    }
    stats
        .into_iter()
        .map(|((id, player_key), (together, shared))| {
            let (u, f) = (us_apps[&id], fpl_apps[&player_key]);
            Pair {
                id,
                player_key,
                together,
                shared,
                us_apps: u,
                fpl_apps: f,
                overlap: shared as f64 / u.min(f) as f64,
                coverage: shared as f64 / u.max(f) as f64,
                name_score: 0.0,
            }
        })
        .collect()
}

/// Best token_set_ratio between the pair's source name and any FPL name variant.
pub fn score_names(
    pairs: &mut [Pair],
    source_names: &HashMap<i64, String>,
    fpl_names: &HashMap<i64, BTreeSet<String>>,
) {
    let source: HashMap<i64, String> =
        source_names.iter().map(|(k, v)| (*k, normalise_name(v))).collect();
    let fpl: HashMap<i64, BTreeSet<String>> = fpl_names
        .iter()
        .map(|(k, v)| (*k, v.iter().map(|n| normalise_name(n)).collect()))
        .collect();
    let mut cache: HashMap<(String, String), f64> = HashMap::new();
    let empty = BTreeSet::new();
    for pair in pairs {
        let a = source.get(&pair.id).cloned().unwrap_or_default();
        let mut best = 0.0f64;
        for b in fpl.get(&pair.player_key).unwrap_or(&empty) {
            let score =
                *cache.entry((a.clone(), b.clone())).or_insert_with(|| token_set_ratio(&a, b));
            best = best.max(score);
        }
        pair.name_score = best;
    }
}

fn acceptable(p: &Pair) -> bool {
    p.name_score >= MIN_AUTO_NAME_SCORE
        && ((p.overlap >= 0.8 && p.shared >= 3)
            || (p.overlap >= 0.5 && p.name_score >= 85.0)
            || (p.together >= 1 && p.name_score >= 95.0))
}

/// One-to-one assignment of acceptable pairs by (coverage, shared, name) descending.
fn greedy_assign(pairs: &[Pair]) -> Vec<Pair> {
    let mut ranked: Vec<&Pair> = pairs.iter().filter(|p| acceptable(p)).collect();
    ranked.sort_by(|a, b| {
        b.coverage
            .total_cmp(&a.coverage)
            .then(b.shared.cmp(&a.shared))
            .then(b.name_score.total_cmp(&a.name_score))
            .then(a.id.cmp(&b.id))
            .then(a.player_key.cmp(&b.player_key))
    });
    let (mut used_id, mut used_key) = (HashSet::new(), HashSet::new());
    let mut out = Vec::new();
    for p in ranked {
        if used_id.contains(&p.id) || used_key.contains(&p.player_key) {
            continue;
        }
        used_id.insert(p.id);
        used_key.insert(p.player_key);
        out.push(p.clone());
    }
    out
}

/// `source` rows of config/overrides.csv: (source id, player_key), either may be blank.
pub fn read_overrides(config_dir: &Path, source: &str) -> Result<Vec<(Option<i64>, Option<i64>)>> {
    let path = config_dir.join(OVERRIDES_FILE);
    if !path.exists() {
        return Ok(Vec::new());
    }
    let mut reader = csv::Reader::from_path(&path)?;
    let header: Vec<String> = reader.headers()?.iter().map(str::to_string).collect();
    let col = |name: &str| {
        header
            .iter()
            .position(|h| h == name)
            .with_context(|| format!("{}: no {name} column", path.display()))
    };
    let (src, id, key) = (col("source")?, col("source_id")?, col("player_key")?);
    let mut out = Vec::new();
    for record in reader.records() {
        let record = record?;
        if record.get(src).map(str::trim) != Some(source) {
            continue;
        }
        let int = |i: usize| -> Result<Option<i64>> {
            let text = record.get(i).unwrap_or("").trim();
            if text.is_empty() { Ok(None) } else { Ok(Some(text.parse()?)) }
        };
        out.push((int(id)?, int(key)?));
    }
    let ids: Vec<i64> = out.iter().filter_map(|o| o.0).collect();
    let keys: Vec<i64> = out.iter().filter_map(|o| o.1).collect();
    if out.iter().any(|o| o.0.is_none() && o.1.is_none())
        || ids.len() != ids.iter().collect::<HashSet<_>>().len()
    {
        bail!("{}: {source} overrides need a unique source_id or a player_key", path.display());
    }
    if keys.len() != keys.iter().collect::<HashSet<_>>().len() {
        bail!("{}: player_key overridden twice for {source}", path.display());
    }
    Ok(out)
}

/// A map row: (source id, player_key or none, method, shared, overlap, coverage, name).
#[derive(Clone, Debug)]
pub struct MapRow {
    pub id: i64,
    pub player_key: Option<i64>,
    pub method: &'static str,
    pub shared: Option<i64>,
    pub overlap: Option<f64>,
    pub coverage: Option<f64>,
    pub name_score: Option<f64>,
}

/// Greedy auto assignment, then overrides, which win.
pub fn map_source(pairs: &[Pair], overrides: &[(Option<i64>, Option<i64>)]) -> Vec<MapRow> {
    let mut chosen: Vec<MapRow> = greedy_assign(pairs)
        .into_iter()
        .map(|p| MapRow {
            id: p.id,
            player_key: Some(p.player_key),
            method: "auto",
            shared: Some(p.shared),
            overlap: Some(p.overlap),
            coverage: Some(p.coverage),
            name_score: Some(p.name_score),
        })
        .collect();
    if !overrides.is_empty() {
        let ids: HashSet<i64> = overrides.iter().filter_map(|o| o.0).collect();
        let keys: HashSet<i64> = overrides.iter().filter_map(|o| o.1).collect();
        chosen.retain(|m| !ids.contains(&m.id) && !m.player_key.is_some_and(|k| keys.contains(&k)));
        for &(id, key) in overrides {
            let Some(id) = id else { continue };
            let pair = key.and_then(|k| pairs.iter().find(|p| p.id == id && p.player_key == k));
            chosen.push(MapRow {
                id,
                player_key: key,
                method: "override",
                shared: pair.map(|p| p.shared),
                overlap: pair.map(|p| p.overlap),
                coverage: pair.map(|p| p.coverage),
                name_score: pair.map(|p| p.name_score),
            });
        }
    }
    chosen
}

pub fn map_table(rows: &[MapRow], id_col: &str) -> Table {
    let n = rows.len();
    Table::new()
        .with(id_col, Column::I64(rows.iter().map(|r| r.id).collect()))
        .with("player_key", Column::OptI64(rows.iter().map(|r| r.player_key).collect()))
        .with("method", Column::Str(rows.iter().map(|r| r.method.to_string()).collect()))
        .with("shared", Column::OptI64(rows.iter().map(|r| r.shared).collect()))
        .with("overlap", Column::OptF64(rows.iter().map(|r| r.overlap).collect()))
        .with("coverage", Column::OptF64(rows.iter().map(|r| r.coverage).collect()))
        .with("name_score", Column::OptF64(rows.iter().map(|r| r.name_score).collect()))
        .with("event_time", Column::Ts(vec![EPOCH; n]))
        .with("available_at", Column::Ts(vec![EPOCH; n]))
}

pub fn validate_map(rows: &[MapRow], name: &str) -> Result<()> {
    let ids: HashSet<i64> = rows.iter().map(|r| r.id).collect();
    let keys: Vec<i64> = rows.iter().filter_map(|r| r.player_key).collect();
    if ids.len() != rows.len() || keys.len() != keys.iter().collect::<HashSet<_>>().len() {
        bail!("table '{name}' failed validation: ids and player_keys must be unique");
    }
    Ok(())
}

// --- validation ------------------------------------------------------------------------------

/// vaastav id_dict pairs: (understat_id, player_key, season).
fn id_dict_oracle(run: &Path, player_season: &[PlayerSeason]) -> Result<Vec<(i64, i64, i64)>> {
    let mut out = Vec::new();
    for season_dir in subdirs(run)? {
        let path = season_dir.join("id_dict.csv.gz");
        if !path.exists() {
            continue;
        }
        let season = i64::from(parse_season_label(file_name(&season_dir))?);
        let players: HashMap<i64, i64> = player_season
            .iter()
            .filter(|p| p.season == season)
            .map(|p| (p.element_id, p.player_key))
            .collect();
        let csv = Csv::read(&path)?;
        let mut missing = Vec::new();
        for (us, fpl) in csv.cells("Understat_ID").into_iter().zip(csv.cells("FPL_ID")) {
            let fpl = parse_int(fpl, "FPL_ID")?;
            match players.get(&fpl) {
                Some(&key) => out.push((parse_int(us, "Understat_ID")?, key, season)),
                None => missing.push(fpl),
            }
        }
        if !missing.is_empty() {
            bail!("id_dict {season}: FPL id(s) {missing:?} not in player_season");
        }
    }
    Ok(out)
}

fn mapping_problems(
    mapping: &[MapRow],
    upm: &[Upm],
    fpl_apps: &[FplApp],
    oracle: &[(i64, i64, i64)],
    exempt: &HashSet<i64>,
) -> Vec<String> {
    let mut problems = Vec::new();
    let mapped: Vec<(i64, i64)> =
        mapping.iter().filter_map(|m| m.player_key.map(|k| (m.id, k))).collect();
    let by_us: HashMap<i64, i64> = mapped.iter().copied().collect();
    let by_fpl: HashMap<i64, i64> = mapped.iter().map(|&(u, k)| (k, u)).collect();
    if by_us.len() != mapped.len() {
        problems.push("understat_id mapped more than once".to_string());
    }
    if by_fpl.len() != mapped.len() {
        problems.push("player_key mapped more than once".to_string());
    }
    let decided: HashSet<i64> = mapping.iter().map(|m| m.id).collect();
    let wrong: Vec<String> = oracle
        .iter()
        .filter(|(u, k, _)| {
            by_us.get(u).is_some_and(|x| x != k) || by_fpl.get(k).is_some_and(|x| x != u)
        })
        .map(|o| format!("{o:?}"))
        .collect();
    if !wrong.is_empty() {
        problems.push(format!(
            "{} id_dict pair(s) contradicted:\n{}",
            wrong.len(),
            wrong[..wrong.len().min(20)].join("\n")
        ));
    }
    let us_ids: HashSet<i64> = upm.iter().map(|r| r.understat_id).collect();
    let fpl_keys: HashSet<i64> = fpl_apps.iter().map(|a| a.player_key).collect();
    let unmapped: Vec<String> = oracle
        .iter()
        .filter(|(u, k, _)| us_ids.contains(u) && fpl_keys.contains(k) && !decided.contains(u))
        .map(|o| format!("{o:?}"))
        .collect();
    if !unmapped.is_empty() {
        problems.push(format!(
            "{} id_dict pair(s) not mapped:\n{}",
            unmapped.len(),
            unmapped[..unmapped.len().min(20)].join("\n")
        ));
    }
    let missing: BTreeSet<(i64, i64)> = fpl_apps
        .iter()
        .filter(|a| VALIDATED_SEASONS.contains(&a.season))
        .filter(|a| !by_fpl.contains_key(&a.player_key) && !exempt.contains(&a.player_key))
        .map(|a| (a.season, a.player_key))
        .collect();
    if !missing.is_empty() {
        let players: HashSet<i64> = missing.iter().map(|m| m.1).collect();
        problems.push(format!(
            "{} FPL player(s) with minutes in {}-{} have no Understat id (add an override if that is right): {:?}",
            players.len(),
            VALIDATED_SEASONS.start,
            VALIDATED_SEASONS.end - 1,
            missing.iter().take(20).collect::<Vec<_>>()
        ));
    }
    problems
}

fn goal_agreement_problems(mapping: &[MapRow], upm: &[Upm], fpl_apps: &[FplApp]) -> Vec<String> {
    let auto: HashMap<i64, i64> = mapping
        .iter()
        .filter(|m| m.method == "auto")
        .filter_map(|m| m.player_key.map(|k| (m.id, k)))
        .collect();
    let goals: HashMap<(i64, i64), i64> =
        fpl_apps.iter().map(|a| ((a.player_key, a.fixture_key), a.goals_scored)).collect();
    let mut per_pair: BTreeMap<(i64, i64), (i64, i64)> = BTreeMap::new();
    for r in upm {
        let Some(&key) = auto.get(&r.understat_id) else { continue };
        let Some(&g) = goals.get(&(key, r.fixture_key)) else { continue };
        let e = per_pair.entry((r.understat_id, key)).or_default();
        e.0 += 1;
        e.1 += i64::from(r.us_goals == g);
    }
    if !per_pair.is_empty() {
        let min =
            per_pair.values().map(|(a, g)| *g as f64 / *a as f64).fold(f64::INFINITY, f64::min);
        let below = per_pair.values().filter(|(a, g)| g < a).count();
        let disagree: i64 = per_pair.values().map(|(a, g)| a - g).sum();
        let total: i64 = per_pair.values().map(|(a, _)| a).sum();
        log::info!(
            "understat: goal agreement of {} auto pair(s): min {min:.3}, {below} below 100%, {disagree} of {total} common appearance(s) disagree",
            per_pair.len()
        );
    }
    let low: Vec<String> = per_pair
        .iter()
        .filter(|(_, (a, g))| (*g as f64 / *a as f64) < MIN_GOAL_AGREEMENT)
        .map(|(k, v)| format!("{k:?} {v:?}"))
        .collect();
    if low.is_empty() {
        Vec::new()
    } else {
        vec![format!(
            "{} auto pair(s) whose goals agree in < 90% of their common appearances (add an override if the pair is right):\n{}",
            low.len(),
            low.join("\n")
        )]
    }
}

// --- the `understat` builder -----------------------------------------------------------------

/// Writes understat_player_match and understat_map, rewrites player_dim (understat_id) and
/// player_match (us_* columns). Returns understat_map's row count.
pub fn build_understat(ctx: &mut BuildContext) -> Result<usize> {
    let run = vaastav_run(ctx)?;
    let resolver = TeamResolver::new(ctx.table("team_dim")?)?;
    let fixtures = Fixture::from_table(ctx.table("fixture")?)?;
    let player_season = PlayerSeason::from_table(ctx.table("player_season")?)?;
    let mut player_match = PlayerMatch::from_table(ctx.table("player_match")?)?;

    let upm = understat_player_match(
        read_player_files(&run)?,
        &read_player_names(&run)?,
        &fixtures,
        &resolver,
    )?;
    validate_upm(&upm)?;
    crate::write_table(ctx, "understat_player_match", upm_table(&upm), UPM_SORT_BY)?;

    let fpl_apps = fpl_appearances(&player_match);
    let apps: Vec<(i64, i64, i64)> =
        upm.iter().map(|r| (r.understat_id, r.fixture_key, r.us_minutes)).collect();
    let mut pairs = candidate_pairs(&apps, &fpl_apps);
    let us_names: HashMap<i64, String> =
        upm.iter().map(|r| (r.understat_id, r.understat_name.clone())).collect();
    score_names(&mut pairs, &us_names, &fpl_name_variants(&player_season));
    let overrides = read_overrides(&ctx.config_dir, "understat")?;
    let mapping = map_source(&pairs, &overrides);
    let exempt: HashSet<i64> =
        overrides.iter().filter(|o| o.0.is_none()).filter_map(|o| o.1).collect();
    let mut problems = mapping_problems(
        &mapping,
        &upm,
        &fpl_apps,
        &id_dict_oracle(&run, &player_season)?,
        &exempt,
    );
    problems.extend(goal_agreement_problems(&mapping, &upm, &fpl_apps));
    if !problems.is_empty() {
        bail!("Understat mapping failed: {}", problems.join("\n"));
    }
    validate_map(&mapping, "understat_map")?;
    crate::write_table(ctx, "understat_map", map_table(&mapping, "understat_id"), MAP_SORT_BY)?;
    let us_ids: HashSet<i64> = upm.iter().map(|r| r.understat_id).collect();
    log::info!(
        "understat_map: {} of {} Understat ids mapped ({} override row(s))",
        mapping.iter().filter(|m| m.player_key.is_some()).count(),
        us_ids.len(),
        mapping.iter().filter(|m| m.method == "override").count()
    );

    let ids: HashMap<i64, i64> =
        mapping.iter().filter_map(|m| m.player_key.map(|k| (k, m.id))).collect();
    let dim = player_dim_from_seasons(&player_season, &ids);
    crate::write_table(ctx, "player_dim", dim, crate::players::PLAYER_DIM_SORT_BY)?;

    // player_match with us_* filled by (player_key, fixture_key).
    let by_us: HashMap<i64, i64> = ids.iter().map(|(k, u)| (*u, *k)).collect();
    let mut keyed: HashMap<(i64, i64), &Upm> = HashMap::new();
    for r in &upm {
        if let Some(&key) = by_us.get(&r.understat_id) {
            keyed.insert((key, r.fixture_key), r);
        }
    }
    let mut joined = 0;
    for r in &mut player_match {
        r.understat = match keyed.get(&(r.player_key, r.fixture_key)) {
            Some(u) => {
                joined += 1;
                [
                    Some(u.us_minutes as f64),
                    Some(u.us_goals as f64),
                    Some(u.us_npg as f64),
                    Some(u.us_xg),
                    Some(u.us_npxg),
                    Some(u.us_xa),
                    Some(u.us_shots as f64),
                    Some(u.us_key_passes as f64),
                ]
            }
            None => [None; 8],
        };
    }
    if keyed.len() > joined {
        log::info!(
            "understat: {} mapped appearance(s) without a player_match row",
            keyed.len() - joined
        );
    }
    let table = validated_player_match_table(&player_match)?;
    crate::write_table(ctx, "player_match", table, crate::players::PLAYER_MATCH_SORT_BY)?;
    let mut minutes: BTreeMap<i64, (i64, i64)> = BTreeMap::new();
    for r in &player_match {
        let e = minutes.entry(r.season).or_default();
        e.1 += r.count("minutes");
        if r.understat[0].is_some() {
            e.0 += r.count("minutes");
        }
    }
    let coverage: BTreeMap<i64, f64> = minutes
        .iter()
        .map(|(s, (c, t))| (*s, (*c as f64 / *t as f64 * 1000.0).round() / 1000.0))
        .collect();
    log::info!("understat: share of FPL minutes with Understat data by season: {coverage:?}");
    ctx.forget("player_dim");
    ctx.forget("player_match");
    Ok(mapping.len())
}

// --- team_match ------------------------------------------------------------------------------

struct Side {
    fixture_key: i64,
    season: i64,
    kickoff_time: Micros,
    available_at: Micros,
    team_key: i64,
    opponent_key: i64,
    is_home: bool,
    goals_for: i64,
    goals_against: i64,
}

fn team_sides(fixtures: &[Fixture]) -> Result<Vec<Side>> {
    let played: Vec<&Fixture> = fixtures.iter().filter(|f| f.finished).collect();
    let mut sides = Vec::new();
    for home in [true, false] {
        for f in &played {
            let (hg, ag) =
                (f.home_goals.context("finished without goals")?, f.away_goals.context("goals")?);
            sides.push(Side {
                fixture_key: f.fixture_key,
                season: f.season,
                kickoff_time: f.kickoff_time.context("finished without a kickoff")?,
                available_at: f.available_at,
                team_key: if home { f.home_team_key } else { f.away_team_key },
                opponent_key: if home { f.away_team_key } else { f.home_team_key },
                is_home: home,
                goals_for: if home { hg } else { ag },
                goals_against: if home { ag } else { hg },
            });
        }
    }
    Ok(sides)
}

/// Understat team-file rows joined to sides on (team, is_home, UK-local kickoff date), one
/// per side (the fixture's season folder first, else the newest): (xG, xGA, npxG, npxGA).
fn understat_team_rows(
    run: &Path,
    sides: &[Side],
    resolver: &TeamResolver,
) -> Result<HashMap<(i64, i64), [Option<f64>; 4]>> {
    let by_key: HashMap<(i64, bool, i32), &Side> = sides
        .iter()
        .map(|s| ((s.team_key, s.is_home, days_from_date(uk_date(s.kickoff_time))), s))
        .collect();
    let mut joined: Vec<(bool, i64, (i64, i64), [Option<f64>; 4])> = Vec::new();
    let (mut unmatched, mut differ) = (Vec::new(), Vec::new());
    for path in understat_files(run)? {
        let stem = stem(&path);
        if !stem.starts_with("understat_") || UNDERSTAT_AGGREGATE_FILES.contains(&stem) {
            continue;
        }
        let team_name = stem.trim_start_matches("understat_").replace('_', " ");
        let team_key = resolver.understat(&team_name)?;
        let folder = folder_season(&path)?;
        let csv = Csv::read(&path)?;
        let (h_a, date, scored, missed) =
            (csv.cells("h_a"), csv.cells("date"), csv.cells("scored"), csv.cells("missed"));
        let values: Vec<Vec<Option<&str>>> =
            ["xG", "xGA", "npxG", "npxGA"].iter().map(|c| csv.cells(c)).collect();
        for i in 0..csv.len() {
            let is_home = h_a[i] == Some("h");
            let day = days_from_date(understat_date(date[i].unwrap_or(""))?);
            let Some(side) = by_key.get(&(team_key, is_home, day)) else {
                unmatched.push(format!("{team_name} {:?} {:?} {folder}", h_a[i], date[i]));
                continue;
            };
            if parse_int(scored[i], "scored")? != side.goals_for
                || parse_int(missed[i], "missed")? != side.goals_against
            {
                differ.push(format!("{team_name} {:?}", date[i]));
            }
            let xg: [Option<f64>; 4] =
                std::array::from_fn(|j| parse_opt_float(values[j][i], "xG").ok().flatten());
            joined.push((folder == side.season, folder, (side.fixture_key, side.team_key), xg));
        }
    }
    if !unmatched.is_empty() {
        bail!(
            "{} Understat team row(s) match no fixture:\n{}",
            unmatched.len(),
            unmatched[..unmatched.len().min(20)].join("\n")
        );
    }
    if !differ.is_empty() {
        bail!(
            "{} Understat team row(s) disagree with the result:\n{}",
            differ.len(),
            differ[..differ.len().min(20)].join("\n")
        );
    }
    // Own folder first, then the newest; stable.
    joined.sort_by_key(|(own, folder, _, _)| std::cmp::Reverse((*own, *folder)));
    let mut out = HashMap::new();
    for (_, _, key, xg) in joined {
        out.entry(key).or_insert(xg);
    }
    Ok(out)
}

/// football-data `HxG`/`AxG` (seasons whose files carry them) by fixture_key.
fn football_data_xg(
    ctx: &BuildContext,
    fixtures: &[Fixture],
    resolver: &TeamResolver,
) -> Result<HashMap<i64, (Option<f64>, Option<f64>)>> {
    let keys: HashMap<(i64, i64, i64), i64> = fixtures
        .iter()
        .map(|f| ((f.season, f.home_team_key, f.away_team_key), f.fixture_key))
        .collect();
    let mut out = HashMap::new();
    for season_dir in subdirs(&ctx.root().join("football-data").join("E0"))? {
        let name = file_name(&season_dir);
        let season = 2000 + name[..2].parse::<i64>()?;
        let Some(path) =
            latest_entry(&ctx.store, "football-data", &format!("E0/{name}"), ".csv.gz")?
        else {
            continue;
        };
        if season < FIRST_SEASON {
            continue;
        }
        let csv = Csv::read(&path)?;
        if !csv.has("HxG") {
            continue;
        }
        let csv = csv.drop_na("Date");
        let (home, away, hxg, axg) =
            (csv.cells("HomeTeam"), csv.cells("AwayTeam"), csv.cells("HxG"), csv.cells("AxG"));
        for i in 0..csv.len() {
            let h = resolver.football_data(home[i].unwrap_or("nan"))?;
            let a = resolver.football_data(away[i].unwrap_or("nan"))?;
            let key = keys
                .get(&(season, h, a))
                .context("football-data xG rows without a fixture (see `fixture` cross-check)")?;
            out.insert(*key, (parse_opt_float(hxg[i], "HxG")?, parse_opt_float(axg[i], "AxG")?));
        }
    }
    Ok(out)
}

pub fn build_team_match(ctx: &mut BuildContext) -> Result<Table> {
    let resolver = TeamResolver::new(ctx.table("team_dim")?)?;
    let fixtures = Fixture::from_table(ctx.table("fixture")?)?;
    let sides = team_sides(&fixtures)?;
    let us = understat_team_rows(&vaastav_run(ctx)?, &sides, &resolver)?;
    let fd = football_data_xg(ctx, &fixtures, &resolver)?;
    let player_match = PlayerMatch::from_table(ctx.table("player_match")?)?;
    // Sum of a side's fpl_xg; null if any player row lacks it.
    let mut fpl: HashMap<(i64, i64), (f64, usize, usize)> = HashMap::new();
    for r in &player_match {
        let e = fpl.entry((r.fixture_key, r.team_key)).or_insert((0.0, 0, 0));
        e.2 += 1;
        if let Some(x) = r.fpl_xg() {
            e.0 += x;
            e.1 += 1;
        }
    }
    let side_xg = |fixture: i64, team: i64| -> Option<f64> {
        fpl.get(&(fixture, team))
            .and_then(|(sum, count, size)| (*count > 0 && count == size).then_some(*sum))
    };
    let mut cols: Vec<Vec<Option<f64>>> = vec![Vec::new(); 8];
    for s in &sides {
        let u = us.get(&(s.fixture_key, s.team_key)).copied().unwrap_or([None; 4]);
        let (hxg, axg) = fd.get(&s.fixture_key).copied().unwrap_or((None, None));
        let (fd_xg, fd_xga) = if s.is_home { (hxg, axg) } else { (axg, hxg) };
        let values = [
            u[0],
            u[1],
            u[2],
            u[3],
            fd_xg,
            fd_xga,
            side_xg(s.fixture_key, s.team_key),
            side_xg(s.fixture_key, s.opponent_key),
        ];
        for (j, v) in values.into_iter().enumerate() {
            cols[j].push(v);
        }
    }
    let mut table = Table::new()
        .with("fixture_key", Column::I64(sides.iter().map(|s| s.fixture_key).collect()))
        .with("team_key", Column::I64(sides.iter().map(|s| s.team_key).collect()))
        .with("season", Column::I64(sides.iter().map(|s| s.season).collect()))
        .with("is_home", Column::Bool(sides.iter().map(|s| s.is_home).collect()))
        .with("goals_for", Column::I64(sides.iter().map(|s| s.goals_for).collect()))
        .with("goals_against", Column::I64(sides.iter().map(|s| s.goals_against).collect()));
    for (name, values) in
        ["us_xg", "us_xga", "us_npxg", "us_npxga", "fd_xg", "fd_xga", "fpl_xg", "fpl_xga"]
            .iter()
            .zip(cols)
    {
        table.push(name, Column::OptF64(values));
    }
    table.push("event_time", Column::Ts(sides.iter().map(|s| s.kickoff_time).collect()));
    table.push("available_at", Column::Ts(sides.iter().map(|s| s.available_at).collect()));
    Ok(table)
}
