//! `team_dim` from `config/teams.csv`, name -> team_key resolvers per source, and the
//! cross-source coverage check that fails the build on any unknown club. Port of
//! `fplopt.build.teams` (see its module docstring for the sampling and sources).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use fplopt_archive::raw_store::RawStore;
use fplopt_core::seasons::parse_season_label;
use serde_json::Value;

use crate::common::{
    Bootstrap, BuildContext, Csv, EPOCH, file_name, fplcache_and_own_bootstraps, json_i64,
    json_str, latest_complete_run, latest_entry, parse_int, subdirs,
};
use crate::table::{Column, Table};

pub const CONFIG_FILE: &str = "teams.csv";
const CONFIG_COLUMNS: [&str; 7] =
    ["team_key", "short_name", "fpl_names", "football_data", "understat", "odds_api", "espn"];
pub const SYNTHETIC_KEY_MIN: i64 = 1000;
pub const UNDERSTAT_AGGREGATE_FILES: [&str; 2] = ["understat_player", "understat_team"];
const MAX_WHERE_SHOWN: usize = 3;
pub const SORT_BY: &[&str] = &["team_key"];

struct ConfigRow {
    team_key: i64,
    short_name: Option<String>,
    fpl_names: Option<String>,
    football_data: Option<String>,
    understat: Option<String>,
    odds_api: Option<String>,
    espn: Option<String>,
}

fn read_teams_config(path: &Path) -> Result<Vec<ConfigRow>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut reader = csv::ReaderBuilder::new().has_headers(true).from_reader(text.as_bytes());
    let header: Vec<String> = reader.headers()?.iter().map(str::to_string).collect();
    if header != CONFIG_COLUMNS {
        bail!("{}: columns must be {}, got {header:?}", path.display(), CONFIG_COLUMNS.join(","));
    }
    let mut problems = Vec::new();
    let mut seen: HashMap<(String, String), i64> = HashMap::new();
    let mut rows = Vec::new();
    for (i, record) in reader.records().enumerate() {
        let record = record?;
        let line = i + 2;
        let cell = |j: usize| -> Option<String> {
            record.get(j).map(str::trim).filter(|s| !s.is_empty()).map(str::to_string)
        };
        let key_text = cell(0).unwrap_or_default();
        let key = match key_text.parse::<i64>() {
            Ok(k) if k > 0 && key_text.chars().all(|c| c.is_ascii_digit()) => k,
            _ => {
                problems.push(format!("line {line}: bad team_key {key_text:?}"));
                continue;
            }
        };
        let names: Vec<String> = cell(2)
            .unwrap_or_default()
            .split(';')
            .map(str::trim)
            .filter(|n| !n.is_empty())
            .map(str::to_string)
            .collect();
        let fpl_names = (!names.is_empty()).then(|| names.join(";"));
        let synthetic = key >= SYNTHETIC_KEY_MIN;
        if synthetic && fpl_names.is_some() {
            problems.push(format!("team_key {key}: synthetic (>= 1000) but has fpl_names"));
        }
        if !synthetic && fpl_names.is_none() {
            problems.push(format!("team_key {key}: FPL club without fpl_names"));
        }
        let row = ConfigRow {
            team_key: key,
            short_name: cell(1),
            fpl_names,
            football_data: cell(3),
            understat: cell(4),
            odds_api: cell(5),
            espn: cell(6),
        };
        if row.football_data.is_none() {
            problems.push(format!("team_key {key}: no football_data name"));
        }
        let mut values = vec![("team_key".to_string(), key.to_string())];
        for (column, value) in [
            ("short_name", &row.short_name),
            ("football_data", &row.football_data),
            ("understat", &row.understat),
            ("odds_api", &row.odds_api),
            ("espn", &row.espn),
        ] {
            if let Some(v) = value {
                values.push((column.to_string(), v.clone()));
            }
        }
        values.extend(names.iter().map(|n| ("fpl_names".to_string(), n.clone())));
        for (column, value) in values {
            let other = *seen.entry((column.clone(), value.clone())).or_insert(key);
            if other != key {
                problems.push(format!("{column} '{value}' used by team_key {other} and {key}"));
            }
        }
        rows.push(row);
    }
    if !problems.is_empty() {
        bail!("{}: {}", path.display(), problems.join("; "));
    }
    Ok(rows)
}

/// The `team_dim` table for a teams.csv (static: event_time = available_at = epoch).
pub fn team_dim_from_config(path: &Path) -> Result<Table> {
    let rows = read_teams_config(path)?;
    let n = rows.len();
    let opt = |f: fn(&ConfigRow) -> &Option<String>| -> Vec<Option<String>> {
        rows.iter().map(|r| f(r).clone()).collect()
    };
    let short_name: Vec<String> =
        rows.iter().map(|r| r.short_name.clone().unwrap_or_else(|| "None".into())).collect();
    let football_data: Vec<String> =
        rows.iter().map(|r| r.football_data.clone().unwrap_or_else(|| "None".into())).collect();
    Ok(Table::new()
        .with("team_key", Column::I64(rows.iter().map(|r| r.team_key).collect()))
        .with("short_name", Column::Str(short_name))
        .with("fpl_names", Column::OptStr(opt(|r| &r.fpl_names)))
        .with("football_data_name", Column::Str(football_data))
        .with("understat_name", Column::OptStr(opt(|r| &r.understat)))
        .with("odds_api_name", Column::OptStr(opt(|r| &r.odds_api)))
        .with("espn_id", Column::OptStr(opt(|r| &r.espn)))
        .with("in_fpl", Column::Bool(rows.iter().map(|r| r.team_key < SYNTHETIC_KEY_MIN).collect()))
        .with("event_time", Column::Ts(vec![EPOCH; n]))
        .with("available_at", Column::Ts(vec![EPOCH; n])))
}

/// Name/code -> team_key lookups per source ('football_data', 'understat', 'odds_api',
/// 'espn', 'fpl_name').
#[derive(Clone, Debug)]
pub struct TeamResolver {
    maps: HashMap<&'static str, HashMap<String, i64>>,
    fpl_names: HashMap<i64, HashSet<String>>,
}

impl TeamResolver {
    pub fn new(team_dim: &Table) -> Result<Self> {
        let keys = team_dim.i64("team_key")?;
        let mut maps: HashMap<&'static str, HashMap<String, i64>> = HashMap::new();
        for (source, column) in [
            ("football_data", "football_data_name"),
            ("understat", "understat_name"),
            ("odds_api", "odds_api_name"),
            ("espn", "espn_id"),
        ] {
            let values = team_dim.opt_str(column)?;
            let map = maps.entry(source).or_default();
            for (value, &key) in values.iter().zip(keys) {
                if let Some(v) = value {
                    map.insert(v.clone(), key);
                }
            }
        }
        let in_fpl = team_dim.bool("in_fpl")?;
        let names = team_dim.opt_str("fpl_names")?;
        let mut fpl_map = HashMap::new();
        let mut fpl_names = HashMap::new();
        for ((&key, &fpl), names) in keys.iter().zip(in_fpl).zip(&names) {
            if !fpl {
                continue;
            }
            let names = names.clone().unwrap_or_default();
            let set: HashSet<String> = names.split(';').map(str::to_string).collect();
            for name in &set {
                fpl_map.insert(name.clone(), key);
            }
            fpl_names.insert(key, set);
        }
        maps.insert("fpl_name", fpl_map);
        Ok(Self { maps, fpl_names })
    }

    pub fn find(&self, source: &str, name: &str) -> Option<i64> {
        self.maps.get(source).and_then(|m| m.get(name)).copied()
    }

    fn get(&self, source: &str, label: &str, name: &str) -> Result<i64> {
        self.find(source, name)
            .with_context(|| format!("unknown {label} team name '{name}' (config/teams.csv)"))
    }

    pub fn football_data(&self, name: &str) -> Result<i64> {
        self.get("football_data", "football-data", name)
    }

    pub fn understat(&self, name: &str) -> Result<i64> {
        self.get("understat", "Understat", name)
    }

    pub fn odds_api(&self, name: &str) -> Result<i64> {
        self.get("odds_api", "Odds API", name)
    }

    pub fn fpl_code(&self, code: i64) -> Result<i64> {
        if !self.fpl_names.contains_key(&code) {
            bail!("unknown FPL team code {code} (config/teams.csv)");
        }
        Ok(code)
    }

    pub fn fpl_names(&self, code: i64) -> Result<&HashSet<String>> {
        Ok(&self.fpl_names[&self.fpl_code(code)?])
    }
}

// --- coverage --------------------------------------------------------------------------------

fn vaastav_season_dirs(run: &Path) -> Result<Vec<(String, PathBuf)>> {
    Ok(subdirs(run)?
        .into_iter()
        .filter(|p| parse_season_label(file_name(p)).is_ok())
        .map(|p| (file_name(&p).to_string(), p))
        .collect())
}

/// First and last bootstrap of each calendar month in each archive.
fn sample_bootstraps(store: &RawStore) -> Result<Vec<Bootstrap>> {
    use chrono::Datelike;
    let mut months: BTreeMap<(&'static str, i32, u32), Vec<Bootstrap>> = BTreeMap::new();
    let mut order = Vec::new();
    for entry in fplcache_and_own_bootstraps(store)? {
        let key = (entry.source, entry.taken_at.year(), entry.taken_at.month());
        if !months.contains_key(&key) {
            order.push(key);
        }
        months.entry(key).or_default().push(entry);
    }
    let mut sample = Vec::new();
    for key in order {
        let entries = &months[&key];
        sample.push(entries[0].clone());
        if entries.len() > 1 {
            sample.push(entries[entries.len() - 1].clone());
        }
    }
    sample.sort_by(|a, b| (a.taken_at, a.source).cmp(&(b.taken_at, b.source)));
    Ok(sample)
}

type Flags = BTreeMap<String, BTreeSet<String>>;

fn fpl_teams(
    store: &RawStore,
    run: &Path,
    resolver: &TeamResolver,
    flag: &mut Flags,
) -> Result<()> {
    let check = |code: i64, name: Option<String>, flag: &mut Flags, where_: String| match resolver
        .fpl_names(code)
    {
        Err(_) => {
            flag.entry(format!("FPL team code {code}")).or_default().insert(where_);
        }
        Ok(known) => {
            if let Some(name) = name
                && !known.contains(&name)
            {
                flag.entry(format!("FPL name '{name}' for team code {code}"))
                    .or_default()
                    .insert(where_);
            }
        }
    };
    let master_path = run.join("master_team_list.csv.gz");
    let master = if master_path.exists() { Some(Csv::read(&master_path)?) } else { None };
    for (season, path) in vaastav_season_dirs(run)? {
        let players_path = path.join("players_raw.csv.gz");
        if players_path.exists() {
            let players = Csv::read(&players_path)?;
            let mut codes: Vec<(i64, i64)> = Vec::new();
            let mut seen = HashSet::new();
            for (team, code) in players.cells("team").into_iter().zip(players.cells("team_code")) {
                let pair = (parse_int(team, "team")?, parse_int(code, "team_code")?);
                if seen.insert(pair) {
                    codes.push(pair);
                }
            }
            for &(_, code) in &codes {
                check(code, None, flag, format!("vaastav {season} players_raw"));
            }
            if let Some(master) = &master {
                let m_season = master.cells("season");
                let m_team = master.cells("team");
                let m_name = master.cells("team_name");
                for &(team, code) in &codes {
                    for i in 0..master.len() {
                        if m_season[i] == Some(season.as_str())
                            && m_team[i].and_then(|t| t.parse::<f64>().ok()) == Some(team as f64)
                        {
                            let name = m_name[i].unwrap_or("nan").to_string();
                            check(
                                code,
                                Some(name),
                                flag,
                                format!("vaastav master_team_list {season}"),
                            );
                        }
                    }
                }
            }
        }
        let teams_path = path.join("teams.csv.gz");
        if teams_path.exists() {
            let teams = Csv::read(&teams_path)?;
            for (code, name) in teams.cells("code").into_iter().zip(teams.cells("name")) {
                check(
                    parse_int(code, "code")?,
                    Some(name.unwrap_or("nan").to_string()),
                    flag,
                    format!("vaastav {season} teams"),
                );
            }
        }
    }
    for entry in sample_bootstraps(store)? {
        let where_ =
            format!("{} bootstrap {}", entry.source, entry.taken_at.format("%Y-%m-%dT%H%MZ"));
        let payload = RawStore::read_json(&entry.path)?;
        for team in payload["teams"].as_array().context("bootstrap without teams")? {
            check(
                json_i64(&team["code"], "team code")?,
                Some(json_str(&team["name"])),
                flag,
                where_.clone(),
            );
        }
    }
    Ok(())
}

fn football_data_names(store: &RawStore) -> Result<Vec<(String, String)>> {
    let base = store.root.join("football-data").join("E0");
    let mut out = Vec::new();
    for season_dir in subdirs(&base)? {
        let code = file_name(&season_dir).to_string();
        let Some(path) = latest_entry(store, "football-data", &format!("E0/{code}"), ".csv.gz")?
        else {
            continue;
        };
        let df = Csv::read(&path)?.drop_na("Date");
        let names: BTreeSet<String> = df
            .cells("HomeTeam")
            .into_iter()
            .chain(df.cells("AwayTeam"))
            .map(|n| n.unwrap_or("nan").to_string())
            .collect();
        for name in names {
            out.push((name, format!("football-data E0/{code}")));
        }
    }
    Ok(out)
}

/// `*/understat/*.csv.gz` files of a vaastav run, sorted by path.
pub fn understat_files(run: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for season in subdirs(run)? {
        let dir = season.join("understat");
        if !dir.is_dir() {
            continue;
        }
        for entry in std::fs::read_dir(&dir)? {
            let path = entry?.path();
            if path.is_file() && file_name(&path).ends_with(".csv.gz") {
                out.push(path);
            }
        }
    }
    out.sort();
    Ok(out)
}

fn understat_names(run: &Path, resolver: &TeamResolver) -> Result<Vec<(String, String)>> {
    let mut out = Vec::new();
    for path in understat_files(run)? {
        let stem = file_name(&path).trim_end_matches(".csv.gz").to_string();
        let season = file_name(path.parent().unwrap().parent().unwrap()).to_string();
        if let Some(team) = stem.strip_prefix("understat_") {
            if !UNDERSTAT_AGGREGATE_FILES.contains(&stem.as_str()) {
                out.push((team.replace('_', " "), format!("vaastav {season} {stem}")));
            }
            continue;
        }
        let df = Csv::read(&path)?;
        let pairs: HashSet<(String, String)> = df
            .cells("h_team")
            .into_iter()
            .zip(df.cells("a_team"))
            .map(|(h, a)| (h.unwrap_or("nan").to_string(), a.unwrap_or("nan").to_string()))
            .collect();
        for (home, away) in pairs {
            let home_known = resolver.find("understat", &home).is_some();
            let away_known = resolver.find("understat", &away).is_some();
            if home_known != away_known {
                let name = if home_known { away } else { home };
                out.push((name, format!("vaastav {season} understat/{stem}")));
            }
        }
    }
    Ok(out)
}

fn odds_api_names(store: &RawStore) -> Result<Vec<(String, String)>> {
    let mut out = Vec::new();
    for (taken_at, path) in store.entries("odds", "soccer_epl", ".json.gz")? {
        let where_ = format!("odds soccer_epl {}", taken_at.format("%Y-%m-%dT%H%MZ"));
        let payload: Value = RawStore::read_json(&path)?;
        for event in payload.as_array().context("odds snapshot is not a list")? {
            out.push((json_str(&event["home_team"]), where_.clone()));
            out.push((json_str(&event["away_team"]), where_.clone()));
        }
    }
    Ok(out)
}

/// Fail (one line listing every unknown) if a team code or name in raw/ is not in team_dim.
pub fn validate_team_coverage(ctx: &BuildContext, team_dim: &Table) -> Result<()> {
    let resolver = TeamResolver::new(team_dim)?;
    let mut problems: Flags = BTreeMap::new();
    let run = latest_complete_run(&ctx.store, "vaastav", "data")?;
    fpl_teams(&ctx.store, &run, &resolver, &mut problems)?;
    for (name, where_) in football_data_names(&ctx.store)? {
        if resolver.find("football_data", &name).is_none() {
            problems.entry(format!("football-data name '{name}'")).or_default().insert(where_);
        }
    }
    for (name, where_) in understat_names(&run, &resolver)? {
        if resolver.find("understat", &name).is_none() {
            problems.entry(format!("Understat name '{name}'")).or_default().insert(where_);
        }
    }
    for (name, where_) in odds_api_names(&ctx.store)? {
        if resolver.find("odds_api", &name).is_none() {
            problems.entry(format!("Odds API name '{name}'")).or_default().insert(where_);
        }
    }
    if !problems.is_empty() {
        let details: Vec<String> = problems
            .iter()
            .map(|(problem, wheres)| {
                let shown: Vec<&str> =
                    wheres.iter().take(MAX_WHERE_SHOWN).map(String::as_str).collect();
                let more = if wheres.len() > MAX_WHERE_SHOWN {
                    format!(" +{} more", wheres.len() - MAX_WHERE_SHOWN)
                } else {
                    String::new()
                };
                format!("{problem} ({}{more})", shown.join(", "))
            })
            .collect();
        bail!(
            "{} unknown team code(s)/name(s), add them to config/teams.csv: {}",
            problems.len(),
            details.join("; ")
        );
    }
    Ok(())
}

pub fn build_team_dim(ctx: &mut BuildContext) -> Result<Table> {
    let team_dim = team_dim_from_config(&ctx.config_dir.join(CONFIG_FILE))?;
    validate_team_coverage(ctx, &team_dim)?;
    Ok(team_dim)
}
