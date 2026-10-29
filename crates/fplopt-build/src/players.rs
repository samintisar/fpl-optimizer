//! `player_season`, `player_dim`, `player_match`, `player_gw` and `player_gw_ownership`.
//! Port of `fplopt.build.players` (its module docstring describes sources, cleaning, timing
//! and checks).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result, bail};
use fplopt_archive::raw_store::{RawStore, parse_ts};
use fplopt_core::seasons::{bootstrap_season, season_label};
use serde_json::Value;

use crate::common::{
    BuildContext, Csv, EPOCH, KNOWN_BEFORE_DEADLINE, MANIFEST, as_str, file_name, json_bool,
    json_i64, json_opt_f64, json_opt_i64, latest_complete_season_run, micros, parse_bool,
    parse_int, parse_opt_float, parse_opt_int,
};
use crate::fixtures::{
    FIRST_SEASON, Fixture, Gameweek, bootstrap_team_codes, last_bootstrap_per_season,
    season_team_codes, vaastav_run, vaastav_seasons,
};
use crate::scoring::{NATIVE_SEASONS, StatLine, backtest_rules, score};
use crate::table::{Column, Micros, Table, read_parquet};
use crate::teams::TeamResolver;

const MANAGER_TYPE: i64 = 5;
pub const MAX_MINUTES: i64 = 90;
const MAX_GOAL_SUM_MISMATCHES: usize = 0;

pub const PLAYER_SEASON_SORT_BY: &[&str] = &["season", "player_key"];
pub const PLAYER_DIM_SORT_BY: &[&str] = &["player_key"];
pub const PLAYER_MATCH_SORT_BY: &[&str] = &["fixture_key", "player_key"];
pub const PLAYER_GW_SORT_BY: &[&str] = &["season", "gw", "player_key"];

/// First GW whose `starts` and FPL expected stats hold data (vaastav 2022-23: GW1-15 are 0
/// placeholders).
fn late_columns_first_gw(season: i64) -> i64 {
    if season == 2022 { 16 } else { 0 }
}

pub const COUNT_STATS: [&str; 14] = [
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "bps",
    "total_points",
];
const SIGNED_STATS: [&str; 2] = ["bps", "total_points"];
const OPTIONAL_INTS: [&str; 5] = [
    "starts",
    "clearances_blocks_interceptions",
    "recoveries",
    "tackles",
    "defensive_contribution",
];
const OPTIONAL_FLOATS: [(&str, &str); 3] = [
    ("expected_goals", "fpl_xg"),
    ("expected_assists", "fpl_xa"),
    ("expected_goals_conceded", "fpl_xgc"),
];
const GW_STATE: [&str; 4] = ["value", "selected", "transfers_in", "transfers_out"];
pub const UNDERSTAT_INTS: [&str; 5] =
    ["us_minutes", "us_goals", "us_npg", "us_shots", "us_key_passes"];
/// The `us_*` columns in table order, with whether each is a float.
pub const UNDERSTAT_COLUMNS: [(&str, bool); 8] = [
    ("us_minutes", false),
    ("us_goals", false),
    ("us_npg", false),
    ("us_xg", true),
    ("us_npxg", true),
    ("us_xa", true),
    ("us_shots", false),
    ("us_key_passes", false),
];
const DEFENSIVE: [&str; 4] =
    ["clearances_blocks_interceptions", "recoveries", "tackles", "defensive_contribution"];

/// Seasons in `fixture` that vaastav doesn't cover.
fn current_seasons(ctx: &mut BuildContext) -> Result<Vec<i64>> {
    let covered: HashSet<i64> = vaastav_seasons(&vaastav_run(ctx)?)?.into_keys().collect();
    let seasons: BTreeSet<i64> = ctx.table("fixture")?.i64("season")?.iter().copied().collect();
    Ok(seasons.into_iter().filter(|s| !covered.contains(s)).collect())
}

// --- player_season ---------------------------------------------------------------------------

/// A registered player of a season (`player_season` without timing).
#[derive(Clone, Debug)]
pub struct Registered {
    pub player_key: i64,
    pub season: i64,
    pub element_id: i64,
    pub element_type: i64,
    pub first_name: String,
    pub second_name: String,
    pub web_name: String,
    pub listed_at: Option<Micros>,
}

fn players_from_csv(csv: &Csv, season: i64) -> Result<Vec<Registered>> {
    let (id, code, et) = (csv.cells("id"), csv.cells("code"), csv.cells("element_type"));
    let (first, second, web) =
        (csv.cells("first_name"), csv.cells("second_name"), csv.cells("web_name"));
    let mut out = Vec::new();
    for i in 0..csv.len() {
        let element_type = parse_int(et[i], "element_type")?;
        if element_type == MANAGER_TYPE {
            continue;
        }
        out.push(Registered {
            player_key: parse_int(code[i], "code")?,
            season,
            element_id: parse_int(id[i], "id")?,
            element_type,
            first_name: as_str(first[i]),
            second_name: as_str(second[i]),
            web_name: as_str(web[i]),
            listed_at: None,
        });
    }
    Ok(out)
}

fn json_text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Null => "None".into(),
        other => other.to_string(),
    }
}

fn players_from_bootstrap(
    bootstrap: &Value,
    season: i64,
    listed_at: Micros,
) -> Result<Vec<Registered>> {
    let mut out = Vec::new();
    for e in bootstrap["elements"].as_array().context("bootstrap without elements")? {
        let element_type = json_i64(&e["element_type"], "element_type")?;
        if element_type == MANAGER_TYPE {
            continue;
        }
        out.push(Registered {
            player_key: json_i64(&e["code"], "code")?,
            season,
            element_id: json_i64(&e["id"], "id")?,
            element_type,
            first_name: json_text(&e["first_name"]),
            second_name: json_text(&e["second_name"]),
            web_name: json_text(&e["web_name"]),
            listed_at: Some(listed_at),
        });
    }
    Ok(out)
}

fn players_raw(season_dir: &Path) -> Result<Csv> {
    Csv::read(&season_dir.join("players_raw.csv.gz"))
}

/// Registered players of every season: vaastav players_raw, and for current seasons the
/// newest bootstrap of the season.
pub fn registered_players(ctx: &mut BuildContext) -> Result<Vec<Registered>> {
    let mut out = Vec::new();
    for (season, dir) in vaastav_seasons(&vaastav_run(ctx)?)? {
        out.extend(players_from_csv(&players_raw(&dir)?, season)?);
    }
    let current = current_seasons(ctx)?;
    let bootstraps = last_bootstrap_per_season(&ctx.store)?;
    for season in current {
        let entry = bootstraps
            .get(&season)
            .with_context(|| format!("no archived bootstrap for season {season}"))?;
        out.extend(players_from_bootstrap(
            &RawStore::read_json(&entry.path)?,
            season,
            micros(&entry.taken_at),
        )?);
    }
    Ok(out)
}

pub struct PlayerSeason {
    pub player_key: i64,
    pub season: i64,
    pub element_id: i64,
    pub element_type: i64,
    pub first_name: String,
    pub second_name: String,
    pub web_name: String,
    pub event_time: Micros,
    pub available_at: Micros,
}

impl PlayerSeason {
    pub fn from_table(t: &Table) -> Result<Vec<PlayerSeason>> {
        let (key, season, element, et) =
            (t.i64("player_key")?, t.i64("season")?, t.i64("element_id")?, t.i64("element_type")?);
        let (first, second, web) =
            (t.str("first_name")?, t.str("second_name")?, t.str("web_name")?);
        let (event_time, available_at) = (t.i64("event_time")?, t.i64("available_at")?);
        Ok((0..t.height())
            .map(|i| PlayerSeason {
                player_key: key[i],
                season: season[i],
                element_id: element[i],
                element_type: et[i],
                first_name: first[i].clone(),
                second_name: second[i].clone(),
                web_name: web[i].clone(),
                event_time: event_time[i],
                available_at: available_at[i],
            })
            .collect())
    }
}

pub fn build_player_season(ctx: &mut BuildContext) -> Result<Table> {
    let players = registered_players(ctx)?;
    let gameweek = Gameweek::from_table(ctx.table("gameweek")?)?;
    let player_gw = ctx.owned("player_gw")?;
    let gw_seasons: HashSet<i64> = gameweek.iter().map(|g| g.season).collect();
    let missing: BTreeSet<i64> =
        players.iter().map(|p| p.season).filter(|s| !gw_seasons.contains(s)).collect();
    if !missing.is_empty() {
        bail!("no gameweek rows for season(s) {missing:?}");
    }
    // Per player and season the player_gw row with the earliest available_at (ties: the
    // earliest deadline).
    let mut first: HashMap<(i64, i64), (Micros, Micros)> = HashMap::new();
    let (keys, seasons) = (player_gw.i64("player_key")?, player_gw.i64("season")?);
    let (events, available) = (player_gw.i64("event_time")?, player_gw.i64("available_at")?);
    for i in 0..player_gw.height() {
        let candidate = (available[i], events[i]);
        let e = first.entry((seasons[i], keys[i])).or_insert(candidate);
        if candidate < *e {
            *e = candidate;
        }
    }
    let mut last_lockdown: HashMap<i64, Micros> = HashMap::new();
    for g in &gameweek {
        let e = last_lockdown.entry(g.season).or_insert(g.lockdown_time);
        *e = (*e).max(g.lockdown_time);
    }
    let mut seen = HashSet::new();
    let mut no_rows: BTreeMap<i64, usize> = BTreeMap::new();
    let mut rows = Vec::new();
    for p in players {
        if !seen.insert((p.season, p.player_key)) {
            bail!("player_season: player {} listed twice in season {}", p.player_key, p.season);
        }
        let (event_time, available_at) = match first.get(&(p.season, p.player_key)) {
            Some(&(available_at, event_time)) => (event_time, available_at),
            None => {
                *no_rows.entry(p.season).or_default() += 1;
                let fallback = p.listed_at.unwrap_or(last_lockdown[&p.season]);
                (fallback, fallback)
            }
        };
        rows.push(PlayerSeason {
            player_key: p.player_key,
            season: p.season,
            element_id: p.element_id,
            element_type: p.element_type,
            first_name: p.first_name,
            second_name: p.second_name,
            web_name: p.web_name,
            event_time,
            available_at,
        });
    }
    if !no_rows.is_empty() {
        log::info!(
            "player_season: {} player-season(s) without player_gw rows (per season: {no_rows:?})",
            no_rows.values().sum::<usize>()
        );
    }
    let mut elements = HashSet::new();
    for r in &rows {
        if !elements.insert((r.season, r.element_id))
            || !(1..=4).contains(&r.element_type)
            || r.player_key <= 0
            || r.element_id <= 0
            || r.season < FIRST_SEASON
            || r.available_at < r.event_time - KNOWN_BEFORE_DEADLINE
        {
            bail!(
                "table 'player_season' failed validation at player {} season {}",
                r.player_key,
                r.season
            );
        }
    }
    Ok(Table::new()
        .with("player_key", Column::I64(rows.iter().map(|r| r.player_key).collect()))
        .with("season", Column::I64(rows.iter().map(|r| r.season).collect()))
        .with("element_id", Column::I64(rows.iter().map(|r| r.element_id).collect()))
        .with("element_type", Column::I64(rows.iter().map(|r| r.element_type).collect()))
        .with("first_name", Column::Str(rows.iter().map(|r| r.first_name.clone()).collect()))
        .with("second_name", Column::Str(rows.iter().map(|r| r.second_name.clone()).collect()))
        .with("web_name", Column::Str(rows.iter().map(|r| r.web_name.clone()).collect()))
        .with("event_time", Column::Ts(rows.iter().map(|r| r.event_time).collect()))
        .with("available_at", Column::Ts(rows.iter().map(|r| r.available_at).collect())))
}

// --- player_dim ------------------------------------------------------------------------------

pub fn player_dim_from_seasons(
    player_season: &[PlayerSeason],
    understat_ids: &HashMap<i64, i64>,
) -> Table {
    // Names from each player's newest season.
    let mut newest: BTreeMap<i64, &PlayerSeason> = BTreeMap::new();
    for p in player_season {
        match newest.get(&p.player_key) {
            Some(q) if q.season > p.season => {}
            _ => {
                newest.insert(p.player_key, p);
            }
        }
    }
    let rows: Vec<&PlayerSeason> = newest.into_values().collect();
    let n = rows.len();
    Table::new()
        .with("player_key", Column::I64(rows.iter().map(|r| r.player_key).collect()))
        .with("first_name", Column::Str(rows.iter().map(|r| r.first_name.clone()).collect()))
        .with("second_name", Column::Str(rows.iter().map(|r| r.second_name.clone()).collect()))
        .with("web_name", Column::Str(rows.iter().map(|r| r.web_name.clone()).collect()))
        .with("opta_code", Column::Str(rows.iter().map(|r| format!("p{}", r.player_key)).collect()))
        .with(
            "understat_id",
            Column::OptI64(
                rows.iter().map(|r| understat_ids.get(&r.player_key).copied()).collect(),
            ),
        )
        .with("event_time", Column::Ts(vec![EPOCH; n]))
        .with("available_at", Column::Ts(vec![EPOCH; n]))
}

pub fn build_player_dim(ctx: &mut BuildContext) -> Result<Table> {
    let seasons = PlayerSeason::from_table(ctx.table("player_season")?)?;
    let dim = player_dim_from_seasons(&seasons, &HashMap::new());
    let keys: HashSet<i64> = dim.i64("player_key")?.iter().copied().collect();
    crate::snapshots::report_missing_players(&keys, &ctx.table_path("player_snapshot"))?;
    Ok(dim)
}

// --- source rows -----------------------------------------------------------------------------

/// A source row (merged_gw or element-summary history) in the common raw shape.
#[derive(Clone, Debug)]
pub struct MatchRow {
    pub season: i64,
    pub element_id: i64,
    pub fpl_fixture_id: i64,
    pub round: i64,
    pub was_home: bool,
    pub opponent_id: i64,
    pub team_name: Option<String>,
    pub counts: [i64; 14],
    pub optional_ints: [Option<i64>; 5],
    pub optional_floats: [Option<f64>; 3],
    pub gw_state: [Option<i64>; 4],
    pub source: &'static str,
}

fn match_rows_from_csv(csv: &Csv, rows: &[usize], season: i64) -> Result<Vec<MatchRow>> {
    let col = |name: &str| csv.index(name);
    let need = |name: &str| csv.require(name);
    let (element, fixture, round, was_home, opponent) = (
        need("element")?,
        need("fixture")?,
        need("round")?,
        need("was_home")?,
        need("opponent_team")?,
    );
    let team = col("team");
    let counts: Vec<usize> = COUNT_STATS.iter().map(|c| need(c)).collect::<Result<_>>()?;
    let ints: Vec<Option<usize>> = OPTIONAL_INTS.iter().map(|c| col(c)).collect();
    let floats: Vec<Option<usize>> = OPTIONAL_FLOATS.iter().map(|(c, _)| col(c)).collect();
    let state: Vec<Option<usize>> = GW_STATE.iter().map(|c| col(c)).collect();
    rows.iter()
        .map(|&r| {
            let row = &csv.rows[r];
            let cell = |i: usize| row[i].as_deref();
            let mut c = [0i64; 14];
            for (j, &i) in counts.iter().enumerate() {
                c[j] = parse_int(cell(i), COUNT_STATS[j])?;
            }
            let mut oi = [None; 5];
            for (j, i) in ints.iter().enumerate() {
                oi[j] = i.map(|i| parse_opt_int(cell(i), OPTIONAL_INTS[j])).transpose()?.flatten();
            }
            let mut of = [None; 3];
            for (j, i) in floats.iter().enumerate() {
                of[j] = i
                    .map(|i| parse_opt_float(cell(i), OPTIONAL_FLOATS[j].0))
                    .transpose()?
                    .flatten();
            }
            let mut gs = [None; 4];
            for (j, i) in state.iter().enumerate() {
                gs[j] = i.map(|i| parse_opt_int(cell(i), GW_STATE[j])).transpose()?.flatten();
            }
            Ok(MatchRow {
                season,
                element_id: parse_int(cell(element), "element")?,
                fpl_fixture_id: parse_int(cell(fixture), "fixture")?,
                round: parse_int(cell(round), "round")?,
                was_home: parse_bool(cell(was_home), "was_home")?,
                opponent_id: parse_int(cell(opponent), "opponent_team")?,
                team_name: team.map(|i| as_str(cell(i))),
                counts: c,
                optional_ints: oi,
                optional_floats: of,
                gw_state: gs,
                source: "vaastav",
            })
        })
        .collect()
}

fn match_row_from_json(h: &Value, season: i64) -> Result<MatchRow> {
    let mut c = [0i64; 14];
    for (j, name) in COUNT_STATS.iter().enumerate() {
        c[j] = json_i64(h.get(*name).with_context(|| format!("KeyError('{name}')"))?, name)?;
    }
    let field = |name: &str| h.get(name).unwrap_or(&Value::Null);
    let mut oi = [None; 5];
    for (j, name) in OPTIONAL_INTS.iter().enumerate() {
        oi[j] = json_opt_i64(field(name), name)?;
    }
    let mut of = [None; 3];
    for (j, (name, _)) in OPTIONAL_FLOATS.iter().enumerate() {
        of[j] = json_opt_f64(field(name), name)?;
    }
    let mut gs = [None; 4];
    for (j, name) in GW_STATE.iter().enumerate() {
        gs[j] = json_opt_i64(field(name), name)?;
    }
    Ok(MatchRow {
        season,
        element_id: json_i64(field("element"), "element")?,
        fpl_fixture_id: json_i64(field("fixture"), "fixture")?,
        round: json_i64(field("round"), "round")?,
        was_home: json_bool(field("was_home")),
        opponent_id: json_i64(field("opponent_team"), "opponent_team")?,
        team_name: h.get("team").map(json_text),
        counts: c,
        optional_ints: oi,
        optional_floats: of,
        gw_state: gs,
        source: "fpl",
    })
}

fn vaastav_match_rows(season: i64, season_dir: &Path) -> Result<Vec<MatchRow>> {
    let merged = Csv::read(&season_dir.join("gws").join("merged_gw.csv.gz"))?;
    let players = players_raw(season_dir)?;
    let mut managers = HashSet::new();
    for (id, et) in players.cells("id").into_iter().zip(players.cells("element_type")) {
        if parse_int(et, "element_type")? == MANAGER_TYPE {
            managers.insert(parse_int(id, "id")?);
        }
    }
    let element = merged.require("element")?;
    let mut seen = HashSet::new();
    let (mut exact, mut manager_rows) = (0, 0);
    let mut keep = Vec::new();
    for (i, row) in merged.rows.iter().enumerate() {
        if !seen.insert(row) {
            exact += 1;
            continue;
        }
        if managers.contains(&parse_int(row[element].as_deref(), "element")?) {
            manager_rows += 1;
            continue;
        }
        keep.push(i);
    }
    if exact > 0 {
        log::info!("season {season}: dropped {exact} exact duplicate row(s)");
    }
    if manager_rows > 0 {
        log::info!("season {season}: dropped {manager_rows} manager row(s)");
    }
    match_rows_from_csv(&merged, &keep, season)
}

/// The bootstrap archived with an element-summary run, else the season's newest.
fn element_summary_bootstrap(ctx: &BuildContext, run: &Path, season: i64) -> Result<Value> {
    let at = parse_ts(file_name(run))?;
    let path = ctx.store.path_for("fpl", "bootstrap-static", &at, ".json.gz", None)?;
    if path.exists() {
        let bootstrap = RawStore::read_json(&path)?;
        if i64::from(bootstrap_season(&bootstrap)?) == season {
            return Ok(bootstrap);
        }
    }
    let bootstraps = last_bootstrap_per_season(&ctx.store)?;
    let entry = bootstraps
        .get(&season)
        .with_context(|| format!("no archived bootstrap for season {season}"))?;
    RawStore::read_json(&entry.path)
}

/// A current season's rows from its newest complete element-summary run and the team id
/// -> code map for them.
fn element_summary_match_rows(
    ctx: &BuildContext,
    season: i64,
    fixtures: &[Fixture],
) -> Result<(Vec<MatchRow>, HashMap<i64, i64>)> {
    let finished: HashSet<i64> = fixtures
        .iter()
        .filter(|f| f.season == season && f.finished)
        .map(|f| f.fpl_fixture_id)
        .collect();
    let Some(run) =
        latest_complete_season_run(&ctx.store, "fpl", "element-summary", season as i32)?
    else {
        if !finished.is_empty() {
            bail!(
                "no complete element-summary run for season {season}, which has {} finished fixture(s)",
                finished.len()
            );
        }
        log::info!("season {season}: no element-summary run and no finished fixture yet");
        return Ok((Vec::new(), HashMap::new()));
    };
    let manifest = RawStore::read_json(&run.join(MANIFEST))?;
    let mut paths: Vec<_> = std::fs::read_dir(&run)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| file_name(p).ends_with(".json.gz") && file_name(p) != MANIFEST)
        .collect();
    paths.sort();
    let mut rows = Vec::new();
    for path in paths {
        let payload = RawStore::read_json(&path)?;
        for h in payload["history"].as_array().context("KeyError('history')")? {
            rows.push(match_row_from_json(h, season)?);
        }
    }
    let through = match manifest.get("through_event") {
        None | Some(Value::Null) => None,
        Some(v) => Some(json_i64(v, "through_event")?),
    };
    let before = rows.len();
    rows.retain(|r| finished.contains(&r.fpl_fixture_id) && through.is_none_or(|t| r.round <= t));
    if rows.len() < before {
        log::info!(
            "element-summary {}: skipped {} unfinished row(s)",
            file_name(&run),
            before - rows.len()
        );
    }
    Ok((rows, bootstrap_team_codes(&element_summary_bootstrap(ctx, &run, season)?)?))
}

// --- assembled rows --------------------------------------------------------------------------

/// A `player_match` row (with the source's GW state carried along for player_gw).
#[derive(Clone, Debug)]
pub struct PlayerMatch {
    pub player_key: i64,
    pub season: i64,
    pub fixture_key: i64,
    pub gw: i64,
    pub element_id: i64,
    pub team_key: i64,
    pub opponent_team_key: i64,
    pub was_home: bool,
    pub kickoff_time: Micros,
    pub counts: [i64; 14],
    pub optional_ints: [Option<i64>; 5],
    pub optional_floats: [Option<f64>; 3],
    pub understat: [Option<f64>; 8],
    pub rescored_points: i64,
    pub source: String,
    pub event_time: Micros,
    pub available_at: Micros,
    pub gw_state: [Option<i64>; 4],
}

impl PlayerMatch {
    pub fn count(&self, name: &str) -> i64 {
        self.counts[COUNT_STATS.iter().position(|c| *c == name).unwrap()]
    }

    pub fn opt_int(&self, name: &str) -> Option<i64> {
        self.optional_ints[OPTIONAL_INTS.iter().position(|c| *c == name).unwrap()]
    }

    pub fn fpl_xg(&self) -> Option<f64> {
        self.optional_floats[0]
    }

    pub fn from_table(t: &Table) -> Result<Vec<PlayerMatch>> {
        let n = t.height();
        let ints = |name: &str| t.i64(name);
        let (key, season, fixture, gw, element) = (
            ints("player_key")?,
            ints("season")?,
            ints("fixture_key")?,
            ints("gw")?,
            ints("element_id")?,
        );
        let (team, opponent, home, kickoff) = (
            ints("team_key")?,
            ints("opponent_team_key")?,
            t.bool("was_home")?,
            ints("kickoff_time")?,
        );
        let counts: Vec<&[i64]> = COUNT_STATS.iter().map(|c| ints(c)).collect::<Result<_>>()?;
        let opt_ints: Vec<Vec<Option<i64>>> =
            OPTIONAL_INTS.iter().map(|c| t.opt_i64(c)).collect::<Result<_>>()?;
        let opt_floats: Vec<Vec<Option<f64>>> =
            OPTIONAL_FLOATS.iter().map(|(_, c)| t.opt_f64(c)).collect::<Result<_>>()?;
        let us: Vec<Vec<Option<f64>>> =
            UNDERSTAT_COLUMNS.iter().map(|(c, _)| t.opt_f64(c)).collect::<Result<_>>()?;
        let (rescored, source) = (ints("rescored_points")?, t.str("source")?);
        let (event_time, available_at) = (ints("event_time")?, ints("available_at")?);
        Ok((0..n)
            .map(|i| PlayerMatch {
                player_key: key[i],
                season: season[i],
                fixture_key: fixture[i],
                gw: gw[i],
                element_id: element[i],
                team_key: team[i],
                opponent_team_key: opponent[i],
                was_home: home[i],
                kickoff_time: kickoff[i],
                counts: std::array::from_fn(|j| counts[j][i]),
                optional_ints: std::array::from_fn(|j| opt_ints[j][i]),
                optional_floats: std::array::from_fn(|j| opt_floats[j][i]),
                understat: std::array::from_fn(|j| us[j][i]),
                rescored_points: rescored[i],
                source: source[i].clone(),
                event_time: event_time[i],
                available_at: available_at[i],
                gw_state: [None; 4],
            })
            .collect())
    }
}

pub fn player_match_table(rows: &[PlayerMatch]) -> Table {
    let mut t = Table::new()
        .with("player_key", Column::I64(rows.iter().map(|r| r.player_key).collect()))
        .with("season", Column::I64(rows.iter().map(|r| r.season).collect()))
        .with("fixture_key", Column::I64(rows.iter().map(|r| r.fixture_key).collect()))
        .with("gw", Column::I64(rows.iter().map(|r| r.gw).collect()))
        .with("element_id", Column::I64(rows.iter().map(|r| r.element_id).collect()))
        .with("team_key", Column::I64(rows.iter().map(|r| r.team_key).collect()))
        .with("opponent_team_key", Column::I64(rows.iter().map(|r| r.opponent_team_key).collect()))
        .with("was_home", Column::Bool(rows.iter().map(|r| r.was_home).collect()))
        .with("kickoff_time", Column::Ts(rows.iter().map(|r| r.kickoff_time).collect()))
        .with("minutes", Column::I64(rows.iter().map(|r| r.counts[0]).collect()))
        .with("starts", Column::OptI64(rows.iter().map(|r| r.optional_ints[0]).collect()));
    for (j, name) in COUNT_STATS.iter().enumerate().skip(1) {
        t.push(name, Column::I64(rows.iter().map(|r| r.counts[j]).collect()));
    }
    t.push("rescored_points", Column::I64(rows.iter().map(|r| r.rescored_points).collect()));
    for (j, name) in OPTIONAL_INTS.iter().enumerate().skip(1) {
        debug_assert_eq!(*name, DEFENSIVE[j - 1]);
        t.push(name, Column::OptI64(rows.iter().map(|r| r.optional_ints[j]).collect()));
    }
    for (j, (_, name)) in OPTIONAL_FLOATS.iter().enumerate() {
        t.push(name, Column::OptF64(rows.iter().map(|r| r.optional_floats[j]).collect()));
    }
    for (j, (name, float)) in UNDERSTAT_COLUMNS.iter().enumerate() {
        let column = if *float {
            Column::OptF64(rows.iter().map(|r| r.understat[j]).collect())
        } else {
            Column::OptI64(rows.iter().map(|r| r.understat[j].map(|v| v as i64)).collect())
        };
        t.push(name, column);
    }
    t.push("source", Column::Str(rows.iter().map(|r| r.source.clone()).collect()));
    t.push("event_time", Column::Ts(rows.iter().map(|r| r.event_time).collect()));
    t.push("available_at", Column::Ts(rows.iter().map(|r| r.available_at).collect()));
    t
}

fn fail_if(rows: &[String], what: &str) -> Result<()> {
    if !rows.is_empty() {
        let sample: Vec<&str> = rows.iter().take(10).map(String::as_str).collect();
        bail!("{} {what}:\n{}", rows.len(), sample.join("\n"));
    }
    Ok(())
}

fn describe(r: &MatchRow) -> String {
    format!(
        "season {} element {} fixture {} round {}",
        r.season, r.element_id, r.fpl_fixture_id, r.round
    )
}

/// player_match rows from raw source rows (fails on unknown fixtures, GWs or elements,
/// leftover duplicates and team inconsistencies).
fn assemble_player_match(
    raw: Vec<MatchRow>,
    fixtures: &[Fixture],
    gameweek: &[Gameweek],
    players: &[Registered],
    team_codes: &HashMap<i64, HashMap<i64, i64>>,
    resolver: &TeamResolver,
) -> Result<Vec<PlayerMatch>> {
    let fx: HashMap<(i64, i64), &Fixture> =
        fixtures.iter().map(|f| ((f.season, f.fpl_fixture_id), f)).collect();
    let unplayed: Vec<String> = raw
        .iter()
        .filter(|r| fx.get(&(r.season, r.fpl_fixture_id)).is_none_or(|f| f.gw.is_none()))
        .map(describe)
        .collect();
    fail_if(&unplayed, "row(s) without a played fixture")?;
    let gw_of = |r: &MatchRow| fx[&(r.season, r.fpl_fixture_id)].gw.unwrap();

    // A duplicated (element, fixture): keep the row filed under the fixture's GW.
    let mut pairs: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for (i, r) in raw.iter().enumerate() {
        pairs.entry((r.season, r.element_id, r.fpl_fixture_id)).or_default().push(i);
    }
    let mut phantom = vec![false; raw.len()];
    let mut lost = Vec::new();
    for rows in pairs.values().filter(|rows| rows.len() > 1) {
        for &i in rows {
            phantom[i] = raw[i].round != gw_of(&raw[i]);
        }
        if rows.iter().all(|&i| phantom[i]) {
            lost.extend(rows.iter().map(|&i| describe(&raw[i])));
        }
    }
    fail_if(&lost, "duplicate row(s) filed under another GW only (the pair would be lost)")?;
    let dropped = phantom.iter().filter(|&&p| p).count();
    if dropped > 0 {
        log::info!("dropped {dropped} duplicate row(s) filed under another GW");
    }
    let mut remaining: HashMap<(i64, i64, i64), usize> = HashMap::new();
    for (i, r) in raw.iter().enumerate() {
        if !phantom[i] {
            *remaining.entry((r.season, r.element_id, r.fpl_fixture_id)).or_default() += 1;
        }
    }
    let dupes: Vec<String> = raw
        .iter()
        .enumerate()
        .filter(|(i, r)| !phantom[*i] && remaining[&(r.season, r.element_id, r.fpl_fixture_id)] > 1)
        .map(|(_, r)| describe(r))
        .collect();
    fail_if(&dupes, "duplicate (element, fixture) row(s)")?;

    let mut player_keys: HashMap<(i64, i64), i64> = HashMap::new();
    for p in players {
        if player_keys.insert((p.season, p.element_id), p.player_key).is_some() {
            bail!("players are not unique on (season, element_id): {} {}", p.season, p.element_id);
        }
    }
    let lockdown: HashMap<(i64, i64), Micros> =
        gameweek.iter().map(|g| ((g.season, g.gw), g.lockdown_time)).collect();
    let (mut unknown, mut opponent_bad, mut name_bad, mut no_gw) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut out = Vec::new();
    for (i, r) in raw.into_iter().enumerate() {
        if phantom[i] {
            continue;
        }
        let f = fx[&(r.season, r.fpl_fixture_id)];
        let Some(&player_key) = player_keys.get(&(r.season, r.element_id)) else {
            unknown.push(describe(&r));
            continue;
        };
        let (team_key, opponent_team_key) = if r.was_home {
            (f.home_team_key, f.away_team_key)
        } else {
            (f.away_team_key, f.home_team_key)
        };
        let opponent_code =
            team_codes.get(&r.season).and_then(|m| m.get(&r.opponent_id)).copied().unwrap_or(-1);
        if opponent_code != opponent_team_key {
            opponent_bad.push(describe(&r));
        }
        if let Some(name) = &r.team_name
            && resolver.find("fpl_name", name) != Some(team_key)
        {
            name_bad.push(describe(&r));
        }
        let gw = f.gw.unwrap();
        let mut optional_ints = r.optional_ints;
        let mut optional_floats = r.optional_floats;
        if gw < late_columns_first_gw(r.season) {
            optional_ints[0] = None; // starts
            optional_floats = [None; 3];
        }
        let Some(&available_at) = lockdown.get(&(r.season, gw)) else {
            no_gw.push(describe(&r));
            continue;
        };
        let kickoff = f.kickoff_time.context("played fixture without a kickoff")?;
        out.push(PlayerMatch {
            player_key,
            season: r.season,
            fixture_key: f.fixture_key,
            gw,
            element_id: r.element_id,
            team_key,
            opponent_team_key,
            was_home: r.was_home,
            kickoff_time: kickoff,
            counts: r.counts,
            optional_ints,
            optional_floats,
            understat: [None; 8],
            rescored_points: 0,
            source: r.source.to_string(),
            event_time: kickoff,
            available_at,
            gw_state: r.gw_state,
        });
    }
    fail_if(&unknown, "row(s) whose element is not a registered player")?;
    fail_if(&opponent_bad, "opponent_team mismatch(es)")?;
    fail_if(&name_bad, "`team` name mismatch(es)")?;
    fail_if(&no_gw, "row(s) without a gameweek")?;
    Ok(out)
}

/// (`assemble_player_match` over every source season, `registered_players`).
fn assembled_match_rows(ctx: &mut BuildContext) -> Result<(Vec<PlayerMatch>, Vec<Registered>)> {
    let fixtures = Fixture::from_table(ctx.table("fixture")?)?;
    let players = registered_players(ctx)?;
    let resolver = TeamResolver::new(ctx.table("team_dim")?)?;
    let mut raw = Vec::new();
    let mut team_codes = HashMap::new();
    for (season, dir) in vaastav_seasons(&vaastav_run(ctx)?)? {
        raw.extend(vaastav_match_rows(season, &dir)?);
        team_codes.insert(season, season_team_codes(&dir)?);
    }
    for season in current_seasons(ctx)? {
        let (rows, codes) = element_summary_match_rows(ctx, season, &fixtures)?;
        raw.extend(rows);
        team_codes.insert(season, codes);
    }
    let gameweek = Gameweek::from_table(ctx.table("gameweek")?)?;
    let rows = assemble_player_match(raw, &fixtures, &gameweek, &players, &team_codes, &resolver)?;
    Ok((rows, players))
}

// --- player_match ----------------------------------------------------------------------------

fn check_goal_sums(rows: &[PlayerMatch], fixtures: &[Fixture]) -> Result<()> {
    // fixture -> [home goals_scored, away goals_scored, home own_goals, away own_goals]
    let mut sums: BTreeMap<i64, [i64; 4]> = BTreeMap::new();
    for r in rows {
        let s = sums.entry(r.fixture_key).or_default();
        let side = usize::from(!r.was_home);
        s[side] += r.count("goals_scored");
        s[2 + side] += r.count("own_goals");
    }
    let goals: HashMap<i64, (Option<i64>, Option<i64>)> =
        fixtures.iter().map(|f| (f.fixture_key, (f.home_goals, f.away_goals))).collect();
    let mut mismatches = Vec::new();
    for (key, s) in sums {
        let (home_sum, away_sum) = (s[0] + s[3], s[1] + s[2]);
        let (hg, ag) = goals.get(&key).copied().unwrap_or((None, None));
        if hg != Some(home_sum) || ag != Some(away_sum) {
            mismatches
                .push(format!("fixture {key}: sums {home_sum}-{away_sum}, result {hg:?}-{ag:?}"));
        }
    }
    if !mismatches.is_empty() {
        log::warn!(
            "{} fixture(s) with goal-sum mismatches:\n{}",
            mismatches.len(),
            mismatches.join("\n")
        );
    }
    if mismatches.len() > MAX_GOAL_SUM_MISMATCHES {
        bail!(
            "{} fixture(s) whose result != players' goals + opponents' own goals (allowed {MAX_GOAL_SUM_MISMATCHES}):\n{}",
            mismatches.len(),
            mismatches.iter().take(20).cloned().collect::<Vec<_>>().join("\n")
        );
    }
    Ok(())
}

/// (season, gw, column) where goals were scored but every non-null value of an optional
/// stat column is 0 (a source placeholder).
fn check_placeholder_zeros(rows: &[PlayerMatch]) -> Result<()> {
    // Optional stats: starts, the defensive counts, the FPL expected stats.
    let names: Vec<&str> =
        OPTIONAL_INTS.iter().copied().chain(OPTIONAL_FLOATS.iter().map(|(_, n)| *n)).collect();
    let mut by_gw: BTreeMap<(i64, i64), (i64, Vec<(bool, bool)>)> = BTreeMap::new();
    for r in rows {
        let entry =
            by_gw.entry((r.season, r.gw)).or_insert_with(|| (0, vec![(false, false); names.len()]));
        entry.0 += r.count("goals_scored");
        let values: Vec<Option<f64>> = r
            .optional_ints
            .iter()
            .map(|v| v.map(|x| x as f64))
            .chain(r.optional_floats.iter().copied())
            .collect();
        for (j, v) in values.iter().enumerate() {
            if let Some(v) = v {
                entry.1[j].0 = true; // present
                if *v != 0.0 {
                    entry.1[j].1 = true; // nonzero
                }
            }
        }
    }
    let mut blocks = Vec::new();
    for (j, name) in names.iter().enumerate() {
        for ((season, gw), (goals, flags)) in &by_gw {
            if *goals > 0 && flags[j].0 && !flags[j].1 {
                blocks.push(format!("{season} {gw} {name}"));
            }
        }
    }
    if !blocks.is_empty() {
        bail!(
            "{} placeholder-zero block(s): goals scored but the column is 0 for every player of the GW (null it, see LATE_COLUMNS_FIRST_GW):\n{}",
            blocks.len(),
            blocks.iter().take(20).cloned().collect::<Vec<_>>().join("\n")
        );
    }
    Ok(())
}

fn validate_player_match(rows: &[PlayerMatch]) -> Result<()> {
    let mut keys = HashSet::new();
    for r in rows {
        let counts_ok = COUNT_STATS
            .iter()
            .enumerate()
            .all(|(j, name)| SIGNED_STATS.contains(name) || r.counts[j] >= 0);
        let ok = keys.insert((r.player_key, r.fixture_key))
            && r.team_key != r.opponent_team_key
            && r.event_time == r.kickoff_time
            && r.available_at > r.event_time
            && (0..=MAX_MINUTES).contains(&r.counts[0])
            && counts_ok
            && r.optional_ints[0].is_none_or(|s| s == 0 || s == 1)
            && r.optional_ints[1..].iter().all(|v| v.is_none_or(|v| v >= 0))
            && r.optional_floats.iter().all(|v| v.is_none_or(|v| v >= 0.0))
            && r.understat.iter().all(|v| v.is_none_or(|v| v >= 0.0))
            && r.player_key > 0
            && r.element_id > 0
            && (1..=47).contains(&r.gw)
            && r.season >= FIRST_SEASON;
        if !ok {
            bail!(
                "table 'player_match' failed validation at player {} fixture {}",
                r.player_key,
                r.fixture_key
            );
        }
    }
    Ok(())
}

pub fn validated_player_match_table(rows: &[PlayerMatch]) -> Result<Table> {
    validate_player_match(rows)?;
    Ok(player_match_table(rows))
}

pub fn build_player_match(ctx: &mut BuildContext) -> Result<Table> {
    let (mut rows, _) = assembled_match_rows(ctx)?;
    let fixtures = Fixture::from_table(ctx.table("fixture")?)?;
    check_goal_sums(&rows, &fixtures)?;
    check_placeholder_zeros(&rows)?;
    let player_season = PlayerSeason::from_table(ctx.table("player_season")?)?;
    let listed: HashMap<(i64, i64), (Micros, i64)> = player_season
        .iter()
        .map(|p| ((p.season, p.player_key), (p.available_at, p.element_type)))
        .collect();
    // available_at waits for the player's first listing.
    let mut later: BTreeMap<i64, usize> = BTreeMap::new();
    let mut without = Vec::new();
    for r in &mut rows {
        match listed.get(&(r.season, r.player_key)) {
            None => without.push(format!("season {} player {}", r.season, r.player_key)),
            Some(&(at, _)) => {
                if at > r.available_at {
                    *later.entry(r.season).or_default() += 1;
                    r.available_at = at;
                }
            }
        }
    }
    fail_if(&without, "row(s) without a player_season row")?;
    if !later.is_empty() {
        log::info!(
            "player_match: {} row(s) wait for the player's first listing (per season: {later:?})",
            later.values().sum::<usize>()
        );
    }
    // Points re-scored under each season's backtest rules.
    let config_dir = ctx.config_dir.join("scoring");
    let mut rules_by_season = HashMap::new();
    for season in rows.iter().map(|r| r.season).collect::<BTreeSet<_>>() {
        let rules = backtest_rules(season, &config_dir).map_err(|err| {
            let label = season_label(season as i32);
            anyhow::anyhow!(
                "season {label} has no backtest rules to re-score its points ({err}): run `fplopt rules export {label}`, add config/scoring/{label}.supplement.json and add {season} to fplopt.backtest.rules.NATIVE_SEASONS (now {NATIVE_SEASONS:?})"
            )
        })?;
        rules_by_season.insert(season, rules);
    }
    for r in &mut rows {
        let line = StatLine {
            element_type: listed[&(r.season, r.player_key)].1,
            minutes: r.count("minutes"),
            goals_scored: r.count("goals_scored"),
            assists: r.count("assists"),
            clean_sheets: r.count("clean_sheets"),
            goals_conceded: r.count("goals_conceded"),
            own_goals: r.count("own_goals"),
            penalties_saved: r.count("penalties_saved"),
            penalties_missed: r.count("penalties_missed"),
            yellow_cards: r.count("yellow_cards"),
            red_cards: r.count("red_cards"),
            saves: r.count("saves"),
            bonus: r.count("bonus"),
            defensive_contribution: r.opt_int("defensive_contribution"),
            clearances_blocks_interceptions: r.opt_int("clearances_blocks_interceptions"),
            tackles: r.opt_int("tackles"),
            recoveries: r.opt_int("recoveries"),
        };
        r.rescored_points = score(&line, &rules_by_season[&r.season])?;
    }
    validated_player_match_table(&rows)
}

// --- player_gw / player_gw_ownership ---------------------------------------------------------

/// One row per (player_key, season, gw): club, position, GW state and the GW's deadline.
struct GwRow {
    player_key: i64,
    season: i64,
    gw: i64,
    team_key: i64,
    element_type: i64,
    gw_index: i64,
    deadline_time: Micros,
    state: [i64; 4],
}

fn player_gw_rows(ctx: &mut BuildContext) -> Result<Vec<GwRow>> {
    let (rows, players) = assembled_match_rows(ctx)?;
    let gameweek = Gameweek::from_table(ctx.table("gameweek")?)?;
    let mut groups: BTreeMap<(i64, i64, i64), Vec<&PlayerMatch>> = BTreeMap::new();
    for r in &rows {
        groups.entry((r.player_key, r.season, r.gw)).or_default().push(r);
    }
    let mut differ = Vec::new();
    for (key, rows) in &groups {
        let first = rows[0];
        if rows.iter().any(|r| r.team_key != first.team_key || r.gw_state != first.gw_state) {
            differ.push(format!("{key:?}"));
        }
    }
    if !differ.is_empty() {
        bail!(
            "{} (player, season, GW)(s) whose rows differ within a GW:\n{}",
            differ.len(),
            differ.iter().take(10).cloned().collect::<Vec<_>>().join("\n")
        );
    }
    let mut missing = BTreeSet::new();
    for rows in groups.values() {
        for (j, v) in rows[0].gw_state.iter().enumerate() {
            if v.is_none() {
                missing.insert(GW_STATE[j]);
            }
        }
    }
    if !missing.is_empty() {
        bail!("player GW row(s) without {}", missing.into_iter().collect::<Vec<_>>().join(", "));
    }
    let positions: HashMap<(i64, i64), i64> =
        players.iter().map(|p| ((p.season, p.player_key), p.element_type)).collect();
    let gws: HashMap<(i64, i64), (i64, Micros)> =
        gameweek.iter().map(|g| ((g.season, g.gw), (g.gw_index, g.deadline_time))).collect();
    let mut out = Vec::new();
    let mut no_gw = Vec::new();
    for ((player_key, season, gw), rows) in groups {
        let Some(&(gw_index, deadline_time)) = gws.get(&(season, gw)) else {
            no_gw.push(format!("{player_key} {season} {gw}"));
            continue;
        };
        out.push(GwRow {
            player_key,
            season,
            gw,
            team_key: rows[0].team_key,
            element_type: *positions
                .get(&(season, player_key))
                .context("player GW row without a position")?,
            gw_index,
            deadline_time,
            state: rows[0].gw_state.map(Option::unwrap),
        });
    }
    fail_if(&no_gw, "player GW row(s) whose GW is not in gameweek")?;
    Ok(out)
}

/// Per player and season, the first snapshot of each run of consecutive snapshots listing
/// him with the same club: (player_key, season) -> [(snapshot_at, team_key)] by time.
type Stints = HashMap<(i64, i64), Vec<(Micros, i64)>>;

fn read_registration_stints(ctx: &BuildContext) -> Result<(Stints, Option<Micros>)> {
    let path = ctx.table_path("player_snapshot");
    if !path.exists() {
        bail!(
            "table 'player_snapshot' has not been built ({}); player_gw needs it: run `fplopt build player_snapshot` first",
            path.display()
        );
    }
    let t = read_parquet(&path, Some(&["snapshot_at", "season", "player_key", "team_key"]))?;
    let (at, season, key, team) =
        (t.i64("snapshot_at")?, t.i64("season")?, t.i64("player_key")?, t.i64("team_key")?);
    let start = at.iter().min().copied();
    let mut order: Vec<usize> = (0..t.height()).collect();
    order.sort_by_key(|&i| (key[i], season[i], at[i], i));
    let mut stints: Stints = HashMap::new();
    let mut previous: Option<(i64, i64, i64)> = None;
    for i in order {
        let current = (key[i], season[i], team[i]);
        if previous != Some(current) {
            stints.entry((key[i], season[i])).or_default().push((at[i], team[i]));
        }
        previous = Some(current);
    }
    Ok((stints, start))
}

/// season, team_key -> the GW index of the club's first fixture of the season.
fn club_first_gameweeks(fixtures: &[Fixture]) -> HashMap<(i64, i64), i64> {
    let mut out: HashMap<(i64, i64), i64> = HashMap::new();
    for f in fixtures {
        let Some(gw_index) = f.gw_index else { continue };
        for team in [f.home_team_key, f.away_team_key] {
            let e = out.entry((f.season, team)).or_insert(gw_index);
            *e = (*e).min(gw_index);
        }
    }
    out
}

/// When each player GW row was public (the player registered with that club); see the
/// Python docstring of `registration_available_at`.
fn registration_available_at(
    gw: &[GwRow],
    stints: &Stints,
    coverage_start: Option<Micros>,
    club_first: &HashMap<(i64, i64), i64>,
) -> Vec<Micros> {
    let known: Vec<Micros> = gw.iter().map(|r| r.deadline_time - KNOWN_BEFORE_DEADLINE).collect();
    let mut out = known.clone();
    let with_snapshots: Vec<bool> =
        known.iter().map(|k| coverage_start.is_some_and(|s| *k >= s)).collect();

    let mut early: Vec<usize> = (0..gw.len()).filter(|&i| !with_snapshots[i]).collect();
    early.sort_by_key(|&i| (gw[i].player_key, gw[i].season, gw[i].gw_index, i));
    let mut previous: Option<(i64, i64, i64)> = None; // player, season, team
    for i in early {
        let r = &gw[i];
        let previous_team =
            previous.filter(|p| p.0 == r.player_key && p.1 == r.season).map(|p| p.2);
        let first_gw = club_first.get(&(r.season, r.team_key)).copied();
        let joined_late = previous_team.is_none() && first_gw != Some(r.gw_index);
        let moved = previous_team.is_some_and(|t| t != r.team_key);
        if joined_late || moved {
            out[i] = r.deadline_time;
        }
        previous = Some((r.player_key, r.season, r.team_key));
    }

    for i in (0..gw.len()).filter(|&i| with_snapshots[i]) {
        let r = &gw[i];
        let empty = Vec::new();
        let player_stints = stints.get(&(r.player_key, r.season)).unwrap_or(&empty);
        // Newest stint listed by `known` (backward as-of).
        let listed_team =
            player_stints.iter().rev().find(|(at, _)| *at <= known[i]).map(|(_, t)| *t);
        out[i] = if listed_team == Some(r.team_key) {
            known[i]
        } else {
            // First later stint with that club (forward as-of), else the deadline.
            player_stints
                .iter()
                .find(|(at, team)| *team == r.team_key && *at >= known[i])
                .map(|(at, _)| *at)
                .unwrap_or(r.deadline_time)
        };
    }
    out
}

pub fn build_player_gw(ctx: &mut BuildContext) -> Result<Table> {
    let gw = player_gw_rows(ctx)?;
    let (stints, coverage_start) = read_registration_stints(ctx)?;
    let fixtures = Fixture::from_table(ctx.table("fixture")?)?;
    let available_at =
        registration_available_at(&gw, &stints, coverage_start, &club_first_gameweeks(&fixtures));
    let mut late: BTreeMap<i64, usize> = BTreeMap::new();
    for (r, at) in gw.iter().zip(&available_at) {
        if *at > r.deadline_time - KNOWN_BEFORE_DEADLINE {
            *late.entry(r.season).or_default() += 1;
        }
    }
    if !late.is_empty() {
        log::info!(
            "player_gw: {} row(s) not known an hour before their deadline (per season: {late:?})",
            late.values().sum::<usize>()
        );
    }
    for (r, at) in gw.iter().zip(&available_at) {
        if !(1..=4).contains(&r.element_type)
            || r.state[0] <= 0
            || *at < r.deadline_time - KNOWN_BEFORE_DEADLINE
        {
            bail!(
                "table 'player_gw' failed validation at player {} {} GW{}",
                r.player_key,
                r.season,
                r.gw
            );
        }
    }
    Ok(Table::new()
        .with("player_key", Column::I64(gw.iter().map(|r| r.player_key).collect()))
        .with("season", Column::I64(gw.iter().map(|r| r.season).collect()))
        .with("gw", Column::I64(gw.iter().map(|r| r.gw).collect()))
        .with("team_key", Column::I64(gw.iter().map(|r| r.team_key).collect()))
        .with("element_type", Column::I64(gw.iter().map(|r| r.element_type).collect()))
        .with("value", Column::I64(gw.iter().map(|r| r.state[0]).collect()))
        .with("event_time", Column::Ts(gw.iter().map(|r| r.deadline_time).collect()))
        .with("available_at", Column::Ts(available_at)))
}

pub fn build_player_gw_ownership(ctx: &mut BuildContext) -> Result<Table> {
    let gw = player_gw_rows(ctx)?;
    let player_gw = ctx.table("player_gw")?;
    let (keys, seasons, gws, available) = (
        player_gw.i64("player_key")?,
        player_gw.i64("season")?,
        player_gw.i64("gw")?,
        player_gw.i64("available_at")?,
    );
    let known: HashMap<(i64, i64, i64), Micros> =
        (0..player_gw.height()).map(|i| ((keys[i], seasons[i], gws[i]), available[i])).collect();
    let mut available_at = Vec::with_capacity(gw.len());
    let mut without = Vec::new();
    for r in &gw {
        match known.get(&(r.player_key, r.season, r.gw)) {
            Some(&k) => available_at.push(if k > r.deadline_time { k } else { r.deadline_time }),
            None => {
                without.push(format!("{} {} {}", r.player_key, r.season, r.gw));
                available_at.push(r.deadline_time);
            }
        }
    }
    fail_if(&without, "ownership row(s) without a player_gw row")?;
    if gw.iter().any(|r| r.state[1..].iter().any(|&v| v < 0)) {
        bail!("table 'player_gw_ownership' failed validation: negative ownership counts");
    }
    Ok(Table::new()
        .with("player_key", Column::I64(gw.iter().map(|r| r.player_key).collect()))
        .with("season", Column::I64(gw.iter().map(|r| r.season).collect()))
        .with("gw", Column::I64(gw.iter().map(|r| r.gw).collect()))
        .with("selected", Column::I64(gw.iter().map(|r| r.state[1]).collect()))
        .with("transfers_in", Column::I64(gw.iter().map(|r| r.state[2]).collect()))
        .with("transfers_out", Column::I64(gw.iter().map(|r| r.state[3]).collect()))
        .with("event_time", Column::Ts(gw.iter().map(|r| r.deadline_time).collect()))
        .with("available_at", Column::Ts(available_at)))
}
