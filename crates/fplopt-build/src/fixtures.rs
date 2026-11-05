//! `fixture`, `gameweek`, `gameweek_result`, `schedule` and `fixture_snapshot`. Port of
//! `fplopt.build.fixtures` (its module docstring describes sources, timing and checks).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use chrono::{DateTime, NaiveDate, Utc};
use fplopt_archive::raw_store::RawStore;
use fplopt_core::seasons::{bootstrap_season, parse_season_label, season_start_year};
use serde_json::Value;

use crate::common::{
    Bootstrap, BuildContext, Csv, UNSCHEDULED_AT, date_from_days, days_from_date, file_name,
    fplcache_and_own_bootstraps, from_micros, json_bool, json_i64, json_opt_i64,
    latest_complete_run, latest_entry, lockdown_time, micros, parse_bool, parse_int, parse_iso,
    parse_opt_int, schedule_available_at, subdirs, uk_date,
};
use crate::table::{Column, Days, Micros, Table};
use crate::teams::TeamResolver;

pub const FIRST_SEASON: i64 = 2016;
pub const FIXTURES_PER_SEASON: usize = 380;
pub const MATCHES_PER_TEAM: usize = 38;
const APPROX_DEADLINE_LEAD: Micros = 90 * 60 * 1_000_000;

pub const FIXTURE_SORT_BY: &[&str] = &["fixture_key"];
pub const GAMEWEEK_SORT_BY: &[&str] = &["season", "gw"];
pub const GAMEWEEK_RESULT_SORT_BY: &[&str] = &["season", "gw"];
pub const SCHEDULE_SORT_BY: &[&str] = &["fixture_key"];
pub const FIXTURE_SNAPSHOT_SORT_BY: &[&str] = &["snapshot_at", "fixture_key"];

// --- shared helpers --------------------------------------------------------------------------

pub fn vaastav_run(ctx: &BuildContext) -> Result<PathBuf> {
    latest_complete_run(&ctx.store, "vaastav", "data")
}

/// Season start year -> season folder of a vaastav run (2016 on).
pub fn vaastav_seasons(run: &Path) -> Result<BTreeMap<i64, PathBuf>> {
    let mut seasons = BTreeMap::new();
    for path in subdirs(run)? {
        if let Ok(season) = parse_season_label(file_name(&path)) {
            let season = i64::from(season);
            if season >= FIRST_SEASON {
                seasons.insert(season, path);
            }
        }
    }
    Ok(seasons)
}

/// FPL team id -> team code of a vaastav season folder (`teams.csv`, else `players_raw`).
pub fn season_team_codes(season_dir: &Path) -> Result<HashMap<i64, i64>> {
    let teams_path = season_dir.join("teams.csv.gz");
    let mut pairs: Vec<(i64, i64)> = Vec::new();
    if teams_path.exists() {
        let teams = Csv::read(&teams_path)?;
        for (id, code) in teams.cells("id").into_iter().zip(teams.cells("code")) {
            pairs.push((parse_int(id, "id")?, parse_int(code, "code")?));
        }
    } else {
        let players = Csv::read(&season_dir.join("players_raw.csv.gz"))?;
        let mut seen = HashSet::new();
        for (team, code) in players.cells("team").into_iter().zip(players.cells("team_code")) {
            let pair = (parse_int(team, "team")?, parse_int(code, "team_code")?);
            if seen.insert(pair) {
                pairs.push(pair);
            }
        }
    }
    let teams: HashSet<i64> = pairs.iter().map(|p| p.0).collect();
    let codes: HashSet<i64> = pairs.iter().map(|p| p.1).collect();
    if teams.len() != pairs.len() || codes.len() != pairs.len() {
        bail!("{}: team id -> code is not 1:1", file_name(season_dir));
    }
    Ok(pairs.into_iter().collect())
}

pub fn bootstrap_team_codes(bootstrap: &Value) -> Result<HashMap<i64, i64>> {
    let mut out = HashMap::new();
    for team in bootstrap["teams"].as_array().context("bootstrap without teams")? {
        out.insert(json_i64(&team["id"], "team id")?, json_i64(&team["code"], "team code")?);
    }
    Ok(out)
}

/// The newest of time-ordered `entries` for each season, by binary search per season (the
/// same few files Python reads).
pub fn newest_per_season<T: Clone>(
    entries: &[T],
    mut season_of: impl FnMut(&T) -> Result<i64>,
) -> Result<BTreeMap<i64, T>> {
    let mut seasons: HashMap<usize, i64> = HashMap::new();
    let mut season_at = |i: usize, seasons: &mut HashMap<usize, i64>| -> Result<i64> {
        if let Some(&s) = seasons.get(&i) {
            return Ok(s);
        }
        let s = season_of(&entries[i])?;
        seasons.insert(i, s);
        Ok(s)
    };
    let mut result = BTreeMap::new();
    if entries.is_empty() {
        return Ok(result);
    }
    let first = season_at(0, &mut seasons)?;
    let last = season_at(entries.len() - 1, &mut seasons)?;
    for season in first..=last {
        let (mut lo, mut hi) = (0usize, entries.len());
        while lo < hi {
            let mid = (lo + hi) / 2;
            if season_at(mid, &mut seasons)? <= season {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo >= 1 && season_at(lo - 1, &mut seasons)? == season {
            result.insert(season, entries[lo - 1].clone());
        }
    }
    Ok(result)
}

/// The newest archived bootstrap describing each season.
pub fn last_bootstrap_per_season(store: &RawStore) -> Result<BTreeMap<i64, Bootstrap>> {
    let entries = fplcache_and_own_bootstraps(store)?;
    newest_per_season(&entries, |entry| {
        Ok(i64::from(bootstrap_season(&RawStore::read_json(&entry.path)?)?))
    })
}

/// Season of an FPL fixtures payload: start year of its earliest kickoff (else the
/// snapshot's own season).
pub fn fixtures_snapshot_season(records: &[Value], snapshot_at: &DateTime<Utc>) -> Result<i64> {
    let kickoffs: Vec<&str> = records
        .iter()
        .filter_map(|r| r.get("kickoff_time").and_then(Value::as_str))
        .filter(|s| !s.is_empty())
        .collect();
    match kickoffs.iter().min() {
        None => Ok(i64::from(season_start_year(snapshot_at))),
        Some(first) => Ok(i64::from(season_start_year(&from_micros(parse_iso(first)?)))),
    }
}

pub fn own_fixtures_per_season(
    store: &RawStore,
) -> Result<BTreeMap<i64, (DateTime<Utc>, PathBuf)>> {
    let entries = store.entries("fpl", "fixtures", ".json.gz")?;
    newest_per_season(&entries, |(at, path)| {
        let records = RawStore::read_json(path)?;
        fixtures_snapshot_season(records.as_array().context("fixtures payload is not a list")?, at)
    })
}

// --- fixture ---------------------------------------------------------------------------------

/// A raw fixture row before keys and timing.
#[derive(Clone, Debug)]
struct RawFixture {
    season: i64,
    fpl_fixture_id: i64,
    gw: Option<i64>,
    kickoff_time: Option<Micros>,
    home_team_key: i64,
    away_team_key: i64,
    home_goals: Option<i64>,
    away_goals: Option<i64>,
    finished: bool,
}

/// A `fixture` row.
#[derive(Clone, Debug)]
pub struct Fixture {
    pub fixture_key: i64,
    pub season: i64,
    pub fpl_fixture_id: i64,
    pub gw: Option<i64>,
    pub gw_index: Option<i64>,
    pub kickoff_time: Option<Micros>,
    pub home_team_key: i64,
    pub away_team_key: i64,
    pub home_goals: Option<i64>,
    pub away_goals: Option<i64>,
    pub finished: bool,
    pub fd_date: Option<Days>,
    pub event_time: Micros,
    pub available_at: Micros,
}

impl Fixture {
    pub fn from_table(t: &Table) -> Result<Vec<Fixture>> {
        let (keys, seasons, ids) =
            (t.i64("fixture_key")?, t.i64("season")?, t.i64("fpl_fixture_id")?);
        let (gw, gw_index, kickoff) =
            (t.opt_i64("gw")?, t.opt_i64("gw_index")?, t.opt_i64("kickoff_time")?);
        let (home, away) = (t.i64("home_team_key")?, t.i64("away_team_key")?);
        let (hg, ag) = (t.opt_i64("home_goals")?, t.opt_i64("away_goals")?);
        let (finished, fd_date) = (t.bool("finished")?, t.opt_date("fd_date")?);
        let (event_time, available_at) = (t.i64("event_time")?, t.i64("available_at")?);
        Ok((0..t.height())
            .map(|i| Fixture {
                fixture_key: keys[i],
                season: seasons[i],
                fpl_fixture_id: ids[i],
                gw: gw[i],
                gw_index: gw_index[i],
                kickoff_time: kickoff[i],
                home_team_key: home[i],
                away_team_key: away[i],
                home_goals: hg[i],
                away_goals: ag[i],
                finished: finished[i],
                fd_date: fd_date[i],
                event_time: event_time[i],
                available_at: available_at[i],
            })
            .collect())
    }
}

fn fixture_table(rows: &[Fixture]) -> Table {
    Table::new()
        .with("fixture_key", Column::I64(rows.iter().map(|r| r.fixture_key).collect()))
        .with("season", Column::I64(rows.iter().map(|r| r.season).collect()))
        .with("fpl_fixture_id", Column::I64(rows.iter().map(|r| r.fpl_fixture_id).collect()))
        .with("gw", Column::OptI64(rows.iter().map(|r| r.gw).collect()))
        .with("gw_index", Column::OptI64(rows.iter().map(|r| r.gw_index).collect()))
        .with("kickoff_time", Column::OptTs(rows.iter().map(|r| r.kickoff_time).collect()))
        .with("home_team_key", Column::I64(rows.iter().map(|r| r.home_team_key).collect()))
        .with("away_team_key", Column::I64(rows.iter().map(|r| r.away_team_key).collect()))
        .with("home_goals", Column::OptI64(rows.iter().map(|r| r.home_goals).collect()))
        .with("away_goals", Column::OptI64(rows.iter().map(|r| r.away_goals).collect()))
        .with("finished", Column::Bool(rows.iter().map(|r| r.finished).collect()))
        .with("fd_date", Column::OptDate(rows.iter().map(|r| r.fd_date).collect()))
        .with("event_time", Column::Ts(rows.iter().map(|r| r.event_time).collect()))
        .with("available_at", Column::Ts(rows.iter().map(|r| r.available_at).collect()))
}

fn code(team_codes: &HashMap<i64, i64>, id: i64) -> Option<i64> {
    team_codes.get(&id).copied()
}

fn check_ids(ids: impl Iterator<Item = i64>, team_codes: &HashMap<i64, i64>) -> Result<()> {
    let unknown: BTreeSet<i64> = ids.filter(|id| !team_codes.contains_key(id)).collect();
    if !unknown.is_empty() {
        bail!("team id(s) {:?} missing from the season's team list", unknown);
    }
    Ok(())
}

/// Raw fixture rows of a season's merged_gw (2016-17, 2017-18 have no fixtures.csv).
fn fixtures_from_merged_gw(
    merged: &Csv,
    season: i64,
    team_codes: &HashMap<i64, i64>,
) -> Result<Vec<RawFixture>> {
    let columns = [
        "fixture",
        "round",
        "kickoff_time",
        "was_home",
        "opponent_team",
        "team_h_score",
        "team_a_score",
    ];
    let idx: Vec<usize> = columns.iter().map(|c| merged.require(c)).collect::<Result<_>>()?;
    let mut seen = HashSet::new();
    struct Row {
        fixture: i64,
        round: Option<i64>,
        kickoff: Option<String>,
        was_home: bool,
        opponent: i64,
        h: Option<i64>,
        a: Option<i64>,
    }
    let mut rows = Vec::new();
    for row in &merged.rows {
        let cells: Vec<Option<String>> = idx.iter().map(|&i| row[i].clone()).collect();
        if !seen.insert(cells.clone()) {
            continue;
        }
        rows.push(Row {
            fixture: parse_int(cells[0].as_deref(), "fixture")?,
            round: parse_opt_int(cells[1].as_deref(), "round")?,
            kickoff: cells[2].clone(),
            was_home: parse_bool(cells[3].as_deref(), "was_home")?,
            opponent: parse_int(cells[4].as_deref(), "opponent_team")?,
            h: parse_opt_int(cells[5].as_deref(), "team_h_score")?,
            a: parse_opt_int(cells[6].as_deref(), "team_a_score")?,
        });
    }
    let mut by_fixture: BTreeMap<i64, Vec<&Row>> = BTreeMap::new();
    for row in &rows {
        by_fixture.entry(row.fixture).or_default().push(row);
    }
    fn distinct<T: Eq + std::hash::Hash + Clone>(values: impl Iterator<Item = Option<T>>) -> usize {
        values.flatten().collect::<HashSet<T>>().len()
    }
    for (column, bad) in [
        (
            "round",
            by_fixture
                .iter()
                .filter(|(_, r)| distinct(r.iter().map(|x| x.round)) > 1)
                .map(|(k, _)| *k)
                .collect::<Vec<_>>(),
        ),
        (
            "kickoff_time",
            by_fixture
                .iter()
                .filter(|(_, r)| distinct(r.iter().map(|x| x.kickoff.clone())) > 1)
                .map(|(k, _)| *k)
                .collect(),
        ),
        (
            "team_h_score",
            by_fixture
                .iter()
                .filter(|(_, r)| distinct(r.iter().map(|x| x.h)) > 1)
                .map(|(k, _)| *k)
                .collect(),
        ),
        (
            "team_a_score",
            by_fixture
                .iter()
                .filter(|(_, r)| distinct(r.iter().map(|x| x.a)) > 1)
                .map(|(k, _)| *k)
                .collect(),
        ),
    ] {
        if !bad.is_empty() {
            bail!("merged_gw: {column} differs within fixture(s) {bad:?}");
        }
    }
    let only = |rows: &[&Row], home: bool| -> Result<Option<i64>> {
        let ids: BTreeSet<i64> =
            rows.iter().filter(|r| r.was_home == home).map(|r| r.opponent).collect();
        match ids.len() {
            0 => Ok(None),
            1 => Ok(ids.into_iter().next()),
            _ => bail!("merged_gw: ambiguous opponent ids {:?}", ids),
        }
    };
    let mut out = Vec::new();
    let mut missing = Vec::new();
    for (&fixture, rows) in &by_fixture {
        // The home side's rows name the away club as their opponent and vice versa.
        let away_id = only(rows, true)?;
        let home_id = only(rows, false)?;
        let (Some(home_id), Some(away_id)) = (home_id, away_id) else {
            missing.push(fixture);
            continue;
        };
        let first = |f: &dyn Fn(&Row) -> Option<i64>| rows.iter().find_map(|r| f(r));
        let kickoff = rows.iter().find_map(|r| r.kickoff.clone());
        out.push((
            fixture,
            first(&|r| r.round),
            kickoff,
            home_id,
            away_id,
            first(&|r| r.h),
            first(&|r| r.a),
        ));
    }
    if !missing.is_empty() {
        bail!("merged_gw: fixture(s) {missing:?} lack one side's rows");
    }
    check_ids(out.iter().flat_map(|r| [r.3, r.4]), team_codes)?;
    out.into_iter()
        .map(|(fixture, gw, kickoff, home, away, h, a)| {
            Ok(RawFixture {
                season,
                fpl_fixture_id: fixture,
                gw,
                kickoff_time: kickoff.as_deref().map(parse_iso).transpose()?,
                home_team_key: code(team_codes, home).unwrap(),
                away_team_key: code(team_codes, away).unwrap(),
                home_goals: h,
                away_goals: a,
                finished: true,
            })
        })
        .collect()
}

/// One fixture of FPL's fixtures endpoint shape (vaastav fixtures.csv or our snapshots).
struct ApiFixture {
    id: i64,
    event: Option<i64>,
    kickoff_time: Option<Micros>,
    team_h: i64,
    team_a: i64,
    team_h_score: Option<i64>,
    team_a_score: Option<i64>,
    finished: bool,
}

fn api_fixtures_from_csv(csv: &Csv) -> Result<Vec<ApiFixture>> {
    let (id, event, kickoff) = (csv.cells("id"), csv.cells("event"), csv.cells("kickoff_time"));
    let (h, a) = (csv.cells("team_h"), csv.cells("team_a"));
    let (hs, as_, finished) =
        (csv.cells("team_h_score"), csv.cells("team_a_score"), csv.cells("finished"));
    (0..csv.len())
        .map(|i| {
            Ok(ApiFixture {
                id: parse_int(id[i], "id")?,
                event: parse_opt_int(event[i], "event")?,
                kickoff_time: kickoff[i].map(parse_iso).transpose()?,
                team_h: parse_int(h[i], "team_h")?,
                team_a: parse_int(a[i], "team_a")?,
                team_h_score: parse_opt_int(hs[i], "team_h_score")?,
                team_a_score: parse_opt_int(as_[i], "team_a_score")?,
                finished: finished[i]
                    .map(|f| parse_bool(Some(f), "finished"))
                    .transpose()?
                    .unwrap_or(true),
            })
        })
        .collect()
}

fn opt_iso(value: &Value) -> Result<Option<Micros>> {
    match value {
        Value::Null => Ok(None),
        Value::String(s) if s.is_empty() => Ok(None),
        Value::String(s) => parse_iso(s).map(Some),
        other => bail!("not a time: {other}"),
    }
}

fn api_fixtures_from_json(records: &[Value]) -> Result<Vec<ApiFixture>> {
    records
        .iter()
        .map(|r| {
            Ok(ApiFixture {
                id: json_i64(&r["id"], "id")?,
                event: json_opt_i64(&r["event"], "event")?,
                kickoff_time: opt_iso(&r["kickoff_time"])?,
                team_h: json_i64(&r["team_h"], "team_h")?,
                team_a: json_i64(&r["team_a"], "team_a")?,
                team_h_score: json_opt_i64(&r["team_h_score"], "team_h_score")?,
                team_a_score: json_opt_i64(&r["team_a_score"], "team_a_score")?,
                finished: json_bool(&r["finished"]),
            })
        })
        .collect()
}

fn fixtures_from_api(
    fixtures: &[ApiFixture],
    season: i64,
    team_codes: &HashMap<i64, i64>,
) -> Result<Vec<RawFixture>> {
    check_ids(fixtures.iter().flat_map(|f| [f.team_h, f.team_a]), team_codes)?;
    Ok(fixtures
        .iter()
        .map(|f| RawFixture {
            season,
            fpl_fixture_id: f.id,
            gw: f.event,
            kickoff_time: f.kickoff_time,
            home_team_key: code(team_codes, f.team_h).unwrap(),
            away_team_key: code(team_codes, f.team_a).unwrap(),
            // Goals only once the result is final (a live match has provisional scores).
            home_goals: f.team_h_score.filter(|_| f.finished),
            away_goals: f.team_a_score.filter(|_| f.finished),
            finished: f.finished,
        })
        .collect())
}

fn raw_fixtures(ctx: &BuildContext) -> Result<Vec<RawFixture>> {
    let run = vaastav_run(ctx)?;
    let mut rows = Vec::new();
    let mut have = HashSet::new();
    for (season, season_dir) in vaastav_seasons(&run)? {
        let team_codes = season_team_codes(&season_dir)?;
        let fixtures_path = season_dir.join("fixtures.csv.gz");
        let raw = if fixtures_path.exists() {
            fixtures_from_api(
                &api_fixtures_from_csv(&Csv::read(&fixtures_path)?)?,
                season,
                &team_codes,
            )?
        } else {
            let merged = Csv::read(&season_dir.join("gws").join("merged_gw.csv.gz"))?;
            fixtures_from_merged_gw(&merged, season, &team_codes)?
        };
        if !raw.is_empty() {
            have.insert(season);
        }
        rows.extend(raw);
    }
    rows.extend(current_season_fixtures(ctx, &have)?);
    Ok(rows)
}

/// Per season vaastav doesn't cover, its newest own fixtures snapshot.
fn current_season_fixtures(ctx: &BuildContext, have: &HashSet<i64>) -> Result<Vec<RawFixture>> {
    let mut rows = Vec::new();
    let mut bootstraps = None;
    for (season, (_, path)) in own_fixtures_per_season(&ctx.store)? {
        if have.contains(&season) {
            log::info!(
                "fixtures snapshot {} is for season {season}, covered by vaastav",
                path.display()
            );
            continue;
        }
        if bootstraps.is_none() {
            bootstraps = Some(last_bootstrap_per_season(&ctx.store)?);
        }
        let entry = bootstraps.as_ref().unwrap().get(&season).with_context(|| {
            format!("no archived bootstrap for season {season} (team ids of {})", path.display())
        })?;
        let team_codes = bootstrap_team_codes(&RawStore::read_json(&entry.path)?)?;
        let records = RawStore::read_json(&path)?;
        let fixtures =
            api_fixtures_from_json(records.as_array().context("fixtures payload is not a list")?)?;
        rows.extend(fixtures_from_api(&fixtures, season, &team_codes)?);
    }
    Ok(rows)
}

/// (season, newest file) per football-data E0 season folder; `first_season` None = all.
pub fn football_data_files(
    ctx: &BuildContext,
    first_season: Option<i64>,
) -> Result<Vec<(i64, PathBuf)>> {
    let base = ctx.root().join("football-data").join("E0");
    let mut out = Vec::new();
    for season_dir in subdirs(&base)? {
        let name = file_name(&season_dir);
        let season = 2000 + name.get(..2).context("short folder name")?.parse::<i64>()?;
        if first_season.is_some_and(|first| season < first) {
            continue;
        }
        if let Some(path) =
            latest_entry(&ctx.store, "football-data", &format!("E0/{name}"), ".csv.gz")?
        {
            out.push((season, path));
        }
    }
    Ok(out)
}

/// One football-data row: season, date, team keys and the requested raw cells.
#[derive(Clone, Debug)]
pub struct FdRow {
    pub season: i64,
    pub fd_date: Days,
    pub home_team_key: i64,
    pub away_team_key: i64,
    pub cells: HashMap<String, Option<String>>,
}

/// football-data `Date`: dd/mm/yy (2016/17 and earlier) or dd/mm/yyyy.
pub fn fd_date(text: &str) -> Result<Days> {
    let text = text.trim();
    let parts: Vec<&str> = text.split('/').collect();
    if parts.len() != 3 {
        bail!("bad football-data date {text:?}");
    }
    let (day, month): (u32, u32) = (parts[0].parse()?, parts[1].parse()?);
    let year: i32 = if text.len() == 10 {
        parts[2].parse()?
    } else {
        let yy: i32 = parts[2].parse()?;
        if parts[2].len() != 2 {
            bail!("bad football-data date {text:?}");
        }
        if yy < 69 { 2000 + yy } else { 1900 + yy }
    };
    let date =
        NaiveDate::from_ymd_opt(year, month, day).with_context(|| format!("bad date {text:?}"))?;
    Ok(days_from_date(date))
}

/// One football-data file's rows (rows without a `Date` dropped), keeping `columns`.
pub fn football_data_frame(
    path: &Path,
    season: i64,
    resolver: &TeamResolver,
    columns: &[&str],
) -> Result<Vec<FdRow>> {
    let csv = Csv::read(path)?.drop_na("Date");
    let (date, home, away) = (csv.cells("Date"), csv.cells("HomeTeam"), csv.cells("AwayTeam"));
    let present: Vec<(&str, usize)> =
        columns.iter().filter_map(|c| csv.index(c).map(|i| (*c, i))).collect();
    let mut rows = Vec::new();
    for i in 0..csv.len() {
        let cells = present.iter().map(|(c, j)| (c.to_string(), csv.rows[i][*j].clone())).collect();
        rows.push(FdRow {
            season,
            fd_date: fd_date(date[i].unwrap())?,
            home_team_key: resolver.football_data(home[i].unwrap_or("nan"))?,
            away_team_key: resolver.football_data(away[i].unwrap_or("nan"))?,
            cells,
        });
    }
    Ok(rows)
}

pub fn football_data_frames(
    ctx: &BuildContext,
    resolver: &TeamResolver,
    columns: &[&str],
    first_season: Option<i64>,
) -> Result<Vec<FdRow>> {
    let mut rows = Vec::new();
    for (season, path) in football_data_files(ctx, first_season)? {
        rows.extend(football_data_frame(&path, season, resolver, columns)?);
    }
    Ok(rows)
}

#[derive(Clone, Debug)]
pub struct FdResult {
    pub season: i64,
    pub fd_date: Days,
    pub home_team_key: i64,
    pub away_team_key: i64,
    pub home_goals: i64,
    pub away_goals: i64,
}

pub fn football_data_results(
    ctx: &BuildContext,
    resolver: &TeamResolver,
    first_season: Option<i64>,
) -> Result<Vec<FdResult>> {
    football_data_frames(ctx, resolver, &["FTHG", "FTAG"], first_season)?
        .into_iter()
        .map(|r| {
            Ok(FdResult {
                season: r.season,
                fd_date: r.fd_date,
                home_team_key: r.home_team_key,
                away_team_key: r.away_team_key,
                home_goals: parse_int(r.cells.get("FTHG").and_then(|c| c.as_deref()), "FTHG")?,
                away_goals: parse_int(r.cells.get("FTAG").and_then(|c| c.as_deref()), "FTAG")?,
            })
        })
        .collect()
}

fn show_rows(rows: &[String], what: &str, problems: &mut Vec<String>) {
    if !rows.is_empty() {
        let sample: Vec<&str> = rows.iter().take(10).map(String::as_str).collect();
        problems.push(format!("{} {what}:\n{}", rows.len(), sample.join("\n")));
    }
}

/// Join football-data results onto fixtures by (season, home, away), cross-checking date
/// and goals; sets `fd_date`.
fn attach_football_data(fixtures: &mut [Fixture], fd: &[FdResult]) -> Result<()> {
    let mut by_key: HashMap<(i64, i64, i64), &FdResult> = HashMap::new();
    let mut duplicates = Vec::new();
    for r in fd {
        if by_key.insert((r.season, r.home_team_key, r.away_team_key), r).is_some() {
            duplicates.push(format!("{} {} {}", r.season, r.home_team_key, r.away_team_key));
        }
    }
    if !duplicates.is_empty() {
        bail!("duplicate football-data matches: {duplicates:?}");
    }
    let fixture_keys: HashSet<(i64, i64, i64)> =
        fixtures.iter().map(|f| (f.season, f.home_team_key, f.away_team_key)).collect();
    let describe = |f: &Fixture| {
        format!(
            "season {} home {} away {} fpl_fixture_id {} kickoff {:?} fd_date {:?}",
            f.season,
            f.home_team_key,
            f.away_team_key,
            f.fpl_fixture_id,
            f.kickoff_time.map(from_micros),
            f.fd_date.map(date_from_days)
        )
    };
    let mut problems = Vec::new();
    let orphans: Vec<String> = fd
        .iter()
        .filter(|r| !fixture_keys.contains(&(r.season, r.home_team_key, r.away_team_key)))
        .map(|r| {
            format!(
                "season {} home {} away {} fd_date {}",
                r.season,
                r.home_team_key,
                r.away_team_key,
                date_from_days(r.fd_date)
            )
        })
        .collect();
    show_rows(&orphans, "football-data row(s) without an FPL fixture", &mut problems);
    let (mut date_bad, mut goals_bad, mut missing) = (Vec::new(), Vec::new(), Vec::new());
    let mut unfinished = 0;
    let newest: HashMap<i64, Days> = fd.iter().fold(HashMap::new(), |mut m, r| {
        let e = m.entry(r.season).or_insert(r.fd_date);
        *e = (*e).max(r.fd_date);
        m
    });
    let current = fixtures.iter().map(|f| f.season).max();
    for f in fixtures.iter_mut() {
        match by_key.get(&(f.season, f.home_team_key, f.away_team_key)) {
            Some(r) => {
                f.fd_date = Some(r.fd_date);
                let kickoff_date = f.kickoff_time.map(|k| days_from_date(uk_date(k)));
                if kickoff_date != Some(r.fd_date) {
                    date_bad.push(describe(f));
                }
                if f.finished {
                    if f.home_goals != Some(r.home_goals) || f.away_goals != Some(r.away_goals) {
                        goals_bad.push(describe(f));
                    }
                } else {
                    unfinished += 1;
                }
            }
            None if f.finished => {
                // Current season: football-data publishes with a delay.
                let kickoff_date = f.kickoff_time.map(|k| days_from_date(uk_date(k)));
                let pending = Some(f.season) == current
                    && match (kickoff_date, newest.get(&f.season)) {
                        (Some(k), Some(n)) => k >= *n,
                        (Some(_), None) => true,
                        (None, _) => false,
                    };
                if !pending {
                    missing.push(describe(f));
                }
            }
            None => {}
        }
    }
    show_rows(&date_bad, "date mismatch(es) (UK kickoff date)", &mut problems);
    show_rows(&goals_bad, "goal mismatch(es)", &mut problems);
    if unfinished > 0 {
        log::warn!("{unfinished} football-data result(s) for fixtures FPL has not finished yet");
    }
    show_rows(&missing, "finished FPL fixture(s) without a football-data row", &mut problems);
    if !problems.is_empty() {
        bail!("football-data cross-check failed: {}", problems.join("\n"));
    }
    Ok(())
}

fn assemble_fixtures(raw: Vec<RawFixture>, fd: &[FdResult]) -> Result<Vec<Fixture>> {
    // gw_index: dense rank of gw within the season.
    let mut gws: BTreeMap<i64, BTreeSet<i64>> = BTreeMap::new();
    let mut last_kickoff: HashMap<(i64, i64), Micros> = HashMap::new();
    for r in &raw {
        if let Some(gw) = r.gw {
            gws.entry(r.season).or_default().insert(gw);
            if let Some(k) = r.kickoff_time {
                let e = last_kickoff.entry((r.season, gw)).or_insert(k);
                *e = (*e).max(k);
            }
        }
    }
    let mut fixtures: Vec<Fixture> = raw
        .into_iter()
        .map(|r| {
            let gw_index =
                r.gw.map(|gw| gws[&r.season].iter().position(|&g| g == gw).unwrap() as i64 + 1);
            let last = r.gw.and_then(|gw| last_kickoff.get(&(r.season, gw)).copied());
            Fixture {
                fixture_key: r.season * 1000 + r.fpl_fixture_id,
                season: r.season,
                fpl_fixture_id: r.fpl_fixture_id,
                gw: r.gw,
                gw_index,
                kickoff_time: r.kickoff_time,
                home_team_key: r.home_team_key,
                away_team_key: r.away_team_key,
                home_goals: r.home_goals,
                away_goals: r.away_goals,
                finished: r.finished,
                fd_date: None,
                event_time: r.kickoff_time.unwrap_or(*UNSCHEDULED_AT),
                available_at: last.map(lockdown_time).unwrap_or(*UNSCHEDULED_AT),
            }
        })
        .collect();
    attach_football_data(&mut fixtures, fd)?;
    Ok(fixtures)
}

fn validate_fixtures(rows: &[Fixture]) -> Result<()> {
    let mut problems = Vec::new();
    let mut keys = HashSet::new();
    let mut sides = HashSet::new();
    let mut per_season: HashMap<i64, usize> = HashMap::new();
    let mut per_team: HashMap<(i64, i64), usize> = HashMap::new();
    for f in rows {
        let scheduled = f.gw.is_some();
        let ok = scheduled == f.kickoff_time.is_some()
            && scheduled == f.gw_index.is_some()
            && (!f.finished || (scheduled && f.home_goals.is_some() && f.away_goals.is_some()))
            && (f.finished || (f.home_goals.is_none() && f.away_goals.is_none()))
            && f.home_team_key != f.away_team_key
            && f.fixture_key == f.season * 1000 + f.fpl_fixture_id
            && (1..=999).contains(&f.fpl_fixture_id)
            && f.gw.is_none_or(|g| (1..=47).contains(&g))
            && f.home_goals.is_none_or(|g| g >= 0)
            && f.away_goals.is_none_or(|g| g >= 0)
            && f.season >= FIRST_SEASON
            && (f.available_at >= f.event_time || f.kickoff_time.is_none())
            && f.kickoff_time.is_none_or(|k| crate::common::season_of_aug(k) == f.season);
        if !ok {
            problems.push(format!("fixture {} row consistency", f.fixture_key));
        }
        if !keys.insert(f.fixture_key) {
            problems.push(format!("fixture_key {} not unique", f.fixture_key));
        }
        if !sides.insert((f.season, f.home_team_key, f.away_team_key)) {
            problems.push(format!("(season, home, away) of {} not unique", f.fixture_key));
        }
        *per_season.entry(f.season).or_default() += 1;
        *per_team.entry((f.season, f.home_team_key)).or_default() += 1;
        *per_team.entry((f.season, f.away_team_key)).or_default() += 1;
    }
    if per_season.values().any(|&n| n != FIXTURES_PER_SEASON)
        || per_team.values().any(|&n| n != MATCHES_PER_TEAM)
    {
        problems.push("380 fixtures per season, 38 per team".into());
    }
    if !problems.is_empty() {
        bail!(
            "table 'fixture' failed validation: {}",
            problems.iter().take(20).cloned().collect::<Vec<_>>().join("; ")
        );
    }
    Ok(())
}

pub fn build_fixture(ctx: &mut BuildContext) -> Result<Table> {
    let resolver = TeamResolver::new(ctx.table("team_dim")?)?;
    let raw = raw_fixtures(ctx)?;
    for r in &raw {
        resolver.fpl_code(r.home_team_key)?;
        resolver.fpl_code(r.away_team_key)?;
    }
    let fd = football_data_results(ctx, &resolver, Some(FIRST_SEASON))?;
    let rows = assemble_fixtures(raw, &fd)?;
    validate_fixtures(&rows)?;
    Ok(fixture_table(&rows))
}

// --- schedule availability -------------------------------------------------------------------

/// Per season its last kickoff.
fn last_kickoffs(rows: impl Iterator<Item = (i64, Option<Micros>)>) -> HashMap<i64, Micros> {
    let mut out: HashMap<i64, Micros> = HashMap::new();
    for (season, kickoff) in rows {
        if let Some(k) = kickoff {
            let e = out.entry(season).or_insert(k);
            *e = (*e).max(k);
        }
    }
    out
}

fn schedule_available(season: i64, last: &HashMap<i64, Micros>) -> Micros {
    schedule_available_at(season, last.get(&(season - 1)).copied())
}

/// Rows of season S available by the last deadline of season S - 1.
fn check_published_after_previous_season(
    rows: &[(i64, Micros)],
    gameweek: &[Gameweek],
) -> Result<()> {
    let mut last_deadline: HashMap<i64, Micros> = HashMap::new();
    for g in gameweek {
        let e = last_deadline.entry(g.season).or_insert(g.deadline_time);
        *e = (*e).max(g.deadline_time);
    }
    let early: Vec<&(i64, Micros)> = rows
        .iter()
        .filter(|(season, at)| last_deadline.get(&(season - 1)).is_some_and(|d| at <= d))
        .collect();
    if !early.is_empty() {
        bail!(
            "{} row(s) available before the previous season's last deadline: {:?}",
            early.len(),
            &early[..early.len().min(10)]
        );
    }
    Ok(())
}

// --- gameweek --------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Gameweek {
    pub season: i64,
    pub gw: i64,
    pub gw_index: i64,
    pub deadline_time: Micros,
    pub deadline_source: String,
    pub first_kickoff: Micros,
    pub last_kickoff: Micros,
    pub lockdown_time: Micros,
    pub event_time: Micros,
    pub available_at: Micros,
}

impl Gameweek {
    pub fn from_table(t: &Table) -> Result<Vec<Gameweek>> {
        let (season, gw, gw_index) = (t.i64("season")?, t.i64("gw")?, t.i64("gw_index")?);
        let (deadline, source) = (t.i64("deadline_time")?, t.str("deadline_source")?);
        let (first, last, lockdown) =
            (t.i64("first_kickoff")?, t.i64("last_kickoff")?, t.i64("lockdown_time")?);
        let (event_time, available_at) = (t.i64("event_time")?, t.i64("available_at")?);
        Ok((0..t.height())
            .map(|i| Gameweek {
                season: season[i],
                gw: gw[i],
                gw_index: gw_index[i],
                deadline_time: deadline[i],
                deadline_source: source[i].clone(),
                first_kickoff: first[i],
                last_kickoff: last[i],
                lockdown_time: lockdown[i],
                event_time: event_time[i],
                available_at: available_at[i],
            })
            .collect())
    }
}

fn gameweek_table(rows: &[Gameweek]) -> Table {
    Table::new()
        .with("season", Column::I64(rows.iter().map(|r| r.season).collect()))
        .with("gw", Column::I64(rows.iter().map(|r| r.gw).collect()))
        .with("gw_index", Column::I64(rows.iter().map(|r| r.gw_index).collect()))
        .with("deadline_time", Column::Ts(rows.iter().map(|r| r.deadline_time).collect()))
        .with(
            "deadline_source",
            Column::Str(rows.iter().map(|r| r.deadline_source.clone()).collect()),
        )
        .with("first_kickoff", Column::Ts(rows.iter().map(|r| r.first_kickoff).collect()))
        .with("last_kickoff", Column::Ts(rows.iter().map(|r| r.last_kickoff).collect()))
        .with("lockdown_time", Column::Ts(rows.iter().map(|r| r.lockdown_time).collect()))
        .with("event_time", Column::Ts(rows.iter().map(|r| r.event_time).collect()))
        .with("available_at", Column::Ts(rows.iter().map(|r| r.available_at).collect()))
}

/// gw, deadline_time, average_entry_score (null until the event is finished).
#[derive(Clone, Debug)]
pub struct Event {
    pub gw: i64,
    pub deadline_time: Option<Micros>,
    pub average_entry_score: Option<i64>,
}

pub fn bootstrap_events(bootstrap: &Value) -> Result<Vec<Event>> {
    bootstrap["events"]
        .as_array()
        .context("bootstrap without events")?
        .iter()
        .map(|e| {
            let finished = json_bool(&e["finished"]);
            Ok(Event {
                gw: json_i64(&e["id"], "event id")?,
                deadline_time: opt_iso(&e["deadline_time"])?,
                average_entry_score: json_opt_i64(
                    &e["average_entry_score"],
                    "average_entry_score",
                )?
                .filter(|_| finished),
            })
        })
        .collect()
}

/// Per-event deadlines from a vaastav fixtures.csv that carries `deadline_time`.
fn fixtures_csv_deadlines(fixtures: &Csv) -> Result<HashMap<i64, Option<Micros>>> {
    let mut per_event: BTreeMap<i64, Vec<Option<String>>> = BTreeMap::new();
    for (event, deadline) in
        fixtures.cells("event").into_iter().zip(fixtures.cells("deadline_time"))
    {
        if let Some(event) = parse_opt_int(event, "event")? {
            per_event.entry(event).or_default().push(deadline.map(str::to_string));
        }
    }
    let mut out = HashMap::new();
    for (event, deadlines) in per_event {
        let distinct: HashSet<&String> = deadlines.iter().flatten().collect();
        if distinct.len() > 1 {
            bail!("fixtures.csv: deadline_time differs within an event");
        }
        let first = deadlines.iter().flatten().next().map(|d| parse_iso(d)).transpose()?;
        out.insert(event, first);
    }
    Ok(out)
}

fn assemble_gameweeks(
    fixtures: &[Fixture],
    bootstrap_deadlines: &BTreeMap<i64, Vec<Event>>,
    csv_deadlines: &HashMap<i64, HashMap<i64, Option<Micros>>>,
) -> Result<Vec<Gameweek>> {
    // (season, gw) -> (gw_index first, first kickoff, last kickoff)
    type GwSpan = (Option<i64>, Option<Micros>, Option<Micros>);
    let mut groups: BTreeMap<(i64, i64), GwSpan> = BTreeMap::new();
    for f in fixtures {
        let Some(gw) = f.gw else { continue };
        let entry = groups.entry((f.season, gw)).or_insert((None, None, None));
        if entry.0.is_none() {
            entry.0 = f.gw_index;
        }
        if let Some(k) = f.kickoff_time {
            entry.1 = Some(entry.1.map_or(k, |x| x.min(k)));
            entry.2 = Some(entry.2.map_or(k, |x| x.max(k)));
        }
    }
    let mut by_season: BTreeMap<i64, Vec<(i64, i64, Micros, Micros)>> = BTreeMap::new();
    for ((season, gw), (gw_index, first, last)) in groups {
        by_season.entry(season).or_default().push((
            gw,
            gw_index.context("gw_index")?,
            first.context("GW without kickoffs")?,
            last.context("GW without kickoffs")?,
        ));
    }
    let mut rows = Vec::new();
    for (season, gws) in by_season {
        let (deadlines, source): (Vec<Micros>, &str) =
            if let Some(events) = bootstrap_deadlines.get(&season) {
                let have: HashSet<i64> = gws.iter().map(|g| g.0).collect();
                let extra: BTreeSet<i64> =
                    events.iter().map(|e| e.gw).filter(|gw| !have.contains(gw)).collect();
                if !extra.is_empty() {
                    log::info!(
                        "season {season}: bootstrap event(s) {extra:?} have no fixtures; skipped"
                    );
                }
                let by_gw: HashMap<i64, Option<Micros>> =
                    events.iter().map(|e| (e.gw, e.deadline_time)).collect();
                let missing: Vec<i64> = gws
                    .iter()
                    .filter(|g| by_gw.get(&g.0).copied().flatten().is_none())
                    .map(|g| g.0)
                    .collect();
                if !missing.is_empty() {
                    bail!("season {season}: GW(s) {missing:?} missing from the bootstrap");
                }
                (gws.iter().map(|g| by_gw[&g.0].unwrap()).collect(), "bootstrap")
            } else if let Some(csv) = csv_deadlines.get(&season) {
                if gws.iter().any(|g| csv.get(&g.0).copied().flatten().is_none()) {
                    bail!("season {season}: fixtures.csv lacks deadlines for some GWs");
                }
                (gws.iter().map(|g| csv[&g.0].unwrap()).collect(), "fixtures_csv")
            } else {
                (gws.iter().map(|g| g.2 - APPROX_DEADLINE_LEAD).collect(), "approx")
            };
        for (g, deadline) in gws.iter().zip(deadlines) {
            rows.push(Gameweek {
                season,
                gw: g.0,
                gw_index: g.1,
                deadline_time: deadline,
                deadline_source: source.to_string(),
                first_kickoff: g.2,
                last_kickoff: g.3,
                lockdown_time: lockdown_time(g.3),
                event_time: deadline,
                available_at: 0,
            });
        }
    }
    let last = last_kickoffs(rows.iter().map(|r| (r.season, Some(r.last_kickoff))));
    for r in &mut rows {
        r.available_at = schedule_available(r.season, &last);
    }
    Ok(rows)
}

/// `bootstrap_events` of the newest bootstrap of each of `seasons` that has one.
fn season_bootstrap_events(
    ctx: &BuildContext,
    seasons: &HashSet<i64>,
) -> Result<BTreeMap<i64, Vec<Event>>> {
    let mut out = BTreeMap::new();
    for (season, entry) in last_bootstrap_per_season(&ctx.store)? {
        if seasons.contains(&season) {
            out.insert(season, bootstrap_events(&RawStore::read_json(&entry.path)?)?);
        }
    }
    Ok(out)
}

fn validate_gameweeks(rows: &[Gameweek]) -> Result<()> {
    let mut problems = Vec::new();
    let mut seen = HashSet::new();
    let mut per_season: BTreeMap<i64, Vec<&Gameweek>> = BTreeMap::new();
    for g in rows {
        if !seen.insert((g.season, g.gw)) {
            problems.push(format!("(season, gw) {} {} unique", g.season, g.gw));
        }
        per_season.entry(g.season).or_default().push(g);
        if !(g.deadline_time < g.first_kickoff
            && g.first_kickoff <= g.last_kickoff
            && g.lockdown_time > g.last_kickoff
            && g.available_at < g.event_time
            && (1..=47).contains(&g.gw)
            && (1..=38).contains(&g.gw_index))
        {
            problems.push(format!("gameweek {} GW{} order", g.season, g.gw));
        }
    }
    for (season, gws) in &mut per_season {
        let expected = if *season == 2022 { 37 } else { 38 };
        if gws.len() != expected {
            problems.push(format!("season {season}: {} GWs (expected {expected})", gws.len()));
        }
        gws.sort_by_key(|g| g.gw);
        if gws.windows(2).any(|w| w[0].deadline_time >= w[1].deadline_time) {
            problems.push(format!("season {season}: deadlines not strictly increasing"));
        }
    }
    let early: Vec<(i64, Micros)> = rows.iter().map(|g| (g.season, g.available_at)).collect();
    if let Err(err) = check_published_after_previous_season(&early, rows) {
        problems.push(err.to_string());
    }
    if !problems.is_empty() {
        bail!("table 'gameweek' failed validation: {}", problems.join("; "));
    }
    Ok(())
}

pub fn build_gameweek(ctx: &mut BuildContext) -> Result<Table> {
    let fixtures = Fixture::from_table(ctx.table("fixture")?)?;
    let seasons: HashSet<i64> = fixtures.iter().map(|f| f.season).collect();
    let bootstrap_deadlines = season_bootstrap_events(ctx, &seasons)?;
    let mut csv_deadlines = HashMap::new();
    for (season, season_dir) in vaastav_seasons(&vaastav_run(ctx)?)? {
        let path = season_dir.join("fixtures.csv.gz");
        if seasons.contains(&season) && !bootstrap_deadlines.contains_key(&season) && path.exists()
        {
            let csv = Csv::read(&path)?;
            if csv.has("deadline_time") {
                csv_deadlines.insert(season, fixtures_csv_deadlines(&csv)?);
            }
        }
    }
    let rows = assemble_gameweeks(&fixtures, &bootstrap_deadlines, &csv_deadlines)?;
    validate_gameweeks(&rows)?;
    Ok(gameweek_table(&rows))
}

// --- gameweek_result -------------------------------------------------------------------------

pub fn build_gameweek_result(ctx: &mut BuildContext) -> Result<Table> {
    let gameweek = Gameweek::from_table(ctx.table("gameweek")?)?;
    let seasons: HashSet<i64> = gameweek.iter().map(|g| g.season).collect();
    let events = season_bootstrap_events(ctx, &seasons)?;
    let scores: HashMap<(i64, i64), Option<i64>> = events
        .iter()
        .flat_map(|(season, events)| {
            events.iter().map(move |e| ((*season, e.gw), e.average_entry_score))
        })
        .collect();
    Ok(Table::new()
        .with("season", Column::I64(gameweek.iter().map(|g| g.season).collect()))
        .with("gw", Column::I64(gameweek.iter().map(|g| g.gw).collect()))
        .with(
            "average_entry_score",
            Column::OptI64(
                gameweek.iter().map(|g| scores.get(&(g.season, g.gw)).copied().flatten()).collect(),
            ),
        )
        .with("event_time", Column::Ts(gameweek.iter().map(|g| g.lockdown_time).collect()))
        .with("available_at", Column::Ts(gameweek.iter().map(|g| g.lockdown_time).collect())))
}

// --- schedule --------------------------------------------------------------------------------

pub fn build_schedule(ctx: &mut BuildContext) -> Result<Table> {
    let fixtures = Fixture::from_table(ctx.table("fixture")?)?;
    let gameweek = Gameweek::from_table(ctx.table("gameweek")?)?;
    let last = last_kickoffs(fixtures.iter().map(|f| (f.season, f.kickoff_time)));
    let available: Vec<Micros> =
        fixtures.iter().map(|f| schedule_available(f.season, &last)).collect();
    let event_time: Vec<Micros> =
        fixtures.iter().map(|f| f.kickoff_time.unwrap_or(*UNSCHEDULED_AT)).collect();
    if fixtures.iter().zip(&available).zip(&event_time).any(|((_, a), e)| a >= e) {
        bail!("table 'schedule' failed validation: available_at not before event_time");
    }
    let rows: Vec<(i64, Micros)> =
        fixtures.iter().map(|f| f.season).zip(available.iter().copied()).collect();
    check_published_after_previous_season(&rows, &gameweek)?;
    Ok(Table::new()
        .with("fixture_key", Column::I64(fixtures.iter().map(|f| f.fixture_key).collect()))
        .with("season", Column::I64(fixtures.iter().map(|f| f.season).collect()))
        .with("gw", Column::OptI64(fixtures.iter().map(|f| f.gw).collect()))
        .with("gw_index", Column::OptI64(fixtures.iter().map(|f| f.gw_index).collect()))
        .with("kickoff_time", Column::OptTs(fixtures.iter().map(|f| f.kickoff_time).collect()))
        .with("home_team_key", Column::I64(fixtures.iter().map(|f| f.home_team_key).collect()))
        .with("away_team_key", Column::I64(fixtures.iter().map(|f| f.away_team_key).collect()))
        .with("schedule_source", Column::Str(vec!["final".to_string(); fixtures.len()]))
        .with("event_time", Column::Ts(event_time))
        .with("available_at", Column::Ts(available)))
}

// --- fixture_snapshot ------------------------------------------------------------------------

const FIXTURE_SNAPSHOT_FLAGS: [&str; 3] = ["started", "finished", "finished_provisional"];

fn opt_flag(value: Option<&Value>) -> Option<bool> {
    match value? {
        Value::Null => None,
        other => Some(json_bool(other)),
    }
}

pub fn build_fixture_snapshot(ctx: &mut BuildContext) -> Result<Table> {
    let mut team_codes: HashMap<i64, HashMap<i64, i64>> = HashMap::new();
    let mut bootstraps = None;
    let (mut snapshot_at, mut season_col, mut key, mut id, mut gw, mut kickoff) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut home, mut away) = (Vec::new(), Vec::new());
    let mut flags: Vec<Vec<Option<bool>>> = vec![Vec::new(); 3];
    for (at, path) in ctx.store.entries("fpl", "fixtures", ".json.gz")? {
        let payload = RawStore::read_json(&path)?;
        let records = payload.as_array().context("fixtures payload is not a list")?;
        let season = fixtures_snapshot_season(records, &at)?;
        if let std::collections::hash_map::Entry::Vacant(slot) = team_codes.entry(season) {
            if bootstraps.is_none() {
                bootstraps = Some(last_bootstrap_per_season(&ctx.store)?);
            }
            let entry = bootstraps.as_ref().unwrap().get(&season).with_context(|| {
                format!(
                    "no archived bootstrap for season {season} (team ids of {})",
                    path.display()
                )
            })?;
            slot.insert(bootstrap_team_codes(&RawStore::read_json(&entry.path)?)?);
        }
        let codes = &team_codes[&season];
        let fixtures = api_fixtures_from_json(records)?;
        check_ids(fixtures.iter().flat_map(|f| [f.team_h, f.team_a]), codes)?;
        let at = micros(&at);
        for (f, record) in fixtures.iter().zip(records) {
            snapshot_at.push(at);
            season_col.push(season);
            key.push(season * 1000 + f.id);
            id.push(f.id);
            gw.push(f.event);
            kickoff.push(f.kickoff_time);
            home.push(codes[&f.team_h]);
            away.push(codes[&f.team_a]);
            for (j, flag) in FIXTURE_SNAPSHOT_FLAGS.iter().enumerate() {
                flags[j].push(opt_flag(record.get(*flag)));
            }
        }
    }
    let n = snapshot_at.len();
    let mut table = Table::new()
        .with("snapshot_at", Column::Ts(snapshot_at.clone()))
        .with("season", Column::I64(season_col))
        .with("fixture_key", Column::I64(key))
        .with("fpl_fixture_id", Column::I64(id))
        .with("gw", Column::OptI64(gw))
        .with("kickoff_time", Column::OptTs(kickoff))
        .with("home_team_key", Column::I64(home))
        .with("away_team_key", Column::I64(away));
    for (j, flag) in FIXTURE_SNAPSHOT_FLAGS.iter().enumerate() {
        table.push(flag, Column::OptBool(std::mem::take(&mut flags[j])));
    }
    table.push("event_time", Column::Ts(snapshot_at.clone()));
    table.push("available_at", Column::Ts(snapshot_at));
    debug_assert_eq!(table.height(), n);
    Ok(table)
}
