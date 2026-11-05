//! Baseline feature builders (Python `fplopt.features.baseline`). Each takes one `AsOfView`,
//! reads only through it and returns a deterministic frame (fixed columns and dtypes, sorted
//! by its key). The target gameweek is the one whose deadline is the view's deadline.
//!
//! Before player snapshots exist (2016/17 – 2020/21 GW32) `player_pool` is built from the
//! `player_gw` rows visible at the deadline, and a club can have no as-of source at all
//! (2020/21 GW1: four clubs' GW1 matches were postponed); `pool_coverage` reports it and
//! `player_pool` logs a warning.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use anyhow::{Result, bail};
use fplopt_store::{AsOfView, Column, Micros, Table};

use crate::util::{Kahan, median, opt_floats, opt_ints, positions, strings, take};

pub const SNAPSHOT_COLUMNS: &[&str] = &[
    "snapshot_at",
    "season",
    "team_key",
    "element_type",
    "now_cost",
    "status",
    "chance_of_playing_next_round",
    "news_added",
    "ep_next",
    "form",
];
pub const FORM_STATS: &[&str] = &[
    "minutes",
    "starts",
    "total_points",
    "goals_scored",
    "assists",
    "us_npxg",
    "us_xa",
    "fpl_xg",
    "fpl_xa",
];
pub const FORM_WINDOW: usize = 5;
/// upcoming_fixtures: the target GW and the next HORIZON GWs.
pub const HORIZON: i64 = 5;
pub const OUTCOMES: &[&str] = &["home", "draw", "away"];
pub const ODDS_GROUP: &[&str] =
    &["fixture_key", "source", "bookmaker", "market", "outcome", "line"];

fn rate_stats() -> impl Iterator<Item = &'static str> {
    FORM_STATS.iter().copied().filter(|s| !matches!(*s, "minutes" | "starts"))
}

/// season, gw, gw_index rows of `season`'s visible gameweeks, in visible order.
pub(crate) struct Gameweeks {
    pub gw: Vec<i64>,
    pub gw_index: Vec<i64>,
}

impl Gameweeks {
    pub fn index(&self) -> HashMap<i64, i64> {
        self.gw.iter().copied().zip(self.gw_index.iter().copied()).collect()
    }
}

pub(crate) fn gameweeks(view: &AsOfView, season: i64) -> Result<Gameweeks> {
    let table = view.table("gameweek", Some(&["season", "gw", "gw_index"]))?;
    let seasons = table.i64("season")?;
    let rows = positions(table.height(), |i| seasons[i] == season);
    let table = take(&table, &rows);
    Ok(Gameweeks { gw: table.i64("gw")?.to_vec(), gw_index: table.i64("gw_index")?.to_vec() })
}

/// (season, gw, gw_index) of the gameweek whose deadline is the view's deadline.
pub(crate) fn target(view: &AsOfView) -> Result<(i64, i64, i64)> {
    let (season, gw) = view.gameweek_for_deadline()?;
    let gameweeks = gameweeks(view, season)?;
    let Some(i) = gameweeks.gw.iter().position(|&g| g == gw) else {
        bail!("no gameweek {gw} in season {season}");
    };
    Ok((season, gw, gameweeks.gw_index[i]))
}

/// Rows of `season`'s newest player snapshot before the deadline (empty if none), sorted by
/// player_key: per player the newest row across sources, then only players present at the
/// newest time.
fn newest_snapshot(view: &AsOfView, season: i64) -> Result<Table> {
    let snaps = view.latest("player_snapshot", &["player_key"], Some(SNAPSHOT_COLUMNS))?;
    let seasons = snaps.opt_i64("season")?;
    let rows = positions(snaps.height(), |i| seasons[i] == Some(season));
    let snaps = take(&snaps, &rows);
    let times = snaps.opt_i64("snapshot_at")?;
    let Some(newest) = times.iter().flatten().max().copied() else {
        return Ok(snaps);
    };
    let rows = positions(snaps.height(), |i| times[i] == Some(newest));
    Ok(take(&snaps, &rows))
}

fn sorted_unique(values: impl IntoIterator<Item = i64>) -> Vec<i64> {
    values.into_iter().collect::<BTreeSet<_>>().into_iter().collect()
}

/// Every club of an as-of schedule, sorted.
fn schedule_teams(schedule: &Table) -> Result<Vec<i64>> {
    let home = schedule.i64("home_team_key")?;
    let away = schedule.i64("away_team_key")?;
    Ok(sorted_unique(home.iter().chain(away).copied()))
}

// --- players -------------------------------------------------------------------------------

/// Players pickable for the target GW with position, club and price (tenths of £m): from the
/// newest player snapshot of the season when one exists before the deadline (source
/// 'snapshot'), else from `player_gw` (source 'player_gw'): the target GW's visible rows plus,
/// for players of clubs that blank in the target GW, their newest earlier row of the season.
/// Logs a warning when a club with a fixture in the horizon has no players.
pub fn player_pool(view: &AsOfView) -> Result<Table> {
    let pool = pool(view)?;
    let coverage = coverage(view, &pool)?;
    let horizon = coverage.i64("n_fixtures_horizon")?;
    let players = coverage.i64("n_pool_players")?;
    let gaps: Vec<i64> = positions(coverage.height(), |i| horizon[i] > 0 && players[i] == 0)
        .into_iter()
        .map(|i| coverage.i64("team_key").unwrap()[i])
        .collect();
    if !gaps.is_empty() {
        log::warn!(
            "player_pool at {}: {} club(s) with a fixture in the horizon have no pool players \
             (team_key {gaps:?}); the pool is incomplete",
            view.deadline(),
            gaps.len()
        );
    }
    Ok(pool)
}

fn pool(view: &AsOfView) -> Result<Table> {
    let (season, gw, gw_index) = target(view)?;
    let snaps = newest_snapshot(view, season)?;
    if snaps.height() > 0 {
        let n = snaps.height();
        return Table::new()
            .with("player_key", Column::I64(snaps.i64("player_key")?.to_vec()))
            .with("element_type", Column::I64(snaps.i64("element_type")?.to_vec()))
            .with("team_key", Column::I64(snaps.i64("team_key")?.to_vec()))
            .with("price", Column::I64(snaps.i64("now_cost")?.to_vec()))
            .with("source", strings("snapshot", n))
            .sorted(&["player_key"]);
    }

    let columns = ["player_key", "season", "gw", "team_key", "element_type", "value"];
    let rows = view.table("player_gw", Some(&columns))?;
    let seasons = rows.i64("season")?;
    let rows = take(&rows, &positions(rows.height(), |i| seasons[i] == season));
    let index = gameweeks(view, season)?.index();
    let keys = rows.i64("player_key")?;
    let gws = rows.i64("gw")?;
    let teams = rows.i64("team_key")?;
    let current: Vec<usize> = positions(rows.height(), |i| gws[i] == gw);
    let in_current: HashSet<i64> = current.iter().map(|&i| keys[i]).collect();
    // Newest earlier row per player, kept only if the club blanks in the target GW.
    let mut earlier: BTreeMap<i64, (i64, usize)> = BTreeMap::new();
    for i in 0..rows.height() {
        let Some(&gi) = index.get(&gws[i]) else { continue };
        if gi < gw_index && !in_current.contains(&keys[i]) {
            let entry = earlier.entry(keys[i]).or_insert((gi, i));
            if gi >= entry.0 {
                *entry = (gi, i);
            }
        }
    }
    let schedule = view.schedule(season)?;
    let all_teams: BTreeSet<i64> = schedule_teams(&schedule)?.into_iter().collect();
    let sgw = schedule.opt_i64("gw")?;
    let home = schedule.i64("home_team_key")?;
    let away = schedule.i64("away_team_key")?;
    let mut blank = all_teams;
    for i in 0..schedule.height() {
        if sgw[i] == Some(gw) {
            blank.remove(&home[i]);
            blank.remove(&away[i]);
        }
    }
    let mut picked = current;
    picked.extend(earlier.values().map(|&(_, i)| i).filter(|&i| blank.contains(&teams[i])));
    let picked = take(&rows, &picked);
    let n = picked.height();
    Table::new()
        .with("player_key", Column::I64(picked.i64("player_key")?.to_vec()))
        .with("element_type", Column::I64(picked.i64("element_type")?.to_vec()))
        .with("team_key", Column::I64(picked.i64("team_key")?.to_vec()))
        .with("price", Column::I64(picked.i64("value")?.to_vec()))
        .with("source", strings("player_gw", n))
        .sorted(&["player_key"])
}

/// (team_key, gw, kickoff, fixture_key, opponent, is_home) per side of the fixtures with a GW.
struct Side {
    team_key: i64,
    opponent: i64,
    is_home: bool,
    fixture_key: i64,
    gw: i64,
    kickoff: Option<Micros>,
}

/// One side per fixture and team: home sides in schedule order, then away sides.
fn sides(schedule: &Table) -> Result<Vec<Side>> {
    let keys = schedule.i64("fixture_key")?;
    let gws = schedule.opt_i64("gw")?;
    let kickoffs = schedule.opt_i64("kickoff_time")?;
    let home = schedule.i64("home_team_key")?;
    let away = schedule.i64("away_team_key")?;
    let mut out = Vec::new();
    for is_home in [true, false] {
        for i in 0..schedule.height() {
            let Some(gw) = gws[i] else { continue };
            let (team_key, opponent) =
                if is_home { (home[i], away[i]) } else { (away[i], home[i]) };
            out.push(Side {
                team_key,
                opponent,
                is_home,
                fixture_key: keys[i],
                gw,
                kickoff: kickoffs[i],
            });
        }
    }
    Ok(out)
}

fn coverage(view: &AsOfView, pool: &Table) -> Result<Table> {
    let (season, gw, gw_index) = target(view)?;
    let schedule = view.schedule(season)?;
    let gameweeks = gameweeks(view, season)?;
    let horizon: HashSet<i64> = (0..gameweeks.gw.len())
        .filter(|&i| (gw_index..=gw_index + HORIZON).contains(&gameweeks.gw_index[i]))
        .map(|i| gameweeks.gw[i])
        .collect();
    let sides = sides(&schedule)?;
    let teams = schedule_teams(&schedule)?;
    let mut target_count: HashMap<i64, i64> = HashMap::new();
    let mut ahead: HashMap<i64, i64> = HashMap::new();
    for side in &sides {
        if side.gw == gw {
            *target_count.entry(side.team_key).or_default() += 1;
        }
        if horizon.contains(&side.gw) {
            *ahead.entry(side.team_key).or_default() += 1;
        }
    }
    let mut players: HashMap<i64, i64> = HashMap::new();
    for &team in pool.i64("team_key")? {
        *players.entry(team).or_default() += 1;
    }
    let count = |m: &HashMap<i64, i64>| -> Column {
        Column::I64(teams.iter().map(|t| m.get(t).copied().unwrap_or(0)).collect())
    };
    let n = teams.len();
    Ok(Table::new()
        .with("season", Column::I64(vec![season; n]))
        .with("gw", Column::I64(vec![gw; n]))
        .with("team_key", Column::I64(teams.clone()))
        .with("n_fixtures", count(&target_count))
        .with("n_fixtures_horizon", count(&ahead))
        .with("n_pool_players", count(&players)))
}

/// Per club of the season's as-of schedule: its fixtures in the target GW and in the horizon
/// and its number of `player_pool` players (fixtures and 0 players: the pool is incomplete).
pub fn pool_coverage(view: &AsOfView) -> Result<Table> {
    coverage(view, &pool(view)?)
}

/// Pool players (sorted) with `columns` of their newest-snapshot row (null without one).
fn pool_with_snapshot(view: &AsOfView, columns: &[&str]) -> Result<(Vec<i64>, Table)> {
    let (season, _, _) = target(view)?;
    let keys = pool(view)?.i64("player_key")?.to_vec();
    let snaps = newest_snapshot(view, season)?;
    let at: HashMap<i64, usize> =
        snaps.i64("player_key")?.iter().enumerate().map(|(i, &k)| (k, i)).collect();
    let mut out = Table::new();
    for name in columns {
        let column = snaps.column(name)?;
        let rows: Vec<Option<usize>> = keys.iter().map(|k| at.get(k).copied()).collect();
        out.push(name, take_opt(column, &rows));
    }
    Ok((keys, out))
}

/// The rows at `rows` of `column` as its nullable variant (None: a null).
fn take_opt(column: &Column, rows: &[Option<usize>]) -> Column {
    fn pick<T: Clone>(v: &[T], rows: &[Option<usize>]) -> Vec<Option<T>> {
        rows.iter().map(|r| r.map(|i| v[i].clone())).collect()
    }
    fn pick_opt<T: Clone>(v: &[Option<T>], rows: &[Option<usize>]) -> Vec<Option<T>> {
        rows.iter().map(|r| r.and_then(|i| v[i].clone())).collect()
    }
    match column {
        Column::I64(v) => Column::OptI64(pick(v, rows)),
        Column::OptI64(v) => Column::OptI64(pick_opt(v, rows)),
        Column::F64(v) => Column::OptF64(pick(v, rows)),
        Column::OptF64(v) => Column::OptF64(pick_opt(v, rows)),
        Column::Str(v) => Column::OptStr(pick(v, rows)),
        Column::OptStr(v) => Column::OptStr(pick_opt(v, rows)),
        Column::Bool(v) => Column::OptBool(pick(v, rows)),
        Column::OptBool(v) => Column::OptBool(pick_opt(v, rows)),
        Column::Ts(v) => Column::OptTs(pick(v, rows)),
        Column::OptTs(v) => Column::OptTs(pick_opt(v, rows)),
        Column::OptDate(v) => Column::OptDate(pick_opt(v, rows)),
    }
    .normalized()
}

/// Per pool player: status flag, chance of playing next round and when the news was added,
/// from the newest snapshot (null without one, i.e. before 2021).
pub fn availability(view: &AsOfView) -> Result<Table> {
    let columns = ["status", "chance_of_playing_next_round", "news_added"];
    let (keys, snaps) = pool_with_snapshot(view, &columns)?;
    let mut out = Table::new().with("player_key", Column::I64(keys));
    for (name, column) in snaps.columns {
        out.push(&name, column);
    }
    Ok(out)
}

/// Per pool player: FPL's `ep_next` and `form` from the newest snapshot (null without one).
pub fn ep_next(view: &AsOfView) -> Result<Table> {
    let (keys, snaps) = pool_with_snapshot(view, &["ep_next", "form"])?;
    Ok(Table::new()
        .with("player_key", Column::I64(keys))
        .with("ep_next", opt_floats(snaps.opt_f64("ep_next")?))
        .with("form", opt_floats(snaps.opt_f64("form")?)))
}

/// Sums, counts and per-90 rates of one window (one player's rows).
struct Window {
    matches: i64,
    apps: i64,
    sums: Vec<Option<f64>>,
    rates: Vec<Option<f64>>,
}

fn window(rows: &[usize], minutes: &[i64], stats: &[Vec<Option<f64>>]) -> Window {
    let mut sums = Vec::new();
    let mut rates = Vec::new();
    for (j, stat) in FORM_STATS.iter().enumerate() {
        let mut total = Kahan::default();
        let mut known = false;
        let mut known_minutes = 0i64;
        for &i in rows {
            if let Some(v) = stats[j][i] {
                total.add(v);
                known = true;
                known_minutes += minutes[i];
            }
        }
        let total = known.then(|| total.value());
        sums.push(total);
        if !matches!(*stat, "minutes" | "starts") {
            rates.push(match total {
                Some(t) if known_minutes > 0 => Some(t / known_minutes as f64 * 90.0),
                _ => None,
            });
        }
    }
    Window {
        matches: rows.len() as i64,
        apps: rows.iter().filter(|&&i| minutes[i] > 0).count() as i64,
        sums,
        rates,
    }
}

/// Per player with a visible match this season: sums of FORM_STATS, matches, appearances
/// (minutes > 0) and per-90 rates over the last 5 matches and the season so far.
pub fn recent_form(view: &AsOfView) -> Result<Table> {
    let (season, _, _) = target(view)?;
    let mut columns = vec!["player_key", "season", "fixture_key", "kickoff_time"];
    columns.extend(FORM_STATS);
    let matches = view.table("player_match", Some(&columns))?;
    let seasons = matches.i64("season")?;
    let matches = take(&matches, &positions(matches.height(), |i| seasons[i] == season))
        .sorted(&["player_key", "kickoff_time", "fixture_key"])?;
    let keys = matches.i64("player_key")?;
    let minutes = matches.i64("minutes")?;
    let stats: Vec<Vec<Option<f64>>> =
        FORM_STATS.iter().map(|s| matches.opt_f64(s)).collect::<Result<_>>()?;
    let mut players: Vec<i64> = Vec::new();
    let mut last5 = Vec::new();
    let mut whole = Vec::new();
    let mut start = 0;
    while start < keys.len() {
        let mut end = start;
        while end < keys.len() && keys[end] == keys[start] {
            end += 1;
        }
        let rows: Vec<usize> = (start..end).collect();
        players.push(keys[start]);
        last5.push(window(&rows[rows.len().saturating_sub(FORM_WINDOW)..], minutes, &stats));
        whole.push(window(&rows, minutes, &stats));
        start = end;
    }
    let mut out = Table::new().with("player_key", Column::I64(players));
    for (suffix, windows) in [("last5", &last5), ("season", &whole)] {
        out.push(
            &format!("matches_{suffix}"),
            Column::I64(windows.iter().map(|w| w.matches).collect()),
        );
        out.push(&format!("apps_{suffix}"), Column::I64(windows.iter().map(|w| w.apps).collect()));
        for (j, stat) in FORM_STATS.iter().enumerate() {
            let values = windows.iter().map(|w| w.sums[j]).collect();
            out.push(&format!("{stat}_{suffix}"), Column::OptF64(values));
        }
    }
    for (suffix, windows) in [("last5", &last5), ("season", &whole)] {
        for (j, stat) in rate_stats().enumerate() {
            let values = windows.iter().map(|w| w.rates[j]).collect();
            out.push(&format!("{stat}_per90_{suffix}"), Column::OptF64(values));
        }
    }
    Ok(out)
}

// --- teams ---------------------------------------------------------------------------------

/// Per club of the season and GW from the target GW to HORIZON GWs later (by gw_index): its
/// fixtures in the as-of schedule, one row each, or a single row with null fixture columns
/// for a blank; `n_fixtures` 0 = blank, 2 = double.
pub fn upcoming_fixtures(view: &AsOfView) -> Result<Table> {
    let (season, _, gw_index) = target(view)?;
    let schedule = view.schedule(season)?;
    let gameweeks = gameweeks(view, season)?;
    let horizon: Vec<usize> = (0..gameweeks.gw.len())
        .filter(|&i| (gw_index..=gw_index + HORIZON).contains(&gameweeks.gw_index[i]))
        .collect();
    let teams = schedule_teams(&schedule)?;
    let sides = sides(&schedule)?;
    let mut by_team_gw: HashMap<(i64, i64), Vec<&Side>> = HashMap::new();
    for side in &sides {
        by_team_gw.entry((side.team_key, side.gw)).or_default().push(side);
    }
    let source = if schedule.height() > 0 {
        Some(schedule.str("schedule_source")?[0].clone())
    } else {
        None
    };
    let mut team_key = Vec::new();
    let mut gw = Vec::new();
    let mut gwi = Vec::new();
    let mut n_fixtures = Vec::new();
    let mut fixture_key = Vec::new();
    let mut opponent = Vec::new();
    let mut is_home = Vec::new();
    let mut kickoff = Vec::new();
    for &team in &teams {
        for &g in &horizon {
            let found = by_team_gw.get(&(team, gameweeks.gw[g]));
            let n = found.map_or(0, Vec::len);
            let mut push = |side: Option<&Side>| {
                team_key.push(team);
                gw.push(gameweeks.gw[g]);
                gwi.push(gameweeks.gw_index[g]);
                n_fixtures.push(n as i64);
                fixture_key.push(side.map(|s| s.fixture_key));
                opponent.push(side.map(|s| s.opponent));
                is_home.push(side.map(|s| s.is_home));
                kickoff.push(side.and_then(|s| s.kickoff));
            };
            match found {
                Some(found) => found.iter().for_each(|s| push(Some(s))),
                None => push(None),
            }
        }
    }
    let n = team_key.len();
    Table::new()
        .with("season", Column::I64(vec![season; n]))
        .with("team_key", Column::I64(team_key))
        .with("gw", Column::I64(gw))
        .with("gw_index", Column::I64(gwi.clone()))
        .with("horizon", Column::I64(gwi.iter().map(|g| g - gw_index).collect()))
        .with("n_fixtures", Column::I64(n_fixtures))
        .with("fixture_key", opt_ints(fixture_key))
        .with("opponent_team_key", opt_ints(opponent))
        .with("is_home", Column::OptBool(is_home).normalized())
        .with("kickoff_time", Column::OptTs(kickoff).normalized())
        .with("schedule_source", Column::OptStr(vec![source; n]).normalized())
        .sorted(&["team_key", "gw_index", "kickoff_time", "fixture_key"])
}

/// Prices per fixture and outcome (home, draw, away), only fixtures with all three positive.
fn complete(prices: BTreeMap<i64, [Option<f64>; 3]>) -> BTreeMap<i64, [f64; 3]> {
    prices
        .into_iter()
        .filter_map(|(fixture, p)| match p {
            [Some(h), Some(d), Some(a)] if h > 0.0 && d > 0.0 && a > 0.0 => {
                Some((fixture, [h, d, a]))
            }
            _ => None,
        })
        .collect()
}

/// Median price per (fixture, outcome) of `rows`.
fn pivot_median(rows: &[(i64, usize, f64)]) -> BTreeMap<i64, [Option<f64>; 3]> {
    let mut groups: BTreeMap<(i64, usize), Vec<f64>> = BTreeMap::new();
    for &(fixture, outcome, price) in rows {
        if !price.is_nan() {
            groups.entry((fixture, outcome)).or_default().push(price);
        }
    }
    let mut out: BTreeMap<i64, [Option<f64>; 3]> = BTreeMap::new();
    for ((fixture, outcome), mut prices) in groups {
        out.entry(fixture).or_default()[outcome] = Some(median(&mut prices));
    }
    out
}

/// Per fixture: overround-normalised h2h probabilities (home, draw, away) and their source
/// from the newest odds before the deadline: football-data's market average if complete,
/// else the median across Odds API bookmakers of the fixture's newest Odds API snapshot.
fn match_probabilities(
    view: &AsOfView,
    fixture_keys: &HashSet<i64>,
) -> Result<HashMap<i64, ([f64; 3], &'static str)>> {
    let odds = view.latest("odds_snapshot", ODDS_GROUP, Some(&["price", "snapshot_at"]))?;
    let fixtures = odds.opt_i64("fixture_key")?;
    let sources = odds.opt_str("source")?;
    let bookmakers = odds.opt_str("bookmaker")?;
    let markets = odds.opt_str("market")?;
    let outcomes = odds.opt_str("outcome")?;
    let prices = odds.opt_f64("price")?;
    let times = odds.opt_i64("snapshot_at")?;
    let mut fd = Vec::new();
    let mut api = Vec::new();
    for i in 0..odds.height() {
        let (Some(fixture), Some(outcome)) = (fixtures[i], outcomes[i].as_deref()) else {
            continue;
        };
        let Some(outcome) = OUTCOMES.iter().position(|o| *o == outcome) else { continue };
        if markets[i].as_deref() != Some("h2h") || !fixture_keys.contains(&fixture) {
            continue;
        }
        let price = prices[i].unwrap_or(f64::NAN);
        match sources[i].as_deref() {
            Some("football-data") if bookmakers[i].as_deref() == Some("avg") => {
                fd.push((fixture, outcome, price))
            }
            Some("odds-api") => api.push((fixture, outcome, price, times[i])),
            _ => {}
        }
    }
    let fd = complete(pivot_median(&fd));
    let mut newest: HashMap<i64, Option<Micros>> = HashMap::new();
    for &(fixture, _, _, at) in &api {
        let entry = newest.entry(fixture).or_insert(at);
        if at > *entry {
            *entry = at;
        }
    }
    let api: Vec<(i64, usize, f64)> = api
        .into_iter()
        .filter(|(fixture, _, _, at)| at.is_some() && newest[fixture] == *at)
        .map(|(fixture, outcome, price, _)| (fixture, outcome, price))
        .collect();
    let api = complete(pivot_median(&api));
    let mut out = HashMap::new();
    let chosen = fd
        .iter()
        .map(|(f, p)| (*f, *p, "football-data"))
        .chain(api.iter().filter(|(f, _)| !fd.contains_key(f)).map(|(f, p)| (*f, *p, "odds-api")));
    for (fixture, prices, source) in chosen {
        let implied = prices.map(|p| 1.0 / p);
        let total = implied[0] + implied[1] + implied[2];
        out.insert(fixture, (implied.map(|x| x / total), source));
    }
    Ok(out)
}

/// Per target-GW fixture and side: both clubs' newest Elo `rating_after` before the deadline
/// and the match's as-of h2h probabilities from the side's view (null without odds).
pub fn team_strength(view: &AsOfView) -> Result<Table> {
    let (season, gw, _) = target(view)?;
    let schedule = view.schedule(season)?;
    let gws = schedule.opt_i64("gw")?;
    let schedule = take(&schedule, &positions(schedule.height(), |i| gws[i] == Some(gw)));
    let sides = sides(&schedule)?;
    let ratings = view
        .table("team_rating", Some(&["team_key", "event_time", "rating_after"]))?
        .sorted(&["team_key", "event_time"])?;
    let mut elo: HashMap<i64, Option<f64>> = HashMap::new();
    for (team, rating) in ratings.i64("team_key")?.iter().zip(ratings.opt_f64("rating_after")?) {
        elo.insert(*team, rating);
    }
    let fixture_keys: HashSet<i64> = schedule.i64("fixture_key")?.iter().copied().collect();
    let probabilities = match_probabilities(view, &fixture_keys)?;
    let n = sides.len();
    let get = |s: &Side, outcome: usize| probabilities.get(&s.fixture_key).map(|(p, _)| p[outcome]);
    let lookup = |team: i64| elo.get(&team).copied().flatten();
    Table::new()
        .with("season", Column::I64(vec![season; n]))
        .with("gw", Column::I64(sides.iter().map(|s| s.gw).collect()))
        .with("fixture_key", Column::I64(sides.iter().map(|s| s.fixture_key).collect()))
        .with("kickoff_time", Column::OptTs(sides.iter().map(|s| s.kickoff).collect()).normalized())
        .with("team_key", Column::I64(sides.iter().map(|s| s.team_key).collect()))
        .with("opponent_team_key", Column::I64(sides.iter().map(|s| s.opponent).collect()))
        .with("is_home", Column::Bool(sides.iter().map(|s| s.is_home).collect()))
        .with("elo", opt_floats(sides.iter().map(|s| lookup(s.team_key)).collect()))
        .with("opponent_elo", opt_floats(sides.iter().map(|s| lookup(s.opponent)).collect()))
        .with(
            "p_win",
            opt_floats(sides.iter().map(|s| get(s, if s.is_home { 0 } else { 2 })).collect()),
        )
        .with("p_draw", opt_floats(sides.iter().map(|s| get(s, 1)).collect()))
        .with(
            "p_loss",
            opt_floats(sides.iter().map(|s| get(s, if s.is_home { 2 } else { 0 })).collect()),
        )
        .with(
            "odds_source",
            Column::OptStr(
                sides
                    .iter()
                    .map(|s| probabilities.get(&s.fixture_key).map(|(_, src)| src.to_string()))
                    .collect(),
            )
            .normalized(),
        )
        .sorted(&["fixture_key", "team_key"])
}
