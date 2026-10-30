//! Per player-fixture history for the minutes model (Python `fplopt.features.history`; PLAN
//! §6.3): `training_frame(view)` has one row per visible `player_match` row with its lag
//! features and targets; `prediction_frame(view)` one row per `player_pool` player and
//! horizon fixture of his club with the lags as of the view's deadline. The module docs of
//! the Python version define every feature; this port follows it step by step, keeping
//! pandas' row orders wherever an order decides a value (first/last of duplicates, ties).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use anyhow::{Context, Result};
use fplopt_store::{AsOfView, Column, Micros, Table};

use crate::bans::{
    BanEvent, CupRed, DOMESTIC_CUPS, LeagueRow, RedKind, ban_events, club_timelines, serve_bans,
};
use crate::baseline::{player_pool, upcoming_fixtures};
use crate::util::{Kahan, days_between, floats, positions};

pub use crate::baseline::HORIZON;

const N_STARTERS: usize = 11;
pub const WINDOWS: &[usize] = &[1, 3, 5, 10];
const LONG_WINDOW: usize = 38;
/// League fixtures per club and season (start_share_last).
const SEASON_FIXTURES: f64 = 38.0;
const REST_CAP_DAYS: f64 = 30.0;
const APPEARANCE_CAP_DAYS: f64 = 365.0;
const RUN_CAP: f64 = 10.0;
const CLUB_CAP: f64 = 10.0;
/// 0-minute rows before an appearance that make it a return.
const ABSENCE_RUN: i64 = 3;
const RETURN_WINDOW: usize = 3;
const RED_WINDOW: usize = 3;
/// Club fixtures in club_continuity.
const CONTINUITY_WINDOW: usize = 5;
const DAY: Micros = 86_400_000_000;
/// Prediction: bans from the cards of the last BAN_LOOKBACK only.
const BAN_LOOKBACK: Micros = 200 * DAY;
/// A flag counts once FPL has had this long after the card to set 's' for a running ban.
const FLAG_SETTLE: Micros = 12 * 3_600_000_000;
/// europe_week: a UEFA match this close to the fixture.
const EUROPE_WINDOW_DAYS: f64 = 4.0;
const EUROPEAN: &str = "uefa.";
const CUP_WINDOW: Micros = 7 * DAY;

pub const LAG_FEATURES: &[&str] = &[
    "element_type",
    "price",
    "start_rate_l1",
    "start_rate_l3",
    "start_rate_l5",
    "start_rate_l10",
    "minutes_l1",
    "minutes_l3",
    "minutes_l5",
    "minutes_l10",
    "start_rate_long",
    "n_long",
    "start_share_season",
    "start_share_last",
    "days_since_app",
    "cup_minutes_7d",
    "cup_started_last",
    "rest_days",
    "rest_days_all",
    "days_to_next",
    "europe_week",
    "absent_run",
    "returned",
    "yellows_season",
    "reds_l3",
    "comp_start_sum",
    "comp_rank",
    "fixtures_at_club",
    "club_continuity",
    "club_fixtures_season",
];
pub const TARGETS: &[&str] = &["minutes", "start", "sixty", "sub"];

/// Float days since the epoch, as pandas computes them on nanosecond timedeltas.
fn days(t: Micros) -> f64 {
    days_between(t, 0)
}

fn opt_days(t: Option<Micros>) -> f64 {
    t.map_or(f64::NAN, days)
}

/// numpy `clip(x, lo, hi)` (NaN stays NaN).
fn clip(x: f64, lo: f64, hi: f64) -> f64 {
    if x.is_nan() { x } else { x.max(lo).min(hi) }
}

// --- inputs --------------------------------------------------------------------------------

/// Visible `player_match` rows joined to their gameweek (view order: the order pandas'
/// inner merge keeps), with start, its inferred flag and position/price.
#[derive(Clone, Default)]
struct Matches {
    player_key: Vec<i64>,
    season: Vec<i64>,
    fixture_key: Vec<i64>,
    gw: Vec<i64>,
    gw_index: Vec<i64>,
    team_key: Vec<i64>,
    kickoff: Vec<Micros>,
    deadline: Vec<Micros>,
    minutes: Vec<f64>,
    start: Vec<f64>,
    start_inferred: Vec<bool>,
    yellow: Vec<f64>,
    red: Vec<f64>,
    element_type: Vec<f64>,
    price: Vec<f64>,
}

impl Matches {
    fn len(&self) -> usize {
        self.player_key.len()
    }
}

/// Inferred starters per row: per (fixture, team) the `N_STARTERS` rows with the most
/// minutes, ties by player_key; only rows with minutes > 0.
fn infer_starts(fixture: &[i64], team: &[i64], minutes: &[i64], player: &[i64]) -> Vec<bool> {
    let mut order: Vec<usize> = (0..fixture.len()).collect();
    order.sort_by_key(|&i| (fixture[i], team[i], std::cmp::Reverse(minutes[i]), player[i]));
    let mut out = vec![false; fixture.len()];
    let mut rank = 0;
    for (n, &i) in order.iter().enumerate() {
        if n > 0 {
            let j = order[n - 1];
            rank = if (fixture[j], team[j]) == (fixture[i], team[i]) { rank + 1 } else { 0 };
        }
        out[i] = rank < N_STARTERS && minutes[i] > 0;
    }
    out
}

fn matches(view: &AsOfView) -> Result<Matches> {
    let gameweeks = view.table("gameweek", Some(&["season", "gw", "gw_index", "deadline_time"]))?;
    let mut gw_of: HashMap<(i64, i64), (i64, Micros)> = HashMap::new();
    for i in 0..gameweeks.height() {
        let deadline = gameweeks.i64("deadline_time").context("gameweek.deadline_time")?[i];
        gw_of.insert(
            (gameweeks.i64("season")?[i], gameweeks.i64("gw")?[i]),
            (gameweeks.i64("gw_index")?[i], deadline),
        );
    }
    let columns = [
        "player_key",
        "season",
        "fixture_key",
        "gw",
        "team_key",
        "kickoff_time",
        "minutes",
        "starts",
        "yellow_cards",
        "red_cards",
    ];
    let rows = view.table("player_match", Some(&columns))?;
    let seasons = rows.i64("season")?;
    let gws = rows.i64("gw")?;
    let kept = positions(rows.height(), |i| gw_of.contains_key(&(seasons[i], gws[i])));
    let rows = rows.take(&kept);
    let player = rows.i64("player_key")?;
    let season = rows.i64("season")?;
    let gw = rows.i64("gw")?;
    let fixture = rows.i64("fixture_key")?;
    let team = rows.i64("team_key")?;
    let minutes = rows.i64("minutes")?;
    let inferred = infer_starts(fixture, team, minutes, player);
    let real = rows.opt_f64("starts")?;
    let yellow = rows.opt_f64("yellow_cards")?;
    let red = rows.opt_f64("red_cards")?;

    let registered =
        view.table("player_gw", Some(&["player_key", "season", "gw", "element_type", "value"]))?;
    let mut reg: HashMap<(i64, i64, i64), (Option<f64>, Option<f64>)> = HashMap::new();
    let (rp, rs, rg) =
        (registered.i64("player_key")?, registered.i64("season")?, registered.i64("gw")?);
    let (re, rv) = (registered.opt_f64("element_type")?, registered.opt_f64("value")?);
    for i in 0..registered.height() {
        reg.insert((rp[i], rs[i], rg[i]), (re[i], rv[i]));
    }
    let seasons_table =
        view.table("player_season", Some(&["player_key", "season", "element_type"]))?;
    let mut position: HashMap<(i64, i64), Option<f64>> = HashMap::new();
    let (sp, ss, se) = (
        seasons_table.i64("player_key")?,
        seasons_table.i64("season")?,
        seasons_table.opt_f64("element_type")?,
    );
    for i in 0..seasons_table.height() {
        position.insert((sp[i], ss[i]), se[i]);
    }

    let n = rows.height();
    let mut out = Matches {
        player_key: player.to_vec(),
        season: season.to_vec(),
        fixture_key: fixture.to_vec(),
        gw: gw.to_vec(),
        team_key: team.to_vec(),
        kickoff: rows.i64("kickoff_time").context("player_match.kickoff_time")?.to_vec(),
        ..Matches::default()
    };
    for i in 0..n {
        let (gw_index, deadline) = gw_of[&(season[i], gw[i])];
        out.gw_index.push(gw_index);
        out.deadline.push(deadline);
        out.minutes.push(minutes[i] as f64);
        let known = real[i].is_some_and(|x| !x.is_nan());
        let start = if known { real[i].unwrap() } else { f64::from(u8::from(inferred[i])) };
        out.start.push(start.clamp(0.0, 1.0));
        out.start_inferred.push(!known);
        out.yellow.push(yellow[i].unwrap_or(0.0));
        out.red.push(red[i].unwrap_or(0.0));
        let (element_type, value) =
            reg.get(&(player[i], season[i], gw[i])).copied().unwrap_or((None, None));
        let fallback = position.get(&(player[i], season[i])).copied().flatten();
        out.element_type.push(element_type.or(fallback).unwrap_or(f64::NAN));
        out.price.push(value.unwrap_or(f64::NAN));
    }
    Ok(out)
}

/// One row per (team_key, fixture_key) of the matches (the first), sorted by (team_key,
/// kickoff, fixture_key).
#[derive(Clone, Default)]
struct TeamFixtures {
    team_key: Vec<i64>,
    fixture_key: Vec<i64>,
    season: Vec<i64>,
    gw_index: Vec<i64>,
    kickoff: Vec<Micros>,
}

fn team_fixtures(m: &Matches) -> TeamFixtures {
    let mut seen = HashSet::new();
    let mut rows: Vec<usize> =
        (0..m.len()).filter(|&i| seen.insert((m.team_key[i], m.fixture_key[i]))).collect();
    rows.sort_by_key(|&i| (m.team_key[i], m.kickoff[i], m.fixture_key[i]));
    TeamFixtures {
        team_key: rows.iter().map(|&i| m.team_key[i]).collect(),
        fixture_key: rows.iter().map(|&i| m.fixture_key[i]).collect(),
        season: rows.iter().map(|&i| m.season[i]).collect(),
        gw_index: rows.iter().map(|&i| m.gw_index[i]).collect(),
        kickoff: rows.iter().map(|&i| m.kickoff[i]).collect(),
    }
}

/// Per row (team_key, ref_time): the mean over the team's last `CONTINUITY_WINDOW` fixtures
/// before ref_time of the share of its 11 starters who started its previous fixture (NaN
/// before the team's second fixture).
fn club_continuity(m: &Matches, teams: &[i64], refs: &[Micros]) -> Vec<f64> {
    let fixtures = team_fixtures(m);
    let mut order: HashMap<(i64, i64), usize> = HashMap::new();
    let mut count: HashMap<i64, usize> = HashMap::new();
    for i in 0..fixtures.team_key.len() {
        let n = count.entry(fixtures.team_key[i]).or_default();
        order.insert((fixtures.team_key[i], fixtures.fixture_key[i]), *n);
        *n += 1;
    }
    let mut started: HashSet<(i64, i64, usize)> = HashSet::new();
    let mut starters: Vec<(i64, i64, usize)> = Vec::new();
    for i in 0..m.len() {
        if m.start[i] >= 0.5
            && let Some(&o) = order.get(&(m.team_key[i], m.fixture_key[i]))
        {
            started.insert((m.team_key[i], m.player_key[i], o));
            starters.push((m.team_key[i], m.player_key[i], o));
        }
    }
    let mut kept: HashMap<(i64, usize), f64> = HashMap::new();
    for &(team, player, o) in &starters {
        let entry = kept.entry((team, o)).or_insert(0.0);
        if o > 0 && started.contains(&(team, player, o - 1)) {
            *entry += 1.0;
        }
    }
    // Per team: (kickoff, rolling mean) in fixture order.
    let mut means: HashMap<i64, Vec<(Micros, f64)>> = HashMap::new();
    let mut window: HashMap<i64, Vec<f64>> = HashMap::new();
    for i in 0..fixtures.team_key.len() {
        let team = fixtures.team_key[i];
        let o = order[&(team, fixtures.fixture_key[i])];
        let continuity = if o == 0 {
            f64::NAN
        } else {
            kept.get(&(team, o)).map_or(f64::NAN, |k| k / N_STARTERS as f64)
        };
        let values = window.entry(team).or_default();
        values.push(continuity);
        let recent = &values[values.len().saturating_sub(CONTINUITY_WINDOW)..];
        let known: Vec<f64> = recent.iter().copied().filter(|x| !x.is_nan()).collect();
        let mean = if known.is_empty() {
            f64::NAN
        } else {
            known.iter().sum::<f64>() / known.len() as f64
        };
        means.entry(team).or_default().push((fixtures.kickoff[i], mean));
    }
    teams
        .iter()
        .zip(refs)
        .map(|(team, &at)| {
            let Some(list) = means.get(team) else { return f64::NAN };
            let below = list.partition_point(|(t, _)| *t < at);
            if below == 0 { f64::NAN } else { list[below - 1].1 }
        })
        .collect()
}

/// Per row (team_key, season, gw_index): the team's fixtures of that season in earlier GWs.
fn fixtures_before(fixtures: &TeamFixtures, rows: &[(i64, i64, i64)]) -> Vec<f64> {
    let mut by_club: HashMap<(i64, i64), Vec<i64>> = HashMap::new();
    for i in 0..fixtures.team_key.len() {
        by_club
            .entry((fixtures.team_key[i], fixtures.season[i]))
            .or_default()
            .push(fixtures.gw_index[i]);
    }
    for list in by_club.values_mut() {
        list.sort_unstable();
    }
    rows.iter()
        .map(|&(team, season, gw_index)| {
            by_club
                .get(&(team, season))
                .map_or(0.0, |l| l.partition_point(|&g| g < gw_index) as f64)
        })
        .collect()
}

// --- lags ----------------------------------------------------------------------------------

/// Rows sorted by (player_key, season, gw_index, kickoff, fixture_key) for `lags`.
struct LagRows {
    player_key: Vec<i64>,
    season: Vec<i64>,
    gw_index: Vec<i64>,
    team_key: Vec<i64>,
    kickoff: Vec<Micros>,
    ref_time: Vec<Micros>,
    minutes: Vec<f64>,
    start: Vec<f64>,
    yellow: Vec<f64>,
    red: Vec<f64>,
}

#[derive(Default)]
struct Lags {
    values: BTreeMap<&'static str, Vec<f64>>,
    club_starts: Vec<f64>,
    last_row: Vec<f64>,
}

fn cumulative(values: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(values.len() + 1);
    let mut total = 0.0;
    out.push(0.0);
    for v in values {
        total += v;
        out.push(total);
    }
    out
}

/// Lag features of `rows` (sorted) from each player's rows of earlier GWs.
fn lags(rows: &LagRows) -> Lags {
    let n = rows.player_key.len();
    let mut first = vec![0usize; n];
    let mut block = vec![0usize; n];
    let mut season_first = vec![0usize; n];
    for i in 0..n {
        let same_player = i > 0 && rows.player_key[i - 1] == rows.player_key[i];
        let same_season = same_player && rows.season[i - 1] == rows.season[i];
        first[i] = if same_player { first[i - 1] } else { i };
        season_first[i] = if same_season { season_first[i - 1] } else { i };
        block[i] =
            if same_season && rows.gw_index[i - 1] == rows.gw_index[i] { block[i - 1] } else { i };
    }
    let appeared: Vec<f64> =
        rows.minutes.iter().map(|&m| if m > 0.0 { 1.0 } else { 0.0 }).collect();
    let has_prior: Vec<bool> = (0..n).map(|i| block[i] > first[i]).collect();
    let prev: Vec<usize> = (0..n).map(|i| if has_prior[i] { block[i] - 1 } else { 0 }).collect();
    let starts = cumulative(&rows.start);
    let minutes = cumulative(&rows.minutes);
    let mut out = Lags::default();

    for &k in WINDOWS.iter().chain([&LONG_WINDOW]) {
        let mut rate = Vec::with_capacity(n);
        let mut mean = Vec::with_capacity(n);
        let mut count = Vec::with_capacity(n);
        for i in 0..n {
            let lo = block[i].saturating_sub(k).max(first[i]);
            let c = (block[i] - lo) as f64;
            rate.push((starts[block[i]] - starts[lo]) / c);
            mean.push((minutes[block[i]] - minutes[lo]) / c);
            count.push(c);
        }
        match k {
            1 => {
                out.values.insert("start_rate_l1", rate);
                out.values.insert("minutes_l1", mean);
            }
            3 => {
                out.values.insert("start_rate_l3", rate);
                out.values.insert("minutes_l3", mean);
            }
            5 => {
                out.values.insert("start_rate_l5", rate);
                out.values.insert("minutes_l5", mean);
            }
            10 => {
                out.values.insert("start_rate_l10", rate);
                out.values.insert("minutes_l10", mean);
            }
            _ => {
                out.values.insert("start_rate_long", rate);
                out.values.insert("n_long", count);
            }
        }
    }

    // Starts at the current club this season, over the club's fixtures this season.
    let mut club: HashMap<(i64, i64, i64), f64> = HashMap::new();
    let mut in_block: HashMap<(i64, i64, i64), f64> = HashMap::new();
    for i in 0..n {
        let c = club.entry((rows.player_key[i], rows.season[i], rows.team_key[i])).or_insert(0.0);
        *c += rows.start[i];
        let b =
            in_block.entry((rows.player_key[i], rows.season[i], rows.gw_index[i])).or_insert(0.0);
        *b += rows.start[i];
        out.club_starts.push(*c - *b);
    }

    // Last season (any club): starts / SEASON_FIXTURES, NaN without a row.
    let mut per_season: HashMap<(i64, i64), Kahan> = HashMap::new();
    for i in 0..n {
        per_season.entry((rows.player_key[i], rows.season[i])).or_default().add(rows.start[i]);
    }
    out.values.insert(
        "start_share_last",
        (0..n)
            .map(|i| {
                per_season
                    .get(&(rows.player_key[i], rows.season[i] - 1))
                    .map_or(f64::NAN, |k| k.value() / SEASON_FIXTURES)
            })
            .collect(),
    );

    // Days since the last appearance before the block, from the GW deadline.
    let kickoff: Vec<f64> = rows.kickoff.iter().map(|&t| days(t)).collect();
    let mut last_appearance = vec![f64::NAN; n];
    for i in 0..n {
        let carried = if i > first[i] { last_appearance[i - 1] } else { f64::NAN };
        last_appearance[i] = if appeared[i] > 0.0 { kickoff[i] } else { carried };
    }
    out.values.insert(
        "days_since_app",
        (0..n)
            .map(|i| {
                let last = if has_prior[i] { last_appearance[prev[i]] } else { f64::NAN };
                clip(days(rows.ref_time[i]) - last, 0.0, APPEARANCE_CAP_DAYS)
            })
            .collect(),
    );
    out.last_row = (0..n).map(|i| if has_prior[i] { kickoff[prev[i]] } else { f64::NAN }).collect();

    // Consecutive 0-minute rows ending at each row; the run just before the block.
    let mut run = vec![0i64; n];
    let mut nonzero = 0i64;
    for i in 0..n {
        if i == first[i] {
            nonzero = first[i] as i64 - 1;
        }
        if appeared[i] > 0.0 {
            nonzero = i as i64;
        }
        run[i] = i as i64 - nonzero;
    }
    out.values.insert(
        "absent_run",
        (0..n).map(|i| (if has_prior[i] { run[prev[i]] } else { 0 } as f64).min(RUN_CAP)).collect(),
    );
    let came_back: Vec<f64> = (0..n)
        .map(|i| {
            let previous = if i > first[i] { run[i - 1] } else { 0 };
            if appeared[i] > 0.0 && previous >= ABSENCE_RUN { 1.0 } else { 0.0 }
        })
        .collect();
    let came_back = cumulative(&came_back);
    out.values.insert(
        "returned",
        (0..n)
            .map(|i| {
                let lo = block[i].saturating_sub(RETURN_WINDOW).max(first[i]);
                if came_back[block[i]] - came_back[lo] > 0.0 { 1.0 } else { 0.0 }
            })
            .collect(),
    );

    // Cards: yellows this season before the block, reds in the last RED_WINDOW rows.
    let yellow = cumulative(&rows.yellow);
    let red = cumulative(&rows.red);
    out.values.insert(
        "yellows_season",
        (0..n).map(|i| yellow[block[i]] - yellow[season_first[i]]).collect(),
    );
    out.values.insert(
        "reds_l3",
        (0..n)
            .map(|i| {
                let lo = block[i].saturating_sub(RED_WINDOW).max(first[i]);
                red[block[i]] - red[lo]
            })
            .collect(),
    );

    // Consecutive earlier rows at the current club (the spell), capped.
    let mut spell_first = 0usize;
    let mut at_club = Vec::with_capacity(n);
    for i in 0..n {
        if i == first[i] || rows.team_key[i] != rows.team_key[i - 1] {
            spell_first = i;
        }
        let spell = block[i].saturating_sub(spell_first);
        at_club.push((spell as f64).min(CLUB_CAP));
    }
    out.values.insert("fixtures_at_club", at_club);
    out
}

// --- cup, congestion and bans --------------------------------------------------------------

/// (team_key, league, kickoff, available_at, played) per PL side of every visible
/// `club_match` (scheduled ones included): home sides, then away sides.
struct ClubSide {
    team_key: i64,
    european: bool,
    domestic_cup: bool,
    kickoff: Micros,
    available: Micros,
    played: bool,
}

fn club_sides(view: &AsOfView) -> Result<Vec<ClubSide>> {
    let columns =
        ["league", "kickoff_time", "available_at", "played", "home_team_key", "away_team_key"];
    let clubs = view.table("club_match", Some(&columns))?;
    let leagues = clubs.str("league")?;
    let kickoffs = clubs.i64("kickoff_time")?;
    let available = clubs.i64("available_at")?;
    let played = clubs.bool("played")?;
    let mut out = Vec::new();
    for side in ["home_team_key", "away_team_key"] {
        for (i, team) in clubs.opt_i64(side)?.into_iter().enumerate() {
            let Some(team_key) = team else { continue };
            out.push(ClubSide {
                team_key,
                european: leagues[i].starts_with(EUROPEAN),
                domestic_cup: DOMESTIC_CUPS.contains(&leagues[i].as_str()),
                kickoff: kickoffs[i],
                available: available[i],
                played: played[i],
            });
        }
    }
    Ok(out)
}

/// (team_key, kickoff) per PL side of every visible domestic cup match from `since` on.
fn cup_sides(view: &AsOfView, since: Option<Micros>) -> Result<Vec<(i64, Micros)>> {
    Ok(club_sides(view)?
        .into_iter()
        .filter(|s| s.domestic_cup && since.is_none_or(|t| s.kickoff >= t))
        .map(|s| (s.team_key, s.kickoff))
        .collect())
}

/// (team_key, season, fixture_key, kickoff) per side of the scheduled fixtures with a kickoff,
/// sorted by (team_key, kickoff, fixture_key).
fn league_sides(schedules: &[Table]) -> Result<Vec<(i64, i64, i64, Micros)>> {
    let mut out = Vec::new();
    for side in ["home_team_key", "away_team_key"] {
        for schedule in schedules {
            let kickoffs = schedule.opt_i64("kickoff_time")?;
            let teams = schedule.i64(side)?;
            let seasons = schedule.i64("season")?;
            let fixtures = schedule.i64("fixture_key")?;
            for i in 0..schedule.height() {
                if let Some(kickoff) = kickoffs[i] {
                    out.push((teams[i], seasons[i], fixtures[i], kickoff));
                }
            }
        }
    }
    out.sort_by_key(|&(team, _, fixture, kickoff)| (team, kickoff, fixture));
    Ok(out)
}

/// A fixture row for `congestion`.
struct FixtureRow {
    team_key: i64,
    season: i64,
    kickoff: Micros,
    ref_time: Micros,
    rest_days: f64,
}

/// (rest_days_all, days_to_next, europe_week) per fixture row. A club match counts if known
/// at the row's ref_time (`available_at` before it) and played or not kicked off yet then;
/// the league fixtures are `league` (the schedule) of the row's season.
fn congestion(
    rows: &[FixtureRow],
    league: &[(i64, i64, i64, Micros)],
    clubs: &[ClubSide],
) -> Vec<(f64, f64, f64)> {
    let mut next: HashMap<(i64, i64), Vec<Micros>> = HashMap::new();
    for &(team, season, _, kickoff) in league {
        next.entry((team, season)).or_default().push(kickoff);
    }
    for list in next.values_mut() {
        list.sort_unstable();
    }
    let mut by_team: HashMap<i64, Vec<&ClubSide>> = HashMap::new();
    for side in clubs {
        by_team.entry(side.team_key).or_default().push(side);
    }
    rows.iter()
        .map(|row| {
            let to_league = next
                .get(&(row.team_key, row.season))
                .and_then(|list| list.get(list.partition_point(|&t| t <= row.kickoff)))
                .map_or(f64::INFINITY, |&t| days_between(t, row.kickoff));
            let mut since_club = f64::INFINITY;
            let mut to_club = f64::INFINITY;
            let mut europe = 0.0;
            for side in by_team.get(&row.team_key).map_or(&[][..], Vec::as_slice) {
                let d = days_between(side.kickoff, row.kickoff);
                let known = side.available < row.ref_time
                    && (side.played || side.kickoff >= row.ref_time)
                    && d.abs() <= REST_CAP_DAYS;
                if !known {
                    continue;
                }
                if d < 0.0 {
                    since_club = since_club.min(-d);
                }
                if d > 0.0 {
                    to_club = to_club.min(d);
                }
                if side.european && d.abs() <= EUROPE_WINDOW_DAYS {
                    europe = 1.0;
                }
            }
            (row.rest_days.min(since_club), to_league.min(to_club).min(REST_CAP_DAYS), europe)
        })
        .collect()
}

/// The visible, mapped ESPN rows of non-league matches (one per player and match), sorted by
/// (player_key, event_time); `cum`: cumulative minutes (unknown = 0) in that order.
struct CupRows {
    player_key: Vec<i64>,
    event_time: Vec<Micros>,
    available: Vec<Micros>,
    minutes: Vec<f64>,
    starter: Vec<f64>,
    cum: Vec<f64>,
}

/// espn_athlete_id -> player_key of the visible mapped athletes among `ids`.
fn espn_players(view: &AsOfView, ids: &[i64]) -> Result<HashMap<i64, i64>> {
    let unique: Vec<i64> = ids.iter().copied().collect::<BTreeSet<_>>().into_iter().collect();
    let mapped = view.lookup("espn_map", &unique)?;
    let athletes = mapped.i64("espn_athlete_id")?;
    Ok(mapped
        .opt_i64("player_key")?
        .into_iter()
        .zip(athletes)
        .filter_map(|(player, &athlete)| player.map(|p| (athlete, p)))
        .collect())
}

fn cup_rows(view: &AsOfView) -> Result<CupRows> {
    let columns = [
        "espn_athlete_id",
        "espn_event_id",
        "league",
        "starter",
        "minutes",
        "event_time",
        "available_at",
    ];
    let rows = view.table("espn_player_match", Some(&columns))?;
    let leagues = rows.str("league")?;
    let rows = rows.take(&positions(rows.height(), |i| leagues[i] != "eng.1"));
    let athletes = rows.i64("espn_athlete_id")?;
    let players = espn_players(view, athletes)?;
    let events = rows.i64("espn_event_id")?;
    let mut seen = HashSet::new();
    let mut kept: Vec<(i64, usize)> = Vec::new();
    for (i, athlete) in athletes.iter().enumerate() {
        if let Some(&player) = players.get(athlete)
            && seen.insert((player, events[i]))
        {
            kept.push((player, i));
        }
    }
    let times = rows.i64("event_time")?;
    kept.sort_by_key(|&(player, i)| (player, times[i]));
    let minutes = rows.opt_f64("minutes")?;
    let starter = rows.bool("starter")?;
    let available = rows.i64("available_at")?;
    let mut out = CupRows {
        player_key: Vec::new(),
        event_time: Vec::new(),
        available: Vec::new(),
        minutes: Vec::new(),
        starter: Vec::new(),
        cum: Vec::new(),
    };
    let mut total = 0.0;
    for (n, &(player, i)) in kept.iter().enumerate() {
        if n == 0 || kept[n - 1].0 != player {
            total = 0.0;
        }
        let m = minutes[i].unwrap_or(f64::NAN);
        total += if m.is_nan() { 0.0 } else { m };
        out.player_key.push(player);
        out.event_time.push(times[i]);
        out.available.push(available[i]);
        out.minutes.push(m);
        out.starter.push(if starter[i] { 1.0 } else { 0.0 });
        out.cum.push(total);
    }
    Ok(out)
}

/// Per player the (time, row) pairs of some rows, stably sorted by time, for "the player's
/// latest row strictly before t" (pandas `merge_asof(allow_exact_matches=False)`).
struct LatestBefore(HashMap<i64, Vec<(Micros, usize)>>);

impl LatestBefore {
    fn new(players: &[i64], times: &[Micros], keep: impl Fn(usize) -> bool) -> Self {
        let mut map: HashMap<i64, Vec<(Micros, usize)>> = HashMap::new();
        for i in 0..players.len() {
            if keep(i) {
                map.entry(players[i]).or_default().push((times[i], i));
            }
        }
        for list in map.values_mut() {
            list.sort_by_key(|&(t, _)| t);
        }
        Self(map)
    }

    fn get(&self, player: i64, at: Micros) -> Option<usize> {
        let list = self.0.get(&player)?;
        let below = list.partition_point(|&(t, _)| t < at);
        (below > 0).then(|| list[below - 1].1)
    }
}

/// (cup_minutes_7d, cup_started_last, days_since_app) per row (player_key, ref_time,
/// last_row, days_since_app) from his non-league ESPN rows available before ref_time.
fn cup_features(
    cups: &CupRows,
    players: &[i64],
    refs: &[Micros],
    last_row: &[f64],
    days_since_app: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let by_available = LatestBefore::new(&cups.player_key, &cups.available, |_| true);
    let by_event = LatestBefore::new(&cups.player_key, &cups.event_time, |_| true);
    let appeared = LatestBefore::new(&cups.player_key, &cups.available, |i| cups.minutes[i] > 0.0);
    let mut minutes = Vec::with_capacity(players.len());
    let mut started = Vec::with_capacity(players.len());
    let mut since = Vec::with_capacity(players.len());
    for i in 0..players.len() {
        let latest = by_available.get(players[i], refs[i]);
        let older = by_event.get(players[i], refs[i] - CUP_WINDOW);
        minutes.push(latest.map_or(0.0, |j| cups.cum[j]) - older.map_or(0.0, |j| cups.cum[j]));
        // The latest non-league row counts if it came after his last league row.
        let after = latest.is_some_and(|j| {
            let last = if last_row[i].is_nan() { f64::NEG_INFINITY } else { last_row[i] };
            days(cups.event_time[j]) > last
        });
        started.push(if after { cups.starter[latest.unwrap()] } else { f64::NAN });
        let app = appeared.get(players[i], refs[i]);
        let cup_since = clip(
            days(refs[i]) - opt_days(app.map(|j| cups.event_time[j])),
            0.0,
            APPEARANCE_CAP_DAYS,
        );
        // numpy fmin: NaN only if both are NaN.
        let current = days_since_app[i];
        since.push(if current.is_nan() {
            cup_since
        } else if cup_since.is_nan() {
            current
        } else {
            current.min(cup_since)
        });
    }
    (minutes, started, since)
}

/// (player_key, fixture_key) -> ban_kind of the bans from the cards in `league_rows` and the
/// visible, mapped ESPN domestic cup reds, served over each club's `league_fixtures`
/// (team_key, fixture_key, kickoff) and its visible domestic cup matches; bans, ESPN rows
/// and cup matches from `since` on (None: all). `flags`: drop the bans FPL says are over.
fn bans(
    view: &AsOfView,
    league_rows: &[LeagueRow],
    league_fixtures: &[(i64, i64, Micros)],
    since: Option<Micros>,
    flags: bool,
) -> Result<BTreeMap<(i64, i64), i64>> {
    let columns = [
        "espn_athlete_id",
        "espn_event_id",
        "league",
        "red",
        "second_yellow",
        "team_key",
        "event_time",
    ];
    let reds = view.table("espn_player_match", Some(&columns))?;
    let red = reds.bool("red")?;
    let times = reds.i64("event_time")?;
    let reds =
        reds.take(&positions(reds.height(), |i| red[i] && since.is_none_or(|t| times[i] >= t)));
    let athletes = reds.i64("espn_athlete_id")?;
    let players = espn_players(view, athletes)?;
    let espn_match = view.table("espn_match", Some(&["espn_event_id", "fixture_key"]))?;
    let mut fixture_of: HashMap<i64, Vec<Option<i64>>> = HashMap::new();
    for (event, fixture) in
        espn_match.i64("espn_event_id")?.iter().zip(espn_match.opt_i64("fixture_key")?)
    {
        fixture_of.entry(*event).or_default().push(fixture);
    }
    let leagues = reds.str("league")?;
    let events = reds.i64("espn_event_id")?;
    let second = reds.bool("second_yellow")?;
    let teams = reds.i64("team_key")?;
    let times = reds.i64("event_time")?;
    let mut red_kinds = Vec::new();
    let mut cup_reds = Vec::new();
    for i in 0..reds.height() {
        let Some(&player) = players.get(&athletes[i]) else { continue };
        if leagues[i] == "eng.1" {
            for fixture in fixture_of.get(&events[i]).into_iter().flatten().flatten() {
                red_kinds.push(RedKind {
                    player_key: player,
                    fixture_key: *fixture,
                    second_yellow: second[i],
                });
            }
        } else if DOMESTIC_CUPS.contains(&leagues[i].as_str()) {
            cup_reds.push(CupRed {
                player_key: player,
                team_key: teams[i],
                kickoff: times[i],
                second_yellow: second[i],
            });
        }
    }
    // Rows without a card neither trigger a ban nor change a yellow total.
    let carded: Vec<LeagueRow> =
        league_rows.iter().filter(|r| r.yellow_cards > 0.0 || r.red_cards > 0.0).cloned().collect();
    let mut events = ban_events(&carded, &red_kinds, &cup_reds);
    if let Some(since) = since {
        events.retain(|e| e.time >= since);
    }
    let served = serve_bans(&events, &club_timelines(league_fixtures, &cup_sides(view, since)?));
    if flags { flag_cleared(view, served, &events) } else { Ok(served) }
}

/// `served` without the players FPL shows fully available (status 'a', chance blank or 100)
/// in their newest snapshot before the deadline, taken over `FLAG_SETTLE` after their latest
/// card that bans: the ban is over or was rescinded.
fn flag_cleared(
    view: &AsOfView,
    served: BTreeMap<(i64, i64), i64>,
    events: &[BanEvent],
) -> Result<BTreeMap<(i64, i64), i64>> {
    if served.is_empty() {
        return Ok(served);
    }
    let mut latest_card: HashMap<i64, Micros> = HashMap::new();
    for event in events {
        let entry = latest_card.entry(event.player_key).or_insert(event.time);
        *entry = (*entry).max(event.time);
    }
    let banned: HashSet<i64> = served.keys().map(|&(p, _)| p).collect();
    let columns = ["snapshot_at", "status", "chance_of_playing_next_round"];
    let snaps = view.latest("player_snapshot", &["player_key"], Some(&columns))?;
    let keys = snaps.opt_i64("player_key")?;
    let at = snaps.opt_i64("snapshot_at")?;
    let status = snaps.opt_str("status")?;
    let chance = snaps.opt_i64("chance_of_playing_next_round")?;
    let mut cleared = HashSet::new();
    for i in 0..snaps.height() {
        let Some(player) = keys[i] else { continue };
        if !banned.contains(&player) {
            continue;
        }
        let available = status[i].as_deref() == Some("a") && chance[i].is_none_or(|c| c >= 100);
        let settled = match (at[i], latest_card.get(&player)) {
            (Some(at), Some(&card)) => at > card + FLAG_SETTLE,
            _ => false,
        };
        if available && settled {
            cleared.insert(player);
        }
    }
    Ok(served.into_iter().filter(|((p, _), _)| !cleared.contains(p)).collect())
}

/// comp_start_sum / comp_rank of each row among the rows sharing `group` (+ position).
fn competition(groups: &[(i64, i64, u64)], start_rate_l5: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let rate: Vec<f64> = start_rate_l5.iter().map(|&r| if r.is_nan() { 0.0 } else { r }).collect();
    let mut members: HashMap<(i64, i64, u64), Vec<usize>> = HashMap::new();
    for (i, g) in groups.iter().enumerate() {
        members.entry(*g).or_default().push(i);
    }
    let mut sums = vec![0.0; rate.len()];
    let mut ranks = vec![0.0; rate.len()];
    for rows in members.values() {
        let mut total = Kahan::default();
        for &i in rows {
            total.add(rate[i]);
        }
        for &i in rows {
            sums[i] = total.value() - rate[i];
            ranks[i] = 1.0 + rows.iter().filter(|&&j| rate[j] > rate[i]).count() as f64;
        }
    }
    (sums, ranks)
}

fn element_group(element_type: f64) -> u64 {
    if element_type.is_nan() { u64::MAX } else { element_type.to_bits() }
}

fn rest_days(previous: Option<Micros>, kickoff: Micros) -> f64 {
    previous.map_or(REST_CAP_DAYS, |p| days_between(kickoff, p).min(REST_CAP_DAYS))
}

fn league_rows(m: &Matches, rows: impl Iterator<Item = usize>) -> Vec<LeagueRow> {
    rows.map(|i| LeagueRow {
        player_key: m.player_key[i],
        season: m.season[i],
        gw_index: m.gw_index[i],
        fixture_key: m.fixture_key[i],
        team_key: m.team_key[i],
        kickoff: m.kickoff[i],
        yellow_cards: m.yellow[i],
        red_cards: m.red[i],
    })
    .collect()
}

// --- training ------------------------------------------------------------------------------

/// One row per visible player_match row: keys, LAG_FEATURES, the targets, `start_inferred`,
/// `ban_kind` and `banned`; sorted by (player_key, season, gw_index, kickoff_time,
/// fixture_key).
pub fn training_frame(view: &AsOfView) -> Result<Table> {
    let all = matches(view)?;
    let mut order: Vec<usize> = (0..all.len()).collect();
    order.sort_by_key(|&i| {
        (all.player_key[i], all.season[i], all.gw_index[i], all.kickoff[i], all.fixture_key[i])
    });
    let m = reorder(&all, &order);
    let n = m.len();
    let rows = LagRows {
        player_key: m.player_key.clone(),
        season: m.season.clone(),
        gw_index: m.gw_index.clone(),
        team_key: m.team_key.clone(),
        kickoff: m.kickoff.clone(),
        ref_time: m.deadline.clone(),
        minutes: m.minutes.clone(),
        start: m.start.clone(),
        yellow: m.yellow.clone(),
        red: m.red.clone(),
    };
    let mut lag = lags(&rows);
    let cups = cup_rows(view)?;
    let (cup_minutes, cup_started, since) = cup_features(
        &cups,
        &m.player_key,
        &m.deadline,
        &lag.last_row,
        &lag.values["days_since_app"],
    );
    lag.values.insert("cup_minutes_7d", cup_minutes);
    lag.values.insert("cup_started_last", cup_started);
    lag.values.insert("days_since_app", since);

    let fixtures = team_fixtures(&m);
    let keys: Vec<(i64, i64, i64)> =
        (0..n).map(|i| (m.team_key[i], m.season[i], m.gw_index[i])).collect();
    let club_fixtures = fixtures_before(&fixtures, &keys);
    let share: Vec<f64> =
        (0..n)
            .map(|i| {
                if club_fixtures[i] > 0.0 {
                    lag.club_starts[i] / club_fixtures[i]
                } else {
                    f64::NAN
                }
            })
            .collect();

    // Per team fixture: rest days, its deadline (the first matching row's) and congestion.
    let mut deadline_of: HashMap<(i64, i64), Micros> = HashMap::new();
    for i in 0..n {
        deadline_of.entry((m.team_key[i], m.fixture_key[i])).or_insert(m.deadline[i]);
    }
    let fixture_rows: Vec<FixtureRow> = (0..fixtures.team_key.len())
        .map(|i| {
            let same_team = i > 0 && fixtures.team_key[i - 1] == fixtures.team_key[i];
            FixtureRow {
                team_key: fixtures.team_key[i],
                season: fixtures.season[i],
                kickoff: fixtures.kickoff[i],
                ref_time: deadline_of[&(fixtures.team_key[i], fixtures.fixture_key[i])],
                rest_days: rest_days(
                    same_team.then(|| fixtures.kickoff[i - 1]),
                    fixtures.kickoff[i],
                ),
            }
        })
        .collect();
    let seasons: BTreeSet<i64> = m.season.iter().copied().collect();
    let schedules: Vec<Table> = if seasons.is_empty() {
        vec![view.schedule(0)?]
    } else {
        seasons.iter().map(|&s| view.schedule(s)).collect::<Result<_>>()?
    };
    let crowded = congestion(&fixture_rows, &league_sides(&schedules)?, &club_sides(view)?);
    let mut per_fixture: HashMap<(i64, i64), (f64, (f64, f64, f64))> = HashMap::new();
    for i in 0..fixture_rows.len() {
        per_fixture.insert(
            (fixtures.team_key[i], fixtures.fixture_key[i]),
            (fixture_rows[i].rest_days, crowded[i]),
        );
    }

    let groups: Vec<(i64, i64, u64)> = (0..n)
        .map(|i| (m.fixture_key[i], m.team_key[i], element_group(m.element_type[i])))
        .collect();
    let (comp_sum, comp_rank) = competition(&groups, &lag.values["start_rate_l5"]);
    let continuity = club_continuity(&m, &m.team_key, &m.deadline);
    let league_fixtures: Vec<(i64, i64, Micros)> = (0..fixtures.team_key.len())
        .map(|i| (fixtures.team_key[i], fixtures.fixture_key[i], fixtures.kickoff[i]))
        .collect();
    let served = bans(view, &league_rows(&m, 0..n), &league_fixtures, None, false)?;

    let fixture_value = |i: usize, f: fn(&(f64, (f64, f64, f64))) -> f64| -> f64 {
        per_fixture.get(&(m.team_key[i], m.fixture_key[i])).map_or(f64::NAN, f)
    };
    let mut features: HashMap<&str, Vec<f64>> = lag.values.into_iter().collect();
    features.insert("element_type", m.element_type.clone());
    features.insert("price", m.price.clone());
    features.insert("start_share_season", share);
    features.insert("rest_days", (0..n).map(|i| fixture_value(i, |v| v.0)).collect());
    features.insert("rest_days_all", (0..n).map(|i| fixture_value(i, |v| v.1.0)).collect());
    features.insert("days_to_next", (0..n).map(|i| fixture_value(i, |v| v.1.1)).collect());
    features.insert("europe_week", (0..n).map(|i| fixture_value(i, |v| v.1.2)).collect());
    features.insert("comp_start_sum", comp_sum);
    features.insert("comp_rank", comp_rank);
    features.insert("club_continuity", continuity);
    features.insert("club_fixtures_season", club_fixtures);
    let ban_kind: Vec<f64> = (0..n)
        .map(|i| served.get(&(m.player_key[i], m.fixture_key[i])).map_or(0.0, |&k| k as f64))
        .collect();

    let mut out = Table::new()
        .with("player_key", Column::I64(m.player_key.clone()))
        .with("season", Column::I64(m.season.clone()))
        .with("fixture_key", Column::I64(m.fixture_key.clone()))
        .with("gw", Column::I64(m.gw.clone()))
        .with("gw_index", Column::I64(m.gw_index.clone()))
        .with("team_key", Column::I64(m.team_key.clone()))
        .with("kickoff_time", Column::Ts(m.kickoff.clone()));
    for name in LAG_FEATURES {
        out.push(name, floats(features.remove(name).with_context(|| format!("no lag {name}"))?));
    }
    let minutes = &m.minutes;
    out.push("minutes", floats(minutes.clone()));
    out.push("start", floats(m.start.clone()));
    out.push("sixty", floats(minutes.iter().map(|&x| if x >= 60.0 { 1.0 } else { 0.0 }).collect()));
    out.push(
        "sub",
        floats(
            (0..n).map(|i| if minutes[i] > 0.0 && m.start[i] == 0.0 { 1.0 } else { 0.0 }).collect(),
        ),
    );
    out.push("start_inferred", Column::Bool(m.start_inferred.clone()));
    let banned = ban_kind.iter().map(|&k| if k > 0.0 { 1.0 } else { 0.0 }).collect();
    out.push("ban_kind", floats(ban_kind));
    out.push("banned", floats(banned));
    Ok(out)
}

fn reorder(m: &Matches, order: &[usize]) -> Matches {
    fn pick<T: Copy>(v: &[T], order: &[usize]) -> Vec<T> {
        order.iter().map(|&i| v[i]).collect()
    }
    Matches {
        player_key: pick(&m.player_key, order),
        season: pick(&m.season, order),
        fixture_key: pick(&m.fixture_key, order),
        gw: pick(&m.gw, order),
        gw_index: pick(&m.gw_index, order),
        team_key: pick(&m.team_key, order),
        kickoff: pick(&m.kickoff, order),
        deadline: pick(&m.deadline, order),
        minutes: pick(&m.minutes, order),
        start: pick(&m.start, order),
        start_inferred: pick(&m.start_inferred, order),
        yellow: pick(&m.yellow, order),
        red: pick(&m.red, order),
        element_type: pick(&m.element_type, order),
        price: pick(&m.price, order),
    }
}

// --- prediction ----------------------------------------------------------------------------

/// A club's horizon fixture with its rest days and congestion as known at the deadline.
struct HorizonFixture {
    team_key: i64,
    fixture_key: i64,
    season: i64,
    gw: i64,
    gw_index: i64,
    horizon: i64,
    kickoff: Micros,
    rest_days: f64,
    congestion: (f64, f64, f64),
}

fn horizon_fixtures(view: &AsOfView, season: i64) -> Result<Vec<HorizonFixture>> {
    let upcoming = upcoming_fixtures(view)?;
    let fixture_keys = upcoming.opt_i64("fixture_key")?;
    let sides = league_sides(&[view.schedule(season)?])?;
    let mut rest: HashMap<(i64, i64), f64> = HashMap::new();
    for (i, &(team, _, fixture, kickoff)) in sides.iter().enumerate() {
        let previous = (i > 0 && sides[i - 1].0 == team).then(|| sides[i - 1].3);
        rest.insert((team, fixture), rest_days(previous, kickoff));
    }
    let teams = upcoming.i64("team_key")?;
    let gws = upcoming.i64("gw")?;
    let gw_index = upcoming.i64("gw_index")?;
    let horizon = upcoming.i64("horizon")?;
    let kickoffs = upcoming.opt_i64("kickoff_time")?;
    let mut out = Vec::new();
    for i in 0..upcoming.height() {
        let Some(fixture_key) = fixture_keys[i] else { continue };
        let kickoff = kickoffs[i].context("a horizon fixture without a kickoff time")?;
        out.push(HorizonFixture {
            team_key: teams[i],
            fixture_key,
            season,
            gw: gws[i],
            gw_index: gw_index[i],
            horizon: horizon[i],
            kickoff,
            rest_days: rest.get(&(teams[i], fixture_key)).copied().unwrap_or(REST_CAP_DAYS),
            congestion: (0.0, 0.0, 0.0),
        });
    }
    let rows: Vec<FixtureRow> = out
        .iter()
        .map(|h| FixtureRow {
            team_key: h.team_key,
            season,
            kickoff: h.kickoff,
            ref_time: view.deadline(),
            rest_days: h.rest_days,
        })
        .collect();
    for (h, c) in out.iter_mut().zip(congestion(&rows, &sides, &club_sides(view)?)) {
        h.congestion = c;
    }
    out.sort_by_key(|h| (h.team_key, h.kickoff, h.fixture_key));
    Ok(out)
}

/// One row per pool player and horizon fixture of his club: keys, LAG_FEATURES as of the
/// deadline and `ban_kind`; sorted by (player_key, horizon, kickoff_time, fixture_key).
pub fn prediction_frame(view: &AsOfView) -> Result<Table> {
    let deadline = view.deadline();
    let (season, gw) = view.gameweek_for_deadline()?;
    let gameweeks = view.table("gameweek", Some(&["season", "gw", "gw_index"]))?;
    let target = (0..gameweeks.height())
        .find(|&i| {
            gameweeks.i64("season").unwrap()[i] == season && gameweeks.i64("gw").unwrap()[i] == gw
        })
        .map(|i| gameweeks.i64("gw_index").unwrap()[i])
        .context("no target gameweek")?;
    let pool = player_pool(view)?;
    let pool_keys = pool.i64("player_key")?;
    let pool_teams = pool.i64("team_key")?;
    let m = matches(view)?;
    let fixtures = team_fixtures(&m);
    let in_pool: HashSet<i64> = pool_keys.iter().copied().collect();
    let history: Vec<usize> =
        (0..m.len()).filter(|&i| in_pool.contains(&m.player_key[i])).collect();

    // History rows then one query row per pool player (after every visible row).
    struct Row {
        player_key: i64,
        season: i64,
        gw_index: i64,
        kickoff: Micros,
        fixture_key: i64,
        team_key: i64,
        ref_time: Micros,
        minutes: f64,
        start: f64,
        yellow: f64,
        red: f64,
        query: bool,
    }
    let mut rows: Vec<Row> = history
        .iter()
        .map(|&i| Row {
            player_key: m.player_key[i],
            season: m.season[i],
            gw_index: m.gw_index[i],
            kickoff: m.kickoff[i],
            fixture_key: m.fixture_key[i],
            team_key: m.team_key[i],
            ref_time: m.deadline[i],
            minutes: m.minutes[i],
            start: m.start[i],
            yellow: m.yellow[i],
            red: m.red[i],
            query: false,
        })
        .collect();
    rows.extend(pool_keys.iter().zip(pool_teams).map(|(&player_key, &team_key)| Row {
        player_key,
        season,
        gw_index: target,
        kickoff: deadline,
        fixture_key: -1,
        team_key,
        ref_time: deadline,
        minutes: 0.0,
        start: 0.0,
        yellow: 0.0,
        red: 0.0,
        query: true,
    }));
    rows.sort_by_key(|r| (r.player_key, r.season, r.gw_index, r.kickoff, r.fixture_key));
    let lag_rows = LagRows {
        player_key: rows.iter().map(|r| r.player_key).collect(),
        season: rows.iter().map(|r| r.season).collect(),
        gw_index: rows.iter().map(|r| r.gw_index).collect(),
        team_key: rows.iter().map(|r| r.team_key).collect(),
        kickoff: rows.iter().map(|r| r.kickoff).collect(),
        ref_time: rows.iter().map(|r| r.ref_time).collect(),
        minutes: rows.iter().map(|r| r.minutes).collect(),
        start: rows.iter().map(|r| r.start).collect(),
        yellow: rows.iter().map(|r| r.yellow).collect(),
        red: rows.iter().map(|r| r.red).collect(),
    };
    let lag = lags(&lag_rows);
    let queried: Vec<usize> = positions(rows.len(), |i| rows[i].query);
    let players: Vec<i64> = queried.iter().map(|&i| rows[i].player_key).collect();
    let teams: Vec<i64> = queried.iter().map(|&i| rows[i].team_key).collect();
    let pick = |v: &[f64]| -> Vec<f64> { queried.iter().map(|&i| v[i]).collect() };
    let mut features: HashMap<&str, Vec<f64>> =
        lag.values.iter().map(|(k, v)| (*k, pick(v))).collect();
    let club_starts = pick(&lag.club_starts);
    let last_row = pick(&lag.last_row);
    let refs = vec![deadline; players.len()];
    let cups = cup_rows(view)?;
    let (cup_minutes, cup_started, since) =
        cup_features(&cups, &players, &refs, &last_row, &features["days_since_app"]);
    features.insert("cup_minutes_7d", cup_minutes);
    features.insert("cup_started_last", cup_started);
    features.insert("days_since_app", since);
    let mut pool_of: HashMap<i64, (f64, f64)> = HashMap::new();
    for (i, &k) in pool_keys.iter().enumerate() {
        pool_of.insert(k, (pool.i64("element_type")?[i] as f64, pool.i64("price")?[i] as f64));
    }
    let element_type: Vec<f64> =
        players.iter().map(|p| pool_of.get(p).map_or(f64::NAN, |v| v.0)).collect();
    features.insert(
        "price",
        players.iter().map(|p| pool_of.get(p).map_or(f64::NAN, |v| v.1)).collect(),
    );
    let club: Vec<(i64, i64, i64)> = teams.iter().map(|&t| (t, season, target)).collect();
    let club_fixtures = fixtures_before(&fixtures, &club);
    features.insert(
        "start_share_season",
        (0..players.len())
            .map(
                |i| {
                    if club_fixtures[i] > 0.0 {
                        club_starts[i] / club_fixtures[i]
                    } else {
                        f64::NAN
                    }
                },
            )
            .collect(),
    );
    features.insert("club_fixtures_season", club_fixtures);
    let groups: Vec<(i64, i64, u64)> =
        (0..players.len()).map(|i| (teams[i], 0, element_group(element_type[i]))).collect();
    let (comp_sum, comp_rank) = competition(&groups, &features["start_rate_l5"]);
    features.insert("comp_start_sum", comp_sum);
    features.insert("comp_rank", comp_rank);
    features.insert("club_continuity", club_continuity(&m, &teams, &refs));
    features.insert("element_type", element_type);

    let horizon = horizon_fixtures(view, season)?;
    let mut by_team: HashMap<i64, Vec<usize>> = HashMap::new();
    for (j, h) in horizon.iter().enumerate() {
        by_team.entry(h.team_key).or_default().push(j);
    }

    // Bans from the cards of the last BAN_LOOKBACK, over the fixtures since and the horizon.
    let since = deadline - BAN_LOOKBACK;
    let mut seen = HashSet::new();
    let league_fixtures: Vec<(i64, i64, Micros)> = (0..fixtures.team_key.len())
        .filter(|&i| fixtures.kickoff[i] >= since)
        .map(|i| (fixtures.team_key[i], fixtures.fixture_key[i], fixtures.kickoff[i]))
        .chain(horizon.iter().map(|h| (h.team_key, h.fixture_key, h.kickoff)))
        .filter(|&(team, fixture, _)| seen.insert((team, fixture)))
        .collect();
    let recent = league_rows(&m, history.iter().copied().filter(|&i| m.season[i] >= season - 1));
    let served = bans(view, &recent, &league_fixtures, Some(since), true)?;

    let mut out_rows: Vec<(usize, usize)> = Vec::new();
    for (i, team) in teams.iter().enumerate() {
        for &j in by_team.get(team).map_or(&[][..], Vec::as_slice) {
            out_rows.push((i, j));
        }
    }
    out_rows.sort_by_key(|&(i, j)| {
        (players[i], horizon[j].horizon, horizon[j].kickoff, horizon[j].fixture_key)
    });
    let h = |f: fn(&HorizonFixture) -> i64| -> Column {
        Column::I64(out_rows.iter().map(|&(_, j)| f(&horizon[j])).collect())
    };
    let mut out = Table::new()
        .with("player_key", Column::I64(out_rows.iter().map(|&(i, _)| players[i]).collect()))
        .with("fixture_key", h(|x| x.fixture_key))
        .with("team_key", h(|x| x.team_key))
        .with("season", h(|x| x.season))
        .with("gw", h(|x| x.gw))
        .with("gw_index", h(|x| x.gw_index))
        .with("horizon", h(|x| x.horizon))
        .with(
            "kickoff_time",
            Column::Ts(out_rows.iter().map(|&(_, j)| horizon[j].kickoff).collect()),
        );
    for name in LAG_FEATURES {
        let values: Vec<f64> = match *name {
            "rest_days" => out_rows.iter().map(|&(_, j)| horizon[j].rest_days).collect(),
            "rest_days_all" => out_rows.iter().map(|&(_, j)| horizon[j].congestion.0).collect(),
            "days_to_next" => out_rows.iter().map(|&(_, j)| horizon[j].congestion.1).collect(),
            "europe_week" => out_rows.iter().map(|&(_, j)| horizon[j].congestion.2).collect(),
            _ => {
                let player = features.get(name).with_context(|| format!("no lag {name}"))?;
                out_rows.iter().map(|&(i, _)| player[i]).collect()
            }
        };
        out.push(name, floats(values));
    }
    let ban_kind = out_rows
        .iter()
        .map(|&(i, j)| served.get(&(players[i], horizon[j].fixture_key)).map_or(0.0, |&k| k as f64))
        .collect();
    out.push("ban_kind", floats(ban_kind));
    Ok(out)
}
