//! ESPN summaries and scoreboards -> `espn_match`, `espn_player_match`,
//! `espn_schedule_snapshot`, `espn_map` and `club_match`. Port of `fplopt.build.espn` (its
//! module docstring has the minutes rules, the timing and the build checks).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use fplopt_archive::raw_store::RawStore;
use rayon::prelude::*;
use serde_json::Value;

use crate::common::{
    BuildContext, DAY, HOUR, file_name, json_bool, json_i64, micros, parse_iso, season_of, subdirs,
};
use crate::fixtures::Fixture;
use crate::players::{PlayerMatch, PlayerSeason};
use crate::table::{Column, Micros, Table};
use crate::teams::TeamResolver;
use crate::understat::{
    MINUTES_TOLERANCE, MapRow, candidate_pairs, fpl_appearances, fpl_name_variants, map_source,
    map_table, read_overrides, score_names, validate_map,
};

const AFTER_EXTRA_TIME: &str = "STATUS_FINAL_AET";
const ON_PENALTIES: &str = "STATUS_FINAL_PEN";
const FINISHED_STATUSES: [&str; 3] = ["STATUS_FULL_TIME", AFTER_EXTRA_TIME, ON_PENALTIES];
const MINUTE_EVENTS: [&str; 3] = ["Substitution", "Yellow Card", "Red Card"];
const SOURCE: &str = "espn";
const AVAILABLE_AFTER: Micros = 3 * HOUR;
const FIRST_ROUND_NOTICE: Micros = 14 * DAY;
const ID: &str = "espn_athlete_id";
const MIN_MINUTES_AGREEMENT: f64 = 0.9;
const MAX_RED_DISAGREEMENT: f64 = 0.02;
const MAX_LISTED: usize = 20;
const GRACE: Micros = 2 * DAY;

const MATCH_SORT_BY: &[&str] = &["kickoff_time", "espn_event_id"];
const PLAYER_SORT_BY: &[&str] = &["espn_event_id", "espn_athlete_id"];
const SCHEDULE_SORT_BY: &[&str] = &["snapshot_at", "espn_event_id"];
const MAP_SORT_BY: &[&str] = &["espn_athlete_id"];
const CLUB_MATCH_SORT_BY: &[&str] = &["kickoff_time", "espn_event_id"];

fn minute(seconds: f64, end: i64) -> i64 {
    ((seconds / 60.0).floor() as i64 + 1).min(end)
}

fn obj<'a>(value: &'a Value, key: &str) -> &'a Value {
    value.get(key).unwrap_or(&Value::Null)
}

fn period(ev: &Value) -> i64 {
    obj(ev, "period").get("number").and_then(Value::as_i64).unwrap_or(0)
}

fn event_order(ev: &Value) -> (i64, f64) {
    (period(ev), obj(ev, "clock").get("value").and_then(Value::as_f64).unwrap_or(0.0))
}

fn season(payload: &Value, kickoff: Micros) -> Result<i64> {
    match obj(payload, "season").get("year") {
        Some(year) if !year.is_null() => json_i64(year, "season year"),
        _ => Ok(season_of(kickoff)),
    }
}

fn went_to_extra_time(status: &str, events: &[Value]) -> bool {
    if status == AFTER_EXTRA_TIME {
        return true;
    }
    if status != ON_PENALTIES {
        return false;
    }
    events.iter().any(|ev| {
        !json_bool(obj(ev, "shootout"))
            && (period(ev) >= 3
                || obj(obj(ev, "type"), "text").as_str() == Some("Start Extra Time"))
    })
}

fn sides(competitors: &Value) -> HashMap<String, String> {
    competitors
        .as_array()
        .into_iter()
        .flatten()
        .map(|c| {
            (c["homeAway"].as_str().unwrap_or_default().to_string(), id_text(&c["team"]["id"]))
        })
        .collect()
}

fn id_text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

#[derive(Clone, Debug)]
struct Match {
    espn_event_id: i64,
    league: String,
    season: i64,
    kickoff_time: Micros,
    home_espn_id: String,
    away_espn_id: String,
    home_team_key: Option<i64>,
    away_team_key: Option<i64>,
    fixture_key: Option<i64>,
    status: String,
    extra_time: bool,
}

#[derive(Clone, Debug)]
struct Player {
    espn_athlete_id: i64,
    espn_event_id: i64,
    espn_team_id: String,
    season: i64,
    league: String,
    team_key: i64,
    athlete_name: String,
    starter: bool,
    minute_on: Option<i64>,
    minute_off: Option<i64>,
    minutes: Option<i64>,
    yellows: i64,
    red: bool,
    second_yellow: bool,
    kickoff_time: Micros,
}

/// One summary -> its match row and one row per squad player.
fn parse_summary(payload: &Value, league: &str) -> Result<(Match, Vec<Player>)> {
    let header = &payload["header"];
    let competition = &header["competitions"][0];
    let status = competition["status"]["type"]["name"]
        .as_str()
        .context("summary without a status")?
        .to_string();
    let key_events: Vec<Value> = obj(payload, "keyEvents").as_array().cloned().unwrap_or_default();
    let end = if went_to_extra_time(&status, &key_events) { 120 } else { 90 };
    let kickoff = parse_iso(competition["date"].as_str().context("summary without a date")?)?;
    let sides = sides(&competition["competitors"]);
    let m = Match {
        espn_event_id: json_i64(&header["id"], "event id")?,
        league: league.to_string(),
        season: season(header, kickoff)?,
        kickoff_time: kickoff,
        home_espn_id: sides.get("home").cloned().context("no home side")?,
        away_espn_id: sides.get("away").cloned().context("no away side")?,
        home_team_key: None,
        away_team_key: None,
        fixture_key: None,
        status,
        extra_time: end == 120,
    };
    let (mut on, mut off): (HashMap<i64, i64>, HashMap<i64, i64>) =
        (HashMap::new(), HashMap::new());
    let mut replaced = HashSet::new();
    let mut yellows: HashMap<i64, i64> = HashMap::new();
    let mut red_at: HashMap<i64, i64> = HashMap::new();
    let mut second_yellow = HashSet::new();
    let mut no_clock = 0;
    let mut ordered: Vec<&Value> = key_events.iter().collect();
    ordered.sort_by(|a, b| {
        let (pa, ca) = event_order(a);
        let (pb, cb) = event_order(b);
        pa.cmp(&pb).then(ca.total_cmp(&cb))
    });
    for ev in ordered {
        let text = obj(obj(ev, "type"), "text").as_str().unwrap_or_default();
        if json_bool(obj(ev, "shootout")) || !MINUTE_EVENTS.contains(&text) {
            continue;
        }
        let Some(seconds) = obj(ev, "clock").get("value").and_then(Value::as_f64) else {
            no_clock += 1;
            continue;
        };
        let ids: Vec<i64> = obj(ev, "participants")
            .as_array()
            .into_iter()
            .flatten()
            .map(|p| json_i64(&p["athlete"]["id"], "athlete id"))
            .collect::<Result<_>>()?;
        let minute = minute(seconds, end);
        if text == "Substitution" && ids.len() == 2 {
            let (player_in, player_out) = (ids[0], ids[1]);
            if off.contains_key(&player_out) {
                continue; // already sent off or replaced: a data glitch
            }
            on.entry(player_in).or_insert(minute);
            off.insert(player_out, minute);
            replaced.insert(player_out);
        } else if text == "Yellow Card" && !ids.is_empty() {
            *yellows.entry(ids[0]).or_default() += 1;
        } else if text == "Red Card" && !ids.is_empty() {
            if yellows.get(&ids[0]).copied().unwrap_or(0) > 0 {
                second_yellow.insert(ids[0]);
            }
            red_at.insert(ids[0], minute);
            off.entry(ids[0]).or_insert(minute);
        }
    }
    if no_clock > 0 {
        log::warn!(
            "espn event {}: {no_clock} key event(s) without a clock skipped",
            m.espn_event_id
        );
    }
    let mut players = Vec::new();
    let mut seen = HashSet::new();
    for roster in obj(payload, "rosters").as_array().into_iter().flatten() {
        let team = id_text(&roster["team"]["id"]);
        for entry in obj(roster, "roster").as_array().into_iter().flatten() {
            let athlete = json_i64(&entry["athlete"]["id"], "athlete id")?;
            if !seen.insert(athlete) {
                continue; // ESPN lists some goalkeepers twice
            }
            let unknown = (json_bool(obj(entry, "subbedIn")) && !on.contains_key(&athlete))
                || (json_bool(obj(entry, "subbedOut"))
                    && !replaced.contains(&athlete)
                    && !red_at.contains_key(&athlete));
            let starter = json_bool(obj(entry, "starter"));
            let start = if starter { Some(0) } else { on.get(&athlete).copied() };
            let stop = if start.is_some() && !unknown {
                Some(off.get(&athlete).copied().unwrap_or(end))
            } else {
                None
            };
            let minutes = if unknown {
                None
            } else {
                match start {
                    Some(start) => Some((stop.unwrap() - start).max(1)),
                    None => Some(0),
                }
            };
            players.push(Player {
                espn_athlete_id: athlete,
                espn_event_id: m.espn_event_id,
                espn_team_id: team.clone(),
                season: m.season,
                league: league.to_string(),
                team_key: 0,
                athlete_name: entry["athlete"].get("displayName").map(id_text).unwrap_or_default(),
                starter,
                minute_on: start,
                minute_off: stop,
                minutes,
                yellows: yellows.get(&athlete).copied().unwrap_or(0),
                red: red_at.contains_key(&athlete),
                second_yellow: second_yellow.contains(&athlete),
                kickoff_time: kickoff,
            });
        }
    }
    Ok((m, players))
}

/// (league, item, directory) of every `espn/<league>/<kind>/<item>` directory, sorted.
fn item_dirs(ctx: &BuildContext, kind: &str) -> Result<Vec<(String, String, PathBuf)>> {
    let mut out = Vec::new();
    for league in subdirs(&ctx.root().join(SOURCE))? {
        for item in subdirs(&league.join(kind))? {
            out.push((file_name(&league).to_string(), file_name(&item).to_string(), item));
        }
    }
    out.sort_by(|a, b| a.2.cmp(&b.2));
    Ok(out)
}

fn summaries(
    ctx: &BuildContext,
    resolver: &TeamResolver,
    fixtures: &[Fixture],
    problems: &mut Vec<String>,
) -> Result<(Vec<Match>, Vec<Player>)> {
    let fixture_of: HashMap<(i64, i64, i64), i64> = fixtures
        .iter()
        .map(|f| ((f.season, f.home_team_key, f.away_team_key), f.fixture_key))
        .collect();
    let items = item_dirs(ctx, "summary")?;
    let parsed: Vec<Result<Option<(Match, Vec<Player>)>>> = items
        .par_iter()
        .map(|(league, event, _)| {
            let Some((_, path)) =
                ctx.store.entries(SOURCE, &format!("{league}/summary/{event}"), ".json.gz")?.pop()
            else {
                return Ok(None);
            };
            let payload = RawStore::read_json(&path)?;
            let status = payload["header"]["competitions"][0]["status"]["type"]["name"]
                .as_str()
                .unwrap_or_default();
            if !FINISHED_STATUSES.contains(&status) {
                return Ok(None);
            }
            parse_summary(&payload, league).map(Some)
        })
        .collect();
    let (mut matches, mut players) = (Vec::new(), Vec::new());
    let mut unknown: BTreeMap<String, usize> = BTreeMap::new();
    for result in parsed {
        let Some((mut m, rows)) = result? else { continue };
        let home = resolver.find("espn", &m.home_espn_id);
        let away = resolver.find("espn", &m.away_espn_id);
        if m.league == "eng.1" {
            m.fixture_key =
                fixture_of.get(&(m.season, home.unwrap_or(-1), away.unwrap_or(-1))).copied();
            if m.fixture_key.is_none() {
                problems.push(format!(
                    "eng.1 event {} ({} v {}, {}) has no fixture",
                    m.espn_event_id,
                    m.home_espn_id,
                    m.away_espn_id,
                    crate::common::from_micros(m.kickoff_time).format("%Y-%m-%d")
                ));
                continue;
            }
        }
        if home.is_none() && away.is_none() {
            continue;
        }
        m.home_team_key = home;
        m.away_team_key = away;
        for mut row in rows {
            if let Some(team_key) = resolver.find("espn", &row.espn_team_id) {
                *unknown.entry(m.league.clone()).or_default() += usize::from(row.minutes.is_none());
                row.team_key = team_key;
                players.push(row);
            }
        }
        matches.push(m);
    }
    let mut events_of: BTreeMap<i64, usize> = BTreeMap::new();
    for m in &matches {
        if let Some(k) = m.fixture_key {
            *events_of.entry(k).or_default() += 1;
        }
    }
    for (k, n) in events_of {
        if n > 1 {
            problems.push(format!("fixture {k} has {n} eng.1 events"));
        }
    }
    for (league, n) in unknown {
        if n > 0 {
            log::info!(
                "espn: minutes unknown (substitution flag without its event) in {league}: {n} row(s)"
            );
        }
    }
    Ok((matches, players))
}

#[derive(Clone, Debug)]
struct Scheduled {
    snapshot_at: Micros,
    espn_event_id: i64,
    league: String,
    season: i64,
    kickoff_time: Micros,
    home_team_key: Option<i64>,
    away_team_key: Option<i64>,
}

fn schedule(ctx: &BuildContext, resolver: &TeamResolver) -> Result<Vec<Scheduled>> {
    let mut rows = Vec::new();
    let mut seen = HashSet::new();
    for (league, period, _) in item_dirs(ctx, "scoreboard")? {
        for (ts, path) in
            ctx.store.entries(SOURCE, &format!("{league}/scoreboard/{period}"), ".json.gz")?
        {
            let snapshot_at = micros(&ts);
            let payload = RawStore::read_json(&path)?;
            for ev in obj(&payload, "events").as_array().into_iter().flatten() {
                let kickoff = parse_iso(ev["date"].as_str().context("event without a date")?)?;
                if ev["status"]["type"].get("state").and_then(Value::as_str) != Some("pre")
                    || kickoff <= snapshot_at
                {
                    continue;
                }
                let sides = sides(&ev["competitions"][0]["competitors"]);
                let home = sides.get("home").and_then(|id| resolver.find("espn", id));
                let away = sides.get("away").and_then(|id| resolver.find("espn", id));
                if home.is_none() && away.is_none() {
                    continue;
                }
                let espn_event_id = json_i64(&ev["id"], "event id")?;
                if !seen.insert((snapshot_at, espn_event_id)) {
                    continue;
                }
                rows.push(Scheduled {
                    snapshot_at,
                    espn_event_id,
                    league: league.clone(),
                    season: season(ev, kickoff)?,
                    kickoff_time: kickoff,
                    home_team_key: home,
                    away_team_key: away,
                });
            }
        }
    }
    Ok(rows)
}

/// ESPN rows of matches with a PL fixture: (athlete, fixture_key, minutes, red, name).
fn pl_appearances<'a>(
    matches: &[Match],
    players: &'a [Player],
) -> Vec<(i64, i64, Option<i64>, bool, &'a str)> {
    let fixture: HashMap<i64, i64> =
        matches.iter().filter_map(|m| m.fixture_key.map(|k| (m.espn_event_id, k))).collect();
    players
        .iter()
        .filter_map(|p| {
            fixture
                .get(&p.espn_event_id)
                .map(|&k| (p.espn_athlete_id, k, p.minutes, p.red, p.athlete_name.as_str()))
        })
        .collect()
}

fn map_espn(
    ctx: &BuildContext,
    matches: &[Match],
    players: &[Player],
    player_match: &[PlayerMatch],
    player_season: &[PlayerSeason],
) -> Result<Vec<MapRow>> {
    let played: Vec<_> = pl_appearances(matches, players)
        .into_iter()
        .filter(|a| a.2.is_some_and(|m| m > 0))
        .collect();
    if played.is_empty() {
        return Ok(Vec::new());
    }
    let apps: Vec<(i64, i64, i64)> = played.iter().map(|a| (a.0, a.1, a.2.unwrap())).collect();
    let mut pairs = candidate_pairs(&apps, &fpl_appearances(player_match));
    let names: HashMap<i64, String> = played.iter().map(|a| (a.0, a.4.to_string())).collect();
    score_names(&mut pairs, &names, &fpl_name_variants(player_season));
    let ids: HashSet<i64> = played.iter().map(|a| a.0).collect();
    let overrides: Vec<_> = read_overrides(&ctx.config_dir, SOURCE)?
        .into_iter()
        .filter(|o| o.0.is_none_or(|id| ids.contains(&id)))
        .collect();
    Ok(map_source(&pairs, &overrides))
}

fn player_problems(
    ctx: &BuildContext,
    matches: &[Match],
    players: &[Player],
    mapping: &[MapRow],
    player_match: &[PlayerMatch],
    seasons: &BTreeSet<i64>,
) -> Result<Vec<String>> {
    if seasons.is_empty() {
        return Ok(Vec::new());
    }
    let mut problems = Vec::new();
    let pm: Vec<&PlayerMatch> =
        player_match.iter().filter(|r| seasons.contains(&r.season)).collect();
    let exempt: HashSet<i64> = read_overrides(&ctx.config_dir, SOURCE)?
        .into_iter()
        .filter(|o| o.0.is_none())
        .filter_map(|o| o.1)
        .collect();
    let mapped: HashMap<i64, (i64, &str)> =
        mapping.iter().filter_map(|m| m.player_key.map(|k| (m.id, (k, m.method)))).collect();
    let mapped_keys: HashSet<i64> = mapped.values().map(|v| v.0).collect();
    let unmapped: BTreeSet<i64> = pm
        .iter()
        .filter(|r| r.count("minutes") > 0)
        .map(|r| r.player_key)
        .filter(|k| !mapped_keys.contains(k) && !exempt.contains(k))
        .collect();
    if !unmapped.is_empty() {
        problems.push(format!(
            "{} FPL players with minutes are not mapped: {:?}",
            unmapped.len(),
            unmapped.iter().take(MAX_LISTED).collect::<Vec<_>>()
        ));
    }
    let fpl: HashMap<(i64, i64), (i64, i64)> = pm
        .iter()
        .map(|r| ((r.player_key, r.fixture_key), (r.count("minutes"), r.count("red_cards"))))
        .collect();
    let mut agreement: BTreeMap<i64, (usize, usize)> = BTreeMap::new();
    let (mut wrong, mut n_red) = (Vec::new(), 0usize);
    for (athlete, fixture, minutes_src, red, _) in pl_appearances(matches, players) {
        let Some(&(player_key, method)) = mapped.get(&athlete) else { continue };
        let Some(&(minutes, red_cards)) = fpl.get(&(player_key, fixture)) else { continue };
        if let Some(src) = minutes_src
            && method == "auto"
            && (minutes > 0 || src > 0)
        {
            let e = agreement.entry(athlete).or_default();
            e.0 += 1;
            e.1 += usize::from((minutes - src).abs() <= MINUTES_TOLERANCE);
        }
        let fpl_red = red_cards > 0;
        n_red += usize::from(fpl_red);
        if fpl_red != red {
            wrong.push(format!("{athlete} {player_key} {fixture} {red} {red_cards}"));
        }
    }
    let mut low: Vec<(i64, f64)> = agreement
        .iter()
        .map(|(a, (n, close))| (*a, *close as f64 / *n as f64))
        .filter(|(_, share)| *share < MIN_MINUTES_AGREEMENT)
        .collect();
    low.sort_by(|a, b| a.1.total_cmp(&b.1));
    if !low.is_empty() {
        let shown: Vec<String> = low
            .iter()
            .take(MAX_LISTED)
            .map(|(a, s)| format!("{a}: {}", (s * 1000.0).round() / 1000.0))
            .collect();
        problems.push(format!(
            "{} auto pairs agree on minutes in < 90% of fixtures (add an override if the pair is right): {{{}}}",
            low.len(),
            shown.join(", ")
        ));
    }
    if !wrong.is_empty() {
        log::info!(
            "espn: red cards disagree with FPL in {} row(s): {}",
            wrong.len(),
            wrong.join("; ")
        );
    }
    if wrong.len() as f64 > MAX_RED_DISAGREEMENT * n_red.max(1) as f64 {
        problems.push(format!("red cards disagree in {} of {n_red} FPL reds", wrong.len()));
    }
    Ok(problems)
}

fn espn_problems(
    ctx: &BuildContext,
    matches: &[Match],
    players: &[Player],
    mapping: &[MapRow],
    fixtures: &[Fixture],
    player_match: &[PlayerMatch],
) -> Result<Vec<String>> {
    let pl: Vec<&Match> = matches.iter().filter(|m| m.league == "eng.1").collect();
    let covered: BTreeSet<i64> = pl.iter().map(|m| m.season).collect();
    let Some(&newest) = covered.iter().max() else { return Ok(Vec::new()) };
    let mut problems = Vec::new();
    let cutoff = pl.iter().map(|m| m.kickoff_time).max().unwrap() - GRACE;
    let have: HashSet<i64> = matches.iter().filter_map(|m| m.fixture_key).collect();
    let missing: Vec<i64> = fixtures
        .iter()
        .filter(|f| {
            f.finished && covered.contains(&f.season) && f.kickoff_time.is_some_and(|k| k < cutoff)
        })
        .filter(|f| !have.contains(&f.fixture_key))
        .map(|f| f.fixture_key)
        .collect();
    if !missing.is_empty() {
        problems.push(format!(
            "{} finished PL fixtures have no ESPN match: {:?}",
            missing.len(),
            &missing[..missing.len().min(MAX_LISTED)]
        ));
    }
    let older: BTreeSet<i64> = covered.iter().copied().filter(|s| *s != newest).collect();
    problems.extend(player_problems(ctx, matches, players, mapping, player_match, &older)?);
    for warning in
        player_problems(ctx, matches, players, mapping, player_match, &BTreeSet::from([newest]))?
    {
        log::warn!("espn (season {newest}, in progress or newest): {warning}");
    }
    Ok(problems)
}

/// Every non-league match of a PL club, played or scheduled, with when it was known.
fn club_match(matches: &[Match], schedule: &[Scheduled]) -> Table {
    struct Row {
        espn_event_id: i64,
        league: String,
        season: i64,
        kickoff_time: Micros,
        home: Option<i64>,
        away: Option<i64>,
        played: bool,
    }
    let mut out: Vec<Row> = matches
        .iter()
        .filter(|m| m.league != "eng.1")
        .map(|m| Row {
            espn_event_id: m.espn_event_id,
            league: m.league.clone(),
            season: m.season,
            kickoff_time: m.kickoff_time,
            home: m.home_team_key,
            away: m.away_team_key,
            played: true,
        })
        .collect();
    let done: HashSet<i64> = out.iter().map(|r| r.espn_event_id).collect();
    let mut snaps: Vec<&Scheduled> = schedule.iter().filter(|s| s.league != "eng.1").collect();
    snaps.sort_by_key(|s| s.snapshot_at);
    let mut first_seen: HashMap<i64, Micros> = HashMap::new();
    let mut newest: BTreeMap<i64, &Scheduled> = BTreeMap::new();
    let mut order = Vec::new();
    for s in &snaps {
        first_seen.entry(s.espn_event_id).or_insert(s.snapshot_at);
        if newest.insert(s.espn_event_id, s).is_none() {
            order.push(s.espn_event_id);
        }
    }
    for id in order {
        if done.contains(&id) {
            continue;
        }
        let s = newest[&id];
        out.push(Row {
            espn_event_id: id,
            league: s.league.clone(),
            season: s.season,
            kickoff_time: s.kickoff_time,
            home: s.home_team_key,
            away: s.away_team_key,
            played: false,
        });
    }
    out.sort_by_key(|r| (r.kickoff_time, r.espn_event_id));
    // Rule time per PL side: the club's previous match in the league and season.
    let mut sides: Vec<(i64, i64, &str, i64, Micros)> = Vec::new(); // team, event, league, season, kickoff
    for home in [true, false] {
        for r in &out {
            if let Some(team) = if home { r.home } else { r.away } {
                sides.push((team, r.espn_event_id, &r.league, r.season, r.kickoff_time));
            }
        }
    }
    sides.sort_by_key(|s| (s.4, s.1));
    let mut previous: HashMap<(i64, &str, i64), Micros> = HashMap::new();
    let mut rule: HashMap<i64, Micros> = HashMap::new();
    for (team, event, league, season, kickoff) in sides {
        let time =
            previous.get(&(team, league, season)).copied().unwrap_or(kickoff - FIRST_ROUND_NOTICE);
        previous.insert((team, league, season), kickoff);
        let e = rule.entry(event).or_insert(time);
        *e = (*e).max(time);
    }
    let available: Vec<Micros> = out
        .iter()
        .map(|r| {
            let mut known = rule[&r.espn_event_id];
            if let Some(&seen) = first_seen.get(&r.espn_event_id)
                && seen < known
            {
                known = seen;
            }
            known.min(r.kickoff_time - 1)
        })
        .collect();
    Table::new()
        .with("espn_event_id", Column::I64(out.iter().map(|r| r.espn_event_id).collect()))
        .with("league", Column::Str(out.iter().map(|r| r.league.clone()).collect()))
        .with("season", Column::I64(out.iter().map(|r| r.season).collect()))
        .with("kickoff_time", Column::Ts(out.iter().map(|r| r.kickoff_time).collect()))
        .with("home_team_key", Column::OptI64(out.iter().map(|r| r.home).collect()))
        .with("away_team_key", Column::OptI64(out.iter().map(|r| r.away).collect()))
        .with("played", Column::Bool(out.iter().map(|r| r.played).collect()))
        .with("event_time", Column::Ts(out.iter().map(|r| r.kickoff_time).collect()))
        .with("available_at", Column::Ts(available))
}

fn match_table(rows: &[Match]) -> Table {
    Table::new()
        .with("espn_event_id", Column::I64(rows.iter().map(|r| r.espn_event_id).collect()))
        .with("league", Column::Str(rows.iter().map(|r| r.league.clone()).collect()))
        .with("season", Column::I64(rows.iter().map(|r| r.season).collect()))
        .with("kickoff_time", Column::Ts(rows.iter().map(|r| r.kickoff_time).collect()))
        .with("home_espn_id", Column::Str(rows.iter().map(|r| r.home_espn_id.clone()).collect()))
        .with("away_espn_id", Column::Str(rows.iter().map(|r| r.away_espn_id.clone()).collect()))
        .with("home_team_key", Column::OptI64(rows.iter().map(|r| r.home_team_key).collect()))
        .with("away_team_key", Column::OptI64(rows.iter().map(|r| r.away_team_key).collect()))
        .with("fixture_key", Column::OptI64(rows.iter().map(|r| r.fixture_key).collect()))
        .with("status", Column::Str(rows.iter().map(|r| r.status.clone()).collect()))
        .with("extra_time", Column::Bool(rows.iter().map(|r| r.extra_time).collect()))
        .with("event_time", Column::Ts(rows.iter().map(|r| r.kickoff_time).collect()))
        .with(
            "available_at",
            Column::Ts(rows.iter().map(|r| r.kickoff_time + AVAILABLE_AFTER).collect()),
        )
}

fn player_table(rows: &[Player]) -> Table {
    Table::new()
        .with(ID, Column::I64(rows.iter().map(|r| r.espn_athlete_id).collect()))
        .with("espn_event_id", Column::I64(rows.iter().map(|r| r.espn_event_id).collect()))
        .with("season", Column::I64(rows.iter().map(|r| r.season).collect()))
        .with("league", Column::Str(rows.iter().map(|r| r.league.clone()).collect()))
        .with("team_key", Column::I64(rows.iter().map(|r| r.team_key).collect()))
        .with("athlete_name", Column::Str(rows.iter().map(|r| r.athlete_name.clone()).collect()))
        .with("starter", Column::Bool(rows.iter().map(|r| r.starter).collect()))
        .with("minute_on", Column::OptI64(rows.iter().map(|r| r.minute_on).collect()))
        .with("minute_off", Column::OptI64(rows.iter().map(|r| r.minute_off).collect()))
        .with("minutes", Column::OptI64(rows.iter().map(|r| r.minutes).collect()))
        .with("yellows", Column::I64(rows.iter().map(|r| r.yellows).collect()))
        .with("red", Column::Bool(rows.iter().map(|r| r.red).collect()))
        .with("second_yellow", Column::Bool(rows.iter().map(|r| r.second_yellow).collect()))
        .with("event_time", Column::Ts(rows.iter().map(|r| r.kickoff_time).collect()))
        .with(
            "available_at",
            Column::Ts(rows.iter().map(|r| r.kickoff_time + AVAILABLE_AFTER).collect()),
        )
}

fn schedule_table(rows: &[Scheduled]) -> Table {
    Table::new()
        .with("snapshot_at", Column::Ts(rows.iter().map(|r| r.snapshot_at).collect()))
        .with("espn_event_id", Column::I64(rows.iter().map(|r| r.espn_event_id).collect()))
        .with("league", Column::Str(rows.iter().map(|r| r.league.clone()).collect()))
        .with("season", Column::I64(rows.iter().map(|r| r.season).collect()))
        .with("kickoff_time", Column::Ts(rows.iter().map(|r| r.kickoff_time).collect()))
        .with("home_team_key", Column::OptI64(rows.iter().map(|r| r.home_team_key).collect()))
        .with("away_team_key", Column::OptI64(rows.iter().map(|r| r.away_team_key).collect()))
        .with("event_time", Column::Ts(rows.iter().map(|r| r.snapshot_at).collect()))
        .with("available_at", Column::Ts(rows.iter().map(|r| r.snapshot_at).collect()))
}

/// Writes the ESPN tables (nothing if a check fails). Returns (player rows, their seasons).
pub fn build_espn(ctx: &mut BuildContext) -> Result<(usize, Vec<i64>)> {
    let resolver = TeamResolver::new(ctx.table("team_dim")?)?;
    let fixtures = Fixture::from_table(ctx.table("fixture")?)?;
    let player_match = PlayerMatch::from_table(ctx.table("player_match")?)?;
    let player_season = PlayerSeason::from_table(ctx.table("player_season")?)?;
    let mut problems = Vec::new();
    let (matches, players) = summaries(ctx, &resolver, &fixtures, &mut problems)?;
    if !problems.is_empty() {
        bail!(
            "{} ESPN problems: {}",
            problems.len(),
            problems[..problems.len().min(20)].join("; ")
        );
    }
    let scheduled = schedule(ctx, &resolver)?;
    let mapping = map_espn(ctx, &matches, &players, &player_match, &player_season)?;
    let problems = espn_problems(ctx, &matches, &players, &mapping, &fixtures, &player_match)?;
    if !problems.is_empty() {
        bail!("{} ESPN problems: {}", problems.len(), problems.join("; "));
    }
    let ids: HashSet<i64> = matches.iter().map(|m| m.espn_event_id).collect();
    let keys: HashSet<(i64, i64)> =
        players.iter().map(|p| (p.espn_athlete_id, p.espn_event_id)).collect();
    if ids.len() != matches.len()
        || keys.len() != players.len()
        || players
            .iter()
            .any(|p| p.minutes.is_some_and(|m| !(0..=120).contains(&m)) || p.yellows < 0)
    {
        bail!("ESPN tables failed validation (unique keys, minutes 0-120)");
    }
    validate_map(&mapping, "espn_map")?;
    let clubs = club_match(&matches, &scheduled);
    if clubs.i64("available_at")?.iter().zip(clubs.i64("kickoff_time")?).any(|(a, k)| a >= k) {
        bail!("table 'club_match' failed validation: a club match is known before it kicks off");
    }
    crate::write_table(ctx, "espn_match", match_table(&matches), MATCH_SORT_BY)?;
    crate::write_table(ctx, "espn_player_match", player_table(&players), PLAYER_SORT_BY)?;
    crate::write_table(
        ctx,
        "espn_schedule_snapshot",
        schedule_table(&scheduled),
        SCHEDULE_SORT_BY,
    )?;
    crate::write_table(ctx, "espn_map", map_table(&mapping, ID), MAP_SORT_BY)?;
    crate::write_table(ctx, "club_match", clubs, CLUB_MATCH_SORT_BY)?;
    for name in
        ["espn_match", "espn_player_match", "espn_schedule_snapshot", "espn_map", "club_match"]
    {
        ctx.forget(name);
    }
    let with_minutes: HashSet<i64> = pl_appearances(&matches, &players)
        .into_iter()
        .filter(|a| a.2.is_some_and(|m| m > 0))
        .map(|a| a.0)
        .collect();
    log::info!(
        "espn_map: {} athletes mapped ({} override row(s)) of {} with PL minutes",
        mapping.iter().filter(|m| m.player_key.is_some()).count(),
        mapping.iter().filter(|m| m.method == "override").count(),
        with_minutes.len()
    );
    Ok((players.len(), players.iter().map(|p| p.season).collect()))
}
