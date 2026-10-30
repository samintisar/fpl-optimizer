//! `team_rating`: our own Elo ratings from football-data results, two rows per match plus
//! one pre-season row per club and season. Port of `fplopt.build.elo` (its module docstring
//! has the model, the timing and the checks).

use std::collections::{BTreeMap, BTreeSet, HashMap};

use anyhow::{Result, bail};

use crate::common::{
    BuildContext, HOUR, date_from_days, schedule_available_at, schedule_published_at, uk_wall_clock,
};
use crate::fixtures::{FIRST_SEASON, Fixture, football_data_results};
use crate::table::{Column, Micros, Table};
use crate::teams::TeamResolver;

const INITIAL_RATING: f64 = 1500.0;
const HOME_ADVANTAGE: f64 = 60.0;
const K_FACTOR: f64 = 20.0;
const CLUBS_PER_SEASON: usize = 20;
const MEAN_TOLERANCE: f64 = 50.0;
const PRE_FPL_KICKOFF_HOUR: u32 = 15;
const RESULT_DELAY: Micros = 2 * HOUR;
const ZERO_SUM_TOLERANCE: f64 = 1e-9;

pub const SORT_BY: &[&str] = &["event_time", "team_key"];

pub fn goal_multiplier(goal_difference: i64) -> f64 {
    match goal_difference.abs() {
        0 | 1 => 1.0,
        2 => 1.5,
        d => (11 + d) as f64 / 8.0,
    }
}

pub fn expected_home_score(home: f64, away: f64) -> f64 {
    1.0 / (1.0 + 10f64.powf(-(home + HOME_ADVANTAGE - away) / 400.0))
}

#[derive(Clone, Debug)]
struct Match {
    season: i64,
    fixture_key: Option<i64>,
    kickoff_time: Micros,
    home: i64,
    away: i64,
    home_goals: i64,
    away_goals: i64,
}

#[derive(Clone, Debug)]
struct Rating {
    team_key: i64,
    season: i64,
    fixture_key: Option<i64>,
    kickoff_time: Option<Micros>,
    opponent_team_key: Option<i64>,
    is_home: Option<bool>,
    rating_before: f64,
    rating_after: f64,
    expected_score: Option<f64>,
    event_time: Micros,
    available_at: Micros,
}

fn elo_ratings(matches: &[Match], season_clubs: &BTreeMap<i64, BTreeSet<i64>>) -> Vec<Rating> {
    let mut ordered: Vec<&Match> = matches.iter().collect();
    ordered.sort_by_key(|m| (m.kickoff_time, m.home));
    let mut clubs = season_clubs.clone();
    for m in &ordered {
        clubs.entry(m.season).or_default().extend([m.home, m.away]);
    }
    let mut ratings: HashMap<i64, f64> = HashMap::new();
    let mut previous: Option<&BTreeSet<i64>> = None;
    let mut last_kickoff: Option<Micros> = None;
    let mut out = Vec::new();
    for (&season, current) in &clubs {
        match previous {
            None => {
                for &club in current {
                    ratings.insert(club, INITIAL_RATING);
                }
            }
            Some(previous) => {
                let relegated: Vec<i64> = previous.difference(current).copied().collect();
                let seed = if relegated.is_empty() {
                    INITIAL_RATING
                } else {
                    relegated.iter().map(|c| ratings[c]).sum::<f64>() / relegated.len() as f64
                };
                for club in current.difference(previous) {
                    ratings.insert(*club, seed);
                }
            }
        }
        let start = schedule_available_at(season, last_kickoff);
        for &club in current {
            out.push(Rating {
                team_key: club,
                season,
                fixture_key: None,
                kickoff_time: None,
                opponent_team_key: None,
                is_home: None,
                rating_before: ratings[&club],
                rating_after: ratings[&club],
                expected_score: None,
                event_time: start,
                available_at: start,
            });
        }
        let rows: Vec<&&Match> = ordered.iter().filter(|m| m.season == season).collect();
        for m in &rows {
            let (before_h, before_a) = (ratings[&m.home], ratings[&m.away]);
            let e_home = expected_home_score(before_h, before_a);
            let score = match m.home_goals.cmp(&m.away_goals) {
                std::cmp::Ordering::Greater => 1.0,
                std::cmp::Ordering::Equal => 0.5,
                std::cmp::Ordering::Less => 0.0,
            };
            let delta = K_FACTOR * goal_multiplier(m.home_goals - m.away_goals) * (score - e_home);
            ratings.insert(m.home, before_h + delta);
            ratings.insert(m.away, before_a - delta);
            for (team, opponent, home, before, expected) in [
                (m.home, m.away, true, before_h, e_home),
                (m.away, m.home, false, before_a, 1.0 - e_home),
            ] {
                out.push(Rating {
                    team_key: team,
                    season,
                    fixture_key: m.fixture_key,
                    kickoff_time: Some(m.kickoff_time),
                    opponent_team_key: Some(opponent),
                    is_home: Some(home),
                    rating_before: before,
                    rating_after: ratings[&team],
                    expected_score: Some(expected),
                    event_time: m.kickoff_time,
                    available_at: m.kickoff_time + RESULT_DELAY,
                });
            }
        }
        if let Some(last) = rows.iter().map(|m| m.kickoff_time).max() {
            last_kickoff = Some(last);
        }
        previous = Some(current);
    }
    out
}

/// Mean end-of-season (or latest) rating of each season's clubs.
fn league_means(rows: &[Rating]) -> BTreeMap<i64, f64> {
    let mut ordered: Vec<&Rating> = rows.iter().collect();
    ordered.sort_by_key(|r| r.event_time);
    let mut last: BTreeMap<(i64, i64), f64> = BTreeMap::new();
    for r in ordered {
        last.insert((r.season, r.team_key), r.rating_after);
    }
    let mut sums: BTreeMap<i64, (f64, usize)> = BTreeMap::new();
    for ((season, _), rating) in last {
        let e = sums.entry(season).or_default();
        e.0 += rating;
        e.1 += 1;
    }
    sums.into_iter().map(|(s, (sum, n))| (s, sum / n as f64)).collect()
}

fn validate(rows: &[Rating]) -> Result<()> {
    let mut problems = Vec::new();
    let mut keys = std::collections::HashSet::new();
    let mut clubs: BTreeMap<i64, BTreeSet<i64>> = BTreeMap::new();
    let mut changes: HashMap<(Micros, i64, i64), (usize, f64)> = HashMap::new();
    for r in rows {
        if !keys.insert((r.team_key, r.event_time)) {
            problems.push(format!(
                "(team_key, event_time) not unique: {} {}",
                r.team_key, r.event_time
            ));
        }
        clubs.entry(r.season).or_default().insert(r.team_key);
        match r.kickoff_time {
            Some(kickoff) => {
                let opponent = r.opponent_team_key.unwrap_or(r.team_key);
                let ok = r.event_time == kickoff
                    && r.available_at == kickoff + RESULT_DELAY
                    && r.expected_score.is_some_and(|e| e > 0.0 && e <= 1.0)
                    && opponent != r.team_key
                    && r.fixture_key.is_some() == (r.season >= FIRST_SEASON);
                if !ok {
                    problems.push(format!("match row {} {}", r.team_key, kickoff));
                }
                let pair = (kickoff, r.team_key.min(opponent), r.team_key.max(opponent));
                let e = changes.entry(pair).or_default();
                e.0 += 1;
                e.1 += r.rating_after - r.rating_before;
            }
            None => {
                if r.rating_before != r.rating_after
                    || r.available_at < schedule_published_at(r.season)
                {
                    problems.push(format!("pre-season row {} {}", r.team_key, r.season));
                }
            }
        }
    }
    if changes.values().any(|(n, sum)| *n != 2 || sum.abs() >= ZERO_SUM_TOLERANCE) {
        problems.push("two rows per match with rating changes summing to 0".into());
    }
    if clubs.values().any(|c| c.len() > CLUBS_PER_SEASON) {
        problems.push("at most 20 clubs per season".into());
    }
    if league_means(rows)
        .iter()
        .any(|(s, m)| *s >= FIRST_SEASON && (m - INITIAL_RATING).abs() > MEAN_TOLERANCE)
    {
        problems.push("league mean rating within 1500 ± 50 from 2016/17".into());
    }
    if !problems.is_empty() {
        bail!(
            "table 'team_rating' failed validation: {}",
            problems[..problems.len().min(20)].join("; ")
        );
    }
    Ok(())
}

pub fn build_team_rating(ctx: &mut BuildContext) -> Result<Table> {
    let resolver = TeamResolver::new(ctx.table("team_dim")?)?;
    let fixtures = Fixture::from_table(ctx.table("fixture")?)?;
    let results = football_data_results(ctx, &resolver, None)?;
    let by_sides: HashMap<(i64, i64, i64), &Fixture> =
        fixtures.iter().map(|f| ((f.season, f.home_team_key, f.away_team_key), f)).collect();
    let mut matches = Vec::new();
    let mut bad = Vec::new();
    for r in &results {
        let fixture = by_sides.get(&(r.season, r.home_team_key, r.away_team_key));
        let fpl_era = r.season >= FIRST_SEASON;
        if fpl_era && fixture.is_none_or(|f| f.fd_date != Some(r.fd_date)) {
            bad.push(format!(
                "{} {} {} {}",
                r.season,
                r.home_team_key,
                r.away_team_key,
                date_from_days(r.fd_date)
            ));
            continue;
        }
        let kickoff_time = match (fpl_era, fixture) {
            (true, Some(f)) => match f.kickoff_time {
                Some(k) => k,
                None => bail!("fixture {} has no kickoff", f.fixture_key),
            },
            _ => uk_wall_clock(date_from_days(r.fd_date), PRE_FPL_KICKOFF_HOUR),
        };
        matches.push(Match {
            season: r.season,
            fixture_key: fixture.filter(|_| fpl_era).map(|f| f.fixture_key),
            kickoff_time,
            home: r.home_team_key,
            away: r.away_team_key,
            home_goals: r.home_goals,
            away_goals: r.away_goals,
        });
    }
    if !bad.is_empty() {
        bail!(
            "{} football-data result(s) from 2016/17 without a fixture of the same date (rebuild `fixture` after a football-data refresh):\n{}",
            bad.len(),
            bad[..bad.len().min(10)].join("\n")
        );
    }
    let mut season_clubs: BTreeMap<i64, BTreeSet<i64>> = BTreeMap::new();
    for m in &matches {
        season_clubs.entry(m.season).or_default().extend([m.home, m.away]);
    }
    for f in &fixtures {
        season_clubs.entry(f.season).or_default().extend([f.home_team_key, f.away_team_key]);
    }
    let rows = elo_ratings(&matches, &season_clubs);
    validate(&rows)?;
    let means = league_means(&rows);
    let recent: Vec<f64> =
        means.iter().filter(|(s, _)| **s >= FIRST_SEASON).map(|(_, m)| *m).collect();
    log::info!(
        "team_rating: league mean end-of-season rating {:.3} to {:.3} (2016/17+: {:.3} to {:.3})",
        means.values().copied().fold(f64::INFINITY, f64::min),
        means.values().copied().fold(f64::NEG_INFINITY, f64::max),
        recent.iter().copied().fold(f64::INFINITY, f64::min),
        recent.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    );
    Ok(Table::new()
        .with("team_key", Column::I64(rows.iter().map(|r| r.team_key).collect()))
        .with("season", Column::I64(rows.iter().map(|r| r.season).collect()))
        .with("fixture_key", Column::OptI64(rows.iter().map(|r| r.fixture_key).collect()))
        .with("kickoff_time", Column::OptTs(rows.iter().map(|r| r.kickoff_time).collect()))
        .with(
            "opponent_team_key",
            Column::OptI64(rows.iter().map(|r| r.opponent_team_key).collect()),
        )
        .with("is_home", Column::OptBool(rows.iter().map(|r| r.is_home).collect()))
        .with("rating_before", Column::F64(rows.iter().map(|r| r.rating_before).collect()))
        .with("rating_after", Column::F64(rows.iter().map(|r| r.rating_after).collect()))
        .with("expected_score", Column::OptF64(rows.iter().map(|r| r.expected_score).collect()))
        .with("event_time", Column::Ts(rows.iter().map(|r| r.event_time).collect()))
        .with("available_at", Column::Ts(rows.iter().map(|r| r.available_at).collect())))
}
