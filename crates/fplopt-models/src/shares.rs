//! Player goal and assist shares, and penalties (Python `fplopt.models.shares`).
//!
//! - npxG / xA per player-match from the first source available (Understat, else FPL's xG
//!   minus 0.79 per penalty attempt scaled by `k_goals`; penalty goals from Understat or
//!   inferred from FPL with π per taker group), k and π fitted where both sources exist.
//! - Marcel prior shares: (Σ w·X + μ·D0) / (Σ w·D + D0) over the current season and the 3
//!   before, μ a position + price quasi-Poisson prior, × a finishing factor.
//! - Per team-fixture: non-penalty goals λ_np split by g·f, FPL assists by a·f, penalties
//!   along the club's taker ranking.
//!
//! Sums follow pandas: group sums Kahan-compensated (`group_sum`), `Series.sum` numpy's
//! pairwise sum with NaN as 0, `bincount` / `cumsum` sequential.

use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::Result;
use fplopt_features::baseline::player_pool;
use fplopt_features::util::Kahan;
use fplopt_store::{AsOfView, Column, Micros, Table};

use crate::gbm::float_column;
use crate::minutes::Minutes;
use crate::util::{np_maximum, np_sum, opt_ts, pd_sum};

pub const FPL_PEN_XG: f64 = 0.79;
const US_PEN_XG: f64 = 0.76;
const FIT_SEASONS: i64 = 3;
const DEFAULT_PEN_RATE: f64 = 0.13;
const DEFAULT_CONVERSION: f64 = 0.78;
const DEFAULT_ASSIST_FRACTION: f64 = 0.88;
const DEFAULT_OWN_GOAL_FRACTION: f64 = 0.035;
const DEFAULT_TAKER_Q: f64 = 0.85;
const DEFAULT_PEN_PI: (f64, f64) = (0.5, 0.2);
const LEAGUE_PRIOR_MATCHES: f64 = 100.0;
const LEAGUE_PRIOR_ATTEMPTS: f64 = 20.0;
const TAKER_PRIOR_ATTEMPTS: f64 = 10.0;
const PI_PRIOR: f64 = 10.0;
const K_PRIOR: f64 = 5.0;
const SNAPSHOT_TOLERANCE: Micros = 14 * 86_400_000_000;
const PRICE_SCALE: f64 = 10.0;
const PRICE_ITERATIONS: usize = 30;
const PRICE_RIDGE: f64 = 1.0;
const RATE_EPS: f64 = 1e-9;

#[derive(Clone, Debug, PartialEq)]
pub struct SharesParams {
    pub prior_minutes: f64,
    pub season_weights: Vec<f64>,
    pub club_change_weight: f64,
    pub goals_weight: f64,
    pub price_prior_all: bool,
    pub pen_team_prior: f64,
    pub conversion_prior: f64,
    pub taker_prior: f64,
    pub finishing_prior: Option<f64>,
}

impl Default for SharesParams {
    fn default() -> Self {
        SharesParams {
            prior_minutes: 1920.0,
            season_weights: vec![2.0, 2.0, 1.0, 1.0],
            club_change_weight: 0.5,
            goals_weight: 0.5,
            price_prior_all: true,
            pen_team_prior: 38.0,
            conversion_prior: 20.0,
            taker_prior: 1.0,
            finishing_prior: Some(25.0),
        }
    }
}

/// League-level parameters fitted at the cutoff.
#[derive(Clone, Debug)]
pub struct SharesFit {
    pub params: SharesParams,
    pub k_goals: f64,
    pub k_assists: f64,
    pub pen_pi: (f64, f64),
    pub ratio_goals: f64,
    pub ratio_assists: f64,
    pub ratio_team: f64,
    pub team_output: f64,
    pub position_goal: Vec<(i64, f64)>,
    pub position_assist: Vec<(i64, f64)>,
    pub price_goal: Vec<(i64, f64, f64, f64)>,
    pub price_assist: Vec<(i64, f64, f64, f64)>,
    pub pen_rate: f64,
    pub conversion: f64,
    pub team_pen: Vec<(i64, f64, f64)>,
    pub assist_fraction: f64,
    pub own_goal_fraction: f64,
    pub taker_q: f64,
    pub n_rows: usize,
}

// --- inputs ---------------------------------------------------------------------------------

/// One player-match row with its npxG / xA columns.
#[derive(Clone, Debug)]
struct Row {
    player_key: i64,
    season: i64,
    fixture_key: i64,
    team_key: i64,
    kickoff: Option<Micros>,
    minutes: f64,
    goals_scored: f64,
    assists: f64,
    penalties_missed: f64,
    own_goals: f64,
    fpl_xg: f64,
    element_type: f64,
    price: f64,
    npxg: f64,
    xa: f64,
    pen_goals: f64,
    pen_attempts: f64,
}

fn nan_to_num(v: f64) -> f64 {
    if v.is_nan() { 0.0 } else { v }
}

/// The raw match columns `match_xg` reads.
struct Raw {
    us_goals: f64,
    us_npg: f64,
    us_npxg: f64,
    us_xa: f64,
    fpl_xa: f64,
    main_taker: bool,
}

/// Visible player_match rows of seasons >= `min_season` (all if None), sorted by (player,
/// season, kickoff, fixture); with element_type and the GW price when all seasons are read.
fn matches(view: &AsOfView, min_season: Option<i64>) -> Result<(Vec<Row>, Vec<Raw>)> {
    let columns = [
        "player_key",
        "season",
        "fixture_key",
        "gw",
        "team_key",
        "kickoff_time",
        "minutes",
        "goals_scored",
        "assists",
        "penalties_missed",
        "own_goals",
        "fpl_xg",
        "fpl_xa",
        "us_goals",
        "us_npg",
        "us_npxg",
        "us_xa",
    ]; // fmt: skip
    let t = view.table("player_match", Some(&columns))?;
    let f = |name: &str| float_column(&t, name);
    let (player, season, fixture, gw, team) = (
        t.i64("player_key")?,
        t.i64("season")?,
        t.i64("fixture_key")?,
        t.i64("gw")?,
        t.i64("team_key")?,
    );
    let kickoff = opt_ts(&t, "kickoff_time")?;
    let (minutes, goals, assists, missed, own) =
        (f("minutes")?, f("goals_scored")?, f("assists")?, f("penalties_missed")?, f("own_goals")?);
    let (fpl_xg, fpl_xa, us_goals, us_npg, us_npxg, us_xa) =
        (f("fpl_xg")?, f("fpl_xa")?, f("us_goals")?, f("us_npg")?, f("us_npxg")?, f("us_xa")?);
    let mut rows: Vec<usize> =
        (0..t.height()).filter(|&i| min_season.is_none_or(|m| season[i] >= m)).collect();
    // Element type and price (the fit only): the newest visible player_gw row of the GW,
    // else the player_season's element type.
    let mut registered: HashMap<(i64, i64, i64), (Option<i64>, Option<i64>)> = HashMap::new();
    let mut positions: HashMap<(i64, i64), Option<i64>> = HashMap::new();
    if min_season.is_none() {
        let gws = view
            .table("player_gw", Some(&["player_key", "season", "gw", "element_type", "value"]))?;
        let (gp, gs, gg) = (gws.i64("player_key")?, gws.i64("season")?, gws.i64("gw")?);
        let (ge, gv) = (gws.opt_i64("element_type")?, gws.opt_i64("value")?);
        for i in 0..gws.height() {
            registered.insert((gp[i], gs[i], gg[i]), (ge[i], gv[i]));
        }
        let ps = view.table("player_season", Some(&["player_key", "season", "element_type"]))?;
        let (pp, psn, pe) = (ps.i64("player_key")?, ps.i64("season")?, ps.opt_i64("element_type")?);
        for i in 0..ps.height() {
            positions.insert((pp[i], psn[i]), pe[i]);
        }
    }
    rows.sort_by_key(|&i| (player[i], season[i], kickoff[i].is_none(), kickoff[i], fixture[i]));
    let mut out = Vec::with_capacity(rows.len());
    let mut raw = Vec::with_capacity(rows.len());
    for i in rows {
        let (element_type, price) = if min_season.is_none() {
            let (e, v) =
                registered.get(&(player[i], season[i], gw[i])).copied().unwrap_or((None, None));
            let e = e.or_else(|| positions.get(&(player[i], season[i])).copied().flatten());
            (e.map_or(f64::NAN, |e| e as f64), v.map_or(f64::NAN, |v| v as f64))
        } else {
            (f64::NAN, f64::NAN)
        };
        out.push(Row {
            player_key: player[i],
            season: season[i],
            fixture_key: fixture[i],
            team_key: team[i],
            kickoff: kickoff[i],
            minutes: minutes[i],
            goals_scored: goals[i],
            assists: assists[i],
            penalties_missed: missed[i],
            own_goals: own[i],
            fpl_xg: fpl_xg[i],
            element_type,
            price,
            npxg: f64::NAN,
            xa: f64::NAN,
            pen_goals: f64::NAN,
            pen_attempts: f64::NAN,
        });
        raw.push(Raw {
            us_goals: us_goals[i],
            us_npg: us_npg[i],
            us_npxg: us_npxg[i],
            us_xa: us_xa[i],
            fpl_xa: fpl_xa[i],
            main_taker: false,
        });
    }
    Ok((out, raw))
}

/// Per snapshot time and club: the rank-1 player of the club's `penalties_order` listing
/// (lowest order, ties by player_key); per club, sorted by time.
fn main_takers(view: &AsOfView) -> Result<HashMap<i64, Vec<(Micros, i64)>>> {
    let snaps = view.table(
        "player_snapshot",
        Some(&["snapshot_at", "player_key", "team_key", "penalties_order"]),
    )?;
    let at = opt_ts(&snaps, "snapshot_at")?;
    let (player, team) = (snaps.i64("player_key")?, snaps.opt_i64("team_key")?);
    let order = snaps.opt_i64("penalties_order")?;
    let mut best: BTreeMap<(Micros, i64), (i64, i64)> = BTreeMap::new();
    for i in 0..snaps.height() {
        let (Some(o), Some(t), Some(a)) = (order[i], team[i], at[i]) else { continue };
        let entry = best.entry((a, t)).or_insert((o, player[i]));
        if (o, player[i]) < *entry {
            *entry = (o, player[i]);
        }
    }
    let mut out: HashMap<i64, Vec<(Micros, i64)>> = HashMap::new();
    for ((a, t), (_, p)) in best {
        out.entry(t).or_default().push((a, p));
    }
    Ok(out)
}

/// The club's main taker in the newest listing strictly before kickoff, within
/// SNAPSHOT_TOLERANCE (`merge_asof`); None without one.
fn taker_at(
    takers: &HashMap<i64, Vec<(Micros, i64)>>,
    team: i64,
    kickoff: Option<Micros>,
) -> Option<i64> {
    let kickoff = kickoff?;
    let list = takers.get(&team)?;
    let below = list.partition_point(|(a, _)| *a < kickoff);
    if below == 0 {
        return None;
    }
    let (at, taker) = list[below - 1];
    (kickoff - at <= SNAPSHOT_TOLERANCE).then_some(taker)
}

/// Penalty-goal candidates from FPL alone: min(goals, ⌊(fpl_xg − 0.79·missed)/0.79⌋).
fn candidates(row: &Row) -> f64 {
    let missed = nan_to_num(row.penalties_missed);
    let goals = nan_to_num(row.goals_scored);
    let room = ((row.fpl_xg - FPL_PEN_XG * missed + 1e-6) / FPL_PEN_XG).floor();
    goals.min(nan_to_num(room).max(0.0))
}

/// `match_xg`: fills npxg, xa, pen_goals, pen_attempts.
fn match_xg(rows: &mut [Row], raw: &[Raw], k_goals: f64, k_assists: f64, pen_pi: (f64, f64)) {
    for (row, r) in rows.iter_mut().zip(raw) {
        let missed = nan_to_num(row.penalties_missed);
        let us_pen = r.us_goals - r.us_npg;
        let inferred = candidates(row) * if r.main_taker { pen_pi.0 } else { pen_pi.1 };
        let pen_goals = if !us_pen.is_nan() {
            us_pen
        } else if row.fpl_xg.is_nan() {
            f64::NAN
        } else {
            inferred
        };
        let attempts = pen_goals + missed;
        let clipped = row.fpl_xg - FPL_PEN_XG * attempts;
        let fpl_npxg = k_goals * if clipped.is_nan() { clipped } else { clipped.max(0.0) };
        row.npxg = if !r.us_npxg.is_nan() {
            r.us_npxg
        } else if !row.fpl_xg.is_nan() {
            fpl_npxg
        } else {
            f64::NAN
        };
        row.xa = if !r.us_xa.is_nan() { r.us_xa } else { k_assists * r.fpl_xa };
        row.pen_goals = pen_goals;
        row.pen_attempts = attempts;
    }
}

fn shrunk_ratio(numerator: f64, denominator: f64, prior: f64, weight: f64) -> f64 {
    (numerator + prior * weight) / (denominator + weight)
}

/// k_goals, k_assists and π from the visible rows where Understat and FPL both exist.
fn source_constants(rows: &[Row], raw: &[Raw]) -> (f64, f64, (f64, f64)) {
    let both: Vec<usize> = (0..rows.len())
        .filter(|&i| !raw[i].us_npxg.is_nan() && !rows[i].fpl_xg.is_nan() && rows[i].minutes > 0.0)
        .collect();
    if both.is_empty() {
        return (1.0, 1.0, DEFAULT_PEN_PI);
    }
    let cands: Vec<f64> = both.iter().map(|&i| candidates(&rows[i])).collect();
    let us_pen: Vec<f64> = both.iter().map(|&i| raw[i].us_goals - raw[i].us_npg).collect();
    let main: Vec<bool> = both.iter().map(|&i| raw[i].main_taker).collect();
    let pi = |group: &dyn Fn(usize) -> bool, default: f64| -> f64 {
        let numerator: Vec<f64> =
            (0..both.len()).filter(|&k| group(k) && cands[k] > 0.0).map(|k| us_pen[k]).collect();
        let denominator: Vec<f64> =
            (0..both.len()).filter(|&k| group(k)).map(|k| cands[k]).collect();
        shrunk_ratio(np_sum(&numerator), np_sum(&denominator), default, PI_PRIOR)
    };
    let pi = (pi(&|k| main[k], DEFAULT_PEN_PI.0), pi(&|k| !main[k], DEFAULT_PEN_PI.1));
    let fpl_npxg: Vec<f64> = both
        .iter()
        .map(|&i| {
            let attempts = raw[i].us_goals - raw[i].us_npg + rows[i].penalties_missed;
            let v = rows[i].fpl_xg - FPL_PEN_XG * attempts;
            if v.is_nan() { v } else { v.max(0.0) }
        })
        .collect();
    let us_npxg: Vec<f64> = both.iter().map(|&i| raw[i].us_npxg).collect();
    let k_goals = shrunk_ratio(pd_sum(&us_npxg), np_sum(&fpl_npxg), 1.0, K_PRIOR);
    let xa: Vec<usize> = both
        .iter()
        .copied()
        .filter(|&i| !raw[i].us_xa.is_nan() && !raw[i].fpl_xa.is_nan())
        .collect();
    let us_xa: Vec<f64> = xa.iter().map(|&i| raw[i].us_xa).collect();
    let fpl_xa: Vec<f64> = xa.iter().map(|&i| raw[i].fpl_xa).collect();
    let k_assists = shrunk_ratio(pd_sum(&us_xa), pd_sum(&fpl_xa), 1.0, K_PRIOR);
    (k_goals, k_assists, pi)
}

// --- rows: X and D per player-match ---------------------------------------------------------

/// One visible team-fixture.
#[derive(Clone, Debug)]
struct TeamFixture {
    fixture_key: i64,
    team_key: i64,
    season: i64,
    goals_for: f64,
    team_npxg: f64,
    team_attempts: f64,
    team_pen_goals: f64,
    assists: f64,
    own_goals: f64,
}

fn team_fixtures(
    view: &AsOfView,
    rows: &[Row],
    min_season: Option<i64>,
) -> Result<Vec<TeamFixture>> {
    let t = view.table(
        "team_match",
        Some(&["fixture_key", "team_key", "season", "goals_for", "us_xg", "us_npxg"]),
    )?;
    let (fixture, team, season) = (t.i64("fixture_key")?, t.i64("team_key")?, t.i64("season")?);
    let (goals, us_xg, us_npxg) =
        (float_column(&t, "goals_for")?, float_column(&t, "us_xg")?, float_column(&t, "us_npxg")?);
    // Played rows' sums per (fixture, team), Kahan as pandas' groupby sum.
    #[derive(Default)]
    struct Sums {
        npxg: Kahan,
        npxg_missing: usize,
        att: Kahan,
        pen: Kahan,
        att_missing: usize,
    }
    let mut sums: HashMap<(i64, i64), Sums> = HashMap::new();
    let mut totals: HashMap<(i64, i64), (Kahan, Kahan)> = HashMap::new();
    for r in rows {
        let key = (r.fixture_key, r.team_key);
        if r.minutes > 0.0 {
            let s = sums.entry(key).or_default();
            if r.npxg.is_nan() {
                s.npxg_missing += 1;
            } else {
                s.npxg.add(r.npxg);
            }
            if r.pen_attempts.is_nan() {
                s.att_missing += 1;
            } else {
                s.att.add(r.pen_attempts);
            }
            if !r.pen_goals.is_nan() {
                s.pen.add(r.pen_goals);
            }
        }
        let tot = totals.entry(key).or_default();
        if !r.assists.is_nan() {
            tot.0.add(r.assists);
        }
        if !r.own_goals.is_nan() {
            tot.1.add(r.own_goals);
        }
    }
    let mut out = Vec::new();
    for i in 0..t.height() {
        if min_season.is_some_and(|m| season[i] < m) {
            continue;
        }
        let key = (fixture[i], team[i]);
        let s = sums.get(&key);
        let npxg_known = s.is_some_and(|s| s.npxg_missing == 0);
        let att_known = s.is_some_and(|s| s.att_missing == 0);
        let team_npxg = if !us_npxg[i].is_nan() {
            us_npxg[i]
        } else if npxg_known {
            s.unwrap().npxg.value()
        } else {
            f64::NAN
        };
        let us_attempts = ((us_xg[i] - us_npxg[i]) / US_PEN_XG).round_ties_even();
        let attempts = if !us_attempts.is_nan() {
            us_attempts
        } else if att_known {
            s.unwrap().att.value()
        } else {
            f64::NAN
        };
        let (assists, own) = totals.get(&key).map_or((0.0, 0.0), |t| (t.0.value(), t.1.value()));
        out.push(TeamFixture {
            fixture_key: fixture[i],
            team_key: team[i],
            season: season[i],
            goals_for: goals[i],
            team_npxg,
            team_attempts: attempts,
            team_pen_goals: if att_known { s.unwrap().pen.value() } else { f64::NAN },
            assists,
            own_goals: own,
        });
    }
    Ok(out)
}

/// One played row with X and D (`_share_rows`).
#[derive(Clone, Debug)]
struct ShareRow {
    row: Row,
    x_goals: f64,
    fb_goals: bool,
    x_assists: f64,
    fb_assists: bool,
    d: f64,
    on_pitch: f64,
}

fn share_rows(rows: &[Row], teams: &[TeamFixture], ratios: (f64, f64, f64)) -> Vec<ShareRow> {
    let (ratio_goals, ratio_assists, ratio_team) = ratios;
    let by_key: HashMap<(i64, i64), &TeamFixture> =
        teams.iter().map(|t| ((t.fixture_key, t.team_key), t)).collect();
    rows.iter()
        .filter(|r| r.minutes > 0.0)
        .filter_map(|r| {
            let team = by_key.get(&(r.fixture_key, r.team_key))?;
            let on_pitch = r.minutes / 90.0;
            let team_goals = team.goals_for * ratio_team;
            let d = if team.team_npxg.is_nan() { team_goals } else { team.team_npxg } * on_pitch;
            let goals = nan_to_num(r.goals_scored) * ratio_goals;
            let assists = nan_to_num(r.assists) * ratio_assists;
            Some(ShareRow {
                row: r.clone(),
                x_goals: if r.npxg.is_nan() { goals } else { r.npxg },
                fb_goals: r.npxg.is_nan(),
                x_assists: if r.xa.is_nan() { assists } else { r.xa },
                fb_assists: r.xa.is_nan(),
                d,
                on_pitch,
            })
        })
        .collect()
}

type Constants = (f64, f64, (f64, f64));

/// (player rows, team-fixtures, source constants) of seasons >= `min_season` (all if
/// None); the constants are estimated when `fit` is None, else taken from it.
fn rows(
    view: &AsOfView,
    fit: Option<&SharesFit>,
    min_season: Option<i64>,
) -> Result<(Vec<Row>, Vec<TeamFixture>, Constants)> {
    let (mut matches, mut raw) = matches(view, min_season)?;
    if matches.iter().any(|r| !r.fpl_xg.is_nan()) {
        let takers = main_takers(view)?;
        for (row, r) in matches.iter().zip(raw.iter_mut()) {
            if !row.fpl_xg.is_nan() {
                r.main_taker = taker_at(&takers, row.team_key, row.kickoff) == Some(row.player_key);
            }
        }
    }
    let constants = match fit {
        None => source_constants(&matches, &raw),
        Some(fit) => (fit.k_goals, fit.k_assists, fit.pen_pi),
    };
    match_xg(&mut matches, &raw, constants.0, constants.1, constants.2);
    let teams = team_fixtures(view, &matches, min_season)?;
    Ok((matches, teams, constants))
}

// --- the Marcel prior -----------------------------------------------------------------------

/// (a, b, center): quasi-Poisson X ~ D·exp(a + b·(price − center)/10) by Newton steps.
fn price_model(seasons: &[(f64, f64, f64)]) -> (f64, f64, f64) {
    // seasons: (price, x, d)
    let rows: Vec<&(f64, f64, f64)> =
        seasons.iter().filter(|(p, _, d)| *d > 0.0 && !p.is_nan()).collect();
    let xs: Vec<f64> = rows.iter().map(|r| r.1).collect();
    if rows.is_empty() || pd_sum(&xs) <= 0.0 {
        let all_x: Vec<f64> = seasons.iter().map(|r| r.1).collect();
        let all_d: Vec<f64> = seasons.iter().map(|r| r.2).collect();
        let total = pd_sum(&all_x) / pd_sum(&all_d).max(RATE_EPS);
        return (total.max(RATE_EPS).ln(), 0.0, 0.0);
    }
    let weights: Vec<f64> = rows.iter().map(|r| r.2).collect();
    let price: Vec<f64> = rows.iter().map(|r| r.0).collect();
    let products: Vec<f64> = price.iter().zip(&weights).map(|(p, w)| p * w).collect();
    let center = np_sum(&products) / np_sum(&weights);
    let z: Vec<f64> = price.iter().map(|p| (p - center) / PRICE_SCALE).collect();
    let mut theta = [(np_sum(&xs) / np_sum(&weights)).ln(), 0.0];
    for _ in 0..PRICE_ITERATIONS {
        let mu: Vec<f64> =
            (0..z.len()).map(|i| weights[i] * (theta[0] + z[i] * theta[1]).exp()).collect();
        let resid: Vec<f64> = (0..z.len()).map(|i| mu[i] - xs[i]).collect();
        let g0 = resid.iter().sum::<f64>();
        let g1 = (0..z.len()).map(|i| z[i] * resid[i]).sum::<f64>() + PRICE_RIDGE * theta[1];
        let h00 = mu.iter().sum::<f64>();
        let h01 = (0..z.len()).map(|i| mu[i] * z[i]).sum::<f64>();
        let h11 = (0..z.len()).map(|i| mu[i] * z[i] * z[i]).sum::<f64>() + PRICE_RIDGE;
        let step = solve2([[h00, h01], [h01, h11]], [g0, g1]);
        theta = [theta[0] - step[0], theta[1] - step[1]];
    }
    (theta[0], theta[1], center)
}

/// `np.linalg.solve` of a 2 × 2 system (LU with partial pivoting, as LAPACK's getrf).
fn solve2(a: [[f64; 2]; 2], b: [f64; 2]) -> [f64; 2] {
    let (mut a, mut b) = (a, b);
    if a[1][0].abs() > a[0][0].abs() {
        a.swap(0, 1);
        b.swap(0, 1);
    }
    let l = a[1][0] * (1.0 / a[0][0]);
    let u11 = a[1][1] - l * a[0][1];
    let y1 = b[1] - l * b[0];
    let x1 = y1 / u11;
    let x0 = (b[0] - a[0][1] * x1) / a[0][0];
    [x0, x1]
}

fn price_prior(table: &[(i64, f64, f64, f64)], element_type: f64, price: f64) -> f64 {
    for &(position, a, b, center) in table {
        if element_type == position as f64 {
            return (a + b * (price - center) / PRICE_SCALE).exp();
        }
    }
    f64::NAN
}

fn position_prior(table: &[(i64, f64)], element_type: f64) -> f64 {
    let fallback = if table.is_empty() {
        0.0
    } else {
        np_sum(&table.iter().map(|t| t.1).collect::<Vec<_>>()) / table.len() as f64
    };
    if element_type.is_nan() {
        return fallback;
    }
    table.iter().find(|t| t.0 as f64 == element_type).map_or(fallback, |t| t.1)
}

/// A pool player's posterior shares and recent penalty attempts.
#[derive(Clone, Debug)]
struct Player {
    player_key: i64,
    team_key: i64,
    g: f64,
    a: f64,
    attempts: f64,
}

fn marcel_shares(
    rows: &[ShareRow],
    pool: &Table,
    fit: &SharesFit,
    season: i64,
) -> Result<Vec<Player>> {
    let params = &fit.params;
    let weights = &params.season_weights;
    let keys = pool.i64("player_key")?;
    let teams = pool.i64("team_key")?;
    let element = float_column(pool, "element_type")?;
    let prices = float_column(pool, "price")?;
    let club: HashMap<i64, i64> = keys.iter().copied().zip(teams.iter().copied()).collect();
    #[derive(Default)]
    struct Sums {
        xg: Kahan,
        dg: Kahan,
        xa: Kahan,
        da: Kahan,
        minutes: Kahan,
        attempts: Kahan,
        fin_goals: Kahan,
        fin_xg: Kahan,
    }
    let add = |k: &mut Kahan, v: f64| {
        if !v.is_nan() {
            k.add(v);
        }
    };
    let mut sums: HashMap<i64, Sums> = HashMap::new();
    for s in rows {
        let r = &s.row;
        let Some(&home) = club.get(&r.player_key) else { continue };
        let age = season - r.season;
        if age < 0 || age >= weights.len() as i64 {
            continue;
        }
        let moved = r.team_key != home;
        let base = weights[age as usize] * if moved { params.club_change_weight } else { 1.0 };
        let w_goals = base * if s.fb_goals { params.goals_weight } else { 1.0 };
        let w_assists = base * if s.fb_assists { params.goals_weight } else { 1.0 };
        let pen_goals = nan_to_num(r.pen_goals);
        let attempts = pen_goals + nan_to_num(r.penalties_missed);
        let own_xg = !s.fb_goals;
        let np_goals = nan_to_num(r.goals_scored) - pen_goals;
        let e = sums.entry(r.player_key).or_default();
        add(&mut e.xg, w_goals * s.x_goals);
        add(&mut e.dg, w_goals * s.d);
        add(&mut e.xa, w_assists * s.x_assists);
        add(&mut e.da, w_assists * s.d);
        add(&mut e.minutes, s.on_pitch);
        add(&mut e.attempts, base * attempts);
        add(&mut e.fin_goals, if own_xg { np_goals.max(0.0) } else { 0.0 });
        add(&mut e.fin_xg, if own_xg { s.x_goals } else { 0.0 });
    }
    let d0 = params.prior_minutes / 90.0 * fit.team_output;
    let mut out = Vec::with_capacity(keys.len());
    for i in 0..keys.len() {
        let s = sums.get(&keys[i]);
        let v = |f: fn(&Sums) -> &Kahan| s.map_or(0.0, |s| f(s).value());
        let (xg, dg, xa, da) = (v(|s| &s.xg), v(|s| &s.dg), v(|s| &s.xa), v(|s| &s.da));
        let minutes = v(|s| &s.minutes);
        let new = minutes <= 0.0;
        let use_price = params.price_prior_all || new;
        let mut prior_g = if use_price {
            price_prior(&fit.price_goal, element[i], prices[i])
        } else {
            position_prior(&fit.position_goal, element[i])
        };
        let mut prior_a = if use_price {
            price_prior(&fit.price_assist, element[i], prices[i])
        } else {
            position_prior(&fit.position_assist, element[i])
        };
        if prior_g.is_nan() {
            prior_g = position_prior(&fit.position_goal, element[i]);
        }
        if prior_a.is_nan() {
            prior_a = position_prior(&fit.position_assist, element[i]);
        }
        let finishing = match params.finishing_prior {
            None => 1.0,
            Some(k) => (v(|s| &s.fin_goals) + k) / (v(|s| &s.fin_xg) + k),
        };
        out.push(Player {
            player_key: keys[i],
            team_key: teams[i],
            g: (xg + prior_g * d0) / (dg + d0) * finishing,
            a: (xa + prior_a * d0) / (da + d0),
            attempts: v(|s| &s.attempts),
        });
    }
    Ok(out)
}

// --- the fit --------------------------------------------------------------------------------

/// `_ratio`: Σ numerator / Σ denominator (pandas sums), `default` when Σ denominator ≤ 0.
fn ratio(numerator: &[f64], denominator: &[f64], default: f64) -> f64 {
    let total = pd_sum(denominator);
    if total > 0.0 { pd_sum(numerator) / total } else { default }
}

fn taker_q(rows: &[Row], teams: &[TeamFixture], takers: &HashMap<i64, Vec<(Micros, i64)>>) -> f64 {
    let known: Vec<&TeamFixture> = teams.iter().filter(|t| t.team_attempts > 0.0).collect();
    if known.is_empty() || takers.is_empty() {
        return DEFAULT_TAKER_Q;
    }
    let mut kickoffs: HashMap<(i64, i64), Option<Micros>> = HashMap::new();
    for r in rows {
        kickoffs.entry((r.fixture_key, r.team_key)).or_insert(r.kickoff);
    }
    // own: (fixture, player) → rows, in row order.
    let mut own: HashMap<(i64, i64), Vec<&Row>> = HashMap::new();
    for r in rows {
        own.entry((r.fixture_key, r.player_key)).or_default().push(r);
    }
    let (mut attempts, mut team_attempts) = (Vec::new(), Vec::new());
    for t in known {
        let Some(&kickoff) = kickoffs.get(&(t.fixture_key, t.team_key)) else { continue };
        let Some(taker) = taker_at(takers, t.team_key, kickoff) else { continue };
        for r in own.get(&(t.fixture_key, taker)).into_iter().flatten() {
            if r.minutes >= 90.0 && !r.pen_attempts.is_nan() {
                attempts.push(r.pen_attempts);
                team_attempts.push(t.team_attempts);
            }
        }
    }
    shrunk_ratio(pd_sum(&attempts), pd_sum(&team_attempts), DEFAULT_TAKER_Q, TAKER_PRIOR_ATTEMPTS)
}

/// League-level parameters from every row visible in `view`.
pub fn fit_shares(view: &AsOfView, params: &SharesParams) -> Result<SharesFit> {
    let (season, _) = view.gameweek_for_deadline()?;
    let (rows, teams, (k_goals, k_assists, pen_pi)) = rows(view, None, None)?;
    let played: Vec<&Row> = rows.iter().filter(|r| r.minutes > 0.0).collect();
    let with_xg: Vec<&&Row> = played.iter().filter(|r| !r.npxg.is_nan()).collect();
    let with_xa: Vec<&&Row> = played.iter().filter(|r| !r.xa.is_nan()).collect();
    let ratio_goals = ratio(
        &with_xg.iter().map(|r| r.npxg).collect::<Vec<_>>(),
        &with_xg.iter().map(|r| r.goals_scored).collect::<Vec<_>>(),
        1.0,
    );
    let ratio_assists = ratio(
        &with_xa.iter().map(|r| r.xa).collect::<Vec<_>>(),
        &with_xa.iter().map(|r| r.assists).collect::<Vec<_>>(),
        1.0,
    );
    let team_xg: Vec<&TeamFixture> = teams.iter().filter(|t| !t.team_npxg.is_nan()).collect();
    let ratio_team = ratio(
        &team_xg.iter().map(|t| t.team_npxg).collect::<Vec<_>>(),
        &team_xg.iter().map(|t| t.goals_for).collect::<Vec<_>>(),
        ratio_goals,
    );
    let shares = share_rows(&rows, &teams, (ratio_goals, ratio_assists, ratio_team));
    let recent: Vec<&ShareRow> =
        shares.iter().filter(|s| s.row.season > season - FIT_SEASONS).collect();
    let team_output = ratio(
        &recent.iter().map(|s| s.d).collect::<Vec<_>>(),
        &recent.iter().map(|s| s.on_pitch).collect::<Vec<_>>(),
        1.3,
    );
    let mut by_position: BTreeMap<i64, Vec<&ShareRow>> = BTreeMap::new();
    for s in &recent {
        if !s.row.element_type.is_nan() {
            by_position.entry(s.row.element_type as i64).or_default().push(s);
        }
    }
    let (mut position_goal, mut position_assist) = (Vec::new(), Vec::new());
    let (mut price_goal, mut price_assist) = (Vec::new(), Vec::new());
    for (position, group) in by_position {
        let wg: Vec<f64> =
            group.iter().map(|s| if s.fb_goals { params.goals_weight } else { 1.0 }).collect();
        let wa: Vec<f64> =
            group.iter().map(|s| if s.fb_assists { params.goals_weight } else { 1.0 }).collect();
        let xg: Vec<f64> = (0..group.len()).map(|i| group[i].x_goals * wg[i]).collect();
        let dg: Vec<f64> = (0..group.len()).map(|i| group[i].d * wg[i]).collect();
        let xa: Vec<f64> = (0..group.len()).map(|i| group[i].x_assists * wa[i]).collect();
        let da: Vec<f64> = (0..group.len()).map(|i| group[i].d * wa[i]).collect();
        position_goal.push((position, ratio(&xg, &dg, 0.0)));
        position_assist.push((position, ratio(&xa, &da, 0.0)));
        // His first price of the season (the position's rows, any minutes).
        let mut first_price: HashMap<(i64, i64), f64> = HashMap::new();
        for r in rows.iter().filter(|r| r.element_type == position as f64 && !r.price.is_nan()) {
            first_price.entry((r.player_key, r.season)).or_insert(r.price);
        }
        // Per (player, season): Kahan sums, sorted by key (groupby + reset_index).
        let mut seasons: BTreeMap<(i64, i64), [Kahan; 4]> = BTreeMap::new();
        for (i, s) in group.iter().enumerate() {
            let e = seasons.entry((s.row.player_key, s.row.season)).or_default();
            for (k, v) in [xg[i], dg[i], xa[i], da[i]].into_iter().enumerate() {
                if !v.is_nan() {
                    e[k].add(v);
                }
            }
        }
        let table: Vec<(f64, [f64; 4])> = seasons
            .iter()
            .map(|(key, s)| {
                let price = first_price.get(key).copied().unwrap_or(f64::NAN);
                (price, [s[0].value(), s[1].value(), s[2].value(), s[3].value()])
            })
            .collect();
        let goal = price_model(&table.iter().map(|(p, v)| (*p, v[0], v[1])).collect::<Vec<_>>());
        let assist = price_model(&table.iter().map(|(p, v)| (*p, v[2], v[3])).collect::<Vec<_>>());
        price_goal.push((position, goal.0, goal.1, goal.2));
        price_assist.push((position, assist.0, assist.1, assist.2));
    }
    let window: Vec<&TeamFixture> =
        teams.iter().filter(|t| t.season > season - FIT_SEASONS).collect();
    let known: Vec<&&TeamFixture> = window.iter().filter(|t| !t.team_attempts.is_nan()).collect();
    let pen_rate = shrunk_ratio(
        pd_sum(&known.iter().map(|t| t.team_attempts).collect::<Vec<_>>()),
        known.len() as f64,
        DEFAULT_PEN_RATE,
        LEAGUE_PRIOR_MATCHES,
    );
    let scored: Vec<&&TeamFixture> =
        window.iter().filter(|t| !t.team_pen_goals.is_nan() && !t.team_attempts.is_nan()).collect();
    let conversion = shrunk_ratio(
        pd_sum(&scored.iter().map(|t| t.team_pen_goals).collect::<Vec<_>>()),
        pd_sum(&scored.iter().map(|t| t.team_attempts).collect::<Vec<_>>()),
        DEFAULT_CONVERSION,
        LEAGUE_PRIOR_ATTEMPTS,
    );
    let mut by_team: BTreeMap<i64, Vec<&TeamFixture>> = BTreeMap::new();
    for t in &known {
        by_team.entry(t.team_key).or_default().push(t);
    }
    let team_pen = by_team
        .into_iter()
        .map(|(team, group)| {
            let goals: Vec<&&TeamFixture> =
                group.iter().filter(|t| !t.team_pen_goals.is_nan()).collect();
            (
                team,
                shrunk_ratio(
                    pd_sum(&group.iter().map(|t| t.team_attempts).collect::<Vec<_>>()),
                    group.len() as f64,
                    pen_rate,
                    params.pen_team_prior,
                ),
                shrunk_ratio(
                    pd_sum(&goals.iter().map(|t| t.team_pen_goals).collect::<Vec<_>>()),
                    pd_sum(&goals.iter().map(|t| t.team_attempts).collect::<Vec<_>>()),
                    conversion,
                    params.conversion_prior,
                ),
            )
        })
        .collect();
    let goals_for: Vec<f64> = window.iter().map(|t| t.goals_for).collect();
    let assist_fraction = ratio(
        &window.iter().map(|t| t.assists).collect::<Vec<_>>(),
        &goals_for,
        DEFAULT_ASSIST_FRACTION,
    );
    let own_goal_fraction = ratio(
        &window.iter().map(|t| t.own_goals).collect::<Vec<_>>(),
        &goals_for,
        DEFAULT_OWN_GOAL_FRACTION,
    );
    let taker_q = taker_q(&rows, &teams, &main_takers(view)?);
    Ok(SharesFit {
        params: params.clone(),
        k_goals,
        k_assists,
        pen_pi,
        ratio_goals,
        ratio_assists,
        ratio_team,
        team_output,
        position_goal,
        position_assist,
        price_goal,
        price_assist,
        pen_rate,
        conversion,
        team_pen,
        assist_fraction,
        own_goal_fraction,
        taker_q,
        n_rows: shares.len(),
    })
}

// --- prediction -----------------------------------------------------------------------------

/// Per row (a club's player in one fixture; `groups` the club-fixture) the probability that
/// he takes a penalty the club is awarded.
fn taker_probabilities(groups: &[i64], rank: &[f64], q: &[f64], f: &[f64], g: &[f64]) -> Vec<f64> {
    let n = groups.len();
    let mut index: HashMap<i64, usize> = HashMap::new();
    let codes: Vec<usize> = groups
        .iter()
        .map(|g| {
            let next = index.len();
            *index.entry(*g).or_insert(next)
        })
        .collect();
    let n_groups = index.len();
    let listed: Vec<bool> = rank.iter().map(|r| !r.is_nan()).collect();
    let log_term: Vec<f64> = (0..n)
        .map(|i| {
            let fq = (f[i] * q[i]).clamp(0.0, 1.0);
            let v = if listed[i] { 1.0 - fq } else { 1.0 };
            np_maximum(v, 1e-300).ln()
        })
        .collect();
    // np.lexsort((rank or inf, codes)): by code, then rank, stable.
    let mut order: Vec<usize> = (0..n).collect();
    let rank_key = |i: usize| if listed[i] { rank[i] } else { f64::INFINITY };
    order.sort_by(|&a, &b| codes[a].cmp(&codes[b]).then(rank_key(a).total_cmp(&rank_key(b))));
    let mut cumulative = vec![0.0; n];
    let mut running = 0.0;
    for (k, &i) in order.iter().enumerate() {
        running += log_term[i];
        cumulative[k] = running;
    }
    let mut before = vec![0.0; n];
    let mut start = 0;
    for k in 0..n {
        if k > 0 && codes[order[k]] != codes[order[k - 1]] {
            start = k;
        }
        let block = if start == 0 { 0.0 } else { cumulative[start - 1] };
        before[order[k]] = cumulative[k] - log_term[order[k]] - block;
    }
    let mut group_log = vec![0.0; n_groups];
    let mut sums = vec![0.0; n_groups];
    let mut on_pitch = vec![0.0; n_groups];
    for i in 0..n {
        group_log[codes[i]] += log_term[i];
        sums[codes[i]] += g[i] * f[i];
        on_pitch[codes[i]] += f[i];
    }
    (0..n)
        .map(|i| {
            let c = codes[i];
            let leftover = group_log[c].exp();
            let direct = if listed[i] { f[i] * q[i] * before[i].exp() } else { 0.0 };
            let weight = g[i] * f[i];
            let spread = if sums[c] > 0.0 {
                weight / sums[c]
            } else if on_pitch[c] > 0.0 {
                f[i] / on_pitch[c]
            } else {
                0.0
            };
            direct + leftover * spread
        })
        .collect()
}

/// Per pool player: (rank, q) in his club's taker list (rank NaN = not a candidate).
fn listing(
    view: &AsOfView,
    season: i64,
    players: &[Player],
    fit: &SharesFit,
) -> Result<HashMap<i64, (f64, f64)>> {
    let snaps = view.latest(
        "player_snapshot",
        &["player_key"],
        Some(&["snapshot_at", "season", "team_key", "penalties_order"]),
    )?;
    let keys = snaps.i64("player_key")?;
    let seasons = snaps.i64("season")?;
    let at = opt_ts(&snaps, "snapshot_at")?;
    let teams = snaps.opt_i64("team_key")?;
    let order = snaps.opt_i64("penalties_order")?;
    let in_season: Vec<usize> = (0..snaps.height()).filter(|&i| seasons[i] == season).collect();
    let newest = in_season.iter().filter_map(|&i| at[i]).max();
    let listed: Vec<usize> =
        in_season.into_iter().filter(|&i| at[i] == newest && order[i].is_some()).collect();
    let by_player_team: HashMap<(i64, i64), f64> =
        listed.iter().filter_map(|&i| Some(((keys[i], teams[i]?), order[i]? as f64))).collect();
    let snapshot_teams: HashSet<i64> = listed.iter().filter_map(|&i| teams[i]).collect();
    let keyed: Vec<(i64, f64, i64, bool, f64)> = players
        .iter()
        .map(|p| {
            let from_snapshot = snapshot_teams.contains(&p.team_key);
            let order =
                by_player_team.get(&(p.player_key, p.team_key)).copied().unwrap_or(f64::NAN);
            let key = if from_snapshot {
                order
            } else if p.attempts > 0.0 {
                -p.attempts
            } else {
                f64::NAN
            };
            (p.team_key, key, p.player_key, from_snapshot, p.attempts)
        })
        .collect();
    let mut ranked: Vec<&(i64, f64, i64, bool, f64)> =
        keyed.iter().filter(|k| !k.1.is_nan()).collect();
    ranked.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)).then(a.2.cmp(&b.2)));
    let mut rank: HashMap<i64, f64> = HashMap::new();
    let mut count: HashMap<i64, f64> = HashMap::new();
    for k in ranked {
        let c = count.entry(k.0).or_insert(0.0);
        *c += 1.0;
        rank.insert(k.2, *c);
    }
    Ok(keyed
        .iter()
        .map(|&(_, _, player, from_snapshot, attempts)| {
            let r = rank.get(&player).copied().unwrap_or(f64::NAN);
            let q = if from_snapshot {
                fit.taker_q
            } else {
                attempts / (attempts + fit.params.taker_prior)
            };
            (player, (r, if r.is_nan() { 0.0 } else { q }))
        })
        .collect())
}

/// The shares frame (Python's `SHARES_COLUMNS`), column by column.
#[derive(Clone, Debug, Default)]
pub struct Shares {
    pub keys: Vec<crate::minutes::Key>,
    pub e_np_goals: Vec<f64>,
    pub e_pen_goals: Vec<f64>,
    pub e_goals: Vec<f64>,
    pub e_assists: Vec<f64>,
    pub goal_share: Vec<f64>,
    pub assist_share: Vec<f64>,
    pub p_taker: Vec<f64>,
}

impl Shares {
    pub fn to_table(&self) -> Table {
        let k = |f: fn(&crate::minutes::Key) -> i64| Column::I64(self.keys.iter().map(f).collect());
        Table::new()
            .with("player_key", k(|k| k.player_key))
            .with("fixture_key", k(|k| k.fixture_key))
            .with("team_key", k(|k| k.team_key))
            .with("season", k(|k| k.season))
            .with("gw", k(|k| k.gw))
            .with("gw_index", k(|k| k.gw_index))
            .with("horizon", k(|k| k.horizon))
            .with("e_np_goals", Column::F64(self.e_np_goals.clone()))
            .with("e_pen_goals", Column::F64(self.e_pen_goals.clone()))
            .with("e_goals", Column::F64(self.e_goals.clone()))
            .with("e_assists", Column::F64(self.e_assists.clone()))
            .with("goal_share", Column::F64(self.goal_share.clone()))
            .with("assist_share", Column::F64(self.assist_share.clone()))
            .with("p_taker", Column::F64(self.p_taker.clone()))
    }
}

/// Per row of `minutes`: the shares (sorted by player, horizon, fixture as `minutes` is).
/// `team`: the `team_lambdas` frame of the same view.
pub fn predict_shares(
    view: &AsOfView,
    fit: &SharesFit,
    minutes: &Minutes,
    team: &Table,
) -> Result<Shares> {
    let (season, _) = view.gameweek_for_deadline()?;
    let min_season = season - fit.params.season_weights.len() as i64 + 1;
    let (rows, teams, _) = rows(view, Some(fit), Some(min_season))?;
    let shares = share_rows(&rows, &teams, (fit.ratio_goals, fit.ratio_assists, fit.ratio_team));
    let pool = player_pool(view)?;
    let players = marcel_shares(&shares, &pool, fit, season)?;
    let listing = listing(view, season, &players, fit)?;
    let by_player: HashMap<i64, &Player> = players.iter().map(|p| (p.player_key, p)).collect();
    let (tf, tt, tl) = (team.i64("fixture_key")?, team.i64("team_key")?, team.f64("lambda_for")?);
    let lambdas: HashMap<(i64, i64), f64> =
        (0..team.height()).map(|i| ((tf[i], tt[i]), tl[i])).collect();
    let penalties: HashMap<i64, (f64, f64)> =
        fit.team_pen.iter().map(|&(t, r, c)| (t, (r, c))).collect();
    let n = minutes.len();
    let mut missing = 0;
    let mut lambda_for = vec![0.0; n];
    let mut team_pen_goals = vec![0.0; n];
    let (mut g, mut a, mut rank, mut q, mut f) =
        (vec![0.0; n], vec![0.0; n], vec![f64::NAN; n], vec![0.0; n], vec![0.0; n]);
    let mut groups = vec![0i64; n];
    for i in 0..n {
        let (fixture, club) = (minutes.fixture_key[i], minutes.team_key[i]);
        match lambdas.get(&(fixture, club)) {
            Some(&l) if !l.is_nan() => lambda_for[i] = l,
            _ => {
                missing += 1;
                lambda_for[i] = 0.0;
            }
        }
        let (rate, conversion) =
            penalties.get(&club).copied().unwrap_or((fit.pen_rate, fit.conversion));
        team_pen_goals[i] = rate * conversion;
        f[i] = np_maximum(minutes.e_minutes[i] / 90.0, 0.0);
        if let Some(p) = by_player.get(&minutes.player_key[i]) {
            g[i] = if p.g.is_nan() { 0.0 } else { p.g };
            a[i] = if p.a.is_nan() { 0.0 } else { p.a };
        }
        if let Some(&(r, qq)) = listing.get(&minutes.player_key[i]) {
            rank[i] = r;
            q[i] = if qq.is_nan() { 0.0 } else { qq };
        }
        groups[i] = fixture * 1_000_000 + club;
    }
    if missing > 0 {
        log::warn!("shares: {missing} row(s) have no team λ (not in the team frame); set to 0");
    }
    let mut index: HashMap<i64, usize> = HashMap::new();
    let codes: Vec<usize> = groups
        .iter()
        .map(|g| {
            let next = index.len();
            *index.entry(*g).or_insert(next)
        })
        .collect();
    let mut sum_g = vec![0.0; index.len()];
    let mut sum_a = vec![0.0; index.len()];
    for i in 0..n {
        sum_g[codes[i]] += g[i] * f[i];
        sum_a[codes[i]] += a[i] * f[i];
    }
    let p_take = taker_probabilities(&groups, &rank, &q, &f, &g);
    let mut out = Shares::default();
    for i in 0..n {
        let lambda_np =
            (lambda_for[i] * (1.0 - fit.own_goal_fraction) - team_pen_goals[i]).max(0.0);
        let goal_share = if sum_g[codes[i]] > 0.0 { g[i] / sum_g[codes[i]] } else { 0.0 };
        let assist_share = if sum_a[codes[i]] > 0.0 { a[i] / sum_a[codes[i]] } else { 0.0 };
        let e_np = lambda_np * goal_share * f[i];
        let e_pen = p_take[i] * team_pen_goals[i];
        out.keys.push(crate::minutes::Key {
            player_key: minutes.player_key[i],
            fixture_key: minutes.fixture_key[i],
            team_key: minutes.team_key[i],
            season: minutes.season[i],
            gw: minutes.gw[i],
            gw_index: minutes.gw_index[i],
            horizon: minutes.horizon[i],
        });
        out.e_np_goals.push(e_np);
        out.e_pen_goals.push(e_pen);
        out.e_goals.push(e_np + e_pen);
        out.e_assists.push(lambda_for[i] * fit.assist_fraction * assist_share * f[i]);
        out.goal_share.push(goal_share);
        out.assist_share.push(assist_share);
        out.p_taker.push(if f[i] > 0.0 { p_take[i] / f[i] } else { 0.0 });
    }
    Ok(out)
}

impl SharesFit {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "k_goals": self.k_goals,
            "k_assists": self.k_assists,
            "pen_pi": [self.pen_pi.0, self.pen_pi.1],
            "ratio_goals": self.ratio_goals,
            "ratio_assists": self.ratio_assists,
            "ratio_team": self.ratio_team,
            "team_output": self.team_output,
            "position_goal": self.position_goal.iter().map(|(p, v)| serde_json::json!([p, v])).collect::<Vec<_>>(),
            "position_assist": self.position_assist.iter().map(|(p, v)| serde_json::json!([p, v])).collect::<Vec<_>>(),
            "price_goal": self.price_goal.iter().map(|(p, a, b, c)| serde_json::json!([p, a, b, c])).collect::<Vec<_>>(),
            "price_assist": self.price_assist.iter().map(|(p, a, b, c)| serde_json::json!([p, a, b, c])).collect::<Vec<_>>(),
            "pen_rate": self.pen_rate,
            "conversion": self.conversion,
            "team_pen": self.team_pen.iter().map(|(t, r, c)| serde_json::json!([t, r, c])).collect::<Vec<_>>(),
            "assist_fraction": self.assist_fraction,
            "own_goal_fraction": self.own_goal_fraction,
            "taker_q": self.taker_q,
            "n_rows": self.n_rows,
        })
    }
}
