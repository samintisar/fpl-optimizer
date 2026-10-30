//! The remaining xP components (Python `fplopt.models.components`): bonus (per position
//! weighted ridge least squares on BPS event indicators), goalkeeper saves (Poisson curve in
//! the opponent's market λ by L-BFGS-B, keeper effects shrunk toward 1) and penalty saves,
//! card and own-goal rates per 90 shrunk toward the position's, the penalty-miss ratio from
//! the club's conversion, and defensive contributions (in-season rates; negative-binomial
//! tail in the assembly).

use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::Result;
use fplopt_build::scoring::{backtest_rules, default_config_dir};
use fplopt_features::baseline::player_pool;
use fplopt_features::util::Kahan;
use fplopt_store::{AsOfView, Column, Table};

use crate::gbm::float_column;
use crate::minutes::{Key, Minutes};
use crate::optim::lbfgsb;
use crate::shares::SharesFit;
use crate::team::{Devig, RHO, market_lambdas, market_probabilities, visible_odds};
use crate::util::{np_maximum, np_sum};

const GOALKEEPER: i64 = 1;
const POSITIONS: [i64; 4] = [1, 2, 3, 4];
pub const BONUS_FEATURES: [&str; 9] =
    ["play", "sixty", "goal1", "goal2", "goal3", "assist1", "assist2", "cs", "saves3"];
const BONUS_RIDGE: f64 = 1e-6;
const MIN_BONUS_ROWS: usize = 200;
const MIN_SAVES_ROWS: usize = 200;
const DEFAULT_SAVES_PER_90: f64 = 2.8;
const DEFAULT_PEN_SAVE_RATE: f64 = 0.02;
const PEN_SAVE_PRIOR: f64 = 200.0;
const SAVES_B_START: f64 = 0.5;
const LAMBDA_FLOOR: f64 = 0.05;
const DEFCON_POSITIONS: [i64; 3] = [2, 3, 4];

#[derive(Clone, Debug, PartialEq)]
pub struct ComponentsParams {
    pub seasons: i64,
    pub season_decay: f64,
    pub keeper_prior: f64,
    pub yellow_prior: f64,
    pub red_prior: f64,
    pub own_goal_prior: f64,
    pub defcon_k: Vec<(i64, f64)>,
    pub defcon_r: Vec<(i64, f64)>,
    pub defcon_prior_mean: Vec<(i64, f64)>,
}

impl Default for ComponentsParams {
    fn default() -> Self {
        ComponentsParams {
            seasons: 4,
            season_decay: 0.7,
            keeper_prior: 30.0,
            yellow_prior: 10.0,
            red_prior: 60.0,
            own_goal_prior: 60.0,
            defcon_k: vec![(2, 3.49), (3, 1.94), (4, 1.94)],
            defcon_r: vec![(2, 13.2), (3, 15.2), (4, 15.2)],
            defcon_prior_mean: vec![(2, 6.57), (3, 7.73), (4, 4.07)],
        }
    }
}

#[derive(Clone, Debug)]
pub struct SavesFit {
    pub a: f64,
    pub b: f64,
    pub keepers: Vec<(i64, f64)>,
    pub pen_save_rate: f64,
    pub n_rows: usize,
}

#[derive(Clone, Debug)]
pub struct ComponentsFit {
    pub params: ComponentsParams,
    pub bonus: Vec<(i64, Vec<f64>)>,
    pub bonus_rows: usize,
    pub saves: SavesFit,
    pub positions: Vec<(i64, f64, f64, f64)>,
    pub players: Vec<(i64, f64, f64, f64)>,
}

/// One played training row.
#[derive(Clone, Debug)]
struct Row {
    player_key: i64,
    season: i64,
    fixture_key: i64,
    was_home: bool,
    minutes: f64,
    goals: f64,
    assists: f64,
    clean_sheets: f64,
    saves: f64,
    bonus: f64,
    yellow: f64,
    red: f64,
    own_goals: f64,
    penalties_saved: f64,
    element_type: i64,
    weight: f64,
}

fn nan_to_num(v: f64) -> f64 {
    if v.is_nan() { 0.0 } else { v }
}

/// Played `player_match` rows of the last `params.seasons` seasons with the player_season
/// position and the season-decay weight, sorted by (player, season, fixture).
fn training_rows(view: &AsOfView, params: &ComponentsParams) -> Result<Vec<Row>> {
    let (season, _) = view.gameweek_for_deadline()?;
    let columns = [
        "player_key",
        "season",
        "fixture_key",
        "team_key",
        "opponent_team_key",
        "was_home",
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "saves",
        "bonus",
        "yellow_cards",
        "red_cards",
        "own_goals",
        "penalties_saved",
    ]; // fmt: skip
    let t = view.table("player_match", Some(&columns))?;
    let (player, seasons, fixture) =
        (t.i64("player_key")?, t.i64("season")?, t.i64("fixture_key")?);
    let was_home = t.opt_bool("was_home")?;
    let f = |c: &str| -> Result<Vec<f64>> {
        Ok(float_column(&t, c)?.into_iter().map(nan_to_num).collect())
    };
    let (minutes, goals, assists, cs, saves) =
        (f("minutes")?, f("goals_scored")?, f("assists")?, f("clean_sheets")?, f("saves")?);
    let (bonus, yellow, red, own, pen_saved) =
        (f("bonus")?, f("yellow_cards")?, f("red_cards")?, f("own_goals")?, f("penalties_saved")?);
    let ps = view.table("player_season", Some(&["player_key", "season", "element_type"]))?;
    let mut positions: HashMap<(i64, i64), Option<i64>> = HashMap::new();
    let (pp, psn, pe) = (ps.i64("player_key")?, ps.i64("season")?, ps.opt_i64("element_type")?);
    for i in 0..ps.height() {
        positions.insert((pp[i], psn[i]), pe[i]);
    }
    let mut rows = Vec::new();
    for i in 0..t.height() {
        if seasons[i] <= season - params.seasons || minutes[i] <= 0.0 {
            continue;
        }
        let Some(&element) = positions.get(&(player[i], seasons[i])) else { continue };
        let age = ((season - seasons[i]) as f64).max(0.0);
        rows.push(Row {
            player_key: player[i],
            season: seasons[i],
            fixture_key: fixture[i],
            was_home: was_home[i].unwrap_or(false),
            minutes: minutes[i],
            goals: goals[i],
            assists: assists[i],
            clean_sheets: cs[i],
            saves: saves[i],
            bonus: bonus[i],
            yellow: yellow[i],
            red: red[i],
            own_goals: own[i],
            penalties_saved: pen_saved[i],
            element_type: element.unwrap_or(0),
            weight: params.season_decay.powf(age),
        });
    }
    rows.sort_by_key(|r| (r.player_key, r.season, r.fixture_key));
    Ok(rows)
}

/// The realized bonus indicators of a played row.
fn bonus_features(r: &Row) -> [f64; 9] {
    let b = |x: bool| if x { 1.0 } else { 0.0 };
    [
        b(r.minutes > 0.0),
        b(r.minutes >= 60.0),
        b(r.goals >= 1.0),
        b(r.goals >= 2.0),
        b(r.goals >= 3.0),
        b(r.assists >= 1.0),
        b(r.assists >= 2.0),
        b(r.clean_sheets >= 1.0),
        if r.element_type == GOALKEEPER { (r.saves / 3.0).floor() } else { 0.0 },
    ]
}

/// `np.linalg.solve`: LU with partial pivoting (LAPACK getrf / getrs).
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for k in 0..n {
        let mut pivot = k;
        for i in k + 1..n {
            if a[i][k].abs() > a[pivot][k].abs() {
                pivot = i;
            }
        }
        a.swap(k, pivot);
        b.swap(k, pivot);
        let inverse = 1.0 / a[k][k];
        for i in k + 1..n {
            a[i][k] *= inverse;
            let l = a[i][k];
            for j in k + 1..n {
                a[i][j] -= l * a[k][j];
            }
        }
    }
    for i in 0..n {
        for j in 0..i {
            b[i] -= a[i][j] * b[j];
        }
    }
    for i in (0..n).rev() {
        for j in i + 1..n {
            b[i] -= a[i][j] * b[j];
        }
        b[i] /= a[i][i];
    }
    b
}

fn fit_bonus(rows: &[Row]) -> Vec<(i64, Vec<f64>)> {
    POSITIONS
        .iter()
        .map(|&position| {
            let part: Vec<&Row> = rows.iter().filter(|r| r.element_type == position).collect();
            if part.len() < MIN_BONUS_ROWS {
                return (position, vec![0.0; BONUS_FEATURES.len()]);
            }
            let k = BONUS_FEATURES.len();
            let x: Vec<[f64; 9]> = part.iter().map(|r| bonus_features(r)).collect();
            let w: Vec<f64> = part.iter().map(|r| r.weight).collect();
            let mut gram = vec![vec![0.0; k]; k];
            let mut rhs = vec![0.0; k];
            for (i, r) in part.iter().enumerate() {
                for a in 0..k {
                    for c in 0..k {
                        gram[a][c] += x[i][a] * (x[i][c] * w[i]);
                    }
                    rhs[a] += x[i][a] * (w[i] * r.bonus);
                }
            }
            let total = np_sum(&w);
            for (a, row) in gram.iter_mut().enumerate() {
                row[a] += BONUS_RIDGE * total;
            }
            (position, solve(gram, rhs))
        })
        .collect()
}

/// Per goalkeeper row the opponent's market λ from the fixture's own visible pre-match odds
/// (NaN without odds or a converged solve).
fn opponent_lambdas(view: &AsOfView, rows: &[&Row]) -> Result<Vec<f64>> {
    let keys: HashSet<i64> = rows.iter().map(|r| r.fixture_key).collect();
    let odds = visible_odds(view, &keys)?;
    if odds.is_empty() {
        return Ok(vec![f64::NAN; rows.len()]);
    }
    let solved: HashMap<i64, (f64, f64)> =
        market_lambdas(&market_probabilities(&odds, Devig::Power), RHO)?
            .into_iter()
            .filter(|s| s.3)
            .map(|s| (s.0, (s.1, s.2)))
            .collect();
    Ok(rows
        .iter()
        .map(|r| match solved.get(&r.fixture_key) {
            Some(&(home, away)) => {
                if r.was_home {
                    away
                } else {
                    home
                }
            }
            None => f64::NAN,
        })
        .collect())
}

fn fit_saves(keepers: &[&Row], lambda_against: &[f64], keeper_prior: f64) -> SavesFit {
    let products = |f: &dyn Fn(&Row) -> f64| -> f64 {
        np_sum(&keepers.iter().map(|r| r.weight * f(r)).collect::<Vec<_>>())
    };
    let pen_saves = products(&|r| r.penalties_saved);
    let exposure = products(&|r| r.minutes / 90.0);
    let pen_save_rate =
        (pen_saves + PEN_SAVE_PRIOR * DEFAULT_PEN_SAVE_RATE) / (exposure + PEN_SAVE_PRIOR);
    let saves_all = products(&|r| r.saves);
    let flat = if saves_all > 0.0 && exposure > 0.0 {
        (saves_all / exposure).ln()
    } else {
        DEFAULT_SAVES_PER_90.ln()
    };
    let known: Vec<usize> = (0..keepers.len()).filter(|&i| lambda_against[i].is_finite()).collect();
    if known.len() < MIN_SAVES_ROWS {
        return SavesFit {
            a: flat,
            b: 0.0,
            keepers: Vec::new(),
            pen_save_rate,
            n_rows: known.len(),
        };
    }
    let w: Vec<f64> = known.iter().map(|&i| keepers[i].weight).collect();
    let frac: Vec<f64> = known.iter().map(|&i| keepers[i].minutes / 90.0).collect();
    let saves: Vec<f64> = known.iter().map(|&i| keepers[i].saves).collect();
    let log_lambda: Vec<f64> =
        known.iter().map(|&i| np_maximum(lambda_against[i], LAMBDA_FLOOR).ln()).collect();
    let scale = np_sum(&w);
    let n = w.len();
    let loss = |theta: &[f64]| -> (f64, Vec<f64>) {
        let eta: Vec<f64> = (0..n).map(|i| theta[0] + theta[1] * log_lambda[i]).collect();
        let mu: Vec<f64> = (0..n).map(|i| frac[i] * eta[i].exp()).collect();
        let value =
            np_sum(&(0..n).map(|i| w[i] * (mu[i] - saves[i] * eta[i])).collect::<Vec<_>>()) / scale;
        let residual: Vec<f64> = (0..n).map(|i| w[i] * (mu[i] - saves[i]) / scale).collect();
        let g0 = np_sum(&residual);
        let g1 = np_sum(&(0..n).map(|i| residual[i] * log_lambda[i]).collect::<Vec<_>>());
        (value, vec![g0, g1])
    };
    let mean_log_lambda = np_sum(&(0..n).map(|i| w[i] * log_lambda[i]).collect::<Vec<_>>()) / scale;
    let start = vec![flat - SAVES_B_START * mean_log_lambda, SAVES_B_START];
    let x = match lbfgsb(loss, start, 200) {
        Ok(result) => result.x,
        Err(_) => vec![flat, 0.0],
    };
    let (a, b) = (x[0], x[1]);
    // Keeper effects: per player Kahan sums (pandas' groupby), sorted.
    let mut totals: BTreeMap<i64, (Kahan, Kahan)> = BTreeMap::new();
    for (k, &i) in known.iter().enumerate() {
        let mu = frac[k] * (a + b * log_lambda[k]).exp();
        let e = totals.entry(keepers[i].player_key).or_default();
        e.0.add(w[k] * saves[k]);
        e.1.add(w[k] * mu);
    }
    let effects = totals
        .into_iter()
        .map(|(key, (s, m))| (key, (s.value() + keeper_prior) / (m.value() + keeper_prior)))
        .collect();
    SavesFit { a, b, keepers: effects, pen_save_rate, n_rows: known.len() }
}

/// (position rates, shrunk player rates) per 90 of yellows, reds and own goals.
fn fit_rates(
    rows: &[Row],
    params: &ComponentsParams,
) -> (Vec<(i64, f64, f64, f64)>, Vec<(i64, f64, f64, f64)>) {
    let priors = [params.yellow_prior, params.red_prior, params.own_goal_prior];
    let values = |r: &Row| -> [f64; 4] {
        [r.weight * r.minutes / 90.0, r.weight * r.yellow, r.weight * r.red, r.weight * r.own_goals]
    };
    let mut by_position: BTreeMap<i64, [Kahan; 4]> = BTreeMap::new();
    let mut by_player: BTreeMap<i64, [Kahan; 4]> = BTreeMap::new();
    let mut latest: HashMap<i64, i64> = HashMap::new();
    for r in rows {
        let v = values(r);
        let p = by_position.entry(r.element_type).or_default();
        let q = by_player.entry(r.player_key).or_default();
        for k in 0..4 {
            p[k].add(v[k]);
            q[k].add(v[k]);
        }
        latest.insert(r.player_key, r.element_type);
    }
    let positions: Vec<(i64, f64, f64, f64)> = POSITIONS
        .iter()
        .map(|&position| match by_position.get(&position) {
            Some(s) if s[0].value() > 0.0 => {
                let e = s[0].value();
                (position, s[1].value() / e, s[2].value() / e, s[3].value() / e)
            }
            _ => (position, 0.0, 0.0, 0.0),
        })
        .collect();
    let rate_of: HashMap<i64, [f64; 3]> =
        positions.iter().map(|&(p, y, r, o)| (p, [y, r, o])).collect();
    let players = by_player
        .into_iter()
        .map(|(key, s)| {
            let base = rate_of.get(&latest[&key]).copied().unwrap_or([0.0; 3]);
            let rate =
                |k: usize| (s[k + 1].value() + priors[k] * base[k]) / (s[0].value() + priors[k]);
            (key, rate(0), rate(1), rate(2))
        })
        .collect();
    (positions, players)
}

/// P(X ≥ k) for X ~ NB(mean, dispersion r): 1 − Σ_{n<k} P(n).
pub fn negbin_tail(mean: f64, r: f64, k: i64) -> f64 {
    let mean = np_maximum(mean, 0.0);
    let q = mean / (r + mean);
    let mut pmf = (r * (-q).ln_1p()).exp();
    let mut cdf = 0.0;
    for n in 0..k.max(0) {
        cdf += pmf;
        pmf = pmf * (n as f64 + r) / (n as f64 + 1.0) * q;
    }
    (1.0 - cdf).clamp(0.0, 1.0)
}

/// (position means, per (player, position) shrunk rates) of defensive contributions per 90
/// from the played rows of the deadline's season.
fn defcon_rates(
    view: &AsOfView,
    params: &ComponentsParams,
) -> Result<(HashMap<i64, f64>, HashMap<(i64, i64), f64>)> {
    let (season, _) = view.gameweek_for_deadline()?;
    let columns = [
        "player_key",
        "season",
        "minutes",
        "defensive_contribution",
        "clearances_blocks_interceptions",
        "tackles",
        "recoveries",
    ]; // fmt: skip
    let t = view.table("player_match", Some(&columns))?;
    let (player, seasons) = (t.i64("player_key")?, t.i64("season")?);
    let minutes = float_column(&t, "minutes")?;
    let fpl = float_column(&t, "defensive_contribution")?;
    let cbi = float_column(&t, "clearances_blocks_interceptions")?;
    let tackles = float_column(&t, "tackles")?;
    let recoveries = float_column(&t, "recoveries")?;
    let ps = view.table("player_season", Some(&["player_key", "season", "element_type"]))?;
    let (pp, psn, pe) = (ps.i64("player_key")?, ps.i64("season")?, ps.opt_i64("element_type")?);
    let mut position: HashMap<i64, Option<i64>> = HashMap::new();
    for i in 0..ps.height() {
        if psn[i] == season {
            position.insert(pp[i], pe[i]);
        }
    }
    let mut totals: BTreeMap<i64, (Kahan, Kahan)> = BTreeMap::new();
    let mut sums: BTreeMap<(i64, i64), (Kahan, Kahan)> = BTreeMap::new();
    for i in 0..t.height() {
        if seasons[i] != season || nan_to_num(minutes[i]) <= 0.0 {
            continue;
        }
        let Some(&element) = position.get(&player[i]) else { continue };
        let element = element.unwrap_or(0);
        let stats = match element {
            2 => cbi[i] + tackles[i],
            3 | 4 => cbi[i] + tackles[i] + recoveries[i],
            _ => f64::NAN,
        };
        let count = if fpl[i].is_nan() { stats } else { fpl[i] };
        if !count.is_finite() || !DEFCON_POSITIONS.contains(&element) {
            continue;
        }
        let exposure = minutes[i] / 90.0;
        let e = totals.entry(element).or_default();
        e.0.add(count);
        e.1.add(exposure);
        let s = sums.entry((player[i], element)).or_default();
        s.0.add(count);
        s.1.add(exposure);
    }
    let prior: HashMap<i64, f64> = params.defcon_prior_mean.iter().copied().collect();
    let means: HashMap<i64, f64> = DEFCON_POSITIONS
        .iter()
        .map(|&p| match totals.get(&p) {
            Some((c, e)) if e.value() > 0.0 => (p, c.value() / e.value()),
            _ => (p, prior[&p]),
        })
        .collect();
    let shrink: HashMap<i64, f64> = params.defcon_k.iter().copied().collect();
    let players = sums
        .into_iter()
        .map(|((key, p), (c, e))| {
            let k = shrink[&p];
            ((key, p), (c.value() + k * means[&p]) / (e.value() + k))
        })
        .collect();
    Ok((means, players))
}

/// Bonus, saves and card / own-goal fits on the rows visible in `view`.
pub fn fit_components(view: &AsOfView, params: &ComponentsParams) -> Result<ComponentsFit> {
    let rows = training_rows(view, params)?;
    let keepers: Vec<&Row> = rows.iter().filter(|r| r.element_type == GOALKEEPER).collect();
    let lambdas = opponent_lambdas(view, &keepers)?;
    let (positions, players) = fit_rates(&rows, params);
    Ok(ComponentsFit {
        params: params.clone(),
        bonus: fit_bonus(&rows),
        bonus_rows: rows.len(),
        saves: fit_saves(&keepers, &lambdas, params.keeper_prior),
        positions,
        players,
    })
}

/// Per row of `minutes`: the component rates (`COMPONENT_COLUMNS`).
#[derive(Clone, Debug, Default)]
pub struct Components {
    pub keys: Vec<Key>,
    pub element_type: Vec<i64>,
    pub opponent_team_key: Vec<i64>,
    pub saves_rate: Vec<f64>,
    pub pen_save_rate: Vec<f64>,
    pub yellow_rate: Vec<f64>,
    pub red_rate: Vec<f64>,
    pub own_goal_rate: Vec<f64>,
    pub pen_miss_ratio: Vec<f64>,
    pub defcon_rate: Vec<f64>,
    pub defcon_r: Vec<f64>,
    pub bonus: Vec<[f64; 9]>,
}

impl Components {
    pub fn to_table(&self) -> Table {
        let k = |f: fn(&Key) -> i64| Column::I64(self.keys.iter().map(f).collect());
        let mut t = Table::new()
            .with("player_key", k(|k| k.player_key))
            .with("fixture_key", k(|k| k.fixture_key))
            .with("team_key", k(|k| k.team_key))
            .with("season", k(|k| k.season))
            .with("gw", k(|k| k.gw))
            .with("gw_index", k(|k| k.gw_index))
            .with("horizon", k(|k| k.horizon))
            .with("element_type", Column::I64(self.element_type.clone()))
            .with("opponent_team_key", Column::I64(self.opponent_team_key.clone()))
            .with("saves_rate", Column::F64(self.saves_rate.clone()))
            .with("pen_save_rate", Column::F64(self.pen_save_rate.clone()))
            .with("yellow_rate", Column::F64(self.yellow_rate.clone()))
            .with("red_rate", Column::F64(self.red_rate.clone()))
            .with("own_goal_rate", Column::F64(self.own_goal_rate.clone()))
            .with("pen_miss_ratio", Column::F64(self.pen_miss_ratio.clone()))
            .with("defcon_rate", Column::F64(self.defcon_rate.clone()))
            .with("defcon_r", Column::F64(self.defcon_r.clone()));
        for (j, name) in BONUS_FEATURES.iter().enumerate() {
            t.push(&format!("b_{name}"), Column::F64(self.bonus.iter().map(|b| b[j]).collect()));
        }
        t
    }
}

pub fn predict_components(
    view: &AsOfView,
    fit: &ComponentsFit,
    minutes: &Minutes,
    team: &Table,
    shares_fit: &SharesFit,
) -> Result<Components> {
    let pool = player_pool(view)?;
    let element: HashMap<i64, i64> = pool
        .i64("player_key")?
        .iter()
        .copied()
        .zip(pool.i64("element_type")?.iter().copied())
        .collect();
    let (tf, tt) = (team.i64("fixture_key")?, team.i64("team_key")?);
    let (to, tl) = (team.i64("opponent_team_key")?, team.f64("lambda_against")?);
    let sides: HashMap<(i64, i64), (i64, f64)> =
        (0..team.height()).map(|i| ((tf[i], tt[i]), (to[i], tl[i]))).collect();
    let thetas: HashMap<i64, f64> = fit.saves.keepers.iter().copied().collect();
    let club_pen: HashMap<i64, (f64, f64)> =
        shares_fit.team_pen.iter().map(|&(t, r, c)| (t, (r, c))).collect();
    let league_rate = if shares_fit.pen_rate > 0.0 { shares_fit.pen_rate } else { 1.0 };
    let position_rates: HashMap<i64, [f64; 3]> =
        fit.positions.iter().map(|&(p, y, r, o)| (p, [y, r, o])).collect();
    let player_rates: HashMap<i64, [f64; 3]> =
        fit.players.iter().map(|&(p, y, r, o)| (p, [y, r, o])).collect();
    let (season, _) = view.gameweek_for_deadline()?;
    let defcon = if backtest_rules(season, &default_config_dir())?.defcon_enabled {
        Some(defcon_rates(view, &fit.params)?)
    } else {
        None
    };
    let dispersion: HashMap<i64, f64> = fit.params.defcon_r.iter().copied().collect();
    let coefficients: HashMap<i64, &Vec<f64>> = fit.bonus.iter().map(|(p, c)| (*p, c)).collect();
    let mut out = Components::default();
    for i in 0..minutes.len() {
        let key = minutes.player_key[i];
        let e = element.get(&key).copied().unwrap_or(0);
        let keeper = e == GOALKEEPER;
        let (opponent, lambda) = sides
            .get(&(minutes.fixture_key[i], minutes.team_key[i]))
            .map_or((-1, f64::NAN), |&(o, l)| (o, l));
        let theta = thetas.get(&key).copied().unwrap_or(1.0);
        let lam = np_maximum(lambda, LAMBDA_FLOOR);
        let saves = if keeper { theta * (fit.saves.a + fit.saves.b * lam.ln()).exp() } else { 0.0 };
        let saves = if saves.is_nan() { 0.0 } else { saves };
        let opp_rate = club_pen.get(&opponent).map_or(shares_fit.pen_rate, |p| p.0);
        let pen_save = if keeper { fit.saves.pen_save_rate * opp_rate / league_rate } else { 0.0 };
        let conversion = club_pen
            .get(&minutes.team_key[i])
            .map_or(shares_fit.conversion, |p| p.1)
            .clamp(1e-3, 1.0);
        let rates =
            player_rates.get(&key).or_else(|| position_rates.get(&e)).copied().unwrap_or([0.0; 3]);
        let defcon_rate = match &defcon {
            Some((means, players)) if DEFCON_POSITIONS.contains(&e) => players
                .get(&(key, e))
                .copied()
                .unwrap_or_else(|| means.get(&e).copied().unwrap_or(0.0)),
            _ => 0.0,
        };
        let mut bonus = [0.0; 9];
        if let Some(c) = coefficients.get(&e) {
            bonus.copy_from_slice(c);
        }
        out.keys.push(Key {
            player_key: key,
            fixture_key: minutes.fixture_key[i],
            team_key: minutes.team_key[i],
            season: minutes.season[i],
            gw: minutes.gw[i],
            gw_index: minutes.gw_index[i],
            horizon: minutes.horizon[i],
        });
        out.element_type.push(e);
        out.opponent_team_key.push(opponent);
        out.saves_rate.push(saves);
        out.pen_save_rate.push(pen_save);
        out.yellow_rate.push(rates[0]);
        out.red_rate.push(rates[1]);
        out.own_goal_rate.push(rates[2]);
        out.pen_miss_ratio.push((1.0 - conversion) / conversion);
        out.defcon_rate.push(defcon_rate);
        out.defcon_r.push(dispersion.get(&e).copied().unwrap_or(0.0));
        out.bonus.push(bonus);
    }
    Ok(out)
}

impl ComponentsFit {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "bonus": {
                "coefficients": self.bonus.iter().map(|(p, c)| serde_json::json!([p, c])).collect::<Vec<_>>(),
                "n_rows": self.bonus_rows,
            },
            "saves": {
                "a": self.saves.a,
                "b": self.saves.b,
                "keepers": self.saves.keepers.iter().map(|(k, v)| serde_json::json!([k, v])).collect::<Vec<_>>(),
                "pen_save_rate": self.saves.pen_save_rate,
                "n_rows": self.saves.n_rows,
            },
            "rates": {
                "positions": self.positions.iter().map(|(p, y, r, o)| serde_json::json!([p, y, r, o])).collect::<Vec<_>>(),
                "players": self.players.iter().map(|(p, y, r, o)| serde_json::json!([p, y, r, o])).collect::<Vec<_>>(),
            },
        })
    }
}
