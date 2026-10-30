//! xP assembly and the `v1` model (Python `fplopt.models.assemble`).
//!
//! **Fit** (`fit_v1`, at the refit cutoff): team, minutes, availability, shares, components.
//! **Per fixture**: the components joined per player-fixture, calibrated (isotonic P(start),
//! P(CS), P(goal ≥ 1)), turned into expected points under the season's backtest rules
//! (minutes split into "60+" and "1–59", Poisson goals conceded / saves, the bonus
//! regression on expected indicators, defcon's negative-binomial tail), then the per-position
//! xP map. **Per GW** (`gw_frame`): every pool player × GW of the horizon, fixtures summed
//! (a double sums two, a blank is 0), single-fixture probabilities, P(plays in ≥ 1 fixture).

use std::collections::{BTreeMap, HashMap};

use anyhow::{Result, bail};
use fplopt_build::scoring::{Rules, backtest_rules, default_config_dir};
use fplopt_features::baseline::{HORIZON, player_pool, upcoming_fixtures};
use fplopt_features::history::training_frame;
use fplopt_features::util::{Kahan, floats};
use fplopt_store::{AsOfView, Column, Table};

use crate::availability::{AvailabilityFit, adjust_minutes, fit_availability};
use crate::calibration::{Calibration, apply_isotonic, calibration_for};
use crate::components::{
    ComponentsFit, ComponentsParams, fit_components, negbin_tail, predict_components,
};
use crate::fitted::{REFIT_EVERY, refit_deadline};
use crate::minutes::{MinutesFit, MinutesParams, conditionals, fit_minutes, predict_minutes};
use crate::shares::{SharesFit, SharesParams, fit_shares, predict_shares};
use crate::team::{TeamFit, TeamParams, fit_team, team_lambdas};
use crate::util::{np_clip, np_maximum, np_minimum, np_sum};

const GOALKEEPER: i64 = 1;
const POISSON_CAP: usize = 40;
const SHORT_MAX_FRACTION: f64 = 59.0 / 90.0;
const SAVES_FOR_BONUS: i64 = 3;

pub const POINT_COLUMNS: [&str; 13] = [
    "pts_appearance",
    "pts_goals",
    "pts_assists",
    "pts_clean_sheet",
    "pts_conceded",
    "pts_saves",
    "pts_pen_saves",
    "pts_bonus",
    "pts_yellow",
    "pts_red",
    "pts_own_goals",
    "pts_pen_miss",
    "pts_defcon",
]; // fmt: skip
pub const GW_PROBABILITIES: [&str; 9] = [
    "p_start",
    "p_play",
    "p_60",
    "p_min_0",
    "p_min_1_59",
    "p_min_60",
    "p_cs",
    "p_goal",
    "p_defcon",
];
pub const GW_SUMS: [&str; 4] = ["e_minutes", "e_goals", "e_assists", "e_bonus"];

/// v1's settings: the module defaults, with the minutes model's ban residual.
#[derive(Clone, Debug, PartialEq)]
pub struct V1Params {
    pub team: TeamParams,
    pub minutes: MinutesParams,
    pub shares: SharesParams,
    pub components: ComponentsParams,
    pub back_split: bool,
    pub calibrate: bool,
}

impl Default for V1Params {
    fn default() -> Self {
        V1Params {
            team: TeamParams::default(),
            minutes: MinutesParams { ban_residual: true, ..MinutesParams::default() },
            shares: SharesParams::default(),
            components: ComponentsParams::default(),
            back_split: true,
            calibrate: true,
        }
    }
}

pub struct V1Fit {
    pub params: V1Params,
    pub team: TeamFit,
    pub minutes: MinutesFit,
    pub availability: AvailabilityFit,
    pub shares: SharesFit,
    pub components: ComponentsFit,
}

/// Every component fit on the rows visible in `view` (the refit cutoff's view).
pub fn fit_v1(view: &AsOfView, params: &V1Params) -> Result<V1Fit> {
    let team = fit_team(view, &params.team)?;
    let history = training_frame(view)?;
    let minutes = fit_minutes(view, &params.minutes, Some(&history))?;
    let availability = fit_availability(view, &minutes, Some(&history), params.back_split)?;
    let shares = fit_shares(view, &params.shares)?;
    let components = fit_components(view, &params.components)?;
    Ok(V1Fit { params: params.clone(), team, minutes, availability, shares, components })
}

// --- Poisson helpers ------------------------------------------------------------------------

/// P(N = n), n = 0..=POISSON_CAP (rate 0 gives P(0) = 1).
fn poisson_pmf(rate: f64) -> [f64; POISSON_CAP + 1] {
    let rate = rate.max(0.0);
    let mut out = [0.0; POISSON_CAP + 1];
    let base = (-rate).exp();
    out[0] = base;
    let mut product = 1.0;
    for n in 1..=POISSON_CAP {
        product *= rate / n as f64;
        out[n] = base * product;
    }
    out
}

/// E[floor(N / k)] for N ~ Poisson(rate).
fn poisson_expected_floor(rate: f64, k: i64) -> f64 {
    let pmf = poisson_pmf(rate);
    (0..=POISSON_CAP).map(|n| pmf[n] * (n as i64 / k) as f64).sum()
}

/// P(N ≥ k) for N ~ Poisson(rate).
fn poisson_tail(rate: f64, k: usize) -> f64 {
    let pmf = poisson_pmf(rate);
    1.0 - np_sum(&pmf[..k])
}

// --- per fixture ----------------------------------------------------------------------------

/// One player-fixture (`FIXTURE_COLUMNS`), then its points.
#[derive(Clone, Debug, Default)]
pub struct Fixture {
    pub key: crate::minutes::Key,
    pub element_type: i64,
    pub opponent_team_key: i64,
    pub p_start: f64,
    pub p_60: f64,
    pub p_sub: f64,
    pub p_play: f64,
    pub e_minutes: f64,
    pub c60: f64,
    pub csub: f64,
    pub m_start: f64,
    pub m_sub: f64,
    pub m_sixty: f64,
    pub lambda_for: f64,
    pub lambda_against: f64,
    pub p_cs: f64,
    pub e_np_goals: f64,
    pub e_pen_goals: f64,
    pub e_assists: f64,
    pub saves_rate: f64,
    pub pen_save_rate: f64,
    pub yellow_rate: f64,
    pub red_rate: f64,
    pub own_goal_rate: f64,
    pub pen_miss_ratio: f64,
    pub defcon_rate: f64,
    pub defcon_r: f64,
    pub bonus: [f64; 9],
    // After `fixture_points`:
    pub p_goal: f64,
    pub e_goals: f64,
    pub e_bonus: f64,
    pub p_defcon: f64,
    pub points: [f64; 13],
    pub xp: f64,
}

/// Uncalibrated per player-fixture inputs, sorted by (player_key, horizon, fixture_key).
pub fn fixture_components(view: &AsOfView, fit: &V1Fit) -> Result<Vec<Fixture>> {
    let team = team_lambdas(view, &fit.team)?;
    let minutes = predict_minutes(view, &fit.minutes)?;
    let minutes = adjust_minutes(view, &minutes, &fit.availability)?;
    let shares = predict_shares(view, &fit.shares, &minutes, &team)?;
    let components = predict_components(view, &fit.components, &minutes, &team, &fit.shares)?;
    let cond = conditionals(&minutes, fit.minutes.minutes_sub);
    let (tf, tt) = (team.i64("fixture_key")?, team.i64("team_key")?);
    let (lf, la, cs) = (team.f64("lambda_for")?, team.f64("lambda_against")?, team.f64("p_cs")?);
    let sides: HashMap<(i64, i64), (f64, f64, f64)> =
        (0..team.height()).map(|i| ((tf[i], tt[i]), (lf[i], la[i], cs[i]))).collect();
    let mut out = Vec::with_capacity(minutes.len());
    let mut missing = std::collections::BTreeSet::new();
    for i in 0..minutes.len() {
        let side = sides.get(&(minutes.fixture_key[i], minutes.team_key[i])).copied();
        let (lambda_for, lambda_against, p_cs) = side.unwrap_or((f64::NAN, f64::NAN, f64::NAN));
        if lambda_for.is_nan() {
            missing.insert(minutes.fixture_key[i]);
        }
        out.push(Fixture {
            key: shares.keys[i],
            element_type: components.element_type[i],
            opponent_team_key: components.opponent_team_key[i],
            p_start: minutes.p_start[i],
            p_60: minutes.p_60[i],
            p_sub: minutes.p_sub[i],
            p_play: minutes.p_play[i],
            e_minutes: minutes.e_minutes[i],
            c60: cond.c60[i],
            csub: cond.csub[i],
            m_start: cond.m_start[i],
            m_sub: fit.minutes.minutes_sub,
            m_sixty: fit.minutes.minutes_sixty,
            lambda_for,
            lambda_against,
            p_cs,
            e_np_goals: shares.e_np_goals[i],
            e_pen_goals: shares.e_pen_goals[i],
            e_assists: shares.e_assists[i],
            saves_rate: components.saves_rate[i],
            pen_save_rate: components.pen_save_rate[i],
            yellow_rate: components.yellow_rate[i],
            red_rate: components.red_rate[i],
            own_goal_rate: components.own_goal_rate[i],
            pen_miss_ratio: components.pen_miss_ratio[i],
            defcon_rate: components.defcon_rate[i],
            defcon_r: components.defcon_r[i],
            bonus: components.bonus[i],
            ..Fixture::default()
        });
    }
    if !missing.is_empty() {
        let first: Vec<i64> = missing.iter().take(10).copied().collect();
        bail!(
            "v1 at {}: no team λ for {} horizon fixture(s) {first:?} (team_lambdas and the \
             minutes frame disagree)",
            view.deadline(),
            missing.len()
        );
    }
    Ok(out)
}

/// P(goals ≥ 1) = p_play · (1 − exp(−e_goals / p_play)) (0 when p_play is 0).
fn p_goal(e_goals: f64, p_play: f64) -> f64 {
    let rate = if p_play > 0.0 { e_goals / p_play } else { 0.0 };
    p_play * -(-rate).exp_m1()
}

/// Steps 1–3 of the calibration (isotonic P(start), P(CS), P(goal ≥ 1)).
fn calibrate_inputs(frame: &mut [Fixture], calibration: Option<&Calibration>) {
    let Some(calibration) = calibration else { return };
    for f in frame.iter_mut() {
        let h = f.key.horizon;
        if let Some(maps) = &calibration.p_start {
            let old_p = f.p_start;
            let p = np_clip(apply_isotonic(maps, old_p, h), 0.0, 1.0);
            let changed = p != old_p;
            let p_sub = (1.0 - p) * f.csub;
            let e_minutes = p * f.m_start + p_sub * f.m_sub;
            let old = f.e_minutes;
            let ratio = if changed && old > 0.0 { e_minutes / old } else { 1.0 };
            f.p_start = p;
            if changed {
                f.p_60 = p * f.c60;
                f.p_sub = p_sub;
                f.p_play = p + p_sub;
                f.e_minutes = e_minutes;
            }
            f.e_np_goals *= ratio;
            f.e_pen_goals *= ratio;
            f.e_assists *= ratio;
        }
        if let Some(maps) = &calibration.p_cs {
            f.p_cs = np_clip(apply_isotonic(maps, f.p_cs, h), 0.0, 1.0);
        }
        if let Some(maps) = &calibration.p_goal {
            let e_goals = f.e_np_goals + f.e_pen_goals;
            let mapped = apply_isotonic(maps, p_goal(e_goals, f.p_play), h);
            let share = np_clip(if f.p_play > 0.0 { mapped / f.p_play } else { 0.0 }, 0.0, 1.0);
            let new_goals = f.p_play * -(-np_minimum(share, 1.0 - 1e-9)).ln_1p();
            let ratio = if e_goals > 0.0 { new_goals / e_goals } else { 1.0 };
            f.e_np_goals *= ratio;
            f.e_pen_goals *= ratio;
        }
    }
}

fn by_position(map: &BTreeMap<i64, i64>, element_type: i64) -> f64 {
    map.get(&element_type).copied().unwrap_or(0) as f64
}

/// Expected points per fixture under `rules` (Python's `fixture_points`).
fn fixture_points(frame: &mut [Fixture], rules: &Rules) {
    let thresholds: HashMap<i64, i64> =
        rules.defcon_threshold.iter().filter_map(|(&e, &t)| t.map(|t| (e, t))).collect();
    for f in frame.iter_mut() {
        let et = f.element_type;
        let p_play = f.p_play;
        let p_60 = np_minimum(f.p_60, p_play);
        let p_short = np_maximum(p_play - p_60, 0.0);
        let f_long = np_clip(f.m_sixty / 90.0, 0.0, 1.0);
        let rest = np_maximum(f.e_minutes - p_60 * f_long * 90.0, 0.0);
        let f_short = np_clip(
            if p_short > 0.0 { rest / (90.0 * p_short) } else { 0.0 },
            0.0,
            SHORT_MAX_FRACTION,
        );
        let split_floor = |rate: f64, k: i64| -> f64 {
            p_60 * poisson_expected_floor(rate * f_long, k)
                + p_short * poisson_expected_floor(rate * f_short, k)
        };
        let e_goals = f.e_np_goals + f.e_pen_goals;
        let keeper = et == GOALKEEPER;
        let goal_rate = if p_play > 0.0 { e_goals / p_play } else { 0.0 };
        let assist_rate = if p_play > 0.0 { f.e_assists / p_play } else { 0.0 };
        let e_saves_floor = if keeper { split_floor(f.saves_rate, SAVES_FOR_BONUS) } else { 0.0 };
        let features = [
            p_play,
            p_60,
            p_play * poisson_tail(goal_rate, 1),
            p_play * poisson_tail(goal_rate, 2),
            p_play * poisson_tail(goal_rate, 3),
            p_play * poisson_tail(assist_rate, 1),
            p_play * poisson_tail(assist_rate, 2),
            p_60 * f.p_cs,
            e_saves_floor,
        ];
        let products: Vec<f64> = (0..9).map(|j| features[j] * f.bonus[j]).collect();
        let e_bonus = np_maximum(np_sum(&products), 0.0);
        let saves_points =
            if keeper { split_floor(f.saves_rate, rules.saves_per_point) } else { 0.0 };
        let exposure = f.e_minutes / 90.0;
        let p_defcon = match thresholds.get(&et) {
            Some(&threshold)
                if rules.defcon_enabled
                    && threshold > 0
                    && f.defcon_rate > 0.0
                    && f.defcon_r > 0.0 =>
            {
                p_60 * negbin_tail(f.defcon_rate * f_long, f.defcon_r, threshold)
                    + p_short * negbin_tail(f.defcon_rate * f_short, f.defcon_r, threshold)
            }
            _ => 0.0,
        };
        let points = [
            p_play * rules.short_play as f64 + p_60 * (rules.long_play - rules.short_play) as f64,
            e_goals * by_position(&rules.goals_scored, et),
            f.e_assists * rules.assists as f64,
            p_60 * f.p_cs * by_position(&rules.clean_sheets, et),
            by_position(&rules.goals_conceded, et)
                * split_floor(f.lambda_against, rules.goals_conceded_per_point),
            rules.saves as f64 * saves_points,
            rules.penalties_saved as f64 * (f.pen_save_rate * exposure),
            rules.bonus as f64 * e_bonus,
            rules.yellow_cards as f64 * (f.yellow_rate * exposure),
            rules.red_cards as f64 * (f.red_rate * exposure),
            rules.own_goals as f64 * (f.own_goal_rate * exposure),
            rules.penalties_missed as f64 * f.e_pen_goals * f.pen_miss_ratio,
            p_defcon * by_position(&rules.defensive_contribution, et),
        ];
        // np.sum over the stacked columns (axis 0): added in order.
        let mut xp = points[0];
        for p in &points[1..] {
            xp += p;
        }
        f.p_goal = p_goal(e_goals, p_play);
        f.e_goals = e_goals;
        f.e_bonus = e_bonus;
        f.p_defcon = p_defcon;
        f.points = points;
        f.xp = xp;
    }
}

/// Step 4: xp' = a · p_play + b · xp per position.
fn calibrate_xp(frame: &mut [Fixture], calibration: Option<&Calibration>) {
    let Some(map) = calibration.and_then(|c| c.xp.as_ref()) else { return };
    let lookup: HashMap<i64, (f64, f64)> = map.iter().map(|&(p, a, b)| (p, (a, b))).collect();
    for f in frame.iter_mut() {
        let (a, b) = lookup.get(&f.element_type).copied().unwrap_or((0.0, 1.0));
        f.xp = a * f.p_play + b * f.xp;
    }
}

/// Per player-fixture components and points at the view's deadline (calibrated with the
/// season's table entry unless `calibrate` is off).
pub fn predict_fixtures(view: &AsOfView, fit: &V1Fit) -> Result<Vec<Fixture>> {
    let (season, _) = view.gameweek_for_deadline()?;
    let calibration = if fit.params.calibrate { calibration_for(season)? } else { None };
    if calibration.is_some() && fit.params != V1Params::default() {
        bail!("the v1 calibration table was fitted for the default V1Params; use calibrate=false");
    }
    let mut frame = fixture_components(view, fit)?;
    calibrate_inputs(&mut frame, calibration.as_ref());
    fixture_points(&mut frame, &backtest_rules(season, &default_config_dir())?);
    calibrate_xp(&mut frame, calibration.as_ref());
    Ok(frame)
}

// --- per GW ---------------------------------------------------------------------------------

/// The xP frame from per-fixture rows (Python's `gw_frame`).
pub fn gw_frame(view: &AsOfView, fixtures: &[Fixture]) -> Result<Table> {
    let pool = player_pool(view)?;
    let upcoming = upcoming_fixtures(view)?;
    let (ugw, ugi, uh, ut, un) = (
        upcoming.i64("gw")?,
        upcoming.i64("gw_index")?,
        upcoming.i64("horizon")?,
        upcoming.i64("team_key")?,
        upcoming.i64("n_fixtures")?,
    );
    // The horizon's gameweeks, in first-appearance order (drop_duplicates), horizon ≤ HORIZON.
    let mut gameweeks: Vec<(i64, i64, i64)> = Vec::new();
    for i in 0..upcoming.height() {
        let g = (ugw[i], ugi[i], uh[i]);
        if uh[i] <= HORIZON && !gameweeks.contains(&g) {
            gameweeks.push(g);
        }
    }
    // Per (player, gw, gw_index, horizon): sums (Kahan), first non-null probabilities, the
    // fixture count and Π (1 − p_play).
    #[derive(Default)]
    struct Group {
        sums: Vec<Kahan>,
        firsts: [Option<f64>; 9],
        n: usize,
        dnp: f64,
        unknown: bool,
    }
    let mut groups: BTreeMap<(i64, i64, i64, i64), Group> = BTreeMap::new();
    for f in fixtures {
        let k = (f.key.player_key, f.key.gw, f.key.gw_index, f.key.horizon);
        let g = groups.entry(k).or_insert_with(|| Group {
            sums: vec![Kahan::default(); 1 + GW_SUMS.len() + POINT_COLUMNS.len()],
            dnp: 1.0,
            ..Group::default()
        });
        let sums = [f.xp, f.e_minutes, f.e_goals, f.e_assists, f.e_bonus];
        for (j, v) in sums.iter().chain(&f.points).enumerate() {
            if !v.is_nan() {
                g.sums[j].add(*v);
            }
        }
        let probabilities = [
            f.p_start,
            f.p_play,
            f.p_60,
            1.0 - f.p_play,
            np_maximum(f.p_play - f.p_60, 0.0),
            f.p_60,
            f.p_60 * f.p_cs,
            f.p_goal,
            f.p_defcon,
        ];
        for (j, v) in probabilities.iter().enumerate() {
            if g.firsts[j].is_none() && !v.is_nan() {
                g.firsts[j] = Some(*v);
            }
        }
        g.n += 1;
        let dnp = 1.0 - f.p_play;
        if dnp.is_nan() {
            g.unknown = true;
        } else {
            g.dnp *= dnp;
        }
    }
    let keys = pool.i64("player_key")?;
    let teams = pool.i64("team_key")?;
    let mut order: Vec<usize> = (0..pool.height()).collect();
    order.sort_by_key(|&i| keys[i]);
    let club_gws: std::collections::HashSet<(i64, i64)> =
        (0..upcoming.height()).filter(|&i| un[i] > 0).map(|i| (ut[i], ugw[i])).collect();
    let n_columns = 1 + GW_PROBABILITIES.len() + GW_SUMS.len() + 1 + POINT_COLUMNS.len();
    let mut columns: Vec<Vec<f64>> = vec![Vec::new(); n_columns];
    let (mut out_player, mut out_gw, mut out_index, mut out_horizon) =
        (vec![], vec![], vec![], vec![]);
    // Rows by (player, horizon): pool sorted by key, gameweeks sorted by horizon.
    let mut sorted_gws = gameweeks.clone();
    sorted_gws.sort_by_key(|g| g.2);
    let mut dropped = Vec::new();
    for &i in &order {
        for &(gw, gwi, h) in &sorted_gws {
            let key = (keys[i], gw, gwi, h);
            out_player.push(keys[i]);
            out_gw.push(gw);
            out_index.push(gwi);
            out_horizon.push(h);
            match groups.get(&key) {
                None => {
                    if club_gws.contains(&(teams[i], gw)) {
                        dropped.push(keys[i]);
                    }
                    // A blank: 0, P(0 minutes) 1.
                    columns[0].push(0.0);
                    for (j, name) in GW_PROBABILITIES.iter().enumerate() {
                        columns[1 + j].push(if *name == "p_min_0" { 1.0 } else { 0.0 });
                    }
                    for c in columns.iter_mut().skip(1 + GW_PROBABILITIES.len()) {
                        c.push(0.0);
                    }
                }
                Some(g) => {
                    let double = g.n > 1;
                    columns[0].push(g.sums[0].value());
                    for j in 0..GW_PROBABILITIES.len() {
                        columns[1 + j].push(if double {
                            f64::NAN
                        } else {
                            g.firsts[j].unwrap_or(f64::NAN)
                        });
                    }
                    let base = 1 + GW_PROBABILITIES.len();
                    for j in 0..GW_SUMS.len() {
                        columns[base + j].push(g.sums[1 + j].value());
                    }
                    columns[base + GW_SUMS.len()].push(if g.unknown {
                        f64::NAN
                    } else {
                        1.0 - g.dnp
                    });
                    for j in 0..POINT_COLUMNS.len() {
                        columns[base + GW_SUMS.len() + 1 + j]
                            .push(g.sums[1 + GW_SUMS.len() + j].value());
                    }
                }
            }
        }
    }
    if !dropped.is_empty() {
        dropped.sort_unstable();
        dropped.dedup();
        bail!(
            "v1 at {}: {} pool player-GW(s) with a fixture have no per-fixture rows, e.g. \
             players {:?}",
            view.deadline(),
            dropped.len(),
            &dropped[..dropped.len().min(10)]
        );
    }
    let mut table = Table::new()
        .with("player_key", Column::I64(out_player))
        .with("gw", Column::I64(out_gw))
        .with("gw_index", Column::I64(out_index))
        .with("horizon", Column::I64(out_horizon));
    let names: Vec<&str> = std::iter::once("xp")
        .chain(GW_PROBABILITIES)
        .chain(GW_SUMS)
        .chain(std::iter::once("p_play_gw"))
        .chain(POINT_COLUMNS)
        .collect();
    // NaN as null, as pandas writes it (single-fixture probabilities in a double).
    for (name, values) in names.into_iter().zip(columns) {
        table.push(name, floats(values));
    }
    Ok(table)
}

/// The v1 xP frame at the view's deadline, from `fit`.
pub fn predict_v1(view: &AsOfView, fit: &V1Fit) -> Result<Table> {
    gw_frame(view, &predict_fixtures(view, fit)?)
}

/// `MODELS["v1"](view)`: fit at the refit cutoff, predict at the view.
pub fn xp_v1(view: &AsOfView) -> Result<Table> {
    let cutoff = refit_deadline(view, REFIT_EVERY)?;
    let fit = fit_v1(&view.earlier(cutoff)?, &V1Params::default())?;
    predict_v1(view, &fit)
}

impl V1Fit {
    /// Every component's fitted values (boosters as model text).
    pub fn to_json(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "team": self.team.to_json(),
            "minutes": self.minutes.to_json()?,
            "availability": self.availability.to_json(),
            "shares": self.shares.to_json(),
            "components": self.components.to_json(),
        }))
    }
}

/// `xp_v1` with the fit's values (the parity gate's gate 1).
pub fn xp_v1_with_fit(view: &AsOfView) -> Result<(Table, serde_json::Value)> {
    let cutoff = refit_deadline(view, REFIT_EVERY)?;
    let fit = fit_v1(&view.earlier(cutoff)?, &V1Params::default())?;
    Ok((predict_v1(view, &fit)?, fit.to_json()?))
}
