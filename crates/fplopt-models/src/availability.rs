//! The availability adjustment layer (Python `fplopt.models.availability`): FPL's status
//! flags and news applied to `predict_minutes` output.
//!
//! - Flag categories (newest snapshot of the season before the deadline): `a`, `a100`, `d25`,
//!   `d50`, `d75`, `i`, `s`, `u`, `n`; with `back_split`, `a_back` / `a100_back` for a player
//!   available again after missing his club's last match (plus β · (start_rate_l10 − 0.5)).
//! - Return dates parsed from "Expected back DD Mon" / "Suspended until DD Mon": fixtures
//!   before the date get p_start = p_sub = 0.
//! - The mapping logit p' = slope · logit p + γ[category, min(h, 3)], fitted walk-forward on
//!   every past deadline's flags against the training rows' starts (out-of-fold P(start) as
//!   the base), log loss with a Gaussian prior on the offsets, `optim::trust_exact`.

use std::collections::HashMap;
use std::sync::LazyLock;

use anyhow::Result;
use chrono::{DateTime, Datelike, NaiveDate};
use fplopt_features::history::{prediction_frame, training_frame};
use fplopt_store::{AsOfView, Micros, Table};
use regex::Regex;

use crate::gbm::float_column;
use crate::minutes::{
    Minutes, MinutesFit, MinutesParams, conditionals, finish_minutes, part, season_weights,
};
use crate::optim::{Mat, trust_exact};
use crate::util::{expit, logit, np_clip, np_sum, opt_ts};

const EPS: f64 = 1e-6;
pub const CATEGORIES: [&str; 9] = ["a", "a100", "d25", "d50", "d75", "i", "s", "u", "n"];
const BACK_CATEGORIES: [&str; 2] = ["a", "a100"];
const BACK_SUFFIX: &str = "_back";
const REGULAR_CENTER: f64 = 0.5;
const MIN_BACK_ROWS: usize = 2000;
const HORIZON_BUCKETS: i64 = 4;
const LOOKAHEAD: i64 = 5;
const MIN_ROWS: usize = 500;
const OFFSET_SD: f64 = 4.0;
const PAST_DAYS: i64 = 60;
const YEAR_DAYS: i64 = 366;
const DAY: i64 = 86_400_000_000;
const MONTHS: [&str; 12] =
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];
static RETURN_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:expected back|suspended until)\s+(\d{1,2})\s+([a-z]{3})").unwrap()
});

/// The fitted mapping.
#[derive(Clone, Debug)]
pub struct AvailabilityFit {
    pub slope: f64,
    pub offsets: Vec<(String, i64, f64)>,
    pub minutes_sub: f64,
    pub max_shift: f64,
    pub max_scale: f64,
    pub n_rows: usize,
    pub back_split: bool,
    pub back_slopes: Vec<(String, i64, f64)>,
}

/// The CATEGORIES label ('' = none) for a status and chance of playing.
fn flag_category(status: Option<&str>, chance: Option<f64>) -> String {
    let status = status.unwrap_or("");
    let chance = chance.filter(|c| !c.is_nan());
    match status {
        "i" | "s" | "u" | "n" => status.to_string(),
        "a" if chance.is_none() => "a".to_string(),
        "a" if chance == Some(100.0) => "a100".to_string(),
        "d" => match chance {
            Some(c) if c == 25.0 || c == 50.0 || c == 75.0 => format!("d{c}"),
            _ => String::new(),
        },
        _ => String::new(),
    }
}

/// UTC midnight of an instant (`dt.normalize()`).
fn midnight(at: Micros) -> Micros {
    at.div_euclid(DAY) * DAY
}

fn date_micros(year: i32, month: u32, day: u32) -> Option<Micros> {
    let date = NaiveDate::from_ymd_opt(year, month, day)?;
    Some(date.and_hms_opt(0, 0, 0)?.and_utc().timestamp_micros())
}

/// The return date (UTC midnight) in the news, if any: the first of the reference's year −
/// 1, year, year + 1 that puts it in [reference's day − PAST_DAYS, that + YEAR_DAYS).
fn expected_back(news: Option<&str>, reference: Option<Micros>) -> Option<Micros> {
    let captures = RETURN_PATTERN.captures(news?)?;
    let day: u32 = captures[1].parse().ok()?;
    let month = MONTHS.iter().position(|m| *m == captures[2].to_lowercase())? as u32 + 1;
    let reference = reference?;
    let earliest = midnight(reference) - PAST_DAYS * DAY;
    let year = DateTime::from_timestamp_micros(reference)?.year();
    let mut out = None;
    for offset in [1, 0, -1] {
        if let Some(candidate) = date_micros(year + offset, month, day)
            && candidate >= earliest
            && candidate < earliest + YEAR_DAYS * DAY
        {
            out = Some(candidate);
        }
    }
    out
}

/// Per player of the newest snapshot of the view's season before its deadline: (category,
/// news, reference time).
struct SnapshotFlag {
    player_key: i64,
    category: String,
    news: Option<String>,
    reference: Option<Micros>,
}

fn snapshot_flags(view: &AsOfView) -> Result<Vec<SnapshotFlag>> {
    let (season, _) = view.gameweek_for_deadline()?;
    let columns =
        ["snapshot_at", "season", "status", "chance_of_playing_next_round", "news", "news_added"];
    let snaps = view.latest("player_snapshot", &["player_key"], Some(&columns))?;
    let keys = snaps.i64("player_key")?;
    let seasons = snaps.i64("season")?;
    let at = opt_ts(&snaps, "snapshot_at")?;
    let status = snaps.opt_str("status")?;
    let chance = snaps.opt_f64("chance_of_playing_next_round")?;
    let news = snaps.opt_str("news")?;
    let added = opt_ts(&snaps, "news_added")?;
    let in_season: Vec<usize> = (0..snaps.height()).filter(|&i| seasons[i] == season).collect();
    let Some(newest) = in_season.iter().filter_map(|&i| at[i]).max() else { return Ok(Vec::new()) };
    Ok(in_season
        .into_iter()
        .filter(|&i| at[i] == Some(newest))
        .map(|i| SnapshotFlag {
            player_key: keys[i],
            category: flag_category(status[i].as_deref(), chance[i]),
            news: news[i].clone(),
            reference: added[i].or(at[i]),
        })
        .collect())
}

/// (category with BACK_SUFFIX where an `a`/`a100` player missed his last match, the β
/// covariate there and 0 elsewhere).
fn back_label(category: &str, absent_run: f64, regular: f64) -> (String, f64) {
    let absent = if absent_run.is_nan() { 0.0 } else { absent_run };
    if BACK_CATEGORIES.contains(&category) && absent >= 1.0 {
        let regular = if regular.is_nan() { 0.0 } else { regular };
        (format!("{category}{BACK_SUFFIX}"), regular - REGULAR_CENTER)
    } else {
        (category.to_string(), 0.0)
    }
}

/// `out_of_fold_start`: per training row a P(start) from the start part refitted on the
/// rows of the other season parity (non-banned, season-decay weights relative to `season`).
fn out_of_fold_start(rows: &Table, season: i64, params: &MinutesParams) -> Result<Vec<f64>> {
    let weight = season_weights(rows, season, params)?;
    let seasons = float_column(rows, "season")?;
    let banned = float_column(rows, "banned")?;
    let odd: Vec<bool> = seasons.iter().map(|s| (*s as i64).rem_euclid(2) == 1).collect();
    let mut out = vec![0.0; rows.height()];
    for fold in [false, true] {
        let train: Vec<usize> =
            (0..rows.height()).filter(|&i| odd[i] != fold && banned[i] == 0.0).collect();
        let w: Vec<f64> = train.iter().map(|&i| weight[i]).collect();
        let fitted = part(&rows.take(&train), &w, "start", &params.start)?;
        let target: Vec<usize> = (0..rows.height()).filter(|&i| odd[i] == fold).collect();
        let predicted = fitted.predict(&rows.take(&target))?;
        for (k, &i) in target.iter().enumerate() {
            out[i] = predicted[k];
        }
    }
    Ok(out)
}

/// One fit row: base p, category, bucket, start outcome, and the back covariates.
struct FitRow {
    base: f64,
    category: String,
    bucket: i64,
    start: f64,
    absent_run: f64,
    start_rate_l10: f64,
}

fn fit_rows(view: &AsOfView, minutes_fit: &MinutesFit, rows: &Table) -> Result<Vec<FitRow>> {
    let gameweeks = view.table("gameweek", Some(&["season", "gw", "gw_index", "deadline_time"]))?;
    let snapshots = view.table("player_snapshot", Some(&["snapshot_at"]))?;
    let Some(first) = opt_ts(&snapshots, "snapshot_at")?.into_iter().flatten().min() else {
        return Ok(Vec::new());
    };
    let deadlines = gameweeks.opt_i64("deadline_time")?;
    let (seasons, gw_index) = (gameweeks.i64("season")?, gameweeks.i64("gw_index")?);
    let mut picked: Vec<usize> = (0..gameweeks.height())
        .filter(|&i| deadlines[i].is_some_and(|d| d < view.deadline() && d > first))
        .collect();
    picked.sort_by_key(|&i| deadlines[i]);
    // (player, season, base_index) → (category, return date), in deadline order.
    let mut flags: Vec<(i64, i64, i64, String)> = Vec::new();
    for i in picked {
        for f in snapshot_flags(&view.earlier(deadlines[i].unwrap())?)? {
            if f.category.is_empty() || expected_back(f.news.as_deref(), f.reference).is_some() {
                continue;
            }
            flags.push((f.player_key, seasons[i], gw_index[i], f.category));
        }
    }
    if flags.is_empty() {
        return Ok(Vec::new());
    }
    let params = &minutes_fit.params;
    let (season, _) = view.gameweek_for_deadline()?;
    let p_model = out_of_fold_start(rows, season, params)?;
    let n_long = float_column(rows, "n_long")?;
    let rate_long = float_column(rows, "start_rate_long")?;
    let keys = float_column(rows, "player_key")?;
    let row_season = float_column(rows, "season")?;
    let row_index = float_column(rows, "gw_index")?;
    let absent = float_column(rows, "absent_run")?;
    let regular = float_column(rows, "start_rate_l10")?;
    let banned = float_column(rows, "banned")?;
    let start = float_column(rows, "start")?;
    // Bases: the first training row per (player, season, gw_index).
    let mut bases: HashMap<(i64, i64, i64), usize> = HashMap::new();
    for i in 0..rows.height() {
        bases.entry((keys[i] as i64, row_season[i] as i64, row_index[i] as i64)).or_insert(i);
    }
    // Outcomes: non-banned training rows per (player, season, gw_index), in row order.
    let mut outcomes: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for i in 0..rows.height() {
        if banned[i] == 0.0 {
            outcomes
                .entry((keys[i] as i64, row_season[i] as i64, row_index[i] as i64))
                .or_default()
                .push(i);
        }
    }
    // (season, base_index, player, gw_index, horizon) order; ties (a double) in row order.
    let mut out: Vec<((i64, i64, i64, i64, i64), FitRow)> = Vec::new();
    for (player, s, base_index, category) in flags {
        let Some(&b) = bases.get(&(player, s, base_index)) else { continue };
        let starts_long = if rate_long[b].is_nan() { 0.0 } else { rate_long[b] } * n_long[b];
        let long_run = (starts_long + params.long_run_prior * p_model[b])
            / (n_long[b] + params.long_run_prior);
        for horizon in 0..=LOOKAHEAD {
            let gwi = base_index + horizon;
            let Some(found) = outcomes.get(&(player, s, gwi)) else { continue };
            let weight = params.horizon_decay.powf(horizon as f64);
            for &o in found {
                out.push((
                    (s, base_index, player, gwi, horizon),
                    FitRow {
                        base: weight * p_model[b] + (1.0 - weight) * long_run,
                        category: category.clone(),
                        bucket: horizon.min(HORIZON_BUCKETS - 1),
                        start: start[o],
                        absent_run: absent[b],
                        start_rate_l10: regular[b],
                    },
                ));
            }
        }
    }
    out.sort_by_key(|(key, _)| *key);
    Ok(out.into_iter().map(|(_, row)| row).collect())
}

/// `np.logaddexp(0, x)`.
fn logaddexp0(x: f64) -> f64 {
    if x == 0.0 {
        std::f64::consts::LN_2
    } else {
        let tmp = -x;
        if tmp > 0.0 {
            (x.exp()).ln_1p()
        } else if tmp <= 0.0 {
            x + (-x).exp().ln_1p()
        } else {
            tmp
        }
    }
}

/// Fit the flag mapping walk-forward on the view (`rows`: its training frame).
pub fn fit_availability(
    view: &AsOfView,
    minutes_fit: &MinutesFit,
    rows: Option<&Table>,
    back_split: bool,
) -> Result<AvailabilityFit> {
    let built;
    let rows = match rows {
        Some(rows) => rows,
        None => {
            built = training_frame(view)?;
            &built
        }
    };
    let mut fit_rows = fit_rows(view, minutes_fit, rows)?;
    let params = &minutes_fit.params;
    if fit_rows.len() < MIN_ROWS {
        return Ok(AvailabilityFit {
            slope: 1.0,
            offsets: Vec::new(),
            minutes_sub: minutes_fit.minutes_sub,
            max_shift: params.max_shift,
            max_scale: params.max_scale,
            n_rows: fit_rows.len(),
            back_split,
            back_slopes: Vec::new(),
        });
    }
    let mut categories: Vec<String> = CATEGORIES.iter().map(|c| c.to_string()).collect();
    let mut backs: Vec<String> = Vec::new();
    let mut x = vec![0.0; fit_rows.len()];
    if back_split {
        for (k, row) in fit_rows.iter_mut().enumerate() {
            let (label, cov) = back_label(&row.category, row.absent_run, row.start_rate_l10);
            row.category = label;
            x[k] = cov;
        }
        for name in BACK_CATEGORIES {
            let back = format!("{name}{BACK_SUFFIX}");
            let thin = fit_rows.iter().filter(|r| r.category == back).count();
            if thin < MIN_BACK_ROWS {
                for (k, row) in fit_rows.iter_mut().enumerate() {
                    if row.category == back {
                        row.category = name.to_string();
                        x[k] = 0.0;
                    }
                }
            }
        }
        backs = BACK_CATEGORIES.iter().map(|c| format!("{c}{BACK_SUFFIX}")).collect();
        categories.extend(backs.iter().cloned());
    }
    let cells: Vec<(String, i64)> =
        categories.iter().flat_map(|c| (0..HORIZON_BUCKETS).map(move |b| (c.clone(), b))).collect();
    let cell_index: HashMap<(String, i64), usize> =
        cells.iter().enumerate().map(|(k, c)| (c.clone(), k)).collect();
    let cell: Vec<usize> =
        fit_rows.iter().map(|r| cell_index[&(r.category.clone(), r.bucket)]).collect();
    let z: Vec<f64> = fit_rows.iter().map(|r| logit(np_clip(r.base, EPS, 1.0 - EPS))).collect();
    let y: Vec<f64> = fit_rows.iter().map(|r| r.start).collect();
    let n_cells = cells.len();
    let back_cells: Vec<usize> = (0..n_cells).filter(|&k| backs.contains(&cells[k].0)).collect();
    let n_back = back_cells.len();
    let mut position = vec![0usize; n_cells];
    for (j, &k) in back_cells.iter().enumerate() {
        position[k] = j;
    }
    let bcell: Vec<usize> = cell.iter().map(|&c| position[c]).collect();
    let n_theta = 1 + n_cells + n_back;
    let n = y.len() as f64;
    let sd2 = OFFSET_SD * OFFSET_SD;
    let eta_of = |theta: &[f64]| -> Vec<f64> {
        (0..y.len())
            .map(|i| {
                let mut eta = theta[0] * z[i] + theta[1 + cell[i]];
                if n_back > 0 {
                    eta += theta[1 + n_cells + bcell[i]] * x[i];
                }
                eta
            })
            .collect()
    };
    let bincount = |codes: &[usize], weights: &dyn Fn(usize) -> f64, len: usize| -> Vec<f64> {
        let mut out = vec![0.0; len];
        for (i, &c) in codes.iter().enumerate() {
            out[c] += weights(i);
        }
        out
    };
    let loss = |theta: &[f64]| -> (f64, Vec<f64>) {
        let eta = eta_of(theta);
        let p: Vec<f64> = eta.iter().map(|&e| expit(e)).collect();
        let terms: Vec<f64> = (0..y.len()).map(|i| logaddexp0(eta[i]) - y[i] * eta[i]).collect();
        let squares: Vec<f64> = theta[1..].iter().map(|t| t * t).collect();
        let value = np_sum(&terms) + 0.5 * np_sum(&squares) / sd2;
        let residual: Vec<f64> = (0..y.len()).map(|i| p[i] - y[i]).collect();
        let mut grad = vec![0.0; n_theta];
        grad[0] = np_sum(&(0..y.len()).map(|i| residual[i] * z[i]).collect::<Vec<_>>());
        let by_cell = bincount(&cell, &|i| residual[i], n_cells);
        grad[1..1 + n_cells].copy_from_slice(&by_cell);
        if n_back > 0 {
            let by_back = bincount(&bcell, &|i| residual[i] * x[i], n_back);
            grad[1 + n_cells..].copy_from_slice(&by_back);
        }
        for j in 1..n_theta {
            grad[j] += theta[j] / sd2;
        }
        (value / n, grad.iter().map(|g| g / n).collect())
    };
    let hessian = |theta: &[f64]| -> Mat {
        let p: Vec<f64> = eta_of(theta).iter().map(|&e| expit(e)).collect();
        let w: Vec<f64> = p.iter().map(|p| p * (1.0 - p)).collect();
        let mut out = Mat::zeros(n_theta, n_theta);
        out.set(0, 0, np_sum(&(0..y.len()).map(|i| w[i] * z[i] * z[i]).collect::<Vec<_>>()));
        let cross = bincount(&cell, &|i| w[i] * z[i], n_cells);
        let diagonal = bincount(&cell, &|i| w[i], n_cells);
        for k in 0..n_cells {
            out.set(0, 1 + k, cross[k]);
            out.set(1 + k, 0, cross[k]);
            out.set(1 + k, 1 + k, diagonal[k]);
        }
        if n_back > 0 {
            let b0 = 1 + n_cells;
            let zx = bincount(&bcell, &|i| w[i] * z[i] * x[i], n_back);
            let joint_codes: Vec<usize> =
                (0..y.len()).map(|i| cell[i] * n_back + bcell[i]).collect();
            let joint = bincount(&joint_codes, &|i| w[i] * x[i], n_cells * n_back);
            let xx = bincount(&bcell, &|i| w[i] * x[i] * x[i], n_back);
            for j in 0..n_back {
                out.set(0, b0 + j, zx[j]);
                out.set(b0 + j, 0, zx[j]);
                for k in 0..n_cells {
                    out.set(1 + k, b0 + j, joint[k * n_back + j]);
                    out.set(b0 + j, 1 + k, joint[k * n_back + j]);
                }
                out.set(b0 + j, b0 + j, xx[j]);
            }
        }
        for j in 1..n_theta {
            out.add(j, j, 1.0 / sd2);
        }
        for v in out.data.iter_mut() {
            *v /= n;
        }
        out
    };
    let mut start = vec![0.0; n_theta];
    start[0] = 1.0;
    let theta = trust_exact(loss, hessian, start, 1e-9, None)?.x;
    let mut counts = vec![0usize; n_cells];
    for &c in &cell {
        counts[c] += 1;
    }
    Ok(AvailabilityFit {
        slope: theta[0],
        offsets: (0..n_cells)
            .filter(|&k| counts[k] > 0)
            .map(|k| (cells[k].0.clone(), cells[k].1, theta[1 + k]))
            .collect(),
        minutes_sub: minutes_fit.minutes_sub,
        max_shift: params.max_shift,
        max_scale: params.max_scale,
        n_rows: fit_rows.len(),
        back_split,
        back_slopes: back_cells
            .iter()
            .enumerate()
            .filter(|&(_, &k)| counts[k] > 0)
            .map(|(j, &k)| (cells[k].0.clone(), cells[k].1, theta[1 + n_cells + j]))
            .collect(),
    })
}

/// `predict_minutes` output with the flags and return dates applied and the team
/// normalization redone (unchanged without a snapshot of the season before the deadline).
pub fn adjust_minutes(
    view: &AsOfView,
    minutes: &Minutes,
    fit: &AvailabilityFit,
) -> Result<Minutes> {
    let flags = snapshot_flags(view)?;
    if flags.is_empty() || minutes.is_empty() {
        return Ok(minutes.clone());
    }
    let by_player: HashMap<i64, (String, Option<Micros>)> = flags
        .into_iter()
        .map(|f| (f.player_key, (f.category, expected_back(f.news.as_deref(), f.reference))))
        .collect();
    let (season, _) = view.gameweek_for_deadline()?;
    let schedule = view.schedule(season)?;
    let kickoff: HashMap<i64, Option<Micros>> = schedule
        .i64("fixture_key")?
        .iter()
        .copied()
        .zip(opt_ts(&schedule, "kickoff_time")?)
        .collect();
    let mut cond = conditionals(minutes, fit.minutes_sub);
    let offsets: HashMap<(String, i64), f64> =
        fit.offsets.iter().map(|(c, b, v)| ((c.clone(), *b), *v)).collect();
    let slopes: HashMap<(String, i64), f64> =
        fit.back_slopes.iter().map(|(c, b, v)| ((c.clone(), *b), *v)).collect();
    let lags: HashMap<(i64, i64), (f64, f64)> = if fit.back_split {
        let frame = prediction_frame(view)?;
        let (p, f) = (float_column(&frame, "player_key")?, float_column(&frame, "fixture_key")?);
        let (a, r) = (float_column(&frame, "absent_run")?, float_column(&frame, "start_rate_l10")?);
        (0..frame.height()).map(|i| ((p[i] as i64, f[i] as i64), (a[i], r[i]))).collect()
    } else {
        HashMap::new()
    };
    for i in 0..minutes.len() {
        let (category, back) =
            by_player.get(&minutes.player_key[i]).cloned().unwrap_or((String::new(), None));
        let kickoff_day = kickoff.get(&minutes.fixture_key[i]).copied().flatten().map(midnight);
        let out_until_back = matches!((back, kickoff_day), (Some(b), Some(k)) if k < b);
        let bucket = minutes.horizon[i].min(HORIZON_BUCKETS - 1);
        let (category, beta) = if fit.back_split {
            let (absent, regular) = lags
                .get(&(minutes.player_key[i], minutes.fixture_key[i]))
                .copied()
                .unwrap_or((f64::NAN, f64::NAN));
            let (label, x) = back_label(&category, absent, regular);
            let slope = slopes.get(&(label.clone(), bucket)).copied().unwrap_or(0.0);
            (label, x * slope)
        } else {
            (category, 0.0)
        };
        let offset = offsets.get(&(category.clone(), bucket)).copied().or_else(|| {
            let plain = category.strip_suffix(BACK_SUFFIX).unwrap_or(&category).to_string();
            offsets.get(&(plain, bucket)).copied()
        });
        let p = cond.p_start[i];
        let csub = cond.csub[i];
        let banned = cond.banned[i];
        let mapped = back.is_none() && offset.is_some() && p > 0.0 && !banned;
        let adjusted =
            expit(fit.slope * logit(np_clip(p, EPS, 1.0 - EPS)) + offset.unwrap_or(0.0) + beta);
        let ratio = if p > 0.0 { adjusted / p } else { 1.0 };
        let mut new_p = if mapped { adjusted } else { p };
        let mut new_csub = if mapped { csub * crate::util::np_minimum(ratio, 1.0) } else { csub };
        if out_until_back {
            new_p = 0.0;
            new_csub = 0.0;
        }
        cond.p_start[i] = new_p;
        cond.csub[i] = new_csub;
        cond.fixed[i] = banned || new_p <= 0.0;
    }
    Ok(finish_minutes(&cond, fit.minutes_sub, fit.max_shift, fit.max_scale))
}

impl AvailabilityFit {
    pub fn to_json(&self) -> serde_json::Value {
        let cells = |v: &[(String, i64, f64)]| -> Vec<serde_json::Value> {
            v.iter().map(|(c, b, x)| serde_json::json!([c, b, x])).collect()
        };
        serde_json::json!({
            "slope": self.slope,
            "offsets": cells(&self.offsets),
            "minutes_sub": self.minutes_sub,
            "max_shift": self.max_shift,
            "max_scale": self.max_scale,
            "n_rows": self.n_rows,
            "back_split": self.back_split,
            "back_slopes": cells(&self.back_slopes),
        })
    }
}
