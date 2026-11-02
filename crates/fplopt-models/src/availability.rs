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
//!   With `category_slopes`, one slope per category (a Gaussian prior around 1, sd
//!   `SLOPE_SD`); a back category uses its plain one's at prediction if the fit never saw it.

use std::collections::HashMap;
use std::sync::LazyLock;

use anyhow::Result;
use chrono::{DateTime, Datelike, NaiveDate};
use fplopt_features::history::prediction_frame;
use fplopt_store::{AsOfView, Micros, Table};
use regex::Regex;

use crate::gbm::float_column;
use crate::minutes::{
    Minutes, MinutesFit, MinutesParams, booster_features, conditionals, finish_minutes,
    history_frame, horizon_start, part, season_weights,
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
const SLOPE_SD: f64 = 1.0; // `category_slopes`: prior sd of each category's slope around 1
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
    /// The shared slope; with `category_slopes`, 1.0: the fallback for a category without
    /// its own slope.
    pub slope: f64,
    pub offsets: Vec<(String, i64, f64)>,
    pub minutes_sub: f64,
    pub max_shift: f64,
    pub max_scale: f64,
    pub n_rows: usize,
    pub back_split: bool,
    pub back_slopes: Vec<(String, i64, f64)>,
    /// With `category_slopes`, each seen category's slope (Python `slopes`); empty otherwise.
    pub slopes: Vec<(String, f64)>,
}

/// The slope a category uses: its own, its plain category's for a back category, else the
/// shared slope (Python `adjust_minutes`).
fn slope_for(fit: &AvailabilityFit, category: &str) -> f64 {
    let find = |c: &str| fit.slopes.iter().find(|(k, _)| k == c).map(|(_, s)| *s);
    find(category)
        .or_else(|| category.strip_suffix(BACK_SUFFIX).and_then(find))
        .unwrap_or(fit.slope)
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
        let fitted =
            part(&rows.take(&train), &w, "start", &params.start, &booster_features(params))?;
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
            for &o in found {
                out.push((
                    (s, base_index, player, gwi, horizon),
                    FitRow {
                        base: horizon_start(p_model[b], long_run, horizon as f64, params),
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

/// Fit the flag mapping walk-forward on the view (`rows`: its training frame;
/// `category_slopes`: one slope per category instead of a shared one).
pub fn fit_availability(
    view: &AsOfView,
    minutes_fit: &MinutesFit,
    rows: Option<&Table>,
    back_split: bool,
    category_slopes: bool,
) -> Result<AvailabilityFit> {
    let built;
    let rows = match rows {
        Some(rows) => rows,
        None => {
            built = history_frame(view, &minutes_fit.params)?;
            &built
        }
    };
    let fit_rows = fit_rows(view, minutes_fit, rows)?;
    let carried = Carried {
        minutes_sub: minutes_fit.minutes_sub,
        max_shift: minutes_fit.params.max_shift,
        max_scale: minutes_fit.params.max_scale,
    };
    fit_mapping(fit_rows, &carried, back_split, category_slopes)
}

/// The minutes fit's settings an `AvailabilityFit` carries.
struct Carried {
    minutes_sub: f64,
    max_shift: f64,
    max_scale: f64,
}

/// `np.bincount(codes, weights, minlength=len)`: sequential sums.
fn bincount(codes: &[usize], weights: &dyn Fn(usize) -> f64, len: usize) -> Vec<f64> {
    let mut out = vec![0.0; len];
    for (i, &c) in codes.iter().enumerate() {
        out[c] += weights(i);
    }
    out
}

/// The flag mapping's objective on the fit rows: per-row log loss with the Gaussian priors,
/// its gradient and Hessian, all divided by the row count. θ = [slopes (one shared, or one
/// per category), offsets per cell, back slopes].
struct Objective {
    category_slopes: bool,
    categories: Vec<String>,
    cells: Vec<(String, i64)>,
    back_cells: Vec<usize>,
    y: Vec<f64>,
    z: Vec<f64>,
    x: Vec<f64>,
    cell: Vec<usize>,
    bcell: Vec<usize>,
    sidx: Vec<usize>,
    cell_slope: Vec<usize>,
    n_s: usize,
    n_cells: usize,
    n_back: usize,
}

impl Objective {
    /// Relabels the rows' back categories (`back_split`) and indexes their cells.
    fn new(fit_rows: &mut [FitRow], back_split: bool, category_slopes: bool) -> Objective {
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
        let cells: Vec<(String, i64)> = categories
            .iter()
            .flat_map(|c| (0..HORIZON_BUCKETS).map(move |b| (c.clone(), b)))
            .collect();
        let cell_index: HashMap<(String, i64), usize> =
            cells.iter().enumerate().map(|(k, c)| (c.clone(), k)).collect();
        let cell: Vec<usize> =
            fit_rows.iter().map(|r| cell_index[&(r.category.clone(), r.bucket)]).collect();
        let z: Vec<f64> = fit_rows.iter().map(|r| logit(np_clip(r.base, EPS, 1.0 - EPS))).collect();
        let y: Vec<f64> = fit_rows.iter().map(|r| r.start).collect();
        let n_cells = cells.len();
        let back_cells: Vec<usize> =
            (0..n_cells).filter(|&k| backs.contains(&cells[k].0)).collect();
        let n_back = back_cells.len();
        let mut position = vec![0usize; n_cells];
        for (j, &k) in back_cells.iter().enumerate() {
            position[k] = j;
        }
        let bcell: Vec<usize> = cell.iter().map(|&c| position[c]).collect();
        // The slopes come first in θ: one shared, or one per category (a cell's category is
        // cell / HORIZON_BUCKETS, the product's order).
        let n_s = if category_slopes { categories.len() } else { 1 };
        let cell_slope: Vec<usize> = (0..n_cells)
            .map(|k| if category_slopes { k / HORIZON_BUCKETS as usize } else { 0 })
            .collect();
        let sidx: Vec<usize> = cell.iter().map(|&c| cell_slope[c]).collect();
        Objective {
            category_slopes,
            categories,
            cells,
            back_cells,
            y,
            z,
            x,
            cell,
            bcell,
            sidx,
            cell_slope,
            n_s,
            n_cells,
            n_back,
        }
    }

    fn n_theta(&self) -> usize {
        self.n_s + self.n_cells + self.n_back
    }

    fn eta_of(&self, theta: &[f64]) -> Vec<f64> {
        let (n_s, n_cells) = (self.n_s, self.n_cells);
        (0..self.y.len())
            .map(|i| {
                let mut eta = theta[self.sidx[i]] * self.z[i] + theta[n_s + self.cell[i]];
                if self.n_back > 0 {
                    eta += theta[n_s + n_cells + self.bcell[i]] * self.x[i];
                }
                eta
            })
            .collect()
    }

    /// The shared slope's sums are numpy sums (np_sum, as before); per category a bincount.
    fn per_slope(&self, weights: &dyn Fn(usize) -> f64) -> Vec<f64> {
        if self.n_s == 1 {
            vec![np_sum(&(0..self.y.len()).map(weights).collect::<Vec<_>>())]
        } else {
            bincount(&self.sidx, weights, self.n_s)
        }
    }

    fn loss(&self, theta: &[f64]) -> (f64, Vec<f64>) {
        let (y, z, x) = (&self.y, &self.z, &self.x);
        let (n_s, n_cells, n_back) = (self.n_s, self.n_cells, self.n_back);
        let n_theta = self.n_theta();
        let n = y.len() as f64;
        let sd2 = OFFSET_SD * OFFSET_SD;
        let slope_sd2 = SLOPE_SD * SLOPE_SD;
        let eta = self.eta_of(theta);
        let p: Vec<f64> = eta.iter().map(|&e| expit(e)).collect();
        let terms: Vec<f64> = (0..y.len()).map(|i| logaddexp0(eta[i]) - y[i] * eta[i]).collect();
        let squares: Vec<f64> = theta[n_s..].iter().map(|t| t * t).collect();
        let mut value = np_sum(&terms) + 0.5 * np_sum(&squares) / sd2;
        let residual: Vec<f64> = (0..y.len()).map(|i| p[i] - y[i]).collect();
        let mut grad = vec![0.0; n_theta];
        grad[..n_s].copy_from_slice(&self.per_slope(&|i| residual[i] * z[i]));
        let by_cell = bincount(&self.cell, &|i| residual[i], n_cells);
        grad[n_s..n_s + n_cells].copy_from_slice(&by_cell);
        if n_back > 0 {
            let by_back = bincount(&self.bcell, &|i| residual[i] * x[i], n_back);
            grad[n_s + n_cells..].copy_from_slice(&by_back);
        }
        for j in n_s..n_theta {
            grad[j] += theta[j] / sd2;
        }
        if self.category_slopes {
            let squares: Vec<f64> = theta[..n_s].iter().map(|t| (t - 1.0) * (t - 1.0)).collect();
            value += 0.5 * np_sum(&squares) / slope_sd2;
            for j in 0..n_s {
                grad[j] += (theta[j] - 1.0) / slope_sd2;
            }
        }
        (value / n, grad.iter().map(|g| g / n).collect())
    }

    fn hessian(&self, theta: &[f64]) -> Mat {
        let (y, z, x) = (&self.y, &self.z, &self.x);
        let (n_s, n_cells, n_back) = (self.n_s, self.n_cells, self.n_back);
        let (cell, bcell, cell_slope) = (&self.cell, &self.bcell, &self.cell_slope);
        let n_theta = self.n_theta();
        let n = y.len() as f64;
        let sd2 = OFFSET_SD * OFFSET_SD;
        let slope_sd2 = SLOPE_SD * SLOPE_SD;
        let p: Vec<f64> = self.eta_of(theta).iter().map(|&e| expit(e)).collect();
        let w: Vec<f64> = p.iter().map(|p| p * (1.0 - p)).collect();
        let mut out = Mat::zeros(n_theta, n_theta);
        for (j, v) in self.per_slope(&|i| w[i] * z[i] * z[i]).into_iter().enumerate() {
            out.set(j, j, v);
        }
        let cross = bincount(cell, &|i| w[i] * z[i], n_cells);
        let diagonal = bincount(cell, &|i| w[i], n_cells);
        // Assignments, as Python's `out[cell_slope, offset_index] = cross` (each offset index
        // appears once).
        for k in 0..n_cells {
            out.set(cell_slope[k], n_s + k, cross[k]);
            out.set(n_s + k, cell_slope[k], cross[k]);
            out.set(n_s + k, n_s + k, diagonal[k]);
        }
        if n_back > 0 {
            let b0 = n_s + n_cells;
            let zx = bincount(bcell, &|i| w[i] * z[i] * x[i], n_back);
            let joint_codes: Vec<usize> =
                (0..y.len()).map(|i| cell[i] * n_back + bcell[i]).collect();
            let joint = bincount(&joint_codes, &|i| w[i] * x[i], n_cells * n_back);
            let xx = bincount(bcell, &|i| w[i] * x[i] * x[i], n_back);
            for j in 0..n_back {
                let s = cell_slope[self.back_cells[j]];
                out.set(s, b0 + j, zx[j]);
                out.set(b0 + j, s, zx[j]);
                for k in 0..n_cells {
                    out.set(n_s + k, b0 + j, joint[k * n_back + j]);
                    out.set(b0 + j, n_s + k, joint[k * n_back + j]);
                }
                out.set(b0 + j, b0 + j, xx[j]);
            }
        }
        for j in n_s..n_theta {
            out.add(j, j, 1.0 / sd2);
        }
        if self.category_slopes {
            for j in 0..n_s {
                out.add(j, j, 1.0 / slope_sd2);
            }
        }
        for v in out.data.iter_mut() {
            *v /= n;
        }
        out
    }
}

/// The optimization part of `fit_availability`, on its fit rows.
fn fit_mapping(
    mut fit_rows: Vec<FitRow>,
    carried: &Carried,
    back_split: bool,
    category_slopes: bool,
) -> Result<AvailabilityFit> {
    if fit_rows.len() < MIN_ROWS {
        return Ok(AvailabilityFit {
            slope: 1.0,
            offsets: Vec::new(),
            minutes_sub: carried.minutes_sub,
            max_shift: carried.max_shift,
            max_scale: carried.max_scale,
            n_rows: fit_rows.len(),
            back_split,
            back_slopes: Vec::new(),
            slopes: Vec::new(),
        });
    }
    let objective = Objective::new(&mut fit_rows, back_split, category_slopes);
    let Objective {
        n_s,
        n_cells,
        ref cell,
        ref sidx,
        ref back_cells,
        ref categories,
        ref cells,
        ..
    } = objective;
    let mut start = vec![0.0; objective.n_theta()];
    start[..n_s].fill(1.0);
    let theta = trust_exact(
        |theta: &[f64]| objective.loss(theta),
        |theta: &[f64]| objective.hessian(theta),
        start,
        1e-9,
        None,
    )?
    .x;
    let mut counts = vec![0usize; n_cells];
    for &c in cell {
        counts[c] += 1;
    }
    let mut slope_counts = vec![0usize; n_s];
    for &j in sidx {
        slope_counts[j] += 1;
    }
    Ok(AvailabilityFit {
        slope: if category_slopes { 1.0 } else { theta[0] },
        offsets: (0..n_cells)
            .filter(|&k| counts[k] > 0)
            .map(|k| (cells[k].0.clone(), cells[k].1, theta[n_s + k]))
            .collect(),
        minutes_sub: carried.minutes_sub,
        max_shift: carried.max_shift,
        max_scale: carried.max_scale,
        n_rows: fit_rows.len(),
        back_split,
        back_slopes: back_cells
            .iter()
            .enumerate()
            .filter(|&(_, &k)| counts[k] > 0)
            .map(|(j, &k)| (cells[k].0.clone(), cells[k].1, theta[n_s + n_cells + j]))
            .collect(),
        slopes: if category_slopes {
            (0..n_s)
                .filter(|&j| slope_counts[j] > 0)
                .map(|j| (categories[j].clone(), theta[j]))
                .collect()
        } else {
            Vec::new()
        },
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
        let slope = slope_for(fit, &category);
        let adjusted =
            expit(slope * logit(np_clip(p, EPS, 1.0 - EPS)) + offset.unwrap_or(0.0) + beta);
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
        let mut out = serde_json::json!({
            "slope": self.slope,
            "offsets": cells(&self.offsets),
            "minutes_sub": self.minutes_sub,
            "max_shift": self.max_shift,
            "max_scale": self.max_scale,
            "n_rows": self.n_rows,
            "back_split": self.back_split,
            "back_slopes": cells(&self.back_slopes),
        });
        // Only with `category_slopes`, so v1's fit JSON keeps its keys.
        if !self.slopes.is_empty() {
            out["slopes"] = serde_json::json!(
                self.slopes.iter().map(|(c, s)| serde_json::json!([c, s])).collect::<Vec<_>>()
            );
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    const CARRIED: Carried = Carried { minutes_sub: 25.0, max_shift: 0.5, max_scale: 1.5 };

    /// A deterministic uniform draw in [0, 1) (64-bit LCG, top 53 bits).
    fn uniform(state: &mut u64) -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*state >> 11) as f64 / (1u64 << 53) as f64
    }

    /// 6000 synthetic fit rows: `a` (half, 80% back from an absence: `a_back` keeps its
    /// cells), `a100` (its back rows too few: merged), `d75`, `d50`, `i`; starts drawn from a
    /// per-category slope on the base's logit plus a horizon offset.
    fn synthetic_rows() -> Vec<FitRow> {
        let mut state = 7u64;
        (0..6000)
            .map(|i| {
                let u = uniform(&mut state);
                let (category, slope, offset) = match u {
                    u if u < 0.5 => ("a", 1.1, 0.2),
                    u if u < 0.7 => ("a100", 0.9, 0.3),
                    u if u < 0.8 => ("d75", 0.7, -0.4),
                    u if u < 0.9 => ("d50", 0.6, -1.0),
                    _ => ("i", 0.4, -2.5),
                };
                let base = 0.05 + 0.9 * uniform(&mut state);
                let bucket = (i % 4) as i64;
                let absent_run = if uniform(&mut state) < 0.8 { 1.0 } else { 0.0 };
                let start_rate_l10 = uniform(&mut state);
                let eta = slope * logit(base) + offset - 0.1 * bucket as f64
                    + absent_run * (start_rate_l10 - 0.5);
                let start = if uniform(&mut state) < expit(eta) { 1.0 } else { 0.0 };
                FitRow {
                    base,
                    category: category.to_string(),
                    bucket,
                    start,
                    absent_run,
                    start_rate_l10,
                }
            })
            .collect()
    }

    fn identity_fit() -> AvailabilityFit {
        AvailabilityFit {
            slope: 1.0,
            offsets: Vec::new(),
            minutes_sub: CARRIED.minutes_sub,
            max_shift: CARRIED.max_shift,
            max_scale: CARRIED.max_scale,
            n_rows: 0,
            back_split: false,
            back_slopes: Vec::new(),
            slopes: Vec::new(),
        }
    }

    fn cells(v: &[(&str, i64, f64)]) -> Vec<(String, i64, f64)> {
        v.iter().map(|(c, b, x)| (c.to_string(), *b, *x)).collect()
    }

    /// The shared-slope fit on the synthetic rows, recorded after the `fit_mapping`
    /// extraction, before `category_slopes`.
    #[test]
    fn shared_slope_fit_is_unchanged() {
        let fit = fit_mapping(synthetic_rows(), &CARRIED, true, false).unwrap();
        assert_eq!(fit.slope, 0.9459744052097245);
        assert_eq!(
            fit.offsets,
            cells(&[
                ("a", 0, 0.1649484373009503),
                ("a", 1, -0.09059659755906703),
                ("a", 2, -0.02002707684973299),
                ("a", 3, -0.157492639843249),
                ("a100", 0, 0.3803512615703158),
                ("a100", 1, 0.15210283677726458),
                ("a100", 2, -0.01988843318464118),
                ("a100", 3, 0.15481065611155675),
                ("d50", 0, -0.9722874611469743),
                ("d50", 1, -1.2923273586534567),
                ("d50", 2, -1.213093370446729),
                ("d50", 3, -1.2458837969621697),
                ("d75", 0, -0.24697800660979868),
                ("d75", 1, -0.40395709176451444),
                ("d75", 2, -0.4513148544171166),
                ("d75", 3, -0.8242693072635527),
                ("i", 0, -2.2407271799713833),
                ("i", 1, -2.7360831871059577),
                ("i", 2, -3.084612095767751),
                ("i", 3, -2.9672959917592077),
                ("a_back", 0, 0.2016153603935348),
                ("a_back", 1, 0.2190704135678721),
                ("a_back", 2, -0.134861509383363),
                ("a_back", 3, -0.37887865559997386),
            ])
        );
        assert_eq!(
            fit.back_slopes,
            cells(&[
                ("a_back", 0, 1.0302162754422977),
                ("a_back", 1, 0.4540067344299013),
                ("a_back", 2, 0.9126070906778803),
                ("a_back", 3, 0.9139223237878391),
            ])
        );
        assert!(fit.slopes.is_empty());

        let fit = fit_mapping(synthetic_rows(), &CARRIED, false, false).unwrap();
        assert_eq!(fit.slope, 0.9387090509896416);
        assert_eq!(
            fit.offsets,
            cells(&[
                ("a", 0, 0.18743939672447246),
                ("a", 1, 0.15171971297293105),
                ("a", 2, -0.10182310943405913),
                ("a", 3, -0.3417042555636646),
                ("a100", 0, 0.37984365449891666),
                ("a100", 1, 0.15194706756354742),
                ("a100", 2, -0.01998214088830841),
                ("a100", 3, 0.15437879071278857),
                ("d50", 0, -0.9690158265365397),
                ("d50", 1, -1.2887063123536566),
                ("d50", 2, -1.2095162698926931),
                ("d50", 3, -1.2422517884697393),
                ("d75", 0, -0.24696296954591707),
                ("d75", 1, -0.4023741127164286),
                ("d75", 2, -0.44939420969691246),
                ("d75", 3, -0.8209921713120373),
                ("i", 0, -2.234917063730278),
                ("i", 1, -2.7287290478957225),
                ("i", 2, -3.077344389746442),
                ("i", 3, -2.959866626845373),
            ])
        );
        assert!(fit.back_slopes.is_empty());
        assert!(fit.slopes.is_empty());
        assert!(fit.to_json().get("slopes").is_none());
    }

    #[test]
    fn category_slopes_one_per_seen_category() {
        let fit = fit_mapping(synthetic_rows(), &CARRIED, true, true).unwrap();
        assert!(fit.n_rows >= MIN_ROWS);
        assert_eq!(fit.slope, 1.0);
        let seen: HashSet<&str> = fit.offsets.iter().map(|(c, _, _)| c.as_str()).collect();
        let names: Vec<&str> = fit.slopes.iter().map(|(c, _)| c.as_str()).collect();
        // Every category with rows, in `categories` order (back categories last); the
        // merged `a100_back` and the unused `d25`, `s`, `u`, `n` have none.
        assert_eq!(names, ["a", "a100", "d50", "d75", "i", "a_back"]);
        for (category, slope) in &fit.slopes {
            assert!(seen.contains(category.as_str()));
            assert!(slope.is_finite());
        }
        // The fitted slopes recover the drawn ones' order and sit within 0.15 of them (the
        // back rows were drawn with `a`'s slope; ~600 rows per flagged category, prior
        // around 1).
        let slope = |c: &str| fit.slopes.iter().find(|(k, _)| k == c).unwrap().1;
        let order = ["i", "d50", "d75", "a100", "a"];
        for pair in order.windows(2) {
            assert!(slope(pair[0]) < slope(pair[1]), "{pair:?}");
        }
        for (category, drawn) in
            [("i", 0.4), ("d50", 0.6), ("d75", 0.7), ("a100", 0.9), ("a", 1.1), ("a_back", 1.1)]
        {
            assert!((slope(category) - drawn).abs() < 0.15, "{category}: {}", slope(category));
        }
        assert_eq!(fit.to_json()["slopes"][0][0], "a");
        // Too few rows: the identity, without slopes.
        let rows: Vec<FitRow> = synthetic_rows().into_iter().take(MIN_ROWS - 1).collect();
        let identity = fit_mapping(rows, &CARRIED, true, true).unwrap();
        assert!(identity.slopes.is_empty() && identity.offsets.is_empty());
    }

    /// Gradient against central differences of the loss value, Hessian against central
    /// differences of the gradient, at a non-trivial θ. The loss is O(1) (per row) and the
    /// gradient and Hessian O(0.01–1): with step 1e-5 the truncation error is O(h²) ≈ 1e-10
    /// and the rounding error ≈ ε·|f|/h ≈ 1e-11, so 1e-6 (absolute, or relative above 1)
    /// leaves four orders of margin while catching any missing or misplaced term.
    #[test]
    fn objective_derivatives_match_finite_differences() {
        const STEP: f64 = 1e-5;
        const TOL: f64 = 1e-6;
        for category_slopes in [true, false] {
            let mut rows = synthetic_rows();
            let objective = Objective::new(&mut rows, true, category_slopes);
            let (n_s, n_cells) = (objective.n_s, objective.n_cells);
            let n_theta = objective.n_theta();
            assert!(objective.n_back > 0);
            let theta: Vec<f64> = (0..n_theta)
                .map(|j| {
                    let t = ((j * 7) % 11) as f64 / 10.0 - 0.5; // −0.5..0.5
                    if j < n_s {
                        1.0 + 0.4 * t // slopes 0.8..1.2
                    } else if j < n_s + n_cells {
                        t // offsets ±0.5
                    } else {
                        0.6 * t // back slopes ±0.3
                    }
                })
                .collect();
            let (_, grad) = objective.loss(&theta);
            let hessian = objective.hessian(&theta);
            let close = |a: f64, b: f64| (a - b).abs() <= TOL * b.abs().max(1.0);
            let (mut max_grad, mut max_hess) = (0.0f64, 0.0f64);
            for j in 0..n_theta {
                let (mut up, mut down) = (theta.clone(), theta.clone());
                up[j] += STEP;
                down[j] -= STEP;
                let ((f_up, g_up), (f_down, g_down)) = (objective.loss(&up), objective.loss(&down));
                let fd = (f_up - f_down) / (2.0 * STEP);
                max_grad = max_grad.max((fd - grad[j]).abs());
                assert!(close(grad[j], fd), "grad[{j}] {} vs {fd}", grad[j]);
                for i in 0..n_theta {
                    let fd = (g_up[i] - g_down[i]) / (2.0 * STEP);
                    let h = hessian.at(i, j);
                    max_hess = max_hess.max((fd - h).abs());
                    assert!(
                        close(h, fd),
                        "H[{i},{j}] {h} vs {fd} (category_slopes {category_slopes})"
                    );
                }
            }
            println!(
                "category_slopes {category_slopes}: max |FD − grad| {max_grad:e}, max |FD − H| {max_hess:e}"
            );
        }
    }

    #[test]
    fn slope_lookup_falls_back_to_plain_then_shared() {
        let fit = AvailabilityFit {
            slope: 0.7,
            slopes: vec![("a".into(), 0.95), ("d50".into(), 0.6)],
            ..identity_fit()
        };
        assert_eq!(slope_for(&fit, "a"), 0.95);
        assert_eq!(slope_for(&fit, "a_back"), 0.95);
        assert_eq!(slope_for(&fit, "i"), 0.7);
        let shared = AvailabilityFit { slope: 0.7, ..identity_fit() };
        assert_eq!(slope_for(&shared, "a"), 0.7);
    }
}
