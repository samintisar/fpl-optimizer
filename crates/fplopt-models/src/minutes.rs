//! The minutes model (Python `fplopt.models.minutes`): a LightGBM hurdle model with
//! post-processing.
//!
//! **Parts** (each a `gbm::train` fit on `history::training_frame` rows, features
//! `booster_features`: `LAG_FEATURES`, then `ELO_FEATURES` with `elo_features`): P(start),
//! P(60+ | start), P(sub appearance | no start); expected minutes from three weighted-mean
//! constants. Training rows: every visible row outside a ban, weighted
//! `season_decay ** age`; a part with fewer than `MIN_ROWS` rows (or one label value)
//! predicts its weighted label mean.
//!
//! **Post-processing**: horizon decay toward the long-run start rate; suspensions (p_start =
//! p_sub = 0, or the ban residual rates); team normalization per (fixture, team): a bounded
//! logit shift so Σ p_start = 11, then a bounded scale so Σ e_minutes = 990, each capped at
//! 90 · p_play.

use std::collections::{HashMap, HashSet};

use anyhow::Result;
use fplopt_features::bans::{BAN_CERTAIN, BAN_LIKELY};
use fplopt_features::history::{
    ELO_FEATURES, LAG_FEATURES, horizon_elo, match_elo, prediction_frame, training_frame,
};
use fplopt_features::util::floats;
use fplopt_store::{AsOfView, Column, Table};

use crate::gbm::{Gbm, GbmParams, float_column, train};
use crate::util::{expit, logit, np_clip, np_minimum, pd_sum};

pub const MIN_ROWS: usize = 1000;
pub const DEFAULT_RATE: f64 = 0.5;
pub const TEAM_STARTERS: f64 = 11.0;
pub const TEAM_MINUTES: f64 = 990.0;
pub const MAX_MINUTES: f64 = 90.0;
const EPS: f64 = 1e-6;
const BISECTIONS: usize = 60;

#[derive(Clone, Debug, PartialEq)]
pub struct MinutesParams {
    pub start: GbmParams,
    pub sixty: GbmParams,
    pub sub: GbmParams,
    pub season_decay: f64,
    pub horizon_decay: f64,
    pub long_run_prior: f64,
    pub max_shift: f64,
    pub max_scale: f64,
    pub ban_residual: bool,
    /// Append `history::ELO_FEATURES` to the lag features (v1r).
    pub elo_features: bool,
    /// Logit shrink of P(start) by `horizon_shrink ** h` after the decay (1: off; v1r 0.92).
    pub horizon_shrink: f64,
}

impl Default for MinutesParams {
    fn default() -> Self {
        MinutesParams {
            start: GbmParams::default(),
            sixty: GbmParams::default(),
            sub: GbmParams::default(),
            season_decay: 0.7,
            horizon_decay: 0.85,
            long_run_prior: 5.0,
            max_shift: 1.0,
            max_scale: 1.25,
            ban_residual: false,
            elo_features: false,
            horizon_shrink: 1.0,
        }
    }
}

/// P(start) at a horizon: the decay toward `long_run`, then logit x `horizon_shrink ** h`
/// (post-processing step 1; Python `horizon_start`).
pub fn horizon_start(p_model: f64, long_run: f64, horizon: f64, params: &MinutesParams) -> f64 {
    let weight = params.horizon_decay.powf(horizon);
    let p_start = weight * p_model + (1.0 - weight) * long_run;
    if params.horizon_shrink == 1.0 {
        return p_start;
    }
    expit(params.horizon_shrink.powf(horizon) * logit(np_clip(p_start, EPS, 1.0 - EPS)))
}

/// The boosters' features: `LAG_FEATURES`, then `ELO_FEATURES` with `elo_features`.
pub fn booster_features(params: &MinutesParams) -> Vec<&'static str> {
    let mut out = LAG_FEATURES.to_vec();
    if params.elo_features {
        out.extend_from_slice(ELO_FEATURES);
    }
    out
}

/// `table` (rows with team_key, fixture_key) with the Elo columns from `elo` (NaN when a
/// key is missing); unchanged without `elo_features`.
fn with_elo(
    table: Table,
    params: &MinutesParams,
    elo: impl FnOnce() -> Result<HashMap<(i64, i64), (f64, f64)>>,
) -> Result<Table> {
    if !params.elo_features {
        return Ok(table);
    }
    let elo = elo()?;
    let (team, fixture) = (table.i64("team_key")?, table.i64("fixture_key")?);
    let get = |i: usize| elo.get(&(team[i], fixture[i])).copied().unwrap_or((f64::NAN, f64::NAN));
    let (we, rating): (Vec<f64>, Vec<f64>) = (0..table.height()).map(get).unzip();
    Ok(table.with("win_expectancy", floats(we)).with("club_rating", floats(rating)))
}

/// The minutes model's training frame: `training_frame` plus the Elo columns if asked
/// (shared with the availability fit, like Python's `history`).
pub fn history_frame(view: &AsOfView, params: &MinutesParams) -> Result<Table> {
    with_elo(training_frame(view)?, params, || match_elo(view))
}

/// A fitted part: a booster, or its weighted label mean on too few rows.
pub enum Part {
    Gbm(Gbm),
    Constant(f64),
}

impl Part {
    pub fn predict(&self, table: &Table) -> Result<Vec<f64>> {
        match self {
            Part::Gbm(gbm) => gbm.predict(table),
            Part::Constant(rate) => Ok(vec![*rate; table.height()]),
        }
    }
}

pub struct MinutesFit {
    pub start: Part,
    pub sixty: Part,
    pub sub: Part,
    pub minutes_sixty: f64,
    pub minutes_short: f64,
    pub minutes_sub: f64,
    pub params: MinutesParams,
    pub n_rows: usize,
    pub ban_start: f64,
    pub ban_sub: f64,
    pub ban_start_likely: f64,
    pub ban_sub_likely: f64,
}

fn weighted_mean(values: &[f64], weights: &[f64], default: f64) -> f64 {
    let total = pd_sum(weights);
    if total > 0.0 {
        let products: Vec<f64> = values.iter().zip(weights).map(|(v, w)| v * w).collect();
        pd_sum(&products) / total
    } else {
        default
    }
}

/// Distinct non-null values (`Series.nunique()`).
fn nunique(values: &[f64]) -> usize {
    values.iter().filter(|v| !v.is_nan()).map(|v| v.to_bits()).collect::<HashSet<_>>().len()
}

/// `_part`: a booster on `rows` (a training frame subset, its weights), or a constant.
pub(crate) fn part(
    table: &Table,
    weight: &[f64],
    label: &str,
    params: &GbmParams,
    features: &[&'static str],
) -> Result<Part> {
    let labels = float_column(table, label)?;
    if table.height() < MIN_ROWS || nunique(&labels) < 2 {
        return Ok(Part::Constant(weighted_mean(&labels, weight, DEFAULT_RATE)));
    }
    let with_weight = table.clone().with("weight", Column::F64(weight.to_vec()));
    Ok(Part::Gbm(train(&with_weight, features, label, Some("weight"), params)?))
}

/// `_weighted`: season_decay ** max(season − row season, 0).
pub(crate) fn season_weights(
    table: &Table,
    season: i64,
    params: &MinutesParams,
) -> Result<Vec<f64>> {
    Ok(float_column(table, "season")?
        .iter()
        .map(|s| params.season_decay.powf((season as f64 - s).max(0.0)))
        .collect())
}

fn subset(table: &Table, weight: &[f64], keep: impl Fn(usize) -> bool) -> (Table, Vec<f64>) {
    let rows: Vec<usize> = (0..table.height()).filter(|&i| keep(i)).collect();
    (table.take(&rows), rows.iter().map(|&i| weight[i]).collect())
}

/// (P(start), P(sub | no start)) of banned training rows, weighted (0 without rows).
fn ban_rates(table: &Table, weight: &[f64]) -> Result<(f64, f64)> {
    let start = float_column(table, "start")?;
    let (benched, bw) = subset(table, weight, |i| start[i] == 0.0);
    Ok((
        weighted_mean(&start, weight, 0.0),
        weighted_mean(&float_column(&benched, "sub")?, &bw, 0.0),
    ))
}

/// Fit the hurdle model on every row visible in `view` (`rows`: its training frame if the
/// caller already built it).
pub fn fit_minutes(
    view: &AsOfView,
    params: &MinutesParams,
    rows: Option<&Table>,
) -> Result<MinutesFit> {
    let (season, _) = view.gameweek_for_deadline()?;
    let built;
    let all = match rows {
        Some(rows) => rows,
        None => {
            built = history_frame(view, params)?;
            &built
        }
    };
    let weight = season_weights(all, season, params)?;
    let banned_col = float_column(all, "banned")?;
    let mut residuals: HashMap<i64, (f64, f64)> =
        HashMap::from([(BAN_CERTAIN, (0.0, 0.0)), (BAN_LIKELY, (0.0, 0.0))]);
    if params.ban_residual {
        let kind = float_column(all, "ban_kind")?;
        for k in [BAN_CERTAIN, BAN_LIKELY] {
            let (t, w) = subset(all, &weight, |i| banned_col[i] != 0.0 && kind[i] == k as f64);
            residuals.insert(k, ban_rates(&t, &w)?);
        }
    }
    let (rows, weight) = subset(all, &weight, |i| banned_col[i] == 0.0);
    let start = float_column(&rows, "start")?;
    let (started, sw) = subset(&rows, &weight, |i| start[i] == 1.0);
    let (benched, bw) = subset(&rows, &weight, |i| start[i] == 0.0);
    let sixty = float_column(&started, "sixty")?;
    let started_minutes = float_column(&started, "minutes")?;
    let pick =
        |values: &[f64], weights: &[f64], keep: &dyn Fn(usize) -> bool| -> (Vec<f64>, Vec<f64>) {
            (0..values.len()).filter(|&i| keep(i)).map(|i| (values[i], weights[i])).unzip()
        };
    let (long_m, long_w) = pick(&started_minutes, &sw, &|i| sixty[i] == 1.0);
    let (short_m, short_w) = pick(&started_minutes, &sw, &|i| sixty[i] != 1.0);
    let sub = float_column(&benched, "sub")?;
    let (sub_m, sub_w) = pick(&float_column(&benched, "minutes")?, &bw, &|i| sub[i] == 1.0);
    let features = booster_features(params);
    Ok(MinutesFit {
        start: part(&rows, &weight, "start", &params.start, &features)?,
        sixty: part(&started, &sw, "sixty", &params.sixty, &features)?,
        sub: part(&benched, &bw, "sub", &params.sub, &features)?,
        minutes_sixty: weighted_mean(&long_m, &long_w, 87.0),
        minutes_short: weighted_mean(&short_m, &short_w, 45.0),
        minutes_sub: weighted_mean(&sub_m, &sub_w, 18.0),
        params: params.clone(),
        n_rows: rows.height(),
        ban_start: residuals[&BAN_CERTAIN].0,
        ban_sub: residuals[&BAN_CERTAIN].1,
        ban_start_likely: residuals[&BAN_LIKELY].0,
        ban_sub_likely: residuals[&BAN_LIKELY].1,
    })
}

/// The minutes frame (Python's `MINUTES_COLUMNS`), column by column.
#[derive(Clone, Debug, Default)]
pub struct Minutes {
    pub player_key: Vec<i64>,
    pub fixture_key: Vec<i64>,
    pub team_key: Vec<i64>,
    pub season: Vec<i64>,
    pub gw: Vec<i64>,
    pub gw_index: Vec<i64>,
    pub horizon: Vec<i64>,
    pub p_start: Vec<f64>,
    pub p_60: Vec<f64>,
    pub p_sub: Vec<f64>,
    pub e_minutes: Vec<f64>,
    pub p_play: Vec<f64>,
    pub banned: Vec<bool>,
}

impl Minutes {
    pub fn len(&self) -> usize {
        self.player_key.len()
    }

    pub fn is_empty(&self) -> bool {
        self.player_key.is_empty()
    }

    pub fn to_table(&self) -> Table {
        Table::new()
            .with("player_key", Column::I64(self.player_key.clone()))
            .with("fixture_key", Column::I64(self.fixture_key.clone()))
            .with("team_key", Column::I64(self.team_key.clone()))
            .with("season", Column::I64(self.season.clone()))
            .with("gw", Column::I64(self.gw.clone()))
            .with("gw_index", Column::I64(self.gw_index.clone()))
            .with("horizon", Column::I64(self.horizon.clone()))
            .with("p_start", Column::F64(self.p_start.clone()))
            .with("p_60", Column::F64(self.p_60.clone()))
            .with("p_sub", Column::F64(self.p_sub.clone()))
            .with("e_minutes", Column::F64(self.e_minutes.clone()))
            .with("p_play", Column::F64(self.p_play.clone()))
            .with("banned", Column::Bool(self.banned.clone()))
    }
}

/// The keys of a minutes row (`KEYS`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Key {
    pub player_key: i64,
    pub fixture_key: i64,
    pub team_key: i64,
    pub season: i64,
    pub gw: i64,
    pub gw_index: i64,
    pub horizon: i64,
}

/// `finish_minutes`' input per row (`p_start`, `c60`, `csub`, `m_start`, `fixed`, `banned`).
#[derive(Clone, Debug, Default)]
pub struct Parts {
    pub keys: Vec<Key>,
    pub p_start: Vec<f64>,
    pub c60: Vec<f64>,
    pub csub: Vec<f64>,
    pub m_start: Vec<f64>,
    pub fixed: Vec<bool>,
    pub banned: Vec<bool>,
}

/// Group codes in order of first appearance (`pd.factorize`) and their count.
fn factorize(groups: &[i64]) -> (Vec<usize>, usize) {
    let mut index: HashMap<i64, usize> = HashMap::new();
    let codes = groups
        .iter()
        .map(|g| {
            let next = index.len();
            *index.entry(*g).or_insert(next)
        })
        .collect();
    (codes, index.len())
}

/// `np.bincount(codes, weights)`: per group the sum in row order.
fn bincount(codes: &[usize], weights: impl Iterator<Item = f64>, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n];
    for (&c, w) in codes.iter().zip(weights) {
        out[c] += w;
    }
    out
}

/// Per group the logit shift δ ∈ [−bound, bound] with Σ_free expit(logit(p) + δ) + Σ_fixed p
/// closest to `target` (bisection); returns the new p (fixed rows unchanged).
fn shift_to_sum(p: &[f64], groups: &[i64], free: &[bool], target: f64, bound: f64) -> Vec<f64> {
    let (codes, n) = factorize(groups);
    let z: Vec<f64> = p.iter().map(|&v| logit(np_clip(v, EPS, 1.0 - EPS))).collect();
    let fixed_sum = bincount(&codes, (0..p.len()).map(|i| if free[i] { 0.0 } else { p[i] }), n);
    let (mut lo, mut hi) = (vec![-bound; n], vec![bound; n]);
    for _ in 0..BISECTIONS {
        let mid: Vec<f64> = (0..n).map(|g| (lo[g] + hi[g]) / 2.0).collect();
        let q = (0..p.len()).map(|i| if free[i] { expit(z[i] + mid[codes[i]]) } else { 0.0 });
        let total = bincount(&codes, q, n);
        for g in 0..n {
            if total[g] + fixed_sum[g] > target {
                hi[g] = mid[g];
            } else {
                lo[g] = mid[g];
            }
        }
    }
    let delta: Vec<f64> = (0..n).map(|g| (lo[g] + hi[g]) / 2.0).collect();
    (0..p.len()).map(|i| if free[i] { expit(z[i] + delta[codes[i]]) } else { p[i] }).collect()
}

/// Team normalization and the output columns, sorted by (player_key, horizon, fixture_key).
pub fn finish_minutes(parts: &Parts, minutes_sub: f64, max_shift: f64, max_scale: f64) -> Minutes {
    let n = parts.keys.len();
    let groups: Vec<i64> =
        parts.keys.iter().map(|k| k.fixture_key * 1_000_000 + k.team_key).collect();
    let free: Vec<bool> = parts.fixed.iter().map(|f| !f).collect();
    let p_start = shift_to_sum(&parts.p_start, &groups, &free, TEAM_STARTERS, max_shift);
    let p_sub: Vec<f64> = (0..n).map(|i| (1.0 - p_start[i]) * parts.csub[i]).collect();
    let p_play: Vec<f64> = (0..n).map(|i| p_start[i] + p_sub[i]).collect();
    let e_minutes: Vec<f64> =
        (0..n).map(|i| p_start[i] * parts.m_start[i] + p_sub[i] * minutes_sub).collect();
    let (codes, n_groups) = factorize(&groups);
    let total = bincount(&codes, e_minutes.iter().copied(), n_groups);
    let scale: Vec<f64> = total
        .iter()
        .map(|&t| np_clip(if t > 0.0 { TEAM_MINUTES / t } else { 1.0 }, 1.0 / max_scale, max_scale))
        .collect();
    let e_minutes: Vec<f64> = (0..n)
        .map(|i| np_minimum(e_minutes[i] * scale[codes[i]], MAX_MINUTES * p_play[i]))
        .collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| {
        (parts.keys[i].player_key, parts.keys[i].horizon, parts.keys[i].fixture_key)
    });
    let mut out = Minutes::default();
    for i in order {
        let k = parts.keys[i];
        out.player_key.push(k.player_key);
        out.fixture_key.push(k.fixture_key);
        out.team_key.push(k.team_key);
        out.season.push(k.season);
        out.gw.push(k.gw);
        out.gw_index.push(k.gw_index);
        out.horizon.push(k.horizon);
        out.p_start.push(p_start[i]);
        out.p_60.push(p_start[i] * parts.c60[i]);
        out.p_sub.push(p_sub[i]);
        out.e_minutes.push(e_minutes[i]);
        out.p_play.push(p_play[i]);
        out.banned.push(parts.banned[i]);
    }
    out
}

/// The inverse of `finish_minutes`' last step: per row p_start, c60, csub, m_start (0 where
/// p_start is 0; 1 / 0 for c60 / csub when undefined), banned, and fixed (= banned).
pub fn conditionals(minutes: &Minutes, minutes_sub: f64) -> Parts {
    let n = minutes.len();
    let mut parts = Parts::default();
    for i in 0..n {
        let (p_start, p_sub) = (minutes.p_start[i], minutes.p_sub[i]);
        let started = p_start > 0.0;
        let c60 = if started { minutes.p_60[i] / p_start } else { 1.0 };
        let csub = if p_start < 1.0 { p_sub / (1.0 - p_start) } else { 0.0 };
        let start_minutes = minutes.e_minutes[i] - p_sub * minutes_sub;
        let m_start = if started { start_minutes / p_start } else { 0.0 };
        parts.keys.push(Key {
            player_key: minutes.player_key[i],
            fixture_key: minutes.fixture_key[i],
            team_key: minutes.team_key[i],
            season: minutes.season[i],
            gw: minutes.gw[i],
            gw_index: minutes.gw_index[i],
            horizon: minutes.horizon[i],
        });
        parts.p_start.push(p_start);
        parts.c60.push(np_clip(c60, 0.0, 1.0));
        parts.csub.push(np_clip(csub, 0.0, 1.0));
        parts.m_start.push(np_clip(m_start, 0.0, MAX_MINUTES));
        parts.banned.push(minutes.banned[i]);
        parts.fixed.push(minutes.banned[i]);
    }
    parts
}

/// The keys of a prediction (or training) frame's rows.
pub(crate) fn keys(frame: &Table) -> Result<Vec<Key>> {
    let (p, f, t, s, g, gi, h) = (
        float_column(frame, "player_key")?,
        float_column(frame, "fixture_key")?,
        float_column(frame, "team_key")?,
        float_column(frame, "season")?,
        float_column(frame, "gw")?,
        float_column(frame, "gw_index")?,
        float_column(frame, "horizon")?,
    );
    Ok((0..frame.height())
        .map(|i| Key {
            player_key: p[i] as i64,
            fixture_key: f[i] as i64,
            team_key: t[i] as i64,
            season: s[i] as i64,
            gw: g[i] as i64,
            gw_index: gi[i] as i64,
            horizon: h[i] as i64,
        })
        .collect())
}

/// Per pool player and horizon fixture: the minutes frame.
pub fn predict_minutes(view: &AsOfView, fit: &MinutesFit) -> Result<Minutes> {
    let params = &fit.params;
    let frame = with_elo(prediction_frame(view)?, params, || horizon_elo(view))?;
    let p_model = fit.start.predict(&frame)?;
    let c60 = fit.sixty.predict(&frame)?;
    let mut csub = fit.sub.predict(&frame)?;
    let n_long = float_column(&frame, "n_long")?;
    let rate_long = float_column(&frame, "start_rate_long")?;
    let horizon = float_column(&frame, "horizon")?;
    let kind = float_column(&frame, "ban_kind")?;
    let n = frame.height();
    let mut p_start = Vec::with_capacity(n);
    let mut banned = Vec::with_capacity(n);
    for i in 0..n {
        let starts_long = if rate_long[i].is_nan() { 0.0 } else { rate_long[i] } * n_long[i];
        let long_run = (starts_long + params.long_run_prior * p_model[i])
            / (n_long[i] + params.long_run_prior);
        let blended = horizon_start(p_model[i], long_run, horizon[i], params);
        let (certain, likely) = (kind[i] == BAN_CERTAIN as f64, kind[i] == BAN_LIKELY as f64);
        p_start.push(if certain {
            fit.ban_start
        } else if likely {
            fit.ban_start_likely
        } else {
            blended
        });
        if certain {
            csub[i] = fit.ban_sub;
        } else if likely {
            csub[i] = fit.ban_sub_likely;
        }
        banned.push(certain || likely);
    }
    let m_start: Vec<f64> =
        c60.iter().map(|c| c * fit.minutes_sixty + (1.0 - c) * fit.minutes_short).collect();
    let parts =
        Parts { keys: keys(&frame)?, p_start, c60, csub, m_start, fixed: banned.clone(), banned };
    Ok(finish_minutes(&parts, fit.minutes_sub, params.max_shift, params.max_scale))
}

impl Part {
    fn to_json(&self) -> Result<serde_json::Value> {
        Ok(match self {
            Part::Gbm(gbm) => serde_json::json!({"model": gbm.booster.model_string()?}),
            Part::Constant(rate) => serde_json::json!({"rate": rate}),
        })
    }
}

impl MinutesFit {
    /// The fitted values (boosters as their model text).
    pub fn to_json(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "start": self.start.to_json()?,
            "sixty": self.sixty.to_json()?,
            "sub": self.sub.to_json()?,
            "minutes_sixty": self.minutes_sixty,
            "minutes_short": self.minutes_short,
            "minutes_sub": self.minutes_sub,
            "n_rows": self.n_rows,
            "ban_start": self.ban_start,
            "ban_sub": self.ban_sub,
            "ban_start_likely": self.ban_start_likely,
            "ban_sub_likely": self.ban_sub_likely,
        }))
    }
}

#[cfg(test)]
mod tests {
    use fplopt_features::history::ELO_FEATURES;

    use super::*;

    #[test]
    fn horizon_start_decays_then_shrinks() {
        let off = MinutesParams { horizon_decay: 0.5, ..MinutesParams::default() };
        let on = MinutesParams { horizon_shrink: 0.8, ..off.clone() };
        for (p, long, h) in [(0.9, 0.6, 0.0), (0.9, 0.6, 1.0), (0.3, 0.5, 3.0)] {
            let w = 0.5f64.powf(h);
            let blended = w * p + (1.0 - w) * long;
            assert_eq!(horizon_start(p, long, h, &off), blended);
            let z = logit(np_clip(blended, EPS, 1.0 - EPS));
            let shrunk = expit(0.8f64.powf(h) * z);
            assert!((horizon_start(p, long, h, &on) - shrunk).abs() < 1e-15);
        }
        // h = 0 is not moved by the shrink beyond the clip round trip.
        assert!((horizon_start(0.9, 0.6, 0.0, &on) - 0.9).abs() < 1e-12);
    }

    #[test]
    fn booster_features_add_elo_only_when_asked() {
        let off = MinutesParams::default();
        let on = MinutesParams { elo_features: true, ..MinutesParams::default() };
        assert_eq!(booster_features(&off), LAG_FEATURES.to_vec());
        let with = booster_features(&on);
        assert_eq!(&with[..LAG_FEATURES.len()], LAG_FEATURES);
        assert_eq!(&with[LAG_FEATURES.len()..], ELO_FEATURES);
    }

    #[test]
    fn with_elo_appends_the_columns_or_nan() {
        let table = Table::new()
            .with("team_key", Column::I64(vec![1, 2]))
            .with("fixture_key", Column::I64(vec![10, 10]));
        let elo = HashMap::from([((1, 10), (0.6, 1500.0))]);
        let off = with_elo(table.clone(), &MinutesParams::default(), || unreachable!()).unwrap();
        assert_eq!(off, table);
        let on = MinutesParams { elo_features: true, ..MinutesParams::default() };
        let out = with_elo(table, &on, || Ok(elo)).unwrap();
        let we = float_column(&out, "win_expectancy").unwrap();
        let rating = float_column(&out, "club_rating").unwrap();
        assert_eq!((we[0], rating[0]), (0.6, 1500.0));
        assert!(we[1].is_nan() && rating[1].is_nan());
    }
}
