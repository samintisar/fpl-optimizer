//! Gate N (`docs/superpowers/specs/2026-11-09-move-timing-design.md`, "The nightly model"):
//! walk-forward evaluation of the night price model (`research night-eval`).
//!
//! For each evaluated season S (develop only: 2023/24–2024/25 are kept clean and the holdout
//! is never read) the model and the baselines are fitted on the nights of first_fit..S−1 at
//! S's first night (labels before it), then score every night of S whose label is known
//! (from a view after the season: evaluation only; unlabelled rows, NaN `rise`, are left
//! out). Baselines: B0 the training base rate of each target; B1 the same boosters on
//! `net_per_own` alone (a threshold rule on net transfers per owner). The model and B1 are
//! Platt-recalibrated (`fit_calibrated`: maps fitted on out-of-fold predictions over the
//! training rows; spec's "Calibration", amended 2026-11-09); B0 stays the raw base rate.
//! Metrics per target: log loss of the calibrated model and both baselines (and of the raw
//! model and B1), one-sided Diebold–Mariano tests of the calibrated model against each on
//! per-night mean log loss (lag 7, clustered by night), calibration tables (10 bins of the
//! model's calibrated and raw probability) and the calibration condition (among rows with
//! calibrated p ≥ 0.3: n ≥ 30 and rate ≥ mean p − 0.10). Gate N passes when the model beats
//! both baselines on both targets with p < 0.10 (a test without a p, e.g. on identical
//! losses, does not pass) and both calibration conditions hold.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, bail};
use fplopt_core::pyjson::Ordered;
use fplopt_core::seasons::{HOLDOUT_SEASONS, season_label};
use fplopt_models::night::{
    FEATURES, NightParams, Platt, fit_calibrated, nights, predict_calibrated, training_rows,
};
use fplopt_store::{Column, DataStore, Micros, Table};

use crate::eval::diebold_mariano;

/// Nights' log losses overlap for about a week (the counter carries over nights). The lag
/// counts nights that have labelled rows (the DM clusters), not calendar nights: a night
/// without a labelled row (no morning snapshot, a stale evening price) is no cluster, so
/// across gaps in the snapshots 7 lags span more than 7 calendar nights.
pub const DM_LAG: usize = 7;
/// The gate's one-sided level.
pub const ALPHA: f64 = 0.10;
/// The last develop season with snapshots; later seasons are kept clean (spec, Measurement).
pub const LAST_DEVELOP: i64 = 2022;
/// Snapshots start in April 2021 (2020/21's last nights).
pub const FIRST_FIT_MIN: i64 = 2020;
/// The feature of baseline B1.
pub const B1_FEATURES: [&str; 1] = ["net_per_own"];
/// The calibration condition (per target, the model's calibrated probabilities): among the
/// scored rows with p ≥ `CAL_P_MIN` (where policy T can act), at least `CAL_MIN_N` rows and
/// an observed rate ≥ their mean p − `CAL_TOL`.
pub const CAL_P_MIN: f64 = 0.3;
pub const CAL_MIN_N: usize = 30;
pub const CAL_TOL: f64 = 0.10;
const EPS: f64 = 1e-15;

/// One scored player-night: the truths and each arm's probabilities.
#[derive(Clone, Debug, PartialEq)]
pub struct NightRow {
    pub season: i64,
    pub night: Micros,
    pub player_key: i64,
    pub rise: f64,
    pub fall: f64,
    pub p_rise: f64,
    pub p_fall: f64,
    pub b0_rise: f64,
    pub b0_fall: f64,
    pub b1_rise: f64,
    pub b1_fall: f64,
    /// The boosters' own probabilities, before the Platt maps.
    pub raw_p_rise: f64,
    pub raw_p_fall: f64,
    pub raw_b1_rise: f64,
    pub raw_b1_fall: f64,
}

/// Refuses a `first_fit` before the snapshots, the holdout, seasons after `LAST_DEVELOP`
/// and seasons not after `first_fit` (S is fitted on first_fit..S−1).
pub fn check_seasons(seasons: &[i64], first_fit: i64) -> Result<()> {
    if first_fit < FIRST_FIT_MIN {
        bail!(
            "--first-fit {first_fit}: no snapshots before {}",
            season_label(FIRST_FIT_MIN as i32)
        );
    }
    for &s in seasons {
        if HOLDOUT_SEASONS.contains(&(s as i32)) {
            bail!("{} is the holdout season", season_label(s as i32));
        }
        if s > LAST_DEVELOP {
            bail!(
                "{} is kept clean until a frozen policy passes on develop (last develop season \
                 {})",
                season_label(s as i32),
                season_label(LAST_DEVELOP as i32)
            );
        }
        if s <= first_fit {
            bail!(
                "{} is not after --first-fit {}",
                season_label(s as i32),
                season_label(first_fit as i32)
            );
        }
    }
    Ok(())
}

/// Mean binary cross-entropy of (truth, probability) pairs, probabilities clipped to
/// [1e-15, 1 − 1e-15]; NaN for no pairs.
pub fn log_loss(pairs: &[(f64, f64)]) -> f64 {
    if pairs.is_empty() {
        return f64::NAN;
    }
    pairs
        .iter()
        .map(|&(y, p)| {
            let p = p.clamp(EPS, 1.0 - EPS);
            -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
        })
        .sum::<f64>()
        / pairs.len() as f64
}

/// The Platt maps of one season's fits: (target, arm (model or b1), map).
pub type SeasonPlatt = Vec<(&'static str, &'static str, Platt)>;

/// The scored rows of season `s`: fit on first_fit..s−1 at s's first night (the model and
/// B1 calibrated by `fit_calibrated`, B0 the raw base rate), predict every night of s with a
/// known label; with the season's Platt maps.
pub fn season_rows(
    store: &DataStore,
    s: i64,
    first_fit: i64,
    params: &NightParams,
) -> Result<(Vec<NightRow>, SeasonPlatt)> {
    check_seasons(&[s], first_fit)?;
    // the evaluated season's nights and labels from a view after the season: evaluation
    // only, never fitted on (check_seasons keeps s a develop season, never the holdout)
    let season_nights = nights(&store.as_of(Micros::MAX), s)?;
    let Some(first) = season_nights.first() else {
        bail!("no nights in {}", season_label(s as i32))
    };
    let fit_seasons: Vec<i64> = (first_fit..s).collect();
    let train_rows = training_rows(store, &fit_seasons, first.cutoff)?;
    let model = fit_calibrated(&train_rows, FEATURES, params)?;
    let b1 = fit_calibrated(&train_rows, &B1_FEATURES, params)?;
    let base = |c: &str| -> Result<f64> {
        let v: Vec<f64> = train_rows.f64(c)?.iter().copied().filter(|x| x.is_finite()).collect();
        Ok(if v.is_empty() { f64::NAN } else { v.iter().sum::<f64>() / v.len() as f64 })
    };
    let (b0_rise, b0_fall) = (base("rise")?, base("fall")?);
    let scored = training_rows(store, &[s], Micros::MAX)?;
    let p = predict_calibrated(&model, &scored)?;
    let q = predict_calibrated(&b1, &scored)?;
    let (night, keys) = (scored.i64("night")?, scored.i64("player_key")?);
    let (rise, fall) = (scored.f64("rise")?, scored.f64("fall")?);
    let (pr, pf) = (p.f64("p_rise")?, p.f64("p_fall")?);
    let (qr, qf) = (q.f64("p_rise")?, q.f64("p_fall")?);
    let (raw_pr, raw_pf) = (p.f64("raw_p_rise")?, p.f64("raw_p_fall")?);
    let (raw_qr, raw_qf) = (q.f64("raw_p_rise")?, q.f64("raw_p_fall")?);
    let platt = vec![
        ("rise", "model", model.rise),
        ("fall", "model", model.fall),
        ("rise", "b1", b1.rise),
        ("fall", "b1", b1.fall),
    ];
    let rows = (0..scored.height())
        .filter(|&i| rise[i].is_finite())
        .map(|i| NightRow {
            season: s,
            night: night[i],
            player_key: keys[i],
            rise: rise[i],
            fall: fall[i],
            p_rise: pr[i],
            p_fall: pf[i],
            b0_rise,
            b0_fall,
            b1_rise: qr[i],
            b1_fall: qf[i],
            raw_p_rise: raw_pr[i],
            raw_p_fall: raw_pf[i],
            raw_b1_rise: raw_qr[i],
            raw_b1_fall: raw_qf[i],
        })
        .collect();
    Ok((rows, platt))
}

/// One row of the gate: the model against a baseline on a target.
#[derive(Clone, Debug, PartialEq)]
pub struct GateRow {
    pub target: &'static str,
    pub baseline: &'static str,
    /// The one-sided DM p of the model being better (NaN: no test).
    pub p: f64,
    pub pass: bool,
}

/// One calibration bin: [lo, hi) of the probability (the last bin includes 1).
#[derive(Clone, Debug, PartialEq)]
pub struct Bin {
    pub lo: f64,
    pub hi: f64,
    pub n: usize,
    pub mean_p: f64,
    pub rate: f64,
}

/// 10 equal-width bins of the probability `p`, with the mean probability and the rate of
/// `y` in each (NaN in an empty bin).
pub fn calibration(
    rows: &[NightRow],
    p: impl Fn(&NightRow) -> f64,
    y: impl Fn(&NightRow) -> f64,
) -> Vec<Bin> {
    (0..10)
        .map(|b| {
            let (lo, hi) = (b as f64 / 10.0, (b + 1) as f64 / 10.0);
            let inside: Vec<&NightRow> = rows
                .iter()
                .filter(|r| {
                    let v = p(r);
                    v >= lo && (v < hi || (b == 9 && v <= hi))
                })
                .collect();
            let n = inside.len();
            let mean = |f: &dyn Fn(&NightRow) -> f64| {
                if n == 0 { f64::NAN } else { inside.iter().map(|r| f(r)).sum::<f64>() / n as f64 }
            };
            Bin { lo, hi, n, mean_p: mean(&|r| p(r)), rate: mean(&|r| y(r)) }
        })
        .collect()
}

/// The calibration condition on a target: the rows with calibrated p ≥ `CAL_P_MIN`, their
/// mean p and observed rate (NaN without rows), and whether it holds.
#[derive(Clone, Debug, PartialEq)]
pub struct CalGateRow {
    pub target: &'static str,
    pub n: usize,
    pub mean_p: f64,
    pub rate: f64,
    pub pass: bool,
}

/// The calibration condition over (truth, calibrated p) pairs: among those with
/// p ≥ `CAL_P_MIN`, n ≥ `CAL_MIN_N` and rate ≥ mean_p − `CAL_TOL`.
pub fn calibration_condition(target: &'static str, pairs: &[(f64, f64)]) -> CalGateRow {
    let high: Vec<(f64, f64)> = pairs.iter().copied().filter(|&(_, p)| p >= CAL_P_MIN).collect();
    let n = high.len();
    let mean = |k: fn(&(f64, f64)) -> f64| {
        if n == 0 { f64::NAN } else { high.iter().map(k).sum::<f64>() / n as f64 }
    };
    let (mean_p, rate) = (mean(|x| x.1), mean(|x| x.0));
    // NaN (no rows) fails the comparison
    let pass = n >= CAL_MIN_N && rate >= mean_p - CAL_TOL;
    CalGateRow { target, n, mean_p, rate, pass }
}

/// (truth, model, B0, B1) of a target.
type Pick = fn(&NightRow) -> (f64, f64, f64, f64);
/// (truth, raw model, raw B1) of a target.
type RawPick = fn(&NightRow) -> (f64, f64, f64);

const TARGETS: [(&str, Pick, RawPick); 2] = [
    (
        "rise",
        |r| (r.rise, r.p_rise, r.b0_rise, r.b1_rise),
        |r| (r.rise, r.raw_p_rise, r.raw_b1_rise),
    ),
    (
        "fall",
        |r| (r.fall, r.p_fall, r.b0_fall, r.b1_fall),
        |r| (r.fall, r.raw_p_fall, r.raw_b1_fall),
    ),
];
const BASELINES: [&str; 2] = ["b0", "b1"];

fn f(x: f64) -> Ordered {
    if x.is_finite() { Ordered::Float(x) } else { Ordered::Null }
}

fn int(x: usize) -> Ordered {
    Ordered::Int(x as i128)
}

fn text(s: &str) -> Ordered {
    Ordered::Str(s.to_owned())
}

fn rec(pairs: Vec<(&str, Ordered)>) -> Ordered {
    Ordered::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}

/// Per night (in night order) the model's and baseline `baseline`'s (0: B0, 1: B1) mean log
/// loss: the Diebold–Mariano input.
pub fn per_night(rows: &[NightRow], pick: Pick, baseline: usize) -> Vec<(Micros, f64, f64)> {
    let mut by: BTreeMap<Micros, (Vec<(f64, f64)>, Vec<(f64, f64)>)> = BTreeMap::new();
    for r in rows {
        let (y, p, b0, b1) = pick(r);
        let e = by.entry(r.night).or_default();
        e.0.push((y, p));
        e.1.push((y, if baseline == 0 { b0 } else { b1 }));
    }
    by.into_iter().map(|(n, (a, b))| (n, log_loss(&a), log_loss(&b))).collect()
}

/// The metrics object: `n_rows`, `n_nights`, `log_loss` (per target, calibrated: model, b0,
/// b1), `log_loss_raw` (per target, before the Platt maps: model, b1), `dm` (calibrated
/// model vs each baseline: target, baseline, value_a, value_b (mean per-night log loss), then
/// `diebold_mariano`'s mean_diff, t_stat, p_a_better, p_b_better, n_clusters, lag),
/// `calibration` and `calibration_raw` (the model's calibrated and raw p, per target and
/// bin: lo, hi, n, mean_p, rate), `gate` (per target and baseline: p, pass),
/// `calibration_gate` (per target: n, mean_p, rate, pass; `calibration_condition`) and
/// `gate_pass` (every DM gate and both calibration conditions).
pub fn night_metrics(rows: &[NightRow]) -> Ordered {
    let nights: BTreeSet<Micros> = rows.iter().map(|r| r.night).collect();
    let (mut losses, mut dm, mut cal, mut gate) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut raw_losses, mut cal_raw, mut cal_gate) = (Vec::new(), Vec::new(), Vec::new());
    let mut all_pass = true;
    for (target, pick, raw) in TARGETS {
        let arm = |k: usize| {
            let pairs: Vec<(f64, f64)> = rows
                .iter()
                .map(|r| {
                    let (y, p, b0, b1) = pick(r);
                    (y, [p, b0, b1][k])
                })
                .collect();
            log_loss(&pairs)
        };
        losses.push(rec(vec![
            ("target", text(target)),
            ("model", f(arm(0))),
            ("b0", f(arm(1))),
            ("b1", f(arm(2))),
        ]));
        let raw_arm = |k: usize| {
            let pairs: Vec<(f64, f64)> = rows
                .iter()
                .map(|r| {
                    let (y, p, b1) = raw(r);
                    (y, [p, b1][k])
                })
                .collect();
            log_loss(&pairs)
        };
        raw_losses.push(rec(vec![
            ("target", text(target)),
            ("model", f(raw_arm(0))),
            ("b1", f(raw_arm(1))),
        ]));
        for (baseline, name) in BASELINES.into_iter().enumerate() {
            let loss = per_night(rows, pick, baseline);
            let mean = |side: fn(&(Micros, f64, f64)) -> f64| {
                if loss.is_empty() {
                    f64::NAN
                } else {
                    loss.iter().map(side).sum::<f64>() / loss.len() as f64
                }
            };
            let test = diebold_mariano(&loss, DM_LAG);
            let p = test
                .iter()
                .find(|(k, _)| *k == "p_a_better")
                .and_then(|(_, v)| match v {
                    Ordered::Float(x) => Some(*x),
                    _ => None,
                })
                .unwrap_or(f64::NAN);
            // NaN (no test) is not < ALPHA
            let pass = p < ALPHA;
            all_pass &= pass;
            let mut pairs = vec![
                ("target", text(target)),
                ("baseline", text(name)),
                ("value_a", f(mean(|l| l.1))),
                ("value_b", f(mean(|l| l.2))),
            ];
            pairs.extend(test);
            dm.push(rec(pairs));
            gate.push(rec(vec![
                ("target", text(target)),
                ("baseline", text(name)),
                ("p", f(p)),
                ("pass", Ordered::Bool(pass)),
            ]));
        }
        let bin = |b: Bin| {
            rec(vec![
                ("target", text(target)),
                ("lo", f(b.lo)),
                ("hi", f(b.hi)),
                ("n", int(b.n)),
                ("mean_p", f(b.mean_p)),
                ("rate", f(b.rate)),
            ])
        };
        cal.extend(calibration(rows, |r| pick(r).1, |r| pick(r).0).into_iter().map(bin));
        cal_raw.extend(calibration(rows, |r| raw(r).1, |r| raw(r).0).into_iter().map(bin));
        let pairs: Vec<(f64, f64)> = rows.iter().map(|r| (pick(r).0, pick(r).1)).collect();
        let c = calibration_condition(target, &pairs);
        all_pass &= c.pass;
        cal_gate.push(rec(vec![
            ("target", text(target)),
            ("n", int(c.n)),
            ("mean_p", f(c.mean_p)),
            ("rate", f(c.rate)),
            ("pass", Ordered::Bool(c.pass)),
        ]));
    }
    rec(vec![
        ("n_rows", int(rows.len())),
        ("n_nights", int(nights.len())),
        ("log_loss", Ordered::Array(losses)),
        ("log_loss_raw", Ordered::Array(raw_losses)),
        ("dm", Ordered::Array(dm)),
        ("calibration", Ordered::Array(cal)),
        ("calibration_raw", Ordered::Array(cal_raw)),
        ("gate", Ordered::Array(gate)),
        ("calibration_gate", Ordered::Array(cal_gate)),
        ("gate_pass", Ordered::Bool(all_pass)),
    ])
}

/// The `gate` rows of `night_metrics`' object.
pub fn gate_of(metrics: &Ordered) -> Vec<GateRow> {
    let Ordered::Object(top) = metrics else { return vec![] };
    let Some((_, Ordered::Array(rows))) = top.iter().find(|(k, _)| k == "gate") else {
        return vec![];
    };
    rows.iter()
        .filter_map(|r| {
            let Ordered::Object(r) = r else { return None };
            let get = |k: &str| r.iter().find(|(n, _)| n == k).map(|(_, v)| v);
            let name = |k: &str, options: [&'static str; 2]| match get(k) {
                Some(Ordered::Str(s)) => options.into_iter().find(|o| o == s),
                _ => None,
            };
            let p = match get("p") {
                Some(Ordered::Float(x)) => *x,
                _ => f64::NAN,
            };
            Some(GateRow {
                target: name("target", ["rise", "fall"])?,
                baseline: name("baseline", BASELINES)?,
                p,
                pass: matches!(get("pass"), Some(Ordered::Bool(true))),
            })
        })
        .collect()
}

/// The `calibration_gate` rows of `night_metrics`' object.
pub fn calibration_gate_of(metrics: &Ordered) -> Vec<CalGateRow> {
    let Ordered::Object(top) = metrics else { return vec![] };
    let Some((_, Ordered::Array(rows))) = top.iter().find(|(k, _)| k == "calibration_gate") else {
        return vec![];
    };
    rows.iter()
        .filter_map(|r| {
            let Ordered::Object(r) = r else { return None };
            let get = |k: &str| r.iter().find(|(n, _)| n == k).map(|(_, v)| v);
            let num = |k: &str| match get(k) {
                Some(Ordered::Float(x)) => *x,
                Some(Ordered::Int(x)) => *x as f64,
                _ => f64::NAN,
            };
            let target = match get("target") {
                Some(Ordered::Str(s)) => ["rise", "fall"].into_iter().find(|o| o == s)?,
                _ => return None,
            };
            Some(CalGateRow {
                target,
                n: match get("n") {
                    Some(Ordered::Int(x)) => usize::try_from(*x).ok()?,
                    _ => return None,
                },
                mean_p: num("mean_p"),
                rate: num("rate"),
                pass: matches!(get("pass"), Some(Ordered::Bool(true))),
            })
        })
        .collect()
}

/// `predictions.parquet`: season, night (Ts), player_key, rise, fall, p_rise, p_fall,
/// b0_rise, b0_fall, b1_rise, b1_fall (calibrated, B0 raw), raw_p_rise, raw_p_fall,
/// raw_b1_rise, raw_b1_fall (before the Platt maps).
pub fn rows_table(rows: &[NightRow]) -> Table {
    let col = |pick: &dyn Fn(&NightRow) -> f64| Column::F64(rows.iter().map(pick).collect());
    Table::new()
        .with("season", Column::I64(rows.iter().map(|r| r.season).collect()))
        .with("night", Column::Ts(rows.iter().map(|r| r.night).collect()))
        .with("player_key", Column::I64(rows.iter().map(|r| r.player_key).collect()))
        .with("rise", col(&|r| r.rise))
        .with("fall", col(&|r| r.fall))
        .with("p_rise", col(&|r| r.p_rise))
        .with("p_fall", col(&|r| r.p_fall))
        .with("b0_rise", col(&|r| r.b0_rise))
        .with("b0_fall", col(&|r| r.b0_fall))
        .with("b1_rise", col(&|r| r.b1_rise))
        .with("b1_fall", col(&|r| r.b1_fall))
        .with("raw_p_rise", col(&|r| r.raw_p_rise))
        .with("raw_p_fall", col(&|r| r.raw_p_fall))
        .with("raw_b1_rise", col(&|r| r.raw_b1_rise))
        .with("raw_b1_fall", col(&|r| r.raw_b1_fall))
}

/// Fits and scores each of `seasons` (sorted, deduplicated) in order, then the metrics:
/// `first_fit`, `fit_seasons` (per season label), `nights` (scored nights per season label),
/// `platt` (per season, target and arm (model, b1): season, target, arm, a, b), then
/// `night_metrics`' keys.
pub fn evaluate_nights(
    store: &DataStore,
    seasons: &[i64],
    first_fit: i64,
    params: &NightParams,
) -> Result<(Vec<NightRow>, Ordered)> {
    let seasons: Vec<i64> = seasons.iter().copied().collect::<BTreeSet<_>>().into_iter().collect();
    check_seasons(&seasons, first_fit)?;
    let label = |s: i64| season_label(s as i32);
    let mut rows = Vec::new();
    let mut fitted = Vec::new();
    let mut per_season = Vec::new();
    let mut platt = Vec::new();
    for &s in &seasons {
        let (part, maps) = season_rows(store, s, first_fit, params)?;
        for (target, arm, m) in maps {
            platt.push(rec(vec![
                ("season", Ordered::Int(i128::from(s))),
                ("target", text(target)),
                ("arm", text(arm)),
                ("a", f(m.a)),
                ("b", f(m.b)),
            ]));
        }
        let n: BTreeSet<Micros> = part.iter().map(|r| r.night).collect();
        per_season.push((label(s), int(n.len())));
        fitted.push((
            label(s),
            Ordered::Array((first_fit..s).map(|x| Ordered::Int(i128::from(x))).collect()),
        ));
        rows.extend(part);
    }
    let mut metrics = vec![
        ("first_fit".to_owned(), Ordered::Int(i128::from(first_fit))),
        ("fit_seasons".to_owned(), Ordered::Object(fitted)),
        ("nights".to_owned(), Ordered::Object(per_season)),
        ("platt".to_owned(), Ordered::Array(platt)),
    ];
    if let Ordered::Object(pairs) = night_metrics(&rows) {
        metrics.extend(pairs);
    }
    Ok((rows, Ordered::Object(metrics)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(night: Micros, truth_rise: f64, p: f64, b0: f64, b1: f64) -> NightRow {
        NightRow {
            season: 2022,
            night,
            player_key: night,
            rise: truth_rise,
            fall: 0.0,
            p_rise: p,
            p_fall: 0.02,
            b0_rise: b0,
            b0_fall: 0.02,
            b1_rise: b1,
            b1_fall: 0.02,
            raw_p_rise: p,
            raw_p_fall: 0.02,
            raw_b1_rise: b1,
            raw_b1_fall: 0.02,
        }
    }

    fn gate_pass(m: &Ordered) -> bool {
        matches!(
            m,
            Ordered::Object(top)
                if top.iter().any(|(k, v)| k == "gate_pass" && *v == Ordered::Bool(true))
        )
    }

    /// `n` nights, one row each: the model better than both baselines on every night by a
    /// varying margin, on rise; on fall too when `both` (else the fall arms are all 0.02).
    fn beaten(n: i64, both: bool) -> Vec<NightRow> {
        (0..n)
            .map(|n| {
                let y = (n % 2) as f64;
                let sure = 0.9 - 0.05 * (n % 3) as f64;
                let p = if y == 1.0 { sure } else { 1.0 - sure };
                let r = row(n, y, p, 0.5, 0.4 + 0.1 * y);
                if !both {
                    return r;
                }
                NightRow {
                    fall: 1.0 - r.rise,
                    p_fall: 1.0 - r.p_rise,
                    b0_fall: 0.5,
                    b1_fall: 0.6 - 0.1 * r.rise,
                    raw_p_fall: 1.0 - r.p_rise,
                    raw_b1_fall: 0.6 - 0.1 * r.rise,
                    ..r
                }
            })
            .collect()
    }

    #[test]
    fn log_loss_is_the_mean_binary_cross_entropy() {
        let l = log_loss(&[(1.0, 0.8), (0.0, 0.2)]);
        assert!((l - (-(0.8f64).ln())).abs() < 1e-12);
        // clipped: a certain wrong prediction is finite
        assert!(log_loss(&[(1.0, 0.0)]).is_finite());
    }

    #[test]
    fn seasons_refused_outside_develop() {
        assert!(check_seasons(&[2022], 2020).is_ok());
        assert!(check_seasons(&[2021], 2021).is_err()); // not after --first-fit
        assert!(check_seasons(&[2023], 2020).is_err()); // kept clean
        assert!(check_seasons(&[2025], 2020).is_err()); // holdout
        assert!(check_seasons(&[2022], 2019).is_err()); // no snapshots before 2020/21
    }

    #[test]
    fn the_gate_needs_both_targets_to_beat_both_baselines() {
        // The model is better on every night, by a margin that varies across nights (a
        // constant margin has zero variance, so no test): the rise half passes. The fall
        // probabilities are equal across arms (0.02): identical losses, no test (p is null),
        // so the fall half fails and so does the gate.
        let m = night_metrics(&beaten(80, false));
        let gate = gate_of(&m);
        assert_eq!(gate.len(), 4);
        assert!(gate.iter().filter(|g| g.target == "rise").all(|g| g.pass && g.p < ALPHA));
        assert!(gate.iter().filter(|g| g.target == "fall").all(|g| !g.pass && g.p.is_nan()));
        assert!(!gate_pass(&m));
        // with the fall half beaten as well (and 40 rows per target with p ≥ 0.3, all
        // realized), the gate passes
        let m = night_metrics(&beaten(80, true));
        assert!(gate_of(&m).iter().all(|g| g.pass));
        let cal = calibration_gate_of(&m);
        assert_eq!(cal.len(), 2);
        assert!(cal.iter().all(|c| c.pass && c.n == 40 && c.rate == 1.0), "{cal:?}");
        assert!(gate_pass(&m));
    }

    #[test]
    fn the_gate_needs_the_calibration_condition_too() {
        // 40 nights: every DM test passes, but only 20 rows per target have p ≥ 0.3
        let m = night_metrics(&beaten(40, true));
        assert!(gate_of(&m).iter().all(|g| g.pass && g.p < ALPHA));
        let cal = calibration_gate_of(&m);
        assert!(cal.iter().all(|c| !c.pass && c.n == 20), "{cal:?}");
        assert!(!gate_pass(&m));
        // overconfident on rises (calibrated 0.8–0.9 against an observed 0.5, 80 such rows):
        // the rise condition fails, so the gate does, whatever the DM tests say
        let mut rows = beaten(80, true);
        rows.extend((0..80).map(|n| {
            let r = row(100 + n, (n % 2) as f64, 0.85, 0.5, 0.5);
            NightRow { fall: 0.0, p_fall: 0.02, b0_fall: 0.02, b1_fall: 0.02, ..r }
        }));
        let m = night_metrics(&rows);
        let cal = calibration_gate_of(&m);
        let rise = cal.iter().find(|c| c.target == "rise").unwrap();
        assert!(!rise.pass && rise.rate < rise.mean_p - CAL_TOL, "{rise:?}");
        assert!(cal.iter().find(|c| c.target == "fall").unwrap().pass);
        assert!(!gate_pass(&m));
    }

    #[test]
    fn the_calibration_condition() {
        let pairs = |n: usize, p: f64, hits: usize| -> Vec<(f64, f64)> {
            (0..n).map(|i| (f64::from(u8::from(i < hits)), p)).collect()
        };
        // 30 rows at p 0.5, rate 0.4 = 0.5 − 0.10: passes (at the boundary)
        let c = calibration_condition("rise", &pairs(30, 0.5, 12));
        assert!(c.pass && c.n == 30 && (c.mean_p - 0.5).abs() < 1e-12, "{c:?}");
        // 29 rows: fails on n
        assert!(!calibration_condition("rise", &pairs(29, 0.5, 15)).pass);
        // rate 11/30 < 0.4: fails on the rate
        assert!(!calibration_condition("rise", &pairs(30, 0.5, 11)).pass);
        // rows below p 0.3 don't count
        let mut low = pairs(30, 0.5, 15);
        low.extend(pairs(100, 0.29, 0));
        let c = calibration_condition("fall", &low);
        assert!(c.pass && c.n == 30 && c.rate == 0.5, "{c:?}");
        // no rows at all: fails, NaN means
        let c = calibration_condition("fall", &pairs(10, 0.1, 0));
        assert!(!c.pass && c.n == 0 && c.mean_p.is_nan());
    }

    #[test]
    fn calibration_bins_cover_every_scored_row() {
        let rows: Vec<NightRow> = (0..100)
            .map(|n| row(n, (n % 3 == 0) as i64 as f64, n as f64 / 100.0, 0.3, 0.3))
            .collect();
        let bins = calibration(&rows, |r| r.p_rise, |r| r.rise);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins.iter().map(|b| b.n).sum::<usize>(), 100);
        // a probability of exactly 1 lands in the last bin
        let one = [row(0, 1.0, 1.0, 0.3, 0.3)];
        assert_eq!(calibration(&one, |r| r.p_rise, |r| r.rise)[9].n, 1);
    }

    #[test]
    fn rows_table_columns() {
        let rows: Vec<NightRow> = (0..3).map(|n| row(n, 0.0, 0.1, 0.2, 0.3)).collect();
        let t = rows_table(&rows);
        assert_eq!(
            t.names(),
            [
                "season",
                "night",
                "player_key",
                "rise",
                "fall",
                "p_rise",
                "p_fall",
                "b0_rise",
                "b0_fall",
                "b1_rise",
                "b1_fall",
                "raw_p_rise",
                "raw_p_fall",
                "raw_b1_rise",
                "raw_b1_fall"
            ]
        );
        assert_eq!(t.height(), 3);
    }
}
