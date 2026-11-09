//! The night price model (move timing, `docs/superpowers/specs/2026-11-09-move-timing-design.md`):
//! P(rise tonight) and P(fall tonight) per player from `fplopt_features::night` at 22:00 UK,
//! labelled by the next morning's price (the first snapshot from 04:00 to 12:00 UK; FPL
//! changes prices at about 02:30). Two binary boosters, fitted on the nights of earlier
//! seasons only; the holdout season is refused. The boosters' probabilities are recalibrated
//! by Platt scaling (`fit_calibrated`; spec's "Calibration", amended 2026-11-09), fitted on
//! out-of-fold predictions over the training rows (5 contiguous blocks of training nights).

use std::collections::BTreeSet;

use anyhow::{Result, bail};
use fplopt_build::common::{uk_date, uk_wall_clock};
use fplopt_features::night::{COLUMNS, SeasonSnaps, features_at, gw_start};
use fplopt_store::{AsOfView, Column, DataStore, Micros, Table};

use crate::gbm::{Gbm, GbmParams, train};
use crate::price::refuse_holdout;
use crate::util::{expit, logit};

/// The decision moment and the label window, UK wall-clock hours.
pub const NIGHT_HOUR: u32 = 22;
pub const MORNING_FROM_HOUR: u32 = 4;
pub const MORNING_TO_HOUR: u32 = 12;

/// The model's feature columns (all of the night features).
pub const FEATURES: &[&str] = COLUMNS;

/// One night of a season: the decision moment, the GW it is before and the label window.
#[derive(Clone, Debug, PartialEq)]
pub struct Night {
    pub season: i64,
    pub cutoff: Micros,
    pub deadline: Micros,
    pub gw_index: i64,
    pub label_from: Micros,
    pub label_to: Micros,
}

/// Every night of `season` in the view's gameweek table: each UK date from the date of the
/// season's first deadline, at 22:00 UK, that is before a later deadline of the season (the
/// GW it decides for is that next deadline's).
pub fn nights(view: &AsOfView, season: i64) -> Result<Vec<Night>> {
    let g = view.table("gameweek", Some(&["season", "gw_index", "deadline_time"]))?;
    let seasons = g.i64("season")?;
    let indices = g.i64("gw_index")?;
    let deadlines = g.opt_i64("deadline_time")?;
    let mut gws: Vec<(Micros, i64)> = (0..g.height())
        .filter(|&i| seasons[i] == season)
        .filter_map(|i| deadlines[i].map(|d| (d, indices[i])))
        .collect();
    gws.sort_unstable();
    gws.dedup();
    let (Some(first), Some(last)) = (gws.first(), gws.last()) else { return Ok(vec![]) };
    let mut out = Vec::new();
    let mut day = uk_date(first.0);
    while uk_wall_clock(day, NIGHT_HOUR) < last.0 {
        let cutoff = uk_wall_clock(day, NIGHT_HOUR);
        let next = day.succ_opt().expect("a next day");
        if let Some(&(deadline, gw_index)) = gws.iter().find(|(d, _)| *d > cutoff) {
            out.push(Night {
                season,
                cutoff,
                deadline,
                gw_index,
                label_from: uk_wall_clock(next, MORNING_FROM_HOUR),
                label_to: uk_wall_clock(next, MORNING_TO_HOUR),
            });
        }
        day = next;
    }
    Ok(out)
}

/// The labelled rows of `seasons`' nights whose label window ends at or before `cutoff`:
/// `season`, `night` (Ts), `player_key`, `FEATURES`, `dp` (the first snapshot price in the
/// label window minus the newest before the night; NaN without one, and NaN when the newest
/// before the night is before 04:00 UK that day, since its price may predate that day's
/// change and the label could then include an earlier night's change; such a row keeps its
/// features), `rise` (dp > 0), `fall`
/// (dp < 0) (1/0, NaN with dp). Everything is read as of `cutoff` (available_at < cutoff),
/// and a label's snapshot is before its window's end, so at or before `cutoff`. Features
/// see only snapshots before each night (`features_at`, equal to `night_features` on that
/// night's view). Sorted by (season, night, player_key).
pub fn training_rows(store: &DataStore, seasons: &[i64], cutoff: Micros) -> Result<Table> {
    refuse_holdout(seasons)?;
    let view = store.as_of(cutoff);
    let mut cols: Vec<Vec<f64>> = vec![Vec::new(); FEATURES.len()];
    let (mut ss, mut ns, mut ks, mut dps) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for &season in seasons.iter().collect::<BTreeSet<_>>() {
        let snaps = SeasonSnaps::read(&view, season)?;
        for night in nights(&view, season)? {
            if night.label_to > cutoff {
                continue;
            }
            let start = gw_start(&view, season, night.cutoff)?;
            let f = features_at(&snaps, night.cutoff, night.deadline, start, night.gw_index);
            let keys = f.i64("player_key")?;
            let price = f.f64("price")?;
            for (j, c) in FEATURES.iter().enumerate() {
                cols[j].extend_from_slice(f.f64(c)?);
            }
            // the evening price must postdate this day's change (~02:30 UK)
            let fresh_from = uk_wall_clock(uk_date(night.cutoff), MORNING_FROM_HOUR);
            for (i, &key) in keys.iter().enumerate() {
                let morning = snaps.by_player.get(&key).and_then(|rows| {
                    let evening = rows.partition_point(|s| s.at < night.cutoff);
                    if evening == 0 || rows[evening - 1].at < fresh_from {
                        return None;
                    }
                    let k = rows.partition_point(|s| s.at < night.label_from);
                    rows.get(k).filter(|s| s.at < night.label_to).map(|s| s.cost)
                });
                ss.push(season);
                ns.push(night.cutoff);
                ks.push(key);
                dps.push(morning.map_or(f64::NAN, |m| m - price[i]));
            }
        }
    }
    let flag = |hit: fn(f64) -> bool| -> Vec<f64> {
        dps.iter()
            .map(|&d| if d.is_nan() { f64::NAN } else { f64::from(u8::from(hit(d))) })
            .collect()
    };
    let (rise, fall) = (flag(|d| d > 0.0), flag(|d| d < 0.0));
    let mut t = Table::new()
        .with("season", Column::I64(ss))
        .with("night", Column::Ts(ns))
        .with("player_key", Column::I64(ks));
    for (name, values) in FEATURES.iter().zip(cols) {
        t = t.with(name, Column::F64(values));
    }
    Ok(t.with("dp", Column::F64(dps))
        .with("rise", Column::F64(rise))
        .with("fall", Column::F64(fall)))
}

/// The boosters' learning parameters (one setting, no grid: Gate N is pre-registered).
#[derive(Clone, Debug, PartialEq)]
pub struct NightParams {
    pub gbm: GbmParams,
}

impl Default for NightParams {
    fn default() -> Self {
        NightParams {
            gbm: GbmParams {
                objective: "binary",
                num_boost_round: 300,
                learning_rate: 0.05,
                num_leaves: 31,
                min_data_in_leaf: 200,
                feature_fraction: 0.9,
                lambda_l2: 1.0,
            },
        }
    }
}

/// The two boosters and the seasons fitted on.
pub struct NightFit {
    pub rise: Gbm,
    pub fall: Gbm,
    pub seasons: Vec<i64>,
}

/// One binary booster per target on the rows with a finite label, in the rows' order, on
/// `features`; the holdout season is refused.
pub fn fit_on(rows: &Table, features: &[&'static str], params: &NightParams) -> Result<NightFit> {
    let seasons: Vec<i64> =
        rows.i64("season")?.iter().copied().collect::<BTreeSet<_>>().into_iter().collect();
    refuse_holdout(&seasons)?;
    let boost = |label: &str| -> Result<Gbm> {
        let y = rows.f64(label)?;
        let keep: Vec<usize> = (0..rows.height()).filter(|&i| y[i].is_finite()).collect();
        if keep.is_empty() {
            bail!("no training row has a finite {label}");
        }
        train(&rows.take(&keep), features, label, None, &params.gbm)
    };
    Ok(NightFit { rise: boost("rise")?, fall: boost("fall")?, seasons })
}

/// The model: both boosters on every night feature.
pub fn fit_night(rows: &Table, params: &NightParams) -> Result<NightFit> {
    fit_on(rows, FEATURES, params)
}

/// `player_key, p_rise, p_fall` for `features`' rows, in order.
pub fn predict_night(fit: &NightFit, features: &Table) -> Result<Table> {
    Ok(Table::new()
        .with("player_key", Column::I64(features.i64("player_key")?.to_vec()))
        .with("p_rise", Column::F64(fit.rise.predict(features)?))
        .with("p_fall", Column::F64(fit.fall.predict(features)?)))
}

/// The out-of-fold blocks of the Platt calibration (contiguous blocks of training nights).
pub const CALIBRATION_FOLDS: usize = 5;
/// Probabilities are clipped to [PLATT_CLIP, 1 − PLATT_CLIP] before the logit.
pub const PLATT_CLIP: f64 = 1e-6;
const PLATT_RIDGE: f64 = 1e-9;
const PLATT_TOL: f64 = 1e-10;
const PLATT_MAX_ITER: usize = 100;

/// Platt scaling: p' = σ(a + b·logit(p)), p clipped to [1e-6, 1 − 1e-6].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Platt {
    pub a: f64,
    pub b: f64,
}

impl Platt {
    pub fn apply(&self, p: f64) -> f64 {
        expit(self.a + self.b * logit(p.clamp(PLATT_CLIP, 1.0 - PLATT_CLIP)))
    }
}

/// The two-parameter logistic regression of y on logit(p) over (y, p) pairs (y 0/1, p
/// finite), by Newton's method (IRLS) from (a, b) = (0, 1), the identity map, with a 1e-9
/// ridge on the Hessian and step-halving (a step is halved until the log-likelihood does not
/// fall); it stops when the step's largest component is below 1e-10, after at most 100
/// iterations. Refuses pairs without both classes (no finite fit).
pub fn fit_platt(pairs: &[(f64, f64)]) -> Result<Platt> {
    if let Some(&(y, p)) = pairs.iter().find(|&&(y, p)| !(y == 0.0 || y == 1.0) || !p.is_finite()) {
        bail!("Platt: a pair (y {y}, p {p}) is not a 0/1 label with a finite probability");
    }
    let positives = pairs.iter().filter(|&&(y, _)| y == 1.0).count();
    if positives == 0 || positives == pairs.len() {
        bail!("Platt: {positives} positive(s) of {} pairs; both classes are needed", pairs.len());
    }
    let xs: Vec<(f64, f64)> =
        pairs.iter().map(|&(y, p)| (y, logit(p.clamp(PLATT_CLIP, 1.0 - PLATT_CLIP)))).collect();
    let (mut a, mut b) = (0.0, 1.0);
    for _ in 0..PLATT_MAX_ITER {
        let (mut ga, mut gb, mut haa, mut hab, mut hbb) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for &(y, x) in &xs {
            let mu = expit(a + b * x);
            let w = mu * (1.0 - mu);
            ga += y - mu;
            gb += (y - mu) * x;
            haa += w;
            hab += w * x;
            hbb += w * x * x;
        }
        haa += PLATT_RIDGE;
        hbb += PLATT_RIDGE;
        let det = haa * hbb - hab * hab;
        if !(det.is_finite() && det > 0.0) {
            bail!("Platt: singular Hessian (det {det})");
        }
        let (mut da, mut db) = ((hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det);
        // step-halving: a full Newton step can overshoot far from the optimum (e.g. from
        // the identity on overconfident p, where the weights are tiny), so halve it until
        // the negative log-likelihood does not increase
        let before = nll(&xs, a, b);
        while nll(&xs, a + da, b + db) > before && da.abs().max(db.abs()) >= PLATT_TOL {
            da /= 2.0;
            db /= 2.0;
        }
        a += da;
        b += db;
        if !(a.is_finite() && b.is_finite()) {
            bail!("Platt: the fit diverged");
        }
        if da.abs().max(db.abs()) < PLATT_TOL {
            break;
        }
    }
    Ok(Platt { a, b })
}

/// The negative log-likelihood of y given σ(a + b·x), summed (softplus kept stable).
fn nll(xs: &[(f64, f64)], a: f64, b: f64) -> f64 {
    let softplus = |z: f64| if z > 0.0 { z + (-z).exp().ln_1p() } else { z.exp().ln_1p() };
    xs.iter().map(|&(y, x)| softplus(a + b * x) - y * (a + b * x)).sum()
}

/// The fold of each of `n` nights in time order: `folds` contiguous blocks covering all of
/// them (night i is in block ⌊i·folds/n⌋). Refuses fewer nights than folds.
pub fn night_folds(n: usize, folds: usize) -> Result<Vec<usize>> {
    if folds == 0 || n < folds {
        bail!("{n} night(s) for {folds} fold(s)");
    }
    Ok((0..n).map(|i| i * folds / n).collect())
}

/// Out-of-fold predictions (rise, fall) for every row of `rows`, in the rows' order: the
/// distinct nights in time order are split into `folds` contiguous blocks (`night_folds`),
/// and each block's rows are predicted by `fit_on` fitted on the other blocks' rows. Rows
/// with a NaN label are predicted too (`fit_on` trains on the finite labels only).
pub fn oof_predictions(
    rows: &Table,
    features: &[&'static str],
    params: &NightParams,
    folds: usize,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let night = rows.i64("night")?;
    let distinct: Vec<Micros> =
        night.iter().copied().collect::<BTreeSet<_>>().into_iter().collect();
    let block = night_folds(distinct.len(), folds)?;
    let fold_of = |t: Micros| block[distinct.partition_point(|&d| d < t)];
    let row_fold: Vec<usize> = night.iter().map(|&t| fold_of(t)).collect();
    let (mut rise, mut fall) = (vec![f64::NAN; rows.height()], vec![f64::NAN; rows.height()]);
    for k in 0..folds {
        let (test, train): (Vec<usize>, Vec<usize>) =
            (0..rows.height()).partition(|&i| row_fold[i] == k);
        let fit = fit_on(&rows.take(&train), features, params)?;
        let held = rows.take(&test);
        let (r, f) = (fit.rise.predict(&held)?, fit.fall.predict(&held)?);
        for (j, &i) in test.iter().enumerate() {
            rise[i] = r[j];
            fall[i] = f[j];
        }
    }
    Ok((rise, fall))
}

/// The boosters fitted on all training rows and the Platt maps of their probabilities.
pub struct CalibratedFit {
    pub fit: NightFit,
    pub rise: Platt,
    pub fall: Platt,
}

/// Boosters on `features` (`fit_on` on all of `rows`) with Platt maps fitted on the
/// out-of-fold predictions over the same rows (`CALIBRATION_FOLDS` contiguous night blocks,
/// the same parameters) against the finite labels.
pub fn fit_calibrated(
    rows: &Table,
    features: &[&'static str],
    params: &NightParams,
) -> Result<CalibratedFit> {
    let (oof_rise, oof_fall) = oof_predictions(rows, features, params, CALIBRATION_FOLDS)?;
    let platt = |label: &str, p: &[f64]| -> Result<Platt> {
        let y = rows.f64(label)?;
        let pairs: Vec<(f64, f64)> =
            (0..rows.height()).filter(|&i| y[i].is_finite()).map(|i| (y[i], p[i])).collect();
        fit_platt(&pairs)
    };
    let (rise, fall) = (platt("rise", &oof_rise)?, platt("fall", &oof_fall)?);
    Ok(CalibratedFit { fit: fit_on(rows, features, params)?, rise, fall })
}

/// `player_key, p_rise, p_fall, raw_p_rise, raw_p_fall` for `features`' rows, in order:
/// `p_*` calibrated, `raw_p_*` the boosters' own.
pub fn predict_calibrated(fit: &CalibratedFit, features: &Table) -> Result<Table> {
    let raw_rise = fit.fit.rise.predict(features)?;
    let raw_fall = fit.fit.fall.predict(features)?;
    let map = |platt: &Platt, raw: &[f64]| raw.iter().map(|&p| platt.apply(p)).collect();
    Ok(Table::new()
        .with("player_key", Column::I64(features.i64("player_key")?.to_vec()))
        .with("p_rise", Column::F64(map(&fit.rise, &raw_rise)))
        .with("p_fall", Column::F64(map(&fit.fall, &raw_fall)))
        .with("raw_p_rise", Column::F64(raw_rise))
        .with("raw_p_fall", Column::F64(raw_fall)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use fplopt_build::common::uk_wall_clock;
    use fplopt_features::night::night_features;
    use fplopt_store::DataStore;
    use std::collections::HashMap;

    const DAY: Micros = 86_400_000_000;

    fn date(m: u32, d: u32) -> chrono::NaiveDate {
        chrono::NaiveDate::from_ymd_opt(2021, m, d).unwrap()
    }

    /// GW1 deadline Aug 13 17:30 UTC, GW2 Aug 21 10:00 UTC; player 1 costs 75 until a
    /// snapshot at 06:30 UK on Aug 16 shows 76; player 2 stays 50. Snapshots at 19:00 and
    /// 06:30 UK each day Aug 13–22.
    fn store() -> DataStore {
        store_with(16, None)
    }

    /// `store()` with player 1's rise first seen at 06:30 UK on Aug `rise_day`, and no
    /// snapshots at all on Aug `skip_day`.
    fn store_with(rise_day: u32, skip_day: Option<u32>) -> DataStore {
        let mut rows: Vec<(Micros, i64, i64)> = Vec::new();
        for d in (13..=22).filter(|&d| Some(d) != skip_day) {
            for (h, m) in [(6, 30), (19, 0)] {
                let at = uk_wall_clock(date(8, d), h) + m * 60_000_000;
                let up = at >= uk_wall_clock(date(8, rise_day), 6);
                rows.push((at, 1, if up { 76 } else { 75 }));
                rows.push((at, 2, 50));
            }
        }
        let n = rows.len();
        let snapshots = Table::new()
            .with("snapshot_at", Column::Ts(rows.iter().map(|r| r.0).collect()))
            .with("source", Column::Str(vec!["fpl".to_string(); n]))
            .with("season", Column::I64(vec![2021; n]))
            .with("player_key", Column::I64(rows.iter().map(|r| r.1).collect()))
            .with("now_cost", Column::I64(rows.iter().map(|r| r.2).collect()))
            .with("transfers_in_event", Column::I64(vec![0; n]))
            .with("transfers_out_event", Column::I64(vec![0; n]))
            .with("selected_by_percent", Column::OptF64(vec![Some(5.0); n]))
            .with("status", Column::Str(vec!["a".to_string(); n]))
            .with("chance_of_playing_next_round", Column::OptI64(vec![None; n]))
            .with("available_at", Column::Ts(rows.iter().map(|r| r.0).collect()));
        let gw1 = uk_wall_clock(date(8, 13), 18) + 30 * 60_000_000;
        let gw2 = uk_wall_clock(date(8, 21), 11);
        let gameweek = Table::new()
            .with("season", Column::I64(vec![2021; 2]))
            .with("gw", Column::I64(vec![1, 2]))
            .with("gw_index", Column::I64(vec![1, 2]))
            .with("deadline_time", Column::Ts(vec![gw1, gw2]))
            .with("available_at", Column::Ts(vec![0; 2]));
        DataStore::from_tables(HashMap::from([
            ("gameweek".to_string(), gameweek),
            ("player_snapshot".to_string(), snapshots),
        ]))
        .unwrap()
    }

    #[test]
    fn nights_run_from_the_first_deadlines_date_to_the_night_before_the_last_deadline() {
        let s = store();
        let view = s.as_of(uk_wall_clock(date(8, 25), 0));
        let n = nights(&view, 2021).unwrap();
        // Aug 13 22:00 UK is after GW1's deadline and before GW2's; the last night with a
        // later deadline is Aug 20.
        assert_eq!(n.first().unwrap().cutoff, uk_wall_clock(date(8, 13), 22));
        assert_eq!(n.last().unwrap().cutoff, uk_wall_clock(date(8, 20), 22));
        assert_eq!(n.len(), 8);
        assert!(n.iter().all(|x| x.deadline == uk_wall_clock(date(8, 21), 11)));
        assert!(n.iter().all(|x| x.gw_index == 2));
        let first = &n[0];
        assert_eq!(first.label_from, uk_wall_clock(date(8, 14), 4));
        assert_eq!(first.label_to, uk_wall_clock(date(8, 14), 12));
    }

    #[test]
    fn labels_are_the_next_mornings_price_minus_the_evenings() {
        let s = store();
        let rows = training_rows(&s, &[2021], uk_wall_clock(date(8, 25), 0)).unwrap();
        let nights = rows.opt_i64("night").unwrap();
        let keys = rows.i64("player_key").unwrap();
        let dp = rows.f64("dp").unwrap();
        let rise = rows.f64("rise").unwrap();
        let fall = rows.f64("fall").unwrap();
        let aug15 = uk_wall_clock(date(8, 15), 22);
        for i in 0..rows.height() {
            let expect = if keys[i] == 1 && nights[i] == Some(aug15) { 1.0 } else { 0.0 };
            assert_eq!(dp[i], expect, "row {i}");
            assert_eq!(rise[i], expect);
            assert_eq!(fall[i], 0.0);
        }
        // 8 nights × 2 players
        assert_eq!(rows.height(), 16);
    }

    #[test]
    fn a_stale_evening_price_has_no_label() {
        // no snapshot on Aug 17: on its night the newest price is Aug 16 19:00's, before Aug
        // 17's change at ~02:30, so Aug 18's 76 may be that earlier change; the row keeps its
        // features but its label is NaN
        let s = store_with(18, Some(17));
        let rows = training_rows(&s, &[2021], uk_wall_clock(date(8, 25), 0)).unwrap();
        let nights = rows.opt_i64("night").unwrap();
        let keys = rows.i64("player_key").unwrap();
        let price = rows.f64("price").unwrap();
        let (dp, rise, fall) =
            (rows.f64("dp").unwrap(), rows.f64("rise").unwrap(), rows.f64("fall").unwrap());
        let aug17 = uk_wall_clock(date(8, 17), 22);
        let stale: Vec<usize> = (0..rows.height()).filter(|&i| nights[i] == Some(aug17)).collect();
        assert_eq!(stale.len(), 2);
        for &i in &stale {
            assert!(dp[i].is_nan() && rise[i].is_nan() && fall[i].is_nan(), "row {i}");
            assert_eq!(price[i], if keys[i] == 1 { 75.0 } else { 50.0 });
        }
        // Aug 18's night is fresh again: 76 then, 76 the next morning
        let aug18 = uk_wall_clock(date(8, 18), 22);
        let i = (0..rows.height()).find(|&i| nights[i] == Some(aug18) && keys[i] == 1).unwrap();
        assert_eq!(dp[i], 0.0);
    }

    #[test]
    fn training_rows_see_nothing_at_or_after_the_fit_cutoff() {
        let s = store();
        // fit at Aug 17 00:00 UK: nights whose label window ends before it (Aug 13..15)
        let cut = uk_wall_clock(date(8, 17), 0);
        let rows = training_rows(&s, &[2021], cut).unwrap();
        let nights = rows.opt_i64("night").unwrap();
        assert!(nights.iter().all(|n| n.unwrap() + DAY / 2 <= cut));
        assert_eq!(rows.height(), 3 * 2);
    }

    #[test]
    fn training_rows_refuse_the_holdout() {
        let s = store();
        assert!(training_rows(&s, &[2025], uk_wall_clock(date(8, 25), 0)).is_err());
    }

    /// On real `data/` (the worktree's `../../../data`; skipped if absent): for 5 nights spread
    /// over 2021/22, `training_rows`' features equal `night_features` on the early view.
    /// Reads nothing from 2022/23 on: the views end 2022-06-01.
    #[test]
    #[ignore = "needs the real data/ directory"]
    fn night_real() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../../data");
        if !dir.join("gameweek.parquet").exists() {
            eprintln!("skipped: no data/ at {}", dir.display());
            return;
        }
        let s = DataStore::open(&dir).unwrap();
        // 2022-06-01 00:00 UTC: after 2021/22's last GW, before 2022/23
        let june_2022: Micros = 1_654_041_600_000_000;
        let all = nights(&s.as_of(june_2022), 2021).unwrap();
        assert!(all.len() > 250, "{} nights", all.len());
        let cutoff = all.last().unwrap().label_to;
        assert!(cutoff < june_2022);
        let rows = training_rows(&s, &[2021], cutoff).unwrap();
        let row_nights = rows.opt_i64("night").unwrap();
        let row_keys = rows.i64("player_key").unwrap();
        // labels: one-step changes only (FPL moves a price at most 0.1m a night)
        let dp = rows.f64("dp").unwrap();
        let labelled: Vec<usize> = (0..rows.height()).filter(|&i| dp[i].is_finite()).collect();
        let multi = labelled.iter().filter(|&&i| dp[i].abs() > 1.0).count();
        let labelled_nights: BTreeSet<Micros> =
            labelled.iter().filter_map(|&i| row_nights[i]).collect();
        eprintln!(
            "{} rows, {} labelled, {} rises, {} falls, {} with |dp| > 1, {} of {} nights unlabelled",
            rows.height(),
            labelled.len(),
            labelled.iter().filter(|&&i| dp[i] > 0.0).count(),
            labelled.iter().filter(|&&i| dp[i] < 0.0).count(),
            multi,
            all.len() - labelled_nights.len(),
            all.len()
        );
        assert!(multi * 1000 <= labelled.len(), "{multi} multi-step labels");
        for k in 0..5 {
            let night = &all[(k * (all.len() - 1)) / 4];
            let idx: Vec<usize> =
                (0..rows.height()).filter(|&i| row_nights[i] == Some(night.cutoff)).collect();
            let view = s.as_of_early(night.deadline, night.cutoff).unwrap();
            let want = night_features(&view).unwrap();
            assert!(want.height() > 0, "night {k}: no features");
            assert_eq!(idx.len(), want.height(), "night {k}: row counts");
            let keys: Vec<i64> = idx.iter().map(|&i| row_keys[i]).collect();
            assert_eq!(keys, want.i64("player_key").unwrap(), "night {k}: player keys");
            for c in FEATURES {
                let got = rows.f64(c).unwrap();
                let exp = want.f64(c).unwrap();
                for (j, &i) in idx.iter().enumerate() {
                    let (a, b) = (got[i], exp[j]);
                    assert!(
                        a == b || (a.is_nan() && b.is_nan()),
                        "night {k} {c} row {j}: {a} vs {b}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_fit_predicts_probabilities_for_every_row() {
        let s = store();
        let rows = training_rows(&s, &[2021], uk_wall_clock(date(8, 25), 0)).unwrap();
        let params = NightParams {
            gbm: GbmParams {
                num_boost_round: 5,
                min_data_in_leaf: 1,
                ..NightParams::default().gbm
            },
        };
        let fit = fit_night(&rows, &params).unwrap();
        let p = predict_night(&fit, &rows).unwrap();
        assert_eq!(p.height(), rows.height());
        for c in ["p_rise", "p_fall"] {
            assert!(p.f64(c).unwrap().iter().all(|v| (0.0..=1.0).contains(v)), "{c}");
        }
    }

    /// A 64-bit LCG (Knuth's MMIX constants): uniform draws in [0, 1).
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    #[test]
    fn platt_recovers_known_coefficients() {
        let truth = Platt { a: -1.0, b: 0.6 };
        let mut rng = Lcg(42);
        let pairs: Vec<(f64, f64)> = (0..20_000)
            .map(|_| {
                let p = 0.01 + 0.98 * rng.next();
                let y = f64::from(u8::from(rng.next() < truth.apply(p)));
                (y, p)
            })
            .collect();
        let fit = fit_platt(&pairs).unwrap();
        assert!((fit.a - truth.a).abs() < 0.1, "a {}", fit.a);
        assert!((fit.b - truth.b).abs() < 0.1, "b {}", fit.b);
        // already calibrated probabilities map to about the identity
        let mut rng = Lcg(7);
        let calibrated: Vec<(f64, f64)> = (0..20_000)
            .map(|_| {
                let p = 0.01 + 0.98 * rng.next();
                (f64::from(u8::from(rng.next() < p)), p)
            })
            .collect();
        let id = fit_platt(&calibrated).unwrap();
        assert!(id.a.abs() < 0.1 && (id.b - 1.0).abs() < 0.1, "{id:?}");
    }

    #[test]
    fn platt_converges_from_very_overconfident_probabilities() {
        // p 1e-5 or 1 − 1e-5, realized 20% and 40%: from the identity the weights are ~1e-5
        // and a full Newton step overshoots into saturation. The MLE maps the two logits to
        // exactly 0.2 and 0.4.
        let mut pairs = Vec::new();
        for i in 0..1000 {
            pairs.push((f64::from(u8::from(i % 5 == 0)), 1e-5));
            pairs.push((f64::from(u8::from(i % 5 < 2)), 1.0 - 1e-5));
        }
        let fit = fit_platt(&pairs).unwrap();
        assert!((fit.apply(1e-5) - 0.2).abs() < 1e-6, "{fit:?}");
        assert!((fit.apply(1.0 - 1e-5) - 0.4).abs() < 1e-6, "{fit:?}");
    }

    #[test]
    fn platt_needs_both_classes() {
        assert!(fit_platt(&[(1.0, 0.3), (1.0, 0.6)]).is_err());
        assert!(fit_platt(&[(0.0, 0.3), (0.0, 0.6)]).is_err());
        assert!(fit_platt(&[]).is_err());
        assert!(fit_platt(&[(0.0, 0.3), (1.0, f64::NAN)]).is_err());
        assert!(fit_platt(&[(0.0, 0.3), (1.0, 0.6), (0.0, 0.4)]).is_ok());
    }

    #[test]
    fn night_folds_are_contiguous_blocks_covering_every_night() {
        let f = night_folds(12, 5).unwrap();
        assert_eq!(f, [0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4]);
        assert!(f.windows(2).all(|w| w[1] == w[0] || w[1] == w[0] + 1));
        assert_eq!(night_folds(5, 5).unwrap(), [0, 1, 2, 3, 4]);
        assert!(night_folds(4, 5).is_err());
        assert!(night_folds(3, 0).is_err());
    }

    /// 10 nights × 20 rows, feature `price` uniform-ish: on nights 0–7 rise = price < 0.5;
    /// on nights 8–9 (the last fold) price is 10–11 and every row rises. Fitted without
    /// that fold, its rows land on the high-price side (no rise): low probabilities; any
    /// leak of the fold's own labels would push them up.
    #[test]
    fn out_of_fold_predictions_never_see_their_own_fold() {
        let (mut night, mut key, mut price, mut rise) = (vec![], vec![], vec![], vec![]);
        for n in 0..10i64 {
            for j in 0..20i64 {
                let x = j as f64 / 20.0;
                night.push(n * DAY);
                key.push(j);
                if n >= 8 {
                    price.push(10.0 + x);
                    rise.push(1.0);
                } else {
                    price.push(x);
                    rise.push(f64::from(u8::from(x < 0.5)));
                }
            }
        }
        let n = night.len();
        let fall: Vec<f64> = rise.iter().map(|r| 1.0 - r).collect();
        let rows = Table::new()
            .with("season", Column::I64(vec![2021; n]))
            .with("night", Column::Ts(night))
            .with("player_key", Column::I64(key))
            .with("price", Column::F64(price))
            .with("rise", Column::F64(rise))
            .with("fall", Column::F64(fall));
        let params = NightParams {
            gbm: GbmParams {
                num_boost_round: 30,
                learning_rate: 0.3,
                min_data_in_leaf: 5,
                ..NightParams::default().gbm
            },
        };
        let (r, f) = oof_predictions(&rows, &["price"], &params, 5).unwrap();
        assert!(r.iter().chain(&f).all(|v| v.is_finite()));
        // the last fold (rows 160..200): its own labels (all rises) are never seen
        assert!(r[160..].iter().all(|&p| p < 0.2), "{:?}", &r[160..165]);
        assert!(f[160..].iter().all(|&p| p > 0.8));
        // the first fold is predicted from the pattern of the others
        assert!(r[..10].iter().all(|&p| p > 0.5) && r[10..20].iter().all(|&p| p < 0.5));
        // fitted on everything instead, the last fold's rows do rise
        let all = fit_on(&rows, &["price"], &params).unwrap();
        let held = rows.take(&(160..200).collect::<Vec<_>>());
        assert!(all.rise.predict(&held).unwrap().iter().all(|&p| p > 0.8));
    }

    #[test]
    fn calibrated_predictions_map_the_raw_ones() {
        let s = store();
        let rows = training_rows(&s, &[2021], uk_wall_clock(date(8, 25), 0)).unwrap();
        // a single rise in the fixture: give the Platt fits both classes on every fold
        let params = NightParams {
            gbm: GbmParams {
                num_boost_round: 5,
                min_data_in_leaf: 1,
                ..NightParams::default().gbm
            },
        };
        let mut rise = rows.f64("rise").unwrap().to_vec();
        let mut fall = rows.f64("fall").unwrap().to_vec();
        for i in (0..rows.height()).step_by(3) {
            rise[i] = 1.0;
            fall[i] = 1.0;
        }
        let mut relabelled = Table::new();
        for name in rows.names().into_iter().filter(|n| !["rise", "fall"].contains(n)) {
            relabelled = relabelled.with(name, rows.column(name).unwrap().clone());
        }
        let rows = relabelled.with("rise", Column::F64(rise)).with("fall", Column::F64(fall));
        let fit = fit_calibrated(&rows, FEATURES, &params).unwrap();
        let p = predict_calibrated(&fit, &rows).unwrap();
        assert_eq!(p.names(), ["player_key", "p_rise", "p_fall", "raw_p_rise", "raw_p_fall"]);
        let (cal, raw) = (p.f64("p_rise").unwrap(), p.f64("raw_p_rise").unwrap());
        for i in 0..p.height() {
            assert!((cal[i] - fit.rise.apply(raw[i])).abs() < 1e-15);
        }
        assert_eq!(raw, fit.fit.rise.predict(&rows).unwrap());
    }
}
