//! The night price model (move timing, `docs/superpowers/specs/2026-11-09-move-timing-design.md`):
//! P(rise tonight) and P(fall tonight) per player from `fplopt_features::night` at 22:00 UK,
//! labelled by the next morning's price (the first snapshot from 04:00 to 12:00 UK; FPL
//! changes prices at about 02:30). Two binary boosters, fitted on the nights of earlier
//! seasons only; the holdout season is refused.

use std::collections::BTreeSet;

use anyhow::{Result, bail};
use fplopt_build::common::{uk_date, uk_wall_clock};
use fplopt_features::night::{COLUMNS, SeasonSnaps, features_at, gw_start};
use fplopt_store::{AsOfView, Column, DataStore, Micros, Table};

use crate::gbm::{Gbm, GbmParams, train};
use crate::price::refuse_holdout;

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

#[cfg(test)]
mod tests {
    use super::*;
    use fplopt_build::common::uk_wall_clock;
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
}
