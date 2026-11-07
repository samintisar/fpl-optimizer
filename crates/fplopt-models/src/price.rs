//! Price forecast (Stage B, `docs/superpowers/plans/2026-11-07-price-forecast-stage-b.md`):
//! per deadline feature rows (snapshot price features, v1 xP, calendar) and training rows
//! with the price change 1–6 deadlines ahead as targets.

use std::collections::{BTreeMap, HashMap};

use anyhow::{Result, bail};
use fplopt_core::seasons::{HOLDOUT_SEASONS, season_label};
use fplopt_features::baseline::{pool, target};
use fplopt_features::price::{COLUMNS as PRICE_COLUMNS, price_features};
use fplopt_features::util::days_between;
use fplopt_store::{AsOfView, Column, DataStore, Micros, Table};

use crate::gbm::float_column;

/// Horizons forecast: price change from deadline t to the deadline of gw_index t + h.
pub const HORIZONS: [usize; 6] = [1, 2, 3, 4, 5, 6];

/// Feature columns, in model order (price_features' columns + xP features + calendar).
pub const FEATURES: &[&str] = &[
    "price",
    "dprice_1d",
    "dprice_7d",
    "dprice_28d",
    "net_event",
    "net_event_1d",
    "net_event_3d",
    "net_per_own",
    "net_share",
    "own",
    "price_rel_pos",
    "status_code",
    "chance_next",
    "days_since_news",
    "xp_1",
    "xp_3",
    "xp_rank_pos",
    "gw_index",
    "days_to_next",
];

/// The target columns of the training rows, `dp_h` for each of `HORIZONS`.
pub fn target_name(h: usize) -> String {
    format!("dp_{h}")
}

/// Refuses the holdout season (PLAN §9): nothing here fits on, tunes on or evaluates it.
pub fn refuse_holdout(seasons: &[i64]) -> Result<()> {
    if let Some(s) = seasons.iter().find(|s| HOLDOUT_SEASONS.contains(&(**s as i32))) {
        bail!("season {} is the holdout; the price forecast refuses it", season_label(*s as i32));
    }
    Ok(())
}

/// The view's gameweeks: (season, gw_index) → (gw, deadline). A (season, gw) listed more than
/// once takes its newest visible row.
fn calendar(view: &AsOfView) -> Result<HashMap<(i64, i64), (i64, Option<Micros>)>> {
    let t = view.table("gameweek", Some(&["season", "gw", "gw_index", "deadline_time"]))?;
    let (seasons, gws, indices) = (t.opt_i64("season")?, t.opt_i64("gw")?, t.opt_i64("gw_index")?);
    let deadlines = t.opt_i64("deadline_time")?;
    let mut out = HashMap::new();
    for i in 0..t.height() {
        if let (Some(season), Some(gw), Some(index)) = (seasons[i], gws[i], indices[i]) {
            out.insert((season, index), (gw, deadlines[i]));
        }
    }
    Ok(out)
}

/// Deadline prices (`player_gw.value`, tenths of £m) visible in the view: (season, gw,
/// player_key) → value. A double GW's rows share the value; the last visible one is kept.
fn deadline_prices(view: &AsOfView) -> Result<HashMap<(i64, i64, i64), f64>> {
    let t = view.table("player_gw", Some(&["season", "gw", "player_key", "value"]))?;
    let (seasons, gws, keys) = (t.opt_i64("season")?, t.opt_i64("gw")?, t.opt_i64("player_key")?);
    let values = t.opt_f64("value")?;
    let mut out = HashMap::new();
    for i in 0..t.height() {
        if let (Some(s), Some(gw), Some(k), Some(v)) = (seasons[i], gws[i], keys[i], values[i]) {
            out.insert((s, gw, k), v);
        }
    }
    Ok(out)
}

/// One deadline's feature rows: price_features(view) joined with v1 xP features
/// (`xp_1` = xP at horizon 0, `xp_3` = Σ horizons 0..3, `xp_rank_pos` = rank of xp_3 within
/// element_type, 1 = best, ties by player_key) and calendar (`gw_index` of the target GW,
/// `days_to_next` = days from this deadline to the next one; NaN at the season's last GW).
///
/// Columns `player_key` then `FEATURES` (all F64), one row per pool player sorted by
/// player_key. A player without rows in `xp` has NaN `xp_1`, `xp_3` and `xp_rank_pos` and
/// takes no rank; the others are ranked among themselves (descending xp_3 by `total_cmp`,
/// then ascending player_key), so the ranks are deterministic.
pub fn deadline_features(view: &AsOfView, xp: &Table) -> Result<Table> {
    let (season, _, gw_index) = target(view)?;
    let deadline = view.deadline();
    let prices = price_features(view)?;
    let keys = prices.i64("player_key")?.to_vec();
    let n = keys.len();

    let pool = pool(view)?;
    let positions: HashMap<i64, i64> = pool
        .i64("player_key")?
        .iter()
        .copied()
        .zip(pool.i64("element_type")?.iter().copied())
        .collect();

    let (xp_keys, horizons, values) =
        (xp.i64("player_key")?, xp.i64("horizon")?, float_column(xp, "xp")?);
    let mut sums: HashMap<i64, (f64, f64)> = HashMap::new();
    for i in 0..xp.height() {
        let sum = sums.entry(xp_keys[i]).or_insert((0.0, 0.0));
        if horizons[i] == 0 {
            sum.0 += values[i];
        }
        if (0..3).contains(&horizons[i]) {
            sum.1 += values[i];
        }
    }
    let xp_1: Vec<f64> = keys.iter().map(|k| sums.get(k).map_or(f64::NAN, |s| s.0)).collect();
    let xp_3: Vec<f64> = keys.iter().map(|k| sums.get(k).map_or(f64::NAN, |s| s.1)).collect();

    let mut groups: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for j in (0..n).filter(|&j| !xp_3[j].is_nan()) {
        groups.entry(positions.get(&keys[j]).copied().unwrap_or(0)).or_default().push(j);
    }
    let mut rank = vec![f64::NAN; n];
    for mut members in groups.into_values() {
        members.sort_by(|&a, &b| xp_3[b].total_cmp(&xp_3[a]).then(keys[a].cmp(&keys[b])));
        for (r, j) in members.into_iter().enumerate() {
            rank[j] = (r + 1) as f64;
        }
    }

    let next = calendar(view)?.get(&(season, gw_index + 1)).and_then(|(_, d)| *d);
    let days_to_next = next.map_or(f64::NAN, |next| days_between(next, deadline));

    let mut out = Table::new().with("player_key", Column::I64(keys));
    for name in &FEATURES[..1 + PRICE_COLUMNS.len()] {
        out.push(name, Column::F64(prices.f64(name)?.to_vec()));
    }
    out.push("xp_1", Column::F64(xp_1));
    out.push("xp_3", Column::F64(xp_3));
    out.push("xp_rank_pos", Column::F64(rank));
    out.push("gw_index", Column::F64(vec![gw_index as f64; n]));
    out.push("days_to_next", Column::F64(vec![days_to_next; n]));
    Ok(out)
}

/// Training rows for a fit at `cutoff`: for every deadline D of `seasons` with D < cutoff,
/// the features at D plus `dp_h` for each h (deadline price at gw_index(D)+h − price at D,
/// from `player_gw` as visible at the cutoff; NaN when that GW's price isn't visible yet or
/// is past the season's end). `xp_at` returns v1 xP for a deadline view.
///
/// The features at D come from `store.as_of(D)` only, and the targets from
/// `store.as_of(cutoff)` (deadline prices are visible from deadline − 1h), so no feature or
/// target from at or after the cutoff enters. "Price at D" is `player_gw.value` at D's GW
/// too, so both ends of a target come from one source; a player without a `player_gw` row at
/// either end (a blank GW) has a NaN target there. Rows also carry `season`, `deadline` (Ts)
/// and `player_key`; they are sorted by (season, gw_index, player_key). The holdout season is
/// refused.
pub fn training_rows(
    store: &DataStore,
    cutoff: Micros,
    seasons: &[i64],
    xp_at: &dyn Fn(&AsOfView) -> Result<Table>,
) -> Result<Table> {
    refuse_holdout(seasons)?;
    let at_cutoff = store.as_of(cutoff);
    let calendar = calendar(&at_cutoff)?;
    let values = deadline_prices(&at_cutoff)?;
    let mut deadlines: Vec<(i64, i64, i64, Micros)> = calendar
        .iter()
        .filter(|((season, _), _)| seasons.contains(season))
        .filter_map(|(&(season, index), &(gw, d))| d.map(|d| (season, index, gw, d)))
        .filter(|r| r.3 < cutoff)
        .collect();
    deadlines.sort_unstable();

    let mut season_col = Vec::new();
    let mut deadline_col = Vec::new();
    let mut key_col = Vec::new();
    let mut features: Vec<Vec<f64>> = vec![Vec::new(); FEATURES.len()];
    let mut targets: Vec<Vec<f64>> = vec![Vec::new(); HORIZONS.len()];
    for (season, index, gw, deadline) in deadlines {
        let view = store.as_of(deadline);
        let frame = deadline_features(&view, &xp_at(&view)?)?;
        let keys = frame.i64("player_key")?;
        for (column, name) in features.iter_mut().zip(FEATURES) {
            column.extend_from_slice(frame.f64(name)?);
        }
        for &key in keys {
            let base = values.get(&(season, gw, key));
            for (column, &h) in targets.iter_mut().zip(&HORIZONS) {
                let later = calendar
                    .get(&(season, index + h as i64))
                    .and_then(|(gw, _)| values.get(&(season, *gw, key)));
                column.push(match (base, later) {
                    (Some(b), Some(l)) => l - b,
                    _ => f64::NAN,
                });
            }
        }
        season_col.extend(std::iter::repeat_n(season, keys.len()));
        deadline_col.extend(std::iter::repeat_n(deadline, keys.len()));
        key_col.extend_from_slice(keys);
    }
    let mut out = Table::new()
        .with("season", Column::I64(season_col))
        .with("deadline", Column::Ts(deadline_col))
        .with("player_key", Column::I64(key_col));
    for (name, column) in FEATURES.iter().zip(features) {
        out.push(name, Column::F64(column));
    }
    for (&h, column) in HORIZONS.iter().zip(targets) {
        out.push(&target_name(h), Column::F64(column));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::HashMap;

    use fplopt_store::{Column, DataStore, Micros};

    use super::*;

    const MINUTE: Micros = 60_000_000;
    const HOUR: Micros = 60 * MINUTE;
    const DAY: Micros = 24 * HOUR;
    /// 2021-08-01 00:00 UTC.
    const AUG_1: Micros = 1_627_776_000_000_000;

    /// 2021 from August 1st, UTC (day 32 = September 1st).
    fn aug(day: i64, hour: i64, minute: i64) -> Micros {
        AUG_1 + (day - 1) * DAY + hour * HOUR + minute * MINUTE
    }

    /// GW1–GW4 deadlines: 08-13 17:30, 08-21 10:00, 08-28 10:00, 09-11 10:00.
    fn deadlines() -> [Micros; 4] {
        [aug(13, 17, 30), aug(21, 10, 0), aug(28, 10, 0), aug(42, 10, 0)]
    }

    /// gw_index 1–4 are gws 1, 2, 3 and 5 (gw ≠ gw_index, as in 2019/20).
    const GWS: [i64; 4] = [1, 2, 3, 5];

    fn empty_snapshots() -> Table {
        let i = || Column::I64(vec![]);
        Table::new()
            .with("snapshot_at", Column::Ts(vec![]))
            .with("source", Column::Str(vec![]))
            .with("season", i())
            .with("player_key", i())
            .with("team_key", i())
            .with("element_type", i())
            .with("now_cost", i())
            .with("status", Column::Str(vec![]))
            .with("chance_of_playing_next_round", Column::OptI64(vec![]))
            .with("news_added", Column::OptTs(vec![]))
            .with("selected_by_percent", Column::OptF64(vec![]))
            .with("transfers_in_event", i())
            .with("transfers_out_event", i())
            .with("cost_change_event", i())
            .with("ep_next", Column::OptF64(vec![]))
            .with("form", Column::OptF64(vec![]))
            .with("available_at", Column::Ts(vec![]))
    }

    /// Season 2021, four GWs, two MIDs: player 1 at 50, 51, 51, 53 and player 2 at 60
    /// throughout at the GW1–4 deadlines (`player_gw` rows visible from deadline − 1h).
    fn store() -> DataStore {
        let d = deadlines();
        let gameweek = Table::new()
            .with("season", Column::I64(vec![2021; 4]))
            .with("gw", Column::I64(GWS.to_vec()))
            .with("gw_index", Column::I64(vec![1, 2, 3, 4]))
            .with("deadline_time", Column::Ts(d.to_vec()))
            .with("available_at", Column::Ts(vec![0; 4]));
        let schedule = Table::new()
            .with("fixture_key", Column::I64(vec![1, 2, 3, 4]))
            .with("season", Column::I64(vec![2021; 4]))
            .with("gw", Column::I64(GWS.to_vec()))
            .with("gw_index", Column::I64(vec![1, 2, 3, 4]))
            .with("kickoff_time", Column::Ts(d.iter().map(|x| x + DAY).collect()))
            .with("home_team_key", Column::I64(vec![11, 12, 11, 12]))
            .with("away_team_key", Column::I64(vec![12, 11, 12, 11]))
            .with("schedule_source", Column::Str(vec!["final".to_string(); 4]))
            .with("available_at", Column::Ts(vec![0; 4]));
        let fixture_snapshot = Table::new()
            .with("snapshot_at", Column::Ts(vec![]))
            .with("fixture_key", Column::I64(vec![]))
            .with("season", Column::I64(vec![]))
            .with("gw", Column::I64(vec![]))
            .with("kickoff_time", Column::Ts(vec![]))
            .with("home_team_key", Column::I64(vec![]))
            .with("away_team_key", Column::I64(vec![]))
            .with("available_at", Column::Ts(vec![]));
        // (player, gw_index, value)
        let mut rows = Vec::new();
        for (i, (p1, p2)) in [(50, 60), (51, 60), (51, 60), (53, 60)].into_iter().enumerate() {
            rows.push((1, i, p1));
            rows.push((2, i, p2));
        }
        let player_gw = Table::new()
            .with("player_key", Column::I64(rows.iter().map(|r| r.0).collect()))
            .with("season", Column::I64(vec![2021; rows.len()]))
            .with("gw", Column::I64(rows.iter().map(|r| GWS[r.1]).collect()))
            .with("team_key", Column::I64(rows.iter().map(|r| 10 + r.0).collect()))
            .with("element_type", Column::I64(vec![3; rows.len()]))
            .with("value", Column::I64(rows.iter().map(|r| r.2).collect()))
            .with("available_at", Column::Ts(rows.iter().map(|r| d[r.1] - HOUR).collect()));
        DataStore::from_tables(HashMap::from([
            ("gameweek".to_string(), gameweek),
            ("schedule".to_string(), schedule),
            ("fixture_snapshot".to_string(), fixture_snapshot),
            ("player_gw".to_string(), player_gw),
            ("player_snapshot".to_string(), empty_snapshots()),
        ]))
        .unwrap()
    }

    /// A constant xP frame: player 1 at 2, 1, 1, 5, ... and player 2 at 1, 2, 1, 0, ... per
    /// horizon (both xp_3 = 4: a tie, broken by player_key).
    fn xp() -> Table {
        let per = |p: i64, h: i64| match (p, h) {
            (1, 0) => 2.0,
            (1, 3) => 5.0,
            (2, 1) => 2.0,
            (2, 3) => 0.0,
            _ => 1.0,
        };
        let rows: Vec<(i64, i64)> =
            [1, 2].iter().flat_map(|&p| (0..6).map(move |h| (p, h))).collect();
        Table::new()
            .with("player_key", Column::I64(rows.iter().map(|r| r.0).collect()))
            .with("gw", Column::I64(rows.iter().map(|r| r.1 + 1).collect()))
            .with("gw_index", Column::I64(rows.iter().map(|r| r.1 + 1).collect()))
            .with("horizon", Column::I64(rows.iter().map(|r| r.1).collect()))
            .with("xp", Column::F64(rows.iter().map(|&(p, h)| per(p, h)).collect()))
    }

    /// The rows of `table` at `deadline` for `player`.
    fn row(table: &Table, deadline: Micros, player: i64) -> usize {
        let ds = table.i64("deadline").unwrap();
        let ps = table.i64("player_key").unwrap();
        let rows: Vec<usize> =
            (0..table.height()).filter(|&i| ds[i] == deadline && ps[i] == player).collect();
        assert_eq!(rows.len(), 1, "rows at {deadline} for player {player}");
        rows[0]
    }

    fn dp(table: &Table, h: usize, i: usize) -> f64 {
        table.f64(&format!("dp_{h}")).unwrap()[i]
    }

    #[test]
    fn deadline_features_join_xp_and_calendar() {
        let store = store();
        let d = deadlines();
        let out = deadline_features(&store.as_of(d[0]), &xp()).unwrap();
        assert_eq!(out.i64("player_key").unwrap(), &[1, 2]);
        let mut names = vec!["player_key"];
        names.extend(FEATURES);
        assert_eq!(out.names(), names);
        let f = |name: &str| out.f64(name).unwrap().to_vec();
        assert_eq!(f("price"), [50.0, 60.0]);
        assert_eq!(f("xp_1"), [2.0, 1.0]);
        assert_eq!(f("xp_3"), [4.0, 4.0]);
        assert_eq!(f("xp_rank_pos"), [1.0, 2.0]);
        assert_eq!(f("gw_index"), [1.0, 1.0]);
        // 08-13 17:30 to 08-21 10:00 = 7 days 16.5 hours.
        assert_eq!(f("days_to_next"), [7.6875, 7.6875]);
        // A player missing from the xP frame has NaN xP and rank; the others rank among
        // themselves.
        let partial = xp().take(&(6..12).collect::<Vec<_>>());
        let out = deadline_features(&store.as_of(d[3]), &partial).unwrap();
        let f = |name: &str| out.f64(name).unwrap().to_vec();
        assert!(f("xp_1")[0].is_nan() && f("xp_3")[0].is_nan() && f("xp_rank_pos")[0].is_nan());
        assert_eq!(f("xp_rank_pos")[1], 1.0);
        assert_eq!(f("gw_index"), [4.0, 4.0]);
        // The season's last GW has no next deadline.
        assert!(f("days_to_next").iter().all(|x| x.is_nan()));
    }

    #[test]
    fn targets_are_deadline_price_differences() {
        let store = store();
        let d = deadlines();
        let rows = training_rows(&store, d[3] + DAY, &[2021], &|_| Ok(xp())).unwrap();
        assert_eq!(rows.height(), 8);
        assert_eq!(rows.i64("season").unwrap(), &[2021; 8]);
        let i = row(&rows, d[0], 1);
        assert_eq!([dp(&rows, 1, i), dp(&rows, 2, i), dp(&rows, 3, i)], [1.0, 1.0, 3.0]);
        // Past the season's end.
        assert!((4..=6).all(|h| dp(&rows, h, i).is_nan()));
        assert_eq!(dp(&rows, 1, row(&rows, d[1], 1)), 0.0);
        assert_eq!(dp(&rows, 2, row(&rows, d[1], 1)), 2.0);
        assert_eq!(dp(&rows, 3, row(&rows, d[0], 2)), 0.0);
        assert!(HORIZONS.iter().all(|&h| dp(&rows, h, row(&rows, d[3], 1)).is_nan()));
        for name in FEATURES {
            assert!(rows.has(name), "{name}");
        }
    }

    #[test]
    fn no_target_past_the_cutoff() {
        let store = store();
        let d = deadlines();
        let cutoff = d[1] + MINUTE;
        let seen = RefCell::new(Vec::new());
        let xp_at = |view: &AsOfView| {
            seen.borrow_mut().push(view.deadline());
            Ok(xp())
        };
        let rows = training_rows(&store, cutoff, &[2021], &xp_at).unwrap();
        // The xP (and every feature) comes from views at deadlines before the cutoff.
        assert_eq!(*seen.borrow(), [d[0], d[1]]);
        let ds = rows.i64("deadline").unwrap();
        assert!(ds.iter().all(|&x| x == d[0] || x == d[1]));
        assert_eq!(rows.height(), 4);
        // GW2's price is visible (from its deadline − 1h), GW3's is not.
        let i = row(&rows, d[0], 1);
        assert_eq!(dp(&rows, 1, i), 1.0);
        assert!(dp(&rows, 2, i).is_nan());
        for player in [1, 2] {
            let i = row(&rows, d[1], player);
            assert!(HORIZONS.iter().all(|&h| dp(&rows, h, i).is_nan()));
        }
    }

    #[test]
    fn holdout_is_refused() {
        let store = store();
        let err = training_rows(&store, deadlines()[3], &[2024, 2025], &|_| Ok(xp())).unwrap_err();
        assert!(err.to_string().contains("2025"), "{err}");
    }
}
