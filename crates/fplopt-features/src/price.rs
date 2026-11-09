//! Snapshot price features (price forecast Stage B, `docs/superpowers/plans/
//! 2026-11-07-price-forecast-stage-b.md` Task 1): per pool player, price, recent price and
//! net-transfer changes, ownership and availability from the season's player snapshots
//! before the deadline.

use std::collections::HashMap;

use anyhow::Result;
use fplopt_store::{AsOfView, Column, Micros, Table};

use crate::baseline::{pool, target};
use crate::util::{days_between, median};

const DAY: Micros = 86_400_000_000;

/// The feature columns after `player_key` and `price`, in output order.
pub const COLUMNS: &[&str] = &[
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
];

const SNAPSHOT_READ: &[&str] = &[
    "snapshot_at",
    "season",
    "player_key",
    "now_cost",
    "transfers_in_event",
    "transfers_out_event",
    "selected_by_percent",
    "status",
    "chance_of_playing_next_round",
    "news_added",
    "source",
];

/// One snapshot row of a player.
struct Snap {
    at: Micros,
    fpl: bool,
    cost: f64,
    net: f64,
    own: Option<f64>,
    status: Option<String>,
    chance: Option<f64>,
    news: Option<Micros>,
}

pub(crate) fn status_code(status: Option<&str>) -> f64 {
    match status {
        Some("a") => 0.0,
        Some("d") => 1.0,
        Some("i") => 2.0,
        Some("s") => 3.0,
        Some("u") => 4.0,
        Some("n") => 5.0,
        _ => 6.0,
    }
}

/// The newest of `rows` (sorted by time, an 'fpl' row last at a tie) at or before `at`.
fn at_or_before(rows: &[Snap], at: Micros) -> Option<&Snap> {
    let k = rows.partition_point(|s| s.at <= at);
    (k > 0).then(|| &rows[k - 1])
}

/// Per pool player (sorted by player_key), as of the view's deadline D and in the current
/// season only:
/// - `price`: the pool price (tenths of £m; `now_cost` of the newest snapshot);
/// - `dprice_{1,7,28}d`: price − `now_cost` at the newest snapshot at or before D − d days;
/// - `net_event`: `transfers_in_event − transfers_out_event` at the newest snapshot, and
///   `net_event_{1,3}d` at the newest snapshot at or before D − 1 / D − 3 days;
/// - `net_per_own`: net_event / (selected_by_percent + 0.1); `net_share`: net_event over the
///   pool's Σ |net_event| (0 if that is 0); `own`: selected_by_percent;
/// - `price_rel_pos`: price over the median pool price of the player's position;
/// - `status_code` (a d i s u n = 0–5, else 6), `chance_next` (chance of playing next round),
///   `days_since_news` (days from `news_added` to D).
///
/// All F64, NaN where undefined. A player without a snapshot this season (seasons before
/// 2021/22) has NaN in every column but `price`. Where two sources share a snapshot time the
/// 'fpl' row wins, as in `AsOfView::latest`.
pub fn price_features(view: &AsOfView) -> Result<Table> {
    let (season, _, _) = target(view)?;
    let deadline = view.deadline();
    let pool = pool(view)?;
    let keys = pool.i64("player_key")?.to_vec();
    let prices: Vec<f64> = pool.i64("price")?.iter().map(|&p| p as f64).collect();
    let positions = pool.i64("element_type")?;

    let snaps = view.table("player_snapshot", Some(SNAPSHOT_READ))?;
    let seasons = snaps.opt_i64("season")?;
    let players = snaps.opt_i64("player_key")?;
    let times = snaps.opt_i64("snapshot_at")?;
    let sources = snaps.opt_str("source")?;
    let costs = snaps.opt_f64("now_cost")?;
    let ins = snaps.opt_f64("transfers_in_event")?;
    let outs = snaps.opt_f64("transfers_out_event")?;
    let owns = snaps.opt_f64("selected_by_percent")?;
    let statuses = snaps.opt_str("status")?;
    let chances = snaps.opt_f64("chance_of_playing_next_round")?;
    let news = snaps.opt_i64("news_added")?;
    let mut by_player: HashMap<i64, Vec<Snap>> = HashMap::new();
    for i in 0..snaps.height() {
        let (Some(player), Some(at)) = (players[i], times[i]) else { continue };
        if seasons[i] != Some(season) {
            continue;
        }
        let net = match (ins[i], outs[i]) {
            (Some(a), Some(b)) => a - b,
            _ => f64::NAN,
        };
        by_player.entry(player).or_default().push(Snap {
            at,
            fpl: sources[i].as_deref() == Some("fpl"),
            cost: costs[i].unwrap_or(f64::NAN),
            net,
            own: owns[i],
            status: statuses[i].clone(),
            chance: chances[i],
            news: news[i],
        });
    }
    for rows in by_player.values_mut() {
        rows.sort_by_key(|s| (s.at, s.fpl));
    }

    let mut medians: HashMap<i64, f64> = HashMap::new();
    for &position in positions {
        medians.entry(position).or_insert_with(|| {
            let mut same: Vec<f64> =
                (0..keys.len()).filter(|&j| positions[j] == position).map(|j| prices[j]).collect();
            median(&mut same)
        });
    }
    let newest: Vec<Option<&Snap>> =
        keys.iter().map(|k| by_player.get(k).and_then(|rows| rows.last())).collect();
    let total: f64 =
        newest.iter().flatten().map(|s| s.net).filter(|n| !n.is_nan()).map(f64::abs).sum();

    let n = keys.len();
    let mut columns: Vec<Vec<f64>> = vec![vec![f64::NAN; n]; COLUMNS.len()];
    for j in 0..n {
        let (Some(rows), Some(now)) = (by_player.get(&keys[j]), newest[j]) else { continue };
        let back = |days: i64| at_or_before(rows, deadline - days * DAY);
        let values = [
            back(1).map_or(f64::NAN, |s| now.cost - s.cost),
            back(7).map_or(f64::NAN, |s| now.cost - s.cost),
            back(28).map_or(f64::NAN, |s| now.cost - s.cost),
            now.net,
            back(1).map_or(f64::NAN, |s| s.net),
            back(3).map_or(f64::NAN, |s| s.net),
            now.own.map_or(f64::NAN, |own| now.net / (own + 0.1)),
            if total > 0.0 || now.net.is_nan() { now.net / total } else { 0.0 },
            now.own.unwrap_or(f64::NAN),
            prices[j] / medians[&positions[j]],
            status_code(now.status.as_deref()),
            now.chance.unwrap_or(f64::NAN),
            now.news.map_or(f64::NAN, |at| days_between(deadline, at)),
        ];
        for (column, value) in columns.iter_mut().zip(values) {
            column[j] = value;
        }
    }
    let mut out =
        Table::new().with("player_key", Column::I64(keys)).with("price", Column::F64(prices));
    for (name, values) in COLUMNS.iter().zip(columns) {
        out.push(name, Column::F64(values));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use fplopt_store::{Column, DataStore, Micros};

    use super::*;

    const MINUTE: Micros = 60_000_000;
    const HOUR: Micros = 60 * MINUTE;
    const DAY: Micros = 24 * HOUR;
    /// 2021-08-01 00:00 UTC.
    const AUG_1: Micros = 1_627_776_000_000_000;

    /// August 2021, UTC.
    fn aug(day: i64, hour: i64, minute: i64) -> Micros {
        AUG_1 + (day - 1) * DAY + hour * HOUR + minute * MINUTE
    }

    fn gw2_deadline() -> Micros {
        aug(21, 10, 0)
    }

    /// One snapshot row: (at, source, player, cost, in, out, own, status, chance, news).
    type Snap =
        (Micros, &'static str, i64, i64, i64, i64, f64, &'static str, Option<i64>, Option<Micros>);

    fn snapshots(rows: &[Snap]) -> Table {
        let col = |f: &dyn Fn(&Snap) -> i64| Column::I64(rows.iter().map(f).collect());
        Table::new()
            .with("snapshot_at", Column::Ts(rows.iter().map(|r| r.0).collect()))
            .with("source", Column::Str(rows.iter().map(|r| r.1.to_string()).collect()))
            .with("season", col(&|_| 2021))
            .with("player_key", col(&|r| r.2))
            .with("team_key", col(&|r| 10 + r.2))
            .with("element_type", col(&|_| 3))
            .with("now_cost", col(&|r| r.3))
            .with("status", Column::Str(rows.iter().map(|r| r.7.to_string()).collect()))
            .with(
                "chance_of_playing_next_round",
                Column::OptI64(rows.iter().map(|r| r.8).collect()),
            )
            .with("news_added", Column::OptTs(rows.iter().map(|r| r.9).collect()))
            .with("selected_by_percent", Column::OptF64(rows.iter().map(|r| Some(r.6)).collect()))
            .with("transfers_in_event", col(&|r| r.4))
            .with("transfers_out_event", col(&|r| r.5))
            .with("cost_change_event", col(&|_| 0))
            .with("ep_next", Column::OptF64(vec![None; rows.len()]))
            .with("form", Column::OptF64(vec![None; rows.len()]))
            .with("available_at", Column::Ts(rows.iter().map(|r| r.0).collect()))
    }

    /// GW1 and GW2 of `season` (deadlines 2021-08-13 17:30 and 2021-08-21 10:00 UTC), two
    /// MIDs (players 1 and 2, clubs 11 and 12) with `player_gw` rows for both GWs and, with
    /// `with_snapshots`, the snapshots of the plan's fixture.
    fn store(season: i64, with_snapshots: bool) -> DataStore {
        let deadlines = [aug(13, 17, 30), gw2_deadline()];
        let gameweek = Table::new()
            .with("season", Column::I64(vec![season; 2]))
            .with("gw", Column::I64(vec![1, 2]))
            .with("gw_index", Column::I64(vec![1, 2]))
            .with("deadline_time", Column::Ts(deadlines.to_vec()))
            .with("available_at", Column::Ts(vec![0; 2]));
        let schedule = Table::new()
            .with("fixture_key", Column::I64(vec![1, 2]))
            .with("season", Column::I64(vec![season; 2]))
            .with("gw", Column::I64(vec![1, 2]))
            .with("gw_index", Column::I64(vec![1, 2]))
            .with("kickoff_time", Column::Ts(deadlines.iter().map(|d| d + DAY).collect()))
            .with("home_team_key", Column::I64(vec![11, 12]))
            .with("away_team_key", Column::I64(vec![12, 11]))
            .with("schedule_source", Column::Str(vec!["final".to_string(); 2]))
            .with("available_at", Column::Ts(vec![0; 2]));
        let fixture_snapshot = Table::new()
            .with("snapshot_at", Column::Ts(vec![]))
            .with("fixture_key", Column::I64(vec![]))
            .with("season", Column::I64(vec![]))
            .with("gw", Column::I64(vec![]))
            .with("kickoff_time", Column::Ts(vec![]))
            .with("home_team_key", Column::I64(vec![]))
            .with("away_team_key", Column::I64(vec![]))
            .with("available_at", Column::Ts(vec![]));
        // (player, gw, value)
        let gw_rows = [(1, 1, 75), (2, 1, 50), (1, 2, 76), (2, 2, 50)];
        let player_gw = Table::new()
            .with("player_key", Column::I64(gw_rows.iter().map(|r| r.0).collect()))
            .with("season", Column::I64(vec![season; gw_rows.len()]))
            .with("gw", Column::I64(gw_rows.iter().map(|r| r.1).collect()))
            .with("team_key", Column::I64(gw_rows.iter().map(|r| 10 + r.0).collect()))
            .with("element_type", Column::I64(vec![3; gw_rows.len()]))
            .with("value", Column::I64(gw_rows.iter().map(|r| r.2).collect()))
            .with(
                "available_at",
                Column::Ts(gw_rows.iter().map(|r| deadlines[r.1 as usize - 1] - HOUR).collect()),
            );
        let news = Some(aug(19, 10, 0));
        let rows: Vec<Snap> = if with_snapshots {
            vec![
                (aug(14, 0, 0), "fpl", 1, 75, 1_000, 0, 10.0, "a", None, None),
                (aug(18, 0, 0), "fpl", 1, 76, 5_000, 0, 12.0, "a", None, None),
                // The fplcache mirror at the same time loses to 'fpl'.
                (aug(20, 12, 0), "fplcache", 1, 80, 0, 0, 1.0, "i", Some(0), None),
                (aug(20, 12, 0), "fpl", 1, 76, 9_000, 1_000, 13.9, "a", None, None),
                (aug(14, 0, 0), "fpl", 2, 50, 0, 500, 5.0, "a", None, None),
                (aug(18, 0, 0), "fpl", 2, 50, 0, 1_000, 5.0, "a", None, None),
                (aug(20, 12, 0), "fpl", 2, 50, 0, 2_000, 5.0, "d", Some(75), news),
                // After the GW2 deadline: invisible.
                (aug(22, 0, 0), "fpl", 1, 99, 50_000, 0, 40.0, "a", None, None),
                (aug(22, 0, 0), "fpl", 2, 99, 50_000, 0, 40.0, "a", None, None),
            ]
        } else {
            vec![]
        };
        DataStore::from_tables(HashMap::from([
            ("gameweek".to_string(), gameweek),
            ("schedule".to_string(), schedule),
            ("fixture_snapshot".to_string(), fixture_snapshot),
            ("player_gw".to_string(), player_gw),
            ("player_snapshot".to_string(), snapshots(&rows)),
        ]))
        .unwrap()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn price_features_at_the_gw2_deadline() {
        let store = store(2021, true);
        let out = price_features(&store.as_of(gw2_deadline())).unwrap();
        assert_eq!(out.i64("player_key").unwrap(), &[1, 2]);
        let f = |name: &str| out.f64(name).unwrap().to_vec();
        assert_eq!(f("price"), [76.0, 50.0]);
        // D − 1 day = 08-20 10:00: the 08-18 snapshot (76); D − 7 days = 08-14 10:00: 75.
        assert_eq!(f("dprice_1d"), [0.0, 0.0]);
        assert_eq!(f("dprice_7d"), [1.0, 0.0]);
        // No snapshot at or before D − 28 days.
        assert!(f("dprice_28d").iter().all(|x| x.is_nan()));
        assert_eq!(f("net_event"), [8_000.0, -2_000.0]);
        assert_eq!(f("net_event_1d"), [5_000.0, -1_000.0]);
        assert_eq!(f("net_event_3d"), [5_000.0, -1_000.0]);
        assert!(close(f("net_per_own")[0], 8_000.0 / 14.0));
        assert!(close(f("net_per_own")[1], -2_000.0 / 5.1));
        assert!(close(f("net_share")[0], 0.8));
        assert!(close(f("net_share")[1], -0.2));
        assert_eq!(f("own"), [13.9, 5.0]);
        // Median MID price (76 + 50) / 2 = 63.
        assert!(close(f("price_rel_pos")[0], 76.0 / 63.0));
        assert!(close(f("price_rel_pos")[1], 50.0 / 63.0));
        assert_eq!(f("status_code"), [0.0, 1.0]);
        assert!(f("chance_next")[0].is_nan());
        assert_eq!(f("chance_next")[1], 75.0);
        assert!(f("days_since_news")[0].is_nan());
        assert!(close(f("days_since_news")[1], 2.0));
    }

    #[test]
    fn price_features_without_snapshots_is_nan_but_the_price() {
        let store = store(2020, false);
        let out = price_features(&store.as_of(gw2_deadline())).unwrap();
        assert_eq!(out.i64("player_key").unwrap(), &[1, 2]);
        assert_eq!(out.f64("price").unwrap(), &[76.0, 50.0]);
        for name in out.names() {
            if name != "player_key" && name != "price" {
                assert!(out.f64(name).unwrap().iter().all(|x| x.is_nan()), "{name}");
            }
        }
        assert_eq!(out.names().len(), 2 + COLUMNS.len());
    }
}
