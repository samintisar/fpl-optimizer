//! Night price features (move timing, `docs/superpowers/specs/2026-11-09-move-timing-design.md`):
//! per player in the season's newest snapshot before the view's cutoff (a night's decision
//! moment, 22:00 UK), the state of FPL's price-change counter. `night_features(view)` is the
//! view-only builder (`FRAMES`, so `check leakage` covers it); `features_at` computes any
//! cutoff from one `SeasonSnaps` read, so training reads the snapshots once a season. Both
//! use only snapshots strictly before the cutoff.

use std::collections::BTreeMap;

use anyhow::Result;
use fplopt_store::{AsOfView, Column, Micros, Table};

use crate::baseline::target;
use crate::price::status_code;

const DAY: Micros = 86_400_000_000;
const HOUR: Micros = 3_600_000_000;
/// A red flag cleared straight to available locks the price for a week.
const LOCK: Micros = 7 * DAY;

/// The feature columns after `player_key`, in output order (all F64, NaN where undefined).
pub const COLUMNS: &[&str] = &[
    "price",
    "net_event",
    "net_since_change",
    "net_24h",
    "own",
    "net_per_own",
    "nights_since_change",
    "rises_gw",
    "falls_gw",
    "status_code",
    "chance_next",
    "red_lock",
    "hours_to_deadline",
    "gw_index",
];

const READ: &[&str] = &[
    "snapshot_at",
    "season",
    "player_key",
    "now_cost",
    "transfers_in_event",
    "transfers_out_event",
    "selected_by_percent",
    "status",
    "chance_of_playing_next_round",
    "source",
];

/// One snapshot of a player (one per time: an `fpl` row wins a tie with `fplcache`).
#[derive(Clone, Debug)]
pub struct NightSnap {
    pub at: Micros,
    pub cost: f64,
    pub net: f64,
    pub own: Option<f64>,
    pub status: Option<String>,
    pub chance: Option<f64>,
}

/// A season's player snapshots as one read sees them: per player, by time.
pub struct SeasonSnaps {
    pub by_player: BTreeMap<i64, Vec<NightSnap>>,
}

impl SeasonSnaps {
    /// `season`'s snapshots visible in `view` (available_at < its cutoff).
    pub fn read(view: &AsOfView, season: i64) -> Result<SeasonSnaps> {
        let t = view.table("player_snapshot", Some(READ))?;
        let seasons = t.opt_i64("season")?;
        let players = t.opt_i64("player_key")?;
        let times = t.opt_i64("snapshot_at")?;
        let sources = t.opt_str("source")?;
        let costs = t.opt_f64("now_cost")?;
        let ins = t.opt_f64("transfers_in_event")?;
        let outs = t.opt_f64("transfers_out_event")?;
        let owns = t.opt_f64("selected_by_percent")?;
        let statuses = t.opt_str("status")?;
        let chances = t.opt_f64("chance_of_playing_next_round")?;
        let mut raw: BTreeMap<i64, Vec<(Micros, bool, NightSnap)>> = BTreeMap::new();
        for i in 0..t.height() {
            let (Some(player), Some(at)) = (players[i], times[i]) else { continue };
            if seasons[i] != Some(season) {
                continue;
            }
            let net = match (ins[i], outs[i]) {
                (Some(a), Some(b)) => a - b,
                _ => f64::NAN,
            };
            let snap = NightSnap {
                at,
                cost: costs[i].unwrap_or(f64::NAN),
                net,
                own: owns[i],
                status: statuses[i].clone(),
                chance: chances[i],
            };
            raw.entry(player).or_default().push((at, sources[i].as_deref() == Some("fpl"), snap));
        }
        let mut by_player = BTreeMap::new();
        for (player, mut rows) in raw {
            rows.sort_by_key(|r| (r.0, r.1));
            let mut kept: Vec<NightSnap> = Vec::with_capacity(rows.len());
            for (at, _, snap) in rows {
                // sorted with the 'fpl' row last at a tie: a later row at the same time wins
                if kept.last().is_some_and(|s| s.at == at) {
                    kept.pop();
                }
                kept.push(snap);
            }
            by_player.insert(player, kept);
        }
        Ok(SeasonSnaps { by_player })
    }

    /// The newest snapshot time strictly before `cutoff` (None before the first).
    fn newest_before(&self, cutoff: Micros) -> Option<Micros> {
        self.by_player
            .values()
            .filter_map(|rows| {
                let k = rows.partition_point(|s| s.at < cutoff);
                (k > 0).then(|| rows[k - 1].at)
            })
            .max()
    }
}

/// The latest deadline of `season` at or before `cutoff` (the current GW's start: FPL resets
/// `transfers_in_event` there), i64::MIN before the first.
pub fn gw_start(view: &AsOfView, season: i64, cutoff: Micros) -> Result<Micros> {
    let g = view.table("gameweek", Some(&["season", "deadline_time"]))?;
    let seasons = g.i64("season")?;
    let deadlines = g.opt_i64("deadline_time")?;
    Ok((0..g.height())
        .filter(|&i| seasons[i] == season)
        .filter_map(|i| deadlines[i])
        .filter(|&d| d <= cutoff)
        .max()
        .unwrap_or(i64::MIN))
}

/// One player's first 12 feature columns (`COLUMNS[..12]`, price … red_lock) from his
/// snapshots before `cutoff` (`rows` sorted by time, all before the cutoff, non-empty).
fn player_row(rows: &[NightSnap], cutoff: Micros, gw_start: Micros) -> [f64; 12] {
    let last = &rows[rows.len() - 1];
    let since_gw = |s: &NightSnap| if s.at >= gw_start { s.net } else { 0.0 };
    let changes: Vec<usize> = (1..rows.len())
        .filter(|&i| {
            let (a, b) = (rows[i - 1].cost, rows[i].cost);
            a.is_finite() && b.is_finite() && a != b
        })
        .collect();
    let net_since_change = match changes.last() {
        Some(&c) if rows[c].at >= gw_start => last.net - since_gw(&rows[c - 1]),
        _ => last.net,
    };
    let k = rows.partition_point(|s| s.at <= cutoff - DAY);
    let net_24h = if k > 0 { last.net - since_gw(&rows[k - 1]) } else { last.net };
    let nights_since_change =
        changes.last().map_or(f64::NAN, |&c| (cutoff - rows[c].at) as f64 / DAY as f64);
    let in_gw = |up: bool| {
        changes
            .iter()
            .filter(|&&c| rows[c].at >= gw_start && (rows[c].cost > rows[c - 1].cost) == up)
            .count() as f64
    };
    let red = |s: &NightSnap| matches!(s.status.as_deref(), Some("i" | "s" | "u"));
    let locked = (1..rows.len()).any(|i| {
        rows[i].at >= cutoff - LOCK && rows[i].status.as_deref() == Some("a") && red(&rows[i - 1])
    });
    let own = last.own.unwrap_or(f64::NAN);
    [
        last.cost,
        last.net,
        net_since_change,
        net_24h,
        own,
        net_since_change / (own + 0.1),
        nights_since_change,
        in_gw(true),
        in_gw(false),
        status_code(last.status.as_deref()),
        last.chance.unwrap_or(f64::NAN),
        if locked { 1.0 } else { 0.0 },
    ]
}

/// Features at `cutoff` for the GW whose deadline is `deadline` (index `gw_index`, started at
/// `gw_start`): one row per player in the newest snapshot before the cutoff, sorted by
/// player_key; empty before the season's first snapshot.
pub fn features_at(
    season: &SeasonSnaps,
    cutoff: Micros,
    deadline: Micros,
    gw_start: Micros,
    gw_index: i64,
) -> Table {
    let newest = season.newest_before(cutoff);
    let mut keys = Vec::new();
    let mut cols: Vec<Vec<f64>> = vec![Vec::new(); COLUMNS.len()];
    if let Some(newest) = newest {
        for (&player, rows) in &season.by_player {
            let k = rows.partition_point(|s| s.at < cutoff);
            if k == 0 || rows[k - 1].at != newest {
                continue;
            }
            keys.push(player);
            let r = player_row(&rows[..k], cutoff, gw_start);
            for (j, v) in r.into_iter().enumerate() {
                cols[j].push(v);
            }
            cols[12].push((deadline - cutoff) as f64 / HOUR as f64);
            cols[13].push(gw_index as f64);
        }
    }
    let mut t = Table::new().with("player_key", Column::I64(keys));
    for (name, values) in COLUMNS.iter().zip(cols) {
        t = t.with(name, Column::F64(values));
    }
    t
}

/// The night features at the view's cutoff, for the GW of its deadline.
pub fn night_features(view: &AsOfView) -> Result<Table> {
    let (season, _, gw_index) = target(view)?;
    let snaps = SeasonSnaps::read(view, season)?;
    let start = gw_start(view, season, view.cutoff())?;
    Ok(features_at(&snaps, view.cutoff(), view.deadline(), start, gw_index))
}

#[cfg(test)]
mod tests {
    use super::*;
    use fplopt_store::DataStore;
    use std::collections::HashMap;

    const MINUTE: Micros = 60_000_000;
    /// 2021-08-01 00:00 UTC.
    const AUG_1: Micros = 1_627_776_000_000_000;

    fn aug(day: i64, hour: i64, minute: i64) -> Micros {
        AUG_1 + (day - 1) * DAY + hour * HOUR + minute * MINUTE
    }

    /// (at, source, player, cost, in, out, own, status, chance)
    type Row = (Micros, &'static str, i64, i64, i64, i64, f64, &'static str, Option<i64>);

    fn store(rows: &[Row]) -> DataStore {
        let col = |f: &dyn Fn(&Row) -> i64| Column::I64(rows.iter().map(f).collect());
        let snapshots = Table::new()
            .with("snapshot_at", Column::Ts(rows.iter().map(|r| r.0).collect()))
            .with("source", Column::Str(rows.iter().map(|r| r.1.to_string()).collect()))
            .with("season", col(&|_| 2021))
            .with("player_key", col(&|r| r.2))
            .with("now_cost", col(&|r| r.3))
            .with("transfers_in_event", col(&|r| r.4))
            .with("transfers_out_event", col(&|r| r.5))
            .with("selected_by_percent", Column::OptF64(rows.iter().map(|r| Some(r.6)).collect()))
            .with("status", Column::Str(rows.iter().map(|r| r.7.to_string()).collect()))
            .with(
                "chance_of_playing_next_round",
                Column::OptI64(rows.iter().map(|r| r.8).collect()),
            )
            .with("available_at", Column::Ts(rows.iter().map(|r| r.0).collect()));
        let gameweek = Table::new()
            .with("season", Column::I64(vec![2021; 2]))
            .with("gw", Column::I64(vec![1, 2]))
            .with("gw_index", Column::I64(vec![1, 2]))
            .with("deadline_time", Column::Ts(vec![aug(13, 17, 30), aug(21, 10, 0)]))
            .with("available_at", Column::Ts(vec![0; 2]));
        DataStore::from_tables(HashMap::from([
            ("gameweek".to_string(), gameweek),
            ("player_snapshot".to_string(), snapshots),
        ]))
        .unwrap()
    }

    fn rows() -> Vec<Row> {
        vec![
            (aug(14, 0, 0), "fpl", 1, 75, 1_000, 0, 10.0, "a", None),
            (aug(18, 0, 0), "fpl", 1, 76, 5_000, 0, 12.0, "a", None),
            (aug(20, 12, 0), "fplcache", 1, 80, 0, 0, 1.0, "i", Some(0)),
            (aug(20, 12, 0), "fpl", 1, 76, 9_000, 1_000, 13.9, "a", None),
            (aug(14, 0, 0), "fpl", 2, 50, 0, 500, 5.0, "a", None),
            (aug(18, 0, 0), "fpl", 2, 50, 0, 1_000, 5.0, "a", None),
            (aug(20, 12, 0), "fpl", 2, 50, 0, 2_000, 5.0, "d", Some(75)),
            (aug(22, 0, 0), "fpl", 1, 99, 50_000, 0, 40.0, "a", None),
            (aug(22, 0, 0), "fpl", 2, 99, 50_000, 0, 40.0, "a", None),
        ]
    }

    fn close(a: f64, b: f64) -> bool {
        (a.is_nan() && b.is_nan()) || (a - b).abs() < 1e-9
    }

    #[test]
    fn night_features_on_an_early_view_before_gw2() {
        let s = store(&rows());
        let cutoff = aug(20, 21, 0);
        let out = night_features(&s.as_of_early(aug(21, 10, 0), cutoff).unwrap()).unwrap();
        assert_eq!(out.i64("player_key").unwrap(), &[1, 2]);
        let v = |c: &str| out.f64(c).unwrap().to_vec();
        let expect: &[(&str, [f64; 2])] = &[
            ("price", [76.0, 50.0]),
            ("net_event", [8_000.0, -2_000.0]),
            ("net_since_change", [7_000.0, -2_000.0]),
            ("net_24h", [3_000.0, -1_000.0]),
            ("own", [13.9, 5.0]),
            ("net_per_own", [7_000.0 / 14.0, -2_000.0 / 5.1]),
            ("nights_since_change", [2.875, f64::NAN]),
            ("rises_gw", [1.0, 0.0]),
            ("falls_gw", [0.0, 0.0]),
            ("status_code", [0.0, 1.0]),
            ("chance_next", [f64::NAN, 75.0]),
            ("red_lock", [0.0, 0.0]),
            ("hours_to_deadline", [13.0, 13.0]),
            ("gw_index", [2.0, 2.0]),
        ];
        assert_eq!(out.names().len(), COLUMNS.len() + 1);
        for (c, want) in expect {
            let got = v(c);
            assert!(close(got[0], want[0]) && close(got[1], want[1]), "{c}: {got:?} vs {want:?}");
        }
    }

    #[test]
    fn features_at_equals_night_features_on_the_view() {
        let s = store(&rows());
        let late = s.as_of(aug(23, 0, 0));
        let season = SeasonSnaps::read(&late, 2021).unwrap();
        for cutoff in [aug(15, 21, 0), aug(18, 21, 0), aug(20, 21, 0)] {
            let view = s.as_of_early(aug(21, 10, 0), cutoff).unwrap();
            let a = night_features(&view).unwrap();
            let b = features_at(&season, cutoff, aug(21, 10, 0), aug(13, 17, 30), 2);
            assert_eq!(a.i64("player_key").unwrap(), b.i64("player_key").unwrap());
            for c in COLUMNS {
                let (x, y) = (a.f64(c).unwrap(), b.f64(c).unwrap());
                assert!(x.iter().zip(y).all(|(p, q)| close(*p, *q)), "{c} at {cutoff}");
            }
        }
    }

    #[test]
    fn red_lock_marks_a_red_flag_cleared_straight_to_available_within_a_week() {
        let mut r = rows();
        // player 2: injured on Aug 14, available again on Aug 18 (lock); Aug 20 'd' after it
        r[4].7 = "i";
        r[5].7 = "a";
        let s = store(&r);
        let out = night_features(&s.as_of_early(aug(21, 10, 0), aug(20, 21, 0)).unwrap()).unwrap();
        assert_eq!(out.f64("red_lock").unwrap(), &[0.0, 1.0]);
        // more than a week after Aug 18 the lock has passed (the fixture has no GW3, so call
        // features_at directly rather than through a view)
        let season = SeasonSnaps::read(&s.as_of(aug(23, 0, 0)), 2021).unwrap();
        let later = features_at(&season, aug(25, 21, 0), aug(28, 10, 0), aug(21, 10, 0), 3);
        assert_eq!(later.f64("red_lock").unwrap()[1], 0.0);
    }

    #[test]
    fn a_view_before_any_snapshot_has_no_rows() {
        let s = store(&rows());
        let out = night_features(&s.as_of_early(aug(21, 10, 0), aug(13, 21, 0)).unwrap()).unwrap();
        assert_eq!(out.height(), 0);
        assert_eq!(out.names().len(), COLUMNS.len() + 1);
    }
}
