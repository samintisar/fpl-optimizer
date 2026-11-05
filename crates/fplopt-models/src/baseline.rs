//! Baseline xP models (Python `fplopt.models.baseline`). Each takes one `AsOfView`, reads only
//! through it and returns an xP frame: one row per `player_pool` player and GW from the target
//! GW to `HORIZON` GWs later by `gw_index` (from `upcoming_fixtures`; fewer near the season's
//! end), columns `player_key, gw, gw_index, horizon (0 = target), xp` (int64 × 4, float64),
//! sorted by (player_key, horizon). `xp` is 0 when unknown and in a GW where the player's club
//! blanks; in a double it counts both fixtures.
//!
//! - `xp_rolling`: per-fixture rate = mean `rescored_points` over the player's last
//!   `ROLLING_N` visible `player_match` rows (any season, by kickoff; 0-minute rows count), 0
//!   without rows; xP = rate × his club's fixtures in the GW.
//! - `xp_ep_next`: FPL's `ep_next` from the newest player snapshot. Per-fixture rate =
//!   `ep_next / n_fixtures` of the target GW if his club plays in it, else the snapshot's
//!   `form`; nulls count as 0. Without a snapshot of the season every xP is 0 (a warning).
//! - `xp_ep_next_fade`: `xp_ep_next`'s rate in the target GW; in horizon GW h ≥ 1 it fades
//!   toward the long-run rate (last `LONG_RUN_N` matches, as `xp_rolling`):
//!   `FADE**h · ep_rate + (1 − FADE**h) · long_run_rate`, × fixtures.

use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::Result;
use fplopt_features::baseline::{ep_next, player_pool, upcoming_fixtures};
use fplopt_features::util::Kahan;
use fplopt_store::{AsOfView, Column, Table};

use crate::util::opt_ts;

pub use fplopt_features::baseline::HORIZON;

pub const ROLLING_N: usize = 5;
pub const LONG_RUN_N: usize = 10;
pub const FADE: f64 = 0.5;
pub const XP_COLUMNS: &[&str] = &["player_key", "gw", "gw_index", "horizon", "xp"];

/// Pool player × horizon GW with the club's fixtures in that GW (0 = blank), sorted by
/// (player_key, horizon) as the finished frame is.
pub(crate) struct Grid {
    pub player_key: Vec<i64>,
    pub gw: Vec<i64>,
    pub gw_index: Vec<i64>,
    pub horizon: Vec<i64>,
    pub n_fixtures: Vec<i64>,
}

impl Grid {
    pub fn len(&self) -> usize {
        self.player_key.len()
    }

    pub fn new(pool: &Table, view: &AsOfView) -> Result<Grid> {
        let fixtures = upcoming_fixtures(view)?;
        let (team, gw, gwi, horizon, n) = (
            fixtures.i64("team_key")?,
            fixtures.i64("gw")?,
            fixtures.i64("gw_index")?,
            fixtures.i64("horizon")?,
            fixtures.i64("n_fixtures")?,
        );
        // The horizon's gameweeks (one per horizon) and each club's fixtures per gameweek.
        let mut gameweeks: BTreeMap<i64, (i64, i64)> = BTreeMap::new();
        let mut club: HashMap<(i64, i64), i64> = HashMap::new();
        for i in 0..fixtures.height() {
            gameweeks.insert(horizon[i], (gw[i], gwi[i]));
            club.insert((team[i], horizon[i]), n[i]);
        }
        let keys = pool.i64("player_key")?;
        let teams = pool.i64("team_key")?;
        let mut order: Vec<usize> = (0..pool.height()).collect();
        order.sort_by_key(|&i| keys[i]);
        let mut grid = Grid {
            player_key: Vec::new(),
            gw: Vec::new(),
            gw_index: Vec::new(),
            horizon: Vec::new(),
            n_fixtures: Vec::new(),
        };
        for i in order {
            for (&h, &(g, gi)) in &gameweeks {
                grid.player_key.push(keys[i]);
                grid.gw.push(g);
                grid.gw_index.push(gi);
                grid.horizon.push(h);
                grid.n_fixtures.push(club.get(&(teams[i], h)).copied().unwrap_or(0));
            }
        }
        Ok(grid)
    }

    /// The xP frame with `xp` (NaN → 0).
    pub fn finish(&self, xp: Vec<f64>) -> Table {
        let xp = xp.into_iter().map(|v| if v.is_nan() { 0.0 } else { v }).collect();
        Table::new()
            .with("player_key", Column::I64(self.player_key.clone()))
            .with("gw", Column::I64(self.gw.clone()))
            .with("gw_index", Column::I64(self.gw_index.clone()))
            .with("horizon", Column::I64(self.horizon.clone()))
            .with("xp", Column::F64(xp))
    }
}

/// Per pool player: mean `rescored_points` over his last `n` visible `player_match` rows
/// (any season, by kickoff then fixture; 0-minute rows count). Players without rows (or
/// with only null points) are absent.
pub(crate) fn rolling_rate(view: &AsOfView, pool: &Table, n: usize) -> Result<HashMap<i64, f64>> {
    let columns = ["player_key", "kickoff_time", "fixture_key", "rescored_points"];
    let matches = view.table("player_match", Some(&columns))?;
    let in_pool: HashSet<i64> = pool.i64("player_key")?.iter().copied().collect();
    let keys = matches.i64("player_key")?;
    let kickoff = opt_ts(&matches, "kickoff_time")?;
    let fixture = matches.i64("fixture_key")?;
    let points = matches.opt_f64("rescored_points")?;
    let mut rows: Vec<usize> =
        (0..matches.height()).filter(|&i| in_pool.contains(&keys[i])).collect();
    // pandas' stable sort, NaT last.
    rows.sort_by_key(|&i| (keys[i], kickoff[i].is_none(), kickoff[i], fixture[i]));
    let mut by_player: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for i in rows {
        by_player.entry(keys[i]).or_default().push(i);
    }
    let mut rate = HashMap::new();
    for (key, rows) in by_player {
        let recent = &rows[rows.len().saturating_sub(n)..];
        let (mut sum, mut count) = (Kahan::default(), 0usize);
        for &i in recent {
            if let Some(p) = points[i].filter(|p| !p.is_nan()) {
                sum.add(p);
                count += 1;
            }
        }
        if count > 0 {
            rate.insert(key, sum.value() / count as f64);
        }
    }
    Ok(rate)
}

pub fn xp_rolling(view: &AsOfView) -> Result<Table> {
    let pool = player_pool(view)?;
    let grid = Grid::new(&pool, view)?;
    let rate = rolling_rate(view, &pool, ROLLING_N)?;
    let xp = (0..grid.len())
        .map(|i| rate.get(&grid.player_key[i]).copied().unwrap_or(0.0) * grid.n_fixtures[i] as f64)
        .collect();
    Ok(grid.finish(xp))
}

/// Per grid row, `xp_ep_next`'s per-fixture rate; None (with a warning) without a snapshot.
fn ep_rate(view: &AsOfView, pool: &Table, grid: &Grid) -> Result<Option<Vec<f64>>> {
    if !pool.str("source")?.iter().any(|s| s == "snapshot") {
        log::warn!(
            "ep_next xP at {}: no player snapshot of the season before the deadline (ep_next \
             exists from 2020/21 GW32); every xP is 0",
            view.deadline()
        );
        return Ok(None);
    }
    let snapshot = ep_next(view)?;
    let keys = snapshot.i64("player_key")?;
    let expected = snapshot.opt_f64("ep_next")?;
    let form = snapshot.opt_f64("form")?;
    let by_key: HashMap<i64, usize> = keys.iter().enumerate().map(|(i, &k)| (k, i)).collect();
    let target: HashMap<i64, i64> = (0..grid.len())
        .filter(|&i| grid.horizon[i] == 0)
        .map(|i| (grid.player_key[i], grid.n_fixtures[i]))
        .collect();
    let value = |column: &[Option<f64>], key: i64| {
        by_key.get(&key).and_then(|&i| column[i]).filter(|v| !v.is_nan()).unwrap_or(0.0)
    };
    Ok(Some(
        (0..grid.len())
            .map(|i| {
                let key = grid.player_key[i];
                let n = target.get(&key).copied().unwrap_or(0);
                if n > 0 { value(&expected, key) / n as f64 } else { value(&form, key) }
            })
            .collect(),
    ))
}

pub fn xp_ep_next(view: &AsOfView) -> Result<Table> {
    let pool = player_pool(view)?;
    let grid = Grid::new(&pool, view)?;
    let Some(rate) = ep_rate(view, &pool, &grid)? else {
        return Ok(grid.finish(vec![0.0; grid.len()]));
    };
    let xp = (0..grid.len()).map(|i| rate[i] * grid.n_fixtures[i] as f64).collect();
    Ok(grid.finish(xp))
}

pub fn xp_ep_next_fade(view: &AsOfView) -> Result<Table> {
    let pool = player_pool(view)?;
    let grid = Grid::new(&pool, view)?;
    let Some(rate) = ep_rate(view, &pool, &grid)? else {
        return Ok(grid.finish(vec![0.0; grid.len()]));
    };
    let long_run = rolling_rate(view, &pool, LONG_RUN_N)?;
    let xp = (0..grid.len())
        .map(|i| {
            let weight = FADE.powf(grid.horizon[i] as f64);
            let long = long_run.get(&grid.player_key[i]).copied().unwrap_or(0.0);
            (weight * rate[i] + (1.0 - weight) * long) * grid.n_fixtures[i] as f64
        })
        .collect();
    Ok(grid.finish(xp))
}
