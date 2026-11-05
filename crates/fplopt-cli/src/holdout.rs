//! `holdout check` (Python `cli._holdout_check`, `backtest.holdout.preflight`): the 2025/26
//! inputs at every deadline (pool size, pool players with an `ep_next`, clubs with a fixture
//! and no pool player), no results read. Fails (and alerts) on any problem. `holdout run`,
//! the one-off Phase 6 run, stays Python's: it has run (the experiment log holds it).

use std::path::Path;

use anyhow::{Result, bail};
use fplopt_backtest::simulator::season_schedule;
use fplopt_core::seasons::season_label;
use fplopt_features::baseline::{ep_next, player_pool, pool_coverage};
use fplopt_optimize::start::fmt_timestamp;
use fplopt_store::DataStore;

/// The holdout season (Python `holdout.SEASON`).
pub const SEASON: i64 = 2025;

pub fn check(data_dir: &Path) -> Result<()> {
    let store = DataStore::open(data_dir)?;
    let schedule = season_schedule(&store, SEASON, 1)?;
    let mut rows: Vec<[String; 5]> = Vec::new();
    let mut problems = Vec::new();
    for g in &schedule {
        let view = store.as_of(g.deadline);
        let pool = player_pool(&view)?;
        let ep = ep_next(&view)?;
        let coverage = pool_coverage(&view)?;
        let (teams, fixtures, players) = (
            coverage.i64("team_key")?,
            coverage.i64("n_fixtures")?,
            coverage.i64("n_pool_players")?,
        );
        let missing: Vec<i64> = (0..coverage.height())
            .filter(|&i| fixtures[i] > 0 && players[i] == 0)
            .map(|i| teams[i])
            .collect();
        let n_ep = ep.opt_f64("ep_next")?.iter().filter(|v| v.is_some_and(|x| !x.is_nan())).count();
        let n_pool = pool.height();
        rows.push([
            g.gw.to_string(),
            fmt_timestamp(g.deadline),
            n_pool.to_string(),
            n_ep.to_string(),
            missing.len().to_string(),
        ]);
        if n_pool == 0 {
            problems.push(format!("GW{}: empty player pool", g.gw));
        } else if (n_ep as f64) < 0.95 * n_pool as f64 {
            problems.push(format!("GW{}: ep_next for {n_ep} of {n_pool} pool players", g.gw));
        }
        if !missing.is_empty() {
            let list: Vec<String> = missing.iter().map(i64::to_string).collect();
            problems.push(format!(
                "GW{}: clubs with a fixture and no pool players [{}]",
                g.gw,
                list.join(", ")
            ));
        }
    }
    // pandas' to_string(index=False): a column is as wide as its header + 1 or its widest value
    let headers = ["gw", "deadline", "pool", "ep_next", "clubs_without_players"];
    let widths: Vec<usize> = (0..5)
        .map(|c| rows.iter().map(|r| r[c].len()).chain([headers[c].len() + 1]).max().unwrap())
        .collect();
    let line = |cells: &[&str]| {
        cells
            .iter()
            .enumerate()
            .map(|(i, c)| format!("{c:>w$}", w = widths[i]))
            .collect::<Vec<_>>()
            .join(" ")
    };
    println!("{} inputs per deadline", season_label(SEASON as i32));
    println!("{}", line(&headers));
    for r in &rows {
        println!("{}", line(&r.iter().map(String::as_str).collect::<Vec<_>>()));
    }
    if !problems.is_empty() {
        bail!("{} problem(s): {}", problems.len(), problems.join("; "));
    }
    println!("\nNo problems: the holdout inputs are complete.");
    Ok(())
}
