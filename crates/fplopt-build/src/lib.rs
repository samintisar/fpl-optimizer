//! raw/ -> `data/*.parquet` (migration batch 2): the table builders, cross-source ID mapping
//! and `rules export`. Port of `fplopt.build`, writing the same tables (rows, columns and
//! dtypes) from the same `raw/` and `config/`.
//!
//! `ORDER` is the dependency order `build()` runs the builders in. Rebuilding
//! `player_season` also rebuilds `player_match`; rebuilding `player_match` or `player_dim`
//! clears their Understat columns, so `understat` and `team_match` rerun too.

pub mod common;
pub mod fixtures;
pub mod rules;
pub mod table;
pub mod teams;

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Result, bail};

pub use common::BuildContext;
use table::{Column, Table, write_parquet};

pub const ORDER: &[&str] = &[
    "team_dim",
    "fixture",
    "gameweek",
    "gameweek_result",
    "schedule",
    "player_snapshot",
    "player_gw",
    "player_gw_ownership",
    "player_season",
    "player_match",
    "player_dim",
    "understat",
    "team_match",
    "fixture_snapshot",
    "odds_snapshot",
    "team_rating",
    "espn",
];

const DEPENDENTS: &[(&str, &[&str])] = &[("player_season", &["player_match"])];
const UNDERSTAT_TARGETS: &[&str] = &["player_match", "player_dim"];
const UNDERSTAT_AND_DEPENDENTS: &[&str] = &["understat", "team_match"];

/// What a builder returns: a table for `build()` to sort and write, or (for builders that
/// write their own outputs) only the row count and seasons for the log.
pub enum Built {
    Table(Table, &'static [&'static str]),
    Written { rows: usize, seasons: Vec<i64> },
}

fn run_builder(name: &str, ctx: &mut BuildContext) -> Result<Built> {
    Ok(match name {
        "team_dim" => Built::Table(teams::build_team_dim(ctx)?, teams::SORT_BY),
        "fixture" => Built::Table(fixtures::build_fixture(ctx)?, fixtures::FIXTURE_SORT_BY),
        "gameweek" => Built::Table(fixtures::build_gameweek(ctx)?, fixtures::GAMEWEEK_SORT_BY),
        "gameweek_result" => {
            Built::Table(fixtures::build_gameweek_result(ctx)?, fixtures::GAMEWEEK_RESULT_SORT_BY)
        }
        "schedule" => Built::Table(fixtures::build_schedule(ctx)?, fixtures::SCHEDULE_SORT_BY),
        "fixture_snapshot" => {
            Built::Table(fixtures::build_fixture_snapshot(ctx)?, fixtures::FIXTURE_SNAPSHOT_SORT_BY)
        }
        other => bail!("builder {other:?} is not ported yet"),
    })
}

fn resolve(names: &[String]) -> Result<Vec<&'static str>> {
    let mut requested: Vec<&'static str> = Vec::new();
    for name in names {
        if name == "all" {
            requested.extend(ORDER);
        } else if let Some(&known) = ORDER.iter().find(|n| **n == name) {
            requested.push(known);
        } else {
            bail!("unknown table '{name}'; valid: all, {}", ORDER.join(", "));
        }
    }
    let sources: Vec<&str> = ORDER.iter().copied().filter(|n| requested.contains(n)).collect();
    for source in sources {
        if let Some((_, deps)) = DEPENDENTS.iter().find(|(s, _)| *s == source) {
            let added: Vec<&str> =
                deps.iter().copied().filter(|d| !requested.contains(d)).collect();
            if !added.is_empty() {
                log::info!("also rebuilding {}: its rows depend on {source}", added.join(", "));
                requested.extend(added);
            }
        }
    }
    let targets: Vec<&str> =
        UNDERSTAT_TARGETS.iter().copied().filter(|t| requested.contains(t)).collect();
    if !targets.is_empty() {
        let added: Vec<&str> =
            UNDERSTAT_AND_DEPENDENTS.iter().copied().filter(|d| !requested.contains(d)).collect();
        if !added.is_empty() {
            log::info!(
                "also rebuilding {}: rebuilding {} clears the Understat columns",
                added.join(", "),
                targets.join(" and ")
            );
            requested.extend(added);
        }
    }
    Ok(ORDER.iter().copied().filter(|n| requested.contains(n)).collect())
}

fn seasons_text(seasons: &[i64]) -> String {
    let mut unique: Vec<i64> = seasons.to_vec();
    unique.sort_unstable();
    unique.dedup();
    match (unique.first(), unique.last()) {
        (Some(first), Some(last)) => format!("{first}-{last} ({})", unique.len()),
        _ => "-".into(),
    }
}

/// Sort `table` by `sort_by` and write `<data_dir>/<name>.parquet`.
pub fn write_table(
    ctx: &mut BuildContext,
    name: &str,
    table: Table,
    sort_by: &[&str],
) -> Result<Table> {
    let table = table.sorted(sort_by)?;
    std::fs::create_dir_all(&ctx.data_dir)?;
    write_parquet(&table, &ctx.table_path(name))?;
    Ok(table)
}

pub fn table_seasons(table: &Table) -> Vec<i64> {
    match table.column("season") {
        Ok(Column::I64(v)) => v.clone(),
        Ok(Column::OptI64(v)) => v.iter().flatten().copied().collect(),
        _ => Vec::new(),
    }
}

/// Build the named tables ('all' = every table) in dependency order. Unknown names fail
/// before anything runs. Returns the written paths.
pub fn build(names: &[String], ctx: &mut BuildContext) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for name in resolve(names)? {
        let started = Instant::now();
        let (rows, seasons) = match run_builder(name, ctx)? {
            Built::Table(table, sort_by) => {
                let table = write_table(ctx, name, table, sort_by)?;
                let summary = (table.height(), table_seasons(&table));
                ctx.forget(name);
                ctx.remember(name, table);
                summary
            }
            Built::Written { rows, seasons } => {
                ctx.forget(name);
                (rows, seasons)
            }
        };
        paths.push(ctx.table_path(name));
        log::info!(
            "{name}: {rows} rows, seasons {}, {:.1}s",
            seasons_text(&seasons),
            started.elapsed().as_secs_f64()
        );
    }
    Ok(paths)
}
