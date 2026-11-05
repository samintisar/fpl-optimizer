//! `player_snapshot`: one row per player per archived bootstrap-static snapshot (fplcache
//! mirror and our own archive, `source` tells them apart). Port of `fplopt.build.snapshots`.
//!
//! Snapshots are parsed in parallel and written in chunks of at most `CHUNK_SNAPSHOTS`
//! snapshots that never straddle a season, one Parquet row group per chunk, rows in key
//! order (snapshot_at, source, element_id). Each chunk is validated before it is written;
//! the file is renamed into place only after every check passed.

use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result, bail};
use chrono::NaiveDate;
use fplopt_archive::raw_store::RawStore;
use fplopt_core::seasons::bootstrap_season;
use rayon::prelude::*;
use serde_json::Value;

use crate::common::{
    Bootstrap, BuildContext, days_from_date, fplcache_and_own_bootstraps, json_f64, json_i64,
    json_opt_i64, micros, parse_iso,
};
use crate::table::{ChunkedWriter, Column, Days, Micros, Table};

pub const NAME: &str = "player_snapshot";
pub const FIRST_SEASON: i64 = 2020;
const CHUNK_SNAPSHOTS: usize = 500;
const MANAGER: i64 = 5;
const STATUSES: [&str; 6] = ["a", "d", "i", "n", "s", "u"];

/// One snapshot's player rows, column by column.
#[derive(Default)]
struct Rows {
    snapshot_at: Vec<Micros>,
    source: Vec<String>,
    season: Vec<i64>,
    element_id: Vec<i64>,
    player_key: Vec<i64>,
    team_key: Vec<i64>,
    element_type: Vec<i64>,
    now_cost: Vec<i64>,
    status: Vec<String>,
    chance_next: Vec<Option<i64>>,
    chance_this: Vec<Option<i64>>,
    news: Vec<Option<String>>,
    news_added: Vec<Option<Micros>>,
    selected_by_percent: Vec<Option<f64>>,
    ep_next: Vec<Option<f64>>,
    ep_this: Vec<Option<f64>>,
    form: Vec<Option<f64>>,
    penalties_order: Vec<Option<i64>>,
    corners_order: Vec<Option<i64>>,
    direct_order: Vec<Option<i64>>,
    team_join_date: Vec<Option<Days>>,
    transfers_in_event: Vec<i64>,
    transfers_out_event: Vec<i64>,
    cost_change_event: Vec<i64>,
}

impl Rows {
    fn extend(&mut self, other: Rows) {
        macro_rules! ext {
            ($($f:ident),*) => { $( self.$f.extend(other.$f); )* };
        }
        ext!(
            snapshot_at,
            source,
            season,
            element_id,
            player_key,
            team_key,
            element_type,
            now_cost,
            status,
            chance_next,
            chance_this,
            news,
            news_added,
            selected_by_percent,
            ep_next,
            ep_this,
            form,
            penalties_order,
            corners_order,
            direct_order,
            team_join_date,
            transfers_in_event,
            transfers_out_event,
            cost_change_event
        );
    }

    fn len(&self) -> usize {
        self.element_id.len()
    }

    fn into_table(self) -> Table {
        let at = self.snapshot_at;
        Table::new()
            .with("snapshot_at", Column::Ts(at.clone()))
            .with("source", Column::Str(self.source))
            .with("season", Column::I64(self.season))
            .with("element_id", Column::I64(self.element_id))
            .with("player_key", Column::I64(self.player_key))
            .with("team_key", Column::I64(self.team_key))
            .with("element_type", Column::I64(self.element_type))
            .with("now_cost", Column::I64(self.now_cost))
            .with("status", Column::Str(self.status))
            .with("chance_of_playing_next_round", Column::OptI64(self.chance_next))
            .with("chance_of_playing_this_round", Column::OptI64(self.chance_this))
            .with("news", Column::OptStr(self.news))
            .with("news_added", Column::OptTs(self.news_added))
            .with("selected_by_percent", Column::OptF64(self.selected_by_percent))
            .with("ep_next", Column::OptF64(self.ep_next))
            .with("ep_this", Column::OptF64(self.ep_this))
            .with("form", Column::OptF64(self.form))
            .with("penalties_order", Column::OptI64(self.penalties_order))
            .with("corners_and_indirect_freekicks_order", Column::OptI64(self.corners_order))
            .with("direct_freekicks_order", Column::OptI64(self.direct_order))
            .with("team_join_date", Column::OptDate(self.team_join_date))
            .with("transfers_in_event", Column::I64(self.transfers_in_event))
            .with("transfers_out_event", Column::I64(self.transfers_out_event))
            .with("cost_change_event", Column::I64(self.cost_change_event))
            .with("event_time", Column::Ts(at.clone()))
            .with("available_at", Column::Ts(at))
    }
}

fn required<'a>(element: &'a Value, key: &str) -> Result<&'a Value> {
    element.get(key).with_context(|| format!("KeyError('{key}')"))
}

fn opt<'a>(element: &'a Value, key: &str) -> &'a Value {
    element.get(key).unwrap_or(&Value::Null)
}

fn opt_float(value: &Value, what: &str) -> Result<Option<f64>> {
    if value.is_null() { Ok(None) } else { json_f64(value, what).map(Some) }
}

fn opt_str(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(s) => Some(s.clone()),
        other => Some(other.to_string()),
    }
}

fn opt_time(value: &Value) -> Result<Option<Micros>> {
    match value {
        Value::Null => Ok(None),
        Value::String(s) if s.trim().is_empty() => Ok(None),
        Value::String(s) => parse_iso(s).map(Some),
        other => bail!("not a time: {other}"),
    }
}

fn opt_date(value: &Value) -> Result<Option<Days>> {
    match value {
        Value::Null => Ok(None),
        Value::String(s) => {
            let day = NaiveDate::parse_from_str(s.trim(), "%Y-%m-%d")
                .with_context(|| format!("ArrowInvalid: cannot cast {s:?} to date32"))?;
            Ok(Some(days_from_date(day)))
        }
        other => bail!("not a date: {other}"),
    }
}

/// (season, player rows) of one archived bootstrap-static file.
fn parse_snapshot(entry: &Bootstrap) -> Result<(i64, Rows)> {
    let parse = || -> Result<(i64, Rows)> {
        let payload = RawStore::read_json(&entry.path)?;
        let season = i64::from(bootstrap_season(&payload)?);
        let mut team_codes = std::collections::HashMap::new();
        for team in payload["teams"].as_array().context("KeyError('teams')")? {
            team_codes.insert(json_i64(&team["id"], "id")?, json_i64(&team["code"], "code")?);
        }
        let mut elements: Vec<(i64, &Value)> = Vec::new();
        for e in payload["elements"].as_array().context("KeyError('elements')")? {
            if json_i64(required(e, "element_type")?, "element_type")? != MANAGER {
                elements.push((json_i64(required(e, "id")?, "id")?, e));
            }
        }
        elements.sort_by_key(|(id, _)| *id);
        let at = micros(&entry.taken_at);
        let mut rows = Rows::default();
        for (id, e) in elements {
            let int = |key: &str| -> Result<i64> { json_i64(required(e, key)?, key) };
            rows.snapshot_at.push(at);
            rows.source.push(entry.source.to_string());
            rows.season.push(season);
            rows.element_id.push(id);
            rows.player_key.push(int("code")?);
            let team = int("team")?;
            rows.team_key
                .push(*team_codes.get(&team).with_context(|| format!("KeyError({team})"))?);
            rows.element_type.push(int("element_type")?);
            rows.now_cost.push(int("now_cost")?);
            rows.status.push(match required(e, "status")? {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            });
            rows.chance_next.push(json_opt_i64(opt(e, "chance_of_playing_next_round"), "chance")?);
            rows.chance_this.push(json_opt_i64(opt(e, "chance_of_playing_this_round"), "chance")?);
            rows.news.push(opt_str(opt(e, "news")));
            rows.news_added.push(opt_time(opt(e, "news_added"))?);
            rows.selected_by_percent
                .push(opt_float(opt(e, "selected_by_percent"), "selected_by_percent")?);
            rows.ep_next.push(opt_float(opt(e, "ep_next"), "ep_next")?);
            rows.ep_this.push(opt_float(opt(e, "ep_this"), "ep_this")?);
            rows.form.push(opt_float(opt(e, "form"), "form")?);
            rows.penalties_order.push(json_opt_i64(opt(e, "penalties_order"), "penalties_order")?);
            rows.corners_order.push(json_opt_i64(
                opt(e, "corners_and_indirect_freekicks_order"),
                "corners_and_indirect_freekicks_order",
            )?);
            rows.direct_order
                .push(json_opt_i64(opt(e, "direct_freekicks_order"), "direct_freekicks_order")?);
            rows.team_join_date.push(opt_date(opt(e, "team_join_date"))?);
            rows.transfers_in_event.push(int("transfers_in_event")?);
            rows.transfers_out_event.push(int("transfers_out_event")?);
            rows.cost_change_event.push(int("cost_change_event")?);
        }
        Ok((season, rows))
    };
    let name = entry.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    parse().map_err(|err| anyhow::anyhow!("{} bootstrap {name}: {err:#}", entry.source))
}

fn validate_chunk(rows: &Rows) -> Result<()> {
    let mut problems: BTreeMap<&str, usize> = BTreeMap::new();
    let mut flag = |what: &'static str, bad: bool| {
        if bad {
            *problems.entry(what).or_default() += 1;
        }
    };
    let mut keys = HashSet::new();
    for i in 0..rows.len() {
        flag("season", rows.season[i] < 2016);
        flag("element_id", rows.element_id[i] <= 0);
        flag("player_key", rows.player_key[i] <= 0);
        flag("team_key", rows.team_key[i] <= 0);
        flag("element_type", !(1..=4).contains(&rows.element_type[i]));
        flag("now_cost", rows.now_cost[i] <= 0);
        flag("status", !STATUSES.contains(&rows.status[i].as_str()));
        for chance in [rows.chance_next[i], rows.chance_this[i]] {
            flag("chance_of_playing", chance.is_some_and(|c| !(0..=100).contains(&c)));
        }
        flag(
            "selected_by_percent",
            rows.selected_by_percent[i].is_some_and(|s| !(0.0..=100.0).contains(&s)),
        );
        for order in [rows.penalties_order[i], rows.corners_order[i], rows.direct_order[i]] {
            flag("set-piece order", order.is_some_and(|o| o < 1));
        }
        flag("transfers_in_event", rows.transfers_in_event[i] < 0);
        flag("transfers_out_event", rows.transfers_out_event[i] < 0);
        flag(
            "(snapshot_at, source, element_id) unique",
            !keys.insert((rows.snapshot_at[i], rows.source[i].as_str(), rows.element_id[i])),
        );
    }
    if !problems.is_empty() {
        bail!("table '{NAME}' failed validation: {problems:?}");
    }
    Ok(())
}

/// Write `data/player_snapshot.parquet` chunk by chunk. Returns (rows, the season of every
/// row as season -> count) for the build log.
pub fn build_player_snapshot(ctx: &mut BuildContext) -> Result<(usize, BTreeMap<i64, usize>)> {
    let entries = fplcache_and_own_bootstraps(&ctx.store)?;
    if entries.is_empty() {
        bail!("{NAME}: no bootstrap-static snapshots in raw/");
    }
    std::fs::create_dir_all(&ctx.data_dir)?;
    let mut writer = ChunkedWriter::new(&ctx.table_path(NAME));
    let mut season_counts: BTreeMap<i64, usize> = BTreeMap::new();
    let mut pending: Option<(i64, Rows, usize)> = None; // season, rows, snapshots
    let flush =
        |pending: &mut Option<(i64, Rows, usize)>, writer: &mut ChunkedWriter| -> Result<()> {
            if let Some((_, rows, _)) = pending.take() {
                validate_chunk(&rows)?;
                writer.write(&rows.into_table())?;
            }
            Ok(())
        };
    // Parse a window of snapshots in parallel, consume them in order.
    for window in entries.chunks(256) {
        let parsed: Vec<Result<(i64, Rows)>> = window.par_iter().map(parse_snapshot).collect();
        for result in parsed {
            let (season, rows) = result?;
            *season_counts.entry(season).or_default() += rows.len();
            let start_new = match &pending {
                Some((current, _, count)) => *current != season || *count >= CHUNK_SNAPSHOTS,
                None => true,
            };
            if start_new {
                flush(&mut pending, &mut writer)?;
                pending = Some((season, Rows::default(), 0));
            }
            let chunk = pending.as_mut().unwrap();
            chunk.1.extend(rows);
            chunk.2 += 1;
        }
    }
    flush(&mut pending, &mut writer)?;
    let expected: Vec<i64> =
        (FIRST_SEASON..=season_counts.keys().max().copied().unwrap_or(FIRST_SEASON)).collect();
    let missing: Vec<i64> =
        expected.into_iter().filter(|s| !season_counts.contains_key(s)).collect();
    if !missing.is_empty() {
        bail!(
            "{NAME}: missing season(s) {} (expected every season from {FIRST_SEASON}; got {:?})",
            missing.iter().map(i64::to_string).collect::<Vec<_>>().join(", "),
            season_counts.keys().collect::<Vec<_>>()
        );
    }
    writer.finish()?;
    let rows = season_counts.values().sum();
    log::info!("{NAME}: {rows} rows from {} snapshots", entries.len());
    Ok((rows, season_counts))
}

/// Warn about snapshot player_keys missing from player_dim (`player_dim` builder).
pub fn report_missing_players(player_dim_keys: &HashSet<i64>, path: &Path) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    let keys = crate::table::read_parquet(path, Some(&["season", "player_key"]))?;
    let (seasons, players) = (keys.i64("season")?, keys.i64("player_key")?);
    let mut missing: BTreeMap<i64, HashSet<i64>> = BTreeMap::new();
    let mut all = std::collections::BTreeSet::new();
    for (&s, &p) in seasons.iter().zip(players) {
        if !player_dim_keys.contains(&p) {
            missing.entry(s).or_default().insert(p);
            all.insert(p);
        }
    }
    if !all.is_empty() {
        let by_season: BTreeMap<i64, usize> = missing.iter().map(|(s, k)| (*s, k.len())).collect();
        let sample: Vec<i64> = all.iter().take(20).copied().collect();
        log::warn!(
            "{NAME}: {} player_key(s) not in player_dim (per season: {by_season:?}); first: {sample:?}",
            all.len()
        );
    }
    Ok(())
}
