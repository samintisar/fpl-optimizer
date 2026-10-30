//! The plan context (Python `fplopt.bot.context`; Phase 7 plan §1): everything a user's plan
//! for the next deadline needs except the squad, computed once per change of the archive and
//! shared by every user.
//!
//! A context lives in `<root>/<context_id>/`: `meta.json` and the frames `xp`, `pool`,
//! `players` (element_id, player_key, web_name of the season), `prices` (gw, element_id,
//! price: each GW's deadline price up to the planned one) and `gameweeks` (gw, gw_index). The
//! newest `context_id` (a sortable UTC stamp) is the current one. Python's `live prepare`
//! writes them until the models move (batch 6); the bot only reads them.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use fplopt_build::table::read_parquet;
use fplopt_optimize::XpFrame;
use fplopt_optimize::entry::GwIndex;
use fplopt_optimize::start::pool_players;
use fplopt_optimize::state::PoolPlayer;
use serde_json::Value;

use crate::util::parse_isoformat;

/// KEEP_CONTEXTS: `ContextStore::prune` keeps this many newest contexts.
pub const KEEP_CONTEXTS: usize = 12;

/// A row of the context's `players` frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContextPlayer {
    pub element_id: i64,
    pub player_key: i64,
    pub web_name: String,
}

/// Inputs for every plan at one deadline (module docs). `prices_at`: when the pool's prices
/// were replaced by FPL's live ones (`with_live_prices`); None = the snapshot's prices.
#[derive(Clone, Debug)]
pub struct PlanContext {
    pub context_id: String,
    pub season: i64,
    pub gw: i64,
    pub gw_index: i64,
    pub deadline: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
    pub fingerprint: String,
    pub bootstrap_at: Option<DateTime<Utc>>,
    pub git_sha: Option<String>,
    pub xp_model: String,
    pub xp: Arc<XpFrame>,
    pub pool: Vec<PoolPlayer>,
    pub players: Arc<Vec<ContextPlayer>>,
    /// (gw, element_id, price)
    pub prices: Arc<Vec<(i64, i64, i64)>>,
    /// (gw, gw_index)
    pub gameweeks: Vec<(i64, i64)>,
    pub prices_at: Option<DateTime<Utc>>,
}

impl PlanContext {
    pub fn gw_index_map(&self) -> GwIndex {
        self.gameweeks.iter().copied().collect()
    }

    /// element_id -> player_key.
    pub fn element_keys(&self) -> HashMap<i64, i64> {
        self.players.iter().map(|p| (p.element_id, p.player_key)).collect()
    }

    /// player_key -> web name (the last row of a key wins, as Python's dict).
    pub fn names(&self) -> HashMap<i64, String> {
        self.players.iter().map(|p| (p.player_key, p.web_name.clone())).collect()
    }

    /// element_id -> price at GW `gw`'s deadline (the entry import's `initial_prices`).
    pub fn initial_prices(&self, gw: i64) -> HashMap<i64, i64> {
        self.prices.iter().filter(|r| r.0 == gw).map(|r| (r.1, r.2)).collect()
    }
}

/// The context with the pool's prices replaced by FPL's current ones (`prices`: element_id
/// -> now_cost, from a bootstrap fetched at `fetched_at`). FPL changes prices nightly and a
/// context is rebuilt only after the next archived snapshot, so plans fetch prices live.
/// Players missing from `prices` keep the context's price; the xP is unchanged.
pub fn with_live_prices(
    context: &PlanContext,
    prices: &HashMap<i64, i64>,
    fetched_at: DateTime<Utc>,
) -> PlanContext {
    // {key: prices[element] for element, key in element_keys.items() if element in prices}
    let mut by_key: HashMap<i64, i64> = HashMap::new();
    for p in context.players.iter() {
        if let Some(&price) = prices.get(&p.element_id) {
            by_key.insert(p.player_key, price);
        }
    }
    let mut out = context.clone();
    for p in &mut out.pool {
        if let Some(&price) = by_key.get(&p.player_key) {
            p.price = price;
        }
    }
    out.prices_at = Some(fetched_at);
    out
}

/// Plan contexts under `root` (module docs); `latest` caches the newest.
pub struct ContextStore {
    pub root: PathBuf,
    cache: RefCell<Option<Rc<PlanContext>>>,
}

impl ContextStore {
    pub fn new(root: impl Into<PathBuf>) -> ContextStore {
        ContextStore { root: root.into(), cache: RefCell::new(None) }
    }

    pub fn ids(&self) -> Vec<String> {
        let Ok(entries) = std::fs::read_dir(&self.root) else { return Vec::new() };
        let mut ids: Vec<String> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir() && e.path().join("meta.json").is_file())
            .filter_map(|e| e.file_name().into_string().ok())
            .filter(|name| !name.starts_with('.'))
            .collect();
        ids.sort();
        ids
    }

    /// The meta of `context_id` (default: the newest), without reading its frames.
    pub fn meta(&self, context_id: Option<&str>) -> Result<Option<Value>> {
        let id = match context_id {
            Some(id) => id.to_string(),
            None => match self.ids().pop() {
                Some(id) => id,
                None => return Ok(None),
            },
        };
        let path = self.root.join(&id).join("meta.json");
        let text = std::fs::read_to_string(&path).with_context(|| path.display().to_string())?;
        Ok(Some(serde_json::from_str(&text)?))
    }

    pub fn read(&self, context_id: &str) -> Result<PlanContext> {
        read_context(&self.root.join(context_id))
    }

    /// The newest context (cached until a newer one appears).
    pub fn latest(&self) -> Result<Option<Rc<PlanContext>>> {
        let Some(newest) = self.ids().pop() else { return Ok(None) };
        let mut cache = self.cache.borrow_mut();
        if cache.as_ref().is_none_or(|c| c.context_id != newest) {
            *cache = Some(Rc::new(self.read(&newest)?));
        }
        Ok(cache.clone())
    }

    /// Delete all but the `keep` newest contexts (the recommendations in SQLite stay);
    /// returns the deleted ids.
    pub fn prune(&self, keep: usize) -> Vec<String> {
        let ids = self.ids();
        let old: Vec<String> = ids[..ids.len().saturating_sub(keep)].to_vec();
        for id in &old {
            let _ = std::fs::remove_dir_all(self.root.join(id));
        }
        old
    }
}

fn int(meta: &Value, key: &str) -> Result<i64> {
    meta.get(key).and_then(Value::as_i64).with_context(|| format!("meta.json: no integer '{key}'"))
}

fn text(meta: &Value, key: &str) -> Result<Option<String>> {
    match meta.get(key) {
        None => anyhow::bail!("meta.json: no '{key}'"),
        Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(other) => Ok(Some(other.to_string())),
    }
}

fn time(meta: &Value, key: &str) -> Result<Option<DateTime<Utc>>> {
    text(meta, key)?.map(|t| parse_isoformat(&t)).transpose()
}

fn int_column(table: &fplopt_build::table::Table, name: &str, frame: &str) -> Result<Vec<i64>> {
    table
        .opt_i64(name)?
        .into_iter()
        .map(|v| v.with_context(|| format!("{frame}.{name}: a null")))
        .collect()
}

/// A context directory (module docs).
pub fn read_context(directory: &Path) -> Result<PlanContext> {
    let meta_path = directory.join("meta.json");
    let meta: Value = serde_json::from_str(
        &std::fs::read_to_string(&meta_path).with_context(|| meta_path.display().to_string())?,
    )?;
    let frame = |name: &str| read_parquet(&directory.join(format!("{name}.parquet")), None);
    let players = frame("players")?;
    let names = players.opt_str("web_name")?;
    let players: Vec<ContextPlayer> = int_column(&players, "element_id", "players")?
        .into_iter()
        .zip(int_column(&players, "player_key", "players")?)
        .zip(names)
        .map(|((element_id, player_key), name)| ContextPlayer {
            element_id,
            player_key,
            web_name: name.unwrap_or_else(|| "None".to_string()),
        })
        .collect();
    let prices = frame("prices")?;
    let prices: Vec<(i64, i64, i64)> = int_column(&prices, "gw", "prices")?
        .into_iter()
        .zip(int_column(&prices, "element_id", "prices")?)
        .zip(int_column(&prices, "price", "prices")?)
        .map(|((g, e), p)| (g, e, p))
        .collect();
    let gameweeks = frame("gameweeks")?;
    let gameweeks: Vec<(i64, i64)> = int_column(&gameweeks, "gw", "gameweeks")?
        .into_iter()
        .zip(int_column(&gameweeks, "gw_index", "gameweeks")?)
        .collect();
    Ok(PlanContext {
        context_id: text(&meta, "context_id")?.context("meta.json: null context_id")?,
        season: int(&meta, "season")?,
        gw: int(&meta, "gw")?,
        gw_index: int(&meta, "gw_index")?,
        deadline: time(&meta, "deadline")?.context("meta.json: null deadline")?,
        created_at: time(&meta, "created_at")?.context("meta.json: null created_at")?,
        fingerprint: text(&meta, "fingerprint")?.unwrap_or_else(|| "None".to_string()),
        bootstrap_at: time(&meta, "bootstrap_at")?,
        git_sha: text(&meta, "git_sha")?,
        xp_model: text(&meta, "xp_model")?.unwrap_or_else(|| "None".to_string()),
        xp: Arc::new(XpFrame::from_table(&frame("xp")?)?),
        pool: pool_players(&frame("pool")?)?,
        players: Arc::new(players),
        prices: Arc::new(prices),
        gameweeks,
        prices_at: None,
    })
}
