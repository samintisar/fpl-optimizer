//! Building and writing plan contexts (Python `fplopt.bot.context`: `build_context`,
//! `ContextStore.write`, `raw_fingerprint`, `git_sha`, `next_gameweek`).
//!
//! `build_context(store, now)` takes the next deadline after `now` in `data/` and, at that
//! deadline's view: the `LIVE_XP` frame, the player pool, the season's players, GW →
//! gw_index and each GW's deadline prices up to the planned one. A context is written under
//! a temporary name and renamed into place, so readers never see half a context.

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use chrono::{DateTime, Utc};
use fplopt_build::table::write_parquet;
use fplopt_features::baseline::player_pool;
use fplopt_optimize::start::{FAR_FUTURE, start_deadline};
use fplopt_store::{Column, DataStore, Micros, Table};
use sha2::{Digest, Sha256};

use crate::context::{ContextStore, KEEP_CONTEXTS};
use crate::util::py_isoformat;

/// The xP model of the system that passed Phase 6 (`holdout.SYSTEM`).
pub const LIVE_XP: &str = "v1";
/// The raw sources whose files identify the archive's content.
pub const FINGERPRINT_SOURCES: &[&str] = &["fpl", "odds", "football-data"];
const FRAMES: [&str; 6] = ["xp", "pool", "players", "prices", "gameweeks", "fixtures"];

/// A context to write: its meta and its frames.
pub struct NewContext {
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
    pub xp: Table,
    pub pool: Table,
    pub players: Table,
    pub prices: Table,
    pub gameweeks: Table,
    pub fixtures: Table,
}

impl NewContext {
    /// `meta.json` as Python's `json.dumps(context.meta(), indent=2)`.
    pub fn meta_json(&self) -> String {
        let text = |v: Option<String>| {
            v.map_or("null".to_string(), |s| serde_json::to_string(&s).unwrap())
        };
        format!(
            "{{\n  \"context_id\": {},\n  \"season\": {},\n  \"gw\": {},\n  \"gw_index\": {},\n  \
             \"deadline\": {},\n  \"created_at\": {},\n  \"fingerprint\": {},\n  \
             \"bootstrap_at\": {},\n  \"git_sha\": {},\n  \"xp_model\": {}\n}}\n",
            text(Some(self.context_id.clone())),
            self.season,
            self.gw,
            self.gw_index,
            text(Some(py_isoformat(self.deadline))),
            text(Some(py_isoformat(self.created_at))),
            text(Some(self.fingerprint.clone())),
            text(self.bootstrap_at.map(py_isoformat)),
            text(self.git_sha.clone()),
            text(Some(self.xp_model.clone())),
        )
    }

    fn frame(&self, name: &str) -> &Table {
        match name {
            "xp" => &self.xp,
            "pool" => &self.pool,
            "players" => &self.players,
            "prices" => &self.prices,
            "fixtures" => &self.fixtures,
            _ => &self.gameweeks,
        }
    }
}

fn micros_to_utc(t: Micros) -> DateTime<Utc> {
    DateTime::from_timestamp_micros(t).expect("a timestamp in range")
}

/// (season, gw_index) of the first deadline after `now` in the data, or None.
pub fn next_gameweek(store: &DataStore, now: DateTime<Utc>) -> Result<Option<(i64, i64)>> {
    let gameweeks = store
        .as_of(FAR_FUTURE)
        .table("gameweek", Some(&["season", "gw", "gw_index", "deadline_time"]))?;
    let (season, index) = (gameweeks.opt_i64("season")?, gameweeks.opt_i64("gw_index")?);
    let deadline = gameweeks.opt_i64("deadline_time")?;
    let now = now.timestamp_micros();
    let best = (0..gameweeks.height())
        .filter(|&i| deadline[i].is_some_and(|d| d > now))
        .min_by_key(|&i| (deadline[i], season[i], index[i]));
    Ok(best.and_then(|i| Some((season[i]?, index[i]?))))
}

/// The `fixtures` frame (context docs): both sides of every scheduled game of `season` in
/// GWs >= `gw`, sorted by (gw, team_key, fixture_key), from the as-of fixture list (`View::schedule`) and
/// `team_dim`'s short names.
pub fn fixture_rows(schedule: &Table, teams: &Table, season: i64, gw: i64) -> Result<Table> {
    let short: std::collections::HashMap<i64, String> = teams
        .i64("team_key")?
        .iter()
        .copied()
        .zip(teams.opt_str("short_name")?.into_iter().map(Option::unwrap_or_default))
        .collect();
    let (seasons, gws) = (schedule.i64("season")?, schedule.opt_i64("gw")?);
    let fixture_keys = schedule.i64("fixture_key")?;
    let (home, away) = (schedule.i64("home_team_key")?, schedule.i64("away_team_key")?);
    let mut rows: Vec<(i64, i64, i64, String, bool)> = Vec::new();
    for i in 0..schedule.height() {
        let Some(g) = gws[i] else { continue };
        if seasons[i] != season || g < gw {
            continue;
        }
        let name = |k: i64| short.get(&k).cloned().unwrap_or_default();
        rows.push((g, home[i], fixture_keys[i], name(away[i]), true));
        rows.push((g, away[i], fixture_keys[i], name(home[i]), false));
    }
    rows.sort_by_key(|r| (r.0, r.1, r.2));
    Ok(Table::new()
        .with("gw", Column::I64(rows.iter().map(|r| r.0).collect()))
        .with("team_key", Column::I64(rows.iter().map(|r| r.1).collect()))
        .with("opponent", Column::Str(rows.iter().map(|r| r.3.clone()).collect()))
        .with("home", Column::Bool(rows.iter().map(|r| r.4).collect())))
}

/// The context for the next deadline after `now` (None if the data has none).
pub fn build_context(
    store: &DataStore,
    now: DateTime<Utc>,
    fingerprint: &str,
    bootstrap_at: Option<DateTime<Utc>>,
    sha: Option<String>,
    xp_model: &str,
) -> Result<Option<NewContext>> {
    let Some((season, gw_index)) = next_gameweek(store, now)? else { return Ok(None) };
    let deadline = start_deadline(store, season, gw_index)?;
    let view = store.as_of(deadline);
    let (_, gw) = view.gameweek_for_deadline()?;
    let pool = player_pool(&view)?;
    // The season's players, one row per element (the newest wins), sorted by element.
    let rows =
        view.table("player_season", Some(&["player_key", "season", "element_id", "web_name"]))?;
    let (keys, seasons) = (rows.i64("player_key")?, rows.i64("season")?);
    let (elements, names) = (rows.opt_i64("element_id")?, rows.opt_str("web_name")?);
    let mut by_element: std::collections::BTreeMap<i64, (i64, String)> = Default::default();
    for i in 0..rows.height() {
        if seasons[i] == season
            && let Some(e) = elements[i]
        {
            by_element.insert(e, (keys[i], names[i].clone().unwrap_or_default()));
        }
    }
    let players = Table::new()
        .with("element_id", Column::I64(by_element.keys().copied().collect()))
        .with("player_key", Column::I64(by_element.values().map(|v| v.0).collect()))
        .with("web_name", Column::Str(by_element.values().map(|v| v.1.clone()).collect()));
    let gw_rows = view.table("gameweek", Some(&["season", "gw", "gw_index"]))?;
    let (gs, gg, gi) = (gw_rows.i64("season")?, gw_rows.i64("gw")?, gw_rows.i64("gw_index")?);
    let mut gameweeks: Vec<(i64, i64)> =
        (0..gw_rows.height()).filter(|&i| gs[i] == season).map(|i| (gg[i], gi[i])).collect();
    gameweeks.sort_by_key(|g| g.1);
    let key_elements: std::collections::HashMap<i64, i64> =
        by_element.iter().map(|(&e, &(k, _))| (k, e)).collect();
    let (mut p_gw, mut p_element, mut p_price) = (Vec::new(), Vec::new(), Vec::new());
    for &(past_gw, past_index) in &gameweeks {
        if past_index > gw_index {
            continue;
        }
        let past = player_pool(&store.as_of(start_deadline(store, season, past_index)?))?;
        for (key, price) in past.i64("player_key")?.iter().zip(past.i64("price")?) {
            if let Some(&element) = key_elements.get(key) {
                p_gw.push(past_gw);
                p_element.push(element);
                p_price.push(*price);
            }
        }
    }
    let prices = Table::new()
        .with("gw", Column::I64(p_gw))
        .with("element_id", Column::I64(p_element))
        .with("price", Column::I64(p_price));
    let gameweeks = Table::new()
        .with("gw", Column::I64(gameweeks.iter().map(|g| g.0).collect()))
        .with("gw_index", Column::I64(gameweeks.iter().map(|g| g.1).collect()));
    // The as-of fixture list (newest snapshot before the deadline, else the final schedule),
    // the one the models see: not the final table, which would show later reschedules.
    let schedule = view.schedule(season)?;
    let teams = view.table("team_dim", Some(&["team_key", "short_name"]))?;
    let fixtures = fixture_rows(&schedule, &teams, season, gw)?;
    let xp = fplopt_models::predict(xp_model, &view)?;
    let created = Utc::now();
    Ok(Some(NewContext {
        context_id: created.format("%Y-%m-%dT%H%M%S%6fZ").to_string(),
        season,
        gw,
        gw_index,
        deadline: micros_to_utc(deadline),
        created_at: created,
        fingerprint: fingerprint.to_string(),
        bootstrap_at,
        git_sha: sha,
        xp_model: xp_model.to_string(),
        xp,
        pool,
        players,
        prices,
        gameweeks,
        fixtures,
    }))
}

impl ContextStore {
    /// Write `context` atomically; returns its directory.
    pub fn write(&self, context: &NewContext) -> Result<PathBuf> {
        std::fs::create_dir_all(&self.root)?;
        let final_dir = self.root.join(&context.context_id);
        if final_dir.exists() {
            bail!("context {} exists", context.context_id);
        }
        let tmp = self.root.join(format!(".{}.{}.tmp", context.context_id, std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir(&tmp)?;
        let result = (|| -> Result<()> {
            for name in FRAMES {
                write_parquet(context.frame(name), &tmp.join(format!("{name}.parquet")))?;
            }
            std::fs::write(tmp.join("meta.json"), context.meta_json())?;
            std::fs::rename(&tmp, &final_dir)?;
            Ok(())
        })();
        let _ = std::fs::remove_dir_all(&tmp);
        result.map(|()| final_dir)
    }

    /// `prune(KEEP_CONTEXTS)`.
    pub fn prune_default(&self) -> Vec<String> {
        self.prune(KEEP_CONTEXTS)
    }
}

/// sha256 over the sorted relative paths ('/'-separated) of every archived file of
/// `sources` under `raw_dir` (files whose name starts with '.' excluded).
pub fn raw_fingerprint(raw_dir: &Path, sources: &[&str]) -> Result<String> {
    fn walk(dir: &Path, raw_dir: &Path, out: &mut Vec<String>) -> Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if entry.file_type()?.is_dir() {
                walk(&path, raw_dir, out)?;
            } else if !entry.file_name().to_string_lossy().starts_with('.') {
                let relative = path.strip_prefix(raw_dir)?;
                let parts: Vec<String> = relative
                    .components()
                    .map(|c| c.as_os_str().to_string_lossy().into_owned())
                    .collect();
                out.push(parts.join("/"));
            }
        }
        Ok(())
    }
    let mut paths = Vec::new();
    for source in sources {
        let root = raw_dir.join(source);
        if root.is_dir() {
            walk(&root, raw_dir, &mut paths)?;
        }
    }
    paths.sort();
    let mut digest = Sha256::new();
    for path in paths {
        digest.update(path.as_bytes());
        digest.update(b"\n");
    }
    Ok(digest.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// The code checkout's HEAD commit (`git rev-parse HEAD` in `root`), or None.
pub fn git_sha(root: &Path) -> Option<String> {
    let out = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(root)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let sha = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!sha.is_empty()).then_some(sha)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_rows_are_both_sides_of_each_game_from_the_planned_gw() {
        let schedule = Table::new()
            .with("season", Column::I64(vec![2026, 2026, 2026, 2025]))
            .with("gw", Column::OptI64(vec![Some(11), Some(12), None, Some(12)]))
            .with("fixture_key", Column::I64(vec![1, 2, 3, 4]))
            .with("home_team_key", Column::I64(vec![10, 20, 10, 10]))
            .with("away_team_key", Column::I64(vec![20, 30, 30, 20]));
        let teams = Table::new()
            .with("team_key", Column::I64(vec![10, 20, 30]))
            .with("short_name", Column::Str(vec!["ARS".into(), "BRE".into(), "LEE".into()]));
        let rows = fixture_rows(&schedule, &teams, 2026, 12).unwrap();
        // Only GW12 of 2026 (GW11 is past, the unscheduled game has no GW, 2025 is another
        // season): BRE (20) at home to LEE (30), both sides.
        assert_eq!(rows.i64("gw").unwrap(), &[12, 12]);
        assert_eq!(rows.i64("team_key").unwrap(), &[20, 30]);
        assert_eq!(rows.str("opponent").unwrap(), &["LEE".to_string(), "BRE".to_string()]);
        assert_eq!(rows.bool("home").unwrap(), &[true, false]);
    }

    // Python test_raw_fingerprint_changes_with_new_files.
    #[test]
    fn fingerprint_changes_with_new_archived_files() {
        let root = std::env::temp_dir().join(format!("fplopt-fingerprint-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let write = |path: &str| {
            let path = root.join(path);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, b"{}").unwrap();
        };
        let empty = raw_fingerprint(&root, FINGERPRINT_SOURCES).unwrap();
        write("fpl/bootstrap-static/2026/10/01/2026-10-01T000000Z.json.gz");
        let one = raw_fingerprint(&root, FINGERPRINT_SOURCES).unwrap();
        assert_ne!(one, empty);
        write("fpl/bootstrap-static/.partial.tmp");
        assert_eq!(raw_fingerprint(&root, FINGERPRINT_SOURCES).unwrap(), one);
        write("vaastav/data/2026-10-02T000000Z.csv.gz");
        assert_eq!(raw_fingerprint(&root, FINGERPRINT_SOURCES).unwrap(), one);
        write("odds/soccer_epl/2026-10-02T000000Z.json.gz");
        assert_ne!(raw_fingerprint(&root, FINGERPRINT_SOURCES).unwrap(), one);
        let _ = std::fs::remove_dir_all(&root);
    }

    // The fingerprint is Python's: sha256 of the sorted '/' paths, one per line.
    #[test]
    fn fingerprint_is_pythons_digest() {
        let root =
            std::env::temp_dir().join(format!("fplopt-fingerprint-py-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        for path in ["fpl/b/2.json.gz", "fpl/a/1.json.gz", "football-data/x/3.csv.gz"] {
            let path = root.join(path);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, b"").unwrap();
        }
        let mut digest = Sha256::new();
        for line in ["football-data/x/3.csv.gz\n", "fpl/a/1.json.gz\n", "fpl/b/2.json.gz\n"] {
            digest.update(line.as_bytes());
        }
        let want: String = digest.finalize().iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(raw_fingerprint(&root, FINGERPRINT_SOURCES).unwrap(), want);
        let _ = std::fs::remove_dir_all(&root);
    }
}
