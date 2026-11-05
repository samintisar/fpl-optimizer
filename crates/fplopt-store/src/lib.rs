//! The as-of read path (migration batch 3; Python `fplopt.features.store`):
//! `DataStore::open(data_dir).as_of(deadline)` is how everything downstream of `data/` reads
//! tables, and the only code in the feature layer that reads files.
//!
//! A view exposes only rows with `available_at < deadline` (strict; the column, never
//! `event_time`). Snapshot tables are read with `latest()` (newest snapshot per group before
//! the deadline); the as-of fixture list with `schedule()`. Static tables are not as-of, so
//! `table()` refuses them: `lookup()` returns only their identity columns
//! (`TableSpec::public_columns`) for keys visible at the deadline.
//!
//! Each table is loaded once per store, lazily and column by column, and kept sorted by
//! `available_at` (snapshot tables: at equal times the preferred source last). A view's
//! visible rows are therefore a prefix found by binary search, and `latest()` uses a group
//! index built once per (table, by). Stores are not invalidated: open a new one after `data/`
//! changes.
//!
//! `files_blocked()`: while it runs, opening `data/` through a `DataStore` fails; the leakage
//! check runs the feature builders inside it.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
pub use fplopt_build::table::{Column, Micros, Table};
use fplopt_build::table::{parquet_columns, read_parquet};
pub use fplopt_build::tables::{Kind, TABLES, TableSpec, table_names, table_spec};

/// At equal snapshot times, rows of these sources win in `latest()` (our own archive is
/// authoritative over the fplcache mirror).
pub const PREFERRED_SOURCES: &[&str] = &["fpl"];
pub const SCHEDULE_COLUMNS: &[&str] = &[
    "fixture_key",
    "season",
    "gw",
    "gw_index",
    "kickoff_time",
    "home_team_key",
    "away_team_key",
    "schedule_source",
];

thread_local! {
    static FILES_BLOCKED: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Run `f` with reading `data/` through a `DataStore` blocked (naming `reason` in the
/// error). In-memory stores still work.
pub fn files_blocked<T>(reason: &str, f: impl FnOnce() -> T) -> T {
    struct Reset(Option<String>);
    impl Drop for Reset {
        fn drop(&mut self) {
            FILES_BLOCKED.with(|cell| *cell.borrow_mut() = self.0.take());
        }
    }
    let previous = FILES_BLOCKED.with(|cell| cell.borrow_mut().replace(reason.to_string()));
    let _reset = Reset(previous);
    f()
}

fn check_files_allowed() -> Result<()> {
    if let Some(reason) = FILES_BLOCKED.with(|cell| cell.borrow().clone()) {
        bail!("reading data/ through a DataStore is blocked {reason}");
    }
    Ok(())
}

fn unknown(name: &str) -> anyhow::Error {
    anyhow::anyhow!("unknown table {name:?}; known tables: {}", table_names().join(", "))
}

enum Source {
    Dir(PathBuf),
    Memory(Arc<HashMap<String, Table>>),
}

/// Positions grouped by the `by` columns (nulls form groups), ascending within a group.
struct Groups {
    offsets: Vec<usize>,
    positions: Vec<usize>,
}

/// One table's columns, loaded on demand and stored in `available_at` order.
struct Stored {
    spec: &'static TableSpec,
    columns: Vec<String>,
    order: Option<Vec<usize>>,
    available: Vec<Micros>,
    file: Option<PathBuf>,
    memory: Option<Arc<HashMap<String, Table>>>,
    arrays: Mutex<HashMap<String, Arc<Column>>>,
    groups: Mutex<HashMap<Vec<String>, Arc<Groups>>>,
}

impl Stored {
    fn load(&self, names: &[&str]) -> Result<Table> {
        if let Some(memory) = &self.memory {
            let table = &memory[self.spec.name];
            let mut out = Table::new();
            for name in names {
                out.push(name, table.column(name)?.clone());
            }
            return Ok(out);
        }
        check_files_allowed()?; // columns load lazily, possibly later
        let path = self.file.as_ref().expect("a file-backed table");
        read_parquet(path, Some(names))
    }

    fn open(spec: &'static TableSpec, source: &Source) -> Result<Stored> {
        let (columns, file, memory) = match source {
            Source::Memory(tables) => {
                let table = tables.get(spec.name).with_context(|| {
                    let mut provided: Vec<_> = tables.keys().cloned().collect();
                    provided.sort();
                    format!(
                        "table {:?} was not provided; provided: {}",
                        spec.name,
                        provided.join(", ")
                    )
                })?;
                let columns = table.names().iter().map(|c| c.to_string()).collect();
                (columns, None, Some(tables.clone()))
            }
            Source::Dir(dir) => {
                check_files_allowed()?;
                let path = dir.join(format!("{}.parquet", spec.name));
                if !path.exists() {
                    bail!(
                        "table {:?} has not been built ({}); run `fplopt build all`",
                        spec.name,
                        path.display()
                    );
                }
                (parquet_columns(&path)?, Some(path), None)
            }
        };
        let mut stored = Stored {
            spec,
            columns,
            order: None,
            available: Vec::new(),
            file,
            memory,
            arrays: Mutex::new(HashMap::new()),
            groups: Mutex::new(HashMap::new()),
        };
        let with_source =
            spec.kind == Kind::Snapshot && stored.columns.iter().any(|c| c == "source");
        let first: Vec<&str> =
            if with_source { vec!["available_at", "source"] } else { vec!["available_at"] };
        let raw = stored.load(&first)?;
        let available = match raw.column("available_at")? {
            Column::Ts(v) => v.clone(),
            Column::OptTs(v) => {
                let missing = v.iter().filter(|x| x.is_none()).count();
                bail!("{}.available_at has {missing} missing value(s)", spec.name)
            }
            other => {
                bail!("{}.available_at must be tz-aware datetimes, got {}", spec.name, other.kind())
            }
        };
        let mut order: Vec<usize> = (0..available.len()).collect();
        if with_source {
            let preferred: Vec<bool> = raw
                .opt_str("source")?
                .iter()
                .map(|s| s.as_deref().is_some_and(|s| PREFERRED_SOURCES.contains(&s)))
                .collect();
            order.sort_by_key(|&i| (available[i], preferred[i]));
        } else {
            order.sort_by_key(|&i| available[i]);
        }
        let identity = order.iter().enumerate().all(|(i, &j)| i == j);
        stored.order = (!identity).then_some(order);
        stored.available = match &stored.order {
            None => available,
            Some(order) => order.iter().map(|&i| available[i]).collect(),
        };
        stored.keep(raw);
        Ok(stored)
    }

    fn n(&self) -> usize {
        self.available.len()
    }

    fn keep(&self, raw: Table) {
        let mut arrays = self.arrays.lock().unwrap();
        for (name, column) in raw.columns {
            let column = match &self.order {
                None => column,
                Some(order) => column.take(order),
            };
            arrays.insert(name, Arc::new(column));
        }
    }

    fn require(&self, names: &[&str]) -> Result<Vec<Arc<Column>>> {
        let unknown: Vec<_> =
            names.iter().filter(|c| !self.columns.iter().any(|n| n == *c)).collect();
        if !unknown.is_empty() {
            bail!(
                "table {:?} has no column(s) {unknown:?}; columns: {}",
                self.spec.name,
                self.columns.join(", ")
            );
        }
        let missing: Vec<&str> = {
            let arrays = self.arrays.lock().unwrap();
            let mut seen = BTreeSet::new();
            names.iter().copied().filter(|c| !arrays.contains_key(*c) && seen.insert(*c)).collect()
        };
        if !missing.is_empty() {
            let loaded = self.load(&missing)?;
            self.keep(loaded);
        }
        let arrays = self.arrays.lock().unwrap();
        Ok(names.iter().map(|c| arrays[*c].clone()).collect())
    }

    /// Number of rows with available_at < deadline (they are the first rows).
    fn visible(&self, deadline: Micros) -> usize {
        self.available.partition_point(|&t| t < deadline)
    }

    fn frame(&self, names: &[&str], rows: &[usize]) -> Result<Table> {
        let columns = self.require(names)?;
        let mut out = Table::new();
        for (name, column) in names.iter().zip(columns) {
            out.push(name, column.take(rows).normalized());
        }
        Ok(out)
    }

    fn prefix(&self, names: &[&str], k: usize) -> Result<Table> {
        let rows: Vec<usize> = (0..k).collect();
        self.frame(names, &rows)
    }

    fn groups(&self, by: &[&str]) -> Result<Arc<Groups>> {
        let id: Vec<String> = by.iter().map(|c| c.to_string()).collect();
        if let Some(groups) = self.groups.lock().unwrap().get(&id) {
            return Ok(groups.clone());
        }
        let columns = self.require(by)?;
        let n = self.n();
        let mut codes = vec![0u64; n];
        for column in &columns {
            let part = factorize(column);
            let mut combined: HashMap<(u64, u64), u64> = HashMap::new();
            for (code, p) in codes.iter_mut().zip(part) {
                let next = combined.len() as u64;
                *code = *combined.entry((*code, p)).or_insert(next);
            }
        }
        let n_groups = codes.iter().max().map_or(0, |&m| m as usize + 1);
        let mut counts = vec![0usize; n_groups + 1];
        for &code in &codes {
            counts[code as usize + 1] += 1;
        }
        for g in 0..n_groups {
            counts[g + 1] += counts[g];
        }
        let mut fill = counts.clone();
        let mut positions = vec![0usize; n];
        for (i, &code) in codes.iter().enumerate() {
            positions[fill[code as usize]] = i;
            fill[code as usize] += 1;
        }
        let groups = Arc::new(Groups { offsets: counts, positions });
        self.groups.lock().unwrap().insert(id, groups.clone());
        Ok(groups)
    }
}

/// Per row a code equal for equal values (nulls one code, NaN one code).
fn factorize(column: &Column) -> Vec<u64> {
    fn codes<T: std::hash::Hash + Eq, I: Iterator<Item = T>>(values: I) -> Vec<u64> {
        let mut seen: HashMap<T, u64> = HashMap::new();
        values
            .map(|v| {
                let next = seen.len() as u64;
                *seen.entry(v).or_insert(next)
            })
            .collect()
    }
    fn float(x: Option<f64>) -> Option<u64> {
        x.map(|x| if x.is_nan() { f64::NAN.to_bits() } else { (x + 0.0).to_bits() })
    }
    match column {
        Column::I64(v) | Column::Ts(v) => codes(v.iter().copied()),
        Column::OptI64(v) | Column::OptTs(v) => codes(v.iter().copied()),
        Column::F64(v) => codes(v.iter().map(|&x| float(Some(x)))),
        Column::OptF64(v) => codes(v.iter().map(|&x| float(x))),
        Column::Str(v) => codes(v.iter().map(|s| Some(s.as_str()))),
        Column::OptStr(v) => codes(v.iter().map(|s| s.as_deref())),
        Column::Bool(v) => codes(v.iter().copied()),
        Column::OptBool(v) => codes(v.iter().copied()),
        Column::OptDate(v) => codes(v.iter().copied()),
    }
}

/// Every column of the tables `names` from `data_dir` (the leakage check holds them in
/// memory to corrupt them).
pub fn load_tables(data_dir: impl AsRef<Path>, names: &[&str]) -> Result<HashMap<String, Table>> {
    check_files_allowed()?;
    let mut out = HashMap::new();
    for name in names {
        table_spec(name).ok_or_else(|| unknown(name))?;
        let path = data_dir.as_ref().join(format!("{name}.parquet"));
        if !path.exists() {
            bail!("table {name:?} has not been built ({})", path.display());
        }
        let columns = parquet_columns(&path)?;
        let columns: Vec<&str> = columns.iter().map(String::as_str).collect();
        out.insert(name.to_string(), read_parquet(&path, Some(&columns))?);
    }
    Ok(out)
}

/// Loads `data/` tables once (lazily, selected columns) and hands out deadline views.
pub struct DataStore {
    source: Source,
    tables: Mutex<HashMap<&'static str, Arc<Stored>>>,
}

impl DataStore {
    /// A store over the Parquet tables in `data_dir`.
    pub fn open(data_dir: impl AsRef<Path>) -> Result<DataStore> {
        check_files_allowed()?;
        Ok(DataStore {
            source: Source::Dir(data_dir.as_ref().to_path_buf()),
            tables: Mutex::new(HashMap::new()),
        })
    }

    /// A store over in-memory tables (tests, the leakage check); only those tables exist.
    pub fn from_tables(tables: HashMap<String, Table>) -> Result<DataStore> {
        for name in tables.keys() {
            if table_spec(name).is_none() {
                return Err(unknown(name));
            }
        }
        Ok(DataStore {
            source: Source::Memory(Arc::new(tables)),
            tables: Mutex::new(HashMap::new()),
        })
    }

    pub fn as_of(&self, deadline: Micros) -> AsOfView<'_> {
        AsOfView { store: self, deadline }
    }

    /// The data directory of a file-backed store.
    pub fn data_dir(&self) -> Option<&Path> {
        match &self.source {
            Source::Dir(dir) => Some(dir),
            Source::Memory(_) => None,
        }
    }

    fn table(&self, name: &str) -> Result<Arc<Stored>> {
        let spec = table_spec(name).ok_or_else(|| unknown(name))?;
        if let Some(stored) = self.tables.lock().unwrap().get(spec.name) {
            return Ok(stored.clone());
        }
        let stored = Arc::new(Stored::open(spec, &self.source)?);
        self.tables.lock().unwrap().insert(spec.name, stored.clone());
        Ok(stored)
    }
}

/// The data as known strictly before `deadline` (µs since the epoch, UTC).
#[derive(Clone, Copy)]
pub struct AsOfView<'a> {
    store: &'a DataStore,
    deadline: Micros,
}

impl<'a> AsOfView<'a> {
    pub fn deadline(&self) -> Micros {
        self.deadline
    }

    /// The same store as of `cutoff` (walk-forward fits); a cutoff after this view's deadline
    /// fails, so nothing derived from a view sees more than the view.
    pub fn earlier(&self, cutoff: Micros) -> Result<AsOfView<'a>> {
        if cutoff > self.deadline {
            bail!("cutoff {cutoff} is after the view's deadline {}", self.deadline);
        }
        Ok(AsOfView { store: self.store, deadline: cutoff })
    }

    /// Rows with available_at < deadline, in available_at order (all columns when `columns`
    /// is None). Not for static tables (use `lookup`).
    pub fn table(&self, name: &str, columns: Option<&[&str]>) -> Result<Table> {
        if let Some(spec) = table_spec(name)
            && spec.kind == Kind::Static
        {
            bail!(
                "{name:?} is a static table (not as-of: it covers every season); use \
                 lookup({name:?}, keys) for its identity columns"
            );
        }
        let table = self.store.table(name)?;
        let k = table.visible(self.deadline);
        match columns {
            Some(columns) => table.prefix(columns, k),
            None => {
                let all: Vec<&str> = table.columns.iter().map(String::as_str).collect();
                table.prefix(&all, k)
            }
        }
    }

    /// Static tables only: the `public_columns` of the rows whose key is in `keys` and is
    /// visible as of the deadline (appears in a visible row of the spec's `visible_via`
    /// table); one row per key (the first), sorted by key. Other keys are dropped.
    pub fn lookup(&self, name: &str, keys: &[i64]) -> Result<Table> {
        let spec = table_spec(name).ok_or_else(|| unknown(name))?;
        let (Kind::Static, Some((via_table, via_column))) = (spec.kind, spec.visible_via) else {
            bail!("lookup() is for static tables; {name:?} is {}", spec.kind.name());
        };
        let visible = self.table(via_table, Some(&[via_column]))?;
        let visible: BTreeSet<i64> = visible.opt_i64(via_column)?.into_iter().flatten().collect();
        let wanted: BTreeSet<i64> = keys.iter().copied().filter(|k| visible.contains(k)).collect();
        let table = self.store.table(name)?;
        let key = spec.key[0];
        let all = table.prefix(&[key], table.n())?;
        let mut first: HashMap<i64, usize> = HashMap::new();
        for (i, k) in all.opt_i64(key)?.into_iter().enumerate() {
            if let Some(k) = k
                && wanted.contains(&k)
            {
                first.entry(k).or_insert(i);
            }
        }
        let mut rows: Vec<(i64, usize)> = first.into_iter().collect();
        rows.sort_unstable();
        let rows: Vec<usize> = rows.into_iter().map(|(_, i)| i).collect();
        table.frame(spec.public_columns, &rows)
    }

    /// Snapshot tables only: per `by` group (nulls are a group; `by` empty = one group) the
    /// row of the newest snapshot before the deadline, an 'fpl' row winning a tie. One row
    /// per group, sorted by `by`; columns = `by` then `columns` (default: all).
    pub fn latest(&self, name: &str, by: &[&str], columns: Option<&[&str]>) -> Result<Table> {
        let table = self.store.table(name)?;
        if table.spec.kind != Kind::Snapshot {
            bail!("latest() is for snapshot tables; {name:?} is {}", table.spec.kind.name());
        }
        let wanted: Vec<&str> = match columns {
            Some(columns) => columns.to_vec(),
            None => table.columns.iter().map(String::as_str).collect(),
        };
        let mut out_columns: Vec<&str> = Vec::new();
        for c in by.iter().chain(&wanted) {
            if !out_columns.contains(c) {
                out_columns.push(c);
            }
        }
        let k = table.visible(self.deadline);
        let groups = table.groups(by)?;
        let mut rows = Vec::new();
        for g in 0..groups.offsets.len().saturating_sub(1) {
            let positions = &groups.positions[groups.offsets[g]..groups.offsets[g + 1]];
            let below = positions.partition_point(|&p| p < k);
            if below > 0 {
                rows.push(positions[below - 1]);
            }
        }
        rows.sort_unstable();
        let out = table.frame(&out_columns, &rows)?;
        if by.is_empty() { Ok(out) } else { out.sorted(by) }
    }

    /// The as-of fixture list of `season`: the newest `fixture_snapshot` of that season before
    /// the deadline (schedule_source 'snapshot'; GW index from `gameweek`), else the final
    /// `schedule` ('final'). Sorted by fixture_key.
    pub fn schedule(&self, season: i64) -> Result<Table> {
        let newest = self.latest("fixture_snapshot", &["season"], Some(&["snapshot_at"]))?;
        let at = newest
            .opt_i64("season")?
            .iter()
            .zip(newest.opt_i64("snapshot_at")?)
            .find(|(s, _)| **s == Some(season))
            .and_then(|(_, at)| at);
        let out = match at {
            None => {
                let all = self.table("schedule", Some(SCHEDULE_COLUMNS))?;
                let rows: Vec<usize> = all
                    .opt_i64("season")?
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| **s == Some(season))
                    .map(|(i, _)| i)
                    .collect();
                let mut out = Table::new();
                for (name, column) in all.columns {
                    out.push(&name, column.take(&rows).normalized());
                }
                out
            }
            Some(at) => {
                // Visible rows are sorted by available_at (= snapshot_at): the snapshot is a
                // block.
                let snapshots = self.store.table("fixture_snapshot")?;
                let k = snapshots.visible(self.deadline);
                let visible = &snapshots.available[..k];
                let lo = visible.partition_point(|&t| t < at);
                let hi = visible.partition_point(|&t| t <= at);
                let columns = [
                    "fixture_key",
                    "season",
                    "gw",
                    "kickoff_time",
                    "home_team_key",
                    "away_team_key",
                    "snapshot_at",
                ];
                let block = snapshots.frame(&columns, &(lo..hi).collect::<Vec<_>>())?;
                let seasons = block.opt_i64("season")?;
                let times = block.opt_i64("snapshot_at")?;
                let rows: Vec<usize> = (0..block.height())
                    .filter(|&i| seasons[i] == Some(season) && times[i] == Some(at))
                    .collect();
                let gameweeks = self.table("gameweek", Some(&["season", "gw", "gw_index"]))?;
                let mut index: HashMap<i64, i64> = HashMap::new();
                for ((s, gw), gi) in gameweeks
                    .i64("season")?
                    .iter()
                    .zip(gameweeks.i64("gw")?)
                    .zip(gameweeks.i64("gw_index")?)
                {
                    if *s == season {
                        index.insert(*gw, *gi);
                    }
                }
                let gws = block.opt_i64("gw")?;
                let gw_index: Vec<Option<i64>> =
                    rows.iter().map(|&i| gws[i].and_then(|gw| index.get(&gw).copied())).collect();
                let pick = |name: &str| -> Result<Column> {
                    Ok(block.column(name)?.take(&rows).normalized())
                };
                Table::new()
                    .with("fixture_key", pick("fixture_key")?)
                    .with("season", pick("season")?)
                    .with("gw", pick("gw")?)
                    .with("gw_index", Column::OptI64(gw_index).normalized())
                    .with("kickoff_time", pick("kickoff_time")?)
                    .with("home_team_key", pick("home_team_key")?)
                    .with("away_team_key", pick("away_team_key")?)
                    .with("schedule_source", Column::Str(vec!["snapshot".to_string(); rows.len()]))
            }
        };
        out.sorted(&["fixture_key"])
    }

    /// (season, gw) of the gameweek whose deadline is this view's deadline.
    pub fn gameweek_for_deadline(&self) -> Result<(i64, i64)> {
        let gameweeks = self.table("gameweek", Some(&["season", "gw", "deadline_time"]))?;
        let deadlines = gameweeks.opt_i64("deadline_time")?;
        let matches: Vec<usize> =
            (0..gameweeks.height()).filter(|&i| deadlines[i] == Some(self.deadline)).collect();
        if matches.len() != 1 {
            bail!("{} gameweeks have the deadline {}", matches.len(), self.deadline);
        }
        let i = matches[0];
        Ok((gameweeks.i64("season")?[i], gameweeks.i64("gw")?[i]))
    }
}

#[cfg(test)]
mod tests;
