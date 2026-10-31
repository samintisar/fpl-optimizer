//! Corrupt-the-future check (Python `fplopt.features.leakcheck`, the feature part; PLAN §4).
//!
//! For a deadline every builder is computed three times, each from its own `DataStore`, in a
//! per-deadline random order:
//! - **clean**: the tables as they are;
//! - **corrupted** (`corrupt_future`): every row with `available_at >= deadline` gets random
//!   values of its column's type in every column except `available_at` (drawn partly from the
//!   column's own values, so corrupted keys collide with real past keys; nulls included),
//!   plus scrambled copies of random past rows moved to or after the deadline (keeping their
//!   key more often than not, so they become the newest row of it), and the future rows stored
//!   interleaved with the past ones;
//! - **truncated** (`truncate_future`): the future rows deleted.
//!
//! All three must give identical frames (names, column types, nulls, values bit for bit). The
//! builders run inside `files_blocked()`. Past rows are never modified and keep their order.
//!
//! The random draws are this port's own (a seeded xoshiro256**), not numpy's: the check is a
//! property of each implementation, so the corruption need not match Python's; the edge
//! deadlines are the same, the sampled ones differ.

use std::collections::{BTreeSet, HashMap};
use std::time::Instant;

use anyhow::{Result, bail};
use fplopt_store::{Column, DataStore, Micros, Table, files_blocked, table_spec};

use crate::FeatureBuilder;

const AVAILABLE: &str = "available_at";
/// Ghosts per table: this share of the past rows, at most MAX_GHOSTS.
const GHOST_FRACTION: f64 = 0.01;
const MAX_GHOSTS: usize = 2_000;
const GHOST_HORIZON: Micros = 400 * 86_400_000_000;
/// Corrupted cells drawn from the column's own values: from this many rows.
const POOL_SIZE: usize = 4096;
/// A corrupted cell is a fresh random value with this probability, else one of the column's.
const P_FRESH: f64 = 0.45;
const P_NULL: f64 = 0.1;
/// A ghost cell keeps the past row's value with this probability (key columns / others).
const P_KEEP_KEY: f64 = 0.75;
const P_KEEP: f64 = 0.4;
pub const DEFAULT_DEADLINES: usize = 12;
pub const RUNS: [&str; 3] = ["clean", "corrupted", "truncated"];
/// The tables the feature builders read: what `check leakage` loads.
pub const FEATURE_TABLES: &[&str] = &[
    "gameweek",
    "schedule",
    "fixture_snapshot",
    "player_snapshot",
    "player_gw",
    "player_match",
    "player_season",
    "team_rating",
    "odds_snapshot",
    "espn_player_match",
    "espn_match",
    "espn_map",
    "club_match",
];
/// (season, gw) always checked when present: the 2019/20 COVID restart (GW39) and its last GW
/// (GW47), 2021/22 GW18 (COVID postponements) and 2022/23 GW8 (after the cancelled GW7).
pub const EDGE_GAMEWEEKS: &[(i64, i64)] = &[(2019, 39), (2019, 47), (2021, 18), (2022, 8)];

#[derive(Clone, Debug)]
pub struct Leak {
    pub deadline: Micros,
    pub feature: String,
    /// 'clean' (the builder failed on the real data), 'corrupted' or 'truncated'.
    pub variant: &'static str,
    pub detail: String,
}

/// xoshiro256** seeded through splitmix64.
pub struct Rng([u64; 4]);

impl Rng {
    pub fn new(seeds: &[u64]) -> Rng {
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        for s in seeds {
            x = x.rotate_left(17) ^ s.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        }
        let mut next = || {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        Rng([next(), next(), next(), next()])
    }

    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.0;
        let out = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        out
    }

    /// Uniform in [0, 1).
    pub fn random(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in [0, n).
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }

    /// Uniform in [lo, hi].
    pub fn between(&mut self, lo: i64, hi: i64) -> i64 {
        let span = (hi as i128 - lo as i128 + 1).max(1) as u128;
        (lo as i128 + (self.next_u64() as u128 % span) as i128) as i64
    }
}

fn name_seed(name: &str) -> u64 {
    name.bytes()
        .fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3))
}

fn available(table: &Table, name: &str) -> Result<Vec<Micros>> {
    match table.column(AVAILABLE)? {
        Column::Ts(v) => Ok(v.clone()),
        other => bail!("{name}.{AVAILABLE} must be non-null datetimes, got {}", other.kind()),
    }
}

/// `table` without its rows with available_at >= deadline.
fn truncate_table(table: &Table, name: &str, deadline: Micros) -> Result<Table> {
    let at = available(table, name)?;
    let rows: Vec<usize> = (0..at.len()).filter(|&i| at[i] < deadline).collect();
    Ok(table.take(&rows))
}

/// `tables` without their rows with available_at >= deadline.
pub fn truncate_future(
    tables: &HashMap<String, Table>,
    deadline: Micros,
) -> Result<HashMap<String, Table>> {
    tables.iter().map(|(name, t)| Ok((name.clone(), truncate_table(t, name, deadline)?))).collect()
}

/// A cell value of any column type (None: null).
#[derive(Clone)]
enum Cell {
    Int(Option<i64>),
    Float(Option<f64>),
    Text(Option<String>),
    Flag(Option<bool>),
    Date(Option<i32>),
}

fn cell(column: &Column, i: usize) -> Cell {
    match column {
        Column::I64(v) | Column::Ts(v) => Cell::Int(Some(v[i])),
        Column::OptI64(v) | Column::OptTs(v) => Cell::Int(v[i]),
        Column::F64(v) => Cell::Float(Some(v[i])),
        Column::OptF64(v) => Cell::Float(v[i]),
        Column::Str(v) => Cell::Text(Some(v[i].clone())),
        Column::OptStr(v) => Cell::Text(v[i].clone()),
        Column::Bool(v) => Cell::Flag(Some(v[i])),
        Column::OptBool(v) => Cell::Flag(v[i]),
        Column::OptDate(v) => Cell::Date(v[i]),
    }
}

/// `column` with `cells` appended, as the nullable variant of its type.
fn extend(column: &Column, cells: Vec<Cell>) -> Column {
    macro_rules! grow {
        ($variant:ident, $values:expr, $pat:ident) => {{
            let mut out: Vec<_> = $values;
            out.extend(cells.into_iter().map(|c| match c {
                Cell::$pat(x) => x,
                _ => unreachable!("a cell of another type"),
            }));
            Column::$variant(out)
        }};
    }
    match column {
        Column::I64(v) => grow!(OptI64, v.iter().map(|&x| Some(x)).collect(), Int),
        Column::OptI64(v) => grow!(OptI64, v.clone(), Int),
        Column::Ts(v) => grow!(OptTs, v.iter().map(|&x| Some(x)).collect(), Int),
        Column::OptTs(v) => grow!(OptTs, v.clone(), Int),
        Column::F64(v) => grow!(OptF64, v.iter().map(|&x| Some(x)).collect(), Float),
        Column::OptF64(v) => grow!(OptF64, v.clone(), Float),
        Column::Str(v) => grow!(OptStr, v.iter().map(|x| Some(x.clone())).collect(), Text),
        Column::OptStr(v) => grow!(OptStr, v.clone(), Text),
        Column::Bool(v) => grow!(OptBool, v.iter().map(|&x| Some(x)).collect(), Flag),
        Column::OptBool(v) => grow!(OptBool, v.clone(), Flag),
        Column::OptDate(v) => grow!(OptDate, v.clone(), Date),
    }
}

/// The range fresh random values of a column are drawn from: its value range widened by its
/// span (at least a minimum) on both sides.
#[derive(Clone, Copy)]
enum Range {
    Int(i64, i64),
    Float(f64, f64),
    Date(i64, i64),
    Other,
}

fn bounds<T: PartialOrd + Copy>(values: impl Iterator<Item = T>, default: (T, T)) -> (T, T) {
    let mut out: Option<(T, T)> = None;
    for v in values {
        out = Some(match out {
            None => (v, v),
            Some((lo, hi)) => (if v < lo { v } else { lo }, if v > hi { v } else { hi }),
        });
    }
    out.unwrap_or(default)
}

fn range(column: &Column) -> Range {
    const YEAR: i64 = 365 * 86_400_000_000;
    const TIMES: (i64, i64) = (946_684_800_000_000, 1_893_456_000_000_000);
    let ints = |(lo, hi): (i64, i64), minimum: i64| {
        let span = (hi - lo).max(minimum);
        Range::Int(lo.saturating_sub(span), hi.saturating_add(span))
    };
    match column {
        Column::I64(v) => ints(bounds(v.iter().copied(), (0, 100)), 10),
        Column::OptI64(v) => ints(bounds(v.iter().flatten().copied(), (0, 100)), 10),
        Column::Ts(v) => ints(bounds(v.iter().copied(), TIMES), YEAR),
        Column::OptTs(v) => ints(bounds(v.iter().flatten().copied(), TIMES), YEAR),
        Column::F64(_) | Column::OptF64(_) => {
            let values: Box<dyn Iterator<Item = f64>> = match column {
                Column::F64(v) => Box::new(v.iter().copied()),
                Column::OptF64(v) => Box::new(v.iter().flatten().copied()),
                _ => unreachable!(),
            };
            let (lo, hi) = bounds(values.filter(|x| !x.is_nan()), (0.0, 1.0));
            let span = (hi - lo).max(1.0);
            Range::Float(lo - span, hi + span)
        }
        Column::OptDate(v) => {
            let (lo, hi) = bounds(v.iter().flatten().copied(), (10_957, 21_915));
            let span = (hi - lo).max(365);
            Range::Date(i64::from(lo - span), i64::from(hi + span))
        }
        _ => Range::Other,
    }
}

/// A fresh random value of the column's type in `range`.
fn fresh(column: &Column, range: Range, rng: &mut Rng) -> Cell {
    match (column, range) {
        (_, Range::Int(lo, hi)) => Cell::Int(Some(rng.between(lo, hi))),
        (_, Range::Float(lo, hi)) => Cell::Float(Some(lo + rng.random() * (hi - lo))),
        (_, Range::Date(lo, hi)) => Cell::Date(Some(rng.between(lo, hi) as i32)),
        (Column::Str(_) | Column::OptStr(_), _) => {
            const ALPHABET: &[u8] =
                b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_'";
            let len = 1 + rng.below(12);
            Cell::Text(Some(
                (0..len).map(|_| ALPHABET[rng.below(ALPHABET.len())] as char).collect(),
            ))
        }
        _ => Cell::Flag(Some(rng.random() < 0.5)),
    }
}

fn null_of(column: &Column) -> Cell {
    match column {
        Column::I64(_) | Column::OptI64(_) | Column::Ts(_) | Column::OptTs(_) => Cell::Int(None),
        Column::F64(_) | Column::OptF64(_) => Cell::Float(None),
        Column::Str(_) | Column::OptStr(_) => Cell::Text(None),
        Column::Bool(_) | Column::OptBool(_) => Cell::Flag(None),
        Column::OptDate(_) => Cell::Date(None),
    }
}

/// k corrupted cells for `column`: its own values (from POOL_SIZE rows, half of them past
/// rows), fresh random values of its type, or nulls.
fn random_cells(
    column: &Column,
    range: Range,
    k: usize,
    rng: &mut Rng,
    past: &[usize],
) -> Vec<Cell> {
    let n = column.len();
    let mut pool: Vec<usize> = (0..POOL_SIZE).map(|_| rng.below(n)).collect();
    if !past.is_empty() {
        for slot in pool.iter_mut().take(POOL_SIZE / 2) {
            *slot = past[rng.below(past.len())];
        }
    }
    (0..k)
        .map(|_| {
            let draw = rng.random();
            if draw < P_NULL {
                null_of(column)
            } else if n == 0 || draw < P_NULL + P_FRESH {
                fresh(column, range, rng)
            } else {
                cell(column, pool[rng.below(POOL_SIZE)])
            }
        })
        .collect()
}

fn corrupt_table(name: &str, table: &Table, deadline: Micros, rng: &mut Rng) -> Result<Table> {
    let at = available(table, name)?;
    let n = at.len();
    let future: Vec<usize> = (0..n).filter(|&i| at[i] >= deadline).collect();
    let past: Vec<usize> = (0..n).filter(|&i| at[i] < deadline).collect();
    let n_ghosts = if past.is_empty() {
        0
    } else {
        ((GHOST_FRACTION * past.len() as f64).ceil() as usize).clamp(1, MAX_GHOSTS)
    };
    let sources: Vec<usize> = (0..n_ghosts).map(|_| past[rng.below(past.len())]).collect();
    let ghost_at: Vec<Micros> = (0..n_ghosts)
        .map(|_| {
            let close = rng.random();
            if close < 0.1 {
                deadline
            } else if close < 0.3 {
                deadline + rng.between(0, 3_600_000_000 - 1)
            } else {
                deadline + rng.between(0, GHOST_HORIZON - 1)
            }
        })
        .collect();
    let keys: BTreeSet<&str> =
        table_spec(name).map(|s| s.key.iter().copied().collect()).unwrap_or_default();
    let mut out = Table::new();
    for (column_name, column) in &table.columns {
        if column_name == AVAILABLE {
            let mut values = at.clone();
            values.extend(&ghost_at);
            out.push(column_name, Column::Ts(values));
            continue;
        }
        // Future rows: corrupted in place.
        let range = range(column);
        let replaced = random_cells(column, range, future.len(), rng, &past);
        let mut cells: Vec<Cell> = (0..n).map(|i| cell(column, i)).collect();
        for (&i, c) in future.iter().zip(replaced) {
            cells[i] = c;
        }
        // Ghosts: past rows with some cells scrambled; times often moved to the ghost's
        // availability, so the ghost is the newest row of its key.
        let keep = if keys.contains(column_name.as_str()) { P_KEEP_KEY } else { P_KEEP };
        let scrambled = random_cells(column, range, n_ghosts, rng, &past);
        let is_time = matches!(column, Column::Ts(_) | Column::OptTs(_));
        for (g, (&source, scramble)) in sources.iter().zip(scrambled).enumerate() {
            let mut c = if rng.random() >= keep { scramble } else { cell(column, source) };
            if is_time && rng.random() < 0.4 {
                c = Cell::Int(Some(ghost_at[g]));
            }
            cells.push(c);
        }
        out.push(column_name, extend(&empty_like(column), cells));
    }
    // Random storage order in which the past rows keep their relative order.
    let total = n + n_ghosts;
    let mut position: Vec<f64> = (0..total).map(|_| rng.random()).collect();
    let mut past_positions: Vec<f64> = past.iter().map(|&i| position[i]).collect();
    past_positions.sort_by(f64::total_cmp);
    for (&i, p) in past.iter().zip(past_positions) {
        position[i] = p;
    }
    let mut order: Vec<usize> = (0..total).collect();
    order.sort_by(|&a, &b| position[a].total_cmp(&position[b]));
    Ok(out.take(&order))
}

fn empty_like(column: &Column) -> Column {
    match column {
        Column::I64(_) | Column::OptI64(_) => Column::OptI64(Vec::new()),
        Column::Ts(_) | Column::OptTs(_) => Column::OptTs(Vec::new()),
        Column::F64(_) | Column::OptF64(_) => Column::OptF64(Vec::new()),
        Column::Str(_) | Column::OptStr(_) => Column::OptStr(Vec::new()),
        Column::Bool(_) | Column::OptBool(_) => Column::OptBool(Vec::new()),
        Column::OptDate(_) => Column::OptDate(Vec::new()),
    }
}

/// A corrupted copy of `tables` for `deadline` (see the module docs); the same seed gives the
/// same copy.
pub fn corrupt_future(
    tables: &HashMap<String, Table>,
    deadline: Micros,
    seed: u64,
) -> Result<HashMap<String, Table>> {
    tables
        .iter()
        .map(|(name, table)| {
            let mut rng = Rng::new(&[seed, name_seed(name)]);
            Ok((name.clone(), corrupt_table(name, table, deadline, &mut rng)?))
        })
        .collect()
}

/// How two frames differ (None: identical, floats bit for bit with NaN equal to NaN).
pub fn describe_difference(clean: &Table, other: &Table) -> Option<String> {
    if clean.names() != other.names() {
        return Some(format!("columns differ: {:?} vs {:?}", clean.names(), other.names()));
    }
    if clean.height() != other.height() {
        return Some(format!("{} rows vs {}", clean.height(), other.height()));
    }
    let mut differing = Vec::new();
    for ((name, a), (_, b)) in clean.columns.iter().zip(&other.columns) {
        if a.kind() != b.kind() {
            differing.push(format!("{name} (type {} vs {})", a.kind(), b.kind()));
            continue;
        }
        let rows = (0..a.len()).filter(|&i| !same_cell(&cell(a, i), &cell(b, i))).count();
        if rows > 0 {
            differing.push(format!("{name} ({rows} rows)"));
        }
    }
    (!differing.is_empty()).then(|| format!("values differ: {}", differing.join(", ")))
}

fn same_cell(a: &Cell, b: &Cell) -> bool {
    match (a, b) {
        (Cell::Int(x), Cell::Int(y)) => x == y,
        (Cell::Float(x), Cell::Float(y)) => match (x, y) {
            (Some(x), Some(y)) => x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()),
            (None, None) => true,
            _ => false,
        },
        (Cell::Text(x), Cell::Text(y)) => x == y,
        (Cell::Flag(x), Cell::Flag(y)) => x == y,
        (Cell::Date(x), Cell::Date(y)) => x == y,
        _ => false,
    }
}

/// The order in which the clean, corrupted and truncated frames of the i-th deadline are
/// computed: random, deterministic for (seed, i).
pub fn variant_order(seed: u64, i: usize) -> [&'static str; 3] {
    let mut rng = Rng::new(&[seed, i as u64, 0x0DE7]);
    let mut runs = RUNS;
    for j in (1..runs.len()).rev() {
        runs.swap(j, rng.below(j + 1));
    }
    runs
}

type Results = Vec<(&'static str, Result<Table, String>)>;

fn compute(
    store: &DataStore,
    deadline: Micros,
    builders: &[(&'static str, FeatureBuilder)],
) -> Results {
    let view = store.as_of(deadline);
    builders
        .iter()
        .map(|(name, builder)| {
            let result = builder(&view).map_err(|e| {
                let message = e.to_string();
                format!("error: {}", message.lines().next().unwrap_or(""))
            });
            (*name, result)
        })
        .collect()
}

/// For each deadline compute `builders` on the clean, corrupted and truncated tables (each
/// variant from a fresh store; the clean one shared across deadlines), in `variant_order`,
/// with `files_blocked()` active; every difference is a Leak, and so is a builder failing on
/// the clean data (variant 'clean').
pub fn check_leakage(
    tables: HashMap<String, Table>,
    deadlines: &[Micros],
    seed: u64,
    builders: &[(&'static str, FeatureBuilder)],
) -> Result<Vec<Leak>> {
    let clean_store = DataStore::from_tables(tables.clone())?;
    let mut leaks = Vec::new();
    files_blocked(
        "while the leakage check runs (features read only the view)",
        || -> Result<()> {
            for (i, &deadline) in deadlines.iter().enumerate() {
                let started = Instant::now();
                let mut results: HashMap<&str, Results> = HashMap::new();
                for run in variant_order(seed, i) {
                    let computed = match run {
                        "clean" => compute(&clean_store, deadline, builders),
                        "corrupted" => {
                            let store = DataStore::from_tables(corrupt_future(
                                &tables,
                                deadline,
                                seed + i as u64,
                            )?)?;
                            compute(&store, deadline, builders)
                        }
                        _ => {
                            let store =
                                DataStore::from_tables(truncate_future(&tables, deadline)?)?;
                            compute(&store, deadline, builders)
                        }
                    };
                    results.insert(run, computed);
                }
                let before = leaks.len();
                let clean = &results["clean"];
                for (name, result) in clean {
                    if let Err(error) = result {
                        leaks.push(Leak {
                            deadline,
                            feature: name.to_string(),
                            variant: "clean",
                            detail: error.clone(),
                        });
                    }
                }
                for variant in ["corrupted", "truncated"] {
                    for ((name, expected), (_, got)) in clean.iter().zip(&results[variant]) {
                        let detail = match (expected, got) {
                            (Ok(a), Ok(b)) => describe_difference(a, b),
                            (Err(a), Err(b)) if a == b => None,
                            (a, b) => Some(format!("clean: {} | variant: {}", short(a), short(b))),
                        };
                        if let Some(detail) = detail {
                            leaks.push(Leak {
                                deadline,
                                feature: name.to_string(),
                                variant,
                                detail,
                            });
                        }
                    }
                }
                log::info!(
                    "leakage check {}/{} at {}: {} ({:.1} s)",
                    i + 1,
                    deadlines.len(),
                    fmt_micros(deadline),
                    if leaks.len() == before { "ok" } else { "LEAKS" },
                    started.elapsed().as_secs_f64()
                );
            }
            Ok(())
        },
    )?;
    Ok(leaks)
}

fn short(result: &Result<Table, String>) -> String {
    match result {
        Ok(t) => format!("{} rows", t.height()),
        Err(e) => e.clone(),
    }
}

/// An instant as ISO 8601 UTC (log lines).
pub fn fmt_micros(t: Micros) -> String {
    let secs = t.div_euclid(1_000_000);
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    // Civil date from days since the epoch (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}+00:00", rem / 3600, rem % 3600 / 60, rem % 60)
}

// --- real data -----------------------------------------------------------------------------

/// GW deadlines outside `holdout` up to the first one after the newest result (`player_match`
/// available_at): the newest is the live decision. Sorted, unique.
fn candidates(tables: &HashMap<String, Table>, holdout: &[i64]) -> Result<Vec<Micros>> {
    let gameweeks = table(tables, "gameweek")?;
    let newest = available(table(tables, "player_match")?, "player_match")?.into_iter().max();
    let seasons = gameweeks.i64("season")?;
    let mut deadlines: Vec<Micros> = gameweeks
        .opt_i64("deadline_time")?
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !holdout.contains(&seasons[*i]))
        .filter_map(|(_, d)| d)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if let Some(newest) = newest
        && let Some(&first) = deadlines.iter().find(|&&d| d > newest)
    {
        deadlines.retain(|&d| d <= first);
    }
    Ok(deadlines)
}

fn table<'a>(tables: &'a HashMap<String, Table>, name: &str) -> Result<&'a Table> {
    tables.get(name).ok_or_else(|| anyhow::anyhow!("table {name:?} was not loaded"))
}

/// Deadlines every check includes (among the candidates): each season's GW1, EDGE_GAMEWEEKS,
/// the first deadline after player snapshot coverage starts and the newest candidate.
pub fn edge_deadlines(tables: &HashMap<String, Table>, holdout: &[i64]) -> Result<Vec<Micros>> {
    let candidates = candidates(tables, holdout)?;
    let Some(&newest) = candidates.last() else { return Ok(Vec::new()) };
    let gameweeks = table(tables, "gameweek")?;
    let seasons = gameweeks.i64("season")?;
    let gws = gameweeks.i64("gw")?;
    let deadlines = gameweeks.opt_i64("deadline_time")?;
    let mut by_gw: HashMap<(i64, i64), Micros> = HashMap::new();
    let mut picked_seasons = BTreeSet::new();
    for i in 0..gameweeks.height() {
        if let Some(d) = deadlines[i] {
            by_gw.insert((seasons[i], gws[i]), d);
            if candidates.binary_search(&d).is_ok() {
                picked_seasons.insert(seasons[i]);
            }
        }
    }
    let wanted = picked_seasons.iter().map(|&s| (s, 1)).chain(EDGE_GAMEWEEKS.iter().copied());
    let mut picked: BTreeSet<Micros> = wanted.filter_map(|gw| by_gw.get(&gw).copied()).collect();
    if let Some(snapshots) = tables.get("player_snapshot")
        && let Some(start) = snapshots.opt_i64("snapshot_at")?.into_iter().flatten().min()
        && let Some(&after) = candidates.iter().find(|&&d| d > start)
    {
        picked.insert(after);
    }
    picked.insert(newest);
    Ok(picked.into_iter().filter(|d| candidates.binary_search(d).is_ok()).collect())
}

/// `edge_deadlines` plus n more GW deadlines spread over the data: the non-edge candidates
/// split into n consecutive chunks (numpy `array_split` sizes), one random deadline per
/// chunk. Sorted.
pub fn sample_deadlines(
    tables: &HashMap<String, Table>,
    n: usize,
    seed: u64,
    holdout: &[i64],
) -> Result<Vec<Micros>> {
    let candidates = candidates(tables, holdout)?;
    let edges = edge_deadlines(tables, holdout)?;
    let rest: Vec<Micros> = candidates.into_iter().filter(|d| !edges.contains(d)).collect();
    let mut picked = edges;
    if n > 0 && !rest.is_empty() {
        let mut rng = Rng::new(&[seed]);
        let chunks = n.min(rest.len());
        let (size, extra) = (rest.len() / chunks, rest.len() % chunks);
        let mut start = 0;
        for c in 0..chunks {
            let len = size + usize::from(c < extra);
            picked.push(rest[start + rng.below(len)]);
            start += len;
        }
    }
    picked.sort_unstable();
    Ok(picked)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tables() -> HashMap<String, Table> {
        let h = 3_600_000_000i64;
        let rows = Table::new()
            .with("player_key", Column::I64(vec![1, 2, 3, 4]))
            .with("fixture_key", Column::I64(vec![10, 10, 11, 11]))
            .with("name", Column::Str(vec!["a".into(), "b".into(), "c".into(), "d".into()]))
            .with("kickoff_time", Column::Ts(vec![h, 2 * h, 3 * h, 4 * h]))
            .with("available_at", Column::Ts(vec![h, 2 * h, 3 * h, 4 * h]));
        HashMap::from([("player_match".to_string(), rows)])
    }

    #[test]
    fn corruption_keeps_past_rows_and_their_order() {
        let deadline = 3 * 3_600_000_000;
        let corrupted = corrupt_future(&tables(), deadline, 7).unwrap();
        let table = &corrupted["player_match"];
        assert!(table.height() > 4);
        let mut past = truncate_table(table, "player_match", deadline).unwrap();
        for (_, column) in &mut past.columns {
            *column = column.clone().normalized();
        }
        let expected = &truncate_future(&tables(), deadline).unwrap()["player_match"];
        assert_eq!(describe_difference(&past, expected), None);
        assert_eq!(corrupt_future(&tables(), deadline, 7).unwrap()["player_match"], *table);
    }

    fn visible_rows(view: &fplopt_store::AsOfView) -> Result<Table> {
        view.table("player_match", Some(&["player_key", "kickoff_time"]))
    }

    /// Keeps state between calls: its result depends on how often it ran.
    fn stateful(view: &fplopt_store::AsOfView) -> Result<Table> {
        static CALLS: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
        let calls = CALLS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let rows = visible_rows(view)?;
        let n = rows.height();
        Ok(rows.with("calls", Column::I64(vec![calls; n])))
    }

    fn failing(_: &fplopt_store::AsOfView) -> Result<Table> {
        bail!("no data")
    }

    #[test]
    fn honest_builders_pass_and_misbehaving_ones_are_reported() {
        let deadlines = [2 * 3_600_000_000 + 1, 3 * 3_600_000_000];
        let builders: &[(&'static str, FeatureBuilder)] = &[("visible", visible_rows)];
        assert!(check_leakage(tables(), &deadlines, 0, builders).unwrap().is_empty());
        let builders: &[(&'static str, FeatureBuilder)] =
            &[("stateful", stateful), ("failing", failing)];
        let leaks = check_leakage(tables(), &deadlines, 0, builders).unwrap();
        let found: BTreeSet<_> = leaks.iter().map(|l| (l.feature.as_str(), l.variant)).collect();
        assert!(found.contains(&("failing", "clean")));
        assert!(found.iter().any(|(f, v)| *f == "stateful" && *v != "clean"));
    }

    #[test]
    fn variant_orders_are_permutations() {
        for i in 0..20 {
            let mut runs = variant_order(0, i).to_vec();
            runs.sort_unstable();
            assert_eq!(runs, ["clean", "corrupted", "truncated"]);
        }
    }

    #[test]
    fn iso_times() {
        assert_eq!(fmt_micros(1_785_502_800_000_000), "2026-07-31T13:00:00+00:00");
    }
}
