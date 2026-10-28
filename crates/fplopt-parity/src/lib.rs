//! Python vs Rust output comparison (migration plan §3, the parity gate).
//!
//! `diff_tables` matches rows on the table's keys and compares every shared column:
//! integers, strings, dates and booleans must be equal, floats within
//! `atol + rtol * max(|a|, |b|)`. Nulls equal nulls and NaN equals NaN, but a null never
//! equals a NaN (pandas writes NaN as null; the Rust side has to do the same).

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::{Context, Result, bail};
use polars::prelude::*;
use serde::Serialize;

const SUFFIX: &str = "__rust";
const SAMPLES: usize = 5;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Tolerance {
    pub rtol: f64,
    pub atol: f64,
}

impl Default for Tolerance {
    fn default() -> Self {
        Self { rtol: 1e-9, atol: 0.0 }
    }
}

impl Tolerance {
    fn close(&self, a: f64, b: f64) -> bool {
        a == b
            || (a.is_nan() && b.is_nan())
            || (a - b).abs() <= self.atol + self.rtol * a.abs().max(b.abs())
    }
}

#[derive(Debug, Serialize, PartialEq)]
pub struct DtypeMismatch {
    pub column: String,
    pub python: String,
    pub rust: String,
}

#[derive(Debug, Serialize)]
pub struct ColumnDiff {
    pub column: String,
    pub mismatches: usize,
    /// Largest |python - rust| over rows where both are finite (float columns only).
    pub max_abs_diff: Option<f64>,
    /// Up to five mismatching rows: "keys: python | rust".
    pub samples: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct TableDiff {
    pub keys: Vec<String>,
    pub rows_python: usize,
    pub rows_rust: usize,
    pub missing_in_rust: Vec<String>,
    pub missing_in_python: Vec<String>,
    pub dtype_mismatches: Vec<DtypeMismatch>,
    pub duplicate_keys_python: usize,
    pub duplicate_keys_rust: usize,
    pub only_in_python: usize,
    pub only_in_rust: usize,
    pub sample_only_in_python: Vec<String>,
    pub sample_only_in_rust: Vec<String>,
    /// Shared non-key columns with at least one mismatch (empty when values agree).
    pub columns: Vec<ColumnDiff>,
    pub columns_compared: usize,
    pub rows_compared: usize,
}

impl TableDiff {
    pub fn passed(&self) -> bool {
        self.rows_python == self.rows_rust
            && self.missing_in_rust.is_empty()
            && self.missing_in_python.is_empty()
            && self.dtype_mismatches.is_empty()
            && self.duplicate_keys_python == 0
            && self.duplicate_keys_rust == 0
            && self.only_in_python == 0
            && self.only_in_rust == 0
            && self.columns.is_empty()
    }
}

pub fn read_parquet(path: &Path) -> Result<DataFrame> {
    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    ParquetReader::new(file)
        .finish()
        .map_err(anyhow::Error::from)
        .with_context(|| format!("read {}", path.display()))
}

/// Compare `python` (the reference) with `rust`, rows matched on `keys`.
pub fn diff_tables(
    python: &DataFrame,
    rust: &DataFrame,
    keys: &[String],
    tol: Tolerance,
) -> Result<TableDiff> {
    let py_schema = python.schema();
    let rs_schema = rust.schema();
    for key in keys {
        if py_schema.get(key).is_none() || rs_schema.get(key).is_none() {
            bail!("key column {key:?} is missing from one of the tables");
        }
    }
    let names = |s: &Schema| s.iter_names().map(|n| n.to_string()).collect::<BTreeSet<_>>();
    let (py_cols, rs_cols) = (names(py_schema), names(rs_schema));
    let shared: Vec<String> = python
        .get_column_names()
        .iter()
        .map(|n| n.to_string())
        .filter(|n| rs_cols.contains(n))
        .collect();
    let dtype_mismatches = shared
        .iter()
        .filter_map(|c| {
            let (p, r) = (py_schema.get(c).unwrap(), rs_schema.get(c).unwrap());
            (p != r).then(|| DtypeMismatch {
                column: c.clone(),
                python: p.to_string(),
                rust: r.to_string(),
            })
        })
        .collect();

    let mut diff = TableDiff {
        keys: keys.to_vec(),
        rows_python: python.height(),
        rows_rust: rust.height(),
        missing_in_rust: py_cols.difference(&rs_cols).cloned().collect(),
        missing_in_python: rs_cols.difference(&py_cols).cloned().collect(),
        dtype_mismatches,
        duplicate_keys_python: duplicate_key_rows(python, keys)?,
        duplicate_keys_rust: duplicate_key_rows(rust, keys)?,
        only_in_python: 0,
        only_in_rust: 0,
        sample_only_in_python: Vec::new(),
        sample_only_in_rust: Vec::new(),
        columns: Vec::new(),
        columns_compared: 0,
        rows_compared: 0,
    };
    // Rust keys cast to Python's dtypes so a dtype mismatch on a key doesn't hide every row.
    let rust = cast_keys(rust, python, keys)?;
    let (only_py, only_rs) = (anti_join(python, &rust, keys)?, anti_join(&rust, python, keys)?);
    diff.only_in_python = only_py.height();
    diff.only_in_rust = only_rs.height();
    diff.sample_only_in_python = key_samples(&only_py, keys)?;
    diff.sample_only_in_rust = key_samples(&only_rs, keys)?;
    if diff.duplicate_keys_python > 0 || diff.duplicate_keys_rust > 0 {
        return Ok(diff); // row matching is ambiguous: don't compare values
    }

    let values: Vec<&String> = shared.iter().filter(|c| !keys.contains(c)).collect();
    let mut rust_side: Vec<Expr> = keys.iter().map(|k| col(k.as_str())).collect();
    rust_side.extend(values.iter().map(|c| col(c.as_str()).alias(format!("{c}{SUFFIX}"))));
    let joined = python
        .clone()
        .lazy()
        .select(
            keys.iter().chain(values.iter().copied()).map(|c| col(c.as_str())).collect::<Vec<_>>(),
        )
        .join(
            rust.lazy().select(rust_side),
            key_exprs(keys),
            key_exprs(keys),
            join_args(JoinType::Inner),
        )
        .collect()?;
    diff.rows_compared = joined.height();
    diff.columns_compared = values.len();
    for column in values {
        let a = joined.column(column)?.as_materialized_series().clone();
        let b = joined.column(&format!("{column}{SUFFIX}"))?.as_materialized_series().clone();
        let column_diff = compare_column(column, &a, &b, &joined, keys, tol)?;
        if column_diff.mismatches > 0 {
            diff.columns.push(column_diff);
        }
    }
    Ok(diff)
}

fn compare_column(
    name: &str,
    a: &Series,
    b: &Series,
    frame: &DataFrame,
    keys: &[String],
    tol: Tolerance,
) -> Result<ColumnDiff> {
    let mut bad: Vec<usize> = Vec::new();
    let mut max_abs_diff = None;
    if a.dtype().is_float() || b.dtype().is_float() {
        let a64 = a.cast(&DataType::Float64)?;
        let b64 = b.cast(&DataType::Float64)?;
        let mut max: f64 = 0.0;
        for (i, (x, y)) in a64.f64()?.iter().zip(b64.f64()?.iter()).enumerate() {
            let same = match (x, y) {
                (None, None) => true,
                (Some(x), Some(y)) => {
                    if x.is_finite() && y.is_finite() {
                        max = max.max((x - y).abs());
                    }
                    tol.close(x, y)
                }
                _ => false,
            };
            if !same {
                bad.push(i);
            }
        }
        max_abs_diff = Some(max);
    } else {
        let b = if b.dtype() == a.dtype() { b.clone() } else { b.cast(a.dtype())? };
        let equal = a.equal_missing(&b)?;
        bad.extend(equal.iter().enumerate().filter(|(_, e)| *e != Some(true)).map(|(i, _)| i));
    }
    let mut samples = Vec::new();
    for &i in bad.iter().take(SAMPLES) {
        samples.push(format!("{}: {} | {}", row_keys(frame, keys, i)?, a.get(i)?, b.get(i)?));
    }
    Ok(ColumnDiff { column: name.to_owned(), mismatches: bad.len(), max_abs_diff, samples })
}

fn key_exprs(keys: &[String]) -> Vec<Expr> {
    keys.iter().map(|k| col(k.as_str())).collect()
}

fn join_args(how: JoinType) -> JoinArgs {
    let mut args = JoinArgs::new(how);
    args.nulls_equal = true;
    args
}

fn cast_keys(rust: &DataFrame, python: &DataFrame, keys: &[String]) -> Result<DataFrame> {
    let mut rust = rust.clone();
    for key in keys {
        let target = python.schema().get(key).unwrap().clone();
        let column = rust.column(key)?;
        if column.dtype() != &target {
            let cast = column.as_materialized_series().cast(&target)?;
            rust.with_column(cast.into())?;
        }
    }
    Ok(rust)
}

fn anti_join(left: &DataFrame, right: &DataFrame, keys: &[String]) -> Result<DataFrame> {
    let right_keys =
        right.clone().lazy().select(key_exprs(keys)).unique(None, UniqueKeepStrategy::First);
    Ok(left
        .clone()
        .lazy()
        .select(key_exprs(keys))
        .join(right_keys, key_exprs(keys), key_exprs(keys), join_args(JoinType::Anti))
        .collect()?)
}

/// Rows whose key appears more than once.
fn duplicate_key_rows(frame: &DataFrame, keys: &[String]) -> Result<usize> {
    let counts = frame
        .clone()
        .lazy()
        .group_by(key_exprs(keys))
        .agg([len().alias("__n")])
        .filter(col("__n").gt(lit(1)))
        .select([col("__n").sum()])
        .collect()?;
    Ok(counts.column("__n")?.get(0)?.extract::<u64>().unwrap_or(0) as usize)
}

fn key_samples(frame: &DataFrame, keys: &[String]) -> Result<Vec<String>> {
    (0..frame.height().min(SAMPLES)).map(|i| row_keys(frame, keys, i)).collect()
}

fn row_keys(frame: &DataFrame, keys: &[String], i: usize) -> Result<String> {
    let parts: Vec<String> = keys
        .iter()
        .map(|k| Ok(format!("{k}={}", frame.column(k)?.get(i)?)))
        .collect::<Result<_>>()?;
    Ok(parts.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    fn base() -> DataFrame {
        df!(
            "season" => [2024i64, 2024, 2025],
            "player_key" => [1i64, 2, 1],
            "xp" => [Some(1.5f64), None, Some(f64::NAN)],
            "name" => ["a", "b", "c"],
        )
        .unwrap()
    }

    #[test]
    fn identical_tables_pass_in_any_row_order() {
        let python = base();
        let rust = python.sort(["player_key", "season"], SortMultipleOptions::default()).unwrap();
        let diff =
            diff_tables(&python, &rust, &keys(&["season", "player_key"]), Tolerance::default())
                .unwrap();
        assert!(diff.passed(), "{diff:?}");
        assert_eq!((diff.rows_compared, diff.columns_compared), (3, 2));
    }

    #[test]
    fn floats_within_tolerance_pass_and_beyond_fail() {
        let python = base();
        let near = df!("season" => [2024i64, 2024, 2025], "player_key" => [1i64, 2, 1],
            "xp" => [Some(1.5 + 1e-12), None, Some(f64::NAN)], "name" => ["a", "b", "c"])
        .unwrap();
        let k = keys(&["season", "player_key"]);
        assert!(diff_tables(&python, &near, &k, Tolerance::default()).unwrap().passed());
        let far = df!("season" => [2024i64, 2024, 2025], "player_key" => [1i64, 2, 1],
            "xp" => [Some(1.6), None, None], "name" => ["a", "b", "c"])
        .unwrap();
        let diff = diff_tables(&python, &far, &k, Tolerance::default()).unwrap();
        assert!(!diff.passed());
        let xp = &diff.columns[0];
        // 1.5 vs 1.6, and NaN vs null (never equal).
        assert_eq!((xp.column.as_str(), xp.mismatches), ("xp", 2));
        assert!((xp.max_abs_diff.unwrap() - 0.1).abs() < 1e-12);
        assert!(xp.samples[0].starts_with("season=2024, player_key=1:"), "{:?}", xp.samples);
    }

    #[test]
    fn missing_rows_columns_and_dtypes_are_reported() {
        let python = base();
        let rust = df!("season" => [2024i32, 2024], "player_key" => [1i64, 2],
            "xp" => [Some(1.5f64), None], "extra" => [1i64, 2])
        .unwrap();
        let diff =
            diff_tables(&python, &rust, &keys(&["season", "player_key"]), Tolerance::default())
                .unwrap();
        assert!(!diff.passed());
        assert_eq!(diff.missing_in_rust, ["name"]);
        assert_eq!(diff.missing_in_python, ["extra"]);
        assert_eq!(diff.dtype_mismatches[0].column, "season");
        assert_eq!((diff.only_in_python, diff.only_in_rust), (1, 0));
        assert_eq!(diff.sample_only_in_python, ["season=2025, player_key=1"]);
        assert!(diff.columns.is_empty(), "{:?}", diff.columns); // matched rows agree
    }

    #[test]
    fn timezone_aware_datetime_keys() {
        let utc = DataType::Datetime(TimeUnit::Microseconds, Some(TimeZone::UTC));
        let at = Series::new("at".into(), [1_700_000_000_000_000i64, 1_700_000_100_000_000])
            .cast(&utc)
            .unwrap();
        let python =
            DataFrame::new_infer_height(vec![at.into(), Column::new("v".into(), [1.0f64, 2.0])])
                .unwrap();
        let diff =
            diff_tables(&python, &python.clone(), &keys(&["at"]), Tolerance::default()).unwrap();
        assert!(diff.passed(), "{diff:?}");
    }

    #[test]
    fn null_keys_match_each_other() {
        let python = df!("k" => [Some(1i64), None], "v" => ["x", "y"]).unwrap();
        let diff =
            diff_tables(&python, &python.clone(), &keys(&["k"]), Tolerance::default()).unwrap();
        assert!(diff.passed(), "{diff:?}");
    }

    #[test]
    fn duplicate_keys_fail_without_comparing_values() {
        let python = df!("k" => [1i64, 1, 2], "v" => ["x", "y", "z"]).unwrap();
        let rust = df!("k" => [1i64, 2], "v" => ["x", "q"]).unwrap();
        let diff = diff_tables(&python, &rust, &keys(&["k"]), Tolerance::default()).unwrap();
        assert_eq!((diff.duplicate_keys_python, diff.duplicate_keys_rust), (2, 0));
        assert!(!diff.passed() && diff.columns.is_empty());
    }

    #[test]
    fn unknown_key_is_an_error() {
        assert!(diff_tables(&base(), &base(), &keys(&["nope"]), Tolerance::default()).is_err());
    }
}
