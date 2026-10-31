//! Column-oriented tables: what the builders produce and read back, and their Parquet I/O.
//! Types follow the pandas dtypes the Python build writes (int64, nullable Int64/Float64,
//! str, bool, datetime64[us, UTC], date), so both builds write the same Parquet schema.

use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, bail};
use polars::prelude::*;

/// Microseconds since the epoch, UTC (pandas `datetime64[us, UTC]`).
pub type Micros = i64;
/// Days since the epoch (Arrow date32).
pub type Days = i32;

#[derive(Clone, Debug, PartialEq)]
pub enum Column {
    I64(Vec<i64>),
    OptI64(Vec<Option<i64>>),
    F64(Vec<f64>),
    OptF64(Vec<Option<f64>>),
    Str(Vec<String>),
    OptStr(Vec<Option<String>>),
    Bool(Vec<bool>),
    OptBool(Vec<Option<bool>>),
    Ts(Vec<Micros>),
    OptTs(Vec<Option<Micros>>),
    OptDate(Vec<Option<Days>>),
}

impl Column {
    pub fn len(&self) -> usize {
        match self {
            Column::I64(v) => v.len(),
            Column::OptI64(v) => v.len(),
            Column::F64(v) => v.len(),
            Column::OptF64(v) => v.len(),
            Column::Str(v) => v.len(),
            Column::OptStr(v) => v.len(),
            Column::Bool(v) => v.len(),
            Column::OptBool(v) => v.len(),
            Column::Ts(v) => v.len(),
            Column::OptTs(v) => v.len(),
            Column::OptDate(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The nullable variant's values as the non-null variant when no value is null, so code
    /// reading a slice sees the same type whether the whole column holds nulls or not.
    pub fn normalized(self) -> Column {
        fn all<T>(v: Vec<Option<T>>) -> Result<Vec<T>, Vec<Option<T>>> {
            if v.iter().all(Option::is_some) {
                Ok(v.into_iter().map(Option::unwrap).collect())
            } else {
                Err(v)
            }
        }
        match self {
            Column::OptI64(v) => all(v).map_or_else(Column::OptI64, Column::I64),
            Column::OptF64(v) => all(v).map_or_else(Column::OptF64, Column::F64),
            Column::OptStr(v) => all(v).map_or_else(Column::OptStr, Column::Str),
            Column::OptBool(v) => all(v).map_or_else(Column::OptBool, Column::Bool),
            Column::OptTs(v) => all(v).map_or_else(Column::OptTs, Column::Ts),
            other => other,
        }
    }

    /// The pandas dtype name of the variant (error messages).
    pub fn kind(&self) -> &'static str {
        kind(self)
    }

    fn to_series(&self, name: &str) -> Series {
        let name: PlSmallStr = name.into();
        match self {
            Column::I64(v) => Series::new(name, v),
            Column::OptI64(v) => Series::new(name, v),
            Column::F64(v) => Series::new(name, v),
            Column::OptF64(v) => Series::new(name, v),
            Column::Str(v) => Series::new(name, v),
            Column::OptStr(v) => Series::new(name, v),
            Column::Bool(v) => Series::new(name, v),
            Column::OptBool(v) => Series::new(name, v),
            Column::Ts(v) => Int64Chunked::from_slice(name, v)
                .into_datetime(TimeUnit::Microseconds, Some(TimeZone::UTC))
                .into_series(),
            Column::OptTs(v) => Int64Chunked::from_iter_options(name, v.iter().copied())
                .into_datetime(TimeUnit::Microseconds, Some(TimeZone::UTC))
                .into_series(),
            Column::OptDate(v) => {
                Int32Chunked::from_iter_options(name, v.iter().copied()).into_date().into_series()
            }
        }
    }

    /// The rows at `order`, in that order.
    pub fn take(&self, order: &[usize]) -> Column {
        fn pick<T: Clone>(v: &[T], order: &[usize]) -> Vec<T> {
            order.iter().map(|&i| v[i].clone()).collect()
        }
        match self {
            Column::I64(v) => Column::I64(pick(v, order)),
            Column::OptI64(v) => Column::OptI64(pick(v, order)),
            Column::F64(v) => Column::F64(pick(v, order)),
            Column::OptF64(v) => Column::OptF64(pick(v, order)),
            Column::Str(v) => Column::Str(pick(v, order)),
            Column::OptStr(v) => Column::OptStr(pick(v, order)),
            Column::Bool(v) => Column::Bool(pick(v, order)),
            Column::OptBool(v) => Column::OptBool(pick(v, order)),
            Column::Ts(v) => Column::Ts(pick(v, order)),
            Column::OptTs(v) => Column::OptTs(pick(v, order)),
            Column::OptDate(v) => Column::OptDate(pick(v, order)),
        }
    }
}

/// Named columns of equal length, in schema order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Table {
    pub columns: Vec<(String, Column)>,
}

/// A sort key value; nulls sort last, as pandas' `sort_values` does.
#[derive(Clone, Debug, PartialEq, PartialOrd)]
enum SortKey {
    Int(Option<i64>),
    Float(Option<f64>),
    Text(Option<String>),
}

fn null_last<T: PartialOrd>(a: &Option<T>, b: &Option<T>) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a, b) {
        (Some(x), Some(y)) => x.partial_cmp(y).unwrap_or(Ordering::Equal),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

impl SortKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (SortKey::Int(a), SortKey::Int(b)) => null_last(a, b),
            (SortKey::Float(a), SortKey::Float(b)) => null_last(a, b),
            (SortKey::Text(a), SortKey::Text(b)) => null_last(a, b),
            _ => std::cmp::Ordering::Equal,
        }
    }
}

impl Table {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with(mut self, name: &str, column: Column) -> Self {
        self.columns.push((name.to_string(), column));
        self
    }

    pub fn push(&mut self, name: &str, column: Column) {
        self.columns.push((name.to_string(), column));
    }

    pub fn height(&self) -> usize {
        self.columns.first().map_or(0, |(_, c)| c.len())
    }

    pub fn names(&self) -> Vec<&str> {
        self.columns.iter().map(|(n, _)| n.as_str()).collect()
    }

    pub fn column(&self, name: &str) -> Result<&Column> {
        self.columns
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, c)| c)
            .with_context(|| format!("no column {name:?}"))
    }

    pub fn has(&self, name: &str) -> bool {
        self.columns.iter().any(|(n, _)| n == name)
    }

    pub fn i64(&self, name: &str) -> Result<&[i64]> {
        match self.column(name)? {
            Column::I64(v) => Ok(v),
            Column::Ts(v) => Ok(v),
            other => bail!("column {name:?} is not int64: {:?}", kind(other)),
        }
    }

    pub fn opt_i64(&self, name: &str) -> Result<Vec<Option<i64>>> {
        Ok(match self.column(name)? {
            Column::I64(v) | Column::Ts(v) => v.iter().map(|&x| Some(x)).collect(),
            Column::OptI64(v) | Column::OptTs(v) => v.clone(),
            other => bail!("column {name:?} is not an integer: {:?}", kind(other)),
        })
    }

    pub fn opt_f64(&self, name: &str) -> Result<Vec<Option<f64>>> {
        Ok(match self.column(name)? {
            Column::F64(v) => v.iter().map(|&x| Some(x)).collect(),
            Column::OptF64(v) => v.clone(),
            Column::I64(v) => v.iter().map(|&x| Some(x as f64)).collect(),
            Column::OptI64(v) => v.iter().map(|x| x.map(|x| x as f64)).collect(),
            other => bail!("column {name:?} is not numeric: {:?}", kind(other)),
        })
    }

    pub fn f64(&self, name: &str) -> Result<&[f64]> {
        match self.column(name)? {
            Column::F64(v) => Ok(v),
            other => bail!("column {name:?} is not float64: {:?}", kind(other)),
        }
    }

    pub fn str(&self, name: &str) -> Result<&[String]> {
        match self.column(name)? {
            Column::Str(v) => Ok(v),
            other => bail!("column {name:?} is not a non-null str: {:?}", kind(other)),
        }
    }

    pub fn opt_str(&self, name: &str) -> Result<Vec<Option<String>>> {
        Ok(match self.column(name)? {
            Column::Str(v) => v.iter().map(|s| Some(s.clone())).collect(),
            Column::OptStr(v) => v.clone(),
            other => bail!("column {name:?} is not str: {:?}", kind(other)),
        })
    }

    pub fn bool(&self, name: &str) -> Result<&[bool]> {
        match self.column(name)? {
            Column::Bool(v) => Ok(v),
            other => bail!("column {name:?} is not bool: {:?}", kind(other)),
        }
    }

    pub fn opt_bool(&self, name: &str) -> Result<Vec<Option<bool>>> {
        Ok(match self.column(name)? {
            Column::Bool(v) => v.iter().map(|&x| Some(x)).collect(),
            Column::OptBool(v) => v.clone(),
            other => bail!("column {name:?} is not bool: {:?}", kind(other)),
        })
    }

    pub fn opt_date(&self, name: &str) -> Result<&[Option<Days>]> {
        match self.column(name)? {
            Column::OptDate(v) => Ok(v),
            other => bail!("column {name:?} is not a date: {:?}", kind(other)),
        }
    }

    /// Rows reordered by `keys` (stable; nulls last), as `sort_values(kind="mergesort")`.
    pub fn sorted(&self, keys: &[&str]) -> Result<Table> {
        let mut columns = Vec::new();
        for key in keys {
            let values: Vec<SortKey> = match self.column(key)? {
                Column::I64(v) | Column::Ts(v) => {
                    v.iter().map(|&x| SortKey::Int(Some(x))).collect()
                }
                Column::OptI64(v) | Column::OptTs(v) => {
                    v.iter().map(|&x| SortKey::Int(x)).collect()
                }
                Column::OptDate(v) => v.iter().map(|&x| SortKey::Int(x.map(i64::from))).collect(),
                Column::Bool(v) => v.iter().map(|&x| SortKey::Int(Some(i64::from(x)))).collect(),
                Column::OptBool(v) => v.iter().map(|&x| SortKey::Int(x.map(i64::from))).collect(),
                Column::F64(v) => {
                    v.iter().map(|&x| SortKey::Float((!x.is_nan()).then_some(x))).collect()
                }
                Column::OptF64(v) => {
                    v.iter().map(|&x| SortKey::Float(x.filter(|x| !x.is_nan()))).collect()
                }
                Column::Str(v) => v.iter().map(|x| SortKey::Text(Some(x.clone()))).collect(),
                Column::OptStr(v) => v.iter().map(|x| SortKey::Text(x.clone())).collect(),
            };
            columns.push(values);
        }
        let mut order: Vec<usize> = (0..self.height()).collect();
        order.sort_by(|&a, &b| {
            for values in &columns {
                let ord = values[a].cmp(&values[b]);
                if ord != std::cmp::Ordering::Equal {
                    return ord;
                }
            }
            std::cmp::Ordering::Equal
        });
        Ok(self.take(&order))
    }

    pub fn take(&self, order: &[usize]) -> Table {
        Table { columns: self.columns.iter().map(|(n, c)| (n.clone(), c.take(order))).collect() }
    }

    pub fn to_dataframe(&self) -> Result<DataFrame> {
        let columns: Vec<polars::prelude::Column> =
            self.columns.iter().map(|(n, c)| c.to_series(n).into_column()).collect();
        Ok(DataFrame::new_infer_height(columns)?)
    }

    pub fn from_dataframe(df: &DataFrame) -> Result<Table> {
        let mut table = Table::new();
        for column in df.columns() {
            let series = column.as_materialized_series();
            let name = series.name().as_str();
            let nulls = series.null_count() > 0;
            let col = match series.dtype() {
                DataType::Int64 | DataType::Int32 | DataType::Int16 | DataType::Int8 => {
                    let ca = series.cast(&DataType::Int64)?;
                    let ca = ca.i64()?;
                    if nulls {
                        Column::OptI64(ca.iter().collect())
                    } else {
                        Column::I64(ca.iter().map(Option::unwrap_or_default).collect())
                    }
                }
                DataType::Float64 | DataType::Float32 => {
                    let ca = series.cast(&DataType::Float64)?;
                    let ca = ca.f64()?;
                    if nulls {
                        Column::OptF64(ca.iter().collect())
                    } else {
                        Column::F64(ca.iter().map(Option::unwrap_or_default).collect())
                    }
                }
                DataType::String => {
                    let ca = series.str()?;
                    if nulls {
                        Column::OptStr(ca.iter().map(|s| s.map(str::to_string)).collect())
                    } else {
                        Column::Str(ca.iter().map(|s| s.unwrap_or_default().to_string()).collect())
                    }
                }
                DataType::Boolean => {
                    let ca = series.bool()?;
                    if nulls {
                        Column::OptBool(ca.iter().collect())
                    } else {
                        Column::Bool(ca.iter().map(Option::unwrap_or_default).collect())
                    }
                }
                DataType::Datetime(unit, _) => {
                    let factor = match unit {
                        TimeUnit::Microseconds => 1,
                        TimeUnit::Milliseconds => 1_000,
                        TimeUnit::Nanoseconds => 0,
                    };
                    let physical = series.to_physical_repr();
                    let ca = physical.i64()?;
                    let conv = |x: i64| if factor == 0 { x.div_euclid(1000) } else { x * factor };
                    if nulls {
                        Column::OptTs(ca.iter().map(|x| x.map(conv)).collect())
                    } else {
                        Column::Ts(ca.iter().map(|x| conv(x.unwrap_or_default())).collect())
                    }
                }
                DataType::Date => {
                    let physical = series.to_physical_repr();
                    Column::OptDate(physical.i32()?.iter().collect())
                }
                other => bail!("column {name:?}: unsupported dtype {other}"),
            };
            table.push(name, col);
        }
        Ok(table)
    }
}

fn kind(column: &Column) -> &'static str {
    match column {
        Column::I64(_) => "int64",
        Column::OptI64(_) => "Int64",
        Column::F64(_) => "float64",
        Column::OptF64(_) => "Float64",
        Column::Str(_) => "str",
        Column::OptStr(_) => "str (nullable)",
        Column::Bool(_) => "bool",
        Column::OptBool(_) => "boolean",
        Column::Ts(_) => "datetime",
        Column::OptTs(_) => "datetime (nullable)",
        Column::OptDate(_) => "date",
    }
}

fn writer_options<W: std::io::Write>(writer: ParquetWriter<W>) -> ParquetWriter<W> {
    writer
        .with_compression(ParquetCompression::Zstd(None))
        .with_statistics(StatisticsOptions::full())
}

/// Write `table` to `path` (zstd) through a `.tmp` file renamed into place.
pub fn write_parquet(table: &Table, path: &Path) -> Result<()> {
    let mut df = table.to_dataframe()?;
    let tmp = path.with_extension("parquet.tmp");
    let result = (|| -> Result<()> {
        let file = File::create(&tmp).with_context(|| format!("create {}", tmp.display()))?;
        writer_options(ParquetWriter::new(file)).finish(&mut df)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

/// A Parquet file written one row group per `write` (chunked tables such as
/// `player_snapshot`), renamed into place by `finish`.
pub struct ChunkedWriter {
    writer: Option<polars::io::parquet::write::BatchedWriter<File>>,
    tmp: std::path::PathBuf,
    path: std::path::PathBuf,
}

impl ChunkedWriter {
    pub fn new(path: &Path) -> Self {
        Self { writer: None, tmp: path.with_extension("parquet.tmp"), path: path.to_path_buf() }
    }

    pub fn write(&mut self, table: &Table) -> Result<()> {
        let df = table.to_dataframe()?;
        if self.writer.is_none() {
            let file = File::create(&self.tmp)?;
            self.writer = Some(writer_options(ParquetWriter::new(file)).batched(df.schema())?);
        }
        self.writer.as_mut().unwrap().write_batch(&df)?;
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        if let Some(writer) = self.writer.take() {
            writer.finish()?;
        }
        std::fs::rename(&self.tmp, &self.path)?;
        Ok(())
    }
}

impl Drop for ChunkedWriter {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.tmp);
    }
}

pub fn read_parquet(path: &Path, columns: Option<&[&str]>) -> Result<Table> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut reader = ParquetReader::new(file);
    if let Some(columns) = columns {
        reader = reader.with_columns(Some(columns.iter().map(|c| c.to_string()).collect()));
    }
    let df = anyhow::Context::with_context(reader.finish(), || format!("read {}", path.display()))?;
    Table::from_dataframe(&df)
}

/// The column names of a Parquet file, without pandas' index columns.
pub fn parquet_columns(path: &Path) -> Result<Vec<String>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let schema = anyhow::Context::with_context(ParquetReader::new(file).schema(), || {
        format!("read the schema of {}", path.display())
    })?;
    Ok(schema
        .iter_names()
        .map(|name| name.to_string())
        .filter(|name| !name.starts_with("__index_level_"))
        .collect())
}
