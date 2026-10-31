//! Small helpers over `Table` shared by the models.

use anyhow::{Result, bail};
pub use fplopt_core::numpy::np_sum;
use fplopt_store::{Column, Micros, Table};

/// A timestamp column with nulls as None (a column without nulls is stored as `Ts`).
pub fn opt_ts(table: &Table, name: &str) -> Result<Vec<Option<Micros>>> {
    match table.column(name)? {
        Column::Ts(v) => Ok(v.iter().copied().map(Some).collect()),
        Column::OptTs(v) => Ok(v.clone()),
        other => bail!("{name}: expected timestamps, got {}", other.kind()),
    }
}

/// `Series.mean()` (NaN for no values).
pub fn np_mean(values: &[f64]) -> f64 {
    np_sum(values) / values.len() as f64
}

/// pandas' `Series.mean()` with NaN: NaN counts as 0 in numpy's sum, divided by the number
/// of non-NaN values (NaN when there are none).
pub fn pd_mean(values: &[f64]) -> f64 {
    let filled: Vec<f64> = values.iter().map(|v| if v.is_nan() { 0.0 } else { *v }).collect();
    let count = values.iter().filter(|v| !v.is_nan()).count();
    if count == 0 { f64::NAN } else { np_sum(&filled) / count as f64 }
}

/// pandas' `Series.sum()`: NaN counts as 0 in numpy's sum.
pub fn pd_sum(values: &[f64]) -> f64 {
    let filled: Vec<f64> = values.iter().map(|v| if v.is_nan() { 0.0 } else { *v }).collect();
    np_sum(&filled)
}

/// `np.minimum` (NaN propagates; `f64::min` would drop it).
pub fn np_minimum(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() { f64::NAN } else { a.min(b) }
}

/// `np.maximum` (NaN propagates).
pub fn np_maximum(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() { f64::NAN } else { a.max(b) }
}

/// `np.clip(x, lo, hi)` (NaN propagates).
pub fn np_clip(x: f64, lo: f64, hi: f64) -> f64 {
    np_minimum(np_maximum(x, lo), hi)
}

/// scipy's `expit`: 1 / (1 + exp(−x)).
pub fn expit(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// scipy's `logit`: log(x / (1 − x)), with log1p(s) − log1p(−s), s = 2(x − ½), on
/// [0.3, 0.65].
pub fn logit(x: f64) -> f64 {
    if !(0.3..=0.65).contains(&x) {
        (x / (1.0 - x)).ln()
    } else {
        let s = 2.0 * (x - 0.5);
        s.ln_1p() - (-s).ln_1p()
    }
}
