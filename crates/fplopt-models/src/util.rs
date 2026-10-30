//! Small helpers over `Table` shared by the models.

use anyhow::{Result, bail};
use fplopt_store::{Column, Micros, Table};

/// A timestamp column with nulls as None (a column without nulls is stored as `Ts`).
pub fn opt_ts(table: &Table, name: &str) -> Result<Vec<Option<Micros>>> {
    match table.column(name)? {
        Column::Ts(v) => Ok(v.iter().copied().map(Some).collect()),
        Column::OptTs(v) => Ok(v.clone()),
        other => bail!("{name}: expected timestamps, got {}", other.kind()),
    }
}

/// numpy's float64 sum (`np.sum`, `Series.sum`; `Series.mean` is it over the count): its
/// pairwise summation, 8 accumulators in blocks of up to 128.
pub fn np_sum(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 8 {
        let mut sum = 0.0;
        for &v in values {
            sum += v;
        }
        sum
    } else if n <= 128 {
        let mut r = [0.0; 8];
        r.copy_from_slice(&values[..8]);
        let mut i = 8;
        while i < n - n % 8 {
            for j in 0..8 {
                r[j] += values[i + j];
            }
            i += 8;
        }
        let mut sum = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            sum += values[i];
            i += 1;
        }
        sum
    } else {
        let half = n / 2;
        let half = half - half % 8;
        np_sum(&values[..half]) + np_sum(&values[half..])
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
