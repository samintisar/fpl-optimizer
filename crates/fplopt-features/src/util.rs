//! Small helpers over `Table` shared by the feature builders.

use fplopt_store::{Column, Table};

/// The rows of `table` at `rows`, in that order.
pub fn take(table: &Table, rows: &[usize]) -> Table {
    table.take(rows)
}

/// The row positions where `keep` holds.
pub fn positions(n: usize, keep: impl Fn(usize) -> bool) -> Vec<usize> {
    (0..n).filter(|&i| keep(i)).collect()
}

/// A float column as pandas writes it: NaN as null.
pub fn floats(values: Vec<f64>) -> Column {
    Column::OptF64(values.into_iter().map(|x| (!x.is_nan()).then_some(x)).collect()).normalized()
}

pub fn opt_floats(values: Vec<Option<f64>>) -> Column {
    Column::OptF64(values.into_iter().map(|x| x.filter(|x| !x.is_nan())).collect()).normalized()
}

pub fn opt_ints(values: Vec<Option<i64>>) -> Column {
    Column::OptI64(values).normalized()
}

pub fn strings(value: &str, n: usize) -> Column {
    Column::Str(vec![value.to_string(); n])
}

/// pandas' grouped float sum (Kahan-compensated, as `group_sum`).
#[derive(Clone, Copy, Default)]
pub struct Kahan {
    sum: f64,
    compensation: f64,
}

impl Kahan {
    pub fn add(&mut self, value: f64) {
        let y = value - self.compensation;
        let t = self.sum + y;
        self.compensation = t - self.sum - y;
        if self.compensation.is_nan() {
            self.compensation = 0.0;
        }
        self.sum = t;
    }

    pub fn value(&self) -> f64 {
        self.sum
    }
}

/// pandas' median of a group (mean of the two middle values for an even count).
pub fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 1 { values[n / 2] } else { (values[n / 2] + values[n / 2 - 1]) / 2.0 }
}

/// Float days between two µs instants, as pandas computes `(a - b) / Timedelta(days=1)` on
/// nanosecond timedeltas.
pub fn days_between(later: i64, earlier: i64) -> f64 {
    ((later - earlier) * 1000) as f64 / 86_400_000_000_000.0
}

/// pandas' `Series.rolling(window, min_periods).mean()`: its online algorithm (a running
/// Kahan sum with separate compensations for added and removed values, the count of
/// negatives, and a run of equal values returned as that value), so the means match
/// pandas' bit for bit rather than a fresh sum per window.
pub fn rolling_mean(values: &[f64], window: usize, min_periods: usize) -> Vec<f64> {
    let (mut nobs, mut neg) = (0usize, 0usize);
    let (mut sum, mut comp_add, mut comp_remove) = (0.0f64, 0.0f64, 0.0f64);
    let mut prev = values.first().copied().unwrap_or(0.0);
    let mut same = 0usize;
    let mut out = Vec::with_capacity(values.len());
    for i in 0..values.len() {
        if i >= window {
            let v = values[i - window];
            if !v.is_nan() {
                nobs -= 1;
                let y = -v - comp_remove;
                let t = sum + y;
                comp_remove = t - sum - y;
                sum = t;
                if v.is_sign_negative() {
                    neg -= 1;
                }
            }
        }
        let v = values[i];
        if !v.is_nan() {
            nobs += 1;
            let y = v - comp_add;
            let t = sum + y;
            comp_add = t - sum - y;
            sum = t;
            if v.is_sign_negative() {
                neg += 1;
            }
            same = if v == prev { same + 1 } else { 1 };
            prev = v;
        }
        out.push(if nobs >= min_periods && nobs > 0 {
            let mean = sum / nobs as f64;
            if same >= nobs {
                prev
            } else if (neg == 0 && mean < 0.0) || (neg == nobs && mean > 0.0) {
                0.0
            } else {
                mean
            }
        } else {
            f64::NAN
        });
    }
    out
}
