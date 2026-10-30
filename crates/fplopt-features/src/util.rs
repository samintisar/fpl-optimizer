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
