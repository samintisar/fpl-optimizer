//! Float helpers that reproduce Python's and pandas' arithmetic.

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

/// Python's built-in `sum()` of floats (3.12+: Neumaier's compensated sum, the compensation
/// added at the end when it is finite and non-zero).
pub fn py_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let (mut sum, mut compensation) = (0.0f64, 0.0f64);
    for x in values {
        let t = sum + x;
        if sum.abs() >= x.abs() {
            compensation += (sum - t) + x;
        } else {
            compensation += (x - t) + sum;
        }
        sum = t;
    }
    if compensation != 0.0 && compensation.is_finite() { sum + compensation } else { sum }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_sum() {
        // sum([0.1] * 10) == 1.0 in Python 3.12 (0.9999999999999999 with plain addition)
        assert_eq!(py_sum([0.1; 10]), 1.0);
        assert_eq!(py_sum([1e100, 1.0, -1e100]), 1.0);
        assert_eq!(py_sum([]), 0.0);
        assert_eq!(py_sum([f64::INFINITY, 1.0]), f64::INFINITY);
    }
}
