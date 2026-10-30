//! Float helpers that reproduce pandas' arithmetic.

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
