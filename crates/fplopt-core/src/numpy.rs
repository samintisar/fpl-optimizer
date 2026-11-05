//! numpy's float reductions where their order shows in the last bits.

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
