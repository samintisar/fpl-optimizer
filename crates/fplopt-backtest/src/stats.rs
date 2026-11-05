//! The paired-comparison statistics (Python `fplopt.backtest.evaluate`: `block_bootstrap`,
//! `season_points`, `season_t_test`, `window_t_test`, `full_run_window_test`,
//! `window_test_null_size`, `deflated_mean`; PLAN §5 *Comparing policies*). A difference
//! frame is a list of `Diff`s: one paired difference per (season, start, gw_index) row, with
//! the weight that turns it into season points (stride/k for overlapping windows).
//! Reductions follow the numpy / pandas ones they port (Kahan group sums, pairwise sums), and
//! the bootstrap draws numpy's generator draw for draw, so CIs match Python's.

use std::collections::BTreeMap;

use fplopt_core::nprng::Generator;
use fplopt_core::numpy::np_sum;

/// One paired-difference row: (season, gw_index) cell, the value (None = null) and its
/// season-points weight.
#[derive(Clone, Copy, Debug)]
pub struct Diff {
    pub season: i64,
    pub gw_index: i64,
    pub value: Option<f64>,
    pub weight: f64,
}

/// pandas' grouped mean (Kahan sum over the count, in row order).
#[derive(Clone, Copy, Default)]
struct KahanMean {
    sum: f64,
    comp: f64,
    n: usize,
}

impl KahanMean {
    fn add(&mut self, v: f64) {
        let y = v - self.comp;
        let t = self.sum + y;
        self.comp = t - self.sum - y;
        self.sum = t;
        self.n += 1;
    }

    fn mean(&self) -> f64 {
        self.sum / self.n as f64
    }

    fn sum(&self) -> f64 {
        self.sum
    }
}

/// Per (season, gw_index), sorted, the mean of the non-null `value × weight`.
pub fn cell_means(diffs: &[Diff], weighted: bool) -> BTreeMap<(i64, i64), f64> {
    let mut cells: BTreeMap<(i64, i64), KahanMean> = BTreeMap::new();
    for d in diffs {
        if let Some(v) = d.value {
            let v = if weighted { v * d.weight } else { v };
            cells.entry((d.season, d.gw_index)).or_default().add(v);
        }
    }
    cells.into_iter().map(|(k, m)| (k, m.mean())).collect()
}

/// numpy's mean (pairwise sum over n; NaN for none).
pub fn np_mean(values: &[f64]) -> f64 {
    np_sum(values) / values.len() as f64
}

/// numpy's `std(ddof=1)`.
pub fn np_std1(values: &[f64]) -> f64 {
    let mean = np_mean(values);
    let squares: Vec<f64> = values.iter().map(|v| (v - mean) * (v - mean)).collect();
    (np_sum(&squares) / (values.len() as f64 - 1.0)).sqrt()
}

/// numpy's `quantile(values, q)` (linear interpolation).
pub fn np_quantile(values: &[f64], q: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    let pos = (n - 1) as f64 * q;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = pos - lo as f64;
    // numpy's _lerp: a + (b - a) * t, flipped to b - (b - a) * (1 - t) for t >= 0.5
    let (a, b) = (sorted[lo], sorted[hi]);
    let diff = b - a;
    if frac >= 0.5 { b - diff * (1.0 - frac) } else { a + diff * frac }
}

/// The block bootstrap's result: mean per cell, percentile CI, one-sided p and the sample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BootstrapResult {
    pub mean: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    pub p_one_sided: f64,
    pub n_seasons: usize,
    pub n_gws: usize,
}

/// GW-block bootstrap clustered by season of the mean of the cells' start-averaged values:
/// each replicate rebuilds every season's series from circular blocks of `block_length`
/// consecutive cells (blocks never cross seasons) and takes the mean over all cells.
pub fn block_bootstrap(
    diffs: &[Diff],
    block_length: usize,
    n_boot: usize,
    seed: u64,
    ci: f64,
) -> BootstrapResult {
    let cells = cell_means(diffs, false);
    if cells.is_empty() {
        let nan = f64::NAN;
        return BootstrapResult {
            mean: nan,
            ci_low: nan,
            ci_high: nan,
            p_one_sided: nan,
            n_seasons: 0,
            n_gws: 0,
        };
    }
    let mut rng = Generator::new(u128::from(seed));
    let mut total = vec![0.0; n_boot];
    let mut by_season: BTreeMap<i64, Vec<f64>> = BTreeMap::new();
    for ((season, _), v) in &cells {
        by_season.entry(*season).or_default().push(*v);
    }
    for series in by_season.values() {
        let n = series.len();
        let n_blocks = n.div_ceil(block_length);
        let starts = rng.integers(0, n as i64, n_boot * n_blocks);
        let mut row = Vec::with_capacity(n_blocks * block_length);
        for (b, slot) in total.iter_mut().enumerate() {
            row.clear();
            for s in &starts[b * n_blocks..(b + 1) * n_blocks] {
                for j in 0..block_length {
                    row.push(series[(*s as usize + j) % n]);
                }
            }
            *slot += np_sum(&row[..n]);
        }
    }
    let n_cells = cells.len() as f64;
    let means: Vec<f64> = total.iter().map(|t| t / n_cells).collect();
    let alpha = (1.0 - ci) / 2.0;
    let values: Vec<f64> = cells.values().copied().collect();
    BootstrapResult {
        mean: np_mean(&values),
        ci_low: np_quantile(&means, alpha),
        ci_high: np_quantile(&means, 1.0 - alpha),
        p_one_sided: means.iter().filter(|m| **m <= 0.0).count() as f64 / n_boot as f64,
        n_seasons: by_season.len(),
        n_gws: cells.len(),
    }
}

/// Per season, the difference in season points: Σ over its cells of the start-averaged
/// weighted value (nulls dropped).
pub fn season_points(diffs: &[Diff]) -> BTreeMap<i64, f64> {
    let mut out: BTreeMap<i64, KahanMean> = BTreeMap::new();
    for ((season, _), v) in cell_means(diffs, true) {
        out.entry(season).or_default().add(v);
    }
    out.into_iter().map(|(s, m)| (s, m.sum())).collect()
}

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    let (tiny, qab, qap, qam) = (1e-300, a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    d = 1.0 / if d.abs() > tiny { d } else { tiny };
    let mut h = d;
    for m in 1..500 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let even = m * (b - m) * x / ((qam + m2) * (a + m2));
        let odd = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        for aa in [even, odd] {
            d = 1.0 + aa * d;
            d = 1.0 / if d.abs() > tiny { d } else { tiny };
            c = 1.0 + aa / c;
            c = if c.abs() > tiny { c } else { tiny };
            h *= d * c;
        }
        if (d * c - 1.0).abs() < 1e-15 {
            break;
        }
    }
    h
}

fn betainc(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let log_front = libm::lgamma(a + b) - libm::lgamma(a) - libm::lgamma(b);
    let front = (log_front + a * x.ln() + b * (-x).ln_1p()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * betacf(a, b, x) / a
    } else {
        1.0 - front * betacf(b, a, 1.0 - x) / b
    }
}

/// P(T > t) for Student's t with `df` degrees of freedom.
pub fn t_sf(t: f64, df: usize) -> f64 {
    let df = df as f64;
    let tail = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t));
    if t >= 0.0 { tail } else { 1.0 - tail }
}

/// One-sided t-test (H1: A > B) on the per-season means of the cells: (p, seasons); p NaN
/// with fewer than 2 seasons or no spread.
pub fn season_t_test(diffs: &[Diff]) -> (f64, usize) {
    let mut seasons: BTreeMap<i64, KahanMean> = BTreeMap::new();
    for ((season, _), v) in cell_means(diffs, false) {
        seasons.entry(season).or_default().add(v);
    }
    let means: Vec<f64> = seasons.values().map(KahanMean::mean).collect();
    let n = means.len();
    if n < 2 {
        return (f64::NAN, n);
    }
    let sd = np_std1(&means);
    if sd == 0.0 {
        return (f64::NAN, n);
    }
    (t_sf(np_mean(&means) / (sd / (n as f64).sqrt()), n - 1), n)
}

/// `window_t_test`: the mean over cells, one-sided p, the cells and the mean's SE.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindowTTest {
    pub mean: f64,
    pub p_one_sided: f64,
    pub n_cells: usize,
    pub se: f64,
}

/// One-sided t-test (H1: A > B) over the (season, gw_index) cells.
pub fn window_t_test(diffs: &[Diff]) -> WindowTTest {
    let cells: Vec<f64> = cell_means(diffs, false).into_values().collect();
    let n = cells.len();
    if n == 0 {
        return WindowTTest { mean: f64::NAN, p_one_sided: f64::NAN, n_cells: 0, se: f64::NAN };
    }
    let mean = np_mean(&cells);
    let sd = if n >= 2 { np_std1(&cells) } else { 0.0 };
    let se = if n >= 2 { sd / (n as f64).sqrt() } else { f64::NAN };
    if n < 2 || sd == 0.0 {
        return WindowTTest { mean, p_one_sided: f64::NAN, n_cells: n, se };
    }
    WindowTTest { mean, p_one_sided: t_sf(mean / se, n - 1), n_cells: n, se }
}

/// GWs per cell of `full_run_window_test` (chip pre-registration §4).
pub const FULL_RUN_WINDOW: i64 = 8;

/// `window_t_test` on full-run per-GW differences grouped into non-overlapping
/// `window`-GW cells.
pub fn full_run_window_test(diffs: &[Diff], window: i64) -> WindowTTest {
    let cells: Vec<Diff> = diffs
        .iter()
        .map(|d| Diff { gw_index: 1 + (d.gw_index - 1).div_euclid(window) * window, ..*d })
        .collect();
    window_t_test(&cells)
}

/// Python's `statistics.NormalDist().inv_cdf(p)` (Wichura's AS241).
pub fn normal_inv_cdf(p: f64) -> f64 {
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180625 - q * q;
        let num = (((((((2.509_080_928_730_122_7e3 * r + 3.343_057_558_358_813e4) * r
            + 6.726_577_092_700_87e4)
            * r
            + 4.592_195_393_154_987e4)
            * r
            + 1.373_169_376_550_946e4)
            * r
            + 1.971_590_950_306_551_3e3)
            * r
            + 1.331_416_678_917_843_8e2)
            * r
            + 3.387_132_872_796_366_5)
            * q;
        let den = ((((((5.226_495_278_852_545e3 * r + 2.872_908_573_572_194_3e4) * r
            + 3.930_789_580_009_271e4)
            * r
            + 2.121_379_430_158_659_7e4)
            * r
            + 5.394_196_021_424_751e3)
            * r
            + 6.871_870_074_920_579e2)
            * r
            + 4.231_333_070_160_091e1)
            * r
            + 1.0;
        return num / den;
    }
    let mut r = if q <= 0.0 { p } else { 1.0 - p };
    r = (-r.ln()).sqrt();
    let (num, den) = if r <= 5.0 {
        r -= 1.6;
        (
            ((((((7.745_450_142_783_414e-4 * r + 2.272_384_498_926_918_4e-2) * r
                + 2.417_807_251_774_506e-1)
                * r
                + 1.270_458_252_452_368_4)
                * r
                + 3.647_848_324_763_204_5)
                * r
                + 5.769_497_221_460_691)
                * r
                + 4.630_337_846_156_545)
                * r
                + 1.423_437_110_749_683_5,
            ((((((1.050_750_071_644_416_9e-9 * r + 5.475_938_084_995_345e-4) * r
                + 1.519_866_656_361_645_7e-2)
                * r
                + 1.481_039_764_274_800_8e-1)
                * r
                + 6.897_673_349_851e-1)
                * r
                + 1.676_384_830_183_803_8)
                * r
                + 2.053_191_626_637_759)
                * r
                + 1.0,
        )
    } else {
        r -= 5.0;
        (
            ((((((2.010_334_399_292_288_1e-7 * r + 2.711_555_568_743_487_6e-5) * r
                + 1.242_660_947_388_078_4e-3)
                * r
                + 2.653_218_952_657_612_4e-2)
                * r
                + 2.965_605_718_285_048_7e-1)
                * r
                + 1.784_826_539_917_291_3)
                * r
                + 5.463_784_911_164_114)
                * r
                + 6.657_904_643_501_103,
            ((((((2.044_263_103_389_939_7e-15 * r + 1.421_511_758_316_445_9e-7) * r
                + 1.846_318_317_510_054_8e-5)
                * r
                + 7.868_691_311_456_133e-4)
                * r
                + 1.487_536_129_085_061_5e-2)
                * r
                + 1.369_298_809_227_358e-1)
                * r
                + 5.998_322_065_558_88e-1)
                * r
                + 1.0,
        )
    };
    let x = num / den;
    if q < 0.0 { -x } else { x }
}

/// The best-of-N haircut: mean − SE·√(2 ln N), SE from the two-sided `ci` interval.
pub fn deflated_mean(mean: f64, ci_low: f64, ci_high: f64, n_variants: usize, ci: f64) -> f64 {
    if n_variants <= 1 {
        return mean;
    }
    let z = normal_inv_cdf((1.0 + ci) / 2.0);
    let se = (ci_high - ci_low) / (2.0 * z);
    mean - se * (2.0 * (n_variants as f64).ln()).sqrt()
}

/// `window_test_null_size`: rejection rates at `alphas` and the calibrated critical p of the
/// full-run window test under no effect (centred cells resampled `n_cells` at a time).
pub fn window_test_null_size(
    diffs: &[Diff],
    window: i64,
    n_cells: usize,
    alphas: &[f64],
    n_rep: usize,
    seed: u64,
) -> anyhow::Result<(usize, Vec<(f64, f64, f64)>)> {
    let grouped: Vec<Diff> = diffs
        .iter()
        .map(|d| Diff { gw_index: 1 + (d.gw_index - 1).div_euclid(window) * window, ..*d })
        .collect();
    let cells: Vec<f64> = cell_means(&grouped, false).into_values().collect();
    if cells.len() < 2 {
        anyhow::bail!("need at least 2 cells");
    }
    let mean = np_mean(&cells);
    let centred: Vec<f64> = cells.iter().map(|c| c - mean).collect();
    let mut rng = Generator::new(u128::from(seed));
    let draws = rng.choice_indices(centred.len(), n_rep * n_cells);
    let ps: Vec<f64> = (0..n_rep)
        .map(|r| {
            let row: Vec<f64> =
                draws[r * n_cells..(r + 1) * n_cells].iter().map(|&i| centred[i]).collect();
            let (m, s) = (np_mean(&row), np_std1(&row));
            if s > 0.0 { t_sf(m / (s / (n_cells as f64).sqrt()), n_cells - 1) } else { 1.0 }
        })
        .collect();
    let out = alphas
        .iter()
        .map(|&a| {
            let rate = ps.iter().filter(|p| **p < a).count() as f64 / ps.len() as f64;
            (a, rate, np_quantile(&ps, a))
        })
        .collect();
    Ok((cells.len(), out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_normal_and_t_tail_match_python() {
        // statistics.NormalDist().inv_cdf(0.9), (0.975), (0.001)
        assert_eq!(normal_inv_cdf(0.9), 1.2815515655446008);
        assert!((normal_inv_cdf(0.975) - 1.9599639845400536).abs() < 1e-15);
        assert_eq!(normal_inv_cdf(0.001), -3.090232306167813);
        // evaluate.t_sf(1.5, 8), t_sf(-0.3, 3)
        assert!((t_sf(1.5, 8) - 0.08600164597595594).abs() < 1e-14);
        assert!((t_sf(-0.3, 3) - 0.6081183539800405).abs() < 1e-14);
    }

    #[test]
    fn quantile_interpolates_as_numpy() {
        let v = [3.0, 1.0, 4.0, 1.0, 5.0];
        assert_eq!(np_quantile(&v, 0.1), 1.0);
        assert_eq!(np_quantile(&v, 0.9), 4.6);
    }
}
