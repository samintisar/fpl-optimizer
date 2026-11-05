//! The v1 family's calibration, applied (Python `fplopt.models.calibration`): isotonic maps
//! of P(start), the team's P(CS) and P(goal ≥ 1) per horizon group, and a per-position linear
//! map of the per-fixture xP. The entry for a season was fitted on the seasons before it; a
//! later season uses the last entry, an earlier one none. `v1-2026-10`'s maps are
//! `calibration_table_v1-2026-10.json` (Python's `calibration_table`, exported by
//! tests/test_rust_golden.py); `v1`'s are `calibration_table.json`, P(start) only, written by
//! `fplopt-rs models calibrate fit --write` (the ported `isotonic`) and fingerprinted with
//! `V1Params::fingerprint`.

use std::sync::LazyLock;

use anyhow::{Context, Result, bail};
use serde_json::Value;

use crate::assemble::V1Params;

/// At most this many equal-count bins go into the PAV fit.
pub const MAX_BINS: usize = 200;
/// Breakpoints are rounded to this many decimals in the table.
const DIGITS: usize = 6;

/// v1's table (`fplopt-rs models calibrate fit --model v1 --write`).
pub static TABLE: LazyLock<Value> = LazyLock::new(|| {
    serde_json::from_str(include_str!("../calibration_table.json")).expect("calibration_table.json")
});

/// `v1-2026-10`'s table: Python's, with its `CALIBRATION_PARAMS` fingerprint.
pub static TABLE_2026_10: LazyLock<Value> = LazyLock::new(|| {
    serde_json::from_str(include_str!("../calibration_table_v1-2026-10.json"))
        .expect("calibration_table_v1-2026-10.json")
});

/// The table of `model`: None if a v1-family model has none (yet), an error for any other name.
pub fn table_for(model: &str) -> Result<Option<&'static Value>> {
    match model {
        "v1" | "v1r" => Ok(Some(&TABLE)),
        "v1-2026-10" => Ok(Some(&TABLE_2026_10)),
        other => bail!("not a v1-family model: {other:?}"),
    }
}

/// A non-decreasing map given by breakpoints.
#[derive(Clone, Debug, Default)]
pub struct Isotonic {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

impl Isotonic {
    /// `np.interp(value, x, y)` (flat beyond the ends), a value ≤ 0 (or NaN) unchanged.
    pub fn apply(&self, value: f64) -> f64 {
        // NaN and values ≤ 0 stay as they are (`np.where(values > 0, mapped, values)`).
        if self.x.is_empty() || value.is_nan() || value <= 0.0 {
            return value;
        }
        interp(value, &self.x, &self.y)
    }
}

/// numpy's `interp` for one value.
fn interp(value: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    if value.is_nan() {
        return value;
    }
    if value > xp[n - 1] {
        return fp[n - 1];
    }
    if value < xp[0] {
        return fp[0];
    }
    let j = xp.partition_point(|&x| x <= value).saturating_sub(1);
    if j == n - 1 || xp[j] == value {
        return fp[j];
    }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    let mut result = slope * (value - xp[j]) + fp[j];
    if result.is_nan() {
        result = slope * (value - xp[j + 1]) + fp[j + 1];
        if result.is_nan() && fp[j] == fp[j + 1] {
            result = fp[j];
        }
    }
    result
}

/// The maps applied in one season (None = not recalibrated): isotonic parts as (first
/// horizon, map) ascending; `xp` as (element_type, a, b).
#[derive(Clone, Debug, Default)]
pub struct Calibration {
    pub p_start: Option<Vec<(i64, Isotonic)>>,
    pub p_cs: Option<Vec<(i64, Isotonic)>>,
    pub p_goal: Option<Vec<(i64, Isotonic)>>,
    pub xp: Option<Vec<(i64, f64, f64)>>,
}

/// `value` mapped by the map of its horizon group.
pub fn apply_isotonic(maps: &[(i64, Isotonic)], value: f64, horizon: i64) -> f64 {
    for (i, (first, curve)) in maps.iter().enumerate() {
        let last = maps.get(i + 1).map_or(i64::MAX, |m| m.0);
        if horizon >= *first && horizon < last {
            return curve.apply(value);
        }
    }
    value
}

fn floats(value: &Value) -> Result<Vec<f64>> {
    value
        .as_array()
        .context("not a list")?
        .iter()
        .map(|v| v.as_f64().context("not a number"))
        .collect()
}

fn maps(value: &Value) -> Result<Vec<(i64, Isotonic)>> {
    value
        .as_array()
        .context("isotonic part: not a list")?
        .iter()
        .map(|m| {
            let m = m.as_array().context("isotonic map: not a list")?;
            Ok((
                m[0].as_i64().context("first horizon")?,
                Isotonic { x: floats(&m[1])?, y: floats(&m[2])? },
            ))
        })
        .collect()
}

/// Python's `round(v, DIGITS)`: correctly rounded from the exact binary value, ties to even
/// (Rust's fixed-precision formatting rounds the same way).
fn round_digits(v: f64) -> f64 {
    format!("{v:.DIGITS$}").parse().expect("a formatted float parses")
}

/// Weighted pool-adjacent-violators: the non-decreasing fit of `y` (in order) per point
/// (Python `_pav`, the same merge order).
fn pav(y: &[f64], w: &[f64]) -> Vec<f64> {
    let (mut means, mut weights, mut sizes): (Vec<f64>, Vec<f64>, Vec<usize>) =
        (Vec::new(), Vec::new(), Vec::new());
    for (&value, &weight) in y.iter().zip(w) {
        means.push(value);
        weights.push(weight);
        sizes.push(1);
        while means.len() > 1 && means[means.len() - 2] > means[means.len() - 1] {
            let (m1, w1, s1) = (means.pop().unwrap(), weights.pop().unwrap(), sizes.pop().unwrap());
            let k = means.len() - 1;
            let total = weights[k] + w1;
            means[k] = (means[k] * weights[k] + m1 * w1) / total;
            weights[k] = total;
            sizes[k] += s1;
        }
    }
    means.iter().zip(&sizes).flat_map(|(&m, &s)| std::iter::repeat_n(m, s)).collect()
}

/// `np.unique(np.linspace(0, n, min(bins, n) + 1).astype(int64))`: linspace's points are
/// k · (n / m) with the last set to n exactly; the cast truncates.
fn bin_edges(n: usize, bins: usize) -> Vec<usize> {
    let m = bins.min(n);
    let step = n as f64 / m as f64;
    let mut edges: Vec<usize> = (0..m).map(|k| (k as f64 * step) as usize).collect();
    edges.push(n);
    edges.dedup();
    edges
}

/// The isotonic map of `outcome` on `pred` (Python `isotonic`): rows with pred ≤ 0, a
/// non-finite value or weight ≤ 0 left out (zeros are never mapped); the rest sorted by pred
/// (stable, ties keep their row order), pooled into at most `bins` consecutive equal-count
/// bins (weighted mean pred and outcome), PAV on the bins, one breakpoint per PAV block (its
/// weighted mean pred and fitted value), rounded to `DIGITS`. Sums run in row order, as
/// numpy's `bincount`.
pub fn isotonic(pred: &[f64], outcome: &[f64], weight: Option<&[f64]>, bins: usize) -> Isotonic {
    let ones = vec![1.0; pred.len()];
    let weight = weight.unwrap_or(&ones);
    let mut rows: Vec<usize> = (0..pred.len())
        .filter(|&i| {
            pred[i] > 0.0 && pred[i].is_finite() && outcome[i].is_finite() && weight[i] > 0.0
        })
        .collect();
    if rows.is_empty() {
        return Isotonic::default();
    }
    rows.sort_by(|&a, &b| pred[a].total_cmp(&pred[b])); // stable, as kind="mergesort"
    let edges = bin_edges(rows.len(), bins);
    let n_bins = edges.len() - 1;
    let (mut w, mut wx, mut wy) = (vec![0.0; n_bins], vec![0.0; n_bins], vec![0.0; n_bins]);
    let mut code = 0;
    for (pos, &i) in rows.iter().enumerate() {
        // searchsorted(edges, pos, side="right") - 1
        while code + 1 < n_bins && pos >= edges[code + 1] {
            code += 1;
        }
        w[code] += weight[i];
        wx[code] += weight[i] * pred[i];
        wy[code] += weight[i] * outcome[i];
    }
    let x: Vec<f64> = (0..n_bins).map(|k| wx[k] / w[k]).collect();
    let y: Vec<f64> = (0..n_bins).map(|k| wy[k] / w[k]).collect();
    let fitted = pav(&y, &w);
    let (mut bw, mut bx, mut by): (Vec<f64>, Vec<f64>, Vec<f64>) =
        (Vec::new(), Vec::new(), Vec::new());
    for k in 0..n_bins {
        if k == 0 || fitted[k] != fitted[k - 1] {
            bw.push(0.0);
            bx.push(0.0);
            by.push(0.0);
        }
        let b = bw.len() - 1;
        bw[b] += w[k];
        bx[b] += w[k] * x[k];
        by[b] += w[k] * fitted[k];
    }
    Isotonic {
        x: (0..bw.len()).map(|b| round_digits(bx[b] / bw[b])).collect(),
        y: (0..bw.len()).map(|b| round_digits(by[b] / bw[b])).collect(),
    }
}

/// v1's table must have been fitted with v1's current params (`params` is
/// `V1Params::fingerprint`); `v1-2026-10`'s holds Python's fingerprint string and is not
/// checked here.
fn check_params(model: &str, table: &Value) -> Result<()> {
    let params = V1Params::for_model(model)?;
    if params.model == "v1" && table["params"].as_str() != Some(params.fingerprint().as_str()) {
        bail!(
            "the v1 calibration table was fitted with other params than v1's \
             (refit it: fplopt-rs models calibrate walk|fit --model v1)"
        );
    }
    Ok(())
}

/// The calibration applied in `season`: the table's entry for it, the last entry for a
/// later season, None before the first.
pub fn calibration_for(model: &str, season: i64) -> Result<Option<Calibration>> {
    let Some(table) = table_for(model)? else { return Ok(None) };
    check_params(model, table)?;
    let table = table["table"].as_array().context("calibration table: no 'table'")?;
    let mut chosen = None;
    for entry in table {
        if entry[0].as_i64().context("calibration table: a season")? <= season {
            chosen = Some(&entry[1]);
        }
    }
    let Some(parts) = chosen else { return Ok(None) };
    let part = |name: &str| parts.get(name).map(maps).transpose();
    let xp = parts
        .get("xp")
        .map(|v| -> Result<Vec<(i64, f64, f64)>> {
            v.as_array()
                .context("xp map: not a list")?
                .iter()
                .map(|t| {
                    let t = t.as_array().context("xp map row")?;
                    Ok((
                        t[0].as_i64().context("position")?,
                        t[1].as_f64().context("a")?,
                        t[2].as_f64().context("b")?,
                    ))
                })
                .collect()
        })
        .transpose()?;
    Ok(Some(Calibration {
        p_start: part("p_start")?,
        p_cs: part("p_cs")?,
        p_goal: part("p_goal")?,
        xp,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interp_is_numpys() {
        let (x, y) = ([0.1, 0.2, 0.4], [0.0, 0.5, 0.6]);
        assert_eq!(interp(0.05, &x, &y), 0.0);
        assert_eq!(interp(0.5, &x, &y), 0.6);
        assert_eq!(interp(0.2, &x, &y), 0.5);
        assert_eq!(interp(0.4, &x, &y), 0.6);
        assert!((interp(0.3, &x, &y) - 0.55).abs() < 1e-15);
        let map = Isotonic { x: x.to_vec(), y: y.to_vec() };
        assert_eq!(map.apply(0.0), 0.0);
        assert!(calibration_for("v1", 2016).unwrap().is_none());
        assert!(calibration_for("v1", 2026).unwrap().unwrap().p_start.is_some());
    }

    fn golden(name: &str) -> Vec<(f64, f64)> {
        let path =
            format!("{}/../../tests/fixtures/rust_golden/{name}", env!("CARGO_MANIFEST_DIR"));
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|l| {
                let mut it = l.split(',').map(|v| v.parse::<f64>().unwrap());
                (it.next().unwrap(), it.next().unwrap())
            })
            .collect()
    }

    /// Python's `isotonic` on 1000 random rows (tests/fixtures/rust_golden/isotonic_*.csv),
    /// breakpoint for breakpoint.
    #[test]
    fn isotonic_matches_python() {
        let (p, y): (Vec<f64>, Vec<f64>) = golden("isotonic_in.csv").into_iter().unzip();
        let expected = golden("isotonic_out.csv");
        let map = isotonic(&p, &y, None, MAX_BINS);
        assert_eq!(map.x.len(), expected.len());
        for (k, (x, v)) in expected.iter().enumerate() {
            assert_eq!(map.x[k], *x, "x[{k}]");
            assert_eq!(map.y[k], *v, "y[{k}]");
        }
    }

    #[test]
    fn isotonic_skips_zeros_and_is_monotone() {
        let p = [0.0, 0.2, 0.4, 0.6, 0.8, f64::NAN];
        let y = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0];
        let map = isotonic(&p, &y, None, MAX_BINS);
        assert!(!map.x.is_empty() && map.x.iter().all(|&x| x > 0.0));
        assert!(map.y.windows(2).all(|w| w[0] <= w[1]));
        assert!(isotonic(&[0.0], &[1.0], None, MAX_BINS).x.is_empty());
    }

    /// `np.unique(np.linspace(0, n, bins + 1).astype(int64))`: k · (n / bins), not k · n / bins.
    #[test]
    fn bin_edges_are_numpys() {
        assert_eq!(bin_edges(7, 200), (0..=7).collect::<Vec<_>>());
        let e = bin_edges(205, 200);
        assert_eq!(e.len(), 201);
        assert_eq!(&e[115..125], &[117, 118, 119, 120, 121, 122, 124, 125, 126, 127]);
        let e = bin_edges(1000, 200);
        assert_eq!(&e[..4], &[0, 5, 10, 15]);
        assert_eq!(*e.last().unwrap(), 1000);
    }

    /// Python's `round(v, 6)` (correctly rounded, ties to even on the exact binary value).
    #[test]
    fn round_digits_is_pythons_round() {
        let cases = [
            (0.0078125, 0.007812),
            (0.0234375, 0.023438),
            (0.1234565, 0.123456),
            (0.1234575, 0.123457),
            (0.4999995, 0.499999),
            (1.0 / 3.0, 0.333333),
            (2.0 / 3.0, 0.666667),
            (5e-07, 0.0),
            (1.5e-06, 2e-06),
            (2.5e-06, 3e-06),
            (0.123456500000001, 0.123457),
            (1e-07, 0.0),
        ];
        for (v, want) in cases {
            assert_eq!(round_digits(v), want, "round({v}, 6)");
        }
    }

    #[test]
    fn pav_pools_violators() {
        assert_eq!(pav(&[1.0, 3.0, 2.0], &[1.0, 1.0, 1.0]), vec![1.0, 2.5, 2.5]);
        assert_eq!(pav(&[3.0, 1.0], &[1.0, 3.0]), vec![1.5, 1.5]);
    }

    #[test]
    fn v1_tables_must_match_the_params() {
        let good = V1Params::default().fingerprint();
        let table = serde_json::json!({"params": good, "table": []});
        assert!(check_params("v1", &table).is_ok());
        assert!(check_params("v1r", &table).is_ok());
        let stale = serde_json::json!({"params": "V1Params { other }", "table": []});
        assert!(check_params("v1", &stale).is_err());
        // v1-2026-10's table keeps Python's fingerprint string.
        assert!(check_params("v1-2026-10", &stale).is_ok());
        // The embedded tables pass.
        assert!(check_params("v1", &TABLE).is_ok());
    }
}
