//! v1's calibration, applied (Python `fplopt.models.calibration`): isotonic maps of P(start),
//! the team's P(CS) and P(goal ≥ 1) per horizon group, and a per-position linear map of the
//! per-fixture xP. The maps come from `calibration_table.json` (Python's
//! `calibration_table`, exported by tests/test_rust_golden.py): the entry for a season was
//! fitted on the seasons before it; a later season uses the last entry, an earlier one none.
//! Fitting the maps stays in Python (`dev/calibrate_v1.py`).

use std::sync::LazyLock;

use anyhow::{Context, Result};
use serde_json::Value;

/// The V1Params fingerprint the table was fitted with (`CALIBRATION_PARAMS`).
pub static TABLE: LazyLock<Value> = LazyLock::new(|| {
    serde_json::from_str(include_str!("../calibration_table.json")).expect("calibration_table.json")
});

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

/// The calibration applied in `season`: the table's entry for it, the last entry for a
/// later season, None before the first.
pub fn calibration_for(season: i64) -> Result<Option<Calibration>> {
    let table = TABLE["table"].as_array().context("calibration table: no 'table'")?;
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
        assert!(calibration_for(2016).unwrap().is_none());
        assert!(calibration_for(2026).unwrap().unwrap().p_start.is_some());
    }
}
