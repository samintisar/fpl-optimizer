//! The only LightGBM entry point (Python `fplopt.models.gbm`). Every fit uses `DETERMINISTIC`
//! (seed 0, one thread, `deterministic`, row-wise histograms), so the same rows give the
//! wheel's byte-identical booster (`fplopt-lgbm`). Callers pass rows in a fixed order.

use anyhow::{Result, bail};
use fplopt_core::pyjson::py_float_repr;
use fplopt_lgbm::{Booster, Train};
use fplopt_store::{Column, Table};

/// Learning parameters (Python's `GbmParams`); `DETERMINISTIC` is always added.
#[derive(Clone, Debug, PartialEq)]
pub struct GbmParams {
    pub objective: &'static str,
    pub num_boost_round: usize,
    pub learning_rate: f64,
    pub num_leaves: i64,
    pub min_data_in_leaf: i64,
    pub feature_fraction: f64,
    pub lambda_l2: f64,
}

impl Default for GbmParams {
    fn default() -> Self {
        GbmParams {
            objective: "binary",
            num_boost_round: 200,
            learning_rate: 0.05,
            num_leaves: 15,
            min_data_in_leaf: 200,
            feature_fraction: 1.0,
            lambda_l2: 1.0,
        }
    }
}

impl GbmParams {
    /// `lightgbm_params()` as `lightgbm.basic._param_dict_to_str` writes it (Python's
    /// `str` of each value, in the dict's order).
    fn pairs(&self) -> Vec<(&'static str, String)> {
        vec![
            ("objective", self.objective.to_string()),
            ("learning_rate", py_float_repr(self.learning_rate)),
            ("num_leaves", self.num_leaves.to_string()),
            ("min_data_in_leaf", self.min_data_in_leaf.to_string()),
            ("feature_fraction", py_float_repr(self.feature_fraction)),
            ("lambda_l2", py_float_repr(self.lambda_l2)),
            ("seed", "0".to_string()),
            ("num_threads", "1".to_string()),
            ("deterministic", "True".to_string()),
            ("force_row_wise", "True".to_string()),
            ("verbosity", "-1".to_string()),
        ]
    }

    /// The dataset's parameter string: `{"verbosity": -1}` updated with the training
    /// parameters, then `num_iterations` (what `lightgbm.train` hands `Dataset.construct`).
    pub fn dataset_params(&self) -> String {
        let mut pairs = vec![("verbosity", "-1".to_string())];
        for (k, v) in self.pairs() {
            match pairs.iter_mut().find(|(name, _)| *name == k) {
                Some(slot) => slot.1 = v,
                None => pairs.push((k, v)),
            }
        }
        pairs.push(("num_iterations", self.num_boost_round.to_string()));
        join(&pairs)
    }

    /// The booster's parameter string: the training parameters, then `num_iterations`.
    pub fn booster_params(&self) -> String {
        let mut pairs = self.pairs();
        pairs.push(("num_iterations", self.num_boost_round.to_string()));
        join(&pairs)
    }
}

fn join(pairs: &[(&str, String)]) -> String {
    pairs.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(" ")
}

/// A column as pandas' `to_numpy(dtype="float64")`: nulls as NaN, bools as 0/1.
pub fn float_column(table: &Table, name: &str) -> Result<Vec<f64>> {
    Ok(match table.column(name)? {
        Column::F64(v) => v.clone(),
        Column::OptF64(v) => v.iter().map(|x| x.unwrap_or(f64::NAN)).collect(),
        Column::I64(v) => v.iter().map(|&x| x as f64).collect(),
        Column::OptI64(v) => v.iter().map(|x| x.map_or(f64::NAN, |x| x as f64)).collect(),
        Column::Bool(v) => v.iter().map(|&x| f64::from(u8::from(x))).collect(),
        Column::OptBool(v) => {
            v.iter().map(|x| x.map_or(f64::NAN, |x| f64::from(u8::from(x)))).collect()
        }
        other => bail!("column {name:?} is not numeric ({})", other.kind()),
    })
}

/// `frame[features].to_numpy(dtype="float64")`, row-major.
fn matrix(table: &Table, features: &[&str]) -> Result<Vec<f64>> {
    let columns: Vec<Vec<f64>> =
        features.iter().map(|f| float_column(table, f)).collect::<Result<_>>()?;
    let n = table.height();
    let mut out = Vec::with_capacity(n * features.len());
    for i in 0..n {
        for column in &columns {
            out.push(column[i]);
        }
    }
    Ok(out)
}

/// A fitted booster and the feature columns it reads, in order.
pub struct Gbm {
    pub booster: Booster,
    pub features: Vec<&'static str>,
}

impl Gbm {
    pub fn predict(&self, table: &Table) -> Result<Vec<f64>> {
        self.booster.predict(&matrix(table, &self.features)?)
    }
}

/// `train(frame, features, label, weight, params)`: fit on `table`'s rows in their order.
pub fn train(
    table: &Table,
    features: &[&'static str],
    label: &str,
    weight: Option<&str>,
    params: &GbmParams,
) -> Result<Gbm> {
    let x = matrix(table, features)?;
    let label = float_column(table, label)?;
    let weight = weight.map(|w| float_column(table, w)).transpose()?;
    let booster = Booster::train(&Train {
        x: &x,
        ncol: features.len(),
        label: &label,
        weight: weight.as_deref(),
        feature_names: features,
        dataset_params: &params.dataset_params(),
        booster_params: &params.booster_params(),
        rounds: params.num_boost_round,
    })?;
    Ok(Gbm { booster, features: features.to_vec() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameter_strings_are_pythons() {
        let p = GbmParams::default();
        assert_eq!(
            p.dataset_params(),
            "verbosity=-1 objective=binary learning_rate=0.05 num_leaves=15 min_data_in_leaf=200 \
             feature_fraction=1.0 lambda_l2=1.0 seed=0 num_threads=1 deterministic=True \
             force_row_wise=True num_iterations=200"
        );
        assert_eq!(
            p.booster_params(),
            "objective=binary learning_rate=0.05 num_leaves=15 min_data_in_leaf=200 \
             feature_fraction=1.0 lambda_l2=1.0 seed=0 num_threads=1 deterministic=True \
             force_row_wise=True verbosity=-1 num_iterations=200"
        );
    }
}
