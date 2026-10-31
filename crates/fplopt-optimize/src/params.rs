//! Planner settings (Python `fplopt.optimize.params`; PLAN §7 *Objective*, *Defaults*). The
//! defaults and their reasons are documented there; this port keeps the same values and
//! validation.

use std::collections::BTreeMap;

use anyhow::{Result, bail};

pub const DEFAULT_FT_VALUE: [(i64, f64); 4] = [(2, 2.0), (3, 1.6), (4, 1.3), (5, 1.1)];
pub const DEFAULT_BENCH_WEIGHTS: [f64; 4] = [0.03, 0.21, 0.06, 0.002];
/// Terminal value (points, undecayed) of an unused chip whose window outlasts the horizon.
pub const DEFAULT_CHIP_VALUE: [(&str, f64); 4] =
    [("wildcard", 6.0), ("freehit", 4.0), ("bboost", 4.0), ("3xc", 3.0)];
/// Candidates kept per position by horizon xP and by horizon xP per price.
pub const DEFAULT_PRUNE_N: [(i64, i64); 4] = [(1, 20), (2, 60), (3, 60), (4, 30)];
pub const DEFAULT_NODE_LIMIT: i64 = 20_000;
pub const DEFAULT_TIE_EPSILON: f64 = 1e-4;
pub const DEFAULT_MIN_MINUTES: f64 = 180.0;
/// FPL's chip names (Python `backtest.rules.CHIP_NAMES`).
pub const CHIP_NAMES: [&str; 4] = ["wildcard", "freehit", "bboost", "3xc"];

/// Planner settings (Python `OptimizerParams`). Maps are key-sorted. `prune_n` None
/// disables top-N pruning; `max_hits`, `time_limit`, `node_limit` None mean no limit.
#[derive(Clone, Debug, PartialEq)]
pub struct OptimizerParams {
    pub horizon: i64,
    pub decay: f64,
    pub ft_value: BTreeMap<i64, f64>,
    pub itb_value: f64,
    pub hit_margin: f64,
    pub max_hits: Option<i64>,
    pub bench_weights: Vec<f64>,
    pub chip_value: BTreeMap<String, f64>,
    pub prune_n: Option<BTreeMap<i64, i64>>,
    pub prune_dominated: bool,
    pub mip_gap: f64,
    pub threads: i64,
    pub time_limit: Option<f64>,
    pub node_limit: Option<i64>,
    pub tie_epsilon: f64,
    pub bench_from_minutes: bool,
    pub min_minutes: f64,
    /// Per chip name, the terminal value by GWs left in its window after the horizon.
    pub chip_value_left: Option<BTreeMap<String, Vec<f64>>>,
    /// Chip names left out of the chip scenario search (sorted).
    pub exclude_chips: Vec<String>,
    /// The share of a Free Hit's gain over the no-chip plan that counts when chip scenarios
    /// are ranked (`search::score`). 1 = all.
    pub fh_scale: f64,
}

impl Default for OptimizerParams {
    fn default() -> Self {
        OptimizerParams {
            horizon: 6,
            decay: 0.85,
            ft_value: DEFAULT_FT_VALUE.into_iter().collect(),
            itb_value: 0.0,
            hit_margin: 0.0,
            max_hits: Some(0),
            bench_weights: DEFAULT_BENCH_WEIGHTS.to_vec(),
            chip_value: DEFAULT_CHIP_VALUE.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            prune_n: Some(DEFAULT_PRUNE_N.into_iter().collect()),
            prune_dominated: true,
            mip_gap: 0.005,
            threads: 1,
            time_limit: None,
            node_limit: Some(DEFAULT_NODE_LIMIT),
            tie_epsilon: DEFAULT_TIE_EPSILON,
            bench_from_minutes: true,
            min_minutes: DEFAULT_MIN_MINUTES,
            chip_value_left: None,
            exclude_chips: Vec::new(),
            fh_scale: 1.0,
        }
    }
}

impl OptimizerParams {
    /// Python's `__post_init__` checks.
    pub fn validate(&self) -> Result<()> {
        if self.horizon < 1 {
            bail!("horizon must be >= 1, got {}", self.horizon);
        }
        if let Some(h) = self.max_hits
            && h < 0
        {
            bail!("max_hits must be >= 0 or None, got {h}");
        }
        if !(self.decay > 0.0 && self.decay <= 1.0) {
            bail!("decay must be in (0, 1], got {}", self.decay);
        }
        for (name, value) in [
            ("itb_value", self.itb_value),
            ("hit_margin", self.hit_margin),
            ("tie_epsilon", self.tie_epsilon),
            ("min_minutes", self.min_minutes),
        ] {
            if !value.is_finite() || value < 0.0 {
                bail!("{name} must be a finite number >= 0, got {value}");
            }
        }
        if self.ft_value.iter().any(|(k, v)| *k < 1 || *v < 0.0) {
            bail!("ft_value keys must be >= 1 and values >= 0, got {:?}", self.ft_value);
        }
        if self.bench_weights.len() != 4 {
            bail!("bench_weights needs 4 values, got {:?}", self.bench_weights);
        }
        if let Some(table) = &self.chip_value_left {
            for (name, values) in table {
                if values.is_empty() || values.iter().any(|x| !x.is_finite() || *x < 0.0) {
                    bail!("chip_value_left['{name}'] needs finite values >= 0, got {values:?}");
                }
            }
        }
        let unknown: Vec<&String> =
            self.exclude_chips.iter().filter(|c| !CHIP_NAMES.contains(&c.as_str())).collect();
        if !unknown.is_empty() {
            bail!("exclude_chips: unknown chip names {unknown:?}");
        }
        if !self.fh_scale.is_finite() || !(0.0..=1.0).contains(&self.fh_scale) {
            bail!("fh_scale must be in [0, 1], got {}", self.fh_scale);
        }
        if let Some(prune_n) = &self.prune_n
            && prune_n.values().any(|v| *v < 0)
        {
            bail!("prune_n values must be >= 0, got {prune_n:?}");
        }
        if !(0.0..1.0).contains(&self.mip_gap) {
            bail!("mip_gap must be in [0, 1), got {}", self.mip_gap);
        }
        if self.threads < 1 {
            bail!("threads must be >= 1, got {}", self.threads);
        }
        if let Some(t) = self.time_limit
            && (t.is_nan() || t <= 0.0)
        {
            bail!("time_limit must be > 0 or None, got {t}");
        }
        if let Some(n) = self.node_limit
            && n < 1
        {
            bail!("node_limit must be >= 1 or None, got {n}");
        }
        Ok(())
    }

    /// V(s) = Σ_{n ≤ s} ft_value[n]: the value of having s free transfers (summed in key
    /// order from 0, as Python's `sum`).
    pub fn ft_state_value(&self, s: i64) -> f64 {
        let mut total = 0.0;
        for (n, v) in &self.ft_value {
            if *n <= s {
                total += v;
            }
        }
        total
    }
}
