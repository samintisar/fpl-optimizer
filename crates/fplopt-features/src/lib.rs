//! Point-in-time features (migration batch 3; Python `fplopt.features`). Every builder takes
//! one argument, an `AsOfView`, and reads only through it (PLAN §4); this crate depends on
//! `fplopt-store` only, so nothing here can reach `data/` another way.

// Builders key maps by tuples of table columns; naming each tuple type would not help.
#![allow(clippy::type_complexity)]

pub mod bans;
pub mod baseline;
pub mod history;
pub mod leakcheck;
mod util;

use anyhow::Result;
use fplopt_store::{AsOfView, Table};

pub type FeatureBuilder = fn(&AsOfView) -> Result<Table>;

/// The registered features, in Python's `FEATURES` order.
pub const FEATURES: &[(&str, FeatureBuilder)] = &[
    ("player_pool", baseline::player_pool),
    ("availability", baseline::availability),
    ("ep_next", baseline::ep_next),
    ("recent_form", baseline::recent_form),
    ("upcoming_fixtures", baseline::upcoming_fixtures),
    ("team_strength", baseline::team_strength),
    ("pool_coverage", baseline::pool_coverage),
];

/// Every registered feature plus the minutes model's history frames: what the batch 3 parity
/// gate compares (`fplopt-rs features dump`).
pub const FRAMES: &[(&str, FeatureBuilder)] = &[
    ("player_pool", baseline::player_pool),
    ("availability", baseline::availability),
    ("ep_next", baseline::ep_next),
    ("recent_form", baseline::recent_form),
    ("upcoming_fixtures", baseline::upcoming_fixtures),
    ("team_strength", baseline::team_strength),
    ("pool_coverage", baseline::pool_coverage),
    ("history_training", history::training_frame),
    ("history_prediction", history::prediction_frame),
];

/// Every registered feature at the view's deadline, in FEATURES order.
pub fn compute_features(view: &AsOfView) -> Result<Vec<(&'static str, Table)>> {
    FEATURES.iter().map(|(name, builder)| Ok((*name, builder(view)?))).collect()
}
