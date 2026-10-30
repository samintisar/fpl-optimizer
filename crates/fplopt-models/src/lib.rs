//! xP models (migration batch 6; Python `fplopt.models`). Every model takes one argument, an
//! `AsOfView`, reads only through it and returns an xP frame: one row per `player_pool`
//! player and GW from the target GW to `HORIZON` GWs later, columns `player_key, gw,
//! gw_index, horizon, xp` (int64 × 4, float64; 0 when unknown), sorted by (player_key,
//! horizon). `v1` adds its per player-GW components after those.

// The numerical ports index like the reference code (MINPACK, scipy, numpy) on purpose, so
// they can be read against it line by line.
#![allow(clippy::needless_range_loop)]
// Models key maps and sorts by tuples of table columns, as the feature builders do.
#![allow(clippy::type_complexity)]

pub mod availability;
pub mod baseline;
pub mod fitted;
pub mod gbm;
pub mod minutes;
pub mod optim;
pub mod shares;
pub mod team;
pub mod util;

use anyhow::{Result, bail};
use fplopt_store::{AsOfView, Table};

pub use baseline::HORIZON;

/// GWs in an xP frame at most: the target GW and `HORIZON` more.
pub const MAX_HORIZON: i64 = HORIZON + 1;

/// The registered models (Python's `MODELS`), by name.
pub const MODELS: &[&str] = &["rolling", "ep_next", "ep_next_fade"];

/// Model components the parity gate checks on their own, fitted at the view's deadline
/// (not the refit cutoff): name → (frame, fitted values).
pub const COMPONENTS: &[&str] = &["team", "minutes", "availability", "shares"];

pub fn component(name: &str, view: &AsOfView) -> Result<(Table, serde_json::Value)> {
    match name {
        "team" => {
            let fit = team::fit_team(view, &team::TeamParams::default())?;
            Ok((team::team_lambdas(view, &fit)?, fit.to_json()))
        }
        "minutes" => {
            let fit = minutes::fit_minutes(view, &minutes::MinutesParams::default(), None)?;
            Ok((minutes::predict_minutes(view, &fit)?.to_table(), fit.to_json()?))
        }
        "availability" => {
            let history = fplopt_features::history::training_frame(view)?;
            let params = minutes::MinutesParams::default();
            let fit = minutes::fit_minutes(view, &params, Some(&history))?;
            let layer = availability::fit_availability(view, &fit, Some(&history), true)?;
            let adjusted =
                availability::adjust_minutes(view, &minutes::predict_minutes(view, &fit)?, &layer)?;
            Ok((adjusted.to_table(), layer.to_json()))
        }
        "shares" => {
            let team_fit = team::fit_team(view, &team::TeamParams::default())?;
            let lambdas = team::team_lambdas(view, &team_fit)?;
            let history = fplopt_features::history::training_frame(view)?;
            let params = minutes::MinutesParams::default();
            let fit = minutes::fit_minutes(view, &params, Some(&history))?;
            let layer = availability::fit_availability(view, &fit, Some(&history), true)?;
            let adjusted =
                availability::adjust_minutes(view, &minutes::predict_minutes(view, &fit)?, &layer)?;
            let shares_fit = shares::fit_shares(view, &shares::SharesParams::default())?;
            let frame = shares::predict_shares(view, &shares_fit, &adjusted, &lambdas)?;
            Ok((frame.to_table(), shares_fit.to_json()))
        }
        other => bail!("unknown component {other:?} (known: {})", COMPONENTS.join(", ")),
    }
}

/// The xP frame of model `name` at the view.
pub fn predict(name: &str, view: &AsOfView) -> Result<Table> {
    match name {
        "rolling" => baseline::xp_rolling(view),
        "ep_next" => baseline::xp_ep_next(view),
        "ep_next_fade" => baseline::xp_ep_next_fade(view),
        other => bail!("unknown xP model {other:?} (known: {})", MODELS.join(", ")),
    }
}
