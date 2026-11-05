//! The backtester (Python `fplopt.backtest`; migration batch 7): gameweek scoring with
//! autosubs, decisions applied under the FPL rules, xG scoring, the season simulator and the
//! policies. Money is int tenths of £m; ties are broken by `player_key` everywhere, as in
//! Python, so runs are deterministic and match Python's.

#![allow(clippy::type_complexity, clippy::excessive_precision)]

pub mod bench;
pub mod chip_values;
pub mod eval;
pub mod evaluate;
pub mod gw_score;
pub mod outcomes;
pub mod policies;
pub mod probes;
pub mod pyrepr;
pub mod simulator;
pub mod specs;
pub mod state;
pub mod stats;
pub mod xp_cache;
