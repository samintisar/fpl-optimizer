//! The MILP transfer planner (migration batch 4; Python `fplopt.optimize`): parameters,
//! planner input, candidate pruning, the minutes model's bench weights, chip scenarios, the
//! model and its plans, solved by HiGHS 1.15.1 (`fplopt-highs`) exactly as Python's PuLP +
//! highspy solve them.
//!
//! `PlanInput::from_context(state, pool, xp, rules, params)` → `optimize(problem, params,
//! top_k, chips, roll, Search::Bound)` → `PlanSet`.

pub mod chips;
pub mod entry;
pub mod io;
pub mod minutes;
pub mod model;
pub mod params;
pub mod plans;
pub mod problem;
pub mod prune;
pub mod search;
pub mod start;
pub mod state;
mod util;

pub use chips::{ChipScenario, scenarios, terminal_value};
pub use model::{SolveError, TransferSet, solve_plan};
pub use params::OptimizerParams;
pub use plans::{GwPlan, Plan, PlanSet, optimize};
pub use problem::{HorizonGw, PlanInput, Player, XpFrame};
pub use search::Search;
