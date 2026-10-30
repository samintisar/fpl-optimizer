//! The Telegram bot (Python `fplopt.bot`, migration batch 5): the user store (`bot.sqlite`,
//! same schema), the plan context reader (contexts written by Python's `live prepare` until
//! batch 6), the planner (`live_params`, FPL's live prices), the message texts, the command
//! logic and the long-polling Telegram layer. `replay` runs scripted sessions for the parity
//! gate.

pub mod context;
pub mod messages;
pub mod planner;
pub mod replay;
pub mod service;
pub mod store;
pub mod telegram;
pub mod util;
