//! The Telegram bot (Python `fplopt.bot`, migration batches 5 and 6): the user store
//! (`bot.sqlite`, same schema), the plan contexts (`builder` writes them from the v1 model,
//! `context` reads them), the planner (`live_params`, FPL's live prices), the message texts,
//! the command logic, the long-polling Telegram layer and the deadline alert job (`live`).
//! `replay` runs scripted sessions for the parity gate.

pub mod builder;
pub mod context;
pub mod live;
pub mod messages;
pub mod planner;
pub mod replay;
pub mod service;
pub mod store;
pub mod telegram;
pub mod util;
