//! The bot's commands as plain functions (Python `fplopt.bot.service`; Phase 7 plan §4, §6):
//! each takes the sender's chat id and the command's arguments and returns the replies to
//! send, so the logic is tested without Telegram (`telegram` is the thin layer around it).
//!
//! Access is invite-only: `/start` from a new chat makes a `pending` user and tells the admin
//! (`TELEGRAM_ADMIN_CHAT_ID`, active from the start), who answers `/allow <chat_id>`.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::time::Instant;

use anyhow::{Context, Result};
use chrono::{DateTime, Duration, Utc};
use fplopt_optimize::entry::SquadSpec;
use log::warn;
use serde::Serialize;
use serde_json::Value;

use crate::chip_calls::{ChipCalls, V6_FROM_GW_INDEX, compute_chip_calls, v6_params};
use crate::context::{ContextStore, PlanContext, with_live_prices};
use crate::messages::{
    MAX_MESSAGE, UNCHANGED, chip_failure_section, chip_section, chips_text, plan_text,
    split_message, squad_text, update_text,
};
use crate::planner::{
    BotFpl, ChipOptions, UnknownTeam, UserPlan, chip_options, chip_params, decision_summary,
    entry_message, import_user_squad, is_not_found, live_params, plan_user, plans_json, rules_for,
};
use crate::store::{ChipCallRow, Record, User, UserStore, dumps, ordered_object};
use crate::summary::plan_change;
use crate::util::{casefold, escape as e, is_digits, py_repr_str, py_split, uk_time};

pub const HELP: &str = "<b>FPL optimizer</b>: transfer plans for your FPL team, from the model that passed its 2025/26 test. You make the final call.

/register &lt;team ID&gt; · your FPL team (the number in your team's URL)
/plan · plans for the next deadline
/chips · what the (untested) chip planner would do: information only
/transfer &lt;out&gt; -&gt; &lt;in&gt; · a transfer you already made for the next GW (FPL doesn't show it before the deadline); /transfer list, /transfer undo, /transfer clear
/setft &lt;n&gt; · correct the free transfers for the next GW (/setft clear to undo)
/alerts on|off · the message 24 hours before each deadline (and a re-run about 2 hours before, only if the plan changes)
/help · this list";
pub const ADMIN_HELP: &str = "Admin: /allow &lt;chat id&gt;, /block &lt;chat id&gt;, /users";
pub const PENDING: &str = "Waiting for access: the admin has been asked.";
pub const NOT_READY: &str = "The data for the next deadline isn't ready yet (it's rebuilt after each new snapshot). Try again in a few minutes.";
const REGISTER_FIRST: &str = "Register your team first: /register &lt;team ID&gt;";

/// FPL's live prices are fetched at most this often (one bootstrap for every plan in an alert
/// run, or a burst of commands).
pub fn price_ttl() -> Duration {
    Duration::minutes(5)
}

/// The commands and whether they take arguments (Python `app.COMMANDS`).
pub const COMMANDS: [(&str, bool); 11] = [
    ("start", false),
    ("help", false),
    ("register", true),
    ("plan", false),
    ("chips", false),
    ("setft", true),
    ("transfer", true),
    ("alerts", true),
    ("allow", true),
    ("block", true),
    ("users", false),
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reply {
    pub chat_id: i64,
    pub text: String,
}

fn reply(chat_id: i64, text: impl Into<String>) -> Vec<Reply> {
    vec![Reply { chat_id, text: text.into() }]
}

/// `Wissa -> Gonzalo` (also →, >, or a comma) or `Wissa Gonzalo` -> (out, in).
pub fn split_transfer(text: &str) -> Option<(String, String)> {
    for separator in ["->", "→", ">", ","] {
        if let Some((out, into)) = text.split_once(separator) {
            let (out, into) = (py_strip(out), py_strip(into));
            return (!out.is_empty() && !into.is_empty())
                .then(|| (out.to_string(), into.to_string()));
        }
    }
    match py_split(text).as_slice() {
        [out, into] => Some((out.to_string(), into.to_string())),
        _ => None,
    }
}

fn py_strip(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace() || ('\x1c'..='\x1f').contains(&c))
}

/// The element id of the season's player called `name` (web name, any case and accents as
/// typed; or an element id), or why not.
fn resolve(context: &PlanContext, name: &str) -> std::result::Result<i64, String> {
    let players = &context.players;
    if is_digits(name)
        && let Ok(id) = name.parse::<i64>()
        && players.iter().any(|p| p.element_id == id)
    {
        return Ok(id);
    }
    let wanted = casefold(name);
    let mut hits: Vec<_> = players.iter().filter(|p| casefold(&p.web_name) == wanted).collect();
    if hits.is_empty() {
        hits = players.iter().filter(|p| casefold(&p.web_name).contains(&wanted)).collect();
    }
    match hits.len() {
        1 => Ok(hits[0].element_id),
        0 => Err(format!("No player called {} this season.", py_repr_str(name))),
        _ => {
            let mut options: Vec<&str> = hits.iter().map(|p| p.web_name.as_str()).collect();
            options.sort();
            options.truncate(8);
            Err(format!(
                "{} matches several players ({}): be more specific.",
                py_repr_str(name),
                options.join(", ")
            ))
        }
    }
}

pub type Planner =
    Box<dyn Fn(&dyn BotFpl, i64, &PlanContext, Option<i64>, &[(i64, i64)]) -> Result<UserPlan>>;
pub type ChipPlanner =
    Box<dyn Fn(&dyn BotFpl, i64, &PlanContext, Option<i64>, &[(i64, i64)]) -> Result<ChipOptions>>;
pub type Clock = Box<dyn Fn() -> DateTime<Utc>>;
/// FPL's prices (element_id -> now_cost) and when they were fetched.
type LivePrices = (DateTime<Utc>, HashMap<i64, i64>);

/// The live planner: `plan_user` with `live_params`.
pub fn default_planner() -> Planner {
    let params = live_params();
    Box::new(move |fpl, entry, context, ft, made| plan_user(fpl, entry, context, ft, &params, made))
}

/// `/chips`'s planner: `chip_options` with `chip_params()` (read when the bot starts).
pub fn default_chip_planner() -> ChipPlanner {
    let params = chip_params();
    Box::new(move |fpl, entry, context, ft, made| {
        chip_options(fpl, entry, context, ft, &params, made)
    })
}

/// The alerts' V6 planner (chip policy 2's frozen V6, logged from GW20): `chip_options` with
/// `v6_params()` (read when the bot starts; if that fails, every call returns the error).
pub fn default_v6_planner() -> ChipPlanner {
    let params = v6_params().map_err(|err| format!("{err:#}"));
    Box::new(move |fpl, entry, context, ft, made| match &params {
        Ok(params) => chip_options(fpl, entry, context, ft, params, made),
        Err(err) => Err(anyhow::anyhow!("V6 settings unavailable: {err}")),
    })
}

/// One alert run's chip calls, P's scenarios as `chip_checks` stores them and V6's error
/// (`"v6: ..."`), or why P's calculation failed.
enum ChipStep {
    Calls { calls: Box<ChipCalls>, scenarios: String, v6_error: Option<String> },
    Failed(String),
}

/// A chip search skipped because the tick's chip time budget ran out.
pub const BUDGET_SKIP: &str = "skipped (tick time budget)";

/// `f()` with a panic turned into an error (a chip search must never take the alert down).
fn guarded<T>(f: impl FnOnce() -> Result<T>) -> std::result::Result<T, String> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(err)) => Err(format!("{err:#}")),
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown".to_string());
            Err(format!("panicked: {message}"))
        }
    }
}

impl ChipStep {
    fn calls(&self) -> Option<&ChipCalls> {
        match self {
            ChipStep::Calls { calls, .. } => Some(calls),
            ChipStep::Failed(_) => None,
        }
    }
}

#[derive(Serialize)]
struct SavedSquad<'a> {
    players: Vec<[i64; 2]>,
    bank: i64,
    free_transfers: i64,
    chips_used: Vec<(&'a str, i64)>,
    notes: &'a [String],
}

/// How a user's alert went (Python's `TickResult` lists).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AlertOutcome {
    /// Sent (the texts, each split to Telegram's limit).
    Sent,
    /// A re-run with the same decision: logged, not sent.
    Silent,
    /// The squad couldn't be imported; the user was told (or telling them failed).
    Told,
    /// Telegram refused the chat: alerts turned off.
    Undeliverable,
}

/// A message that wasn't sent (Python `TelegramSendError`): `permanent` when Telegram refused
/// the chat (403, 400), so retrying won't help. The text never holds the bot token.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SendError {
    pub message: String,
    pub permanent: bool,
}

impl std::fmt::Display for SendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for SendError {}

/// Command logic over the user store, the plan contexts and FPL's public API.
pub struct BotService {
    pub users: UserStore,
    pub contexts: ContextStore,
    pub fpl: Box<dyn BotFpl>,
    pub admin_chat_id: Option<i64>,
    pub now: Clock,
    pub planner: Planner,
    pub chip_planner: ChipPlanner,
    /// Alerts carry the chip section and log `chip_calls` (spec 2026-11-05-chip-alerts; on in
    /// `bot run` / `live tick`, off by default so `bot replay` matches Python's golden).
    pub chip_alerts: bool,
    /// V6's chip search for the `chip_calls` log (alert24 from GW20).
    pub v6_planner: ChipPlanner,
    /// No chip search starts after this (`live tick` sets it, so a slow search can't run the
    /// job past its timeout); None: no budget.
    pub chip_budget_until: Cell<Option<Instant>>,
    prices: RefCell<Option<LivePrices>>,
}

impl BotService {
    pub fn new(
        users: UserStore,
        contexts: ContextStore,
        fpl: Box<dyn BotFpl>,
        admin_chat_id: Option<i64>,
    ) -> BotService {
        BotService {
            users,
            contexts,
            fpl,
            admin_chat_id,
            now: Box::new(Utc::now),
            planner: default_planner(),
            chip_planner: default_chip_planner(),
            chip_alerts: false,
            v6_planner: default_v6_planner(),
            chip_budget_until: Cell::new(None),
            prices: RefCell::new(None),
        }
    }

    /// Run `command` (one of `COMMANDS`) for `chat_id`.
    pub fn run(&self, command: &str, chat_id: i64, args: &[String]) -> Result<Vec<Reply>> {
        match command {
            "start" => self.start(chat_id),
            "help" => self.help(chat_id),
            "register" => self.register(chat_id, args),
            "plan" => self.plan(chat_id),
            "chips" => self.chips(chat_id),
            "setft" => self.setft(chat_id, args),
            "transfer" => self.transfer(chat_id, args),
            "alerts" => self.alerts(chat_id, args),
            "allow" => self.set_status(chat_id, args, "allow", "active"),
            "block" => self.set_status(chat_id, args, "block", "blocked"),
            "users" => self.list_users(chat_id),
            other => anyhow::bail!("unknown command {other}"),
        }
    }

    // --- access -----------------------------------------------------------------------------

    pub fn is_admin(&self, chat_id: i64) -> bool {
        self.admin_chat_id == Some(chat_id)
    }

    /// The active user for `chat_id`, or the replies that say why not.
    fn active(&self, chat_id: i64) -> Result<std::result::Result<User, Vec<Reply>>> {
        let mut user = self.users.user(chat_id)?;
        if user.is_none() && self.is_admin(chat_id) {
            user = Some(self.users.add_user(chat_id, "active")?);
        }
        Ok(match user {
            None => Err(reply(chat_id, "Send /start first.")),
            Some(u) if u.status == "pending" => Err(reply(chat_id, PENDING)),
            Some(u) if u.status == "blocked" => Err(Vec::new()),
            Some(u) => Ok(u),
        })
    }

    pub fn start(&self, chat_id: i64) -> Result<Vec<Reply>> {
        let existing = self.users.user(chat_id)?;
        if self.is_admin(chat_id) {
            if existing.as_ref().is_none_or(|u| u.status != "active") {
                self.users.add_user(chat_id, "active")?;
                self.users.set_status(chat_id, "active")?;
            }
            return Ok(reply(chat_id, format!("{HELP}\n\n{ADMIN_HELP}")));
        }
        let Some(existing) = existing else {
            self.users.add_user(chat_id, "pending")?;
            let mut replies = reply(chat_id, PENDING);
            if let Some(admin) = self.admin_chat_id {
                replies.push(Reply {
                    chat_id: admin,
                    text: format!("New user asks for access: chat {chat_id}. /allow {chat_id}"),
                });
            }
            return Ok(replies);
        };
        Ok(match existing.status.as_str() {
            "pending" => reply(chat_id, PENDING),
            "blocked" => Vec::new(),
            _ => reply(chat_id, HELP),
        })
    }

    pub fn help(&self, chat_id: i64) -> Result<Vec<Reply>> {
        let extra =
            if self.is_admin(chat_id) { format!("\n\n{ADMIN_HELP}") } else { String::new() };
        Ok(reply(chat_id, format!("{HELP}{extra}")))
    }

    /// The chat id argument of /allow and /block (group chats' ids are negative); an error
    /// where Python's `int()` raises.
    fn admin_target(args: &[String]) -> Result<Option<i64>> {
        if args.len() != 1 || !is_digits(args[0].trim_start_matches('-')) {
            return Ok(None);
        }
        Ok(Some(
            args[0]
                .parse::<i64>()
                .with_context(|| format!("invalid literal for int(): {:?}", args[0]))?,
        ))
    }

    fn set_status(
        &self,
        chat_id: i64,
        args: &[String],
        command: &str,
        status: &str,
    ) -> Result<Vec<Reply>> {
        if !self.is_admin(chat_id) {
            return self.unknown(chat_id);
        }
        let Some(target) = Self::admin_target(args)? else {
            return Ok(reply(chat_id, format!("Usage: /{command} &lt;chat id&gt;")));
        };
        if self.is_admin(target) && status == "blocked" {
            return Ok(reply(chat_id, "That's you."));
        }
        if self.users.user(target)?.is_none() {
            self.users.add_user(target, status)?;
        }
        self.users.set_status(target, status)?;
        let mut replies = reply(chat_id, format!("Chat {target}: {status}."));
        if status == "active" {
            replies.push(Reply { chat_id: target, text: format!("You have access.\n\n{HELP}") });
        }
        Ok(replies)
    }

    pub fn list_users(&self, chat_id: i64) -> Result<Vec<Reply>> {
        if !self.is_admin(chat_id) {
            return self.unknown(chat_id);
        }
        let rows: Vec<String> = self
            .users
            .users(None)?
            .iter()
            .map(|u| {
                let team = match u.entry_id {
                    Some(id) if id != 0 => id.to_string(),
                    _ => "-".to_string(),
                };
                format!(
                    "chat {}: {}, team {team}, alerts {}",
                    u.chat_id,
                    u.status,
                    if u.alerts { "on" } else { "off" }
                )
            })
            .collect();
        Ok(reply(chat_id, if rows.is_empty() { "No users.".to_string() } else { rows.join("\n") }))
    }

    pub fn unknown(&self, chat_id: i64) -> Result<Vec<Reply>> {
        Ok(match self.active(chat_id)? {
            Err(replies) => replies,
            Ok(_) => reply(chat_id, "Unknown command. /help lists them."),
        })
    }

    // --- the user's team --------------------------------------------------------------------

    /// The newest context, if it is for a deadline still ahead, with FPL's live prices.
    fn context(&self) -> Result<Option<PlanContext>> {
        let Some(context) = self.contexts.latest()? else { return Ok(None) };
        if context.deadline <= (self.now)() {
            return Ok(None);
        }
        Ok(Some(self.priced(&context)))
    }

    /// `context` with FPL's current prices (`with_live_prices`), fetched at most every
    /// `price_ttl`. If FPL can't be reached the context's own prices are kept (logged; the
    /// message then says prices are from the data's snapshot).
    pub fn priced(&self, context: &PlanContext) -> PlanContext {
        if context.prices_at.is_some() {
            return context.clone();
        }
        let now = (self.now)();
        let stale = self.prices.borrow().as_ref().is_none_or(|(at, _)| now - *at > price_ttl());
        if stale {
            match self.live_prices() {
                Ok(prices) => *self.prices.borrow_mut() = Some((now, prices)),
                Err(err) => {
                    warn!("live prices unavailable ({err}): using the context's");
                    return context.clone();
                }
            }
        }
        let cached = self.prices.borrow();
        let (fetched_at, prices) = cached.as_ref().expect("prices just fetched");
        with_live_prices(context, prices, *fetched_at)
    }

    fn live_prices(&self) -> Result<HashMap<i64, i64>> {
        let bootstrap: Value = serde_json::from_slice(&self.fpl.bootstrap_static()?)?;
        let elements = bootstrap["elements"].as_array().context("bootstrap: no elements")?;
        elements
            .iter()
            .map(|el| {
                let id = py_int(&el["id"]).context("bootstrap: element id")?;
                let cost = py_int(&el["now_cost"]).context("bootstrap: now_cost")?;
                Ok((id, cost))
            })
            .collect()
    }

    pub fn register(&self, chat_id: i64, args: &[String]) -> Result<Vec<Reply>> {
        let user = match self.active(chat_id)? {
            Ok(u) => u,
            Err(replies) => return Ok(replies),
        };
        let entry_id = match args {
            [arg] if is_digits(arg) => arg.parse::<i64>().ok().filter(|id| *id >= 1),
            _ => None,
        };
        let Some(entry_id) = entry_id else {
            return Ok(reply(
                chat_id,
                "Usage: /register &lt;team ID&gt;, the number in your FPL team's URL \
                 (fantasy.premierleague.com/entry/<b>1234567</b>/event/…).",
            ));
        };
        let Some(context) = self.context()? else {
            if let Err(err) = self.fpl.entry_history(entry_id) {
                if is_not_found(&err) {
                    return Ok(reply(chat_id, format!("FPL has no team with ID {entry_id}.")));
                }
                return Err(err);
            }
            self.users.set_entry(user.user_id, entry_id)?;
            return Ok(reply(chat_id, format!("Registered team {entry_id}.\n{NOT_READY}")));
        };
        let ft = self.users.free_transfers(user.user_id, context.season, context.gw)?;
        let spec = match import_user_squad(self.fpl.as_ref(), entry_id, &context, ft, &[]) {
            Ok((spec, _, _)) => spec,
            Err(err) => {
                if let Some(unknown) = err.downcast_ref::<UnknownTeam>() {
                    return Ok(reply(chat_id, format!("{}.", e(&unknown.0))));
                }
                let Some(message) = entry_message(&err) else { return Err(err) };
                self.users.set_entry(user.user_id, entry_id)?;
                return Ok(reply(
                    chat_id,
                    format!(
                        "Registered team {entry_id}, but its squad can't be planned yet: {}",
                        e(&message)
                    ),
                ));
            }
        };
        self.users.set_entry(user.user_id, entry_id)?;
        self.save_squad(&user, entry_id, &spec, &context)?;
        let text = squad_text(&context, &spec, entry_id);
        Ok(reply(chat_id, format!("Registered.\n\n{text}\n\n/plan for the GW{} plan.", context.gw)))
    }

    fn save_squad(
        &self,
        user: &User,
        entry_id: i64,
        spec: &SquadSpec,
        context: &PlanContext,
    ) -> Result<()> {
        let squad = dumps(&SavedSquad {
            players: spec.players.iter().map(|p| [p.element_id, p.purchase_price]).collect(),
            bank: spec.bank,
            free_transfers: spec.free_transfers,
            chips_used: spec.chips_used.iter().map(|(n, g)| (n.as_str(), *g)).collect(),
            notes: &spec.notes,
        })?;
        self.users.save_squad(
            user.user_id,
            context.season,
            context.gw,
            entry_id,
            &squad,
            Some(&context.context_id),
        )
    }

    pub fn plan(&self, chat_id: i64) -> Result<Vec<Reply>> {
        let user = match self.active(chat_id)? {
            Ok(u) => u,
            Err(replies) => return Ok(replies),
        };
        let Some(entry_id) = user.entry_id else { return Ok(reply(chat_id, REGISTER_FIRST)) };
        let Some(context) = self.context()? else { return Ok(reply(chat_id, NOT_READY)) };
        match self.make_plan(&user, &context, "plan") {
            Ok(result) => Ok(reply(chat_id, plan_text(&result))),
            Err(err) => match entry_message(&err) {
                Some(message) => {
                    Ok(reply(chat_id, format!("Can't plan team {entry_id}: {}", e(&message))))
                }
                None => Err(err),
            },
        }
    }

    /// `/chips`: the chip scenario search on the user's squad, as information (the bot's plans
    /// don't play chips until a chip policy passes its own test).
    pub fn chips(&self, chat_id: i64) -> Result<Vec<Reply>> {
        let user = match self.active(chat_id)? {
            Ok(u) => u,
            Err(replies) => return Ok(replies),
        };
        let Some(entry_id) = user.entry_id else { return Ok(reply(chat_id, REGISTER_FIRST)) };
        let Some(context) = self.context()? else { return Ok(reply(chat_id, NOT_READY)) };
        let ft = self.users.free_transfers(user.user_id, context.season, context.gw)?;
        let made = self.users.transfers(user.user_id, context.season, context.gw)?;
        let options = match (self.chip_planner)(self.fpl.as_ref(), entry_id, &context, ft, &made) {
            Ok(options) => options,
            Err(err) => match entry_message(&err) {
                Some(message) => {
                    return Ok(reply(
                        chat_id,
                        format!("Can't plan team {entry_id}: {}", e(&message)),
                    ));
                }
                None => return Err(err),
            },
        };
        let mut scenarios = options.plans.scenario_objectives.clone();
        scenarios.extend(options.singles.iter().map(|(_, (label, value))| (label.clone(), *value)));
        self.users.record_chip_check(
            user.user_id,
            context.season,
            context.gw,
            Some(&context.context_id),
            &options.plans.best().scenario,
            &scenarios,
        )?;
        Ok(reply(chat_id, chips_text(&options, 6)))
    }

    /// Plan the user's team on `context` (with FPL's live prices) and log it (`kind`; an entry
    /// error is logged as the recommendation's error and returned).
    pub fn make_plan(&self, user: &User, context: &PlanContext, kind: &str) -> Result<UserPlan> {
        let entry_id = user.entry_id.context("make_plan: the user has no team")?;
        let context = self.priced(context);
        let ft = self.users.free_transfers(user.user_id, context.season, context.gw)?;
        let made = self.users.transfers(user.user_id, context.season, context.gw)?;
        let result = match (self.planner)(self.fpl.as_ref(), entry_id, &context, ft, &made) {
            Ok(result) => result,
            Err(err) => {
                if let Some(message) = entry_message(&err) {
                    self.users.record(
                        user.user_id,
                        context.season,
                        context.gw,
                        kind,
                        &Record {
                            context_id: Some(context.context_id.clone()),
                            error: Some(format!("EntryError: {message}")),
                            ..Record::default()
                        },
                    )?;
                }
                return Err(err);
            }
        };
        self.save_squad(user, entry_id, &result.spec, &context)?;
        if kind == "plan" {
            self.users.record(
                user.user_id,
                context.season,
                context.gw,
                kind,
                &Record {
                    context_id: Some(context.context_id.clone()),
                    plans: Some(plans_json(&result)?),
                    decision: Some(dumps(&decision_summary(&result))?),
                    message: Some(plan_text(&result)),
                    sent: true,
                    error: None,
                },
            )?;
        }
        Ok(result)
    }

    /// `/transfer <out> -> <in>` (or `/transfer <out> <in>` for one-word names): record a
    /// transfer already made for the next GW; `list`, `undo`, `clear`. Checked against the
    /// squad (position, club limit, money) before it is stored.
    pub fn transfer(&self, chat_id: i64, args: &[String]) -> Result<Vec<Reply>> {
        let user = match self.active(chat_id)? {
            Ok(u) => u,
            Err(replies) => return Ok(replies),
        };
        let Some(entry_id) = user.entry_id else { return Ok(reply(chat_id, REGISTER_FIRST)) };
        let Some(context) = self.context()? else { return Ok(reply(chat_id, NOT_READY)) };
        let (season, gw) = (context.season, context.gw);
        let made = self.users.transfers(user.user_id, season, gw)?;
        let names = context.names();
        let keys = context.element_keys();
        let listed = |moves: &[(i64, i64)]| {
            let label = |el: i64| {
                keys.get(&el).and_then(|k| names.get(k)).cloned().unwrap_or_else(|| el.to_string())
            };
            let text: Vec<String> =
                moves.iter().map(|&(o, i)| format!("{} → {}", label(o), label(i))).collect();
            let text = text.join(", ");
            e(if text.is_empty() { "none" } else { &text })
        };
        let command = args.join(" ");
        let command = py_strip(&command);
        let lower = command.to_lowercase();
        if lower.is_empty() || lower == "list" {
            return Ok(reply(
                chat_id,
                format!("GW{gw} transfers you've told me about: {}", listed(&made)),
            ));
        }
        if lower == "undo" || lower == "clear" {
            let n = self.users.drop_transfers(user.user_id, season, gw, lower == "undo")?;
            let left = self.users.transfers(user.user_id, season, gw)?;
            return Ok(reply(
                chat_id,
                format!("Removed {n}. GW{gw} transfers now: {}", listed(&left)),
            ));
        }
        let Some((out, into)) = split_transfer(command) else {
            return Ok(reply(
                chat_id,
                "Usage: /transfer Wissa -&gt; Gonzalo (or /transfer Wissa Gonzalo); /transfer list, \
                 /transfer undo, /transfer clear",
            ));
        };
        let mut resolved = Vec::new();
        for name in [&out, &into] {
            match resolve(&context, name) {
                Ok(element) => resolved.push(element),
                Err(problem) => return Ok(reply(chat_id, e(&problem))),
            }
        }
        let new_move = (resolved[0], resolved[1]);
        let ft = self.users.free_transfers(user.user_id, season, gw)?;
        let mut all = made.clone();
        all.push(new_move);
        let spec = match import_user_squad(self.fpl.as_ref(), entry_id, &context, ft, &all) {
            Ok((spec, _, _)) => spec,
            Err(err) => match entry_message(&err) {
                Some(message) => {
                    return Ok(reply(
                        chat_id,
                        format!("Can't record that transfer: {}", e(&message)),
                    ));
                }
                None => return Err(err),
            },
        };
        self.users.add_transfer(user.user_id, season, gw, new_move.0, new_move.1)?;
        Ok(reply(
            chat_id,
            format!(
                "Recorded for GW{gw}: {}. All GW{gw} transfers: {}. Bank £{:.1}m, {} FT left. /plan \
                 and the alerts now plan from this squad.",
                listed(&[new_move]),
                listed(&all),
                spec.bank as f64 / 10.0,
                spec.free_transfers
            ),
        ))
    }

    pub fn setft(&self, chat_id: i64, args: &[String]) -> Result<Vec<Reply>> {
        let user = match self.active(chat_id)? {
            Ok(u) => u,
            Err(replies) => return Ok(replies),
        };
        let Some(context) = self.context()? else { return Ok(reply(chat_id, NOT_READY)) };
        let cap = rules_for(context.season)?.max_free_transfers;
        if args.len() == 1 && args[0].to_lowercase() == "clear" {
            self.users.set_free_transfers(user.user_id, context.season, context.gw, None)?;
            return Ok(reply(
                chat_id,
                format!("GW{}: free transfers from the import again.", context.gw),
            ));
        }
        let n = match args {
            [arg] if is_digits(arg) => arg.parse::<i64>().ok().filter(|n| (0..=cap).contains(n)),
            _ => None,
        };
        let Some(n) = n else {
            return Ok(reply(chat_id, format!("Usage: /setft &lt;0-{cap}&gt; or /setft clear")));
        };
        self.users.set_free_transfers(user.user_id, context.season, context.gw, Some(n))?;
        Ok(reply(
            chat_id,
            format!(
                "GW{} (deadline {}): planning with {n} free transfers.",
                context.gw,
                uk_time(context.deadline)
            ),
        ))
    }

    pub fn alerts(&self, chat_id: i64, args: &[String]) -> Result<Vec<Reply>> {
        let user = match self.active(chat_id)? {
            Ok(u) => u,
            Err(replies) => return Ok(replies),
        };
        let choice = match args {
            [arg] => Some(arg.to_lowercase()).filter(|a| a == "on" || a == "off"),
            _ => None,
        };
        let Some(choice) = choice else {
            let state = if user.alerts { "on" } else { "off" };
            return Ok(reply(chat_id, format!("Alerts are {state}. Usage: /alerts on|off")));
        };
        self.users.set_alerts(user.user_id, choice == "on")?;
        Ok(reply(chat_id, format!("Alerts {choice}.")))
    }

    // --- the deadline alerts (Python `live._alert_user`) --------------------------------------

    /// Plan, log and send one user's alert of `kind` (`alert24`, or the `alert2` re-run that
    /// is sent only if plan 1's decision changed since the last alert). `send(chat_id, text)`
    /// sends one piece.
    pub fn alert_user(
        &self,
        kind: &str,
        context: &PlanContext,
        user: &User,
        send: &mut dyn FnMut(i64, &str) -> std::result::Result<(), SendError>,
    ) -> Result<AlertOutcome> {
        let plan = match self.make_plan(user, context, kind) {
            Ok(plan) => plan,
            Err(err) => {
                let Some(message) = entry_message(&err) else { return Err(err) };
                let entry = user.entry_id.map_or("None".to_string(), |id| id.to_string());
                let text =
                    format!("Couldn't plan GW{} for team {entry}: {}", context.gw, e(&message));
                if let Err(err) = send_pieces(send, user.chat_id, &text) {
                    warn!("telling user {} about their squad failed: {err}", user.user_id);
                }
                return Ok(AlertOutcome::Told);
            }
        };
        let decision = decision_summary(&plan);
        let last = self.users.last_sent_alert(user.user_id, context.season, context.gw)?;
        let base = Record {
            context_id: Some(context.context_id.clone()),
            plans: Some(plans_json(&plan)?),
            decision: Some(dumps(&decision)?),
            ..Record::default()
        };
        let chips = (self.chip_alerts && (kind == "alert24" || kind == "alert2"))
            .then(|| self.chip_step(user, &plan, kind));
        let section =
            chips.as_ref().and_then(|step| self.chip_text(kind, last.is_none(), user, &plan, step));
        // Log the run's chip calls after its recommendation row; best effort, never failing
        // the alert.
        let record_chips = || {
            if let Some(step) = &chips
                && let Err(err) = self.record_chip_step(user, &plan, kind, step)
            {
                warn!(
                    "GW{} {kind} chip_calls row for user {} not written: {err:#}",
                    context.gw, user.user_id
                );
            }
        };
        let finish = |record: Record, outcome: AlertOutcome| -> Result<AlertOutcome> {
            self.users.record(user.user_id, context.season, context.gw, kind, &record)?;
            record_chips();
            Ok(outcome)
        };
        let text = match (kind, &last) {
            ("alert2", Some(last)) => match plan_change(last.decision.as_ref(), &plan) {
                None if section.is_none() => return finish(base, AlertOutcome::Silent),
                None => UNCHANGED.to_string(),
                Some(change) => update_text(&change),
            },
            _ => plan_text(&plan),
        };
        let text = match &section {
            Some(section) => format!("{section}\n\n{text}"),
            None => text,
        };
        if let Err(err) = send_pieces(send, user.chat_id, &text) {
            if !err.permanent {
                return Err(err.into());
            }
            // The user blocked the bot or the chat is gone: stop alerting them (/alerts on
            // turns it back on) and don't retry.
            warn!("GW{} {kind} undeliverable to user {}: {err}", context.gw, user.user_id);
            self.users.set_alerts(user.user_id, false)?;
            let record = Record {
                message: Some(text),
                error: Some(format!("undeliverable: {err}")),
                ..base
            };
            return finish(record, AlertOutcome::Undeliverable);
        }
        finish(Record { message: Some(text), sent: true, ..base }, AlertOutcome::Sent)
    }

    /// The chip section for an alert run (spec §5): on `alert24`, and on an `alert2` that is
    /// the user's first alert this GW (`first`), the calls or the failure line. On any other
    /// `alert2`, the calls only if this run's succeeded and the user hasn't seen this call:
    /// no logged `alert24` call (or it can't be read), the `alert24` calculation failed, or
    /// the call shown (planner, rules, forced) changed since.
    fn chip_text(
        &self,
        kind: &str,
        first: bool,
        user: &User,
        plan: &UserPlan,
        step: &ChipStep,
    ) -> Option<String> {
        if kind == "alert24" || first {
            return Some(step.calls().map_or_else(chip_failure_section, chip_section));
        }
        let calls = step.calls()?;
        let context = &plan.context;
        let logged = match self.users.chip_call(user.user_id, context.season, context.gw, "alert24")
        {
            Ok(logged) => logged,
            Err(err) => {
                warn!(
                    "GW{} alert24 chip call of user {} unreadable: {err:#}",
                    context.gw, user.user_id
                );
                None
            }
        };
        let seen = logged.is_some_and(|row| {
            !planner_failed(&row) && shown_call(&row) == shown_call(&chip_call_row(calls))
        });
        (!seen).then(|| chip_section(calls))
    }

    /// The chip calls for `plan`'s alert (spec 2026-11-05-chip-alerts §1-3): P's search on the
    /// same priced context, free transfers and made transfers as the plan, and on `alert24`
    /// from GW20 V6's (logged only). A failure or panic of P gives no calls (the failure
    /// line); V6's only sets the row's error. Both are skipped once the tick's chip time budget
    /// (`chip_budget_until`) has run out. Never an error.
    fn chip_step(&self, user: &User, plan: &UserPlan, kind: &str) -> ChipStep {
        let started = Instant::now();
        let step = self.chip_step_inner(user, plan, kind);
        log::info!(
            "GW{} {kind} chip calls for user {}: {:.1} s",
            plan.context.gw,
            user.user_id,
            started.elapsed().as_secs_f64()
        );
        step
    }

    fn chip_step_inner(&self, user: &User, plan: &UserPlan, kind: &str) -> ChipStep {
        let context = &plan.context;
        let failed = |err: String| {
            warn!("GW{} {kind} chip calls for user {} failed: {err}", context.gw, user.user_id);
            ChipStep::Failed(err)
        };
        let over_budget = || self.chip_budget_until.get().is_some_and(|t| Instant::now() >= t);
        if over_budget() {
            return failed(BUDGET_SKIP.to_string());
        }
        let inputs = (|| -> Result<_> {
            let entry_id = user.entry_id.context("chip calls: the user has no team")?;
            let ft = self.users.free_transfers(user.user_id, context.season, context.gw)?;
            let made = self.users.transfers(user.user_id, context.season, context.gw)?;
            Ok((entry_id, ft, made))
        })();
        let (entry_id, ft, made) = match inputs {
            Ok(inputs) => inputs,
            Err(err) => return failed(format!("{err:#}")),
        };
        let options = match guarded(|| {
            (self.chip_planner)(self.fpl.as_ref(), entry_id, context, ft, &made)
        }) {
            Ok(options) => options,
            Err(err) => return failed(err),
        };
        // H's numbers solve more MILPs (`chip_gains`): budgeted and panic-isolated too. P and
        // H come before V6, which is only logged.
        if over_budget() {
            return failed(BUDGET_SKIP.to_string());
        }
        let mut calls = match guarded(|| compute_chip_calls(plan, &options, None)) {
            Ok(calls) => calls,
            Err(err) => return failed(err),
        };
        let mut v6_error = None;
        if kind == "alert24" && context.gw_index >= V6_FROM_GW_INDEX {
            let result = if over_budget() {
                Err(BUDGET_SKIP.to_string())
            } else {
                guarded(|| (self.v6_planner)(self.fpl.as_ref(), entry_id, context, ft, &made))
            };
            match result {
                // as `compute_chip_calls` reads V6's search
                Ok(v6) => calls.v6 = Some(v6.plans.best().first().chip.clone()),
                Err(err) => {
                    warn!("GW{} V6 chip call for user {} failed: {err}", context.gw, user.user_id);
                    v6_error = Some(format!("v6: {err}"));
                }
            }
        }
        ChipStep::Calls {
            calls: Box::new(calls),
            scenarios: chip_scenarios_json(&options),
            v6_error,
        }
    }

    /// Log one alert run's chip calls (or their failure) in `chip_calls`.
    fn record_chip_step(
        &self,
        user: &User,
        plan: &UserPlan,
        kind: &str,
        step: &ChipStep,
    ) -> Result<()> {
        let context = &plan.context;
        let row = match step {
            ChipStep::Calls { calls, scenarios, v6_error } => ChipCallRow {
                gains: dumps(&calls.gains)?,
                scenarios: scenarios.clone(),
                error: v6_error.clone(),
                ..chip_call_row(calls)
            },
            ChipStep::Failed(err) => {
                ChipCallRow { error: Some(err.clone()), ..chip_call_row_base() }
            }
        };
        self.users.record_chip_call(&ChipCallRow {
            user_id: user.user_id,
            season: context.season,
            gw: context.gw,
            kind: kind.to_string(),
            context_id: Some(context.context_id.clone()),
            ..row
        })
    }
}

fn send_pieces(
    send: &mut dyn FnMut(i64, &str) -> std::result::Result<(), SendError>,
    chat_id: i64,
    text: &str,
) -> std::result::Result<(), SendError> {
    for piece in split_message(text, MAX_MESSAGE) {
        send(chat_id, &piece)?;
    }
    Ok(())
}

/// Python's `int(x)` of a JSON value (an integer, a whole float or a numeric string).
fn py_int(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f.trunc() as i64)),
        Value::String(s) => py_strip(s).parse().ok(),
        Value::Bool(b) => Some(i64::from(*b)),
        _ => None,
    }
}

/// An empty `chip_calls` row: every call `"none"`, no gains or scenarios (the caller fills in
/// the user, GW and calls).
fn chip_call_row_base() -> ChipCallRow {
    ChipCallRow {
        user_id: 0,
        season: 0,
        gw: 0,
        kind: String::new(),
        context_id: None,
        planner_call: "none".to_string(),
        planner_gain: None,
        rules_call: "none".to_string(),
        forced: None,
        v6_call: None,
        gains: "{}".to_string(),
        scenarios: "{}".to_string(),
        error: None,
        created_at: String::new(),
    }
}

/// `calls` as a `chip_calls` row's call columns (`"none"` for hold; `forced` null when no
/// chip is forced; `v6_call` null when V6 wasn't computed).
fn chip_call_row(calls: &ChipCalls) -> ChipCallRow {
    let name = |c: &Option<String>| c.clone().unwrap_or_else(|| "none".to_string());
    ChipCallRow {
        planner_call: name(&calls.planner),
        planner_gain: Some(calls.planner_gain),
        rules_call: name(&calls.rules),
        forced: calls.forced.clone(),
        v6_call: calls.v6.as_ref().map(name),
        ..chip_call_row_base()
    }
}

/// Whether a logged row's planner calculation failed (an error not from V6 alone): its calls
/// are placeholders the user never saw.
fn planner_failed(row: &ChipCallRow) -> bool {
    row.error.as_deref().is_some_and(|e| !e.starts_with("v6: "))
}

/// The call the user sees (planner, rules, forced; `"none"` for None): an `alert2` shows the
/// chip section only if it differs from this GW's `alert24` call.
fn shown_call(row: &ChipCallRow) -> (&str, &str, &str) {
    (&row.planner_call, &row.rules_call, row.forced.as_deref().unwrap_or("none"))
}

/// `{**objectives, **singles}` as stored (re-exported for tests).
pub fn chip_scenarios_json(options: &ChipOptions) -> String {
    let mut scenarios = options.plans.scenario_objectives.clone();
    scenarios.extend(options.singles.iter().map(|(_, (label, value))| (label.clone(), *value)));
    ordered_object(&scenarios)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Python test_split_transfer.
    #[test]
    fn transfers_are_split() {
        let pair = |a: &str, b: &str| Some((a.to_string(), b.to_string()));
        assert_eq!(split_transfer("Wissa -> Gonzalo"), pair("Wissa", "Gonzalo"));
        assert_eq!(
            split_transfer("João Pedro → Calvert-Lewin"),
            pair("João Pedro", "Calvert-Lewin")
        );
        assert_eq!(split_transfer("Wissa Gonzalo"), pair("Wissa", "Gonzalo"));
        assert_eq!(split_transfer("Joao Pedro Gonzalo"), None); // ambiguous without a separator
        assert_eq!(split_transfer("Wissa ->"), None);
    }
}
