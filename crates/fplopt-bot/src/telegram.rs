//! The Telegram bot (`fplopt-rs bot run`, long polling; Python `fplopt.bot.app` over
//! python-telegram-bot): a thin layer over `service::BotService`, on the Bot API over plain
//! HTTPS.
//!
//! Updates are handled one at a time, in order (python-telegram-bot's default), so a `/plan`
//! waits for the one before it. Private chats only; messages without text and group
//! messages are ignored. Routing is python-telegram-bot's: a command entity at the start
//! (`/plan`, `/plan@thisbot`) runs its command; any other text goes to the text handler, which
//! still runs a command pasted with formatting (`parse_command`) and answers anything else
//! with "Unknown command". An unexpected error (or panic) is logged, alerted to the admin and
//! answered with a short apology.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::thread::sleep;
use std::time::Duration;

use anyhow::{Result, bail};
use fplopt_core::alerts::send_admin_alert;
use fplopt_core::http::failure_detail;
use fplopt_core::redact::{redact, register_secret};
use log::{error, warn};
use reqwest::blocking::Client;
use serde_json::{Value, json};

use crate::messages::{MAX_MESSAGE, split_message};
use crate::service::{BotService, COMMANDS, Reply, SendError};
use crate::util::py_split;

pub const API: &str = "https://api.telegram.org";
/// getUpdates' long-polling timeout (seconds; python-telegram-bot's default).
const POLL_TIMEOUT: u64 = 10;
pub const APOLOGY: &str = "Something went wrong. The admin has been told; try again later.";

/// A failed Bot API call, named as python-telegram-bot names it. The text never holds the
/// request URL (it contains the bot token).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TgError {
    BadRequest(String),
    Forbidden(String),
    InvalidToken,
    Conflict(String),
    RetryAfter(u64),
    /// Another HTTP status (python-telegram-bot raises NetworkError for these).
    Status(u16, String),
    /// No response (the transport's error kind).
    Network(String),
}

impl TgError {
    pub fn kind(&self) -> &'static str {
        match self {
            TgError::BadRequest(_) => "BadRequest",
            TgError::Forbidden(_) => "Forbidden",
            TgError::InvalidToken => "InvalidToken",
            TgError::Conflict(_) => "Conflict",
            TgError::RetryAfter(_) => "RetryAfter",
            TgError::Status(..) | TgError::Network(_) => "NetworkError",
        }
    }
}

impl std::fmt::Display for TgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TgError::BadRequest(m)
            | TgError::Forbidden(m)
            | TgError::Conflict(m)
            | TgError::Network(m) => f.write_str(m),
            TgError::InvalidToken => f.write_str("Invalid token"),
            TgError::RetryAfter(s) => write!(f, "Flood control exceeded. Retry in {s} seconds"),
            TgError::Status(code, m) => write!(f, "{m} ({code})"),
        }
    }
}

impl std::error::Error for TgError {}

/// The Bot API calls the bot makes (a trait, so the update handling is tested offline).
pub trait TelegramApi {
    fn send_message(&self, chat_id: i64, text: &str) -> Result<(), TgError>;
    fn send_chat_action(&self, chat_id: i64, action: &str) -> Result<(), TgError>;
}

/// The Bot API over HTTPS.
pub struct Telegram {
    http: Client,
    base: String,
}

impl Telegram {
    pub fn new(token: &str) -> Result<Telegram> {
        Telegram::with_api(API, token)
    }

    pub fn with_api(api: &str, token: &str) -> Result<Telegram> {
        let token = token.trim();
        register_secret(token);
        let http = Client::builder()
            .timeout(Duration::from_secs(POLL_TIMEOUT + 20))
            .connect_timeout(Duration::from_secs(10))
            .build()?;
        Ok(Telegram { http, base: format!("{}/bot{token}", api.trim_end_matches('/')) })
    }

    /// One Bot API call; its `result`.
    pub fn call(&self, method: &str, body: &Value) -> Result<Value, TgError> {
        let response = self
            .http
            .post(format!("{}/{method}", self.base))
            .json(body)
            .send()
            .map_err(|e| TgError::Network(failure_detail(&e)))?;
        let status = response.status().as_u16();
        let payload: Value = response.json().unwrap_or(Value::Null);
        if (200..300).contains(&status) && payload["ok"] == json!(true) {
            return Ok(payload["result"].clone());
        }
        let description = payload["description"].as_str().unwrap_or("").to_string();
        Err(match status {
            400 => TgError::BadRequest(description),
            403 => TgError::Forbidden(description),
            401 | 404 => TgError::InvalidToken,
            409 => TgError::Conflict(description),
            429 => TgError::RetryAfter(payload["parameters"]["retry_after"].as_u64().unwrap_or(1)),
            _ => TgError::Status(status, description),
        })
    }

    pub fn username(&self) -> Result<String, TgError> {
        Ok(self.call("getMe", &json!({}))?["username"].as_str().unwrap_or("").to_string())
    }

    pub fn get_updates(&self, offset: Option<i64>) -> Result<Vec<Value>, TgError> {
        let mut body = json!({"timeout": POLL_TIMEOUT});
        if let Some(offset) = offset {
            body["offset"] = json!(offset);
        }
        Ok(self.call("getUpdates", &body)?.as_array().cloned().unwrap_or_default())
    }

    /// `send(chat_id, html)` for the alert job (Python `sender.telegram_sender`): one
    /// sendMessage; SendError (permanent for 400 and 403) on any failure.
    pub fn send_html(&self, chat_id: i64, text: &str) -> Result<(), SendError> {
        self.send_message(chat_id, text).map_err(|e| {
            let status = match &e {
                TgError::BadRequest(_) => 400,
                TgError::Forbidden(_) => 403,
                TgError::InvalidToken => 401,
                TgError::Conflict(_) => 409,
                TgError::RetryAfter(_) => 429,
                TgError::Status(code, _) => *code,
                TgError::Network(kind) => {
                    return SendError {
                        message: format!("sendMessage to chat {chat_id} failed: {kind}"),
                        permanent: false,
                    };
                }
            };
            SendError {
                message: format!(
                    "sendMessage to chat {chat_id} failed: HTTPStatusError (HTTP {status})"
                ),
                permanent: status == 400 || status == 403,
            }
        })
    }
}

impl TelegramApi for Telegram {
    fn send_message(&self, chat_id: i64, text: &str) -> Result<(), TgError> {
        self.call(
            "sendMessage",
            &json!({
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",
                "link_preview_options": {"is_disabled": true},
            }),
        )
        .map(|_| ())
    }

    fn send_chat_action(&self, chat_id: i64, action: &str) -> Result<(), TgError> {
        self.call("sendChatAction", &json!({"chat_id": chat_id, "action": action})).map(|_| ())
    }
}

fn is_command(name: &str) -> bool {
    COMMANDS.iter().any(|(c, _)| *c == name)
}

/// (command, arguments) of a message that reads as one of `COMMANDS` but that Telegram didn't
/// mark as a command: pasted with formatting (e.g. copied from a code block), so it reaches
/// the text handler. The command is case-insensitive and may carry `@botname`.
pub fn parse_command(text: Option<&str>) -> Option<(String, Vec<String>)> {
    let words = py_split(text.unwrap_or(""));
    let first = words.first()?.strip_prefix('/')?;
    let command = first.split('@').next().unwrap_or("").to_lowercase();
    is_command(&command).then(|| (command, words[1..].iter().map(|w| w.to_string()).collect()))
}

/// What an update asks for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Route {
    Command(String, Vec<String>),
    Unknown,
}

/// The chat and route of an update (None: not handled: no private text message).
pub fn route(update: &Value, bot_username: &str) -> Option<(i64, Route)> {
    let message = update.get("message").or_else(|| update.get("edited_message"))?;
    if message["chat"]["type"] != "private" {
        return None;
    }
    let chat_id = message["chat"]["id"].as_i64()?;
    let text = message["text"].as_str()?;
    // python-telegram-bot's CommandHandler: a bot_command entity at offset 0 naming one of
    // the commands, for this bot (`@thisbot`) or no bot; arguments: the text's other words.
    if let Some(entity) = message["entities"].as_array().and_then(|e| e.first())
        && entity["type"] == "bot_command"
        && entity["offset"] == 0
    {
        let length = entity["length"].as_u64().unwrap_or(0) as usize;
        let units: Vec<u16> = text.encode_utf16().collect();
        let command = String::from_utf16_lossy(&units[1.min(units.len())..length.min(units.len())]);
        let mut parts = command.splitn(2, '@');
        let name = parts.next().unwrap_or("").to_lowercase();
        let bot = parts.next().unwrap_or(bot_username).to_lowercase();
        if is_command(&name) && bot == bot_username.to_lowercase() {
            let args = py_split(text)[1..].iter().map(|w| w.to_string()).collect();
            return Some((chat_id, Route::Command(name, args)));
        }
    }
    Some(match parse_command(Some(text)) {
        Some((command, args)) => (chat_id, Route::Command(command, args)),
        None => (chat_id, Route::Unknown),
    })
}

/// Send each reply (split to Telegram's length limit). A chat that refuses messages (blocked
/// the bot, or never started it: e.g. the "you have access" note after /allow) is logged and
/// skipped; the other replies still go.
pub fn send_replies(api: &dyn TelegramApi, replies: &[Reply]) -> Result<(), TgError> {
    for reply in replies {
        for piece in split_message(&reply.text, MAX_MESSAGE) {
            match api.send_message(reply.chat_id, &piece) {
                Ok(()) => {}
                Err(e @ (TgError::Forbidden(_) | TgError::BadRequest(_))) => {
                    warn!("reply to chat {} refused: {}", reply.chat_id, e.kind());
                    break;
                }
                Err(e) => return Err(e),
            }
        }
    }
    Ok(())
}

/// Where errors go: the admin alert's token and chat.
pub struct AdminAlerts {
    pub token: Option<String>,
    pub chat_id: Option<String>,
}

fn panic_text(panic: Box<dyn std::any::Any + Send>) -> String {
    panic
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "panic".to_string())
}

/// Handle one update (module docs).
pub fn handle_update(
    service: &BotService,
    api: &dyn TelegramApi,
    update: &Value,
    bot_username: &str,
    admin: &AdminAlerts,
) {
    let Some((chat_id, route)) = route(update, bot_username) else { return };
    let outcome = catch_unwind(AssertUnwindSafe(|| -> Result<()> {
        let replies = match &route {
            Route::Command(command, args) => {
                if command == "plan" || command == "chips" {
                    api.send_chat_action(chat_id, "typing")?;
                }
                service.run(command, chat_id, args)?
            }
            Route::Unknown => service.unknown(chat_id)?,
        };
        send_replies(api, &replies)?;
        Ok(())
    }));
    let (kind, text) = match outcome {
        Ok(Ok(())) => return,
        Ok(Err(err)) => {
            let kind = err.downcast_ref::<TgError>().map_or("Error", TgError::kind);
            (kind, format!("{err:#}"))
        }
        Err(panic) => ("Panic", panic_text(panic)),
    };
    error!("bot update failed: {kind}: {}", redact(&text));
    let summary = text.lines().next().unwrap_or("");
    send_admin_alert(
        &redact(&format!("fplopt bot: {kind}: {summary}")),
        admin.token.as_deref(),
        admin.chat_id.as_deref(),
    );
    if let Err(e) = api.send_message(chat_id, APOLOGY) {
        error!("couldn't tell the user about the error: {}", e.kind());
    }
}

/// Long polling until `stopped()` (checked between polls; SIGTERM sets it): Ok then. Pending
/// updates are processed (`drop_pending_updates=False`).
pub fn run_polling(
    service: &BotService,
    bot: &Telegram,
    admin: &AdminAlerts,
    stopped: &dyn Fn() -> bool,
) -> Result<()> {
    let username = match bot.username() {
        Ok(name) => name,
        Err(TgError::InvalidToken) => bail!("Telegram rejected the bot token"),
        Err(e) => bail!("getMe failed: {}: {e}", e.kind()),
    };
    if let Err(e) = bot.call("deleteWebhook", &json!({"drop_pending_updates": false})) {
        warn!("deleteWebhook: {}: {e}", e.kind());
    }
    let mut offset = None;
    let mut wait = 1.0f64;
    while !stopped() {
        match bot.get_updates(offset) {
            Ok(updates) => {
                wait = 1.0;
                for update in updates {
                    if let Some(id) = update["update_id"].as_i64() {
                        offset = Some(id + 1);
                    }
                    handle_update(service, bot, &update, &username, admin);
                }
            }
            Err(TgError::InvalidToken) => bail!("Telegram rejected the bot token"),
            Err(
                e @ (TgError::Network(_)
                | TgError::Status(..)
                | TgError::Conflict(_)
                | TgError::RetryAfter(_)),
            ) => {
                // Polling errors (a Telegram 5xx, a network blip, two instances during a
                // restart): retried with backoff; alerting each would spam.
                warn!("polling: {}: {e}", e.kind());
                let pause = if let TgError::RetryAfter(s) = e { s as f64 } else { wait };
                sleep(Duration::from_secs_f64(pause));
                wait = (wait * 2.0).min(30.0);
            }
            Err(e) => {
                error!("polling failed: {}: {e}", e.kind());
                send_admin_alert(
                    &redact(&format!("fplopt bot: {}: {e}", e.kind())),
                    admin.token.as_deref(),
                    admin.chat_id.as_deref(),
                );
                sleep(Duration::from_secs_f64(wait));
                wait = (wait * 2.0).min(30.0);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Python test_parse_command_reads_a_pasted_command.
    #[test]
    fn pasted_commands() {
        let parsed = |t: Option<&str>| parse_command(t);
        assert_eq!(
            parsed(Some("/transfer Saka -> Mbeumo")),
            Some((
                "transfer".to_string(),
                vec!["Saka".to_string(), "->".to_string(), "Mbeumo".to_string()]
            ))
        );
        assert_eq!(parsed(Some("  /PLAN@fplopt_bot ")), Some(("plan".to_string(), vec![])));
        assert_eq!(parsed(Some("/nope x")), None);
        assert_eq!(parsed(Some("transfer Saka")), None);
        assert_eq!(parsed(Some("")), None);
        assert_eq!(parsed(None), None);
    }

    fn message(text: &str, entity: Option<usize>, chat_type: &str) -> Value {
        let mut m = json!({"message": {"chat": {"id": 5, "type": chat_type}, "text": text}});
        if let Some(length) = entity {
            m["message"]["entities"] =
                json!([{"type": "bot_command", "offset": 0, "length": length}]);
        }
        m
    }

    #[test]
    fn routes() {
        let cmd = |c: &str, a: &[&str]| {
            Some((5, Route::Command(c.to_string(), a.iter().map(|s| s.to_string()).collect())))
        };
        assert_eq!(route(&message("/plan", Some(5), "private"), "fplopt_bot"), cmd("plan", &[]));
        assert_eq!(
            route(&message("/PLAN@FPLOPT_bot x", Some(16), "private"), "fplopt_bot"),
            cmd("plan", &["x"])
        );
        // Another bot's command still reaches the text handler, which parses it.
        assert_eq!(
            route(&message("/plan@other", Some(11), "private"), "fplopt_bot"),
            cmd("plan", &[])
        );
        // A command entity shorter than the first word: the CommandHandler's reading.
        assert_eq!(
            route(&message("/plan,now", Some(5), "private"), "fplopt_bot"),
            cmd("plan", &[])
        );
        assert_eq!(
            route(&message("/plan,now", None, "private"), "fplopt_bot"),
            Some((5, Route::Unknown))
        );
        assert_eq!(
            route(&message("hello", None, "private"), "fplopt_bot"),
            Some((5, Route::Unknown))
        );
        assert_eq!(
            route(&message("/setft 2", None, "private"), "fplopt_bot"),
            cmd("setft", &["2"])
        );
        assert_eq!(route(&message("/plan", Some(5), "group"), "fplopt_bot"), None);
        assert_eq!(route(&json!({"message": {"chat": {"id": 5, "type": "private"}}}), "b"), None);
        let edited =
            json!({"edited_message": {"chat": {"id": 5, "type": "private"}, "text": "/help"}});
        assert_eq!(route(&edited, "b"), cmd("help", &[]));
    }

    struct Fake {
        sent: std::cell::RefCell<Vec<(i64, String)>>,
        refuse: i64,
    }

    impl TelegramApi for Fake {
        fn send_message(&self, chat_id: i64, text: &str) -> Result<(), TgError> {
            if chat_id == self.refuse {
                return Err(TgError::Forbidden("Forbidden: bot was blocked by the user".into()));
            }
            self.sent.borrow_mut().push((chat_id, text.to_string()));
            Ok(())
        }

        fn send_chat_action(&self, _: i64, _: &str) -> Result<(), TgError> {
            Ok(())
        }
    }

    // Python test_send_replies_splits_long_messages, test_send_replies_skips_chats_that_refuse.
    #[test]
    fn replies_are_split_and_refusals_skipped() {
        let api = Fake { sent: Default::default(), refuse: 9 };
        let long = vec!["x".repeat(3000); 2].join("\n");
        let replies = [
            Reply { chat_id: 9, text: long.clone() },
            Reply { chat_id: 1, text: long },
            Reply { chat_id: 2, text: "hi".into() },
        ];
        send_replies(&api, &replies).unwrap();
        let sent = api.sent.borrow();
        assert_eq!(
            sent.iter().map(|(c, t)| (*c, t.len())).collect::<Vec<_>>(),
            [(1, 3000), (1, 3000), (2, 2)]
        );
    }

    /// One HTTP request on localhost answered with `status`; the request body comes back.
    fn serve_once(status: u16) -> (String, std::sync::mpsc::Receiver<String>) {
        use std::io::{BufRead, BufReader, Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut length = 0;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if line.trim().is_empty() {
                    break;
                }
                if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    length = v.trim().parse().unwrap();
                }
            }
            let mut body = vec![0; length];
            reader.read_exact(&mut body).unwrap();
            tx.send(String::from_utf8(body).unwrap()).unwrap();
            let ok = status == 200;
            let payload = format!(r#"{{"ok": {ok}, "result": true, "description": "x"}}"#);
            write!(
                stream,
                "HTTP/1.1 {status} X
Content-Type: application/json
Content-Length: {}

{payload}",
                payload.len()
            )
            .unwrap();
        });
        (url, rx)
    }

    // Python test_telegram_sender_posts_html_and_never_leaks_the_token.
    #[test]
    fn sender_posts_html_and_never_leaks_the_token() {
        let (url, seen) = serve_once(200);
        let bot = Telegram::with_api(&url, "123:SECRET").unwrap();
        bot.send_html(1, "<b>hi</b>").unwrap();
        let body: Value = serde_json::from_str(&seen.recv().unwrap()).unwrap();
        assert_eq!(
            body,
            json!({"chat_id": 1, "text": "<b>hi</b>", "parse_mode": "HTML", "link_preview_options": {"is_disabled": true}})
        );
        let (url, _seen) = serve_once(403);
        let err = Telegram::with_api(&url, "123:SECRET").unwrap().send_html(2, "x").unwrap_err();
        assert!(err.permanent);
        assert_eq!(err.message, "sendMessage to chat 2 failed: HTTPStatusError (HTTP 403)");
        // No response at all: the transport's error kind, never the URL.
        let err = Telegram::with_api("http://127.0.0.1:9", "123:SECRET")
            .unwrap()
            .send_html(3, "x")
            .unwrap_err();
        assert!(!err.permanent && !err.message.contains("SECRET"), "{}", err.message);
    }
}
