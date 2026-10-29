//! Runtime settings from environment variables (the CLI loads .env first). Port of
//! `fplopt.settings`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Settings {
    pub raw_dir: PathBuf,
    pub odds_api_key: Option<String>,
    pub telegram_bot_token: Option<String>,
    pub telegram_admin_chat_id: Option<String>,
    pub data_dir: PathBuf,
    pub healthcheck_ping_url: Option<String>,
    /// The bot's SQLite user state and plan contexts (gitignored).
    pub state_dir: PathBuf,
}

impl Settings {
    /// From the process environment.
    pub fn from_env() -> Self {
        Self::from_map(&std::env::vars().collect())
    }

    pub fn from_map(env: &HashMap<String, String>) -> Self {
        let get = |key: &str| {
            let value = env.get(key).map(|v| v.trim()).unwrap_or("");
            (!value.is_empty()).then(|| value.to_owned())
        };
        let dir = |key: &str, default: &str| resolve(&get(key).unwrap_or_else(|| default.into()));
        Self {
            raw_dir: dir("FPLOPT_RAW_DIR", "raw"),
            odds_api_key: get("ODDS_API_KEY"),
            telegram_bot_token: get("TELEGRAM_BOT_TOKEN"),
            telegram_admin_chat_id: get("TELEGRAM_ADMIN_CHAT_ID"),
            data_dir: dir("FPLOPT_DATA_DIR", "data"),
            healthcheck_ping_url: get("HEALTHCHECK_PING_URL"),
            state_dir: dir("FPLOPT_STATE_DIR", "state"),
        }
    }
}

/// Python's `Path(p).expanduser().resolve()`, without following symlinks.
fn resolve(path: &str) -> PathBuf {
    let expanded = match path.strip_prefix('~') {
        Some(rest) if rest.is_empty() || rest.starts_with(['/', '\\']) => {
            let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"));
            match home {
                Some(home) => Path::new(&home).join(rest.trim_start_matches(['/', '\\'])),
                None => PathBuf::from(path),
            }
        }
        _ => PathBuf::from(path),
    };
    std::path::absolute(&expanded).unwrap_or(expanded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_and_blank_values() {
        let env = HashMap::from([
            ("TELEGRAM_BOT_TOKEN".to_owned(), "  ".to_owned()),
            ("ODDS_API_KEY".to_owned(), " k ".to_owned()),
        ]);
        let s = Settings::from_map(&env);
        assert_eq!(s.telegram_bot_token, None);
        assert_eq!(s.odds_api_key.as_deref(), Some("k"));
        assert!(s.raw_dir.is_absolute() && s.raw_dir.ends_with("raw"));
        assert!(s.data_dir.ends_with("data") && s.state_dir.ends_with("state"));
    }

    #[test]
    fn dirs_from_env() {
        let env = HashMap::from([("FPLOPT_DATA_DIR".to_owned(), "elsewhere/d".to_owned())]);
        assert!(Settings::from_map(&env).data_dir.ends_with("elsewhere/d"));
    }
}
