//! HTTP fetching with retries for flaky public APIs. Port of `fplopt.adapters.http`.

use std::fmt;
use std::io::Read;
use std::time::Duration;

use reqwest::blocking::Client;
use reqwest::header::HeaderMap;

pub const USER_AGENT: &str = "fpl-optimizer/0.1 (+https://github.com/samintisar/fpl-optimizer)";
/// For the connection, the response headers and each read of the body, like httpx's
/// `Timeout(30.0)` (not a cap on the whole body).
pub const TIMEOUT: Duration = Duration::from_secs(30);
pub const RETRY_AFTER_CAP_S: f64 = 120.0;
pub const DEFAULT_ATTEMPTS: u32 = 5;
const ERROR_BODY_CAP: u64 = 64 << 10;

/// A failed GET. Messages name the URL without its query string (the odds API key travels
/// there).
#[derive(Debug)]
pub enum FetchError {
    /// A non-2xx response (`httpx.HTTPStatusError`).
    /// `body`: the first `ERROR_BODY_CAP` bytes, for APIs that explain errors there.
    Status { status: u16, reason: String, url: String, retry_after: Option<u64>, body: Vec<u8> },
    /// Connect, timeout or read failure (`httpx.TransportError`).
    Transport { kind: &'static str, url: String },
    /// A 2xx response whose body failed validation (`InvalidPayload`).
    Invalid { message: String, url: String },
}

impl FetchError {
    /// The Python exception name.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Status { .. } => "HTTPStatusError",
            Self::Transport { kind, .. } => kind,
            Self::Invalid { .. } => "InvalidPayload",
        }
    }

    pub fn status(&self) -> Option<u16> {
        match self {
            Self::Status { status, .. } => Some(*status),
            _ => None,
        }
    }

    fn retryable(&self) -> bool {
        match self {
            Self::Status { status, .. } => *status == 429 || *status >= 500,
            Self::Transport { .. } | Self::Invalid { .. } => true,
        }
    }

    fn from_reqwest(err: &reqwest::Error, url: &str) -> Self {
        let kind = if err.is_timeout() {
            if err.is_connect() { "ConnectTimeout" } else { "ReadTimeout" }
        } else if err.is_connect() {
            "ConnectError"
        } else if err.is_redirect() {
            "TooManyRedirects"
        } else if err.is_body() || err.is_decode() {
            "ReadError"
        } else {
            "TransportError"
        };
        Self::Transport { kind, url: url.to_owned() }
    }
}

impl fmt::Display for FetchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Status { status, reason, url, .. } => {
                let what = match status {
                    100..=199 => "Informational response",
                    300..=399 => "Redirect response",
                    400..=499 => "Client error",
                    500..=599 => "Server error",
                    _ => "Error",
                };
                write!(f, "{what} '{status} {reason}' for url '{url}'")
            }
            Self::Transport { kind, url } => write!(f, "{kind} for url '{url}'"),
            Self::Invalid { message, url } => write!(f, "{message}: {url}"),
        }
    }
}

impl std::error::Error for FetchError {}

/// Body validation: `Err(message)` for a bad payload.
pub type Validator<'a> = &'a dyn Fn(&[u8]) -> Result<(), String>;

pub fn require_json(content: &[u8]) -> Result<(), String> {
    serde_json::from_slice::<serde::de::IgnoredAny>(strip_bom(content))
        .map(|_| ())
        .map_err(|_| "response is not JSON".to_owned())
}

/// Python's `json.loads(bytes)` accepts a UTF-8 BOM.
pub fn strip_bom(content: &[u8]) -> &[u8] {
    content.strip_prefix(b"\xef\xbb\xbf").unwrap_or(content)
}

#[derive(Debug)]
pub struct Response {
    pub status: u16,
    pub headers: HeaderMap,
    pub body: Vec<u8>,
}

type Wait = Box<dyn Fn(u32, &FetchError) -> Duration + Send + Sync>;

pub struct Http {
    client: Client,
    wait: Wait,
}

impl Http {
    /// The archiver's client: our User-Agent, 30 s connect/read timeouts, redirects followed,
    /// `default_wait` between attempts.
    pub fn new() -> reqwest::Result<Self> {
        Ok(Self::with_wait(make_client()?, Box::new(default_wait)))
    }

    pub fn with_wait(client: Client, wait: Wait) -> Self {
        Self { client, wait }
    }

    /// GET `url`, retrying transient failures (429, 5xx, transport errors, invalid
    /// payloads) up to `attempts` times in all; the last error is returned.
    pub fn get(
        &self,
        url: &str,
        params: &[(&str, &str)],
        validate: Option<Validator<'_>>,
        attempts: u32,
    ) -> Result<Response, FetchError> {
        let mut attempt = 1;
        loop {
            match self.get_once(url, params, validate) {
                Ok(response) => return Ok(response),
                Err(err) if attempt < attempts && err.retryable() => {
                    let wait = (self.wait)(attempt, &err);
                    log::debug!("{err}; attempt {attempt} of {attempts}, retrying in {wait:?}");
                    std::thread::sleep(wait);
                    attempt += 1;
                }
                Err(err) => return Err(err),
            }
        }
    }

    /// `get` with the body guaranteed to be JSON.
    pub fn get_json(
        &self,
        url: &str,
        params: &[(&str, &str)],
        attempts: u32,
    ) -> Result<Response, FetchError> {
        self.get(url, params, Some(&require_json), attempts)
    }

    /// A streamed GET (no retries): the response, for reading the body incrementally.
    pub fn stream(&self, url: &str) -> Result<reqwest::blocking::Response, FetchError> {
        let response =
            self.client.get(url).send().map_err(|e| FetchError::from_reqwest(&e, url))?;
        check_status(response, url)
    }

    fn get_once(
        &self,
        url: &str,
        params: &[(&str, &str)],
        validate: Option<Validator<'_>>,
    ) -> Result<Response, FetchError> {
        let response = self
            .client
            .get(url)
            .query(params)
            .send()
            .map_err(|e| FetchError::from_reqwest(&e, url))?;
        let mut response = check_status(response, url)?;
        let status = response.status().as_u16();
        let headers = response.headers().clone();
        // Read, not `bytes()`: the blocking client's timeout then applies per read.
        let mut body = Vec::new();
        response.read_to_end(&mut body).map_err(|e| read_error(&e, url))?;
        if let Some(validate) = validate {
            validate(&body)
                .map_err(|message| FetchError::Invalid { message, url: url.to_owned() })?;
        }
        Ok(Response { status, headers, body })
    }
}

fn read_error(err: &std::io::Error, url: &str) -> FetchError {
    let kind = if err.kind() == std::io::ErrorKind::TimedOut { "ReadTimeout" } else { "ReadError" };
    FetchError::Transport { kind, url: url.to_owned() }
}

pub fn make_client() -> reqwest::Result<Client> {
    Client::builder().user_agent(USER_AGENT).timeout(TIMEOUT).connect_timeout(TIMEOUT).build()
}

fn check_status(
    response: reqwest::blocking::Response,
    url: &str,
) -> Result<reqwest::blocking::Response, FetchError> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let retry_after = response
        .headers()
        .get("Retry-After")
        .and_then(|v| v.to_str().ok())
        .filter(|v| !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit()))
        .and_then(|v| v.parse().ok());
    let mut body = Vec::new();
    let _ = response.take(ERROR_BODY_CAP).read_to_end(&mut body); // best effort
    Err(FetchError::Status {
        status: status.as_u16(),
        reason: status.canonical_reason().unwrap_or("").to_owned(),
        url: url.to_owned(),
        retry_after,
        body,
    })
}

/// Honour a 429's Retry-After (capped); otherwise exponential backoff with jitter
/// (tenacity's `wait_exponential_jitter(initial=2, max=60)`: 2·2^(n-1) + U(0, 1), capped).
pub fn default_wait(attempt: u32, err: &FetchError) -> Duration {
    if let FetchError::Status { status: 429, retry_after: Some(seconds), .. } = err {
        return Duration::from_secs_f64((*seconds as f64).min(RETRY_AFTER_CAP_S));
    }
    let exp = 2.0 * 2f64.powi(attempt.saturating_sub(1).min(30) as i32);
    Duration::from_secs_f64((exp + fastrand::f64()).min(60.0))
}

#[cfg(test)]
pub(crate) mod testing {
    //! A local HTTP server answering from a handler, for adapter tests.

    use std::sync::{Arc, Mutex};
    use std::thread;

    pub struct Request {
        pub path: String,
        pub query: String,
    }

    /// (status, headers, body) for a request.
    pub type Reply = (u16, Vec<(&'static str, String)>, Vec<u8>);

    pub struct Server {
        pub base: String,
        pub seen: Arc<Mutex<Vec<Request>>>,
    }

    impl Server {
        pub fn requests(&self) -> usize {
            self.seen.lock().unwrap().len()
        }
    }

    pub fn serve(handler: impl Fn(&Request, usize) -> Reply + Send + 'static) -> Server {
        let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
        let base = format!("http://{}", server.server_addr().to_ip().unwrap());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::clone(&seen);
        thread::spawn(move || {
            for request in server.incoming_requests() {
                let (path, query) = match request.url().split_once('?') {
                    Some((p, q)) => (p.to_owned(), q.to_owned()),
                    None => (request.url().to_owned(), String::new()),
                };
                let req = Request { path, query };
                let n = log.lock().unwrap().len();
                let (status, headers, body) = handler(&req, n);
                log.lock().unwrap().push(req);
                let mut response = tiny_http::Response::from_data(body).with_status_code(status);
                for (name, value) in headers {
                    response.add_header(
                        tiny_http::Header::from_bytes(name.as_bytes(), value.as_bytes()).unwrap(),
                    );
                }
                let _ = request.respond(response);
            }
        });
        Server { base, seen }
    }

    pub fn no_wait() -> super::Http {
        super::Http::with_wait(super::make_client().unwrap(), Box::new(|_, _| Default::default()))
    }
}

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;

    #[test]
    fn retries_5xx_then_returns_body() {
        let server =
            serve(|_, n| if n < 2 { (503, vec![], vec![]) } else { (200, vec![], b"{}".to_vec()) });
        let r = no_wait().get_json(&format!("{}/x/", server.base), &[], 5).unwrap();
        assert_eq!(r.body, b"{}");
        assert_eq!(server.requests(), 3);
    }

    #[test]
    fn retries_invalid_payloads_and_gives_up() {
        let server = serve(|_, _| (200, vec![], b"<html>updating</html>".to_vec()));
        let err = no_wait().get_json(&format!("{}/x/", server.base), &[], 3).unwrap_err();
        assert_eq!(err.kind(), "InvalidPayload");
        assert!(err.to_string().starts_with("response is not JSON: http://"), "{err}");
        assert_eq!(server.requests(), 3);
    }

    #[test]
    fn does_not_retry_404_and_hides_the_query() {
        let server = serve(|_, _| (404, vec![], vec![]));
        let url = format!("{}/x/", server.base);
        let err = no_wait().get(&url, &[("apiKey", "SECRET")], None, 5).unwrap_err();
        assert_eq!(err.status(), Some(404));
        assert_eq!(err.to_string(), format!("Client error '404 Not Found' for url '{url}'"));
        assert_eq!(server.requests(), 1);
        assert_eq!(server.seen.lock().unwrap()[0].query, "apiKey=SECRET");
    }

    #[test]
    fn honours_retry_after_capped() {
        let err = FetchError::Status {
            status: 429,
            reason: String::new(),
            url: String::new(),
            retry_after: Some(500),
            body: vec![],
        };
        assert_eq!(default_wait(1, &err), Duration::from_secs(120));
        let other = FetchError::Transport { kind: "ConnectError", url: String::new() };
        let w = default_wait(2, &other).as_secs_f64();
        assert!((4.0..5.0).contains(&w), "{w}");
        assert_eq!(default_wait(10, &other), Duration::from_secs(60));
    }

    #[test]
    fn connection_refused_is_a_transport_error() {
        let err = no_wait().get("http://127.0.0.1:9/x", &[], None, 2).unwrap_err();
        assert_eq!(err.kind(), "ConnectError");
    }

    #[test]
    fn json_validation_accepts_a_bom() {
        assert!(require_json(b"\xef\xbb\xbf{\"a\": 1}").is_ok());
        assert!(require_json(b"nope").is_err());
    }
}
