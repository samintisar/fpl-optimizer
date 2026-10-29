//! Shared bits for the best-effort HTTP calls (alerts, heartbeats).

use std::time::Duration;

use reqwest::blocking::Client;

pub const TIMEOUT: Duration = Duration::from_secs(10);
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

pub fn client() -> reqwest::Result<Client> {
    Client::builder().timeout(TIMEOUT).connect_timeout(CONNECT_TIMEOUT).build()
}

/// What failed, safe to log: the error kind and the HTTP status, never the error's message
/// (it contains the URL, and with it bot tokens or ping keys).
pub fn failure_detail(err: &reqwest::Error) -> String {
    let kind = if err.is_timeout() {
        "Timeout"
    } else if err.is_connect() {
        "ConnectError"
    } else if err.is_status() {
        "HTTPStatusError"
    } else if err.is_builder() {
        "BuilderError"
    } else if err.is_request() {
        "RequestError"
    } else {
        "Error"
    };
    match err.status() {
        Some(status) => format!("{kind} (HTTP {})", status.as_u16()),
        None => kind.to_owned(),
    }
}

#[cfg(test)]
pub(crate) mod testing {
    //! A one-request HTTP server on localhost for the alert and heartbeat tests.

    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::thread;

    pub struct Seen {
        pub method: String,
        pub path: String,
        pub body: String,
    }

    /// Serves one request with `status`; returns the base URL and a receiver for what the
    /// request was.
    pub fn serve_once(status: u16) -> (String, mpsc::Receiver<Seen>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            let mut parts = line.split_whitespace();
            let method = parts.next().unwrap_or_default().to_owned();
            let path = parts.next().unwrap_or_default().to_owned();
            let mut length = 0;
            loop {
                let mut header = String::new();
                reader.read_line(&mut header).unwrap();
                if header.trim().is_empty() {
                    break;
                }
                if let Some((k, v)) = header.split_once(':')
                    && k.eq_ignore_ascii_case("content-length")
                {
                    length = v.trim().parse().unwrap();
                }
            }
            let mut body = vec![0; length];
            reader.read_exact(&mut body).unwrap();
            let mut stream = stream;
            write!(
                stream,
                "HTTP/1.1 {status} X\r\ncontent-length: 2\r\nconnection: close\r\n\r\nok"
            )
            .unwrap();
            let body = String::from_utf8(body).unwrap();
            tx.send(Seen { method, path, body }).unwrap();
        });
        (base, rx)
    }
}
