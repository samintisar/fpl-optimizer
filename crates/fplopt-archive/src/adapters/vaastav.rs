//! vaastav/Fantasy-Premier-League at a pinned commit: git tree listing + raw file bytes.
//! Port of `fplopt.adapters.vaastav`.

use std::collections::BTreeMap;

use anyhow::Result;
use sha1::{Digest, Sha1};

use super::VaastavSource;
use crate::Failure;
use crate::fetch::{DEFAULT_ATTEMPTS, Http};

pub const REPO: &str = "vaastav/Fantasy-Premier-League";
pub const PINNED_COMMIT: &str = "9779cdbc0c07f6c900c2d0c181ddf6bb9c800f88"; // 2026-08-28, "Add 26/27 gw1 data"
pub const API_URL: &str = "https://api.github.com";
pub const RAW_URL: &str = "https://raw.githubusercontent.com";

/// The sha git uses for a file's content, to verify downloads against the tree listing.
pub fn git_blob_sha(data: &[u8]) -> String {
    let mut hasher = Sha1::new();
    hasher.update(format!("blob {}\0", data.len()).as_bytes());
    hasher.update(data);
    hasher.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// `urllib.parse.quote(path, safe='/')`.
pub fn quote_path(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    for b in path.bytes() {
        if b.is_ascii_alphanumeric() || b"_.-~/".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

pub struct VaastavClient<'a> {
    http: &'a Http,
    api: String,
    raw: String,
}

impl<'a> VaastavClient<'a> {
    pub fn new(http: &'a Http, api_url: &str, raw_url: &str) -> Self {
        Self {
            http,
            api: api_url.trim_end_matches('/').to_owned(),
            raw: raw_url.trim_end_matches('/').to_owned(),
        }
    }
}

impl VaastavSource for VaastavClient<'_> {
    /// Refuses a truncated listing.
    fn tree(&self, commit: &str) -> Result<BTreeMap<String, String>> {
        let url = format!("{}/repos/{REPO}/git/trees/{commit}", self.api);
        let body = self.http.get_json(&url, &[("recursive", "1")], DEFAULT_ATTEMPTS)?.body;
        let data: serde_json::Value = serde_json::from_slice(crate::fetch::strip_bom(&body))?;
        if crate::truthy(&data["truncated"]) {
            return Err(Failure::error(
                "RuntimeError",
                format!("git tree listing for {REPO}@{commit} is truncated"),
            ));
        }
        let entries =
            data["tree"].as_array().ok_or_else(|| Failure::error("KeyError", "'tree'"))?;
        let mut blobs = BTreeMap::new();
        for entry in entries {
            if entry["type"] == "blob" {
                let path =
                    entry["path"].as_str().ok_or_else(|| Failure::error("KeyError", "'path'"))?;
                let sha =
                    entry["sha"].as_str().ok_or_else(|| Failure::error("KeyError", "'sha'"))?;
                blobs.insert(path.to_owned(), sha.to_owned());
            }
        }
        Ok(blobs)
    }

    fn file(&self, commit: &str, path: &str, blob_sha: &str) -> Result<Vec<u8>> {
        let url = format!("{}/{REPO}/{commit}/{}", self.raw, quote_path(path));
        let content = self.http.get(&url, &[], None, DEFAULT_ATTEMPTS)?.body;
        if git_blob_sha(&content) != blob_sha {
            return Err(Failure::error(
                "BlobMismatch",
                format!("{path}: content does not match blob {blob_sha}"),
            ));
        }
        Ok(content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fetch::testing::{no_wait, serve};

    #[test]
    fn blob_sha_matches_git() {
        // `printf 'hello\n' | git hash-object --stdin`
        assert_eq!(git_blob_sha(b"hello\n"), "ce013625030ba8dba906f756967f9e9ca394464a");
    }

    #[test]
    fn quoting() {
        assert_eq!(quote_path("data/2016-17/gws/merged_gw.csv"), "data/2016-17/gws/merged_gw.csv");
        assert_eq!(quote_path("data/a b/ü.csv"), "data/a%20b/%C3%BC.csv");
    }

    #[test]
    fn tree_and_verified_files() {
        let server = serve(|req, _| {
            match req.path.as_str() {
            p if p.contains("/git/trees/") => (
                200,
                vec![],
                br#"{"tree": [{"path": "data/a.csv", "type": "blob", "sha": "ce013625030ba8dba906f756967f9e9ca394464a"},
                              {"path": "data", "type": "tree", "sha": "x"}], "truncated": false}"#
                    .to_vec(),
            ),
            _ => (200, vec![], b"hello\n".to_vec()),
        }
        });
        let http = no_wait();
        let v = VaastavClient::new(&http, &server.base, &server.base);
        let tree = v.tree("abc").unwrap();
        assert_eq!(tree.len(), 1);
        let sha = &tree["data/a.csv"];
        assert_eq!(v.file("abc", "data/a.csv", sha).unwrap(), b"hello\n");
        let err = v.file("abc", "data/a.csv", "0000").unwrap_err();
        assert_eq!(crate::kind_name(&err), "BlobMismatch");
        let seen = server.seen.lock().unwrap();
        assert_eq!(seen[0].query, "recursive=1");
        assert_eq!(seen[1].path, format!("/{REPO}/abc/data/a.csv"));
    }

    #[test]
    fn truncated_tree_is_refused() {
        let server = serve(|_, _| (200, vec![], br#"{"tree": [], "truncated": true}"#.to_vec()));
        let http = no_wait();
        let err = VaastavClient::new(&http, &server.base, &server.base).tree("abc").unwrap_err();
        assert!(err.to_string().ends_with("is truncated"));
    }
}
