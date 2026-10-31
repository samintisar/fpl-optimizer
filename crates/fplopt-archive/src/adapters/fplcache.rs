//! Randdalf/fplcache: bootstrap-static snapshots ~4x/day since April 2021 (public domain).
//! Port of `fplopt.adapters.fplcache`.

use std::io::{self, BufRead, BufReader, Read};
use std::sync::LazyLock;

use anyhow::Result;
use regex::Regex;

use super::FplcacheSource;
use crate::Failure;
use crate::fetch::{DEFAULT_ATTEMPTS, Http};

pub const REPO: &str = "Randdalf/fplcache";
pub const API_URL: &str = "https://api.github.com";
pub const CODELOAD_URL: &str = "https://codeload.github.com";
pub const READ_BUFFER: usize = 1 << 20;

static SHA: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[0-9a-f]{40}$").unwrap());

pub struct FplcacheClient<'a> {
    http: &'a Http,
    api: String,
    codeload: String,
}

impl<'a> FplcacheClient<'a> {
    pub fn new(http: &'a Http, api_url: &str, codeload_url: &str) -> Self {
        Self {
            http,
            api: api_url.trim_end_matches('/').to_owned(),
            codeload: codeload_url.trim_end_matches('/').to_owned(),
        }
    }
}

impl FplcacheSource for FplcacheClient<'_> {
    /// The commit sha `main` points at now.
    fn head_commit(&self) -> Result<String> {
        let url = format!("{}/repos/{REPO}/commits/main", self.api);
        let body = self.http.get_json(&url, &[], DEFAULT_ATTEMPTS)?.body;
        let data: serde_json::Value = serde_json::from_slice(crate::fetch::strip_bom(&body))?;
        match data["sha"].as_str() {
            Some(sha) if SHA.is_match(sha) => Ok(sha.to_owned()),
            _ => Err(Failure::error("ValueError", format!("no commit sha for {REPO}@main"))),
        }
    }

    /// The repo tarball at `commit` (~0.9 GB), streamed. No retries: a failed download fails
    /// the caller, and so does a truncated one (see `for_each_member`).
    fn tarball(&self, commit: &str) -> Result<Box<dyn Read>> {
        let url = format!("{}/{REPO}/tar.gz/{commit}", self.codeload);
        Ok(Box::new(self.http.stream(&url)?))
    }
}

/// One tar member: its name, size, whether it is a regular file, and its content.
pub struct Member<'a> {
    pub name: String,
    pub size: u64,
    pub is_file: bool,
    pub content: &'a mut dyn Read,
}

/// Stream a `.tar.gz` and call `visit` on each member in order, without holding the archive
/// in memory. The gzip stream must be a single complete member: a body cut short fails
/// (`EOFError`, even at a 512-byte boundary, where tar alone would see a clean end of
/// archive) and so do bytes after it (`ValueError`). After the end-of-archive blocks the
/// rest of the body is read and checked, so the whole download is always verified.
pub fn for_each_member(
    stream: Box<dyn Read>,
    mut visit: impl FnMut(Member<'_>) -> Result<()>,
) -> Result<()> {
    let mut gz = flate2::bufread::GzDecoder::new(BufReader::with_capacity(READ_BUFFER, stream));
    {
        let mut archive = tar::Archive::new(&mut gz);
        for entry in archive.entries().map_err(truncation)? {
            let mut entry = entry.map_err(truncation)?;
            let name = String::from_utf8_lossy(&entry.path_bytes()).into_owned();
            let size = entry.size();
            let is_file = matches!(
                entry.header().entry_type(),
                tar::EntryType::Regular | tar::EntryType::Continuous
            );
            visit(Member { name, size, is_file, content: &mut entry })?;
        }
    }
    io::copy(&mut gz, &mut io::sink()).map_err(truncation)?;
    let mut rest = gz.into_inner();
    if !rest.fill_buf()?.is_empty() {
        return Err(Failure::error(
            "ValueError",
            "unexpected data after the end of the fplcache gzip stream",
        ));
    }
    Ok(())
}

fn truncation(err: io::Error) -> anyhow::Error {
    if err.kind() == io::ErrorKind::UnexpectedEof {
        Failure::error("EOFError", "fplcache tarball truncated (gzip stream ended early)")
    } else {
        err.into()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Write;

    /// A .tar.gz with the given (name, content) regular files.
    pub fn tar_gz(files: &[(&str, &[u8])]) -> Vec<u8> {
        let mut builder = tar::Builder::new(Vec::new());
        for (name, content) in files {
            let mut header = tar::Header::new_gnu();
            header.set_size(content.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder.append_data(&mut header, name, *content).unwrap();
        }
        let tar = builder.into_inner().unwrap();
        let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        gz.write_all(&tar).unwrap();
        gz.finish().unwrap()
    }

    fn names(data: Vec<u8>) -> Result<Vec<(String, Vec<u8>)>> {
        let mut out = Vec::new();
        for_each_member(Box::new(io::Cursor::new(data)), |m| {
            let mut content = Vec::new();
            m.content.read_to_end(&mut content)?;
            out.push((m.name, content));
            Ok(())
        })?;
        Ok(out)
    }

    #[test]
    fn reads_members_in_order() {
        let data = tar_gz(&[("r/a.txt", b"one"), ("r/b.txt", b"two")]);
        let got = names(data).unwrap();
        assert_eq!(
            got,
            vec![("r/a.txt".into(), b"one".to_vec()), ("r/b.txt".into(), b"two".to_vec())]
        );
    }

    #[test]
    fn truncation_fails_even_at_a_block_boundary() {
        let data = tar_gz(&[("r/a.txt", &[7u8; 5000]), ("r/b.txt", b"two")]);
        for cut in [10, data.len() / 2, data.len() - 8, data.len() - 1] {
            let err = names(data[..cut].to_vec()).unwrap_err();
            assert!(
                ["EOFError", "OSError", "Error"].contains(&crate::kind_name(&err)),
                "cut {cut}: {err:#}"
            );
        }
    }

    #[test]
    fn trailing_bytes_are_an_error() {
        let mut data = tar_gz(&[("r/a.txt", b"one")]);
        data.extend_from_slice(b"garbage");
        let err = names(data).unwrap_err();
        assert!(err.to_string().contains("unexpected data after the end"), "{err:#}");
    }
}
