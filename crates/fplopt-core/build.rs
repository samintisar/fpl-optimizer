//! Embeds the checkout's commit as `FPLOPT_GIT_SHA` (the experiment log's `git_sha`): the
//! commit the binary was built from, which is what ran. Rebuilds when HEAD moves.

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    let text = String::from_utf8(out.stdout).ok()?.trim().to_owned();
    (out.status.success() && !text.is_empty()).then_some(text)
}

fn main() {
    let sha = git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=FPLOPT_GIT_SHA={sha}");
    let mut watched = vec!["HEAD".to_owned(), "packed-refs".to_owned()];
    watched.extend(git(&["symbolic-ref", "-q", "HEAD"]));
    for name in watched {
        if let Some(path) = git(&["rev-parse", "--path-format=absolute", "--git-path", &name]) {
            println!("cargo:rerun-if-changed={path}");
        }
    }
}
