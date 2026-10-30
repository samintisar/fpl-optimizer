//! `fplopt-rs`: the Rust port's CLI. Subcommands and flags match the Python `fplopt` CLI
//! so a migrated job only swaps the binary name (migration plan §2). Each batch adds the
//! commands it ports: batch 0 the parity check, batch 1 the archiver (`snapshot`,
//! `backfill`, `check freshness`), batch 2 the build (`build`, `rules export`), batch 3 the
//! as-of store and features (`check leakage`, and `features dump` for the parity gate).
//!
//! Archiver jobs run like Python's: a failure is logged and alerted to Telegram (exit 1);
//! SIGTERM stops a long run cleanly, after its manifest is written (exit 143, no alert); a
//! successful `snapshot daily|tick` pings HEALTHCHECK_PING_URL if set.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::Result;
use chrono::{TimeDelta, Utc};
use clap::{CommandFactory, FromArgMatches, Parser, Subcommand, ValueEnum};
use fplopt_archive::adapters::{self, OddsSource, base_url};
use fplopt_archive::fetch::Http;
use fplopt_archive::raw_store::RawStore;
use fplopt_archive::{SystemClock, espn, health, history, jobs, stop};
use fplopt_core::alerts::send_admin_alert;
use fplopt_core::experiments::{GIT_SHA, family_variants, log_experiment};
use fplopt_core::heartbeat::send_heartbeat;
use fplopt_core::redact::redact;
use fplopt_core::settings::Settings;
use fplopt_parity::{Tolerance, diff_tables, read_parquet};
use serde_json::json;

const DEFAULT_EXPERIMENTS: &str = "results/experiments.csv";
const PARITY_FAMILY: &str = "rust parity";
/// Jobs whose success pings HEALTHCHECK_PING_URL: the scheduled archiver runs.
const HEARTBEAT_JOBS: [&str; 2] = ["snapshot daily", "snapshot tick"];
const TEAMS_CSV: &str = "config/teams.csv";

#[derive(Parser)]
#[command(name = "fplopt-rs", about = "Fantasy Premier League decision optimizer (Rust port)")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Archive API snapshots into raw/.
    Snapshot {
        #[arg(value_enum)]
        command: SnapshotJob,
    },
    /// One-off backfills into raw/.
    Backfill {
        #[arg(value_enum)]
        command: BackfillJob,
        /// football-data, espn: first season's start year.
        #[arg(long, default_value_t = history::FIRST_SEASON, value_name = "YEAR")]
        from_season: i32,
    },
    /// Per-season rules config.
    Rules {
        #[arg(value_enum)]
        command: RulesJob,
        /// Season label, e.g. 2026-27.
        season: String,
        /// Output directory.
        #[arg(long, default_value = "config/scoring")]
        out: PathBuf,
    },
    /// Build data/<table>.parquet from raw/.
    Build {
        /// Table name, or 'all'.
        target: String,
    },
    /// Archive health and parity checks.
    Check {
        #[command(subcommand)]
        check: Check,
    },
    /// Feature frames at given deadlines (the batch 3 parity gate; not in the Python CLI).
    Features {
        #[command(subcommand)]
        command: FeaturesJob,
    },
}

#[derive(Subcommand)]
enum FeaturesJob {
    /// Write every feature frame and the history frames at each deadline to
    /// <out>/<NN>/<frame>.parquet (NN: the deadline's line number in the file, from 00).
    Dump {
        /// A file with one ISO 8601 deadline per line.
        #[arg(long)]
        deadlines: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum SnapshotJob {
    Daily,
    Tick,
}

#[derive(Clone, Copy, ValueEnum)]
enum RulesJob {
    Export,
}

#[derive(Clone, Copy, ValueEnum)]
enum BackfillJob {
    ElementSummary,
    FootballData,
    Vaastav,
    Fplcache,
    Espn,
}

#[derive(Subcommand)]
enum Check {
    /// Fail if the newest bootstrap snapshot is missing or too old (a dead-man's switch for
    /// the timers).
    Freshness {
        /// Fail if the newest bootstrap snapshot is older than this.
        #[arg(long, default_value_t = health::DEFAULT_MAX_AGE_HOURS, value_name = "H")]
        max_age_hours: f64,
    },
    /// Corrupt-the-future check of every feature and the history frames on data/ (the
    /// feature part of Python's `check leakage`; models and probes are ported later).
    Leakage {
        /// Sampled deadlines on top of the edge list.
        #[arg(long, default_value_t = fplopt_features::leakcheck::DEFAULT_DEADLINES)]
        deadlines: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Compare Python and Rust outputs (the migration's parity gate).
    Parity {
        #[command(subcommand)]
        area: Parity,
    },
}

#[derive(Subcommand)]
enum Parity {
    /// Two Parquet tables, rows matched on --keys. Exit 1 if they differ.
    Table {
        /// The Python output (the reference).
        #[arg(long)]
        python: PathBuf,
        /// The Rust output.
        #[arg(long)]
        rust: PathBuf,
        /// Key columns, comma-separated.
        #[arg(long, value_delimiter = ',', required = true)]
        keys: Vec<String>,
        #[arg(long, default_value_t = 1e-9)]
        rtol: f64,
        #[arg(long, default_value_t = 0.0)]
        atol: f64,
        /// Report directory (default results/parity/<UTC>-table-<name>).
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long, default_value = DEFAULT_EXPERIMENTS)]
        experiments: PathBuf,
    },
}

fn main() -> ExitCode {
    // Like the Python CLI: .env from the working directory, never overriding the environment.
    let _ = dotenvy::from_path(Path::new(".env"));
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let version = format!("{} ({GIT_SHA})", env!("CARGO_PKG_VERSION"));
    let matches = Cli::command().version(version).get_matches();
    let cli = match Cli::from_arg_matches(&matches) {
        Ok(cli) => cli,
        Err(err) => err.exit(),
    };
    match run(cli) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("error: {err:#}");
            ExitCode::from(2)
        }
    }
}

fn run(cli: Cli) -> Result<ExitCode> {
    match cli.command {
        Command::Snapshot { command } => {
            let name = format!("snapshot {}", value_name(command));
            Ok(run_job(&name, |ctx| match command {
                SnapshotJob::Daily => snapshot_daily(ctx),
                SnapshotJob::Tick => {
                    let odds = ctx.odds();
                    let odds = odds.as_ref().map(|o| o as &dyn OddsSource);
                    jobs::run_tick(&ctx.store, &ctx.fpl(), odds, &SystemClock).map(drop)
                }
            }))
        }
        Command::Backfill { command, from_season } => {
            let name = format!("backfill {}", value_name(command));
            Ok(run_job(&name, |ctx| backfill(ctx, command, from_season)))
        }
        Command::Rules { command: RulesJob::Export, season, out } => {
            // The label is parsed inside the job, so a bad label is alerted like any failure.
            Ok(run_job("rules export", |ctx| {
                let season = fplopt_core::seasons::parse_season_label(&season)?;
                fplopt_build::rules::export_rules(&ctx.store, season, &out).map(drop)
            }))
        }
        Command::Build { target } => Ok(run_job("build", |ctx| {
            let mut build_ctx = fplopt_build::BuildContext::new(
                ctx.store.clone(),
                &ctx.settings.data_dir,
                "config",
            );
            fplopt_build::build(&[target], &mut build_ctx).map(drop)
        })),
        Command::Check { check: Check::Freshness { max_age_hours } } => {
            Ok(run_job("check freshness", |ctx| {
                let max_age = TimeDelta::microseconds((max_age_hours * 3.6e9).round() as i64);
                health::check_freshness(&ctx.store, Utc::now(), max_age).map(drop)
            }))
        }
        Command::Check { check: Check::Leakage { deadlines, seed } } => {
            Ok(run_job("check leakage", |ctx| {
                check_leakage(&ctx.settings.data_dir, deadlines, seed)
            }))
        }
        Command::Features { command: FeaturesJob::Dump { deadlines, out } } => {
            Ok(run_job("features dump", |ctx| {
                features_dump(&ctx.settings.data_dir, &deadlines, &out)
            }))
        }
        Command::Check { check: Check::Parity { area } } => match area {
            Parity::Table { python, rust, keys, rtol, atol, out, experiments } => {
                parity_table(&python, &rust, &keys, Tolerance { rtol, atol }, out, &experiments)
            }
        },
    }
}

/// An ISO 8601 instant with an offset as µs since the epoch (UTC).
fn parse_deadline(text: &str) -> Result<i64> {
    let parsed = chrono::DateTime::parse_from_rfc3339(text.trim())
        .map_err(|e| anyhow::anyhow!("bad deadline {text:?}: {e}"))?;
    Ok(parsed.timestamp_micros())
}

fn features_dump(data_dir: &Path, deadlines: &Path, out: &Path) -> Result<()> {
    let text = std::fs::read_to_string(deadlines)?;
    let deadlines: Vec<i64> =
        text.lines().filter(|l| !l.trim().is_empty()).map(parse_deadline).collect::<Result<_>>()?;
    let store = fplopt_store::DataStore::open(data_dir)?;
    for (i, &deadline) in deadlines.iter().enumerate() {
        let dir = out.join(format!("{i:02}"));
        std::fs::create_dir_all(&dir)?;
        let view = store.as_of(deadline);
        for (name, builder) in fplopt_features::FRAMES {
            let started = std::time::Instant::now();
            let frame = builder(&view)
                .map_err(|e| e.context(format!("{name} at {}", fmt_micros(deadline))))?;
            fplopt_build::table::write_parquet(&frame, &dir.join(format!("{name}.parquet")))?;
            log::debug!(
                "{name}: {} rows, {:.2} s",
                frame.height(),
                started.elapsed().as_secs_f64()
            );
        }
        log::info!("features at {}: written", fmt_micros(deadline));
    }
    Ok(())
}

/// `check leakage`: every feature and the history frames at the edge deadlines plus
/// `n_deadlines` sampled ones; fails listing the leaks.
fn check_leakage(data_dir: &Path, n_deadlines: usize, seed: u64) -> Result<()> {
    use fplopt_features::leakcheck::{FEATURE_TABLES, check_leakage, sample_deadlines};
    let started = std::time::Instant::now();
    let tables = fplopt_store::load_tables(data_dir, FEATURE_TABLES)?;
    let holdout: Vec<i64> =
        fplopt_core::seasons::HOLDOUT_SEASONS.iter().map(|&s| i64::from(s)).collect();
    let deadlines = sample_deadlines(&tables, n_deadlines, seed, &holdout)?;
    let (Some(&first), Some(&last)) = (deadlines.first(), deadlines.last()) else {
        // Checking nothing must not read as a pass.
        anyhow::bail!("no eligible deadlines: the leakage check was not performed");
    };
    let builders = fplopt_features::FRAMES;
    log::info!(
        "leakage check: {} deadline(s) from {} to {}, {} frame(s), seed {seed}",
        deadlines.len(),
        fmt_micros(first),
        fmt_micros(last),
        builders.len()
    );
    let leaks = check_leakage(tables, &deadlines, seed, builders)?;
    if let Some(leak) = leaks.first() {
        for leak in &leaks {
            log::error!(
                "leak: {} @ {} ({}): {}",
                leak.feature,
                fmt_micros(leak.deadline),
                leak.variant,
                leak.detail
            );
        }
        let features: std::collections::BTreeSet<_> = leaks.iter().map(|l| &l.feature).collect();
        let at: std::collections::BTreeSet<_> = leaks.iter().map(|l| l.deadline).collect();
        anyhow::bail!(
            "{} leak(s) in {} builder(s) at {} deadline(s); first: {} @ {} ({}): {}",
            leaks.len(),
            features.len(),
            at.len(),
            leak.feature,
            fmt_micros(leak.deadline),
            leak.variant,
            leak.detail
        );
    }
    log::info!(
        "leakage check passed: {} deadline(s) x {} frame(s) x 2 variants in {:.0} s",
        deadlines.len(),
        builders.len(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

use fplopt_features::leakcheck::fmt_micros;

fn value_name(value: impl ValueEnum) -> String {
    value.to_possible_value().map(|v| v.get_name().to_owned()).unwrap_or_default()
}

/// What an archiver job runs with. Base URLs can be redirected with `FPLOPT_<NAME>_URL`
/// (`adapters::base_url`), for the parity harness only.
struct JobContext {
    store: RawStore,
    http: Http,
    settings: Settings,
}

impl JobContext {
    fn fpl(&self) -> adapters::fpl::FplClient<'_> {
        adapters::fpl::FplClient::new(&self.http, &base_url("FPL", adapters::fpl::BASE_URL))
    }

    fn odds(&self) -> Option<adapters::odds::OddsClient<'_>> {
        let key = self.settings.odds_api_key.as_deref()?;
        let url = base_url("ODDS", adapters::odds::BASE_URL);
        Some(adapters::odds::OddsClient::new(&self.http, key, &url))
    }

    fn football_data(&self) -> adapters::football_data::FootballDataClient<'_> {
        let url = base_url("FOOTBALL_DATA", adapters::football_data::BASE_URL);
        adapters::football_data::FootballDataClient::new(&self.http, &url)
    }

    fn espn(&self) -> adapters::espn::EspnClient<'_> {
        adapters::espn::EspnClient::new(&self.http, &base_url("ESPN", adapters::espn::BASE_URL))
    }
}

fn snapshot_daily(ctx: &JobContext) -> Result<()> {
    let (fpl, odds, fd, espn) = (ctx.fpl(), ctx.odds(), ctx.football_data(), ctx.espn());
    let odds = odds.as_ref().map(|o| o as &dyn OddsSource);
    let daily_espn = jobs::DailyEspn { source: &espn, teams_csv: Path::new(TEAMS_CSV) };
    jobs::run_daily(&ctx.store, &fpl, odds, &SystemClock, Some(&fd), Some(daily_espn))
}

fn backfill(ctx: &JobContext, command: BackfillJob, from_season: i32) -> Result<()> {
    let clock = &SystemClock;
    let github = || base_url("GITHUB_API", adapters::vaastav::API_URL);
    match command {
        BackfillJob::ElementSummary => jobs::backfill_element_summaries(
            &ctx.store,
            &ctx.fpl(),
            clock,
            jobs::ELEMENT_SUMMARY_PAUSE,
            jobs::MAX_CONSECUTIVE_FAILURES,
        )
        .map(drop),
        BackfillJob::FootballData => history::backfill_football_data(
            &ctx.store,
            &ctx.football_data(),
            clock,
            history::FOOTBALL_DATA_PAUSE,
            from_season,
        )
        .map(drop),
        BackfillJob::Espn => {
            let team_ids = espn::espn_team_ids(Path::new(TEAMS_CSV))?;
            espn::backfill_espn(&ctx.store, &ctx.espn(), &team_ids, from_season, clock, espn::PAUSE)
        }
        BackfillJob::Vaastav => {
            let raw = base_url("GITHUB_RAW", adapters::vaastav::RAW_URL);
            let source = adapters::vaastav::VaastavClient::new(&ctx.http, &github(), &raw);
            history::backfill_vaastav(
                &ctx.store,
                &source,
                adapters::vaastav::PINNED_COMMIT,
                clock,
                history::VAASTAV_PAUSE,
                jobs::MAX_CONSECUTIVE_FAILURES,
            )
            .map(drop)
        }
        BackfillJob::Fplcache => {
            let codeload = base_url("CODELOAD", adapters::fplcache::CODELOAD_URL);
            let source = adapters::fplcache::FplcacheClient::new(&ctx.http, &github(), &codeload);
            history::backfill_fplcache(&ctx.store, &source, clock).map(drop)
        }
    }
}

/// Run an archiver job the way Python's `main` does: alert on failure (exit 1), exit 143
/// without an alert when stopped by SIGTERM, heartbeat after a scheduled run succeeds.
fn run_job(name: &str, job: impl FnOnce(&JobContext) -> Result<()>) -> ExitCode {
    let settings = Settings::from_env();
    if settings.telegram_bot_token.is_none() || settings.telegram_admin_chat_id.is_none() {
        log::warn!(
            "TELEGRAM_BOT_TOKEN/TELEGRAM_ADMIN_CHAT_ID not set: failures will not be alerted"
        );
    }
    log::info!("job '{name}' starting (raw dir {})", settings.raw_dir.display());
    register_sigterm();
    let result = Http::new().map_err(anyhow::Error::from).and_then(|http| {
        let store = RawStore::new(&settings.raw_dir);
        job(&JobContext { store, http, settings: settings.clone() })
    });
    match result {
        Ok(()) => {
            log::info!("job '{name}' done");
            if let Some(url) = &settings.healthcheck_ping_url
                && HEARTBEAT_JOBS.contains(&name)
            {
                send_heartbeat(url, None);
            }
            ExitCode::SUCCESS
        }
        Err(err) if err.is::<stop::Terminated>() => {
            log::warn!("job '{name}' stopped by SIGTERM");
            ExitCode::from(143)
        }
        Err(err) => {
            log::error!("job '{name}' failed: {}", redact(&format!("{err:#}")));
            let host = gethostname::gethostname().to_string_lossy().into_owned();
            let text = format!(
                "fplopt-rs {name} failed on {host}: {}: {}",
                fplopt_archive::kind_name(&err),
                fplopt_archive::summary(&err)
            );
            send_admin_alert(
                &redact(&text),
                settings.telegram_bot_token.as_deref(),
                settings.telegram_admin_chat_id.as_deref(),
            );
            ExitCode::from(1)
        }
    }
}

/// SIGTERM sets the stop flag the long loops check (`fplopt_archive::stop`).
fn register_sigterm() {
    #[cfg(unix)]
    if let Err(err) = signal_hook::flag::register(signal_hook::consts::SIGTERM, stop::flag()) {
        log::warn!("could not install the SIGTERM handler: {err}");
    }
}

fn parity_table(
    python: &Path,
    rust: &Path,
    keys: &[String],
    tol: Tolerance,
    out: Option<PathBuf>,
    experiments: &Path,
) -> Result<ExitCode> {
    let diff = diff_tables(&read_parquet(python)?, &read_parquet(rust)?, keys, tol)?;
    let passed = diff.passed();
    let name = python.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let out = out.unwrap_or_else(|| {
        let stamp = Utc::now().format("%Y%m%dT%H%M%SZ");
        PathBuf::from("results").join("parity").join(format!("{stamp}-table-{name}"))
    });
    std::fs::create_dir_all(&out)?;
    let report =
        json!({"passed": passed, "python": python, "rust": rust, "tolerance": tol, "diff": diff});
    std::fs::write(out.join("report.json"), serde_json::to_string_pretty(&report)? + "\n")?;

    let command = std::iter::once("fplopt-rs".to_owned())
        .chain(std::env::args().skip(1))
        .collect::<Vec<_>>()
        .join(" ");
    let config = json!({"python": python, "rust": rust, "keys": keys, "rtol": tol.rtol,
        "atol": tol.atol, "out": out});
    let mismatches: serde_json::Map<String, serde_json::Value> =
        diff.columns.iter().map(|c| (c.column.clone(), json!(c.mismatches))).collect();
    let metrics = json!({"table": name, "passed": passed, "rows_python": diff.rows_python,
        "rows_rust": diff.rows_rust, "only_in_python": diff.only_in_python,
        "only_in_rust": diff.only_in_rust, "dtype_mismatches": diff.dtype_mismatches.len(),
        "missing_columns": diff.missing_in_rust.len() + diff.missing_in_python.len(),
        "column_mismatches": mismatches});
    let n_variants = family_variants(experiments, PARITY_FAMILY)? + 1;
    log_experiment(experiments, &command, &config, &metrics, n_variants, PARITY_FAMILY)?;

    println!("{}: {}", name, if passed { "PASS" } else { "FAIL" });
    println!(
        "  rows: python {} / rust {} (compared {})",
        diff.rows_python, diff.rows_rust, diff.rows_compared
    );
    for (label, list) in
        [("missing in rust", &diff.missing_in_rust), ("missing in python", &diff.missing_in_python)]
    {
        if !list.is_empty() {
            println!("  columns {label}: {}", list.join(", "));
        }
    }
    for m in &diff.dtype_mismatches {
        println!("  dtype {}: python {} / rust {}", m.column, m.python, m.rust);
    }
    if diff.duplicate_keys_python + diff.duplicate_keys_rust > 0 {
        println!(
            "  duplicate-key rows: python {} / rust {} (values not compared)",
            diff.duplicate_keys_python, diff.duplicate_keys_rust
        );
    }
    for (label, n, samples) in [
        ("only in python", diff.only_in_python, &diff.sample_only_in_python),
        ("only in rust", diff.only_in_rust, &diff.sample_only_in_rust),
    ] {
        if n > 0 {
            println!("  rows {label}: {n} (e.g. {})", samples.join("; "));
        }
    }
    for c in &diff.columns {
        let max = c.max_abs_diff.map(|m| format!(", max |diff| {m:.3e}")).unwrap_or_default();
        println!("  {}: {} mismatches{max}", c.column, c.mismatches);
        for sample in &c.samples {
            println!("    {sample}");
        }
    }
    println!("  report: {}", out.join("report.json").display());
    Ok(if passed { ExitCode::SUCCESS } else { ExitCode::from(1) })
}
