//! `fplopt-rs`: the Rust port's CLI. Subcommands and flags match the Python `fplopt` CLI
//! so a migrated job only swaps the binary name (migration plan §2). Batch 0 has only the
//! parity check; each batch adds the commands it ports.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::Result;
use chrono::Utc;
use clap::{CommandFactory, FromArgMatches, Parser, Subcommand};
use fplopt_core::experiments::{GIT_SHA, family_variants, log_experiment};
use fplopt_parity::{Tolerance, diff_tables, read_parquet};
use serde_json::json;

const DEFAULT_EXPERIMENTS: &str = "results/experiments.csv";
const PARITY_FAMILY: &str = "rust parity";

#[derive(Parser)]
#[command(name = "fplopt-rs", about = "Fantasy Premier League decision optimizer (Rust port)")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Checks.
    Check {
        #[command(subcommand)]
        check: Check,
    },
}

#[derive(Subcommand)]
enum Check {
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
        Command::Check { check: Check::Parity { area } } => match area {
            Parity::Table { python, rust, keys, rtol, atol, out, experiments } => {
                parity_table(&python, &rust, &keys, Tolerance { rtol, atol }, out, &experiments)
            }
        },
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
