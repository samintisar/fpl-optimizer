//! `chips values` and `chips size` (Python `cli._chips_values`, `_chips_size`; the chip
//! study): the terminal value table from a chipgains run's GW rows, and the full-run window
//! test's null rejection rate on a develop run's paired differences.

use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::Subcommand;
use fplopt_backtest::chip_values::{
    GainRows, estimate_loso, estimate_terminal_values, known_doubles_by_gw,
};
use fplopt_backtest::policies::DOUBLE_LEAD;
use fplopt_backtest::specs::{CHIP_SPLITS, DOUBLE_TABLE_HORIZON, chip_table_file};
use fplopt_backtest::stats::{Diff, FULL_RUN_WINDOW, window_test_null_size};
use fplopt_build::table::read_parquet;
use fplopt_core::pyjson::{Ordered, dumps_indent};
use fplopt_core::seasons::{DEVELOP_SEASONS, season_label};
use fplopt_store::DataStore;

#[derive(Subcommand)]
pub enum ChipsJob {
    /// Estimate chip terminal values per GWs left in the window from a chipgains run.
    Values {
        /// A `backtest run --spec chipgains:...` output directory (develop seasons only).
        #[arg(long)]
        gains: PathBuf,
        /// Where to write the table (default config/chips/terminal-values[-window|-double].json).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Table keys: chip (pooled), chip@window, chip@window+double (reads data/).
        #[arg(long, default_value = "pooled", value_parser = CHIP_SPLITS)]
        split: String,
        /// Add leave-one-season-out tables.
        #[arg(long)]
        loso: bool,
    },
    /// The full-run window test's null rejection rate on resampled real differences.
    Size {
        /// A develop `backtest compare` run's paired.parquet.
        #[arg(long)]
        paired: PathBuf,
        /// Cells per resample.
        #[arg(long, default_value_t = 10)]
        cells: usize,
    },
}

/// Python's `str(Path)`: the platform's separators.
fn py_path(p: &std::path::Path) -> String {
    let text = p.display().to_string();
    if cfg!(windows) { text.replace('/', "\\") } else { text }
}

fn table_json(table: &[(String, Vec<f64>)]) -> Ordered {
    let round4 = |v: f64| format!("{v:.4}").parse::<f64>().unwrap_or(v);
    Ordered::Object(
        table
            .iter()
            .map(|(k, v)| {
                (k.clone(), Ordered::Array(v.iter().map(|x| Ordered::Float(round4(*x))).collect()))
            })
            .collect(),
    )
}

pub fn run(job: &ChipsJob, data_dir: &std::path::Path) -> Result<()> {
    match job {
        ChipsJob::Values { gains, out, split, loso } => {
            let rows = GainRows::from_table(&read_parquet(&gains.join("gws.parquet"), None)?)?;
            let horizon = DOUBLE_TABLE_HORIZON;
            let mut doubles = None;
            let mut extra: Vec<(String, Ordered)> = Vec::new();
            if split == "double" {
                let outside: Vec<i64> = rows
                    .seasons
                    .iter()
                    .copied()
                    .filter(|s| !DEVELOP_SEASONS.contains(&(*s as i32)))
                    .collect();
                if !outside.is_empty() {
                    bail!("chip values are estimated on develop seasons only, got {outside:?}");
                }
                let store = DataStore::open(data_dir)?;
                doubles = Some(known_doubles_by_gw(&store, &rows.seasons, DOUBLE_LEAD)?);
                extra = vec![
                    ("lead".into(), Ordered::Int(DOUBLE_LEAD.into())),
                    ("horizon".into(), Ordered::Int(horizon.into())),
                ];
            }
            let table = estimate_terminal_values(&rows, split, doubles.as_ref(), horizon)?;
            let mut payload: Vec<(String, Ordered)> = vec![
                ("source".into(), Ordered::Str(py_path(gains))),
                ("seasons".into(), Ordered::Array(rows.seasons.iter().map(|s| Ordered::Str(season_label(*s as i32))).collect())),
                (
                    "note".into(),
                    Ordered::Str(
                        "mean best ex-ante one-GW gain of each chip over the GWs left in its window (index = GWs left - 1); \
                         unshrunk (the policy spec's chip_shrink scales it)"
                            .into(),
                    ),
                ),
            ];
            if split != "pooled" || *loso {
                payload.push(("format".into(), Ordered::Int(2)));
                payload.push(("split".into(), Ordered::Str(split.clone())));
                payload.extend(extra);
            }
            payload.push(("values".into(), table_json(&table)));
            if *loso {
                let tables = estimate_loso(&rows, split, doubles.as_ref(), horizon)?;
                payload.push((
                    "loso".into(),
                    Ordered::Object(
                        tables.iter().map(|(s, t)| (s.to_string(), table_json(t))).collect(),
                    ),
                ));
            }
            let index = CHIP_SPLITS.iter().position(|s| s == split).unwrap_or(0) as i64;
            let out = out.clone().unwrap_or_else(|| chip_table_file(index));
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&out, dumps_indent(&Ordered::Object(payload), 2)? + "\n")?;
            for (name, values) in &table {
                let shown: Vec<String> = values.iter().take(6).map(|v| format!("{v:.1}")).collect();
                println!(
                    "{name}: L=1..6 {} ... L={} {:.1}",
                    shown.join(", "),
                    values.len(),
                    values[values.len() - 1]
                );
            }
            println!("Written to {}", out.display());
            Ok(())
        }
        ChipsJob::Size { paired, cells } => {
            let t = read_parquet(paired, None)?;
            let (season, gw_index) = (t.i64("season")?, t.i64("gw_index")?);
            let mut outside: Vec<i64> = season
                .iter()
                .copied()
                .filter(|s| !DEVELOP_SEASONS.contains(&(*s as i32)))
                .collect();
            outside.sort_unstable();
            outside.dedup();
            if !outside.is_empty() {
                bail!("size is checked on develop seasons only, got {outside:?}");
            }
            let diff: Vec<Option<f64>> =
                t.opt_i64("diff")?.iter().map(|v| v.map(|x| x as f64)).collect();
            for (metric, values) in [("realized", diff), ("xg", t.opt_f64("diff_xg")?)] {
                let diffs: Vec<Diff> = (0..t.height())
                    .map(|i| Diff {
                        season: season[i],
                        gw_index: gw_index[i],
                        value: values[i].filter(|v| !v.is_nan()),
                        weight: 1.0,
                    })
                    .collect();
                let (n, rates) = window_test_null_size(
                    &diffs,
                    FULL_RUN_WINDOW,
                    *cells,
                    &[0.05, 0.10],
                    20000,
                    0,
                )?;
                let parts: Vec<String> = rates
                    .iter()
                    .map(|(a, r, p)| format!("alpha {a:.2}: rate {r:.3}, calibrated p {p:.4}"))
                    .collect();
                println!("{metric}: {n} develop cells; {}", parts.join(", "));
            }
            Ok(())
        }
    }
}
