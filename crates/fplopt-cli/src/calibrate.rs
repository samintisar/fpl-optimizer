//! `models calibrate walk|fit`: the walk-forward rows of a v1-family model and the
//! expanding-window P(start) calibration fitted on them (Python `dev/calibrate_v1.py`'s
//! `walk` and `fit --write`, part `p_start` only; `score`, the variant report, `ban` and the
//! other parts are not ported: neither v1's nor v1r's table uses them).
//!
//! **walk**: per season, at every deadline in gw_index order, the uncalibrated per
//! player-fixture components (`fixture_components`, fitted at the refit cutoff) of the
//! fixtures that were played (inner join to `team_match`), in Python's row order, with the
//! realized start (0 when the player has no match row). One `<season>.parquet` per season
//! and `params.txt` (the model's `V1Params::fingerprint`).
//!
//! **fit**: for each season S from first_fit + 1 through LAST_FIT_SEASON + 1, the isotonic
//! P(start) map per horizon group fitted on the walk rows of the seasons in [first_fit, S),
//! concatenated in season order (the row order matters: the fit's sort keeps ties in it).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result, bail};
use clap::Subcommand;
use fplopt_build::table::{Column, Table, read_parquet, write_parquet};
use fplopt_core::pyjson::{Ordered, dumps};
use fplopt_core::settings::Settings;
use fplopt_models::assemble::{Fixture, V1Fit, V1Params, fit_v1, fixture_components};
use fplopt_models::calibration::{Isotonic, MAX_BINS, isotonic};
use fplopt_models::fitted::{REFIT_EVERY, refit_deadline};
use fplopt_store::DataStore;
use serde_json::Value;

use crate::optimize_plan::{exit_code, parse_seasons, usage};

/// The last season a walk covers and a table is fitted on (never the 2025/26 holdout).
pub const LAST_FIT_SEASON: i64 = 2024;
/// The gate's tolerance (`--compare`): breakpoints equal after the 6-digit rounding, but a
/// rounding tie may move one unit in the 6th digit at ≤ 1% of them.
const COMPARE_ATOL: f64 = 1e-6 + 1e-12;
const COMPARE_SHARE: f64 = 0.01;

#[derive(Subcommand)]
pub enum CalibrateJob {
    /// Per season, the uncalibrated per player-fixture P(start) at every deadline and the
    /// realized start; writes <out>/<season>.parquet and <out>/params.txt.
    Walk {
        /// v1 or v1r.
        #[arg(long)]
        model: String,
        /// Start years, e.g. 2016-2024 (the 2025/26 holdout is refused).
        #[arg(long)]
        seasons: String,
        #[arg(long)]
        out: PathBuf,
        /// Threads over seasons (default: CPUs - 1); results equal --jobs 1.
        #[arg(long, default_value_t = std::thread::available_parallelism().map_or(2, |n| n.get()).saturating_sub(1).max(1))]
        jobs: usize,
        /// The data directory (default: FPLOPT_DATA_DIR, else data).
        #[arg(long)]
        data_dir: Option<PathBuf>,
    },
    /// From a walk, the table: for each season S, the P(start) map(s) fitted on the walk
    /// rows of seasons [first_fit, S); written to <walk>/calibration-table.json, and with
    /// --write to crates/fplopt-models/calibration_table_<model>.json.
    Fit {
        /// v1 or v1r (the walk's params.txt must be this model's fingerprint).
        #[arg(long)]
        model: String,
        #[arg(long)]
        walk: PathBuf,
        /// Horizon groups: "1+" (v1's: horizons ≥ 1 pooled, h0 left as is), "0", "all"
        /// (every horizon pooled) or "0,1+".
        #[arg(long, default_value = "1+")]
        groups: String,
        #[arg(long, default_value_t = 2016)]
        first_fit: i64,
        #[arg(long)]
        write: bool,
        /// Another table JSON to diff every entry's breakpoints against (exit 1 beyond the
        /// gate's tolerance).
        #[arg(long)]
        compare: Option<PathBuf>,
    },
}

pub fn run(job: &CalibrateJob) -> ExitCode {
    exit_code(match job {
        CalibrateJob::Walk { model, seasons, out, jobs, data_dir } => {
            walk(model, seasons, out, *jobs, data_dir.as_deref())
        }
        CalibrateJob::Fit { model, walk, groups, first_fit, write, compare } => {
            fit(model, walk, groups, *first_fit, *write, compare.as_deref())
        }
    })
}

fn params_for(model: &str) -> Result<V1Params> {
    match V1Params::for_model(model) {
        Ok(p) => Ok(p),
        Err(e) => usage(format!("argument --model: {e}")),
    }
}

// --- walk -----------------------------------------------------------------------------------

fn walk(
    model: &str,
    seasons: &str,
    out: &Path,
    jobs: usize,
    data_dir: Option<&Path>,
) -> Result<()> {
    let params = params_for(model)?;
    let seasons = parse_seasons(seasons, "--seasons")?;
    if seasons.iter().any(|&s| s > LAST_FIT_SEASON) {
        return usage("walk covers develop and validate only (never the 2025/26 holdout)");
    }
    if jobs < 1 {
        return usage("--jobs must be >= 1");
    }
    let data_dir = data_dir.map_or_else(|| Settings::from_env().data_dir, Path::to_path_buf);
    std::fs::create_dir_all(out)?;
    std::fs::write(out.join("params.txt"), params.fingerprint())?;
    let store = DataStore::open(&data_dir)?;
    // Longest seasons first, as Python's pool (the output does not depend on it).
    let order: Vec<i64> = seasons.iter().rev().copied().collect();
    let next = AtomicUsize::new(0);
    let failures: Mutex<Vec<anyhow::Error>> = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..jobs.min(order.len()) {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= order.len() {
                        break;
                    }
                    let season = order[i];
                    let started = std::time::Instant::now();
                    let written = walk_season(&store, &data_dir, &params, season).and_then(|t| {
                        write_parquet(&t, &out.join(format!("{season}.parquet")))?;
                        Ok(t.height())
                    });
                    match written {
                        Ok(rows) => log::info!(
                            "season {season} done: {rows} rows, {:.0} s",
                            started.elapsed().as_secs_f64()
                        ),
                        Err(e) => {
                            log::error!("season {season} failed: {e:#}");
                            failures.lock().unwrap().push(e.context(format!("season {season}")));
                        }
                    }
                }
            });
        }
    });
    match failures.into_inner().unwrap().into_iter().next() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// One walk row's columns besides the components.
#[derive(Default)]
struct WalkRows {
    deadline_gw_index: Vec<i64>,
    fixtures: Vec<Fixture>,
    r_start: Vec<f64>,
}

impl WalkRows {
    fn into_table(self) -> Table {
        let f = &self.fixtures;
        let ints = |g: fn(&Fixture) -> i64| Column::I64(f.iter().map(g).collect());
        let floats = |g: fn(&Fixture) -> f64| Column::F64(f.iter().map(g).collect());
        Table::new()
            .with("season", ints(|x| x.key.season))
            .with("deadline_gw_index", Column::I64(self.deadline_gw_index))
            .with("player_key", ints(|x| x.key.player_key))
            .with("fixture_key", ints(|x| x.key.fixture_key))
            .with("team_key", ints(|x| x.key.team_key))
            .with("gw", ints(|x| x.key.gw))
            .with("gw_index", ints(|x| x.key.gw_index))
            .with("horizon", ints(|x| x.key.horizon))
            .with("element_type", ints(|x| x.element_type))
            .with("p_start", floats(|x| x.p_start))
            .with("p_60", floats(|x| x.p_60))
            .with("p_play", floats(|x| x.p_play))
            .with("e_minutes", floats(|x| x.e_minutes))
            .with("p_cs", floats(|x| x.p_cs))
            .with("r_start", Column::F64(self.r_start))
    }
}

/// `season`'s walk rows (Python `walk_season`, the p_start columns). The fit is redone when
/// the refit cutoff moves (cutoffs only grow along the season, so one fit is kept).
fn walk_season(store: &DataStore, data: &Path, params: &V1Params, season: i64) -> Result<Table> {
    let params = V1Params { calibrate: false, ..params.clone() };
    let read = |name: &str, columns: &[&str]| {
        read_parquet(&data.join(format!("{name}.parquet")), Some(columns))
    };
    let realized = realized_starts(
        &read(
            "player_match",
            &["player_key", "season", "fixture_key", "team_key", "minutes", "starts"],
        )?,
        &read("player_season", &["player_key", "season"])?,
        season,
    )?;
    let played = played_sides(
        &read("team_match", &["season", "fixture_key", "team_key", "goals_against"])?,
        season,
    )?;
    let mut out = WalkRows::default();
    let mut fit: Option<(i64, V1Fit)> = None;
    for (gw_index, deadline) in
        season_deadlines(&read("gameweek", &["season", "gw_index", "deadline_time"])?, season)?
    {
        let view = store.as_of(deadline);
        let cutoff = refit_deadline(&view, REFIT_EVERY)?;
        if fit.as_ref().is_none_or(|(c, _)| *c != cutoff) {
            fit = Some((cutoff, fit_v1(&view.earlier(cutoff)?, &params)?));
        }
        let mut rows = fixture_components(&view, &fit.as_ref().unwrap().1)?;
        // Python's order (`sort_values([player_key, horizon, fixture_key], kind="mergesort")`),
        // whatever order the components come in; the inner merge then keeps it.
        rows.sort_by_key(|f| (f.key.player_key, f.key.horizon, f.key.fixture_key));
        for f in rows {
            if !played.contains(&(f.key.fixture_key, f.key.team_key)) {
                continue;
            }
            out.r_start
                .push(realized.get(&(f.key.player_key, f.key.fixture_key)).copied().unwrap_or(0.0));
            out.deadline_gw_index.push(gw_index);
            out.fixtures.push(f);
        }
    }
    Ok(out.into_table())
}

/// The season's (gw_index, deadline) in gw_index order.
fn season_deadlines(gameweeks: &Table, season: i64) -> Result<Vec<(i64, i64)>> {
    let (s, g, d) = (
        gameweeks.opt_i64("season")?,
        gameweeks.opt_i64("gw_index")?,
        gameweeks.opt_i64("deadline_time")?,
    );
    let mut out: Vec<(i64, i64)> = (0..gameweeks.height())
        .filter(|&i| s[i] == Some(season))
        .map(|i| -> Result<(i64, i64)> {
            Ok((g[i].context("gameweek: null gw_index")?, d[i].context("gameweek: null deadline")?))
        })
        .collect::<Result<_>>()?;
    out.sort_by_key(|&(gw_index, _)| gw_index);
    Ok(out)
}

/// The (fixture_key, team_key) of the season's team_match rows with goals_against (Python's
/// `against`, after `dropna`).
fn played_sides(teams: &Table, season: i64) -> Result<HashSet<(i64, i64)>> {
    let (s, f, t) =
        (teams.opt_i64("season")?, teams.opt_i64("fixture_key")?, teams.opt_i64("team_key")?);
    let against = teams.opt_f64("goals_against")?;
    Ok((0..teams.height())
        .filter(|&i| s[i] == Some(season) && against[i].is_some_and(|a| !a.is_nan()))
        .filter_map(|i| Some((f[i]?, t[i]?)))
        .collect())
}

/// Per (player_key, fixture_key), the season's realized start (Python `realized`'s
/// `r_start`): the season's player_match rows of players with a player_season row that
/// season (inner join), `starts` where known, else inferred from minutes among those rows
/// (`infer_starts`, minutes null as 0); a repeated key keeps its last row.
fn realized_starts(
    matches: &Table,
    positions: &Table,
    season: i64,
) -> Result<HashMap<(i64, i64), f64>> {
    let (pp, ps) = (positions.opt_i64("player_key")?, positions.opt_i64("season")?);
    let registered: HashSet<i64> =
        (0..positions.height()).filter(|&i| ps[i] == Some(season)).filter_map(|i| pp[i]).collect();
    let (player, seasons) = (matches.opt_i64("player_key")?, matches.opt_i64("season")?);
    let (fixture, team) = (matches.opt_i64("fixture_key")?, matches.opt_i64("team_key")?);
    let (minutes, starts) = (matches.opt_i64("minutes")?, matches.opt_f64("starts")?);
    let rows: Vec<usize> = (0..matches.height())
        .filter(|&i| {
            seasons[i] == Some(season) && player[i].is_some_and(|p| registered.contains(&p))
        })
        .collect();
    let take = |v: &[Option<i64>], what: &str| -> Result<Vec<i64>> {
        rows.iter().map(|&i| v[i].with_context(|| format!("player_match: null {what}"))).collect()
    };
    let (rp, rf, rt) =
        (take(&player, "player_key")?, take(&fixture, "fixture_key")?, take(&team, "team_key")?);
    let rm: Vec<i64> = rows.iter().map(|&i| minutes[i].unwrap_or(0)).collect();
    let inferred = fplopt_features::history::infer_starts(&rf, &rt, &rm, &rp);
    let mut out = HashMap::with_capacity(rows.len());
    for (k, &i) in rows.iter().enumerate() {
        let real = starts[i].filter(|s| !s.is_nan());
        out.insert((rp[k], rf[k]), real.unwrap_or(f64::from(u8::from(inferred[k]))));
    }
    Ok(out)
}

// --- fit ------------------------------------------------------------------------------------

/// The walk columns the fit reads, rows in file order.
#[derive(Default)]
struct FitRows {
    horizon: Vec<i64>,
    p_start: Vec<f64>,
    r_start: Vec<f64>,
}

/// One table entry: the season it applies from, the seasons fitted on, the maps.
struct Entry {
    season: i64,
    fitted_on: Vec<i64>,
    p_start: Vec<(i64, Isotonic)>,
}

/// Python `horizon_groups`: (first horizon, stop) per map.
fn parse_groups(spec: &str) -> Result<Vec<(i64, Option<i64>)>> {
    Ok(match spec {
        "1+" => vec![(1, None)],
        "0" => vec![(0, Some(1))],
        "all" => vec![(0, None)],
        "0,1+" => vec![(0, Some(1)), (1, None)],
        other => return usage(format!("argument --groups: '{other}' (1+, 0, all or 0,1+)")),
    })
}

/// The `<season>.parquet` files of a walk, sorted.
fn walk_seasons(walk: &Path) -> Result<Vec<i64>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(walk).with_context(|| format!("read {}", walk.display()))? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "parquet")
            && let Some(season) = path.file_stem().and_then(|s| s.to_str()?.parse::<i64>().ok())
        {
            out.push(season);
        }
    }
    out.sort_unstable();
    Ok(out)
}

fn load_walk(walk: &Path, season: i64) -> Result<FitRows> {
    let t = read_parquet(
        &walk.join(format!("{season}.parquet")),
        Some(&["horizon", "p_start", "r_start"]),
    )?;
    let nan = |v: Vec<Option<f64>>| v.into_iter().map(|x| x.unwrap_or(f64::NAN)).collect();
    Ok(FitRows {
        horizon: t.opt_i64("horizon")?.into_iter().map(|h| h.unwrap_or(-1)).collect(),
        p_start: nan(t.opt_f64("p_start")?),
        r_start: nan(t.opt_f64("r_start")?),
    })
}

/// Python `fit --write`'s entries (the p_start part): S = first_fit + 1 ..= LAST_FIT_SEASON
/// + 1, each fitted on the walk seasons in [first_fit, S).
fn fit_table(walk: &Path, groups: &[(i64, Option<i64>)], first_fit: i64) -> Result<Vec<Entry>> {
    let available = walk_seasons(walk)?;
    let mut frames: BTreeMap<i64, FitRows> = BTreeMap::new();
    for &s in &available {
        frames.insert(s, load_walk(walk, s)?);
    }
    let mut table = Vec::new();
    for season in first_fit + 1..=LAST_FIT_SEASON + 1 {
        let used: Vec<i64> =
            available.iter().copied().filter(|&s| first_fit <= s && s < season).collect();
        if used.is_empty() {
            bail!("no walk season in [{first_fit}, {season}) under {}", walk.display());
        }
        let mut maps = Vec::new();
        for &(first, stop) in groups {
            let (mut p, mut y) = (Vec::new(), Vec::new());
            for s in &used {
                let rows = &frames[s];
                for i in 0..rows.horizon.len() {
                    if rows.horizon[i] >= first && rows.horizon[i] < stop.unwrap_or(99) {
                        p.push(rows.p_start[i]);
                        y.push(rows.r_start[i]);
                    }
                }
            }
            maps.push((first, isotonic(&p, &y, None, MAX_BINS)));
        }
        table.push(Entry { season, fitted_on: used, p_start: maps });
    }
    Ok(table)
}

/// The table in `calibration_table.json`'s layout: {"params", "table": [[S, {"fitted_on",
/// "p_start": [[first, x, y], ...]}], ...]}.
fn table_json(params: &str, entries: &[Entry]) -> Ordered {
    let int = |i: i64| Ordered::Int(i128::from(i));
    let floats = |v: &[f64]| Ordered::Array(v.iter().map(|&x| Ordered::Float(x)).collect());
    let rows = entries
        .iter()
        .map(|e| {
            let maps = e
                .p_start
                .iter()
                .map(|(first, m)| Ordered::Array(vec![int(*first), floats(&m.x), floats(&m.y)]))
                .collect();
            Ordered::Array(vec![
                int(e.season),
                Ordered::Object(vec![
                    (
                        "fitted_on".into(),
                        Ordered::Array(e.fitted_on.iter().map(|&s| int(s)).collect()),
                    ),
                    ("p_start".into(), Ordered::Array(maps)),
                ]),
            ])
        })
        .collect();
    Ordered::Object(vec![
        ("params".into(), Ordered::Str(params.into())),
        ("table".into(), Ordered::Array(rows)),
    ])
}

fn fit(
    model: &str,
    walk: &Path,
    groups: &str,
    first_fit: i64,
    write: bool,
    compare: Option<&Path>,
) -> Result<()> {
    if write && model == "v1" {
        bail!("v1's table is calibration_table.json, exported from Python's; --write is for v1r");
    }
    let params = params_for(model)?;
    let groups = parse_groups(groups)?;
    let fingerprint = std::fs::read_to_string(walk.join("params.txt"))
        .with_context(|| format!("{}: no params.txt (not a walk?)", walk.display()))?;
    if fingerprint != params.fingerprint() {
        bail!("the walk in {} ran with other params than {model}'s; walk again", walk.display());
    }
    let entries = fit_table(walk, &groups, first_fit)?;
    for e in &entries {
        let sizes: Vec<String> =
            e.p_start.iter().map(|(first, m)| format!("h{first}+: {}", m.x.len())).collect();
        let seasons: Vec<String> = e.fitted_on.iter().map(i64::to_string).collect();
        println!("{}: fitted on {}; breakpoints {}", e.season, seasons.join(","), sizes.join(", "));
    }
    let text = dumps(&table_json(&fingerprint, &entries))?;
    let local = walk.join("calibration-table.json");
    std::fs::write(&local, &text)?;
    println!("wrote {}", local.display());
    if write {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("fplopt-models")
            .join(format!("calibration_table_{model}.json"));
        std::fs::write(&path, &text)?;
        println!("wrote {}", path.display());
    }
    if let Some(other) = compare {
        let theirs: Value = serde_json::from_str(&std::fs::read_to_string(other)?)
            .with_context(|| format!("parse {}", other.display()))?;
        let ours: Value = serde_json::from_str(&text)?;
        let c = compare_tables(&ours, &theirs)?;
        println!(
            "compare with {}: {} breakpoints, {} differ, max |diff| {:.3e}, max length mismatch {}, \
             seasons missing {:?}",
            other.display(),
            c.breakpoints,
            c.differ,
            c.max_abs,
            c.max_length_mismatch,
            c.missing
        );
        if !c.passes() {
            bail!("the tables differ beyond the gate's tolerance");
        }
        println!("compare: pass");
    }
    Ok(())
}

/// Breakpoint differences between two tables' p_start maps.
#[derive(Debug, Default)]
struct Comparison {
    /// Breakpoints (x, y pairs) of ours.
    breakpoints: usize,
    /// Breakpoints at a common position whose x or y differ.
    differ: usize,
    max_abs: f64,
    max_length_mismatch: usize,
    /// Seasons (or maps) present in one table only.
    missing: Vec<i64>,
}

impl Comparison {
    fn passes(&self) -> bool {
        self.missing.is_empty()
            && self.max_length_mismatch == 0
            && self.max_abs <= COMPARE_ATOL
            && self.differ as f64 <= COMPARE_SHARE * self.breakpoints as f64
    }
}

type Maps = Vec<(i64, Vec<f64>, Vec<f64>)>;

/// Per season, the p_start maps of a table JSON.
fn p_start_maps(table: &Value) -> Result<BTreeMap<i64, Maps>> {
    let floats = |v: &Value| -> Result<Vec<f64>> {
        v.as_array()
            .context("not a list")?
            .iter()
            .map(|x| x.as_f64().context("not a number"))
            .collect()
    };
    let mut out = BTreeMap::new();
    for entry in table["table"].as_array().context("no 'table'")? {
        let season = entry[0].as_i64().context("a season")?;
        let mut maps = Vec::new();
        if let Some(parts) = entry[1]["p_start"].as_array() {
            for m in parts {
                maps.push((
                    m[0].as_i64().context("first horizon")?,
                    floats(&m[1])?,
                    floats(&m[2])?,
                ));
            }
        }
        out.insert(season, maps);
    }
    Ok(out)
}

fn compare_tables(ours: &Value, theirs: &Value) -> Result<Comparison> {
    let (a, b) = (p_start_maps(ours)?, p_start_maps(theirs)?);
    let mut c = Comparison::default();
    let seasons: std::collections::BTreeSet<i64> = a.keys().chain(b.keys()).copied().collect();
    for season in seasons {
        let (Some(ma), Some(mb)) = (a.get(&season), b.get(&season)) else {
            c.missing.push(season);
            continue;
        };
        if ma.len() != mb.len() || ma.iter().zip(mb).any(|(x, y)| x.0 != y.0) {
            c.missing.push(season); // other horizon groups
            continue;
        }
        for ((_, ax, ay), (_, bx, by)) in ma.iter().zip(mb) {
            c.breakpoints += ax.len();
            c.max_length_mismatch = c.max_length_mismatch.max(ax.len().abs_diff(bx.len()));
            for k in 0..ax.len().min(bx.len()) {
                let d = (ax[k] - bx[k]).abs().max((ay[k] - by[k]).abs());
                if d != 0.0 {
                    c.differ += 1;
                }
                c.max_abs = c.max_abs.max(d);
            }
        }
    }
    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(columns: Vec<(&str, Column)>) -> Table {
        columns.into_iter().fold(Table::new(), |t, (n, c)| t.with(n, c))
    }

    /// Python `realized`'s r_start: real starts where known, else inferred among the
    /// registered players' rows only; other seasons and unregistered players left out.
    #[test]
    fn realized_starts_is_pythons() {
        // Fixture 10, team 1: twelve players with minutes 90..=79 (player 12 is the 12th),
        // player 1 with a real start of 0, player 13 not registered (90 minutes: would push
        // player 11 out of the inferred XI if it counted).
        let mut player: Vec<i64> = (1..=13).collect();
        let mut minutes: Vec<Option<i64>> = (0..12).map(|k| Some(90 - k)).collect();
        minutes.push(Some(90));
        let mut starts: Vec<Option<f64>> = vec![None; 13];
        starts[0] = Some(0.0);
        let mut season = vec![2020; 13];
        // An other season's row for player 2 and a null-minutes row (fixture 11).
        player.extend([2, 3]);
        minutes.extend([Some(90), None]);
        starts.extend([Some(1.0), None]);
        season.extend([2019, 2020]);
        let n = player.len();
        let mut fixture = vec![10; 13];
        fixture.extend([9, 11]);
        let matches = table(vec![
            ("player_key", Column::I64(player)),
            ("season", Column::I64(season)),
            ("fixture_key", Column::I64(fixture)),
            ("team_key", Column::I64(vec![1; n])),
            ("minutes", Column::OptI64(minutes)),
            ("starts", Column::OptF64(starts)),
        ]);
        let positions = table(vec![
            ("player_key", Column::I64((1..=12).collect())),
            ("season", Column::I64(vec![2020; 12])),
        ]);
        let r = realized_starts(&matches, &positions, 2020).unwrap();
        assert_eq!(r[&(1, 10)], 0.0); // real start wins
        for p in 2..=11 {
            assert_eq!(r[&(p, 10)], 1.0, "player {p}");
        }
        assert_eq!(r[&(12, 10)], 0.0); // 12th by minutes (player 1 still ranks)
        assert!(!r.contains_key(&(13, 10)));
        assert!(!r.contains_key(&(2, 9)));
        assert_eq!(r[&(3, 11)], 0.0); // null minutes: 0, never a starter
    }

    #[test]
    fn played_sides_drops_null_goals_and_other_seasons() {
        let teams = table(vec![
            ("season", Column::I64(vec![2020, 2020, 2019])),
            ("fixture_key", Column::I64(vec![1, 2, 3])),
            ("team_key", Column::I64(vec![5, 5, 5])),
            ("goals_against", Column::OptF64(vec![Some(0.0), None, Some(1.0)])),
        ]);
        assert_eq!(played_sides(&teams, 2020).unwrap(), HashSet::from([(1, 5)]));
    }

    #[test]
    fn deadlines_in_gw_index_order() {
        let gameweeks = table(vec![
            ("season", Column::I64(vec![2020, 2020, 2021])),
            ("gw_index", Column::I64(vec![2, 1, 1])),
            ("deadline_time", Column::Ts(vec![20, 10, 30])),
        ]);
        assert_eq!(season_deadlines(&gameweeks, 2020).unwrap(), vec![(1, 10), (2, 20)]);
    }

    #[test]
    fn groups_are_pythons_horizon_groups() {
        assert_eq!(parse_groups("1+").unwrap(), vec![(1, None)]);
        assert_eq!(parse_groups("0,1+").unwrap(), vec![(0, Some(1)), (1, None)]);
        assert!(parse_groups("2+").is_err());
    }

    #[test]
    fn compare_counts_differences() {
        let a = serde_json::json!({"table": [[2017, {"p_start": [[1, [0.1, 0.2], [0.0, 0.5]]]}]]});
        let b =
            serde_json::json!({"table": [[2017, {"p_start": [[1, [0.1, 0.200001], [0.0, 0.5]]]}]]});
        let c = compare_tables(&a, &b).unwrap();
        assert_eq!((c.breakpoints, c.differ, c.max_length_mismatch), (2, 1, 0));
        assert!((c.max_abs - 1e-6).abs() < 1e-12);
        assert!(!c.passes()); // 1 of 2 differs: over 1%
        assert!(compare_tables(&a, &a).unwrap().passes());
        let short =
            serde_json::json!({"table": [[2017, {"p_start": [[1, [0.1], [0.0]]]}], [2018, {}]]});
        let c = compare_tables(&a, &short).unwrap();
        assert_eq!((c.max_length_mismatch, c.missing.clone()), (1, vec![2018]));
    }
}
