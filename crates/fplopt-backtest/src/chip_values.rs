//! Chip terminal values estimated from backtests (Python `fplopt.backtest.chip_values`; chip
//! pre-registration §3). From a `chipgains` run's GW rows: for a chip and L GWs left in its
//! window after the horizon, the mean over runs (season, policy, start) of the best ex-ante
//! gain among the window's last L GWs. Develop seasons only. Splits: `pooled` (keys by chip),
//! `window` (`name@k`, k the window's ordinal), `double` (also `name@k+d`, d = a double known
//! at the decision GW in those L GWs; a `+d` cell under `MIN_CELL` observations takes the
//! `name@k` value).

use std::collections::{BTreeMap, BTreeSet, HashMap};

use anyhow::{Result, bail};
use fplopt_core::seasons::DEVELOP_SEASONS;
use fplopt_store::DataStore;

use crate::evaluate::rules_for;

/// Chip name -> its gain column, in Python's order.
pub const GAIN_COLUMNS: [(&str, &str); 4] =
    [("3xc", "gain_tc"), ("bboost", "gain_bb"), ("freehit", "gain_fh"), ("wildcard", "gain_wc")];
pub const SPLITS: [&str; 3] = ["pooled", "window", "double"];
pub const MIN_CELL: usize = 30;

/// One run's GW gains: (season, policy, start_id) -> per chip, gw_index -> gain.
pub struct GainRows {
    pub runs: Vec<(i64, Vec<HashMap<i64, f64>>)>,
    pub seasons: Vec<i64>,
}

impl GainRows {
    /// From a chipgains run's gws.parquet table.
    pub fn from_table(t: &fplopt_store::Table) -> Result<GainRows> {
        for (_, c) in GAIN_COLUMNS {
            if !t.has(c) {
                bail!("no chip gains in these GW rows: run `backtest run --spec chipgains:...`");
            }
        }
        let (season, policy, start, gw_index) =
            (t.i64("season")?, t.str("policy")?, t.str("start_id")?, t.i64("gw_index")?);
        let gains: Vec<Vec<Option<f64>>> =
            GAIN_COLUMNS.iter().map(|(_, c)| t.opt_f64(c)).collect::<Result<_>>()?;
        if gains.iter().all(|g| g.iter().all(|v| v.is_none_or(f64::is_nan))) {
            bail!("no chip gains in these GW rows: run `backtest run --spec chipgains:...`");
        }
        let mut runs: BTreeMap<(i64, String, String), Vec<HashMap<i64, f64>>> = BTreeMap::new();
        for i in 0..t.height() {
            let run = runs
                .entry((season[i], policy[i].clone(), start[i].clone()))
                .or_insert_with(|| vec![HashMap::new(); 4]);
            for (c, g) in gains.iter().enumerate() {
                if let Some(v) = g[i].filter(|v| !v.is_nan()) {
                    run[c].insert(gw_index[i], v);
                }
            }
        }
        let seasons = runs.keys().map(|k| k.0).collect::<BTreeSet<_>>().into_iter().collect();
        Ok(GainRows { runs: runs.into_iter().map(|(k, v)| (k.0, v)).collect(), seasons })
    }

    fn without(&self, season: i64) -> GainRows {
        GainRows {
            runs: self.runs.iter().filter(|(s, _)| *s != season).cloned().collect(),
            seasons: self.seasons.iter().copied().filter(|s| *s != season).collect(),
        }
    }
}

/// Per table key (in Python's order), the values for L = 1 .. the longest window.
pub fn estimate_terminal_values(
    rows: &GainRows,
    split: &str,
    doubles: Option<&HashMap<(i64, i64), BTreeSet<i64>>>,
    horizon: i64,
) -> Result<Vec<(String, Vec<f64>)>> {
    if !SPLITS.contains(&split) {
        bail!("split must be one of ('pooled', 'window', 'double'), got '{split}'");
    }
    if split == "double" && doubles.is_none() {
        bail!("split='double' needs the known doubles (known_doubles_by_gw)");
    }
    let outside: Vec<i64> =
        rows.seasons.iter().copied().filter(|s| !DEVELOP_SEASONS.contains(&(*s as i32))).collect();
    if !outside.is_empty() {
        bail!("chip values are estimated on develop seasons only, got {outside:?}");
    }
    let rules = rules_for(i64::from(*DEVELOP_SEASONS.end()))?;
    let mut longest: Vec<(String, i64)> = Vec::new();
    let mut chip_windows: Vec<Vec<(i64, i64)>> = Vec::new();
    for (name, _) in GAIN_COLUMNS {
        let mut windows: Vec<(i64, i64)> =
            rules.chips.iter().filter(|w| w.name == name).map(|w| (w.start, w.stop)).collect();
        windows.sort_unstable();
        for (k, (start, stop)) in windows.iter().enumerate() {
            let length = stop - start + 1;
            let mut keys =
                vec![if split == "pooled" { name.to_owned() } else { format!("{name}@{}", k + 1) }];
            if split == "double" {
                keys.push(format!("{name}@{}+0", k + 1));
                keys.push(format!("{name}@{}+1", k + 1));
            }
            for key in keys {
                match longest.iter_mut().find(|(k, _)| *k == key) {
                    Some(e) => e.1 = e.1.max(length),
                    None => longest.push((key, length)),
                }
            }
        }
        chip_windows.push(windows);
    }
    let mut sums: HashMap<(String, i64), f64> = HashMap::new();
    let mut counts: HashMap<(String, i64), usize> = HashMap::new();
    for (c, (name, _)) in GAIN_COLUMNS.iter().enumerate() {
        for (season, gains) in &rows.runs {
            for (k, &(start, stop)) in chip_windows[c].iter().enumerate() {
                let base =
                    if split == "pooled" { name.to_string() } else { format!("{name}@{}", k + 1) };
                let mut best = f64::NEG_INFINITY;
                for left in 1..=(stop - start + 1) {
                    if let Some(g) = gains[c].get(&(stop - left + 1)) {
                        best = best.max(*g);
                    }
                    if best == f64::NEG_INFINITY {
                        continue;
                    }
                    let mut cells = vec![base.clone()];
                    if split == "double" {
                        let decision = (stop - left - horizon + 1).max(1);
                        let ahead = doubles.and_then(|d| d.get(&(*season, decision)));
                        let d =
                            ahead.is_some_and(|a| a.iter().any(|g| stop - left < *g && *g <= stop));
                        cells.push(format!("{base}+{}", i64::from(d)));
                    }
                    for key in cells {
                        *sums.entry((key.clone(), left)).or_default() += best;
                        *counts.entry((key, left)).or_default() += 1;
                    }
                }
            }
        }
    }
    let mut out = Vec::new();
    for (key, length) in longest {
        let parent = key.split_once('+').map(|(p, _)| p.to_owned());
        let mut values: Vec<f64> = Vec::new();
        for left in 1..=length {
            let mut src = key.clone();
            if let Some(p) = &parent
                && counts.get(&(key.clone(), left)).copied().unwrap_or(0) < MIN_CELL
            {
                src.clone_from(p);
            }
            let n = counts.get(&(src.clone(), left)).copied().unwrap_or(0);
            let v = if n > 0 {
                sums[&(src, left)] / n as f64
            } else {
                values.last().copied().unwrap_or(0.0)
            };
            values.push(v);
        }
        out.push((key, values.into_iter().map(|v| v.max(0.0)).collect()));
    }
    Ok(out)
}

/// Season s -> the table estimated on every other season's rows.
pub fn estimate_loso(
    rows: &GainRows,
    split: &str,
    doubles: Option<&HashMap<(i64, i64), BTreeSet<i64>>>,
    horizon: i64,
) -> Result<Vec<(i64, Vec<(String, Vec<f64>)>)>> {
    if rows.seasons.len() < 2 {
        bail!("leave-one-season-out needs at least two seasons, got {:?}", rows.seasons);
    }
    rows.seasons
        .iter()
        .map(|&s| Ok((s, estimate_terminal_values(&rows.without(s), split, doubles, horizon)?)))
        .collect()
}

/// (season, gw_index) -> the doubles known at that GW's deadline (lookahead `lead`).
pub fn known_doubles_by_gw(
    store: &DataStore,
    seasons: &[i64],
    lead: i64,
) -> Result<HashMap<(i64, i64), BTreeSet<i64>>> {
    let gameweeks = store
        .as_of(fplopt_optimize::start::FAR_FUTURE)
        .table("gameweek", Some(&["season", "gw_index"]))?;
    let (s, idx) = (gameweeks.opt_i64("season")?, gameweeks.opt_i64("gw_index")?);
    let mut out = HashMap::new();
    for &season in seasons {
        let indices: BTreeSet<i64> = (0..gameweeks.height())
            .filter(|&i| s[i] == Some(season))
            .filter_map(|i| idx[i])
            .collect();
        for gw_index in indices {
            let deadline = crate::simulator::season_schedule(store, season, gw_index)?[0].deadline;
            let view = store.as_of(deadline);
            out.insert(
                (season, gw_index),
                crate::policies::known_doubles(&view, season, gw_index, lead)?,
            );
        }
    }
    Ok(out)
}
