//! Walk-forward evaluation of the price forecast (`research price-eval`; Stage B of
//! `docs/superpowers/specs/2026-11-06-price-planner-design.md`, plan
//! `docs/superpowers/plans/2026-11-07-price-forecast-stage-b.md` Task 4).
//!
//! For each evaluated season S: the six regressions are fitted on the training rows of seasons
//! first_fit..S−1 (the holdout left out) at S's first deadline, with v1 xP from the
//! backtester's caches; at every deadline of S the forecast and the baselines (N0 no change,
//! N1 momentum) are made for the candidates (the union of v1's top `DEFAULT_PRUNE_N` per
//! position) and set against the realized change `dp_h` from `player_gw` as visible after the
//! season (`RealizedChanges`, the training targets' definition; evaluation only, never a model
//! input). Rows whose truth is NaN (a blank GW, past the season's end) are written but left
//! out of the metrics: MSE per h and pooled over h = 1..6, and one-sided Diebold–Mariano tests
//! of the forecast against each baseline on per-deadline mean squared error pooled over h
//! (lag 6, clustered by deadline). Seasons run on threads; output in season order, the same
//! for any `jobs`.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Result, bail};
use fplopt_core::numpy::np_sum;
use fplopt_core::pyjson::Ordered;
use fplopt_core::seasons::{HOLDOUT_SEASONS, season_label};
use fplopt_models::price::{
    HORIZONS, PriceParams, RealizedChanges, baseline_changes, deadline_features, fit_price,
    predict_changes, refuse_holdout, training_rows,
};
use fplopt_store::{AsOfView, Column, DataStore, Micros, Table};

use crate::eval::{candidate_keys, diebold_mariano, split_parts};
use crate::simulator::{Caches, season_schedule};

/// The xP model whose features enter the forecast and whose top-N picks the candidates.
pub const XP_MODEL: &str = "v1";
/// Diebold–Mariano lag: the longest horizon, since h-step errors overlap up to h − 1 GWs.
pub const DM_LAG: usize = 6;
/// The baselines, in metric order.
pub const BASELINES: [&str; 2] = ["n0", "n1"];

/// One evaluated (deadline, player, h): realized and forecast changes in tenths of £m.
#[derive(Clone, Debug, PartialEq)]
pub struct PriceRow {
    pub season: i64,
    pub deadline: Micros,
    pub player_key: i64,
    pub h: i64,
    pub truth: f64,
    pub forecast: f64,
    pub n0: f64,
    pub n1: f64,
}

impl PriceRow {
    fn baseline(&self, name: &str) -> f64 {
        if name == "n0" { self.n0 } else { self.n1 }
    }
}

pub struct PriceEval {
    pub rows: Vec<PriceRow>,
    pub metrics: Ordered,
    /// (season label, seconds).
    pub timings: Vec<(String, f64)>,
}

/// Refuses the holdout and any season not after `first_fit` (S is fitted on
/// first_fit..S−1).
pub fn check_seasons(seasons: &[i64], first_fit: i64) -> Result<()> {
    refuse_holdout(seasons)?;
    if let Some(s) = seasons.iter().find(|s| **s <= first_fit) {
        bail!(
            "season {} is not after --first-fit {}: each season is fitted on the seasons from \
             --first-fit to the one before it",
            season_label(*s as i32),
            season_label(first_fit as i32)
        );
    }
    Ok(())
}

/// The seasons S's forecast is fitted on: first_fit..S−1 without the holdout.
pub fn fit_seasons(season: i64, first_fit: i64) -> Vec<i64> {
    (first_fit..season).filter(|s| !HOLDOUT_SEASONS.contains(&(*s as i32))).collect()
}

struct SeasonResult {
    rows: Vec<PriceRow>,
    n_deadlines: usize,
    seconds: f64,
}

fn season_unit(
    store: &DataStore,
    caches: &Caches,
    truth: &RealizedChanges,
    season: i64,
    first_fit: i64,
    params: &PriceParams,
) -> Result<SeasonResult> {
    let began = Instant::now();
    let schedule = season_schedule(store, season, 1)?;
    let Some(first) = schedule.first() else {
        bail!("no deadlines in {}", season_label(season as i32))
    };
    let xp_at = |view: &AsOfView| -> Result<Table> { Ok(caches.xp(XP_MODEL, view)?.table.clone()) };
    let train = training_rows(store, first.deadline, &fit_seasons(season, first_fit), &xp_at)?;
    let fit = fit_price(&train, params)?;
    let mut rows = Vec::new();
    for g in &schedule {
        let view = store.as_of(g.deadline);
        let xp = caches.xp(XP_MODEL, &view)?;
        let pool = caches.pool(&view)?;
        let positions: HashMap<i64, i64> =
            pool.iter().map(|p| (p.player_key, p.element_type)).collect();
        let candidates = candidate_keys(&[xp.as_ref()], &positions);
        let features = deadline_features(&view, &xp.table)?;
        let keep: Vec<usize> = features
            .i64("player_key")?
            .iter()
            .enumerate()
            .filter(|(_, k)| candidates.contains(k))
            .map(|(i, _)| i)
            .collect();
        let features = features.take(&keep);
        let keys = features.i64("player_key")?.to_vec();
        let forecast = predict_changes(&fit, &features)?;
        let (n0, n1) = baseline_changes(&view, &keys)?;
        let (fk, fh) = (forecast.i64("player_key")?, forecast.i64("h")?);
        let (fdp, n0dp, n1dp) = (forecast.f64("dp")?, n0.f64("dp")?, n1.f64("dp")?);
        for i in 0..forecast.height() {
            rows.push(PriceRow {
                season,
                deadline: g.deadline,
                player_key: fk[i],
                h: fh[i],
                truth: truth.dp(season, g.gw_index, fk[i], fh[i] as usize),
                forecast: fdp[i],
                n0: n0dp[i],
                n1: n1dp[i],
            });
        }
    }
    Ok(SeasonResult { rows, n_deadlines: schedule.len(), seconds: began.elapsed().as_secs_f64() })
}

/// Fits and evaluates the forecast on each of `seasons` (see the module docs).
pub fn evaluate_prices(
    store: &DataStore,
    caches: &Caches,
    seasons: &[i64],
    first_fit: i64,
    params: &PriceParams,
    jobs: usize,
) -> Result<PriceEval> {
    let seasons: Vec<i64> = seasons.iter().copied().collect::<BTreeSet<_>>().into_iter().collect();
    check_seasons(&seasons, first_fit)?;
    let truth = RealizedChanges::new(&store.as_of(Micros::MAX))?;
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<Result<SeasonResult>>>> =
        Mutex::new((0..seasons.len()).map(|_| None).collect());
    std::thread::scope(|scope| {
        for _ in 0..jobs.max(1).min(seasons.len()) {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= seasons.len() {
                        break;
                    }
                    let r = season_unit(store, caches, &truth, seasons[i], first_fit, params);
                    results.lock().unwrap()[i] = Some(r);
                }
            });
        }
    });
    let mut parts = Vec::new();
    for r in results.into_inner().unwrap() {
        parts.push(r.expect("every season ran")?);
    }
    let label = |s: i64| season_label(s as i32);
    let int = |x: i64| Ordered::Int(i128::from(x));
    let timings: Vec<(String, f64)> = seasons
        .iter()
        .zip(&parts)
        .map(|(s, p)| (label(*s), (p.seconds * 10.0).round() / 10.0))
        .collect();
    let deadlines: Vec<(String, Ordered)> =
        seasons.iter().zip(&parts).map(|(s, p)| (label(*s), int(p.n_deadlines as i64))).collect();
    let fitted: Vec<(String, Ordered)> = seasons
        .iter()
        .map(|s| {
            let fs = fit_seasons(*s, first_fit).into_iter().map(int).collect();
            (label(*s), Ordered::Array(fs))
        })
        .collect();
    let n_deadlines: usize = parts.iter().map(|p| p.n_deadlines).sum();
    let rows: Vec<PriceRow> = parts.into_iter().flat_map(|p| p.rows).collect();
    let mut metrics = vec![
        ("params".to_owned(), Ordered::Str(params.name.clone())),
        ("first_fit".to_owned(), int(first_fit)),
        ("fit_seasons".to_owned(), Ordered::Object(fitted)),
        ("deadlines".to_owned(), Ordered::Object(deadlines)),
        ("n_deadlines".to_owned(), int(n_deadlines as i64)),
        ("n_rows".to_owned(), int(rows.len() as i64)),
    ];
    if let Ordered::Object(pairs) = price_metrics(&rows) {
        metrics.extend(pairs);
    }
    Ok(PriceEval { rows, metrics: Ordered::Object(metrics), timings })
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { f64::NAN } else { np_sum(v) / v.len() as f64 }
}

fn f(x: f64) -> Ordered {
    if x.is_finite() { Ordered::Float(x) } else { Ordered::Null }
}

fn rec(pairs: Vec<(&str, Ordered)>) -> Ordered {
    Ordered::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}

/// Mean squared error of the forecast, N0 and N1 over `rows` (their truth must be finite).
pub fn mse_rows(rows: &[&PriceRow]) -> [f64; 3] {
    let mse = |pick: &dyn Fn(&PriceRow) -> f64| {
        mean(&rows.iter().map(|r| (pick(r) - r.truth).powi(2)).collect::<Vec<_>>())
    };
    [mse(&|r| r.forecast), mse(&|r| r.n0), mse(&|r| r.n1)]
}

/// The Diebold–Mariano input of forecast vs `baseline`: one row per deadline (in deadline
/// order) with the forecast's and the baseline's mean squared error over that deadline's
/// `rows`, pooled over h (their truth must be finite).
pub fn dm_inputs(rows: &[&PriceRow], baseline: &str) -> Vec<(Micros, f64, f64)> {
    let mut by_deadline: BTreeMap<Micros, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    for r in rows {
        let e = by_deadline.entry(r.deadline).or_default();
        e.0.push((r.forecast - r.truth).powi(2));
        e.1.push((r.baseline(baseline) - r.truth).powi(2));
    }
    by_deadline.into_iter().map(|(d, (a, b))| (d, mean(&a), mean(&b))).collect()
}

/// The metric tables over the rows with a finite truth: `n_scored`, `mse` (per split and h,
/// "1".."6" and pooled "1-6": n, n_deadlines, forecast, n0, n1) and `dm` (per split, forecast
/// vs each baseline on per-deadline pooled MSE: value_a/value_b are the mean per-deadline
/// MSEs, then the test's mean_diff, t_stat, p_a_better, p_b_better, n_clusters and lag).
pub fn price_metrics(rows: &[PriceRow]) -> Ordered {
    let scored: Vec<&PriceRow> = rows.iter().filter(|r| r.truth.is_finite()).collect();
    let mut mse = Vec::new();
    let mut dm = Vec::new();
    for (split, part) in split_parts(&scored, |r| r.season) {
        let mut horizons: Vec<(String, Vec<&PriceRow>)> = HORIZONS
            .iter()
            .map(|&h| (h.to_string(), part.iter().copied().filter(|r| r.h == h as i64).collect()))
            .collect();
        horizons.push(("1-6".to_owned(), part.clone()));
        for (label, rows) in horizons {
            if rows.is_empty() {
                continue;
            }
            let deadlines: BTreeSet<Micros> = rows.iter().map(|r| r.deadline).collect();
            let [fc, n0, n1] = mse_rows(&rows);
            mse.push(rec(vec![
                ("split", Ordered::Str(split.to_owned())),
                ("horizon", Ordered::Str(label)),
                ("n", Ordered::Int(rows.len() as i128)),
                ("n_deadlines", Ordered::Int(deadlines.len() as i128)),
                ("forecast", f(fc)),
                ("n0", f(n0)),
                ("n1", f(n1)),
            ]));
        }
        for baseline in BASELINES {
            let loss = dm_inputs(&part, baseline);
            let a: Vec<f64> = loss.iter().map(|l| l.1).collect();
            let b: Vec<f64> = loss.iter().map(|l| l.2).collect();
            let mut pairs = vec![
                ("a", Ordered::Str("forecast".to_owned())),
                ("b", Ordered::Str(baseline.to_owned())),
                ("split", Ordered::Str(split.to_owned())),
                ("horizon", Ordered::Str("1-6".to_owned())),
                ("value_a", f(mean(&a))),
                ("value_b", f(mean(&b))),
            ];
            pairs.extend(diebold_mariano(&loss, DM_LAG));
            dm.push(rec(pairs));
        }
    }
    rec(vec![
        ("n_scored", Ordered::Int(scored.len() as i128)),
        ("mse", Ordered::Array(mse)),
        ("dm", Ordered::Array(dm)),
    ])
}

/// `predictions.parquet`: season, deadline, player_key, h, truth, forecast, n0, n1.
pub fn rows_table(rows: &[PriceRow]) -> Table {
    let col = |pick: &dyn Fn(&PriceRow) -> f64| Column::F64(rows.iter().map(pick).collect());
    Table::new()
        .with("season", Column::I64(rows.iter().map(|r| r.season).collect()))
        .with("deadline", Column::Ts(rows.iter().map(|r| r.deadline).collect()))
        .with("player_key", Column::I64(rows.iter().map(|r| r.player_key).collect()))
        .with("h", Column::I64(rows.iter().map(|r| r.h).collect()))
        .with("truth", col(&|r| r.truth))
        .with("forecast", col(&|r| r.forecast))
        .with("n0", col(&|r| r.n0))
        .with("n1", col(&|r| r.n1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(season: i64, deadline: Micros, h: i64, truth: f64, fc: f64, n1: f64) -> PriceRow {
        PriceRow { season, deadline, player_key: 1, h, truth, forecast: fc, n0: 0.0, n1 }
    }

    /// Two deadlines of 2022, h = 1 and 2, plus a NaN truth (excluded) and a 2023 row.
    fn rows() -> Vec<PriceRow> {
        vec![
            row(2022, 10, 1, 1.0, 0.5, 2.0),  // sq: fc 0.25, n0 1, n1 1
            row(2022, 10, 2, 2.0, 1.0, 4.0),  // fc 1, n0 4, n1 4
            row(2022, 20, 1, 0.0, 0.5, 0.0),  // fc 0.25, n0 0, n1 0
            row(2022, 20, 2, -1.0, 0.0, 0.0), // fc 1, n0 1, n1 1
            row(2022, 20, 3, f64::NAN, 3.0, 9.0),
            row(2023, 30, 1, 1.0, 1.0, 1.0), // all 0 but n0 1
        ]
    }

    fn get<'a>(o: &'a Ordered, key: &str) -> &'a Ordered {
        match o {
            Ordered::Object(pairs) => &pairs.iter().find(|(k, _)| k == key).unwrap().1,
            _ => panic!("not an object"),
        }
    }

    fn items(o: &Ordered) -> &[Ordered] {
        match o {
            Ordered::Array(v) => v,
            _ => panic!("not an array"),
        }
    }

    fn num(o: &Ordered) -> f64 {
        match o {
            Ordered::Float(x) => *x,
            Ordered::Int(i) => *i as f64,
            _ => f64::NAN,
        }
    }

    fn text(o: &Ordered) -> &str {
        match o {
            Ordered::Str(s) => s,
            _ => "",
        }
    }

    #[test]
    fn mse_per_h_and_pooled() {
        let rows = rows();
        let scored: Vec<&PriceRow> = rows.iter().filter(|r| r.truth.is_finite()).collect();
        assert_eq!(mse_rows(&scored), [2.5 / 5.0, 7.0 / 5.0, 6.0 / 5.0]);
        let metrics = price_metrics(&rows);
        assert_eq!(num(get(&metrics, "n_scored")), 5.0);
        let find = |split: &str, h: &str| {
            items(get(&metrics, "mse"))
                .iter()
                .find(|r| text(get(r, "split")) == split && text(get(r, "horizon")) == h)
                .cloned()
        };
        let h1 = find("all", "1").unwrap();
        assert_eq!(num(get(&h1, "n")), 3.0);
        assert_eq!(num(get(&h1, "n_deadlines")), 3.0);
        assert_eq!(
            [num(get(&h1, "forecast")), num(get(&h1, "n0")), num(get(&h1, "n1"))],
            [0.5 / 3.0, 2.0 / 3.0, 1.0 / 3.0]
        );
        let pooled = find("develop", "1-6").unwrap();
        assert_eq!(num(get(&pooled, "n")), 4.0);
        assert_eq!(num(get(&pooled, "forecast")), 2.5 / 4.0);
        assert_eq!(num(get(&pooled, "n0")), 6.0 / 4.0);
        // h = 3's only row has no truth; 2023 is a validate season.
        assert!(find("all", "3").is_none());
        let v = find("validate", "1-6").unwrap();
        assert_eq!(num(get(&v, "n")), 1.0);
    }

    #[test]
    fn dm_inputs_are_per_deadline_pooled_over_h() {
        let rows = rows();
        let scored: Vec<&PriceRow> = rows.iter().filter(|r| r.truth.is_finite()).collect();
        assert_eq!(dm_inputs(&scored, "n0"), [(10, 0.625, 2.5), (20, 0.625, 0.5), (30, 0.0, 1.0)]);
        assert_eq!(dm_inputs(&scored, "n1")[0], (10, 0.625, 2.5));
        let metrics = price_metrics(&rows);
        let dm = items(get(&metrics, "dm"));
        let all_n0 = dm
            .iter()
            .find(|r| text(get(r, "split")) == "all" && text(get(r, "b")) == "n0")
            .unwrap();
        assert_eq!(num(get(all_n0, "n_clusters")), 3.0);
        assert_eq!(num(get(all_n0, "lag")), 6.0);
        assert_eq!(num(get(all_n0, "value_a")), 1.25 / 3.0);
        assert_eq!(num(get(all_n0, "value_b")), 4.0 / 3.0);
        // Three splits with rows (all, develop, validate) × two baselines.
        assert_eq!(dm.len(), 6);
    }

    #[test]
    fn seasons_not_after_first_fit_and_the_holdout_are_refused() {
        assert!(check_seasons(&[2022, 2023], 2021).is_ok());
        let err = check_seasons(&[2021, 2022], 2021).unwrap_err().to_string();
        assert!(err.contains("2021-22"), "{err}");
        let err = check_seasons(&[2024, 2025], 2021).unwrap_err().to_string();
        assert!(err.contains("2025"), "{err}");
        assert_eq!(fit_seasons(2022, 2021), [2021]);
        assert_eq!(fit_seasons(2026, 2021), [2021, 2022, 2023, 2024]);
    }

    #[test]
    fn rows_table_columns() {
        let t = rows_table(&rows());
        assert_eq!(
            t.names(),
            ["season", "deadline", "player_key", "h", "truth", "forecast", "n0", "n1"]
        );
        assert_eq!(t.height(), 6);
    }
}
