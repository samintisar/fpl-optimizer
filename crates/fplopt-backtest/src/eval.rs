//! Walk-forward evaluation of xP models against realized outcomes (Python `fplopt.evaluate`:
//! `run`, `metrics`, `outcomes`; Phase 5 *criterion 1*, PLAN §5 *Metrics*).
//!
//! At every deadline of every season: each model's xP frame (the backtester's caches, so v1
//! memoizes its fit by cutoff), joined per player-GW to the realized outcome of its target
//! GW (null if not played; a pool player without a row in a played GW scored 0), the
//! candidate flags (the union over the models of each one's top-N per position by
//! horizon-summed xP, and the model's own), and the template and random squads' XI and
//! captain regret under each model's lineup. `compute_metrics` turns these into the metric
//! tables (xP, bands, by season, regret, pairwise Diebold-Mariano, components). Seasons run
//! on threads; output in season order, the same for any `jobs`.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Result, bail};
use fplopt_build::scoring::Rules;
use fplopt_core::numpy::np_sum;
use fplopt_core::pyjson::Ordered;
use fplopt_core::seasons::{DEVELOP_SEASONS, HOLDOUT_SEASONS, VALIDATE_SEASONS, season_label};
use fplopt_optimize::params::DEFAULT_PRUNE_N;
use fplopt_optimize::start::{MAX_BANK, StartStateError, random_state, template_state};
use fplopt_store::{DataStore, Micros};

use crate::evaluate::rules_for;
use crate::gw_score::score_gameweek;
use crate::outcomes::OUTCOME_DELAY;
use crate::outcomes::gw_matches;
use crate::policies::{best_lineup, target_xp};
use crate::simulator::{Caches, hindsight_points, season_schedule};
use crate::state::SquadState;
use crate::stats::t_sf;

/// Models whose inputs start later (FPL's ep_next is in the snapshots from 2021/22).
pub fn first_season(model: &str) -> i64 {
    if model == "ep_next" || model == "ep_next_fade" { 2021 } else { 0 }
}

pub const N_RANDOM: usize = 3;
pub const XP_BANDS: [f64; 10] =
    [f64::NEG_INFINITY, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, f64::INFINITY];
/// Realized per player-GW columns joined to the predictions.
pub const REALIZED: [&str; 9] = [
    "points",
    "minutes",
    "starts",
    "n_fixtures",
    "goals_scored",
    "assists",
    "clean_sheets",
    "bonus",
    "defcon",
];
const NULLABLE: [usize; 2] = [2, 8]; // starts, defcon
const XP_BASE: [&str; 5] = ["player_key", "gw", "gw_index", "horizon", "xp"];
pub const HORIZON_LABELS: [&str; 7] = ["0", "1", "2", "3", "4", "5", "1-5"];
const LOG_EPS: f64 = 1e-15;
const RATE_EPS: f64 = 1e-12;

/// One prediction row (Python `PREDICTION_COLUMNS` + the model's extra columns).
#[derive(Clone, Debug)]
pub struct Prediction {
    pub model: usize,
    pub season: i64,
    pub deadline: Micros,
    pub gw: i64,
    pub gw_index: i64,
    pub player_key: i64,
    pub element_type: i64,
    pub horizon: i64,
    pub target_gw: i64,
    pub target_gw_index: i64,
    pub xp: f64,
    pub candidate: bool,
    pub own_candidate: bool,
    /// `REALIZED`, NaN for null.
    pub realized: [f64; 9],
    /// The model's extra columns (`EvalResult::extras[model]`), NaN for null.
    pub extras: Vec<f64>,
}

/// One regret row (Python `REGRET_COLUMNS`).
#[derive(Clone, Debug)]
pub struct Regret {
    pub model: usize,
    pub season: i64,
    pub deadline: Micros,
    pub gw: i64,
    pub gw_index: i64,
    pub squad: String,
    pub points: i64,
    pub best_points: i64,
    pub xi_regret: i64,
    pub captain_points: i64,
    pub best_captain_points: i64,
    pub captain_regret: i64,
}

/// Per player_key, the GW's realized totals (`REALIZED`).
type GwTotals = HashMap<i64, [f64; 9]>;

/// The GW's realized per player totals, None if not played: points re-scored under the
/// season's rules, stats summed over fixtures, `starts` null if every fixture's is, `defcon`
/// 1/0 from the summed defensive contributions and the position's threshold (null without
/// either).
fn gw_totals(
    store: &DataStore,
    rules: &Rules,
    season: i64,
    gw: i64,
    lockdown: Micros,
) -> Result<Option<GwTotals>> {
    let view = store.as_of(lockdown + OUTCOME_DELAY);
    let rows = gw_matches(&view, season, gw)?;
    if rows.is_empty() {
        return Ok(None);
    }
    // (points, minutes, starts, n, goals, assists, cs, bonus, dc) with nullable sums
    let mut acc: BTreeMap<i64, (i64, [f64; 9], Option<i64>, Option<i64>)> = BTreeMap::new();
    let mut order: Vec<&crate::outcomes::MatchRow> = rows.iter().collect();
    order.sort_by_key(|r| (r.player_key, r.fixture_key));
    for r in order {
        let points = fplopt_build::scoring::score(&r.line, rules)?;
        let e = acc.entry(r.player_key).or_insert((r.element_type, [0.0; 9], None, None));
        e.1[0] += points as f64;
        e.1[1] += r.line.minutes as f64;
        e.1[3] += 1.0;
        e.1[4] += r.line.goals_scored as f64;
        e.1[5] += r.line.assists as f64;
        e.1[6] += r.line.clean_sheets as f64;
        e.1[7] += r.line.bonus as f64;
        if let Some(s) = r.starts {
            e.2 = Some(e.2.unwrap_or(0) + s);
        }
        if let Some(d) = r.line.defensive_contribution {
            e.3 = Some(e.3.unwrap_or(0) + d);
        }
    }
    Ok(Some(
        acc.into_iter()
            .map(|(k, (et, mut v, starts, dc))| {
                v[2] = starts.map_or(f64::NAN, |s| s as f64);
                let threshold = rules.defcon_threshold.get(&et).copied().flatten();
                v[8] = match (dc, threshold) {
                    (Some(d), Some(t)) => f64::from(u8::from(d >= t)),
                    _ => f64::NAN,
                };
                (k, v)
            })
            .collect(),
    ))
}

/// Sorted player_keys in the union over the frames of each one's top `DEFAULT_PRUNE_N` pool
/// players per position by horizon-summed xP (ties: smaller key first).
fn candidate_keys(
    frames: &[&crate::simulator::Xp],
    positions: &HashMap<i64, i64>,
) -> BTreeSet<i64> {
    let prune_n: BTreeMap<i64, i64> = DEFAULT_PRUNE_N.into_iter().collect();
    let mut chosen = BTreeSet::new();
    for xp in frames {
        let f = &xp.frame;
        let mut sums: BTreeMap<i64, (f64, f64)> = BTreeMap::new();
        for i in 0..f.len() {
            let e = sums.entry(f.player_key[i]).or_insert((0.0, 0.0));
            if let Some(v) = f.xp[i].filter(|v| !v.is_nan()) {
                let y = v - e.1;
                let t = e.0 + y;
                e.1 = t - e.0 - y;
                e.0 = t;
            }
        }
        let mut by_et: BTreeMap<i64, Vec<(f64, i64)>> = BTreeMap::new();
        for (k, (s, _)) in sums {
            if let Some(et) = positions.get(&k) {
                by_et.entry(*et).or_default().push((s, k));
            }
        }
        for (et, mut players) in by_et {
            players.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            let n = prune_n.get(&et).map_or(players.len(), |n| *n as usize);
            chosen.extend(players.iter().take(n).map(|p| p.1));
        }
    }
    chosen
}

/// The template squad and random squads seed..seed + n_random − 1 at the deadline (refused
/// ones skipped).
fn deadline_squads(
    view: &fplopt_store::AsOfView,
    rules: &Rules,
    n_random: usize,
    seed: u64,
) -> Result<Vec<(String, SquadState)>> {
    let mut out = Vec::new();
    let mut push = |id: String, state: Result<SquadState>| -> Result<()> {
        match state {
            Ok(s) => out.push((id, s)),
            Err(e) if e.downcast_ref::<StartStateError>().is_some() => {}
            Err(e) => return Err(e),
        }
        Ok(())
    };
    push("template".to_owned(), template_state(view, rules))?;
    for s in seed..seed + n_random as u64 {
        push(format!("random{s}"), random_state(view, rules, s, Some(MAX_BANK)))?;
    }
    Ok(out)
}

struct SeasonResult {
    predictions: Vec<Prediction>,
    regrets: Vec<Regret>,
    seconds: f64,
    n_deadlines: usize,
    extras: Vec<Vec<String>>,
}

fn season_unit(
    store: &DataStore,
    caches: &Caches,
    rules: &Rules,
    season: i64,
    models: &[String],
    n_random: usize,
    seed: u64,
) -> Result<SeasonResult> {
    let began = std::time::Instant::now();
    let schedule = season_schedule(store, season, 1)?;
    let lockdowns: HashMap<i64, Micros> = schedule.iter().map(|g| (g.gw, g.lockdown)).collect();
    let mut realized: HashMap<i64, Option<GwTotals>> = HashMap::new();
    let get_gw = |gw: i64, realized: &mut HashMap<i64, Option<GwTotals>>| -> Result<()> {
        if let std::collections::hash_map::Entry::Vacant(e) = realized.entry(gw) {
            e.insert(match lockdowns.get(&gw) {
                Some(l) => gw_totals(store, rules, season, gw, *l)?,
                None => None,
            });
        }
        Ok(())
    };
    let active: Vec<usize> =
        (0..models.len()).filter(|&m| season >= first_season(&models[m])).collect();
    let mut extras: Vec<Vec<String>> = vec![Vec::new(); models.len()];
    let (mut predictions, mut regrets) = (Vec::new(), Vec::new());
    for g in &schedule {
        let view = store.as_of(g.deadline);
        let pool = caches.pool(&view)?;
        let positions: HashMap<i64, i64> =
            pool.iter().map(|p| (p.player_key, p.element_type)).collect();
        let frames: Vec<std::sync::Arc<crate::simulator::Xp>> =
            active.iter().map(|&m| caches.xp(&models[m], &view)).collect::<Result<_>>()?;
        let refs: Vec<&crate::simulator::Xp> = frames.iter().map(|f| f.as_ref()).collect();
        let candidates = candidate_keys(&refs, &positions);
        for (&m, xp) in active.iter().zip(&frames) {
            let f = &xp.frame;
            let missing = f.player_key.iter().filter(|k| !positions.contains_key(k)).count();
            if missing > 0 {
                bail!(
                    "{} at {}: {missing} xP row(s) for players outside the pool",
                    models[m],
                    fplopt_optimize::start::fmt_timestamp(g.deadline)
                );
            }
            let names: Vec<String> = xp
                .table
                .names()
                .into_iter()
                .filter(|n| !XP_BASE.contains(n))
                .map(str::to_owned)
                .collect();
            let columns: Vec<Vec<Option<f64>>> =
                names.iter().map(|n| xp.table.opt_f64(n)).collect::<Result<_>>()?;
            extras[m].clone_from(&names);
            let own = candidate_keys(&[xp.as_ref()], &positions);
            for i in 0..f.len() {
                let key = f.player_key[i];
                get_gw(f.gw[i], &mut realized)?;
                let values = match &realized[&f.gw[i]] {
                    None => [f64::NAN; 9],
                    Some(t) => match t.get(&key) {
                        Some(v) => *v,
                        None => {
                            let mut v = [0.0; 9];
                            for n in NULLABLE {
                                v[n] = f64::NAN;
                            }
                            v
                        }
                    },
                };
                predictions.push(Prediction {
                    model: m,
                    season,
                    deadline: g.deadline,
                    gw: g.gw,
                    gw_index: g.gw_index,
                    player_key: key,
                    element_type: positions[&key],
                    horizon: f.horizon[i],
                    target_gw: f.gw[i],
                    target_gw_index: f.gw_index[i],
                    xp: f.xp[i].unwrap_or(f64::NAN),
                    candidate: candidates.contains(&key),
                    own_candidate: own.contains(&key),
                    realized: values,
                    extras: columns.iter().map(|c| c[i].unwrap_or(f64::NAN)).collect(),
                });
            }
        }
        get_gw(g.gw, &mut realized)?;
        let Some(outcomes) = &realized[&g.gw] else { continue };
        let points: HashMap<i64, (i64, i64)> =
            outcomes.iter().map(|(k, v)| (*k, (v[0] as i64, v[1] as i64))).collect();
        for (squad_id, state) in deadline_squads(&view, rules, n_random, seed)? {
            let squad: Vec<(i64, i64)> =
                state.holdings.iter().map(|h| (h.player_key, h.element_type)).collect();
            let positions: HashMap<i64, i64> = squad.iter().copied().collect();
            for (&m, xp) in active.iter().zip(&frames) {
                let lineup = best_lineup(&squad, &target_xp(&xp.frame), rules)?;
                let score = score_gameweek(&lineup, &points, &positions, rules, None)?;
                let raw = |k: Option<i64>| k.map_or(0, |k| points.get(&k).map_or(0, |p| p.0));
                let best = hindsight_points(&squad, &points, rules)?;
                let captain = raw(score.captain_used);
                let best_captain = score.counted.iter().map(|k| raw(Some(*k))).max().unwrap_or(0);
                regrets.push(Regret {
                    model: m,
                    season,
                    deadline: g.deadline,
                    gw: g.gw,
                    gw_index: g.gw_index,
                    squad: squad_id.clone(),
                    points: score.points,
                    best_points: best,
                    xi_regret: best - score.points,
                    captain_points: captain,
                    best_captain_points: best_captain,
                    captain_regret: best_captain - captain,
                });
            }
        }
    }
    let seconds = began.elapsed().as_secs_f64();
    log::info!(
        "{}: {} deadlines, models {:?}, {seconds:.1}s",
        season_label(season as i32),
        schedule.len(),
        active.iter().map(|&m| &models[m]).collect::<Vec<_>>()
    );
    Ok(SeasonResult { predictions, regrets, seconds, n_deadlines: schedule.len(), extras })
}

/// `evaluate`'s output.
pub struct EvalResult {
    pub models: Vec<String>,
    pub predictions: Vec<Prediction>,
    pub regrets: Vec<Regret>,
    /// Per model, its frames' extra column names.
    pub extras: Vec<Vec<String>>,
    pub metrics: Ordered,
    /// Per season label, seconds.
    pub timings: Vec<(String, f64)>,
}

/// Walk-forward predictions and metrics of `models` over every deadline of `seasons`.
pub fn evaluate(
    store: &DataStore,
    models: &[String],
    seasons: &[i64],
    jobs: usize,
    n_random: usize,
    seed: u64,
    caches: &Caches,
) -> Result<EvalResult> {
    if models.is_empty() {
        bail!("no models to evaluate");
    }
    let seasons: Vec<i64> = seasons.iter().copied().collect::<BTreeSet<_>>().into_iter().collect();
    let holdout: Vec<String> = seasons
        .iter()
        .filter(|s| HOLDOUT_SEASONS.contains(&(**s as i32)))
        .map(|s| season_label(*s as i32))
        .collect();
    if !holdout.is_empty() {
        bail!(
            "HoldoutError: {} is a holdout season; the evaluation refuses it until Phase 6",
            holdout.join(", ")
        );
    }
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
                    let r = rules_for(seasons[i]).and_then(|rules| {
                        season_unit(store, caches, &rules, seasons[i], models, n_random, seed)
                    });
                    results.lock().unwrap()[i] = Some(r);
                }
            });
        }
    });
    let mut parts = Vec::new();
    for r in results.into_inner().unwrap() {
        parts.push(r.expect("every season ran")?);
    }
    let mut extras: Vec<Vec<String>> = vec![Vec::new(); models.len()];
    for part in &parts {
        for (m, names) in part.extras.iter().enumerate() {
            if extras[m].is_empty() {
                extras[m].clone_from(names);
            }
        }
    }
    let timings = seasons
        .iter()
        .zip(&parts)
        .map(|(s, p)| (season_label(*s as i32), (p.seconds * 10.0).round() / 10.0))
        .collect();
    let deadlines: Vec<(String, Ordered)> = seasons
        .iter()
        .zip(&parts)
        .map(|(s, p)| (season_label(*s as i32), Ordered::Int(p.n_deadlines as i128)))
        .collect();
    let mut predictions: Vec<Prediction> = Vec::new();
    let mut regrets: Vec<Regret> = Vec::new();
    for p in parts {
        predictions.extend(p.predictions);
        regrets.extend(p.regrets);
    }
    predictions.sort_by_key(|p| (p.model, p.deadline, p.player_key, p.horizon));
    regrets.sort_by(|a, b| (a.model, a.deadline, &a.squad).cmp(&(b.model, b.deadline, &b.squad)));
    let mut metrics = compute_metrics(&predictions, &regrets, models, &extras);
    if let Ordered::Object(pairs) = &mut metrics {
        let model_seasons = models
            .iter()
            .map(|m| {
                let s = seasons
                    .iter()
                    .filter(|s| **s >= first_season(m))
                    .map(|s| Ordered::Int(i128::from(*s)))
                    .collect();
                (m.clone(), Ordered::Array(s))
            })
            .collect();
        pairs.push(("seasons".to_owned(), Ordered::Object(model_seasons)));
        pairs.push(("deadlines".to_owned(), Ordered::Object(deadlines)));
    }
    Ok(EvalResult { models: models.to_vec(), predictions, regrets, extras, metrics, timings })
}

// --- metrics ------------------------------------------------------------------------------------

fn f(x: f64) -> Ordered {
    if x.is_finite() { Ordered::Float(x) } else { Ordered::Null }
}

fn i(x: usize) -> Ordered {
    Ordered::Int(x as i128)
}

fn s(x: &str) -> Ordered {
    Ordered::Str(x.to_owned())
}

fn rec(pairs: Vec<(&str, Ordered)>) -> Ordered {
    Ordered::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}

fn merge(a: Ordered, b: Ordered) -> Ordered {
    match (a, b) {
        (Ordered::Object(mut x), Ordered::Object(y)) => {
            x.extend(y);
            Ordered::Object(x)
        }
        (a, _) => a,
    }
}

/// numpy's mean (NaN propagates).
fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { f64::NAN } else { np_sum(v) / v.len() as f64 }
}

/// pandas' mean (NaN skipped).
fn nanmean(v: &[f64]) -> f64 {
    let kept: Vec<f64> = v.iter().copied().filter(|x| !x.is_nan()).collect();
    mean(&kept)
}

fn mse(p: &[f64], a: &[f64]) -> f64 {
    mean(&p.iter().zip(a).map(|(x, y)| (x - y) * (x - y)).collect::<Vec<_>>())
}

fn mae(p: &[f64], a: &[f64]) -> f64 {
    mean(&p.iter().zip(a).map(|(x, y)| (x - y).abs()).collect::<Vec<_>>())
}

fn splits() -> [(&'static str, Option<Vec<i64>>); 3] {
    [
        ("all", None),
        ("develop", Some(DEVELOP_SEASONS.map(i64::from).collect())),
        ("validate", Some(VALIDATE_SEASONS.map(i64::from).collect())),
    ]
}

fn split_parts<'a, T>(
    rows: &[&'a T],
    season: impl Fn(&T) -> i64,
) -> Vec<(&'static str, Vec<&'a T>)> {
    splits()
        .into_iter()
        .filter_map(|(name, seasons)| {
            let part: Vec<&T> = rows
                .iter()
                .copied()
                .filter(|r| seasons.as_ref().is_none_or(|s| s.contains(&season(r))))
                .collect();
            (!part.is_empty()).then_some((name, part))
        })
        .collect()
}

fn horizon_parts<'a>(rows: &[&'a Prediction]) -> Vec<(&'static str, Vec<&'a Prediction>, usize)> {
    HORIZON_LABELS
        .iter()
        .filter_map(|label| {
            let (part, lag): (Vec<&Prediction>, usize) = if *label == "1-5" {
                (rows.iter().copied().filter(|r| (1..=5).contains(&r.horizon)).collect(), 5)
            } else {
                let h: i64 = label.parse().unwrap();
                (rows.iter().copied().filter(|r| r.horizon == h).collect(), h as usize)
            };
            (!part.is_empty()).then_some((*label, part, lag))
        })
        .collect()
}

fn xp_row(part: &[&Prediction]) -> Vec<(&'static str, Ordered)> {
    let xp: Vec<f64> = part.iter().map(|r| r.xp).collect();
    let pts: Vec<f64> = part.iter().map(|r| r.realized[0]).collect();
    let cand: Vec<&&Prediction> = part.iter().filter(|r| r.candidate).collect();
    let cxp: Vec<f64> = cand.iter().map(|r| r.xp).collect();
    let cpts: Vec<f64> = cand.iter().map(|r| r.realized[0]).collect();
    let deadlines: BTreeSet<Micros> = part.iter().map(|r| r.deadline).collect();
    vec![
        ("n", i(part.len())),
        ("n_deadlines", i(deadlines.len())),
        ("mse", f(mse(&xp, &pts))),
        ("mse_candidates", f(mse(&cxp, &cpts))),
        ("n_candidates", i(cand.len())),
        ("mae", f(mae(&xp, &pts))),
        ("mean_xp", f(nanmean(&xp))),
        ("mean_points", f(nanmean(&pts))),
    ]
}

/// Bin k holds edges[k] ≤ v < edges[k + 1]; the last bin includes its upper edge.
fn bin_index(v: f64, edges: &[f64]) -> usize {
    let idx = edges.partition_point(|e| *e <= v) as i64 - 1;
    idx.clamp(0, edges.len() as i64 - 2) as usize
}

/// Per band of the predicted value: band, lo, hi, n, mean_pred, mean_actual, mse.
fn band_table(p: &[f64], a: &[f64], edges: &[f64]) -> Vec<Ordered> {
    let mut groups: BTreeMap<usize, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    for (x, y) in p.iter().zip(a) {
        let g = groups.entry(bin_index(*x, edges)).or_default();
        g.0.push(*x);
        g.1.push(*y);
    }
    groups
        .into_iter()
        .map(|(k, (x, y))| {
            rec(vec![
                ("band", i(k)),
                ("lo", f(edges[k])),
                ("hi", f(edges[k + 1])),
                ("n", i(x.len())),
                ("mean_pred", f(mean(&x))),
                ("mean_actual", f(mean(&y))),
                ("mse", f(mse(&x, &y))),
            ])
        })
        .collect()
}

fn probability_edges(bins: usize) -> Vec<f64> {
    let step = 1.0 / bins as f64;
    let mut edges: Vec<f64> = (0..=bins).map(|k| k as f64 * step).collect();
    edges[bins] = 1.0;
    edges
}

fn brier(p: &[f64], y: &[f64], bins: usize) -> Vec<(&'static str, Ordered)> {
    if p.is_empty() {
        return vec![
            ("brier", Ordered::Null),
            ("reliability", Ordered::Null),
            ("resolution", Ordered::Null),
            ("uncertainty", Ordered::Null),
            ("n", i(0)),
            ("n_bins", i(bins)),
        ];
    }
    let edges = probability_edges(bins);
    let base = mean(y);
    let mut groups: BTreeMap<usize, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    for (x, o) in p.iter().zip(y) {
        let g = groups.entry(bin_index(*x, &edges)).or_default();
        g.0.push(*x);
        g.1.push(*o);
    }
    let (mut reliability, mut resolution) = (0.0, 0.0);
    for (x, o) in groups.values() {
        let n = x.len() as f64;
        reliability += n * (mean(x) - mean(o)).powi(2);
        resolution += n * (mean(o) - base).powi(2);
    }
    let n = p.len() as f64;
    vec![
        ("brier", f(mse(p, y))),
        ("reliability", f(reliability / n)),
        ("resolution", f(resolution / n)),
        ("uncertainty", f(base * (1.0 - base))),
        ("n", i(p.len())),
        ("n_bins", i(bins)),
    ]
}

/// The component scored against a realized outcome (Python `metrics.COMPONENTS`).
struct Component {
    name: &'static str,
    kind: &'static str,
    columns: &'static [&'static str],
    target: usize,
    threshold: f64,
    single_fixture: bool,
}

const COMPONENTS: [Component; 11] = [
    Component {
        name: "p_start",
        kind: "prob",
        columns: &["p_start"],
        target: 2,
        threshold: 1.0,
        single_fixture: true,
    },
    Component {
        name: "p_play",
        kind: "prob",
        columns: &["p_play"],
        target: 1,
        threshold: 1.0,
        single_fixture: true,
    },
    Component {
        name: "p_60",
        kind: "prob",
        columns: &["p_60"],
        target: 1,
        threshold: 60.0,
        single_fixture: true,
    },
    Component {
        name: "minutes",
        kind: "ordinal",
        columns: &["p_min_0", "p_min_1_59", "p_min_60"],
        target: 1,
        threshold: 1.0,
        single_fixture: true,
    },
    Component {
        name: "e_minutes",
        kind: "value",
        columns: &["e_minutes"],
        target: 1,
        threshold: 1.0,
        single_fixture: false,
    },
    Component {
        name: "p_cs",
        kind: "prob",
        columns: &["p_cs"],
        target: 6,
        threshold: 1.0,
        single_fixture: true,
    },
    Component {
        name: "p_goal",
        kind: "prob",
        columns: &["p_goal"],
        target: 4,
        threshold: 1.0,
        single_fixture: true,
    },
    Component {
        name: "e_goals",
        kind: "count",
        columns: &["e_goals"],
        target: 4,
        threshold: 1.0,
        single_fixture: false,
    },
    Component {
        name: "e_assists",
        kind: "count",
        columns: &["e_assists"],
        target: 5,
        threshold: 1.0,
        single_fixture: false,
    },
    Component {
        name: "e_bonus",
        kind: "count",
        columns: &["e_bonus"],
        target: 7,
        threshold: 1.0,
        single_fixture: false,
    },
    Component {
        name: "p_defcon",
        kind: "prob",
        columns: &["p_defcon"],
        target: 8,
        threshold: 1.0,
        single_fixture: true,
    },
];

fn component_metrics(
    rows: &[&Prediction],
    c: &Component,
    cols: &[usize],
) -> Result<Vec<(&'static str, Ordered)>> {
    let kept: Vec<&&Prediction> = rows
        .iter()
        .filter(|r| !r.realized[c.target].is_nan())
        .filter(|r| !c.single_fixture || r.realized[3] == 1.0)
        .filter(|r| cols.iter().all(|&k| !r.extras[k].is_nan()))
        .collect();
    let target: Vec<f64> = kept.iter().map(|r| r.realized[c.target]).collect();
    let col = |j: usize| kept.iter().map(|r| r.extras[cols[j]]).collect::<Vec<f64>>();
    let mut out = vec![("n", i(kept.len()))];
    match c.kind {
        "prob" => {
            let p = col(0);
            let y: Vec<f64> =
                target.iter().map(|t| f64::from(u8::from(*t >= c.threshold))).collect();
            out.extend(brier(&p, &y, 10));
            let table = band_table(&p, &y, &probability_edges(10))
                .into_iter()
                .map(|r| match r {
                    Ordered::Object(pairs) => Ordered::Object(
                        pairs
                            .into_iter()
                            .filter(|(k, _)| k != "mse")
                            .map(|(k, v)| {
                                let k = match k.as_str() {
                                    "band" => "bin".to_owned(),
                                    "mean_actual" => "observed_rate".to_owned(),
                                    _ => k,
                                };
                                (k, v)
                            })
                            .collect(),
                    ),
                    other => other,
                })
                .collect();
            out.push(("reliability_table", Ordered::Array(table)));
        }
        "ordinal" => {
            let probs: Vec<[f64; 3]> = kept
                .iter()
                .map(|r| [r.extras[cols[0]], r.extras[cols[1]], r.extras[cols[2]]])
                .collect();
            for p in &probs {
                if p.iter().any(|x| *x < -1e-9) || (p.iter().sum::<f64>() - 1.0).abs() > 1e-6 + 1e-5
                {
                    bail!("each row of probs must be >= 0 and sum to 1");
                }
            }
            let class = |m: f64| {
                if m >= 60.0 {
                    2
                } else if m > 0.0 {
                    1
                } else {
                    0
                }
            };
            let ll: Vec<f64> = probs
                .iter()
                .zip(&target)
                .map(|(p, t)| -p[class(*t)].clamp(LOG_EPS, 1.0).ln())
                .collect();
            let rps: Vec<f64> = probs
                .iter()
                .zip(&target)
                .map(|(p, t)| {
                    let c = class(*t);
                    let f0 = p[0];
                    let f1 = p[0] + p[1];
                    let o0 = f64::from(u8::from(c == 0));
                    let o1 = f64::from(u8::from(c <= 1));
                    ((f0 - o0).powi(2) + (f1 - o1).powi(2)) / 2.0
                })
                .collect();
            out.push(("log_loss", f(mean(&ll))));
            out.push(("rps", f(mean(&rps))));
        }
        "count" => {
            let rate = col(0);
            let max = target.iter().copied().fold(0.0, f64::max) as usize;
            let mut log_fact = vec![0.0; max + 1];
            for k in 1..=max {
                log_fact[k] = log_fact[k - 1] + (k as f64).ln();
            }
            let ll: Vec<f64> = rate
                .iter()
                .zip(&target)
                .map(|(l, k)| {
                    let l = l.max(RATE_EPS);
                    k * l.ln() - l - log_fact[*k as usize]
                })
                .collect();
            out.push(("log_likelihood", f(mean(&ll))));
            out.push(("mse", f(mse(&rate, &target))));
            out.push(("mean_pred", f(mean(&rate))));
            out.push(("mean_actual", f(mean(&target))));
        }
        _ => {
            let v = col(0);
            out.push(("mse", f(mse(&v, &target))));
            out.push(("mae", f(mae(&v, &target))));
        }
    }
    Ok(out)
}

/// One-sided Diebold-Mariano test on paired losses clustered by deadline (Harvey-Leybourne-
/// Newbold corrected, Student t on G − 1 df).
fn diebold_mariano(loss: &[(Micros, f64, f64)], lag: usize) -> Vec<(&'static str, Ordered)> {
    let mut clusters: BTreeMap<Micros, (f64, f64, usize)> = BTreeMap::new();
    for (d, a, b) in loss {
        let e = clusters.entry(*d).or_insert((0.0, 0.0, 0));
        let v = a - b;
        let y = v - e.1;
        let t = e.0 + y;
        e.1 = t - e.0 - y;
        e.0 = t;
        e.2 += 1;
    }
    let d: Vec<f64> = clusters.values().map(|(s, _, n)| s / *n as f64).collect();
    let n = d.len();
    let nan = Ordered::Null;
    let dm = |mean_diff: Ordered, t: Ordered, pa: Ordered, pb: Ordered| {
        vec![
            ("mean_diff", mean_diff),
            ("t_stat", t),
            ("p_a_better", pa),
            ("p_b_better", pb),
            ("n_clusters", i(n)),
            ("lag", i(lag)),
        ]
    };
    if n < 2 {
        return dm(if n == 1 { f(d[0]) } else { Ordered::Null }, nan.clone(), nan.clone(), nan);
    }
    let m = mean(&d);
    let dev: Vec<f64> = d.iter().map(|x| x - m).collect();
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).fold(0.0, |acc, (x, y)| acc + x * y);
    let mut variance = dot(&dev, &dev) / n as f64;
    for k in 1..=lag.min(n - 1) {
        variance +=
            2.0 * (1.0 - k as f64 / (lag + 1) as f64) * dot(&dev[k..], &dev[..n - k]) / n as f64;
    }
    let k = (lag + 1) as f64;
    let nf = n as f64;
    let correction = (nf + 1.0 - 2.0 * k + k * (k - 1.0) / nf) / nf;
    if variance.is_nan() || variance <= 0.0 || correction.is_nan() || correction <= 0.0 {
        return dm(f(m), nan.clone(), nan.clone(), nan);
    }
    let t = m / (variance / nf).sqrt() * correction.sqrt();
    dm(f(m), f(t), f(t_sf(-t, n - 1)), f(t_sf(t, n - 1)))
}

fn dm_row(
    a: &str,
    b: &str,
    rows: &[(Micros, f64, f64, i64)],
    metric: &str,
    split: &str,
    horizon: &str,
    lag: usize,
) -> Ordered {
    let seasons: BTreeSet<i64> = rows.iter().map(|r| r.3).collect();
    let la: Vec<f64> = rows.iter().map(|r| r.1).collect();
    let lb: Vec<f64> = rows.iter().map(|r| r.2).collect();
    let loss: Vec<(Micros, f64, f64)> = rows.iter().map(|r| (r.0, r.1, r.2)).collect();
    let mut pairs = vec![
        ("a", s(a)),
        ("b", s(b)),
        ("metric", s(metric)),
        ("split", s(split)),
        ("horizon", s(horizon)),
        ("n", i(rows.len())),
        (
            "seasons",
            Ordered::Array(seasons.into_iter().map(|x| Ordered::Int(i128::from(x))).collect()),
        ),
        ("value_a", f(nanmean(&la))),
        ("value_b", f(nanmean(&lb))),
    ];
    pairs.extend(diebold_mariano(&loss, lag));
    rec(pairs)
}

/// The metric tables from the predictions and regrets (rows of GWs not played left out).
pub fn compute_metrics(
    predictions: &[Prediction],
    regrets: &[Regret],
    models: &[String],
    extras: &[Vec<String>],
) -> Ordered {
    let scored: Vec<&Prediction> = predictions.iter().filter(|p| !p.realized[0].is_nan()).collect();
    let (mut xp_rows, mut band_rows, mut season_rows, mut regret_rows, mut component_rows) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (m, model) in models.iter().enumerate() {
        let mine: Vec<&Prediction> = scored.iter().copied().filter(|p| p.model == m).collect();
        for (split, part) in split_parts(&mine, |p| p.season) {
            for (label, rows, _) in horizon_parts(&part) {
                let head = vec![("model", s(model)), ("split", s(split)), ("horizon", s(label))];
                xp_rows.push(merge(rec(head.clone()), rec(xp_row(&rows))));
                let p: Vec<f64> = rows.iter().map(|r| r.xp).collect();
                let a: Vec<f64> = rows.iter().map(|r| r.realized[0]).collect();
                for band in band_table(&p, &a, &XP_BANDS) {
                    band_rows.push(merge(rec(head.clone()), band));
                }
                for c in &COMPONENTS {
                    let Some(cols) = c
                        .columns
                        .iter()
                        .map(|n| extras[m].iter().position(|e| e == n))
                        .collect::<Option<Vec<usize>>>()
                    else {
                        continue;
                    };
                    if rows.iter().all(|r| cols.iter().all(|&k| r.extras[k].is_nan())) {
                        continue;
                    }
                    if rows.iter().all(|r| r.realized[c.target].is_nan()) {
                        continue;
                    }
                    if let Ok(values) = component_metrics(&rows, c, &cols) {
                        component_rows.push(merge(
                            rec(vec![
                                ("model", s(model)),
                                ("component", s(c.name)),
                                ("split", s(split)),
                                ("horizon", s(label)),
                            ]),
                            rec(values),
                        ));
                    }
                }
            }
        }
        let h0: Vec<&Prediction> = mine.iter().copied().filter(|p| p.horizon == 0).collect();
        let seasons: BTreeSet<i64> = h0.iter().map(|p| p.season).collect();
        for season in seasons {
            let part: Vec<&Prediction> =
                h0.iter().copied().filter(|p| p.season == season).collect();
            let row: BTreeMap<&str, Ordered> = xp_row(&part).into_iter().collect();
            let mut pairs = vec![("model", s(model)), ("season", Ordered::Int(i128::from(season)))];
            for k in ["n", "mse", "mse_candidates", "mae", "mean_xp", "mean_points"] {
                pairs.push((k, row[k].clone()));
            }
            season_rows.push(rec(pairs));
        }
        let theirs: Vec<&Regret> = regrets.iter().filter(|r| r.model == m).collect();
        for (split, part) in split_parts(&theirs, |r| r.season) {
            let col =
                |g: fn(&Regret) -> i64| mean(&part.iter().map(|r| g(r) as f64).collect::<Vec<_>>());
            let deadlines: BTreeSet<Micros> = part.iter().map(|r| r.deadline).collect();
            regret_rows.push(rec(vec![
                ("model", s(model)),
                ("split", s(split)),
                ("n", i(part.len())),
                ("n_deadlines", i(deadlines.len())),
                ("xi_regret", f(col(|r| r.xi_regret))),
                ("captain_regret", f(col(|r| r.captain_regret))),
                ("points", f(col(|r| r.points))),
                ("best_points", f(col(|r| r.best_points))),
            ]));
        }
    }
    let mut dm_rows = Vec::new();
    for (ia, a) in models.iter().enumerate() {
        for (ib, b) in models.iter().enumerate().skip(ia + 1) {
            let side_b: HashMap<(Micros, i64, i64), &Prediction> = scored
                .iter()
                .filter(|p| p.model == ib)
                .map(|p| ((p.deadline, p.player_key, p.horizon), *p))
                .collect();
            // (deadline, loss_a, loss_b, season, horizon, candidate)
            let merged: Vec<(Micros, f64, f64, i64, i64, bool)> = scored
                .iter()
                .filter(|p| p.model == ia)
                .filter_map(|p| {
                    let q = side_b.get(&(p.deadline, p.player_key, p.horizon))?;
                    let pts = p.realized[0];
                    Some((
                        p.deadline,
                        (p.xp - pts).powi(2),
                        (q.xp - pts).powi(2),
                        p.season,
                        p.horizon,
                        p.own_candidate || q.own_candidate,
                    ))
                })
                .collect();
            let refs: Vec<&(Micros, f64, f64, i64, i64, bool)> = merged.iter().collect();
            for (split, part) in split_parts(&refs, |r| r.3) {
                for label in HORIZON_LABELS {
                    let (rows, lag): (Vec<_>, usize) = if label == "1-5" {
                        (part.iter().filter(|r| (1..=5).contains(&r.4)).collect(), 5)
                    } else {
                        let h: i64 = label.parse().unwrap();
                        (part.iter().filter(|r| r.4 == h).collect(), h as usize)
                    };
                    if rows.is_empty() {
                        continue;
                    }
                    let all: Vec<(Micros, f64, f64, i64)> =
                        rows.iter().map(|r| (r.0, r.1, r.2, r.3)).collect();
                    dm_rows.push(dm_row(a, b, &all, "mse", split, label, lag));
                    let cand: Vec<(Micros, f64, f64, i64)> =
                        rows.iter().filter(|r| r.5).map(|r| (r.0, r.1, r.2, r.3)).collect();
                    if !cand.is_empty() {
                        dm_rows.push(dm_row(a, b, &cand, "mse_candidates", split, label, lag));
                    }
                }
            }
            let rb: HashMap<(Micros, &str, i64), &Regret> = regrets
                .iter()
                .filter(|r| r.model == ib)
                .map(|r| ((r.deadline, r.squad.as_str(), r.season), r))
                .collect();
            for (metric, g) in [
                ("xi_regret", (|r: &Regret| r.xi_regret) as fn(&Regret) -> i64),
                ("captain_regret", |r: &Regret| r.captain_regret),
            ] {
                let pair: Vec<(Micros, f64, f64, i64)> = regrets
                    .iter()
                    .filter(|r| r.model == ia)
                    .filter_map(|r| {
                        let q = rb.get(&(r.deadline, r.squad.as_str(), r.season))?;
                        Some((r.deadline, g(r) as f64, g(q) as f64, r.season))
                    })
                    .collect();
                let refs: Vec<&(Micros, f64, f64, i64)> = pair.iter().collect();
                for (split, part) in split_parts(&refs, |r| r.3) {
                    let rows: Vec<(Micros, f64, f64, i64)> = part.iter().map(|r| **r).collect();
                    dm_rows.push(dm_row(a, b, &rows, metric, split, "0", 0));
                }
            }
        }
    }
    rec(vec![
        ("models", Ordered::Array(models.iter().map(|m| s(m)).collect())),
        ("xp", Ordered::Array(xp_rows)),
        ("bands", Ordered::Array(band_rows)),
        ("by_season", Ordered::Array(season_rows)),
        ("regret", Ordered::Array(regret_rows)),
        ("dm", Ordered::Array(dm_rows)),
        ("components", Ordered::Array(component_rows)),
    ])
}
