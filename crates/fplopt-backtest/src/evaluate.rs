//! Paired policy evaluation (Python `fplopt.backtest.evaluate`; PLAN §5 *Comparing
//! policies*): every policy from the same start states (`run_grid`), full-run paired
//! differences, per-decision differences from a reference trajectory, and the summary
//! (season totals, comparisons with block-bootstrap CIs per split, transfer gains).
//!
//! Units are (season, start state); with `jobs > 1` they run on threads sharing the store
//! and the caches, and results are concatenated in unit order, so the output equals
//! `jobs = 1`'s.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Result, bail};
use fplopt_build::scoring::{Rules, backtest_rules, default_config_dir};
use fplopt_core::numpy::np_sum;
use fplopt_core::pyjson::Ordered;
use fplopt_core::seasons::{DEVELOP_SEASONS, HOLDOUT_SEASONS, VALIDATE_SEASONS, season_label};
use fplopt_optimize::start::{MAX_BANK, StartStateError, random_state, template_state};
use fplopt_store::DataStore;

use crate::simulator::{
    Caches, Decider, GwRow, decide_step, run_gameweeks, season_schedule, simulate,
};
use crate::state::{Decision, SquadState};
use crate::stats::{
    BootstrapResult, Diff, block_bootstrap, deflated_mean, season_points, season_t_test,
};

// --- start states ---------------------------------------------------------------------------

/// One start state: the template squad, or a random squad with `seed`, at `gw_index`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StartSpec {
    pub random: bool,
    pub gw_index: i64,
    pub seed: u64,
}

impl StartSpec {
    pub fn start_id(&self, gw_index: Option<i64>) -> String {
        let gw = gw_index.unwrap_or(self.gw_index);
        if self.random { format!("random{}@{gw}", self.seed) } else { format!("template@{gw}") }
    }

    fn build(
        &self,
        store: &DataStore,
        season: i64,
        gw_index: i64,
        rules: &Rules,
    ) -> Result<SquadState> {
        let deadline = season_schedule(store, season, gw_index)?[0].deadline;
        let view = store.as_of(deadline);
        if self.random {
            random_state(&view, rules, self.seed, Some(MAX_BANK))
        } else {
            template_state(&view, rules)
        }
    }
}

/// `"template@1,random:5@1,random:3@20"` -> the template at gw_index 1, random seeds 0-4 at
/// 1 and seeds 0-2 at 20 (`random@g` = seed 0). Duplicates raise.
pub fn parse_start_specs(text: &str) -> Result<Vec<StartSpec>> {
    let mut specs = Vec::new();
    for item in text.split(',').map(str::trim) {
        let bad = || {
            anyhow::anyhow!(
                "bad start spec '{item}': expected template@<gw>, random@<gw> or random:<n>@<gw>"
            )
        };
        let (kind, gw) = item.split_once('@').ok_or_else(bad)?;
        if gw.is_empty() || !gw.bytes().all(|b| b.is_ascii_digit()) {
            return Err(bad());
        }
        let gw_index: i64 = gw.parse()?;
        if gw_index < 1 {
            bail!("bad start spec '{item}': gw_index must be >= 1");
        }
        if kind == "template" {
            specs.push(StartSpec { random: false, gw_index, seed: 0 });
        } else if kind == "random" || kind.starts_with("random:") {
            let n: u64 = match kind.strip_prefix("random:") {
                None => 1,
                Some(n) if !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()) => n.parse()?,
                Some(_) => return Err(bad()),
            };
            if n < 1 {
                bail!("bad start spec '{item}': need at least one seed");
            }
            specs.extend((0..n).map(|seed| StartSpec { random: true, gw_index, seed }));
        } else {
            return Err(bad());
        }
    }
    let ids: Vec<String> = specs.iter().map(|s| s.start_id(None)).collect();
    let duplicates: BTreeSet<&String> =
        ids.iter().filter(|i| ids.iter().filter(|j| j == i).count() > 1).collect();
    if !duplicates.is_empty() {
        let list: Vec<String> = duplicates.iter().map(|d| format!("'{d}'")).collect();
        bail!("duplicate start specs: [{}]", list.join(", "));
    }
    if specs.is_empty() {
        bail!("no start specs");
    }
    Ok(specs)
}

fn refused(err: &anyhow::Error) -> bool {
    err.downcast_ref::<StartStateError>().is_some()
}

/// (start_id, state) per spec for `season`. Refused states are skipped with a warning; at
/// gw_index 1 a refusal falls back to gw_index 2 (the id records the GW used).
pub fn build_start_states(
    store: &DataStore,
    season: i64,
    specs: &[StartSpec],
    rules: &Rules,
) -> Result<Vec<(String, SquadState)>> {
    let label = season_label(season as i32);
    let mut out: Vec<(String, SquadState)> = Vec::new();
    for spec in specs {
        let (state, gw_index) = match spec.build(store, season, spec.gw_index, rules) {
            Ok(state) => (state, spec.gw_index),
            Err(e) if refused(&e) => {
                if spec.gw_index != 1 {
                    log::warn!("{label} {} refused, skipped: {e}", spec.start_id(None));
                    continue;
                }
                log::info!("{label} {} refused ({e}); starting at gw_index 2", spec.start_id(None));
                match spec.build(store, season, 2, rules) {
                    Ok(state) => (state, 2),
                    Err(e2) if refused(&e2) => {
                        log::warn!("{label} {} refused, skipped: {e2}", spec.start_id(Some(2)));
                        continue;
                    }
                    Err(e2) => return Err(e2),
                }
            }
            Err(e) => return Err(e),
        };
        let id = spec.start_id(Some(gw_index));
        if out.iter().any(|(i, _)| *i == id) {
            log::warn!("{label} {id}: duplicate start id after fallback, skipped");
            continue;
        }
        out.push((id, state));
    }
    Ok(out)
}

/// (season, start_id, start state).
pub type Unit = (i64, String, SquadState);

/// The backtest rules of a season (from the repo's `config/scoring`).
pub fn rules_for(season: i64) -> Result<Rules> {
    backtest_rules(season, &default_config_dir())
}

/// The units of `seasons` × start specs, in order.
pub fn units(store: &DataStore, seasons: &[i64], specs: &[StartSpec]) -> Result<Vec<Unit>> {
    let mut out = Vec::new();
    for &season in seasons {
        let rules = rules_for(season)?;
        for (id, state) in build_start_states(store, season, specs, &rules)? {
            out.push((season, id, state));
        }
    }
    Ok(out)
}

/// Run `f` over the units on `jobs` threads; results in unit order. The first error stops
/// the run.
pub fn run_units<T: Send>(
    units: &[Unit],
    jobs: usize,
    f: impl Fn(&Unit, &Rules) -> Result<T> + Sync,
) -> Result<Vec<T>> {
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<Result<T>>>> =
        Mutex::new((0..units.len()).map(|_| None).collect());
    let failed = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|scope| {
        for _ in 0..jobs.max(1).min(units.len().max(1)) {
            scope.spawn(|| {
                let mut rules: BTreeMap<i64, Rules> = BTreeMap::new();
                loop {
                    if failed.load(Ordering::Relaxed) {
                        break;
                    }
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= units.len() {
                        break;
                    }
                    let unit = &units[i];
                    let result = match rules.get(&unit.0) {
                        Some(r) => f(unit, r),
                        None => rules_for(unit.0).and_then(|r| {
                            let out = f(unit, &r);
                            rules.insert(unit.0, r);
                            out
                        }),
                    };
                    if result.is_err() {
                        failed.store(true, Ordering::Relaxed);
                    }
                    results.lock().unwrap()[i] = Some(result);
                }
            });
        }
    });
    let mut out = Vec::with_capacity(units.len());
    for r in results.into_inner().unwrap() {
        match r {
            Some(Ok(v)) => out.push(v),
            Some(Err(e)) => return Err(e),
            None => {} // not run after a failure; the failure is returned above
        }
    }
    Ok(out)
}

fn refuse_holdout(seasons: &[i64]) -> Result<()> {
    let holdout: Vec<String> = seasons
        .iter()
        .filter(|s| HOLDOUT_SEASONS.contains(&(**s as i32)))
        .map(|s| season_label(*s as i32))
        .collect();
    if !holdout.is_empty() {
        bail!(
            "HoldoutError: {} is a holdout season; backtests refuse it until Phase 6",
            holdout.join(", ")
        );
    }
    Ok(())
}

// --- the grid -------------------------------------------------------------------------------

/// One GW of one policy's run from one start state.
#[derive(Clone, Debug)]
pub struct GridRow {
    pub start_id: String,
    pub policy: String,
    pub row: GwRow,
}

/// Every policy from every start state of every season (one `simulate` each).
pub fn run_grid(
    store: &DataStore,
    seasons: &[i64],
    specs: &[StartSpec],
    policies: &[crate::policies::Policy],
    caches: &Caches,
    jobs: usize,
) -> Result<Vec<GridRow>> {
    refuse_holdout(seasons)?;
    let names: Vec<String> = policies.iter().map(|p| p.name()).collect();
    if names.iter().collect::<BTreeSet<_>>().len() != names.len() {
        bail!("policy names must be unique: {names:?}");
    }
    let units = units(store, seasons, specs)?;
    let results = run_units(&units, jobs, |(_, start_id, state), rules| {
        let mut rows = Vec::new();
        for (policy, name) in policies.iter().zip(&names) {
            let run = simulate(store, rules, policy, state, None, caches, false)?;
            rows.extend(run.rows.into_iter().map(|row| GridRow {
                start_id: start_id.clone(),
                policy: name.clone(),
                row,
            }));
        }
        Ok(rows)
    })?;
    Ok(results.into_iter().flatten().collect())
}

/// One full-run paired difference.
#[derive(Clone, Debug, PartialEq)]
pub struct Paired {
    pub season: i64,
    pub start_id: String,
    pub gw: i64,
    pub gw_index: i64,
    pub points_a: i64,
    pub points_b: i64,
    pub diff: i64,
    pub xg_a: Option<f64>,
    pub xg_b: Option<f64>,
    pub diff_xg: Option<f64>,
}

/// Per (season, start_id, gw_index) where both ran: net points of A and B and A − B
/// (realized and xG), sorted by season, start_id, gw_index.
pub fn paired_full_run(results: &[GridRow], a: &str, b: &str) -> Result<Vec<Paired>> {
    let side = |name: &str| -> Result<BTreeMap<(i64, String, i64, i64), &GwRow>> {
        let rows: BTreeMap<_, _> = results
            .iter()
            .filter(|r| r.policy == name)
            .map(|r| ((r.row.season, r.start_id.clone(), r.row.gw, r.row.gw_index), &r.row))
            .collect();
        if rows.is_empty() {
            bail!("no results for policy '{name}'");
        }
        Ok(rows)
    };
    let (sa, sb) = (side(a)?, side(b)?);
    let mut out: Vec<Paired> = sa
        .iter()
        .filter_map(|(key, ra)| {
            let rb = sb.get(key)?;
            let diff_xg = match (ra.xg_net_points, rb.xg_net_points) {
                (Some(x), Some(y)) => Some(x - y),
                _ => None,
            };
            Some(Paired {
                season: key.0,
                start_id: key.1.clone(),
                gw: key.2,
                gw_index: key.3,
                points_a: ra.net_points,
                points_b: rb.net_points,
                diff: ra.net_points - rb.net_points,
                xg_a: ra.xg_net_points,
                xg_b: rb.xg_net_points,
                diff_xg,
            })
        })
        .collect();
    out.sort_by(|x, y| {
        (x.season, &x.start_id, x.gw_index).cmp(&(y.season, &y.start_id, y.gw_index))
    });
    Ok(out)
}

// --- per decision ---------------------------------------------------------------------------

/// One per-decision row (Python `PER_DECISION_COLUMNS`).
#[derive(Clone, Debug, PartialEq)]
pub struct PerDecision {
    pub season: i64,
    pub start_id: String,
    pub policy_a: String,
    pub policy_b: String,
    pub reference: String,
    pub continuation: String,
    pub gw: i64,
    pub gw_index: i64,
    pub k: i64,
    pub stride: i64,
    pub same_decision: bool,
    pub transfers_a: i64,
    pub transfers_b: i64,
    pub points_a: i64,
    pub points_b: i64,
    pub diff: i64,
    pub xg_a: Option<f64>,
    pub xg_b: Option<f64>,
    pub diff_xg: Option<f64>,
}

/// Σ net points and Σ xG net points (None if any GW's is null) over an arm's GWs.
fn window_scores(rows: &[GwRow]) -> (i64, Option<f64>) {
    let points = rows.iter().map(|r| r.net_points).sum();
    let xg: Option<Vec<f64>> = rows.iter().map(|r| r.xg_net_points).collect();
    (points, xg.map(|v| v.iter().fold(0.0, |acc, x| acc + x)))
}

/// Per-decision paired differences: states from the reference policy's own trajectory; at
/// each decision GW on the season's grid (gw_index ≡ 1 mod `stride`) both arms start from
/// that state, play their own decision, then follow `continuation` (None: each its own
/// policy) for k − 1 GWs. Each decision is made once.
#[allow(clippy::too_many_arguments)]
pub fn per_decision(
    store: &DataStore,
    seasons: &[i64],
    specs: &[StartSpec],
    policy_a: &crate::policies::Policy,
    policy_b: &crate::policies::Policy,
    continuation: Option<&crate::policies::Policy>,
    k: usize,
    stride: usize,
    reference: &str,
    caches: &Caches,
    jobs: usize,
) -> Result<Vec<PerDecision>> {
    if k < 1 || stride < 1 {
        bail!("k and stride must be >= 1");
    }
    if reference != "a" && reference != "b" {
        bail!("reference must be 'a' or 'b', got '{reference}'");
    }
    refuse_holdout(seasons)?;
    let (name_a, name_b) = (policy_a.name(), policy_b.name());
    let cont_name = continuation.map_or("own".to_owned(), |c| c.name());
    let units = units(store, seasons, specs)?;
    let results = run_units(&units, jobs, |(season, start_id, start), rules| {
        let ref_policy = if reference == "a" { policy_a } else { policy_b };
        let r = simulate(store, rules, ref_policy, start, None, caches, false)?;
        let mut schedule = season_schedule(store, *season, start.gw_index)?;
        schedule.truncate(r.rows.len());
        let mut rows = Vec::new();
        for (i, g) in schedule.iter().enumerate() {
            if (g.gw_index - 1) % stride as i64 != 0 {
                continue;
            }
            let state = &r.states[i];
            let window = &schedule[i..(i + k).min(schedule.len())];
            let deadline = window[0].deadline;
            let mut decisions: BTreeMap<&str, Decision> = BTreeMap::new();
            for (side, policy) in [("a", policy_a), ("b", policy_b)] {
                let d = if side == reference {
                    r.decisions[i].clone()
                } else {
                    decide_step(store, rules, Decider::Policy(policy), state, deadline, caches)?
                        .decision
                };
                decisions.insert(side, d);
            }
            let same = decisions["a"] == decisions["b"];
            let mut scores: BTreeMap<&str, ((i64, Option<f64>), i64)> = BTreeMap::new();
            for (side, policy) in [("a", policy_a), ("b", policy_b)] {
                if side == "b" && same && continuation.is_some() {
                    let a = scores["a"];
                    scores.insert("b", a);
                    continue;
                }
                if side == reference && continuation.is_none() {
                    let arm = &r.rows[i..i + window.len()];
                    scores.insert(side, (window_scores(arm), arm[0].n_transfers));
                    continue;
                }
                let first = &decisions[side];
                let then = continuation.unwrap_or(policy);
                let (arm, ..) = run_gameweeks(
                    store,
                    rules,
                    |j| if j == 0 { Decider::Decided(first) } else { Decider::Policy(then) },
                    state,
                    window,
                    caches,
                )?;
                if arm.len() != window.len() {
                    bail!("{}: arm stopped early", season_label(*season as i32));
                }
                scores.insert(side, (window_scores(&arm), arm[0].n_transfers));
            }
            let ((points_a, xg_a), n_a) = scores["a"];
            let ((points_b, xg_b), n_b) = scores["b"];
            rows.push(PerDecision {
                season: *season,
                start_id: start_id.clone(),
                policy_a: name_a.clone(),
                policy_b: name_b.clone(),
                reference: reference.to_owned(),
                continuation: cont_name.clone(),
                gw: window[0].gw,
                gw_index: window[0].gw_index,
                k: window.len() as i64,
                stride: stride as i64,
                same_decision: same,
                transfers_a: n_a,
                transfers_b: n_b,
                points_a,
                points_b,
                diff: points_a - points_b,
                xg_a,
                xg_b,
                diff_xg: match (xg_a, xg_b) {
                    (Some(x), Some(y)) => Some(x - y),
                    _ => None,
                },
            });
        }
        log::info!(
            "{} {start_id} per-decision {name_a} vs {name_b}: {} decisions, {} differ",
            season_label(*season as i32),
            rows.len(),
            rows.iter().filter(|r| !r.same_decision).count()
        );
        Ok(rows)
    })?;
    Ok(results.into_iter().flatten().collect())
}

// --- summary --------------------------------------------------------------------------------

/// A JSON number as pandas' `to_json` writes it (10 decimals) read back by `json.loads`;
/// NaN as null.
pub fn num(x: f64) -> Ordered {
    if !x.is_finite() {
        return Ordered::Null;
    }
    let rounded: f64 = format!("{x:.10}").parse().unwrap_or(x);
    Ordered::Float(rounded)
}

pub fn opt(x: Option<f64>) -> Ordered {
    x.map_or(Ordered::Null, num)
}

fn int(x: i64) -> Ordered {
    Ordered::Int(i128::from(x))
}

fn text(s: &str) -> Ordered {
    Ordered::Str(s.to_owned())
}

pub fn record(pairs: Vec<(&str, Ordered)>) -> Ordered {
    Ordered::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}

/// pandas' mean of the non-NaN values (NaN when none).
fn nanmean(values: &[f64]) -> f64 {
    let kept: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    kahan_mean(&kept)
}

fn kahan_mean(values: &[f64]) -> f64 {
    let (mut sum, mut comp) = (0.0, 0.0);
    for v in values {
        let y = v - comp;
        let t = sum + y;
        comp = t - sum - y;
        sum = t;
    }
    sum / values.len() as f64
}

/// Per (season, policy): the means over start states of each run's total, xG total (null if
/// any start has a null GW), GWs, transfers, hit points and regrets.
#[derive(Clone, Debug)]
pub struct SeasonTotal {
    pub season: i64,
    pub policy: String,
    pub n_starts: usize,
    pub total: f64,
    pub xg_total: f64,
    pub n_gws: f64,
    pub n_transfers: f64,
    pub hit_points: f64,
    pub captain_regret: f64,
    pub xi_regret: f64,
}

/// One run's aggregates (season, policy, start).
#[derive(Clone, Debug)]
pub struct RunTotal {
    pub season: i64,
    pub policy: String,
    pub start_id: String,
    pub total: f64,
    pub xg_total: f64,
    pub xg_null: f64,
    pub n_gws: f64,
    pub n_transfers: f64,
    pub hit_points: f64,
    pub captain_regret: f64,
    pub xi_regret: f64,
}

/// Per (season, policy, start_id), sorted.
pub fn run_totals(results: &[GridRow]) -> Vec<RunTotal> {
    let mut groups: BTreeMap<(i64, String, String), Vec<&GwRow>> = BTreeMap::new();
    for r in results {
        groups
            .entry((r.row.season, r.policy.clone(), r.start_id.clone()))
            .or_default()
            .push(&r.row);
    }
    groups
        .into_iter()
        .map(|((season, policy, start_id), rows)| {
            let xg: Option<Vec<f64>> = rows.iter().map(|r| r.xg_net_points).collect();
            let xi: Vec<f64> = rows.iter().filter_map(|r| r.xi_regret).map(|v| v as f64).collect();
            let captain: Vec<f64> = rows.iter().map(|r| r.captain_regret as f64).collect();
            RunTotal {
                season,
                policy,
                start_id,
                total: rows.iter().map(|r| r.net_points).sum::<i64>() as f64,
                xg_total: xg.map_or(f64::NAN, |v| np_sum(&v)),
                xg_null: rows.iter().filter(|r| r.xg_net_points.is_none()).count() as f64,
                n_gws: rows.len() as f64,
                n_transfers: rows.iter().map(|r| r.n_transfers).sum::<i64>() as f64,
                hit_points: rows.iter().map(|r| r.hit_points).sum::<i64>() as f64,
                captain_regret: kahan_mean(&captain),
                xi_regret: if xi.is_empty() { f64::NAN } else { kahan_mean(&xi) },
            }
        })
        .collect()
}

pub fn season_totals(results: &[GridRow]) -> Vec<SeasonTotal> {
    let mut groups: BTreeMap<(i64, String), Vec<RunTotal>> = BTreeMap::new();
    for r in run_totals(results) {
        groups.entry((r.season, r.policy.clone())).or_default().push(r);
    }
    groups
        .into_iter()
        .map(|((season, policy), runs)| {
            let col = |f: fn(&RunTotal) -> f64| runs.iter().map(f).collect::<Vec<f64>>();
            let xg = col(|r| r.xg_total);
            SeasonTotal {
                season,
                policy,
                n_starts: runs.len(),
                total: nanmean(&col(|r| r.total)),
                xg_total: if xg.iter().any(|v| v.is_nan()) { f64::NAN } else { kahan_mean(&xg) },
                n_gws: nanmean(&col(|r| r.n_gws)),
                n_transfers: nanmean(&col(|r| r.n_transfers)),
                hit_points: nanmean(&col(|r| r.hit_points)),
                captain_regret: nanmean(&col(|r| r.captain_regret)),
                xi_regret: nanmean(&col(|r| r.xi_regret)),
            }
        })
        .collect()
}

/// Per policy: the season totals averaged over seasons; the xG total over the seasons where
/// every policy's is defined.
pub fn policy_means(totals: &[SeasonTotal]) -> Ordered {
    let policies: BTreeSet<&String> = totals.iter().map(|t| &t.policy).collect();
    let seasons: BTreeSet<i64> = totals.iter().map(|t| t.season).collect();
    let common: Vec<i64> = seasons
        .iter()
        .copied()
        .filter(|s| {
            policies.iter().all(|p| {
                totals.iter().any(|t| t.season == *s && &&t.policy == p && !t.xg_total.is_nan())
            })
        })
        .collect();
    let rows = policies
        .iter()
        .map(|p| {
            let part: Vec<&SeasonTotal> = totals.iter().filter(|t| &&t.policy == p).collect();
            let col = |f: fn(&SeasonTotal) -> f64| {
                nanmean(&part.iter().map(|t| f(t)).collect::<Vec<_>>())
            };
            let xg: Vec<f64> =
                part.iter().filter(|t| common.contains(&t.season)).map(|t| t.xg_total).collect();
            record(vec![
                ("policy", text(p)),
                ("total", num(col(|t| t.total))),
                ("xg_total", num(if xg.is_empty() { f64::NAN } else { nanmean(&xg) })),
                ("n_gws", num(col(|t| t.n_gws))),
                ("n_transfers", num(col(|t| t.n_transfers))),
                ("hit_points", num(col(|t| t.hit_points))),
                ("captain_regret", num(col(|t| t.captain_regret))),
                ("xi_regret", num(col(|t| t.xi_regret))),
                (
                    "n_seasons",
                    int(part.iter().map(|t| t.season).collect::<BTreeSet<_>>().len() as i64),
                ),
                ("xg_seasons", int(common.len() as i64)),
            ])
        })
        .collect();
    Ordered::Array(rows)
}

pub fn season_totals_json(totals: &[SeasonTotal]) -> Ordered {
    Ordered::Array(
        totals
            .iter()
            .map(|t| {
                record(vec![
                    ("season", int(t.season)),
                    ("policy", text(&t.policy)),
                    ("n_starts", int(t.n_starts as i64)),
                    ("total", num(t.total)),
                    ("xg_total", num(t.xg_total)),
                    ("n_gws", num(t.n_gws)),
                    ("n_transfers", num(t.n_transfers)),
                    ("hit_points", num(t.hit_points)),
                    ("captain_regret", num(t.captain_regret)),
                    ("xi_regret", num(t.xi_regret)),
                ])
            })
            .collect(),
    )
}

/// One transfer-gain summary row.
#[derive(Clone, Debug)]
pub struct GainRow {
    pub policy: String,
    pub n_decisions: usize,
    pub transfers: f64,
    pub hit_points: f64,
    pub pred: f64,
    pub real: f64,
    pub pred_net: f64,
    pub real_net: f64,
    pub slope: f64,
    pub intercept: f64,
}

/// Per policy: GWs after gw_index 1 with transfers, their predicted vs realized gain (gross
/// and net of hits) and the least-squares slope and intercept of realized on predicted.
pub fn transfer_gain_summary(results: &[GridRow]) -> Vec<GainRow> {
    let policies: BTreeSet<&String> = results.iter().map(|r| &r.policy).collect();
    let mean = |v: &[f64]| if v.is_empty() { f64::NAN } else { np_sum(v) / v.len() as f64 };
    policies
        .into_iter()
        .map(|policy| {
            let part: Vec<&GwRow> = results
                .iter()
                .filter(|r| &r.policy == policy)
                .map(|r| &r.row)
                .filter(|r| {
                    r.n_transfers > 0
                        && r.gw_index > 1
                        && r.pred_gain.is_some()
                        && r.real_gain.is_some()
                })
                .collect();
            let pred: Vec<f64> = part.iter().map(|r| r.pred_gain.unwrap()).collect();
            let real: Vec<f64> = part.iter().map(|r| r.real_gain.unwrap()).collect();
            let hits: Vec<f64> = part.iter().map(|r| r.hit_points as f64).collect();
            let transfers: Vec<f64> = part.iter().map(|r| r.n_transfers as f64).collect();
            let (mut slope, mut intercept) = (f64::NAN, f64::NAN);
            if pred.len() >= 2 {
                let (mp, mr) = (mean(&pred), mean(&real));
                let dev: Vec<f64> = pred.iter().map(|p| (p - mp) * (p - mp)).collect();
                let var = mean(&dev);
                if var > 0.0 {
                    let cov: Vec<f64> =
                        pred.iter().zip(&real).map(|(p, r)| (p - mp) * (r - mr)).collect();
                    slope = mean(&cov) / var;
                    intercept = mr - slope * mp;
                }
            }
            let net = |v: &[f64]| v.iter().zip(&hits).map(|(x, h)| x - h).collect::<Vec<f64>>();
            GainRow {
                policy: policy.clone(),
                n_decisions: part.len(),
                transfers: mean(&transfers),
                hit_points: mean(&hits),
                pred: mean(&pred),
                real: mean(&real),
                pred_net: mean(&net(&pred)),
                real_net: mean(&net(&real)),
                slope,
                intercept,
            }
        })
        .collect()
}

pub fn gains_json(gains: &[GainRow]) -> Ordered {
    Ordered::Array(
        gains
            .iter()
            .map(|g| {
                record(vec![
                    ("policy", text(&g.policy)),
                    ("n_decisions", int(g.n_decisions as i64)),
                    ("transfers", num(g.transfers)),
                    ("hit_points", num(g.hit_points)),
                    ("pred", num(g.pred)),
                    ("real", num(g.real)),
                    ("pred_net", num(g.pred_net)),
                    ("real_net", num(g.real_net)),
                    ("slope", num(g.slope)),
                    ("intercept", num(g.intercept)),
                ])
            })
            .collect(),
    )
}

/// PLAN §5 splits: all seasons, develop, validate.
pub fn splits() -> [(&'static str, Option<Vec<i64>>); 3] {
    [
        ("all", None),
        ("develop", Some(DEVELOP_SEASONS.map(i64::from).collect())),
        ("validate", Some(VALIDATE_SEASONS.map(i64::from).collect())),
    ]
}

/// The bootstrap settings and the comparison family's variant count.
#[derive(Clone, Copy, Debug)]
pub struct Bootstrap {
    pub block_length: usize,
    pub n_boot: usize,
    pub seed: u64,
    pub ci: f64,
    pub n_variants: usize,
}

/// A difference frame's rows for one metric: (season, gw_index, value, k, stride).
#[derive(Clone, Debug)]
pub struct DiffRow {
    pub season: i64,
    pub gw_index: i64,
    pub diff: f64,
    pub diff_xg: Option<f64>,
    pub weight: f64,
}

/// One comparison row (Python `_comparison`).
#[derive(Clone, Debug)]
pub struct Comparison {
    pub a: String,
    pub b: String,
    pub method: String,
    pub split: String,
    pub metric: String,
    pub result: BootstrapResult,
    pub season_diff: f64,
    pub p_season_t: f64,
    pub deflated_mean: f64,
}

/// Rows per split and metric (realized, realized on the xG sample, xG-scored).
pub fn comparison(
    rows: &[DiffRow],
    a: &str,
    b: &str,
    method: &str,
    stride: usize,
    bs: &Bootstrap,
) -> Vec<Comparison> {
    let cell_block = bs.block_length.div_ceil(stride);
    let mut out = Vec::new();
    for (split, seasons) in splits() {
        let part: Vec<&DiffRow> = rows
            .iter()
            .filter(|r| seasons.as_ref().is_none_or(|s| s.contains(&r.season)))
            .collect();
        if part.is_empty() {
            continue;
        }
        let realized: Vec<Diff> = part
            .iter()
            .map(|r| Diff {
                season: r.season,
                gw_index: r.gw_index,
                value: Some(r.diff),
                weight: r.weight,
            })
            .collect();
        let on_xg: Vec<Diff> = part
            .iter()
            .filter(|r| r.diff_xg.is_some())
            .map(|r| Diff {
                season: r.season,
                gw_index: r.gw_index,
                value: Some(r.diff),
                weight: r.weight,
            })
            .collect();
        let xg: Vec<Diff> = part
            .iter()
            .map(|r| Diff {
                season: r.season,
                gw_index: r.gw_index,
                value: r.diff_xg,
                weight: r.weight,
            })
            .collect();
        for (metric, frame) in [("realized", realized), ("realized@xg", on_xg), ("xg", xg)] {
            let result = block_bootstrap(&frame, cell_block, bs.n_boot, bs.seed, bs.ci);
            let per_season: Vec<f64> = season_points(&frame).into_values().collect();
            let season_diff = if per_season.is_empty() {
                f64::NAN
            } else {
                np_sum(&per_season) / per_season.len() as f64
            };
            let (p_t, _) = season_t_test(&frame);
            out.push(Comparison {
                a: a.to_owned(),
                b: b.to_owned(),
                method: method.to_owned(),
                split: split.to_owned(),
                metric: metric.to_owned(),
                deflated_mean: deflated_mean(
                    result.mean,
                    result.ci_low,
                    result.ci_high,
                    bs.n_variants,
                    bs.ci,
                ),
                result,
                season_diff,
                p_season_t: p_t,
            });
        }
    }
    out
}

pub fn comparisons_json(rows: &[Comparison]) -> Ordered {
    Ordered::Array(
        rows.iter()
            .map(|c| {
                record(vec![
                    ("a", text(&c.a)),
                    ("b", text(&c.b)),
                    ("method", text(&c.method)),
                    ("split", text(&c.split)),
                    ("metric", text(&c.metric)),
                    ("mean", num(c.result.mean)),
                    ("ci_low", num(c.result.ci_low)),
                    ("ci_high", num(c.result.ci_high)),
                    ("p_one_sided", num(c.result.p_one_sided)),
                    ("n_seasons", int(c.result.n_seasons as i64)),
                    ("n_gws", int(c.result.n_gws as i64)),
                    ("season_diff", num(c.season_diff)),
                    ("p_season_t", num(c.p_season_t)),
                    ("deflated_mean", num(c.deflated_mean)),
                ])
            })
            .collect(),
    )
}

/// The full-run difference rows of a paired frame.
pub fn paired_rows(paired: &[Paired]) -> Vec<DiffRow> {
    paired
        .iter()
        .map(|p| DiffRow {
            season: p.season,
            gw_index: p.gw_index,
            diff: p.diff as f64,
            diff_xg: p.diff_xg,
            weight: 1.0,
        })
        .collect()
}

/// The per-decision difference rows (overlapping windows weighted stride/k).
pub fn decision_rows(rows: &[PerDecision]) -> Vec<DiffRow> {
    rows.iter()
        .map(|r| DiffRow {
            season: r.season,
            gw_index: r.gw_index,
            diff: r.diff as f64,
            diff_xg: r.diff_xg,
            weight: (r.stride as f64 / r.k as f64).min(1.0),
        })
        .collect()
}

/// The summary: season totals, per-policy means, comparisons and transfer gains.
pub struct Summary {
    pub season_totals: Vec<SeasonTotal>,
    pub comparisons: Vec<Comparison>,
    pub gains: Vec<GainRow>,
}

impl Summary {
    pub fn to_json(&self) -> Ordered {
        record(vec![
            ("season_totals", season_totals_json(&self.season_totals)),
            ("policies", policy_means(&self.season_totals)),
            ("comparisons", comparisons_json(&self.comparisons)),
            ("transfer_gains", gains_json(&self.gains)),
        ])
    }

    /// The experiment log's metrics: per-policy means, season totals, comparisons, gains.
    pub fn metrics(&self) -> Ordered {
        let totals = self
            .season_totals
            .iter()
            .map(|t| {
                let key = format!("{} {}", season_label(t.season as i32), t.policy);
                (key, num((t.total * 100.0).round() / 100.0))
            })
            .collect();
        record(vec![
            ("policies", policy_means(&self.season_totals)),
            ("season_totals", Ordered::Object(totals)),
            ("comparisons", comparisons_json(&self.comparisons)),
            ("transfer_gains", gains_json(&self.gains)),
        ])
    }
}

/// Season totals per policy, the full-run comparison of each pair and each per-decision
/// frame's, and the transfer gains.
pub fn summarize(
    results: &[GridRow],
    pairs: &[(String, String)],
    decisions: Option<&[PerDecision]>,
    bs: &Bootstrap,
) -> Result<Summary> {
    let mut comparisons = Vec::new();
    for (a, b) in pairs {
        let paired = paired_full_run(results, a, b)?;
        comparisons.extend(comparison(&paired_rows(&paired), a, b, "full_run", 1, bs));
    }
    if let Some(rows) = decisions
        && !rows.is_empty()
    {
        let mut groups: BTreeMap<(String, String), Vec<PerDecision>> = BTreeMap::new();
        for r in rows {
            groups.entry((r.policy_a.clone(), r.policy_b.clone())).or_default().push(r.clone());
        }
        for ((a, b), part) in groups {
            let strides: BTreeSet<i64> = part.iter().map(|r| r.stride).collect();
            if strides.len() != 1 {
                bail!("per-decision frame mixes strides {strides:?}");
            }
            let stride = *strides.iter().next().unwrap() as usize;
            comparisons.extend(comparison(
                &decision_rows(&part),
                &a,
                &b,
                "per_decision",
                stride,
                bs,
            ));
        }
    }
    Ok(Summary {
        season_totals: season_totals(results),
        comparisons,
        gains: transfer_gain_summary(results),
    })
}
