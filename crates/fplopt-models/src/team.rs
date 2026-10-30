//! Team model (Python `fplopt.models.team`): per-fixture expected goals and clean-sheet
//! probabilities.
//!
//! **Market λ** (fixtures with pre-match odds visible at the deadline): per fixture the
//! newest visible pre-match snapshot (football-data's `avg` when its h2h is complete, else
//! the newest Odds API snapshot), each bookmaker de-vigged on its own (power method or
//! Shin), the median per outcome renormalized; λ_home, λ_away by MINPACK Levenberg–Marquardt
//! on log λ (`optim::lmder`, fixed start, fixed tolerances, analytic Jacobian) of the
//! Dixon-Coles score distribution against the 1X2, O/U 2.5 and Asian-handicap probabilities;
//! a solution outside `LAMBDA_RANGE` counts as not converged.
//!
//! **Ratings** (fixtures without visible odds): `log λ_home = base + home + attack[home] −
//! defence[away]`, `log λ_away = base + attack[away] − defence[home]`, attack/defence = slope
//! · Elo z + a ridge-penalized deviation; time-decayed weighted quasi-Poisson MLE
//! (`optim::trust_exact`) on every finished match visible, targets blending the match's
//! market λ with xG and goals. **P(CS)** under Dixon-Coles with `RHO`.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use anyhow::{Result, bail};
use fplopt_features::baseline::upcoming_fixtures;
use fplopt_features::util::median;
use fplopt_store::{AsOfView, Column, Micros, Table};

use crate::optim::{Mat, lmder, trust_exact};
use crate::util::{np_sum, opt_ts, pd_mean};

pub const MAX_GOALS: usize = 10;
pub const RHO: f64 = -0.03;
pub const TOTALS_LINE: f64 = 2.5;
pub const LAMBDA_RANGE: (f64, f64) = (0.05, 8.0);
pub const MARKET_START: (f64, f64) = (1.4, 1.1);
const WEIGHT_1X2: f64 = 1.0;
const WEIGHT_TOTALS: f64 = 1.0;
const WEIGHT_AH: f64 = 1.0;
const LSQ_TOL: f64 = 1e-12;
pub const XG_WEIGHT: f64 = 0.7;
pub const ELO_SCALE: f64 = 100.0;
pub const MIN_WEIGHT: f64 = 0.01;
const MAX_WINDOW_DAYS: f64 = 365.25 * 50.0;
const FIT_GTOL: f64 = 1e-6;
const FIT_MAXITER: usize = 200;
const BISECTION_STEPS: usize = 100;
const OUTCOMES_1X2: [&str; 3] = ["home", "draw", "away"];

#[derive(Clone, Debug, PartialEq)]
pub struct TeamParams {
    pub half_life_days: f64,
    pub market_weight: f64,
    pub prior_strength: f64,
    pub rho: f64,
    pub devig: Devig,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Devig {
    Power,
    Shin,
}

impl Default for TeamParams {
    fn default() -> Self {
        TeamParams {
            half_life_days: 45.0,
            market_weight: 0.75,
            prior_strength: 2.0,
            rho: RHO,
            devig: Devig::Power,
        }
    }
}

/// Ratings fitted at `cutoff`: `teams` sorted, `attack` / `defence` aligned with it (Elo
/// prior included); `source` 'ratings' when market λ entered the targets, else 'stats'.
#[derive(Clone, Debug)]
pub struct TeamFit {
    pub cutoff: Micros,
    pub params: TeamParams,
    pub source: &'static str,
    pub base: f64,
    pub home: f64,
    pub attack_slope: f64,
    pub defence_slope: f64,
    pub elo_center: f64,
    pub teams: Vec<i64>,
    pub attack: Vec<f64>,
    pub defence: Vec<f64>,
    pub n_matches: usize,
    pub n_market: usize,
}

// --- de-vig -------------------------------------------------------------------------------

/// (1/prices, valid); a row is valid when every price is finite and > 0 (an invalid row's
/// prices count as 2, as Python's).
fn implied(prices: &[f64]) -> (Vec<f64>, bool) {
    let valid = prices.iter().all(|p| p.is_finite() && *p > 0.0);
    (prices.iter().map(|&p| 1.0 / if valid { p } else { 2.0 }).collect(), valid)
}

fn devig_power(prices: &[f64]) -> Vec<f64> {
    let (implied, valid) = implied(prices);
    if !valid {
        return vec![f64::NAN; prices.len()];
    }
    let (mut lo, mut hi) = (1e-6f64, 50.0f64);
    for _ in 0..BISECTION_STEPS {
        let mid = (lo + hi) / 2.0;
        let powered: Vec<f64> = implied.iter().map(|p| p.powf(mid)).collect();
        if np_sum(&powered) > 1.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let k = (lo + hi) / 2.0;
    let out: Vec<f64> = implied.iter().map(|p| p.powf(k)).collect();
    let total = np_sum(&out);
    out.iter().map(|p| p / total).collect()
}

fn shin(implied: &[f64], z: f64) -> Vec<f64> {
    let beta = np_sum(implied);
    implied
        .iter()
        .map(|p| ((z * z + 4.0 * (1.0 - z) * p * p / beta).sqrt() - z) / (2.0 * (1.0 - z)))
        .collect()
}

fn devig_shin(prices: &[f64]) -> Vec<f64> {
    let (implied, valid) = implied(prices);
    if !valid {
        return vec![f64::NAN; prices.len()];
    }
    let margin = np_sum(&implied) > 1.0;
    let (mut lo, mut hi) = (0.0f64, 1.0 - 1e-9);
    for _ in 0..BISECTION_STEPS {
        let mid = (lo + hi) / 2.0;
        if np_sum(&shin(&implied, mid)) > 1.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let out = if margin { shin(&implied, (lo + hi) / 2.0) } else { implied };
    let total = np_sum(&out);
    out.iter().map(|p| p / total).collect()
}

fn devig(prices: &[f64], method: Devig) -> Vec<f64> {
    match method {
        Devig::Power => devig_power(prices),
        Devig::Shin => devig_shin(prices),
    }
}

// --- Dixon-Coles ------------------------------------------------------------------------

/// ln k! for k = 0..=MAX_GOALS (scipy's `gammaln(k + 1)` is `log` of the exact factorial).
fn log_factorials() -> [f64; MAX_GOALS + 1] {
    let mut out = [0.0; MAX_GOALS + 1];
    let mut factorial = 1.0f64;
    for (k, slot) in out.iter_mut().enumerate() {
        if k > 0 {
            factorial *= k as f64;
        }
        *slot = factorial.ln();
    }
    out
}

/// The score matrix [home goals][away goals] (renormalized; negative cells clipped).
pub fn dc_matrix(
    lambda_home: f64,
    lambda_away: f64,
    rho: f64,
) -> [[f64; MAX_GOALS + 1]; MAX_GOALS + 1] {
    let log_fact = log_factorials();
    let (log_h, log_a) = (lambda_home.ln(), lambda_away.ln());
    let mut m = [[0.0; MAX_GOALS + 1]; MAX_GOALS + 1];
    let ph: Vec<f64> =
        (0..=MAX_GOALS).map(|g| (g as f64 * log_h - lambda_home - log_fact[g]).exp()).collect();
    let pa: Vec<f64> =
        (0..=MAX_GOALS).map(|g| (g as f64 * log_a - lambda_away - log_fact[g]).exp()).collect();
    for i in 0..=MAX_GOALS {
        for j in 0..=MAX_GOALS {
            m[i][j] = ph[i] * pa[j];
        }
    }
    m[0][0] *= 1.0 - lambda_home * lambda_away * rho;
    m[0][1] *= 1.0 + lambda_home * rho;
    m[1][0] *= 1.0 + lambda_away * rho;
    m[1][1] *= 1.0 - rho;
    let mut flat = Vec::with_capacity((MAX_GOALS + 1) * (MAX_GOALS + 1));
    for row in m.iter_mut() {
        for cell in row.iter_mut() {
            *cell = cell.max(0.0);
            flat.push(*cell);
        }
    }
    let total = np_sum(&flat);
    for row in m.iter_mut() {
        for cell in row.iter_mut() {
            *cell /= total;
        }
    }
    m
}

/// (won, lost) share of a home Asian-handicap stake per score, for home handicap `line`.
fn ah_stakes(line: f64) -> Result<(Vec<f64>, Vec<f64>)> {
    if (line * 4.0).rem_euclid(1.0) != 0.0 {
        bail!("Asian handicap lines are multiples of 0.25, got {line}");
    }
    let parts: Vec<f64> = if (line * 4.0).rem_euclid(2.0) == 1.0 {
        vec![line - 0.25, line + 0.25]
    } else {
        vec![line]
    };
    let share = 1.0 / parts.len() as f64;
    let size = MAX_GOALS + 1;
    let (mut won, mut lost) = (vec![0.0; size * size], vec![0.0; size * size]);
    for i in 0..size {
        for j in 0..size {
            let diff = i as f64 - j as f64;
            // Python: sum(share * (diff + part > 0) for part in parts), from 0.
            let (mut w, mut l) = (0.0, 0.0);
            for part in &parts {
                w += share * f64::from(u8::from(diff + part > 0.0));
                l += share * f64::from(u8::from(diff + part < 0.0));
            }
            won[i * size + j] = w;
            lost[i * size + j] = l;
        }
    }
    Ok((won, lost))
}

/// Unnormalized DC matrix B (flattened) at (log λ_home, log λ_away) = x and dB/dx.
fn dc_with_gradient(x: [f64; 2], rho: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (lh, la) = (x[0].exp(), x[1].exp());
    let log_fact = log_factorials();
    let size = MAX_GOALS + 1;
    let ph: Vec<f64> = (0..size).map(|g| (g as f64 * x[0] - lh - log_fact[g]).exp()).collect();
    let pa: Vec<f64> = (0..size).map(|g| (g as f64 * x[1] - la - log_fact[g]).exp()).collect();
    let mut base = vec![0.0; size * size];
    for i in 0..size {
        for j in 0..size {
            base[i * size + j] = ph[i] * pa[j];
        }
    }
    let mut tau = vec![1.0; size * size];
    tau[0] = 1.0 - lh * la * rho;
    tau[1] = 1.0 + lh * rho;
    tau[size] = 1.0 + la * rho;
    tau[size + 1] = 1.0 - rho;
    let b: Vec<f64> = base.iter().zip(&tau).map(|(b, t)| b * t).collect();
    let mut db_h = vec![0.0; size * size];
    let mut db_a = vec![0.0; size * size];
    for i in 0..size {
        for j in 0..size {
            db_h[i * size + j] = b[i * size + j] * (i as f64 - lh);
            db_a[i * size + j] = b[i * size + j] * (j as f64 - la);
        }
    }
    db_h[0] += base[0] * (-lh * la * rho);
    db_h[1] += base[1] * (lh * rho);
    db_a[0] += base[0] * (-lh * la * rho);
    db_a[size] += base[size] * (la * rho);
    (b, db_h, db_a)
}

/// Least-squares (λ_home, λ_away, converged) for one fixture: the linear events (masks over
/// the flattened score matrix) and optionally the AH pair against `target`.
fn solve_market(
    target: &[f64],
    events: &[Vec<f64>],
    ah: Option<&(Vec<f64>, Vec<f64>)>,
    weights: &[f64],
    rho: f64,
) -> Result<(f64, f64, bool)> {
    let sqrt_w: Vec<f64> = weights.iter().map(|w| w.sqrt()).collect();
    let (log_lo, log_hi) = (LAMBDA_RANGE.0.ln(), LAMBDA_RANGE.1.ln());
    let n_events = events.len();
    let mut masks: Vec<&[f64]> = events.iter().map(Vec::as_slice).collect();
    let ones = vec![1.0; (MAX_GOALS + 1) * (MAX_GOALS + 1)];
    masks.push(&ones);
    if let Some((won, lost)) = ah {
        masks.push(won);
        masks.push(lost);
    }
    let dot = |a: &[f64], b: &[f64]| -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
    let model = |x: &[f64]| -> (Vec<f64>, Mat) {
        let clipped = [x[0].clamp(log_lo, log_hi), x[1].clamp(log_lo, log_hi)];
        let (b, db_h, db_a) = dc_with_gradient(clipped, rho);
        let s: Vec<f64> = masks.iter().map(|m| dot(m, &b)).collect();
        let ds: Vec<[f64; 2]> = masks.iter().map(|m| [dot(m, &db_h), dot(m, &db_a)]).collect();
        let (z, dz) = (s[n_events], ds[n_events]);
        let mut p: Vec<f64> = s[..n_events].iter().map(|v| v / z).collect();
        let mut dp: Vec<[f64; 2]> = (0..n_events)
            .map(|e| [(ds[e][0] - p[e] * dz[0]) / z, (ds[e][1] - p[e] * dz[1]) / z])
            .collect();
        if ah.is_some() {
            let (w, lo) = (s[n_events + 1], s[n_events + 2]);
            let (dw, dl) = (ds[n_events + 1], ds[n_events + 2]);
            let q = w / (w + lo);
            let denom = (w + lo).powi(2);
            let dq = [(dw[0] * lo - w * dl[0]) / denom, (dw[1] * lo - w * dl[1]) / denom];
            p.push(q);
            p.push(1.0 - q);
            dp.push(dq);
            dp.push([-dq[0], -dq[1]]);
        }
        let residuals: Vec<f64> = (0..p.len()).map(|k| sqrt_w[k] * (p[k] - target[k])).collect();
        let mut jac = Mat::zeros(p.len(), 2);
        for (k, d) in dp.iter().enumerate() {
            jac.set(k, 0, sqrt_w[k] * d[0]);
            jac.set(k, 1, sqrt_w[k] * d[1]);
        }
        (residuals, jac)
    };
    let start = [MARKET_START.0.ln(), MARKET_START.1.ln()];
    let (x, info) = lmder(model, &start, LSQ_TOL, LSQ_TOL, LSQ_TOL, 200)?;
    let inside = log_lo <= x[0].min(x[1]) && x[0].max(x[1]) <= log_hi;
    let ok = (1..=4).contains(&info) && x.iter().all(|v| v.is_finite()) && inside;
    Ok((x[0].exp(), x[1].exp(), ok))
}

/// One fixture's de-vigged markets (NaN = missing).
#[derive(Clone, Copy, Debug)]
struct Markets {
    fixture_key: i64,
    p_home: f64,
    p_draw: f64,
    p_away: f64,
    p_over: f64,
    ah_line: f64,
    q_ah_home: f64,
}

/// Per fixture (sorted): (fixture_key, λ_home, λ_away, success); NaN without complete 1X2.
fn market_lambdas(markets: &[Markets], rho: f64) -> Result<Vec<(i64, f64, f64, bool)>> {
    let size = MAX_GOALS + 1;
    let grid = |f: &dyn Fn(f64, f64) -> bool| -> Vec<f64> {
        let mut out = vec![0.0; size * size];
        for i in 0..size {
            for j in 0..size {
                let (diff, total) = (i as f64 - j as f64, i as f64 + j as f64);
                out[i * size + j] = if f(diff, total) { 1.0 } else { 0.0 };
            }
        }
        out
    };
    let win = grid(&|d, _| d > 0.0);
    let draw = grid(&|d, _| d == 0.0);
    let loss = grid(&|d, _| d < 0.0);
    let over = grid(&|_, t| t > TOTALS_LINE);
    let under: Vec<f64> = over.iter().map(|o| 1.0 - o).collect();
    let mut rows = markets.to_vec();
    rows.sort_by_key(|m| m.fixture_key);
    let mut out = Vec::with_capacity(rows.len());
    for m in rows {
        if ![m.p_home, m.p_draw, m.p_away].iter().all(|p| p.is_finite()) {
            out.push((m.fixture_key, f64::NAN, f64::NAN, false));
            continue;
        }
        let mut events = vec![win.clone(), draw.clone(), loss.clone()];
        let mut target = vec![m.p_home, m.p_draw, m.p_away];
        let mut weights = vec![WEIGHT_1X2; 3];
        if m.p_over.is_finite() {
            events.push(over.clone());
            events.push(under.clone());
            target.push(m.p_over);
            target.push(1.0 - m.p_over);
            weights.extend([WEIGHT_TOTALS; 2]);
        }
        let mut ah = None;
        if m.ah_line.is_finite() && m.q_ah_home.is_finite() {
            ah = Some(ah_stakes(m.ah_line)?);
            target.push(m.q_ah_home);
            target.push(1.0 - m.q_ah_home);
            weights.extend([WEIGHT_AH; 2]);
        }
        let (lh, la, ok) = solve_market(&target, &events, ah.as_ref(), &weights, rho)?;
        out.push((m.fixture_key, lh, la, ok));
    }
    Ok(out)
}

// --- odds ---------------------------------------------------------------------------------

/// One visible pre-match odds row.
#[derive(Clone, Debug)]
struct OddsRow {
    fixture_key: i64,
    bookmaker: String,
    market: String,
    outcome: String,
    line: Option<f64>,
    price: f64,
}

/// Pre-match odds of `fixture_keys` visible in the view, one snapshot per fixture:
/// football-data `avg` (newest snapshot) where its h2h is complete, else the newest Odds API
/// snapshot (every bookmaker in it).
fn visible_odds(view: &AsOfView, fixture_keys: &HashSet<i64>) -> Result<Vec<OddsRow>> {
    let columns = [
        "fixture_key",
        "source",
        "bookmaker",
        "market",
        "outcome",
        "line",
        "price",
        "is_closing",
        "snapshot_at",
    ]; // fmt: skip
    let odds = view.table("odds_snapshot", Some(&columns))?;
    let fixture = odds.i64("fixture_key")?;
    let source = odds.str("source")?;
    let bookmaker = odds.str("bookmaker")?;
    let market = odds.str("market")?;
    let outcome = odds.str("outcome")?;
    let line = odds.opt_f64("line")?;
    let price = odds.opt_f64("price")?;
    let closing = odds.opt_bool("is_closing")?;
    let snapshot = opt_ts(&odds, "snapshot_at")?;
    let rows: Vec<usize> = (0..odds.height())
        .filter(|&i| fixture_keys.contains(&fixture[i]) && closing[i] == Some(false))
        .collect();
    // Newest snapshot per fixture, per source group (NaT never equals the max).
    let newest = |keep: &dyn Fn(usize) -> bool| -> Vec<usize> {
        let mut max: HashMap<i64, Micros> = HashMap::new();
        for &i in rows.iter().filter(|&&i| keep(i)) {
            if let Some(t) = snapshot[i] {
                let entry = max.entry(fixture[i]).or_insert(t);
                *entry = (*entry).max(t);
            }
        }
        rows.iter()
            .copied()
            .filter(|&i| {
                keep(i) && snapshot[i].is_some() && max.get(&fixture[i]) == snapshot[i].as_ref()
            })
            .collect()
    };
    let fd = newest(&|i| source[i] == "football-data" && bookmaker[i] == "avg");
    let mut outcomes: HashMap<i64, HashSet<&str>> = HashMap::new();
    for &i in &fd {
        if market[i] == "h2h" && OUTCOMES_1X2.contains(&outcome[i].as_str()) {
            outcomes.entry(fixture[i]).or_default().insert(outcome[i].as_str());
        }
    }
    let complete: HashSet<i64> =
        outcomes.iter().filter(|(_, o)| o.len() == OUTCOMES_1X2.len()).map(|(k, _)| *k).collect();
    let fd: Vec<usize> = fd.into_iter().filter(|i| complete.contains(&fixture[*i])).collect();
    let api = newest(&|i| source[i] == "odds-api" && !complete.contains(&fixture[i]));
    Ok(fd
        .into_iter()
        .chain(api)
        .map(|i| OddsRow {
            fixture_key: fixture[i],
            bookmaker: bookmaker[i].clone(),
            market: market[i].clone(),
            outcome: outcome[i].clone(),
            line: line[i],
            price: price[i].unwrap_or(f64::NAN),
        })
        .collect())
}

/// A market's key: the fixture and, for 'ah', the line.
type MarketKey = (i64, Option<OrdF64>);

/// Per fixture (and line for 'ah', on the fixture's most quoted line): the median over
/// bookmakers of each outcome's de-vigged probability, renormalized.
fn devig_market(
    odds: &[OddsRow],
    market: &str,
    outcomes: &[&str],
    method: Devig,
) -> BTreeMap<MarketKey, Vec<f64>> {
    let is_ah = market == "ah";
    // pivot_table(aggfunc="median"): per (fixture, bookmaker[, line]) and outcome.
    let mut cells: BTreeMap<(i64, String, Option<OrdF64>), Vec<Vec<f64>>> = BTreeMap::new();
    for row in odds.iter().filter(|r| r.market == market) {
        let Some(k) = outcomes.iter().position(|o| *o == row.outcome) else { continue };
        let line = if is_ah {
            match row.line {
                Some(l) if !l.is_nan() => Some(OrdF64(l)),
                _ => continue, // groupby drops a null key
            }
        } else {
            None
        };
        let entry = cells
            .entry((row.fixture_key, row.bookmaker.clone(), line))
            .or_insert_with(|| vec![Vec::new(); outcomes.len()]);
        if !row.price.is_nan() {
            entry[k].push(row.price);
        }
    }
    let mut probabilities: Vec<(MarketKey, Vec<f64>)> = Vec::new();
    for ((fixture, _, line), mut prices) in cells {
        if prices.iter().any(Vec::is_empty) {
            continue;
        }
        let row: Vec<f64> = prices.iter_mut().map(|p| median(p)).collect();
        probabilities.push(((fixture, line), devig(&row, method)));
    }
    if is_ah {
        let mut counts: BTreeMap<(i64, OrdF64), usize> = BTreeMap::new();
        for ((fixture, line), _) in &probabilities {
            *counts.entry((*fixture, line.unwrap())).or_default() += 1;
        }
        let mut best: BTreeMap<i64, (usize, OrdF64)> = BTreeMap::new();
        for (&(fixture, line), &n) in &counts {
            let better = match best.get(&fixture) {
                None => true,
                Some(&(bn, bl)) => {
                    n > bn
                        || (n == bn
                            && (line.0.abs() < bl.0.abs()
                                || (line.0.abs() == bl.0.abs() && line.0 < bl.0)))
                }
            };
            if better {
                best.insert(fixture, (n, line));
            }
        }
        probabilities.retain(|((fixture, line), _)| best.get(fixture).map(|b| b.1) == *line);
    }
    let mut grouped: BTreeMap<MarketKey, Vec<Vec<f64>>> = BTreeMap::new();
    for (key, p) in probabilities {
        let entry = grouped.entry(key).or_insert_with(|| vec![Vec::new(); outcomes.len()]);
        for (k, v) in p.into_iter().enumerate() {
            if !v.is_nan() {
                entry[k].push(v);
            }
        }
    }
    grouped
        .into_iter()
        .map(|(key, mut values)| {
            let medians: Vec<f64> = values
                .iter_mut()
                .map(|v| if v.is_empty() { f64::NAN } else { median(v) })
                .collect();
            let present: Vec<f64> = medians.iter().copied().filter(|v| !v.is_nan()).collect();
            let total = np_sum(&present);
            (key, medians.iter().map(|m| m / total).collect())
        })
        .collect()
}

/// A float with a total order (for line keys).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrdF64(pub f64);

impl Eq for OrdF64 {}

impl PartialOrd for OrdF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Per fixture with complete 1X2: the de-vigged markets (NaN where missing), sorted.
fn market_probabilities(odds: &[OddsRow], method: Devig) -> Vec<Markets> {
    let h2h = devig_market(odds, "h2h", &OUTCOMES_1X2, method);
    let totals_rows: Vec<OddsRow> = odds
        .iter()
        .filter(|r| r.market != "totals" || r.line == Some(TOTALS_LINE))
        .cloned()
        .collect();
    let totals = devig_market(&totals_rows, "totals", &["over", "under"], method);
    let ah = devig_market(odds, "ah", &["home", "away"], method);
    let ah: HashMap<i64, (f64, f64)> =
        ah.into_iter().map(|((fixture, line), p)| (fixture, (line.unwrap().0, p[0]))).collect();
    h2h.into_iter()
        .map(|((fixture, _), p)| {
            let p_over = totals.get(&(fixture, None)).map_or(f64::NAN, |t| t[0]);
            let (ah_line, q_ah_home) = ah.get(&fixture).copied().unwrap_or((f64::NAN, f64::NAN));
            Markets {
                fixture_key: fixture,
                p_home: p[0],
                p_draw: p[1],
                p_away: p[2],
                p_over,
                ah_line,
                q_ah_home,
            }
        })
        .collect()
}

/// fixture_key → (λ_home, λ_away) for fixtures with visible odds whose solve converged.
fn visible_market_lambdas(
    view: &AsOfView,
    fixture_keys: &HashSet<i64>,
    params: &TeamParams,
) -> Result<BTreeMap<i64, (f64, f64)>> {
    let odds = visible_odds(view, fixture_keys)?;
    if odds.is_empty() {
        return Ok(BTreeMap::new());
    }
    let solved = market_lambdas(&market_probabilities(&odds, params.devig), params.rho)?;
    let failed: Vec<i64> = solved.iter().filter(|s| !s.3 && !s.1.is_nan()).map(|s| s.0).collect();
    if !failed.is_empty() {
        log::warn!(
            "team model at {}: market λ did not converge for fixture(s) {failed:?}; ratings used",
            view.deadline()
        );
    }
    Ok(solved.into_iter().filter(|s| s.3).map(|s| (s.0, (s.1, s.2))).collect())
}

// --- ratings ------------------------------------------------------------------------------

/// team_key → the Elo of the club's newest visible `team_rating` row.
fn elo(view: &AsOfView) -> Result<HashMap<i64, f64>> {
    let ratings = view.table("team_rating", Some(&["team_key", "event_time", "rating_after"]))?;
    let team = ratings.i64("team_key")?;
    let time = opt_ts(&ratings, "event_time")?;
    let rating = ratings.opt_f64("rating_after")?;
    let mut order: Vec<usize> = (0..ratings.height()).collect();
    order.sort_by_key(|&i| (team[i], time[i].is_none(), time[i]));
    let mut out = HashMap::new();
    for i in order {
        out.insert(team[i], rating[i].unwrap_or(f64::NAN));
    }
    Ok(out)
}

/// Clubs of the newest season with a visible schedule (sorted; empty without one).
fn current_teams(view: &AsOfView) -> Result<Vec<i64>> {
    let gameweeks = view.table("gameweek", Some(&["season"]))?;
    let Some(&season) = gameweeks.i64("season")?.iter().max() else { return Ok(Vec::new()) };
    let schedule = view.schedule(season)?;
    let mut teams: BTreeSet<i64> = schedule.i64("home_team_key")?.iter().copied().collect();
    teams.extend(schedule.i64("away_team_key")?.iter().copied());
    Ok(teams.into_iter().collect())
}

/// One finished match (`match_history`).
#[derive(Clone, Debug)]
struct Match {
    fixture_key: i64,
    kickoff: Option<Micros>,
    available_at: Option<Micros>,
    home: i64,
    away: i64,
    home_goals: f64,
    away_goals: f64,
    home_xg: f64,
    away_xg: f64,
    home_market: f64,
    away_market: f64,
}

/// One row per finished match visible in the view (sorted by fixture_key), within the
/// look-back window, with the market λ of its pre-match odds (NaN without).
fn match_history(view: &AsOfView, params: &TeamParams) -> Result<Vec<Match>> {
    let columns = [
        "fixture_key",
        "team_key",
        "is_home",
        "goals_for",
        "us_xg",
        "fpl_xg",
        "fd_xg",
        "event_time",
        "available_at",
    ]; // fmt: skip
    let sides = view.table("team_match", Some(&columns))?;
    let fixture = sides.i64("fixture_key")?;
    let team = sides.i64("team_key")?;
    let is_home = sides.opt_bool("is_home")?;
    let goals = sides.opt_f64("goals_for")?;
    let sources = [sides.opt_f64("us_xg")?, sides.opt_f64("fpl_xg")?, sides.opt_f64("fd_xg")?];
    let event = opt_ts(&sides, "event_time")?;
    let available = opt_ts(&sides, "available_at")?;
    let xg = |i: usize| -> f64 {
        sources.iter().find_map(|s| s[i].filter(|v| !v.is_nan())).unwrap_or(f64::NAN)
    };
    // set_index + .loc: the last row per (fixture, side) wins only if keys repeat (they
    // don't: team_match is keyed by fixture and team).
    let mut home: BTreeMap<i64, usize> = BTreeMap::new();
    let mut away: BTreeMap<i64, usize> = BTreeMap::new();
    for i in 0..sides.height() {
        match is_home[i] {
            Some(true) => home.insert(fixture[i], i),
            Some(false) => away.insert(fixture[i], i),
            None => None,
        };
    }
    let window = (params.half_life_days * (1.0 / MIN_WEIGHT).log2()).min(MAX_WINDOW_DAYS);
    let oldest = view.deadline() - (window * 86_400_000_000.0).round() as i64;
    let mut matches = Vec::new();
    for (&key, &h) in &home {
        let Some(&a) = away.get(&key) else { continue };
        let kickoff = event[h];
        // NaT compares False: dropped by the window filter.
        if !kickoff.is_some_and(|k| k >= oldest) {
            continue;
        }
        matches.push(Match {
            fixture_key: key,
            kickoff,
            // np.maximum propagates NaT.
            available_at: available[h].zip(available[a]).map(|(x, y)| x.max(y)),
            home: team[h],
            away: team[a],
            home_goals: goals[h].unwrap_or(f64::NAN),
            away_goals: goals[a].unwrap_or(f64::NAN),
            home_xg: xg(h),
            away_xg: xg(a),
            home_market: f64::NAN,
            away_market: f64::NAN,
        });
    }
    let keys: HashSet<i64> = matches.iter().map(|m| m.fixture_key).collect();
    let market = visible_market_lambdas(view, &keys, params)?;
    for m in &mut matches {
        if let Some(&(lh, la)) = market.get(&m.fixture_key) {
            m.home_market = lh;
            m.away_market = la;
        }
    }
    Ok(matches)
}

/// Per match the home and away targets: w · market + (1 − w) · stats (stats alone without
/// a market λ).
fn targets(matches: &[Match], market_weight: f64) -> (Vec<f64>, Vec<f64>) {
    let side = |goals: f64, xg: f64, market: f64| -> f64 {
        let stats = if xg.is_nan() { goals } else { XG_WEIGHT * xg + (1.0 - XG_WEIGHT) * goals };
        if market.is_nan() { stats } else { market_weight * market + (1.0 - market_weight) * stats }
    };
    (
        matches.iter().map(|m| side(m.home_goals, m.home_xg, m.home_market)).collect(),
        matches.iter().map(|m| side(m.away_goals, m.away_xg, m.away_market)).collect(),
    )
}

/// The ratings fit on `matches` available before `cutoff`, with Elo priors.
fn fit_ratings(
    matches: &[Match],
    elo: &HashMap<i64, f64>,
    teams: &[i64],
    cutoff: Micros,
    params: &TeamParams,
) -> Result<TeamFit> {
    let mut rows: Vec<&Match> =
        matches.iter().filter(|m| m.available_at.is_some_and(|a| a < cutoff)).collect();
    rows.sort_by_key(|m| m.fixture_key);
    let weight_of = |m: &Match| -> f64 {
        // (cutoff − kickoff).total_seconds() / 86400 on nanoseconds.
        let age = match m.kickoff {
            Some(k) => ((cutoff - k) * 1000) as f64 / 1e9 / 86400.0,
            None => f64::NAN,
        };
        0.5f64.powf(age / params.half_life_days)
    };
    let weighted: Vec<(&Match, f64)> =
        rows.into_iter().map(|m| (m, weight_of(m))).filter(|(_, w)| *w >= MIN_WEIGHT).collect();
    let rows: Vec<Match> = weighted.iter().map(|(m, _)| (*m).clone()).collect();
    let weight: Vec<f64> = weighted.iter().map(|(_, w)| *w).collect();
    let has_market = rows.iter().any(|m| !m.home_market.is_nan());
    let source = if has_market { "ratings" } else { "stats" };
    let (y_home, y_away) = targets(&rows, if has_market { params.market_weight } else { 0.0 });

    let mut all: BTreeSet<i64> = teams.iter().copied().collect();
    for m in &rows {
        all.insert(m.home);
        all.insert(m.away);
    }
    let all_teams: Vec<i64> = all.into_iter().collect();
    let n = all_teams.len();
    let index: HashMap<i64, usize> = all_teams.iter().enumerate().map(|(i, &t)| (t, i)).collect();
    let mut current: Vec<i64> = teams.iter().copied().filter(|t| elo.contains_key(t)).collect();
    if current.is_empty() {
        current = all_teams.iter().copied().filter(|t| elo.contains_key(t)).collect();
    }
    let center = if current.is_empty() {
        0.0
    } else {
        pd_mean(&current.iter().map(|t| elo[t]).collect::<Vec<_>>())
    };
    let z: Vec<f64> = all_teams
        .iter()
        .map(|t| {
            let v = (elo.get(t).copied().unwrap_or(center) - center) / ELO_SCALE;
            if v.is_nan() { 0.0 } else { v }
        })
        .collect();

    // Observations: home sides then away sides; x = [base, home, attack_slope,
    // defence_slope, u (n), v (n)].
    let k = rows.len();
    let mut team = Vec::with_capacity(2 * k);
    let mut opp = Vec::with_capacity(2 * k);
    for m in &rows {
        team.push(index[&m.home]);
        opp.push(index[&m.away]);
    }
    for m in &rows {
        team.push(index[&m.away]);
        opp.push(index[&m.home]);
    }
    let obs = team.len();
    let is_home = |r: usize| if r < k { 1.0 } else { 0.0 };
    let y: Vec<f64> = y_home.iter().chain(&y_away).copied().collect();
    let w_raw: Vec<f64> = weight.iter().chain(&weight).copied().collect();
    let scale = 1.0 / np_sum(&w_raw).max(1.0);
    let w: Vec<f64> = w_raw.iter().map(|v| v * scale).collect();
    let dim = 4 + 2 * n;
    let ridge: Vec<f64> =
        (0..dim).map(|j| if j < 4 { 0.0 } else { params.prior_strength } * scale).collect();
    // Each observation's design row: (column, value) pairs.
    let design = |r: usize| -> [(usize, f64); 6] {
        [
            (0, 1.0),
            (1, is_home(r)),
            (2, z[team[r]]),
            (3, -z[opp[r]]),
            (4 + team[r], 1.0),
            (4 + n + opp[r], -1.0),
        ]
    };
    let eta = |x: &[f64]| -> Vec<f64> {
        (0..obs).map(|r| design(r).iter().map(|&(c, v)| v * x[c]).sum()).collect()
    };
    let objective = |x: &[f64]| -> (f64, Vec<f64>) {
        let e = eta(x);
        let mu: Vec<f64> = e.iter().map(|v| v.exp()).collect();
        let loss_terms: Vec<f64> = (0..obs).map(|r| w[r] * (mu[r] - y[r] * e[r])).collect();
        let penalty: f64 = (0..dim).map(|j| ridge[j] * x[j] * x[j]).sum();
        let loss = loss_terms.iter().sum::<f64>() + 0.5 * penalty;
        let mut grad: Vec<f64> = (0..dim).map(|j| ridge[j] * x[j]).collect();
        let mut g = vec![0.0; dim];
        for r in 0..obs {
            let residual = w[r] * (mu[r] - y[r]);
            for (c, v) in design(r) {
                g[c] += v * residual;
            }
        }
        for j in 0..dim {
            grad[j] += g[j];
        }
        (loss, grad)
    };
    let hessian = |x: &[f64]| -> Mat {
        let e = eta(x);
        let mut h = Mat::zeros(dim, dim);
        for r in 0..obs {
            let weight = w[r] * e[r].exp();
            let row = design(r);
            for &(a, va) in &row {
                for &(b, vb) in &row {
                    h.add(a, b, va * weight * vb);
                }
            }
        }
        for j in 0..dim {
            h.add(j, j, ridge[j]);
        }
        h
    };
    let mut x0 = vec![0.0; dim];
    x0[0] = 1.35f64.ln();
    let result = trust_exact(objective, hessian, x0, FIT_GTOL, Some(FIT_MAXITER))?;
    if !result.success {
        log::warn!("team ratings fit at {cutoff}: {}", result.message);
    }
    let x = result.x;
    Ok(TeamFit {
        cutoff,
        params: params.clone(),
        source,
        base: x[0],
        home: x[1],
        attack_slope: x[2],
        defence_slope: x[3],
        elo_center: center,
        attack: (0..n).map(|i| x[2] * z[i] + x[4 + i]).collect(),
        defence: (0..n).map(|i| x[3] * z[i] + x[4 + n + i]).collect(),
        teams: all_teams,
        n_matches: rows.len(),
        n_market: rows.iter().filter(|m| !m.home_market.is_nan()).count(),
    })
}

/// The walk-forward team fit at the view's deadline.
pub fn fit_team(view: &AsOfView, params: &TeamParams) -> Result<TeamFit> {
    fit_ratings(
        &match_history(view, params)?,
        &elo(view)?,
        &current_teams(view)?,
        view.deadline(),
        params,
    )
}

/// (attack, defence) per team key; clubs outside the fit from their Elo.
fn strengths(fit: &TeamFit, teams: &[i64], elo: &HashMap<i64, f64>) -> Vec<(f64, f64)> {
    let known: HashMap<i64, (f64, f64)> =
        fit.teams.iter().enumerate().map(|(i, &t)| (t, (fit.attack[i], fit.defence[i]))).collect();
    let mut missing = BTreeSet::new();
    let out = teams
        .iter()
        .map(|t| {
            if let Some(&s) = known.get(t) {
                return s;
            }
            missing.insert(*t);
            let rating = elo.get(t).copied().unwrap_or(f64::NAN);
            let z = if rating.is_finite() { (rating - fit.elo_center) / ELO_SCALE } else { 0.0 };
            (fit.attack_slope * z, fit.defence_slope * z)
        })
        .collect();
    if !missing.is_empty() {
        log::warn!("team model: club(s) {missing:?} not in the fit; rated from Elo");
    }
    out
}

/// One row per fixture and side of `upcoming_fixtures(view)`: λ for / against, P(CS) and
/// the source ('market' with visible odds, else 'ratings' or 'stats'); sorted by (team_key,
/// gw_index, fixture_key).
pub fn team_lambdas(view: &AsOfView, fit: &TeamFit) -> Result<Table> {
    let sides = upcoming_fixtures(view)?;
    let fixture = sides.opt_i64("fixture_key")?;
    let opponent = sides.opt_i64("opponent_team_key")?;
    let is_home = sides.opt_bool("is_home")?;
    let (season, gw, gwi, horizon, team) = (
        sides.i64("season")?,
        sides.i64("gw")?,
        sides.i64("gw_index")?,
        sides.i64("horizon")?,
        sides.i64("team_key")?,
    );
    let rows: Vec<usize> = (0..sides.height()).filter(|&i| fixture[i].is_some()).collect();
    // The fixtures (home rows), sorted by fixture_key.
    let mut fixtures: Vec<(i64, i64, i64)> = rows
        .iter()
        .filter(|&&i| is_home[i] == Some(true))
        .map(|&i| (fixture[i].unwrap(), team[i], opponent[i].unwrap_or(0)))
        .collect();
    fixtures.sort_by_key(|f| f.0);
    let elo = elo(view)?;
    let home_s = strengths(fit, &fixtures.iter().map(|f| f.1).collect::<Vec<_>>(), &elo);
    let away_s = strengths(fit, &fixtures.iter().map(|f| f.2).collect::<Vec<_>>(), &elo);
    let keys: HashSet<i64> = fixtures.iter().map(|f| f.0).collect();
    let market = visible_market_lambdas(view, &keys, &fit.params)?;
    let ratings_source = if fit.source == "stats" { "stats" } else { "ratings" };
    // fixture_key → (λ_home, λ_away, P(home CS), P(away CS), source).
    let mut solved: HashMap<i64, (f64, f64, f64, f64, &str)> = HashMap::new();
    for (k, &(key, _, _)) in fixtures.iter().enumerate() {
        let (lh, la, source) = match market.get(&key) {
            Some(&(lh, la)) => (lh, la, "market"),
            None => (
                (fit.base + fit.home + home_s[k].0 - away_s[k].1).exp(),
                (fit.base + away_s[k].0 - home_s[k].1).exp(),
                ratings_source,
            ),
        };
        let m = dc_matrix(lh, la, fit.params.rho);
        let cs_home = np_sum(&(0..=MAX_GOALS).map(|i| m[i][0]).collect::<Vec<_>>());
        let cs_away = np_sum(&m[0]);
        solved.insert(key, (lh, la, cs_home, cs_away, source));
    }
    let mut out: Vec<usize> = rows;
    out.sort_by_key(|&i| (team[i], gwi[i], fixture[i]));
    let mut cols = TeamColumns::default();
    for i in out {
        let key = fixture[i].unwrap();
        let home = is_home[i] == Some(true);
        let (lh, la, cs_h, cs_a, source) =
            solved.get(&key).copied().unwrap_or((f64::NAN, f64::NAN, f64::NAN, f64::NAN, "nan"));
        cols.fixture_key.push(key);
        cols.season.push(season[i]);
        cols.gw.push(gw[i]);
        cols.gw_index.push(gwi[i]);
        cols.horizon.push(horizon[i]);
        cols.team_key.push(team[i]);
        cols.opponent.push(opponent[i].unwrap_or(0));
        cols.is_home.push(home);
        cols.lambda_for.push(if home { lh } else { la });
        cols.lambda_against.push(if home { la } else { lh });
        cols.p_cs.push(if home { cs_h } else { cs_a });
        cols.source.push(source.to_string());
    }
    Ok(Table::new()
        .with("fixture_key", Column::I64(cols.fixture_key))
        .with("season", Column::I64(cols.season))
        .with("gw", Column::I64(cols.gw))
        .with("gw_index", Column::I64(cols.gw_index))
        .with("horizon", Column::I64(cols.horizon))
        .with("team_key", Column::I64(cols.team_key))
        .with("opponent_team_key", Column::I64(cols.opponent))
        .with("is_home", Column::Bool(cols.is_home))
        .with("lambda_for", Column::F64(cols.lambda_for))
        .with("lambda_against", Column::F64(cols.lambda_against))
        .with("p_cs", Column::F64(cols.p_cs))
        .with("source", Column::Str(cols.source)))
}

#[derive(Default)]
struct TeamColumns {
    fixture_key: Vec<i64>,
    season: Vec<i64>,
    gw: Vec<i64>,
    gw_index: Vec<i64>,
    horizon: Vec<i64>,
    team_key: Vec<i64>,
    opponent: Vec<i64>,
    is_home: Vec<bool>,
    lambda_for: Vec<f64>,
    lambda_against: Vec<f64>,
    p_cs: Vec<f64>,
    source: Vec<String>,
}

impl TeamFit {
    /// The fitted values (the parity gate compares them with Python's `TeamFit`).
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "source": self.source,
            "base": self.base,
            "home": self.home,
            "attack_slope": self.attack_slope,
            "defence_slope": self.defence_slope,
            "elo_center": self.elo_center,
            "teams": self.teams,
            "attack": self.attack,
            "defence": self.defence,
            "n_matches": self.n_matches,
            "n_market": self.n_market,
        })
    }
}
