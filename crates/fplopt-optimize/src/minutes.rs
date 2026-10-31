//! The minutes model in the planner (Python `fplopt.optimize.minutes`; PLAN §7 *Bench*,
//! *Pruning*): per-GW bench weights from P(starter doesn't play) for frames with `p_play`
//! (`v1`), and the expected-minutes floor for frames with `e_minutes`. The model and its
//! evaluation are documented in the Python module docstring.

use std::collections::{BTreeMap, HashMap};

use anyhow::{Result, bail};
use fplopt_build::scoring::Rules;

use crate::problem::XpFrame;
use crate::state::{GOALKEEPER, best_xi};
use crate::util::Kahan;

/// The distribution of N + B for dist[n] = P(N = n) and B ~ Bernoulli(value) (clipped).
fn add_bernoulli(dist: &[f64], value: f64) -> Vec<f64> {
    let p = value.clamp(0.0, 1.0);
    let mut out: Vec<f64> = dist.iter().map(|d| d * (1.0 - p)).collect();
    out.push(0.0);
    for (i, d) in dist.iter().enumerate() {
        out[i + 1] += d * p;
    }
    out
}

/// (P(N ≥ 1), ..., P(N ≥ k_max)) for N = Σ Bernoulli(q_i), independent.
pub fn poisson_binomial_tail(q: &[f64], k_max: usize) -> Vec<f64> {
    let mut dist = vec![1.0];
    for &value in q {
        dist = add_bernoulli(&dist, value);
    }
    // tail[k] = P(N >= k), a cumulative sum from the end (numpy's cumsum of the reverse).
    let mut tail = vec![0.0; dist.len()];
    let mut acc = 0.0;
    for k in (0..dist.len()).rev() {
        acc += dist[k];
        tail[k] = acc;
    }
    (1..=k_max).map(|k| if k < tail.len() { tail[k].clamp(0.0, 1.0) } else { 0.0 }).collect()
}

/// Per bench slot k: P(M ≥ 1 + S_{k−1}), M = Σ Bernoulli(q_starters), S_{k−1} = Σ_{j<k}
/// Bernoulli(p_bench[j]), all independent.
pub fn skip_aware_tail(q_starters: &[f64], p_bench: &[f64]) -> Vec<f64> {
    let mut tail = vec![1.0];
    tail.extend(poisson_binomial_tail(q_starters, q_starters.len() + p_bench.len() + 1));
    let mut out = Vec::new();
    let mut dist = vec![1.0];
    for &value in p_bench {
        let mut dot = 0.0;
        for (s, d) in dist.iter().enumerate() {
            let reach = if s + 1 < tail.len() { tail[s + 1] } else { 0.0 };
            dot += d * reach;
        }
        out.push(dot.clamp(0.0, 1.0));
        dist = add_bernoulli(&dist, value);
    }
    out
}

fn xp_of(xp: &HashMap<i64, f64>, key: i64) -> f64 {
    let x = xp.get(&key).copied().unwrap_or(0.0);
    if x.is_nan() { 0.0 } else { x }
}

/// The XI with the most xP among `players` ((player_key, element_type)) under the formation
/// limits; missing or NaN xP counts 0; ties by player_key.
pub fn projected_xi(players: &[(i64, i64)], xp: &HashMap<i64, f64>, rules: &Rules) -> Vec<i64> {
    let mut order = players.to_vec();
    order.sort_by(|a, b| {
        (-xp_of(xp, a.0)).partial_cmp(&-xp_of(xp, b.0)).unwrap().then(a.0.cmp(&b.0))
    });
    best_xi(&order, rules).into_iter().map(|(k, _)| k).collect()
}

/// P(doesn't play) per (player_key, horizon): 1 − `p_play_gw` (else `p_play`), clipped;
/// rows of one player-horizon multiply; NaN where any is unknown. None without `p_play`.
pub fn not_play_probabilities(xp: &XpFrame) -> Option<BTreeMap<(i64, i64), f64>> {
    let p_play = xp.p_play.as_ref()?;
    let mut out: BTreeMap<(i64, i64), (f64, bool)> = BTreeMap::new();
    for i in 0..xp.len() {
        let mut p = p_play[i].unwrap_or(f64::NAN);
        if let Some(gw) = &xp.p_play_gw
            && let Some(v) = gw[i]
            && !v.is_nan()
        {
            p = v;
        }
        let q = 1.0 - p.clamp(0.0, 1.0);
        let entry = out.entry((xp.player_key[i], xp.horizon[i])).or_insert((1.0, false));
        if q.is_nan() {
            entry.1 = true;
        } else {
            entry.0 *= q;
        }
    }
    Some(out.into_iter().map(|(k, (q, unknown))| (k, if unknown { f64::NAN } else { q })).collect())
}

/// Bench weights (GK slot, outfield slots) per horizon GW for the incumbent squad from the
/// frame's minutes; `fallback` for a GW where a projected starter's P(play) is unknown.
/// None when the frame has no `p_play`.
pub fn minutes_bench_weights(
    squad: &[(i64, i64)],
    xp: &XpFrame,
    horizons: &[i64],
    rules: &Rules,
    fallback: &[f64],
) -> Option<Vec<Vec<f64>>> {
    let q_map = not_play_probabilities(xp)?;
    let n_outfield = fallback.len() - 1;
    let et: HashMap<i64, i64> = squad.iter().copied().collect();
    let mut out = Vec::new();
    for &h in horizons {
        // The GW's xP per player: rows of one player add up (NaN if all are NaN).
        let mut sums: BTreeMap<i64, (Kahan, bool)> = BTreeMap::new();
        for i in 0..xp.len() {
            if xp.horizon[i] != h {
                continue;
            }
            let entry = sums.entry(xp.player_key[i]).or_default();
            if let Some(x) = xp.xp[i]
                && !x.is_nan()
            {
                entry.0.add(x);
                entry.1 = true;
            }
        }
        let target: HashMap<i64, f64> = sums
            .into_iter()
            .map(|(k, (s, any))| (k, if any { s.value() } else { f64::NAN }))
            .collect();
        let xi = projected_xi(squad, &target, rules);
        let mut bench: Vec<i64> = squad
            .iter()
            .filter(|(k, e)| !xi.contains(k) && *e != GOALKEEPER)
            .map(|(k, _)| *k)
            .collect();
        bench.sort_by(|a, b| {
            (-xp_of(&target, *a)).partial_cmp(&-xp_of(&target, *b)).unwrap().then(a.cmp(b))
        });
        bench.truncate(n_outfield);
        let risk = |k: i64| q_map.get(&(k, h)).copied().unwrap_or(1.0);
        if xi.iter().chain(&bench).any(|&k| risk(k).is_nan()) {
            out.push(fallback.to_vec());
            continue;
        }
        let keepers: Vec<f64> =
            xi.iter().filter(|k| et[k] == GOALKEEPER).map(|&k| risk(k)).collect();
        let outfield: Vec<f64> =
            xi.iter().filter(|k| et[k] != GOALKEEPER).map(|&k| risk(k)).collect();
        let gk = poisson_binomial_tail(&keepers, 1)[0];
        let mut plays: Vec<f64> = bench.iter().map(|&k| 1.0 - risk(k)).collect();
        plays.resize(n_outfield, 0.0);
        let mut weights = vec![gk];
        weights.extend(skip_aware_tail(&outfield, &plays));
        out.push(weights);
    }
    Some(out)
}

/// Expected minutes per player of `keys` summed over `horizons` (0 without rows); None when
/// the frame has no `e_minutes`. A NaN is an error (it would silently drop the player).
pub fn horizon_minutes(xp: &XpFrame, horizons: &[i64], keys: &[i64]) -> Result<Option<Vec<f64>>> {
    let Some(e_minutes) = &xp.e_minutes else { return Ok(None) };
    let mut sums: HashMap<i64, Kahan> = HashMap::new();
    let mut bad = Vec::new();
    for ((key, horizon), minutes) in xp.player_key.iter().zip(&xp.horizon).zip(e_minutes) {
        if !horizons.contains(horizon) {
            continue;
        }
        match minutes {
            Some(v) if !v.is_nan() => sums.entry(*key).or_default().add(*v),
            _ => bad.push(*key),
        }
    }
    if !bad.is_empty() {
        bad.sort_unstable();
        bad.dedup();
        bad.truncate(10);
        bail!("NaN e_minutes in the xP frame, e.g. players {bad:?}");
    }
    Ok(Some(keys.iter().map(|k| sums.get(k).map_or(0.0, Kahan::value)).collect()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tails() {
        let t = poisson_binomial_tail(&[0.5, 0.5], 3);
        assert_eq!(t, vec![0.75, 0.25, 0.0]);
        // One starter missing for sure: slot 1 comes on; slot 2 only if slot 1 doesn't play.
        let s = skip_aware_tail(&[1.0], &[0.8, 1.0]);
        assert_eq!(s, vec![1.0, 1.0 - 0.8]);
    }
}
