//! Candidate pruning (Python `fplopt.optimize.prune`; PLAN §7 *Practicalities*): owned
//! players always kept; a minutes floor first (frames with `e_minutes`); then per position
//! the top-N by horizon xP and by horizon xP per price; then dominance (a non-owned player
//! dominated by buyable players of his position from enough distinct clubs is dropped).
//! Ties by `player_key`. The rules and their reasons are in the Python module docstring.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashSet};

use anyhow::{Result, bail};
use fplopt_build::scoring::Rules;

/// Σ_t decay**horizons[t] · xp[t] per row (summed in horizon order).
pub fn horizon_xp(xp: &[Vec<f64>], horizons: &[i64], decay: f64) -> Vec<f64> {
    let weights: Vec<f64> = horizons.iter().map(|&h| decay.powf(h as f64)).collect();
    xp.iter()
        .map(|row| {
            let mut total = 0.0;
            for (x, w) in row.iter().zip(&weights) {
                total += x * w;
            }
            total
        })
        .collect()
}

/// The n entries of `idx` with the highest score (ties: smaller key first).
fn top(idx: &[usize], score: &[f64], keys: &[i64], n: usize) -> Vec<usize> {
    let mut order = idx.to_vec();
    order.sort_by(|&a, &b| {
        (-score[a]).partial_cmp(&-score[b]).unwrap_or(Ordering::Equal).then(keys[a].cmp(&keys[b]))
    });
    order.truncate(n);
    order
}

/// Per-player arrays for `prune` (aligned; `xp` is players × horizon GWs).
pub struct Candidates<'a> {
    pub keys: &'a [i64],
    pub element_type: &'a [i64],
    pub team_key: &'a [i64],
    pub price: &'a [i64],
    pub xp: &'a [Vec<f64>],
    pub owned: &'a [bool],
    pub buyable: &'a [bool],
    /// Expected minutes over the horizon (None: a frame without minutes).
    pub minutes: Option<&'a [f64]>,
}

/// The kept `player_key`s, sorted.
#[allow(clippy::too_many_arguments)]
pub fn prune(
    c: &Candidates,
    decay: f64,
    horizons: &[i64],
    rules: &Rules,
    prune_n: Option<&BTreeMap<i64, i64>>,
    dominated: bool,
    min_minutes: f64,
) -> Result<Vec<i64>> {
    let n = c.keys.len();
    if c.keys.iter().collect::<HashSet<_>>().len() != n {
        bail!("prune: duplicate player keys");
    }
    let hxp = horizon_xp(c.xp, horizons, decay);
    let mut eligible = vec![true; n];
    if let Some(minutes) = c.minutes
        && min_minutes > 0.0
    {
        for i in 0..n {
            let m = if minutes[i].is_nan() { 0.0 } else { minutes[i] };
            eligible[i] = c.owned[i] || m >= min_minutes;
        }
    }
    let positions: BTreeSet<i64> = c.element_type.iter().copied().collect();
    let mut keep = c.owned.to_vec();
    match prune_n {
        None => keep.clone_from(&eligible),
        Some(prune_n) => {
            let value: Vec<f64> = (0..n).map(|i| hxp[i] / (c.price[i].max(1) as f64)).collect();
            for &et in &positions {
                let idx: Vec<usize> = (0..n)
                    .filter(|&i| {
                        c.element_type[i] == et && !c.owned[i] && c.buyable[i] && eligible[i]
                    })
                    .collect();
                match prune_n.get(&et) {
                    None => idx.iter().for_each(|&i| keep[i] = true),
                    Some(&k) => {
                        let k = k.max(0) as usize;
                        for i in top(&idx, &hxp, c.keys, k) {
                            keep[i] = true;
                        }
                        for i in top(&idx, &value, c.keys, k) {
                            keep[i] = true;
                        }
                    }
                }
            }
        }
    }

    if dominated {
        let full_clubs = rules.squad_size / rules.team_limit.max(1);
        for &et in &positions {
            let idx: Vec<usize> = (0..n).filter(|&i| keep[i] && c.element_type[i] == et).collect();
            let need = rules.squad_select.get(&et).copied().unwrap_or(0) + full_clubs;
            if idx.len() as i64 <= need {
                continue;
            }
            let mut drop = Vec::new();
            for &j in &idx {
                if c.owned[j] {
                    continue;
                }
                let mut clubs = HashSet::new();
                for &d in &idx {
                    if d != j && c.buyable[d] && dominates(c, d, j) {
                        clubs.insert(c.team_key[d]);
                    }
                }
                if clubs.len() as i64 >= need {
                    drop.push(j);
                }
            }
            for j in drop {
                keep[j] = false;
            }
        }
    }
    let mut kept: Vec<i64> = (0..n).filter(|&i| keep[i]).map(|i| c.keys[i]).collect();
    kept.sort_unstable();
    Ok(kept)
}

/// d dominates j: price ≤ and xP ≥ in every GW, strictly better somewhere or (identical)
/// a smaller key.
fn dominates(c: &Candidates, d: usize, j: usize) -> bool {
    let (xd, xj) = (&c.xp[d], &c.xp[j]);
    let weakly = c.price[d] <= c.price[j] && xd.iter().zip(xj).all(|(a, b)| a >= b);
    let strictly = c.price[d] < c.price[j] || xd.iter().zip(xj).any(|(a, b)| a > b);
    weakly && (strictly || c.keys[d] < c.keys[j])
}
