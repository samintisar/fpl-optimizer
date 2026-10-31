//! Candidate pruning (Python `tests/test_optimize_prune.py`).

mod common;

use std::collections::{BTreeMap, BTreeSet};

use common::*;
use fplopt_optimize::prune::{Candidates, prune};
use fplopt_optimize::{OptimizerParams, PlanInput};

/// `prune` on lists; every player has his own club unless `clubs` says otherwise.
struct Run<'a> {
    keys: &'a [i64],
    et: &'a [i64],
    price: &'a [i64],
    xp: Vec<Vec<f64>>,
    owned: &'a [i64],
    buyable: Option<Vec<bool>>,
    prune_n: Option<BTreeMap<i64, i64>>,
    dominated: bool,
    clubs: Option<&'a [i64]>,
}

impl<'a> Run<'a> {
    fn new(keys: &'a [i64], et: &'a [i64], price: &'a [i64], xp: Vec<Vec<f64>>) -> Run<'a> {
        Run {
            keys,
            et,
            price,
            xp,
            owned: &[],
            buyable: None,
            prune_n: None,
            dominated: true,
            clubs: None,
        }
    }

    fn run(&self) -> Vec<i64> {
        let n = self.keys.len();
        let owned: Vec<bool> = self.keys.iter().map(|k| self.owned.contains(k)).collect();
        let buyable = self.buyable.clone().unwrap_or_else(|| vec![true; n]);
        let horizons: Vec<i64> = (0..self.xp[0].len() as i64).collect();
        prune(
            &Candidates {
                keys: self.keys,
                element_type: self.et,
                team_key: self.clubs.unwrap_or(self.keys),
                price: self.price,
                xp: &self.xp,
                owned: &owned,
                buyable: &buyable,
                minutes: None,
            },
            0.85,
            &horizons,
            &rules(),
            self.prune_n.as_ref(),
            self.dominated,
            0.0,
        )
        .unwrap()
    }
}

fn column(values: &[f64]) -> Vec<Vec<f64>> {
    values.iter().map(|x| vec![*x]).collect()
}

#[test]
fn top_n_by_xp_and_by_value_per_position() {
    // 6 FWDs: xp 10, 9, 8, 1, 0.9, 0.1; prices 100, 100, 100, 10, 45, 45.
    let keys = [1, 2, 3, 4, 5, 6];
    let r = Run {
        prune_n: Some(BTreeMap::from([(4, 2)])),
        dominated: false,
        ..Run::new(
            &keys,
            &[4; 6],
            &[100, 100, 100, 10, 45, 45],
            column(&[10.0, 9.0, 8.0, 1.0, 0.9, 0.1]),
        )
    };
    // best two by xp (1, 2) and by xp/price (4: 0.1, 1: 0.1 tie → 1)
    assert_eq!(r.run(), vec![1, 2, 4]);
}

#[test]
fn owned_always_kept_and_unlisted_positions_untouched() {
    let keys = [1, 2, 3, 4, 5];
    let r = Run {
        owned: &[3],
        prune_n: Some(BTreeMap::from([(4, 1)])),
        ..Run::new(&keys, &[4, 4, 4, 3, 3], &[50; 5], column(&[5.0, 4.0, 0.0, 0.0, 0.0]))
    };
    let kept = r.run();
    assert!(kept.contains(&3) && kept.contains(&1) && !kept.contains(&2));
    // MIDs (3) aren't in prune_n: both kept by top-N; neither dominated by 5 (squad_select).
    assert!(kept.contains(&4) && kept.contains(&5));
}

#[test]
fn dominance_needs_dominators_from_enough_clubs() {
    // GKs: squad_select 2 + 15 // 3 full clubs = dominators from 7 clubs needed.
    let keys: Vec<i64> = (1..=8).collect();
    let mut xp = vec![vec![5.0, 5.0]; 7];
    xp.push(vec![3.0, 3.0]);
    let price = [40, 40, 40, 40, 40, 40, 40, 45];
    // 1-7 (identical: tie by key, so 7 has 6 dominators) dominate 8, from 7 clubs.
    assert_eq!(Run::new(&keys, &[1; 8], &price, xp.clone()).run(), keys[..7].to_vec());
    // The same 7 dominators from 3 clubs: the club cap could block them all, so 8 stays.
    let clubs = [1, 1, 1, 2, 2, 2, 3, 4];
    assert_eq!(
        Run { clubs: Some(&clubs), ..Run::new(&keys, &[1; 8], &price, xp.clone()) }.run(),
        keys
    );
    // Six dominators: kept.
    assert_eq!(
        Run::new(&keys[1..], &[1; 7], &price[1..], xp[1..].to_vec()).run(),
        keys[1..].to_vec()
    );
    // Cheaper but worse in one GW: not dominated.
    let mut worse = vec![vec![5.0, 5.0]; 6];
    worse.extend([vec![5.0, 1.0], vec![3.0, 3.0]]);
    assert_eq!(Run::new(&keys, &[1; 8], &price, worse).run(), keys);
}

#[test]
fn identical_players_tie_by_key_and_departed_never_dominate() {
    let keys = [7, 3, 5, 9, 11, 13, 15, 17, 19];
    // Sorted 3, 5, ..., 19: each is dominated by the smaller keys; 17 and 19 have 7+.
    assert_eq!(
        Run::new(&keys, &[1; 9], &[40; 9], column(&[2.0; 9])).run(),
        vec![3, 5, 7, 9, 11, 13, 15]
    );
    // 3 and 5 can't be bought (left the game; owned): they dominate nobody.
    let buyable: Vec<bool> = keys.iter().map(|k| !matches!(k, 3 | 5)).collect();
    let r = Run {
        owned: &[3, 5],
        buyable: Some(buyable),
        ..Run::new(&keys, &[1; 9], &[40; 9], column(&[2.0; 9]))
    };
    let mut sorted = keys.to_vec();
    sorted.sort_unstable();
    assert_eq!(r.run(), sorted);
}

#[test]
fn deterministic_under_input_order() {
    let mut rng = Rng::new(0);
    let n = 60;
    let keys: Vec<i64> = rng.permutation(1000).into_iter().take(n).map(|k| k as i64).collect();
    let et: Vec<i64> = (0..n).map(|_| rng.integers(1, 5)).collect();
    let price: Vec<i64> = (0..n).map(|_| rng.integers(40, 120)).collect();
    let xp: Vec<Vec<f64>> =
        (0..n).map(|_| (0..3).map(|_| round_to(rng.uniform(0.0, 8.0), 1)).collect()).collect();
    let prune_n = Some(BTreeMap::from([(1, 3), (2, 5), (3, 5), (4, 3)]));
    let a = Run { prune_n: prune_n.clone(), ..Run::new(&keys, &et, &price, xp.clone()) }.run();
    let order = rng.permutation(n);
    let pick = |v: &[i64]| order.iter().map(|&i| v[i]).collect::<Vec<_>>();
    let (k2, e2, p2) = (pick(&keys), pick(&et), pick(&price));
    let x2: Vec<Vec<f64>> = order.iter().map(|&i| xp[i].clone()).collect();
    let b = Run { prune_n, ..Run::new(&k2, &e2, &p2, x2) }.run();
    assert_eq!(a, b);
}

#[test]
fn pruning_keeps_the_optimum_on_random_instances() {
    // (pruned, dominance-only, unpruned) plan objectives on random instances.
    let rules = rules();
    for seed in 0..6 {
        // 12 clubs: dominance needs dominators from up to 10 clubs (squad_select + 5).
        let pool = grid_pool(1..=12, &[(1, 1), (2, 3), (3, 3), (4, 2)], Some(seed));
        let xp = correlated_xp(&pool, 2, seed, 5);
        let state = state_with(&pool, &first_valid_squad(&pool), 5, 2);
        let base = OptimizerParams { horizon: 2, mip_gap: 0.0, ..Default::default() };
        let mut out = Vec::new();
        for params in [
            base.clone(),
            OptimizerParams { prune_n: None, ..base.clone() },
            OptimizerParams { prune_n: None, prune_dominated: false, ..base.clone() },
        ] {
            let problem: PlanInput = problem(&state, &pool, &xp, &params, &rules);
            let kept: BTreeSet<i64> = problem.players.iter().map(|p| p.player_key).collect();
            assert!(state.player_keys().iter().all(|k| kept.contains(k)));
            out.push((problem.players.len(), solve(&problem, &params).objective));
        }
        let [(n_pruned, pruned), (n_dom, dominance), (n_all, unpruned)] = out[..] else {
            unreachable!()
        };
        assert!(n_pruned <= n_dom && n_dom <= n_all, "seed {seed}");
        assert!((dominance - unpruned).abs() <= 1e-6, "seed {seed}: {dominance} vs {unpruned}");
        // Top-N is a heuristic: allow the default MIP gap.
        assert!(
            pruned >= unpruned * (1.0 - OptimizerParams::default().mip_gap) - 1e-6,
            "seed {seed}"
        );
    }
}

#[test]
fn default_params() {
    let params = OptimizerParams::default();
    assert_eq!(params.prune_n, Some(BTreeMap::from([(1, 20), (2, 60), (3, 60), (4, 30)])));
    assert_eq!(params.ft_state_value(1), 0.0);
    assert!((params.ft_state_value(3) - 3.6).abs() < 1e-12);
    let bad = OptimizerParams { ft_value: BTreeMap::from([(2, -1.0)]), ..Default::default() };
    assert!(bad.validate().is_err());
}
