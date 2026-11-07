//! The placebo wobble on xP (`xp_jitter`, `jitter_seed` on the optimizer policy; noise study,
//! `docs/superpowers/specs/2026-11-07-noise-study-design.md`): each player's xP for a GW is
//! multiplied by (1 + sigma * eps), eps ~ N(0, 1) drawn from a hash of (seed, season, decision
//! gw_index, player_key, GW). Two seeds give two planners that are equal in expectation, so
//! their compared difference is 0 in truth. Deterministic and independent of thread order; one
//! eps per player and GW (a double GW's fixture rows share it).

use fplopt_optimize::XpFrame;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct XpJitter {
    pub sigma: f64,
    pub seed: u64,
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn unit(h: u64) -> f64 {
    // (0, 1): never 0, so ln is finite.
    ((h >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

/// eps ~ N(0, 1) for (seed, season, decision gw_index, player_key, GW), by Box-Muller.
pub fn epsilon(seed: u64, season: i64, gw_index: i64, player_key: i64, gw: i64) -> f64 {
    let mut h = splitmix64(seed);
    for v in [season, gw_index, player_key, gw] {
        h = splitmix64(h ^ v as u64);
    }
    let (u1, u2) = (unit(h), unit(splitmix64(h ^ 0xD1B5_4A32_D192_ED03)));
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

impl XpJitter {
    /// A copy of `frame` with every non-null xP multiplied by (1 + sigma * eps).
    pub fn apply(&self, frame: &XpFrame, season: i64, gw_index: i64) -> XpFrame {
        let mut out = frame.clone();
        for i in 0..out.len() {
            if let Some(x) = out.xp[i] {
                let eps = epsilon(self.seed, season, gw_index, out.player_key[i], out.gw[i]);
                out.xp[i] = Some(x * (1.0 + self.sigma * eps));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame() -> XpFrame {
        XpFrame {
            player_key: vec![1, 1, 1, 2],
            gw: vec![5, 5, 6, 5],
            gw_index: vec![5, 5, 6, 5],
            horizon: vec![0, 0, 1, 0],
            xp: vec![Some(4.0), Some(2.0), Some(3.0), None],
            ..XpFrame::default()
        }
    }

    #[test]
    fn deterministic_and_seed_dependent() {
        assert_eq!(epsilon(1, 2022, 5, 10, 5), epsilon(1, 2022, 5, 10, 5));
        assert_ne!(epsilon(1, 2022, 5, 10, 5), epsilon(2, 2022, 5, 10, 5));
        assert_ne!(epsilon(1, 2022, 5, 10, 5), epsilon(1, 2022, 6, 10, 5));
    }

    #[test]
    fn standard_normal_moments() {
        let n = 200_000;
        let xs: Vec<f64> = (0..n).map(|k| epsilon(7, 2022, 1, k, 3)).collect();
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!((var - 1.0).abs() < 0.02, "var {var}");
    }

    #[test]
    fn apply_scales_xp_one_eps_per_player_gw() {
        let j = XpJitter { sigma: 0.05, seed: 3 };
        let out = j.apply(&frame(), 2022, 5);
        let e = epsilon(3, 2022, 5, 1, 5);
        assert_eq!(out.xp[0], Some(4.0 * (1.0 + 0.05 * e)));
        // the double GW's second fixture row shares the eps
        assert_eq!(out.xp[1], Some(2.0 * (1.0 + 0.05 * e)));
        assert_eq!(out.xp[3], None);
        assert_eq!(out.player_key, frame().player_key);
    }

    #[test]
    fn zero_sigma_is_identity() {
        let out = XpJitter { sigma: 0.0, seed: 9 }.apply(&frame(), 2022, 5);
        assert_eq!(out.xp, frame().xp);
    }
}
