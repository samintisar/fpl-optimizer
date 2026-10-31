//! A squad's gameweek score: lineup, autosubs, captaincy and chips (Python
//! `fplopt.backtest.gw_score`; PLAN §5 *Simulator*).
//!
//! "Played" = minutes > 0 summed over the GW's fixtures; a blank (no row) is did-not-play.
//! The bench GK (bench[0]) can only replace the starting GK; the outfield bench players, in
//! bench order, each replace the first non-playing outfield starter (FPL pick order: by
//! position, then lineup order) whose replacement keeps the formation within the limits.
//! Bench Boost: all 15 count, no autosubs. Captain ×2 (Triple Captain ×3) if he played, else
//! the vice if he played, else nobody. Points are generic: realized int points or xG-scored
//! floats (`Points`), summed in Python's order (from 0, in counted order).

use std::collections::{BTreeMap, HashMap};
use std::fmt;

use fplopt_build::scoring::Rules;

use crate::pyrepr;

pub const CHIP_NAMES: [&str; 4] = ["wildcard", "freehit", "bboost", "3xc"];

/// A lineup that breaks the formation or selection rules (Python `InvalidLineup`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InvalidLineup(pub String);

impl fmt::Display for InvalidLineup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for InvalidLineup {}

/// The 11 starters, the 4 bench players in autosub order (bench[0] = the GK), and the
/// captain and vice (both starters), as player_keys.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Lineup {
    pub starters: Vec<i64>,
    pub bench: Vec<i64>,
    pub captain: i64,
    pub vice: i64,
}

/// Realized int points or xG-scored floats.
pub trait Points:
    Copy + Default + std::ops::Add<Output = Self> + std::ops::Sub<Output = Self>
{
    fn times(self, multiplier: i64) -> Self;
}

impl Points for i64 {
    fn times(self, multiplier: i64) -> i64 {
        self * multiplier
    }
}

impl Points for f64 {
    fn times(self, multiplier: i64) -> f64 {
        self * multiplier as f64
    }
}

/// `points` = Σ counted players' points × their multiplier; `counted` the players whose
/// points count (the XI after autosubs, or all 15 under Bench Boost) with `multipliers`;
/// `captain_used` the player who got the armband's multiplier; `bench_points` the points of
/// the squad players who did not count.
#[derive(Clone, Debug, PartialEq)]
pub struct GwScore<P> {
    pub points: P,
    pub counted: Vec<i64>,
    pub multipliers: HashMap<i64, i64>,
    pub captain_used: Option<i64>,
    pub autosubs: Vec<(i64, i64)>,
    pub bench_points: P,
}

fn formation_ok(counts: &BTreeMap<i64, i64>, rules: &Rules) -> bool {
    rules.play_min.iter().all(|(et, min)| {
        let n = counts.get(et).copied().unwrap_or(0);
        *min <= n && n <= rules.play_max[et]
    })
}

/// InvalidLineup unless: 11 starters and 4 bench players, all distinct with a known
/// position; `squad_select` players per position; the starters' formation within the
/// limits; bench[0] the GK; captain ≠ vice, both starters.
pub fn validate_lineup(
    lineup: &Lineup,
    positions: &HashMap<i64, i64>,
    rules: &Rules,
) -> Result<(), InvalidLineup> {
    let bad = |msg: String| Err(InvalidLineup(msg));
    let (starters, bench) = (&lineup.starters, &lineup.bench);
    if starters.len() as i64 != rules.squad_play {
        return bad(format!("{} starters, need {}", starters.len(), rules.squad_play));
    }
    if bench.len() as i64 != rules.squad_size - rules.squad_play {
        return bad(format!(
            "{} bench players, need {}",
            bench.len(),
            rules.squad_size - rules.squad_play
        ));
    }
    let squad: Vec<i64> = starters.iter().chain(bench).copied().collect();
    let mut sorted = squad.clone();
    sorted.sort_unstable();
    if sorted.windows(2).any(|w| w[0] == w[1]) {
        return bad(format!("duplicate players in lineup: {}", pyrepr::list(&sorted)));
    }
    let unknown: Vec<i64> = squad.iter().copied().filter(|k| !positions.contains_key(k)).collect();
    if !unknown.is_empty() {
        return bad(format!("no position for players {}", pyrepr::list(&unknown)));
    }
    let counts = pyrepr::counter(squad.iter().map(|k| positions[k]));
    let select: Vec<(i64, i64)> = rules.squad_select.iter().map(|(k, v)| (*k, *v)).collect();
    let as_map = |c: &[(i64, i64)]| c.iter().copied().collect::<BTreeMap<i64, i64>>();
    if as_map(&counts) != as_map(&select) {
        return bad(format!(
            "squad positions {} != {}",
            pyrepr::counter_dict(&counts),
            pyrepr::counter_dict(&select)
        ));
    }
    let formation = as_map(&pyrepr::counter(starters.iter().map(|k| positions[k])));
    if !formation_ok(&formation, rules) {
        return bad(format!("invalid formation {}", pyrepr::dict(&formation)));
    }
    if positions[&bench[0]] != 1 {
        return bad(format!("bench[0] ({}) must be the goalkeeper", bench[0]));
    }
    if lineup.captain == lineup.vice {
        return bad("captain and vice must differ".to_owned());
    }
    for (role, key) in [("captain", lineup.captain), ("vice", lineup.vice)] {
        if !starters.contains(&key) {
            return bad(format!("{role} {key} is not a starter"));
        }
    }
    Ok(())
}

/// Score a lineup with `outcomes` (player_key -> (points, minutes), summed over the GW's
/// fixtures); players missing from it scored 0 in 0 minutes. Only `bboost` and `3xc` change
/// the score.
pub fn score_gameweek<P: Points>(
    lineup: &Lineup,
    outcomes: &HashMap<i64, (P, i64)>,
    positions: &HashMap<i64, i64>,
    rules: &Rules,
    chip: Option<&str>,
) -> anyhow::Result<GwScore<P>> {
    if let Some(chip) = chip
        && !CHIP_NAMES.contains(&chip)
    {
        anyhow::bail!("unknown chip '{chip}'");
    }
    validate_lineup(lineup, positions, rules)?;
    let points = |key: i64| outcomes.get(&key).map_or_else(P::default, |o| o.0);
    let played = |key: i64| outcomes.get(&key).is_some_and(|o| o.1 > 0);

    let mut xi = lineup.starters.clone();
    let mut autosubs = Vec::new();
    let counted: Vec<i64> = if chip == Some("bboost") {
        xi.iter().chain(&lineup.bench).copied().collect()
    } else {
        let mut formation: BTreeMap<i64, i64> = BTreeMap::new();
        for k in &xi {
            *formation.entry(positions[k]).or_default() += 1;
        }
        let mut order: Vec<usize> = (0..xi.len()).collect();
        order.sort_by_key(|&i| (positions[&xi[i]], i));
        for &sub in &lineup.bench {
            if !played(sub) {
                continue;
            }
            let sub_et = positions[&sub];
            for &i in &order {
                let out = xi[i];
                let out_et = positions[&out];
                if played(out) || (out_et == 1) != (sub_et == 1) {
                    continue;
                }
                let mut trial = formation.clone();
                *trial.entry(out_et).or_default() -= 1;
                *trial.entry(sub_et).or_default() += 1;
                if formation_ok(&trial, rules) {
                    xi[i] = sub;
                    formation = trial;
                    autosubs.push((out, sub));
                    break;
                }
            }
        }
        xi
    };

    let multiplier = if chip == Some("3xc") { 3 } else { 2 };
    let captain_used = [lineup.captain, lineup.vice].into_iter().find(|&k| played(k));
    let mut multipliers: HashMap<i64, i64> = counted.iter().map(|&k| (k, 1)).collect();
    if let Some(c) = captain_used {
        multipliers.insert(c, multiplier);
    }
    let total = counted.iter().fold(P::default(), |acc, &k| acc + points(k).times(multipliers[&k]));
    let bench_points = lineup
        .starters
        .iter()
        .chain(&lineup.bench)
        .filter(|k| !multipliers.contains_key(k))
        .fold(P::default(), |acc, &k| acc + points(k));
    Ok(GwScore { points: total, counted, multipliers, captain_used, autosubs, bench_points })
}

#[cfg(test)]
mod tests {
    use super::*;
    use fplopt_build::scoring::backtest_rules;

    fn rules() -> Rules {
        let dir =
            std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../config/scoring"));
        backtest_rules(2024, dir).unwrap()
    }

    /// GK 1-2, DEF 3-7, MID 8-12, FWD 13-15; starters 1, 3-6, 8-11, 13-14 (4-4-2).
    fn squad() -> (Lineup, HashMap<i64, i64>) {
        let positions: HashMap<i64, i64> = (1..=15)
            .map(|k| {
                (
                    k,
                    if k <= 2 {
                        1
                    } else if k <= 7 {
                        2
                    } else if k <= 12 {
                        3
                    } else {
                        4
                    },
                )
            })
            .collect();
        let lineup = Lineup {
            starters: vec![1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14],
            bench: vec![2, 7, 12, 15],
            captain: 13,
            vice: 8,
        };
        (lineup, positions)
    }

    fn all_play(points: i64) -> HashMap<i64, (i64, i64)> {
        (1..=15).map(|k| (k, (points, 90))).collect()
    }

    #[test]
    fn captain_doubles_and_bench_counts_apart() {
        let (lineup, pos) = squad();
        let s = score_gameweek(&lineup, &all_play(2), &pos, &rules(), None).unwrap();
        assert_eq!(s.points, 2 * 11 + 2);
        assert_eq!(s.bench_points, 8);
        assert_eq!(s.captain_used, Some(13));
    }

    #[test]
    fn autosubs_keep_the_formation_and_the_vice_takes_over() {
        let (lineup, pos) = squad();
        let mut out = all_play(1);
        out.insert(13, (0, 0)); // captain FWD out: the bench FWD (15) can't come before 7, 12
        out.insert(3, (0, 0)); // a DEF out
        let s = score_gameweek(&lineup, &out, &pos, &rules(), None).unwrap();
        // bench order 7 (DEF) replaces DEF 3 first; 12 (MID) replaces FWD 13 (keeps 1 FWD)
        assert_eq!(s.autosubs, vec![(3, 7), (13, 12)]);
        assert_eq!(s.captain_used, Some(8));
        assert_eq!(s.points, 11 + 1);
    }

    #[test]
    fn bench_boost_and_triple_captain() {
        let (lineup, pos) = squad();
        let bb = score_gameweek(&lineup, &all_play(1), &pos, &rules(), Some("bboost")).unwrap();
        assert_eq!((bb.points, bb.bench_points), (16, 0));
        let tc = score_gameweek(&lineup, &all_play(1), &pos, &rules(), Some("3xc")).unwrap();
        assert_eq!(tc.points, 13);
    }

    #[test]
    fn invalid_lineups_are_named() {
        let (mut lineup, pos) = squad();
        lineup.vice = 13;
        let err = validate_lineup(&lineup, &pos, &rules()).unwrap_err();
        assert_eq!(err.0, "captain and vice must differ");
        let (mut lineup, pos) = squad();
        lineup.bench = vec![7, 2, 12, 15];
        let err = validate_lineup(&lineup, &pos, &rules()).unwrap_err();
        assert_eq!(err.0, "bench[0] (7) must be the goalkeeper");
    }
}
