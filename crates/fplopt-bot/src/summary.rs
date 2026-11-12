//! The facts of a plan for the bot's messages (spec
//! docs/superpowers/specs/2026-11-11-bot-messaging-ux-design.md): plain data, no text.
//! `messages` words them.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use fplopt_optimize::state::selling_price;
use fplopt_optimize::{GwPlan, Plan, XpFrame};

use crate::context::Fixture;
use crate::planner::{UserPlan, ranked};

/// A club's games per GW: (gw, [(opponent, home)]), empty for a blank.
pub type GwGames = Vec<(i64, Vec<(String, bool)>)>;

/// A player's expected points summed over `gws` (every row of a player-GW counts, so a
/// double adds both games; NaN or missing counts 0); None if the frame has no row for him in
/// those GWs.
pub fn horizon_points(xp: &XpFrame, player_key: i64, gws: &[i64]) -> Option<f64> {
    let mut total = None;
    for i in 0..xp.len() {
        if xp.player_key[i] == player_key && gws.contains(&xp.gw[i]) {
            let x = xp.xp[i].filter(|v| !v.is_nan()).unwrap_or(0.0);
            total = Some(total.unwrap_or(0.0) + x);
        }
    }
    total
}

/// A club's games per GW in `gws` (in that order): (gw, [(opponent, home)]), empty for a
/// blank.
pub fn gw_fixtures(fixtures: &[Fixture], team_key: i64, gws: &[i64]) -> GwGames {
    gws.iter()
        .map(|&gw| {
            let games = fixtures
                .iter()
                .filter(|f| f.gw == gw && f.team_key == team_key)
                .map(|f| (f.opponent.clone(), f.home))
                .collect();
            (gw, games)
        })
        .collect()
}

/// Plans whose objectives differ by less than the solver's gap: equally good as far as the
/// solver can tell (PLAN §8 live finding).
pub fn tied(a: &Plan, b: &Plan) -> bool {
    let gap = a.mip_gap.max(b.mip_gap) * a.total_objective().abs();
    (a.total_objective() - b.total_objective()).abs() <= gap
}

/// Positions in lineup order.
pub const POSITIONS: [(i64, &str); 4] = [(1, "GK"), (2, "DEF"), (3, "MID"), (4, "FWD")];

/// One of this GW's transfers in plan 1.
#[derive(Clone, Debug, PartialEq)]
pub struct Move {
    pub out_name: String,
    pub in_name: String,
    /// What the sale earns and the buy costs (tenths of £m).
    pub sell_price: i64,
    pub buy_price: i64,
    /// Expected points over the plan's horizon GWs (undecayed).
    pub out_points: f64,
    pub in_points: f64,
    /// The incoming player's games in his next 3 GWs, or None without a fixtures frame.
    pub in_fixtures: Option<GwGames>,
}

/// A player and his expected points this GW.
#[derive(Clone, Debug, PartialEq)]
pub struct Pick {
    pub name: String,
    pub points: f64,
}

/// Another plan: its first-GW moves (out, in names), hit points, gain over rolling and
/// whether the solver can't tell it from plan 1.
#[derive(Clone, Debug, PartialEq)]
pub struct Alternative {
    pub moves: Vec<(String, String)>,
    pub hit_points: i64,
    pub gain: Option<f64>,
    pub tied: bool,
}

/// The facts of a user's plan (spec, *Summary step*).
#[derive(Clone, Debug, PartialEq)]
pub struct PlanSummary {
    pub gw: i64,
    pub deadline: DateTime<Utc>,
    /// GWs in plan 1's horizon.
    pub horizon: usize,
    pub moves: Vec<Move>,
    pub hit_points: i64,
    pub captain: Pick,
    pub vice: Pick,
    pub gain_vs_roll: Option<f64>,
    pub bank_after: i64,
    pub free_transfers_next: i64,
    /// Starters by position ("GK", names with " (C)"/" (V)"), positions without starters left
    /// out.
    pub team: Vec<(&'static str, Vec<String>)>,
    pub bench: Vec<String>,
    pub alternatives: Vec<Alternative>,
    /// Rolling the transfer: free transfers next GW, and whether it ties plan 1. None when the
    /// planner gave no roll plan or plan 1 makes no transfers.
    pub roll: Option<(i64, bool)>,
    /// Plan 1's later moves: (gw, out, in).
    pub later: Vec<(i64, String, String)>,
    pub stopped_early: bool,
    /// When FPL's live prices couldn't be fetched: the time of the prices used.
    pub prices_fallback: Option<DateTime<Utc>>,
    /// The plan starts from transfers the user recorded with /transfer.
    pub told_transfers: bool,
}

fn name(names: &HashMap<i64, String>, key: i64) -> String {
    names.get(&key).cloned().unwrap_or_else(|| format!("player {key}"))
}

fn moves_of(g: &GwPlan, names: &HashMap<i64, String>) -> Vec<(String, String)> {
    g.transfers.iter().map(|t| (name(names, t.out_key), name(names, t.in_key))).collect()
}

/// Free transfers at the next GW after making `n` transfers with `ft` available (FPL's
/// roll-over, capped; top-ups ignored).
fn next_free_transfers(ft: i64, n: i64, cap: i64) -> i64 {
    ((ft - n).max(0) + 1).min(cap)
}

/// The summary of `plan`, or None when the planner found no plan.
pub fn plan_summary(plan: &UserPlan) -> Option<PlanSummary> {
    let context = &plan.context;
    if plan.plans.plans.is_empty() {
        return None;
    }
    let names = context.names();
    let order = ranked(&plan.plans);
    let best = order[0];
    let first = best.first();
    let gws: Vec<i64> = best.gws.iter().map(|g| g.gw).collect();
    let points = |key: i64, gws: &[i64]| horizon_points(&context.xp, key, gws).unwrap_or(0.0);
    let pool: HashMap<i64, (i64, i64, i64)> = context
        .pool
        .iter()
        .map(|p| (p.player_key, (p.price, p.team_key, p.element_type)))
        .collect();
    let next3: Vec<i64> =
        context.gameweeks.iter().map(|g| g.0).filter(|&g| g >= first.gw).take(3).collect();
    let moves = first
        .transfers
        .iter()
        .map(|t| {
            let sell_price = plan
                .state
                .holdings
                .iter()
                .find(|h| h.player_key == t.out_key)
                .map_or(0, |h| selling_price(h, plan.rules.sell_on_fee));
            let (buy_price, team_key, _) = pool.get(&t.in_key).copied().unwrap_or((0, 0, 0));
            Move {
                out_name: name(&names, t.out_key),
                in_name: name(&names, t.in_key),
                sell_price,
                buy_price,
                out_points: points(t.out_key, &gws),
                in_points: points(t.in_key, &gws),
                in_fixtures: context.fixtures.as_ref().map(|f| gw_fixtures(f, team_key, &next3)),
            }
        })
        .collect();
    let mark = |k: i64| {
        if k == first.vice {
            " (V)"
        } else if k == first.captain {
            " (C)"
        } else {
            ""
        }
    };
    let team = POSITIONS
        .iter()
        .filter_map(|&(pos, label)| {
            let players: Vec<String> = first
                .starters
                .iter()
                .filter(|k| pool.get(k).map_or(0, |p| p.2) == pos)
                .map(|&k| format!("{}{}", name(&names, k), mark(k)))
                .collect();
            (!players.is_empty()).then_some((label, players))
        })
        .collect();
    let cap = plan.rules.max_free_transfers;
    let ft = plan.state.free_transfers;
    let roll = match &plan.plans.roll {
        Some(r) if !first.transfers.is_empty() => {
            Some((next_free_transfers(ft, 0, cap), tied(best, r)))
        }
        _ => None,
    };
    Some(PlanSummary {
        gw: context.gw,
        deadline: context.deadline,
        horizon: best.gws.len(),
        moves,
        hit_points: plan.rules.hit_cost * first.hits,
        captain: Pick {
            name: name(&names, first.captain),
            points: points(first.captain, &[first.gw]),
        },
        vice: Pick { name: name(&names, first.vice), points: points(first.vice, &[first.gw]) },
        gain_vs_roll: best.gain_vs_roll,
        bank_after: first.bank,
        free_transfers_next: next_free_transfers(ft, first.transfers.len() as i64, cap),
        team,
        bench: first.bench.iter().map(|&k| name(&names, k)).collect(),
        alternatives: order[1..]
            .iter()
            .map(|p| Alternative {
                moves: moves_of(p.first(), &names),
                hit_points: plan.rules.hit_cost * p.first().hits,
                gain: p.gain_vs_roll,
                tied: tied(best, p),
            })
            .collect(),
        roll,
        later: best.gws[1..]
            .iter()
            .flat_map(|g| moves_of(g, &names).into_iter().map(move |(o, i)| (g.gw, o, i)))
            .collect(),
        stopped_early: !best.optimal(),
        prices_fallback: match context.prices_at {
            Some(_) => None,
            None => Some(context.bootstrap_at.unwrap_or(context.created_at)),
        },
        told_transfers: plan.spec.notes.iter().any(|n| n.starts_with("transfers you told the bot")),
    })
}

/// What changed in plan 1's decision since the last alert (spec, *Update alert*).
#[derive(Clone, Debug, PartialEq)]
pub struct PlanChange {
    pub gw: i64,
    pub deadline: DateTime<Utc>,
    pub horizon: usize,
    pub was_moves: Vec<(String, String)>,
    pub now_moves: Vec<(String, String)>,
    pub moves_changed: bool,
    /// (now, Some(was) if it changed)
    pub captain: (String, Option<String>),
    pub vice: (String, Option<String>),
    /// Incoming players new to the plan / dropped from it, with their horizon points (players
    /// without xP rows left out).
    pub now_points: Vec<(String, f64)>,
    pub was_points: Vec<(String, f64)>,
}

fn pairs(transfers: &serde_json::Value) -> Vec<(i64, i64)> {
    transfers
        .as_array()
        .map(|items| {
            items
                .iter()
                .map(|p| (p[0].as_i64().unwrap_or_default(), p[1].as_i64().unwrap_or_default()))
                .collect()
        })
        .unwrap_or_default()
}

/// The change from `before` (the stored decision JSON) to `after`, or None if nothing
/// changed (or there is no earlier decision). `points(key)`: a player's horizon points.
pub fn change_of(
    before: Option<&serde_json::Value>,
    after: &crate::planner::Decision,
    names: &HashMap<i64, String>,
    points: &dyn Fn(i64) -> Option<f64>,
) -> Option<PlanChange> {
    let before = before?;
    let was = pairs(&before["transfers"]);
    let now: Vec<(i64, i64)> = after.transfers.iter().map(|p| (p[0], p[1])).collect();
    let (mut a, mut b) = (was.clone(), now.clone());
    a.sort();
    b.sort();
    let captain = before["captain"].as_i64().unwrap_or_default();
    let vice = before["vice"].as_i64().unwrap_or_default();
    let moves_changed = a != b;
    if !moves_changed && captain == after.captain && vice == after.vice {
        return None;
    }
    let named = |m: &[(i64, i64)]| -> Vec<(String, String)> {
        m.iter().map(|&(o, i)| (name(names, o), name(names, i))).collect()
    };
    let ins = |m: &[(i64, i64)], other: &[(i64, i64)]| -> Vec<(String, f64)> {
        m.iter()
            .filter(|(_, i)| !other.iter().any(|(_, j)| j == i))
            .filter_map(|&(_, i)| points(i).map(|p| (name(names, i), p)))
            .collect()
    };
    let changed = |was: i64, now: i64| (name(names, now), (was != now).then(|| name(names, was)));
    Some(PlanChange {
        gw: 0,
        deadline: DateTime::<Utc>::MIN_UTC,
        horizon: 0,
        was_moves: named(&was),
        now_moves: named(&now),
        moves_changed,
        captain: changed(captain, after.captain),
        vice: changed(vice, after.vice),
        now_points: if moves_changed { ins(&now, &was) } else { Vec::new() },
        was_points: if moves_changed { ins(&was, &now) } else { Vec::new() },
    })
}

/// `change_of` for `plan` (its GW, deadline, horizon and xP filled in).
pub fn plan_change(before: Option<&serde_json::Value>, plan: &UserPlan) -> Option<PlanChange> {
    let best = *ranked(&plan.plans).first()?;
    let gws: Vec<i64> = best.gws.iter().map(|g| g.gw).collect();
    let decision = crate::planner::decision_summary(plan);
    let points = |k: i64| horizon_points(&plan.context.xp, k, &gws);
    let mut change = change_of(before, &decision, &plan.context.names(), &points)?;
    change.gw = plan.context.gw;
    change.deadline = plan.context.deadline;
    change.horizon = best.gws.len();
    Some(change)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame() -> XpFrame {
        XpFrame {
            player_key: vec![1, 1, 1, 1, 2],
            gw: vec![6, 7, 7, 8, 6],
            gw_index: vec![6, 7, 7, 8, 6],
            horizon: vec![0, 1, 1, 2, 0],
            xp: vec![Some(2.0), Some(3.0), Some(1.5), Some(f64::NAN), None],
            p_play: None,
            p_play_gw: None,
            e_minutes: None,
        }
    }

    #[test]
    fn horizon_points_add_a_players_rows_in_the_gws() {
        // GW7 is a double (two rows); NaN and missing count 0.
        assert_eq!(horizon_points(&frame(), 1, &[6, 7]), Some(6.5));
        assert_eq!(horizon_points(&frame(), 1, &[6, 7, 8]), Some(6.5));
        assert_eq!(horizon_points(&frame(), 2, &[6]), Some(0.0));
        assert_eq!(horizon_points(&frame(), 3, &[6]), None);
    }

    #[test]
    fn gw_fixtures_group_games_per_gw_and_mark_blanks() {
        let f = |gw, team_key, opponent: &str, home| Fixture {
            gw,
            team_key,
            opponent: opponent.into(),
            home,
        };
        let fixtures = vec![
            f(6, 1, "BRE", true),
            f(6, 1, "LEE", false),
            f(8, 1, "WOL", true),
            f(6, 2, "X", true),
        ];
        assert_eq!(
            gw_fixtures(&fixtures, 1, &[6, 7, 8]),
            vec![
                (6, vec![("BRE".to_string(), true), ("LEE".to_string(), false)]),
                (7, vec![]),
                (8, vec![("WOL".to_string(), true)]),
            ]
        );
    }

    #[test]
    fn plan_change_compares_moves_captain_and_vice() {
        let names: HashMap<i64, String> = [(1, "A"), (2, "B"), (3, "C"), (4, "D"), (5, "E")]
            .into_iter()
            .map(|(k, v)| (k, v.to_string()))
            .collect();
        let before = serde_json::json!({"transfers": [[1, 2]], "captain": 3, "vice": 4});
        let same = crate::planner::Decision { transfers: vec![[1, 2]], captain: 3, vice: 4 };
        assert!(change_of(Some(&before), &same, &names, &|_| None).is_none());
        assert!(change_of(None, &same, &names, &|_| None).is_none());
        let after = crate::planner::Decision { transfers: vec![[1, 5]], captain: 3, vice: 2 };
        let points = |k: i64| Some(k as f64 * 10.0);
        let c = change_of(Some(&before), &after, &names, &points).unwrap();
        assert_eq!(c.was_moves, vec![("A".to_string(), "B".to_string())]);
        assert_eq!(c.now_moves, vec![("A".to_string(), "E".to_string())]);
        assert!(c.moves_changed);
        assert_eq!(c.captain, ("C".to_string(), None));
        assert_eq!(c.vice, ("B".to_string(), Some("D".to_string())));
        assert_eq!(c.now_points, vec![("E".to_string(), 50.0)]);
        assert_eq!(c.was_points, vec![("B".to_string(), 20.0)]);
    }
}
