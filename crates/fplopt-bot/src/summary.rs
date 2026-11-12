//! The facts of a plan for the bot's messages (spec
//! docs/superpowers/specs/2026-11-11-bot-messaging-ux-design.md): plain data, no text.
//! `messages` words them.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use fplopt_optimize::state::{Holding, selling_price};
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
/// whether it is as good as plan 1 (the solver can't tell them apart, or the gains read the
/// same).
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
    /// The most free transfers one can bank.
    pub ft_cap: i64,
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

/// What the pairing of a GW's transfers needs about players: position, sale and buy price.
/// Falls back from the pool to the squad's holdings, so a player outside the pool still has
/// a position and a price.
pub struct Lookup<'a> {
    /// player_key -> (price, team_key, element_type)
    pub pool: HashMap<i64, (i64, i64, i64)>,
    pub holdings: &'a [Holding],
    pub sell_on_fee: f64,
}

impl Lookup<'_> {
    fn holding(&self, key: i64) -> Option<&Holding> {
        self.holdings.iter().find(|h| h.player_key == key)
    }

    /// 1..=4, or 0 when neither the pool nor the squad knows him.
    pub fn position(&self, key: i64) -> i64 {
        match self.pool.get(&key) {
            Some(p) => p.2,
            None => self.holding(key).map_or(0, |h| h.element_type),
        }
    }

    /// His price to buy: the pool's, else the squad's price (0 if unknown).
    pub fn buy_price(&self, key: i64) -> i64 {
        match self.pool.get(&key) {
            Some(p) => p.0,
            None => self.holding(key).map_or(0, |h| h.price),
        }
    }

    /// What selling him earns: only for a player in the squad (0 otherwise).
    pub fn sell_price(&self, key: i64) -> i64 {
        self.holding(key).map_or(0, |h| selling_price(h, self.sell_on_fee))
    }

    /// His price to sort sales by: the sale price, else (bought earlier in the plan) the buy
    /// price.
    fn sort_price(&self, key: i64) -> i64 {
        match self.holding(key) {
            Some(h) => selling_price(h, self.sell_on_fee),
            None => self.buy_price(key),
        }
    }
}

/// `g`'s transfers as (out_key, in_key), re-paired within each position: the optimizer pairs
/// by player key, which is arbitrary, so the dearest sale goes with the dearest buy, and so on
/// (ties by name). Positions in lineup order.
pub fn pair_transfers(
    g: &GwPlan,
    lookup: &Lookup,
    names: &HashMap<i64, String>,
) -> Vec<(i64, i64)> {
    let mut out = Vec::new();
    let mut positions: Vec<i64> = g
        .transfers
        .iter()
        .flat_map(|t| [lookup.position(t.out_key), lookup.position(t.in_key)])
        .collect();
    positions.sort_unstable();
    positions.dedup();
    for pos in positions {
        let mut outs: Vec<i64> =
            g.transfers.iter().map(|t| t.out_key).filter(|&k| lookup.position(k) == pos).collect();
        let mut ins: Vec<i64> =
            g.transfers.iter().map(|t| t.in_key).filter(|&k| lookup.position(k) == pos).collect();
        outs.sort_by(|&a, &b| {
            lookup
                .sort_price(b)
                .cmp(&lookup.sort_price(a))
                .then_with(|| name(names, a).cmp(&name(names, b)))
        });
        ins.sort_by(|&a, &b| {
            lookup
                .buy_price(b)
                .cmp(&lookup.buy_price(a))
                .then_with(|| name(names, a).cmp(&name(names, b)))
        });
        out.extend(outs.into_iter().zip(ins));
    }
    out
}

fn moves_of(g: &GwPlan, lookup: &Lookup, names: &HashMap<i64, String>) -> Vec<(String, String)> {
    pair_transfers(g, lookup, names)
        .into_iter()
        .map(|(o, i)| (name(names, o), name(names, i)))
        .collect()
}

/// Whether two gains read the same at one decimal (both must exist).
fn same_gain(a: Option<f64>, b: Option<f64>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => (a * 10.0).round() == (b * 10.0).round(),
        _ => false,
    }
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
    let lookup = Lookup {
        pool: context
            .pool
            .iter()
            .map(|p| (p.player_key, (p.price, p.team_key, p.element_type)))
            .collect(),
        holdings: &plan.state.holdings,
        sell_on_fee: plan.rules.sell_on_fee,
    };
    let next3: Vec<i64> =
        context.gameweeks.iter().map(|g| g.0).filter(|&g| g >= first.gw).take(3).collect();
    let moves = pair_transfers(first, &lookup, &names)
        .into_iter()
        .map(|(out_key, in_key)| {
            let team_key = lookup
                .pool
                .get(&in_key)
                .map(|p| p.1)
                .or_else(|| lookup.holding(in_key).map(|h| h.team_key))
                .unwrap_or(0);
            Move {
                out_name: name(&names, out_key),
                in_name: name(&names, in_key),
                sell_price: lookup.sell_price(out_key),
                buy_price: lookup.buy_price(in_key),
                out_points: points(out_key, &gws),
                in_points: points(in_key, &gws),
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
                .filter(|&&k| lookup.position(k) == pos)
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
        ft_cap: cap,
        team,
        bench: first.bench.iter().map(|&k| name(&names, k)).collect(),
        alternatives: order[1..]
            .iter()
            .map(|p| Alternative {
                moves: moves_of(p.first(), &lookup, &names),
                hit_points: plan.rules.hit_cost * p.first().hits,
                gain: p.gain_vs_roll,
                tied: tied(best, p) || same_gain(best.gain_vs_roll, p.gain_vs_roll),
            })
            // The roll line covers "no transfers".
            .filter(|a| roll.is_none() || !a.moves.is_empty())
            .collect(),
        roll,
        later: best.gws[1..]
            .iter()
            .flat_map(|g| moves_of(g, &lookup, &names).into_iter().map(move |(o, i)| (g.gw, o, i)))
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
    /// Points the plan's hits cost (0 without hits).
    pub hit_points: i64,
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
        hit_points: 0,
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
    change.hit_points = plan.rules.hit_cost * best.first().hits;
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

    fn holding(key: i64, element_type: i64, purchase: i64, price: i64) -> Holding {
        Holding { player_key: key, element_type, team_key: 1, purchase_price: purchase, price }
    }

    fn gw_with(transfers: &[(i64, i64)]) -> GwPlan {
        GwPlan {
            gw: 6,
            gw_index: 6,
            horizon: 0,
            transfers: transfers
                .iter()
                .map(|&(out_key, in_key)| fplopt_optimize::state::Transfer { out_key, in_key })
                .collect(),
            starters: vec![],
            bench: vec![],
            captain: 0,
            vice: 0,
            chip: None,
            xp: 0.0,
            bench_xp: 0.0,
            objective: 0.0,
            hits: 0,
            bank: 0,
            ft: 0,
        }
    }

    fn names_of(pairs: &[(i64, &str)]) -> HashMap<i64, String> {
        pairs.iter().map(|&(k, n)| (k, n.to_string())).collect()
    }

    #[test]
    fn transfers_are_paired_by_price_within_a_position() {
        // Two midfield swaps, paired by key the cheap sale goes with the dear buy. Sales: 1
        // sells at 12.0, 2 at 5.0; buys: 11 costs 5.5, 12 costs 13.0.
        let holdings = vec![holding(1, 3, 120, 120), holding(2, 3, 50, 50), holding(3, 4, 80, 80)];
        let pool = [(11, (55, 9, 3)), (12, (130, 9, 3)), (13, (90, 9, 4))].into_iter().collect();
        let lookup = Lookup { pool, holdings: &holdings, sell_on_fee: 0.5 };
        let names = names_of(&[(1, "A"), (2, "B"), (3, "F"), (11, "X"), (12, "Y"), (13, "G")]);
        let g = gw_with(&[(1, 11), (2, 12), (3, 13)]);
        assert_eq!(pair_transfers(&g, &lookup, &names), vec![(1, 12), (2, 11), (3, 13)]);
        assert_eq!(
            moves_of(&g, &lookup, &names),
            vec![
                ("A".to_string(), "Y".to_string()),
                ("B".to_string(), "X".to_string()),
                ("F".to_string(), "G".to_string())
            ]
        );
    }

    #[test]
    fn players_outside_the_pool_fall_back_to_the_squad() {
        // Out player 2 is not in the pool (position and price from his holding); ties in price
        // go by name.
        let holdings = vec![holding(1, 2, 60, 60), holding(2, 2, 60, 60)];
        let pool = [(11, (60, 9, 2)), (12, (60, 9, 2))].into_iter().collect();
        let lookup = Lookup { pool, holdings: &holdings, sell_on_fee: 0.5 };
        assert_eq!(lookup.position(2), 2);
        assert_eq!(lookup.buy_price(2), 60);
        assert_eq!(lookup.position(99), 0);
        let names = names_of(&[(1, "B"), (2, "A"), (11, "D"), (12, "C")]);
        let g = gw_with(&[(1, 11), (2, 12)]);
        // Outs by name A(2), B(1); ins by name C(12), D(11).
        assert_eq!(pair_transfers(&g, &lookup, &names), vec![(2, 12), (1, 11)]);
    }

    #[test]
    fn same_gain_compares_the_shown_decimal() {
        assert!(same_gain(Some(2.44), Some(2.36)));
        assert!(!same_gain(Some(2.44), Some(2.3)));
        assert!(!same_gain(None, None));
        assert!(!same_gain(Some(1.0), None));
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
        assert_eq!(c.hit_points, 0);
        assert_eq!(c.now_points, vec![("E".to_string(), 50.0)]);
        assert_eq!(c.was_points, vec![("B".to_string(), 20.0)]);
    }
}
