//! Telegram message texts (Python `fplopt.bot.messages`; Phase 7 plan §5): Telegram HTML
//! (`<b>` and escaped text only), short lines for a phone. `recommendation_text` is the full
//! recommendation (alerts and `/plan`); `split_message` cuts a text to Telegram's 4,096
//! character limit at line breaks. The texts are Python's, byte for byte.

use std::collections::{BTreeSet, HashMap};

use fplopt_optimize::GwPlan;
use fplopt_optimize::chips::chip_value;
use fplopt_optimize::entry::{SquadSpec, season_label};
use serde_json::Value;

use crate::context::PlanContext;
use crate::planner::{
    CHIP_LABELS, ChipOptions, Decision, UserPlan, chip_label, chips_left, ranked,
};
use crate::util::{escape as e, py_g, uk_time};

pub const MAX_MESSAGE: usize = 4096;
const POSITIONS: [(i64, &str); 4] = [(1, "GK"), (2, "DEF"), (3, "MID"), (4, "FWD")];
const ARROW: &str = "→";

/// `names[key]` (Python raises KeyError on a missing player; here the update fails the same
/// way: the admin is alerted).
fn name(names: &HashMap<i64, String>, key: i64) -> &str {
    names.get(&key).map(String::as_str).unwrap_or_else(|| panic!("no name for player_key {key}"))
}

fn moves(g: &GwPlan, names: &HashMap<i64, String>, hit_cost: i64) -> String {
    let moves: Vec<String> = g
        .transfers
        .iter()
        .map(|t| format!("{} {ARROW} {}", name(names, t.out_key), name(names, t.in_key)))
        .collect();
    let moves = moves.join(", ");
    let hit = if g.hits != 0 { format!(" (−{} hit)", hit_cost * g.hits) } else { String::new() };
    e(if moves.is_empty() { "no transfers" } else { &moves }) + &hit
}

/// Plans whose objectives differ by less than the solver's gap: equally good as far as the
/// solver can tell (PLAN §8 live finding).
fn tied(a: &fplopt_optimize::Plan, b: &fplopt_optimize::Plan) -> bool {
    let gap = a.mip_gap.max(b.mip_gap) * a.total_objective().abs();
    (a.total_objective() - b.total_objective()).abs() <= gap
}

fn lineup(g: &GwPlan, plan: &UserPlan, names: &HashMap<i64, String>) -> Vec<String> {
    let positions: HashMap<i64, i64> =
        plan.context.pool.iter().map(|p| (p.player_key, p.element_type)).collect();
    let mark = |k: i64| {
        if k == g.vice {
            " (V)"
        } else if k == g.captain {
            " (C)"
        } else {
            ""
        }
    };
    let mut lines = Vec::new();
    for (pos, label) in POSITIONS {
        let players: Vec<String> = g
            .starters
            .iter()
            .filter(|k| positions.get(k).copied().unwrap_or(0) == pos)
            .map(|&k| format!("{}{}", name(names, k), mark(k)))
            .collect();
        if !players.is_empty() {
            lines.push(format!("{label} {}", e(&players.join(", "))));
        }
    }
    let bench: Vec<&str> = g.bench.iter().map(|&k| name(names, k)).collect();
    lines.push(format!("Bench {}", e(&bench.join(", "))));
    lines
}

/// The recommendation for `plan` (module docs); `title` replaces the first line's lead,
/// `changes` (from `changes_text`) go right under it.
pub fn recommendation_text(plan: &UserPlan, title: Option<&str>, changes: &[String]) -> String {
    let (context, spec, plans) = (&plan.context, &plan.spec, &plan.plans);
    let names = context.names();
    let gw = context.gw;
    let lead = title.map_or_else(|| format!("GW{gw} plan"), str::to_string);
    let mut lines =
        vec![format!("<b>{}</b> · deadline {} (UK)", e(&lead), uk_time(context.deadline))];
    lines.extend(changes.iter().map(|c| e(c)));
    let chips = chips_left(&plan.state, &plan.rules).join(", ");
    lines.extend([
        format!(
            "Team {} · bank £{:.1}m · {} FT",
            plan.entry_id,
            spec.bank as f64 / 10.0,
            spec.free_transfers
        ),
        format!(
            "Chips you can play this GW: {}",
            e(if chips.is_empty() { "none" } else { &chips })
        ),
        format!(
            "xP data from {} (UK)",
            uk_time(context.bootstrap_at.unwrap_or(context.created_at))
        ),
        prices_line(context),
        String::new(),
    ]);
    if plans.plans.is_empty() {
        lines.push("The solver found no plan. The admin has been told.".into());
        return lines.join("\n");
    }
    let order = ranked(plans);
    let best = order[0];
    let roll = plans.roll.as_ref();
    let hit_cost = plan.rules.hit_cost;
    let gain =
        best.gain_vs_roll.map_or(String::new(), |g| format!(" · {g:+.1} xP vs saving the FT"));
    lines.push(format!("<b>Plan 1</b>{gain} (next {} GWs)", best.gws.len()));
    for g in &best.gws {
        lines.push(format!("GW{}: {}", g.gw, moves(g, &names, hit_cost)));
    }
    let first = best.first();
    lines.push(format!(
        "Captain {} · vice {}",
        e(name(&names, first.captain)),
        e(name(&names, first.vice))
    ));
    lines.push(String::new());
    lines.push(format!("<b>GW{gw} team</b> (plan 1)"));
    lines.extend(lineup(first, plan, &names));
    lines.push(String::new());
    for (i, other) in order.iter().enumerate().skip(1) {
        let gain = other.gain_vs_roll.map_or(String::new(), |g| format!(" ({g:+.1})"));
        lines.push(format!(
            "<b>Plan {}</b>{gain}: GW{gw} {}",
            i + 1,
            moves(other.first(), &names, hit_cost)
        ));
    }
    // Python's `roll is not p` holds for every plan: `optimize` returns fresh copies.
    if let Some(roll) = roll {
        lines.push(format!(
            "<b>Save the FT</b>: GW{gw} xP {:.1} (plan 1: {:.1})",
            roll.first().xp,
            first.xp
        ));
    }
    lines.push(String::new());
    let mut ties = vec![1];
    ties.extend(
        order.iter().enumerate().skip(1).filter(|(_, o)| tied(best, o)).map(|(i, _)| i + 1),
    );
    if ties.len() > 1 {
        let head: Vec<String> = ties[..ties.len() - 1].iter().map(ToString::to_string).collect();
        lines.push(format!(
            "Plans {} and {} are within the solver's precision of each other: treat them as \
             equally good.",
            head.join(", "),
            ties[ties.len() - 1]
        ));
    }
    if let Some(roll) = roll
        && tied(best, roll)
    {
        lines.push("Saving the FT is as good as plan 1 within the solver's precision.".into());
    }
    if !best.optimal() {
        lines.push(format!(
            "The solver stopped early ({}): plan 1 may not be the best.",
            e(&best.status)
        ));
    }
    lines.push("Chips are your call: the system doesn't plan them yet.".into());
    lines.push(format!(
        "Made transfers for GW{gw} already? FPL doesn't show them before the deadline: tell the \
         bot with /transfer."
    ));
    let notes: Vec<&String> =
        spec.notes.iter().filter(|n| !n.starts_with("transfers or a chip made for")).collect();
    if !notes.is_empty() {
        lines.push(String::new());
        lines.push("Notes:".into());
        lines.extend(notes.iter().map(|n| format!("· {}", e(n))));
    }
    lines.join("\n")
}

/// Where the prices came from: FPL live (when) or the data's snapshot.
pub fn prices_line(context: &PlanContext) -> String {
    match context.prices_at {
        Some(at) => format!("Prices live from FPL at {} (UK)", uk_time(at)),
        None => format!(
            "Prices from the data's snapshot ({} UK): FPL couldn't be reached",
            uk_time(context.bootstrap_at.unwrap_or(context.created_at))
        ),
    }
}

fn pairs(transfers: &Value) -> Vec<(i64, i64)> {
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

/// What changed in plan 1's decision for this GW (`planner::decision_summary`) since the last
/// alert (`before`: its stored decision); empty if nothing did.
pub fn changes_text(
    before: Option<&Value>,
    after: &Decision,
    names: &HashMap<i64, String>,
) -> Vec<String> {
    let Some(before) = before else { return Vec::new() };
    let moves = |pairs: &[(i64, i64)]| {
        let text: Vec<String> = pairs
            .iter()
            .map(|&(o, i)| format!("{} {ARROW} {}", name(names, o), name(names, i)))
            .collect();
        if text.is_empty() { "no transfers".to_string() } else { text.join(", ") }
    };
    let was = pairs(&before["transfers"]);
    let now: Vec<(i64, i64)> = after.transfers.iter().map(|p| (p[0], p[1])).collect();
    let (mut a, mut b) = (was.clone(), now.clone());
    a.sort();
    b.sort();
    let mut out = Vec::new();
    if a != b {
        out.push(format!("Transfers changed: was {}, now {}.", moves(&was), moves(&now)));
    }
    let captain = before["captain"].as_i64().unwrap_or_default();
    if captain != after.captain {
        out.push(format!(
            "Captain changed: was {}, now {}.",
            name(names, captain),
            name(names, after.captain)
        ));
    }
    let vice = before["vice"].as_i64().unwrap_or_default();
    if vice != after.vice {
        out.push(format!(
            "Vice changed: was {}, now {}.",
            name(names, vice),
            name(names, after.vice)
        ));
    }
    out
}

/// The imported squad, for /register's reply.
pub fn squad_text(context: &PlanContext, spec: &SquadSpec, entry_id: i64) -> String {
    let names: HashMap<i64, &str> =
        context.players.iter().map(|p| (p.element_id, p.web_name.as_str())).collect();
    let players: Vec<String> = spec
        .players
        .iter()
        .map(|p| {
            names.get(&p.element_id).map_or_else(|| p.element_id.to_string(), |n| n.to_string())
        })
        .collect();
    let used: Vec<String> = spec.chips_used.iter().map(|(n, gw)| format!("{n} (GW{gw})")).collect();
    let used = used.join(", ");
    [
        format!("<b>Team {entry_id}</b> before GW{} ({})", spec.gw, e(&season_label(spec.season))),
        format!("Squad: {}", e(&players.join(", "))),
        format!("Bank £{:.1}m · {} free transfers", spec.bank as f64 / 10.0, spec.free_transfers),
        format!("Chips played: {}", e(if used.is_empty() { "none" } else { &used })),
    ]
    .join("\n")
}

/// `bboost@gw9+freehit@gw11` -> 'Bench Boost GW9 + Free Hit GW11'; `none` -> 'no chip'.
pub fn scenario_label(label: &str) -> String {
    if label == "none" {
        return "no chip".into();
    }
    label
        .split('+')
        .map(|part| {
            let (name, gw) = part.split_once("@gw").unwrap_or((part, ""));
            format!("{} GW{gw}", chip_label(name))
        })
        .collect::<Vec<_>>()
        .join(" + ")
}

/// `/chips` (information only): the chips still unused, the best chip plan over the horizon
/// and the solved scenarios against holding every chip.
pub fn chips_text(options: &ChipOptions, shown: usize) -> String {
    let (context, rules, params) = (&options.context, &options.rules, &options.params);
    let gw_map = context.gw_index_map();
    let gw_of: HashMap<i64, i64> = gw_map.iter().map(|(g, i)| (*i, *g)).collect();
    let used: BTreeSet<i64> = options.state.chips_used.iter().map(|(cid, _)| *cid).collect();
    let mut windows: Vec<_> = rules.chips.iter().collect();
    windows.sort_by_key(|w| (w.start, w.chip_id));
    let unused: Vec<String> = windows
        .iter()
        .filter(|w| !used.contains(&w.chip_id) && w.stop >= context.gw_index)
        .map(|w| {
            format!(
                "{} (until GW{})",
                chip_label(&w.name),
                gw_of.get(&w.stop).copied().unwrap_or(w.stop)
            )
        })
        .collect();
    let objectives = &options.plans.scenario_objectives;
    let objective =
        |label: &str| objectives.iter().rev().find(|(l, _)| l == label).map(|(_, v)| *v);
    let base = objective("none").unwrap_or(0.0);
    let best = options.plans.best();
    let horizon = best.gws.len();
    let last = best.gws.last().expect("a plan has GWs").gw;
    let unused = unused.join(", ");
    let mut best_line =
        format!("<b>Best over the next {horizon} GWs:</b> {}", e(&scenario_label(&best.scenario)));
    if best.scenario != "none" {
        best_line += &format!(", {:+.1} xP vs holding every chip", best.total_objective() - base);
    }
    let mut lines = vec![
        format!("<b>GW{} chips</b> · information only", context.gw),
        if params.chip_value_left.is_some() {
            "Tested on 2023-25: this planner beat no chips, but not simple rules convincingly."
                .to_string()
        } else {
            "The chip planner isn't tested yet: weigh this against your own judgement.".to_string()
        },
        format!("Unused chips: {}", e(if unused.is_empty() { "none" } else { &unused })),
        prices_line(context),
        String::new(),
        best_line,
    ];
    if best.scenario != "none" && best.first().chip.is_some() {
        let names = context.names();
        lines.push(format!(
            "This GW with it: {}; captain {}",
            moves(best.first(), &names, rules.hit_cost),
            e(name(&names, best.first().captain))
        ));
    }
    if !options.singles.is_empty() {
        lines.push(String::new());
        lines.push("Each chip on its own, in its most promising GW (xP vs no chip):".into());
        for (chip, _) in CHIP_LABELS {
            if let Some((label, value)) = options.single(chip) {
                lines.push(format!("· {}: {:+.1}", e(&scenario_label(label)), value - base));
            }
        }
    }
    // dict(scenario_objectives), sorted by objective (stable)
    let mut unique: Vec<(String, f64)> = Vec::new();
    for (label, value) in objectives {
        match unique.iter_mut().find(|(l, _)| l == label) {
            Some(slot) => slot.1 = *value,
            None => unique.push((label.clone(), *value)),
        }
    }
    let mut ranked_scenarios = unique.clone();
    ranked_scenarios.sort_by(|a, b| (-a.1).partial_cmp(&-b.1).unwrap_or(std::cmp::Ordering::Equal));
    lines.push(String::new());
    lines.push("Options the solver checked (xP vs no chip):".into());
    for (label, value) in ranked_scenarios.iter().take(shown) {
        lines.push(format!("· {}: {:+.1}", e(&scenario_label(label)), value - base));
    }
    let mut solved: BTreeSet<&str> = unique.iter().map(|(l, _)| l.as_str()).collect();
    solved.extend(options.singles.iter().map(|(_, (l, _))| l.as_str()));
    let unsolved: BTreeSet<&str> = options
        .plans
        .scenario_bounds
        .iter()
        .map(|(l, _)| l.as_str())
        .filter(|l| !solved.contains(l))
        .collect();
    let not_optimal: BTreeSet<&str> =
        options.plans.search.not_optimal.iter().map(|(l, _)| l.as_str()).collect();
    let stopped = unsolved.intersection(&not_optimal).count();
    let skipped = unsolved.len() - stopped;
    if ranked_scenarios.len() > shown {
        lines.push(format!("· … {} more, all lower", ranked_scenarios.len() - shown));
    }
    if skipped > 0 {
        lines.push(format!(
            "{skipped} other options were ruled out: provably no better than the best."
        ));
    }
    if stopped > 0 {
        lines.push(format!(
            "{stopped} option(s) not settled: the solver hit its time limit before finding a plan \
             for them."
        ));
    }
    let how = if params.chip_value_left.is_some() {
        let last_index =
            gw_map.get(&last).copied().unwrap_or(context.gw_index + best.gws.len() as i64 - 1);
        let mut held = Vec::new();
        for w in &windows {
            if used.contains(&w.chip_id) || w.stop <= last_index || w.start > last_index + 1 {
                continue;
            }
            let value = chip_value(params, &w.name, w.stop - last_index);
            held.push(format!("{} {value:.1}", chip_label(&w.name)));
        }
        let held = held.join(", ");
        format!(
            "How to read it: a chip still held after GW{last} counts as worth its estimated best \
             use in the rest of its window ({} xP, from backtests); a chip whose window ends \
             sooner is worth nothing if held. So a chip wins only if it beats holding it. The \
             bot's plans don't play chips.",
            e(if held.is_empty() { "none" } else { &held })
        )
    } else {
        let values: Vec<String> = CHIP_LABELS
            .iter()
            .filter_map(|(chip, label)| {
                params.chip_value.get(*chip).map(|v| format!("{label} {}", py_g(*v)))
            })
            .collect();
        format!(
            "How to read it: a chip still held after GW{last} counts as worth a guessed value ({} \
             xP); a chip whose window ends sooner is worth nothing if held. So a chip wins only if \
             it beats that guess. The bot's plans don't play chips.",
            values.join(", ")
        )
    };
    lines.push(String::new());
    lines.push(how);
    lines.join("\n")
}

/// `text` in pieces of at most `limit` characters, cut at line breaks (a single longer line
/// is cut hard).
pub fn split_message(text: &str, limit: usize) -> Vec<String> {
    let mut pieces: Vec<String> = Vec::new();
    let mut current = String::new();
    for line in text.split('\n') {
        let mut line: Vec<char> = line.chars().collect();
        while line.len() > limit {
            if !current.is_empty() {
                pieces.push(std::mem::take(&mut current));
            }
            pieces.push(line[..limit].iter().collect());
            line.drain(..limit);
        }
        let line: String = line.into_iter().collect();
        let candidate =
            if current.is_empty() { line.clone() } else { format!("{current}\n{line}") };
        if candidate.chars().count() > limit {
            pieces.push(std::mem::replace(&mut current, line));
        } else {
            current = candidate;
        }
    }
    if !current.is_empty() || pieces.is_empty() {
        pieces.push(current);
    }
    pieces
}

#[cfg(test)]
mod tests {
    use super::*;

    // Python test_split_message.
    #[test]
    fn split() {
        assert_eq!(split_message("short", MAX_MESSAGE), ["short"]);
        assert_eq!(split_message("", MAX_MESSAGE), [""]);
        let lines: Vec<String> = (0..100).map(|i| format!("line {i:03}")).collect();
        let pieces = split_message(
            &lines.join(
                "
",
            ),
            50,
        );
        assert!(pieces.iter().all(|p| p.chars().count() <= 50));
        assert_eq!(
            pieces.join(
                "
"
            ),
            lines.join(
                "
"
            )
        );
        let long = "x".repeat(120);
        assert_eq!(
            split_message(
                &format!(
                    "a
{long}
b"
                ),
                50
            ),
            [
                "a".to_string(),
                "x".repeat(50),
                "x".repeat(50),
                format!(
                    "{}
b",
                    "x".repeat(20)
                )
            ]
        );
        assert_eq!(
            split_message(
                "ab
éééééé",
                4
            ),
            ["ab", "éééé", "éé"]
        ); // characters, not bytes
    }

    // Python test_scenario_label.
    #[test]
    fn labels() {
        assert_eq!(scenario_label("none"), "no chip");
        assert_eq!(scenario_label("bboost@gw9"), "Bench Boost GW9");
        assert_eq!(scenario_label("wildcard@gw5+3xc@gw9"), "Wildcard GW5 + Triple Captain GW9");
    }

    // Python test_changes_text.
    #[test]
    fn changes() {
        let names: HashMap<i64, String> = [(1, "A"), (2, "B"), (3, "C"), (4, "D"), (5, "E")]
            .into_iter()
            .map(|(k, v)| (k, v.to_string()))
            .collect();
        let before = serde_json::json!({"transfers": [[1, 2]], "captain": 3, "vice": 4});
        let same = Decision { transfers: vec![[1, 2]], captain: 3, vice: 4 };
        assert!(changes_text(None, &same, &names).is_empty());
        assert!(changes_text(Some(&before), &same, &names).is_empty());
        let after = Decision { transfers: vec![[1, 5]], captain: 4, vice: 3 };
        assert_eq!(
            changes_text(Some(&before), &after, &names),
            [
                "Transfers changed: was A → B, now A → E.",
                "Captain changed: was C, now D.",
                "Vice changed: was D, now C."
            ]
        );
    }
}
