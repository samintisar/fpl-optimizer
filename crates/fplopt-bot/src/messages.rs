//! Telegram message texts (Python `fplopt.bot.messages`; Phase 7 plan §5): Telegram HTML
//! (`<b>` and escaped text only), short lines for a phone. `plan_text` is the plan message
//! (alerts and `/plan`); `split_message` cuts a text to Telegram's 4,096
//! character limit at line breaks. The texts are Python's, byte for byte.

use std::collections::{BTreeSet, HashMap};

use fplopt_optimize::GwPlan;
use fplopt_optimize::chips::chip_value;
use fplopt_optimize::entry::{SquadSpec, season_label};

use crate::chip_calls::ChipCalls;
use crate::context::PlanContext;
use crate::planner::{CHIP_LABELS, ChipOptions, UserPlan, chip_label};
use crate::summary::{PlanChange, PlanSummary, plan_summary};
use crate::util::{escape as e, py_g, uk_time};

pub const MAX_MESSAGE: usize = 4096;
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

fn money(tenths: i64) -> String {
    format!("£{:.1}m", tenths as f64 / 10.0)
}

fn free_transfers(n: i64) -> String {
    if n == 1 { "1 free transfer".into() } else { format!("{n} free transfers") }
}

/// "this GW" or "over the next N GWs".
pub fn over(horizon: usize) -> String {
    if horizon == 1 { "this GW".into() } else { format!("over the next {horizon} GWs") }
}

/// "sell A → buy B, sell C → buy D" (escaped); "no transfers" when empty.
fn sell_buy(moves: &[(String, String)]) -> String {
    if moves.is_empty() {
        return "no transfers".into();
    }
    let text: Vec<String> =
        moves.iter().map(|(o, i)| format!("sell {} {ARROW} buy {}", e(o), e(i))).collect();
    text.join(", ")
}

/// `text` with its first letter in upper case.
fn capitalized(text: &str) -> String {
    let mut chars = text.chars();
    chars.next().map_or(String::new(), |c| c.to_uppercase().collect::<String>() + chars.as_str())
}

fn games(fixtures: &[(i64, Vec<(String, bool)>)]) -> String {
    let per_gw: Vec<String> = fixtures
        .iter()
        .map(|(_, games)| {
            if games.is_empty() {
                return "no game".to_string();
            }
            let g: Vec<String> = games
                .iter()
                .map(|(o, home)| format!("{} ({})", e(o), if *home { "H" } else { "A" }))
                .collect();
            g.join(" + ")
        })
        .collect();
    per_gw.join(", ")
}

/// The plan message (`/plan`, the 24-hour alert).
pub fn render_plan(s: &PlanSummary) -> String {
    let mut lines = vec![
        format!("<b>GW{} plan</b> · deadline {} UK", s.gw, uk_time(s.deadline)),
        String::new(),
    ];
    if s.moves.is_empty() {
        lines.push(format!(
            "<b>This week's moves:</b> none, save your transfer ({} next GW)",
            free_transfers(s.free_transfers_next)
        ));
    } else {
        let hit = if s.hit_points > 0 {
            format!(" (costs a {}-pt hit)", s.hit_points)
        } else {
            String::new()
        };
        lines.push(format!("<b>This week's moves</b>{hit}"));
        for m in &s.moves {
            lines.push(format!(
                "Sell {} ({}) {ARROW} buy {} ({})",
                e(&m.out_name),
                money(m.sell_price),
                e(&m.in_name),
                money(m.buy_price)
            ));
            lines.push(format!(
                "  {:.1} {ARROW} {:.1} pts {}",
                m.out_points,
                m.in_points,
                over(s.horizon)
            ));
            if let Some(f) = &m.in_fixtures {
                lines.push(format!("  {}'s next games: {}", e(&m.in_name), games(f)));
            }
        }
    }
    lines.push(format!(
        "Captain {} ({:.1} pts expected) · vice {} ({:.1})",
        e(&s.captain.name),
        s.captain.points,
        e(&s.vice.name),
        s.vice.points
    ));
    if !s.moves.is_empty() {
        if let Some(g) = s.gain_vs_roll {
            lines.push(format!("Gain: about {g:+.1} pts vs saving your free transfer"));
        }
        lines.push(format!(
            "Afterwards: {} in the bank, {} next GW",
            money(s.bank_after),
            free_transfers(s.free_transfers_next)
        ));
    }
    lines.push(String::new());
    lines.push("<b>Your team</b>".into());
    for (label, names) in &s.team {
        lines.push(format!("{label} {}", e(&names.join(", "))));
    }
    lines.push(format!("Bench {}", e(&s.bench.join(", "))));
    if !s.alternatives.is_empty() || s.roll.is_some() {
        lines.push(String::new());
        lines.push("<b>Close alternatives</b>".into());
        for a in &s.alternatives {
            let hit = if a.hit_points > 0 {
                format!(" (−{} pts hit)", a.hit_points)
            } else {
                String::new()
            };
            let gain = a.gain.map_or(String::new(), |g| format!(": {g:+.1} pts"));
            let tie = if a.tied { " (just as good)" } else { "" };
            lines.push(format!("· {}{hit}{gain}{tie}", capitalized(&sell_buy(&a.moves))));
        }
        if let Some((ft, tie)) = s.roll {
            lines.push(if tie {
                format!("· Save your transfer: just as good ({} next GW)", free_transfers(ft))
            } else {
                format!("· Save your transfer: {} next GW", free_transfers(ft))
            });
        }
    }
    if !s.later.is_empty() {
        let mut by_gw: Vec<(i64, Vec<(String, String)>)> = Vec::new();
        for (gw, o, i) in &s.later {
            match by_gw.last_mut() {
                Some((g, moves)) if g == gw => moves.push((o.clone(), i.clone())),
                _ => by_gw.push((*gw, vec![(o.clone(), i.clone())])),
            }
        }
        let parts: Vec<String> =
            by_gw.iter().map(|(gw, moves)| format!("GW{gw} {}", sell_buy(moves))).collect();
        lines.push(String::new());
        lines.push(format!("Looking ahead (may change): {}", parts.join("; ")));
    }
    lines.push(String::new());
    if s.stopped_early {
        lines.push(
            "This plan was found under a time limit; there may be a slightly better one.".into(),
        );
    }
    if let Some(at) = s.prices_fallback {
        lines.push(format!("Couldn't reach FPL: prices from {} UK.", uk_time(at)));
    }
    if s.told_transfers {
        lines.push("Includes the transfers you told me about (/transfer list).".into());
    }
    lines.push(
        "Made a transfer already? FPL hides it until the deadline, so tell me with /transfer."
            .into(),
    );
    lines.join("\n")
}

/// `render_plan` of `plan`'s summary, or the no-plan text.
pub fn plan_text(plan: &UserPlan) -> String {
    match plan_summary(plan) {
        Some(s) => render_plan(&s),
        None => format!(
            "<b>GW{} plan</b> · deadline {} UK\n\nI couldn't find a plan this time. The admin has been told.",
            plan.context.gw,
            uk_time(plan.context.deadline)
        ),
    }
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

/// The re-run's message when only the chip call changed (after the chip section).
pub const UNCHANGED: &str = "Your transfers and captain are unchanged: /plan";

fn plain_moves(moves: &[(String, String)]) -> String {
    if moves.is_empty() {
        return "no transfers".into();
    }
    let parts: Vec<String> =
        moves.iter().map(|(o, i)| format!("sell {}, buy {}", e(o), e(i))).collect();
    parts.join("; ")
}

/// The "plan changed" update (spec, *Update alert*).
pub fn update_text(c: &PlanChange) -> String {
    let mut lines =
        vec![format!("<b>GW{} plan changed</b> · deadline {} UK", c.gw, uk_time(c.deadline))];
    if c.moves_changed {
        lines.push(format!(
            "Now: {} (was: {})",
            plain_moves(&c.now_moves),
            plain_moves(&c.was_moves)
        ));
    }
    for (label, (now, was)) in [("Captain", &c.captain), ("Vice", &c.vice)] {
        lines.push(match was {
            Some(was) => format!("{label}: {} (was {})", e(now), e(was)),
            None => format!("{label}: {} (unchanged)", e(now)),
        });
    }
    if !c.now_points.is_empty() || !c.was_points.is_empty() {
        let list = |p: &[(String, f64)]| {
            let parts: Vec<String> = p.iter().map(|(n, x)| format!("{} {x:.1}", e(n))).collect();
            parts.join(", ")
        };
        let what = if c.horizon == 1 {
            "pts this GW".to_string()
        } else {
            format!("pts over the next {} GWs", c.horizon)
        };
        let mut parts = Vec::new();
        if !c.now_points.is_empty() {
            parts.push(format!("now {}", list(&c.now_points)));
        }
        if !c.was_points.is_empty() {
            parts.push(format!("before {}", list(&c.was_points)));
        }
        lines.push(format!("Why ({what}): {}", parts.join("; ")));
    }
    lines.push("Full plan and team: /plan".into());
    lines.join("\n")
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
            let value = chip_value(params, &w.name, w.stop - last_index, 1, false);
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

/// The alert's chip section (Telegram HTML; the caller adds the blank line after it).
///
/// The headline is P's chip when P plays one. A chip H's last-chance rule forces (more chips
/// left in a window than GWs) never overrides it, since P may play another chip of that window
/// now and the forced one next GW: the section then says how many chips and GWs are left.
/// When P holds, the forced chip is the headline.
pub fn chip_section(calls: &ChipCalls) -> String {
    let forced = calls.forced.as_deref();
    let headline = calls.planner.as_deref().or(forced);
    // The forced chip's window: open chips ending with it, GWs to its end.
    let window = forced.and_then(|f| {
        let last = calls.open.iter().find(|o| o.name == f)?.last_gw;
        let left = calls.open.iter().filter(|o| o.last_gw == last).count();
        Some((left, last - calls.gw + 1, last))
    });
    let mut lines = Vec::new();
    lines.push(match (calls.planner.as_deref(), forced, window) {
        (Some(c), _, _) => format!(
            "<b>Chip this week:</b> play {} ({:+.1} xP vs keeping it)",
            e(chip_label(c)),
            calls.planner_gain
        ),
        (None, Some(c), Some((left, gws, last))) => format!(
            "<b>Chip this week:</b> play {} now ({left} chips left, {gws} GWs to GW{last})",
            e(chip_label(c))
        ),
        (None, Some(c), None) => {
            format!("<b>Chip this week:</b> play {} now, or lose it", e(chip_label(c)))
        }
        (None, None, _) => "<b>Chip this week:</b> hold".to_string(),
    });
    // A Free Hit or Wildcard squad differs from the alert's no-chip plan, forced or not.
    if matches!(headline, Some("freehit" | "wildcard")) {
        lines[0].push_str(" \u{b7} see /chips for the chip squad");
    }
    if calls.planner.is_some()
        && let Some((left, gws, last)) = window
    {
        lines.push(format!("  {left} chips left and {gws} GWs to GW{last}: play one each GW"));
    }

    let gain = |key: &str| match calls.gains.get(key).copied().flatten() {
        Some(v) => format!("{v:.1}"),
        None => "n/a".to_string(),
    };
    let rules = calls.rules.as_deref();
    let reason = match rules {
        Some(c) if calls.forced.as_deref() == Some(c) => {
            "last chance before the window closes".to_string()
        }
        Some("3xc") => format!("captain xP {} \u{2265} 8.3", gain("captain_xp")),
        Some("bboost") => format!("bench xP {} \u{2265} 10.9", gain("bench_xp")),
        Some("freehit") => format!(
            "{} squad players without a match \u{2265} 4",
            gain("blanks").trim_end_matches(".0")
        ),
        Some("wildcard") => format!("wildcard gain {} \u{2265} 12.9", gain("gain_wc")),
        Some(_) => String::new(),
        None => format!("captain xP {}, bench xP {}", gain("captain_xp"), gain("bench_xp")),
    };
    if rules == headline {
        lines.push(match rules {
            Some(_) => format!("  Simple rules agree: {}", e(&reason)),
            None => "  Simple rules agree: hold".to_string(),
        });
    } else {
        let play = match rules {
            Some(c) => format!("play {}", e(chip_label(c))),
            None => "hold".to_string(),
        };
        lines.push(format!("  Simple rules: {play} ({})", e(&reason)));
    }

    if calls.open.is_empty() {
        lines.push("Unused chips: none".to_string());
    } else {
        let parts: Vec<String> = calls
            .open
            .iter()
            .map(|o| {
                let mut note = if o.expiring {
                    format!("expires after GW{}", o.last_gw)
                } else {
                    format!("until GW{}", o.last_gw)
                };
                if headline == Some(o.name.as_str()) {
                    note.push_str(", playing now");
                } else if let Some(best) = o.best_gw
                    // "best GW: this GW" would contradict P's "hold"
                    && !(calls.planner.is_none() && best == calls.gw)
                {
                    note.push_str(&format!(", best GW: GW{best}"));
                }
                format!("{} ({note})", e(chip_label(&o.name)))
            })
            .collect();
        lines.push(format!("Unused chips: {}", parts.join(", ")));
    }
    lines.push(
        "Evidence: on 2023/24 and 2024/25 this planner scored +4.2 pts/GW over playing no chips; \
         not proven better than the simple rules."
            .to_string(),
    );
    lines.join("\n")
}

/// The chip section when the calls could not be computed.
pub fn chip_failure_section() -> String {
    "<b>Chip this week:</b> chip check failed this time: try /chips".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chip_calls::OpenChip;
    use crate::summary::{Alternative, Move, Pick, PlanSummary};
    use chrono::TimeZone;

    fn summary() -> PlanSummary {
        PlanSummary {
            gw: 12,
            deadline: chrono::Utc.with_ymd_and_hms(2026, 11, 21, 11, 0, 0).unwrap(),
            horizon: 6,
            moves: vec![Move {
                out_name: "Wissa".into(),
                in_name: "Gonzalo".into(),
                sell_price: 69,
                buy_price: 71,
                out_points: 26.1,
                in_points: 31.4,
                in_fixtures: Some(vec![
                    (12, vec![("BRE".into(), true)]),
                    (13, vec![("LEE".into(), false), ("WOL".into(), true)]),
                    (14, vec![]),
                ]),
            }],
            hit_points: 0,
            captain: Pick { name: "Haaland".into(), points: 7.8 },
            vice: Pick { name: "Salah".into(), points: 6.9 },
            gain_vs_roll: Some(2.44),
            bank_after: 3,
            free_transfers_next: 1,
            team: vec![
                ("GK", vec!["Raya".into()]),
                ("DEF", vec!["Gabriel".into(), "Timber".into()]),
                ("FWD", vec!["Haaland (C)".into(), "Gonzalo".into()]),
            ],
            bench: vec!["Dúbravka".into(), "Ings".into()],
            alternatives: vec![
                Alternative {
                    moves: vec![("Wissa".into(), "Strand Larsen".into())],
                    hit_points: 0,
                    gain: Some(2.3),
                    tied: true,
                },
                Alternative {
                    moves: vec![("Mbeumo".into(), "Palmer".into()), ("Ings".into(), "Wood".into())],
                    hit_points: 4,
                    gain: Some(1.8),
                    tied: false,
                },
            ],
            roll: Some((2, false)),
            later: vec![(14, "Saka".into(), "Kudus".into())],
            stopped_early: false,
            prices_fallback: None,
            told_transfers: false,
        }
    }

    #[test]
    fn plan_text_in_full() {
        assert_eq!(
            render_plan(&summary()),
            "<b>GW12 plan</b> · deadline Sat 21 Nov 11:00 UK\n\
             \n\
             <b>This week's moves</b>\n\
             Sell Wissa (£6.9m) → buy Gonzalo (£7.1m)\n\
             \x20 26.1 → 31.4 pts over the next 6 GWs\n\
             \x20 Gonzalo's next games: BRE (H), LEE (A) + WOL (H), no game\n\
             Captain Haaland (7.8 pts expected) · vice Salah (6.9)\n\
             Gain: about +2.4 pts vs saving your free transfer\n\
             Afterwards: £0.3m in the bank, 1 free transfer next GW\n\
             \n\
             <b>Your team</b>\n\
             GK Raya\n\
             DEF Gabriel, Timber\n\
             FWD Haaland (C), Gonzalo\n\
             Bench Dúbravka, Ings\n\
             \n\
             <b>Close alternatives</b>\n\
             · Sell Wissa → buy Strand Larsen: +2.3 pts (just as good)\n\
             · Sell Mbeumo → buy Palmer, sell Ings → buy Wood (−4 pts hit): +1.8 pts\n\
             · Save your transfer: 2 free transfers next GW\n\
             \n\
             Looking ahead (may change): GW14 sell Saka → buy Kudus\n\
             \n\
             Made a transfer already? FPL hides it until the deadline, so tell me with /transfer."
        );
    }

    #[test]
    fn plan_text_variants() {
        let mut s = summary();
        s.hit_points = 4;
        s.horizon = 1;
        s.stopped_early = true;
        s.told_transfers = true;
        s.prices_fallback = Some(chrono::Utc.with_ymd_and_hms(2026, 11, 21, 9, 0, 0).unwrap());
        s.roll = Some((2, true));
        s.later.clear();
        let text = render_plan(&s);
        assert!(text.contains("<b>This week's moves</b> (costs a 4-pt hit)\n"), "{text}");
        assert!(text.contains("\n  26.1 → 31.4 pts this GW\n"), "{text}");
        assert!(text.contains("· Save your transfer: just as good (2 free transfers next GW)"));
        assert!(!text.contains("Looking ahead"));
        assert!(text.ends_with(
            "This plan was found under a time limit; there may be a slightly better one.\n\
             Couldn't reach FPL: prices from Sat 21 Nov 09:00 UK.\n\
             Includes the transfers you told me about (/transfer list).\n\
             Made a transfer already? FPL hides it until the deadline, so tell me with /transfer."
        ));

        let mut s = summary();
        s.moves.clear();
        s.roll = None;
        s.gain_vs_roll = None;
        s.free_transfers_next = 2;
        let text = render_plan(&s);
        assert!(
            text.contains(
                "<b>This week's moves:</b> none, save your transfer (2 free transfers next GW)\n\
                 Captain Haaland"
            ),
            "{text}"
        );
        assert!(!text.contains("Gain:"));
        assert!(!text.contains("Afterwards"));
    }

    #[test]
    fn plan_text_escapes_names_and_fits_one_message() {
        let mut s = summary();
        s.moves[0].in_name = "A&B <x>".into();
        let text = render_plan(&s);
        assert!(text.contains("buy A&amp;B &lt;x&gt;"), "{text}");
        // Worst case: 3 hits' worth of moves with long names and two alternatives.
        let long = "Abcdefghijklmnopqrstuvwxyz-Abcdefghij";
        s.moves = vec![s.moves[0].clone(); 5];
        for m in &mut s.moves {
            m.out_name = long.into();
            m.in_name = long.into();
        }
        s.hit_points = 12;
        assert!(render_plan(&s).chars().count() < MAX_MESSAGE);
    }

    fn chip_calls(planner: Option<&str>, rules: Option<&str>, forced: Option<&str>) -> ChipCalls {
        ChipCalls {
            gw: 13,
            planner: planner.map(Into::into),
            planner_gain: 7.44,
            rules: rules.map(Into::into),
            gains: [("captain_xp", 7.0), ("bench_xp", 11.6), ("blanks", 0.0), ("gain_wc", 3.0)]
                .iter()
                .map(|(k, v)| (k.to_string(), Some(*v)))
                .collect(),
            forced: forced.map(Into::into),
            open: vec![
                OpenChip {
                    name: "freehit".into(),
                    last_gw: 19,
                    best_gw: Some(16),
                    expiring: false,
                },
                OpenChip { name: "bboost".into(), last_gw: 19, best_gw: Some(13), expiring: false },
            ],
            v6: None,
        }
    }

    #[test]
    fn chip_section_play_and_agree() {
        assert_eq!(
            chip_section(&chip_calls(Some("bboost"), Some("bboost"), None)),
            "<b>Chip this week:</b> play Bench Boost (+7.4 xP vs keeping it)\n\
             \x20 Simple rules agree: bench xP 11.6 \u{2265} 10.9\n\
             Unused chips: Free Hit (until GW19, best GW: GW16), Bench Boost (until GW19, playing now)\n\
             Evidence: on 2023/24 and 2024/25 this planner scored +4.2 pts/GW over playing no chips; \
             not proven better than the simple rules."
        );
    }

    #[test]
    fn chip_section_hold_and_differ() {
        let text = chip_section(&chip_calls(None, Some("bboost"), None));
        assert!(text.starts_with("<b>Chip this week:</b> hold\n  Simple rules: play Bench Boost (bench xP 11.6 \u{2265} 10.9)\n"));
        // Bench Boost's best GW is this one (GW13): not shown next to "hold".
        assert!(
            text.contains(
                "Unused chips: Free Hit (until GW19, best GW: GW16), Bench Boost (until GW19)\n"
            ),
            "{text}"
        );
    }

    #[test]
    fn chip_section_forced_and_expiring() {
        let mut calls = chip_calls(None, Some("freehit"), Some("freehit"));
        calls.gw = 18;
        calls.open[0].expiring = true;
        let text = chip_section(&calls);
        assert!(text.starts_with("<b>Chip this week:</b> play Free Hit now (2 chips left, 2 GWs to GW19) · see /chips for the chip squad\n  Simple rules agree: last chance before the window closes\n"), "{text}");
        assert!(text.contains("Free Hit (expires after GW19, playing now)"));
    }

    // P plays Bench Boost at GW18 while H's last-chance rule forces Free Hit (two chips left,
    // window ends GW19): P's call stays the headline.
    #[test]
    fn chip_section_forced_chip_does_not_override_the_planner() {
        let mut calls = chip_calls(Some("bboost"), Some("freehit"), Some("freehit"));
        calls.gw = 18;
        for o in &mut calls.open {
            o.expiring = true;
        }
        calls.open[0].best_gw = Some(19);
        assert_eq!(
            chip_section(&calls),
            "<b>Chip this week:</b> play Bench Boost (+7.4 xP vs keeping it)\n\
             \x20 2 chips left and 2 GWs to GW19: play one each GW\n\
             \x20 Simple rules: play Free Hit (last chance before the window closes)\n\
             Unused chips: Free Hit (expires after GW19, best GW: GW19), Bench Boost (expires after GW19, playing now)\n\
             Evidence: on 2023/24 and 2024/25 this planner scored +4.2 pts/GW over playing no chips; \
             not proven better than the simple rules."
        );
    }

    #[test]
    fn chip_section_free_hit_points_to_chips() {
        let text = chip_section(&chip_calls(Some("freehit"), None, None));
        assert!(text.starts_with("<b>Chip this week:</b> play Free Hit (+7.4 xP vs keeping it) \u{b7} see /chips for the chip squad\n"));
        assert!(text.contains("\n  Simple rules: hold (captain xP 7.0, bench xP 11.6)\n"));
    }

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

    #[test]
    fn update_text_says_what_changed() {
        let change = crate::summary::PlanChange {
            gw: 12,
            deadline: chrono::Utc.with_ymd_and_hms(2026, 11, 21, 11, 0, 0).unwrap(),
            horizon: 6,
            was_moves: vec![("Wissa".into(), "Strand Larsen".into())],
            now_moves: vec![("Wissa".into(), "Gonzalo".into())],
            moves_changed: true,
            captain: ("Haaland".into(), None),
            vice: ("Salah".into(), Some("Saka".into())),
            now_points: vec![("Gonzalo".into(), 31.4)],
            was_points: vec![("Strand Larsen".into(), 30.6)],
        };
        assert_eq!(
            update_text(&change),
            "<b>GW12 plan changed</b> · deadline Sat 21 Nov 11:00 UK\n\
             Now: sell Wissa, buy Gonzalo (was: sell Wissa, buy Strand Larsen)\n\
             Captain: Haaland (unchanged)\n\
             Vice: Salah (was Saka)\n\
             Why (pts over the next 6 GWs): now Gonzalo 31.4; before Strand Larsen 30.6\n\
             Full plan and team: /plan"
        );
        let mut c = change;
        c.moves_changed = false;
        c.now_points.clear();
        c.was_points.clear();
        let text = update_text(&c);
        assert!(!text.contains("Now:") && !text.contains("Why"), "{text}");
    }
}
