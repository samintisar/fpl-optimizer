//! Telegram message texts (spec docs/superpowers/specs/2026-11-11-bot-messaging-ux-design.md):
//! Telegram HTML (`<b>` and escaped text only), short plain-FPL lines for a phone, worded
//! from `summary`'s facts. `plan_text` is the plan (`/plan`, the 24-hour alert),
//! `update_text` the re-run's "plan changed", `chip_section` the alerts' chip lines,
//! `chips_text` `/chips`; `split_message` cuts a text to Telegram's 4,096 character limit at
//! line breaks. The texts are the Rust bot's own since 2026-11-11 (no longer Python's).

use std::collections::{BTreeSet, HashMap};

use fplopt_optimize::GwPlan;
use fplopt_optimize::entry::SquadSpec;

use crate::chip_calls::ChipCalls;
use crate::context::PlanContext;
use crate::planner::{CHIP_LABELS, ChipOptions, UserPlan, chip_label};
use crate::summary::{PlanChange, PlanSummary, plan_summary};
use crate::util::{escape as e, uk_time};

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
    let hit =
        if g.hits != 0 { format!(" (−{} pts hit)", hit_cost * g.hits) } else { String::new() };
    e(if moves.is_empty() { "no transfers" } else { &moves }) + &hit
}

pub fn money(tenths: i64) -> String {
    format!("£{:.1}m", tenths as f64 / 10.0)
}

/// A points figure with its sign at one decimal: "+1.2", "−1.0" (U+2212), "+0.0" for zero.
pub fn signed(x: f64) -> String {
    let digits = format!("{:.1}", x.abs());
    if x < 0.0 && digits != "0.0" { format!("\u{2212}{digits}") } else { format!("+{digits}") }
}

/// "1 chip" / "2 chips".
fn count(n: usize, noun: &str) -> String {
    if n == 1 { format!("1 {noun}") } else { format!("{n} {noun}s") }
}

/// "5 free transfers next GW", with ", the most you can bank" at the cap.
fn next_gw_transfers(n: i64, cap: i64) -> String {
    let text = format!("{} next GW", free_transfers(n));
    if n == cap { format!("{text}, the most you can bank") } else { text }
}

pub fn free_transfers(n: i64) -> String {
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
            "<b>This week's moves:</b> none, save your transfer ({})",
            next_gw_transfers(s.free_transfers_next, s.ft_cap)
        ));
    } else {
        let hit = if s.hit_points > 0 {
            format!(" (\u{2212}{} pts hit)", s.hit_points)
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
            lines.push(format!("Gain: about {} pts vs saving your free transfer", signed(g)));
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
            let gain = a.gain.map_or(String::new(), |g| format!(": {} pts", signed(g)));
            let tie = if a.tied { " (just as good)" } else { "" };
            lines.push(format!("· {}{hit}{gain}{tie}", capitalized(&sell_buy(&a.moves))));
        }
        if let Some((ft, tie)) = s.roll {
            let next = next_gw_transfers(ft, s.ft_cap);
            lines.push(if tie {
                format!("· Save your transfer: just as good ({next})")
            } else {
                format!("· Save your transfer: {next}")
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
        let hit = if c.hit_points > 0 {
            format!(" (\u{2212}{} pts hit)", c.hit_points)
        } else {
            String::new()
        };
        lines.push(format!(
            "Now: {}{hit} (was: {})",
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
    // "Why" needs a newly bought player to point to; "before X" alone explains nothing.
    if !c.now_points.is_empty() {
        let list = |p: &[(String, f64)]| {
            let parts: Vec<String> = p.iter().map(|(n, x)| format!("{} {x:.1}", e(n))).collect();
            parts.join(", ")
        };
        let what = if c.horizon == 1 {
            "pts this GW".to_string()
        } else {
            format!("pts over the next {} GWs", c.horizon)
        };
        let mut parts = vec![format!("now {}", list(&c.now_points))];
        if !c.was_points.is_empty() {
            parts.push(format!("before {}", list(&c.was_points)));
        }
        lines.push(format!("Why ({what}): {}", parts.join("; ")));
    }
    lines.push("Full plan and team: /plan".into());
    lines.join("\n")
}

/// The imported squad by position, bank and free transfers, chips played (/register).
pub fn squad_text(context: &PlanContext, spec: &SquadSpec) -> String {
    let names: HashMap<i64, &str> =
        context.players.iter().map(|p| (p.element_id, p.web_name.as_str())).collect();
    let keys = context.element_keys();
    let positions: HashMap<i64, i64> =
        context.pool.iter().map(|p| (p.player_key, p.element_type)).collect();
    let position = |element: i64| keys.get(&element).and_then(|k| positions.get(k)).copied();
    let label =
        |element: i64| names.get(&element).map_or_else(|| element.to_string(), |n| n.to_string());
    let mut lines = Vec::new();
    for (pos, tag) in crate::summary::POSITIONS {
        let players: Vec<String> = spec
            .players
            .iter()
            .filter(|p| position(p.element_id) == Some(pos))
            .map(|p| label(p.element_id))
            .collect();
        if !players.is_empty() {
            lines.push(format!("{tag} {}", e(&players.join(", "))));
        }
    }
    let other: Vec<String> = spec
        .players
        .iter()
        .filter(|p| position(p.element_id).is_none())
        .map(|p| label(p.element_id))
        .collect();
    if !other.is_empty() {
        lines.push(format!("Others {}", e(&other.join(", "))));
    }
    let used: Vec<String> =
        spec.chips_used.iter().map(|(n, gw)| format!("{} (GW{gw})", chip_label(n))).collect();
    let used = used.join(", ");
    lines.push(format!(
        "{} in the bank · {}",
        money(spec.bank),
        free_transfers(spec.free_transfers)
    ));
    lines.push(format!("Chips played: {}", e(if used.is_empty() { "none" } else { &used })));
    lines.join("\n")
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

/// `/chips` (information only, spec *`/chips`*).
pub fn chips_text(options: &ChipOptions) -> String {
    let (context, rules, params) = (&options.context, &options.rules, &options.params);
    let gw_map = context.gw_index_map();
    let gw_of: HashMap<i64, i64> = gw_map.iter().map(|(g, i)| (*i, *g)).collect();
    let used: BTreeSet<i64> = options.state.chips_used.iter().map(|(cid, _)| *cid).collect();
    let mut windows: Vec<_> = rules.chips.iter().collect();
    windows.sort_by_key(|w| (w.stop, w.start, w.chip_id));
    let objectives = &options.plans.scenario_objectives;
    let base = objectives.iter().rev().find(|(l, _)| l == "none").map_or(0.0, |(_, v)| *v);
    let best = options.plans.best();
    let names = context.names();
    let mut lines = vec![
        format!("<b>GW{} chips</b> · for information: your /plan doesn't use them", context.gw),
        if best.scenario == "none" {
            format!("Best {}: keep your chips", over(best.gws.len()))
        } else {
            format!(
                "Best {}: {} ({} pts vs keeping your chips)",
                over(best.gws.len()),
                e(&scenario_label(&best.scenario)),
                signed(best.total_objective() - base)
            )
        },
    ];
    let first = best.first();
    match first.chip.as_deref() {
        Some(chip @ ("freehit" | "wildcard")) if best.scenario != "none" => {
            let positions: HashMap<i64, i64> =
                context.pool.iter().map(|p| (p.player_key, p.element_type)).collect();
            lines.push(String::new());
            lines.push(format!("<b>Your {} team</b>", chip_label(chip)));
            for (pos, label) in crate::summary::POSITIONS {
                let players: Vec<&str> = first
                    .starters
                    .iter()
                    .filter(|k| positions.get(k).copied() == Some(pos))
                    .map(|&k| name(&names, k))
                    .collect();
                if !players.is_empty() {
                    lines.push(format!("{label} {}", e(&players.join(", "))));
                }
            }
            let bench: Vec<&str> = first.bench.iter().map(|&k| name(&names, k)).collect();
            lines.push(format!("Bench {}", e(&bench.join(", "))));
            lines.push(format!("Captain {}", e(name(&names, first.captain))));
        }
        Some(_) if best.scenario != "none" => {
            lines.push(format!(
                "This week with it: {}; captain {}",
                moves(first, &names, rules.hit_cost),
                e(name(&names, first.captain))
            ));
        }
        _ => {}
    }
    if !options.singles.is_empty() {
        let mut singles: Vec<(&str, f64)> = CHIP_LABELS
            .iter()
            .filter_map(|(chip, _)| options.single(chip))
            .map(|(label, value)| (label.as_str(), value - base))
            .collect();
        singles.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        lines.push(String::new());
        lines.push("<b>Each chip in its best week</b>".into());
        for (label, gain) in singles {
            lines.push(format!("· {}: {}", e(&scenario_label(label)), signed(gain)));
        }
    }
    // Chips left, grouped by the GW their window ends.
    let mut groups: Vec<(i64, Vec<&str>)> = Vec::new();
    for w in windows.iter().filter(|w| !used.contains(&w.chip_id) && w.stop >= context.gw_index) {
        let last = gw_of.get(&w.stop).copied().unwrap_or(w.stop);
        match groups.last_mut() {
            Some((g, chips)) if *g == last => chips.push(w.name.as_str()),
            _ => groups.push((last, vec![w.name.as_str()])),
        }
    }
    // Within a group, chips in the usual order (Wildcard, Free Hit, Bench Boost, Triple
    // Captain).
    let rank = |name: &str| CHIP_LABELS.iter().position(|(n, _)| *n == name).unwrap_or(usize::MAX);
    for (_, chips) in &mut groups {
        chips.sort_by_key(|n| rank(n));
    }
    let left: Vec<String> = groups
        .iter()
        .map(|(last, chips)| {
            let labels: Vec<&str> = chips.iter().map(|n| chip_label(n)).collect();
            format!("{} (until GW{last})", labels.join(", "))
        })
        .collect();
    lines.push(String::new());
    let left = if left.is_empty() { "none".to_string() } else { left.join("; ") };
    lines.push(format!("Chips left: {}", e(&left)));
    lines.push("A chip only shows a gain if it beats saving it for later.".into());
    let solved: BTreeSet<&str> = objectives
        .iter()
        .map(|(l, _)| l.as_str())
        .chain(options.singles.iter().map(|(_, (l, _))| l.as_str()))
        .collect();
    let stopped = options
        .plans
        .search
        .not_optimal
        .iter()
        .filter(|(l, _)| !solved.contains(l.as_str()))
        .count();
    if stopped > 0 {
        lines.push("Some options ran out of time, so this may be incomplete.".into());
    }
    lines.push(if params.chip_value_left.is_some() {
        "Tested on 2023–25: beat playing no chips, not clearly better than simple rules.".into()
    } else {
        "The chip planner isn't tested yet: weigh it against your own judgement.".into()
    });
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
    let window = forced.and_then(|f| {
        let last = calls.open.iter().find(|o| o.name == f)?.last_gw;
        let left = calls.open.iter().filter(|o| o.last_gw == last).count();
        Some((left, last - calls.gw + 1, last))
    });
    let mut lines = Vec::new();
    lines.push(match (calls.planner.as_deref(), forced, window) {
        (Some(c), _, _) => format!(
            "<b>Chip:</b> play {} this week ({} pts vs keeping it)",
            e(chip_label(c)),
            signed(calls.planner_gain)
        ),
        (None, Some(c), Some((left, gws, last))) => format!(
            "<b>Chip:</b> play {} now ({} left and {} before they expire after GW{last})",
            e(chip_label(c)),
            count(left, "chip"),
            count(gws as usize, "GW")
        ),
        (None, Some(c), None) => format!("<b>Chip:</b> play {} now, or lose it", e(chip_label(c))),
        (None, None, _) => "<b>Chip:</b> keep your chips this week".to_string(),
    });
    if let Some(c @ ("freehit" | "wildcard")) = headline {
        lines[0].push_str(&format!(" \u{b7} your {} team: /chips", chip_label(c)));
    }
    if calls.planner.is_some()
        && let Some((left, gws, last)) = window
    {
        lines.push(format!(
            "  {} left and {} before they expire after GW{last}: play one each GW",
            count(left, "chip"),
            count(gws as usize, "GW")
        ));
    }
    let gain = |key: &str| match calls.gains.get(key).copied().flatten() {
        Some(v) => format!("{v:.1}"),
        None => "n/a".to_string(),
    };
    let rules = calls.rules.as_deref();
    // (what the rule saw, its threshold) — joined with " (… )" when agreeing, "; " inside
    // the brackets when not.
    let (seen, threshold): (String, Option<&str>) = match rules {
        Some(c) if calls.forced.as_deref() == Some(c) => {
            ("last chance before they expire".to_string(), None)
        }
        Some("3xc") => {
            (format!("your captain is expected to score {}", gain("captain_xp")), Some("8.3+"))
        }
        Some("bboost") => {
            (format!("your bench is expected to score {}", gain("bench_xp")), Some("10.9+"))
        }
        Some("freehit") => (
            format!("{} of your players have no game", gain("blanks").trim_end_matches(".0")),
            Some("4+"),
        ),
        Some("wildcard") => (format!("a Wildcard gains {} pts", gain("gain_wc")), Some("12.9+")),
        Some(_) => (String::new(), None),
        None => (
            format!("captain expected {}, bench expected {}", gain("captain_xp"), gain("bench_xp")),
            None,
        ),
    };
    if rules == headline {
        lines.push(match (rules, threshold) {
            (None, _) => "  The simple rule agrees: keep them".to_string(),
            (Some(_), Some(t)) => {
                format!("  The simple rule agrees: {} (it plays at {t})", e(&seen))
            }
            (Some(_), None) => format!("  The simple rule agrees: {}", e(&seen)),
        });
    } else {
        let play = match rules {
            Some(c) => format!("play {}", e(chip_label(c))),
            None => "keep them".to_string(),
        };
        let why = match threshold {
            Some(t) => format!("{}; it plays at {t}", e(&seen)),
            None => e(&seen),
        };
        lines.push(if why.is_empty() {
            format!("  A simpler rule would {play}")
        } else {
            format!("  A simpler rule would {play} ({why})")
        });
    }
    if calls.open.is_empty() {
        lines.push("Chips left: none".to_string());
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
                    && !(calls.planner.is_none() && best == calls.gw)
                {
                    note.push_str(&format!(", best week GW{best}"));
                }
                format!("{} ({note})", e(chip_label(&o.name)))
            })
            .collect();
        lines.push(format!("Chips left: {}", parts.join(", ")));
    }
    lines.join("\n")
}

/// The chip section when the calls could not be computed.
pub fn chip_failure_section() -> String {
    "<b>Chip:</b> couldn't check chips this time, try /chips".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chip_calls::OpenChip;
    use crate::summary::{Alternative, Move, Pick, PlanSummary};
    use chrono::TimeZone;

    #[test]
    fn squad_text_lists_the_squad_by_position() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/rust_golden/bot/state/context");
        let id = std::fs::read_dir(&dir).unwrap().next().unwrap().unwrap().file_name();
        let context = crate::context::read_context(&dir.join(id)).unwrap();
        let types: HashMap<i64, i64> =
            context.pool.iter().map(|p| (p.player_key, p.element_type)).collect();
        let pick = |t: i64, n: usize| -> Vec<i64> {
            context
                .players
                .iter()
                .filter(|p| types.get(&p.player_key) == Some(&t))
                .take(n)
                .map(|p| p.element_id)
                .collect()
        };
        let mut elements = pick(1, 2);
        elements.extend(pick(4, 3));
        let players = elements
            .iter()
            .map(|&element_id| fplopt_optimize::entry::SquadPlayer {
                element_id,
                purchase_price: 50,
                name: String::new(),
            })
            .collect();
        let mut spec = SquadSpec::new(context.season, context.gw, players, 5, 2);
        spec.chips_used = vec![("wildcard".into(), 4)];
        let text = squad_text(&context, &spec);
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines[0].starts_with("GK "), "{text}");
        assert!(lines[1].starts_with("FWD "), "{text}");
        assert_eq!(lines[2], "£0.5m in the bank · 2 free transfers");
        assert_eq!(lines[3], "Chips played: Wildcard (GW4)");
    }

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
            ft_cap: 5,
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
        assert!(text.contains("<b>This week's moves</b> (\u{2212}4 pts hit)\n"), "{text}");
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
            "<b>Chip:</b> play Bench Boost this week (+7.4 pts vs keeping it)\n\
             \x20 The simple rule agrees: your bench is expected to score 11.6 (it plays at 10.9+)\n\
             Chips left: Free Hit (until GW19, best week GW16), Bench Boost (until GW19, playing now)"
        );
    }

    #[test]
    fn chip_section_hold_and_differ() {
        let text = chip_section(&chip_calls(None, Some("bboost"), None));
        assert!(text.starts_with(
            "<b>Chip:</b> keep your chips this week\n  A simpler rule would play Bench Boost (your bench is expected to score 11.6; it plays at 10.9+)\n"
        ), "{text}");
        assert!(
            text.ends_with(
                "Chips left: Free Hit (until GW19, best week GW16), Bench Boost (until GW19)"
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
        assert!(text.starts_with(
            "<b>Chip:</b> play Free Hit now (2 chips left and 2 GWs before they expire after GW19) · your Free Hit team: /chips\n  The simple rule agrees: last chance before they expire\n"
        ), "{text}");
        assert!(text.contains("Free Hit (expires after GW19, playing now)"));
    }

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
            "<b>Chip:</b> play Bench Boost this week (+7.4 pts vs keeping it)\n\
             \x20 2 chips left and 2 GWs before they expire after GW19: play one each GW\n\
             \x20 A simpler rule would play Free Hit (last chance before they expire)\n\
             Chips left: Free Hit (expires after GW19, best week GW19), Bench Boost (expires after GW19, playing now)"
        );
    }

    #[test]
    fn chip_section_free_hit_points_to_chips() {
        let text = chip_section(&chip_calls(Some("freehit"), None, None));
        assert!(text.starts_with("<b>Chip:</b> play Free Hit this week (+7.4 pts vs keeping it) · your Free Hit team: /chips\n"), "{text}");
        assert!(
            text.contains(
                "\n  A simpler rule would keep them (captain expected 7.0, bench expected 11.6)\n"
            ),
            "{text}"
        );
    }

    #[test]
    fn signed_uses_a_real_minus() {
        assert_eq!(signed(1.24), "+1.2");
        assert_eq!(signed(-1.0), "\u{2212}1.0");
        assert_eq!(signed(0.0), "+0.0");
        assert_eq!(signed(-0.02), "+0.0");
    }

    #[test]
    fn plan_text_at_the_free_transfer_cap() {
        let mut s = summary();
        s.free_transfers_next = 5;
        s.roll = Some((5, false));
        let text = render_plan(&s);
        assert!(
            text.contains(
                "· Save your transfer: 5 free transfers next GW, the most you can bank\n"
            ),
            "{text}"
        );
        s.roll = Some((5, true));
        assert!(render_plan(&s).contains(
            "· Save your transfer: just as good (5 free transfers next GW, the most you can bank)"
        ));
        s.moves.clear();
        assert!(render_plan(&s).contains(
            "<b>This week's moves:</b> none, save your transfer (5 free transfers next GW, the most you can bank)\n"
        ));
        // Below the cap: no suffix.
        s.free_transfers_next = 2;
        assert!(!render_plan(&s).contains("most you can bank)\nCaptain"));
    }

    #[test]
    fn chip_section_says_one_chip_and_one_gw() {
        let mut calls = chip_calls(None, Some("freehit"), Some("freehit"));
        calls.gw = 19;
        calls.open.remove(1);
        calls.open[0].expiring = true;
        let text = chip_section(&calls);
        assert!(
            text.starts_with(
                "<b>Chip:</b> play Free Hit now (1 chip left and 1 GW before they expire after GW19)"
            ),
            "{text}"
        );
        let mut calls = chip_calls(Some("bboost"), Some("freehit"), Some("freehit"));
        calls.gw = 19;
        calls.open.remove(1);
        let text = chip_section(&calls);
        assert!(
            text.contains(
                "\n  1 chip left and 1 GW before they expire after GW19: play one each GW\n"
            ),
            "{text}"
        );
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
            hit_points: 0,
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
        // A hit shows between the new moves and what they replace.
        let mut hit = change.clone();
        hit.hit_points = 4;
        assert!(
            update_text(&hit).contains(
                "Now: sell Wissa, buy Gonzalo (\u{2212}4 pts hit) (was: sell Wissa, buy Strand Larsen)\n"
            ),
            "{}",
            update_text(&hit)
        );
        // Only a dropped buy, no new one: no "Why" line.
        let mut c = change;
        c.now_points.clear();
        assert!(!update_text(&c).contains("Why"));
        c.moves_changed = false;
        c.now_points.clear();
        c.was_points.clear();
        let text = update_text(&c);
        assert!(!text.contains("Now:") && !text.contains("Why"), "{text}");
    }
}
