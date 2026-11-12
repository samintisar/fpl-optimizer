//! `plan_summary` on the synthetic golden league (`tests/fixtures/rust_golden/bot/`): team 7
//! at GW6 with the script's planner settings (horizon 1).

use std::path::{Path, PathBuf};

use fplopt_bot::context::ContextStore;
use fplopt_bot::planner::{live_params, plan_user};
use fplopt_bot::replay::FileFpl;
use fplopt_bot::summary::{horizon_points, plan_summary};
use fplopt_optimize::io::params_with;
use serde_json::Value;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

#[test]
fn plan_summary_of_the_golden_team() {
    let root = repo();
    std::env::set_current_dir(&root).unwrap(); // rules are read relative to the repo root
    let golden = root.join("tests/fixtures/rust_golden/bot");
    let script: Value =
        serde_json::from_str(&std::fs::read_to_string(golden.join("script.json")).unwrap())
            .unwrap();
    let context = ContextStore::new(golden.join("state/context")).latest().unwrap().unwrap();
    let fpl = FileFpl { dir: golden.join("fpl"), down: Default::default() };
    let params = params_with(live_params(), &script["params"]).unwrap();
    let plan = plan_user(&fpl, 7, &context, None, &params, &[]).unwrap();
    let s = plan_summary(&plan).unwrap();

    assert_eq!((s.gw, s.horizon), (6, 1));
    let best = fplopt_bot::planner::ranked(&plan.plans)[0];
    let first = best.first();
    assert_eq!(s.moves.len(), first.transfers.len());
    assert_eq!(s.hit_points, plan.rules.hit_cost * first.hits);
    let names = context.names();
    // The moves are re-paired within position, so compare the sets of outs and ins.
    let sorted = |mut v: Vec<String>| {
        v.sort();
        v
    };
    assert_eq!(
        sorted(s.moves.iter().map(|m| m.out_name.clone()).collect()),
        sorted(first.transfers.iter().map(|t| names[&t.out_key].clone()).collect())
    );
    assert_eq!(
        sorted(s.moves.iter().map(|m| m.in_name.clone()).collect()),
        sorted(first.transfers.iter().map(|t| names[&t.in_key].clone()).collect())
    );
    for m in &s.moves {
        let t = first.transfers.iter().find(|t| names[&t.in_key] == m.in_name).unwrap();
        assert_eq!(Some(m.in_points), horizon_points(&context.xp, t.in_key, &[6]));
        assert!(m.in_fixtures.is_none(), "the golden context has no fixtures frame");
        let pool_price = context.pool.iter().find(|p| p.player_key == t.in_key).unwrap().price;
        assert_eq!(m.buy_price, pool_price);
    }
    assert_eq!(s.captain.name, names[&first.captain]);
    assert_eq!(Some(s.captain.points), horizon_points(&context.xp, first.captain, &[6]));
    assert_eq!(s.gain_vs_roll, best.gain_vs_roll);
    assert_eq!(s.bank_after, first.bank);
    // 5 free transfers, 5 moves: max(5 - 5, 0) + 1.
    assert_eq!(
        s.free_transfers_next,
        (plan.state.free_transfers - first.transfers.len() as i64).max(0) + 1
    );
    assert_eq!(s.team.iter().map(|(_, names)| names.len()).sum::<usize>(), 11);
    assert_eq!(s.bench.len(), 4);
    let starters: Vec<&String> = s.team.iter().flat_map(|(_, names)| names).collect();
    assert_eq!(starters.iter().filter(|n| n.ends_with(" (C)")).count(), 1);
    assert_eq!(starters.iter().filter(|n| n.ends_with(" (V)")).count(), 1);
    assert_eq!(s.alternatives.len(), plan.plans.plans.len() - 1);
    assert!(s.later.is_empty(), "a 1-GW horizon has no later moves");
    assert!(!s.told_transfers);
    assert_eq!(s.prices_fallback, Some(context.bootstrap_at.unwrap_or(context.created_at)));
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

/// A context with a `fixtures.parquet` (a single, a double and a blank for the club of plan 1's
/// first incoming player) read back through `read_context`: the move's `in_fixtures` and the
/// plan text show the next games.
#[test]
fn fixtures_reach_the_plan_text() {
    use fplopt_bot::context::read_context;
    use fplopt_bot::messages::render_plan;
    use fplopt_build::table::{Column, Table, write_parquet};

    let root = repo();
    std::env::set_current_dir(&root).unwrap();
    let golden = root.join("tests/fixtures/rust_golden/bot");
    let script: Value =
        serde_json::from_str(&std::fs::read_to_string(golden.join("script.json")).unwrap())
            .unwrap();
    let source = ContextStore::new(golden.join("state/context")).latest().unwrap().unwrap();
    let tmp = std::env::temp_dir().join(format!("fplopt-fixtures-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    copy_dir(&golden.join("state/context").join(&source.context_id), &tmp);

    let fpl = FileFpl { dir: golden.join("fpl"), down: Default::default() };
    let params = params_with(live_params(), &script["params"]).unwrap();
    // Plan once without fixtures to find the club of the first incoming player.
    let plain = plan_user(&fpl, 7, &source, None, &params, &[]).unwrap();
    let best = fplopt_bot::planner::ranked(&plain.plans)[0];
    let in_key = best.first().transfers[0].in_key;
    let team_key = source.pool.iter().find(|p| p.player_key == in_key).unwrap().team_key;
    let in_name = source.names()[&in_key].clone();

    let next3: Vec<i64> =
        source.gameweeks.iter().map(|g| g.0).filter(|&g| g >= source.gw).take(3).collect();
    assert_eq!(next3.len(), 3, "the golden context has three GWs from the planned one");
    let games: Vec<Vec<(&str, bool)>> = vec![
        vec![("AAA", true)],
        vec![("BBB", false), ("CCC", true)],
        vec![], // a blank
    ];
    let (mut gw, mut team, mut opponent, mut home) = (vec![], vec![], vec![], vec![]);
    for (g, list) in next3.iter().zip(&games) {
        for (o, h) in list {
            gw.push(*g);
            team.push(team_key);
            opponent.push(o.to_string());
            home.push(*h);
        }
    }
    let table = Table::new()
        .with("gw", Column::I64(gw))
        .with("team_key", Column::I64(team))
        .with("opponent", Column::Str(opponent))
        .with("home", Column::Bool(home));
    write_parquet(&table, &tmp.join("fixtures.parquet")).unwrap();

    let context = read_context(&tmp).unwrap();
    let plan = plan_user(&fpl, 7, &context, None, &params, &[]).unwrap();
    let s = plan_summary(&plan).unwrap();
    let m = s.moves.iter().find(|m| m.in_name == in_name).expect("the move is still planned");
    let expected: Vec<(i64, Vec<(String, bool)>)> = next3
        .iter()
        .zip(&games)
        .map(|(g, list)| (*g, list.iter().map(|(o, h)| (o.to_string(), *h)).collect()))
        .collect();
    assert_eq!(m.in_fixtures, Some(expected));

    let text = render_plan(&s);
    assert!(
        text.contains(&format!("{in_name}'s next games: AAA (H), BBB (A) + CCC (H), no game")),
        "{text}"
    );
    let _ = std::fs::remove_dir_all(&tmp);
}
