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
    for (m, t) in s.moves.iter().zip(&first.transfers) {
        assert_eq!(m.out_name, names[&t.out_key]);
        assert_eq!(m.in_name, names[&t.in_key]);
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
