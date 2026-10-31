//! Decision probes for the corrupt-the-future check (Python `fplopt.backtest.probes`; PLAN
//! §4): a builder like a feature, one view in, a deterministic frame out. It builds a start
//! state at the view's deadline, runs a policy on it (the season's backtest rules, the
//! policy's xP model) and encodes the start state and the decision, so a policy or start
//! state that peeks past the deadline changes the frame. Registered in `check leakage` as
//! `probe:<name>`.
//!
//! Frame: kind, slot, player_key, value, note: `held` (slot = holding index, value =
//! purchase price), `bank`, `free_transfers`, `out`/`in` per transfer, `starter`/`bench`
//! (slot = lineup position), `captain`, `vice`, `chip` (note = its name). A refused start
//! state is one `refused` row with the reason.

use std::collections::BTreeMap;

use anyhow::{Result, bail};
use fplopt_build::table::{Column, Table};
use fplopt_core::seasons::{HOLDOUT_SEASONS, season_label};
use fplopt_features::FeatureBuilder;
use fplopt_features::baseline::player_pool;
use fplopt_optimize::start::{
    MAX_BANK, StartStateError, pool_players, random_state, target_gameweek, template_state,
};
use fplopt_optimize::{OptimizerParams, XpFrame};
use fplopt_store::AsOfView;

use crate::evaluate::rules_for;
use crate::policies::{ChipPolicy2, DecisionContext, Policy};
use crate::state::{Decision, SquadState, apply_decision};

type Row = (String, i64, i64, i64, String);

fn frame(rows: Vec<Row>) -> Table {
    Table::new()
        .with("kind", Column::Str(rows.iter().map(|r| r.0.clone()).collect()))
        .with("slot", Column::I64(rows.iter().map(|r| r.1).collect()))
        .with("player_key", Column::I64(rows.iter().map(|r| r.2).collect()))
        .with("value", Column::I64(rows.iter().map(|r| r.3).collect()))
        .with("note", Column::Str(rows.iter().map(|r| r.4.clone()).collect()))
}

/// The start state and decision as a probe frame.
pub fn encode(state: &SquadState, decision: &Decision) -> Table {
    let r = |k: &str, slot: usize, key: i64, value: i64, note: &str| {
        (k.to_owned(), slot as i64, key, value, note.to_owned())
    };
    let mut rows: Vec<Row> = state
        .holdings
        .iter()
        .enumerate()
        .map(|(i, h)| r("held", i, h.player_key, h.purchase_price, ""))
        .collect();
    rows.push(r("bank", 0, -1, state.bank, ""));
    rows.push(r("free_transfers", 0, -1, state.free_transfers, ""));
    for (i, t) in decision.transfers.iter().enumerate() {
        rows.push(r("out", i, t.out_key, 0, ""));
        rows.push(r("in", i, t.in_key, 0, ""));
    }
    let l = &decision.lineup;
    rows.extend(l.starters.iter().enumerate().map(|(i, k)| r("starter", i, *k, 0, "")));
    rows.extend(l.bench.iter().enumerate().map(|(i, k)| r("bench", i, *k, 0, "")));
    rows.push(r("captain", 0, l.captain, 0, ""));
    rows.push(r("vice", 0, l.vice, 0, ""));
    rows.push(r("chip", 0, -1, 0, decision.chip.as_deref().unwrap_or("")));
    frame(rows)
}

type Start = fn(&AsOfView, &fplopt_build::scoring::Rules) -> Result<SquadState>;

/// `policy`'s decision from `start` at the view's deadline, encoded; the decision is
/// validated with `apply_decision`.
pub fn run_probe(view: &AsOfView, policy: &Policy, start: Start) -> Result<Table> {
    let (season, _, _) = target_gameweek(view)?;
    if HOLDOUT_SEASONS.contains(&(season as i32)) {
        bail!("decision probes do not run on the holdout season {}", season_label(season as i32));
    }
    let rules = rules_for(season)?;
    let state = match start(view, &rules) {
        Ok(s) => s,
        Err(e) if e.downcast_ref::<StartStateError>().is_some() => {
            return Ok(frame(vec![("refused".to_owned(), 0, -1, 0, e.to_string())]));
        }
        Err(e) => return Err(e),
    };
    let pool = pool_players(&player_pool(view)?)?;
    let xp = fplopt_models::predict(policy.xp_model(), view)?;
    let xp_frame = XpFrame::from_table(&xp)?;
    let ctx = DecisionContext {
        view,
        state: &state,
        rules: &rules,
        pool: &pool,
        xp: &xp,
        frame: &xp_frame,
    };
    let decision = policy.decide(&ctx)?;
    apply_decision(&state, &decision, &pool, &rules)?;
    Ok(encode(&state, &decision))
}

fn random0(view: &AsOfView, rules: &fplopt_build::scoring::Rules) -> Result<SquadState> {
    random_state(view, rules, 0, Some(MAX_BANK))
}

fn greedy(xp: &str) -> Policy {
    Policy::Greedy {
        xp_model: xp.to_owned(),
        threshold: 1.0,
        horizon: 6,
        decay: 0.85,
        max_transfers: 1,
    }
}

/// The optimizer probes' settings: no chips, a 3-GW horizon and a small candidate pool.
fn probe_params() -> OptimizerParams {
    OptimizerParams {
        horizon: 3,
        prune_n: Some(BTreeMap::from([(1, 8), (2, 20), (3, 20), (4, 10)])),
        ..OptimizerParams::default()
    }
}

fn optimizer(xp: &str) -> Policy {
    Policy::Optimizer {
        xp_model: xp.to_owned(),
        params: probe_params(),
        chips: false,
        extra: ChipPolicy2::default(),
    }
}

fn greedy_rolling_random0(view: &AsOfView) -> Result<Table> {
    run_probe(view, &greedy("rolling"), random0)
}

fn greedy_ep_next_template(view: &AsOfView) -> Result<Table> {
    run_probe(view, &greedy("ep_next"), template_state)
}

fn roll_rolling_template(view: &AsOfView) -> Result<Table> {
    run_probe(view, &Policy::Roll { xp_model: "rolling".to_owned() }, template_state)
}

fn optimizer_ep_next_template(view: &AsOfView) -> Result<Table> {
    run_probe(view, &optimizer("ep_next"), template_state)
}

fn optimizer_rolling_random0(view: &AsOfView) -> Result<Table> {
    run_probe(view, &optimizer("rolling"), random0)
}

/// The probes as leakage-check builders.
pub const PROBE_BUILDERS: &[(&str, FeatureBuilder)] = &[
    ("probe:greedy_rolling_random0", greedy_rolling_random0),
    ("probe:greedy_ep_next_template", greedy_ep_next_template),
    ("probe:roll_rolling_template", roll_rolling_template),
    ("probe:optimizer_ep_next_template", optimizer_ep_next_template),
    ("probe:optimizer_rolling_random0", optimizer_rolling_random0),
];
