//! Points per £1m without simulation noise (price sizing study, addendum of 2026-11-06,
//! `docs/superpowers/specs/2026-11-06-price-sizing-design.md`; Rust only).
//!
//! At each (season, gw_index, start) state the live planner solves twice at that deadline:
//! the state as is and with `extra_bank` tenths of £m more in the bank. From each solve,
//! plan #1's undecayed expected points over its horizon GWs net of hit points; v is the
//! difference per horizon GW, scaled to £1m. States are built as the backtester builds them
//! (`StartSpec::build`), except that a refused state is skipped, never moved to another GW.

use anyhow::{Result, bail};
use fplopt_build::table::{Column, Table};
use fplopt_core::pyjson::Ordered;
use fplopt_core::seasons::season_label;
use fplopt_optimize::Plan;
use fplopt_store::{DataStore, Micros};

use crate::evaluate::{Unit, parse_start_specs, refuse_holdout, refused, rules_for, run_units};
use crate::policies::{DecisionContext, Policy};
use crate::simulator::{Caches, season_schedule};
use crate::state::{SquadState, refresh};
use crate::stats::{np_mean, np_std1};

/// The live planner (`fplopt.bot.planner.LIVE_PARAMS` as a backtest spec).
pub const LIVE_SPEC: &str = "optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2";
/// The gw_index from which money is spent through normal transfers (GW1 is a free rebuild).
pub const POOLED_FROM: i64 = 8;

/// What one solve contributes: plan #1's undecayed xP net of hit points, its objective, and
/// its hits and transfers over the horizon.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Solved {
    pub xp_net: f64,
    pub objective: f64,
    pub hits: i64,
    pub n_transfers: i64,
    pub horizon_gws: i64,
}

/// Σ over the plan's GWs of its lineup xP (captain included), minus `hit_cost` per hit.
pub fn plan_outcome(plan: &Plan, hit_cost: i64) -> Solved {
    let hits: i64 = plan.gws.iter().map(|g| g.hits).sum();
    let xp: f64 = plan.gws.iter().map(|g| g.xp).sum();
    Solved {
        xp_net: xp - (hit_cost * hits) as f64,
        objective: plan.objective,
        hits,
        n_transfers: plan.gws.iter().map(|g| g.n_transfers() as i64).sum(),
        horizon_gws: plan.gws.len() as i64,
    }
}

/// One state's pair of solves.
#[derive(Clone, Debug, PartialEq)]
pub struct MoneyRow {
    pub season: i64,
    pub gw_index: i64,
    pub start_id: String,
    pub horizon_gws: i64,
    pub xp_base: f64,
    pub xp_bank: f64,
    pub obj_base: f64,
    pub obj_bank: f64,
    pub hits_base: i64,
    pub hits_bank: i64,
    pub n_transfers_base: i64,
    pub n_transfers_bank: i64,
    /// Expected points per horizon GW that £1m buys.
    pub v: f64,
}

pub fn money_row(
    season: i64,
    gw_index: i64,
    start_id: &str,
    base: &Solved,
    bank: &Solved,
    extra_bank: i64,
) -> Result<MoneyRow> {
    if base.horizon_gws != bank.horizon_gws || base.horizon_gws < 1 {
        bail!(
            "{} {start_id}: horizons differ or are empty ({} vs {} GWs)",
            season_label(season as i32),
            base.horizon_gws,
            bank.horizon_gws
        );
    }
    let v = (bank.xp_net - base.xp_net) / base.horizon_gws as f64 * 10.0 / extra_bank as f64;
    Ok(MoneyRow {
        season,
        gw_index,
        start_id: start_id.to_owned(),
        horizon_gws: base.horizon_gws,
        xp_base: base.xp_net,
        xp_bank: bank.xp_net,
        obj_base: base.objective,
        obj_bank: bank.objective,
        hits_base: base.hits,
        hits_bank: bank.hits,
        n_transfers_base: base.n_transfers,
        n_transfers_bank: bank.n_transfers,
        v,
    })
}

/// `"template,random:3"` at each gw_index -> the states, in (season, gw_index, start) order.
/// A refused state (e.g. the template at gw_index 1 before 2021/22) is skipped and logged.
pub fn start_units(
    store: &DataStore,
    seasons: &[i64],
    gws: &[i64],
    starts: &str,
) -> Result<Vec<Unit>> {
    let mut out = Vec::new();
    for &season in seasons {
        let rules = rules_for(season)?;
        let label = season_label(season as i32);
        for &gw in gws {
            for spec in specs_at(starts, gw)? {
                let id = spec.start_id(None);
                match spec.build(store, season, gw, &rules) {
                    Ok(state) => out.push((season, id, state)),
                    Err(e) if refused(&e) => log::warn!("{label} {id} refused, skipped: {e}"),
                    Err(e) => return Err(e),
                }
            }
        }
    }
    Ok(out)
}

/// The start specs `starts` (kinds without `@`) at `gw`.
pub fn specs_at(starts: &str, gw: i64) -> Result<Vec<crate::evaluate::StartSpec>> {
    if starts.contains('@') || starts.contains("+bank") {
        bail!("--starts takes kinds without @<gw> or +bank (e.g. template,random:3): {starts}");
    }
    let text: Vec<String> = starts.split(',').map(|s| format!("{}@{gw}", s.trim())).collect();
    parse_start_specs(&text.join(","))
}

/// Plan #1 of `policy` at `deadline` from `state`: the simulator's decision path
/// (`decide_step`: refreshed state, cached pool and xP frame) up to the planner's plans.
fn solve_at(
    store: &DataStore,
    rules: &fplopt_build::scoring::Rules,
    policy: &Policy,
    state: &SquadState,
    deadline: Micros,
    caches: &Caches,
) -> Result<Solved> {
    let view = store.as_of(deadline);
    let pool = caches.pool(&view)?;
    let state = refresh(state, &pool);
    let xp = caches.xp(policy.xp_model(), &view)?;
    let ctx = DecisionContext {
        view: &view,
        state: &state,
        rules,
        pool: &pool,
        xp: &xp.table,
        frame: &xp.frame,
    };
    let (_, _, plans) = policy.optimizer_plans(&ctx)?;
    let status = plans.solver_status();
    if status != "Optimal" {
        log::warn!(
            "{} gw_index {} bank {}: solver stopped at a limit ({status}, gap {:.2}%)",
            season_label(state.season as i32),
            state.gw_index,
            state.bank,
            100.0 * plans.best().mip_gap
        );
    }
    Ok(plan_outcome(plans.best(), rules.hit_cost))
}

/// Every state's pair of solves, in unit order (`jobs` threads; equal to `jobs = 1`).
#[allow(clippy::too_many_arguments)]
pub fn measure(
    store: &DataStore,
    seasons: &[i64],
    gws: &[i64],
    starts: &str,
    extra_bank: i64,
    policy: &Policy,
    caches: &Caches,
    jobs: usize,
) -> Result<Vec<MoneyRow>> {
    refuse_holdout(seasons)?;
    if !matches!(policy, Policy::Optimizer { .. }) {
        bail!("money value needs an optimizer policy, got {}", policy.name());
    }
    if extra_bank < 1 {
        bail!("extra_bank must be >= 1 (tenths of £m), got {extra_bank}");
    }
    let units = start_units(store, seasons, gws, starts)?;
    log::info!("money value: {} state(s), 2 solves each", units.len());
    run_units(&units, jobs, |(season, start_id, state), rules| {
        let deadline = season_schedule(store, *season, state.gw_index)?[0].deadline;
        let base = solve_at(store, rules, policy, state, deadline, caches)?;
        let mut more = state.clone();
        more.bank += extra_bank;
        let bank = solve_at(store, rules, policy, &more, deadline, caches)?;
        money_row(*season, state.gw_index, start_id, &base, &bank, extra_bank)
    })
}

pub fn rows_table(rows: &[MoneyRow]) -> Table {
    let i = |f: &dyn Fn(&MoneyRow) -> i64| Column::I64(rows.iter().map(f).collect());
    let f = |f: &dyn Fn(&MoneyRow) -> f64| Column::F64(rows.iter().map(f).collect());
    Table::new()
        .with("season", i(&|r| r.season))
        .with("gw_index", i(&|r| r.gw_index))
        .with("start_id", Column::Str(rows.iter().map(|r| r.start_id.clone()).collect()))
        .with("horizon_gws", i(&|r| r.horizon_gws))
        .with("xp_base", f(&|r| r.xp_base))
        .with("xp_bank", f(&|r| r.xp_bank))
        .with("obj_base", f(&|r| r.obj_base))
        .with("obj_bank", f(&|r| r.obj_bank))
        .with("hits_base", i(&|r| r.hits_base))
        .with("hits_bank", i(&|r| r.hits_bank))
        .with("n_transfers_base", i(&|r| r.n_transfers_base))
        .with("n_transfers_bank", i(&|r| r.n_transfers_bank))
        .with("v", f(&|r| r.v))
}

/// Mean, standard error (std with ddof 1 over √n; null below 2) and n.
pub fn stat(values: &[f64]) -> Ordered {
    let n = values.len();
    let float = |x: f64| if x.is_finite() { Ordered::Float(x) } else { Ordered::Null };
    let mean = if n == 0 { Ordered::Null } else { float(np_mean(values)) };
    let se = if n < 2 { Ordered::Null } else { float(np_std1(values) / (n as f64).sqrt()) };
    Ordered::Object(vec![
        ("mean".into(), mean),
        ("se".into(), se),
        ("n".into(), Ordered::Int(n as i128)),
    ])
}

/// v per gw_index and pooled over gw_index ≥ `POOLED_FROM` (the decision's number).
pub fn summary(rows: &[MoneyRow], extra_bank: i64, spec: &str) -> Ordered {
    let mut gws: Vec<i64> = rows.iter().map(|r| r.gw_index).collect();
    gws.sort_unstable();
    gws.dedup();
    let values = |keep: &dyn Fn(&MoneyRow) -> bool| -> Vec<f64> {
        rows.iter().filter(|r| keep(r)).map(|r| r.v).collect()
    };
    let by_gw = gws.iter().map(|&g| (g.to_string(), stat(&values(&|r| r.gw_index == g)))).collect();
    Ordered::Object(vec![
        ("spec".into(), Ordered::Str(spec.to_owned())),
        ("extra_bank".into(), Ordered::Int(extra_bank as i128)),
        ("n_states".into(), Ordered::Int(rows.len() as i128)),
        ("by_gw_index".into(), Ordered::Object(by_gw)),
        (format!("gw_index_ge_{POOLED_FROM}"), stat(&values(&|r| r.gw_index >= POOLED_FROM))),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use fplopt_optimize::GwPlan;

    fn gw(xp: f64, hits: i64) -> GwPlan {
        GwPlan {
            gw: 1,
            gw_index: 1,
            horizon: 0,
            transfers: Vec::new(),
            starters: Vec::new(),
            bench: Vec::new(),
            captain: 0,
            vice: 0,
            chip: None,
            xp,
            bench_xp: 1.0,
            objective: 0.0,
            hits,
            bank: 0,
            ft: 1,
        }
    }

    fn solved(xp_net: f64, horizon_gws: i64) -> Solved {
        Solved { xp_net, objective: xp_net, hits: 0, n_transfers: 0, horizon_gws }
    }

    #[test]
    fn outcome_is_undecayed_xp_net_of_hits() {
        let plan = Plan {
            gws: vec![gw(50.0, 1), gw(48.0, 0), gw(47.5, 2)],
            objective: 120.0,
            status: "Optimal".into(),
            mip_gap: 0.0,
            build_seconds: 0.0,
            solve_seconds: 0.0,
            n_candidates: 0,
            scenario: String::new(),
            terminal_value: 0.0,
            gain_vs_roll: None,
            bound: 0.0,
            n_nodes: 0,
        };
        let s = plan_outcome(&plan, 4);
        assert_eq!(s.xp_net, 145.5 - 12.0); // bench xP not counted
        assert_eq!((s.hits, s.horizon_gws, s.objective), (3, 3, 120.0));
    }

    #[test]
    fn v_is_per_gw_and_per_million() {
        let r = money_row(2021, 8, "template@8", &solved(300.0, 6), &solved(303.0, 6), 10).unwrap();
        assert!((r.v - 0.5).abs() < 1e-12);
        // £0.5m buying 1.5 points over 6 GWs is 0.5 per £1m too
        let r = money_row(2021, 8, "t", &solved(300.0, 6), &solved(301.5, 6), 5).unwrap();
        assert!((r.v - 0.5).abs() < 1e-12);
        assert!(money_row(2021, 8, "t", &solved(1.0, 6), &solved(1.0, 5), 10).is_err());
    }

    #[test]
    fn summary_by_gw_and_pooled() {
        let row = |gw_index: i64, v: f64| MoneyRow {
            season: 2020,
            gw_index,
            start_id: String::new(),
            horizon_gws: 6,
            xp_base: 0.0,
            xp_bank: 0.0,
            obj_base: 0.0,
            obj_bank: 0.0,
            hits_base: 0,
            hits_bank: 0,
            n_transfers_base: 0,
            n_transfers_bank: 0,
            v,
        };
        let rows = [row(1, 9.0), row(8, 0.2), row(8, 0.4), row(15, 0.6)];
        let Ordered::Object(s) = summary(&rows, 10, LIVE_SPEC) else { panic!() };
        let get = |k: &str| s.iter().find(|(n, _)| n == k).unwrap().1.clone();
        let Ordered::Object(by_gw) = get("by_gw_index") else { panic!() };
        let keys: Vec<&str> = by_gw.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, ["1", "8", "15"]);
        // gw_index 1: one row, no SE
        assert_eq!(
            by_gw[0].1,
            Ordered::Object(vec![
                ("mean".into(), Ordered::Float(9.0)),
                ("se".into(), Ordered::Null),
                ("n".into(), Ordered::Int(1)),
            ])
        );
        // pooled ≥ 8: 0.2, 0.4, 0.6 -> mean 0.4, sd 0.2, se 0.2/√3
        let Ordered::Object(pooled) = get("gw_index_ge_8") else { panic!() };
        let num = |k: &str| match pooled.iter().find(|(n, _)| n == k).unwrap().1 {
            Ordered::Float(x) => x,
            Ordered::Int(n) => n as f64,
            _ => panic!(),
        };
        assert!((num("mean") - 0.4).abs() < 1e-12);
        assert!((num("se") - 0.2 / 3f64.sqrt()).abs() < 1e-12);
        assert_eq!(num("n"), 3.0);
        assert_eq!(stat(&[]), {
            Ordered::Object(vec![
                ("mean".into(), Ordered::Null),
                ("se".into(), Ordered::Null),
                ("n".into(), Ordered::Int(0)),
            ])
        });
    }

    #[test]
    fn starts_are_kinds_at_each_gw() {
        let ids: Vec<String> =
            specs_at("template,random:3", 8).unwrap().iter().map(|s| s.start_id(None)).collect();
        assert_eq!(ids, ["template@8", "random0@8", "random1@8", "random2@8"]);
        assert!(specs_at("template@1", 8).is_err());
        assert!(specs_at("template+bank10", 8).is_err());
    }
}
