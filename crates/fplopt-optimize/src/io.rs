//! Planner cases and plans as files, for the batch 4 parity gate (`dev/optimize_parity.py`):
//! a case directory holds `case.json` (season, squad state, planner settings, top_k, chips),
//! `pool.parquet` and `xp.parquet`; `planset_json` writes what the gate compares with
//! Python's `PlanSet` (the same JSON layout the gate writes for Python).

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use fplopt_build::table::read_parquet;
use serde_json::{Value, json};

use crate::params::OptimizerParams;
use crate::plans::{Plan, PlanSet};
use crate::problem::{PlanInput, XpFrame};
use crate::search::Search;
use crate::state::{Holding, PoolPlayer, SquadState};

/// One planner call: inputs and settings.
pub struct Case {
    pub season: i64,
    pub state: SquadState,
    pub params: OptimizerParams,
    pub top_k: usize,
    pub chips: bool,
    pub search: Search,
    pub pool: Vec<PoolPlayer>,
    pub xp: XpFrame,
}

fn int(v: &Value, key: &str) -> Result<i64> {
    v.get(key).and_then(Value::as_i64).with_context(|| format!("case: '{key}' is not an integer"))
}

fn pairs(v: &Value, key: &str) -> Result<Vec<Vec<i64>>> {
    let items =
        v.get(key).and_then(Value::as_array).with_context(|| format!("case: no '{key}'"))?;
    items
        .iter()
        .map(|item| {
            item.as_array()
                .context("case: expected a list")?
                .iter()
                .map(|x| x.as_i64().context("case: expected integers"))
                .collect()
        })
        .collect()
}

/// `OptimizerParams` from a JSON object of Python's field names; missing keys keep the
/// defaults, null means None where Python allows it.
pub fn params_from_json(v: &Value) -> Result<OptimizerParams> {
    let mut p = OptimizerParams::default();
    let Some(map) = v.as_object() else { bail!("params: expected an object") };
    let float =
        |x: &Value, k: &str| x.as_f64().with_context(|| format!("params.{k}: not a number"));
    for (key, x) in map {
        match key.as_str() {
            "horizon" => p.horizon = x.as_i64().context("params.horizon")?,
            "decay" => p.decay = float(x, key)?,
            "itb_value" => p.itb_value = float(x, key)?,
            "hit_margin" => p.hit_margin = float(x, key)?,
            "max_hits" => p.max_hits = x.as_i64(),
            "mip_gap" => p.mip_gap = float(x, key)?,
            "threads" => p.threads = x.as_i64().context("params.threads")?,
            "time_limit" => p.time_limit = x.as_f64(),
            "node_limit" => p.node_limit = x.as_i64(),
            "tie_epsilon" => p.tie_epsilon = float(x, key)?,
            "bench_from_minutes" => {
                p.bench_from_minutes =
                    x.as_bool().or(x.as_i64().map(|i| i != 0)).context(key.clone())?
            }
            "min_minutes" => p.min_minutes = float(x, key)?,
            "prune_dominated" => p.prune_dominated = x.as_bool().context(key.clone())?,
            "bench_weights" => {
                p.bench_weights = x
                    .as_array()
                    .context("params.bench_weights")?
                    .iter()
                    .map(|w| float(w, key))
                    .collect::<Result<_>>()?
            }
            "ft_value" => p.ft_value = int_keyed(x, key)?,
            "chip_value" => {
                p.chip_value = x
                    .as_object()
                    .context("params.chip_value")?
                    .iter()
                    .map(|(k, w)| Ok((k.clone(), float(w, key)?)))
                    .collect::<Result<_>>()?
            }
            "prune_n" => {
                p.prune_n = if x.is_null() {
                    None
                } else {
                    Some(
                        int_keyed(x, key)?
                            .into_iter()
                            .map(|(k, v)| (k, v as i64))
                            .collect::<BTreeMap<i64, i64>>(),
                    )
                }
            }
            "chip_value_left" => {
                p.chip_value_left = if x.is_null() {
                    None
                } else {
                    Some(
                        x.as_object()
                            .context("params.chip_value_left")?
                            .iter()
                            .map(|(k, w)| {
                                let values = w.as_array().context("chip_value_left")?;
                                Ok((
                                    k.clone(),
                                    values.iter().map(|v| float(v, key)).collect::<Result<_>>()?,
                                ))
                            })
                            .collect::<Result<_>>()?,
                    )
                }
            }
            other => bail!("params: unknown key '{other}'"),
        }
    }
    p.validate()?;
    Ok(p)
}

fn int_keyed(x: &Value, key: &str) -> Result<BTreeMap<i64, f64>> {
    x.as_object()
        .with_context(|| format!("params.{key}: expected an object"))?
        .iter()
        .map(|(k, v)| {
            let k: i64 = k.parse().with_context(|| format!("params.{key}: key '{k}'"))?;
            Ok((k, v.as_f64().with_context(|| format!("params.{key}[{k}]"))?))
        })
        .collect()
}

pub fn read_case(dir: &Path) -> Result<Case> {
    let text = std::fs::read_to_string(dir.join("case.json"))
        .with_context(|| format!("{}/case.json", dir.display()))?;
    let v: Value = serde_json::from_str(&text)?;
    let s = v.get("state").context("case: no 'state'")?;
    let holdings = pairs(s, "holdings")?
        .into_iter()
        .map(|h| match h[..] {
            [player_key, element_type, team_key, purchase_price, price] => {
                Ok(Holding { player_key, element_type, team_key, purchase_price, price })
            }
            _ => bail!("case: a holding needs 5 values"),
        })
        .collect::<Result<Vec<_>>>()?;
    let mut state = SquadState::new(
        int(s, "season")?,
        int(s, "gw_index")?,
        holdings,
        int(s, "bank")?,
        int(s, "free_transfers")?,
    )?;
    state.chips_used = pairs(s, "chips_used")?.into_iter().map(|c| (c[0], c[1])).collect();
    let search = match v.get("search").and_then(Value::as_str).unwrap_or("bound") {
        "bound" => Search::Bound,
        "all" => Search::All,
        other => bail!("case: unknown search '{other}'"),
    };
    let pool = read_parquet(&dir.join("pool.parquet"), None)?;
    let (keys, et, team, price) = (
        pool.i64("player_key")?,
        pool.i64("element_type")?,
        pool.i64("team_key")?,
        pool.i64("price")?,
    );
    let pool = (0..keys.len())
        .map(|i| PoolPlayer {
            player_key: keys[i],
            element_type: et[i],
            team_key: team[i],
            price: price[i],
        })
        .collect();
    let xp = XpFrame::from_table(&read_parquet(&dir.join("xp.parquet"), None)?)?;
    Ok(Case {
        season: int(&v, "season")?,
        state,
        params: params_from_json(v.get("params").unwrap_or(&json!({})))?,
        top_k: int(&v, "top_k")? as usize,
        chips: v.get("chips").and_then(Value::as_bool).context("case: no 'chips'")?,
        search,
        pool,
        xp,
    })
}

pub fn plan_json(p: &Plan) -> Value {
    json!({
        "scenario": p.scenario,
        "objective": p.objective,
        "terminal_value": p.terminal_value,
        "total_objective": p.total_objective(),
        "status": p.status,
        "mip_gap": p.mip_gap,
        "n_nodes": p.n_nodes,
        "bound": if p.bound.is_finite() { json!(p.bound) } else { json!("inf") },
        "gain_vs_roll": p.gain_vs_roll,
        "n_candidates": p.n_candidates,
        "gws": p.gws.iter().map(|g| json!({
            "gw": g.gw,
            "gw_index": g.gw_index,
            "horizon": g.horizon,
            "transfers": g.transfers.iter().map(|t| [t.out_key, t.in_key]).collect::<Vec<_>>(),
            "starters": g.starters,
            "bench": g.bench,
            "captain": g.captain,
            "vice": g.vice,
            "chip": g.chip,
            "xp": g.xp,
            "bench_xp": g.bench_xp,
            "objective": g.objective,
            "hits": g.hits,
            "bank": g.bank,
            "ft": g.ft,
        })).collect::<Vec<_>>(),
    })
}

fn finite(x: f64) -> Value {
    if x.is_finite() { json!(x) } else { json!("inf") }
}

/// The problem summary (candidates, horizon, bench weights) and the plan set.
pub fn planset_json(set: &PlanSet, problem: &PlanInput) -> Value {
    json!({
        "problem": {
            "candidates": problem.players.iter().map(|p| p.player_key).collect::<Vec<_>>(),
            "gws": problem.gws.iter().map(|g| [g.gw, g.gw_index, g.horizon]).collect::<Vec<_>>(),
            "bench_weights": problem.bench_weights,
            "n_pool": problem.n_pool,
        },
        "plans": set.plans.iter().map(plan_json).collect::<Vec<_>>(),
        "roll": set.roll.as_ref().map(plan_json),
        "scenario_objectives": set.scenario_objectives.iter().map(|(k, v)| json!([k, v])).collect::<Vec<_>>(),
        "scenario_bounds": set.scenario_bounds.iter().map(|(k, v)| json!([k, finite(*v)])).collect::<Vec<_>>(),
        "search": {
            "search": set.search.search,
            "n_scenarios": set.search.n_scenarios,
            "n_relaxations": set.search.n_relaxations,
            "n_solves": set.search.n_solves,
            "not_optimal": set.search.not_optimal,
        },
        "solver_status": set.solver_status(),
        "seconds": set.seconds,
    })
}
