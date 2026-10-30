//! Chip scenarios and the terminal value of unused chips (Python `fplopt.optimize.chips`;
//! PLAN §7 *Chips*). Each allowed chip assignment in the horizon is a fixed scenario solved
//! separately: no chip; each chip at each horizon GW; pairs in two different GWs. A
//! scenario's keys `t` are positions in `PlanInput.gws`.

use std::collections::BTreeMap;

use fplopt_build::scoring::ChipWindow;

use crate::params::OptimizerParams;
use crate::problem::PlanInput;
use crate::state::{InvalidDecision, chip_window};
use crate::util::py_sum;

/// A fixed chip assignment: `chips` (t, name) sorted by t, `chip_ids` the windows they use,
/// `label` e.g. `none`, `bboost@gw7`, `wildcard@gw5+freehit@gw9`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ChipScenario {
    pub chips: Vec<(usize, String)>,
    pub chip_ids: Vec<i64>,
    pub label: String,
}

impl ChipScenario {
    pub fn none() -> ChipScenario {
        ChipScenario { chips: Vec::new(), chip_ids: Vec::new(), label: "none".to_string() }
    }

    pub fn by_t(&self) -> BTreeMap<usize, &str> {
        self.chips.iter().map(|(t, n)| (*t, n.as_str())).collect()
    }

    pub fn chip_at(&self, t: usize) -> Option<&str> {
        self.chips.iter().find(|(s, _)| *s == t).map(|(_, n)| n.as_str())
    }

    /// The scenario with the chip at position `t` removed (chip ids re-assigned).
    pub fn without(&self, t: usize, problem: &PlanInput) -> Result<ChipScenario, InvalidDecision> {
        let kept: BTreeMap<usize, String> =
            self.chips.iter().filter(|(s, _)| *s != t).map(|(s, n)| (*s, n.clone())).collect();
        make_scenario(problem, &kept)
    }
}

/// Validate a chip assignment (position → name) in GW order and assign each chip its
/// `chip_id`. Positions outside the horizon are a programming error (panic).
pub fn make_scenario(
    problem: &PlanInput,
    chips: &BTreeMap<usize, String>,
) -> Result<ChipScenario, InvalidDecision> {
    let mut state = problem.state.clone();
    let (mut pairs, mut ids, mut parts) = (Vec::new(), Vec::new(), Vec::new());
    for (&t, name) in chips {
        assert!(t < problem.gws.len(), "chip position {t} outside the horizon");
        let gw = problem.gws[t];
        state.gw_index = gw.gw_index;
        let chip_id = chip_window(&state, name, &problem.rules)?;
        state.chips_used.push((chip_id, gw.gw_index));
        pairs.push((t, name.clone()));
        ids.push(chip_id);
        parts.push(format!("{name}@gw{}", gw.gw));
    }
    let label = if parts.is_empty() { "none".to_string() } else { parts.join("+") };
    Ok(ChipScenario { chips: pairs, chip_ids: ids, label })
}

/// Every scenario the rules allow: no chip; singles by (position, name); pairs by (first
/// position, first name, second position, second name); names in `params.exclude_chips`
/// are left out.
pub fn scenarios(problem: &PlanInput, params: &OptimizerParams) -> Vec<ChipScenario> {
    let mut names: Vec<&str> = problem
        .rules
        .chips
        .iter()
        .map(|c| c.name.as_str())
        .filter(|n| !params.exclude_chips.iter().any(|e| e == n))
        .collect();
    names.sort_unstable();
    names.dedup();
    let one =
        |t: usize, name: &str| make_scenario(problem, &BTreeMap::from([(t, name.to_string())]));
    let singles: Vec<ChipScenario> = (0..problem.gws.len())
        .flat_map(|t| names.iter().filter_map(move |n| one(t, n).ok()))
        .collect();
    let mut pairs = Vec::new();
    for (i, s1) in singles.iter().enumerate() {
        for s2 in &singles[i + 1..] {
            let (t1, n1) = &s1.chips[0];
            let (t2, n2) = &s2.chips[0];
            if t1 != t2 {
                let chips = BTreeMap::from([(*t1, n1.clone()), (*t2, n2.clone())]);
                if let Ok(pair) = make_scenario(problem, &chips) {
                    pairs.push(pair);
                }
            }
        }
    }
    let mut out = vec![ChipScenario::none()];
    out.extend(singles);
    out.extend(pairs);
    out
}

/// Σ chip values over the windows unused after the scenario that outlast the last horizon
/// gw_index (in chip_id order, summed from 0). Each window is valued by its ordinal among the
/// chip's windows and whether `problem.doubles` has a double in its part after the horizon.
pub fn terminal_value(
    problem: &PlanInput,
    params: &OptimizerParams,
    scenario: &ChipScenario,
) -> f64 {
    let used = |cid: i64| {
        problem.state.chips_used.iter().any(|(c, _)| *c == cid) || scenario.chip_ids.contains(&cid)
    };
    let last = problem.gws.last().map_or(0, |g| g.gw_index);
    let all = &problem.rules.chips;
    let ordinal = |w: &ChipWindow| {
        1 + all.iter().filter(|o| o.name == w.name && o.start < w.start).count() as i64
    };
    let mut windows: Vec<_> = all.iter().collect();
    windows.sort_by_key(|w| w.chip_id);
    py_sum(windows.into_iter().filter(|w| !used(w.chip_id) && w.stop > last).map(|w| {
        let from = last.max(w.start - 1);
        let double = problem.doubles.iter().any(|&g| from < g && g <= w.stop);
        chip_value(params, &w.name, w.stop - last, ordinal(w), double)
    }))
}

/// The value of holding chip `name` with `left` (≥ 1) GWs of its window after the horizon,
/// from `params.chip_value_left` (its last entry for a longer window): the first key of
/// `name@window+double` (0/1), `name@window`, `name` the table has; else the flat
/// `params.chip_value[name]` (0 if missing).
pub fn chip_value(
    params: &OptimizerParams,
    name: &str,
    left: i64,
    window: i64,
    double: bool,
) -> f64 {
    if let Some(table) = &params.chip_value_left {
        let keys = [
            format!("{name}@{window}+{}", i64::from(double)),
            format!("{name}@{window}"),
            name.to_string(),
        ];
        if let Some(values) = keys.iter().find_map(|k| table.get(k)) {
            let i = (left.min(values.len() as i64) - 1).max(0) as usize;
            return values[i];
        }
    }
    params.chip_value.get(name).copied().unwrap_or(0.0)
}
