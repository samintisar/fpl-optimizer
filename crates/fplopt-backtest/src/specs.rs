//! Policies on the command line (Python `cli.PolicySpec`, `make_policy_spec`,
//! `parse_policy_spec`): `name:xp[:key=value,...]`, e.g. `greedy:rolling:threshold=2.0`,
//! `optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2,chips=1`. Values are typed as
//! Python's parsers type them, and a spec prints as Python's `str(spec)` (the experiment
//! log's family keys on it).

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use fplopt_core::pyjson::py_float_repr;
use fplopt_optimize::OptimizerParams;

use crate::policies::{ChipPolicy2, DOUBLE_LEAD, Policy};

pub const XP_MODELS: [&str; 4] = ["rolling", "ep_next", "ep_next_fade", "v1"];
pub const MINUTES_XP_MODELS: [&str; 1] = ["v1"];
pub const MINUTES_KEYS: [&str; 2] = ["bench_from_minutes", "min_minutes"];
pub const MAX_PLAN_HORIZON: i64 = 6;
pub const CHIP_SPLITS: [&str; 3] = ["pooled", "window", "double"];
/// The horizon a double-split table is estimated for.
pub const DOUBLE_TABLE_HORIZON: i64 = 6;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Float,
    Int,
    MaxHits,
    Flag,
}

impl Kind {
    fn label(self) -> &'static str {
        match self {
            Kind::Float => "float",
            Kind::Int => "int",
            Kind::MaxHits => "int or none",
            Kind::Flag => "0/1 flag",
        }
    }
}

/// A parsed parameter value (Python float, int or None).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value {
    Float(f64),
    Int(i64),
    None,
}

impl Value {
    pub fn f(self) -> f64 {
        match self {
            Value::Float(x) => x,
            Value::Int(i) => i as f64,
            Value::None => f64::NAN,
        }
    }

    pub fn i(self) -> i64 {
        match self {
            Value::Int(i) => i,
            Value::Float(x) => x as i64,
            Value::None => 0,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Float(x) => f.write_str(&py_float_repr(*x)),
            Value::Int(i) => write!(f, "{i}"),
            Value::None => f.write_str("None"),
        }
    }
}

fn params_of(name: &str) -> Option<Vec<(&'static str, Kind)>> {
    let optimizer_keys = vec![
        ("horizon", Kind::Int),
        ("decay", Kind::Float),
        ("hit_margin", Kind::Float),
        ("max_hits", Kind::MaxHits),
        ("itb_value", Kind::Float),
        ("bench_from_minutes", Kind::Flag),
        ("min_minutes", Kind::Float),
    ];
    Some(match name {
        "greedy" => vec![
            ("threshold", Kind::Float),
            ("horizon", Kind::Int),
            ("decay", Kind::Float),
            ("max_transfers", Kind::Int),
        ],
        "roll" => vec![],
        "optimizer" => {
            let mut keys = optimizer_keys;
            keys.splice(7..7, [("chips", Kind::Flag)]);
            keys.extend([
                ("chip_shrink", Kind::Float),
                ("chip_split", Kind::Int),
                ("loso", Kind::Flag),
                ("tc_rule", Kind::Float),
                ("fh_scale", Kind::Float),
            ]);
            keys
        }
        "chipgains" => optimizer_keys,
        "chiprules" => {
            let mut keys = optimizer_keys;
            keys.extend([
                ("t_tc", Kind::Float),
                ("t_bb", Kind::Float),
                ("k_fh", Kind::Int),
                ("t_wc", Kind::Float),
            ]);
            keys
        }
        _ => return None,
    })
}

const POLICIES: [&str; 5] = ["greedy", "roll", "optimizer", "chipgains", "chiprules"];

/// Python's `float(text)`.
fn py_float(text: &str) -> Option<f64> {
    let t = text.trim().replace('_', "");
    match t.to_ascii_lowercase().as_str() {
        "inf" | "+inf" | "infinity" | "+infinity" => return Some(f64::INFINITY),
        "-inf" | "-infinity" => return Some(f64::NEG_INFINITY),
        "nan" | "+nan" | "-nan" => return Some(f64::NAN),
        _ => {}
    }
    t.parse().ok()
}

/// Python's `int(text)`.
fn py_int(text: &str) -> Option<i64> {
    text.trim().replace('_', "").parse().ok()
}

fn parse_value(kind: Kind, raw: &str) -> Option<Value> {
    match kind {
        Kind::Float => py_float(raw).map(Value::Float),
        Kind::Int => py_int(raw).map(Value::Int),
        Kind::MaxHits => {
            if matches!(raw.trim().to_ascii_lowercase().as_str(), "none" | "inf") {
                return Some(Value::None);
            }
            py_int(raw).filter(|v| *v >= 0).map(Value::Int)
        }
        Kind::Flag => match raw.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" => Some(Value::Int(1)),
            "0" | "false" | "no" => Some(Value::Int(0)),
            _ => None,
        },
    }
}

/// A policy spec: name, xP model and the parameters given (sorted by key).
#[derive(Clone, Debug, PartialEq)]
pub struct PolicySpec {
    pub name: String,
    pub xp: String,
    pub params: Vec<(String, Value)>,
}

impl fmt::Display for PolicySpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.name, self.xp)?;
        if !self.params.is_empty() {
            let items: Vec<String> = self.params.iter().map(|(k, v)| format!("{k}={v}")).collect();
            write!(f, ":{}", items.join(","))?;
        }
        Ok(())
    }
}

/// The repo's `config/chips` directory.
pub fn chips_dir() -> PathBuf {
    fplopt_build::scoring::default_config_dir()
        .parent()
        .map_or_else(|| PathBuf::from("config/chips"), |c| c.join("chips"))
}

/// The terminal value table file of `chip_split`.
pub fn chip_table_file(split: i64) -> PathBuf {
    chips_dir().join(match split {
        1 => "terminal-values-window.json",
        2 => "terminal-values-double.json",
        _ => "terminal-values.json",
    })
}

fn read_json(path: &Path) -> Result<serde_json::Value> {
    Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
}

fn scaled(table: &serde_json::Value, shrink: f64) -> Result<BTreeMap<String, Vec<f64>>> {
    let Some(map) = table.as_object() else { bail!("chip values: not a table") };
    map.iter()
        .map(|(k, v)| {
            let values = v.as_array().ok_or_else(|| anyhow::anyhow!("chip values: {k}"))?;
            Ok((
                k.clone(),
                values.iter().map(|x| shrink * x.as_f64().unwrap_or(f64::NAN)).collect(),
            ))
        })
        .collect()
}

/// `chip_values.load_terminal_values`: the estimated table × `shrink`.
pub fn load_terminal_values(shrink: f64, path: &Path) -> Result<BTreeMap<String, Vec<f64>>> {
    if !shrink.is_finite() || shrink < 0.0 {
        bail!("chip_shrink must be a finite number >= 0, got {}", py_float_repr(shrink));
    }
    if !path.is_file() {
        bail!("chip_shrink: no estimated chip values at {} (fplopt chips values)", path.display());
    }
    scaled(&read_json(path)?["values"], shrink)
}

/// `chip_values.load_loso_values`: season → its leave-one-season-out table × `shrink`.
pub fn load_loso_values(
    shrink: f64,
    path: &Path,
) -> Result<BTreeMap<i64, BTreeMap<String, Vec<f64>>>> {
    if !shrink.is_finite() || shrink < 0.0 {
        bail!("chip_shrink must be a finite number >= 0, got {}", py_float_repr(shrink));
    }
    if !path.is_file() {
        bail!("loso: no estimated chip values at {} (fplopt chips values)", path.display());
    }
    let payload = read_json(path)?;
    let Some(loso) = payload.get("loso").and_then(|l| l.as_object()) else {
        bail!("{}: no leave-one-season-out tables (chips values --loso)", path.display());
    };
    loso.iter().map(|(s, t)| Ok((s.parse()?, scaled(t, shrink)?))).collect()
}

fn check_table_header(path: &Path, split: i64, lead: i64, horizon: i64) -> Result<()> {
    if !path.is_file() {
        return Ok(());
    }
    let payload = read_json(path)?;
    let name = CHIP_SPLITS[split as usize];
    if payload.get("split").and_then(|s| s.as_str()) != Some(name) {
        bail!(
            "chip_split={split}: {} is not a {name}-split table (chips values --split {name})",
            path.display()
        );
    }
    if split != 2 {
        return Ok(());
    }
    let show = |k: &str| payload.get(k).map_or("None".to_owned(), |v| v.to_string());
    if payload.get("lead").and_then(|v| v.as_i64()) != Some(lead) {
        bail!(
            "chip_split=2: {} was estimated with known-double lead {}, the policy uses {lead}",
            path.display(),
            show("lead")
        );
    }
    if payload.get("horizon").and_then(|v| v.as_i64()) != Some(horizon) {
        bail!(
            "chip_split=2: {} was estimated for horizon {}, the policy plans {horizon} GWs",
            path.display(),
            show("horizon")
        );
    }
    Ok(())
}

fn optimizer_params(params: &BTreeMap<String, Value>) -> Result<OptimizerParams> {
    let mut p = OptimizerParams::default();
    for (key, v) in params {
        match key.as_str() {
            "horizon" => p.horizon = v.i(),
            "decay" => p.decay = v.f(),
            "hit_margin" => p.hit_margin = v.f(),
            "max_hits" => p.max_hits = if *v == Value::None { None } else { Some(v.i()) },
            "itb_value" => p.itb_value = v.f(),
            "bench_from_minutes" => p.bench_from_minutes = v.i() != 0,
            "min_minutes" => p.min_minutes = v.f(),
            "fh_scale" => p.fh_scale = v.f(),
            other => bail!("unexpected keyword argument '{other}'"),
        }
    }
    p.validate()?;
    Ok(p)
}

impl PolicySpec {
    /// The policy this spec describes (Python `PolicySpec.build`).
    pub fn build(&self) -> Result<Policy> {
        let mut params: BTreeMap<String, Value> = self.params.iter().cloned().collect();
        let xp_model = self.xp.clone();
        let policy = match self.name.as_str() {
            "optimizer" => {
                let chips = params.remove("chips").is_some_and(|v| v.i() != 0);
                let shrink = params.remove("chip_shrink").map(Value::f);
                let split = params.remove("chip_split").map_or(0, Value::i);
                let loso = params.remove("loso").is_some_and(|v| v.i() != 0);
                let tc_rule = params.remove("tc_rule").map(Value::f);
                let mut extra = ChipPolicy2::default();
                if !(0..CHIP_SPLITS.len() as i64).contains(&split) {
                    bail!("chip_split must be one of 0..{}", CHIP_SPLITS.len() - 1);
                }
                if (split != 0 || loso) && shrink.is_none() {
                    bail!("chip_split and loso need chip_shrink");
                }
                let mut opt = optimizer_params(&params)?;
                if let Some(shrink) = shrink {
                    if !chips {
                        bail!("chip_shrink needs chips=1");
                    }
                    let path = chip_table_file(split);
                    if split != 0 {
                        check_table_header(&path, split, DOUBLE_LEAD, opt.horizon)?;
                    }
                    if loso {
                        extra.season_tables = Some(load_loso_values(shrink, &path)?);
                    } else {
                        opt.chip_value_left = Some(load_terminal_values(shrink, &path)?);
                    }
                }
                if split == 2 {
                    extra.double_lead = Some(DOUBLE_LEAD);
                }
                if let Some(t) = tc_rule {
                    opt.exclude_chips = vec!["3xc".to_owned()];
                    extra.tc_rule = Some(t);
                }
                opt.validate()?;
                Policy::Optimizer { xp_model, params: opt, chips, extra }
            }
            "chipgains" | "chiprules" => {
                let mut rule = |k: &str, default: Value| params.remove(k).unwrap_or(default);
                let (t_tc, t_bb, k_fh, t_wc) = (
                    rule("t_tc", Value::Float(10.0)),
                    rule("t_bb", Value::Float(12.0)),
                    rule("k_fh", Value::Int(3)),
                    rule("t_wc", Value::Float(10.0)),
                );
                let opt = optimizer_params(&params)?;
                if self.name == "chipgains" {
                    Policy::ChipGains { xp_model, params: opt }
                } else {
                    Policy::ChipRules {
                        xp_model,
                        params: opt,
                        t_tc: t_tc.f(),
                        t_bb: t_bb.f(),
                        k_fh: k_fh.i(),
                        t_wc: t_wc.f(),
                    }
                }
            }
            "greedy" => Policy::Greedy {
                xp_model,
                threshold: params.get("threshold").map_or(1.0, |v| v.f()),
                horizon: params.get("horizon").map_or(6, |v| v.i()),
                decay: params.get("decay").map_or(0.85, |v| v.f()),
                max_transfers: params.get("max_transfers").map_or(1, |v| v.i()),
            },
            _ => Policy::Roll { xp_model },
        };
        policy.validate()?;
        Ok(policy)
    }
}

/// Chip spec keys dropped when given at their default.
fn chip_key_default(key: &str) -> Option<Value> {
    match key {
        "chip_split" => Some(Value::Int(0)),
        "loso" => Some(Value::Int(0)),
        "fh_scale" => Some(Value::Float(1.0)),
        _ => None,
    }
}

fn values_equal(a: Value, b: Value) -> bool {
    match (a, b) {
        (Value::None, Value::None) => true,
        (Value::None, _) | (_, Value::None) => false,
        (x, y) => x.f() == y.f(),
    }
}

/// A checked spec (Python `make_policy_spec`; errors as argparse's messages).
pub fn make_policy_spec(name: &str, xp: &str, given: &[(String, String)]) -> Result<PolicySpec> {
    let Some(allowed) = params_of(name) else {
        bail!("unknown policy '{name}' (policies: {})", POLICIES.join(", "));
    };
    if !XP_MODELS.contains(&xp) {
        bail!("unknown xP model '{xp}' (models: {})", XP_MODELS.join(", "));
    }
    let mut values: Vec<(String, Value)> = Vec::new();
    for (key, raw) in given {
        let Some((_, kind)) = allowed.iter().find(|(k, _)| k == key) else {
            let known: Vec<&str> = allowed.iter().map(|(k, _)| *k).collect();
            let known = if known.is_empty() { "none".to_owned() } else { known.join(", ") };
            bail!("policy '{name}' has no parameter '{key}' (parameters: {known})");
        };
        let Some(value) = parse_value(*kind, raw) else {
            bail!("{name} {key}='{raw}': not a valid {}", kind.label());
        };
        if let Value::Float(x) = value
            && !x.is_finite()
        {
            bail!("{name} {key}='{raw}': not finite");
        }
        values.push((key.clone(), value));
    }
    if name == "optimizer" {
        values.retain(|(k, v)| !chip_key_default(k).is_some_and(|d| values_equal(*v, d)));
    }
    values.sort_by(|a, b| a.0.cmp(&b.0));
    let spec = PolicySpec { name: name.to_owned(), xp: xp.to_owned(), params: values };
    let get = |k: &str| spec.params.iter().find(|(key, _)| key == k).map(|(_, v)| *v);
    let horizon = get("horizon").map_or(1, Value::i);
    if !(1..=MAX_PLAN_HORIZON).contains(&horizon) {
        bail!("bad policy {spec}: need 1 <= horizon <= {MAX_PLAN_HORIZON} (the xP frames' GWs)");
    }
    if name == "greedy" && get("max_transfers").map_or(0, Value::i) < 0 {
        bail!("bad policy {spec}: need max_transfers >= 0");
    }
    let planner = matches!(name, "optimizer" | "chipgains" | "chiprules");
    if planner && !MINUTES_XP_MODELS.contains(&xp) {
        let ignored: Vec<&str> =
            MINUTES_KEYS.iter().copied().filter(|k| get(k).is_some()).collect();
        if !ignored.is_empty() {
            bail!(
                "bad policy {spec}: {} only apply to xP models with minutes ({})",
                ignored.join(", "),
                MINUTES_XP_MODELS.join(", ")
            );
        }
    }
    if name == "optimizer"
        && get("chip_shrink").is_some()
        && get("chips").is_none_or(|v| v.i() == 0)
    {
        bail!("bad policy {spec}: chip_shrink needs chips=1");
    }
    if planner {
        let decay = get("decay").map_or(0.5, Value::f);
        if !(decay > 0.0 && decay <= 1.0) {
            bail!("bad policy {spec}: need 0 < decay <= 1");
        }
    }
    if let Err(e) = spec.build() {
        bail!("bad policy {spec}: {e}");
    }
    Ok(spec)
}

/// `name:xp[:key=value,...]` -> PolicySpec.
pub fn parse_policy_spec(text: &str) -> Result<PolicySpec> {
    let parts: Vec<&str> = text.trim().split(':').map(str::trim).collect();
    if !(2..=3).contains(&parts.len()) || parts[0].is_empty() || parts[1].is_empty() {
        bail!(
            "bad policy '{text}': expected name:xp[:key=value,...], e.g. greedy:rolling or \
             greedy:rolling:threshold=2.0"
        );
    }
    let mut given: Vec<(String, String)> = Vec::new();
    if parts.len() == 3 {
        for item in parts[2].split(',') {
            let (key, value) =
                item.split_once('=').map_or((item.trim(), ""), |(k, v)| (k.trim(), v.trim()));
            if key.is_empty() || value.is_empty() || !item.contains('=') {
                bail!("bad policy parameter '{item}' in '{text}'");
            }
            if given.iter().any(|(k, _)| k == key) {
                bail!("duplicate policy parameter '{key}' in '{text}'");
            }
            given.push((key.to_owned(), value.to_owned()));
        }
    }
    make_policy_spec(parts[0], parts[1], &given)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs_print_and_name_as_python() {
        let s =
            parse_policy_spec("optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2").unwrap();
        assert_eq!(s.to_string(), "optimizer:v1:decay=0.75,hit_margin=2.0,horizon=6,max_hits=1");
        assert_eq!(s.build().unwrap().name(), "optimizer(v1,mh=1,m=2.0,d=0.75)");
        let g = parse_policy_spec("greedy:rolling:threshold=2").unwrap();
        assert_eq!(g.build().unwrap().name(), "greedy(rolling,t=2.0)");
        let n = parse_policy_spec("optimizer:ep_next:max_hits=none,chip_split=0").unwrap();
        assert_eq!(n.to_string(), "optimizer:ep_next:max_hits=None");
        assert!(parse_policy_spec("optimizer:ep_next:min_minutes=90").is_err());
        assert!(parse_policy_spec("greedy:rolling:horizon=7").is_err());
    }
}
