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

use crate::policies::{ChipPolicy2, DOUBLE_LEAD, Policy, PriceSource};

pub const XP_MODELS: [&str; 6] = ["rolling", "ep_next", "ep_next_fade", "v1", "v1r", "v1-2026-10"];
pub const MINUTES_XP_MODELS: &[&str] = fplopt_models::V1_FAMILY;
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
    Prices,
}

impl Kind {
    fn label(self) -> &'static str {
        match self {
            Kind::Float => "float",
            Kind::Int => "int",
            Kind::MaxHits => "int or none",
            Kind::Flag => "0/1 flag",
            Kind::Prices => "price source (fixed, oracle)",
        }
    }
}

/// A parsed parameter value (Python float, int or None; `Text` for a named choice, e.g.
/// `prices=oracle`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value {
    Float(f64),
    Int(i64),
    None,
    Text(&'static str),
}

impl Value {
    pub fn f(self) -> f64 {
        match self {
            Value::Float(x) => x,
            Value::Int(i) => i as f64,
            Value::None | Value::Text(_) => f64::NAN,
        }
    }

    pub fn i(self) -> i64 {
        match self {
            Value::Int(i) => i,
            Value::Float(x) => x as i64,
            Value::None | Value::Text(_) => 0,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Float(x) => f.write_str(&py_float_repr(*x)),
            Value::Int(i) => write!(f, "{i}"),
            Value::None => f.write_str("None"),
            Value::Text(t) => f.write_str(t),
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
        ("price_k", Kind::Float),
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
                ("prices", Kind::Prices),
                ("xp_jitter", Kind::Float),
                ("jitter_seed", Kind::Int),
                ("early", Kind::Int),
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
        Kind::Prices => {
            let raw = raw.trim().to_ascii_lowercase();
            PRICE_SOURCES.iter().find(|s| **s == raw).map(|s| Value::Text(s))
        }
    }
}

/// `prices=` values: `fixed` (default), `oracle` and `forecast` (v1 xP only).
const PRICE_SOURCES: [&str; 3] = ["fixed", "oracle", "forecast"];

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
            "price_k" => p.price_value_k = v.f(),
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
                let sigma = params.remove("xp_jitter").map_or(0.0, Value::f);
                let seed = params.remove("jitter_seed").map(Value::i);
                if sigma < 0.0 {
                    bail!("xp_jitter must be >= 0, got {}", py_float_repr(sigma));
                }
                if sigma > 0.3 {
                    bail!("xp_jitter must be <= 0.3: larger values can flip the sign of xP");
                }
                if seed.is_some_and(|s| s < 0) {
                    bail!("jitter_seed must be >= 0");
                }
                if seed.is_some() && sigma == 0.0 {
                    bail!("jitter_seed needs xp_jitter > 0");
                }
                let jitter = (sigma > 0.0)
                    .then(|| crate::jitter::XpJitter { sigma, seed: seed.unwrap_or(0) as u64 });
                if jitter.is_some() && tc_rule.is_some() {
                    bail!("xp_jitter can't be combined with tc_rule (the rule reads the true xP)");
                }
                let early = match params.remove("early").map_or(0, Value::i) {
                    0 => false,
                    1 => true,
                    v => bail!("early must be 0 or 1, got {v}"),
                };
                let prices = match params.remove("prices") {
                    None | Some(Value::Text("fixed")) => PriceSource::Fixed,
                    Some(Value::Text("oracle")) => PriceSource::Oracle,
                    Some(Value::Text("forecast")) => PriceSource::Forecast,
                    Some(other) => {
                        bail!("prices={other}: not a price source (fixed, oracle, forecast)")
                    }
                };
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
                Policy::Optimizer { xp_model, params: opt, chips, extra, prices, jitter, early }
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

/// Spec keys dropped when given at their default: the optimizer's chip keys, and the price
/// keys on every planner.
fn key_default(name: &str, key: &str) -> Option<Value> {
    match (name, key) {
        ("optimizer", "chip_split") => Some(Value::Int(0)),
        ("optimizer", "loso") => Some(Value::Int(0)),
        ("optimizer", "fh_scale") => Some(Value::Float(1.0)),
        ("optimizer", "xp_jitter") => Some(Value::Float(0.0)),
        ("optimizer", "jitter_seed") => Some(Value::Int(0)),
        ("optimizer", "early") => Some(Value::Int(0)),
        (_, "price_k") => Some(Value::Float(0.0)),
        (_, "prices") => Some(Value::Text("fixed")),
        _ => None,
    }
}

fn values_equal(a: Value, b: Value) -> bool {
    match (a, b) {
        (Value::None, Value::None) => true,
        (Value::Text(x), Value::Text(y)) => x == y,
        (Value::None | Value::Text(_), _) | (_, Value::None | Value::Text(_)) => false,
        (x, y) => x.f() == y.f(),
    }
}

/// A checked spec (Python `make_policy_spec`; errors as argparse's messages).
pub fn make_policy_spec(name: &str, xp: &str, given: &[(String, String)]) -> Result<PolicySpec> {
    let Some(allowed) = params_of(name) else {
        bail!("unknown policy '{name}' (policies: {})", POLICIES.join(", "));
    };
    // A value-of-information oracle (`info_oracle`: v1 with one input's truth) is a backtest
    // model only, for the plain no-chip, fixed-price, deadline planner.
    let oracle = crate::info_oracle::Arm::parse(xp).is_some();
    if !XP_MODELS.contains(&xp) && !oracle {
        bail!(
            "unknown xP model '{xp}' (models: {}; backtest oracles: {})",
            XP_MODELS.join(", "),
            crate::info_oracle::model_names().join(", ")
        );
    }
    if oracle && name != "optimizer" {
        bail!("xP model '{xp}' is a backtest oracle for the optimizer policy only");
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
    values.retain(|(k, v)| !key_default(name, k).is_some_and(|d| values_equal(*v, d)));
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
    if oracle {
        let refused: Vec<&str> = ["chips", "prices", "early", "price_k", "tc_rule"]
            .into_iter()
            .filter(|k| get(k).is_some())
            .collect();
        if !refused.is_empty() {
            bail!("bad policy {spec}: an oracle xP model takes no {}", refused.join(", "));
        }
    }
    if planner && !MINUTES_XP_MODELS.contains(&xp) && !oracle {
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
    fn chip_rules_h_is_the_preregistered_spec() {
        let spec = parse_policy_spec(
            "chiprules:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2,t_tc=8.3,t_bb=10.9,\
             k_fh=4,t_wc=12.87",
        )
        .unwrap();
        let Policy::ChipRules { t_tc, t_bb, k_fh, t_wc, .. } = spec.build().unwrap() else {
            panic!("not a chiprules policy");
        };
        let h = fplopt_optimize::chip_rules::H;
        assert_eq!((t_tc, t_bb, k_fh, t_wc), (h.t_tc, h.t_bb, h.k_fh, h.t_wc));
    }

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

    #[test]
    fn price_keys_parse_and_label() {
        let base = "optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2";
        let spec = parse_policy_spec(&format!("{base},prices=oracle,price_k=1")).unwrap();
        assert_eq!(
            spec.to_string(),
            "optimizer:v1:decay=0.75,hit_margin=2.0,horizon=6,max_hits=1,price_k=1.0,prices=oracle"
        );
        let policy = spec.build().unwrap();
        assert_eq!(policy.name(), "optimizer(v1,mh=1,m=2.0,d=0.75,pk=1.0,oracle)");
        assert_eq!(policy.price_source(), PriceSource::Oracle);
        let Policy::Optimizer { params, prices, .. } = &policy else { panic!("not an optimizer") };
        assert_eq!((params.price_value_k, *prices), (1.0, PriceSource::Oracle));
        // defaults are dropped: same spec, same name as without the keys
        let plain = parse_policy_spec(&format!("{base},prices=fixed,price_k=0")).unwrap();
        let bare = parse_policy_spec(base).unwrap();
        assert_eq!(plain, bare);
        assert_eq!(plain.build().unwrap().name(), bare.build().unwrap().name());
        assert_eq!(bare.build().unwrap().price_source(), PriceSource::Fixed);
        // the price forecast's paths: optimizer on v1 xP only
        let forecast = parse_policy_spec(&format!("{base},prices=forecast,price_k=0.5")).unwrap();
        let policy = forecast.build().unwrap();
        assert_eq!(policy.name(), "optimizer(v1,mh=1,m=2.0,d=0.75,pk=0.5,forecast)");
        assert_eq!(policy.price_source(), PriceSource::Forecast);
        let err = parse_policy_spec("optimizer:rolling:prices=forecast").unwrap_err().to_string();
        assert!(err.contains("the price forecast uses v1 xP"), "{err}");
        assert!(parse_policy_spec("chipgains:v1:prices=forecast").is_err());
        assert!(parse_policy_spec("optimizer:v1:prices=bogus").is_err());
        assert!(parse_policy_spec("optimizer:v1:price_k=-1").is_err());
        // price_k on the chip study's planners; prices only on the optimizer
        let gains = parse_policy_spec("chipgains:v1:price_k=0.5").unwrap().build().unwrap();
        assert!(gains.name().contains("pk=0.5"), "{}", gains.name());
        assert!(parse_policy_spec("chipgains:v1:price_k=0").unwrap().params.is_empty());
        assert!(parse_policy_spec("chiprules:v1:prices=oracle").is_err());
    }

    #[test]
    fn jitter_keys_parse_and_label() {
        let base = "optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2";
        let spec = parse_policy_spec(&format!("{base},xp_jitter=0.05,jitter_seed=3")).unwrap();
        let policy = spec.build().unwrap();
        assert_eq!(policy.name(), "optimizer(v1,mh=1,m=2.0,d=0.75,jit=0.05/3)");
        let Policy::Optimizer { jitter, .. } = &policy else { panic!("not an optimizer") };
        assert_eq!(*jitter, Some(crate::jitter::XpJitter { sigma: 0.05, seed: 3 }));
        // the default (0) is dropped and leaves the live planner's name and spec unchanged
        let plain = parse_policy_spec(&format!("{base},xp_jitter=0")).unwrap();
        assert_eq!(plain, parse_policy_spec(base).unwrap());
        assert_eq!(plain.build().unwrap().name(), "optimizer(v1,mh=1,m=2.0,d=0.75)");
        // seed without a wobble, negative sigma, and the key on other policies are refused
        assert!(parse_policy_spec(&format!("{base},jitter_seed=3")).is_err());
        assert!(parse_policy_spec(&format!("{base},xp_jitter=-0.1")).is_err());
        assert!(parse_policy_spec(&format!("{base},xp_jitter=0.05,jitter_seed=-1")).is_err());
        assert!(parse_policy_spec("chipgains:v1:xp_jitter=0.05").is_err());
        // tc_rule reads the true xP, so it can't be combined with the wobble
        let err =
            parse_policy_spec("optimizer:v1:tc_rule=8,xp_jitter=0.05").unwrap_err().to_string();
        assert!(err.contains("xp_jitter can't be combined with tc_rule"), "{err}");
        // 0.3 is the cap
        assert!(parse_policy_spec(&format!("{base},xp_jitter=0.31")).is_err());
        assert!(parse_policy_spec(&format!("{base},xp_jitter=0.3")).is_ok());
        // seed defaults to 0 and is dropped there: one canonical spec
        assert_eq!(
            parse_policy_spec(&format!("{base},xp_jitter=0.05")).unwrap(),
            parse_policy_spec(&format!("{base},xp_jitter=0.05,jitter_seed=0")).unwrap()
        );
        let s0 = parse_policy_spec(&format!("{base},xp_jitter=0.05")).unwrap().build().unwrap();
        assert!(s0.name().ends_with("jit=0.05/0)"), "{}", s0.name());
    }

    #[test]
    fn oracle_xp_models_are_optimizer_only_without_chips_prices_or_early() {
        let live = "horizon=6,decay=0.75,max_hits=1,hit_margin=2";
        let spec = parse_policy_spec(&format!("optimizer:v1+oracle-M0:{live}")).unwrap();
        assert_eq!(spec.xp, "v1+oracle-M0");
        assert!(spec.build().unwrap().name().contains("v1+oracle-M0"));
        // the minutes keys apply, as for v1
        assert!(parse_policy_spec("optimizer:v1+oracle-MA:min_minutes=90").is_ok());
        let err = |s: &str| parse_policy_spec(s).unwrap_err().to_string();
        assert!(err("optimizer:v1+oracle-ZZ").contains("unknown xP model"));
        assert!(err("optimizer:v1+oracle-ZZ").contains("v1+oracle-M0"));
        assert!(err("greedy:v1+oracle-M0").contains("optimizer policy only"));
        assert!(err("optimizer:v1+oracle-M0:chips=1").contains("takes no chips"));
        assert!(err("optimizer:v1+oracle-M0:prices=oracle,price_k=1").contains("takes no"));
        assert!(err("optimizer:v1+oracle-M0:early=1").contains("takes no early"));
    }

    #[test]
    fn early_key_parses_labels_and_refuses_what_it_cannot_combine_with() {
        let base = "optimizer:v1:horizon=6,decay=0.75,max_hits=1,hit_margin=2";
        let policy = parse_policy_spec(&format!("{base},early=1")).unwrap().build().unwrap();
        assert_eq!(policy.name(), "optimizer(v1,mh=1,m=2.0,d=0.75,early)");
        let Policy::Optimizer { early, .. } = &policy else { panic!("not an optimizer") };
        assert!(*early);
        // the default (0) is dropped: the live planner's spec and name are unchanged
        let plain = parse_policy_spec(&format!("{base},early=0")).unwrap();
        assert_eq!(plain, parse_policy_spec(base).unwrap());
        assert_eq!(plain.build().unwrap().name(), "optimizer(v1,mh=1,m=2.0,d=0.75)");
        // only 0/1, and not with chips, moving prices or the rule Triple Captain
        assert!(parse_policy_spec(&format!("{base},early=2")).is_err());
        for extra in ["chips=1", "prices=forecast", "prices=oracle", "tc_rule=8"] {
            let spec = parse_policy_spec(&format!("{base},early=1,{extra}"));
            assert!(spec.and_then(|s| s.build()).is_err(), "{extra}");
        }
        assert!(parse_policy_spec("chipgains:v1:early=1").is_err());
    }
}
