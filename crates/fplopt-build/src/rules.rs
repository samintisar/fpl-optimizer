//! `rules export`: config/scoring/<season>.json from an archived bootstrap-static (PLAN §3
//! Rules config). Port of `fplopt.build.rules`; the file is written like Python's
//! `json.dumps(config, sort_keys=True, indent=2) + "\n"`.

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use chrono::{DateTime, TimeZone, Utc};
use fplopt_archive::raw_store::RawStore;
use fplopt_core::seasons::{bootstrap_season, season_label};
use serde_json::{Map, Value};

/// Archives searched in order: our own snapshots first, then the fplcache mirror.
const ARCHIVES: [(&str, &str, &str); 2] =
    [("fpl", "bootstrap-static", ".json.gz"), ("fplcache", "bootstrap-static", ".json.xz")];

/// Newest archived bootstrap describing season `start_year` (own archive first), skipping
/// snapshots from 1 Aug of start_year + 1 and stopping before 1 Jun of start_year.
pub fn find_bootstrap(store: &RawStore, start_year: i32) -> Result<(DateTime<Utc>, PathBuf)> {
    let too_new = Utc.with_ymd_and_hms(start_year + 1, 8, 1, 0, 0, 0).unwrap();
    let too_old = Utc.with_ymd_and_hms(start_year, 6, 1, 0, 0, 0).unwrap();
    for (source, endpoint, suffix) in ARCHIVES {
        for (taken_at, path) in store.entries(source, endpoint, suffix)?.into_iter().rev() {
            if taken_at >= too_new {
                continue;
            }
            if taken_at < too_old {
                break;
            }
            match bootstrap_season(&RawStore::read_json(&path)?) {
                Ok(season) if season == start_year => return Ok((taken_at, path)),
                Ok(_) => {}
                Err(_) => {
                    log::warn!("skipping {}: no events to date its season", path.display());
                }
            }
        }
    }
    bail!("no archived bootstrap-static for season {}", season_label(start_year))
}

/// True if an overrides entry sets anything: a non-null leaf (0 and false count).
fn has_value(value: &Value) -> bool {
    match value {
        Value::Object(map) => map.values().any(has_value),
        Value::Array(items) => items.iter().any(has_value),
        Value::Null => false,
        _ => true,
    }
}

/// The season's rules from a bootstrap payload (without the `source` provenance).
pub fn rules_config(bootstrap: &Value, start_year: i32) -> Result<Map<String, Value>> {
    for key in ["game_config", "chips"] {
        if bootstrap.get(key).is_none() {
            bail!("bootstrap has no {key} (pre-2025/26 snapshot?)");
        }
    }
    let game_config = &bootstrap["game_config"];
    let missing: Vec<&str> = ["rules", "scoring", "settings"]
        .into_iter()
        .filter(|k| game_config.get(k).is_none())
        .collect();
    if !missing.is_empty() {
        bail!("bootstrap game_config lacks {}", missing.join(", "));
    }
    let element_types: Vec<Value> = bootstrap["element_types"]
        .as_array()
        .map(|items| {
            items
                .iter()
                .map(|item| {
                    let mut item = item.as_object().cloned().unwrap_or_default();
                    item.remove("element_count");
                    Value::Object(item)
                })
                .collect()
        })
        .unwrap_or_default();
    let mut overrides = Map::new();
    for event in bootstrap["events"].as_array().into_iter().flatten() {
        let value = event.get("overrides").unwrap_or(&Value::Null);
        if has_value(value) {
            let id = match &event["id"] {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            overrides.insert(id, value.clone());
        }
    }
    let mut config = Map::new();
    config.insert("season".into(), Value::String(season_label(start_year)));
    config.insert("scoring".into(), game_config["scoring"].clone());
    config.insert("rules".into(), game_config["rules"].clone());
    config.insert("settings".into(), game_config["settings"].clone());
    config.insert("chips".into(), bootstrap["chips"].clone());
    config.insert("element_types".into(), Value::Array(element_types));
    config.insert("event_overrides".into(), Value::Object(overrides));
    Ok(config)
}

/// Write `out_dir/<season>.json` (sorted keys, indent 2, LF, trailing newline).
pub fn export_rules(store: &RawStore, start_year: i32, out_dir: &Path) -> Result<PathBuf> {
    let (taken_at, path) = find_bootstrap(store, start_year)?;
    let mut config = rules_config(&RawStore::read_json(&path)?, start_year)?;
    let relative = path.strip_prefix(&store.root).unwrap_or(&path);
    let relative: Vec<String> =
        relative.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
    let mut source = Map::new();
    source.insert("path".into(), Value::String(relative.join("/")));
    source.insert(
        "snapshot_at".into(),
        Value::String(taken_at.format("%Y-%m-%dT%H:%M:%SZ").to_string()),
    );
    config.insert("source".into(), Value::Object(source));
    std::fs::create_dir_all(out_dir)?;
    let out_path = out_dir.join(format!("{}.json", season_label(start_year)));
    let mut text = String::new();
    write_indented(&Value::Object(config), 0, &mut text);
    text.push('\n');
    std::fs::write(&out_path, text.as_bytes())?;
    log::info!("wrote {} from {}", out_path.display(), relative.join("/"));
    Ok(out_path)
}

/// `json.dumps(value, sort_keys=True, indent=2)` (ensure_ascii, Python float repr).
pub fn write_indented(value: &Value, depth: usize, out: &mut String) {
    let pad = |n: usize| "  ".repeat(n);
    match value {
        Value::Object(map) if !map.is_empty() => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push_str("{\n");
            for (i, key) in keys.iter().enumerate() {
                out.push_str(&pad(depth + 1));
                write_string(key, out);
                out.push_str(": ");
                write_indented(&map[*key], depth + 1, out);
                if i + 1 < keys.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            out.push_str(&pad(depth));
            out.push('}');
        }
        Value::Array(items) if !items.is_empty() => {
            out.push_str("[\n");
            for (i, item) in items.iter().enumerate() {
                out.push_str(&pad(depth + 1));
                write_indented(item, depth + 1, out);
                if i + 1 < items.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            out.push_str(&pad(depth));
            out.push(']');
        }
        Value::Object(_) => out.push_str("{}"),
        Value::Array(_) => out.push_str("[]"),
        Value::String(s) => write_string(s, out),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Null => out.push_str("null"),
        Value::Number(n) => match n.as_f64() {
            Some(f) if !n.is_i64() && !n.is_u64() => out.push_str(&py_float_repr(f)),
            _ => out.push_str(&n.to_string()),
        },
    }
}

fn write_string(s: &str, out: &mut String) {
    out.push_str(&fplopt_core::pyjson::dumps(s).expect("a string always serializes"));
}

/// Python's `repr(float)`: shortest round-trip digits, scientific below 1e-4 or from 1e16.
pub fn py_float_repr(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let sci = format!("{f:e}"); // shortest digits, e.g. "1.5e-5"
    let (mantissa, exp) = sci.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let negative = mantissa.starts_with('-');
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let sign = if negative { "-" } else { "" };
    if (-4..16).contains(&exp) {
        let point = exp + 1; // digits before the decimal point
        let body = if point <= 0 {
            format!("0.{}{digits}", "0".repeat((-point) as usize))
        } else if point as usize >= digits.len() {
            format!("{digits}{}.0", "0".repeat(point as usize - digits.len()))
        } else {
            format!("{}.{}", &digits[..point as usize], &digits[point as usize..])
        };
        format!("{sign}{body}")
    } else {
        let rest = if digits.len() > 1 { format!(".{}", &digits[1..]) } else { String::new() };
        let esign = if exp < 0 { '-' } else { '+' };
        format!("{sign}{}{rest}e{esign}{:02}", &digits[..1], exp.abs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn float_repr_matches_python() {
        let cases = [
            (0.5, "0.5"),
            (2.0, "2.0"),
            (1e-5, "1e-05"),
            (0.0001, "0.0001"),
            (1e16, "1e+16"),
            (123456789012345.6, "123456789012345.6"),
            (-1.25e-7, "-1.25e-07"),
            (0.1, "0.1"),
        ];
        for (f, text) in cases {
            assert_eq!(py_float_repr(f), text, "{f}");
        }
    }

    #[test]
    fn indented_sorted_output() {
        let mut out = String::new();
        write_indented(&json!({"b": [1, {}], "a": {"y": [], "x": 0.5}}), 0, &mut out);
        assert_eq!(
            out,
            "{\n  \"a\": {\n    \"x\": 0.5,\n    \"y\": []\n  },\n  \"b\": [\n    1,\n    {}\n  ]\n}"
        );
    }

    #[test]
    fn overrides_need_a_non_null_leaf() {
        assert!(!has_value(&json!({"rules": {}, "scoring": {}, "pick_multiplier": null})));
        assert!(has_value(&json!({"rules": {"x": 0}})));
        assert!(has_value(&json!([false])));
    }
}
