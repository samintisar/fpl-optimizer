//! `fplopt_core::seasons` against Python's outputs (tests/fixtures/rust_golden/seasons.json,
//! written and kept current by tests/test_rust_golden.py).

use chrono::DateTime;
use fplopt_core::seasons::*;
use serde_json::Value;

fn golden() -> Value {
    let path =
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures/rust_golden/seasons.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn labels_and_codes() {
    let g = golden();
    for (year, label) in g["season_label"].as_object().unwrap() {
        let year: i32 = year.parse().unwrap();
        assert_eq!(season_label(year), label.as_str().unwrap(), "{year}");
        assert_eq!(football_data_code(year), g["football_data_code"][year.to_string()], "{year}");
    }
}

#[test]
fn parse() {
    for (label, expected) in golden()["parse_season_label"].as_object().unwrap() {
        let got = parse_season_label(label).ok();
        assert_eq!(got.map(i64::from), expected.as_i64(), "{label:?}");
    }
}

#[test]
fn start_years() {
    for case in golden()["season_start_year"].as_array().unwrap() {
        let when = DateTime::parse_from_rfc3339(case[0].as_str().unwrap()).unwrap();
        assert_eq!(i64::from(season_start_year(&when)), case[1].as_i64().unwrap(), "{when}");
    }
}

#[test]
fn bootstraps() {
    for case in golden()["bootstrap_season"].as_array().unwrap() {
        assert_eq!(i64::from(bootstrap_season(&case[0]).unwrap()), case[1].as_i64().unwrap());
    }
}

#[test]
fn splits() {
    let g = golden();
    let ints = |key: &str| -> Vec<i32> {
        g[key].as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as i32).collect()
    };
    assert_eq!(DEVELOP_SEASONS.collect::<Vec<_>>(), ints("develop"));
    assert_eq!(VALIDATE_SEASONS.collect::<Vec<_>>(), ints("validate"));
    assert_eq!(HOLDOUT_SEASONS.to_vec(), ints("holdout"));
}
