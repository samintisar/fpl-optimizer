//! `tests/fixtures/rust_golden/lgbm.json.gz` (written by Python's tests/test_rust_golden.py):
//! boosters the LightGBM wheel trained through `fplopt.models.gbm`. The Rust build of the
//! same LightGBM, given the same rows and parameter strings, must write the same model text
//! byte for byte and predict the same values bit for bit, on every platform CI runs.

use std::io::Read;
use std::path::Path;

use serde_json::Value;

fn golden() -> Value {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/rust_golden/lgbm.json.gz");
    let mut text = String::new();
    flate2::read::GzDecoder::new(std::fs::File::open(path).unwrap())
        .read_to_string(&mut text)
        .unwrap();
    serde_json::from_str(&text).unwrap()
}

fn floats(value: &Value) -> Vec<f64> {
    value.as_array().unwrap().iter().map(|v| v.as_f64().unwrap_or(f64::NAN)).collect()
}

#[test]
fn boosters_match_the_wheel() {
    for case in golden()["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let features: Vec<&str> =
            case["features"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
        let x: Vec<f64> = case["x"].as_array().unwrap().iter().flat_map(floats).collect();
        let label = floats(&case["label"]);
        let weight = (!case["weight"].is_null()).then(|| floats(&case["weight"]));
        let booster = fplopt_lgbm::Booster::train(&fplopt_lgbm::Train {
            x: &x,
            ncol: features.len(),
            label: &label,
            weight: weight.as_deref(),
            feature_names: &features,
            dataset_params: case["dataset_params"].as_str().unwrap(),
            booster_params: case["booster_params"].as_str().unwrap(),
            rounds: case["rounds"].as_u64().unwrap() as usize,
        })
        .unwrap();
        let text = booster.model_string().unwrap();
        let want = case["model"].as_str().unwrap();
        if text != want {
            let line = text.lines().zip(want.lines()).position(|(a, b)| a != b);
            panic!("{name}: model text differs (first differing line {line:?})");
        }
        let got = booster.predict(&x).unwrap();
        let want = floats(&case["predictions"]);
        assert_eq!(got.len(), want.len(), "{name}");
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "{name} row {i}: {g} vs {w}");
        }
    }
}

#[test]
fn predict_checks_the_width() {
    let case = &golden()["cases"][0];
    let booster = fplopt_lgbm::Booster::from_string(case["model"].as_str().unwrap(), 4).unwrap();
    assert!(booster.predict(&[1.0, 2.0, 3.0]).is_err());
    assert!(booster.predict(&[]).unwrap().is_empty());
    assert_eq!(booster.predict(&[0.1, 40.0, f64::NAN, 2.0]).unwrap().len(), 1);
}
