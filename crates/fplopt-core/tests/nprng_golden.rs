//! `fplopt_core::nprng` against numpy's draws (`tests/fixtures/rust_golden/nprng.json.gz`,
//! from `nprng_golden` in tests/test_rust_golden.py): every op, in order, on one generator.

use std::io::Read;

use fplopt_core::nprng::Generator;
use serde_json::{Value, json};

#[test]
fn draws_match_numpy() {
    let path =
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures/rust_golden/nprng.json.gz");
    let mut text = String::new();
    flate2::read::GzDecoder::new(std::fs::File::open(path).unwrap())
        .read_to_string(&mut text)
        .unwrap();
    let golden: Value = serde_json::from_str(&text).unwrap();
    let ops = golden["ops"].as_array().unwrap();
    let int = |v: &Value| v.as_i64().unwrap();
    for case in golden["cases"].as_array().unwrap() {
        let seed: u128 = case["seed"].as_str().unwrap().parse().unwrap();
        let mut g = Generator::new(seed);
        for (i, (op, want)) in ops.iter().zip(case["draws"].as_array().unwrap()).enumerate() {
            let op = op.as_array().unwrap();
            let got = match op[0].as_str().unwrap() {
                "raw" => {
                    json!((0..int(&op[1])).map(|_| g.next_u64().to_string()).collect::<Vec<_>>())
                }
                "integers" => {
                    json!(g.integers(int(&op[1]), int(&op[2]), int(&op[3]) as usize))
                }
                "random" => json!((0..int(&op[1])).map(|_| g.random()).collect::<Vec<_>>()),
                "permutation" => json!(g.permutation(int(&op[1]) as usize)),
                "choice" => json!(
                    g.choice_indices(int(&op[1]) as usize, int(&op[2]) as usize)
                        .into_iter()
                        .map(|k| 10 * k)
                        .collect::<Vec<_>>()
                ),
                "choice_p" => {
                    let w: Vec<f64> =
                        op[1].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
                    let total: f64 = w.iter().sum();
                    let p: Vec<f64> = w.iter().map(|x| x / total).collect();
                    json!(g.choice_weighted(&p))
                }
                other => panic!("unknown op {other}"),
            };
            assert_eq!(&got, want, "seed {seed}, op {i} {op:?}");
        }
    }
}
