//! `models calibrate fit` on a synthetic two-season walk, and `walk`'s refusals. (A walk
//! itself needs every table v1 fits on: it is checked on real data by the Task 12 gate.)

use std::path::{Path, PathBuf};
use std::process::Command;

use fplopt_build::table::{Column, Table, write_parquet};
use fplopt_models::assemble::V1Params;

fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_fplopt-rs"))
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("fplopt-calibrate-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A season of walk rows: horizons 0..=5, P(start) on a grid, a start when the
/// deterministic "draw" is below p (h0 rows get p 0.99 and never start, so a map fitted on
/// them would not be monotone up to 0.99).
fn write_season(dir: &Path, season: i64, n: usize) {
    let (mut horizon, mut p, mut r) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        let h = (i % 6) as i64;
        let pi = 0.01 + 0.98 * ((i * 7919) % n) as f64 / n as f64;
        let draw = ((i * 104_729) % 1000) as f64 / 1000.0;
        horizon.push(h);
        if h == 0 {
            p.push(0.99);
            r.push(0.0);
        } else {
            p.push(if i % 50 == 0 { 0.0 } else { pi });
            r.push(if draw < pi { 1.0 } else { 0.0 });
        }
    }
    let t = Table::new()
        .with("season", Column::I64(vec![season; n]))
        .with("horizon", Column::I64(horizon))
        .with("p_start", Column::F64(p))
        .with("r_start", Column::F64(r));
    write_parquet(&t, &dir.join(format!("{season}.parquet"))).unwrap();
}

#[test]
fn fit_writes_one_monotone_map_per_season() {
    let dir = scratch("fit");
    write_season(&dir, 2016, 3000);
    write_season(&dir, 2017, 2400);
    std::fs::write(dir.join("params.txt"), V1Params::default().fingerprint()).unwrap();
    let out = binary()
        .args(["models", "calibrate", "fit", "--model", "v1", "--groups", "1+", "--walk"])
        .arg(&dir)
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let table: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("calibration-table.json")).unwrap())
            .unwrap();
    assert_eq!(table["params"], V1Params::default().fingerprint());
    let entries = table["table"].as_array().unwrap();
    // Python's entries: S = 2017 ..= 2025, each fitted on the walk seasons before S.
    let seasons: Vec<i64> = entries.iter().map(|e| e[0].as_i64().unwrap()).collect();
    assert_eq!(seasons, (2017..=2025).collect::<Vec<_>>());
    assert_eq!(entries[0][1]["fitted_on"], serde_json::json!([2016]));
    assert_eq!(entries[1][1]["fitted_on"], serde_json::json!([2016, 2017]));
    for e in entries {
        let maps = e[1]["p_start"].as_array().unwrap();
        assert_eq!(maps.len(), 1);
        assert_eq!(maps[0][0], 1);
        let x: Vec<f64> =
            maps[0][1].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let y: Vec<f64> =
            maps[0][2].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        assert!(x.len() > 1 && x.len() == y.len());
        assert!(x.iter().all(|&v| v > 0.0 && v < 0.99));
        assert!(x.windows(2).all(|w| w[0] < w[1]));
        assert!(y.windows(2).all(|w| w[0] <= w[1]));
    }
    // Compared with itself, the gate passes.
    let again = binary()
        .args(["models", "calibrate", "fit", "--model", "v1", "--walk"])
        .arg(&dir)
        .arg("--compare")
        .arg(dir.join("calibration-table.json"))
        .output()
        .unwrap();
    assert!(again.status.success(), "{}", String::from_utf8_lossy(&again.stderr));
    assert!(String::from_utf8_lossy(&again.stdout).contains("compare: pass"));
    // A walk of other params is refused.
    let refused = binary()
        .args(["models", "calibrate", "fit", "--model", "v1-2026-10", "--walk"])
        .arg(&dir)
        .output()
        .unwrap();
    assert!(!refused.status.success());
    // --write is for v1 only (v1-2026-10's table is Python's export).
    let refused = binary()
        .args(["models", "calibrate", "fit", "--model", "v1-2026-10", "--write", "--walk"])
        .arg(&dir)
        .output()
        .unwrap();
    assert!(!refused.status.success());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn walk_refuses_the_holdout_and_unknown_models() {
    let dir = scratch("walk");
    for args in
        [["--model", "v1", "--seasons", "2024-2025"], ["--model", "v9", "--seasons", "2020"]]
    {
        let out = binary()
            .args(["models", "calibrate", "walk"])
            .args(args)
            .arg("--out")
            .arg(&dir)
            .output()
            .unwrap();
        assert_eq!(out.status.code(), Some(2), "{}", String::from_utf8_lossy(&out.stderr));
    }
    assert!(!dir.join("params.txt").exists());
    let _ = std::fs::remove_dir_all(&dir);
}
