//! Builds the HiGHS static library from the pinned submodule (`third_party/HiGHS`) with
//! CMake, in Release whatever the cargo profile, as highspy's wheel is built.

use std::env;
use std::path::PathBuf;

fn main() {
    let source = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("..")
        .join("..")
        .join("third_party")
        .join("HiGHS");
    assert!(
        source.join("highs/interfaces/highs_c_api.h").exists(),
        "third_party/HiGHS is empty: run `git submodule update --init`"
    );
    let target = env::var("TARGET").unwrap();
    let crt_static = env::var("CARGO_CFG_TARGET_FEATURE")
        .unwrap_or_default()
        .split(',')
        .any(|f| f == "crt-static");
    // No cmake-crate flags: CMake's own Release flags (/O2 on MSVC, -O3 on gcc), as the
    // highspy wheel's scikit-build-core build uses.
    let dst = cmake::Config::new(&source)
        .no_default_flags(true)
        .profile("Release")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        .define("BUILD_CXX_EXE", "OFF")
        .define("BUILD_EXAMPLES", "OFF")
        .define("BUILD_TESTING", "OFF")
        .define("FAST_BUILD", "ON")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("ZLIB", "OFF")
        .define("CMAKE_INTERPROCEDURAL_OPTIMIZATION", "FALSE")
        .define(
            "CMAKE_MSVC_RUNTIME_LIBRARY",
            if crt_static { "MultiThreaded" } else { "MultiThreadedDLL" },
        )
        .build();
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-search=native={}/lib64", dst.display());
    println!("cargo:rustc-link-lib=static=highs");
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=c++");
    } else if target.contains("linux") || target.contains("windows-gnu") {
        println!("cargo:rustc-link-lib=stdc++");
    }
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../../third_party/HiGHS/Version.txt");
}
