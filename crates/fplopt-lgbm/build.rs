//! Builds LightGBM's static library from the pinned submodule (`third_party/LightGBM`) with
//! CMake, in Release whatever the cargo profile, with OpenMP like the Python wheel (every fit
//! runs on one thread; OpenMP only keeps the wheel's code paths).
//!
//! LightGBM's CMakeLists writes its libraries into the source directory, so the source is
//! copied into OUT_DIR first and the submodule's checkout stays clean.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn copy_tree(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap();
    for entry in fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name();
        if name == ".git" {
            continue;
        }
        let target = to.join(&name);
        if entry.file_type().unwrap().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), target).unwrap();
        }
    }
}

/// The first file called `name` under `dir`.
fn find(dir: &Path, name: &str) -> Option<PathBuf> {
    for entry in fs::read_dir(dir).ok()? {
        let path = entry.ok()?.path();
        if path.is_dir() {
            if let Some(found) = find(&path, name) {
                return Some(found);
            }
        } else if path.file_name().is_some_and(|n| n == name) {
            return Some(path);
        }
    }
    None
}

fn main() {
    let submodule = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("..")
        .join("..")
        .join("third_party")
        .join("LightGBM");
    assert!(
        submodule.join("external_libs/nanoarrow/CMakeLists.txt").exists(),
        "third_party/LightGBM is incomplete: run `git submodule update --init --recursive`"
    );
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let source = out.join("LightGBM");
    let version = fs::read_to_string(submodule.join("VERSION.txt")).unwrap();
    if fs::read_to_string(source.join("VERSION.txt")).ok().as_deref() != Some(version.as_str()) {
        let _ = fs::remove_dir_all(&source);
        copy_tree(&submodule, &source);
    }
    let target = env::var("TARGET").unwrap();
    let crt_static = env::var("CARGO_CFG_TARGET_FEATURE")
        .unwrap_or_default()
        .split(',')
        .any(|f| f == "crt-static");
    // No cmake-crate flags: CMake's own Release flags, as the wheel's scikit-build-core build.
    let dst = cmake::Config::new(&source)
        .no_default_flags(true)
        .profile("Release")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        .define("BUILD_STATIC_LIB", "ON")
        .define("BUILD_CLI", "OFF")
        .define("INSTALL_HEADERS", "OFF")
        .define("USE_OPENMP", "ON")
        .define(
            "CMAKE_MSVC_RUNTIME_LIBRARY",
            if crt_static { "MultiThreaded" } else { "MultiThreadedDLL" },
        )
        .build();
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    let msvc = target.contains("msvc");
    println!("cargo:rustc-link-lib=static={}", if msvc { "lib_lightgbm" } else { "_lightgbm" });
    // The Arrow entry points of the C API need nanoarrow, which CMake builds but doesn't install.
    let nanoarrow = if msvc { "nanoarrow_static.lib" } else { "libnanoarrow_static.a" };
    let nanoarrow = find(&dst.join("build"), nanoarrow).expect("nanoarrow_static not built");
    println!("cargo:rustc-link-search=native={}", nanoarrow.parent().unwrap().display());
    println!("cargo:rustc-link-lib=static=nanoarrow_static");
    // OpenMP: MSVC's objects name vcomp themselves; gcc needs libgomp, clang libomp.
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-lib=omp");
    } else if target.contains("linux") || target.contains("windows-gnu") {
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=gomp");
    }
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../../third_party/LightGBM/VERSION.txt");
}
