//! LightGBM through its C API (the static library build.rs compiles from the pinned
//! `third_party/LightGBM`, v4.7.0 like the Python wheel). Only the calls `lightgbm.train` and
//! `Booster.predict` make for a dense float64 matrix are declared, in the order Python makes
//! them, so the same rows and parameter strings give a byte-identical booster:
//! 1. `LGBM_DatasetCreateFromMat` (row-major float64, the dataset's parameter string), the
//!    label and the weight as float32 (Python's `set_label` / `set_weight` downcast; an
//!    all-ones weight is dropped), the feature names;
//! 2. `LGBM_BoosterCreate` (the booster's parameter string), `LGBM_BoosterUpdateOneIter` once
//!    per round (an `is_finished` doesn't stop the loop, as in `lightgbm.train`);
//! 3. the model saved to text and loaded back (`lightgbm.train` without
//!    `keep_training_booster` returns that copy), which predicts.

use std::ffi::{CStr, CString, c_char, c_int, c_void};

use anyhow::{Result, bail};

const C_API_DTYPE_FLOAT32: c_int = 0;
const C_API_DTYPE_FLOAT64: c_int = 1;
const C_API_PREDICT_NORMAL: c_int = 0;
const C_API_FEATURE_IMPORTANCE_SPLIT: c_int = 0;

unsafe extern "C" {
    fn LGBM_GetLastError() -> *const c_char;
    fn LGBM_DatasetCreateFromMat(
        data: *const c_void,
        data_type: c_int,
        nrow: i32,
        ncol: i32,
        is_row_major: c_int,
        parameters: *const c_char,
        reference: *const c_void,
        out: *mut *mut c_void,
    ) -> c_int;
    fn LGBM_DatasetSetField(
        handle: *mut c_void,
        field_name: *const c_char,
        field_data: *const c_void,
        num_element: c_int,
        data_type: c_int,
    ) -> c_int;
    fn LGBM_DatasetSetFeatureNames(
        handle: *mut c_void,
        feature_names: *const *const c_char,
        num_feature_names: c_int,
    ) -> c_int;
    fn LGBM_DatasetFree(handle: *mut c_void) -> c_int;
    fn LGBM_BoosterCreate(
        train_data: *const c_void,
        parameters: *const c_char,
        out: *mut *mut c_void,
    ) -> c_int;
    fn LGBM_BoosterUpdateOneIter(handle: *mut c_void, is_finished: *mut c_int) -> c_int;
    fn LGBM_BoosterSaveModelToString(
        handle: *mut c_void,
        start_iteration: c_int,
        num_iteration: c_int,
        feature_importance_type: c_int,
        buffer_len: i64,
        out_len: *mut i64,
        out_str: *mut c_char,
    ) -> c_int;
    fn LGBM_BoosterLoadModelFromString(
        model_str: *const c_char,
        out_num_iterations: *mut c_int,
        out: *mut *mut c_void,
    ) -> c_int;
    fn LGBM_BoosterPredictForMat(
        handle: *mut c_void,
        data: *const c_void,
        data_type: c_int,
        nrow: i32,
        ncol: i32,
        is_row_major: c_int,
        predict_type: c_int,
        start_iteration: c_int,
        num_iteration: c_int,
        parameter: *const c_char,
        out_len: *mut i64,
        out_result: *mut f64,
    ) -> c_int;
    fn LGBM_BoosterFree(handle: *mut c_void) -> c_int;
}

fn check(code: c_int, call: &str) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    // SAFETY: LightGBM returns a pointer to its thread-local, NUL-terminated message.
    let message = unsafe { CStr::from_ptr(LGBM_GetLastError()) }.to_string_lossy().into_owned();
    bail!("LightGBM {call}: {message}")
}

fn c_string(text: &str) -> Result<CString> {
    Ok(CString::new(text)?)
}

/// A training set: a row-major float64 matrix (NaN = missing), its label and weight.
struct Dataset(*mut c_void);

impl Drop for Dataset {
    fn drop(&mut self) {
        // SAFETY: the handle came from LGBM_DatasetCreateFromMat and is freed once.
        unsafe { LGBM_DatasetFree(self.0) };
    }
}

impl Dataset {
    fn new(
        x: &[f64],
        ncol: usize,
        label: &[f64],
        weight: Option<&[f64]>,
        feature_names: &[&str],
        params: &str,
    ) -> Result<Dataset> {
        let nrow = label.len();
        if x.len() != nrow * ncol || feature_names.len() != ncol {
            bail!("LightGBM dataset: {} values for {nrow} rows × {ncol} features", x.len());
        }
        if weight.is_some_and(|w| w.len() != nrow) {
            bail!("LightGBM dataset: the weight has a different length from the label");
        }
        let params = c_string(params)?;
        let mut handle = std::ptr::null_mut();
        // SAFETY: `x` holds nrow × ncol doubles, row-major; the strings outlive the call.
        check(
            unsafe {
                LGBM_DatasetCreateFromMat(
                    x.as_ptr().cast(),
                    C_API_DTYPE_FLOAT64,
                    i32::try_from(nrow)?,
                    i32::try_from(ncol)?,
                    1,
                    params.as_ptr(),
                    std::ptr::null(),
                    &mut handle,
                )
            },
            "DatasetCreateFromMat",
        )?;
        let dataset = Dataset(handle);
        dataset.set_field("label", label)?;
        if let Some(weight) = weight.filter(|w| !w.iter().all(|&v| v == 1.0)) {
            dataset.set_field("weight", weight)?;
        }
        let names: Vec<CString> =
            feature_names.iter().map(|n| c_string(n)).collect::<Result<_>>()?;
        let pointers: Vec<*const c_char> = names.iter().map(|n| n.as_ptr()).collect();
        // SAFETY: `pointers` holds ncol NUL-terminated names that outlive the call.
        check(
            unsafe {
                LGBM_DatasetSetFeatureNames(
                    dataset.0,
                    pointers.as_ptr(),
                    c_int::try_from(pointers.len())?,
                )
            },
            "DatasetSetFeatureNames",
        )?;
        Ok(dataset)
    }

    /// A float32 field (Python downcasts label and weight).
    fn set_field(&self, name: &str, values: &[f64]) -> Result<()> {
        let values: Vec<f32> = values.iter().map(|&v| v as f32).collect();
        let name = c_string(name)?;
        // SAFETY: `values` holds len floats; the handle is live.
        check(
            unsafe {
                LGBM_DatasetSetField(
                    self.0,
                    name.as_ptr(),
                    values.as_ptr().cast(),
                    c_int::try_from(values.len())?,
                    C_API_DTYPE_FLOAT32,
                )
            },
            "DatasetSetField",
        )
    }
}

/// A fitted booster (the copy loaded back from its text, as `lightgbm.train` returns).
pub struct Booster {
    handle: *mut c_void,
    num_features: usize,
}

// SAFETY: a booster handle isn't tied to the thread that made it, and LightGBM's Booster
// guards its state with its own lock, so `predict(&self)` may run from several threads.
unsafe impl Send for Booster {}
unsafe impl Sync for Booster {}

impl Drop for Booster {
    fn drop(&mut self) {
        // SAFETY: the handle came from LightGBM and is freed once.
        unsafe { LGBM_BoosterFree(self.handle) };
    }
}

/// What `lightgbm.train(params, Dataset(x, label, weight, feature_name, params=dataset
/// params), num_boost_round=rounds)` does, with the parameter strings Python passes
/// (`dataset_params` to the dataset, `booster_params` to the booster).
pub struct Train<'a> {
    pub x: &'a [f64],
    pub ncol: usize,
    pub label: &'a [f64],
    pub weight: Option<&'a [f64]>,
    pub feature_names: &'a [&'a str],
    pub dataset_params: &'a str,
    pub booster_params: &'a str,
    pub rounds: usize,
}

impl Booster {
    pub fn train(spec: &Train) -> Result<Booster> {
        let dataset = Dataset::new(
            spec.x,
            spec.ncol,
            spec.label,
            spec.weight,
            spec.feature_names,
            spec.dataset_params,
        )?;
        let params = c_string(spec.booster_params)?;
        let mut handle = std::ptr::null_mut();
        // SAFETY: the dataset is live; the string outlives the call.
        check(
            unsafe { LGBM_BoosterCreate(dataset.0, params.as_ptr(), &mut handle) },
            "BoosterCreate",
        )?;
        let training = Booster { handle, num_features: spec.ncol };
        for _ in 0..spec.rounds {
            let mut finished: c_int = 0;
            // SAFETY: the booster is live.
            check(
                unsafe { LGBM_BoosterUpdateOneIter(training.handle, &mut finished) },
                "BoosterUpdateOneIter",
            )?;
        }
        let text = training.model_string()?;
        drop(training);
        drop(dataset);
        Booster::from_string(&text, spec.ncol)
    }

    /// A booster from its model text (`num_features`: the matrix width `predict` takes).
    pub fn from_string(text: &str, num_features: usize) -> Result<Booster> {
        let text = c_string(text)?;
        let (mut iterations, mut handle): (c_int, *mut c_void) = (0, std::ptr::null_mut());
        // SAFETY: the string outlives the call.
        check(
            unsafe { LGBM_BoosterLoadModelFromString(text.as_ptr(), &mut iterations, &mut handle) },
            "BoosterLoadModelFromString",
        )?;
        Ok(Booster { handle, num_features })
    }

    /// The model text (`Booster.model_to_string()`: every iteration, split importance).
    pub fn model_string(&self) -> Result<String> {
        let mut len: i64 = 0;
        // SAFETY: a zero-length buffer only asks for the length.
        check(
            unsafe {
                LGBM_BoosterSaveModelToString(
                    self.handle,
                    0,
                    -1,
                    C_API_FEATURE_IMPORTANCE_SPLIT,
                    0,
                    &mut len,
                    std::ptr::null_mut(),
                )
            },
            "BoosterSaveModelToString",
        )?;
        let mut buffer = vec![0u8; usize::try_from(len)?];
        // SAFETY: `buffer` holds `len` bytes, the length LightGBM asked for.
        check(
            unsafe {
                LGBM_BoosterSaveModelToString(
                    self.handle,
                    0,
                    -1,
                    C_API_FEATURE_IMPORTANCE_SPLIT,
                    len,
                    &mut len,
                    buffer.as_mut_ptr().cast(),
                )
            },
            "BoosterSaveModelToString",
        )?;
        let text = CStr::from_bytes_until_nul(&buffer)?;
        Ok(text.to_str()?.to_string())
    }

    /// `Booster.predict(x, num_threads=1)` on a row-major float64 matrix (NaN = missing):
    /// one value per row (the probability for `binary`).
    pub fn predict(&self, x: &[f64]) -> Result<Vec<f64>> {
        let ncol = self.num_features;
        if ncol == 0 || !x.len().is_multiple_of(ncol) {
            bail!("LightGBM predict: {} values for {ncol} features", x.len());
        }
        let nrow = x.len() / ncol;
        let mut out = vec![0.0; nrow];
        if nrow == 0 {
            return Ok(out);
        }
        let params = c_string("num_threads=1")?;
        let mut len: i64 = 0;
        // SAFETY: `x` holds nrow × ncol doubles; `out` has room for one value per row.
        check(
            unsafe {
                LGBM_BoosterPredictForMat(
                    self.handle,
                    x.as_ptr().cast(),
                    C_API_DTYPE_FLOAT64,
                    i32::try_from(nrow)?,
                    i32::try_from(ncol)?,
                    1,
                    C_API_PREDICT_NORMAL,
                    0,
                    -1,
                    params.as_ptr(),
                    &mut len,
                    out.as_mut_ptr(),
                )
            },
            "BoosterPredictForMat",
        )?;
        if usize::try_from(len)? != nrow {
            bail!("LightGBM predict: {len} values for {nrow} rows");
        }
        Ok(out)
    }
}
