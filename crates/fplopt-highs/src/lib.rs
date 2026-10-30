//! HiGHS through its C API (the static library build.rs compiles from the pinned
//! `third_party/HiGHS`, v1.15.1 like highspy). Only the calls the optimizer makes are
//! declared: create, options, `addCols`/`addRows`/integrality in bulk (the hand-over
//! Python's `_BulkHiGHS` uses), run, status, solution and info values.

use std::ffi::{CStr, CString, c_char, c_void};

use anyhow::{Result, bail};

type HighsInt = i32; // HIGHSINT64 is OFF in the build

const K_OK: HighsInt = 0;
const K_ERROR: HighsInt = -1;
const K_INTEGER: HighsInt = 1;

unsafe extern "C" {
    fn Highs_create() -> *mut c_void;
    fn Highs_destroy(highs: *mut c_void);
    fn Highs_version() -> *const c_char;
    fn Highs_githash() -> *const c_char;
    fn Highs_setBoolOptionValue(
        highs: *mut c_void,
        option: *const c_char,
        value: HighsInt,
    ) -> HighsInt;
    fn Highs_setIntOptionValue(
        highs: *mut c_void,
        option: *const c_char,
        value: HighsInt,
    ) -> HighsInt;
    fn Highs_setDoubleOptionValue(
        highs: *mut c_void,
        option: *const c_char,
        value: f64,
    ) -> HighsInt;
    fn Highs_addCols(
        highs: *mut c_void,
        num_new_col: HighsInt,
        costs: *const f64,
        lower: *const f64,
        upper: *const f64,
        num_new_nz: HighsInt,
        starts: *const HighsInt,
        index: *const HighsInt,
        value: *const f64,
    ) -> HighsInt;
    fn Highs_addRows(
        highs: *mut c_void,
        num_new_row: HighsInt,
        lower: *const f64,
        upper: *const f64,
        num_new_nz: HighsInt,
        starts: *const HighsInt,
        index: *const HighsInt,
        value: *const f64,
    ) -> HighsInt;
    fn Highs_changeColsIntegralityBySet(
        highs: *mut c_void,
        num_set_entries: HighsInt,
        set: *const HighsInt,
        integrality: *const HighsInt,
    ) -> HighsInt;
    fn Highs_run(highs: *mut c_void) -> HighsInt;
    fn Highs_getModelStatus(highs: *const c_void) -> HighsInt;
    fn Highs_getObjectiveValue(highs: *const c_void) -> f64;
    fn Highs_getNumCol(highs: *const c_void) -> HighsInt;
    fn Highs_getNumRow(highs: *const c_void) -> HighsInt;
    fn Highs_getSolution(
        highs: *const c_void,
        col_value: *mut f64,
        col_dual: *mut f64,
        row_value: *mut f64,
        row_dual: *mut f64,
    ) -> HighsInt;
    fn Highs_getDoubleInfoValue(
        highs: *const c_void,
        info: *const c_char,
        value: *mut f64,
    ) -> HighsInt;
    fn Highs_getInt64InfoValue(
        highs: *const c_void,
        info: *const c_char,
        value: *mut i64,
    ) -> HighsInt;
    fn Highs_writeModel(highs: *mut c_void, filename: *const c_char) -> HighsInt;
}

/// `kHighsModelStatus*` values of the C API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelStatus {
    Notset,
    LoadError,
    ModelError,
    PresolveError,
    SolveError,
    PostsolveError,
    ModelEmpty,
    Optimal,
    Infeasible,
    UnboundedOrInfeasible,
    Unbounded,
    ObjectiveBound,
    ObjectiveTarget,
    TimeLimit,
    IterationLimit,
    Unknown,
    SolutionLimit,
    Interrupt,
    /// A code this binding does not know (a newer HiGHS).
    Other(i32),
}

impl ModelStatus {
    fn from_code(code: HighsInt) -> ModelStatus {
        use ModelStatus::*;
        match code {
            0 => Notset,
            1 => LoadError,
            2 => ModelError,
            3 => PresolveError,
            4 => SolveError,
            5 => PostsolveError,
            6 => ModelEmpty,
            7 => Optimal,
            8 => Infeasible,
            9 => UnboundedOrInfeasible,
            10 => Unbounded,
            11 => ObjectiveBound,
            12 => ObjectiveTarget,
            13 => TimeLimit,
            14 => IterationLimit,
            15 => Unknown,
            16 => SolutionLimit,
            17 => Interrupt,
            other => Other(other),
        }
    }
}

/// "v1.15.1" and the githash of the linked HiGHS.
pub fn version() -> (String, String) {
    // SAFETY: both return static NUL-terminated strings.
    unsafe {
        (
            CStr::from_ptr(Highs_version()).to_string_lossy().into_owned(),
            CStr::from_ptr(Highs_githash()).to_string_lossy().into_owned(),
        )
    }
}

fn c_name(name: &str) -> Result<CString> {
    Ok(CString::new(name)?)
}

fn len(n: usize) -> Result<HighsInt> {
    match HighsInt::try_from(n) {
        Ok(v) => Ok(v),
        Err(_) => bail!("HiGHS: {n} entries exceed the 32-bit index"),
    }
}

/// One HiGHS instance (`Highs_create` .. `Highs_destroy`).
pub struct Highs {
    ptr: *mut c_void,
}

impl Default for Highs {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Highs {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from Highs_create and is destroyed once.
        unsafe { Highs_destroy(self.ptr) }
    }
}

impl Highs {
    pub fn new() -> Highs {
        // SAFETY: no arguments; a null return would be an allocation failure.
        let ptr = unsafe { Highs_create() };
        assert!(!ptr.is_null(), "Highs_create returned null");
        Highs { ptr }
    }

    fn check(&self, status: HighsInt, what: &str) -> Result<()> {
        if status == K_ERROR {
            bail!("HiGHS: {what} failed");
        }
        Ok(())
    }

    pub fn set_bool(&mut self, option: &str, value: bool) -> Result<()> {
        let name = c_name(option)?;
        // SAFETY: valid instance and NUL-terminated name.
        let status =
            unsafe { Highs_setBoolOptionValue(self.ptr, name.as_ptr(), HighsInt::from(value)) };
        self.check(status, &format!("option {option}"))
    }

    pub fn set_int(&mut self, option: &str, value: i64) -> Result<()> {
        let name = c_name(option)?;
        let value = len(usize::try_from(value)?)?;
        // SAFETY: valid instance and NUL-terminated name.
        let status = unsafe { Highs_setIntOptionValue(self.ptr, name.as_ptr(), value) };
        self.check(status, &format!("option {option}"))
    }

    pub fn set_double(&mut self, option: &str, value: f64) -> Result<()> {
        let name = c_name(option)?;
        // SAFETY: valid instance and NUL-terminated name.
        let status = unsafe { Highs_setDoubleOptionValue(self.ptr, name.as_ptr(), value) };
        self.check(status, &format!("option {option}"))
    }

    /// Columns without matrix entries (the rows bring them).
    pub fn add_cols(&mut self, cost: &[f64], lower: &[f64], upper: &[f64]) -> Result<()> {
        let n = cost.len();
        assert!(lower.len() == n && upper.len() == n);
        let starts = vec![0 as HighsInt; n];
        // SAFETY: arrays of length n (starts too); no nonzeros, so index/value unused.
        let status = unsafe {
            Highs_addCols(
                self.ptr,
                len(n)?,
                cost.as_ptr(),
                lower.as_ptr(),
                upper.as_ptr(),
                0,
                starts.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        self.check(status, "addCols")
    }

    /// Rows in compressed row form (`starts[r]` the first entry of row r).
    pub fn add_rows(
        &mut self,
        lower: &[f64],
        upper: &[f64],
        starts: &[i32],
        index: &[i32],
        value: &[f64],
    ) -> Result<()> {
        let n = lower.len();
        assert!(upper.len() == n && starts.len() == n && index.len() == value.len());
        // SAFETY: arrays of the declared lengths, indices validated by HiGHS.
        let status = unsafe {
            Highs_addRows(
                self.ptr,
                len(n)?,
                lower.as_ptr(),
                upper.as_ptr(),
                len(index.len())?,
                starts.as_ptr(),
                index.as_ptr(),
                value.as_ptr(),
            )
        };
        self.check(status, "addRows")
    }

    /// Marks `columns` integer.
    pub fn set_integer(&mut self, columns: &[i32]) -> Result<()> {
        if columns.is_empty() {
            return Ok(());
        }
        let kinds = vec![K_INTEGER; columns.len()];
        // SAFETY: both arrays have columns.len() entries.
        let status = unsafe {
            Highs_changeColsIntegralityBySet(
                self.ptr,
                len(columns.len())?,
                columns.as_ptr(),
                kinds.as_ptr(),
            )
        };
        self.check(status, "changeColsIntegrality")
    }

    /// `Highs_run`; its return status is ignored like PuLP does (the model status says what
    /// happened). True when it returned kOk.
    pub fn run(&mut self) -> bool {
        // SAFETY: valid instance.
        unsafe { Highs_run(self.ptr) == K_OK }
    }

    pub fn model_status(&self) -> ModelStatus {
        // SAFETY: valid instance.
        ModelStatus::from_code(unsafe { Highs_getModelStatus(self.ptr) })
    }

    pub fn objective_value(&self) -> f64 {
        // SAFETY: valid instance.
        unsafe { Highs_getObjectiveValue(self.ptr) }
    }

    /// The solution's column values.
    pub fn col_values(&self) -> Vec<f64> {
        // SAFETY: valid instance; buffers sized to the model.
        unsafe {
            let n = Highs_getNumCol(self.ptr).max(0) as usize;
            let m = Highs_getNumRow(self.ptr).max(0) as usize;
            let (mut cv, mut cd) = (vec![0.0; n], vec![0.0; n]);
            let (mut rv, mut rd) = (vec![0.0; m], vec![0.0; m]);
            Highs_getSolution(
                self.ptr,
                cv.as_mut_ptr(),
                cd.as_mut_ptr(),
                rv.as_mut_ptr(),
                rd.as_mut_ptr(),
            );
            cv
        }
    }

    pub fn double_info(&self, info: &str) -> Result<f64> {
        let name = c_name(info)?;
        let mut value = 0.0;
        // SAFETY: valid instance, name and out pointer.
        let status = unsafe { Highs_getDoubleInfoValue(self.ptr, name.as_ptr(), &mut value) };
        self.check(status, &format!("info {info}"))?;
        Ok(value)
    }

    pub fn int64_info(&self, info: &str) -> Result<i64> {
        let name = c_name(info)?;
        let mut value = 0i64;
        // SAFETY: valid instance, name and out pointer.
        let status = unsafe { Highs_getInt64InfoValue(self.ptr, name.as_ptr(), &mut value) };
        self.check(status, &format!("info {info}"))?;
        Ok(value)
    }

    /// HiGHS's own model writer (`.mps`/`.lp` by extension), for debugging model parity.
    pub fn write_model(&mut self, path: &str) -> Result<()> {
        let name = c_name(path)?;
        // SAFETY: valid instance and NUL-terminated path.
        let status = unsafe { Highs_writeModel(self.ptr, name.as_ptr()) };
        self.check(status, "writeModel")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn links_the_pinned_version() {
        let (version, githash) = version();
        assert_eq!(version, "1.15.1");
        assert!(githash.starts_with("04024d7"), "{githash}");
    }

    #[test]
    fn solves_a_small_mip() {
        // max x + 2y st x + y <= 1.5, x,y binary -> y = 1 (as a minimisation of -x - 2y).
        let mut h = Highs::new();
        h.set_bool("output_flag", false).unwrap();
        h.add_cols(&[-1.0, -2.0], &[0.0, 0.0], &[1.0, 1.0]).unwrap();
        h.add_rows(&[f64::NEG_INFINITY], &[1.5], &[0], &[0, 1], &[1.0, 1.0]).unwrap();
        h.set_integer(&[0, 1]).unwrap();
        h.run();
        assert_eq!(h.model_status(), ModelStatus::Optimal);
        assert_eq!(h.col_values(), vec![0.0, 1.0]);
        assert_eq!(h.objective_value(), -2.0);
        assert_eq!(h.int64_info("mip_node_count").unwrap(), 0);
        let _ = K_OK;
    }
}
