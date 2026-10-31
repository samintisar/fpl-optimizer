//! SIGTERM (systemd stop, or `TimeoutStartSec`) as a flag the long loops check, so a
//! stopped run still writes its manifest: Python turns SIGTERM into `SystemExit` for the
//! same reason. The CLI registers the signal handler; jobs call [`check`].

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock};

static FLAG: LazyLock<Arc<AtomicBool>> = LazyLock::new(|| Arc::new(AtomicBool::new(false)));

/// The flag a signal handler sets.
pub fn flag() -> Arc<AtomicBool> {
    Arc::clone(&FLAG)
}

pub fn requested() -> bool {
    FLAG.load(Ordering::Relaxed)
}

/// Err(Terminated) once a stop was requested.
pub fn check() -> Result<(), Terminated> {
    if requested() { Err(Terminated) } else { Ok(()) }
}

/// The run was stopped by SIGTERM: exit 143, no alert (as for Python's `SystemExit`).
#[derive(Debug, Clone, Copy)]
pub struct Terminated;

impl fmt::Display for Terminated {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("143")
    }
}

impl std::error::Error for Terminated {}
