//! Strategy string parsing for Python bindings.
//!
//! Converts Python string arguments (e.g. `"ffd"`, `"OBFD"`, `"FirstFitDecreasing"`)
//! into `PackStrategy` enum values. All lookup is case-insensitive.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use seqpacker::PackStrategy;

/// Parse a strategy string into a PackStrategy enum.
///
/// Accepts short names (e.g. "ffd", "obfd") and full names (e.g. "FirstFitDecreasing").
/// Case-insensitive.
pub fn parse_strategy(name: &str) -> PyResult<PackStrategy> {
    // Try short name first (most common usage).
    if let Some(strategy) = PackStrategy::from_short_name(name) {
        return Ok(strategy);
    }

    // Try matching full name (case-insensitive).
    for &strategy in &seqpacker::strategy::ALL_STRATEGIES {
        if strategy.name().eq_ignore_ascii_case(name) {
            return Ok(strategy);
        }
    }

    // Build helpful error message showing both short and full names.
    let valid: Vec<String> = seqpacker::strategy::ALL_STRATEGIES
        .iter()
        .filter_map(|s| {
            s.short_name()
                .map(|short| format!("{short} ({name})", name = s.name()))
        })
        .collect();
    Err(PyValueError::new_err(format!(
        "unknown strategy '{name}'. Valid strategies: {}",
        valid.join(", ")
    )))
}

/// Return list of all available strategy (short_name, full_name) pairs.
pub fn available_strategies() -> Vec<(&'static str, &'static str)> {
    seqpacker::strategy::ALL_STRATEGIES
        .iter()
        .filter_map(|s| s.short_name().map(|short| (short, s.name())))
        .collect()
}
