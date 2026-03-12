//! Python wrapper for `PackMetrics`.
//!
//! Exposes packing quality metrics as read-only Python properties.
//! The inner `PackMetrics` struct is owned; Python sees only getters.

use pyo3::prelude::*;
use seqpacker::PackMetrics;

/// Packing quality metrics (read-only).
///
/// Wraps the Rust `PackMetrics` struct. All fields are exposed as
/// Python `@property` getters. Computed methods like `padding_ratio()`
/// and `throughput()` are also exposed as properties for ergonomic access.
#[pyclass(name = "PackMetrics", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyMetrics {
    pub(crate) inner: PackMetrics,
}

#[pymethods]
impl PyMetrics {
    // ── Struct fields (direct access) ───────────────────────────

    #[getter]
    fn num_sequences(&self) -> usize {
        self.inner.num_sequences
    }

    #[getter]
    fn total_tokens(&self) -> usize {
        self.inner.total_tokens
    }

    #[getter]
    fn num_packs(&self) -> usize {
        self.inner.num_packs
    }

    #[getter]
    fn padding_tokens(&self) -> usize {
        self.inner.padding_tokens
    }

    #[getter]
    fn efficiency(&self) -> f64 {
        self.inner.efficiency
    }

    #[getter]
    fn avg_utilisation(&self) -> f64 {
        self.inner.avg_utilisation
    }

    #[getter]
    fn utilisation_std(&self) -> f64 {
        self.inner.utilisation_std
    }

    #[getter]
    fn min_utilisation(&self) -> f64 {
        self.inner.min_utilisation
    }

    #[getter]
    fn max_utilisation(&self) -> f64 {
        self.inner.max_utilisation
    }

    #[getter]
    fn avg_sequences_per_pack(&self) -> f64 {
        self.inner.avg_sequences_per_pack
    }

    #[getter]
    fn packing_time_ms(&self) -> f64 {
        self.inner.packing_time_ms
    }

    // ── Computed methods (delegated to inner) ───────────────────

    #[getter]
    fn padding_ratio(&self) -> f64 {
        self.inner.padding_ratio()
    }

    #[getter]
    fn throughput(&self) -> f64 {
        self.inner.throughput()
    }

    // ── Python dunder methods ───────────────────────────────────

    fn __repr__(&self) -> String {
        format!(
            "PackMetrics(bins={}, efficiency={:.2}%, time={:.2}ms)",
            self.inner.num_packs,
            self.inner.efficiency * 100.0,
            self.inner.packing_time_ms,
        )
    }
}

impl From<PackMetrics> for PyMetrics {
    fn from(inner: PackMetrics) -> Self {
        Self { inner }
    }
}
