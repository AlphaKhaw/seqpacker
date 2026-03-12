//! Python wrapper for `Pack`.
//!
//! Exposes individual bin contents as read-only Python properties.
//! Each `PyPack` wraps a Rust `Pack` and provides access to sequence
//! IDs, lengths, and utilisation without exposing the inner struct.

use pyo3::prelude::*;
use seqpacker::Pack;

/// A single packed bin (read-only).
///
/// Wraps the Rust `Pack` struct. Provides sequence IDs, lengths,
/// and total token usage as Python properties.
#[pyclass(name = "Pack", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyPack {
    pub(crate) inner: Pack,
}

#[pymethods]
impl PyPack {
    /// Sequence IDs in this pack.
    #[getter]
    fn sequence_ids(&self) -> Vec<usize> {
        self.inner.sequences.iter().map(|s| s.id).collect()
    }

    /// Sequence lengths in this pack.
    #[getter]
    fn lengths(&self) -> Vec<usize> {
        self.inner.sequences.iter().map(|s| s.length).collect()
    }

    /// Total tokens used in this pack.
    #[getter]
    fn used(&self) -> usize {
        self.inner.sequences.iter().map(|s| s.length).sum()
    }

    /// Number of sequences in this pack.
    fn __len__(&self) -> usize {
        self.inner.sequences.len()
    }

    fn __repr__(&self) -> String {
        let lens: Vec<usize> = self.lengths();
        format!("Pack(sequences={}, used={})", lens.len(), self.used())
    }
}

impl From<Pack> for PyPack {
    fn from(inner: Pack) -> Self {
        Self { inner }
    }
}
