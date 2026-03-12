//! Python wrapper for `StreamPacker`.
//!
//! Exposes streaming packing via `add()` and `finish()` methods.
//! Only bounded-space algorithms (NextFit, Harmonic-K) are supported.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use seqpacker::stream::{StreamPacker, StreamStrategy};

use crate::pack::PyPack;

/// Streaming packer for incremental sequence packing.
///
/// Only bounded-space online algorithms are supported:
/// ``"nf"`` (NextFit) and ``"hk"`` (Harmonic-K).
///
/// Example::
///
///     sp = StreamPacker(capacity=2048, strategy="nf")
///     for length in dataset_lengths:
///         for pack in sp.add(length):
///             process(pack)
///     for pack in sp.finish():
///         process(pack)
#[pyclass(name = "StreamPacker")]
pub struct PyStreamPacker {
    inner: Option<StreamPacker>,
}

#[pymethods]
impl PyStreamPacker {
    /// Create a new streaming packer.
    ///
    /// Args:
    ///     capacity (int): Maximum bin capacity in tokens.
    ///     strategy (str): Algorithm: ``"nf"`` (NextFit) or ``"hk"`` (Harmonic-K).
    ///     k (int | None): Number of size classes for Harmonic-K (default: 10).
    ///
    /// Raises:
    ///     ValueError: If strategy is not ``"nf"`` or ``"hk"``.
    #[new]
    #[pyo3(signature = (capacity, strategy="nf", k=None))]
    fn new(capacity: usize, strategy: &str, k: Option<usize>) -> PyResult<Self> {
        let stream_strategy = parse_stream_strategy(strategy)?;
        let k = k.unwrap_or(10);
        Ok(Self {
            inner: Some(StreamPacker::with_k(capacity, stream_strategy, k)),
        })
    }

    /// Add a sequence length and return any completed packs.
    ///
    /// Args:
    ///     length (int): Sequence length to add.
    ///
    /// Returns:
    ///     list[Pack]: Packs that are now complete (may be empty).
    ///
    /// Raises:
    ///     ValueError: If length exceeds capacity or packer is already finished.
    fn add(&mut self, length: usize) -> PyResult<Vec<PyPack>> {
        let sp = self
            .inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("StreamPacker already finished"))?;
        let closed = sp
            .add(length)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(closed.into_iter().map(PyPack::from).collect())
    }

    /// Flush all remaining open bins and return them as packs.
    ///
    /// After calling ``finish()``, the packer cannot be used again.
    ///
    /// Returns:
    ///     list[Pack]: All remaining packs.
    ///
    /// Raises:
    ///     ValueError: If packer is already finished.
    fn finish(&mut self) -> PyResult<Vec<PyPack>> {
        let sp = self
            .inner
            .take()
            .ok_or_else(|| PyValueError::new_err("StreamPacker already finished"))?;
        Ok(sp.finish().into_iter().map(PyPack::from).collect())
    }

    /// Number of sequences added so far.
    #[getter]
    fn sequences_added(&self) -> PyResult<usize> {
        let sp = self
            .inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("StreamPacker already finished"))?;
        Ok(sp.sequences_added())
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            Some(sp) => format!(
                "StreamPacker(strategy={:?}, added={})",
                sp.strategy(),
                sp.sequences_added()
            ),
            None => "StreamPacker(finished)".to_string(),
        }
    }
}

/// Parse streaming strategy from string.
fn parse_stream_strategy(name: &str) -> PyResult<StreamStrategy> {
    match name.to_lowercase().as_str() {
        "nf" | "nextfit" | "next_fit" => Ok(StreamStrategy::NextFit),
        "hk" | "harmonic" => Ok(StreamStrategy::Harmonic),
        _ => Err(PyValueError::new_err(format!(
            "unknown streaming strategy '{name}'. \
             Only bounded-space algorithms are supported: \"nf\" (NextFit), \"hk\" (Harmonic-K). \
             For other algorithms, use Packer.pack() instead."
        ))),
    }
}
