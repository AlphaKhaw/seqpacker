//! Python wrappers for `Packer` and `PackResult`.
//!
//! `PyPacker` is the main entry point — Python users create one with a
//! capacity and strategy, then call `pack()` or `pack_flat()`.
//! Both methods accept either a Python list or a NumPy array.
//!
//! `PyPackResult` is the return type — combines bins and metrics with
//! convenience getters so Python users don't need to drill into sub-objects.

use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use seqpacker::{PackStrategy, Packer};

use crate::metrics::PyMetrics;
use crate::pack::PyPack;
use crate::strategy::parse_strategy;

/// Return type for `pack_flat()`: `(items_flat, bin_offsets)` as NumPy arrays.
type PackFlatResult<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>);

/// Combined packing result (read-only).
///
/// Contains the list of packed bins and associated metrics.
/// Convenience getters like `efficiency` and `bins` delegate
/// to the inner `PyMetrics` and `PyPack` objects.
#[pyclass(name = "PackResult", frozen)]
pub struct PyPackResult {
    packs: Vec<PyPack>,
    metrics: PyMetrics,
}

#[pymethods]
impl PyPackResult {
    /// List of Pack objects.
    #[getter]
    fn packs(&self) -> Vec<PyPack> {
        self.packs.clone()
    }

    /// Packing metrics.
    #[getter]
    fn metrics(&self) -> PyMetrics {
        self.metrics.clone()
    }

    /// Number of bins used.
    #[getter]
    fn num_bins(&self) -> usize {
        self.packs.len()
    }

    /// Packing efficiency (0.0-1.0).
    #[getter]
    fn efficiency(&self) -> f64 {
        self.metrics.inner.efficiency
    }

    /// Packing time in milliseconds.
    #[getter]
    fn time_ms(&self) -> f64 {
        self.metrics.inner.packing_time_ms
    }

    /// Bins as nested list of sequence IDs: [[0, 3], [1, 2], ...].
    #[getter]
    fn bins(&self) -> Vec<Vec<usize>> {
        self.packs
            .iter()
            .map(|p| p.inner.sequences.iter().map(|s| s.id).collect())
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "PackResult(bins={}, efficiency={:.2}%, time={:.2}ms)",
            self.packs.len(),
            self.metrics.inner.efficiency * 100.0,
            self.metrics.inner.packing_time_ms,
        )
    }

    fn __len__(&self) -> usize {
        self.packs.len()
    }
}

/// Main packing interface.
///
/// Wraps the Rust `Packer` with a string-based strategy API.
#[pyclass(name = "Packer", frozen)]
pub struct PyPacker {
    capacity: usize,
    strategy: PackStrategy,
    seed: Option<u64>,
}

#[pymethods]
impl PyPacker {
    #[new]
    #[pyo3(signature = (capacity, strategy="obfd", seed=None))]
    fn new(capacity: usize, strategy: &str, seed: Option<u64>) -> PyResult<Self> {
        let strategy = parse_strategy(strategy)?;
        Ok(Self {
            capacity,
            strategy,
            seed,
        })
    }

    /// Pack sequence lengths into bins.
    ///
    /// Accepts a Python list or a NumPy int64 array.
    ///
    /// Args:
    ///     lengths (list[int] | numpy.ndarray): Sequence lengths to pack.
    ///
    /// Returns:
    ///     PackResult: Packing result with bins, metrics, and efficiency.
    ///
    /// Raises:
    ///     TypeError: If lengths is not a list or NumPy array.
    ///     ValueError: If any length exceeds capacity or input is empty.
    #[pyo3(signature = (lengths))]
    fn pack(&self, lengths: &Bound<'_, PyAny>) -> PyResult<PyPackResult> {
        let lens = self.extract_lengths(lengths)?;
        self.pack_impl(&lens)
    }

    /// Pack into flat NumPy arrays for maximum performance.
    ///
    /// Accepts a Python list or a NumPy int64 array. Returns flat arrays
    /// instead of nested lists. Reconstruct bins with:
    ///     ``bins = np.split(items_flat, bin_offsets)``
    ///
    /// Args:
    ///     lengths (list[int] | numpy.ndarray): Sequence lengths to pack.
    ///
    /// Returns:
    ///     tuple[numpy.ndarray, numpy.ndarray]: (item_ids_flat, bin_offsets) as int64 arrays.
    ///
    /// Raises:
    ///     TypeError: If lengths is not a list or NumPy array.
    ///     ValueError: If any length exceeds capacity or input is empty.
    #[pyo3(signature = (lengths))]
    fn pack_flat<'py>(
        &self,
        py: Python<'py>,
        lengths: &Bound<'py, PyAny>,
    ) -> PyResult<PackFlatResult<'py>> {
        let lens = self.extract_lengths(lengths)?;
        let result = self.run_packer(&lens)?;

        let mut items_flat: Vec<i64> = Vec::new();
        let mut bin_offsets: Vec<i64> = Vec::new();
        let mut offset = 0i64;

        for pack in &result.packs {
            for seq in &pack.sequences {
                items_flat.push(seq.id as i64);
            }
            offset += pack.sequences.len() as i64;
            bin_offsets.push(offset);
        }
        // Remove last offset (np.split doesn't need it)
        bin_offsets.pop();

        Ok((
            PyArray1::from_vec(py, items_flat),
            PyArray1::from_vec(py, bin_offsets),
        ))
    }

    /// Return all available strategy names.
    ///
    /// Returns:
    ///     list[tuple[str, str]]: List of (short_name, full_name) pairs.
    #[staticmethod]
    fn strategies() -> Vec<(String, String)> {
        crate::strategy::available_strategies()
            .into_iter()
            .map(|(short, full)| (short.to_string(), full.to_string()))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Packer(capacity={}, strategy='{}')",
            self.capacity,
            self.strategy.short_name().unwrap_or(self.strategy.name()),
        )
    }
}

impl PyPacker {
    /// Create a `PyPacker` from Rust code (used by `pack_sequences`).
    pub(crate) fn create(capacity: usize, strategy: &str, seed: Option<u64>) -> PyResult<Self> {
        let strategy = parse_strategy(strategy)?;
        Ok(Self {
            capacity,
            strategy,
            seed,
        })
    }

    /// Pack from Rust code (used by `pack_sequences`).
    pub(crate) fn pack_any(&self, lengths: &Bound<'_, PyAny>) -> PyResult<PyPackResult> {
        let lens = self.extract_lengths(lengths)?;
        self.pack_impl(&lens)
    }

    /// Extract lengths from either a Python list or NumPy array.
    ///
    /// Tries NumPy first (zero-copy fast path), then falls back to list.
    /// Raises TypeError with a clear message if neither works.
    fn extract_lengths(&self, lengths: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
        // Fast path: try NumPy array first (zero-copy read)
        if let Ok(array) = lengths.cast::<numpy::PyArray1<i64>>() {
            let readonly = array.readonly();
            let slice = readonly
                .as_slice()
                .map_err(|e| PyValueError::new_err(format!("non-contiguous array: {e}")))?;
            return Ok(slice.iter().map(|&v| v as usize).collect());
        }

        // Fallback: try Python list/sequence → Vec<usize>
        if let Ok(vec) = lengths.extract::<Vec<usize>>() {
            return Ok(vec);
        }

        // Neither worked — give a helpful error
        Err(PyTypeError::new_err(format!(
            "expected list[int] or numpy.ndarray, got {}",
            lengths.get_type().qualname()?
        )))
    }

    fn pack_impl(&self, lengths: &[usize]) -> PyResult<PyPackResult> {
        let result = self.run_packer(lengths)?;
        let packs: Vec<PyPack> = result.packs.into_iter().map(PyPack::from).collect();
        let metrics = PyMetrics::from(result.metrics);
        Ok(PyPackResult { packs, metrics })
    }

    fn run_packer(&self, lengths: &[usize]) -> PyResult<seqpacker::PackResult> {
        let mut packer = Packer::new(self.capacity).with_strategy(self.strategy);
        if let Some(seed) = self.seed {
            packer = packer.with_seed(seed);
        }
        packer
            .pack_lengths(lengths)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}
