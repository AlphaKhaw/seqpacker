//! Python bindings for seqpacker via PyO3.
//!
//! This crate compiles into a cdylib (`_core.so`) that Maturin installs as
//! `seqpacker._core`. The public Python package (`python/seqpacker/__init__.py`)
//! re-exports symbols from this module.
//!
//! ## Exported classes
//!
//! | Python name   | Rust wrapper        | Core type              |
//! |---------------|---------------------|------------------------|
//! | `Packer`      | `packer::PyPacker`  | `seqpacker::Packer`    |
//! | `PackResult`  | `packer::PyPackResult` | —                   |
//! | `Pack`        | `pack::PyPack`      | `seqpacker::Pack`      |
//! | `Metrics`     | `metrics::PyMetrics`| `seqpacker::PackMetrics`|
//! | `StreamPacker`| `stream_packer::PyStreamPacker` | `seqpacker::StreamPacker` |
//!
//! ## Convenience function
//!
//! `pack_sequences()` — one-shot packing without constructing a `Packer`.

mod metrics;
mod pack;
mod packer;
mod strategy;
mod stream_packer;

use pyo3::prelude::*;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<packer::PyPacker>()?;
    m.add_class::<packer::PyPackResult>()?;
    m.add_class::<pack::PyPack>()?;
    m.add_class::<metrics::PyMetrics>()?;
    m.add_class::<stream_packer::PyStreamPacker>()?;

    // Convenience function
    m.add_function(wrap_pyfunction!(pack_sequences, m)?)?;

    Ok(())
}

/// Pack sequence lengths into bins.
///
/// Args:
///     lengths (list[int] | numpy.ndarray): Sequence lengths to pack.
///     capacity (int): Maximum bin capacity.
///     strategy (str): Algorithm short name (default: "obfd"). See Packer.strategies().
///     seed (int | None): Random seed for shuffle-based algorithms.
///
/// Returns:
///     PackResult: Packing result with bins and metrics.
///
/// Raises:
///     ValueError: If strategy is unknown, any length exceeds capacity, or input is empty.
#[pyfunction]
#[pyo3(signature = (lengths, capacity, strategy="obfd", seed=None))]
fn pack_sequences(
    lengths: &Bound<'_, PyAny>,
    capacity: usize,
    strategy: &str,
    seed: Option<u64>,
) -> PyResult<packer::PyPackResult> {
    let packer = packer::PyPacker::create(capacity, strategy, seed)?;
    packer.pack_any(lengths)
}
