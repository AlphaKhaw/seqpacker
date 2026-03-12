# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-07

### Added

- **11 bin-packing algorithms:** NF, FF, BF, WF, FFD, BFD, FFS, MFFD, OBFD, OBFDP, HK
- **StreamPacker API** for bounded-space streaming with NextFit and Harmonic-K
- **Python bindings** via PyO3/Maturin with full type stubs (PEP 561)
- **PyTorch integration** (`seqpacker.torch_utils`) with `PackedBatch`, `pack_result_to_tensors`, and `packed_collate_fn`
- **NumPy support** with zero-copy array input and flat output (`pack_flat`)
- **Comprehensive metrics:** efficiency, utilisation stats, padding ratio, throughput
- **Cross-platform support:** Linux, macOS, Windows
- **400 Rust tests** (381 unit, 9 golden integration, 5 property-based, 5 doc tests) and **249 Python tests**
