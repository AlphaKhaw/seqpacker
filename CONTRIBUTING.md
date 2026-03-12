# Contributing to Seqpacker

## Prerequisites

### Required

1. **Rust** (latest stable)
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **Python 3.9+**
   ```bash
   python --version
   ```

3. **UV** (package manager)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

### Optional

- **pre-commit** - Automatic formatting/linting on commit
- **rust-analyzer** - LSP for Rust development
- **pyright** - Type checking for Python

---

## Quick Start

```bash
# 1. Clone and enter the repository
git clone https://github.com/AlphaKhaw/seqpacker.git
cd seqpacker

# 2. Install Python dependencies (includes maturin)
make install

# 3. Build the Rust extension in dev mode
make build-dev

# 4. Run all tests to verify everything works
make test

# 5. (Optional) Set up pre-commit hooks
make pre-commit-install
```

---

## Project Structure

```
seqpacker/
├── Cargo.toml                     # Rust workspace root
├── pyproject.toml                 # Python package config (maturin build backend)
├── Makefile                       # Development commands (run `make help`)
│
├── crates/seqpacker/              # Pure Rust core library (publishable to crates.io)
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs                 # Crate root, public API re-exports
│   │   ├── error.rs               # PackError (thiserror)
│   │   ├── sequence.rs            # Item type
│   │   ├── pack.rs                # Bin type
│   │   ├── metrics.rs             # PackMetrics + MetricsBuilder
│   │   ├── strategy.rs            # PackStrategy enum (11 variants)
│   │   ├── engine.rs              # Generic greedy_pack() engine
│   │   ├── packer.rs              # Main Packer interface
│   │   ├── stream.rs              # StreamPacker (bounded-space streaming)
│   │   ├── validation.rs          # validate_solution() invariant checker
│   │   ├── dev.rs                 # Dev helpers
│   │   ├── placement/             # Bin-selection data structures
│   │   │   ├── mod.rs             # PlacementIndex trait
│   │   │   ├── segment_tree.rs    # O(log B) first-fit
│   │   │   ├── capacity_segment_tree.rs  # O(log L) best-fit by capacity
│   │   │   ├── btree.rs           # O(log B) best-fit / worst-fit
│   │   │   └── linear.rs          # O(B) linear scan (debugging)
│   │   └── algorithms/            # 11 packing algorithms
│   │       ├── mod.rs
│   │       ├── next_fit.rs        # NF  — O(n), single open bin
│   │       ├── first_fit.rs       # FF  — segment tree + greedy_pack
│   │       ├── best_fit.rs        # BF  — BTree + greedy_pack
│   │       ├── worst_fit.rs       # WF  — BTree + greedy_pack
│   │       ├── first_fit_decreasing.rs   # FFD  — sort desc + FF
│   │       ├── best_fit_decreasing.rs    # BFD  — sort desc + BF
│   │       ├── first_fit_shuffle.rs      # FFS  — shuffle + FF (NeMo-style)
│   │       ├── modified_first_fit_decreasing.rs  # MFFD — 5-phase Johnson & Garey
│   │       ├── optimized_best_fit_decreasing.rs  # OBFD — capacity segment tree (default)
│   │       ├── optimized_best_fit_decreasing_parallel.rs  # OBFDP — Rayon parallel
│   │       ├── harmonic.rs        # HK  — bounded-space online
│   │       └── counting_sort.rs   # Shared utility
│   └── tests/
│       ├── golden.rs              # Golden invariant tests
│       └── proptest_packing.rs    # Property-based tests
│
├── bindings/python/               # PyO3 cdylib wrapper (produces _core.abi3.so)
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                 # #[pymodule] fn _core
│       ├── packer.rs              # PyPacker, pack_sequences()
│       ├── pack.rs                # PyPack
│       ├── strategy.rs            # Strategy string parsing
│       ├── metrics.rs             # PyMetrics
│       └── stream_packer.rs       # PyStreamPacker
│
├── python/seqpacker/              # Python package (re-exports from _core)
│   ├── __init__.py                # Public API
│   ├── __init__.pyi               # Type stubs (PEP 561)
│   ├── py.typed                   # PEP 561 marker
│   └── torch_utils.py             # PyTorch integration (opt-in, lazy import)
│
├── tests/python/                  # Python tests
│   ├── test_bindings.py           # Core API tests (~166)
│   ├── test_stream_packer.py      # StreamPacker tests (30)
│   ├── test_torch_utils.py        # PyTorch tests (requires torch)
│   └── benchmarks/                # Benchmark harness tests
│       ├── test_metrics.py
│       ├── test_runners.py
│       └── test_synthetic.py
│
├── benchmarks/
│   ├── python/                    # Python benchmark suite
│   │   ├── bench_seqpacker.py     # Algorithm comparison
│   │   ├── run_benchmarks.py      # Competitor benchmarks
│   │   ├── generate_report.py     # Report generation
│   │   ├── datasets/              # Real + synthetic data generators
│   │   ├── runners/               # Competitor algorithm wrappers
│   │   ├── metrics/               # Benchmark metric models
│   │   └── utils/                 # Logging utilities
│   ├── rust/                      # Criterion benchmarks
│   │   ├── Cargo.toml
│   │   └── src/
│   └── results/                   # Pre-computed benchmark data (JSON)
│
├── docs/benchmarks/               # Static benchmark dashboard (Observable Framework)
│
└── .github/workflows/
    ├── ci.yml                     # Lint, test (Rust + Python 3.9/3.12/3.13), MSRV check
    └── release.yml                # Wheel builds + PyPI/crates.io publish
```

**Build chain:** `pyproject.toml` → Maturin → compiles `bindings/python/` → produces `python/seqpacker/_core.abi3.so`

**Important:** Only ONE `pyproject.toml` at root. The `bindings/python/` folder contains only a `Cargo.toml` for the PyO3 cdylib crate.

---

## Development Workflow

```bash
# 1. Make changes to Rust or Python code
# 2. Rebuild the extension
make build-dev
# 3. Run tests
make test
# 4. Format and lint
make fmt && make lint
# 5. Commit (pre-commit hooks run automatically if installed)
```

---

## Commands Reference

Run `make help` for all commands. Key ones:

### Building
```bash
make build-dev       # Dev build (fast compile, debug symbols)
make build-release   # Release build (optimized)
make build-wheel     # Build distributable wheel
```

### Testing
```bash
make test            # All tests (Rust + Python)
make test-rust       # Rust tests only
make test-python     # Python tests only
```

Run a single test:
```bash
cargo test -p seqpacker test_name                              # Rust
uv run pytest tests/python/test_bindings.py::test_name -v      # Python
```

### Code Quality
```bash
make fmt             # Format all (Rust + Python)
make lint            # Lint all (clippy + ruff)
make check-rust      # cargo check
```

### Benchmarking

Benchmark dependencies (torch, transformers, competitor libraries) are not installed by default due to their size (~2GB). Install them explicitly when needed:

```bash
uv sync --group bench
make bench-python    # Python benchmarks (builds release first)
make bench-rust      # Rust benchmarks (Criterion)
make bench-all       # All benchmarks
make bench-report    # Generate reports
```

---

## Common Tasks

### Adding a New Algorithm

1. Create `crates/seqpacker/src/algorithms/your_algorithm.rs`
2. Implement the algorithm (see existing files for patterns)
3. Export in `crates/seqpacker/src/algorithms/mod.rs`
4. Add variant to `PackStrategy` enum in `crates/seqpacker/src/strategy.rs`
5. Add Python string mapping in `bindings/python/src/strategy.rs`
6. Add tests and verify: `make build-dev && make test`

### Adding Dependencies

**Rust** — add to workspace root `Cargo.toml`:
```toml
[workspace.dependencies]
your-crate = "1.0"
```
Then use in member crate:
```toml
[dependencies]
your-crate.workspace = true
```

**Python:**
```bash
uv add package-name        # Production
uv add --dev dev-package   # Development
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `maturin: command not found` | `make install` (maturin is a dev dependency) |
| `linker cc not found` (macOS) | `xcode-select --install` |
| `linker cc not found` (Linux) | `sudo apt-get install build-essential` |
| Import error after Rust changes | `make build-dev` to rebuild the extension |
| Pre-commit hooks failing | `make pre-commit-update` to sync hook versions |
| Slow Rust compilation | Use `make build-dev` (not `build-release`) during development |

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
