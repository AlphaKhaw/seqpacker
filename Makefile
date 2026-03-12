# Prevent Maturin "Both VIRTUAL_ENV and CONDA_PREFIX are set" error.
unexport CONDA_PREFIX

# --------------------------------
# Setup
# --------------------------------

install:
	uv sync

# --------------------------------
# Build
# --------------------------------

build-dev:
	uv run maturin develop --uv

build-release:
	uv run maturin develop --release --uv

build-wheel:
	uv run maturin build --release

# --------------------------------
# Test
# --------------------------------

test: test-rust test-python

test-rust:
	cd crates/seqpacker && cargo test

test-python:
	PYTHONPATH=$(CURDIR) uv run pytest tests/

# --------------------------------
# Code quality
# --------------------------------

fmt: fmt-rust fmt-python

fmt-rust:
	cargo fmt --all

fmt-python:
	uv run ruff format .

lint: lint-rust lint-python

lint-rust:
	cargo clippy --workspace -- -D warnings

lint-python:
	uv run ruff check .

check-rust:
	cargo check --workspace

# --------------------------------
# Pre-commit
# --------------------------------

pre-commit-install:
	pre-commit install

pre-commit-update:
	pre-commit autoupdate

pre-commit-run:
	pre-commit run --all-files

# --------------------------------
# Benchmarks
# --------------------------------

bench-python: build-release
	PYTHONPATH=$(CURDIR) uv run python -m benchmarks.python.run_benchmarks
	PYTHONPATH=$(CURDIR) uv run python -m benchmarks.python.bench_seqpacker

bench-rust:
	cargo run --manifest-path benchmarks/rust/Cargo.toml --release

bench-all: bench-python bench-rust

bench-report:
	PYTHONPATH=$(CURDIR) uv run python -m benchmarks.python.generate_report

# --------------------------------
# Docs
# --------------------------------

docs-serve:
	uv run --group docs mkdocs serve

docs-build:
	uv run --group docs mkdocs build

# --------------------------------
# Cleanup
# --------------------------------

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf target/ *.egg-info dist/ build/ .pytest_cache/

# --------------------------------
# Help
# --------------------------------

help:
	@echo "Seqpacker Development Commands"
	@echo "==============================="
	@echo ""
	@echo "Setup:"
	@echo "  make install            Install Python dependencies (requires Rust + UV)"
	@echo ""
	@echo "Build:"
	@echo "  make build-dev          Dev build (fast, debug symbols)"
	@echo "  make build-release      Release build (optimized)"
	@echo "  make build-wheel        Build distributable wheel"
	@echo ""
	@echo "Test:"
	@echo "  make test               All tests (Rust + Python)"
	@echo "  make test-rust          Rust tests only"
	@echo "  make test-python        Python tests only"
	@echo ""
	@echo "Code Quality:"
	@echo "  make fmt                Format all (Rust + Python)"
	@echo "  make lint               Lint all (Rust + Python)"
	@echo "  make check-rust         cargo check"
	@echo ""
	@echo "Pre-commit:"
	@echo "  make pre-commit-install Install hooks"
	@echo "  make pre-commit-run     Run hooks on all files"
	@echo ""
	@echo "Benchmarks:"
	@echo "  make bench-python       Python benchmarks"
	@echo "  make bench-rust         Rust benchmarks (Criterion)"
	@echo "  make bench-all          All benchmarks"
	@echo "  make bench-report       Generate reports"
	@echo ""
	@echo "Docs:"
	@echo "  make docs-serve         Preview docs locally"
	@echo "  make docs-build         Build static docs site"
	@echo ""
	@echo "Other:"
	@echo "  make clean              Remove build artifacts"

.PHONY: install \
	build-dev build-release build-wheel \
	test test-rust test-python \
	fmt fmt-rust fmt-python \
	lint lint-rust lint-python check-rust \
	pre-commit-install pre-commit-update pre-commit-run \
	bench-python bench-rust bench-all bench-report \
	docs-serve docs-build \
	clean help
