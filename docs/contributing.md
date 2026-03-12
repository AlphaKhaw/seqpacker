# Contributing

See [CONTRIBUTING.md](https://github.com/AlphaKhaw/seqpacker/blob/main/CONTRIBUTING.md) for the full development guide, including:

- Repository structure
- Build and test commands
- Code style conventions
- How to add new algorithms

## Quick Setup

```bash
git clone https://github.com/AlphaKhaw/seqpacker.git
cd seqpacker
make install       # uv sync
make build-dev     # maturin develop
make test          # Run all tests
make help          # See all commands
```

## Running Tests

```bash
make test          # All tests (Rust + Python)
make test-rust     # Rust only
make test-python   # Python only

# Single Rust test
cargo test -p seqpacker test_name

# Single Python test
uv run pytest tests/python/test_bindings.py::test_name -v
```
