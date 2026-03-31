# SeqPacker

**High-performance sequence packing for LLM training, written in Rust with Python bindings.**

---

Training LLMs on variable-length sequences? Naive padding wastes **20-40% of GPU compute**. SeqPacker packs sequences into fixed-size bins, achieving **95-99% utilization** with 11 bin-packing algorithms — from O(n) streaming to near-optimal offline.

## Key Features

- **11 algorithms** — NF, FF, BF, WF, FFD, BFD, FFS, MFFD, OBFD, OBFDP, HK
- **Streaming API** — bounded-space packing with incremental output
- **HuggingFace integration** — one-call `pack_dataset` for SFTTrainer / TRL
- **PyTorch integration** — GPU-ready tensors out of the box
- **NumPy zero-copy** — pass arrays directly, no conversion overhead
- **Cross-platform** — Linux, macOS, Windows; Python 3.9-3.13

## Quick Example

=== "Python"

    ```python
    from seqpacker import pack_sequences

    lengths = [1000, 800, 600, 500, 400, 300, 200, 100]
    result = pack_sequences(lengths, capacity=1024)

    print(result.bins)        # [[0], [1, 7], [2, 4], [3, 5, 6]]
    print(result.efficiency)  # 0.952...
    ```

=== "Rust"

    ```rust
    use seqpacker::{Packer, PackStrategy};

    let packer = Packer::new(1024)
        .with_strategy(PackStrategy::OptimizedBestFitDecreasing);

    let result = packer.pack_lengths(&[1000, 800, 600, 500, 400, 300, 200, 100]).unwrap();
    println!("Efficiency: {:.2}%", result.metrics.efficiency * 100.0);
    ```

## Next Steps

- [Getting Started](getting-started.md) — installation and first steps
- [User Guide](user-guide/algorithms.md) — algorithm selection and usage patterns
- [Python API Reference](api.md) — full Python API documentation
- [Rust API Reference](https://docs.rs/seqpacker) — auto-generated on docs.rs
- [Performance](performance.md) — benchmarks and comparisons
