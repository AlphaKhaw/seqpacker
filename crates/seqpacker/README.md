<h1 align="center">seqpacker</h1>

<p align="center">
  <em>High-performance sequence packing for LLM training.</em>
</p>

<p align="center">
  <a href="https://crates.io/crates/seqpacker"><img src="https://img.shields.io/crates/v/seqpacker" alt="Crates.io"></a>
  <a href="https://docs.rs/seqpacker"><img src="https://img.shields.io/docsrs/seqpacker" alt="docs.rs"></a>
  <a href="https://github.com/AlphaKhaw/seqpacker/blob/main/LICENSE"><img src="https://img.shields.io/crates/l/seqpacker" alt="License"></a>
</p>

---

The core of [SeqPacker](https://github.com/AlphaKhaw/seqpacker), written in Rust. This crate provides 11 bin-packing algorithms for packing variable-length sequences into fixed-size bins, reducing padding waste from 20-40% down to 1-5%.

Python bindings are available via the [`seqpacker` PyPI package](https://pypi.org/project/seqpacker/).

## Quick Start

```rust
use seqpacker::{Packer, PackStrategy, Sequence};

let packer = Packer::new(2048)
    .with_strategy(PackStrategy::OptimizedBestFitDecreasing);

let sequences = vec![
    Sequence::new(0, 500),
    Sequence::new(1, 600),
    Sequence::new(2, 400),
    Sequence::new(3, 1000),
];

let result = packer.pack(sequences).unwrap();
println!("Bins: {}", result.packs.len());
println!("Efficiency: {:.2}%", result.metrics.efficiency * 100.0);
```

### Pack from lengths

```rust
use seqpacker::{Packer, PackStrategy};

let packer = Packer::new(1024)
    .with_strategy(PackStrategy::FirstFitDecreasing);

let result = packer.pack_lengths(&[1000, 800, 600, 500, 400, 300, 200, 100]).unwrap();

for pack in &result.packs {
    let ids: Vec<usize> = pack.items.iter().map(|s| s.id).collect();
    println!("Pack {}: items {:?}, used {}/{}", pack.id, ids, pack.used, pack.capacity);
}
```

### Streaming

For online / bounded-space packing, use `StreamPacker` with `NextFit` or `Harmonic`:

```rust
use seqpacker::{StreamPacker, StreamStrategy, Sequence};

let mut stream = StreamPacker::new(2048, StreamStrategy::NextFit);

let sequences = vec![
    Sequence::new(0, 500),
    Sequence::new(1, 600),
    Sequence::new(2, 1500),
    Sequence::new(3, 400),
];

for seq in sequences {
    for completed_pack in stream.add(seq).unwrap() {
        println!("Completed: {} items, {}/{} used", completed_pack.items.len(), completed_pack.used, completed_pack.capacity);
    }
}

// Flush remaining
for pack in stream.finish() {
    println!("Remaining: {} items", pack.items.len());
}
```

## Algorithms

11 bin-packing algorithms from O(n) online to near-optimal offline:

| Algorithm | Enum Variant | Time | Approx. Ratio | Best For |
|-----------|-------------|------|---------------|----------|
| NextFit | `NextFit` | O(n) | 2.0 | Memory-constrained streaming |
| FirstFit | `FirstFit` | O(n log B) | 1.7 | Online baseline |
| BestFit | `BestFit` | O(n log B) | 1.7 | Tighter online packing |
| WorstFit | `WorstFit` | O(n log B) | 2.0 | Even distribution |
| FirstFitDecreasing | `FirstFitDecreasing` | O(n log n) | 1.22 | Good offline default |
| BestFitDecreasing | `BestFitDecreasing` | O(n log n) | 1.22 | Tighter offline packing |
| FirstFitShuffle | `FirstFitShuffle` | O(n log n) | ~1.3 | Training randomness |
| ModifiedFFD | `ModifiedFirstFitDecreasing` | O(n log n) | 1.18 | Mixed-size distributions |
| **OptimizedBFD** | **`OptimizedBestFitDecreasing`** | **O(n log n)** | **1.22** | **Default (recommended)** |
| ParallelOBFD | `OptimizedBestFitDecreasingParallel` | O(n log n) | 1.22 | Large datasets (multi-threaded) |
| Harmonic-K | `Harmonic` | O(n) | ~1.69 | Bounded-space online |

Select an algorithm via `PackStrategy`:

```rust
use seqpacker::{Packer, PackStrategy};

let packer = Packer::new(2048)
    .with_strategy(PackStrategy::ModifiedFirstFitDecreasing);
```

## Performance

SeqPacker achieves equal packing efficiency to competitors while being significantly faster:

| Comparison | Speedup | Efficiency |
|------------|---------|------------|
| vs LightBinPack (C++) | ~1.2-1.5x faster | Equal (98.76%) |
| vs greedy_ffd (Python) | ~400x faster | Equal |
| vs binpacking (Python) | ~1,700x faster | Equal |

See the [interactive benchmark dashboard](https://alphakhaw.github.io/seqpacker/benchmarks/) for detailed results.

## License

MIT
