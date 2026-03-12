# seqpacker

High-performance sequence packing for LLM training, written in Rust with Python bindings.

[![CI](https://github.com/AlphaKhaw/seqpacker/actions/workflows/ci.yml/badge.svg)](https://github.com/AlphaKhaw/seqpacker/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/seqpacker)](https://crates.io/crates/seqpacker)
[![PyPI](https://img.shields.io/pypi/v/seqpacker)](https://pypi.org/project/seqpacker/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

Training LLMs on variable-length sequences? Naive padding wastes **20-40% of GPU compute**. seqpacker packs sequences into fixed-size bins, achieving **95-99% utilization** with 11 bin-packing algorithms — from O(n) streaming to near-optimal offline.

**Key features:**
- 11 algorithms (NF, FF, BF, WF, FFD, BFD, FFS, MFFD, OBFD, OBFDP, HK)
- Streaming API for bounded-space packing
- PyTorch integration with GPU-ready tensors
- NumPy zero-copy input support
- Cross-platform: Linux, macOS, Windows
- Python 3.9-3.13

## Installation

```bash
# Python (pip)
pip install seqpacker

# Python (uv)
uv add seqpacker

# Rust
cargo add seqpacker
```

## Quick Start

### Python

```python
from seqpacker import pack_sequences

lengths = [1000, 800, 600, 500, 400, 300, 200, 100]
result = pack_sequences(lengths, capacity=1024)

print(result.bins)        # [[0], [1, 7], [2, 4], [3, 5, 6]]
print(result.efficiency)  # 0.952...
```

### Rust

```rust
use seqpacker::{Packer, PackStrategy};

let packer = Packer::new(1024)
    .with_strategy(PackStrategy::OptimizedBestFitDecreasing);

let result = packer.pack_lengths(&[1000, 800, 600, 500, 400, 300, 200, 100]).unwrap();
println!("Efficiency: {:.2}%", result.metrics.efficiency * 100.0);
```

## Algorithms

11 bin-packing algorithms from O(n) online to optimal offline:

| Algorithm | Short | Time | Approx. Ratio | Best For |
|-----------|-------|------|---------------|----------|
| NextFit | `nf` | O(n) | 2.0 | Memory-constrained streaming |
| FirstFit | `ff` | O(n log B) | 1.7 | Online baseline |
| BestFit | `bf` | O(n log B) | 1.7 | Tighter online packing |
| WorstFit | `wf` | O(n log B) | 2.0 | Even distribution |
| FirstFitDecreasing | `ffd` | O(n log n) | 1.22 | Good offline default |
| BestFitDecreasing | `bfd` | O(n log n) | 1.22 | Tighter offline packing |
| FirstFitShuffle | `ffs` | O(n log n) | ~1.3 | Training randomness |
| ModifiedFFD | `mffd` | O(n log n) | 1.18 | Mixed-size distributions |
| OptimizedBFD | **`obfd`** | O(n log n) | 1.22 | **Default (recommended)** |
| ParallelOBFD | `obfdp` | O(n log n) | 1.22 | Large datasets (multi-threaded) |
| Harmonic-K | `hk` | O(n) | ~1.69 | Bounded-space online |

```python
from seqpacker import Packer

# Use any algorithm by short name (default: obfd)
packer = Packer(capacity=2048, strategy="obfd")
result = packer.pack([500, 600, 400, 1000])

# List all available strategies
print(Packer.strategies())
```

## Usage Modes

### Batch Packing

Pack all sequences at once. Best for offline dataset preprocessing. All 11 algorithms available.

```python
from seqpacker import Packer

packer = Packer(capacity=2048, strategy="obfd")
result = packer.pack(sequence_lengths)

for pack in result.packs:
    print(pack.sequence_ids, pack.lengths, pack.used)

print(f"Efficiency: {result.efficiency:.2%}")
print(f"Packs: {result.num_bins}")
```

### Streaming

Feed sequences one at a time. Completed packs are emitted incrementally. Only bounded-space algorithms supported: NextFit (`nf`) and Harmonic-K (`hk`).

```python
from seqpacker import StreamPacker

sp = StreamPacker(capacity=2048, strategy="nf")

for length in dataset_lengths:
    for pack in sp.add(length):
        process(pack)  # completed packs emitted as they fill

for pack in sp.finish():
    process(pack)      # flush remaining
```

### Buffer + Batch

Accumulate sequences into a buffer and pack periodically. Requires no special library support -- just call `pack()` on each buffer. All algorithms available.

```python
from seqpacker import Packer

packer = Packer(capacity=2048, strategy="obfd")
buffer = []

for sample in dataset_stream:
    buffer.append(len(sample["input_ids"]))
    if len(buffer) >= 10_000:
        result = packer.pack(buffer)
        for pack in result.packs:
            yield pack
        buffer.clear()

if buffer:
    result = packer.pack(buffer)
    for pack in result.packs:
        yield pack
```

## PyTorch Integration

`seqpacker.torch_utils` provides helpers for converting pack results into GPU-ready tensors. Torch is not a dependency -- import only when you need it.

```python
from seqpacker.torch_utils import packed_collate_fn
from torch.utils.data import DataLoader

collate = packed_collate_fn(capacity=2048, strategy="obfd")
loader = DataLoader(dataset, collate_fn=collate, batch_size=256)

for batch in loader:
    outputs = model(
        input_ids=batch.input_ids,
        position_ids=batch.position_ids,
        labels=batch.labels,
    )
```

Or convert a `PackResult` directly:

```python
from seqpacker import pack_sequences
from seqpacker.torch_utils import pack_result_to_tensors

result = pack_sequences(lengths, capacity=2048)
batch = pack_result_to_tensors(result=result, token_ids=token_ids)
# batch.input_ids, batch.cu_seqlens, batch.position_ids, batch.labels, batch.attention_mask
```

## NumPy Support

Both list and NumPy array inputs are supported with zero-copy for NumPy:

```python
import numpy as np
from seqpacker import Packer

packer = Packer(capacity=2048)
lengths = np.array([500, 600, 400, 1000], dtype=np.int64)
result = packer.pack(lengths)

# Flat NumPy output for maximum performance
items_flat, bin_offsets = packer.pack_flat(lengths)
bins = np.split(items_flat, bin_offsets)
```

## Performance

seqpacker achieves equal packing efficiency to competitors while being significantly faster:

| Comparison | Speedup | Efficiency |
|------------|---------|------------|
| vs LightBinPack (C++) | ~1.2-1.5x faster | Equal (98.76%) |
| vs greedy_ffd (Python) | ~400x faster | Equal |
| vs binpacking (Python) | ~1,700x faster | Equal |
| vs prtpy (Python) | ~1,900x faster | Equal |

> Benchmarked on 10,000 sequences across real-world datasets (Alpaca, UltraChat, C4).
> See the [interactive benchmark dashboard](https://alphakhaw.github.io/seqpacker/benchmarks/) for detailed results.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions and development workflow.

```bash
make install       # Install dependencies
make build-dev     # Build the Rust extension
make test          # Run all tests (400 Rust + 249 Python)
make help          # See all commands
```

## License

MIT
