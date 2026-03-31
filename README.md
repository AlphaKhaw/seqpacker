<h1 align="center">SeqPacker</h1>

<p align="center">
  <em>High-performance sequence packing for LLM training, written in Rust with Python bindings.</em>
</p>

<p align="center">
  <a href="https://github.com/AlphaKhaw/seqpacker/actions/workflows/ci.yml"><img src="https://github.com/AlphaKhaw/seqpacker/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://crates.io/crates/seqpacker"><img src="https://img.shields.io/crates/v/seqpacker" alt="Crates.io"></a>
  <a href="https://pypi.org/project/seqpacker/"><img src="https://img.shields.io/pypi/v/seqpacker" alt="PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
</p>

<p align="center">
  <a href="https://alphakhaw.github.io/seqpacker/docs/">Documentation</a>
  &ensp;|&ensp;
  <a href="https://docs.rs/seqpacker">Rust API</a>
  &ensp;|&ensp;
  <a href="https://alphakhaw.github.io/seqpacker/benchmarks/">Benchmarks</a>
  &ensp;|&ensp;
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

---

Training LLMs on variable-length sequences? Naive padding wastes **20-40% of GPU compute**. SeqPacker packs sequences into fixed-size bins, achieving **95-99% utilization** with 11 bin-packing algorithms — from O(n) streaming to near-optimal offline.

- **11 algorithms** — NF, FF, BF, WF, FFD, BFD, FFS, MFFD, OBFD, OBFDP, HK
- **Streaming API** — bounded-space packing with incremental output
- **HuggingFace integration** — one-call `pack_dataset` for SFTTrainer / TRL
- **PyTorch integration** — GPU-ready tensors out of the box
- **NumPy zero-copy** — pass arrays directly, no conversion overhead
- **Cross-platform** — Linux, macOS, Windows; Python 3.9-3.13

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

## Training Integration

### HuggingFace Trainer

`seqpacker.hf_utils` builds a packed `datasets.Dataset` in one call -- ready for SFTTrainer, TRL, or any HF Trainer workflow. `datasets` is not a dependency -- import only when you need it.

```python
from seqpacker.hf_utils import pack_dataset

tokenized = tokenizer(texts, truncation=True, max_length=2048)
ds = pack_dataset(tokenized["input_ids"], capacity=2048)

trainer = SFTTrainer(model=model, train_dataset=ds, ...)
```

The returned dataset includes `input_ids`, `attention_mask`, `labels` (shifted with boundary masking), and `position_ids` (per-sequence reset).

### PyTorch DataLoader

`seqpacker.torch_utils` provides helpers for converting pack results into GPU-ready tensors. `torch` is not a dependency -- import only when you need it.

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

SeqPacker achieves equal packing efficiency to competitors while being significantly faster:

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
