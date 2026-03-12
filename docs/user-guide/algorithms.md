# Algorithms

SeqPacker implements 11 bin-packing algorithms, from O(n) online to near-optimal offline.

## Algorithm Comparison

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

!!! tip "Which algorithm should I use?"
    Use **`obfd`** (the default) for most use cases. It matches the packing quality of FFD while being significantly faster thanks to a capacity-indexed segment tree.

    Use **`obfdp`** for datasets over 100k sequences — it parallelizes OBFD across threads via Rayon.

    Use **`ffs`** if you need training randomness — it shuffles sequences before packing, giving different packings each epoch with a deterministic seed.

## Online vs Offline

**Online algorithms** (NF, FF, BF, WF, HK) process sequences in arrival order. They work with streaming input but produce less efficient packings.

**Offline algorithms** (FFD, BFD, FFS, MFFD, OBFD, OBFDP) sort or preprocess all sequences before packing. They require all input upfront but produce tighter packings.

```python
from seqpacker import Packer

# Offline (recommended) — sees all data, packs optimally
packer = Packer(capacity=2048, strategy="obfd")
result = packer.pack(all_lengths)

# Online (streaming) — processes one at a time
from seqpacker import StreamPacker
sp = StreamPacker(capacity=2048, strategy="nf")
```

## Algorithm Details

### NextFit (NF)

Maintains a single open bin. Each item either fits in the current bin or starts a new one. O(n) time, O(1) space. Worst packing quality but fastest and simplest.

### FirstFit (FF)

Scans all open bins left-to-right, placing the item in the first bin with enough space. Uses a segment tree for O(log B) lookups.

### BestFit (BF)

Places each item in the bin with the **smallest remaining space** that still fits. Produces tighter packings than FF by minimizing wasted space per bin. Uses a B-tree index.

### WorstFit (WF)

Places each item in the bin with the **largest remaining space**. Spreads items more evenly across bins. Uses a B-tree index.

### FirstFitDecreasing (FFD)

Sorts items largest-to-smallest, then applies FirstFit. The sorting step ensures large items are placed first, leaving small gaps that smaller items can fill. Approximation ratio: 11/9 OPT + 6/9.

### BestFitDecreasing (BFD)

Sorts items largest-to-smallest, then applies BestFit. Similar quality to FFD with slightly tighter bin selection.

### FirstFitShuffle (FFS)

Shuffles items with a deterministic seed, then applies FirstFit. Useful for training — gives different packings per epoch while remaining reproducible.

```python
packer = Packer(capacity=2048, strategy="ffs", seed=42)
```

### ModifiedFFD (MFFD)

Johnson & Garey's 5-phase algorithm. Classifies items into size classes (large, medium, small, tiny) and pairs them strategically. Theoretical ratio: 71/60 OPT + 1. Most beneficial for mixed-size distributions.

### OptimizedBFD (OBFD)

The recommended default. Uses counting sort O(N) + a capacity-indexed segment tree for O(log L) best-fit queries. Same quality as BFD but significantly faster for large inputs.

### ParallelOBFD (OBFDP)

Parallelizes OBFD across threads using Rayon. Splits items cyclically across workers, runs OBFD independently, then repacks boundary bins. Adaptive thread count based on input size:

- N <= 20k: 1 thread (delegates to OBFD)
- N <= 100k: 2 threads
- N <= 500k: 4 threads
- N > 500k: all available threads

### Harmonic-K (HK)

Bounded-space online algorithm. Classifies items by size ratio and maintains one open bin per class. O(n) time, O(k) space. Can use more bins than FF/BF due to class segregation — this is expected behavior, not a bug.

```python
from seqpacker import StreamPacker

# Configurable k (number of size classes, default: 10)
sp = StreamPacker(capacity=2048, strategy="hk", k=10)
```
