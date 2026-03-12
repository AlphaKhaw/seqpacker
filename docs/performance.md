# Performance

SeqPacker achieves equal packing efficiency to competitors while being significantly faster.

## Comparison

| Comparison | Speedup | Efficiency |
|------------|---------|------------|
| vs LightBinPack (C++) | ~1.2-1.5x faster | Equal (98.76%) |
| vs greedy_ffd (Python) | ~400x faster | Equal |
| vs binpacking (Python) | ~1,700x faster | Equal |
| vs prtpy (Python) | ~1,900x faster | Equal |

> Benchmarked on 10,000 sequences across real-world datasets (Alpaca, UltraChat, C4).

## Interactive Dashboard

Explore detailed benchmark results with the [interactive benchmark dashboard](https://alphakhaw.github.io/seqpacker/benchmarks/), including:

- Algorithm comparison across datasets
- Build optimization analysis (release vs PGO vs PGO+BOLT)
- Per-dataset efficiency and throughput breakdowns

## Why is SeqPacker fast?

The default OBFD algorithm achieves O(N log L) time complexity through:

1. **Counting sort** — O(N) instead of comparison sort O(N log N)
2. **Capacity-indexed segment tree** — O(log L) best-fit queries instead of O(log B)
3. **Integer-only arithmetic** — no floating-point operations in the hot path
4. **SmallVec optimization** — stack-allocated small vectors avoid heap allocation
5. **Early termination** — segment tree prunes search space during updates

For large datasets (>100k sequences), OBFDP parallelizes across threads via Rayon with adaptive thread count.
