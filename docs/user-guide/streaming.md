# Streaming

Feed sequences one at a time. Completed packs are emitted incrementally. Only bounded-space algorithms are supported: **NextFit** (`nf`) and **Harmonic-K** (`hk`).

## Why Only NF and HK?

FF, BF, WF, and all offline algorithms keep all bins open until the entire input is seen — they cannot emit completed packs incrementally. Only NF and Harmonic have bounded open bins, enabling true streaming output.

## NextFit Streaming

NextFit maintains a single open bin. When an item doesn't fit, the current bin is emitted and a new one opens.

```python
from seqpacker import StreamPacker

sp = StreamPacker(capacity=2048, strategy="nf")

for length in dataset_lengths:
    for pack in sp.add(length):
        process(pack)  # completed packs emitted as they fill

for pack in sp.finish():
    process(pack)      # flush remaining
```

## Harmonic-K Streaming

Harmonic-K maintains `k` open bins (one per size class). Items are routed to their class bin. When a class bin fills, it's emitted.

```python
sp = StreamPacker(capacity=2048, strategy="hk", k=10)

for length in dataset_lengths:
    for pack in sp.add(length):
        process(pack)

for pack in sp.finish():
    process(pack)
```

!!! note
    Harmonic-K can use **more** bins than NextFit due to class segregation. This is expected — theoretical bounds are for worst-case, not average-case.

## Lifecycle

The `StreamPacker` has a strict lifecycle:

1. **Adding** — call `sp.add(length)` repeatedly
2. **Finishing** — call `sp.finish()` once to flush remaining bins
3. **Done** — subsequent calls to `add()` or `finish()` raise `ValueError`

```python
sp = StreamPacker(capacity=2048, strategy="nf")
sp.add(500)    # OK
sp.add(600)    # OK
sp.finish()    # OK — flushes remaining
sp.add(100)    # ValueError: StreamPacker already finished
```

## Tracking Progress

```python
sp = StreamPacker(capacity=2048, strategy="nf")

for length in lengths:
    sp.add(length)

print(f"Sequences added: {sp.sequences_added}")
```

## Error Handling

```python
sp = StreamPacker(capacity=2048, strategy="nf")

# Sequence too long
sp.add(3000)  # ValueError: sequence length 3000 exceeds capacity 2048

# Invalid strategy
StreamPacker(capacity=2048, strategy="obfd")
# ValueError: strategy 'obfd' not supported for streaming.
# Use Packer.pack() for offline algorithms.
```
