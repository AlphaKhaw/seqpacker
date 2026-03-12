# Batch Packing

Pack all sequences at once. Best for offline dataset preprocessing. All 11 algorithms available.

## Basic Usage

```python
from seqpacker import Packer

packer = Packer(capacity=2048, strategy="obfd")
result = packer.pack(sequence_lengths)

for pack in result.packs:
    print(pack.sequence_ids, pack.lengths, pack.used)

print(f"Efficiency: {result.efficiency:.2%}")
print(f"Packs: {result.num_bins}")
```

## Convenience Function

For one-shot packing without constructing a `Packer`:

```python
from seqpacker import pack_sequences

result = pack_sequences(lengths, capacity=2048, strategy="obfd")
```

## Buffer + Batch Pattern

Accumulate sequences into a buffer and pack periodically. This requires no special library support — just call `pack()` on each buffer. All algorithms available.

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

# Don't forget the last batch
if buffer:
    result = packer.pack(buffer)
    for pack in result.packs:
        yield pack
```

!!! tip
    The buffer + batch pattern is useful for large streaming datasets where you can't load everything into memory. Set the buffer size based on your memory budget — 10k-100k sequences is typical.

## Working with Results

### Accessing bins

```python
result = packer.pack(lengths)

# Nested list of sequence IDs per bin
print(result.bins)  # [[0, 3], [1, 2], ...]

# Detailed per-pack info
for pack in result.packs:
    print(f"IDs: {pack.sequence_ids}")
    print(f"Lengths: {pack.lengths}")
    print(f"Used: {pack.used}/{packer.capacity}")
```

### Metrics

```python
metrics = result.metrics

print(f"Efficiency: {metrics.efficiency:.2%}")
print(f"Bins: {metrics.num_packs}")
print(f"Padding tokens: {metrics.padding_tokens}")
print(f"Time: {metrics.packing_time_ms:.3f} ms")
print(f"Throughput: {metrics.throughput:.0f} seq/ms")
```

### Flat output for performance

For maximum performance with large datasets, use `pack_flat()` which returns NumPy arrays directly:

```python
items_flat, bin_offsets = packer.pack_flat(lengths)
bins = np.split(items_flat, bin_offsets)
```
