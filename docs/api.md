# API Reference

Complete Python API documentation for SeqPacker.

For Rust API docs, see [docs.rs/seqpacker](https://docs.rs/seqpacker).

---

## `pack_sequences`

```python
def pack_sequences(
    lengths: list[int] | NDArray[int64],
    capacity: int,
    strategy: str = "obfd",
    seed: int | None = None,
) -> PackResult
```

Pack sequence lengths into bins (convenience function).

One-shot packing without constructing a `Packer`. Equivalent to `Packer(capacity, strategy, seed).pack(lengths)`.

**Args:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lengths` | `list[int] \| NDArray[int64]` | required | Sequence lengths to pack |
| `capacity` | `int` | required | Maximum bin capacity in tokens |
| `strategy` | `str` | `"obfd"` | Algorithm short name |
| `seed` | `int \| None` | `None` | Random seed for shuffle-based algorithms |

**Returns:** [`PackResult`](#packresult)

**Raises:**

- `ValueError` — if strategy is unknown, any length exceeds capacity, or input is empty
- `TypeError` — if lengths is not a list or NumPy int64 array

---

## `Packer`

```python
class Packer(capacity: int, strategy: str = "obfd", seed: int | None = None)
```

Main packing interface. Create a packer with a capacity and strategy, then call `pack()` or `pack_flat()` with sequence lengths.

```python
packer = Packer(capacity=2048, strategy="obfd")
result = packer.pack([500, 600, 400, 1000])
```

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `capacity` | `int` | required | Maximum bin capacity in tokens |
| `strategy` | `str` | `"obfd"` | Algorithm short name. Use `Packer.strategies()` to list all |
| `seed` | `int \| None` | `None` | Random seed for shuffle-based algorithms (e.g. `"ffs"`) |

### Methods

#### `pack(lengths) -> PackResult`

Pack sequence lengths into bins. Accepts a Python list or NumPy int64 array.

```python
result = packer.pack([500, 600, 400, 1000])
```

**Raises:** `ValueError` if any length exceeds capacity or input is empty. `TypeError` if lengths is not a list or NumPy array.

#### `pack_flat(lengths) -> tuple[NDArray, NDArray]`

Pack into flat NumPy arrays for maximum performance.

Returns `(items_flat, bin_offsets)` as int64 arrays. Reconstruct bins with `np.split(items_flat, bin_offsets)`.

```python
items_flat, bin_offsets = packer.pack_flat(lengths)
bins = np.split(items_flat, bin_offsets)
```

#### `strategies() -> list[tuple[str, str]]` *(static method)*

Return all available strategy names as `(short_name, full_name)` pairs.

```python
Packer.strategies()
# [('nf', 'NF'), ('ff', 'FF'), ..., ('hk', 'HK')]
```

---

## `PackResult`

```python
class PackResult
```

Combined packing result (read-only). Contains the list of packed bins and associated metrics.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `packs` | `list[Pack]` | List of `Pack` objects (one per bin) |
| `metrics` | `PackMetrics` | Packing quality metrics |
| `num_bins` | `int` | Number of bins used |
| `efficiency` | `float` | Packing efficiency (0.0–1.0). Shortcut for `metrics.efficiency` |
| `time_ms` | `float` | Packing time in milliseconds. Shortcut for `metrics.packing_time_ms` |
| `bins` | `list[list[int]]` | Bins as nested list of sequence IDs: `[[0, 3], [1, 2], ...]` |

---

## `Pack`

```python
class Pack
```

A single packed bin (read-only). Contains the sequence IDs and lengths assigned to this bin.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `sequence_ids` | `list[int]` | Sequence IDs in this bin (original input order) |
| `lengths` | `list[int]` | Sequence lengths in this bin |
| `used` | `int` | Total tokens used in this bin |

`len(pack)` returns the number of sequences in the bin.

---

## `PackMetrics`

```python
class PackMetrics
```

Packing quality metrics (read-only).

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `num_sequences` | `int` | Total number of input sequences |
| `total_tokens` | `int` | Total tokens across all input sequences |
| `num_packs` | `int` | Number of bins produced |
| `padding_tokens` | `int` | Total padding tokens added across all bins |
| `efficiency` | `float` | Packing efficiency (0.0–1.0). Higher is better |
| `padding_ratio` | `float` | Ratio of padding tokens to total capacity. Lower is better |
| `avg_utilisation` | `float` | Mean bin utilisation across all bins (0.0–1.0) |
| `utilisation_std` | `float` | Standard deviation of bin utilisation |
| `min_utilisation` | `float` | Minimum bin utilisation |
| `max_utilisation` | `float` | Maximum bin utilisation |
| `avg_sequences_per_pack` | `float` | Mean number of sequences per bin |
| `packing_time_ms` | `float` | Packing time in milliseconds |
| `throughput` | `float` | Sequences packed per millisecond |

---

## `StreamPacker`

```python
class StreamPacker(capacity: int, strategy: str = "nf", k: int | None = None)
```

Streaming packer for incremental sequence packing. Only bounded-space online algorithms are supported: `"nf"` (NextFit) and `"hk"` (Harmonic-K).

```python
sp = StreamPacker(capacity=2048, strategy="nf")
for length in dataset_lengths:
    for pack in sp.add(length):
        process(pack)
for pack in sp.finish():
    process(pack)
```

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `capacity` | `int` | required | Maximum bin capacity in tokens |
| `strategy` | `str` | `"nf"` | Algorithm: `"nf"` (NextFit) or `"hk"` (Harmonic-K) |
| `k` | `int \| None` | `None` | Number of size classes for Harmonic-K (default: 10) |

**Raises:** `ValueError` if strategy is not `"nf"` or `"hk"`.

### Methods

#### `add(length) -> list[Pack]`

Add a sequence length and return any completed packs. May return an empty list if no packs are complete yet.

**Raises:** `ValueError` if length exceeds capacity or packer is already finished.

#### `finish() -> list[Pack]`

Flush all remaining open bins and return them as packs. After calling `finish()`, the packer cannot be used again.

**Raises:** `ValueError` if packer is already finished.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `sequences_added` | `int` | Number of sequences added so far |

---

## `hf_utils`

Optional module for HuggingFace Trainer integration. Requires `datasets` to be installed.

```python
from seqpacker.hf_utils import pack_dataset, pack_dataset_from_result
```

See [Training Integration](user-guide/training.md#huggingface-trainer) for full documentation.

### `pack_dataset`

One-call convenience that packs token sequences and returns a `datasets.Dataset` with `input_ids`, `attention_mask`, `labels`, and `position_ids`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `token_ids` | `Sequence[Sequence[int]]` | required | Token ID sequences to pack |
| `capacity` | `int` | required | Maximum bin capacity in tokens |
| `strategy` | `str` | `"obfd"` | Packing algorithm short name |
| `seed` | `int \| None` | `None` | Random seed for shuffle-based algorithms |
| `padding_value` | `int` | `0` | Value for padding positions in `input_ids` |
| `label_padding_value` | `int` | `-100` | Value for masked label positions |

### `pack_dataset_from_result`

Converts a pre-computed `PackResult` into a `datasets.Dataset`. Use when you want to inspect metrics before building the dataset.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `PackResult` | required | Packing result from `pack_sequences` or `Packer.pack` |
| `token_ids` | `Sequence[Sequence[int]]` | required | Token ID sequences indexed by original sequence ID |
| `padding_value` | `int` | `0` | Value for padding positions in `input_ids` |
| `label_padding_value` | `int` | `-100` | Value for masked label positions |

---

## `torch_utils`

Optional module for PyTorch integration. Requires `torch` to be installed.

```python
from seqpacker.torch_utils import PackedBatch, pack_result_to_tensors, packed_collate_fn
```

See [Training Integration](user-guide/training.md#pytorch-dataloader) for full documentation.

### `PackedBatch`

Dataclass containing GPU-ready tensors:

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `input_ids` | `Tensor` | `(num_packs, capacity)` | Padded token IDs |
| `cu_seqlens` | `list[Tensor]` | `(num_seqs + 1,)` each | Cumulative sequence lengths |
| `max_seqlen` | `int` | — | Maximum individual sequence length |
| `position_ids` | `Tensor` | `(num_packs, capacity)` | Per-sequence reset position IDs |
| `labels` | `Tensor \| None` | `(num_packs, capacity)` | Shifted labels (-100 at boundaries) |
| `attention_mask` | `Tensor \| None` | `(num_packs, capacity)` | Binary mask (1=real, 0=padding) |
