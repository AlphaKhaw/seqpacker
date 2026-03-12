# PyTorch Integration

`seqpacker.torch_utils` provides helpers for converting pack results into GPU-ready tensors for packed LLM training. Torch is **not** a dependency — import only when you need it.

## DataLoader Collate Function

The simplest way to integrate with PyTorch training:

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

The collate function:

1. Extracts lengths from each sample
2. Packs them using the specified algorithm
3. Returns a `PackedBatch` with all tensors ready for training

## Manual Conversion

For more control, convert a `PackResult` directly:

```python
from seqpacker import pack_sequences
from seqpacker.torch_utils import pack_result_to_tensors

result = pack_sequences(lengths, capacity=2048)
batch = pack_result_to_tensors(result=result, token_ids=token_ids)
```

## PackedBatch

The `PackedBatch` dataclass contains all tensors needed for packed training:

| Field | Shape | Description |
|-------|-------|-------------|
| `input_ids` | `(num_packs, capacity)` | Padded token IDs |
| `cu_seqlens` | list of `(num_seqs + 1,)` | Cumulative sequence lengths (Flash Attention varlen API) |
| `max_seqlen` | `int` | Maximum individual sequence length |
| `position_ids` | `(num_packs, capacity)` | Per-sequence reset position IDs |
| `labels` | `(num_packs, capacity)` | Shifted labels with -100 at boundaries/padding |
| `attention_mask` | `(num_packs, capacity)` | Binary mask (1=real, 0=padding) |

## Flash Attention Integration

Use `cu_seqlens` with Flash Attention's variable-length API:

```python
from flash_attn import flash_attn_varlen_func

for i, pack_cu in enumerate(batch.cu_seqlens):
    # pack_cu = [0, len_0, len_0+len_1, ..., total]
    output = flash_attn_varlen_func(
        q=q[i],
        k=k[i],
        v=v[i],
        cu_seqlens_q=pack_cu,
        cu_seqlens_k=pack_cu,
        max_seqlen_q=batch.max_seqlen,
        max_seqlen_k=batch.max_seqlen,
    )
```

## Parameters

### `pack_result_to_tensors`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `PackResult` | required | Packing result from `pack_sequences` or `Packer.pack` |
| `token_ids` | `Sequence[Sequence[int]]` | required | Token ID sequences indexed by original sequence ID |
| `padding_value` | `int` | `0` | Value for padding positions in `input_ids` |
| `label_padding_value` | `int` | `-100` | Value for masked label positions |
| `device` | `torch.device \| str \| None` | `None` | Device for output tensors |

### `packed_collate_fn`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `capacity` | `int` | required | Maximum bin capacity in tokens |
| `strategy` | `str` | `"obfd"` | Packing algorithm short name |
| `seed` | `int \| None` | `None` | Random seed for shuffle-based algorithms |
| `padding_value` | `int` | `0` | Value for padding positions |
| `label_padding_value` | `int` | `-100` | Value for masked label positions |
