# Training Integration

SeqPacker provides framework-specific utilities for converting packing results into training-ready formats. Both modules are optional -- import only what you need.

## Which integration?

| Workflow | Module | Dependency | Import |
|----------|--------|------------|--------|
| HF Trainer / SFTTrainer / TRL | `hf_utils` | `datasets` | `from seqpacker.hf_utils import pack_dataset` |
| Custom PyTorch training loop | `torch_utils` | `torch` | `from seqpacker.torch_utils import packed_collate_fn` |
| PyTorch + Flash Attention | `torch_utils` | `torch` | `from seqpacker.torch_utils import pack_result_to_tensors` |
| Framework-agnostic | core API | none | `from seqpacker import pack_sequences` |

---

## HuggingFace Trainer

`seqpacker.hf_utils` converts packing results directly into a HuggingFace `datasets.Dataset` with correctly computed `input_ids`, `attention_mask`, `labels` (shifted with boundary masking), and `position_ids` (per-sequence reset).

### One-call packing

```python
from seqpacker.hf_utils import pack_dataset

tokenized = tokenizer(texts, truncation=True, max_length=2048)
ds = pack_dataset(
    token_ids=tokenized["input_ids"],
    capacity=2048,
    padding_value=tokenizer.pad_token_id or tokenizer.eos_token_id,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=ds,
    args=SFTConfig(..., dataset_text_field=None),
    tokenizer=tokenizer,
)
trainer.train()
```

`pack_dataset` handles the full pipeline: packing sequences into bins, concatenating token IDs, padding, and generating all metadata columns. See [examples/sft_trainer.py](https://github.com/AlphaKhaw/seqpacker/blob/main/examples/sft_trainer.py) for a complete fine-tuning script.

### Pre-computed packing

If you want to inspect metrics before building the dataset:

```python
from seqpacker import pack_sequences
from seqpacker.hf_utils import pack_dataset_from_result

lengths = [len(ids) for ids in tokenized["input_ids"]]
result = pack_sequences(lengths=lengths, capacity=2048)
print(f"Efficiency: {result.efficiency:.2%}")
print(f"Packed {len(lengths)} sequences into {result.num_bins} bins")

ds = pack_dataset_from_result(
    result=result,
    token_ids=tokenized["input_ids"],
    padding_value=tokenizer.pad_token_id or tokenizer.eos_token_id,
)
```

### Dataset columns

The returned `datasets.Dataset` contains these columns (each row is one packed bin):

| Column | Type | Description |
|--------|------|-------------|
| `input_ids` | `list[int]` | Padded token IDs, length = capacity |
| `attention_mask` | `list[int]` | 1 for real tokens, 0 for padding |
| `labels` | `list[int]` | Shifted labels with -100 at sequence boundaries and padding |
| `position_ids` | `list[int]` | Per-sequence reset position IDs |

### Parameters

#### `pack_dataset`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `token_ids` | `Sequence[Sequence[int]]` | required | Token ID sequences to pack |
| `capacity` | `int` | required | Maximum bin capacity in tokens |
| `strategy` | `str` | `"obfd"` | Packing algorithm short name |
| `seed` | `int \| None` | `None` | Random seed for shuffle-based algorithms |
| `padding_value` | `int` | `0` | Value for padding positions in `input_ids` |
| `label_padding_value` | `int` | `-100` | Value for masked label positions |

#### `pack_dataset_from_result`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `PackResult` | required | Packing result from `pack_sequences` or `Packer.pack` |
| `token_ids` | `Sequence[Sequence[int]]` | required | Token ID sequences indexed by original sequence ID |
| `padding_value` | `int` | `0` | Value for padding positions in `input_ids` |
| `label_padding_value` | `int` | `-100` | Value for masked label positions |

---

## PyTorch DataLoader

`seqpacker.torch_utils` provides helpers for converting pack results into GPU-ready tensors for packed LLM training.

### Collate function

The simplest way to integrate with a custom PyTorch training loop:

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

See [examples/pytorch_training.py](https://github.com/AlphaKhaw/seqpacker/blob/main/examples/pytorch_training.py) for a complete training loop.

### Manual conversion

For more control, convert a `PackResult` directly:

```python
from seqpacker import pack_sequences
from seqpacker.torch_utils import pack_result_to_tensors

result = pack_sequences(lengths, capacity=2048)
batch = pack_result_to_tensors(result=result, token_ids=token_ids)
```

### PackedBatch

The `PackedBatch` dataclass contains all tensors needed for packed training:

| Field | Shape | Description |
|-------|-------|-------------|
| `input_ids` | `(num_packs, capacity)` | Padded token IDs |
| `cu_seqlens` | list of `(num_seqs + 1,)` | Cumulative sequence lengths (Flash Attention varlen API) |
| `max_seqlen` | `int` | Maximum individual sequence length |
| `position_ids` | `(num_packs, capacity)` | Per-sequence reset position IDs |
| `labels` | `(num_packs, capacity)` | Shifted labels with -100 at boundaries/padding |
| `attention_mask` | `(num_packs, capacity)` | Binary mask (1=real, 0=padding) |

### Parameters

#### `pack_result_to_tensors`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `PackResult` | required | Packing result from `pack_sequences` or `Packer.pack` |
| `token_ids` | `Sequence[Sequence[int]]` | required | Token ID sequences indexed by original sequence ID |
| `padding_value` | `int` | `0` | Value for padding positions in `input_ids` |
| `label_padding_value` | `int` | `-100` | Value for masked label positions |
| `device` | `torch.device \| str \| None` | `None` | Device for output tensors |

#### `packed_collate_fn`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `capacity` | `int` | required | Maximum bin capacity in tokens |
| `strategy` | `str` | `"obfd"` | Packing algorithm short name |
| `seed` | `int \| None` | `None` | Random seed for shuffle-based algorithms |
| `padding_value` | `int` | `0` | Value for padding positions |
| `label_padding_value` | `int` | `-100` | Value for masked label positions |

---

## Flash Attention

Use `cu_seqlens` from `PackedBatch` with Flash Attention's variable-length API:

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

This is only available via `torch_utils` since Flash Attention requires raw tensors. The `cu_seqlens` tensor tells Flash Attention where each sequence starts and ends within the packed input, preventing cross-sequence attention.
