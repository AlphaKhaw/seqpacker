# NumPy Support

SeqPacker supports both Python lists and NumPy arrays as input, with zero-copy for NumPy int64 arrays.

## Basic Usage

```python
import numpy as np
from seqpacker import Packer

packer = Packer(capacity=2048)

# NumPy array input (zero-copy)
lengths = np.array([500, 600, 400, 1000], dtype=np.int64)
result = packer.pack(lengths)
```

## Flat Output

For maximum performance with large datasets, `pack_flat()` returns NumPy arrays directly instead of Python objects:

```python
items_flat, bin_offsets = packer.pack_flat(lengths)
bins = np.split(items_flat, bin_offsets)
```

Both `items_flat` and `bin_offsets` are `np.ndarray[np.int64]`:

- `items_flat`: all item indices concatenated
- `bin_offsets`: split points between bins

This avoids Python object overhead and is the fastest way to access packing results programmatically.

## Supported dtypes

The `pack()` method accepts:

- `list[int]` — Python list of integers
- `np.ndarray[np.int64]` — NumPy int64 array (zero-copy)

Other NumPy dtypes will raise `TypeError`. Convert explicitly if needed:

```python
lengths_f32 = np.array([500.0, 600.0], dtype=np.float32)
result = packer.pack(lengths_f32.astype(np.int64))
```
