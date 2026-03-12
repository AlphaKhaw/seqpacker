# Getting Started

## Installation

=== "Python (pip)"

    ```bash
    pip install seqpacker
    ```

=== "Python (uv)"

    ```bash
    uv add seqpacker
    ```

=== "Rust"

    ```bash
    cargo add seqpacker
    ```

## Quick Start

### Pack sequences in 3 lines

```python
from seqpacker import pack_sequences

result = pack_sequences([1000, 800, 600, 500, 400, 300, 200, 100], capacity=1024)
print(result.bins)        # [[0], [1, 7], [2, 4], [3, 5, 6]]
print(result.efficiency)  # 0.952...
```

### Using the Packer class

For more control, use the `Packer` class directly:

```python
from seqpacker import Packer

packer = Packer(capacity=2048, strategy="obfd")
result = packer.pack(sequence_lengths)

for pack in result.packs:
    print(pack.sequence_ids, pack.lengths, pack.used)

print(f"Efficiency: {result.efficiency:.2%}")
print(f"Packs: {result.num_bins}")
```

### Choosing an algorithm

SeqPacker ships 11 algorithms. The default (`obfd`) is recommended for most use cases:

```python
from seqpacker import Packer

# Use any algorithm by short name
packer = Packer(capacity=2048, strategy="obfd")

# List all available strategies
print(Packer.strategies())
# [('nf', 'NF'), ('ff', 'FF'), ('bf', 'BF'), ..., ('hk', 'HK')]
```

See the [Algorithms](user-guide/algorithms.md) page for a full comparison.
