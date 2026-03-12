"""
seqpacker - High-performance sequence packing for LLM training.

Quick start:
    >>> from seqpacker import pack_sequences
    >>> result = pack_sequences([500, 600, 400, 1000], capacity=2048)
    >>> print(result.bins)
    [[0, 3], [1, 2]]
"""

from seqpacker._core import (
    Pack,
    Packer,
    PackMetrics,
    PackResult,
    StreamPacker,
    __version__,
    pack_sequences,
)

__all__ = [
    "__version__",
    "Pack",
    "PackMetrics",
    "PackResult",
    "Packer",
    "StreamPacker",
    "pack_sequences",
]
