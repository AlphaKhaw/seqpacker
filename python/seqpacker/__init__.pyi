"""
Type stubs for seqpacker._core.

These stubs provide static type information for IDE autocompletion,
pyright/mypy type checking, and documentation generation. They describe
the Python-visible API of the Rust extension module — the actual runtime
behavior is implemented in Rust via PyO3.
"""

import numpy as np
import numpy.typing as npt

__version__: str

class PackMetrics:
    """
    Packing quality metrics (read-only).

    All fields are exposed as read-only properties. Includes both
    direct metrics (efficiency, num_packs) and computed metrics
    (padding_ratio, throughput).
    """

    @property
    def num_sequences(self) -> int:
        """Total number of input sequences."""
        ...
    @property
    def total_tokens(self) -> int:
        """Total tokens across all input sequences."""
        ...
    @property
    def num_packs(self) -> int:
        """Number of bins (packs) produced."""
        ...
    @property
    def padding_tokens(self) -> int:
        """Total padding tokens added across all bins."""
        ...
    @property
    def efficiency(self) -> float:
        """Packing efficiency as a ratio (0.0–1.0). Higher is better."""
        ...
    @property
    def avg_utilisation(self) -> float:
        """Mean bin utilisation across all bins (0.0–1.0)."""
        ...
    @property
    def utilisation_std(self) -> float:
        """Standard deviation of bin utilisation."""
        ...
    @property
    def min_utilisation(self) -> float:
        """Minimum bin utilisation."""
        ...
    @property
    def max_utilisation(self) -> float:
        """Maximum bin utilisation."""
        ...
    @property
    def avg_sequences_per_pack(self) -> float:
        """Mean number of sequences per bin."""
        ...
    @property
    def packing_time_ms(self) -> float:
        """Packing time in milliseconds."""
        ...
    @property
    def padding_ratio(self) -> float:
        """Ratio of padding tokens to total capacity. Lower is better."""
        ...
    @property
    def throughput(self) -> float:
        """Sequences packed per millisecond."""
        ...

class Pack:
    """
    A single packed bin (read-only).

    Contains the sequence IDs and lengths assigned to this bin.
    """

    @property
    def sequence_ids(self) -> list[int]:
        """Sequence IDs in this bin (original input order)."""
        ...
    @property
    def lengths(self) -> list[int]:
        """Sequence lengths in this bin."""
        ...
    @property
    def used(self) -> int:
        """Total tokens used in this bin."""
        ...
    def __len__(self) -> int:
        """Number of sequences in this bin."""
        ...
    def __repr__(self) -> str: ...

class PackResult:
    """
    Combined packing result (read-only).

    Contains the list of packed bins and associated metrics.
    Convenience properties like ``efficiency`` and ``bins`` delegate
    to the inner Pack and PackMetrics objects.
    """

    @property
    def packs(self) -> list[Pack]:
        """List of Pack objects (one per bin)."""
        ...
    @property
    def metrics(self) -> PackMetrics:
        """Packing quality metrics."""
        ...
    @property
    def num_bins(self) -> int:
        """Number of bins used."""
        ...
    @property
    def efficiency(self) -> float:
        """Packing efficiency (0.0–1.0). Shortcut for ``metrics.efficiency``."""
        ...
    @property
    def time_ms(self) -> float:
        """Packing time in milliseconds. Shortcut for ``metrics.packing_time_ms``."""
        ...
    @property
    def bins(self) -> list[list[int]]:
        """Bins as nested list of sequence IDs: ``[[0, 3], [1, 2], ...]``."""
        ...
    def __len__(self) -> int:
        """Number of bins."""
        ...
    def __repr__(self) -> str: ...

class Packer:
    """
    Main packing interface.

    Create a packer with a capacity and strategy, then call ``pack()``
    or ``pack_flat()`` with sequence lengths.

    Example::

        packer = Packer(capacity=2048, strategy="obfd")
        result = packer.pack([500, 600, 400, 1000])
        print(result.bins)  # [[0, 3], [1, 2]]
        print(result.efficiency)  # 0.976...
    """

    def __init__(
        self,
        capacity: int,
        strategy: str = "obfd",
        seed: int | None = None,
    ) -> None:
        """
        Create a new packer.

        Args:
            capacity: Maximum bin capacity in tokens.
            strategy: Algorithm short name. Use ``Packer.strategies()`` to list all.
            seed: Random seed for shuffle-based algorithms (e.g. "ffs").
        """
        ...
    def pack(self, lengths: list[int] | npt.NDArray[np.int64]) -> PackResult:
        """
        Pack sequence lengths into bins.

        Accepts a Python list or NumPy int64 array. Returns a full
        PackResult with per-bin details and metrics.

        Args:
            lengths: Sequence lengths to pack.

        Returns:
            Packing result with bins, metrics, and efficiency.

        Raises:
            TypeError: If lengths is not a list or NumPy array.
            ValueError: If any length exceeds capacity or input is empty.
        """
        ...
    def pack_flat(
        self,
        lengths: list[int] | npt.NDArray[np.int64],
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """
        Pack into flat NumPy arrays for maximum performance.

        Returns ``(items_flat, bin_offsets)`` as int64 arrays.
        Reconstruct bins with ``np.split(items_flat, bin_offsets)``.

        Args:
            lengths: Sequence lengths to pack.

        Returns:
            Tuple of (item_ids_flat, bin_offsets) as int64 NumPy arrays.

        Raises:
            TypeError: If lengths is not a list or NumPy array.
            ValueError: If any length exceeds capacity or input is empty.
        """
        ...
    @staticmethod
    def strategies() -> list[tuple[str, str]]:
        """
        Return all available strategy names.

        Returns:
            List of (short_name, full_name) pairs, e.g. ``[("obfd", "OBFD"), ...]``.
        """
        ...
    def __repr__(self) -> str: ...

class StreamPacker:
    """
    Streaming packer for incremental sequence packing.

    Only bounded-space online algorithms are supported:
    ``"nf"`` (NextFit) and ``"hk"`` (Harmonic-K).

    Example::

        sp = StreamPacker(capacity=2048, strategy="nf")
        for length in dataset_lengths:
            for pack in sp.add(length):
                process(pack)
        for pack in sp.finish():
            process(pack)
    """

    def __init__(
        self,
        capacity: int,
        strategy: str = "nf",
        k: int | None = None,
    ) -> None:
        """
        Create a new streaming packer.

        Args:
            capacity: Maximum bin capacity in tokens.
            strategy: Algorithm: ``"nf"`` (NextFit) or ``"hk"`` (Harmonic-K).
            k: Number of size classes for Harmonic-K (default: 10).

        Raises:
            ValueError: If strategy is not ``"nf"`` or ``"hk"``.
        """
        ...
    def add(self, length: int) -> list[Pack]:
        """
        Add a sequence length and return any completed packs.

        Args:
            length: Sequence length to add.

        Returns:
            Packs that are now complete (may be empty).

        Raises:
            ValueError: If length exceeds capacity or packer is already finished.
        """
        ...
    def finish(self) -> list[Pack]:
        """
        Flush all remaining open bins and return them as packs.

        After calling ``finish()``, the packer cannot be used again.

        Returns:
            All remaining packs.

        Raises:
            ValueError: If packer is already finished.
        """
        ...
    @property
    def sequences_added(self) -> int:
        """Number of sequences added so far."""
        ...
    def __repr__(self) -> str: ...

def pack_sequences(
    lengths: list[int] | npt.NDArray[np.int64],
    capacity: int,
    strategy: str = "obfd",
    seed: int | None = None,
) -> PackResult:
    """
    Pack sequence lengths into bins (convenience function).

    One-shot packing without constructing a Packer. Equivalent to::

        Packer(capacity, strategy, seed).pack(lengths)

    Args:
        lengths: Sequence lengths to pack.
        capacity: Maximum bin capacity in tokens.
        strategy: Algorithm short name (default: "obfd").
        seed: Random seed for shuffle-based algorithms.

    Returns:
        Packing result with bins, metrics, and efficiency.

    Raises:
        ValueError: If strategy is unknown, any length exceeds capacity, or input is
                    empty.
    """
    ...
