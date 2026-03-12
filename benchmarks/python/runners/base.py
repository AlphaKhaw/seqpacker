"""
Base protocol for packing runners.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class PackingRunner(Protocol):
    """
    Protocol for all packing algorithm runners.
    """

    @property
    def name(self) -> str:
        """
        Return the runner name.

        Returns:
            str: Human-readable name for this runner.
        """
        ...

    def pack(self, lengths: list[int], max_seq_len: int) -> list[list[int]]:
        """
        Pack sequences into bins.

        Args:
            lengths (list[int]): Sequence lengths to pack.
            max_seq_len (int): Maximum bin capacity.

        Returns:
            list[list[int]]: List of bins, each bin is a list of sequence lengths.
        """
        ...
