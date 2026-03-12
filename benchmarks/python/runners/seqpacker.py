"""
Benchmark runner for seqpacker Rust-backed packing.
"""

from seqpacker import Packer


class SeqpackerRunner:
    """
    Benchmark runner wrapping the seqpacker Rust library.

    Follows the PackingRunner protocol: pack(lengths, max_seq_len) returns
    bins as lists of sequence lengths.
    """

    def __init__(self, strategy: str = "obfd") -> None:
        """
        Initialize the runner with a packing strategy.

        Args:
            strategy (str): Algorithm short name (default: "obfd").
        """
        self.strategy = strategy
        self._name = f"seqpacker_{strategy}"

    @property
    def name(self) -> str:
        """
        Return the runner name.

        Returns:
            str: Runner name.
        """
        return self._name

    def pack(self, lengths: list[int], max_seq_len: int) -> list[list[int]]:
        """
        Pack sequences into bins.

        Args:
            lengths (list[int]): Sequence lengths to pack.
            max_seq_len (int): Maximum bin capacity.

        Returns:
            list[list[int]]: List of bins, each bin is a list of sequence lengths.
        """
        packer = Packer(capacity=max_seq_len, strategy=self.strategy)
        result = packer.pack(lengths)
        # Convert from sequence IDs to lengths
        return [[lengths[id_] for id_ in bin_] for bin_ in result.bins]
