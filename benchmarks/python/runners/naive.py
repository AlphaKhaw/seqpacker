"""
Naive padding runner: one sequence per bin (worst-case baseline).
"""


class NaivePaddingRunner:
    """
    Baseline runner that places each sequence in its own bin.
    """

    @property
    def name(self) -> str:
        """
        Return the runner name.

        Returns:
            str: Runner name.
        """
        return "naive_padding"

    def pack(self, lengths: list[int], max_seq_len: int) -> list[list[int]]:
        """
        Pack each sequence into its own bin.

        Args:
            lengths (list[int]): Sequence lengths to pack.
            max_seq_len (int): Maximum bin capacity.

        Returns:
            list[list[int]]: One bin per sequence.
        """
        return [[length] for length in lengths if length <= max_seq_len]
