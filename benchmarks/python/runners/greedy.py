"""
Pure Python First-Fit-Decreasing packing runner.
"""


class GreedyFFDRunner:
    """
    First-Fit-Decreasing bin packing in pure Python.
    """

    @property
    def name(self) -> str:
        """
        Return the runner name.

        Returns:
            str: Runner name.
        """
        return "greedy_ffd"

    def pack(self, lengths: list[int], max_seq_len: int) -> list[list[int]]:
        """
        Pack sequences using First-Fit-Decreasing.

        Sorts sequences by length descending, then places each into the first
        bin that has enough remaining capacity.

        Args:
            lengths (list[int]): Sequence lengths to pack.
            max_seq_len (int): Maximum bin capacity.

        Returns:
            list[list[int]]: Packed bins.
        """
        sorted_lengths = sorted(
            [length for length in lengths if length <= max_seq_len],
            reverse=True,
        )

        bins: list[list[int]] = []
        bin_remaining: list[int] = []

        for length in sorted_lengths:
            placed = False
            for i, remaining in enumerate(bin_remaining):
                if remaining >= length:
                    bins[i].append(length)
                    bin_remaining[i] -= length
                    placed = True
                    break
            if not placed:
                bins.append([length])
                bin_remaining.append(max_seq_len - length)

        return bins
