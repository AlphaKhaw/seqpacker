"""
Runner wrapping the prtpy library's bin packing algorithms.
"""

import prtpy


class PrtpyFFDRunner:
    """
    Runner using prtpy's First-Fit-Decreasing algorithm.
    """

    @property
    def name(self) -> str:
        """
        Return the runner name.

        Returns:
            str: Runner name.
        """
        return "prtpy_ffd"

    def pack(self, lengths: list[int], max_seq_len: int) -> list[list[int]]:
        """
        Pack sequences using prtpy's First-Fit-Decreasing.

        Args:
            lengths (list[int]): Sequence lengths to pack.
            max_seq_len (int): Maximum bin capacity.

        Returns:
            list[list[int]]: Packed bins.
        """
        filtered = [length for length in lengths if length <= max_seq_len]
        if not filtered:
            return []
        bins = prtpy.pack(
            algorithm=prtpy.packing.first_fit_decreasing,  # type: ignore[attr-defined]
            binsize=max_seq_len,
            items=filtered,
        )
        return [b for b in bins if b]


class PrtpyFirstFitRunner:
    """
    Runner using prtpy's First-Fit algorithm.
    """

    @property
    def name(self) -> str:
        """
        Return the runner name.

        Returns:
            str: Runner name.
        """
        return "prtpy_ff"

    def pack(self, lengths: list[int], max_seq_len: int) -> list[list[int]]:
        """
        Pack sequences using prtpy's First-Fit.

        Args:
            lengths (list[int]): Sequence lengths to pack.
            max_seq_len (int): Maximum bin capacity.

        Returns:
            list[list[int]]: Packed bins.
        """
        filtered = [length for length in lengths if length <= max_seq_len]
        if not filtered:
            return []
        bins = prtpy.pack(
            algorithm=prtpy.packing.first_fit,  # type: ignore[attr-defined]
            binsize=max_seq_len,
            items=filtered,
        )
        return [b for b in bins if b]
