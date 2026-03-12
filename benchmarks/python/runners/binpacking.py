"""
Runner wrapping the binpacking library.
"""

import binpacking


class BinpackingRunner:
    """
    Runner using the binpacking library's greedy algorithm.
    """

    @property
    def name(self) -> str:
        """
        Return the runner name.

        Returns:
            str: Runner name.
        """
        return "binpacking"

    def pack(self, lengths: list[int], max_seq_len: int) -> list[list[int]]:
        """
        Pack sequences using binpacking library.

        The binpacking library works with dict items, so we create
        indexed items and convert back to length lists.

        Args:
            lengths (list[int]): Sequence lengths to pack.
            max_seq_len (int): Maximum bin capacity.

        Returns:
            list[list[int]]: Packed bins.
        """
        filtered = {
            i: length for i, length in enumerate(lengths) if length <= max_seq_len
        }
        if not filtered:
            return []
        result = binpacking.to_constant_volume(d=filtered, V_max=max_seq_len)  # type: ignore[attr-defined]
        return [list(bin_dict.values()) for bin_dict in result if bin_dict]  # type: ignore[union-attr]
