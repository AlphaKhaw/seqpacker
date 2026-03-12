"""
Runner wrapping the LightBinPack library (C++ core with Python bindings).

LightBinPack is specifically designed for LLM training sequence packing.
It requires OpenMP support, so it may not build on macOS with Apple clang.

Install: pip install lightbinpack
    - Requires Linux or a compiler with OpenMP support
    - On macOS: install GCC via `brew install gcc` and build from source
"""

from typing import Callable

from benchmarks.python.utils.logging import logger

try:
    from lightbinpack import bfd, ffd, obfd, obfdp

    LIGHTBINPACK_AVAILABLE = True
    LIGHTBINPACK_STRATEGIES: dict[str, Callable] = {
        "ffd": ffd,
        "bfd": bfd,
        "obfd": obfd,
        "obfdp": obfdp,
    }
except ImportError:
    LIGHTBINPACK_AVAILABLE = False
    LIGHTBINPACK_STRATEGIES = {}
    logger.warning(
        "lightbinpack not installed. LightBinPackRunner will be skipped. "
        "Install with: pip install lightbinpack "
        "(requires Linux or a compiler with OpenMP support; "
        "Apple clang on macOS does not support -fopenmp by default)"
    )


class LightBinPackRunner:
    """
    Runner using LightBinPack's packing algorithms.

    LightBinPack is a C++ library with Python bindings, specifically
    designed for solving packing problems in LLM training.

    Supports strategies: nf, ffd, bfd, obfd, obfdp.
    """

    def __init__(self, strategy: str = "ffd") -> None:
        """
        Initialize the runner with a packing strategy.

        Args:
            strategy (str): Algorithm short name (default: "ffd").
                Supported: "ffd", "bfd", "obfd", "obfdp".
        """
        if LIGHTBINPACK_AVAILABLE and strategy not in LIGHTBINPACK_STRATEGIES:
            raise ValueError(
                f"Unknown LightBinPack strategy '{strategy}'. "
                f"Supported: {list(LIGHTBINPACK_STRATEGIES.keys())}"
            )
        self.strategy = strategy
        self._name = "LightBinPack" if strategy == "ffd" else f"LightBinPack_{strategy}"

    @property
    def name(self) -> str:
        """
        Return the runner name.

        Returns:
            str: Runner name.
        """
        return self._name

    @property
    def available(self) -> bool:
        """
        Check if lightbinpack is installed.

        Returns:
            bool: True if available.
        """
        return LIGHTBINPACK_AVAILABLE

    def pack(self, lengths: list[int], max_seq_len: int) -> list[list[int]]:
        """
        Pack sequences using the selected LightBinPack algorithm.

        Args:
            lengths (list[int]): Sequence lengths to pack.
            max_seq_len (int): Maximum bin capacity.

        Returns:
            list[list[int]]: Packed bins.

        Raises:
            RuntimeError: If lightbinpack is not installed.
        """
        if not LIGHTBINPACK_AVAILABLE:
            raise RuntimeError(
                "lightbinpack is not installed. "
                "Install with: pip install lightbinpack "
                "(requires Linux or a compiler with OpenMP support)"
            )

        filtered = [length for length in lengths if length <= max_seq_len]
        if not filtered:
            return []

        pack_fn = LIGHTBINPACK_STRATEGIES[self.strategy]
        index_bins = pack_fn(filtered, max_seq_len)
        return [[filtered[i] for i in bin_indices] for bin_indices in index_bins]
