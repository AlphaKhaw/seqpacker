"""
Deterministic synthetic dataset generators for benchmarking.
"""

import numpy as np

from benchmarks.python.metrics.models import DatasetInfo


class SyntheticDataGenerator:
    """
    Generator for synthetic sequence length distributions.

    All methods are static and deterministic (seeded). Each returns a
    DatasetInfo with the generated lengths and computed statistics.
    """

    @staticmethod
    def uniform(
        n: int,
        min_len: int = 16,
        max_len: int = 512,
        seed: int = 42,
    ) -> DatasetInfo:
        """
        Generate uniformly distributed sequence lengths.

        Args:
            n (int): Number of sequences to generate.
            min_len (int): Minimum sequence length.
            max_len (int): Maximum sequence length.
            seed (int): Random seed for reproducibility.

        Returns:
            DatasetInfo: Dataset with uniform length distribution.
        """
        rng = np.random.default_rng(seed=seed)
        lengths = rng.integers(low=min_len, high=max_len + 1, size=n).tolist()
        return DatasetInfo(
            name=f"uniform_{n}_{min_len}_{max_len}_seed{seed}",
            lengths=lengths,
        )

    @staticmethod
    def lognormal(
        n: int,
        mean_len: float = 128.0,
        std_len: float = 64.0,
        seed: int = 42,
    ) -> DatasetInfo:
        """
        Generate log-normally distributed sequence lengths.

        Args:
            n (int): Number of sequences to generate.
            mean_len (float): Desired mean length.
            std_len (float): Desired standard deviation.
            seed (int): Random seed for reproducibility.

        Returns:
            DatasetInfo: Dataset with log-normal length distribution.
        """
        rng = np.random.default_rng(seed=seed)
        variance = std_len**2
        mu = np.log(mean_len**2 / np.sqrt(variance + mean_len**2))
        sigma = np.sqrt(np.log(1 + variance / mean_len**2))
        raw = rng.lognormal(mean=mu, sigma=sigma, size=n)
        lengths = np.clip(raw, 1, None).astype(int).tolist()
        return DatasetInfo(
            name=f"lognormal_{n}_mean{mean_len}_std{std_len}_seed{seed}",
            lengths=lengths,
        )

    @staticmethod
    def exponential(
        n: int,
        mean_len: float = 128.0,
        seed: int = 42,
    ) -> DatasetInfo:
        """
        Generate exponentially distributed sequence lengths.

        Args:
            n (int): Number of sequences to generate.
            mean_len (float): Mean sequence length.
            seed (int): Random seed for reproducibility.

        Returns:
            DatasetInfo: Dataset with exponential length distribution.
        """
        rng = np.random.default_rng(seed=seed)
        raw = rng.exponential(scale=mean_len, size=n)
        lengths = np.clip(raw, 1, None).astype(int).tolist()
        return DatasetInfo(
            name=f"exponential_{n}_mean{mean_len}_seed{seed}",
            lengths=lengths,
        )

    @staticmethod
    def bimodal(
        n: int,
        short_mean: float = 64.0,
        long_mean: float = 512.0,
        short_ratio: float = 0.7,
        seed: int = 42,
    ) -> DatasetInfo:
        """
        Generate bimodally distributed sequence lengths.

        Args:
            n (int): Number of sequences to generate.
            short_mean (float): Mean length for short sequences.
            long_mean (float): Mean length for long sequences.
            short_ratio (float): Fraction of short sequences (0-1).
            seed (int): Random seed for reproducibility.

        Returns:
            DatasetInfo: Dataset with bimodal length distribution.
        """
        rng = np.random.default_rng(seed=seed)
        n_short = int(n * short_ratio)
        n_long = n - n_short
        short_seqs = rng.normal(loc=short_mean, scale=short_mean * 0.2, size=n_short)
        long_seqs = rng.normal(loc=long_mean, scale=long_mean * 0.2, size=n_long)
        raw = np.concatenate([short_seqs, long_seqs])
        rng.shuffle(raw)
        lengths = np.clip(raw, 1, None).astype(int).tolist()
        return DatasetInfo(
            name=f"bimodal_{n}_short{short_mean}_long{long_mean}_ratio{short_ratio}_seed{seed}",
            lengths=lengths,
        )
