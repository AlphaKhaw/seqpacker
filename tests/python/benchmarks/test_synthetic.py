"""
Tests for synthetic dataset generators.
"""

from benchmarks.python.datasets.synthetic import SyntheticDataGenerator


class TestUniformGenerator:
    """
    Tests for uniform distribution generator.
    """

    def test_correct_count(self) -> None:
        """
        Verify correct number of sequences generated.
        """
        ds = SyntheticDataGenerator.uniform(n=100, min_len=10, max_len=50, seed=42)
        assert len(ds.lengths) == 100

    def test_range_bounds(self) -> None:
        """
        Verify all lengths within specified range.
        """
        ds = SyntheticDataGenerator.uniform(n=1000, min_len=10, max_len=50, seed=42)
        assert all(10 <= length <= 50 for length in ds.lengths)

    def test_deterministic(self) -> None:
        """
        Verify same seed produces same results.
        """
        ds1 = SyntheticDataGenerator.uniform(n=100, seed=123)
        ds2 = SyntheticDataGenerator.uniform(n=100, seed=123)
        assert ds1.lengths == ds2.lengths

    def test_different_seeds(self) -> None:
        """
        Verify different seeds produce different results.
        """
        ds1 = SyntheticDataGenerator.uniform(n=100, seed=1)
        ds2 = SyntheticDataGenerator.uniform(n=100, seed=2)
        assert ds1.lengths != ds2.lengths


class TestLognormalGenerator:
    """
    Tests for log-normal distribution generator.
    """

    def test_correct_count(self) -> None:
        """
        Verify correct number of sequences generated.
        """
        ds = SyntheticDataGenerator.lognormal(n=100, seed=42)
        assert len(ds.lengths) == 100

    def test_positive_lengths(self) -> None:
        """
        Verify all lengths are positive.
        """
        ds = SyntheticDataGenerator.lognormal(n=1000, seed=42)
        assert all(length >= 1 for length in ds.lengths)

    def test_deterministic(self) -> None:
        """
        Verify same seed produces same results.
        """
        ds1 = SyntheticDataGenerator.lognormal(n=100, seed=42)
        ds2 = SyntheticDataGenerator.lognormal(n=100, seed=42)
        assert ds1.lengths == ds2.lengths


class TestExponentialGenerator:
    """
    Tests for exponential distribution generator.
    """

    def test_correct_count(self) -> None:
        """
        Verify correct number of sequences generated.
        """
        ds = SyntheticDataGenerator.exponential(n=100, seed=42)
        assert len(ds.lengths) == 100

    def test_positive_lengths(self) -> None:
        """
        Verify all lengths are positive.
        """
        ds = SyntheticDataGenerator.exponential(n=1000, seed=42)
        assert all(length >= 1 for length in ds.lengths)

    def test_deterministic(self) -> None:
        """
        Verify same seed produces same results.
        """
        ds1 = SyntheticDataGenerator.exponential(n=100, seed=42)
        ds2 = SyntheticDataGenerator.exponential(n=100, seed=42)
        assert ds1.lengths == ds2.lengths


class TestBimodalGenerator:
    """
    Tests for bimodal distribution generator.
    """

    def test_correct_count(self) -> None:
        """
        Verify correct number of sequences generated.
        """
        ds = SyntheticDataGenerator.bimodal(n=100, seed=42)
        assert len(ds.lengths) == 100

    def test_positive_lengths(self) -> None:
        """
        Verify all lengths are positive.
        """
        ds = SyntheticDataGenerator.bimodal(n=1000, seed=42)
        assert all(length >= 1 for length in ds.lengths)

    def test_deterministic(self) -> None:
        """
        Verify same seed produces same results.
        """
        ds1 = SyntheticDataGenerator.bimodal(n=100, seed=42)
        ds2 = SyntheticDataGenerator.bimodal(n=100, seed=42)
        assert ds1.lengths == ds2.lengths


class TestDatasetInfoStats:
    """
    Tests for DatasetInfo computed statistics.
    """

    def test_stats_computed(self) -> None:
        """
        Verify stats are computed correctly.
        """
        ds = SyntheticDataGenerator.uniform(n=100, min_len=10, max_len=50, seed=42)
        assert ds.num_sequences == 100
        assert 10 <= ds.mean_length <= 50
        assert 10 <= ds.median_length <= 50
        assert ds.std_length >= 0
        assert "p50" in ds.percentiles
        assert "p99" in ds.percentiles
