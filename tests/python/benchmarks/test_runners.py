"""
Tests for packing runners.
"""

import pytest

from benchmarks.python.datasets.synthetic import SyntheticDataGenerator
from benchmarks.python.runners.binpacking import BinpackingRunner
from benchmarks.python.runners.greedy import GreedyFFDRunner
from benchmarks.python.runners.naive import NaivePaddingRunner
from benchmarks.python.runners.prtpy import PrtpyFFDRunner, PrtpyFirstFitRunner


@pytest.fixture
def sample_lengths() -> list[int]:
    """
    Provide a sample list of sequence lengths.

    Returns:
        list[int]: Sample lengths.
    """
    ds = SyntheticDataGenerator.uniform(n=100, min_len=16, max_len=256, seed=42)
    return ds.lengths


class TestNaivePaddingRunner:
    """
    Tests for naive padding runner.
    """

    def test_one_bin_per_sequence(self, sample_lengths: list[int]) -> None:
        """
        Verify naive creates one bin per sequence.

        Args:
            sample_lengths (list[int]): Sample lengths.
        """
        runner = NaivePaddingRunner()
        bins = runner.pack(lengths=sample_lengths, max_seq_len=512)
        assert len(bins) == len(sample_lengths)

    def test_each_bin_has_one_item(self, sample_lengths: list[int]) -> None:
        """
        Verify each bin contains exactly one sequence.

        Args:
            sample_lengths (list[int]): Sample lengths.
        """
        runner = NaivePaddingRunner()
        bins = runner.pack(lengths=sample_lengths, max_seq_len=512)
        assert all(len(b) == 1 for b in bins)


class TestGreedyFFDRunner:
    """Tests for greedy FFD runner."""

    def test_fewer_bins_than_naive(self, sample_lengths: list[int]) -> None:
        """
        Verify FFD creates fewer bins than naive.

        Args:
            sample_lengths (list[int]): Sample lengths.
        """
        naive = NaivePaddingRunner()
        ffd = GreedyFFDRunner()
        naive_bins = naive.pack(lengths=sample_lengths, max_seq_len=512)
        ffd_bins = ffd.pack(lengths=sample_lengths, max_seq_len=512)
        assert len(ffd_bins) < len(naive_bins)

    def test_no_capacity_overflow(self, sample_lengths: list[int]) -> None:
        """
        Verify no bin exceeds capacity.

        Args:
            sample_lengths (list[int]): Sample lengths.
        """
        runner = GreedyFFDRunner()
        max_seq_len = 512
        bins = runner.pack(lengths=sample_lengths, max_seq_len=max_seq_len)
        assert all(sum(b) <= max_seq_len for b in bins)

    def test_all_sequences_packed(self, sample_lengths: list[int]) -> None:
        """
        Verify all sequences are packed.

        Args:
            sample_lengths (list[int]): Sample lengths.
        """
        runner = GreedyFFDRunner()
        bins = runner.pack(lengths=sample_lengths, max_seq_len=512)
        packed_count = sum(len(b) for b in bins)
        assert packed_count == len(sample_lengths)


class TestPrtpyRunners:
    """
    Tests for prtpy runners.
    """

    def test_ffd_no_overflow(self, sample_lengths: list[int]) -> None:
        """
        Verify prtpy FFD doesn't overflow bins.

        Args:
            sample_lengths (list[int]): Sample lengths.
        """
        runner = PrtpyFFDRunner()
        max_seq_len = 512
        bins = runner.pack(lengths=sample_lengths, max_seq_len=max_seq_len)
        assert all(sum(b) <= max_seq_len for b in bins)

    def test_ff_no_overflow(self, sample_lengths: list[int]) -> None:
        """
        Verify prtpy FF doesn't overflow bins.

        Args:
            sample_lengths (list[int]): Sample lengths.
        """
        runner = PrtpyFirstFitRunner()
        max_seq_len = 512
        bins = runner.pack(lengths=sample_lengths, max_seq_len=max_seq_len)
        assert all(sum(b) <= max_seq_len for b in bins)

    def test_ffd_fewer_than_naive(self, sample_lengths: list[int]) -> None:
        """
        Verify prtpy FFD creates fewer bins than naive.

        Args:
            sample_lengths (list[int]): Sample lengths.
        """
        naive = NaivePaddingRunner()
        ffd = PrtpyFFDRunner()
        naive_bins = naive.pack(lengths=sample_lengths, max_seq_len=512)
        ffd_bins = ffd.pack(lengths=sample_lengths, max_seq_len=512)
        assert len(ffd_bins) < len(naive_bins)


class TestBinpackingRunner:
    """
    Tests for binpacking runner.
    """

    def test_no_overflow(self, sample_lengths: list[int]) -> None:
        """
        Verify binpacking doesn't overflow bins.

        Args:
            sample_lengths (list[int]): Sample lengths.
        """
        runner = BinpackingRunner()
        max_seq_len = 512
        bins = runner.pack(lengths=sample_lengths, max_seq_len=max_seq_len)
        assert all(sum(b) <= max_seq_len for b in bins)

    def test_fewer_than_naive(self, sample_lengths: list[int]) -> None:
        """
        Verify binpacking creates fewer bins than naive.

        Args:
            sample_lengths (list[int]): Sample lengths.
        """
        naive = NaivePaddingRunner()
        bp = BinpackingRunner()
        naive_bins = naive.pack(lengths=sample_lengths, max_seq_len=512)
        bp_bins = bp.pack(lengths=sample_lengths, max_seq_len=512)
        assert len(bp_bins) < len(naive_bins)


class TestAllRunnersProtocol:
    """
    Verify all runners conform to the PackingRunner protocol.
    """

    def test_all_have_name(self) -> None:
        """
        Verify all runners have a name property.
        """
        runners = [
            NaivePaddingRunner(),
            GreedyFFDRunner(),
            PrtpyFFDRunner(),
            PrtpyFirstFitRunner(),
            BinpackingRunner(),
        ]
        for runner in runners:
            assert isinstance(runner.name, str)
            assert len(runner.name) > 0

    def test_empty_input(self) -> None:
        """
        Verify all runners handle empty input.
        """
        runners = [
            NaivePaddingRunner(),
            GreedyFFDRunner(),
            PrtpyFFDRunner(),
            PrtpyFirstFitRunner(),
            BinpackingRunner(),
        ]
        for runner in runners:
            bins = runner.pack(lengths=[], max_seq_len=512)
            assert bins == []
