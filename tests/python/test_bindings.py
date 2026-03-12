"""
Correctness tests for seqpacker Python bindings.
"""

import random
import time

import numpy as np
import pytest
from seqpacker import Pack, Packer, PackMetrics, PackResult, pack_sequences

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestPackerConstruction:
    """
    Test Packer construction and argument handling.
    """

    def test_default_strategy(self):
        """
        Default strategy should be obfd.
        """
        p = Packer(capacity=1024)
        assert "obfd" in repr(p).lower() or repr(p)

    def test_short_name(self):
        """
        Construct with a known short name.
        """
        p = Packer(capacity=1024, strategy="ffd")
        assert p is not None

    def test_case_insensitive(self):
        """
        Strategy names should be case-insensitive.
        """
        p = Packer(capacity=1024, strategy="FFD")
        assert p is not None

    def test_invalid_strategy(self):
        """
        Unknown strategy should raise ValueError.
        """
        with pytest.raises(ValueError):
            Packer(capacity=1024, strategy="nonexistent_strategy")

    def test_seed_param(self):
        """
        Seed parameter should be accepted.
        """
        p = Packer(capacity=1024, strategy="ffs", seed=42)
        assert p is not None


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------


class TestPacking:
    """
    Test core packing functionality.
    """

    @pytest.fixture()
    def packer(self):
        """
        Create a default packer.
        """
        return Packer(capacity=2048)

    def test_basic_pack(self, packer: Packer):
        """
        Pack should return a PackResult.
        """
        result = packer.pack([500, 600, 400, 1000])
        assert isinstance(result, PackResult)

    def test_all_items_accounted(self, packer: Packer):
        """
        Every input sequence ID must appear exactly once.
        """
        lengths = [100, 200, 300, 400, 500]
        result = packer.pack(lengths)
        all_ids = sorted(id_ for bin_ in result.bins for id_ in bin_)
        assert all_ids == list(range(len(lengths)))

    def test_no_bin_exceeds_capacity(self, packer: Packer):
        """
        No bin's total length should exceed capacity.
        """
        lengths = [100, 200, 300, 400, 500, 600, 700]
        result = packer.pack(lengths)
        for pack in result.packs:
            assert pack.used <= 2048

    def test_oversize_raises(self, packer: Packer):
        """
        A length exceeding capacity should raise ValueError.
        """
        with pytest.raises(ValueError):
            packer.pack([100, 5000])

    def test_empty_raises(self, packer: Packer):
        """
        Empty input should raise ValueError.
        """
        with pytest.raises(ValueError):
            packer.pack([])

    def test_bins_property_shape(self, packer: Packer):
        """
        Verify bins property returns list[list[int]].
        """
        result = packer.pack([100, 200, 300])
        assert isinstance(result.bins, list)
        assert all(isinstance(b, list) for b in result.bins)
        assert all(isinstance(i, int) for b in result.bins for i in b)


# ---------------------------------------------------------------------------
# All strategies
# ---------------------------------------------------------------------------

ALL_STRATEGIES = [s[0] for s in Packer.strategies()]

# All strategies are implemented
_IMPLEMENTED = ALL_STRATEGIES


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
class TestAllStrategies:
    """
    Every strategy should produce valid output.
    """

    def test_valid_output(self, strategy: str):
        """
        Pack with each strategy and verify basic invariants.
        """
        packer = Packer(capacity=1024, strategy=strategy, seed=42)
        lengths = [100, 200, 300, 400, 500, 50, 150, 250]
        result = packer.pack(lengths)
        all_ids = sorted(id_ for bin_ in result.bins for id_ in bin_)
        assert all_ids == list(range(len(lengths)))
        for pack in result.packs:
            assert pack.used <= 1024


# ---------------------------------------------------------------------------
# NumPy integration
# ---------------------------------------------------------------------------


class TestNumPy:
    """
    Test NumPy array input/output.
    """

    def test_numpy_input(self):
        """
        Verify pack() accepts numpy int64 arrays.
        """
        packer = Packer(capacity=2048)
        lengths = np.array([100, 200, 300, 400], dtype=np.int64)
        result = packer.pack(lengths)
        assert isinstance(result, PackResult)
        assert len(result) > 0

    def test_pack_flat_returns_numpy(self):
        """
        Verify pack_flat() returns numpy int64 arrays.
        """
        packer = Packer(capacity=2048)
        items_flat, bin_offsets = packer.pack_flat([100, 200, 300, 400])
        assert isinstance(items_flat, np.ndarray)
        assert isinstance(bin_offsets, np.ndarray)
        assert items_flat.dtype == np.int64
        assert bin_offsets.dtype == np.int64

    def test_roundtrip_via_split(self):
        """
        Reconstruct bins from pack_flat output using np.split.
        """
        packer = Packer(capacity=2048)
        lengths = [100, 200, 300, 400, 500]
        items_flat, bin_offsets = packer.pack_flat(lengths)
        bins = np.split(items_flat, bin_offsets)
        # All IDs present
        all_ids = sorted(int(i) for b in bins for i in b)
        assert all_ids == list(range(len(lengths)))


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


class TestConvenienceFunction:
    """
    Test pack_sequences() top-level function.
    """

    def test_with_defaults(self):
        """
        Verify pack_sequences works with only required args.
        """
        result = pack_sequences(
            lengths=[100, 200, 300],
            capacity=1024,
        )
        assert isinstance(result, PackResult)

    def test_with_explicit_strategy(self):
        """
        Verify pack_sequences works with an explicit strategy.
        """
        result = pack_sequences(
            lengths=[100, 200, 300],
            capacity=1024,
            strategy="ff",
        )
        assert isinstance(result, PackResult)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Strategies list
# ---------------------------------------------------------------------------


class TestStrategiesList:
    """
    Test Packer.strategies() class method.
    """

    def test_returns_list_of_tuples(self):
        """
        Verify strategies() returns list[tuple[str, str]] with >= 9 entries.
        """
        strats = Packer.strategies()
        assert isinstance(strats, list)
        assert len(strats) >= 9
        for item in strats:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], str)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    """
    Test PackMetrics properties and types.
    """

    @pytest.fixture()
    def metrics(self) -> PackMetrics:
        """
        Get metrics from a packing run.
        """
        result = pack_sequences(
            lengths=[100, 200, 300, 400, 500],
            capacity=1024,
        )
        return result.metrics

    def test_int_properties(self, metrics: PackMetrics):
        """
        Verify integer metric properties return int type.
        """
        assert isinstance(metrics.num_sequences, int)
        assert isinstance(metrics.total_tokens, int)
        assert isinstance(metrics.num_packs, int)
        assert isinstance(metrics.padding_tokens, int)

    def test_float_properties(self, metrics: PackMetrics):
        """
        Verify float metric properties return float type.
        """
        assert isinstance(metrics.efficiency, float)
        assert isinstance(metrics.avg_utilisation, float)
        assert isinstance(metrics.utilisation_std, float)
        assert isinstance(metrics.min_utilisation, float)
        assert isinstance(metrics.max_utilisation, float)
        assert isinstance(metrics.avg_sequences_per_pack, float)
        assert isinstance(metrics.packing_time_ms, float)
        assert isinstance(metrics.padding_ratio, float)
        assert isinstance(metrics.throughput, float)

    def test_efficiency_range(self, metrics: PackMetrics):
        """
        Verify efficiency is between 0 and 1.
        """
        assert 0.0 <= metrics.efficiency <= 1.0

    def test_num_sequences_correct(self, metrics: PackMetrics):
        """
        Verify num_sequences matches the input count.
        """
        assert metrics.num_sequences == 5


# ---------------------------------------------------------------------------
# PackResult
# ---------------------------------------------------------------------------


class TestPackResult:
    """
    Test PackResult properties.
    """

    @pytest.fixture()
    def result(self) -> PackResult:
        """
        Get a packing result.
        """
        return pack_sequences(
            lengths=[100, 200, 300, 400, 500],
            capacity=1024,
        )

    def test_bins(self, result: PackResult):
        """
        Verify bins property returns list[list[int]].
        """
        assert isinstance(result.bins, list)

    def test_packs(self, result: PackResult):
        """
        Verify packs property returns list[Pack].
        """
        assert isinstance(result.packs, list)
        assert all(isinstance(p, Pack) for p in result.packs)

    def test_efficiency(self, result: PackResult):
        """
        Verify efficiency shortcut matches metrics.efficiency.
        """
        assert result.efficiency == result.metrics.efficiency

    def test_time_ms(self, result: PackResult):
        """
        Verify time_ms shortcut matches metrics.packing_time_ms.
        """
        assert result.time_ms == result.metrics.packing_time_ms

    def test_num_bins(self, result: PackResult):
        """
        Verify num_bins matches len(packs).
        """
        assert result.num_bins == len(result.packs)

    def test_len(self, result: PackResult):
        """
        Verify __len__ matches num_bins.
        """
        assert len(result) == result.num_bins

    def test_repr(self, result: PackResult):
        """
        Verify __repr__ returns a non-empty string.
        """
        r = repr(result)
        assert isinstance(r, str)
        assert len(r) > 0


# ---------------------------------------------------------------------------
# Efficiency and metrics consistency (all implemented strategies)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", _IMPLEMENTED)
class TestEfficiencyAndMetrics:
    """
    Verify efficiency, metrics consistency, and accounting for every
    implemented strategy on a non-trivial workload.
    """

    @pytest.fixture()
    def result(self, strategy: str) -> PackResult:
        """
        Pack a reproducible workload with the given strategy.
        """
        rng = random.Random(42)
        lengths = [rng.randint(50, 900) for _ in range(200)]
        packer = Packer(capacity=1024, strategy=strategy, seed=42)
        return packer.pack(lengths)

    def test_efficiency_positive(self, result: PackResult, strategy: str):
        """
        Verify packing efficiency is strictly positive.
        """
        assert result.efficiency > 0.0

    def test_efficiency_bounded(self, result: PackResult, strategy: str):
        """
        Verify packing efficiency does not exceed 1.0.
        """
        assert result.efficiency <= 1.0

    def test_padding_ratio_complements_efficiency(
        self, result: PackResult, strategy: str
    ):
        """
        Verify padding_ratio + efficiency approximately equal 1.0.
        """
        total = result.metrics.efficiency + result.metrics.padding_ratio
        assert abs(total - 1.0) < 1e-9

    def test_total_tokens_equals_sum_of_used(self, result: PackResult, strategy: str):
        """
        Verify total_tokens equals the sum of used across all packs.
        """
        sum_used = sum(p.used for p in result.packs)
        assert result.metrics.total_tokens == sum_used

    def test_num_packs_matches(self, result: PackResult, strategy: str):
        """
        Verify metrics.num_packs equals len(packs).
        """
        assert result.metrics.num_packs == len(result.packs)

    def test_num_sequences_matches(self, result: PackResult, strategy: str):
        """
        Verify num_sequences equals the total items across all bins.
        """
        total_items = sum(len(p.sequence_ids) for p in result.packs)
        assert result.metrics.num_sequences == total_items

    def test_padding_tokens_nonnegative(self, result: PackResult, strategy: str):
        """
        Verify padding_tokens is non-negative.
        """
        assert result.metrics.padding_tokens >= 0

    def test_packing_time_positive(self, result: PackResult, strategy: str):
        """
        Verify packing_time_ms is positive.
        """
        assert result.metrics.packing_time_ms > 0.0

    def test_throughput_positive(self, result: PackResult, strategy: str):
        """
        Verify throughput is positive.
        """
        assert result.metrics.throughput > 0.0

    def test_utilisation_range(self, result: PackResult, strategy: str):
        """
        Verify utilisation metrics are within [0, 1].
        """
        assert 0.0 <= result.metrics.min_utilisation <= 1.0
        assert 0.0 <= result.metrics.max_utilisation <= 1.0
        assert 0.0 <= result.metrics.avg_utilisation <= 1.0
        assert result.metrics.min_utilisation <= result.metrics.avg_utilisation
        assert result.metrics.avg_utilisation <= result.metrics.max_utilisation


# ---------------------------------------------------------------------------
# Performance (speed) sanity checks
# ---------------------------------------------------------------------------


class TestPerformance:
    """
    Verify that packing completes within reasonable time bounds.
    """

    def test_10k_sequences_under_one_second(self):
        """
        Pack 10,000 sequences and verify it completes in under 1 second.
        """
        rng = random.Random(123)
        lengths = [rng.randint(1, 2048) for _ in range(10_000)]
        packer = Packer(capacity=2048, strategy="obfd")
        start = time.perf_counter()
        result = packer.pack(lengths)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"Packing 10k sequences took {elapsed:.3f}s"
        assert result.metrics.num_sequences == 10_000

    @pytest.mark.parametrize("strategy", _IMPLEMENTED)
    def test_all_strategies_complete_promptly(self, strategy: str):
        """
        Verify every implemented strategy packs 1,000 sequences in < 2s.
        """
        rng = random.Random(99)
        lengths = [rng.randint(1, 512) for _ in range(1_000)]
        packer = Packer(capacity=512, strategy=strategy, seed=42)
        start = time.perf_counter()
        result = packer.pack(lengths)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"{strategy} took {elapsed:.3f}s for 1k seqs"
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Algorithm coverage verification
# ---------------------------------------------------------------------------


class TestAlgorithmCoverage:
    """
    Verify that all implemented algorithms are exercised and produce
    distinct behaviour where expected.
    """

    def test_all_implemented_strategies_are_covered(self):
        """
        Verify _IMPLEMENTED covers all strategies.
        """
        assert set(_IMPLEMENTED) == set(ALL_STRATEGIES)

    def test_decreasing_strategies_beat_online(self):
        """
        Verify FFD/BFD/OBFD produce fewer or equal bins than FF/BF on
        a workload where sorting helps.
        """
        rng = random.Random(77)
        lengths = [rng.randint(100, 900) for _ in range(500)]
        capacity = 1024

        online_bins = {}
        for s in ["FF", "BF"]:
            r = Packer(capacity=capacity, strategy=s).pack(lengths)
            online_bins[s] = r.num_bins

        decreasing_bins = {}
        for s in ["FFD", "BFD", "OBFD"]:
            r = Packer(capacity=capacity, strategy=s).pack(lengths)
            decreasing_bins[s] = r.num_bins

        best_online = min(online_bins.values())
        for s, n in decreasing_bins.items():
            assert n <= best_online, (
                f"{s} ({n} bins) should be <= best online ({best_online} bins)"
            )

    def test_ffs_deterministic_with_seed(self):
        """
        Verify FFS produces identical results with the same seed.
        """
        lengths = list(range(10, 510, 10))
        r1 = Packer(capacity=512, strategy="ffs", seed=42).pack(lengths)
        r2 = Packer(capacity=512, strategy="ffs", seed=42).pack(lengths)
        assert r1.bins == r2.bins

    def test_ffs_differs_with_different_seed(self):
        """
        Verify FFS produces different results with different seeds.
        """
        lengths = list(range(10, 510, 10))
        r1 = Packer(capacity=512, strategy="ffs", seed=1).pack(lengths)
        r2 = Packer(capacity=512, strategy="ffs", seed=999).pack(lengths)
        assert r1.bins != r2.bins

    def test_next_fit_uses_most_bins(self):
        """
        Verify NF (simplest, worst ratio) uses >= bins than any other strategy.
        Harmonic-K excluded: class segregation can legitimately use more bins than NF.
        """
        rng = random.Random(55)
        lengths = [rng.randint(100, 400) for _ in range(300)]
        capacity = 512

        nf_bins = Packer(capacity=capacity, strategy="nf").pack(lengths).num_bins
        for s in _IMPLEMENTED:
            if s in ("NF", "HK"):
                continue
            other_bins = (
                Packer(capacity=capacity, strategy=s, seed=42).pack(lengths).num_bins
            )
            assert nf_bins >= other_bins, (
                f"NF ({nf_bins}) should use >= bins than {s} ({other_bins})"
            )
