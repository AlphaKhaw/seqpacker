"""
Tests for metrics models and MetricCollector.
"""

import time

from benchmarks.python.metrics.models import (
    BenchmarkResult,
    DatasetInfo,
    EfficiencyMetrics,
    MetricCollector,
    PackStatistics,
    PerformanceMetrics,
)


class TestMetricCollector:
    """
    Tests for MetricCollector context manager.
    """

    def test_timing_works(self) -> None:
        """
        Verify timing captures elapsed time.
        """
        collector = MetricCollector()
        with collector:
            time.sleep(0.05)
        assert collector.elapsed_ms >= 40  # Allow some tolerance

    def test_memory_tracking(self) -> None:
        """
        Verify memory tracking returns non-negative value.
        """
        collector = MetricCollector()
        with collector:
            _ = [0] * 10000
        assert collector.peak_memory_mb >= 0


class TestEfficiencyMetrics:
    """
    Tests for EfficiencyMetrics validation.
    """

    def test_valid_metrics(self) -> None:
        """
        Verify valid metrics are accepted.
        """
        m = EfficiencyMetrics(
            packing_efficiency=0.85,
            padding_ratio=0.15,
            total_tokens=8500,
            total_capacity=10000,
        )
        assert m.packing_efficiency == 0.85

    def test_serialization(self) -> None:
        """
        Verify metrics serialize to JSON.
        """
        m = EfficiencyMetrics(
            packing_efficiency=0.85,
            padding_ratio=0.15,
            total_tokens=8500,
            total_capacity=10000,
        )
        data = m.model_dump()
        assert "packing_efficiency" in data


class TestBenchmarkResult:
    """
    Tests for BenchmarkResult model.
    """

    def test_full_result(self) -> None:
        """
        Verify full result construction and serialization.
        """
        result = BenchmarkResult(
            runner_name="test",
            dataset_name="test_dataset",
            max_seq_len=512,
            num_sequences=100,
            efficiency=EfficiencyMetrics(
                packing_efficiency=0.9,
                padding_ratio=0.1,
                total_tokens=900,
                total_capacity=1000,
            ),
            performance=PerformanceMetrics(
                packing_time_ms=10.5,
                throughput_seqs_per_sec=9523.8,
                memory_peak_mb=1.2,
            ),
            pack_stats=PackStatistics(
                num_packs=10,
                sequences_per_pack_mean=10.0,
                sequences_per_pack_min=5,
                sequences_per_pack_max=15,
                sequences_per_pack_std=2.5,
            ),
        )
        json_str = result.to_json()
        assert "test" in json_str
        assert "packing_efficiency" in json_str


class TestDatasetInfo:
    """
    Tests for DatasetInfo computed fields.
    """

    def test_empty_lengths(self) -> None:
        """
        Verify handling of empty lengths list.
        """
        ds = DatasetInfo(name="empty", lengths=[])
        assert ds.num_sequences == 0
        assert ds.mean_length == 0.0
        assert ds.median_length == 0.0
        assert ds.std_length == 0.0
        assert ds.percentiles == {}

    def test_single_length(self) -> None:
        """
        Verify handling of single-element lengths.
        """
        ds = DatasetInfo(name="single", lengths=[42])
        assert ds.num_sequences == 1
        assert ds.mean_length == 42.0
        assert ds.std_length == 0.0
