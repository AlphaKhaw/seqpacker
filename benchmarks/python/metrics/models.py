"""
Pydantic v2 models for benchmark metrics and data collection.
"""

import statistics
import time
import tracemalloc
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, computed_field

########################################
# Dataset Information
########################################


class DatasetInfo(BaseModel):
    """
    Information about a dataset including its sequence lengths and statistics.

    Used by both synthetic generators and real dataset loaders to represent
    a dataset's sequence length distribution with auto-computed statistics.
    """

    name: str
    lengths: list[int]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def num_sequences(self) -> int:
        """
        Return the number of sequences.

        Returns:
            int: The number of sequences.
        """
        return len(self.lengths)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mean_length(self) -> float:
        """
        Return the mean sequence length.

        Returns:
            float: The mean sequence length.
        """
        if not self.lengths:
            return 0.0
        return statistics.mean(self.lengths)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def median_length(self) -> float:
        """
        Return the median sequence length.

        Returns:
            float: The median sequence length.
        """
        if not self.lengths:
            return 0.0
        return statistics.median(self.lengths)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def std_length(self) -> float:
        """
        Return the standard deviation of sequence lengths.

        Returns:
            float: The standard deviation of sequence lengths.
        """
        if len(self.lengths) < 2:
            return 0.0
        return statistics.stdev(self.lengths)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def percentiles(self) -> dict[str, float]:
        """
        Return p25, p50, p75, p90, p95, p99 percentiles.

        Returns:
            dict[str, float]: The percentiles.
        """
        if not self.lengths:
            return {}
        arr = np.array(self.lengths)
        return {f"p{p}": float(np.percentile(arr, p)) for p in [25, 50, 75, 90, 95, 99]}


########################################
# Packing Efficiency Metrics
########################################


class EfficiencyMetrics(BaseModel):
    """
    Packing efficiency metrics.

    Measures how well a packing algorithm utilizes bin capacity,
    including the ratio of real tokens to total capacity (including padding).
    """

    packing_efficiency: float = Field(
        description="Ratio of used tokens to total capacity (0-1)"
    )
    padding_ratio: float = Field(
        description="Ratio of padding tokens to total capacity (0-1)"
    )
    total_tokens: int = Field(description="Total number of real tokens packed")
    total_capacity: int = Field(description="Total capacity across all bins")


# #### Performance Metrics ####


class PerformanceMetrics(BaseModel):
    """
    Performance timing and resource metrics.

    Captures wall-clock packing time, throughput, and peak memory usage
    for comparing algorithm implementations.
    """

    packing_time_ms: float = Field(description="Time to pack in milliseconds")
    throughput_seqs_per_sec: float = Field(description="Sequences packed per second")
    memory_peak_mb: float = Field(description="Peak memory usage in megabytes")


########################################
# Pack Statistics
########################################


class PackStatistics(BaseModel):
    """
    Statistics about the resulting packs/bins.

    Describes the shape of the packing output: how many bins were created
    and the distribution of sequences across those bins.
    """

    num_packs: int
    sequences_per_pack_mean: float
    sequences_per_pack_min: int
    sequences_per_pack_max: int
    sequences_per_pack_std: float


########################################
# Combined Benchmark Result
########################################


class BenchmarkResult(BaseModel):
    """
    Combined benchmark result for a single runner x dataset x config run.

    Aggregates efficiency, performance, and pack statistics into a single
    serializable record for comparison across runners and datasets.
    """

    runner_name: str
    dataset_name: str
    max_seq_len: int
    num_sequences: int
    efficiency: EfficiencyMetrics
    performance: PerformanceMetrics
    pack_stats: PackStatistics

    def to_json(self) -> str:
        """
        Serialize to JSON string.

        Returns:
            str: JSON representation.
        """
        return self.model_dump_json(indent=2)


########################################
# Metric Collection
########################################


class MetricCollector:
    """
    Context manager for collecting timing and memory metrics during packing.

    Wraps time.perf_counter for wall-clock timing and tracemalloc for
    peak memory measurement. Use as a context manager around the packing call.
    """

    def __init__(self) -> None:
        """
        Initialize the metric collector.
        """
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._peak_memory: int = 0
        self.elapsed_ms: float = 0.0
        self.peak_memory_mb: float = 0.0

    def __enter__(self) -> "MetricCollector":
        """
        Start timing and memory tracking.

        Returns:
            MetricCollector: The metric collector.
        """
        tracemalloc.start()
        self._start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """
        Stop timing and memory tracking.

        Args:
            exc_type: The type of exception that occurred.
            exc_val: The exception that occurred.
            exc_tb: The traceback of the exception.
        """
        self._end_time = time.perf_counter()
        _, self._peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.elapsed_ms = (self._end_time - self._start_time) * 1000.0
        self.peak_memory_mb = self._peak_memory / (1024 * 1024)
