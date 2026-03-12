"""
Main benchmark orchestration entry point.

Runs all benchmark combinations in parallel using process pools,
with sequential single-run execution for deterministic benchmarks.

Usage: uv run python -m benchmarks.python.run_benchmarks
"""

import json
import os
import platform
from datetime import datetime
from pathlib import Path

from benchmarks.python.datasets.real import RealDataLoader
from benchmarks.python.datasets.synthetic import DatasetInfo, SyntheticDataGenerator
from benchmarks.python.metrics.models import (
    BenchmarkResult,
    EfficiencyMetrics,
    MetricCollector,
    PackStatistics,
    PerformanceMetrics,
)
from benchmarks.python.runners.base import PackingRunner
from benchmarks.python.utils.logging import logger

RESULTS_DIR: Path = Path("benchmarks/results")
MAX_SEQ_LENS: list[int] = [512, 1024, 2048, 4096, 8192]
NUM_SEQUENCES: int = 10000


def _create_runner(runner_type: str, strategy: str | None = None) -> PackingRunner:
    """
    Create a runner instance from a serializable config.

    Args:
        runner_type (str): Type of runner to create.
        strategy (str | None): Strategy for seqpacker runners.

    Returns:
        PackingRunner: A runner instance.
    """
    if runner_type == "naive":
        from benchmarks.python.runners.naive import NaivePaddingRunner

        return NaivePaddingRunner()
    elif runner_type == "greedy_ffd":
        from benchmarks.python.runners.greedy import GreedyFFDRunner

        return GreedyFFDRunner()
    elif runner_type == "binpacking":
        from benchmarks.python.runners.binpacking import BinpackingRunner

        return BinpackingRunner()
    elif runner_type == "prtpy_ffd":
        from benchmarks.python.runners.prtpy import PrtpyFFDRunner

        return PrtpyFFDRunner()
    elif runner_type == "prtpy_ff":
        from benchmarks.python.runners.prtpy import PrtpyFirstFitRunner

        return PrtpyFirstFitRunner()
    elif runner_type == "seqpacker":
        from benchmarks.python.runners.seqpacker import SeqpackerRunner

        return SeqpackerRunner(strategy=strategy or "obfd")
    elif runner_type == "lightbinpack":
        from benchmarks.python.runners.lightbinpack import LightBinPackRunner

        return LightBinPackRunner(strategy=strategy or "ffd")
    else:
        raise ValueError(f"Unknown runner type: {runner_type}")


def get_runner_configs() -> list[tuple[str, str | None]]:
    """
    Get serializable configs for all available runners.

    Returns:
        list[tuple[str, str | None]]: List of (runner_type, strategy) tuples.
    """
    configs: list[tuple[str, str | None]] = [
        ("naive", None),
        ("greedy_ffd", None),
        ("prtpy_ffd", None),
        ("prtpy_ff", None),
        ("binpacking", None),
    ]

    for strategy in [
        "nf",
        "ff",
        "bf",
        "wf",
        "ffd",
        "bfd",
        "ffs",
        "mffd",
        "obfd",
        "obfdp",
        "hk",
    ]:
        configs.append(("seqpacker", strategy))

    # Check LightBinPack availability in the main process
    try:
        from benchmarks.python.runners.lightbinpack import LightBinPackRunner

        if LightBinPackRunner().available:
            for lbp_strategy in ["ffd", "bfd", "obfd", "obfdp"]:
                configs.append(("lightbinpack", lbp_strategy))
    except Exception:
        pass

    return configs


def get_synthetic_datasets() -> list[DatasetInfo]:
    """
    Get all synthetic benchmark datasets.

    Returns:
        list: List of DatasetInfo instances.
    """
    return [
        SyntheticDataGenerator.uniform(
            n=NUM_SEQUENCES, min_len=16, max_len=512, seed=42
        ),
        SyntheticDataGenerator.lognormal(
            n=NUM_SEQUENCES, mean_len=128, std_len=64, seed=42
        ),
        SyntheticDataGenerator.exponential(n=NUM_SEQUENCES, mean_len=128, seed=42),
        SyntheticDataGenerator.bimodal(
            n=NUM_SEQUENCES,
            short_mean=64,
            long_mean=512,
            short_ratio=0.7,
            seed=42,
        ),
    ]


def get_real_datasets() -> list:
    """
    Get all real benchmark datasets from HuggingFace.

    Loads and tokenizes on first run, cached afterwards.

    Returns:
        list: List of DatasetInfo instances.
    """
    datasets = []
    for name in RealDataLoader.available_datasets():
        try:
            logger.info(f"Loading real dataset '{name}'...")
            ds = RealDataLoader.load(dataset_name=name, max_samples=NUM_SEQUENCES)
            datasets.append(ds)
        except Exception as e:
            logger.warning(f"Failed to load '{name}': {e}")
    return datasets


def _run_benchmark_task(
    runner_type: str,
    strategy: str | None,
    dataset_name: str,
    valid_lengths: list[int],
    max_seq_len: int,
) -> BenchmarkResult | None:
    """
    Run a single benchmark task once.

    Args:
        runner_type (str): Type of runner to create.
        strategy (str | None): Strategy for seqpacker runners.
        dataset_name (str): Name of the dataset.
        valid_lengths (list[int]): Pre-filtered sequence lengths.
        max_seq_len (int): Maximum bin capacity.

    Returns:
        BenchmarkResult | None: Result, or None on failure.
    """
    try:
        runner = _create_runner(runner_type=runner_type, strategy=strategy)
        runner_name = runner.name

        collector = MetricCollector()
        with collector:
            bins = runner.pack(max_seq_len=max_seq_len, lengths=valid_lengths)

        num_sequences = sum(len(b) for b in bins)
        total_tokens = sum(sum(b) for b in bins)
        total_capacity = len(bins) * max_seq_len
        packing_efficiency = (
            total_tokens / total_capacity if total_capacity > 0 else 0.0
        )
        padding_ratio = 1.0 - packing_efficiency
        seqs_per_pack = [len(b) for b in bins]
        mean_seqs = sum(seqs_per_pack) / len(seqs_per_pack) if seqs_per_pack else 0.0
        std_seqs = (
            (
                sum((x - mean_seqs) ** 2 for x in seqs_per_pack)
                / (len(seqs_per_pack) - 1)
            )
            ** 0.5
            if len(seqs_per_pack) >= 2
            else 0.0
        )

        return BenchmarkResult(
            runner_name=runner_name,
            dataset_name=dataset_name,
            max_seq_len=max_seq_len,
            num_sequences=num_sequences,
            efficiency=EfficiencyMetrics(
                packing_efficiency=packing_efficiency,
                padding_ratio=padding_ratio,
                total_tokens=total_tokens,
                total_capacity=total_capacity,
            ),
            performance=PerformanceMetrics(
                packing_time_ms=collector.elapsed_ms,
                throughput_seqs_per_sec=(
                    num_sequences / (collector.elapsed_ms / 1000.0)
                    if collector.elapsed_ms > 0
                    else 0.0
                ),
                memory_peak_mb=collector.peak_memory_mb,
            ),
            pack_stats=PackStatistics(
                num_packs=len(bins),
                sequences_per_pack_mean=mean_seqs,
                sequences_per_pack_min=min(seqs_per_pack) if seqs_per_pack else 0,
                sequences_per_pack_max=max(seqs_per_pack) if seqs_per_pack else 0,
                sequences_per_pack_std=std_seqs,
            ),
        )
    except Exception as e:
        runner_label = f"{runner_type}_{strategy}" if strategy else runner_type
        logger.error(
            f"FAILED: {runner_label} on {dataset_name} "
            f"(max_seq_len={max_seq_len}): {type(e).__name__}: {e}"
        )
        return None


def run_all() -> list[BenchmarkResult]:
    """
    Run all benchmark combinations sequentially and return results.

    Sequential execution avoids process-pool overhead that can distort
    timing measurements. Each task runs once (deterministic algorithms).

    Returns:
        list[BenchmarkResult]: All benchmark results.
    """
    runner_configs = get_runner_configs()
    datasets = get_synthetic_datasets() + get_real_datasets()

    # Build all tasks: (runner_config, dataset_name, valid_lengths, max_seq_len)
    tasks: list[tuple[str, str | None, str, list[int], int]] = []
    for dataset in datasets:
        for max_seq_len in MAX_SEQ_LENS:
            valid_lengths = [
                length for length in dataset.lengths if length <= max_seq_len
            ]
            if not valid_lengths:
                logger.warning(
                    f"No valid sequences for {dataset.name} "
                    f"with max_seq_len={max_seq_len}"
                )
                continue

            for runner_type, strategy in runner_configs:
                tasks.append(
                    (runner_type, strategy, dataset.name, valid_lengths, max_seq_len)
                )

    total_tasks = len(tasks)
    logger.info(
        f"Running {total_tasks} benchmark tasks "
        f"({len(runner_configs)} runners x {len(datasets)} datasets x "
        f"{len(MAX_SEQ_LENS)} max_seq_lens), sequential execution"
    )

    all_results: list[BenchmarkResult] = []

    for i, (
        runner_type,
        strategy,
        dataset_name,
        valid_lengths,
        max_seq_len,
    ) in enumerate(tasks, 1):
        runner_label = f"{runner_type}_{strategy}" if strategy else runner_type

        result = _run_benchmark_task(
            runner_type=runner_type,
            strategy=strategy,
            dataset_name=dataset_name,
            valid_lengths=valid_lengths,
            max_seq_len=max_seq_len,
        )

        if result is not None:
            all_results.append(result)
            logger.info(
                f"[{i}/{total_tasks}] {runner_label} on {dataset_name} "
                f"(max_seq_len={max_seq_len}): "
                f"eff={result.efficiency.packing_efficiency:.4f}, "
                f"bins={result.pack_stats.num_packs}, "
                f"time={result.performance.packing_time_ms:.1f}ms"
            )

    return all_results


def get_benchmark_metadata(description: str) -> dict:
    """
    Collect system and benchmark metadata.

    Args:
        description (str): User-provided description of this benchmark run.

    Returns:
        dict: Metadata dictionary.
    """
    return {
        "description": description,
        "timestamp": datetime.now().isoformat(),
        "num_repeats": 1,
        "system": {
            "os": platform.system(),
            "os_version": platform.version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        },
        "environment": {
            "rustflags": os.environ.get("RUSTFLAGS", "(default)"),
        },
    }


def save_results(results: list[BenchmarkResult], metadata: dict) -> None:
    """
    Save benchmark results to JSON files.

    Saves to:
    - Individual result files (overwritten)
    - all_results.json (latest run, overwritten)
    - all_results_history.json (accumulates all runs)

    Args:
        results (list[BenchmarkResult]): Results to save.
        metadata (dict): Benchmark run metadata.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save individual results
    for result in results:
        filename = (
            f"{result.runner_name}_{result.dataset_name}_seq{result.max_seq_len}.json"
        )
        path = RESULTS_DIR / filename
        path.write_text(result.to_json())

    run_data = {
        "metadata": metadata,
        "results": [json.loads(r.to_json()) for r in results],
    }

    # Save latest run (overwrites)
    latest_path = RESULTS_DIR / "all_results.json"
    latest_path.write_text(json.dumps(run_data, indent=2))
    logger.info(f"Saved latest run to {latest_path}")

    # Append to history file
    history_path = RESULTS_DIR / "all_results_history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
    else:
        history = {"runs": []}

    history["runs"].append(run_data)
    history_path.write_text(json.dumps(history, indent=2))
    logger.info(
        f"Appended to history ({len(history['runs'])} total runs) at {history_path}"
    )


def print_summary_table(results: list[BenchmarkResult]) -> None:
    """
    Print a summary table of benchmark results to console.

    Args:
        results (list[BenchmarkResult]): Results to summarize.
    """
    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)
    print(
        f"{'Runner':<20} {'Dataset':<45} {'MaxLen':>6} "
        f"{'Eff%':>7} {'Bins':>6} {'Time(ms)':>10}"
    )
    print("-" * 100)

    for r in sorted(
        results, key=lambda x: (x.dataset_name, x.max_seq_len, x.runner_name)
    ):
        print(
            f"{r.runner_name:<20} {r.dataset_name:<45} {r.max_seq_len:>6} "
            f"{r.efficiency.packing_efficiency * 100:>6.2f}% "
            f"{r.pack_stats.num_packs:>6} "
            f"{r.performance.packing_time_ms:>10.1f}"
        )
    print("=" * 100)


def prompt_for_description() -> str:
    """
    Get benchmark description from BENCH_DESC env var or interactive prompt.

    Set BENCH_DESC environment variable to skip the interactive prompt
    (useful for CI and ``make bench-python``).

    Returns:
        str: Benchmark run description.
    """
    env_desc = os.environ.get("BENCH_DESC", "").strip()
    if env_desc:
        print(f"\nBenchmark description (from BENCH_DESC): {env_desc}\n")
        return env_desc

    print("\n" + "=" * 70)
    print("BENCHMARK CONFIGURATION")
    print("=" * 70)
    print("Please provide a description for this benchmark run.")
    print("(Tip: set BENCH_DESC env var to skip this prompt)")
    print("Examples:")
    print("  - 'Baseline release build on Ubuntu 22.04'")
    print("  - 'With RUSTFLAGS=-C target-cpu=native on macOS M2'")
    print("  - 'After SmallVec optimization'")
    print("=" * 70)
    description = input("\nDescription: ").strip()
    if not description:
        description = "No description provided"
    print()
    return description


def main() -> None:
    """
    Main entry point for benchmark orchestration.
    """
    description = prompt_for_description()
    metadata = get_benchmark_metadata(description=description)

    logger.info(f"Starting benchmark suite: {description}")
    logger.info(f"System: {metadata['system']['platform']}")
    logger.info(f"Config: max_seq_lens={MAX_SEQ_LENS}, num_sequences={NUM_SEQUENCES}")

    results = run_all()
    save_results(results=results, metadata=metadata)
    print_summary_table(results=results)

    print("\nBenchmark metadata:")
    print(f"  Description: {metadata['description']}")
    print(f"  Timestamp: {metadata['timestamp']}")
    print("  Repeats: 1 (deterministic)")
    print(f"  OS: {metadata['system']['os']} ({metadata['system']['platform']})")
    print(f"  RUSTFLAGS: {metadata['environment']['rustflags']}")

    logger.info("Benchmark suite complete.")


if __name__ == "__main__":
    main()
