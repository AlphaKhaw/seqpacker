"""
Standalone benchmark comparing all seqpacker algorithms.

Runs every implemented strategy across synthetic and real-world datasets,
prints a comparison table, identifies the best strategy per dataset, and
optionally compares against LightBinPack if installed.

Usage: uv run python -m benchmarks.python.bench_seqpacker
"""

import json
import os
import platform
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from seqpacker import Packer

from benchmarks.python.datasets.real import RealDataLoader
from benchmarks.python.datasets.synthetic import DatasetInfo, SyntheticDataGenerator
from benchmarks.python.runners.lightbinpack import LightBinPackRunner
from benchmarks.python.utils.logging import logger

RESULTS_DIR: Path = Path("benchmarks/results")
MAX_SEQ_LENS: list[int] = [512, 1024, 2048, 4096, 8192]
NUM_SEQUENCES: int = 10_000

# All implemented strategies
STRATEGIES: list[str] = [
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
]


@dataclass
class StrategyResult:
    """
    Result of a single strategy x dataset x capacity run.
    """

    strategy: str
    dataset: str
    capacity: int
    num_sequences: int
    num_bins: int
    efficiency: float
    padding_ratio: float
    time_ms: float
    throughput: float
    avg_utilisation: float


def get_synthetic_datasets() -> list[DatasetInfo]:
    """
    Generate all synthetic benchmark datasets.

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


def get_real_datasets() -> list[DatasetInfo]:
    """
    Load real-world datasets from HuggingFace.

    Tokenizes on first run, cached afterwards. Failures are logged
    and skipped so the benchmark continues.

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


def bench_strategy(
    strategy: str,
    lengths: list[int],
    capacity: int,
    dataset_name: str,
) -> StrategyResult:
    """
    Benchmark a single strategy on one dataset/capacity combination.

    Args:
        strategy (str): Algorithm short name.
        lengths (list[int]): Sequence lengths to pack.
        capacity (int): Maximum bin capacity.
        dataset_name (str): Name of the dataset for labelling.

    Returns:
        StrategyResult: Benchmark result for this run.
    """
    packer = Packer(capacity=capacity, strategy=strategy, seed=42)
    start = time.perf_counter()
    result = packer.pack(lengths)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return StrategyResult(
        strategy=strategy,
        dataset=dataset_name,
        capacity=capacity,
        num_sequences=result.metrics.num_sequences,
        num_bins=result.num_bins,
        efficiency=result.efficiency,
        padding_ratio=result.metrics.padding_ratio,
        time_ms=elapsed_ms,
        throughput=result.metrics.throughput,
        avg_utilisation=result.metrics.avg_utilisation,
    )


LIGHTBINPACK_STRATEGIES: list[str] = ["ffd", "bfd", "obfd", "obfdp"]


def bench_lightbinpack(
    lengths: list[int],
    capacity: int,
    dataset_name: str,
    strategy: str = "ffd",
) -> StrategyResult | None:
    """
    Benchmark a LightBinPack strategy for comparison if available.

    Args:
        lengths (list[int]): Sequence lengths to pack.
        capacity (int): Maximum bin capacity.
        dataset_name (str): Name of the dataset for labelling.
        strategy (str): LightBinPack strategy (ffd, obfd, obfdp).

    Returns:
        StrategyResult | None: Result if LightBinPack is available, else None.
    """
    runner = LightBinPackRunner(strategy=strategy)
    if not runner.available:
        return None

    start = time.perf_counter()
    bins = runner.pack(lengths=lengths, max_seq_len=capacity)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    total_tokens = sum(sum(b) for b in bins)
    total_capacity = len(bins) * capacity
    efficiency = total_tokens / total_capacity if total_capacity > 0 else 0.0

    strategy_label = "lightbinpack" if strategy == "ffd" else f"lightbinpack_{strategy}"

    return StrategyResult(
        strategy=strategy_label,
        dataset=dataset_name,
        capacity=capacity,
        num_sequences=sum(len(b) for b in bins),
        num_bins=len(bins),
        efficiency=efficiency,
        padding_ratio=1.0 - efficiency,
        time_ms=elapsed_ms,
        throughput=len(lengths) / elapsed_ms if elapsed_ms > 0 else 0.0,
        avg_utilisation=efficiency,
    )


def print_comparison_table(results: list[StrategyResult]) -> None:
    """
    Print a formatted comparison table grouped by dataset and capacity.

    Args:
        results (list[StrategyResult]): All benchmark results.
    """
    # Group by (dataset, capacity)
    groups: dict[tuple[str, int], list[StrategyResult]] = {}
    for r in results:
        key = (r.dataset, r.capacity)
        groups.setdefault(key, []).append(r)

    header = (
        f"{'Strategy':<14} {'Bins':>6} {'Eff%':>7} {'Pad%':>7} "
        f"{'AvgUtil':>8} {'Time(ms)':>10} {'Tput(seq/ms)':>13}"
    )
    sep = "-" * len(header)

    for (dataset, capacity), group in sorted(groups.items()):
        print(f"\n{'=' * len(header)}")
        print(f"Dataset: {dataset}  |  Capacity: {capacity}")
        print(f"{'=' * len(header)}")
        print(header)
        print(sep)

        # Sort by efficiency descending, then by time ascending
        sorted_group = sorted(group, key=lambda r: (-r.efficiency, r.time_ms))
        best_eff = sorted_group[0].efficiency

        for r in sorted_group:
            marker = " *" if r.efficiency == best_eff else ""
            print(
                f"{r.strategy:<14} {r.num_bins:>6} {r.efficiency * 100:>6.2f}% "
                f"{r.padding_ratio * 100:>6.2f}% {r.avg_utilisation * 100:>7.2f}% "
                f"{r.time_ms:>10.2f} {r.throughput:>13.1f}{marker}"
            )

        print(sep)


def print_best_per_dataset(results: list[StrategyResult]) -> None:
    """
    Print the best seqpacker strategy per dataset/capacity.

    Args:
        results (list[StrategyResult]): All benchmark results.
    """
    # Filter out lightbinpack for "best seqpacker" comparison
    seqpacker_results = [
        r for r in results if not r.strategy.startswith("lightbinpack")
    ]

    groups: dict[tuple[str, int], list[StrategyResult]] = {}
    for r in seqpacker_results:
        key = (r.dataset, r.capacity)
        groups.setdefault(key, []).append(r)

    print(f"\n{'=' * 80}")
    print("BEST SEQPACKER STRATEGY PER DATASET/CAPACITY")
    print(f"{'=' * 80}")
    print(f"{'Dataset':<45} {'Cap':>5} {'Best':<8} {'Eff%':>7} {'Time(ms)':>10}")
    print("-" * 80)

    for (dataset, capacity), group in sorted(groups.items()):
        best = max(group, key=lambda r: (r.efficiency, -r.time_ms))
        print(
            f"{dataset:<45} {capacity:>5} {best.strategy:<8} "
            f"{best.efficiency * 100:>6.2f}% {best.time_ms:>10.2f}"
        )

    print("=" * 80)


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


def save_results(results: list[StrategyResult], metadata: dict) -> None:
    """
    Save benchmark results to JSON.

    Saves to both:
    - seqpacker_algorithms.json (latest run, overwritten)
    - seqpacker_history.json (accumulates all runs)

    Args:
        results (list[StrategyResult]): All benchmark results.
        metadata (dict): Benchmark run metadata.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_data = {
        "metadata": metadata,
        "results": [
            {
                "strategy": r.strategy,
                "dataset": r.dataset,
                "capacity": r.capacity,
                "num_sequences": r.num_sequences,
                "num_bins": r.num_bins,
                "efficiency": r.efficiency,
                "padding_ratio": r.padding_ratio,
                "time_ms": r.time_ms,
                "throughput": r.throughput,
                "avg_utilisation": r.avg_utilisation,
            }
            for r in results
        ],
    }

    # Save latest run (overwrites)
    latest_path = RESULTS_DIR / "seqpacker_algorithms.json"
    latest_path.write_text(json.dumps(run_data, indent=2))
    logger.info(f"Saved latest run to {latest_path}")

    # Append to history file
    history_path = RESULTS_DIR / "seqpacker_history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
    else:
        history = {"runs": []}

    history["runs"].append(run_data)
    history_path.write_text(json.dumps(history, indent=2))
    logger.info(
        f"Appended to history ({len(history['runs'])} total runs) at {history_path}"
    )


def prompt_for_description() -> str:
    """
    Prompt user for benchmark description.

    Returns:
        str: User-provided description.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK CONFIGURATION")
    print("=" * 70)
    print("Please provide a description for this benchmark run.")
    print("Examples:")
    print("  - 'Baseline release build on Ubuntu 22.04'")
    print("  - 'With RUSTFLAGS=-C target-cpu=native on macOS M2'")
    print("  - 'After SmallVec optimization'")
    print("  - 'PGO-optimized build'")
    print("=" * 70)
    description = input("\nDescription: ").strip()
    if not description:
        description = "No description provided"
    print()
    return description


def main() -> None:
    """
    Run all seqpacker strategies, compare, and optionally benchmark LightBinPack.
    """
    description = prompt_for_description()
    metadata = get_benchmark_metadata(description)

    logger.info(f"Starting benchmark: {description}")
    logger.info(f"System: {metadata['system']['platform']}")
    logger.info(f"RUSTFLAGS: {metadata['environment']['rustflags']}")

    synthetic = get_synthetic_datasets()
    real = get_real_datasets()
    datasets = synthetic + real
    all_results: list[StrategyResult] = []

    logger.info(
        f"Benchmarking {len(STRATEGIES)} seqpacker strategies "
        f"x {len(datasets)} datasets ({len(synthetic)} synthetic + "
        f"{len(real)} real) x {len(MAX_SEQ_LENS)} capacities"
    )

    for dataset in datasets:
        for capacity in MAX_SEQ_LENS:
            valid_lengths = [length for length in dataset.lengths if length <= capacity]
            if not valid_lengths:
                logger.warning(
                    f"No valid sequences for {dataset.name} with capacity={capacity}"
                )
                continue

            # Benchmark all seqpacker strategies
            for strategy in STRATEGIES:
                logger.info(
                    f"  seqpacker_{strategy} on {dataset.name} "
                    f"(cap={capacity}, n={len(valid_lengths)})"
                )
                result = bench_strategy(
                    strategy=strategy,
                    lengths=valid_lengths,
                    capacity=capacity,
                    dataset_name=dataset.name,
                )
                all_results.append(result)

            # Benchmark LightBinPack strategies for comparison
            for lbp_strategy in LIGHTBINPACK_STRATEGIES:
                logger.info(
                    f"  lightbinpack_{lbp_strategy} on {dataset.name} "
                    f"(cap={capacity}, n={len(valid_lengths)})"
                )
                lbp_result = bench_lightbinpack(
                    lengths=valid_lengths,
                    capacity=capacity,
                    dataset_name=dataset.name,
                    strategy=lbp_strategy,
                )
                if lbp_result is not None:
                    all_results.append(lbp_result)

    print_comparison_table(results=all_results)
    print_best_per_dataset(results=all_results)
    save_results(results=all_results, metadata=metadata)

    print("\nBenchmark metadata:")
    print(f"  Description: {metadata['description']}")
    print(f"  Timestamp: {metadata['timestamp']}")
    print(f"  OS: {metadata['system']['os']} ({metadata['system']['platform']})")
    print(f"  RUSTFLAGS: {metadata['environment']['rustflags']}")

    logger.info("Seqpacker algorithm benchmark complete.")


if __name__ == "__main__":
    main()
