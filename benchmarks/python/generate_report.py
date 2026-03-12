"""
Generate benchmark visualization report from results JSON.

Usage: uv run python -m benchmarks.python.generate_report
"""

import json
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd

from benchmarks.python.utils.logging import logger

RESULTS_DIR = Path("benchmarks/results")
CHARTS_DIR = RESULTS_DIR / "charts"


def load_results() -> tuple[pd.DataFrame, dict | None]:
    """
    Load benchmark results from JSON.

    Returns:
        tuple[pd.DataFrame, dict | None]: Results as a DataFrame and metadata
                                          (if present).

    Raises:
        FileNotFoundError: If all_results.json doesn't exist.
    """
    results_path = RESULTS_DIR / "all_results.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"{results_path} not found. Run benchmarks first: make bench-python"
        )
    data = json.loads(results_path.read_text())

    # Handle both old format (list) and new format (dict with metadata)
    if isinstance(data, dict) and "results" in data:
        metadata = data.get("metadata")
        results_list = data["results"]
    else:
        metadata = None
        results_list = data

    rows = []
    for r in results_list:
        rows.append(
            {
                "runner": r["runner_name"],
                "dataset": r["dataset_name"],
                "max_seq_len": r["max_seq_len"],
                "efficiency": r["efficiency"]["packing_efficiency"],
                "padding_ratio": r["efficiency"]["padding_ratio"],
                "time_ms": r["performance"]["packing_time_ms"],
                "throughput": r["performance"]["throughput_seqs_per_sec"],
                "memory_mb": r["performance"]["memory_peak_mb"],
                "num_packs": r["pack_stats"]["num_packs"],
                "seqs_per_pack_mean": r["pack_stats"]["sequences_per_pack_mean"],
            }
        )
    return pd.DataFrame(rows), metadata


def plot_efficiency_comparison(df: pd.DataFrame) -> None:
    """
    Create grouped bar chart comparing packing efficiency across runners.

    Args:
        df (pd.DataFrame): Benchmark results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Packing Efficiency by Runner and Dataset", fontsize=14)

    for ax, max_len in zip(axes.flat, sorted(df["max_seq_len"].unique())):
        subset = df[df["max_seq_len"] == max_len]
        pivot = subset.pivot_table(
            index="dataset", columns="runner", values="efficiency"
        )
        pivot.plot(kind="bar", ax=ax, rot=25)
        ax.set_title(f"max_seq_len = {max_len}")
        ax.set_ylabel("Packing Efficiency")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "efficiency_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved efficiency_comparison.png")


def plot_performance_comparison(df: pd.DataFrame) -> None:
    """
    Create log-scale bar chart comparing packing time.

    Args:
        df (pd.DataFrame): Benchmark results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Packing Time by Runner and Dataset (log scale)", fontsize=14)

    for ax, max_len in zip(axes.flat, sorted(df["max_seq_len"].unique())):
        subset = df[df["max_seq_len"] == max_len]
        pivot = subset.pivot_table(index="dataset", columns="runner", values="time_ms")
        pivot.plot(kind="bar", ax=ax, rot=25, logy=True)
        ax.set_title(f"max_seq_len = {max_len}")
        ax.set_ylabel("Time (ms, log scale)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "performance_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved performance_comparison.png")


def plot_dataset_distributions() -> None:
    """
    Plot sequence length distributions for synthetic datasets.
    """
    from benchmarks.python.datasets.synthetic import SyntheticDataGenerator

    datasets = [
        SyntheticDataGenerator.uniform(n=10000, seed=42),
        SyntheticDataGenerator.lognormal(n=10000, seed=42),
        SyntheticDataGenerator.exponential(n=10000, seed=42),
        SyntheticDataGenerator.bimodal(n=10000, seed=42),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Synthetic Dataset Length Distributions", fontsize=14)

    for ax, ds in zip(axes.flat, datasets):
        ax.hist(ds.lengths, bins=50, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.set_title(ds.name.split("_seed")[0])
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Count")
        ax.axvline(
            ds.mean_length,
            color="red",
            linestyle="--",
            label=f"mean={ds.mean_length:.0f}",
        )
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "dataset_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved dataset_distributions.png")


def generate_summary_markdown(df: pd.DataFrame, metadata: dict | None = None) -> None:
    """
    Generate summary.md with readable comparison tables.

    Args:
        df (pd.DataFrame): Benchmark results.
        metadata (dict | None): Benchmark run metadata.
    """
    lines = ["# Benchmark Results Summary\n"]

    # Include metadata if present
    if metadata:
        lines.append("\n## Benchmark Metadata\n")
        lines.append(f"- **Description:** {metadata.get('description', 'N/A')}")
        lines.append(f"- **Timestamp:** {metadata.get('timestamp', 'N/A')}")
        if "system" in metadata:
            sys_info = metadata["system"]
            lines.append(
                f"- **OS:** {sys_info.get('os', 'N/A')} "
                f"({sys_info.get('platform', 'N/A')})"
            )
            lines.append(f"- **Machine:** {sys_info.get('machine', 'N/A')}")
            lines.append(f"- **Python:** {sys_info.get('python_version', 'N/A')}")
        if "environment" in metadata:
            env_info = metadata["environment"]
            lines.append(f"- **RUSTFLAGS:** `{env_info.get('rustflags', 'N/A')}`")
        lines.append("")

    # --- Efficiency comparison pivot (runner x dataset, per max_seq_len) ---
    lines.append("\n## Packing Efficiency by Runner\n")
    for max_len in sorted(df["max_seq_len"].unique()):
        subset = cast(pd.DataFrame, df[df["max_seq_len"] == max_len])
        pivot = subset.pivot_table(
            index="dataset", columns="runner", values="efficiency"
        )
        lines.append(f"\n### max_seq_len = {max_len}\n")
        # Shorten dataset names for readability
        pivot.index = [d.split("_seed")[0] if "_seed" in d else d for d in pivot.index]
        lines.append(pivot.map(lambda x: f"{x:.2%}").to_markdown())

    # --- Speed comparison pivot ---
    lines.append("\n\n## Packing Time (ms) by Runner\n")
    for max_len in sorted(df["max_seq_len"].unique()):
        subset = cast(pd.DataFrame, df[df["max_seq_len"] == max_len])
        pivot = subset.pivot_table(index="dataset", columns="runner", values="time_ms")
        lines.append(f"\n### max_seq_len = {max_len}\n")
        pivot.index = [d.split("_seed")[0] if "_seed" in d else d for d in pivot.index]
        lines.append(pivot.map(lambda x: f"{x:.1f}").to_markdown())

    # --- Bin count comparison pivot ---
    lines.append("\n\n## Bin Count by Runner\n")
    for max_len in sorted(df["max_seq_len"].unique()):
        subset = cast(pd.DataFrame, df[df["max_seq_len"] == max_len])
        pivot = subset.pivot_table(
            index="dataset", columns="runner", values="num_packs"
        )
        lines.append(f"\n### max_seq_len = {max_len}\n")
        pivot.index = [d.split("_seed")[0] if "_seed" in d else d for d in pivot.index]
        lines.append(pivot.map(lambda x: f"{int(x)}").to_markdown())

    # --- Overall averages ---
    lines.append("\n\n## Overall Runner Averages\n")
    avg = cast(
        pd.DataFrame,
        df[df["runner"] != "naive_padding"]
        .groupby("runner")
        .agg(
            avg_efficiency=("efficiency", "mean"),
            avg_time_ms=("time_ms", "mean"),
            avg_bins=("num_packs", "mean"),
        ),
    ).sort_values(by="avg_efficiency", ascending=False)
    lines.append("| Runner | Avg Efficiency | Avg Time (ms) | Avg Bins |")
    lines.append("|--------|---------------|--------------|----------|")
    for runner, row in avg.iterrows():
        lines.append(
            f"| {runner} | {row['avg_efficiency']:.2%} | "
            f"{row['avg_time_ms']:.0f} | {row['avg_bins']:.0f} |"
        )

    # --- Full detail table ---
    lines.append("\n\n## Full Results\n")
    for max_len in sorted(df["max_seq_len"].unique()):
        subset = cast(pd.DataFrame, df[df["max_seq_len"] == max_len])
        lines.append(f"\n### max_seq_len = {max_len}\n")
        lines.append(
            "| Runner | Dataset | Efficiency | Bins | Time (ms) | Throughput (seq/s) |"
        )
        lines.append(
            "|--------|---------|-----------|------|-----------|-------------------|"
        )
        for _, row in subset.sort_values(by=["dataset", "runner"]).iterrows():
            lines.append(
                f"| {row['runner']} | {row['dataset']} | "
                f"{row['efficiency']:.4f} | {row['num_packs']} | "
                f"{row['time_ms']:.1f} | {row['throughput']:.0f} |"
            )

    summary_path = RESULTS_DIR / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n")
    logger.info(f"Saved summary to {summary_path}")


def main() -> None:
    """
    Main entry point for report generation.
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading benchmark results...")
    df, metadata = load_results()

    if metadata:
        logger.info(f"Benchmark: {metadata.get('description', 'N/A')}")
        logger.info(f"Timestamp: {metadata.get('timestamp', 'N/A')}")

    logger.info("Generating charts...")
    plot_efficiency_comparison(df=df)
    plot_performance_comparison(df=df)
    plot_dataset_distributions()

    logger.info("Generating summary markdown...")
    generate_summary_markdown(df=df, metadata=metadata)

    logger.info("Report generation complete.")


if __name__ == "__main__":
    main()
