//! Standalone benchmark binary for seqpacker.
//!
//! Runs all 11 implemented packing algorithms (NF, FF, BF, WF, FFD, BFD, FFS, MFFD, OBFD, OBFDP, HK)
//! across 4 synthetic datasets (uniform, log-normal, exponential, bimodal) and
//! 4 capacity values (512, 1024, 2048, 4096), producing a summary table with
//! efficiency, padding, timing, and throughput metrics.
//!
//! The synthetic distributions replicate the Python benchmarks in
//! `benchmarks/python/datasets/synthetic.py` so results are directly comparable.
//!
//! # Usage
//!
//! ```bash
//! # Via Makefile (recommended):
//! make bench-rust
//!
//! # Or directly:
//! cargo run --manifest-path benchmarks/rust/Cargo.toml --release
//! ```
//!
//! # Output
//!
//! Prints a formatted table with 176 rows (4 datasets x 4 capacities x 11 algorithms):
//!
//! ```text
//! Dataset              Capacity  Algo       Bins    Eff%    Pad%   Avg/Pack    Time(ms)    Throughput
//! ---------------------------------------------------------------------------------------------------------
//! uniform                   512  NF         6891   75.09   24.91       1.45       1.180       8471288
//! ...
//! ```

mod dataset;
mod report;

use seqpacker::{PackStrategy, Packer};

use crate::dataset::SyntheticDataGenerator;
use crate::report::print_header;

/// Number of sequences generated per synthetic dataset.
const NUM_SEQUENCES: usize = 10_000;

/// Bin capacities to benchmark against.
const MAX_SEQ_LENS: [usize; 4] = [512, 1024, 2048, 4096];

/// Random seed for deterministic dataset generation.
const SEED: u64 = 42;

/// All 11 implemented algorithms to benchmark.
const STRATEGIES: [PackStrategy; 11] = [
    PackStrategy::NextFit,
    PackStrategy::FirstFit,
    PackStrategy::BestFit,
    PackStrategy::WorstFit,
    PackStrategy::FirstFitDecreasing,
    PackStrategy::BestFitDecreasing,
    PackStrategy::FirstFitShuffle,
    PackStrategy::ModifiedFirstFitDecreasing,
    PackStrategy::OptimizedBestFitDecreasing,
    PackStrategy::ParallelOptimizedBestFitDecreasing,
    PackStrategy::Harmonic,
];

fn main() {
    let datasets = [
        (
            "uniform",
            SyntheticDataGenerator::uniform(NUM_SEQUENCES, 16, 512, SEED),
        ),
        (
            "lognormal",
            SyntheticDataGenerator::lognormal(NUM_SEQUENCES, 128.0, 64.0, SEED),
        ),
        (
            "exponential",
            SyntheticDataGenerator::exponential(NUM_SEQUENCES, 128.0, SEED),
        ),
        (
            "bimodal",
            SyntheticDataGenerator::bimodal(NUM_SEQUENCES, 64.0, 512.0, 0.7, SEED),
        ),
    ];

    print_header();

    for (name, lengths) in &datasets {
        for &capacity in &MAX_SEQ_LENS {
            // Clamp lengths to capacity (skip sequences that exceed it after clamping to 1).
            let clamped: Vec<usize> = lengths.iter().map(|&l| l.min(capacity).max(1)).collect();

            for &strategy in &STRATEGIES {
                let packer = Packer::new(capacity)
                    .with_strategy(strategy)
                    .with_seed(SEED);

                let result = packer.pack_lengths(&clamped);

                match result {
                    Ok(r) => {
                        let m = &r.metrics;
                        println!(
                            "{:<20} {:>8}  {:<8} {:>6}  {:>6.2}  {:>6.2}  {:>9.2}  {:>10.3}  {:>12.0}",
                            name,
                            capacity,
                            strategy.short_name().unwrap_or(strategy.name()),
                            m.num_packs,
                            m.efficiency * 100.0,
                            m.padding_ratio() * 100.0,
                            m.avg_sequences_per_pack,
                            m.packing_time_ms,
                            m.throughput(),
                        );
                    }
                    Err(e) => {
                        println!(
                            "{:<20} {:>8}  {:<8} ERROR: {}",
                            name,
                            capacity,
                            strategy.short_name().unwrap_or(strategy.name()),
                            e,
                        );
                    }
                }
            }
        }
    }
}
