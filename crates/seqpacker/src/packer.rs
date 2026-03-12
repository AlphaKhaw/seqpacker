//! Main Packer interface - entry point for users.

use std::time::Instant;

use crate::algorithms;
use crate::error::{PackError, Result};
use crate::metrics::{MetricsBuilder, PackMetrics};
use crate::pack::Pack;
use crate::sequence::Sequence;
use crate::strategy::{PackStrategy, PackingAlgorithm};

/// Configuration for the packer.
#[derive(Clone, Debug)]
pub struct PackerConfig {
    /// Maximum pack length in tokens.
    pub max_pack_length: usize,
    /// Packing strategy to use.
    pub strategy: PackStrategy,
    /// Random seed for shuffle-based algorithms.
    pub seed: Option<u64>,
}

impl Default for PackerConfig {
    fn default() -> Self {
        Self {
            max_pack_length: 2048,
            strategy: PackStrategy::default(),
            seed: None,
        }
    }
}

/// Result of a packing operation: packs + metrics.
#[derive(Debug)]
pub struct PackResult {
    /// Packed bins.
    pub packs: Vec<Pack>,
    /// Packing quality metrics.
    pub metrics: PackMetrics,
}

/// Main packer interface.
///
/// # Example
///
/// ```
/// use seqpacker::{Packer, PackStrategy, Sequence};
///
/// let packer = Packer::new(2048)
///     .with_strategy(PackStrategy::FirstFitDecreasing);
///
/// let sequences = vec![
///     Sequence::new(0, 500),
///     Sequence::new(1, 600),
///     Sequence::new(2, 400),
/// ];
///
/// let result = packer.pack(sequences).unwrap();
/// println!("Packs: {}, Efficiency: {:.2}%",
///     result.metrics.num_packs,
///     result.metrics.efficiency * 100.0);
/// ```
#[derive(Debug)]
pub struct Packer {
    /// Packer configuration.
    pub config: PackerConfig,
}

impl Packer {
    /// Create a new packer with default strategy (FFD).
    pub fn new(max_pack_length: usize) -> Self {
        Self {
            config: PackerConfig {
                max_pack_length,
                ..Default::default()
            },
        }
    }

    /// Create packer with full configuration.
    pub fn with_config(config: PackerConfig) -> Self {
        Self { config }
    }

    /// Set the packing strategy.
    pub fn with_strategy(mut self, strategy: PackStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// Set the random seed (for shuffle-based algorithms).
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.config.seed = Some(seed);
        self
    }

    /// Pack sequences into fixed-length packs.
    ///
    /// # Errors
    ///
    /// - `PackError::EmptyInput` if no sequences provided.
    /// - `PackError::SequenceTooLong` if any sequence exceeds capacity.
    pub fn pack(&self, sequences: Vec<Sequence>) -> Result<PackResult> {
        if sequences.is_empty() {
            return Err(PackError::EmptyInput);
        }

        // Extract num_sequences, total_tokens and capacity.
        let num_sequences = sequences.len();
        let total_tokens: usize = sequences.iter().map(|s| s.length).sum();
        let capacity = self.config.max_pack_length;

        let start = Instant::now();

        let packs = match self.config.strategy {
            PackStrategy::NextFit => algorithms::NextFit.pack(sequences, capacity)?,
            PackStrategy::FirstFit => algorithms::FirstFit.pack(sequences, capacity)?,
            PackStrategy::BestFit => algorithms::BestFit.pack(sequences, capacity)?,
            PackStrategy::WorstFit => algorithms::WorstFit.pack(sequences, capacity)?,
            PackStrategy::FirstFitDecreasing => {
                algorithms::FirstFitDecreasing.pack(sequences, capacity)?
            }
            PackStrategy::BestFitDecreasing => {
                algorithms::BestFitDecreasing.pack(sequences, capacity)?
            }
            PackStrategy::FirstFitShuffle => {
                let seed = self.config.seed.unwrap_or(42);
                algorithms::FirstFitShuffle::new(seed).pack(sequences, capacity)?
            }
            PackStrategy::OptimizedBestFitDecreasing => {
                algorithms::OptimizedBestFitDecreasing.pack(sequences, capacity)?
            }
            PackStrategy::ParallelOptimizedBestFitDecreasing => {
                algorithms::OptimizedBestFitDecreasingParallel.pack(sequences, capacity)?
            }
            PackStrategy::ModifiedFirstFitDecreasing => {
                algorithms::ModifiedFirstFitDecreasing.pack(sequences, capacity)?
            }
            PackStrategy::Harmonic => algorithms::Harmonic::default().pack(sequences, capacity)?,
        };

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        let metrics = MetricsBuilder::new(num_sequences, total_tokens)
            .with_time(elapsed_ms)
            .build(&packs);

        Ok(PackResult { packs, metrics })
    }

    /// Pack from lengths only (no token data) - faster path.
    pub fn pack_lengths(&self, lengths: &[usize]) -> Result<PackResult> {
        let sequences: Vec<Sequence> = lengths
            .iter()
            .enumerate()
            .map(|(id, &length)| Sequence::new(id, length))
            .collect();
        self.pack(sequences)
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn seqs(lens: &[usize]) -> Vec<Sequence> {
        lens.iter()
            .enumerate()
            .map(|(id, &len)| Sequence::new(id, len))
            .collect()
    }

    // ── Construction and builder ──────────────────────────────────

    #[test]
    fn test_new_default_strategy() {
        let p = Packer::new(2048);
        assert_eq!(p.config.max_pack_length, 2048);
        assert_eq!(p.config.strategy, PackStrategy::OptimizedBestFitDecreasing);
        assert_eq!(p.config.seed, None);
    }

    #[test]
    fn test_with_strategy() {
        let p = Packer::new(1024).with_strategy(PackStrategy::NextFit);
        assert_eq!(p.config.strategy, PackStrategy::NextFit);
    }

    #[test]
    fn test_with_seed() {
        let p = Packer::new(1024).with_seed(123);
        assert_eq!(p.config.seed, Some(123));
    }

    #[test]
    fn test_builder_chaining() {
        let p = Packer::new(512)
            .with_strategy(PackStrategy::FirstFitShuffle)
            .with_seed(42);
        assert_eq!(p.config.max_pack_length, 512);
        assert_eq!(p.config.strategy, PackStrategy::FirstFitShuffle);
        assert_eq!(p.config.seed, Some(42));
    }

    #[test]
    fn test_with_config() {
        let config = PackerConfig {
            max_pack_length: 4096,
            strategy: PackStrategy::BestFit,
            seed: Some(99),
        };
        let p = Packer::with_config(config);
        assert_eq!(p.config.max_pack_length, 4096);
        assert_eq!(p.config.strategy, PackStrategy::BestFit);
        assert_eq!(p.config.seed, Some(99));
    }

    #[test]
    fn test_default_config() {
        let config = PackerConfig::default();
        assert_eq!(config.max_pack_length, 2048);
        assert_eq!(config.strategy, PackStrategy::OptimizedBestFitDecreasing);
        assert_eq!(config.seed, None);
    }

    // ── pack() basic behavior ─────────────────────────────────────

    #[test]
    fn test_pack_returns_packs_and_metrics() {
        let p = Packer::new(10);
        let result = p.pack(seqs(&[3, 7, 5, 5])).unwrap();
        assert!(!result.packs.is_empty());
        assert_eq!(result.metrics.num_sequences, 4);
        assert_eq!(result.metrics.total_tokens, 20);
    }

    #[test]
    fn test_pack_metrics_num_packs_matches() {
        let p = Packer::new(10);
        let result = p.pack(seqs(&[3, 7, 5, 5])).unwrap();
        assert_eq!(result.metrics.num_packs, result.packs.len());
    }

    #[test]
    fn test_pack_metrics_efficiency_reasonable() {
        let p = Packer::new(10);
        let result = p.pack(seqs(&[3, 7, 5, 5])).unwrap();
        assert!(result.metrics.efficiency > 0.0);
        assert!(result.metrics.efficiency <= 1.0);
    }

    #[test]
    fn test_pack_metrics_time_nonnegative() {
        let p = Packer::new(10);
        let result = p.pack(seqs(&[3, 7, 5, 5])).unwrap();
        assert!(result.metrics.packing_time_ms >= 0.0);
    }

    // ── Strategy dispatch ─────────────────────────────────────────

    #[test]
    fn test_dispatch_next_fit() {
        let p = Packer::new(10).with_strategy(PackStrategy::NextFit);
        let result = p.pack(seqs(&[3, 7, 5, 5])).unwrap();
        assert!(!result.packs.is_empty());
    }

    #[test]
    fn test_dispatch_first_fit() {
        let p = Packer::new(10).with_strategy(PackStrategy::FirstFit);
        let result = p.pack(seqs(&[3, 7, 5, 5])).unwrap();
        assert!(!result.packs.is_empty());
    }

    #[test]
    fn test_dispatch_best_fit() {
        let p = Packer::new(10).with_strategy(PackStrategy::BestFit);
        let result = p.pack(seqs(&[3, 7, 5, 5])).unwrap();
        assert!(!result.packs.is_empty());
    }

    #[test]
    fn test_dispatch_worst_fit() {
        let p = Packer::new(10).with_strategy(PackStrategy::WorstFit);
        let result = p.pack(seqs(&[3, 7, 5, 5])).unwrap();
        assert!(!result.packs.is_empty());
    }

    #[test]
    fn test_dispatch_ffd() {
        let p = Packer::new(10).with_strategy(PackStrategy::FirstFitDecreasing);
        let result = p.pack(seqs(&[3, 7, 5, 5])).unwrap();
        assert!(!result.packs.is_empty());
    }

    #[test]
    fn test_dispatch_bfd() {
        let p = Packer::new(10).with_strategy(PackStrategy::BestFitDecreasing);
        let result = p.pack(seqs(&[3, 7, 5, 5])).unwrap();
        assert!(!result.packs.is_empty());
    }

    #[test]
    fn test_dispatch_ffs() {
        let p = Packer::new(10)
            .with_strategy(PackStrategy::FirstFitShuffle)
            .with_seed(42);
        let result = p.pack(seqs(&[3, 7, 5, 5])).unwrap();
        assert!(!result.packs.is_empty());
    }

    #[test]
    fn test_ffs_default_seed() {
        // FFS without explicit seed should use default seed=42.
        let p = Packer::new(10).with_strategy(PackStrategy::FirstFitShuffle);
        let result = p.pack(seqs(&[3, 7, 5, 5])).unwrap();
        assert!(!result.packs.is_empty());
    }

    #[test]
    fn test_dispatch_mffd() {
        let p = Packer::new(10).with_strategy(PackStrategy::ModifiedFirstFitDecreasing);
        let result = p.pack(seqs(&[3, 7, 5, 5])).unwrap();
        assert!(!result.packs.is_empty());
    }

    #[test]
    fn test_dispatch_harmonic() {
        let p = Packer::new(10).with_strategy(PackStrategy::Harmonic);
        let result = p.pack(seqs(&[3, 7, 5, 5])).unwrap();
        assert!(!result.packs.is_empty());
    }

    // ── Error cases ───────────────────────────────────────────────

    #[test]
    fn test_empty_input_error() {
        let p = Packer::new(10);
        let result = p.pack(vec![]);
        assert!(matches!(result, Err(PackError::EmptyInput)));
    }

    #[test]
    fn test_oversize_error() {
        let p = Packer::new(10);
        let result = p.pack(seqs(&[11]));
        assert!(matches!(
            result,
            Err(PackError::SequenceTooLong {
                length: 11,
                capacity: 10
            })
        ));
    }

    // ── pack_lengths ──────────────────────────────────────────────

    #[test]
    fn test_pack_lengths_basic() {
        let p = Packer::new(10);
        let result = p.pack_lengths(&[3, 7, 5, 5]).unwrap();
        assert_eq!(result.metrics.num_sequences, 4);
        assert_eq!(result.metrics.total_tokens, 20);
    }

    #[test]
    fn test_pack_lengths_matches_pack() {
        let p = Packer::new(10);
        let from_lengths = p.pack_lengths(&[3, 7, 5, 5]).unwrap();
        let from_seqs = p.pack(seqs(&[3, 7, 5, 5])).unwrap();
        assert_eq!(from_lengths.packs.len(), from_seqs.packs.len());
    }

    #[test]
    fn test_pack_lengths_empty() {
        let p = Packer::new(10);
        let result = p.pack_lengths(&[]);
        assert!(matches!(result, Err(PackError::EmptyInput)));
    }

    // ── All sequences accounted for ───────────────────────────────

    #[test]
    fn test_all_sequences_in_packs() {
        let p = Packer::new(10);
        let result = p.pack(seqs(&[3, 7, 5, 5])).unwrap();
        let total: usize = result.packs.iter().map(|p| p.sequences.len()).sum();
        assert_eq!(total, 4);
    }

    #[test]
    fn test_all_tokens_accounted() {
        let p = Packer::new(10);
        let result = p.pack(seqs(&[3, 7, 5, 5])).unwrap();
        let packed_tokens: usize = result
            .packs
            .iter()
            .flat_map(|p| p.sequences.iter())
            .map(|s| s.length)
            .sum();
        assert_eq!(packed_tokens, 20);
    }
}
