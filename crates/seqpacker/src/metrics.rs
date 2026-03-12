//! Packing quality metrics.

use serde::{Deserialize, Serialize};

use crate::pack::Pack;

/// Metrics from a packing operation.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PackMetrics {
    /// Total number of sequences packed.
    pub num_sequences: usize,
    /// Total number of packs generated.
    pub num_packs: usize,
    /// Total tokens across all sequences.
    pub total_tokens: usize,
    /// Total padding tokens added.
    pub padding_tokens: usize,
    /// Packing efficiency: total_tokens / total_capacity (0.0 - 1.0).
    pub efficiency: f64,
    /// Average utilisation per pack.
    pub avg_utilisation: f64,
    /// Standard deviation of pack utilisation.
    pub utilisation_std: f64,
    /// Minimum pack utilisation.
    pub min_utilisation: f64,
    /// Maximum pack utilisation.
    pub max_utilisation: f64,
    /// Average sequences per pack.
    pub avg_sequences_per_pack: f64,
    /// Packing time in milliseconds.
    pub packing_time_ms: f64,
}

impl PackMetrics {
    /// Padding ratio: padding_tokens / total_capacity (0.0 - 1.0).
    /// Lower is better. Equal to `1.0 - efficiency`.
    pub fn padding_ratio(&self) -> f64 {
        let total = self.total_tokens + self.padding_tokens;
        if total == 0 {
            0.0
        } else {
            self.padding_tokens as f64 / total as f64
        }
    }

    /// Throughput in sequences per second.
    pub fn throughput(&self) -> f64 {
        if self.packing_time_ms == 0.0 {
            f64::INFINITY
        } else {
            self.num_sequences as f64 / (self.packing_time_ms / 1000.0)
        }
    }
}

/// Builder for computing `PackMetrics` from a set of packs.
pub struct MetricsBuilder {
    num_sequences: usize,
    total_tokens: usize,
    packing_time_ms: f64,
}

impl MetricsBuilder {
    /// Create a new metrics builder.
    pub fn new(num_sequences: usize, total_tokens: usize) -> Self {
        Self {
            num_sequences,
            total_tokens,
            packing_time_ms: 0.0,
        }
    }

    /// Set the packing time in milliseconds.
    pub fn with_time(mut self, time_ms: f64) -> Self {
        self.packing_time_ms = time_ms;
        self
    }

    /// Build metrics from the given packs.
    pub fn build(self, packs: &[Pack]) -> PackMetrics {
        let num_packs = packs.len();
        let padding_tokens: usize = packs.iter().map(|p| p.padding_tokens()).sum();

        // Compute `total_capacity` for efficiency computation.
        let total_capacity: usize = packs.iter().map(|p| p.capacity).sum();
        let efficiency = if total_capacity == 0 {
            0.0
        } else {
            self.total_tokens as f64 / total_capacity as f64
        };

        // Compute `utilisations` for avg, std, min, and max utilisation metrics.
        let utilisations: Vec<f64> = packs.iter().map(|p| p.utilisation()).collect();
        let avg_utilisation = if utilisations.is_empty() {
            0.0
        } else {
            utilisations.iter().sum::<f64>() / utilisations.len() as f64
        };
        let utilisation_std = if utilisations.len() < 2 {
            0.0
        } else {
            let variance = utilisations
                .iter()
                .map(|&u| (u - avg_utilisation).powi(2))
                .sum::<f64>()
                / utilisations.len() as f64;
            variance.sqrt()
        };
        let min_utilisation = utilisations
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);
        let max_utilisation = utilisations
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        // Compute `total_seqs` for average sequences per pack.
        let total_seqs: usize = packs.iter().map(|p| p.len()).sum();
        let avg_sequences_per_pack = if num_packs == 0 {
            0.0
        } else {
            total_seqs as f64 / num_packs as f64
        };

        PackMetrics {
            num_sequences: self.num_sequences,
            num_packs,
            total_tokens: self.total_tokens,
            padding_tokens,
            efficiency,
            avg_utilisation,
            utilisation_std,
            min_utilisation,
            max_utilisation,
            avg_sequences_per_pack,
            packing_time_ms: self.packing_time_ms,
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pack::Pack;
    use crate::sequence::Sequence;

    /// Helper: build a pack with given sequence lengths.
    fn make_pack(capacity: usize, lengths: &[usize]) -> Pack {
        let mut pack = Pack::new(capacity);
        for (i, &len) in lengths.iter().enumerate() {
            pack.add(Sequence::new(i, len)).unwrap();
        }
        pack
    }

    // ── PackMetrics::padding_ratio ────────────────────────────────────

    #[test]
    fn test_padding_ratio_zero_padding() {
        let packs = vec![make_pack(100, &[100])];
        let m = MetricsBuilder::new(1, 100).build(&packs);
        assert!((m.padding_ratio() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_padding_ratio_half_padding() {
        let packs = vec![make_pack(100, &[50])];
        let m = MetricsBuilder::new(1, 50).build(&packs);
        assert!((m.padding_ratio() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_padding_ratio_empty() {
        let packs: Vec<Pack> = vec![];
        let m = MetricsBuilder::new(0, 0).build(&packs);
        assert!((m.padding_ratio() - 0.0).abs() < f64::EPSILON);
    }

    // ── PackMetrics::throughput ────────────────────────────────────────

    #[test]
    fn test_throughput_normal() {
        let packs = vec![make_pack(100, &[100])];
        let m = MetricsBuilder::new(1000, 100)
            .with_time(500.0)
            .build(&packs);
        // 1000 seqs / 0.5 sec = 2000 seq/sec
        assert!((m.throughput() - 2000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_throughput_zero_time() {
        let packs = vec![make_pack(100, &[100])];
        let m = MetricsBuilder::new(10, 100).build(&packs);
        assert!(m.throughput().is_infinite());
    }

    // ── MetricsBuilder defaults and chaining ──────────────────────────

    #[test]
    fn test_builder_defaults() {
        let packs = vec![make_pack(100, &[60])];
        let m = MetricsBuilder::new(5, 60).build(&packs);
        assert_eq!(m.num_sequences, 5);
        assert_eq!(m.total_tokens, 60);
        assert_eq!(m.packing_time_ms, 0.0);
    }

    #[test]
    fn test_builder_with_time() {
        let packs = vec![make_pack(100, &[100])];
        let m = MetricsBuilder::new(1, 100).with_time(42.5).build(&packs);
        assert!((m.packing_time_ms - 42.5).abs() < f64::EPSILON);
    }

    // ── build() computed fields ───────────────────────────────────────

    #[test]
    fn test_build_num_packs() {
        let packs = vec![make_pack(100, &[60, 30]), make_pack(100, &[50])];
        let m = MetricsBuilder::new(3, 140).build(&packs);
        assert_eq!(m.num_packs, 2);
    }

    #[test]
    fn test_build_padding_tokens() {
        // Pack 1: capacity 100, used 90 → padding 10
        // Pack 2: capacity 100, used 50 → padding 50
        let packs = vec![make_pack(100, &[60, 30]), make_pack(100, &[50])];
        let m = MetricsBuilder::new(3, 140).build(&packs);
        assert_eq!(m.padding_tokens, 60);
    }

    #[test]
    fn test_build_efficiency() {
        // total_tokens=140, total_capacity=200 → 0.7
        let packs = vec![make_pack(100, &[60, 30]), make_pack(100, &[50])];
        let m = MetricsBuilder::new(3, 140).build(&packs);
        assert!((m.efficiency - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_build_efficiency_empty() {
        let packs: Vec<Pack> = vec![];
        let m = MetricsBuilder::new(0, 0).build(&packs);
        assert!((m.efficiency - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_build_avg_utilisation() {
        // Pack 1: 90/100 = 0.9, Pack 2: 50/100 = 0.5 → avg = 0.7
        let packs = vec![make_pack(100, &[60, 30]), make_pack(100, &[50])];
        let m = MetricsBuilder::new(3, 140).build(&packs);
        assert!((m.avg_utilisation - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_build_utilisation_std() {
        // utilisations: [0.9, 0.5], mean=0.7
        // variance = ((0.2)² + (0.2)²) / 2 = 0.04
        // std = 0.2
        let packs = vec![make_pack(100, &[60, 30]), make_pack(100, &[50])];
        let m = MetricsBuilder::new(3, 140).build(&packs);
        assert!((m.utilisation_std - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_build_utilisation_std_single_pack() {
        let packs = vec![make_pack(100, &[75])];
        let m = MetricsBuilder::new(1, 75).build(&packs);
        assert!((m.utilisation_std - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_build_min_max_utilisation() {
        let packs = vec![
            make_pack(100, &[90]), // 0.9
            make_pack(100, &[50]), // 0.5
            make_pack(100, &[70]), // 0.7
        ];
        let m = MetricsBuilder::new(3, 210).build(&packs);
        assert!((m.min_utilisation - 0.5).abs() < f64::EPSILON);
        assert!((m.max_utilisation - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_build_avg_sequences_per_pack() {
        // Pack 1: 2 seqs, Pack 2: 1 seq → avg = 1.5
        let packs = vec![make_pack(100, &[60, 30]), make_pack(100, &[50])];
        let m = MetricsBuilder::new(3, 140).build(&packs);
        assert!((m.avg_sequences_per_pack - 1.5).abs() < f64::EPSILON);
    }

    // ── Consistency: padding_ratio ≈ 1.0 - efficiency ─────────────────

    #[test]
    fn test_padding_ratio_plus_efficiency_equals_one() {
        let packs = vec![make_pack(100, &[60, 30]), make_pack(100, &[50])];
        let m = MetricsBuilder::new(3, 140).build(&packs);
        assert!((m.padding_ratio() + m.efficiency - 1.0).abs() < 1e-10);
    }
}
