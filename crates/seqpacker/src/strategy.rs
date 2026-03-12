//! Packing strategy enum and algorithm trait.

use crate::error::Result;
use crate::pack::Pack;
use crate::sequence::Sequence;

/// Packing algorithm strategy.
///
/// Used to select which algorithm the `Packer` dispatches to.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PackStrategy {
    // ── Classical greedy (use greedy_pack engine) ─────────────────
    /// Next Fit: O(n), single open bin.
    /// Worst ratio (2.0) but fastest and lowest memory.
    NextFit,
    /// First Fit: O(n log B), leftmost bin with space.
    /// Online baseline with 1.7 approx ratio.
    FirstFit,
    /// Best Fit: O(n log B), tightest fitting bin.
    /// Slightly better than FF in practice.
    BestFit,
    /// Worst Fit: O(n log B), loosest fitting bin.
    /// Spreads items evenly across bins.
    WorstFit,
    /// First Fit Decreasing: sort desc + FF. 11/9 ≈ 1.22 ratio.
    /// Best general-purpose offline algorithm.
    FirstFitDecreasing,
    /// Best Fit Decreasing: sort desc + BF. Same ratio as FFD.
    BestFitDecreasing,
    /// First Fit Shuffle: shuffle with seed + FF.
    /// Preserves training randomness (used by NeMo).
    FirstFitShuffle,

    // ── Advanced classical ────────────────────────────────────────
    /// Modified First Fit Decreasing: 5-phase with size classes (>1/2, >1/3, etc.).
    /// Tighter worst-case bound than FFD: 71/60 ≈ 1.183.
    ModifiedFirstFitDecreasing,

    // ── OBFD family (capacity-indexed segment tree) ───────────────
    /// Optimized Best-Fit Decreasing: counting sort + capacity segment tree.
    /// O(N log L) where L = max_length. Primary target to beat baseline.
    #[default]
    OptimizedBestFitDecreasing,
    /// Parallel Optimized Best-Fit Decreasing: splits input across threads,
    /// repacks partial bins. Uses Rayon for data-parallel packing.
    ParallelOptimizedBestFitDecreasing,

    // ── Theoretical ───────────────────────────────────────────────
    /// Harmonic-k: classifies items by size into k classes.
    /// Near-optimal asymptotic ratio.
    Harmonic,
}

impl PackStrategy {
    /// Human-readable name for display and logging.
    pub fn name(&self) -> &'static str {
        match self {
            Self::NextFit => "NextFit",
            Self::FirstFit => "FirstFit",
            Self::BestFit => "BestFit",
            Self::WorstFit => "WorstFit",
            Self::FirstFitDecreasing => "FirstFitDecreasing",
            Self::BestFitDecreasing => "BestFitDecreasing",
            Self::FirstFitShuffle => "FirstFitShuffle",
            Self::ModifiedFirstFitDecreasing => "MFFD",
            Self::OptimizedBestFitDecreasing => "OBFD",
            Self::ParallelOptimizedBestFitDecreasing => "OBFDP",
            Self::Harmonic => "Harmonic",
        }
    }

    /// Short-form alias for API usage.
    ///
    /// Returns `None` for strategies that don't have a common abbreviation.
    pub fn short_name(&self) -> Option<&'static str> {
        match self {
            Self::NextFit => Some("NF"),
            Self::FirstFit => Some("FF"),
            Self::BestFit => Some("BF"),
            Self::WorstFit => Some("WF"),
            Self::FirstFitDecreasing => Some("FFD"),
            Self::BestFitDecreasing => Some("BFD"),
            Self::FirstFitShuffle => Some("FFS"),
            Self::ModifiedFirstFitDecreasing => Some("MFFD"),
            Self::OptimizedBestFitDecreasing => Some("OBFD"),
            Self::ParallelOptimizedBestFitDecreasing => Some("OBFDP"),
            Self::Harmonic => Some("HK"),
        }
    }

    /// Look up a strategy by its short name (case-insensitive).
    pub fn from_short_name(name: &str) -> Option<Self> {
        match name.to_uppercase().as_str() {
            "NF" => Some(Self::NextFit),
            "FF" => Some(Self::FirstFit),
            "BF" => Some(Self::BestFit),
            "WF" => Some(Self::WorstFit),
            "FFD" => Some(Self::FirstFitDecreasing),
            "BFD" => Some(Self::BestFitDecreasing),
            "FFS" => Some(Self::FirstFitShuffle),
            "MFFD" => Some(Self::ModifiedFirstFitDecreasing),
            "OBFD" => Some(Self::OptimizedBestFitDecreasing),
            "OBFDP" => Some(Self::ParallelOptimizedBestFitDecreasing),
            "HK" | "HARMONIC" => Some(Self::Harmonic),
            _ => None,
        }
    }
}

/// All available strategies, useful for iteration in tests and benchmarks.
pub const ALL_STRATEGIES: [PackStrategy; 11] = [
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

/// Trait for implementing packing algorithms.
///
/// All algorithms implement this trait, enabling:
/// - **Static dispatch** via generics (zero-cost, used in production)
/// - **Dynamic dispatch** via `Box<dyn PackingAlgorithm>` (used in tests
///   to run all algorithms in a loop)
///
/// # Example
///
/// ```ignore
/// struct MyAlgorithm;
///
/// impl PackingAlgorithm for MyAlgorithm {
///     fn pack(&self, sequences: Vec<Sequence>, capacity: usize) -> Result<Vec<Pack>> {
///         // your implementation here
///         todo!()
///     }
///
///     fn name(&self) -> &'static str {
///         "MyAlgorithm"
///     }
/// }
/// ```
pub trait PackingAlgorithm: Send + Sync {
    /// Pack sequences into bins of the given capacity.
    ///
    /// # Errors
    ///
    /// Returns `PackError::SequenceTooLong` if any sequence exceeds capacity.
    fn pack(&self, sequences: Vec<Sequence>, capacity: usize) -> Result<Vec<Pack>>;

    /// Human-readable algorithm name.
    fn name(&self) -> &'static str;
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_is_obfd() {
        assert_eq!(
            PackStrategy::default(),
            PackStrategy::OptimizedBestFitDecreasing
        );
    }

    #[test]
    fn test_strategy_is_copy() {
        let a = PackStrategy::OptimizedBestFitDecreasing;
        let b = a; // copy, not move
        assert_eq!(a, b); // a still valid
    }

    #[test]
    fn test_name_returns_expected_strings() {
        assert_eq!(PackStrategy::NextFit.name(), "NextFit");
        assert_eq!(PackStrategy::FirstFit.name(), "FirstFit");
        assert_eq!(PackStrategy::BestFit.name(), "BestFit");
        assert_eq!(PackStrategy::WorstFit.name(), "WorstFit");
        assert_eq!(
            PackStrategy::FirstFitDecreasing.name(),
            "FirstFitDecreasing"
        );
        assert_eq!(PackStrategy::BestFitDecreasing.name(), "BestFitDecreasing");
        assert_eq!(PackStrategy::FirstFitShuffle.name(), "FirstFitShuffle");
        assert_eq!(PackStrategy::ModifiedFirstFitDecreasing.name(), "MFFD");
        assert_eq!(PackStrategy::OptimizedBestFitDecreasing.name(), "OBFD");
        assert_eq!(
            PackStrategy::ParallelOptimizedBestFitDecreasing.name(),
            "OBFDP"
        );
        assert_eq!(PackStrategy::Harmonic.name(), "Harmonic");
    }

    #[test]
    fn test_all_variants_have_unique_names() {
        let all = ALL_STRATEGIES;
        let names: Vec<&str> = all.iter().map(|s| s.name()).collect();
        let mut deduped = names.clone();
        deduped.sort();
        deduped.dedup();
        assert_eq!(names.len(), deduped.len(), "duplicate strategy names found");
    }

    #[test]
    fn test_short_name_roundtrip() {
        for strategy in ALL_STRATEGIES {
            if let Some(short) = strategy.short_name() {
                let resolved = PackStrategy::from_short_name(short).unwrap();
                assert_eq!(resolved, strategy);
            }
        }
    }

    #[test]
    fn test_from_short_name_case_insensitive() {
        assert_eq!(
            PackStrategy::from_short_name("obfd"),
            Some(PackStrategy::OptimizedBestFitDecreasing),
        );
        assert_eq!(
            PackStrategy::from_short_name("Ffd"),
            Some(PackStrategy::FirstFitDecreasing),
        );
    }

    #[test]
    fn test_from_short_name_unknown() {
        assert_eq!(PackStrategy::from_short_name("UNKNOWN"), None);
    }
}
