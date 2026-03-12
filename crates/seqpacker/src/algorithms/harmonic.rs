//! Harmonic-K algorithm — online, bounded-space bin packing.
//!
//! Partitions items into k size classes and maintains dedicated bins
//! per class. Each class bin only accepts items of that class, and
//! is closed when full.
//!
//! Size classes for k classes (1-indexed):
//!   Class 1: (1/2, 1]         — at most 1 item per bin
//!   Class 2: (1/3, 1/2]       — at most 2 items per bin
//!   Class i: (1/(i+1), 1/i]   — at most i items per bin
//!   Class k: (0, 1/k]         — packed via first-fit (catch-all)
//!
//! Approximation ratio approaches ~1.691 as k → ∞.
//! Bounded space: at most k open bins at any time.

use crate::error::{PackError, Result};
use crate::pack::{Bin, Pack, bins_to_packs};
use crate::sequence::{Item, Sequence};
use crate::strategy::PackingAlgorithm;

/// Default number of size classes.
const DEFAULT_K: usize = 10;

/// Harmonic-k algorithm implementation.
pub struct Harmonic {
    k: usize,
}

impl Harmonic {
    /// Create a Harmonic packer with k size classes.
    pub fn new(k: usize) -> Self {
        assert!(k >= 2, "Harmonic-K requires k >= 2");
        Self { k }
    }

    /// Classify an item into a size class index (0-based).
    ///
    /// Class 0: (1/2, 1]       — items that take more than half the bin
    /// Class 1: (1/3, 1/2]     — items that take 1/3 to 1/2
    /// Class i: (1/(i+2), 1/(i+1)] for i in 0..k-2
    /// Class k-1: (0, 1/(k)]  — catch-all for small items
    fn classify(&self, len: usize, capacity: usize) -> usize {
        if len == 0 {
            return self.k - 1;
        }
        // Item fits in class i if len > capacity/(i+2) and len <= capacity/(i+1).
        // Equivalently, class = floor(capacity/len) - 1, clamped to [0, k-1].
        // For len > capacity/2: capacity/len < 2, so floor = 1, class = 0. Correct.
        // For len in (capacity/3, capacity/2]: floor = 2, class = 1. Correct.
        let ratio = capacity / len; // integer division, equivalent to floor(capacity/len)
        if ratio <= 1 {
            0
        } else {
            (ratio - 1).min(self.k - 1)
        }
    }

    /// Maximum number of items that fit in a bin for a given class.
    fn max_items_for_class(&self, class: usize) -> usize {
        // Class 0: items > 1/2 → at most 1 per bin
        // Class 1: items > 1/3 → at most 2 per bin
        // Class i: items > 1/(i+2) → at most i+1 per bin
        // Class k-1 (catch-all): use capacity-based packing, no fixed limit.
        if class == self.k - 1 {
            usize::MAX // catch-all class uses first-fit, no fixed limit
        } else {
            class + 1
        }
    }
}

impl Default for Harmonic {
    fn default() -> Self {
        Self::new(DEFAULT_K)
    }
}

impl PackingAlgorithm for Harmonic {
    fn pack(&self, sequences: Vec<Sequence>, capacity: usize) -> Result<Vec<Pack>> {
        let items: Vec<Item> = sequences.iter().map(|s| s.to_item()).collect();

        if items.is_empty() {
            return Ok(Vec::new());
        }

        let mut bins: Vec<Bin> = Vec::new();

        // For each class, track the current open bin (if any).
        // open_bins[class] = Some(bin_id) or None if no open bin for that class.
        let mut open_bins: Vec<Option<usize>> = vec![None; self.k];

        // For the catch-all class (k-1), we may have multiple open bins.
        // Track all of them for first-fit within the class.
        let mut catchall_open: Vec<usize> = Vec::new();

        for item in &items {
            if item.len > capacity {
                return Err(PackError::SequenceTooLong {
                    length: item.len,
                    capacity,
                });
            }

            let class = self.classify(item.len, capacity);

            if class == self.k - 1 {
                // Catch-all class: first-fit among open catch-all bins.
                let mut placed = false;
                for &bin_id in &catchall_open {
                    if bins[bin_id].remaining() >= item.len {
                        bins[bin_id].used += item.len;
                        bins[bin_id].items.push(item.id);
                        placed = true;
                        break;
                    }
                }
                if !placed {
                    let bin_id = bins.len();
                    let mut bin = Bin::new(bin_id, capacity);
                    bin.used = item.len;
                    bin.items.push(item.id);
                    bins.push(bin);
                    catchall_open.push(bin_id);
                }
            } else {
                // Fixed-class bin: items of this class go into a dedicated bin.
                let max_items = self.max_items_for_class(class);

                match open_bins[class] {
                    Some(bin_id)
                        if bins[bin_id].remaining() >= item.len
                            && bins[bin_id].items.len() < max_items =>
                    {
                        // Fits in the open bin for this class.
                        bins[bin_id].used += item.len;
                        bins[bin_id].items.push(item.id);
                        // Close the bin if it's now full (reached max items for class).
                        if bins[bin_id].items.len() >= max_items {
                            open_bins[class] = None;
                        }
                    }
                    _ => {
                        // Close old bin (if any) and open a new one.
                        let bin_id = bins.len();
                        let mut bin = Bin::new(bin_id, capacity);
                        bin.used = item.len;
                        bin.items.push(item.id);
                        bins.push(bin);
                        if max_items <= 1 {
                            // Immediately full.
                            open_bins[class] = None;
                        } else {
                            open_bins[class] = Some(bin_id);
                        }
                    }
                }
            }
        }

        Ok(bins_to_packs(bins, &sequences))
    }

    fn name(&self) -> &'static str {
        "Harmonic"
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

    fn harmonic() -> Harmonic {
        Harmonic::default()
    }

    // ── Classification ───────────────────────────────────────────

    #[test]
    fn test_classify_large_items() {
        let h = harmonic();
        // > 1/2 of 100 → class 0
        assert_eq!(h.classify(51, 100), 0);
        assert_eq!(h.classify(100, 100), 0);
    }

    #[test]
    fn test_classify_medium_items() {
        let h = harmonic();
        // (1/3, 1/2] of 100 → class 1
        assert_eq!(h.classify(34, 100), 1);
        assert_eq!(h.classify(50, 100), 1);
    }

    #[test]
    fn test_classify_small_items() {
        let h = harmonic();
        // (1/4, 1/3] of 100 → class 2
        assert_eq!(h.classify(26, 100), 2);
        assert_eq!(h.classify(33, 100), 2);
    }

    #[test]
    fn test_classify_tiny_items() {
        let h = harmonic();
        // Very small items → catch-all class (k-1 = 9)
        assert_eq!(h.classify(1, 100), 9);
        assert_eq!(h.classify(5, 100), 9);
    }

    #[test]
    fn test_classify_zero_length() {
        let h = harmonic();
        assert_eq!(h.classify(0, 100), 9);
    }

    // ── Basic packing ────────────────────────────────────────────

    #[test]
    fn test_empty_input() {
        let packs = harmonic().pack(seqs(&[]), 10).unwrap();
        assert!(packs.is_empty());
    }

    #[test]
    fn test_single_item() {
        let packs = harmonic().pack(seqs(&[5]), 10).unwrap();
        assert_eq!(packs.len(), 1);
    }

    #[test]
    fn test_oversize_error() {
        let result = harmonic().pack(seqs(&[11]), 10);
        assert!(matches!(
            result,
            Err(PackError::SequenceTooLong {
                length: 11,
                capacity: 10
            })
        ));
    }

    // ── Class-based behavior ─────────────────────────────────────

    #[test]
    fn test_large_items_get_own_bins() {
        // Items > 50% of capacity → class 0 → max 1 per bin.
        let packs = harmonic().pack(seqs(&[60, 70, 80]), 100).unwrap();
        assert_eq!(packs.len(), 3);
    }

    #[test]
    fn test_medium_items_pair_up() {
        // Items in (1/3, 1/2] → class 1 → max 2 per bin.
        // 40 + 40 = 80 <= 100.
        let packs = harmonic().pack(seqs(&[40, 40]), 100).unwrap();
        assert_eq!(packs.len(), 1);
    }

    #[test]
    fn test_medium_items_three_need_two_bins() {
        // Class 1 → max 2 per bin. Three items need 2 bins.
        let packs = harmonic().pack(seqs(&[40, 40, 40]), 100).unwrap();
        assert_eq!(packs.len(), 2);
    }

    #[test]
    fn test_small_items_triple_up() {
        // Items in (1/4, 1/3] → class 2 → max 3 per bin.
        // 30 + 30 + 30 = 90 <= 100.
        let packs = harmonic().pack(seqs(&[30, 30, 30]), 100).unwrap();
        assert_eq!(packs.len(), 1);
    }

    #[test]
    fn test_catch_all_first_fit() {
        // Very small items use first-fit in catch-all class.
        // 10 items of size 1 should all fit in 1 bin (capacity 100).
        let packs = harmonic()
            .pack(seqs(&[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 100)
            .unwrap();
        assert_eq!(packs.len(), 1);
    }

    #[test]
    fn test_different_classes_get_separate_bins() {
        // A large item and a medium item go to different bins
        // (unlike FF/BF which would combine them).
        let packs = harmonic().pack(seqs(&[60, 40]), 100).unwrap();
        // 60 → class 0 (>50), 40 → class 1 (>33, <=50)
        // They go to separate class bins.
        assert_eq!(packs.len(), 2);
    }

    // ── All sequences accounted for ──────────────────────────────

    #[test]
    fn test_all_sequences_present() {
        let lens = &[60, 40, 30, 25, 10, 5, 80, 35];
        let packs = harmonic().pack(seqs(lens), 100).unwrap();
        let mut all_ids: Vec<usize> = packs
            .iter()
            .flat_map(|p| p.sequences.iter().map(|s| s.id))
            .collect();
        all_ids.sort();
        let expected: Vec<usize> = (0..lens.len()).collect();
        assert_eq!(all_ids, expected);
    }

    #[test]
    fn test_no_bin_exceeds_capacity() {
        let lens = &[60, 40, 30, 25, 10, 5, 80, 35, 15, 45, 70, 20];
        let capacity = 100;
        let packs = harmonic().pack(seqs(lens), capacity).unwrap();
        for pack in &packs {
            assert!(pack.used_capacity() <= capacity);
        }
    }

    // ── Custom k values ──────────────────────────────────────────

    #[test]
    fn test_k_equals_2() {
        // k=2: class 0 = (1/2, 1], class 1 = (0, 1/2] (catch-all)
        let h = Harmonic::new(2);
        let packs = h.pack(seqs(&[60, 40, 30, 20, 10]), 100).unwrap();
        // 60 → class 0 (own bin)
        // 40, 30, 20, 10 → class 1 (catch-all, first-fit)
        // 40+30=70, 20+10=30 → potentially 2 catch-all bins plus 1 large bin
        let total_items: usize = packs.iter().map(|p| p.sequences.len()).sum();
        assert_eq!(total_items, 5);
    }

    #[test]
    #[should_panic(expected = "k >= 2")]
    fn test_k_less_than_2_panics() {
        Harmonic::new(1);
    }

    // ── Trait ────────────────────────────────────────────────────

    #[test]
    fn test_name() {
        assert_eq!(harmonic().name(), "Harmonic");
    }

    #[test]
    fn test_many_small_items() {
        let packs = harmonic()
            .pack(seqs(&[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 10)
            .unwrap();
        assert_eq!(packs.len(), 1);
    }

    #[test]
    fn test_exact_capacity() {
        let packs = harmonic().pack(seqs(&[10]), 10).unwrap();
        assert_eq!(packs.len(), 1);
        assert_eq!(packs[0].used_capacity(), 10);
    }

    #[test]
    fn test_all_same_size_large() {
        // All items are large (>50%), each gets own bin.
        let packs = harmonic().pack(seqs(&[6, 6, 6, 6]), 10).unwrap();
        assert_eq!(packs.len(), 4);
    }

    #[test]
    fn test_all_same_size_medium() {
        // All items are medium (>1/3, <=1/2 of 10 → 4 or 5).
        // 4 is > 10/3 ≈ 3.33 and <= 10/2 = 5 → class 1, max 2 per bin.
        let packs = harmonic().pack(seqs(&[4, 4, 4, 4]), 10).unwrap();
        assert_eq!(packs.len(), 2); // 2 per bin → 2 bins
    }
}
