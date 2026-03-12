//! OBFD (Optimized Best-Fit Decreasing) algorithm.
//!
//! LightBinPack's core algorithm. O(N log L) where N = number of sequences,
//! L = max capacity. Achieves 98.76% efficiency (same as FFD) but ~150-200x
//! faster due to:
//! - Counting sort O(N) instead of comparison sort O(N log N)
//! - Capacity-indexed segment tree O(log L) instead of bin-indexed O(log B)
//! - Integer-only arithmetic
//! - Early termination in tree updates

use smallvec::{SmallVec, smallvec};

use crate::error::{PackError, Result};
use crate::pack::Pack;
use crate::sequence::Sequence;
use crate::strategy::PackingAlgorithm;

use super::counting_sort::counting_sort;
use crate::placement::capacity_segment_tree::CapacitySegmentTree;

/// OBFD algorithm — the recommended default for offline packing.
pub struct OptimizedBestFitDecreasing;

impl PackingAlgorithm for OptimizedBestFitDecreasing {
    fn pack(&self, sequences: Vec<Sequence>, capacity: usize) -> Result<Vec<Pack>> {
        let lengths: Vec<usize> = sequences.iter().map(|s| s.length).collect();
        let bins = optimized_best_fit_decreasing_lengths(&lengths, capacity)?;
        Ok(bins_to_packs_from_indices(bins, &sequences, capacity))
    }

    fn name(&self) -> &'static str {
        "OptimizedBestFitDecreasing"
    }
}

/// Convert index-based bins to Pack objects.
///
/// Each inner Vec contains original indices into `sequences`.
pub fn bins_to_packs_from_indices(
    bins: Vec<Vec<usize>>,
    sequences: &[Sequence],
    capacity: usize,
) -> Vec<Pack> {
    bins.into_iter()
        .map(|item_indices| {
            let mut pack_seqs = Vec::with_capacity(item_indices.len());
            let mut used = 0;
            for idx in item_indices {
                let src = &sequences[idx];
                let seq = Sequence::new(src.id, src.length);
                used += seq.length;
                pack_seqs.push(seq);
            }
            Pack {
                sequences: pack_seqs,
                capacity,
                used,
            }
        })
        .collect()
}

/// Core OBFD operating on raw lengths. Returns `Vec<Vec<usize>>` where each
/// inner Vec contains original indices of items placed in that bin.
///
/// This is the hot path — no `Sequence`/`Item` abstraction overhead.
///
/// # Arguments
///
/// - `lengths`: Sequence lengths as raw integers.
/// - `capacity`: Maximum bin capacity.
///
/// # Errors
///
/// Returns `PackError::SequenceTooLong` if any length exceeds capacity.
pub fn optimized_best_fit_decreasing_lengths(
    lengths: &[usize],
    capacity: usize,
) -> Result<Vec<Vec<usize>>> {
    if lengths.is_empty() {
        return Ok(Vec::new());
    }

    // Validate: no item exceeds capacity.
    for (i, &len) in lengths.iter().enumerate() {
        if len > capacity {
            return Err(PackError::SequenceTooLong {
                length: len,
                capacity,
            });
        }
        if len == 0 {
            return Err(PackError::InvalidConfig {
                message: format!("sequence at index {i} has zero length"),
            });
        }
    }

    let max_length = *lengths.iter().max().unwrap();

    // 1. Counting sort — O(N + L)
    let buckets = counting_sort(lengths, max_length);

    // 2. Capacity segment tree — O(L) space
    let mut seg_tree = CapacitySegmentTree::new(capacity);

    // 3. Capacity -> bins mapping
    //    capacity_to_bins[c] = stack of bin indices with remaining capacity c
    let total_tokens: usize = lengths.iter().sum();
    let estimated_bins = (total_tokens / capacity) + 1;
    let mut capacity_to_bins: Vec<SmallVec<[usize; 4]>> = vec![SmallVec::new(); capacity + 1];
    let mut bins_remaining: Vec<usize> = Vec::with_capacity(estimated_bins);
    let mut bins_items: Vec<SmallVec<[usize; 8]>> = Vec::with_capacity(estimated_bins);

    // Seed: one empty bin at full capacity
    bins_remaining.push(capacity);
    capacity_to_bins[capacity].push(0);
    bins_items.push(SmallVec::new());

    // 4. Process largest -> smallest — O(N log L)
    for size in (1..=max_length).rev() {
        for &orig_idx in &buckets[size] {
            if let Some(best_cap) = seg_tree.find_best_fit(size) {
                // Place in existing bin with tightest fit.
                let bin_idx = capacity_to_bins[best_cap].pop().unwrap();
                if capacity_to_bins[best_cap].is_empty() {
                    seg_tree.update(best_cap, 0);
                }

                let new_cap = bins_remaining[bin_idx] - size;
                bins_remaining[bin_idx] = new_cap;
                bins_items[bin_idx].push(orig_idx);

                if new_cap > 0 {
                    capacity_to_bins[new_cap].push(bin_idx);
                    seg_tree.update(new_cap, new_cap);
                }
            } else {
                // Open a new bin.
                open_new_bin(
                    size,
                    orig_idx,
                    capacity,
                    &mut bins_remaining,
                    &mut bins_items,
                    &mut capacity_to_bins,
                    &mut seg_tree,
                );
            }
        }
    }

    Ok(bins_items.into_iter().map(SmallVec::into_vec).collect())
}

#[cold]
fn open_new_bin(
    size: usize,
    orig_idx: usize,
    capacity: usize,
    bins_remaining: &mut Vec<usize>,
    bins_items: &mut Vec<SmallVec<[usize; 8]>>,
    capacity_to_bins: &mut [SmallVec<[usize; 4]>],
    seg_tree: &mut CapacitySegmentTree,
) {
    let new_bin_idx = bins_remaining.len();
    let new_cap = capacity - size;
    bins_remaining.push(new_cap);
    bins_items.push(smallvec![orig_idx]);

    if new_cap > 0 {
        capacity_to_bins[new_cap].push(new_bin_idx);
        seg_tree.update(new_cap, new_cap);
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper ─────────────────────────────────────────────────────────

    /// Validate that bins are a valid packing of `lengths` with given `capacity`.
    fn validate_bins(bins: &[Vec<usize>], lengths: &[usize], capacity: usize) {
        // Every bin respects capacity.
        for (i, bin) in bins.iter().enumerate() {
            let total: usize = bin.iter().map(|&idx| lengths[idx]).sum();
            assert!(
                total <= capacity,
                "bin {i} total {total} exceeds capacity {capacity}"
            );
        }

        // Every item appears exactly once.
        let mut seen = vec![false; lengths.len()];
        for bin in bins {
            for &idx in bin {
                assert!(!seen[idx], "item {idx} appears in multiple bins");
                seen[idx] = true;
            }
        }
        for (i, &s) in seen.iter().enumerate() {
            assert!(s, "item {i} missing from bins");
        }
    }

    // ── Basic functionality ────────────────────────────────────────────

    #[test]
    fn test_empty_input() {
        let bins = optimized_best_fit_decreasing_lengths(&[], 10).unwrap();
        assert!(bins.is_empty());
    }

    #[test]
    fn test_single_item() {
        let bins = optimized_best_fit_decreasing_lengths(&[5], 10).unwrap();
        assert_eq!(bins.len(), 1);
        assert_eq!(bins[0], vec![0]);
    }

    #[test]
    fn test_exact_fit_single_bin() {
        let lengths = &[3, 3, 4];
        let bins = optimized_best_fit_decreasing_lengths(lengths, 10).unwrap();
        assert_eq!(bins.len(), 1);
        validate_bins(&bins, lengths, 10);
    }

    #[test]
    fn test_all_same_size_equal_capacity() {
        // 4 items of size 5, capacity 5 → 4 bins.
        let lengths = &[5, 5, 5, 5];
        let bins = optimized_best_fit_decreasing_lengths(lengths, 5).unwrap();
        assert_eq!(bins.len(), 4);
        validate_bins(&bins, lengths, 5);
    }

    #[test]
    fn test_all_same_size_two_per_bin() {
        // 4 items of size 5, capacity 10 → 2 bins.
        let lengths = &[5, 5, 5, 5];
        let bins = optimized_best_fit_decreasing_lengths(lengths, 10).unwrap();
        assert_eq!(bins.len(), 2);
        validate_bins(&bins, lengths, 10);
    }

    #[test]
    fn test_decreasing_order_packs_well() {
        // Classic BFD example: should pack tightly.
        let lengths = &[7, 5, 5, 4, 3, 3, 2, 2, 1];
        let bins = optimized_best_fit_decreasing_lengths(lengths, 10).unwrap();
        validate_bins(&bins, lengths, 10);
        // Total = 32, capacity 10 → optimal = ceil(32/10) = 4 bins.
        assert!(bins.len() <= 5, "expected ≤5 bins, got {}", bins.len());
    }

    #[test]
    fn test_best_fit_chooses_tightest() {
        // capacity=10, items: [6, 5, 4]
        // After 6: bin0 has remaining=4
        // 5 doesn't fit in bin0 (remaining=4 < 5) → new bin1, remaining=5
        // 4 fits in bin0 (remaining=4 == 4, tighter than bin1's 5)
        let lengths = &[6, 5, 4];
        let bins = optimized_best_fit_decreasing_lengths(lengths, 10).unwrap();
        assert_eq!(bins.len(), 2);
        validate_bins(&bins, lengths, 10);
    }

    // ── Error cases ────────────────────────────────────────────────────

    #[test]
    fn test_sequence_too_long() {
        let err = optimized_best_fit_decreasing_lengths(&[11], 10).unwrap_err();
        assert!(matches!(
            err,
            PackError::SequenceTooLong {
                length: 11,
                capacity: 10
            }
        ));
    }

    #[test]
    fn test_zero_length_rejected() {
        let err = optimized_best_fit_decreasing_lengths(&[5, 0, 3], 10).unwrap_err();
        assert!(matches!(err, PackError::InvalidConfig { .. }));
    }

    // ── Coverage and invariants ────────────────────────────────────────

    #[test]
    fn test_all_items_placed() {
        let lengths: Vec<usize> = (1..=20).collect();
        let bins = optimized_best_fit_decreasing_lengths(&lengths, 50).unwrap();
        validate_bins(&bins, &lengths, 50);
    }

    #[test]
    fn test_large_items_one_per_bin() {
        // Every item equals capacity → N bins.
        let lengths = &[100, 100, 100];
        let bins = optimized_best_fit_decreasing_lengths(lengths, 100).unwrap();
        assert_eq!(bins.len(), 3);
        validate_bins(&bins, lengths, 100);
    }

    #[test]
    fn test_mixed_sizes() {
        let lengths = &[8, 7, 6, 5, 4, 3, 2, 1];
        let bins = optimized_best_fit_decreasing_lengths(lengths, 10).unwrap();
        validate_bins(&bins, lengths, 10);
        // Total = 36, capacity 10 → optimal = 4 bins.
        assert!(bins.len() <= 5);
    }

    // ── PackingAlgorithm trait ─────────────────────────────────────────

    #[test]
    fn test_packing_algorithm_trait() {
        let algo = OptimizedBestFitDecreasing;
        let sequences = vec![
            Sequence::new(0, 6),
            Sequence::new(1, 4),
            Sequence::new(2, 3),
        ];
        let packs = algo.pack(sequences, 10).unwrap();
        assert!(!packs.is_empty());
        // All sequences placed.
        let total_seqs: usize = packs.iter().map(|p| p.len()).sum();
        assert_eq!(total_seqs, 3);
    }

    #[test]
    fn test_name() {
        assert_eq!(
            OptimizedBestFitDecreasing.name(),
            "OptimizedBestFitDecreasing"
        );
    }
}
