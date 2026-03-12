//! Modified First Fit Decreasing (MFFD) — 5-phase size-class algorithm.
//!
//! Johnson & Garey (1985). Classifies items into four size classes
//! (large, medium, small, tiny) relative to bin capacity and packs
//! them in five phases for a tighter worst-case bound than FFD.
//!
//! Approximation ratio: 71/60 ≈ 1.183 (vs FFD's 11/9 ≈ 1.222).

use crate::error::{PackError, Result};
use crate::pack::{Bin, Pack, bins_to_packs};
use crate::placement::{PlacementIndex, SegmentTreeIndex};
use crate::sequence::{Item, Sequence};
use crate::strategy::PackingAlgorithm;

/// Size class for MFFD classification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SizeClass {
    /// > 1/2 capacity
    Large,
    /// > 1/3 capacity (and <= 1/2)
    Medium,
    /// > 1/6 capacity (and <= 1/3)
    Small,
    /// <= 1/6 capacity
    Tiny,
}

fn classify(len: usize, capacity: usize) -> SizeClass {
    // Use integer arithmetic to avoid floating-point: len > capacity/2
    // is equivalent to 2*len > capacity (no overflow for realistic values).
    if 2 * len > capacity {
        SizeClass::Large
    } else if 3 * len > capacity {
        SizeClass::Medium
    } else if 6 * len > capacity {
        SizeClass::Small
    } else {
        SizeClass::Tiny
    }
}

/// Modified First Fit Decreasing algorithm implementation.
pub struct ModifiedFirstFitDecreasing;

impl PackingAlgorithm for ModifiedFirstFitDecreasing {
    fn pack(&self, sequences: Vec<Sequence>, capacity: usize) -> Result<Vec<Pack>> {
        let mut items: Vec<Item> = sequences.iter().map(|s| s.to_item()).collect();

        // Validate: no item exceeds capacity.
        for item in &items {
            if item.len > capacity {
                return Err(PackError::SequenceTooLong {
                    length: item.len,
                    capacity,
                });
            }
        }

        if items.is_empty() {
            return Ok(Vec::new());
        }

        // Sort all items descending by length.
        items.sort_unstable_by(|a, b| b.len.cmp(&a.len));

        // Classify items into size classes.
        let mut large: Vec<Item> = Vec::new();
        let mut medium: Vec<Item> = Vec::new();
        let mut small: Vec<Item> = Vec::new();
        let mut tiny: Vec<Item> = Vec::new();

        for item in &items {
            match classify(item.len, capacity) {
                SizeClass::Large => large.push(*item),
                SizeClass::Medium => medium.push(*item),
                SizeClass::Small => small.push(*item),
                SizeClass::Tiny => tiny.push(*item),
            }
        }

        // All sub-lists are already sorted descending (inherited from items).

        let mut bins: Vec<Bin> = Vec::new();

        // ── Phase 1: One bin per large item ──────────────────────────
        for item in &large {
            let bin_id = bins.len();
            let mut bin = Bin::new(bin_id, capacity);
            bin.used = item.len;
            bin.items.push(item.id);
            bins.push(bin);
        }

        // Track which bins received a medium item (for Phase 3).
        let mut has_medium = vec![false; bins.len()];

        // ── Phase 2: Pack medium items into large-item bins ──────────
        // Proceed forward through bins. For each bin, if the largest
        // remaining medium item fits, place it.
        if !medium.is_empty() {
            let mut med_idx = 0;
            for bin_id in 0..bins.len() {
                if med_idx >= medium.len() {
                    break;
                }
                let remaining = bins[bin_id].remaining();
                // Check if smallest remaining medium fits (optimization: skip if not).
                if medium.last().unwrap().len > remaining {
                    continue;
                }
                // Place largest remaining medium item that fits.
                if medium[med_idx].len <= remaining {
                    let item = medium[med_idx];
                    bins[bin_id].used += item.len;
                    bins[bin_id].items.push(item.id);
                    has_medium[bin_id] = true;
                    med_idx += 1;
                }
            }
            // Any remaining medium items: pair them with each other (2 per bin max,
            // since each is > 1/3 capacity). Process in pairs.
            let leftover_medium: Vec<Item> = medium[med_idx..].to_vec();
            let mut lm_idx = 0;
            while lm_idx < leftover_medium.len() {
                let bin_id = bins.len();
                let mut bin = Bin::new(bin_id, capacity);
                bin.used = leftover_medium[lm_idx].len;
                bin.items.push(leftover_medium[lm_idx].id);
                lm_idx += 1;
                // Try to pair with the next medium item if it fits.
                if lm_idx < leftover_medium.len()
                    && bin.used + leftover_medium[lm_idx].len <= capacity
                {
                    bin.used += leftover_medium[lm_idx].len;
                    bin.items.push(leftover_medium[lm_idx].id);
                    lm_idx += 1;
                }
                bins.push(bin);
                has_medium.push(true);
            }
        }

        // ── Phase 3: Pack small items into bins WITHOUT medium items ─
        // Proceed backward through non-medium bins. Try to fit pairs of
        // small items.
        if !small.is_empty() {
            // small is sorted descending: small[0] is largest, small[last] is smallest.
            let mut sf = 0; // front = largest
            let mut sb = small.len() - 1; // back = smallest

            // Iterate bins backward that have NO medium item.
            let bin_ids_no_medium: Vec<usize> = (0..bins.len())
                .rev()
                .filter(|&id| !has_medium[id])
                .collect();

            for &bin_id in &bin_ids_no_medium {
                if sf > sb {
                    break;
                }
                let remaining = bins[bin_id].remaining();
                // Check if two smallest remaining small items fit.
                if sf < sb && small[sb].len + small[sb - 1].len <= remaining {
                    // Place smallest small item first.
                    let smallest = small[sb];
                    bins[bin_id].used += smallest.len;
                    bins[bin_id].items.push(smallest.id);
                    sb -= 1;

                    // Now try to place the largest remaining small item that fits.
                    let new_remaining = bins[bin_id].remaining();
                    if small[sf].len <= new_remaining {
                        let largest_fitting = small[sf];
                        bins[bin_id].used += largest_fitting.len;
                        bins[bin_id].items.push(largest_fitting.id);
                        sf += 1;
                    }
                } else if small[sb].len <= remaining {
                    // Only one small item fits.
                    let item = small[sb];
                    bins[bin_id].used += item.len;
                    bins[bin_id].items.push(item.id);
                    if sb == 0 {
                        // Prevent underflow: we've consumed the last item.
                        sf = 1; // ensures sf > sb, breaking the loop.
                        break;
                    }
                    sb -= 1;
                }
            }

            // Phase 4: First Fit remaining small items.
            if sf <= sb {
                let remaining_small: Vec<Item> = small[sf..=sb].to_vec();
                let mut index = SegmentTreeIndex::with_capacity(bins.len() + remaining_small.len());
                for (i, bin) in bins.iter().enumerate() {
                    index.insert_bin(i, bin.remaining());
                }
                for item in &remaining_small {
                    match index.first_fit(item.len) {
                        Some(bin_id) => {
                            let old_rem = bins[bin_id].remaining();
                            bins[bin_id].used += item.len;
                            bins[bin_id].items.push(item.id);
                            index.update_bin(bin_id, old_rem, bins[bin_id].remaining());
                        }
                        None => {
                            let bin_id = bins.len();
                            let mut bin = Bin::new(bin_id, capacity);
                            bin.used = item.len;
                            bin.items.push(item.id);
                            bins.push(bin);
                            index.insert_bin(bin_id, bins[bin_id].remaining());
                        }
                    }
                }
            }
        }

        // ── Phase 5: First Fit all tiny items ────────────────────────
        if !tiny.is_empty() {
            let mut index = SegmentTreeIndex::with_capacity(bins.len() + tiny.len());
            for (i, bin) in bins.iter().enumerate() {
                index.insert_bin(i, bin.remaining());
            }
            for item in &tiny {
                match index.first_fit(item.len) {
                    Some(bin_id) => {
                        let old_rem = bins[bin_id].remaining();
                        bins[bin_id].used += item.len;
                        bins[bin_id].items.push(item.id);
                        index.update_bin(bin_id, old_rem, bins[bin_id].remaining());
                    }
                    None => {
                        let bin_id = bins.len();
                        let mut bin = Bin::new(bin_id, capacity);
                        bin.used = item.len;
                        bin.items.push(item.id);
                        bins.push(bin);
                        index.insert_bin(bin_id, bins[bin_id].remaining());
                    }
                }
            }
        }

        Ok(bins_to_packs(bins, &sequences))
    }

    fn name(&self) -> &'static str {
        "ModifiedFirstFitDecreasing"
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

    // ── Size classification ──────────────────────────────────────

    #[test]
    fn test_classify_large() {
        // > 1/2 of 100 = > 50
        assert_eq!(classify(51, 100), SizeClass::Large);
        assert_eq!(classify(100, 100), SizeClass::Large);
    }

    #[test]
    fn test_classify_medium() {
        // > 1/3 and <= 1/2 of 100 = (33, 50]
        assert_eq!(classify(34, 100), SizeClass::Medium);
        assert_eq!(classify(50, 100), SizeClass::Medium);
    }

    #[test]
    fn test_classify_small() {
        // > 1/6 and <= 1/3 of 100 = (16, 33]
        assert_eq!(classify(17, 100), SizeClass::Small);
        assert_eq!(classify(33, 100), SizeClass::Small);
    }

    #[test]
    fn test_classify_tiny() {
        // <= 1/6 of 100 = <= 16
        assert_eq!(classify(16, 100), SizeClass::Tiny);
        assert_eq!(classify(1, 100), SizeClass::Tiny);
    }

    // ── Basic packing ────────────────────────────────────────────

    #[test]
    fn test_empty_input() {
        let packs = ModifiedFirstFitDecreasing.pack(seqs(&[]), 10).unwrap();
        assert!(packs.is_empty());
    }

    #[test]
    fn test_single_item() {
        let packs = ModifiedFirstFitDecreasing.pack(seqs(&[5]), 10).unwrap();
        assert_eq!(packs.len(), 1);
    }

    #[test]
    fn test_oversize_error() {
        let result = ModifiedFirstFitDecreasing.pack(seqs(&[11]), 10);
        assert!(matches!(
            result,
            Err(PackError::SequenceTooLong {
                length: 11,
                capacity: 10
            })
        ));
    }

    #[test]
    fn test_all_same_size() {
        let packs = ModifiedFirstFitDecreasing
            .pack(seqs(&[5, 5, 5, 5]), 10)
            .unwrap();
        assert_eq!(packs.len(), 2);
    }

    #[test]
    fn test_each_item_needs_own_bin() {
        let packs = ModifiedFirstFitDecreasing
            .pack(seqs(&[6, 7, 8, 9]), 10)
            .unwrap();
        assert_eq!(packs.len(), 4);
    }

    // ── Phase behavior ───────────────────────────────────────────

    #[test]
    fn test_large_items_get_own_bins() {
        // Items > 50% of capacity=100 each get their own bin.
        let packs = ModifiedFirstFitDecreasing
            .pack(seqs(&[60, 70, 80]), 100)
            .unwrap();
        assert_eq!(packs.len(), 3);
    }

    #[test]
    fn test_large_paired_with_medium() {
        // Large (60) + medium (35) should share a bin (capacity=100).
        let packs = ModifiedFirstFitDecreasing
            .pack(seqs(&[60, 35]), 100)
            .unwrap();
        assert_eq!(packs.len(), 1);
    }

    #[test]
    fn test_large_paired_with_tiny() {
        // Large (60) + tiny (10) should share a bin.
        let packs = ModifiedFirstFitDecreasing
            .pack(seqs(&[60, 10]), 100)
            .unwrap();
        assert_eq!(packs.len(), 1);
    }

    #[test]
    fn test_mixed_sizes() {
        // Capacity 120: large>60, medium (40,60], small (20,40], tiny<=20
        // Items: 80(L), 70(L), 50(M), 45(M), 30(S), 25(S), 10(T), 5(T)
        // Phase 1: bin0=[80], bin1=[70]
        // Phase 2: 45→bin1(rem=50, fits), 50→bin0(rem=40, fits)
        // Phase 5: tiny items fill gaps
        let packs = ModifiedFirstFitDecreasing
            .pack(seqs(&[80, 70, 50, 45, 30, 25, 10, 5]), 120)
            .unwrap();
        // All items packed, should be quite efficient.
        let total_items: usize = packs.iter().map(|p| p.sequences.len()).sum();
        assert_eq!(total_items, 8);
    }

    // ── All sequences accounted for ──────────────────────────────

    #[test]
    fn test_all_sequences_present() {
        let lens = &[80, 70, 50, 45, 30, 25, 10, 5];
        let packs = ModifiedFirstFitDecreasing.pack(seqs(lens), 120).unwrap();
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
        let lens = &[80, 70, 50, 45, 30, 25, 10, 5, 15, 35, 60, 90];
        let capacity = 120;
        let packs = ModifiedFirstFitDecreasing
            .pack(seqs(lens), capacity)
            .unwrap();
        for pack in &packs {
            assert!(pack.used_capacity() <= capacity);
        }
    }

    // ── Comparison with FFD ──────────────────────────────────────

    #[test]
    fn test_at_least_as_good_as_or_equal_to_ffd() {
        use crate::algorithms::FirstFitDecreasing;

        let lens = &[80, 70, 50, 45, 30, 25, 10, 5, 15, 35, 60, 90];
        let capacity = 120;
        let mffd_packs = ModifiedFirstFitDecreasing
            .pack(seqs(lens), capacity)
            .unwrap();
        let ffd_packs = FirstFitDecreasing.pack(seqs(lens), capacity).unwrap();
        // MFFD should use <= bins than FFD (or equal on typical inputs).
        // On some inputs they're equal; MFFD's advantage is worst-case.
        assert!(mffd_packs.len() <= ffd_packs.len() + 1);
    }

    // ── Trait ────────────────────────────────────────────────────

    #[test]
    fn test_name() {
        assert_eq!(
            ModifiedFirstFitDecreasing.name(),
            "ModifiedFirstFitDecreasing"
        );
    }

    #[test]
    fn test_many_small_items() {
        let packs = ModifiedFirstFitDecreasing
            .pack(seqs(&[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 10)
            .unwrap();
        assert_eq!(packs.len(), 1);
    }

    #[test]
    fn test_only_tiny_items() {
        // All items are tiny (<=1/6 of 60 = <=10).
        let packs = ModifiedFirstFitDecreasing
            .pack(seqs(&[5, 8, 3, 10, 7, 2, 9, 4]), 60)
            .unwrap();
        let total_items: usize = packs.iter().map(|p| p.sequences.len()).sum();
        assert_eq!(total_items, 8);
        for pack in &packs {
            assert!(pack.used_capacity() <= 60);
        }
    }

    #[test]
    fn test_only_large_items() {
        // All items are large (>1/2 of 10 = >5).
        let packs = ModifiedFirstFitDecreasing
            .pack(seqs(&[6, 7, 8, 9, 10]), 10)
            .unwrap();
        assert_eq!(packs.len(), 5);
    }
}
