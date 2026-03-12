//! Best Fit Decreasing — sort descending by length, then Best Fit.

use crate::engine::{ChooseFn, greedy_pack};
use crate::error::Result;
use crate::pack::{Pack, bins_to_packs};
use crate::placement::{BTreeRemainingIndex, PlacementIndex};
use crate::sequence::{Item, Sequence};
use crate::strategy::PackingAlgorithm;

/// Best Fit Decreasing algorithm implementation.
pub struct BestFitDecreasing;

impl PackingAlgorithm for BestFitDecreasing {
    fn pack(&self, sequences: Vec<Sequence>, capacity: usize) -> Result<Vec<Pack>> {
        // Convert to lightweight items, then sort descending by length.
        // Sorting items (16 bytes each) is faster than sorting sequences
        // (which may own Vec<u32> token data). Items keep their original IDs,
        // so bins_to_packs can index into the unchanged sequences array.
        let mut items: Vec<Item> = sequences.iter().map(|s| s.to_item()).collect();
        items.sort_unstable_by(|a, b| b.len.cmp(&a.len));

        let mut index = BTreeRemainingIndex::new();
        let bins = greedy_pack(
            items.into_iter(),
            capacity,
            &mut index,
            BTreeRemainingIndex::best_fit as ChooseFn<BTreeRemainingIndex>,
        )?;
        Ok(bins_to_packs(bins, &sequences))
    }

    fn name(&self) -> &'static str {
        "BestFitDecreasing"
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::greedy_pack;
    use crate::error::PackError;
    use crate::sequence::Item;
    use crate::validation::validate_solution;

    fn seqs(lens: &[usize]) -> Vec<Sequence> {
        lens.iter()
            .enumerate()
            .map(|(id, &len)| Sequence::new(id, len))
            .collect()
    }

    fn items(lens: &[usize]) -> Vec<Item> {
        lens.iter()
            .enumerate()
            .map(|(id, &len)| Item { id, len })
            .collect()
    }

    /// Sort lengths descending, then create items preserving original IDs
    /// (same as the production code: to_item() then sort).
    fn items_sorted_desc(lens: &[usize]) -> Vec<Item> {
        let mut itms: Vec<Item> = lens
            .iter()
            .enumerate()
            .map(|(id, &len)| Item { id, len })
            .collect();
        itms.sort_unstable_by(|a, b| b.len.cmp(&a.len));
        itms
    }

    /// Run BFD at the engine level: sort items descending, then best fit.
    fn bfd_bins(lens: &[usize], capacity: usize) -> Vec<crate::pack::Bin> {
        let itms = items_sorted_desc(lens);
        let mut index = BTreeRemainingIndex::new();
        greedy_pack(
            itms.into_iter(),
            capacity,
            &mut index,
            BTreeRemainingIndex::best_fit as ChooseFn<BTreeRemainingIndex>,
        )
        .unwrap()
    }

    // ── Core behavior: sorting + tightest fit ─────────────────────

    #[test]
    fn test_sorting_reduces_bins() {
        // BF on [3, 8, 5, 7] cap=10:
        //   3→b0(rem=7), 8→b0? rem=7<8→b1(rem=2), 5→b0(rem=7,tightest≥5)→rem=2,
        //   7→b0(rem=2<7)→b1(rem=2<7)→b2 → 3 bins
        //
        // BFD sorted [8, 7, 5, 3]:
        //   8→b0(rem=2), 7→b1(rem=3), 5→b2(rem=5), 3→b1(rem=3,tightest≥3)→rem=0
        //   → 3 bins (same here, but uses best_fit placement)
        let packs = BestFitDecreasing.pack(seqs(&[3, 8, 5, 7]), 10).unwrap();
        assert_eq!(packs.len(), 3);
    }

    #[test]
    fn test_sort_order_verified() {
        // Input [3, 5, 8] → sorted [8, 5, 3]
        // 8→b0(rem=2), 5→b1(rem=5), 3→b1(rem=5,tightest≥3)→rem=2
        let bins = bfd_bins(&[3, 5, 8], 10);
        assert_eq!(bins.len(), 2);
        assert_eq!(bins[0].used, 8);
        assert_eq!(bins[1].used, 8); // 5+3
    }

    #[test]
    fn test_large_items_first_with_tight_placement() {
        // [8, 7, 5, 4, 3] (sorted desc) cap=10
        // 8→b0(rem=2), 7→b1(rem=3), 5→b2(rem=5),
        // 4→b2(rem=5,tightest≥4)→rem=1, 3→b1(rem=3,tightest≥3)→rem=0
        let bins = bfd_bins(&[3, 4, 5, 7, 8], 10);
        assert_eq!(bins.len(), 3);
        // b1 should be exactly full: 7+3=10
        assert_eq!(bins[1].used, 10);
    }

    #[test]
    fn test_already_sorted_input() {
        let bins = bfd_bins(&[8, 7, 5, 4, 3], 10);
        assert_eq!(bins.len(), 3);
    }

    #[test]
    fn test_tight_fit_after_sort() {
        // After sorting, BFD should find exact fits that BF on unsorted misses.
        // [2, 8, 3, 7] cap=10 → sorted [8, 7, 3, 2]
        // 8→b0(rem=2), 7→b1(rem=3), 3→b1(rem=3,exact)→rem=0, 2→b0(rem=2,exact)→rem=0
        let bins = bfd_bins(&[2, 8, 3, 7], 10);
        assert_eq!(bins.len(), 2);
        assert_eq!(bins[0].remaining(), 0); // 8+2=10
        assert_eq!(bins[1].remaining(), 0); // 7+3=10
    }

    // ── Comparison with BF ────────────────────────────────────────

    #[test]
    fn test_bfd_at_least_as_good_as_bf() {
        let lens = &[3, 8, 3, 7, 2, 6, 4, 1];

        let bf = {
            let itms = items(lens);
            let mut index = BTreeRemainingIndex::new();
            greedy_pack(
                itms.into_iter(),
                10,
                &mut index,
                BTreeRemainingIndex::best_fit as ChooseFn<BTreeRemainingIndex>,
            )
            .unwrap()
        };

        let bfd = bfd_bins(lens, 10);
        assert!(bfd.len() <= bf.len());
    }

    // ── Error cases ───────────────────────────────────────────────

    #[test]
    fn test_oversize_error() {
        let result = BestFitDecreasing.pack(seqs(&[11]), 10);
        assert!(matches!(
            result,
            Err(PackError::SequenceTooLong {
                length: 11,
                capacity: 10
            })
        ));
    }

    #[test]
    fn test_oversize_detected_after_sort() {
        let result = BestFitDecreasing.pack(seqs(&[3, 15, 5]), 10);
        assert!(result.is_err());
    }

    // ── Edge cases ────────────────────────────────────────────────

    #[test]
    fn test_empty_input() {
        let packs = BestFitDecreasing.pack(seqs(&[]), 10).unwrap();
        assert!(packs.is_empty());
    }

    #[test]
    fn test_single_item() {
        let packs = BestFitDecreasing.pack(seqs(&[5]), 10).unwrap();
        assert_eq!(packs.len(), 1);
    }

    #[test]
    fn test_exact_capacity() {
        let bins = bfd_bins(&[10], 10);
        assert_eq!(bins[0].remaining(), 0);
    }

    #[test]
    fn test_all_same_size() {
        let bins = bfd_bins(&[5, 5, 5, 5], 10);
        assert_eq!(bins.len(), 2);
    }

    #[test]
    fn test_each_item_needs_own_bin() {
        let bins = bfd_bins(&[6, 7, 8, 9], 10);
        assert_eq!(bins.len(), 4);
    }

    #[test]
    fn test_many_small_items() {
        let bins = bfd_bins(&[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 10);
        assert_eq!(bins.len(), 1);
    }

    // ── Validation ────────────────────────────────────────────────

    #[test]
    fn test_validates_basic() {
        let original = items(&[3, 5, 8, 7, 4]);
        let sorted = items_sorted_desc(&[3, 5, 8, 7, 4]);
        let mut index = BTreeRemainingIndex::new();
        let bins = greedy_pack(
            sorted.into_iter(),
            10,
            &mut index,
            BTreeRemainingIndex::best_fit as ChooseFn<BTreeRemainingIndex>,
        )
        .unwrap();
        // validate_solution indexes items by ID, so pass original (unsorted) items
        validate_solution(&original, &bins, 10).unwrap();
    }

    #[test]
    fn test_validates_many_items() {
        let original = items(&[6, 4, 3, 7, 5, 5, 2, 8, 1, 9]);
        let sorted = items_sorted_desc(&[6, 4, 3, 7, 5, 5, 2, 8, 1, 9]);
        let mut index = BTreeRemainingIndex::new();
        let bins = greedy_pack(
            sorted.into_iter(),
            10,
            &mut index,
            BTreeRemainingIndex::best_fit as ChooseFn<BTreeRemainingIndex>,
        )
        .unwrap();
        validate_solution(&original, &bins, 10).unwrap();
    }

    // ── Trait ─────────────────────────────────────────────────────

    #[test]
    fn test_name() {
        assert_eq!(BestFitDecreasing.name(), "BestFitDecreasing");
    }

    #[test]
    fn test_pack_through_trait() {
        let packs = BestFitDecreasing.pack(seqs(&[3, 7, 5, 5]), 10).unwrap();
        // Sorted: [7, 5, 5, 3] → 7→b0(rem=3), 5→b1(rem=5), 5→b1(rem=5,tightest)→rem=0,
        // 3→b0(rem=3,exact)→rem=0
        assert_eq!(packs.len(), 2);
    }
}
