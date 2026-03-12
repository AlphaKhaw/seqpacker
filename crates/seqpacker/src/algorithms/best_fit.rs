//! Best Fit algorithm — O(n log B), tightest fitting bin.

use crate::engine::{ChooseFn, greedy_pack};
use crate::error::Result;
use crate::pack::{Pack, bins_to_packs};
use crate::placement::{BTreeRemainingIndex, PlacementIndex};
use crate::sequence::Sequence;
use crate::strategy::PackingAlgorithm;

/// Best Fit algorithm implementation.
pub struct BestFit;

impl PackingAlgorithm for BestFit {
    fn pack(&self, sequences: Vec<Sequence>, capacity: usize) -> Result<Vec<Pack>> {
        let items: Vec<_> = sequences.iter().map(|s| s.to_item()).collect();
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
        "BestFit"
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

    /// Run best fit at the engine level and return bins for direct inspection.
    fn bf_bins(lens: &[usize], capacity: usize) -> Vec<crate::pack::Bin> {
        let itms = items(lens);
        let mut index = BTreeRemainingIndex::new();
        greedy_pack(
            itms.into_iter(),
            capacity,
            &mut index,
            BTreeRemainingIndex::best_fit as ChooseFn<BTreeRemainingIndex>,
        )
        .unwrap()
    }

    // ── Core behavior: tightest fit ─────────────────────────────

    #[test]
    fn test_picks_tightest_bin() {
        // Set up two bins with different remaining, then insert item that fits both.
        // BF must pick the tighter (smaller remaining) one.
        let bins = bf_bins(&[7, 4, 3], 10);
        // 7→bin0(rem=3), 4→bin1(rem=6), 3→bin0(rem=3, tightest≥3) not bin1(rem=6)
        assert_eq!(bins.len(), 2);
        assert_eq!(&bins[0].items[..], &[0, 2]); // 7+3=10
        assert_eq!(&bins[1].items[..], &[1]); // 4
    }

    #[test]
    fn test_exact_fit_preferred() {
        // bin0(rem=5), bin1(rem=3). Item len=3 → exact fit in bin1.
        let bins = bf_bins(&[5, 7, 3], 10);
        // 5→bin0(rem=5), 7→bin1(rem=3), 3→bin1(rem=3, exact!)
        assert_eq!(bins.len(), 2);
        assert_eq!(&bins[0].items[..], &[0]); // 5
        assert_eq!(&bins[1].items[..], &[1, 2]); // 7+3=10
    }

    #[test]
    fn test_smallest_remaining_not_leftmost() {
        // Unlike FF which picks leftmost, BF picks tightest.
        // bin0(rem=5), bin1(rem=3). Item len=2.
        // FF would pick bin0 (leftmost with rem≥2).
        // BF picks bin1 (smallest remaining ≥ 2 is 3).
        let bins = bf_bins(&[5, 7, 2], 10);
        // 5→bin0(rem=5), 7→bin1(rem=3), 2→bin1(rem=3≥2, tightest)
        assert_eq!(bins.len(), 2);
        assert_eq!(&bins[0].items[..], &[0]); // 5 alone
        assert_eq!(&bins[1].items[..], &[1, 2]); // 7+2=9
    }

    #[test]
    fn test_five_items_walkthrough() {
        let bins = bf_bins(&[7, 5, 3, 8, 4], 10);
        // 7→bin0(rem=3), 5→bin1(rem=5), 3→bin0(rem=3, tightest≥3), 8→bin2(rem=2), 4→bin1(rem=5≥4, tightest≥4)
        // BTree after step 2: {3:[0], 5:[1]}
        // Item 3 (len=3): range(3..).next() → key=3 → bin0. update: {0:[0], 5:[1]}
        // Item 4 (len=8): range(8..).next() → None. new bin2. {0:[0], 2:[2], 5:[1]}
        // Item 5 (len=4): range(4..).next() → key=5 → bin1. update: {0:[0], 1:[1], 2:[2]}
        assert_eq!(bins.len(), 3);
        assert_eq!(&bins[0].items[..], &[0, 2]); // 7+3=10
        assert_eq!(&bins[1].items[..], &[1, 4]); // 5+4=9
        assert_eq!(&bins[2].items[..], &[3]); // 8
    }

    // ── Comparison with FF ──────────────────────────────────────

    #[test]
    fn test_differs_from_first_fit() {
        // bin0(rem=5), bin1(rem=3). Item len=3.
        // FF: picks bin0 (leftmost with rem≥3).
        // BF: picks bin1 (tightest, rem=3 exact).
        let bins = bf_bins(&[5, 7, 3], 10);
        assert_eq!(&bins[1].items[..], &[1, 2]); // BF puts item 2 in bin1 (tighter)
    }

    // ── Error cases ─────────────────────────────────────────────

    #[test]
    fn test_oversize_error() {
        let result = BestFit.pack(seqs(&[11]), 10);
        assert!(matches!(
            result,
            Err(PackError::SequenceTooLong {
                length: 11,
                capacity: 10
            })
        ));
    }

    // ── Edge cases ──────────────────────────────────────────────

    #[test]
    fn test_empty_input() {
        let packs = BestFit.pack(seqs(&[]), 10).unwrap();
        assert!(packs.is_empty());
    }

    #[test]
    fn test_single_item() {
        let packs = BestFit.pack(seqs(&[5]), 10).unwrap();
        assert_eq!(packs.len(), 1);
    }

    #[test]
    fn test_exact_capacity() {
        let bins = bf_bins(&[10], 10);
        assert_eq!(bins[0].remaining(), 0);
    }

    #[test]
    fn test_all_same_size() {
        let bins = bf_bins(&[5, 5, 5, 5], 10);
        assert_eq!(bins.len(), 2);
    }

    #[test]
    fn test_each_item_needs_own_bin() {
        let bins = bf_bins(&[6, 7, 8, 9], 10);
        assert_eq!(bins.len(), 4);
    }

    #[test]
    fn test_many_small_items() {
        let bins = bf_bins(&[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 10);
        assert_eq!(bins.len(), 1);
        assert_eq!(bins[0].used, 10);
    }

    // ── Validation ──────────────────────────────────────────────

    #[test]
    fn test_validates_basic() {
        let itms = items(&[7, 5, 3, 8, 4]);
        let mut index = BTreeRemainingIndex::new();
        let bins = greedy_pack(
            itms.iter().copied(),
            10,
            &mut index,
            BTreeRemainingIndex::best_fit as ChooseFn<BTreeRemainingIndex>,
        )
        .unwrap();
        validate_solution(&itms, &bins, 10).unwrap();
    }

    #[test]
    fn test_validates_many_items() {
        let itms = items(&[6, 4, 3, 7, 5, 5, 2, 8, 1, 9]);
        let mut index = BTreeRemainingIndex::new();
        let bins = greedy_pack(
            itms.iter().copied(),
            10,
            &mut index,
            BTreeRemainingIndex::best_fit as ChooseFn<BTreeRemainingIndex>,
        )
        .unwrap();
        validate_solution(&itms, &bins, 10).unwrap();
    }

    // ── Trait ───────────────────────────────────────────────────

    #[test]
    fn test_name() {
        assert_eq!(BestFit.name(), "BestFit");
    }

    #[test]
    fn test_pack_through_trait() {
        let packs = BestFit.pack(seqs(&[6, 3, 4]), 10).unwrap();
        assert_eq!(packs.len(), 2);
    }
}
