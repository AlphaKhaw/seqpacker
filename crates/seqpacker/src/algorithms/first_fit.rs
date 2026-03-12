//! First Fit algorithm — O(n log B), leftmost bin with space.

use crate::engine::{ChooseFn, greedy_pack};
use crate::error::Result;
use crate::pack::{Pack, bins_to_packs};
use crate::placement::{PlacementIndex, SegmentTreeIndex};
use crate::sequence::{Item, Sequence};
use crate::strategy::PackingAlgorithm;

/// First Fit algorithm implementation.
pub struct FirstFit;

impl PackingAlgorithm for FirstFit {
    fn pack(&self, sequences: Vec<Sequence>, capacity: usize) -> Result<Vec<Pack>> {
        let items: Vec<Item> = sequences.iter().map(|s| s.to_item()).collect();
        let mut index = SegmentTreeIndex::with_capacity(items.len() / 2 + 1);
        let bins = greedy_pack(
            items.into_iter(),
            capacity,
            &mut index,
            SegmentTreeIndex::first_fit as ChooseFn<SegmentTreeIndex>,
        )?;
        Ok(bins_to_packs(bins, &sequences))
    }

    fn name(&self) -> &'static str {
        "FirstFit"
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

    /// Run first fit at the engine level and return bins for direct inspection.
    fn ff_bins(lens: &[usize], capacity: usize) -> Vec<crate::pack::Bin> {
        let itms = items(lens);
        let mut index = SegmentTreeIndex::with_capacity(itms.len() / 2 + 1);
        greedy_pack(
            itms.into_iter(),
            capacity,
            &mut index,
            SegmentTreeIndex::first_fit as ChooseFn<SegmentTreeIndex>,
        )
        .unwrap()
    }

    // ── Basic behavior ──────────────────────────────────────────

    #[test]
    fn test_basic() {
        let packs = FirstFit.pack(seqs(&[6, 4, 6]), 10).unwrap();
        // 6→bin0(rem=4), 4→bin0(rem=0), 6→bin1
        assert_eq!(packs.len(), 2);
    }

    #[test]
    fn test_leftmost_preference() {
        // Set up two bins with room, then insert an item that fits both.
        // FF must pick the leftmost (smallest bin_id).
        let bins = ff_bins(&[8, 7, 2, 3], 10);
        // 8→bin0(rem=2), 7→bin1(rem=3), 2→bin0(rem=2≥2, leftmost!), 3→bin1(rem=3≥3)
        assert_eq!(bins.len(), 2);
        assert_eq!(&bins[0].items[..], &[0, 2]); // 8+2=10
        assert_eq!(&bins[1].items[..], &[1, 3]); // 7+3=10
    }

    #[test]
    fn test_skips_full_bins() {
        let bins = ff_bins(&[10, 5, 5], 10);
        // 10→bin0(rem=0), 5→bin0? no→bin1(rem=5), 5→bin1(rem=0)
        assert_eq!(bins.len(), 2);
        assert_eq!(&bins[0].items[..], &[0]);
        assert_eq!(&bins[1].items[..], &[1, 2]);
    }

    #[test]
    fn test_revisits_earlier_bins() {
        // Unlike Next Fit, FF can go back to earlier bins.
        let bins = ff_bins(&[7, 8, 3, 2], 10);
        // 7→bin0(rem=3), 8→bin1(rem=2), 3→bin0(rem=3≥3, yes!), 2→bin1(rem=2≥2, yes!)
        assert_eq!(bins.len(), 2);
        assert_eq!(&bins[0].items[..], &[0, 2]); // 7+3=10
        assert_eq!(&bins[1].items[..], &[1, 3]); // 8+2=10
    }

    #[test]
    fn test_five_items_walkthrough() {
        let bins = ff_bins(&[7, 5, 3, 8, 4], 10);
        // 7→bin0(rem=3), 5→bin1(rem=5), 3→bin0(rem=0), 8→bin2(rem=2), 4→bin1(rem=1)
        assert_eq!(bins.len(), 3);
        assert_eq!(&bins[0].items[..], &[0, 2]); // 7+3=10
        assert_eq!(&bins[1].items[..], &[1, 4]); // 5+4=9
        assert_eq!(&bins[2].items[..], &[3]); // 8
    }

    // ── Comparison with Next Fit ────────────────────────────────

    #[test]
    fn test_fewer_bins_than_next_fit() {
        // NF gets 4 bins for [6,5,6,5] because it can't revisit.
        // FF: 6→bin0(rem=4), 5→bin1(rem=5), 6→bin2(rem=4), 5→bin1(rem=0)
        let bins = ff_bins(&[6, 5, 6, 5], 10);
        assert_eq!(bins.len(), 3); // FF gets 3, NF would get 4
    }

    // ── Error cases ─────────────────────────────────────────────

    #[test]
    fn test_oversize_error() {
        let result = FirstFit.pack(seqs(&[11]), 10);
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
        let packs = FirstFit.pack(seqs(&[]), 10).unwrap();
        assert!(packs.is_empty());
    }

    #[test]
    fn test_single_item() {
        let packs = FirstFit.pack(seqs(&[5]), 10).unwrap();
        assert_eq!(packs.len(), 1);
    }

    #[test]
    fn test_exact_capacity() {
        let bins = ff_bins(&[10], 10);
        assert_eq!(bins[0].remaining(), 0);
    }

    #[test]
    fn test_all_same_size() {
        let bins = ff_bins(&[5, 5, 5, 5], 10);
        assert_eq!(bins.len(), 2);
        assert_eq!(&bins[0].items[..], &[0, 1]);
        assert_eq!(&bins[1].items[..], &[2, 3]);
    }

    #[test]
    fn test_each_item_needs_own_bin() {
        let bins = ff_bins(&[6, 7, 8, 9], 10);
        assert_eq!(bins.len(), 4);
    }

    #[test]
    fn test_many_small_items() {
        let bins = ff_bins(&[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 10);
        assert_eq!(bins.len(), 1);
        assert_eq!(bins[0].used, 10);
    }

    // ── Validation ──────────────────────────────────────────────

    #[test]
    fn test_validates_basic() {
        let itms = items(&[7, 5, 3, 8, 4]);
        let mut index = SegmentTreeIndex::with_capacity(5);
        let bins = greedy_pack(
            itms.iter().copied(),
            10,
            &mut index,
            SegmentTreeIndex::first_fit as ChooseFn<SegmentTreeIndex>,
        )
        .unwrap();
        validate_solution(&itms, &bins, 10).unwrap();
    }

    #[test]
    fn test_validates_many_items() {
        let lens: Vec<usize> = vec![6, 4, 3, 7, 5, 5, 2, 8, 1, 9];
        let itms = items(&lens);
        let mut index = SegmentTreeIndex::with_capacity(10);
        let bins = greedy_pack(
            itms.iter().copied(),
            10,
            &mut index,
            SegmentTreeIndex::first_fit as ChooseFn<SegmentTreeIndex>,
        )
        .unwrap();
        validate_solution(&itms, &bins, 10).unwrap();
    }

    // ── Trait ───────────────────────────────────────────────────

    #[test]
    fn test_name() {
        assert_eq!(FirstFit.name(), "FirstFit");
    }
}
