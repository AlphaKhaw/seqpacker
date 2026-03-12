//! First Fit Shuffle — shuffle with deterministic seed, then First Fit.
//!
//! Used by NVIDIA NeMo to preserve training randomness while still
//! getting reasonable packing efficiency. The shuffle prevents the
//! model from always seeing sequences in the same length-based order.

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::engine::{ChooseFn, greedy_pack};
use crate::error::Result;
use crate::pack::{Pack, bins_to_packs};
use crate::placement::{PlacementIndex, SegmentTreeIndex};
use crate::sequence::{Item, Sequence};
use crate::strategy::PackingAlgorithm;

/// First Fit Shuffle with a deterministic seed.
pub struct FirstFitShuffle {
    /// Random seed for deterministic shuffling.
    pub seed: u64,
}

impl FirstFitShuffle {
    /// Create a new First Fit Shuffle packer with the given seed.
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }
}

impl PackingAlgorithm for FirstFitShuffle {
    fn pack(&self, sequences: Vec<Sequence>, capacity: usize) -> Result<Vec<Pack>> {
        // Shuffle lightweight items, not sequences.
        // Items keep original IDs, sequences stay unchanged,
        // so bins_to_packs can index correctly.
        let mut items: Vec<Item> = sequences.iter().map(|s| s.to_item()).collect();

        let mut rng = StdRng::seed_from_u64(self.seed);
        items.shuffle(&mut rng);

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
        "FirstFitShuffle"
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

    // ── Core behavior: deterministic shuffle + first fit ──────────

    #[test]
    fn test_seed_determinism() {
        let input = seqs(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        let packs1 = FirstFitShuffle::new(123).pack(input.clone(), 15).unwrap();
        let packs2 = FirstFitShuffle::new(123).pack(input, 15).unwrap();

        // Same seed → same pack count and same assignments.
        assert_eq!(packs1.len(), packs2.len());
        for (p1, p2) in packs1.iter().zip(packs2.iter()) {
            let ids1: Vec<usize> = p1.sequences.iter().map(|s| s.id).collect();
            let ids2: Vec<usize> = p2.sequences.iter().map(|s| s.id).collect();
            assert_eq!(ids1, ids2);
        }
    }

    #[test]
    fn test_different_seeds_differ() {
        let input = seqs(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        let packs1 = FirstFitShuffle::new(123).pack(input.clone(), 15).unwrap();
        let packs2 = FirstFitShuffle::new(456).pack(input, 15).unwrap();

        let ids1: Vec<Vec<usize>> = packs1
            .iter()
            .map(|p| p.sequences.iter().map(|s| s.id).collect())
            .collect();
        let ids2: Vec<Vec<usize>> = packs2
            .iter()
            .map(|p| p.sequences.iter().map(|s| s.id).collect())
            .collect();

        // With 10 items, probability of identical shuffle is 1/10! ≈ 0.
        assert_ne!(ids1, ids2);
    }

    #[test]
    fn test_shuffle_changes_order() {
        // With unsorted input, shuffled output should differ from FF on same input.
        // (Not guaranteed, but with seed=42 and 10 items it will.)
        let lens = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        // FF (original order)
        let ff_packs = {
            let itms = items(lens);
            let mut index = SegmentTreeIndex::with_capacity(10);
            greedy_pack(
                itms.into_iter(),
                15,
                &mut index,
                SegmentTreeIndex::first_fit as ChooseFn<SegmentTreeIndex>,
            )
            .unwrap()
        };

        // FFS (shuffled order)
        let ffs_packs = FirstFitShuffle::new(42).pack(seqs(lens), 15).unwrap();

        // Bin count may differ or be the same, but item assignment should differ.
        let ff_items: Vec<Vec<usize>> = ff_packs.iter().map(|b| b.items.to_vec()).collect();
        let ffs_items: Vec<Vec<usize>> = ffs_packs
            .iter()
            .map(|p| p.sequences.iter().map(|s| s.id).collect())
            .collect();
        assert_ne!(ff_items, ffs_items);
    }

    #[test]
    fn test_seed_zero_works() {
        // Edge case: seed=0 is valid.
        let packs = FirstFitShuffle::new(0).pack(seqs(&[3, 7, 5]), 10).unwrap();
        assert!(!packs.is_empty());
    }

    #[test]
    fn test_seed_max_works() {
        // Edge case: max u64 seed.
        let packs = FirstFitShuffle::new(u64::MAX)
            .pack(seqs(&[3, 7, 5]), 10)
            .unwrap();
        assert!(!packs.is_empty());
    }

    #[test]
    fn test_single_item_same_regardless_of_seed() {
        // With one item, shuffle has no effect.
        let p1 = FirstFitShuffle::new(1).pack(seqs(&[5]), 10).unwrap();
        let p2 = FirstFitShuffle::new(999).pack(seqs(&[5]), 10).unwrap();
        assert_eq!(p1.len(), 1);
        assert_eq!(p2.len(), 1);
    }

    #[test]
    fn test_two_items_same_length() {
        // Two items of same length — any shuffle produces the same packing.
        let p1 = FirstFitShuffle::new(1).pack(seqs(&[5, 5]), 10).unwrap();
        let p2 = FirstFitShuffle::new(2).pack(seqs(&[5, 5]), 10).unwrap();
        assert_eq!(p1.len(), p2.len());
    }

    // ── Error cases ───────────────────────────────────────────────

    #[test]
    fn test_oversize_error() {
        let result = FirstFitShuffle::new(42).pack(seqs(&[11]), 10);
        assert!(matches!(
            result,
            Err(PackError::SequenceTooLong {
                length: 11,
                capacity: 10
            })
        ));
    }

    #[test]
    fn test_oversize_in_middle() {
        let result = FirstFitShuffle::new(42).pack(seqs(&[3, 15, 5]), 10);
        assert!(result.is_err());
    }

    // ── Edge cases ────────────────────────────────────────────────

    #[test]
    fn test_empty_input() {
        let packs = FirstFitShuffle::new(42).pack(seqs(&[]), 10).unwrap();
        assert!(packs.is_empty());
    }

    #[test]
    fn test_exact_capacity() {
        let packs = FirstFitShuffle::new(42).pack(seqs(&[10]), 10).unwrap();
        assert_eq!(packs.len(), 1);
    }

    #[test]
    fn test_all_same_size() {
        let packs = FirstFitShuffle::new(42)
            .pack(seqs(&[5, 5, 5, 5]), 10)
            .unwrap();
        assert_eq!(packs.len(), 2);
    }

    #[test]
    fn test_each_item_needs_own_bin() {
        let packs = FirstFitShuffle::new(42)
            .pack(seqs(&[6, 7, 8, 9]), 10)
            .unwrap();
        assert_eq!(packs.len(), 4);
    }

    #[test]
    fn test_many_small_items() {
        let packs = FirstFitShuffle::new(42)
            .pack(seqs(&[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 10)
            .unwrap();
        assert_eq!(packs.len(), 1);
    }

    // ── Validation ────────────────────────────────────────────────

    #[test]
    fn test_validates_basic() {
        let original = items(&[3, 5, 8, 7, 4]);
        let mut shuffled = original.clone();
        let mut rng = StdRng::seed_from_u64(42);
        shuffled.shuffle(&mut rng);

        let mut index = SegmentTreeIndex::with_capacity(5);
        let bins = greedy_pack(
            shuffled.into_iter(),
            10,
            &mut index,
            SegmentTreeIndex::first_fit as ChooseFn<SegmentTreeIndex>,
        )
        .unwrap();
        validate_solution(&original, &bins, 10).unwrap();
    }

    #[test]
    fn test_validates_many_items() {
        let original = items(&[6, 4, 3, 7, 5, 5, 2, 8, 1, 9]);
        let mut shuffled = original.clone();
        let mut rng = StdRng::seed_from_u64(42);
        shuffled.shuffle(&mut rng);

        let mut index = SegmentTreeIndex::with_capacity(10);
        let bins = greedy_pack(
            shuffled.into_iter(),
            10,
            &mut index,
            SegmentTreeIndex::first_fit as ChooseFn<SegmentTreeIndex>,
        )
        .unwrap();
        validate_solution(&original, &bins, 10).unwrap();
    }

    // ── Trait ─────────────────────────────────────────────────────

    #[test]
    fn test_name() {
        assert_eq!(FirstFitShuffle::new(42).name(), "FirstFitShuffle");
    }

    #[test]
    fn test_pack_through_trait() {
        let packs = FirstFitShuffle::new(42)
            .pack(seqs(&[3, 7, 5, 5]), 10)
            .unwrap();
        // Shuffled order varies, but total items must be 4.
        let total_items: usize = packs.iter().map(|p| p.sequences.len()).sum();
        assert_eq!(total_items, 4);
    }
}
