//! Generic greedy packing engine.
//!
//! This single function implements FF, BF, WF, FFD, BFD, and FFS —
//! the only differences are preprocessing (sort/shuffle) and
//! which `choose` function is passed.

use crate::error::{PackError, Result};
use crate::pack::Bin;
use crate::placement::PlacementIndex;
use crate::sequence::Item;

/// Choose function type: given an index and a needed capacity,
/// returns the bin_id to place the item in, or None if no bin fits.
pub type ChooseFn<I> = fn(&I, usize) -> Option<usize>;

/// Pack items into bins using a greedy strategy.
///
/// # Arguments
///
/// - `items`: Iterator of items to pack (order matters — FF processes
///   left-to-right, FFD pre-sorts, FFS pre-shuffles).
/// - `capacity`: Maximum bin capacity.
/// - `index`: Mutable reference to a PlacementIndex for bin selection.
/// - `choose`: Function that selects a bin from the index for an item
///   of the given size. One of `first_fit`, `best_fit`, or `worst_fit`.
///
/// # Errors
///
/// Returns `PackError::SequenceTooLong` if any item's length exceeds capacity.
///
/// # Algorithm
///
/// ```text
/// for each item:
///     if item.len > capacity → error
///     if choose(index, item.len) returns Some(bin_id):
///         place item in that bin
///         update index with new remaining
///     else:
///         create new bin with item
///         insert new bin into index
/// ```
#[inline]
pub fn greedy_pack<I: PlacementIndex>(
    items: impl Iterator<Item = Item>,
    capacity: usize,
    index: &mut I,
    choose: ChooseFn<I>,
) -> Result<Vec<Bin>> {
    let mut bins: Vec<Bin> = Vec::new();

    for item in items {
        if item.len > capacity {
            return Err(PackError::SequenceTooLong {
                length: item.len,
                capacity,
            });
        }

        match choose(index, item.len) {
            Some(bin_id) => {
                let old_remaining = bins[bin_id].remaining();
                bins[bin_id].used += item.len;
                bins[bin_id].items.push(item.id);
                index.update_bin(bin_id, old_remaining, bins[bin_id].remaining());
            }
            None => {
                let bin_id = bins.len();
                create_new_bins(&mut bins, bin_id, capacity, item);
                index.insert_bin(bin_id, bins[bin_id].remaining());
            }
        }
    }
    Ok(bins)
}

#[cold]
fn create_new_bins(bins: &mut Vec<Bin>, bin_id: usize, capacity: usize, item: Item) {
    let mut bin = Bin::new(bin_id, capacity);
    bin.used = item.len;
    bin.items.push(item.id);
    bins.push(bin);
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::{BTreeRemainingIndex, LinearScanIndex, SegmentTreeIndex};
    use crate::validation::validate_solution;

    fn make_items(lens: &[usize]) -> Vec<Item> {
        lens.iter()
            .enumerate()
            .map(|(id, &len)| Item { id, len })
            .collect()
    }

    // ── First Fit ────────────────────────────────────────────────

    #[test]
    fn test_first_fit_basic() {
        let items = make_items(&[6, 4, 6]);
        let mut index = LinearScanIndex::new();

        let bins =
            greedy_pack(items.into_iter(), 10, &mut index, PlacementIndex::first_fit).unwrap();

        // 6 → bin 0 (rem=4), 4 → bin 0 (rem=0), 6 → bin 1
        assert_eq!(bins.len(), 2);
        assert_eq!(&bins[0].items[..], &[0, 1]);
        assert_eq!(&bins[1].items[..], &[2]);
    }

    #[test]
    fn test_first_fit_with_segment_tree() {
        let items = make_items(&[7, 5, 3, 8, 4]);
        let mut index = SegmentTreeIndex::new();

        let bins = greedy_pack(
            items.iter().copied(),
            10,
            &mut index,
            SegmentTreeIndex::first_fit,
        )
        .unwrap();

        // 7→bin0(rem3), 5→bin1(rem5), 3→bin0(rem0), 8→bin2(rem2), 4→bin1(rem1)
        assert_eq!(bins.len(), 3);
        assert_eq!(&bins[0].items[..], &[0, 2]); // 7+3=10
        assert_eq!(&bins[1].items[..], &[1, 4]); // 5+4=9
        assert_eq!(&bins[2].items[..], &[3]); // 8
        validate_solution(&items, &bins, 10).unwrap();
    }

    // ── Best Fit ─────────────────────────────────────────────────

    #[test]
    fn test_best_fit_picks_tightest() {
        // Two bins: rem=3, rem=5. Item of len=3 should go to rem=3 (exact fit).
        let items = make_items(&[7, 5, 3]);
        let mut index = BTreeRemainingIndex::new();

        let bins = greedy_pack(
            items.into_iter(),
            10,
            &mut index,
            BTreeRemainingIndex::best_fit,
        )
        .unwrap();

        // 7→bin0(rem3), 5→bin1(rem5), 3→bin0(rem0) — tightest is bin0
        assert_eq!(bins.len(), 2);
        assert_eq!(&bins[0].items[..], &[0, 2]); // 7+3=10, exact fit
        assert_eq!(&bins[1].items[..], &[1]); // 5
    }

    // ── Worst Fit ────────────────────────────────────────────────

    #[test]
    fn test_worst_fit_picks_loosest() {
        // Two bins: rem=3, rem=5. Item of len=3 should go to rem=5 (loosest).
        let items = make_items(&[7, 5, 3]);
        let mut index = BTreeRemainingIndex::new();

        let bins = greedy_pack(
            items.into_iter(),
            10,
            &mut index,
            BTreeRemainingIndex::worst_fit,
        )
        .unwrap();

        // 7→bin0(rem3), 5→bin1(rem5), 3→bin1(rem2) — loosest is bin1
        assert_eq!(bins.len(), 2);
        assert_eq!(&bins[0].items[..], &[0]); // 7
        assert_eq!(&bins[1].items[..], &[1, 2]); // 5+3=8
    }

    // ── Preprocessing (simulating FFD/BFD) ───────────────────────

    #[test]
    fn test_sorted_descending_best_fit() {
        let items = make_items(&[3, 7, 5, 8, 4]);
        let mut sorted: Vec<Item> = items.clone();
        sorted.sort_unstable_by(|a, b| b.len.cmp(&a.len));

        let mut index = BTreeRemainingIndex::new();
        let bins = greedy_pack(
            sorted.into_iter(),
            10,
            &mut index,
            BTreeRemainingIndex::best_fit,
        )
        .unwrap();

        // Sorted: [8,7,5,4,3] → 8→bin0(rem2), 7→bin1(rem3), 5→bin2(rem5),
        //                        4→bin2(rem1, tightest≥4 is 5), 3→bin1(rem0)
        validate_solution(&items, &bins, 10).unwrap();
        assert_eq!(bins.len(), 3);
    }

    // ── Error Cases ──────────────────────────────────────────────

    #[test]
    fn test_oversize_item_returns_error() {
        let items = make_items(&[5, 15, 3]);
        let mut index = LinearScanIndex::new();

        let result = greedy_pack(items.into_iter(), 10, &mut index, PlacementIndex::first_fit);

        assert!(matches!(
            result,
            Err(PackError::SequenceTooLong {
                length: 15,
                capacity: 10
            })
        ));
    }

    #[test]
    fn test_oversize_item_stops_early() {
        // The error should occur at item 1 (len=15), item 2 is never processed.
        let items = make_items(&[5, 15, 3]);
        let mut index = LinearScanIndex::new();

        let result = greedy_pack(items.into_iter(), 10, &mut index, PlacementIndex::first_fit);

        assert!(result.is_err());
    }

    // ── Edge Cases ───────────────────────────────────────────────

    #[test]
    fn test_empty_input() {
        let mut index = LinearScanIndex::new();
        let bins = greedy_pack(
            std::iter::empty(),
            10,
            &mut index,
            PlacementIndex::first_fit,
        )
        .unwrap();
        assert!(bins.is_empty());
    }

    #[test]
    fn test_single_item() {
        let items = make_items(&[5]);
        let mut index = LinearScanIndex::new();
        let bins =
            greedy_pack(items.into_iter(), 10, &mut index, PlacementIndex::first_fit).unwrap();
        assert_eq!(bins.len(), 1);
        assert_eq!(bins[0].used, 5);
        assert_eq!(bins[0].remaining(), 5);
    }

    #[test]
    fn test_exact_capacity_item() {
        let items = make_items(&[10]);
        let mut index = LinearScanIndex::new();
        let bins =
            greedy_pack(items.into_iter(), 10, &mut index, PlacementIndex::first_fit).unwrap();
        assert_eq!(bins.len(), 1);
        assert_eq!(bins[0].remaining(), 0);
    }

    #[test]
    fn test_all_same_size() {
        let items = make_items(&[5, 5, 5, 5]);
        let mut index = LinearScanIndex::new();
        let bins = greedy_pack(
            items.iter().copied(),
            10,
            &mut index,
            PlacementIndex::first_fit,
        )
        .unwrap();
        // Each bin holds exactly 2 items.
        assert_eq!(bins.len(), 2);
        validate_solution(&items, &bins, 10).unwrap();
    }

    #[test]
    fn test_each_item_needs_own_bin() {
        let items = make_items(&[6, 7, 8, 9]);
        let mut index = LinearScanIndex::new();
        let bins = greedy_pack(
            items.iter().copied(),
            10,
            &mut index,
            PlacementIndex::first_fit,
        )
        .unwrap();
        // No two items fit together (6+7=13>10, etc.)
        assert_eq!(bins.len(), 4);
        validate_solution(&items, &bins, 10).unwrap();
    }

    // ── Cross-Index Agreement ────────────────────────────────────

    #[test]
    fn test_all_indexes_same_bin_count_for_first_fit() {
        let items = make_items(&[7, 3, 5, 8, 2, 6, 4, 1]);

        let mut lin = LinearScanIndex::new();
        let bins_lin = greedy_pack(
            items.iter().copied(),
            10,
            &mut lin,
            PlacementIndex::first_fit,
        )
        .unwrap();

        let mut seg = SegmentTreeIndex::new();
        let bins_seg = greedy_pack(
            items.iter().copied(),
            10,
            &mut seg,
            SegmentTreeIndex::first_fit,
        )
        .unwrap();

        let mut bt = BTreeRemainingIndex::new();
        let bins_bt = greedy_pack(
            items.iter().copied(),
            10,
            &mut bt,
            BTreeRemainingIndex::first_fit,
        )
        .unwrap();

        // All three should produce identical results for first_fit.
        assert_eq!(bins_lin.len(), bins_seg.len());
        assert_eq!(bins_lin.len(), bins_bt.len());
        for i in 0..bins_lin.len() {
            assert_eq!(bins_lin[i].items, bins_seg[i].items);
            assert_eq!(bins_lin[i].items, bins_bt[i].items);
        }
    }
}
