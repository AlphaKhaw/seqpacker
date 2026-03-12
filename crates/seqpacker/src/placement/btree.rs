//! BTree-based PlacementIndex for O(log B) best-fit and worst-fit queries.

use std::collections::BTreeMap;

use super::PlacementIndex;

/// BTreeMap keyed by remaining capacity, values are bin IDs.
///
/// Optimized for `best_fit` and `worst_fit` queries via `range()`.
/// Also supports `first_fit` but as O(B) fallback (use `SegmentTreeIndex`
/// for fast first-fit).
#[derive(Clone, Debug, Default)]
pub struct BTreeRemainingIndex {
    /// remaining_capacity → list of bin IDs with that remaining.
    map: BTreeMap<usize, Vec<usize>>,
    /// Direct lookup: bin_id → remaining_capacity.
    remaining: Vec<usize>,
}

impl BTreeRemainingIndex {
    /// Create an empty BTree-based placement index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Remove a bin_id from the vec at the given key. Clean up empty keys.
    #[inline(always)]
    fn remove_from_key(&mut self, key: usize, bin_id: usize) {
        if let Some(ids) = self.map.get_mut(&key) {
            ids.retain(|&id| id != bin_id);
            if ids.is_empty() {
                self.map.remove(&key);
            }
        }
    }

    /// Add a bin_id at the given index.
    #[inline(always)]
    fn add_to_key(&mut self, key: usize, bin_id: usize) {
        self.map.entry(key).or_default().push(bin_id)
    }
}

impl PlacementIndex for BTreeRemainingIndex {
    #[inline(always)]
    fn insert_bin(&mut self, bin_id: usize, remaining: usize) {
        // Extend remaining vec if needed.
        if bin_id >= self.remaining.len() {
            self.remaining.resize(bin_id + 1, 0);
        }
        self.remaining[bin_id] = remaining;
        self.add_to_key(remaining, bin_id);
    }

    #[inline(always)]
    fn update_bin(&mut self, bin_id: usize, old_remaining: usize, new_remaining: usize) {
        self.remove_from_key(old_remaining, bin_id);
        self.remaining[bin_id] = new_remaining;
        self.add_to_key(new_remaining, bin_id);
    }

    fn first_fit(&self, needed: usize) -> Option<usize> {
        // BTree is not optimized for leftmost queries. Fall back to linear.
        self.remaining
            .iter()
            .enumerate()
            .filter(|&(_, &rem)| rem >= needed)
            .min_by_key(|(id, _)| *id)
            .map(|(id, _)| id)
    }

    #[inline(always)]
    fn best_fit(&self, needed: usize) -> Option<usize> {
        // Smallest remaining >= needed.
        // BTreeMap::range(needed..) iterates keys in ascending order.
        // .next() gives the smallest key >= needed.
        self.map
            .range(needed..)
            .next()
            .and_then(|(_, ids)| ids.first().copied())
    }

    #[inline(always)]
    fn worst_fit(&self, needed: usize) -> Option<usize> {
        // Largest remaining >= needed.
        // .next_back() on the range gives the largest key >= needed.
        self.map
            .range(needed..)
            .next_back()
            .and_then(|(_, ids)| ids.first().copied())
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::LinearScanIndex;

    #[test]
    fn test_best_fit_tightest() {
        let mut idx = BTreeRemainingIndex::new();
        idx.insert_bin(0, 9);
        idx.insert_bin(1, 6);
        idx.insert_bin(2, 6);

        // item len=5: smallest remaining >= 5 is 6 (bins 1 or 2).
        let result = idx.best_fit(5);
        assert!(result == Some(1) || result == Some(2));
    }

    #[test]
    fn test_worst_fit_loosest() {
        let mut idx = BTreeRemainingIndex::new();
        idx.insert_bin(0, 3);
        idx.insert_bin(1, 9);
        idx.insert_bin(2, 6);

        // item len=5: largest remaining >= 5 is 9 → bin 1.
        assert_eq!(idx.worst_fit(5), Some(1));
    }

    #[test]
    fn test_update_bin_moves_between_keys() {
        let mut idx = BTreeRemainingIndex::new();
        idx.insert_bin(0, 10);
        idx.insert_bin(1, 10);

        // Place item of size 7 in bin 0.
        idx.update_bin(0, 10, 3);

        // best_fit(5) should pick bin 1 (remaining 10), not bin 0 (remaining 3).
        assert_eq!(idx.best_fit(5), Some(1));
        // best_fit(2) should pick bin 0 (remaining 3, tighter than 10).
        assert_eq!(idx.best_fit(2), Some(0));
    }

    #[test]
    fn test_update_bin_cleans_up_empty_keys() {
        let mut idx = BTreeRemainingIndex::new();
        idx.insert_bin(0, 10);
        idx.insert_bin(1, 5);

        // Remove only bin at key 5.
        idx.update_bin(1, 5, 2);

        // Key 5 should be gone. best_fit(4) should skip to key 10.
        assert_eq!(idx.best_fit(4), Some(0));
        // best_fit(1) should find the tighter key 2.
        assert_eq!(idx.best_fit(1), Some(1));
    }

    #[test]
    fn test_first_fit_leftmost() {
        let mut idx = BTreeRemainingIndex::new();
        idx.insert_bin(0, 2);
        idx.insert_bin(1, 7);
        idx.insert_bin(2, 7);

        // Leftmost bin with remaining >= 6 is bin 1.
        assert_eq!(idx.first_fit(6), Some(1));
        // Nothing fits 8.
        assert_eq!(idx.first_fit(8), None);
    }

    #[test]
    fn test_no_fit_returns_none() {
        let mut idx = BTreeRemainingIndex::new();
        idx.insert_bin(0, 3);
        idx.insert_bin(1, 5);

        assert_eq!(idx.best_fit(10), None);
        assert_eq!(idx.worst_fit(10), None);
        assert_eq!(idx.first_fit(10), None);
    }

    #[test]
    fn test_empty_index() {
        let idx = BTreeRemainingIndex::new();
        assert_eq!(idx.best_fit(1), None);
        assert_eq!(idx.worst_fit(1), None);
        assert_eq!(idx.first_fit(1), None);
    }

    #[test]
    fn test_agrees_with_linear_on_best_and_worst() {
        let remainders = [5, 12, 3, 8, 15, 1, 9, 7];

        let mut btree = BTreeRemainingIndex::new();
        let mut lin = LinearScanIndex::new();

        for (id, &rem) in remainders.iter().enumerate() {
            btree.insert_bin(id, rem);
            lin.insert_bin(id, rem);
        }

        for needed in 0..=20 {
            // Best fit: both should pick a bin with the same remaining.
            // (Tie-breaking may differ, so compare remaining not bin_id.)
            let bt_bf = btree.best_fit(needed);
            let li_bf = lin.best_fit(needed);
            match (bt_bf, li_bf) {
                (Some(a), Some(b)) => assert_eq!(
                    remainders[a], remainders[b],
                    "best_fit({needed}): btree picked bin {a} (rem={}), linear picked bin {b} (rem={})",
                    remainders[a], remainders[b]
                ),
                (None, None) => {}
                _ => panic!("best_fit({needed}): one returned None, other didn't"),
            }

            // Worst fit: same logic.
            let bt_wf = btree.worst_fit(needed);
            let li_wf = lin.worst_fit(needed);
            match (bt_wf, li_wf) {
                (Some(a), Some(b)) => assert_eq!(
                    remainders[a], remainders[b],
                    "worst_fit({needed}): btree picked bin {a} (rem={}), linear picked bin {b} (rem={})",
                    remainders[a], remainders[b]
                ),
                (None, None) => {}
                _ => panic!("worst_fit({needed}): one returned None, other didn't"),
            }
        }
    }
}
