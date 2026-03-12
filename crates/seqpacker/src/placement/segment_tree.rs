//! Segment tree PlacementIndex for O(log B) first-fit queries.
//!
//! The segment tree stores the maximum remaining capacity in each subtree.
//! A first-fit query traverses left-first to find the leftmost bin with
//! remaining capacity >= needed.

use super::PlacementIndex;

/// Segment tree over bin remaining capacities.
///
/// Optimized for `first_fit` queries (leftmost bin with remaining ≥ needed).
/// Also supports `best_fit` and `worst_fit` but these are O(B) fallbacks
/// (use `BTreeRemainingIndex` if you need fast best/worst fit).
///
/// The tree is stored as a flat array in 1-indexed layout:
/// - Node 1 is the root
/// - Node i's children are 2i and 2i+1
/// - Leaves start at index `size` (where `size` is the next power of 2 ≥ B)
#[derive(Clone, Debug)]
pub struct SegmentTreeIndex {
    /// Flat array storing max-remaining per subtree. 1-indexed.
    tree: Vec<usize>,
    /// Number of leaf slots (next power of 2 >= actual bin count).
    size: usize,
    /// Actual number of bins inserted so far.
    num_bins: usize,
    /// Remaining capacity per bin (for direct lookup).
    remaining: Vec<usize>,
}

impl SegmentTreeIndex {
    /// Create an empty segment tree.
    ///
    /// The tree grows dynamically as bins are inserted. Start with
    /// a reasonable initial capacity to avoid frequent reallocations.
    pub fn new() -> Self {
        Self::with_capacity(16)
    }

    /// Create with an expected bin count hint.
    pub fn with_capacity(expected_bins: usize) -> Self {
        let size = expected_bins.next_power_of_two();
        Self {
            tree: vec![0; 2 * size],
            size,
            num_bins: 0,
            remaining: Vec::with_capacity(expected_bins),
        }
    }

    /// Ensure the tree has enough capacity for `bin_id`.
    fn ensure_capacity(&mut self, bin_id: usize) {
        if bin_id < self.size {
            return;
        }

        // Double the size until it fits.
        let mut new_size = self.size;
        while bin_id >= new_size {
            new_size *= 2;
        }

        // Rebuild tree with new size.
        let mut new_tree = vec![0; 2 * new_size];

        // Copy leaf values.
        new_tree[new_size..(self.num_bins + new_size)]
            .copy_from_slice(&self.remaining[..self.num_bins]);
        // Rebuild internal nodes.
        for i in (1..new_size).rev() {
            new_tree[i] = new_tree[2 * i].max(new_tree[2 * i + 1]);
        }
        self.tree = new_tree;
        self.size = new_size;
    }

    /// Update a leaf and propagate up to the root.
    #[inline(always)]
    fn update_leaf(&mut self, bin_id: usize, value: usize) {
        let mut pos = self.size + bin_id;
        self.tree[pos] = value;
        pos /= 2;
        while pos >= 1 {
            self.tree[pos] = self.tree[2 * pos].max(self.tree[2 * pos + 1]);
            pos /= 2;
        }
    }

    /// Find leftmost leaf in subtree rooted at `node` with value >= `needed`.
    ///
    /// `node_left` and `node_right` define the range of leaf indices
    /// covered by this node (inclusive).
    fn query_leftmost(
        &self,
        node: usize,
        node_left: usize,
        node_right: usize,
        needed: usize,
    ) -> Option<usize> {
        // No bin in this subtree has enough remaining.
        if self.tree[node] < needed {
            return None;
        }

        // Leaf node.
        if node_left == node_right {
            return if node_left < self.num_bins {
                Some(node_left)
            } else {
                None
            };
        }

        // Recursive left-first (leftmost preference).
        let mid = (node_left + node_right) / 2;
        if let Some(id) = self.query_leftmost(2 * node, node_left, mid, needed) {
            return Some(id);
        }
        self.query_leftmost(2 * node + 1, mid + 1, node_right, needed)
    }
}

impl Default for SegmentTreeIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl PlacementIndex for SegmentTreeIndex {
    #[inline(always)]
    fn insert_bin(&mut self, bin_id: usize, remaining: usize) {
        self.ensure_capacity(bin_id);
        if bin_id == self.remaining.len() {
            self.remaining.push(remaining);
        } else {
            self.remaining[bin_id] = remaining;
        }
        self.num_bins = self.num_bins.max(bin_id + 1);
        self.update_leaf(bin_id, remaining);
    }

    #[inline(always)]
    fn update_bin(&mut self, bin_id: usize, _old_remaining: usize, new_remaining: usize) {
        self.remaining[bin_id] = new_remaining;
        self.update_leaf(bin_id, new_remaining);
    }

    #[inline(always)]
    fn first_fit(&self, needed: usize) -> Option<usize> {
        if self.num_bins == 0 || self.tree[1] < needed {
            return None;
        }
        self.query_leftmost(1, 0, self.size - 1, needed)
    }

    fn best_fit(&self, needed: usize) -> Option<usize> {
        // Segment tree is not optimized for best-fit. Fall back to linear scan.
        // Use BTreeRemainingIndex for O(log B) for best-fit.
        self.remaining
            .iter()
            .enumerate()
            .filter(|&(_, &rem)| rem >= needed)
            .min_by_key(|&(id, &rem)| (rem, id))
            .map(|(id, _)| id)
    }

    fn worst_fit(&self, needed: usize) -> Option<usize> {
        // Same fallback as best_fit.
        self.remaining
            .iter()
            .enumerate()
            .filter(|&(_, &rem)| rem >= needed)
            .max_by_key(|&(_, &rem)| rem)
            .map(|(id, _)| id)
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::LinearScanIndex;

    // ── insert_bin ──────────────────────────────────────────────────

    #[test]
    fn test_insert_bin() {
        let mut idx = SegmentTreeIndex::new();
        idx.insert_bin(0, 100);
        idx.insert_bin(1, 50);
        assert_eq!(idx.first_fit(50), Some(0));
        assert_eq!(idx.first_fit(100), Some(0));
    }

    #[test]
    fn test_insert_many_bins() {
        let mut idx = SegmentTreeIndex::new();
        for i in 0..100 {
            idx.insert_bin(i, i + 1);
        }
        // Bin 99 has remaining=100, largest.
        assert_eq!(idx.first_fit(100), Some(99));
        // Bin 0 has remaining=1, smallest that fits 1.
        assert_eq!(idx.first_fit(1), Some(0));
    }

    // ── update_bin ──────────────────────────────────────────────────

    #[test]
    fn test_update_bin() {
        let mut idx = SegmentTreeIndex::new();
        idx.insert_bin(0, 10);
        idx.insert_bin(1, 10);

        idx.update_bin(0, 10, 3);

        assert_eq!(idx.first_fit(5), Some(1));
        assert_eq!(idx.first_fit(3), Some(0));
    }

    #[test]
    fn test_update_bin_to_zero() {
        let mut idx = SegmentTreeIndex::new();
        idx.insert_bin(0, 10);

        idx.update_bin(0, 10, 0);

        assert_eq!(idx.first_fit(1), None);
    }

    // ── first_fit ───────────────────────────────────────────────────

    #[test]
    fn test_first_fit_leftmost() {
        let mut idx = SegmentTreeIndex::new();
        idx.insert_bin(0, 2);
        idx.insert_bin(1, 7);
        idx.insert_bin(2, 7);

        // Both bin 1 and 2 fit, first_fit returns leftmost (1).
        assert_eq!(idx.first_fit(6), Some(1));
    }

    #[test]
    fn test_first_fit_none() {
        let mut idx = SegmentTreeIndex::new();
        idx.insert_bin(0, 5);
        idx.insert_bin(1, 3);

        assert_eq!(idx.first_fit(6), None);
    }

    #[test]
    fn test_first_fit_exact() {
        let mut idx = SegmentTreeIndex::new();
        idx.insert_bin(0, 5);

        assert_eq!(idx.first_fit(5), Some(0));
    }

    #[test]
    fn test_first_fit_empty_index() {
        let idx = SegmentTreeIndex::new();
        assert_eq!(idx.first_fit(1), None);
    }

    #[test]
    fn test_first_fit_skips_left_subtree() {
        // Verify the O(log B) pruning behaviour.
        let mut idx = SegmentTreeIndex::with_capacity(4);
        idx.insert_bin(0, 3);
        idx.insert_bin(1, 5);
        idx.insert_bin(2, 8);
        idx.insert_bin(3, 3);

        // needed=6: bins 0(3) and 1(5) can't fit → left subtree pruned.
        // bin 2(8) is the leftmost fit.
        assert_eq!(idx.first_fit(6), Some(2));
    }

    #[test]
    fn test_first_fit_root_prune() {
        // When root max < needed, returns None in O(1).
        let mut idx = SegmentTreeIndex::new();
        idx.insert_bin(0, 3);
        idx.insert_bin(1, 3);
        idx.insert_bin(2, 3);

        assert_eq!(idx.first_fit(5), None);
    }

    // ── best_fit ────────────────────────────────────────────────────

    #[test]
    fn test_best_fit_tightest() {
        let mut idx = SegmentTreeIndex::new();
        idx.insert_bin(0, 9);
        idx.insert_bin(1, 6);
        idx.insert_bin(2, 6);

        // item=5: smallest remaining ≥ 5 is 6, tie-break → bin 1.
        assert_eq!(idx.best_fit(5), Some(1));
    }

    #[test]
    fn test_best_fit_exact_match() {
        let mut idx = SegmentTreeIndex::new();
        idx.insert_bin(0, 10);
        idx.insert_bin(1, 5);
        idx.insert_bin(2, 8);

        assert_eq!(idx.best_fit(5), Some(1));
    }

    #[test]
    fn test_best_fit_none() {
        let mut idx = SegmentTreeIndex::new();
        idx.insert_bin(0, 3);

        assert_eq!(idx.best_fit(5), None);
    }

    // ── worst_fit ───────────────────────────────────────────────────

    #[test]
    fn test_worst_fit_loosest() {
        let mut idx = SegmentTreeIndex::new();
        idx.insert_bin(0, 3);
        idx.insert_bin(1, 9);
        idx.insert_bin(2, 6);

        assert_eq!(idx.worst_fit(5), Some(1));
    }

    #[test]
    fn test_worst_fit_none() {
        let mut idx = SegmentTreeIndex::new();
        idx.insert_bin(0, 3);
        idx.insert_bin(1, 4);

        assert_eq!(idx.worst_fit(5), None);
    }

    // ── Cross-method consistency ────────────────────────────────────

    #[test]
    fn test_all_methods_agree_single_candidate() {
        let mut idx = SegmentTreeIndex::new();
        idx.insert_bin(0, 2);
        idx.insert_bin(1, 5);

        assert_eq!(idx.first_fit(5), Some(1));
        assert_eq!(idx.best_fit(5), Some(1));
        assert_eq!(idx.worst_fit(5), Some(1));
    }

    #[test]
    fn test_all_methods_none_when_nothing_fits() {
        let mut idx = SegmentTreeIndex::new();
        idx.insert_bin(0, 3);

        assert_eq!(idx.first_fit(5), None);
        assert_eq!(idx.best_fit(5), None);
        assert_eq!(idx.worst_fit(5), None);
    }

    // ── Resize / grow ───────────────────────────────────────────────

    #[test]
    fn test_grows_beyond_initial_capacity() {
        let mut idx = SegmentTreeIndex::with_capacity(2);
        // Insert more bins than initial capacity.
        for i in 0..20 {
            idx.insert_bin(i, (i + 1) * 10);
        }
        // Bin 19 has remaining=200 (largest).
        assert_eq!(idx.first_fit(200), Some(19));
        // Bin 0 has remaining=10.
        assert_eq!(idx.first_fit(5), Some(0));
    }

    // ── Engine simulation ───────────────────────────────────────────

    #[test]
    fn test_insert_place_update_cycle() {
        let mut idx = SegmentTreeIndex::new();
        idx.insert_bin(0, 10);

        assert_eq!(idx.best_fit(7), Some(0));
        idx.update_bin(0, 10, 3);

        assert_eq!(idx.best_fit(5), None);

        idx.insert_bin(1, 10);
        assert_eq!(idx.best_fit(5), Some(1));

        idx.update_bin(1, 10, 5);

        // Bin 0 has 3, bin 1 has 5. Best fit for 3 is bin 0 (tighter).
        assert_eq!(idx.best_fit(3), Some(0));
    }

    // ── Test oracle: compare against LinearScanIndex ────────────────

    #[test]
    fn test_matches_linear_scan_first_fit() {
        let mut linear = LinearScanIndex::new();
        let mut segtree = SegmentTreeIndex::new();

        // Insert bins with varying capacities.
        let capacities = [10, 3, 7, 15, 1, 8, 12, 5, 9, 6];
        for (id, &cap) in capacities.iter().enumerate() {
            linear.insert_bin(id, cap);
            segtree.insert_bin(id, cap);
        }

        // Every first_fit query must agree.
        for needed in 1..=20 {
            assert_eq!(
                linear.first_fit(needed),
                segtree.first_fit(needed),
                "first_fit({needed}) disagrees"
            );
        }

        // Simulate some updates and re-check.
        linear.update_bin(0, 10, 2);
        segtree.update_bin(0, 10, 2);
        linear.update_bin(3, 15, 4);
        segtree.update_bin(3, 15, 4);

        for needed in 1..=20 {
            assert_eq!(
                linear.first_fit(needed),
                segtree.first_fit(needed),
                "first_fit({needed}) disagrees after updates"
            );
        }
    }

    #[test]
    fn test_matches_linear_scan_best_fit() {
        let mut linear = LinearScanIndex::new();
        let mut segtree = SegmentTreeIndex::new();

        let capacities = [10, 3, 7, 15, 1, 8, 12, 5, 9, 6];
        for (id, &cap) in capacities.iter().enumerate() {
            linear.insert_bin(id, cap);
            segtree.insert_bin(id, cap);
        }

        for needed in 1..=20 {
            assert_eq!(
                linear.best_fit(needed),
                segtree.best_fit(needed),
                "best_fit({needed}) disagrees"
            );
        }
    }

    #[test]
    fn test_matches_linear_scan_worst_fit() {
        let mut linear = LinearScanIndex::new();
        let mut segtree = SegmentTreeIndex::new();

        let capacities = [10, 3, 7, 15, 1, 8, 12, 5, 9, 6];
        for (id, &cap) in capacities.iter().enumerate() {
            linear.insert_bin(id, cap);
            segtree.insert_bin(id, cap);
        }

        for needed in 1..=20 {
            assert_eq!(
                linear.worst_fit(needed),
                segtree.worst_fit(needed),
                "worst_fit({needed}) disagrees"
            );
        }
    }

    // ── Default trait ───────────────────────────────────────────────

    #[test]
    fn test_default() {
        let idx = SegmentTreeIndex::default();
        assert_eq!(idx.first_fit(1), None);
    }
}
