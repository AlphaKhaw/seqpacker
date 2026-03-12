//! Capacity-indexed segment tree for OBFD.
//!
//! Unlike the bin-indexed `SegmentTreeIndex` (Step 8), this tree is indexed
//! by remaining capacity values (0..=max_capacity). Each leaf stores the
//! capacity value itself if at least one bin has that remaining capacity,
//! or 0 if no bins have that capacity.
//!
//! Internal nodes store the max of their children, enabling O(log L) queries
//! for "smallest capacity ≥ target" (best-fit) via left-first traversal.
//!
//! The tree uses 0-indexed layout to match LightBinPack's C++ implementation:
//! - Root is at index 0
//! - Node i's children are at 2i+1 (left) and 2i+2 (right)
//! - Leaves start at index n-1 (where n = next power of 2 ≥ max_capacity+1)
//! - Leaf for capacity value C is at index n-1+C

/// Iterative segment tree over capacity values for OBFD best-fit queries.
///
/// Fixed-size tree: O(L) space where L = max capacity. Does not grow.
///
/// # Operations
///
/// - `update(capacity, value)` — O(log L), set leaf and propagate with early termination
/// - `find_best_fit(target)` — O(log L), find smallest capacity ≥ target that has bins
///
/// # Example
///
/// ```
/// use seqpacker::placement::capacity_segment_tree::CapacitySegmentTree;
///
/// let mut tree = CapacitySegmentTree::new(10);
/// // Tree starts with one bin at capacity 10.
/// assert_eq!(tree.find_best_fit(5), Some(10));
///
/// // Simulate placing an item of size 3: bin now has remaining 7.
/// tree.update(10, 0);  // remove from capacity 10
/// tree.update(7, 7);   // add at capacity 7
/// assert_eq!(tree.find_best_fit(5), Some(7));
///
/// // No bin can fit an item of size 8.
/// assert_eq!(tree.find_best_fit(8), None);
/// ```
#[derive(Debug, Clone)]
pub struct CapacitySegmentTree {
    /// Number of leaf slots (next power of 2 >= max_capacity + 1).
    n: usize,
    /// Flat array: 0-indexed. Size = 2n. Internal nodes store max of children.
    /// Leaf at n-1+c stores c if bins exist at capacity c, else 0.
    tree: Vec<usize>,
}

impl CapacitySegmentTree {
    /// Create a new tree for capacities in 0..=max_capacity.
    ///
    /// Initializes with a single "bin" at max_capacity (the first empty bin).
    pub fn new(max_capacity: usize) -> Self {
        let mut n = 1;
        while n < max_capacity + 1 {
            n <<= 1;
        }
        let mut tree = vec![0usize; 2 * n];

        // Seed: one empty bin exists at max_capacity.
        tree[n - 1 + max_capacity] = max_capacity;

        // Build internal nodes bottom-up.
        for i in (0..n - 1).rev() {
            tree[i] = tree[2 * i + 1].max(tree[2 * i + 2]);
        }

        Self { n, tree }
    }

    /// Create an empty tree (no seeded bin). Used when the caller manages initialization.
    pub fn new_empty(max_capacity: usize) -> Self {
        let mut n = 1;
        while n < max_capacity + 1 {
            n <<= 1;
        }
        Self {
            n,
            tree: vec![0usize; 2 * n],
        }
    }

    /// Update the leaf for `capacity` to `value`.
    ///
    /// Set `value = capacity` to mark that bins exist at this capacity.
    /// Set `value = 0` to mark that no bins exist at this capacity.
    ///
    /// Propagates upward with **early termination**: stops as soon as
    /// a parent's value doesn't change.
    #[inline(always)]
    pub fn update(&mut self, capacity: usize, value: usize) {
        let mut idx = self.n - 1 + capacity;
        self.tree[idx] = value;
        while idx > 0 {
            idx = (idx - 1) / 2;
            let new_val = self.tree[2 * idx + 1].max(self.tree[2 * idx + 2]);
            if self.tree[idx] == new_val {
                break; // Early termination
            }
            self.tree[idx] = new_val;
        }
    }

    /// Find the smallest capacity ≥ target that has bins available.
    ///
    /// Returns `None` if no capacity ≥ target has any bins.
    ///
    /// Traverses left-first (left child = smaller capacities), yielding
    /// the tightest fit (best-fit semantics).
    #[inline(always)]
    pub fn find_best_fit(&self, target: usize) -> Option<usize> {
        if self.tree[0] < target {
            return None;
        }

        let mut idx = 0;
        while idx < self.n - 1 {
            let left = 2 * idx + 1;
            let right = 2 * idx + 2;
            if self.tree[left] >= target {
                idx = left; // Prefer left (smaller capacity = tighter fit)
            } else {
                idx = right;
            }
        }

        Some(idx - (self.n - 1))
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_seeds_at_max_capacity() {
        let tree = CapacitySegmentTree::new(10);
        // Root should hold max_capacity since one bin is seeded there.
        assert_eq!(tree.find_best_fit(1), Some(10));
        assert_eq!(tree.find_best_fit(10), Some(10));
        assert_eq!(tree.find_best_fit(11), None);
    }

    #[test]
    fn test_new_empty_has_no_bins() {
        let tree = CapacitySegmentTree::new_empty(10);
        // No bins seeded — any target > 0 returns None.
        assert_eq!(tree.find_best_fit(1), None);
        assert_eq!(tree.find_best_fit(10), None);
    }

    #[test]
    fn test_update_and_query() {
        let mut tree = CapacitySegmentTree::new_empty(10);

        // Add a bin at capacity 7.
        tree.update(7, 7);
        assert_eq!(tree.find_best_fit(5), Some(7));
        assert_eq!(tree.find_best_fit(7), Some(7));
        assert_eq!(tree.find_best_fit(8), None);

        // Add a bin at capacity 3.
        tree.update(3, 3);
        assert_eq!(tree.find_best_fit(2), Some(3)); // best fit = tightest
        assert_eq!(tree.find_best_fit(5), Some(7));
    }

    #[test]
    fn test_best_fit_returns_smallest_qualifying() {
        let mut tree = CapacitySegmentTree::new_empty(10);

        // Bins at capacities 3, 5, 8.
        tree.update(3, 3);
        tree.update(5, 5);
        tree.update(8, 8);

        // target=4 → smallest capacity ≥ 4 is 5 (not 8).
        assert_eq!(tree.find_best_fit(4), Some(5));
        // target=1 → smallest capacity ≥ 1 is 3 (not 5 or 8).
        assert_eq!(tree.find_best_fit(1), Some(3));
        // target=6 → smallest capacity ≥ 6 is 8.
        assert_eq!(tree.find_best_fit(6), Some(8));
    }

    #[test]
    fn test_remove_capacity() {
        let mut tree = CapacitySegmentTree::new(10);

        // Remove the seeded bin.
        tree.update(10, 0);
        assert_eq!(tree.find_best_fit(1), None);

        // Add at capacity 5, then remove it.
        tree.update(5, 5);
        assert_eq!(tree.find_best_fit(3), Some(5));
        tree.update(5, 0);
        assert_eq!(tree.find_best_fit(3), None);
    }

    #[test]
    fn test_move_bin_between_capacities() {
        // Simulates placing an item: bin moves from old_cap to new_cap.
        let mut tree = CapacitySegmentTree::new(10);

        // Place item of size 3: bin moves 10 → 7.
        tree.update(10, 0);
        tree.update(7, 7);
        assert_eq!(tree.find_best_fit(8), None);
        assert_eq!(tree.find_best_fit(7), Some(7));

        // Place item of size 4: bin moves 7 → 3.
        tree.update(7, 0);
        tree.update(3, 3);
        assert_eq!(tree.find_best_fit(4), None);
        assert_eq!(tree.find_best_fit(3), Some(3));
    }

    #[test]
    fn test_early_termination_correctness() {
        // Verifies that early termination doesn't break propagation.
        let mut tree = CapacitySegmentTree::new_empty(8);

        tree.update(5, 5);
        tree.update(3, 3);
        assert_eq!(tree.find_best_fit(4), Some(5));

        // Remove 5. Root should now reflect 3 as max.
        tree.update(5, 0);
        assert_eq!(tree.find_best_fit(4), None);
        assert_eq!(tree.find_best_fit(3), Some(3));
    }

    #[test]
    fn test_capacity_one() {
        // Edge case: smallest meaningful capacity.
        let mut tree = CapacitySegmentTree::new(1);
        assert_eq!(tree.find_best_fit(1), Some(1));
        assert_eq!(tree.find_best_fit(2), None);

        tree.update(1, 0);
        assert_eq!(tree.find_best_fit(1), None);
    }

    #[test]
    fn test_power_of_two_capacity() {
        // max_capacity+1 is already a power of 2 — no extra padding needed.
        let tree = CapacitySegmentTree::new(7);
        // n should be 8 (next power of 2 ≥ 8).
        assert_eq!(tree.find_best_fit(7), Some(7));
        assert_eq!(tree.find_best_fit(8), None);
    }

    #[test]
    fn test_multiple_updates_same_leaf() {
        let mut tree = CapacitySegmentTree::new_empty(10);

        // Toggle capacity 5 on and off repeatedly.
        tree.update(5, 5);
        assert_eq!(tree.find_best_fit(5), Some(5));
        tree.update(5, 0);
        assert_eq!(tree.find_best_fit(5), None);
        tree.update(5, 5);
        assert_eq!(tree.find_best_fit(5), Some(5));
    }
}
