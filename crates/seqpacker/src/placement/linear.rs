//! Linear scan PlacementIndex — O(B) per query, used for testing and debugging.

use super::PlacementIndex;

/// Simple linear scan over remaining capacities.
///
/// Stores `(bin_id, remaining)` pairs in a Vec and scans linearly
/// for every query. O(B) per query but trivially correct.
///
/// Used for:
/// - Testing other PlacementIndex implementations against
/// - Small bin counts where O(B) is acceptable
/// - Debugging placement decisions
#[derive(Clone, Debug, Default)]
pub struct LinearScanIndex {
    /// (bin_id, remaining_capacity) pairs.
    bins: Vec<(usize, usize)>,
}

impl LinearScanIndex {
    /// Create an empty linear scan placement index.
    pub fn new() -> Self {
        Self { bins: Vec::new() }
    }
}

impl PlacementIndex for LinearScanIndex {
    fn insert_bin(&mut self, bin_id: usize, remaining: usize) {
        self.bins.push((bin_id, remaining));
    }

    fn update_bin(&mut self, bin_id: usize, _old_remaining: usize, new_remaining: usize) {
        // Linear search for the bin and update its remaining.
        for entry in &mut self.bins {
            if entry.0 == bin_id {
                entry.1 = new_remaining;
                return;
            }
        }
    }

    fn first_fit(&self, needed: usize) -> Option<usize> {
        // Leftmost (lowest bin_id) with remaining >= needed.
        self.bins
            .iter()
            .filter(|(_, rem)| *rem >= needed)
            .min_by_key(|(id, _)| *id)
            .map(|(id, _)| *id)
    }

    fn best_fit(&self, needed: usize) -> Option<usize> {
        // Smallest remaining >= needed (tightest fit).
        // Tie-break by lowest bin_id for determinism.
        self.bins
            .iter()
            .filter(|(_, rem)| *rem >= needed)
            .min_by_key(|(id, rem)| (*rem, *id))
            .map(|(id, _)| *id)
    }

    fn worst_fit(&self, needed: usize) -> Option<usize> {
        // Largest remaining >= needed (loosest fit).
        // Tie-break by lowest bin_id for determinism.
        self.bins
            .iter()
            .filter(|(_, rem)| *rem >= needed)
            .max_by_key(|(_, rem)| *rem)
            .map(|(id, _)| *id)
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── insert_bin ────────────────────────────────────────────────────

    #[test]
    fn test_insert_bin() {
        let mut idx = LinearScanIndex::new();
        idx.insert_bin(0, 100);
        idx.insert_bin(1, 50);
        // Both bins should be queryable.
        assert_eq!(idx.first_fit(50), Some(0));
        assert_eq!(idx.first_fit(100), Some(0));
    }

    // ── update_bin ────────────────────────────────────────────────────

    #[test]
    fn test_update_bin() {
        let mut idx = LinearScanIndex::new();
        idx.insert_bin(0, 10);
        idx.insert_bin(1, 10);

        idx.update_bin(0, 10, 3);

        // Bin 0 now has 3 remaining, can't fit 5.
        assert_eq!(idx.first_fit(5), Some(1));
        // But can still fit 3.
        assert_eq!(idx.first_fit(3), Some(0));
    }

    #[test]
    fn test_update_bin_to_zero() {
        let mut idx = LinearScanIndex::new();
        idx.insert_bin(0, 10);

        idx.update_bin(0, 10, 0);

        // Nothing fits in a bin with 0 remaining.
        assert_eq!(idx.first_fit(1), None);
    }

    // ── first_fit ─────────────────────────────────────────────────────

    #[test]
    fn test_first_fit_leftmost() {
        let mut idx = LinearScanIndex::new();
        idx.insert_bin(0, 2);
        idx.insert_bin(1, 7);
        idx.insert_bin(2, 7);

        // Both bin 1 and 2 fit, first_fit returns leftmost (1).
        assert_eq!(idx.first_fit(6), Some(1));
    }

    #[test]
    fn test_first_fit_none() {
        let mut idx = LinearScanIndex::new();
        idx.insert_bin(0, 5);
        idx.insert_bin(1, 3);

        assert_eq!(idx.first_fit(6), None);
    }

    #[test]
    fn test_first_fit_exact() {
        let mut idx = LinearScanIndex::new();
        idx.insert_bin(0, 5);

        assert_eq!(idx.first_fit(5), Some(0));
    }

    #[test]
    fn test_first_fit_empty_index() {
        let idx = LinearScanIndex::new();
        assert_eq!(idx.first_fit(1), None);
    }

    // ── best_fit ──────────────────────────────────────────────────────

    #[test]
    fn test_best_fit_tightest() {
        let mut idx = LinearScanIndex::new();
        idx.insert_bin(0, 9);
        idx.insert_bin(1, 6);
        idx.insert_bin(2, 6);

        // item=5: all fit, smallest remaining ≥ 5 is 6, tie-break → bin 1.
        assert_eq!(idx.best_fit(5), Some(1));
    }

    #[test]
    fn test_best_fit_exact_match() {
        let mut idx = LinearScanIndex::new();
        idx.insert_bin(0, 10);
        idx.insert_bin(1, 5);
        idx.insert_bin(2, 8);

        // item=5: exact match at bin 1 (remaining=5), tightest possible.
        assert_eq!(idx.best_fit(5), Some(1));
    }

    #[test]
    fn test_best_fit_none() {
        let mut idx = LinearScanIndex::new();
        idx.insert_bin(0, 3);

        assert_eq!(idx.best_fit(5), None);
    }

    // ── worst_fit ─────────────────────────────────────────────────────

    #[test]
    fn test_worst_fit_loosest() {
        let mut idx = LinearScanIndex::new();
        idx.insert_bin(0, 3);
        idx.insert_bin(1, 9);
        idx.insert_bin(2, 6);

        // item=5: bins 1(9) and 2(6) fit, largest remaining = 9 → bin 1.
        assert_eq!(idx.worst_fit(5), Some(1));
    }

    #[test]
    fn test_worst_fit_none() {
        let mut idx = LinearScanIndex::new();
        idx.insert_bin(0, 3);
        idx.insert_bin(1, 4);

        assert_eq!(idx.worst_fit(5), None);
    }

    // ── Cross-method consistency ──────────────────────────────────────

    #[test]
    fn test_all_methods_agree_single_candidate() {
        let mut idx = LinearScanIndex::new();
        idx.insert_bin(0, 2);
        idx.insert_bin(1, 5);

        // Only bin 1 fits item=5. All methods must return it.
        assert_eq!(idx.first_fit(5), Some(1));
        assert_eq!(idx.best_fit(5), Some(1));
        assert_eq!(idx.worst_fit(5), Some(1));
    }

    #[test]
    fn test_all_methods_none_when_nothing_fits() {
        let mut idx = LinearScanIndex::new();
        idx.insert_bin(0, 3);

        assert_eq!(idx.first_fit(5), None);
        assert_eq!(idx.best_fit(5), None);
        assert_eq!(idx.worst_fit(5), None);
    }

    // ── Sequence of operations (simulates engine usage) ───────────────

    #[test]
    fn test_insert_place_update_cycle() {
        let mut idx = LinearScanIndex::new();
        idx.insert_bin(0, 10);

        // Place item of size 7 → remaining drops to 3.
        assert_eq!(idx.best_fit(7), Some(0));
        idx.update_bin(0, 10, 3);

        // Item of size 5 no longer fits bin 0.
        assert_eq!(idx.best_fit(5), None);

        // Create new bin.
        idx.insert_bin(1, 10);
        assert_eq!(idx.best_fit(5), Some(1));

        // Place size 5 → remaining drops to 5.
        idx.update_bin(1, 10, 5);

        // Now bin 0 has 3, bin 1 has 5. Best fit for item=3 is bin 0 (tighter).
        assert_eq!(idx.best_fit(3), Some(0));
    }
}
