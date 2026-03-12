//! PlacementIndex trait and implementations for bin selection.
//!
//! The PlacementIndex is the key abstraction that makes the generic
//! `greedy_pack()` engine possible. Each implementation provides
//! different performance characteristics for different query types.

// Submodules — uncomment as each is implemented (Steps 7-9):
pub mod btree;
pub mod capacity_segment_tree;
pub mod linear;
pub mod segment_tree;

pub use btree::BTreeRemainingIndex;
pub use capacity_segment_tree::CapacitySegmentTree;
pub use linear::LinearScanIndex;
pub use segment_tree::SegmentTreeIndex;

/// Trait for data structures that track bin remaining capacities
/// and support efficient bin selection queries.
///
/// The three query methods correspond to the three classical
/// bin-packing placement strategies. Not all implementations
/// need to be efficient at all three — pick the right index
/// for your algorithm.
pub trait PlacementIndex {
    /// Register a newly created bin with its initial remaining capacity.
    fn insert_bin(&mut self, bin_id: usize, remaining: usize);

    /// Update a bin's remaining capacity after an item is placed in it.
    ///
    /// `old_remaining` is provided for implementations that index by
    /// remaining capacity (e.g., BTreeMap needs to remove from the old
    /// key before inserting at the new key).
    fn update_bin(&mut self, bin_id: usize, old_remaining: usize, new_remaining: usize);

    /// First Fit: find the **leftmost** bin with remaining ≥ needed.
    ///
    /// "Leftmost" means the bin with the smallest `bin_id` that fits.
    /// This is the natural query for segment trees.
    fn first_fit(&self, needed: usize) -> Option<usize>;

    /// Best Fit: find the bin with the **smallest** remaining ≥ needed.
    ///
    /// Ties are broken arbitrarily (any bin with minimum feasible
    /// remaining is acceptable). This minimizes wasted space per placement.
    fn best_fit(&self, needed: usize) -> Option<usize>;

    /// Worst Fit: find the bin with the **largest** remaining ≥ needed.
    ///
    /// Ties are broken arbitrarily. This spreads items evenly across bins.
    fn worst_fit(&self, needed: usize) -> Option<usize>;
}
