//! Counting sort for integer sequence lengths.
//!
//! Groups sequence indices by their length in O(N + L) time,
//! where L is the maximum length. Used by OBFD to process
//! sequences from largest to smallest without comparison sort.

/// Group item indices by their length.
///
/// Returns a Vec of buckets where `buckets[len]` contains the
/// original indices of all items with that length.
///
/// # Arguments
///
/// - `lengths`: Sequence lengths (values must be in 0..=max_length).
/// - `max_length`: Upper bound on length values.
///
/// # Example
///
/// ```
/// use seqpacker::algorithms::counting_sort::counting_sort;
///
/// let lengths = &[5, 3, 5, 1, 3];
/// let buckets = counting_sort(lengths, 5);
///
/// assert_eq!(buckets[1], vec![3]);     // index 3 has length 1
/// assert_eq!(buckets[3], vec![1, 4]);  // indices 1,4 have length 3
/// assert_eq!(buckets[5], vec![0, 2]);  // indices 0,2 have length 5
/// ```
pub fn counting_sort(lengths: &[usize], max_length: usize) -> Vec<Vec<usize>> {
    let mut buckets = vec![Vec::new(); max_length + 1];
    for (i, &length) in lengths.iter().enumerate() {
        buckets[length].push(i);
    }
    buckets
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_grouping() {
        let lengths = &[5, 3, 5, 1, 3];
        let buckets = counting_sort(lengths, 5);

        assert!(buckets[0].is_empty());
        assert_eq!(buckets[1], vec![3]);
        assert!(buckets[2].is_empty());
        assert_eq!(buckets[3], vec![1, 4]);
        assert!(buckets[4].is_empty());
        assert_eq!(buckets[5], vec![0, 2]);
    }

    #[test]
    fn test_all_same_length() {
        let lengths = &[7, 7, 7];
        let buckets = counting_sort(lengths, 7);

        assert_eq!(buckets[7], vec![0, 1, 2]);
        for i in 0..7 {
            assert!(buckets[i].is_empty());
        }
    }

    #[test]
    fn test_empty_input() {
        let lengths: &[usize] = &[];
        let buckets = counting_sort(lengths, 10);
        assert!(buckets.iter().all(|b| b.is_empty()));
    }

    #[test]
    fn test_single_item() {
        let buckets = counting_sort(&[42], 42);
        assert_eq!(buckets[42], vec![0]);
    }

    #[test]
    fn test_preserves_insertion_order() {
        // Items with same length should appear in original order.
        let lengths = &[5, 5, 5, 5];
        let buckets = counting_sort(lengths, 5);
        assert_eq!(buckets[5], vec![0, 1, 2, 3]);
    }
}
