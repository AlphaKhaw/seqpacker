//! Solution validation — checks all universal packing invariants.
//!
//! Run `validate_solution()` in tests, debug builds, and CI to catch
//! invariant violations early. This function verifies that an algorithm's
//! output is actually a valid bin-packing solution.

use crate::error::ValidationError;
use crate::pack::Bin;
use crate::sequence::Item;

/// Validate that a packing solution satisfies all universal invariants.
///
/// # Invariants Checked
///
/// 1. **Capacity:** No bin's total item lengths exceed its capacity.
/// 2. **Coverage:** Every input item appears in exactly one bin.
/// 3. **Uniqueness:** No item appears in more than one bin.
/// 4. **Accounting:** Each bin's `used` field matches the sum of its items' lengths.
///
/// # Errors
///
/// Returns the first invariant violation found, if any.
pub fn validate_solution(
    items: &[Item],
    bins: &[Bin],
    capacity: usize,
) -> std::result::Result<(), ValidationError> {
    let mut seen = vec![false; items.len()];

    for bin in bins {
        // Invariant 1: capacity not exceeded.
        let total: usize = bin.items.iter().map(|&id| items[id].len).sum();
        if total > capacity {
            return Err(ValidationError::CapacityExceeded {
                bin_id: bin.id,
                total,
                capacity,
            });
        }

        // Invariant 4: accounting matches.
        if bin.used != total {
            return Err(ValidationError::AccountingMismatch { bin_id: bin.id });
        }

        // Invariants 2 & 3: each item appears exactly once.
        for &id in &bin.items {
            if seen[id] {
                return Err(ValidationError::DuplicateItem { id });
            }
            seen[id] = true;
        }
    }

    // Invariant 2: no items missing.
    if seen.iter().any(|&s| !s) {
        return Err(ValidationError::MissingItems);
    }

    Ok(())
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn items(lens: &[usize]) -> Vec<Item> {
        lens.iter()
            .enumerate()
            .map(|(id, &len)| Item { id, len })
            .collect()
    }

    #[test]
    fn test_valid_solution() {
        let itms = items(&[6, 4, 6, 4]);
        let bins = vec![
            Bin {
                id: 0,
                capacity: 10,
                used: 10,
                items: vec![0, 1].into(),
            },
            Bin {
                id: 1,
                capacity: 10,
                used: 10,
                items: vec![2, 3].into(),
            },
        ];
        assert!(validate_solution(&itms, &bins, 10).is_ok());
    }

    #[test]
    fn test_capacity_exceeded() {
        let itms = items(&[6, 6]);
        let bins = vec![Bin {
            id: 0,
            capacity: 10,
            used: 12,
            items: vec![0, 1].into(),
        }];
        assert!(matches!(
            validate_solution(&itms, &bins, 10),
            Err(ValidationError::CapacityExceeded { .. })
        ));
    }

    #[test]
    fn test_missing_items() {
        let itms = items(&[5, 5, 5]);
        let bins = vec![Bin {
            id: 0,
            capacity: 10,
            used: 10,
            items: vec![0, 1].into(),
        }];
        assert!(matches!(
            validate_solution(&itms, &bins, 10),
            Err(ValidationError::MissingItems)
        ));
    }

    #[test]
    fn test_duplicate_item() {
        let itms = items(&[5, 5]);
        let bins = vec![
            Bin {
                id: 0,
                capacity: 10,
                used: 10,
                items: vec![0, 1].into(),
            },
            Bin {
                id: 1,
                capacity: 10,
                used: 5,
                items: vec![0].into(),
            },
        ];
        assert!(matches!(
            validate_solution(&itms, &bins, 10),
            Err(ValidationError::DuplicateItem { id: 0 })
        ));
    }
}
