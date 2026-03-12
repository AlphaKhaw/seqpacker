//! Error types for the seqpacker library.

use thiserror::Error;

/// Errors that can occur during packing operations.
///
/// User-facing errors returned by algorithm and packer functions.
#[derive(Error, Debug)]
pub enum PackError {
    /// A sequence is longer than the bin capacity — can never be packed.
    #[error("sequence length {length} exceeds capacity {capacity}")]
    SequenceTooLong {
        /// The sequence length that exceeded capacity.
        length: usize,
        /// The bin capacity.
        capacity: usize,
    },

    /// Attempted to add a sequence to a bin that doesn't have enough remaining space.
    #[error("pack is full, cannot add sequence of length {length}")]
    PackFull {
        /// The sequence length that could not fit.
        length: usize,
    },

    /// The packer was configured with invalid parameters.
    #[error("invalid configuration: {message}")]
    InvalidConfig {
        /// Description of the invalid configuration.
        message: String,
    },

    /// An algorithm-specific error occurred.
    #[error("algorithm error: {message}")]
    AlgorithmError {
        /// Description of the algorithm error.
        message: String,
    },

    /// No sequences were provided to pack.
    #[error("empty input: no sequences to pack")]
    EmptyInput,

    /// A post-packing validation check failed (only in debug/test builds).
    #[error("validation failed: {0}")]
    Validation(#[from] ValidationError),
}

/// Errors from `validate_solution()` — internal correctness checks.
///
/// These indicate bugs in the algorithm implementation, not user errors.
/// They should never occur in production; they're checked in tests and
/// debug builds to catch invariant violations early.
#[derive(Error, Debug)]
pub enum ValidationError {
    /// A bin's total item lengths exceed its capacity.
    #[error("bin {bin_id}: total {total} exceeds capacity {capacity}")]
    CapacityExceeded {
        /// The bin that exceeded capacity.
        bin_id: usize,
        /// The actual total of item lengths in the bin.
        total: usize,
        /// The bin capacity.
        capacity: usize,
    },

    /// A bin's `used` field doesn't match the sum of its item lengths.
    #[error("bin {bin_id}: accounting mismatch between used and actual sum")]
    AccountingMismatch {
        /// The bin with mismatched accounting.
        bin_id: usize,
    },

    /// An item appears in more than one bin.
    #[error("item {id} appears in multiple bins")]
    DuplicateItem {
        /// The duplicated item ID.
        id: usize,
    },

    /// Some items from the input are missing from the output bins.
    #[error("not all items appear in the output bins")]
    MissingItems,
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, PackError>;

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── PackError display messages ───────────────────────────────────

    #[test]
    fn test_sequence_too_long_message() {
        let err = PackError::SequenceTooLong {
            length: 500,
            capacity: 128,
        };
        assert_eq!(err.to_string(), "sequence length 500 exceeds capacity 128");
    }

    #[test]
    fn test_pack_full_message() {
        let err = PackError::PackFull { length: 42 };
        assert_eq!(
            err.to_string(),
            "pack is full, cannot add sequence of length 42"
        );
    }

    #[test]
    fn test_invalid_config_message() {
        let err = PackError::InvalidConfig {
            message: "capacity must be > 0".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "invalid configuration: capacity must be > 0"
        );
    }

    #[test]
    fn test_algorithm_error_message() {
        let err = PackError::AlgorithmError {
            message: "segment tree overflow".to_string(),
        };
        assert_eq!(err.to_string(), "algorithm error: segment tree overflow");
    }

    #[test]
    fn test_empty_input_message() {
        let err = PackError::EmptyInput;
        assert_eq!(err.to_string(), "empty input: no sequences to pack");
    }

    // ── ValidationError display messages ─────────────────────────────

    #[test]
    fn test_capacity_exceeded_message() {
        let err = ValidationError::CapacityExceeded {
            bin_id: 3,
            total: 150,
            capacity: 128,
        };
        assert_eq!(err.to_string(), "bin 3: total 150 exceeds capacity 128");
    }

    #[test]
    fn test_accounting_mismatch_message() {
        let err = ValidationError::AccountingMismatch { bin_id: 7 };
        assert_eq!(
            err.to_string(),
            "bin 7: accounting mismatch between used and actual sum"
        );
    }

    #[test]
    fn test_duplicate_item_message() {
        let err = ValidationError::DuplicateItem { id: 42 };
        assert_eq!(err.to_string(), "item 42 appears in multiple bins");
    }

    #[test]
    fn test_missing_items_message() {
        let err = ValidationError::MissingItems;
        assert_eq!(err.to_string(), "not all items appear in the output bins");
    }

    // ── From conversion (#[from]) ────────────────────────────────────

    #[test]
    fn test_validation_error_converts_to_pack_error() {
        let ve = ValidationError::MissingItems;
        let pe: PackError = ve.into();
        assert!(matches!(
            pe,
            PackError::Validation(ValidationError::MissingItems)
        ));
    }

    #[test]
    fn test_validation_error_wraps_message() {
        let ve = ValidationError::CapacityExceeded {
            bin_id: 0,
            total: 200,
            capacity: 100,
        };
        let pe: PackError = ve.into();
        assert_eq!(
            pe.to_string(),
            "validation failed: bin 0: total 200 exceeds capacity 100"
        );
    }
}
