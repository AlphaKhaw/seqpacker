//! Input sequence types for packing.

/// Lightweight item for the packing engine.
///
/// This is what algorithms operate on internally. Contains only
/// the information needed for bin-packing decisions: an ID to track
/// which input sequence this came from, and a length.
///
/// `Item` is `Copy` because it is just two `usize` fields (16 bytes on 64-bit).
/// This means the engine can freely copy items without allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Item {
    /// Index into the original input sequence array.
    pub id: usize,
    /// Length of sequence in tokens.
    pub len: usize,
}

/// A sequence with optional token data, used in the public API.
///
/// Users create `Sequence` values; the packer internally converts them
/// to `Item` for the engine, then maps back to `Sequence` in the output.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Sequence {
    /// Unique identifier for tracking.
    pub id: usize,
    /// Length of the sequence in tokens.
    pub length: usize,
    /// Optional token data (None for length-only packing).
    pub tokens: Option<Vec<u32>>,
}

impl Sequence {
    /// Create a length-only sequence (no token data).
    ///
    /// This is the fast path — no allocation for token storage.
    pub fn new(id: usize, length: usize) -> Self {
        Self {
            id,
            length,
            tokens: None,
        }
    }

    /// Create a sequence from token data.
    ///
    /// The length is inferred from the token vector's length.
    pub fn with_tokens(id: usize, tokens: Vec<u32>) -> Self {
        let length = tokens.len();
        Self {
            id,
            length,
            tokens: Some(tokens),
        }
    }

    /// Get the length of this sequence.
    pub fn len(&self) -> usize {
        self.length
    }

    /// Check if sequence is empty.
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Convert to a lightweight Item for the engine.
    pub fn to_item(&self) -> Item {
        Item {
            id: self.id,
            len: self.length,
        }
    }
}

/// Ordering by length — used by FFD/BFD and variants for sorting.
impl Ord for Sequence {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.length.cmp(&other.length)
    }
}

impl PartialOrd for Sequence {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl From<&Sequence> for Item {
    fn from(seq: &Sequence) -> Self {
        Item {
            id: seq.id,
            len: seq.length,
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Item tests ───────────────────────────────────────────────────

    #[test]
    fn test_item_is_copy() {
        let a = Item { id: 0, len: 512 };
        let b = a; // copy, not move
        assert_eq!(a.id, b.id); // a is still valid
    }

    // ── Sequence construction ────────────────────────────────────────

    #[test]
    fn test_sequence_new() {
        let seq = Sequence::new(0, 512);
        assert_eq!(seq.id, 0);
        assert_eq!(seq.len(), 512);
        assert!(seq.tokens.is_none());
    }

    #[test]
    fn test_sequence_with_tokens() {
        let seq = Sequence::with_tokens(1, vec![10, 20, 30]);
        assert_eq!(seq.len(), 3);
        assert_eq!(seq.tokens.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_sequence_is_empty() {
        assert!(Sequence::new(0, 0).is_empty());
        assert!(!Sequence::new(0, 1).is_empty());
    }

    // ── Conversion ───────────────────────────────────────────────────

    #[test]
    fn test_to_item() {
        let seq = Sequence::new(5, 42);
        let item = seq.to_item();
        assert_eq!(item.id, 5);
        assert_eq!(item.len, 42);
    }

    #[test]
    fn test_item_from_sequence_ref() {
        let seq = Sequence::new(5, 42);
        let item = Item::from(&seq);
        assert_eq!(item.id, 5);
        assert_eq!(item.len, 42);
    }

    // ── Ordering ─────────────────────────────────────────────────────

    #[test]
    fn test_sequence_ordering() {
        let a = Sequence::new(0, 100);
        let b = Sequence::new(1, 200);
        assert!(a < b);
    }

    #[test]
    fn test_sequence_sort_descending() {
        let mut seqs = vec![
            Sequence::new(0, 30),
            Sequence::new(1, 10),
            Sequence::new(2, 20),
        ];
        seqs.sort_by(|a, b| b.cmp(a)); // descending
        assert_eq!(seqs[0].length, 30);
        assert_eq!(seqs[1].length, 20);
        assert_eq!(seqs[2].length, 10);
    }
}
