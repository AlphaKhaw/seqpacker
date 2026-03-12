//! Bin and Pack types for packing output.

use smallvec::SmallVec;

use crate::error::PackError;
use crate::sequence::Sequence;

/// Lightweight bin for the packing engine.
///
/// Tracks which items (by ID) are placed in this bin and capacity
/// used. The engine only needs this information for placement
/// decisions.
#[derive(Clone, Debug)]
pub struct Bin {
    /// Bin identifier.
    pub id: usize,
    /// Maximum capacity.
    pub capacity: usize,
    /// Used capacity.
    pub used: usize,
    /// Item IDs placed in this bin (indices into the original item array).
    pub items: SmallVec<[usize; 8]>,
}

impl Bin {
    /// Create an empty bin.
    pub fn new(id: usize, capacity: usize) -> Self {
        Self {
            id,
            capacity,
            used: 0,
            items: SmallVec::new(),
        }
    }

    /// Remaining capacity.
    pub fn remaining(&self) -> usize {
        self.capacity - self.used
    }
}

/// A pack containing multiple sequences - the user-facing output type.
///
/// Provides methods for generating LLM training metadata:
/// attention masks, position IDs, cumulative sequence lengths, etc.
#[derive(Clone, Debug)]
pub struct Pack {
    /// Sequences included in this pack.
    pub sequences: Vec<Sequence>,
    /// Maximum capacity of this pack.
    pub capacity: usize,
    /// Current used capacity.
    pub(crate) used: usize,
}

impl Pack {
    /// Create a new empty pack with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            sequences: Vec::new(),
            capacity,
            used: 0,
        }
    }

    /// Add a sequence to this pack.
    ///
    /// # Errors
    ///
    /// Returns `PackError::PackFull` if the sequence does not fit.
    pub fn add(&mut self, seq: Sequence) -> Result<(), PackError> {
        if seq.length > self.remaining_capacity() {
            return Err(PackError::PackFull { length: seq.length });
        }
        self.used += seq.length;
        self.sequences.push(seq);
        Ok(())
    }

    /// Remaining capacity in tokens.
    pub fn remaining_capacity(&self) -> usize {
        self.capacity.saturating_sub(self.used)
    }

    /// Used capacity in tokens.
    pub fn used_capacity(&self) -> usize {
        self.used
    }

    /// Utilisation ratio (0.0 to 1.0).
    pub fn utilisation(&self) -> f64 {
        self.used as f64 / self.capacity as f64
    }

    /// Number of sequences in this pack.
    pub fn len(&self) -> usize {
        self.sequences.len()
    }

    /// Check if pack is empty.
    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }

    /// Padding tokens needed to fill this pack to capacity.
    pub fn padding_tokens(&self) -> usize {
        self.capacity.saturating_sub(self.used)
    }

    // ── LLM Training Metadata ─────────────────────────────────────

    /// Cumulative sequence lengths for Flash Attention's varlen API.
    ///
    /// Returns `[0, len_0, len_0+len_1, ..., total]` — a vector of
    /// length `num_sequences + 1`. This is the primary input format for
    /// `flash_attn_varlen_func`, avoiding N×N mask materialization.
    ///
    /// # Example
    ///
    /// For sequences of lengths [512, 256, 128]:
    /// ```text
    /// cu_seqlens = [0, 512, 768, 896]
    /// ```
    pub fn cu_seqlens(&self) -> Vec<usize> {
        let mut cu = Vec::with_capacity(self.sequences.len() + 1);
        cu.push(0);
        let mut offset = 0;
        for seq in &self.sequences {
            offset += seq.length;
            cu.push(offset);
        }
        cu
    }

    /// Maximum sequence length in this pack.
    ///
    /// Needed alongside `cu_seqlens` for Flash Attention's varlen API
    /// as the `max_seqlen_q` / `max_seqlen_k` argument.
    pub fn max_seqlen_in_pack(&self) -> usize {
        self.sequences.iter().map(|s| s.length).max().unwrap_or(0)
    }

    /// Position IDs with per-sequence reset.
    ///
    /// Each sequence gets position IDs starting from 0. This is
    /// necessary because packed sequences share a single input
    /// tensor but need independent positional encodings.
    ///
    /// For sequences of lengths [3, 2]: `[0, 1, 2, 0, 1]`
    pub fn position_ids(&self) -> Vec<usize> {
        let mut ids = Vec::with_capacity(self.used);
        for seq in &self.sequences {
            ids.extend(0..seq.length);
        }
        ids
    }

    /// Segment IDs for cross-attention prevention.
    ///
    /// Each token gets an integer ID for which sequence it belongs to.
    /// Used by Megatron-LM and DeepSpeed to prevent attention across
    /// sequence boundaries without materialising an NxN mask.
    ///
    /// For sequences of length [3, 2]: `[0, 0, 0, 1, 1]`
    pub fn segment_ids(&self) -> Vec<usize> {
        let mut ids = Vec::with_capacity(self.used);
        for (seq_id, seq) in self.sequences.iter().enumerate() {
            ids.extend(std::iter::repeat_n(seq_id, seq.length));
        }
        ids
    }

    /// Block-diagonal causal attention mask.
    ///
    /// Returns a flattened `NxN` boolean mask (row-major) where `N = used`.
    /// Each sequence gets a lower-triangular (causal) block, and cross-sequence
    /// attention is blocked.
    ///
    /// WARNING: This is O(N²) in memory. For sequences longer than ~4K tokens,
    /// prefer `cu_seqlens()` which Flash Attention accepts directly without
    /// materialising the full mask.
    pub fn attention_mask(&self) -> Vec<bool> {
        let n = self.used;
        let mut mask = vec![false; n * n];

        let mut offset = 0;
        for seq in &self.sequences {
            for i in 0..seq.length {
                for j in 0..=i {
                    mask[(offset + i) * n + (offset + j)] = true;
                }
            }
            offset += seq.length;
        }
        mask
    }
}

/// Convert engine output (Bins) to user output (Packs).
///
/// Looks up each item ID in the original sequences to reconstruct
/// rich Pack objects with full sequence data.
pub fn bins_to_packs(bins: Vec<Bin>, sequences: &[Sequence]) -> Vec<Pack> {
    bins.into_iter()
        .map(|bin| {
            let mut pack_seqs = Vec::with_capacity(bin.items.len());
            let mut used = 0;
            for &item_id in &bin.items {
                let src = &sequences[item_id];
                // Construct directly instead of cloning - avoids Options<Vec> clone overhead.
                let seq = if src.tokens.is_some() {
                    src.clone()
                } else {
                    Sequence::new(src.id, src.length)
                };
                used += seq.length;
                pack_seqs.push(seq);
            }
            Pack {
                sequences: pack_seqs,
                capacity: bin.capacity,
                used,
            }
        })
        .collect()
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence::Sequence;

    // ── Bin tests ────────────────────────────────────────────────────

    #[test]
    fn test_bin_new() {
        let bin = Bin::new(0, 100);
        assert_eq!(bin.id, 0);
        assert_eq!(bin.capacity, 100);
        assert_eq!(bin.used, 0);
        assert!(bin.items.is_empty());
    }

    #[test]
    fn test_bin_remaining() {
        let mut bin = Bin::new(0, 100);
        bin.used = 60;
        bin.items.push(0);
        assert_eq!(bin.remaining(), 40);
    }

    // ── Pack construction and add ────────────────────────────────────

    #[test]
    fn test_pack_new() {
        let pack = Pack::new(100);
        assert_eq!(pack.capacity, 100);
        assert_eq!(pack.used_capacity(), 0);
        assert_eq!(pack.len(), 0);
        assert!(pack.is_empty());
    }

    #[test]
    fn test_pack_add_single() {
        let mut pack = Pack::new(100);
        pack.add(Sequence::new(0, 60)).unwrap();

        assert_eq!(pack.used_capacity(), 60);
        assert_eq!(pack.remaining_capacity(), 40);
        assert_eq!(pack.len(), 1);
        assert!(!pack.is_empty());
    }

    #[test]
    fn test_pack_add_multiple() {
        let mut pack = Pack::new(100);
        pack.add(Sequence::new(0, 60)).unwrap();
        pack.add(Sequence::new(1, 30)).unwrap();

        assert_eq!(pack.used_capacity(), 90);
        assert_eq!(pack.remaining_capacity(), 10);
        assert_eq!(pack.len(), 2);
    }

    #[test]
    fn test_pack_add_exact_fill() {
        let mut pack = Pack::new(100);
        pack.add(Sequence::new(0, 100)).unwrap();

        assert_eq!(pack.used_capacity(), 100);
        assert_eq!(pack.remaining_capacity(), 0);
    }

    #[test]
    fn test_pack_full_error() {
        let mut pack = Pack::new(100);
        pack.add(Sequence::new(0, 60)).unwrap();
        let err = pack.add(Sequence::new(1, 50)).unwrap_err();
        assert!(matches!(err, PackError::PackFull { length: 50 }));
    }

    // ── Utilisation and padding ──────────────────────────────────────

    #[test]
    fn test_utilisation() {
        let mut pack = Pack::new(100);
        pack.add(Sequence::new(0, 75)).unwrap();
        assert!((pack.utilisation() - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_padding_tokens() {
        let mut pack = Pack::new(100);
        pack.add(Sequence::new(0, 60)).unwrap();
        assert_eq!(pack.padding_tokens(), 40);
    }

    // ── cu_seqlens ───────────────────────────────────────────────────

    #[test]
    fn test_cu_seqlens() {
        let mut pack = Pack::new(1024);
        pack.add(Sequence::new(0, 512)).unwrap();
        pack.add(Sequence::new(1, 256)).unwrap();
        pack.add(Sequence::new(2, 128)).unwrap();

        assert_eq!(pack.cu_seqlens(), vec![0, 512, 768, 896]);
    }

    #[test]
    fn test_cu_seqlens_single_sequence() {
        let mut pack = Pack::new(100);
        pack.add(Sequence::new(0, 50)).unwrap();

        assert_eq!(pack.cu_seqlens(), vec![0, 50]);
    }

    #[test]
    fn test_cu_seqlens_empty_pack() {
        let pack = Pack::new(100);
        assert_eq!(pack.cu_seqlens(), vec![0]);
    }

    // ── max_seqlen_in_pack ───────────────────────────────────────────

    #[test]
    fn test_max_seqlen_in_pack() {
        let mut pack = Pack::new(1024);
        pack.add(Sequence::new(0, 512)).unwrap();
        pack.add(Sequence::new(1, 256)).unwrap();
        pack.add(Sequence::new(2, 128)).unwrap();

        assert_eq!(pack.max_seqlen_in_pack(), 512);
    }

    #[test]
    fn test_max_seqlen_empty_pack() {
        let pack = Pack::new(100);
        assert_eq!(pack.max_seqlen_in_pack(), 0);
    }

    // ── position_ids ─────────────────────────────────────────────────

    #[test]
    fn test_position_ids() {
        let mut pack = Pack::new(100);
        pack.add(Sequence::new(0, 3)).unwrap();
        pack.add(Sequence::new(1, 2)).unwrap();

        assert_eq!(pack.position_ids(), vec![0, 1, 2, 0, 1]);
    }

    #[test]
    fn test_position_ids_single_sequence() {
        let mut pack = Pack::new(100);
        pack.add(Sequence::new(0, 4)).unwrap();

        assert_eq!(pack.position_ids(), vec![0, 1, 2, 3]);
    }

    // ── segment_ids ──────────────────────────────────────────────────

    #[test]
    fn test_segment_ids() {
        let mut pack = Pack::new(100);
        pack.add(Sequence::new(0, 3)).unwrap();
        pack.add(Sequence::new(1, 2)).unwrap();

        assert_eq!(pack.segment_ids(), vec![0, 0, 0, 1, 1]);
    }

    #[test]
    fn test_segment_ids_three_sequences() {
        let mut pack = Pack::new(100);
        pack.add(Sequence::new(0, 2)).unwrap();
        pack.add(Sequence::new(1, 1)).unwrap();
        pack.add(Sequence::new(2, 3)).unwrap();

        assert_eq!(pack.segment_ids(), vec![0, 0, 1, 2, 2, 2]);
    }

    // ── attention_mask ───────────────────────────────────────────────

    #[test]
    fn test_attention_mask_block_diagonal() {
        let mut pack = Pack::new(100);
        pack.add(Sequence::new(0, 2)).unwrap();
        pack.add(Sequence::new(1, 2)).unwrap();

        let mask = pack.attention_mask();
        // 4×4 mask:
        //   [T .  .  . ]   seq 0: causal at (0,0), (1,0), (1,1)
        //   [T T  .  . ]
        //   [.  .  T . ]   seq 1: causal at (2,2), (3,2), (3,3)
        //   [.  .  T T ]
        #[rustfmt::skip]
        let expected = vec![
            true,  false, false, false,
            true,  true,  false, false,
            false, false, true,  false,
            false, false, true,  true,
        ];
        assert_eq!(mask, expected);
    }

    #[test]
    fn test_attention_mask_single_sequence() {
        let mut pack = Pack::new(100);
        pack.add(Sequence::new(0, 3)).unwrap();

        let mask = pack.attention_mask();
        // 3×3 lower-triangular:
        //   [T .  . ]
        //   [T T  . ]
        //   [T T  T ]
        #[rustfmt::skip]
        let expected = vec![
            true,  false, false,
            true,  true,  false,
            true,  true,  true,
        ];
        assert_eq!(mask, expected);
    }

    // ── bins_to_packs ────────────────────────────────────────────────

    #[test]
    fn test_bins_to_packs() {
        let sequences = vec![
            Sequence::new(0, 60),
            Sequence::new(1, 40),
            Sequence::new(2, 50),
        ];

        let bins = vec![
            Bin {
                id: 0,
                capacity: 100,
                used: 100,
                items: vec![0, 1].into(),
            },
            Bin {
                id: 1,
                capacity: 100,
                used: 50,
                items: vec![2].into(),
            },
        ];

        let packs = bins_to_packs(bins, &sequences);
        assert_eq!(packs.len(), 2);
        assert_eq!(packs[0].len(), 2);
        assert_eq!(packs[0].used_capacity(), 100);
        assert_eq!(packs[1].len(), 1);
        assert_eq!(packs[1].used_capacity(), 50);
    }

    #[test]
    fn test_bins_to_packs_preserves_sequence_data() {
        let sequences = vec![
            Sequence::with_tokens(0, vec![10, 20, 30]),
            Sequence::new(1, 2),
        ];

        let bins = vec![Bin {
            id: 0,
            capacity: 10,
            used: 5,
            items: vec![0, 1].into(),
        }];

        let packs = bins_to_packs(bins, &sequences);
        assert_eq!(
            packs[0].sequences[0].tokens.as_ref().unwrap(),
            &vec![10, 20, 30]
        );
        assert!(packs[0].sequences[1].tokens.is_none());
    }

    #[test]
    fn test_bins_to_packs_empty() {
        let sequences: Vec<Sequence> = vec![];
        let bins: Vec<Bin> = vec![];
        let packs = bins_to_packs(bins, &sequences);
        assert!(packs.is_empty());
    }

    // ── Metadata consistency ─────────────────────────────────────────

    #[test]
    fn test_metadata_lengths_consistent() {
        let mut pack = Pack::new(100);
        pack.add(Sequence::new(0, 3)).unwrap();
        pack.add(Sequence::new(1, 2)).unwrap();
        pack.add(Sequence::new(2, 4)).unwrap();

        let total = pack.used_capacity(); // 9

        // cu_seqlens last element should equal total used.
        let cu = pack.cu_seqlens();
        assert_eq!(*cu.last().unwrap(), total);

        // position_ids length should equal total used.
        assert_eq!(pack.position_ids().len(), total);

        // segment_ids length should equal total used.
        assert_eq!(pack.segment_ids().len(), total);

        // attention_mask length should equal total² .
        assert_eq!(pack.attention_mask().len(), total * total);
    }
}
