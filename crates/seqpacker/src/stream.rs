//! Streaming packer for bounded-space online algorithms.
//!
//! Only NextFit and Harmonic-K support true streaming — they have a
//! bounded number of open bins at any time, so completed packs can be
//! emitted before all input is seen.
//!
//! For unbounded-space algorithms (FF, BF, WF) or offline algorithms
//! (FFD, BFD, OBFD, etc.), use [`Packer::pack()`](crate::Packer::pack).

use crate::error::{PackError, Result};
use crate::pack::Pack;
use crate::sequence::Sequence;

/// Streaming strategy selector.
///
/// Only bounded-space online algorithms are supported.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamStrategy {
    /// Next Fit: 1 open bin at a time.
    NextFit,
    /// Harmonic-K: k open bins (one per size class).
    Harmonic,
}

/// Streaming packer that emits completed packs incrementally.
///
/// # Example
///
/// ```
/// use seqpacker::stream::{StreamPacker, StreamStrategy};
///
/// let mut sp = StreamPacker::new(10, StreamStrategy::NextFit);
///
/// // Each add() may return closed packs.
/// let closed = sp.add(7).unwrap();
/// assert!(closed.is_empty()); // bin still open
///
/// let closed = sp.add(5).unwrap(); // 5 > remaining 3 → closes bin
/// assert_eq!(closed.len(), 1);
/// assert_eq!(closed[0].used_capacity(), 7);
///
/// // finish() flushes the last open bin.
/// let remaining = sp.finish();
/// assert_eq!(remaining.len(), 1);
/// assert_eq!(remaining[0].used_capacity(), 5);
/// ```
pub struct StreamPacker {
    capacity: usize,
    strategy: StreamStrategy,
    state: StreamState,
    next_id: usize,
}

enum StreamState {
    NextFit {
        current: Option<Pack>,
    },
    Harmonic {
        k: usize,
        /// One open pack per size class (None if no open pack for that class).
        class_packs: Vec<Option<Pack>>,
        /// Catch-all class may have multiple open packs (first-fit).
        catchall_packs: Vec<Pack>,
    },
}

impl StreamPacker {
    /// Create a new streaming packer.
    pub fn new(capacity: usize, strategy: StreamStrategy) -> Self {
        Self::with_k(capacity, strategy, 10)
    }

    /// Create a streaming packer with a custom k for Harmonic.
    ///
    /// `k` is ignored for NextFit.
    pub fn with_k(capacity: usize, strategy: StreamStrategy, k: usize) -> Self {
        assert!(k >= 2, "Harmonic-K requires k >= 2");
        let state = match strategy {
            StreamStrategy::NextFit => StreamState::NextFit { current: None },
            StreamStrategy::Harmonic => StreamState::Harmonic {
                k,
                class_packs: vec![None; k],
                catchall_packs: Vec::new(),
            },
        };
        Self {
            capacity,
            strategy,
            state,
            next_id: 0,
        }
    }

    /// Add a sequence length and return any packs that are now complete.
    ///
    /// # Errors
    ///
    /// Returns `PackError::SequenceTooLong` if `length` exceeds capacity.
    pub fn add(&mut self, length: usize) -> Result<Vec<Pack>> {
        if length > self.capacity {
            return Err(PackError::SequenceTooLong {
                length,
                capacity: self.capacity,
            });
        }
        let id = self.next_id;
        self.next_id += 1;
        let seq = Sequence::new(id, length);

        match &mut self.state {
            StreamState::NextFit { current } => Self::add_next_fit(current, self.capacity, seq),
            StreamState::Harmonic {
                k,
                class_packs,
                catchall_packs,
            } => Self::add_harmonic(*k, class_packs, catchall_packs, self.capacity, seq),
        }
    }

    /// Flush all remaining open bins and return them as packs.
    pub fn finish(self) -> Vec<Pack> {
        match self.state {
            StreamState::NextFit { current } => {
                current.into_iter().filter(|p| !p.is_empty()).collect()
            }
            StreamState::Harmonic {
                class_packs,
                catchall_packs,
                ..
            } => {
                let mut packs: Vec<Pack> = class_packs
                    .into_iter()
                    .flatten()
                    .filter(|p| !p.is_empty())
                    .collect();
                for pack in catchall_packs {
                    if !pack.is_empty() {
                        packs.push(pack);
                    }
                }
                packs
            }
        }
    }

    /// Number of sequences added so far.
    pub fn sequences_added(&self) -> usize {
        self.next_id
    }

    /// The streaming strategy in use.
    pub fn strategy(&self) -> StreamStrategy {
        self.strategy
    }

    // ── NextFit internals ────────────────────────────────────────────

    fn add_next_fit(
        current: &mut Option<Pack>,
        capacity: usize,
        seq: Sequence,
    ) -> Result<Vec<Pack>> {
        let mut closed = Vec::new();

        let needs_new = match current {
            Some(pack) => pack.remaining_capacity() < seq.length,
            None => true,
        };

        if needs_new {
            if let Some(full_pack) = current.take() {
                closed.push(full_pack);
            }
            let mut pack = Pack::new(capacity);
            // add() cannot fail here — we already checked length <= capacity.
            pack.add(seq).expect("length <= capacity");
            *current = Some(pack);
        } else {
            let pack = current.as_mut().unwrap();
            pack.add(seq).expect("checked remaining >= length");
        }

        Ok(closed)
    }

    // ── Harmonic internals ───────────────────────────────────────────

    fn classify(k: usize, len: usize, capacity: usize) -> usize {
        if len == 0 {
            return k - 1;
        }
        let ratio = capacity / len;
        if ratio <= 1 {
            0
        } else {
            (ratio - 1).min(k - 1)
        }
    }

    fn max_items_for_class(k: usize, class: usize) -> usize {
        if class == k - 1 {
            usize::MAX
        } else {
            class + 1
        }
    }

    fn add_harmonic(
        k: usize,
        class_packs: &mut [Option<Pack>],
        catchall_packs: &mut Vec<Pack>,
        capacity: usize,
        seq: Sequence,
    ) -> Result<Vec<Pack>> {
        let mut closed = Vec::new();
        let class = Self::classify(k, seq.length, capacity);

        if class == k - 1 {
            // Catch-all: first-fit among open catch-all packs.
            let mut placed = false;
            for pack in catchall_packs.iter_mut() {
                if pack.remaining_capacity() >= seq.length {
                    pack.add(seq.clone()).expect("checked remaining");
                    placed = true;
                    break;
                }
            }
            if !placed {
                let mut pack = Pack::new(capacity);
                pack.add(seq).expect("length <= capacity");
                catchall_packs.push(pack);
            }
        } else {
            let max_items = Self::max_items_for_class(k, class);

            let fits_existing = match &class_packs[class] {
                Some(pack) => pack.remaining_capacity() >= seq.length && pack.len() < max_items,
                None => false,
            };

            if fits_existing {
                let pack = class_packs[class].as_mut().unwrap();
                pack.add(seq).expect("checked remaining");
                // Close if full.
                if pack.len() >= max_items {
                    let full_pack = class_packs[class].take().unwrap();
                    closed.push(full_pack);
                }
            } else {
                // Close old pack for this class if it exists.
                if let Some(old_pack) = class_packs[class].take() {
                    closed.push(old_pack);
                }
                // Open a new pack.
                let mut pack = Pack::new(capacity);
                pack.add(seq).expect("length <= capacity");
                if max_items <= 1 {
                    // Immediately full (class 0: large items).
                    closed.push(pack);
                } else {
                    class_packs[class] = Some(pack);
                }
            }
        }

        Ok(closed)
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── NextFit streaming ────────────────────────────────────────────

    #[test]
    fn test_nf_single_item() {
        let mut sp = StreamPacker::new(10, StreamStrategy::NextFit);
        let closed = sp.add(5).unwrap();
        assert!(closed.is_empty());
        let remaining = sp.finish();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].used_capacity(), 5);
    }

    #[test]
    fn test_nf_emits_on_overflow() {
        let mut sp = StreamPacker::new(10, StreamStrategy::NextFit);
        sp.add(7).unwrap();
        let closed = sp.add(5).unwrap(); // 5 > remaining 3
        assert_eq!(closed.len(), 1);
        assert_eq!(closed[0].used_capacity(), 7);
        assert_eq!(closed[0].len(), 1);
    }

    #[test]
    fn test_nf_exact_fill_no_emit() {
        let mut sp = StreamPacker::new(10, StreamStrategy::NextFit);
        sp.add(6).unwrap();
        let closed = sp.add(4).unwrap(); // exactly fills
        assert!(closed.is_empty()); // no emit until next item forces it
        let closed = sp.add(1).unwrap(); // forces emit of full bin
        assert_eq!(closed.len(), 1);
        assert_eq!(closed[0].used_capacity(), 10);
    }

    #[test]
    fn test_nf_multiple_items_same_bin() {
        let mut sp = StreamPacker::new(10, StreamStrategy::NextFit);
        sp.add(3).unwrap();
        sp.add(3).unwrap();
        let closed = sp.add(3).unwrap();
        assert!(closed.is_empty()); // 3+3+3=9 still fits
        let remaining = sp.finish();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].used_capacity(), 9);
        assert_eq!(remaining[0].len(), 3);
    }

    #[test]
    fn test_nf_finish_empty() {
        let sp = StreamPacker::new(10, StreamStrategy::NextFit);
        let remaining = sp.finish();
        assert!(remaining.is_empty());
    }

    #[test]
    fn test_nf_oversize_error() {
        let mut sp = StreamPacker::new(10, StreamStrategy::NextFit);
        let result = sp.add(11);
        assert!(matches!(
            result,
            Err(PackError::SequenceTooLong {
                length: 11,
                capacity: 10
            })
        ));
    }

    #[test]
    fn test_nf_all_sequences_accounted() {
        let mut sp = StreamPacker::new(10, StreamStrategy::NextFit);
        let lens = [3, 7, 5, 5, 2, 8];
        let mut all_packs = Vec::new();
        for &len in &lens {
            all_packs.extend(sp.add(len).unwrap());
        }
        all_packs.extend(sp.finish());

        let total_seqs: usize = all_packs.iter().map(|p| p.len()).sum();
        assert_eq!(total_seqs, lens.len());

        let total_tokens: usize = all_packs.iter().map(|p| p.used_capacity()).sum();
        assert_eq!(total_tokens, lens.iter().sum::<usize>());
    }

    #[test]
    fn test_nf_sequence_ids_are_sequential() {
        let mut sp = StreamPacker::new(10, StreamStrategy::NextFit);
        let mut all_packs = Vec::new();
        for &len in &[6, 5, 6, 5] {
            all_packs.extend(sp.add(len).unwrap());
        }
        all_packs.extend(sp.finish());

        let mut ids: Vec<usize> = all_packs
            .iter()
            .flat_map(|p| p.sequences.iter().map(|s| s.id))
            .collect();
        ids.sort();
        assert_eq!(ids, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_nf_sequences_added_counter() {
        let mut sp = StreamPacker::new(10, StreamStrategy::NextFit);
        assert_eq!(sp.sequences_added(), 0);
        sp.add(5).unwrap();
        assert_eq!(sp.sequences_added(), 1);
        sp.add(3).unwrap();
        assert_eq!(sp.sequences_added(), 2);
    }

    // ── Harmonic streaming ───────────────────────────────────────────

    #[test]
    fn test_hk_large_items_emit_immediately() {
        let mut sp = StreamPacker::new(100, StreamStrategy::Harmonic);
        // Items > 50% capacity → class 0 → max 1 per bin → emit immediately.
        let closed = sp.add(60).unwrap();
        assert_eq!(closed.len(), 1);
        assert_eq!(closed[0].used_capacity(), 60);
    }

    #[test]
    fn test_hk_medium_items_pair_then_emit() {
        let mut sp = StreamPacker::new(100, StreamStrategy::Harmonic);
        // Items in (1/3, 1/2] → class 1 → max 2 per bin.
        let closed = sp.add(40).unwrap();
        assert!(closed.is_empty()); // first item opens bin
        let closed = sp.add(40).unwrap();
        assert_eq!(closed.len(), 1); // second item fills and closes
        assert_eq!(closed[0].len(), 2);
    }

    #[test]
    fn test_hk_different_classes_separate() {
        let mut sp = StreamPacker::new(100, StreamStrategy::Harmonic);
        sp.add(60).unwrap(); // class 0, emitted immediately
        sp.add(40).unwrap(); // class 1, opens new bin
        let remaining = sp.finish();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].used_capacity(), 40);
    }

    #[test]
    fn test_hk_catchall_first_fit() {
        let mut sp = StreamPacker::new(100, StreamStrategy::Harmonic);
        // Small items go to catch-all class (first-fit).
        for _ in 0..10 {
            sp.add(5).unwrap();
        }
        let remaining = sp.finish();
        // 10 * 5 = 50, all fit in one catch-all bin.
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].used_capacity(), 50);
    }

    #[test]
    fn test_hk_class_bin_closes_on_new_item() {
        let mut sp = StreamPacker::new(100, StreamStrategy::Harmonic);
        // Class 1 (40%): max 2 items. Add 2 to fill, then a third forces new bin.
        sp.add(40).unwrap();
        let closed = sp.add(40).unwrap(); // closes first class-1 bin
        assert_eq!(closed.len(), 1);
        sp.add(40).unwrap(); // opens new class-1 bin
        let remaining = sp.finish();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].used_capacity(), 40);
    }

    #[test]
    fn test_hk_oversize_error() {
        let mut sp = StreamPacker::new(100, StreamStrategy::Harmonic);
        let result = sp.add(101);
        assert!(matches!(
            result,
            Err(PackError::SequenceTooLong {
                length: 101,
                capacity: 100
            })
        ));
    }

    #[test]
    fn test_hk_all_sequences_accounted() {
        let mut sp = StreamPacker::new(100, StreamStrategy::Harmonic);
        let lens = [60, 40, 30, 25, 10, 5, 80, 35];
        let mut all_packs = Vec::new();
        for &len in &lens {
            all_packs.extend(sp.add(len).unwrap());
        }
        all_packs.extend(sp.finish());

        let total_seqs: usize = all_packs.iter().map(|p| p.len()).sum();
        assert_eq!(total_seqs, lens.len());

        let mut ids: Vec<usize> = all_packs
            .iter()
            .flat_map(|p| p.sequences.iter().map(|s| s.id))
            .collect();
        ids.sort();
        assert_eq!(ids, (0..lens.len()).collect::<Vec<_>>());
    }

    #[test]
    fn test_hk_finish_empty() {
        let sp = StreamPacker::new(100, StreamStrategy::Harmonic);
        let remaining = sp.finish();
        assert!(remaining.is_empty());
    }

    #[test]
    fn test_hk_no_bin_exceeds_capacity() {
        let mut sp = StreamPacker::new(100, StreamStrategy::Harmonic);
        let lens = [60, 40, 30, 25, 10, 5, 80, 35, 15, 45, 70, 20];
        let mut all_packs = Vec::new();
        for &len in &lens {
            all_packs.extend(sp.add(len).unwrap());
        }
        all_packs.extend(sp.finish());

        for pack in &all_packs {
            assert!(pack.used_capacity() <= 100);
        }
    }

    // ── Strategy accessor ────────────────────────────────────────────

    #[test]
    fn test_strategy_accessor() {
        let sp = StreamPacker::new(10, StreamStrategy::NextFit);
        assert_eq!(sp.strategy(), StreamStrategy::NextFit);
    }

    // ── Custom k ─────────────────────────────────────────────────────

    #[test]
    fn test_custom_k() {
        let mut sp = StreamPacker::with_k(100, StreamStrategy::Harmonic, 3);
        // k=3: class 0 = (1/2, 1], class 1 = (1/3, 1/2], class 2 = catch-all
        let closed = sp.add(60).unwrap(); // class 0 → immediate emit
        assert_eq!(closed.len(), 1);
    }

    #[test]
    #[should_panic(expected = "k >= 2")]
    fn test_k_less_than_2_panics() {
        StreamPacker::with_k(100, StreamStrategy::Harmonic, 1);
    }
}
