//! Next Fit algorithm - O(n), single open bin.

use crate::error::{PackError, Result};
use crate::pack::{Bin, Pack, bins_to_packs};
use crate::sequence::{Item, Sequence};
use crate::strategy::PackingAlgorithm;

/// Next Fit algorithm implementation.
pub struct NextFit;

impl PackingAlgorithm for NextFit {
    fn pack(&self, sequences: Vec<Sequence>, capacity: usize) -> Result<Vec<Pack>> {
        let items: Vec<Item> = sequences.iter().map(|s| s.to_item()).collect();
        let bins = next_fit_items(&items, capacity)?;
        Ok(bins_to_packs(bins, &sequences))
    }

    fn name(&self) -> &'static str {
        "NextFit"
    }
}

/// Core Next Fit on lightweight items.
fn next_fit_items(items: &[Item], capacity: usize) -> Result<Vec<Bin>> {
    let mut bins: Vec<Bin> = Vec::new();
    let mut current: Option<Bin> = None;

    for &item in items {
        if item.len > capacity {
            return Err(PackError::SequenceTooLong {
                length: item.len,
                capacity,
            });
        }

        let needs_new_bin = match &current {
            Some(bin) => bin.remaining() < item.len,
            None => true,
        };

        if needs_new_bin {
            if let Some(bin) = current.take() {
                bins.push(bin);
            }
            let mut bin = Bin::new(bins.len(), capacity);
            bin.used = item.len;
            bin.items.push(item.id);
            current = Some(bin);
        } else {
            let bin = current.as_mut().unwrap();
            bin.used += item.len;
            bin.items.push(item.id);
        }
    }

    if let Some(bin) = current {
        bins.push(bin);
    }

    Ok(bins)
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::validate_solution;

    fn seqs(lens: &[usize]) -> Vec<Sequence> {
        lens.iter()
            .enumerate()
            .map(|(id, &len)| Sequence::new(id, len))
            .collect()
    }

    fn items(lens: &[usize]) -> Vec<Item> {
        lens.iter()
            .enumerate()
            .map(|(id, &len)| Item { id, len })
            .collect()
    }

    // ── Basic behavior ──────────────────────────────────────────

    #[test]
    fn test_single_bin_fits() {
        let packs = NextFit.pack(seqs(&[3, 3, 4]), 10).unwrap();
        assert_eq!(packs.len(), 1);
    }

    #[test]
    fn test_two_bins() {
        let itms = items(&[7, 5]);
        let bins = next_fit_items(&itms, 10).unwrap();
        // 7 → bin 0 (rem=3), 5 > 3 → close bin 0, new bin 1
        assert_eq!(bins.len(), 2);
        assert_eq!(&bins[0].items[..], &[0]);
        assert_eq!(&bins[1].items[..], &[1]);
    }

    #[test]
    fn test_exact_fill_then_new_bin() {
        let itms = items(&[6, 4, 3]);
        let bins = next_fit_items(&itms, 10).unwrap();
        // 6 → bin 0 (rem=4), 4 → bin 0 (rem=0), 3 > 0 → bin 1
        assert_eq!(bins.len(), 2);
        assert_eq!(&bins[0].items[..], &[0, 1]);
        assert_eq!(bins[0].used, 10);
        assert_eq!(&bins[1].items[..], &[2]);
    }

    #[test]
    fn test_exact_fill_boundaries() {
        let packs = NextFit.pack(seqs(&[10, 10]), 10).unwrap();
        // Each item exactly fills a bin: 10 → bin 0 (rem=0), 10 > 0 → bin 1
        assert_eq!(packs.len(), 2);
    }

    // ── Pathological / never-revisit ────────────────────────────

    #[test]
    fn test_pathological_inefficiency() {
        // NF can't revisit: alternating 6,5 never fit together
        // 6 → bin 0 (rem=4), 5 > 4 → bin 1 (rem=5), 6 > 5 → bin 2 (rem=4), 5 > 4 → bin 3
        let packs = NextFit.pack(seqs(&[6, 5, 6, 5]), 10).unwrap();
        assert_eq!(packs.len(), 4);
    }

    #[test]
    fn test_never_revisits_closed_bins() {
        // Items [8, 1, 1, 1]: NF puts 8 in bin 0 (rem=2), then 1+1 in bin 0 (rem=0),
        // then 1 > 0 → bin 1. FF would also get 2 bins here.
        // But [8, 3, 2]: 8 → bin 0 (rem=2), 3 > 2 → bin 1 (rem=7), 2 → bin 1 (rem=5)
        // Bin 0 has rem=2 which could fit the 2, but NF doesn't look back.
        let itms = items(&[8, 3, 2]);
        let bins = next_fit_items(&itms, 10).unwrap();
        assert_eq!(bins.len(), 2);
        assert_eq!(&bins[0].items[..], &[0]); // 8 alone, rem=2 wasted
        assert_eq!(&bins[1].items[..], &[1, 2]); // 3+2=5
        assert_eq!(bins[0].remaining(), 2); // could have fit item 2
    }

    // ── Error cases ─────────────────────────────────────────────

    #[test]
    fn test_oversize_error() {
        let result = NextFit.pack(seqs(&[11]), 10);
        assert!(matches!(
            result,
            Err(PackError::SequenceTooLong {
                length: 11,
                capacity: 10
            })
        ));
    }

    #[test]
    fn test_oversize_stops_early() {
        // Error at item 1 (len=15), item 2 never processed
        let itms = items(&[5, 15, 3]);
        let result = next_fit_items(&itms, 10);
        assert!(result.is_err());
    }

    // ── Edge cases ──────────────────────────────────────────────

    #[test]
    fn test_empty_input() {
        let itms = items(&[]);
        let bins = next_fit_items(&itms, 10).unwrap();
        assert!(bins.is_empty());
    }

    #[test]
    fn test_single_item() {
        let itms = items(&[5]);
        let bins = next_fit_items(&itms, 10).unwrap();
        assert_eq!(bins.len(), 1);
        assert_eq!(bins[0].used, 5);
        assert_eq!(&bins[0].items[..], &[0]);
    }

    #[test]
    fn test_exact_capacity_item() {
        let itms = items(&[10]);
        let bins = next_fit_items(&itms, 10).unwrap();
        assert_eq!(bins.len(), 1);
        assert_eq!(bins[0].remaining(), 0);
    }

    #[test]
    fn test_all_same_size_pairs() {
        // 4 items of size 5, capacity 10 → pairs perfectly
        let itms = items(&[5, 5, 5, 5]);
        let bins = next_fit_items(&itms, 10).unwrap();
        assert_eq!(bins.len(), 2);
        assert_eq!(&bins[0].items[..], &[0, 1]);
        assert_eq!(&bins[1].items[..], &[2, 3]);
    }

    #[test]
    fn test_each_item_needs_own_bin() {
        // All items > capacity/2, no two fit together
        let itms = items(&[6, 7, 8, 9]);
        let bins = next_fit_items(&itms, 10).unwrap();
        assert_eq!(bins.len(), 4);
    }

    #[test]
    fn test_many_small_items_one_bin() {
        let itms = items(&[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
        let bins = next_fit_items(&itms, 10).unwrap();
        assert_eq!(bins.len(), 1);
        assert_eq!(bins[0].used, 10);
    }

    // ── Validation ──────────────────────────────────────────────

    #[test]
    fn test_validates_basic() {
        let itms = items(&[3, 7, 5, 5, 2, 8]);
        let bins = next_fit_items(&itms, 10).unwrap();
        validate_solution(&itms, &bins, 10).unwrap();
    }

    #[test]
    fn test_validates_pathological() {
        let itms = items(&[6, 5, 6, 5]);
        let bins = next_fit_items(&itms, 10).unwrap();
        validate_solution(&itms, &bins, 10).unwrap();
    }

    #[test]
    fn test_validates_many_items() {
        let lens: Vec<usize> = (1..=20).collect();
        let itms = items(&lens);
        let bins = next_fit_items(&itms, 25).unwrap();
        validate_solution(&itms, &bins, 25).unwrap();
    }

    // ── PackingAlgorithm trait ──────────────────────────────────

    #[test]
    fn test_name() {
        assert_eq!(NextFit.name(), "NextFit");
    }

    #[test]
    fn test_pack_through_trait() {
        let packs = NextFit.pack(seqs(&[3, 7, 5, 5]), 10).unwrap();
        // 3+7=10 → pack 0, 5+5=10 → pack 1
        assert_eq!(packs.len(), 2);
    }
}
