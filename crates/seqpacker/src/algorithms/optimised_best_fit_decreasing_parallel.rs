//! OBFDP — Parallel OBFD using Rayon.
//!
//! Splits work across threads with cyclical distribution,
//! runs OBFD independently per thread, then repacks partial bins.
//!
//! O(N log L / P) expected time for P threads.

use rayon::prelude::*;

use crate::error::Result;
use crate::pack::Pack;
use crate::sequence::Sequence;
use crate::strategy::PackingAlgorithm;

use super::optimised_best_fit_decreasing::{
    bins_to_packs_from_indices, optimised_best_fit_decreasing_lengths,
};

/// Parallel OBFD with adaptive thread count.
pub struct OptimisedBestFitDecreasingParallel;

impl PackingAlgorithm for OptimisedBestFitDecreasingParallel {
    fn pack(&self, sequences: Vec<Sequence>, capacity: usize) -> Result<Vec<Pack>> {
        let lengths: Vec<usize> = sequences.iter().map(|s| s.length).collect();
        let bins = optimised_best_fit_decreasing_parallel_lengths(&lengths, capacity)?;
        Ok(bins_to_packs_from_indices(bins, &sequences, capacity))
    }

    fn name(&self) -> &'static str {
        "OptimisedBestFitDecreasingParallel"
    }
}

/// Core parallel OBFD on raw lengths.
///
/// Adaptively selects thread count based on input size.
/// Falls back to sequential OBFD for small inputs (N ≤ 20k).
pub fn optimised_best_fit_decreasing_parallel_lengths(
    lengths: &[usize],
    capacity: usize,
) -> Result<Vec<Vec<usize>>> {
    if lengths.is_empty() {
        return Ok(Vec::new());
    }

    let num_threads = match lengths.len() {
        n if n <= 20_000 => 1,
        n if n <= 100_000 => 2,
        n if n <= 500_000 => 4,
        _ => rayon::current_num_threads(),
    };

    if num_threads == 1 {
        return optimised_best_fit_decreasing_lengths(lengths, capacity);
    }

    // Cyclical distribution: thread t gets indices [t, t+P, t+2P, ...]
    let groups: Vec<Vec<usize>> = (0..num_threads)
        .map(|tid| (tid..lengths.len()).step_by(num_threads).collect())
        .collect();

    // Run OBFD independently per thread.
    let results: Vec<Result<Vec<Vec<usize>>>> = groups
        .par_iter()
        .map(|group| obfd_worker(lengths, group, capacity))
        .collect();

    // Check for errors from any thread.
    let thread_results: Vec<Vec<Vec<usize>>> = results.into_iter().collect::<Result<Vec<_>>>()?;

    // Merge: keep all-but-last bins from each thread, repack last bins.
    let mut final_bins: Vec<Vec<usize>> = Vec::new();
    let mut repack_indices: Vec<usize> = Vec::new();

    for group_bins in thread_results {
        if group_bins.is_empty() {
            continue;
        }
        if group_bins.len() == 1 {
            // Only one bin — always repack it.
            repack_indices.extend(&group_bins[0]);
        } else {
            // Keep all but last bin.
            final_bins.extend(group_bins[..group_bins.len() - 1].iter().cloned());
            // Repack items from the last bin.
            repack_indices.extend(&group_bins[group_bins.len() - 1]);
        }
    }

    // Sequential repack of leftovers.
    if !repack_indices.is_empty() {
        let repack_lengths: Vec<usize> = repack_indices.iter().map(|&i| lengths[i]).collect();
        let repacked = optimised_best_fit_decreasing_lengths(&repack_lengths, capacity)?;
        // Map local indices back to original indices.
        for bin in repacked {
            let mapped: Vec<usize> = bin.iter().map(|&local| repack_indices[local]).collect();
            final_bins.push(mapped);
        }
    }

    Ok(final_bins)
}

/// Run OBFD on a subset of items (identified by indices into the original lengths array).
fn obfd_worker(
    all_lengths: &[usize],
    indices: &[usize],
    capacity: usize,
) -> Result<Vec<Vec<usize>>> {
    if indices.is_empty() {
        return Ok(Vec::new());
    }

    // Extract lengths for this worker's items.
    let worker_lengths: Vec<usize> = indices.iter().map(|&i| all_lengths[i]).collect();
    let local_bins = optimised_best_fit_decreasing_lengths(&worker_lengths, capacity)?;

    // Map local indices back to original indices.
    Ok(local_bins
        .into_iter()
        .map(|bin| bin.into_iter().map(|local| indices[local]).collect())
        .collect())
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper ─────────────────────────────────────────────────────────

    fn validate_bins(bins: &[Vec<usize>], lengths: &[usize], capacity: usize) {
        // Every bin respects capacity.
        for (i, bin) in bins.iter().enumerate() {
            let total: usize = bin.iter().map(|&idx| lengths[idx]).sum();
            assert!(
                total <= capacity,
                "bin {i} total {total} exceeds capacity {capacity}"
            );
        }

        // Every item appears exactly once.
        let mut seen = vec![false; lengths.len()];
        for bin in bins {
            for &idx in bin {
                assert!(!seen[idx], "item {idx} appears in multiple bins");
                seen[idx] = true;
            }
        }
        for (i, &s) in seen.iter().enumerate() {
            assert!(s, "item {i} missing from bins");
        }
    }

    // ── Basic functionality ────────────────────────────────────────────

    #[test]
    fn test_empty_input() {
        let bins = optimised_best_fit_decreasing_parallel_lengths(&[], 10).unwrap();
        assert!(bins.is_empty());
    }

    #[test]
    fn test_small_input_delegates_to_obfd() {
        // With ≤20k items, should use 1 thread (same result as OBFD).
        let lengths = &[6, 4, 6, 4];
        let bins = optimised_best_fit_decreasing_parallel_lengths(lengths, 10).unwrap();
        assert_eq!(bins.len(), 2);
        validate_bins(&bins, lengths, 10);
    }

    #[test]
    fn test_all_items_packed() {
        let lengths: Vec<usize> = (1..=100).collect();
        let bins = optimised_best_fit_decreasing_parallel_lengths(&lengths, 200).unwrap();
        validate_bins(&bins, &lengths, 200);
    }

    #[test]
    fn test_all_same_size() {
        let lengths = &[5, 5, 5, 5];
        let bins = optimised_best_fit_decreasing_parallel_lengths(lengths, 10).unwrap();
        assert_eq!(bins.len(), 2);
        validate_bins(&bins, lengths, 10);
    }

    #[test]
    fn test_exact_capacity_items() {
        let lengths = &[10, 10, 10];
        let bins = optimised_best_fit_decreasing_parallel_lengths(lengths, 10).unwrap();
        assert_eq!(bins.len(), 3);
        validate_bins(&bins, lengths, 10);
    }

    #[test]
    fn test_error_propagated() {
        let result = optimised_best_fit_decreasing_parallel_lengths(&[11], 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_validates_with_validation() {
        use crate::pack::Bin;
        use crate::sequence::Item;
        use crate::validation::validate_solution;

        let lengths: Vec<usize> = (1..=50).map(|i| i * 2).collect();
        let bins_items = optimised_best_fit_decreasing_parallel_lengths(&lengths, 100).unwrap();

        let bins: Vec<Bin> = bins_items
            .iter()
            .enumerate()
            .map(|(id, items)| {
                let used: usize = items.iter().map(|&i| lengths[i]).sum();
                Bin {
                    id,
                    capacity: 100,
                    used,
                    items: items.clone().into(),
                }
            })
            .collect();

        let itms: Vec<Item> = lengths
            .iter()
            .enumerate()
            .map(|(id, &len)| Item { id, len })
            .collect();

        validate_solution(&itms, &bins, 100).unwrap();
    }

    // ── PackingAlgorithm trait ─────────────────────────────────────────

    #[test]
    fn test_packing_algorithm_trait() {
        let algo = OptimisedBestFitDecreasingParallel;
        let sequences = vec![
            Sequence::new(0, 6),
            Sequence::new(1, 4),
            Sequence::new(2, 3),
        ];
        let packs = algo.pack(sequences, 10).unwrap();
        assert!(!packs.is_empty());
        let total_seqs: usize = packs.iter().map(|p| p.len()).sum();
        assert_eq!(total_seqs, 3);
    }

    #[test]
    fn test_name() {
        assert_eq!(
            OptimisedBestFitDecreasingParallel.name(),
            "OptimisedBestFitDecreasingParallel"
        );
    }
}
