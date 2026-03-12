//! Golden invariant tests — run all algorithms on known inputs and
//! verify that `validate_solution` passes on every result.

use seqpacker::Sequence;
use seqpacker::algorithms::*;
use seqpacker::pack::{Bin, Pack};
use seqpacker::sequence::Item;
use seqpacker::strategy::PackingAlgorithm;
use seqpacker::validation::validate_solution;

// ── Helpers ───────────────────────────────────────────────────────────

fn make_sequences(lens: &[usize]) -> Vec<Sequence> {
    lens.iter()
        .enumerate()
        .map(|(id, &len)| Sequence::new(id, len))
        .collect()
}

fn make_items(lens: &[usize]) -> Vec<Item> {
    lens.iter()
        .enumerate()
        .map(|(id, &len)| Item { id, len })
        .collect()
}

/// Reconstruct `Bin`s from `Pack`s so we can call `validate_solution`.
fn packs_to_bins(packs: &[Pack], capacity: usize) -> Vec<Bin> {
    packs
        .iter()
        .enumerate()
        .map(|(id, pack)| {
            let item_ids: Vec<usize> = pack.sequences.iter().map(|s| s.id).collect();
            let used: usize = pack.sequences.iter().map(|s| s.length).sum();
            Bin {
                id,
                capacity,
                used,
                items: item_ids.into(),
            }
        })
        .collect()
}

/// All deterministic algorithms (excludes FFS which needs a seed).
fn all_deterministic_algorithms() -> Vec<Box<dyn PackingAlgorithm>> {
    vec![
        Box::new(NextFit),
        Box::new(FirstFit),
        Box::new(BestFit),
        Box::new(WorstFit),
        Box::new(FirstFitDecreasing),
        Box::new(BestFitDecreasing),
        Box::new(OptimisedBestFitDecreasing),
        Box::new(OptimisedBestFitDecreasingParallel),
    ]
}

/// All algorithms including FFS.
fn all_algorithms() -> Vec<Box<dyn PackingAlgorithm>> {
    let mut algos = all_deterministic_algorithms();
    algos.push(Box::new(FirstFitShuffle::new(42)));
    algos
}

// ── Test data ─────────────────────────────────────────────────────────

const GOLDEN_TESTS: &[(usize, &[usize])] = &[
    // Basic
    (10, &[1, 2, 3, 4]),       // easy — all fit in one bin
    (10, &[10, 10, 10]),       // exact fills — 3 bins
    (10, &[6, 4, 6, 4]),       // perfect pairs — 2 bins
    (10, &[6, 5, 6, 5]),       // pathological for NF (3 bins vs 2)
    (10, &[9, 1, 9, 1, 9, 1]), // fragmentation — large + small
    (128, &[64, 64, 64, 64]),  // equal half-capacity items
    // Edge cases
    (10, &[1]),                            // single item
    (10, &[10]),                           // single item = capacity
    (10, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), // many small — 1 bin
    (10, &[6, 7, 8, 9]),                   // all > half capacity — 4 bins
    // Larger capacity
    (100, &[30, 40, 50, 60, 20, 10, 80, 15, 25, 35]),
    (2048, &[500, 600, 400, 1000, 200, 800, 300]),
];

const OVERSIZE_TESTS: &[(usize, &[usize])] = &[
    (10, &[11]),         // single oversize
    (10, &[5, 15, 3]),   // mixed with oversize
    (10, &[10, 10, 11]), // oversize at end
];

// ── Tests ─────────────────────────────────────────────────────────────

#[test]
fn test_all_algorithms_golden() {
    for algo in all_deterministic_algorithms() {
        for &(capacity, lens) in GOLDEN_TESTS {
            let sequences = make_sequences(lens);
            let items = make_items(lens);

            let packs = algo.pack(sequences, capacity).unwrap_or_else(|e| {
                panic!("{}: golden test failed on {:?}: {}", algo.name(), lens, e)
            });

            let bins = packs_to_bins(&packs, capacity);
            validate_solution(&items, &bins, capacity).unwrap_or_else(|e| {
                panic!("{}: validation failed on {:?}: {}", algo.name(), lens, e)
            });
        }
    }
}

#[test]
fn test_all_algorithms_oversize_error() {
    for algo in all_deterministic_algorithms() {
        for &(capacity, lens) in OVERSIZE_TESTS {
            let sequences = make_sequences(lens);
            let result = algo.pack(sequences, capacity);
            assert!(
                result.is_err(),
                "{}: expected error on oversize input {:?} with capacity {}",
                algo.name(),
                lens,
                capacity
            );
        }
    }
}

#[test]
fn test_ffs_golden() {
    let algo = FirstFitShuffle::new(42);
    for &(capacity, lens) in GOLDEN_TESTS {
        let sequences = make_sequences(lens);
        let items = make_items(lens);
        let packs = algo
            .pack(sequences, capacity)
            .unwrap_or_else(|e| panic!("FFS: golden test failed on {:?}: {}", lens, e));

        let bins = packs_to_bins(&packs, capacity);
        validate_solution(&items, &bins, capacity)
            .unwrap_or_else(|e| panic!("FFS: validation failed on {:?}: {}", lens, e));
    }
}

#[test]
fn test_ffs_oversize_error() {
    let algo = FirstFitShuffle::new(42);
    for &(capacity, lens) in OVERSIZE_TESTS {
        let sequences = make_sequences(lens);
        let result = algo.pack(sequences, capacity);
        assert!(
            result.is_err(),
            "FFS: expected error on oversize input {:?} with capacity {}",
            lens,
            capacity
        );
    }
}

#[test]
fn test_all_sequences_accounted_for() {
    for algo in all_algorithms() {
        for &(capacity, lens) in GOLDEN_TESTS {
            let sequences = make_sequences(lens);
            let packs = algo.pack(sequences, capacity).unwrap();

            let total: usize = packs.iter().map(|p| p.sequences.len()).sum();
            assert_eq!(
                total,
                lens.len(),
                "{}: expected {} sequences, got {} on {:?}",
                algo.name(),
                lens.len(),
                total,
                lens
            );
        }
    }
}

#[test]
fn test_no_pack_exceeds_capacity() {
    for algo in all_algorithms() {
        for &(capacity, lens) in GOLDEN_TESTS {
            let sequences = make_sequences(lens);
            let packs = algo.pack(sequences, capacity).unwrap();

            for (i, pack) in packs.iter().enumerate() {
                let used: usize = pack.sequences.iter().map(|s| s.length).sum();
                assert!(
                    used <= capacity,
                    "{}: pack {} used {} > capacity {} on {:?}",
                    algo.name(),
                    i,
                    used,
                    capacity,
                    lens
                );
            }
        }
    }
}

#[test]
fn test_ffd_at_most_as_many_bins_as_nf() {
    // FFD (best offline) should never use more bins than NF (worst online).
    for &(capacity, lens) in GOLDEN_TESTS {
        let nf_packs = NextFit.pack(make_sequences(lens), capacity).unwrap();
        let ffd_packs = FirstFitDecreasing
            .pack(make_sequences(lens), capacity)
            .unwrap();
        assert!(
            ffd_packs.len() <= nf_packs.len(),
            "FFD ({}) > NF ({}) on {:?} cap={}",
            ffd_packs.len(),
            nf_packs.len(),
            lens,
            capacity
        );
    }
}

#[test]
fn test_obfd_matches_ffd_bin_count() {
    // OBFD uses best-fit decreasing — should produce same or fewer bins as FFD.
    for &(capacity, lens) in GOLDEN_TESTS {
        let ffd_packs = FirstFitDecreasing
            .pack(make_sequences(lens), capacity)
            .unwrap();
        let obfd_packs = OptimisedBestFitDecreasing
            .pack(make_sequences(lens), capacity)
            .unwrap();
        assert!(
            obfd_packs.len() <= ffd_packs.len(),
            "OBFD ({}) > FFD ({}) on {:?} cap={}",
            obfd_packs.len(),
            ffd_packs.len(),
            lens,
            capacity
        );
    }
}

#[test]
fn test_obfdp_matches_obfd_on_small_input() {
    // For small inputs (≤20k), OBFDP delegates to OBFD — same result.
    for &(capacity, lens) in GOLDEN_TESTS {
        let obfd_packs = OptimisedBestFitDecreasing
            .pack(make_sequences(lens), capacity)
            .unwrap();
        let obfdp_packs = OptimisedBestFitDecreasingParallel
            .pack(make_sequences(lens), capacity)
            .unwrap();
        assert_eq!(
            obfd_packs.len(),
            obfdp_packs.len(),
            "OBFD ({}) != OBFDP ({}) on {:?} cap={} (should match for small input)",
            obfd_packs.len(),
            obfdp_packs.len(),
            lens,
            capacity
        );
    }
}
