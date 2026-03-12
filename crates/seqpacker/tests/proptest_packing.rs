//! Property-based tests — random inputs, all algorithms must produce valid solutions.

use proptest::prelude::*;
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

fn all_algorithms() -> Vec<Box<dyn PackingAlgorithm>> {
    vec![
        Box::new(NextFit),
        Box::new(FirstFit),
        Box::new(BestFit),
        Box::new(WorstFit),
        Box::new(FirstFitDecreasing),
        Box::new(BestFitDecreasing),
        Box::new(FirstFitShuffle::new(42)),
        Box::new(OptimisedBestFitDecreasing),
        Box::new(OptimisedBestFitDecreasingParallel),
    ]
}

// ── Property tests ────────────────────────────────────────────────────

proptest! {
    #[test]
    fn prop_all_algorithms_valid(
        lens in prop::collection::vec(1usize..1000, 1..200),
        capacity in 100usize..2000,
    ) {
        // Filter to items that fit in capacity.
        let valid_lens: Vec<usize> = lens.into_iter().filter(|&l| l <= capacity).collect();
        if valid_lens.is_empty() {
            return Ok(());
        }

        let items = make_items(&valid_lens);

        for algo in all_algorithms() {
            let sequences = make_sequences(&valid_lens);
            let packs = algo.pack(sequences, capacity)?;

            let bins = packs_to_bins(&packs, capacity);
            validate_solution(&items, &bins, capacity)
                .map_err(|e| TestCaseError::Fail(format!("{}: {}", algo.name(), e).into()))?;
        }
    }

    #[test]
    fn prop_all_items_accounted(
        lens in prop::collection::vec(1usize..500, 1..100),
        capacity in 50usize..1000,
    ) {
        let valid_lens: Vec<usize> = lens.into_iter().filter(|&l| l <= capacity).collect();
        if valid_lens.is_empty() {
            return Ok(());
        }

        for algo in all_algorithms() {
            let sequences = make_sequences(&valid_lens);
            let packs = algo.pack(sequences, capacity)?;

            let total: usize = packs.iter().map(|p| p.sequences.len()).sum();
            prop_assert_eq!(
                total,
                valid_lens.len(),
                "{}: expected {} items, got {}",
                algo.name(),
                valid_lens.len(),
                total
            );
        }
    }

    #[test]
    fn prop_no_pack_exceeds_capacity(
        lens in prop::collection::vec(1usize..500, 1..100),
        capacity in 50usize..1000,
    ) {
        let valid_lens: Vec<usize> = lens.into_iter().filter(|&l| l <= capacity).collect();
        if valid_lens.is_empty() {
            return Ok(());
        }

        for algo in all_algorithms() {
            let sequences = make_sequences(&valid_lens);
            let packs = algo.pack(sequences, capacity)?;

            for pack in &packs {
                let used: usize = pack.sequences.iter().map(|s| s.length).sum();
                prop_assert!(
                    used <= capacity,
                    "{}: pack used {} > capacity {}",
                    algo.name(),
                    used,
                    capacity
                );
            }
        }
    }

    #[test]
    fn prop_ffd_never_worse_than_nf(
        lens in prop::collection::vec(1usize..500, 2..100),
        capacity in 50usize..1000,
    ) {
        let valid_lens: Vec<usize> = lens.into_iter().filter(|&l| l <= capacity).collect();
        if valid_lens.is_empty() {
            return Ok(());
        }

        let nf_packs = NextFit.pack(make_sequences(&valid_lens), capacity)?;
        let ffd_packs = FirstFitDecreasing.pack(make_sequences(&valid_lens), capacity)?;

        prop_assert!(
            ffd_packs.len() <= nf_packs.len(),
            "FFD ({}) > NF ({}) — should never happen",
            ffd_packs.len(),
            nf_packs.len()
        );
    }

    #[test]
    fn prop_oversize_always_rejected(
        small_lens in prop::collection::vec(1usize..50, 0..10),
        oversize in 101usize..500,
    ) {
        let capacity = 100;
        let mut lens = small_lens;
        lens.push(oversize);

        for algo in all_algorithms() {
            let sequences = make_sequences(&lens);
            let result = algo.pack(sequences, capacity);
            prop_assert!(
                result.is_err(),
                "{}: oversize {} should have been rejected (cap={})",
                algo.name(),
                oversize,
                capacity
            );
        }
    }
}
