//! Synthetic dataset generators for benchmarking.
//!
//! Replicates the 4 distributions from `benchmarks/python/datasets/synthetic.py`
//! so that Rust and Python benchmarks operate on comparable data. Note that the
//! random number generators differ (Rust's `StdRng` vs NumPy's PCG64), so the
//! exact sequences won't be identical, but the statistical properties match.
//!
//! # Distributions
//!
//! | Name | Parameters | Description |
//! |------|-----------|-------------|
//! | Uniform | min=16, max=512 | Flat distribution across length range |
//! | Log-normal | mean=128, std=64 | Right-skewed, models natural text lengths |
//! | Exponential | mean=128 | Heavy short-sequence bias |
//! | Bimodal | short=64, long=512, ratio=0.7 | Two peaks (70% short, 30% long) |

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand_distr::{Distribution, Exp, LogNormal, Normal, Uniform};

/// Deterministic synthetic dataset generators for benchmarking.
///
/// All methods are seeded for reproducibility. Each returns a `Vec<usize>` of
/// sequence lengths.
pub struct SyntheticDataGenerator;

impl SyntheticDataGenerator {
    /// Uniformly distributed sequence lengths in [min_len, max_len].
    pub fn uniform(n: usize, min_len: usize, max_len: usize, seed: u64) -> Vec<usize> {
        let mut rng = StdRng::seed_from_u64(seed);
        let dist = Uniform::new_inclusive(min_len, max_len);
        (0..n).map(|_| dist.sample(&mut rng)).collect()
    }

    /// Log-normally distributed sequence lengths.
    ///
    /// Matches Python's parameterisation: given desired mean and std of the
    /// *output* distribution, compute mu/sigma of the underlying normal.
    pub fn lognormal(n: usize, mean_len: f64, std_len: f64, seed: u64) -> Vec<usize> {
        let mut rng = StdRng::seed_from_u64(seed);
        let variance = std_len * std_len;
        let mu = (mean_len * mean_len / (variance + mean_len * mean_len).sqrt()).ln();
        let sigma = (1.0 + variance / (mean_len * mean_len)).ln().sqrt();
        let dist = LogNormal::new(mu, sigma).expect("invalid lognormal params");
        (0..n)
            .map(|_| {
                let v: f64 = dist.sample(&mut rng);
                v.round().max(1.0) as usize
            })
            .collect()
    }

    /// Exponentially distributed sequence lengths.
    ///
    /// Uses `Exp(1/mean)`. Heavy bias towards short sequences with a long tail.
    pub fn exponential(n: usize, mean_len: f64, seed: u64) -> Vec<usize> {
        let mut rng = StdRng::seed_from_u64(seed);
        let dist = Exp::new(1.0 / mean_len).expect("invalid exp param");
        (0..n)
            .map(|_| {
                let v: f64 = dist.sample(&mut rng);
                v.round().max(1.0) as usize
            })
            .collect()
    }

    /// Bimodal distribution: mix of two normal distributions.
    ///
    /// Generates `n * short_ratio` short sequences from `Normal(short_mean, short_mean*0.2)`
    /// and the remainder from `Normal(long_mean, long_mean*0.2)`, then shuffles.
    pub fn bimodal(
        n: usize,
        short_mean: f64,
        long_mean: f64,
        short_ratio: f64,
        seed: u64,
    ) -> Vec<usize> {
        let mut rng = StdRng::seed_from_u64(seed);
        let n_short = (n as f64 * short_ratio) as usize;
        let n_long = n - n_short;

        let short_dist = Normal::new(short_mean, short_mean * 0.2).expect("invalid normal params");
        let long_dist = Normal::new(long_mean, long_mean * 0.2).expect("invalid normal params");

        let mut lengths: Vec<usize> = Vec::with_capacity(n);
        for _ in 0..n_short {
            let v: f64 = short_dist.sample(&mut rng);
            lengths.push(v.round().max(1.0) as usize);
        }
        for _ in 0..n_long {
            let v: f64 = long_dist.sample(&mut rng);
            lengths.push(v.round().max(1.0) as usize);
        }

        lengths.shuffle(&mut rng);
        lengths
    }
}
