//! # seqpacker
//!
//! High-performance sequence packing for LLM training.
//!
//! Seqpacker solves the bin-packing problem for variable-length sequences:
//! given a set of sequences and a maximum pack length, pack them into the
//! fewest bins possible while minimizing padding waste.
//!
//! ## Quick Start
//!
//! ```
//! use seqpacker::{Packer, PackStrategy, Sequence};
//!
//! let packer = Packer::new(2048)
//!     .with_strategy(PackStrategy::FirstFitDecreasing);
//!
//! let sequences = vec![
//!     Sequence::new(0, 500),
//!     Sequence::new(1, 600),
//!     Sequence::new(2, 400),
//! ];
//!
//! let result = packer.pack(sequences).unwrap();
//! println!("Efficiency: {:.2}%", result.metrics.efficiency * 100.0);
//! ```
//!
//! ## Available Algorithms
//!
//! | Algorithm | Time | Approx. Ratio | Best For |
//! |-----------|------|--------------|----------|
//! | NextFit | O(n) | 2.0 | Memory-constrained streaming |
//! | FirstFit | O(n log B) | 1.7 | Online baseline |
//! | BestFit | O(n log B) | 1.7 | Tighter packing |
//! | WorstFit | O(n log B) | 2.0 | Even distribution |
//! | FirstFitDecreasing | O(n log n) | 1.22 | Default offline (recommended) |
//! | BestFitDecreasing | O(n log n) | 1.22 | Tightest offline packing |
//! | FirstFitShuffle | O(n log n) | ~1.3 | Training randomness |
//! | ModifiedFirstFitDecreasing | O(n log n) | 1.18 | Mixed-size distributions |
//! | Harmonic-K | O(n) | ~1.69 | Bounded-space online |

#![forbid(unsafe_code)]

pub mod algorithms;
pub mod dev;
pub mod engine;
pub mod error;
pub mod metrics;
pub mod pack;
pub mod packer;
pub mod placement;
pub mod sequence;
pub mod strategy;
pub mod stream;
pub mod validation;

// Re-export key types for ergonomic imports: `use seqpacker::{Packer, ...};`
pub use error::PackError;
pub use metrics::PackMetrics;
pub use pack::Pack;
pub use packer::{PackResult, Packer, PackerConfig};
pub use sequence::Sequence;
pub use strategy::PackStrategy;
pub use stream::{StreamPacker, StreamStrategy};
