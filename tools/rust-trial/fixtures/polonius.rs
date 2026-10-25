//! Conditional cache borrowing accepted by Polonius Alpha.
//!
//! Adapted to logits storage from the Rust project's conditional map example:
//! <https://blog.rust-lang.org/2026/08/04/enabling-polonius-alpha-on-nightly/>.

#![deny(warnings, missing_docs)]
#![forbid(unsafe_code)]

/// Borrow cached logits, allocating storage only when the cache is empty.
pub fn cached_or_refill(cache: &mut Option<Vec<f32>>, vocabulary_size: usize) -> &mut [f32] {
    match cache.as_mut() {
        Some(logits) => logits.as_mut_slice(),
        None => cache.insert(vec![0.0; vocabulary_size]).as_mut_slice(),
    }
}
