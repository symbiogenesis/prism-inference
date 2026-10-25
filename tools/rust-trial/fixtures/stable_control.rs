//! Positive control: stable Rust can refill and then borrow a logits cache.

#![deny(warnings, missing_docs)]
#![forbid(unsafe_code)]

/// Refill before creating the returned borrow, using a stable-compatible shape.
pub fn cached_or_refill(cache: &mut Option<Vec<f32>>, vocabulary_size: usize) -> &mut [f32] {
    cache
        .get_or_insert_with(|| vec![0.0; vocabulary_size])
        .as_mut_slice()
}

/// Consume a generic borrowed or owned logits view using a stable trait bound.
pub fn view_length<V: AsRef<[f32]>>(view: V) -> usize {
    view.as_ref().len()
}
