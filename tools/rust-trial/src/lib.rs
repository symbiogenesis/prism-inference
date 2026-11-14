//! Isolated borrowed-logits experiment for comparing Rust compilers.
//!
//! The numerical function is deliberately synthetic: this demonstrates safe
//! output ownership and storage reuse, not inference performance or model parity.
//! No production decoder or checkpoint contract depends on this crate.

#![forbid(unsafe_code)]

/// A source whose output may borrow its reusable storage until the next call.
pub trait LogitsSource {
    /// A view tied to the exclusive borrow of the source.
    type Output<'a>: AsRef<[f32]>
    where
        Self: 'a;

    /// Evaluate a token and lend the output to the caller.
    fn evaluate(&mut self, token: u16) -> Self::Output<'_>;
}

/// A tiny deterministic source with one reusable logits buffer.
///
/// Repeated tokens reuse the cached values. A different token or explicit
/// invalidation recomputes values in the same storage. The caller must finish
/// using a borrowed result before evaluating or invalidating the source again.
#[derive(Debug)]
pub struct LogitsCache {
    logits: Vec<f32>,
    cached_token: Option<u16>,
    refills: u64,
}

impl LogitsCache {
    /// Allocate output storage once for the chosen vocabulary size.
    #[must_use]
    pub fn new(vocabulary_size: u16) -> Self {
        Self {
            logits: vec![0.0; usize::from(vocabulary_size)],
            cached_token: None,
            refills: 0,
        }
    }

    /// Forget the cached token while retaining the output allocation.
    pub const fn invalidate(&mut self) {
        self.cached_token = None;
    }

    /// Number of cache refills, saturating at the diagnostic counter's limit.
    #[must_use]
    pub const fn refills(&self) -> u64 {
        self.refills
    }

    /// Capacity of the reusable output buffer in logits.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.logits.capacity()
    }
}

impl LogitsSource for LogitsCache {
    type Output<'a> = &'a [f32];

    fn evaluate(&mut self, token: u16) -> Self::Output<'_> {
        if self.cached_token == Some(token) {
            return &self.logits;
        }
        for (logit, index) in self.logits.iter_mut().zip(0_u16..) {
            *logit = f32::from(index).mul_add(0.5, f32::from(token));
        }
        self.cached_token = Some(token);
        self.refills = self.refills.saturating_add(1);
        &self.logits
    }
}

/// Higher-ranked associated-view projection supported by the next trait solver.
///
/// Rustdoc compiles this example with the selected compiler's default solver,
/// exercising the same projection as the ordinary Cargo compiler gate.
///
/// ```
/// use bonsai_rust_trial::{LogitsCache, SupportsLogitsView};
///
/// fn first(values: &[f32]) -> Option<f32> {
///     values.first().copied()
/// }
///
/// let read: for<'a> fn(&'a [f32]) -> Option<f32> = first;
/// let mut source = (LogitsCache::new(4), read);
/// assert_eq!(source.apply(5), Some(5.0));
/// assert_eq!(source.apply(7), Some(7.0));
/// ```
pub trait SupportsLogitsView {
    /// Evaluate a token through a higher-ranked borrowed-output function pointer.
    fn apply(&mut self, token: u16) -> Option<f32>;
}

impl<S: LogitsSource> SupportsLogitsView for (S, for<'a> fn(S::Output<'a>) -> Option<f32>) {
    fn apply(&mut self, token: u16) -> Option<f32> {
        let view = self.0.evaluate(token);
        (self.1)(view)
    }
}

/// Visit a logits view borrowed for this call, including from a borrowed source.
///
/// The consumer may return the view; its borrow then keeps the source from being
/// evaluated or invalidated until the last use of the returned value.
///
/// ```compile_fail,E0499
/// use bonsai_rust_trial::{LogitsCache, with_logits};
///
/// let mut source = LogitsCache::new(4);
/// let view = with_logits(&mut source, 1, |output| output);
/// source.invalidate(); // The returned view still borrows the source.
/// assert_eq!(view.len(), 4);
/// ```
pub fn with_logits<'a, S: LogitsSource, R>(
    source: &'a mut S,
    token: u16,
    consumer: impl FnOnce(S::Output<'a>) -> R,
) -> R {
    consumer(source.evaluate(token))
}

/// Consume a borrowed output through a generic lending interface.
#[must_use]
pub fn maximum_logit<S: LogitsSource>(source: &mut S, token: u16) -> Option<f32> {
    with_logits(source, token, |output| {
        output.as_ref().iter().copied().reduce(f32::max)
    })
}

#[cfg(test)]
mod tests {
    use super::{LogitsCache, LogitsSource, SupportsLogitsView, maximum_logit, with_logits};

    fn owned_reference(token: u16, vocabulary_size: u16) -> Vec<f32> {
        (0..vocabulary_size)
            .map(|index| f32::from(index).mul_add(0.5, f32::from(token)))
            .collect()
    }

    fn slice_max(values: &[f32]) -> Option<f32> {
        values.iter().copied().reduce(f32::max)
    }

    #[test]
    fn borrowed_outputs_match_reference_and_invalidate() {
        let mut cache = LogitsCache::new(8);
        assert_eq!(cache.evaluate(7), owned_reference(7, 8));
        assert_eq!(cache.evaluate(7), owned_reference(7, 8));
        assert_eq!(cache.refills(), 1);

        cache.invalidate();
        assert_eq!(cache.evaluate(7), owned_reference(7, 8));
        assert_eq!(cache.refills(), 2);
        assert_eq!(cache.evaluate(11), owned_reference(11, 8));
        assert_eq!(cache.refills(), 3);
        assert_eq!(maximum_logit(&mut cache, 11), Some(14.5));
        assert_eq!(with_logits(&mut cache, 11, <[f32]>::len), 8);
        let reduce: for<'a> fn(&'a [f32]) -> Option<f32> = slice_max;
        let mut pair = (cache, reduce);
        assert_eq!(pair.apply(11), Some(14.5));
        assert_eq!(maximum_logit(&mut LogitsCache::new(0), 11), None);
    }

    #[test]
    fn repeated_outputs_reuse_storage() {
        let mut cache = LogitsCache::new(8);
        let capacity = cache.capacity();
        let address = cache.evaluate(7).as_ptr();
        assert_eq!(cache.evaluate(7).as_ptr(), address);
        assert_eq!(cache.evaluate(11).as_ptr(), address);
        cache.invalidate();
        assert_eq!(cache.evaluate(11).as_ptr(), address);
        assert_eq!(cache.capacity(), capacity);
        assert_eq!(cache.refills(), 3);
    }

    struct BorrowedLogits<'a>(&'a [f32]);

    impl LogitsSource for BorrowedLogits<'_> {
        type Output<'a>
            = &'a [f32]
        where
            Self: 'a;

        fn evaluate(&mut self, _token: u16) -> Self::Output<'_> {
            self.0
        }
    }

    #[test]
    fn consumer_can_return_a_view_from_a_nonstatic_source() {
        let values = [4.0, -1.0, 2.0];
        let mut source = BorrowedLogits(&values);
        let view = with_logits(&mut source, 0, |output| output);
        assert_eq!(view, values);
        assert_eq!(view.as_ptr(), values.as_ptr());
        assert_eq!(maximum_logit(&mut source, 0), Some(4.0));
    }
}
