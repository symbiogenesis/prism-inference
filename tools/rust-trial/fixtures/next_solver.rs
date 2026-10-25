//! Normalize a backend view inside a higher-ranked callback type.
//!
//! Adapted from the Rust project's higher-ranked associated type example:
//! <https://blog.rust-lang.org/2026/08/21/enabling-next-solver-on-nightly/>.

#![deny(warnings, missing_docs)]
#![forbid(unsafe_code)]

/// A backend whose view can depend on the callback's lifetime.
pub trait Backend {
    /// Borrowed output passed to a callback.
    type View<'a>;
}

/// A CPU backend lending logits slices.
pub struct Cpu;

impl Backend for Cpu {
    type View<'a> = &'a [f32];
}

/// A backend paired with a callback accepting every view lifetime.
pub trait SupportsView {}

impl<B: Backend> SupportsView for (B, for<'a> fn(B::View<'a>)) {}

fn require_view<T: SupportsView>() {}

/// Prove that a callback over CPU slices satisfies the generic backend bound.
pub fn validate_cpu_callback() {
    require_view::<(Cpu, for<'a> fn(&'a [f32]))>();
}
