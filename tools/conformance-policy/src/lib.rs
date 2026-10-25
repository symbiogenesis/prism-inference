//! Offline policy for the local Prism conformance catalog and CI artifacts.
#![forbid(unsafe_code)]

/// A bounded policy result or a diagnostic on stderr.
pub type Result<T> = std::result::Result<T, String>;

/// Catalog identities, complete source reports and baseline gates.
pub mod prism_baseline;

/// Ordinary source partitioning and bounded revision-bound result merging.
pub mod prism_source;
