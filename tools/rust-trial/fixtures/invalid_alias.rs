//! Negative control: two overlapping exclusive references must be rejected.

#![deny(warnings, missing_docs)]
#![forbid(unsafe_code)]

/// An intentionally invalid alias retained across a second exclusive borrow.
pub fn overlapping_mutable_references(value: &mut u8) {
    let first = &mut *value;
    let second = &mut *value;
    *first = *second;
}
