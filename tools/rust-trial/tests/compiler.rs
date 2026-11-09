//! Behavioral gates for the compiler analyses required by the pinned nightly.
//!
//! Deliberately use the inherited compiler configuration, without opt-in flags:
//! disabling either analysis must fail compilation before any test can run.
#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::fmt::Debug;

use bonsai_rust_trial::{LogitsCache, SupportsLogitsView, with_logits};

// Flow-sensitive lending: the first borrow escapes only on the occupied path.
// https://blog.rust-lang.org/2026/08/04/enabling-polonius-alpha-on-nightly/
fn existing_or_default(values: &mut HashMap<u8, String>, key: u8) -> &mut String {
    if let Some(value) = values.get_mut(&key) {
        return value;
    }
    values.entry(key).or_default()
}

#[test]
fn polonius_lends_existing_or_new_storage() {
    let mut values = HashMap::from([(1, String::from("existing"))]);
    existing_or_default(&mut values, 1).push('!');
    existing_or_default(&mut values, 2).push_str("new");

    assert_eq!(values.len(), 2);
    assert_eq!(values.get(&1).map(String::as_str), Some("existing!"));
    assert_eq!(values.get(&2).map(String::as_str), Some("new"));
}

struct SlotCache {
    slots: [(u64, Vec<u8>); 2],
}

impl SlotCache {
    fn get_or_insert(&mut self, key: u64, fill: &[u8]) -> &[u8] {
        for (slot_key, value) in &mut self.slots {
            if *slot_key == key {
                return value.as_slice();
            }
        }
        self.slots[0] = (key, fill.to_vec());
        self.slots[0].1.as_slice()
    }
}

#[test]
fn polonius_lends_from_mutable_slice_loop_with_fallback_mutation() {
    let mut cache = SlotCache {
        slots: [(1, b"first".to_vec()), (2, b"second".to_vec())],
    };
    assert_eq!(cache.get_or_insert(2, b"unused"), b"second");
    assert_eq!(cache.get_or_insert(3, b"third"), b"third");
    assert_eq!(cache.get_or_insert(3, b"unused"), b"third");
}

// The recursive opaque return is normalized inside its defining function.
// https://blog.rust-lang.org/2026/08/21/enabling-next-solver-on-nightly/
fn recursive_opaque(recursive: bool) -> impl Sized + Debug + PartialEq<u8> {
    if recursive {
        recursive_opaque(false) + 1_u8
    } else {
        0_u8
    }
}

#[test]
fn next_solver_normalizes_recursive_opaque_return() {
    assert_eq!(recursive_opaque(false), 0_u8);
    assert_eq!(recursive_opaque(true), 1_u8);
}

const fn first_logit(values: &[f32]) -> Option<f32> {
    values.first().copied()
}

#[test]
fn next_solver_normalizes_higher_ranked_logits_views() {
    let mut cache = LogitsCache::new(4);
    assert_eq!(with_logits(&mut cache, 5, <[f32]>::len), 4);
    let head: for<'a> fn(&'a [f32]) -> Option<f32> = first_logit;
    let mut pair = (cache, head);
    assert_eq!(pair.apply(5), Some(5.0));
}
