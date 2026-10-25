//! Focused regression for the policy migrated from the Prism C++ overlay.

use bonsai_prism::{Layout, MAX_CACHE_BYTES, MAX_ENTRIES, MAX_TENSOR_BYTES, Pool};

fn layout(count: usize) -> Layout {
    Layout {
        context: 1,
        buffer_type: 2,
        cpu: true,
        f32: true,
        contiguous: true,
        dimensions: [1, i64::try_from(count).unwrap_or(i64::MAX), 1, 1],
        strides: [4, 4, count * 4, count * 4],
        byte_len: count * 4,
    }
}

fn payload(count: usize, bits: u32) -> Vec<u8> {
    bits.to_ne_bytes().repeat(count)
}

fn check_byte_identity() {
    let source = layout(5120);
    let bytes = payload(5120, 0x3f80_0000);
    let mut pool = Pool::new();
    assert!(pool.is_empty());
    assert_eq!(pool.find(&source, &bytes), None);
    assert!(pool.remember(&source, &bytes, 3));
    assert_eq!(pool.find(&source, &bytes), Some(3));
    assert_eq!(pool.shared_tensors(), 1);
    assert_eq!(pool.shared_bytes(), bytes.len());
    assert!(!pool.remember(&source, &bytes, 4));
    assert_eq!(pool.len(), 1);

    for changed in [
        Layout {
            context: 9,
            ..source
        },
        Layout {
            buffer_type: 9,
            ..source
        },
        Layout {
            strides: [4, 8, 40960, 40960],
            ..source
        },
        layout(5119),
    ] {
        assert_eq!(pool.find(&changed, &bytes), None);
    }
    let mut changed = bytes.clone();
    if let Some(last) = changed.last_mut() {
        *last ^= 1;
    }
    assert_eq!(pool.find(&source, &changed), None);

    for (bits, owner) in [(0, 4), (0x8000_0000, 5), (0x7fc0_0001, 6), (0x7fc0_0002, 7)] {
        let value = payload(5120, bits);
        assert_eq!(pool.find(&source, &value), None);
        assert!(pool.remember(&source, &value, owner));
        assert_eq!(pool.find(&source, &value), Some(owner));
    }
    assert_eq!(pool.find(&source, &payload(5120, 0x7fc0_0001)), Some(6));
    assert_eq!(pool.shared_tensors(), 6);
    assert_eq!(pool.shared_bytes(), 6 * bytes.len());
}

fn check_eligibility() {
    let source = layout(5120);
    assert!(source.eligible());
    assert!(Pool::eligible(&source));
    let mut pool = Pool::default();
    for rejected in [
        Layout {
            cpu: false,
            ..source
        },
        Layout {
            f32: false,
            ..source
        },
        Layout {
            contiguous: false,
            ..source
        },
        Layout {
            dimensions: [2, 5120, 1, 1],
            ..source
        },
        Layout {
            dimensions: [1, 0, 1, 1],
            ..source
        },
        Layout {
            dimensions: [1, -1, 1, 1],
            ..source
        },
        Layout {
            dimensions: [1, 5120, 2, 1],
            ..source
        },
        Layout {
            dimensions: [1, 5120, 1, 2],
            ..source
        },
        layout(20000),
    ] {
        assert!(!rejected.eligible());
        assert!(!pool.remember(&rejected, &payload(5120, 0), 3));
        assert_eq!(pool.find(&rejected, &payload(5120, 0)), None);
    }
    assert!(!pool.remember(&source, &payload(5119, 0), 3));
    assert!(!pool.remember(&source, &payload(5120, 0), 0));
    assert_eq!(pool.shared_tensors(), 0);
    assert_eq!(pool.shared_bytes(), 0);
    assert_eq!(pool.cached_bytes(), 0);
    assert!(pool.is_empty());
}

fn check_budgets() {
    let mut bounded = Pool::new();
    let large = layout(MAX_TENSOR_BYTES / 4);
    assert!(large.eligible());
    for bits in 0..17_u32 {
        let bytes = payload(MAX_TENSOR_BYTES / 4, bits);
        assert_eq!(bounded.find(&large, &bytes), None);
        assert_eq!(bounded.remember(&large, &bytes, 3), bits < 16);
    }
    assert_eq!(bounded.cached_bytes(), MAX_CACHE_BYTES);
    assert_eq!(
        bounded.find(&large, &payload(MAX_TENSOR_BYTES / 4, 16)),
        None
    );
    assert_eq!(
        bounded.find(&large, &payload(MAX_TENSOR_BYTES / 4, 0)),
        Some(3)
    );

    let mut bounded = Pool::new();
    let tiny = layout(1);
    for bits in 0..65_u32 {
        let bytes = payload(1, bits);
        assert_eq!(bounded.find(&tiny, &bytes), None);
        assert_eq!(bounded.remember(&tiny, &bytes, 3), bits < 64);
    }
    assert_eq!(bounded.len(), MAX_ENTRIES);
    assert_eq!(bounded.cached_bytes(), MAX_ENTRIES * 4);
    assert_eq!(bounded.find(&tiny, &payload(1, 64)), None);
    assert_eq!(bounded.find(&tiny, &payload(1, 0)), Some(3));
}

#[test]
fn pool_policy_preserves_existing_cpp_cases() {
    check_byte_identity();
    check_eligibility();
    check_budgets();
}
