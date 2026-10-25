//! Focused ownership and malformed-input contracts for batch splitting.
#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use bonsai_prism_runtime::model::batch_allocator::{BatchAllocator, Input};

type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

#[test]
fn token_positions_broadcast_into_owned_mrope_sections() -> TestResult {
    let mut allocator = BatchAllocator::new(4)?;
    let input = Input {
        tokens: Some(vec![1, 2]),
        positions: Some(vec![10, 12]),
        ..Input::default()
    };
    assert!(allocator.init(input, 3, 1, false, None)?);
    let micro = allocator.split_simple(2)?;
    let data = micro.data.as_ref().ok_or("missing token microbatch")?;
    assert_eq!(data.positions, [10, 12, 10, 12, 10, 12, 10, 12]);
    assert_eq!(data.tokens.as_deref(), Some([1, 2].as_slice()));
    allocator.reserve(3, 2)?;
    assert_eq!(data.tokens.as_deref(), Some([1, 2].as_slice()));
    assert_eq!(data.positions, [10, 12, 10, 12, 10, 12, 10, 12]);
    Ok(())
}

#[test]
fn coupled_memory_ranges_fill_positions_after_rejection() -> TestResult {
    let mut allocator = BatchAllocator::new(1)?;
    let input = Input {
        embedding_width: 1,
        embeddings: Some(vec![1.0, 2.0]),
        sequence_ids: Some(vec![vec![0, 1], vec![0, 1]]),
        ..Input::default()
    };
    let mut memory = BTreeMap::from([(0, (0, 4)), (1, (1, 4))]);
    assert!(!allocator.init(input.clone(), 0, 2, false, Some(&memory))?);
    assert!(allocator.batch().is_none());
    assert!(allocator.split_simple(1).is_err());
    memory.insert(1, (0, 4));
    assert!(allocator.init(input, 0, 2, false, Some(&memory))?);
    assert_eq!(allocator.batch().ok_or("missing batch")?.positions, [5, 6]);
    Ok(())
}

#[test]
fn malformed_extents_and_split_capacities_return_errors() -> TestResult {
    assert!(BatchAllocator::new(2).is_err());
    let mut allocator = BatchAllocator::new(1)?;
    assert!(
        allocator
            .init(
                Input {
                    tokens: Some(vec![0]),
                    positions: Some(vec![]),
                    ..Input::default()
                },
                1,
                1,
                false,
                None
            )
            .is_err()
    );
    assert!(
        allocator
            .init(
                Input {
                    tokens: Some(vec![0]),
                    sequence_ids: Some(vec![vec![]]),
                    ..Input::default()
                },
                1,
                1,
                false,
                None
            )
            .is_err()
    );
    assert!(
        allocator
            .init(
                Input {
                    embeddings: Some(vec![f32::NAN]),
                    embedding_width: 1,
                    ..Input::default()
                },
                0,
                1,
                false,
                None
            )
            .is_err()
    );
    assert!(allocator.init(
        Input {
            tokens: Some(vec![0]),
            ..Input::default()
        },
        1,
        1,
        false,
        None
    )?);
    assert!(allocator.split_simple(0).is_err());
    assert!(allocator.split_equal(2, false, 2).is_err());
    assert!(allocator.reserve(usize::MAX, 2).is_err());
    Ok(())
}
