//! Initialized scratch reuse, bounded growth and fallible allocation controls.

use super::{MAX_STATE_BYTES, resize_attention_scores, resize_attention_scores_with};
use crate::{Result, invalid};

#[test]
fn attention_score_growth_preserves_old_storage_and_respects_context_ceiling() -> Result<()> {
    // Non-power-of-two context catches a final doubling past admission.
    for rows in [1, 4, 24, 48] {
        let mut scores = Vec::new();
        let mut reservations = 0;
        for tokens in 1..=67 {
            let old_len = scores.len();
            scores.fill(f32::NAN);
            resize_attention_scores_with(rows, tokens, 67, &mut scores, |scores, additional| {
                reservations += 1;
                scores.try_reserve_exact(additional)?;
                Ok(())
            })?;
            assert_eq!(scores.len(), rows * tokens);
            assert!(scores.capacity() <= rows * 67);
            assert!(scores[..old_len].iter().all(|value| value.is_nan()));
            assert!(scores[old_len..].iter().all(|value| value.to_bits() == 0));
        }
        assert!(reservations <= 8);
        let pointer = scores.as_ptr();
        let capacity = scores.capacity();
        resize_attention_scores(rows, 1, 67, &mut scores)?;
        resize_attention_scores(rows, 67, 67, &mut scores)?;
        assert_eq!(scores.as_ptr(), pointer);
        assert_eq!(scores.capacity(), capacity);
        assert!(scores[rows..].iter().all(|value| value.to_bits() == 0));
    }
    Ok(())
}

#[test]
fn attention_score_growth_rejection_preserves_storage_and_allows_retry() -> Result<()> {
    let mut scores = vec![-0.0, f32::from_bits(1), 7.0];
    scores.shrink_to_fit();
    let original: Vec<_> = scores.iter().map(|value| value.to_bits()).collect();
    let pointer = scores.as_ptr();
    let capacity = scores.capacity();
    let mut attempted = false;
    assert!(
        resize_attention_scores_with(4, 2, 3, &mut scores, |_, _| {
            attempted = true;
            Err(invalid("injected attention reservation refusal"))
        })
        .is_err()
    );
    assert!(attempted);
    assert_eq!(scores.as_ptr(), pointer);
    assert_eq!(scores.capacity(), capacity);
    assert!(scores.iter().map(|value| value.to_bits()).eq(original));
    resize_attention_scores(4, 2, 3, &mut scores)?;
    assert_eq!(scores.len(), 8);
    assert!(scores.capacity() <= 12);
    assert_eq!(scores[0].to_bits(), (-0.0_f32).to_bits());
    assert_eq!(scores[1].to_bits(), 1);
    assert_eq!(scores[2].to_bits(), 7.0_f32.to_bits());
    assert!(scores[3..].iter().all(|value| value.to_bits() == 0));
    Ok(())
}

#[test]
fn attention_score_growth_bounds_reject_before_allocation_or_mutation() {
    let mut scores = vec![-0.0, 7.0];
    for (rows, tokens, context) in [
        (0, 1, 1),
        (1, 0, 1),
        (1, 2, 1),
        (usize::MAX, 2, 2),
        (2, 1, usize::MAX),
        (1, MAX_STATE_BYTES / size_of::<f32>() + 1, usize::MAX),
    ] {
        let mut attempted = false;
        assert!(
            resize_attention_scores_with(rows, tokens, context, &mut scores, |_, _| {
                attempted = true;
                Err(invalid("reservation must not run for invalid bounds"))
            })
            .is_err()
        );
        assert!(!attempted);
        assert_eq!(scores.len(), 2);
        assert_eq!(scores[0].to_bits(), (-0.0_f32).to_bits());
        assert_eq!(scores[1].to_bits(), 7.0_f32.to_bits());
    }
}
