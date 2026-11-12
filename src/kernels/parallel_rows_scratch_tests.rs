//! Exact partition, completion and failure ordering for bounded worker scratch.

use super::{MAX_ROW_WORKERS, RowParallelism};
use crate::{Result, invalid};

#[test]
fn bounded_update_failures_join_every_chunk_and_preserve_first_error() -> Result<()> {
    for workers in [2, 4, MAX_ROW_WORKERS] {
        let policy = RowParallelism::new(workers)?;
        // Eighteen units exercise every error slot with eighteen workers;
        // nineteen also cover a partial final task and unused error slots.
        for units in [MAX_ROW_WORKERS, MAX_ROW_WORKERS + 1] {
            let first_failure = units.div_ceil(workers);
            let mut first = vec![0_u8; units * 2];
            let mut second = vec![0_u8; units * 3];
            let mut third = vec![0_u8; units * 5];
            let failure = |base: usize| {
                if base > 0 {
                    Err(invalid(format!("rejected chunk {base}")))
                } else {
                    Ok(())
                }
            };

            let error = policy.update_chunks(&mut first, 2, |base, chunk| {
                chunk.fill(1);
                failure(base)
            });
            assert_eq!(
                error.map_err(|error| error.to_string()),
                Err(format!("rejected chunk {first_failure}"))
            );
            assert!(first.iter().all(|&value| value == 1));

            let error = policy.update_paired_chunks(&mut first, 2, &mut second, 3, |base, a, b| {
                assert_eq!(a.len() / 2, b.len() / 3);
                a.fill(2);
                b.fill(3);
                failure(base)
            });
            assert_eq!(
                error.map_err(|error| error.to_string()),
                Err(format!("rejected chunk {first_failure}"))
            );
            assert!(first.iter().all(|&value| value == 2));
            assert!(second.iter().all(|&value| value == 3));

            let error = policy.update_triple_chunks(
                &mut first,
                2,
                &mut second,
                3,
                &mut third,
                5,
                |base, a, b, c| {
                    assert_eq!(a.len() / 2, b.len() / 3);
                    assert_eq!(a.len() / 2, c.len() / 5);
                    a.fill(4);
                    b.fill(5);
                    c.fill(6);
                    failure(base)
                },
            );
            assert_eq!(
                error.map_err(|error| error.to_string()),
                Err(format!("rejected chunk {first_failure}"))
            );
            assert!(first.iter().all(|&value| value == 4));
            assert!(second.iter().all(|&value| value == 5));
            assert!(third.iter().all(|&value| value == 6));

            // Fresh successful work must not retain an earlier task's failure.
            policy.update_chunks(&mut first, 2, |_, chunk| {
                chunk.fill(7);
                Ok(())
            })?;
            policy.update_paired_chunks(&mut first, 2, &mut second, 3, |_, a, b| {
                a.fill(8);
                b.fill(9);
                Ok(())
            })?;
            policy.update_triple_chunks(
                &mut first,
                2,
                &mut second,
                3,
                &mut third,
                5,
                |_, a, b, c| {
                    a.fill(10);
                    b.fill(11);
                    c.fill(12);
                    Ok(())
                },
            )?;
            assert!(first.iter().all(|&value| value == 10));
            assert!(second.iter().all(|&value| value == 11));
            assert!(third.iter().all(|&value| value == 12));
        }
    }
    Ok(())
}
