use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};

use super::{MAX_ROW_WORKERS, RowParallelism, SiblingTask};
use crate::{Result, invalid};

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "Keep bounded storage, borrowed rows, deterministic failures and joined caller cancellation in one control."
)]
fn x1_sibling_tile_workers_preserve_borrows_order_and_cancellation() -> Result<()> {
    assert!(
        size_of::<[SiblingTask<'_, 32>; MAX_ROW_WORKERS + 2]>() + 1024
            <= crate::kernels::prefill::ROW_TASK_SCRATCH_BYTES
    );
    let parallelism = RowParallelism::new(4)?;
    let data = [vec![1_u8; 13], vec![2; 9], vec![3; 5]];
    let slices = data.each_ref().map(Vec::as_slice);
    let mut outputs: [Vec<Vec<f32>>; 3] = std::array::from_fn(|_| vec![Vec::new(); 3]);
    for (group, bytes) in outputs.iter_mut().zip(&data) {
        for output in group {
            output.reserve_exact(bytes.len());
        }
    }
    let caller = std::thread::current().id();
    let checks = Cell::new(0);
    let mut blocks = 0;
    let [a, b, c] = &mut outputs;
    parallelism.fill_tile_siblings_four_rows(
        &slices,
        1,
        &mut [a, b, c],
        &mut blocks,
        &|| {
            assert_eq!(std::thread::current().id(), caller);
            checks.set(checks.get() + 1);
            Ok(())
        },
        |_, _| Err(invalid("replay eight rows")),
        |_, _| Err(invalid("replay four rows")),
        |row, blocks| {
            let group = usize::from(row[0]) - 1;
            assert!(
                data[group]
                    .iter()
                    .any(|value| std::ptr::eq(value, row.as_ptr()))
            );
            *blocks += 1;
            let value = f32::from(row[0]);
            Ok([value, -value, value * 0.5])
        },
    )?;
    assert_eq!(blocks, 27);
    assert!(checks.get() > 0);
    for (group, values) in outputs.iter().enumerate() {
        let value = f32::from(data[group][0]);
        assert_eq!(values[0], vec![value; data[group].len()]);
        assert_eq!(values[1], vec![-value; data[group].len()]);
        assert_eq!(values[2], vec![value * 0.5; data[group].len()]);
    }
    // Clear through the owning projection interface in production. This direct
    // worker fixture retains storage and demonstrates sibling-major error order.
    for group in &mut outputs {
        for output in group {
            output.clear();
        }
    }
    let [a, b, c] = &mut outputs;
    let failure = parallelism.fill_tile_siblings_four_rows(
        &slices,
        1,
        &mut [a, b, c],
        &mut blocks,
        &|| Ok(()),
        |_, _| Err(invalid("replay eight rows")),
        |_, _| Err(invalid("replay four rows")),
        |row, _| -> Result<[f32; 3]> {
            Err(invalid(format!("sibling {}", usize::from(row[0]) - 1)))
        },
    );
    assert_eq!(
        failure.map_err(|error| error.to_string()),
        Err("sibling 0".into())
    );

    let started = AtomicBool::new(false);
    let cancelled = AtomicBool::new(false);
    let finished = AtomicBool::new(false);
    let [a, b, c] = &mut outputs;
    let failure = parallelism.fill_tile_siblings_four_rows(
        &slices,
        1,
        &mut [a, b, c],
        &mut blocks,
        &|| {
            assert_eq!(std::thread::current().id(), caller);
            if started.load(Ordering::Acquire) {
                cancelled.store(true, Ordering::Release);
                Err(invalid("caller cancelled siblings"))
            } else {
                Ok(())
            }
        },
        |_, _| Err(invalid("replay eight rows")),
        |_, _| Err(invalid("replay four rows")),
        |_, _| -> Result<[f32; 3]> {
            started.store(true, Ordering::Release);
            let wait = std::time::Instant::now();
            while !cancelled.load(Ordering::Acquire) {
                assert!(wait.elapsed() < std::time::Duration::from_secs(2));
                std::thread::yield_now();
            }
            finished.store(true, Ordering::Release);
            Err(invalid("provisional worker failure"))
        },
    );
    assert_eq!(
        failure.map_err(|error| error.to_string()),
        Err("caller cancelled siblings".into())
    );
    assert!(finished.load(Ordering::Acquire));
    Ok(())
}
