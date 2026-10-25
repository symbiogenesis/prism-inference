//! Source-local numerical proof from test-double-float.cpp, in safe Rust.
//!
//! The upstream source calls no engine API. Its Windows ARM64 path exhausts
//! all binary32 patterns and compares float/double rounding. Its F16C-only
//! half-precision branch is outside the requested ARM64 target.
//!
//! C++ float-to-int8 conversion is undefined outside [-128,127]. Instead of
//! inventing a result for that cast, compare the rounded numbers themselves:
//! equality proves the original assertion wherever its cast is defined and
//! imposes a stronger, defined condition on every other finite input.
//! Upstream MIT attribution: tools/prism/upstream/LICENSE.

use serde_json::{Value, json};
use std::sync::atomic::{AtomicU32, Ordering};

use super::engine::Engine;
use super::{Result, invalid, result};

const ID: &str = "tests/test-double-float.cpp";
const PATTERNS: u64 = 1_u64 << 32;
const FINITE: u64 = 2 * 255 * (1 << 23);
// Each range owns one sign/exponent pair. Queueing ranges keeps workers busy
// across finite and nonfinite regions without synchronization in the hot loop.
const RANGE_BITS: u32 = 23;
const RANGES: u32 = 512;
const MAX_WORKERS: usize = 8;
const FINITE_INTERVALS: [(u32, u32); 2] = [(0, 0x7f7f_ffff), (0x8000_0000, 0xff7f_ffff)];
// round(127.5) is 128; round(-128.5) is -129. The negative endpoint
// differs because -128 is representable by the original signed-byte cast.
const UNDEFINED_CAST_INTERVALS: [(u32, u32); 2] =
    [(0x42ff_0000, 0x7f7f_ffff), (0xc300_8000, 0xff7f_ffff)];

fn rounded(value: f32) -> (f64, f64) {
    (f64::from(value).round(), f64::from(value.round()))
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Proof {
    visited: u64,
    finite: u64,
    undefined_casts: u64,
    failure: Option<(u32, f64, f64)>,
}

impl Proof {
    fn examine(&mut self, patterns: std::ops::RangeInclusive<u32>) {
        let (first, last) = patterns.into_inner();
        if first > last {
            return;
        }
        self.visited += u64::from(last) - u64::from(first) + 1;
        // A sign/exponent range has uniform finiteness. Count intervals once
        // instead of classifying and updating counters billions of
        // times. Every finite bit pattern still reaches both rounding calls
        // and the exact comparison; only nonfinite patterns have no predicate,
        // exactly as in the original proof.
        for (start, end) in UNDEFINED_CAST_INTERVALS {
            self.undefined_casts += intersection_count(first, last, start, end);
        }
        for (start, end) in FINITE_INTERVALS.into_iter().rev() {
            let start = first.max(start);
            let end = last.min(end);
            if start > end {
                continue;
            }
            self.finite += u64::from(end) - u64::from(start) + 1;
            for bits in (start..=end).rev() {
                let value = f32::from_bits(bits);
                let (double, single) = rounded(value);
                #[allow(
                    clippy::float_cmp,
                    reason = "The original proof requires exact rounding equality, not approximate error."
                )]
                if double != single && self.failure.is_none_or(|(previous, _, _)| bits > previous) {
                    self.failure = Some((bits, double, single));
                }
            }
        }
    }

    fn merge(&mut self, other: Self) {
        self.visited += other.visited;
        self.finite += other.finite;
        self.undefined_casts += other.undefined_casts;
        if let Some((bits, _, _)) = other.failure
            && self.failure.is_none_or(|(previous, _, _)| bits > previous)
        {
            self.failure = other.failure;
        }
    }
}

fn intersection_count(first: u32, last: u32, start: u32, end: u32) -> u64 {
    let lower = if first > start { first } else { start };
    let upper = if last < end { last } else { end };
    if lower > upper {
        0
    } else {
        u64::from(upper) - u64::from(lower) + 1
    }
}

fn prove(workers: usize, range_bits: u32, ranges: u32) -> Result<Proof> {
    let next = AtomicU32::new(0);
    std::thread::scope(|scope| {
        let handles = (0..workers)
            .map(|_| {
                scope.spawn(|| {
                    let mut proof = Proof::default();
                    loop {
                        let range = next.fetch_add(1, Ordering::Relaxed);
                        if range >= ranges {
                            return proof;
                        }
                        let start = (ranges - range - 1) << range_bits;
                        proof.examine(start..=start | ((1_u32 << range_bits) - 1));
                    }
                })
            })
            .collect::<Vec<_>>();
        let mut proof = Proof::default();
        for handle in handles {
            proof.merge(
                handle
                    .join()
                    .map_err(|_| invalid("binary32 proof worker panicked"))?,
            );
        }
        Ok(proof)
    })
}

pub(super) fn run(_engine: &Engine) -> Result<Value> {
    if std::env::var("GITHUB_ACTIONS").as_deref() != Ok("true")
        || std::env::var("GITHUB_REF").as_deref() != Ok("refs/heads/main")
    {
        return Err(invalid(
            "the exhaustive binary32 proof runs only in main-branch CI",
        ));
    }
    if !cfg!(all(target_os = "windows", target_arch = "aarch64")) {
        return Err(invalid(
            "this source port covers the requested Windows ARM64 path",
        ));
    }
    let workers = std::thread::available_parallelism()?.get().min(MAX_WORKERS);
    let proof = prove(workers, RANGE_BITS, RANGES)?;
    let passed = proof.visited == PATTERNS && proof.finite == FINITE && proof.failure.is_none();
    let failure = proof.failure.map(|(bits, double, single)| {
        json!({"InputBits":bits,"DoubleRounded":double,"FloatRounded":single})
    });
    Ok(result(ID, passed, json!({
        "SourceSha256":"1a967f5a89c959260d1884740ac0b2f25a8db9dcc96f4c6eed05fcdb35c9627b",
        "Scope":"source-local arithmetic proof; no upstream or Rust runtime engine API exists in this source",
        "VisitedPatterns":proof.visited,"ExpectedPatterns":PATTERNS,"FinitePredicates":proof.finite,"ExpectedFinitePredicates":FINITE,
        "NonfinitePatternsClassified":proof.visited-proof.finite,
        "CoverageAccounting":"every finite bit pattern is enumerated and compared; nonfinite intervals are classified by their IEEE exponent bits",
        "UndefinedOriginalCastInputs":proof.undefined_casts,"DefinedOriginalCastInputs":proof.finite-proof.undefined_casts,
        "Workers":workers,"Ranges":RANGES,
        "Comparison":"exact rounded-number equality; implies original int8 equality on its defined domain",
        "ExcludedBranch":"x86 F16C half-SiLU code does not apply to Windows ARM64",
        "FirstFailure":failure,
    }).to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(patterns: std::ops::RangeInclusive<u32>) -> Proof {
        let mut proof = Proof::default();
        for bits in patterns.rev() {
            proof.visited += 1;
            let value = f32::from_bits(bits);
            if !value.is_finite() {
                continue;
            }
            proof.finite += 1;
            let (double, single) = rounded(value);
            proof.undefined_casts += u64::from(!(-128.0..128.0).contains(&double));
            #[allow(
                clippy::float_cmp,
                reason = "This independent scalar oracle preserves the original exact comparison."
            )]
            if double != single && proof.failure.is_none() {
                proof.failure = Some((bits, double, single));
            }
        }
        proof
    }

    #[test]
    fn interval_accounting_matches_scalar_oracle_at_every_boundary_and_sampled_ranges() {
        for boundary in [
            0_u32,
            0x42ff_0000,
            0x7f80_0000,
            0x8000_0000,
            0xc300_8000,
            0xff80_0000,
            u32::MAX,
        ] {
            for radius in [0, 1, 127, 512] {
                let range = boundary.saturating_sub(radius)..=boundary.saturating_add(radius);
                let mut proof = Proof::default();
                proof.examine(range.clone());
                assert_eq!(
                    proof,
                    reference(range),
                    "boundary {boundary:08x}, radius {radius}"
                );
            }
        }
        let mut first = 0x1234_5678_u32;
        for _ in 0..512 {
            first = first.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let range = first..=first.saturating_add(255);
            let mut proof = Proof::default();
            proof.examine(range.clone());
            assert_eq!(proof, reference(range), "sample {first:08x}");
        }
        let mut empty = Proof::default();
        empty.examine(std::ops::RangeInclusive::new(1, 0));
        assert_eq!(empty, Proof::default());
    }

    #[test]
    fn finite_boundary_and_tie_rounding_witnesses() {
        for bits in [
            0,
            0x8000_0000,
            1,
            0x007f_ffff,
            0x0080_0000,
            0x3eff_ffff,
            0x3f00_0000,
            0x3f00_0001,
            0xbf00_0000,
            0x42ff_0000,
            0xc300_0000,
            0x7f7f_ffff,
            0xff7f_ffff,
        ] {
            let (double, single) = rounded(f32::from_bits(bits));
            assert_eq!(double.to_bits(), single.to_bits());
        }
        assert_eq!(FINITE, 4_278_190_080);
        assert_eq!(PATTERNS - FINITE, 16_777_216);
    }

    #[test]
    fn range_partition_and_reduction_preserve_exact_counts_and_first_failure() -> Result<()> {
        assert_eq!(u64::from(RANGES) << RANGE_BITS, PATTERNS);
        let parallel = prove(3, 7, 5)?;
        assert_eq!(parallel.visited, 640);
        assert_eq!(parallel.finite, 640);
        assert_eq!(parallel.undefined_casts, 0);
        assert!(parallel.failure.is_none());
        let mut combined = Proof::default();
        for bits in [
            0_u32,
            0x3eff_ff80,
            0x42fe_ff80,
            0x7f7f_ff80,
            0x7fff_ff80,
            0x8000_0000,
            0xff7f_ff80,
            0xffff_ff80,
        ] {
            let mut proof = Proof::default();
            proof.examine(bits..=bits + 127);
            assert_eq!(proof.visited, 128);
            assert_eq!(
                proof.finite,
                if matches!(bits, 0x7fff_ff80 | 0xffff_ff80) {
                    0
                } else {
                    128
                }
            );
            assert_eq!(
                proof.undefined_casts,
                if matches!(bits, 0x7f7f_ff80 | 0xff7f_ff80) {
                    128
                } else {
                    0
                }
            );
            assert!(proof.failure.is_none());
            combined.merge(proof);
        }
        assert_eq!(combined.visited, 1024);
        assert_eq!(combined.finite, 768);
        assert_eq!(combined.undefined_casts, 256);
        for bits in [200, 100, 300, 250] {
            combined.merge(Proof {
                failure: Some((bits, 1.0, 2.0)),
                ..Proof::default()
            });
        }
        assert_eq!(combined.failure, Some((300, 1.0, 2.0)));
        Ok(())
    }
}
