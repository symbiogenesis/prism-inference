//! Bounded 256-channel rotated scalar KV consumption.
//!
//! Three-bit codes occupy 96 bytes and four-bit codes occupy 128 bytes. Codes
//! are contiguous, least-significant-bit first; record scales are supplied by
//! callers separately. Consumers decode only four coordinates at a time and
//! never construct an expanded cache. Both Snapdragon X1 and X2 use this same
//! runtime-detected 128-bit NEON path; no optional ISA or speed claim is assumed.

const DIMENSION: usize = 256;

/// An immutable finite query and codebook checked once for a packed-key scan.
///
/// The borrows prevent mutation of either input while the proof is in use.
/// Packed-row extents and finite dot results remain checked on every call.
pub struct PreparedQuery<'a> {
    bits: u8,
    centroids: &'a [f32; 16],
    query: &'a [f32; DIMENSION],
    #[cfg(target_arch = "aarch64")]
    neon: bool,
}

impl<'a> PreparedQuery<'a> {
    /// Checks a query and codebook without allocating or copying either row.
    ///
    /// Returns `None` for unsupported precision or nonfinite input. Optional
    /// native dispatch is selected once and retained only for this borrow.
    #[must_use]
    pub fn new(bits: u8, centroids: &'a [f32; 16], query: &'a [f32; DIMENSION]) -> Option<Self> {
        if !matches!(bits, 3 | 4)
            || !centroids.iter().all(|value| value.is_finite())
            || !query.iter().all(|value| value.is_finite())
        {
            return None;
        }
        Some(Self {
            bits,
            centroids,
            query,
            #[cfg(target_arch = "aarch64")]
            neon: std::arch::is_aarch64_feature_detected!("neon"),
        })
    }

    /// Computes one code-only dot with the ordinary four-lane arithmetic.
    ///
    /// Returns `None` for an incorrect code extent or nonfinite binary32 result.
    /// The current packed bytes are consumed on every call; pointer identity
    /// never permits stale row reuse.
    #[must_use]
    pub fn dot(&self, packed: &[u8]) -> Option<f32> {
        if packed.len() != DIMENSION * usize::from(self.bits) / 8 {
            return None;
        }
        #[cfg(target_arch = "aarch64")]
        if self.neon {
            return finite_binary32(native::dot(packed, self.bits, self.centroids, self.query));
        }
        finite_binary32(portable_dot(packed, self.bits, self.centroids, self.query))
    }
}

/// An immutable group of `G` finite queries and shared codebook checked once for
/// a grouped-query packed-key scan.
///
/// Unpacks each 4-bit or 3-bit packed KV code block once per group of `G` heads
/// while preserving identical per-lane binary64 multiply/add rounding order.
pub struct PreparedQueryGroup<'a, const G: usize> {
    bits: u8,
    centroids: &'a [f32; 16],
    queries: [&'a [f32; DIMENSION]; G],
    #[cfg(target_arch = "aarch64")]
    neon: bool,
}

impl<'a, const G: usize> PreparedQueryGroup<'a, G> {
    /// Checks `G` query rows and shared centroids without allocating or copying.
    #[must_use]
    pub fn new(
        bits: u8,
        centroids: &'a [f32; 16],
        queries: [&'a [f32; DIMENSION]; G],
    ) -> Option<Self> {
        if G == 0
            || !matches!(bits, 3 | 4)
            || !centroids.iter().all(|value| value.is_finite())
            || queries
                .iter()
                .any(|query| !query.iter().all(|value| value.is_finite()))
        {
            return None;
        }
        Some(Self {
            bits,
            centroids,
            queries,
            #[cfg(target_arch = "aarch64")]
            neon: std::arch::is_aarch64_feature_detected!("neon"),
        })
    }

    /// Computes code-only dots for all `G` query heads from a single key row scan.
    #[must_use]
    pub fn dot(&self, packed: &[u8]) -> [Option<f32>; G] {
        if packed.len() != DIMENSION * usize::from(self.bits) / 8 {
            return [None; G];
        }
        #[cfg(target_arch = "aarch64")]
        if self.neon {
            return native::dot_group(packed, self.bits, self.centroids, &self.queries)
                .map(finite_binary32);
        }
        portable_dot_group(packed, self.bits, self.centroids, &self.queries).map(finite_binary32)
    }

    /// Computes code-only dots for `out.len()` contiguous key rows (`N * row_bytes`).
    ///
    /// Returns `false` if `packed_rows.len()` mismatches `out.len() * row_bytes` or
    /// any computed dot overflows finite binary32.
    #[must_use]
    pub fn score_rows(&self, packed_rows: &[u8], out: &mut [[f32; G]]) -> bool {
        let Some(expected_bytes) = out
            .len()
            .checked_mul(DIMENSION * usize::from(self.bits) / 8)
        else {
            return false;
        };
        if packed_rows.len() != expected_bytes {
            return false;
        }
        if out.is_empty() {
            return true;
        }
        #[cfg(target_arch = "aarch64")]
        if self.neon {
            return native::score_rows_group(
                packed_rows,
                self.bits,
                self.centroids,
                &self.queries,
                out,
            );
        }
        portable_score_rows_group(packed_rows, self.bits, self.centroids, &self.queries, out)
    }
}

/// An immutable pair of `G`-head query groups sharing a codebook for 2-token
/// batched packed-key scans.
///
/// Unpacks each 4-bit or 3-bit packed KV code block once across both token groups
/// (`2 * G` heads) while preserving identical per-lane binary64 rounding order.
pub struct PreparedQueryPair<'a, const G: usize> {
    bits: u8,
    centroids: &'a [f32; 16],
    queries: [[&'a [f32; DIMENSION]; G]; 2],
    #[cfg(target_arch = "aarch64")]
    neon: bool,
}

impl<'a, const G: usize> PreparedQueryPair<'a, G> {
    /// Checks two `G`-head query groups and shared centroids without copying.
    #[must_use]
    pub fn new(
        bits: u8,
        centroids: &'a [f32; 16],
        first: [&'a [f32; DIMENSION]; G],
        second: [&'a [f32; DIMENSION]; G],
    ) -> Option<Self> {
        if G == 0
            || !matches!(bits, 3 | 4)
            || !centroids.iter().all(|value| value.is_finite())
            || first
                .iter()
                .chain(second.iter())
                .any(|query| !query.iter().all(|value| value.is_finite()))
        {
            return None;
        }
        Some(Self {
            bits,
            centroids,
            queries: [first, second],
            #[cfg(target_arch = "aarch64")]
            neon: std::arch::is_aarch64_feature_detected!("neon"),
        })
    }

    /// Computes code-only dots for both token groups from a single key row scan.
    #[must_use]
    pub fn dot_both(&self, packed: &[u8]) -> ([Option<f32>; G], [Option<f32>; G]) {
        if packed.len() != DIMENSION * usize::from(self.bits) / 8 {
            return ([None; G], [None; G]);
        }
        #[cfg(target_arch = "aarch64")]
        if self.neon {
            let (first, second) = native::dot_group_pair(
                packed,
                self.bits,
                self.centroids,
                &self.queries[0],
                &self.queries[1],
            );
            return (first.map(finite_binary32), second.map(finite_binary32));
        }
        let (first, second) = portable_dot_group_pair(
            packed,
            self.bits,
            self.centroids,
            &self.queries[0],
            &self.queries[1],
        );
        (first.map(finite_binary32), second.map(finite_binary32))
    }

    /// Computes code-only dots for the second token group from one key row scan.
    #[must_use]
    pub fn dot_second(&self, packed: &[u8]) -> [Option<f32>; G] {
        if packed.len() != DIMENSION * usize::from(self.bits) / 8 {
            return [None; G];
        }
        #[cfg(target_arch = "aarch64")]
        if self.neon {
            return native::dot_group(packed, self.bits, self.centroids, &self.queries[1])
                .map(finite_binary32);
        }
        portable_dot_group(packed, self.bits, self.centroids, &self.queries[1]).map(finite_binary32)
    }

    /// Computes code-only dots for `out0.len()` contiguous key rows (`N * row_bytes`)
    /// across both `G`-head token groups.
    ///
    /// Returns `false` if slice lengths mismatch or any dot overflows finite binary32.
    #[must_use]
    pub fn score_rows(
        &self,
        packed_rows: &[u8],
        out0: &mut [[f32; G]],
        out1: &mut [[f32; G]],
    ) -> bool {
        let Some(expected_bytes) = out0
            .len()
            .checked_mul(DIMENSION * usize::from(self.bits) / 8)
        else {
            return false;
        };
        if out0.len() != out1.len() || packed_rows.len() != expected_bytes {
            return false;
        }
        if out0.is_empty() {
            return true;
        }
        #[cfg(target_arch = "aarch64")]
        if self.neon {
            return native::score_rows_pair(
                packed_rows,
                self.bits,
                self.centroids,
                &self.queries[0],
                &self.queries[1],
                out0,
                out1,
            );
        }
        portable_score_rows_pair(
            packed_rows,
            self.bits,
            self.centroids,
            &self.queries[0],
            &self.queries[1],
            out0,
            out1,
        )
    }
}

/// Computes a dot against one code-only rotated KV row, excluding its scale.
///
/// Rejects unsupported widths, incorrect byte lengths, nonfinite centroids or
/// query coordinates, and a result outside finite binary32. Four binary64 lane
/// sums consume indices modulo four; the final reduction is left-associated.
/// Portable and NEON implementations use the same separate multiply/add order.
#[must_use]
pub fn dot(
    packed: &[u8],
    bits: u8,
    centroids: &[f32; 16],
    query: &[f32; DIMENSION],
) -> Option<f32> {
    PreparedQuery::new(bits, centroids, query)?.dot(packed)
}

/// An exclusively owned running value sum for one packed-attention scan.
///
/// The codebook and native dispatch are checked once. Each row retains its
/// extent, scale and finite-result checks, with separate binary32 multiply/add
/// rounding. A failed append permanently prevents publication; no partially
/// accumulated or nonfinite row can be retrieved through this safe API.
pub struct ValueAccumulator<'a> {
    bits: u8,
    centroids: &'a [f32; 16],
    output: [f32; DIMENSION],
    failed: bool,
    #[cfg(target_arch = "aarch64")]
    neon: bool,
}

impl<'a> ValueAccumulator<'a> {
    /// Starts a zero sum, rejecting unsupported precision or nonfinite centroids.
    #[must_use]
    pub fn new(bits: u8, centroids: &'a [f32; 16]) -> Option<Self> {
        if !matches!(bits, 3 | 4) || !centroids.iter().all(|value| value.is_finite()) {
            return None;
        }
        Some(Self {
            bits,
            centroids,
            output: [0.0; DIMENSION],
            failed: false,
            #[cfg(target_arch = "aarch64")]
            neon: std::arch::is_aarch64_feature_detected!("neon"),
        })
    }

    /// Adds one code-only row without copying or rescanning the previous sum.
    ///
    /// Returns false on invalid input, overflow or any earlier failed append.
    /// Failure makes [`Self::finish`] return `None`, including after later valid
    /// appends. The private sum cannot escape while it is being accumulated.
    #[must_use]
    pub fn accumulate(&mut self, packed: &[u8], scale: f32) -> bool {
        if self.failed
            || packed.len() != DIMENSION * usize::from(self.bits) / 8
            || !scale.is_finite()
        {
            self.failed = true;
            return false;
        }
        #[cfg(target_arch = "aarch64")]
        let finite = if self.neon {
            native::accumulate(packed, self.bits, self.centroids, scale, &mut self.output)
        } else {
            portable_accumulate(packed, self.bits, self.centroids, scale, &mut self.output)
        };
        #[cfg(not(target_arch = "aarch64"))]
        let finite =
            portable_accumulate(packed, self.bits, self.centroids, scale, &mut self.output);
        self.failed = !finite;
        !self.failed
    }

    /// Publishes the completed sum only if every append succeeded.
    #[must_use]
    pub const fn finish(self) -> Option<[f32; DIMENSION]> {
        if self.failed { None } else { Some(self.output) }
    }
}

/// Exclusive running value sums for `G` grouped query heads sharing one packed KV head.
///
/// Unpacks each value row once and updates all `G` independent binary32 accumulators
/// with exact per-head isolation and failure atomicity.
pub struct ValueGroupAccumulator<'a, const G: usize> {
    bits: u8,
    centroids: &'a [f32; 16],
    outputs: [[f32; DIMENSION]; G],
    failed: [bool; G],
    #[cfg(target_arch = "aarch64")]
    neon: bool,
}

impl<'a, const G: usize> ValueGroupAccumulator<'a, G> {
    /// Starts zeroed accumulators for `G` heads, rejecting unsupported width or nonfinite codebook.
    #[must_use]
    pub fn new(bits: u8, centroids: &'a [f32; 16]) -> Option<Self> {
        if G == 0 || !matches!(bits, 3 | 4) || !centroids.iter().all(|value| value.is_finite()) {
            return None;
        }
        Some(Self {
            bits,
            centroids,
            outputs: [[0.0; DIMENSION]; G],
            failed: [false; G],
            #[cfg(target_arch = "aarch64")]
            neon: std::arch::is_aarch64_feature_detected!("neon"),
        })
    }

    /// Accumulates one packed value row across all `G` heads with per-head scales.
    #[must_use]
    pub fn accumulate(&mut self, packed: &[u8], scales: [f32; G]) -> [bool; G] {
        if packed.len() != DIMENSION * usize::from(self.bits) / 8 {
            self.failed = [true; G];
            return [false; G];
        }
        let mut active = [false; G];
        let mut any_active = false;
        for (head, (&scale, failed)) in scales.iter().zip(self.failed.iter_mut()).enumerate() {
            if !*failed && scale.is_finite() {
                active[head] = true;
                any_active = true;
            } else {
                *failed = true;
            }
        }
        if !any_active {
            return [false; G];
        }
        #[cfg(target_arch = "aarch64")]
        let ok = if self.neon {
            native::accumulate_group(
                packed,
                self.bits,
                self.centroids,
                scales,
                active,
                &mut self.outputs,
            )
        } else {
            portable_accumulate_group(
                packed,
                self.bits,
                self.centroids,
                scales,
                active,
                &mut self.outputs,
            )
        };
        #[cfg(not(target_arch = "aarch64"))]
        let ok = portable_accumulate_group(
            packed,
            self.bits,
            self.centroids,
            scales,
            active,
            &mut self.outputs,
        );
        for (head, &row_ok) in ok.iter().enumerate() {
            if !active[head] || !row_ok {
                self.failed[head] = true;
            }
        }
        std::array::from_fn(|head| !self.failed[head])
    }

    /// Accumulates `scales.len()` contiguous packed value rows across all `G` heads.
    ///
    /// Validates slice lengths, prior failure state, and scale finiteness upfront.
    /// Returns `false` and poisons all heads if any input is invalid or any output
    /// becomes nonfinite; otherwise preserves exact per-row binary32 rounding.
    #[must_use]
    pub fn accumulate_rows(&mut self, packed_rows: &[u8], scales: &[[f32; G]]) -> bool {
        let Some(expected_bytes) = scales
            .len()
            .checked_mul(DIMENSION * usize::from(self.bits) / 8)
        else {
            self.failed = [true; G];
            return false;
        };
        if packed_rows.len() != expected_bytes
            || self.failed.iter().any(|&failed| failed)
            || scales
                .iter()
                .any(|row| row.iter().any(|scale| !scale.is_finite()))
        {
            self.failed = [true; G];
            return false;
        }
        if scales.is_empty() {
            return true;
        }
        #[cfg(target_arch = "aarch64")]
        let finite = if self.neon {
            native::accumulate_rows_group(
                packed_rows,
                self.bits,
                self.centroids,
                scales,
                &mut self.outputs,
            )
        } else {
            portable_accumulate_rows_group(
                packed_rows,
                self.bits,
                self.centroids,
                scales,
                &mut self.outputs,
            )
        };
        #[cfg(not(target_arch = "aarch64"))]
        let finite = portable_accumulate_rows_group(
            packed_rows,
            self.bits,
            self.centroids,
            scales,
            &mut self.outputs,
        );
        if !finite {
            self.failed = [true; G];
        }
        finite
    }

    /// Publishes per-head sums for every head whose appends remained valid and finite.
    #[must_use]
    pub fn finish(self) -> [Option<[f32; DIMENSION]>; G] {
        std::array::from_fn(|head| {
            if self.failed[head] {
                None
            } else {
                Some(self.outputs[head])
            }
        })
    }
}

/// Exclusive running value sums for two consecutive tokens sharing `G` heads per KV head.
///
/// Unpacks each shared value row once across both tokens (`2 * G` heads) with exact
/// per-head isolation and failure atomicity.
pub struct ValueGroupPairAccumulator<'a, const G: usize> {
    bits: u8,
    centroids: &'a [f32; 16],
    outputs: [[[f32; DIMENSION]; G]; 2],
    failed: [[bool; G]; 2],
    #[cfg(target_arch = "aarch64")]
    neon: bool,
}

impl<'a, const G: usize> ValueGroupPairAccumulator<'a, G> {
    /// Starts zeroed accumulators for two token groups of `G` heads each.
    #[must_use]
    pub fn new(bits: u8, centroids: &'a [f32; 16]) -> Option<Self> {
        if G == 0 || !matches!(bits, 3 | 4) || !centroids.iter().all(|value| value.is_finite()) {
            return None;
        }
        Some(Self {
            bits,
            centroids,
            outputs: [[[0.0; DIMENSION]; G]; 2],
            failed: [[false; G]; 2],
            #[cfg(target_arch = "aarch64")]
            neon: std::arch::is_aarch64_feature_detected!("neon"),
        })
    }

    /// Accumulates one shared packed value row across both token groups.
    #[must_use]
    pub fn accumulate_both(
        &mut self,
        packed: &[u8],
        scales0: [f32; G],
        scales1: [f32; G],
    ) -> ([bool; G], [bool; G]) {
        if packed.len() != DIMENSION * usize::from(self.bits) / 8 {
            self.failed = [[true; G]; 2];
            return ([false; G], [false; G]);
        }
        let mut active0 = [false; G];
        let mut active1 = [false; G];
        let mut any_active = false;
        for (head, (&scale, failed)) in scales0.iter().zip(self.failed[0].iter_mut()).enumerate() {
            if !*failed && scale.is_finite() {
                active0[head] = true;
                any_active = true;
            } else {
                *failed = true;
            }
        }
        for (head, (&scale, failed)) in scales1.iter().zip(self.failed[1].iter_mut()).enumerate() {
            if !*failed && scale.is_finite() {
                active1[head] = true;
                any_active = true;
            } else {
                *failed = true;
            }
        }
        if !any_active {
            return ([false; G], [false; G]);
        }
        let scales = [scales0, scales1];
        let active = [active0, active1];
        #[cfg(target_arch = "aarch64")]
        let (ok0, ok1) = if self.neon {
            native::accumulate_group_pair(
                packed,
                self.bits,
                self.centroids,
                scales,
                active,
                &mut self.outputs,
            )
        } else {
            portable_accumulate_group_pair(
                packed,
                self.bits,
                self.centroids,
                scales,
                active,
                &mut self.outputs,
            )
        };
        #[cfg(not(target_arch = "aarch64"))]
        let (ok0, ok1) = portable_accumulate_group_pair(
            packed,
            self.bits,
            self.centroids,
            scales,
            active,
            &mut self.outputs,
        );
        for (head, &row_ok) in ok0.iter().enumerate() {
            if !active0[head] || !row_ok {
                self.failed[0][head] = true;
            }
        }
        for (head, &row_ok) in ok1.iter().enumerate() {
            if !active1[head] || !row_ok {
                self.failed[1][head] = true;
            }
        }
        (
            std::array::from_fn(|head| !self.failed[0][head]),
            std::array::from_fn(|head| !self.failed[1][head]),
        )
    }

    /// Accumulates one packed value row across the second token group only.
    #[must_use]
    pub fn accumulate_second(&mut self, packed: &[u8], scales1: [f32; G]) -> [bool; G] {
        if packed.len() != DIMENSION * usize::from(self.bits) / 8 {
            self.failed[1] = [true; G];
            return [false; G];
        }
        let mut active1 = [false; G];
        let mut any_active = false;
        for (head, (&scale, failed)) in scales1.iter().zip(self.failed[1].iter_mut()).enumerate() {
            if !*failed && scale.is_finite() {
                active1[head] = true;
                any_active = true;
            } else {
                *failed = true;
            }
        }
        if !any_active {
            return [false; G];
        }
        #[cfg(target_arch = "aarch64")]
        let ok1 = if self.neon {
            native::accumulate_group(
                packed,
                self.bits,
                self.centroids,
                scales1,
                active1,
                &mut self.outputs[1],
            )
        } else {
            portable_accumulate_group(
                packed,
                self.bits,
                self.centroids,
                scales1,
                active1,
                &mut self.outputs[1],
            )
        };
        #[cfg(not(target_arch = "aarch64"))]
        let ok1 = portable_accumulate_group(
            packed,
            self.bits,
            self.centroids,
            scales1,
            active1,
            &mut self.outputs[1],
        );
        for (head, &row_ok) in ok1.iter().enumerate() {
            if !active1[head] || !row_ok {
                self.failed[1][head] = true;
            }
        }
        std::array::from_fn(|head| !self.failed[1][head])
    }

    /// Accumulates `scales0.len()` contiguous shared packed value rows across both token groups.
    ///
    /// Validates slice lengths, prior failure state, and scale finiteness upfront.
    /// Returns `false` and poisons both token groups if any input is invalid or any
    /// output becomes nonfinite; otherwise preserves exact per-row binary32 rounding.
    #[must_use]
    pub fn accumulate_rows(
        &mut self,
        packed_rows: &[u8],
        scales0: &[[f32; G]],
        scales1: &[[f32; G]],
    ) -> bool {
        let Some(expected_bytes) = scales0
            .len()
            .checked_mul(DIMENSION * usize::from(self.bits) / 8)
        else {
            self.failed = [[true; G]; 2];
            return false;
        };
        if scales0.len() != scales1.len()
            || packed_rows.len() != expected_bytes
            || self.failed.iter().flatten().any(|&failed| failed)
            || scales0
                .iter()
                .chain(scales1.iter())
                .any(|row| row.iter().any(|scale| !scale.is_finite()))
        {
            self.failed = [[true; G]; 2];
            return false;
        }
        if scales0.is_empty() {
            return true;
        }
        #[cfg(target_arch = "aarch64")]
        let finite = if self.neon {
            native::accumulate_rows_group_pair(
                packed_rows,
                self.bits,
                self.centroids,
                scales0,
                scales1,
                &mut self.outputs,
            )
        } else {
            portable_accumulate_rows_group_pair(
                packed_rows,
                self.bits,
                self.centroids,
                scales0,
                scales1,
                &mut self.outputs,
            )
        };
        #[cfg(not(target_arch = "aarch64"))]
        let finite = portable_accumulate_rows_group_pair(
            packed_rows,
            self.bits,
            self.centroids,
            scales0,
            scales1,
            &mut self.outputs,
        );
        if !finite {
            self.failed = [[true; G]; 2];
        }
        finite
    }

    /// Publishes per-head sums for both token groups if their appends remained valid.
    #[must_use]
    pub fn finish(self) -> [[Option<[f32; DIMENSION]>; G]; 2] {
        [
            std::array::from_fn(|head| {
                if self.failed[0][head] {
                    None
                } else {
                    Some(self.outputs[0][head])
                }
            }),
            std::array::from_fn(|head| {
                if self.failed[1][head] {
                    None
                } else {
                    Some(self.outputs[1][head])
                }
            }),
        ]
    }
}

/// Adds a scaled code-only rotated value row into the supplied accumulator.
///
/// Every multiply and addition rounds separately to binary32. Invalid input or
/// nonfinite arithmetic returns false with the accumulator entirely unchanged.
/// This is a per-row operation; no complete cache is expanded or retained.
#[must_use]
pub fn accumulate(
    packed: &[u8],
    bits: u8,
    centroids: &[f32; 16],
    scale: f32,
    output: &mut [f32; DIMENSION],
) -> bool {
    if !valid(packed, bits, centroids)
        || !scale.is_finite()
        || !output.iter().all(|value| value.is_finite())
    {
        return false;
    }
    let mut staged = *output;
    #[cfg(target_arch = "aarch64")]
    let finite = if std::arch::is_aarch64_feature_detected!("neon") {
        native::accumulate(packed, bits, centroids, scale, &mut staged)
    } else {
        portable_accumulate(packed, bits, centroids, scale, &mut staged)
    };
    #[cfg(not(target_arch = "aarch64"))]
    let finite = portable_accumulate(packed, bits, centroids, scale, &mut staged);
    if !finite {
        return false;
    }
    *output = staged;
    true
}

/// Applies a normalized 256-coordinate Walsh-Hadamard transform in place.
///
/// Butterflies visit increasing half-widths from one through 128, followed by
/// exact power-of-two normalization by 1/16. Both backends preserve that order.
/// Nonfinite input or arithmetic returns false without changing any coordinate.
#[must_use]
pub fn hadamard(values: &mut [f32; DIMENSION]) -> bool {
    if !values.iter().all(|value| value.is_finite()) {
        return false;
    }
    let mut staged = *values;
    #[cfg(target_arch = "aarch64")]
    let finite = if std::arch::is_aarch64_feature_detected!("neon") {
        native::hadamard(&mut staged)
    } else {
        portable_hadamard(&mut staged);
        staged.iter().all(|value| value.is_finite())
    };
    #[cfg(not(target_arch = "aarch64"))]
    let finite = {
        portable_hadamard(&mut staged);
        staged.iter().all(|value| value.is_finite())
    };
    if !finite {
        return false;
    }
    *values = staged;
    true
}

/// Applies three signed, normalized Hadamard passes in one transaction.
///
/// Each boolean selects a negative diagonal sign before its forward pass.
/// Inverse execution reverses the passes and applies each sign after its
/// Hadamard. Sign-bit toggles preserve finite multiplication by `-1` exactly,
/// including signed zero and subnormals. Every pass retains increasing-width
/// butterflies, normalization by `1/16` and its finite-result boundary.
///
/// Native dispatch is selected once. The input is checked once and one staging
/// row serves all passes; there are no per-pass row copies. Nonfinite input or
/// arithmetic returns false without changing any caller-owned coordinate.
#[must_use]
pub fn rotate(
    values: &mut [f32; DIMENSION],
    negative_signs: &[[bool; DIMENSION]; 3],
    inverse: bool,
) -> bool {
    if !values.iter().all(|value| value.is_finite()) {
        return false;
    }
    let mut staged = *values;
    #[cfg(target_arch = "aarch64")]
    let neon = std::arch::is_aarch64_feature_detected!("neon");
    for pass in 0..negative_signs.len() {
        let signs = &negative_signs[if inverse {
            negative_signs.len() - 1 - pass
        } else {
            pass
        }];
        if !inverse {
            apply_signs(&mut staged, signs);
        }
        #[cfg(target_arch = "aarch64")]
        let finite = if neon {
            native::hadamard(&mut staged)
        } else {
            portable_hadamard(&mut staged);
            staged.iter().all(|value| value.is_finite())
        };
        #[cfg(not(target_arch = "aarch64"))]
        let finite = {
            portable_hadamard(&mut staged);
            staged.iter().all(|value| value.is_finite())
        };
        if !finite {
            return false;
        }
        if inverse {
            apply_signs(&mut staged, signs);
        }
    }
    *values = staged;
    true
}

/// Divides every coordinate by `scale` in binary64, rounding once to binary32.
///
/// Portable and NEON implementations use IEEE-754 binary64 division followed by
/// round-to-nearest narrowing. Returns `false` without changing `values` when
/// `scale` is nonfinite or nonpositive, or any input/normalized element is nonfinite.
#[must_use]
pub fn normalize(values: &mut [f32; DIMENSION], scale: f32) -> bool {
    if !scale.is_finite() || scale <= 0.0 {
        return false;
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return native::normalize(values, scale);
    }
    portable_normalize(values, scale)
}

/// Quantizes 256 rotated coordinates to nearest centroids and packs 3-bit or 4-bit codes.
///
/// Uses exact binary32 rounded absolute-distance comparisons (`(value - centroid).abs()`)
/// with lowest-code tie-breaking, producing bit-exact packed codes matching scalar encoding.
/// Returns `false` without changing `packed` on unsupported width, mismatched slice length,
/// or nonfinite input/codebook coordinates.
#[must_use]
pub fn quantize_pack(
    values: &[f32; DIMENSION],
    bits: u8,
    centroids: &[f32; 16],
    packed: &mut [u8],
) -> bool {
    if !valid(packed, bits, centroids) {
        return false;
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return native::quantize_pack(values, bits, centroids, packed);
    }
    portable_quantize_pack(values, bits, centroids, packed)
}

fn apply_signs(values: &mut [f32; DIMENSION], negative_signs: &[bool; DIMENSION]) {
    for (value, &negative) in values.iter_mut().zip(negative_signs) {
        *value = f32::from_bits(value.to_bits() ^ (u32::from(negative) << 31));
    }
}

fn valid(packed: &[u8], bits: u8, centroids: &[f32; 16]) -> bool {
    matches!((bits, packed.len()), (3, 96) | (4, 128))
        && centroids.iter().all(|value| value.is_finite())
}

fn coordinate(packed: &[u8], bits: u8, centroids: &[f32; 16], index: usize) -> f32 {
    let bit = index * usize::from(bits);
    let byte = bit / 8;
    let shift = bit % 8;
    let low = u16::from(packed[byte]);
    let high = if shift + usize::from(bits) > 8 {
        u16::from(packed[byte + 1]) << 8
    } else {
        0
    };
    let code = ((low | high) >> shift) & ((1_u16 << bits) - 1);
    centroids[usize::from(code)]
}

fn tile(packed: &[u8], bits: u8, centroids: &[f32; 16], index: usize) -> [f32; 4] {
    std::array::from_fn(|lane| coordinate(packed, bits, centroids, index + lane))
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::as_conversions,
    reason = "Binary64 accumulation is rounded once to binary32 and its finite range is checked immediately."
)]
fn finite_binary32(value: f64) -> Option<f32> {
    let value = value as f32;
    value.is_finite().then_some(value)
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Separate multiplication and addition preserve the deterministic scalar/NEON arithmetic contract."
)]
fn portable_dot(packed: &[u8], bits: u8, centroids: &[f32; 16], query: &[f32; DIMENSION]) -> f64 {
    let mut sums = [0.0_f64; 4];
    for (index, query) in query.as_chunks::<4>().0.iter().enumerate() {
        let values = tile(packed, bits, centroids, index * 4);
        for ((sum, value), query) in sums.iter_mut().zip(values).zip(query) {
            *sum += f64::from(value) * f64::from(*query);
        }
    }
    reduce(sums)
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Separate multiplication and addition preserve the deterministic scalar/NEON arithmetic contract."
)]
fn portable_dot_group<const G: usize>(
    packed: &[u8],
    bits: u8,
    centroids: &[f32; 16],
    queries: &[&[f32; DIMENSION]; G],
) -> [f64; G] {
    let mut sums = [[0.0_f64; 4]; G];
    for index in 0..(DIMENSION / 4) {
        let base = index * 4;
        let values = tile(packed, bits, centroids, base);
        for head in 0..G {
            let query = &queries[head][base..base + 4];
            for ((sum, value), &query_val) in sums[head].iter_mut().zip(values).zip(query) {
                *sum += f64::from(value) * f64::from(query_val);
            }
        }
    }
    sums.map(reduce)
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Separate multiplication and addition preserve the deterministic scalar/NEON arithmetic contract."
)]
fn portable_dot_group_pair<const G: usize>(
    packed: &[u8],
    bits: u8,
    centroids: &[f32; 16],
    first_queries: &[&[f32; DIMENSION]; G],
    second_queries: &[&[f32; DIMENSION]; G],
) -> ([f64; G], [f64; G]) {
    let mut first_sums = [[0.0_f64; 4]; G];
    let mut second_sums = [[0.0_f64; 4]; G];
    for index in 0..(DIMENSION / 4) {
        let base = index * 4;
        let values = tile(packed, bits, centroids, base);
        for head in 0..G {
            let query0 = &first_queries[head][base..base + 4];
            let query1 = &second_queries[head][base..base + 4];
            for (lane, &value) in values.iter().enumerate() {
                first_sums[head][lane] += f64::from(value) * f64::from(query0[lane]);
                second_sums[head][lane] += f64::from(value) * f64::from(query1[lane]);
            }
        }
    }
    (first_sums.map(reduce), second_sums.map(reduce))
}

fn portable_score_rows_group<const G: usize>(
    packed_rows: &[u8],
    bits: u8,
    centroids: &[f32; 16],
    queries: &[&[f32; DIMENSION]; G],
    out: &mut [[f32; G]],
) -> bool {
    let row_bytes = DIMENSION * usize::from(bits) / 8;
    for (dst, row) in out.iter_mut().zip(packed_rows.chunks_exact(row_bytes)) {
        let dots = portable_dot_group(row, bits, centroids, queries);
        for (dst_head, dot_f64) in dst.iter_mut().zip(dots) {
            let Some(val) = finite_binary32(dot_f64) else {
                return false;
            };
            *dst_head = val;
        }
    }
    true
}

fn portable_score_rows_pair<const G: usize>(
    packed_rows: &[u8],
    bits: u8,
    centroids: &[f32; 16],
    first_queries: &[&[f32; DIMENSION]; G],
    second_queries: &[&[f32; DIMENSION]; G],
    out0: &mut [[f32; G]],
    out1: &mut [[f32; G]],
) -> bool {
    let row_bytes = DIMENSION * usize::from(bits) / 8;
    for ((dst0, dst1), row) in out0
        .iter_mut()
        .zip(out1.iter_mut())
        .zip(packed_rows.chunks_exact(row_bytes))
    {
        let (dots0, dots1) =
            portable_dot_group_pair(row, bits, centroids, first_queries, second_queries);
        for (dst_head, dot_f64) in dst0.iter_mut().zip(dots0) {
            let Some(val) = finite_binary32(dot_f64) else {
                return false;
            };
            *dst_head = val;
        }
        for (dst_head, dot_f64) in dst1.iter_mut().zip(dots1) {
            let Some(val) = finite_binary32(dot_f64) else {
                return false;
            };
            *dst_head = val;
        }
    }
    true
}

fn reduce(sums: [f64; 4]) -> f64 {
    ((sums[0] + sums[1]) + sums[2]) + sums[3]
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Value accumulation intentionally retains separate binary32 multiply/add rounding."
)]
fn portable_accumulate(
    packed: &[u8],
    bits: u8,
    centroids: &[f32; 16],
    scale: f32,
    output: &mut [f32; DIMENSION],
) -> bool {
    let mut finite = true;
    for (index, output) in output.iter_mut().enumerate() {
        *output += scale * coordinate(packed, bits, centroids, index);
        finite &= output.is_finite();
    }
    finite
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Value accumulation intentionally retains separate binary32 multiply/add rounding."
)]
fn portable_accumulate_group<const G: usize>(
    packed: &[u8],
    bits: u8,
    centroids: &[f32; 16],
    scales: [f32; G],
    active: [bool; G],
    outputs: &mut [[f32; DIMENSION]; G],
) -> [bool; G] {
    let mut finite = active;
    for tile_index in 0..(DIMENSION / 4) {
        let base = tile_index * 4;
        let values = tile(packed, bits, centroids, base);
        for ((&is_active, &scale), (output, is_finite)) in active
            .iter()
            .zip(&scales)
            .zip(outputs.iter_mut().zip(finite.iter_mut()))
        {
            if is_active {
                for (target, value) in output[base..base + 4].iter_mut().zip(values) {
                    *target += scale * value;
                    *is_finite &= target.is_finite();
                }
            }
        }
    }
    finite
}

#[expect(
    clippy::suboptimal_flops,
    reason = "Value accumulation intentionally retains separate binary32 multiply/add rounding."
)]
fn portable_accumulate_group_pair<const G: usize>(
    packed: &[u8],
    bits: u8,
    centroids: &[f32; 16],
    scales: [[f32; G]; 2],
    active: [[bool; G]; 2],
    outputs: &mut [[[f32; DIMENSION]; G]; 2],
) -> ([bool; G], [bool; G]) {
    let mut finite = active;
    for tile_index in 0..(DIMENSION / 4) {
        let base = tile_index * 4;
        let values = tile(packed, bits, centroids, base);
        for token in 0..2 {
            for ((&is_active, &scale), (output, is_finite)) in active[token]
                .iter()
                .zip(&scales[token])
                .zip(outputs[token].iter_mut().zip(finite[token].iter_mut()))
            {
                if is_active {
                    for (target, value) in output[base..base + 4].iter_mut().zip(values) {
                        *target += scale * value;
                        *is_finite &= target.is_finite();
                    }
                }
            }
        }
    }
    finite.into()
}

fn portable_accumulate_rows_group<const G: usize>(
    packed_rows: &[u8],
    bits: u8,
    centroids: &[f32; 16],
    scales: &[[f32; G]],
    outputs: &mut [[f32; DIMENSION]; G],
) -> bool {
    let row_bytes = DIMENSION * usize::from(bits) / 8;
    for (row, &row_scales) in packed_rows.chunks_exact(row_bytes).zip(scales) {
        let ok = portable_accumulate_group(row, bits, centroids, row_scales, [true; G], outputs);
        if !ok.iter().all(|&head_ok| head_ok) {
            return false;
        }
    }
    true
}

fn portable_accumulate_rows_group_pair<const G: usize>(
    packed_rows: &[u8],
    bits: u8,
    centroids: &[f32; 16],
    scales0: &[[f32; G]],
    scales1: &[[f32; G]],
    outputs: &mut [[[f32; DIMENSION]; G]; 2],
) -> bool {
    let row_bytes = DIMENSION * usize::from(bits) / 8;
    for ((row, &row_scales0), &row_scales1) in packed_rows
        .chunks_exact(row_bytes)
        .zip(scales0)
        .zip(scales1)
    {
        let (ok0, ok1) = portable_accumulate_group_pair(
            row,
            bits,
            centroids,
            [row_scales0, row_scales1],
            [[true; G]; 2],
            outputs,
        );
        if !ok0.iter().chain(ok1.iter()).all(|&head_ok| head_ok) {
            return false;
        }
    }
    true
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::as_conversions,
    reason = "Binary64 division is rounded once to binary32 and its finite range is checked immediately."
)]
fn portable_normalize(values: &mut [f32; DIMENSION], scale: f32) -> bool {
    let mut staged = [0.0_f32; DIMENSION];
    let scale_f64 = f64::from(scale);
    for (dst, &value) in staged.iter_mut().zip(values.iter()) {
        if !value.is_finite() {
            return false;
        }
        let normalized = (f64::from(value) / scale_f64) as f32;
        if !normalized.is_finite() {
            return false;
        }
        *dst = normalized;
    }
    *values = staged;
    true
}

fn portable_quantize_pack(
    values: &[f32; DIMENSION],
    bits: u8,
    centroids: &[f32; 16],
    packed: &mut [u8],
) -> bool {
    if !values.iter().all(|value| value.is_finite()) {
        return false;
    }
    let levels = 1_u8 << bits;
    packed.fill(0);
    for (index, &value) in values.iter().enumerate() {
        let mut best_code = 0_u8;
        let mut best_dist = (value - centroids[0]).abs();
        for candidate in 1..levels {
            let dist = (value - centroids[usize::from(candidate)]).abs();
            if dist < best_dist {
                best_code = candidate;
                best_dist = dist;
            }
        }
        let offset = index * usize::from(bits);
        let byte = offset / 8;
        let shift = offset % 8;
        packed[byte] |= best_code << shift;
        if shift + usize::from(bits) > 8 {
            packed[byte + 1] |= best_code >> (8 - shift);
        }
    }
    true
}

fn butterfly(values: &mut [f32; DIMENSION], half: usize) {
    for block in values.chunks_exact_mut(half * 2) {
        let (left, right) = block.split_at_mut(half);
        for (left, right) in left.iter_mut().zip(right) {
            let a = *left;
            let b = *right;
            *left = a + b;
            *right = a - b;
        }
    }
}

fn portable_hadamard(values: &mut [f32; DIMENSION]) {
    let mut half = 1;
    while half < DIMENSION {
        butterfly(values, half);
        half *= 2;
    }
    for value in values {
        *value *= 0.0625;
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(
    unsafe_code,
    reason = "Private baseline-NEON intrinsics use checked fixed-size tiles behind runtime detection and failure-atomic safe public APIs."
)]
mod native {
    use core::arch::aarch64::{
        float32x4_t, float32x4x4_t, float64x2_t, int32x4_t, uint8x16_t, uint8x16x2_t,
        uint8x16x4_t, uint32x4_t, vabdq_f32, vaddq_f32, vaddq_f64, vaddvq_u32, vandq_u8,
        vandq_u32, vbslq_f32, vbslq_u32, vcltq_f32, vcltq_u32, vcvt_f32_f64, vcvt_f64_f32,
        vcvt_high_f32_f64, vcvt_high_f64_f32, vdivq_f64, vdupq_laneq_f32, vdupq_n_f32,
        vdupq_n_f64, vdupq_n_u8, vdupq_n_u32, vdupq_n_u64, vget_low_f32, vld1q_f32, vld1q_s32,
        vld1q_u8, vld1q_u8_x2, vld1q_u8_x4, vld3q_u8, vld4q_f32, vminvq_u32, vmulq_f32,
        vmulq_f64, vmulq_u32, vorrq_u8, vorrq_u32, vqtbl2q_u8, vqtbl4q_u8, vreinterpretq_f32_u8,
        vreinterpretq_u8_u16, vreinterpretq_u8_u32, vreinterpretq_u8_u64, vreinterpretq_u16_u8,
        vreinterpretq_u32_f32, vshlq_n_u8, vshlq_u32, vshrq_n_u8, vsliq_n_u8, vst1q_f32,
        vst1q_f64, vst4q_f32, vsubq_f32, vzip1q_u8, vzip1q_u16, vzip2q_u8, vzip2q_u16,
    };

    use super::DIMENSION;

    const TILE_OFFSETS: [u8; 16] = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
    const THREE_BIT_UNPACK_SHIFTS_LO: [i32; 4] = [0, -3, -6, -9];
    const THREE_BIT_UNPACK_SHIFTS_HI: [i32; 4] = [-12, -15, -18, -21];
    const THREE_BIT_PACK_SHIFTS_LO: [i32; 4] = [0, 3, 6, 9];
    const THREE_BIT_PACK_SHIFTS_HI: [i32; 4] = [12, 15, 18, 21];
    const FOUR_BIT_PACK_SHIFTS_LO: [i32; 4] = [0, 4, 8, 12];
    const FOUR_BIT_PACK_SHIFTS_HI: [i32; 4] = [16, 20, 24, 28];

    #[inline]
    fn prefetch_l1(ptr: *const u8) {
        // SAFETY: `prfm pldl1keep` is a non-faulting architectural read prefetch hint.
        unsafe {
            core::arch::asm!(
                "prfm pldl1keep, [{ptr}]",
                ptr = in(reg) ptr,
                options(nostack, readonly, preserves_flags),
            );
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn codebook_bytes(centroids: &[f32; 16]) -> uint8x16x4_t {
        // SAFETY: sixteen initialized binary32 centroids occupy exactly the
        // sixty-four bytes required by the four unaligned vector loads.
        unsafe { vld1q_u8_x4(centroids.as_ptr().cast()) }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn codebook_bytes_3(centroids: &[f32; 16]) -> uint8x16x2_t {
        // SAFETY: the first eight initialized binary32 centroids occupy thirty-two
        // bytes required by the two unaligned vector loads.
        unsafe { vld1q_u8_x2(centroids.as_ptr().cast()) }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn tile_offsets() -> uint8x16_t {
        // SAFETY: TILE_OFFSETS is a static sixteen-byte array.
        unsafe { vld1q_u8(TILE_OFFSETS.as_ptr()) }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn expand_4bit_16(
        scaled_codes: uint8x16_t,
        table: uint8x16x4_t,
        offsets: uint8x16_t,
    ) -> [float32x4_t; 4] {
        let dup_lo = vzip1q_u8(scaled_codes, scaled_codes);
        let dup_hi = vzip2q_u8(scaled_codes, scaled_codes);
        [
            vreinterpretq_f32_u8(vqtbl4q_u8(
                table,
                vorrq_u8(vzip1q_u8(dup_lo, dup_lo), offsets),
            )),
            vreinterpretq_f32_u8(vqtbl4q_u8(
                table,
                vorrq_u8(vzip2q_u8(dup_lo, dup_lo), offsets),
            )),
            vreinterpretq_f32_u8(vqtbl4q_u8(
                table,
                vorrq_u8(vzip1q_u8(dup_hi, dup_hi), offsets),
            )),
            vreinterpretq_f32_u8(vqtbl4q_u8(
                table,
                vorrq_u8(vzip2q_u8(dup_hi, dup_hi), offsets),
            )),
        ]
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn unpack_4bit_32(
        chunk: &[u8; 16],
        table: uint8x16x4_t,
        mask_3c: uint8x16_t,
        offsets: uint8x16_t,
    ) -> [float32x4_t; 8] {
        // SAFETY: chunk is a sixteen-byte array slice covering one NEON load.
        let packed_bytes = unsafe { vld1q_u8(chunk.as_ptr()) };
        let low_codes = vandq_u8(vshlq_n_u8::<2>(packed_bytes), mask_3c);
        let high_codes = vandq_u8(vshrq_n_u8::<2>(packed_bytes), mask_3c);
        let first_16 = expand_4bit_16(vzip1q_u8(low_codes, high_codes), table, offsets);
        let second_16 = expand_4bit_16(vzip2q_u8(low_codes, high_codes), table, offsets);
        [
            first_16[0],
            first_16[1],
            first_16[2],
            first_16[3],
            second_16[0],
            second_16[1],
            second_16[2],
            second_16[3],
        ]
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn unpack_3bit_128(chunk: &[u8; 48], mask_1c: uint8x16_t) -> [[uint8x16_t; 2]; 4] {
        // SAFETY: chunk is a forty-eight-byte array covering sixteen 3-byte tuples.
        let tuples = unsafe { vld3q_u8(chunk.as_ptr()) };
        let byte0 = tuples.0;
        let byte1 = tuples.1;
        let byte2 = tuples.2;

        let c0 = vandq_u8(vshlq_n_u8::<2>(byte0), mask_1c);
        let c1 = vandq_u8(vshrq_n_u8::<1>(byte0), mask_1c);
        let c2 = vandq_u8(vsliq_n_u8::<4>(vshrq_n_u8::<4>(byte0), byte1), mask_1c);
        let c3 = vandq_u8(vshlq_n_u8::<1>(byte1), mask_1c);
        let c4 = vandq_u8(vshrq_n_u8::<2>(byte1), mask_1c);
        let c5 = vandq_u8(vsliq_n_u8::<3>(vshrq_n_u8::<5>(byte1), byte2), mask_1c);
        let c6 = vandq_u8(byte2, mask_1c);
        let c7 = vandq_u8(vshrq_n_u8::<3>(byte2), mask_1c);

        let p01_lo = vreinterpretq_u16_u8(vzip1q_u8(c0, c1));
        let p01_hi = vreinterpretq_u16_u8(vzip2q_u8(c0, c1));
        let p23_lo = vreinterpretq_u16_u8(vzip1q_u8(c2, c3));
        let p23_hi = vreinterpretq_u16_u8(vzip2q_u8(c2, c3));
        let p45_lo = vreinterpretq_u16_u8(vzip1q_u8(c4, c5));
        let p45_hi = vreinterpretq_u16_u8(vzip2q_u8(c4, c5));
        let p67_lo = vreinterpretq_u16_u8(vzip1q_u8(c6, c7));
        let p67_hi = vreinterpretq_u16_u8(vzip2q_u8(c6, c7));

        [
            [
                vreinterpretq_u8_u16(vzip1q_u16(p01_lo, p23_lo)),
                vreinterpretq_u8_u16(vzip1q_u16(p45_lo, p67_lo)),
            ],
            [
                vreinterpretq_u8_u16(vzip2q_u16(p01_lo, p23_lo)),
                vreinterpretq_u8_u16(vzip2q_u16(p45_lo, p67_lo)),
            ],
            [
                vreinterpretq_u8_u16(vzip1q_u16(p01_hi, p23_hi)),
                vreinterpretq_u8_u16(vzip1q_u16(p45_hi, p67_hi)),
            ],
            [
                vreinterpretq_u8_u16(vzip2q_u16(p01_hi, p23_hi)),
                vreinterpretq_u8_u16(vzip2q_u16(p45_hi, p67_hi)),
            ],
        ]
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn expand_3bit_32(
        pair: [uint8x16_t; 2],
        table: uint8x16x2_t,
        offsets: uint8x16_t,
    ) -> [float32x4_t; 8] {
        let low_dup0 = vzip1q_u8(pair[0], pair[0]);
        let high_dup0 = vzip1q_u8(pair[1], pair[1]);
        let low_dup1 = vzip2q_u8(pair[0], pair[0]);
        let high_dup1 = vzip2q_u8(pair[1], pair[1]);
        [
            vreinterpretq_f32_u8(vqtbl2q_u8(
                table,
                vorrq_u8(vzip1q_u8(low_dup0, low_dup0), offsets),
            )),
            vreinterpretq_f32_u8(vqtbl2q_u8(
                table,
                vorrq_u8(vzip1q_u8(high_dup0, high_dup0), offsets),
            )),
            vreinterpretq_f32_u8(vqtbl2q_u8(
                table,
                vorrq_u8(vzip2q_u8(low_dup0, low_dup0), offsets),
            )),
            vreinterpretq_f32_u8(vqtbl2q_u8(
                table,
                vorrq_u8(vzip2q_u8(high_dup0, high_dup0), offsets),
            )),
            vreinterpretq_f32_u8(vqtbl2q_u8(
                table,
                vorrq_u8(vzip1q_u8(low_dup1, low_dup1), offsets),
            )),
            vreinterpretq_f32_u8(vqtbl2q_u8(
                table,
                vorrq_u8(vzip1q_u8(high_dup1, high_dup1), offsets),
            )),
            vreinterpretq_f32_u8(vqtbl2q_u8(
                table,
                vorrq_u8(vzip2q_u8(low_dup1, low_dup1), offsets),
            )),
            vreinterpretq_f32_u8(vqtbl2q_u8(
                table,
                vorrq_u8(vzip2q_u8(high_dup1, high_dup1), offsets),
            )),
        ]
    }

    #[cfg(test)]
    #[inline]
    #[target_feature(enable = "neon")]
    fn decode_row<const BITS: u8>(
        packed: &[u8],
        centroids: &[f32; 16],
        out: &mut [f32; DIMENSION],
    ) {
        let offsets = tile_offsets();
        if BITS == 4 {
            let table = codebook_bytes(centroids);
            let mask_3c = vdupq_n_u8(0x3c);
            for (chunk, dst) in packed
                .as_chunks::<16>()
                .0
                .iter()
                .zip(out.as_chunks_mut::<32>().0)
            {
                let tiles = unpack_4bit_32(chunk, table, mask_3c, offsets);
                for (tile, out_quad) in tiles.into_iter().zip(dst.as_chunks_mut::<4>().0) {
                    // SAFETY: out_quad is a unique four-element slice in out.
                    unsafe { vst1q_f32(out_quad.as_mut_ptr(), tile) };
                }
            }
        } else {
            let table = codebook_bytes_3(centroids);
            let mask_1c = vdupq_n_u8(0x1c);
            for (chunk, dst) in packed
                .as_chunks::<48>()
                .0
                .iter()
                .zip(out.as_chunks_mut::<128>().0)
            {
                let pairs = unpack_3bit_128(chunk, mask_1c);
                for (pair, sub_dst) in pairs.into_iter().zip(dst.as_chunks_mut::<32>().0) {
                    let tiles = expand_3bit_32(pair, table, offsets);
                    for (tile, out_quad) in tiles.into_iter().zip(sub_dst.as_chunks_mut::<4>().0) {
                        // SAFETY: out_quad is a unique four-element slice in out.
                        unsafe { vst1q_f32(out_quad.as_mut_ptr(), tile) };
                    }
                }
            }
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn dot_tile_group<const G: usize>(
        values: float32x4_t,
        base: usize,
        queries: &[&[f32; DIMENSION]; G],
        low_sums: &mut [float64x2_t; G],
        high_sums: &mut [float64x2_t; G],
    ) {
        let values_low = vcvt_f64_f32(vget_low_f32(values));
        let values_high = vcvt_high_f64_f32(values);
        for head in 0..G {
            // SAFETY: callers only pass base offsets where base + 4 <= DIMENSION.
            let query = unsafe { vld1q_f32(queries[head][base..].as_ptr()) };
            let low = vmulq_f64(values_low, vcvt_f64_f32(vget_low_f32(query)));
            let high = vmulq_f64(values_high, vcvt_high_f64_f32(query));
            low_sums[head] = vaddq_f64(low_sums[head], low);
            high_sums[head] = vaddq_f64(high_sums[head], high);
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn finish_dot_group<const G: usize>(
        low_sums: &[float64x2_t; G],
        high_sums: &[float64x2_t; G],
    ) -> [f64; G] {
        std::array::from_fn(|head| {
            let mut low = [0.0; 2];
            let mut high = [0.0; 2];
            // SAFETY: low is a unique initialized two-element output array.
            unsafe { vst1q_f64(low.as_mut_ptr(), low_sums[head]) };
            // SAFETY: high is a disjoint unique two-element output array.
            unsafe { vst1q_f64(high.as_mut_ptr(), high_sums[head]) };
            super::reduce([low[0], low[1], high[0], high[1]])
        })
    }

    pub(super) fn dot(
        packed: &[u8],
        bits: u8,
        centroids: &[f32; 16],
        query: &[f32; DIMENSION],
    ) -> f64 {
        dot_group(packed, bits, centroids, &[query])[0]
    }

    pub(super) fn dot_group<const G: usize>(
        packed: &[u8],
        bits: u8,
        centroids: &[f32; 16],
        queries: &[&[f32; DIMENSION]; G],
    ) -> [f64; G] {
        if bits == 4 {
            // SAFETY: the safe public caller proves baseline NEON at runtime and
            // validates exactly 256 four-bit codes and finite fixed-array inputs.
            unsafe { dot_group_neon::<4, G>(packed, centroids, queries) }
        } else {
            // SAFETY: the same caller admits only three- or four-bit codes;
            // this branch has exactly 256 three-bit codes and detected NEON.
            unsafe { dot_group_neon::<3, G>(packed, centroids, queries) }
        }
    }

    #[target_feature(enable = "neon")]
    fn dot_group_neon<const BITS: u8, const G: usize>(
        packed: &[u8],
        centroids: &[f32; 16],
        queries: &[&[f32; DIMENSION]; G],
    ) -> [f64; G] {
        let mut low_sums: [_; G] = std::array::from_fn(|_| vdupq_n_f64(0.0));
        let mut high_sums: [_; G] = std::array::from_fn(|_| vdupq_n_f64(0.0));
        let offsets = tile_offsets();
        if BITS == 4 {
            let table = codebook_bytes(centroids);
            let mask_3c = vdupq_n_u8(0x3c);
            for (chunk_idx, chunk) in packed.as_chunks::<16>().0.iter().take(8).enumerate() {
                let tiles = unpack_4bit_32(chunk, table, mask_3c, offsets);
                let chunk_base = chunk_idx * 32;
                for (tile_idx, values) in tiles.into_iter().enumerate() {
                    dot_tile_group(
                        values,
                        chunk_base + tile_idx * 4,
                        queries,
                        &mut low_sums,
                        &mut high_sums,
                    );
                }
            }
        } else {
            let table = codebook_bytes_3(centroids);
            let mask_1c = vdupq_n_u8(0x1c);
            for (chunk_idx, chunk) in packed.as_chunks::<48>().0.iter().take(2).enumerate() {
                let pairs = unpack_3bit_128(chunk, mask_1c);
                let chunk_base = chunk_idx * 128;
                for (pair_idx, pair) in pairs.into_iter().enumerate() {
                    let tiles = expand_3bit_32(pair, table, offsets);
                    let pair_base = chunk_base + pair_idx * 32;
                    for (tile_idx, values) in tiles.into_iter().enumerate() {
                        dot_tile_group(
                            values,
                            pair_base + tile_idx * 4,
                            queries,
                            &mut low_sums,
                            &mut high_sums,
                        );
                    }
                }
            }
        }
        finish_dot_group(&low_sums, &high_sums)
    }

    pub(super) fn dot_group_pair<const G: usize>(
        packed: &[u8],
        bits: u8,
        centroids: &[f32; 16],
        first_queries: &[&[f32; DIMENSION]; G],
        second_queries: &[&[f32; DIMENSION]; G],
    ) -> ([f64; G], [f64; G]) {
        if bits == 4 {
            // SAFETY: caller detects NEON and validates 256 four-bit codes and queries.
            unsafe { dot_group_pair_neon::<4, G>(packed, centroids, first_queries, second_queries) }
        } else {
            // SAFETY: caller detects NEON and validates 256 three-bit codes and queries.
            unsafe { dot_group_pair_neon::<3, G>(packed, centroids, first_queries, second_queries) }
        }
    }

    #[target_feature(enable = "neon")]
    fn dot_group_pair_neon<const BITS: u8, const G: usize>(
        packed: &[u8],
        centroids: &[f32; 16],
        first_queries: &[&[f32; DIMENSION]; G],
        second_queries: &[&[f32; DIMENSION]; G],
    ) -> ([f64; G], [f64; G]) {
        let mut low0: [_; G] = std::array::from_fn(|_| vdupq_n_f64(0.0));
        let mut high0: [_; G] = std::array::from_fn(|_| vdupq_n_f64(0.0));
        let mut low1: [_; G] = std::array::from_fn(|_| vdupq_n_f64(0.0));
        let mut high1: [_; G] = std::array::from_fn(|_| vdupq_n_f64(0.0));
        let offsets = tile_offsets();
        if BITS == 4 {
            let table = codebook_bytes(centroids);
            let mask_3c = vdupq_n_u8(0x3c);
            for (chunk_idx, chunk) in packed.as_chunks::<16>().0.iter().take(8).enumerate() {
                let tiles = unpack_4bit_32(chunk, table, mask_3c, offsets);
                let chunk_base = chunk_idx * 32;
                for (tile_idx, values) in tiles.into_iter().enumerate() {
                    let base = chunk_base + tile_idx * 4;
                    dot_tile_group(values, base, first_queries, &mut low0, &mut high0);
                    dot_tile_group(values, base, second_queries, &mut low1, &mut high1);
                }
            }
        } else {
            let table = codebook_bytes_3(centroids);
            let mask_1c = vdupq_n_u8(0x1c);
            for (chunk_idx, chunk) in packed.as_chunks::<48>().0.iter().take(2).enumerate() {
                let pairs = unpack_3bit_128(chunk, mask_1c);
                let chunk_base = chunk_idx * 128;
                for (pair_idx, pair) in pairs.into_iter().enumerate() {
                    let tiles = expand_3bit_32(pair, table, offsets);
                    let pair_base = chunk_base + pair_idx * 32;
                    for (tile_idx, values) in tiles.into_iter().enumerate() {
                        let base = pair_base + tile_idx * 4;
                        dot_tile_group(values, base, first_queries, &mut low0, &mut high0);
                        dot_tile_group(values, base, second_queries, &mut low1, &mut high1);
                    }
                }
            }
        }
        (
            finish_dot_group(&low0, &high0),
            finish_dot_group(&low1, &high1),
        )
    }

    #[derive(Clone, Copy)]
    struct FourBitTables {
        table: uint8x16x4_t,
        mask_3c: uint8x16_t,
        offsets: uint8x16_t,
    }

    #[derive(Clone, Copy)]
    struct ThreeBitTables {
        table: uint8x16x2_t,
        mask_1c: uint8x16_t,
        offsets: uint8x16_t,
    }

    #[derive(Clone, Copy)]
    struct ThreeBitStripTables {
        table: uint8x16x2_t,
        sh_lo: int32x4_t,
        sh_hi: int32x4_t,
        mask_7: uint32x4_t,
        mul_04: uint32x4_t,
        offsets: uint8x16_t,
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn widen_queries<const G: usize>(
        queries: &[&[f32; DIMENSION]; G],
    ) -> [[[float64x2_t; 2]; G]; 64] {
        std::array::from_fn(|tile| {
            let base = tile * 4;
            std::array::from_fn(|head| {
                // SAFETY: tile < 64 guarantees base + 4 <= DIMENSION.
                let query = unsafe { vld1q_f32(queries[head][base..].as_ptr()) };
                [vcvt_f64_f32(vget_low_f32(query)), vcvt_high_f64_f32(query)]
            })
        })
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn dot_tile_widened<const G: usize, const H: usize>(
        values: float32x4_t,
        widened_tile: &[[float64x2_t; 2]; G],
        head_start: usize,
        low_sums: &mut [float64x2_t; H],
        high_sums: &mut [float64x2_t; H],
    ) {
        let values_low = vcvt_f64_f32(vget_low_f32(values));
        let values_high = vcvt_high_f64_f32(values);
        for h in 0..H {
            let [q_lo, q_hi] = widened_tile[head_start + h];
            low_sums[h] = vaddq_f64(low_sums[h], vmulq_f64(values_low, q_lo));
            high_sums[h] = vaddq_f64(high_sums[h], vmulq_f64(values_high, q_hi));
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn store_reduced_rows<const G: usize, const R: usize, const H: usize>(
        low_sums: &[[float64x2_t; H]; R],
        high_sums: &[[float64x2_t; H]; R],
        head_start: usize,
        out: &mut [[f32; G]],
    ) -> bool {
        for r in 0..R {
            for h in 0..H {
                let mut low = [0.0; 2];
                let mut high = [0.0; 2];
                // SAFETY: low is a unique initialized two-element output array.
                unsafe { vst1q_f64(low.as_mut_ptr(), low_sums[r][h]) };
                // SAFETY: high is a disjoint unique two-element output array.
                unsafe { vst1q_f64(high.as_mut_ptr(), high_sums[r][h]) };
                let Some(val) =
                    super::finite_binary32(super::reduce([low[0], low[1], high[0], high[1]]))
                else {
                    return false;
                };
                out[r][head_start + h] = val;
            }
        }
        true
    }

    #[target_feature(enable = "neon")]
    fn score_rows_chunk_4bit<const G: usize, const R: usize, const H: usize>(
        rows: [&[u8]; R],
        tables: FourBitTables,
        widened: &[[[float64x2_t; 2]; G]; 64],
        head_start: usize,
        out: &mut [[f32; G]],
    ) -> bool {
        let mut low_sums = [[vdupq_n_f64(0.0); H]; R];
        let mut high_sums = [[vdupq_n_f64(0.0); H]; R];
        for chunk_idx in 0..8 {
            let byte_off = chunk_idx * 16;
            let tile_base = chunk_idx * 8;
            for r in 0..R {
                // SAFETY: caller ensures every row in `rows` has 128 bytes (`byte_off + 16 <= 128`).
                let packed_bytes = unsafe { vld1q_u8(rows[r][byte_off..].as_ptr()) };
                let low_codes = vandq_u8(vshlq_n_u8::<2>(packed_bytes), tables.mask_3c);
                let high_codes = vandq_u8(vshrq_n_u8::<2>(packed_bytes), tables.mask_3c);
                let first_16 = expand_4bit_16(
                    vzip1q_u8(low_codes, high_codes),
                    tables.table,
                    tables.offsets,
                );
                for (tile_idx, values) in first_16.into_iter().enumerate() {
                    dot_tile_widened(
                        values,
                        &widened[tile_base + tile_idx],
                        head_start,
                        &mut low_sums[r],
                        &mut high_sums[r],
                    );
                }
                let second_16 = expand_4bit_16(
                    vzip2q_u8(low_codes, high_codes),
                    tables.table,
                    tables.offsets,
                );
                for (tile_idx, values) in second_16.into_iter().enumerate() {
                    dot_tile_widened(
                        values,
                        &widened[tile_base + 4 + tile_idx],
                        head_start,
                        &mut low_sums[r],
                        &mut high_sums[r],
                    );
                }
            }
        }
        store_reduced_rows(&low_sums, &high_sums, head_start, out)
    }

    #[target_feature(enable = "neon")]
    fn score_rows_chunk_3bit<const G: usize, const R: usize, const H: usize>(
        rows: [&[u8]; R],
        tables: ThreeBitTables,
        widened: &[[[float64x2_t; 2]; G]; 64],
        head_start: usize,
        out: &mut [[f32; G]],
    ) -> bool {
        let mut low_sums = [[vdupq_n_f64(0.0); H]; R];
        let mut high_sums = [[vdupq_n_f64(0.0); H]; R];
        for chunk_idx in 0..2 {
            let chunk_base = chunk_idx * 32;
            for r in 0..R {
                let chunk_arr = &rows[r].as_chunks::<48>().0[chunk_idx];
                let pairs = unpack_3bit_128(chunk_arr, tables.mask_1c);
                for (pair_idx, pair) in pairs.into_iter().enumerate() {
                    let tiles = expand_3bit_32(pair, tables.table, tables.offsets);
                    let pair_base = chunk_base + pair_idx * 8;
                    for (tile_idx, values) in tiles.into_iter().enumerate() {
                        dot_tile_widened(
                            values,
                            &widened[pair_base + tile_idx],
                            head_start,
                            &mut low_sums[r],
                            &mut high_sums[r],
                        );
                    }
                }
            }
        }
        store_reduced_rows(&low_sums, &high_sums, head_start, out)
    }

    #[target_feature(enable = "neon")]
    fn score_rows_sub<const BITS: u8, const G: usize, const R: usize, const H: usize>(
        rows: [&[u8]; R],
        tables4: Option<FourBitTables>,
        tables3: Option<ThreeBitTables>,
        widened: &[[[float64x2_t; 2]; G]; 64],
        head_start: usize,
        out: &mut [[f32; G]],
    ) -> bool {
        if BITS == 4 {
            let Some(tables) = tables4 else {
                return false;
            };
            score_rows_chunk_4bit::<G, R, H>(rows, tables, widened, head_start, out)
        } else {
            let Some(tables) = tables3 else {
                return false;
            };
            score_rows_chunk_3bit::<G, R, H>(rows, tables, widened, head_start, out)
        }
    }

    #[target_feature(enable = "neon")]
    fn score_rows_dispatch_heads<const BITS: u8, const G: usize, const R: usize>(
        rows: [&[u8]; R],
        tables4: Option<FourBitTables>,
        tables3: Option<ThreeBitTables>,
        widened: &[[[float64x2_t; 2]; G]; 64],
        out: &mut [[f32; G]],
    ) -> bool {
        let max_h = if R == 1 { 6 } else if G <= 4 { G } else { 3 };
        let mut head_start = 0_usize;
        while head_start < G {
            let rem = G - head_start;
            let step = rem.min(max_h);
            let ok = match step {
                1 => score_rows_sub::<BITS, G, R, 1>(rows, tables4, tables3, widened, head_start, out),
                2 => score_rows_sub::<BITS, G, R, 2>(rows, tables4, tables3, widened, head_start, out),
                3 => score_rows_sub::<BITS, G, R, 3>(rows, tables4, tables3, widened, head_start, out),
                4 => score_rows_sub::<BITS, G, R, 4>(rows, tables4, tables3, widened, head_start, out),
                5 => score_rows_sub::<BITS, G, R, 5>(rows, tables4, tables3, widened, head_start, out),
                _ => score_rows_sub::<BITS, G, R, 6>(rows, tables4, tables3, widened, head_start, out),
            };
            if !ok {
                return false;
            }
            head_start += step;
        }
        true
    }

    #[target_feature(enable = "neon")]
    fn score_rows_with_widened<const BITS: u8, const G: usize>(
        packed_rows: &[u8],
        centroids: &[f32; 16],
        widened: &[[[float64x2_t; 2]; G]; 64],
        out: &mut [[f32; G]],
    ) -> bool {
        let row_bytes = if BITS == 4 { 128 } else { 96 };
        let offsets = tile_offsets();
        let tables4 = (BITS == 4).then(|| FourBitTables {
            table: codebook_bytes(centroids),
            mask_3c: vdupq_n_u8(0x3c),
            offsets,
        });
        let tables3 = (BITS == 3).then(|| ThreeBitTables {
            table: codebook_bytes_3(centroids),
            mask_1c: vdupq_n_u8(0x1c),
            offsets,
        });
        let mut row_idx = 0_usize;
        if G <= 2 {
            let batch_bytes = 4 * row_bytes;
            let mut batches = packed_rows.chunks_exact(batch_bytes);
            for (batch_idx, chunk) in batches.by_ref().enumerate() {
                let pf_off = (batch_idx + 2) * batch_bytes;
                if pf_off + row_bytes <= packed_rows.len() {
                    prefetch_l1(packed_rows[pf_off..].as_ptr());
                    prefetch_l1(packed_rows[pf_off + 64..].as_ptr());
                }
                let rows = [
                    &chunk[..row_bytes],
                    &chunk[row_bytes..2 * row_bytes],
                    &chunk[2 * row_bytes..3 * row_bytes],
                    &chunk[3 * row_bytes..4 * row_bytes],
                ];
                if !score_rows_dispatch_heads::<BITS, G, 4>(
                    rows,
                    tables4,
                    tables3,
                    widened,
                    &mut out[row_idx..row_idx + 4],
                ) {
                    return false;
                }
                row_idx += 4;
            }
            for row in batches.remainder().chunks_exact(row_bytes) {
                if !score_rows_dispatch_heads::<BITS, G, 1>(
                    [row],
                    tables4,
                    tables3,
                    widened,
                    &mut out[row_idx..=row_idx],
                ) {
                    return false;
                }
                row_idx += 1;
            }
        } else {
            let batch_bytes = 2 * row_bytes;
            let mut batches = packed_rows.chunks_exact(batch_bytes);
            for (batch_idx, chunk) in batches.by_ref().enumerate() {
                let pf_off = (batch_idx + 2) * batch_bytes;
                if pf_off + row_bytes <= packed_rows.len() {
                    prefetch_l1(packed_rows[pf_off..].as_ptr());
                    prefetch_l1(packed_rows[pf_off + 64..].as_ptr());
                }
                let rows = [&chunk[..row_bytes], &chunk[row_bytes..2 * row_bytes]];
                if !score_rows_dispatch_heads::<BITS, G, 2>(
                    rows,
                    tables4,
                    tables3,
                    widened,
                    &mut out[row_idx..row_idx + 2],
                ) {
                    return false;
                }
                row_idx += 2;
            }
            for row in batches.remainder().chunks_exact(row_bytes) {
                if !score_rows_dispatch_heads::<BITS, G, 1>(
                    [row],
                    tables4,
                    tables3,
                    widened,
                    &mut out[row_idx..=row_idx],
                ) {
                    return false;
                }
                row_idx += 1;
            }
        }
        true
    }

    pub(super) fn score_rows_group<const G: usize>(
        packed_rows: &[u8],
        bits: u8,
        centroids: &[f32; 16],
        queries: &[&[f32; DIMENSION]; G],
        out: &mut [[f32; G]],
    ) -> bool {
        if bits == 4 {
            // SAFETY: caller detects NEON and validates packed row lengths and finite inputs.
            unsafe { score_rows_group_neon::<4, G>(packed_rows, centroids, queries, out) }
        } else {
            // SAFETY: caller detects NEON and validates 3-bit packed row lengths and finite inputs.
            unsafe { score_rows_group_neon::<3, G>(packed_rows, centroids, queries, out) }
        }
    }

    #[target_feature(enable = "neon")]
    fn score_rows_group_neon<const BITS: u8, const G: usize>(
        packed_rows: &[u8],
        centroids: &[f32; 16],
        queries: &[&[f32; DIMENSION]; G],
        out: &mut [[f32; G]],
    ) -> bool {
        let widened = widen_queries(queries);
        score_rows_with_widened::<BITS, G>(packed_rows, centroids, &widened, out)
    }

    pub(super) fn score_rows_pair<const G: usize>(
        packed_rows: &[u8],
        bits: u8,
        centroids: &[f32; 16],
        first_queries: &[&[f32; DIMENSION]; G],
        second_queries: &[&[f32; DIMENSION]; G],
        out0: &mut [[f32; G]],
        out1: &mut [[f32; G]],
    ) -> bool {
        if bits == 4 {
            // SAFETY: caller detects NEON and validates packed row lengths and finite inputs.
            unsafe {
                score_rows_pair_neon::<4, G>(
                    packed_rows,
                    centroids,
                    first_queries,
                    second_queries,
                    out0,
                    out1,
                )
            }
        } else {
            // SAFETY: caller detects NEON and validates 3-bit packed row lengths and finite inputs.
            unsafe {
                score_rows_pair_neon::<3, G>(
                    packed_rows,
                    centroids,
                    first_queries,
                    second_queries,
                    out0,
                    out1,
                )
            }
        }
    }

    #[target_feature(enable = "neon")]
    fn score_rows_pair_neon<const BITS: u8, const G: usize>(
        packed_rows: &[u8],
        centroids: &[f32; 16],
        first_queries: &[&[f32; DIMENSION]; G],
        second_queries: &[&[f32; DIMENSION]; G],
        out0: &mut [[f32; G]],
        out1: &mut [[f32; G]],
    ) -> bool {
        let widened0 = widen_queries(first_queries);
        let widened1 = widen_queries(second_queries);
        let row_bytes = if BITS == 4 { 128 } else { 96 };
        let chunk_bytes = 64 * row_bytes;
        for ((row_chunk, out0_chunk), out1_chunk) in packed_rows
            .chunks(chunk_bytes)
            .zip(out0.chunks_mut(64))
            .zip(out1.chunks_mut(64))
        {
            if !score_rows_with_widened::<BITS, G>(row_chunk, centroids, &widened0, out0_chunk) {
                return false;
            }
            if !score_rows_with_widened::<BITS, G>(row_chunk, centroids, &widened1, out1_chunk) {
                return false;
            }
        }
        true
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn accumulate_tile_single(
        values: float32x4_t,
        scale: float32x4_t,
        magnitude: uint32x4_t,
        infinity: uint32x4_t,
        finite: &mut uint32x4_t,
        dst: &mut [f32; 4],
    ) {
        // SAFETY: dst is a unique four-element chunk of the staging row.
        let previous = unsafe { vld1q_f32(dst.as_ptr()) };
        let product = vmulq_f32(scale, values);
        let result = vaddq_f32(previous, product);
        let absolute = vandq_u32(vreinterpretq_u32_f32(result), magnitude);
        *finite = vandq_u32(*finite, vcltq_u32(absolute, infinity));
        // SAFETY: dst covers the same unique four-element destination.
        unsafe { vst1q_f32(dst.as_mut_ptr(), result) };
    }

    pub(super) fn accumulate(
        packed: &[u8],
        bits: u8,
        centroids: &[f32; 16],
        scale: f32,
        output: &mut [f32; DIMENSION],
    ) -> bool {
        if bits == 4 {
            // SAFETY: caller detects NEON, validates 256 four-bit codes and
            // finite inputs, and supplies a unique 256-coordinate staging array.
            unsafe { accumulate_neon::<4>(packed, centroids, scale, output) }
        } else {
            // SAFETY: the same caller admits only three- or four-bit codes;
            // this branch has exactly 256 three-bit codes and detected NEON.
            unsafe { accumulate_neon::<3>(packed, centroids, scale, output) }
        }
    }

    #[target_feature(enable = "neon")]
    fn accumulate_neon<const BITS: u8>(
        packed: &[u8],
        centroids: &[f32; 16],
        scale: f32,
        output: &mut [f32; DIMENSION],
    ) -> bool {
        let scale = vdupq_n_f32(scale);
        let magnitude = vdupq_n_u32(0x7fff_ffff);
        let infinity = vdupq_n_u32(0x7f80_0000);
        let mut finite = vdupq_n_u32(u32::MAX);
        let offsets = tile_offsets();
        if BITS == 4 {
            let table = codebook_bytes(centroids);
            let mask_3c = vdupq_n_u8(0x3c);
            for (chunk, out_chunk) in packed
                .as_chunks::<16>()
                .0
                .iter()
                .zip(output.as_chunks_mut::<32>().0)
            {
                let tiles = unpack_4bit_32(chunk, table, mask_3c, offsets);
                for (values, dst) in tiles.into_iter().zip(out_chunk.as_chunks_mut::<4>().0) {
                    accumulate_tile_single(values, scale, magnitude, infinity, &mut finite, dst);
                }
            }
        } else {
            let table = codebook_bytes_3(centroids);
            let mask_1c = vdupq_n_u8(0x1c);
            for (chunk, out_chunk) in packed
                .as_chunks::<48>()
                .0
                .iter()
                .zip(output.as_chunks_mut::<128>().0)
            {
                let pairs = unpack_3bit_128(chunk, mask_1c);
                for (pair, sub_out) in pairs.into_iter().zip(out_chunk.as_chunks_mut::<32>().0) {
                    let tiles = expand_3bit_32(pair, table, offsets);
                    for (values, dst) in tiles.into_iter().zip(sub_out.as_chunks_mut::<4>().0) {
                        accumulate_tile_single(
                            values,
                            scale,
                            magnitude,
                            infinity,
                            &mut finite,
                            dst,
                        );
                    }
                }
            }
        }
        vminvq_u32(finite) != 0
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn accumulate_tile_group<const G: usize>(
        values: float32x4_t,
        base: usize,
        active: Option<&[bool; G]>,
        scale_vecs: &[float32x4_t; G],
        finite: &mut [uint32x4_t; G],
        outputs: &mut [[f32; DIMENSION]; G],
    ) {
        let magnitude = vdupq_n_u32(0x7fff_ffff);
        let infinity = vdupq_n_u32(0x7f80_0000);
        match active {
            None => {
                for head in 0..G {
                    let chunk = &mut outputs[head][base..base + 4];
                    // SAFETY: chunk has 4 initialized floats in the caller's row.
                    let previous = unsafe { vld1q_f32(chunk.as_ptr()) };
                    let product = vmulq_f32(scale_vecs[head], values);
                    let result = vaddq_f32(previous, product);
                    let absolute = vandq_u32(vreinterpretq_u32_f32(result), magnitude);
                    finite[head] = vandq_u32(finite[head], vcltq_u32(absolute, infinity));
                    // SAFETY: chunk covers the same unique four-element destination.
                    unsafe { vst1q_f32(chunk.as_mut_ptr(), result) };
                }
            }
            Some(flags) => {
                for head in 0..G {
                    if !flags[head] {
                        continue;
                    }
                    let chunk = &mut outputs[head][base..base + 4];
                    // SAFETY: chunk has 4 initialized floats in the caller's row.
                    let previous = unsafe { vld1q_f32(chunk.as_ptr()) };
                    let product = vmulq_f32(scale_vecs[head], values);
                    let result = vaddq_f32(previous, product);
                    let absolute = vandq_u32(vreinterpretq_u32_f32(result), magnitude);
                    finite[head] = vandq_u32(finite[head], vcltq_u32(absolute, infinity));
                    // SAFETY: chunk covers the same unique four-element destination.
                    unsafe { vst1q_f32(chunk.as_mut_ptr(), result) };
                }
            }
        }
    }

    pub(super) fn accumulate_group<const G: usize>(
        packed: &[u8],
        bits: u8,
        centroids: &[f32; 16],
        scales: [f32; G],
        active: [bool; G],
        outputs: &mut [[f32; DIMENSION]; G],
    ) -> [bool; G] {
        if bits == 4 {
            // SAFETY: caller detects NEON, validates 256 four-bit codes and
            // finite inputs, and supplies disjoint 256-coordinate staging rows.
            unsafe { accumulate_group_neon::<4, G>(packed, centroids, scales, active, outputs) }
        } else {
            // SAFETY: the same caller admits only three- or four-bit codes;
            // this branch has exactly 256 three-bit codes and detected NEON.
            unsafe { accumulate_group_neon::<3, G>(packed, centroids, scales, active, outputs) }
        }
    }

    #[target_feature(enable = "neon")]
    fn accumulate_group_neon<const BITS: u8, const G: usize>(
        packed: &[u8],
        centroids: &[f32; 16],
        scales: [f32; G],
        active: [bool; G],
        outputs: &mut [[f32; DIMENSION]; G],
    ) -> [bool; G] {
        let mut scale_vecs = [vdupq_n_f32(0.0); G];
        for (dst, &scale) in scale_vecs.iter_mut().zip(&scales) {
            *dst = vdupq_n_f32(scale);
        }
        let mut finite: [_; G] = std::array::from_fn(|_| vdupq_n_u32(u32::MAX));
        let active_flags = (!active.iter().all(|&flag| flag)).then_some(&active);
        let offsets = tile_offsets();
        if BITS == 4 {
            let table = codebook_bytes(centroids);
            let mask_3c = vdupq_n_u8(0x3c);
            for (chunk_idx, chunk) in packed.as_chunks::<16>().0.iter().take(8).enumerate() {
                let tiles = unpack_4bit_32(chunk, table, mask_3c, offsets);
                let chunk_base = chunk_idx * 32;
                for (tile_idx, values) in tiles.into_iter().enumerate() {
                    accumulate_tile_group(
                        values,
                        chunk_base + tile_idx * 4,
                        active_flags,
                        &scale_vecs,
                        &mut finite,
                        outputs,
                    );
                }
            }
        } else {
            let table = codebook_bytes_3(centroids);
            let mask_1c = vdupq_n_u8(0x1c);
            for (chunk_idx, chunk) in packed.as_chunks::<48>().0.iter().take(2).enumerate() {
                let pairs = unpack_3bit_128(chunk, mask_1c);
                let chunk_base = chunk_idx * 128;
                for (pair_idx, pair) in pairs.into_iter().enumerate() {
                    let tiles = expand_3bit_32(pair, table, offsets);
                    let pair_base = chunk_base + pair_idx * 32;
                    for (tile_idx, values) in tiles.into_iter().enumerate() {
                        accumulate_tile_group(
                            values,
                            pair_base + tile_idx * 4,
                            active_flags,
                            &scale_vecs,
                            &mut finite,
                            outputs,
                        );
                    }
                }
            }
        }
        std::array::from_fn(|head| active[head] && vminvq_u32(finite[head]) != 0)
    }

    pub(super) fn accumulate_group_pair<const G: usize>(
        packed: &[u8],
        bits: u8,
        centroids: &[f32; 16],
        scales: [[f32; G]; 2],
        active: [[bool; G]; 2],
        outputs: &mut [[[f32; DIMENSION]; G]; 2],
    ) -> ([bool; G], [bool; G]) {
        if bits == 4 {
            // SAFETY: caller detects NEON and validates 256 four-bit codes and outputs.
            unsafe {
                accumulate_group_pair_neon::<4, G>(packed, centroids, scales, active, outputs)
            }
        } else {
            // SAFETY: caller detects NEON and validates 256 three-bit codes and outputs.
            unsafe {
                accumulate_group_pair_neon::<3, G>(packed, centroids, scales, active, outputs)
            }
        }
    }

    #[target_feature(enable = "neon")]
    fn accumulate_group_pair_neon<const BITS: u8, const G: usize>(
        packed: &[u8],
        centroids: &[f32; 16],
        scales: [[f32; G]; 2],
        active: [[bool; G]; 2],
        outputs: &mut [[[f32; DIMENSION]; G]; 2],
    ) -> ([bool; G], [bool; G]) {
        let mut first_scale_vecs = [vdupq_n_f32(0.0); G];
        let mut second_scale_vecs = [vdupq_n_f32(0.0); G];
        for (dst, &scale) in first_scale_vecs.iter_mut().zip(&scales[0]) {
            *dst = vdupq_n_f32(scale);
        }
        for (dst, &scale) in second_scale_vecs.iter_mut().zip(&scales[1]) {
            *dst = vdupq_n_f32(scale);
        }
        let mut finite0: [_; G] = std::array::from_fn(|_| vdupq_n_u32(u32::MAX));
        let mut finite1: [_; G] = std::array::from_fn(|_| vdupq_n_u32(u32::MAX));
        let active_flags0 = (!active[0].iter().all(|&flag| flag)).then_some(&active[0]);
        let active_flags1 = (!active[1].iter().all(|&flag| flag)).then_some(&active[1]);
        let offsets = tile_offsets();
        if BITS == 4 {
            let table = codebook_bytes(centroids);
            let mask_3c = vdupq_n_u8(0x3c);
            for (chunk_idx, chunk) in packed.as_chunks::<16>().0.iter().take(8).enumerate() {
                let tiles = unpack_4bit_32(chunk, table, mask_3c, offsets);
                let chunk_base = chunk_idx * 32;
                for (tile_idx, values) in tiles.into_iter().enumerate() {
                    let base = chunk_base + tile_idx * 4;
                    accumulate_tile_group(
                        values,
                        base,
                        active_flags0,
                        &first_scale_vecs,
                        &mut finite0,
                        &mut outputs[0],
                    );
                    accumulate_tile_group(
                        values,
                        base,
                        active_flags1,
                        &second_scale_vecs,
                        &mut finite1,
                        &mut outputs[1],
                    );
                }
            }
        } else {
            let table = codebook_bytes_3(centroids);
            let mask_1c = vdupq_n_u8(0x1c);
            for (chunk_idx, chunk) in packed.as_chunks::<48>().0.iter().take(2).enumerate() {
                let pairs = unpack_3bit_128(chunk, mask_1c);
                let chunk_base = chunk_idx * 128;
                for (pair_idx, pair) in pairs.into_iter().enumerate() {
                    let tiles = expand_3bit_32(pair, table, offsets);
                    let pair_base = chunk_base + pair_idx * 32;
                    for (tile_idx, values) in tiles.into_iter().enumerate() {
                        let base = pair_base + tile_idx * 4;
                        accumulate_tile_group(
                            values,
                            base,
                            active_flags0,
                            &first_scale_vecs,
                            &mut finite0,
                            &mut outputs[0],
                        );
                        accumulate_tile_group(
                            values,
                            base,
                            active_flags1,
                            &second_scale_vecs,
                            &mut finite1,
                            &mut outputs[1],
                        );
                    }
                }
            }
        }
        (
            std::array::from_fn(|head| active[0][head] && vminvq_u32(finite0[head]) != 0),
            std::array::from_fn(|head| active[1][head] && vminvq_u32(finite1[head]) != 0),
        )
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn unpack_4bit_8(bytes: &[u8], tables: FourBitTables) -> [float32x4_t; 2] {
        let word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let packed = vreinterpretq_u8_u32(vdupq_n_u32(word));
        let low_codes = vandq_u8(vshlq_n_u8::<2>(packed), tables.mask_3c);
        let high_codes = vandq_u8(vshrq_n_u8::<2>(packed), tables.mask_3c);
        let interleaved = vzip1q_u8(low_codes, high_codes);
        let dup = vzip1q_u8(interleaved, interleaved);
        [
            vreinterpretq_f32_u8(vqtbl4q_u8(
                tables.table,
                vorrq_u8(vzip1q_u8(dup, dup), tables.offsets),
            )),
            vreinterpretq_f32_u8(vqtbl4q_u8(
                tables.table,
                vorrq_u8(vzip2q_u8(dup, dup), tables.offsets),
            )),
        ]
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn unpack_4bit_16(bytes: &[u8], tables: FourBitTables) -> [float32x4_t; 4] {
        let word = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        let packed = vreinterpretq_u8_u64(vdupq_n_u64(word));
        let low_codes = vandq_u8(vshlq_n_u8::<2>(packed), tables.mask_3c);
        let high_codes = vandq_u8(vshrq_n_u8::<2>(packed), tables.mask_3c);
        expand_4bit_16(
            vzip1q_u8(low_codes, high_codes),
            tables.table,
            tables.offsets,
        )
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn unpack_3bit_8(bytes: &[u8], tables: ThreeBitStripTables) -> [float32x4_t; 2] {
        let word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], 0]);
        let vec = vdupq_n_u32(word);
        let codes0 = vandq_u32(vshlq_u32(vec, tables.sh_lo), tables.mask_7);
        let codes1 = vandq_u32(vshlq_u32(vec, tables.sh_hi), tables.mask_7);
        let bases0 = vreinterpretq_u8_u32(vmulq_u32(codes0, tables.mul_04));
        let bases1 = vreinterpretq_u8_u32(vmulq_u32(codes1, tables.mul_04));
        [
            vreinterpretq_f32_u8(vqtbl2q_u8(
                tables.table,
                vorrq_u8(bases0, tables.offsets),
            )),
            vreinterpretq_f32_u8(vqtbl2q_u8(
                tables.table,
                vorrq_u8(bases1, tables.offsets),
            )),
        ]
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn unpack_3bit_16(bytes: &[u8], tables: ThreeBitStripTables) -> [float32x4_t; 4] {
        let lo = unpack_3bit_8(&bytes[0..3], tables);
        let hi = unpack_3bit_8(&bytes[3..6], tables);
        [lo[0], lo[1], hi[0], hi[1]]
    }

    #[target_feature(enable = "neon")]
    fn accumulate_strip_batch<
        const BITS: u8,
        const G: usize,
        const TILES: usize,
        const H: usize,
    >(
        row_chunk: &[u8],
        scales: &[[f32; G]],
        head_start: usize,
        tables4: Option<FourBitTables>,
        tables3: Option<ThreeBitStripTables>,
        outputs: &mut [[f32; DIMENSION]; G],
    ) -> bool {
        let row_bytes = if BITS == 4 { 128 } else { 96 };
        let strip_chans = TILES * 4;
        let strip_bytes = if BITS == 4 {
            strip_chans / 2
        } else {
            (strip_chans * 3) / 8
        };
        let strips = DIMENSION / strip_chans;
        let magnitude = vdupq_n_u32(0x7fff_ffff);
        let infinity = vdupq_n_u32(0x7f80_0000);
        for strip in 0..strips {
            let chan_base = strip * strip_chans;
            let byte_off = strip * strip_bytes;
            let mut acc: [[float32x4_t; TILES]; H] = std::array::from_fn(|h| {
                std::array::from_fn(|t| {
                    // SAFETY: chan_base + t * 4 + 4 <= DIMENSION.
                    unsafe { vld1q_f32(outputs[head_start + h][chan_base + t * 4..].as_ptr()) }
                })
            });
            for (r, (row, row_scales)) in row_chunk
                .chunks_exact(row_bytes)
                .zip(scales)
                .enumerate()
            {
                if r + 4 < scales.len() {
                    prefetch_l1(row_chunk[(r + 4) * row_bytes + byte_off..].as_ptr());
                }
                let slice = &row[byte_off..byte_off + strip_bytes];
                let tiles: [float32x4_t; TILES] = if BITS == 4 {
                    let Some(t4) = tables4 else {
                        return false;
                    };
                    if TILES == 4 {
                        let unpacked = unpack_4bit_16(slice, t4);
                        std::array::from_fn(|t| unpacked[t])
                    } else {
                        let unpacked = unpack_4bit_8(slice, t4);
                        std::array::from_fn(|t| unpacked[t])
                    }
                } else {
                    let Some(t3) = tables3 else {
                        return false;
                    };
                    if TILES == 4 {
                        let unpacked = unpack_3bit_16(slice, t3);
                        std::array::from_fn(|t| unpacked[t])
                    } else {
                        let unpacked = unpack_3bit_8(slice, t3);
                        std::array::from_fn(|t| unpacked[t])
                    }
                };
                for h in 0..H {
                    let scale_vec = vdupq_n_f32(row_scales[head_start + h]);
                    for t in 0..TILES {
                        acc[h][t] = vaddq_f32(acc[h][t], vmulq_f32(scale_vec, tiles[t]));
                    }
                }
            }
            let mut strip_finite = vdupq_n_u32(u32::MAX);
            for head_acc in &acc {
                for &vec in head_acc {
                    let abs = vandq_u32(vreinterpretq_u32_f32(vec), magnitude);
                    strip_finite = vandq_u32(strip_finite, vcltq_u32(abs, infinity));
                }
            }
            if vminvq_u32(strip_finite) == 0 {
                return false;
            }
            for (h, head_acc) in acc.iter().enumerate() {
                for (t, &vec) in head_acc.iter().enumerate() {
                    // SAFETY: chan_base + t * 4 + 4 <= DIMENSION.
                    unsafe {
                        vst1q_f32(
                            outputs[head_start + h][chan_base + t * 4..].as_mut_ptr(),
                            vec,
                        );
                    }
                }
            }
        }
        true
    }

    pub(super) fn accumulate_rows_group<const G: usize>(
        packed_rows: &[u8],
        bits: u8,
        centroids: &[f32; 16],
        scales: &[[f32; G]],
        outputs: &mut [[f32; DIMENSION]; G],
    ) -> bool {
        if bits == 4 {
            // SAFETY: caller detects NEON and validates packed row lengths and finite inputs.
            unsafe { accumulate_rows_group_neon::<4, G>(packed_rows, centroids, scales, outputs) }
        } else {
            // SAFETY: caller detects NEON and validates 3-bit packed row lengths and finite inputs.
            unsafe { accumulate_rows_group_neon::<3, G>(packed_rows, centroids, scales, outputs) }
        }
    }

    #[target_feature(enable = "neon")]
    fn accumulate_chunk_neon<const BITS: u8, const G: usize>(
        row_chunk: &[u8],
        scale_chunk: &[[f32; G]],
        tables4: Option<FourBitTables>,
        tables3: Option<ThreeBitStripTables>,
        outputs: &mut [[f32; DIMENSION]; G],
    ) -> bool {
        if G <= 4 {
            return match G {
                1 => accumulate_strip_batch::<BITS, G, 4, 1>(
                    row_chunk,
                    scale_chunk,
                    0,
                    tables4,
                    tables3,
                    outputs,
                ),
                2 => accumulate_strip_batch::<BITS, G, 4, 2>(
                    row_chunk,
                    scale_chunk,
                    0,
                    tables4,
                    tables3,
                    outputs,
                ),
                3 => accumulate_strip_batch::<BITS, G, 4, 3>(
                    row_chunk,
                    scale_chunk,
                    0,
                    tables4,
                    tables3,
                    outputs,
                ),
                _ => accumulate_strip_batch::<BITS, G, 4, 4>(
                    row_chunk,
                    scale_chunk,
                    0,
                    tables4,
                    tables3,
                    outputs,
                ),
            };
        }
        let mut head_start = 0_usize;
        while head_start < G {
            let step = (G - head_start).min(6);
            let ok = match step {
                1 => accumulate_strip_batch::<BITS, G, 2, 1>(
                    row_chunk,
                    scale_chunk,
                    head_start,
                    tables4,
                    tables3,
                    outputs,
                ),
                2 => accumulate_strip_batch::<BITS, G, 2, 2>(
                    row_chunk,
                    scale_chunk,
                    head_start,
                    tables4,
                    tables3,
                    outputs,
                ),
                3 => accumulate_strip_batch::<BITS, G, 2, 3>(
                    row_chunk,
                    scale_chunk,
                    head_start,
                    tables4,
                    tables3,
                    outputs,
                ),
                4 => accumulate_strip_batch::<BITS, G, 2, 4>(
                    row_chunk,
                    scale_chunk,
                    head_start,
                    tables4,
                    tables3,
                    outputs,
                ),
                5 => accumulate_strip_batch::<BITS, G, 2, 5>(
                    row_chunk,
                    scale_chunk,
                    head_start,
                    tables4,
                    tables3,
                    outputs,
                ),
                _ => accumulate_strip_batch::<BITS, G, 2, 6>(
                    row_chunk,
                    scale_chunk,
                    head_start,
                    tables4,
                    tables3,
                    outputs,
                ),
            };
            if !ok {
                return false;
            }
            head_start += step;
        }
        true
    }

    #[target_feature(enable = "neon")]
    fn accumulate_rows_group_neon<const BITS: u8, const G: usize>(
        packed_rows: &[u8],
        centroids: &[f32; 16],
        scales: &[[f32; G]],
        outputs: &mut [[f32; DIMENSION]; G],
    ) -> bool {
        let row_bytes = if BITS == 4 { 128 } else { 96 };
        let offsets = tile_offsets();
        let tables4 = (BITS == 4).then(|| FourBitTables {
            table: codebook_bytes(centroids),
            mask_3c: vdupq_n_u8(0x3c),
            offsets,
        });
        let tables3 = (BITS == 3).then(|| ThreeBitStripTables {
            table: codebook_bytes_3(centroids),
            // SAFETY: THREE_BIT_UNPACK_SHIFTS_LO is a static 4-element initialized array.
            sh_lo: unsafe { vld1q_s32(THREE_BIT_UNPACK_SHIFTS_LO.as_ptr()) },
            // SAFETY: THREE_BIT_UNPACK_SHIFTS_HI is a static 4-element initialized array.
            sh_hi: unsafe { vld1q_s32(THREE_BIT_UNPACK_SHIFTS_HI.as_ptr()) },
            mask_7: vdupq_n_u32(7),
            mul_04: vdupq_n_u32(0x0404_0404),
            offsets,
        });
        for (row_chunk, scale_chunk) in packed_rows.chunks(64 * row_bytes).zip(scales.chunks(64)) {
            if !accumulate_chunk_neon::<BITS, G>(row_chunk, scale_chunk, tables4, tables3, outputs) {
                return false;
            }
        }
        true
    }

    pub(super) fn accumulate_rows_group_pair<const G: usize>(
        packed_rows: &[u8],
        bits: u8,
        centroids: &[f32; 16],
        scales0: &[[f32; G]],
        scales1: &[[f32; G]],
        outputs: &mut [[[f32; DIMENSION]; G]; 2],
    ) -> bool {
        if bits == 4 {
            // SAFETY: caller detects NEON and validates packed row lengths and finite inputs.
            unsafe {
                accumulate_rows_group_pair_neon::<4, G>(
                    packed_rows,
                    centroids,
                    scales0,
                    scales1,
                    outputs,
                )
            }
        } else {
            // SAFETY: caller detects NEON and validates 3-bit packed row lengths and finite inputs.
            unsafe {
                accumulate_rows_group_pair_neon::<3, G>(
                    packed_rows,
                    centroids,
                    scales0,
                    scales1,
                    outputs,
                )
            }
        }
    }

    #[target_feature(enable = "neon")]
    fn accumulate_rows_group_pair_neon<const BITS: u8, const G: usize>(
        packed_rows: &[u8],
        centroids: &[f32; 16],
        scales0: &[[f32; G]],
        scales1: &[[f32; G]],
        outputs: &mut [[[f32; DIMENSION]; G]; 2],
    ) -> bool {
        let row_bytes = if BITS == 4 { 128 } else { 96 };
        let chunk_bytes = 64 * row_bytes;
        for ((row_chunk, s0_chunk), s1_chunk) in packed_rows
            .chunks(chunk_bytes)
            .zip(scales0.chunks(64))
            .zip(scales1.chunks(64))
        {
            if !accumulate_rows_group_neon::<BITS, G>(
                row_chunk,
                centroids,
                s0_chunk,
                &mut outputs[0],
            ) {
                return false;
            }
            if !accumulate_rows_group_neon::<BITS, G>(
                row_chunk,
                centroids,
                s1_chunk,
                &mut outputs[1],
            ) {
                return false;
            }
        }
        true
    }

    pub(super) fn normalize(values: &mut [f32; DIMENSION], scale: f32) -> bool {
        // SAFETY: caller detects baseline NEON and passes a unique 256-float array.
        unsafe { normalize_neon(values, scale) }
    }

    #[target_feature(enable = "neon")]
    fn normalize_neon(values: &mut [f32; DIMENSION], scale: f32) -> bool {
        let mut staged = [0.0_f32; DIMENSION];
        let scale_f64 = vdupq_n_f64(f64::from(scale));
        let magnitude = vdupq_n_u32(0x7fff_ffff);
        let infinity = vdupq_n_u32(0x7f80_0000);
        let mut finite = vdupq_n_u32(u32::MAX);
        for (src, dst) in values
            .as_chunks::<4>()
            .0
            .iter()
            .zip(staged.as_chunks_mut::<4>().0)
        {
            // SAFETY: src is a four-element array slice of initialized coordinates.
            let vec = unsafe { vld1q_f32(src.as_ptr()) };
            let in_abs = vandq_u32(vreinterpretq_u32_f32(vec), magnitude);
            finite = vandq_u32(finite, vcltq_u32(in_abs, infinity));
            let low = vdivq_f64(vcvt_f64_f32(vget_low_f32(vec)), scale_f64);
            let high = vdivq_f64(vcvt_high_f64_f32(vec), scale_f64);
            let narrowed = vcvt_high_f32_f64(vcvt_f32_f64(low), high);
            let out_abs = vandq_u32(vreinterpretq_u32_f32(narrowed), magnitude);
            finite = vandq_u32(finite, vcltq_u32(out_abs, infinity));
            // SAFETY: dst is a unique four-element slice of the staging buffer.
            unsafe { vst1q_f32(dst.as_mut_ptr(), narrowed) };
        }
        if vminvq_u32(finite) == 0 {
            return false;
        }
        *values = staged;
        true
    }

    pub(super) fn quantize_pack(
        values: &[f32; DIMENSION],
        bits: u8,
        centroids: &[f32; 16],
        packed: &mut [u8],
    ) -> bool {
        if bits == 4 {
            // SAFETY: caller detects NEON and validates 128-byte output and finite codebook.
            unsafe { quantize_pack_neon::<4>(values, centroids, packed) }
        } else {
            // SAFETY: caller detects NEON and validates 96-byte output and finite codebook.
            unsafe { quantize_pack_neon::<3>(values, centroids, packed) }
        }
    }

    struct NearestState {
        dist0: float32x4_t,
        dist1: float32x4_t,
        code0: uint32x4_t,
        code1: uint32x4_t,
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn step_nearest(
        vals0: float32x4_t,
        vals1: float32x4_t,
        state: &mut NearestState,
        centroid: float32x4_t,
        code: u32,
    ) {
        let cand = vdupq_n_u32(code);
        let dist0 = vabdq_f32(vals0, centroid);
        let dist1 = vabdq_f32(vals1, centroid);
        let lt0 = vcltq_f32(dist0, state.dist0);
        let lt1 = vcltq_f32(dist1, state.dist1);
        state.dist0 = vbslq_f32(lt0, dist0, state.dist0);
        state.dist1 = vbslq_f32(lt1, dist1, state.dist1);
        state.code0 = vbslq_u32(lt0, cand, state.code0);
        state.code1 = vbslq_u32(lt1, cand, state.code1);
    }

    #[target_feature(enable = "neon")]
    fn quantize_pack_neon<const BITS: u8>(
        values: &[f32; DIMENSION],
        centroids: &[f32; 16],
        packed: &mut [u8],
    ) -> bool {
        let mut staged = [0_u8; DIMENSION / 2];
        // SAFETY: centroids[0..16] holds sixteen initialized binary32 values.
        let c0 = unsafe { vld1q_f32(centroids.as_ptr()) };
        // SAFETY: centroids[4..8] is initialized within the sixteen-element array.
        let c1 = unsafe { vld1q_f32(centroids[4..].as_ptr()) };
        // SAFETY: centroids[8..12] is initialized within the sixteen-element array.
        let c2 = unsafe { vld1q_f32(centroids[8..].as_ptr()) };
        // SAFETY: centroids[12..16] is initialized within the sixteen-element array.
        let c3 = unsafe { vld1q_f32(centroids[12..].as_ptr()) };
        let sh_lo_arr = if BITS == 4 {
            &FOUR_BIT_PACK_SHIFTS_LO
        } else {
            &THREE_BIT_PACK_SHIFTS_LO
        };
        let sh_hi_arr = if BITS == 4 {
            &FOUR_BIT_PACK_SHIFTS_HI
        } else {
            &THREE_BIT_PACK_SHIFTS_HI
        };
        // SAFETY: static four-element shift array is initialized.
        let sh_lo = unsafe { vld1q_s32(sh_lo_arr.as_ptr()) };
        // SAFETY: static four-element shift array is initialized.
        let sh_hi = unsafe { vld1q_s32(sh_hi_arr.as_ptr()) };
        let magnitude = vdupq_n_u32(0x7fff_ffff);
        let infinity = vdupq_n_u32(0x7f80_0000);
        let mut finite = vdupq_n_u32(u32::MAX);
        let c0_lane0 = vdupq_laneq_f32::<0>(c0);
        let byte_stride = usize::from(BITS);

        for (block_idx, block) in values.as_chunks::<8>().0.iter().enumerate() {
            // SAFETY: block holds eight initialized coordinates; loading first four.
            let vals0 = unsafe { vld1q_f32(block.as_ptr()) };
            // SAFETY: block[4..8] holds the remaining four initialized coordinates.
            let vals1 = unsafe { vld1q_f32(block[4..].as_ptr()) };
            let abs0 = vandq_u32(vreinterpretq_u32_f32(vals0), magnitude);
            let abs1 = vandq_u32(vreinterpretq_u32_f32(vals1), magnitude);
            finite = vandq_u32(finite, vcltq_u32(abs0, infinity));
            finite = vandq_u32(finite, vcltq_u32(abs1, infinity));

            let mut state = NearestState {
                dist0: vabdq_f32(vals0, c0_lane0),
                dist1: vabdq_f32(vals1, c0_lane0),
                code0: vdupq_n_u32(0),
                code1: vdupq_n_u32(0),
            };

            step_nearest(vals0, vals1, &mut state, vdupq_laneq_f32::<1>(c0), 1);
            step_nearest(vals0, vals1, &mut state, vdupq_laneq_f32::<2>(c0), 2);
            step_nearest(vals0, vals1, &mut state, vdupq_laneq_f32::<3>(c0), 3);
            step_nearest(vals0, vals1, &mut state, vdupq_laneq_f32::<0>(c1), 4);
            step_nearest(vals0, vals1, &mut state, vdupq_laneq_f32::<1>(c1), 5);
            step_nearest(vals0, vals1, &mut state, vdupq_laneq_f32::<2>(c1), 6);
            step_nearest(vals0, vals1, &mut state, vdupq_laneq_f32::<3>(c1), 7);
            if BITS == 4 {
                step_nearest(vals0, vals1, &mut state, vdupq_laneq_f32::<0>(c2), 8);
                step_nearest(vals0, vals1, &mut state, vdupq_laneq_f32::<1>(c2), 9);
                step_nearest(vals0, vals1, &mut state, vdupq_laneq_f32::<2>(c2), 10);
                step_nearest(vals0, vals1, &mut state, vdupq_laneq_f32::<3>(c2), 11);
                step_nearest(vals0, vals1, &mut state, vdupq_laneq_f32::<0>(c3), 12);
                step_nearest(vals0, vals1, &mut state, vdupq_laneq_f32::<1>(c3), 13);
                step_nearest(vals0, vals1, &mut state, vdupq_laneq_f32::<2>(c3), 14);
                step_nearest(vals0, vals1, &mut state, vdupq_laneq_f32::<3>(c3), 15);
            }

            let packed_word = vaddvq_u32(vorrq_u32(
                vshlq_u32(state.code0, sh_lo),
                vshlq_u32(state.code1, sh_hi),
            ));
            let bytes = packed_word.to_le_bytes();
            let offset = block_idx * byte_stride;
            staged[offset..offset + byte_stride].copy_from_slice(&bytes[..byte_stride]);
        }
        if vminvq_u32(finite) == 0 {
            return false;
        }
        let expected_bytes = DIMENSION * usize::from(BITS) / 8;
        packed.copy_from_slice(&staged[..expected_bytes]);
        true
    }

    #[cfg(test)]
    #[inline]
    #[target_feature(enable = "neon")]
    fn four_bit_tile(packed: &[u8], index: usize, table: uint8x16x4_t) -> float32x4_t {
        let byte = index * 2;
        let pair = u16::from_le_bytes([packed[byte], packed[byte + 1]]);
        // SAFETY: this initialized fixed array supplies exactly four shifts.
        let shifts = unsafe { vld1q_s32([0, -4, -8, -12].as_ptr()) };
        let codes = vandq_u32(
            vshlq_u32(vdupq_n_u32(u32::from(pair)), shifts),
            vdupq_n_u32(15),
        );
        // Each nibble selects four consecutive bytes of its binary32 centroid.
        // Multiplication broadcasts code * 4 into every byte of its lane; those
        // bases are multiples of four, so OR appends the offsets 0, 1, 2, 3.
        let bases = vreinterpretq_u8_u32(vmulq_u32(codes, vdupq_n_u32(0x0404_0404)));
        // SAFETY: the initialized array supplies sixteen byte offsets. Loading
        // explicit bytes keeps the table order independent of integer endianness.
        let offsets =
            unsafe { vld1q_u8([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3].as_ptr()) };
        vreinterpretq_f32_u8(vqtbl4q_u8(table, vorrq_u8(bases, offsets)))
    }

    pub(super) fn hadamard(values: &mut [f32; DIMENSION]) -> bool {
        // SAFETY: caller detects baseline NEON and supplies one unique fixed
        // 256-coordinate staging array, retained entirely within this call.
        unsafe { hadamard_neon(values) }
    }

    #[target_feature(enable = "neon")]
    fn hadamard_neon(values: &mut [f32; DIMENSION]) -> bool {
        // Deinterleaving places the same coordinate from four independent
        // four-value blocks in each vector. Preserve the first two butterfly
        // stages in registers, with exactly their original add/subtract order.
        for chunk in values.as_chunks_mut::<16>().0 {
            // SAFETY: the unique chunk contains sixteen initialized values;
            // the four-way unaligned load reads exactly those coordinates.
            let input = unsafe { vld4q_f32(chunk.as_ptr()) };
            let first_sum = vaddq_f32(input.0, input.1);
            let first_difference = vsubq_f32(input.0, input.1);
            let second_sum = vaddq_f32(input.2, input.3);
            let second_difference = vsubq_f32(input.2, input.3);
            let output = float32x4x4_t(
                vaddq_f32(first_sum, second_sum),
                vaddq_f32(first_difference, second_difference),
                vsubq_f32(first_sum, second_sum),
                vsubq_f32(first_difference, second_difference),
            );
            // SAFETY: the same unique sixteen-value chunk admits all four
            // interleaved result vectors, restoring the original row order.
            unsafe { vst4q_f32(chunk.as_mut_ptr(), output) };
        }
        let mut half = 4;
        while half < DIMENSION / 2 {
            for block in values.chunks_exact_mut(half * 2) {
                let (left, right) = block.split_at_mut(half);
                for (left, right) in left
                    .as_chunks_mut::<4>()
                    .0
                    .iter_mut()
                    .zip(right.as_chunks_mut::<4>().0)
                {
                    // SAFETY: both disjoint chunks contain four initialized
                    // values and remain valid until their corresponding stores.
                    let a = unsafe { vld1q_f32(left.as_ptr()) };
                    // SAFETY: right is the disjoint valid four-element chunk.
                    let b = unsafe { vld1q_f32(right.as_ptr()) };
                    let sum = vaddq_f32(a, b);
                    let difference = vsubq_f32(a, b);
                    // SAFETY: left is a unique valid four-element output chunk.
                    unsafe { vst1q_f32(left.as_mut_ptr(), sum) };
                    // SAFETY: right is disjoint and covers four output values.
                    unsafe { vst1q_f32(right.as_mut_ptr(), difference) };
                }
            }
            half *= 2;
        }
        // The final half-width-128 butterfly already visits every output.
        // Normalize and check each result before its only store, retaining the
        // separate binary32 addition/subtraction and multiplication boundary.
        let normalization = vdupq_n_f32(0.0625);
        let magnitude = vdupq_n_u32(0x7fff_ffff);
        let infinity = vdupq_n_u32(0x7f80_0000);
        let mut finite = vdupq_n_u32(u32::MAX);
        let (left, right) = values.split_at_mut(DIMENSION / 2);
        for (left, right) in left
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(right.as_chunks_mut::<4>().0)
        {
            // SAFETY: left and right are disjoint initialized four-value
            // chunks of the unique row and stay valid through both stores.
            let a = unsafe { vld1q_f32(left.as_ptr()) };
            // SAFETY: right supplies its own disjoint initialized vector.
            let b = unsafe { vld1q_f32(right.as_ptr()) };
            let sum = vmulq_f32(vaddq_f32(a, b), normalization);
            let difference = vmulq_f32(vsubq_f32(a, b), normalization);
            let sum_magnitude = vandq_u32(vreinterpretq_u32_f32(sum), magnitude);
            let difference_magnitude = vandq_u32(vreinterpretq_u32_f32(difference), magnitude);
            finite = vandq_u32(finite, vcltq_u32(sum_magnitude, infinity));
            finite = vandq_u32(finite, vcltq_u32(difference_magnitude, infinity));
            // SAFETY: left remains a unique four-value output chunk.
            unsafe { vst1q_f32(left.as_mut_ptr(), sum) };
            // SAFETY: right is disjoint and has room for its four outputs.
            unsafe { vst1q_f32(right.as_mut_ptr(), difference) };
        }
        vminvq_u32(finite) != 0
    }

    #[cfg(test)]
    mod tests {
        #[test]
        fn four_bit_lookup_preserves_every_nibble_lane_and_centroid_bits() {
            if !std::arch::is_aarch64_feature_detected!("neon") {
                return;
            }
            let centroids = [
                -f32::MAX,
                -17.25,
                -1.0,
                -f32::MIN_POSITIVE,
                -f32::from_bits(1),
                -0.0,
                0.0,
                f32::from_bits(1),
                f32::MIN_POSITIVE,
                0.125,
                0.5,
                1.0,
                3.25,
                17.25,
                f32::MAX * 0.5,
                f32::MAX,
            ];
            // SAFETY: NEON was detected above and the array owns all 64 bytes.
            let table = unsafe { super::codebook_bytes(&centroids) };
            for offset in 0_u8..16 {
                let codes: [u8; super::DIMENSION] =
                    std::array::from_fn(|index| index.to_le_bytes()[0].wrapping_add(offset) & 15);
                // Independent bit-at-a-time packing exercises both nibble
                // positions and all four output lanes, including the last tile.
                let mut packed = [0_u8; super::DIMENSION / 2];
                for (index, code) in codes.iter().enumerate() {
                    for bit in 0..4 {
                        let position = index * 4 + bit;
                        packed[position / 8] |= ((code >> bit) & 1) << (position % 8);
                    }
                }
                for (index, codes) in codes.as_chunks::<4>().0.iter().enumerate() {
                    // SAFETY: NEON was detected, packed owns 128 initialized
                    // bytes, and the enumerated tile is within its 64 tiles.
                    let tile = unsafe { super::four_bit_tile(&packed, index, table) };
                    let mut decoded = [0.0; 4];
                    // SAFETY: decoded is a unique initialized four-value array.
                    unsafe { super::vst1q_f32(decoded.as_mut_ptr(), tile) };
                    assert_eq!(
                        decoded.map(f32::to_bits),
                        codes.map(|code| centroids[usize::from(code)].to_bits()),
                    );
                }
            }
        }

        #[test]
        fn three_bit_and_four_bit_vector_unpackers_preserve_every_lane_and_centroid_bits() {
            if !std::arch::is_aarch64_feature_detected!("neon") {
                return;
            }
            let centroids = [
                -f32::MAX,
                -17.25,
                -1.0,
                -f32::MIN_POSITIVE,
                -f32::from_bits(1),
                -0.0,
                0.0,
                f32::MAX,
                f32::MIN_POSITIVE,
                0.125,
                0.5,
                1.0,
                3.25,
                17.25,
                f32::MAX * 0.5,
                -2.0,
            ];
            for offset in 0_u8..8 {
                let codes: [u8; super::DIMENSION] =
                    std::array::from_fn(|index| index.to_le_bytes()[0].wrapping_add(offset) & 7);
                let mut packed = [0_u8; super::DIMENSION * 3 / 8];
                for (index, code) in codes.iter().enumerate() {
                    for bit in 0..3 {
                        let position = index * 3 + bit;
                        packed[position / 8] |= ((code >> bit) & 1) << (position % 8);
                    }
                }
                let mut actual = [0.0_f32; super::DIMENSION];
                // SAFETY: NEON was detected above and packed has 96 initialized bytes.
                unsafe { super::decode_row::<3>(&packed, &centroids, &mut actual) };
                let expected: [u32; super::DIMENSION] =
                    std::array::from_fn(|index| centroids[usize::from(codes[index])].to_bits());
                assert_eq!(actual.map(f32::to_bits), expected);
            }
            for offset in 0_u8..16 {
                let codes: [u8; super::DIMENSION] =
                    std::array::from_fn(|index| index.to_le_bytes()[0].wrapping_add(offset) & 15);
                let mut packed = [0_u8; super::DIMENSION / 2];
                for (index, code) in codes.iter().enumerate() {
                    for bit in 0..4 {
                        let position = index * 4 + bit;
                        packed[position / 8] |= ((code >> bit) & 1) << (position % 8);
                    }
                }
                let mut actual = [0.0_f32; super::DIMENSION];
                // SAFETY: NEON was detected above and packed has 128 initialized bytes.
                unsafe { super::decode_row::<4>(&packed, &centroids, &mut actual) };
                let expected: [u32; super::DIMENSION] =
                    std::array::from_fn(|index| centroids[usize::from(codes[index])].to_bits());
                assert_eq!(actual.map(f32::to_bits), expected);
            }
        }
    }
}

#[cfg(test)]
#[expect(
    clippy::suboptimal_flops,
    reason = "Independent arithmetic oracles preserve the specified separate multiplication and addition rounding."
)]
mod tests {
    use super::{DIMENSION, accumulate, dot, hadamard};

    // Keep this oracle independent of the production butterfly and sign helpers.
    // Match binary32 stage boundaries rather than using a reassociated dense dot.
    fn reference_rotation(
        mut values: [f32; DIMENSION],
        negative_signs: &[[bool; DIMENSION]; 3],
        inverse: bool,
    ) -> Option<[f32; DIMENSION]> {
        if values.iter().any(|value| !value.is_finite()) {
            return None;
        }
        for pass in 0..3 {
            let signs = &negative_signs[if inverse { 2 - pass } else { pass }];
            if !inverse {
                for (value, &negative) in values.iter_mut().zip(signs) {
                    *value *= if negative { -1.0 } else { 1.0 };
                }
            }
            for stage in 0..8 {
                let width = 1 << stage;
                for base in (0..DIMENSION).step_by(2 * width) {
                    for lane in 0..width {
                        let left = values[base + lane];
                        let right = values[base + width + lane];
                        values[base + lane] = left + right;
                        values[base + width + lane] = left - right;
                    }
                }
            }
            for value in &mut values {
                *value *= 0.0625;
            }
            if values.iter().any(|value| !value.is_finite()) {
                return None;
            }
            if inverse {
                for (value, &negative) in values.iter_mut().zip(signs) {
                    *value *= if negative { -1.0 } else { 1.0 };
                }
            }
        }
        Some(values)
    }

    #[test]
    fn x1_rotation_preserves_ordered_passes_signed_zero_and_subnormals() {
        let signs = std::array::from_fn(|pass: usize| {
            std::array::from_fn(|index: usize| (index * (2 * pass + 1) + pass) % 11 < 5)
        });
        let mixed = std::array::from_fn(|index: usize| {
            let value = f32::from(index.to_le_bytes()[0]) - 127.5;
            value / 32.0
        });
        let subnormal =
            std::array::from_fn(|index: usize| f32::from_bits(u32::from(index.to_le_bytes()[0])));
        for input in [mixed, subnormal, [0.0; DIMENSION], [-0.0; DIMENSION]] {
            for inverse in [false, true] {
                let mut actual = input;
                assert!(super::rotate(&mut actual, &signs, inverse));
                assert_eq!(
                    Some(actual.map(f32::to_bits)),
                    reference_rotation(input, &signs, inverse)
                        .map(|values| values.map(f32::to_bits)),
                );
            }
        }
    }

    #[test]
    fn x1_hadamard_fused_stages_match_ordered_arithmetic_and_atomic_failure() {
        for case in 0..32_u32 {
            let mut word = 0x9328_17af_u32.wrapping_add(case);
            let input: [f32; DIMENSION] = std::array::from_fn(|index| {
                word = word.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                match case % 4 {
                    // Both signs of zero and subnormal coordinates.
                    0 => f32::from_bits(word & 0x807f_ffff),
                    // Exclude the all-ones exponent while retaining wide
                    // cancellation, intermediate overflow and rounding cases.
                    1 => f32::from_bits(word & 0xfeff_ffff),
                    2 => (f32::from(index.to_le_bytes()[0]) - 127.5) * 0.03125,
                    _ => {
                        if index.is_multiple_of(2) {
                            f32::MAX * 0.25
                        } else {
                            -f32::MAX * 0.25
                        }
                    }
                }
            });
            // Independent scalar indexing retains every original butterfly
            // stage and the later normalization, with no native tile helper.
            let mut expected = input;
            for stage in 0..8 {
                let width = 1 << stage;
                for base in (0..DIMENSION).step_by(2 * width) {
                    for lane in 0..width {
                        let a = expected[base + lane];
                        let b = expected[base + width + lane];
                        expected[base + lane] = a + b;
                        expected[base + width + lane] = a - b;
                    }
                }
            }
            for value in &mut expected {
                *value *= 0.0625;
            }
            let finite = expected.iter().all(|value| value.is_finite());
            let mut actual = input;
            assert_eq!(hadamard(&mut actual), finite, "case {case}");
            assert_eq!(
                actual.map(f32::to_bits),
                if finite { expected } else { input }.map(f32::to_bits),
                "case {case}",
            );
        }
    }

    #[test]
    fn x1_rotation_rejects_late_pass_overflow_without_publication() {
        let signs = [[false; DIMENSION]; 3];
        let mut late_overflow = [0.0; DIMENSION];
        late_overflow[0] = f32::MAX * 0.5;
        // The impulse's first Hadamard is finite; its second pass over the
        // resulting constant row overflows before normalization.
        let mut first = late_overflow;
        assert!(hadamard(&mut first));
        assert!(!hadamard(&mut first));
        for input in [
            late_overflow,
            [f32::MAX; DIMENSION],
            [f32::NAN; DIMENSION],
            [f32::INFINITY; DIMENSION],
            [f32::NEG_INFINITY; DIMENSION],
        ] {
            for inverse in [false, true] {
                let mut actual = input;
                assert!(!super::rotate(&mut actual, &signs, inverse));
                assert_eq!(actual.map(f32::to_bits), input.map(f32::to_bits));
                assert!(reference_rotation(input, &signs, inverse).is_none());
            }
        }
        let mut recovery = [0.25; DIMENSION];
        assert!(super::rotate(&mut recovery, &signs, false));
    }

    // Independent bit-at-a-time encoder: no production decode/packing helpers.
    fn packed(codes: &[u8; DIMENSION], bits: u8) -> Vec<u8> {
        let mut bytes = vec![0_u8; DIMENSION * usize::from(bits) / 8];
        for (index, code) in codes.iter().enumerate() {
            for bit in 0..bits {
                let offset = index * usize::from(bits) + usize::from(bit);
                bytes[offset / 8] |= ((code >> bit) & 1) << (offset % 8);
            }
        }
        bytes
    }

    fn codebook() -> [f32; 16] {
        std::array::from_fn(|index| f32::from(index.to_le_bytes()[0]) - 7.5)
    }

    fn codes(bits: u8, offset: u8) -> [u8; DIMENSION] {
        std::array::from_fn(|index: usize| {
            index.to_le_bytes()[0].wrapping_add(offset) & ((1 << bits) - 1)
        })
    }

    #[test]
    fn x1_value_scan_matches_independent_rounding_and_poisoned_failure() -> Result<(), &'static str>
    {
        let centroids = codebook();
        for bits in [3, 4] {
            let mut sum = super::ValueAccumulator::new(bits, &centroids)
                .ok_or("finite accumulator must be accepted")?;
            let mut expected = [0.0_f32; DIMENSION];
            for (offset, scale) in [0.125, -0.25, -0.0, f32::from_bits(1), 0.375]
                .into_iter()
                .enumerate()
            {
                let codes = codes(bits, offset.to_le_bytes()[0]);
                assert!(sum.accumulate(&packed(&codes, bits), scale));
                for (value, code) in expected.iter_mut().zip(codes) {
                    *value += scale * centroids[usize::from(code)];
                }
            }
            assert_eq!(
                sum.finish().map(|values| values.map(f32::to_bits)),
                Some(expected.map(f32::to_bits))
            );
            let bytes = packed(&codes(bits, 0), bits);
            for invalid_scale in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut sum = super::ValueAccumulator::new(bits, &centroids)
                    .ok_or("finite accumulator must be accepted")?;
                assert!(sum.accumulate(&bytes, 0.25));
                assert!(!sum.accumulate(&bytes, invalid_scale));
                assert!(!sum.accumulate(&bytes, 0.25));
                assert!(sum.finish().is_none());
            }
            let mut sum = super::ValueAccumulator::new(bits, &centroids)
                .ok_or("finite accumulator must be accepted")?;
            assert!(sum.accumulate(&bytes, 0.25));
            assert!(!sum.accumulate(&bytes[..bytes.len() - 1], 0.25));
            assert!(sum.finish().is_none());
            let huge = [f32::MAX; 16];
            let mut sum = super::ValueAccumulator::new(bits, &huge)
                .ok_or("finite extreme codebook must be accepted")?;
            assert!(sum.accumulate(&bytes, 0.25));
            assert!(!sum.accumulate(&bytes, 2.0));
            assert!(!sum.accumulate(&bytes, 0.0));
            assert!(sum.finish().is_none());
            let mut last_code = [0; DIMENSION];
            last_code[DIMENSION - 1] = (1 << bits) - 1;
            let mut last_huge = [1.0; 16];
            last_huge[usize::from((1_u8 << bits) - 1)] = f32::MAX;
            let bytes = packed(&last_code, bits);
            let mut sum = super::ValueAccumulator::new(bits, &last_huge)
                .ok_or("finite late-lane codebook must be accepted")?;
            assert!(sum.accumulate(&bytes, 0.25));
            // All earlier lanes update finitely before the final addition overflows.
            assert!(!sum.accumulate(&bytes, 1.0));
            assert!(sum.finish().is_none());
        }
        assert!(super::ValueAccumulator::new(2, &centroids).is_none());
        let mut invalid = centroids;
        invalid[15] = f32::NAN;
        assert!(super::ValueAccumulator::new(4, &invalid).is_none());
        Ok(())
    }

    #[test]
    fn packed_value_finite_checks_cover_each_vector_lane_and_late_overflow()
    -> Result<(), &'static str> {
        for bits in [3, 4] {
            for lane in [0, 1, 2, 3, 124, 125, 126, 127, 252, 253, 254, 255] {
                let mut codes = [0; DIMENSION];
                codes[lane] = 1;
                let bytes = packed(&codes, bits);
                for extreme in [f32::MAX, -f32::MAX] {
                    let mut centroids = [1.0; 16];
                    centroids[1] = extreme;
                    for scale in [1.0, 2.0] {
                        let mut sum = super::ValueAccumulator::new(bits, &centroids)
                            .ok_or("finite codebook must be accepted")?;
                        assert!(sum.accumulate(&bytes, 0.25));
                        // Scale one overflows on addition; scale two overflows
                        // on multiplication. Both failures poison the scan.
                        assert!(!sum.accumulate(&bytes, scale));
                        assert!(!sum.accumulate(&bytes, 0.0));
                        assert!(sum.finish().is_none());

                        let mut output = [0.25; DIMENSION];
                        output[lane] = extreme * 0.25;
                        let before = output.map(f32::to_bits);
                        assert!(!accumulate(&bytes, bits, &centroids, scale, &mut output));
                        assert_eq!(output.map(f32::to_bits), before);
                        assert!(!super::portable_accumulate(
                            &bytes,
                            bits,
                            &centroids,
                            scale,
                            &mut output,
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn x1_prepared_kv_dot_matches_independent_lanes_and_checks_every_row()
    -> Result<(), &'static str> {
        let centroids = codebook();
        let query = std::array::from_fn(|index: usize| {
            let value = f32::from(index.to_le_bytes()[0]) - 127.5;
            value * 0.03125 + (value * 0.17).sin()
        });
        for bits in [3, 4] {
            let prepared = super::PreparedQuery::new(bits, &centroids, &query)
                .ok_or("finite prepared query must be accepted")?;
            // Reuse exactly the same packed allocation with different bytes.
            let mut bytes = packed(&codes(bits, 0), bits);
            for offset in 0..(1 << bits) {
                let codes = codes(bits, offset);
                bytes.copy_from_slice(&packed(&codes, bits));
                let mut lanes = [0.0_f64; 4];
                for (index, (&code, &query)) in codes.iter().zip(&query).enumerate() {
                    lanes[index % 4] += f64::from(centroids[usize::from(code)]) * f64::from(query);
                }
                let expected =
                    super::finite_binary32(((lanes[0] + lanes[1]) + lanes[2]) + lanes[3]);
                assert_eq!(
                    prepared.dot(&bytes).map(f32::to_bits),
                    expected.map(f32::to_bits)
                );
            }
            assert!(prepared.dot(&bytes[..bytes.len() - 1]).is_none());
            bytes.push(0);
            assert!(prepared.dot(&bytes).is_none());
        }
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut invalid_query = query;
            invalid_query[255] = invalid;
            assert!(super::PreparedQuery::new(4, &centroids, &invalid_query).is_none());
            let mut invalid_book = centroids;
            invalid_book[15] = invalid;
            assert!(super::PreparedQuery::new(4, &invalid_book, &query).is_none());
        }
        assert!(super::PreparedQuery::new(2, &centroids, &query).is_none());
        let huge = [f32::MAX; DIMENSION];
        let book = [f32::MAX; 16];
        assert!(
            super::PreparedQuery::new(4, &book, &huge)
                .and_then(|query| query.dot(&[0; 128]))
                .is_none()
        );
        Ok(())
    }

    #[test]
    fn all_codes_cross_byte_boundaries_and_native_scalar_consumption_agree() {
        let centroids = codebook();
        let query = std::array::from_fn(|index: usize| {
            (f32::from(index.to_le_bytes()[0]) - 128.0) * 0.03125
        });
        for bits in [3, 4] {
            for offset in 0..(1 << bits) {
                let codes = codes(bits, offset);
                let bytes = packed(&codes, bits);
                let expected = codes.iter().zip(query).fold(0.0_f64, |sum, (code, q)| {
                    sum + f64::from(centroids[usize::from(*code)]) * f64::from(q)
                });
                assert_eq!(
                    dot(&bytes, bits, &centroids, &query).map(f32::to_bits),
                    super::finite_binary32(expected).map(f32::to_bits)
                );
                assert_eq!(
                    dot(&bytes, bits, &centroids, &query).map(f32::to_bits),
                    super::finite_binary32(super::portable_dot(&bytes, bits, &centroids, &query))
                        .map(f32::to_bits)
                );
                let initial = [0.25; DIMENSION];
                let mut output = initial;
                assert!(accumulate(&bytes, bits, &centroids, -0.125, &mut output));
                let expected: [f32; DIMENSION] = std::array::from_fn(|index| {
                    initial[index] - 0.125 * centroids[usize::from(codes[index])]
                });
                assert_eq!(output.map(f32::to_bits), expected.map(f32::to_bits));
                let mut scalar = initial;
                assert!(super::portable_accumulate(
                    &bytes,
                    bits,
                    &centroids,
                    -0.125,
                    &mut scalar,
                ));
                assert_eq!(output.map(f32::to_bits), scalar.map(f32::to_bits));
            }
        }
    }

    #[test]
    fn repeated_zero_and_finite_extremes_preserve_failure_atomicity() {
        for bits in [3, 4] {
            let bytes = packed(&[0; DIMENSION], bits);
            assert_eq!(
                dot(&bytes, bits, &[0.0; 16], &[f32::MAX; DIMENSION]).map(f32::to_bits),
                Some(0.0_f32.to_bits())
            );
            let mut output = [1.0; DIMENSION];
            assert!(accumulate(&bytes, bits, &[1.0; 16], 0.25, &mut output));
            assert!(accumulate(&bytes, bits, &[1.0; 16], 0.25, &mut output));
            assert_eq!(output.map(f32::to_bits), [1.5_f32.to_bits(); DIMENSION]);
            let before = output.map(f32::to_bits);
            assert!(!accumulate(&bytes, bits, &[f32::MAX; 16], 2.0, &mut output));
            assert_eq!(output.map(f32::to_bits), before);
            assert_eq!(
                dot(&bytes, bits, &[f32::MAX; 16], &[f32::MAX; DIMENSION]),
                None
            );
        }
    }

    #[test]
    fn invalid_width_lengths_and_nonfinite_inputs_never_mutate() {
        let centroids = codebook();
        for (bits, length) in [
            (0, 0),
            (2, 64),
            (3, 95),
            (3, 97),
            (4, 127),
            (4, 129),
            (8, 256),
        ] {
            let bytes = vec![0; length];
            let mut output = [1.0; DIMENSION];
            assert!(dot(&bytes, bits, &centroids, &output).is_none());
            assert!(!accumulate(&bytes, bits, &centroids, 1.0, &mut output));
            assert_eq!(output.map(f32::to_bits), [1.0_f32.to_bits(); DIMENSION]);
        }
        let bytes = packed(&codes(3, 0), 3);
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut book = centroids;
            book[15] = invalid;
            let mut output = [1.0; DIMENSION];
            assert!(dot(&bytes, 3, &book, &output).is_none());
            assert!(!accumulate(&bytes, 3, &book, 1.0, &mut output));
            assert!(!accumulate(&bytes, 3, &centroids, invalid, &mut output));
            assert_eq!(output.map(f32::to_bits), [1.0_f32.to_bits(); DIMENSION]);
            output[255] = invalid;
            let before = output.map(f32::to_bits);
            assert!(dot(&bytes, 3, &centroids, &output).is_none());
            assert!(!accumulate(&bytes, 3, &centroids, 1.0, &mut output));
            assert!(!hadamard(&mut output));
            assert_eq!(output.map(f32::to_bits), before);
        }
    }

    #[test]
    fn hadamard_matches_independent_matrix_and_native_scalar_rounding() {
        let original =
            std::array::from_fn(|index: usize| (f32::from(index.to_le_bytes()[0]) - 127.0) * 0.125);
        let expected: [f32; DIMENSION] = std::array::from_fn(|row: usize| {
            original
                .iter()
                .enumerate()
                .fold(0.0, |sum, (column, value)| {
                    if (row & column).count_ones().is_multiple_of(2) {
                        sum + *value
                    } else {
                        sum - *value
                    }
                })
                * 0.0625
        });
        let mut transformed = original;
        assert!(hadamard(&mut transformed));
        assert_eq!(transformed.map(f32::to_bits), expected.map(f32::to_bits));
        let mut scalar = original;
        super::portable_hadamard(&mut scalar);
        assert_eq!(transformed.map(f32::to_bits), scalar.map(f32::to_bits));
        assert!(hadamard(&mut transformed));
        assert_eq!(transformed.map(f32::to_bits), original.map(f32::to_bits));
        let mut overflow = [f32::MAX; DIMENSION];
        assert!(!hadamard(&mut overflow));
        assert_eq!(overflow.map(f32::to_bits), [f32::MAX.to_bits(); DIMENSION]);
    }

    fn verify_group_bit_exact<const G: usize>(bits: u8) -> Result<(), &'static str> {
        use super::{PreparedQuery, PreparedQueryGroup, ValueAccumulator, ValueGroupAccumulator};

        let centroids = codebook();
        let queries: [[f32; DIMENSION]; G] = std::array::from_fn(|head| {
            std::array::from_fn(|idx| {
                let centered = f32::from(idx.wrapping_mul(head + 3).to_le_bytes()[0]) - 127.5;
                if idx % 7 == 0 {
                    -0.0
                } else {
                    centered * 0.03125
                }
            })
        });
        let query_refs: [&[f32; DIMENSION]; G] = std::array::from_fn(|head| &queries[head]);
        let group_query =
            PreparedQueryGroup::new(bits, &centroids, query_refs).ok_or("finite group query")?;
        let mut per_head_queries = Vec::with_capacity(G);
        for &query in &query_refs {
            per_head_queries
                .push(PreparedQuery::new(bits, &centroids, query).ok_or("finite query")?);
        }

        let mut group_values =
            ValueGroupAccumulator::<G>::new(bits, &centroids).ok_or("finite group acc")?;
        let mut per_head_values = Vec::with_capacity(G);
        for _ in 0..G {
            per_head_values.push(ValueAccumulator::new(bits, &centroids).ok_or("finite acc")?);
        }

        for step in 0..8_u8 {
            let bytes = packed(&codes(bits, step), bits);
            let actual_dots = group_query.dot(&bytes);
            for (head, single) in per_head_queries.iter().enumerate() {
                let expected = single.dot(&bytes);
                assert_eq!(
                    actual_dots[head].map(f32::to_bits),
                    expected.map(f32::to_bits)
                );
            }
            let scales: [f32; G] = std::array::from_fn(|head| {
                if (usize::from(step) + head).is_multiple_of(4) {
                    -0.0
                } else {
                    (f32::from(u8::try_from(head + 1).unwrap_or(1)) - 2.5) * 0.125
                }
            });
            let ok_group = group_values.accumulate(&bytes, scales);
            for (head, single) in per_head_values.iter_mut().enumerate() {
                assert_eq!(ok_group[head], single.accumulate(&bytes, scales[head]));
            }
        }
        let actual_out = group_values.finish();
        for (head, single) in per_head_values.into_iter().enumerate() {
            assert_eq!(
                actual_out[head].map(|row| row.map(f32::to_bits)),
                single.finish().map(|row| row.map(f32::to_bits)),
            );
        }

        // Verify per-head isolation when one head in the group receives a nonfinite scale.
        let bytes = packed(&codes(bits, 1), bits);
        let mut isolated =
            ValueGroupAccumulator::<G>::new(bits, &centroids).ok_or("finite iso acc")?;
        let mut ref_single = Vec::with_capacity(G);
        for _ in 0..G {
            ref_single.push(ValueAccumulator::new(bits, &centroids).ok_or("finite single acc")?);
        }
        let mut scales = [0.25_f32; G];
        scales[G - 1] = f32::NAN;
        let ok = isolated.accumulate(&bytes, scales);
        for (head, single) in ref_single.iter_mut().enumerate() {
            assert_eq!(ok[head], single.accumulate(&bytes, scales[head]));
        }
        let actual_isolated = isolated.finish();
        for (head, single) in ref_single.into_iter().enumerate() {
            assert_eq!(
                actual_isolated[head].map(|row| row.map(f32::to_bits)),
                single.finish().map(|row| row.map(f32::to_bits)),
            );
        }
        Ok(())
    }

    #[test]
    fn grouped_query_and_value_match_single_head_bits_exactly() -> Result<(), &'static str> {
        for bits in [3, 4] {
            verify_group_bit_exact::<2>(bits)?;
            verify_group_bit_exact::<3>(bits)?;
            verify_group_bit_exact::<4>(bits)?;
            verify_group_bit_exact::<6>(bits)?;
        }
        Ok(())
    }

    fn make_pair_queries<const G: usize>(
        mul_add: usize,
        zero_mod: usize,
        scale: f32,
    ) -> [[f32; DIMENSION]; G] {
        std::array::from_fn(|head| {
            std::array::from_fn(|idx| {
                let centered = f32::from(idx.wrapping_mul(head + mul_add).to_le_bytes()[0]) - 127.5;
                if idx % zero_mod == 0 {
                    -0.0
                } else {
                    centered * scale
                }
            })
        })
    }

    fn verify_pair_bit_exact<const G: usize>(bits: u8) -> Result<(), &'static str> {
        use super::{
            PreparedQuery, PreparedQueryPair, ValueAccumulator, ValueGroupPairAccumulator,
        };

        let centroids = codebook();
        let queries0: [[f32; DIMENSION]; G] = make_pair_queries(3, 7, 0.03125);
        let queries1: [[f32; DIMENSION]; G] = make_pair_queries(9, 5, 0.0625);
        let refs0: [&[f32; DIMENSION]; G] = std::array::from_fn(|head| &queries0[head]);
        let refs1: [&[f32; DIMENSION]; G] = std::array::from_fn(|head| &queries1[head]);
        let pair_query =
            PreparedQueryPair::new(bits, &centroids, refs0, refs1).ok_or("finite pair query")?;
        let mut single_queries0 = Vec::with_capacity(G);
        let mut single_queries1 = Vec::with_capacity(G);
        for head in 0..G {
            single_queries0
                .push(PreparedQuery::new(bits, &centroids, refs0[head]).ok_or("finite q0")?);
            single_queries1
                .push(PreparedQuery::new(bits, &centroids, refs1[head]).ok_or("finite q1")?);
        }

        let mut pair_acc =
            ValueGroupPairAccumulator::<G>::new(bits, &centroids).ok_or("finite pair acc")?;
        let mut single_acc0 = Vec::with_capacity(G);
        let mut single_acc1 = Vec::with_capacity(G);
        for _ in 0..G {
            single_acc0.push(ValueAccumulator::new(bits, &centroids).ok_or("finite a0")?);
            single_acc1.push(ValueAccumulator::new(bits, &centroids).ok_or("finite a1")?);
        }

        for step in 0..8_u8 {
            let bytes = packed(&codes(bits, step), bits);
            let (dots0, dots1) = pair_query.dot_both(&bytes);
            let dots_sec = pair_query.dot_second(&bytes);
            for head in 0..G {
                assert_eq!(
                    dots0[head].map(f32::to_bits),
                    single_queries0[head].dot(&bytes).map(f32::to_bits)
                );
                assert_eq!(
                    dots1[head].map(f32::to_bits),
                    single_queries1[head].dot(&bytes).map(f32::to_bits)
                );
                assert_eq!(
                    dots_sec[head].map(f32::to_bits),
                    dots1[head].map(f32::to_bits)
                );
            }
            let scales0: [f32; G] = std::array::from_fn(|head| {
                if (usize::from(step) + head).is_multiple_of(4) {
                    -0.0
                } else {
                    (f32::from(u8::try_from(head + 1).unwrap_or(1)) - 2.0) * 0.125
                }
            });
            let scales1: [f32; G] = std::array::from_fn(|head| {
                if (usize::from(step) + head + 1).is_multiple_of(3) {
                    -0.0
                } else {
                    (f32::from(u8::try_from(head + 2).unwrap_or(2)) - 3.0) * 0.0625
                }
            });
            let (ok0, ok1) = pair_acc.accumulate_both(&bytes, scales0, scales1);
            for head in 0..G {
                assert_eq!(
                    ok0[head],
                    single_acc0[head].accumulate(&bytes, scales0[head])
                );
                assert_eq!(
                    ok1[head],
                    single_acc1[head].accumulate(&bytes, scales1[head])
                );
            }
        }
        let extra = packed(&codes(bits, 3), bits);
        let extra_scales1 = [0.25_f32; G];
        let ok_extra = pair_acc.accumulate_second(&extra, extra_scales1);
        for (head, single) in single_acc1.iter_mut().enumerate() {
            assert_eq!(
                ok_extra[head],
                single.accumulate(&extra, extra_scales1[head])
            );
        }
        let [out0, out1] = pair_acc.finish();
        for (head, single) in single_acc0.into_iter().enumerate() {
            assert_eq!(
                out0[head].map(|row| row.map(f32::to_bits)),
                single.finish().map(|row| row.map(f32::to_bits)),
            );
        }
        for (head, single) in single_acc1.into_iter().enumerate() {
            assert_eq!(
                out1[head].map(|row| row.map(f32::to_bits)),
                single.finish().map(|row| row.map(f32::to_bits)),
            );
        }
        Ok(())
    }

    #[test]
    fn pair_group_query_and_value_match_single_head_bits_exactly() -> Result<(), &'static str> {
        for bits in [3, 4] {
            verify_pair_bit_exact::<1>(bits)?;
            verify_pair_bit_exact::<2>(bits)?;
            verify_pair_bit_exact::<3>(bits)?;
            verify_pair_bit_exact::<4>(bits)?;
            verify_pair_bit_exact::<6>(bits)?;
        }
        Ok(())
    }

    #[test]
    fn normalize_and_quantize_pack_preserve_ties_and_atomicity() {
        let mut values: [f32; DIMENSION] = std::array::from_fn(|index| {
            let signed = f32::from(index.to_le_bytes()[0]) - 127.5;
            if index == 0 {
                -0.0
            } else if index == 1 {
                f32::from_bits(1)
            } else {
                signed * 0.25
            }
        });
        let mut portable_norm = values;
        assert!(super::normalize(&mut values, 1.375));
        assert!(super::portable_normalize(&mut portable_norm, 1.375));
        assert_eq!(values.map(f32::to_bits), portable_norm.map(f32::to_bits));
        let before = values.map(f32::to_bits);
        assert!(!super::normalize(&mut values, 0.0));
        assert!(!super::normalize(&mut values, -0.0));
        assert!(!super::normalize(&mut values, -1.0));
        assert!(!super::normalize(&mut values, f32::NAN));
        assert_eq!(values.map(f32::to_bits), before);
        let mut huge = [f32::MAX; DIMENSION];
        let huge_before = huge.map(f32::to_bits);
        assert!(!super::normalize(&mut huge, 0.5));
        assert_eq!(huge.map(f32::to_bits), huge_before);

        let centroids = codebook();
        let mut edge_cases = Vec::new();
        for pair in centroids.windows(2) {
            let mid = f32::midpoint(pair[0], pair[1]);
            for delta in 0..=4 {
                for candidate in [
                    f32::from_bits(mid.to_bits().wrapping_add(delta)),
                    f32::from_bits(mid.to_bits().wrapping_sub(delta)),
                ] {
                    if candidate.is_finite() {
                        edge_cases.push(candidate);
                    }
                }
            }
        }
        edge_cases.extend([
            0.0,
            -0.0,
            f32::from_bits(1),
            -f32::from_bits(1),
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            1.0e10,
            -1.0e10,
            f32::MAX,
            -f32::MAX,
        ]);
        for bits in [3_u8, 4_u8] {
            let byte_len = DIMENSION * usize::from(bits) / 8;
            for chunk in edge_cases.chunks(DIMENSION) {
                let mut row = [0.0_f32; DIMENSION];
                for (dst, &val) in row.iter_mut().zip(chunk) {
                    *dst = val;
                }
                let mut actual = vec![0xaa_u8; byte_len];
                let mut expected = vec![0x55_u8; byte_len];
                assert!(super::quantize_pack(&row, bits, &centroids, &mut actual));
                assert!(super::portable_quantize_pack(
                    &row,
                    bits,
                    &centroids,
                    &mut expected,
                ));
                assert_eq!(actual, expected);
            }
            let mut bad_row = [0.0_f32; DIMENSION];
            bad_row[DIMENSION - 1] = f32::NAN;
            let mut out = vec![0x5a_u8; byte_len];
            assert!(!super::quantize_pack(&bad_row, bits, &centroids, &mut out));
            assert!(out.iter().all(|&byte| byte == 0x5a));
            assert!(!super::quantize_pack(
                &[0.0; DIMENSION],
                bits,
                &centroids,
                &mut out[..byte_len - 1],
            ));
        }
    }

    fn verify_batched_group_and_pair<const G: usize>(bits: u8) -> Result<(), &'static str> {
        use super::{
            PreparedQueryGroup, PreparedQueryPair, ValueGroupAccumulator,
            ValueGroupPairAccumulator,
        };

        let centroids = codebook();
        let row_bytes = DIMENSION * usize::from(bits) / 8;
        let queries0: [[f32; DIMENSION]; G] = make_pair_queries(3, 7, 0.03125);
        let queries1: [[f32; DIMENSION]; G] = make_pair_queries(9, 5, 0.0625);
        let refs0: [&[f32; DIMENSION]; G] = std::array::from_fn(|head| &queries0[head]);
        let refs1: [&[f32; DIMENSION]; G] = std::array::from_fn(|head| &queries1[head]);
        let group_query =
            PreparedQueryGroup::new(bits, &centroids, refs0).ok_or("finite group query")?;
        let pair_query =
            PreparedQueryPair::new(bits, &centroids, refs0, refs1).ok_or("finite pair query")?;

        for len in [0_usize, 1, 2, 3, 7, 16, 63, 64, 65] {
            let mut packed_rows = Vec::with_capacity(len * row_bytes);
            let mut scales0 = Vec::with_capacity(len);
            let mut scales1 = Vec::with_capacity(len);
            for row_idx in 0..len {
                let offset = u8::try_from((row_idx * 5 + 1) & 0xff).unwrap_or(0);
                packed_rows.extend_from_slice(&packed(&codes(bits, offset), bits));
                scales0.push(std::array::from_fn(|head| {
                    if (row_idx + head).is_multiple_of(11) {
                        -0.0
                    } else {
                        let signed = i16::try_from((row_idx * 3 + head * 7) % 19).unwrap_or(0) - 9;
                        f32::from(signed) * 0.0625
                    }
                }));
                scales1.push(std::array::from_fn(|head| {
                    if (row_idx + head + 2).is_multiple_of(13) {
                        0.0
                    } else {
                        let signed = i16::try_from((row_idx * 5 + head * 3) % 17).unwrap_or(0) - 8;
                        f32::from(signed) * 0.03125
                    }
                }));
            }

            let mut actual_scores = vec![[0.0_f32; G]; len];
            assert!(group_query.score_rows(&packed_rows, &mut actual_scores));
            let mut oracle_scores = vec![[0.0_f32; G]; len];
            assert!(super::portable_score_rows_group(
                &packed_rows,
                bits,
                &centroids,
                &refs0,
                &mut oracle_scores,
            ));
            for (actual, expected) in actual_scores.iter().zip(&oracle_scores) {
                assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
            }

            let mut actual_p0 = vec![[0.0_f32; G]; len];
            let mut actual_p1 = vec![[0.0_f32; G]; len];
            assert!(pair_query.score_rows(&packed_rows, &mut actual_p0, &mut actual_p1));
            let mut oracle_p0 = vec![[0.0_f32; G]; len];
            let mut oracle_p1 = vec![[0.0_f32; G]; len];
            assert!(super::portable_score_rows_pair(
                &packed_rows,
                bits,
                &centroids,
                &refs0,
                &refs1,
                &mut oracle_p0,
                &mut oracle_p1,
            ));
            for ((a0, e0), (a1, e1)) in actual_p0
                .iter()
                .zip(&oracle_p0)
                .zip(actual_p1.iter().zip(&oracle_p1))
            {
                assert_eq!(a0.map(f32::to_bits), e0.map(f32::to_bits));
                assert_eq!(a1.map(f32::to_bits), e1.map(f32::to_bits));
            }

            let mut group_acc =
                ValueGroupAccumulator::<G>::new(bits, &centroids).ok_or("finite group acc")?;
            assert!(group_acc.accumulate_rows(&packed_rows, &scales0));
            let mut oracle_group_out = [[0.0_f32; DIMENSION]; G];
            assert!(super::portable_accumulate_rows_group(
                &packed_rows,
                bits,
                &centroids,
                &scales0,
                &mut oracle_group_out,
            ));
            let actual_group_out = group_acc.finish();
            for head in 0..G {
                assert_eq!(
                    actual_group_out[head].map(|row| row.map(f32::to_bits)),
                    Some(oracle_group_out[head].map(f32::to_bits)),
                );
            }

            let mut pair_acc =
                ValueGroupPairAccumulator::<G>::new(bits, &centroids).ok_or("finite pair acc")?;
            assert!(pair_acc.accumulate_rows(&packed_rows, &scales0, &scales1));
            let mut oracle_pair_out = [[[0.0_f32; DIMENSION]; G]; 2];
            assert!(super::portable_accumulate_rows_group_pair(
                &packed_rows,
                bits,
                &centroids,
                &scales0,
                &scales1,
                &mut oracle_pair_out,
            ));
            let [actual_out0, actual_out1] = pair_acc.finish();
            for head in 0..G {
                assert_eq!(
                    actual_out0[head].map(|row| row.map(f32::to_bits)),
                    Some(oracle_pair_out[0][head].map(f32::to_bits)),
                );
                assert_eq!(
                    actual_out1[head].map(|row| row.map(f32::to_bits)),
                    Some(oracle_pair_out[1][head].map(f32::to_bits)),
                );
            }
        }

        // Mismatching slice lengths.
        let one_row = packed(&codes(bits, 0), bits);
        let mut two_out = vec![[0.0_f32; G]; 2];
        assert!(!group_query.score_rows(&one_row, &mut two_out));
        assert!(!group_query.score_rows(&one_row[..row_bytes - 1], &mut two_out[..1]));
        let mut one_out0 = vec![[0.0_f32; G]; 1];
        assert!(!pair_query.score_rows(&one_row, &mut one_out0, &mut two_out));
        assert!(!pair_query.score_rows(&one_row[..row_bytes - 1], &mut one_out0, &mut two_out[..1]));

        let mut acc_bad =
            ValueGroupAccumulator::<G>::new(bits, &centroids).ok_or("finite group acc")?;
        assert!(!acc_bad.accumulate_rows(&one_row, &[[1.0; G]; 2]));
        assert!(acc_bad.finish().iter().all(Option::is_none));

        let mut pair_bad =
            ValueGroupPairAccumulator::<G>::new(bits, &centroids).ok_or("finite pair acc")?;
        assert!(!pair_bad.accumulate_rows(&one_row, &[[1.0; G]; 1], &[[1.0; G]; 2]));
        assert!(
            pair_bad
                .finish()
                .iter()
                .flatten()
                .all(Option::is_none)
        );

        // Non-finite scales rejected upfront and poison accumulator.
        for bad_scale in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut acc =
                ValueGroupAccumulator::<G>::new(bits, &centroids).ok_or("finite group acc")?;
            assert!(acc.accumulate_rows(&one_row, &[[0.25; G]]));
            let mut bad_scales = [0.25; G];
            bad_scales[G - 1] = bad_scale;
            assert!(!acc.accumulate_rows(&one_row, &[bad_scales]));
            assert!(!acc.accumulate_rows(&one_row, &[[0.25; G]]));
            assert!(acc.finish().iter().all(Option::is_none));

            let mut pacc =
                ValueGroupPairAccumulator::<G>::new(bits, &centroids).ok_or("finite pair acc")?;
            assert!(pacc.accumulate_rows(&one_row, &[[0.25; G]], &[[0.25; G]]));
            assert!(!pacc.accumulate_rows(&one_row, &[[0.25; G]], &[bad_scales]));
            assert!(!pacc.accumulate_rows(&one_row, &[[0.25; G]], &[[0.25; G]]));
            assert!(pacc.finish().iter().flatten().all(Option::is_none));
        }

        // Overflow to +/-Inf in scoring and accumulation.
        let huge_q = [f32::MAX; DIMENSION];
        let huge_book = [f32::MAX; 16];
        let huge_gq =
            PreparedQueryGroup::new(bits, &huge_book, [(&huge_q); G]).ok_or("huge group query")?;
        let huge_pq = PreparedQueryPair::new(bits, &huge_book, [(&huge_q); G], [(&huge_q); G])
            .ok_or("huge pair query")?;
        let mut score_overflow = [[0.0_f32; G]; 1];
        assert!(!huge_gq.score_rows(&one_row, &mut score_overflow));
        let mut p0_overflow = [[0.0_f32; G]; 1];
        let mut p1_overflow = [[0.0_f32; G]; 1];
        assert!(!huge_pq.score_rows(&one_row, &mut p0_overflow, &mut p1_overflow));

        for extreme in [f32::MAX, -f32::MAX] {
            let mut extreme_book = [1.0_f32; 16];
            extreme_book[0] = extreme;
            let mut zero_codes = [0_u8; DIMENSION];
            zero_codes[DIMENSION - 1] = 0;
            let zero_row = packed(&zero_codes, bits);
            let mut two_rows = Vec::with_capacity(2 * row_bytes);
            two_rows.extend_from_slice(&zero_row);
            two_rows.extend_from_slice(&zero_row);

            let mut acc =
                ValueGroupAccumulator::<G>::new(bits, &extreme_book).ok_or("extreme acc")?;
            assert!(!acc.accumulate_rows(&two_rows, &[[1.0; G], [1.0; G]]));
            assert!(acc.finish().iter().all(Option::is_none));

            let mut pacc =
                ValueGroupPairAccumulator::<G>::new(bits, &extreme_book).ok_or("extreme pacc")?;
            assert!(!pacc.accumulate_rows(&two_rows, &[[0.5; G], [0.5; G]], &[[1.0; G], [1.0; G]]));
            assert!(pacc.finish().iter().flatten().all(Option::is_none));
        }

        Ok(())
    }

    #[test]
    fn batched_group_and_pair_match_scalar_oracle_bit_for_bit() -> Result<(), &'static str> {
        for bits in [3, 4] {
            verify_batched_group_and_pair::<1>(bits)?;
            verify_batched_group_and_pair::<2>(bits)?;
            verify_batched_group_and_pair::<3>(bits)?;
            verify_batched_group_and_pair::<6>(bits)?;
        }
        Ok(())
    }
}
