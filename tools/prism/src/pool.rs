//! Safe sharing decisions, independently of the native tensor implementation.

#![forbid(unsafe_code)]

/// Largest eligible B-vector payload, in bytes (64 KiB).
pub const MAX_TENSOR_BYTES: usize = 64 * 1024;
/// Maximum copied payload retained by one pool, in bytes (1 MiB).
pub const MAX_CACHE_BYTES: usize = 1024 * 1024;
/// Maximum number of tensor owners retained by one pool.
pub const MAX_ENTRIES: usize = 64;

/// Tensor metadata used to decide whether immutable storage can be shared.
///
/// Context and buffer-type values are opaque identity tokens. The native
/// adapter obtains all fields from ggml; the safe policy never dereferences
/// identities or tensors. Distinct contexts, buffer types, dimensions, or
/// strides cannot share an owner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Layout {
    /// Identity of the destination allocation context.
    pub context: usize,
    /// Identity of the destination buffer type.
    pub buffer_type: usize,
    /// Whether the buffer type is ggml's ordinary CPU buffer type.
    pub cpu: bool,
    /// Whether the source tensor has native F32 element type.
    pub f32: bool,
    /// Whether ggml reports the source tensor as contiguous.
    pub contiguous: bool,
    /// Native tensor extents in ggml dimension order.
    pub dimensions: [i64; 4],
    /// Native tensor byte strides in ggml dimension order.
    pub strides: [usize; 4],
    /// Native source payload size, as reported by ggml.
    pub byte_len: usize,
}

impl Layout {
    /// Returns whether metadata satisfies Prism's CPU rank-one B policy.
    ///
    /// Like the native policy, this trusts the supplied contiguity and payload
    /// size metadata. Lookup and insertion also require the payload slice to
    /// have exactly `byte_len` bytes.
    #[must_use]
    pub const fn eligible(&self) -> bool {
        self.cpu
            && self.f32
            && self.contiguous
            && self.dimensions[0] == 1
            && self.dimensions[1] > 0
            && self.dimensions[2] == 1
            && self.dimensions[3] == 1
            && self.byte_len <= MAX_TENSOR_BYTES
    }
}

#[derive(Debug)]
struct Entry {
    layout: Layout,
    owner: usize,
    bytes: Vec<u8>,
}

/// Bounded cache of immutable B-vector payloads and caller-owned identities.
///
/// A caller first uses [`Self::find`]. After a miss, it allocates a distinct
/// tensor and may register its nonzero identity using [`Self::remember`]. The
/// caller must keep every registered owner alive and immutable while using
/// identities returned by this pool. Dropping a pool releases only its copied
/// payloads; it neither frees nor modifies a tensor.
///
/// Failed cache reservations return `false` and leave the new tensor uncached.
/// A full cache still finds existing owners. Sharing counters saturate at
/// `usize::MAX` rather than overflowing in extremely long-running callers.
#[derive(Debug, Default)]
pub struct Pool {
    entries: Vec<Entry>,
    cached_bytes: usize,
    shared_tensors: usize,
    shared_bytes: usize,
}

impl Pool {
    /// Creates an empty pool without allocating.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
            cached_bytes: 0,
            shared_tensors: 0,
            shared_bytes: 0,
        }
    }

    /// Returns whether metadata satisfies Prism's CPU rank-one B policy.
    #[must_use]
    pub const fn eligible(layout: &Layout) -> bool {
        layout.eligible()
    }

    /// Finds an owner with exactly the same layout and immutable payload bytes.
    ///
    /// Ineligible metadata and mismatched payload lengths always miss. Only a
    /// hit increments the sharing counters. No allocations occur during lookup.
    #[must_use]
    pub fn find(&mut self, layout: &Layout, bytes: &[u8]) -> Option<usize> {
        if !layout.eligible() || bytes.len() != layout.byte_len {
            return None;
        }
        let owner = self
            .entries
            .iter()
            .find(|entry| entry.layout == *layout && entry.bytes == bytes)?
            .owner;
        self.shared_tensors = self.shared_tensors.saturating_add(1);
        self.shared_bytes = self.shared_bytes.saturating_add(bytes.len());
        Some(owner)
    }

    /// Retains an eligible owner if both cache budgets and allocations permit.
    ///
    /// Supply the newly allocated owner's layout: a native duplicate can
    /// normalize source strides. Future lookups compare against this layout.
    ///
    /// Returns `false` for a zero owner identity, duplicate layout and bytes,
    /// ineligible metadata, an incorrect payload length, a full cache, or an
    /// allocation failure. No sharing counters change. The payload is copied,
    /// so subsequent caller changes to the supplied slice do not affect it.
    #[must_use]
    pub fn remember(&mut self, layout: &Layout, bytes: &[u8], owner: usize) -> bool {
        if owner == 0
            || !layout.eligible()
            || bytes.len() != layout.byte_len
            || self.entries.len() >= MAX_ENTRIES
            || bytes.len() > MAX_CACHE_BYTES - self.cached_bytes
            || self
                .entries
                .iter()
                .any(|entry| entry.layout == *layout && entry.bytes == bytes)
        {
            return false;
        }
        if self.entries.try_reserve(1).is_err() {
            return false;
        }
        let mut copy = Vec::new();
        if copy.try_reserve_exact(bytes.len()).is_err() {
            return false;
        }
        copy.extend_from_slice(bytes);
        self.entries.push(Entry {
            layout: *layout,
            owner,
            bytes: copy,
        });
        self.cached_bytes += bytes.len();
        true
    }

    /// Returns the number of successful sharing lookups, saturating on overflow.
    #[must_use]
    pub const fn shared_tensors(&self) -> usize {
        self.shared_tensors
    }

    /// Returns the bytes saved by sharing lookups, saturating on overflow.
    #[must_use]
    pub const fn shared_bytes(&self) -> usize {
        self.shared_bytes
    }

    /// Returns the copied payload bytes currently retained by the pool.
    #[must_use]
    pub const fn cached_bytes(&self) -> usize {
        self.cached_bytes
    }

    /// Returns the number of cached owners.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns whether the pool contains no owners.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
