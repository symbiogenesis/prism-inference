//! Bounded metadata for immutable, append-only native weight residency.

use std::sync::Arc;

use bonsai_prism_runtime::model::Tensor;
use hexagon_native_rpc::HmxQ2WeightHandle;

use crate::{Result, invalid};

/// Observed packed-weight preparation and reuse, without timing claims.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HmxResidencyStatistics {
    /// Complete immutable tensor projections prepared before kernel submission.
    pub tensors_prepared: u64,
    /// Projection attempts finding a previously prepared immutable tensor.
    pub cache_hits: u64,
    /// Packed bytes copied into the native arena, including incomplete preparation.
    pub weight_payload_bytes_prepared: u64,
    /// Retained host vector capacities charged against the metadata reservation.
    pub metadata_payload_bytes: usize,
}

pub(super) struct Projection {
    source: Arc<[u8]>,
    columns: usize,
    rows: usize,
    pub(super) first_handle: usize,
}

pub(super) struct Cache {
    pub(super) projections: Vec<Projection>,
    pub(super) handles: Vec<HmxQ2WeightHandle>,
    pub(super) reservation: usize,
    pub(super) statistics: HmxResidencyStatistics,
}

impl Cache {
    pub(super) fn new(reservation: usize) -> Result<Self> {
        let slots = reservation / (size_of::<Projection>() + size_of::<HmxQ2WeightHandle>());
        if slots == 0 {
            return Err(invalid("HMX resident metadata reservation is too small"));
        }
        let mut projections = Vec::new();
        let mut handles = Vec::new();
        projections
            .try_reserve_exact(slots)
            .map_err(|_| invalid("HMX resident projection metadata allocation failed"))?;
        handles
            .try_reserve_exact(slots)
            .map_err(|_| invalid("HMX resident handle metadata allocation failed"))?;
        let metadata_payload_bytes = projections
            .capacity()
            .checked_mul(size_of::<Projection>())
            .and_then(|bytes| {
                handles
                    .capacity()
                    .checked_mul(size_of::<HmxQ2WeightHandle>())
                    .and_then(|handle_bytes| bytes.checked_add(handle_bytes))
            })
            .filter(|&bytes| bytes <= reservation)
            .ok_or_else(|| invalid("HMX resident metadata allocation exceeds reservation"))?;
        Ok(Self {
            projections,
            handles,
            reservation,
            statistics: HmxResidencyStatistics {
                metadata_payload_bytes,
                ..HmxResidencyStatistics::default()
            },
        })
    }

    pub(super) fn find(&self, tensor: &Tensor) -> Option<usize> {
        self.projections.iter().position(|projection| {
            // The retained strong Arc prevents source allocation mutation and
            // address reuse. Names are presentation metadata, never identities.
            tensor.kind == 142
                && tensor.dimensions.as_slice() == [projection.columns, projection.rows]
                && Arc::ptr_eq(&projection.source, &tensor.data)
        })
    }

    pub(super) const fn has_metadata_capacity(&self, operations: usize) -> bool {
        self.projections.len() < self.projections.capacity()
            && operations <= self.handles.capacity() - self.handles.len()
    }

    pub(super) fn commit(&mut self, tensor: &Tensor, first_handle: usize) -> usize {
        let index = self.projections.len();
        self.projections.push(Projection {
            source: Arc::clone(&tensor.data),
            columns: tensor.dimensions[0],
            rows: tensor.dimensions[1],
            first_handle,
        });
        self.statistics.tensors_prepared = self.statistics.tensors_prepared.saturating_add(1);
        index
    }
}
