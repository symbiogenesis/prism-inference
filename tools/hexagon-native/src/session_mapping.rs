//! Fixed signed-module session and FD mapping receipt; no computation is submitted.

#![allow(
    clippy::redundant_pub_crate,
    reason = "Private receipt construction must not become public native-boundary API"
)]

use crate::{MappingProbeReport, SessionPreparationReport};

/// Terminal native session observation, never an inference/admission permit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "Independent native steps must remain distinct in diagnostic evidence"
)]
pub struct SessionMappingReport {
    /// Fixed completed or entered stage, suitable for bounded progress capture.
    pub stage: &'static str,
    /// The separately named reference-process initializer was explicitly selected.
    pub reference_process_initialization_selected: bool,
    /// Exact fixed process-initialization return; absent means not attempted.
    pub process_initialization_result: Option<i32>,
    /// The per-process unsigned-PD request was actually invoked, not inferred.
    pub unsigned_process_requested: bool,
    /// Exact architecture/domain/URI metadata requests and results.
    pub metadata: SessionPreparationReport,
    /// Native module-open return, absent when not attempted.
    pub module_open_result: Option<i32>,
    /// A successful open returned a handle different from the poisoned sentinel.
    pub handle_observed: bool,
    /// Fixed bundled module was opened through the selected package search route.
    pub remote_module_opened: bool,
    /// Fixed 4096-byte allocation and explicit FD-map cycle observations.
    pub mapping: MappingProbeReport,
    /// Native module-close return, absent when not attempted.
    pub module_close_result: Option<i32>,
    /// The explicit remote handle close completed with success.
    pub module_close_completed: bool,
    /// Driver, module/catalog guards and process lease remain pinned until exit.
    pub owner_pinned_until_exit: bool,
}

impl SessionMappingReport {
    pub(crate) const fn new(metadata: SessionPreparationReport) -> Self {
        Self {
            stage: "session-metadata-incomplete",
            reference_process_initialization_selected: false,
            process_initialization_result: None,
            unsigned_process_requested: false,
            metadata,
            module_open_result: None,
            handle_observed: false,
            remote_module_opened: false,
            mapping: MappingProbeReport::new(),
            module_close_result: None,
            module_close_completed: false,
            owner_pinned_until_exit: false,
        }
    }

    /// Metadata, fixed signed-module open, FD map/unmap/free and close all succeeded.
    ///
    /// This proves neither DSP visibility, DMA/compute, signed deployment identity
    /// nor complete vendor session teardown. The supervisor verifies those separately.
    #[must_use]
    pub const fn completed_cycle(self) -> bool {
        (if self.reference_process_initialization_selected {
            self.unsigned_process_requested && matches!(self.process_initialization_result, Some(0))
        } else {
            !self.unsigned_process_requested && self.process_initialization_result.is_none()
        }) && self.metadata.metadata_prepared()
            && matches!(self.module_open_result, Some(0))
            && self.handle_observed
            && self.remote_module_opened
            && self.mapping.completed_mapping_cycle()
            && matches!(self.module_close_result, Some(0))
            && self.module_close_completed
            && self.owner_pinned_until_exit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_or_uncertain_sessions_never_claim_a_complete_cycle() {
        let mut metadata = SessionPreparationReport::new(0x81);
        metadata.domain_result = Some(0);
        metadata.effective_domain = Some(3);
        metadata.uri_result = Some(0);
        metadata.uri_verified = true;
        let mut report = SessionMappingReport::new(metadata);
        assert!(!report.completed_cycle());
        report.module_open_result = Some(0);
        report.handle_observed = true;
        report.remote_module_opened = true;
        report.mapping.allocation_succeeded = true;
        report.mapping.descriptor_observed = true;
        report.mapping.cpu_roundtrip_verified = true;
        report.mapping.mapping_attempted = true;
        report.mapping.mapping_result = Some(0);
        report.mapping.unmapping_result = Some(0);
        report.mapping.free_completed = true;
        report.module_close_result = Some(0);
        report.module_close_completed = true;
        report.owner_pinned_until_exit = true;
        assert!(report.completed_cycle());
        report.reference_process_initialization_selected = true;
        assert!(!report.completed_cycle());
        report.process_initialization_result = Some(0);
        report.unsigned_process_requested = true;
        assert!(report.completed_cycle());
        report.mapping.poisoned = true;
        assert!(!report.completed_cycle());
        report.mapping.poisoned = false;
        report.module_close_result = Some(512);
        assert!(!report.completed_cycle());
        report.module_close_result = Some(0);
        report.metadata.uri_verified = false;
        assert!(!report.completed_cycle());
    }
}
