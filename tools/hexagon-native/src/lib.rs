//! Minimal native Rust `FastRPC` capability boundary, without an inference library.
//!
//! The ordinary API performs fixed read-only CDSP capability queries. A separate
//! consuming terminal probe can allocate/register/unregister exactly 4KiB; that
//! registration may implicitly open a domain session. Neither path loads custom
//! DSP kernels, changes power/trust or submits inference. Native calls need an
//! external process deadline; synchronous driver calls have no in-process timeout.
//! The trusted caller also owns the child environment and must exclude vendor
//! search-path, unsigned-process, power and runtime overrides before loading.
#![deny(unsafe_code)]
#![deny(warnings)]

use std::fmt;
use std::path::{Path, PathBuf};

#[cfg(all(windows, target_arch = "aarch64"))]
#[allow(
    unsafe_code,
    reason = "Small audited Windows SCM, loader, trust and vendor capability boundary"
)]
mod windows;

/// Exact ggml host-driver source used to verify the Windows loading convention.
pub const GGML_SOURCE: &str = "94256114c229674ef96e76eb2dea596e65b43818";
/// Exact public Qualcomm header used to pin capability fields and constants.
pub const FASTRPC_SOURCE: &str = "a56e9d4de3614a8d8f4b0e15ff561350c4b686e3";
/// Compute-DSP domain from the public Qualcomm remote header.
pub const CDSP_DOMAIN: u32 = 3;
/// `DSPRPC_GET_DSP_INFO`, used only with the handle-free control function.
pub const GET_DSP_INFO: u32 = 2;

/// Explicit rejection; no failed operation becomes a zero-valued capability.
#[derive(Debug)]
pub enum Error {
    /// The candidate supports only native Windows ARM64.
    UnsupportedPlatform,
    /// The SCM path or a filesystem ancestor failed validation.
    InvalidDriverPath,
    /// A native operation failed; stage is fixed and code is unmodified.
    Native(&'static str, i64),
    /// Ordinary file access failed before driver loading.
    File(std::io::Error),
    /// Model-free ABI or decoding checks failed.
    Contract,
    /// The diagnostic child inherited an unsupported vendor/runtime override.
    InheritedOverride,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedPlatform => f.write_str("native-windows-arm64-required"),
            Self::InvalidDriverPath => f.write_str("invalid-system-driver-path"),
            Self::Native(stage, code) => write!(f, "{stage}:{code}"),
            Self::File(_) => f.write_str("system-driver-file-access-failed"),
            Self::Contract => f.write_str("native-rpc-contract-failed"),
            Self::InheritedOverride => f.write_str("inherited-native-runtime-override"),
        }
    }
}

impl std::error::Error for Error {}

/// Fixed query attributes. No arbitrary driver control value is exposed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Capability {
    /// Whether the domain is supported.
    Domain = 0,
    /// Reported number of 64-byte HVX contexts.
    Hvx64 = 2,
    /// Reported number of 128-byte HVX contexts.
    Hvx128 = 3,
    /// Driver-reported VTCM page size; not an allocation guarantee.
    VtcmPage = 4,
    /// Driver-reported VTCM page count; not concurrent free capacity.
    VtcmCount = 5,
    /// Architecture encoding, retained raw and decoded separately.
    Architecture = 6,
    /// HMX depth capability; not measured matrix throughput.
    HmxDepth = 7,
    /// HMX spatial capability; not measured matrix throughput.
    HmxSpatial = 8,
}

impl Capability {
    /// Fixed vendor attribute ID; there is no caller-supplied control value.
    #[must_use]
    pub const fn id(self) -> u32 {
        match self {
            Self::Domain => 0,
            Self::Hvx64 => 2,
            Self::Hvx128 => 3,
            Self::VtcmPage => 4,
            Self::VtcmCount => 5,
            Self::Architecture => 6,
            Self::HmxDepth => 7,
            Self::HmxSpatial => 8,
        }
    }

    /// Stable receipt label.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Domain => "domain-support",
            Self::Hvx64 => "hvx-64b",
            Self::Hvx128 => "hvx-128b",
            Self::VtcmPage => "vtcm-page",
            Self::VtcmCount => "vtcm-count",
            Self::Architecture => "architecture",
            Self::HmxDepth => "hmx-depth",
            Self::HmxSpatial => "hmx-spatial",
        }
    }
}

/// Complete bounded attribute inventory, excluding unsigned-process controls.
pub const CAPABILITIES: [Capability; 8] = [
    Capability::Domain,
    Capability::Hvx64,
    Capability::Hvx128,
    Capability::VtcmPage,
    Capability::VtcmCount,
    Capability::Architecture,
    Capability::HmxDepth,
    Capability::HmxSpatial,
];

/// Exact C `remote_dsp_capability` layout from the pinned Qualcomm header.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CapabilityRequest {
    /// CDSP domain ID.
    pub domain: u32,
    /// One fixed attribute selector.
    pub attribute_id: u32,
    /// Driver output, meaningful only when the native result is zero.
    pub capability: u32,
}

/// Results of exactly one terminal 4KiB registration probe.
///
/// This observes allocation/host access/mapping APIs only, never DSP compute or
/// DMA. A poisoned result pins uncertain resources until supervised process death.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools, reason = "The diagnostic receipt must retain each independent completed or failed native step")]
pub struct MappingProbeReport {
    /// Fixed terminal stage; no provider text or memory address is included.
    pub stage: &'static str,
    /// Requested host scratch extent, always 4096 bytes.
    pub bytes: u32,
    /// The native allocator returned a nonnull pointer.
    pub allocation_succeeded: bool,
    /// A nonnegative descriptor was obtained from that allocation.
    pub descriptor_observed: bool,
    /// Exact negative descriptor result when descriptor lookup failed.
    pub descriptor_error: Option<i32>,
    /// CPU bytes were written/read before any registration attempt.
    pub cpu_roundtrip_verified: bool,
    /// Whether the explicit registration API was invoked.
    pub mapping_attempted: bool,
    /// Exact mapping return code; absent means the call was not attempted.
    pub mapping_result: Option<i32>,
    /// Exact unmapping return code; absent means the call was not attempted.
    pub unmapping_result: Option<i32>,
    /// The normal native free call returned after no mapping or a successful unmap.
    pub free_completed: bool,
    /// Uncertain native buffer, library and file lifetimes remain pinned.
    pub poisoned: bool,
    /// The driver DLL and file guard remain pinned after any mapping attempt.
    pub owner_pinned_until_exit: bool,
}

impl MappingProbeReport {
    const fn new() -> Self {
        Self { stage: "not-started", bytes: 4096, allocation_succeeded: false,
            descriptor_observed: false, descriptor_error: None,
            cpu_roundtrip_verified: false, mapping_attempted: false,
            mapping_result: None, unmapping_result: None,
            free_completed: false, poisoned: false, owner_pinned_until_exit: false }
    }

    /// Whether allocation, host check, mapping, unmapping and normal free completed.
    #[must_use]
    pub const fn completed_mapping_cycle(self) -> bool {
        self.allocation_succeeded && self.descriptor_observed
            && self.cpu_roundtrip_verified && self.mapping_attempted
            && matches!(self.mapping_result, Some(0))
            && matches!(self.unmapping_result, Some(0))
            && self.free_completed && !self.poisoned
    }
}

/// Decode the architecture's packed decimal low byte as in the pinned host.
///
/// Invalid decimal nibbles and zero remain unknown. The raw driver value must
/// also be retained; this supplies no device, kernel or performance eligibility.
#[must_use]
pub const fn decode_architecture(raw: u32) -> Option<u32> {
    let major = (raw & 0xff) >> 4;
    let minor = raw & 0x0f;
    if major > 9 || minor > 9 || major == 0 {
        None
    } else {
        Some(major * 10 + minor)
    }
}

/// Model-free checks that run without opening services, files or DLLs.
///
/// # Errors
/// Returns a fixed contract error if ABI size/alignment/offsets or architecture
/// decoding differ from the pinned C contract.
pub fn self_check() -> Result<(), Error> {
    if size_of::<CapabilityRequest>() != 12
        || align_of::<CapabilityRequest>() != 4
        || std::mem::offset_of!(CapabilityRequest, domain) != 0
        || std::mem::offset_of!(CapabilityRequest, attribute_id) != 4
        || std::mem::offset_of!(CapabilityRequest, capability) != 8
        || decode_architecture(0x81) != Some(81)
        || decode_architecture(0x73) != Some(73)
        || decode_architecture(0).is_some()
        || decode_architecture(0x8a).is_some()
        || MappingProbeReport::new().completed_mapping_cycle()
    {
        return Err(Error::Contract);
    }
    Ok(())
}

/// Validated service-derived path; creating it does not load any driver code.
#[derive(Debug)]
pub struct DriverLocation(PathBuf);

impl DriverLocation {
    /// Read the fixed `qcnspmcdm` service and validate its `DriverStore` location.
    ///
    /// # Errors
    /// Rejects nonnative platforms, missing SCM access, malformed paths and
    /// reparse ancestors. No environment variable or caller path chooses a DLL.
    pub fn inspect_system() -> Result<Self, Error> {
        #[cfg(all(windows, target_arch = "aarch64"))]
        {
            windows::inspect_system().map(Self)
        }
        #[cfg(not(all(windows, target_arch = "aarch64")))]
        {
            Err(Error::UnsupportedPlatform)
        }
    }

    /// Service-derived absolute vendor driver path, suitable for external hashing.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.0
    }
}

/// Single-threaded native capability session with retained file/library lifetime.
pub struct Driver {
    location: DriverLocation,
    #[cfg(all(windows, target_arch = "aarch64"))]
    inner: windows::Driver,
}

impl Driver {
    /// Verify local Authenticode trust and load the fixed service-derived driver.
    ///
    /// The caller must also bind this path, file digest and publisher to its
    /// reviewed driver evidence. Cached Authenticode trust is not a Microsoft
    /// publisher pin, fresh revocation check or inference qualification.
    ///
    /// # Errors
    /// Rejects native/path/file/signature/loader/export failures. Caller-owned
    /// process deadlines must cover signature validation and every native call.
    pub fn open_system() -> Result<Self, Error> {
        #[cfg(all(windows, target_arch = "aarch64"))]
        {
            let location = DriverLocation::inspect_system()?;
            let inner = windows::Driver::open(location.path())?;
            Ok(Self { location, inner })
        }
        #[cfg(not(all(windows, target_arch = "aarch64")))]
        {
            Err(Error::UnsupportedPlatform)
        }
    }

    /// Exact service-derived path whose signed file is retained by this session.
    #[must_use]
    pub fn path(&self) -> &Path {
        self.location.path()
    }

    /// Issue only the fixed handle-free capability control request for CDSP.
    ///
    /// # Errors
    /// Preserves native failure codes; output bytes are not used on failure.
    pub fn capability(&mut self, capability: Capability) -> Result<u32, Error> {
        #[cfg(all(windows, target_arch = "aarch64"))]
        {
            self.inner.capability(capability)
        }
        #[cfg(not(all(windows, target_arch = "aarch64")))]
        {
            let _ = (self, capability);
            Err(Error::UnsupportedPlatform)
        }
    }

    /// Consume this driver in one fixed 4KiB allocation/registration/unmap probe.
    ///
    /// The documented registration call may implicitly open a domain session;
    /// this opens no custom DSP module and invokes no compute. Every subsequent
    /// driver open/query in this process rejects. Any mapping/unmapping error
    /// pins the uncertain allocation until supervised process death. Every mapping
    /// attempt retains the DLL/file owner until process death because implicit
    /// domain-session teardown has not been qualified, even after a successful unmap.
    /// Do not run this terminal diagnostic inside a serving process.
    ///
    /// # Errors
    /// Rejects unsupported platforms, repeated probes and missing native exports
    /// before allocation. Per-step native refusals are retained in the report.
    pub fn scratch_mapping_probe(self) -> Result<MappingProbeReport, Error> {
        #[cfg(all(windows, target_arch = "aarch64"))]
        { windows::mapping_probe(self.inner) }
        #[cfg(not(all(windows, target_arch = "aarch64")))]
        { let _ = self; Err(Error::UnsupportedPlatform) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_capability_layout_and_architecture_contract() -> Result<(), Error> {
        self_check()
    }

    #[test]
    fn raw_architecture_does_not_become_a_family_assumption() {
        assert_eq!(decode_architecture(0xffff_0081), Some(81));
        assert_eq!(decode_architecture(0x8f), None);
        assert_eq!(decode_architecture(0xff), None);
        assert_eq!(decode_architecture(0x02), None);
    }

    #[test]
    fn partial_mapping_receipts_never_claim_a_complete_cycle() {
        let mut report = MappingProbeReport::new();
        report.allocation_succeeded = true;
        report.descriptor_observed = true;
        report.cpu_roundtrip_verified = true;
        report.mapping_attempted = true;
        report.mapping_result = Some(0);
        report.unmapping_result = Some(0);
        report.free_completed = true;
        assert!(report.completed_mapping_cycle());
        report.poisoned = true;
        assert!(!report.completed_mapping_cycle());
        report.poisoned = false;
        report.unmapping_result = Some(-1);
        assert!(!report.completed_mapping_cycle());
        report.unmapping_result = None;
        assert!(!report.completed_mapping_cycle());
    }
}
