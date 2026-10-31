//! Minimal native Rust `FastRPC` capability boundary, without an inference library.
//!
//! This candidate performs only the fixed, read-only CDSP capability control
//! request. It cannot load DSP kernels, create sessions, change power/trust,
//! allocate shared buffers or submit inference. Native calls need an external
//! process deadline; a synchronous driver call has no safe in-process timeout.
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
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedPlatform => f.write_str("native-windows-arm64-required"),
            Self::InvalidDriverPath => f.write_str("invalid-system-driver-path"),
            Self::Native(stage, code) => write!(f, "{stage}:{code}"),
            Self::File(_) => f.write_str("system-driver-file-access-failed"),
            Self::Contract => f.write_str("native-rpc-contract-failed"),
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
            let _ = capability;
            Err(Error::UnsupportedPlatform)
        }
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
}
