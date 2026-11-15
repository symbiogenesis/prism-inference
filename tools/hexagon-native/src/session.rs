//! Fixed default-CDSP session metadata, without opening a remote module.
#![expect(
    clippy::redundant_pub_crate,
    reason = "Private URI and receipt helpers must not become public native-boundary API"
)]

use std::ffi::CStr;

/// Default-session effective-domain query in the pinned public vendor header.
pub const GET_EFFECTIVE_DOMAIN_ID: u32 = 14;
/// Default-session module-URI query in the pinned public vendor header.
pub const GET_URI: u32 = 15;
/// Explicit FD mapping mode in the pinned public vendor header.
pub const MAP_FD: i32 = 2;
/// Delayed FD registration; it does not establish a DSP virtual mapping.
pub const MAP_FD_DELAYED: i32 = 3;

/// Result of bounded default-session metadata preparation.
///
/// This receipt is descriptive, never a session handle or authorization to map.
/// Neither successful metadata queries nor the URI prove installed catalog
/// membership, signed module loading, an established session or DSP execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionPreparationReport {
    /// Fixed stage, including a native refusal or unexpected response.
    pub stage: &'static str,
    /// Unmodified architecture capability value.
    pub architecture_raw: u32,
    /// Supported bundled module family, only when it is exactly v73 or v81.
    pub architecture: Option<u32>,
    /// Exact effective-domain request return; absent means not attempted.
    pub domain_result: Option<i32>,
    /// Effective domain returned for `cdsp`, session zero, after successful checks.
    pub effective_domain: Option<i32>,
    /// Exact URI request return; absent means not attempted.
    pub uri_result: Option<i32>,
    /// Returned URI exactly matches the fixed module, domain and session zero.
    pub uri_verified: bool,
}

/// Return `true` when `architecture` identifies a bundled signed HTP target (`v73` or `v81`).
#[must_use]
pub const fn is_supported_htp_architecture(architecture: u32) -> bool {
    matches!(architecture, 73 | 81)
}

/// Hardware capability profile for an opened Hexagon CDSP HTP session (`v73` or `v81`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HtpArchProfile {
    /// Decoded Hexagon CDSP architecture version (`73` on Snapdragon X1 Elite, `81` on X2 Elite).
    pub architecture: u32,
    /// Bounded HVX-128 worker threads requested at queue startup and in `kparams` solvers.
    pub hvx_workers: u32,
    /// Bounded VTCM scratchpad budget (bytes) requested at queue startup and in `kparams` solvers.
    pub vtcm_bytes: u32,
    /// Whether the DSP hardware supports 64-bit DMA strides beyond `HTP_DMA32_MAX_STRIDE_BYTES`.
    pub dma64_capable: bool,
}

impl HtpArchProfile {
    /// Fixed default profile for Hexagon `v81` (8 HVX-128 threads, 4 MiB VTCM request, DMA64).
    #[must_use]
    pub const fn v81() -> Self {
        Self {
            architecture: 81,
            hvx_workers: crate::HMX_Q2_HVX_WORKERS,
            vtcm_bytes: crate::NATIVE_Q2_VTCM_REQUEST_BYTES,
            dma64_capable: true,
        }
    }

    /// Conservative default profile for Hexagon `v73` when FastRPC HVX/VTCM capability queries return zero.
    #[must_use]
    pub const fn v73_default() -> Self {
        Self {
            architecture: 73,
            hvx_workers: 4,
            vtcm_bytes: crate::HMX_Q2_VTCM_REQUEST_BYTES,
            dma64_capable: false,
        }
    }

    /// Derive a bounded HTP capability profile from the driver's reported architecture,
    /// `Hvx128` context count, and total `VtcmPage * VtcmCount` bytes.
    ///
    /// Returns `None` for any architecture other than `v73` or `v81`.
    #[must_use]
    pub const fn for_architecture(
        architecture: u32,
        hvx128_contexts: u32,
        vtcm_total_bytes: u32,
    ) -> Option<Self> {
        if !is_supported_htp_architecture(architecture) {
            return None;
        }
        let default_workers = match architecture {
            73 => 4,
            81 => crate::HMX_Q2_HVX_WORKERS,
            _ => 1,
        };
        let hvx_workers = if hvx128_contexts == 0 {
            default_workers
        } else if hvx128_contexts > crate::HMX_Q2_HVX_WORKERS {
            crate::HMX_Q2_HVX_WORKERS
        } else {
            hvx128_contexts
        };
        let vtcm_bytes = if vtcm_total_bytes == 0 {
            match architecture {
                73 => crate::HMX_Q2_VTCM_REQUEST_BYTES,
                81 => crate::NATIVE_Q2_VTCM_REQUEST_BYTES,
                _ => 65_536,
            }
        } else if vtcm_total_bytes >= 8 * 1024 * 1024 {
            crate::NATIVE_Q2_VTCM_REQUEST_BYTES
        } else {
            let headroom = vtcm_total_bytes / 8;
            let usable = vtcm_total_bytes.saturating_sub(headroom);
            if usable < 65_536 {
                65_536
            } else if usable > crate::NATIVE_Q2_VTCM_REQUEST_BYTES {
                crate::NATIVE_Q2_VTCM_REQUEST_BYTES
            } else {
                usable
            }
        };
        Some(Self {
            architecture,
            hvx_workers,
            vtcm_bytes,
            dma64_capable: architecture >= 81,
        })
    }
}

impl SessionPreparationReport {
    pub(crate) const fn new(raw: u32) -> Self {
        let architecture = match crate::decode_architecture(raw) {
            Some(arch) if is_supported_htp_architecture(arch) => Some(arch),
            _ => None,
        };
        Self {
            stage: "architecture-outside-selected-targets",
            architecture_raw: raw,
            architecture,
            domain_result: None,
            effective_domain: None,
            uri_result: None,
            uri_verified: false,
        }
    }

    /// Both fixed metadata requests completed and their outputs were validated.
    ///
    /// This does not mean that a session was opened or a buffer was mapped.
    #[must_use]
    pub const fn metadata_prepared(self) -> bool {
        self.architecture.is_some()
            && matches!(self.domain_result, Some(0))
            && self.effective_domain.is_some()
            && matches!(self.uri_result, Some(0))
            && self.uri_verified
    }
}

pub(crate) const fn module_uri(architecture: u32) -> Option<&'static CStr> {
    match architecture {
        73 => Some(c"file:///libggml-htp-v73.so?htp_iface_skel_handle_invoke&_modver=1.0"),
        81 => Some(c"file:///libggml-htp-v81.so?htp_iface_skel_handle_invoke&_modver=1.0"),
        _ => None,
    }
}

pub(crate) fn checked_uri(architecture: u32, bytes: &[u8]) -> bool {
    let Some(module) = module_uri(architecture) else {
        return false;
    };
    let Some(terminator) = bytes.iter().position(|byte| *byte == 0) else {
        return false;
    };
    let mut expected = module.to_bytes().to_vec();
    expected.extend_from_slice(b"&_dom=cdsp&_session=0");
    bytes[..terminator] == expected
}

pub(crate) fn self_check() -> bool {
    let Some(module_81) = module_uri(81) else {
        return false;
    };
    let Some(module_73) = module_uri(73) else {
        return false;
    };
    let mut uri_81 = module_81.to_bytes().to_vec();
    uri_81.extend_from_slice(c"&_dom=cdsp&_session=0".to_bytes_with_nul());
    let mut uri_73 = module_73.to_bytes().to_vec();
    uri_73.extend_from_slice(c"&_dom=cdsp&_session=0".to_bytes_with_nul());
    checked_uri(81, &uri_81)
        && checked_uri(73, &uri_73)
        && !checked_uri(73, &uri_81)
        && !checked_uri(81, &uri_73)
        && !SessionPreparationReport::new(0x81).metadata_prepared()
        && !SessionPreparationReport::new(0x73).metadata_prepared()
        && MAP_FD == 2
        && MAP_FD_DELAYED == 3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uri_is_bound_to_exact_architecture_domain_and_default_session() {
        assert!(self_check());
        for arch in [73_u32, 81_u32] {
            let mut uri = module_uri(arch).map_or_default(CStr::to_bytes).to_vec();
            uri.extend_from_slice(c"&_dom=cdsp&_session=0".to_bytes_with_nul());
            assert!(checked_uri(arch, &uri));
            assert!(!checked_uri(82, &uri));
            assert!(!checked_uri(arch, &uri[..uri.len() - 1]));
        }
        assert_eq!(
            HtpArchProfile::for_architecture(73, 0, 0),
            Some(HtpArchProfile::v73_default())
        );
        assert_eq!(
            HtpArchProfile::for_architecture(73, 6, 4 * 1024 * 1024),
            Some(HtpArchProfile {
                architecture: 73,
                hvx_workers: 6,
                vtcm_bytes: 3_670_016,
                dma64_capable: false,
            })
        );
        assert_eq!(
            HtpArchProfile::for_architecture(81, 8, 8 * 1024 * 1024),
            Some(HtpArchProfile::v81())
        );
        assert_eq!(
            HtpArchProfile::for_architecture(68, 4, 2 * 1024 * 1024),
            None
        );
        let mut tail = module_uri(81).map_or_default(CStr::to_bytes).to_vec();
        tail.extend_from_slice(c"&_dom=cdsp&_session=0".to_bytes_with_nul());
        tail.truncate(tail.len() - 2);
        tail.extend_from_slice(b"1\0");
        assert!(!checked_uri(81, &tail));
        assert!(!checked_uri(81, b"\0"));
    }

    #[test]
    fn partial_metadata_and_new_families_never_become_prepared_sessions() {
        let mut report = SessionPreparationReport::new(0x81);
        report.domain_result = Some(0);
        report.effective_domain = Some(3);
        report.uri_result = Some(0);
        assert!(!report.metadata_prepared());
        report.uri_verified = true;
        assert!(report.metadata_prepared());
        report.domain_result = Some(512);
        assert!(!report.metadata_prepared());
        assert_eq!(SessionPreparationReport::new(0x82).architecture, None);
        assert_eq!(SessionPreparationReport::new(0x8f).architecture, None);
    }
}
