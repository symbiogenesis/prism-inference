//! Fixed default-CDSP session metadata, without opening a remote module.
#![allow(
    clippy::redundant_pub_crate,
    reason = "Private URI and receipt helpers must not become public native-boundary API"
)]

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

impl SessionPreparationReport {
    pub(crate) const fn new(raw: u32) -> Self {
        let architecture = match crate::decode_architecture(raw) {
            Some(73) => Some(73),
            Some(81) => Some(81),
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

pub(crate) const fn module_uri(architecture: u32) -> Option<&'static [u8]> {
    match architecture {
        73 => Some(b"file:///libggml-htp-v73.so?htp_iface_skel_handle_invoke&_modver=1.0\0"),
        81 => Some(b"file:///libggml-htp-v81.so?htp_iface_skel_handle_invoke&_modver=1.0\0"),
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
    let mut expected = module[..module.len() - 1].to_vec();
    expected.extend_from_slice(b"&_dom=cdsp&_session=0");
    bytes[..terminator] == expected
}

pub(crate) fn self_check() -> bool {
    let Some(module) = module_uri(81) else {
        return false;
    };
    let mut uri = module[..module.len() - 1].to_vec();
    uri.extend_from_slice(b"&_dom=cdsp&_session=0\0");
    checked_uri(81, &uri)
        && !checked_uri(73, &uri)
        && !SessionPreparationReport::new(0x81).metadata_prepared()
        && MAP_FD == 2
        && MAP_FD_DELAYED == 3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uri_is_bound_to_exact_architecture_domain_and_default_session() {
        assert!(self_check());
        let mut uri = module_uri(81).unwrap_or_default().to_vec();
        uri.pop();
        uri.extend_from_slice(b"&_dom=cdsp&_session=0\0");
        assert!(checked_uri(81, &uri));
        assert!(!checked_uri(82, &uri));
        assert!(!checked_uri(81, &uri[..uri.len() - 1]));
        uri.truncate(uri.len() - 2);
        uri.extend_from_slice(b"1\0");
        assert!(!checked_uri(81, &uri));
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
