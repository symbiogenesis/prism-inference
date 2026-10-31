//! Two fixed session-metadata controls; no remote handle or mapping is created.

use super::{Driver, GetProcAddress, last_error};
use crate::{Capability, Error, SessionPreparationReport, session};
use std::ffi::{c_char, c_void};

#[allow(
    clippy::redundant_pub_crate,
    reason = "Only sibling fixed-session paths share this private ABI alias"
)]
pub(super) type SessionControl = unsafe extern "C" fn(u32, *mut c_void, u32) -> i32;

#[repr(C)]
#[allow(
    clippy::struct_field_names,
    reason = "The pinned public C ABI names its output effective_domain_id"
)]
struct EffectiveDomain {
    domain_name: *mut c_char,
    domain_name_len: u32,
    session_id: u32,
    effective_domain_id: u32,
}

#[repr(C)]
struct GetUri {
    domain_name: *mut c_char,
    domain_name_len: u32,
    session_id: u32,
    module_uri: *mut c_char,
    module_uri_len: u32,
    uri: *mut c_char,
    uri_len: u32,
}

const _: () = assert!(size_of::<EffectiveDomain>() == 24);
const _: () = assert!(align_of::<EffectiveDomain>() == 8);
const _: () = assert!(std::mem::offset_of!(EffectiveDomain, effective_domain_id) == 16);
const _: () = assert!(size_of::<GetUri>() == 48);
const _: () = assert!(align_of::<GetUri>() == 8);
const _: () = assert!(std::mem::offset_of!(GetUri, module_uri) == 16);
const _: () = assert!(std::mem::offset_of!(GetUri, uri) == 32);
const _: () = assert!(std::mem::offset_of!(GetUri, uri_len) == 40);

#[allow(
    clippy::redundant_pub_crate,
    reason = "The native helper remains private to its safe parent API"
)]
pub(super) fn prepare(driver: &mut Driver) -> Result<SessionPreparationReport, Error> {
    prepare_with_uri(driver, &mut [0xa5; 256])
}

#[allow(
    clippy::redundant_pub_crate,
    reason = "The validated returned URI is private input to the fixed native open"
)]
pub(super) fn prepare_with_uri(
    driver: &mut Driver,
    uri: &mut [u8; 256],
) -> Result<SessionPreparationReport, Error> {
    let raw = driver.capability(Capability::Architecture)?;
    let report = SessionPreparationReport::new(raw);
    if report.architecture.is_none() {
        return Ok(report);
    }
    prepare_fixed_metadata(raw, session_control(driver)?, uri)
}

#[allow(
    clippy::redundant_pub_crate,
    reason = "Only the private fixed session paths use this pinned export"
)]
pub(super) fn session_control(driver: &Driver) -> Result<SessionControl, Error> {
    // SAFETY: the retained driver owns the DLL and the fixed export is NUL terminated.
    let symbol =
        unsafe { GetProcAddress(driver.library.as_ptr(), c"remote_session_control".as_ptr()) };
    if symbol.is_null() {
        return Err(last_error("session-metadata-export"));
    }
    // SAFETY: the pinned vendor header specifies this exact C function signature.
    Ok(unsafe { std::mem::transmute::<*mut c_void, SessionControl>(symbol) })
}

fn prepare_fixed_metadata(
    raw: u32,
    control: SessionControl,
    uri: &mut [u8; 256],
) -> Result<SessionPreparationReport, Error> {
    prepare_fixed_uri(prepare_effective_domain(raw, control), control, uri)
}

#[allow(
    clippy::redundant_pub_crate,
    reason = "The reference initializer must occur after domain metadata and before URI formatting"
)]
pub(super) fn prepare_effective_domain(
    raw: u32,
    control: SessionControl,
) -> SessionPreparationReport {
    let mut report = SessionPreparationReport::new(raw);
    if report.architecture.is_none() {
        return report;
    }

    let mut domain = *b"cdsp\0";
    let domain_ptr = domain.as_mut_ptr().cast::<c_char>();
    let mut request = EffectiveDomain {
        domain_name: domain_ptr,
        domain_name_len: 4,
        session_id: 0,
        effective_domain_id: u32::MAX,
    };
    // SAFETY: request has the pinned 24-byte C layout and both writable buffers
    // stay alive. Only cdsp/session0 metadata is requested; no process is opened.
    let result = unsafe {
        control(
            session::GET_EFFECTIVE_DOMAIN_ID,
            std::ptr::from_mut(&mut request).cast(),
            24,
        )
    };
    report.domain_result = Some(result);
    if result != 0 {
        report.stage = "effective-domain-refused";
        return report;
    }
    if request.domain_name != domain_ptr
        || request.domain_name_len != 4
        || request.session_id != 0
        || domain != *b"cdsp\0"
    {
        report.stage = "effective-domain-contract-changed";
        return report;
    }
    let Ok(effective_domain) = i32::try_from(request.effective_domain_id) else {
        report.stage = "effective-domain-out-of-range";
        return report;
    };
    report.effective_domain = Some(effective_domain);

    report.stage = "effective-domain-prepared";
    report
}

#[allow(
    clippy::redundant_pub_crate,
    reason = "Only the two private fixed session paths may format this module URI"
)]
pub(super) fn prepare_fixed_uri(
    mut report: SessionPreparationReport,
    control: SessionControl,
    uri: &mut [u8; 256],
) -> Result<SessionPreparationReport, Error> {
    if report.effective_domain.is_none() || report.domain_result != Some(0) {
        return Ok(report);
    }
    let Some(architecture) = report.architecture else {
        return Ok(report);
    };
    let module = session::module_uri(architecture).ok_or(Error::Contract)?;
    // The vendor ABI excludes the terminator from the declared input length,
    // while the independently writable backing allocation retains that NUL.
    let module_len = u32::try_from(module.to_bytes().len()).map_err(|_| Error::Contract)?;
    let mut domain = *b"cdsp\0";
    let domain_ptr = domain.as_mut_ptr().cast::<c_char>();
    // Mutable owned storage satisfies the C ABI without allowing a provider to
    // write into a Rust string literal. The input is checked again after return.
    let mut module_bytes = module.to_bytes_with_nul().to_vec();
    let module_ptr = module_bytes.as_mut_ptr().cast::<c_char>();
    uri.fill(0xa5);
    let uri_ptr = uri.as_mut_ptr().cast::<c_char>();
    let mut request = GetUri {
        domain_name: domain_ptr,
        domain_name_len: 4,
        session_id: 0,
        module_uri: module_ptr,
        module_uri_len: module_len,
        uri: uri_ptr,
        uri_len: 256,
    };
    // SAFETY: this is the pinned 48-byte C layout, with writable and live input
    // and output arrays. GET_URI formats metadata; it does not load the module.
    let result = unsafe {
        control(
            session::GET_URI,
            std::ptr::from_mut(&mut request).cast(),
            48,
        )
    };
    report.uri_result = Some(result);
    if result != 0 {
        report.stage = "session-uri-refused";
        return Ok(report);
    }
    if request.domain_name != domain_ptr
        || request.domain_name_len != 4
        || request.session_id != 0
        || request.module_uri != module_ptr
        || request.module_uri_len != module_len
        || request.uri != uri_ptr
        || request.uri_len != 256
        || domain != *b"cdsp\0"
        || module_bytes != module.to_bytes_with_nul()
    {
        report.stage = "session-uri-contract-changed";
        return Ok(report);
    }
    report.uri_verified = session::checked_uri(architecture, uri);
    report.stage = if report.uri_verified {
        "session-metadata-prepared"
    } else {
        "session-uri-not-exact"
    };
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" fn accepted(request: u32, data: *mut c_void, bytes: u32) -> i32 {
        match (request, bytes) {
            (session::GET_EFFECTIVE_DOMAIN_ID, 24) => {
                // SAFETY: the tested caller provides this exact writable C request.
                unsafe { (*data.cast::<EffectiveDomain>()).effective_domain_id = 3 };
            }
            (session::GET_URI, 48) => {
                // SAFETY: the tested caller provides this exact live request.
                let input = unsafe { &mut *data.cast::<GetUri>() };
                // SAFETY: the fixed caller owns this complete declared input extent.
                let module = unsafe {
                    std::slice::from_raw_parts(
                        input.module_uri.cast::<u8>(),
                        usize::try_from(input.module_uri_len).unwrap_or_default(),
                    )
                };
                // SAFETY: the tested caller supplies the exact 256-byte output array.
                let output = unsafe { std::slice::from_raw_parts_mut(input.uri.cast::<u8>(), 256) };
                output[..module.len()].copy_from_slice(module);
                let suffix = b"&_dom=cdsp&_session=0\0";
                output[module.len()..module.len() + suffix.len()].copy_from_slice(suffix);
            }
            _ => return 14,
        }
        0
    }

    unsafe extern "C" fn domain_refused(_: u32, _: *mut c_void, _: u32) -> i32 {
        512
    }

    unsafe extern "C" fn uri_refused(request: u32, data: *mut c_void, bytes: u32) -> i32 {
        if request == session::GET_EFFECTIVE_DOMAIN_ID {
            // SAFETY: this forwards the exact request and extent to the fixed mock.
            unsafe { accepted(request, data, bytes) }
        } else {
            512
        }
    }

    unsafe extern "C" fn input_changed(request: u32, data: *mut c_void, bytes: u32) -> i32 {
        // SAFETY: this forwards the exact request and extent to the fixed mock.
        let result = unsafe { accepted(request, data, bytes) };
        if request == session::GET_URI {
            // SAFETY: the tested caller provides this exact writable C request.
            unsafe { (*data.cast::<GetUri>()).uri = std::ptr::null_mut() };
        }
        result
    }

    #[test]
    fn session_metadata_retains_both_codes_without_failure_fallback() -> Result<(), Error> {
        for architecture in [0x73, 0x81] {
            let success = prepare_fixed_metadata(architecture, accepted, &mut [0xa5; 256])?;
            assert!(success.metadata_prepared());
            assert_eq!(success.effective_domain, Some(3));
        }
        let refused = prepare_fixed_metadata(0x81, domain_refused, &mut [0xa5; 256])?;
        assert_eq!(refused.domain_result, Some(512));
        assert_eq!(refused.uri_result, None);
        assert!(!refused.metadata_prepared());
        let refused = prepare_fixed_metadata(0x81, uri_refused, &mut [0xa5; 256])?;
        assert_eq!(refused.domain_result, Some(0));
        assert_eq!(refused.uri_result, Some(512));
        assert!(!refused.metadata_prepared());
        let unsupported = prepare_fixed_metadata(0x82, domain_refused, &mut [0xa5; 256])?;
        assert_eq!(unsupported.domain_result, None);
        Ok(())
    }

    #[test]
    fn changed_native_pointer_is_rejected_without_dereferencing_it() -> Result<(), Error> {
        let report = prepare_fixed_metadata(0x73, input_changed, &mut [0xa5; 256])?;
        assert_eq!(report.stage, "session-uri-contract-changed");
        assert!(!report.metadata_prepared());
        Ok(())
    }
}
