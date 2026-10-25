//! Shared HTTP wire mechanics; request and response policy stays with callers.

#![allow(
    clippy::redundant_pub_crate,
    reason = "Keep shared HTTP helpers explicitly crate scoped"
)]

use http::{HeaderMap, HeaderName, HeaderValue};

pub(super) fn valid_line_endings(raw: &[u8]) -> bool {
    !raw.iter().enumerate().any(|(i, byte)| match byte {
        b'\r' => raw.get(i + 1) != Some(&b'\n'),
        b'\n' => i == 0 || raw[i - 1] != b'\r',
        _ => false,
    })
}

pub(super) fn append_headers(
    target: &mut HeaderMap,
    headers: &[httparse::Header<'_>],
) -> Result<(), ()> {
    for header in headers {
        let name = HeaderName::from_bytes(header.name.as_bytes()).map_err(|_| ())?;
        let value = HeaderValue::from_bytes(header.value).map_err(|_| ())?;
        // insert() would hide duplicate critical headers, including identical CL.
        target.append(name, value);
    }
    Ok(())
}
