//! Shared HTTP wire mechanics; request and response policy stays with callers.

#![expect(
    clippy::redundant_pub_crate,
    reason = "Keep shared HTTP helpers explicitly crate scoped"
)]

pub(super) fn valid_line_endings(raw: &[u8]) -> bool {
    !raw.iter().enumerate().any(|(i, byte)| match byte {
        b'\r' => raw.get(i + 1) != Some(&b'\n'),
        b'\n' => i == 0 || raw[i - 1] != b'\r',
        _ => false,
    })
}

const fn valid_header_name_byte(byte: u8) -> bool {
    matches!(
        byte,
        b'0'..=b'9'
            | b'a'..=b'z'
            | b'A'..=b'Z'
            | b'!'
            | b'#'
            | b'$'
            | b'%'
            | b'&'
            | b'\''
            | b'*'
            | b'+'
            | b'-'
            | b'.'
            | b'^'
            | b'_'
            | b'`'
            | b'|'
            | b'~'
    )
}

pub(super) fn valid_headers(headers: &[httparse::Header<'_>]) -> bool {
    headers.iter().all(|header| {
        !header.name.is_empty()
            && header.name.bytes().all(valid_header_name_byte)
            && header
                .value
                .iter()
                .all(|&byte| byte == b'\t' || (byte >= 0x20 && byte != 0x7f))
    })
}
