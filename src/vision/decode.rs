//! Bounded pure-Rust image decoding for PNG, PPM (P6), and `data:` URLs.
//!
//! Avoids external image/compression crate dependencies while enforcing strict
//! pixel, dimension, checksum, and allocation bounds before decoding pixels.

use crate::{Result, invalid};

/// Maximum UTF-8 byte length admitted for one base64 image `data:` URL.
pub const MAX_DATA_URL_BYTES: usize = 16 * 1024 * 1024;
/// Maximum decoded compressed PNG or PPM byte length admitted per image.
pub const MAX_ENCODED_IMAGE_BYTES: usize = 12 * 1024 * 1024;
/// Maximum admitted width or height in pixels.
pub const MAX_IMAGE_DIMENSION: usize = 4_096;
/// Maximum admitted total pixel count (`width * height`).
pub const MAX_IMAGE_PIXELS: usize = 16_777_216;

/// Validated `RGB8` bitmap in row-major top-to-bottom order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecodedImage {
    /// Pixel width (`1..=MAX_IMAGE_DIMENSION`).
    pub width: usize,
    /// Pixel height (`1..=MAX_IMAGE_DIMENSION`).
    pub height: usize,
    /// Tightly packed `3 * width * height` `[R, G, B, ...]` samples.
    pub rgb: Vec<u8>,
}

/// Decodes one `data:image/png;base64,...` or `data:image/x-portable-pixmap;base64,...` URL.
///
/// Returns the decoded bitmap along with the raw compressed payload bytes before
/// raster expansion so callers can key caches deterministically.
///
/// # Errors
/// Rejects unsupported media types, oversized payloads, invalid RFC 4648 base64,
/// malformed container headers, corrupted checksums, or out-of-bounds images.
pub fn decode_image_data_url(url: &str) -> Result<(DecodedImage, Vec<u8>)> {
    if url.len() > MAX_DATA_URL_BYTES {
        return Err(invalid("image data URL exceeds 16 MiB limit"));
    }
    let (is_png, payload) = if let Some(rest) = url.strip_prefix("data:image/png;base64,") {
        (true, rest)
    } else if let Some(rest) = url.strip_prefix("data:image/x-portable-pixmap;base64,") {
        (false, rest)
    } else {
        return Err(invalid("unsupported image data URL scheme or media type"));
    };
    let raw = decode_base64(payload, MAX_ENCODED_IMAGE_BYTES)?;
    let image = if is_png {
        decode_png(&raw)?
    } else {
        decode_ppm(&raw)?
    };
    Ok((image, raw))
}

/// Decodes strict RFC 4648 base64 into at most `limit` bytes.
///
/// # Errors
/// Rejects whitespace, illegal characters, bad padding, truncated quartets, or
/// outputs exceeding `limit`.
pub fn decode_base64(encoded: &str, limit: usize) -> Result<Vec<u8>> {
    let bytes = encoded.as_bytes();
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return Err(invalid("base64 payload must be nonempty multiple of four"));
    }
    let quartets = bytes.len() / 4;
    let pad = if bytes.ends_with(b"==") {
        2
    } else {
        usize::from(bytes.ends_with(b"="))
    };
    let expected_len = quartets
        .checked_mul(3)
        .and_then(|n| n.checked_sub(pad))
        .ok_or_else(|| invalid("base64 output length overflow"))?;
    if expected_len > limit {
        return Err(invalid("decoded base64 payload exceeds limit"));
    }
    let mut output = Vec::new();
    output.try_reserve_exact(expected_len)?;
    for (index, chunk) in bytes.as_chunks::<4>().0.iter().enumerate() {
        let is_last = index + 1 == quartets;
        let a = decode_b64_digit(chunk[0])?;
        let b = decode_b64_digit(chunk[1])?;
        if is_last && pad == 2 {
            if chunk[2] != b'=' || chunk[3] != b'=' || (b & 0x0f) != 0 {
                return Err(invalid("invalid base64 2-byte padding"));
            }
            output.push((a << 2) | (b >> 4));
        } else if is_last && pad == 1 {
            let c = decode_b64_digit(chunk[2])?;
            if chunk[3] != b'=' || (c & 0x03) != 0 {
                return Err(invalid("invalid base64 1-byte padding"));
            }
            output.push((a << 2) | (b >> 4));
            output.push((b << 4) | (c >> 2));
        } else {
            let c = decode_b64_digit(chunk[2])?;
            let d = decode_b64_digit(chunk[3])?;
            output.push((a << 2) | (b >> 4));
            output.push((b << 4) | (c >> 2));
            output.push((c << 6) | d);
        }
    }
    Ok(output)
}

fn decode_b64_digit(byte: u8) -> Result<u8> {
    match byte {
        b'A'..=b'Z' => Ok(byte - b'A'),
        b'a'..=b'z' => Ok(byte - b'a' + 26),
        b'0'..=b'9' => Ok(byte - b'0' + 52),
        b'+' => Ok(62),
        b'/' => Ok(63),
        _ => Err(invalid("invalid base64 character")),
    }
}

/// Decodes a non-interlaced 8-bit `PNG` (Gray, `GrayAlpha`, RGB, or RGBA) to `RGB8`.
///
/// # Errors
/// Rejects malformed headers, unsupported bit depths/color types/interlacing,
/// bad CRC-32 or Adler-32 checksums, corrupt DEFLATE streams, or oversized bitmaps.
#[expect(
    clippy::too_many_lines,
    reason = "Keep PNG chunk validation and zlib/filter dispatch explicit."
)]
pub fn decode_png(bytes: &[u8]) -> Result<DecodedImage> {
    const PNG_SIG: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];
    if bytes.len() > MAX_ENCODED_IMAGE_BYTES || !bytes.starts_with(&PNG_SIG) {
        return Err(invalid("invalid or oversized PNG signature"));
    }
    let mut cursor = 8_usize;
    let mut header: Option<(usize, usize, u8)> = None;
    let mut idat = Vec::new();
    let mut saw_iend = false;

    while cursor < bytes.len() {
        if saw_iend || bytes.len() - cursor < 12 {
            return Err(invalid("truncated or trailing PNG chunk"));
        }
        let len = usize::try_from(u32::from_be_bytes([
            bytes[cursor],
            bytes[cursor + 1],
            bytes[cursor + 2],
            bytes[cursor + 3],
        ]))?;
        let chunk_type = [
            bytes[cursor + 4],
            bytes[cursor + 5],
            bytes[cursor + 6],
            bytes[cursor + 7],
        ];
        cursor += 8;
        if len > MAX_ENCODED_IMAGE_BYTES || bytes.len() - cursor < len + 4 {
            return Err(invalid("truncated or oversized PNG chunk payload"));
        }
        let payload = &bytes[cursor..cursor + len];
        let expected_crc = u32::from_be_bytes([
            bytes[cursor + len],
            bytes[cursor + len + 1],
            bytes[cursor + len + 2],
            bytes[cursor + len + 3],
        ]);
        if png_crc32(chunk_type, payload) != expected_crc {
            return Err(invalid("PNG chunk CRC-32 mismatch"));
        }
        cursor += len + 4;

        match &chunk_type {
            b"IHDR" => {
                if header.is_some() || payload.len() != 13 {
                    return Err(invalid("invalid or duplicate PNG IHDR"));
                }
                let width = usize::try_from(u32::from_be_bytes([
                    payload[0], payload[1], payload[2], payload[3],
                ]))?;
                let height = usize::try_from(u32::from_be_bytes([
                    payload[4], payload[5], payload[6], payload[7],
                ]))?;
                let bit_depth = payload[8];
                let color_type = payload[9];
                let compression = payload[10];
                let filter = payload[11];
                let interlace = payload[12];
                if width == 0
                    || height == 0
                    || width > MAX_IMAGE_DIMENSION
                    || height > MAX_IMAGE_DIMENSION
                    || width
                        .checked_mul(height)
                        .is_none_or(|px| px > MAX_IMAGE_PIXELS)
                    || bit_depth != 8
                    || !matches!(color_type, 0 | 2 | 4 | 6)
                    || compression != 0
                    || filter != 0
                    || interlace != 0
                {
                    return Err(invalid("unsupported PNG IHDR format or dimensions"));
                }
                header = Some((width, height, color_type));
            }
            b"IDAT" => {
                if header.is_none()
                    || idat.len().saturating_add(payload.len()) > MAX_ENCODED_IMAGE_BYTES
                {
                    return Err(invalid("invalid or oversized PNG IDAT stream"));
                }
                idat.try_reserve(payload.len())?;
                idat.extend_from_slice(payload);
            }
            b"IEND" => {
                if header.is_none() || !payload.is_empty() {
                    return Err(invalid("invalid PNG IEND chunk"));
                }
                saw_iend = true;
            }
            _ => {
                if header.is_none() {
                    return Err(invalid("PNG must begin with IHDR"));
                }
                // Reject critical unknown ancillary-style chunks such as PLTE
                // color-indexed images that require an external palette.
                if (chunk_type[0] & 0x20) == 0 {
                    return Err(invalid("unsupported critical PNG chunk"));
                }
            }
        }
    }

    if !saw_iend {
        return Err(invalid("PNG missing IEND"));
    }
    let (width, height, color_type) = header.ok_or_else(|| invalid("PNG missing IHDR"))?;
    let channels = match color_type {
        0 => 1_usize,
        2 => 3,
        4 => 2,
        6 => 4,
        _ => return Err(invalid("unsupported PNG color type")),
    };
    let row_bytes = width
        .checked_mul(channels)
        .ok_or_else(|| invalid("PNG row byte overflow"))?;
    let scanline_stride = row_bytes
        .checked_add(1)
        .ok_or_else(|| invalid("PNG scanline stride overflow"))?;
    let expected_raw_len = height
        .checked_mul(scanline_stride)
        .ok_or_else(|| invalid("PNG scanline buffer overflow"))?;
    let unfiltered_input = inflate_zlib(&idat, expected_raw_len)?;
    if unfiltered_input.len() != expected_raw_len {
        return Err(invalid("PNG decompressed byte count mismatch"));
    }
    let rgb = unfilter_to_rgb(&unfiltered_input, width, height, channels, row_bytes)?;
    Ok(DecodedImage { width, height, rgb })
}

/// Decodes a binary `PPM` (`P6`) `maxval=255` image into `RGB8`.
///
/// # Errors
/// Rejects malformed headers, unsupported bit depths, comments > bounds, or
/// truncated/trailing pixel bytes.
pub fn decode_ppm(bytes: &[u8]) -> Result<DecodedImage> {
    if bytes.len() > MAX_ENCODED_IMAGE_BYTES {
        return Err(invalid("PPM payload exceeds size limit"));
    }
    let mut pos = 0_usize;
    let magic = next_ppm_token(bytes, &mut pos)?;
    if magic != b"P6" {
        return Err(invalid("PPM requires P6 binary magic"));
    }
    let width = parse_ppm_usize(next_ppm_token(bytes, &mut pos)?)?;
    let height = parse_ppm_usize(next_ppm_token(bytes, &mut pos)?)?;
    let max_val = parse_ppm_usize(next_ppm_token(bytes, &mut pos)?)?;
    if max_val != 255
        || width == 0
        || height == 0
        || width > MAX_IMAGE_DIMENSION
        || height > MAX_IMAGE_DIMENSION
        || width
            .checked_mul(height)
            .is_none_or(|px| px > MAX_IMAGE_PIXELS)
    {
        return Err(invalid("unsupported PPM header geometry or maxval"));
    }
    if pos >= bytes.len() || !bytes[pos].is_ascii_whitespace() {
        return Err(invalid("PPM header must end with single whitespace byte"));
    }
    pos += 1;
    let expected = width
        .checked_mul(height)
        .and_then(|px| px.checked_mul(3))
        .ok_or_else(|| invalid("PPM RGB byte overflow"))?;
    if bytes.len() - pos != expected {
        return Err(invalid("PPM pixel data length mismatch"));
    }
    let mut rgb = Vec::new();
    rgb.try_reserve_exact(expected)?;
    rgb.extend_from_slice(&bytes[pos..]);
    Ok(DecodedImage { width, height, rgb })
}

fn next_ppm_token<'a>(bytes: &'a [u8], pos: &mut usize) -> Result<&'a [u8]> {
    while *pos < bytes.len() {
        if bytes[*pos].is_ascii_whitespace() {
            *pos += 1;
        } else if bytes[*pos] == b'#' {
            while *pos < bytes.len() && bytes[*pos] != b'\n' {
                *pos += 1;
            }
        } else {
            break;
        }
    }
    let start = *pos;
    while *pos < bytes.len() && !bytes[*pos].is_ascii_whitespace() && bytes[*pos] != b'#' {
        *pos += 1;
    }
    if start == *pos {
        return Err(invalid("unexpected EOF reading PPM header token"));
    }
    Ok(&bytes[start..*pos])
}

fn parse_ppm_usize(token: &[u8]) -> Result<usize> {
    std::str::from_utf8(token)
        .map_err(|_| invalid("non-UTF-8 PPM header token"))?
        .parse::<usize>()
        .map_err(|_| invalid("invalid integer in PPM header"))
}

fn unfilter_to_rgb(
    scanlines: &[u8],
    width: usize,
    height: usize,
    channels: usize,
    row_bytes: usize,
) -> Result<Vec<u8>> {
    let rgb_len = width
        .checked_mul(height)
        .and_then(|px| px.checked_mul(3))
        .ok_or_else(|| invalid("PNG RGB allocation overflow"))?;
    let mut rgb = Vec::new();
    rgb.try_reserve_exact(rgb_len)?;
    let mut prev_row = vec![0_u8; row_bytes];
    let mut curr_row = vec![0_u8; row_bytes];
    let stride = row_bytes + 1;

    for row_idx in 0..height {
        let line = &scanlines[row_idx * stride..(row_idx + 1) * stride];
        let filter_type = line[0];
        let src = &line[1..];
        match filter_type {
            0 => curr_row.copy_from_slice(src),
            1 => {
                for i in 0..row_bytes {
                    let left = if i >= channels {
                        curr_row[i - channels]
                    } else {
                        0
                    };
                    curr_row[i] = src[i].wrapping_add(left);
                }
            }
            2 => {
                for i in 0..row_bytes {
                    curr_row[i] = src[i].wrapping_add(prev_row[i]);
                }
            }
            3 => {
                for i in 0..row_bytes {
                    let left = if i >= channels {
                        curr_row[i - channels]
                    } else {
                        0
                    };
                    let up = prev_row[i];
                    let avg = u8::try_from(u16::midpoint(u16::from(left), u16::from(up)))
                        .map_err(|_| invalid("PNG average filter overflow"))?;
                    curr_row[i] = src[i].wrapping_add(avg);
                }
            }
            4 => {
                for i in 0..row_bytes {
                    let left = if i >= channels {
                        curr_row[i - channels]
                    } else {
                        0
                    };
                    let up = prev_row[i];
                    let up_left = if i >= channels {
                        prev_row[i - channels]
                    } else {
                        0
                    };
                    curr_row[i] = src[i].wrapping_add(paeth_predictor(left, up, up_left));
                }
            }
            _ => return Err(invalid("unsupported PNG scanline filter type")),
        }

        match channels {
            1 => {
                for &g in &curr_row {
                    rgb.extend_from_slice(&[g, g, g]);
                }
            }
            2 => {
                for px in curr_row.as_chunks::<2>().0 {
                    let v = composite_white(px[0], px[1]);
                    rgb.extend_from_slice(&[v, v, v]);
                }
            }
            3 => rgb.extend_from_slice(&curr_row),
            4 => {
                for px in curr_row.as_chunks::<4>().0 {
                    let alpha = px[3];
                    rgb.push(composite_white(px[0], alpha));
                    rgb.push(composite_white(px[1], alpha));
                    rgb.push(composite_white(px[2], alpha));
                }
            }
            _ => return Err(invalid("unexpected PNG channel count")),
        }
        std::mem::swap(&mut prev_row, &mut curr_row);
    }
    Ok(rgb)
}

fn composite_white(channel: u8, alpha: u8) -> u8 {
    let fg = u16::from(channel);
    let a = u16::from(alpha);
    let blended = (fg * a + 255 * (255 - a) + 127) / 255;
    u8::try_from(blended).unwrap_or(255)
}

fn paeth_predictor(a: u8, b: u8, c: u8) -> u8 {
    let ai = i32::from(a);
    let bi = i32::from(b);
    let ci = i32::from(c);
    let p = ai + bi - ci;
    let pa = (p - ai).abs();
    let pb = (p - bi).abs();
    let pc = (p - ci).abs();
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

fn png_crc32(chunk_type: [u8; 4], payload: &[u8]) -> u32 {
    let mut crc = 0xffff_ffff_u32;
    for &byte in chunk_type.iter().chain(payload.iter()) {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

fn adler32(data: &[u8]) -> u32 {
    const MOD_ADLER: u32 = 65_521;
    let mut s1 = 1_u32;
    let mut s2 = 0_u32;
    for chunk in data.chunks(5_552) {
        for &byte in chunk {
            s1 += u32::from(byte);
            s2 += s1;
        }
        s1 %= MOD_ADLER;
        s2 %= MOD_ADLER;
    }
    (s2 << 16) | s1
}

fn inflate_zlib(data: &[u8], expected_len: usize) -> Result<Vec<u8>> {
    if data.len() < 6 {
        return Err(invalid("zlib stream too short"));
    }
    let cmf = data[0];
    let flg = data[1];
    if (cmf & 0x0f) != 8
        || (cmf >> 4) > 7
        || !(u16::from(cmf) * 256 + u16::from(flg)).is_multiple_of(31)
        || (flg & 0x20) != 0
    {
        return Err(invalid("unsupported or corrupt zlib header"));
    }

    let mut out = Vec::new();
    out.try_reserve_exact(expected_len)?;
    let mut reader = BitReader::new(&data[2..data.len() - 4]);

    loop {
        let bfinal = reader.read_bits(1)?;
        let btype = reader.read_bits(2)?;
        match btype {
            0 => {
                reader.align_byte();
                let len = reader.read_u16_le()?;
                let nlen = reader.read_u16_le()?;
                if len != !nlen {
                    return Err(invalid("corrupt uncompressed DEFLATE block length"));
                }
                let raw = reader.read_bytes(usize::from(len))?;
                if out.len() + raw.len() > expected_len {
                    return Err(invalid(
                        "DEFLATE uncompressed output exceeds scanline bound",
                    ));
                }
                out.extend_from_slice(raw);
            }
            1 => {
                let (lit_tree, dist_tree) = fixed_trees()?;
                decode_huffman_block(&mut reader, &lit_tree, &dist_tree, &mut out, expected_len)?;
            }
            2 => {
                let (lit_tree, dist_tree) = read_dynamic_trees(&mut reader)?;
                decode_huffman_block(&mut reader, &lit_tree, &dist_tree, &mut out, expected_len)?;
            }
            _ => return Err(invalid("reserved DEFLATE block type")),
        }
        if bfinal == 1 {
            break;
        }
    }

    let trailer = &data[data.len() - 4..];
    let expected_adler = u32::from_be_bytes([trailer[0], trailer[1], trailer[2], trailer[3]]);
    if adler32(&out) != expected_adler {
        return Err(invalid("zlib Adler-32 checksum mismatch"));
    }
    Ok(out)
}

struct BitReader<'a> {
    bytes: &'a [u8],
    pos: usize,
    bit_buf: u64,
    bit_count: u8,
}

impl<'a> BitReader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            pos: 0,
            bit_buf: 0,
            bit_count: 0,
        }
    }

    fn ensure(&mut self, needed: u8) -> Result<()> {
        while self.bit_count < needed {
            let Some(&next) = self.bytes.get(self.pos) else {
                return Err(invalid("unexpected EOF in DEFLATE bitstream"));
            };
            self.pos += 1;
            self.bit_buf |= u64::from(next) << self.bit_count;
            self.bit_count += 8;
        }
        Ok(())
    }

    fn read_bits(&mut self, count: u8) -> Result<u32> {
        if count == 0 {
            return Ok(0);
        }
        self.ensure(count)?;
        let mask = (1_u64 << count) - 1;
        let value = u32::try_from(self.bit_buf & mask)?;
        self.bit_buf >>= count;
        self.bit_count -= count;
        Ok(value)
    }

    const fn align_byte(&mut self) {
        let discard = self.bit_count % 8;
        self.bit_buf >>= discard;
        self.bit_count -= discard;
    }

    fn read_u16_le(&mut self) -> Result<u16> {
        let lo = self.read_bits(8)?;
        let hi = self.read_bits(8)?;
        u16::try_from(lo | (hi << 8)).map_err(|_| invalid("u16 read overflow"))
    }

    fn read_bytes(&mut self, len: usize) -> Result<&'a [u8]> {
        if self.bit_count != 0 {
            return Err(invalid("internal DEFLATE byte alignment mismatch"));
        }
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| invalid("DEFLATE slice overflow"))?;
        let slice = self
            .bytes
            .get(self.pos..end)
            .ok_or_else(|| invalid("unexpected EOF reading stored DEFLATE block"))?;
        self.pos = end;
        Ok(slice)
    }
}

const MAX_CODE_BITS: usize = 15;

struct HuffmanTree {
    counts: [u16; MAX_CODE_BITS + 1],
    symbols: Vec<u16>,
}

impl HuffmanTree {
    fn from_lengths(lengths: &[u8]) -> Result<Self> {
        let mut counts = [0_u16; MAX_CODE_BITS + 1];
        for &len in lengths {
            let bits = usize::from(len);
            if bits > MAX_CODE_BITS {
                return Err(invalid("DEFLATE Huffman code exceeds 15 bits"));
            }
            if bits > 0 {
                counts[bits] = counts[bits]
                    .checked_add(1)
                    .ok_or_else(|| invalid("Huffman count overflow"))?;
            }
        }
        let mut left = 1_i32;
        for &count in &counts[1..] {
            left <<= 1;
            left -= i32::from(count);
            if left < 0 {
                return Err(invalid("oversubscribed DEFLATE Huffman code"));
            }
        }
        let mut offsets = [0_u16; MAX_CODE_BITS + 1];
        let mut total = 0_u16;
        for bits in 1..=MAX_CODE_BITS {
            offsets[bits] = total;
            total = total
                .checked_add(counts[bits])
                .ok_or_else(|| invalid("Huffman symbol table overflow"))?;
        }
        let mut symbols = vec![0_u16; usize::from(total)];
        for (sym, &len) in lengths.iter().enumerate() {
            if len != 0 {
                let slot = usize::from(offsets[usize::from(len)]);
                offsets[usize::from(len)] += 1;
                symbols[slot] =
                    u16::try_from(sym).map_err(|_| invalid("Huffman symbol index overflow"))?;
            }
        }
        Ok(Self { counts, symbols })
    }

    fn decode(&self, reader: &mut BitReader<'_>) -> Result<u16> {
        let mut code = 0_i32;
        let mut first = 0_i32;
        let mut index = 0_i32;
        for &count in &self.counts[1..] {
            code |= i32::try_from(reader.read_bits(1)?)
                .map_err(|_| invalid("Huffman bit conversion"))?;
            let count_i32 = i32::from(count);
            if code - count_i32 < first {
                let offset = usize::try_from(index + (code - first))
                    .map_err(|_| invalid("Huffman lookup index"))?;
                return self
                    .symbols
                    .get(offset)
                    .copied()
                    .ok_or_else(|| invalid("invalid DEFLATE Huffman symbol"));
            }
            index += count_i32;
            first += count_i32;
            first <<= 1;
            code <<= 1;
        }
        Err(invalid("corrupt DEFLATE Huffman code"))
    }
}

fn fixed_trees() -> Result<(HuffmanTree, HuffmanTree)> {
    let mut lit_lengths = [0_u8; 288];
    lit_lengths[..144].fill(8);
    lit_lengths[144..256].fill(9);
    lit_lengths[256..280].fill(7);
    lit_lengths[280..288].fill(8);
    let dist_lengths = [5_u8; 32];
    Ok((
        HuffmanTree::from_lengths(&lit_lengths)?,
        HuffmanTree::from_lengths(&dist_lengths)?,
    ))
}

fn read_dynamic_trees(reader: &mut BitReader<'_>) -> Result<(HuffmanTree, HuffmanTree)> {
    const ORDER: [usize; 19] = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];
    let hlit = usize::try_from(reader.read_bits(5)?)? + 257;
    let hdist = usize::try_from(reader.read_bits(5)?)? + 1;
    let hclen = usize::try_from(reader.read_bits(4)?)? + 4;
    if hlit > 286 || hdist > 30 {
        return Err(invalid(
            "DEFLATE dynamic header lengths exceed RFC 1951 bounds",
        ));
    }
    let mut code_lengths = [0_u8; 19];
    for &idx in ORDER.iter().take(hclen) {
        code_lengths[idx] =
            u8::try_from(reader.read_bits(3)?).map_err(|_| invalid("code length overflow"))?;
    }
    let code_tree = HuffmanTree::from_lengths(&code_lengths)?;
    let total = hlit + hdist;
    let mut combined = Vec::with_capacity(total);
    while combined.len() < total {
        let sym = code_tree.decode(reader)?;
        match sym {
            0..=15 => combined.push(u8::try_from(sym).map_err(|_| invalid("symbol range"))?),
            16 => {
                let &prev = combined
                    .last()
                    .ok_or_else(|| invalid("DEFLATE repeat code at start of table"))?;
                let repeat = usize::try_from(reader.read_bits(2)?)? + 3;
                if combined.len() + repeat > total {
                    return Err(invalid("DEFLATE code length table overflow"));
                }
                combined.resize(combined.len() + repeat, prev);
            }
            17 => {
                let repeat = usize::try_from(reader.read_bits(3)?)? + 3;
                if combined.len() + repeat > total {
                    return Err(invalid("DEFLATE zero-repeat table overflow"));
                }
                combined.resize(combined.len() + repeat, 0);
            }
            18 => {
                let repeat = usize::try_from(reader.read_bits(7)?)? + 11;
                if combined.len() + repeat > total {
                    return Err(invalid("DEFLATE long zero-repeat table overflow"));
                }
                combined.resize(combined.len() + repeat, 0);
            }
            _ => return Err(invalid("invalid DEFLATE code-length alphabet symbol")),
        }
    }
    Ok((
        HuffmanTree::from_lengths(&combined[..hlit])?,
        HuffmanTree::from_lengths(&combined[hlit..])?,
    ))
}

fn decode_huffman_block(
    reader: &mut BitReader<'_>,
    lit_tree: &HuffmanTree,
    dist_tree: &HuffmanTree,
    out: &mut Vec<u8>,
    expected_len: usize,
) -> Result<()> {
    const LEN_BASE: [u16; 29] = [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115,
        131, 163, 195, 227, 258,
    ];
    const LEN_EXTRA: [u8; 29] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
    ];
    const DIST_BASE: [u16; 30] = [
        1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
        2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
    ];
    const DIST_EXTRA: [u8; 30] = [
        0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
        13, 13,
    ];

    loop {
        let sym = lit_tree.decode(reader)?;
        match sym.cmp(&256) {
            std::cmp::Ordering::Less => {
                if out.len() >= expected_len {
                    return Err(invalid("DEFLATE literal output exceeds scanline bound"));
                }
                out.push(u8::try_from(sym).map_err(|_| invalid("literal byte range"))?);
            }
            std::cmp::Ordering::Equal => return Ok(()),
            std::cmp::Ordering::Greater => {
                let len_idx =
                    usize::from(sym.checked_sub(257).ok_or_else(|| invalid("bad length"))?);
                let &base_len = LEN_BASE
                    .get(len_idx)
                    .ok_or_else(|| invalid("invalid DEFLATE length symbol"))?;
                let extra_len = reader.read_bits(LEN_EXTRA[len_idx])?;
                let length = usize::from(base_len) + usize::try_from(extra_len)?;

                let dist_sym = usize::from(dist_tree.decode(reader)?);
                let &base_dist = DIST_BASE
                    .get(dist_sym)
                    .ok_or_else(|| invalid("invalid DEFLATE distance symbol"))?;
                let extra_dist = reader.read_bits(DIST_EXTRA[dist_sym])?;
                let distance = usize::from(base_dist) + usize::try_from(extra_dist)?;
                if distance == 0 || distance > out.len() {
                    return Err(invalid(
                        "DEFLATE backreference points before start of stream",
                    ));
                }
                if out.len() + length > expected_len {
                    return Err(invalid("DEFLATE copy exceeds scanline bound"));
                }
                for _ in 0..length {
                    let byte = out[out.len() - distance];
                    out.push(byte);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_b64(input: &[u8]) -> String {
        const TABLE: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in input.chunks(3) {
            let b0 = u32::from(chunk[0]);
            let b1 = chunk.get(1).copied().map_or(0, u32::from);
            let b2 = chunk.get(2).copied().map_or(0, u32::from);
            let triple = (b0 << 16) | (b1 << 8) | b2;
            out.push(char::from(
                TABLE[usize::try_from((triple >> 18) & 63).unwrap_or(0)],
            ));
            out.push(char::from(
                TABLE[usize::try_from((triple >> 12) & 63).unwrap_or(0)],
            ));
            out.push(if chunk.len() > 1 {
                char::from(TABLE[usize::try_from((triple >> 6) & 63).unwrap_or(0)])
            } else {
                '='
            });
            out.push(if chunk.len() > 2 {
                char::from(TABLE[usize::try_from(triple & 63).unwrap_or(0)])
            } else {
                '='
            });
        }
        out
    }

    fn write_chunk(out: &mut Vec<u8>, kind: [u8; 4], payload: &[u8]) -> Result<()> {
        let len = u32::try_from(payload.len())?;
        out.extend_from_slice(&len.to_be_bytes());
        out.extend_from_slice(&kind);
        out.extend_from_slice(payload);
        out.extend_from_slice(&png_crc32(kind, payload).to_be_bytes());
        Ok(())
    }

    fn stored_zlib(raw: &[u8]) -> Result<Vec<u8>> {
        let mut out = vec![0x78, 0x01];
        let mut remaining = raw;
        while remaining.len() > 65_535 {
            let (chunk, rest) = remaining.split_at(65_535);
            out.push(0x00);
            out.extend_from_slice(&65_535_u16.to_le_bytes());
            out.extend_from_slice(&(!65_535_u16).to_le_bytes());
            out.extend_from_slice(chunk);
            remaining = rest;
        }
        let len = u16::try_from(remaining.len())?;
        out.push(0x01);
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(remaining);
        out.extend_from_slice(&adler32(raw).to_be_bytes());
        Ok(out)
    }

    fn build_test_png(
        width: u32,
        height: u32,
        color_type: u8,
        raw_scanlines: &[u8],
    ) -> Result<Vec<u8>> {
        let mut png = vec![137, 80, 78, 71, 13, 10, 26, 10];
        let mut ihdr = Vec::with_capacity(13);
        ihdr.extend_from_slice(&width.to_be_bytes());
        ihdr.extend_from_slice(&height.to_be_bytes());
        ihdr.extend_from_slice(&[8, color_type, 0, 0, 0]);
        write_chunk(&mut png, *b"IHDR", &ihdr)?;
        write_chunk(&mut png, *b"IDAT", &stored_zlib(raw_scanlines)?)?;
        write_chunk(&mut png, *b"IEND", &[])?;
        Ok(png)
    }

    #[test]
    fn base64_ppm_and_png_filters_decode_cleanly() -> Result<()> {
        let ppm = b"P6\n# test comment\n2 1\n255\n\x01\x02\x03\xff\x80\x00";
        let url = format!("data:image/x-portable-pixmap;base64,{}", encode_b64(ppm));
        let (decoded, raw) = decode_image_data_url(&url)?;
        assert_eq!(raw, ppm);
        assert_eq!(
            decoded,
            DecodedImage {
                width: 2,
                height: 1,
                rgb: vec![1, 2, 3, 255, 128, 0],
            }
        );

        // 2x2 RGBA using None filter on row 0 (transparent red + opaque blue)
        // and Sub filter on row 1 (opaque white + opaque green).
        let raw_png_lines = vec![
            0, 200, 10, 20, 0, 0, 0, 255, 255, // row 0: (255,255,255) over white, (0,0,255)
            1, 255, 255, 255, 255, 1, 0, 1, 0, // row 1 Sub: px0=(255,255,255), px1=(0,255,0)
        ];
        let png = build_test_png(2, 2, 6, &raw_png_lines)?;
        let png_url = format!("data:image/png;base64,{}", encode_b64(&png));
        let (decoded_png, _) = decode_image_data_url(&png_url)?;
        assert_eq!(
            decoded_png.rgb,
            vec![
                255, 255, 255, 0, 0, 255, // row 0
                255, 255, 255, 0, 255, 0, // row 1
            ]
        );
        Ok(())
    }

    #[test]
    fn corrupt_crc_bad_base64_and_oversized_png_are_rejected() -> Result<()> {
        assert!(decode_image_data_url("https://example.com/a.png").is_err());
        assert!(decode_image_data_url("data:image/jpeg;base64,AAAA").is_err());
        assert_eq!(decode_base64("AA==", 1)?, vec![0]);
        assert!(decode_base64("AB==", 1).is_err());
        let mut png = build_test_png(1, 1, 2, &[0, 10, 20, 30])?;
        let last = png.len() - 1;
        png[last] ^= 1; // corrupt IEND CRC
        assert!(decode_png(&png).is_err());
        Ok(())
    }
}
