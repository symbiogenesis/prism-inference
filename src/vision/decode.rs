//! Bounded pure-Rust image decoding for PNG, PPM (P6), and `data:` URLs.
//!
//! Uses `#![forbid(unsafe_code)]` `base64ct` and `fdeflate` alongside `crc32fast`
//! while enforcing strict pixel, dimension, aspect-ratio, checksum, and allocation
//! bounds before expanding pixels.

use base64ct::{Base64, Encoding as _};

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

/// Supported container formats encoded in `data:` image URLs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageContainerFormat {
    /// Portable Network Graphics (`data:image/png;base64,...`).
    Png,
    /// Binary Portable Pixmap (`data:image/x-portable-pixmap;base64,...`).
    Ppm,
}

/// Visual token and grid budget enforced on container headers before pixel inflation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImageDimensionBudget {
    /// Minimum planned vision tokens (`>= 1`).
    pub min_tokens: usize,
    /// Maximum planned vision tokens (`>= min_tokens`).
    pub max_tokens: usize,
    /// Spatial patch-merge factor (`patch_size * spatial_merge`).
    pub grid_factor: usize,
}

impl ImageDimensionBudget {
    /// Validate static token/grid bounds before parsing container payloads.
    ///
    /// # Errors
    /// Rejects zero factors/bounds, `min_tokens > max_tokens`, or pixel-bound overflows.
    pub fn validate_bounds(self) -> Result<()> {
        if self.grid_factor == 0 || self.min_tokens == 0 || self.min_tokens > self.max_tokens {
            return Err(invalid("invalid vision grid planning bounds"));
        }
        let factor_sq = self
            .grid_factor
            .checked_mul(self.grid_factor)
            .ok_or_else(|| invalid("vision factor area overflow"))?;
        self.max_tokens
            .checked_mul(factor_sq)
            .ok_or_else(|| invalid("vision max pixel bound overflow"))?;
        Ok(())
    }

    /// Validate header image dimensions against budget and `200:1` aspect ratio rules.
    ///
    /// # Errors
    /// Rejects invalid bounds, zero dimensions, or extreme aspect ratios before pixel decode.
    pub fn validate_dimensions(self, width: usize, height: usize) -> Result<()> {
        self.validate_bounds()?;
        if width == 0 || height == 0 {
            return Err(invalid("invalid vision grid planning bounds"));
        }
        let min_side = width.min(height);
        let max_side = width.max(height);
        if max_side > min_side.saturating_mul(200) {
            return Err(invalid("image aspect ratio exceeds 200:1 bound"));
        }
        Ok(())
    }
}

/// Extracts and decodes the raw compressed container bytes from one `data:` image URL.
///
/// Allows callers to hash compressed bytes and consult caches before PNG/PPM pixel inflation.
///
/// # Errors
/// Rejects unsupported media types, oversized URLs or payloads, and invalid RFC 4648 base64.
pub fn decode_image_data_url_raw(url: &str) -> Result<(ImageContainerFormat, Vec<u8>)> {
    if url.len() > MAX_DATA_URL_BYTES {
        return Err(invalid("image data URL exceeds 16 MiB limit"));
    }
    let (format, payload) = if let Some(rest) = url.strip_prefix("data:image/png;base64,") {
        (ImageContainerFormat::Png, rest)
    } else if let Some(rest) = url.strip_prefix("data:image/x-portable-pixmap;base64,") {
        (ImageContainerFormat::Ppm, rest)
    } else {
        return Err(invalid("unsupported image data URL scheme or media type"));
    };
    let raw = decode_base64(payload, MAX_ENCODED_IMAGE_BYTES)?;
    Ok((format, raw))
}

/// Decodes raw PNG or PPM container bytes into `RGB8`, guarding dimensions before pixel inflation.
///
/// # Errors
/// Rejects invalid budgets, out-of-bounds header dimensions, malformed headers,
/// corrupted checksums, or invalid compressed streams.
pub fn decode_encoded_image_within(
    format: ImageContainerFormat,
    raw: &[u8],
    budget: ImageDimensionBudget,
) -> Result<DecodedImage> {
    budget.validate_bounds()?;
    match format {
        ImageContainerFormat::Png => decode_png_with_budget(raw, Some(budget)),
        ImageContainerFormat::Ppm => decode_ppm_with_budget(raw, Some(budget)),
    }
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
    let (format, raw) = decode_image_data_url_raw(url)?;
    let image = match format {
        ImageContainerFormat::Png => decode_png(&raw)?,
        ImageContainerFormat::Ppm => decode_ppm(&raw)?,
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
    output.resize(expected_len, 0);
    let decoded_len = Base64::decode(bytes, &mut output)
        .map_err(|_| invalid("invalid base64 payload"))?
        .len();
    if decoded_len != expected_len {
        return Err(invalid("invalid base64 padding length"));
    }
    Ok(output)
}

/// Decodes a non-interlaced 8-bit `PNG` (Gray, `GrayAlpha`, RGB, or RGBA) to `RGB8`.
///
/// # Errors
/// Rejects malformed headers, unsupported bit depths/color types/interlacing,
/// bad CRC-32 or Adler-32 checksums, corrupt DEFLATE streams, or oversized bitmaps.
pub fn decode_png(bytes: &[u8]) -> Result<DecodedImage> {
    decode_png_with_budget(bytes, None)
}

#[expect(
    clippy::too_many_lines,
    reason = "Keep PNG chunk validation and zlib/filter dispatch explicit."
)]
fn decode_png_with_budget(
    bytes: &[u8],
    budget: Option<ImageDimensionBudget>,
) -> Result<DecodedImage> {
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
                if let Some(dim_budget) = budget {
                    dim_budget.validate_dimensions(width, height)?;
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
    decode_ppm_with_budget(bytes, None)
}

fn decode_ppm_with_budget(
    bytes: &[u8],
    budget: Option<ImageDimensionBudget>,
) -> Result<DecodedImage> {
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
    if let Some(dim_budget) = budget {
        dim_budget.validate_dimensions(width, height)?;
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
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&chunk_type);
    hasher.update(payload);
    hasher.finalize()
}

fn inflate_zlib(data: &[u8], expected_len: usize) -> Result<Vec<u8>> {
    const CHUNK_GROWTH: usize = 32 * 1024;
    if data.len() < 6 {
        return Err(invalid("zlib stream too short"));
    }
    let max_out = expected_len
        .checked_add(1)
        .ok_or_else(|| invalid("PNG scanline buffer overflow"))?;
    let initial = max_out.min(CHUNK_GROWTH);
    let mut output = Vec::new();
    output.try_reserve_exact(initial)?;
    output.resize(initial, 0);

    let mut decoder = fdeflate::Decompressor::new();
    let body_end = data.len() - 4;
    let mut input_index = 0_usize;
    let mut output_index = 0_usize;
    while !decoder.is_done() {
        let (chunk, end_of_input) = if input_index < body_end {
            (&data[input_index..body_end], false)
        } else if input_index < data.len() {
            (
                &data[input_index..=input_index],
                input_index + 1 == data.len(),
            )
        } else {
            (&[][..], true)
        };
        let (consumed, produced) = decoder
            .read(chunk, &mut output, output_index, end_of_input)
            .map_err(|_| invalid("corrupt PNG zlib stream"))?;
        input_index += consumed;
        output_index += produced;
        if output_index > expected_len {
            return Err(invalid("DEFLATE output exceeds scanline bound"));
        }
        if decoder.is_done() {
            break;
        }
        if consumed == 0 && produced == 0 {
            if output.len() >= max_out {
                return Err(invalid("DEFLATE output exceeds scanline bound"));
            }
            let next_len = output.len().saturating_add(CHUNK_GROWTH).min(max_out);
            output.try_reserve(next_len - output.len())?;
            output.resize(next_len, 0);
        }
    }
    if !decoder.is_done() || input_index != data.len() || output_index != expected_len {
        return Err(invalid("PNG decompressed byte count mismatch"));
    }
    output.truncate(expected_len);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_base64(bytes: &[u8]) -> Result<String> {
        let mut out = vec![0; Base64::encoded_len(bytes)];
        let encoded =
            Base64::encode(bytes, &mut out).map_err(|_| invalid("test base64 encode failed"))?;
        Ok(encoded.to_owned())
    }

    fn test_adler32(data: &[u8]) -> u32 {
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
        out.extend_from_slice(&test_adler32(raw).to_be_bytes());
        Ok(out)
    }

    fn build_png_from_zlib(
        width: u32,
        height: u32,
        color_type: u8,
        zlib: &[u8],
    ) -> Result<Vec<u8>> {
        let mut png = vec![137, 80, 78, 71, 13, 10, 26, 10];
        let mut ihdr = Vec::with_capacity(13);
        ihdr.extend_from_slice(&width.to_be_bytes());
        ihdr.extend_from_slice(&height.to_be_bytes());
        ihdr.extend_from_slice(&[8, color_type, 0, 0, 0]);
        write_chunk(&mut png, *b"IHDR", &ihdr)?;
        write_chunk(&mut png, *b"IDAT", zlib)?;
        write_chunk(&mut png, *b"IEND", &[])?;
        Ok(png)
    }

    fn build_test_png(
        width: u32,
        height: u32,
        color_type: u8,
        raw_scanlines: &[u8],
    ) -> Result<Vec<u8>> {
        build_png_from_zlib(width, height, color_type, &stored_zlib(raw_scanlines)?)
    }

    #[test]
    fn base64_ppm_and_png_filters_decode_cleanly() -> Result<()> {
        let ppm = b"P6\n# test comment\n2 1\n255\n\x01\x02\x03\xff\x80\x00";
        let url = format!(
            "data:image/x-portable-pixmap;base64,{}",
            encode_base64(ppm)?
        );
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
        let png_url = format!("data:image/png;base64,{}", encode_base64(&png)?);
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

    fn bitwise_png_crc32(chunk_type: [u8; 4], payload: &[u8]) -> u32 {
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

    fn push_bits_lsb(out: &mut Vec<u8>, acc: &mut u32, acc_bits: &mut u8, val: u32, bits: u8) {
        *acc |= val << *acc_bits;
        *acc_bits += bits;
        while *acc_bits >= 8 {
            out.push((*acc & 0xff).to_le_bytes()[0]);
            *acc >>= 8;
            *acc_bits -= 8;
        }
    }

    fn fixed_lit_code(sym: u16) -> (u32, u8) {
        let (msb_code, bits) = match sym {
            0..=143 => (u32::from(sym) + 0b0011_0000, 8_u8),
            144..=255 => (u32::from(sym - 144) + 0b1_1001_0000, 9_u8),
            256..=279 => (u32::from(sym - 256), 7_u8),
            _ => (u32::from(sym - 280) + 0b1100_0000, 8_u8),
        };
        let rev = msb_code.reverse_bits() >> (32 - bits);
        (rev, bits)
    }

    #[test]
    fn png_crc32_matches_bitwise_reference() {
        let mut payload = Vec::new();
        for len in 0..=40_usize {
            assert_eq!(
                png_crc32(*b"IDAT", &payload),
                bitwise_png_crc32(*b"IDAT", &payload),
                "CRC mismatch at payload length {len}"
            );
            let next_byte = u8::try_from((len * 73 + 19) & 0xff).unwrap_or(0);
            payload.push(next_byte);
        }
        assert_eq!(png_crc32(*b"IEND", &[]), bitwise_png_crc32(*b"IEND", &[]));
    }

    #[test]
    fn deflate_decodes_compressed_and_mixed_blocks_and_rejects_corrupt_streams() -> Result<()> {
        // Test mixed DEFLATE zlib stream: Block 1 fixed Huffman (BFINAL=0) [0, 10, 20] + EOB,
        // followed by Block 2 stored (BFINAL=1) [30].
        let mut zlib = vec![0x78, 0x9c];
        let mut deflate_acc = 0_u32;
        let mut deflate_bits = 0_u8;
        push_bits_lsb(&mut zlib, &mut deflate_acc, &mut deflate_bits, 0, 1); // BFINAL=0
        push_bits_lsb(&mut zlib, &mut deflate_acc, &mut deflate_bits, 1, 2); // BTYPE=1 (fixed)
        for sym in [0_u16, 10, 20, 256] {
            let (cw, len) = fixed_lit_code(sym);
            push_bits_lsb(&mut zlib, &mut deflate_acc, &mut deflate_bits, cw, len);
        }
        push_bits_lsb(&mut zlib, &mut deflate_acc, &mut deflate_bits, 1, 1); // BFINAL=1
        push_bits_lsb(&mut zlib, &mut deflate_acc, &mut deflate_bits, 0, 2); // BTYPE=0 (stored)
        if deflate_bits > 0 {
            zlib.push((deflate_acc & 0xff).to_le_bytes()[0]);
        }
        let len_u16 = 1_u16;
        zlib.extend_from_slice(&len_u16.to_le_bytes());
        zlib.extend_from_slice(&(!len_u16).to_le_bytes());
        zlib.push(30_u8);
        zlib.extend_from_slice(&test_adler32(&[0, 10, 20, 30]).to_be_bytes());

        let decoded = decode_png(&build_png_from_zlib(1, 1, 2, &zlib)?)?;
        assert_eq!(decoded.rgb, vec![10, 20, 30]);

        // Compressed fdeflate stream round-trip (larger than initial 32 KiB buffer).
        let row_count = 130_u32;
        let row_bytes = 256_usize * 3;
        let mut raw_large = Vec::with_capacity(usize::try_from(row_count)? * (row_bytes + 1));
        for y in 0..row_count {
            raw_large.push(0); // None filter
            for x in 0..row_bytes {
                raw_large.push(u8::try_from((usize::try_from(y)? * 17 + x) & 0xff).unwrap_or(0));
            }
        }
        let compressed_zlib = fdeflate::compress_to_vec(&raw_large);
        let decoded_large = decode_png(&build_png_from_zlib(256, row_count, 2, &compressed_zlib)?)?;
        assert_eq!(
            decoded_large.rgb.len(),
            256 * usize::try_from(row_count)? * 3
        );

        // Corrupt Adler-32 trailer must be rejected even with valid chunk CRC.
        let mut bad_adler = zlib.clone();
        let last_idx = bad_adler.len() - 1;
        bad_adler[last_idx] ^= 1;
        assert!(decode_png(&build_png_from_zlib(1, 1, 2, &bad_adler)?).is_err());

        // Trailing garbage after valid zlib trailer must be rejected.
        let mut trailing = zlib;
        trailing.push(0x00);
        assert!(decode_png(&build_png_from_zlib(1, 1, 2, &trailing)?).is_err());

        // Overlong uncompressed IDAT payload exceeding expected scanline length must be rejected.
        let overlong = stored_zlib(&[0, 10, 20, 30, 40])?;
        assert!(decode_png(&build_png_from_zlib(1, 1, 2, &overlong)?).is_err());

        // Stored (>32 KiB) zlib stream crossing initial output capacity boundary near trailer.
        let stored_rows = 128_u32;
        let stored_row_bytes = 256_usize * 3;
        let mut raw_stored_large =
            Vec::with_capacity(usize::try_from(stored_rows)? * (stored_row_bytes + 1));
        for y in 0..stored_rows {
            raw_stored_large.push(0);
            for x in 0..stored_row_bytes {
                raw_stored_large
                    .push(u8::try_from((usize::try_from(y)? * 13 + x) & 0xff).unwrap_or(0));
            }
        }
        let stored_large_zlib = stored_zlib(&raw_stored_large)?;
        let decoded_stored_large =
            decode_png(&build_png_from_zlib(256, stored_rows, 2, &stored_large_zlib)?)?;
        assert_eq!(
            decoded_stored_large.rgb.len(),
            256 * usize::try_from(stored_rows)? * 3
        );
        Ok(())
    }

    #[test]
    fn decode_image_data_url_raw_and_early_ihdr_budget_guard() -> Result<()> {
        let png = build_test_png(1, 1, 2, &[0, 11, 22, 33])?;
        let url = format!("data:image/png;base64,{}", encode_base64(&png)?);
        let (fmt, raw) = decode_image_data_url_raw(&url)?;
        assert_eq!(fmt, ImageContainerFormat::Png);
        assert_eq!(raw, png);
        let budget = ImageDimensionBudget {
            min_tokens: 1,
            max_tokens: 64,
            grid_factor: 32,
        };
        let decoded = decode_encoded_image_within(fmt, &raw, budget)?;
        assert_eq!(decoded.rgb, vec![11, 22, 33]);

        // Craft a 300x1 IHDR (> 200:1 aspect ratio) followed by corrupt IDAT bytes;
        // early IHDR validation must fail before inspecting IDAT/zlib payloads.
        let mut wide_corrupt = vec![137, 80, 78, 71, 13, 10, 26, 10];
        let mut ihdr = Vec::with_capacity(13);
        ihdr.extend_from_slice(&300_u32.to_be_bytes());
        ihdr.extend_from_slice(&1_u32.to_be_bytes());
        ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
        write_chunk(&mut wide_corrupt, *b"IHDR", &ihdr)?;
        wide_corrupt.extend_from_slice(&999_u32.to_be_bytes()); // truncated/corrupt IDAT
        assert!(
            decode_encoded_image_within(ImageContainerFormat::Png, &wide_corrupt, budget).is_err()
        );
        Ok(())
    }
}
