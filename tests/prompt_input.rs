#![forbid(unsafe_code)]
//! Focused exact prompt input, resource bounds and failure regressions.

use std::io::{self, Cursor, Read};

use bonsai_prism_runtime::{Result, bpe::MAX_BPE_INPUT_BYTES, prompt_input::read_prompt};

#[test]
fn exact_utf8_preserves_bom_newlines_nul_and_literal_control_spellings() -> Result<()> {
    for text in [
        "a",
        "\u{feff}🌱 café\r\n中文\n\0",
        " \t\r\n ",
        "<|im_start|>user\nplain text<|im_end|>\n",
        "a\u{0301} é",
    ] {
        let mut reader = Cursor::new(text.as_bytes());
        let output = read_prompt(&mut reader)?;
        assert_eq!(output.as_bytes(), text.as_bytes());
        assert_eq!(reader.position(), u64::try_from(text.len())?);
    }
    Ok(())
}

#[test]
fn exact_byte_limit_accepts_ascii_and_multibyte_prompts() -> Result<()> {
    for text in [
        "a".repeat(MAX_BPE_INPUT_BYTES),
        "🌱".repeat(MAX_BPE_INPUT_BYTES / "🌱".len()),
        format!("{}€", "a".repeat(MAX_BPE_INPUT_BYTES - "€".len())),
    ] {
        assert_eq!(text.len(), MAX_BPE_INPUT_BYTES);
        assert_eq!(read_prompt(&mut text.as_bytes())?, text);
    }
    Ok(())
}

#[test]
fn empty_malformed_and_truncated_utf8_reject_without_exposing_prompt_bytes() {
    for bytes in [
        &b""[..],
        &[0xff][..],
        &[0xe2, 0x82][..],
        &b"private prompt\xff"[..],
        &b"private prompt\xc0\xaf"[..],
        &b"private prompt\xed\xa0\x80"[..],
        &b"private prompt\xf4\x90\x80\x80"[..],
        &b"private prompt\xf0\x9f\x8c"[..],
    ] {
        let result = read_prompt(&mut &bytes[..]);
        assert!(result.is_err());
        if let Err(error) = result {
            assert!(!error.to_string().contains("private"));
            assert!(!format!("{error:?}").contains("private"));
            assert!(error.source().is_none());
        }
    }
}

struct RepeatingReader {
    consumed: usize,
    maximum_chunk: usize,
}

impl Read for RepeatingReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        assert_ne!(buffer, [0_u8; 0]);
        let count = buffer.len().min(self.maximum_chunk);
        buffer[..count].fill(b'a');
        self.consumed += count;
        Ok(count)
    }
}

#[test]
fn infinite_oversized_input_stops_after_exactly_one_byte_beyond_limit() {
    for maximum_chunk in [1, 3, 8192, usize::MAX] {
        let mut reader = RepeatingReader {
            consumed: 0,
            maximum_chunk,
        };
        assert!(read_prompt(&mut reader).is_err());
        assert_eq!(reader.consumed, MAX_BPE_INPUT_BYTES + 1);
    }
}

#[test]
fn finite_oversized_input_leaves_suffix_unread_even_when_utf8_crosses_limit() -> Result<()> {
    let text = format!("{}🌱private suffix", "a".repeat(MAX_BPE_INPUT_BYTES - 1));
    let mut reader = Cursor::new(text.as_bytes());
    assert!(read_prompt(&mut reader).is_err());
    assert_eq!(reader.position(), u64::try_from(MAX_BPE_INPUT_BYTES + 1)?);
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    assert_eq!(bytes, &text.as_bytes()[MAX_BPE_INPUT_BYTES + 1..]);
    Ok(())
}

struct ShortReader<'a> {
    bytes: &'a [u8],
    maximum_chunk: usize,
}

impl Read for ShortReader<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let count = buffer.len().min(self.maximum_chunk).min(self.bytes.len());
        buffer[..count].copy_from_slice(&self.bytes[..count]);
        self.bytes = &self.bytes[count..];
        Ok(count)
    }
}

#[test]
fn short_reads_and_trait_object_readers_preserve_split_utf8() -> Result<()> {
    let text = "\u{feff}🌱 café\r\n中文\n\0";
    for maximum_chunk in 1..=text.len() {
        let mut reader = ShortReader {
            bytes: text.as_bytes(),
            maximum_chunk,
        };
        let erased: &mut dyn Read = &mut reader;
        assert_eq!(read_prompt(erased)?, text);
        assert_eq!(reader.bytes, [0_u8; 0]);
    }
    Ok(())
}

struct FailingReader<'a> {
    prefix: &'a [u8],
    kind: io::ErrorKind,
    failures: usize,
}

impl Read for FailingReader<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if !self.prefix.is_empty() {
            return self.prefix.read(buffer);
        }
        self.failures += 1;
        Err(io::Error::new(self.kind, "private prompt reader message"))
    }
}

#[test]
fn io_failures_and_interruptions_return_once_without_partial_prompt_or_message_leaks() -> Result<()>
{
    for kind in [
        io::ErrorKind::Interrupted,
        io::ErrorKind::WouldBlock,
        io::ErrorKind::PermissionDenied,
        io::ErrorKind::UnexpectedEof,
        io::ErrorKind::Other,
    ] {
        for prefix in [&b""[..], &b"private prompt"[..], &b"split\xf0\x9f"[..]] {
            let mut reader = FailingReader {
                prefix,
                kind,
                failures: 0,
            };
            let result = read_prompt(&mut reader);
            assert!(result.is_err());
            if let Err(error) = result {
                assert!(!error.to_string().contains("private"));
                assert!(!format!("{error:?}").contains("private"));
                assert!(error.source().is_none());
                let io_error = error.downcast_ref::<io::Error>();
                assert_eq!(io_error.map(io::Error::kind), Some(kind));
            }
            assert_eq!(reader.failures, 1);
            assert_eq!(read_prompt(&mut &b"next request"[..])?, "next request");
        }
    }
    Ok(())
}
