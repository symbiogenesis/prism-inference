#![forbid(unsafe_code)]
//! Focused strict raw-token JSON and bounded-stream checks.

use std::io::{self, Cursor, Read};

use bonsai_prism_runtime::Result;
use bonsai_prism_runtime::token_input::{MAX_TOKEN_INPUT_BYTES, read_tokens};
use bonsai_prism_runtime::tokenization::MAX_TOKENIZATION_CONTEXT;

#[test]
fn exact_ids_empty_arrays_and_json_whitespace_are_preserved() -> Result<()> {
    assert_eq!(
        read_tokens(&mut &b" \r\n\t[0, 4294967295,\r\n258,258,0]\t"[..], 5)?,
        [0, u32::MAX, 258, 258, 0]
    );
    for input in ["[]", " \t[ \r\n ]\n"] {
        assert_eq!(read_tokens(&mut input.as_bytes(), 1)?, Vec::<u32>::new());
    }
    let ids: Vec<_> = (0..MAX_TOKENIZATION_CONTEXT)
        .map(|index| u32::try_from(index % 3))
        .collect::<std::result::Result<_, _>>()?;
    let serialized = serde_json::to_vec(&ids)?;
    assert_eq!(
        read_tokens(&mut serialized.as_slice(), MAX_TOKENIZATION_CONTEXT)?,
        ids
    );
    assert!(read_tokens(&mut serialized.as_slice(), MAX_TOKENIZATION_CONTEXT - 1).is_err());
    Ok(())
}

#[test]
fn strict_json_u32_arrays_reject_malformed_or_other_json_without_private_data() {
    for input in [
        "",
        " ",
        "null",
        "0",
        "{}",
        "[",
        "]",
        "[1",
        "[1,]",
        "[,1]",
        "[1,,2]",
        "[1 2]",
        "[00]",
        "[01]",
        "[-0]",
        "[-1]",
        "[+1]",
        "[1.0]",
        "[1e0]",
        "[4294967296]",
        "[999999999999999999999999999999999999]",
        "[true]",
        "[null]",
        "[[1]]",
        "[\"private token sentinel\"]",
        "[1] [2]",
        "[1]private token sentinel",
        "[1]// private token sentinel",
        "\u{feff}[1]",
        "[1\u{b}]",
        "[1\u{c}]",
        "[1\u{a0}]",
        "[１２]",
    ] {
        let result = read_tokens(&mut input.as_bytes(), 8);
        assert!(result.is_err(), "accepted malformed token input");
        if let Err(error) = result {
            assert!(!error.to_string().contains("private"));
            assert!(!format!("{error:?}").contains("private"));
            assert!(error.source().is_none());
        }
    }
    assert!(read_tokens(&mut &b"[1,\xff]"[..], 8).is_err());
}

#[test]
fn exact_byte_limit_is_admitted_and_overflow_stops_at_one_extra_byte() -> Result<()> {
    let mut serialized = b"[4294967295,0]".to_vec();
    serialized.resize(MAX_TOKEN_INPUT_BYTES, b' ');
    assert_eq!(read_tokens(&mut serialized.as_slice(), 2)?, [u32::MAX, 0]);
    serialized.extend_from_slice(b" private token suffix");
    let mut reader = Cursor::new(serialized.as_slice());
    assert!(read_tokens(&mut reader, 2).is_err());
    assert_eq!(reader.position(), u64::try_from(MAX_TOKEN_INPUT_BYTES + 1)?);
    let mut remainder = Vec::new();
    reader.read_to_end(&mut remainder)?;
    assert_eq!(remainder, b"private token suffix");
    Ok(())
}

struct RepeatingReader {
    consumed: usize,
    maximum_chunk: usize,
}

impl Read for RepeatingReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        assert_ne!(buffer, [0_u8; 0]);
        let count = buffer.len().min(self.maximum_chunk);
        buffer[..count].fill(b' ');
        self.consumed += count;
        Ok(count)
    }
}

#[test]
fn infinite_streams_are_bounded_and_invalid_limits_reject_before_reading() {
    for maximum_chunk in [1, 3, 8192, usize::MAX] {
        let mut reader = RepeatingReader {
            consumed: 0,
            maximum_chunk,
        };
        assert!(read_tokens(&mut reader, 2).is_err());
        assert_eq!(reader.consumed, MAX_TOKEN_INPUT_BYTES + 1);
    }
    for limit in [0, MAX_TOKENIZATION_CONTEXT + 1, usize::MAX] {
        let mut reader = RepeatingReader {
            consumed: 0,
            maximum_chunk: 1,
        };
        assert!(read_tokens(&mut reader, limit).is_err());
        assert_eq!(reader.consumed, 0);
    }
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
fn short_reads_and_trait_object_readers_preserve_split_numbers() -> Result<()> {
    let text = b"\r\n[4294967295,0,258,258]\t";
    for maximum_chunk in 1..=text.len() {
        let mut reader = ShortReader {
            bytes: text,
            maximum_chunk,
        };
        let erased: &mut dyn Read = &mut reader;
        assert_eq!(read_tokens(erased, 4)?, [u32::MAX, 0, 258, 258]);
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
        Err(io::Error::new(self.kind, "private token reader message"))
    }
}

#[test]
fn io_errors_and_interruptions_return_once_without_partial_ids_or_message_leaks() {
    for kind in [
        io::ErrorKind::Interrupted,
        io::ErrorKind::WouldBlock,
        io::ErrorKind::PermissionDenied,
        io::ErrorKind::UnexpectedEof,
        io::ErrorKind::Other,
    ] {
        for prefix in [&b""[..], &b"[1,2]"[..], &b"[429496"[..]] {
            let mut reader = FailingReader {
                prefix,
                kind,
                failures: 0,
            };
            let result = read_tokens(&mut reader, 8);
            assert!(result.is_err());
            if let Err(error) = result {
                assert!(!error.to_string().contains("private"));
                assert!(!format!("{error:?}").contains("private"));
                assert!(error.source().is_none());
                assert_eq!(
                    error.downcast_ref::<io::Error>().map(io::Error::kind),
                    Some(kind)
                );
            }
            assert_eq!(reader.failures, 1);
        }
    }
}
