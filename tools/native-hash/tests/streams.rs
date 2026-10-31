//! Independent SHA-256 vectors and fail-closed stream/checkpoint controls.
#![forbid(unsafe_code)]

use std::io::{self, Cursor, Read};

use bonsai_native_hash::hash_reader;
use sha2::{Digest, Sha256};

fn hex(digest: [u8; 32]) -> String {
    digest
        .into_iter()
        .flat_map(|byte| [byte >> 4, byte & 15])
        .map(|nibble| char::from(b"0123456789abcdef"[usize::from(nibble)]))
        .collect()
}

#[test]
fn empty_short_multiblock_and_million_byte_vectors_match() -> io::Result<()> {
    for (input, expected) in [
        (
            b"".as_slice(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        ),
        (
            b"abc".as_slice(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        ),
        (
            b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq".as_slice(),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
        ),
    ] {
        for size in [1, 7, 64, 65_536] {
            let digest = hash_reader(
                &mut Cursor::new(input),
                u64::try_from(input.len()).map_err(io::Error::other)?,
                &mut vec![0; size],
                &mut || Ok(()),
            )?;
            assert_eq!(hex(digest), expected);
        }
    }
    let input = vec![b'a'; 1_000_000];
    assert_eq!(
        hex(hash_reader(
            &mut Cursor::new(&input),
            1_000_000,
            &mut [0; 4093],
            &mut || Ok(()),
        )?),
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
    );
    Ok(())
}

#[test]
fn binary_blocks_match_independent_rust_for_every_boundary() -> io::Result<()> {
    let input: Vec<u8> = (0..=255).cycle().take(131_073).collect();
    let expected: [u8; 32] = Sha256::digest(&input).into();
    for size in [1, 63, 64, 65, 65_535, 65_536, 65_537] {
        assert_eq!(
            hash_reader(
                &mut Cursor::new(&input),
                131_073,
                &mut vec![0; size],
                &mut || Ok(()),
            )?,
            expected
        );
    }
    Ok(())
}

struct ScriptedReader {
    source: Cursor<&'static [u8]>,
    calls: usize,
    failure: io::ErrorKind,
}

impl Read for ScriptedReader {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        self.calls += 1;
        if self.calls == 2 {
            Err(io::Error::from(self.failure))
        } else {
            self.source.read(output)
        }
    }
}

#[test]
fn read_failure_and_cancellation_never_publish_a_partial_hash() {
    let mut failing = ScriptedReader {
        source: Cursor::new(b"abcdef"),
        calls: 0,
        failure: io::ErrorKind::PermissionDenied,
    };
    let error = hash_reader(&mut failing, 6, &mut [0; 2], &mut || Ok(())).err();
    assert_eq!(
        error.map(|error| error.kind()),
        Some(io::ErrorKind::PermissionDenied)
    );
    assert_eq!(failing.source.position(), 2);

    // Cancel at startup, after partial input, and immediately before EOF.
    for cancellation in [1, 2, 4] {
        let mut checkpoints = 0;
        let mut source = Cursor::new(b"abcdef");
        let error = hash_reader(&mut source, 6, &mut [0; 2], &mut || {
            checkpoints += 1;
            if checkpoints == cancellation {
                Err(io::Error::from(io::ErrorKind::TimedOut))
            } else {
                Ok(())
            }
        })
        .err();
        assert_eq!(
            error.map(|error| error.kind()),
            Some(io::ErrorKind::TimedOut)
        );
        assert_eq!(checkpoints, cancellation);
    }
}

#[test]
fn interrupted_read_retries_with_a_new_cancellation_checkpoint() -> io::Result<()> {
    let mut source = ScriptedReader {
        source: Cursor::new(b"abcdef"),
        calls: 0,
        failure: io::ErrorKind::Interrupted,
    };
    let mut checkpoints = 0;
    let digest = hash_reader(&mut source, 6, &mut [0; 2], &mut || {
        checkpoints += 1;
        Ok(())
    })?;
    assert_eq!(digest, <[u8; 32]>::from(Sha256::digest(b"abcdef")));
    assert_eq!(source.calls, 5);
    assert_eq!(checkpoints, 5);
    Ok(())
}

#[test]
fn exact_length_empty_buffer_and_bounded_range_are_checked() -> io::Result<()> {
    for (expected, kind) in [
        (2, io::ErrorKind::InvalidData),
        (4, io::ErrorKind::UnexpectedEof),
    ] {
        let error = hash_reader(&mut Cursor::new(b"abc"), expected, &mut [0; 2], &mut || {
            Ok(())
        })
        .err();
        assert_eq!(error.map(|error| error.kind()), Some(kind));
    }
    let mut source = Cursor::new(b"abcZ");
    let error = hash_reader(&mut source, 4, &mut [], &mut || Ok(())).err();
    assert_eq!(
        error.map(|error| error.kind()),
        Some(io::ErrorKind::InvalidInput)
    );
    assert_eq!(source.position(), 0);
    assert_eq!(
        hash_reader(
            &mut Read::by_ref(&mut source).take(3),
            3,
            &mut [0; 2],
            &mut || Ok(())
        )?,
        <[u8; 32]>::from(Sha256::digest(b"abc"))
    );
    assert_eq!(source.position(), 3);
    Ok(())
}
