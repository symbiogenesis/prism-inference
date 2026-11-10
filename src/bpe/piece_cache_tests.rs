//! Independent ranked-merge and bounded request-cache controls.

use super::*;

fn fixture() -> Result<Bpe> {
    let mut bytes = [None; 256];
    for (index, token) in bytes.iter_mut().enumerate() {
        *token = Some(u32::try_from(index)?);
    }
    let mut merges = HashMap::new();
    for (rank, left, right, token) in [
        (0, 98, 99, 256),
        (1, 97, 97, 257),
        (2, 97, 98, 258),
        (3, 257, 97, 259),
        (4, 258, 99, 260),
    ] {
        merges.insert((left, right), Merge { rank, token });
    }
    Ok(Bpe { bytes, merges })
}

// Deliberately use a complete forward scan and Vec removal after every merge,
// independently of production's linked nodes, heap, stale checks and cache.
fn oracle(bpe: &Bpe, bytes: &[u8]) -> Result<Vec<u32>> {
    let mut tokens = bytes
        .iter()
        .map(|&byte| bpe.bytes[usize::from(byte)].ok_or_else(|| invalid("missing fixture byte")))
        .collect::<Result<Vec<_>>>()?;
    loop {
        let next = tokens
            .windows(2)
            .enumerate()
            .filter_map(|(index, pair)| {
                bpe.merges
                    .get(&(pair[0], pair[1]))
                    .map(|merge| (merge.rank, index, merge.token))
            })
            .min_by_key(|&(rank, index, _)| (rank, index));
        let Some((_, index, token)) = next else {
            return Ok(tokens);
        };
        tokens[index] = token;
        tokens.remove(index + 1);
    }
}

#[test]
fn piece_cache_matches_independent_ranked_oracle_and_skips_repeated_merges() -> Result<()> {
    let bpe = fixture()?;
    let mut scratch = BpeScratch::default();
    for length in 1..=7 {
        for mut code in 0..3_usize.pow(u32::try_from(length)?) {
            let mut piece = vec![0; length];
            for byte in &mut piece {
                *byte = b'a' + u8::try_from(code % 3)?;
                code /= 3;
            }
            let expected = oracle(&bpe, &piece)?;
            let mut output = Vec::new();
            bpe.encode_piece_into(&piece, expected.len(), &mut output, &mut scratch)?;
            assert_eq!(output, expected);
            let uncached = scratch.uncached_pieces;
            bpe.encode_piece_into(&piece, expected.len(), &mut output, &mut scratch)?;
            assert_eq!(scratch.uncached_pieces, uncached);
            assert_eq!(&output[expected.len()..], expected);
        }
    }
    Ok(())
}

#[test]
fn piece_cache_collisions_owners_unicode_and_long_bypass_preserve_identity() -> Result<()> {
    let original = fixture()?;
    let mut changed = fixture()?;
    changed.merges.insert(
        (97, 97),
        Merge {
            rank: 1,
            token: 999,
        },
    );
    let mut scratch = BpeScratch::default();
    assert_eq!(piece_slot(b"aaa"), piece_slot(b"qqq"));
    for (bpe, piece) in [
        (&original, b"aaa".as_slice()),
        (&original, b"qqq".as_slice()),
        (&original, b"aaa".as_slice()),
        (&changed, b"aaa".as_slice()),
        (&original, b"aaa".as_slice()),
    ] {
        let before = scratch.uncached_pieces;
        let mut output = Vec::new();
        bpe.encode_piece_into(piece, piece.len(), &mut output, &mut scratch)?;
        assert_eq!(output, oracle(bpe, piece)?);
        assert_eq!(scratch.uncached_pieces, before + 1);
    }
    for piece in [
        "🌱 café".as_bytes(),
        &[0, 255, 128, 97, 97],
        &[b'a'; PIECE_CACHE_BYTES],
    ] {
        let mut output = Vec::new();
        original.encode_piece_into(piece, piece.len(), &mut output, &mut scratch)?;
        let before = scratch.uncached_pieces;
        let expected = oracle(&original, piece)?;
        assert_eq!(output, expected);
        output.clear();
        original.encode_piece_into(piece, expected.len(), &mut output, &mut scratch)?;
        assert_eq!(output, expected);
        assert_eq!(scratch.uncached_pieces, before);
    }
    let long = [b'a'; PIECE_CACHE_BYTES + 1];
    for _ in 0..2 {
        let before = scratch.uncached_pieces;
        let mut output = Vec::new();
        original.encode_piece_into(&long, long.len(), &mut output, &mut scratch)?;
        assert_eq!(output, oracle(&original, &long)?);
        assert_eq!(scratch.uncached_pieces, before + 1);
    }
    Ok(())
}

#[test]
fn piece_cache_failures_preserve_output_bounds_and_missing_byte_precedence() -> Result<()> {
    let bpe = fixture()?;
    let mut missing = fixture()?;
    missing.bytes[usize::from(b'x')] = None;
    let mut scratch = BpeScratch::default();
    let mut output = vec![42];
    let expected = oracle(&bpe, b"aaab")?;
    for _ in 0..2 {
        let before = scratch.uncached_pieces;
        assert!(
            bpe.encode_piece_into(b"aaab", expected.len() - 1, &mut output, &mut scratch)
                .is_err_and(|error| error.is::<crate::TokenLimitExceeded>())
        );
        assert_eq!(output, [42]);
        assert_eq!(scratch.uncached_pieces, before + 1);
    }
    bpe.encode_piece_into(b"aaab", expected.len(), &mut output, &mut scratch)?;
    let retained = output.clone();
    let before = scratch.uncached_pieces;
    for limit in [0, expected.len() - 1] {
        assert!(
            bpe.encode_piece_into(b"aaab", limit, &mut output, &mut scratch)
                .is_err_and(|error| error.is::<crate::TokenLimitExceeded>())
        );
        assert_eq!(output, retained);
        assert_eq!(scratch.uncached_pieces, before);
    }
    bpe.encode_piece_into(b"x", 1, &mut output, &mut scratch)?;
    let retained = output.clone();
    for limit in [0, 1] {
        assert!(
            missing
                .encode_piece_into(b"x", limit, &mut output, &mut scratch)
                .is_err_and(|error| !error.is::<crate::TokenLimitExceeded>())
        );
        assert_eq!(output, retained);
    }
    assert!(
        bpe.encode_piece_into(
            &vec![b'a'; MAX_BPE_PIECE_BYTES + 1],
            0,
            &mut output,
            &mut scratch
        )
        .is_err_and(|error| !error.is::<crate::TokenLimitExceeded>())
    );
    assert_eq!(output, retained);
    bpe.encode_piece_into(b"", 0, &mut output, &mut scratch)?;
    assert_eq!(output, retained);
    Ok(())
}

#[test]
fn piece_cache_storage_and_public_piece_composition_remain_bounded() -> Result<()> {
    // The cache adds no backing allocation: sixteen inline entries plus one
    // borrowed owner. Node/heap Vecs keep their existing largest-piece bounds.
    assert_eq!(size_of::<CachedPiece>(), 160 + 2 * size_of::<usize>());
    assert_eq!(
        PIECE_CACHE_SLOTS * size_of::<CachedPiece>() + size_of::<&Bpe>(),
        16 * (160 + 2 * size_of::<usize>()) + size_of::<usize>()
    );
    let bpe = fixture()?;
    let pieces = [b"aaab".as_slice(); 128];
    let expected = oracle(&bpe, b"aaab")?.repeat(pieces.len());
    let mut scratch = BpeScratch::default();
    let mut output = Vec::new();
    for piece in pieces {
        bpe.encode_piece_into(piece, usize::MAX, &mut output, &mut scratch)?;
    }
    assert_eq!(output, expected);
    assert_eq!(scratch.uncached_pieces, 1);
    assert_eq!(bpe.encode_pieces(&pieces, expected.len())?, expected);
    assert!(
        bpe.encode_pieces(&pieces, expected.len() - 1)
            .is_err_and(|error| error.is::<crate::TokenLimitExceeded>())
    );
    assert_eq!(bpe.encode_pieces(&pieces, expected.len())?, expected);
    Ok(())
}
