//! Focused owned grammar execution, partial UTF-8 and candidate-order checks.
#![forbid(unsafe_code)]

use std::sync::Arc;

use bonsai_prism_runtime::grammar::{
    Element, Grammar,
    runtime::{
        Candidate, GrammarGss, GrammarMaskCache, GrammarState, PartialUtf8, VocabularyRadixTrie,
    },
};

type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

fn grammar(source: &str) -> bonsai_prism_runtime::Result<GrammarState> {
    GrammarState::from_grammar(Grammar::parse(source)?, "root")
}

fn candidate(index: u32, points: Vec<u32>) -> Candidate {
    Candidate {
        index,
        code_points: points,
        partial_utf8: PartialUtf8::default(),
        id: 0,
    }
}

#[test]
fn candidate_rejection_keeps_native_immediate_then_recursive_order() -> TestResult {
    let grammar = grammar("root ::= \"ab\"")?;
    let candidates = [
        candidate(0, vec![97, 100, 0]),
        candidate(1, vec![120, 0]),
        candidate(2, vec![97, 98, 0]),
    ];
    let rejected = grammar.reject_candidates(0, &candidates)?;
    assert_eq!(
        rejected
            .iter()
            .map(|candidate| candidate.index)
            .collect::<Vec<_>>(),
        [1, 0]
    );
    assert_eq!(rejected[1].code_points, [97, 100, 0]);
    Ok(())
}

#[test]
fn split_utf8_tokens_and_clones_own_independent_rule_storage() -> TestResult {
    let mut state = grammar("root ::= \"é\" \"a\"")?;
    let checkpoint = state.save_stacks();
    state.accept_token(0, &[0xc3])?;
    let mut clone = state.clone();
    assert_ne!(state.rules()[0].as_ptr(), clone.rules()[0].as_ptr());
    assert!(clone.restore_stacks(&checkpoint).is_err());
    state.accept_token(0, &[0xa9])?;
    state.accept_token(0, b"a")?;
    assert!(state.stacks().iter().any(Vec::is_empty));
    clone.accept_token(0, &[0xa9])?;
    assert!(clone.accept_token(0, b"b").is_err());
    assert_eq!(clone.stacks().len(), 0);
    assert!(state.stacks().iter().any(Vec::is_empty));
    state.restore_stacks(&checkpoint)?;
    state.accept_token(0, "éa".as_bytes())?;
    assert!(state.stacks().iter().any(Vec::is_empty));
    Ok(())
}

#[test]
fn explicit_token_rules_and_partial_character_ranges_keep_distinct_semantics() -> TestResult {
    let mut tokens = grammar("root ::= <[1]> \"a\"")?;
    tokens.accept_token(1, b"ignored-piece")?;
    tokens.accept_token(0, b"a")?;
    assert!(tokens.stacks().iter().any(Vec::is_empty));
    let range = grammar("root ::= [\\u0080-\\u07ff]")?;
    let mut valid = candidate(0, vec![0]);
    valid.partial_utf8 = PartialUtf8 {
        value: 2,
        remaining: 1,
    };
    let mut overlong = valid.clone();
    overlong.index = 1;
    overlong.partial_utf8.value = 1;
    assert_eq!(
        range.reject_candidates(0, &[valid, overlong.clone()])?,
        [overlong]
    );
    Ok(())
}

#[test]
fn long_candidates_use_heap_work_and_invalid_rules_are_rejected() -> TestResult {
    let state = grammar("root ::= \"a\"*")?;
    let stack = state
        .stacks()
        .iter()
        .position(|stack| !stack.is_empty())
        .ok_or("missing recursive terminal stack")?;
    let mut points = vec![97; 1000];
    points.push(0);
    assert_eq!(state.reject_candidates(stack, &[candidate(0, points)])?, []);
    assert!(grammar("root ::= root \"a\"").is_err());
    assert!(
        GrammarState::from_rules(
            vec![vec![
                Element { kind: 2, value: 1 },
                Element { kind: 0, value: 0 },
            ]],
            0
        )
        .is_err()
    );
    assert!(
        state
            .reject_candidates(stack, &[candidate(0, vec![97])])
            .is_err()
    );
    Ok(())
}

const fn is_valid_utf8_cont(partial: PartialUtf8, byte: u8) -> bool {
    if (byte & 0xc0) != 0x80 {
        return false;
    }
    match (partial.remaining, partial.value) {
        (1, 0..=1) => false,
        (2, 0) => byte >= 0xa0,
        (2, 0x0d) => byte <= 0x9f,
        (3, 0) => byte >= 0x90,
        (3, 4) => byte <= 0x8f,
        (1..=3, _) => true,
        _ => false,
    }
}

fn oracle_decode_utf8(bytes: &[u8], mut partial: PartialUtf8) -> (Vec<u32>, PartialUtf8) {
    let mut points = Vec::new();
    let mut pos = 0;
    while pos < bytes.len() && bytes[pos] != 0 {
        let byte = bytes[pos];
        if partial.remaining > 0 {
            if is_valid_utf8_cont(partial, byte) {
                partial.value = (partial.value << 6) | u32::from(byte & 0x3f);
                partial.remaining -= 1;
                pos += 1;
                if partial.remaining == 0 {
                    if partial.value <= 0x10_ffff && !(0xd800..=0xdfff).contains(&partial.value) {
                        points.push(partial.value);
                    } else {
                        points.push(0xfffd);
                    }
                    partial.value = 0;
                }
            } else {
                points.push(0xfffd);
                partial = PartialUtf8::default();
            }
            continue;
        }
        match byte {
            0x01..=0x7f => points.push(u32::from(byte)),
            0xc2..=0xdf => {
                partial = PartialUtf8 {
                    value: u32::from(byte & 0x1f),
                    remaining: 1,
                };
            }
            0xe0..=0xef => {
                partial = PartialUtf8 {
                    value: u32::from(byte & 0x0f),
                    remaining: 2,
                };
            }
            0xf0..=0xf4 => {
                partial = PartialUtf8 {
                    value: u32::from(byte & 0x07),
                    remaining: 3,
                };
            }
            _ => {
                points.push(0xfffd);
                partial = PartialUtf8::default();
            }
        }
        pos += 1;
    }
    points.push(0);
    (points, partial)
}

fn oracle_allowed_tokens(
    state: &GrammarState,
    vocab: &[Vec<u8>],
    excluded: &[u32],
) -> Result<Vec<bool>, Box<dyn std::error::Error + Send + Sync>> {
    let candidates: Vec<Candidate> = vocab
        .iter()
        .enumerate()
        .map(|(idx, piece)| {
            let (code_points, partial_utf8) = oracle_decode_utf8(piece, state.partial_utf8());
            let id = i32::try_from(idx)?;
            Ok(Candidate {
                index: u32::try_from(idx)?,
                code_points,
                partial_utf8,
                id,
            })
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error + Send + Sync>>>()?;

    let mut allowed = vec![false; vocab.len()];
    for stack_idx in 0..state.stacks().len() {
        let mut rejected_flags = vec![false; vocab.len()];
        for rejected in state.reject_candidates(stack_idx, &candidates)? {
            rejected_flags[usize::try_from(rejected.index)?] = true;
        }
        for (idx, flag) in rejected_flags.into_iter().enumerate() {
            if !flag {
                allowed[idx] = true;
            }
        }
    }
    for &ex in excluded {
        allowed[usize::try_from(ex)?] = false;
    }
    Ok(allowed)
}

#[test]
fn gss_checkpoint_rollback_and_compact_match_grammar_state_across_speculative_branches()
-> TestResult {
    let json_grammar = Grammar::parse(
        r#"root ::= "{" key ":" value "}"
key ::= "\"é\""
value ::= [0-9]+ | "\"ok\""
"#,
    )?;
    let mut state = GrammarState::from_grammar(json_grammar.clone(), "root")?;
    let mut gss = GrammarGss::from_grammar(json_grammar, "root")?;
    assert_eq!(gss.materialize_stacks(), state.stacks());
    assert_eq!(gss.partial_utf8(), state.partial_utf8());

    let cp0 = gss.checkpoint();
    let base_nodes = gss.arena_node_count();

    for piece in [b"{".as_slice(), b"\"".as_slice(), &[0xc3]] {
        state.accept_token(0, piece)?;
        gss.accept_token(0, piece)?;
        assert_eq!(gss.materialize_stacks(), state.stacks());
        assert_eq!(gss.partial_utf8(), state.partial_utf8());
    }

    let cp1 = gss.checkpoint();
    let split_nodes = gss.arena_node_count();
    assert!(split_nodes > base_nodes);

    // Speculative rejection leaves state advanced, then O(1) rollback restores cp1.
    let mut err_state = state.clone();
    assert!(err_state.accept_token(0, b"z").is_err());
    assert!(gss.accept_token(0, b"z").is_err());
    gss.rollback(&cp1)?;
    assert_eq!(gss.arena_node_count(), split_nodes);
    assert_eq!(gss.materialize_stacks(), state.stacks());
    assert_eq!(gss.partial_utf8(), state.partial_utf8());

    // Continue through split UTF-8 completion and completion to empty stack.
    for piece in [&[0xa9][..], b"\":", b"4", b"2}"] {
        state.accept_token(0, piece)?;
        gss.accept_token(0, piece)?;
        assert_eq!(gss.materialize_stacks(), state.stacks());
        assert_eq!(gss.partial_utf8(), state.partial_utf8());
    }

    // Clone cannot share checkpoints; compaction reclaims unreachable nodes and bumps epoch.
    let mut clone = gss.clone();
    assert!(clone.rollback(&cp0).is_err());
    gss.rollback(&cp0)?;
    assert_eq!(gss.arena_node_count(), base_nodes);
    gss.accept_token(0, b"{\"")?;
    gss.compact()?;
    assert!(gss.rollback(&cp0).is_err());

    // Nested nullable repetition also matches GrammarState step-for-step.
    let nested = Grammar::parse("root ::= ((\"a\")*)* \"!\"")?;
    let mut ref_nested = GrammarState::from_grammar(nested.clone(), "root")?;
    let mut gss_nested = GrammarGss::from_grammar(nested, "root")?;
    assert_eq!(gss_nested.materialize_stacks(), ref_nested.stacks());
    for piece in [b"a".as_slice(), b"a".as_slice(), b"!".as_slice()] {
        ref_nested.accept_token(0, piece)?;
        gss_nested.accept_token(0, piece)?;
        assert_eq!(gss_nested.materialize_stacks(), ref_nested.stacks());
    }
    Ok(())
}

fn assert_bitmask_matches_oracle(
    cache: &mut GrammarMaskCache,
    state: &GrammarState,
    vocab: &[Vec<u8>],
    excluded: &[u32],
    bitmask: &mut [u64],
    label: &str,
) -> TestResult {
    cache.fill_allowed_bitmask(state, bitmask)?;
    let oracle = oracle_allowed_tokens(state, vocab, excluded)?;
    for (idx, expected) in oracle.iter().enumerate() {
        assert_eq!(
            GrammarMaskCache::is_token_allowed(bitmask, u32::try_from(idx)?),
            *expected,
            "{label} mismatch at token {idx}"
        );
    }
    Ok(())
}

fn test_vocab() -> Vec<Vec<u8>> {
    vec![
        b"".to_vec(),             // 0: empty piece
        b"{".to_vec(),            // 1
        b"}".to_vec(),            // 2
        b":".to_vec(),            // 3
        b",".to_vec(),            // 4
        b"\"".to_vec(),           // 5
        b"a".to_vec(),            // 6
        b"ab".to_vec(),           // 7
        b"az".to_vec(),           // 8
        b"\"ok\"".to_vec(),       // 9
        vec![0xc3],               // 10: split lead of 'é'
        vec![0xa9],               // 11: continuation of 'é'
        b"x".to_vec(),            // 12
        b"xy".to_vec(),           // 13
        vec![b'x', 0xc3],         // 14: valid prefix + split UTF-8 tail
        vec![0x80],               // 15: invalid UTF-8 lead (0xfffd)
        b"a\x00ignored".to_vec(), // 16: NUL-truncated -> "a"
        b"<pad>".to_vec(),        // 17: excluded control token
    ]
}

#[test]
fn mask_cache_and_radix_trie_match_reject_candidates_across_generation_trace() -> TestResult {
    let vocab = test_vocab();
    let excluded = [17_u32];
    let trie = Arc::new(VocabularyRadixTrie::build(&vocab, &excluded)?);
    assert_eq!(trie.vocab_size(), vocab.len());
    assert_eq!(trie.token_piece(16), Some(b"a".as_slice()));
    assert!(trie.is_excluded(17));

    // Trace 1: Multi-step JSON + split UTF-8 generation trace up to grammar completion.
    let json_grammar = Grammar::parse(
        r#"root ::= "{" key ":" val "}"
key ::= "\"é\""
val ::= "ab" | "\"ok\""
"#,
    )?;
    let mut state = GrammarState::from_grammar(json_grammar, "root")?;
    let mut gss = GrammarGss::from_state(&state)?;
    let mut cache = GrammarMaskCache::from_state(&state, Arc::clone(&trie))?;
    let mut bitmask = vec![0_u64; trie.required_bitmask_words()];
    let mut gss_bitmask = vec![0_u64; trie.required_bitmask_words()];

    let trace: &[(i32, &[u8])] = &[
        (1, b"{"),
        (5, b"\""),
        (10, &[0xc3]),
        (11, &[0xa9]),
        (5, b"\""),
        (3, b":"),
        (9, b"\"ok\""),
        (2, b"}"),
    ];
    for &(tok_id, piece) in trace {
        assert_bitmask_matches_oracle(
            &mut cache,
            &state,
            &vocab,
            &excluded,
            &mut bitmask,
            &format!("trace step {piece:?}"),
        )?;
        cache.fill_allowed_bitmask_gss(&gss, &mut gss_bitmask)?;
        assert_eq!(bitmask, gss_bitmask);
        state.accept_token(tok_id, piece)?;
        gss.accept_token(tok_id, piece)?;
    }
    assert_bitmask_matches_oracle(
        &mut cache,
        &state,
        &vocab,
        &excluded,
        &mut bitmask,
        "completed state",
    )?;
    assert!(cache.cached_position_count() > 0);

    // Logits application sets disallowed logits to -inf and preserves allowed logits.
    let mut logits = vec![1.5_f32; vocab.len()];
    GrammarMaskCache::apply_bitmask_to_logits(&mut logits, &bitmask)?;
    assert_eq!(logits[0], 1.5);
    assert_eq!(logits[1], f32::NEG_INFINITY);

    // Trace 2: Escaped nonterminal boundary: `item` returns after "a", caller requires "z".
    let context_grammar = Grammar::parse(
        r#"root ::= item "z" | "y"
item ::= "a" | "ax"
"#,
    )?;
    let ctx_state = GrammarState::from_grammar(context_grammar, "root")?;
    let mut ctx_cache = GrammarMaskCache::from_state(&ctx_state, Arc::clone(&trie))?;
    assert_bitmask_matches_oracle(
        &mut ctx_cache,
        &ctx_state,
        &vocab,
        &excluded,
        &mut bitmask,
        "context boundary",
    )?;
    assert!(!GrammarMaskCache::is_token_allowed(&bitmask, 7)); // "ab" rejected
    assert!(GrammarMaskCache::is_token_allowed(&bitmask, 8)); // "az" allowed
    Ok(())
}

#[test]
fn mask_cache_matches_reject_candidates_for_explicit_token_and_token_not_rules() -> TestResult {
    let vocab = test_vocab();
    let excluded = [17_u32];
    let trie = Arc::new(VocabularyRadixTrie::build(&vocab, &excluded)?);
    let mut bitmask = vec![0_u64; trie.required_bitmask_words()];

    let token_rules = vec![vec![
        Element {
            kind: 3,
            value: u32::from(b'x'),
        },
        Element { kind: 8, value: 13 }, // <[13]> ("xy") after 'x' (or just "x")
        Element { kind: 0, value: 0 },
    ]];
    let tok_state = GrammarState::from_rules(token_rules, 0)?;
    let mut tok_cache = GrammarMaskCache::from_state(&tok_state, Arc::clone(&trie))?;
    assert_bitmask_matches_oracle(
        &mut tok_cache,
        &tok_state,
        &vocab,
        &excluded,
        &mut bitmask,
        "TOKEN rule",
    )?;

    let token_not_rules = vec![vec![
        Element { kind: 9, value: 13 }, // !<[13]>
        Element { kind: 0, value: 0 },
    ]];
    let not_state = GrammarState::from_rules(token_not_rules, 0)?;
    let mut not_cache = GrammarMaskCache::from_state(&not_state, trie)?;
    assert_bitmask_matches_oracle(
        &mut not_cache,
        &not_state,
        &vocab,
        &excluded,
        &mut bitmask,
        "TOKEN_NOT rule",
    )?;
    Ok(())
}
