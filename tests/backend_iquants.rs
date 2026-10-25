//! Route native IQ1/IQ2 observations through owned CPU graph execution.

use bonsai_prism_runtime::Result;
use bonsai_prism_runtime::backend::{Argument, CpuBackend, Observation};
use serde_json::Value;
use std::collections::BTreeSet;

fn bytes(value: &Value) -> Result<Vec<u8>> {
    value
        .as_str()
        .ok_or_else(|| std::io::Error::other("fixture hex missing"))?
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| Ok(u8::from_str_radix(std::str::from_utf8(pair)?, 16)?))
        .collect()
}

#[test]
fn cpu_graph_iq1_iq2_matmul_matches_native_fixture_results() -> Result<()> {
    let mut seen = BTreeSet::new();
    for (fixture, weight_key) in [
        (include_str!("fixtures/upstream-iq1.json"), "WeightsHex"),
        (include_str!("fixtures/upstream-iq2.json"), "OutputHex"),
    ] {
        let fixture: Value = serde_json::from_str(fixture)?;
        let cases = fixture["Cases"]
            .as_array()
            .ok_or_else(|| std::io::Error::other("fixture cases missing"))?;
        for case in cases
            .iter()
            .filter(|case| case["ActivationInputHex"].is_string())
        {
            let kind = i32::try_from(
                case["TypeId"]
                    .as_u64()
                    .ok_or_else(|| std::io::Error::other("fixture type missing"))?,
            )?;
            let count = usize::try_from(
                case["Count"]
                    .as_u64()
                    .ok_or_else(|| std::io::Error::other("fixture count missing"))?,
            )?;
            let input: Vec<_> = bytes(&case["ActivationInputHex"])?
                .as_chunks::<4>()
                .0
                .iter()
                .map(|value| f32::from_le_bytes(*value))
                .collect();
            let mut backend = CpuBackend::new();
            let weights = backend.tensor(1, kind, [count, 1, 1, 1])?;
            backend.allocate(1)?;
            backend.set(weights, 0, &bytes(&case[weight_key])?)?;
            let activation = backend.tensor(1, 0, [count, 1, 1, 1])?;
            backend.set_values(activation, &input)?;
            let Observation::Tensor(Some(output)) = backend.call(
                "ggml_mul_mat",
                &[
                    Argument::Context(1),
                    Argument::Tensor(weights),
                    Argument::Tensor(activation),
                ],
            )?
            else {
                return Err(std::io::Error::other("expected output").into());
            };
            assert!(backend.supports(output)?);
            let Observation::Graph(graph) =
                backend.call("ggml_new_graph", &[Argument::Context(1)])?
            else {
                return Err(std::io::Error::other("expected graph").into());
            };
            backend.call(
                "ggml_build_forward_expand",
                &[Argument::Graph(graph), Argument::Tensor(output)],
            )?;
            backend.compute(graph)?;
            assert_eq!(
                backend.values(output)?[0].to_le_bytes().as_slice(),
                bytes(&case["GenericDotHex"])?
            );
            seen.insert(kind);
        }
    }
    assert_eq!(seen, BTreeSet::from([16, 17, 19, 22, 29]));
    Ok(())
}

#[test]
fn iq_copy_support_distinguishes_raw_storage_from_unavailable_encoding() -> Result<()> {
    for kind in [16, 17, 18, 19, 21, 22, 29] {
        let mut backend = CpuBackend::new();
        let input = backend.tensor(1, 0, [256, 1, 1, 1])?;
        let packed = backend.tensor(1, kind, [256, 1, 1, 1])?;
        let Observation::Tensor(Some(converted)) = backend.call(
            "ggml_cpy",
            &[
                Argument::Context(1),
                Argument::Tensor(input),
                Argument::Tensor(packed),
            ],
        )?
        else {
            return Err(std::io::Error::other("expected conversion").into());
        };
        assert!(!backend.supports(converted)?);
        let Observation::Tensor(Some(copied)) = backend.call(
            "ggml_dup",
            &[Argument::Context(1), Argument::Tensor(packed)],
        )?
        else {
            return Err(std::io::Error::other("expected raw copy").into());
        };
        assert!(backend.supports(copied)?);
    }
    Ok(())
}
