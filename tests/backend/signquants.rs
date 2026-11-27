//! Check `Q1_0`/`Q2_0` storage and packed arithmetic through the production graph API.

use bonsai_prism_runtime::Result;
use bonsai_prism_runtime::backend::{Argument, CpuBackend, Observation};

#[test]
fn graph_matmul_uses_q1_and_q2_packed_rows() -> Result<()> {
    for (kind, block) in [(41, 128), (42, 64)] {
        let mut backend = CpuBackend::new();
        let weights = backend.tensor(1, kind, [256, 2, 1, 1])?;
        let input = backend.tensor(1, 0, [256, 1, 1, 1])?;
        let mut values = vec![1.0; 256];
        values.extend_from_slice(&[-1.0; 256]);
        backend.set_values(weights, &values)?;
        backend.set_values(input, &[1.0; 256])?;
        assert_eq!(backend.info(weights)?.bytes()?, 512 / block * 18);
        assert_eq!(backend.values(weights)?, values);
        let Observation::Tensor(Some(product)) = backend.call(
            "ggml_mul_mat",
            &[
                Argument::Context(1),
                Argument::Tensor(weights),
                Argument::Tensor(input),
            ],
        )?
        else {
            return Err(std::io::Error::other("expected tensor").into());
        };
        assert!(backend.supports(product)?);
        let Observation::Graph(graph) = backend.call("ggml_new_graph", &[Argument::Context(1)])?
        else {
            return Err(std::io::Error::other("expected graph").into());
        };
        backend.call(
            "ggml_build_forward_expand",
            &[Argument::Graph(graph), Argument::Tensor(product)],
        )?;
        backend.compute(graph)?;
        for (value, expected) in backend.values(product)?.iter().zip([256.0, -256.0]) {
            assert!((value - expected).abs() < 0.1);
        }
    }
    Ok(())
}
