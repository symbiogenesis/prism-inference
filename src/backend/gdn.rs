// Copyright (c) 2023-2026 The ggml/llama.cpp authors
// SPDX-License-Identifier: MIT
//! Safe gated delta state updates, cache rows, raw gates and rollback records.

use super::{CpuBackend, TensorInfo, linear};
use crate::{Result, invalid};

fn integer(params: &[u8; 64], index: usize) -> i32 {
    i32::from_le_bytes(std::array::from_fn(|offset| params[index * 4 + offset]))
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(&left, &right)| f64::from(left) * f64::from(right))
        .sum::<f64>() as f32
}

impl CpuBackend {
    #[expect(
        clippy::too_many_lines,
        reason = "GDN snapshot and replay state transitions retain their pinned output layout and arithmetic order."
    )]
    pub(super) fn delta_values(&self, info: &TensorInfo, mut output: Vec<f32>) -> Result<Vec<f32>> {
        let ids = info.source_slots;
        let source = |index: usize| ids[index].ok_or_else(|| invalid("CPU GDN input is absent"));
        let input = (0..6)
            .map(|index| self.values(source(index)?))
            .collect::<Result<Vec<_>>>()?;
        let shapes = (0..6)
            .map(|index| self.info(source(index)?).map(|info| info.shape))
            .collect::<Result<Vec<_>>>()?;
        let [width, heads, tokens, sequences] = shapes[2];
        let snapshots = usize::try_from(integer(&info.parameters, 0))?;
        let replay = info.operation == 101;
        if snapshots == 0 || (replay && snapshots < 2) {
            return Err(invalid("CPU GDN snapshot count is invalid"));
        }
        if width == 0
            || heads == 0
            || sequences == 0
            || (replay && tokens == 0)
            || shapes[0][3] == 0
            || shapes[1][3] == 0
        {
            return Err(invalid("CPU GDN geometry must be positive"));
        }
        let raw = integer(&info.parameters, 1) != 0;
        let rows = if replay {
            None
        } else {
            ids[6].map(|id| self.integer_values(id)).transpose()?
        };
        let validated_rows = if let Some(rows) = &rows {
            if rows.len() != sequences {
                return Err(invalid("CPU GDN cache row count differs from sequences"));
            }
            let mut seen = std::collections::HashSet::with_capacity(sequences);
            let mut mapped = Vec::with_capacity(sequences);
            for &raw_row in rows {
                let row = usize::try_from(raw_row)?;
                if row >= shapes[5][1] {
                    return Err(invalid("CPU GDN cache row exceeds state table"));
                }
                if !seen.insert(row) {
                    return Err(invalid(
                        "CPU GDN cache rows must be distinct across sequences",
                    ));
                }
                mapped.push(row);
            }
            Some(mapped)
        } else {
            None
        };
        let bias = if raw {
            Some(self.values(source(7)?)?)
        } else {
            None
        };
        let decay_weight = if raw {
            Some(self.values(source(8)?)?)
        } else {
            None
        };
        let records = if replay {
            Some(self.values(source(6)?)?)
        } else {
            None
        };
        let control = if replay {
            Some(self.integer_values(source(7)?)?)
        } else {
            None
        };
        let record_shape = if replay {
            Some(self.info(source(6)?)?.shape)
        } else {
            None
        };
        let state_per_head = width
            .checked_mul(width)
            .ok_or_else(|| invalid("CPU GDN head state extent overflow"))?;
        let state_per_sequence = state_per_head
            .checked_mul(heads)
            .ok_or_else(|| invalid("CPU GDN sequence state extent overflow"))?;
        let all_state = state_per_sequence
            .checked_mul(sequences)
            .ok_or_else(|| invalid("CPU GDN total state extent overflow"))?;
        let attention_elements = width
            .checked_mul(heads)
            .and_then(|value| value.checked_mul(tokens))
            .and_then(|value| value.checked_mul(sequences))
            .ok_or_else(|| invalid("CPU GDN attention output extent overflow"))?;
        let record_width = width
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .and_then(|value| value.checked_mul(heads))
            .ok_or_else(|| invalid("CPU GDN record width overflow"))?;
        let expected_output = if replay {
            (snapshots - 1)
                .checked_mul(record_width)
                .and_then(|value| value.checked_mul(sequences))
                .and_then(|value| value.checked_add(all_state))
                .and_then(|value| value.checked_add(attention_elements))
                .ok_or_else(|| invalid("CPU GDN replay output extent overflow"))?
        } else {
            snapshots
                .checked_mul(all_state)
                .and_then(|value| value.checked_add(attention_elements))
                .ok_or_else(|| invalid("CPU GDN output extent overflow"))?
        };
        if output.len() != expected_output {
            return Err(invalid("CPU GDN output extent differs"));
        }
        let written_depth = tokens.min(snapshots);
        let base_token = if replay {
            tokens
                .checked_sub(written_depth)
                .ok_or_else(|| invalid("CPU GDN base token underflow"))?
        } else {
            0
        };
        let scale = 1.0 / (width as f32).sqrt();
        for sequence in 0..sequences {
            let state_row = validated_rows
                .as_ref()
                .map_or(sequence, |mapped| mapped[sequence]);
            for head in 0..heads {
                let start = state_row * state_per_sequence + head * state_per_head;
                let mut state = input[5]
                    .get(start..start + state_per_head)
                    .ok_or_else(|| invalid("CPU GDN initial state extent differs"))?
                    .to_vec();
                if let (Some(records), Some(control), Some(record_shape)) =
                    (&records, &control, record_shape)
                {
                    let raw_count = *control
                        .get(2 * sequence)
                        .ok_or_else(|| invalid("CPU GDN replay control extent differs"))?;
                    let raw_bank = *control
                        .get(2 * sequence + 1)
                        .ok_or_else(|| invalid("CPU GDN replay control extent differs"))?;
                    let count = usize::try_from(raw_count)?;
                    let bank = usize::try_from(raw_bank)?;
                    if count >= snapshots || bank > 1 {
                        return Err(invalid("CPU GDN replay count or bank differs"));
                    }
                    let record_stride_seq = record_shape[0]
                        .checked_mul(record_shape[1])
                        .ok_or_else(|| invalid("CPU GDN replay record stride overflow"))?;
                    let record_base = bank
                        .checked_mul(record_shape[0])
                        .and_then(|offset| {
                            sequence
                                .checked_mul(record_stride_seq)
                                .and_then(|seq_off| offset.checked_add(seq_off))
                        })
                        .ok_or_else(|| invalid("CPU GDN replay record base overflow"))?;
                    let record_end = count
                        .checked_mul(record_width)
                        .and_then(|span| record_base.checked_add(span))
                        .ok_or_else(|| invalid("CPU GDN replay record extent overflow"))?;
                    if record_end > records.len() {
                        return Err(invalid("CPU GDN replay records buffer too small"));
                    }
                    for record in 0..count {
                        let base = record_base + record * record_width;
                        let decay = records[base + 2 * heads * width + head];
                        for value in &mut state {
                            *value *= decay;
                        }
                        let keys = &records[base + heads * width + head * width
                            ..base + heads * width + (head + 1) * width];
                        for row in 0..width {
                            let delta = records[base + head * width + row];
                            for column in 0..width {
                                state[row * width + column] += keys[column] * delta;
                            }
                        }
                    }
                }
                for token in 0..tokens {
                    let query_start = linear(
                        [
                            0,
                            head % shapes[0][1],
                            token,
                            sequence / (sequences / shapes[0][3]),
                        ],
                        shapes[0],
                    );
                    let query = &input[0][query_start..query_start + width];
                    let key_start = linear(
                        [
                            0,
                            head % shapes[1][1],
                            token,
                            sequence / (sequences / shapes[1][3]),
                        ],
                        shapes[1],
                    );
                    let keys = &input[1][key_start..key_start + width];
                    let value_start = linear([0, head, token, sequence], shapes[2]);
                    let values = &input[2][value_start..value_start + width];
                    let mut beta = input[4][linear([0, head, token, sequence], shapes[4])];
                    let gate_start = linear([0, head, token, sequence], shapes[3]);
                    let mut gate = input[3][gate_start];
                    if let (Some(bias), Some(weight)) = (&bias, &decay_weight) {
                        beta = 1.0 / (1.0 + (-beta).exp());
                        if weight[head] == 0.0 && gate.is_finite() && bias[head].is_finite() {
                            gate = 0.0;
                        } else {
                            let value = gate + bias[head];
                            #[expect(
                                clippy::imprecise_flops,
                                reason = "The pinned raw gate helper rounds 1+exp before logf."
                            )]
                            let softplus = if value > 20.0 {
                                value
                            } else {
                                (1.0 + value.exp()).ln()
                            };
                            gate = weight[head] * softplus;
                        }
                    }
                    let decay = gate.exp();
                    for row in 0..width {
                        for column in 0..width {
                            let factor = if shapes[3][0] == width {
                                input[3][gate_start + column].exp()
                            } else {
                                decay
                            };
                            state[row * width + column] *= factor;
                        }
                    }
                    let delta = (0..width)
                        .map(|row| {
                            (values[row] - dot(&state[row * width..(row + 1) * width], keys)) * beta
                        })
                        .collect::<Vec<_>>();
                    for row in 0..width {
                        for column in 0..width {
                            state[row * width + column] += keys[column] * delta[row];
                        }
                    }
                    for row in 0..width {
                        output[linear([row, head, token, sequence], shapes[2])] =
                            dot(&state[row * width..(row + 1) * width], query) * scale;
                    }
                    if replay {
                        if token == base_token {
                            let start = attention_elements
                                + sequence * state_per_sequence
                                + head * state_per_head;
                            output[start..start + state_per_head].copy_from_slice(&state);
                        } else if token > base_token {
                            let record = token - base_token - 1;
                            let start = attention_elements
                                + all_state
                                + sequence * (snapshots - 1) * record_width
                                + record * record_width;
                            output[start + head * width..start + (head + 1) * width]
                                .copy_from_slice(&delta);
                            output[start + heads * width + head * width
                                ..start + heads * width + (head + 1) * width]
                                .copy_from_slice(keys);
                            output[start + 2 * heads * width + head] = decay;
                        }
                    } else {
                        let slot = tokens - 1 - token;
                        if slot < snapshots {
                            let start = attention_elements
                                + slot * all_state
                                + sequence * state_per_sequence
                                + head * state_per_head;
                            output[start..start + state_per_head].copy_from_slice(&state);
                        }
                    }
                }
                if !replay && snapshots == 1 {
                    let start =
                        attention_elements + sequence * state_per_sequence + head * state_per_head;
                    output[start..start + state_per_head].copy_from_slice(&state);
                }
            }
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Argument, CpuBackend, GraphId, Observation, TensorId};
    use crate::{Result, invalid};

    fn tensor(value: &Observation) -> Result<TensorId> {
        match value {
            Observation::Tensor(Some(id)) => Ok(*id),
            _ => Err(invalid("expected tensor")),
        }
    }

    fn new_graph(backend: &mut CpuBackend) -> Result<GraphId> {
        match backend.call("ggml_new_graph", &[Argument::Context(0)])? {
            Observation::Graph(graph) => Ok(graph),
            _ => Err(invalid("expected graph")),
        }
    }

    struct GdnFixture {
        backend: CpuBackend,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        g: TensorId,
        b: TensorId,
        s: TensorId,
        out: TensorId,
        graph: GraphId,
    }

    fn build_gdn(
        width: usize,
        heads: usize,
        tokens: usize,
        snapshots: usize,
    ) -> Result<GdnFixture> {
        let mut backend = CpuBackend::new();
        let query = backend.tensor(0, 0, [width, heads, tokens, 1])?;
        let key = backend.tensor(0, 0, [width, heads, tokens, 1])?;
        let value = backend.tensor(0, 0, [width, heads, tokens, 1])?;
        let gate = backend.tensor(0, 0, [1, heads, tokens, 1])?;
        let beta = backend.tensor(0, 0, [1, heads, tokens, 1])?;
        let state = backend.tensor(0, 0, [width, width, heads, 1])?;
        let out = tensor(&backend.call(
            "ggml_gated_delta_net",
            &[
                Argument::Context(0),
                Argument::Tensor(query),
                Argument::Tensor(key),
                Argument::Tensor(value),
                Argument::Tensor(gate),
                Argument::Tensor(beta),
                Argument::Tensor(state),
                Argument::Integer(i64::try_from(snapshots)?),
            ],
        )?)?;
        let graph = new_graph(&mut backend)?;
        backend.call(
            "ggml_build_forward_expand",
            &[Argument::Graph(graph), Argument::Tensor(out)],
        )?;
        Ok(GdnFixture {
            backend,
            q: query,
            k: key,
            v: value,
            g: gate,
            b: beta,
            s: state,
            out,
            graph,
        })
    }

    fn set_i32(backend: &mut CpuBackend, id: TensorId, values: &[i32]) -> Result<()> {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        backend.set(id, 0, &bytes)
    }

    #[test]
    fn repeated_identical_keys_and_nearly_collinear_keys_stay_bounded() -> Result<()> {
        let width = 4;
        let heads = 1;
        let tokens = 32;
        // 32 identical unit keys with nonzero gate & beta converge stably without blow-up.
        let mut identical = build_gdn(width, heads, tokens, 1)?;
        identical.backend.allocate(0)?;
        let mut unit_key = vec![0.0_f32; width * tokens];
        for step in 0..tokens {
            unit_key[step * width] = 1.0;
        }
        identical.backend.set_values(identical.q, &unit_key)?;
        identical.backend.set_values(identical.k, &unit_key)?;
        identical
            .backend
            .set_values(identical.v, &vec![0.75_f32; width * tokens])?;
        identical
            .backend
            .set_values(identical.g, &vec![-0.25_f32; tokens])?;
        identical
            .backend
            .set_values(identical.b, &vec![0.5_f32; tokens])?;
        identical
            .backend
            .set_values(identical.s, &vec![0.0_f32; width * width])?;
        identical.backend.compute(identical.graph)?;
        let out_identical = identical.backend.values(identical.out)?;
        assert!(
            out_identical
                .iter()
                .all(|value| value.is_finite() && value.abs() < 2.0)
        );

        // 32 nearly collinear keys (cos > 0.9999) remain finite and bounded.
        let mut collinear = build_gdn(width, heads, tokens, 1)?;
        collinear.backend.allocate(0)?;
        let mut near_keys = Vec::with_capacity(width * tokens);
        for step in 0..tokens {
            let eps = 0.005_f32 * f32::from(u8::try_from(step % 8)?);
            let norm = (1.0_f32 + eps * eps).sqrt();
            near_keys.extend_from_slice(&[1.0 / norm, eps / norm, 0.0, 0.0]);
        }
        collinear.backend.set_values(collinear.q, &near_keys)?;
        collinear.backend.set_values(collinear.k, &near_keys)?;
        collinear
            .backend
            .set_values(collinear.v, &vec![0.5_f32; width * tokens])?;
        collinear
            .backend
            .set_values(collinear.g, &vec![-0.01_f32; tokens])?;
        collinear
            .backend
            .set_values(collinear.b, &vec![0.9_f32; tokens])?;
        collinear
            .backend
            .set_values(collinear.s, &vec![0.1_f32; width * width])?;
        collinear.backend.compute(collinear.graph)?;
        let out_collinear = collinear.backend.values(collinear.out)?;
        assert!(
            out_collinear
                .iter()
                .all(|value| value.is_finite() && value.abs() < 10.0)
        );
        Ok(())
    }

    #[test]
    fn extreme_gate_erasure_and_zero_weight_raw_gate_overflow_produce_finite_state() -> Result<()> {
        // Test raw gate overflow when weight[head] == 0.0 (`gate + bias = +inf` -> finite `decay = 1.0`)
        // alongside `beta -> 1.0`.
        let width = 2;
        let mut raw_fixture = CpuBackend::new();
        let query = raw_fixture.tensor(0, 0, [width, 1, 1, 1])?;
        let key = raw_fixture.tensor(0, 0, [width, 1, 1, 1])?;
        let value = raw_fixture.tensor(0, 0, [width, 1, 1, 1])?;
        let gate = raw_fixture.tensor(0, 0, [1, 1, 1, 1])?;
        let beta = raw_fixture.tensor(0, 0, [1, 1, 1, 1])?;
        let state = raw_fixture.tensor(0, 0, [width, width, 1, 1])?;
        let bias = raw_fixture.tensor(0, 0, [1, 1, 1, 1])?;
        let weight = raw_fixture.tensor(0, 0, [1, 1, 1, 1])?;
        let out = tensor(&raw_fixture.call(
            "ggml_gated_delta_net",
            &[
                Argument::Context(0),
                Argument::Tensor(query),
                Argument::Tensor(key),
                Argument::Tensor(value),
                Argument::Tensor(gate),
                Argument::Tensor(beta),
                Argument::Tensor(state),
                Argument::Integer(1),
            ],
        )?)?;
        raw_fixture.call(
            "ggml_gated_delta_net_set_raw_gates",
            &[
                Argument::Tensor(out),
                Argument::Tensor(bias),
                Argument::Tensor(weight),
            ],
        )?;
        let graph = new_graph(&mut raw_fixture)?;
        raw_fixture.call(
            "ggml_build_forward_expand",
            &[Argument::Graph(graph), Argument::Tensor(out)],
        )?;
        raw_fixture.allocate(0)?;
        raw_fixture.set_values(query, &[1.0, 0.0])?;
        raw_fixture.set_values(key, &[0.0, 1.0])?;
        raw_fixture.set_values(value, &[2.0, -3.0])?;
        raw_fixture.set_values(gate, &[3e38_f32])?;
        raw_fixture.set_values(beta, &[40.0_f32])?;
        raw_fixture.set_values(state, &[0.5, 0.0, 0.0, 0.5])?;
        raw_fixture.set_values(bias, &[3e38_f32])?;
        raw_fixture.set_values(weight, &[0.0_f32])?;
        raw_fixture.compute(graph)?;
        let values = raw_fixture.values(out)?;
        assert!(values.iter().all(|value| value.is_finite()));
        // decay = 1.0 preserves state[0,0] = 0.5 and replaces row 1 col 1 (`k = [0,1]`) with `v[1] = -3`.
        assert!((values[2] - 0.5).abs() <= 1e-6);
        assert!((values[5] - (-3.0)).abs() <= 1e-5);

        // Extreme erasure (`decay -> 0`, `beta -> 1`) wipes prior orthogonal state component cleanly.
        let mut erase = build_gdn(width, 1, 1, 1)?;
        erase.backend.allocate(0)?;
        erase.backend.set_values(erase.q, &[1.0, 0.0])?;
        erase.backend.set_values(erase.k, &[0.0, 1.0])?;
        erase.backend.set_values(erase.v, &[0.0, 1.0])?;
        erase.backend.set_values(erase.g, &[-100.0])?;
        erase.backend.set_values(erase.b, &[1.0])?;
        erase
            .backend
            .set_values(erase.s, &[10.0, 20.0, 30.0, 40.0])?;
        erase.backend.compute(erase.graph)?;
        let erased = erase.backend.values(erase.out)?;
        assert!(erased[2].abs() <= 1e-20);
        assert!((erased[5] - 1.0).abs() <= 1e-6);
        Ok(())
    }

    #[test]
    fn chunked_256_step_trajectory_matches_single_shot() -> Result<()> {
        let width = 4;
        let heads = 2;
        let total_tokens = 256;
        let half = 128;
        let elems = width * heads * total_tokens;
        let scalar_elems = heads * total_tokens;
        let mut q_vals = Vec::with_capacity(elems);
        let mut k_vals = Vec::with_capacity(elems);
        let mut v_vals = Vec::with_capacity(elems);
        let mut g_vals = Vec::with_capacity(scalar_elems);
        let mut b_vals = Vec::with_capacity(scalar_elems);
        for step in 0..total_tokens {
            for head in 0..heads {
                let phase = f32::from(u16::try_from((step * 7 + head * 13) % 29)?) * 0.05;
                g_vals.push(-0.02 - 0.01 * phase);
                b_vals.push(0.25 + 0.1 * phase);
                let base_k = [phase.cos(), phase.sin(), 0.5 * phase, -0.25];
                let norm = base_k.iter().map(|x| x * x).sum::<f32>().sqrt();
                for lane in 0..width {
                    q_vals.push(base_k[(lane + 1) % width] / norm);
                    k_vals.push(base_k[lane] / norm);
                    v_vals.push(0.1 * f32::from(u8::try_from((step + lane + head) % 11)?));
                }
            }
        }
        let init_state: Vec<f32> = (0..width * width * heads)
            .map(|idx| Ok(0.05 * f32::from(u8::try_from(idx % 7)?)))
            .collect::<Result<Vec<f32>>>()?;

        let mut full = build_gdn(width, heads, total_tokens, 1)?;
        full.backend.allocate(0)?;
        full.backend.set_values(full.q, &q_vals)?;
        full.backend.set_values(full.k, &k_vals)?;
        full.backend.set_values(full.v, &v_vals)?;
        full.backend.set_values(full.g, &g_vals)?;
        full.backend.set_values(full.b, &b_vals)?;
        full.backend.set_values(full.s, &init_state)?;
        full.backend.compute(full.graph)?;
        let full_values = full.backend.values(full.out)?;

        let mut first = build_gdn(width, heads, half, 1)?;
        first.backend.allocate(0)?;
        first.backend.set_values(first.q, &q_vals[..elems / 2])?;
        first.backend.set_values(first.k, &k_vals[..elems / 2])?;
        first.backend.set_values(first.v, &v_vals[..elems / 2])?;
        first
            .backend
            .set_values(first.g, &g_vals[..scalar_elems / 2])?;
        first
            .backend
            .set_values(first.b, &b_vals[..scalar_elems / 2])?;
        first.backend.set_values(first.s, &init_state)?;
        first.backend.compute(first.graph)?;
        let first_values = first.backend.values(first.out)?;
        let mid_state = &first_values[elems / 2..];

        let mut second = build_gdn(width, heads, half, 1)?;
        second.backend.allocate(0)?;
        second.backend.set_values(second.q, &q_vals[elems / 2..])?;
        second.backend.set_values(second.k, &k_vals[elems / 2..])?;
        second.backend.set_values(second.v, &v_vals[elems / 2..])?;
        second
            .backend
            .set_values(second.g, &g_vals[scalar_elems / 2..])?;
        second
            .backend
            .set_values(second.b, &b_vals[scalar_elems / 2..])?;
        second.backend.set_values(second.s, mid_state)?;
        second.backend.compute(second.graph)?;
        let second_values = second.backend.values(second.out)?;

        let combined: Vec<f32> = first_values[..elems / 2]
            .iter()
            .copied()
            .chain(second_values)
            .collect();
        assert_eq!(combined.len(), full_values.len());
        for (&actual, &expected) in combined.iter().zip(&full_values) {
            assert!((actual - expected).abs() <= 1e-5);
        }
        Ok(())
    }

    #[test]
    fn snapshots_and_distinct_rows_reject_invalid_counts_and_aliasing() -> Result<()> {
        // Zero and negative snapshots are rejected.
        for bad_snap in [0_i32, -1] {
            let mut bad = build_gdn(2, 1, 2, 2)?;
            bad.backend.allocate(0)?;
            bad.backend.set_values(bad.q, &[0.5; 4])?;
            bad.backend.set_values(bad.k, &[0.5; 4])?;
            bad.backend.set_values(bad.v, &[0.5; 4])?;
            bad.backend.set_values(bad.g, &[0.0; 2])?;
            bad.backend.set_values(bad.b, &[0.5; 2])?;
            bad.backend.set_values(bad.s, &[0.0; 4])?;
            bad.backend.info_mut(bad.out)?.parameters[..4].copy_from_slice(&bad_snap.to_le_bytes());
            assert!(bad.backend.compute(bad.graph).is_err());
        }

        // Duplicate sequence cache rows (`[0, 0]`) must be rejected; distinct `[1, 0]` succeeds.
        for (rows, should_pass) in [([0_i32, 0], false), ([1_i32, 0], true)] {
            let mut backend = CpuBackend::new();
            let query = backend.tensor(0, 0, [2, 1, 1, 2])?;
            let key = backend.tensor(0, 0, [2, 1, 1, 2])?;
            let value = backend.tensor(0, 0, [2, 1, 1, 2])?;
            let gate = backend.tensor(0, 0, [1, 1, 1, 2])?;
            let beta = backend.tensor(0, 0, [1, 1, 1, 2])?;
            let state = backend.tensor(0, 0, [4, 2, 1, 1])?;
            let row_ids = backend.tensor(0, 26, [2, 1, 1, 1])?;
            let out = tensor(&backend.call(
                "ggml_gated_delta_net_rows",
                &[
                    Argument::Context(0),
                    Argument::Tensor(query),
                    Argument::Tensor(key),
                    Argument::Tensor(value),
                    Argument::Tensor(gate),
                    Argument::Tensor(beta),
                    Argument::Tensor(state),
                    Argument::Tensor(row_ids),
                    Argument::Integer(1),
                ],
            )?)?;
            let graph = new_graph(&mut backend)?;
            backend.call(
                "ggml_build_forward_expand",
                &[Argument::Graph(graph), Argument::Tensor(out)],
            )?;
            backend.allocate(0)?;
            backend.set_values(query, &[1.0; 4])?;
            backend.set_values(key, &[0.5; 4])?;
            backend.set_values(value, &[0.25; 4])?;
            backend.set_values(gate, &[0.0; 2])?;
            backend.set_values(beta, &[0.5; 2])?;
            backend.set_values(state, &[0.0; 8])?;
            set_i32(&mut backend, row_ids, &rows)?;
            let outcome = backend.compute(graph);
            assert_eq!(outcome.is_ok(), should_pass);
        }
        Ok(())
    }

    fn eval_single_token_replay(
        replay_count: i32,
        bank: i32,
        truncate_records: bool,
    ) -> Result<Vec<f32>> {
        let width = 2;
        let heads = 1;
        let tokens = 1;
        let snapshots = 4;
        let record_width = heads * (2 * width + 1);
        let mut backend = CpuBackend::new();
        let query = backend.tensor(0, 0, [width, heads, tokens, 1])?;
        let key = backend.tensor(0, 0, [width, heads, tokens, 1])?;
        let value = backend.tensor(0, 0, [width, heads, tokens, 1])?;
        let gate = backend.tensor(0, 0, [1, heads, tokens, 1])?;
        let beta = backend.tensor(0, 0, [1, heads, tokens, 1])?;
        let state = backend.tensor(0, 0, [width, width, heads, 1])?;
        let records = backend.tensor(0, 0, [record_width * (snapshots - 1), 2, 1, 1])?;
        let control = backend.tensor(0, 26, [2, 1, 1, 1])?;
        let out = tensor(&backend.call(
            "ggml_gated_delta_net_replay",
            &[
                Argument::Context(0),
                Argument::Tensor(query),
                Argument::Tensor(key),
                Argument::Tensor(value),
                Argument::Tensor(gate),
                Argument::Tensor(beta),
                Argument::Tensor(state),
                Argument::Tensor(records),
                Argument::Tensor(control),
                Argument::Integer(i64::try_from(snapshots)?),
            ],
        )?)?;
        let graph = new_graph(&mut backend)?;
        backend.call(
            "ggml_build_forward_expand",
            &[Argument::Graph(graph), Argument::Tensor(out)],
        )?;
        backend.allocate(0)?;
        backend.set_values(query, &[1.0, 0.0])?;
        backend.set_values(key, &[0.0, 0.0])?;
        backend.set_values(value, &[0.0, 0.0])?;
        backend.set_values(gate, &[0.0])?;
        backend.set_values(beta, &[0.0])?;
        backend.set_values(state, &[2.0, 0.0, 0.0, 2.0])?;
        let mut rec_data = vec![9.0_f32; record_width * (snapshots - 1) * 2];
        rec_data[..record_width].copy_from_slice(&[1.0, 2.0, 1.0, 0.0, 0.5]);
        rec_data[record_width..2 * record_width].copy_from_slice(&[0.5, -1.0, 0.0, 1.0, 0.5]);
        backend.set_values(records, &rec_data)?;
        set_i32(&mut backend, control, &[replay_count, bank])?;
        if truncate_records {
            backend.info_mut(records)?.shape = [record_width, 1, 1, 1];
        }
        backend.compute(graph)?;
        backend.values(out)
    }

    #[test]
    fn replay_allows_multi_record_prefix_on_single_token_and_rejects_bad_bounds() -> Result<()> {
        // `tokens = 1, snapshots = 4` has `written_depth = 1`, yet prior step's `count = 2 < snapshots` succeeds.
        let values = eval_single_token_replay(2, 0, false)?;
        let expected_attn = std::f32::consts::FRAC_1_SQRT_2;
        assert!((values[0] - expected_attn).abs() <= 1e-6);
        assert!((values[1] - expected_attn).abs() <= 1e-6);
        let expected_state = [1.0_f32, 0.5, 1.0, -0.5];
        for (&actual, &expected) in values[2..6].iter().zip(&expected_state) {
            assert!((actual - expected).abs() <= 1e-6);
        }

        // `count == snapshots`, `bank > 1`, and truncated `records` all fail cleanly.
        assert!(eval_single_token_replay(4, 0, false).is_err());
        assert!(eval_single_token_replay(1, 2, false).is_err());
        assert!(eval_single_token_replay(2, 0, true).is_err());
        Ok(())
    }
}
