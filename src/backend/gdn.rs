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
    #[allow(
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
        let snapshots = integer(&info.parameters, 0) as usize;
        let replay = info.operation == 101;
        let raw = integer(&info.parameters, 1) != 0;
        let rows = if replay {
            None
        } else {
            ids[6].map(|id| self.integer_values(id)).transpose()?
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
        let state_per_head = width * width;
        let state_per_sequence = state_per_head * heads;
        let all_state = state_per_sequence * sequences;
        let attention_elements = width * heads * tokens * sequences;
        let record_width = heads * (2 * width + 1);
        let base_token = if replay {
            tokens - 1 - (tokens - 1).min(snapshots - 1)
        } else {
            0
        };
        let scale = 1.0 / (width as f32).sqrt();
        for sequence in 0..sequences {
            for head in 0..heads {
                let state_row = if let Some(rows) = &rows {
                    let row = usize::try_from(rows[sequence])?;
                    if row >= shapes[5][1] {
                        return Err(invalid("CPU GDN cache row exceeds state table"));
                    }
                    row
                } else {
                    sequence
                };
                let start = state_row * state_per_sequence + head * state_per_head;
                let mut state = input[5]
                    .get(start..start + state_per_head)
                    .ok_or_else(|| invalid("CPU GDN initial state extent differs"))?
                    .to_vec();
                if let (Some(records), Some(control), Some(record_shape)) =
                    (&records, &control, record_shape)
                {
                    let count = usize::try_from(control[2 * sequence])?;
                    let bank = usize::try_from(control[2 * sequence + 1])?;
                    if count >= snapshots || bank > 1 {
                        return Err(invalid("CPU GDN replay count or bank differs"));
                    }
                    let record_base = linear([0, bank, sequence, 0], record_shape);
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
                        let value = gate + bias[head];
                        #[allow(
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
