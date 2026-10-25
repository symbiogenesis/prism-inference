// Copyright (c) 2023-2026 The ggml/llama.cpp authors
// SPDX-License-Identifier: MIT
//! Safe reference state scans, recurrent linear attention and hyperconnections.

use super::{CpuBackend, TensorInfo, coordinates, linear};
use crate::{Result, invalid};

fn integer(params: &[u8; 64], index: usize) -> i32 {
    i32::from_le_bytes(std::array::from_fn(|offset| params[index * 4 + offset]))
}
fn float(params: &[u8; 64], index: usize) -> f32 {
    f32::from_bits(integer(params, index) as u32)
}

impl CpuBackend {
    #[allow(
        clippy::too_many_lines,
        reason = "The state scan retains source traversal, recurrence order, and reverse snapshot slots."
    )]
    pub(super) fn scan_values(&self, info: &TensorInfo, mut output: Vec<f32>) -> Result<Vec<f32>> {
        let inputs = info
            .sources
            .iter()
            .map(|&id| self.values(id))
            .collect::<Result<Vec<_>>>()?;
        let shapes = info
            .sources
            .iter()
            .map(|&id| self.info(id).map(|info| info.shape))
            .collect::<Result<Vec<_>>>()?;
        let [state_width, head_width, heads, _] = shapes[0];
        let tokens = shapes[1][2];
        let sequences = shapes[1][3];
        let groups = shapes[4][1];
        let snapshots = integer(&info.parameters, 0) as usize;
        let ids = self.integer_values(info.sources[6])?;
        let state_elements = state_width * head_width * heads;
        let output_elements = inputs[1].len();
        if output.len() != output_elements + snapshots * state_elements * sequences {
            return Err(invalid("CPU state scan output extent differs"));
        }
        for sequence in 0..sequences {
            let initial = usize::try_from(ids[sequence])?;
            if initial >= shapes[0][3] {
                return Err(invalid("CPU state scan row exceeds state extent"));
            }
            let mut state =
                inputs[0][initial * state_elements..(initial + 1) * state_elements].to_vec();
            for token in 0..tokens {
                for head in 0..heads {
                    let delta = inputs[2][linear([head, token, sequence, 0], shapes[2])];
                    let delta = if delta > 20.0 {
                        delta
                    } else {
                        #[allow(
                            clippy::imprecise_flops,
                            reason = "The pinned scan helper rounds 1+exp before logf."
                        )]
                        {
                            (1.0 + delta.exp()).ln()
                        }
                    };
                    let group = head / (heads / groups);
                    for dimension in 0..head_width {
                        let activation = inputs[1]
                            [linear([dimension, head, token, sequence], shapes[1])]
                            * delta;
                        let mut sum = 0.0_f32;
                        for component in 0..state_width {
                            let state_index =
                                component + state_width * (dimension + head_width * head);
                            let decay_index = if shapes[3][0] == 1 {
                                head
                            } else {
                                component + state_width * head
                            };
                            let decay = (delta * inputs[3][decay_index]).exp();
                            let projection_index =
                                linear([component, group, token, sequence], shapes[4]);
                            let value = state[state_index] * decay
                                + inputs[4][projection_index] * activation;
                            state[state_index] = value;
                            sum += value * inputs[5][projection_index];
                        }
                        output[linear([dimension, head, token, sequence], shapes[1])] = sum;
                    }
                }
                let slot = tokens - 1 - token;
                if snapshots > 1 && slot > 0 && slot < snapshots {
                    let start = output_elements + (slot * sequences + sequence) * state_elements;
                    output[start..start + state_elements].copy_from_slice(&state);
                }
            }
            let start = output_elements + sequence * state_elements;
            output[start..start + state_elements].copy_from_slice(&state);
        }
        Ok(output)
    }

    pub(super) fn linear_attention_values(&self, info: &TensorInfo) -> Result<Vec<f32>> {
        let inputs = info
            .sources
            .iter()
            .map(|&id| self.values(id))
            .collect::<Result<Vec<_>>>()?;
        let first_shape = self.info(info.sources[0])?.shape;
        let state_index = match info.operation {
            82 => 5,
            83 => 4,
            84 => 6,
            _ => return Err(invalid("CPU recurrent attention operation differs")),
        };
        let sequences = self.info(info.sources[state_index])?.shape[1];
        let width = first_shape[0];
        let heads = first_shape[1];
        let tokens = first_shape[2];
        let channels = width * heads;
        if sequences == 0 || !tokens.is_multiple_of(sequences) {
            return Err(invalid("CPU recurrent attention sequence count differs"));
        }
        let mut state = inputs[state_index].clone();
        let mut output = vec![0.0_f32; info.elements()];
        for token in 0..tokens {
            let sequence = token / (tokens / sequences);
            let token_start = token * channels;
            let sequence_start = sequence * width * channels;
            for head in 0..heads {
                let head_start = token_start + head * width;
                let state_start = sequence_start + head * width * width;
                for row in 0..width {
                    if info.operation == 84 {
                        let mut auxiliary = 0.0_f32;
                        for column in 0..width {
                            auxiliary += inputs[4][head_start + column]
                                * state[state_start + row * width + column];
                        }
                        let mut sum = 0.0_f32;
                        for column in 0..width {
                            let index = head_start + column;
                            let state_index = state_start + row * width + column;
                            let product = inputs[3][head_start + row] * inputs[2][index];
                            let value = state[state_index] * inputs[1][index]
                                + product
                                + auxiliary * inputs[5][index];
                            state[state_index] = value;
                            sum += value * inputs[0][index];
                        }
                        output[head_start + row] = sum;
                    } else {
                        let key = inputs[0][head_start + row];
                        let query = inputs[2][head_start + row]
                            * if info.operation == 83 {
                                float(&info.parameters, 0)
                            } else {
                                1.0
                            };
                        let gate =
                            inputs[if info.operation == 82 { 4 } else { 3 }][head_start + row];
                        for column in 0..width {
                            let state_index = state_start + row * width + column;
                            let product = inputs[1][head_start + column] * key;
                            let previous = state[state_index];
                            let value = previous * gate + product;
                            state[state_index] = value;
                            let contribution = if info.operation == 82 {
                                product * inputs[3][head * width + row] + previous
                            } else {
                                value
                            };
                            output[head_start + column] += contribution * query;
                        }
                    }
                }
            }
        }
        output[channels * tokens..].copy_from_slice(&state);
        Ok(output)
    }

    pub(super) fn hyperconnection_values(
        &self,
        info: &TensorInfo,
        input: &[f32],
    ) -> Result<Vec<f32>> {
        let mut output = vec![0.0_f32; info.elements()];
        let first_shape = self.info(info.sources[0])?.shape;
        match info.operation {
            88 => {
                let scale = self.values(info.sources[1])?;
                let base = self.values(info.sources[2])?;
                let epsilon = float(&info.parameters, 0);
                let iterations = integer(&info.parameters, 1);
                for token in 0..first_shape[1] {
                    let mut matrix = [0.0_f32; 16];
                    for source in 0..4 {
                        let mut maximum = f32::NEG_INFINITY;
                        for destination in 0..4 {
                            let index = destination + 4 * source;
                            let value = input[8 + index + 24 * token] * scale[2] + base[8 + index];
                            matrix[index] = value;
                            maximum = maximum.max(value);
                        }
                        let mut sum = 0.0_f32;
                        for destination in 0..4 {
                            let index = destination + 4 * source;
                            matrix[index] = (matrix[index] - maximum).exp();
                            sum += matrix[index];
                        }
                        let inverse = 1.0 / sum;
                        for destination in 0..4 {
                            let index = destination + 4 * source;
                            matrix[index] = matrix[index] * inverse + epsilon;
                        }
                    }
                    for iteration in 0..iterations {
                        if iteration != 0 {
                            for source in 0..4 {
                                let mut sum = epsilon;
                                for destination in 0..4 {
                                    sum += matrix[destination + 4 * source];
                                }
                                let inverse = 1.0 / sum;
                                for destination in 0..4 {
                                    matrix[destination + 4 * source] *= inverse;
                                }
                            }
                        }
                        for destination in 0..4 {
                            let mut sum = epsilon;
                            for source in 0..4 {
                                sum += matrix[destination + 4 * source];
                            }
                            let inverse = 1.0 / sum;
                            for source in 0..4 {
                                matrix[destination + 4 * source] *= inverse;
                            }
                        }
                    }
                    output[token * 16..(token + 1) * 16].copy_from_slice(&matrix);
                }
            }
            89 => {
                let weights = self.values(info.sources[1])?;
                for (index, value) in output.iter_mut().enumerate() {
                    let [dimension, token, _, _] = coordinates(index, info.shape);
                    let mut sum = 0.0_f32;
                    for channel in 0..first_shape[1] {
                        sum += input[linear([dimension, channel, token, 0], first_shape)]
                            * weights[channel + first_shape[1] * token];
                    }
                    *value = sum;
                }
            }
            90 => {
                let residual = self.values(info.sources[1])?;
                let residual_shape = self.info(info.sources[1])?.shape;
                let post = self.values(info.sources[2])?;
                let combination = self.values(info.sources[3])?;
                let channels = info.shape[1];
                for (index, value) in output.iter_mut().enumerate() {
                    let [dimension, destination, token, _] = coordinates(index, info.shape);
                    let mut sum = input[dimension + first_shape[0] * token]
                        * post[destination + channels * token];
                    for source in 0..channels {
                        sum += residual[linear([dimension, source, token, 0], residual_shape)]
                            * combination[destination + channels * (source + channels * token)];
                    }
                    *value = sum;
                }
            }
            _ => return Err(invalid("CPU hyperconnection operation differs")),
        }
        Ok(output)
    }
}
