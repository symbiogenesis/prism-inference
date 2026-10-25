// Copyright (c) 2023-2026 The ggml/llama.cpp authors
// SPDX-License-Identifier: MIT
//! Safe online attention using the runtime's packed activation dot contracts.

use super::{CpuBackend, TensorInfo, linear};
use crate::{Result, invalid};

fn float(params: &[u8; 64], index: usize) -> f32 {
    f32::from_le_bytes(std::array::from_fn(|offset| params[index * 4 + offset]))
}

impl CpuBackend {
    pub(super) fn indexer_values(&self, info: &TensorInfo, queries: &[f32]) -> Result<Vec<f32>> {
        let query_shape = self.info(info.sources[0])?.shape;
        let key_shape = self.info(info.sources[1])?.shape;
        let weight_shape = self.info(info.sources[2])?.shape;
        let mask_shape = self.info(info.sources[3])?.shape;
        let keys = self.values(info.sources[1])?;
        let weights = self.values(info.sources[2])?;
        let mask = self.values(info.sources[3])?;
        let mut output = vec![0.0; info.elements()];
        for stream in 0..query_shape[3] {
            for token in 0..query_shape[2] {
                for key in 0..key_shape[2] {
                    let key_start = linear([0, 0, key, stream], key_shape);
                    let mut score = 0.0_f32;
                    for head in 0..query_shape[1] {
                        let query_start = linear([0, head, token, stream], query_shape);
                        let dot = queries[query_start..query_start + query_shape[0]]
                            .iter()
                            .zip(&keys[key_start..key_start + query_shape[0]])
                            .map(|(&query, &key)| f64::from(query) * f64::from(key))
                            .sum::<f64>() as f32;
                        score +=
                            dot.max(0.0) * weights[linear([head, token, 0, stream], weight_shape)];
                    }
                    output[linear([key, token, 0, stream], info.shape)] =
                        score + mask[linear([key, token, 0, stream % mask_shape[3]], mask_shape)];
                }
            }
        }
        Ok(output)
    }

    #[allow(
        clippy::too_many_lines,
        reason = "The pinned online softmax, half accumulator and sink updates retain their operation order."
    )]
    pub(super) fn attention_values(&self, info: &TensorInfo, queries: &[f32]) -> Result<Vec<f32>> {
        let query = *info
            .sources
            .first()
            .ok_or_else(|| invalid("CPU attention queries missing"))?;
        let keys = *info
            .sources
            .get(1)
            .ok_or_else(|| invalid("CPU attention keys missing"))?;
        let values = *info
            .sources
            .get(2)
            .ok_or_else(|| invalid("CPU attention values missing"))?;
        let query_shape = self.info(query)?.shape;
        let key_shape = self.info(keys)?.shape;
        let value_info = self.info(values)?;
        let value_shape = value_info.shape;
        let value_data = self.values(values)?;
        let mask = info.source_slots[3].map(|id| self.values(id)).transpose()?;
        let mask_shape = info.source_slots[3]
            .map(|id| self.info(id).map(|info| info.shape))
            .transpose()?;
        let sinks = info.source_slots[4].map(|id| self.values(id)).transpose()?;
        let softcap = float(&info.parameters, 2);
        let mut scale = float(&info.parameters, 0);
        if softcap != 0.0 {
            scale /= softcap;
        }
        let max_bias = float(&info.parameters, 1);
        let power_heads = 1_usize << query_shape[2].ilog2();
        let first_slope = 2.0_f32.powf(-max_bias / power_heads as f32);
        let second_slope = 2.0_f32.powf(-(max_bias / 2.0) / power_heads as f32);
        let mut output = vec![0.0; info.elements()];
        for batch in 0..query_shape[3] {
            for head in 0..query_shape[2] {
                let key_head = head / (query_shape[2] / key_shape[2]);
                let value_head = head / (query_shape[2] / value_shape[2]);
                let matrix = self.matrix(keys, key_head, batch)?;
                let slope = if max_bias <= 0.0 {
                    1.0
                } else if head < power_heads {
                    first_slope.powi((head + 1) as i32)
                } else {
                    second_slope.powi((2 * (head - power_heads) + 1) as i32)
                };
                for token in 0..query_shape[1] {
                    let query_start = linear([0, token, head, batch], query_shape);
                    let scores = crate::kernels::matvec(
                        &matrix,
                        &queries[query_start..query_start + query_shape[0]],
                    )?;
                    let mut accumulator = vec![0.0_f32; value_shape[0]];
                    let mut maximum = f32::NEG_INFINITY;
                    let mut sum = 0.0_f32;
                    for (key, &dot) in scores.iter().enumerate() {
                        let bias = if let (Some(mask), Some(shape)) = (&mask, mask_shape) {
                            slope
                                * mask
                                    [linear([key, token, head % shape[2], batch % shape[3]], shape)]
                        } else {
                            0.0
                        };
                        if bias == f32::NEG_INFINITY {
                            continue;
                        }
                        let score = dot * scale;
                        let score = if softcap == 0.0 {
                            score
                        } else {
                            softcap * score.tanh()
                        };
                        let score = score + bias;
                        let mut previous_scale = 1.0;
                        let mut weight = 1.0;
                        if score > maximum {
                            previous_scale = (maximum - score).exp();
                            maximum = score;
                            for value in &mut accumulator {
                                *value *= previous_scale;
                                if value_info.kind == 1 {
                                    *value = crate::kernels::round_f16(*value);
                                }
                            }
                        } else {
                            weight = (score - maximum).exp();
                        }
                        for (dimension, value) in accumulator.iter_mut().enumerate() {
                            *value += value_data
                                [linear([dimension, key, value_head, batch], value_shape)]
                                * weight;
                            if value_info.kind == 1 {
                                *value = crate::kernels::round_f16(*value);
                            }
                        }
                        sum = sum * previous_scale + weight;
                    }
                    if let Some(sinks) = &sinks {
                        let score = sinks[head];
                        let mut previous_scale = 1.0;
                        let mut weight = 1.0;
                        if score > maximum {
                            previous_scale = (maximum - score).exp();
                            for value in &mut accumulator {
                                *value *= previous_scale;
                            }
                        } else {
                            weight = (score - maximum).exp();
                        }
                        sum = sum * previous_scale + weight;
                    }
                    let inverse = if sum == 0.0 { 0.0 } else { 1.0 / sum };
                    for (dimension, &value) in accumulator.iter().enumerate() {
                        output[linear([dimension, head, token, batch], info.shape)] =
                            value * inverse;
                    }
                }
            }
        }
        Ok(output)
    }
}
