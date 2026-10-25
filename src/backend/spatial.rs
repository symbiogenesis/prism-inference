// Copyright (c) 2023-2026 The ggml authors
// SPDX-License-Identifier: MIT
//! Window and relative position kernels from the pinned production CPU backend.

use super::{CpuBackend, TensorId, TensorInfo, coordinates, linear};
use crate::{Result, invalid};

fn integer(parameters: &[u8; 64], index: usize) -> Result<usize> {
    Ok(usize::try_from(i32::from_le_bytes(std::array::from_fn(
        |offset| parameters[index * 4 + offset],
    )))?)
}

impl CpuBackend {
    pub(super) fn spatial_values(&self, info: &TensorInfo, input: &[f32]) -> Result<Vec<f32>> {
        let source = *info
            .sources
            .first()
            .ok_or_else(|| invalid("CPU spatial input missing"))?;
        let source_shape = self.info(source)?.shape;
        let mut output = vec![0.0; info.elements()];
        match info.operation {
            78 => {
                let columns = integer(&info.parameters, 0)?;
                let window = integer(&info.parameters, 2)?;
                if columns == 0 || window == 0 {
                    return Err(invalid("CPU window partition dimensions must be positive"));
                }
                for (index, value) in output.iter_mut().enumerate() {
                    let [channel, x, y, patch] = coordinates(index, info.shape);
                    let x = patch % columns * window + x;
                    let y = patch / columns * window + y;
                    if x < source_shape[1] && y < source_shape[2] {
                        *value = input[linear([channel, x, y, 0], source_shape)];
                    }
                }
            }
            79 => {
                let window = integer(&info.parameters, 0)?;
                if window == 0 {
                    return Err(invalid("CPU window size must be positive"));
                }
                let columns = info.shape[1].div_ceil(window);
                for (index, value) in output.iter_mut().enumerate() {
                    let [channel, x, y, _] = coordinates(index, info.shape);
                    let patch = y / window * columns + x / window;
                    *value = input[linear([channel, x % window, y % window, patch], source_shape)];
                }
            }
            81 => {
                let width_id = *info
                    .sources
                    .get(1)
                    .ok_or_else(|| invalid("CPU width positions missing"))?;
                let height_id = *info
                    .sources
                    .get(2)
                    .ok_or_else(|| invalid("CPU height positions missing"))?;
                let width = self.values(width_id)?;
                let height = self.values(height_id)?;
                let position_shape = self.info(width_id)?.shape;
                let side = position_shape[0];
                output.copy_from_slice(input);
                // Preserve the native interleaving of height and width additions;
                // reassociating to input + (height + width) changes F32 results.
                for (index, (&width, &height)) in width.iter().zip(&height).enumerate() {
                    let coordinate = coordinates(index, position_shape);
                    let height_start = index * side;
                    let width_start = height_start - (side - 1) * coordinate[0];
                    for offset in 0..side {
                        output[height_start + offset] += height;
                        output[width_start + offset * side] += width;
                    }
                }
            }
            _ => return Err(invalid("unsupported CPU spatial operation")),
        }
        Ok(output)
    }

    pub(super) fn relative_positions(&mut self, tensor: TensorId) -> Result<()> {
        let info = self.info(tensor)?.clone();
        let source = *info
            .sources
            .first()
            .ok_or_else(|| invalid("CPU relative positions missing"))?;
        let [channels, keys, queries, _] = info.shape;
        let mut output = Vec::new();
        output.try_reserve_exact(info.bytes()?)?;
        // Native GET_REL_POS copies 16-bit payloads, including BF16 input,
        // into an F16 result. Retain every bit rather than converting values.
        for query in 0..queries {
            for key in 0..keys {
                let position = keys - key - 1 + query;
                output.extend_from_slice(self.get(
                    source,
                    position * channels * 2,
                    channels * 2,
                )?);
            }
        }
        self.ensure_storage(tensor)?;
        self.set(tensor, 0, &output)
    }
}
