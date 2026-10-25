// Copyright (c) 2023-2026 The ggml/llama.cpp authors
// SPDX-License-Identifier: MIT
//! Safe image-gradient kernels adapted from the pinned GGML CPU contracts.

use super::{CpuBackend, TensorInfo, linear};
use crate::{Result, invalid};

fn integer(parameters: &[u8; 64], index: usize) -> i32 {
    i32::from_le_bytes(std::array::from_fn(|offset| parameters[index * 4 + offset]))
}

impl CpuBackend {
    pub(super) fn backward_image_values(
        &self,
        info: &TensorInfo,
        gradient: &[f32],
    ) -> Result<Vec<f32>> {
        let gradient_info = self.info(
            *info
                .sources
                .first()
                .ok_or_else(|| invalid("CPU image gradient missing"))?,
        )?;
        if info.kind != 0 || gradient_info.kind != 0 || gradient.len() != gradient_info.elements() {
            return Err(invalid("CPU image gradients require binary32 storage"));
        }
        match info.operation {
            53 => self.im2col_backward_values(info, gradient_info.shape, gradient),
            62 => self.pool_backward_values(info, gradient_info.shape, gradient),
            _ => Err(invalid("CPU image backward operation differs")),
        }
    }

    #[allow(
        clippy::too_many_lines,
        reason = "Pinned im2col backward retains its pixel and kernel loop ordering."
    )]
    fn im2col_backward_values(
        &self,
        info: &TensorInfo,
        gradient_shape: [usize; 4],
        gradient: &[f32],
    ) -> Result<Vec<f32>> {
        let kernel = self.info(
            *info
                .sources
                .get(1)
                .ok_or_else(|| invalid("CPU im2col backward kernel missing"))?,
        )?;
        if !matches!(kernel.kind, 0 | 1) {
            return Err(invalid("CPU im2col backward kernel type differs"));
        }
        let parameters = &info.parameters;
        let two_dimensional = integer(parameters, 6) == 1;
        let shape = info.shape;
        let batches = shape[if two_dimensional { 3 } else { 2 }];
        let channels = shape[if two_dimensional { 2 } else { 1 }];
        let height = if two_dimensional { shape[1] } else { 1 };
        let kernel_height = if two_dimensional { kernel.shape[1] } else { 1 };
        let kernel_width = kernel.shape[0];
        let stride = [integer(parameters, 0), integer(parameters, 1)];
        let padding = [integer(parameters, 2), integer(parameters, 3)];
        let dilation = [integer(parameters, 4), integer(parameters, 5)];
        let gradient_width = channels
            .checked_mul(kernel_height)
            .and_then(|count| count.checked_mul(kernel_width))
            .ok_or_else(|| invalid("CPU im2col backward channel extent overflow"))?;
        if stride[0] <= 0
            || dilation[0] <= 0
            || two_dimensional && (stride[1] <= 0 || dilation[1] <= 0)
            || gradient_shape[0] != gradient_width
            || two_dimensional && gradient_shape[3] != batches
            || !two_dimensional
                && (shape[3] != 1 || gradient_shape[2] != batches || gradient_shape[3] != 1)
        {
            return Err(invalid("CPU im2col backward geometry differs"));
        }
        let output_height = if two_dimensional {
            gradient_shape[2]
        } else {
            1
        };
        let output_width = gradient_shape[1];
        let mut output = vec![0.0_f32; info.elements()];
        for batch in 0..batches {
            for channel in 0..channels {
                for y in 0..height {
                    for x in 0..shape[0] {
                        let mut sum = 0.0_f32;
                        for kernel_y in 0..kernel_height {
                            for kernel_x in 0..kernel_width {
                                let numerator_x = x as i64 + i64::from(padding[0])
                                    - kernel_x as i64 * i64::from(dilation[0]);
                                if numerator_x % i64::from(stride[0]) != 0 {
                                    continue;
                                }
                                let source_x = numerator_x / i64::from(stride[0]);
                                let source_y = if two_dimensional {
                                    let numerator_y = y as i64 + i64::from(padding[1])
                                        - kernel_y as i64 * i64::from(dilation[1]);
                                    if numerator_y % i64::from(stride[1]) != 0 {
                                        continue;
                                    }
                                    numerator_y / i64::from(stride[1])
                                } else {
                                    0
                                };
                                if source_x < 0
                                    || source_x >= output_width as i64
                                    || source_y < 0
                                    || source_y >= output_height as i64
                                {
                                    continue;
                                }
                                let kernel_index =
                                    (channel * kernel_height + kernel_y) * kernel_width + kernel_x;
                                let source = if two_dimensional {
                                    [kernel_index, source_x as usize, source_y as usize, batch]
                                } else {
                                    [kernel_index, source_x as usize, batch, 0]
                                };
                                sum += gradient[linear(source, gradient_shape)];
                            }
                        }
                        let destination = if two_dimensional {
                            [x, y, channel, batch]
                        } else {
                            [x, channel, batch, 0]
                        };
                        output[linear(destination, shape)] = sum;
                    }
                }
            }
        }
        Ok(output)
    }

    #[allow(
        clippy::too_many_lines,
        reason = "Pinned pooling backward retains window visitation and first-maximum selection."
    )]
    fn pool_backward_values(
        &self,
        info: &TensorInfo,
        gradient_shape: [usize; 4],
        gradient: &[f32],
    ) -> Result<Vec<f32>> {
        let forward = *info
            .sources
            .get(1)
            .ok_or_else(|| invalid("CPU pooling backward forward input missing"))?;
        let forward_info = self.info(forward)?;
        let parameters = &info.parameters;
        let average = match integer(parameters, 0) {
            0 => false,
            1 => true,
            _ => return Err(invalid("CPU pooling backward operation differs")),
        };
        let kernel = [integer(parameters, 1), integer(parameters, 2)];
        let stride = [integer(parameters, 3), integer(parameters, 4)];
        let padding = [integer(parameters, 5), integer(parameters, 6)];
        let shape = info.shape;
        if forward_info.kind != 0
            || forward_info.shape != shape
            || gradient_shape[2..] != shape[2..]
            || kernel.iter().chain(&stride).any(|&value| value <= 0)
        {
            return Err(invalid("CPU pooling backward geometry differs"));
        }
        let kernel_area = kernel[0]
            .checked_mul(kernel[1])
            .ok_or_else(|| invalid("CPU pooling backward kernel area overflow"))?;
        let forward = if average {
            Vec::new()
        } else {
            self.values(forward)?
        };
        let mut output = vec![0.0_f32; info.elements()];
        for batch in 0..shape[3] {
            for channel in 0..shape[2] {
                for output_y in 0..gradient_shape[1] {
                    for output_x in 0..gradient_shape[0] {
                        let gradient =
                            gradient[linear([output_x, output_y, channel, batch], gradient_shape)];
                        let origin_x =
                            output_x as i64 * i64::from(stride[0]) - i64::from(padding[0]);
                        let origin_y =
                            output_y as i64 * i64::from(stride[1]) - i64::from(padding[1]);
                        let contribution = gradient / kernel_area as f32;
                        let mut maximum = -f32::MAX;
                        let mut maximum_index = None;
                        for kernel_y in 0..kernel[1] {
                            let y = origin_y + i64::from(kernel_y);
                            if y < 0 || y >= shape[1] as i64 {
                                continue;
                            }
                            for kernel_x in 0..kernel[0] {
                                let x = origin_x + i64::from(kernel_x);
                                if x < 0 || x >= shape[0] as i64 {
                                    continue;
                                }
                                let index = linear([x as usize, y as usize, channel, batch], shape);
                                if average {
                                    output[index] += contribution;
                                } else {
                                    let value = forward[index];
                                    if value <= maximum {
                                        continue;
                                    }
                                    maximum = value;
                                    maximum_index = Some(index);
                                }
                            }
                        }
                        if !average && let Some(index) = maximum_index {
                            output[index] += gradient;
                        }
                    }
                }
            }
        }
        Ok(output)
    }
}
