// Copyright (c) 2023-2026 The ggml/llama.cpp authors
// SPDX-License-Identifier: MIT
//! Safe image extraction, convolution, pooling and interpolation kernels.

use super::{CpuBackend, TensorInfo, coordinates, linear};
use crate::{Result, invalid};

fn integer(params: &[u8; 64], index: usize) -> i32 {
    i32::from_le_bytes(std::array::from_fn(|offset| params[index * 4 + offset]))
}

fn cubic(values: [f32; 4], position: f32) -> f32 {
    let alpha = -0.75;
    let first = |x: f32| ((alpha + 2.0) * x - (alpha + 3.0)) * x * x + 1.0;
    let second = |x: f32| ((alpha * x - 5.0 * alpha) * x + 8.0 * alpha) * x - 4.0 * alpha;
    values[0] * second(position + 1.0)
        + values[1] * first(position)
        + values[2] * first(1.0 - position)
        + values[3] * second(2.0 - position)
}

impl CpuBackend {
    #[allow(
        clippy::too_many_lines,
        clippy::many_single_char_names,
        reason = "Pinned image kernels retain conventional pixel coordinates and exact arithmetic order."
    )]
    pub(super) fn image_values(&self, info: &TensorInfo, input: &[f32]) -> Result<Vec<f32>> {
        let p = &info.parameters;
        let shape = info.shape;
        let source = *info
            .sources
            .first()
            .ok_or_else(|| invalid("CPU image input missing"))?;
        let source_shape = self.info(source)?.shape;
        let mut output = vec![0.0_f32; info.elements()];
        match info.operation {
            52 => {
                let image = *info
                    .sources
                    .get(1)
                    .ok_or_else(|| invalid("CPU im2col image missing"))?;
                let image_shape = self.info(image)?.shape;
                let values = self.values(image)?;
                let two_dimensional = integer(p, 6) != 0;
                let kernel_height = if two_dimensional { source_shape[1] } else { 1 };
                for (index, value) in output.iter_mut().enumerate() {
                    let coord = coordinates(index, shape);
                    let kernel = coordinates(
                        coord[0],
                        [
                            source_shape[0],
                            kernel_height,
                            shape[0] / (source_shape[0] * kernel_height),
                            1,
                        ],
                    );
                    let x = coord[1] as i64 * i64::from(integer(p, 0))
                        + kernel[0] as i64 * i64::from(integer(p, 4))
                        - i64::from(integer(p, 2));
                    let y = if two_dimensional {
                        coord[2] as i64 * i64::from(integer(p, 1))
                            + kernel[1] as i64 * i64::from(integer(p, 5))
                            - i64::from(integer(p, 3))
                    } else {
                        kernel[2] as i64
                    };
                    let source_index = if two_dimensional {
                        [x, y, kernel[2] as i64, coord[3] as i64]
                    } else {
                        [x, y, coord[2] as i64, 0]
                    };
                    if source_index
                        .iter()
                        .zip(image_shape)
                        .all(|(&coordinate, dimension)| {
                            coordinate >= 0 && coordinate < dimension as i64
                        })
                    {
                        *value =
                            values[linear(source_index.map(|index| index as usize), image_shape)];
                    }
                }
            }
            54 => {
                let image = *info
                    .sources
                    .get(1)
                    .ok_or_else(|| invalid("CPU 3D im2col image missing"))?;
                let image_shape = self.info(image)?.shape;
                let image = self.values(image)?;
                let channels = integer(p, 9) as usize;
                let batches = image_shape[3] / channels;
                let depth = shape[3] / batches;
                for (index, value) in output.iter_mut().enumerate() {
                    let coordinate = coordinates(index, shape);
                    let kernel = coordinates(
                        coordinate[0],
                        [source_shape[0], source_shape[1], source_shape[2], channels],
                    );
                    let position = [coordinate[1], coordinate[2], coordinate[3] % depth];
                    let batch = coordinate[3] / depth;
                    let source_index: [i64; 3] = std::array::from_fn(|axis| {
                        position[axis] as i64 * i64::from(integer(p, axis))
                            + kernel[axis] as i64 * i64::from(integer(p, axis + 6))
                            - i64::from(integer(p, axis + 3))
                    });
                    if source_index
                        .iter()
                        .zip(image_shape)
                        .all(|(&index, dimension)| index >= 0 && index < dimension as i64)
                    {
                        *value = image[linear(
                            [
                                source_index[0] as usize,
                                source_index[1] as usize,
                                source_index[2] as usize,
                                kernel[3] + channels * batch,
                            ],
                            image_shape,
                        )];
                    }
                }
            }
            56 | 58 => {
                let image = *info
                    .sources
                    .get(1)
                    .ok_or_else(|| invalid("CPU convolution input missing"))?;
                let image_shape = self.info(image)?.shape;
                let image = self.values(image)?;
                for (index, value) in output.iter_mut().enumerate() {
                    let [x, y, channel, batch] = coordinates(index, shape);
                    let mut sum = 0.0_f32;
                    let input_channels = if info.operation == 58 {
                        1
                    } else {
                        source_shape[2]
                    };
                    for input_channel in 0..input_channels {
                        for ky in 0..source_shape[1] {
                            for kx in 0..source_shape[0] {
                                let ix = x as i64 * i64::from(integer(p, 0))
                                    + kx as i64 * i64::from(integer(p, 4))
                                    - i64::from(integer(p, 2));
                                let iy = y as i64 * i64::from(integer(p, 1))
                                    + ky as i64 * i64::from(integer(p, 5))
                                    - i64::from(integer(p, 3));
                                if ix >= 0
                                    && ix < image_shape[0] as i64
                                    && iy >= 0
                                    && iy < image_shape[1] as i64
                                {
                                    let image_channel = if info.operation == 58 {
                                        channel
                                    } else {
                                        input_channel
                                    };
                                    sum += input
                                        [linear([kx, ky, input_channel, channel], source_shape)]
                                        * image[linear(
                                            [ix as usize, iy as usize, image_channel, batch],
                                            image_shape,
                                        )];
                                }
                            }
                        }
                    }
                    *value = sum;
                }
            }
            57 => {
                let image = *info
                    .sources
                    .get(1)
                    .ok_or_else(|| invalid("CPU 3D convolution input missing"))?;
                let image_shape = self.info(image)?.shape;
                let image = self.values(image)?;
                let channels = integer(p, 9) as usize;
                let output_channels = integer(p, 11) as usize;
                for (index, value) in output.iter_mut().enumerate() {
                    let [x, y, z, channel_batch] = coordinates(index, shape);
                    let channel = channel_batch % output_channels;
                    let batch = channel_batch / output_channels;
                    let mut sum = 0.0_f32;
                    for input_channel in 0..channels {
                        for kz in 0..source_shape[2] {
                            for ky in 0..source_shape[1] {
                                for kx in 0..source_shape[0] {
                                    let position = [x, y, z];
                                    let kernel = [kx, ky, kz];
                                    let source_index: [i64; 3] = std::array::from_fn(|axis| {
                                        position[axis] as i64 * i64::from(integer(p, axis))
                                            + kernel[axis] as i64 * i64::from(integer(p, axis + 6))
                                            - i64::from(integer(p, axis + 3))
                                    });
                                    if source_index.iter().zip(image_shape).all(
                                        |(&coordinate, dimension)| {
                                            coordinate >= 0 && coordinate < dimension as i64
                                        },
                                    ) {
                                        sum += input[linear(
                                            [kx, ky, kz, input_channel + channels * channel],
                                            source_shape,
                                        )] * image[linear(
                                            [
                                                source_index[0] as usize,
                                                source_index[1] as usize,
                                                source_index[2] as usize,
                                                input_channel + channels * batch,
                                            ],
                                            image_shape,
                                        )];
                                    }
                                }
                            }
                        }
                    }
                    *value = sum;
                }
            }
            59 => {
                let image = *info
                    .sources
                    .get(1)
                    .ok_or_else(|| invalid("CPU transposed convolution input missing"))?;
                let image_shape = self.info(image)?.shape;
                let image = self.values(image)?;
                let stride = integer(p, 0) as usize;
                for batch in 0..shape[3] {
                    for channel in 0..shape[2] {
                        for iy in 0..image_shape[1] {
                            for ix in 0..image_shape[0] {
                                for ky in 0..source_shape[1] {
                                    for kx in 0..source_shape[0] {
                                        let mut dot = 0.0_f64;
                                        for input_channel in 0..image_shape[2] {
                                            dot += f64::from(
                                                input[linear(
                                                    [kx, ky, channel, input_channel],
                                                    source_shape,
                                                )],
                                            ) * f64::from(
                                                image[linear(
                                                    [ix, iy, input_channel, batch],
                                                    image_shape,
                                                )],
                                            );
                                        }
                                        output[linear(
                                            [ix * stride + kx, iy * stride + ky, channel, batch],
                                            shape,
                                        )] += dot as f32;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            60 | 61 => {
                let two_dimensional = info.operation == 61;
                let average = integer(p, 0) == 1;
                let kernel = [
                    integer(p, 1),
                    if two_dimensional { integer(p, 2) } else { 1 },
                ];
                let stride = [
                    integer(p, if two_dimensional { 3 } else { 2 }),
                    if two_dimensional { integer(p, 4) } else { 1 },
                ];
                let padding = [
                    integer(p, if two_dimensional { 5 } else { 3 }),
                    if two_dimensional { integer(p, 6) } else { 0 },
                ];
                for (index, value) in output.iter_mut().enumerate() {
                    let coord = coordinates(index, shape);
                    let mut sum = if average { 0.0 } else { -f32::MAX };
                    let mut count = 0;
                    for ky in 0..kernel[1] {
                        for kx in 0..kernel[0] {
                            let x =
                                coord[0] as i64 * i64::from(stride[0]) + i64::from(kx - padding[0]);
                            let y =
                                coord[1] as i64 * i64::from(stride[1]) + i64::from(ky - padding[1]);
                            if x >= 0
                                && x < source_shape[0] as i64
                                && y >= 0
                                && y < source_shape[1] as i64
                            {
                                let sample = input[linear(
                                    [x as usize, y as usize, coord[2], coord[3]],
                                    source_shape,
                                )];
                                if average {
                                    sum += sample;
                                } else {
                                    sum = sum.max(sample);
                                }
                                count += 1;
                            }
                        }
                    }
                    if average {
                        let denominator = if two_dimensional {
                            kernel[0] * kernel[1]
                        } else {
                            count
                        };
                        sum = if denominator > 0 {
                            sum / denominator as f32
                        } else {
                            0.0
                        };
                    }
                    *value = sum;
                }
            }
            63 => {
                let flags = integer(p, 0);
                let mode = flags & 255;
                let aligned = flags & 256 != 0;
                let antialias = flags & 512 != 0;
                if !(0..=2).contains(&mode) || antialias && mode != 1 {
                    return Err(invalid("unsupported CPU interpolation mode"));
                }
                let mut scale: [f32; 4] =
                    std::array::from_fn(|axis| shape[axis] as f32 / source_shape[axis] as f32);
                let pixel = if aligned { 0.0 } else { 0.5 };
                if aligned {
                    for axis in 0..2 {
                        if shape[axis] > 1 && source_shape[axis] > 1 {
                            scale[axis] =
                                (shape[axis] - 1) as f32 / (source_shape[axis] - 1) as f32;
                        }
                    }
                }
                for (index, value) in output.iter_mut().enumerate() {
                    let coord = coordinates(index, shape);
                    let channel = (coord[2] as f32 / scale[2]) as usize;
                    let batch = (coord[3] as f32 / scale[3]) as usize;
                    let sample = |x: i64, y: i64| {
                        input[linear(
                            [
                                x.clamp(0, source_shape[0] as i64 - 1) as usize,
                                y.clamp(0, source_shape[1] as i64 - 1) as usize,
                                channel,
                                batch,
                            ],
                            source_shape,
                        )]
                    };
                    if mode == 0 {
                        *value = sample(
                            (coord[0] as f32 / scale[0]) as i64,
                            (coord[1] as f32 / scale[1]) as i64,
                        );
                        continue;
                    }
                    let x = (coord[0] as f32 + pixel) / scale[0] - pixel;
                    let y = (coord[1] as f32 + pixel) / scale[1] - pixel;
                    if antialias {
                        let center_x = x + pixel;
                        let center_y = y + pixel;
                        let support_x = (1.0 / scale[0]).max(1.0);
                        let support_y = (1.0 / scale[1]).max(1.0);
                        let first_x = ((center_x - support_x + pixel) as i64).max(0);
                        let last_x =
                            ((center_x + support_x + pixel) as i64).min(source_shape[0] as i64);
                        let first_y = ((center_y - support_y + pixel) as i64).max(0);
                        let last_y =
                            ((center_y + support_y + pixel) as i64).min(source_shape[1] as i64);
                        let mut sum = 0.0;
                        let mut weights = 0.0;
                        for sy in first_y..last_y {
                            let wy =
                                (1.0 - ((sy as f32 - center_y + pixel) / support_y).abs()).max(0.0);
                            for sx in first_x..last_x {
                                let wx = (1.0 - ((sx as f32 - center_x + pixel) / support_x).abs())
                                    .max(0.0);
                                let weight = wx * wy;
                                if weight > 0.0 {
                                    sum += sample(sx, sy) * weight;
                                    weights += weight;
                                }
                            }
                        }
                        *value = if weights > 0.0 { sum / weights } else { sum };
                    } else if mode == 1 {
                        let x0 = (x.floor() as i64).clamp(0, source_shape[0] as i64 - 1);
                        let x1 = (x.floor() as i64 + 1).clamp(0, source_shape[0] as i64 - 1);
                        let y0 = (y.floor() as i64).clamp(0, source_shape[1] as i64 - 1);
                        let y1 = (y.floor() as i64 + 1).clamp(0, source_shape[1] as i64 - 1);
                        let dx = (x - x0 as f32).clamp(0.0, 1.0);
                        let dy = (y - y0 as f32).clamp(0.0, 1.0);
                        *value = sample(x0, y0) * (1.0 - dx) * (1.0 - dy)
                            + sample(x1, y0) * dx * (1.0 - dy)
                            + sample(x0, y1) * (1.0 - dx) * dy
                            + sample(x1, y1) * dx * dy;
                    } else {
                        let x0 = x.floor() as i64;
                        let y0 = y.floor() as i64;
                        let dx = x - x0 as f32;
                        let dy = y - y0 as f32;
                        *value = cubic(
                            std::array::from_fn(|row| {
                                cubic(
                                    std::array::from_fn(|column| {
                                        sample(x0 + column as i64 - 1, y0 + row as i64 - 1)
                                    }),
                                    dx,
                                )
                            }),
                            dy,
                        );
                    }
                }
            }
            65 => {
                let left = i64::from(integer(p, 0));
                for (index, value) in output.iter_mut().enumerate() {
                    let mut coord = coordinates(index, shape);
                    let position = coord[0] as i64 - left;
                    coord[0] = if position < 0 {
                        -position as usize
                    } else if position >= source_shape[0] as i64 {
                        (2 * source_shape[0] as i64 - 2 - position) as usize
                    } else {
                        position as usize
                    };
                    *value = input[linear(coord, source_shape)];
                }
            }
            _ => return Err(invalid("unsupported CPU image operation")),
        }
        Ok(output)
    }
}
