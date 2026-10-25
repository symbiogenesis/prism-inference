// Copyright (c) 2023-2026 The ggml/llama.cpp authors
// SPDX-License-Identifier: MIT
//! Safe CPU normal, NEOX, multidimensional, vision and interleaved rotary kernels.

use super::{CpuBackend, TensorInfo, linear};
use crate::{Result, invalid};

fn integer(p: &[u8; 64], index: usize) -> i32 {
    i32::from_le_bytes(std::array::from_fn(|offset| p[index * 4 + offset]))
}
fn float(p: &[u8; 64], index: usize) -> f32 {
    f32::from_bits(integer(p, index) as u32)
}

impl CpuBackend {
    #[allow(
        clippy::too_many_lines,
        reason = "The pinned rotary cache and pair traversal retain their numerical order."
    )]
    pub(super) fn rope_values(&self, info: &TensorInfo, input: &[f32]) -> Result<Vec<f32>> {
        let p = &info.parameters;
        let dims = usize::try_from(integer(p, 1))?;
        let mode = integer(p, 2);
        let offset = usize::try_from(integer(p, 15))?;
        if dims == 0
            || !dims.is_multiple_of(2)
            || !offset.is_multiple_of(2)
            || offset
                .checked_add(dims)
                .is_none_or(|end| end > info.shape[0])
            || !matches!(mode, 0 | 2 | 8 | 24 | 40)
        {
            return Err(invalid("CPU rotary dimensions, offset or mode differ"));
        }
        let positions = self.values(
            *info
                .sources
                .get(1)
                .ok_or_else(|| invalid("CPU rotary positions missing"))?,
        )?;
        let factors = info.sources.get(2).map(|&id| self.values(id)).transpose()?;
        let freq_base = float(p, 5);
        let freq_scale = float(p, 6);
        let extension = float(p, 7);
        let mut magnitude = float(p, 8);
        let theta_scale = freq_base.powf(-2.0 / dims as f32);
        let sections: [usize; 4] = std::array::from_fn(|axis| integer(p, 11 + axis) as usize);
        let section_count: usize = sections.iter().sum();
        let multi = mode & 8 != 0;
        let vision = mode == 24;
        let rotated = if vision { info.shape[0] } else { dims };
        if multi && (section_count == 0 || positions.len() != 4 * info.shape[2])
            || !multi && positions.len() != info.shape[2]
            || vision && (dims * 2 != info.shape[0] || offset != 0)
        {
            return Err(invalid("CPU rotary position or section extent differs"));
        }
        if factors
            .as_ref()
            .is_some_and(|factors| factors.len() < rotated / 2)
        {
            return Err(invalid("CPU rotary frequency factors are incomplete"));
        }
        let correction = |beta: f32| {
            dims as f32 * (integer(p, 4) as f32 / (beta * 2.0 * std::f32::consts::PI)).ln()
                / (2.0 * freq_base.ln())
        };
        let low = correction(float(p, 9)).floor().max(0.0);
        let high = correction(float(p, 10)).ceil().min(dims as f32 - 1.0);
        if extension != 0.0 {
            magnitude *= 1.0 + 0.1 * (1.0 / freq_scale).ln();
        }
        let mut output = input.to_vec();
        let mut cache = vec![(0.0_f32, 0.0_f32); rotated / 2];
        for sequence in 0..info.shape[2] {
            let positions4: [f32; 4] = std::array::from_fn(|axis| {
                if multi {
                    positions[sequence + axis * info.shape[2]]
                } else {
                    positions[sequence]
                }
            });
            let mut theta = positions4;
            for (pair, entry) in cache.iter_mut().enumerate() {
                let sector = if multi { pair % section_count } else { 0 };
                if vision {
                    let mut boundary = 0;
                    for axis in 0..4 {
                        if sector == boundary {
                            theta[axis] = positions4[axis];
                        }
                        boundary += sections[axis];
                    }
                }
                let axis = if !multi {
                    0
                } else if mode == 40 {
                    if sector % 3 == 1 && sector < 3 * sections[1] {
                        1
                    } else if sector % 3 == 2 && sector < 3 * sections[2] {
                        2
                    } else if sector % 3 == 0 && sector < 3 * sections[0] {
                        0
                    } else {
                        3
                    }
                } else if sector < sections[0] {
                    0
                } else if sector < sections[0] + sections[1] {
                    1
                } else if sector < sections[0] + sections[1] + sections[2] {
                    2
                } else {
                    3
                };
                let extrapolated =
                    theta[axis] / factors.as_ref().map_or(1.0, |factors| factors[pair]);
                let interpolated = freq_scale * extrapolated;
                let angle = if extension == 0.0 {
                    interpolated
                } else {
                    let ramp = (1.0
                        - ((pair as f32 - low) / (high - low).max(0.001)).clamp(0.0, 1.0))
                        * extension;
                    interpolated * (1.0 - ramp) + extrapolated * ramp
                };
                *entry = (
                    angle.cos() * magnitude,
                    angle.sin() * magnitude * if info.operation == 49 { -1.0 } else { 1.0 },
                );
                for angle in &mut theta {
                    *angle *= theta_scale;
                }
            }
            for batch in 0..info.shape[3] {
                for head in 0..info.shape[1] {
                    let base = linear([offset, head, sequence, batch], info.shape);
                    for (pair, &(cosine, sine)) in cache.iter().enumerate() {
                        let first = if mode == 0 { 2 * pair } else { pair };
                        let second = first
                            + if mode == 0 {
                                1
                            } else if vision {
                                dims
                            } else {
                                dims / 2
                            };
                        let x = input[base + first];
                        let y = input[base + second];
                        output[base + first] = x * cosine - y * sine;
                        output[base + second] = x * sine + y * cosine;
                    }
                }
            }
        }
        Ok(output)
    }
}
