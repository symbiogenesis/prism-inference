//! Safe scalar `IQ2_S/IQ3` reference and importance fitting.
//!
//! Ported from pinned Prism `b87ca7225456b06b0975b8239c66b5aecf07e37c`,
//! ggml/src/ggml-quants.c (MIT; see root NOTICE). Geometric neighbour shells,
//! index tie breaking, weighted refitting and source scalar association are
//! retained. Search neighbours are cached only for the duration of one row.

#![allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::suspicious_operation_groupings,
    reason = "Pinned fitting specifies bounded integer bit fields and exact scalar multiply/add association."
)]

use std::collections::HashMap;

use bonsai_ternary::half_to_float;

use super::{finite, half_bits};
use crate::{Result, invalid};

struct Grid<'a, const N: usize> {
    points: &'a [[u8; N]],
    map: Vec<i16>,
    neighbours: HashMap<usize, Vec<usize>>,
    bits: usize,
    shells: usize,
}

impl<'a, const N: usize> Grid<'a, N> {
    fn new(points: &'a [[u8; N]], bits: usize, shells: usize) -> Result<Self> {
        let mut map = Vec::new();
        map.try_reserve_exact(1 << (bits * N))?;
        map.resize(1 << (bits * N), -1);
        for (index, point) in points.iter().enumerate() {
            let key = Self::key(&point.map(|value| (value - 1) / 2), bits);
            map[key] = i16::try_from(index)?;
        }
        Ok(Self {
            points,
            map,
            neighbours: HashMap::new(),
            bits,
            shells,
        })
    }

    fn key(levels: &[u8], bits: usize) -> usize {
        levels.iter().enumerate().fold(0, |key, (index, &level)| {
            key | (usize::from(level) << (bits * index))
        })
    }

    fn index(&self, levels: &[u8]) -> Result<usize> {
        usize::try_from(self.map[Self::key(levels, self.bits)])
            .map_err(|_| invalid("IQ fitted point is absent from native grid"))
    }

    fn candidates(&mut self, key: usize, levels: &[u8]) -> Result<&[usize]> {
        if !self.neighbours.contains_key(&key) {
            let mut distances = Vec::new();
            distances.try_reserve_exact(self.points.len())?;
            for (index, point) in self.points.iter().enumerate() {
                let distance = point.iter().zip(levels).fold(0_i32, |sum, (&q, &l)| {
                    let difference = i32::from(q) - (2 * i32::from(l) + 1);
                    sum + difference * difference
                });
                distances.push((distance, index));
            }
            // Native qsort compares squared distance and then original index.
            distances.sort_unstable();
            let mut previous = distances[0].0;
            let mut shells = 1;
            let mut neighbours = Vec::new();
            neighbours.try_reserve_exact(distances.len())?;
            for (distance, index) in distances {
                if distance > previous {
                    if shells == self.shells {
                        break;
                    }
                    previous = distance;
                    shells += 1;
                }
                neighbours.push(index);
            }
            self.neighbours.try_reserve(1)?;
            self.neighbours.insert(key, neighbours);
        }
        self.neighbours
            .get(&key)
            .map(Vec::as_slice)
            .ok_or_else(|| invalid("IQ neighbour cache entry missing"))
    }

    fn fit(
        &mut self,
        levels: &mut [u8],
        values: &[f32],
        weights: &[f32],
        scale: f32,
    ) -> Result<bool> {
        let key = Self::key(levels, self.bits);
        if self.map[key] >= 0 {
            return Ok(true);
        }
        let points = self.points;
        let candidates = self.candidates(key, levels)?;
        let mut best_distance = f32::MAX;
        let mut best_index = None;
        for &candidate in candidates {
            let mut distance = 0.0_f32;
            for ((&q, &value), &weight) in points[candidate].iter().zip(values).zip(weights) {
                let difference = scale * f32::from(q) - value;
                distance += weight * difference * difference;
            }
            if distance < best_distance {
                best_distance = distance;
                best_index = Some(candidate);
            }
        }
        let index = best_index.ok_or_else(|| invalid("IQ neighbour fit overflowed"))?;
        for (level, &point) in levels.iter_mut().zip(&points[index]) {
            *level = (point - 1) / 2;
        }
        Ok(false)
    }
}

fn nearest(value: f32) -> Result<i32> {
    if !value.is_finite() || value.abs() > 4_194_303.0 {
        return Err(invalid("IQ rounding exceeds native nearest_int range"));
    }
    Ok(((value + 12_582_912.0).to_bits() & 0x007f_ffff).cast_signed() - 0x0040_0000)
}

fn levels(values: &[f32], inverse: f32, maximum: u8, output: &mut [u8]) -> Result<()> {
    for (level, &value) in output.iter_mut().zip(values) {
        *level = nearest(0.5 * (inverse * value - 1.0))?.clamp(0, i32::from(maximum)) as u8;
    }
    Ok(())
}

fn sums(levels: &[u8], values: &[f32], weights: &[f32]) -> Result<(f32, f32)> {
    let mut sum_qx = 0.0_f32;
    let mut sum_q2 = 0.0_f32;
    for ((&level, &value), &weight) in levels.iter().zip(values).zip(weights) {
        let q = f32::from(2 * level + 1);
        sum_qx += weight * value * q;
        sum_q2 += weight * q * q;
    }
    Ok((finite(sum_qx)?, finite(sum_q2)?))
}

struct Configuration {
    kind: u32,
    bytes: usize,
    width: usize,
    maximum: u8,
    trials: i32,
    delta: f32,
    epsilon: f32,
    factor: f32,
}

impl Configuration {
    fn new(kind: u32) -> Result<Self> {
        match kind {
            18 => Ok(Self {
                kind,
                bytes: 98,
                width: 32,
                maximum: 7,
                trials: 15,
                delta: 0.2,
                epsilon: 1.0e-8,
                factor: 1.0125,
            }),
            21 => Ok(Self {
                kind,
                bytes: 110,
                width: 32,
                maximum: 7,
                trials: 9,
                delta: 0.2,
                epsilon: 0.0,
                factor: 1.033,
            }),
            22 => Ok(Self {
                kind,
                bytes: 82,
                width: 16,
                maximum: 2,
                trials: 9,
                delta: 0.1,
                epsilon: 1.0e-8,
                factor: 0.9875,
            }),
            _ => Err(invalid("unsupported IQ fitting type")),
        }
    }

    fn pack<const N: usize>(
        &self,
        grid: &Grid<'_, N>,
        group: usize,
        written_group: usize,
        levels: &[u8],
        signs: &[u8],
        output: &mut [u8],
    ) -> Result<()> {
        for (local, levels) in levels.as_chunks::<N>().0.iter().enumerate() {
            let index = grid.index(levels)?;
            let absolute = group * (self.width / N) + local;
            let written = written_group * (self.width / N) + local;
            output[2 + written] = index as u8;
            if self.kind == 21 {
                output[66 + absolute / 8] |= ((index >> 8) as u8) << (absolute % 8);
            } else if self.kind == 22 {
                output[66 + absolute / 4] |= ((index >> 8) as u8) << (2 * (absolute % 4));
            }
        }
        if self.kind == 18 {
            let word = u32::from(signs[0])
                | (u32::from(signs[1]) << 7)
                | (u32::from(signs[2]) << 14)
                | (u32::from(signs[3]) << 21);
            output[66 + group * 4..70 + group * 4].copy_from_slice(&word.to_le_bytes());
        } else {
            let offset = if self.kind == 21 { 74 } else { 34 };
            let offset = offset + written_group * (self.width / 8);
            output[offset..offset + signs.len()].copy_from_slice(signs);
        }
        Ok(())
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "Retain native sign correction, scale search, refit and byte packing in their source order."
)]
pub(super) fn quantize<const N: usize>(
    kind: u32,
    input: &[f32],
    importance: Option<&[f32]>,
    points: &[[u8; N]],
) -> Result<Vec<u8>> {
    let config = Configuration::new(kind)?;
    if input.is_empty()
        || !input.len().is_multiple_of(256)
        || input.iter().any(|value| !value.is_finite())
        || importance.is_some_and(|weights| {
            weights.len() != input.len()
                || weights
                    .iter()
                    .any(|value| !value.is_finite() || *value < 0.0)
        })
    {
        return Err(invalid(
            "IQ fitting needs complete finite blocks and nonnegative importance",
        ));
    }
    let mut grid = Grid::new(
        points,
        if kind == 22 { 2 } else { 3 },
        match kind {
            18 => 2,
            21 => 3,
            _ => 1,
        },
    )?;
    let bytes = (input.len() / 256)
        .checked_mul(config.bytes)
        .ok_or_else(|| invalid("IQ fitting byte extent overflow"))?;
    let mut output = Vec::new();
    output.try_reserve_exact(bytes)?;
    output.resize(bytes, 0);
    for (block_index, (input, output)) in input
        .as_chunks::<256>()
        .0
        .iter()
        .zip(output.chunks_exact_mut(config.bytes))
        .enumerate()
    {
        let mut sum_x2 = 0.0_f32;
        for &value in input {
            sum_x2 += value * value;
        }
        let sigma2 = finite(2.0 * sum_x2 / 256.0)?;
        let importance =
            importance.map(|weights| &weights[block_index * 256..(block_index + 1) * 256]);
        let mut scales = [0.0_f32; 16];
        let mut maximum_scale = 0.0_f32;
        let mut written_groups = 0;
        for (group, source) in input.chunks_exact(config.width).enumerate() {
            let mut values = [0.0_f32; 32];
            let mut weight = [0.0_f32; 32];
            let mut auxiliary_weight = [0.0_f32; 32];
            let mut signs = [0_u8; 4];
            for index in 0..config.width {
                let square = source[index] * source[index];
                weight[index] = finite(importance.map_or_else(
                    || {
                        if kind == 22 {
                            0.25 * sigma2 + square
                        } else {
                            square
                        }
                    },
                    |importance| {
                        importance[group * config.width + index] * (sigma2 + square).sqrt()
                    },
                ))?;
                auxiliary_weight[index] = weight[index].sqrt();
            }
            for (octet, source) in source.as_chunks::<8>().0.iter().enumerate() {
                for (lane, &value) in source.iter().enumerate() {
                    if value >= 0.0 {
                        values[octet * 8 + lane] = value;
                    } else {
                        values[octet * 8 + lane] = -value;
                        signs[octet] |= 1 << lane;
                    }
                }
                if kind == 18 && !signs[octet].count_ones().is_multiple_of(2) {
                    let mut minimum_index = 0;
                    let offset = octet * 8;
                    let mut minimum = weight[offset] * source[0] * source[0];
                    for index in 1..8 {
                        let candidate = weight[offset + index] * source[index] * source[index];
                        if candidate < minimum {
                            minimum = candidate;
                            minimum_index = index;
                        }
                    }
                    values[offset + minimum_index] = -values[offset + minimum_index];
                    signs[octet] ^= 1 << minimum_index;
                }
                if kind == 18 {
                    signs[octet] &= 127;
                }
            }
            let values = &values[..config.width];
            let weight = &weight[..config.width];
            let auxiliary_weight = &auxiliary_weight[..config.width];
            let maximum = values.iter().copied().fold(values[0], f32::max);
            if maximum == 0.0 || maximum < config.epsilon {
                continue;
            }
            let mut best = 0.0_f32;
            let mut scale = maximum / f32::from(2 * config.maximum + 1);
            let mut chosen = [0_u8; 32];
            let mut on_grid = [kind != 21; 8];
            for trial in -config.trials..=config.trials {
                let inverse =
                    (f32::from(2 * config.maximum + 1) + trial as f32 * config.delta) / maximum;
                let candidate_scale = finite(1.0 / inverse)?;
                let mut candidate = [0_u8; 32];
                let mut candidate_on_grid = [true; 8];
                levels(
                    values,
                    inverse,
                    config.maximum,
                    &mut candidate[..config.width],
                )?;
                for (index, levels) in candidate[..config.width]
                    .as_chunks_mut::<N>()
                    .0
                    .iter_mut()
                    .enumerate()
                {
                    candidate_on_grid[index] = grid.fit(
                        levels,
                        &values[index * N..(index + 1) * N],
                        &auxiliary_weight[index * N..(index + 1) * N],
                        candidate_scale,
                    )?;
                }
                let (sum_qx, sum_q2) = sums(&candidate[..config.width], values, weight)?;
                if sum_q2 > 0.0 && finite(sum_qx * sum_qx)? > finite(best * sum_q2)? {
                    scale = finite(sum_qx / sum_q2)?;
                    best = finite(scale * sum_qx)?;
                    chosen = candidate;
                    on_grid = candidate_on_grid;
                }
            }
            if on_grid[..config.width / N].iter().any(|&value| !value) && scale > 0.0 {
                let inverse = finite(1.0 / scale)?;
                for index in 0..config.width / N {
                    if kind != 21 && on_grid[index] {
                        continue;
                    }
                    let selected = &mut chosen[index * N..(index + 1) * N];
                    levels(
                        &values[index * N..(index + 1) * N],
                        inverse,
                        config.maximum,
                        selected,
                    )?;
                    grid.fit(
                        selected,
                        &values[index * N..(index + 1) * N],
                        &auxiliary_weight[index * N..(index + 1) * N],
                        scale,
                    )?;
                }
                let (sum_qx, sum_q2) = sums(&chosen[..config.width], values, weight)?;
                if sum_q2 > 0.0 {
                    scale = finite(sum_qx / sum_q2)?;
                }
            }
            if scale < 0.0 {
                scale = -scale;
                for sign in &mut signs[..config.width / 8] {
                    *sign = !*sign;
                    if kind == 18 {
                        *sign &= 127;
                    }
                }
            }
            config.pack(
                &grid,
                group,
                if kind == 21 { written_groups } else { group },
                &chosen[..config.width],
                &signs[..config.width / 8],
                output,
            )?;
            // Native IQ3_S advances its low-code/sign pointers only after a
            // nonzero group. High bits and scales still use the original group.
            // Preserve these defined bytes, including mixed zero/nonzero groups.
            written_groups += 1;
            scales[group] = scale;
            maximum_scale = maximum_scale.max(scale);
        }
        if maximum_scale == 0.0 {
            continue;
        }
        let d = maximum_scale / 31.0;
        let half = half_bits(finite(d * config.factor)?);
        finite(half_to_float(half))?;
        output[..2].copy_from_slice(&half.to_le_bytes());
        let inverse = finite(1.0 / d)?;
        for (group, &scale) in scales[..256 / config.width].iter().enumerate() {
            let level = nearest(0.5 * (inverse * scale - 1.0))?.clamp(0, 15) as u8;
            if kind == 18 {
                output[69 + group * 4] |= level << 4;
            } else {
                let offset = if kind == 21 { 106 } else { 74 };
                output[offset + group / 2] |= level << (4 * (group % 2));
            }
        }
    }
    Ok(output)
}
