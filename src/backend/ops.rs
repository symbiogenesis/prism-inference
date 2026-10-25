//! Numerical kernels for owned CPU graphs.

use super::{CpuBackend, TensorId, coordinates, linear};
use crate::{Result, invalid};

pub(super) const fn supports(operation: i32) -> bool {
    matches!(operation, 0..=74 | 76..=91 | 96..=101)
}

pub(super) const fn packed_add1(kind: i32) -> bool {
    // Only advertise formats with both a decoder and a native CPU encoder.
    // Q8_1 has no native decoder; Q8_K is absent from pinned ADD1 dispatch.
    matches!(kind, 2 | 3 | 6..=8 | 10..=14 | 20 | 23 | 34 | 35 | 39..=42 | 142 | 143)
}

fn integer(parameters: &[u8; 64], index: usize) -> i32 {
    i32::from_le_bytes(std::array::from_fn(|offset| parameters[index * 4 + offset]))
}

fn float(parameters: &[u8; 64], index: usize) -> f32 {
    f32::from_bits(integer(parameters, index) as u32)
}

fn erf(x: f32) -> f32 {
    // Abramowitz and Stegun 7.1.26; maximum absolute approximation error 1.5e-7.
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let polynomial = ((((1.061_405_4 * t - 1.453_152_1) * t + 1.421_413_8) * t - 0.284_496_72) * t
        + 0.254_829_6)
        * t;
    sign * (1.0 - polynomial * (-x * x).exp())
}

fn unary(op: i32, x: f32, p: &[u8; 64]) -> Result<f32> {
    Ok(match op {
        0 => x.abs(),
        1 => {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        2 => -x,
        3 => {
            if x > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        4 => x.tanh(),
        5 => {
            if x > 0.0 {
                x
            } else {
                x.exp_m1()
            }
        }
        6 => x.max(0.0),
        7 => 1.0 / (1.0 + (-x).exp()),
        8 => {
            0.5 * x
                * (1.0
                    + (std::f32::consts::FRAC_2_SQRT_PI
                        * std::f32::consts::FRAC_1_SQRT_2
                        * x
                        * (1.0 + 0.044_715 * x * x))
                        .tanh())
        }
        9 => x / (1.0 + (-1.702 * x).exp()),
        10 => x / (1.0 + (-x).exp()),
        11 => x * (x + 3.0).clamp(0.0, 6.0) / 6.0,
        12 => (x + 3.0).clamp(0.0, 6.0) / 6.0,
        13 => x.exp(),
        14 => x.exp_m1(),
        15 => {
            if x > 20.0 {
                x
            } else {
                x.exp().ln_1p()
            }
        }
        16 => 0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2)),
        17 => {
            let an = float(p, 1);
            let ap = float(p, 2);
            let beta = float(p, 3);
            let eps = float(p, 4);
            if x > 0.0 {
                ap * x * x + beta * x
            } else {
                an * (x.min(eps).exp_m1() - x) + beta * x
            }
        }
        18 => x.floor(),
        19 => x.ceil(),
        20 => x.round(),
        21 => x.trunc(),
        _ => return Err(invalid(format!("unsupported CPU unary operation {op}"))),
    })
}

impl CpuBackend {
    #[allow(
        clippy::too_many_lines,
        clippy::many_single_char_names,
        reason = "Pinned numerical operation dispatch retains conventional operand and matrix-axis names."
    )]
    pub(super) fn evaluate(&mut self, tensor: TensorId) -> Result<()> {
        let info = self.info(tensor)?.clone();
        let op = info.operation;
        // The pinned CPU dispatcher skips empty nodes before reading operands
        // or interpreting kernel parameters (some derive divisors from shape).
        if matches!(op, 0 | 36..=39) || info.elements() == 0 {
            return Ok(());
        }
        let source = |index: usize| {
            info.sources
                .get(index)
                .copied()
                .ok_or_else(|| invalid("CPU operation input is missing"))
        };
        let p = &info.parameters;
        let shape = info.shape;
        let n = info.elements();
        if matches!(op, 1 | 34 | 35)
            && info.sources.first().is_some_and(|&source| {
                self.info(source)
                    .is_ok_and(|source| source.kind == info.kind)
            })
        {
            return self.copy_storage(tensor, info.sources[0]);
        }
        if op == 80 {
            return self.relative_positions(tensor);
        }
        let a_id = source(0).ok();
        let a = if let Some(id) = a_id.filter(|_| !matches!(op, 29 | 30)) {
            self.values(id)?
        } else {
            Vec::new()
        };
        let a_shape = if let Some(id) = a_id {
            self.info(id)?.shape
        } else {
            [1; 4]
        };
        let mut out = Vec::new();
        out.try_reserve_exact(n)?;
        out.resize(n, 0.0);
        match op {
            1 | 34 | 35 => {
                if a.len() != n {
                    return Err(invalid("CPU copy element count differs"));
                }
                out.copy_from_slice(&a);
            }
            2 | 6..=8 => {
                let b_id = source(1)?;
                let b = self.values(b_id)?;
                let b_shape = self.info(b_id)?.shape;
                if a.len() != n
                    || b_shape
                        .iter()
                        .zip(shape)
                        .any(|(&b, a)| b == 0 || !a.is_multiple_of(b))
                {
                    return Err(invalid("CPU binary broadcast shapes differ"));
                }
                for (index, value) in out.iter_mut().enumerate() {
                    let coordinate = coordinates(index, shape);
                    let b_index = linear(
                        std::array::from_fn(|axis| coordinate[axis] % b_shape[axis]),
                        b_shape,
                    );
                    *value = match op {
                        2 => a[index] + b[b_index],
                        6 => a[index] - b[b_index],
                        7 => a[index] * b[b_index],
                        _ => a[index] / b[b_index],
                    };
                }
            }
            4 => {
                let scalar = self.values(source(1)?)?;
                if scalar.len() != 1 || a.len() != n {
                    return Err(invalid("CPU scalar add shapes differ"));
                }
                for (&input, output) in a.iter().zip(&mut out) {
                    *output = input + scalar[0];
                }
            }
            78 | 79 | 81 => out = self.spatial_values(&info, &a)?,
            3 => {
                let b_id = source(1)?;
                let b = self.values(b_id)?;
                let b_shape = self.info(b_id)?.shape;
                let ids_id = source(2)?;
                let ids = self.integer_values(ids_id)?;
                let ids_shape = self.info(ids_id)?.shape;
                for (index, value) in out.iter_mut().enumerate() {
                    let coordinate = coordinates(index, shape);
                    let row = usize::try_from(
                        ids[linear([coordinate[1], coordinate[2], 0, 0], ids_shape)],
                    )?;
                    if row >= b_shape[1] {
                        return Err(invalid("CPU indexed add expert exceeds row count"));
                    }
                    *value = a[index] + b[coordinate[0] + b_shape[0] * row];
                }
            }
            9..=13 | 32 | 50 | 71 | 73 | 91 => {
                if op != 73 && a.len() != n {
                    return Err(invalid("CPU unary shape differs"));
                }
                for (index, value) in out.iter_mut().enumerate() {
                    let x = a.get(index).copied().unwrap_or(0.0);
                    *value = match op {
                        9 => x * x,
                        10 => x.sqrt(),
                        11 => x.ln(),
                        12 => x.sin(),
                        13 => x.cos(),
                        32 => x * float(p, 0) + float(p, 1),
                        50 => x.clamp(float(p, 0), float(p, 1)),
                        71 => {
                            if x > 0.0 {
                                x
                            } else {
                                x * float(p, 0)
                            }
                        }
                        73 => float(p, 0),
                        _ => unary(integer(p, 0), x, p)?,
                    };
                }
            }
            14 => out[0] = a.iter().map(|&x| f64::from(x)).sum::<f64>() as f32,
            15 | 17 | 18 => {
                for (row, value) in out.iter_mut().enumerate() {
                    let values = &a[row * a_shape[0]..(row + 1) * a_shape[0]];
                    *value = if op == 18 {
                        let mut best = 0;
                        for index in 1..values.len() {
                            if values[index] > values[best] {
                                best = index;
                            }
                        }
                        best as f32
                    } else {
                        let sum = values.iter().map(|&x| f64::from(x)).sum::<f64>();
                        (if op == 17 {
                            sum / values.len() as f64
                        } else {
                            sum
                        }) as f32
                    };
                }
            }
            16 => {
                for (input, output) in a
                    .chunks_exact(a_shape[0])
                    .zip(out.chunks_exact_mut(shape[0]))
                {
                    let mut sum = 0.0;
                    for (&x, y) in input.iter().zip(output) {
                        sum += x;
                        *y = sum;
                    }
                }
            }
            19 => {
                let other = source(1)?;
                if (24..=27).contains(&self.info(source(0)?)?.kind)
                    && (24..=27).contains(&self.info(other)?.kind)
                {
                    let left = self.integer_values(source(0)?)?;
                    let right = self.integer_values(other)?;
                    if left.len() != right.len() {
                        return Err(invalid("CPU equal count shapes differ"));
                    }
                    let equal =
                        i64::try_from(left.iter().zip(&right).filter(|(x, y)| x == y).count())?;
                    return self.set(tensor, 0, &equal.to_le_bytes());
                }
                let b = self.values(other)?;
                if a.len() != b.len() {
                    return Err(invalid("CPU equal count shapes differ"));
                }
                out[0] = a.iter().zip(b).filter(|(x, y)| **x == *y).count() as f32;
            }
            20 | 21 => {
                let (large, small) = if op == 20 {
                    (shape, a_shape)
                } else {
                    (a_shape, shape)
                };
                if small
                    .iter()
                    .zip(large)
                    .any(|(&small, large)| small == 0 || !large.is_multiple_of(small))
                {
                    return Err(invalid("CPU repeat shapes differ"));
                }
                if op == 20 {
                    for (index, value) in out.iter_mut().enumerate() {
                        let coord = coordinates(index, shape);
                        *value = a[linear(
                            std::array::from_fn(|axis| coord[axis] % a_shape[axis]),
                            a_shape,
                        )];
                    }
                } else {
                    for (index, &value) in a.iter().enumerate() {
                        let coord = coordinates(index, a_shape);
                        out[linear(
                            std::array::from_fn(|axis| coord[axis] % shape[axis]),
                            shape,
                        )] += value;
                    }
                }
            }
            22 => {
                let axis = usize::try_from(integer(p, 0))?;
                let b_id = source(1)?;
                let b_shape = self.info(b_id)?.shape;
                let b = self.values(b_id)?;
                for (index, value) in out.iter_mut().enumerate() {
                    let mut coord = coordinates(index, shape);
                    *value = if coord[axis] < a_shape[axis] {
                        a[linear(coord, a_shape)]
                    } else {
                        coord[axis] -= a_shape[axis];
                        b[linear(coord, b_shape)]
                    };
                }
            }
            23 => {
                let b = self.values(source(1)?)?;
                if a.len() != n || b.len() != n {
                    return Err(invalid("CPU silu backward shapes differ"));
                }
                for ((&dy, &x), value) in a.iter().zip(&b).zip(&mut out) {
                    let sigmoid = 1.0 / (1.0 + (-x).exp());
                    *value = dy * sigmoid * (1.0 + x * (1.0 - sigmoid));
                }
            }
            24 | 25 | 28 => {
                let eps = float(p, 0);
                for (input, output) in a
                    .chunks_exact(a_shape[0])
                    .zip(out.chunks_exact_mut(shape[0]))
                {
                    let mean = if op == 24 {
                        (input.iter().map(|&x| f64::from(x)).sum::<f64>() / input.len() as f64)
                            as f32
                    } else {
                        0.0
                    };
                    let sum = input
                        .iter()
                        .map(|&x| {
                            let delta = f64::from(x - mean);
                            delta * delta
                        })
                        .sum::<f64>();
                    let scale = if op == 28 {
                        1.0 / (sum as f32).sqrt().max(eps)
                    } else {
                        1.0 / ((sum / input.len() as f64) as f32 + eps).sqrt()
                    };
                    for (&x, y) in input.iter().zip(output) {
                        *y = (x - mean) * scale;
                    }
                }
            }
            26 => {
                let b = self.values(source(1)?)?;
                if a.len() != n || b.len() != n {
                    return Err(invalid("CPU rms backward shapes differ"));
                }
                for ((gradient, input), output) in a
                    .chunks_exact(a_shape[0])
                    .zip(b.chunks_exact(a_shape[0]))
                    .zip(out.chunks_exact_mut(shape[0]))
                {
                    let square = input
                        .iter()
                        .map(|&x| f64::from(x) * f64::from(x))
                        .sum::<f64>()
                        / input.len() as f64;
                    let mean_product = input
                        .iter()
                        .zip(gradient)
                        .map(|(&x, &dy)| f64::from(x) * f64::from(dy))
                        .sum::<f64>()
                        / input.len() as f64;
                    let scale = 1.0 / (square + f64::from(float(p, 0))).sqrt();
                    for ((&x, &dy), value) in input.iter().zip(gradient).zip(output) {
                        *value = (f64::from(dy) * scale
                            - f64::from(x) * mean_product * scale * scale * scale)
                            as f32;
                    }
                }
            }
            27 => {
                let groups = usize::try_from(integer(p, 0))?;
                let group_channels = shape[2].div_ceil(groups);
                for batch in 0..shape[3] {
                    for group in 0..groups {
                        let first = group * group_channels * shape[0] * shape[1]
                            + batch * shape[0] * shape[1] * shape[2];
                        if group * group_channels >= shape[2] {
                            continue;
                        }
                        let count = (shape[2] - group * group_channels).min(group_channels)
                            * shape[0]
                            * shape[1];
                        if count == 0 {
                            continue;
                        }
                        let input = &a[first..first + count];
                        let mean = (input.iter().map(|&x| f64::from(x)).sum::<f64>() / count as f64)
                            as f32;
                        let variance = input
                            .iter()
                            .map(|&x| {
                                let d = f64::from(x - mean);
                                d * d
                            })
                            .sum::<f64>()
                            / count as f64;
                        let scale = 1.0 / (variance as f32 + float(p, 1)).sqrt();
                        for (offset, &x) in input.iter().enumerate() {
                            out[first + offset] = (x - mean) * scale;
                        }
                    }
                }
            }
            29 => {
                let b_id = source(1)?;
                let b_shape = self.info(b_id)?.shape;
                let b = self.values(b_id)?;
                if a_shape[0] != b_shape[0]
                    || !b_shape[2].is_multiple_of(a_shape[2])
                    || !b_shape[3].is_multiple_of(a_shape[3])
                {
                    return Err(invalid("CPU matmul shapes differ"));
                }
                for batch3 in 0..shape[3] {
                    for batch2 in 0..shape[2] {
                        let a2 = batch2 / (b_shape[2] / a_shape[2]);
                        let a3 = batch3 / (b_shape[3] / a_shape[3]);
                        let matrix = self.matrix(source(0)?, a2, a3)?;
                        for column in 0..shape[1] {
                            let start = linear([0, column, batch2, batch3], b_shape);
                            let values =
                                crate::kernels::matvec(&matrix, &b[start..start + b_shape[0]])?;
                            let output_start = linear([0, column, batch2, batch3], shape);
                            out[output_start..output_start + shape[0]].copy_from_slice(&values);
                        }
                    }
                }
            }
            30 => {
                let b_id = source(1)?;
                let b = self.values(b_id)?;
                let b_shape = self.info(b_id)?.shape;
                let ids_id = source(2)?;
                let ids = self.integer_values(ids_id)?;
                let ids_shape = self.info(ids_id)?.shape;
                for token in 0..shape[2] {
                    for slot in 0..shape[1] {
                        let expert = usize::try_from(ids[linear([slot, token, 0, 0], ids_shape)])?;
                        if expert >= a_shape[2] {
                            return Err(invalid("CPU matmul expert exceeds expert count"));
                        }
                        let matrix = self.matrix(source(0)?, expert, 0)?;
                        let start = linear([0, slot % b_shape[1], token, 0], b_shape);
                        let values =
                            crate::kernels::matvec(&matrix, &b[start..start + b_shape[0]])?;
                        let output_start = linear([0, slot, token, 0], shape);
                        out[output_start..output_start + shape[0]].copy_from_slice(&values);
                    }
                }
            }
            31 => {
                let b_id = source(1)?;
                let b_shape = self.info(b_id)?.shape;
                let b = self.values(b_id)?;
                for (index, value) in out.iter_mut().enumerate() {
                    let [m, n, b2, b3] = coordinates(index, shape);
                    let mut sum = 0.0_f64;
                    for k in 0..a_shape[1] {
                        sum +=
                            f64::from(a[linear([m, k, b2 % a_shape[2], b3 % a_shape[3]], a_shape)])
                                * f64::from(
                                    b[linear([n, k, b2 % b_shape[2], b3 % b_shape[3]], b_shape)],
                                );
                    }
                    *value = sum as f32;
                }
            }
            5 | 33 => {
                out.copy_from_slice(&a);
                let b_id = source(1)?;
                let b_shape = self.info(b_id)?.shape;
                let b = self.values(b_id)?;
                let strides = [
                    4,
                    integer(p, 0) as usize,
                    integer(p, 1) as usize,
                    integer(p, 2) as usize,
                ];
                for (index, &x) in b.iter().enumerate() {
                    let coord = coordinates(index, b_shape);
                    let offset = integer(p, 3) as usize
                        + coord
                            .iter()
                            .zip(strides)
                            .map(|(&x, stride)| x * stride)
                            .sum::<usize>();
                    let destination = out
                        .get_mut(offset / 4)
                        .ok_or_else(|| invalid("CPU set destination exceeds extent"))?;
                    if op == 5 {
                        *destination += x;
                    } else {
                        *destination = x;
                    }
                }
            }
            40 => {
                let ids_id = source(1)?;
                let ids = self.integer_values(ids_id)?;
                let ids_shape = self.info(ids_id)?.shape;
                if info.kind == 26 {
                    let source_info = self.info(source(0)?)?.clone();
                    let mut bytes = Vec::new();
                    bytes.try_reserve_exact(n * 4)?;
                    for index in 0..n {
                        let coord = coordinates(index, shape);
                        let row = usize::try_from(
                            ids[linear([coord[1], coord[2], coord[3], 0], ids_shape)],
                        )?;
                        if row >= a_shape[1] {
                            return Err(invalid("CPU gather index exceeds row count"));
                        }
                        let coordinate = [coord[0], row, coord[2], coord[3]];
                        let offset: usize = coordinate
                            .iter()
                            .zip(source_info.strides)
                            .map(|(&index, stride)| index * stride)
                            .sum();
                        bytes.extend_from_slice(self.get(source(0)?, offset, 4)?);
                    }
                    return self.set(tensor, 0, &bytes);
                }
                for (index, value) in out.iter_mut().enumerate() {
                    let coord = coordinates(index, shape);
                    let row =
                        usize::try_from(ids[linear([coord[1], coord[2], coord[3], 0], ids_shape)])?;
                    if row >= a_shape[1] {
                        return Err(invalid("CPU gather index exceeds row count"));
                    }
                    *value = a[linear([coord[0], row, coord[2], coord[3]], a_shape)];
                }
            }
            41 => {
                let ids = self.integer_values(source(1)?)?;
                for (row, &id) in ids.iter().enumerate() {
                    let id = usize::try_from(id)?;
                    if id >= shape[1] {
                        return Err(invalid("CPU scatter index exceeds row count"));
                    }
                    for column in 0..shape[0] {
                        out[column + shape[0] * id] += a[column + shape[0] * row];
                    }
                }
            }
            42 => {
                let b = self.values(source(2)?)?;
                let ids_id = source(1)?;
                let ids = self.integer_values(ids_id)?;
                let ids_shape = self.info(ids_id)?.shape;
                out.copy_from_slice(&b);
                for row in 0..a.len() / a_shape[0] {
                    let coord = coordinates(row, [a_shape[1], a_shape[2], a_shape[3], 1]);
                    let id = usize::try_from(
                        ids[linear(
                            [
                                coord[0],
                                coord[1] % ids_shape[1],
                                coord[2] % ids_shape[2],
                                0,
                            ],
                            ids_shape,
                        )],
                    )?;
                    if id >= shape[1] {
                        return Err(invalid("CPU row assignment exceeds extent"));
                    }
                    for column in 0..shape[0] {
                        out[linear([column, id, coord[1], coord[2]], shape)] =
                            a[column + a_shape[0] * row];
                    }
                }
            }
            43 => {
                for (index, value) in out.iter_mut().enumerate() {
                    let coord = coordinates(index, shape);
                    if coord[0] == coord[1] {
                        *value = a[linear([coord[0], 0, coord[2], coord[3]], a_shape)];
                    }
                }
            }
            44 | 45 => {
                out.copy_from_slice(&a);
                for (index, value) in out.iter_mut().enumerate() {
                    let coord = coordinates(index, shape);
                    if coord[0] as i64 > coord[1] as i64 + i64::from(integer(p, 0)) {
                        *value = if op == 44 { f32::NEG_INFINITY } else { 0.0 };
                    }
                }
            }
            46 => {
                let mask = info.source_slots[1].map(|id| self.values(id)).transpose()?;
                let mask_shape = info.source_slots[1]
                    .map(|id| self.info(id).map(|info| info.shape))
                    .transpose()?;
                let sinks = info.source_slots[2].map(|id| self.values(id)).transpose()?;
                let scale = float(p, 0);
                let max_bias = float(p, 1);
                let heads = shape[2];
                let power_heads = 1_usize << heads.ilog2();
                let m0 = 2.0_f32.powf(-max_bias / power_heads as f32);
                let m1 = 2.0_f32.powf(-(max_bias / 2.0) / power_heads as f32);
                for (row, output) in out.chunks_exact_mut(shape[0]).enumerate() {
                    let coord = coordinates(row, [shape[1], shape[2], shape[3], 1]);
                    let head = coord[1];
                    let slope = if max_bias <= 0.0 {
                        1.0
                    } else if head < power_heads {
                        m0.powi((head + 1) as i32)
                    } else {
                        m1.powi((2 * (head - power_heads) + 1) as i32)
                    };
                    for (column, value) in output.iter_mut().enumerate() {
                        let bias = if let (Some(mask), Some(ms)) = (&mask, mask_shape) {
                            slope
                                * mask[linear(
                                    [
                                        column % ms[0],
                                        coord[0] % ms[1],
                                        coord[1] % ms[2],
                                        coord[2] % ms[3],
                                    ],
                                    ms,
                                )]
                        } else {
                            0.0
                        };
                        *value = a[row * shape[0] + column] * scale + bias;
                    }
                    let sink = sinks.as_ref().map(|s| s[head % s.len()]);
                    let maximum = output
                        .iter()
                        .copied()
                        .chain(sink)
                        .fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0_f64;
                    for value in output.iter_mut() {
                        *value = (*value - maximum).exp();
                        sum += f64::from(*value);
                    }
                    if let Some(sink) = sink {
                        sum += f64::from((sink - maximum).exp());
                    }
                    for value in output {
                        *value *= (1.0 / sum) as f32;
                    }
                }
            }
            47 => {
                let b = self.values(source(1)?)?;
                for ((gradient, y), output) in a
                    .chunks_exact(shape[0])
                    .zip(b.chunks_exact(shape[0]))
                    .zip(out.chunks_exact_mut(shape[0]))
                {
                    let dot = gradient
                        .iter()
                        .zip(y)
                        .map(|(&x, &y)| f64::from(x) * f64::from(y))
                        .sum::<f64>() as f32;
                    for ((&dy, &y), value) in gradient.iter().zip(y).zip(output) {
                        *value = (dy - dot) * y * float(p, 0);
                    }
                }
            }
            48 | 49 => out = self.rope_values(&info, &a)?,
            51 => {
                let b_id = source(1)?;
                let b_shape = self.info(b_id)?.shape;
                let b = self.values(b_id)?;
                let stride = integer(p, 0) as usize;
                for channel in 0..a_shape[1] {
                    for input_time in 0..b_shape[0] {
                        for kernel in 0..a_shape[0] {
                            let mut dot = 0.0_f64;
                            for input_channel in 0..a_shape[2] {
                                dot += f64::from(
                                    a[linear([kernel, channel, input_channel, 0], a_shape)],
                                ) * f64::from(
                                    b[linear([input_time, input_channel, 0, 0], b_shape)],
                                );
                            }
                            out[input_time * stride + kernel + channel * shape[0]] += dot as f32;
                        }
                    }
                }
            }
            55 => {
                let stride = i64::from(integer(p, 0));
                let channels = integer(p, 1) as usize;
                let padding = i64::from(integer(p, 2));
                let kernel_width = a_shape[0] / channels;
                for channel in 0..channels {
                    for time in 0..shape[0] {
                        let absolute = time as i64 + padding;
                        let first = ((absolute - kernel_width as i64 + stride) / stride).max(0);
                        let last = (absolute / stride).min(a_shape[1] as i64 - 1);
                        let mut sum = 0.0_f32;
                        for input in first..=last {
                            let kernel = absolute - input * stride;
                            if kernel >= 0 && kernel < kernel_width as i64 {
                                sum += a[channel * kernel_width
                                    + kernel as usize
                                    + input as usize * a_shape[0]];
                            }
                        }
                        out[time + channel * shape[0]] = sum;
                    }
                }
            }
            53 | 62 => out = self.backward_image_values(&info, &a)?,
            52 | 54 | 56..=61 | 63 | 65 => out = self.image_values(&info, &a)?,
            64 => {
                let circular = integer(p, 8) != 0;
                for (index, value) in out.iter_mut().enumerate() {
                    let coordinate = coordinates(index, shape);
                    let mut src = [0; 4];
                    let mut valid = true;
                    for axis in 0..4 {
                        let shifted = coordinate[axis] as i64 - i64::from(integer(p, 2 * axis));
                        if circular {
                            src[axis] = shifted.rem_euclid(a_shape[axis] as i64) as usize;
                        } else if shifted < 0 || shifted >= a_shape[axis] as i64 {
                            valid = false;
                        } else {
                            src[axis] = shifted as usize;
                        }
                    }
                    if valid {
                        *value = a[linear(src, a_shape)];
                    }
                }
            }
            66 => {
                for (index, value) in out.iter_mut().enumerate() {
                    let coordinate = coordinates(index, shape);
                    let src = std::array::from_fn(|axis| {
                        (coordinate[axis] as i64 - i64::from(integer(p, axis)))
                            .rem_euclid(shape[axis] as i64) as usize
                    });
                    *value = a[linear(src, shape)];
                }
            }
            67 => {
                for (index, value) in out.iter_mut().enumerate() {
                    *value = float(p, 0) + index as f32 * float(p, 2);
                }
            }
            68 => {
                let dim = integer(p, 0) as usize;
                let half = dim / 2;
                for (row, &time) in a.iter().enumerate() {
                    for index in 0..half {
                        let angle = time
                            * (-(integer(p, 1) as f32).ln() * index as f32 / half as f32).exp();
                        out[row * shape[0] + index] = angle.cos();
                        out[row * shape[0] + index + half] = angle.sin();
                    }
                }
            }
            69 | 70 => {
                for (row, output) in out.chunks_exact_mut(shape[0]).enumerate() {
                    let values = &a[row * a_shape[0]..(row + 1) * a_shape[0]];
                    let mut indices: Vec<_> = (0..a_shape[0]).collect();
                    let descending = integer(p, 0) != 0 || op == 70;
                    indices.sort_by(|&x, &y| {
                        let order = values[x]
                            .partial_cmp(&values[y])
                            .unwrap_or(std::cmp::Ordering::Equal);
                        if descending { order.reverse() } else { order }
                    });
                    for (value, index) in output.iter_mut().zip(indices) {
                        *value = index as f32;
                    }
                }
            }
            72 => {
                for (index, value) in out.iter_mut().enumerate() {
                    let c = coordinates(index, shape);
                    let keep = match integer(p, 0) {
                        0 => c[0] >= c[1],
                        1 => c[0] > c[1],
                        2 => c[0] <= c[1],
                        3 => c[0] < c[1],
                        _ => return Err(invalid("unknown CPU triangular operation")),
                    };
                    if keep {
                        *value = a[index];
                    }
                }
            }
            74 => out = self.attention_values(&info, &a)?,
            87 => out = self.indexer_values(&info, &a)?,
            77 => out = self.scan_values(&info, self.values(tensor).unwrap_or(out))?,
            82..=84 => out = self.linear_attention_values(&info)?,
            88..=90 => out = self.hyperconnection_values(&info, &a)?,
            86 | 101 => out = self.delta_values(&info, self.values(tensor).unwrap_or(out))?,
            76 => {
                let weights_id = source(1)?;
                let weights = self.values(weights_id)?;
                let weights_shape = self.info(weights_id)?.shape;
                for (index, value) in out.iter_mut().enumerate() {
                    let [channel, time, sequence, _] = coordinates(index, shape);
                    let mut sum = 0.0_f32;
                    for tap in 0..weights_shape[0] {
                        sum += a[linear([time + tap, channel, sequence, 0], a_shape)]
                            * weights[tap + weights_shape[0] * channel];
                    }
                    *value = sum;
                }
            }
            85 => {
                let b_id = source(1)?;
                let b = self.values(b_id)?;
                let width = a_shape[0];
                out.copy_from_slice(&b);
                for batch in 0..shape[2] * shape[3] {
                    for column in 0..shape[0] {
                        for step in 0..width {
                            let row = step;
                            let output_index = column + shape[0] * (row + shape[1] * batch);
                            let mut value = out[output_index];
                            for previous in 0..step {
                                let k = previous;
                                value -= a[k + a_shape[0]
                                    * (row + a_shape[1] * (batch % (a_shape[2] * a_shape[3])))]
                                    * out[column + shape[0] * (k + shape[1] * batch)];
                            }
                            value /= a[row
                                + a_shape[0]
                                    * (row + a_shape[1] * (batch % (a_shape[2] * a_shape[3])))];
                            out[output_index] = value;
                        }
                    }
                }
            }
            96 | 97 => {
                let (logits, logits_shape, targets, gradient) = if op == 97 {
                    let logits = source(1)?;
                    (
                        self.values(logits)?,
                        self.info(logits)?.shape,
                        self.values(source(2)?)?,
                        a[0],
                    )
                } else {
                    (a.clone(), a_shape, self.values(source(1)?)?, 1.0)
                };
                let rows = logits.len() / logits_shape[0];
                let mut loss = 0.0_f64;
                for (row, logits) in logits.chunks_exact(logits_shape[0]).enumerate() {
                    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let sum = logits
                        .iter()
                        .map(|&x| f64::from((x - maximum).exp()))
                        .sum::<f64>();
                    for (column, &logit) in logits.iter().enumerate() {
                        let index = row * logits_shape[0] + column;
                        let probability = f64::from((logit - maximum).exp()) / sum;
                        if op == 96 {
                            loss -=
                                f64::from(targets[index]) * (f64::from(logit - maximum) - sum.ln());
                        } else {
                            out[index] =
                                gradient * (probability as f32 - targets[index]) / rows as f32;
                        }
                    }
                }
                if op == 96 {
                    out[0] = (loss / rows as f64) as f32;
                }
            }
            100 => {
                let b = info.sources.get(1).map(|&id| self.values(id)).transpose()?;
                let glu = integer(p, 0);
                let swapped = integer(p, 1) != 0;
                for (index, value) in out.iter_mut().enumerate() {
                    let coord = coordinates(index, shape);
                    let ai = linear(coord, a_shape);
                    let (mut gate, mut up) = b
                        .as_ref()
                        .map_or_else(|| (a[ai], a[ai + shape[0]]), |b| (a[ai], b[index]));
                    if swapped {
                        std::mem::swap(&mut gate, &mut up);
                    }
                    *value = match glu {
                        0 => gate.max(0.0) * up,
                        1 => unary(8, gate, p)? * up,
                        2 => unary(10, gate, p)? * up,
                        3 => {
                            let alpha = float(p, 2);
                            let limit = float(p, 3);
                            let gate = gate.min(limit);
                            let up = up.clamp(-limit, limit);
                            gate / (1.0 + (-alpha * gate).exp()) * (up + 1.0)
                        }
                        4 => unary(16, gate, p)? * up,
                        5 => unary(9, gate, p)? * up,
                        _ => return Err(invalid("unknown CPU GLU operation")),
                    };
                }
            }
            98 | 99 => {
                let gradient = self.values(source(1)?)?;
                let params = self.values(source(if op == 98 { 4 } else { 2 })?)?;
                let alpha = params[0];
                let decay = params[if op == 98 { 4 } else { 1 }];
                let keep = 1.0 - alpha * decay;
                if op == 98 {
                    let moment_id = source(2)?;
                    let variance_id = source(3)?;
                    let mut moment = self.values(moment_id)?;
                    let mut variance = self.values(variance_id)?;
                    for (index, value) in out.iter_mut().enumerate() {
                        let grad = gradient[index];
                        moment[index] = moment[index] * params[1] + grad * (1.0 - params[1]);
                        variance[index] =
                            variance[index] * params[2] + grad * grad * (1.0 - params[2]);
                        let corrected_moment = moment[index] * params[5];
                        let corrected_variance = (variance[index] * params[6]).sqrt() + params[3];
                        *value = a[index] * keep - alpha * corrected_moment / corrected_variance;
                    }
                    self.set_values(moment_id, &moment)?;
                    self.set_values(variance_id, &variance)?;
                } else {
                    for (index, value) in out.iter_mut().enumerate() {
                        *value = a[index] * keep - alpha * gradient[index];
                    }
                }
            }
            _ => return Err(invalid(format!("unsupported CPU numerical operation {op}"))),
        }
        self.set_values(tensor, &out)
    }
}
