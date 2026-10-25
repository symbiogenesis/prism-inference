// Copyright (c) 2023-2026 The ggml/llama.cpp authors
// SPDX-License-Identifier: MIT
//! Symbolic derivatives of the pinned production GGML graph operations.
//!
//! Ported from ggml.c at b87ca7225456b06b0975b8239c66b5aecf07e37c.
//! Derivatives are ordinary owned CPU nodes; execution, storage and graph
//! allocation therefore use the same production paths as forward computation.

use std::collections::BTreeSet;

use super::{Argument, CpuBackend, GraphId, Observation, TensorId, TensorInfo, type_layout};
use crate::{Result, invalid};

fn integer(parameters: &[u8; 64], index: usize) -> i32 {
    i32::from_le_bytes(std::array::from_fn(|byte| parameters[index * 4 + byte]))
}

fn ignored(info: &TensorInfo, position: usize) -> bool {
    match info.operation {
        52 | 53 => position == 0,
        34 | 40 | 41 | 48 => position == 1,
        91 => position == 0 && matches!(integer(&info.parameters, 0), 1 | 3),
        _ => false,
    }
}

impl CpuBackend {
    /// Returns the symbolic derivative for a tensor in one gradient graph.
    ///
    /// A constant, disconnected parameter, or piecewise-constant source has no
    /// derivative node. The returned tensor remains owned by this backend.
    ///
    /// # Errors
    /// Rejects an unknown graph or tensor identity.
    pub fn graph_gradient(&self, graph: GraphId, tensor: TensorId) -> Result<Option<TensorId>> {
        self.info(tensor)?;
        Ok(self
            .graphs
            .get(graph.0)
            .ok_or_else(|| invalid("unknown CPU gradient graph"))?
            .gradients
            .get(&tensor)
            .copied())
    }

    /// Returns the independently retained gradient accumulator, when present.
    ///
    /// Losses always have an accumulator. Other tensors acquire one only when
    /// explicitly supplied to [`Self::build_backward`].
    ///
    /// # Errors
    /// Rejects an unknown graph or tensor identity.
    pub fn graph_gradient_accumulator(
        &self,
        graph: GraphId,
        tensor: TensorId,
    ) -> Result<Option<TensorId>> {
        self.info(tensor)?;
        Ok(self
            .graphs
            .get(graph.0)
            .ok_or_else(|| invalid("unknown CPU gradient graph"))?
            .gradient_accumulators
            .get(&tensor)
            .copied())
    }

    /// Expands a forward graph with symbolic derivatives of supported operations.
    ///
    /// The graph must have gradient storage, a parameter and a scalar F32 loss.
    /// Optional accumulators correspond to the original forward node order;
    /// each present accumulator must be F32 with the corresponding node shape.
    /// Existing forward values are untouched. Newly created metadata and graph
    /// edits are rolled back on failure, including unsupported derivatives.
    ///
    /// # Errors
    /// Rejects unknown identities, missing gradient prerequisites, unsupported
    /// derivatives/in-place operations, incompatible shapes, or graph capacity.
    pub fn build_backward(
        &mut self,
        context: u32,
        graph: GraphId,
        accumulators: Option<&[Option<TensorId>]>,
    ) -> Result<()> {
        let original = self
            .graphs
            .get(graph.0)
            .ok_or_else(|| invalid("unknown CPU backward graph"))?
            .clone();
        let tensor_count = self.tensors.len();
        let result = self.backward_expand(context, graph, accumulators);
        if result.is_err() {
            self.tensors.truncate(tensor_count);
            self.graphs[graph.0] = original;
        }
        result
    }

    fn backward_expand(
        &mut self,
        context: u32,
        graph: GraphId,
        accumulators: Option<&[Option<TensorId>]>,
    ) -> Result<()> {
        let forward = self.graphs[graph.0].nodes.clone();
        if !self.graphs[graph.0].gradients_enabled
            || forward.is_empty()
            || accumulators.is_some_and(|values| values.len() != forward.len())
            || !forward
                .iter()
                .any(|id| self.tensors[id.0].info.flags & 16 != 0)
            || !forward
                .iter()
                .any(|id| self.tensors[id.0].info.flags & 8 != 0)
        {
            return Err(invalid(
                "CPU backward graph lacks gradient storage, parameters or loss",
            ));
        }
        self.graphs[graph.0].gradients.clear();
        self.graphs[graph.0].gradient_accumulators.clear();
        let mut needed = BTreeSet::new();
        for (position, &tensor) in forward.iter().enumerate() {
            let info = self.info(tensor)?.clone();
            if info.kind == 26 {
                continue;
            }
            let mut needs = info.flags & (16 | 8) != 0;
            for (position, &source) in info.sources.iter().enumerate() {
                if !ignored(&info, position) && needed.contains(&source) {
                    if !matches!(self.info(source)?.kind, 0 | 1) {
                        return Err(invalid("CPU backward source requires F32 or F16 storage"));
                    }
                    needs = true;
                }
            }
            if !needs {
                continue;
            }
            if info.view.is_some() && !matches!(info.operation, 34 | 36..=39) {
                return Err(invalid(
                    "CPU backward does not support in-place forward operations",
                ));
            }
            if info.flags & 8 != 0 && (info.kind != 0 || info.elements() != 1) {
                return Err(invalid("CPU backward loss must be scalar F32"));
            }
            let accumulator = accumulators.and_then(|values| values[position]);
            let accumulator = if let Some(accumulator) = accumulator {
                let value = self.info(accumulator)?;
                if value.kind != 0 || value.shape != info.shape {
                    return Err(invalid("CPU gradient accumulator shape or storage differs"));
                }
                Some(accumulator)
            } else if info.flags & 8 != 0 {
                Some(self.tensor(context, 0, info.shape)?)
            } else {
                None
            };
            if let Some(accumulator) = accumulator {
                self.graphs[graph.0].gradients.insert(tensor, accumulator);
                self.graphs[graph.0]
                    .gradient_accumulators
                    .insert(tensor, accumulator);
            }
            needed.insert(tensor);
        }
        for &tensor in forward.iter().rev() {
            let Some(gradient) = self.graph_gradient(graph, tensor)? else {
                continue;
            };
            self.differentiate(context, graph, tensor, gradient, &needed)?;
        }
        Ok(())
    }

    /// Resets gradient accumulators and `AdamW` moment tensors in one graph.
    ///
    /// Scalar loss accumulators become one; other accumulators and both `AdamW`
    /// moments become zero. Parameters and ordinary forward tensors retain their
    /// values. Gradient expressions are recomputed by the next graph execution.
    ///
    /// # Errors
    /// Rejects an unknown graph, missing gradient storage, invalid loss extent,
    /// or malformed accumulator/moment storage.
    pub fn reset_graph(&mut self, graph: GraphId) -> Result<()> {
        let graph = self
            .graphs
            .get(graph.0)
            .ok_or_else(|| invalid("unknown CPU gradient graph"))?
            .clone();
        if !graph.gradients_enabled {
            return Err(invalid("CPU graph reset requires gradient storage"));
        }
        for tensor in graph.nodes {
            let info = self.info(tensor)?.clone();
            if info.operation == 98 {
                for &moment in info
                    .sources
                    .get(2..4)
                    .ok_or_else(|| invalid("CPU AdamW moments missing"))?
                {
                    self.set_values(moment, &vec![0.0; self.info(moment)?.elements()])?;
                }
            }
            if let Some(&accumulator) = graph.gradient_accumulators.get(&tensor) {
                let loss = info.flags & 8 != 0;
                let accumulator_info = self.info(accumulator)?;
                if loss && (accumulator_info.kind != 0 || accumulator_info.elements() != 1) {
                    return Err(invalid("CPU loss accumulator must be scalar F32"));
                }
                self.set_values(
                    accumulator,
                    &vec![if loss { 1.0 } else { 0.0 }; accumulator_info.elements()],
                )?;
            }
        }
        Ok(())
    }

    fn derivative_call(
        &mut self,
        context: u32,
        name: &str,
        arguments: &[Argument],
    ) -> Result<TensorId> {
        let mut args = Vec::with_capacity(arguments.len() + 1);
        args.push(Argument::Context(context));
        args.extend_from_slice(arguments);
        match self.call(name, &args)? {
            Observation::Tensor(Some(tensor)) => Ok(tensor),
            _ => Err(invalid("CPU derivative operation returned no tensor")),
        }
    }

    fn derivative_unary(&mut self, context: u32, name: &str, input: TensorId) -> Result<TensorId> {
        self.derivative_call(context, name, &[Argument::Tensor(input)])
    }

    fn derivative_binary(
        &mut self,
        context: u32,
        name: &str,
        a: TensorId,
        b: TensorId,
    ) -> Result<TensorId> {
        self.derivative_call(context, name, &[Argument::Tensor(a), Argument::Tensor(b)])
    }

    fn derivative_scale(&mut self, context: u32, input: TensorId, scale: f32) -> Result<TensorId> {
        self.derivative_call(
            context,
            "ggml_scale",
            &[Argument::Tensor(input), Argument::Float(scale)],
        )
    }

    fn derivative_reshape(
        &mut self,
        context: u32,
        input: TensorId,
        shape: [usize; 4],
    ) -> Result<TensorId> {
        let input = if self.info(input)?.contiguous() {
            input
        } else {
            self.derivative_unary(context, "ggml_cont", input)?
        };
        let mut arguments = vec![Argument::Tensor(input)];
        arguments.extend(shape.map(|value| Argument::Integer(value as i64)));
        self.derivative_call(context, "ggml_reshape_4d", &arguments)
    }

    fn derivative_view(
        &mut self,
        context: u32,
        input: TensorId,
        shape: [usize; 4],
        strides: [usize; 3],
        offset: usize,
    ) -> Result<TensorId> {
        let mut arguments = vec![Argument::Tensor(input)];
        arguments.extend(shape.map(|value| Argument::Integer(value as i64)));
        arguments.extend(strides.map(|value| Argument::Integer(value as i64)));
        arguments.push(Argument::Integer(offset as i64));
        self.derivative_call(context, "ggml_view_4d", &arguments)
    }

    fn add_derivative(
        &mut self,
        context: u32,
        graph: GraphId,
        source: TensorId,
        contribution: TensorId,
    ) -> Result<()> {
        if self.info(source)?.shape != self.info(contribution)?.shape {
            return Err(invalid(
                "CPU symbolic derivative shape differs from its source",
            ));
        }
        let result = if let Some(existing) = self.graph_gradient(graph, source)? {
            let inplace = self.graph_gradient_accumulator(graph, source)?.is_some();
            self.derivative_binary(
                context,
                if inplace {
                    "ggml_add_inplace"
                } else {
                    "ggml_add"
                },
                existing,
                contribution,
            )?
        } else {
            contribution
        };
        self.graphs[graph.0].gradients.insert(source, result);
        self.expand(graph, result)
    }

    fn scatter_derivative(
        &mut self,
        context: u32,
        graph: GraphId,
        source: TensorId,
        gradient: TensorId,
        strides: [usize; 3],
        offset: usize,
    ) -> Result<()> {
        let existing = self.graph_gradient(graph, source)?;
        let base = if let Some(existing) = existing {
            existing
        } else {
            let mut parameters = [0; 64];
            parameters[..4].copy_from_slice(&0.0_f32.to_le_bytes());
            let Observation::Tensor(Some(zero)) = self.derived(
                context,
                0,
                self.info(source)?.shape,
                32,
                vec![source],
                parameters,
                None,
            )?
            else {
                return Err(invalid("CPU gradient zero expression missing"));
            };
            zero
        };
        let mut parameters = [0; 64];
        for (index, value) in strides.into_iter().chain([offset]).enumerate() {
            parameters[index * 4..index * 4 + 4]
                .copy_from_slice(&i32::try_from(value)?.to_le_bytes());
        }
        let inplace = self.graph_gradient_accumulator(graph, source)?.is_some();
        parameters[16..20].copy_from_slice(&i32::from(inplace).to_le_bytes());
        let view = if inplace {
            Some((base, self.info(base)?.strides, 0))
        } else {
            None
        };
        let Observation::Tensor(Some(result)) = self.derived(
            context,
            0,
            self.info(source)?.shape,
            5,
            vec![base, gradient],
            parameters,
            view,
        )?
        else {
            return Err(invalid("CPU gradient scatter expression missing"));
        };
        self.graphs[graph.0].gradients.insert(source, result);
        self.expand(graph, result)
    }

    #[allow(
        clippy::too_many_lines,
        reason = "The symbolic derivative dispatch follows the pinned GGML backward operation switch."
    )]
    fn differentiate(
        &mut self,
        context: u32,
        graph: GraphId,
        tensor: TensorId,
        gradient: TensorId,
        needed: &BTreeSet<TensorId>,
    ) -> Result<()> {
        let info = self.info(tensor)?.clone();
        let inputs = info
            .sources
            .iter()
            .enumerate()
            .filter(|&(position, source)| !ignored(&info, position) && needed.contains(source))
            .map(|(position, &source)| (position, source))
            .collect::<Vec<_>>();
        if inputs.is_empty() {
            return Ok(());
        }
        let a = info.sources[0];
        let b = info.sources.get(1).copied();
        let source_b = || b.ok_or_else(|| invalid("CPU derivative second input missing"));
        for (position, source) in inputs {
            let shape = self.info(source)?.shape;
            let contribution = match info.operation {
                0 => continue,
                1 | 2 | 4 | 5 | 6 if position == 0 => gradient,
                2 => {
                    if self.info(gradient)?.shape == shape {
                        gradient
                    } else {
                        self.derivative_binary(context, "ggml_repeat_back", gradient, source)?
                    }
                }
                4 => self.derivative_unary(context, "ggml_mean", gradient)?,
                5 | 33 => {
                    let other = source_b()?;
                    let other_shape = self.info(other)?.shape;
                    let strides =
                        std::array::from_fn(|axis| integer(&info.parameters, axis) as usize);
                    let offset = integer(&info.parameters, 3) as usize;
                    let view =
                        self.derivative_view(context, gradient, other_shape, strides, offset)?;
                    if position == 0 {
                        let negative = self.derivative_scale(context, view, -1.0)?;
                        let mut arguments =
                            vec![Argument::Tensor(gradient), Argument::Tensor(negative)];
                        arguments.extend(
                            strides
                                .into_iter()
                                .chain([offset])
                                .map(|value| Argument::Integer(value as i64)),
                        );
                        self.derivative_call(context, "ggml_acc", &arguments)?
                    } else {
                        self.derivative_reshape(context, view, shape)?
                    }
                }
                6 => self.derivative_call(
                    context,
                    "ggml_unary",
                    &[Argument::Tensor(gradient), Argument::Integer(2)],
                )?,
                7 => {
                    let other = if position == 0 { source_b()? } else { a };
                    let product = self.derivative_binary(context, "ggml_mul", gradient, other)?;
                    if self.info(product)?.shape == shape {
                        product
                    } else {
                        self.derivative_binary(context, "ggml_repeat_back", product, source)?
                    }
                }
                8 if position == 0 => {
                    self.derivative_binary(context, "ggml_div", gradient, source_b()?)?
                }
                8 => {
                    let quotient = self.derivative_binary(context, "ggml_div", tensor, source)?;
                    let product =
                        self.derivative_binary(context, "ggml_mul", gradient, quotient)?;
                    self.derivative_scale(context, product, -1.0)?
                }
                9 => {
                    let product = self.derivative_binary(context, "ggml_mul", a, gradient)?;
                    self.derivative_scale(context, product, 2.0)?
                }
                10 => {
                    let quotient = self.derivative_binary(context, "ggml_div", gradient, tensor)?;
                    self.derivative_scale(context, quotient, 0.5)?
                }
                11 => self.derivative_binary(context, "ggml_div", gradient, a)?,
                12 | 13 => {
                    let derivative = self.derivative_unary(
                        context,
                        if info.operation == 12 {
                            "ggml_cos"
                        } else {
                            "ggml_sin"
                        },
                        a,
                    )?;
                    let product =
                        self.derivative_binary(context, "ggml_mul", gradient, derivative)?;
                    if info.operation == 13 {
                        self.derivative_scale(context, product, -1.0)?
                    } else {
                        product
                    }
                }
                14 | 15 | 17 | 20 | 21 => {
                    let input = if info.operation == 17 {
                        self.derivative_scale(context, gradient, 1.0 / shape[0] as f32)?
                    } else {
                        gradient
                    };
                    self.derivative_binary(
                        context,
                        if info.operation == 20 {
                            "ggml_repeat_back"
                        } else {
                            "ggml_repeat"
                        },
                        input,
                        source,
                    )?
                }
                25 => self.derivative_call(
                    context,
                    "ggml_rms_norm_back",
                    &[
                        Argument::Tensor(gradient),
                        Argument::Tensor(a),
                        Argument::Float(f32::from_bits(integer(&info.parameters, 0) as u32)),
                    ],
                )?,
                29 if position == 0 => {
                    let product =
                        self.derivative_binary(context, "ggml_out_prod", source_b()?, gradient)?;
                    if self.info(product)?.shape == shape {
                        product
                    } else {
                        let metadata = self.info(product)?.clone();
                        if metadata.shape[0..2] != shape[0..2]
                            || metadata.shape[3] != 1
                            || shape[2] == 0
                            || !metadata.shape[2].is_multiple_of(shape[2])
                        {
                            return Err(invalid("CPU matmul gradient broadcast geometry differs"));
                        }
                        let repeats = metadata.shape[2] / shape[2];
                        let view = self.derivative_view(
                            context,
                            product,
                            [shape[0], shape[1], shape[2], repeats],
                            [
                                metadata.strides[1],
                                metadata.strides[2] * repeats,
                                metadata.strides[2],
                            ],
                            0,
                        )?;
                        self.derivative_binary(context, "ggml_repeat_back", view, source)?
                    }
                }
                29 => {
                    let transpose = self.derivative_unary(context, "ggml_transpose", gradient)?;
                    self.derivative_binary(context, "ggml_out_prod", a, transpose)?
                }
                32 => self.derivative_scale(
                    context,
                    gradient,
                    f32::from_bits(integer(&info.parameters, 0) as u32),
                )?,
                34..=36 => self.derivative_reshape(context, gradient, shape)?,
                37 => {
                    let width = type_layout(self.info(source)?.kind)?.bytes;
                    let strides = [info.strides[1], info.strides[2], info.strides[3]];
                    let relative = u64::from_le_bytes(info.parameters[..8].try_into()?);
                    if strides
                        .into_iter()
                        .chain([relative as usize])
                        .any(|value| !value.is_multiple_of(width))
                    {
                        return Err(invalid(
                            "CPU view gradient byte geometry is not element aligned",
                        ));
                    }
                    let strides = strides.map(|value| value / width * 4);
                    let offset = usize::try_from(relative)? / width * 4;
                    self.scatter_derivative(context, graph, source, gradient, strides, offset)?;
                    continue;
                }
                38 => {
                    let mut inverse = [0_i64; 4];
                    for axis in 0..4 {
                        inverse[(integer(&info.parameters, axis) & 3) as usize] = axis as i64;
                    }
                    let mut arguments = vec![Argument::Tensor(gradient)];
                    arguments.extend(inverse.map(Argument::Integer));
                    self.derivative_call(context, "ggml_permute", &arguments)?
                }
                39 => self.derivative_unary(context, "ggml_transpose", gradient)?,
                40 => self.derivative_call(
                    context,
                    "ggml_get_rows_back",
                    &[
                        Argument::Tensor(gradient),
                        Argument::Tensor(source_b()?),
                        Argument::Tensor(a),
                    ],
                )?,
                44 | 45 => {
                    let mut p = [0; 64];
                    p[..4].copy_from_slice(&info.parameters[..4]);
                    let Observation::Tensor(Some(id)) =
                        self.derived(context, 0, shape, 45, vec![gradient], p, None)?
                    else {
                        return Err(invalid("CPU mask derivative missing"));
                    };
                    id
                }
                46 => {
                    if position != 0 {
                        return Err(invalid(
                            "CPU softmax mask gradients are unsupported by the pinned source",
                        ));
                    }
                    self.derivative_call(
                        context,
                        "ggml_soft_max_ext_back",
                        &[
                            Argument::Tensor(gradient),
                            Argument::Tensor(tensor),
                            Argument::Float(f32::from_bits(integer(&info.parameters, 0) as u32)),
                            Argument::Float(f32::from_bits(integer(&info.parameters, 1) as u32)),
                        ],
                    )?
                }
                48 => {
                    if position != 0 {
                        return Err(invalid(
                            "CPU rotary frequency gradients are unsupported by the pinned source",
                        ));
                    }
                    let mut sources = info.sources.clone();
                    sources[0] = gradient;
                    let mut parameters = info.parameters;
                    parameters[60..64].fill(0);
                    let Observation::Tensor(Some(id)) =
                        self.derived(context, 0, shape, 49, sources, parameters, None)?
                    else {
                        return Err(invalid("CPU rotary derivative missing"));
                    };
                    id
                }
                52 => {
                    let mut arguments = vec![
                        Argument::Tensor(gradient),
                        Argument::Tensor(a),
                        Argument::Integers(shape.map(|value| value as i64).to_vec()),
                    ];
                    arguments.extend((0..6).map(|index| {
                        Argument::Integer(i64::from(integer(&info.parameters, index)))
                    }));
                    arguments.push(Argument::Bool(integer(&info.parameters, 6) == 1));
                    self.derivative_call(context, "ggml_im2col_back", &arguments)?
                }
                61 => {
                    let mut arguments = vec![Argument::Tensor(gradient), Argument::Tensor(a)];
                    arguments.extend((0..5).map(|index| {
                        Argument::Integer(i64::from(integer(&info.parameters, index)))
                    }));
                    arguments.extend(
                        (5..7)
                            .map(|index| Argument::Float(integer(&info.parameters, index) as f32)),
                    );
                    self.derivative_call(context, "ggml_pool_2d_back", &arguments)?
                }
                91 => self.unary_derivative(context, &info, tensor, gradient)?,
                96 => {
                    if position != 0 {
                        return Err(invalid(
                            "CPU cross-entropy label gradients are unsupported by the pinned source",
                        ));
                    }
                    self.derivative_call(
                        context,
                        "ggml_cross_entropy_loss_back",
                        &[
                            Argument::Tensor(gradient),
                            Argument::Tensor(a),
                            Argument::Tensor(source_b()?),
                        ],
                    )?
                }
                100 if integer(&info.parameters, 0) == 2 && b.is_some() => {
                    if position == 0 {
                        let product =
                            self.derivative_binary(context, "ggml_mul", gradient, source_b()?)?;
                        self.derivative_binary(context, "ggml_silu_back", product, a)?
                    } else {
                        let silu = self.derivative_unary(context, "ggml_silu", a)?;
                        self.derivative_binary(context, "ggml_mul", silu, gradient)?
                    }
                }
                _ => {
                    return Err(invalid(format!(
                        "unsupported CPU symbolic derivative for operation {}",
                        info.operation
                    )));
                }
            };
            self.add_derivative(context, graph, source, contribution)?;
        }
        Ok(())
    }

    fn unary_derivative(
        &mut self,
        context: u32,
        info: &TensorInfo,
        output: TensorId,
        gradient: TensorId,
    ) -> Result<TensorId> {
        let input = info.sources[0];
        match integer(&info.parameters, 0) {
            0 | 6 | 14 | 15 => {
                let operation = match integer(&info.parameters, 0) {
                    0 => 1,
                    6 => 3,
                    14 => 13,
                    _ => 7,
                };
                let derivative = self.derivative_call(
                    context,
                    "ggml_unary",
                    &[Argument::Tensor(input), Argument::Integer(operation)],
                )?;
                self.derivative_binary(context, "ggml_mul", derivative, gradient)
            }
            2 => self.derivative_scale(context, gradient, -1.0),
            10 => self.derivative_binary(context, "ggml_silu_back", gradient, input),
            13 => self.derivative_binary(context, "ggml_mul", output, gradient),
            operation => Err(invalid(format!(
                "unsupported pinned CPU unary derivative {operation}"
            ))),
        }
    }
}
