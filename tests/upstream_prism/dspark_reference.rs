//! Independent scalar reference for the tiny `DeepSpec` Qwen3 `DSpark` fixture.
//! Equations follow `DeepSpec` revision 005e03b81cec38b7da6399833d609ee89a2587f2,
//! `qwen3/modeling.py`, `markov_head.py` and `eval/dspark/draft_ops.py` (MIT).
//! No production engine calls or recorded production answers appear here.
use super::{Result, invalid};
use serde_json::Value;
use std::collections::BTreeMap;

const EMBEDDING: usize = 64;
const HEAD: usize = 32;
const VOCABULARY: usize = 256;
const RANK: usize = 8;
#[allow(
    clippy::cast_possible_truncation,
    clippy::as_conversions,
    reason = "independent scalar reference rounds each tensor result to its source F32 dtype"
)]
const fn f32_round(value: f64) -> f32 {
    value as f32
}
fn size(value: usize) -> Result<f64> {
    Ok(f64::from(u32::try_from(value)?))
}
fn checked(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(invalid(message)) }
}
struct Tensor {
    dimensions: Vec<usize>,
    values: Vec<f32>,
}
fn tensor(manifest: &Value, layout: &str, name: &str) -> Result<Tensor> {
    let row = manifest["Layouts"]
        .as_array()
        .and_then(|rows| rows.iter().find(|r| r["Name"] == layout))
        .and_then(|r| r["Tensors"].as_array())
        .and_then(|rows| rows.iter().find(|r| r["Name"] == name))
        .ok_or_else(|| invalid(format!("independent DSpark tensor absent {name}")))?;
    let integer = |key: &str| -> Result<usize> {
        Ok(usize::try_from(row[key].as_u64().ok_or_else(|| {
            invalid("independent tensor extent absent")
        })?)?)
    };
    let begin = integer("Offset")?;
    let end = begin
        .checked_add(integer("Bytes")?)
        .ok_or_else(|| invalid("independent tensor input extent overflow"))?;
    let raw = include_bytes!("dspark-inputs.bin")
        .get(begin..end)
        .ok_or_else(|| invalid("independent tensor input outside corpus"))?;
    checked(raw.len() % 4 == 0, "independent F32 input misaligned")?;
    let values = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&v| f32::from_le_bytes(v))
        .collect::<Vec<_>>();
    let dimensions = row["Dimensions"]
        .as_array()
        .ok_or_else(|| invalid("independent tensor dimensions absent"))?
        .iter()
        .map(|v| {
            Ok(usize::try_from(v.as_u64().ok_or_else(|| {
                invalid("independent tensor dimension is not unsigned")
            })?)?)
        })
        .collect::<Result<Vec<_>>>()?;
    let count = dimensions.iter().try_fold(1_usize, |n, &d| {
        n.checked_mul(d)
            .ok_or_else(|| invalid("independent tensor count overflow"))
    })?;
    checked(
        count == values.len(),
        "independent tensor byte count differs",
    )?;
    Ok(Tensor { dimensions, values })
}
fn linear(input: &[f32], matrix: &Tensor) -> Result<Vec<f32>> {
    checked(
        matrix.dimensions.len() == 2 && matrix.dimensions[0] == input.len(),
        "independent linear shape mismatch",
    )?;
    Ok(matrix
        .values
        .chunks_exact(input.len())
        .map(|weight| {
            f32_round(
                input
                    .iter()
                    .zip(weight)
                    .map(|(&x, &w)| f64::from(x) * f64::from(w))
                    .sum(),
            )
        })
        .collect())
}
fn normalize(input: &[f32], weight: &[f32]) -> Result<Vec<f32>> {
    checked(
        !input.is_empty() && input.len() == weight.len(),
        "independent RMS shape mismatch",
    )?;
    let variance = input.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / size(input.len())?;
    let inverse = (variance + 1e-6).sqrt().recip();
    Ok(input
        .iter()
        .zip(weight)
        .map(|(&x, &w)| f32_round(f64::from(x) * inverse * f64::from(w)))
        .collect())
}
#[allow(
    clippy::suboptimal_flops,
    reason = "keep the published separate rotate-half products in the scalar reference"
)]
fn rotary(values: &mut [f32], position: i32) -> Result<()> {
    checked(
        values.len().is_multiple_of(HEAD),
        "independent RoPE head mismatch",
    )?;
    for head in values.as_chunks_mut::<HEAD>().0 {
        for i in 0..HEAD / 2 {
            let angle = f64::from(position) * 10000_f64.powf(-size(2 * i)? / size(HEAD)?);
            let (sin, cos) = angle.sin_cos();
            let (a, b) = (f64::from(head[i]), f64::from(head[i + HEAD / 2]));
            head[i] = f32_round(a * cos - b * sin);
            head[i + HEAD / 2] = f32_round(b * cos + a * sin);
        }
    }
    Ok(())
}
#[derive(Default)]
struct Cache {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}
pub(super) struct Reference {
    tensors: BTreeMap<String, Tensor>,
    caches: [Cache; 2],
}
impl Reference {
    pub(super) fn provenance() -> Result<Value> {
        let value: Value = serde_json::from_str(include_str!("dspark-reference-provenance.json"))?;
        checked(
            value["Revision"] == "005e03b81cec38b7da6399833d609ee89a2587f2"
                && value["LicenseText"]
                    .as_str()
                    .is_some_and(|v| v.contains("MIT License")),
            "independent reference provenance invalid",
        )?;
        Ok(value)
    }
    pub(super) fn new() -> Result<Self> {
        super::dspark_inputs::Corpus::load()?;
        Self::provenance()?;
        let manifest: Value = serde_json::from_str(include_str!("dspark-inputs.json"))?;
        let mut tensors = BTreeMap::new();
        let rows = manifest["Layouts"]
            .as_array()
            .and_then(|r| r.iter().find(|r| r["Name"] == "unconditioned"))
            .and_then(|r| r["Tensors"].as_array())
            .ok_or_else(|| invalid("independent tensor corpus missing"))?;
        for row in rows {
            let name = row["Name"]
                .as_str()
                .ok_or_else(|| invalid("independent tensor name absent"))?;
            tensors.insert(name.into(), tensor(&manifest, "unconditioned", name)?);
        }
        for name in ["dspark.markov_head_a.weight", "dspark.markov_head_b.weight"] {
            tensors.insert(name.into(), tensor(&manifest, "corrected", name)?);
        }
        Ok(Self {
            tensors,
            caches: std::array::from_fn(|_| Cache::default()),
        })
    }
    fn get(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| invalid(format!("independent weight absent {name}")))
    }
    fn projection(&self, name: &str, input: &[f32]) -> Result<Vec<f32>> {
        linear(input, self.get(name)?)
    }
    fn norm(&self, name: &str, input: &[f32]) -> Result<Vec<f32>> {
        normalize(input, &self.get(name)?.values)
    }
    #[allow(
        clippy::too_many_lines,
        reason = "independent published backbone equations and cache crop stay in one auditable sequence"
    )]
    pub(super) fn forward(
        &mut self,
        context_features: &[f32],
        context_positions: &[i32],
        draft_tokens: &[i32],
        draft_positions: &[i32],
        crop: Option<usize>,
    ) -> Result<Vec<f32>> {
        checked(
            !draft_tokens.is_empty()
                && draft_tokens.len() <= 8
                && draft_tokens.len() == draft_positions.len()
                && context_features.len() == context_positions.len() * 128
                && context_positions.len() <= 8,
            "independent DSpark forward input shape mismatch",
        )?;
        let mut target = Vec::new();
        for row in context_features.as_chunks::<128>().0 {
            target.push(self.norm(
                "dspark.hidden_norm.weight",
                &self.projection("dspark.fc.weight", row)?,
            )?);
        }
        let embedding = self.get("token_embd.weight")?;
        let mut hidden = draft_tokens
            .iter()
            .map(|&token| {
                let token = usize::try_from(token)?;
                checked(token < VOCABULARY, "independent token outside vocabulary")?;
                Ok(embedding.values[token * EMBEDDING..(token + 1) * EMBEDDING].to_vec())
            })
            .collect::<Result<Vec<_>>>()?;
        for layer in 0..2 {
            let prefix = format!("blk.{layer}");
            let normalized = hidden
                .iter()
                .map(|row| self.norm(&format!("{prefix}.attn_norm.weight"), row))
                .collect::<Result<Vec<_>>>()?;
            let mut keys = Vec::new();
            let mut values = Vec::new();
            for (row, &position) in target
                .iter()
                .chain(&normalized)
                .zip(context_positions.iter().chain(draft_positions))
            {
                let mut key = self.norm(
                    &format!("{prefix}.attn_k_norm.weight"),
                    &self.projection(&format!("{prefix}.attn_k.weight"), row)?,
                )?;
                rotary(&mut key, position)?;
                keys.push(key);
                values.push(self.projection(&format!("{prefix}.attn_v.weight"), row)?);
            }
            self.caches[layer].keys.extend(keys);
            self.caches[layer].values.extend(values);
            checked(
                self.caches[layer].keys.len() <= 128,
                "independent cache exceeds128 rows",
            )?;
            let mut next = Vec::new();
            for ((residual, row), &position) in hidden.iter().zip(&normalized).zip(draft_positions)
            {
                let mut query = self.projection(&format!("{prefix}.attn_q.weight"), row)?;
                for head in query.as_chunks_mut::<HEAD>().0 {
                    let normalized = self.norm(&format!("{prefix}.attn_q_norm.weight"), head)?;
                    head.copy_from_slice(&normalized);
                }
                rotary(&mut query, position)?;
                let mut attention = Vec::new();
                for head in query.as_chunks::<HEAD>().0 {
                    let scores = self.caches[layer]
                        .keys
                        .iter()
                        .map(|key| {
                            head.iter()
                                .zip(key)
                                .map(|(&q, &k)| f64::from(q) * f64::from(k))
                                .sum::<f64>()
                                / 32_f64.sqrt()
                        })
                        .collect::<Vec<_>>();
                    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let exponentials = scores
                        .iter()
                        .map(|v| (v - maximum).exp())
                        .collect::<Vec<_>>();
                    let denominator: f64 = exponentials.iter().sum();
                    for d in 0..HEAD {
                        attention.push(f32_round(
                            exponentials
                                .iter()
                                .zip(&self.caches[layer].values)
                                .map(|(&p, value)| p * f64::from(value[d]) / denominator)
                                .sum(),
                        ));
                    }
                }
                let projected =
                    self.projection(&format!("{prefix}.attn_output.weight"), &attention)?;
                let residual = residual
                    .iter()
                    .zip(projected)
                    .map(|(&a, b)| a + b)
                    .collect::<Vec<_>>();
                let norm = self.norm(&format!("{prefix}.ffn_norm.weight"), &residual)?;
                let gate = self.projection(&format!("{prefix}.ffn_gate.weight"), &norm)?;
                let up = self.projection(&format!("{prefix}.ffn_up.weight"), &norm)?;
                let activation = gate
                    .iter()
                    .zip(up)
                    .map(|(&g, u)| f32_round(f64::from(g) / (1.0 + (-f64::from(g)).exp())) * u)
                    .collect::<Vec<_>>();
                let down = self.projection(&format!("{prefix}.ffn_down.weight"), &activation)?;
                next.push(residual.iter().zip(down).map(|(&a, b)| a + b).collect());
            }
            if let Some(keep) = crop {
                checked(
                    keep <= self.caches[layer].keys.len(),
                    "independent crop exceeds cache",
                )?;
                self.caches[layer].keys.truncate(keep);
                self.caches[layer].values.truncate(keep);
            }
            hidden = next;
        }
        let mut logits = Vec::new();
        for row in hidden {
            logits.extend(
                self.projection("token_embd.weight", &self.norm("output_norm.weight", &row)?)?,
            );
        }
        checked(
            logits.iter().all(|v| v.is_finite()),
            "independent logits nonfinite",
        )?;
        Ok(logits)
    }
    pub(super) fn sample(&self, logits: &[f32], anchor: i32) -> Result<Vec<i32>> {
        checked(
            logits.len().is_multiple_of(VOCABULARY) && (0..255).contains(&anchor),
            "independent Markov input shape/anchor invalid",
        )?;
        let a = self.get("dspark.markov_head_a.weight")?;
        let b = self.get("dspark.markov_head_b.weight")?;
        let mut previous = usize::try_from(anchor)?;
        let mut sampled = Vec::new();
        for row in logits.as_chunks::<VOCABULARY>().0 {
            let mut best = f64::NEG_INFINITY;
            let mut winner = 0;
            for (token, &base) in row.iter().enumerate().take(255) {
                let bias: f64 = (0..RANK)
                    .map(|r| {
                        f64::from(a.values[previous * RANK + r])
                            * f64::from(b.values[token * RANK + r])
                    })
                    .sum();
                let score = f64::from(base) + bias;
                if score > best {
                    best = score;
                    winner = token;
                }
            }
            sampled.push(i32::try_from(winner)?);
            previous = winner;
        }
        Ok(sampled)
    }
}
