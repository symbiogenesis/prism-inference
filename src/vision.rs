//! Bounded Qwen3-VL vision encoder (`mmproj`) for native multimodal Swift inference.
//!
//! Decodes PNG and PPM inputs without external imaging dependencies, applies
//! dynamic aspect-preserving bicubic preprocessing, runs the 27-layer `ViT` with
//! independent-sector 2D `RoPE`, and projects 2x2 merged patch groups into the
//! unrotated decoder residual stream.

#[path = "vision/decode.rs"]
pub mod decode;

use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex};

use bonsai_native_hash::Hasher;
#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
use fearless_simd::{Neon, f32x4, prelude::*};

use crate::kernels::prefill::{MAX_TILE_TOKENS, ProjectionTile};
use crate::kernels::{
    MatvecBackend, RowParallelism, is_finite_slice, row_into, softmax, validate_tensor,
};
use crate::model::swift::MmprojLoadReceipt;
use crate::model::{Model, Tensor};
use crate::{Result, invalid};

use decode::{
    DecodedImage, ImageDimensionBudget, decode_encoded_image_within, decode_image_data_url_raw,
};

const DEFAULT_MIN_IMAGE_TOKENS: usize = 4;
const MAX_CACHE_ENTRIES: usize = 8;
const VIT_ROPE_FREQ_BASE: f32 = 10_000.0;
const SQRT_2_OVER_PI: f32 = 0.797_884_6;

/// Validated encoded vision tokens for one image in the decoder residual stream.
#[derive(Clone, Debug)]
pub struct EncodedImage {
    /// Post-merge vision token grid width (`resized_width / (patch_size * spatial_merge)`).
    pub grid_width: usize,
    /// Post-merge vision token grid height (`resized_height / (patch_size * spatial_merge)`).
    pub grid_height: usize,
    /// SHA-256 digest of the raw encoded image bytes before decompression.
    pub sha256: [u8; 32],
    /// Flattened `(grid_width * grid_height) * projection_dim` decoder-space embeddings.
    pub embeddings: Arc<[f32]>,
}

/// Architecture hyperparameters verified from a `clip` `qwen3vl_merger` projector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VisionConfig {
    /// Square spatial patch side length in pixels (`16`).
    pub patch_size: usize,
    /// Spatial merger grouping factor (`2`, merging `2x2 = 4` patches per token).
    pub spatial_merge: usize,
    /// `ViT` hidden width (`1152`).
    pub hidden_size: usize,
    /// Attention head count (`16`, `72` floats per head).
    pub num_heads: usize,
    /// `ViT` feed-forward intermediate width (`4304`).
    pub intermediate_size: usize,
    /// Number of `ViT` transformer blocks (`27`).
    pub num_layers: usize,
    /// Output projection width matching the text decoder (`5120`).
    pub projection_dim: usize,
    /// `LayerNorm` epsilon (`1e-6`).
    pub eps: f32,
    /// Per-channel normalization mean (`[0.5, 0.5, 0.5]`).
    pub mean: [f32; 3],
    /// Per-channel normalization standard deviation (`[0.5, 0.5, 0.5]`).
    pub std: [f32; 3],
    /// Learned positional embedding grid side (`48` for `48x48 = 2304`).
    pub pos_grid_side: usize,
}

struct VitBlock {
    ln1_w: Vec<f32>,
    ln1_b: Vec<f32>,
    attn_qkv_w: Tensor,
    attn_qkv_b: Vec<f32>,
    attn_out_w: Tensor,
    attn_out_b: Vec<f32>,
    ln2_w: Vec<f32>,
    ln2_b: Vec<f32>,
    ffn_up_w: Tensor,
    ffn_up_b: Vec<f32>,
    ffn_down_w: Tensor,
    ffn_down_b: Vec<f32>,
}

struct CachedImage {
    sha256: [u8; 32],
    max_tokens: usize,
    encoded: EncodedImage,
}

/// Immutable native vision encoder loaded from the pinned `mmproj` artifact.
pub struct SwiftVisionEncoder {
    config: VisionConfig,
    patch_embd: Tensor,
    patch_bias: Vec<f32>,
    pos_embd: Vec<f32>,
    blocks: Vec<VitBlock>,
    post_ln_w: Vec<f32>,
    post_ln_b: Vec<f32>,
    mm_0_w: Tensor,
    mm_0_b: Vec<f32>,
    mm_2_w: Tensor,
    mm_2_b: Vec<f32>,
    backend: MatvecBackend,
    row_parallelism: RowParallelism,
    cache: Mutex<VecDeque<CachedImage>>,
}

impl SwiftVisionEncoder {
    /// Loads and cryptographically verifies the pinned `mmproj` artifact.
    ///
    /// # Errors
    /// Rejects hash/size drift, malformed GGUF metadata or tensor shapes, nonfinite
    /// parameters, or memory budget exhaustion.
    pub fn load(
        path: &Path,
        memory_limit: usize,
        backend: MatvecBackend,
        row_parallelism: RowParallelism,
    ) -> Result<(Self, MmprojLoadReceipt)> {
        let (model, receipt) = crate::model::swift::load_mmproj(path, memory_limit)?;
        let encoder = Self::from_model(&model, backend, row_parallelism)?;
        Ok((encoder, receipt))
    }

    /// Constructs a vision encoder from an already parsed `clip` `qwen3vl_merger` model.
    ///
    /// # Errors
    /// Rejects incompatible metadata, missing/reshaped/nonfinite tensors, or
    /// invalid head/merge geometry.
    pub fn from_model(
        model: &Model,
        backend: MatvecBackend,
        row_parallelism: RowParallelism,
    ) -> Result<Self> {
        let config = parse_vision_config(model)?;
        let patch_pixels = config
            .patch_size
            .checked_mul(config.patch_size)
            .and_then(|area| area.checked_mul(3))
            .ok_or_else(|| invalid("vision patch pixel count overflow"))?;
        let patch0 = require_tensor(
            model,
            "v.patch_embd.weight",
            &[config.patch_size, config.patch_size, 3, config.hidden_size],
        )?;
        let patch1 = require_tensor(
            model,
            "v.patch_embd.weight.1",
            &[config.patch_size, config.patch_size, 3, config.hidden_size],
        )?;
        let patch_embd =
            fuse_temporal_patch_weights(patch0, patch1, patch_pixels, config.hidden_size)?;
        let patch_bias = load_f32_vector(model, "v.patch_embd.bias", config.hidden_size)?;
        let pos_rows = config
            .pos_grid_side
            .checked_mul(config.pos_grid_side)
            .ok_or_else(|| invalid("vision position grid overflow"))?;
        let pos_tensor = require_tensor(
            model,
            "v.position_embd.weight",
            &[config.hidden_size, pos_rows],
        )?;
        let pos_embd =
            load_matrix_rows(pos_tensor, pos_rows, config.hidden_size, &row_parallelism)?;

        let mut blocks = Vec::with_capacity(config.num_layers);
        for block in 0..config.num_layers {
            blocks.push(load_block(model, &config, block)?);
        }

        let post_ln_w = load_f32_vector(model, "v.post_ln.weight", config.hidden_size)?;
        let post_ln_b = load_f32_vector(model, "v.post_ln.bias", config.hidden_size)?;

        let merge_dim = config
            .hidden_size
            .checked_mul(config.spatial_merge * config.spatial_merge)
            .ok_or_else(|| invalid("vision merger width overflow"))?;
        let mm_0_w = require_tensor(model, "mm.0.weight", &[merge_dim, merge_dim])?.clone();
        let mm_0_b = load_f32_vector(model, "mm.0.bias", merge_dim)?;
        let mm_2_w =
            require_tensor(model, "mm.2.weight", &[merge_dim, config.projection_dim])?.clone();
        let mm_2_b = load_f32_vector(model, "mm.2.bias", config.projection_dim)?;

        Ok(Self {
            config,
            patch_embd,
            patch_bias,
            pos_embd,
            blocks,
            post_ln_w,
            post_ln_b,
            mm_0_w,
            mm_0_b,
            mm_2_w,
            mm_2_b,
            backend,
            row_parallelism,
            cache: Mutex::new(VecDeque::with_capacity(MAX_CACHE_ENTRIES)),
        })
    }

    /// Returns the verified vision encoder hyperparameters.
    #[must_use]
    pub const fn config(&self) -> &VisionConfig {
        &self.config
    }

    /// Returns the output embedding dimension projected into the language decoder.
    #[must_use]
    pub const fn projection_dim(&self) -> usize {
        self.config.projection_dim
    }

    /// Clears all cached encoded image embeddings.
    ///
    /// # Errors
    /// Returns an error if the internal cache mutex is poisoned.
    pub fn clear_cache(&self) -> Result<()> {
        self.cache
            .lock()
            .map_err(|_| invalid("vision cache mutex poisoned"))?
            .clear();
        Ok(())
    }

    /// Decodes a validated `data:image/...;base64,...` URL and encodes its vision tokens.
    ///
    /// Hashes the raw container payload before PNG/PPM pixel decoding so cache hits
    /// avoid decompressing image pixels, and validates visual budget/aspect bounds early.
    ///
    /// # Errors
    /// Rejects invalid image payloads, token bounds, cache lock failures, or
    /// nonfinite intermediate activations.
    pub fn encode_data_url(&self, url: &str, max_tokens: usize) -> Result<EncodedImage> {
        let factor = self.config.patch_size * self.config.spatial_merge;
        let min_tokens = DEFAULT_MIN_IMAGE_TOKENS.min(max_tokens).max(1);
        let budget = ImageDimensionBudget {
            min_tokens,
            max_tokens,
            grid_factor: factor,
        };
        budget.validate_bounds()?;
        let (format, raw) = decode_image_data_url_raw(url)?;
        let sha256 = sha256_bytes(&raw)?;
        if let Some(cached) = self.cache_lookup(sha256, max_tokens)? {
            return Ok(cached);
        }
        let decoded = decode_encoded_image_within(format, &raw, budget)?;
        let encoded = self.encode_decoded(&decoded, sha256, max_tokens)?;
        self.cache_store(sha256, max_tokens, encoded.clone())?;
        Ok(encoded)
    }

    /// Encodes a pre-decoded `RGB8` bitmap directly into decoder residual embeddings.
    ///
    /// # Errors
    /// Rejects dimension/length mismatches, invalid token caps, or nonfinite values.
    pub fn encode_rgb_image(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
        max_tokens: usize,
    ) -> Result<EncodedImage> {
        let expected_len = width
            .checked_mul(height)
            .and_then(|px| px.checked_mul(3))
            .ok_or_else(|| invalid("RGB bitmap size overflow"))?;
        if rgb.len() != expected_len
            || width == 0
            || height == 0
            || width > decode::MAX_IMAGE_DIMENSION
            || height > decode::MAX_IMAGE_DIMENSION
        {
            return Err(invalid("invalid RGB image dimensions or buffer length"));
        }
        let sha256 = sha256_rgb_image(rgb, width, height)?;
        if let Some(cached) = self.cache_lookup(sha256, max_tokens)? {
            return Ok(cached);
        }
        let decoded = DecodedImage {
            width,
            height,
            rgb: rgb.to_vec(),
        };
        let encoded = self.encode_decoded(&decoded, sha256, max_tokens)?;
        self.cache_store(sha256, max_tokens, encoded.clone())?;
        Ok(encoded)
    }

    fn cache_lookup(&self, sha256: [u8; 32], max_tokens: usize) -> Result<Option<EncodedImage>> {
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| invalid("vision cache mutex poisoned"))?;
        let Some(pos) = cache
            .iter()
            .position(|entry| entry.sha256 == sha256 && entry.max_tokens == max_tokens)
        else {
            return Ok(None);
        };
        let entry = cache
            .remove(pos)
            .ok_or_else(|| invalid("vision cache entry missing"))?;
        let encoded = entry.encoded.clone();
        cache.push_back(entry);
        drop(cache);
        Ok(Some(encoded))
    }

    fn cache_store(
        &self,
        sha256: [u8; 32],
        max_tokens: usize,
        encoded: EncodedImage,
    ) -> Result<()> {
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| invalid("vision cache mutex poisoned"))?;
        if cache.len() >= MAX_CACHE_ENTRIES {
            let _ = cache.pop_front();
        }
        cache.push_back(CachedImage {
            sha256,
            max_tokens,
            encoded,
        });
        drop(cache);
        Ok(())
    }

    const fn use_pinned_projection(&self, tensor: &Tensor) -> bool {
        !matches!(self.backend, MatvecBackend::Scalar) && tensor.kind == crate::kernels::F16
    }

    fn encode_decoded(
        &self,
        decoded: &DecodedImage,
        sha256: [u8; 32],
        max_tokens: usize,
    ) -> Result<EncodedImage> {
        let factor = self.config.patch_size * self.config.spatial_merge;
        let min_tokens = DEFAULT_MIN_IMAGE_TOKENS.min(max_tokens).max(1);
        let (resized_w, resized_h) = plan_grid_with_factor(
            decoded.width,
            decoded.height,
            factor,
            min_tokens,
            max_tokens,
        )?;
        let planar = resize_bicubic_normalize(
            &decoded.rgb,
            decoded.width,
            decoded.height,
            resized_w,
            resized_h,
            self.config.mean,
            self.config.std,
            &self.row_parallelism,
        )?;

        let patch_cols = resized_w / self.config.patch_size;
        let patch_rows = resized_h / self.config.patch_size;
        let nx = resized_w / factor;
        let ny = resized_h / factor;
        let n_pos = patch_cols * patch_rows;
        let n_tokens = nx * ny;

        let mut tile = ProjectionTile::with_backend(self.backend);
        tile.set_row_parallelism(self.row_parallelism.clone());
        let mut projector = TiledProjector {
            tile,
            inputs: vec![Vec::new(); MAX_TILE_TOKENS],
            outputs: vec![Vec::new(); MAX_TILE_TOKENS],
        };

        let (mut hidden, patch_coords) = self.extract_and_embed_patches(
            &planar,
            resized_w,
            patch_cols,
            patch_rows,
            nx,
            ny,
            &mut projector,
        )?;

        let total_hidden = hidden.len();
        let mut residual = vec![0.0_f32; total_hidden];
        let mut norm_buf = vec![0.0_f32; total_hidden];
        let mut attn = AttentionScratch::new(
            total_hidden,
            self.config.num_heads,
            n_pos,
            self.row_parallelism.workers(),
        )?;
        let mut attn_out_buf = vec![0.0_f32; total_hidden];
        let rope_cache = build_vit_rope_cache(&self.config, &patch_coords)?;

        for block in &self.blocks {
            self.run_vit_block(
                block,
                n_pos,
                &rope_cache,
                &mut hidden,
                &mut residual,
                &mut norm_buf,
                &mut attn,
                &mut attn_out_buf,
                &mut projector,
            )?;
        }

        layer_norm_rows(
            &hidden,
            n_pos,
            self.config.hidden_size,
            &self.post_ln_w,
            &self.post_ln_b,
            self.config.eps,
            &mut norm_buf,
            &self.row_parallelism,
            self.backend,
        )?;
        let embeddings = self.run_merger(&norm_buf, n_tokens, &mut projector)?;
        Ok(EncodedImage {
            grid_width: nx,
            grid_height: ny,
            sha256,
            embeddings: Arc::from(embeddings),
        })
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Reuse layer-level ViT scratch buffers across transformer blocks."
    )]
    fn run_vit_block(
        &self,
        block: &VitBlock,
        n_pos: usize,
        rope_cache: &[(f32, f32)],
        hidden: &mut [f32],
        residual: &mut [f32],
        norm_buf: &mut [f32],
        attn: &mut AttentionScratch,
        attn_out_buf: &mut [f32],
        projector: &mut TiledProjector,
    ) -> Result<()> {
        residual.copy_from_slice(hidden);
        layer_norm_rows(
            hidden,
            n_pos,
            self.config.hidden_size,
            &block.ln1_w,
            &block.ln1_b,
            self.config.eps,
            norm_buf,
            &self.row_parallelism,
            self.backend,
        )?;
        run_self_attention(
            block,
            &self.config,
            n_pos,
            rope_cache,
            norm_buf,
            attn,
            attn_out_buf,
            projector,
            &self.row_parallelism,
            self.backend,
        )?;
        for (h, (&res, &attn_val)) in hidden.iter_mut().zip(residual.iter().zip(&*attn_out_buf)) {
            *h = res + attn_val;
        }
        if !is_finite_slice(hidden) {
            return Err(invalid("nonfinite values after ViT attention residual"));
        }

        residual.copy_from_slice(hidden);
        layer_norm_rows(
            hidden,
            n_pos,
            self.config.hidden_size,
            &block.ln2_w,
            &block.ln2_b,
            self.config.eps,
            norm_buf,
            &self.row_parallelism,
            self.backend,
        )?;
        run_mlp_block(
            block,
            &self.config,
            n_pos,
            norm_buf,
            residual,
            hidden,
            projector,
            &self.row_parallelism,
            self.backend,
        )?;
        if !is_finite_slice(hidden) {
            return Err(invalid("nonfinite values after ViT MLP residual"));
        }
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Keep spatial-merge ordering, temporal sum, and bilinear pos-embd explicit."
    )]
    fn extract_and_embed_patches(
        &self,
        planar: &[f32],
        resized_w: usize,
        patch_cols: usize,
        patch_rows: usize,
        nx: usize,
        ny: usize,
        projector: &mut TiledProjector,
    ) -> Result<PatchEmbeddings> {
        let p = self.config.patch_size;
        let sm = self.config.spatial_merge;
        let d = self.config.hidden_size;
        let n_pos = patch_cols * patch_rows;
        let resized_h = patch_rows * p;
        let plane_len = resized_w * resized_h;
        let patch_pixels = p * p * 3;

        let mut coords = Vec::with_capacity(n_pos);
        for by in 0..ny {
            for bx in 0..nx {
                for dy in 0..sm {
                    for dx in 0..sm {
                        coords.push((by * sm + dy, bx * sm + dx));
                    }
                }
            }
        }

        let mut hidden = vec![0.0_f32; n_pos * d];
        let mut pos_row = vec![0.0_f32; d];
        for (tile_idx, coord_chunk) in coords.chunks(MAX_TILE_TOKENS).enumerate() {
            let chunk = coord_chunk.len();
            for (t, &(py, px)) in coord_chunk.iter().enumerate() {
                let patch_vec = &mut projector.inputs[t];
                patch_vec.resize(patch_pixels, 0.0);
                for c in 0..3 {
                    let c_off = c * plane_len;
                    let dst_c = c * p * p;
                    for ky in 0..p {
                        let src_row = c_off + (py * p + ky) * resized_w + px * p;
                        let dst_row = dst_c + ky * p;
                        patch_vec[dst_row..dst_row + p]
                            .copy_from_slice(&planar[src_row..src_row + p]);
                    }
                }
            }
            projector.tile.project_with_previous(
                None,
                &self.patch_embd,
                &projector.inputs[..chunk],
                &mut projector.outputs[..chunk],
                self.use_pinned_projection(&self.patch_embd),
                &|| Ok(()),
            )?;
            let base_idx = tile_idx * MAX_TILE_TOKENS;
            for (t, (&(py, px), patch_out)) in coord_chunk
                .iter()
                .zip(&projector.outputs[..chunk])
                .enumerate()
            {
                bilinear_pos_embd(
                    &self.pos_embd,
                    self.config.pos_grid_side,
                    d,
                    patch_cols,
                    patch_rows,
                    px,
                    py,
                    &mut pos_row,
                )?;
                let patch_idx = base_idx + t;
                let dst = &mut hidden[patch_idx * d..(patch_idx + 1) * d];
                for ((out, (&val, &bias)), &pos) in dst
                    .iter_mut()
                    .zip(patch_out.iter().zip(&self.patch_bias))
                    .zip(&pos_row)
                {
                    *out = val + bias + pos;
                }
            }
        }
        if !is_finite_slice(&hidden) {
            return Err(invalid("nonfinite patch/position embedding output"));
        }
        Ok((hidden, coords))
    }

    fn run_merger(
        &self,
        normed: &[f32],
        n_tokens: usize,
        projector: &mut TiledProjector,
    ) -> Result<Vec<f32>> {
        let merge_group = self.config.spatial_merge * self.config.spatial_merge;
        let merge_dim = self.config.hidden_size * merge_group;
        let proj_dim = self.config.projection_dim;
        let mut out = Vec::new();
        out.try_reserve_exact(n_tokens * proj_dim)?;

        for tile_start in (0..n_tokens).step_by(MAX_TILE_TOKENS) {
            let chunk = (n_tokens - tile_start).min(MAX_TILE_TOKENS);
            copy_tile_rows(&mut projector.inputs, chunk, normed, tile_start, merge_dim);
            projector.tile.project_with_previous(
                None,
                &self.mm_0_w,
                &projector.inputs[..chunk],
                &mut projector.outputs[..chunk],
                self.use_pinned_projection(&self.mm_0_w),
                &|| Ok(()),
            )?;
            apply_bias_gelu_rows(
                &mut projector.outputs[..chunk],
                &self.mm_0_b,
                &self.row_parallelism,
                self.backend,
            )?;
            std::mem::swap(&mut projector.inputs, &mut projector.outputs);
            projector.tile.project_with_previous(
                None,
                &self.mm_2_w,
                &projector.inputs[..chunk],
                &mut projector.outputs[..chunk],
                self.use_pinned_projection(&self.mm_2_w),
                &|| Ok(()),
            )?;
            for proj_row in &projector.outputs[..chunk] {
                for (&val, &bias) in proj_row.iter().zip(&self.mm_2_b) {
                    let final_val = val + bias;
                    if !final_val.is_finite() {
                        return Err(invalid("nonfinite value in multimodal merger projection"));
                    }
                    out.push(final_val);
                }
            }
        }
        Ok(out)
    }
}

struct TiledProjector {
    tile: ProjectionTile,
    inputs: Vec<Vec<f32>>,
    outputs: Vec<Vec<f32>>,
}

struct AttentionScratch {
    q_heads: Vec<f32>,
    k_heads: Vec<f32>,
    v_heads: Vec<f32>,
    head_out: Vec<f32>,
    head_scores: Vec<f32>,
}

impl AttentionScratch {
    fn new(len: usize, num_heads: usize, n_pos: usize, workers: usize) -> Result<Self> {
        let score_slots = num_heads.max(workers);
        let score_len = score_slots
            .checked_mul(4)
            .and_then(|n| n.checked_mul(n_pos))
            .ok_or_else(|| invalid("vision attention score scratch overflow"))?;
        let mut head_scores = Vec::new();
        head_scores.try_reserve_exact(score_len)?;
        head_scores.resize(score_len, 0.0_f32);
        Ok(Self {
            q_heads: vec![0.0_f32; len],
            k_heads: vec![0.0_f32; len],
            v_heads: vec![0.0_f32; len],
            head_out: vec![0.0_f32; len],
            head_scores,
        })
    }
}

fn copy_tile_rows(
    inputs: &mut [Vec<f32>],
    chunk: usize,
    src: &[f32],
    tile_start: usize,
    width: usize,
) {
    for (t, dst) in inputs.iter_mut().take(chunk).enumerate() {
        let tok = tile_start + t;
        dst.resize(width, 0.0);
        dst.copy_from_slice(&src[tok * width..(tok + 1) * width]);
    }
}

/// Plans target `(resized_width, resized_height)` aligned to `32` for `qwen3vl`.
///
/// Matches `qwen3vl` `calc_size_preserved_ratio`: both outputs are multiples of
/// `32` with area bounded within `[min_tokens * 1024, max_tokens * 1024]`.
///
/// # Errors
/// Rejects zero dimensions, empty token ranges, or aspect ratios exceeding `200:1`.
pub fn plan_image_grid(
    width: usize,
    height: usize,
    min_tokens: usize,
    max_tokens: usize,
) -> Result<(usize, usize)> {
    plan_grid_with_factor(width, height, 32, min_tokens, max_tokens)
}

#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "Bounded pixel counts and token bounds map cleanly between integer grids and binary64."
)]
fn plan_grid_with_factor(
    width: usize,
    height: usize,
    factor: usize,
    min_tokens: usize,
    max_tokens: usize,
) -> Result<(usize, usize)> {
    if width == 0 || height == 0 || factor == 0 || min_tokens == 0 || min_tokens > max_tokens {
        return Err(invalid("invalid vision grid planning bounds"));
    }
    let min_side = width.min(height);
    let max_side = width.max(height);
    if max_side > min_side.saturating_mul(200) {
        return Err(invalid("image aspect ratio exceeds 200:1 bound"));
    }
    let factor_sq = factor
        .checked_mul(factor)
        .ok_or_else(|| invalid("vision factor area overflow"))?;
    let min_pixels = min_tokens
        .checked_mul(factor_sq)
        .ok_or_else(|| invalid("vision min pixel bound overflow"))?;
    let max_pixels = max_tokens
        .checked_mul(factor_sq)
        .ok_or_else(|| invalid("vision max pixel bound overflow"))?;

    let f_f64 = factor as f64;
    let w_f64 = width as f64;
    let h_f64 = height as f64;
    let round_by = |dim: f64| -> usize { ((dim / f_f64).round() as usize).max(1) * factor };

    let mut h_bar = round_by(h_f64);
    let mut w_bar = round_by(w_f64);
    let area = h_bar.saturating_mul(w_bar);
    if area > max_pixels {
        let beta = ((w_f64 * h_f64) / (max_pixels as f64)).sqrt();
        h_bar = ((h_f64 / beta / f_f64).floor() as usize).max(1) * factor;
        w_bar = ((w_f64 / beta / f_f64).floor() as usize).max(1) * factor;
    } else if area < min_pixels {
        let beta = ((min_pixels as f64) / (w_f64 * h_f64)).sqrt();
        h_bar = ((h_f64 * beta / f_f64).ceil() as usize).max(1) * factor;
        w_bar = ((w_f64 * beta / f_f64).ceil() as usize).max(1) * factor;
    }
    while (w_bar / factor).saturating_mul(h_bar / factor) > max_tokens {
        if w_bar >= h_bar && w_bar > factor {
            w_bar -= factor;
        } else if h_bar > factor {
            h_bar -= factor;
        } else {
            return Err(invalid(
                "max_image_tokens too small for a single vision block",
            ));
        }
    }
    Ok((w_bar, h_bar))
}

type PatchEmbeddings = (Vec<f32>, Vec<(usize, usize)>);

#[expect(
    clippy::too_many_arguments,
    reason = "Pass explicit source/target geometry, normalization constants and worker pool."
)]
#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "Pixel coordinates and Catmull-Rom weights match reference floating preprocessing."
)]
fn resize_bicubic_normalize(
    rgb: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    mean: [f32; 3],
    std: [f32; 3],
    row_parallelism: &RowParallelism,
) -> Result<Vec<f32>> {
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return Err(invalid("invalid resize image geometry"));
    }
    let mut horiz = vec![0.0_f32; src_h * dst_w * 3];
    let scale_x = (src_w as f64) / (dst_w as f64);
    let mut x_taps = Vec::with_capacity(dst_w);
    for dx in 0..dst_w {
        let sx = (dx as f64 + 0.5).mul_add(scale_x, -0.5);
        let ix = sx.floor() as isize;
        let fx = sx - (ix as f64);
        let idx = [
            clamp_index(ix - 1, src_w),
            clamp_index(ix, src_w),
            clamp_index(ix + 1, src_w),
            clamp_index(ix + 2, src_w),
        ];
        let weights = catmull_rom_weights(fx);
        x_taps.push((idx, weights));
    }
    let horiz_row_len = dst_w * 3;
    row_parallelism.update_chunks(&mut horiz, horiz_row_len, |base_sy, chunk| {
        for (offset, dst_row) in chunk.chunks_exact_mut(horiz_row_len).enumerate() {
            let sy = base_sy + offset;
            let src_row = &rgb[sy * src_w * 3..(sy + 1) * src_w * 3];
            for (dst_px, &(idx, weights)) in dst_row.as_chunks_mut::<3>().0.iter_mut().zip(&x_taps)
            {
                for (c, dst_ch) in dst_px.iter_mut().enumerate() {
                    let mut sum = 0.0_f64;
                    for (&ix, &w) in idx.iter().zip(&weights) {
                        sum = f64::from(src_row[ix * 3 + c]).mul_add(w, sum);
                    }
                    *dst_ch = sum as f32;
                }
            }
        }
        Ok(())
    })?;

    let plane_len = dst_w * dst_h;
    let mut out = vec![0.0_f32; 3 * plane_len];
    let scale_y = (src_h as f64) / (dst_h as f64);
    let mut y_taps = Vec::with_capacity(dst_h);
    for dy in 0..dst_h {
        let sy = (dy as f64 + 0.5).mul_add(scale_y, -0.5);
        let iy = sy.floor() as isize;
        let fy = sy - (iy as f64);
        let idx = [
            clamp_index(iy - 1, src_h),
            clamp_index(iy, src_h),
            clamp_index(iy + 1, src_h),
            clamp_index(iy + 2, src_h),
        ];
        let weights = catmull_rom_weights(fy);
        y_taps.push((idx, weights));
    }
    let inv_255 = 1.0_f32 / 255.0_f32;
    for (c, plane) in out.chunks_exact_mut(plane_len).enumerate() {
        let mean_c = mean[c];
        let std_c = std[c];
        row_parallelism.update_chunks(plane, dst_w, |base_dy, chunk| {
            for (offset, dst_row) in chunk.chunks_exact_mut(dst_w).enumerate() {
                let &(idx, weights) = &y_taps[base_dy + offset];
                for (dx, dst_ch) in dst_row.iter_mut().enumerate() {
                    let mut sum = 0.0_f64;
                    for (&iy, &w) in idx.iter().zip(&weights) {
                        sum = f64::from(horiz[(iy * dst_w + dx) * 3 + c]).mul_add(w, sum);
                    }
                    let pixel = (sum as f32).round_ties_even().clamp(0.0, 255.0);
                    *dst_ch = pixel.mul_add(inv_255, -mean_c) / std_c;
                }
            }
            Ok(())
        })?;
    }
    Ok(out)
}

#[expect(
    clippy::as_conversions,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    reason = "Clamp signed tap index into valid image axis extent."
)]
fn clamp_index(idx: isize, len: usize) -> usize {
    let max = (len - 1) as isize;
    idx.clamp(0, max) as usize
}

fn catmull_rom_weights(t: f64) -> [f64; 4] {
    let mut w = [
        cubic_kernel(t + 1.0),
        cubic_kernel(t),
        cubic_kernel(1.0 - t),
        cubic_kernel(2.0 - t),
    ];
    let sum: f64 = w.iter().sum();
    if sum != 0.0 {
        for item in &mut w {
            *item /= sum;
        }
    }
    w
}

fn cubic_kernel(x: f64) -> f64 {
    let ax = x.abs();
    if ax <= 1.0 {
        (1.5_f64.mul_add(ax, -2.5) * ax).mul_add(ax, 1.0)
    } else if ax < 2.0 {
        ((-0.5_f64).mul_add(ax, 2.5).mul_add(ax, -4.0)).mul_add(ax, 2.0)
    } else {
        0.0
    }
}

#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "Align-corners bilinear interpolation of bounded positional embedding grid."
)]
#[expect(
    clippy::too_many_arguments,
    reason = "Pass explicit grid coordinates and reusable embedding scratch vector."
)]
fn bilinear_pos_embd(
    pos_embd: &[f32],
    grid_side: usize,
    hidden: usize,
    patch_cols: usize,
    patch_rows: usize,
    px: usize,
    py: usize,
    out: &mut [f32],
) -> Result<()> {
    if out.len() != hidden || grid_side == 0 {
        return Err(invalid(
            "invalid positional embedding interpolation geometry",
        ));
    }
    let fy = if patch_rows <= 1 || grid_side <= 1 {
        0.0_f64
    } else {
        (py as f64) * ((grid_side - 1) as f64) / ((patch_rows - 1) as f64)
    };
    let fx = if patch_cols <= 1 || grid_side <= 1 {
        0.0_f64
    } else {
        (px as f64) * ((grid_side - 1) as f64) / ((patch_cols - 1) as f64)
    };
    let y0 = (fy.floor() as usize).min(grid_side - 1);
    let x0 = (fx.floor() as usize).min(grid_side - 1);
    let y1 = (y0 + 1).min(grid_side - 1);
    let x1 = (x0 + 1).min(grid_side - 1);
    let wy1 = (fy - y0 as f64) as f32;
    let wy0 = 1.0_f32 - wy1;
    let wx1 = (fx - x0 as f64) as f32;
    let wx0 = 1.0_f32 - wx1;

    let w00 = wy0 * wx0;
    let w01 = wy0 * wx1;
    let w10 = wy1 * wx0;
    let w11 = wy1 * wx1;

    let r00 = &pos_embd[(y0 * grid_side + x0) * hidden..(y0 * grid_side + x0 + 1) * hidden];
    let r01 = &pos_embd[(y0 * grid_side + x1) * hidden..(y0 * grid_side + x1 + 1) * hidden];
    let r10 = &pos_embd[(y1 * grid_side + x0) * hidden..(y1 * grid_side + x0 + 1) * hidden];
    let r11 = &pos_embd[(y1 * grid_side + x1) * hidden..(y1 * grid_side + x1 + 1) * hidden];

    for (dst, ((&a, &b), (&c, &d))) in out
        .iter_mut()
        .zip(r00.iter().zip(r01).zip(r10.iter().zip(r11)))
    {
        let top = b.mul_add(w01, a * w00);
        let bottom = c.mul_add(w10, top);
        *dst = d.mul_add(w11, bottom);
    }
    Ok(())
}

#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "Bounded ViT patch coordinates convert cleanly to binary32 for trigonometric RoPE."
)]
fn build_vit_rope_cache(
    config: &VisionConfig,
    coords: &[(usize, usize)],
) -> Result<Vec<(f32, f32)>> {
    let d_head = config.hidden_size / config.num_heads;
    let n_dims = d_head / 2;
    let half = n_dims / 2;
    if d_head % 4 != 0 || half == 0 {
        return Err(invalid(
            "ViT head dimension must be a positive multiple of 4",
        ));
    }
    let theta_scale = VIT_ROPE_FREQ_BASE.powf(-2.0_f32 / (n_dims as f32));
    let mut cache = Vec::with_capacity(coords.len() * n_dims);
    for &(py, px) in coords {
        let mut theta_y = py as f32;
        let mut theta_x = px as f32;
        for ic in 0..n_dims {
            if ic == half {
                theta_x = px as f32;
            }
            let angle = if ic < half { theta_y } else { theta_x };
            cache.push((angle.cos(), angle.sin()));
            theta_y *= theta_scale;
            theta_x *= theta_scale;
        }
    }
    Ok(cache)
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "Retained for standalone RoPE unit test coverage alongside head-major apply_vit_rope_head."
    )
)]
fn apply_vit_rope_patch(values: &mut [f32], num_heads: usize, d_head: usize, freqs: &[(f32, f32)]) {
    for head in values.chunks_exact_mut(d_head).take(num_heads) {
        apply_vit_rope_head(head, d_head, freqs);
    }
}

fn apply_vit_rope_head(head: &mut [f32], d_head: usize, freqs: &[(f32, f32)]) {
    let n_dims = d_head / 2;
    for (ic, &(cos_theta, sin_theta)) in freqs.iter().enumerate() {
        let x0 = head[ic];
        let x1 = head[ic + n_dims];
        head[ic] = x1.mul_add(-sin_theta, x0 * cos_theta);
        head[ic + n_dims] = x1.mul_add(cos_theta, x0 * sin_theta);
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Reuse preallocated head-major scratch buffers across all 27 ViT layers without inner heap allocation."
)]
fn run_self_attention(
    block: &VitBlock,
    config: &VisionConfig,
    n_pos: usize,
    rope_cache: &[(f32, f32)],
    normed: &[f32],
    attn: &mut AttentionScratch,
    attn_out_buf: &mut [f32],
    projector: &mut TiledProjector,
    row_parallelism: &RowParallelism,
    backend: MatvecBackend,
) -> Result<()> {
    let d = config.hidden_size;
    let d_head = d / config.num_heads;
    fill_qkv_heads(
        block,
        config,
        n_pos,
        rope_cache,
        normed,
        attn,
        projector,
        row_parallelism,
        backend,
    )?;

    dispatch_self_attention_heads(
        config.num_heads,
        n_pos,
        d_head,
        attn,
        row_parallelism,
        backend,
    )?;

    project_attn_output(
        block,
        config,
        n_pos,
        &attn.head_out,
        attn_out_buf,
        projector,
        backend,
    )
}

struct AttnWorkUnit<'a> {
    head_tiles: usize,
    n_pos: usize,
    d_head: usize,
    tile_start: usize,
    tile_end: usize,
    out_chunk: &'a mut [f32],
    scores_buf: &'a mut [f32],
}

fn run_attn_work_unit(
    unit: &mut AttnWorkUnit<'_>,
    q_heads: &[f32],
    k_heads: &[f32],
    v_heads: &[f32],
    kq_scale: f32,
    backend: MatvecBackend,
) -> Result<()> {
    let head_plane = unit.n_pos * unit.d_head;
    let first_head = unit.tile_start / unit.head_tiles;
    let last_head = (unit.tile_end - 1) / unit.head_tiles;
    let mut out_cursor = &mut unit.out_chunk[..];
    for head in first_head..=last_head {
        let q_begin = if head == first_head {
            (unit.tile_start % unit.head_tiles) * 4
        } else {
            0
        };
        let q_end = if head == last_head {
            let rem = unit.tile_end % unit.head_tiles;
            if rem == 0 { unit.n_pos } else { rem * 4 }
        } else {
            unit.n_pos
        };
        let q_count = q_end - q_begin;
        let q_elems = q_count * unit.d_head;
        let (out_sub, rest_out) = out_cursor.split_at_mut(q_elems);
        out_cursor = rest_out;
        let q_start_elem = (head * unit.n_pos + q_begin) * unit.d_head;
        let q_sub = &q_heads[q_start_elem..q_start_elem + q_elems];
        let k_head = &k_heads[head * head_plane..(head + 1) * head_plane];
        let v_head = &v_heads[head * head_plane..(head + 1) * head_plane];
        attend_head_query_slice(
            q_sub,
            k_head,
            v_head,
            unit.n_pos,
            q_count,
            unit.d_head,
            kq_scale,
            unit.scores_buf,
            out_sub,
            backend,
        )?;
    }
    Ok(())
}

fn dispatch_self_attention_heads(
    num_heads: usize,
    n_pos: usize,
    d_head: usize,
    attn: &mut AttentionScratch,
    row_parallelism: &RowParallelism,
    backend: MatvecBackend,
) -> Result<()> {
    let head_plane = n_pos * d_head;
    let kq_scale = 1.0_f32 / head_sqrt(d_head)?;
    let score_stride = 4 * n_pos;
    let head_tiles = n_pos.div_ceil(4);
    let total_tiles = num_heads * head_tiles;
    let workers = row_parallelism.workers();

    if workers > 1 && total_tiles > 1 {
        let num_tasks = workers.min(total_tiles);
        let mut units: [Option<AttnWorkUnit<'_>>; crate::kernels::MAX_ROW_WORKERS] =
            std::array::from_fn(|_| None);
        let mut out_rem = &mut attn.head_out[..];
        let mut score_rem = &mut attn.head_scores[..];
        let mut prev_elem = 0_usize;
        for (w, slot) in units.iter_mut().take(num_tasks).enumerate() {
            let tile_start = (w * total_tiles) / num_tasks;
            let tile_end = ((w + 1) * total_tiles) / num_tasks;
            let end_elem = if tile_end == total_tiles {
                num_heads * head_plane
            } else {
                let h = tile_end / head_tiles;
                let q = (tile_end % head_tiles) * 4;
                (h * n_pos + q) * d_head
            };
            let (out_chunk, next_out) = out_rem.split_at_mut(end_elem - prev_elem);
            out_rem = next_out;
            let (scores_buf, next_score) = score_rem.split_at_mut(score_stride);
            score_rem = next_score;
            prev_elem = end_elem;
            *slot = Some(AttnWorkUnit {
                head_tiles,
                n_pos,
                d_head,
                tile_start,
                tile_end,
                out_chunk,
                scores_buf,
            });
        }
        let q_heads = &attn.q_heads;
        let k_heads = &attn.k_heads;
        let v_heads = &attn.v_heads;
        return row_parallelism.update_chunks(&mut units[..num_tasks], 1, |_, batch| {
            for unit in batch.iter_mut().flatten() {
                run_attn_work_unit(unit, q_heads, k_heads, v_heads, kq_scale, backend)?;
            }
            Ok(())
        });
    }

    let scores_buf = &mut attn.head_scores[..score_stride];
    for (head, out_head) in attn.head_out.chunks_exact_mut(head_plane).enumerate() {
        let q_head = &attn.q_heads[head * head_plane..(head + 1) * head_plane];
        let k_head = &attn.k_heads[head * head_plane..(head + 1) * head_plane];
        let v_head = &attn.v_heads[head * head_plane..(head + 1) * head_plane];
        attend_head_four_queries(
            q_head, k_head, v_head, n_pos, d_head, kq_scale, scores_buf, out_head, backend,
        )?;
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "Keep tiled QKV bias/RoPE scatter parallel across attention heads."
)]
fn fill_qkv_heads(
    block: &VitBlock,
    config: &VisionConfig,
    n_pos: usize,
    rope_cache: &[(f32, f32)],
    normed: &[f32],
    attn: &mut AttentionScratch,
    projector: &mut TiledProjector,
    row_parallelism: &RowParallelism,
    backend: MatvecBackend,
) -> Result<()> {
    let d = config.hidden_size;
    let d_head = d / config.num_heads;
    let n_dims = d_head / 2;
    let head_plane = n_pos * d_head;
    let pinned_qkv =
        !matches!(backend, MatvecBackend::Scalar) && block.attn_qkv_w.kind == crate::kernels::F16;
    for tile_start in (0..n_pos).step_by(MAX_TILE_TOKENS) {
        let chunk = (n_pos - tile_start).min(MAX_TILE_TOKENS);
        copy_tile_rows(&mut projector.inputs, chunk, normed, tile_start, d);
        projector.tile.project_with_previous(
            None,
            &block.attn_qkv_w,
            &projector.inputs[..chunk],
            &mut projector.outputs[..chunk],
            pinned_qkv,
            &|| Ok(()),
        )?;
        let proj_rows = &projector.outputs[..chunk];
        row_parallelism.update_triple_chunks(
            &mut attn.q_heads,
            head_plane,
            &mut attn.k_heads,
            head_plane,
            &mut attn.v_heads,
            head_plane,
            |base_head, q_chunk, k_chunk, v_chunk| {
                for (offset, ((q_head, k_head), v_head)) in q_chunk
                    .chunks_exact_mut(head_plane)
                    .zip(k_chunk.chunks_exact_mut(head_plane))
                    .zip(v_chunk.chunks_exact_mut(head_plane))
                    .enumerate()
                {
                    let head = base_head + offset;
                    let q_off = head * d_head;
                    let k_off = d + q_off;
                    let v_off = 2 * d + q_off;
                    for (t, proj_row) in proj_rows.iter().enumerate() {
                        let tok = tile_start + t;
                        let freqs = &rope_cache[tok * n_dims..(tok + 1) * n_dims];
                        let tok_off = tok * d_head;

                        let q_dst = &mut q_head[tok_off..tok_off + d_head];
                        for (dst, (&val, &bias)) in q_dst.iter_mut().zip(
                            proj_row[q_off..q_off + d_head]
                                .iter()
                                .zip(&block.attn_qkv_b[q_off..q_off + d_head]),
                        ) {
                            *dst = val + bias;
                        }
                        apply_vit_rope_head(q_dst, d_head, freqs);

                        let k_dst = &mut k_head[tok_off..tok_off + d_head];
                        for (dst, (&val, &bias)) in k_dst.iter_mut().zip(
                            proj_row[k_off..k_off + d_head]
                                .iter()
                                .zip(&block.attn_qkv_b[k_off..k_off + d_head]),
                        ) {
                            *dst = val + bias;
                        }
                        apply_vit_rope_head(k_dst, d_head, freqs);

                        let v_dst = &mut v_head[tok_off..tok_off + d_head];
                        for (dst, (&val, &bias)) in v_dst.iter_mut().zip(
                            proj_row[v_off..v_off + d_head]
                                .iter()
                                .zip(&block.attn_qkv_b[v_off..v_off + d_head]),
                        ) {
                            *dst = val + bias;
                        }
                    }
                }
                Ok(())
            },
        )?;
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "Keep 4-query tiled ViT attention kernels self-contained with caller-owned scratch."
)]
fn attend_head_four_queries(
    q_head: &[f32],
    k_head: &[f32],
    v_head: &[f32],
    n_pos: usize,
    d_head: usize,
    kq_scale: f32,
    scores_buf: &mut [f32],
    out_head: &mut [f32],
    backend: MatvecBackend,
) -> Result<()> {
    attend_head_query_slice(
        q_head, k_head, v_head, n_pos, n_pos, d_head, kq_scale, scores_buf, out_head, backend,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "Execute a contiguous query tile slice for one head with caller-owned scratch."
)]
fn attend_head_query_slice(
    q_slice: &[f32],
    k_head: &[f32],
    v_head: &[f32],
    n_pos: usize,
    q_count: usize,
    d_head: usize,
    kq_scale: f32,
    scores_buf: &mut [f32],
    out_slice: &mut [f32],
    backend: MatvecBackend,
) -> Result<()> {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if !matches!(backend, MatvecBackend::Scalar)
        && let Some(neon) = fearless_simd::Level::baseline().as_neon()
    {
        return attend_head_four_queries_neon(
            neon, q_slice, k_head, v_head, n_pos, q_count, d_head, kq_scale, scores_buf, out_slice,
        );
    }
    let _ = backend;

    let quad_end = (q_count / 4) * 4;
    let (s0, rest) = scores_buf.split_at_mut(n_pos);
    let (s1, rest) = rest.split_at_mut(n_pos);
    let (s2, s3) = rest.split_at_mut(n_pos);
    let s3 = &mut s3[..n_pos];

    for q_base in (0..quad_end).step_by(4) {
        let q0 = &q_slice[q_base * d_head..(q_base + 1) * d_head];
        let q1 = &q_slice[(q_base + 1) * d_head..(q_base + 2) * d_head];
        let q2 = &q_slice[(q_base + 2) * d_head..(q_base + 3) * d_head];
        let q3 = &q_slice[(q_base + 3) * d_head..(q_base + 4) * d_head];
        for (k_tok, k) in k_head.chunks_exact(d_head).enumerate() {
            let (mut d0, mut d1, mut d2, mut d3) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
            for ((((&qv0, &qv1), &qv2), &qv3), &kv) in q0.iter().zip(q1).zip(q2).zip(q3).zip(k) {
                let kv_f64 = f64::from(kv);
                d0 = f64::from(qv0).mul_add(kv_f64, d0);
                d1 = f64::from(qv1).mul_add(kv_f64, d1);
                d2 = f64::from(qv2).mul_add(kv_f64, d2);
                d3 = f64::from(qv3).mul_add(kv_f64, d3);
            }
            s0[k_tok] = narrow_f64_to_f32(d0)? * kq_scale;
            s1[k_tok] = narrow_f64_to_f32(d1)? * kq_scale;
            s2[k_tok] = narrow_f64_to_f32(d2)? * kq_scale;
            s3[k_tok] = narrow_f64_to_f32(d3)? * kq_scale;
        }
        softmax(s0)?;
        softmax(s1)?;
        softmax(s2)?;
        softmax(s3)?;
        let (o0, rest) = out_slice[q_base * d_head..(q_base + 4) * d_head].split_at_mut(d_head);
        let (o1, rest) = rest.split_at_mut(d_head);
        let (o2, o3) = rest.split_at_mut(d_head);
        o0.fill(0.0);
        o1.fill(0.0);
        o2.fill(0.0);
        o3.fill(0.0);
        for (v_tok, v) in v_head.chunks_exact(d_head).enumerate() {
            let (p0, p1, p2, p3) = (s0[v_tok], s1[v_tok], s2[v_tok], s3[v_tok]);
            for ((((a0, a1), a2), a3), &vv) in o0
                .iter_mut()
                .zip(o1.iter_mut())
                .zip(o2.iter_mut())
                .zip(o3.iter_mut())
                .zip(v)
            {
                *a0 = p0.mul_add(vv, *a0);
                *a1 = p1.mul_add(vv, *a1);
                *a2 = p2.mul_add(vv, *a2);
                *a3 = p3.mul_add(vv, *a3);
            }
        }
    }

    for q_tok in quad_end..q_count {
        let q = &q_slice[q_tok * d_head..(q_tok + 1) * d_head];
        let out_row = &mut out_slice[q_tok * d_head..(q_tok + 1) * d_head];
        attend_single_query(q, k_head, v_head, d_head, kq_scale, s0, out_row)?;
    }
    Ok(())
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
#[inline]
fn reduce4_f32x4(
    a0: f32x4<Neon>,
    a1: f32x4<Neon>,
    a2: f32x4<Neon>,
    a3: f32x4<Neon>,
) -> f32x4<Neon> {
    let p01 = a0.unzip_low(a1) + a0.unzip_high(a1);
    let p23 = a2.unzip_low(a3) + a2.unzip_high(a3);
    p01.unzip_low(p23) + p01.unzip_high(p23)
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
fn softmax_four_queries_max_neon(neon: Neon, scores: [&[f32]; 4]) -> Option<[f32; 4]> {
    let [s0, s1, s2, s3] = scores;
    let (c0, t0) = s0.as_chunks::<4>();
    let (c1, t1) = s1.as_chunks::<4>();
    let (c2, t2) = s2.as_chunks::<4>();
    let (c3, t3) = s3.as_chunks::<4>();

    let mut max_v = [f32x4::splat(neon, f32::NEG_INFINITY); 4];
    let mut all_finite = f32x4::splat(neon, 0.0).is_finite();
    for (((v0_sl, v1_sl), v2_sl), v3_sl) in c0.iter().zip(c1).zip(c2).zip(c3) {
        let v0 = f32x4::from_slice(neon, v0_sl);
        let v1 = f32x4::from_slice(neon, v1_sl);
        let v2 = f32x4::from_slice(neon, v2_sl);
        let v3 = f32x4::from_slice(neon, v3_sl);
        all_finite &= (v0.is_finite() & v1.is_finite()) & (v2.is_finite() & v3.is_finite());
        max_v[0] = max_v[0].max(v0);
        max_v[1] = max_v[1].max(v1);
        max_v[2] = max_v[2].max(v2);
        max_v[3] = max_v[3].max(v3);
    }

    let mut tail_finite = true;
    let mut tail_max = [f32::NEG_INFINITY; 4];
    for (((&a0, &a1), &a2), &a3) in t0.iter().zip(t1).zip(t2).zip(t3) {
        tail_finite &= a0.is_finite() && a1.is_finite() && a2.is_finite() && a3.is_finite();
        tail_max[0] = tail_max[0].max(a0);
        tail_max[1] = tail_max[1].max(a1);
        tail_max[2] = tail_max[2].max(a2);
        tail_max[3] = tail_max[3].max(a3);
    }
    if !all_finite.all_true() || !tail_finite {
        return None;
    }

    Some(std::array::from_fn(|row| {
        let a = max_v[row].to_array();
        a[0].max(a[1]).max(a[2]).max(a[3]).max(tail_max[row])
    }))
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Vectorized 4-row softmax accumulates lane sums in binary64 before inverse scaling."
)]
fn softmax_four_queries_neon(neon: Neon, scores: [&mut [f32]; 4]) -> Result<()> {
    let [s0, s1, s2, s3] = scores;
    let n = s0.len();
    let Some(max_vals) = (n >= 4 && s1.len() == n && s2.len() == n && s3.len() == n)
        .then(|| softmax_four_queries_max_neon(neon, [s0, s1, s2, s3]))
        .flatten()
    else {
        softmax(s0)?;
        softmax(s1)?;
        softmax(s2)?;
        softmax(s3)?;
        return Ok(());
    };

    let (c0, t0) = s0.as_chunks_mut::<4>();
    let (c1, t1) = s1.as_chunks_mut::<4>();
    let (c2, t2) = s2.as_chunks_mut::<4>();
    let (c3, t3) = s3.as_chunks_mut::<4>();

    let m_vec = max_vals.map(|m| f32x4::splat(neon, m));
    let mut sum_v = [f32x4::splat(neon, 0.0); 4];
    for (((v0_sl, v1_sl), v2_sl), v3_sl) in c0
        .iter_mut()
        .zip(c1.iter_mut())
        .zip(c2.iter_mut())
        .zip(c3.iter_mut())
    {
        for (sl, (&mv, sv)) in [v0_sl, v1_sl, v2_sl, v3_sl]
            .into_iter()
            .zip(m_vec.iter().zip(sum_v.iter_mut()))
        {
            let e = vector_exp_neg_x4(neon, f32x4::from_slice(neon, sl) - mv);
            e.store_slice(sl);
            *sv += e;
        }
    }

    let mut tail_sum = [0.0_f64; 4];
    for (((a0, a1), a2), a3) in t0
        .iter_mut()
        .zip(t1.iter_mut())
        .zip(t2.iter_mut())
        .zip(t3.iter_mut())
    {
        for ((val, &max_val), sum_slot) in [a0, a1, a2, a3]
            .into_iter()
            .zip(max_vals.iter())
            .zip(tail_sum.iter_mut())
        {
            *val = (*val - max_val).exp();
            *sum_slot += f64::from(*val);
        }
    }

    let inv_vals: [f32; 4] = std::array::from_fn(|row| {
        let a = sum_v[row].to_array();
        let total =
            f64::from(a[0]) + f64::from(a[1]) + f64::from(a[2]) + f64::from(a[3]) + tail_sum[row];
        (1.0_f64 / total) as f32
    });
    let inv_vec = inv_vals.map(|inv| f32x4::splat(neon, inv));
    for (((v0_sl, v1_sl), v2_sl), v3_sl) in c0
        .iter_mut()
        .zip(c1.iter_mut())
        .zip(c2.iter_mut())
        .zip(c3.iter_mut())
    {
        for (sl, &iv) in [v0_sl, v1_sl, v2_sl, v3_sl].into_iter().zip(inv_vec.iter()) {
            (f32x4::from_slice(neon, sl) * iv).store_slice(sl);
        }
    }
    for (((a0, a1), a2), a3) in t0
        .iter_mut()
        .zip(t1.iter_mut())
        .zip(t2.iter_mut())
        .zip(t3.iter_mut())
    {
        for (val, &inv) in [a0, a1, a2, a3].into_iter().zip(inv_vals.iter()) {
            *val *= inv;
        }
    }
    Ok(())
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
#[expect(
    clippy::too_many_arguments,
    reason = "Vectorized 4-query tiled ViT attention on baseline NEON."
)]
fn attend_head_four_queries_neon(
    neon: Neon,
    q_slice: &[f32],
    k_head: &[f32],
    v_head: &[f32],
    n_pos: usize,
    q_count: usize,
    d_head: usize,
    kq_scale: f32,
    scores_buf: &mut [f32],
    out_slice: &mut [f32],
) -> Result<()> {
    let quad_end = (q_count / 4) * 4;
    let (s0, rest) = scores_buf.split_at_mut(n_pos);
    let (s1, rest) = rest.split_at_mut(n_pos);
    let (s2, s3) = rest.split_at_mut(n_pos);
    let s3 = &mut s3[..n_pos];

    for q_base in (0..quad_end).step_by(4) {
        let q0 = &q_slice[q_base * d_head..(q_base + 1) * d_head];
        let q1 = &q_slice[(q_base + 1) * d_head..(q_base + 2) * d_head];
        let q2 = &q_slice[(q_base + 2) * d_head..(q_base + 3) * d_head];
        let q3 = &q_slice[(q_base + 3) * d_head..(q_base + 4) * d_head];
        qk_four_queries_neon(
            neon,
            [q0, q1, q2, q3],
            k_head,
            n_pos,
            d_head,
            kq_scale,
            &mut [s0, s1, s2, s3],
        )?;
        softmax_four_queries_neon(neon, [s0, s1, s2, s3])?;
        let (o0, rest) = out_slice[q_base * d_head..(q_base + 4) * d_head].split_at_mut(d_head);
        let (o1, rest) = rest.split_at_mut(d_head);
        let (o2, o3) = rest.split_at_mut(d_head);
        pv_four_queries_neon(
            neon,
            [s0, s1, s2, s3],
            v_head,
            n_pos,
            d_head,
            [o0, o1, o2, o3],
        );
    }

    for q_tok in quad_end..q_count {
        let q = &q_slice[q_tok * d_head..(q_tok + 1) * d_head];
        let out_row = &mut out_slice[q_tok * d_head..(q_tok + 1) * d_head];
        attend_single_query(q, k_head, v_head, d_head, kq_scale, s0, out_row)?;
    }
    Ok(())
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
fn qk_four_queries_neon(
    neon: Neon,
    qs: [&[f32]; 4],
    k_head: &[f32],
    n_pos: usize,
    d_head: usize,
    kq_scale: f32,
    scores: &mut [&mut [f32]; 4],
) -> Result<()> {
    let k_quad_end = (n_pos / 4) * 4;
    let d_vec_end = (d_head / 4) * 4;
    let scale_vec = f32x4::splat(neon, kq_scale);
    let mut finite_mask = scale_vec.is_finite();

    for k_base in (0..k_quad_end).step_by(4) {
        let ks = [
            &k_head[k_base * d_head..(k_base + 1) * d_head],
            &k_head[(k_base + 1) * d_head..(k_base + 2) * d_head],
            &k_head[(k_base + 2) * d_head..(k_base + 3) * d_head],
            &k_head[(k_base + 3) * d_head..(k_base + 4) * d_head],
        ];
        let mut acc = [[f32x4::splat(neon, 0.0); 4]; 4];
        for d in (0..d_vec_end).step_by(4) {
            let qv = [
                f32x4::from_slice(neon, &qs[0][d..d + 4]),
                f32x4::from_slice(neon, &qs[1][d..d + 4]),
                f32x4::from_slice(neon, &qs[2][d..d + 4]),
                f32x4::from_slice(neon, &qs[3][d..d + 4]),
            ];
            let kv = [
                f32x4::from_slice(neon, &ks[0][d..d + 4]),
                f32x4::from_slice(neon, &ks[1][d..d + 4]),
                f32x4::from_slice(neon, &ks[2][d..d + 4]),
                f32x4::from_slice(neon, &ks[3][d..d + 4]),
            ];
            for qi in 0..4 {
                for ki in 0..4 {
                    acc[qi][ki] = qv[qi].mul_add_precise(kv[ki], acc[qi][ki]);
                }
            }
        }
        for qi in 0..4 {
            let mut dot = reduce4_f32x4(acc[qi][0], acc[qi][1], acc[qi][2], acc[qi][3]);
            if d_vec_end < d_head {
                let mut arr = dot.to_array();
                for (ki, dst) in arr.iter_mut().enumerate() {
                    for d in d_vec_end..d_head {
                        *dst = qs[qi][d].mul_add(ks[ki][d], *dst);
                    }
                }
                dot = f32x4::from_slice(neon, &arr);
            }
            let scaled = dot * scale_vec;
            finite_mask &= scaled.is_finite();
            scaled.store_slice(&mut scores[qi][k_base..k_base + 4]);
        }
    }

    for k_tok in k_quad_end..n_pos {
        let k = &k_head[k_tok * d_head..(k_tok + 1) * d_head];
        let mut acc = [f32x4::splat(neon, 0.0); 4];
        for d in (0..d_vec_end).step_by(4) {
            let kv = f32x4::from_slice(neon, &k[d..d + 4]);
            for (qi, &q) in qs.iter().enumerate() {
                acc[qi] = f32x4::from_slice(neon, &q[d..d + 4]).mul_add_precise(kv, acc[qi]);
            }
        }
        let mut arr = reduce4_f32x4(acc[0], acc[1], acc[2], acc[3]).to_array();
        for (qi, dst) in arr.iter_mut().enumerate() {
            for d in d_vec_end..d_head {
                *dst = qs[qi][d].mul_add(k[d], *dst);
            }
            *dst *= kq_scale;
            if !dst.is_finite() {
                return Err(invalid("nonfinite ViT attention dot product"));
            }
            scores[qi][k_tok] = *dst;
        }
    }

    if !finite_mask.all_true() {
        return Err(invalid("nonfinite ViT attention dot product"));
    }
    Ok(())
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
#[expect(
    clippy::too_many_arguments,
    reason = "Tile P*V over k-chunks resident in L1D across d_head sub-tiles."
)]
fn pv_tile_chunks<const W: usize>(
    neon: Neon,
    scores: [&[f32]; 4],
    v_head: &[f32],
    k_start: usize,
    k_end: usize,
    d_head: usize,
    d_off: usize,
    out: &mut [&mut [f32]; 4],
) {
    let mut acc = if k_start == 0 {
        [[f32x4::splat(neon, 0.0); W]; 4]
    } else {
        std::array::from_fn(|row| {
            std::array::from_fn(|c| {
                f32x4::from_slice(neon, &out[row][d_off + c * 4..d_off + (c + 1) * 4])
            })
        })
    };
    for v_tok in k_start..k_end {
        let p0 = f32x4::splat(neon, scores[0][v_tok]);
        let p1 = f32x4::splat(neon, scores[1][v_tok]);
        let p2 = f32x4::splat(neon, scores[2][v_tok]);
        let p3 = f32x4::splat(neon, scores[3][v_tok]);
        let v_base = v_tok * d_head + d_off;
        for c in 0..W {
            let vv = f32x4::from_slice(neon, &v_head[v_base + c * 4..v_base + (c + 1) * 4]);
            acc[0][c] = p0.mul_add_precise(vv, acc[0][c]);
            acc[1][c] = p1.mul_add_precise(vv, acc[1][c]);
            acc[2][c] = p2.mul_add_precise(vv, acc[2][c]);
            acc[3][c] = p3.mul_add_precise(vv, acc[3][c]);
        }
    }
    for (dst, row_acc) in out.iter_mut().zip(acc) {
        for (c, vec) in row_acc.into_iter().enumerate() {
            vec.store_slice(&mut dst[d_off + c * 4..d_off + (c + 1) * 4]);
        }
    }
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
fn pv_four_queries_neon(
    neon: Neon,
    scores: [&[f32]; 4],
    v_head: &[f32],
    n_pos: usize,
    d_head: usize,
    mut out: [&mut [f32]; 4],
) {
    let d_16 = (d_head / 16) * 16;
    let d_4 = (d_head / 4) * 4;
    // Tile v_head over k chunks fitting in L1D (~16 KiB) so v_head is read once
    // across all d_head sub-tiles (e.g., 5 sub-tiles for d_head = 72).
    let k_tile = (4096 / d_head.max(1)).clamp(16, 64);
    if d_4 < d_head {
        for row in &mut out {
            row[d_4..d_head].fill(0.0);
        }
    }
    for k_start in (0..n_pos).step_by(k_tile) {
        let k_end = (k_start + k_tile).min(n_pos);
        for d_off in (0..d_16).step_by(16) {
            pv_tile_chunks::<4>(
                neon, scores, v_head, k_start, k_end, d_head, d_off, &mut out,
            );
        }
        match (d_4 - d_16) / 4 {
            1 => pv_tile_chunks::<1>(neon, scores, v_head, k_start, k_end, d_head, d_16, &mut out),
            2 => pv_tile_chunks::<2>(neon, scores, v_head, k_start, k_end, d_head, d_16, &mut out),
            3 => pv_tile_chunks::<3>(neon, scores, v_head, k_start, k_end, d_head, d_16, &mut out),
            _ => {}
        }
        if d_4 < d_head {
            for v_tok in k_start..k_end {
                let v = &v_head[v_tok * d_head..(v_tok + 1) * d_head];
                let (p0, p1, p2, p3) = (
                    scores[0][v_tok],
                    scores[1][v_tok],
                    scores[2][v_tok],
                    scores[3][v_tok],
                );
                for d in d_4..d_head {
                    let vv = v[d];
                    out[0][d] = p0.mul_add(vv, out[0][d]);
                    out[1][d] = p1.mul_add(vv, out[1][d]);
                    out[2][d] = p2.mul_add(vv, out[2][d]);
                    out[3][d] = p3.mul_add(vv, out[3][d]);
                }
            }
        }
    }
}

fn attend_single_query(
    q: &[f32],
    k_head: &[f32],
    v_head: &[f32],
    d_head: usize,
    kq_scale: f32,
    scores: &mut [f32],
    out_row: &mut [f32],
) -> Result<()> {
    for (k_tok, k) in k_head.chunks_exact(d_head).enumerate() {
        let mut dot = 0.0_f64;
        for (&qv, &kv) in q.iter().zip(k) {
            dot = f64::from(qv).mul_add(f64::from(kv), dot);
        }
        scores[k_tok] = narrow_f64_to_f32(dot)? * kq_scale;
    }
    softmax(scores)?;
    out_row.fill(0.0);
    for (v_tok, &prob) in scores.iter().enumerate() {
        let v = &v_head[v_tok * d_head..(v_tok + 1) * d_head];
        for (acc, &vv) in out_row.iter_mut().zip(v) {
            *acc = prob.mul_add(vv, *acc);
        }
    }
    Ok(())
}

fn project_attn_output(
    block: &VitBlock,
    config: &VisionConfig,
    n_pos: usize,
    head_out: &[f32],
    attn_out_buf: &mut [f32],
    projector: &mut TiledProjector,
    backend: MatvecBackend,
) -> Result<()> {
    let d = config.hidden_size;
    let d_head = d / config.num_heads;
    let head_plane = n_pos * d_head;
    let pinned_out =
        !matches!(backend, MatvecBackend::Scalar) && block.attn_out_w.kind == crate::kernels::F16;
    for tile_start in (0..n_pos).step_by(MAX_TILE_TOKENS) {
        let chunk = (n_pos - tile_start).min(MAX_TILE_TOKENS);
        for (t, dst) in projector.inputs.iter_mut().take(chunk).enumerate() {
            let tok = tile_start + t;
            dst.resize(d, 0.0);
            for head in 0..config.num_heads {
                let src_off = head * head_plane + tok * d_head;
                dst[head * d_head..(head + 1) * d_head]
                    .copy_from_slice(&head_out[src_off..src_off + d_head]);
            }
        }
        projector.tile.project_with_previous(
            None,
            &block.attn_out_w,
            &projector.inputs[..chunk],
            &mut projector.outputs[..chunk],
            pinned_out,
            &|| Ok(()),
        )?;
        for (t, proj_row) in projector.outputs[..chunk].iter().enumerate() {
            let out_tok = &mut attn_out_buf[(tile_start + t) * d..(tile_start + t + 1) * d];
            for (dst, (&val, &bias)) in out_tok
                .iter_mut()
                .zip(proj_row.iter().zip(&block.attn_out_b))
            {
                *dst = val + bias;
            }
        }
    }
    Ok(())
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
#[inline]
fn vector_exp_neg_x4(neon: Neon, neg_abs: f32x4<Neon>) -> f32x4<Neon> {
    use fearless_simd::u32x4;

    let min_arg = f32x4::splat(neon, -87.0);
    let clamped = neg_abs.simd_lt(min_arg).select(min_arg, neg_abs);
    let magic = f32x4::splat(neon, 12_582_912.0);
    let z = clamped.mul_add(f32x4::splat(neon, f32::from_bits(0x3fb8_aa3b)), magic);
    let n = z - magic;
    let neg_n = -n;
    let remainder = neg_n.mul_add(f32x4::splat(neon, f32::from_bits(0x3f31_7200)), clamped);
    let remainder = neg_n.mul_add(f32x4::splat(neon, f32::from_bits(0x35bf_be8e)), remainder);
    let exponent = z.bitcast::<u32x4<Neon>>() << 23;
    let k = (exponent + u32x4::splat(neon, 1.0_f32.to_bits())).bitcast::<f32x4<Neon>>();
    let square = remainder * remainder;
    let inner = f32x4::splat(neon, f32::from_bits(0x3c07_2010))
        .mul_add(remainder, f32x4::splat(neon, f32::from_bits(0x3d2b_9f17)))
        .mul_add(
            square,
            f32x4::splat(neon, f32::from_bits(0x3e2a_af33))
                .mul_add(remainder, f32x4::splat(neon, f32::from_bits(0x3eff_fedb))),
        );
    let polynomial = inner.mul_add(
        square,
        f32x4::splat(neon, f32::from_bits(0x3f7f_fff6)) * remainder,
    );
    let exp_val = polynomial.mul_add(k, k);
    // Clamp exp(-inf / very negative) underflow cleanly to 0 while keeping NaNs.
    neg_abs
        .simd_lt(min_arg)
        .select(f32x4::splat(neon, 0.0), exp_val)
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
#[inline]
fn gelu_tanh_x4(neon: Neon, x: f32x4<Neon>) -> f32x4<Neon> {
    let one = f32x4::splat(neon, 1.0);
    let zero = f32x4::splat(neon, 0.0);
    let coeff = f32x4::splat(neon, 0.044_715);
    let sqrt_2_over_pi = f32x4::splat(neon, SQRT_2_OVER_PI);
    let z = sqrt_2_over_pi * x * coeff.mul_add_precise(x * x, one);
    let e = vector_exp_neg_x4(neon, f32x4::splat(neon, -2.0) * z.abs());
    let num = z.simd_ge(zero).select(one, e);
    x * num / (one + e)
}

fn apply_bias_gelu_rows(
    rows: &mut [Vec<f32>],
    bias: &[f32],
    row_parallelism: &RowParallelism,
    backend: MatvecBackend,
) -> Result<()> {
    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if !matches!(backend, MatvecBackend::Scalar)
        && let Some(neon) = fearless_simd::Level::baseline().as_neon()
    {
        return row_parallelism.update_chunks(rows, 1, |_, row_chunk| {
            let (bias_chunks, bias_tail) = bias.as_chunks::<4>();
            for proj_row in row_chunk {
                let (dst_chunks, dst_tail) = proj_row.as_chunks_mut::<4>();
                for (dst, b) in dst_chunks.iter_mut().zip(bias_chunks) {
                    let x = f32x4::from_slice(neon, dst) + f32x4::from_slice(neon, b);
                    gelu_tanh_x4(neon, x).store_slice(dst);
                }
                for (dst, &b) in dst_tail.iter_mut().zip(bias_tail) {
                    *dst = gelu_tanh(*dst + b);
                }
            }
            Ok(())
        });
    }
    let _ = backend;

    row_parallelism.update_chunks(rows, 1, |_, row_chunk| {
        for proj_row in row_chunk {
            for (dst, &b) in proj_row.iter_mut().zip(bias) {
                *dst = gelu_tanh(*dst + b);
            }
        }
        Ok(())
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "Explicit ViT MLP scratch, tile buffers and row-worker pool."
)]
fn run_mlp_block(
    block: &VitBlock,
    config: &VisionConfig,
    n_pos: usize,
    normed: &[f32],
    residual: &[f32],
    hidden: &mut [f32],
    projector: &mut TiledProjector,
    row_parallelism: &RowParallelism,
    backend: MatvecBackend,
) -> Result<()> {
    let d = config.hidden_size;
    let pinned_up =
        !matches!(backend, MatvecBackend::Scalar) && block.ffn_up_w.kind == crate::kernels::F16;
    let pinned_down =
        !matches!(backend, MatvecBackend::Scalar) && block.ffn_down_w.kind == crate::kernels::F16;
    for tile_start in (0..n_pos).step_by(MAX_TILE_TOKENS) {
        let chunk = (n_pos - tile_start).min(MAX_TILE_TOKENS);
        copy_tile_rows(&mut projector.inputs, chunk, normed, tile_start, d);
        projector.tile.project_with_previous(
            None,
            &block.ffn_up_w,
            &projector.inputs[..chunk],
            &mut projector.outputs[..chunk],
            pinned_up,
            &|| Ok(()),
        )?;
        apply_bias_gelu_rows(
            &mut projector.outputs[..chunk],
            &block.ffn_up_b,
            row_parallelism,
            backend,
        )?;
        std::mem::swap(&mut projector.inputs, &mut projector.outputs);
        projector.tile.project_with_previous(
            None,
            &block.ffn_down_w,
            &projector.inputs[..chunk],
            &mut projector.outputs[..chunk],
            pinned_down,
            &|| Ok(()),
        )?;
        for (t, proj_row) in projector.outputs[..chunk].iter().enumerate() {
            let tok = tile_start + t;
            let res = &residual[tok * d..(tok + 1) * d];
            let out = &mut hidden[tok * d..(tok + 1) * d];
            for (dst, (&r, (&val, &bias))) in out
                .iter_mut()
                .zip(res.iter().zip(proj_row.iter().zip(&block.ffn_down_b)))
            {
                *dst = r + val + bias;
            }
        }
    }
    Ok(())
}

#[cfg(all(
    feature = "experimental-neon",
    target_arch = "aarch64",
    target_os = "windows",
    target_env = "msvc",
    target_feature = "neon"
))]
#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "Vectorized LayerNorm mean/variance accumulation in binary32 across 4-wide NEON lanes."
)]
fn layer_norm_row_neon(
    neon: Neon,
    src: &[f32],
    dst: &mut [f32],
    weight: &[f32],
    bias: &[f32],
    eps: f32,
) {
    let inv_cols = 1.0_f32 / (src.len() as f32);
    let (src8, src_rem8) = src.as_chunks::<8>();
    let mut sum0 = f32x4::splat(neon, 0.0);
    let mut sum1 = f32x4::splat(neon, 0.0);
    for chunk in src8 {
        sum0 += f32x4::from_slice(neon, &chunk[..4]);
        sum1 += f32x4::from_slice(neon, &chunk[4..]);
    }
    let (src4, src_tail) = src_rem8.as_chunks::<4>();
    for chunk in src4 {
        sum0 += f32x4::from_slice(neon, chunk);
    }
    let mut sum = (sum0 + sum1).reduce_sum();
    for &v in src_tail {
        sum += v;
    }
    let mean = sum * inv_cols;
    let mean_vec = f32x4::splat(neon, mean);

    let mut var0 = f32x4::splat(neon, 0.0);
    let mut var1 = f32x4::splat(neon, 0.0);
    for chunk in src8 {
        let d0 = f32x4::from_slice(neon, &chunk[..4]) - mean_vec;
        let d1 = f32x4::from_slice(neon, &chunk[4..]) - mean_vec;
        var0 = d0.mul_add_precise(d0, var0);
        var1 = d1.mul_add_precise(d1, var1);
    }
    for chunk in src4 {
        let d0 = f32x4::from_slice(neon, chunk) - mean_vec;
        var0 = d0.mul_add_precise(d0, var0);
    }
    let mut var = (var0 + var1).reduce_sum();
    for &v in src_tail {
        let diff = v - mean;
        var = diff.mul_add(diff, var);
    }
    let inv_std = 1.0_f32 / var.mul_add(inv_cols, eps).sqrt();
    let inv_std_vec = f32x4::splat(neon, inv_std);

    let (src4_all, src_tail_all) = src.as_chunks::<4>();
    let (dst4_all, dst_tail_all) = dst.as_chunks_mut::<4>();
    let (w4_all, w_tail_all) = weight.as_chunks::<4>();
    let (b4_all, b_tail_all) = bias.as_chunks::<4>();
    for (((s_chunk, d_chunk), w_chunk), b_chunk) in
        src4_all.iter().zip(dst4_all).zip(w4_all).zip(b4_all)
    {
        let normalized = (f32x4::from_slice(neon, s_chunk) - mean_vec) * inv_std_vec;
        let out = normalized.mul_add_precise(
            f32x4::from_slice(neon, w_chunk),
            f32x4::from_slice(neon, b_chunk),
        );
        out.store_slice(d_chunk);
    }
    for ((d, &s), (&w, &b)) in dst_tail_all
        .iter_mut()
        .zip(src_tail_all)
        .zip(w_tail_all.iter().zip(b_tail_all))
    {
        let normalized = (s - mean) * inv_std;
        *d = normalized.mul_add(w, b);
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Explicit dimensions, parameters and worker pool for row-parallel LayerNorm."
)]
#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "Standard LayerNorm mean/variance accumulation in binary64 rounded back to binary32."
)]
fn layer_norm_rows(
    input: &[f32],
    rows: usize,
    cols: usize,
    weight: &[f32],
    bias: &[f32],
    eps: f32,
    output: &mut [f32],
    row_parallelism: &RowParallelism,
    backend: MatvecBackend,
) -> Result<()> {
    if rows == 0
        || cols == 0
        || input.len() != rows * cols
        || output.len() != input.len()
        || weight.len() != cols
        || bias.len() != cols
    {
        return Err(invalid("invalid LayerNorm dimensions"));
    }

    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    if !matches!(backend, MatvecBackend::Scalar)
        && let Some(neon) = fearless_simd::Level::baseline().as_neon()
    {
        row_parallelism.update_chunks(output, cols, |base_row, chunk| {
            for (offset, dst) in chunk.chunks_exact_mut(cols).enumerate() {
                let r = base_row + offset;
                let src = &input[r * cols..(r + 1) * cols];
                layer_norm_row_neon(neon, src, dst, weight, bias, eps);
            }
            Ok(())
        })?;
        if !is_finite_slice(output) {
            return Err(invalid("nonfinite LayerNorm output"));
        }
        return Ok(());
    }
    let _ = backend;

    let inv_cols = 1.0_f64 / (cols as f64);
    row_parallelism.update_chunks(output, cols, |base_row, chunk| {
        for (offset, dst) in chunk.chunks_exact_mut(cols).enumerate() {
            let r = base_row + offset;
            let src = &input[r * cols..(r + 1) * cols];
            let mut sum = 0.0_f64;
            for &v in src {
                sum += f64::from(v);
            }
            let mean = sum * inv_cols;
            let mut var = 0.0_f64;
            for &v in src {
                let diff = f64::from(v) - mean;
                var = diff.mul_add(diff, var);
            }
            let inv_std = 1.0_f64 / var.mul_add(inv_cols, f64::from(eps)).sqrt();
            for ((d, &s), (&w, &b)) in dst.iter_mut().zip(src).zip(weight.iter().zip(bias)) {
                let normalized = ((f64::from(s) - mean) * inv_std) as f32;
                *d = normalized.mul_add(w, b);
            }
        }
        Ok(())
    })?;
    if !is_finite_slice(output) {
        return Err(invalid("nonfinite LayerNorm output"));
    }
    Ok(())
}

fn gelu_tanh(x: f32) -> f32 {
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * x * (0.044_715 * x).mul_add(x, 1.0)).tanh())
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Bounded attention dot product safely narrows to binary32."
)]
fn narrow_f64_to_f32(value: f64) -> Result<f32> {
    let out = value as f32;
    if !out.is_finite() {
        return Err(invalid("nonfinite ViT attention dot product"));
    }
    Ok(out)
}

#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "Head dimension square root for QK scaling."
)]
fn head_sqrt(d_head: usize) -> Result<f32> {
    if d_head == 0 {
        return Err(invalid("ViT head dimension cannot be zero"));
    }
    Ok((d_head as f32).sqrt())
}

fn sha256_bytes(bytes: &[u8]) -> Result<[u8; 32]> {
    let mut hasher = Hasher::new()?;
    hasher.update(bytes)?;
    Ok(hasher.finish()?)
}

fn sha256_rgb_image(rgb: &[u8], width: usize, height: usize) -> Result<[u8; 32]> {
    let mut hasher = Hasher::new()?;
    hasher.update(b"prism-vision-rgb-v1\0")?;
    hasher.update(&u64::try_from(width)?.to_le_bytes())?;
    hasher.update(&u64::try_from(height)?.to_le_bytes())?;
    hasher.update(rgb)?;
    Ok(hasher.finish()?)
}

fn parse_vision_config(model: &Model) -> Result<VisionConfig> {
    let md = &model.metadata;
    if md.get("general.architecture").and_then(|v| v.as_str()) != Some("clip")
        || md.get("general.type").and_then(|v| v.as_str()) != Some("mmproj")
        || md.get("clip.projector_type").and_then(|v| v.as_str()) != Some("qwen3vl_merger")
        || md
            .get("clip.has_vision_encoder")
            .and_then(bonsai_gguf::MetadataValue::as_bool)
            != Some(true)
    {
        return Err(invalid(
            "mmproj model is not a clip qwen3vl_merger vision encoder",
        ));
    }
    let meta_usize = |key: &str| -> Result<usize> {
        let raw = md
            .get(key)
            .and_then(bonsai_gguf::MetadataValue::as_u64)
            .ok_or_else(|| invalid(format!("missing integer vision metadata: {key}")))?;
        usize::try_from(raw).map_err(|_| invalid(format!("overflow in vision metadata: {key}")))
    };
    let patch_size = meta_usize("clip.vision.patch_size")?;
    let spatial_merge = meta_usize("clip.vision.spatial_merge_size")?;
    let hidden_size = meta_usize("clip.vision.embedding_length")?;
    let num_heads = meta_usize("clip.vision.attention.head_count")?;
    let intermediate_size = meta_usize("clip.vision.feed_forward_length")?;
    let num_layers = meta_usize("clip.vision.block_count")?;
    let projection_dim = meta_usize("clip.vision.projection_dim")?;
    if patch_size == 0
        || spatial_merge < 2
        || hidden_size == 0
        || num_heads == 0
        || !hidden_size.is_multiple_of(num_heads)
        || !(hidden_size / num_heads).is_multiple_of(4)
        || num_layers == 0
    {
        return Err(invalid("invalid ViT hyperparameter geometry"));
    }
    let eps = narrow_f64_to_f32(
        md.get("clip.vision.attention.layer_norm_epsilon")
            .and_then(bonsai_gguf::MetadataValue::as_f64)
            .unwrap_or(1e-6),
    )?;
    if !eps.is_finite() || eps <= 0.0 {
        return Err(invalid("invalid ViT LayerNorm epsilon"));
    }
    let mean = read_rgb_triple(md, "clip.vision.image_mean", [0.5, 0.5, 0.5])?;
    let std = read_rgb_triple(md, "clip.vision.image_std", [0.5, 0.5, 0.5])?;
    if std.iter().any(|&v| !v.is_finite() || v <= 0.0) {
        return Err(invalid("invalid ViT normalization std"));
    }

    let pos_tensor = model
        .tensors
        .get("v.position_embd.weight")
        .ok_or_else(|| invalid("missing v.position_embd.weight tensor"))?;
    let &[c0, c1] = pos_tensor.dimensions.as_slice() else {
        return Err(invalid("v.position_embd.weight must be 2-dimensional"));
    };
    if c0 != hidden_size || c1 == 0 {
        return Err(invalid("v.position_embd.weight dimension mismatch"));
    }
    let pos_grid_side = integer_sqrt(c1)
        .ok_or_else(|| invalid("v.position_embd.weight row count is not a square grid"))?;

    Ok(VisionConfig {
        patch_size,
        spatial_merge,
        hidden_size,
        num_heads,
        intermediate_size,
        num_layers,
        projection_dim,
        eps,
        mean,
        std,
        pos_grid_side,
    })
}

fn read_rgb_triple(md: &bonsai_gguf::Metadata, key: &str, default: [f32; 3]) -> Result<[f32; 3]> {
    let Some(val) = md.get(key) else {
        return Ok(default);
    };
    let Some(bonsai_gguf::MetadataArray::F32(vals)) = val.as_array() else {
        return Err(invalid(format!("expected F32 array for {key}")));
    };
    let &[a, b, c] = vals.as_slice() else {
        return Err(invalid(format!("expected 3 floats for {key}")));
    };
    if !a.is_finite() || !b.is_finite() || !c.is_finite() {
        return Err(invalid(format!("nonfinite float in {key}")));
    }
    Ok([a, b, c])
}

fn integer_sqrt(n: usize) -> Option<usize> {
    let mut candidate = 1_usize;
    while candidate.saturating_mul(candidate) < n {
        candidate += 1;
    }
    (candidate.saturating_mul(candidate) == n).then_some(candidate)
}

fn require_tensor<'a>(model: &'a Model, name: &str, expected_dims: &[usize]) -> Result<&'a Tensor> {
    let tensor = model
        .tensors
        .get(name)
        .ok_or_else(|| invalid(format!("missing vision tensor: {name}")))?;
    validate_tensor(tensor)?;
    if tensor.dimensions != expected_dims {
        return Err(invalid(format!("unexpected dimensions for {name}")));
    }
    Ok(tensor)
}

fn fuse_temporal_patch_weights(
    p0: &Tensor,
    p1: &Tensor,
    patch_pixels: usize,
    hidden_size: usize,
) -> Result<Tensor> {
    if p0.kind != 0 || p1.kind != 0 {
        return Err(invalid("patch embedding weights must be F32"));
    }
    let expected_elements = patch_pixels
        .checked_mul(hidden_size)
        .ok_or_else(|| invalid("fused patch weight count overflow"))?;
    let expected_bytes = expected_elements
        .checked_mul(4)
        .ok_or_else(|| invalid("fused patch byte count overflow"))?;
    if p0.data.len() != expected_bytes || p1.data.len() != expected_bytes {
        return Err(invalid("patch embedding byte length mismatch"));
    }
    let mut fused_bytes = Vec::new();
    fused_bytes.try_reserve_exact(expected_bytes)?;
    for (a, b) in p0
        .data
        .as_chunks::<4>()
        .0
        .iter()
        .zip(p1.data.as_chunks::<4>().0)
    {
        let va = f32::from_le_bytes(*a);
        let vb = f32::from_le_bytes(*b);
        let sum = va + vb;
        if !sum.is_finite() {
            return Err(invalid("nonfinite temporal patch embedding sum"));
        }
        fused_bytes.extend_from_slice(&sum.to_le_bytes());
    }
    Ok(Tensor {
        name: "v.patch_embd.fused".to_owned(),
        dimensions: vec![patch_pixels, hidden_size],
        kind: 0,
        data: crate::model::TensorBytes::from(fused_bytes),
    })
}

fn load_f32_vector(model: &Model, name: &str, len: usize) -> Result<Vec<f32>> {
    let tensor = require_tensor(model, name, &[len])?;
    let mut out = Vec::new();
    row_into(tensor, 0, &mut out)?;
    if out.len() != len || !is_finite_slice(&out) {
        return Err(invalid(format!("invalid vector tensor values in {name}")));
    }
    Ok(out)
}

fn load_matrix_rows(
    tensor: &Tensor,
    rows: usize,
    cols: usize,
    row_parallelism: &RowParallelism,
) -> Result<Vec<f32>> {
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| invalid("matrix row element count overflow"))?;
    let mut flat = Vec::new();
    flat.try_reserve_exact(total)?;
    flat.resize(total, 0.0_f32);
    row_parallelism.update_chunks(&mut flat, cols, |base_row, chunk| {
        let mut row = Vec::new();
        for (offset, dst) in chunk.chunks_exact_mut(cols).enumerate() {
            row_into(tensor, base_row + offset, &mut row)?;
            if row.len() != cols || !is_finite_slice(&row) {
                return Err(invalid(format!("invalid matrix row in {}", tensor.name)));
            }
            dst.copy_from_slice(&row);
        }
        Ok(())
    })?;
    Ok(flat)
}

fn load_block(model: &Model, config: &VisionConfig, index: usize) -> Result<VitBlock> {
    let d = config.hidden_size;
    let m = config.intermediate_size;
    Ok(VitBlock {
        ln1_w: load_f32_vector(model, &format!("v.blk.{index}.ln1.weight"), d)?,
        ln1_b: load_f32_vector(model, &format!("v.blk.{index}.ln1.bias"), d)?,
        attn_qkv_w: require_tensor(
            model,
            &format!("v.blk.{index}.attn_qkv.weight"),
            &[d, 3 * d],
        )?
        .clone(),
        attn_qkv_b: load_f32_vector(model, &format!("v.blk.{index}.attn_qkv.bias"), 3 * d)?,
        attn_out_w: require_tensor(model, &format!("v.blk.{index}.attn_out.weight"), &[d, d])?
            .clone(),
        attn_out_b: load_f32_vector(model, &format!("v.blk.{index}.attn_out.bias"), d)?,
        ln2_w: load_f32_vector(model, &format!("v.blk.{index}.ln2.weight"), d)?,
        ln2_b: load_f32_vector(model, &format!("v.blk.{index}.ln2.bias"), d)?,
        ffn_up_w: require_tensor(model, &format!("v.blk.{index}.ffn_up.weight"), &[d, m])?.clone(),
        ffn_up_b: load_f32_vector(model, &format!("v.blk.{index}.ffn_up.bias"), m)?,
        ffn_down_w: require_tensor(model, &format!("v.blk.{index}.ffn_down.weight"), &[m, d])?
            .clone(),
        ffn_down_b: load_f32_vector(model, &format!("v.blk.{index}.ffn_down.bias"), d)?,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use bonsai_gguf::{Metadata, MetadataArray, MetadataValue};

    use super::*;
    use crate::kernels::quantize_row;

    fn f32_tensor(name: &str, dimensions: Vec<usize>, values: &[f32]) -> Tensor {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Tensor {
            name: name.to_owned(),
            dimensions,
            kind: 0,
            data: Arc::<[u8]>::from(bytes).into(),
        }
    }

    fn q8_0_tensor(name: &str, cols: usize, rows: usize, values: &[f32]) -> Result<Tensor> {
        let mut data = Vec::new();
        for r in 0..rows {
            let encoded = quantize_row(8, &values[r * cols..(r + 1) * cols])?;
            data.extend_from_slice(&encoded);
        }
        Ok(Tensor {
            name: name.to_owned(),
            dimensions: vec![cols, rows],
            kind: 8,
            data: Arc::<[u8]>::from(data).into(),
        })
    }

    fn f16_tensor(name: &str, cols: usize, rows: usize, values: &[f32]) -> Result<Tensor> {
        let mut data = Vec::new();
        for r in 0..rows {
            let encoded = quantize_row(1, &values[r * cols..(r + 1) * cols])?;
            data.extend_from_slice(&encoded);
        }
        Ok(Tensor {
            name: name.to_owned(),
            dimensions: vec![cols, rows],
            kind: 1,
            data: Arc::<[u8]>::from(data).into(),
        })
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Construct complete synthetic GGUF mmproj fixture."
    )]
    fn synthetic_mmproj_model() -> Result<Model> {
        // Use hidden_size = 32 (divisible by Q8_0 block 32 and heads = 4 -> d_head = 8 % 4 == 0).
        let mut metadata = Metadata::new();
        metadata.insert(
            "general.architecture".to_owned(),
            MetadataValue::from("clip"),
        );
        metadata.insert("general.type".to_owned(), MetadataValue::from("mmproj"));
        metadata.insert(
            "clip.projector_type".to_owned(),
            MetadataValue::from("qwen3vl_merger"),
        );
        metadata.insert(
            "clip.has_vision_encoder".to_owned(),
            MetadataValue::from(true),
        );
        metadata.insert(
            "clip.vision.patch_size".to_owned(),
            MetadataValue::from(2_u32),
        );
        metadata.insert(
            "clip.vision.spatial_merge_size".to_owned(),
            MetadataValue::from(2_u32),
        );
        metadata.insert(
            "clip.vision.embedding_length".to_owned(),
            MetadataValue::from(32_u32),
        );
        metadata.insert(
            "clip.vision.attention.head_count".to_owned(),
            MetadataValue::from(4_u32),
        );
        metadata.insert(
            "clip.vision.feed_forward_length".to_owned(),
            MetadataValue::from(64_u32),
        );
        metadata.insert(
            "clip.vision.block_count".to_owned(),
            MetadataValue::from(1_u32),
        );
        metadata.insert(
            "clip.vision.projection_dim".to_owned(),
            MetadataValue::from(32_u32),
        );
        metadata.insert(
            "clip.vision.image_mean".to_owned(),
            MetadataValue::Array(MetadataArray::F32(vec![0.5, 0.5, 0.5])),
        );
        metadata.insert(
            "clip.vision.image_std".to_owned(),
            MetadataValue::Array(MetadataArray::F32(vec![0.5, 0.5, 0.5])),
        );

        let mut tensors = BTreeMap::new();
        let patch_elems = 2 * 2 * 3 * 32;
        tensors.insert(
            "v.patch_embd.weight".to_owned(),
            f32_tensor(
                "v.patch_embd.weight",
                vec![2, 2, 3, 32],
                &vec![0.05; patch_elems],
            ),
        );
        tensors.insert(
            "v.patch_embd.weight.1".to_owned(),
            f32_tensor(
                "v.patch_embd.weight.1",
                vec![2, 2, 3, 32],
                &vec![0.05; patch_elems],
            ),
        );
        tensors.insert(
            "v.patch_embd.bias".to_owned(),
            f32_tensor("v.patch_embd.bias", vec![32], &[0.01; 32]),
        );
        tensors.insert(
            "v.position_embd.weight".to_owned(),
            f32_tensor("v.position_embd.weight", vec![32, 4], &vec![0.02; 32 * 4]),
        );

        let ones = vec![1.0_f32; 32];
        let zeros32 = vec![0.0_f32; 32];
        tensors.insert(
            "v.blk.0.ln1.weight".to_owned(),
            f32_tensor("v.blk.0.ln1.weight", vec![32], &ones),
        );
        tensors.insert(
            "v.blk.0.ln1.bias".to_owned(),
            f32_tensor("v.blk.0.ln1.bias", vec![32], &zeros32),
        );
        tensors.insert(
            "v.blk.0.attn_qkv.weight".to_owned(),
            q8_0_tensor("v.blk.0.attn_qkv.weight", 32, 96, &vec![0.03; 32 * 96])?,
        );
        tensors.insert(
            "v.blk.0.attn_qkv.bias".to_owned(),
            f32_tensor("v.blk.0.attn_qkv.bias", vec![96], &vec![0.0; 96]),
        );
        tensors.insert(
            "v.blk.0.attn_out.weight".to_owned(),
            q8_0_tensor("v.blk.0.attn_out.weight", 32, 32, &vec![0.02; 32 * 32])?,
        );
        tensors.insert(
            "v.blk.0.attn_out.bias".to_owned(),
            f32_tensor("v.blk.0.attn_out.bias", vec![32], &zeros32),
        );
        tensors.insert(
            "v.blk.0.ln2.weight".to_owned(),
            f32_tensor("v.blk.0.ln2.weight", vec![32], &ones),
        );
        tensors.insert(
            "v.blk.0.ln2.bias".to_owned(),
            f32_tensor("v.blk.0.ln2.bias", vec![32], &zeros32),
        );
        tensors.insert(
            "v.blk.0.ffn_up.weight".to_owned(),
            q8_0_tensor("v.blk.0.ffn_up.weight", 32, 64, &vec![0.02; 32 * 64])?,
        );
        tensors.insert(
            "v.blk.0.ffn_up.bias".to_owned(),
            f32_tensor("v.blk.0.ffn_up.bias", vec![64], &vec![0.0; 64]),
        );
        tensors.insert(
            "v.blk.0.ffn_down.weight".to_owned(),
            f16_tensor("v.blk.0.ffn_down.weight", 64, 32, &vec![0.02; 64 * 32])?,
        );
        tensors.insert(
            "v.blk.0.ffn_down.bias".to_owned(),
            f32_tensor("v.blk.0.ffn_down.bias", vec![32], &zeros32),
        );

        tensors.insert(
            "v.post_ln.weight".to_owned(),
            f32_tensor("v.post_ln.weight", vec![32], &ones),
        );
        tensors.insert(
            "v.post_ln.bias".to_owned(),
            f32_tensor("v.post_ln.bias", vec![32], &zeros32),
        );
        tensors.insert(
            "mm.0.weight".to_owned(),
            q8_0_tensor("mm.0.weight", 128, 128, &vec![0.01; 128 * 128])?,
        );
        tensors.insert(
            "mm.0.bias".to_owned(),
            f32_tensor("mm.0.bias", vec![128], &vec![0.0; 128]),
        );
        tensors.insert(
            "mm.2.weight".to_owned(),
            q8_0_tensor("mm.2.weight", 128, 32, &vec![0.02; 128 * 32])?,
        );
        tensors.insert(
            "mm.2.bias".to_owned(),
            f32_tensor("mm.2.bias", vec![32], &zeros32),
        );

        Ok(Model {
            metadata,
            tensors,
            identity: "synthetic-mmproj".to_owned(),
        })
    }

    #[test]
    fn dynamic_grid_planning_preserves_bounds_and_alignment() -> Result<()> {
        let (w, h) = plan_image_grid(1920, 1080, 4, 1024)?;
        assert_eq!(w % 32, 0);
        assert_eq!(h % 32, 0);
        let tokens = (w / 32) * (h / 32);
        assert!((4..=1024).contains(&tokens));

        // Tiny image upscales at least to min_tokens.
        let (sw, sh) = plan_image_grid(16, 16, 4, 64)?;
        assert_eq!((sw / 32) * (sh / 32), 4);
        assert!(plan_image_grid(5000, 10, 4, 64).is_err());
        Ok(())
    }

    #[test]
    fn vit_rope_uses_independent_y_x_sectors_and_preserves_norm() -> Result<()> {
        let cfg = VisionConfig {
            patch_size: 16,
            spatial_merge: 2,
            hidden_size: 1152,
            num_heads: 16,
            intermediate_size: 4304,
            num_layers: 27,
            projection_dim: 5120,
            eps: 1e-6,
            mean: [0.5; 3],
            std: [0.5; 3],
            pos_grid_side: 48,
        };
        let rope = build_vit_rope_cache(&cfg, &[(3, 7)])?;
        assert_eq!(rope.len(), 36);
        // Sector 0 index 0 rotates with py=3 (cos(3), sin(3));
        // Sector 1 index 18 resets independent frequency scale and rotates with px=7 (cos(7), sin(7)).
        assert!((rope[0].0 - 3.0_f32.cos()).abs() < 1e-6);
        assert!((rope[18].0 - 7.0_f32.cos()).abs() < 1e-6);

        let mut head = (0..72)
            .map(|i| f32::from(u16::try_from(i).unwrap_or(0)).mul_add(0.1, 0.05))
            .collect::<Vec<_>>();
        let before_norm: f32 = head.iter().map(|v| v * v).sum();
        apply_vit_rope_patch(&mut head, 1, 72, &rope);
        let after_norm: f32 = head.iter().map(|v| v * v).sum();
        assert!((before_norm - after_norm).abs() < 1e-3);
        Ok(())
    }

    #[test]
    fn synthetic_mmproj_end_to_end_encoding_and_cache_work() -> Result<()> {
        let model = synthetic_mmproj_model()?;
        let encoder =
            SwiftVisionEncoder::from_model(&model, MatvecBackend::Scalar, RowParallelism::new(1)?)?;
        let rgb = vec![180_u8; 8 * 8 * 3];
        let image_out = encoder.encode_rgb_image(&rgb, 8, 8, 16)?;
        assert!(image_out.grid_width >= 1 && image_out.grid_height >= 1);
        assert_eq!(
            image_out.embeddings.len(),
            image_out.grid_width * image_out.grid_height * 32
        );
        assert!(is_finite_slice(&image_out.embeddings));

        // Second identical encoding hits the SHA-256 cache cleanly.
        let second = encoder.encode_rgb_image(&rgb, 8, 8, 16)?;
        assert_eq!(second.sha256, image_out.sha256);
        assert!(Arc::ptr_eq(&second.embeddings, &image_out.embeddings));

        // Same byte length with swapped dimensions must not collide in cache.
        let rect_rgb = vec![42_u8; 8 * 16 * 3];
        let wide = encoder.encode_rgb_image(&rect_rgb, 16, 8, 16)?;
        let tall = encoder.encode_rgb_image(&rect_rgb, 8, 16, 16)?;
        assert_ne!(wide.sha256, tall.sha256);
        assert!(!Arc::ptr_eq(&wide.embeddings, &tall.embeddings));
        assert_ne!(
            (wide.grid_width, wide.grid_height),
            (tall.grid_width, tall.grid_height)
        );
        encoder.clear_cache()?;
        Ok(())
    }

    fn varied_values(len: usize, seed: usize, scale: f32) -> Vec<f32> {
        (0..len)
            .map(|idx| {
                let signed = i16::try_from((idx.wrapping_mul(17) + seed) % 29).unwrap_or(0) - 14;
                f32::from(signed) * scale
            })
            .collect()
    }

    fn non_uniform_mmproj_model() -> Result<Model> {
        let mut model = synthetic_mmproj_model()?;
        for (name, dims, seed, scale) in [
            ("v.patch_embd.weight", vec![2, 2, 3, 32], 3, 0.015),
            ("v.patch_embd.weight.1", vec![2, 2, 3, 32], 11, 0.012),
            ("v.patch_embd.bias", vec![32], 5, 0.02),
            ("v.position_embd.weight", vec![32, 4], 7, 0.01),
            ("v.blk.0.attn_qkv.bias", vec![96], 2, 0.01),
            ("v.blk.0.attn_out.bias", vec![32], 9, 0.01),
            ("v.blk.0.ffn_up.bias", vec![64], 4, 0.01),
            ("v.blk.0.ffn_down.bias", vec![32], 15, 0.01),
        ] {
            let count = dims.iter().product();
            model.tensors.insert(
                name.to_owned(),
                f32_tensor(name, dims, &varied_values(count, seed, scale)),
            );
        }
        for (name, cols, rows, seed, scale) in [
            ("v.blk.0.attn_qkv.weight", 32, 96, 13, 0.025),
            ("v.blk.0.attn_out.weight", 32, 32, 19, 0.02),
            ("v.blk.0.ffn_up.weight", 32, 64, 23, 0.02),
            ("mm.0.weight", 128, 128, 31, 0.012),
            ("mm.2.weight", 128, 32, 37, 0.015),
        ] {
            model.tensors.insert(
                name.to_owned(),
                q8_0_tensor(name, cols, rows, &varied_values(cols * rows, seed, scale))?,
            );
        }
        model.tensors.insert(
            "v.blk.0.ffn_down.weight".to_owned(),
            f16_tensor(
                "v.blk.0.ffn_down.weight",
                64,
                32,
                &varied_values(64 * 32, 29, 0.018),
            )?,
        );
        Ok(model)
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "Preexisting end-to-end multi-worker scalar/Auto ViT encoder parity test."
    )]
    fn parallel_tiled_encoder_and_four_query_attention_match_scalar_bit_for_bit() -> Result<()> {
        let model = non_uniform_mmproj_model()?;
        let rgb: Vec<u8> = (0..16 * 12 * 3)
            .map(|i| u8::try_from((i * 43 + 17) & 0xff).unwrap_or(0))
            .collect();
        let baseline =
            SwiftVisionEncoder::from_model(&model, MatvecBackend::Scalar, RowParallelism::new(1)?)?;
        // 16x12 with factor 4 (patch_size 2 * spatial_merge 2) preserves (16, 12) => n_pos = 48 > 32.
        let expected = baseline.encode_rgb_image(&rgb, 16, 12, 16)?;
        assert_eq!(expected.grid_width, 4);
        assert_eq!(expected.grid_height, 3);

        let workers_list: &[usize] = if cfg!(feature = "parallel-rows") {
            &[1, 6, 8, 12]
        } else {
            &[1]
        };
        for &workers in workers_list {
            let scalar_parallel = SwiftVisionEncoder::from_model(
                &model,
                MatvecBackend::Scalar,
                RowParallelism::new(workers)?,
            )?;
            let actual_scalar = scalar_parallel.encode_rgb_image(&rgb, 16, 12, 16)?;
            assert_eq!(actual_scalar.grid_width, expected.grid_width);
            assert_eq!(actual_scalar.grid_height, expected.grid_height);
            assert_eq!(
                actual_scalar
                    .embeddings
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .embeddings
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                "vision encoder bit mismatch for Scalar/{workers} workers"
            );

            let encoder = SwiftVisionEncoder::from_model(
                &model,
                MatvecBackend::Auto,
                RowParallelism::new(workers)?,
            )?;
            let actual = encoder.encode_rgb_image(&rgb, 16, 12, 16)?;
            assert_eq!(actual.grid_width, expected.grid_width);
            assert_eq!(actual.grid_height, expected.grid_height);
            let max_diff = actual
                .embeddings
                .iter()
                .zip(expected.embeddings.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                max_diff < 1e-3,
                "vision encoder Auto/{workers} max_diff={max_diff} exceeded tolerance"
            );
        }

        // Verify 4-query tiled attention + odd tail at n_pos = 37 against single-query reference.
        let n_pos = 37;
        let d_head = 16;
        let kq_scale = 1.0_f32 / head_sqrt(d_head)?;
        let q_head = varied_values(n_pos * d_head, 5, 0.2);
        let k_head = varied_values(n_pos * d_head, 11, 0.2);
        let v_head = varied_values(n_pos * d_head, 19, 0.2);

        let mut single_scores = vec![0.0_f32; n_pos];
        let mut expected_head = vec![0.0_f32; n_pos * d_head];
        for q_tok in 0..n_pos {
            attend_single_query(
                &q_head[q_tok * d_head..(q_tok + 1) * d_head],
                &k_head,
                &v_head,
                d_head,
                kq_scale,
                &mut single_scores,
                &mut expected_head[q_tok * d_head..(q_tok + 1) * d_head],
            )?;
        }

        let mut quad_scores = vec![0.0_f32; 4 * n_pos];
        let mut actual_head = vec![0.0_f32; n_pos * d_head];
        attend_head_four_queries(
            &q_head,
            &k_head,
            &v_head,
            n_pos,
            d_head,
            kq_scale,
            &mut quad_scores,
            &mut actual_head,
            MatvecBackend::Scalar,
        )?;
        assert_eq!(
            actual_head.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected_head
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );

        let mut auto_head = vec![0.0_f32; n_pos * d_head];
        attend_head_four_queries(
            &q_head,
            &k_head,
            &v_head,
            n_pos,
            d_head,
            kq_scale,
            &mut quad_scores,
            &mut auto_head,
            MatvecBackend::Auto,
        )?;
        let attn_diff = auto_head
            .iter()
            .zip(expected_head.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            attn_diff < 1e-4,
            "4-query Auto attention diff={attn_diff} exceeded tolerance"
        );
        Ok(())
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "Preexisting end-to-end F16 ViT encoder and SIMD operator parity test."
    )]
    fn auto_f16_vit_encoder_and_simd_ops_match_scalar_within_tolerance() -> Result<()> {
        // Test vectorized attention at ViT d_head = 72 (not a multiple of 16) with non-divisible n_pos.
        let n_pos = 19;
        let d_head = 72;
        let kq_scale = 1.0_f32 / head_sqrt(d_head)?;
        let q_head = varied_values(n_pos * d_head, 3, 0.15);
        let k_head = varied_values(n_pos * d_head, 7, 0.15);
        let v_head = varied_values(n_pos * d_head, 13, 0.15);
        let mut s_scores = vec![0.0_f32; 4 * n_pos];
        let mut expected_attn = vec![0.0_f32; n_pos * d_head];
        let mut actual_attn = vec![0.0_f32; n_pos * d_head];
        attend_head_four_queries(
            &q_head,
            &k_head,
            &v_head,
            n_pos,
            d_head,
            kq_scale,
            &mut s_scores,
            &mut expected_attn,
            MatvecBackend::Scalar,
        )?;
        attend_head_four_queries(
            &q_head,
            &k_head,
            &v_head,
            n_pos,
            d_head,
            kq_scale,
            &mut s_scores,
            &mut actual_attn,
            MatvecBackend::Auto,
        )?;
        let attn_max_diff = actual_attn
            .iter()
            .zip(&expected_attn)
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            attn_max_diff < 1e-4,
            "d_head=72 attention max_diff={attn_max_diff}"
        );

        // Test vectorized GELU across [-100, 100] and tail elements.
        let gelu_inputs: Vec<f32> = (-200..=200)
            .map(|i| f32::from(i16::try_from(i).unwrap_or(0)) * 0.05)
            .chain([-100.0, -25.0, 25.0, 100.0])
            .collect();
        let gelu_bias = varied_values(gelu_inputs.len(), 9, 0.05);
        let mut scalar_gelu = vec![gelu_inputs.clone()];
        let mut auto_gelu = vec![gelu_inputs];
        let serial = RowParallelism::new(1)?;
        apply_bias_gelu_rows(&mut scalar_gelu, &gelu_bias, &serial, MatvecBackend::Scalar)?;
        apply_bias_gelu_rows(&mut auto_gelu, &gelu_bias, &serial, MatvecBackend::Auto)?;
        let gelu_max_diff = auto_gelu[0]
            .iter()
            .zip(&scalar_gelu[0])
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            gelu_max_diff < 1e-5,
            "GELU SIMD max_diff={gelu_max_diff} exceeded 1e-5"
        );

        // Test vectorized LayerNorm across 1152-wide and odd 35-wide rows.
        for cols in [35_usize, 1152_usize] {
            let rows = 3;
            let ln_in = varied_values(rows * cols, 17, 0.3);
            let ln_w = varied_values(cols, 5, 0.1)
                .into_iter()
                .map(|v| v + 1.0)
                .collect::<Vec<_>>();
            let ln_b = varied_values(cols, 11, 0.05);
            let mut ln_scalar = vec![0.0_f32; rows * cols];
            let mut ln_auto = vec![0.0_f32; rows * cols];
            layer_norm_rows(
                &ln_in,
                rows,
                cols,
                &ln_w,
                &ln_b,
                1e-6,
                &mut ln_scalar,
                &serial,
                MatvecBackend::Scalar,
            )?;
            layer_norm_rows(
                &ln_in,
                rows,
                cols,
                &ln_w,
                &ln_b,
                1e-6,
                &mut ln_auto,
                &serial,
                MatvecBackend::Auto,
            )?;
            let ln_max_diff = ln_auto
                .iter()
                .zip(&ln_scalar)
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                ln_max_diff < 1e-5,
                "LayerNorm cols={cols} max_diff={ln_max_diff} exceeded 1e-5"
            );
        }

        // Full multi-head F16 ViT block + merger encoder parity (Auto vs Scalar).
        let mut model = non_uniform_mmproj_model()?;
        for (name, cols, rows, seed, scale) in [
            ("v.blk.0.attn_qkv.weight", 32, 96, 13, 0.025),
            ("v.blk.0.attn_out.weight", 32, 32, 19, 0.02),
            ("v.blk.0.ffn_up.weight", 32, 64, 23, 0.02),
            ("mm.0.weight", 128, 128, 31, 0.012),
            ("mm.2.weight", 128, 32, 37, 0.015),
        ] {
            model.tensors.insert(
                name.to_owned(),
                f16_tensor(name, cols, rows, &varied_values(cols * rows, seed, scale))?,
            );
        }
        let rgb: Vec<u8> = (0..16 * 12 * 3)
            .map(|i| u8::try_from((i * 43 + 17) & 0xff).unwrap_or(0))
            .collect();
        let scalar_enc =
            SwiftVisionEncoder::from_model(&model, MatvecBackend::Scalar, RowParallelism::new(1)?)?;
        let scalar_out = scalar_enc.encode_rgb_image(&rgb, 16, 12, 16)?;
        let workers = if cfg!(feature = "parallel-rows") {
            4
        } else {
            1
        };
        let auto_enc = SwiftVisionEncoder::from_model(
            &model,
            MatvecBackend::Auto,
            RowParallelism::new(workers)?,
        )?;
        let auto_out = auto_enc.encode_rgb_image(&rgb, 16, 12, 16)?;
        assert_eq!(auto_out.embeddings.len(), scalar_out.embeddings.len());
        let f16_max_diff = auto_out
            .embeddings
            .iter()
            .zip(scalar_out.embeddings.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            f16_max_diff < 1e-3,
            "full F16 ViT encoder Auto vs Scalar max_diff={f16_max_diff} exceeded 1e-3"
        );
        Ok(())
    }

    fn encode_b64_test(input: &[u8]) -> String {
        const TABLE: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in input.chunks(3) {
            let b0 = usize::from(chunk[0]);
            let b1 = usize::from(*chunk.get(1).unwrap_or(&0));
            let b2 = usize::from(*chunk.get(2).unwrap_or(&0));
            let triple = (b0 << 16) | (b1 << 8) | b2;
            out.push(char::from(TABLE[(triple >> 18) & 63]));
            out.push(char::from(TABLE[(triple >> 12) & 63]));
            out.push(if chunk.len() > 1 {
                char::from(TABLE[(triple >> 6) & 63])
            } else {
                '='
            });
            out.push(if chunk.len() > 2 {
                char::from(TABLE[triple & 63])
            } else {
                '='
            });
        }
        out
    }

    #[test]
    fn encode_data_url_hashes_before_png_decode_and_rejects_early_budget() -> Result<()> {
        let model = synthetic_mmproj_model()?;
        let encoder =
            SwiftVisionEncoder::from_model(&model, MatvecBackend::Auto, RowParallelism::new(1)?)?;
        let raw_non_png = b"not-even-close-to-png";
        let sha256 = sha256_bytes(raw_non_png)?;
        let sentinel = EncodedImage {
            grid_width: 2,
            grid_height: 2,
            sha256,
            embeddings: Arc::from(vec![1.5_f32; 4 * 32]),
        };
        encoder.cache_store(sha256, 16, sentinel.clone())?;

        // Cache hit must succeed before attempting PNG header/zlib decode.
        let cached_url = format!("data:image/png;base64,{}", encode_b64_test(raw_non_png));
        let cached_out = encoder.encode_data_url(&cached_url, 16)?;
        assert!(Arc::ptr_eq(&cached_out.embeddings, &sentinel.embeddings));

        // Zero max_tokens is rejected upfront before decoding image bytes.
        assert!(encoder.encode_data_url(&cached_url, 0).is_err());

        // Real PPM data URL cache miss populates cache and subsequent lookup returns identical Arc.
        let mut ppm = b"P6\n4 4\n255\n".to_vec();
        ppm.extend((0..4 * 4 * 3).map(|i| u8::try_from((i * 19) & 0xff).unwrap_or(0)));
        let ppm_url = format!(
            "data:image/x-portable-pixmap;base64,{}",
            encode_b64_test(&ppm)
        );
        let first = encoder.encode_data_url(&ppm_url, 16)?;
        let second = encoder.encode_data_url(&ppm_url, 16)?;
        assert!(Arc::ptr_eq(&first.embeddings, &second.embeddings));
        Ok(())
    }

    #[test]
    fn four_query_neon_attention_matches_scalar_across_n_pos_and_d_head() -> Result<()> {
        for &d_head in &[64_usize, 72] {
            let kq_scale = 1.0_f32 / head_sqrt(d_head)?;
            for &n_pos in &[1_usize, 3, 4, 17, 64, 256] {
                let q_head = varied_values(n_pos * d_head, 5 + n_pos, 0.12);
                let k_head = varied_values(n_pos * d_head, 11 + d_head, 0.12);
                let v_head = varied_values(n_pos * d_head, 19 + n_pos, 0.12);
                let mut scores_buf = vec![0.0_f32; 4 * n_pos];
                let mut scalar_out = vec![0.0_f32; n_pos * d_head];
                let mut neon_out = vec![0.0_f32; n_pos * d_head];
                attend_head_four_queries(
                    &q_head,
                    &k_head,
                    &v_head,
                    n_pos,
                    d_head,
                    kq_scale,
                    &mut scores_buf,
                    &mut scalar_out,
                    MatvecBackend::Scalar,
                )?;
                attend_head_four_queries(
                    &q_head,
                    &k_head,
                    &v_head,
                    n_pos,
                    d_head,
                    kq_scale,
                    &mut scores_buf,
                    &mut neon_out,
                    MatvecBackend::Auto,
                )?;
                let max_diff = neon_out
                    .iter()
                    .zip(&scalar_out)
                    .map(|(&a, &b)| (a - b).abs())
                    .fold(0.0_f32, f32::max);
                assert!(
                    max_diff <= 1e-5,
                    "attention parity failed for n_pos={n_pos}, d_head={d_head}: max_diff={max_diff}"
                );
                for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                    let mut bad_q = q_head.clone();
                    bad_q[n_pos * d_head - 1] = bad;
                    let mut dummy = vec![0.0_f32; n_pos * d_head];
                    for backend in [MatvecBackend::Scalar, MatvecBackend::Auto] {
                        assert!(
                            attend_head_four_queries(
                                &bad_q,
                                &k_head,
                                &v_head,
                                n_pos,
                                d_head,
                                kq_scale,
                                &mut scores_buf,
                                &mut dummy,
                                backend,
                            )
                            .is_err()
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn self_attention_2d_parallelism_matches_scalar_across_workers() -> Result<()> {
        let num_heads = 16_usize;
        let workers_list: &[usize] = if cfg!(feature = "parallel-rows") {
            &[1, 4, 12]
        } else {
            &[1]
        };
        for &d_head in &[64_usize, 72] {
            for &n_pos in &[1_usize, 3, 4, 17, 64, 256] {
                let total_len = num_heads * n_pos * d_head;
                let mut baseline_auto_bits: Option<Vec<u32>> = None;
                let mut baseline_scalar_bits: Option<Vec<u32>> = None;
                for &workers in workers_list {
                    let policy = RowParallelism::new(workers)?;
                    let mut attn_scalar =
                        AttentionScratch::new(total_len, num_heads, n_pos, workers)?;
                    attn_scalar.q_heads = varied_values(total_len, 3, 0.1);
                    attn_scalar.k_heads = varied_values(total_len, 7, 0.1);
                    attn_scalar.v_heads = varied_values(total_len, 13, 0.1);
                    dispatch_self_attention_heads(
                        num_heads,
                        n_pos,
                        d_head,
                        &mut attn_scalar,
                        &policy,
                        MatvecBackend::Scalar,
                    )?;
                    let scalar_bits: Vec<u32> =
                        attn_scalar.head_out.iter().map(|v| v.to_bits()).collect();
                    assert_eq!(
                        baseline_scalar_bits.get_or_insert_with(|| scalar_bits.clone()),
                        &scalar_bits,
                        "Scalar multi-worker mismatch for n_pos={n_pos}, d_head={d_head}, workers={workers}"
                    );

                    let mut attn_auto =
                        AttentionScratch::new(total_len, num_heads, n_pos, workers)?;
                    attn_auto.q_heads = attn_scalar.q_heads.clone();
                    attn_auto.k_heads = attn_scalar.k_heads.clone();
                    attn_auto.v_heads = attn_scalar.v_heads.clone();
                    dispatch_self_attention_heads(
                        num_heads,
                        n_pos,
                        d_head,
                        &mut attn_auto,
                        &policy,
                        MatvecBackend::Auto,
                    )?;
                    let auto_bits: Vec<u32> =
                        attn_auto.head_out.iter().map(|v| v.to_bits()).collect();
                    assert_eq!(
                        baseline_auto_bits.get_or_insert_with(|| auto_bits.clone()),
                        &auto_bits,
                        "Auto multi-worker bit mismatch for n_pos={n_pos}, d_head={d_head}, workers={workers}"
                    );
                    let worker_diff = attn_auto
                        .head_out
                        .iter()
                        .zip(&attn_scalar.head_out)
                        .map(|(&a, &b)| (a - b).abs())
                        .fold(0.0_f32, f32::max);
                    assert!(
                        worker_diff <= 1e-5,
                        "2D attention parity failed for n_pos={n_pos}, d_head={d_head}, workers={workers}: max_diff={worker_diff}"
                    );
                }
            }
        }
        Ok(())
    }

    #[cfg(all(
        feature = "experimental-neon",
        target_arch = "aarch64",
        target_os = "windows",
        target_env = "msvc",
        target_feature = "neon"
    ))]
    #[test]
    fn softmax_four_queries_neon_matches_scalar_and_rejects_nonfinite() -> Result<()> {
        if let Some(neon) = fearless_simd::Level::baseline().as_neon() {
            for &n_pos in &[1_usize, 3, 4, 17, 64, 256] {
                let mut ref_rows = [
                    varied_values(n_pos, 1, 1.5),
                    varied_values(n_pos, 7, 2.0),
                    varied_values(n_pos, 13, 0.5),
                    varied_values(n_pos, 23, 3.0),
                ];
                let mut neon_rows = ref_rows.clone();
                for row in &mut ref_rows {
                    softmax(row)?;
                }
                let [r0, r1, r2, r3] = &mut neon_rows;
                softmax_four_queries_neon(neon, [r0, r1, r2, r3])?;
                for row_idx in 0..4 {
                    let diff = neon_rows[row_idx]
                        .iter()
                        .zip(&ref_rows[row_idx])
                        .map(|(&a, &b)| (a - b).abs())
                        .fold(0.0_f32, f32::max);
                    assert!(
                        diff <= 1e-5,
                        "softmax_four_queries_neon diff={diff} for n_pos={n_pos}, row={row_idx}"
                    );
                }
                if n_pos >= 2 {
                    let mut p_ref = [
                        vec![0.0_f32; n_pos],
                        vec![1.0_f32; n_pos],
                        vec![-2.0_f32; n_pos],
                        vec![3.0_f32; n_pos],
                    ];
                    p_ref[0][0] = f32::NEG_INFINITY;
                    p_ref[2][n_pos - 1] = f32::NEG_INFINITY;
                    let mut p_neon = p_ref.clone();
                    for row in &mut p_ref {
                        softmax(row)?;
                    }
                    let [n0, n1, n2, n3] = &mut p_neon;
                    softmax_four_queries_neon(neon, [n0, n1, n2, n3])?;
                    for row_idx in 0..4 {
                        let diff = p_neon[row_idx]
                            .iter()
                            .zip(&p_ref[row_idx])
                            .map(|(&a, &b)| (a - b).abs())
                            .fold(0.0_f32, f32::max);
                        assert!(diff <= 1e-5);
                    }
                }
                for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                    let mut bad_rows = [
                        vec![0.0_f32; n_pos],
                        vec![0.0_f32; n_pos],
                        vec![0.0_f32; n_pos],
                        vec![0.0_f32; n_pos],
                    ];
                    if bad == f32::NEG_INFINITY {
                        bad_rows[1].fill(f32::NEG_INFINITY);
                    } else {
                        bad_rows[1][n_pos - 1] = bad;
                    }
                    let [b0, b1, b2, b3] = &mut bad_rows;
                    assert!(softmax_four_queries_neon(neon, [b0, b1, b2, b3]).is_err());
                }
            }
        }
        Ok(())
    }

    #[test]
    fn head_sqrt_and_rgb_digest_enforce_dimension_bounds() -> Result<()> {
        assert!(head_sqrt(0).is_err());
        assert_eq!(head_sqrt(64)?, 8.0_f32);
        let rgb = [10_u8, 20, 30, 40, 50, 60];
        let a = sha256_rgb_image(&rgb, 2, 1)?;
        let b = sha256_rgb_image(&rgb, 1, 2)?;
        assert_ne!(a, b);
        assert_ne!(a, sha256_bytes(&rgb)?);
        Ok(())
    }
}
