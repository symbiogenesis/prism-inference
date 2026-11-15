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

use crate::kernels::{
    MatvecBackend, MatvecScratch, RowParallelism, is_finite_slice, row_into, softmax,
    validate_tensor,
};
use crate::model::swift::MmprojLoadReceipt;
use crate::model::{Model, Tensor};
use crate::{Result, invalid};

use decode::{DecodedImage, decode_image_data_url};

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
        let pos_embd = load_matrix_rows(pos_tensor, pos_rows, config.hidden_size)?;

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
    /// Reuses an exact cached `(sha256, max_tokens)` result when available.
    ///
    /// # Errors
    /// Rejects invalid image payloads, token bounds, cache lock failures, or
    /// nonfinite intermediate activations.
    pub fn encode_data_url(&self, url: &str, max_tokens: usize) -> Result<EncodedImage> {
        let (decoded, raw) = decode_image_data_url(url)?;
        let sha256 = sha256_bytes(&raw)?;
        if let Some(cached) = self.cache_lookup(sha256, max_tokens)? {
            return Ok(cached);
        }
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
        let sha256 = sha256_bytes(rgb)?;
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

    #[expect(
        clippy::too_many_lines,
        reason = "Keep vision encoding pipeline stages sequential and bounded."
    )]
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
        )?;

        let patch_cols = resized_w / self.config.patch_size;
        let patch_rows = resized_h / self.config.patch_size;
        let nx = resized_w / factor;
        let ny = resized_h / factor;
        let n_pos = patch_cols * patch_rows;
        let n_tokens = nx * ny;

        let mut scratch = MatvecScratch::with_backend(self.backend);
        scratch.set_row_parallelism(self.row_parallelism.clone());

        let (mut hidden, patch_coords) = self.extract_and_embed_patches(
            &planar,
            resized_w,
            patch_cols,
            patch_rows,
            nx,
            ny,
            &mut scratch,
        )?;

        let mut residual = vec![0.0_f32; hidden.len()];
        let mut norm_buf = vec![0.0_f32; hidden.len()];
        let mut qkv_buf = vec![0.0_f32; n_pos * 3 * self.config.hidden_size];
        let mut q_buf = vec![0.0_f32; n_pos * self.config.hidden_size];
        let mut k_buf = vec![0.0_f32; n_pos * self.config.hidden_size];
        let mut v_buf = vec![0.0_f32; n_pos * self.config.hidden_size];
        let mut attn_out_buf = vec![0.0_f32; n_pos * self.config.hidden_size];
        let mut ffn_buf = vec![0.0_f32; self.config.intermediate_size];
        let mut proj_buf = Vec::new();
        let rope_cache = build_vit_rope_cache(&self.config, &patch_coords)?;

        for block in &self.blocks {
            residual.copy_from_slice(&hidden);
            layer_norm_rows(
                &hidden,
                n_pos,
                self.config.hidden_size,
                &block.ln1_w,
                &block.ln1_b,
                self.config.eps,
                &mut norm_buf,
            )?;
            run_self_attention(
                block,
                &self.config,
                n_pos,
                &rope_cache,
                &norm_buf,
                &mut qkv_buf,
                &mut q_buf,
                &mut k_buf,
                &mut v_buf,
                &mut attn_out_buf,
                &mut proj_buf,
                &mut scratch,
            )?;
            for (h, (&res, &attn)) in hidden.iter_mut().zip(residual.iter().zip(&attn_out_buf)) {
                *h = res + attn;
            }
            if !is_finite_slice(&hidden) {
                return Err(invalid("nonfinite values after ViT attention residual"));
            }

            residual.copy_from_slice(&hidden);
            layer_norm_rows(
                &hidden,
                n_pos,
                self.config.hidden_size,
                &block.ln2_w,
                &block.ln2_b,
                self.config.eps,
                &mut norm_buf,
            )?;
            run_mlp_block(
                block,
                &self.config,
                n_pos,
                &norm_buf,
                &residual,
                &mut ffn_buf,
                &mut proj_buf,
                &mut hidden,
                &mut scratch,
            )?;
            if !is_finite_slice(&hidden) {
                return Err(invalid("nonfinite values after ViT MLP residual"));
            }
        }

        layer_norm_rows(
            &hidden,
            n_pos,
            self.config.hidden_size,
            &self.post_ln_w,
            &self.post_ln_b,
            self.config.eps,
            &mut norm_buf,
        )?;
        let embeddings = self.run_merger(&norm_buf, n_tokens, &mut proj_buf, &mut scratch)?;
        Ok(EncodedImage {
            grid_width: nx,
            grid_height: ny,
            sha256,
            embeddings: Arc::from(embeddings),
        })
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
        scratch: &mut MatvecScratch,
    ) -> Result<PatchEmbeddings> {
        let p = self.config.patch_size;
        let sm = self.config.spatial_merge;
        let d = self.config.hidden_size;
        let n_pos = patch_cols * patch_rows;
        let resized_h = patch_rows * p;
        let plane_len = resized_w * resized_h;

        let mut hidden = vec![0.0_f32; n_pos * d];
        let mut coords = Vec::with_capacity(n_pos);
        let mut patch_vec = vec![0.0_f32; p * p * 3];
        let mut patch_out = Vec::new();
        let mut pos_row = vec![0.0_f32; d];

        let mut patch_idx = 0_usize;
        for by in 0..ny {
            for bx in 0..nx {
                for dy in 0..sm {
                    for dx in 0..sm {
                        let py = by * sm + dy;
                        let px = bx * sm + dx;
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
                        scratch
                            .prepare(&self.patch_embd, &patch_vec)?
                            .matvec_into(&self.patch_embd, &mut patch_out)?;
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
                        let dst = &mut hidden[patch_idx * d..(patch_idx + 1) * d];
                        for ((out, (&val, &bias)), &pos) in dst
                            .iter_mut()
                            .zip(patch_out.iter().zip(&self.patch_bias))
                            .zip(&pos_row)
                        {
                            *out = val + bias + pos;
                        }
                        coords.push((py, px));
                        patch_idx += 1;
                    }
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
        proj_buf: &mut Vec<f32>,
        scratch: &mut MatvecScratch,
    ) -> Result<Vec<f32>> {
        let merge_group = self.config.spatial_merge * self.config.spatial_merge;
        let merge_dim = self.config.hidden_size * merge_group;
        let proj_dim = self.config.projection_dim;
        let mut hidden_merge = Vec::new();
        let mut out = Vec::new();
        out.try_reserve_exact(n_tokens * proj_dim)?;

        for tok in 0..n_tokens {
            let group = &normed[tok * merge_dim..(tok + 1) * merge_dim];
            scratch
                .prepare(&self.mm_0_w, group)?
                .matvec_into(&self.mm_0_w, &mut hidden_merge)?;
            for (v, &bias) in hidden_merge.iter_mut().zip(&self.mm_0_b) {
                *v = gelu_tanh(*v + bias);
            }
            scratch
                .prepare(&self.mm_2_w, &hidden_merge)?
                .matvec_into(&self.mm_2_w, proj_buf)?;
            for (&val, &bias) in proj_buf.iter().zip(&self.mm_2_b) {
                let final_val = val + bias;
                if !final_val.is_finite() {
                    return Err(invalid("nonfinite value in multimodal merger projection"));
                }
                out.push(final_val);
            }
        }
        Ok(out)
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
    for sy in 0..src_h {
        let src_row = &rgb[sy * src_w * 3..(sy + 1) * src_w * 3];
        let dst_row = &mut horiz[sy * dst_w * 3..(sy + 1) * dst_w * 3];
        for (dx, &(idx, weights)) in x_taps.iter().enumerate() {
            for c in 0..3 {
                let mut sum = 0.0_f64;
                for t in 0..4 {
                    sum = f64::from(src_row[idx[t] * 3 + c]).mul_add(weights[t], sum);
                }
                dst_row[dx * 3 + c] = sum as f32;
            }
        }
    }

    let plane_len = dst_w * dst_h;
    let mut out = vec![0.0_f32; 3 * plane_len];
    let scale_y = (src_h as f64) / (dst_h as f64);
    let inv_255 = 1.0_f32 / 255.0_f32;
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
        for dx in 0..dst_w {
            for c in 0..3 {
                let mut sum = 0.0_f64;
                for t in 0..4 {
                    sum = f64::from(horiz[(idx[t] * dst_w + dx) * 3 + c]).mul_add(weights[t], sum);
                }
                let pixel = (sum as f32).round_ties_even().clamp(0.0, 255.0);
                out[c * plane_len + dy * dst_w + dx] = pixel.mul_add(inv_255, -mean[c]) / std[c];
            }
        }
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

fn apply_vit_rope_patch(values: &mut [f32], num_heads: usize, d_head: usize, freqs: &[(f32, f32)]) {
    let n_dims = d_head / 2;
    for head in values.chunks_exact_mut(d_head).take(num_heads) {
        for (ic, &(cos_theta, sin_theta)) in freqs.iter().enumerate() {
            let x0 = head[ic];
            let x1 = head[ic + n_dims];
            head[ic] = x1.mul_add(-sin_theta, x0 * cos_theta);
            head[ic + n_dims] = x1.mul_add(cos_theta, x0 * sin_theta);
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Reuse preallocated scratch buffers across all 27 ViT layers without inner heap allocation."
)]
fn run_self_attention(
    block: &VitBlock,
    config: &VisionConfig,
    n_pos: usize,
    rope_cache: &[(f32, f32)],
    normed: &[f32],
    qkv_buf: &mut [f32],
    q_buf: &mut [f32],
    k_buf: &mut [f32],
    v_buf: &mut [f32],
    attn_out_buf: &mut [f32],
    proj_buf: &mut Vec<f32>,
    scratch: &mut MatvecScratch,
) -> Result<()> {
    let d = config.hidden_size;
    let d_head = d / config.num_heads;
    let n_dims = d_head / 2;

    for tok in 0..n_pos {
        let x = &normed[tok * d..(tok + 1) * d];
        scratch
            .prepare(&block.attn_qkv_w, x)?
            .matvec_into(&block.attn_qkv_w, proj_buf)?;
        let qkv_tok = &mut qkv_buf[tok * 3 * d..(tok + 1) * 3 * d];
        for (dst, (&val, &bias)) in qkv_tok
            .iter_mut()
            .zip(proj_buf.iter().zip(&block.attn_qkv_b))
        {
            *dst = val + bias;
        }
        let freqs = &rope_cache[tok * n_dims..(tok + 1) * n_dims];
        let q_tok = &mut q_buf[tok * d..(tok + 1) * d];
        q_tok.copy_from_slice(&qkv_tok[..d]);
        apply_vit_rope_patch(q_tok, config.num_heads, d_head, freqs);

        let k_tok = &mut k_buf[tok * d..(tok + 1) * d];
        k_tok.copy_from_slice(&qkv_tok[d..2 * d]);
        apply_vit_rope_patch(k_tok, config.num_heads, d_head, freqs);

        v_buf[tok * d..(tok + 1) * d].copy_from_slice(&qkv_tok[2 * d..3 * d]);
    }

    let kq_scale = 1.0_f32 / head_sqrt(d_head)?;
    let mut scores = vec![0.0_f32; n_pos];
    let mut concat = vec![0.0_f32; n_pos * d];
    for head in 0..config.num_heads {
        let h_off = head * d_head;
        for q_tok in 0..n_pos {
            let q = &q_buf[q_tok * d + h_off..q_tok * d + h_off + d_head];
            for k_tok in 0..n_pos {
                let k = &k_buf[k_tok * d + h_off..k_tok * d + h_off + d_head];
                let mut dot = 0.0_f64;
                for (&qv, &kv) in q.iter().zip(k) {
                    dot = f64::from(qv).mul_add(f64::from(kv), dot);
                }
                scores[k_tok] = narrow_f64_to_f32(dot)? * kq_scale;
            }
            softmax(&mut scores)?;
            let out_head = &mut concat[q_tok * d + h_off..q_tok * d + h_off + d_head];
            out_head.fill(0.0);
            for (v_tok, &prob) in scores.iter().enumerate() {
                let v = &v_buf[v_tok * d + h_off..v_tok * d + h_off + d_head];
                for (acc, &vv) in out_head.iter_mut().zip(v) {
                    *acc = prob.mul_add(vv, *acc);
                }
            }
        }
    }

    for tok in 0..n_pos {
        let x = &concat[tok * d..(tok + 1) * d];
        scratch
            .prepare(&block.attn_out_w, x)?
            .matvec_into(&block.attn_out_w, proj_buf)?;
        let out_tok = &mut attn_out_buf[tok * d..(tok + 1) * d];
        for (dst, (&val, &bias)) in out_tok
            .iter_mut()
            .zip(proj_buf.iter().zip(&block.attn_out_b))
        {
            *dst = val + bias;
        }
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "Reuse layer scratch allocations across patches."
)]
fn run_mlp_block(
    block: &VitBlock,
    config: &VisionConfig,
    n_pos: usize,
    normed: &[f32],
    residual: &[f32],
    ffn_buf: &mut [f32],
    proj_buf: &mut Vec<f32>,
    hidden: &mut [f32],
    scratch: &mut MatvecScratch,
) -> Result<()> {
    let d = config.hidden_size;
    for tok in 0..n_pos {
        let x = &normed[tok * d..(tok + 1) * d];
        scratch
            .prepare(&block.ffn_up_w, x)?
            .matvec_into(&block.ffn_up_w, proj_buf)?;
        for (dst, (&val, &bias)) in ffn_buf.iter_mut().zip(proj_buf.iter().zip(&block.ffn_up_b)) {
            *dst = gelu_tanh(val + bias);
        }
        scratch
            .prepare(&block.ffn_down_w, ffn_buf)?
            .matvec_into(&block.ffn_down_w, proj_buf)?;
        let res = &residual[tok * d..(tok + 1) * d];
        let out = &mut hidden[tok * d..(tok + 1) * d];
        for (dst, (&r, (&val, &bias))) in out
            .iter_mut()
            .zip(res.iter().zip(proj_buf.iter().zip(&block.ffn_down_b)))
        {
            *dst = r + val + bias;
        }
    }
    Ok(())
}

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
) -> Result<()> {
    if rows == 0 || cols == 0 || input.len() != rows * cols || output.len() != input.len() {
        return Err(invalid("invalid LayerNorm dimensions"));
    }
    let inv_cols = 1.0_f64 / (cols as f64);
    for r in 0..rows {
        let src = &input[r * cols..(r + 1) * cols];
        let dst = &mut output[r * cols..(r + 1) * cols];
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
        data: Arc::from(fused_bytes),
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

fn load_matrix_rows(tensor: &Tensor, rows: usize, cols: usize) -> Result<Vec<f32>> {
    let mut flat = Vec::new();
    flat.try_reserve_exact(rows * cols)?;
    let mut row = Vec::new();
    for r in 0..rows {
        row_into(tensor, r, &mut row)?;
        if row.len() != cols || !is_finite_slice(&row) {
            return Err(invalid(format!("invalid matrix row in {}", tensor.name)));
        }
        flat.extend_from_slice(&row);
    }
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
            data: Arc::from(bytes),
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
            data: Arc::from(data),
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
            data: Arc::from(data),
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
        encoder.clear_cache()?;
        Ok(())
    }
}
