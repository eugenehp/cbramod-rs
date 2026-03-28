/// CBraMod (Criss-Cross Brain Model) — full model.
///
/// Architecture:
///   1. Rearrange input into patches: [B, C, T] → [B, C, N, P]
///   2. PatchEmbedding: conv pipeline + spectral FFT + positional conv → [B, C, N, d_model]
///   3. Criss-Cross Transformer Encoder (12 layers, spatial + temporal attention)
///   4. proj_out: Linear(d_model, emb_dim) → [B, C, N, emb_dim]
///   5. final_layer: Flatten → Linear → [B, n_outputs]

use burn::prelude::*;
#[allow(unused_imports)]
use burn::nn::{Linear, LinearConfig, LayerNorm, LayerNormConfig};

use crate::model::patch_embedding::PatchEmbedding;
use crate::model::transformer::TransformerEncoder;

#[derive(Module, Debug)]
pub struct CBraMod<B: Backend> {
    pub patch_embedding: PatchEmbedding<B>,
    pub encoder: TransformerEncoder<B>,
    pub proj_out: Linear<B>,
    pub final_ln: Option<LayerNorm<B>>,
    pub final_linear: Linear<B>,
    pub patch_size: usize,
    pub n_outputs: usize,
    pub n_chans: usize,
    pub n_times: usize,
    pub emb_dim: usize,
    pub return_encoder_output: bool,
}

impl<B: Backend> CBraMod<B> {
    pub fn new(
        n_outputs: usize,
        n_chans: usize,
        n_times: usize,
        patch_size: usize,
        dim_feedforward: usize,
        n_layer: usize,
        nhead: usize,
        emb_dim: usize,
        return_encoder_output: bool,
        device: &B::Device,
    ) -> Self {
        let patch_embedding = PatchEmbedding::new(patch_size, device);
        let d_model = patch_embedding.d_model;
        let encoder = TransformerEncoder::new(d_model, nhead, dim_feedforward, n_layer, device);
        let proj_out = LinearConfig::new(d_model, emb_dim).with_bias(true).init(device);

        let n_patches = n_times / patch_size;
        let final_dim = n_chans * n_patches * emb_dim;

        let (final_ln, final_linear) = if return_encoder_output {
            // Identity — just need placeholders
            (None, LinearConfig::new(emb_dim, emb_dim).with_bias(true).init(device))
        } else {
            (None, LinearConfig::new(final_dim, n_outputs).with_bias(true).init(device))
        };

        Self {
            patch_embedding, encoder, proj_out, final_ln, final_linear,
            patch_size, n_outputs, n_chans, n_times, emb_dim, return_encoder_output,
        }
    }

    /// x: [B, C, T] → [B, n_outputs] or [B, C, N, emb_dim]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch, n_chans, n_times] = x.dims();
        let n_patches = n_times / self.patch_size;

        // 1. Rearrange to patches: [B, C, T] → [B, C, N, P]
        let x = x.reshape([batch, n_chans, n_patches, self.patch_size]);

        // 2. Patch embedding
        let x = self.patch_embedding.forward(x);

        // 3. Encoder
        let x = self.encoder.forward(x);

        // 4. Projection
        let x = self.proj_out.forward(x); // [B, C, N, emb_dim]

        if self.return_encoder_output {
            // Flatten for now — return [B, C*N*emb_dim]
            let flat_dim = n_chans * n_patches * self.emb_dim;
            return x.reshape([batch, flat_dim]);
        }

        // 5. Flatten + Linear
        let flat_dim = n_chans * n_patches * self.emb_dim;
        let x = x.reshape([batch, flat_dim]);
        self.final_linear.forward(x)
    }
}
