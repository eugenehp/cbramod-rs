/// Criss-Cross Transformer Encoder Layer for CBraMod.
///
/// Python:
///   - Splits embedding in half: first half → S-Attention (spatial), second half → T-Attention (temporal)
///   - S-Attention: rearrange to (B*N, C, d/2), self-attention across channels
///   - T-Attention: rearrange to (B*C, N, d/2), self-attention across time patches
///   - Concat results, then FFN with residual
///   - norm_first=True: x = x + sa(norm1(x)); x = x + ff(norm2(x))

use burn::prelude::*;
use burn::nn::{Linear, LinearConfig, LayerNorm, LayerNormConfig};
use burn::tensor::activation::{gelu, softmax};

/// Fused multi-head attention (nn.MultiheadAttention with fused in_proj).
#[derive(Module, Debug)]
pub struct FusedMha<B: Backend> {
    pub in_proj: Linear<B>,
    pub out_proj: Linear<B>,
    pub n_heads: usize,
    pub head_dim: usize,
}

impl<B: Backend> FusedMha<B> {
    pub fn new(dim: usize, n_heads: usize, device: &B::Device) -> Self {
        let head_dim = dim / n_heads;
        Self {
            in_proj: LinearConfig::new(dim, dim * 3).with_bias(true).init(device),
            out_proj: LinearConfig::new(dim, dim).with_bias(true).init(device),
            n_heads,
            head_dim,
        }
    }

    /// x: [B, S, D] → [B, S, D]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, s, _] = x.dims();
        let (h, dh) = (self.n_heads, self.head_dim);
        let dim = h * dh;

        let qkv = self.in_proj.forward(x);
        let q = qkv.clone().narrow(2, 0, dim).reshape([b, s, h, dh]).swap_dims(1, 2);
        let k = qkv.clone().narrow(2, dim, dim).reshape([b, s, h, dh]).swap_dims(1, 2);
        let v = qkv.narrow(2, dim * 2, dim).reshape([b, s, h, dh]).swap_dims(1, 2);

        let scale = (dh as f64).powf(-0.5) as f32;
        let attn = softmax(q.matmul(k.transpose()).mul_scalar(scale), 3);
        let out = attn.matmul(v);
        let out = out.swap_dims(1, 2).reshape([b, s, dim]);
        self.out_proj.forward(out)
    }
}

#[derive(Module, Debug)]
pub struct CrissCrossEncoderLayer<B: Backend> {
    pub self_attn_s: FusedMha<B>,
    pub self_attn_t: FusedMha<B>,
    pub linear1: Linear<B>,
    pub linear2: Linear<B>,
    pub norm1: LayerNorm<B>,
    pub norm2: LayerNorm<B>,
    pub d_model: usize,
}

impl<B: Backend> CrissCrossEncoderLayer<B> {
    pub fn new(
        d_model: usize, nhead: usize, dim_feedforward: usize,
        device: &B::Device,
    ) -> Self {
        let half = d_model / 2;
        Self {
            self_attn_s: FusedMha::new(half, nhead / 2, device),
            self_attn_t: FusedMha::new(half, nhead / 2, device),
            linear1: LinearConfig::new(d_model, dim_feedforward).with_bias(true).init(device),
            linear2: LinearConfig::new(dim_feedforward, d_model).with_bias(true).init(device),
            norm1: LayerNormConfig::new(d_model).init(device),
            norm2: LayerNormConfig::new(d_model).init(device),
            d_model,
        }
    }

    /// x: [B, C, N, D] → [B, C, N, D]
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let sa = self.sa_block(self.norm1.forward(x.clone()));
        let x = x + sa;
        let ff = self.ff_block(self.norm2.forward(x.clone()));
        x + ff
    }

    fn sa_block(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [bz, ch_num, patch_num, d] = x.dims();
        let half = d / 2;

        // Split: first half → spatial, second half → temporal
        let xs = x.clone().narrow(3, 0, half);
        let xt = x.narrow(3, half, half);

        // S-Attention: [B, C, N, d/2] → [B*N, C, d/2]
        let xs = xs.swap_dims(1, 2) // [B, N, C, d/2]
            .reshape([bz * patch_num, ch_num, half]);
        let xs = self.self_attn_s.forward(xs);
        let xs = xs.reshape([bz, patch_num, ch_num, half])
            .swap_dims(1, 2); // [B, C, N, d/2]

        // T-Attention: [B, C, N, d/2] → [B*C, N, d/2]
        let xt = xt.reshape([bz * ch_num, patch_num, half]);
        let xt = self.self_attn_t.forward(xt);
        let xt = xt.reshape([bz, ch_num, patch_num, half]);

        // Concat: [B, C, N, d]
        Tensor::cat(vec![xs, xt], 3)
    }

    fn ff_block(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.linear2.forward(gelu(self.linear1.forward(x)))
    }
}
