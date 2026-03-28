/// Patch Embedding for CBraMod.
///
/// Python:
///   class _PatchEmbedding(nn.Module):
///     proj_in: 3-layer Conv2d+GroupNorm+GELU pipeline (1→25→25→25)
///     spectral_proj: rfft → abs → Linear(101, d_model)
///     positional_encoding: depthwise Conv2d(d_model, d_model, (19,7), groups=d_model)
///     mask_encoding: zeros(patch_size) — unused at inference
///
///   Forward:
///     x: [B, C, N, P] → proj_in on [B, 1, C*N, P] → reshape → [B, C, N, d_model]
///     spectral: rfft(x) → abs → linear → [B, C, N, d_model]
///     patch_emb = proj + spectral
///     pos = depthwise_conv2d(patch_emb transposed) → add
///     return patch_emb

use burn::prelude::*;
use burn::nn::{
    conv::{Conv2d, Conv2dConfig},
    GroupNorm, GroupNormConfig,
    Linear, LinearConfig,
};
use burn::tensor::activation::gelu;
use rustfft::{FftPlanner, num_complex::Complex};

/// Conv2d + GroupNorm + GELU block
#[derive(Module, Debug)]
pub struct ConvGnGelu<B: Backend> {
    pub conv: Conv2d<B>,
    pub gn: GroupNorm<B>,
}

impl<B: Backend> ConvGnGelu<B> {
    pub fn new(
        in_ch: usize, out_ch: usize,
        kernel: [usize; 2], stride: [usize; 2], padding: [usize; 2],
        gn_groups: usize, gn_channels: usize,
        device: &B::Device,
    ) -> Self {
        let conv = Conv2dConfig::new([in_ch, out_ch], kernel)
            .with_stride(stride)
            .with_padding(burn::nn::PaddingConfig2d::Explicit(padding[0], padding[1]))
            .with_bias(true)
            .init(device);
        let gn = GroupNormConfig::new(gn_groups, gn_channels)
            .with_epsilon(1e-5)
            .init(device);
        Self { conv, gn }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        gelu(self.gn.forward(self.conv.forward(x)))
    }
}

#[derive(Module, Debug)]
pub struct PatchEmbedding<B: Backend> {
    pub conv1: ConvGnGelu<B>,
    pub conv2: ConvGnGelu<B>,
    pub conv3: ConvGnGelu<B>,
    pub spectral_linear: Linear<B>,
    pub pos_conv: Conv2d<B>,
    pub patch_size: usize,
    pub d_model: usize,
}

impl<B: Backend> PatchEmbedding<B> {
    pub fn new(patch_size: usize, device: &B::Device) -> Self {
        // Default: channels_kernel_stride_padding_norm = (25,49,25,24,(5,25)), (25,3,1,1,(5,25)), (25,3,1,1,(5,25))
        // Conv layers: Conv2d(in, out, (1,kernel), (1,stride), (0,padding))
        let conv1 = ConvGnGelu::new(1, 25, [1, 49], [1, 25], [0, 24], 5, 25, device);
        let conv2 = ConvGnGelu::new(25, 25, [1, 3], [1, 1], [0, 1], 5, 25, device);
        let conv3 = ConvGnGelu::new(25, 25, [1, 3], [1, 1], [0, 1], 5, 25, device);

        // d_model computation: after conv pipeline on patch_size=200:
        // (200 + 2*24 - 49) / 25 + 1 = 199/25 + 1 = 7+1 = 8
        // (8 + 2*1 - 3) / 1 + 1 = 8
        // (8 + 2*1 - 3) / 1 + 1 = 8
        // d_model = 25 * 8 = 200
        let d_model = 200; // 25 channels * 8 spatial

        let n_freq = patch_size / 2 + 1; // 101 for patch_size=200
        let spectral_linear = LinearConfig::new(n_freq, d_model).with_bias(true).init(device);

        // Depthwise Conv2d for positional encoding: (d_model, d_model, (19,7), groups=d_model, pad=(9,3))
        let pos_conv = Conv2dConfig::new([d_model, d_model], [19, 7])
            .with_stride([1, 1])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(9, 3))
            .with_groups(d_model)
            .with_bias(true)
            .init(device);

        Self { conv1, conv2, conv3, spectral_linear, pos_conv, patch_size, d_model }
    }

    /// x: [B, C, N, P] → [B, C, N, d_model]
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [bz, ch_num, patch_num, patch_size] = x.dims();
        let device = x.device();

        // 1. Patch embedding via conv pipeline
        // Rearrange: [B, C, N, P] → [B, 1, C*N, P]
        let x_flat = x.clone().reshape([bz, 1, ch_num * patch_num, patch_size]);
        let conv_out = self.conv3.forward(self.conv2.forward(self.conv1.forward(x_flat)));
        // conv_out: [B, 25, C*N, P2] where P2 = 8 for patch_size=200
        let [_, d_ch, _, p2] = conv_out.dims();
        // Rearrange: [B, d_ch, C*N, P2] → [B, C, N, d_ch*P2]
        let patch_emb = conv_out
            .reshape([bz, d_ch, ch_num, patch_num, p2])
            .swap_dims(1, 2) // [B, C, d_ch, N, P2]
            .swap_dims(2, 3) // [B, C, N, d_ch, P2]
            .reshape([bz, ch_num, patch_num, d_ch * p2]);

        // 2. Spectral embedding via FFT
        let spectral_emb = self.compute_spectral(x, bz, ch_num, patch_num, &device);
        // [B, C, N, d_model]

        // 3. Sum
        let patch_emb = patch_emb + spectral_emb;

        // 4. Positional encoding via depthwise conv
        // Rearrange: [B, C, N, d_model] → [B, d_model, C, N]
        let pos_input = patch_emb.clone()
            .swap_dims(1, 3) // [B, d_model, N, C]
            .swap_dims(2, 3); // [B, d_model, C, N]
        let pos_emb = self.pos_conv.forward(pos_input);
        // [B, d_model, C, N] → [B, C, N, d_model]
        let pos_emb = pos_emb
            .swap_dims(2, 3) // [B, d_model, N, C]
            .swap_dims(1, 3); // [B, C, N, d_model]

        patch_emb + pos_emb
    }

    fn compute_spectral(
        &self,
        x: Tensor<B, 4>,
        bz: usize, ch_num: usize, patch_num: usize,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        let n_elements = bz * ch_num * patch_num;
        let p = self.patch_size;
        let n_freq = p / 2 + 1;

        // Extract data for FFT
        let x_flat = x.reshape([n_elements, p]);
        let tensor_data = x_flat.into_data();
        let x_f32: Vec<f32> = tensor_data.to_vec::<f64>()
            .map(|v| v.into_iter().map(|x| x as f32).collect())
            .or_else(|_| tensor_data.to_vec::<f32>())
            .expect("extract tensor data");

        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(p);

        // Python: norm="forward" means divide by N
        let norm_factor = 1.0 / p as f64;
        let mut magnitudes = vec![0.0f32; n_elements * n_freq];

        for i in 0..n_elements {
            let mut buf: Vec<Complex<f64>> = x_f32[i * p..(i + 1) * p]
                .iter()
                .map(|&v| Complex { re: v as f64, im: 0.0 })
                .collect();
            fft.process(&mut buf);

            for k in 0..n_freq {
                let re = buf[k].re * norm_factor;
                let im = buf[k].im * norm_factor;
                magnitudes[i * n_freq + k] = (re * re + im * im).sqrt() as f32;
            }
        }

        // [n_elements, n_freq] → Linear → [n_elements, d_model]
        let spec_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(magnitudes, vec![n_elements, n_freq]),
            device,
        );
        let spec_emb = self.spectral_linear.forward(spec_tensor);
        spec_emb.reshape([bz, ch_num, patch_num, self.d_model])
    }
}
