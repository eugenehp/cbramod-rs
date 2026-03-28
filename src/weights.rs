/// Load CBraMod weights from safetensors.

use std::collections::HashMap;
use burn::prelude::*;
use half::bf16;
use safetensors::SafeTensors;
use crate::model::cbramod::CBraMod;
use crate::config::ModelConfig;

pub struct WeightMap {
    pub tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl WeightMap {
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)?;
        let st = SafeTensors::deserialize(&bytes)?;
        let mut tensors = HashMap::with_capacity(st.len());
        for (key, view) in st.tensors() {
            let key = key.strip_prefix("model.").unwrap_or(&key).to_string();
            let shape: Vec<usize> = view.shape().to_vec();
            let data = view.data();
            let f32s: Vec<f32> = match view.dtype() {
                safetensors::Dtype::BF16 => data.chunks_exact(2)
                    .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32()).collect(),
                safetensors::Dtype::F32 => data.chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
                safetensors::Dtype::F16 => data.chunks_exact(2)
                    .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32()).collect(),
                other => anyhow::bail!("unsupported dtype {:?} for key {key}", other),
            };
            tensors.insert(key, (f32s, shape));
        }
        Ok(Self { tensors })
    }

    pub fn take<B: Backend, const N: usize>(&mut self, key: &str, device: &B::Device) -> anyhow::Result<Tensor<B, N>> {
        let (data, shape) = self.tensors.remove(key)
            .ok_or_else(|| anyhow::anyhow!("key not found: {key}"))?;
        if shape.len() != N { anyhow::bail!("rank mismatch for {key}: expected {N}, got {}", shape.len()); }
        Ok(Tensor::<B, N>::from_data(TensorData::new(data, shape), device))
    }

    pub fn has(&self, key: &str) -> bool { self.tensors.contains_key(key) }

    pub fn print_keys(&self) {
        let mut keys: Vec<&str> = self.tensors.keys().map(String::as_str).collect();
        keys.sort();
        for k in keys { let (_, s) = &self.tensors[k]; println!("  {k:70}  {s:?}"); }
    }
}

fn set_linear_wb<B: Backend>(l: &mut burn::nn::Linear<B>, w: Tensor<B, 2>, b: Tensor<B, 1>) {
    l.weight = l.weight.clone().map(|_| w.transpose());
    if let Some(ref bias) = l.bias { l.bias = Some(bias.clone().map(|_| b)); }
}

#[allow(dead_code)]
fn set_linear_w<B: Backend>(l: &mut burn::nn::Linear<B>, w: Tensor<B, 2>) {
    l.weight = l.weight.clone().map(|_| w.transpose());
}

fn set_layernorm<B: Backend>(n: &mut burn::nn::LayerNorm<B>, w: Tensor<B, 1>, b: Tensor<B, 1>) {
    n.gamma = n.gamma.clone().map(|_| w);
    if let Some(ref beta) = n.beta { n.beta = Some(beta.clone().map(|_| b)); }
}

fn set_conv2d_wb<B: Backend>(c: &mut burn::nn::conv::Conv2d<B>, w: Tensor<B, 4>, b: Tensor<B, 1>) {
    c.weight = c.weight.clone().map(|_| w);
    if let Some(ref bias) = c.bias { c.bias = Some(bias.clone().map(|_| b)); }
}

fn set_groupnorm<B: Backend>(g: &mut burn::nn::GroupNorm<B>, w: Tensor<B, 1>, b: Tensor<B, 1>) {
    if let Some(ref gamma) = g.gamma { g.gamma = Some(gamma.clone().map(|_| w)); }
    if let Some(ref beta) = g.beta { g.beta = Some(beta.clone().map(|_| b)); }
}

pub fn load_model<B: Backend>(cfg: &ModelConfig, path: &str, device: &B::Device) -> anyhow::Result<CBraMod<B>> {
    let mut wm = WeightMap::from_file(path)?;
    eprintln!("Loading {} weight tensors...", wm.tensors.len());
    load_model_from_wm(cfg, &mut wm, device)
}

pub fn load_model_from_wm<B: Backend>(cfg: &ModelConfig, wm: &mut WeightMap, device: &B::Device) -> anyhow::Result<CBraMod<B>> {
    let mut model = CBraMod::new(
        cfg.n_outputs, cfg.n_chans, cfg.n_times, cfg.patch_size,
        cfg.dim_feedforward, cfg.n_layer, cfg.nhead, cfg.emb_dim, false, device,
    );
    load_weights(wm, &mut model, cfg, device)?;
    Ok(model)
}

fn load_weights<B: Backend>(wm: &mut WeightMap, model: &mut CBraMod<B>, cfg: &ModelConfig, device: &B::Device) -> anyhow::Result<()> {
    // Patch embedding conv layers
    macro_rules! load_conv_gn {
        ($block:expr, $conv_key:expr, $gn_key:expr) => {
            if let (Ok(w), Ok(b)) = (wm.take::<B,4>(&format!("{}.weight", $conv_key), device),
                                     wm.take::<B,1>(&format!("{}.bias", $conv_key), device)) {
                set_conv2d_wb(&mut $block.conv, w, b);
            }
            if let (Ok(w), Ok(b)) = (wm.take::<B,1>(&format!("{}.weight", $gn_key), device),
                                     wm.take::<B,1>(&format!("{}.bias", $gn_key), device)) {
                set_groupnorm(&mut $block.gn, w, b);
            }
        };
    }
    load_conv_gn!(model.patch_embedding.conv1, "patch_embedding.proj_in.0", "patch_embedding.proj_in.1");
    load_conv_gn!(model.patch_embedding.conv2, "patch_embedding.proj_in.3", "patch_embedding.proj_in.4");
    load_conv_gn!(model.patch_embedding.conv3, "patch_embedding.proj_in.6", "patch_embedding.proj_in.7");

    // Spectral projection
    if let (Ok(w), Ok(b)) = (wm.take::<B,2>("patch_embedding.spectral_proj.0.weight", device),
                             wm.take::<B,1>("patch_embedding.spectral_proj.0.bias", device)) {
        set_linear_wb(&mut model.patch_embedding.spectral_linear, w, b);
    }

    // Positional encoding conv
    if let (Ok(w), Ok(b)) = (wm.take::<B,4>("patch_embedding.positional_encoding.0.weight", device),
                             wm.take::<B,1>("patch_embedding.positional_encoding.0.bias", device)) {
        set_conv2d_wb(&mut model.patch_embedding.pos_conv, w, b);
    }

    // Encoder layers
    for i in 0..cfg.n_layer {
        let layer = &mut model.encoder.layers[i];
        let p = format!("encoder.layers.{i}");

        // S-Attention
        if let (Ok(w), Ok(b)) = (wm.take::<B,2>(&format!("{p}.self_attn_s.in_proj_weight"), device),
                                 wm.take::<B,1>(&format!("{p}.self_attn_s.in_proj_bias"), device)) {
            set_linear_wb(&mut layer.self_attn_s.in_proj, w, b);
        }
        if let (Ok(w), Ok(b)) = (wm.take::<B,2>(&format!("{p}.self_attn_s.out_proj.weight"), device),
                                 wm.take::<B,1>(&format!("{p}.self_attn_s.out_proj.bias"), device)) {
            set_linear_wb(&mut layer.self_attn_s.out_proj, w, b);
        }

        // T-Attention
        if let (Ok(w), Ok(b)) = (wm.take::<B,2>(&format!("{p}.self_attn_t.in_proj_weight"), device),
                                 wm.take::<B,1>(&format!("{p}.self_attn_t.in_proj_bias"), device)) {
            set_linear_wb(&mut layer.self_attn_t.in_proj, w, b);
        }
        if let (Ok(w), Ok(b)) = (wm.take::<B,2>(&format!("{p}.self_attn_t.out_proj.weight"), device),
                                 wm.take::<B,1>(&format!("{p}.self_attn_t.out_proj.bias"), device)) {
            set_linear_wb(&mut layer.self_attn_t.out_proj, w, b);
        }

        // FFN
        if let (Ok(w), Ok(b)) = (wm.take::<B,2>(&format!("{p}.linear1.weight"), device),
                                 wm.take::<B,1>(&format!("{p}.linear1.bias"), device)) {
            set_linear_wb(&mut layer.linear1, w, b);
        }
        if let (Ok(w), Ok(b)) = (wm.take::<B,2>(&format!("{p}.linear2.weight"), device),
                                 wm.take::<B,1>(&format!("{p}.linear2.bias"), device)) {
            set_linear_wb(&mut layer.linear2, w, b);
        }

        // Norms
        if let (Ok(w), Ok(b)) = (wm.take::<B,1>(&format!("{p}.norm1.weight"), device),
                                 wm.take::<B,1>(&format!("{p}.norm1.bias"), device)) {
            set_layernorm(&mut layer.norm1, w, b);
        }
        if let (Ok(w), Ok(b)) = (wm.take::<B,1>(&format!("{p}.norm2.weight"), device),
                                 wm.take::<B,1>(&format!("{p}.norm2.bias"), device)) {
            set_layernorm(&mut layer.norm2, w, b);
        }
    }

    // proj_out
    if let (Ok(w), Ok(b)) = (wm.take::<B,2>("proj_out.0.weight", device),
                             wm.take::<B,1>("proj_out.0.bias", device)) {
        set_linear_wb(&mut model.proj_out, w, b);
    }

    // final_layer (Flatten → LazyLinear mapped to final_layer.1)
    if let (Ok(w), Ok(b)) = (wm.take::<B,2>("final_layer.1.weight", device),
                             wm.take::<B,1>("final_layer.1.bias", device)) {
        set_linear_wb(&mut model.final_linear, w, b);
    }

    Ok(())
}
