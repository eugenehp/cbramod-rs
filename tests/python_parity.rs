/// Python parity test for CBraMod.
/// Requires /tmp/cbramod_parity.safetensors from Python.

use burn::backend::NdArray as B;
use burn::prelude::*;
use std::collections::HashMap;

fn load_data() -> Option<HashMap<String, (Vec<f32>, Vec<usize>)>> {
    let path = "/tmp/cbramod_parity.safetensors";
    if !std::path::Path::new(path).exists() {
        eprintln!("Skipping: {path} not found"); return None;
    }
    let bytes = std::fs::read(path).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let mut m = HashMap::new();
    for (k, v) in st.tensors() {
        let shape: Vec<usize> = v.shape().to_vec();
        let f32s: Vec<f32> = v.data().chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        m.insert(k.to_string(), (f32s, shape));
    }
    Some(m)
}

fn t1(data: &HashMap<String, (Vec<f32>, Vec<usize>)>, key: &str, dev: &burn::backend::ndarray::NdArrayDevice) -> Tensor<B, 1> {
    let (d, s) = &data[key]; Tensor::from_data(TensorData::new(d.clone(), s.clone()), dev)
}
fn t2(data: &HashMap<String, (Vec<f32>, Vec<usize>)>, key: &str, dev: &burn::backend::ndarray::NdArrayDevice) -> Tensor<B, 2> {
    let (d, s) = &data[key]; Tensor::from_data(TensorData::new(d.clone(), s.clone()), dev)
}
fn t4(data: &HashMap<String, (Vec<f32>, Vec<usize>)>, key: &str, dev: &burn::backend::ndarray::NdArrayDevice) -> Tensor<B, 4> {
    let (d, s) = &data[key]; Tensor::from_data(TensorData::new(d.clone(), s.clone()), dev)
}

#[test]
fn test_python_parity() {
    let dev = burn::backend::ndarray::NdArrayDevice::Cpu;
    let data = match load_data() { Some(d) => d, None => return };

    let (inp_data, inp_shape) = &data["_input"];
    let (out_data, _out_shape) = &data["_output"];
    let n_chans = inp_shape[1];
    let n_times = inp_shape[2];

    let mut model = cbramod_rs::model::cbramod::CBraMod::<B>::new(
        4, n_chans, n_times, 200, 800, 2, 8, 200, false, &dev,
    );

    // Load weights
    macro_rules! set_linear_wb {
        ($l:expr, $wk:expr, $bk:expr) => {
            let w = t2(&data, $wk, &dev);
            let b = t1(&data, $bk, &dev);
            $l.weight = $l.weight.clone().map(|_| w.transpose());
            if let Some(ref bias) = $l.bias { $l.bias = Some(bias.clone().map(|_| b)); }
        };
    }
    macro_rules! set_conv_wb {
        ($c:expr, $wk:expr, $bk:expr) => {
            let w = t4(&data, $wk, &dev);
            let b = t1(&data, $bk, &dev);
            $c.weight = $c.weight.clone().map(|_| w);
            if let Some(ref bias) = $c.bias { $c.bias = Some(bias.clone().map(|_| b)); }
        };
    }
    macro_rules! set_gn {
        ($g:expr, $wk:expr, $bk:expr) => {
            let w = t1(&data, $wk, &dev);
            let b = t1(&data, $bk, &dev);
            if let Some(ref gamma) = $g.gamma { $g.gamma = Some(gamma.clone().map(|_| w)); }
            if let Some(ref beta) = $g.beta { $g.beta = Some(beta.clone().map(|_| b)); }
        };
    }
    macro_rules! set_ln {
        ($n:expr, $wk:expr, $bk:expr) => {
            let w = t1(&data, $wk, &dev);
            let b = t1(&data, $bk, &dev);
            $n.gamma = $n.gamma.clone().map(|_| w);
            if let Some(ref beta) = $n.beta { $n.beta = Some(beta.clone().map(|_| b)); }
        };
    }

    // Patch embedding
    set_conv_wb!(model.patch_embedding.conv1.conv, "patch_embedding.proj_in.0.weight", "patch_embedding.proj_in.0.bias");
    set_gn!(model.patch_embedding.conv1.gn, "patch_embedding.proj_in.1.weight", "patch_embedding.proj_in.1.bias");
    set_conv_wb!(model.patch_embedding.conv2.conv, "patch_embedding.proj_in.3.weight", "patch_embedding.proj_in.3.bias");
    set_gn!(model.patch_embedding.conv2.gn, "patch_embedding.proj_in.4.weight", "patch_embedding.proj_in.4.bias");
    set_conv_wb!(model.patch_embedding.conv3.conv, "patch_embedding.proj_in.6.weight", "patch_embedding.proj_in.6.bias");
    set_gn!(model.patch_embedding.conv3.gn, "patch_embedding.proj_in.7.weight", "patch_embedding.proj_in.7.bias");
    set_linear_wb!(model.patch_embedding.spectral_linear, "patch_embedding.spectral_proj.0.weight", "patch_embedding.spectral_proj.0.bias");
    set_conv_wb!(model.patch_embedding.pos_conv, "patch_embedding.positional_encoding.0.weight", "patch_embedding.positional_encoding.0.bias");

    // Encoder layers
    for i in 0..2 {
        let layer = &mut model.encoder.layers[i];
        let p = format!("encoder.layers.{i}");
        set_linear_wb!(layer.self_attn_s.in_proj, &format!("{p}.self_attn_s.in_proj_weight"), &format!("{p}.self_attn_s.in_proj_bias"));
        set_linear_wb!(layer.self_attn_s.out_proj, &format!("{p}.self_attn_s.out_proj.weight"), &format!("{p}.self_attn_s.out_proj.bias"));
        set_linear_wb!(layer.self_attn_t.in_proj, &format!("{p}.self_attn_t.in_proj_weight"), &format!("{p}.self_attn_t.in_proj_bias"));
        set_linear_wb!(layer.self_attn_t.out_proj, &format!("{p}.self_attn_t.out_proj.weight"), &format!("{p}.self_attn_t.out_proj.bias"));
        set_linear_wb!(layer.linear1, &format!("{p}.linear1.weight"), &format!("{p}.linear1.bias"));
        set_linear_wb!(layer.linear2, &format!("{p}.linear2.weight"), &format!("{p}.linear2.bias"));
        set_ln!(layer.norm1, &format!("{p}.norm1.weight"), &format!("{p}.norm1.bias"));
        set_ln!(layer.norm2, &format!("{p}.norm2.weight"), &format!("{p}.norm2.bias"));
    }

    set_linear_wb!(model.proj_out, "proj_out.0.weight", "proj_out.0.bias");
    set_linear_wb!(model.final_linear, "final_layer.1.weight", "final_layer.1.bias");

    // Run forward
    let input = Tensor::<B, 3>::from_data(TensorData::new(inp_data.clone(), inp_shape.clone()), &dev);
    let output = model.forward(input);
    let out_vec = output.into_data().to_vec::<f32>().unwrap();

    eprintln!("Expected: {:?}", out_data);
    eprintln!("Got:      {:?}", out_vec);

    let max_diff: f32 = out_vec.iter().zip(out_data.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!("Max diff: {:.6e}", max_diff);
    assert!(max_diff < 0.01, "Parity failed: max_diff={:.6e}", max_diff);
}
