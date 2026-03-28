use burn::prelude::*;
use std::time::Instant;

#[cfg(feature = "ndarray")]
mod backend {
    pub use burn::backend::NdArray as B;
    pub fn device() -> burn::backend::ndarray::NdArrayDevice { burn::backend::ndarray::NdArrayDevice::Cpu }
    #[cfg(feature = "blas-accelerate")]
    pub const NAME: &str = "ndarray-accelerate";
    #[cfg(not(feature = "blas-accelerate"))]
    pub const NAME: &str = "ndarray";
}
#[cfg(all(feature = "wgpu", not(feature = "ndarray")))]
mod backend {
    pub use burn::backend::Wgpu as B;
    pub fn device() -> burn::backend::wgpu::WgpuDevice { burn::backend::wgpu::WgpuDevice::DefaultDevice }
    pub const NAME: &str = "wgpu";
}
use backend::{B, device, NAME};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 { eprintln!("Usage: benchmark <n_chans> <n_times> <warmup> <repeats>"); std::process::exit(1); }
    let n_chans: usize = args[1].parse().unwrap();
    let n_times: usize = args[2].parse().unwrap();
    let warmup: usize = args[3].parse().unwrap();
    let repeats: usize = args[4].parse().unwrap();
    let dev = device();

    let model = cbramod_rs::model::cbramod::CBraMod::<B>::new(
        4, n_chans, n_times, 200, 800, 2, 8, 200, false, &dev,
    );
    let x = Tensor::<B, 3>::ones([1, n_chans, n_times], &dev).mul_scalar(0.1f32);

    for _ in 0..warmup { let _ = model.forward(x.clone()); }
    let mut times = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let t0 = Instant::now();
        let _ = model.forward(x.clone());
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let ts: Vec<String> = times.iter().map(|t| format!("{:.4}", t)).collect();
    println!("{{\"times_ms\": [{}], \"backend\": \"{}\"}}", ts.join(", "), NAME);
}
