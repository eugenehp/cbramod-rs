/// CBraMod inference CLI.

use burn::prelude::*;
use std::time::Instant;

#[cfg(feature = "ndarray")]
mod backend {
    pub use burn::backend::NdArray as B;
    pub fn device() -> burn::backend::ndarray::NdArrayDevice { burn::backend::ndarray::NdArrayDevice::Cpu }
    pub const NAME: &str = "CPU (NdArray)";
}
#[cfg(all(feature = "wgpu", not(feature = "ndarray")))]
mod backend {
    pub use burn::backend::Wgpu as B;
    pub fn device() -> burn::backend::wgpu::WgpuDevice { burn::backend::wgpu::WgpuDevice::DefaultDevice }
    pub const NAME: &str = "GPU (wgpu)";
}
use backend::{B, device};

fn main() -> anyhow::Result<()> {
    let dev = device();
    println!("Backend: {}", backend::NAME);

    let n_chans = 22;
    let n_times = 1000;
    let model = cbramod_rs::model::cbramod::CBraMod::<B>::new(
        4, n_chans, n_times, 200, 800, 2, 8, 200, false, &dev,
    );

    let x = Tensor::<B, 3>::ones([1, n_chans, n_times], &dev).mul_scalar(0.1f32);

    let t0 = Instant::now();
    let out = model.forward(x);
    let ms = t0.elapsed().as_secs_f64() * 1000.0;

    println!("Output shape: {:?}  ({ms:.1} ms)", out.dims());
    Ok(())
}
