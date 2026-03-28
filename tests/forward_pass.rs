use burn::backend::NdArray as B;
use burn::prelude::*;

#[test]
fn test_forward_basic() {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let model = cbramod_rs::model::cbramod::CBraMod::<B>::new(
        4, 8, 1000, 200, 800, 2, 8, 200, false, &device,
    );
    let x = Tensor::<B, 3>::ones([1, 8, 1000], &device).mul_scalar(0.1);
    let out = model.forward(x);
    assert_eq!(out.dims(), [1, 4]);
    eprintln!("Output shape: {:?}", out.dims());
}

#[test]
fn test_different_channels() {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    for n_chans in [4, 16, 22] {
        let model = cbramod_rs::model::cbramod::CBraMod::<B>::new(
            4, n_chans, 1000, 200, 800, 2, 8, 200, false, &device,
        );
        let x = Tensor::<B, 3>::ones([1, n_chans, 1000], &device).mul_scalar(0.1);
        let out = model.forward(x);
        assert_eq!(out.dims(), [1, 4]);
        eprintln!("n_chans={n_chans}: {:?}", out.dims());
    }
}
