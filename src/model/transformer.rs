/// Transformer Encoder stack for CBraMod.

use burn::prelude::*;
use crate::model::criss_cross_attention::CrissCrossEncoderLayer;

#[derive(Module, Debug)]
pub struct TransformerEncoder<B: Backend> {
    pub layers: Vec<CrissCrossEncoderLayer<B>>,
}

impl<B: Backend> TransformerEncoder<B> {
    pub fn new(
        d_model: usize, nhead: usize, dim_feedforward: usize,
        num_layers: usize, device: &B::Device,
    ) -> Self {
        let layers = (0..num_layers)
            .map(|_| CrissCrossEncoderLayer::new(d_model, nhead, dim_feedforward, device))
            .collect();
        Self { layers }
    }

    /// x: [B, C, N, D] → [B, C, N, D]
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(x);
        }
        x
    }
}
