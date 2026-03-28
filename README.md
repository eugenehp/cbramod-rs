# cbramod-rs

Pure-Rust inference for the **CBraMod** (Criss-Cross Brain Model) EEG foundation model, built on [Burn 0.20](https://burn.dev).

CBraMod is a compact (~4M params) foundation model pretrained on the Temple University Hospital EEG Corpus using masked patch reconstruction. It uses **criss-cross attention** to separately model spatial (channel) and temporal (patch) dependencies.

## Architecture

```
EEG [B, C, T]
    │
    ├─ Rearrange to patches: [B, C, N, P]
    │
    ├─ Patch Embedding
    │  ├─ Conv2d pipeline (3 layers) → time-domain features
    │  ├─ FFT → abs → Linear → spectral features
    │  └─ Depthwise Conv2d → positional encoding
    │  → [B, C, N, d_model=200]
    │
    ├─ Criss-Cross Transformer (12 layers)
    │  ├─ S-Attention: self-attention across channels (spatial)
    │  └─ T-Attention: self-attention across patches (temporal)
    │  → [B, C, N, d_model]
    │
    ├─ Linear projection → [B, C, N, emb_dim]
    │
    └─ Flatten + Linear → [B, n_outputs]
```

## Quick Start

```rust
use cbramod_rs::model::cbramod::CBraMod;

let model = CBraMod::<B>::new(
    4,    // n_outputs
    22,   // n_chans
    1000, // n_times (5s @ 200Hz)
    200,  // patch_size
    800,  // dim_feedforward
    12,   // n_layers
    8,    // nhead
    200,  // emb_dim
    false, // return_encoder_output
    &device,
);

let output = model.forward(eeg_tensor); // [B, n_outputs]
```

## Build

```bash
cargo build --release                          # CPU (NdArray)
cargo build --release --features blas-accelerate  # macOS Accelerate
cargo build --release --no-default-features --features metal  # Metal GPU
```

## Numerical Parity

Python ↔ Rust output difference: **< 3×10⁻⁶** (f32 precision limit).

## Pretrained Weights

Available on [HuggingFace](https://huggingface.co/braindecode/cbramod-pretrained).

## Citation

```bibtex
@inproceedings{wang2025cbramod,
    title     = {{CBraMod}: A Criss-Cross Brain Foundation Model for {EEG} Decoding},
    author    = {Wang, Jiquan and Zhao, Sha and Luo, Zhiling and Zhou, Yangxuan and Jiang, Haiteng and Li, Shijian and Li, Tao and Pan, Gang},
    booktitle = {The Thirteenth International Conference on Learning Representations (ICLR 2025)},
    year      = {2025},
    url       = {https://arxiv.org/abs/2412.07236}
}

@software{hauptmann2025cbramodrustinference,
    title     = {cbramod-rs: {CBraMod} {EEG} Foundation Model Inference in Rust},
    author    = {Hauptmann, Eugene},
    year      = {2025},
    url       = {https://github.com/eugenehp/cbramod-rs},
    version   = {0.0.1}
}
```

## Author

[Eugene Hauptmann](https://github.com/eugenehp)

## License

Apache-2.0
