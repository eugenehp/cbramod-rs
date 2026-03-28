//! # cbramod-rs — CBraMod EEG Foundation Model inference in Rust
//!
//! Pure-Rust inference for the CBraMod (Criss-Cross Brain Model)
//! EEG foundation model, built on [Burn 0.20](https://burn.dev).
//!
//! CBraMod uses criss-cross attention to separately model spatial (channel)
//! and temporal (patch) dependencies, with asymmetric conditional positional
//! encoding via depthwise convolution.

pub mod config;
pub mod model;
pub mod weights;

pub use config::ModelConfig;
pub use weights::WeightMap;
