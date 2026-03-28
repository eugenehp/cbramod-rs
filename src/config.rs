/// Model configuration for CBraMod.

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelConfig {
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_dim_feedforward")]
    pub dim_feedforward: usize,
    #[serde(default = "default_n_layer")]
    pub n_layer: usize,
    #[serde(default = "default_nhead")]
    pub nhead: usize,
    #[serde(default = "default_emb_dim")]
    pub emb_dim: usize,
    #[serde(default)]
    pub n_outputs: usize,
    #[serde(default)]
    pub n_chans: usize,
    #[serde(default)]
    pub n_times: usize,
}

fn default_patch_size() -> usize { 200 }
fn default_dim_feedforward() -> usize { 800 }
fn default_n_layer() -> usize { 12 }
fn default_nhead() -> usize { 8 }
fn default_emb_dim() -> usize { 200 }

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            patch_size: 200, dim_feedforward: 800, n_layer: 12, nhead: 8,
            emb_dim: 200, n_outputs: 4, n_chans: 22, n_times: 1000,
        }
    }
}
