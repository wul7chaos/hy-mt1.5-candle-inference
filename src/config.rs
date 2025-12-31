use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub use_qk_norm: bool,
    pub tie_word_embeddings: bool,
    pub head_dim: usize,
    pub eos_token_id: usize,
}
