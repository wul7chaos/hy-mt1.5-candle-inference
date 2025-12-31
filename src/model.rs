use anyhow::Result;
use candle_core::{Device, Tensor, DType, D};
use candle_nn::{VarBuilder, Module, Embedding, RmsNorm};
use candle_nn::ops::softmax;
use crate::config::Config;
use crate::utils::repeat_kv;

// Helper to load Linear layers that might have scales
pub struct QLinear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl QLinear {
    pub fn new(vb: VarBuilder, in_dim: usize, out_dim: usize, name: &str) -> Result<Self> {
        let w_name = if name.is_empty() { "weight".to_string() } else { format!("{}.weight", name) };
        let s_name = if name.is_empty() { "weight_scale".to_string() } else { format!("{}.weight_scale", name) };
        let b_name = if name.is_empty() { "bias".to_string() } else { format!("{}.bias", name) };
        
        let mut weight = vb.get((out_dim, in_dim), &w_name)?;
        let dtype = weight.dtype();

        // Dequantize weight if scale exists
        if let Ok(scale) = vb.get(out_dim, &s_name).or_else(|_| vb.get(1, &s_name)) {
            weight = weight.broadcast_mul(&scale.to_dtype(dtype)?.unsqueeze(1)?)?;
        }
        
        // Transpose and make contiguous for faster matmul
        let weight = weight.t()?.contiguous()?;
        
        let bias = vb.get(out_dim, &b_name).ok();
        
        Ok(Self { weight, bias })
    }

    pub fn from_weights(weight: Tensor, bias: Option<Tensor>) -> Self {
        let weight = weight.t().unwrap_or_else(|_| weight.clone()).contiguous().unwrap_or_else(|_| weight.clone());
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x_dims = x.dims();
        let last_dim = x_dims[x_dims.len() - 1];
        let (in_dim, out_dim) = self.weight.dims2()?;
        
        if last_dim != in_dim {
            return Err(candle_core::Error::ShapeMismatchBinaryOp {
                lhs: x.shape().clone(),
                rhs: self.weight.shape().clone(),
                op: "matmul",
            });
        }

        let res = if x.rank() == 2 {
            x.matmul(&self.weight)?
        } else {
            let batch_prod: usize = x_dims[..x_dims.len() - 1].iter().product();
            let mut out_dims = x_dims[..x_dims.len() - 1].to_vec();
            out_dims.push(out_dim);
            x.reshape((batch_prod, last_dim))?.matmul(&self.weight)?.reshape(out_dims)?
        };

        match &self.bias {
            None => Ok(res),
            Some(bias) => res.broadcast_add(bias),
        }
    }
}

pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    pub fn new(config: &Config, device: &Device) -> Result<Self> {
        let head_dim = config.head_dim;
        let max_seq_len = 4096; // 增加最大长度以防万一
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / config.rope_theta.powf(i as f32 / head_dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq, device)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, device)?.to_dtype(DType::F32)?;
        let freqs = t.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        let freqs = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    pub fn apply(&self, x: &Tensor, index: usize) -> candle_core::Result<Tensor> {
        let (_b_sz, _h, seq_len, _d) = x.dims4()?;
        
        // 优化：确保 narrow 后的 tensor 类型一致，避免不必要的转换
        let cos = self.cos.narrow(0, index, seq_len)?.to_dtype(x.dtype())?;
        let sin = self.sin.narrow(0, index, seq_len)?.to_dtype(x.dtype())?;
        
        let x1 = x.narrow(D::Minus1, 0, x.dim(D::Minus1)? / 2)?;
        let x2 = x.narrow(D::Minus1, x.dim(D::Minus1)? / 2, x.dim(D::Minus1)? / 2)?;
        let rotate_x = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
        
        // Broadcasting
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
        
        x.broadcast_mul(&cos)? + rotate_x.broadcast_mul(&sin)?
    }
}

pub struct MLP {
    gate_proj: QLinear,
    up_proj: QLinear,
    down_proj: QLinear,
}

impl MLP {
    pub fn new(vb: VarBuilder, config: &Config) -> Result<Self> {
        let gate_proj = QLinear::new(vb.pp("gate_proj"), config.hidden_size, config.intermediate_size, "")?;
        let up_proj = QLinear::new(vb.pp("up_proj"), config.hidden_size, config.intermediate_size, "")?;
        let down_proj = QLinear::new(vb.pp("down_proj"), config.intermediate_size, config.hidden_size, "")?;
        Ok(Self { gate_proj, up_proj, down_proj })
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = (candle_nn::ops::silu(&self.gate_proj.forward(x)?)? * self.up_proj.forward(x)?)?;
        self.down_proj.forward(&x)
    }
}

#[derive(Clone)]
pub struct LayerCache {
    pub k: Tensor,
    pub v: Tensor,
}

pub struct Attention {
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    query_layernorm: Option<RmsNorm>,
    key_layernorm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    pub fn new(vb: VarBuilder, config: &Config) -> Result<Self> {
        let head_dim = config.head_dim;
        let q_proj = QLinear::new(vb.pp("q_proj"), config.hidden_size, config.num_attention_heads * head_dim, "")?;
        let k_proj = QLinear::new(vb.pp("k_proj"), config.hidden_size, config.num_key_value_heads * head_dim, "")?;
        let v_proj = QLinear::new(vb.pp("v_proj"), config.hidden_size, config.num_key_value_heads * head_dim, "")?;
        let o_proj = QLinear::new(vb.pp("o_proj"), config.num_attention_heads * head_dim, config.hidden_size, "")?;
        
        let (query_layernorm, key_layernorm) = if config.use_qk_norm {
            let qn = candle_nn::rms_norm(head_dim, config.rms_norm_eps, vb.pp("query_layernorm"))?;
            let kn = candle_nn::rms_norm(head_dim, config.rms_norm_eps, vb.pp("key_layernorm"))?;
            (Some(qn), Some(kn))
        } else {
            (None, None)
        };

        Ok(Self {
            q_proj, k_proj, v_proj, o_proj,
            query_layernorm, key_layernorm,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim,
        })
    }

    pub fn forward(&self, x: &Tensor, rope: &RotaryEmbedding, index: usize, cache: &mut Option<LayerCache>) -> candle_core::Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let mut q = q.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let mut k = k.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;

        if let Some(ln) = &self.query_layernorm {
            q = ln.forward(&q)?;
        }
        if let Some(ln) = &self.key_layernorm {
            k = ln.forward(&k)?;
        }

        let q = rope.apply(&q, index)?;
        let k = rope.apply(&k, index)?;

        let (k, v) = match cache {
            Some(c) => {
                let k = Tensor::cat(&[&c.k, &k], 2)?;
                let v = Tensor::cat(&[&c.v, &v], 2)?;
                c.k = k.clone();
                c.v = v.clone();
                (k, v)
            }
            None => {
                *cache = Some(LayerCache { k: k.clone(), v: v.clone() });
                (k, v)
            }
        };

        let seq_len_full = k.dim(2)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        
        // GQA repeat
        let k = repeat_kv(k, self.num_heads / self.num_kv_heads)?;
        let v = repeat_kv(v, self.num_heads / self.num_kv_heads)?;

        let att = (q.matmul(&k.t()?)? * scale)?;

        let att = if seq_len > 1 {
            let device = att.device();
            let q_indices = Tensor::arange(0u32, seq_len as u32, device)?.to_dtype(DType::F32)?.affine(1.0, index as f64)?;
            let k_indices = Tensor::arange(0u32, seq_len_full as u32, device)?.to_dtype(DType::F32)?;
            let mask = q_indices.unsqueeze(1)?.broadcast_lt(&k_indices.unsqueeze(0)?)?;
            let on_true = Tensor::full(f32::NEG_INFINITY, mask.shape(), device)?.to_dtype(att.dtype())?;
            let on_false = Tensor::full(0.0, mask.shape(), device)?.to_dtype(att.dtype())?;
            let mask = mask.where_cond(&on_true, &on_false)?;
            att.broadcast_add(&mask)?
        } else {
            att
        };

        let att = softmax(&att, D::Minus1)?;
        let x = att.matmul(&v)?;
        let x = x.transpose(1, 2)?.reshape((b_sz, seq_len, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&x)
    }
}

pub struct Layer {
    attention: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Layer {
    pub fn new(vb: VarBuilder, config: &Config) -> Result<Self> {
        let attention = Attention::new(vb.pp("self_attn"), config)?;
        let mlp = MLP::new(vb.pp("mlp"), config)?;
        let input_layernorm = candle_nn::rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = candle_nn::rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("post_attention_layernorm"))?;
        Ok(Self { attention, mlp, input_layernorm, post_attention_layernorm })
    }

    pub fn forward(&self, x: &Tensor, rope: &RotaryEmbedding, index: usize, cache: &mut Option<LayerCache>) -> candle_core::Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = (self.attention.forward(&x, rope, index, cache)? + residual)?;
        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = (self.mlp.forward(&x)? + residual)?;
        Ok(x)
    }
}

pub struct HunyuanModel {
    embed_tokens: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    lm_head: QLinear,
    rope: RotaryEmbedding,
}

impl HunyuanModel {
    pub fn new(vb: VarBuilder, config: &Config) -> Result<Self> {
        let embed_tokens = candle_nn::embedding(config.vocab_size, config.hidden_size, vb.pp("model.embed_tokens"))?;
        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            layers.push(Layer::new(vb.pp(format!("model.layers.{}", i)), config)?);
        }
        let norm = candle_nn::rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("model.norm"))?;
        
        let lm_head = if config.tie_word_embeddings {
            QLinear::from_weights(embed_tokens.embeddings().clone(), None)
        } else {
            QLinear::new(vb.pp("lm_head"), config.hidden_size, config.vocab_size, "")?
        };

        let rope = RotaryEmbedding::new(config, vb.device())?;
        Ok(Self { embed_tokens, layers, norm, lm_head, rope })
    }

    pub fn forward(&self, input_ids: &Tensor, index: usize, caches: &mut [Option<LayerCache>]) -> candle_core::Result<Tensor> {
        let mut x = self.embed_tokens.forward(input_ids)?;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, &self.rope, index, &mut caches[i])?;
        }
        x = self.norm.forward(&x)?;
        let x = x.narrow(1, x.dim(1)? - 1, 1)?; // Only take last token for lm_head
        self.lm_head.forward(&x)
    }
}