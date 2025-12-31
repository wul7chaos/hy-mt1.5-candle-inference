use anyhow::Result;
use candle_core::{Device, Tensor, DType};
use candle_nn::VarBuilder;
use std::path::Path;
use safetensors::SafeTensors;

pub fn fp8_e4m3_to_f32(bits: u8) -> f32 {
    let s = (bits >> 7) as i32;
    let e = ((bits >> 3) & 0x0F) as i32;
    let m = (bits & 0x07) as i32;

    if e == 0 {
        // Subnormal
        (if s == 0 { 1.0 } else { -1.0 }) * (m as f32) * (2.0f32.powi(-6 - 3))
    } else if e == 15 && m == 7 {
        f32::NAN
    } else {
        // Normal
        (if s == 0 { 1.0 } else { -1.0 }) * (1.0 + (m as f32) / 8.0) * (2.0f32.powi(e - 7))
    }
}

pub fn load_safetensors_manual(path: &Path, device: &Device) -> Result<VarBuilder<'static>> {
    let file = std::fs::read(path)?;
    let st = SafeTensors::deserialize(&file)?;
    let mut tensors = std::collections::HashMap::new();

    for (name, view) in st.tensors() {
        let dtype = view.dtype();
        let shape = view.shape();
        let data = view.data();

        let tensor = match dtype {
            safetensors::Dtype::F32 => {
                let f16_data: Vec<half::f16> = data
                    .chunks_exact(4)
                    .map(|c| half::f16::from_f32(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                    .collect();
                Tensor::from_vec(f16_data, shape, device)?
            }
            safetensors::Dtype::F16 => {
                let f16_data: Vec<half::f16> = data
                    .chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]))
                    .collect();
                Tensor::from_vec(f16_data, shape, device)?
            }
            safetensors::Dtype::BF16 => {
                let f16_data: Vec<half::f16> = data
                    .chunks_exact(2)
                    .map(|c| half::f16::from_f32(half::bf16::from_le_bytes([c[0], c[1]]).to_f32()))
                    .collect();
                Tensor::from_vec(f16_data, shape, device)?
            }
            safetensors::Dtype::F8_E4M3 => {
                let f16_data: Vec<half::f16> = data.iter().map(|&b| half::f16::from_f32(fp8_e4m3_to_f32(b))).collect();
                Tensor::from_vec(f16_data, shape, device)?
            }
            _ => {
                return Err(anyhow::anyhow!("不支持的张量 {} 数据类型 {:?}", name, dtype));
            }
        };
        tensors.insert(name.to_string(), tensor);
    }

    Ok(VarBuilder::from_tensors(tensors, DType::F16, device))
}

pub fn repeat_kv(x: Tensor, n_rep: usize) -> candle_core::Result<Tensor> {
    if n_rep == 1 {
        Ok(x)
    } else {
        let (b_sz, n_kv_heads, seq_len, head_dim) = x.dims4()?;
        x.unsqueeze(2)?
            .expand((b_sz, n_kv_heads, n_rep, seq_len, head_dim))?
            .reshape((b_sz, n_kv_heads * n_rep, seq_len, head_dim))
    }
}
