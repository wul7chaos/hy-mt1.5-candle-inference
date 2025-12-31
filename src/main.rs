mod config;
mod model;
mod utils;

use anyhow::Result;
use axum::{
    extract::State,
    routing::post,
    Json, Router,
};
use candle_core::{Device, Tensor};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tower_http::cors::CorsLayer;
use tracing::{info, warn};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use crate::config::Config;
use crate::model::{HunyuanModel, LayerCache};
use crate::utils::load_safetensors_manual;

const VERSION: &str = "0.2.0";

#[derive(Parser, Debug)]
#[command(author, version = VERSION, about = "AI Translate Service (OpenAI API Compatible)")]
struct Args {
    /// 模型路径
    #[arg(short, long, default_value = "HY-MT1.5-1.8B-FP8")]
    model_path: PathBuf,

    /// 监听地址
    #[arg(short, long, default_value = "0.0.0.0:8000")]
    addr: String,

    /// 是否运行性能测试
    #[arg(short, long, default_value_t = false)]
    test: bool,

    /// 强制使用 CPU
    #[arg(long, default_value_t = false)]
    cpu: bool,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    messages: Vec<ChatMessage>,
    #[serde(default = "default_model")]
    model: String,
    #[allow(dead_code)]
    #[serde(default)]
    stream: bool,
}

fn default_model() -> String {
    "hy-translate".to_string()
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct ChatChoice {
    index: usize,
    message: ChatMessage,
    finish_reason: String,
}

#[derive(Debug, Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Debug, Serialize)]
struct ModelList {
    object: String,
    data: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
struct ModelInfo {
    id: String,
    object: String,
    created: u64,
    owned_by: String,
}

struct AppState {
    model: HunyuanModel,
    tokenizer: Tokenizer,
    config: Config,
    device: Device,
}

#[tokio::main]
async fn main() -> Result<()> {
    // 初始化日志
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .init();

    let args = Args::parse();

    info!("========================================");
    info!("AI Translate v{} - 程序信息: Hunyuan-MT 1.5 翻译服务", VERSION);
    info!("========================================");

    // 设备检测
    let device = if args.cpu {
        info!("提示: 强制使用 CPU 模式");
        Device::Cpu
    } else if candle_core::utils::cuda_is_available() {
        let d = Device::new_cuda(0)?;
        info!("提示: 检测到 GPU 可用，正在使用设备: {:?}", d);
        d
    } else {
        warn!("提示: 未检测到可用 GPU，回退到 CPU 模式");
        Device::Cpu
    };

    // 加载配置
    info!("正在加载模型配置: {:?}/config.json", args.model_path);
    let config_path = args.model_path.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)?;
    let config: Config = serde_json::from_str(&config_str)?;
    info!("配置加载成功: 隐藏层数={}, 词表大小={}", config.num_hidden_layers, config.vocab_size);

    // 加载分词器
    info!("正在加载分词器: {:?}/tokenizer.json", args.model_path);
    let tokenizer_path = args.model_path.join("tokenizer.json");
    let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("加载分词器失败: {}", e))?;
    tokenizer.with_padding(None);
    let _ = tokenizer.with_truncation(None);
    info!("分词器加载成功");

    // 加载模型权重 (只需加载一次)
    info!("正在加载权重文件: {:?}/model.safetensors", args.model_path);
    let weights_path = args.model_path.join("model.safetensors");
    let load_start = std::time::Instant::now();
    let vb = load_safetensors_manual(&weights_path, &device)?;
    let load_duration = load_start.elapsed();
    info!("权重加载成功，耗时: {:?}", load_duration);

    info!("正在初始化模型架构...");
    let init_start = std::time::Instant::now();
    let model = HunyuanModel::new(vb, &config)?;
    let init_duration = init_start.elapsed();
    info!("模型初始化成功，耗时: {:?}", init_duration);
    info!("模型加载完成，总耗时: {:?}", load_duration + init_duration);

    let state = Arc::new(AppState {
        model,
        tokenizer,
        config,
        device,
    });

    // 性能测试
    if args.test {
        info!("正在运行性能测试 (test=true)...");
        run_performance_test(state.clone()).await?;
    } else {
        info!("性能测试功能已禁用 (默认)");
    }

    // 设置路由
    let app = Router::new()
        .route("/v1/chat/completions", post(v1_chat_completions))
        .route("/v1/models", axum::routing::get(v1_models))
        .layer(CorsLayer::permissive())
        .with_state(state);

    info!("API 服务启动成功!");
    info!("监听地址: {}", args.addr);
    info!("OpenAI 兼容端点: http://{}/v1/chat/completions", args.addr);
    
    let listener = tokio::net::TcpListener::bind(&args.addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn v1_models() -> Json<ModelList> {
    Json(ModelList {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: "hy-translate".to_string(),
            object: "model".to_string(),
            created: 1700000000,
            owned_by: "hy".to_string(),
        }],
    })
}

async fn v1_chat_completions(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChatCompletionRequest>,
) -> Json<ChatCompletionResponse> {
    // 构造 Prompt：将所有消息拼接，模拟对话
    let mut prompt = String::new();
    for msg in &payload.messages {
        if msg.role == "user" {
            prompt.push_str(&msg.content);
        } else if msg.role == "system" {
            // 系统提示词放在前面
            prompt = format!("{}\n\n{}", msg.content, prompt);
        }
    }
    
    info!("接收到翻译请求，Prompt 长度: {}", prompt.len());
    
    // 使用 spawn_blocking 处理计算密集型的推理任务
    let state_clone = state.clone();
    let (translated_text, prompt_tokens, completion_tokens) = match tokio::task::spawn_blocking(move || {
        do_translate_sync(&state_clone, prompt)
    }).await {
        Ok(Ok(res)) => res,
        Ok(Err(e)) => {
            warn!("翻译过程中出错: {}", e);
            (format!("翻译错误: {}", e), 0, 0)
        },
        Err(e) => {
            warn!("任务调度出错: {}", e);
            (format!("服务繁忙: {}", e), 0, 0)
        }
    };

    info!("翻译完成: {} tokens -> {} tokens", prompt_tokens, completion_tokens);

    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{}", chrono::Utc::now().timestamp()),
        object: "chat.completion".to_string(),
        created: chrono::Utc::now().timestamp() as u64,
        model: payload.model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: translated_text,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    };

    Json(response)
}

fn do_translate_sync(state: &AppState, prompt: String) -> Result<(String, usize, usize)> {
    let mut tokens = vec![120000, 120006]; // <｜hy_begin▁of▁sentence｜>, <｜hy_User｜>
    let content_tokens = state.tokenizer.encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("分词失败: {}", e))?;
    let prompt_tokens_count = content_tokens.len();
    tokens.extend(content_tokens.get_ids());
    tokens.push(120007); // <｜hy_Assistant｜>
    
    let mut index = 0;
    let mut generated_tokens = 0;
    let mut layer_caches: Vec<Option<LayerCache>> = (0..state.config.num_hidden_layers).map(|_| None).collect();
    let mut result_text = String::new();

    // 限制最大生成长度
    for _i in 0..1024 {
        let input = Tensor::new(&tokens[index..], &state.device)?.unsqueeze(0)?;
        let logits = state.model.forward(&input, index, &mut layer_caches)?;
        let logits = logits.squeeze(0)?.squeeze(0)?;
        
        let next_token = logits.argmax(0)?.to_scalar::<u32>()?;
        
        if next_token == state.config.eos_token_id as u32 { break; }
        
        tokens.push(next_token);
        index = tokens.len() - 1;
        generated_tokens += 1;

        let decoded = state.tokenizer.decode(&[next_token], true)
            .map_err(|e| anyhow::anyhow!("解码失败: {}", e))?;
        result_text.push_str(&decoded);
    }

    Ok((result_text, prompt_tokens_count, generated_tokens))
}

async fn run_performance_test(state: Arc<AppState>) -> Result<()> {
    let prompt_content = "It's on the house.";
    let target_language = "中文";
    let prompt = format!("将以下文本翻译为{}，注意只需要输出翻译后的结果，不要额外解释：\n\n{}", target_language, prompt_content);
    
    info!("测试 Prompt: {}", prompt);
    let start = std::time::Instant::now();
    let state_clone = state.clone();
    let (res, p_tokens, c_tokens) = tokio::task::spawn_blocking(move || {
        do_translate_sync(&state_clone, prompt)
    }).await??;
    let duration = start.elapsed();
    
    info!("测试结果: {}", res);
    info!("性能统计: Prompt Tokens: {}, Completion Tokens: {}, 耗时: {:?}, 吞吐量: {:.2} tokens/s", 
        p_tokens, c_tokens, duration, c_tokens as f64 / duration.as_secs_f64());
    
    Ok(())
}
