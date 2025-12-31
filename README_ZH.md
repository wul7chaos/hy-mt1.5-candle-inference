# hy-mt1.5-candle-inference

[中文](README_ZH.md) | [English Version](README.md)

基于 [Candle](https://github.com/huggingface/candle) 机器学习框架的腾讯 **混元翻译 (Hunyuan-MT) 1.5** 模型高性能 Rust 推理实现。

本项目提供了一个 OpenAI 兼容的 API 接口和命令行工具，旨在实现高效的翻译任务。

## 🔗 官方项目
本项目基于腾讯混元官方开源项目实现：
[Tencent-Hunyuan/HY-MT](https://github.com/Tencent-Hunyuan/HY-MT)

## ✨ 功能特性
- **高性能**: 使用 Rust 和 Candle 框架开发，提供极低的推理延迟。
- **GPU 加速**: 支持 CUDA，实现高速翻译。
- **自动回退**: 如果未检测到可用 GPU，程序将自动回退到 CPU 模式。
- **OpenAI 兼容**: 提供 `/v1/chat/completions` 接口，兼容 OpenAI API 格式。
- **内存优化**: 模型权重仅在启动时加载一次到内存/显存，显著提升后续请求响应速度。
- **命令行控制**: 支持丰富的命令行参数，方便进行配置和测试。
- **性能测试**: 内置测试模式，可实时评估硬件的吞吐量和延迟。

## 🚀 快速上手

### 环境要求
- Rust (最新稳定版)
- CUDA Toolkit (可选，用于 GPU 加速)
- 混元 1.5 模型权重 (例如：`HY-MT1.5-1.8B-FP8`)

### 安装步骤
1. 克隆仓库:
   ```bash
   git clone https://github.com/your-username/hy-mt1.5-candle-inference.git
   cd hy-mt1.5-candle-inference
   ```
2. 编译项目:
   ```bash
   cargo build --release
   ```

### 使用方法

#### 启动 API 服务
```bash
./target/release/ai_translate --model-path path/to/model --addr 0.0.0.0:8000
```

#### 命令行参数说明
- `-m, --model-path <PATH>`: 模型目录路径（默认：`HY-MT1.5-1.8B-FP8`）。
- `-a, --addr <ADDR>`: 服务监听地址（默认：`0.0.0.0:8000`）。
- `-t, --test`: 启动时运行一次性能测试。
- `--cpu`: 强制使用 CPU，即使 GPU 可用。

### API 调用示例
您可以使用任何支持 OpenAI 格式的客户端调用本服务。

**请求示例 (cURL):**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "将以下文本翻译为中文：It is on the house."}
    ]
  }'
```

## 📊 性能表现
1.8B FP8 模型在支持 CUDA 的 GPU 上表现优异。您可以使用 `--test` 标志在您的硬件上测量实际性能。

## 📄 开源协议
模型相关的许可信息请参考[官方混元翻译项目](https://github.com/Tencent-Hunyuan/HY-MT)。
