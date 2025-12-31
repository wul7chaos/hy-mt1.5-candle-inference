# hy-mt1.5-candle-inference

High-performance Rust inference implementation for Tencent's **Hunyuan-MT 1.5** translation models using the [Candle](https://github.com/huggingface/candle) ML framework.

This project provides an OpenAI-compatible API and a command-line interface for efficient translation tasks.

## 🔗 Official Project
This implementation is based on the official Hunyuan-MT project:
[Tencent-Hunyuan/HY-MT](https://github.com/Tencent-Hunyuan/HY-MT)

## ✨ Features
- **High Performance**: Built with Rust and Candle for low-latency inference.
- **GPU Acceleration**: Supports CUDA for high-speed translation.
- **Automatic Fallback**: Automatically falls back to CPU if no compatible GPU is detected.
- **OpenAI Compatible**: Provides a `/v1/chat/completions` endpoint compatible with OpenAI's API format.
- **Efficient Loading**: Loads model weights only once into memory/VRAM for subsequent requests.
- **CLI Control**: Easy-to-use command-line arguments for configuration and testing.
- **Performance Testing**: Built-in benchmarking mode to evaluate throughput and latency.

## 🚀 Getting Started

### Prerequisites
- Rust (latest stable version)
- CUDA Toolkit (optional, for GPU acceleration)
- Hunyuan-MT 1.5 model weights (e.g., `HY-MT1.5-1.8B-FP8`)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/hy-mt1.5-candle-inference.git
   cd hy-mt1.5-candle-inference
   ```
2. Build the project:
   ```bash
   cargo build --release
   ```

### Usage

#### Start the API Server
```bash
./target/release/ai_translate --model-path path/to/model --addr 0.0.0.0:8000
```

#### CLI Options
- `-m, --model-path <PATH>`: Path to the model directory (default: `HY-MT1.5-1.8B-FP8`).
- `-a, --addr <ADDR>`: Socket address to listen on (default: `0.0.0.0:8000`).
- `-t, --test`: Run a performance test on startup.
- `--cpu`: Force the use of CPU even if GPU is available.

### API Example
You can interact with the service using any OpenAI-compatible client.

**Request:**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "将以下文本翻译为中文：It is on the house."}
    ]
  }'
```

## 📊 Performance
The 1.8B FP8 model achieves significant throughput on compatible GPUs. Use the `--test` flag to measure performance on your hardware.

## 📄 License
Please refer to the [official Hunyuan-MT project](https://github.com/Tencent-Hunyuan/HY-MT) for model licensing information.
