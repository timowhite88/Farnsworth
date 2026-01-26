# Farnsworth Setup Guide 🛠️

Welcome to Farnsworth! This guide will help you get the system running on your local machine or server.

## 📋 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Windows 10+/Linux/macOS | Ubuntu 22.04 LTS |
| **Python** | 3.10 | 3.11 |
| **RAM** | 8 GB | 16+ GB (for local models) |
| **GPU** | None (Cloud only) | NVIDIA RTX 3060+ (8GB VRAM) |
| **Disk** | 10 GB | 50 GB NVMe |

---

## 🔧 Step-by-Step Installation

### 1. Environment Setup

First, ensure you have Python 3.10+ installed.

```bash
# Check Python version
python --version
```

Create a virtual environment to keep dependencies clean:

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies

Install the required Python packages.

```bash
pip install -r requirements.txt
```

> **Note**: If you want to use local GPU acceleration for `llama-cpp-python`, ensure you have the CUDA toolkit installed and install with:
> `CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python`

### 3. Local Model Setup (Ollama)

For the best "Hybrid" experience (fast local inference + smart cloud), install [Ollama](https://ollama.ai).

1.  Download & Install Ollama.
2.  Pull recommended models:

```bash
ollama pull mistral      # General reasoning
ollama pull qwen2.5-coder # Coding specialist
ollama pull nomic-embed-text # Embeddings
```

### 4. Configuration

Run the automated setup wizard. This is the easiest way to configure your `.env` file.

```bash
python main.py --setup
```

You will be asked for:
*   **Anthropic/OpenAI Keys**: Optional, for cloud fallback.
*   **Solana RPC**: Optional, for DeGen Mob features.
*   **Memory Path**: Where to store the database (default: `./data`).

---

## 🐳 Docker Setup

If you prefer containers, we provide a production-ready `docker-compose`.

```yaml
# docker-compose.yml included in repo
version: '3.8'
services:
  farnsworth:
    build: .
    ports:
      - "8501:8501" # Dashboard
      - "8000:8000" # MCP Server
    volumes:
      - ./data:/app/data
      - ./config:/app/config
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Run it:

```bash
docker-compose up -d
```

---

## 🏃‍♂️ Running Farnsworth

### Mode A: Full System (Dashboard + Core)
This starts the Streamlit UI, MCP Server, and background Swarm.

```bash
python main.py
```
> Access UI at: `http://localhost:8501`

### Mode B: MCP Server Only (for Claude Code)
Use this if you are connecting from Claude Desktop or VS Code.

```bash
python main.py --mcp
```

### Mode C: CLI Mode
Interactive terminal chat.

```bash
python main.py --cli
```

---

## 🐛 Troubleshooting

*   **"Ollama connection refused"**: Ensure Ollama is running (`ollama serve`). By default it runs on port 11434.
*   **"CUDA out of memory"**: Adjust `max_gpu_layers` in `config/models.yaml` or switch to a smaller model (e.g., `phi-2`).
*   **"Missing dependency"**: Run `pip install .` in the root directory to install the package in editable mode.

---

*"I trust this setup guide is sufficiently foolproof. If not, I can always build a robot to read it for you."*
