# 🧠 Farnsworth: Your Claude Companion AI
9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS
<div align="center">

**Give Claude superpowers: persistent memory, model swarms, multimodal understanding, and self-evolution.**

[![Version](https://img.shields.io/badge/version-0.5.0-blue.svg)](https://github.com/timowhite88/Farnsworth)
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Dual%20(Free%20%2B%20Commercial)-purple.svg)](LICENSE)
[![Claude Code](https://img.shields.io/badge/Claude%20Code-MCP%20Integration-orange.svg)](https://claude.ai)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](docker/)
[![Models](https://img.shields.io/badge/Models-12%2B%20Supported-green.svg)](configs/models.yaml)

[**Documentation**](docs/USER_GUIDE.md) • [**Roadmap**](ROADMAP.md) • [**Contributing**](CONTRIBUTING.md) • [**Docker**](docker/)

</div>

---

## 🎯 What is Farnsworth?

Farnsworth is a **companion AI system** that integrates with [Claude Code](https://claude.ai) to give Claude capabilities it doesn't have on its own:

| Without Farnsworth | With Farnsworth |
|:------------------:|:---------------:|
| 🚫 Claude forgets everything between sessions | ✅ Claude remembers your preferences forever |
| 🚫 Claude is a single model | ✅ **Model Swarm**: 12+ models collaborate via PSO |
| 🚫 Claude can't see images or hear audio | ✅ Multimodal: vision (CLIP/BLIP) + voice (Whisper) |
| 🚫 Claude never learns from feedback | ✅ Claude evolves and adapts to you |
| 🚫 Single user only | ✅ Team collaboration with shared memory |
| 🚫 High RAM/VRAM requirements | ✅ Runs on **<2GB RAM** with efficient models |

**All processing happens locally on your machine.** Your data never leaves your computer.

---

## ✨ What's New in v0.5.0

- 🐝 **Model Swarm** - PSO-based collaborative inference with multiple small models
- 🔮 **Proactive Intelligence** - Anticipatory suggestions based on context and habits
- 🚀 **12+ New Models** - Phi-4-mini, SmolLM2, Qwen3-4B, TinyLlama, BitNet 2B
- ⚡ **Ultra-Efficient** - Run on <2GB RAM with TinyLlama, Qwen3-0.6B
- 🎯 **Smart Routing** - Mixture-of-Experts automatically picks best model per task
- 🔄 **Speculative Decoding** - 2.5x speedup with draft+verify pairs
- 📊 **Hardware Profiles** - Auto-configure based on your available resources

### Previously Added (v0.4.0)
- 🖼️ **Vision Module** - CLIP/BLIP image understanding, VQA, OCR
- 🎤 **Voice Module** - Whisper transcription, speaker diarization, TTS
- 📦 **Docker Support** - One-command deployment with GPU support
- 👥 **Team Collaboration** - Shared memory pools, multi-user sessions

---

## 🐝 Model Swarm: Collaborative Multi-Model Inference

The **Model Swarm** system enables multiple small models to work together, achieving better results than any single model:

### Swarm Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| **PSO Collaborative** | Particle Swarm Optimization guides model selection | Complex tasks |
| **Parallel Vote** | Run 3+ models, vote on best response | Quality-critical |
| **Mixture of Experts** | Route to specialist per task type | General use |
| **Speculative Ensemble** | Fast model drafts, strong model verifies | Speed + quality |
| **Fastest First** | Start fast, escalate if confidence low | Low latency |
| **Confidence Fusion** | Weighted combination of outputs | High reliability |

---

## 🏗️ Architecture & Privacy

**Farnsworth runs 100% locally on your machine.**

- **No Server Costs:** You do not need to pay for hosting.
- **Your Data:** All memories and files stay on your computer.
- **How it connects:** The [Claude Desktop App](https://claude.ai/download) spawns Farnsworth as a background process using the Model Context Protocol (MCP).



---

### Supported Models (Jan 2025)

| Model | Params | RAM | Strengths |
|-------|--------|-----|-----------|
| **Phi-4-mini-reasoning** | 3.8B | 6GB | Rivals o1-mini in math/reasoning |
| **Phi-4-mini** | 3.8B | 6GB | GPT-3.5 class, 128K context |
| **DeepSeek-R1-1.5B** | 1.5B | 4GB | o1-style reasoning, MIT license |
| **Qwen3-4B** | 4B | 5GB | MMLU-Pro 74%, multilingual |
| **SmolLM2-1.7B** | 1.7B | 3GB | Best quality at size |
| **Qwen3-0.6B** | 0.6B | 2GB | Ultra-light, 100+ languages |
| **TinyLlama-1.1B** | 1.1B | 2GB | Fastest, edge devices |
| **BitNet-2B** | 2B | 1GB | Native 1-bit, 5-7x CPU speedup |
| **Gemma-3n-E2B** | 2B eff | 4GB | Multimodal (text/image/audio) |
| **Phi-4-multimodal** | 5.6B | 8GB | Vision + speech + reasoning |

### Hardware Profiles

Farnsworth auto-configures based on your hardware:

```yaml
minimal:     # <4GB RAM: TinyLlama, Qwen3-0.6B
cpu_only:    # 8GB+ RAM, no GPU: BitNet, SmolLM2
low_vram:    # 2-4GB VRAM: DeepSeek-R1, Qwen3-0.6B
medium_vram: # 4-8GB VRAM: Phi-4-mini, Qwen3-4B
high_vram:   # 8GB+ VRAM: Full swarm with verification
```

---

## ⚡ Quick Start

### 📦 Option 1: One-Line Install (Recommended)

Farnsworth is available on PyPI. This is the easiest way to get started.

```bash
pip install farnsworth-ai
```

**Running the Server:**
```bash
# Start the MCP server
farnsworth-server

# Or customize configuration
farnsworth-server --debug --port 8000
```

### 🐳 Option 2: Docker

```bash
git clone https://github.com/timowhite88/Farnsworth.git
cd Farnsworth
docker-compose -f docker/docker-compose.yml up -d
```

### 🛠️ Option 3: Source (For Developers)

```bash
git clone https://github.com/timowhite88/Farnsworth.git
cd Farnsworth
pip install -r requirements.txt
```

### 🔌 Configure Claude Code

Add to your Claude Code MCP settings (usually found in `claude_desktop_config.json`):

**For PyPI Install:**
```json
{
  "mcpServers": {
    "farnsworth": {
      "command": "farnsworth-server",
      "args": [],
      "env": {
        "FARNSWORTH_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### 📖 [Full Installation Guide →](docs/USER_GUIDE.md#installation)

---

## 🌟 Key Features

### 🧠 Advanced Memory System

Claude finally remembers! Multi-tier hierarchical memory:

| Memory Type | Description |
|-------------|-------------|
| **Working Memory** | Current conversation context |
| **Episodic Memory** | Timeline of interactions, "on this day" recall |
| **Semantic Layers** | 5-level abstraction hierarchy |
| **Knowledge Graph** | Entities, relationships, temporal edges |
| **Archival Memory** | Permanent vector-indexed storage |
| **Memory Dreaming** | Background consolidation during idle time |

### 🤖 Agent Swarm (11 Specialists)

Claude can delegate tasks to AI agents:

| Core Agents | Description |
|-------------|-------------|
| **Code Agent** | Programming, debugging, code review |
| **Reasoning Agent** | Logic, math, step-by-step analysis |
| **Research Agent** | Information gathering, summarization |
| **Creative Agent** | Writing, brainstorming, ideation |

| Advanced Agents (v0.3+) | Description |
|-------------------------|-------------|
| **Planner Agent** | Task decomposition, dependency tracking |
| **Critic Agent** | Quality scoring, iterative refinement |
| **Web Agent** | Intelligent browsing, form filling |
| **FileSystem Agent** | Project understanding, smart search |

| Collaboration (v0.3+) | Description |
|-----------------------|-------------|
| **Agent Debates** | Multi-perspective synthesis |
| **Specialization Learning** | Skill development, task routing |
| **Hierarchical Teams** | Manager coordination, load balancing |

### 🖼️ Vision Understanding (v0.4+)

See and understand images:

- **CLIP Integration** - Zero-shot classification, image embeddings
- **BLIP Integration** - Captioning, visual question answering
- **OCR** - Extract text from images (EasyOCR)
- **Scene Graphs** - Extract objects and relationships
- **Image Similarity** - Compare and search images

### 🎤 Voice Interaction (v0.4+)

Hear and speak:

- **Whisper Transcription** - Real-time and batch processing
- **Speaker Diarization** - Identify different speakers
- **Text-to-Speech** - Multiple voice options
- **Voice Commands** - Natural language control
- **Continuous Listening** - Hands-free mode

### 👥 Team Collaboration (v0.4+)

Work together with shared AI:

- **Shared Memory Pools** - Team knowledge bases
- **Multi-User Support** - Individual profiles and preferences
- **Permission System** - Role-based access control
- **Collaborative Sessions** - Real-time multi-user interaction
- **Audit Logging** - Compliance-ready access trails

### 📈 Self-Evolution

Farnsworth learns from your feedback and improves automatically:

- **Fitness Tracking** - Monitors task success, efficiency, satisfaction
- **Genetic Optimization** - Evolves better configurations over time
- **User Avatar** - Builds a model of your preferences
- **LoRA Evolution** - Adapts model weights to your usage

### 🔍 Smart Retrieval (RAG 2.0)

Self-refining retrieval that gets better at finding relevant information:

- **Hybrid Search** - Semantic + BM25 keyword search
- **Query Understanding** - Intent classification, expansion
- **Multi-hop Retrieval** - Complex question answering
- **Context Compression** - Token-efficient memory injection
- **Source Attribution** - Confidence scoring

---

## 🛠️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Claude Code                             │
│              (Your AI Programming Partner)                   │
└─────────────────────────────────────────────────────────────┘
                              │ MCP Protocol
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Farnsworth MCP Server                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ Memory   │ │ Agent    │ │Evolution │ │Multimodal│       │
│  │ Tools    │ │ Tools    │ │ Tools    │ │ Tools    │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
└─────────────────────────────────────────────────────────────┘
          │                │                │
          ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Memory     │  │    Agent     │  │  Multimodal  │
│   System     │  │    Swarm     │  │   Engine     │
│              │  │              │  │              │
│ • Episodic   │  │ • Planner    │  │ • Vision     │
│ • Semantic   │  │ • Critic     │  │   (CLIP/BLIP)│
│ • Knowledge  │  │ • Web        │  │ • Voice      │
│   Graph v2   │  │ • FileSystem │  │   (Whisper)  │
│ • Archival   │  │ • Debates    │  │ • OCR        │
│ • Sharing    │  │ • Teams      │  │ • TTS        │
└──────────────┘  └──────────────┘  └──────────────┘
          │                │                │
          ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Evolution   │  │Collaboration │  │   Storage    │
│   Engine     │  │   System     │  │   Backends   │
│              │  │              │  │              │
│ • Genetic    │  │ • Multi-User │  │ • FAISS      │
│   Optimizer  │  │ • Shared     │  │ • ChromaDB   │
│ • Fitness    │  │   Memory     │  │ • Redis      │
│   Tracker    │  │ • Sessions   │  │ • SQLite     │
│ • LoRA       │  │ • Permissions│  │              │
└──────────────┘  └──────────────┘  └──────────────┘
          │                │                │
          └────────────────┴────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Model Swarm (v0.5+)                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              PSO Collaborative Engine                │   │
│  │   • Particle positions = model configs              │   │
│  │   • Velocity = adaptation direction                 │   │
│  │   • Global/personal best tracking                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ Phi-4    │ │DeepSeek  │ │ Qwen3    │ │ SmolLM2  │       │
│  │ mini     │ │ R1-1.5B  │ │ 0.6B/4B  │ │ 1.7B     │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │TinyLlama │ │ BitNet   │ │ Gemma    │ │ Cascade  │       │
│  │ 1.1B     │ │ 2B(1-bit)│ │ 3n-E2B   │ │ (hybrid) │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Tools Available to Claude

Once connected, Claude has access to these tools:

| Tool | Description |
|------|-------------|
| `farnsworth_remember(content, tags)` | Store information in long-term memory |
| `farnsworth_recall(query, limit)` | Search and retrieve relevant memories |
| `farnsworth_delegate(task, agent_type)` | Delegate to specialist agent |
| `farnsworth_evolve(feedback)` | Provide feedback for system improvement |
| `farnsworth_status()` | Get system health and statistics |
| `farnsworth_vision(image, task)` | Analyze images (caption, VQA, OCR) |
| `farnsworth_voice(audio, task)` | Process audio (transcribe, diarize) |
| `farnsworth_collaborate(action, ...)` | Team collaboration operations |
| `farnsworth_swarm(prompt, strategy)` | **NEW:** Multi-model collaborative inference |

---

## 📦 Docker Deployment

Multiple deployment profiles available:

```bash
# Basic deployment
docker-compose -f docker/docker-compose.yml up -d

# With GPU support
docker-compose -f docker/docker-compose.yml --profile gpu up -d

# With Ollama + ChromaDB
docker-compose -f docker/docker-compose.yml --profile ollama --profile chromadb up -d

# Development mode (hot reload + debugger)
docker-compose -f docker/docker-compose.yml --profile dev up -d
```

See [docker/docker-compose.yml](docker/docker-compose.yml) for all options.

---

## 📊 Dashboard

Farnsworth includes a Streamlit dashboard for visualization:

```bash
python main.py --ui
# Or with Docker:
docker-compose -f docker/docker-compose.yml --profile ui-only up -d
```

<details>
<summary>📸 Dashboard Features</summary>

- **Memory Browser** - Search and explore all stored memories
- **Episodic Timeline** - Visual history of interactions
- **Knowledge Graph** - 3D entity relationships
- **Agent Monitor** - Active agents and task history
- **Evolution Dashboard** - Fitness metrics and improvement trends
- **Team Collaboration** - Shared pools and active sessions
- **Model Swarm Monitor** - PSO state, model performance, strategy stats

</details>

---

## 🚀 Roadmap

See [ROADMAP.md](ROADMAP.md) for detailed plans.

### Completed ✅
- v0.1.0 - Core memory, agents, evolution
- v0.2.0 - Enhanced memory (episodic, semantic, sharing)
- v0.3.0 - Advanced agents (planner, critic, web, filesystem, debates, teams)
- v0.4.0 - Multimodal (vision, voice) + collaboration + Docker
- v0.5.0 - **Model Swarm** + 12 new models + hardware profiles

### Coming Next
- 🎬 Video understanding and summarization
- 🔐 Encryption at rest (AES-256)
- ☁️ Cloud deployment templates (AWS, Azure, GCP)
- 📊 Performance optimization (<100ms recall)

---

## 💡 Why "Farnsworth"?

Named after Professor Hubert J. Farnsworth from *Futurama* - a brilliant inventor who created countless gadgets and whose catchphrase "Good news, everyone!" perfectly captures what we hope you'll feel when using this tool with Claude.

---

## 📋 Requirements

| Minimum | Recommended | With Full Swarm |
|---------|-------------|-----------------|
| Python 3.10+ | Python 3.11+ | Python 3.11+ |
| 4GB RAM | 8GB RAM | 16GB RAM |
| 2-core CPU | 4-core CPU | 8-core CPU |
| 5GB storage | 20GB storage | 50GB storage |
| - | 4GB VRAM | 8GB+ VRAM |

**Supported Platforms:** Windows 10+, macOS 11+, Linux

**Optional Dependencies:**
- `ollama` - Local LLM inference (recommended)
- `llama-cpp-python` - Direct GGUF inference
- `torch` - GPU acceleration
- `transformers` - Vision/Voice models
- `playwright` - Web browsing agent
- `whisper` - Voice transcription

---

## 📄 License

**Farnsworth is dual-licensed:**

| Use Case | License |
|----------|---------|
| Personal / Educational / Non-commercial | **FREE** |
| Commercial (revenue > $1M or enterprise) | **Commercial License Required** |

See [LICENSE](LICENSE) for details. For commercial licensing, contact via GitHub.

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Priority Areas:**
- Video understanding module
- Cloud deployment templates
- Performance benchmarks
- Additional model integrations
- Documentation improvements

---

## 📚 Documentation

- 📖 [User Guide](docs/USER_GUIDE.md) - Complete usage documentation
- 🗺️ [Roadmap](ROADMAP.md) - Future plans and features
- 🤝 [Contributing](CONTRIBUTING.md) - How to contribute
- 📜 [License](LICENSE) - License terms
- 🐳 [Docker Guide](docker/) - Container deployment
- 🐝 [Model Configs](configs/models.yaml) - Supported models and swarm configs

---

## 🔗 Research References

Model Swarm implementation inspired by:
- [Model Swarms: Collaborative Search via Swarm Intelligence](https://arxiv.org/abs/2410.11163)
- [Harnessing Multiple LLMs: Survey on LLM Ensemble](https://arxiv.org/abs/2502.18036)
- [Small Language Models - MIT Tech Review](https://www.technologyreview.com/2025/01/03/1108800/small-language-models-ai-breakthrough-technologies-2025/)

---

## ⭐ Star History

If Farnsworth helps you, consider giving it a star! ⭐

---

<div align="center">

**Built with ❤️ for the Claude community**

*"Good news, everyone!"* - Professor Farnsworth

[Report Bug](https://github.com/timowhite88/Farnsworth/issues) • [Request Feature](https://github.com/timowhite88/Farnsworth/issues) • [Get Commercial License](https://github.com/timowhite88)

</div>
