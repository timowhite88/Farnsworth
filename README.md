# 🧠 Farnsworth: Your Claude Companion AI

<div align="center">

**Give Claude superpowers: persistent memory, specialist agents, and self-evolution.**

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/timowhite88/Farnsworth)
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Dual%20(Free%20%2B%20Commercial)-purple.svg)](LICENSE)
[![Claude Code](https://img.shields.io/badge/Claude%20Code-MCP%20Integration-orange.svg)](https://claude.ai)

[**Documentation**](docs/USER_GUIDE.md) • [**Roadmap**](ROADMAP.md) • [**Contributing**](CONTRIBUTING.md)

</div>

---

## 🎯 What is Farnsworth?

Farnsworth is a **companion AI system** that integrates with [Claude Code](https://claude.ai) to give Claude capabilities it doesn't have on its own:

| Without Farnsworth | With Farnsworth |
|:------------------:|:---------------:|
| 🚫 Claude forgets everything between sessions | ✅ Claude remembers your preferences forever |
| 🚫 Claude is a single model | ✅ Claude can delegate to specialist agents |
| 🚫 Claude never learns from feedback | ✅ Claude evolves and adapts to you |
| 🚫 You can't see what Claude "knows" | ✅ Visual dashboard shows everything |

**All processing happens locally on your machine.** Your data never leaves your computer.

---

## ⚡ Quick Start

### 1. Install

```bash
git clone https://github.com/timowhite88/Farnsworth.git
cd Farnsworth
pip install -r requirements.txt
```

### 2. Download a Local LLM

```bash
# Install Ollama from https://ollama.ai, then:
ollama pull deepseek-r1:1.5b
```

### 3. Configure Claude Code

Add to your Claude Code MCP settings:

```json
{
  "mcpServers": {
    "farnsworth": {
      "command": "python",
      "args": ["-m", "farnsworth.mcp_server"],
      "cwd": "/path/to/Farnsworth"
    }
  }
}
```

### 4. Start Using!

```
You: "Remember that I prefer TypeScript over JavaScript"
Claude: ✓ I'll remember that preference.

[Next week, new session]

You: "What language should I use for this project?"
Claude: "Based on your preference for TypeScript..."
```

📖 **[Full Installation Guide →](docs/USER_GUIDE.md#installation)**

---

## 🌟 Key Features

### 🧠 Persistent Memory

Claude finally remembers! Farnsworth gives Claude a hierarchical memory system:

- **Working Memory** - Current conversation context
- **Archival Memory** - Permanent storage of facts and preferences
- **Knowledge Graph** - Entities and relationships
- **Memory Dreaming** - Background consolidation during idle time

### 🤖 Agent Swarm

Claude can delegate tasks to specialist AI agents:

| Agent | Specialty |
|-------|-----------|
| **Code Agent** | Programming, debugging, code review |
| **Reasoning Agent** | Logic, math, step-by-step analysis |
| **Research Agent** | Information gathering, summarization |
| **Creative Agent** | Writing, brainstorming, ideation |

### 📈 Self-Evolution

Farnsworth learns from your feedback and improves automatically:

- **Fitness Tracking** - Monitors task success, efficiency, satisfaction
- **Genetic Optimization** - Evolves better configurations over time
- **User Avatar** - Builds a model of your preferences

### 🔍 Smart Retrieval

Self-refining RAG that gets better at finding relevant information:

- Hybrid semantic + keyword search
- Genetic evolution of retrieval strategies
- Automatic query expansion

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
│  │ Memory   │ │ Agent    │ │Evolution │ │Resources │       │
│  │ Tools    │ │ Tools    │ │ Tools    │ │(streams) │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
└─────────────────────────────────────────────────────────────┘
          │                │                │
          ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Memory     │  │    Agent     │  │  Evolution   │
│   System     │  │    Swarm     │  │   Engine     │
│              │  │              │  │              │
│ • Virtual    │  │ • Code       │  │ • Genetic    │
│   Context    │  │ • Reasoning  │  │   Optimizer  │
│ • Archival   │  │ • Research   │  │ • Fitness    │
│ • Knowledge  │  │ • Creative   │  │   Tracker    │
│   Graph      │  │ • User       │  │ • Behavior   │
│ • Dreaming   │  │   Avatar     │  │   Mutation   │
└──────────────┘  └──────────────┘  └──────────────┘
          │                │                │
          └────────────────┴────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Local LLM Backends                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ Ollama   │ │llama.cpp │ │ BitNet   │ │ Cascade  │       │
│  │(default) │ │ (GGUF)   │ │ (1-bit)  │ │ (hybrid) │       │
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

---

## 📊 Dashboard

Farnsworth includes a Streamlit dashboard for visualization:

```bash
python main.py --ui
```

<details>
<summary>📸 Dashboard Screenshots</summary>

- **Memory Browser** - Search and explore all stored memories
- **Knowledge Graph** - Visual entity relationships
- **Agent Monitor** - Active agents and task history
- **Evolution Dashboard** - Fitness metrics and improvement trends

</details>

---

## 🚀 Roadmap

See [ROADMAP.md](ROADMAP.md) for planned features:

**Coming Soon:**
- 🖼️ Image understanding (CLIP/BLIP integration)
- 🎤 Voice interaction (Whisper real-time)
- 🌐 Web browsing agent
- 📦 Docker deployment
- 👥 Team collaboration features

---

## 💡 Why "Farnsworth"?

Named after Professor Hubert J. Farnsworth from *Futurama* - a brilliant inventor who created countless gadgets and whose catchphrase "Good news, everyone!" perfectly captures what we hope you'll feel when using this tool with Claude.

---

## 📋 Requirements

| Minimum | Recommended |
|---------|-------------|
| Python 3.10+ | Python 3.11+ |
| 8GB RAM | 16GB RAM |
| 4-core CPU | 8-core CPU |
| 10GB storage | 50GB storage |
| - | NVIDIA GPU (4GB+ VRAM) |

**Supported Platforms:** Windows 10+, macOS 11+, Linux

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

**Good First Issues:**
- Adding tests
- Documentation improvements
- Bug fixes

---

## 📚 Documentation

- 📖 [User Guide](docs/USER_GUIDE.md) - Complete usage documentation
- 🗺️ [Roadmap](ROADMAP.md) - Future plans and features
- 🤝 [Contributing](CONTRIBUTING.md) - How to contribute
- 📜 [License](LICENSE) - License terms

---

## ⭐ Star History

If Farnsworth helps you, consider giving it a star! ⭐

---

<div align="center">

**Built with ❤️ for the Claude community**

*"Good news, everyone!"* - Professor Farnsworth

[Report Bug](https://github.com/timowhite88/Farnsworth/issues) • [Request Feature](https://github.com/timowhite88/Farnsworth/issues) • [Get Commercial License](https://github.com/timowhite88)

</div>
