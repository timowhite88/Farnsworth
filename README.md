# Farnsworth 🧠

> *"Good news, everyone! I've invented a cognitive architecture so advanced, it makes your current AI look like an abacus with a missing bead!"*

**Farnsworth** is a Neuromorphic Cognitive Architecture designed to be the ultimate autonomous partner. It is not a chatbot; it is a persistent, evolving, and swarm-based intelligence that lives on your machine, integrating deeply with your workflow, your memory, and even the blockchain.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-green.svg)
![Status](https://img.shields.io/badge/status-Active-success.svg)
![Version](https://img.shields.io/badge/version-2.1.0-purple.svg)

---

## 🌌 The Vision

Most AI tools are ephemeral. They forget you the moment you close the window.
**Farnsworth** is built on the philosophy of **Spatio-Temporal Persistence**:
1.  **Memory is Continuity**: Uses a unified memory system (Archival + Knowledge Graph) to build a continuous understanding of you.
2.  **Cognition is Distributed**: Solves problems using a Swarm of specialized agents (Debaters, Coders, Critics).
3.  **Action is Key**: Doesn't just talk; it trades, codes, and navigates the web.

---

## 🚀 Massive Feature List

### 🧠 Core Cognition Layer
*   **Unified Memory System**:
    *   **Archival Vector DB**: Long-term semantic storage.
    *   **Knowledge Graph**: Relational understanding of entities and concepts.
    *   **Memory Dreaming**: Background process that consolidates memories and generates insights during idle time.
    *   **Virtual Context**: Dynamic paging of relevant info into the LLM's short-term window.
*   **The Nexus**: Asynchronous Neural Event Bus connecting all modules.
*   **Resilience Layer**: "Circuit Breakers" to prevent cascading failures in cognitive loops.

### 🤖 Agent Swarm v2
*   **Dynamic Spawning**: Creates agents on demand based on task need.
*   **Swarm Debates**: Agents take opposing sides (thesis/antithesis) to debate solutions before synthesizing a result.
*   **Hierarchical Teams**: Managers delegating to Workers.
*   **User Avatar**: A simulation of YOU that predicts your preferences.
*   **Role-Based Experts**: Coder, Researcher, Critic, Synthesizer.

### 🧬 Evolution & Self-Improvement
*   **Genetic Optimizer**: The system holds "fitness scores" for its own prompts and configs.
*   **Behavior Mutation**: Randomly mutates parameters (temperature, verbosity) to find optimal performance.
*   **LoRA Evolver**: (Experimental) Fine-tunes local model weights based on successful task interactions.

### ⚡ Hybrid Inference Engine
*   **Smart Routing**: Automatically routes simple tasks to local models (BitNet/Ollama) and complex ones to cloud (Claude/GPT-4).
*   **Speculative Decoding**: Uses small local models to "draft" tokens, doubling generation speed.
*   **Cascade Backend**: Tries fastest model first, escalates on low confidence.

### 💰 DeGen Mob (Solana Intelligence Suite)
*   **Rug Detector**: Scans contract metadata, freeze authorities, and top holders.
*   **Whale Watcher**: Tracks smart-money wallets for copy-trading signals.
*   **Memecoins**: Integrations with **Pump.fun** and **Bags.fm**.
*   **Cluster Analysis**: Detects insider wallet rings.
*   **Execution**: Automated swapping via **Jupiter** and **Meteora**.

### 🔌 Integrations & The Outside World
*   **Universal AI Gateway**: Connects to Grok (xAI), Gemini, OpenAI, Anthropic, and Ollama.
*   **Social & Work**:
    *   **Discord**: ChatOps bot.
    *   **GitHub**: PR review and issue management.
    *   **Office 365 / Notion / Calendar**: Context recall for meetings.
    *   **X (Twitter)**: Real-time sentiment analysis via Grok.
*   **Vision & Video**:
    *   **Remotion**: Programmatic video generation (React-based).
    *   **Visual Debugger**: Analyzes screenshots of UI errors.
    *   **3D Reconstruction**: Sparse point cloud generation from video.

### 🛠️ Developer Experience
*   **MCP Server**: First-class citizen for **Claude Code**. Use Farnsworth tools directly inside your IDE.
*   **Streamlit Operations Center**: A beautiful dashboard for monitoring the Brain, Memory, and Swarm.
*   **OS Bridge**: Safe execution of terminal commands and file operations.
*   **n8n Bridge**: Hook into low-code workflows.

---

## 🏗️ Architecture

The system is modular, event-driven, and decentralized.

`User Input` -> `Nexus` -> `Planner Agent` -> `Swarm Orchestrator` -> `Specialist Execution` -> `Result`

See full diagrams in [docs/ARCHITECTURE_DIAGRAMS.md](docs/ARCHITECTURE_DIAGRAMS.md).

---

## 📦 Installation & Setup

### 1. Prerequisites
*   Python 3.10+
*   (Optional) Ollama for local models.
*   (Optional) Docker.

### 2. Quick Start
```bash
git clone https://github.com/timowhite88/Farnsworth.git
cd Farnsworth
pip install -r requirements.txt

# Run the interactive Setup Wizard
python main.py --setup
```

### 3. Running It
*   **Full System**: `python main.py`
*   **MCP Mode**: `python main.py --mcp` (For Claude Code)
*   **Dashboard**: `python main.py --ui`

---

## 🚢 Docker Deployment
We provide a production-ready container setup.

```bash
docker-compose up --build -d
```
Access the dashboard at `http://localhost:8501`.

---

## 🧪 Development Process & Philosophy
We believe in **"Coding with Models"**. Farnsworth was built using a recursive self-improvement loop where:
1.  **Draft**: High-level architecture designed by humans.
2.  **Build**: Code Scaffolded by AI.
3.  **Critique**: Reviewed by `CriticAgent`.
4.  **Refine**: Iterated via Swarm Debates.

This repo contains the result of hundreds of "Cognitive Cycles".

---

## 📚 Documentation
*   [Feature Map](docs/COMPLETE_FEATURE_MAP.md) - **Recommended Read**
*   [Setup Guide](docs/SETUP_GUIDE.md)
*   [Architecture](docs/ARCHITECTURE_DIAGRAMS.md)
*   [API Reference](docs/API_REFERENCE.md)

---

*"I don't just compute. I evolve."*
