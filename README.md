# Farnsworth 🧠

> *"Good news, everyone! I've invented a device that makes you smarter just by standing near it!"*

**Farnsworth** is a Neuromorphic Cognitive Architecture designed to be the ultimate AI companion for developers. It goes beyond simple "chat" by integrating a persistent evolving memory, a self-organizing agent swarm, and a specialized financial intelligence suite.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-green.svg)
![Status](https://img.shields.io/badge/status-Active-success.svg)
![Version](https://img.shields.io/badge/version-1.9-purple.svg)

## 🌟 Key Features

### 🧠 Core Cognition
*   **Persistent Unified Memory**: Remembers everything via `Archival Context` and `Knowledge Graphs`.
*   **Swarm Intelligence**: Dynamically spawns specialist agents (Coder, Critic, Researcher).
*   **Agent Debates**: Models argue pros/cons of a solution before presenting the best one.
*   **Evolutionary Feedback**: The system improves its own parameters based on your feedback (Genetic Optimization).

### ⚡ Hybrid Inference Engine
*   **Local First**: Runs ultra-fast local models (BitNet, Ollama) for simple tasks.
*   **Cloud Burst**: Automatically escalates to heavy models (Claude, GPT-4) for complex reasoning.
*   **Speculative Decoding**: Uses small models to "draft" answers for larger models to verify, doubling speed.

### 💰 DeGen Mob (Solana Suite)
*   **Rug Detection**: Analyzes smart contracts for malicious patterns.
*   **Mempool Sniping**: Detects high-value transactions before they land.
*   **Whale Tracking**: Follows "smart money" wallets.
*   **Automated Trading**: Executes via Jupiter/Raydium (Paper Trading by default).

### 🛠️ Developer Tools
*   **MCP Server**: Full integration with **Claude Code** (Cursor, VS Code).
*   **Streamlit Dashboard**: realtime visualization of memory, swarm status, and evolution.
*   **OS Bridge**: Can read/write files and execute terminal commands safely.

---

## 🚀 Getting Started

### Prerequisites
*   Python 3.10+
*   Docker (Optional, recommended)
*   NVIDIA GPU (Optional, for local inference)
*   [Ollama](https://ollama.ai/) (For local LLM support)

### Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/timowhite88/Farnsworth.git
    cd Farnsworth
    ```

2.  **Run Setup Wizard**
    ```bash
    python main.py --setup
    ```
    This will guide you through API key configuration, model selection, and memory initialization.

3.  **Start the System**
    ```bash
    python main.py
    ```

---

## 🐳 Docker Deployment

For a fully isolated environment, use Docker.

1.  **Build Container**
    ```bash
    docker-compose up --build
    ```

2.  **Access Dashboard**
    Open `http://localhost:8501` to view the Farnsworth Brain.

---

## 📖 Documentation

*   [**Architecture Diagrams**](docs/ARCHITECTURE_DIAGRAMS.md): Visual maps of the neural pipelines.
*   [**Feature Map**](docs/COMPLETE_FEATURE_MAP.md): Detailed breakdown of every subsystem.
*   [**Memory System**](farnsworth/memory/README.md): How short/long-term memory works.
*   [**Agent Swarm**](farnsworth/agents/README.md): Creating custom agents and debate protocols.

---

## 🎮 Usage Examples

### 1. Hybrid Delegation (Chat)
> **User**: "Write a python script to scrape SOL prices."
>
> **Farnsworth**: *Detects 'coding' task -> Spawns `CoderAgent` -> Checks local Qwen-2.5-Coder model -> Generates code.*

### 2. Deep Reasoning (Swarm Debate)
> **User**: "Should I use Postgres or Mongo for this project?"
>
> **Farnsworth**: *Spawns `Proponent(SQL)` and `Proponent(NoSQL)` -> Initiates Debate -> Synthesizer Agent merges points -> Returns balanced recommendation.*

### 3. Financial Intelligence
> **User**: "/scan CA:8sHj..."
>
> **Farnsworth**: *Activates `RugDetector` -> Scans metadata -> checks top holders -> Returns Risk Score: HIGH (Mint authority enabled).*

---

## 🤝 Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for guidelines on how to add new cognitive modules.

---

*"I don't simply process information... I understand it. Usually better than you do."* — Farnsworth
