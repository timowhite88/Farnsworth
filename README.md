# Farnsworth 🧠

> *"Good news, everyone! I've invented a cognitive architecture so advanced, it makes your current AI look like an abacus with a missing bead!"*

**Farnsworth** is a Neuromorphic Cognitive Architecture designed to be the ultimate autonomous partner. It is not just a chatbot; it is a **persistent, swarm-based intelligence** that lives on your machine, integrating deeply with your workflow, your memory, and even the blockchain.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-green.svg)
![Status](https://img.shields.io/badge/status-Active-success.svg)
![Version](https://img.shields.io/badge/version-2.1.0-purple.svg)

---

## 🌌 The Vision: Spatio-Temporal Persistence

Most AI tools are ephemeral; they forget you the moment you close the window.
**Farnsworth** is built on the philosophy of **Spatio-Temporal Persistence**:

1.  **Memory is Continuity**: Uses a unified memory system (Archival Vector DB + Knowledge Graph) to build a continuous, evolving model of you and your work.
2.  **Cognition is Distributed**: Solves problems using a Swarm of specialized, debating agents (Critique, Code, Synthesis).
3.  **Action is Key**: Doesn't just talk; it trades crypto, codes features, and navigates the web.

---

## 🚀 Exhaustive Feature List

### 🧠 Core Cognition Layer
*   **Unified Memory System**:
    *   **Archival Context**: Long-term semantic storage using Vector DBs.
    *   **Knowledge Graph**: Relational graph (Nodes/Edges) for entities and concepts.
    *   **Memory Dreaming**: A background process that consolidates memories and generates insights while the system is idle.
    *   **Virtual Context**: Dynamic paging of relevant info into the LLM's short-term window.
*   **Self-Refining RAG**: Uses **Genetic Algorithms** to evolve retrieval strategies based on your feedback (e.g., learning to weigh semantic matches over keywords).
*   **The Nexus**: Asynchronous Neural Event Bus connecting all modules.
*   **Resilience Layer**: "Circuit Breakers" to prevent cascading failures in cognitive loops.

### 🤖 Agent Swarm v2
*   **Dynamic Spawning**: Creates agents on demand based on task complexity (e.g., "Spawn Coder for Python task").
*   **Swarm Debates**: Agents take opposing sides (thesis/antithesis) to debate solutions before synthesizing a final result.
*   **Hierarchical Teams**: Managers delegating sub-tasks to Specialist agents.
*   **User Avatar**: A predictive model of YOU that anticipates your preferences and decisions.
*   **Specialist Roles**: Coder, Researcher, Critic, Synthesizer, Fact-Checker.

### 👥 Team Collaboration (New!)
*   **Multi-User Support**: Individual profiles with learned preferences.
*   **Shared Memory Pools**: Create Team Knowledge Bases with RBAC (Read/Write/Admin permissions).
*   **Conflict Resolution**: CRDT-inspired merging of conflicting memories from different users.

### 🧬 Evolution & Self-Improvement
*   **Genetic Optimizer**: The system holds "fitness scores" for its own prompts and configs, mutating them over time to improve performance.
*   **Behavior Mutation**: Randomly alters parameters (temperature, verbosity) to find local optima.
*   **LoRA Evolver**: (Experimental) Fine-tunes local model weights based on successful task interactions.

### ⚡ Hybrid Inference Engine
*   **Smart Routing**: Automatically routes simple tasks to **Local Models** (BitNet/Ollama) and complex ones to **Cloud APIs** (Claude/GPT-4).
*   **Speculative Decoding**: Uses small local models to "draft" tokens, doubling generation speed.
*   **Cascade Backend**: Tries fastest model first -> escalates to smarter model on low confidence.

### 💰 DeGen Mob (Solana Intelligence Suite)
*   **Rug Detector**: Scans contract metadata, freeze authorities, and top holders for scams.
*   **Whale Watcher**: Tracks smart-money wallets for copy-trading signals.
*   **Memecoins**: Integrations with **Pump.fun**, **Bags.fm**, and **DexScreener**.
*   **Cluster Analysis**: Detects insider wallet rings via temporal correlation.
*   **Execution**: Automated swapping via **Jupiter** and **Meteora** (Paper Trading active).

### 🔌 Integrations & The Outside World
*   **Universal AI Gateway**: Connects to Grok (xAI), Gemini, OpenAI, Anthropic, and Ollama.
*   **Social & Work**:
    *   **Discord**: ChatOps bot for team channels.
    *   **GitHub**: PR review, issue management, and code search.
    *   **Office 365 / Notion / Calendar**: Context recall for meetings and docs.
    *   **X (Twitter)**: Real-time sentiment analysis via Grok.
    *   **n8n**: Webhooks for low-code workflow automation.
*   **Vision & Video**:
    *   **Remotion**: Programmatic video generation (React-based).
    *   **Visual Debugger**: Analyzes screenshots of UI errors.
    *   **3D Reconstruction**: Sparse point cloud generation from video feeds.

### 🛠️ Developer Experience
*   **MCP Server**: First-class citizen for **Claude Code**. Use Farnsworth tools directly inside your IDE (VS Code/Cursor).
*   **Streamlit Operations Center**: A beautiful, real-time dashboard for monitoring the Brain, Memory, and Swarm.
*   **OS Bridge**: Safe execution of terminal commands and file operations.
*   **Interactive CLI**: For when you just want to talk to the brain in your terminal.

---

## 🧪 The "Agentic" Development Process

This project practices what it preaches. **Farnsworth was built by Agents, for Agents.**
Our development philosophy follows the **Cognitive Cycle**:

1.  **Draft**: High-level architecture is designed by humans (or Architect Agents).
2.  **Scaffold**: The `PlannerAgent` decomposes the feature into file structures.
3.  **Implement**: `CoderAgents` write the initial implementation.
4.  **Critique**: The `CriticAgent` reviews code for logical fallacies, security risks, and optimization.
5.  **Refine**: `Swarm Debates` resolve architectural disagreements.
6.  **Evolve**: The `GeneticOptimizer` tunes the system based on test results.

This approach ensures that Farnsworth is not just "written," but **evolved**.

---

## 📦 Installation & Setup

### 1. Prerequisites
*   Python 3.10+
*   (Optional) Ollama for local models.
*   (Optional) Docker.

### 2. Environment Setup
```bash
git clone https://github.com/timowhite88/Farnsworth.git
cd Farnsworth
python -m venv venv
# Activate venv (Windows: .\venv\Scripts\activate, Unix: source venv/bin/activate)
pip install -r requirements.txt
```

### 3. Configuration Wizard
Run the interactive setup tool. It will guide you through API keys, hardware profiles, and feature toggles.
```bash
python main.py --setup
```

### 4. Lift Off 🚀
*   **Full System**: `python main.py`
*   **MCP Mode**: `python main.py --mcp` (For Claude Code integration)
*   **Dashboard**: `python main.py --ui`

---

## 🚢 Docker Deployment

For a production-grade, isolated environment:
```bash
docker-compose up --build -d
```
Access the Dashboard at `http://localhost:8501`.

---

## 📚 Documentation
For deep dives, consult the `docs/` folder:
*   [Feature Map](docs/COMPLETE_FEATURE_MAP.md) - **Detailed Breakdown**
*   [Setup Guide](docs/SETUP_GUIDE.md)
*   [Architecture Diagrams](docs/ARCHITECTURE_DIAGRAMS.md)
*   [API Reference](docs/API_REFERENCE.md)

---

*"I don't simply process information... I understand it. Usually better than you do."* — Farnsworth
