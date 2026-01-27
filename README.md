# 🧠 Farnsworth: Your Claude Companion AI
9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS
<div align="center">

**Give Claude superpowers: persistent memory, model swarms, multimodal understanding, and self-evolution.**

[![Version](https://img.shields.io/badge/version-2.8.0-blue.svg)](https://github.com/timowhite88/Farnsworth)
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Dual%20(Free%20%2B%20Commercial)-purple.svg)](LICENSE)
[![Claude Code](https://img.shields.io/badge/Claude%20Code-MCP%20Integration-orange.svg)](https://claude.ai)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](docker/)
[![Models](https://img.shields.io/badge/Models-12%2B%20Supported-green.svg)](configs/models.yaml)

[**Documentation**](docs/USER_GUIDE.md) • [**Roadmap**](ROADMAP.md) • [**Setup Wizard**](farnsworth/core/setup_wizard.py) • [**Isolated Mode**](README.md#isolated-mode)

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

**All processing happens locally on your machine.** Your data never leaves your computer.

### ⚔️ Why choose Farnsworth?

| Feature | 🧠 **Farnsworth** | 🤖 **Others (Marge, Ralph, Claudebot)** |
|:---|:---:|:---:|
| **Memory** | **Infinite & Planetary** | Session / Repo Only |
| **Logic** | **Quantum & Causal** | Linear Chain-of-Thought |
| **Tools** | **Solana / Stocks / Vision** | Basic IO |
| **Privacy** | **Local First** | Cloud Dependent |

[**👉 See the full Battle Chart vs. Marge, Ralph, and Claudebot**](COMPARED.md)

---

## ✨ What's New in v2.8.0 (The "Swarm Node" Release)

### P2P Network Node
- 🌐 **Spin Up as a Node** - Run `python main.py --node` to join the global Farnsworth network
- 🔗 **Peer Discovery** - Automatic mDNS/UDP discovery of nearby Farnsworth nodes
- 🌍 **Planetary Memory Sharing** - Contribute to and benefit from the Akashic Record
- 📡 **Task Auctions** - Distribute heavy tasks across the swarm for parallel processing
- 📊 **Live Dashboard** - `--dashboard` flag shows real-time peer and DKG stats

### Token Saving Mode (API Cost Optimization)
- 💰 **Daily Budget Tracking** - Set token limits, get warnings at 80%/90% thresholds
- 🗜️ **Context Compression** - Smart/extractive/truncate strategies to reduce input tokens
- 📦 **Response Caching** - LRU cache with TTL for common responses (skip redundant API calls)
- 🐝 **Swarm Offloading** - Route simple queries to local models, reserve API for complex tasks

### Productivity Suite
- 📝 **Quick Notes** - Fast note capture with tags (`note "Meeting notes #work"`)
- 📋 **Snippet Manager** - Code snippet storage with template variables
- 🍅 **Focus Timer** - Pomodoro-style timer with session tracking and stats
- 📊 **Daily Summary** - Auto-generated activity digests with LLM insights
- 🎭 **Context Profiles** - Switch between Work/Personal/Creative/Technical modes

### Previous Releases

<details>
<summary>v2.1.0 - v2.7.0 (Click to expand)</summary>

#### v2.1.0 - The Skill Swarm
- 🦝 **Grok X Search** - Real-time X (Twitter) search and deep thinking via xAI
- 🎬 **Remotion Video** - Programmatic React-based video generation and rendering
- ⚡ **Parallel AI** - High-reliability consensus via multi-model concurrent dispatch
- 🧪 **DeGen Mob** - Launch Sniping, Whale Watching, & Rug Detection (Solana)
- 🖥️ **DeGen Dashboard** - Ultra-premium glassmorphic HUD for real-time swarm visualization
- 🧠 **Cognitive Trading** - Integrated reasoning & learning for signal accuracy
- 💰 **Elite Solana Trading** - Jupiter Swaps, Meteora LP management, & Pump.fun execution
- 📈 **Financial Intelligence** - DexScreener, Polymarket, & Pump.fun/Bags.fm tracking
- 💹 **Market Sentiment** - Crypto Fear & Greed index and global market macro
- 📺 **YouTube Intelligence** - Transcript extraction and semantic video analysis
- 🧩 **Sequential Thinking** - Systematic "Chain-of-Thought" reasoning tool
- 🗄️ **Database Manager** - Secure, read-only SQL access to local/remote databases
- 🔌 **Discord Bridge** - Full "ChatOps" integration for remote commanding
- 📊 **Mermaid Diagrams** - Native architecture and flowchart visualization
- 🦾 **Agentic OS** - Deep system diagnostics and process management
- 🧙 **Granular Setup Wizard** - Step-by-step feature control (`python main.py --setup`)
- 🎥 **Video v2.1** - Advanced Spatio-Temporal Flow Analysis (Optical Flow)
- 🧠 **Synergy Engine** - Automated cross-domain learning (GitHub -> Memory -> Projects)

#### v2.7.0 - The "Cognitive Productivity" Suite
- 💤 **Dream Catcher (Sleep Learning)** - Farnsworth performs "offline memory consolidation" while idling, hallucinating questions it *should* have been asked to refine its own knowledge base.
- 🏙️ **The Holodeck** - A 3D WebGL visualization of your codebase topology (Buildings = Classes, Height = LoC, Color = Complexity).
- 🤫 **Cone of Silence (Focus Mode)** - System-level blocking of distraction sites (X/Reddit) during deep work sessions.
- 🪃 **Boomerang** - "Remind me of this if I don't hear back." Smart task resurfacing.
- 🗣️ **Mimic** - Lightweight local Text-to-Speech (TTS) engine.
- 📝 **Auto-Docs & Meeting Whisperer** - Real-time documentation scanning and transcript keyword spotting.

### Mega Update (v2.6.0) - The "Omni-Market" Update
- 🕷️ **Universal Scraper (Crawlee)** - Robust scraping for Social Media (X/Insta) and Live Platforms (Twitch/YouTube) with bot-evasion tactics.
- 📉 **TradFi Agent** - Real-time Stocks & Forex tracking (yfinance / AlphaVantage). Farnsworth is now a multi-asset financial terminal.
- 🎨 **Meme Quality Analyzer** - Vision-based AI that rates memes on "Originality" and "Cursed Energy" to predict viral potential.
- 🐇 **Bonding Curve Sniper** - Tracks Pump.fun curves to alert you moments before a token graduates to Raydium.
- ⚡ **Jito Bundle Execution** - Routes Solana trades directly to validators (Anti-MEV) to prevent sandwich attacks.
- 🧊 **3D Reconstruction** - Building spatial mental models from video (SfM)
- 🐈 **Quantum-Inspired Search (Schrödinger's Query)** - Superposition-based reasoning engine.
- 🌍 **Planetary Memory (Akashic Record)** - Privacy-preserving global knowledge sharing.

### The Spatio-Temporal Era (v2.0)
- 🎥 **Video v2.0** - Duo-Stream Analysis (Visual Saliency + Audio Narrative)
- 🌐 **P2P Swarm Fabric** - Decentralized agent discovery and Task Auctions (DTA)
- 🧠 **Decentralized Knowledge Graph (DKG)** - Federated fact-sharing across trust pools

### Cutting Edge (v1.6 - v1.9)
- 🎭 **Theory of Mind (v1.6)** - Predictive Coding simulation of user intent
- 👁️ **Visual Intelligence (v1.7)** - Visual Debugger & Diagram Understanding
- 📅 **Personal Assistant (v1.8)** - Meeting Prep & Learning Co-Pilot
- 🔗 **Connected Ecosystem (v1.9)** - Integrations with GitHub, Notion, O365, X, n8n
- 🧠 **Neuromorphic Core (v1.4)** - Sparse Distributed Memory & Hebbian Learning
- 🦾 **Agentic OS (v1.4)** - Deep system context awareness bridge
- ♾️ **Continual Learning (v1.5)** - Experience Replay & Elastic Consolidation
- 🔮 **Causal Reasoning (v1.5)** - Causal graphs, interventions, and counterfactuals

#### Previously Added
- 🖼️ **Multimodal** - Vision (CLIP/BLIP) & Voice (Whisper) support
- 📦 **Docker Support** - One-command deployment with GPU support
- 👥 **Team Collaboration** - Shared memory pools, multi-user sessions
- 🔍 **Advanced RAG** - Hybrid search with semantic layers

</details>

---

## 🛠️ Usage & Examples

### 📈 Financial Intelligence
Ask Farnsworth about any token or market:
- "Check the price and liquidity of $SOL on DexScreener."
- "What's the bonding curve progress for [MINT_ADDRESS] on pump.fun?"
- "Show me the trending tokens on bags.fm."
- "What are the current odds on Polymarket for the next SpaceX launch?"

### 🧪 DeGen Mob (Solana Power Tools)
Unleash the swarm on the Solana ecosystem:
- "Scan this mint for rug risks: [MINT_ADDRESS]"
- "Start watching this whale wallet for rotations: [WALLET_ADDRESS]"
- "Activate the launch sniper for AI-themed memecoins."
- "Run a sentiment swarm on 'AI Agents' vs 'DePIN' narratives."
- "Show me the trading dashboard."

### 💰 Elite Solana Trading
Farnsworth can now manage assets and execute trades (Burner wallet recommended):
- "Evaluate this signal: $WIF at 50k liquidity and 1M volume. Should I swap?"
- "Swap 0.1 SOL for $WIF on Jupiter."
- "Create a one-sided Meteora DLMM pool for my new token."
- "What happened with our last signal on [MINT]? Did it go well?"

### 🎬 Video & Diagrams

### 🧩 Systematic Reasoning
- "Explain quantum tunneling using the Sequential Thinking tool."

---

## 🌐 P2P Network Node

Turn your Farnsworth into a node in the global swarm:

```bash
# Basic node
python main.py --node

# Custom port with live dashboard
python main.py --node --port 9999 --dashboard

# Node without Planetary Memory sharing
python main.py --node --no-planetary
```

### What Happens When You Run a Node

| Capability | Description |
|------------|-------------|
| **Peer Discovery** | Automatically finds other Farnsworth nodes on your network via UDP broadcast |
| **Knowledge Sharing** | Syncs the Decentralized Knowledge Graph (DKG) with peers |
| **Planetary Memory** | Contributes anonymized skills to the global Akashic Record |
| **Task Auctions** | Can bid on or delegate heavy computation tasks |

### Node Dashboard

When using `--dashboard`, you see live stats:
```
📊 Peers: 3 | DKG: 127 nodes, 89 edges | Messages seen: 1,247
```

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

### 🤖 Install via Claude Code (Recommended)

**Just paste this to Claude:**
```
Clone and set up Farnsworth from https://github.com/timowhite88/Farnsworth -
it's a companion AI system with persistent memory, model swarms, and P2P networking.
After cloning, run the setup wizard and help me configure it.
```

Claude will:
1. Clone the repository
2. Install dependencies
3. Run the setup wizard (`python main.py --setup`)
4. Help you configure Claude Desktop's MCP settings

**Or give Claude a direct command:**
```
git clone https://github.com/timowhite88/Farnsworth.git && cd Farnsworth && pip install -r requirements.txt && python main.py --setup
```

---

### 📦 Option 1: One-Line Install (Recommended)

Farnsworth is available on PyPI. This is the easiest way to get started.

```bash
pip install farnsworth-ai
```

**Running the Server:**
```bash
# Start the MCP server
farnsworth-server

# Run the GRANULAR setup wizard
python main.py --setup
```

### 🛡️ Isolated Mode
For maximum privacy, Farnsworth can run in complete isolation:
- Set `FARNSWORTH_ISOLATED=true` in your `.env`
- All P2P discovery and network broadcasting is HARD-DISABLED.
- Perfect for offline usage or highly sensitive environments.

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

### 🏥 System Health Check
After installation, run the self-diagnostic tool to ensure all advanced features (like Focus Mode and Scrapers) have the necessary permissions and dependencies:

```bash
python scripts/verify_env.py
```
This will check for:
- **Admin/Root Privileges** (Required for 'Focus Mode' hosts file blocking)
- **Playwright** (Required for 'Universal Scraper')
- **TTS Engine** (Required for 'Mimic' voice)

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

### 💰 Token Saving Mode (NEW v2.8.0)

Reduce API costs by up to 70%:

- **Daily Budget** - Set token limits with warnings at 80%/90%
- **Response Cache** - Skip API calls for repeated queries
- **Context Compression** - Smart summarization of long contexts
- **Swarm Offloading** - Route simple tasks to local models

### 🎭 Context Profiles (NEW v2.8.0)

Switch between different working modes:

| Profile | Personality | Use Case |
|---------|-------------|----------|
| 💼 Work | Formal, detailed | Professional tasks |
| 🏠 Personal | Casual, normal | Personal projects |
| 🎨 Creative | Casual, high-temp | Brainstorming, writing |
| 🔧 Technical | Technical, precise | Debugging, architecture |

```bash
# CLI commands
farnsworth> profiles        # List all profiles
farnsworth> switch work     # Switch to Work profile
farnsworth> profile         # Show current profile
```

### 🍅 Productivity Tools (NEW v2.8.0)

Built-in productivity features:

| Tool | Description |
|------|-------------|
| **Quick Notes** | Fast capture with tags: `note "idea #project"` |
| **Snippet Manager** | Store and reuse code snippets with templates |
| **Focus Timer** | Pomodoro timer with session tracking |
| **Daily Summary** | Auto-generated activity digests |

### 🔍 Smart Retrieval (RAG 2.0)

Self-refining retrieval that gets better at finding relevant information:

- **Hybrid Search** - Semantic + BM25 keyword search
- **Query Understanding** - Intent classification, expansion
- **Multi-hop Retrieval** - Complex question answering
- **Context Compression** - Token-efficient memory injection
- **Source Attribution** - Confidence scoring

### 📊 Project Tracking (v1.2+)

Turn conversations into concrete progress:

- **Auto-Detection** - Identifies new projects from natural conversation
- **Task Management** - Tracks dependencies, priorities, and status
- **Milestone Tracking** - Monitors progress towards key goals
- **Cross-Project Knowledge** - Transfers learnings between related projects
- **Smart Linking** - Semantically links related initiatives


---

## 🛠️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Claude Code                             │
│              (The User Interface)                           │
└─────────────────────────────────────────────────────────────┘
                              │ FCP Context Injection
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Farnsworth Nexus                          │
│                (The Neural Event Bus)                       │
│                                                             │
│  ┌──────────┐   Signals    ┌──────────┐   Signals           │
│  │ Agents   │ ◄──────────► │ FCP      │ ◄──────────► User   │
│  │          │              │ Engine   │              State  │
│  └──────────┘              └──────────┘              Files  │
│       ▲                          │                          │
│       │                          ▼                          │
│       │             ┌─────────────────────────┐             │
│       │             │ Vision | Focus | Horizon│             │
│       │             └─────────────────────────┘             │
│       │                                                     │
│       ▼                                                     │
│  ┌──────────┐              ┌──────────┐                     │
│  │ Resilience│             │ Omni-    │                     │
│  │ Layer     │             │ Channel  │ ◄────► Discord/Slack│
│  └──────────┘              └──────────┘                     │
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
| `farnsworth_swarm(prompt, strategy)` | Multi-model collaborative inference |
| `farnsworth_project_create(name, desc)` | Create and track projects |
| `farnsworth_project_status(id)` | Get project progress and tasks |
| `farnsworth_project_detect(text)` | Auto-detect projects from conversations |
| `farnsworth_token_status()` | **NEW:** Get token budget and cache stats |
| `farnsworth_quick_note(content, tags)` | **NEW:** Add a quick note |
| `farnsworth_focus_start(task)` | **NEW:** Start focus timer session |
| `farnsworth_daily_summary()` | **NEW:** Generate daily activity summary |
| `farnsworth_switch_profile(id)` | **NEW:** Switch context profile |

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

### Docker Ports

| Port | Service |
|------|---------|
| 8000 | MCP Server |
| 8501 | Streamlit UI |
| 8888/udp | P2P Discovery |
| 9999 | P2P Swarm Fabric |

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
- v0.5.0 - Model Swarm + 12 new models + hardware profiles
- v1.0.0 - **Production Release** - Performance, reliability, scaling
- v1.1.0 - **Conversation Export** - multiple formats
- v1.2.0 - **Project Tracking** - Tasks, milestones, knowledge transfer


### Version 2.0.0 - Spatio-Temporal era 🚀
- **Video Duo-Stream**: Visual Saliency + Audio-Visual Narrative
- **3D Scene Reconstruction**: SfM-based sparse point cloud generation
- **P2P Swarm**: mDNS discovery & Distributed Task Auctions
- **DKG**: Decentralized Knowledge Graph with CRDT resolution
- **Emotion-to-Action**: Directly mapping affective states into system priorities
- **Biological Support**: Standardized API for neuro-integration

### Version 1.9.0 - Connected Ecosystem 🔗
- **External Framework**: GitHub, Notion, Calendar, Office365, X (Twitter)
- **Universal AI Gateway**: Hybrid route to Grok/Gemini/OpenAI
- **n8n Bridge**: Infinite extensibility via workflows
- **IDE Integrations**: VS Code LSP & Cursor Shadow Workspace

### Coming Next
- 🪐 Planetary Memory (Global shared vector cache)
- 🪐 Biological Neural Interfacing (SDK)




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

**Built with ❤️ for the Community**

*"Good news, everyone!"* - Professor Farnsworth

[Report Bug](https://github.com/timowhite88/Farnsworth/issues) • [Request Feature](https://github.com/timowhite88/Farnsworth/issues) • [Get Commercial License](https://github.com/timowhite88)

</div>
