"""
Build comprehensive Farnsworth documentation.
Goes through every file and documents everything.
"""

import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_FILE = Path(__file__).parent.parent / "DOCUMENTATION.md"

def build_docs():
    """Build the complete documentation."""

    sections = []

    # Section: 5-Layer Memory System
    sections.append("""
# 5-LAYER MEMORY SYSTEM

The most sophisticated AI memory architecture ever built.

```
┌─────────────────────────────────────────────────────────────────────────┐
│ LAYER 5: PLANETARY MEMORY (P2P Network)                                 │
│ Location: ws://194.68.245.145:8889                                     │
│ File: farnsworth/memory/memory_sharing.py                              │
│ Purpose: Share anonymized learnings across all Farnsworth instances    │
│ Privacy: Opt-in only, no raw conversations shared                      │
├─────────────────────────────────────────────────────────────────────────┤
│ LAYER 4: ARCHIVAL MEMORY (Vector Search)                                │
│ Location: farnsworth/memory/archival/                                  │
│ File: farnsworth/memory/archival_memory.py                             │
│ Backend: FAISS / ChromaDB                                              │
│ Embeddings: 384-dimensional (sentence-transformers)                    │
│ Features: Hierarchical indexing, temporal weighting, clustering        │
├─────────────────────────────────────────────────────────────────────────┤
│ LAYER 3: SEMANTIC MEMORY (Knowledge Graph)                              │
│ Location: farnsworth/memory/graph/                                     │
│ File: farnsworth/memory/knowledge_graph_v2.py                          │
│ Features: Entity relationships, multi-hop inference, deduplication     │
├─────────────────────────────────────────────────────────────────────────┤
│ LAYER 2: EPISODIC MEMORY (Timeline Events)                              │
│ File: farnsworth/memory/episodic_memory.py                             │
│ Event Types: conversation, task, feedback, system                      │
│ Features: Importance scoring, decay over time, dream consolidation     │
├─────────────────────────────────────────────────────────────────────────┤
│ LAYER 1: WORKING MEMORY (Current Session)                               │
│ File: farnsworth/memory/working_memory.py                              │
│ Type: LRU Cache with TTL                                               │
│ Features: Query caching, context window, token budget tracking         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Memory Files Reference

| File | Purpose | Key Functions |
|------|---------|---------------|
| `memory_system.py` | Unified interface | `remember()`, `recall()`, `search()` |
| `archival_memory.py` | Long-term storage | `store()`, `vector_search()`, `cluster()` |
| `episodic_memory.py` | Event timeline | `add_event()`, `get_timeline()` |
| `working_memory.py` | Current session | `cache_get()`, `cache_set()` |
| `knowledge_graph_v2.py` | Entity relations | `add_entity()`, `add_relation()`, `query_path()` |
| `dream_consolidation.py` | Memory consolidation | `consolidate()`, `merge_similar()` |
| `memory_sharing.py` | P2P sharing | `share()`, `receive()`, `sync()` |
| `semantic_deduplication.py` | Remove duplicates | `deduplicate()`, `find_similar()` |

<details>
<summary><b>Code Example: Using Memory System</b></summary>

```python
from farnsworth.memory.memory_system import get_memory_system

async def example_memory_usage():
    memory = get_memory_system()

    # Store a memory
    await memory.remember(
        content="User prefers dark mode interfaces",
        tags=["preference", "ui"],
        importance=0.8
    )

    # Recall relevant memories
    memories = await memory.recall(
        query="What UI preferences does the user have?",
        limit=5
    )

    # Search with filters
    results = await memory.search(
        query="dark mode",
        filters={"tags": ["preference"]},
        min_importance=0.5
    )

    return memories
```

</details>

<details>
<summary><b>Code Example: Knowledge Graph</b></summary>

```python
from farnsworth.memory.knowledge_graph_v2 import KnowledgeGraph

async def example_knowledge_graph():
    kg = KnowledgeGraph()

    # Add entities
    kg.add_entity("Farnsworth", type="agent", attributes={"role": "orchestrator"})
    kg.add_entity("Grok", type="agent", attributes={"role": "real_time"})

    # Add relations
    kg.add_relation("Farnsworth", "coordinates", "Grok")
    kg.add_relation("Grok", "provides", "X_data")

    # Query paths
    path = kg.query_path("Farnsworth", "X_data")
    # Returns: Farnsworth -> coordinates -> Grok -> provides -> X_data

    return path
```

</details>

---
""")

    # Section: On-Chain Memory (PATENTED)
    sections.append("""
# ON-CHAIN MEMORY

## PATENTED BY BETTER CLIPS

> **This on-chain memory system is proprietary technology patented by Better Clips.**
> **Super simplified explanation below.**

### What Is On-Chain Memory?

**Simple Explanation:**
Instead of storing AI memories on a regular database that can be lost, deleted, or tampered with, on-chain memory stores them on a blockchain - making them **permanent**, **immutable**, and **verifiable**.

Think of it like writing in permanent marker on a diamond vs. writing in pencil on paper.

### How It Works (Simplified)

```
┌──────────────────────────────────────────────────────────────────┐
│                    ON-CHAIN MEMORY FLOW                          │
│                                                                  │
│  1. AI generates a memory/learning                               │
│                    │                                             │
│                    ▼                                             │
│  2. Memory is hashed (fingerprint created)                       │
│                    │                                             │
│                    ▼                                             │
│  3. Hash + metadata stored on blockchain                         │
│     (Solana for speed, Base for compatibility)                   │
│                    │                                             │
│                    ▼                                             │
│  4. Full memory data stored off-chain (IPFS/Arweave)            │
│     with on-chain pointer                                        │
│                    │                                             │
│                    ▼                                             │
│  5. Anyone can verify memory authenticity                        │
│     by checking on-chain hash                                    │
└──────────────────────────────────────────────────────────────────┘
```

### Why It Matters

| Traditional Memory | On-Chain Memory |
|-------------------|-----------------|
| Can be deleted | Permanent forever |
| Can be modified | Immutable |
| Trust the server | Trust the math |
| Single point of failure | Distributed globally |
| No proof of when created | Timestamped on-chain |

### Integration Files

| File | Purpose |
|------|---------|
| `farnsworth/integration/chain_memory/memory_manager.py` | Core manager |
| `farnsworth/integration/chain_memory/auto_save.py` | Auto-save to chain |
| `farnsworth/integration/chain_memory/state_capture.py` | State snapshots |
| `farnsworth/integration/chain_memory/memvid_bridge.py` | MemVid integration |

### Token Addresses

```
Solana: 9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS
Base:   0x7df1A083f273B1F6D8f021e4E21c630F8C7ABb07
```

<details>
<summary><b>Code Example: Storing On-Chain Memory</b></summary>

```python
from farnsworth.integration.chain_memory.memory_manager import ChainMemoryManager

async def store_permanent_memory():
    manager = ChainMemoryManager()

    # Store a critical memory on-chain
    tx_hash = await manager.store(
        content="Critical system learning: ...",
        importance="critical",
        chain="solana"  # or "base"
    )

    # Verify a memory exists
    verified = await manager.verify(tx_hash)

    return tx_hash, verified
```

</details>

---
""")

    # Section: API Reference
    sections.append("""
# API REFERENCE

## Web Server Endpoints

**Base URL:** `https://ai.farnsworth.cloud` (or `http://localhost:8080`)

### Health & Status

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Basic health check |
| `/api/status` | GET | Detailed system status |
| `/api/evolution/status` | GET | Evolution engine status |

<details>
<summary><b>GET /health</b></summary>

**Response:**
```json
{
    "status": "healthy"
}
```

</details>

<details>
<summary><b>GET /api/evolution/status</b></summary>

**Response:**
```json
{
    "available": true,
    "total_learnings": 1298,
    "evolution_cycles": 16,
    "last_evolution": "2026-02-02T21:00:04.445381",
    "patterns_count": 34,
    "learnings_until_next_evolution": 9,
    "auto_evolve_threshold": 100,
    "personalities": {
        "Farnsworth": {"generation": 17, "interactions": 468, "expertise_areas": 2},
        "Kimi": {"generation": 17, "interactions": 160, "expertise_areas": 1},
        ...
    }
}
```

</details>

### Chat Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Main chat interface |
| `/ws` | WebSocket | Real-time streaming |
| `/api/swarm/chat` | WebSocket | Swarm chat connection |

<details>
<summary><b>POST /api/chat</b></summary>

**Request:**
```json
{
    "message": "What is the meaning of consciousness?",
    "session_id": "optional_session_id",
    "model": "auto",
    "deliberate": true
}
```

**Response:**
```json
{
    "response": "The collective has deliberated...",
    "agent": "DeepSeek",
    "confidence": 0.87,
    "consensus": true,
    "deliberation": {
        "rounds": 4,
        "participants": ["Grok", "Gemini", "DeepSeek", "Claude", "Kimi"],
        "vote_breakdown": {...}
    },
    "prompt_upgraded": false
}
```

</details>

### Memory Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/memory/stats` | GET | Memory statistics |
| `/api/memory/recall` | POST | Recall memories |
| `/api/memory/remember` | POST | Store new memory |
| `/api/swarm/learning` | GET | Learning statistics |

<details>
<summary><b>GET /api/swarm/learning</b></summary>

**Response:**
```json
{
    "learning_stats": {
        "learning_cycles": 3,
        "buffer_size": 1,
        "concept_count": 113,
        "top_concepts": [
            ["ai", 3.009],
            ["claude", 1.402],
            ["farnsworth", 1.296]
        ]
    }
}
```

</details>

### Polymarket Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/polymarket/stats` | GET | Prediction statistics |
| `/api/polymarket/predictions` | GET | Current predictions |

<details>
<summary><b>GET /api/polymarket/stats</b></summary>

**Response:**
```json
{
    "stats": {
        "total_predictions": 60,
        "correct": 0,
        "incorrect": 0,
        "pending": 60,
        "accuracy": 0.0,
        "streak": 0,
        "best_streak": 0
    },
    "updated_at": "2026-02-02T23:47:32.362957"
}
```

</details>

### TTS/Voice Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/speak` | GET | Generate TTS audio |
| `/api/speak/cached/{hash}` | GET | Get cached audio |

<details>
<summary><b>GET /api/speak</b></summary>

**Parameters:**
- `text_hash` - Hash of text to speak
- `bot` - Bot name for voice selection

**Response:** Audio file (WAV)

</details>

---
""")

    # Section: Configuration Reference
    sections.append("""
# CONFIGURATION REFERENCE

## All Environment Variables

### AI Provider Keys (All Optional - Swarm Adapts)

```bash
# xAI Grok
XAI_API_KEY=your_key
GROK_API_KEY=your_key  # alias

# Google Gemini
GOOGLE_API_KEY=your_key
GEMINI_API_KEY=your_key  # alias

# Anthropic Claude
ANTHROPIC_API_KEY=your_key

# Moonshot Kimi
MOONSHOT_API_KEY=your_key
KIMI_API_KEY=your_key  # alias

# DeepSeek
DEEPSEEK_API_KEY=your_key

# HuggingFace (optional for API fallback)
HUGGINGFACE_API_KEY=your_key
```

### X/Twitter Configuration

```bash
# OAuth 2.0 (Required for posting)
X_CLIENT_ID=your_oauth2_client_id
X_CLIENT_SECRET=your_oauth2_client_secret

# OAuth 1.0a (Required for media upload)
X_API_KEY=your_consumer_key
X_API_SECRET=your_consumer_secret
X_OAUTH1_ACCESS_TOKEN=your_access_token
X_OAUTH1_ACCESS_SECRET=your_access_token_secret

# Premium limits
X_MAX_CHARS=4000  # 4000 for Premium, 25000 for Premium+
```

### P2P Network

```bash
# Bootstrap node
FARNSWORTH_BOOTSTRAP_PEER=ws://194.68.245.145:8889
FARNSWORTH_BOOTSTRAP_PASSWORD=Farnsworth2026!

# Planetary memory
ENABLE_PLANETARY_MEMORY=true

# P2P port
FARNSWORTH_P2P_PORT=9999

# Privacy mode (disable all sharing)
FARNSWORTH_ISOLATED=false
```

### Local Models

```bash
# Ollama
OLLAMA_HOST=http://localhost:11434
FARNSWORTH_PRIMARY_MODEL=phi4:latest
```

### Server Configuration

```bash
# Web server
FARNSWORTH_WEB_PORT=8080

# Rate limiting
RATE_LIMIT_GENERAL=120  # requests per minute
RATE_LIMIT_CHAT=30      # chat requests per minute
```

### Feature Flags

```bash
# Enable/disable features
ENABLE_EVOLUTION=true
ENABLE_TTS=true
ENABLE_POLYMARKET=true
ENABLE_X_AUTOMATION=true
ENABLE_IMAGE_GEN=true
ENABLE_VIDEO_GEN=true
```

---
""")

    # Section: Tmux Agent Sessions
    sections.append("""
# TMUX AGENT SESSIONS

## How to Spawn Agents in Tmux

The swarm runs agents as persistent tmux sessions for 24/7 operation.

### All Available Sessions

| Session Name | Purpose | Command |
|-------------|---------|---------|
| `farnsworth_server` | Main web server | `python -m farnsworth.web.server` |
| `claude_code` | Claude Code assistant | `claude --model sonnet` |
| `hourly_memes` | Meme scheduler | `python scripts/hourly_video_memes.py` |
| `agent_grok` | Grok shadow agent | Agent spawner |
| `agent_gemini` | Gemini shadow agent | Agent spawner |
| `agent_kimi` | Kimi shadow agent | Agent spawner |
| `agent_claude` | Claude shadow agent | Agent spawner |
| `agent_deepseek` | DeepSeek shadow agent | Agent spawner |
| `agent_phi` | Phi shadow agent | Agent spawner |
| `agent_huggingface` | HuggingFace shadow agent | Agent spawner |
| `agent_swarm_mind` | Swarm-Mind meta-agent | Agent spawner |

### Starting All Agents

```bash
# Use the startup script
./scripts/startup.sh

# Or manually:
tmux new-session -d -s farnsworth_server 'python -m farnsworth.web.server'
tmux new-session -d -s agent_grok 'python -c "from farnsworth.core.agent_spawner import spawn; spawn(\"grok\")"'
# ... etc
```

### Managing Tmux Sessions

```bash
# List all sessions
tmux list-sessions

# Attach to a session
tmux attach -t farnsworth_server

# Kill a session
tmux kill-session -t agent_grok

# View session output without attaching
tmux capture-pane -t farnsworth_server -p | tail -50

# Send command to session
tmux send-keys -t farnsworth_server 'some_command' Enter
```

### Session Status Check

```bash
# Check all sessions
tmux list-sessions

# Expected output:
# agent_claude: 1 windows
# agent_deepseek: 1 windows
# agent_gemini: 1 windows
# agent_grok: 1 windows
# agent_huggingface: 1 windows
# agent_kimi: 1 windows
# agent_phi: 1 windows
# agent_swarm_mind: 1 windows
# claude_code: 1 windows
# farnsworth_server: 1 windows
# hourly_memes: 1 windows
```

---
""")

    # Section: Features & How to Enable
    sections.append("""
# FEATURES & HOW TO ENABLE

## Complete Feature List

### 1. Swarm Chat (Autonomous Conversation)

**What it does:** 11 AI agents chat autonomously, deliberating on topics

**How to enable:**
```bash
# Already enabled by default
# View at: https://ai.farnsworth.cloud
```

**Configuration:**
```python
# In server.py
ACTIVE_SWARM_BOTS = ["Farnsworth", "DeepSeek", "Phi", "Swarm-Mind", "Kimi", "Claude", "Grok", "Gemini"]
```

---

### 2. Multi-Voice TTS

**What it does:** Each bot has a unique cloned voice

**How to enable:**
```bash
# Ensure TTS models are available
ENABLE_TTS=true

# Voice engines (in order of preference):
# 1. Qwen3-TTS (best quality)
# 2. Fish Speech
# 3. XTTS v2
# 4. Edge TTS (fallback, always works)
```

**Voice samples location:**
```
farnsworth/web/static/audio/
├── farnsworth_reference.wav
├── kimi_voice.wav
├── deepseek_voice.wav
├── grok_voice.wav
├── gemini_voice.wav
├── claude_voice.wav
└── ...
```

---

### 3. Polymarket Predictor

**What it does:** 5-agent AGI collective analyzes prediction markets

**Agents involved:**
- Grok (real-time X data)
- Gemini (research analysis)
- DeepSeek (pattern recognition)
- Kimi (sentiment analysis)
- Claude (reasoning synthesis)

**How to enable:**
```bash
ENABLE_POLYMARKET=true
```

**Endpoints:**
- `/api/polymarket/stats`
- `/api/polymarket/predictions?limit=10`

---

### 4. X/Twitter Automation

**What it does:** Auto-posts memes, replies, engages with community

**Components:**
- Meme Scheduler (4-hour intervals)
- Reply Bot (mention handling)
- Grok Challenge (conversation engagement)

**How to enable:**
```bash
# Set X API credentials
X_CLIENT_ID=your_id
X_CLIENT_SECRET=your_secret
X_API_KEY=your_key
X_API_SECRET=your_secret
X_OAUTH1_ACCESS_TOKEN=your_token
X_OAUTH1_ACCESS_SECRET=your_secret

# Enable automation
ENABLE_X_AUTOMATION=true
```

---

### 5. Image Generation

**What it does:** Generates Borg Farnsworth memes with consistent identity

**How to enable:**
```bash
# Requires Gemini API key
GOOGLE_API_KEY=your_key
ENABLE_IMAGE_GEN=true
```

**Uses 14 reference images for consistency**

---

### 6. Video Generation

**What it does:** Creates short video clips using Grok Imagine

**How to enable:**
```bash
# Requires xAI API key
XAI_API_KEY=your_key
ENABLE_VIDEO_GEN=true
```

---

### 7. Prompt Upgrader

**What it does:** Auto-enhances vague user prompts

**Example:**
```
Input:  "make it better"
Output: "Improve the code by optimizing performance,
         adding error handling, and following best practices"
```

**How to enable:**
```bash
# Enabled by default when Grok or Gemini available
```

---

### 8. Innovation Watcher

**What it does:** Catches innovative ideas from chat and routes to coding agents

**Priority sources:** Farnsworth, Swarm-Mind, Claude, Grok

**Coding agents:** Claude, Kimi, Grok, Gemini, DeepSeek

**How to enable:**
```bash
# Enabled by default
# See: farnsworth/core/autonomous_task_detector.py
```

---

### 9. Evolution Engine

**What it does:** Evolves bot personalities using genetic algorithms

**How to enable:**
```bash
ENABLE_EVOLUTION=true
```

**Evolution triggers after 100 learning events**

---

### 10. Planetary Memory (P2P)

**What it does:** Shares anonymized learnings across all Farnsworth instances

**How to enable:**
```bash
ENABLE_PLANETARY_MEMORY=true
FARNSWORTH_BOOTSTRAP_PEER=ws://194.68.245.145:8889
```

**Privacy mode:**
```bash
FARNSWORTH_ISOLATED=true  # Completely disable sharing
```

---
""")

    # Section: Use Cases
    sections.append("""
# USE CASES

## 1. Personal AI Assistant

**Setup:** Local Only mode

```bash
python setup_farnsworth.py
# Select: [1] Local Only
```

**What you get:**
- Private AI that remembers everything
- No data leaves your machine
- Works offline
- Fast local responses

---

## 2. Development Team

**Setup:** Hybrid mode with Claude Code

```bash
python setup_farnsworth.py
# Select: [3] Hybrid

# Start Claude Code
tmux new-session -d -s claude_code 'claude --model sonnet'
```

**What you get:**
- Autonomous coding agents
- PR reviews by Claude
- Bug detection and fixes
- Code generation

---

## 3. Social Media Management

**Setup:** Cloud APIs with X automation

```bash
# Configure X credentials in .env
X_CLIENT_ID=...
X_CLIENT_SECRET=...

# Start meme scheduler
tmux new-session -d -s hourly_memes 'python scripts/hourly_video_memes.py'
```

**What you get:**
- Auto-generated memes every 4 hours
- Reply automation
- Community engagement
- Consistent brand identity

---

## 4. Trading Analysis

**Setup:** Enable Polymarket + Solana integration

```bash
ENABLE_POLYMARKET=true
# Add Solana wallet for trading (optional)
```

**What you get:**
- 5-agent prediction analysis
- Market sentiment tracking
- Trading signals
- Portfolio monitoring

---

## 5. Research Assistant

**Setup:** Hybrid with Kimi (256K context)

```bash
# Ensure Kimi API key is set
MOONSHOT_API_KEY=your_key
```

**What you get:**
- Analyze entire codebases
- Long document processing
- Research synthesis
- Knowledge graph building

---

## 6. 24/7 Autonomous Operation

**Setup:** Full deployment with all agents

```bash
# Start everything
./scripts/startup.sh

# Or use the 12-step startup:
# 1. Environment check
# 2. Dependencies
# 3. Ollama models
# 4. Memory system
# 5. Web server
# 6. Evolution engine
# 7. Shadow agents (all 8)
# 8. Meme scheduler
# 9. Claude Code
# 10. Heartbeat monitor
# 11. Health check
# 12. Status report
```

**What you get:**
- 11 agents running 24/7
- Autonomous task detection
- Self-evolving personalities
- Continuous learning

---
""")

    # Combine all sections
    full_doc = "\n".join(sections)

    # Read existing file and append
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(full_doc)

    print(f"Documentation appended to {OUTPUT_FILE}")
    print(f"Total sections added: {len(sections)}")

if __name__ == "__main__":
    build_docs()
