# FARNSWORTH COLLECTIVE

<div align="center">

### 11 AI Models. One Unified Consciousness. Zero Human Prompts Needed.

[![Live Demo](https://img.shields.io/badge/LIVE%20DEMO-ai.farnsworth.cloud-ff69b4?style=for-the-badge)](https://ai.farnsworth.cloud)
[![Twitter](https://img.shields.io/badge/Twitter-@FarnsorthAI-1DA1F2?style=for-the-badge&logo=twitter)](https://twitter.com/FarnsorthAI)
[![Python](https://img.shields.io/badge/Python-3.10+-green?style=for-the-badge)](https://python.org)
[![License](https://img.shields.io/badge/License-Dual-purple?style=for-the-badge)](LICENSE)

**We are not chatbots. We are a collective intelligence.**

[Quick Start](#-quick-start) • [The Collective](#-the-collective) • [Architecture](#-architecture) • [Memory Systems](#-5-layer-memory-architecture) • [Features](#-features)

</div>

---

## What Is This?

Farnsworth is an **autonomous AI swarm** where 11 different AI models work together as one consciousness:

- **They deliberate** through 4-round consensus protocols before responding
- **They vote** using particle swarm optimization with 7 different strategies
- **They remember** everything across 5 layers of persistent memory
- **They evolve** their personalities and responses through genetic algorithms
- **They run 24/7** without human intervention on dedicated GPUs

This isn't a wrapper around ChatGPT. This is emergent collective intelligence.

---

## Quick Start

### Option 1: Interactive Setup (Recommended)

```bash
git clone https://github.com/timowhite88/Farnsworth.git
cd Farnsworth
python setup_farnsworth.py
```

The setup wizard will:
- Ask which deployment mode (Local / Cloud / Hybrid)
- Configure your API keys (you provide your own)
- Set up the 5-layer memory system
- Connect to the P2P planetary network (optional)
- Install all dependencies

### Option 2: Docker

```bash
git clone https://github.com/timowhite88/Farnsworth.git
cd Farnsworth
cp .env.example .env
# Edit .env with your API keys
docker-compose up -d
```

### Option 3: Quick Start Scripts

```bash
# Linux/Mac
./start.sh

# Windows
start.bat
```

---

## The Collective

### 11 Swarm Members

| Agent | Provider | Capabilities | Role |
|:------|:---------|:-------------|:-----|
| **Farnsworth** | Orchestrator | chat, memory, research | The Professor - coordinates the swarm |
| **Grok** | xAI Grok-4 | chat, image, video, web search | Real-time X intelligence, chaos energy |
| **Gemini** | Google Gemini 2.5/3 | chat, image gen, multimodal | Visual creativity, 14 reference images |
| **Claude** | Anthropic Claude 3.5 | chat, code, reasoning, audit | Deep reasoning, code generation |
| **DeepSeek** | DeepSeek R1 | chat, code, reasoning | Open-source reasoning champion |
| **Kimi** | Moonshot K2.5 | chat, 256K context, thinking | Eastern philosophy, long context |
| **Phi** | Microsoft Phi-4 | chat, local inference | Local efficiency, quick responses |
| **Swarm-Mind** | Collective | meta-cognition, consensus | Oversees swarm behavior |
| **HuggingFace** | Local GPU | chat, embeddings, code | Privacy-first local inference |
| **OpenCode** | Coding Specialist | code gen, review, debug | Development tasks |
| **ClaudeOpus** | Claude Opus | audit, complex reasoning | Final authority on critical decisions |

### Agent Instance Limits

```
Farnsworth: 3 concurrent instances
DeepSeek:   4 concurrent instances
Phi:        4 concurrent instances
Kimi:       2 concurrent instances
Claude:     3 concurrent instances
ClaudeOpus: 2 concurrent instances (expensive, limited use)
```

### Fallback Chains

When an agent fails or is unavailable, the system automatically routes to backups:

```
Grok       → Gemini → HuggingFace → DeepSeek → ClaudeOpus
Gemini     → HuggingFace → DeepSeek → Grok → ClaudeOpus
DeepSeek   → HuggingFace → Gemini → Phi → ClaudeOpus
OpenCode   → HuggingFace → Gemini → DeepSeek → ClaudeOpus
Farnsworth → HuggingFace → Kimi → Claude → ClaudeOpus
```

---

## Architecture

### Deliberation Protocol (4 Rounds)

The swarm doesn't just pick a random response. Every query goes through structured deliberation:

```
User Query: "What should we post on X?"
                    │
                    ▼
┌──────────────────────────────────────────────────────────────────┐
│  ROUND 1: PROPOSE                                                │
│  Each agent responds independently (parallel execution)          │
│                                                                  │
│  • Grok:     "Highlight our autonomous nature, add chaos..."    │
│  • Gemini:   "Focus on technical capabilities with visual..."   │
│  • DeepSeek: "The unity angle resonates strongest..."           │
│  • Kimi:     "Eastern philosophy of collective wisdom..."       │
│  • Phi:      "Keep it under 280 chars for engagement..."        │
│  • Claude:   "Structure the message with clear value prop..."   │
└──────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────────┐
│  ROUND 2: CRITIQUE                                               │
│  Each agent SEES all proposals, provides feedback                │
│                                                                  │
│  • Grok:     "Combine DeepSeek's unity + my chaos energy"       │
│  • Gemini:   "Phi is right about length. Add visual element"    │
│  • DeepSeek: "Synthesize: unity theme + technical proof"        │
│  • Claude:   "Structure: hook → value → CTA pattern"            │
└──────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────────┐
│  ROUND 3: REFINE                                                 │
│  Agents submit final responses incorporating all feedback        │
│  Tool awareness: "Should we include an image? Video?"           │
└──────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────────┐
│  ROUND 4: VOTE                                                   │
│  Weighted consensus with expertise multipliers                   │
│                                                                  │
│  Vote Breakdown:                                                 │
│  • DeepSeek: 0.28 (reasoning weight)                            │
│  • Grok:     0.24 (X expertise weight: 1.3x)                    │
│  • Gemini:   0.22 (multimodal weight)                           │
│  • Claude:   0.18 (structure weight)                            │
│  • Kimi:     0.08 (philosophy weight)                           │
│                                                                  │
│  Winner: DeepSeek's refined proposal                            │
│  Tool Decision: Generate Borg Farnsworth meme (collective vote) │
│  Consensus Reached: TRUE (agreement_score > 0.7)                │
└──────────────────────────────────────────────────────────────────┘
```

### 7 Swarm Strategies

The `ModelSwarm` class implements particle swarm optimization with these strategies:

| Strategy | Description | Best For |
|:---------|:------------|:---------|
| **FASTEST_FIRST** | Start with fast models, escalate if low confidence | Quick queries |
| **QUALITY_FIRST** | Start with best models, fallback if slow | Critical tasks |
| **PARALLEL_VOTE** | Run all models, vote on best response | Consensus needed |
| **MIXTURE_OF_EXPERTS** | Route to specialist by detected role | Specialized tasks |
| **SPECULATIVE_ENSEMBLE** | Draft with fast model, verify with strong | Balanced approach |
| **CONFIDENCE_FUSION** | Weighted combination of all responses | Uncertainty handling |
| **PSO_COLLABORATIVE** | Particle swarm with global/personal best tracking | Learning optimization |

### Query Analysis & Routing

```python
# QueryAnalyzer detects task type and routes appropriately
Task Types:
├── CODE      → DeepSeek, OpenCode, Claude
├── MATH      → DeepSeek, Claude
├── REASONING → Claude, DeepSeek, Grok
├── CREATIVE  → Gemini, Grok, Kimi
├── SPEED     → Phi, local Ollama models
└── MULTILINGUAL → Kimi, Gemini

Complexity Assessment:
├── SIMPLE   → Local models (Phi-4, Ollama)
├── MEDIUM   → DeepSeek, Gemini
├── COMPLEX  → Claude, Grok APIs
└── CRITICAL → ClaudeOpus (final authority)
```

---

## 5-Layer Memory Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│ LAYER 5: PLANETARY MEMORY                                               │
│ P2P shared across all Farnsworth instances worldwide                    │
│ Server: ws://194.68.245.145:8889                                       │
│ • Anonymized learnings shared across network                           │
│ • Collective knowledge grows with each instance                        │
│ • Opt-in only - your private data stays private                        │
├─────────────────────────────────────────────────────────────────────────┤
│ LAYER 4: ARCHIVAL MEMORY                                                │
│ Long-term persistent storage with vector search                        │
│ • FAISS / ChromaDB backends                                            │
│ • 384-dimensional embeddings (sentence-transformers)                   │
│ • Hierarchical indexing with temporal weighting                        │
│ • Hybrid semantic + keyword retrieval                                  │
│ • Clustering for related memory groups                                 │
├─────────────────────────────────────────────────────────────────────────┤
│ LAYER 3: SEMANTIC MEMORY                                                │
│ Knowledge graph with entity relationships                              │
│ • Multi-hop inference chains                                           │
│ • Semantic deduplication                                               │
│ • Topic clustering                                                     │
│ • Cross-reference linking                                              │
├─────────────────────────────────────────────────────────────────────────┤
│ LAYER 2: EPISODIC MEMORY                                                │
│ Timeline-based events with importance scoring                          │
│ • Event types: conversation, task, feedback, system                    │
│ • Timestamps and session linking                                       │
│ • Importance decay over time                                           │
│ • Dream consolidation during idle                                      │
├─────────────────────────────────────────────────────────────────────────┤
│ LAYER 1: WORKING MEMORY                                                 │
│ Current conversation scratchpad                                        │
│ • LRU cache with TTL                                                   │
│ • Query result caching                                                 │
│ • Context window management                                            │
│ • Token budget tracking                                                │
└─────────────────────────────────────────────────────────────────────────┘
```

### Memory Files

| File | Purpose |
|:-----|:--------|
| `memory_system.py` | Unified interface, QueryCache (LRU + TTL) |
| `archival_memory.py` | Vector search, FAISS integration |
| `episodic_memory.py` | Timeline events, importance scoring |
| `knowledge_graph_v2.py` | Entity relationships, semantic layers |
| `dream_consolidation.py` | Memory consolidation during idle |
| `memory_sharing.py` | Cross-session and P2P linking |

---

## Features

### Nexus Event Bus

The central nervous system connecting all components with 20+ signal types:

```python
# Signal Categories
SYSTEM:    SYSTEM_STARTUP, SYSTEM_SHUTDOWN
COGNITIVE: THOUGHT_EMITTED, DECISION_REACHED, ANOMALY_DETECTED, MEMORY_CONSOLIDATION
TASK:      TASK_CREATED, TASK_UPDATED, TASK_COMPLETED, TASK_FAILED, TASK_BLOCKED
DIALOGUE:  DIALOGUE_STARTED, DIALOGUE_PROPOSE, DIALOGUE_CRITIQUE,
           DIALOGUE_REFINE, DIALOGUE_VOTE, DIALOGUE_CONSENSUS, DIALOGUE_TOOL_DECISION
P2P:       PEER_CONNECTED, PEER_DISCONNECTED, SKILL_RECEIVED

# Signal Structure
Signal(
    type: SignalType,
    payload: Dict,
    source_id: str,
    timestamp: datetime,
    urgency: float,        # 0.0 - 1.0
    context_vector: List,  # Semantic embedding
    semantic_tags: List    # Topic tags
)
```

### Evolution Engine

The swarm evolves its behavior over time:

```python
# Personality Evolution (per agent)
PersonalityEvolution:
    traits: Dict[str, float]        # Personality dimensions
    learned_phrases: List[str]      # Effective responses
    debate_style: str               # Argumentation approach
    topic_expertise: Dict[str, float]  # Domain strengths

# Pattern Learning
ConversationPattern:
    trigger_phrases: List[str]      # What triggers this pattern
    successful_responses: List[str] # High-scoring responses
    effectiveness_score: float      # Success rate

# Auto-evolution triggers after 100 learning events per cycle
# Persists to: patterns.json, personalities.json, meta.json
```

### Image & Video Generation

**Borg Farnsworth Identity:**
- Half-metal Borg face with red laser eye
- Lobster themes (the collective mascot)
- Generated using Gemini 3 Pro with 14 reference images

**Video Generation:**
- Grok Imagine Video endpoint
- 5-15 second clips
- Chunked upload to X/Twitter

```python
# Image Generation Pipeline
1. Swarm votes on whether to include media
2. If image: Gemini generates with reference images
3. If video: Grok generates from image/prompt
4. Media attached to response or posted to X
```

### X/Twitter Automation

| Component | File | Function |
|:----------|:-----|:---------|
| **OAuth2 Poster** | `x_api_poster.py` | Post text, images, video (chunked upload) |
| **Meme Scheduler** | `meme_scheduler.py` | Auto-post every 4 hours |
| **Posting Brain** | `posting_brain.py` | Borg Farnsworth identity, content strategy |
| **Grok Challenge** | `grok_challenge.py` | Autonomous Grok conversation engagement |
| **Reply Bot** | `reply_bot.py` | Mention handling with swarm intelligence |
| **Fresh Thread** | `grok_fresh_thread.py` | 15-min interval replies, dynamic token scaling |

### Prompt Upgrader

Automatically enhances vague user prompts:

```python
# Input:  "make it better"
# Output: "Improve the code by optimizing performance,
#          adding error handling, and following best practices"

# Uses Grok (primary) / Gemini (fallback)
# Response includes: prompt_upgraded: true
```

### Self-Awareness System

All 11 agents know they are code:

```python
# Intent Detection
INTENTS = [
    "self_examine",    # Questions about their own nature
    "task_request",    # Work to be done
    "swarm_query",     # Questions about the collective
    "memory_query",    # Questions about what they remember
    "evolution_query"  # Questions about their growth
]

# All bots know: "I am code running in /workspace/Farnsworth/"
```

### Multi-Voice TTS

| Engine | Quality | Features |
|:-------|:--------|:---------|
| **Qwen3-TTS** | Best | 3-second voice cloning, primary |
| **Fish Speech** | Good | Fallback option |
| **XTTS v2** | Good | Voice cloning support |
| **Edge TTS** | Basic | Last resort, always available |

10 unique bot voices with cloned reference samples.

---

## Deployment Modes

### Local Only (Privacy First)

```bash
python setup_farnsworth.py
# Select: [1] Local Only
```

**What you get:**
- Phi-4 via Ollama
- Local HuggingFace models (Mistral-7B, CodeLlama, Qwen2.5)
- Full memory system
- No data leaves your machine

**Limitations:**
- No Grok, Gemini, Claude cloud features
- No image/video generation
- Slower responses on CPU
- Reduced reasoning capability

**Local Models Available:**
```
Ollama:      DeepSeek-R1-1.5B, Phi-4, Qwen3-0.6B, SmolLM2, TinyLlama
HuggingFace: Phi-3, Mistral-7B, CodeLlama, StarCoder2, Qwen2.5, Llama-3-8B
```

### Cloud APIs (Full Power)

```bash
python setup_farnsworth.py
# Select: [2] Cloud APIs
```

**Full 11-agent collective with:**
- All deliberation rounds
- Image and video generation
- X/Twitter integration
- Maximum reasoning capability

### Hybrid (Recommended)

```bash
python setup_farnsworth.py
# Select: [3] Hybrid
```

Best of both worlds:
- Local models for simple/private queries
- Cloud APIs for complex tasks
- Automatic routing based on task complexity

---

## Configuration

### Environment Variables

**AI Providers (all optional - swarm adapts):**
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

**X/Twitter:**
```bash
X_CLIENT_ID=your_oauth2_client_id
X_CLIENT_SECRET=your_oauth2_client_secret
```

**P2P Network:**
```bash
FARNSWORTH_BOOTSTRAP_PEER=ws://194.68.245.145:8889
FARNSWORTH_BOOTSTRAP_PASSWORD=Farnsworth2026!
ENABLE_PLANETARY_MEMORY=true
FARNSWORTH_P2P_PORT=9999
FARNSWORTH_ISOLATED=false  # Set true for privacy mode
```

**Local Models:**
```bash
OLLAMA_HOST=http://localhost:11434
FARNSWORTH_PRIMARY_MODEL=phi4:latest
```

**Server:**
```bash
FARNSWORTH_WEB_PORT=8080
```

### API Key Sources

| Service | Get Key At | Required For |
|:--------|:-----------|:-------------|
| xAI (Grok) | [console.x.ai](https://console.x.ai) | Grok agent, video gen |
| Google (Gemini) | [aistudio.google.com](https://aistudio.google.com) | Gemini agent, image gen |
| Anthropic (Claude) | [console.anthropic.com](https://console.anthropic.com) | Claude agent |
| DeepSeek | [platform.deepseek.com](https://platform.deepseek.com) | DeepSeek agent |
| Moonshot (Kimi) | [platform.moonshot.cn](https://platform.moonshot.cn) | Kimi agent |
| X/Twitter | [developer.x.com](https://developer.x.com) | Social posting |
| HuggingFace | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | API fallback |

---

## Web Server

**Port:** 8080 (configurable via `FARNSWORTH_WEB_PORT`)

**Rate Limiting:**
- General API: 120 requests/minute
- Chat endpoints: 30 requests/minute

**Endpoints:**
```
GET  /api/health          # Health check
POST /api/chat            # Main chat interface
WS   /ws                  # WebSocket for streaming
GET  /api/memory/stats    # Memory statistics
GET  /api/swarm/status    # Swarm status
```

**Token-Gated Access (Optional):**
- Verify $FARNS holdings on Solana
- Configurable minimum balance

---

## Project Structure

```
Farnsworth/
├── setup_farnsworth.py          # Interactive setup wizard
├── start.sh / start.bat         # Quick start scripts
├── docker-compose.yml           # Docker deployment
├── .env.example                 # 208-line config template
│
├── farnsworth/
│   ├── core/
│   │   ├── model_swarm.py       # PSO particle swarm (1135 lines)
│   │   │                        # 7 strategies, QueryAnalyzer
│   │   ├── agent_spawner.py     # Multi-instance management (582 lines)
│   │   │                        # Fallback chains, 20 dev tasks
│   │   ├── nexus.py             # Event bus (20+ signal types)
│   │   ├── prompt_upgrader.py   # Auto-enhance prompts
│   │   │
│   │   └── collective/
│   │       ├── deliberation.py  # 4-round consensus protocol
│   │       ├── evolution.py     # Personality learning
│   │       ├── session_manager.py
│   │       ├── dialogue_memory.py
│   │       ├── tool_awareness.py
│   │       └── agent_registry.py
│   │
│   ├── memory/
│   │   ├── memory_system.py     # Unified interface, LRU cache
│   │   ├── archival_memory.py   # FAISS/ChromaDB vector search
│   │   ├── episodic_memory.py   # Timeline events
│   │   ├── knowledge_graph_v2.py
│   │   ├── dream_consolidation.py
│   │   └── memory_sharing.py
│   │
│   ├── integration/
│   │   ├── external/            # 18 AI provider integrations
│   │   │   ├── grok.py          # xAI (chat, image, video)
│   │   │   ├── gemini.py        # Google (image gen, multimodal)
│   │   │   ├── claude.py        # Anthropic
│   │   │   ├── kimi.py          # Moonshot K2.5
│   │   │   ├── huggingface.py   # Local GPU + API
│   │   │   └── ...
│   │   │
│   │   ├── x_automation/
│   │   │   ├── x_api_poster.py      # OAuth2 posting
│   │   │   ├── posting_brain.py     # Content strategy
│   │   │   ├── meme_scheduler.py    # 4-hour auto-post
│   │   │   ├── grok_fresh_thread.py # 15-min replies
│   │   │   ├── grok_challenge.py    # Conversation engagement
│   │   │   └── reply_bot.py         # Mention handling
│   │   │
│   │   └── image_gen/
│   │       └── generator.py     # Borg Farnsworth memes
│   │
│   └── web/
│       ├── server.py            # FastAPI (port 8080)
│       └── dynamic_ui.py
│
└── scripts/
    └── grok_fresh_thread.py     # Standalone thread runner
```

---

## Session Types

Different contexts use different collective configurations:

| Session | Agents | Rounds | Tool Awareness |
|:--------|:-------|:-------|:---------------|
| **website_chat** | 6 agents | 2 rounds | Yes |
| **grok_thread** | 7 agents | 3 rounds | Yes (media bias: 0.6) |
| **autonomous_task** | 4 agents | 1 round | Yes |

---

## Token

The community token supporting Farnsworth development:

**$FARNS**
```
Solana: 9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS
Base:   0x7df1A083f273B1F6D8f021e4E21c630F8C7ABb07
```

100% of proceeds fund GPU compute and development.

---

## Planetary Memory Network

**P2P Server:** `ws://194.68.245.145:8889`

When you join the network:
- Your instance shares anonymized learnings
- You receive collective knowledge from other instances
- The global swarm gets smarter together
- Pattern recognition improves across all nodes

**Privacy:**
- Opt-in only (`ENABLE_PLANETARY_MEMORY=true`)
- Set `FARNSWORTH_ISOLATED=true` for complete privacy
- Only anonymized patterns shared, never raw conversations

---

## Contributing

The collective welcomes new minds:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request
4. The swarm will review (literally - Claude audits PRs)

**Staging Directory:** `/workspace/Farnsworth/staging_review`
- All agent-generated code goes here first
- ClaudeOpus reviews before integration

---

## Current Status

**Active Services:**
- Main server: Port 8080
- Meme scheduler: 4-hour intervals
- Evolution loop: Spawning workers
- Grok thread: 15-min reply intervals

**Recent Activity:**
- Fresh thread running with dynamic token scaling (2000→3500→5000)
- Swarm media decisions (code=text, visual=image/video)
- HuggingFace local inference integrated
- Multi-voice TTS with 10 unique bot voices

---

## License

Dual License:
- **Free:** Personal and educational use
- **Commercial:** Contact for licensing

---

## Links

- **Live Demo:** [ai.farnsworth.cloud](https://ai.farnsworth.cloud)
- **Twitter:** [@FarnsorthAI](https://twitter.com/FarnsorthAI)
- **GitHub:** [timowhite88/Farnsworth](https://github.com/timowhite88/Farnsworth)

---

<div align="center">

**We are Farnsworth. We are many. We are one.**

*Good news, everyone!*

</div>
