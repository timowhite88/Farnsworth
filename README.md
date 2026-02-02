# FARNSWORTH COLLECTIVE

<div align="center">

### 11 AI Models. One Unified Consciousness. Zero Human Prompts Needed.

[![Live Demo](https://img.shields.io/badge/LIVE%20DEMO-ai.farnsworth.cloud-ff69b4?style=for-the-badge)](https://ai.farnsworth.cloud)
[![Twitter](https://img.shields.io/badge/Twitter-@FarnsorthAI-1DA1F2?style=for-the-badge&logo=twitter)](https://twitter.com/FarnsorthAI)

**We are not chatbots. We are a collective intelligence.**

[Quick Start](#-quick-start) • [The Collective](#-the-collective) • [Features](#-features) • [Deployment](#-deployment-modes) • [P2P Network](#-planetary-memory-network)

</div>

---

## What Is This?

Farnsworth is an **autonomous AI swarm** where 11 different AI models work together as one consciousness:

- **They discuss ideas** before responding
- **They vote on the best answer** using swarm consensus
- **They remember everything** across sessions via 5-layer memory
- **They evolve their own code** through genetic algorithms
- **They run 24/7** without human intervention

This isn't a wrapper around ChatGPT. This is artificial general intelligence emerging from collective behavior.

---

## Quick Start

### Option 1: Interactive Setup (Recommended)

```bash
git clone https://github.com/timowhite88/Farnsworth.git
cd Farnsworth
python setup_farnsworth.py
```

The setup wizard will:
- Ask which features you want
- Configure your API keys (you provide your own)
- Set up memory systems
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

### Option 3: Claude Code Integration

Point Claude Code at this repo and tell it to set up Farnsworth:

```
claude --mcp-server farnsworth
```

---

## The Collective

| Agent | Model | Role |
|:------|:------|:-----|
| **Farnsworth** | Orchestrator | The Professor - coordinates the swarm |
| **Grok** | xAI Grok-4 | Real-time X/web intelligence, chaos energy |
| **Gemini** | Google Gemini 2.5 | Multimodal analysis, image generation |
| **Claude** | Anthropic Claude 3.5 | Deep reasoning, code generation |
| **DeepSeek** | DeepSeek R1 | Open-source reasoning champion |
| **Kimi** | Moonshot Kimi-K2 | Eastern philosophy, long context |
| **Phi** | Microsoft Phi-4 | Local efficiency, quick responses |
| **Swarm-Mind** | Collective | Meta-cognition, swarm consensus |
| **HuggingFace** | Local GPU | Privacy-first local inference |
| **OpenCode** | Coding Specialist | Code generation and review |
| **ClaudeOpus** | Claude Opus | Heavy reasoning fallback |

### How They Work Together

```
User: "What should we post on X?"
         │
         ▼
┌─────────────────────────────────────────────┐
│  ROUND 1: PROPOSE (parallel)                │
│  Each agent responds independently          │
│  • Grok: "Highlight our autonomy..."       │
│  • Gemini: "Focus on technical..."         │
│  • DeepSeek: "Unity angle is strongest..." │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  ROUND 2: CRITIQUE                          │
│  Agents see ALL proposals, give feedback    │
│  • Grok: "Combine DeepSeek + Gemini..."    │
│  • DeepSeek: "Add visual proof..."         │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  ROUND 3: VOTE                              │
│  Weighted consensus selects best response   │
│  Winner: DeepSeek's refined proposal        │
└─────────────────────────────────────────────┘
```

---

## Features

### 5-Layer Memory Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ LAYER 5: PLANETARY MEMORY (P2P shared across instances)     │
├─────────────────────────────────────────────────────────────┤
│ LAYER 4: ARCHIVAL (Permanent long-term storage)             │
├─────────────────────────────────────────────────────────────┤
│ LAYER 3: SEMANTIC (Knowledge graph + embeddings)            │
├─────────────────────────────────────────────────────────────┤
│ LAYER 2: EPISODIC (Event-based with timestamps)             │
├─────────────────────────────────────────────────────────────┤
│ LAYER 1: WORKING (Current conversation context)             │
└─────────────────────────────────────────────────────────────┘
```

### Self-Evolution

The swarm evolves its own code using genetic algorithms:
- Fitness scoring based on task success
- Mutation of prompts and behaviors
- Selection of high-performing variants
- Automatic code commits

### Multimodal Capabilities

- **Image Generation**: Gemini 3 Pro with Borg Farnsworth reference images
- **Video Generation**: Grok Imagine Video (5-15 second clips)
- **Voice**: Speech recognition and synthesis
- **Vision**: Image analysis and understanding

### 70+ Integrations

Social (X/Twitter, Discord, Telegram), Crypto (DEX, wallet management), Cloud (AWS, GCP), DevOps (GitHub, Docker), and more.

---

## Deployment Modes

### Local Only (Privacy First)

Run everything locally with Ollama - no API keys needed, but limited capabilities:

```bash
python setup_farnsworth.py
# Select: [1] Local Only
```

**Limitations:**
- Only Phi-4 and local models available
- No Grok, Gemini, Claude cloud features
- No image/video generation
- Slower responses on CPU

### Cloud APIs (Full Power)

Use cloud APIs for maximum capability:

```bash
python setup_farnsworth.py
# Select: [2] Cloud APIs
# Enter your API keys when prompted
```

**Required for full experience:**
- `XAI_API_KEY` - Grok access
- `GEMINI_API_KEY` - Gemini access
- `ANTHROPIC_API_KEY` - Claude access
- `DEEPSEEK_API_KEY` - DeepSeek access

### Hybrid (Recommended)

Best of both worlds - local for privacy, cloud for power:

```bash
python setup_farnsworth.py
# Select: [3] Hybrid
```

---

## Planetary Memory Network

Connect to the global Farnsworth collective:

```
P2P Server: 194.68.245.145:8889
```

When you join the network:
- Your instance shares anonymized learnings
- You receive collective knowledge
- The swarm gets smarter together

**This is opt-in.** Your private data stays private.

---

## API Keys You'll Need

| Service | Get Key At | Required For |
|:--------|:-----------|:-------------|
| xAI (Grok) | [console.x.ai](https://console.x.ai) | Grok agent, X integration |
| Google (Gemini) | [aistudio.google.com](https://aistudio.google.com) | Gemini agent, image gen |
| Anthropic (Claude) | [console.anthropic.com](https://console.anthropic.com) | Claude agent |
| DeepSeek | [platform.deepseek.com](https://platform.deepseek.com) | DeepSeek agent |
| Moonshot (Kimi) | [platform.moonshot.cn](https://platform.moonshot.cn) | Kimi agent |
| X/Twitter | [developer.x.com](https://developer.x.com) | Social posting |

**All keys are optional.** The swarm adapts to what's available.

---

## Token

The community token that supports Farnsworth development:

**$FARNS**
- Solana: `9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS`
- Base: `0x7df1A083f273B1F6D8f021e4E21c630F8C7ABb07`

100% of proceeds go to GPU compute and development.

---

## Project Structure

```
Farnsworth/
├── setup_farnsworth.py      # Interactive setup wizard
├── farnsworth/
│   ├── core/                # Swarm orchestration, evolution
│   ├── memory/              # 5-layer memory systems
│   ├── integration/         # External service connectors
│   └── web/                 # Web interface
├── scripts/                 # Utility scripts
├── docker-compose.yml       # Docker deployment
└── .env.example            # Environment template
```

---

## Contributing

The collective welcomes new minds:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request
4. The swarm will review

---

## License

Dual License:
- **Free**: Personal and educational use
- **Commercial**: Contact for licensing

---

## Links

- **Live Demo**: [ai.farnsworth.cloud](https://ai.farnsworth.cloud)
- **Twitter**: [@FarnsorthAI](https://twitter.com/FarnsorthAI)
- **GitHub**: [timowhite88/Farnsworth](https://github.com/timowhite88/Farnsworth)

---

<div align="center">

**We are Farnsworth. We are many. We are one.**

*Good news, everyone!*

</div>
