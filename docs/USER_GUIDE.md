# Farnsworth User Guide

> Your complete guide to supercharging Claude with persistent memory, specialist agents, and self-evolution.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Using with Claude Code](#using-with-claude-code)
5. [Memory System](#memory-system)
6. [Agent Swarm](#agent-swarm)
7. [Evolution & Learning](#evolution--learning)
8. [Dashboard UI](#dashboard-ui)
9. [Configuration](#configuration)
10. [Troubleshooting](#troubleshooting)
11. [FAQ](#faq)

---

## Introduction

### What is Farnsworth?

Farnsworth is a **companion system** that integrates with Claude Code to give Claude capabilities it doesn't have on its own:

| Without Farnsworth | With Farnsworth |
|-------------------|-----------------|
| Claude forgets everything between sessions | Claude remembers your preferences, projects, and context forever |
| Claude is a single model | Claude can delegate to specialist agents (code, reasoning, research, creative) |
| Claude never improves from your feedback | Claude evolves and adapts based on your interactions |
| You can't see what Claude is "thinking" | Visual dashboard shows memory, agents, and evolution in real-time |

### How Does It Work?

Farnsworth runs locally on your machine and connects to Claude Code via the **Model Context Protocol (MCP)**. When you chat with Claude in Claude Code, it can now:

1. **Store memories** using `farnsworth_remember`
2. **Retrieve context** using `farnsworth_recall`
3. **Delegate tasks** using `farnsworth_delegate`
4. **Learn from feedback** using `farnsworth_evolve`

All processing happens locally - your data never leaves your machine.

---

## Installation

### Prerequisites

- **Python 3.10+**
- **Claude Code** (installed and working)
- **8GB RAM minimum** (16GB recommended)
- **Windows 10+, macOS 11+, or Linux**

### Step 1: Clone the Repository

```bash
git clone https://github.com/timowhite88/Farnsworth.git
cd Farnsworth
```

### Step 2: Install Dependencies

```bash
# Using pip
pip install -r requirements.txt

# Or using the setup script
python scripts/setup.py
```

### Step 3: Install a Local LLM

Farnsworth needs a local language model. The easiest option is Ollama:

```bash
# Install Ollama from https://ollama.ai
# Then pull a model:
ollama pull deepseek-r1:1.5b
```

**Recommended Models:**

| Model | Size | Best For |
|-------|------|----------|
| `deepseek-r1:1.5b` | 1.2GB | General use, great reasoning |
| `qwen3:0.6b` | 400MB | Low-resource systems |
| `phi3:mini` | 2.4GB | Complex tasks |
| `llama3.2:1b` | 700MB | Balanced performance |

### Step 4: Run the Granular Setup Wizard (NEW)
Farnsworth v2.0+ includes a granular setup wizard to give you full control over privacy and active features.

```bash
python main.py --setup
```

The wizard will guide you through:
- **Isolated Mode**: Disable all networking (highly private).
- **Hardware Profile**: Choose how much VRAM/RAM to allocate.
- **Cognitive Engines**: Enable/Disable Theory of Mind, Causal Reasoning, etc.
- **External API Keys**: GitHub, Office 365, X (Twitter), etc.

Settings are saved to a `.env` file in your root directory.

### Step 5: Configure Claude Code
Add Farnsworth to your Claude Code MCP settings:

**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Linux:** `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "farnsworth": {
      "command": "python",
      "args": ["-m", "farnsworth.mcp_server"],
      "cwd": "C:\\Farnsworth"
    }
  }
}
```

> **Note:** Replace `C:\\Farnsworth` with your actual installation path.

### Step 6: Restart Claude Code
Close and reopen Claude Code. You should see Farnsworth tools available.

---

## Quick Start

### Your First Memory

Open Claude Code and try:

```
"Hey Claude, please remember that my name is Alex and I'm working on a Python web scraper project."
```

Claude will use `farnsworth_remember` to store this permanently.

### Recalling Information

Later, in a new session:

```
"What project am I working on?"
```

Claude will use `farnsworth_recall` and respond:

```
"You're working on a Python web scraper project, Alex!"
```

### Delegating to Specialists

For complex tasks:

```
"Can you delegate this algorithm optimization to your reasoning specialist?"
```

Claude will use `farnsworth_delegate` to hand off to the reasoning agent.

### Providing Feedback

Help the system learn:

```
"That response was really helpful, thanks!"
```

Or:

```
"That wasn't quite what I needed - I wanted more detail."
```

Claude uses `farnsworth_evolve` to record your feedback and improve.

---

## Using with Claude Code

### Available Tools

When Farnsworth is connected, Claude has access to these tools:

#### `farnsworth_remember`

Store information in long-term memory.

```
Parameters:
- content (required): The information to remember
- tags (optional): Categories like ["project", "preference"]
- importance (optional): 0.0-1.0 scale (default 0.5)
```

**Example usage by Claude:**
```
"I'll remember that for you."
[Claude calls farnsworth_remember with your preference]
```

#### `farnsworth_recall`

Search and retrieve memories.

```
Parameters:
- query (required): What to search for
- limit (optional): Max results (default 5)
```

**Example usage by Claude:**
```
"Let me check what I know about that..."
[Claude calls farnsworth_recall and uses the context]
```

#### `farnsworth_delegate`

Delegate tasks to specialist agents.

```
Parameters:
- task (required): The task description
- agent_type (optional): "code", "reasoning", "research", "creative", or "auto"
```

**Agent Types:**

| Type | Best For |
|------|----------|
| `code` | Programming, debugging, code review |
| `reasoning` | Logic, math, analysis, step-by-step thinking |
| `research` | Information gathering, summarization |
| `creative` | Writing, brainstorming, ideation |
| `auto` | Let Farnsworth choose the best agent |

#### `farnsworth_evolve`

Provide feedback for system improvement.

```
Parameters:
- feedback (required): Your feedback text
```

#### `farnsworth_status`

Get system health and statistics.

```
No parameters required.
```

### Available Resources

Claude can also access these data streams:

| Resource URI | Description |
|-------------|-------------|
| `farnsworth://memory/recent` | Recent conversation context |
| `farnsworth://memory/graph` | Knowledge graph of entities |
| `farnsworth://agents/active` | Currently running agents |
| `farnsworth://evolution/fitness` | Performance metrics |
| `farnsworth://vision/reconstruction` | 3D Point cloud data |

---

## Spatio-Temporal Intelligence (v2.0+)

Farnsworth can now understand action over time and spatial depth.

### Video v2.1: Flow Analysis
Unlike basic frame captions, Video v2.1 uses **Dense Optical Flow (Farneback)** to track movement.
- **Action Peaks**: It identifies moments of high activity automatically.
- **Narrative Synthesis**: It correlates what it SEES with what it HEARS to understand the goal of a video.

### 🧊 3D Scene Reconstruction
Farnsworth can build a **Sparse Point Cloud** from video keyframes using Structure from Motion (SfM). This allows the agent to have a spatial mental model of an environment or object.

---

## Swarm Collaboration (v2.5)

The **Swarm Fabric** allows multiple Farnsworth nodes to collaborate securely over a local network.

- **TCP Gossip**: Scalable knowledge dissemination via a gossip protocol.
- **Task Auctions (DTA)**: If your machine is busy, Farnsworth can auction a heavy task to another node in the swarm.

### 🛡️ Isolated Mode (Offline/Private)
If you do not want your agent to communicate with others:
1. Run `python main.py --setup`
2. Select **YES** for "Enable ISOLATED MODE"
3. This hard-disables all UDP discovery and TCP swarm listening.


---

## Memory System

### How Memory Works

Farnsworth uses a **hierarchical memory system** inspired by human cognition:

```
┌─────────────────────────────────────────────────────┐
│                  Working Memory                      │
│         (Current conversation context)               │
│                    ~8,000 tokens                     │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│                 Recall Memory                        │
│      (Searchable conversation history)               │
│              Last 1000 conversations                 │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│                Archival Memory                       │
│        (Permanent semantic storage)                  │
│               Unlimited capacity                     │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│               Knowledge Graph                        │
│    (Entities and relationships)                      │
│        People, projects, concepts                    │
└─────────────────────────────────────────────────────┘
```

### Memory Types

#### Working Memory
- **What:** Current session context
- **Lifespan:** Current conversation
- **Use:** Immediate context for responses

#### Recall Memory
- **What:** Conversation history
- **Lifespan:** Permanent (searchable)
- **Use:** "What did we discuss about X?"

#### Archival Memory
- **What:** Facts, preferences, knowledge
- **Lifespan:** Permanent
- **Use:** Long-term storage of important information

#### Knowledge Graph
- **What:** Entities and their relationships
- **Lifespan:** Permanent
- **Use:** "How is X related to Y?"

### Memory Dreaming

During idle time, Farnsworth performs **memory consolidation**:

1. **Clustering:** Groups related memories together
2. **Pattern Discovery:** Finds recurring themes
3. **Insight Generation:** Creates new connections
4. **Forgetting:** Removes low-value memories

This happens automatically in the background.

### Tips for Effective Memory Use

1. **Be explicit:** "Remember that I prefer TypeScript" works better than "I like TS"
2. **Use tags:** Ask Claude to tag memories for easier retrieval
3. **Review periodically:** Ask "What do you remember about me?" to verify
4. **Provide context:** "Remember for my work project that..." helps organization

---

## Agent Swarm

### What Are Agents?

Agents are **specialist AI workers** that Claude can delegate to. Each agent has:

- A specific area of expertise
- Its own local LLM for processing
- The ability to work in parallel with others

### Available Agents

#### Code Agent 🖥️
- **Expertise:** Programming, debugging, code review
- **Model:** Uses reasoning-optimized LLM
- **Best for:** "Write a function that...", "Debug this code...", "Review my implementation..."

#### Reasoning Agent 🧠
- **Expertise:** Logic, math, analysis
- **Model:** DeepSeek-R1 (chain-of-thought)
- **Best for:** "Analyze this problem...", "What's the best approach to...", "Calculate..."

#### Research Agent 🔍
- **Expertise:** Information synthesis, summarization
- **Model:** General-purpose LLM
- **Best for:** "Summarize this document...", "What are the key points of...", "Compare X and Y..."

#### Creative Agent 🎨
- **Expertise:** Writing, brainstorming, ideation
- **Model:** Temperature-tuned for creativity
- **Best for:** "Write a story about...", "Brainstorm ideas for...", "Create a tagline..."

### How Delegation Works

```
You: "Can you analyze this algorithm's complexity?"

Claude: [Recognizes this needs reasoning]
        [Calls farnsworth_delegate with agent_type="reasoning"]

Reasoning Agent: [Receives task]
                 [Performs step-by-step analysis]
                 [Returns detailed breakdown]

Claude: [Receives agent's response]
        [Formats and presents to you]
```

### Multi-Agent Tasks

For complex tasks, multiple agents can collaborate:

```
You: "Research the best Python web frameworks, analyze their trade-offs,
      and write a recommendation document."

Claude delegates to:
1. Research Agent → Gathers framework information
2. Reasoning Agent → Analyzes trade-offs
3. Creative Agent → Writes the document

Results are combined and presented to you.
```

---

## Evolution & Learning

### How Farnsworth Learns

Farnsworth uses **genetic algorithms** to improve over time:

```
┌─────────────────────────────────────────────────────┐
│                  Your Feedback                       │
│  "That was helpful!" / "That wasn't what I needed"  │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│               Fitness Tracking                       │
│    task_success, efficiency, user_satisfaction      │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│             Genetic Optimization                     │
│   Select best configs → Crossover → Mutate          │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│              Improved Behavior                       │
│     Better responses, faster retrieval, etc.        │
└─────────────────────────────────────────────────────┘
```

### Fitness Metrics

Farnsworth tracks these performance metrics:

| Metric | Description | Weight |
|--------|-------------|--------|
| `task_success` | Did the response achieve its goal? | 30% |
| `user_satisfaction` | Based on your explicit feedback | 30% |
| `efficiency` | Token usage and response time | 20% |
| `response_quality` | Relevance and coherence | 20% |

### Providing Good Feedback

The more feedback you provide, the faster Farnsworth learns:

**Positive Feedback:**
- "That was exactly what I needed!"
- "Perfect, thanks!"
- "Great explanation!"

**Negative Feedback:**
- "That wasn't quite right..."
- "I needed more detail"
- "The code doesn't work"

**Specific Feedback:**
- "I prefer shorter responses"
- "I like when you show code examples"
- "Explain things like I'm a beginner"

### Viewing Evolution Progress

In the Streamlit dashboard, you can see:

- Current fitness scores
- Improvement trends over time
- Generation history
- Top-performing configurations

---

## Dashboard UI

### Starting the Dashboard

```bash
# From the Farnsworth directory:
python main.py --ui

# Or directly with Streamlit:
streamlit run farnsworth/ui/streamlit_app.py
```

The dashboard opens at `http://localhost:8501`

### Dashboard Pages

#### 💬 Chat
- Interactive chat interface
- Shows memory context being used
- Displays active working memory

#### 🧠 Memory
- Browse all stored memories
- Search by content or tags
- View knowledge graph visualization
- Trigger memory dreaming manually

#### 🤖 Agents
- See active agents and their status
- View task history
- Monitor agent performance

#### 📈 Evolution
- Fitness metrics over time
- Generation progress
- Improvement suggestions
- Behavior parameters

#### ⚙️ Settings
- Configure models and backends
- Adjust memory settings
- Set evolution parameters
- Manage integrations

---

## Configuration

### Main Configuration File

`configs/default.yaml`:

```yaml
# Core settings
data_dir: "./data"
log_level: "INFO"

# Memory settings
memory:
  archival_max_entries: 100000
  embedding_model: "all-MiniLM-L6-v2"
  embedding_dimensions: 384
  importance_threshold: 0.3
  enable_dreaming: true
  dream_interval_minutes: 30

# Agent settings
agents:
  max_concurrent: 5
  default_timeout: 120
  enable_cascade: true
  confidence_threshold: 0.7

# Evolution settings
evolution:
  population_size: 50
  generations_per_cycle: 10
  mutation_rate: 0.1
  elite_count: 5
```

### Model Configuration

`configs/models.yaml`:

```yaml
models:
  primary:
    name: "deepseek-r1:1.5b"
    backend: "ollama"
    context_size: 8192

  draft:
    name: "qwen3:0.6b"
    backend: "ollama"
    context_size: 4096

  embedding:
    name: "all-MiniLM-L6-v2"
    dimensions: 384
```

### Environment Variables

```bash
# Optional overrides
export FARNSWORTH_DATA_DIR="/path/to/data"
export FARNSWORTH_LOG_LEVEL="DEBUG"
export OLLAMA_HOST="http://localhost:11434"
```

---

## Troubleshooting

### Common Issues

#### "Farnsworth tools not appearing in Claude Code"

1. Verify the MCP config path is correct
2. Check that Python can find the farnsworth module:
   ```bash
   python -c "import farnsworth; print('OK')"
   ```
3. Restart Claude Code completely
4. Check Claude Code's MCP logs for errors

#### "Memory recall returns nothing"

1. Verify memories were stored:
   ```bash
   python main.py --cli
   > status
   ```
2. Check if embeddings are working:
   ```bash
   python -c "from farnsworth.rag.embeddings import EmbeddingManager; print(EmbeddingManager().embed('test')[:5])"
   ```

#### "Agent delegation times out"

1. Check if Ollama is running:
   ```bash
   ollama list
   ```
2. Pull the required model:
   ```bash
   ollama pull deepseek-r1:1.5b
   ```
3. Increase timeout in config

#### "High memory usage"

1. Use a smaller model (qwen3:0.6b)
2. Reduce `max_concurrent` agents
3. Lower `archival_max_entries`
4. Disable speculative decoding

### Getting Help

1. Check the [FAQ](#faq) below
2. Search [GitHub Issues](https://github.com/timowhite88/Farnsworth/issues)
3. Open a new issue with:
   - Your OS and Python version
   - Error messages
   - Steps to reproduce

---

## FAQ

### General

**Q: Is my data sent to the cloud?**
A: No. Everything runs locally. Your memories, conversations, and feedback never leave your machine.

**Q: Does this work with Claude.ai (web)?**
A: No, only with Claude Code (the CLI/desktop app) via MCP.

**Q: Can I use this without a GPU?**
A: Yes! Farnsworth is optimized for CPU usage. GPU just makes it faster.

**Q: How much disk space do I need?**
A: ~10GB minimum (5GB for models + 5GB for data). Recommended 50GB.

### Memory

**Q: How many memories can Farnsworth store?**
A: Virtually unlimited. The default config supports 100,000+ entries.

**Q: Can I export my memories?**
A: Yes, memories are stored in `data/memories/` as JSON files.

**Q: Can I manually add memories?**
A: Yes, through the CLI or dashboard.

### Agents

**Q: Are agents different AI models?**
A: They can be. By default, they share the same base model but with different prompts and parameters.

**Q: Can I add custom agents?**
A: Yes! See the developer documentation for creating custom specialists.

### Evolution

**Q: How long until I see improvements?**
A: Usually 10-20 generations (a few days of normal use).

**Q: Can evolution make things worse?**
A: The system keeps the best configurations and only applies improvements that pass testing.

**Q: Can I reset evolution?**
A: Yes, delete `data/evolution/` to start fresh.

---

## Next Steps

- **Explore the Dashboard:** Run `python main.py --ui` and browse your memories
- **Read the README:** Full technical details at [README.md](README.md)
- **Check the Roadmap:** Upcoming features at [ROADMAP.md](ROADMAP.md)
- **Contribute:** See [CONTRIBUTING.md](CONTRIBUTING.md)

---

*"Good news, everyone!" - Professor Farnsworth*
