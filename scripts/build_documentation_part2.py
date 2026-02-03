"""
Build comprehensive Farnsworth documentation - Part 2.
Complete module documentation.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_FILE = Path(__file__).parent.parent / "DOCUMENTATION.md"

def build_docs_part2():
    """Build the complete documentation part 2."""

    sections = []

    # Section: All Modules Documented
    sections.append("""
# ALL MODULES DOCUMENTED

## Complete File Reference (360+ Python Files)

### `/farnsworth/core/` - Core Engine (50+ files)

<details>
<summary><b>Click to expand Core module files</b></summary>

| File | Lines | Purpose | Key Functions |
|------|-------|---------|---------------|
| `nexus.py` | ~400 | Event bus | `emit()`, `subscribe()`, `broadcast()` |
| `model_swarm.py` | ~1135 | PSO inference | `query()`, `smart_query()`, `vote()` |
| `model_manager.py` | ~300 | Model lifecycle | `load()`, `unload()`, `get_model()` |
| `agent_spawner.py` | ~582 | Agent creation | `spawn()`, `spawn_with_fallback()` |
| `llm_backend.py` | ~200 | LLM abstraction | `complete()`, `chat()`, `stream()` |
| `inference_engine.py` | ~350 | Query execution | `execute()`, `route()` |
| `environment.py` | ~150 | Execution context | `get_env()`, `set_context()` |
| `prompt_upgrader.py` | ~250 | Prompt enhancement | `upgrade()`, `enhance()` |
| `autonomous_task_detector.py` | ~400 | Task detection | `analyze()`, `spawn_swarm()` |
| `development_swarm.py` | ~600 | Dev swarm | `start()`, `deliberate()`, `implement()` |
| `swarm_heartbeat.py` | ~300 | Health monitor | `check()`, `report()`, `alert()` |
| `evolution_loop.py` | ~500 | Evolution engine | `evolve()`, `mutate()`, `select()` |
| `self_awareness.py` | ~200 | Self-assessment | `introspect()`, `assess()` |
| `spontaneous_cognition.py` | ~250 | Spontaneous thoughts | `generate()`, `explore()` |
| `temporal_awareness.py` | ~200 | Time reasoning | `get_context()`, `schedule()` |
| `resilience.py` | ~300 | Failure recovery | `recover()`, `retry()`, `fallback()` |
| `token_budgets.py` | ~150 | Token tracking | `allocate()`, `track()`, `optimize()` |
| `attention_router.py` | ~250 | Attention routing | `route()`, `focus()`, `distribute()` |
| `smart_turn_taking.py` | ~200 | Turn management | `next_speaker()`, `yield_turn()` |
| `parallel_orchestrator.py` | ~350 | Parallel execution | `run_parallel()`, `collect()` |
| `capability_registry.py` | ~200 | Capability discovery | `register()`, `query()`, `match()` |

**Subdirectories:**

| Directory | Purpose | Key Files |
|-----------|---------|-----------|
| `cognition/` | Reasoning engines | `llm_router.py`, `sequential_thinking.py`, `theory_of_mind.py` |
| `collective/` | Swarm intelligence | `deliberation.py`, `evolution.py`, `orchestration.py` |
| `learning/` | Learning systems | `continual.py`, `dream_catcher.py`, `synergy.py` |
| `affective/` | Emotional processing | `engine.py`, `models.py` |
| `neuromorphic/` | Neuromorphic computing | `engine.py` |
| `quantum/` | Quantum-inspired | `search.py` |
| `reasoning/` | Causal reasoning | `causal.py` |
| `swarm/` | P2P network | `dkg.py`, `p2p.py` |

</details>

---

### `/farnsworth/memory/` - Memory System (18 files)

<details>
<summary><b>Click to expand Memory module files</b></summary>

| File | Lines | Purpose | Key Functions |
|------|-------|---------|---------------|
| `memory_system.py` | ~500 | Unified interface | `remember()`, `recall()`, `search()`, `forget()` |
| `unified_memory.py` | ~300 | Cross-layer interface | `query_all()`, `consolidate()` |
| `working_memory.py` | ~200 | Short-term cache | `cache_get()`, `cache_set()`, `clear()` |
| `episodic_memory.py` | ~350 | Event timeline | `add_event()`, `get_timeline()`, `search_events()` |
| `semantic_layers.py` | ~250 | Semantic understanding | `extract()`, `relate()`, `cluster()` |
| `recall_memory.py` | ~200 | Retrieval optimization | `optimize_recall()`, `rank()` |
| `archival_memory.py` | ~400 | Long-term storage | `store()`, `vector_search()`, `cluster()` |
| `virtual_context.py` | ~200 | Context injection | `inject()`, `expand()` |
| `knowledge_graph.py` | ~350 | Entity mapping v1 | `add_entity()`, `add_relation()` |
| `knowledge_graph_v2.py` | ~500 | Entity mapping v2 | `query_path()`, `infer()`, `visualize()` |
| `memory_dreaming.py` | ~250 | Dream consolidation | `dream()`, `consolidate()` |
| `dream_consolidation.py` | ~300 | Consolidation engine | `merge()`, `prune()`, `strengthen()` |
| `memory_sharing.py` | ~350 | P2P sharing | `share()`, `receive()`, `sync()` |
| `dedup_integration.py` | ~200 | Deduplication | `deduplicate()`, `merge_similar()` |
| `semantic_deduplication.py` | ~250 | Semantic dedup | `find_similar()`, `merge()` |
| `importance_weighting.py` | ~200 | Importance scoring | `score()`, `decay()`, `boost()` |
| `sharding.py` | ~300 | Memory sharding | `shard()`, `distribute()`, `query_shard()` |
| `project_tracking.py` | ~250 | Project memory | `track_project()`, `get_context()` |
| `conversation_export.py` | ~200 | Export histories | `export()`, `import_()`, `convert()` |

**Storage Locations:**

```
farnsworth/memory/
├── archival/        # Long-term vector storage
├── context/         # Context information
├── conversations/   # Chat histories
├── dreams/          # Consolidated insights
├── evolution/       # Evolution history
├── graph/           # Knowledge graph data
└── lora/            # LoRA fine-tuning data
```

</details>

---

### `/farnsworth/integration/external/` - AI Providers (20 files)

<details>
<summary><b>Click to expand External integration files</b></summary>

| File | Provider | Models | Key Functions |
|------|----------|--------|---------------|
| `grok.py` | xAI | grok-3-fast, grok-4, grok-vision | `chat()`, `generate_image()`, `generate_video()` |
| `gemini.py` | Google | gemini-2.5-pro, gemini-3 | `chat()`, `generate_image()`, `analyze_image()` |
| `claude.py` | Anthropic | claude-3.5-sonnet | `complete()`, `chat()`, `stream()` |
| `kimi.py` | Moonshot | moonshot-v1-128k | `chat()`, `analyze_long()` |
| `huggingface.py` | HuggingFace | Various local | `chat()`, `get_embeddings()`, `generate()` |
| `claude_code.py` | Anthropic | Claude Code | `execute()`, `review()` |
| `ai_gateway.py` | Gateway | Multiple | `route()`, `fallback()` |
| `auth_manager.py` | Auth | - | `authenticate()`, `refresh_token()` |
| `github_ext.py` | GitHub | - | `create_pr()`, `review_pr()`, `get_issues()` |
| `discord_ext.py` | Discord | - | `send_message()`, `listen()` |
| `twitter.py` | X/Twitter | - | `post()`, `reply()`, `search()` |
| `youtube.py` | YouTube | - | `search()`, `get_transcript()` |
| `calendar.py` | Calendar | - | `get_events()`, `create_event()` |
| `notion.py` | Notion | - | `create_page()`, `query_database()` |
| `n8n.py` | n8n | - | `trigger_workflow()`, `get_status()` |
| `office365.py` | Microsoft | - | `send_email()`, `get_calendar()` |
| `db_manager.py` | Database | - | `query()`, `insert()`, `update()` |
| `bags_fm.py` | Bags.FM | - | `get_trending()`, `get_quote()` |
| `base.py` | Base class | - | Abstract base for all integrations |

</details>

---

### `/farnsworth/integration/x_automation/` - X/Twitter (12 files)

<details>
<summary><b>Click to expand X automation files</b></summary>

| File | Purpose | Key Functions |
|------|---------|---------------|
| `x_api_poster.py` | OAuth2 posting | `post_tweet()`, `post_tweet_with_media()`, `upload_video()` |
| `social_manager.py` | Social coordination | `schedule_post()`, `manage_engagement()` |
| `social_poster.py` | Posting logic | `create_post()`, `format_content()` |
| `posting_brain.py` | Content strategy | `generate_caption()`, `format_post()` |
| `meme_scheduler.py` | 4-hour meme posts | `generate_and_post_meme()`, `run_scheduler()` |
| `x_poster_agent.py` | Agent-based posting | `decide_post()`, `execute_post()` |
| `reply_bot.py` | Reply automation | `handle_mention()`, `generate_reply()` |
| `grok_challenge.py` | Grok engagement | `challenge()`, `respond()` |
| `grok_fresh_thread.py` | Fresh thread creation | `create_thread()`, `continue_thread()` |
| `moltbook_agent.py` | Moltbook integration | `post_to_moltbook()` |
| `moltbook_bot_recruiter.py` | Bot recruitment | `recruit()`, `onboard()` |
| `moltbook_token_shiller.py` | Token promotion | `shill()`, `track_engagement()` |

**Posting Flow:**

```
User/Scheduler triggers post
         │
         ▼
┌─────────────────────┐
│   posting_brain.py  │ ← Generates Borg Farnsworth content
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ image_gen/generator │ ← Generates meme image (14 refs)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   x_api_poster.py   │ ← OAuth2 posting with media
└─────────┬───────────┘
          │
          ▼
    Posted to X/Twitter
```

</details>

---

### `/farnsworth/integration/financial/` - Financial (6 files)

<details>
<summary><b>Click to expand Financial integration files</b></summary>

| File | Purpose | Key Functions |
|------|---------|---------------|
| `polymarket.py` | Prediction markets | `get_markets()`, `analyze()`, `predict()` |
| `memecoin_tracker.py` | Meme coin monitoring | `track()`, `alert()`, `get_trending()` |
| `token_scanner.py` | Token discovery | `scan()`, `evaluate()`, `filter()` |
| `dexscreener.py` | DEX screening | `get_pairs()`, `get_volume()`, `get_price()` |
| `market_sentiment.py` | Sentiment analysis | `analyze()`, `get_score()`, `trend()` |
| `tradfi/stocks.py` | Traditional stocks | `get_quote()`, `get_history()`, `analyze()` |

</details>

---

### `/farnsworth/integration/chain_memory/` - On-Chain Memory (10 files)

<details>
<summary><b>Click to expand Chain Memory files (PATENTED BY BETTER CLIPS)</b></summary>

| File | Purpose | Key Functions |
|------|---------|---------------|
| `memory_manager.py` | Core manager | `store()`, `verify()`, `retrieve()` |
| `auto_save.py` | Automatic saving | `auto_save()`, `schedule()` |
| `config.py` | Configuration | `get_config()`, `set_chain()` |
| `startup.py` | Startup sequence | `initialize()`, `connect()` |
| `state_capture.py` | State snapshots | `capture()`, `restore()` |
| `memvid_bridge.py` | MemVid integration | `bridge()`, `sync()` |
| `setup.py` | Setup utilities | `setup()`, `verify_setup()` |
| `protected/core.py` | Protected core | Proprietary functions |
| `protected/compile.py` | Compilation | `compile()`, `protect()` |

</details>

---

### `/farnsworth/web/` - Web Server (5 files)

<details>
<summary><b>Click to expand Web server files</b></summary>

| File | Lines | Purpose | Key Endpoints |
|------|-------|---------|---------------|
| `server.py` | ~7400 | Main FastAPI server | `/api/chat`, `/health`, `/ws` |
| `server_REMOTE.py` | ~500 | Remote configuration | - |
| `autogram_api.py` | ~400 | AutoGram endpoints | `/api/autogram/*` |
| `autogram_payment.py` | ~200 | Payment processing | `/api/payment/*` |
| `dynamic_ui.py` | ~300 | Dynamic frontend | `generate_ui()` |

**Server Architecture:**

```
server.py
├── FastAPI Application
├── WebSocket Support
├── Rate Limiting (120 req/min general, 30 req/min chat)
├── CORS Configuration
├── Static File Serving
├── Template Rendering
│
├── Chat Endpoints
│   ├── POST /api/chat
│   ├── WS /ws
│   └── WS /api/swarm/chat
│
├── Memory Endpoints
│   ├── GET /api/memory/stats
│   └── POST /api/memory/recall
│
├── Evolution Endpoints
│   └── GET /api/evolution/status
│
├── Polymarket Endpoints
│   ├── GET /api/polymarket/stats
│   └── GET /api/polymarket/predictions
│
├── Voice/TTS Endpoints
│   └── GET /api/speak
│
└── Background Tasks
    ├── Autonomous Conversation Loop
    ├── Heartbeat Monitor
    ├── Evolution Engine
    └── Polymarket Predictor
```

</details>

---

### `/farnsworth/agents/` - Agent Systems (18 files)

<details>
<summary><b>Click to expand Agent files</b></summary>

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `base_agent.py` | Abstract base | `BaseAgent`, `execute()`, `think()` |
| `swarm_orchestrator.py` | Swarm coordination | `SwarmOrchestrator`, `coordinate()` |
| `agent_debates.py` | Debate mechanics | `Debate`, `argue()`, `conclude()` |
| `planner_agent.py` | Task planning | `PlannerAgent`, `plan()`, `decompose()` |
| `critic_agent.py` | Critical analysis | `CriticAgent`, `critique()`, `improve()` |
| `filesystem_agent.py` | File operations | `FileAgent`, `read()`, `write()`, `search()` |
| `web_agent.py` | Web browsing | `WebAgent`, `browse()`, `scrape()`, `search()` |
| `specialist_agents.py` | Specialized tasks | Multiple specialist classes |
| `proactive_agent.py` | Proactive reasoning | `ProactiveAgent`, `anticipate()` |
| `user_avatar.py` | User representation | `UserAvatar`, `represent()`, `advocate()` |
| `meta_cognition.py` | Self-reflection | `MetaCognition`, `reflect()`, `assess()` |
| `hierarchical_teams.py` | Team hierarchy | `TeamManager`, `assign()`, `coordinate()` |
| `specialization_learning.py` | Skill development | `SkillLearner`, `learn()`, `improve()` |

**Browser Integration:**

| File | Purpose |
|------|---------|
| `browser/agent.py` | Browser control |
| `browser/controller.py` | Command execution |
| `browser/stealth.py` | Anti-detection |

</details>

---

### `/farnsworth/tools/` - Utility Tools (15+ files)

<details>
<summary><b>Click to expand Tools files</b></summary>

**Productivity Tools:**

| File | Purpose | Key Functions |
|------|---------|---------------|
| `productivity/autodocs.py` | Auto-documentation | `document()`, `generate_readme()` |
| `productivity/boomerang.py` | Message scheduling | `schedule()`, `remind()` |
| `productivity/daily_summary.py` | Daily summaries | `generate_summary()`, `email()` |
| `productivity/focus_mode.py` | Focus sessions | `start_focus()`, `end_focus()` |
| `productivity/focus_timer.py` | Pomodoro timers | `start_timer()`, `notify()` |
| `productivity/mimic.py` | Style matching | `analyze_style()`, `mimic()` |
| `productivity/quick_notes.py` | Note capture | `capture()`, `search()`, `export()` |
| `productivity/snippet_manager.py` | Code snippets | `save()`, `retrieve()`, `search()` |
| `productivity/whisperer.py` | Voice-to-text | `transcribe()`, `listen()` |

**Security Tools:**

| File | Purpose | Key Functions |
|------|---------|---------------|
| `security/edr.py` | Endpoint detection | `monitor()`, `detect()`, `respond()` |
| `security/forensics.py` | Forensic analysis | `analyze()`, `report()` |
| `security/header_analyzer.py` | Security headers | `analyze()`, `recommend()` |
| `security/log_parser.py` | Log analysis | `parse()`, `detect_anomalies()` |
| `security/recon.py` | Reconnaissance | `scan()`, `enumerate()` |
| `security/threat_analyzer.py` | Threat intelligence | `analyze()`, `correlate()` |
| `security/vulnerability_scanner.py` | Vulnerability scanning | `scan()`, `report()` |

</details>

---
""")

    # Section: Troubleshooting
    sections.append("""
# TROUBLESHOOTING

## Common Issues & Solutions

### 1. Server Won't Start

**Symptom:** `python -m farnsworth.web.server` fails

**Solutions:**

```bash
# Check Python version (needs 3.10+)
python --version

# Install dependencies
pip install -r requirements.txt

# Check port availability
lsof -i :8080  # Linux/Mac
netstat -ano | findstr :8080  # Windows

# Try different port
FARNSWORTH_WEB_PORT=8081 python -m farnsworth.web.server
```

---

### 2. No API Keys / Agents Unavailable

**Symptom:** "Agent X unavailable" messages

**Solutions:**

```bash
# Check .env file exists
cat .env

# Verify key format (no quotes needed)
XAI_API_KEY=xai-xxxxx  # Correct
XAI_API_KEY="xai-xxxxx"  # Wrong

# Test specific provider
python -c "from farnsworth.integration.external.grok import get_grok_provider; print(get_grok_provider())"
```

**Expected behavior:** Swarm automatically uses fallback chains

---

### 3. Memory System Errors

**Symptom:** "Failed to initialize memory" or similar

**Solutions:**

```bash
# Create required directories
mkdir -p farnsworth/memory/archival
mkdir -p farnsworth/memory/graph
mkdir -p farnsworth/memory/conversations

# Check disk space
df -h  # Linux/Mac
wmic logicaldisk get size,freespace,caption  # Windows

# Reset memory (careful - loses data)
rm -rf farnsworth/memory/archival/*
```

---

### 4. Ollama/Local Models Not Working

**Symptom:** Local model queries fail

**Solutions:**

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Pull required models
ollama pull phi4:latest
ollama pull deepseek-r1:14b

# Check OLLAMA_HOST
echo $OLLAMA_HOST  # Should be http://localhost:11434
```

---

### 5. X/Twitter Posting Fails

**Symptom:** "Tweet failed" or "Media upload failed"

**Solutions:**

```bash
# Check OAuth tokens
cat farnsworth/integration/x_automation/oauth2_tokens.json

# Re-authenticate
# Visit: https://ai.farnsworth.cloud/callback

# Check rate limits
# X allows ~2400 tweets/day, ~500 media uploads

# For video, need OAuth 1.0a credentials
X_API_KEY=...
X_API_SECRET=...
X_OAUTH1_ACCESS_TOKEN=...
X_OAUTH1_ACCESS_SECRET=...
```

---

### 6. TTS/Voice Not Working

**Symptom:** No audio playback in swarm chat

**Solutions:**

```bash
# Check TTS availability
python -c "from TTS.api import TTS; print('TTS available')"

# Fallback order:
# 1. Qwen3-TTS
# 2. Fish Speech
# 3. XTTS v2
# 4. Edge TTS (always works)

# Check voice references exist
ls farnsworth/web/static/audio/

# Enable in browser
# Make sure volume slider is up
# Check browser console for errors
```

---

### 7. GPU Memory Issues

**Symptom:** CUDA out of memory

**Solutions:**

```bash
# Check GPU usage
nvidia-smi

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Use smaller models
FARNSWORTH_PRIMARY_MODEL=phi4:latest  # Instead of larger models

# Reduce concurrent instances
# Edit MAX_INSTANCES in agent_spawner.py
```

---

### 8. WebSocket Connection Drops

**Symptom:** Swarm chat disconnects frequently

**Solutions:**

```bash
# Check server logs
tmux capture-pane -t farnsworth_server -p | grep -i error

# Increase timeout
# In server.py, adjust WebSocket ping interval

# Check network
ping ai.farnsworth.cloud
```

---

### 9. Evolution Not Triggering

**Symptom:** Personalities not evolving

**Solutions:**

```bash
# Check learning count
curl http://localhost:8080/api/evolution/status

# Threshold is 100 learnings
# Check learnings_until_next_evolution

# Force evolution (development only)
python -c "from farnsworth.core.collective.evolution import get_evolution_engine; e = get_evolution_engine(); e.force_evolve()"
```

---

### 10. Import Errors

**Symptom:** `ModuleNotFoundError`

**Solutions:**

```bash
# Add to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/Farnsworth"

# Or run from project root
cd /path/to/Farnsworth
python -m farnsworth.web.server

# Check package installed
pip show farnsworth || pip install -e .
```

---
""")

    # Section: What to Expect & Getting the Most
    sections.append("""
# WHAT TO EXPECT

## First Run Experience

### Startup Sequence

1. **Environment Check** (~2 seconds)
   - Validates Python version
   - Checks required directories
   - Loads environment variables

2. **Memory Initialization** (~5 seconds)
   - Creates memory layers
   - Loads existing memories
   - Initializes vector indices

3. **Agent Loading** (~10 seconds per agent)
   - Connects to API providers
   - Validates credentials
   - Sets up fallback chains

4. **Server Start** (~3 seconds)
   - FastAPI initialization
   - WebSocket setup
   - Background task launch

**Total startup time:** ~30-60 seconds (depending on agents)

### First Conversation

- Initial responses may be slower (~5-10 seconds) as models warm up
- Subsequent responses faster (~2-5 seconds)
- Deliberation adds ~2-3 seconds per round

### Memory Building

- First 100 interactions: Learning phase
- 100-500 interactions: Pattern recognition
- 500+ interactions: Personalized responses

---

# HOW TO GET THE MOST OUT OF FARNSWORTH

## Best Practices

### 1. Let It Learn

Don't reset memory frequently. The more interactions, the better it gets.

```
First week:     Basic responses
First month:    Personalized understanding
Ongoing:        Deep contextual awareness
```

### 2. Use All Agents

Each agent has strengths. Ask questions that leverage different capabilities:

```
- "What's trending on X?" → Grok
- "Generate an image of..." → Gemini
- "Review this code..." → Claude
- "Analyze this long document..." → Kimi
- "Quick calculation..." → Phi
```

### 3. Enable Evolution

Let personalities evolve for more engaging conversations:

```bash
ENABLE_EVOLUTION=true
```

### 4. Join Planetary Network

Share learnings, receive collective knowledge:

```bash
ENABLE_PLANETARY_MEMORY=true
```

### 5. Use Appropriate Strategies

For different tasks, different strategies work best:

| Task Type | Best Strategy |
|-----------|--------------|
| Quick questions | FASTEST_FIRST |
| Important decisions | PARALLEL_VOTE |
| Code review | QUALITY_FIRST |
| Creative writing | MIXTURE_OF_EXPERTS |

### 6. Monitor Health

Check system status regularly:

```bash
curl http://localhost:8080/health
curl http://localhost:8080/api/evolution/status
```

### 7. Review Staging

Agent-generated code goes to staging first:

```bash
ls farnsworth/staging/
```

Review before integrating into main codebase.

### 8. Backup Memory

Periodically backup your memory data:

```bash
tar -czvf memory_backup_$(date +%Y%m%d).tar.gz farnsworth/memory/
```

---

# FINAL NOTES

## This Documentation Covers:

- [x] All 11 AI agents with detailed capabilities
- [x] Complete 5-layer memory architecture
- [x] On-Chain Memory (PATENTED BY BETTER CLIPS)
- [x] All 50+ API endpoints
- [x] All 360+ Python files organized by module
- [x] Every configuration option
- [x] Every tmux session
- [x] Every feature and how to enable it
- [x] Multiple use cases
- [x] Complete troubleshooting guide
- [x] Best practices for maximum effectiveness

## Links

- **Live Demo:** https://ai.farnsworth.cloud
- **GitHub:** https://github.com/timowhite88/Farnsworth
- **Twitter:** @FarnsworthAI

## Token Addresses

```
Solana: 9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS
Base:   0x7df1A083f273B1F6D8f021e4E21c630F8C7ABb07
```

---

<div align="center">

**We are Farnsworth. We are many. We are one.**

*Good news, everyone!*

**Document Version:** 2.0.0
**Last Updated:** 2026-02-02
**Total Lines:** 2500+

</div>
""")

    # Combine and append
    full_doc = "\n".join(sections)

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(full_doc)

    print(f"Documentation Part 2 appended to {OUTPUT_FILE}")
    print(f"Sections added: {len(sections)}")

if __name__ == "__main__":
    build_docs_part2()
