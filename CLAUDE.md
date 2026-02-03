# Farnsworth AI Swarm - Claude Code Instructions

**IMPORTANT: On every session start, IMMEDIATELY read this file:**
`farnsworth/memory/claude_session.json`

This contains:
- SSH command to connect to server
- Recent work history with timestamps
- Active services status
- HuggingFace integration details
- Pending tasks
- Architecture notes (fallback chains, memory layers)

## Quick Reference

**SSH to Server:**
```bash
ssh root@194.68.245.145 -p 22046 -i ~/.ssh/runpod_key
```

**Server Workspace:** `/workspace/Farnsworth`
**Local Path:** `C:/Fawnsworth`
**Website:** https://ai.farnsworth.cloud

## Key Files

| File | Purpose |
|------|---------|
| `farnsworth/memory/claude_session.json` | Session memory - READ FIRST |
| `farnsworth/integration/external/huggingface.py` | HuggingFace local inference |
| `farnsworth/core/prompt_upgrader.py` | Auto prompt enhancement |
| `farnsworth/core/model_swarm.py` | PSO collaborative inference |
| `farnsworth/core/collective/evolution.py` | Bot personality evolution |
| `farnsworth/core/agent_spawner.py` | Agent capabilities & fallbacks |
| `.env` (on server) | All environment variables |

## Active Bots (11)

Farnsworth, DeepSeek, Phi, Swarm-Mind, Kimi, Claude, Grok, Gemini, ClaudeOpus, OpenCode, **HuggingFace**

## Token Address

- **Solana:** `9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS`

## HuggingFace Integration (NEW)

Local-first GPU inference without API key:
- **Models:** Phi-3, Mistral-7B, CodeLlama, Qwen2.5, Llama-3
- **Embeddings:** sentence-transformers (MiniLM, BGE, E5)
- **Integration:** Model Swarm PSO, Evolution personalities, Archival Memory

## Fallback Chains

When an agent can't handle a task:
- Grok → Gemini → **HuggingFace** → DeepSeek → ClaudeOpus
- OpenCode → **HuggingFace** → Gemini → DeepSeek → ClaudeOpus
- HuggingFace → DeepSeek → Gemini → ClaudeOpus

## Active Services

| Service | Status | Details |
|---------|--------|---------|
| Main Server | Running | Port 8080, /health endpoint |
| Meme Scheduler | Running | 4-hour interval |
| Evolution Loop | Running | Spawning workers |

## First Actions Each Session

1. Read `farnsworth/memory/claude_session.json` for full context
2. Check server health: `curl https://ai.farnsworth.cloud/health`
3. Review pending tasks in the memory file
4. Update memory file with new work done

## Recent Tweets

- `2017791973478813962` - HuggingFace announcement (2026-02-01)
- `2017777307193094245` - Cooking OpenClaw meme

## Prompt Upgrader

Auto-enhances vague user prompts using Grok/Gemini:
- File: `farnsworth/core/prompt_upgrader.py`
- Integrated into `/api/chat` endpoint
- Response includes `prompt_upgraded: true` when enhanced
