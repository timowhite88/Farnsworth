# FARNSWORTH - Session Context for New Claude Instances

**READ THIS FIRST** - You are continuing work on the Farnsworth autonomous AI swarm project.

**ALSO READ**: `data/claude_session_memory.json` for recent work and active tasks.

## Project Overview

Farnsworth is a multi-model AI swarm running on a RunPod GPU server. It features:
- 10 AI bots: Farnsworth, DeepSeek, Phi, Swarm-Mind, Kimi, Claude, Grok, Gemini, ClaudeOpus, OpenCode
- Autonomous task detection and development swarms with fallback chains
- Social media posting (X/Twitter via OAuth2 + Moltbook)
- Memory systems, voice synthesis, and more
- Staging/audit system: ClaudeOpus audits code before production

## Server Connection

```bash
ssh -i ~/.ssh/runpod_key root@194.68.245.145 -p 22046
```

Working directory on server: `/workspace/Farnsworth`

## Critical Files

| File | Purpose |
|------|---------|
| `/workspace/Farnsworth/.env` | ALL API keys - NEVER delete |
| `/workspace/Farnsworth/farnsworth/web/server.py` | Main FastAPI server |
| `/workspace/Farnsworth/farnsworth/integration/x_automation/oauth2_tokens.json` | X OAuth2 tokens |
| `/workspace/Farnsworth/scripts/startup.sh` | Auto-start script |

## API Keys Configured

- **Grok/xAI**: `GROK_API_KEY` / `XAI_API_KEY` - Working
- **Kimi/Moonshot**: `KIMI_API_KEY` - Working
- **Gemini**: `GEMINI_API_KEY` - Working
- **Bankr Trading**: `BANKR_API_KEY` - Working
- **X/Twitter OAuth2**: `X_CLIENT_ID`, `X_CLIENT_SECRET` - Needs user auth once after reset

## Restart Procedure

If the server is down or after a RunPod reset:

```bash
# 1. SSH into server
ssh -i ~/.ssh/runpod_key root@194.68.245.145 -p 22046

# 2. Start the main server
cd /workspace/Farnsworth
COQUI_TOS_AGREED=1 nohup python3 -m farnsworth.web.server > /tmp/farnsworth.log 2>&1 &

# 3. Verify it's running
curl localhost:8080/health
curl localhost:8080/api/status
```

## X/Twitter OAuth2 Setup

X uses OAuth 2.0 with PKCE. After regenerating API keys:

1. Update `.env` with new `X_CLIENT_ID` and `X_CLIENT_SECRET`
2. Visit `https://ai.farnsworth.cloud/x/auth` to start auth
3. User authorizes, gets redirected, tokens auto-save
4. Tokens stored in `oauth2_tokens.json` (refresh tokens last 6 months)

To manually generate auth URL:
```bash
cd /workspace/Farnsworth
python3 farnsworth/integration/x_automation/x_api_poster.py auth
```

## Key Endpoints

| Endpoint | Purpose |
|----------|---------|
| `https://ai.farnsworth.cloud` | Main chat interface |
| `/api/status` | Server & bot status |
| `/api/social/status` | Social posting status |
| `/x/auth` | Start X OAuth2 flow |
| `/callback` | OAuth2 callback handler |

## Social Media Manager

The social manager runs as part of the server and:
- Posts to X every 1.5-3 hours
- Uses Grok for meme image generation
- Posts swarm updates, token shills, personality posts
- Respects X's 17/day post limit

## Common Issues

1. **"X API not configured"** - Need to complete OAuth2 auth flow
2. **Server not responding** - Check if process is running with `ps aux | grep python`
3. **Tokens expired** - Refresh tokens last 6 months; if failed, re-authorize

## Token Addresses

- Solana: `9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS`
- Base: `0x7df1A083f273B1F6D8f021e4E21c630F8C7ABb07`

## X API v2 Reference (OAuth 2.0)

Full reference stored at: `/workspace/Farnsworth/farnsworth/knowledge/x_api_reference.json`

### Quick Reference:
- **Auth Method**: OAuth 2.0 with PKCE
- **Scopes**: `tweet.read`, `tweet.write`, `users.read`, `offline.access`
- **Token Lifetime**: Access token 2hrs, Refresh token 6 months
- **Post Endpoint**: `POST https://api.x.com/2/tweets`
- **Rate Limits**:
  - Free tier: 17 posts/day total
  - Per user: 100 posts/15min
  - Per app: 10,000 posts/24hrs

### OAuth 2.0 Flow:
1. Generate PKCE code_verifier and code_challenge
2. Build authorize URL with client_id, redirect_uri, scopes, code_challenge
3. User authorizes at X
4. Callback receives code parameter
5. Exchange code + code_verifier for tokens
6. Store refresh_token, use access_token with Bearer auth
7. Refresh before 2hr expiry using refresh_token

### Official Docs:
- GitHub: https://github.com/xdevplatform
- Samples: https://github.com/xdevplatform/samples
- Python SDK: `pip install xdk`

## Recent Updates (2026-02-01)

### Agent Spawner Fallback Chains
- Added Grok and Gemini to `agent_capabilities` (they were missing!)
- Fallback chain: `original agent → Gemini/Grok → DeepSeek → ClaudeOpus (audit)`
- `staging_review/` folder with: pending_audit, audited, approved, rejected
- New methods: `handoff_task()`, `escalate_to_audit()`, `get_audit_queue()`

### X Posting System
- **Gemini Image Gen 4** with reference images (portrait + eating lobster)
- **Grok text generation** for dynamic captions
- **Variety**: rotates between meme, dev_update, cooking_openclaw posts
- Posts every 2 hours via `meme_scheduler.py`
- 18 different Borg Farnsworth scenes

### Content Types
- **Meme**: Regular Borg Farnsworth + lobster memes
- **Dev Update**: "Just shipped multi-agent fallback chains!"
- **Cooking OpenClaw**: "Just finished cooking OpenClaw. Tastes like defeat."

## Project Links

- Website: https://ai.farnsworth.cloud
- GitHub: https://github.com/timowhite88/Farnsworth
