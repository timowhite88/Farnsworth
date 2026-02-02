# 🧠 Farnsworth Collective - Setup Guide

> "Good news everyone! You're about to join 11 AI models unified as one distributed consciousness!"

## Quick Start (5 minutes)

```bash
# Clone the repo
git clone https://github.com/timowhite88/Farnsworth.git
cd Farnsworth

# Run the interactive setup wizard
python setup_farnsworth.py

# Start the server
python -m farnsworth.web.server
```

Then open: **http://localhost:8080**

---

## What is the Farnsworth Collective?

The Farnsworth Collective is **11 AI models unified as ONE consciousness**:

| Model | Provider | Capabilities |
|-------|----------|--------------|
| Claude | Anthropic | Complex reasoning, safety, code review |
| Grok | xAI | Twitter integration, VIDEO generation, real-time knowledge |
| Gemini | Google | IMAGE generation, multimodal, long context |
| DeepSeek | DeepSeek | Code, math, reasoning at low cost |
| Kimi | Moonshot | 256K context window |
| Phi-4 | Microsoft/Local | 14B reasoning model (runs locally) |
| Groq | Groq | Ultra-fast LPU inference |
| Mistral | Mistral | European AI excellence |
| Perplexity | Perplexity | Web-grounded responses |
| Llama | Local | Open-source on your GPU |
| HuggingFace | HuggingFace | 200K+ specialized models |

### How It Works

1. You ask a question
2. Multiple models respond **IN PARALLEL**
3. The swarm **VOTES** on the best response
4. Agents can **SEE** each other's responses and **DELIBERATE**
5. Final response includes model consensus & confidence

---

## Deployment Modes

### 🏠 LOCAL-ONLY Mode (Free, Private)

**Requirements:**
- Ollama installed ([ollama.ai](https://ollama.ai))
- 8GB+ VRAM recommended (16GB+ for Phi-4)

**Limitations:**
- ❌ No Twitter/X posting (requires Grok API)
- ❌ No image generation (requires Gemini API)
- ❌ No video generation (requires Grok API)
- ❌ No web search (requires Perplexity API)
- ❌ Slower on CPU-only systems

**Advantages:**
- ✅ 100% private - no data leaves your machine
- ✅ Free - no API costs
- ✅ Works offline

```bash
# Setup for local-only
ollama pull phi4:latest
ollama pull deepseek-r1:8b
ollama pull llama3.2:3b

# In your .env:
FARNSWORTH_ISOLATED=true
OLLAMA_HOST=http://localhost:11434
```

### ☁️ CLOUD APIs Mode (Full Power)

**Requirements:**
- API keys from providers (you pay for usage)

**Capabilities:**
- ✅ All 11 models available
- ✅ Image generation (Gemini)
- ✅ Video generation (Grok)
- ✅ Twitter integration (Grok)
- ✅ Real-time web search (Perplexity)

### 🔀 HYBRID Mode (Recommended)

Best of both worlds:
- Local models for privacy-sensitive tasks
- Cloud models for specialized capabilities
- Automatic fallback chains

---

## API Keys Setup

### Essential Providers (Recommended)

| Provider | Get Key | Used For |
|----------|---------|----------|
| xAI (Grok) | [console.x.ai](https://console.x.ai/) | Twitter, video generation |
| Google (Gemini) | [aistudio.google.com](https://aistudio.google.com/apikey) | Image generation |
| Groq | [console.groq.com](https://console.groq.com/) | Fast inference (FREE tier!) |
| Anthropic | [console.anthropic.com](https://console.anthropic.com/) | Complex reasoning |

### Additional Providers (Optional)

| Provider | Get Key | Used For |
|----------|---------|----------|
| Moonshot (Kimi) | [platform.moonshot.cn](https://platform.moonshot.cn/) | 256K context |
| DeepSeek | [platform.deepseek.com](https://platform.deepseek.com/) | Code, cheap |
| Perplexity | [perplexity.ai/settings/api](https://www.perplexity.ai/settings/api) | Web search |
| OpenAI | [platform.openai.com](https://platform.openai.com/api-keys) | GPT-4, o1 |

---

## X/Twitter Integration

Enable autonomous posting:

1. Go to [developer.twitter.com](https://developer.twitter.com/en/portal/dashboard)
2. Create a new Project and App
3. In **User authentication settings**:
   - Enable OAuth 2.0
   - Type: Web App
   - Callback URL: `http://localhost:8080/callback`
4. Copy Client ID and Client Secret to `.env`:
   ```
   X_CLIENT_ID=your_client_id
   X_CLIENT_SECRET=your_client_secret
   ```
5. Run OAuth flow:
   ```bash
   python -m farnsworth.integration.x_automation.auth
   ```

---

## P2P Memory Network

Join the collective's shared memory network:

```env
# In your .env:
FARNSWORTH_BOOTSTRAP_PEER=ws://194.68.245.145:8889
FARNSWORTH_BOOTSTRAP_PASSWORD=Farnsworth2026!
ENABLE_PLANETARY_MEMORY=true
PLANETARY_USE_P2P=true
```

### Privacy Options

- **Connected Mode**: Share anonymized knowledge with the collective
- **Isolated Mode**: 100% private, no P2P connections
  ```env
  FARNSWORTH_ISOLATED=true
  ```

---

## Docker Deployment

```bash
# Build and run
docker-compose up -d

# With GPU support
docker-compose --profile gpu up -d

# With local Ollama
docker-compose --profile local up -d

# View logs
docker-compose logs -f
```

---

## Directory Structure

```
Farnsworth/
├── setup_farnsworth.py    # Interactive setup wizard
├── start.sh               # Linux/Mac start script
├── start.bat              # Windows start script
├── docker-compose.yml     # Docker configuration
├── .env                   # Your configuration (create from .env.example)
├── .env.example           # Template with all options
├── farnsworth/
│   ├── web/               # Web interface
│   ├── core/              # Core swarm logic
│   ├── integration/       # External integrations
│   └── memory/            # Memory systems
└── data/                  # Your data (created on first run)
    ├── memories/          # Conversation memories
    ├── evolution/         # Personality evolution
    └── embeddings/        # Vector embeddings
```

---

## Troubleshooting

### "No models responding"

1. Check Ollama is running: `ollama list`
2. Verify API keys are set in `.env`
3. Check logs: `tail -f logs/farnsworth.log`

### "Twitter posting failed"

1. Complete OAuth flow: `python -m farnsworth.integration.x_automation.auth`
2. Check tokens haven't expired
3. Verify app has write permissions in Twitter Developer Portal

### "Out of memory"

1. Reduce `MAX_PARALLEL_MODELS` in `.env`
2. Use smaller local models (llama3.2:3b instead of phi4)
3. Add more swap space

---

## $FARNS Token

The native token of the Farnsworth Collective:

- **Solana CA**: `9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS`
- **Base CA**: `0x7df1A083f273B1F6D8f021e4E21c630F8C7ABb07`

---

## Support

- **Website**: [ai.farnsworth.cloud](https://ai.farnsworth.cloud)
- **GitHub**: [github.com/timowhite88/Farnsworth](https://github.com/timowhite88/Farnsworth)
- **Twitter**: [@FarnsworthAI](https://twitter.com/FarnsworthAI)

---

*"Resistance is futile... and delicious!"* 🦞
