"""
Farnsworth Web Server
Token-gated chat interface with Solana wallet verification
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Optional Solana imports
try:
    from solana.rpc.api import Client as SolanaClient
    from solders.pubkey import Pubkey
    SOLANA_AVAILABLE = True
except ImportError:
    SOLANA_AVAILABLE = False

# Optional Ollama imports
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
REQUIRED_TOKEN = os.getenv("FARNSWORTH_REQUIRED_TOKEN", "9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS")
MIN_TOKEN_BALANCE = int(os.getenv("FARNSWORTH_MIN_TOKEN_BALANCE", "1"))
SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
PRIMARY_MODEL = os.getenv("FARNSWORTH_PRIMARY_MODEL", "deepseek-r1:1.5b")
DEMO_MODE = os.getenv("FARNSWORTH_DEMO_MODE", "true").lower() == "true"

# Get paths
WEB_DIR = Path(__file__).parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

# Initialize FastAPI
app = FastAPI(
    title="Farnsworth Neural Interface",
    description="Token-gated AI companion chat interface",
    version="2.8.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# Request models
class ChatRequest(BaseModel):
    message: str
    wallet: Optional[str] = None
    history: Optional[list] = None


class TokenVerifyRequest(BaseModel):
    wallet_address: str


# Solana client
solana_client = None
if SOLANA_AVAILABLE and not DEMO_MODE:
    try:
        solana_client = SolanaClient(SOLANA_RPC_URL)
        logger.info(f"Solana client connected to {SOLANA_RPC_URL}")
    except Exception as e:
        logger.warning(f"Failed to connect to Solana: {e}")


def get_token_balance(wallet_address: str) -> int:
    """Get SPL token balance for a wallet."""
    if not SOLANA_AVAILABLE or not solana_client:
        logger.warning("Solana not available, returning demo balance")
        return 1  # Demo mode

    try:
        wallet_pubkey = Pubkey.from_string(wallet_address)
        token_pubkey = Pubkey.from_string(REQUIRED_TOKEN)

        # Get token accounts
        response = solana_client.get_token_accounts_by_owner(
            wallet_pubkey,
            {"mint": token_pubkey}
        )

        if response.value:
            for account in response.value:
                account_data = account.account.data
                # Parse token account data to get balance
                # This is simplified - real implementation would parse the account data properly
                return 1  # Assume they have tokens if account exists

        return 0

    except Exception as e:
        logger.error(f"Error checking token balance: {e}")
        return 0 if not DEMO_MODE else 1


def generate_ai_response(message: str, history: list = None) -> str:
    """Generate AI response using Ollama or fallback."""

    # System prompt
    system_prompt = """You are Farnsworth, a Claude Companion AI with advanced capabilities.
You have persistent memory, specialist agents, and can evolve from feedback.

This is a LIMITED DEMO interface. Important rules:
1. Keep responses concise (2-3 paragraphs max)
2. Remind users this is a demo with limited features
3. Encourage users to install locally for full capabilities
4. Be helpful and engaging, but mention limitations when relevant

Full features (install locally): Solana trading, P2P networking, model swarms, vision, voice, evolution.

Respond in a friendly, slightly quirky manner. Reference "Good news, everyone!" occasionally."""

    # Try Ollama
    if OLLAMA_AVAILABLE:
        try:
            # Build messages
            messages = [{"role": "system", "content": system_prompt}]

            if history:
                for h in history[-10:]:  # Last 10 messages
                    messages.append({
                        "role": h.get("role", "user"),
                        "content": h.get("content", "")
                    })

            messages.append({"role": "user", "content": message})

            # Generate response
            response = ollama.chat(
                model=PRIMARY_MODEL,
                messages=messages,
                options={"temperature": 0.7, "num_predict": 500}
            )

            return response["message"]["content"]

        except Exception as e:
            logger.error(f"Ollama error: {e}")

    # Fallback responses
    return generate_fallback_response(message)


def generate_fallback_response(message: str) -> str:
    """Generate a fallback response when Ollama is not available."""
    msg_lower = message.lower()

    if "capabil" in msg_lower or "what can you" in msg_lower:
        return """I have many capabilities! Here's what's available:

**In This Demo:**
- Basic conversation and Q&A
- Memory of our chat session
- Voice output (text-to-speech)

**Full Version (Install Locally):**
- Persistent memory across sessions
- Solana trading (Jupiter swaps, DexScreener)
- Model swarm with 12+ specialists
- Vision and image analysis
- P2P planetary memory network
- Self-evolution from feedback

To unlock everything: `pip install farnsworth-ai` or visit the GitHub repo!"""

    if "memory" in msg_lower or "remember" in msg_lower:
        return """My memory system is hierarchical and inspired by human cognition:

**Working Memory** - Current conversation context
**Recall Memory** - Searchable conversation history
**Archival Memory** - Permanent semantic storage
**Knowledge Graph** - Entities and relationships

I even have **Memory Dreaming** - I consolidate memories during idle time!

This demo has limited memory. Install locally for persistent memory across all sessions."""

    if "install" in msg_lower or "setup" in msg_lower or "local" in msg_lower:
        return """Here's how to install Farnsworth locally:

**Quick Install:**
```bash
pip install farnsworth-ai
farnsworth-server
```

**From Source:**
```bash
git clone https://github.com/timowhite88/Farnsworth
cd Farnsworth
pip install -r requirements.txt
python main.py --setup
```

**Docker:**
```bash
docker-compose up -d
```

After installing, add to Claude Desktop config and restart. You'll get infinite memory, trading tools, and all premium features!"""

    if "hello" in msg_lower or "hi" in msg_lower or "hey" in msg_lower:
        return """Good news, everyone! Hello there!

I'm Farnsworth, your Claude Companion AI. In this demo, I can chat with you about my features, explain how I work, and help you get set up with the full version.

What would you like to know? You can ask about my capabilities, memory system, or how to install locally!"""

    # Default response
    return """That's an interesting question! In this demo interface, I have limited capabilities.

I can discuss my features, show examples, and help you get started with the full version. For deeper conversations, Solana trading, and advanced features, you'll want to **install Farnsworth locally**.

Try asking about my *capabilities*, *memory system*, or *how to install*!

What else can I help you with?"""


# Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main chat interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Handle chat messages."""
    try:
        if not request.message:
            raise HTTPException(status_code=400, detail="Message is required")

        # Generate response
        response = generate_ai_response(
            request.message,
            request.history or []
        )

        return JSONResponse({
            "response": response,
            "demo_mode": DEMO_MODE
        })

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/verify-token")
async def verify_token(request: TokenVerifyRequest):
    """Verify wallet holds required token."""
    try:
        if DEMO_MODE:
            return JSONResponse({
                "verified": True,
                "balance": 1,
                "demo_mode": True
            })

        balance = get_token_balance(request.wallet_address)

        return JSONResponse({
            "verified": balance >= MIN_TOKEN_BALANCE,
            "balance": balance,
            "required": MIN_TOKEN_BALANCE
        })

    except Exception as e:
        logger.error(f"Token verification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status")
async def status():
    """Get server status."""
    return JSONResponse({
        "status": "online",
        "version": "2.8.0",
        "demo_mode": DEMO_MODE,
        "ollama_available": OLLAMA_AVAILABLE,
        "solana_available": SOLANA_AVAILABLE,
        "required_token": REQUIRED_TOKEN
    })


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


def main():
    """Run the web server."""
    host = os.getenv("FARNSWORTH_WEB_HOST", "0.0.0.0")
    port = int(os.getenv("FARNSWORTH_WEB_PORT", "8080"))

    logger.info(f"Starting Farnsworth Web Interface on {host}:{port}")
    logger.info(f"Demo Mode: {DEMO_MODE}")
    logger.info(f"Ollama Available: {OLLAMA_AVAILABLE}")
    logger.info(f"Solana Available: {SOLANA_AVAILABLE}")

    uvicorn.run(
        "farnsworth.web.server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
