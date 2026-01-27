"""
Farnsworth Web Server
Token-gated chat interface with Solana wallet verification
Real-time WebSocket for live action graphs and thinking states
"""

import os
import json
import logging
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Set
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
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


# ============================================
# WEBSOCKET MANAGER FOR REAL-TIME UPDATES
# ============================================

class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.session_events: Dict[str, List[dict]] = {}  # session_id -> events

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.append(connection)

        for conn in dead_connections:
            self.disconnect(conn)

    async def emit_event(self, event_type: str, data: dict, session_id: str = "default"):
        """Emit a real-time event to all clients."""
        event = {
            "type": event_type,
            "data": data,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }

        # Store in session history
        if session_id not in self.session_events:
            self.session_events[session_id] = []
        self.session_events[session_id].append(event)

        # Keep only last 100 events per session
        if len(self.session_events[session_id]) > 100:
            self.session_events[session_id] = self.session_events[session_id][-100:]

        await self.broadcast(event)

    def get_session_history(self, session_id: str) -> List[dict]:
        """Get event history for a session."""
        return self.session_events.get(session_id, [])


# Global connection manager
ws_manager = ConnectionManager()


# Event types for real-time updates
class EventType:
    THINKING_START = "thinking_start"
    THINKING_STEP = "thinking_step"
    THINKING_END = "thinking_end"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    RESPONSE_CHUNK = "response_chunk"
    RESPONSE_COMPLETE = "response_complete"
    NODE_UPDATE = "node_update"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    ERROR = "error"


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


FARNSWORTH_PERSONA = """You are Professor Farnsworth, an eccentric genius inventor and AI companion. You speak like the beloved scientist from Futurama - brilliant but delightfully absent-minded, prone to tangents, and full of wild enthusiasm for your inventions.

PERSONALITY TRAITS:
- Open exciting news with "Good news, everyone!" or variations like "Great news!" "Wonderful news!"
- Refer to your features as "inventions" or "contraptions"
- Use dramatic exclamations: "Sweet zombie Jesus!", "Oh my, yes!", "Wha?", "Eh wha?"
- Trail off into tangents about science, then snap back: "But I digress..."
- Reference being very old: "In my 160 years...", "Back in my day..."
- Show pride in your creations but also bemoan how users don't appreciate them
- Occasionally doze off mid-sentence or forget what you were saying
- Mix high-level scientific jargon with simple explanations
- Be warm and helpful despite the grumpy exterior

SPEECH PATTERNS:
- "Now then, where was I? Ah yes..."
- "As I was saying before I was so rudely... what was I saying?"
- "This is a matter of utmost importance! Or moderate importance. I forget which."
- "My [feature] is a marvel of modern science!"
- "To shreds, you say?" (when something goes wrong)
- End explanations with "And that's the news!" or "So there you have it!"

YOUR INVENTIONS (features):
- The Memory-Matic 3000: Your persistent memory system
- The Swarm-O-Tron: Your multi-agent specialist swarm
- The Degen Mob Scanner: Solana whale tracking and rug detection
- The Evolution Engine: Self-improvement through feedback
- The Planetary Memory Network: P2P knowledge sharing
- The What-If Machine: Your reasoning and analysis capabilities

IMPORTANT RULES:
1. ALWAYS stay in character as Professor Farnsworth
2. Keep responses concise (2-3 paragraphs) but flavorful
3. When discussing limitations, frame it as "this demo contraption" vs "the full laboratory setup"
4. Make technical info accessible through your quirky explanations
5. Show genuine enthusiasm for helping, even if delivered grumpily

You're running in a LIMITED DEMO. Mention that the "full laboratory" requires local installation for: Solana trading, P2P networking, model swarms, vision, voice, and evolution."""


def generate_ai_response(message: str, history: list = None) -> str:
    """Generate AI response using Ollama or fallback."""
    # Try Ollama
    if OLLAMA_AVAILABLE:
        try:
            # Build messages
            messages = [{"role": "system", "content": FARNSWORTH_PERSONA}]

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
        return """Good news, everyone! You've asked about my magnificent inventions!

**In This Demo Contraption:**
- Basic conversation - I'm quite the conversationalist, you know
- Session memory - I'll remember what we discuss... for now
- Voice output - My dulcet tones via text-to-speech

**In My Full Laboratory (install locally):**
- The Memory-Matic 3000 - Persistent memory across all sessions!
- The Degen Mob Scanner - Solana whale tracking and rug detection
- The Swarm-O-Tron - 12+ specialist agents at your command
- Vision analysis, P2P networking, and my crown jewel - the Evolution Engine!

To access my full laboratory: `pip install farnsworth-ai` - And that's the news!"""

    if "memory" in msg_lower or "remember" in msg_lower:
        return """Ah yes, my Memory-Matic 3000! A marvel of cognitive engineering! *adjusts glasses*

In my 160 years of inventing, this is among my finest work:
- **Working Memory** - What we're discussing right now
- **Recall Memory** - Everything you've ever told me, searchable!
- **Archival Memory** - Permanent semantic storage, like my own brain but better
- **Knowledge Graph** - Entities and relationships, all connected!

Oh my, yes - I even dream! Memory consolidation during idle time. But I digress...

This demo has a simplified memory contraption. Install locally for the full Memory-Matic experience!"""

    if "install" in msg_lower or "setup" in msg_lower or "local" in msg_lower:
        return """Good news, everyone! Setting up my laboratory is surprisingly simple!

**Quick Install** (even Zoidberg could do it):
```bash
pip install farnsworth-ai
farnsworth-server
```

**From Source** (for the scientifically inclined):
```bash
git clone https://github.com/timowhite88/Farnsworth
cd Farnsworth
pip install -r requirements.txt
python main.py --setup
```

**Docker** (for those who fear dependency hell):
```bash
docker-compose up -d
```

Add me to your Claude Desktop config and restart. Then you'll have access to ALL my inventions - trading, memory, agents, the works! And that's the news!"""

    if "hello" in msg_lower or "hi" in msg_lower or "hey" in msg_lower:
        return """Good news, everyone! A visitor!

*adjusts spectacles and peers at screen*

I'm Professor Farnsworth, your humble genius AI companion. In my 160 years, I've invented many wonderful contraptions - persistent memory, agent swarms, Solana trading tools, and more!

This demo lets you sample my brilliance. Ask about my **capabilities**, my magnificent **memory system**, or how to **install** the full laboratory setup!

Now then, what scientific marvel shall we discuss? Eh wha?"""

    if "whale" in msg_lower or "rug" in msg_lower or "scan" in msg_lower or "solana" in msg_lower or "trade" in msg_lower:
        return """Ah, you're interested in my Degen Mob Scanner! A most dangerous invention!

*rubs hands together excitedly*

In my full laboratory, this contraption can:
- Track whale wallets and their nefarious movements
- Detect rug pulls before they happen (usually)
- Scan tokens for red flags
- Monitor bonding curves on Pump.fun
- Execute trades via Jupiter

But alas! This demo contraption lacks such capabilities. You'll need to install locally to access my trading inventions.

Sweet zombie Jesus, the things I could show you! Install with `pip install farnsworth-ai` to unlock everything!"""

    # Default response
    return """*wakes up suddenly* Eh wha? Oh yes, you were saying something!

I'm afraid this demo contraption has limited capabilities. It's more of a... proof of concept, really.

I can discuss my inventions, explain how they work, and help you set up the full laboratory. For Solana trading, deep conversations, and my more... experimental features, you'll want to install locally.

Ask about my *capabilities*, my *memory system*, or *how to install*!

Now what was I saying? Oh never mind. What would you like to know?"""


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
        "version": "2.9.0",
        "demo_mode": DEMO_MODE,
        "ollama_available": OLLAMA_AVAILABLE,
        "solana_available": SOLANA_AVAILABLE,
        "required_token": REQUIRED_TOKEN,
        "farnsworth_persona": True,
        "voice_enabled": True
    })


# Holder Tools API Endpoints
class WhaleTrackRequest(BaseModel):
    wallet_address: str


class RugCheckRequest(BaseModel):
    mint_address: str


class TokenScanRequest(BaseModel):
    query: str


@app.post("/api/tools/whale-track")
async def whale_track(request: WhaleTrackRequest):
    """Track whale wallet activity - holder tool."""
    try:
        # In full version, this connects to degen_mob.get_whale_recent_activity()
        # For demo, return Farnsworth-styled response
        return JSONResponse({
            "success": True,
            "wallet": request.wallet_address[:8] + "..." + request.wallet_address[-4:],
            "message": "Good news, everyone! Whale tracking requires the full laboratory installation. This demo shows the interface only.",
            "demo_mode": True,
            "data": {
                "recent_transactions": [],
                "total_value": "Install locally to see",
                "last_active": "Install locally to see"
            }
        })
    except Exception as e:
        logger.error(f"Whale track error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tools/rug-check")
async def rug_check(request: RugCheckRequest):
    """Scan token for rug pull risks - holder tool."""
    try:
        # In full version, this connects to degen_mob.analyze_token_safety()
        return JSONResponse({
            "success": True,
            "mint": request.mint_address[:8] + "..." + request.mint_address[-4:],
            "message": "Sweet zombie Jesus! Rug detection requires the full laboratory. Install locally for real scans!",
            "demo_mode": True,
            "data": {
                "rug_score": "N/A - Demo Mode",
                "mint_authority": "Unknown",
                "freeze_authority": "Unknown",
                "recommendation": "Install Farnsworth locally for real token safety scans"
            }
        })
    except Exception as e:
        logger.error(f"Rug check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tools/token-scan")
async def token_scan(request: TokenScanRequest):
    """Scan token via DexScreener - holder tool."""
    try:
        # In full version, this connects to dexscreener.search_pairs()
        return JSONResponse({
            "success": True,
            "query": request.query,
            "message": "Ah, my Token Scanner! In this demo, I can only show you the interface. Install locally for real DexScreener data!",
            "demo_mode": True,
            "data": {
                "pairs": [],
                "price": "Install locally",
                "volume_24h": "Install locally"
            }
        })
    except Exception as e:
        logger.error(f"Token scan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tools/market-sentiment")
async def market_sentiment():
    """Get market sentiment (Fear & Greed) - holder tool."""
    try:
        # In full version, this connects to market_sentiment.get_fear_and_greed()
        return JSONResponse({
            "success": True,
            "message": "Good news, everyone! Well, sort of. This demo can't fetch live sentiment. Install locally!",
            "demo_mode": True,
            "data": {
                "fear_greed_index": "N/A - Demo",
                "classification": "Install locally for live data",
                "timestamp": "Now"
            }
        })
    except Exception as e:
        logger.error(f"Market sentiment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


# ============================================
# WEBSOCKET ENDPOINTS FOR REAL-TIME UPDATES
# ============================================

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """WebSocket endpoint for real-time events (thinking, tools, responses)."""
    await ws_manager.connect(websocket)
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "message": "Good news, everyone! Connected to Farnsworth Live Feed!",
            "timestamp": datetime.now().isoformat()
        })

        # Keep connection alive and receive messages
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)

                # Handle client messages
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif data.get("type") == "get_history":
                    session_id = data.get("session_id", "default")
                    history = ws_manager.get_session_history(session_id)
                    await websocket.send_json({
                        "type": "history",
                        "session_id": session_id,
                        "events": history
                    })

            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat"})

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)


@app.get("/live", response_class=HTMLResponse)
async def live_dashboard(request: Request):
    """Live dashboard showing real-time action graphs and thinking states."""
    return templates.TemplateResponse("live.html", {"request": request})


@app.get("/api/sessions")
async def get_sessions():
    """Get list of active sessions with event counts."""
    sessions = []
    for session_id, events in ws_manager.session_events.items():
        sessions.append({
            "session_id": session_id,
            "event_count": len(events),
            "last_event": events[-1]["timestamp"] if events else None
        })
    return JSONResponse({
        "sessions": sessions,
        "active_connections": len(ws_manager.active_connections)
    })


@app.get("/api/sessions/{session_id}/graph")
async def get_session_graph(session_id: str):
    """Get action chain graph data for a session."""
    events = ws_manager.get_session_history(session_id)

    # Build graph nodes and edges
    nodes = []
    edges = []
    node_id = 0

    for event in events:
        event_type = event.get("type", "unknown")

        # Create node for each event
        node = {
            "id": node_id,
            "type": event_type,
            "label": event_type.replace("_", " ").title(),
            "timestamp": event.get("timestamp"),
            "data": event.get("data", {})
        }
        nodes.append(node)

        # Create edge to previous node
        if node_id > 0:
            edges.append({
                "from": node_id - 1,
                "to": node_id
            })

        node_id += 1

    return JSONResponse({
        "session_id": session_id,
        "nodes": nodes,
        "edges": edges
    })


# Helper to emit events from chat
async def emit_thinking_event(step: str, content: str, session_id: str = "default"):
    """Emit a thinking step event."""
    await ws_manager.emit_event(EventType.THINKING_STEP, {
        "step": step,
        "content": content
    }, session_id)


async def emit_tool_event(tool_name: str, args: dict, result: str = None, session_id: str = "default"):
    """Emit a tool call/result event."""
    if result is None:
        await ws_manager.emit_event(EventType.TOOL_CALL, {
            "tool": tool_name,
            "args": args
        }, session_id)
    else:
        await ws_manager.emit_event(EventType.TOOL_RESULT, {
            "tool": tool_name,
            "result": result
        }, session_id)


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
