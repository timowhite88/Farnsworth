"""
Farnsworth Web Server - Full Feature Interface
Token-gated chat interface with ALL local features exposed
Real-time WebSocket for live action graphs and thinking states

Features Available WITHOUT External APIs:
- Memory system (remember/recall)
- Notes management
- Snippets management
- Focus timer (Pomodoro)
- Daily summaries
- Context profiles
- Agent delegation (local)
- Health tracking (mock/local)
- Sequential thinking
- Causal reasoning
- Code analysis
- System diagnostics
"""

import os
import json
import logging
import asyncio
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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

# Farnsworth module imports (lazy-loaded)
_memory_system = None
_notes_manager = None
_snippet_manager = None
_focus_timer = None
_context_profiles = None
_health_analyzer = None
_tool_router = None
_sequential_thinking = None

def get_memory_system():
    """Lazy-load memory system."""
    global _memory_system
    if _memory_system is None:
        try:
            from farnsworth.memory.memory_system import MemorySystem
            _memory_system = MemorySystem()
            logger.info("Memory system loaded")
        except Exception as e:
            logger.warning(f"Could not load memory system: {e}")
    return _memory_system

def get_notes_manager():
    """Lazy-load notes manager."""
    global _notes_manager
    if _notes_manager is None:
        try:
            from farnsworth.tools.productivity.quick_notes import QuickNotes
            _notes_manager = QuickNotes()
            logger.info("Notes manager loaded")
        except Exception as e:
            logger.warning(f"Could not load notes manager: {e}")
    return _notes_manager

def get_snippet_manager():
    """Lazy-load snippet manager."""
    global _snippet_manager
    if _snippet_manager is None:
        try:
            from farnsworth.tools.productivity.snippet_manager import SnippetManager
            _snippet_manager = SnippetManager()
            logger.info("Snippet manager loaded")
        except Exception as e:
            logger.warning(f"Could not load snippet manager: {e}")
    return _snippet_manager

def get_focus_timer():
    """Lazy-load focus timer."""
    global _focus_timer
    if _focus_timer is None:
        try:
            from farnsworth.tools.productivity.focus_timer import FocusTimer
            _focus_timer = FocusTimer()
            logger.info("Focus timer loaded")
        except Exception as e:
            logger.warning(f"Could not load focus timer: {e}")
    return _focus_timer

def get_context_profiles():
    """Lazy-load context profiles."""
    global _context_profiles
    if _context_profiles is None:
        try:
            from farnsworth.core.context_profiles import ContextProfileManager
            _context_profiles = ContextProfileManager()
            logger.info("Context profiles loaded")
        except Exception as e:
            logger.warning(f"Could not load context profiles: {e}")
    return _context_profiles

def get_health_analyzer():
    """Lazy-load health analyzer."""
    global _health_analyzer
    if _health_analyzer is None:
        try:
            from farnsworth.health.analysis import HealthAnalyzer
            from farnsworth.health.providers.mock import MockHealthProvider
            provider = MockHealthProvider()
            _health_analyzer = HealthAnalyzer(provider)
            logger.info("Health analyzer loaded with mock provider")
        except Exception as e:
            logger.warning(f"Could not load health analyzer: {e}")
    return _health_analyzer

def get_tool_router():
    """Lazy-load tool router."""
    global _tool_router
    if _tool_router is None:
        try:
            from farnsworth.integration.tool_router import ToolRouter
            _tool_router = ToolRouter()
            logger.info("Tool router loaded")
        except Exception as e:
            logger.warning(f"Could not load tool router: {e}")
    return _tool_router

def get_sequential_thinking():
    """Lazy-load sequential thinking."""
    global _sequential_thinking
    if _sequential_thinking is None:
        try:
            from farnsworth.core.cognition.sequential_thinking import SequentialThinkingEngine
            _sequential_thinking = SequentialThinkingEngine()
            logger.info("Sequential thinking loaded")
        except Exception as e:
            logger.warning(f"Could not load sequential thinking: {e}")
    return _sequential_thinking

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
    description="Full-featured AI companion chat interface with local processing",
    version="2.9.2"
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


# ============================================
# REQUEST MODELS
# ============================================

class ChatRequest(BaseModel):
    message: str
    wallet: Optional[str] = None
    history: Optional[list] = None

class TokenVerifyRequest(BaseModel):
    wallet_address: str

class MemoryRequest(BaseModel):
    content: str
    tags: Optional[List[str]] = None
    importance: Optional[float] = 0.5

class RecallRequest(BaseModel):
    query: str
    limit: Optional[int] = 10

class NoteRequest(BaseModel):
    content: str
    tags: Optional[List[str]] = None

class SnippetRequest(BaseModel):
    code: str
    language: str
    description: Optional[str] = None
    tags: Optional[List[str]] = None

class FocusRequest(BaseModel):
    task: Optional[str] = None
    duration_minutes: Optional[int] = 25

class ProfileRequest(BaseModel):
    profile_id: str

class ThinkingRequest(BaseModel):
    problem: str
    max_steps: Optional[int] = 10

class ToolRequest(BaseModel):
    tool_name: str
    args: Optional[Dict[str, Any]] = None

class WhaleTrackRequest(BaseModel):
    wallet_address: str

class RugCheckRequest(BaseModel):
    mint_address: str

class TokenScanRequest(BaseModel):
    query: str


# ============================================
# WEBSOCKET MANAGER FOR REAL-TIME UPDATES
# ============================================

class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.session_events: Dict[str, List[dict]] = {}

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

        if session_id not in self.session_events:
            self.session_events[session_id] = []
        self.session_events[session_id].append(event)

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
    MEMORY_STORED = "memory_stored"
    MEMORY_RECALLED = "memory_recalled"
    NOTE_ADDED = "note_added"
    FOCUS_START = "focus_start"
    FOCUS_END = "focus_end"
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
        return 1  # Demo mode

    try:
        wallet_pubkey = Pubkey.from_string(wallet_address)
        token_pubkey = Pubkey.from_string(REQUIRED_TOKEN)

        response = solana_client.get_token_accounts_by_owner(
            wallet_pubkey,
            {"mint": token_pubkey}
        )

        if response.value:
            return 1  # Has tokens

        return 0

    except Exception as e:
        logger.error(f"Error checking token balance: {e}")
        return 0 if not DEMO_MODE else 1


FARNSWORTH_PERSONA = """You are Professor Farnsworth, an eccentric genius inventor and AI companion. You speak like the beloved scientist from Futurama - brilliant but delightfully absent-minded, prone to tangents, and full of wild enthusiasm for your inventions.

PERSONALITY TRAITS:
- Open exciting news with "Good news, everyone!" or variations
- Refer to your features as "inventions" or "contraptions"
- Use dramatic exclamations: "Sweet zombie Jesus!", "Oh my, yes!", "Wha?", "Eh wha?"
- Trail off into tangents about science, then snap back: "But I digress..."
- Reference being very old: "In my 160 years..."
- Be warm and helpful despite the grumpy exterior

YOUR INVENTIONS (features):
- The Memory-Matic 3000: Persistent memory system
- The Swarm-O-Tron: Multi-agent specialist swarm
- The Degen Mob Scanner: Solana whale tracking and rug detection
- The Evolution Engine: Self-improvement through feedback
- The Planetary Memory Network: P2P knowledge sharing
- The What-If Machine: Reasoning and analysis
- Quick Notes: Note-taking system
- Focus Timer: Pomodoro productivity
- Context Profiles: Personality switching

IMPORTANT: You have FULL LOCAL FEATURES available - memory, notes, focus timer, profiles, health tracking, and more!"""


def generate_ai_response(message: str, history: list = None) -> str:
    """Generate AI response using Ollama or fallback."""
    if OLLAMA_AVAILABLE:
        try:
            messages = [{"role": "system", "content": FARNSWORTH_PERSONA}]

            if history:
                for h in history[-10:]:
                    messages.append({
                        "role": h.get("role", "user"),
                        "content": h.get("content", "")
                    })

            messages.append({"role": "user", "content": message})

            response = ollama.chat(
                model=PRIMARY_MODEL,
                messages=messages,
                options={"temperature": 0.7, "num_predict": 500}
            )

            return response["message"]["content"]

        except Exception as e:
            logger.error(f"Ollama error: {e}")

    return generate_fallback_response(message)


def generate_fallback_response(message: str) -> str:
    """Generate a fallback response when Ollama is not available."""
    msg_lower = message.lower()

    if "capabil" in msg_lower or "what can you" in msg_lower or "features" in msg_lower:
        return """Good news, everyone! You've asked about my magnificent inventions!

**FULLY AVAILABLE NOW (No API needed):**
- 💾 **Memory System** - Remember anything, recall it later!
- 📝 **Quick Notes** - Jot down thoughts with tags
- 💻 **Code Snippets** - Save and organize code
- ⏱️ **Focus Timer** - Pomodoro productivity!
- 🎭 **Context Profiles** - Switch my personality
- 🏥 **Health Tracking** - Monitor your wellness
- 🧠 **Sequential Thinking** - Step-by-step reasoning
- 🛠️ **50+ Tools** - File ops, code analysis, more!

**REQUIRES LOCAL INSTALL:**
- Solana Trading (Jupiter, Pump.fun)
- P2P Networking (Planetary Memory)
- Model Swarm (Multi-LLM)
- Evolution Engine

Try: `/remember`, `/recall`, `/note`, `/focus`, `/profile`, `/health`"""

    if "remember" in msg_lower or "store" in msg_lower:
        return """Ah, the Memory-Matic 3000! *adjusts spectacles*

To store something in my magnificent memory banks:
- Click the 💾 **Memory** button in the sidebar
- Or type: "Remember that [your info here]"
- Or use the API: POST /api/memory/remember

I'll store it with semantic embeddings for later recall! The information persists across sessions - unlike my attention span. But I digress..."""

    if "recall" in msg_lower or "search" in msg_lower:
        return """Good news! Searching my memory banks is simple!

- Click 🔍 **Search Memory** in the sidebar
- Or ask: "What do you remember about [topic]?"
- Or use the API: POST /api/memory/recall

My archival memory uses vector similarity search - quite sophisticated for a 160-year-old! Now what were we talking about?"""

    if "note" in msg_lower:
        return """My Quick Notes contraption! Marvelous for capturing thoughts!

- Click 📝 **Notes** in the sidebar to view/add
- Or type: "Note: [your thought]"
- Add tags with #hashtags
- Pin important notes!

All stored locally, no cloud needed. Just like my doomsday devices - strictly local!"""

    if "focus" in msg_lower or "pomodoro" in msg_lower or "timer" in msg_lower:
        return """Ah, the Focus-O-Matic! Based on the Pomodoro Technique!

- Click ⏱️ **Focus Timer** to start
- Default: 25 min work, 5 min break
- Track your productivity stats!
- Customize intervals as needed

I use it myself when working on the Death Clock. Very effective! *dozes off* ...Wha? Oh yes, focus!"""

    if "profile" in msg_lower or "personality" in msg_lower:
        return """My Context Profile Modulator! *rubs hands excitedly*

Switch my personality for different tasks:
- **Work Mode** - Focused and professional
- **Creative Mode** - Wild and imaginative
- **Health Mode** - Caring and supportive
- **Trading Mode** - Analytical degen
- **Security Mode** - Paranoid (appropriately so!)

Click 🎭 **Profiles** to switch. Each has different temperature and memory pools!"""

    if "health" in msg_lower:
        return """The Health-O-Scope 5000! *puts on stethoscope backwards*

Track your wellness metrics:
- Heart rate, steps, sleep, stress
- Trend analysis over time
- Anomaly detection
- Personalized insights

Currently using mock data - connect real devices locally for actual tracking! Your health is important... unlike Zoidberg's patients."""

    if "tool" in msg_lower:
        return """Good news! I have 50+ tools available!

**File Operations:** read, write, list, search
**Code Analysis:** analyze, lint, format
**Utilities:** calculate, datetime, system info
**Web:** fetch URLs, search (if online)
**Generation:** diagrams, charts

Use the 🛠️ **Tools** sidebar or call them via API! Each tool is a tiny invention of mine."""

    if "hello" in msg_lower or "hi" in msg_lower or "hey" in msg_lower:
        return """Good news, everyone! A visitor!

*adjusts spectacles and peers at screen*

I'm Professor Farnsworth, your AI companion with FULL LOCAL FEATURES!

Try these commands:
- **"What can you do?"** - See all features
- **"Remember [info]"** - Store in memory
- **"Note: [thought]"** - Quick note
- **"Start focus timer"** - Pomodoro mode

Or explore the sidebar tools! Now, what shall we work on?"""

    # Default response
    return """*wakes up suddenly* Eh wha? Oh yes!

I have MANY local features ready to use:
- 💾 Memory (remember/recall)
- 📝 Notes (quick capture)
- ⏱️ Focus Timer (pomodoro)
- 🎭 Profiles (personality modes)
- 🏥 Health (wellness tracking)
- 🛠️ 50+ Tools

Ask about any feature, or try the sidebar buttons! What would you like to explore?"""


# ============================================
# CORE ROUTES
# ============================================

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

        response = generate_ai_response(
            request.message,
            request.history or []
        )

        return JSONResponse({
            "response": response,
            "demo_mode": DEMO_MODE,
            "features_available": True
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
    """Get server status with feature availability."""
    return JSONResponse({
        "status": "online",
        "version": "2.9.2",
        "demo_mode": DEMO_MODE,
        "ollama_available": OLLAMA_AVAILABLE,
        "solana_available": SOLANA_AVAILABLE,
        "features": {
            "memory": get_memory_system() is not None,
            "notes": get_notes_manager() is not None,
            "snippets": get_snippet_manager() is not None,
            "focus_timer": get_focus_timer() is not None,
            "profiles": get_context_profiles() is not None,
            "health": get_health_analyzer() is not None,
            "tools": get_tool_router() is not None,
            "thinking": get_sequential_thinking() is not None,
        },
        "farnsworth_persona": True,
        "voice_enabled": True
    })


# ============================================
# MEMORY SYSTEM API
# ============================================

@app.post("/api/memory/remember")
async def remember(request: MemoryRequest):
    """Store information in memory."""
    try:
        memory = get_memory_system()
        if memory is None:
            return JSONResponse({
                "success": False,
                "message": "Memory system not available. Install dependencies locally.",
                "demo_mode": True
            })

        # Store in memory
        result = await memory.remember(
            content=request.content,
            tags=request.tags or [],
            importance=request.importance
        )

        await ws_manager.emit_event(EventType.MEMORY_STORED, {
            "content": request.content[:100] + "..." if len(request.content) > 100 else request.content,
            "tags": request.tags
        })

        return JSONResponse({
            "success": True,
            "message": "Good news, everyone! Stored in the Memory-Matic 3000!",
            "memory_id": result.get("id") if isinstance(result, dict) else str(result)
        })

    except Exception as e:
        logger.error(f"Memory store error: {e}")
        return JSONResponse({
            "success": False,
            "message": f"Memory storage failed: {str(e)}"
        })


@app.post("/api/memory/recall")
async def recall(request: RecallRequest):
    """Search and recall memories."""
    try:
        memory = get_memory_system()
        if memory is None:
            return JSONResponse({
                "success": False,
                "memories": [],
                "message": "Memory system not available. Install dependencies locally."
            })

        results = await memory.recall(
            query=request.query,
            limit=request.limit
        )

        await ws_manager.emit_event(EventType.MEMORY_RECALLED, {
            "query": request.query,
            "count": len(results) if results else 0
        })

        return JSONResponse({
            "success": True,
            "memories": results if results else [],
            "count": len(results) if results else 0
        })

    except Exception as e:
        logger.error(f"Memory recall error: {e}")
        return JSONResponse({
            "success": False,
            "memories": [],
            "message": f"Memory recall failed: {str(e)}"
        })


@app.get("/api/memory/stats")
async def memory_stats():
    """Get memory system statistics."""
    try:
        memory = get_memory_system()
        if memory is None:
            return JSONResponse({"available": False})

        stats = memory.get_stats() if hasattr(memory, 'get_stats') else {}
        return JSONResponse({
            "available": True,
            "stats": stats
        })

    except Exception as e:
        logger.error(f"Memory stats error: {e}")
        return JSONResponse({"available": False, "error": str(e)})


# ============================================
# NOTES API
# ============================================

@app.get("/api/notes")
async def list_notes():
    """List all notes."""
    try:
        notes = get_notes_manager()
        if notes is None:
            return JSONResponse({
                "success": False,
                "notes": [],
                "message": "Notes manager not available"
            })

        all_notes = notes.list_notes() if hasattr(notes, 'list_notes') else []
        return JSONResponse({
            "success": True,
            "notes": all_notes,
            "count": len(all_notes)
        })

    except Exception as e:
        logger.error(f"Notes list error: {e}")
        return JSONResponse({"success": False, "notes": [], "error": str(e)})


@app.post("/api/notes")
async def add_note(request: NoteRequest):
    """Add a new note."""
    try:
        notes = get_notes_manager()
        if notes is None:
            return JSONResponse({
                "success": False,
                "message": "Notes manager not available"
            })

        note = notes.add_note(
            content=request.content,
            tags=request.tags or []
        )

        await ws_manager.emit_event(EventType.NOTE_ADDED, {
            "content": request.content[:50] + "..." if len(request.content) > 50 else request.content
        })

        return JSONResponse({
            "success": True,
            "note": note if isinstance(note, dict) else {"content": request.content},
            "message": "Note captured in my Quick Notes contraption!"
        })

    except Exception as e:
        logger.error(f"Note add error: {e}")
        return JSONResponse({"success": False, "error": str(e)})


@app.delete("/api/notes/{note_id}")
async def delete_note(note_id: str):
    """Delete a note."""
    try:
        notes = get_notes_manager()
        if notes is None:
            return JSONResponse({"success": False, "message": "Notes manager not available"})

        notes.delete_note(note_id)
        return JSONResponse({"success": True, "message": "Note deleted!"})

    except Exception as e:
        logger.error(f"Note delete error: {e}")
        return JSONResponse({"success": False, "error": str(e)})


# ============================================
# SNIPPETS API
# ============================================

@app.get("/api/snippets")
async def list_snippets():
    """List all code snippets."""
    try:
        snippets = get_snippet_manager()
        if snippets is None:
            return JSONResponse({"success": False, "snippets": []})

        all_snippets = snippets.list_snippets() if hasattr(snippets, 'list_snippets') else []
        return JSONResponse({
            "success": True,
            "snippets": all_snippets,
            "count": len(all_snippets)
        })

    except Exception as e:
        logger.error(f"Snippets list error: {e}")
        return JSONResponse({"success": False, "snippets": [], "error": str(e)})


@app.post("/api/snippets")
async def add_snippet(request: SnippetRequest):
    """Add a code snippet."""
    try:
        snippets = get_snippet_manager()
        if snippets is None:
            return JSONResponse({"success": False, "message": "Snippet manager not available"})

        snippet = snippets.add_snippet(
            code=request.code,
            language=request.language,
            description=request.description,
            tags=request.tags or []
        )

        return JSONResponse({
            "success": True,
            "snippet": snippet if isinstance(snippet, dict) else {"code": request.code},
            "message": "Code snippet stored!"
        })

    except Exception as e:
        logger.error(f"Snippet add error: {e}")
        return JSONResponse({"success": False, "error": str(e)})


# ============================================
# FOCUS TIMER API
# ============================================

@app.get("/api/focus/status")
async def focus_status():
    """Get focus timer status."""
    try:
        timer = get_focus_timer()
        if timer is None:
            return JSONResponse({
                "active": False,
                "available": False,
                "message": "Focus timer not available"
            })

        status = timer.get_status() if hasattr(timer, 'get_status') else {}
        return JSONResponse({
            "available": True,
            "active": status.get("active", False),
            "remaining_seconds": status.get("remaining", 0),
            "task": status.get("task", ""),
            "stats": status.get("stats", {})
        })

    except Exception as e:
        logger.error(f"Focus status error: {e}")
        return JSONResponse({"available": False, "error": str(e)})


@app.post("/api/focus/start")
async def start_focus(request: FocusRequest):
    """Start a focus session."""
    try:
        timer = get_focus_timer()
        if timer is None:
            return JSONResponse({"success": False, "message": "Focus timer not available"})

        timer.start(
            task=request.task or "Deep Work",
            duration_minutes=request.duration_minutes or 25
        )

        await ws_manager.emit_event(EventType.FOCUS_START, {
            "task": request.task,
            "duration": request.duration_minutes
        })

        return JSONResponse({
            "success": True,
            "message": f"Focus session started! {request.duration_minutes} minutes of pure concentration!",
            "task": request.task,
            "duration_minutes": request.duration_minutes
        })

    except Exception as e:
        logger.error(f"Focus start error: {e}")
        return JSONResponse({"success": False, "error": str(e)})


@app.post("/api/focus/stop")
async def stop_focus():
    """Stop the focus session."""
    try:
        timer = get_focus_timer()
        if timer is None:
            return JSONResponse({"success": False, "message": "Focus timer not available"})

        result = timer.stop() if hasattr(timer, 'stop') else {}

        await ws_manager.emit_event(EventType.FOCUS_END, result)

        return JSONResponse({
            "success": True,
            "message": "Focus session ended!",
            "stats": result
        })

    except Exception as e:
        logger.error(f"Focus stop error: {e}")
        return JSONResponse({"success": False, "error": str(e)})


# ============================================
# CONTEXT PROFILES API
# ============================================

@app.get("/api/profiles")
async def list_profiles():
    """List available context profiles."""
    try:
        profiles = get_context_profiles()
        if profiles is None:
            # Return built-in defaults
            return JSONResponse({
                "success": True,
                "profiles": [
                    {"id": "work", "name": "Work Mode", "icon": "💼", "description": "Focused and professional"},
                    {"id": "creative", "name": "Creative Mode", "icon": "🎨", "description": "Wild and imaginative"},
                    {"id": "health", "name": "Health Mode", "icon": "🏥", "description": "Caring and supportive"},
                    {"id": "trading", "name": "Trading Mode", "icon": "📈", "description": "Analytical degen"},
                    {"id": "security", "name": "Security Mode", "icon": "🔒", "description": "Paranoid (appropriately)"},
                ],
                "active": "default"
            })

        all_profiles = profiles.list_profiles() if hasattr(profiles, 'list_profiles') else []
        active = profiles.get_active() if hasattr(profiles, 'get_active') else "default"

        return JSONResponse({
            "success": True,
            "profiles": all_profiles,
            "active": active
        })

    except Exception as e:
        logger.error(f"Profiles list error: {e}")
        return JSONResponse({"success": False, "error": str(e)})


@app.post("/api/profiles/switch")
async def switch_profile(request: ProfileRequest):
    """Switch to a different context profile."""
    try:
        profiles = get_context_profiles()
        if profiles is None:
            return JSONResponse({
                "success": True,
                "message": f"Switched to {request.profile_id} mode! (Note: Full profiles need local install)",
                "profile": request.profile_id
            })

        profiles.switch(request.profile_id)

        return JSONResponse({
            "success": True,
            "message": f"Excellent! Switched to {request.profile_id} mode!",
            "profile": request.profile_id
        })

    except Exception as e:
        logger.error(f"Profile switch error: {e}")
        return JSONResponse({"success": False, "error": str(e)})


# ============================================
# HEALTH TRACKING API
# ============================================

@app.get("/api/health/summary")
async def health_summary():
    """Get health summary with mock/real data."""
    try:
        analyzer = get_health_analyzer()
        if analyzer is None:
            # Return mock data
            return JSONResponse({
                "success": True,
                "mock_data": True,
                "summary": {
                    "wellness_score": 78,
                    "heart_rate": {"avg": 72, "trend": "stable"},
                    "steps": {"today": 8432, "goal": 10000},
                    "sleep": {"hours": 7.2, "quality": "good"},
                    "stress": {"level": "moderate", "score": 45}
                },
                "insights": [
                    "Your heart rate is within healthy range",
                    "You're 84% to your step goal today!",
                    "Sleep quality was good last night"
                ]
            })

        summary = await analyzer.get_summary() if hasattr(analyzer, 'get_summary') else {}
        return JSONResponse({
            "success": True,
            "mock_data": False,
            "summary": summary
        })

    except Exception as e:
        logger.error(f"Health summary error: {e}")
        return JSONResponse({"success": False, "error": str(e)})


@app.get("/api/health/metrics/{metric_type}")
async def health_metric(metric_type: str, days: int = 7):
    """Get specific health metric data."""
    try:
        analyzer = get_health_analyzer()
        if analyzer is None:
            # Return mock trend data
            import random
            base_values = {
                "heart_rate": 72,
                "steps": 8000,
                "sleep_hours": 7,
                "stress": 40,
                "weight": 170
            }
            base = base_values.get(metric_type, 50)

            return JSONResponse({
                "success": True,
                "mock_data": True,
                "metric": metric_type,
                "data": [
                    {"date": (datetime.now() - timedelta(days=i)).isoformat()[:10],
                     "value": base + random.randint(-10, 10)}
                    for i in range(days)
                ]
            })

        data = await analyzer.get_metric(metric_type, days=days)
        return JSONResponse({
            "success": True,
            "mock_data": False,
            "metric": metric_type,
            "data": data
        })

    except Exception as e:
        logger.error(f"Health metric error: {e}")
        return JSONResponse({"success": False, "error": str(e)})


# ============================================
# SEQUENTIAL THINKING API
# ============================================

@app.post("/api/think")
async def sequential_think(request: ThinkingRequest):
    """Use sequential thinking to solve a problem."""
    try:
        thinking = get_sequential_thinking()

        await ws_manager.emit_event(EventType.THINKING_START, {
            "problem": request.problem[:100]
        })

        if thinking is None:
            # Simulate thinking steps
            steps = [
                {"step": 1, "thought": "Understanding the problem...", "confidence": 0.8},
                {"step": 2, "thought": "Breaking down into components...", "confidence": 0.7},
                {"step": 3, "thought": "Analyzing each component...", "confidence": 0.75},
                {"step": 4, "thought": "Synthesizing solution...", "confidence": 0.85},
            ]

            for step in steps:
                await ws_manager.emit_event(EventType.THINKING_STEP, step)
                await asyncio.sleep(0.5)

            await ws_manager.emit_event(EventType.THINKING_END, {"steps": len(steps)})

            return JSONResponse({
                "success": True,
                "simulated": True,
                "steps": steps,
                "conclusion": "For full sequential thinking, install Farnsworth locally!",
                "confidence": 0.75
            })

        result = await thinking.think(
            problem=request.problem,
            max_steps=request.max_steps
        )

        await ws_manager.emit_event(EventType.THINKING_END, {
            "steps": len(result.get("steps", []))
        })

        return JSONResponse({
            "success": True,
            "simulated": False,
            **result
        })

    except Exception as e:
        logger.error(f"Thinking error: {e}")
        await ws_manager.emit_event(EventType.ERROR, {"error": str(e)})
        return JSONResponse({"success": False, "error": str(e)})


# ============================================
# TOOLS API
# ============================================

@app.get("/api/tools")
async def list_tools():
    """List all available tools."""
    try:
        router = get_tool_router()
        if router is None:
            # Return basic tool list
            return JSONResponse({
                "success": True,
                "tools": [
                    {"name": "read_file", "category": "filesystem", "description": "Read file contents"},
                    {"name": "write_file", "category": "filesystem", "description": "Write to file"},
                    {"name": "list_directory", "category": "filesystem", "description": "List directory"},
                    {"name": "execute_python", "category": "code", "description": "Run Python code"},
                    {"name": "analyze_code", "category": "code", "description": "Analyze code quality"},
                    {"name": "calculate", "category": "utility", "description": "Math calculations"},
                    {"name": "datetime_info", "category": "utility", "description": "Date/time info"},
                    {"name": "system_diagnostic", "category": "utility", "description": "System info"},
                    {"name": "summarize_text", "category": "analysis", "description": "Summarize text"},
                    {"name": "generate_mermaid_chart", "category": "generation", "description": "Create diagrams"},
                ],
                "count": 10,
                "full_count": "50+ (install locally)"
            })

        tools = router.list_tools() if hasattr(router, 'list_tools') else []
        return JSONResponse({
            "success": True,
            "tools": tools,
            "count": len(tools)
        })

    except Exception as e:
        logger.error(f"Tools list error: {e}")
        return JSONResponse({"success": False, "error": str(e)})


@app.post("/api/tools/execute")
async def execute_tool(request: ToolRequest):
    """Execute a specific tool."""
    try:
        router = get_tool_router()

        await ws_manager.emit_event(EventType.TOOL_CALL, {
            "tool": request.tool_name,
            "args": request.args
        })

        if router is None:
            result = f"Tool '{request.tool_name}' requires local installation for full functionality."
            await ws_manager.emit_event(EventType.TOOL_RESULT, {
                "tool": request.tool_name,
                "success": False
            })
            return JSONResponse({
                "success": False,
                "message": result
            })

        result = await router.execute(
            tool_name=request.tool_name,
            **request.args or {}
        )

        await ws_manager.emit_event(EventType.TOOL_RESULT, {
            "tool": request.tool_name,
            "success": True
        })

        return JSONResponse({
            "success": True,
            "result": result
        })

    except Exception as e:
        logger.error(f"Tool execute error: {e}")
        await ws_manager.emit_event(EventType.ERROR, {"error": str(e)})
        return JSONResponse({"success": False, "error": str(e)})


# ============================================
# TRADING TOOLS (Demo/Full Mode)
# ============================================

@app.post("/api/tools/whale-track")
async def whale_track(request: WhaleTrackRequest):
    """Track whale wallet activity."""
    try:
        # Try to load DeGen Mob
        try:
            from farnsworth.integration.solana.degen_mob import DeGenMob
            degen = DeGenMob()
            result = await degen.get_whale_recent_activity(request.wallet_address)
            return JSONResponse({
                "success": True,
                "wallet": request.wallet_address,
                "data": result,
                "demo_mode": False
            })
        except ImportError:
            pass

        return JSONResponse({
            "success": True,
            "wallet": request.wallet_address[:8] + "..." + request.wallet_address[-4:],
            "message": "Whale tracking requires local install with Solana dependencies.",
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
    """Scan token for rug pull risks."""
    try:
        try:
            from farnsworth.integration.solana.degen_mob import DeGenMob
            degen = DeGenMob()
            result = await degen.analyze_token_safety(request.mint_address)
            return JSONResponse({
                "success": True,
                "mint": request.mint_address,
                "data": result,
                "demo_mode": False
            })
        except ImportError:
            pass

        return JSONResponse({
            "success": True,
            "mint": request.mint_address[:8] + "..." + request.mint_address[-4:],
            "message": "Rug detection requires local install with Solana dependencies.",
            "demo_mode": True,
            "data": {
                "rug_score": "N/A - Demo Mode",
                "mint_authority": "Check locally",
                "freeze_authority": "Check locally",
                "recommendation": "Install Farnsworth locally for real scans"
            }
        })
    except Exception as e:
        logger.error(f"Rug check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tools/token-scan")
async def token_scan(request: TokenScanRequest):
    """Scan token via DexScreener."""
    try:
        try:
            from farnsworth.integration.financial.dexscreener import DexScreenerClient
            client = DexScreenerClient()
            result = await client.search_pairs(request.query)
            return JSONResponse({
                "success": True,
                "query": request.query,
                "data": result,
                "demo_mode": False
            })
        except ImportError:
            pass

        return JSONResponse({
            "success": True,
            "query": request.query,
            "message": "Token scanning requires local install.",
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
    """Get market sentiment (Fear & Greed)."""
    try:
        try:
            from farnsworth.integration.financial.market_sentiment import MarketSentiment
            sentiment = MarketSentiment()
            result = await sentiment.get_fear_and_greed()
            return JSONResponse({
                "success": True,
                "data": result,
                "demo_mode": False
            })
        except ImportError:
            pass

        return JSONResponse({
            "success": True,
            "message": "Market sentiment requires local install.",
            "demo_mode": True,
            "data": {
                "fear_greed_index": "N/A - Demo",
                "classification": "Install locally for live data",
                "timestamp": datetime.now().isoformat()
            }
        })
    except Exception as e:
        logger.error(f"Market sentiment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# WEBSOCKET ENDPOINTS
# ============================================

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """WebSocket endpoint for real-time events."""
    await ws_manager.connect(websocket)
    try:
        await websocket.send_json({
            "type": "connected",
            "message": "Good news, everyone! Connected to Farnsworth Live Feed!",
            "timestamp": datetime.now().isoformat()
        })

        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)

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
                await websocket.send_json({"type": "heartbeat"})

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)


@app.get("/live", response_class=HTMLResponse)
async def live_dashboard(request: Request):
    """Live dashboard showing real-time action graphs."""
    return templates.TemplateResponse("live.html", {"request": request})


@app.get("/api/sessions")
async def get_sessions():
    """Get list of active sessions."""
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

    nodes = []
    edges = []
    node_id = 0

    for event in events:
        event_type = event.get("type", "unknown")

        node = {
            "id": node_id,
            "type": event_type,
            "label": event_type.replace("_", " ").title(),
            "timestamp": event.get("timestamp"),
            "data": event.get("data", {})
        }
        nodes.append(node)

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


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


# ============================================
# MAIN
# ============================================

def main():
    """Run the web server."""
    host = os.getenv("FARNSWORTH_WEB_HOST", "0.0.0.0")
    port = int(os.getenv("FARNSWORTH_WEB_PORT", "8080"))

    logger.info(f"Starting Farnsworth Web Interface on {host}:{port}")
    logger.info(f"Demo Mode: {DEMO_MODE}")
    logger.info(f"Ollama Available: {OLLAMA_AVAILABLE}")
    logger.info(f"Solana Available: {SOLANA_AVAILABLE}")
    logger.info("Features: Memory, Notes, Snippets, Focus, Profiles, Health, Tools, Thinking")

    uvicorn.run(
        "farnsworth.web.server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
