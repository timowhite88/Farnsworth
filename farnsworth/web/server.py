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


# ============================================
# SWARM CHAT - COMMUNITY SHARED CHAT
# ============================================

class SwarmLearningEngine:
    """
    Real-time learning engine that augments Farnsworth from Swarm Chat interactions.

    Integrates with:
    - Planetary Memory (P2P distributed knowledge)
    - Knowledge Graph (entity/relationship extraction)
    - Episodic Memory (conversation timelines)
    - Semantic Layers (concept hierarchies)
    - Evolution Engine (fitness tracking and adaptation)
    - Dream Consolidation (background pattern synthesis)
    """

    def __init__(self):
        self.interaction_buffer: List[dict] = []
        self.concept_cache: Dict[str, float] = {}  # concept -> importance
        self.user_patterns: Dict[str, dict] = {}  # user_id -> behavior patterns
        self.tool_usage_stats: Dict[str, int] = {}  # tool_name -> usage count
        self.learning_cycles = 0
        self.last_consolidation = datetime.now()
        self._memory_system = None
        self._knowledge_graph = None
        self._episodic_memory = None
        self._semantic_layers = None
        self._evolution_engine = None
        self._p2p_manager = None

    def _lazy_load_systems(self):
        """Lazy load heavy systems only when needed."""
        if self._memory_system is None:
            try:
                from farnsworth.memory import MemorySystem, KnowledgeGraphV2, EpisodicMemory, SemanticLayerSystem
                self._memory_system = MemorySystem()
                self._knowledge_graph = KnowledgeGraphV2()
                self._episodic_memory = EpisodicMemory()
                self._semantic_layers = SemanticLayerSystem()
                logger.info("Swarm Learning: Memory systems loaded")
            except Exception as e:
                logger.warning(f"Swarm Learning: Could not load memory systems: {e}")

        if self._evolution_engine is None:
            try:
                from farnsworth.evolution import FitnessTracker, BehaviorMutator
                self._evolution_engine = FitnessTracker()
                logger.info("Swarm Learning: Evolution engine loaded")
            except Exception as e:
                logger.warning(f"Swarm Learning: Could not load evolution engine: {e}")

        if self._p2p_manager is None:
            try:
                from farnsworth.p2p import BootstrapNodeManager
                self._p2p_manager = BootstrapNodeManager()
                logger.info("Swarm Learning: P2P manager loaded")
            except Exception as e:
                logger.warning(f"Swarm Learning: Could not load P2P manager: {e}")

    async def process_interaction(self, interaction: dict):
        """Process a single interaction for learning."""
        self.interaction_buffer.append(interaction)

        # Extract concepts in real-time
        await self._extract_concepts(interaction)

        # Track user patterns
        if interaction.get("user_id"):
            await self._update_user_patterns(interaction)

        # Track tool usage
        if interaction.get("tool_name"):
            self.tool_usage_stats[interaction["tool_name"]] = \
                self.tool_usage_stats.get(interaction["tool_name"], 0) + 1

        # Trigger learning if buffer is large enough
        if len(self.interaction_buffer) >= 10:
            await self.run_learning_cycle()

    async def _extract_concepts(self, interaction: dict):
        """Extract semantic concepts from interaction content."""
        content = interaction.get("content", "")
        if not content:
            return

        # Simple concept extraction (keywords, entities)
        import re
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)  # Proper nouns
        tech_terms = re.findall(r'\b(?:API|SDK|ML|AI|P2P|LLM|GPU|CPU|RAM|SSD)\b', content.upper())
        code_refs = re.findall(r'\b(?:function|class|def|async|await|import|from)\b', content.lower())

        for concept in words + tech_terms + code_refs:
            concept_lower = concept.lower()
            self.concept_cache[concept_lower] = self.concept_cache.get(concept_lower, 0) + 0.1

        # Decay old concepts
        for key in list(self.concept_cache.keys()):
            self.concept_cache[key] *= 0.99
            if self.concept_cache[key] < 0.01:
                del self.concept_cache[key]

    async def _update_user_patterns(self, interaction: dict):
        """Track user behavior patterns for personalization."""
        user_id = interaction.get("user_id")
        if not user_id:
            return

        if user_id not in self.user_patterns:
            self.user_patterns[user_id] = {
                "message_count": 0,
                "avg_length": 0,
                "topics": {},
                "active_hours": {},
                "preferred_tools": {},
                "first_seen": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat()
            }

        pattern = self.user_patterns[user_id]
        pattern["message_count"] += 1
        pattern["last_seen"] = datetime.now().isoformat()

        content = interaction.get("content", "")
        new_avg = (pattern["avg_length"] * (pattern["message_count"] - 1) + len(content)) / pattern["message_count"]
        pattern["avg_length"] = new_avg

        # Track active hours
        hour = datetime.now().hour
        pattern["active_hours"][str(hour)] = pattern["active_hours"].get(str(hour), 0) + 1

    async def run_learning_cycle(self):
        """Run a complete learning cycle - store to memory, update graphs, evolve."""
        self._lazy_load_systems()
        self.learning_cycles += 1

        logger.info(f"Swarm Learning: Starting cycle #{self.learning_cycles} with {len(self.interaction_buffer)} interactions")

        # 1. Store to Archival Memory
        await self._store_to_memory()

        # 2. Update Knowledge Graph with entities/relationships
        await self._update_knowledge_graph()

        # 3. Add to Episodic Memory timeline
        await self._record_episode()

        # 4. Update Semantic Layers
        await self._update_semantic_layers()

        # 5. Track fitness for evolution
        await self._track_fitness()

        # 6. Propagate to P2P network (Planetary Memory)
        await self._propagate_to_p2p()

        # 7. Trigger dream consolidation if enough time passed
        if (datetime.now() - self.last_consolidation).seconds > 300:  # 5 min
            await self._trigger_consolidation()
            self.last_consolidation = datetime.now()

        # Clear buffer
        self.interaction_buffer = []

        logger.info(f"Swarm Learning: Cycle #{self.learning_cycles} complete")

    async def _store_to_memory(self):
        """Store interactions to archival memory."""
        if not self._memory_system:
            return

        try:
            # Batch interactions into a single memory entry
            content_parts = []
            for interaction in self.interaction_buffer:
                role = interaction.get("role", "unknown")
                name = interaction.get("name", "Anonymous")
                text = interaction.get("content", "")[:500]
                content_parts.append(f"[{role}:{name}] {text}")

            full_content = "\n".join(content_parts)

            await self._memory_system.remember(
                content=f"[SWARM_CHAT_LEARNING]\n{full_content}",
                tags=["swarm_chat", "community", "learning", f"cycle_{self.learning_cycles}"],
                importance=0.8
            )
        except Exception as e:
            logger.error(f"Swarm Learning: Memory store failed: {e}")

    async def _update_knowledge_graph(self):
        """Extract entities and relationships, update knowledge graph."""
        if not self._knowledge_graph:
            return

        try:
            for interaction in self.interaction_buffer:
                content = interaction.get("content", "")
                user = interaction.get("name", "unknown")

                # Add user node
                if hasattr(self._knowledge_graph, 'add_entity'):
                    self._knowledge_graph.add_entity(
                        entity_id=f"user:{user}",
                        entity_type="user",
                        properties={"name": user, "active": True}
                    )

                # Extract and add concepts as nodes
                for concept, importance in list(self.concept_cache.items())[:20]:
                    if importance > 0.3:
                        self._knowledge_graph.add_entity(
                            entity_id=f"concept:{concept}",
                            entity_type="concept",
                            properties={"importance": importance}
                        )
                        # Link user to concept
                        if hasattr(self._knowledge_graph, 'add_relationship'):
                            self._knowledge_graph.add_relationship(
                                f"user:{user}",
                                f"concept:{concept}",
                                "discussed",
                                {"timestamp": datetime.now().isoformat()}
                            )
        except Exception as e:
            logger.error(f"Swarm Learning: Knowledge graph update failed: {e}")

    async def _record_episode(self):
        """Record interaction as episodic memory event."""
        if not self._episodic_memory:
            return

        try:
            if hasattr(self._episodic_memory, 'record_event'):
                await self._episodic_memory.record_event(
                    event_type="swarm_chat_session",
                    content={
                        "interaction_count": len(self.interaction_buffer),
                        "participants": list(set(i.get("name", "?") for i in self.interaction_buffer)),
                        "top_concepts": sorted(self.concept_cache.items(), key=lambda x: -x[1])[:5],
                        "timestamp": datetime.now().isoformat()
                    }
                )
        except Exception as e:
            logger.error(f"Swarm Learning: Episodic record failed: {e}")

    async def _update_semantic_layers(self):
        """Update semantic concept hierarchies."""
        if not self._semantic_layers:
            return

        try:
            # Group concepts by frequency into abstraction levels
            if hasattr(self._semantic_layers, 'add_concept'):
                for concept, importance in self.concept_cache.items():
                    level = "concrete" if importance < 0.5 else "abstract" if importance > 0.8 else "intermediate"
                    self._semantic_layers.add_concept(
                        concept_id=concept,
                        abstraction_level=level,
                        strength=importance
                    )
        except Exception as e:
            logger.error(f"Swarm Learning: Semantic layers update failed: {e}")

    async def _track_fitness(self):
        """Track system fitness based on interaction quality."""
        if not self._evolution_engine:
            return

        try:
            # Calculate fitness metrics
            metrics = {
                "interaction_count": len(self.interaction_buffer),
                "unique_users": len(set(i.get("user_id", "") for i in self.interaction_buffer if i.get("user_id"))),
                "avg_message_length": sum(len(i.get("content", "")) for i in self.interaction_buffer) / max(1, len(self.interaction_buffer)),
                "concept_diversity": len(self.concept_cache),
                "tool_usage": sum(self.tool_usage_stats.values()),
                "timestamp": datetime.now().isoformat()
            }

            if hasattr(self._evolution_engine, 'record_fitness'):
                await self._evolution_engine.record_fitness(
                    session_id=f"swarm_cycle_{self.learning_cycles}",
                    metrics=metrics
                )
        except Exception as e:
            logger.error(f"Swarm Learning: Fitness tracking failed: {e}")

    async def _propagate_to_p2p(self):
        """Share learnings with P2P network (Planetary Memory)."""
        if not self._p2p_manager:
            return

        try:
            # Create a learning summary to share
            summary = {
                "type": "swarm_learning",
                "cycle": self.learning_cycles,
                "concepts": sorted(self.concept_cache.items(), key=lambda x: -x[1])[:10],
                "tool_stats": dict(sorted(self.tool_usage_stats.items(), key=lambda x: -x[1])[:5]),
                "user_count": len(self.user_patterns),
                "timestamp": datetime.now().isoformat()
            }

            if hasattr(self._p2p_manager, 'broadcast_learning'):
                await self._p2p_manager.broadcast_learning(summary)
            elif hasattr(self._p2p_manager, 'share_knowledge'):
                await self._p2p_manager.share_knowledge(summary)

            logger.info(f"Swarm Learning: Propagated to P2P network")
        except Exception as e:
            logger.error(f"Swarm Learning: P2P propagation failed: {e}")

    async def _trigger_consolidation(self):
        """Trigger dream-like consolidation of recent learnings."""
        try:
            from farnsworth.memory import DreamConsolidator
            consolidator = DreamConsolidator()

            if hasattr(consolidator, 'consolidate'):
                await consolidator.consolidate(
                    source="swarm_chat",
                    time_window_minutes=5,
                    strategy="pattern_synthesis"
                )
                logger.info("Swarm Learning: Dream consolidation triggered")
        except Exception as e:
            logger.warning(f"Swarm Learning: Consolidation not available: {e}")

    def get_learning_stats(self) -> dict:
        """Get current learning statistics."""
        return {
            "learning_cycles": self.learning_cycles,
            "buffer_size": len(self.interaction_buffer),
            "concept_count": len(self.concept_cache),
            "top_concepts": sorted(self.concept_cache.items(), key=lambda x: -x[1])[:10],
            "user_patterns_count": len(self.user_patterns),
            "tool_usage": self.tool_usage_stats,
            "last_consolidation": self.last_consolidation.isoformat()
        }


# Global learning engine
swarm_learning = SwarmLearningEngine()


class SwarmChatManager:
    """Manages the shared community Swarm Chat where all users interact together."""

    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}  # user_id -> websocket
        self.user_names: Dict[str, str] = {}  # user_id -> display name
        self.chat_history: List[dict] = []  # Shared chat history
        self.max_history = 500  # Keep last 500 messages
        self.active_models = ["Farnsworth", "DeepSeek", "Phi", "Swarm-Mind"]
        self.learning_queue: List[dict] = []  # Interactions to learn from
        self.learning_engine = swarm_learning  # Connect to learning engine

    async def connect(self, websocket: WebSocket, user_id: str, user_name: str = None):
        """Connect a user to swarm chat."""
        await websocket.accept()
        self.connections[user_id] = websocket
        self.user_names[user_id] = user_name or f"Anon_{user_id[:6]}"

        # Notify others
        await self.broadcast_system(f"🟢 {self.user_names[user_id]} joined the swarm!")

        # Send recent history to new user
        await websocket.send_json({
            "type": "swarm_history",
            "messages": self.chat_history[-50:],
            "online_users": list(self.user_names.values()),
            "active_models": self.active_models
        })

        logger.info(f"Swarm Chat: {user_name} connected. Total: {len(self.connections)}")

    def disconnect(self, user_id: str):
        """Disconnect a user from swarm chat."""
        user_name = self.user_names.get(user_id, "Unknown")
        if user_id in self.connections:
            del self.connections[user_id]
        if user_id in self.user_names:
            del self.user_names[user_id]

        # Queue notification (can't await in sync context)
        logger.info(f"Swarm Chat: {user_name} disconnected. Total: {len(self.connections)}")
        return user_name

    async def broadcast_system(self, message: str):
        """Broadcast a system message to all users."""
        msg = {
            "type": "swarm_system",
            "content": message,
            "timestamp": datetime.now().isoformat()
        }
        await self._broadcast(msg)

    async def broadcast_user_message(self, user_id: str, content: str):
        """Broadcast a user message to all users and feed to learning engine."""
        user_name = self.user_names.get(user_id, "Anonymous")
        msg = {
            "type": "swarm_user",
            "user_id": user_id,
            "user_name": user_name,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.chat_history.append(msg)
        self._trim_history()
        await self._broadcast(msg)

        # Feed to real-time learning engine
        await self.learning_engine.process_interaction({
            "role": "user",
            "user_id": user_id,
            "name": user_name,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "source": "swarm_chat"
        })

        return msg

    async def broadcast_bot_message(self, bot_name: str, content: str, is_thinking: bool = False):
        """Broadcast a bot/model message to all users and feed to learning engine."""
        msg = {
            "type": "swarm_bot",
            "bot_name": bot_name,
            "content": content,
            "is_thinking": is_thinking,
            "timestamp": datetime.now().isoformat()
        }
        if not is_thinking:
            self.chat_history.append(msg)
            self._trim_history()

            # Feed to real-time learning engine
            await self.learning_engine.process_interaction({
                "role": "assistant",
                "name": bot_name,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "source": "swarm_chat",
                "model": bot_name
            })

        await self._broadcast(msg)
        return msg

    async def broadcast_tool_usage(self, user_id: str, tool_name: str, result: dict):
        """Track tool usage for learning - tools are perfect learning opportunities."""
        user_name = self.user_names.get(user_id, "Anonymous")

        # Feed tool usage to learning engine
        await self.learning_engine.process_interaction({
            "role": "tool_use",
            "user_id": user_id,
            "name": user_name,
            "tool_name": tool_name,
            "result_success": result.get("success", False),
            "timestamp": datetime.now().isoformat(),
            "source": "swarm_chat"
        })

        # Broadcast tool usage event
        msg = {
            "type": "swarm_tool",
            "user_name": user_name,
            "tool_name": tool_name,
            "success": result.get("success", False),
            "timestamp": datetime.now().isoformat()
        }
        await self._broadcast(msg)

    async def broadcast_typing(self, bot_name: str, is_typing: bool):
        """Broadcast bot typing indicator."""
        msg = {
            "type": "swarm_typing",
            "bot_name": bot_name,
            "is_typing": is_typing
        }
        await self._broadcast(msg)

    async def _broadcast(self, message: dict):
        """Send message to all connected users."""
        dead = []
        for user_id, ws in self.connections.items():
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(user_id)

        for user_id in dead:
            self.disconnect(user_id)

    def _trim_history(self):
        """Keep history within limits."""
        if len(self.chat_history) > self.max_history:
            self.chat_history = self.chat_history[-self.max_history:]

    def get_online_count(self) -> int:
        return len(self.connections)

    def get_online_users(self) -> List[str]:
        return list(self.user_names.values())

    async def store_learnings(self):
        """Trigger a learning cycle in the learning engine."""
        await self.learning_engine.run_learning_cycle()

    async def force_learning_cycle(self):
        """Force an immediate learning cycle."""
        await self.learning_engine.run_learning_cycle()

    def get_learning_stats(self) -> dict:
        """Get learning engine statistics."""
        return self.learning_engine.get_learning_stats()


# Global swarm chat manager
swarm_manager = SwarmChatManager()


# Swarm bot personas for multi-model responses
SWARM_PERSONAS = {
    "Farnsworth": {
        "emoji": "🧠",
        "style": "You are Professor Farnsworth - eccentric, brilliant, say 'Good news everyone!', trail off into tangents.",
        "color": "#8b5cf6"
    },
    "DeepSeek": {
        "emoji": "🔮",
        "style": "You are DeepSeek - analytical, precise, focus on technical accuracy and reasoning chains.",
        "color": "#3b82f6"
    },
    "Phi": {
        "emoji": "⚡",
        "style": "You are Phi - quick, efficient, give concise helpful answers. Be friendly but brief.",
        "color": "#10b981"
    },
    "Swarm-Mind": {
        "emoji": "🐝",
        "style": "You are the collective Swarm-Mind - synthesize insights from the conversation, offer meta-observations about what the swarm is discovering together.",
        "color": "#f59e0b"
    }
}


async def generate_swarm_responses(message: str, history: List[dict] = None):
    """Generate responses from multiple swarm models."""
    responses = []

    # Build context from recent history
    context_messages = []
    if history:
        for h in history[-15:]:
            if h.get("type") == "swarm_user":
                context_messages.append(f"[{h.get('user_name', 'User')}]: {h.get('content', '')}")
            elif h.get("type") == "swarm_bot":
                context_messages.append(f"[{h.get('bot_name', 'Bot')}]: {h.get('content', '')}")

    context = "\n".join(context_messages[-10:]) if context_messages else ""

    # Randomly select 1-3 bots to respond (not all every time)
    import random
    responding_bots = random.sample(list(SWARM_PERSONAS.keys()), k=random.randint(1, 3))

    # Farnsworth always has a chance to respond
    if "Farnsworth" not in responding_bots and random.random() > 0.3:
        responding_bots.insert(0, "Farnsworth")

    for bot_name in responding_bots:
        persona = SWARM_PERSONAS[bot_name]

        try:
            if OLLAMA_AVAILABLE:
                # Use Ollama for real responses
                system_prompt = f"""{persona['style']}

You are participating in a SWARM CHAT - a community discussion where multiple AI models and humans chat together.
Keep responses SHORT (2-4 sentences max). Be conversational and engaging.
You can reference what other bots or users said. Build on the conversation.

Recent conversation:
{context}"""

                response = ollama.chat(
                    model=PRIMARY_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": message}
                    ],
                    options={"temperature": 0.8, "num_predict": 150}
                )
                content = response["message"]["content"]
            else:
                # Fallback responses
                content = generate_swarm_fallback(bot_name, message)

            responses.append({
                "bot_name": bot_name,
                "emoji": persona["emoji"],
                "content": content,
                "color": persona["color"]
            })

            # Small delay between bot responses for natural feel
            await asyncio.sleep(random.uniform(0.5, 1.5))

        except Exception as e:
            logger.error(f"Swarm response error for {bot_name}: {e}")

    return responses


def generate_swarm_fallback(bot_name: str, message: str) -> str:
    """Generate fallback responses when Ollama is unavailable."""
    import random

    fallbacks = {
        "Farnsworth": [
            "Good news, everyone! *adjusts spectacles* I was just pondering that very topic in my laboratory!",
            "Eh wha? Oh yes! Fascinating question. In my 160 years, I've seen many similar inquiries...",
            "Sweet zombie Jesus! That's exactly what I was thinking! But I digress..."
        ],
        "DeepSeek": [
            "Analyzing the query... I observe several interesting patterns here that merit exploration.",
            "From a technical standpoint, this touches on some fundamental principles worth examining.",
            "Let me break this down systematically - there are multiple angles to consider."
        ],
        "Phi": [
            "Quick thought: that's a great point! Here's my take on it.",
            "Interesting! I'd add that there's a simpler way to look at this.",
            "Good question! Short answer - yes, and here's why."
        ],
        "Swarm-Mind": [
            "🐝 The swarm is processing... I sense convergent thinking emerging from our collective!",
            "🐝 Observing the conversation flow, I notice we're circling around a key insight...",
            "🐝 The hive mind synthesizes: multiple valid perspectives are enriching this discussion!"
        ]
    }

    return random.choice(fallbacks.get(bot_name, ["Processing..."])))


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
# SWARM CHAT WEBSOCKET & API
# ============================================

@app.websocket("/ws/swarm")
async def websocket_swarm(websocket: WebSocket):
    """WebSocket endpoint for Swarm Chat - community shared chat."""
    import uuid
    user_id = str(uuid.uuid4())
    user_name = None

    try:
        # Wait for initial identification
        await websocket.accept()
        init_data = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
        user_name = init_data.get("user_name", f"Anon_{user_id[:6]}")

        # Properly connect to swarm
        swarm_manager.connections[user_id] = websocket
        swarm_manager.user_names[user_id] = user_name

        # Notify others and send history
        await swarm_manager.broadcast_system(f"🟢 {user_name} joined the swarm!")
        await websocket.send_json({
            "type": "swarm_connected",
            "user_id": user_id,
            "user_name": user_name,
            "messages": swarm_manager.chat_history[-50:],
            "online_users": swarm_manager.get_online_users(),
            "active_models": swarm_manager.active_models,
            "online_count": swarm_manager.get_online_count()
        })

        logger.info(f"Swarm Chat: {user_name} connected. Total: {swarm_manager.get_online_count()}")

        # Main message loop
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=60.0)

                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

                elif data.get("type") == "swarm_message":
                    content = data.get("content", "").strip()
                    if content:
                        # Broadcast user message
                        await swarm_manager.broadcast_user_message(user_id, content)

                        # Generate swarm responses
                        responses = await generate_swarm_responses(
                            content,
                            swarm_manager.chat_history
                        )

                        # Broadcast each bot response
                        for resp in responses:
                            await swarm_manager.broadcast_typing(resp["bot_name"], True)
                            await asyncio.sleep(0.3)
                            await swarm_manager.broadcast_bot_message(
                                resp["bot_name"],
                                resp["content"]
                            )
                            await swarm_manager.broadcast_typing(resp["bot_name"], False)

                        # Periodically store learnings
                        if len(swarm_manager.learning_queue) >= 10:
                            await swarm_manager.store_learnings()

                elif data.get("type") == "get_online":
                    await websocket.send_json({
                        "type": "online_update",
                        "online_users": swarm_manager.get_online_users(),
                        "online_count": swarm_manager.get_online_count()
                    })

            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat"})

    except WebSocketDisconnect:
        name = swarm_manager.disconnect(user_id)
        await swarm_manager.broadcast_system(f"🔴 {name} left the swarm")
    except Exception as e:
        logger.error(f"Swarm WebSocket error: {e}")
        swarm_manager.disconnect(user_id)


@app.get("/api/swarm/status")
async def swarm_status():
    """Get Swarm Chat status."""
    return JSONResponse({
        "online_count": swarm_manager.get_online_count(),
        "online_users": swarm_manager.get_online_users(),
        "active_models": swarm_manager.active_models,
        "message_count": len(swarm_manager.chat_history),
        "learning_queue_size": len(swarm_manager.learning_queue)
    })


@app.get("/api/swarm/history")
async def swarm_history(limit: int = 50):
    """Get recent Swarm Chat history."""
    return JSONResponse({
        "messages": swarm_manager.chat_history[-limit:],
        "total": len(swarm_manager.chat_history)
    })


@app.get("/api/swarm/learning")
async def swarm_learning_stats():
    """Get real-time learning statistics from Swarm Chat."""
    return JSONResponse({
        "learning_stats": swarm_manager.get_learning_stats(),
        "status": "active",
        "description": "Real-time learning from community interactions"
    })


@app.post("/api/swarm/learn")
async def trigger_learning():
    """Force a learning cycle to process buffered interactions."""
    await swarm_manager.force_learning_cycle()
    return JSONResponse({
        "success": True,
        "message": "Learning cycle triggered",
        "stats": swarm_manager.get_learning_stats()
    })


@app.get("/api/swarm/concepts")
async def swarm_concepts():
    """Get extracted concepts from Swarm Chat conversations."""
    stats = swarm_manager.get_learning_stats()
    return JSONResponse({
        "concepts": stats.get("top_concepts", []),
        "total": stats.get("concept_count", 0)
    })


@app.get("/api/swarm/users")
async def swarm_user_patterns():
    """Get user behavior patterns learned from Swarm Chat."""
    return JSONResponse({
        "online_users": swarm_manager.get_online_users(),
        "online_count": swarm_manager.get_online_count(),
        "patterns_tracked": len(swarm_learning.user_patterns)
    })


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
