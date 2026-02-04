# Development Plan

Task: build this madness together!

Here's a **concrete implementation plan** for a **Real-time Collective Deliberation Dashboard** - a WebSocket-driven UI that visualizes multi-agent thought processes as they happen.

## 1. FILES TO CREATE

### Backend Infrastructure
```
farnsworth/web/websocket/manager.py
farnsworth/web/routers/deliberation.py
farnsworth/web/schemas/deliberation.py
farnsworth/core/collective/streaming.py
farnsworth/core/collective/visualizer.py
```

### Frontend Assets
```
farnsworth/web/static/templates/deliberation_dashboard.html
farnsworth/web/static/css/deliberation.css
farnsworth/web/static/js/deliberation-client.js
```

### Tests
```
tests/web/test_deliberation_ws.py
tests/core/test_collective_streaming.py
```

## 2. FUNCTIONS TO IMPLEMENT

### farnsworth/web/websocket/manager.py
```python
from typing import Dict, List, Set
from fastapi import WebSocket
from starlette.websockets import WebSocketState

class DeliberationConnectionManager:
    def __init__(self) -> None:
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.session_metadata: Dict[str, Dict] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        """Accept connection and register to session pool"""
        
    async def disconnect(self, websocket: WebSocket, session_id: str) -> None:
        """Clean up disconnected client"""
        
    async def broadcast_to_session(
        self, 
        session_id: str, 
        message: Dict[str, any]
    ) -> None:
        """Broadcast JSON event to all session subscribers"""
        
    async def send_personal_message(
        self, 
        message: str, 
        websocket: WebSocket
    ) -> None:
        """Send targeted message to specific client"""
```

### farnsworth/web/routers/deliberation.py
```python
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

router = APIRouter(tags=["deliberation"])

@router.websocket("/ws/deliberation/{session_id}")
async def deliberation_websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    db: AsyncSession = Depends(get_db)
) -> None:
    """
    WebSocket endpoint for real-time deliberation streaming.
    Accepts: thought_subscribe, thought_react, consensus_vote
    Emits: thought_spawned, consensus_shift, agent_speaking, memory_recall
    """

@router.post("/api/v1/deliberation/session")
async def create_deliberation_session(
    topic: str,
    agent_ids: List[str],
    max_iterations: int = 10,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, str]:
    """Initialize new deliberation session and return session_id"""

@router.get("/api/v1/deliberation/session/{session_id}/state")
async def get_session_state(
    session_id: str,
    db: AsyncSession = Depends(get_db)
) -> DeliberationStateSchema:
    """Fetch current deliberation graph and consensus metrics"""
```

### farnsworth/core/collective/streaming.py
```python
from typing import Callable, Awaitable
from dataclasses import dataclass
import asyncio

@dataclass
class DeliberationEvent:
    event_type: str  # 'thought_generated', 'consensus_update', 'memory_access'
    payload: Dict[str, any]
    timestamp: float
    agent_id: Optional[str]

class StreamingDeliberationBridge:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.subscribers: List[Callable[[DeliberationEvent], Awaitable[None]]] = []
        
    def register_callback(
        self, 
        callback: Callable[[DeliberationEvent], Awaitable[None]]
    ) -> None:
        """Register WebSocket manager callback for event streaming"""
        
    async def emit_thought_spawn(
        self,
        agent_id: str,
        thought_content: str,
        parent_thought_id: Optional[str],
        confidence_score: float
    ) -> None:
        """Called by collective engine when agent generates thought"""
        
    async def emit_memory_recall(
        self,
        agent_id: str,
        memory_ids: List[str],
        relevance_scores: List[float]
    ) -> None:
        """Stream memory retrieval events to UI"""
```

### farnsworth/core/collective/visualizer.py
```python
from farnsworth.core.collective.graph import ThoughtGraph
from farnsworth.memory.models import MemoryFragment
import networkx as nx

async def serialize_thought_graph(
    graph: ThoughtGraph,
    include_weights: bool = True
) -> Dict[str, any]:
    """
    Convert internal thought graph to D3.js compatible format
    Returns: {'nodes': [...], 'links': [...], 'consensus_clusters': [...]}
    """

def calculate_consensus_trajectory(
    thought_history: List[Dict],
    window_size: int = 5
) -> List[Dict[str, float]]:
    """
    Calculate consensus velocity and divergence metrics over time
    Returns: [{'timestamp': float, 'consensus_score': float, 'divergence': float}]
    """

def generate_agent_color_map(
    agent_ids: List[str]
) -> Dict[str, str]:
    """Assign unique HSL colors to agents for UI consistency"""
```

## 3. IMPORTS REQUIRED

### From Existing Farnsworth Modules
```python
# From core collective system
from farnsworth.core.collective.engine import CollectiveDeliberationEngine
from farnsworth.core.collective.graph import ThoughtGraph, ThoughtNode
from farnsworth.core.cognition.agent import AgentInstance

# From memory systems
from farnsworth.memory.archival import ArchivalMemory
from farnsworth.memory.recall import RecallMechanism

# From existing web infrastructure
from farnsworth.web.server import app, get_db
from farnsworth.web.auth.dependencies import get_current_user_ws  # For WS auth

# Models (assuming SQLAlchemy)
from farnsworth.db.models.deliberation import DeliberationSession
```

## 4. INTEGRATION POINTS

### Modify: farnsworth/web/server.py
```python
# Add after existing router imports
from farnsworth.web.routers import deliberation
from farnsworth.web.websocket.manager import DeliberationConnectionManager

# Initialize manager as singleton
deliberation_manager = DeliberationConnectionManager()

# Add routers
app.include_router(deliberation.router, prefix="/api/v1")

# Mount static files if not already done
app.mount("/static", StaticFiles(directory="farnsworth/web/static"), name="static")
```

### Modify: farnsworth/core/collective/engine.py (Existing)
```python
# Add streaming bridge integration
from farnsworth.core.collective.streaming import StreamingDeliberationBridge

class CollectiveDeliberationEngine:
    def __init__(self, session_id: str, enable_streaming: bool = False):
        # ... existing init ...
        self.streaming_bridge: Optional[StreamingDeliberationBridge] = None
        if enable_streaming:
            self.streaming_bridge = StreamingDeliberationBridge(session_id)
    
    async def process_agent_turn(self, agent: AgentInstance) -> None:
        # ... existing logic ...
        
        # Add emission hook after thought generation
        if self.streaming_bridge:
            await self.streaming_bridge.emit_thought_spawn(
                agent_id=agent.id,
                thought_content=thought.content,
                parent_thought_id=thought.parent_id,
                confidence_score=thought.confidence
            )
```

### Modify: farnsworth/agents/base.py (Hook injection)
```python
# Add callback hook for memory access visualization
async def recall_memories(self, query: str, limit: int = 5) -> List[MemoryFragment]:
    results = await super().recall_memories(query, limit)
    
    # Emit to streaming bridge if in deliberation mode
    if hasattr(self, 'deliberation_bridge') and self.deliberation_bridge:
        await self.deliberation_bridge.emit_memory_recall(
            agent_id=self.id,
            memory_ids=[r.id for r in results],
            relevance_scores=[r.relevance for r in results]
        )
    return results
```

## 5. TEST COMMANDS

### Unit Tests
```bash
# Test WebSocket manager logic
pytest tests/web/test_deliberation_ws.py::test_connection_manager_broadcast -v

# Test serialization
pytest tests/core/test_collective_streaming.py::test_thought_serialization -v

# Test full deliberation flow with streaming
pytest tests/integration/test_deliberation_end_to_end.py -v --timeout=60
```

### Manual Verification
```bash
# 1. Start server
uvicorn farnsworth.web.server:app --reload --port 8000

# 2. Create deliberation session
curl -X POST "http://localhost:8000/api/v1/deliberation/session" \
  -H "Content-Type: application/json" \
  -d '{"topic": "AI safety frameworks", "agent_ids": ["agent_1", "agent_2", "agent_3"], "max_iterations": 5}'

# Response: {"session_id": "delib_abc123", "status": "initialized"}

# 3. Open dashboard (Browser)
open http://localhost:8000/static/templates/deliberation_dashboard.html?session=delib_abc123

# 4. Connect WebSocket client (CLI test)
wscat -c "ws://localhost:8000