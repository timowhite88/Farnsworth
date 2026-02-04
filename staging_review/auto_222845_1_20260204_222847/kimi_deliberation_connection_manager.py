"""
Manages WebSocket connections for real-time collective deliberation sessions.
"""

import asyncio
from typing import Dict, Set, Any
from fastapi import WebSocket
from starlette.websockets import WebSocketState

class DeliberationConnectionManager:
    def __init__(self) -> None:
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.session_metadata: Dict[str, Dict] = {}

    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        """Accept connection and register to session pool."""
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = set()
        
        self.active_connections[session_id].add(websocket)
        logger.info(f"WebSocket connected for session {session_id}")

    async def disconnect(self, websocket: WebSocket, session_id: str) -> None:
        """Clean up disconnected client."""
        if session_id in self.active_connections and websocket in self.active_connections[session_id]:
            await websocket.close()
            self.active_connections[session_id].remove(websocket)
            logger.info(f"WebSocket disconnected from session {session_id}")

    async def broadcast_to_session(
        self, 
        session_id: str, 
        message: Dict[str, Any]
    ) -> None:
        """Broadcast JSON event to all session subscribers."""
        if session_id in self.active_connections:
            for websocket in list(self.active_connections[session_id]):
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting message: {e}")
                    await self.disconnect(websocket, session_id)

    async def send_personal_message(
        self, 
        message: Dict[str, Any], 
        websocket: WebSocket
    ) -> None:
        """Send targeted message to specific client."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")

# filename: deliberation_router.py
"""
API router for creating and managing deliberation sessions via WebSockets.
"""

from fastapi import APIRouter, WebSocket, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional

from .connection_manager import DeliberationConnectionManager
from farnsworth.db.models.deliberation import DeliberationSession

router = APIRouter(tags=["deliberation"])

@router.websocket("/ws/deliberation/{session_id}")
async def deliberation_websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    db: AsyncSession = Depends(get_db),
    manager: DeliberationConnectionManager = Depends(DeliberationConnectionManager())
) -> None:
    """
    WebSocket endpoint for real-time deliberation streaming.
    Accepts: thought_subscribe, thought_react, consensus_vote
    Emits: thought_spawned, consensus_shift, agent_speaking, memory_recall
    """
    try:
        await manager.connect(websocket, session_id)
        while True:
            data = await websocket.receive_json()
            # Handle incoming messages and respond accordingly
            if data["action"] == "thought_subscribe":
                await manager.send_personal_message({"event": "subscribed"}, websocket)
            elif data["action"] == "thought_react":
                await manager.broadcast_to_session(session_id, {"event": "thought_reacted"})
    except WebSocketDisconnect:
        await manager.disconnect(websocket, session_id)
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket endpoint: {e}")

@router.post("/api/v1/deliberation/session")
async def create_deliberation_session(
    topic: str,
    agent_ids: List[str],
    max_iterations: int = 10,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, str]:
    """Initialize new deliberation session and return session_id."""
    try:
        # Create a new deliberation session
        new_session = DeliberationSession(topic=topic, agent_ids=agent_ids, max_iterations=max_iterations)
        db.add(new_session)
        await db.commit()
        await db.refresh(new_session)
        
        logger.info(f"Created new deliberation session {new_session.id} for topic: {topic}")
        return {"session_id": str(new_session.id), "status": "initialized"}
    except Exception as e:
        logger.error(f"Error creating deliberation session: {e}")
        raise HTTPException(status_code=500, detail="Failed to create deliberation session")

@router.get("/api/v1/deliberation/session/{session_id}/state")
async def get_session_state(
    session_id: str,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Fetch current deliberation graph and consensus metrics."""
    try:
        # Retrieve session state from the database
        session = await db.get(DeliberationSession, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Placeholder for actual state retrieval logic
        state = {"graph": "data", "consensus_metrics": "metrics"}
        
        logger.info(f"Fetched state for session {session_id}")
        return state
    except Exception as e:
        logger.error(f"Error fetching session state: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch session state")

# filename: streaming_deliberation_bridge.py
"""
Bridges the gap between internal deliberation events and WebSocket streaming.
"""

from typing import Callable, Awaitable, List, Dict, Optional

import asyncio
from dataclasses import dataclass

@dataclass
class DeliberationEvent:
    event_type: str  # 'thought_generated', 'consensus_update', 'memory_access'
    payload: Dict[str, Any]
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
        """Register WebSocket manager callback for event streaming."""
        self.subscribers.append(callback)

    async def emit_thought_spawn(
        self,
        agent_id: str,
        thought_content: str,
        parent_thought_id: Optional[str],
        confidence_score: float
    ) -> None:
        """Called by collective engine when agent generates thought."""
        event = DeliberationEvent(
            event_type="thought_generated",
            payload={"agent_id": agent_id, "content": thought_content, "parent_id": parent_thought_id, "confidence": confidence_score},
            timestamp=asyncio.get_event_loop().time(),
            agent_id=agent_id
        )
        await self._notify_subscribers(event)

    async def emit_memory_recall(
        self,
        agent_id: str,
        memory_ids: List[str],
        relevance_scores: List[float]
    ) -> None:
        """Stream memory retrieval events to UI."""
        event = DeliberationEvent(
            event_type="memory_access",
            payload={"agent_id": agent_id, "memory_ids": memory_ids, "relevance_scores": relevance_scores},
            timestamp=asyncio.get_event_loop().time(),
            agent_id=agent_id
        )
        await self._notify_subscribers(event)

    async def _notify_subscribers(self, event: DeliberationEvent) -> None:
        """Notify all registered subscribers of a new event."""
        for subscriber in self.subscribers:
            try:
                await subscriber(event)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")

# filename: collective_deliberation_engine.py
"""
Engine that manages the deliberation process among multiple agents.
"""

from typing import Optional

import asyncio
from farnsworth.core.collective.graph import ThoughtGraph, ThoughtNode
from farnsworth.memory.models import MemoryFragment

from .streaming_bridge import StreamingDeliberationBridge


class CollectiveDeliberationEngine:
    def __init__(self, session_id: str, enable_streaming: bool = False) -> None:
        self.session_id = session_id
        self.graph = ThoughtGraph()
        self.streaming_bridge: Optional[StreamingDeliberationBridge] = None
        if enable_streaming:
            self.streaming_bridge = StreamingDeliberationBridge(session_id)

    async def process_agent_turn(self, agent: "AgentInstance") -> None:
        """Process the turn for a given agent."""
        # Placeholder logic for thought generation
        thought = ThoughtNode(content="Generated thought", parent_id=None)
        self.graph.add_node(thought)
        
        if self.streaming_bridge:
            await self.streaming_bridge.emit_thought_spawn(
                agent_id=agent.id,
                thought_content=thought.content,
                parent_thought_id=thought.parent_id,
                confidence_score=1.0
            )

# filename: deliberation_visualizer.py
"""
Visualizes the internal state of collective deliberations for external interfaces.
"""

import asyncio
from typing import Dict, List

from farnsworth.core.collective.graph import ThoughtGraph


async def serialize_thought_graph(
    graph: ThoughtGraph,
    include_weights: bool = True
) -> Dict[str, Any]:
    """
    Convert internal thought graph to D3.js compatible format.
    Returns: {'nodes': [...], 'links': [...], 'consensus_clusters': [...]}
    """
    nodes = [{"id": node.id, "content": node.content} for node in graph.nodes]
    links = [
        {"source": link.source.id, "target": link.target.id, "weight": getattr(link, "weight", 1)}
        for link in graph.links
    ]
    return {"nodes": nodes, "links": links}

def calculate_consensus_trajectory(
    thought_history: List[Dict[str, Any]],
    window_size: int = 5
) -> List[Dict[str, float]]:
    """
    Calculate consensus velocity and divergence metrics over time.
    Returns: [{'timestamp': float, 'consensus_score': float, 'divergence': float}]
    """
    trajectory = []
    for i in range(0, len(thought_history), window_size):
        window = thought_history[i:i + window_size]
        timestamp = window[-1]['timestamp']
        consensus_score = sum(entry['score'] for entry in window) / len(window)
        divergence = max(entry['divergence'] for entry in window)
        trajectory.append({"timestamp": timestamp, "consensus_score": consensus_score, "divergence": divergence})
    return trajectory

def generate_agent_color_map(
    agent_ids: List[str]
) -> Dict[str, str]:
    """Assign unique HSL colors to agents for UI consistency."""
    import colorsys
    hue_step = 1.0 / len(agent_ids)
    return {agent_id: f"hsl({i * hue_step * 360}, 100%, 50%)" for i, agent_id in enumerate(agent_ids)}

if __name__ == "__main__":
    # Test code to demonstrate usage of functions
    async def main():
        graph = ThoughtGraph()
        graph.add_node(ThoughtNode(id="1", content="Root"))
        graph.add_node(ThoughtNode(id="2", content="Child 1", parent_id="1"))

        serialized_graph = await serialize_thought_graph(graph)
        print(serialized_graph)

    asyncio.run(main())