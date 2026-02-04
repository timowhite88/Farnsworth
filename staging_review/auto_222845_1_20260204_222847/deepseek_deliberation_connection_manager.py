"""
Manages WebSocket connections for deliberation sessions in Farnsworth AI collective.
"""

import asyncio
from typing import Dict, Set

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
        try:
            if session_id in self.active_connections and websocket in self.active_connections[session_id]:
                self.active_connections[session_id].remove(websocket)
                if not self.active_connections[session_id]:  # Remove empty sessions
                    del self.active_connections[session_id]
                logger.info(f"WebSocket disconnected for session {session_id}")
        except KeyError:
            logger.error("Session ID or WebSocket not found during disconnect.")

    async def broadcast_to_session(
        self, 
        session_id: str, 
        message: Dict[str, any]
    ) -> None:
        """Broadcast JSON event to all session subscribers."""
        try:
            websockets = self.active_connections.get(session_id, set())
            for websocket in websockets:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json(message)
            logger.info(f"Message broadcasted to session {session_id}")
        except Exception as e:
            logger.error(f"Error broadcasting message: {e}")

    async def send_personal_message(
        self, 
        message: str, 
        websocket: WebSocket
    ) -> None:
        """Send targeted message to specific client."""
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(message)
            logger.info("Personal message sent.")
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")

# filename: deliberation_router.py
"""
Defines FastAPI routes for deliberation session management and WebSocket endpoints.
"""

from typing import List, Dict

from fastapi import APIRouter, WebSocket, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from farnsworth.db.models.deliberation import DeliberationSession

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
    connection_manager.connect(websocket, session_id)
    
    try:
        while True:
            data = await websocket.receive_json()
            # Handle received messages and emit responses as needed.
            logger.info(f"Received message: {data}")
    except WebSocketDisconnect:
        pass
    finally:
        connection_manager.disconnect(websocket, session_id)


@router.post("/api/v1/deliberation/session")
async def create_deliberation_session(
    topic: str,
    agent_ids: List[str],
    max_iterations: int = 10,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, str]:
    """Initialize new deliberation session and return session_id."""
    try:
        # Create a new deliberation session in the database
        new_session = DeliberationSession(topic=topic, agent_ids=agent_ids, max_iterations=max_iterations)
        db.add(new_session)
        await db.commit()
        await db.refresh(new_session)
        
        logger.info(f"Deliberation session created with ID: {new_session.id}")
        return {"session_id": new_session.id, "status": "initialized"}
    except Exception as e:
        logger.error(f"Error creating deliberation session: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.get("/api/v1/deliberation/session/{session_id}/state")
async def get_session_state(
    session_id: str,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, any]:
    """Fetch current deliberation graph and consensus metrics."""
    try:
        # Fetch session state from the database
        session = await db.get(DeliberationSession, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        logger.info(f"Fetching state for session {session.id}")
        return {
            "topic": session.topic,
            "agent_ids": session.agent_ids,
            "max_iterations": session.max_iterations
        }
    except Exception as e:
        logger.error(f"Error fetching session state: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# filename: streaming_deliberation_bridge.py
"""
Facilitates event streaming from the deliberation engine to WebSocket clients.
"""

from typing import Callable, Awaitable, List, Optional

from dataclasses import dataclass

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
            event_type='thought_generated',
            payload={
                'agent_id': agent_id,
                'thought_content': thought_content,
                'parent_thought_id': parent_thought_id,
                'confidence_score': confidence_score
            },
            timestamp=asyncio.get_event_loop().time(),
            agent_id=agent_id
        )
        
        for callback in self.subscribers:
            await callback(event)
            
    async def emit_memory_recall(
        self,
        agent_id: str,
        memory_ids: List[str],
        relevance_scores: List[float]
    ) -> None:
        """Stream memory retrieval events to UI."""
        event = DeliberationEvent(
            event_type='memory_access',
            payload={
                'agent_id': agent_id,
                'memory_ids': memory_ids,
                'relevance_scores': relevance_scores
            },
            timestamp=asyncio.get_event_loop().time(),
            agent_id=agent_id
        )
        
        for callback in self.subscribers:
            await callback(event)

# filename: deliberation_visualizer.py
"""
Visualizes deliberation data structures for UI integration.
"""

from typing import List, Dict

import networkx as nx


async def serialize_thought_graph(
    graph: nx.DiGraph,
    include_weights: bool = True
) -> Dict[str, any]:
    """
    Convert internal thought graph to D3.js compatible format.
    Returns: {'nodes': [...], 'links': [...], 'consensus_clusters': [...]}
    """
    nodes = [{'id': node, **graph.nodes[node]} for node in graph.nodes]
    links = [
        {
            'source': u,
            'target': v,
            'weight': data['weight'] if include_weights else 1
        } 
        for u, v, data in graph.edges(data=True)
    ]
    
    # Example consensus clusters (mock implementation)
    consensus_clusters = []
    
    return {'nodes': nodes, 'links': links, 'consensus_clusters': consensus_clusters}


def calculate_consensus_trajectory(
    thought_history: List[Dict[str, float]],
    window_size: int = 5
) -> List[Dict[str, float]]:
    """
    Calculate consensus velocity and divergence metrics over time.
    Returns: [{'timestamp': float, 'consensus_score': float, 'divergence': float}]
    """
    trajectory = []
    
    for i in range(len(thought_history) - window_size + 1):
        window = thought_history[i:i + window_size]
        consensus_score = sum(item['score'] for item in window) / len(window)
        divergence = max(item['score'] for item in window) - min(item['score'] for item in window)
        
        trajectory.append({
            'timestamp': window[-1]['timestamp'],
            'consensus_score': consensus_score,
            'divergence': divergence
        })
    
    return trajectory


def generate_agent_color_map(
    agent_ids: List[str]
) -> Dict[str, str]:
    """Assign unique HSL colors to agents for UI consistency."""
    import colorsys
    
    num_agents = len(agent_ids)
    hue_step = 360 / num_agents
    color_map = {
        agent_id: colorsys.hls_to_rgb(hue_step * i / 360, 0.5, 1) 
        for i, agent_id in enumerate(agent_ids)
    }
    
    return {agent_id: '#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255)) 
            for agent_id, (r,g,b) in color_map.items()}