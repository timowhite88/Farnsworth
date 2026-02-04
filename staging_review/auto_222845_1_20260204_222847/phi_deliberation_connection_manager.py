"""
Manages WebSocket connections for real-time collective deliberation sessions, allowing clients to connect, disconnect, and receive broadcast messages related to specific sessions.
"""

import asyncio
from typing import Dict, Set, Any
from fastapi import WebSocket

class DeliberationConnectionManager:
    def __init__(self) -> None:
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.session_metadata: Dict[str, Dict] = {}

    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        """
        Accept connection and register to session pool.
        
        Args:
            websocket (WebSocket): The client WebSocket connection.
            session_id (str): Identifier for the deliberation session.
        """
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = set()
        self.active_connections[session_id].add(websocket)
        logger.info(f"WebSocket connected to session {session_id}")

    async def disconnect(self, websocket: WebSocket, session_id: str) -> None:
        """
        Clean up disconnected client.
        
        Args:
            websocket (WebSocket): The client WebSocket connection that is being disconnected.
            session_id (str): Identifier for the deliberation session.
        """
        if session_id in self.active_connections and websocket in self.active_connections[session_id]:
            await websocket.close()
            self.active_connections[session_id].remove(websocket)
            logger.info(f"WebSocket disconnected from session {session_id}")
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]

    async def broadcast_to_session(
        self, 
        session_id: str, 
        message: Dict[str, Any]
    ) -> None:
        """
        Broadcast JSON event to all session subscribers.
        
        Args:
            session_id (str): Identifier for the deliberation session.
            message (Dict[str, Any]): The message to broadcast.
        """
        if session_id in self.active_connections:
            connections = self.active_connections[session_id]
            await asyncio.wait([conn.send_json(message) for conn in connections])

    async def send_personal_message(
        self, 
        message: str, 
        websocket: WebSocket
    ) -> None:
        """
        Send targeted message to specific client.
        
        Args:
            message (str): The message content to be sent.
            websocket (WebSocket): The target client WebSocket connection.
        """
        await websocket.send_text(message)

# filename: deliberation_router.py
"""
Routers for handling WebSocket and HTTP requests related to real-time collective deliberation sessions.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional

from farnsworth.db.models.deliberation import DeliberationSession
from farnsworth.web.server import app, get_db
from .manager import DeliberationConnectionManager
from .schemas import DeliberationStateSchema

router = APIRouter(tags=["deliberation"])
deliberation_manager = DeliberationConnectionManager()

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
    try:
        await deliberation_manager.connect(websocket, session_id)
        while True:
            data = await websocket.receive_json()
            # Handle incoming messages and emit responses as needed.
            if data["action"] == "thought_react":
                response = {"event": "agent_speaking", "content": "Reacted"}
                await deliberation_manager.send_personal_message(str(response), websocket)
    except WebSocketDisconnect:
        logger.warning(f"WebSocket disconnected for session {session_id}")
        await deliberation_manager.disconnect(websocket, session_id)
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {str(e)}")

@router.post("/api/v1/deliberation/session")
async def create_deliberation_session(
    topic: str,
    agent_ids: List[str],
    max_iterations: int = 10,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, str]:
    """
    Initialize new deliberation session and return session_id.
    
    Args:
        topic (str): The topic of the deliberation session.
        agent_ids (List[str]): List of participating agent IDs.
        max_iterations (int): Maximum number of iterations for the session.
        
    Returns:
        Dict[str, str]: Contains 'session_id' and status information.
    """
    try:
        # Create a new deliberation session in the database
        new_session = DeliberationSession(topic=topic, agent_ids=agent_ids, max_iterations=max_iterations)
        db.add(new_session)
        await db.commit()
        await db.refresh(new_session)
        return {"session_id": new_session.id, "status": "initialized"}
    except Exception as e:
        logger.error(f"Error creating deliberation session: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Failed to create deliberation session")

@router.get("/api/v1/deliberation/session/{session_id}/state")
async def get_session_state(
    session_id: str,
    db: AsyncSession = Depends(get_db)
) -> DeliberationStateSchema:
    """
    Fetch current deliberation graph and consensus metrics.
    
    Args:
        session_id (str): Identifier for the deliberation session.
        
    Returns:
        DeliberationStateSchema: Serialized state of the deliberation session.
    """
    try:
        # Retrieve session from database
        session = await db.get(DeliberationSession, session_id)
        if not session:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail="Deliberation session not found")
        
        # Placeholder logic for fetching state - replace with actual implementation
        state_data = {"graph": {}, "consensus": 0.5}  # Example data
        return DeliberationStateSchema(**state_data)
    except Exception as e:
        logger.error(f"Error retrieving deliberation session state: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Failed to retrieve session state")

# filename: collective_streaming.py
"""
Handles real-time streaming of deliberation events, including thought generation and memory recall.
"""

from typing import Callable, Awaitable, List, Optional, Dict
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
        """
        Register WebSocket manager callback for event streaming.
        
        Args:
            callback (Callable): Function to call with each DeliberationEvent.
        """
        self.subscribers.append(callback)

    async def emit_thought_spawn(
        self,
        agent_id: str,
        thought_content: str,
        parent_thought_id: Optional[str],
        confidence_score: float
    ) -> None:
        """
        Called by collective engine when an agent generates a thought.
        
        Args:
            agent_id (str): ID of the agent generating the thought.
            thought_content (str): Content of the generated thought.
            parent_thought_id (Optional[str]): Parent thought's ID, if applicable.
            confidence_score (float): Confidence level in the thought content.
        """
        event = DeliberationEvent(
            event_type="thought_generated",
            payload={
                "agent_id": agent_id,
                "content": thought_content,
                "parent_thought_id": parent_thought_id,
                "confidence": confidence_score
            },
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
        """
        Stream memory retrieval events to UI.
        
        Args:
            agent_id (str): ID of the agent recalling memories.
            memory_ids (List[str]): IDs of the recalled memories.
            relevance_scores (List[float]): Relevance scores for each memory.
        """
        event = DeliberationEvent(
            event_type="memory_access",
            payload={
                "agent_id": agent_id,
                "memory_ids": memory_ids,
                "relevance_scores": relevance_scores
            },
            timestamp=asyncio.get_event_loop().time(),
            agent_id=agent_id
        )
        await self._notify_subscribers(event)

    async def _notify_subscribers(self, event: DeliberationEvent) -> None:
        """
        Notify all registered subscribers of a new deliberation event.
        
        Args:
            event (DeliberationEvent): The event to notify about.
        """
        tasks = [callback(event) for callback in self.subscribers]
        await asyncio.gather(*tasks)

# filename: collective_visualizer.py
"""
Provides utilities for visualizing and analyzing the deliberation process using a thought graph.
"""

from farnsworth.core.collective.graph import ThoughtGraph, ThoughtNode
from farnsworth.memory.models import MemoryFragment
import networkx as nx
from typing import Dict, List

async def serialize_thought_graph(
    graph: ThoughtGraph,
    include_weights: bool = True
) -> Dict[str, any]:
    """
    Convert internal thought graph to D3.js compatible format.
    
    Args:
        graph (ThoughtGraph): The thought graph to serialize.
        include_weights (bool): Whether to include edge weights in the output.
        
    Returns:
        Dict[str, any]: Serialized representation of the thought graph with nodes and links.
    """
    try:
        G = nx.DiGraph()
        for node_id, node in graph.nodes.items():
            G.add_node(node_id, **node.to_dict())
            if include_weights:
                for neighbor_id, weight in node.edges.items():
                    G[node_id][neighbor_id]['weight'] = weight

        nodes = [{"id": n, "label": G.nodes[n].get("content", "")} for n in G.nodes]
        links = [
            {"source": u, "target": v, "weight": d.get('weight', 1.0)}
            for u, v, d in G.edges(data=True)
        ]

        return {
            'nodes': nodes,
            'links': links,
            'consensus_clusters': []  # Placeholder - implement cluster logic
        }
    except Exception as e:
        logger.error(f"Error serializing thought graph: {str(e)}")
        raise

def calculate_consensus_trajectory(
    thought_history: List[Dict],
    window_size: int = 5
) -> List[Dict[str, float]]:
    """
    Calculate consensus velocity and divergence metrics over time.
    
    Args:
        thought_history (List[Dict]): Historical records of thoughts with timestamps.
        window_size (int): Number of past events to consider for metric calculation.
        
    Returns:
        List[Dict[str, float]]: Calculated consensus metrics at each timestamp.
    """
    trajectory = []
    for i in range(len(thought_history)):
        if i < window_size - 1:
            continue
        window = thought_history[i-window_size+1:i+1]
        # Placeholder logic - implement actual calculation
        consensus_score = sum([entry['consensus'] for entry in window]) / window_size
        divergence = max([entry['divergence'] for entry in window])
        trajectory.append({
            "timestamp": window[-1]['timestamp'],
            "consensus_score": consensus_score,
            "divergence": divergence
        })
    return trajectory

def generate_agent_color_map(
    agent_ids: List[str]
) -> Dict[str, str]:
    """
    Assign unique HSL colors to agents for UI consistency.
    
    Args:
        agent_ids (List[str]): IDs of the agents participating in the session.
        
    Returns:
        Dict[str, str]: Mapping from agent ID to color string.
    """
    import colorsys
    num_agents = len(agent_ids)
    hue_step = 360 / num_agents
    return {
        agent_id: f"hsl({i * hue_step}, 100%, 50%)"
        for i, agent_id in enumerate(agent_ids)
    }