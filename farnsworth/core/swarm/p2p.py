"""
Farnsworth P2P Swarm Protocol - The "Antigravity" Decentralized Fabric.

"I don't need a server. I AM the server. And so are you!"

This module enables Farnsworth instances to discover and collaborate with each other
over a local network or via a zero-trust overlay.

Concepts:
1. Distributed Task Auction (DTA): Agents bid on tasks they are best suited for.
2. Federated Learning Fragments: Sharing knowledge updates without sharing raw data.
3. Swarm Consensus: Majority voting on critical logic across multiple machines.
"""

import asyncio
import json
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from loguru import logger

from farnsworth.core.nexus import nexus, Signal, SignalType

@dataclass
class SwarmNode:
    id: str
    capabilities: List[str]
    latency: float = 0.0
    trust_score: float = 1.0

import socket
from farnsworth.core.swarm.dkg import DecentralizedKnowledgeGraph

class P2PSwarmProtocol:
    def __init__(self, node_id: Optional[str] = None):
        self.node_id = node_id or str(uuid.uuid4())[:8]
        self.peers: Dict[str, SwarmNode] = {}
        self.dkg = DecentralizedKnowledgeGraph(self.node_id)
        self.port = 8888
        nexus.subscribe(SignalType.TASK_CREATED, self._on_local_task_created)

    async def start_discovery(self):
        """Announce presence and listen for other Farnsworth instances via UDP."""
        logger.info(f"P2P: Node {self.node_id} entering the fabric on port {self.port}...")
        
        # 1. Start Listener
        loop = asyncio.get_event_loop()
        transport, protocol = await loop.create_datagram_endpoint(
            lambda: SwarmProtocol(self),
            local_addr=('0.0.0.0', self.port)
        )
        
        # 2. Start Broadcaster
        asyncio.create_task(self._broadcast_presence())

    async def _broadcast_presence(self):
        """Send periodic HELLO packets."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        
        while True:
            msg = json.dumps({"type": "HELLO", "id": self.node_id, "caps": ["RECONSTRUCTION", "NLP"]})
            sock.sendto(msg.encode(), ('<broadcast>', self.port))
            await asyncio.sleep(30)

    def _add_peer(self, peer_data: Dict):
        pid = peer_data["id"]
        if pid != self.node_id:
            logger.info(f"P2P: Discovered peer '{pid}'")
            self.peers[pid] = SwarmNode(id=pid, capabilities=peer_data.get("caps", []))

    async def share_knowledge(self):
        """Broadcast DKG fragment to all peers."""
        if not self.peers: return
        
        fragment = self.dkg.create_sync_fragment()
        msg = json.dumps({"type": "DKG_SYNC", "id": self.node_id, "data": fragment})
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.sendto(msg.encode(), ('<broadcast>', self.port))
        logger.debug("P2P: Knowledge fragment broadcasted.")

class SwarmProtocol(asyncio.DatagramProtocol):
    def __init__(self, handler: P2PSwarmProtocol):
        self.handler = handler

    def datagram_received(self, data: bytes, addr: Tuple[str, int]):
        try:
            msg = json.loads(data.decode())
            m_type = msg.get("type")
            
            if m_type == "HELLO":
                self.handler._add_peer(msg)
            elif m_type == "DKG_SYNC":
                self.handler.dkg.merge_fragment(msg["data"])
        except Exception as e:
            pass

# Global Instance
swarm_p2p = P2PSwarmProtocol()
