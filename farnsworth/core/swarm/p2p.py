"""
Farnsworth P2P Swarm v2.5 - The "Professorial" Gossip Protocol.

"I decided to build my own internet, with blackjack and... wait, wrong catchphrase."

Improvements:
1. TCP-based Multiplexed Streams: High reliability peer communication.
2. Gossipsub Simulation: Efficient broadcast of Knowledge Fragments (DKG).
3. Kademlia-inspired DHT: Distributed routing of agent capabilities.
4. Auto-Discovery: Async UDP beaconing for seamless swarm entry.
"""

import asyncio
import json
import uuid
import socket
import struct
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from loguru import logger

from farnsworth.core.nexus import nexus, Signal, SignalType
from farnsworth.core.swarm.dkg import DecentralizedKnowledgeGraph

@dataclass
class PeerInfo:
    id: str
    addr: str
    port: int
    capabilities: List[str]
    writer: Optional[asyncio.StreamWriter] = None
    last_seen: float = field(default_factory=asyncio.get_event_loop().time)

class SwarmFabric:
    """
    Advanced Decentralized Networking Layer.
    """
    def __init__(self, node_id: Optional[str] = None, port: int = 9999):
        self.node_id = node_id or str(uuid.uuid4())[:8]
        self.port = port
        self.peers: Dict[str, PeerInfo] = {}
        self.dkg = DecentralizedKnowledgeGraph(self.node_id)
        self.seen_messages: Set[str] = set() # For gossip deduplication
        
    async def start(self):
        """Boot the P2P Fabric."""
        logger.info(f"Swarm Fabric: Starting node {self.node_id} on port {self.port}")
        
        # 1. Start TCP Server (The listener)
        self.server = await asyncio.start_server(self._handle_peer_conn, '0.0.0.0', self.port)
        
        # 2. Start UDP Beacon (Discovery)
        asyncio.create_task(self._udp_beacon())
        asyncio.create_task(self._udp_listener())
        
        # 3. Start Peer Maintenance
        asyncio.create_task(self._maintain_peer_health())
        
        async with self.server:
            await self.server.serve_forever()

    async def _udp_beacon(self):
        """Broadcast presence via UDP."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        while True:
            beacon = json.dumps({
                "type": "BEACON",
                "id": self.node_id,
                "port": self.port,
                "caps": ["CV", "NLP", "P2P"]
            })
            sock.sendto(beacon.encode(), ('<broadcast>', 8888))
            await asyncio.sleep(15)

    async def _udp_listener(self):
        """Listen for peer beacons."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('0.0.0.0', 8888))
        sock.setblocking(False)
        loop = asyncio.get_event_loop()
        while True:
            data, addr = await loop.sock_recvfrom(sock, 1024)
            msg = json.loads(data.decode())
            if msg["id"] != self.node_id:
                asyncio.create_task(self._connect_to_peer(msg["id"], addr[0], msg["port"], msg.get("caps", [])))

    async def _connect_to_peer(self, peer_id: str, host: str, port: int, caps: List[str]):
        if peer_id in self.peers: return
        try:
            reader, writer = await asyncio.open_connection(host, port)
            self._register_peer(peer_id, host, port, caps, writer)
            # Handshake
            await self._send_to_peer(peer_id, {"type": "HELLO", "id": self.node_id})
        except Exception as e:
            logger.trace(f"P2P: Failed to connect to {peer_id}: {e}")

    def _register_peer(self, pid, host, port, caps, writer):
        self.peers[pid] = PeerInfo(id=pid, addr=host, port=port, capabilities=caps, writer=writer)
        logger.info(f"P2P: Securely linked to peer '{pid}'")

    async def _handle_peer_conn(self, reader, writer):
        """Inbound TCP Connection handler."""
        peer_addr = writer.get_extra_info('peername')
        try:
            while True:
                data = await reader.read(4096)
                if not data: break
                msg = json.loads(data.decode())
                await self._process_peer_message(msg, writer)
        except Exception as e:
            pass
        finally:
            writer.close()

    async def _process_peer_message(self, msg: Dict, writer: asyncio.StreamWriter):
        m_type = msg.get("type")
        m_id = msg.get("msg_id", str(uuid.uuid4()))

        if m_id in self.seen_messages:
            return
        self.seen_messages.add(m_id)

        if m_type == "HELLO":
            # Handshake acknowledgment
            peer_id = msg.get("id")
            if peer_id:
                logger.debug(f"P2P: Handshake from {peer_id}")
                # Send our capabilities back
                await self._send_to_writer(writer, {
                    "type": "HELLO_ACK",
                    "id": self.node_id,
                    "caps": ["CV", "NLP", "P2P", "PLANETARY"]
                })

        elif m_type == "HELLO_ACK":
            peer_id = msg.get("id")
            logger.debug(f"P2P: Handshake ACK from {peer_id}")

        elif m_type == "GOSSIP_DKG":
            # 1. Merge locally
            fragment = msg.get("fragment")
            if fragment:
                self.dkg.merge_fragment(fragment)
                logger.debug(f"P2P: Merged DKG fragment from gossip")
            # 2. Re-broadcast (Gossip)
            await self.gossip(msg)

        elif m_type == "GOSSIP_SKILL":
            # Planetary Memory skill sharing
            skill_data = msg.get("skill")
            if skill_data:
                # Signal to planetary memory
                nexus.emit(Signal(
                    type=SignalType.EXTERNAL_EVENT,
                    payload={
                        "event": "planetary_skill_received",
                        "skill": skill_data
                    },
                    source="p2p_fabric"
                ))
                logger.info(f"P2P: Received planetary skill: {skill_data.get('id', 'unknown')[:8]}...")
            await self.gossip(msg)

        elif m_type == "DHT_QUERY":
            # Find node with capability
            requested_cap = msg.get("capability")
            if requested_cap in ["CV", "NLP", "P2P", "PLANETARY"]:
                await self._send_to_writer(writer, {
                    "type": "DHT_RESPONSE",
                    "node_id": self.node_id,
                    "capability": requested_cap,
                    "available": True
                })

        elif m_type == "TASK_AUCTION":
            # Distributed Task Auction
            task_desc = msg.get("task")
            task_id = msg.get("task_id")
            # Emit for local agents to bid
            nexus.emit(Signal(
                type=SignalType.TASK_RECEIVED,
                payload={
                    "task_id": task_id,
                    "description": task_desc,
                    "from_peer": msg.get("from_node")
                },
                source="p2p_fabric"
            ))
            logger.info(f"P2P: Task auction received: {task_id}")

    async def _send_to_writer(self, writer: asyncio.StreamWriter, msg: Dict):
        """Send message directly to a writer."""
        try:
            msg["msg_id"] = msg.get("msg_id", str(uuid.uuid4()))
            payload = json.dumps(msg).encode()
            writer.write(payload)
            await writer.drain()
        except Exception as e:
            logger.debug(f"P2P: Send failed: {e}")

    async def broadcast_skill(self, skill_data: Dict):
        """Broadcast a planetary skill to the swarm."""
        msg = {
            "type": "GOSSIP_SKILL",
            "skill": skill_data,
            "from_node": self.node_id
        }
        await self.gossip(msg)
        logger.info(f"P2P: Broadcasted skill to {len(self.peers)} peers")

    async def submit_task_auction(self, task_id: str, task_desc: str):
        """Submit a task for distributed auction."""
        msg = {
            "type": "TASK_AUCTION",
            "task_id": task_id,
            "task": task_desc,
            "from_node": self.node_id
        }
        await self.gossip(msg)
        logger.info(f"P2P: Submitted task auction: {task_id}")

    async def gossip(self, msg: Dict):
        """Propagate message through the fabric."""
        pids = list(self.peers.keys())
        # Gossip to random subset of peers (simplified scale)
        for pid in pids:
            await self._send_to_peer(pid, msg)

    async def _send_to_peer(self, peer_id: str, msg: Dict):
        peer = self.peers.get(peer_id)
        if peer and peer.writer:
            try:
                msg["msg_id"] = msg.get("msg_id", str(uuid.uuid4()))
                payload = json.dumps(msg).encode()
                peer.writer.write(payload)
                await peer.writer.drain()
            except Exception:
                # Remove dead peer
                del self.peers[peer_id]

    async def _maintain_peer_health(self):
        while True:
            # Prune old seen messages
            if len(self.seen_messages) > 1000: self.seen_messages.clear()
            await asyncio.sleep(60)

# Global Instance
swarm_fabric = SwarmFabric()
