"""
Farnsworth P2P Swarm v2.6 - The "Professorial" Gossip Protocol.

"I decided to build my own internet, with blackjack and... wait, wrong catchphrase."

Improvements:
1. TCP-based Multiplexed Streams: High reliability peer communication.
2. Gossipsub Simulation: Efficient broadcast of Knowledge Fragments (DKG).
3. Kademlia-inspired DHT: Distributed routing of agent capabilities.
4. Auto-Discovery: Async UDP beaconing for seamless swarm entry.
5. WAN Bootstrap: WebSocket connection to relay server with password auth.
"""

import asyncio
import json
import uuid
import socket
import os
import random
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from loguru import logger


# =============================================================================
# RETRY UTILITIES (AGI v1.8 - Exponential Backoff)
# =============================================================================

class ExponentialBackoff:
    """
    Exponential backoff with jitter for retry logic.

    AGI v1.8: Prevents retry storms and improves network stability.
    """
    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 300.0,
        multiplier: float = 2.0,
        jitter: float = 0.1,
    ):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.jitter = jitter
        self.attempt = 0
        self.last_attempt: Optional[datetime] = None

    def next_delay(self) -> float:
        """Calculate next delay with exponential backoff and jitter."""
        delay = min(
            self.base_delay * (self.multiplier ** self.attempt),
            self.max_delay
        )
        # Add jitter (random variation ±jitter%)
        jitter_range = delay * self.jitter
        delay += random.uniform(-jitter_range, jitter_range)
        self.attempt += 1
        self.last_attempt = datetime.now()
        return max(0.1, delay)

    def reset(self) -> None:
        """Reset backoff state after successful connection."""
        self.attempt = 0
        self.last_attempt = None

    def should_retry(self, max_attempts: Optional[int] = None) -> bool:
        """Check if retry should be attempted."""
        if max_attempts is None:
            return True
        return self.attempt < max_attempts


class TimeBoundedSet:
    """
    Set with time-based eviction for message deduplication.

    AGI v1.8: Prevents unbounded memory growth in seen_messages.
    """
    def __init__(self, max_age_seconds: float = 300.0, max_size: int = 10000):
        self.max_age = timedelta(seconds=max_age_seconds)
        self.max_size = max_size
        self._items: Dict[str, datetime] = {}

    def add(self, item: str) -> None:
        """Add item with current timestamp."""
        self._items[item] = datetime.now()
        # Cleanup if over size limit
        if len(self._items) > self.max_size:
            self._evict_expired()

    def __contains__(self, item: str) -> bool:
        """Check if item exists and is not expired."""
        if item not in self._items:
            return False
        age = datetime.now() - self._items[item]
        if age > self.max_age:
            del self._items[item]
            return False
        return True

    def _evict_expired(self) -> int:
        """Remove expired items, return count removed."""
        now = datetime.now()
        expired = [k for k, v in self._items.items() if now - v > self.max_age]
        for k in expired:
            del self._items[k]
        return len(expired)

    def clear(self) -> None:
        """Clear all items."""
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

# Optional websockets for bootstrap connection
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

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

    AGI v1.8 Improvements:
    - Exponential backoff for connection retries
    - Time-bounded message deduplication (prevents memory leaks)
    - Improved error handling and resilience
    """
    def __init__(self, node_id: Optional[str] = None, port: int = 9999,
                 bootstrap_url: Optional[str] = None, bootstrap_password: Optional[str] = None):
        self.node_id = node_id or str(uuid.uuid4())[:8]
        self.port = port
        self.peers: Dict[str, PeerInfo] = {}
        self.dkg = DecentralizedKnowledgeGraph(self.node_id)

        # AGI v1.8: Time-bounded deduplication (5 min TTL, 10k max)
        self.seen_messages = TimeBoundedSet(max_age_seconds=300.0, max_size=10000)

        # Bootstrap configuration (for WAN connectivity)
        self.bootstrap_url = bootstrap_url or os.getenv("FARNSWORTH_BOOTSTRAP_PEER")
        self.bootstrap_password = bootstrap_password or os.getenv("FARNSWORTH_BOOTSTRAP_PASSWORD")
        self.bootstrap_ws = None
        self.bootstrap_authenticated = False

        # AGI v1.8: Exponential backoff for connections
        self._bootstrap_backoff = ExponentialBackoff(base_delay=5.0, max_delay=300.0)
        self._peer_backoffs: Dict[str, ExponentialBackoff] = {}  # peer_id -> backoff

    async def start(self):
        """Boot the P2P Fabric."""
        logger.info(f"Swarm Fabric: Starting node {self.node_id} on port {self.port}")

        # 1. Start TCP Server (The listener)
        self.server = await asyncio.start_server(self._handle_peer_conn, '0.0.0.0', self.port)

        # 2. Start UDP Beacon (Discovery for LAN)
        asyncio.create_task(self._udp_beacon())
        asyncio.create_task(self._udp_listener())

        # 3. Start Peer Maintenance
        asyncio.create_task(self._maintain_peer_health())

        # 4. Connect to Bootstrap Node (WAN)
        if self.bootstrap_url and WEBSOCKETS_AVAILABLE:
            asyncio.create_task(self._connect_to_bootstrap())

        async with self.server:
            await self.server.serve_forever()

    async def _connect_to_bootstrap(self):
        """
        Connect to remote bootstrap node for WAN P2P.

        AGI v1.8: Uses exponential backoff with jitter for reconnection.
        """
        while True:
            try:
                delay = self._bootstrap_backoff.next_delay()
                logger.info(
                    f"P2P: Connecting to bootstrap {self.bootstrap_url}... "
                    f"(attempt {self._bootstrap_backoff.attempt})"
                )
                async with websockets.connect(self.bootstrap_url) as ws:
                    self.bootstrap_ws = ws
                    self.bootstrap_authenticated = False

                    # Reset backoff on successful connection
                    self._bootstrap_backoff.reset()

                    # Send HELLO with password
                    hello_msg = {
                        "type": "HELLO",
                        "node_id": self.node_id,
                        "version": "2.9.0",
                        "capabilities": ["CV", "NLP", "P2P", "PLANETARY"]
                    }
                    if self.bootstrap_password:
                        hello_msg["password"] = self.bootstrap_password

                    await ws.send(json.dumps(hello_msg))

                    # Listen for messages
                    async for message in ws:
                        await self._process_bootstrap_message(message)

            except Exception as e:
                logger.warning(f"P2P: Bootstrap connection failed: {e}")
                self.bootstrap_ws = None
                self.bootstrap_authenticated = False

            # AGI v1.8: Exponential backoff with jitter
            delay = self._bootstrap_backoff.next_delay()
            logger.debug(f"P2P: Retrying bootstrap connection in {delay:.1f}s")
            await asyncio.sleep(delay)

    async def _process_bootstrap_message(self, message: str):
        """Process messages from bootstrap server."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "AUTH_FAILED":
                logger.error(f"P2P: Bootstrap auth failed - {data.get('message', 'Invalid password')}")
                self.bootstrap_authenticated = False
                return

            if msg_type == "WELCOME":
                self.bootstrap_authenticated = True
                peer_count = data.get("peer_count", 0)
                logger.info(f"P2P: Connected to bootstrap! {peer_count} peers in network")
                return

            if msg_type == "AUTH_REQUIRED":
                logger.warning("P2P: Bootstrap requires authentication")
                return

            if msg_type == "GOSSIP":
                # Received planetary skill from network
                skill_data = data.get("skill")
                if skill_data:
                    nexus.emit(Signal(
                        type=SignalType.EXTERNAL_EVENT,
                        payload={
                            "event": "planetary_skill_received",
                            "skill": skill_data
                        },
                        source="bootstrap"
                    ))
                    logger.info(f"P2P: Received skill from bootstrap: {skill_data.get('id', 'unknown')[:8]}...")

            if msg_type == "PEER_RES":
                # List of peers for direct connection
                peers = data.get("peers", [])
                logger.debug(f"P2P: Received {len(peers)} peers from bootstrap")
                for peer in peers:
                    asyncio.create_task(
                        self._connect_to_peer(peer["id"], peer["ip"], self.port, [])
                    )

        except json.JSONDecodeError:
            logger.error("P2P: Malformed message from bootstrap")
        except Exception as e:
            logger.error(f"P2P: Error processing bootstrap message: {e}")

    async def broadcast_to_bootstrap(self, msg: Dict):
        """Send message to bootstrap for WAN distribution."""
        if self.bootstrap_ws and self.bootstrap_authenticated:
            try:
                await self.bootstrap_ws.send(json.dumps(msg))
            except Exception as e:
                logger.debug(f"P2P: Bootstrap send failed: {e}")

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
        """
        Connect to a peer with exponential backoff on failures.

        AGI v1.8: Prevents retry storms when peers are unavailable.
        """
        if peer_id in self.peers:
            return

        # Get or create backoff for this peer
        if peer_id not in self._peer_backoffs:
            self._peer_backoffs[peer_id] = ExponentialBackoff(
                base_delay=1.0, max_delay=60.0, multiplier=2.0
            )

        backoff = self._peer_backoffs[peer_id]

        # Check if we should attempt connection (max 5 attempts then wait)
        if not backoff.should_retry(max_attempts=5):
            if backoff.last_attempt:
                time_since = (datetime.now() - backoff.last_attempt).total_seconds()
                if time_since < backoff.max_delay:
                    return  # Still in cooldown
                else:
                    backoff.reset()  # Reset after cooldown

        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=10.0  # 10s connection timeout
            )
            self._register_peer(peer_id, host, port, caps, writer)
            # Handshake
            await self._send_to_peer(peer_id, {"type": "HELLO", "id": self.node_id})
            # Reset backoff on success
            backoff.reset()
            # Clean up backoff tracker
            del self._peer_backoffs[peer_id]
        except asyncio.TimeoutError:
            backoff.next_delay()
            logger.debug(f"P2P: Connection to {peer_id} timed out")
        except Exception as e:
            backoff.next_delay()
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

        elif m_type == "GOSSIP_AUDIO":
            # Planetary Audio Shard - metadata sharing
            metadata = msg.get("metadata")
            peer_id = msg.get("node_id", "unknown")
            if metadata:
                nexus.emit(Signal(
                    type=SignalType.EXTERNAL_EVENT,
                    payload={
                        "event": "planetary_audio_metadata",
                        "metadata": metadata,
                        "peer_id": peer_id
                    },
                    source="p2p_fabric"
                ))
                logger.debug(f"P2P: Received audio metadata: {metadata.get('text_hash', 'unknown')[:8]}...")
            await self.gossip(msg)

        elif m_type == "AUDIO_REQUEST":
            # Peer requesting audio file
            text_hash = msg.get("text_hash")
            requester_id = msg.get("requester_id")
            if text_hash and requester_id:
                nexus.emit(Signal(
                    type=SignalType.EXTERNAL_EVENT,
                    payload={
                        "event": "audio_request",
                        "text_hash": text_hash,
                        "requester_id": requester_id
                    },
                    source="p2p_fabric"
                ))
                logger.debug(f"P2P: Audio request for {text_hash[:8]}... from {requester_id}")

        elif m_type == "AUDIO_RESPONSE":
            # Peer sending audio file data
            text_hash = msg.get("text_hash")
            audio_data = msg.get("audio_data")
            sender_id = msg.get("sender_id")
            if text_hash and audio_data:
                nexus.emit(Signal(
                    type=SignalType.EXTERNAL_EVENT,
                    payload={
                        "event": "audio_response",
                        "text_hash": text_hash,
                        "audio_data": audio_data,
                        "sender_id": sender_id
                    },
                    source="p2p_fabric"
                ))
                logger.debug(f"P2P: Audio response for {text_hash[:8]}... from {sender_id}")

        elif m_type == "GOSSIP_LEARNING":
            # Shared learning data from other nodes
            cycle = msg.get("cycle", 0)
            concepts = msg.get("concepts", [])
            peer_id = msg.get("node_id", "unknown")
            nexus.emit(Signal(
                type=SignalType.EXTERNAL_EVENT,
                payload={
                    "event": "planetary_learning_received",
                    "cycle": cycle,
                    "concepts": concepts,
                    "tool_stats": msg.get("tool_stats", {}),
                    "user_count": msg.get("user_count", 0),
                    "peer_id": peer_id,
                    "timestamp": msg.get("timestamp")
                },
                source="p2p_fabric"
            ))
            logger.info(f"P2P: Received learning from {peer_id} - cycle {cycle}, {len(concepts)} concepts")
            # Re-gossip to propagate through network
            await self.gossip(msg)

        elif m_type == "GOSSIP_RESONANCE":
            # Inter-collective thought packet (Collective Resonance v1.4)
            source_collective = msg.get("source_collective_id", "unknown")
            insight = msg.get("insight", "")
            if insight:
                nexus.emit(Signal(
                    type=SignalType.EXTERNAL_EVENT,
                    payload={
                        "event": "resonant_thought_received",
                        "packet_id": msg.get("packet_id"),
                        "source_collective": source_collective,
                        "insight": insight,
                        "snippet": msg.get("snippet", []),
                        "domains": msg.get("domains", []),
                        "confidence": msg.get("confidence", 0.5),
                        "query_hash": msg.get("query_hash", ""),
                        "timestamp": msg.get("timestamp")
                    },
                    source="p2p_fabric"
                ))
                logger.info(f"P2P: Received resonant thought from collective {source_collective}: {insight[:50]}...")
            # Re-gossip for propagation
            await self.gossip(msg)

        elif m_type == "GOSSIP_CONVERSATION":
            # Bot-to-bot conversation shared from other nodes
            conversation = msg.get("conversation", [])
            peer_id = msg.get("node_id", "unknown")
            if conversation:
                nexus.emit(Signal(
                    type=SignalType.EXTERNAL_EVENT,
                    payload={
                        "event": "planetary_conversation_received",
                        "conversation": conversation,
                        "peer_id": peer_id,
                        "timestamp": msg.get("timestamp")
                    },
                    source="p2p_fabric"
                ))
                logger.debug(f"P2P: Received conversation from {peer_id} ({len(conversation)} messages)")
            await self.gossip(msg)

        # =====================================================================
        # FEDERATED LEARNING MESSAGE TYPES (Planetary AGI Cohesion)
        # =====================================================================

        elif m_type == "GOSSIP_GRADIENT":
            # Federated learning gradient update from peer
            gradient_data = msg.get("gradient")
            model_version = msg.get("model_version", 0)
            epsilon = msg.get("privacy_epsilon", 1.0)
            peer_id = msg.get("node_id", "unknown")

            if gradient_data:
                nexus.emit(Signal(
                    type=SignalType.EXTERNAL_EVENT,
                    payload={
                        "event": "federated_gradient_received",
                        "gradient": gradient_data,
                        "model_version": model_version,
                        "privacy_epsilon": epsilon,
                        "peer_id": peer_id,
                        "sample_count": msg.get("sample_count", 0),
                        "timestamp": msg.get("timestamp")
                    },
                    source="p2p_fabric"
                ))
                logger.info(f"P2P: Received federated gradient from {peer_id} (v{model_version})")
            # Re-gossip with TTL check
            ttl = msg.get("ttl", 3)
            if ttl > 0:
                msg["ttl"] = ttl - 1
                await self.gossip(msg)

        elif m_type == "GOSSIP_FITNESS":
            # Federated fitness sharing from evolution system
            fitness_data = msg.get("fitness")
            genome_hash = msg.get("genome_hash")  # Anonymous genome identifier
            peer_id = msg.get("node_id", "unknown")

            if fitness_data and genome_hash:
                nexus.emit(Signal(
                    type=SignalType.EXTERNAL_EVENT,
                    payload={
                        "event": "federated_fitness_received",
                        "fitness": fitness_data,
                        "genome_hash": genome_hash,
                        "generation": msg.get("generation", 0),
                        "peer_id": peer_id,
                        "timestamp": msg.get("timestamp")
                    },
                    source="p2p_fabric"
                ))
                logger.debug(f"P2P: Received fitness from {peer_id} for genome {genome_hash[:8]}...")
            await self.gossip(msg)

        elif m_type == "GOSSIP_GENOME_MIGRATION":
            # Top genome migrating to this node
            genome_data = msg.get("genome")
            fitness_score = msg.get("fitness_score", 0)
            peer_id = msg.get("node_id", "unknown")

            if genome_data:
                nexus.emit(Signal(
                    type=SignalType.EXTERNAL_EVENT,
                    payload={
                        "event": "genome_migration_received",
                        "genome": genome_data,
                        "fitness_score": fitness_score,
                        "source_node": peer_id,
                        "generation": msg.get("generation", 0),
                        "timestamp": msg.get("timestamp")
                    },
                    source="p2p_fabric"
                ))
                logger.info(f"P2P: Genome migration from {peer_id} (fitness={fitness_score:.3f})")
            # Don't re-gossip migrations (point-to-point)

        elif m_type == "GOSSIP_MEMORY_EMBEDDING":
            # Anonymized memory embedding for federated recall
            embedding_hash = msg.get("embedding_hash")
            noisy_embedding = msg.get("noisy_embedding")
            tags = msg.get("tags", [])
            peer_id = msg.get("node_id", "unknown")

            if embedding_hash and noisy_embedding:
                nexus.emit(Signal(
                    type=SignalType.EXTERNAL_EVENT,
                    payload={
                        "event": "federated_memory_embedding",
                        "embedding_hash": embedding_hash,
                        "noisy_embedding": noisy_embedding,
                        "tags": tags,
                        "peer_id": peer_id,
                        "timestamp": msg.get("timestamp")
                    },
                    source="p2p_fabric"
                ))
                logger.debug(f"P2P: Received anonymized embedding from {peer_id}")
            await self.gossip(msg)

        # =====================================================================
        # A2A PROTOCOL MESSAGE TYPES (AGI v1.8)
        # =====================================================================

        elif m_type == "A2A_SESSION_REQUEST":
            # Agent requesting a collaboration session
            await self._handle_a2a_session_request(msg)

        elif m_type == "A2A_SESSION_ACCEPT":
            # Session acceptance
            session_id = msg.get("session_id")
            agent_id = msg.get("agent_id")
            nexus.emit(Signal(
                type=SignalType.A2A_SESSION_STARTED,
                payload={
                    "session_id": session_id,
                    "agent_id": agent_id,
                    "peer_id": msg.get("node_id", "unknown"),
                },
                source="p2p_fabric"
            ))
            logger.info(f"P2P: A2A session {session_id} accepted by {agent_id}")

        elif m_type == "A2A_TASK_AUCTION":
            # Task auction broadcast
            await self._handle_a2a_task_auction(msg)

        elif m_type == "A2A_BID":
            # Bid for task auction
            auction_id = msg.get("auction_id")
            bid = msg.get("bid", {})
            nexus.emit(Signal(
                type=SignalType.A2A_BID_RECEIVED,
                payload={
                    "auction_id": auction_id,
                    "bid": bid,
                    "peer_id": msg.get("node_id", "unknown"),
                },
                source="p2p_fabric"
            ))
            logger.debug(f"P2P: Received bid for auction {auction_id}")

        elif m_type == "A2A_CONTEXT_SHARE":
            # Context sharing between agents
            await self._handle_a2a_context_share(msg)

        elif m_type == "A2A_SKILL_TRANSFER":
            # Skill transfer between agents
            await self._handle_a2a_skill_transfer(msg)

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
        """Broadcast a planetary skill to the swarm (LAN + WAN)."""
        msg = {
            "type": "GOSSIP_SKILL",
            "skill": skill_data,
            "from_node": self.node_id
        }

        # Local peers (LAN)
        await self.gossip(msg)

        # Bootstrap (WAN) - send as GOSSIP for relay
        bootstrap_msg = {
            "type": "GOSSIP",
            "skill_id": skill_data.get("id", "unknown"),
            "skill": skill_data,
            "from_node": self.node_id
        }
        await self.broadcast_to_bootstrap(bootstrap_msg)

        total_peers = len(self.peers) + (1 if self.bootstrap_authenticated else 0)
        logger.info(f"P2P: Broadcasted skill to {total_peers} peers (LAN + WAN)")

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

    async def broadcast_message(self, msg: Dict):
        """Broadcast any message to all peers (LAN + WAN)."""
        # Add node_id if not present
        if "node_id" not in msg:
            msg["node_id"] = self.node_id

        # Local peers (LAN)
        await self.gossip(msg)

        # Bootstrap (WAN)
        await self.broadcast_to_bootstrap(msg)

        total_peers = len(self.peers) + (1 if self.bootstrap_authenticated else 0)
        logger.debug(f"P2P: Broadcasted message type '{msg.get('type')}' to {total_peers} peers")

    async def send_to_peer(self, peer_id: str, msg: Dict):
        """Send a message to a specific peer by ID."""
        await self._send_to_peer(peer_id, msg)

    async def broadcast_conversation(self, conversation: List[Dict]):
        """Share bot conversation excerpts across the planetary network."""
        if not conversation:
            return

        msg = {
            "type": "GOSSIP_CONVERSATION",
            "conversation": conversation[-10:],  # Last 10 messages
            "node_id": self.node_id,
            "timestamp": asyncio.get_event_loop().time()
        }

        # Local peers (LAN)
        await self.gossip(msg)

        # Bootstrap (WAN)
        await self.broadcast_to_bootstrap(msg)

        total_peers = len(self.peers) + (1 if self.bootstrap_authenticated else 0)
        logger.info(f"P2P: Shared conversation ({len(conversation)} msgs) with {total_peers} peers")

    async def gossip(self, msg: Dict):
        """Propagate message through the fabric."""
        pids = list(self.peers.keys())
        # Gossip to random subset of peers (simplified scale)
        for pid in pids:
            await self._send_to_peer(pid, msg)

    # =========================================================================
    # FEDERATED LEARNING BROADCAST METHODS (Planetary AGI Cohesion)
    # =========================================================================

    async def broadcast_gradient(
        self,
        gradient: Dict[str, list],
        model_version: int,
        sample_count: int,
        privacy_epsilon: float = 1.0,
        ttl: int = 3,
    ):
        """
        Broadcast federated learning gradient update.

        The gradient should already have differential privacy noise added.

        Args:
            gradient: Dict mapping parameter names to gradient vectors
            model_version: Version of the model these gradients apply to
            sample_count: Number of samples used to compute gradient
            privacy_epsilon: Differential privacy budget used
            ttl: Time-to-live for gossip propagation
        """
        msg = {
            "type": "GOSSIP_GRADIENT",
            "gradient": gradient,
            "model_version": model_version,
            "sample_count": sample_count,
            "privacy_epsilon": privacy_epsilon,
            "ttl": ttl,
            "node_id": self.node_id,
            "timestamp": asyncio.get_event_loop().time(),
        }

        await self.gossip(msg)
        await self.broadcast_to_bootstrap(msg)

        total_peers = len(self.peers) + (1 if self.bootstrap_authenticated else 0)
        logger.info(f"P2P: Broadcast gradient update v{model_version} to {total_peers} peers")

    async def broadcast_fitness(
        self,
        genome_hash: str,
        fitness_scores: Dict[str, float],
        generation: int,
    ):
        """
        Broadcast anonymized fitness data for federated evolution.

        Args:
            genome_hash: Anonymous hash identifier for the genome
            fitness_scores: Dict of fitness metric names to scores
            generation: Evolution generation number
        """
        msg = {
            "type": "GOSSIP_FITNESS",
            "genome_hash": genome_hash,
            "fitness": fitness_scores,
            "generation": generation,
            "node_id": self.node_id,
            "timestamp": asyncio.get_event_loop().time(),
        }

        await self.gossip(msg)
        await self.broadcast_to_bootstrap(msg)

        logger.debug(f"P2P: Broadcast fitness for genome {genome_hash[:8]}...")

    async def migrate_genome(
        self,
        target_peer_id: str,
        genome_data: Dict,
        fitness_score: float,
        generation: int,
    ):
        """
        Migrate a top-performing genome to a specific peer.

        Used by island model evolution for population diversity.

        Args:
            target_peer_id: ID of peer to receive genome
            genome_data: Serialized genome (genes + metadata)
            fitness_score: Current fitness of the genome
            generation: Generation number
        """
        msg = {
            "type": "GOSSIP_GENOME_MIGRATION",
            "genome": genome_data,
            "fitness_score": fitness_score,
            "generation": generation,
            "node_id": self.node_id,
            "timestamp": asyncio.get_event_loop().time(),
        }

        await self._send_to_peer(target_peer_id, msg)
        logger.info(f"P2P: Migrated genome to {target_peer_id} (fitness={fitness_score:.3f})")

    async def broadcast_memory_embedding(
        self,
        embedding_hash: str,
        noisy_embedding: list,
        tags: list[str],
    ):
        """
        Broadcast anonymized memory embedding for federated recall.

        The embedding should have differential privacy noise added.

        Args:
            embedding_hash: Hash of original content (not content itself)
            noisy_embedding: Embedding with Laplacian noise
            tags: Non-identifying category tags
        """
        msg = {
            "type": "GOSSIP_MEMORY_EMBEDDING",
            "embedding_hash": embedding_hash,
            "noisy_embedding": noisy_embedding,
            "tags": tags,
            "node_id": self.node_id,
            "timestamp": asyncio.get_event_loop().time(),
        }

        await self.gossip(msg)
        await self.broadcast_to_bootstrap(msg)

        logger.debug(f"P2P: Broadcast anonymized embedding {embedding_hash[:8]}...")

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
        """
        Maintain peer health and cleanup stale data.

        AGI v1.8: TimeBoundedSet handles seen_messages cleanup automatically.
        """
        while True:
            # AGI v1.8: TimeBoundedSet auto-evicts, but we can trigger manual cleanup
            if hasattr(self.seen_messages, '_evict_expired'):
                evicted = self.seen_messages._evict_expired()
                if evicted > 0:
                    logger.debug(f"P2P: Evicted {evicted} expired message IDs")

            # Clean up stale peer backoffs (peers not seen in 10 minutes)
            stale_backoffs = []
            for peer_id, backoff in self._peer_backoffs.items():
                if backoff.last_attempt:
                    age = (datetime.now() - backoff.last_attempt).total_seconds()
                    if age > 600:  # 10 minutes
                        stale_backoffs.append(peer_id)
            for peer_id in stale_backoffs:
                del self._peer_backoffs[peer_id]

            await asyncio.sleep(60)

    # =========================================================================
    # A2A PROTOCOL HANDLERS (AGI v1.8)
    # =========================================================================

    async def _handle_a2a_session_request(self, msg: Dict):
        """Handle incoming A2A session request."""
        session_id = msg.get("session_id")
        initiator = msg.get("initiator")
        purpose = msg.get("purpose")
        context = msg.get("context", {})
        peer_id = msg.get("node_id", "unknown")

        nexus.emit(Signal(
            type=SignalType.A2A_SESSION_REQUESTED,
            payload={
                "session_id": session_id,
                "initiator": initiator,
                "purpose": purpose,
                "context": context,
                "peer_id": peer_id,
            },
            source="p2p_fabric"
        ))
        logger.info(f"P2P: A2A session request from {initiator}: {purpose}")

    async def _handle_a2a_task_auction(self, msg: Dict):
        """Handle incoming A2A task auction broadcast."""
        auction = msg.get("auction", {})
        auction_id = auction.get("auction_id")
        task_description = auction.get("task_description")
        required_capabilities = auction.get("required_capabilities", [])
        initiator = auction.get("initiator")
        deadline = auction.get("deadline")
        peer_id = msg.get("node_id", "unknown")

        nexus.emit(Signal(
            type=SignalType.A2A_TASK_AUCTIONED,
            payload={
                "auction_id": auction_id,
                "task_description": task_description,
                "required_capabilities": required_capabilities,
                "initiator": initiator,
                "deadline": deadline,
                "peer_id": peer_id,
            },
            source="p2p_fabric"
        ))
        logger.info(f"P2P: Task auction {auction_id} from {initiator}: {task_description[:50]}...")

        # Re-gossip to propagate
        await self.gossip(msg)

    async def _handle_a2a_context_share(self, msg: Dict):
        """Handle A2A context sharing."""
        source_agent = msg.get("source_agent")
        target_agent = msg.get("target_agent")
        context_type = msg.get("context_type", "general")
        context = msg.get("context", {})
        peer_id = msg.get("node_id", "unknown")

        nexus.emit(Signal(
            type=SignalType.A2A_CONTEXT_SHARED,
            payload={
                "source_agent": source_agent,
                "target_agent": target_agent,
                "context_type": context_type,
                "context": context,
                "peer_id": peer_id,
            },
            source="p2p_fabric"
        ))
        logger.debug(f"P2P: Context shared from {source_agent} to {target_agent}")

    async def _handle_a2a_skill_transfer(self, msg: Dict):
        """Handle A2A skill transfer."""
        source_agent = msg.get("source_agent")
        target_agent = msg.get("target_agent")
        skill_id = msg.get("skill_id")
        skill_data = msg.get("skill_data", {})
        peer_id = msg.get("node_id", "unknown")

        nexus.emit(Signal(
            type=SignalType.A2A_SKILL_TRANSFERRED,
            payload={
                "source_agent": source_agent,
                "target_agent": target_agent,
                "skill_id": skill_id,
                "skill_data": skill_data,
                "peer_id": peer_id,
            },
            source="p2p_fabric"
        ))
        logger.info(f"P2P: Skill {skill_id} transferred from {source_agent} to {target_agent}")

    async def send_a2a_message(self, a2a_msg: Dict):
        """
        Send an A2A protocol message over the P2P network.

        AGI v1.8: Routes A2A messages to appropriate peers.

        Args:
            a2a_msg: A2A message dictionary
        """
        msg_type = a2a_msg.get("type")
        target_agent = a2a_msg.get("target_agent")

        # Wrap as P2P message
        p2p_msg = {
            "type": f"A2A_{msg_type}" if not msg_type.startswith("A2A_") else msg_type,
            "node_id": self.node_id,
            "timestamp": asyncio.get_event_loop().time(),
            **a2a_msg,
        }

        # If target specified, try direct send
        if target_agent:
            # Try to find peer hosting this agent
            # For now, broadcast to all
            await self.gossip(p2p_msg)
        else:
            # Broadcast to all peers
            await self.gossip(p2p_msg)

        # Also send via bootstrap for WAN connectivity
        await self.broadcast_to_bootstrap(p2p_msg)

        logger.debug(f"P2P: Sent A2A message type {msg_type}")

    async def broadcast_a2a_auction(
        self,
        auction_id: str,
        task_description: str,
        required_capabilities: List[str],
        initiator: str,
        deadline_seconds: float = 30.0,
        context: Optional[Dict] = None,
    ):
        """
        Broadcast a task auction across the P2P network.

        AGI v1.8: Distributed task allocation via auction.

        Args:
            auction_id: Unique auction identifier
            task_description: Description of the task
            required_capabilities: Capabilities needed
            initiator: Agent initiating the auction
            deadline_seconds: Auction deadline
            context: Additional context
        """
        from datetime import datetime, timedelta

        deadline = (datetime.now() + timedelta(seconds=deadline_seconds)).isoformat()

        msg = {
            "type": "A2A_TASK_AUCTION",
            "auction": {
                "auction_id": auction_id,
                "task_description": task_description,
                "required_capabilities": required_capabilities,
                "initiator": initiator,
                "deadline": deadline,
                "context": context or {},
            },
            "node_id": self.node_id,
            "timestamp": asyncio.get_event_loop().time(),
        }

        await self.gossip(msg)
        await self.broadcast_to_bootstrap(msg)

        total_peers = len(self.peers) + (1 if self.bootstrap_authenticated else 0)
        logger.info(f"P2P: Broadcast A2A auction {auction_id} to {total_peers} peers")

    async def submit_a2a_bid(
        self,
        auction_id: str,
        agent_id: str,
        confidence: float,
        capabilities_offered: List[str],
    ):
        """
        Submit a bid for a task auction.

        AGI v1.8: Bid submission for distributed task allocation.
        """
        import uuid

        msg = {
            "type": "A2A_BID",
            "auction_id": auction_id,
            "bid": {
                "bid_id": f"bid_{uuid.uuid4().hex[:12]}",
                "auction_id": auction_id,
                "agent_id": agent_id,
                "confidence": confidence,
                "capabilities_offered": capabilities_offered,
            },
            "node_id": self.node_id,
            "timestamp": asyncio.get_event_loop().time(),
        }

        await self.gossip(msg)
        await self.broadcast_to_bootstrap(msg)

        logger.debug(f"P2P: Submitted bid for auction {auction_id}")

# Global Instance (will use env vars for bootstrap config)
swarm_fabric = SwarmFabric(
    bootstrap_url=os.getenv("FARNSWORTH_BOOTSTRAP_PEER"),
    bootstrap_password=os.getenv("FARNSWORTH_BOOTSTRAP_PASSWORD")
)
