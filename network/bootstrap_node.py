"""
Farnsworth Bootstrap Node (Relay Server)
----------------------------------------

"The intergalactic phone switchboard!"

This script runs a lightweight specialized server that introduces Farnsworth nodes 
to each other across the internet (WAN), enabling Planetary Memory sharing 
outside of a local network.

Usage:
    python network/bootstrap_node.py --port 8888

Clients connect by setting:
    FARNSWORTH_BOOTSTRAP_PEER=ws://<YOUR_SERVER_IP>:8888
"""

import asyncio
import json
import logging
import argparse
from typing import Set, Dict
import websockets
from websockets.server import WebSocketServerProtocol

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [BOOTSTRAP] - %(message)s")
logger = logging.getLogger("Bootstrap")

class BootstrapServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8888):
        self.host = host
        self.port = port
        self.peers: Set[WebSocketServerProtocol] = set()
        
        # Peer Info Map: {websocket: {"id": "node_x", "ip": "1.2.3.4"}}
        self.peer_metadata: Dict[WebSocketServerProtocol, dict] = {}

    async def register(self, websocket: WebSocketServerProtocol):
        """Register a new connection."""
        self.peers.add(websocket)
        logger.info(f"New Peer Connected: {websocket.remote_address}")

    async def unregister(self, websocket: WebSocketServerProtocol):
        """Unregister a connection."""
        self.peers.remove(websocket)
        if websocket in self.peer_metadata:
            data = self.peer_metadata[websocket]
            logger.info(f"Peer Disconnected: {data.get('id', 'Unknown')} ({websocket.remote_address})")
            del self.peer_metadata[websocket]

    async def handler(self, websocket: WebSocketServerProtocol, path: str):
        """Handle incoming messages."""
        await self.register(websocket)
        try:
            async for message in websocket:
                await self.process_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister(websocket)

    async def process_message(self, sender: WebSocketServerProtocol, message: str):
        """
        Process signaling messages.
        
        Types:
        - HELLO: Node announcing itself.
        - GOSSIP: A Planetary Memory update (Skill Vector).
        - PEER_REQ: Requesting list of peers.
        """
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "HELLO":
                # Save metadata
                self.peer_metadata[sender] = {
                    "id": data.get("node_id"),
                    "version": data.get("version"),
                    "capabilities": data.get("capabilities", [])
                }
                logger.info(f"Handshake: Node {data.get('node_id')} (v{data.get('version')})")
                
                # Send back welcome with known peers count
                await sender.send(json.dumps({
                    "type": "WELCOME",
                    "peer_count": len(self.peers)
                }))

            elif msg_type == "GOSSIP":
                # Broadcast memory updates to ALL other peers (Floodsub)
                # In prod, use intelligent routing (Kademlia/Gossipsub)
                skill_id = data.get("skill_id", "unknown")
                logger.debug(f"Relaying GOSSIP (Skill: {skill_id}) to {len(self.peers)-1} peers")
                
                tasks = []
                for peer in self.peers:
                    if peer != sender:
                        tasks.append(peer.send(message))
                
                if tasks:
                    await asyncio.gather(*tasks)

            elif msg_type == "PEER_REQ":
                # Send a list of active peers (for direct P2P hole-punching)
                peer_list = [
                    {"id": meta["id"], "ip": ws.remote_address[0]}
                    for ws, meta in self.peer_metadata.items()
                    if ws != sender
                ]
                await sender.send(json.dumps({
                    "type": "PEER_RES",
                    "peers": peer_list
                }))

        except json.JSONDecodeError:
            logger.error("Received malformed JSON")
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    async def start(self):
        """Start the WebSocket server."""
        logger.info(f"🚀 Starting Bootstrap Node on {self.host}:{self.port}")
        logger.info("Ready to relay Planetary Memory...")
        
        async with websockets.serve(self.handler, self.host, self.port):
            await asyncio.Future()  # Run forever

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Farnsworth Relay Node")
    parser.add_argument("--port", type=int, default=8888, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    
    args = parser.parse_args()
    
    try:
        asyncio.run(BootstrapServer(host=args.host, port=args.port).start())
    except KeyboardInterrupt:
        logger.info("Bootstrap Node Stopped.")
