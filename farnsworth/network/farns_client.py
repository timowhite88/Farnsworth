"""
FARNS Client — Async client for querying remote bots via the FARNS mesh.

Simple API for the rest of the codebase:

    from farnsworth.network.farns_client import FARNSClient

    client = FARNSClient()
    await client.connect()

    # Complete response
    response = await client.query("qwen3-coder-next", "Write a parser")

    # Streaming (fast path)
    async for chunk in client.stream("qwen3-coder-next", "Write a parser"):
        print(chunk, end="", flush=True)
"""
import asyncio
from typing import Optional, AsyncIterator, Dict, List
from loguru import logger

from .farns_node import get_farns_node, FARNSNode


class FARNSClient:
    """
    High-level async client for FARNS mesh queries.

    Uses the local FARNS node to route requests to remote bots.
    If no node is running, attempts to connect directly.
    """

    def __init__(self, node: Optional[FARNSNode] = None):
        self._node = node

    def _get_node(self) -> Optional[FARNSNode]:
        """Get the FARNS node (lazy)."""
        if self._node is None:
            self._node = get_farns_node()
        return self._node

    async def query(self, bot_name: str, prompt: str,
                    max_tokens: int = 4000,
                    timeout: float = 120.0) -> Optional[str]:
        """
        Query a remote bot and get the complete response.

        Args:
            bot_name: Name of the bot (e.g., "qwen3-coder-next")
            prompt: The prompt to send
            max_tokens: Max response tokens
            timeout: Max wait time

        Returns:
            Response string or None if unavailable
        """
        node = self._get_node()
        if not node:
            logger.warning("No FARNS node available")
            return None

        # Check if bot is local first (faster)
        if bot_name in node.get_local_bots():
            local_bots = node._local_bots
            if bot_name in local_bots:
                return await local_bots[bot_name](prompt, max_tokens)

        # Route to remote
        return await node.query_remote_bot(bot_name, prompt, max_tokens, timeout)

    async def stream(self, bot_name: str, prompt: str,
                     max_tokens: int = 4000) -> AsyncIterator[str]:
        """
        Stream dialogue from a remote bot. Yields chunks.

        This is the FAST PATH — minimum latency per chunk.

        Usage:
            async for chunk in client.stream("qwen3-coder-next", prompt):
                print(chunk, end="", flush=True)
        """
        node = self._get_node()
        if not node:
            return

        async for chunk in node.stream_remote_bot(bot_name, prompt, max_tokens):
            yield chunk

    def list_bots(self) -> Dict[str, str]:
        """
        List all available bots with their location.

        Returns:
            Dict of bot_name → node_name
        """
        node = self._get_node()
        if not node:
            return {}
        return node.get_all_bots()

    def list_remote_bots(self) -> List[str]:
        """List only remote (non-local) bots."""
        node = self._get_node()
        if not node:
            return []
        local = set(node.get_local_bots())
        all_bots = node.get_all_bots()
        return [name for name in all_bots if name not in local]

    def status(self) -> Dict:
        """Get FARNS mesh status."""
        node = self._get_node()
        if not node:
            return {"error": "No FARNS node running"}
        return node.get_status()


# ── Global client singleton ───────────────────────────────────

_farns_client: Optional[FARNSClient] = None


def get_farns_client() -> FARNSClient:
    """Get or create the global FARNS client."""
    global _farns_client
    if _farns_client is None:
        _farns_client = FARNSClient()
    return _farns_client
