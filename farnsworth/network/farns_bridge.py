"""
FARNS Bridge — Connects FARNS mesh to existing Farnsworth systems.

Bridges:
  - PersistentAgent shadow agents → remote bots via FARNS
  - DialogueBus → FARNS mesh (remote agents appear in bus)
  - AgentSpawner → FARNS-aware fallback chains
"""
import asyncio
from typing import Optional, Dict, List
from loguru import logger

from .farns_client import get_farns_client, FARNSClient
from .farns_node import get_farns_node


class FARNSRemoteProvider:
    """
    Provider that routes queries through the FARNS mesh to a remote bot.

    Drop-in replacement for OllamaWithToolsProvider / API providers.
    Used by PersistentAgent for remote bot access.
    """

    def __init__(self, bot_name: str):
        self.bot_name = bot_name
        self.system_prompt = None
        self._client = get_farns_client()

    async def chat(self, prompt: str, max_tokens: int = 4000) -> Dict:
        """Query the remote bot via FARNS mesh."""
        # Prepend system prompt if set
        full_prompt = prompt
        if self.system_prompt:
            full_prompt = f"{self.system_prompt}\n\n{prompt}"

        response = await self._client.query(self.bot_name, full_prompt, max_tokens)
        return {"content": response or ""}

    async def chat_stream(self, prompt: str, max_tokens: int = 4000):
        """Stream from the remote bot. Yields content chunks."""
        full_prompt = prompt
        if self.system_prompt:
            full_prompt = f"{self.system_prompt}\n\n{prompt}"

        async for chunk in self._client.stream(self.bot_name, full_prompt, max_tokens):
            yield chunk


def register_farns_bots_as_shadow_agents():
    """
    Register all remote FARNS bots as shadow agents in the PersistentAgent system.

    This makes remote bots callable via:
        from farnsworth.core.collective.persistent_agent import call_shadow_agent
        result = await call_shadow_agent("qwen3-coder-next", "Write a parser")
    """
    try:
        from farnsworth.core.collective.persistent_agent import (
            AGENT_CONFIGS, PersistentAgent, _SHADOW_AGENTS, _SHADOW_LOCK,
        )

        client = get_farns_client()
        remote_bots = client.list_remote_bots()

        for bot_name in remote_bots:
            safe_name = bot_name.replace("-", "_").replace(":", "_").replace(".", "_")

            if safe_name not in AGENT_CONFIGS:
                AGENT_CONFIGS[safe_name] = {
                    "provider": "farns_remote",
                    "personality": f"Remote bot via FARNS mesh ({bot_name})",
                    "thinking_interval": 60,
                    "specialties": ["remote", "farns"],
                    "farns_bot_name": bot_name,
                }
                logger.info(f"Registered FARNS remote bot as shadow agent: {safe_name} → {bot_name}")

    except Exception as e:
        logger.warning(f"Could not register FARNS bots as shadow agents: {e}")


def register_farns_bots_with_spawner():
    """
    Register remote FARNS bots with the AgentSpawner system.
    """
    try:
        from farnsworth.core.agent_spawner import get_spawner, TaskType

        spawner = get_spawner()
        client = get_farns_client()
        remote_bots = client.list_remote_bots()

        for bot_name in remote_bots:
            display_name = bot_name.replace("-", " ").replace("_", " ").title().replace(" ", "")

            if display_name not in spawner.agent_capabilities:
                spawner.agent_capabilities[display_name] = [
                    TaskType.CHAT, TaskType.DEVELOPMENT, TaskType.RESEARCH,
                ]
                spawner.max_instances[display_name] = 2
                # Add to fallback chains
                spawner.fallback_chains[display_name] = ["DeepSeek", "ClaudeOpus"]
                logger.info(f"Registered FARNS bot with spawner: {display_name}")

    except Exception as e:
        logger.warning(f"Could not register FARNS bots with spawner: {e}")


async def start_farns_bridge():
    """
    Initialize the FARNS bridge.

    Call this after the FARNS node is started and connected to peers.
    Registers remote bots with all Farnsworth systems.
    """
    # Wait a moment for peer discovery
    await asyncio.sleep(5)

    register_farns_bots_as_shadow_agents()
    register_farns_bots_with_spawner()

    # Register local shadow agents as FARNS bots (reverse bridge)
    node = get_farns_node()
    if node:
        try:
            from farnsworth.core.collective.persistent_agent import (
                call_shadow_agent, get_shadow_agents,
            )
            for agent_id in get_shadow_agents():
                async def make_fn(aid=agent_id):
                    async def query_fn(prompt: str, max_tokens: int = 4000) -> str:
                        result = await call_shadow_agent(aid, prompt, max_tokens)
                        return result[1] if result else ""
                    return query_fn

                node.register_bot(agent_id, await make_fn())
                logger.debug(f"Registered shadow agent '{agent_id}' as FARNS bot")

        except Exception as e:
            logger.debug(f"Could not register shadow agents as FARNS bots: {e}")

    logger.info("FARNS bridge initialized")
