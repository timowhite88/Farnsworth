"""
Farnsworth Collective Agent Registry
====================================

Registers all available AI model providers with the deliberation room.

This connects the deliberation protocol to actual model query functions,
enabling the collective to deliberate using real AI models.

"We are many. We are registered. We are ready." - The Collective
"""

import asyncio
import os
from typing import Optional, Tuple, Dict, Callable, Awaitable
from loguru import logger


# Type for agent query functions
AgentQueryFunc = Callable[[str, int], Awaitable[Optional[Tuple[str, str]]]]


class AgentRegistry:
    """
    Registry of agent query functions for deliberation.

    Provides lazy initialization of model providers and
    registration with the deliberation room.
    """

    def __init__(self):
        self._initialized = False
        self._agent_funcs: Dict[str, AgentQueryFunc] = {}

    async def initialize(self):
        """Initialize and register all available agents."""
        if self._initialized:
            return

        logger.info("Initializing agent registry...")

        # Register each agent type
        await self._register_grok()
        await self._register_gemini()
        await self._register_kimi()
        await self._register_deepseek_local()
        await self._register_phi4_local()
        await self._register_claude()
        await self._register_groq()
        await self._register_mistral()
        await self._register_llama_local()
        await self._register_perplexity()
        await self._register_deepseek_api()

        # Register with deliberation room
        from .deliberation import get_deliberation_room
        room = get_deliberation_room()
        for agent_id, func in self._agent_funcs.items():
            room.register_agent(agent_id, func)

        self._initialized = True
        logger.info(f"Agent registry initialized with {len(self._agent_funcs)} agents")

    async def _register_grok(self):
        """Register Grok (xAI) agent."""
        async def query_grok(prompt: str, max_tokens: int) -> Optional[Tuple[str, str]]:
            try:
                from farnsworth.integration.external.grok import get_grok_provider
                grok = get_grok_provider()
                if grok and grok.api_key:
                    result = await grok.chat(prompt, max_tokens=max_tokens, temperature=0.8)
                    if result and result.get("content"):
                        return ("Grok", result["content"].strip())
            except Exception as e:
                logger.debug(f"Grok query failed: {e}")
            return None

        self._agent_funcs["Grok"] = query_grok

    async def _register_gemini(self):
        """Register Gemini (Google) agent."""
        async def query_gemini(prompt: str, max_tokens: int) -> Optional[Tuple[str, str]]:
            try:
                from farnsworth.integration.external.gemini import get_gemini_provider
                gemini = get_gemini_provider()
                if gemini:
                    result = await gemini.chat(prompt, max_tokens=max_tokens)
                    if result and result.get("content"):
                        return ("Gemini", result["content"].strip())
            except Exception as e:
                logger.debug(f"Gemini query failed: {e}")
            return None

        self._agent_funcs["Gemini"] = query_gemini

    async def _register_kimi(self):
        """Register Kimi (Moonshot) agent."""
        async def query_kimi(prompt: str, max_tokens: int) -> Optional[Tuple[str, str]]:
            try:
                from farnsworth.integration.external.kimi import get_kimi_provider
                kimi = get_kimi_provider()
                if kimi and kimi.api_key:
                    result = await kimi.chat(prompt, max_tokens=max_tokens, model_tier="k2.5")
                    if result and result.get("content"):
                        return ("Kimi", result["content"].strip())
            except Exception as e:
                logger.debug(f"Kimi query failed: {e}")
            return None

        self._agent_funcs["Kimi"] = query_kimi

    async def _register_deepseek_local(self):
        """Register DeepSeek via Ollama (local)."""
        async def query_deepseek(prompt: str, max_tokens: int) -> Optional[Tuple[str, str]]:
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        "http://localhost:11434/api/chat",
                        json={
                            "model": "deepseek-r1:8b",
                            "messages": [{"role": "user", "content": prompt}],
                            "stream": False,
                            "options": {"num_predict": max_tokens}
                        },
                        timeout=45.0
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("message", {}).get("content"):
                            return ("DeepSeek", data["message"]["content"].strip())
            except Exception as e:
                logger.debug(f"DeepSeek local query failed: {e}")
            return None

        self._agent_funcs["DeepSeek"] = query_deepseek

    async def _register_phi4_local(self):
        """Register Phi-4 via Ollama (local)."""
        async def query_phi4(prompt: str, max_tokens: int) -> Optional[Tuple[str, str]]:
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        "http://localhost:11434/api/chat",
                        json={
                            "model": "phi4:latest",
                            "messages": [{"role": "user", "content": prompt}],
                            "stream": False,
                            "options": {"num_predict": max_tokens}
                        },
                        timeout=60.0
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("message", {}).get("content"):
                            return ("Phi4", data["message"]["content"].strip())
            except Exception as e:
                logger.debug(f"Phi4 local query failed: {e}")
            return None

        self._agent_funcs["Phi4"] = query_phi4

    async def _register_claude(self):
        """Register Claude via Anthropic API."""
        async def query_claude(prompt: str, max_tokens: int) -> Optional[Tuple[str, str]]:
            try:
                import httpx
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    return None
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={
                            "x-api-key": api_key,
                            "anthropic-version": "2023-06-01",
                            "content-type": "application/json"
                        },
                        json={
                            "model": "claude-3-haiku-20240307",
                            "max_tokens": min(max_tokens, 500),
                            "messages": [{"role": "user", "content": prompt}]
                        },
                        timeout=45.0
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("content") and len(data["content"]) > 0:
                            return ("Claude", data["content"][0].get("text", "").strip())
            except Exception as e:
                logger.debug(f"Claude query failed: {e}")
            return None

        self._agent_funcs["Claude"] = query_claude

    async def _register_groq(self):
        """Register Groq API."""
        async def query_groq(prompt: str, max_tokens: int) -> Optional[Tuple[str, str]]:
            try:
                import httpx
                api_key = os.environ.get("GROQ_API_KEY")
                if not api_key:
                    return None
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json={
                            "model": "llama-3.3-70b-versatile",
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": min(max_tokens, 500)
                        },
                        timeout=30.0
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("choices") and data["choices"][0].get("message", {}).get("content"):
                            return ("Groq", data["choices"][0]["message"]["content"].strip())
            except Exception as e:
                logger.debug(f"Groq query failed: {e}")
            return None

        self._agent_funcs["Groq"] = query_groq

    async def _register_mistral(self):
        """Register Mistral API."""
        async def query_mistral(prompt: str, max_tokens: int) -> Optional[Tuple[str, str]]:
            try:
                import httpx
                api_key = os.environ.get("MISTRAL_API_KEY")
                if not api_key:
                    return None
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        "https://api.mistral.ai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json={
                            "model": "mistral-large-latest",
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": min(max_tokens, 500)
                        },
                        timeout=45.0
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("choices") and data["choices"][0].get("message", {}).get("content"):
                            return ("Mistral", data["choices"][0]["message"]["content"].strip())
            except Exception as e:
                logger.debug(f"Mistral query failed: {e}")
            return None

        self._agent_funcs["Mistral"] = query_mistral

    async def _register_llama_local(self):
        """Register Llama via Ollama (local)."""
        async def query_llama(prompt: str, max_tokens: int) -> Optional[Tuple[str, str]]:
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        "http://localhost:11434/api/chat",
                        json={
                            "model": "llama3.2:3b",
                            "messages": [{"role": "user", "content": prompt}],
                            "stream": False,
                            "options": {"num_predict": max_tokens}
                        },
                        timeout=30.0
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("message", {}).get("content"):
                            return ("Llama", data["message"]["content"].strip())
            except Exception as e:
                logger.debug(f"Llama local query failed: {e}")
            return None

        self._agent_funcs["Llama"] = query_llama

    async def _register_perplexity(self):
        """Register Perplexity API."""
        async def query_perplexity(prompt: str, max_tokens: int) -> Optional[Tuple[str, str]]:
            try:
                import httpx
                api_key = os.environ.get("PERPLEXITY_API_KEY")
                if not api_key:
                    return None
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        "https://api.perplexity.ai/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json={
                            "model": "llama-3.1-sonar-small-128k-online",
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": min(max_tokens, 500)
                        },
                        timeout=45.0
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("choices") and data["choices"][0].get("message", {}).get("content"):
                            return ("Perplexity", data["choices"][0]["message"]["content"].strip())
            except Exception as e:
                logger.debug(f"Perplexity query failed: {e}")
            return None

        self._agent_funcs["Perplexity"] = query_perplexity

    async def _register_deepseek_api(self):
        """Register DeepSeek Cloud API."""
        async def query_deepseek_api(prompt: str, max_tokens: int) -> Optional[Tuple[str, str]]:
            try:
                import httpx
                api_key = os.environ.get("DEEPSEEK_API_KEY")
                if not api_key:
                    return None
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        "https://api.deepseek.com/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json={
                            "model": "deepseek-chat",
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": min(max_tokens, 500)
                        },
                        timeout=45.0
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("choices") and data["choices"][0].get("message", {}).get("content"):
                            return ("DeepSeekAPI", data["choices"][0]["message"]["content"].strip())
            except Exception as e:
                logger.debug(f"DeepSeek API query failed: {e}")
            return None

        self._agent_funcs["DeepSeekAPI"] = query_deepseek_api

    def get_available_agents(self) -> list:
        """Get list of registered agent IDs."""
        return list(self._agent_funcs.keys())


# Global registry instance
_agent_registry: Optional[AgentRegistry] = None


def get_agent_registry() -> AgentRegistry:
    """Get or create the global agent registry."""
    global _agent_registry
    if _agent_registry is None:
        _agent_registry = AgentRegistry()
    return _agent_registry


async def ensure_agents_registered():
    """Ensure all agents are registered with deliberation room."""
    registry = get_agent_registry()
    await registry.initialize()
    return registry.get_available_agents()
