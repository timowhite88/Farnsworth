"""
Farnsworth Collective Session Manager
=====================================

Manages multiple concurrent collective instances for different use cases:
- Website chat: Interactive user conversations
- Grok thread: Public X/Twitter deliberations
- Autonomous tasks: Background processing

Each session type has its own configuration and agent mix.

"We think in many places at once." - The Collective
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from loguru import logger

from .deliberation import DeliberationRoom, DeliberationResult, get_deliberation_room


@dataclass
class CollectiveConfig:
    """Configuration for a collective session."""
    agents: List[str]
    deliberation_rounds: int = 2
    tool_awareness: bool = True
    max_tokens: int = 5000
    timeout: float = 120.0
    media_bias: float = 0.4  # Probability of including media
    require_consensus: bool = False

    # Context-specific settings
    x_content: bool = False  # Optimize for Twitter/X format
    technical_depth: str = "medium"  # "low", "medium", "high"


@dataclass
class CollectiveSession:
    """An active collective session."""
    session_id: str
    session_type: str
    config: CollectiveConfig
    created_at: datetime
    last_active: datetime
    deliberation_count: int = 0
    total_turns: int = 0
    history: List[DeliberationResult] = field(default_factory=list)

    def record_deliberation(self, result: DeliberationResult):
        """Record a completed deliberation."""
        self.deliberation_count += 1
        self.total_turns += len(result.participating_agents)
        self.last_active = datetime.now()
        # AGI v1.8: Keep last 100 deliberations in history (increased from 10)
        # More context enables better pattern recognition across sessions
        self.history.append(result)
        if len(self.history) > 100:
            self.history = self.history[-100:]


class CollectiveSessionManager:
    """
    Manage multiple concurrent collective instances.

    Provides session management for different contexts:
    - website_chat: User conversations on ai.farnsworth.cloud
    - grok_thread: Public conversations with @grok on X
    - autonomous_task: Background task processing
    """

    # Pre-defined session configurations
    # Local models: DeepSeek (Ollama), Phi4 (Ollama), Llama (Ollama)
    # API models: Grok, Gemini, Kimi, Claude, Groq, Mistral, Perplexity, DeepSeekAPI
    DEFAULT_CONFIGS = {
        "website_chat": CollectiveConfig(
            # Include both local and API models for website chat
            # Local: DeepSeek, Phi4 (fast, private, no API cost)
            # API: Grok, Gemini, Kimi, Claude (high quality)
            agents=["Grok", "Gemini", "DeepSeek", "Phi4", "Kimi", "Claude"],
            deliberation_rounds=2,
            tool_awareness=True,
            max_tokens=5000,
            media_bias=0.3,
            technical_depth="medium",
        ),
        "grok_thread": CollectiveConfig(
            # More agents for public X conversations
            agents=["Grok", "Gemini", "Kimi", "DeepSeek", "Phi4", "Claude", "Groq"],
            deliberation_rounds=3,  # More thorough for public posts
            tool_awareness=True,
            max_tokens=5000,
            media_bias=0.6,  # Higher chance of media for engagement
            x_content=True,  # Optimize for Twitter format
            technical_depth="high",
        ),
        "autonomous_task": CollectiveConfig(
            # Fast local-first for background tasks
            agents=["DeepSeek", "Phi4", "Gemini", "Claude"],
            deliberation_rounds=1,  # Fast for background tasks
            tool_awareness=True,
            max_tokens=3000,
            media_bias=0.2,
            technical_depth="high",
        ),
        "quick_response": CollectiveConfig(
            # Ultra-fast with local models prioritized
            agents=["DeepSeek", "Phi4", "Grok"],
            deliberation_rounds=1,
            tool_awareness=False,
            max_tokens=2000,
            timeout=30.0,
            media_bias=0.1,
            technical_depth="low",
        ),
    }

    def __init__(self):
        self.sessions: Dict[str, CollectiveSession] = {}
        self._deliberation_room = get_deliberation_room()
        self._lock = asyncio.Lock()
        self._agents_initialized = False

        logger.info("CollectiveSessionManager initialized")

    async def _ensure_agents_registered(self):
        """Ensure all agents are registered with the deliberation room."""
        if self._agents_initialized:
            return

        try:
            from .agent_registry import ensure_agents_registered
            agents = await ensure_agents_registered()
            logger.info(f"Registered {len(agents)} agents for deliberation: {agents}")
            self._agents_initialized = True
        except Exception as e:
            logger.warning(f"Could not register agents: {e}")

    async def get_or_create_session(
        self,
        session_type: str,
        session_id: str = None
    ) -> CollectiveSession:
        """
        Get existing session or create a new one.

        Args:
            session_type: Type of session (website_chat, grok_thread, etc.)
            session_id: Optional specific session ID

        Returns:
            CollectiveSession instance
        """
        async with self._lock:
            # Generate session ID if not provided
            if session_id is None:
                session_id = f"{session_type}_{str(uuid.uuid4())[:8]}"

            # Return existing session if found
            if session_id in self.sessions:
                session = self.sessions[session_id]
                session.last_active = datetime.now()
                return session

            # Get configuration for this session type
            config = self.DEFAULT_CONFIGS.get(
                session_type,
                self.DEFAULT_CONFIGS["website_chat"]  # Default fallback
            )

            # Create new session
            session = CollectiveSession(
                session_id=session_id,
                session_type=session_type,
                config=config,
                created_at=datetime.now(),
                last_active=datetime.now(),
            )

            self.sessions[session_id] = session
            logger.info(f"Created new {session_type} session: {session_id}")

            return session

    async def deliberate_in_session(
        self,
        session_type: str,
        prompt: str,
        context: Dict[str, Any] = None,
        session_id: str = None,
    ) -> DeliberationResult:
        """
        Run deliberation in the appropriate session.

        Args:
            session_type: Type of session to use
            prompt: The prompt to deliberate on
            context: Optional context dictionary
            session_id: Optional specific session ID

        Returns:
            DeliberationResult from the deliberation
        """
        # Ensure agents are registered
        await self._ensure_agents_registered()

        # Get or create session
        session = await self.get_or_create_session(session_type, session_id)
        config = session.config

        # Build tool context if enabled
        tool_context = None
        if config.tool_awareness:
            from .tool_awareness import get_tool_awareness
            tool_ctx = get_tool_awareness()
            tool_context = tool_ctx.get_tool_context_for_agents()

        # Run deliberation
        result = await self._deliberation_room.deliberate(
            prompt=prompt,
            agents=config.agents,
            max_rounds=config.deliberation_rounds,
            require_consensus=config.require_consensus,
            max_tokens=config.max_tokens,
            tool_context=tool_context,
            timeout=config.timeout,
        )

        # Record the deliberation to session (in-memory)
        session.record_deliberation(result)

        # AGI v1.8: Record to persistent DialogueMemory for learning
        # This triggers the full learning pipeline:
        # 1. DialogueMemory.store_exchange() → saves to exchanges.json
        # 2. _archive_to_long_term() → stores to ArchivalMemory with embeddings
        try:
            from .dialogue_memory import record_deliberation as persist_deliberation
            asyncio.create_task(persist_deliberation(result, session_type=session.session_type))
            logger.debug(f"Queued deliberation {result.deliberation_id} for persistent storage")
        except Exception as e:
            logger.warning(f"Could not persist deliberation: {e}")

        return result

    async def deliberate_with_tools(
        self,
        session_type: str,
        prompt: str,
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Run deliberation and handle tool decisions.

        Returns both the response and any tool decisions made by the collective.

        Args:
            session_type: Type of session
            prompt: The prompt to deliberate on
            context: Optional context

        Returns:
            Dict with 'response', 'tool_decision', 'deliberation_summary', etc.
        """
        result = await self.deliberate_in_session(session_type, prompt, context)

        # Analyze deliberation for tool suggestions
        from .tool_awareness import get_tool_awareness
        tool_awareness = get_tool_awareness()
        tool_decision = await tool_awareness.analyze_deliberation_for_tools(result)

        return {
            "response": result.final_response,
            "tool_decision": tool_decision,
            "deliberation_summary": result.get_summary(),
            "participating_agents": result.participating_agents,
            "winning_agent": result.winning_agent,
            "consensus_reached": result.consensus_reached,
            "vote_breakdown": result.vote_breakdown,
        }

    def get_session_stats(self, session_id: str = None) -> Dict[str, Any]:
        """Get statistics for a session or all sessions."""
        if session_id:
            session = self.sessions.get(session_id)
            if session:
                return {
                    "session_id": session.session_id,
                    "type": session.session_type,
                    "deliberations": session.deliberation_count,
                    "total_turns": session.total_turns,
                    "created": session.created_at.isoformat(),
                    "last_active": session.last_active.isoformat(),
                }
            return {}

        # All sessions
        return {
            "total_sessions": len(self.sessions),
            "sessions": {
                sid: {
                    "type": s.session_type,
                    "deliberations": s.deliberation_count,
                    "last_active": s.last_active.isoformat(),
                }
                for sid, s in self.sessions.items()
            }
        }

    async def cleanup_stale_sessions(self, max_age_hours: int = 24):
        """Remove sessions older than max_age_hours."""
        async with self._lock:
            from datetime import timedelta
            cutoff = datetime.now() - timedelta(hours=max_age_hours)

            stale = [
                sid for sid, session in self.sessions.items()
                if session.last_active < cutoff
            ]

            for sid in stale:
                del self.sessions[sid]
                logger.info(f"Cleaned up stale session: {sid}")

            return len(stale)


# Global session manager instance
_session_manager: Optional[CollectiveSessionManager] = None


def get_session_manager() -> CollectiveSessionManager:
    """Get or create the global session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = CollectiveSessionManager()
    return _session_manager


async def website_deliberate(prompt: str, context: Dict = None) -> DeliberationResult:
    """Quick helper for website chat deliberation."""
    manager = get_session_manager()
    return await manager.deliberate_in_session("website_chat", prompt, context)


async def grok_thread_deliberate(prompt: str, context: Dict = None) -> DeliberationResult:
    """Quick helper for Grok thread deliberation."""
    manager = get_session_manager()
    return await manager.deliberate_in_session("grok_thread", prompt, context)


async def autonomous_deliberate(prompt: str, context: Dict = None) -> DeliberationResult:
    """Quick helper for autonomous task deliberation."""
    manager = get_session_manager()
    return await manager.deliberate_in_session("autonomous_task", prompt, context)
