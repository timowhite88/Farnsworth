"""
Farnsworth Collective Dialogue Memory
=====================================

Store and retrieve agent-to-agent conversations and deliberation history.

This module provides:
- Storage of deliberation exchanges
- Agent turn history tracking
- Consensus logging
- Context retrieval for agents

"We remember every thought we've shared." - The Collective
"""

import json
import asyncio
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger

from .deliberation import AgentTurn, DeliberationResult, DeliberationRound


@dataclass
class DeliberationExchange:
    """Complete deliberation session record for storage."""
    exchange_id: str
    timestamp: str
    prompt: str
    participating_agents: List[str]
    rounds: Dict[str, List[Dict]]  # Serialized AgentTurn dicts
    final_response: str
    winning_agent: str
    vote_breakdown: Dict[str, float]
    tool_used: Optional[str] = None
    consensus_reached: bool = False
    duration_ms: float = 0
    session_type: Optional[str] = None  # website_chat, grok_thread, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


class DialogueMemory:
    """
    Store and retrieve agent-to-agent conversations.

    Provides:
    - Persistent storage of deliberation exchanges
    - Agent history tracking
    - Context retrieval for future deliberations
    """

    def __init__(self, storage_path: Optional[Path] = None):
        if storage_path is None:
            import os
            if os.path.exists("/workspace/farnsworth_memory"):
                storage_path = Path("/workspace/farnsworth_memory/dialogue")
            else:
                storage_path = Path("data/dialogue_memory")
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory caches
        self._exchanges: List[DeliberationExchange] = []
        self._agent_stats: Dict[str, Dict[str, Any]] = {}

        # Load existing data
        self._load_state()

        logger.info(f"DialogueMemory initialized with {len(self._exchanges)} exchanges")

    def _load_state(self):
        """Load persisted dialogue memory."""
        try:
            exchanges_file = self.storage_path / "exchanges.json"
            if exchanges_file.exists():
                data = json.loads(exchanges_file.read_text())
                self._exchanges = [
                    DeliberationExchange(**e) for e in data
                ]

            stats_file = self.storage_path / "agent_stats.json"
            if stats_file.exists():
                self._agent_stats = json.loads(stats_file.read_text())

        except Exception as e:
            logger.warning(f"Could not load dialogue memory: {e}")

    def _save_state(self):
        """Persist dialogue memory to storage."""
        try:
            # Save exchanges (keep last 500)
            exchanges_to_save = self._exchanges[-500:]
            (self.storage_path / "exchanges.json").write_text(
                json.dumps([asdict(e) for e in exchanges_to_save], indent=2)
            )

            # Save stats
            (self.storage_path / "agent_stats.json").write_text(
                json.dumps(self._agent_stats, indent=2)
            )

        except Exception as e:
            logger.error(f"Could not save dialogue memory: {e}")

    async def store_exchange(self, result: DeliberationResult, session_type: str = None) -> str:
        """
        Store a completed deliberation exchange.

        Args:
            result: The DeliberationResult to store
            session_type: Type of session (website_chat, grok_thread, etc.)

        Returns:
            exchange_id of the stored exchange
        """
        # Convert rounds to serializable format
        serialized_rounds = {}
        for round_name, turns in result.rounds.items():
            serialized_rounds[round_name] = [
                {
                    "turn_id": t.turn_id,
                    "timestamp": t.timestamp.isoformat(),
                    "agent_id": t.agent_id,
                    "content": t.content,
                    "round_type": t.round_type.value,
                    "addressing": t.addressing,
                    "references": t.references,
                    "vote_for": t.vote_for,
                }
                for t in turns
            ]

        exchange = DeliberationExchange(
            exchange_id=result.deliberation_id,
            timestamp=datetime.now().isoformat(),
            prompt=result.prompt,
            participating_agents=result.participating_agents,
            rounds=serialized_rounds,
            final_response=result.final_response,
            winning_agent=result.winning_agent,
            vote_breakdown=result.vote_breakdown,
            tool_used=result.tool_decision.get("tool_name") if result.tool_decision else None,
            consensus_reached=result.consensus_reached,
            duration_ms=result.total_duration_ms,
            session_type=session_type,
        )

        self._exchanges.append(exchange)

        # Update agent stats
        for agent in result.participating_agents:
            if agent not in self._agent_stats:
                self._agent_stats[agent] = {
                    "total_deliberations": 0,
                    "wins": 0,
                    "total_score": 0,
                    "topics": {},
                }

            stats = self._agent_stats[agent]
            stats["total_deliberations"] += 1
            stats["total_score"] += result.vote_breakdown.get(agent, 0)

            if agent == result.winning_agent:
                stats["wins"] += 1

        # Save periodically (every 10 exchanges)
        if len(self._exchanges) % 10 == 0:
            self._save_state()

        logger.info(f"Stored exchange {exchange.exchange_id}")
        return exchange.exchange_id

    async def get_recent_exchanges(
        self,
        limit: int = 10,
        session_type: str = None
    ) -> List[DeliberationExchange]:
        """
        Get recent deliberation exchanges.

        Args:
            limit: Maximum number to return
            session_type: Filter by session type

        Returns:
            List of recent exchanges
        """
        exchanges = self._exchanges

        if session_type:
            exchanges = [e for e in exchanges if e.session_type == session_type]

        return exchanges[-limit:]

    async def get_agent_history(
        self,
        agent_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get all contributions from a specific agent.

        Args:
            agent_id: The agent to get history for
            limit: Maximum number of turns to return

        Returns:
            List of agent turns with context
        """
        history = []

        for exchange in reversed(self._exchanges):
            if agent_id in exchange.participating_agents:
                for round_name, turns in exchange.rounds.items():
                    for turn in turns:
                        if turn["agent_id"] == agent_id:
                            history.append({
                                "exchange_id": exchange.exchange_id,
                                "timestamp": turn["timestamp"],
                                "round": round_name,
                                "content": turn["content"],
                                "prompt": exchange.prompt[:100],
                                "was_winner": exchange.winning_agent == agent_id,
                            })

                            if len(history) >= limit:
                                return history

        return history

    async def get_context_for_agent(
        self,
        agent_id: str,
        topic: str = None,
        limit: int = 5
    ) -> str:
        """
        Get relevant context for an agent from previous deliberations.

        Useful for providing agents with their "memory" of past discussions.

        Args:
            agent_id: The agent to get context for
            topic: Optional topic to filter by
            limit: Maximum number of exchanges to include

        Returns:
            Formatted context string
        """
        context_parts = []

        # Get agent stats
        if agent_id in self._agent_stats:
            stats = self._agent_stats[agent_id]
            win_rate = stats["wins"] / max(1, stats["total_deliberations"]) * 100
            context_parts.append(
                f"YOUR TRACK RECORD: {stats['total_deliberations']} deliberations, "
                f"{stats['wins']} wins ({win_rate:.0f}% win rate)"
            )

        # Get recent successful contributions
        history = await self.get_agent_history(agent_id, limit * 2)
        wins = [h for h in history if h["was_winner"]][:limit]

        if wins:
            context_parts.append("\nYOUR RECENT WINNING RESPONSES:")
            for win in wins[:3]:
                context_parts.append(f"- {win['content'][:100]}...")

        # Get topic-relevant exchanges if topic provided
        if topic:
            topic_lower = topic.lower()
            relevant = [
                e for e in self._exchanges
                if topic_lower in e.prompt.lower() or topic_lower in e.final_response.lower()
            ][-limit:]

            if relevant:
                context_parts.append(f"\nPAST DISCUSSIONS ON '{topic}':")
                for ex in relevant:
                    context_parts.append(
                        f"- Winner ({ex.winning_agent}): {ex.final_response[:80]}..."
                    )

        return "\n".join(context_parts) if context_parts else ""

    async def get_consensus_patterns(self) -> Dict[str, Any]:
        """
        Analyze consensus patterns from past deliberations.

        Returns insights about what leads to consensus.
        """
        total = len(self._exchanges)
        if total == 0:
            return {"error": "No exchanges recorded"}

        consensus_count = sum(1 for e in self._exchanges if e.consensus_reached)
        by_session_type = {}

        for e in self._exchanges:
            st = e.session_type or "unknown"
            if st not in by_session_type:
                by_session_type[st] = {"total": 0, "consensus": 0}
            by_session_type[st]["total"] += 1
            if e.consensus_reached:
                by_session_type[st]["consensus"] += 1

        return {
            "total_exchanges": total,
            "consensus_rate": consensus_count / total * 100,
            "by_session_type": {
                st: {
                    "total": data["total"],
                    "consensus_rate": data["consensus"] / max(1, data["total"]) * 100
                }
                for st, data in by_session_type.items()
            }
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get overall dialogue memory statistics."""
        return {
            "total_exchanges": len(self._exchanges),
            "unique_agents": len(self._agent_stats),
            "agent_stats": {
                agent: {
                    "deliberations": stats["total_deliberations"],
                    "wins": stats["wins"],
                    "avg_score": stats["total_score"] / max(1, stats["total_deliberations"]),
                }
                for agent, stats in self._agent_stats.items()
            }
        }


# Global dialogue memory instance
_dialogue_memory: Optional[DialogueMemory] = None


def get_dialogue_memory() -> DialogueMemory:
    """Get or create the global dialogue memory instance."""
    global _dialogue_memory
    if _dialogue_memory is None:
        _dialogue_memory = DialogueMemory()
    return _dialogue_memory


async def record_deliberation(
    result: DeliberationResult,
    session_type: str = None
) -> str:
    """Quick helper to record a deliberation."""
    memory = get_dialogue_memory()
    return await memory.store_exchange(result, session_type)
