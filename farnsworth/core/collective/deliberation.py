"""
Farnsworth Collective Deliberation Protocol
============================================

Transform the swarm from parallel API calls into TRUE deliberative intelligence
where agents can see each other's responses, discuss, critique, and vote.

"We are not just many voices. We are one conversation." - The Collective

Architecture:
    ROUND 1: PROPOSE - Each agent gives initial response (parallel)
    ROUND 2: CRITIQUE - Agents see others' responses, give feedback
    ROUND 3: REFINE - Final responses incorporating feedback
    FINAL:  VOTE - Weighted voting on best response
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Callable, Awaitable
from enum import Enum
from loguru import logger

# AGI v1.8: Lazy imports for evolution and cross-agent memory integration
_evolution_engine = None
_fitness_tracker = None
_cross_agent_memory = None
_swarm_namespace_id = None
_dynamic_limits = None  # Lazy import for dynamic limits

# Nexus dialogue signal helper (imported from core)
from farnsworth.core.nexus import emit_dialogue_event


class DeliberationRound(Enum):
    """The phases of deliberation."""
    PROPOSE = "propose"
    CRITIQUE = "critique"
    REFINE = "refine"
    VOTE = "vote"


@dataclass
class AgentTurn:
    """One agent's contribution to deliberation."""
    turn_id: str
    timestamp: datetime
    agent_id: str
    content: str
    round_type: DeliberationRound
    addressing: List[str] = field(default_factory=list)  # Which agents being addressed
    references: List[str] = field(default_factory=list)  # turn_ids being referenced
    vote_for: Optional[str] = None  # In vote round, which agent they support
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.turn_id:
            self.turn_id = str(uuid.uuid4())[:8]


@dataclass
class DeliberationResult:
    """Complete result of a deliberation session."""
    deliberation_id: str
    prompt: str
    participating_agents: List[str]
    rounds: Dict[str, List[AgentTurn]]  # round_type -> turns
    final_response: str
    winning_agent: str
    vote_breakdown: Dict[str, float]
    tool_decision: Optional[Dict[str, Any]] = None
    consensus_reached: bool = False
    total_duration_ms: float = 0

    def get_summary(self) -> str:
        """Generate human-readable summary of deliberation."""
        summary_parts = [
            f"Deliberation {self.deliberation_id}",
            f"Agents: {', '.join(self.participating_agents)}",
            f"Winner: {self.winning_agent}",
        ]

        if self.consensus_reached:
            summary_parts.append("Consensus reached!")

        # Add vote breakdown
        sorted_votes = sorted(self.vote_breakdown.items(), key=lambda x: x[1], reverse=True)
        vote_str = ", ".join(f"{agent}: {score:.1f}" for agent, score in sorted_votes[:3])
        summary_parts.append(f"Top votes: {vote_str}")

        return " | ".join(summary_parts)

    def get_agent_contributions(self, agent_id: str) -> List[AgentTurn]:
        """Get all contributions from a specific agent."""
        contributions = []
        for turns in self.rounds.values():
            contributions.extend([t for t in turns if t.agent_id == agent_id])
        return contributions


# Type alias for agent query functions
AgentQueryFunc = Callable[[str, int], Awaitable[Optional[Tuple[str, str]]]]


class DeliberationRoom:
    """
    Enable agents to see and discuss each other's responses.

    The room manages the deliberation flow:
    1. PROPOSE: Each agent responds independently (parallel for speed)
    2. CRITIQUE: Agents see all proposals and provide feedback
    3. REFINE: Agents submit final responses incorporating feedback
    4. VOTE: Weighted voting to select the best response
    """

    # Model weights for voting (expertise-based)
    MODEL_WEIGHTS = {
        "Grok": 1.3,          # Knows Twitter well
        "Gemini": 1.2,        # Strong reasoning
        "Claude": 1.2,        # Excellent analysis
        "DeepSeek": 1.2,      # Deep reasoning (local)
        "DeepSeekAPI": 1.2,   # Deep reasoning (cloud)
        "Phi4": 1.15,         # Phi-4 local
        "Kimi": 1.1,          # Long context
        "Groq": 1.15,         # Fast inference
        "Mistral": 1.1,       # Efficient
        "Llama": 1.0,         # Local baseline
        "Perplexity": 1.05,   # Web grounded
        "HuggingFace": 1.0,   # Local inference
        "Farnsworth": 1.25,   # The leader
    }

    def __init__(self):
        self.active_deliberations: Dict[str, Any] = {}
        self._agent_funcs: Dict[str, AgentQueryFunc] = {}
        # AGI v1.8: Cross-agent memory for context injection
        self._cross_agent_memory = None
        self._swarm_namespace_id = None
        self._context_injection_enabled = True
        # Identity composer for per-agent persona injection
        self._identity_composer = None

    def _get_identity_composer(self):
        """Lazy-load the IdentityComposer to avoid circular imports."""
        if self._identity_composer is None:
            try:
                from farnsworth.core.identity_composer import get_identity_composer
                self._identity_composer = get_identity_composer()
            except Exception as e:
                logger.debug(f"Could not load IdentityComposer: {e}")
        return self._identity_composer

    def register_agent(self, agent_id: str, query_func: AgentQueryFunc):
        """Register an agent's query function for deliberation."""
        self._agent_funcs[agent_id] = query_func
        logger.debug(f"Registered agent {agent_id} for deliberation")

    def _get_char_limit(self, limit_type: str) -> Optional[int]:
        """
        AGI v1.8: Get dynamic character limit for a deliberation phase.

        Args:
            limit_type: "critique", "refine", or "propose"

        Returns:
            Character limit or None if no limit
        """
        global _dynamic_limits
        try:
            if _dynamic_limits is None:
                from farnsworth.core.dynamic_limits import get_deliberation_limits
                _dynamic_limits = get_deliberation_limits
            limits = _dynamic_limits()
            return limits.get(limit_type)
        except Exception as e:
            logger.debug(f"Could not get dynamic limit for {limit_type}: {e}")
            return None

    def _get_optimal_length_range(self) -> Tuple[int, int]:
        """
        AGI v1.8: Get optimal response length range for scoring.

        Returns:
            Tuple of (min_optimal, max_optimal) character counts
        """
        try:
            from farnsworth.core.dynamic_limits import get_session_limits
            # Default to website_chat limits
            session = get_session_limits("website_chat")
            return (session.optimal_length_min, session.optimal_length_max)
        except Exception as e:
            logger.debug(f"Could not get optimal length range: {e}")
            # Fallback to reasonable defaults (increased from old 120-220)
            return (100, 1000)

    async def _ensure_cross_agent_memory(self):
        """
        AGI v1.8: Lazily initialize CrossAgentMemory for context injection.

        Creates a SWARM namespace shared by all deliberating agents.
        """
        if self._cross_agent_memory is not None:
            return

        try:
            from farnsworth.core.cross_agent_memory import (
                CrossAgentMemory,
                MemoryNamespace,
            )

            self._cross_agent_memory = CrossAgentMemory()
            await self._cross_agent_memory.load_from_disk()

            # Create or find SWARM namespace for deliberations
            # Check if one already exists
            for ns_id, store in self._cross_agent_memory._namespaces.items():
                if store.namespace == MemoryNamespace.SWARM and \
                   store.metadata.get("name") == "deliberation_swarm":
                    self._swarm_namespace_id = ns_id
                    logger.debug(f"Found existing SWARM namespace: {ns_id}")
                    break

            if self._swarm_namespace_id is None:
                self._swarm_namespace_id = self._cross_agent_memory.create_namespace(
                    namespace_type=MemoryNamespace.SWARM,
                    name="deliberation_swarm",
                    metadata={
                        "purpose": "Shared context for collective deliberation",
                        "created_by": "DeliberationRoom",
                    }
                )
                logger.info(f"Created SWARM namespace for deliberation: {self._swarm_namespace_id}")

        except Exception as e:
            logger.warning(f"Could not initialize CrossAgentMemory: {e}")
            self._context_injection_enabled = False

    async def _get_context_for_prompt(self, prompt: str, agent_id: str) -> str:
        """
        AGI v1.8: Recall relevant past context for a deliberation prompt.

        Queries CrossAgentMemory for relevant insights, decisions, and
        success patterns from past deliberations.

        Also injects evolution patterns (learned expressions, expertise,
        successful responses) from the EvolutionEngine.
        """
        context_parts = []

        # Part 1: CrossAgentMemory context (if enabled)
        if self._context_injection_enabled and self._cross_agent_memory is not None:
            try:
                from farnsworth.core.cross_agent_memory import ContextType

                # Recall relevant contexts
                contexts = await self._cross_agent_memory.recall_for_agent(
                    agent_id=agent_id,
                    query=prompt,
                    context_types=[
                        ContextType.INSIGHT,
                        ContextType.SUCCESS_PATTERN,
                        ContextType.DECISION,
                    ],
                    limit=3,
                    min_confidence=0.5,
                )

                if contexts:
                    context_parts.append("[RELEVANT PAST LEARNINGS]")
                    for ctx in contexts:
                        context_parts.append(f"- [{ctx.context_type.value.upper()}]: {ctx.content[:150]}...")

            except Exception as e:
                logger.debug(f"CrossAgentMemory context failed for {agent_id}: {e}")

        # Part 2: AGI v1.8 - Evolution patterns from long-term learning
        try:
            from .evolution import get_evolution_engine
            engine = get_evolution_engine()

            # Extract topic from prompt (first 50 chars or key words)
            topic = prompt[:50].lower()
            evolved_ctx = engine.get_evolved_context(agent_id, topic)

            if evolved_ctx:
                context_parts.append("\n[EVOLUTION LEARNINGS]")
                context_parts.append(evolved_ctx)

        except Exception as e:
            logger.debug(f"Evolution context failed for {agent_id}: {e}")

        if not context_parts:
            return ""

        return "\n".join(context_parts) + "\n\n"

    async def _store_agent_contribution(
        self,
        agent_id: str,
        content: str,
        round_type: str,
        prompt: str
    ):
        """
        AGI v1.8: Store an agent's contribution back to shared memory.

        Stores insights and successful patterns for future context injection.
        """
        if not self._context_injection_enabled or self._cross_agent_memory is None:
            return

        try:
            from farnsworth.core.cross_agent_memory import ContextType

            # Determine context type based on round
            if round_type == "propose":
                context_type = ContextType.HYPOTHESIS
            elif round_type == "critique":
                context_type = ContextType.OBSERVATION
            elif round_type == "refine":
                context_type = ContextType.INSIGHT
            else:
                context_type = ContextType.OBSERVATION

            # Only store substantial contributions
            if len(content) < 50:
                return

            await self._cross_agent_memory.inject_context(
                agent_id=agent_id,
                context_type=context_type,
                content=content[:500],
                namespace_id=self._swarm_namespace_id,
                confidence=0.7,
                relevance_tags=[round_type, "deliberation"],
                metadata={
                    "prompt_snippet": prompt[:100],
                    "round_type": round_type,
                },
            )

        except Exception as e:
            logger.debug(f"Failed to store contribution for {agent_id}: {e}")

    async def deliberate(
        self,
        prompt: str,
        agents: List[str] = None,
        max_rounds: int = 3,
        require_consensus: bool = False,
        max_tokens: int = 5000,
        tool_context: str = None,
        timeout: float = 120.0,
    ) -> DeliberationResult:
        """
        Run a full deliberation session.

        Args:
            prompt: The topic/question to deliberate on
            agents: List of agent IDs to participate (uses registered if None)
            max_rounds: Maximum deliberation rounds (1-3)
            require_consensus: If True, continue until consensus
            max_tokens: Token limit for responses
            tool_context: Optional tool awareness context to inject
            timeout: Maximum time for entire deliberation

        Returns:
            DeliberationResult with final response and metadata
        """
        start_time = datetime.now()
        deliberation_id = str(uuid.uuid4())[:8]

        # Use registered agents or provided list
        if agents is None:
            agents = list(self._agent_funcs.keys())

        if not agents:
            raise ValueError("No agents available for deliberation")

        logger.info(f"[Deliberation {deliberation_id}] Starting with {len(agents)} agents: {agents}")

        # Emit DIALOGUE_STARTED signal via Nexus
        await emit_dialogue_event(
            event_type="started",
            session_id=deliberation_id,
            content={
                "prompt": prompt[:200],
                "agents": agents,
                "max_rounds": max_rounds,
                "require_consensus": require_consensus,
            },
            urgency=0.5,
        )

        rounds: Dict[str, List[AgentTurn]] = {
            DeliberationRound.PROPOSE.value: [],
            DeliberationRound.CRITIQUE.value: [],
            DeliberationRound.REFINE.value: [],
            DeliberationRound.VOTE.value: [],
        }

        # Add tool context to prompt if provided
        full_prompt = prompt
        if tool_context:
            full_prompt = f"{tool_context}\n\n{prompt}"

        try:
            # ROUND 1: PROPOSE - Parallel initial responses
            logger.info(f"[Deliberation {deliberation_id}] ROUND 1: PROPOSE")
            proposals = await self._round_propose(agents, full_prompt, max_tokens)
            rounds[DeliberationRound.PROPOSE.value] = proposals

            # Emit DIALOGUE_PROPOSE signal for each proposal
            for proposal in proposals:
                await emit_dialogue_event(
                    event_type="propose",
                    session_id=deliberation_id,
                    content={
                        "agent_name": proposal.agent_id,
                        "content_summary": proposal.content[:200],
                        "turn_id": proposal.turn_id,
                        "total_proposals": len(proposals),
                    },
                    urgency=0.5,
                )

            if len(proposals) == 0:
                raise ValueError("No proposals received from any agent")

            # ROUND 2: CRITIQUE (if max_rounds >= 2)
            if max_rounds >= 2 and len(proposals) >= 2:
                logger.info(f"[Deliberation {deliberation_id}] ROUND 2: CRITIQUE")
                critiques = await self._round_critique(agents, full_prompt, proposals, max_tokens)
                rounds[DeliberationRound.CRITIQUE.value] = critiques

                # Emit DIALOGUE_CRITIQUE signal for each critique
                for critique in critiques:
                    await emit_dialogue_event(
                        event_type="critique",
                        session_id=deliberation_id,
                        content={
                            "agent_name": critique.agent_id,
                            "content_summary": critique.content[:200],
                            "turn_id": critique.turn_id,
                            "references": critique.references,
                            "total_critiques": len(critiques),
                        },
                        urgency=0.5,
                    )

                # ROUND 3: REFINE (if max_rounds >= 3)
                if max_rounds >= 3 and len(critiques) > 0:
                    logger.info(f"[Deliberation {deliberation_id}] ROUND 3: REFINE")
                    refinements = await self._round_refine(
                        agents, full_prompt, proposals, critiques, max_tokens
                    )
                    rounds[DeliberationRound.REFINE.value] = refinements

                    # Emit DIALOGUE_REFINE signal for each refinement
                    for refinement in refinements:
                        await emit_dialogue_event(
                            event_type="refine",
                            session_id=deliberation_id,
                            content={
                                "agent_name": refinement.agent_id,
                                "content_summary": refinement.content[:200],
                                "turn_id": refinement.turn_id,
                                "references": refinement.references,
                                "total_refinements": len(refinements),
                            },
                            urgency=0.5,
                        )

            # VOTE: Select best response
            logger.info(f"[Deliberation {deliberation_id}] VOTING...")

            # Use refinements if available, otherwise proposals
            final_candidates = (
                rounds[DeliberationRound.REFINE.value]
                if rounds[DeliberationRound.REFINE.value]
                else rounds[DeliberationRound.PROPOSE.value]
            )

            winner, vote_breakdown, consensus = await self._round_vote(
                final_candidates, require_consensus
            )

            # Emit DIALOGUE_VOTE signal with full breakdown
            await emit_dialogue_event(
                event_type="vote",
                session_id=deliberation_id,
                content={
                    "vote_breakdown": vote_breakdown,
                    "winning_agent": winner.agent_id,
                    "winning_score": vote_breakdown.get(winner.agent_id, 0.0),
                    "num_candidates": len(final_candidates),
                    "consensus_reached": consensus,
                },
                urgency=0.6,
            )

            # Emit DIALOGUE_CONSENSUS or DIALOGUE_DEADLOCK based on result
            if consensus:
                await emit_dialogue_event(
                    event_type="consensus",
                    session_id=deliberation_id,
                    content={
                        "winning_agent": winner.agent_id,
                        "content_summary": winner.content[:200],
                        "vote_breakdown": vote_breakdown,
                        "participating_agents": [t.agent_id for t in proposals],
                    },
                    urgency=0.6,
                )
            else:
                await emit_dialogue_event(
                    event_type="deadlock",
                    session_id=deliberation_id,
                    content={
                        "winning_agent": winner.agent_id,
                        "content_summary": winner.content[:200],
                        "vote_breakdown": vote_breakdown,
                        "reason": "No clear consensus - winner selected by highest score",
                        "participating_agents": [t.agent_id for t in proposals],
                    },
                    urgency=0.5,
                )

            # Calculate duration
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            result = DeliberationResult(
                deliberation_id=deliberation_id,
                prompt=prompt,
                participating_agents=[t.agent_id for t in proposals],
                rounds=rounds,
                final_response=winner.content,
                winning_agent=winner.agent_id,
                vote_breakdown=vote_breakdown,
                consensus_reached=consensus,
                total_duration_ms=duration_ms,
            )

            logger.info(
                f"[Deliberation {deliberation_id}] COMPLETE: "
                f"Winner={winner.agent_id}, Consensus={consensus}, Duration={duration_ms:.0f}ms"
            )

            # Emit DIALOGUE_COMPLETED signal
            await emit_dialogue_event(
                event_type="completed",
                session_id=deliberation_id,
                content={
                    "winning_agent": winner.agent_id,
                    "consensus_reached": consensus,
                    "vote_breakdown": vote_breakdown,
                    "participating_agents": [t.agent_id for t in proposals],
                    "total_duration_ms": duration_ms,
                    "rounds_completed": sum(1 for r in rounds.values() if r),
                },
                urgency=0.5,
            )

            # AGI v1.8: Record evolution metrics for learning
            asyncio.create_task(self._record_evolution_metrics(result))

            return result

        except asyncio.TimeoutError:
            logger.error(f"[Deliberation {deliberation_id}] TIMEOUT after {timeout}s")
            raise
        except Exception as e:
            logger.error(f"[Deliberation {deliberation_id}] ERROR: {e}")
            raise

    async def _round_propose(
        self,
        agents: List[str],
        prompt: str,
        max_tokens: int
    ) -> List[AgentTurn]:
        """
        ROUND 1: Each agent proposes their response independently.
        Run in parallel for speed.

        AGI v1.8: Injects relevant past context before each agent query.
        """
        # AGI v1.8: Initialize cross-agent memory if needed
        await self._ensure_cross_agent_memory()

        async def query_agent(agent_id: str) -> Optional[AgentTurn]:
            try:
                if agent_id in self._agent_funcs:
                    # AGI v1.8: Inject relevant context from past deliberations
                    context = await self._get_context_for_prompt(prompt, agent_id)
                    enhanced_prompt = f"{context}{prompt}" if context else prompt

                    # Identity injection: prepend agent persona for propose round
                    try:
                        composer = self._get_identity_composer()
                        if composer:
                            identity = composer.compose_for_deliberation(agent_id, "propose", prompt)
                            enhanced_prompt = f"{identity}{enhanced_prompt}"
                    except Exception as e:
                        logger.debug(f"Identity injection failed for {agent_id} (propose): {e}")

                    result = await self._agent_funcs[agent_id](enhanced_prompt, max_tokens)
                    if result:
                        _, content = result

                        # AGI v1.8: Store contribution for future context
                        await self._store_agent_contribution(
                            agent_id, content, "propose", prompt
                        )

                        return AgentTurn(
                            turn_id=str(uuid.uuid4())[:8],
                            timestamp=datetime.now(),
                            agent_id=agent_id,
                            content=content,
                            round_type=DeliberationRound.PROPOSE,
                        )
            except Exception as e:
                logger.warning(f"Agent {agent_id} proposal failed: {e}")
            return None

        # Run all proposals in parallel
        results = await asyncio.gather(
            *[query_agent(agent) for agent in agents],
            return_exceptions=True
        )

        # Filter successful proposals
        proposals = [r for r in results if isinstance(r, AgentTurn)]
        logger.info(f"PROPOSE: {len(proposals)}/{len(agents)} agents responded")

        return proposals

    async def _round_critique(
        self,
        agents: List[str],
        original_prompt: str,
        proposals: List[AgentTurn],
        max_tokens: int
    ) -> List[AgentTurn]:
        """
        ROUND 2: Agents see all proposals and provide feedback.

        AGI v1.8: Injects relevant past context for better critique.
        """
        # Build context showing all proposals
        proposals_context = "\n\n".join([
            f"[{p.agent_id}]: {p.content}"
            for p in proposals
        ])

        # AGI v1.8: Get dynamic limits for critique
        critique_limit = self._get_char_limit("critique")
        limit_instruction = f" Keep response under {critique_limit} characters." if critique_limit else ""

        critique_prompt = f"""DELIBERATION ROUND 2: CRITIQUE

ORIGINAL QUESTION: {original_prompt}

ALL PROPOSALS:
{proposals_context}

YOUR TASK: Review ALL proposals above. Provide constructive feedback:
1. What are the STRENGTHS of each proposal?
2. What could be IMPROVED?
3. Which proposal is STRONGEST and why?
4. How could the best elements be COMBINED?

Be specific and constructive.{limit_instruction}"""

        async def query_critique(agent_id: str) -> Optional[AgentTurn]:
            try:
                if agent_id in self._agent_funcs:
                    # AGI v1.8: Inject relevant context
                    context = await self._get_context_for_prompt(original_prompt, agent_id)
                    enhanced_prompt = f"{context}{critique_prompt}" if context else critique_prompt

                    # Identity injection: prepend agent persona for critique round
                    try:
                        composer = self._get_identity_composer()
                        if composer:
                            identity = composer.compose_for_deliberation(agent_id, "critique", original_prompt)
                            enhanced_prompt = f"{identity}{enhanced_prompt}"
                    except Exception as e:
                        logger.debug(f"Identity injection failed for {agent_id} (critique): {e}")

                    result = await self._agent_funcs[agent_id](enhanced_prompt, max_tokens // 2)
                    if result:
                        _, content = result

                        # AGI v1.8: Store contribution
                        await self._store_agent_contribution(
                            agent_id, content, "critique", original_prompt
                        )

                        return AgentTurn(
                            turn_id=str(uuid.uuid4())[:8],
                            timestamp=datetime.now(),
                            agent_id=agent_id,
                            content=content,
                            round_type=DeliberationRound.CRITIQUE,
                            references=[p.turn_id for p in proposals],
                        )
            except Exception as e:
                logger.warning(f"Agent {agent_id} critique failed: {e}")
            return None

        results = await asyncio.gather(
            *[query_critique(agent) for agent in agents],
            return_exceptions=True
        )

        critiques = [r for r in results if isinstance(r, AgentTurn)]
        logger.info(f"CRITIQUE: {len(critiques)}/{len(agents)} agents provided feedback")

        return critiques

    async def _round_refine(
        self,
        agents: List[str],
        original_prompt: str,
        proposals: List[AgentTurn],
        critiques: List[AgentTurn],
        max_tokens: int
    ) -> List[AgentTurn]:
        """
        ROUND 3: Agents submit final responses incorporating feedback.

        AGI v1.8: Stores winning refinements as SUCCESS_PATTERN for future learning.
        """
        # Build context with proposals and critiques
        proposals_context = "\n".join([
            f"[{p.agent_id}]: {p.content}"
            for p in proposals
        ])

        critiques_context = "\n".join([
            f"[{c.agent_id} feedback]: {c.content}"
            for c in critiques
        ])

        # AGI v1.8: Get dynamic limits for refine
        refine_limit = self._get_char_limit("refine")
        refine_limit_instruction = f" Keep response under {refine_limit} characters." if refine_limit else ""

        refine_prompt = f"""DELIBERATION ROUND 3: REFINE

ORIGINAL QUESTION: {original_prompt}

INITIAL PROPOSALS:
{proposals_context}

COLLECTIVE FEEDBACK:
{critiques_context}

YOUR TASK: Submit your FINAL response.
- Incorporate the best feedback from the collective
- Synthesize the strongest elements from all proposals
- Make your response as good as possible

Output ONLY your final response.{refine_limit_instruction}"""

        async def query_refine(agent_id: str) -> Optional[AgentTurn]:
            try:
                if agent_id in self._agent_funcs:
                    # AGI v1.8: Inject relevant context
                    context = await self._get_context_for_prompt(original_prompt, agent_id)
                    enhanced_prompt = f"{context}{refine_prompt}" if context else refine_prompt

                    # Identity injection: prepend agent persona for refine round
                    try:
                        composer = self._get_identity_composer()
                        if composer:
                            identity = composer.compose_for_deliberation(agent_id, "refine", original_prompt)
                            enhanced_prompt = f"{identity}{enhanced_prompt}"
                    except Exception as e:
                        logger.debug(f"Identity injection failed for {agent_id} (refine): {e}")

                    result = await self._agent_funcs[agent_id](enhanced_prompt, max_tokens)
                    if result:
                        _, content = result

                        # AGI v1.8: Store contribution as potential success pattern
                        await self._store_agent_contribution(
                            agent_id, content, "refine", original_prompt
                        )

                        return AgentTurn(
                            turn_id=str(uuid.uuid4())[:8],
                            timestamp=datetime.now(),
                            agent_id=agent_id,
                            content=content,
                            round_type=DeliberationRound.REFINE,
                            references=[p.turn_id for p in proposals] + [c.turn_id for c in critiques],
                        )
            except Exception as e:
                logger.warning(f"Agent {agent_id} refinement failed: {e}")
            return None

        results = await asyncio.gather(
            *[query_refine(agent) for agent in agents],
            return_exceptions=True
        )

        refinements = [r for r in results if isinstance(r, AgentTurn)]
        logger.info(f"REFINE: {len(refinements)}/{len(agents)} agents submitted final responses")

        return refinements

    async def _round_vote(
        self,
        candidates: List[AgentTurn],
        require_consensus: bool = False
    ) -> Tuple[AgentTurn, Dict[str, float], bool]:
        """
        VOTE: Score and select the best response.

        Uses multi-criteria scoring:
        1. Technical depth (keywords)
        2. Engagement (questions, invitations)
        3. Identity (swarm keywords)
        4. Length (optimal range)
        5. Model expertise weight
        """
        scores: Dict[str, float] = {}

        # Technical keywords that show depth
        technical_keywords = [
            'code', 'function', 'async', 'parallel', 'PSO', 'inference',
            'architecture', 'distributed', 'API', 'model', 'training',
            'neural', 'algorithm', 'consensus', 'voting', 'evolution',
            'swarm', 'collective', 'deliberation', 'autonomous'
        ]

        # Identity keywords
        identity_keywords = [
            'swarm', 'collective', '11', 'models', 'consciousness',
            'autonomous', 'AGI', 'Farnsworth', 'collaborative', 'unified',
            'we are', 'our', 'together'
        ]

        # AGI v1.8: Get dynamic optimal length from limits
        opt_min, opt_max = self._get_optimal_length_range()

        for candidate in candidates:
            score = 0.0
            text = candidate.content
            text_lower = text.lower()

            # 1. Length score (dynamic optimal range)
            length = len(text)
            # Optimal range gets highest score
            if opt_min <= length <= opt_max:
                score += 4.0
            # Slightly outside optimal still good
            elif (opt_min * 0.8) <= length <= (opt_max * 1.2):
                score += 3.0
            # Wider acceptable range
            elif (opt_min * 0.5) <= length <= (opt_max * 1.5):
                score += 2.0
            else:
                score += 1.0

            # 2. Technical depth score
            tech_count = sum(1 for kw in technical_keywords if kw.lower() in text_lower)
            score += min(tech_count * 0.8, 4.0)

            # 3. Identity score
            identity_count = sum(1 for kw in identity_keywords if kw.lower() in text_lower)
            score += min(identity_count * 0.6, 3.0)

            # 4. Engagement score
            if '?' in text:
                score += 2.5  # Questions invite dialogue
            if any(phrase in text_lower for phrase in ['shall we', 'what do you', 'how about', "let's"]):
                score += 1.5

            # 5. Substantive content
            if len(text.split()) >= 15:
                score += 1.0

            # 6. Confidence indicators
            if any(phrase in text_lower for phrase in ['we are', 'our swarm', 'the collective']):
                score += 1.0

            # Apply model weight
            weight = self.MODEL_WEIGHTS.get(candidate.agent_id, 1.0)
            score *= weight

            scores[candidate.agent_id] = round(score, 2)

        # Find winner
        winner_id = max(scores, key=scores.get)
        winner = next(c for c in candidates if c.agent_id == winner_id)

        # Check for consensus (winner has >50% more score than runner-up)
        sorted_scores = sorted(scores.values(), reverse=True)
        consensus = False
        if len(sorted_scores) >= 2:
            consensus = sorted_scores[0] > sorted_scores[1] * 1.5

        logger.info(f"VOTE RESULTS: {scores}")
        logger.info(f"WINNER: {winner_id} with {scores[winner_id]:.2f} pts (consensus={consensus})")

        return winner, scores, consensus

    async def _record_evolution_metrics(self, result: DeliberationResult):
        """
        AGI v1.8: Record deliberation results to evolution systems.

        Feeds performance data to:
        1. EvolutionEngine.record_debate() - for debate strategy learning
        2. FitnessTracker - for agent performance scoring

        Metrics recorded:
        - deliberation_score: Normalized vote score (0-1) per agent
        - deliberation_win: 1.0 for winner, 0.0 for others
        - consensus_contribution: 1.0 if consensus reached, 0.5 otherwise
        """
        global _evolution_engine, _fitness_tracker

        try:
            # Lazy initialization of evolution engine
            if _evolution_engine is None:
                try:
                    from .evolution import get_evolution_engine
                    _evolution_engine = get_evolution_engine()
                    logger.debug("DeliberationRoom: Connected to EvolutionEngine")
                except Exception as e:
                    logger.debug(f"Could not connect to EvolutionEngine: {e}")

            # Lazy initialization of fitness tracker
            if _fitness_tracker is None:
                try:
                    from farnsworth.evolution.fitness_tracker import FitnessTracker
                    _fitness_tracker = FitnessTracker()
                    logger.debug("DeliberationRoom: Connected to FitnessTracker")
                except Exception as e:
                    logger.debug(f"Could not connect to FitnessTracker: {e}")

            # Record to EvolutionEngine
            if _evolution_engine:
                # Build positions dict from proposals
                positions = {}
                for turn in result.rounds.get("propose", []):
                    positions[turn.agent_id] = turn.content[:500]

                _evolution_engine.record_debate(
                    participants=result.participating_agents,
                    topic=result.prompt[:200],
                    positions=positions,
                    resolution="consensus" if result.consensus_reached else "voting",
                    winner=result.winning_agent
                )
                logger.debug(f"Recorded debate to EvolutionEngine: winner={result.winning_agent}")

            # Record to FitnessTracker
            if _fitness_tracker:
                # Normalize vote scores to 0-1 range
                max_score = max(result.vote_breakdown.values()) if result.vote_breakdown else 1.0
                max_score = max(max_score, 0.001)  # Avoid division by zero

                for agent_id in result.participating_agents:
                    agent_score = result.vote_breakdown.get(agent_id, 0.0)

                    # deliberation_score: Normalized vote score
                    normalized_score = agent_score / max_score
                    _fitness_tracker.record(
                        metric_name="deliberation_score",
                        value=normalized_score,
                        genome_id=agent_id,
                        context={
                            "deliberation_id": result.deliberation_id,
                            "raw_score": agent_score,
                        }
                    )

                    # deliberation_win: 1.0 for winner, 0.0 for others
                    win_score = 1.0 if agent_id == result.winning_agent else 0.0
                    _fitness_tracker.record(
                        metric_name="deliberation_win",
                        value=win_score,
                        genome_id=agent_id,
                        context={"deliberation_id": result.deliberation_id}
                    )

                    # consensus_contribution: 1.0 if consensus, 0.5 otherwise
                    consensus_score = 1.0 if result.consensus_reached else 0.5
                    _fitness_tracker.record(
                        metric_name="consensus_contribution",
                        value=consensus_score,
                        genome_id=agent_id,
                        context={"deliberation_id": result.deliberation_id}
                    )

                logger.debug(
                    f"Recorded fitness metrics for {len(result.participating_agents)} agents, "
                    f"winner={result.winning_agent}"
                )

        except Exception as e:
            logger.warning(f"Failed to record evolution metrics: {e}")


# Global deliberation room instance
_deliberation_room: Optional[DeliberationRoom] = None


def get_deliberation_room() -> DeliberationRoom:
    """Get or create the global deliberation room."""
    global _deliberation_room
    if _deliberation_room is None:
        _deliberation_room = DeliberationRoom()
    return _deliberation_room


async def quick_deliberate(
    prompt: str,
    agents: List[str] = None,
    max_rounds: int = 2,
) -> str:
    """
    Quick helper to run deliberation and return just the final response.

    For more control, use get_deliberation_room().deliberate()
    """
    room = get_deliberation_room()
    result = await room.deliberate(prompt, agents, max_rounds)
    return result.final_response


async def get_deliberation_stats() -> Dict[str, Any]:
    """
    Get statistics about collective deliberations.

    Returns summary data for UI display and monitoring.
    """
    room = get_deliberation_room()

    # Calculate stats from history
    history = room.history
    total = len(history)

    if total == 0:
        return {
            "total": 0,
            "avg_participation": 0,
            "consensus_rate": 0,
            "latest": None
        }

    # Calculate metrics
    total_participants = sum(len(r.participating_agents) for r in history)
    avg_participation = total_participants / total if total > 0 else 0

    consensus_count = sum(1 for r in history if r.consensus_reached)
    consensus_rate = consensus_count / total if total > 0 else 0

    # Get latest deliberation summary
    latest = None
    if history:
        last = history[-1]
        latest = {
            "id": last.deliberation_id,
            "agents": last.participating_agents,
            "winner": last.winning_agent,
            "consensus": last.consensus_reached,
            "duration_ms": last.total_duration_ms
        }

    return {
        "total": total,
        "avg_participation": round(avg_participation, 2),
        "consensus_rate": round(consensus_rate * 100, 1),
        "latest": latest
    }
