"""
Farnsworth Swarm Orchestrator - Multi-Agent Coordination

Novel Approaches:
1. Dynamic Agent Spawning - Create agents on demand
2. Capability-Based Routing - Match tasks to best agent
3. Parallel Execution - Run independent subtasks concurrently
4. Emergent Behavior - Allow agent team compositions to evolve

AGI UPGRADES (v1.4):
- Fully event-driven via Nexus (central nervous system)
- Context vector routing (semantic agent matching)
- Memory-aware task assignment (recall before routing)
- Speculative agent spawning on spontaneous thoughts
- Evolution triggers on dialogue consensus and anomalies
- Closed AGI loop: Memory ↔ Nexus ↔ Swarm
"""

import asyncio
import random
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Callable, List, Dict
import uuid

from loguru import logger

from farnsworth.agents.base_agent import BaseAgent, AgentCapability, AgentStatus, TaskResult
from farnsworth.core.nexus import (
    nexus, Signal, SignalType,
    emit_thought, emit_memory_consolidation,
)


# =============================================================================
# POPULATION-BASED EVOLUTION (AGI Upgrade)
# =============================================================================

@dataclass
class AgentVariant:
    """
    A variant of an agent for population-based evolution.

    Tracks genetic traits and fitness for natural selection.
    """
    variant_id: str
    base_agent_type: str
    generation: int = 0

    # Genetic traits (can be mutated)
    temperature: float = 0.7
    capability_weights: dict = field(default_factory=dict)
    prompt_style: str = "balanced"  # "concise", "detailed", "creative", "balanced"
    confidence_threshold: float = 0.6

    # Fitness tracking
    fitness_score: float = 0.5
    tasks_completed: int = 0
    tasks_failed: int = 0
    avg_confidence: float = 0.5
    avg_execution_time: float = 0.0

    # Lineage
    parent_id: Optional[str] = None
    mutation_history: list = field(default_factory=list)

    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "variant_id": self.variant_id,
            "base_agent_type": self.base_agent_type,
            "generation": self.generation,
            "fitness_score": self.fitness_score,
            "temperature": self.temperature,
            "prompt_style": self.prompt_style,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
        }


@dataclass
class EvolutionConfig:
    """Configuration for population-based evolution."""
    population_size: int = 10
    generations: int = 5
    mutation_rate: float = 0.2
    elite_ratio: float = 0.2  # Top 20% survive unchanged
    crossover_rate: float = 0.3
    fitness_weights: dict = field(default_factory=lambda: {
        "success_rate": 0.4,
        "confidence": 0.2,
        "speed": 0.2,
        "handoff_efficiency": 0.2,
    })


class TaskStatus(Enum):
    """Status of a swarm task."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SwarmTask:
    """A task managed by the swarm."""
    id: str
    description: str
    required_capabilities: set[AgentCapability] = field(default_factory=set)
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    result: Optional[TaskResult] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    parent_task: Optional[str] = None
    subtasks: list[str] = field(default_factory=list)
    context: dict = field(default_factory=dict)
    priority: int = 5  # 1-10, higher = more priority

    # AGI: Semantic routing
    context_vector: Optional[List[float]] = None  # For neural routing
    semantic_tags: List[str] = field(default_factory=list)
    memory_refs: List[str] = field(default_factory=list)  # Related memory IDs


@dataclass
class SwarmState:
    """State of the entire swarm."""
    active_agents: dict[str, BaseAgent] = field(default_factory=dict)
    tasks: dict[str, SwarmTask] = field(default_factory=dict)
    task_queue: list[str] = field(default_factory=list)  # Task IDs
    completed_tasks: list[str] = field(default_factory=list)
    total_tasks_processed: int = 0
    total_handoffs: int = 0


class SwarmOrchestrator:
    """
    Orchestrates multiple agents working together.

    Features:
    - Dynamic agent creation and lifecycle management
    - Intelligent task routing based on capabilities
    - Handoff protocols between specialists
    - Shared state and memory management
    - Parallel subtask execution

    AGI Upgrades:
    - Fully event-driven via Nexus signals
    - Context vector routing (semantic agent matching)
    - Memory-aware task assignment
    - Speculative agent spawning on spontaneous thoughts
    - Evolution triggers on dialogue consensus/anomalies
    """

    def __init__(
        self,
        max_concurrent_agents: int = 5,
        handoff_timeout_seconds: float = 30.0,
        enable_nexus: bool = True,
    ):
        self.max_concurrent = max_concurrent_agents
        self.handoff_timeout = handoff_timeout_seconds
        self.enable_nexus = enable_nexus

        self.state = SwarmState()

        # Agent factories
        self._agent_factories: dict[str, Callable[[], BaseAgent]] = {}

        # Shared resources
        self.llm_backend = None
        self.memory_system = None

        # Event handlers (legacy - kept for backwards compatibility)
        self._on_task_complete: list[Callable] = []
        self._on_handoff: list[Callable] = []

        self._lock = asyncio.Lock()

        # AGI: Nexus integration
        self._nexus_subscribed = False
        self._speculative_spawn_probability = 0.3  # Chance to spawn on thought
        self._evolution_on_consensus = True
        self._evolution_on_anomaly = True
        self._memory_aware_routing = True

        # AGI: Agent capability vectors (for semantic matching)
        self._agent_type_vectors: Dict[str, List[float]] = {}

        # AGI: Speculative agent tracking
        self._speculative_agents: Dict[str, str] = {}  # agent_id -> thought_id

        logger.info("SwarmOrchestrator initialized with AGI upgrades")

    # =========================================================================
    # NEXUS INTEGRATION (AGI Upgrade)
    # =========================================================================

    async def connect_to_nexus(self):
        """
        Connect the swarm to the Nexus event bus.

        Subscribes to relevant signals for event-driven operation:
        - THOUGHT_EMITTED: Speculative agent spawning
        - DIALOGUE_CONSENSUS: Evolution triggers
        - ANOMALY_DETECTED: Evolution and adaptation
        - MEMORY_CONSOLIDATION: Memory-aware routing
        - TASK_* signals: Task lifecycle events
        """
        if self._nexus_subscribed:
            logger.warning("Already connected to Nexus")
            return

        # Subscribe to cognitive signals
        nexus.subscribe(SignalType.THOUGHT_EMITTED, self._on_thought_emitted)
        nexus.subscribe(SignalType.DIALOGUE_CONSENSUS, self._on_dialogue_consensus)
        nexus.subscribe(SignalType.ANOMALY_DETECTED, self._on_anomaly_detected)
        nexus.subscribe(SignalType.MEMORY_CONSOLIDATION, self._on_memory_consolidation)

        # Subscribe to P2P signals
        nexus.subscribe(SignalType.TASK_RECEIVED, self._on_external_task)
        nexus.subscribe(SignalType.SKILL_RECEIVED, self._on_skill_received)

        # Subscribe to resonance signals (collective thoughts)
        nexus.subscribe(SignalType.RESONANCE_RECEIVED, self._on_resonance_received)

        self._nexus_subscribed = True
        logger.info("SwarmOrchestrator connected to Nexus")

        # Emit startup signal
        await nexus.emit(
            SignalType.SYSTEM_STARTUP,
            {"component": "swarm_orchestrator", "agent_types": list(self._agent_factories.keys())},
            source="swarm_orchestrator",
            urgency=0.8,
        )

    async def disconnect_from_nexus(self):
        """Disconnect from Nexus."""
        if not self._nexus_subscribed:
            return

        # Note: Nexus doesn't currently support unsubscribe by handler
        # In production, we'd track subscription IDs
        self._nexus_subscribed = False
        logger.info("SwarmOrchestrator disconnected from Nexus")

    async def _on_thought_emitted(self, signal: Signal):
        """
        Handle spontaneous thought signals.

        May trigger speculative agent spawning for interesting thoughts.
        """
        if not self.enable_nexus:
            return

        thought_content = signal.payload.get("content", "")
        thought_type = signal.payload.get("thought_type", "general")
        relevance = signal.payload.get("relevance", 0.5)

        # Only consider high-relevance thoughts
        if relevance < 0.6:
            return

        # Probabilistic speculative spawning
        if random.random() > self._speculative_spawn_probability:
            return

        # Determine if thought suggests a useful task
        agent_type = self._infer_agent_type_from_thought(thought_content, thought_type)

        if agent_type and agent_type in self._agent_factories:
            # Spawn speculative agent
            agent = await self.spawn_agent(agent_type)
            if agent:
                self._speculative_agents[agent.agent_id] = signal.id
                logger.info(
                    f"Speculative agent spawned: {agent.name} "
                    f"(triggered by thought: {thought_content[:50]}...)"
                )

                # Create exploratory task from thought
                if thought_type == "question":
                    task_desc = f"Investigate: {thought_content}"
                elif thought_type == "insight":
                    task_desc = f"Explore implications: {thought_content}"
                else:
                    task_desc = f"Research: {thought_content}"

                await self.submit_task(
                    description=task_desc,
                    context={"speculative": True, "source_thought": signal.id},
                    context_vector=signal.context_vector,
                    priority=3,  # Lower priority for speculative tasks
                )

    async def _on_dialogue_consensus(self, signal: Signal):
        """
        Handle dialogue consensus signals.

        High-confidence consensus may trigger evolution of agent population.
        """
        if not self._evolution_on_consensus:
            return

        session_id = signal.payload.get("session_id")
        confidence = signal.payload.get("confidence", 0.0)
        decision = signal.payload.get("decision", "")

        # High-confidence consensus triggers evolution
        if confidence >= 0.85:
            logger.info(f"High-confidence consensus detected - considering evolution")

            if hasattr(self, '_evolution_config') and self._population:
                # Boost fitness of agents that contributed to consensus
                contributors = signal.payload.get("contributors", [])
                for agent_id in contributors:
                    for variant in self._population.values():
                        if variant.base_agent_type in agent_id:
                            variant.fitness_score *= 1.1  # 10% fitness boost
                            variant.mutation_history.append(f"consensus_boost:{session_id}")

    async def _on_anomaly_detected(self, signal: Signal):
        """
        Handle anomaly signals.

        Anomalies may trigger defensive evolution or adaptation.
        """
        if not self._evolution_on_anomaly:
            return

        anomaly_type = signal.payload.get("anomaly_type", "unknown")
        severity = signal.payload.get("severity", 0.5)

        if severity >= 0.7:
            logger.warning(f"High-severity anomaly: {anomaly_type} - triggering adaptation")

            # Trigger evolution with stricter fitness threshold
            if hasattr(self, '_evolution_config'):
                self.evolve_subscriptions_internal(fitness_threshold=0.5)

            # Potentially spawn defensive agents
            if anomaly_type in ["security", "resource_exhaustion", "cascade_failure"]:
                if "defensive" in self._agent_factories:
                    agent = await self.spawn_agent("defensive")
                    if agent:
                        await self.submit_task(
                            description=f"Investigate and mitigate anomaly: {anomaly_type}",
                            context={"anomaly_signal": signal.id},
                            priority=9,  # High priority
                        )

    async def _on_memory_consolidation(self, signal: Signal):
        """
        Handle memory consolidation signals.

        Updates routing knowledge based on memory patterns.
        """
        if not self._memory_aware_routing:
            return

        memory_ids = signal.payload.get("memory_ids", [])
        new_vector = signal.payload.get("new_vector")
        session_ref = signal.payload.get("session_ref")

        # If we have a context vector, update agent type vectors
        if new_vector and session_ref:
            # Associate vector with successful agent types
            if session_ref in self.state.tasks:
                task = self.state.tasks[session_ref]
                if task.assigned_agent:
                    agent = self.state.active_agents.get(task.assigned_agent)
                    if agent:
                        agent_type = agent.name
                        self._update_agent_type_vector(agent_type, new_vector)

    async def _on_external_task(self, signal: Signal):
        """Handle tasks received from P2P network."""
        task_data = signal.payload.get("task", {})
        description = task_data.get("description", "")
        priority = task_data.get("priority", 5)
        context_vector = signal.context_vector

        if description:
            await self.submit_task(
                description=description,
                context={"p2p_source": signal.source_id},
                context_vector=context_vector,
                priority=priority,
            )

    async def _on_skill_received(self, signal: Signal):
        """Handle new skill received from network."""
        skill_type = signal.payload.get("skill_type")
        factory_code = signal.payload.get("factory")

        # Note: In production, would validate/sandbox received skills
        logger.info(f"Skill received from network: {skill_type}")

    async def _on_resonance_received(self, signal: Signal):
        """Handle collective thoughts from other Farnsworth instances."""
        thought = signal.payload.get("thought", "")
        source_collective = signal.payload.get("source_collective")

        if thought and signal.context_vector:
            # Consider spawning based on collective resonance
            await self._on_thought_emitted(signal)

    def _infer_agent_type_from_thought(self, content: str, thought_type: str) -> Optional[str]:
        """Infer what agent type might be useful for a thought."""
        content_lower = content.lower()

        if thought_type == "connection":
            return "reasoning"
        elif thought_type == "question":
            if any(kw in content_lower for kw in ["code", "implement", "function"]):
                return "code"
            elif any(kw in content_lower for kw in ["research", "find", "search"]):
                return "research"
            return "reasoning"
        elif thought_type == "insight":
            return "reasoning"
        elif thought_type == "idea":
            if any(kw in content_lower for kw in ["creative", "design", "write"]):
                return "creative"
            return "general"

        return None

    def _update_agent_type_vector(self, agent_type: str, vector: List[float], alpha: float = 0.1):
        """Update the semantic vector for an agent type (exponential moving average)."""
        if agent_type not in self._agent_type_vectors:
            self._agent_type_vectors[agent_type] = vector
        else:
            existing = self._agent_type_vectors[agent_type]
            if len(existing) == len(vector):
                # EMA update
                self._agent_type_vectors[agent_type] = [
                    alpha * v + (1 - alpha) * e
                    for v, e in zip(vector, existing)
                ]

    def evolve_subscriptions_internal(self, fitness_threshold: float = 0.3):
        """Trigger evolution of agent population based on fitness."""
        if not hasattr(self, '_population') or not self._population:
            return

        # Flag low-fitness variants for replacement
        to_replace = []
        for variant_id, variant in self._population.items():
            if variant.fitness_score < fitness_threshold:
                to_replace.append(variant_id)

        if to_replace:
            logger.info(f"Evolution: {len(to_replace)} low-fitness variants flagged")

    # =========================================================================
    # NEXUS SIGNAL EMISSION
    # =========================================================================

    async def _emit_task_created(self, task: SwarmTask):
        """Emit TASK_CREATED signal."""
        if self.enable_nexus:
            await nexus.emit(
                SignalType.TASK_CREATED,
                {
                    "task_id": task.id,
                    "description": task.description[:200],
                    "priority": task.priority,
                    "capabilities": [c.value for c in task.required_capabilities],
                },
                source="swarm_orchestrator",
                urgency=0.5 + task.priority * 0.05,
                context_vector=task.context_vector,
            )

    async def _emit_task_completed(self, task: SwarmTask, result: TaskResult):
        """Emit TASK_COMPLETED signal."""
        if self.enable_nexus:
            await nexus.emit(
                SignalType.TASK_COMPLETED,
                {
                    "task_id": task.id,
                    "success": result.success,
                    "confidence": result.confidence,
                    "assigned_agent": task.assigned_agent,
                    "execution_time": result.execution_time,
                },
                source="swarm_orchestrator",
                urgency=0.6,
                context_vector=task.context_vector,
            )

    async def _emit_task_failed(self, task: SwarmTask, error: str):
        """Emit TASK_FAILED signal."""
        if self.enable_nexus:
            await nexus.emit(
                SignalType.TASK_FAILED,
                {
                    "task_id": task.id,
                    "error": error,
                    "assigned_agent": task.assigned_agent,
                    "retry_count": task.context.get("retry_count", 0),
                },
                source="swarm_orchestrator",
                urgency=0.8,
                context_vector=task.context_vector,
            )

    async def _emit_handoff(self, from_agent: str, to_type: str, reason: str):
        """Emit handoff signal."""
        if self.enable_nexus:
            self.state.total_handoffs += 1
            # Use external event for handoffs
            await nexus.emit(
                SignalType.EXTERNAL_EVENT,
                {
                    "event_type": "agent_handoff",
                    "from_agent": from_agent,
                    "to_agent_type": to_type,
                    "reason": reason,
                },
                source="swarm_orchestrator",
                urgency=0.6,
            )

    # =========================================================================
    # ORIGINAL METHODS (Enhanced)
    # =========================================================================

    def register_agent_factory(
        self,
        agent_type: str,
        factory: Callable[[], BaseAgent],
        capability_vector: Optional[List[float]] = None,
    ):
        """
        Register an agent factory for dynamic creation.

        Args:
            agent_type: Name of the agent type
            factory: Factory function to create agent instances
            capability_vector: Optional semantic vector for this agent type
        """
        self._agent_factories[agent_type] = factory
        if capability_vector:
            self._agent_type_vectors[agent_type] = capability_vector
        logger.info(f"Registered agent factory: {agent_type}")

    async def spawn_agent(self, agent_type: str) -> Optional[BaseAgent]:
        """Spawn a new agent of the given type."""
        if agent_type not in self._agent_factories:
            logger.error(f"Unknown agent type: {agent_type}")
            return None

        if len(self.state.active_agents) >= self.max_concurrent:
            # Try to recycle an idle agent
            await self._recycle_idle_agents()

            if len(self.state.active_agents) >= self.max_concurrent:
                logger.warning("Max concurrent agents reached")
                return None

        # Create agent
        agent = self._agent_factories[agent_type]()

        # Configure agent
        agent.llm_backend = self.llm_backend
        agent.memory = self.memory_system
        agent.set_handoff_callback(self._handle_handoff)

        self.state.active_agents[agent.agent_id] = agent
        logger.info(f"Spawned agent: {agent.name} ({agent.agent_id})")

        return agent

    async def _recycle_idle_agents(self):
        """Recycle idle agents to free up capacity."""
        idle_agents = [
            agent_id for agent_id, agent in self.state.active_agents.items()
            if agent.state.status == AgentStatus.IDLE
        ]

        # Keep at least one of each type
        type_counts: dict[str, int] = {}
        for agent in self.state.active_agents.values():
            type_counts[agent.name] = type_counts.get(agent.name, 0) + 1

        for agent_id in idle_agents:
            agent = self.state.active_agents[agent_id]
            if type_counts.get(agent.name, 0) > 1:
                del self.state.active_agents[agent_id]
                type_counts[agent.name] -= 1
                logger.debug(f"Recycled idle agent: {agent_id}")

    async def submit_task(
        self,
        description: str,
        required_capabilities: Optional[set[AgentCapability]] = None,
        context: Optional[dict] = None,
        priority: int = 5,
        context_vector: Optional[List[float]] = None,
        semantic_tags: Optional[List[str]] = None,
    ) -> str:
        """
        Submit a task to the swarm.

        Args:
            description: Task description
            required_capabilities: Required agent capabilities
            context: Additional context dict
            priority: Task priority (1-10)
            context_vector: Semantic vector for neural routing
            semantic_tags: Tags for categorization

        Returns:
            Task ID
        """
        task_id = f"task_{uuid.uuid4().hex[:8]}"

        task = SwarmTask(
            id=task_id,
            description=description,
            required_capabilities=required_capabilities or set(),
            context=context or {},
            priority=priority,
            context_vector=context_vector,
            semantic_tags=semantic_tags or [],
        )

        # AGI: Memory-aware routing - recall relevant memories
        if self._memory_aware_routing and self.memory_system and context_vector:
            try:
                memories = await self._recall_relevant_memories(description, context_vector)
                task.memory_refs = [m.get("id", "") for m in memories if m.get("id")]
                task.context["memory_context"] = memories[:3]  # Top 3 for context
            except Exception as e:
                logger.debug(f"Memory recall failed: {e}")

        async with self._lock:
            self.state.tasks[task_id] = task

            # Insert into queue based on priority
            inserted = False
            for i, queued_id in enumerate(self.state.task_queue):
                queued_task = self.state.tasks[queued_id]
                if task.priority > queued_task.priority:
                    self.state.task_queue.insert(i, task_id)
                    inserted = True
                    break

            if not inserted:
                self.state.task_queue.append(task_id)

        logger.info(f"Task submitted: {task_id} - {description[:50]}...")

        # Emit Nexus signal
        await self._emit_task_created(task)

        # Try to process immediately
        asyncio.create_task(self._process_queue())

        return task_id

    async def _recall_relevant_memories(
        self,
        description: str,
        context_vector: List[float],
        limit: int = 5,
    ) -> List[Dict]:
        """Recall relevant memories for a task."""
        if not self.memory_system:
            return []

        try:
            # Try to use memory system's recall method
            if hasattr(self.memory_system, 'recall'):
                results = await self.memory_system.recall(
                    query=description,
                    context_vector=context_vector,
                    limit=limit,
                )
                return [r.to_dict() if hasattr(r, 'to_dict') else r for r in results]
            elif hasattr(self.memory_system, 'query'):
                results = await self.memory_system.query(description, top_k=limit)
                return results
        except Exception as e:
            logger.debug(f"Memory recall error: {e}")

        return []

    async def _process_queue(self):
        """Process tasks in the queue."""
        async with self._lock:
            while self.state.task_queue:
                task_id = self.state.task_queue[0]
                task = self.state.tasks.get(task_id)

                if not task or task.status != TaskStatus.PENDING:
                    self.state.task_queue.pop(0)
                    continue

                # Find best agent for task
                agent = await self._find_best_agent(task)

                if agent is None:
                    # Try to spawn a new agent
                    agent_type = self._infer_agent_type(task)
                    agent = await self.spawn_agent(agent_type)

                if agent is None:
                    # No capacity, wait
                    break

                # Assign task
                self.state.task_queue.pop(0)
                task.status = TaskStatus.ASSIGNED
                task.assigned_agent = agent.agent_id

                # Execute task (don't await here to allow parallel processing)
                asyncio.create_task(self._execute_task(task, agent))

    async def _find_best_agent(self, task: SwarmTask) -> Optional[BaseAgent]:
        """
        Find the best available agent for a task.

        Uses hybrid scoring:
        1. Capability matching (original)
        2. Context vector similarity (AGI)
        3. Agent performance history
        """
        best_agent = None
        best_score = 0.0

        for agent in self.state.active_agents.values():
            if agent.state.status not in (AgentStatus.IDLE, AgentStatus.COMPLETED):
                continue

            # Original capability score
            capability_score = agent.can_handle(task.required_capabilities)

            # AGI: Context vector similarity
            vector_score = 0.0
            if task.context_vector and agent.name in self._agent_type_vectors:
                agent_vector = self._agent_type_vectors[agent.name]
                vector_score = self._cosine_similarity(task.context_vector, agent_vector)

            # Combine scores
            if task.context_vector and agent.name in self._agent_type_vectors:
                # Weighted combination when vector available
                score = capability_score * 0.4 + vector_score * 0.4 + agent.state.avg_confidence * 0.2
            else:
                # Original scoring when no vector
                score = capability_score * (0.5 + 0.5 * agent.state.avg_confidence)

            if score > best_score:
                best_score = score
                best_agent = agent

        return best_agent if best_score > 0.3 else None

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _infer_agent_type(self, task: SwarmTask) -> str:
        """Infer the best agent type for a task."""
        if not task.required_capabilities:
            # Try to infer from description
            desc_lower = task.description.lower()

            if any(kw in desc_lower for kw in ["code", "implement", "function", "debug"]):
                return "code"
            elif any(kw in desc_lower for kw in ["reason", "think", "analyze", "math"]):
                return "reasoning"
            elif any(kw in desc_lower for kw in ["research", "find", "search"]):
                return "research"
            elif any(kw in desc_lower for kw in ["write", "creative", "story"]):
                return "creative"

        # Map capabilities to agent types
        cap_to_type = {
            AgentCapability.CODE_GENERATION: "code",
            AgentCapability.CODE_ANALYSIS: "code",
            AgentCapability.REASONING: "reasoning",
            AgentCapability.MATH: "reasoning",
            AgentCapability.RESEARCH: "research",
            AgentCapability.CREATIVE_WRITING: "creative",
        }

        for cap in task.required_capabilities:
            if cap in cap_to_type:
                return cap_to_type[cap]

        return "general"

    async def _execute_task(
        self,
        task: SwarmTask,
        agent: BaseAgent,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """Execute a task with an agent, with retry logic and fallback."""
        task.status = TaskStatus.IN_PROGRESS
        task.context["retry_count"] = 0

        last_error = None
        for attempt in range(max_retries):
            task.context["retry_count"] = attempt
            try:
                result = await agent.execute(task.description, task.context)

                # Check if result indicates failure that might be retryable
                if result.success:
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.now()

                    self.state.total_tasks_processed += 1
                    self.state.completed_tasks.append(task.id)

                    # Notify legacy listeners
                    for handler in self._on_task_complete:
                        try:
                            await handler(task, result)
                        except Exception as e:
                            logger.error(f"Task complete handler error: {e}")

                    # AGI: Emit Nexus signal
                    await self._emit_task_completed(task, result)

                    # AGI: Update agent type vector with successful task
                    if task.context_vector:
                        self._update_agent_type_vector(agent.name, task.context_vector)

                    # AGI: Emit memory consolidation for successful tasks
                    if self.enable_nexus and task.memory_refs:
                        await emit_memory_consolidation(
                            memory_ids=task.memory_refs,
                            session_ref=task.id,
                            context_vector=task.context_vector,
                        )

                    logger.info(f"Task {task.id} completed: success={result.success}")
                    return

                # Result was not successful - try with different agent if available
                last_error = result.output
                logger.warning(f"Task {task.id} attempt {attempt + 1} failed: {last_error}")

                if attempt < max_retries - 1:
                    # Try to find a different agent
                    alt_agent = await self._find_alternative_agent(task, agent.agent_id)
                    if alt_agent:
                        # AGI: Emit handoff signal
                        await self._emit_handoff(agent.agent_id, alt_agent.name, last_error)
                        agent = alt_agent
                        logger.info(f"Retrying with alternative agent: {alt_agent.name}")
                    await asyncio.sleep(retry_delay * (attempt + 1))

            except asyncio.TimeoutError:
                last_error = "Task execution timed out"
                logger.warning(f"Task {task.id} attempt {attempt + 1} timed out")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))

            except Exception as e:
                last_error = str(e)
                logger.error(f"Task {task.id} attempt {attempt + 1} error: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))

        # All retries exhausted
        logger.error(f"Task {task.id} failed after {max_retries} attempts")
        task.status = TaskStatus.FAILED
        task.result = TaskResult(success=False, output=f"Failed after {max_retries} attempts: {last_error}")

        # AGI: Emit failure signal
        await self._emit_task_failed(task, last_error or "Unknown error")

    async def _find_alternative_agent(
        self,
        task: SwarmTask,
        exclude_agent_id: str
    ) -> Optional[BaseAgent]:
        """Find an alternative agent for a failed task."""
        for agent in self.state.active_agents.values():
            if agent.agent_id == exclude_agent_id:
                continue
            if agent.state.status not in (AgentStatus.IDLE, AgentStatus.COMPLETED):
                continue
            if agent.can_handle(task.required_capabilities) > 0.3:
                return agent
        return None

        # Process more tasks
        asyncio.create_task(self._process_queue())

    async def _handle_handoff(
        self,
        target_agent_type: str,
        task_description: str,
        reason: str,
        context: Optional[dict],
    ):
        """Handle a handoff request from an agent."""
        logger.info(f"Handoff requested: {target_agent_type} - {reason}")

        # Notify legacy listeners
        for handler in self._on_handoff:
            try:
                await handler(target_agent_type, task_description, reason)
            except Exception as e:
                logger.error(f"Handoff handler error: {e}")

        # AGI: Emit handoff signal via Nexus
        await self._emit_handoff("unknown", target_agent_type, reason)

        # Extract context_vector if available
        context_vector = None
        if context and "context_vector" in context:
            context_vector = context["context_vector"]

        # Submit as new task
        await self.submit_task(
            description=task_description,
            context=context,
            context_vector=context_vector,
            priority=6,  # Slightly higher priority for handoffs
        )

    async def execute_with_subtasks(
        self,
        main_task: str,
        subtasks: list[str],
        context: Optional[dict] = None,
    ) -> list[TaskResult]:
        """
        Execute a main task with parallel subtasks.

        Subtasks are executed concurrently when possible.
        """
        # Submit main task
        main_id = await self.submit_task(main_task, context=context, priority=7)

        # Submit subtasks
        subtask_ids = []
        for subtask in subtasks:
            subtask_id = await self.submit_task(
                subtask,
                context={**(context or {}), "parent_task": main_id},
                priority=5,
            )
            subtask_ids.append(subtask_id)
            self.state.tasks[main_id].subtasks.append(subtask_id)

        # Wait for all subtasks to complete
        results = await asyncio.gather(*[
            self.wait_for_task(task_id)
            for task_id in subtask_ids
        ])

        return results

    async def wait_for_task(self, task_id: str, timeout: float = 300.0) -> TaskResult:
        """Wait for a task to complete."""
        start_time = asyncio.get_event_loop().time()

        while True:
            task = self.state.tasks.get(task_id)

            if task and task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                return task.result or TaskResult(success=False, output="No result")

            if asyncio.get_event_loop().time() - start_time > timeout:
                return TaskResult(success=False, output="Task timeout")

            await asyncio.sleep(0.1)

    def get_task_status(self, task_id: str) -> Optional[dict]:
        """Get the status of a task."""
        task = self.state.tasks.get(task_id)
        if not task:
            return None

        return {
            "id": task.id,
            "description": task.description[:100],
            "status": task.status.value,
            "assigned_agent": task.assigned_agent,
            "created_at": task.created_at.isoformat(),
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        }

    def on_task_complete(self, handler: Callable):
        """Register a task completion handler."""
        self._on_task_complete.append(handler)

    def on_handoff(self, handler: Callable):
        """Register a handoff handler."""
        self._on_handoff.append(handler)

    def get_swarm_status(self) -> dict:
        """Get comprehensive swarm status."""
        return {
            "active_agents": {
                agent_id: agent.get_status()
                for agent_id, agent in self.state.active_agents.items()
            },
            "queue_length": len(self.state.task_queue),
            "pending_tasks": sum(
                1 for t in self.state.tasks.values()
                if t.status == TaskStatus.PENDING
            ),
            "in_progress_tasks": sum(
                1 for t in self.state.tasks.values()
                if t.status == TaskStatus.IN_PROGRESS
            ),
            "total_tasks_processed": self.state.total_tasks_processed,
            "total_handoffs": self.state.total_handoffs,
            "agent_types": list(self._agent_factories.keys()),
            # AGI metrics
            "nexus_connected": self._nexus_subscribed,
            "speculative_agents": len(self._speculative_agents),
            "agent_type_vectors": len(self._agent_type_vectors),
            "memory_aware_routing": self._memory_aware_routing,
            "evolution_on_consensus": self._evolution_on_consensus,
            "evolution_on_anomaly": self._evolution_on_anomaly,
        }

    # =========================================================================
    # POPULATION-BASED EVOLUTION (AGI Upgrade)
    # =========================================================================

    def _init_evolution(self, config: Optional[EvolutionConfig] = None):
        """Initialize evolution tracking."""
        self._evolution_config = config or EvolutionConfig()
        self._population: dict[str, AgentVariant] = {}
        self._evolution_history: list[dict] = []

    async def evolve_population(
        self,
        agent_type: str,
        evaluation_tasks: list[str],
        generations: Optional[int] = None,
    ) -> list[AgentVariant]:
        """
        Run population-based evolution on a set of agent variants.

        Uses genetic algorithm:
        1. Initialize population with random variants
        2. Evaluate fitness on test tasks
        3. Select elite survivors
        4. Crossover and mutate to create next generation
        5. Repeat for N generations

        Returns the final evolved population sorted by fitness.
        """
        if not hasattr(self, '_evolution_config'):
            self._init_evolution()

        config = self._evolution_config
        num_generations = generations or config.generations

        # Initialize population if empty
        if not self._population:
            for i in range(config.population_size):
                variant = AgentVariant(
                    variant_id=f"{agent_type}_v{i}_{uuid.uuid4().hex[:6]}",
                    base_agent_type=agent_type,
                    generation=0,
                    temperature=random.uniform(0.3, 1.0),
                    prompt_style=random.choice(["concise", "detailed", "creative", "balanced"]),
                    confidence_threshold=random.uniform(0.4, 0.8),
                )
                self._population[variant.variant_id] = variant

        logger.info(f"Starting evolution: {config.population_size} variants, {num_generations} generations")

        # Evolution loop
        for gen in range(num_generations):
            logger.info(f"Generation {gen + 1}/{num_generations}")

            # Evaluate all variants
            for variant in self._population.values():
                if variant.generation == gen:
                    await self._evaluate_variant(variant, evaluation_tasks)

            # Sort by fitness
            sorted_variants = sorted(
                self._population.values(),
                key=lambda v: v.fitness_score,
                reverse=True,
            )

            # Elite selection - top performers survive unchanged
            elite_count = max(1, int(config.population_size * config.elite_ratio))
            elites = sorted_variants[:elite_count]

            # Create next generation
            next_gen: list[AgentVariant] = []

            # Elites pass through
            for elite in elites:
                next_gen.append(elite)

            # Fill rest with crossover and mutation
            while len(next_gen) < config.population_size:
                # Tournament selection for parents
                parent1 = self._tournament_select(sorted_variants)
                parent2 = self._tournament_select(sorted_variants)

                # Crossover
                if random.random() < config.crossover_rate:
                    child = self._crossover(parent1, parent2, gen + 1)
                else:
                    child = AgentVariant(
                        variant_id=f"{agent_type}_v{len(next_gen)}_{uuid.uuid4().hex[:6]}",
                        base_agent_type=agent_type,
                        generation=gen + 1,
                        parent_id=parent1.variant_id,
                        temperature=parent1.temperature,
                        prompt_style=parent1.prompt_style,
                        confidence_threshold=parent1.confidence_threshold,
                    )

                # Mutation
                if random.random() < config.mutation_rate:
                    child = self._mutate_variant(child)

                next_gen.append(child)

            # Update population
            self._population = {v.variant_id: v for v in next_gen}

            # Record history
            self._evolution_history.append({
                "generation": gen + 1,
                "best_fitness": sorted_variants[0].fitness_score,
                "avg_fitness": sum(v.fitness_score for v in sorted_variants) / len(sorted_variants),
                "best_variant": sorted_variants[0].variant_id,
            })

        # Return final sorted population
        return sorted(
            self._population.values(),
            key=lambda v: v.fitness_score,
            reverse=True,
        )

    async def _evaluate_variant(
        self,
        variant: AgentVariant,
        evaluation_tasks: list[str],
    ):
        """Evaluate a variant's fitness on test tasks."""
        config = self._evolution_config
        start_time = datetime.now()

        successes = 0
        total_confidence = 0.0

        for task_desc in evaluation_tasks:
            try:
                # Spawn agent with variant's traits
                agent = await self.spawn_agent(variant.base_agent_type)
                if not agent:
                    continue

                # Apply variant traits
                agent.temperature = variant.temperature
                agent.confidence_threshold = variant.confidence_threshold

                # Execute task
                result = await asyncio.wait_for(
                    agent.execute(task_desc, {}),
                    timeout=self.handoff_timeout,
                )

                if result.success:
                    successes += 1
                    variant.tasks_completed += 1
                else:
                    variant.tasks_failed += 1

                total_confidence += result.confidence

            except asyncio.TimeoutError:
                variant.tasks_failed += 1
            except Exception as e:
                logger.warning(f"Evaluation error for {variant.variant_id}: {e}")
                variant.tasks_failed += 1

        # Calculate fitness components
        total_tasks = len(evaluation_tasks)
        success_rate = successes / total_tasks if total_tasks > 0 else 0
        avg_confidence = total_confidence / total_tasks if total_tasks > 0 else 0.5
        execution_time = (datetime.now() - start_time).total_seconds()

        # Speed score (faster = better, max 1.0)
        speed_score = max(0, 1 - (execution_time / (total_tasks * self.handoff_timeout)))

        # Handoff efficiency (fewer handoffs = better)
        handoff_efficiency = 1 / (1 + variant.tasks_failed * 0.1)

        # Weighted fitness
        weights = config.fitness_weights
        variant.fitness_score = (
            weights["success_rate"] * success_rate +
            weights["confidence"] * avg_confidence +
            weights["speed"] * speed_score +
            weights["handoff_efficiency"] * handoff_efficiency
        )

        variant.avg_confidence = avg_confidence
        variant.avg_execution_time = execution_time / total_tasks if total_tasks > 0 else 0

        logger.debug(
            f"Variant {variant.variant_id}: fitness={variant.fitness_score:.3f}, "
            f"success={success_rate:.2f}, confidence={avg_confidence:.2f}"
        )

    def _tournament_select(
        self,
        population: list[AgentVariant],
        tournament_size: int = 3,
    ) -> AgentVariant:
        """Select a variant using tournament selection."""
        tournament = random.sample(
            population,
            min(tournament_size, len(population)),
        )
        return max(tournament, key=lambda v: v.fitness_score)

    def _crossover(
        self,
        parent1: AgentVariant,
        parent2: AgentVariant,
        generation: int,
    ) -> AgentVariant:
        """Create a child variant by crossing over two parents."""
        child = AgentVariant(
            variant_id=f"{parent1.base_agent_type}_v{generation}_{uuid.uuid4().hex[:6]}",
            base_agent_type=parent1.base_agent_type,
            generation=generation,
            parent_id=parent1.variant_id,
            # Crossover traits
            temperature=(parent1.temperature + parent2.temperature) / 2,
            prompt_style=random.choice([parent1.prompt_style, parent2.prompt_style]),
            confidence_threshold=(parent1.confidence_threshold + parent2.confidence_threshold) / 2,
        )
        return child

    def _mutate_variant(self, variant: AgentVariant) -> AgentVariant:
        """Apply random mutations to a variant."""
        mutations = []

        # Temperature mutation
        if random.random() < 0.5:
            delta = random.gauss(0, 0.1)
            variant.temperature = max(0.1, min(1.5, variant.temperature + delta))
            mutations.append(f"temperature:{delta:+.2f}")

        # Prompt style mutation
        if random.random() < 0.3:
            styles = ["concise", "detailed", "creative", "balanced"]
            old_style = variant.prompt_style
            variant.prompt_style = random.choice(styles)
            if variant.prompt_style != old_style:
                mutations.append(f"style:{variant.prompt_style}")

        # Confidence threshold mutation
        if random.random() < 0.5:
            delta = random.gauss(0, 0.05)
            variant.confidence_threshold = max(0.3, min(0.9, variant.confidence_threshold + delta))
            mutations.append(f"conf_thresh:{delta:+.2f}")

        variant.mutation_history.extend(mutations)
        return variant

    async def gossip_state(
        self,
        peer_orchestrators: list["SwarmOrchestrator"],
        share_ratio: float = 0.3,
    ) -> dict:
        """
        Share evolutionary state with peer orchestrators.

        Implements gossip protocol for decentralized knowledge sharing:
        1. Select top performers to share
        2. Send to random subset of peers
        3. Merge received variants into local population

        Returns: {sent_count, received_count, merged_count}
        """
        if not hasattr(self, '_population') or not self._population:
            return {"sent_count": 0, "received_count": 0, "merged_count": 0}

        # Select top variants to share
        sorted_variants = sorted(
            self._population.values(),
            key=lambda v: v.fitness_score,
            reverse=True,
        )
        share_count = max(1, int(len(sorted_variants) * share_ratio))
        to_share = sorted_variants[:share_count]

        sent_count = 0
        received_count = 0
        merged_count = 0

        # Share with each peer
        for peer in peer_orchestrators:
            if not hasattr(peer, '_population'):
                peer._init_evolution()

            # Send our top variants
            for variant in to_share:
                variant_data = variant.to_dict()
                sent_count += 1

                # Check if peer should adopt this variant
                if variant.fitness_score > 0.5:  # Only share good variants
                    # Create new variant for peer with slightly mutated traits
                    peer_variant = AgentVariant(
                        variant_id=f"gossip_{variant.variant_id}_{uuid.uuid4().hex[:4]}",
                        base_agent_type=variant.base_agent_type,
                        generation=variant.generation,
                        temperature=variant.temperature,
                        prompt_style=variant.prompt_style,
                        confidence_threshold=variant.confidence_threshold,
                        fitness_score=variant.fitness_score * 0.9,  # Slight penalty for transferred
                        parent_id=variant.variant_id,
                    )

                    # Only add if peer doesn't have similar variant
                    similar_exists = any(
                        abs(v.temperature - peer_variant.temperature) < 0.1 and
                        v.prompt_style == peer_variant.prompt_style
                        for v in peer._population.values()
                    )

                    if not similar_exists:
                        peer._population[peer_variant.variant_id] = peer_variant
                        merged_count += 1

            # Receive from peer (symmetric exchange)
            received_count += len(peer._population)

        logger.info(
            f"Gossip complete: sent={sent_count}, received={received_count}, merged={merged_count}"
        )

        return {
            "sent_count": sent_count,
            "received_count": received_count,
            "merged_count": merged_count,
        }

    def get_best_variant(self, agent_type: Optional[str] = None) -> Optional[AgentVariant]:
        """Get the best performing variant, optionally filtered by agent type."""
        if not hasattr(self, '_population') or not self._population:
            return None

        variants = list(self._population.values())

        if agent_type:
            variants = [v for v in variants if v.base_agent_type == agent_type]

        if not variants:
            return None

        return max(variants, key=lambda v: v.fitness_score)

    def get_evolution_stats(self) -> dict:
        """Get statistics about the evolution process."""
        if not hasattr(self, '_evolution_history'):
            return {"status": "not_initialized"}

        return {
            "population_size": len(self._population) if hasattr(self, '_population') else 0,
            "generations_completed": len(self._evolution_history),
            "history": self._evolution_history[-10:],  # Last 10 generations
            "best_variant": self.get_best_variant().to_dict() if self.get_best_variant() else None,
        }
