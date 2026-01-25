"""
Farnsworth Swarm Orchestrator - Multi-Agent Coordination

Novel Approaches:
1. Dynamic Agent Spawning - Create agents on demand
2. Capability-Based Routing - Match tasks to best agent
3. Parallel Execution - Run independent subtasks concurrently
4. Emergent Behavior - Allow agent team compositions to evolve
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Callable
import uuid

from loguru import logger

from farnsworth.agents.base_agent import BaseAgent, AgentCapability, AgentStatus, TaskResult


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
    """

    def __init__(
        self,
        max_concurrent_agents: int = 5,
        handoff_timeout_seconds: float = 30.0,
    ):
        self.max_concurrent = max_concurrent_agents
        self.handoff_timeout = handoff_timeout_seconds

        self.state = SwarmState()

        # Agent factories
        self._agent_factories: dict[str, Callable[[], BaseAgent]] = {}

        # Shared resources
        self.llm_backend = None
        self.memory_system = None

        # Event handlers
        self._on_task_complete: list[Callable] = []
        self._on_handoff: list[Callable] = []

        self._lock = asyncio.Lock()

    def register_agent_factory(self, agent_type: str, factory: Callable[[], BaseAgent]):
        """Register an agent factory for dynamic creation."""
        self._agent_factories[agent_type] = factory
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
    ) -> str:
        """
        Submit a task to the swarm.

        Returns task ID.
        """
        task_id = f"task_{uuid.uuid4().hex[:8]}"

        task = SwarmTask(
            id=task_id,
            description=description,
            required_capabilities=required_capabilities or set(),
            context=context or {},
            priority=priority,
        )

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

        # Try to process immediately
        asyncio.create_task(self._process_queue())

        return task_id

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
        """Find the best available agent for a task."""
        best_agent = None
        best_score = 0.0

        for agent in self.state.active_agents.values():
            if agent.state.status not in (AgentStatus.IDLE, AgentStatus.COMPLETED):
                continue

            score = agent.can_handle(task.required_capabilities)

            # Adjust for agent performance
            score *= (0.5 + 0.5 * agent.state.avg_confidence)

            if score > best_score:
                best_score = score
                best_agent = agent

        return best_agent if best_score > 0.3 else None

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

    async def _execute_task(self, task: SwarmTask, agent: BaseAgent):
        """Execute a task with an agent."""
        task.status = TaskStatus.IN_PROGRESS

        try:
            result = await agent.execute(task.description, task.context)
            task.result = result
            task.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
            task.completed_at = datetime.now()

            self.state.total_tasks_processed += 1
            self.state.completed_tasks.append(task.id)

            # Notify listeners
            for handler in self._on_task_complete:
                try:
                    await handler(task, result)
                except Exception as e:
                    logger.error(f"Task complete handler error: {e}")

            logger.info(f"Task {task.id} completed: success={result.success}")

        except Exception as e:
            logger.error(f"Task execution error: {e}")
            task.status = TaskStatus.FAILED
            task.result = TaskResult(success=False, output=str(e))

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
        self.state.total_handoffs += 1

        # Notify listeners
        for handler in self._on_handoff:
            try:
                await handler(target_agent_type, task_description, reason)
            except Exception as e:
                logger.error(f"Handoff handler error: {e}")

        # Submit as new task
        await self.submit_task(
            description=task_description,
            context=context,
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
        }
