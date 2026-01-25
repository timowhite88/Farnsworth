"""
Farnsworth Base Agent - Foundation for Specialist Agents

Novel Approaches:
1. Capability-Based Dispatch - Dynamic capability registration
2. Confidence-Aware Handoff - Know when to delegate
3. State Persistence - Maintain context across invocations
4. Learning Hooks - Track performance for evolution
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Callable

from loguru import logger


class AgentCapability(Enum):
    """Capabilities an agent can have."""
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    CODE_DEBUGGING = "code_debugging"
    REASONING = "reasoning"
    MATH = "math"
    RESEARCH = "research"
    CREATIVE_WRITING = "creative_writing"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"
    PLANNING = "planning"
    TOOL_USE = "tool_use"
    MEMORY_MANAGEMENT = "memory_management"
    META_COGNITION = "meta_cognition"
    USER_MODELING = "user_modeling"


class AgentStatus(Enum):
    """Agent execution status."""
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"  # Waiting for handoff or resource
    COMPLETED = "completed"
    FAILED = "failed"
    DELEGATED = "delegated"


@dataclass
class AgentState:
    """State container for an agent."""
    agent_id: str
    agent_type: str
    status: AgentStatus = AgentStatus.IDLE
    current_task: str = ""
    progress: float = 0.0
    context: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Performance tracking
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_tokens_used: int = 0
    avg_confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": self.status.value,
            "current_task": self.current_task,
            "progress": self.progress,
            "context": self.context,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "avg_confidence": self.avg_confidence,
        }


@dataclass
class AgentMessage:
    """Message passed between agents."""
    from_agent: str
    to_agent: str
    content: str
    message_type: str = "task"  # task, result, handoff, query
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TaskResult:
    """Result of agent task execution."""
    success: bool
    output: Any
    confidence: float = 0.0
    tokens_used: int = 0
    execution_time: float = 0.0
    should_handoff: bool = False
    handoff_reason: str = ""
    suggested_agent: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base class for all Farnsworth agents.

    Provides:
    - Capability registration and matching
    - State management
    - Handoff logic
    - Performance tracking
    """

    def __init__(
        self,
        name: str,
        capabilities: list[AgentCapability],
        confidence_threshold: float = 0.6,
    ):
        self.agent_id = f"{name}_{uuid.uuid4().hex[:8]}"
        self.name = name
        self.capabilities = set(capabilities)
        self.confidence_threshold = confidence_threshold

        self.state = AgentState(
            agent_id=self.agent_id,
            agent_type=self.__class__.__name__,
        )

        # LLM backend (set by orchestrator)
        self.llm_backend = None

        # Memory system (set by orchestrator)
        self.memory = None

        # Message handlers
        self._message_handlers: dict[str, Callable] = {}

        # Handoff callback (set by orchestrator)
        self._handoff_callback: Optional[Callable] = None

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        pass

    @abstractmethod
    async def process(self, task: str, context: Optional[dict] = None) -> TaskResult:
        """
        Process a task and return result.

        Must be implemented by subclasses.
        """
        pass

    def can_handle(self, required_capabilities: set[AgentCapability]) -> float:
        """
        Check if agent can handle task with given capabilities.

        Returns confidence score (0-1).
        """
        if not required_capabilities:
            return 0.5  # Unknown capability requirement

        overlap = self.capabilities & required_capabilities
        if not overlap:
            return 0.0

        return len(overlap) / len(required_capabilities)

    async def execute(self, task: str, context: Optional[dict] = None) -> TaskResult:
        """
        Execute a task with full lifecycle management.

        Handles:
        - State updates
        - Error handling
        - Handoff decisions
        - Performance tracking
        """
        import time
        start_time = time.time()

        self.state.status = AgentStatus.RUNNING
        self.state.current_task = task
        self.state.progress = 0.0
        self.state.updated_at = datetime.now()

        try:
            # Execute the task
            result = await self.process(task, context)

            # Update state based on result
            execution_time = time.time() - start_time
            result.execution_time = execution_time

            if result.success:
                self.state.tasks_completed += 1
                self.state.status = AgentStatus.COMPLETED
            else:
                self.state.tasks_failed += 1
                self.state.status = AgentStatus.FAILED

            # Update rolling average confidence
            n = self.state.tasks_completed + self.state.tasks_failed
            self.state.avg_confidence = (
                (self.state.avg_confidence * (n - 1) + result.confidence) / n
            )

            self.state.total_tokens_used += result.tokens_used
            self.state.progress = 1.0

            # Record in history
            self.state.history.append({
                "task": task[:100],
                "success": result.success,
                "confidence": result.confidence,
                "time": execution_time,
                "timestamp": datetime.now().isoformat(),
            })

            # Check for handoff
            if result.should_handoff and self._handoff_callback:
                self.state.status = AgentStatus.DELEGATED
                await self._handoff_callback(
                    result.suggested_agent,
                    task,
                    result.handoff_reason,
                    context,
                )

            return result

        except Exception as e:
            logger.error(f"Agent {self.name} error: {e}")
            self.state.status = AgentStatus.FAILED
            self.state.tasks_failed += 1
            self.state.errors.append(str(e))

            return TaskResult(
                success=False,
                output=str(e),
                confidence=0.0,
                execution_time=time.time() - start_time,
            )

        finally:
            self.state.updated_at = datetime.now()

    async def should_handoff(self, task: str, confidence: float) -> tuple[bool, str, str]:
        """
        Determine if task should be handed off to another agent.

        Returns (should_handoff, reason, suggested_agent).
        """
        # Low confidence handoff
        if confidence < self.confidence_threshold:
            return True, f"Low confidence ({confidence:.2f})", self._suggest_handoff_agent(task)

        # Capability-based handoff (check if task requires capabilities we don't have)
        required_caps = self._infer_required_capabilities(task)
        missing_caps = required_caps - self.capabilities

        if missing_caps:
            return True, f"Missing capabilities: {missing_caps}", self._suggest_handoff_agent(task)

        return False, "", ""

    def _infer_required_capabilities(self, task: str) -> set[AgentCapability]:
        """Infer required capabilities from task description."""
        task_lower = task.lower()
        required = set()

        capability_keywords = {
            AgentCapability.CODE_GENERATION: ["write code", "implement", "create function", "generate"],
            AgentCapability.CODE_ANALYSIS: ["analyze code", "review", "explain code"],
            AgentCapability.CODE_DEBUGGING: ["debug", "fix bug", "error", "issue"],
            AgentCapability.REASONING: ["think", "reason", "logic", "deduce"],
            AgentCapability.MATH: ["calculate", "math", "equation", "formula"],
            AgentCapability.RESEARCH: ["research", "find", "search", "look up"],
            AgentCapability.CREATIVE_WRITING: ["write", "story", "creative", "compose"],
            AgentCapability.SUMMARIZATION: ["summarize", "summary", "brief"],
            AgentCapability.PLANNING: ["plan", "strategy", "steps", "roadmap"],
        }

        for cap, keywords in capability_keywords.items():
            if any(kw in task_lower for kw in keywords):
                required.add(cap)

        return required or {AgentCapability.QUESTION_ANSWERING}  # Default

    def _suggest_handoff_agent(self, task: str) -> str:
        """Suggest the best agent for handoff."""
        required = self._infer_required_capabilities(task)

        # Map capabilities to agent types
        capability_agents = {
            AgentCapability.CODE_GENERATION: "code",
            AgentCapability.CODE_ANALYSIS: "code",
            AgentCapability.CODE_DEBUGGING: "code",
            AgentCapability.REASONING: "reasoning",
            AgentCapability.MATH: "reasoning",
            AgentCapability.RESEARCH: "research",
            AgentCapability.CREATIVE_WRITING: "creative",
            AgentCapability.PLANNING: "reasoning",
        }

        for cap in required:
            if cap in capability_agents:
                return capability_agents[cap]

        return "general"

    async def generate_response(
        self,
        prompt: str,
        context: Optional[dict] = None,
    ) -> tuple[str, float]:
        """
        Generate a response using the LLM backend.

        Returns (response_text, confidence).
        """
        if self.llm_backend is None:
            raise RuntimeError("LLM backend not configured")

        # Build full prompt with system prompt
        full_prompt = f"{self.system_prompt}\n\n{prompt}"

        # Include context if available
        if context:
            context_str = "\n".join(f"{k}: {v}" for k, v in context.items())
            full_prompt = f"{self.system_prompt}\n\nContext:\n{context_str}\n\nTask: {prompt}"

        result = await self.llm_backend.generate(full_prompt)

        return result.text, result.confidence_score

    def set_handoff_callback(self, callback: Callable):
        """Set the callback for handoff requests."""
        self._handoff_callback = callback

    def get_status(self) -> dict:
        """Get agent status for monitoring."""
        return self.state.to_dict()

    def reset(self):
        """Reset agent state."""
        self.state.status = AgentStatus.IDLE
        self.state.current_task = ""
        self.state.progress = 0.0
        self.state.context.clear()
