"""
Farnsworth Planner Agent - Task Decomposition & Orchestration

Novel Approaches:
1. Hierarchical Task Networks - Multi-level task breakdown
2. Dependency Graph - Automatic dependency detection
3. Progress Monitoring - Real-time status tracking
4. Dynamic Replanning - Adapt to failures and changes
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any, Callable
from collections import defaultdict
import json

from loguru import logger


class TaskStatus(Enum):
    PENDING = "pending"
    READY = "ready"  # All dependencies satisfied
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"  # Waiting on dependencies
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class SubTask:
    """A single task in the plan."""
    id: str
    title: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM

    # Dependencies
    depends_on: list[str] = field(default_factory=list)  # Task IDs
    blocks: list[str] = field(default_factory=list)  # Tasks waiting on this

    # Assignment
    assigned_agent: Optional[str] = None
    agent_type: Optional[str] = None  # "code", "reasoning", "research", etc.

    # Execution
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None

    # Estimation
    estimated_complexity: float = 1.0  # 1.0 = simple, 5.0 = complex
    actual_duration_seconds: Optional[float] = None

    # Metadata
    metadata: dict = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "depends_on": self.depends_on,
            "blocks": self.blocks,
            "assigned_agent": self.assigned_agent,
            "agent_type": self.agent_type,
            "estimated_complexity": self.estimated_complexity,
            "metadata": self.metadata,
        }


@dataclass
class Plan:
    """A complete plan with tasks and structure."""
    id: str
    goal: str
    tasks: dict[str, SubTask] = field(default_factory=dict)

    # Status
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Hierarchy
    root_task_ids: list[str] = field(default_factory=list)

    # Progress
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0

    def get_progress(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return self.completed_tasks / self.total_tasks

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "goal": self.goal,
            "status": self.status.value,
            "progress": self.get_progress(),
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "tasks": {k: v.to_dict() for k, v in self.tasks.items()},
        }


class PlannerAgent:
    """
    Intelligent task planning and orchestration agent.

    Features:
    - Breaks complex goals into actionable sub-tasks
    - Builds dependency graphs automatically
    - Monitors progress and replans on failures
    - Coordinates with specialist agents
    """

    def __init__(
        self,
        llm_fn: Optional[Callable] = None,
        max_concurrent_tasks: int = 3,
    ):
        self.llm_fn = llm_fn
        self.max_concurrent_tasks = max_concurrent_tasks

        self.plans: dict[str, Plan] = {}
        self.active_plan_id: Optional[str] = None

        self._task_counter = 0
        self._plan_counter = 0
        self._lock = asyncio.Lock()

        # Callbacks
        self.on_task_complete: Optional[Callable] = None
        self.on_task_failed: Optional[Callable] = None
        self.on_plan_complete: Optional[Callable] = None

    async def create_plan(
        self,
        goal: str,
        context: Optional[str] = None,
        constraints: Optional[list[str]] = None,
    ) -> Plan:
        """
        Create a plan to achieve a goal.

        Uses LLM to break down the goal into sub-tasks.
        """
        async with self._lock:
            self._plan_counter += 1
            plan_id = f"plan_{self._plan_counter}"

            plan = Plan(id=plan_id, goal=goal)

            if self.llm_fn:
                # Use LLM to decompose the goal
                tasks = await self._decompose_goal(goal, context, constraints)
                for task in tasks:
                    plan.tasks[task.id] = task
                    if not task.depends_on:
                        plan.root_task_ids.append(task.id)

                # Build dependency graph
                self._build_dependency_graph(plan)
            else:
                # Create a single task for the whole goal
                task = await self._create_task(goal, goal)
                plan.tasks[task.id] = task
                plan.root_task_ids.append(task.id)

            plan.total_tasks = len(plan.tasks)
            self.plans[plan_id] = plan

            logger.info(f"Created plan {plan_id} with {plan.total_tasks} tasks")
            return plan

    async def _decompose_goal(
        self,
        goal: str,
        context: Optional[str],
        constraints: Optional[list[str]],
    ) -> list[SubTask]:
        """Use LLM to decompose goal into tasks."""
        prompt = f"""Break down this goal into specific, actionable sub-tasks.

Goal: {goal}

{f"Context: {context}" if context else ""}
{f"Constraints: {', '.join(constraints)}" if constraints else ""}

Return a JSON array of tasks, each with:
- title: Brief task title
- description: Detailed description of what to do
- agent_type: "code", "reasoning", "research", "creative", or "general"
- depends_on: Array of task indices (0-based) this task depends on
- complexity: 1-5 (1=simple, 5=very complex)

Example:
[
  {{"title": "Research API options", "description": "Find available APIs for...", "agent_type": "research", "depends_on": [], "complexity": 2}},
  {{"title": "Design schema", "description": "Create data model based on research", "agent_type": "reasoning", "depends_on": [0], "complexity": 3}}
]

Return ONLY the JSON array, no other text."""

        try:
            if asyncio.iscoroutinefunction(self.llm_fn):
                response = await self.llm_fn(prompt)
            else:
                response = self.llm_fn(prompt)

            # Parse JSON from response
            json_str = self._extract_json(response)
            task_data = json.loads(json_str)

            tasks = []
            task_ids = []

            for i, td in enumerate(task_data):
                task = await self._create_task(
                    title=td.get("title", f"Task {i+1}"),
                    description=td.get("description", ""),
                    agent_type=td.get("agent_type", "general"),
                    complexity=td.get("complexity", 1.0),
                )
                tasks.append(task)
                task_ids.append(task.id)

            # Resolve dependencies by index
            for i, td in enumerate(task_data):
                dep_indices = td.get("depends_on", [])
                for dep_idx in dep_indices:
                    if 0 <= dep_idx < len(task_ids):
                        tasks[i].depends_on.append(task_ids[dep_idx])

            return tasks

        except Exception as e:
            logger.error(f"Goal decomposition failed: {e}")
            # Fallback to single task
            task = await self._create_task(goal, goal)
            return [task]

    async def _create_task(
        self,
        title: str,
        description: str,
        agent_type: str = "general",
        complexity: float = 1.0,
    ) -> SubTask:
        """Create a new sub-task."""
        self._task_counter += 1
        return SubTask(
            id=f"task_{self._task_counter}",
            title=title,
            description=description,
            agent_type=agent_type,
            estimated_complexity=complexity,
        )

    def _build_dependency_graph(self, plan: Plan):
        """Build reverse dependency links (blocks)."""
        for task in plan.tasks.values():
            for dep_id in task.depends_on:
                if dep_id in plan.tasks:
                    plan.tasks[dep_id].blocks.append(task.id)

    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response."""
        # Find JSON array
        start = text.find('[')
        end = text.rfind(']') + 1
        if start >= 0 and end > start:
            return text[start:end]

        # Try object
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            return '[' + text[start:end] + ']'

        return '[]'

    async def execute_plan(
        self,
        plan_id: str,
        agent_executor: Optional[Callable] = None,
    ) -> Plan:
        """
        Execute a plan, coordinating task execution.

        Args:
            plan_id: ID of the plan to execute
            agent_executor: Async function(task, agent_type) -> result
        """
        if plan_id not in self.plans:
            raise ValueError(f"Unknown plan: {plan_id}")

        plan = self.plans[plan_id]
        plan.status = TaskStatus.IN_PROGRESS
        plan.started_at = datetime.now()
        self.active_plan_id = plan_id

        logger.info(f"Executing plan {plan_id}: {plan.goal}")

        try:
            while not self._is_plan_complete(plan):
                # Get ready tasks
                ready_tasks = self._get_ready_tasks(plan)

                if not ready_tasks:
                    # Check for deadlock
                    if self._has_unfinished_tasks(plan):
                        logger.warning("Plan appears deadlocked")
                        await self._handle_deadlock(plan)
                    break

                # Execute ready tasks (up to max concurrent)
                batch = ready_tasks[:self.max_concurrent_tasks]

                if agent_executor:
                    await asyncio.gather(*[
                        self._execute_task(plan, task, agent_executor)
                        for task in batch
                    ])
                else:
                    # Mark as completed (no executor)
                    for task in batch:
                        task.status = TaskStatus.COMPLETED
                        task.completed_at = datetime.now()
                        plan.completed_tasks += 1

            # Finalize plan
            if plan.failed_tasks == 0:
                plan.status = TaskStatus.COMPLETED
            else:
                plan.status = TaskStatus.FAILED

            plan.completed_at = datetime.now()

            if self.on_plan_complete:
                await self._call_async(self.on_plan_complete, plan)

            logger.info(f"Plan {plan_id} finished: {plan.status.value}")
            return plan

        except Exception as e:
            logger.error(f"Plan execution failed: {e}")
            plan.status = TaskStatus.FAILED
            raise
        finally:
            self.active_plan_id = None

    async def _execute_task(
        self,
        plan: Plan,
        task: SubTask,
        agent_executor: Callable,
    ):
        """Execute a single task."""
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()

        logger.info(f"Executing task {task.id}: {task.title}")

        try:
            if asyncio.iscoroutinefunction(agent_executor):
                result = await agent_executor(task, task.agent_type)
            else:
                result = agent_executor(task, task.agent_type)

            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.actual_duration_seconds = (
                task.completed_at - task.started_at
            ).total_seconds()

            plan.completed_tasks += 1

            if self.on_task_complete:
                await self._call_async(self.on_task_complete, task)

            logger.info(f"Task {task.id} completed")

        except Exception as e:
            task.error = str(e)
            task.retry_count += 1

            if task.retry_count >= task.max_retries:
                task.status = TaskStatus.FAILED
                plan.failed_tasks += 1

                if self.on_task_failed:
                    await self._call_async(self.on_task_failed, task)

                logger.error(f"Task {task.id} failed after {task.retry_count} retries")

                # Attempt replanning
                await self._handle_task_failure(plan, task)
            else:
                task.status = TaskStatus.PENDING
                logger.warning(f"Task {task.id} failed, will retry ({task.retry_count}/{task.max_retries})")

    async def _call_async(self, fn: Callable, *args):
        """Call function, handling both sync and async."""
        if asyncio.iscoroutinefunction(fn):
            await fn(*args)
        else:
            fn(*args)

    def _get_ready_tasks(self, plan: Plan) -> list[SubTask]:
        """Get tasks ready for execution (all dependencies satisfied)."""
        ready = []

        for task in plan.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue

            # Check all dependencies are completed
            deps_satisfied = all(
                plan.tasks.get(dep_id, SubTask(id="", title="", description="")).status == TaskStatus.COMPLETED
                for dep_id in task.depends_on
            )

            if deps_satisfied:
                task.status = TaskStatus.READY
                ready.append(task)

        # Sort by priority
        ready.sort(key=lambda t: t.priority.value)

        return ready

    def _is_plan_complete(self, plan: Plan) -> bool:
        """Check if plan is complete."""
        for task in plan.tasks.values():
            if task.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                return False
        return True

    def _has_unfinished_tasks(self, plan: Plan) -> bool:
        """Check if there are unfinished tasks."""
        for task in plan.tasks.values():
            if task.status in (TaskStatus.PENDING, TaskStatus.READY, TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED):
                return True
        return False

    async def _handle_deadlock(self, plan: Plan):
        """Handle a deadlocked plan."""
        # Find blocked tasks
        blocked = [
            t for t in plan.tasks.values()
            if t.status in (TaskStatus.PENDING, TaskStatus.BLOCKED)
        ]

        for task in blocked:
            # Check if blocking dependencies failed
            failed_deps = [
                dep_id for dep_id in task.depends_on
                if plan.tasks.get(dep_id, SubTask(id="", title="", description="")).status == TaskStatus.FAILED
            ]

            if failed_deps:
                task.status = TaskStatus.BLOCKED
                task.error = f"Blocked by failed dependencies: {failed_deps}"
                plan.failed_tasks += 1

    async def _handle_task_failure(self, plan: Plan, failed_task: SubTask):
        """Handle a failed task with replanning."""
        logger.info(f"Attempting to replan after task {failed_task.id} failure")

        # Cancel dependent tasks
        for blocked_id in failed_task.blocks:
            if blocked_id in plan.tasks:
                task = plan.tasks[blocked_id]
                if task.status in (TaskStatus.PENDING, TaskStatus.READY):
                    task.status = TaskStatus.CANCELLED
                    task.error = f"Cancelled due to dependency failure: {failed_task.id}"

        # Optionally create alternative tasks via LLM
        if self.llm_fn:
            await self._create_recovery_plan(plan, failed_task)

    async def _create_recovery_plan(self, plan: Plan, failed_task: SubTask):
        """Create alternative tasks to recover from failure."""
        prompt = f"""A task in our plan has failed. Suggest an alternative approach.

Original task: {failed_task.title}
Description: {failed_task.description}
Error: {failed_task.error}

Plan goal: {plan.goal}

Suggest 1-2 alternative tasks that could achieve the same outcome.
Return JSON array with title, description, agent_type, complexity.
Return empty array [] if no alternatives are viable."""

        try:
            if asyncio.iscoroutinefunction(self.llm_fn):
                response = await self.llm_fn(prompt)
            else:
                response = self.llm_fn(prompt)

            json_str = self._extract_json(response)
            alternatives = json.loads(json_str)

            for alt in alternatives:
                new_task = await self._create_task(
                    title=alt.get("title", "Recovery task"),
                    description=alt.get("description", ""),
                    agent_type=alt.get("agent_type", "general"),
                    complexity=alt.get("complexity", 2.0),
                )

                # Inherit dependencies and blocked tasks
                new_task.depends_on = [
                    d for d in failed_task.depends_on
                    if plan.tasks.get(d, SubTask(id="", title="", description="")).status == TaskStatus.COMPLETED
                ]
                new_task.blocks = failed_task.blocks.copy()

                plan.tasks[new_task.id] = new_task
                plan.total_tasks += 1

                # Unblock dependent tasks
                for blocked_id in new_task.blocks:
                    if blocked_id in plan.tasks:
                        blocked_task = plan.tasks[blocked_id]
                        blocked_task.depends_on = [
                            d if d != failed_task.id else new_task.id
                            for d in blocked_task.depends_on
                        ]
                        if blocked_task.status == TaskStatus.CANCELLED:
                            blocked_task.status = TaskStatus.PENDING
                            blocked_task.error = None

                logger.info(f"Created recovery task {new_task.id}: {new_task.title}")

        except Exception as e:
            logger.error(f"Recovery planning failed: {e}")

    async def add_task(
        self,
        plan_id: str,
        title: str,
        description: str,
        depends_on: Optional[list[str]] = None,
        agent_type: str = "general",
    ) -> SubTask:
        """Add a new task to an existing plan."""
        if plan_id not in self.plans:
            raise ValueError(f"Unknown plan: {plan_id}")

        plan = self.plans[plan_id]

        async with self._lock:
            task = await self._create_task(title, description, agent_type)
            task.depends_on = depends_on or []

            plan.tasks[task.id] = task
            plan.total_tasks += 1

            if not task.depends_on:
                plan.root_task_ids.append(task.id)

            # Update dependency graph
            for dep_id in task.depends_on:
                if dep_id in plan.tasks:
                    plan.tasks[dep_id].blocks.append(task.id)

            return task

    async def cancel_task(self, plan_id: str, task_id: str) -> bool:
        """Cancel a task and its dependents."""
        if plan_id not in self.plans:
            return False

        plan = self.plans[plan_id]

        if task_id not in plan.tasks:
            return False

        task = plan.tasks[task_id]

        if task.status in (TaskStatus.COMPLETED, TaskStatus.IN_PROGRESS):
            return False

        async with self._lock:
            await self._cancel_task_recursive(plan, task_id)

        return True

    async def _cancel_task_recursive(self, plan: Plan, task_id: str):
        """Recursively cancel a task and dependents."""
        task = plan.tasks.get(task_id)
        if not task:
            return

        task.status = TaskStatus.CANCELLED

        for blocked_id in task.blocks:
            await self._cancel_task_recursive(plan, blocked_id)

    def get_plan_status(self, plan_id: str) -> Optional[dict]:
        """Get detailed status of a plan."""
        if plan_id not in self.plans:
            return None

        plan = self.plans[plan_id]

        status_counts = defaultdict(int)
        for task in plan.tasks.values():
            status_counts[task.status.value] += 1

        return {
            "id": plan_id,
            "goal": plan.goal,
            "status": plan.status.value,
            "progress": plan.get_progress(),
            "tasks": {
                "total": plan.total_tasks,
                "by_status": dict(status_counts),
            },
            "timing": {
                "created": plan.created_at.isoformat(),
                "started": plan.started_at.isoformat() if plan.started_at else None,
                "completed": plan.completed_at.isoformat() if plan.completed_at else None,
            },
        }

    def get_dependency_graph(self, plan_id: str) -> Optional[dict]:
        """Get dependency graph for visualization."""
        if plan_id not in self.plans:
            return None

        plan = self.plans[plan_id]

        nodes = []
        edges = []

        for task in plan.tasks.values():
            nodes.append({
                "id": task.id,
                "label": task.title,
                "status": task.status.value,
                "agent_type": task.agent_type,
            })

            for dep_id in task.depends_on:
                edges.append({
                    "from": dep_id,
                    "to": task.id,
                })

        return {"nodes": nodes, "edges": edges}

    def get_stats(self) -> dict:
        """Get planner statistics."""
        total_tasks = sum(len(p.tasks) for p in self.plans.values())
        completed_tasks = sum(p.completed_tasks for p in self.plans.values())

        return {
            "total_plans": len(self.plans),
            "active_plan": self.active_plan_id,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "success_rate": completed_tasks / total_tasks if total_tasks > 0 else 0,
        }
