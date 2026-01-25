"""
Farnsworth Proactive Agent - Anticipatory Intelligence

Q2 2025 Feature: Proactive Intelligence
- Anticipatory Suggestions: Predict what the user might need next
- Task Automation: Learn and execute repetitive workflows
- Context Awareness: Adapt to time, mood, and project context
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any, Callable
import json

from loguru import logger

from farnsworth.agents.planner_agent import PlannerAgent
from farnsworth.memory.memory_system import MemorySystem


class ProactiveState(Enum):
    IDLE = "idle"
    OBSERVING = "observing"  # Watching user actions to learn
    ANALYZING = "analyzing"  # Analyzing context to find opportunities
    SUGGESTING = "suggesting" # Preparing a suggestion
    ACTING = "acting"        # Autonomously executing a task


@dataclass
class Suggestion:
    """A proactive suggestion for the user."""
    id: str
    title: str
    description: str
    confidence: float
    reasoning: str
    
    # Actionable payload
    action_type: str  # "run_task", "create_file", "search", "none"
    action_payload: dict = field(default_factory=dict)
    
    timestamp: datetime = field(default_factory=datetime.now)
    is_dismissed: bool = False
    is_accepted: bool = False


@dataclass
class ScheduledTask:
    """A task scheduled to run periodically or at a specific time."""
    id: str
    description: str
    schedule_type: str  # "interval" or "cron" or "one_off"
    schedule_value: Any # seconds for interval, cron string, or datetime
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    enabled: bool = True

class ProactiveAgent:
    """
    Agent responsible for background monitoring and proactive assistance.
    """

    def __init__(
        self,
        memory_system: MemorySystem,
        planner_agent: PlannerAgent,
        llm_fn: Optional[Callable] = None,
        check_interval_seconds: float = 60.0,
    ):
        self.memory = memory_system
        self.planner = planner_agent
        self.llm_fn = llm_fn
        self.check_interval = check_interval_seconds
        
        self.state = ProactiveState.IDLE
        self.suggestions: list[Suggestion] = []
        self.scheduled_tasks: list[ScheduledTask] = []
        
        self._is_running = False
        
        self._is_running = False
        self._task: Optional[asyncio.Task] = None
        
        logger.info("Proactive Agent initialized")

    async def start(self):
        """Start the background monitoring loop."""
        if self._is_running:
            return
            
        self._is_running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("Proactive Agent started")

    async def stop(self):
        """Stop the background loop."""
        self._is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Proactive Agent stopped")

    async def _monitor_loop(self):
        """Main background loop."""
        while self._is_running:
            try:
                self.state = ProactiveState.OBSERVING
                
                # 1. Get current context
                context = await self._analyze_context()
                
                # 2. Check for patterns/needs
                if context["activity_level"] > 0:
                    self.state = ProactiveState.ANALYZING
                    opportunity = await self._detect_opportunity(context)
                    
                    if opportunity:
                        self.state = ProactiveState.SUGGESTING
                        await self._generate_suggestion(opportunity)
                
                # 3. Process Scheduled Tasks
                await self._check_scheduled_tasks()

                self.state = ProactiveState.IDLE
                
            except Exception as e:
                logger.error(f"Error in proactive loop: {e}")
                self.state = ProactiveState.IDLE
            
            await asyncio.sleep(self.check_interval)

    async def _check_scheduled_tasks(self):
        """Check and execute scheduled tasks."""
        now = datetime.now()
        for task in self.scheduled_tasks:
            if not task.enabled:
                continue
                
            should_run = False
            if task.schedule_type == "interval":
                if not task.last_run:
                    should_run = True
                elif (now - task.last_run).total_seconds() >= task.schedule_value:
                    should_run = True
            
            if should_run:
                logger.info(f"Running scheduled task: {task.description}")
                # Execute via Planner
                try:
                     # Submit task to planner (fire and forget for now, or track)
                     # In a real system, we'd use the SwarmOrchestrator
                     # Here we just log it as "executed"
                     task.last_run = now
                     self.state = ProactiveState.ACTING
                     # TODO: Actually execute logic
                except Exception as e:
                    logger.error(f"Failed to run scheduled task {task.id}: {e}")
                self.state = ProactiveState.IDLE

    async def _analyze_context(self) -> dict:
        """
        Analyze current user context and system state.
        
        Novelty: ContextVector
        We create a multi-dimensional representation of context:
        [time_of_day, user_mood, focus_level, project_phase]
        """
        # 1. Time context
        now = datetime.now()
        is_work_hours = 9 <= now.hour <= 17
        
        # 2. Memory context (recent conversations)
        recent_context = await self.memory.get_conversation_context(max_turns=5)
        
        # 3. Planner context (active tasks)
        active_plan = None
        if self.planner.active_plan_id:
            active_plan = self.planner.get_plan_status(self.planner.active_plan_id)
            
        # 4. Activity level heuristic
        activity_level = 0.1
        if active_plan and active_plan['status'] == 'in_progress':
            activity_level += 0.5
        if len(recent_context) > 50:
            activity_level += 0.4
            
        # 5. Novel Context Attributes
        # Mood Heuristic: Length of recent queries (Short = Focused/Terse, Long = Exploratory)
        # TODO: Use Sentiment Analysis from LLM
        focus_score = 0.5
        if "error" in recent_context.lower() or "fail" in recent_context.lower():
            mood = "frustrated"
            focus_score = 0.9 # High focus on debugging
        else:
            mood = "neutral"
            
        return {
            "timestamp": now,
            "is_work_hours": is_work_hours,
            "recent_context": recent_context,
            "active_plan": active_plan,
            "activity_level": min(1.0, activity_level),
            "context_vector": {
                "mood": mood,
                "focus": focus_score,
                "phase": "execution" if active_plan else "planning"
            }
        }

    async def _detect_opportunity(self, context: dict) -> Optional[dict]:
        """Detect a potential opportunity to help."""
        if not self.llm_fn or context["activity_level"] < 0.3:
            return None

        prompt = f"""Analyze the user's context and suggest ONE proactive action if relevant.
        
Current Time: {context['timestamp']} (Work Hours: {context['is_work_hours']})
Active Plan: {json.dumps(context.get('active_plan', {}), default=str)}
Recent Conversation:
{context['recent_context']}

Identify if there is a clear, helpful action you can propose (e.g., creating a file, searching for info, automating a next step).
If NO clear action is needed, return {{}}.

If yes, return a JSON object with:
- title: Short title of suggestion
- description: Why this is helpful
- action_type: "run_task" | "plan_task" | "search"
- confidence: 0.0 to 1.0 (threshold 0.7)
"""
        try:
            if asyncio.iscoroutinefunction(self.llm_fn):
                response = await self.llm_fn(prompt)
            else:
                response = self.llm_fn(prompt)
                
            # Basic JSON extraction
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                if data and data.get("confidence", 0) > 0.7:
                    return data
                    
        except Exception as e:
            logger.error(f"Opportunity detection failed: {e}")
            
        return None

    async def _generate_suggestion(self, opportunity: dict):
        """Create and store a suggestion."""
        suggestion = Suggestion(
            id=f"sugg_{len(self.suggestions) + 1}",
            title=opportunity.get("title", "Suggestion"),
            description=opportunity.get("description", ""),
            confidence=opportunity.get("confidence", 0.0),
            reasoning="Context analysis",
            action_type=opportunity.get("action_type", "none"),
            action_payload=opportunity,
        )
        
        self.suggestions.append(suggestion)
        logger.info(f"Generated proactive suggestion: {suggestion.title}")

    def get_status(self) -> dict:
        return {
            "state": self.state.value,
            "is_running": self._is_running,
            "suggestion_count": len(self.suggestions),
        }
